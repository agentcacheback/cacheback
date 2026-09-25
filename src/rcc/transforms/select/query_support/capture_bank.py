"""The layer bank: per-layer capture statistics and the replay that scores them.

A bank is filled during a capture and can be recut afterwards at any registered
selector, on disk or in memory.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from rcc.transforms.select.core import kernels
from rcc.transforms.select.core.types import (
    EnergyAccumulator,
    LayerDescriptor,
)
from rcc.transforms.select.query_support import capture_layers
from rcc.transforms.select.query_support.capture_bank_format import (
    BANK_SCOPE_ALL_LAYER,
    BANK_SCOPE_GLOBAL_ONLY,
    LAYER_BANK_SCHEMA,
    payload_from_bank,
    rows_from_payload,
    validate_payload,
)
from rcc.transforms.select.query_support.methods import reducers
from rcc.transforms.select.query_support.methods.compose import SupportMomentBundle, selector_score

#: Per family, the registered ratio ladder, the one span width, and the selectors
#: that family registers. A listed family replays exactly the arms it was scored
#: under; an unlisted family accepts any arm its selector name parses to.
_REGISTERED_FAMILY_ARMS: dict[str, tuple[frozenset[int], int, frozenset[str]]] = {
    "gemma4_12b": (frozenset({2, 4, 8, 16, 32, 64, 128}), 16, frozenset({"support"})),
    "ministral3_14b": (
        frozenset({2, 4, 8, 16, 32, 64, 128}),
        16,
        frozenset({"snap", "support"}),
    ),
}


def _bank_rows() -> list[LayerBankRow]:
    return []


def _snap_chunks() -> list[torch.Tensor]:
    return []


def _frozen(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(dtype=torch.float32).cpu().clone()


def _frozen_scatter(
    value: torch.Tensor | None,
    descriptor: LayerDescriptor,
) -> torch.Tensor | None:
    if value is None:
        return None
    scattered = capture_layers.scatter_memory(value, descriptor, descriptor.logical_memory_length)
    return _frozen(scattered)


def _frozen_map(
    values: dict[float, torch.Tensor], descriptor: LayerDescriptor
) -> dict[float, torch.Tensor]:
    return {
        order: _frozen(
            capture_layers.scatter_memory(value, descriptor, descriptor.logical_memory_length)
        )
        for order, value in values.items()
    }


@dataclass(frozen=True)
class LayerBankRow:
    """One labeled layer row with exact replay sufficient statistics."""

    descriptor: LayerDescriptor
    snap: torch.Tensor
    energy: torch.Tensor | None
    row_energy: torch.Tensor | None
    support_e: dict[float, torch.Tensor]
    support_eprime: dict[float, torch.Tensor]
    support_r: dict[float, torch.Tensor]
    support_einf: torch.Tensor | None
    support_rinf: torch.Tensor | None
    prepool_e2: torch.Tensor | None
    prepool_r2: torch.Tensor | None
    heads: int
    query_rows: int
    kv_groups: int
    snap_chunks: tuple[torch.Tensor, ...] = ()

    @property
    def normalized_snap(self) -> torch.Tensor:
        """Return this layer's snap mass normalized by attention geometry."""
        return self.snap / float(self.heads * max(1, self.query_rows))

    @property
    def w16_snap(self) -> torch.Tensor:
        """Return the width-16 diagnostic projection of normalized snap."""
        return capture_layers.w16_mean(self.normalized_snap)

    @property
    def schema(self) -> str:
        """Return the fixed bank schema tag."""
        return LAYER_BANK_SCHEMA


@dataclass
class LayerBankRowAccumulator:
    """Mutable row state while one callback layer is being captured.

    Holds the snap partials only; the energy and support terms come from the
    capture's per-layer reducer fold, so no chunk is folded twice.
    """

    descriptor: LayerDescriptor
    device: torch.device
    snap: torch.Tensor = field(init=False)
    heads: int = 1
    query_rows: int = 0
    kv_groups: int = 1
    snap_chunks: list[torch.Tensor] = field(default_factory=_snap_chunks)

    def __post_init__(self) -> None:
        """Initialize an empty logical snap vector on the capture device."""
        self.snap = torch.zeros(
            self.descriptor.logical_memory_length,
            dtype=torch.float32,
            device=self.device,
        )

    def add_snap(
        self,
        contribution: torch.Tensor,
        *,
        heads: int,
        query_rows: int,
        kv_groups: int,
    ) -> None:
        """Append one exact capture-order snap contribution."""
        if contribution.ndim != 1 or int(contribution.shape[0]) != self.snap.shape[0]:
            raise RuntimeError("layer bank snap contribution has the wrong logical geometry")
        if heads < 1 or query_rows < 1 or kv_groups < 1 or heads % kv_groups:
            raise RuntimeError("layer bank attention geometry is invalid")
        self.snap = self.snap + contribution.float()
        self.snap_chunks.append(contribution.detach().float().clone())
        self.heads = heads
        self.query_rows += query_rows
        self.kv_groups = kv_groups


def stream_weights(rows: Sequence[LayerBankRow]) -> tuple[float, ...]:
    """Return per-row weights that give every covering layer one stream population.

    See docs/selectors.md, Stream weights. On a uniform bank every weight is exactly
    1.0 and callers skip the multiply, so uniform-bank arithmetic is bit-identical.
    """
    if not rows:
        raise RuntimeError("stream weights need at least one banked row")
    streams: list[int] = []
    for row in rows:
        if row.kv_groups < 1 or row.heads % row.kv_groups:
            raise RuntimeError("layer bank row kv groups must divide its heads")
        streams.append(row.heads // row.kv_groups)
    top = max(streams)
    return tuple(float(top) / float(count) for count in streams)


@dataclass
class LayerBank:
    """Durable capture rows with stored pooling geometry and scope identity."""

    pool_kernel: int = 1
    scope: str = BANK_SCOPE_GLOBAL_ONLY
    energy_pool_kernel: int | None = None
    row_energy_pool_kernel: int | None = None
    support_orders: tuple[float, ...] = ()
    rows: list[LayerBankRow] = field(default_factory=_bank_rows)

    def __post_init__(self) -> None:
        """Validate the stored replay identity before capture begins."""
        reducers.validate_pool_kernel(self.pool_kernel)
        if self.scope not in {BANK_SCOPE_GLOBAL_ONLY, BANK_SCOPE_ALL_LAYER}:
            raise ValueError(f"unknown layer bank scope {self.scope!r}")
        for kernel in (self.energy_pool_kernel, self.row_energy_pool_kernel):
            if kernel is not None:
                reducers.validate_pool_kernel(kernel)
        orders = tuple(float(order) for order in self.support_orders)
        if orders != tuple(sorted(set(orders))) or any(order <= 1 for order in orders):
            raise ValueError("layer bank support orders must be sorted and greater than one")
        self.support_orders = orders

    @property
    def schema(self) -> str:
        """Return the fixed bank schema tag."""
        return LAYER_BANK_SCHEMA

    def accepts(self, descriptor: LayerDescriptor) -> bool:
        """Return whether the configured scope records this layer."""
        return descriptor.is_global or self.scope == BANK_SCOPE_ALL_LAYER

    def begin(
        self, descriptor: LayerDescriptor, *, device: torch.device
    ) -> LayerBankRowAccumulator | None:
        """Start one row, refusing duplicate callback layer indices."""
        if not self.accepts(descriptor):
            return None
        if any(row.descriptor.layer_idx == descriptor.layer_idx for row in self.rows):
            raise RuntimeError(f"layer bank received duplicate layer {descriptor.layer_idx}")
        return LayerBankRowAccumulator(descriptor, device)

    def finish(self, row: LayerBankRowAccumulator, energies: EnergyAccumulator | None) -> None:
        """Freeze one row from the capture's own per-layer fold products."""
        descriptor = row.descriptor
        moments = EnergyAccumulator(mem_len=0) if energies is None else energies
        self.rows.append(
            LayerBankRow(
                descriptor=descriptor,
                snap=_frozen(row.snap),
                energy=_frozen_scatter(moments.energy_sum, descriptor),
                row_energy=_frozen_scatter(moments.row_energy_sum, descriptor),
                support_e=_frozen_map(moments.support_e, descriptor),
                support_eprime=_frozen_map(moments.support_eprime, descriptor),
                support_r=_frozen_map(moments.support_r, descriptor),
                support_einf=_frozen_scatter(moments.support_einf, descriptor),
                support_rinf=_frozen_scatter(moments.support_rinf, descriptor),
                prepool_e2=_frozen_scatter(moments.support_prepool_e2, descriptor),
                prepool_r2=_frozen_scatter(moments.support_prepool_r2, descriptor),
                heads=row.heads,
                query_rows=row.query_rows,
                kv_groups=row.kv_groups,
                snap_chunks=tuple(_frozen(item) for item in row.snap_chunks),
            )
        )

    def selected_rows(self, scope: str | None = None) -> tuple[LayerBankRow, ...]:
        """Return the rows one scoring scope reads, validated as one geometry."""
        return self._selected(scope)

    def require_window_support(self) -> None:
        """Refuse any row carrying mass outside the interval it could have written."""
        for row in self.rows:
            require_window_support(row)

    def _selected(self, scope: str | None) -> tuple[LayerBankRow, ...]:
        # Checked here as well as at load, because a bank is scored in memory
        # before it is ever written to a file.
        self.require_window_support()
        requested = self.scope if scope is None else scope
        if requested not in {BANK_SCOPE_GLOBAL_ONLY, BANK_SCOPE_ALL_LAYER}:
            raise RuntimeError(f"unknown layer bank replay scope {requested!r}")
        if self.scope == BANK_SCOPE_GLOBAL_ONLY and requested == BANK_SCOPE_ALL_LAYER:
            raise RuntimeError("global-only layer bank cannot replay sliding scope")
        selected = tuple(
            row
            for row in self.rows
            if requested == BANK_SCOPE_ALL_LAYER or row.descriptor.is_global
        )
        if not selected:
            raise RuntimeError("layer bank has no rows for the requested scoring scope")
        lengths = {row.descriptor.logical_memory_length for row in selected}
        # kv_groups is not part of the shared geometry: a hybrid model mixes GQA factors
        # per layer type. Cross-row folds of the per-stream finite moments reweight
        # through `stream_weights`; snap needs only the heads and query rows checked here.
        geometry = {(row.heads, row.query_rows) for row in selected}
        if len(lengths) != 1 or len(geometry) != 1:
            raise RuntimeError(
                "layer bank rows do not share one capture geometry: "
                f"lengths={sorted(lengths)} heads_and_query_rows={sorted(geometry)}"
            )
        return selected

    @staticmethod
    def _sum_moment(
        rows: tuple[LayerBankRow, ...],
        name: str,
        order: float,
        weights: tuple[float, ...],
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        values = [getattr(row, name).get(order) for row in rows]
        if any(value is None for value in values):
            raise RuntimeError(f"layer bank is missing {name} order {order:g}")
        tensors = [value for value in values if value is not None]
        if device is not None:
            tensors = [value.to(device=device) for value in tensors]
        total = tensors[0].clone() if weights[0] == 1.0 else tensors[0] * weights[0]
        for value, weight in zip(tensors[1:], weights[1:], strict=True):
            total = total + (value if weight == 1.0 else value * weight)
        return total

    def replay_snap(
        self,
        *,
        scope: str | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Reconstruct snap on an optional device using the stored fold order."""
        selected = self._selected(scope)
        chunks = [
            chunk if device is None else chunk.to(device=device)
            for row in selected
            for chunk in row.snap_chunks
        ]
        if not chunks:
            raise RuntimeError("layer bank has no snap chunks")
        raw = chunks[0].clone()
        for chunk in chunks[1:]:
            raw = raw + chunk
        heads, query_rows = selected[0].heads, selected[0].query_rows
        return kernels.pool_scores(raw / float(heads * max(1, query_rows)), self.pool_kernel)

    def replay_moments(
        self,
        *,
        scope: str | None = None,
        device: torch.device | str | None = None,
    ) -> SupportMomentBundle:
        """Reconstruct moments on an optional device in the captured layer order."""
        selected = self._selected(scope)
        if not self.support_orders:
            raise RuntimeError("layer bank has no finite support moments")
        einf = selected[0].support_einf
        rinf = selected[0].support_rinf
        if einf is None or rinf is None:
            raise RuntimeError("layer bank is missing infinity support moments")
        if device is not None:
            einf = einf.to(device=device)
            rinf = rinf.to(device=device)
        for row in selected[1:]:
            if row.support_einf is None or row.support_rinf is None:
                raise RuntimeError("layer bank is missing infinity support moments")
            row_einf = row.support_einf
            row_rinf = row.support_rinf
            if device is not None:
                row_einf = row_einf.to(device=device)
                row_rinf = row_rinf.to(device=device)
            einf = torch.maximum(einf, row_einf)
            rinf = torch.maximum(rinf, row_rinf)
        weights = stream_weights(selected)
        return SupportMomentBundle(
            snap=self.replay_snap(scope=scope, device=device),
            e={
                order: self._sum_moment(selected, "support_e", order, weights, device)
                for order in self.support_orders
            },
            eprime={
                order: self._sum_moment(selected, "support_eprime", order, weights, device)
                for order in self.support_orders
            },
            r={
                order: self._sum_moment(selected, "support_r", order, weights, device)
                for order in self.support_orders
            },
            einf=einf,
            rinf=rinf,
        )

    @staticmethod
    def _replay_selector(selector: str) -> tuple[str, str | None]:
        """Resolve a selector name to its composition and optional variant token."""
        if selector == "support":
            return "support-p2-a2", None
        if selector == "snap" or selector.startswith("support-p"):
            return selector, None
        match = re.fullmatch(
            r"(qwen3_8b|gemma4_12b|ministral3_14b)_r([1-9]\d*)(?:_w([1-9]\d*))?"
            r"(?:_(snap|support))?(?:_(slidetail|slidenorm))?",
            selector,
        )
        if match is None:
            raise ValueError(f"selector {selector!r} is not a registered bank selector")
        family, ratio_text, width_text, parsed_selector, variant = match.groups()
        ratio = int(ratio_text)
        width = None if width_text is None else int(width_text)
        parsed_selector = parsed_selector or "snap"
        # A registered family accepts only its own registered arms; an
        # unregistered ratio or width would replay a bank at an arm it was never
        # scored under.
        registered = _REGISTERED_FAMILY_ARMS.get(family)
        if registered is not None:
            ratios, registered_width, selectors = registered
            if ratio not in ratios or width != registered_width or parsed_selector not in selectors:
                raise ValueError(f"selector {selector!r} is not a registered bank selector")
        # Variants are registered on the Gemma support arms only: they read the
        # sliding rows, which exist only on a hybrid layer schedule.
        if variant is not None and (family != "gemma4_12b" or parsed_selector != "support"):
            raise ValueError(f"selector {selector!r} is not a registered bank selector")
        if parsed_selector == "support":
            return "support-p2-a2", variant
        if parsed_selector == "snap":
            return "snap", None
        raise ValueError(f"selector {selector!r} is not a banked replay selector")

    def replay_score(self, selector: str, *, scope: str | None = None) -> torch.Tensor:
        """Replay a plain, grammar-produced, or variant support selector."""
        resolved, variant = self._replay_selector(selector)
        if variant is not None:
            from rcc.transforms.select.query_support.methods import variants

            if scope is not None and scope != BANK_SCOPE_ALL_LAYER:
                raise RuntimeError(
                    f"selector variant {variant!r} scores the sliding rows and cannot "
                    f"replay at scope {scope!r}"
                )
            return variants.variant_score(self, variant, resolved)
        if resolved == "snap":
            return self.replay_snap(scope=scope)
        return selector_score(self.replay_moments(scope=scope), resolved)

    def to_payload(self) -> dict[str, Any]:
        """Return the plain-tensor payload used by durable serialization."""
        return payload_from_bank(self)

    def save(self, path: Path) -> None:
        """Write the plain-tensor payload to a torch file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_payload(), path)

    @classmethod
    def from_payload(cls, payload: object) -> LayerBank:
        """Validate a payload completely before constructing replay rows."""
        checked = validate_payload(payload)
        rows = rows_from_payload(checked)
        # Refuse a malformed bank at the load boundary as well as at scoring.
        for row in rows:
            require_window_support(row)
        return cls(
            pool_kernel=checked["pool_kernel"],
            scope=checked["scope"],
            energy_pool_kernel=checked["energy_pool_kernel"],
            row_energy_pool_kernel=checked["row_energy_pool_kernel"],
            support_orders=tuple(float(order) for order in checked["support_orders"]),
            rows=list(rows),
        )

    @classmethod
    def load(cls, path: Path) -> LayerBank:
        """Load and validate one durable bank with weights-only torch loading."""
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, EOFError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(f"layer bank could not be loaded: {path}") from exc
        return cls.from_payload(payload)


def _row_tensors(row: LayerBankRow) -> list[tuple[str, torch.Tensor]]:
    """Return every stored vector of one row, by name, in a stable order."""
    named: list[tuple[str, torch.Tensor]] = [("snap", row.snap)]
    named.extend((f"snap_chunk[{index}]", chunk) for index, chunk in enumerate(row.snap_chunks))
    for name, value in (
        ("energy", row.energy),
        ("row_energy", row.row_energy),
        ("support_einf", row.support_einf),
        ("support_rinf", row.support_rinf),
        ("prepool_e2", row.prepool_e2),
        ("prepool_r2", row.prepool_r2),
    ):
        if value is not None:
            named.append((name, value))
    for field_name, mapping in (
        ("support_e", row.support_e),
        ("support_eprime", row.support_eprime),
        ("support_r", row.support_r),
    ):
        named.extend(
            (f"{field_name}[{order:g}]", value) for order, value in sorted(mapping.items())
        )
    return named


def covered_span(row: LayerBankRow) -> tuple[int, int]:
    """Return the half-open logical interval this row's attention covers.

    The physical range and the declared window are intersected, since either can be
    the loose bound depending on whether the cache cropped the sliding layer.
    """
    descriptor = row.descriptor
    logical = descriptor.logical_memory_length
    start = min(max(0, descriptor.absolute_key_offset), logical)
    end = min(logical, start + descriptor.physical_memory_length)
    if descriptor.window is not None:
        start = max(start, logical - descriptor.window)
    return (start, end) if end > start else (0, 0)


def require_window_support(row: LayerBankRow) -> None:
    """Raise if a row's tensors carry mass outside its own covered interval.

    `scatter_memory` writes only the row's physical span and leaves the rest zero, so
    outside mass is impossible; snap, every snap chunk, and every moment is checked.
    """
    descriptor = row.descriptor
    logical = descriptor.logical_memory_length
    start, end = covered_span(row)
    if start == 0 and end == logical:
        return
    for name, tensor in _row_tensors(row):
        if int(tensor.shape[-1]) != logical:
            raise RuntimeError(
                f"layer bank row {descriptor.layer_idx} tensor {name} has length "
                f"{int(tensor.shape[-1])}, not the {logical} logical memory columns"
            )
        for label, outside in (("before", tensor[..., :start]), ("after", tensor[..., end:])):
            if outside.numel() and bool(outside.any()):
                raise RuntimeError(
                    f"layer bank row {descriptor.layer_idx} ({descriptor.layer_type}) carries "
                    f"mass {label} its covered interval [{start}, {end}) in tensor {name}; a "
                    "sliding layer cannot vote outside its own attention window, so this bank "
                    "is malformed and no selector may score it"
                )


def needs_reducers(bank: LayerBank | None) -> bool:
    """Return whether a bank row needs energy or support reducer state."""
    return bank is not None and (
        bank.energy_pool_kernel is not None
        or bank.row_energy_pool_kernel is not None
        or bool(bank.support_orders)
    )


__all__ = [
    "BANK_SCOPE_ALL_LAYER",
    "BANK_SCOPE_GLOBAL_ONLY",
    "LAYER_BANK_SCHEMA",
    "LayerBank",
    "LayerBankRow",
    "LayerBankRowAccumulator",
    "covered_span",
    "needs_reducers",
    "require_window_support",
]
