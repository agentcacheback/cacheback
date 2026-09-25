"""The Query-Support moments, the correction, and the score composed from them.

A selector name such as ``support-p2-a2`` resolves to an order p and correction
strength alpha. The score is mean query attention times C_p raised to alpha.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

import torch

Order = float | Literal["inf"]

SUPPORT_MOMENT_SCALE_SCHEMA = "query-support-fixed-power-of-two-v1"
SUPPORT_MOMENT_SCALE = 256.0

SUPPORT_DEFAULT_ORDER = 2.0
"""The default finite moment order p.

The plain ``support`` name resolves to ``support-p2-a2``, so a bank collected for
later replay must carry this order or it holds no moments that name can compose.
"""

SUPPORT_DEFAULT_ALPHA = 2.0
"""The default correction strength: the (p=2, alpha=2) point of the family."""

__all__ = [
    "SUPPORT_DEFAULT_ALPHA",
    "SUPPORT_DEFAULT_ORDER",
    "SUPPORT_MOMENT_SCALE",
    "SUPPORT_MOMENT_SCALE_SCHEMA",
    "Order",
    "SupportMomentBundle",
    "_pin_sink",
    "compose_score",
    "concentration_and_alignment",
    "correction",
    "moment_power",
    "parse_selector",
    "selector_score",
    "support_corrected_scores",
]


def moment_power(values: torch.Tensor, order: float) -> torch.Tensor:
    """Scale one attention base by `SUPPORT_MOMENT_SCALE`, then raise it to `order`."""
    return (values * SUPPORT_MOMENT_SCALE).pow(order)


def _tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


@dataclass(frozen=True)
class SupportMomentBundle:
    """The stream-pooled moment vectors from one query-aware forward."""

    snap: torch.Tensor
    e: dict[float, torch.Tensor]
    eprime: dict[float, torch.Tensor]
    r: dict[float, torch.Tensor]
    einf: torch.Tensor
    rinf: torch.Tensor

    def __post_init__(self) -> None:
        """Check that every vector shares one shape and device and is finite and nonnegative."""
        orders = set(self.e)
        if not orders or orders != set(self.eprime) or orders != set(self.r):
            raise ValueError("E, Eprime, and R must share at least one moment order")
        if any(order <= 1 or not math.isfinite(order) for order in orders):
            raise ValueError("finite support orders must be finite and greater than one")
        vectors = [self.snap, self.einf, self.rinf]
        vectors.extend(self.e[order] for order in sorted(orders))
        vectors.extend(self.eprime[order] for order in sorted(orders))
        vectors.extend(self.r[order] for order in sorted(orders))
        if any(vector.ndim != 1 for vector in vectors):
            raise ValueError("support moment vectors must be one-dimensional")
        if any(vector.shape != self.snap.shape for vector in vectors):
            raise ValueError("support moment vectors must share one length")
        if any(vector.device != self.snap.device for vector in vectors):
            raise ValueError("support moment vectors must share one device")
        if any(not bool(torch.isfinite(vector).all()) for vector in vectors):
            raise ValueError("support moment vectors must be finite")
        if any(bool((vector < 0).any()) for vector in vectors):
            raise ValueError("support moment vectors must be nonnegative")

    @property
    def orders(self) -> tuple[float, ...]:
        """Return the finite moment orders, ascending."""
        return tuple(sorted(self.e))

    def as_vectors(self) -> dict[str, torch.Tensor]:
        """Return the bundle as a flat name-to-vector mapping."""
        vectors = {"snap": self.snap, "e-pinf": self.einf, "r-pinf": self.rinf}
        for order in self.orders:
            tag = _tag(order)
            vectors[f"e-p{tag}"] = self.e[order]
            vectors[f"eprime-p{tag}"] = self.eprime[order]
            vectors[f"r-p{tag}"] = self.r[order]
        return vectors

    @classmethod
    def from_vectors(cls, vectors: dict[str, torch.Tensor]) -> SupportMomentBundle:
        """Rebuild a bundle from that mapping, raising unless the roster is exact."""
        orders: set[float] = set()
        for name in vectors:
            match = re.fullmatch(r"e-p([0-9]+(?:p[0-9]+)?)", name)
            if match:
                orders.add(float(match.group(1).replace("p", ".")))
        expected = {"snap", "e-pinf", "r-pinf"}
        for order in orders:
            tag = _tag(order)
            expected.update((f"e-p{tag}", f"eprime-p{tag}", f"r-p{tag}"))
        if set(vectors) != expected:
            raise ValueError("score bank does not contain the canonical support moment roster")
        return cls(
            snap=vectors["snap"],
            e={order: vectors[f"e-p{_tag(order)}"] for order in orders},
            eprime={order: vectors[f"eprime-p{_tag(order)}"] for order in orders},
            r={order: vectors[f"r-p{_tag(order)}"] for order in orders},
            einf=vectors["e-pinf"],
            rinf=vectors["r-pinf"],
        )


def _ratio_power(
    numerator: torch.Tensor, denominator: torch.Tensor, exponent: float
) -> torch.Tensor:
    """Raise a positive ratio to a power, with one as the zero-over-zero limit."""
    numerator = numerator.float()
    denominator = denominator.float()
    tiny = torch.finfo(torch.float32).tiny
    both_zero = (numerator == 0) & (denominator == 0)
    powered = (numerator / denominator.clamp_min(tiny)).clamp_min(tiny).pow(exponent)
    return torch.where(both_zero, torch.ones_like(powered), powered)


def correction(bundle: SupportMomentBundle, order: Order) -> torch.Tensor:
    """Return C_p, including the exact global-max order-infinity endpoint."""
    if order == "inf":
        return _ratio_power(bundle.rinf, bundle.einf, 1.0)
    finite = float(order)
    if finite not in bundle.e:
        raise ValueError(f"moment order {finite:g} is absent from the capture")
    return _ratio_power(bundle.r[finite], bundle.e[finite], 1.0 / (finite - 1.0))


def concentration_and_alignment(
    bundle: SupportMomentBundle,
    order: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor C_p into row concentration and pooling disagreement."""
    if order not in bundle.e:
        raise ValueError(f"moment order {order:g} is absent from the capture")
    root = 1.0 / (order - 1.0)
    pure = _ratio_power(bundle.r[order], bundle.eprime[order], root)
    alignment = _ratio_power(bundle.eprime[order], bundle.e[order], root)
    return pure, alignment


def compose_score(
    bundle: SupportMomentBundle,
    *,
    order: Order,
    alpha: float,
) -> torch.Tensor:
    """Return mean query attention times the correction raised to alpha."""
    if alpha < 0 or not math.isfinite(alpha):
        raise ValueError("alpha must be finite and nonnegative")
    if alpha == 0.0:
        return bundle.snap
    return bundle.snap.float() * correction(bundle, order).pow(alpha)


def parse_selector(name: str) -> tuple[Order, float] | None:
    """Parse a selector name into (order, alpha), or None; plain snap is alpha zero."""
    if name == "snap":
        return 2.0, 0.0
    match = re.fullmatch(r"support-p([0-9]+(?:p[0-9]+)?|inf)-a([0-9]+(?:p[0-9]+)?)", name)
    if match is None:
        return None
    raw_order, raw_alpha = match.groups()
    order: Order = "inf" if raw_order == "inf" else float(raw_order.replace("p", "."))
    return order, float(raw_alpha.replace("p", "."))


def selector_score(bundle: SupportMomentBundle, name: str) -> torch.Tensor:
    """Score one bundle under a selector name."""
    parsed = parse_selector(name)
    if parsed is None:
        raise ValueError(f"selector {name!r} is not query-aware")
    order, alpha = parsed
    return compose_score(bundle, order=order, alpha=alpha)


def support_corrected_scores(
    snap: torch.Tensor,
    energy: torch.Tensor,
    row_energy: torch.Tensor,
    *,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Correct mean query attention by the query-row support ratio ``row_energy / energy``.

    ``alpha`` controls the correction strength: raise the ratio to alpha, then
    multiply mean query attention by it. Composition uses float32 with the dtype's
    tiny as its floor. See docs/selectors.md, CacheBack.
    """
    if alpha <= 0 or not math.isfinite(alpha):
        raise ValueError("alpha must be finite and positive")
    if snap.ndim != 1 or energy.ndim != 1 or row_energy.ndim != 1:
        raise ValueError("support-corrected scores must be one-dimensional")
    if snap.shape != energy.shape or snap.shape != row_energy.shape:
        raise ValueError("support-corrected score vectors must have the same shape")
    if snap.device != energy.device or snap.device != row_energy.device:
        raise ValueError("support-corrected score vectors must use the same device")

    snap_float = snap.float()
    energy_float = energy.float()
    row_float = row_energy.float()
    floor = torch.finfo(energy_float.dtype).tiny
    correction_value = torch.where(
        (energy_float != 0) | (row_float != 0),
        row_float / (energy_float + floor),
        torch.ones_like(energy_float),
    )
    if alpha != 1.0:
        correction_value = correction_value.pow(alpha)
    return snap_float * correction_value


def _pin_sink(scores: torch.Tensor, n_sink: int) -> torch.Tensor:
    """Return a copy with the leading `n_sink` columns pinned to the vector maximum."""
    if n_sink < 0:
        raise ValueError(f"n_sink must be nonnegative, got {n_sink}")
    if n_sink == 0 or scores.numel() == 0:
        return scores
    pinned = scores.clone()
    pinned[: min(n_sink, int(scores.numel()))] = scores.max()
    return pinned
