"""Score adapters over the single-item capture.

Each runs one capture and returns the scored memory vectors with the leading sink
columns pinned.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from typing import Any, cast

import torch

from rcc.transforms.select.core import kernels

# Re-exported here so a caller can reach the reducer state types from one place.
from rcc.transforms.select.core.types import (
    CaptureResult,
    CaptureSpec,
    CaptureStatistics,
    EnergyAccumulator,
    EnergyChunkState,
    MutableHFCache,
    SupportMomentState,
    TokenIds,
)
from rcc.transforms.select.query_support import capture_single
from rcc.transforms.select.query_support.methods import compose, reducers
from rcc.transforms.select.query_support.methods.compose import (
    SUPPORT_DEFAULT_ORDER,
    SupportMomentBundle,
    moment_power,
)
from rcc.transforms.select.query_support.methods.single import capture_query_support

_accumulate_capture_energies = reducers.accumulate_capture_energies
_fold_capture_energies = reducers.fold_capture_energies
_finish_capture_energies = reducers.finish_capture_energies
_legacy_reducer_hook: Callable[[], Any] | None = None

__all__ = [
    "EnergyAccumulator",
    "EnergyChunkState",
    "SupportMomentBundle",
    "SupportMomentState",
    "accumulate_capture_energies",
    "capture_query_support",
    "finish_capture_energies",
    "fold_capture_energies",
    "memory_energy_votes",
    "memory_row_energy_votes",
    "memory_votes_with_energies",
    "memory_votes_with_support_moments",
    "moment_power",
]


def _validate_sink(n_sink: int) -> None:
    """Raise for a negative sink count."""
    if n_sink < 0:
        raise ValueError(f"n_sink must be nonnegative, got {n_sink}")


def _pin(value: torch.Tensor, n_sink: int) -> torch.Tensor:
    """Return a copy with the leading sink columns pinned to the vector maximum."""
    return compose._pin_sink(value, n_sink)


def _chunk_state_from_legacy(state: dict[str, Any]) -> EnergyChunkState:
    """Convert a plain row-chunk dictionary into typed reducer state."""
    raw_support = state.get("support")
    support: SupportMomentState | None = None
    if isinstance(raw_support, dict):
        typed_support = cast(dict[str, Any], raw_support)
        raw_moments = typed_support.get("row_moments", {})
        support = SupportMomentState(
            pooled_rowsum=typed_support.get("pooled_rowsum"),
            rowmax=typed_support.get("rowmax"),
            row_moments=(
                cast(dict[float, torch.Tensor], raw_moments)
                if isinstance(raw_moments, dict)
                else {}
            ),
            prepool_row2=typed_support.get("prepool_row2"),
        )
    return EnergyChunkState(
        grouped_rowsum=state.get("grouped_rowsum"),
        row_sq_sum=state.get("row_sq_sum"),
        rows=int(state.get("rows", 0)),
        support=support,
    )


def _write_legacy_state(state: dict[str, Any], typed: EnergyChunkState) -> None:
    """Write typed reducer state back into the caller's dictionary."""
    state.clear()
    if typed.grouped_rowsum is not None:
        state["grouped_rowsum"] = typed.grouped_rowsum
    if typed.row_sq_sum is not None:
        state["row_sq_sum"] = typed.row_sq_sum
    if typed.support is not None:
        state["support"] = {
            **(
                {"pooled_rowsum": typed.support.pooled_rowsum}
                if typed.support.pooled_rowsum is not None
                else {}
            ),
            **({"rowmax": typed.support.rowmax} if typed.support.rowmax is not None else {}),
            "row_moments": dict(typed.support.row_moments),
            **(
                {"prepool_row2": typed.support.prepool_row2}
                if typed.support.prepool_row2 is not None
                else {}
            ),
        }
    if typed.rows:
        state["rows"] = typed.rows


def accumulate_capture_energies(
    ctx: EnergyAccumulator, weights: torch.Tensor, *, kv_groups: int
) -> None:
    """Fold one layer's weights into the accumulator."""
    _accumulate_capture_energies(ctx, weights, kv_groups=kv_groups)


def fold_capture_energies(
    ctx: EnergyAccumulator,
    weights: torch.Tensor,
    *,
    kv_groups: int,
    state: dict[str, Any] | EnergyChunkState,
) -> None:
    """Fold one row chunk, with `state` either a dictionary or the typed state."""
    if isinstance(state, EnergyChunkState):
        _fold_capture_energies(ctx, weights, kv_groups=kv_groups, state=state)
        return
    typed = _chunk_state_from_legacy(state)
    _fold_capture_energies(ctx, weights, kv_groups=kv_groups, state=typed)
    _write_legacy_state(state, typed)


def finish_capture_energies(
    ctx: EnergyAccumulator,
    state: dict[str, Any] | EnergyChunkState,
) -> None:
    """Close one layer, with `state` either a dictionary or the typed state."""
    if isinstance(state, EnergyChunkState):
        _finish_capture_energies(ctx, state)
        return
    typed = _chunk_state_from_legacy(state)
    _finish_capture_energies(ctx, typed)
    _write_legacy_state(state, typed)


@contextmanager
def _reducer_context() -> Generator[None, None, None]:
    """Run under the installed reducer patch, if a caller set one."""
    if _legacy_reducer_hook is None:
        yield
        return
    with _legacy_reducer_hook():
        yield


def _finish_vector(
    value: torch.Tensor | None,
    memory_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Return one float32 memory vector, zeros when the capture collected none."""
    if value is None:
        # Zeros on the host: in this branch the vector is empty or discarded.
        del device
        return torch.zeros(memory_length, dtype=torch.float32)
    return value


def finish_capture_result(
    context: capture_single.CaptureContext,
    *,
    pool_kernel: int,
    spec: CaptureSpec,
    found: bool,
    device: torch.device,
) -> CaptureResult:
    """Freeze the mutable capture state into its `CaptureResult`."""
    raw_snap = _finish_vector(context.votes, context.mem_len, device)
    raw_snap = raw_snap / float(context.heads * max(1, context.hi - context.lo))
    pooled_snap = kernels.pool_scores(raw_snap, pool_kernel)
    snap = pooled_snap if spec.collect_snap else torch.zeros_like(pooled_snap)
    energy = (
        _finish_vector(context.energy_sum, context.mem_len, device)
        if spec.energy_pool_kernel is not None
        else None
    )
    row_energy = (
        _finish_vector(context.row_energy_sum, context.mem_len, device)
        if spec.row_energy_pool_kernel is not None
        else None
    )
    orders = tuple(float(order) for order in spec.support_orders)
    moments = (
        SupportMomentBundle(
            snap=snap,
            e={
                order: _finish_vector(context.support_e.get(order), context.mem_len, device)
                for order in orders
            },
            eprime={
                order: _finish_vector(context.support_eprime.get(order), context.mem_len, device)
                for order in orders
            },
            r={
                order: _finish_vector(context.support_r.get(order), context.mem_len, device)
                for order in orders
            },
            einf=_finish_vector(context.support_einf, context.mem_len, device),
            rinf=_finish_vector(context.support_rinf, context.mem_len, device),
        )
        if orders
        else None
    )
    statistics = None
    if spec.collect_statistics or context.observer is not None:
        pairs = max(1, context.pairs)
        zero = torch.zeros(context.mem_len, dtype=torch.float32, device=device)
        statistics = CaptureStatistics(
            rank_mean=(context.rank_sum / float(pairs) if context.rank_sum is not None else zero),
            nomination_fraction=(
                context.nomination_sum / float(pairs)
                if context.nomination_sum is not None
                else torch.zeros_like(zero)
            ),
            raw_snap=raw_snap,
            prepool_e2=context.support_prepool_e2,
            prepool_r2=context.support_prepool_r2,
        )
    return CaptureResult(
        snap=snap,
        found=found,
        energy=energy,
        row_energy=row_energy,
        moments=moments,
        statistics=statistics,
        layer_bank=context.layer_bank,
    )


def _capture_energy(
    model: object,
    past: MutableHFCache | None,
    judger_ids: torch.Tensor,
    judger_mask: torch.Tensor,
    question_ids: TokenIds,
    *,
    pool_kernel: int,
    n_sink: int,
    row: bool,
) -> tuple[torch.Tensor, bool]:
    """Run one energy-only capture and pin its sink columns."""
    reducers.validate_pool_kernel(pool_kernel)
    _validate_sink(n_sink)
    spec = CaptureSpec(
        collect_snap=False,
        energy_pool_kernel=None if row else pool_kernel,
        row_energy_pool_kernel=pool_kernel if row else None,
    )
    with _reducer_context():
        result = capture_query_support(
            model,
            past,
            judger_ids,
            judger_mask,
            question_ids,
            pool_kernel=1,
            spec=spec,
        )
    value = result.row_energy if row else result.energy
    if value is None:
        raise AssertionError("energy capture returned no requested vector")
    return _pin(value, n_sink), result.found


def memory_energy_votes(
    model: object,
    past: MutableHFCache | None,
    judger_ids: torch.Tensor,
    judger_mask: torch.Tensor,
    question_ids: TokenIds,
    *,
    pool_kernel: int = 7,
    n_sink: int = 1,
) -> tuple[torch.Tensor, bool]:
    """Score memory columns with stream-preserving energy-based query attention."""
    return _capture_energy(
        model,
        past,
        judger_ids,
        judger_mask,
        question_ids,
        pool_kernel=pool_kernel,
        n_sink=n_sink,
        row=False,
    )


def memory_row_energy_votes(
    model: object,
    past: MutableHFCache | None,
    judger_ids: torch.Tensor,
    judger_mask: torch.Tensor,
    question_ids: TokenIds,
    *,
    pool_kernel: int = 7,
    n_sink: int = 1,
) -> tuple[torch.Tensor, bool]:
    """Score memory columns with the question rows preserved through the square."""
    return _capture_energy(
        model,
        past,
        judger_ids,
        judger_mask,
        question_ids,
        pool_kernel=pool_kernel,
        n_sink=n_sink,
        row=True,
    )


def memory_votes_with_energies(
    model: object,
    past: MutableHFCache | None,
    judger_ids: torch.Tensor,
    judger_mask: torch.Tensor,
    question_ids: TokenIds,
    *,
    pool_kernel: int = 7,
    n_sink: int = 1,
    layer_bank_sink: list[Any] | None = None,
    bank_include_sliding: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Produce unshared snap, Energy, and Row-Energy vectors in one capture.

    With a bank sink the capture also accumulates the default finite support order,
    since a bank collected without those moments cannot be scored under its own name.
    """
    reducers.validate_pool_kernel(pool_kernel)
    _validate_sink(n_sink)
    with _reducer_context():
        result = capture_query_support(
            model,
            past,
            judger_ids,
            judger_mask,
            question_ids,
            pool_kernel=pool_kernel,
            spec=CaptureSpec(
                energy_pool_kernel=pool_kernel,
                row_energy_pool_kernel=pool_kernel,
                support_orders=() if layer_bank_sink is None else (SUPPORT_DEFAULT_ORDER,),
                collect_layer_bank=layer_bank_sink is not None,
                bank_include_sliding=bank_include_sliding,
            ),
        )
    if result.energy is None or result.row_energy is None:
        raise AssertionError("fused capture returned no energy vectors")
    if layer_bank_sink is not None:
        if result.layer_bank is None:
            raise AssertionError("fused capture returned no requested layer bank")
        layer_bank_sink.append(result.layer_bank)
    return (
        _pin(result.snap, n_sink),
        _pin(result.energy, n_sink),
        _pin(result.row_energy, n_sink),
        result.found,
    )


def _orders(values: Sequence[float]) -> tuple[float, ...]:
    """Return the support orders sorted, raising unless they are unique and above one."""
    orders = tuple(float(value) for value in values)
    if (
        not orders
        or len(set(orders)) != len(orders)
        or any(order <= 1 or not math.isfinite(order) for order in orders)
    ):
        raise ValueError("support orders must be unique finite values greater than one")
    return tuple(sorted(orders))


def _pin_bundle(bundle: SupportMomentBundle, n_sink: int) -> SupportMomentBundle:
    """Pin the sink columns of every vector in a moment bundle."""
    return SupportMomentBundle(
        snap=_pin(bundle.snap, n_sink),
        e={order: _pin(value, n_sink) for order, value in bundle.e.items()},
        eprime={order: _pin(value, n_sink) for order, value in bundle.eprime.items()},
        r={order: _pin(value, n_sink) for order, value in bundle.r.items()},
        einf=_pin(bundle.einf, n_sink),
        rinf=_pin(bundle.rinf, n_sink),
    )


def memory_votes_with_support_moments(
    model: object,
    past: MutableHFCache | None,
    judger_ids: torch.Tensor,
    judger_mask: torch.Tensor,
    question_ids: TokenIds,
    *,
    orders: Sequence[float],
    pool_kernel: int = 7,
    n_sink: int = 0,
    consume_past: bool = False,
) -> tuple[SupportMomentBundle, bool]:
    """Capture the finite support moments and the infinity endpoint."""
    reducers.validate_pool_kernel(pool_kernel)
    _validate_sink(n_sink)
    finite_orders = _orders(orders)
    with _reducer_context():
        result = capture_query_support(
            model,
            past,
            judger_ids,
            judger_mask,
            question_ids,
            pool_kernel=pool_kernel,
            spec=CaptureSpec(
                energy_pool_kernel=pool_kernel,
                row_energy_pool_kernel=pool_kernel,
                support_orders=finite_orders,
            ),
            consume_past=consume_past,
        )
    if result.moments is None:
        raise AssertionError("support capture returned no moment bundle")
    return _pin_bundle(result.moments, n_sink), result.found
