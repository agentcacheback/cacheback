"""Score adapters over the single-item capture.

Each runs one capture and returns the scored memory vectors with the leading sink
columns pinned.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from rcc.selectors.core import kernels
from rcc.selectors.core.types import (
    CaptureResult,
    CaptureSpec,
    CaptureStatistics,
    MutableHFCache,
    TokenIds,
)
from rcc.selectors.query_support import capture_single
from rcc.selectors.query_support.methods import compose, reducers
from rcc.selectors.query_support.methods.compose import (
    SupportMomentBundle,
)
from rcc.selectors.query_support.methods.single import capture_query_support


def _validate_sink(n_sink: int) -> None:
    """Raise for a negative sink count."""
    if n_sink < 0:
        raise ValueError(f"n_sink must be nonnegative, got {n_sink}")


def _pin(value: torch.Tensor, n_sink: int) -> torch.Tensor:
    """Return a copy with the leading sink columns pinned to the vector maximum."""
    return compose.pin_sink(value, n_sink)


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
