"""Stream-preserving reducers for query-support attention captures.

One layer's question-row attention reduces to a `[mem_len]` score vector with each
KV stream kept separate until the final sum.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from rcc.transforms.select.core.types import (
    EnergyAccumulator,
    EnergyChunkState,
    SupportMomentState,
)
from rcc.transforms.select.query_support.methods.compose import moment_power

__all__ = [
    "EnergyAccumulator",
    "EnergyChunkState",
    "SupportMomentState",
    "accumulate_capture_energies",
    "finish_capture_energies",
    "finish_energy_scores",
    "fold_capture_energies",
    "merge_layer_energies",
    "normalized_row_energy_scores",
    "normalized_stream_energy_scores",
    "row_energy_scores",
    "stream_energy_scores",
    "stream_pool_sum_scores",
    "validate_pool_kernel",
]


def validate_pool_kernel(pool_kernel: int) -> None:
    """Raise unless the pooling kernel is odd and positive, which keeps tokens aligned."""
    if pool_kernel < 1 or pool_kernel % 2 == 0:
        raise ValueError(f"pool kernel must be a positive odd integer, got {pool_kernel}")


def _grouped_memory(
    weights: torch.Tensor,
    *,
    mem_len: int,
    kv_groups: int,
) -> torch.Tensor:
    """Check the attention geometry and average query heads within each KV head."""
    if weights.ndim != 3:
        raise ValueError(
            "weights must be three-dimensional [query_heads, question_rows, keys], "
            f"got {tuple(weights.shape)}"
        )
    query_heads, question_rows, keys = (int(size) for size in weights.shape)
    if question_rows < 1:
        raise ValueError("weights must contain at least one question row")
    if mem_len < 0 or mem_len > keys:
        raise ValueError(f"mem_len {mem_len} is outside the key length {keys}")
    if kv_groups < 1 or query_heads % kv_groups:
        raise ValueError(f"query heads {query_heads} must be divisible by kv_groups {kv_groups}")
    if mem_len == 0:
        return torch.zeros(
            query_heads // kv_groups,
            question_rows,
            0,
            dtype=torch.float32,
            device=weights.device,
        )

    memory = weights[:, :, :mem_len].to(torch.float32)
    # unflatten returns a view, so the values are unchanged.
    grouped = torch.unflatten(
        memory,
        0,
        (query_heads // kv_groups, kv_groups),
    )
    return grouped.mean(dim=1)


def _pool_streams(streams: torch.Tensor, pool_kernel: int) -> torch.Tensor:
    """Max-pool the final token axis independently for every leading stream."""
    validate_pool_kernel(pool_kernel)
    if streams.shape[-1] == 0 or pool_kernel == 1:
        return streams
    original_shape = streams.shape
    flat = streams.reshape(-1, 1, original_shape[-1])
    pooled = functional.max_pool1d(
        flat,
        kernel_size=pool_kernel,
        stride=1,
        padding=pool_kernel // 2,
    )[..., : original_shape[-1]]
    return pooled.reshape(original_shape)


def _sum_unit_energy_streams(stream_energy: torch.Tensor) -> torch.Tensor:
    """Give each leading stream unit tokenwise energy before summing streams."""
    totals = stream_energy.sum(dim=-1, keepdim=True)
    normalized = torch.where(
        totals > 0,
        stream_energy / totals.clamp_min(torch.finfo(stream_energy.dtype).tiny),
        torch.zeros_like(stream_energy),
    )
    return normalized.sum(dim=0)


def _elementwise_max(prior: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
    """Accumulate one tokenwise maximum without inventing a zero baseline."""
    return value if prior is None else torch.maximum(prior, value)


def stream_pool_sum_scores(
    weights: torch.Tensor,
    *,
    mem_len: int,
    kv_groups: int,
    pool_kernel: int = 7,
) -> torch.Tensor:
    """Pool each KV stream and combine the streams linearly, without the square."""
    grouped = _grouped_memory(weights, mem_len=mem_len, kv_groups=kv_groups)
    streams = grouped.mean(dim=1)
    return _pool_streams(streams, pool_kernel).sum(dim=0)


def stream_energy_scores(
    weights: torch.Tensor,
    *,
    mem_len: int,
    kv_groups: int,
    pool_kernel: int = 7,
) -> torch.Tensor:
    """Reduce one layer's question attention to squared energy density.

    Args:
        weights: Attention weights shaped `[query_heads, question_rows, keys]`.
        mem_len: Leading key columns belonging to the memory being scored.
        kv_groups: Query heads mapped to each KV head under grouped-query attention.
        pool_kernel: Odd local max-pooling width applied independently per KV stream.

    Returns:
        Float32 token scores of shape `[mem_len]`, summed over this layer's KV heads.
    """
    grouped = _grouped_memory(weights, mem_len=mem_len, kv_groups=kv_groups)
    streams = grouped.mean(dim=1)
    streams = _pool_streams(streams, pool_kernel)
    return streams.square().sum(dim=0)


def row_energy_scores(
    weights: torch.Tensor,
    *,
    mem_len: int,
    kv_groups: int,
    pool_kernel: int = 7,
) -> torch.Tensor:
    """Score with the query rows preserved through the pool and the square.

    Query heads sharing one KV head are still averaged, then each question row is
    pooled and squared on its own; averaging at the end keeps duplication invariance.
    """
    grouped = _grouped_memory(weights, mem_len=mem_len, kv_groups=kv_groups)
    pooled = _pool_streams(grouped, pool_kernel)
    return pooled.square().mean(dim=1).sum(dim=0)


def normalized_stream_energy_scores(
    weights: torch.Tensor,
    *,
    mem_len: int,
    kv_groups: int,
    pool_kernel: int = 7,
) -> torch.Tensor:
    """Energy-based query attention with unit energy per layer-by-KV-head stream."""
    grouped = _grouped_memory(weights, mem_len=mem_len, kv_groups=kv_groups)
    pooled = _pool_streams(grouped.mean(dim=1), pool_kernel)
    return _sum_unit_energy_streams(pooled.square())


def normalized_row_energy_scores(
    weights: torch.Tensor,
    *,
    mem_len: int,
    kv_groups: int,
    pool_kernel: int = 7,
) -> torch.Tensor:
    """Row-Energy with unit total energy per layer-by-KV-head stream."""
    grouped = _grouped_memory(weights, mem_len=mem_len, kv_groups=kv_groups)
    pooled = _pool_streams(grouped, pool_kernel)
    stream_energy = pooled.square().mean(dim=1)
    return _sum_unit_energy_streams(stream_energy)


def accumulate_capture_energies(
    accumulator: EnergyAccumulator,
    weights: torch.Tensor,
    *,
    kv_groups: int,
) -> None:
    """Fold one layer's question-row softmax into typed accumulators.

    The GQA grouping runs once and both reductions read the shared grouped tensor,
    each collapsing to one `[memory]` float32 vector; an unset kernel is left alone.
    """
    energy_kernel = accumulator.energy_pool_kernel
    row_kernel = accumulator.row_energy_pool_kernel
    support_orders = tuple(float(order) for order in accumulator.support_orders)
    if energy_kernel is None and row_kernel is None and not support_orders:
        return
    if support_orders and (energy_kernel is None or row_kernel != energy_kernel):
        raise RuntimeError("generalized support capture requires one shared pooling kernel")
    grouped = _grouped_memory(weights, mem_len=accumulator.mem_len, kv_groups=kv_groups)
    if energy_kernel is not None:
        validate_pool_kernel(energy_kernel)
        energy = _pool_streams(grouped.mean(dim=1), energy_kernel).square().sum(dim=0)
        accumulator.energy_sum = (
            energy if accumulator.energy_sum is None else accumulator.energy_sum + energy
        )
    if row_kernel is not None:
        validate_pool_kernel(row_kernel)
        pooled = _pool_streams(grouped, row_kernel)
        if pooled is grouped:
            row = pooled.square().mean(dim=1).sum(dim=0)
        else:
            rows = int(pooled.shape[1])
            row = pooled.square_().sum(dim=(0, 1)) / rows
        accumulator.row_energy_sum = (
            row if accumulator.row_energy_sum is None else accumulator.row_energy_sum + row
        )
    if support_orders:
        if energy_kernel is None or row_kernel is None:
            raise AssertionError("support capture lost its pooling kernel")
        pooled = _pool_streams(grouped, row_kernel)
        pooled_mean = _pool_streams(grouped.mean(dim=1), energy_kernel)
        row_mean = pooled.mean(dim=1)
        for order in support_orders:
            _accumulate_moment(
                accumulator.support_e,
                order,
                moment_power(pooled_mean, order).sum(dim=0),
            )
            _accumulate_moment(
                accumulator.support_eprime,
                order,
                moment_power(row_mean, order).sum(dim=0),
            )
            _accumulate_moment(
                accumulator.support_r,
                order,
                moment_power(pooled, order).mean(dim=1).sum(dim=0),
            )
        einf = pooled_mean.amax(dim=0)
        rinf = pooled.amax(dim=(0, 1))
        accumulator.support_einf = _elementwise_max(accumulator.support_einf, einf)
        accumulator.support_rinf = _elementwise_max(accumulator.support_rinf, rinf)
        if 2.0 in support_orders:
            raw_mean = grouped.mean(dim=1)
            pre_e2 = moment_power(raw_mean, 2.0).sum(dim=0)
            pre_r2 = moment_power(grouped, 2.0).mean(dim=1).sum(dim=0)
            accumulator.support_prepool_e2 = (
                pre_e2
                if accumulator.support_prepool_e2 is None
                else accumulator.support_prepool_e2 + pre_e2
            )
            accumulator.support_prepool_r2 = (
                pre_r2
                if accumulator.support_prepool_r2 is None
                else accumulator.support_prepool_r2 + pre_r2
            )


def _accumulate_moment(
    target: dict[float, torch.Tensor],
    order: float,
    value: torch.Tensor,
) -> None:
    """Add one layer vector into a typed moment dictionary."""
    prior = target.get(order)
    target[order] = value if prior is None else prior + value


def fold_capture_energies(
    accumulator: EnergyAccumulator,
    weights: torch.Tensor,
    *,
    kv_groups: int,
    state: EnergyChunkState,
) -> None:
    """Fold one row chunk of a layer's softmax into typed layer state.

    The row-chunked twin of `accumulate_capture_energies`. The energy branch folds a row
    sum per chunk because its mean precedes the square; the row-energy squares just add.
    """
    energy_kernel = accumulator.energy_pool_kernel
    row_kernel = accumulator.row_energy_pool_kernel
    support_orders = tuple(float(order) for order in accumulator.support_orders)
    if energy_kernel is None and row_kernel is None and not support_orders:
        return
    if support_orders and (energy_kernel is None or row_kernel != energy_kernel):
        raise RuntimeError("generalized support capture requires one shared pooling kernel")
    grouped = _grouped_memory(weights, mem_len=accumulator.mem_len, kv_groups=kv_groups)
    if energy_kernel is not None:
        validate_pool_kernel(energy_kernel)
        rowsum = grouped.sum(dim=1)
        state.grouped_rowsum = (
            rowsum if state.grouped_rowsum is None else state.grouped_rowsum + rowsum
        )
    pooled: torch.Tensor | None = None
    if row_kernel is not None:
        validate_pool_kernel(row_kernel)
        pooled = _pool_streams(grouped, row_kernel)
        if pooled is grouped or support_orders:
            square = pooled.square().sum(dim=(0, 1))
        else:
            square = pooled.square_().sum(dim=(0, 1))
        state.row_sq_sum = square if state.row_sq_sum is None else state.row_sq_sum + square
    if support_orders:
        if pooled is None:
            raise AssertionError("support capture lost its row-pooled streams")
        support = state.support
        if support is None:
            support = SupportMomentState()
            state.support = support
        pooled_rowsum = pooled.sum(dim=1)
        support.pooled_rowsum = (
            pooled_rowsum
            if support.pooled_rowsum is None
            else support.pooled_rowsum + pooled_rowsum
        )
        rowmax = pooled.amax(dim=1)
        support.rowmax = rowmax if support.rowmax is None else torch.maximum(support.rowmax, rowmax)
        for order in support_orders:
            moment = moment_power(pooled, order).sum(dim=(0, 1))
            _accumulate_moment(support.row_moments, order, moment)
        if 2.0 in support_orders:
            raw2 = moment_power(grouped, 2.0).sum(dim=(0, 1))
            support.prepool_row2 = (
                raw2 if support.prepool_row2 is None else support.prepool_row2 + raw2
            )
    state.rows += int(weights.shape[1])


def finish_capture_energies(
    accumulator: EnergyAccumulator,
    state: EnergyChunkState,
) -> None:
    """Close one layer of row-chunked capture by folding typed state into accumulators.

    Produces the same layer vectors as one `accumulate_capture_energies`
    call over the full row set, up to float summation order across chunks.
    """
    rows = state.rows
    if rows < 1:
        return
    if accumulator.energy_pool_kernel is not None and state.grouped_rowsum is not None:
        energy = (
            _pool_streams(
                state.grouped_rowsum / rows,
                accumulator.energy_pool_kernel,
            )
            .square()
            .sum(dim=0)
        )
        accumulator.energy_sum = (
            energy if accumulator.energy_sum is None else accumulator.energy_sum + energy
        )
    if accumulator.row_energy_pool_kernel is not None and state.row_sq_sum is not None:
        row = state.row_sq_sum / rows
        accumulator.row_energy_sum = (
            row if accumulator.row_energy_sum is None else accumulator.row_energy_sum + row
        )
    support_orders = tuple(float(order) for order in accumulator.support_orders)
    support = state.support
    if support_orders and support is not None:
        if state.grouped_rowsum is None:
            raise RuntimeError("generalized support capture lost its unpooled row sum")
        energy_kernel = accumulator.energy_pool_kernel
        if energy_kernel is None or support.pooled_rowsum is None or support.rowmax is None:
            raise RuntimeError("generalized support capture lost its pooling kernel")
        pooled_mean = _pool_streams(state.grouped_rowsum / rows, energy_kernel)
        row_mean = support.pooled_rowsum / rows
        for order in support_orders:
            _accumulate_moment(
                accumulator.support_e,
                order,
                moment_power(pooled_mean, order).sum(dim=0),
            )
            _accumulate_moment(
                accumulator.support_eprime,
                order,
                moment_power(row_mean, order).sum(dim=0),
            )
            moment = support.row_moments.get(order)
            if moment is None:
                raise RuntimeError("generalized support capture lost a row moment")
            _accumulate_moment(accumulator.support_r, order, moment / rows)
        einf = pooled_mean.amax(dim=0)
        rinf = support.rowmax.amax(dim=0)
        accumulator.support_einf = _elementwise_max(accumulator.support_einf, einf)
        accumulator.support_rinf = _elementwise_max(accumulator.support_rinf, rinf)
        if 2.0 in support_orders:
            if support.prepool_row2 is None:
                raise RuntimeError("generalized support capture lost pre-pool moments")
            pre_e2 = moment_power(state.grouped_rowsum / rows, 2.0).sum(dim=0)
            pre_r2 = support.prepool_row2 / rows
            accumulator.support_prepool_e2 = (
                pre_e2
                if accumulator.support_prepool_e2 is None
                else accumulator.support_prepool_e2 + pre_e2
            )
            accumulator.support_prepool_r2 = (
                pre_r2
                if accumulator.support_prepool_r2 is None
                else accumulator.support_prepool_r2 + pre_r2
            )


def _accumulate_sum(prior: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
    """Add one vector into an accumulator slot without a zero baseline."""
    return value if prior is None else prior + value


def merge_layer_energies(
    accumulator: EnergyAccumulator,
    layer: EnergyAccumulator,
) -> None:
    """Add one finished layer's vectors into the capture-wide accumulator.

    The terms are added in the same order and with the same reductions
    `finish_capture_energies` applies directly, so the two paths are bit-identical.
    """
    if layer.energy_sum is not None:
        accumulator.energy_sum = _accumulate_sum(accumulator.energy_sum, layer.energy_sum)
    if layer.row_energy_sum is not None:
        accumulator.row_energy_sum = _accumulate_sum(
            accumulator.row_energy_sum, layer.row_energy_sum
        )
    for target, source in (
        (accumulator.support_e, layer.support_e),
        (accumulator.support_eprime, layer.support_eprime),
        (accumulator.support_r, layer.support_r),
    ):
        for order, value in source.items():
            _accumulate_moment(target, order, value)
    if layer.support_einf is not None:
        accumulator.support_einf = _elementwise_max(accumulator.support_einf, layer.support_einf)
    if layer.support_rinf is not None:
        accumulator.support_rinf = _elementwise_max(accumulator.support_rinf, layer.support_rinf)
    if layer.support_prepool_e2 is not None:
        accumulator.support_prepool_e2 = _accumulate_sum(
            accumulator.support_prepool_e2, layer.support_prepool_e2
        )
    if layer.support_prepool_r2 is not None:
        accumulator.support_prepool_r2 = _accumulate_sum(
            accumulator.support_prepool_r2, layer.support_prepool_r2
        )


def finish_energy_scores(
    energy_sum: torch.Tensor | None,
    memory_length: int,
    sink_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Materialise an accumulated energy vector and pin leading sink columns."""
    scores = energy_sum
    if scores is None:
        scores = torch.zeros(memory_length, dtype=torch.float32, device=device)
    if sink_count > 0 and memory_length > 0:
        scores = scores.clone()
        scores[: min(sink_count, memory_length)] = scores.max()
    return scores
