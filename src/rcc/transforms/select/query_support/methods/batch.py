"""The batched capture entry points and the conversion of their results."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from rcc.transforms.select.core import kernels
from rcc.transforms.select.core.types import (
    BatchedPastLike,
    CaptureResult,
    TokenIds,
)
from rcc.transforms.select.query_support import capture_batch
from rcc.transforms.select.query_support.methods import reducers

__all__ = [
    "capture_query_support_batched_from_past",
]


def _results(
    items: Sequence[capture_batch.BatchedCaptureItem],
    judger_ids: Sequence[torch.Tensor],
    *,
    pool_kernel: int,
) -> tuple[CaptureResult, ...]:
    """Convert one item's raw capture state into its score vectors."""
    device = judger_ids[0].device
    results: list[CaptureResult] = []
    for item in items:
        if item.votes is None:
            live = torch.zeros(item.mem_len, dtype=torch.float32, device=device)
        else:
            live = item.votes.index_select(0, item.live_indices)
        snap = kernels.pool_scores(
            live / float(item.heads * item.question_rows),
            pool_kernel,
        )
        energy = None
        row_energy = None
        if item.energy is not None:
            if item.energy.energy_pool_kernel is not None:
                energy = reducers.finish_energy_scores(
                    item.energy.energy_sum,
                    item.mem_len,
                    sink_count=0,
                    device=device,
                )
            if item.energy.row_energy_pool_kernel is not None:
                row_energy = reducers.finish_energy_scores(
                    item.energy.row_energy_sum,
                    item.mem_len,
                    sink_count=0,
                    device=device,
                )
        results.append(
            CaptureResult(
                snap=snap,
                found=item.found,
                energy=energy,
                row_energy=row_energy,
            )
        )
    return tuple(results)


def _capture(
    model: object,
    padded: BatchedPastLike,
    judger_ids: Sequence[torch.Tensor],
    judger_masks: Sequence[torch.Tensor],
    question_ids: Sequence[TokenIds],
    *,
    pool_kernel: int,
    use_flex: bool,
    energy_pool_kernel: int | None,
    row_energy_pool_kernel: int | None,
) -> tuple[CaptureResult, ...]:
    """Run one batched capture and convert each item's state."""
    items = capture_batch.capture_batched(
        model,
        padded,
        judger_ids,
        judger_masks,
        question_ids,
        pool_kernel=pool_kernel,
        use_flex=use_flex,
        energy_pool_kernel=energy_pool_kernel,
        row_energy_pool_kernel=row_energy_pool_kernel,
    )
    return _results(items, judger_ids, pool_kernel=pool_kernel)


def capture_query_support_batched_from_past(
    model: object,
    padded: BatchedPastLike,
    judger_ids: Sequence[torch.Tensor],
    judger_masks: Sequence[torch.Tensor],
    question_ids: Sequence[TokenIds],
    *,
    pool_kernel: int = 7,
    energy_pool_kernel: int | None = None,
    row_energy_pool_kernel: int | None = None,
) -> tuple[CaptureResult, ...]:
    """Capture mean query attention and any energy vectors the caller asked for."""
    return _capture(
        model,
        padded,
        judger_ids,
        judger_masks,
        question_ids,
        pool_kernel=pool_kernel,
        use_flex=False,
        energy_pool_kernel=energy_pool_kernel,
        row_energy_pool_kernel=row_energy_pool_kernel,
    )
