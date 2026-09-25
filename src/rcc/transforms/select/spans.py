"""Span keep schedule: whole-span selection on the length axis for Select.

A schedule is the keep-step analog of a scorer, mapping `(scores, budget, sink)`
to a sorted keep-index list. See docs/selectors.md, The span keep law.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, cast

import torch

KeepSchedule = Callable[[torch.Tensor, int, tuple[int, ...]], list[int]]
"""The keep-step contract: (scores, budget, sink) -> sorted ascending keep indices."""


def sink_within_budget(sink_in_range: set[int], budget: int) -> None:
    """Raise when the force-kept sink alone exceeds the budget.

    Every keep builder retains the sink unconditionally, so an oversized sink
    would return more slots than the budget and misprice the ratio.
    """
    if len(sink_in_range) > budget:
        raise ValueError(
            f"sink holds {len(sink_in_range)} in-range positions but budget is {budget}; "
            "a keep that silently exceeds its budget would misprice the compression"
        )


def _span_bounds(n: int, span_size: int) -> list[tuple[int, int]]:
    """Return near-equal contiguous spans tiling `[0, n)`, count = round(n / span_size).

    Args:
        n: Number of scored positions.
        span_size: Target span width; the span count is `round(n / span_size)`,
            clamped to `[1, n]`.

    Returns:
        Half-open `(lo, hi)` bounds whose union is exactly `[0, n)`; the last
        span absorbs any remainder.
    """
    count = max(1, min(round(n / max(1, span_size)), n))
    base = n // count
    bounds: list[tuple[int, int]] = []
    lo = 0
    for i in range(count):
        hi = n if i == count - 1 else lo + base
        bounds.append((lo, hi))
        lo = hi
    return bounds


def span_keep(
    scores: torch.Tensor,
    budget: int,
    sink: tuple[int, ...] = (0,),
    *,
    span_size: int = 32,
) -> list[int]:
    """Keep whole near-equal spans by descending mean score up to `budget`.

    Adds whole spans until the budget is spent, and gives the overshooting span one
    contiguous window around its peak. See docs/selectors.md, The span keep law.

    Args:
        scores: Per-position score vector of shape `[n]`.
        budget: Absolute keep count; clamped to `[1, n]`.
        sink: Positions always kept, counted inside the budget.
        span_size: Target span width for the near-equal tiling.

    Returns:
        Sorted ascending kept indices of length `min(budget, n)`.
    """
    n = int(scores.shape[0])
    budget = max(1, min(budget, n))
    if budget == n:
        return list(range(n))
    keep: set[int] = {s for s in sink if 0 <= s < n}
    sink_within_budget(keep, budget)
    bounds = _span_bounds(n, span_size)
    # One `.mean()` per span, read back to the host in a single stack, so the
    # call costs one device sync rather than one per span.
    stacked = torch.stack([scores[lo:hi].mean() for lo, hi in bounds])
    means = [float(m) for m in cast(Any, stacked).tolist()]
    ranked = sorted(
        ((means[index], lo, hi) for index, (lo, hi) in enumerate(bounds)),
        reverse=True,
    )
    for _mean, lo, hi in ranked:
        room = budget - len(keep)
        if room <= 0:
            break
        if hi - lo <= room:
            keep.update(range(lo, hi))
        else:
            peak = lo + int(scores[lo:hi].argmax())
            start = max(lo, min(peak - room // 2, hi - room))
            end = start + room
            # The window is `room` wide, but a sink position already kept inside
            # it adds no new slot, so the contiguous run widens outward from the
            # peak until it holds exactly `room` not-yet-kept positions.
            new_in = sum(1 for p in range(start, end) if p not in keep)
            while new_in < room:
                if end < hi:
                    new_in += end not in keep
                    end += 1
                elif start > lo:
                    start -= 1
                    new_in += start not in keep
                else:
                    break
            keep.update(range(start, end))
            if len(keep) >= budget:
                break
    if len(keep) != budget:
        raise AssertionError(f"span keep returned {len(keep)} positions for budget {budget}")
    return sorted(keep)


def span_schedule(span_size: int = 32) -> KeepSchedule:
    """Bind `span_size` into a `(scores, budget, sink)` -> keep-list schedule.

    Args:
        span_size: Target span width for the near-equal tiling.

    Returns:
        A `KeepSchedule` callable suitable for `Select`'s `schedule` argument.
    """
    return functools.partial(span_keep, span_size=span_size)
