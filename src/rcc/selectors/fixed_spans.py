"""Fixed-width span schedules: fill an exact budget with whole spans of the score vector."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

__all__ = ["fixed_span_bounds", "fixed_span_keep"]


def fixed_span_bounds(length: int, span_size: int) -> tuple[tuple[int, int], ...]:
    """Return the fixed-width span grid over ``range(length)``."""
    if length < 0:
        raise ValueError("length must be non-negative")
    if span_size <= 0:
        raise ValueError("span_size must be positive")
    return tuple((start, min(length, start + span_size)) for start in range(0, length, span_size))


def _partial_interval(available: Sequence[int], room: int, peak: int) -> range:
    """Return ``room`` free positions in one contiguous interval nearest ``peak``."""
    if room <= 0 or room > len(available):
        raise ValueError("partial interval room must fit the available positions")
    candidates: list[tuple[tuple[float, int, int], range]] = []
    for offset in range(len(available) - room + 1):
        start = available[offset]
        end = available[offset + room - 1]
        center = (start + end) / 2
        candidates.append(((abs(center - peak), end - start, start), range(start, end + 1)))
    return min(candidates, key=lambda item: item[0])[1]


def fixed_span_keep(
    scores: torch.Tensor,
    budget: int,
    protected: Iterable[int] = (0,),
    *,
    span_size: int = 32,
) -> list[int]:
    """Select exactly ``budget`` positions from spans ranked by mean score.

    Protected in-range positions are always kept; whole spans are added in descending
    mean-score order, and the overshooting span fills the rest around its peak.
    """
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    length = int(scores.numel())
    if length == 0:
        return []
    if span_size <= 0:
        raise ValueError("span_size must be positive")

    target = min(length, max(1, int(budget)))
    protected_set = {int(position) for position in protected if 0 <= int(position) < length}
    if len(protected_set) > target:
        raise ValueError("protected positions exceed the selection budget")
    if target == length:
        return list(range(length))

    bounds = fixed_span_bounds(length, span_size)
    ranked = sorted(
        range(len(bounds)),
        key=lambda index: (
            -float(scores[bounds[index][0] : bounds[index][1]].float().mean()),
            index,
        ),
    )
    keep = set(protected_set)
    score_values = scores.float()
    for span_index in ranked:
        room = target - len(keep)
        if room == 0:
            break
        start, end = bounds[span_index]
        available = [position for position in range(start, end) if position not in keep]
        if not available:
            continue
        if len(available) <= room:
            keep.update(available)
            continue

        peak = start + int(torch.argmax(score_values[start:end]).item())
        keep.update(_partial_interval(available, room, peak))
        break

    if len(keep) != target:
        raise RuntimeError(f"fixed-span selection emitted {len(keep)} positions, expected {target}")
    return sorted(keep)
