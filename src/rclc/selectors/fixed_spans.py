"""Fixed-width span schedules: fill an exact budget with whole spans of the score vector."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as functional

if TYPE_CHECKING:
    from rclc.transport import SenderState


def _partial_interval(available: Sequence[int], room: int, peak: int) -> range:
    """Return ``room`` free positions in one contiguous interval nearest ``peak``."""
    windows = (
        (available[offset], available[offset + room - 1])
        for offset in range(len(available) - room + 1)
    )
    start, end = min(windows, key=lambda w: (abs((w[0] + w[1]) / 2 - peak), w[1] - w[0], w[0]))
    return range(start, end + 1)


def fixed_span_keep(
    scores: torch.Tensor, budget: int, protected: Iterable[int], *, span_size: int
) -> list[int]:
    """Select exactly ``budget`` positions from spans ranked by mean score.

    Protected positions are always kept; whole spans are added in descending
    mean-score order, and the overshooting span fills the rest around its peak.
    """
    length = scores.numel()
    keep = set(protected)
    if len(keep) > budget:
        raise ValueError("protected positions exceed the selection budget")
    if budget >= length:
        return list(range(length))
    values = scores.detach().float()
    starts = range(0, length, span_size)
    sums = functional.pad(values, (0, -length % span_size)).view(-1, span_size).sum(1)
    sizes = torch.tensor([min(span_size, length - start) for start in starts], device=sums.device)
    means = [float(mean) for mean in (sums / sizes).cpu()]
    for index in sorted(range(len(means)), key=lambda i: -means[i]):
        room = budget - len(keep)
        if room == 0:
            break
        start = starts[index]
        end = min(length, start + span_size)
        available = [position for position in range(start, end) if position not in keep]
        if len(available) <= room:
            keep.update(available)
        else:
            peak = start + int(values[start:end].argmax())
            keep.update(_partial_interval(available, room, peak))
    return sorted(keep)


def selects(sender: SenderState, budget: int, span_size: int) -> bool:
    """Validate selector options, returning False when the budget keeps the whole state."""
    if type(span_size) is not int or span_size < 1:
        raise ValueError("span_size must be a positive integer")
    if budget < sender.latent_steps + 1:
        raise ValueError("budget must cover the first source position and the latent tail")
    return budget < sender.input_embeds.shape[0]


def keep_spans(
    sender: SenderState, scores: torch.Tensor, budget: int, *, span_size: int
) -> list[int]:
    """Keep the first position, pinned to the top score, the latent tail and whole spans."""
    length = sender.input_embeds.shape[0]
    pinned = scores.clone()
    pinned[0] = pinned.max()
    protected = (0, *range(length - sender.latent_steps, length))
    return fixed_span_keep(pinned, budget, protected, span_size=span_size)


def span_means(scores: torch.Tensor, span_size: int) -> torch.Tensor:
    """Replace every score with the mean of its fixed-width span."""
    length = scores.numel()
    sums = functional.pad(scores.float(), (0, -length % span_size)).view(-1, span_size).sum(1)
    sizes = torch.full_like(sums, span_size)
    sizes[-1] = length - span_size * (sums.numel() - 1)
    return (sums / sizes).repeat_interleave(span_size)[:length]
