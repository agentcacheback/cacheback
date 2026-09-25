"""The flat-interleave payload layout: which rows a receiver reads, in what order.

One flat block per worker, its kept evidence followed by its rolled cargo. The
receiver reads embedding rows only: no delimiter row and no foreign KV.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

SPAN_WIDTH = 16
QWEN_BLOCK_KIND = "qwen_interleave"
FLAT_INTERLEAVE_LAYOUT = "flat-interleave-v1"


@dataclass(frozen=True)
class ArmLayout:
    """One arm's evidence spans in order, with an optional tier boundary."""

    span_order: tuple[tuple[int, int], ...]
    tier_start: int
    cargo_first: bool = False

    def __post_init__(self) -> None:
        """Raise for an out-of-range tier boundary or a repeated span."""
        if not 0 <= self.tier_start <= len(self.span_order):
            raise ValueError("tier start is outside the span order")
        if len(set(self.span_order)) != len(self.span_order):
            raise ValueError("span order repeats a span")


def evidence_spans(keep: tuple[int, ...], prompt_tokens: int) -> dict[int, list[int]]:
    """Group one worker's kept evidence positions into W16 grid spans."""
    spans: dict[int, list[int]] = {}
    for position in keep:
        if position < prompt_tokens:
            spans.setdefault(position // SPAN_WIDTH, []).append(position)
    return spans


def span_table(
    keeps: tuple[tuple[int, ...], ...],
    prompt_tokens_by_worker: tuple[int, ...],
) -> dict[tuple[int, int], list[int]]:
    """Return kept evidence spans on one shared ``(worker, block)`` axis."""
    return {
        (worker, block): positions
        for worker, keep in enumerate(keeps)
        for block, positions in evidence_spans(keep, prompt_tokens_by_worker[worker]).items()
    }


def source_layout(
    keeps: tuple[tuple[int, ...], ...],
    prompt_tokens_by_worker: tuple[int, ...],
) -> ArmLayout:
    """Order evidence spans by worker and then by their source position."""
    order = tuple(
        (worker, block)
        for worker in range(len(keeps))
        for block in sorted(evidence_spans(keeps[worker], prompt_tokens_by_worker[worker]))
    )
    return ArmLayout(span_order=order, tier_start=len(order))


def flat_interleave_layout(
    keeps: tuple[tuple[int, ...], ...],
    prompt_tokens_by_worker: tuple[int, ...],
) -> ArmLayout:
    """Return the flat-interleave source order, the same for every family."""
    return source_layout(keeps, prompt_tokens_by_worker)


def interleaved_blocks(
    prompt_ids: tuple[torch.Tensor, ...],
    layout: ArmLayout,
    keeps: tuple[tuple[int, ...], ...],
    rolled_by_worker: tuple[torch.Tensor, ...],
    embed: Callable[[list[int]], torch.Tensor],
) -> tuple[list[torch.Tensor], list[str], dict[str, int]]:
    """Emit one flat block per worker: its kept evidence, then its rolled cargo."""
    prompt_tokens = tuple(int(ids.numel()) for ids in prompt_ids)
    if layout != flat_interleave_layout(keeps, prompt_tokens):
        raise RuntimeError(
            "the flat-interleave payload needs the per-worker source-order layout; "
            "the supplied layout seats spans this recipe never seats"
        )
    blocks: list[torch.Tensor] = []
    kinds: list[str] = []
    stats = {"delimiter_rows": 0, "tier_spans": 0, "tier_rows": 0, "kept_evidence_rows": 0}
    for worker, keep in enumerate(keeps):
        evidence = [position for position in keep if position < prompt_tokens[worker]]
        parts: list[torch.Tensor] = []
        if evidence:
            parts.append(embed([int(prompt_ids[worker][position]) for position in evidence]))
            stats["kept_evidence_rows"] += len(evidence)
        parts.append(rolled_by_worker[worker])
        blocks.append(torch.cat(parts))
        kinds.append(f"worker_memory_w{worker}")
    return blocks, kinds, stats


__all__ = (
    "FLAT_INTERLEAVE_LAYOUT",
    "QWEN_BLOCK_KIND",
    "SPAN_WIDTH",
    "ArmLayout",
    "evidence_spans",
    "flat_interleave_layout",
    "interleaved_blocks",
    "source_layout",
    "span_table",
)
