"""The flat-interleave payload layout for the Gemma split route."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch

from rcc.models.gemma.contract import QWEN_PAYLOAD_LAYOUT
from rcc.topologies.fanout.layout import (
    QWEN_BLOCK_KIND,
    ArmLayout,
    evidence_spans,
    flat_interleave_layout,
    interleaved_blocks,
    span_table,
)

SOURCE_PAYLOAD_LAYOUT = "source"
PAYLOAD_LAYOUT_BY_BLOCK_KIND = {QWEN_BLOCK_KIND: QWEN_PAYLOAD_LAYOUT}


def validate_layout(
    layout: ArmLayout,
    keeps: tuple[tuple[int, ...], ...],
    prompt_tokens_by_worker: tuple[int, ...],
) -> None:
    """Refuse a layout that is not a permutation of the kept evidence spans."""
    expected = {
        (worker, block)
        for worker, keep in enumerate(keeps)
        for block in evidence_spans(keep, prompt_tokens_by_worker[worker])
    }
    if set(layout.span_order) != expected:
        raise ValueError("layout is not a permutation of the kept evidence spans")


def arm_payload_layout(
    arm: str,
    layouts_by_arm: Mapping[str, ArmLayout],
    *,
    flat_interleave: bool = False,
) -> tuple[ArmLayout, str]:
    """Return the banked flat-interleave layout for one live cut arm."""
    try:
        layout = layouts_by_arm[arm]
    except KeyError as exc:
        raise ValueError(f"{arm}: no banked payload layout for this arm") from exc
    if not flat_interleave:
        raise ValueError(f"{arm}: live Gemma payloads require flat-interleave layout")
    return layout, QWEN_BLOCK_KIND


def payload_blocks(
    block_kind: str,
    prompt_ids: tuple[torch.Tensor, ...],
    layout: ArmLayout,
    keeps: tuple[tuple[int, ...], ...],
    rolled_by_worker: tuple[torch.Tensor, ...],
    embed: Callable[[Sequence[int]], torch.Tensor],
    delimiter: torch.Tensor,
) -> tuple[list[torch.Tensor], list[str], dict[str, int]]:
    """Materialize the live flat-interleave payload."""
    if block_kind != QWEN_BLOCK_KIND:
        raise ValueError(f"unregistered Gemma payload block kind {block_kind!r}")
    return interleaved_blocks(prompt_ids, layout, keeps, rolled_by_worker, embed)


__all__ = (
    "PAYLOAD_LAYOUT_BY_BLOCK_KIND",
    "QWEN_BLOCK_KIND",
    "SOURCE_PAYLOAD_LAYOUT",
    "ArmLayout",
    "arm_payload_layout",
    "evidence_spans",
    "flat_interleave_layout",
    "interleaved_blocks",
    "payload_blocks",
    "span_table",
    "validate_layout",
)
