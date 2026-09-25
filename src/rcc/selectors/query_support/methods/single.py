"""The single-item capture entry point over a native sender cache."""

from __future__ import annotations

import torch

from rcc.selectors.core.types import (
    AttentionLayerObserver,
    CaptureResult,
    CaptureSpec,
    MutableHFCache,
    TokenIds,
)
from rcc.selectors.query_support import capture_single

__all__ = [
    "capture_query_support",
]

_DEFAULT_CAPTURE_SPEC = CaptureSpec()


def _flash_available(judger_mask: torch.Tensor) -> bool:
    """Return whether the lower-right SDPA mask can be used for this prompt."""
    return (
        bool(torch.all(judger_mask != 0))
        and capture_single.kernels.lower_right_causal_bias(1, 1) is not None
    )


def _capture(
    model: object,
    past: MutableHFCache | None,
    judger_ids: torch.Tensor,
    judger_mask: torch.Tensor,
    question_ids: TokenIds,
    *,
    pool_kernel: int,
    spec: CaptureSpec,
    observer: AttentionLayerObserver | None,
    lower_right: bool,
    collect_votes: bool | None = None,
    consume_past: bool = False,
) -> tuple[CaptureResult, capture_single.CaptureContext]:
    """Run one capture with the chosen output-attention mask."""
    return capture_single.capture_context(
        model,
        past,
        judger_ids,
        judger_mask,
        question_ids,
        pool_kernel=pool_kernel,
        spec=spec,
        observer=observer,
        lower_right=lower_right,
        collect_votes=collect_votes,
        consume_past=consume_past,
    )


def capture_query_support(
    model: object,
    past: MutableHFCache | None,
    judger_ids: torch.Tensor,
    judger_mask: torch.Tensor,
    question_ids: TokenIds,
    *,
    pool_kernel: int = 7,
    spec: CaptureSpec = _DEFAULT_CAPTURE_SPEC,
    observer: AttentionLayerObserver | None = None,
    consume_past: bool = False,
) -> CaptureResult:
    """Capture snap, energy, row energy, the moments, and the diagnostics in one pass."""
    result, _context = _capture(
        model,
        past,
        judger_ids,
        judger_mask,
        question_ids,
        pool_kernel=pool_kernel,
        spec=spec,
        observer=observer,
        lower_right=_flash_available(judger_mask),
        consume_past=consume_past,
    )
    return result
