"""CacheBack's request-conditioned support score and protected fixed-span selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rclc.selectors import _support
from rclc.selectors.fixed_spans import keep_spans, selects

if TYPE_CHECKING:
    from rclc.transport import SenderState


def check_context(sender: SenderState, request_ids: torch.Tensor) -> None:
    """Require the sender state plus request to fit the model's context limit."""
    length = sender.input_embeds.shape[0] + request_ids.shape[1]
    if length > sender.model.config.max_position_embeddings:
        raise ValueError("sender state plus request exceeds the model's context limit")


def finite(scores: torch.Tensor, name: str) -> torch.Tensor:
    """Return the scores after one finiteness check."""
    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError(f"{name} produced non-finite scores")
    return scores


def support_score(
    votes: torch.Tensor, energy: torch.Tensor, row_energy: torch.Tensor
) -> torch.Tensor:
    """Return pooled votes times the squared support correction, one where both are zero."""
    tiny = torch.finfo(torch.float32).tiny
    ratio = (row_energy / energy.clamp_min(tiny)).clamp_min(tiny)
    ratio = torch.where((row_energy == 0) & (energy == 0), 1.0, ratio)
    return _support.pool(votes) * ratio.pow(2.0)


def cacheback_scores(sender: SenderState, request_ids: torch.Tensor) -> torch.Tensor:
    """Score every sender position against the request, leaving the sender unchanged."""
    check_context(sender, request_ids)
    with torch.inference_mode():
        moments = _support.support_moments(sender.model, sender.past_key_values, request_ids)
        return finite(support_score(*moments), "CacheBack")


def cacheback(
    sender: SenderState, request_ids: torch.Tensor, budget: int, *, span_size: int = 16
) -> list[int]:
    """Select the resolved budget from validated state, keeping its first row and latent tail."""
    if not selects(sender, budget, span_size):
        return list(range(sender.input_embeds.shape[0]))
    scores = cacheback_scores(sender, request_ids)
    return keep_spans(sender, scores, budget, span_size=span_size)
