"""QSnap: the request rows' mean attention to each sender position, kept in fixed spans."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rcc.selectors import _support
from rcc.selectors.cacheback import check_context, finite
from rcc.selectors.fixed_spans import keep_spans, selects

if TYPE_CHECKING:
    from rcc.transport import SenderState


def qsnap_scores(sender: SenderState, request_ids: torch.Tensor) -> torch.Tensor:
    """Return pooled query attention summed over global layers, leaving the sender unchanged."""
    check_context(sender, request_ids)
    with torch.inference_mode():
        votes = _support.mean_query_attention(sender.model, sender.past_key_values, request_ids)
        return finite(_support.pool(votes), "QSnap")


def qsnap(
    sender: SenderState, request_ids: torch.Tensor, budget: int, *, span_size: int = 16
) -> list[int]:
    """Select the budget by mean query attention, keeping the first row and latent tail."""
    if not selects(sender, budget, span_size):
        return list(range(sender.input_embeds.shape[0]))
    scores = qsnap_scores(sender, request_ids)
    return keep_spans(sender, scores, budget, span_size=span_size)
