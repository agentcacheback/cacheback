"""CacheBack's request-conditioned support score and protected fixed-span selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rcc.selectors.core.kernels import ATTENTION_LOCK
from rcc.selectors.fixed_spans import fixed_span_keep
from rcc.selectors.query_support.methods.compose import compose_score, pin_sink
from rcc.selectors.query_support.methods.scorers import memory_votes_with_support_moments

if TYPE_CHECKING:
    from rcc.transport import SenderState


def cacheback(
    sender: SenderState, request_ids: torch.Tensor, budget: int, *, span_size: int = 16
) -> list[int]:
    """Select the resolved budget from validated state, keeping its first row and latent tail."""
    if type(span_size) is not int or span_size < 1:
        raise ValueError("span_size must be a positive integer")
    length = sender.input_embeds.shape[0]
    if budget < sender.latent_steps + 1:
        raise ValueError("budget must cover the first source position and the latent tail")
    if budget >= length:
        return list(range(length))
    if length + request_ids.shape[1] > sender.model.config.max_position_embeddings:
        raise ValueError("sender state plus request exceeds the model's context limit")
    with ATTENTION_LOCK, torch.inference_mode():
        moments, found = memory_votes_with_support_moments(
            sender.model,
            sender.past_key_values,
            request_ids,
            torch.ones_like(request_ids),
            request_ids[0],
            orders=(2.0,),
            n_sink=0,
            consume_past=False,
        )
        if not found:
            raise RuntimeError("the receiver request was not located in the capture")
        scores = pin_sink(compose_score(moments, order=2.0, alpha=2.0), 1)
        protected = (0, *range(length - sender.latent_steps, length))
        return fixed_span_keep(scores, budget, protected, span_size=span_size)
