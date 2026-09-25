"""Reprefill a query after a handed-off cache and greedy-decode the continuation.

By default the loop runs the full `max_new_tokens`, which makes the continuation
identical token for token to a single forward over the same sequence.
"""

from __future__ import annotations

from typing import Any, cast

import torch

from rcc.cache import KVCache


def eos_ids(model: Any) -> frozenset[int]:
    """Return the model's end-of-sequence ids, possibly none."""
    raw: Any = None
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        raw = generation_config.eos_token_id
    if raw is None:
        raw = getattr(model.config, "eos_token_id", None)
    if raw is None:
        return frozenset[int]()
    if isinstance(raw, (list, tuple)):
        return frozenset(int(v) for v in cast("list[int]", raw))
    return frozenset({int(cast("int", raw))})


def handoff(
    cache: KVCache,
    model: Any,
    query_ids: torch.Tensor,
    max_new_tokens: int = 64,
    *,
    stop_at_eos: bool = False,
) -> torch.Tensor:
    """Reprefill query_ids after cache and greedy-decode up to max_new_tokens ids.

    With `stop_at_eos` the decode halts at the first end-of-turn id and drops it, so
    the result can be shorter than `max_new_tokens`, or length zero.
    """
    if query_ids.ndim == 1:
        query_ids = query_ids.unsqueeze(0)
    if query_ids.ndim != 2 or query_ids.shape[0] != 1:
        raise ValueError(f"handoff is single-sequence, got shape {tuple(query_ids.shape)}")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
    device = query_ids.device
    start_pos = cache.length
    query_len = int(query_ids.shape[1])
    past = cache.to_hf_cache()
    attn = torch.ones(1, start_pos + query_len, dtype=torch.long, device=device)
    position_ids = torch.arange(start_pos, start_pos + query_len, device=device).unsqueeze(0)
    with torch.no_grad():
        out = model(
            input_ids=query_ids,
            attention_mask=attn,
            past_key_values=past,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
    eos = eos_ids(model) if stop_at_eos else frozenset[int]()
    return _decode_from_prefix(model, out, attn, max_new_tokens, eos)


def _decode_from_prefix(
    model: Any, out: Any, attn: torch.Tensor, max_new_tokens: int, eos: frozenset[int]
) -> torch.Tensor:
    """Greedy-decode up to max_new_tokens ids from a primed pass, stopping at any eos id."""
    device = attn.device
    cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past = out.past_key_values
    pos_next = int(attn.shape[1])
    generated: list[torch.Tensor] = []
    for _ in range(max_new_tokens):
        if eos and int(cur[0, 0]) in eos:
            break
        generated.append(cur)
        if len(generated) >= max_new_tokens:
            break
        attn = torch.cat([attn, torch.ones(1, 1, dtype=attn.dtype, device=device)], dim=1)
        position_ids = torch.full((1, 1), pos_next, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(
                input_ids=cur,
                attention_mask=attn,
                past_key_values=past,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )
        cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past = out.past_key_values
        pos_next += 1
    if not generated:
        return torch.empty(1, 0, dtype=torch.long, device=device)
    return torch.cat(generated, dim=1)
