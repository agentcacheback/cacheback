"""H2O scoring: the attention mass each cached position accumulates.

The attention every query row sends to a position is summed in row chunks, so
the full attention matrix never materializes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch

from rcc.transforms.select.core import kernels
from rcc.transforms.select.core.types import MutableHFCache

__all__ = ["chunk_accumulated_scores", "h2o_row_scores", "memory_h2o_scores"]

_CAPTURE_IMPL = "rcc_h2o_score_capture"


def h2o_row_scores(weights: torch.Tensor, *, query_heads: int | None = None) -> torch.Tensor:
    """Sum softmaxed attention down its query rows, one score per column."""
    if weights.ndim != 3:
        raise ValueError(
            f"weights must have shape [query heads, rows, keys], got {tuple(weights.shape)}"
        )
    heads, rows, _ = (int(size) for size in weights.shape)
    if heads < 1 or rows < 1:
        raise ValueError("weights must contain at least one query head and one row")
    divisor = heads if query_heads is None else int(query_heads)
    if divisor < heads:
        raise ValueError(f"query_heads {divisor} is smaller than the {heads} heads present")
    dtype = torch.float64 if weights.dtype == torch.float64 else torch.float32
    return weights.sum(dim=(0, 1), dtype=dtype) / float(divisor)


def chunk_accumulated_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    scaling: float,
) -> torch.Tensor:
    """Return one layer's contribution from one chunk of query rows."""
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query and key must both have shape [batch, heads, length, head dim]")
    batch, query_heads, rows, head_dim = (int(size) for size in query.shape)
    if batch != 1 or int(key.shape[0]) != 1:
        raise ValueError("the H2O capture scores one sequence at a time")
    if int(key.shape[3]) != head_dim:
        raise ValueError("query and key head dimensions do not align")
    kv_heads, key_length = int(key.shape[1]), int(key.shape[2])
    if kv_heads < 1 or query_heads % kv_heads:
        raise ValueError(f"{query_heads} query heads do not divide across {kv_heads} KV heads")
    memory_length = key_length - rows
    if memory_length < 0:
        raise ValueError(f"{rows} query rows exceed the {key_length} keys they attend to")
    groups = query_heads // kv_heads
    dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    bias = kernels.causal_bias(rows, memory_length, dtype, query.device)
    total = torch.zeros(key_length, dtype=dtype, device=query.device)
    for group in range(kv_heads):
        rows_of_group = query[:, group * groups : (group + 1) * groups]
        logits = torch.matmul(rows_of_group, key[:, group : group + 1].transpose(-2, -1))
        logits = logits.to(dtype) * scaling + bias
        total += h2o_row_scores(torch.softmax(logits, dim=-1)[0], query_heads=query_heads)
    return total


@dataclass
class _H2OScoreContext:
    """The scores one H2O capture accumulates."""

    context_length: int
    scores: torch.Tensor


_ctx: _H2OScoreContext | None = None
_registered = False


def _score_capture_attention(
    module: object,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: object,
) -> tuple[torch.Tensor, None]:
    """Return the normal SDPA output and accumulate this chunk's attention mass."""
    del attention_mask, dropout, kwargs
    repeats = int(getattr(module, "num_key_value_groups", 1))
    keys_full = kernels.repeat_kv(key, repeats)
    values_full = kernels.repeat_kv(value, repeats)
    query_length = int(query.shape[2])
    memory_length = int(key.shape[2]) - query_length
    output_mask = kernels.lower_right_causal_bias(query_length, int(keys_full.shape[2]))
    if output_mask is None:
        output_mask = kernels.causal_bias(query_length, memory_length, query.dtype, query.device)
    output = torch.nn.functional.scaled_dot_product_attention(
        query,
        keys_full,
        values_full,
        attn_mask=output_mask,
        dropout_p=0.0,
        scale=scaling,
    )
    output = output.transpose(1, 2).contiguous()

    context = _ctx
    if context is None:
        return output, None
    key_length = int(key.shape[2])
    if key_length > context.context_length:
        raise RuntimeError(
            f"H2O-score chunk reached key length {key_length} past the scored "
            f"context length {context.context_length}"
        )
    context.scores[:key_length] += chunk_accumulated_scores(query, key, scaling=scaling)
    return output, None


def _register_capture() -> None:
    """Register the scoring attention function with Transformers, once."""
    global _registered
    if _registered:
        return
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    register = getattr(ALL_ATTENTION_FUNCTIONS, "register", None)
    if not callable(register):
        raise RuntimeError("Transformers attention registry has no register method")
    register_fn = cast(Callable[[str, Callable[..., object]], object], register)
    register_fn(_CAPTURE_IMPL, _score_capture_attention)
    _registered = True


def _set_attn_impl(model: object, implementation: str) -> None:
    """Set the model attention implementation through the package kernel."""
    kernels.set_attention_implementation(model, implementation)


def memory_h2o_scores(
    model: object,
    past: MutableHFCache,
    context_ids: torch.Tensor,
    tokenizer: object | None = None,
    *,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Return one float32 H2O score per cache position, on the host."""
    global _ctx
    del tokenizer
    kernels.refuse_layer_aware_attention(model, "H2O")
    if context_ids.ndim != 2 or int(context_ids.shape[0]) != 1:
        raise ValueError(f"context_ids must have shape [1, length], got {tuple(context_ids.shape)}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    memory_length = kernels.cache_length(past)
    context_length = int(context_ids.shape[1])
    if context_length > memory_length:
        raise ValueError(f"context length {context_length} exceeds cache length {memory_length}")
    scores = torch.zeros(memory_length, dtype=torch.float32, device=context_ids.device)
    if memory_length == 0 or context_length == 0:
        return scores.cpu()
    from transformers import DynamicCache

    _register_capture()
    backbone = kernels.get_backbone(model)
    config = getattr(model, "config", None)
    previous_impl = str(getattr(config, "_attn_implementation", "sdpa"))
    scoring_cache = DynamicCache()
    try:
        _set_attn_impl(model, _CAPTURE_IMPL)
        _ctx = _H2OScoreContext(context_length=context_length, scores=scores)
        for start in range(0, context_length, chunk_size):
            end = min(start + chunk_size, context_length)
            chunk = context_ids[:, start:end]
            with torch.no_grad():
                backbone(
                    input_ids=chunk,
                    past_key_values=scoring_cache,
                    position_ids=torch.arange(start, end, device=chunk.device).unsqueeze(0),
                    attention_mask=torch.ones(1, end, dtype=torch.long, device=chunk.device),
                    use_cache=True,
                    output_attentions=False,
                    return_dict=True,
                )
    finally:
        _ctx = None
        _set_attn_impl(model, previous_impl)
        del scoring_cache
    return scores.cpu()
