"""ChunkKV scoring, aligned to cache positions.

The attention each cached position receives over the observation window is
averaged within fixed chunks and broadcast back to the chunk's members.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch

from rcc.transforms.select.baselines.chunkkv_native import (
    native_window_scores,
    window_stream_scores,
)
from rcc.transforms.select.baselines.kvzip import project_position_scores
from rcc.transforms.select.core import kernels
from rcc.transforms.select.core.types import KVPair, MutableHFCache

__all__ = [
    "chunk_means",
    "chunk_position_scores",
    "memory_chunkkv_scores",
    "native_window_scores",
    "project_position_scores",
    "window_stream_scores",
]

_CAPTURE_IMPL = "rcc_chunkkv_score_capture"


def chunk_means(scores: torch.Tensor, *, chunk_size: int) -> torch.Tensor:
    """Return the mean score of every full chunk, plus the short tail chunk."""
    if scores.ndim != 1:
        raise ValueError(f"scores must have shape [positions], got {tuple(scores.shape)}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    length = int(scores.shape[0])
    values = scores.to(torch.float32)
    complete = length // chunk_size
    remainder = length % chunk_size
    parts: list[torch.Tensor] = []
    if complete:
        parts.append(values[: complete * chunk_size].view(complete, chunk_size).mean(dim=-1))
    if remainder:
        parts.append(values[-remainder:].mean().view(1))
    if not parts:
        return torch.zeros(0, dtype=torch.float32, device=scores.device)
    return torch.cat(parts)


def chunk_position_scores(scores: torch.Tensor, *, chunk_size: int) -> torch.Tensor:
    """Broadcast each chunk score back to the positions in that chunk."""
    means = chunk_means(scores, chunk_size=chunk_size)
    length = int(scores.shape[0])
    remainder = length % chunk_size
    sizes = [chunk_size] * (length // chunk_size) + ([remainder] if remainder else [])
    counts = torch.tensor(sizes, dtype=torch.long, device=means.device)
    return torch.repeat_interleave(means, counts)


@dataclass
class _ChunkKVScoreContext:
    """The state the ChunkKV attention function accumulates."""

    prefix_length: int
    window_size: int
    kernel_size: int
    column_chunk: int
    score_sum: torch.Tensor
    layer_count: torch.Tensor


_ctx: _ChunkKVScoreContext | None = None
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
    """Return the normal SDPA output and accumulate the observation-window scores."""
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
        query, keys_full, values_full, attn_mask=output_mask, dropout_p=0.0, scale=scaling
    ).transpose(1, 2)
    output = output.contiguous()

    context = _ctx
    if context is None:
        return output, None
    if memory_length != context.prefix_length:
        raise RuntimeError(
            f"chunkkv cache length drifted from {context.prefix_length} to {memory_length}"
        )
    native = native_window_scores(
        query,
        key,
        scaling=scaling,
        window_size=context.window_size,
        kernel_size=context.kernel_size,
        column_chunk=context.column_chunk,
    )
    context.score_sum += project_position_scores(native)
    context.layer_count += 1
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


def _restore_cache(
    past: MutableHFCache,
    tail: list[KVPair],
    *,
    memory_length: int,
    prefix_length: int,
) -> None:
    """Crop the scoring rows, re-append the saved tail, and check the cache is unchanged."""
    if kernels.cache_length(past) != prefix_length:
        past.crop(prefix_length)
    for layer_index, (tail_keys, tail_values) in enumerate(tail):
        past.update(tail_keys, tail_values, layer_index)
    if kernels.cache_length(past) != memory_length:
        raise RuntimeError(
            f"chunkkv restored cache length {kernels.cache_length(past)}, expected {memory_length}"
        )
    for restored, expected in zip(kernels.cache_kv(past), tail, strict=True):
        restored_keys, restored_values = restored
        expected_keys, expected_values = expected
        if not torch.equal(restored_keys[:, :, prefix_length:], expected_keys):
            raise RuntimeError("chunkkv changed a protected key tail")
        if not torch.equal(restored_values[:, :, prefix_length:], expected_values):
            raise RuntimeError("chunkkv changed a protected value tail")


def memory_chunkkv_scores(
    model: object,
    past: MutableHFCache,
    context_ids: torch.Tensor,
    tokenizer: object,
    *,
    chunk_size: int = 16,
    window_size: int = 32,
    kernel_size: int = 5,
    column_chunk: int = 4096,
) -> torch.Tensor:
    """Return one float32 ChunkKV score per cache position, on the host."""
    global _ctx
    del tokenizer
    kernels.refuse_layer_aware_attention(model, "ChunkKV")
    memory_length = kernels.cache_length(past)
    context_length = int(context_ids.shape[1])
    if context_ids.ndim != 2 or int(context_ids.shape[0]) != 1:
        raise ValueError(f"context_ids must have shape [1, length], got {tuple(context_ids.shape)}")
    if context_length > memory_length:
        raise ValueError(f"context length {context_length} exceeds cache length {memory_length}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if memory_length == 0:
        return torch.zeros(0, dtype=torch.float32)
    prefix_length = context_length - window_size
    if window_size < 1 or prefix_length < 1:
        raise ValueError(
            f"observation window {window_size} leaves no scored prompt in {context_length} tokens"
        )
    _register_capture()
    score_sum = torch.zeros(context_length, dtype=torch.float32, device=context_ids.device)
    layer_count = torch.zeros((), dtype=torch.int32, device=context_ids.device)
    backbone = kernels.get_backbone(model)
    config = getattr(model, "config", None)
    previous_impl = str(getattr(config, "_attn_implementation", "sdpa"))
    body_error: BaseException | None = None
    tail = [
        (keys[:, :, prefix_length:].clone(), values[:, :, prefix_length:].clone())
        for keys, values in kernels.cache_kv(past)
    ]
    try:
        past.crop(prefix_length)
        _set_attn_impl(model, _CAPTURE_IMPL)
        _ctx = _ChunkKVScoreContext(
            prefix_length=prefix_length,
            window_size=window_size,
            kernel_size=kernel_size,
            column_chunk=column_chunk,
            score_sum=score_sum,
            layer_count=layer_count,
        )
        with torch.no_grad():
            backbone(
                input_ids=context_ids[:, prefix_length:context_length],
                past_key_values=past,
                position_ids=torch.arange(
                    prefix_length, context_length, device=context_ids.device
                ).unsqueeze(0),
                attention_mask=torch.ones(
                    1, context_length, dtype=torch.long, device=context_ids.device
                ),
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
    except BaseException as error:
        body_error = error
        raise
    finally:
        _ctx = None
        _set_attn_impl(model, previous_impl)
        try:
            _restore_cache(past, tail, memory_length=memory_length, prefix_length=prefix_length)
        except RuntimeError as restore_error:
            if body_error is None:
                raise
            raise restore_error from body_error
    layers = int(layer_count)
    if layers < 1:
        raise RuntimeError("chunkkv captured no layer")
    prompt = chunk_position_scores(score_sum / float(layers), chunk_size=chunk_size)
    scores = torch.zeros(memory_length, dtype=torch.float32, device=prompt.device)
    scores[:context_length] = prompt
    return scores.cpu()
