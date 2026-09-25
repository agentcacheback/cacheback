"""KVzip reconstruction scoring, aligned to cache positions.

The model is prompted to reconstruct its own context, and the attention that
reconstruction pays to each cached position becomes that position's score.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, cast

import torch

from rcc.transforms.select.core import kernels
from rcc.transforms.select.core.types import KVPair, MutableHFCache

__all__ = [
    "memory_kvzip_scores",
    "native_reconstruction_scores",
    "project_position_scores",
    "reconstruction_inputs",
    "reconstruction_stream_scores",
]

_CAPTURE_IMPL = "rcc_kvzip_score_capture"
_FIRST_PROMPT = "\n\nRepeat the previous context exactly."
_LATER_PROMPT = "\n\nRepeat the part of the previous context exactly, starting with "
_QWEN3_POSTFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class _Tokenizer(Protocol):
    """The tokenizer calls the reconstruction prompt makes."""

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        add_special_tokens: bool,
    ) -> Mapping[str, torch.Tensor]:
        """Encode one prompt fragment without adding special tokens."""
        ...


def project_position_scores(scores: torch.Tensor) -> torch.Tensor:
    """Average a ``[layers, KV heads, positions]`` score down to one row per position."""
    if scores.ndim != 3:
        raise ValueError(
            "native scores must have shape [layers, KV heads, positions], "
            f"got {tuple(scores.shape)}"
        )
    if int(scores.shape[0]) < 1 or int(scores.shape[1]) < 1:
        raise ValueError("native scores must contain at least one layer and KV head")
    scores = scores.to(torch.float32)
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("native scores must be finite")
    return scores.mean(dim=(0, 1))


def reconstruction_stream_scores(
    weights: torch.Tensor,
    *,
    source_length: int,
    kv_groups: int,
) -> torch.Tensor:
    """Reduce one layer's reconstruction attention to one score per KV head."""
    if weights.ndim != 3:
        raise ValueError(
            "weights must have shape [query heads, reconstruction rows, keys], "
            f"got {tuple(weights.shape)}"
        )
    query_heads, rows, keys = (int(size) for size in weights.shape)
    if rows < 1:
        raise ValueError("weights must contain at least one reconstruction row")
    if source_length < 0 or source_length > keys:
        raise ValueError(f"source length {source_length} is outside the key length {keys}")
    if kv_groups < 1 or query_heads % kv_groups:
        raise ValueError(f"{query_heads} query heads do not divide into groups of {kv_groups}")
    kv_heads = query_heads // kv_groups
    if source_length == 0:
        return torch.zeros(kv_heads, 0, dtype=torch.float32, device=weights.device)
    source = weights[:, :, :source_length].to(torch.float32)
    grouped = source.reshape(kv_heads, kv_groups, rows, source_length)
    return grouped.amax(dim=(1, 2))


def _encode(tokenizer: _Tokenizer, text: str, device: torch.device) -> torch.Tensor:
    """Encode one reconstruction prompt fragment on the context device."""
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = encoded["input_ids"]
    if ids.ndim != 2 or int(ids.shape[0]) != 1:
        raise ValueError(f"tokenizer returned invalid input_ids shape {tuple(ids.shape)}")
    return ids.to(device)


def reconstruction_inputs(
    context_ids: torch.Tensor,
    tokenizer: _Tokenizer,
    *,
    chunk_size: int = 2000,
    previous_suffix_size: int = 8,
    sink_size: int = 1,
) -> tuple[tuple[int, int, torch.Tensor], ...]:
    """Build the self-reconstruction prompt chunks for a Qwen3 context."""
    if context_ids.ndim != 2 or int(context_ids.shape[0]) != 1:
        raise ValueError(f"context_ids must have shape [1, length], got {tuple(context_ids.shape)}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if previous_suffix_size < 0:
        raise ValueError(f"previous_suffix_size must be nonnegative, got {previous_suffix_size}")
    if sink_size < 0 or sink_size > int(context_ids.shape[1]):
        raise ValueError(f"sink_size {sink_size} is outside the context")
    length = int(context_ids.shape[1])
    postfix = _encode(tokenizer, _QWEN3_POSTFIX, context_ids.device)
    chunks: list[tuple[int, int, torch.Tensor]] = []
    for start in range(sink_size, length, chunk_size):
        end = min(start + chunk_size, length)
        source = context_ids[:, start:end]
        if start == sink_size:
            prefix = _encode(tokenizer, _FIRST_PROMPT, context_ids.device)
        else:
            prefix = _encode(tokenizer, _LATER_PROMPT, context_ids.device)
            previous = context_ids[:, max(0, start - previous_suffix_size) : start]
            prefix = torch.cat((prefix, previous), dim=1)
        repeat = torch.cat((prefix, postfix, source), dim=1)
        chunks.append((start, end, repeat))
    return tuple(chunks)


@dataclass
class _KVzipScoreContext:
    """The reconstruction scores one attention function accumulates."""

    source_start: int
    source_end: int
    memory_length: int
    sink_size: int
    score_sum: torch.Tensor
    score_count: torch.Tensor


_ctx: _KVzipScoreContext | None = None
_registered = False


def native_reconstruction_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    source_start: int,
    source_end: int,
    scaling: float,
    sink_size: int = 0,
) -> torch.Tensor:
    """Return KVzip's native score for one layer as ``[batch, KV heads, source]``."""
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query and key must both have shape [batch, heads, length, head_dim]")
    batch, query_heads, query_length, head_dim = (int(size) for size in query.shape)
    if int(key.shape[0]) != batch or int(key.shape[3]) != head_dim:
        raise ValueError("query and key batch/head dimensions do not align")
    kv_heads = int(key.shape[1])
    if query_heads % kv_heads:
        raise ValueError(f"{query_heads} query heads do not divide across {kv_heads} KV heads")
    memory_length = int(key.shape[2]) - query_length
    if not 0 <= sink_size <= source_start < source_end <= memory_length:
        raise ValueError(
            f"source range [{source_start}, {source_end}) is outside memory length {memory_length}"
        )
    sink = key[:, :, :sink_size, :]
    target = key[:, :, source_start:source_end, :]
    current = key[:, :, -query_length:, :]
    selected = torch.cat((sink, target, current), dim=2)
    grouped_query = query.reshape(batch, kv_heads, query_heads // kv_heads, query_length, head_dim)
    logits = torch.matmul(grouped_query, selected.unsqueeze(2).transpose(-2, -1)) * scaling
    causal = torch.full(
        (query_length, query_length),
        torch.finfo(logits.dtype).min,
        dtype=logits.dtype,
        device=logits.device,
    )
    causal.masked_fill_(
        torch.arange(query_length, device=logits.device).unsqueeze(0)
        <= torch.arange(query_length, device=logits.device).unsqueeze(1),
        0,
    )
    logits[..., -query_length:] += causal.view(1, 1, 1, query_length, query_length)
    weights = torch.softmax(logits, dim=-1)
    target_length = source_end - source_start
    return weights[..., sink_size : sink_size + target_length].amax(dim=(-3, -2))


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
    """Return the normal SDPA output and accumulate the reconstruction scores."""
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
    if memory_length != context.memory_length:
        raise RuntimeError(
            f"KVzip-score cache length drifted from {context.memory_length} to {memory_length}"
        )
    native = native_reconstruction_scores(
        query,
        key,
        source_start=context.source_start,
        source_end=context.source_end,
        scaling=scaling,
        sink_size=context.sink_size,
    ).to(torch.float32)
    position_scores = native.mean(dim=(0, 1))
    context.score_sum[context.source_start : context.source_end] += position_scores
    context.score_count[context.source_start : context.source_end] += 1
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
    context_length: int,
) -> None:
    """Crop the scoring rows, re-append the latent tail, and check the cache is unchanged."""
    if kernels.cache_length(past) != context_length:
        past.crop(context_length)
    if memory_length == context_length:
        return
    for layer_index, (tail_keys, tail_values) in enumerate(tail):
        past.update(tail_keys, tail_values, layer_index)
    if kernels.cache_length(past) != memory_length:
        raise RuntimeError(
            f"KVzip-score restored cache length {kernels.cache_length(past)}, "
            f"expected {memory_length}"
        )
    for restored, expected in zip(kernels.cache_kv(past), tail, strict=True):
        restored_keys, restored_values = restored
        expected_keys, expected_values = expected
        if not torch.equal(restored_keys[:, :, context_length:], expected_keys):
            raise RuntimeError("KVzip-score changed a protected latent key tail")
        if not torch.equal(restored_values[:, :, context_length:], expected_values):
            raise RuntimeError("KVzip-score changed a protected latent value tail")


def memory_kvzip_scores(
    model: object,
    past: MutableHFCache,
    context_ids: torch.Tensor,
    tokenizer: _Tokenizer,
    *,
    chunk_size: int = 2000,
    previous_suffix_size: int = 8,
    sink_size: int = 1,
) -> torch.Tensor:
    """Return one float32 reconstruction score per cache position, on the host."""
    global _ctx
    config = getattr(model, "config", None)
    kernels.refuse_layer_aware_attention(model, "KVzip")
    if str(getattr(config, "model_type", "")) != "qwen3":
        raise ValueError("KVzip-score currently freezes the Qwen3 reconstruction template")
    memory_length = kernels.cache_length(past)
    context_length = int(context_ids.shape[1])
    if context_length > memory_length:
        raise ValueError(f"context length {context_length} exceeds cache length {memory_length}")
    if sink_size < 0 or sink_size > context_length:
        raise ValueError(f"sink_size {sink_size} is outside context length {context_length}")
    if memory_length == 0:
        return torch.zeros(0, dtype=torch.float32)
    _register_capture()
    score_sum = torch.zeros(memory_length, dtype=torch.float32, device=context_ids.device)
    score_count = torch.zeros(memory_length, dtype=torch.int32, device=context_ids.device)
    backbone = kernels.get_backbone(model)
    previous_impl = str(getattr(config, "_attn_implementation", "sdpa"))
    body_error: BaseException | None = None
    tail = [
        (
            keys[:, :, context_length:].clone(),
            values[:, :, context_length:].clone(),
        )
        for keys, values in kernels.cache_kv(past)
    ]
    try:
        if memory_length != context_length:
            past.crop(context_length)
        _set_attn_impl(model, _CAPTURE_IMPL)
        for start, end, repeat_ids in reconstruction_inputs(
            context_ids,
            tokenizer,
            chunk_size=chunk_size,
            previous_suffix_size=previous_suffix_size,
            sink_size=sink_size,
        ):
            _ctx = _KVzipScoreContext(
                source_start=start,
                source_end=end,
                memory_length=context_length,
                sink_size=sink_size,
                score_sum=score_sum,
                score_count=score_count,
            )
            query_length = int(repeat_ids.shape[1])
            base = torch.ones(1, context_length, dtype=torch.long, device=repeat_ids.device)
            current = torch.ones_like(repeat_ids)
            with torch.no_grad():
                backbone(
                    input_ids=repeat_ids,
                    past_key_values=past,
                    position_ids=torch.arange(
                        context_length,
                        context_length + query_length,
                        device=repeat_ids.device,
                    ).unsqueeze(0),
                    attention_mask=torch.cat((base, current), dim=1),
                    use_cache=True,
                    output_attentions=False,
                    return_dict=True,
                )
            if kernels.cache_length(past) != context_length:
                past.crop(context_length)
    except BaseException as error:
        body_error = error
        raise
    finally:
        _ctx = None
        _set_attn_impl(model, previous_impl)
        try:
            _restore_cache(past, tail, memory_length=memory_length, context_length=context_length)
        except RuntimeError as restore_error:
            if body_error is None:
                raise
            raise restore_error from body_error
    scores = torch.where(
        score_count > 0,
        score_sum / score_count.clamp_min(1).to(torch.float32),
        torch.zeros_like(score_sum),
    )
    return scores.cpu()
