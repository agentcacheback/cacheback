"""Cache, attention, and question-location helpers shared across the Select code.

The cache readers cover every supported Hugging Face cache, the bias builders
state the mask explicitly, and the attention helpers move a model to eager.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from importlib import import_module
from typing import Protocol, cast

import torch
import torch.nn.functional as functional

from rcc.transforms.select.core.types import (
    DecoderBackbone,
    KVPair,
    MutableHFCache,
    TokenIds,
)


class _CacheLayer(Protocol):
    """Typed view of one layered Hugging Face cache entry."""

    keys: torch.Tensor
    values: torch.Tensor


class _ModelWithConfig(Protocol):
    """Typed view of a model carrying an attention configuration."""

    config: object


__all__ = [
    "DecoderBackbone",
    "KVPair",
    "MutableHFCache",
    "TokenIds",
    "cache_kv",
    "cache_length",
    "cache_logical_length",
    "causal_bias",
    "get_backbone",
    "has_layer_aware_attention",
    "locate_question",
    "lower_right_causal_bias",
    "pool_scores",
    "refuse_layer_aware_attention",
    "repeat_kv",
    "restore_attention_implementations",
    "set_attention_implementation",
    "snapshot_attention_implementations",
    "use_eager_attention",
]


def cache_kv(past: object | None) -> list[KVPair]:
    """Return per-layer key/value tensors from any supported HF cache flavor."""
    if past is None:
        return []
    layers = getattr(past, "layers", None)
    if layers is not None:
        layer_values = cast(Sequence[_CacheLayer], layers)
        return [(layer.keys, layer.values) for layer in layer_values]
    to_legacy = getattr(past, "to_legacy_cache", None)
    if to_legacy is not None:
        legacy = cast(Callable[[], object], to_legacy)()
        return [
            (key, value) for key, value in cast(Sequence[tuple[torch.Tensor, torch.Tensor]], legacy)
        ]
    return [(key, value) for key, value in cast(Sequence[tuple[torch.Tensor, torch.Tensor]], past)]


def cache_length(past: object | None) -> int:
    """Return the length axis of an HF cache, or zero when it is empty."""
    kv = cache_kv(past)
    if not kv:
        return 0
    return int(kv[0][0].shape[2])


def cache_logical_length(past: object | None) -> int:
    """Return cumulative memory length without collapsing sliding cache layers."""
    if past is None:
        return 0
    getter = getattr(past, "get_seq_length", None)
    if callable(getter):
        try:
            value = getter()
        except (TypeError, ValueError, AttributeError):
            value = None
        if isinstance(value, torch.Tensor):
            return int(value.item())
        if isinstance(value, int):
            return value
    return cache_length(past)


def get_backbone(model: object) -> DecoderBackbone:
    """Return a model's decoder backbone, falling back to its `model` attribute."""
    get_decoder = getattr(model, "get_decoder", None)
    if callable(get_decoder):
        return cast(DecoderBackbone, get_decoder())
    base_model = getattr(model, "model", model)
    return cast(DecoderBackbone, base_model)


def has_layer_aware_attention(model: object) -> bool:
    """Return whether a model needs per-layer masks or cache handling."""
    config = getattr(get_backbone(model), "config", getattr(model, "config", None))
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        return False
    return any(
        "sliding" in str(layer_type) or "chunked" in str(layer_type) for layer_type in layer_types
    )


def refuse_layer_aware_attention(model: object, scorer: str) -> None:
    """Raise when a scorer's mask logic cannot represent this model's hybrid attention."""
    if has_layer_aware_attention(model):
        raise RuntimeError(f"{scorer} scorer refuses hybrid attention until its mask is reworked")


def repeat_kv(value: torch.Tensor, repeats: int) -> torch.Tensor:
    """Expand grouped-query KV heads to the query-head count."""
    if repeats == 1:
        return value
    batch, heads, sequence, dimension = value.shape
    return (
        value[:, :, None, :, :]
        .expand(batch, heads, repeats, sequence, dimension)
        .reshape(batch, heads * repeats, sequence, dimension)
    )


def causal_bias(
    query_length: int,
    memory_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Additive bottom-right causal bias over `[memory | current]` keys.

    Shape `[1, 1, q_len, mem_len + q_len]`: memory columns are never masked, current
    column `j` is visible to row `i` only when `j <= i`, bottom-right on non-square.
    """
    key_length = memory_length + query_length
    bias = torch.zeros(query_length, key_length, dtype=dtype, device=device)
    if query_length > 0:
        query_positions = torch.arange(query_length, device=device).unsqueeze(1)
        key_positions = torch.arange(query_length, device=device).unsqueeze(0)
        bias[:, memory_length:] = torch.where(
            key_positions > query_positions,
            torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=device),
            torch.zeros((), dtype=dtype, device=device),
        )
    return bias.view(1, 1, query_length, key_length)


def lower_right_causal_bias(
    query_length: int,
    key_length: int,
) -> torch.Tensor | None:
    """Return Torch's optional lower-right causal bias, or ``None`` if unavailable."""
    try:
        bias_module = import_module("torch.nn.attention.bias")
        lower_right = cast(
            Callable[[int, int], torch.Tensor], bias_module.__dict__["causal_lower_right"]
        )
    except Exception:  # pragma: no cover - depends on the installed torch
        return None
    return lower_right(query_length, key_length)


def locate_question(
    judger_ids: torch.Tensor,
    question_ids: TokenIds,
) -> tuple[int, int, bool]:
    """Locate the question rows inside a judger prompt.

    Returns `(start, end, exact)`; when the ids are not an exact subsequence the span
    falls back to the last `len(question_ids)` rows and `exact` is False.
    """

    def token_list(values: torch.Tensor) -> list[int]:
        tolist = cast(Callable[[], list[int]], values.tolist)  # pyright: ignore[reportUnknownMemberType]
        return tolist()

    haystack = token_list(judger_ids[0])
    if isinstance(question_ids, torch.Tensor):
        needle = token_list(question_ids.reshape(-1))
    else:
        needle = list(question_ids)
    question_length = len(haystack)
    if needle and len(needle) <= question_length:
        for start in range(question_length - len(needle) + 1):
            if haystack[start : start + len(needle)] == needle:
                return start, start + len(needle), True
    span = min(len(needle) or question_length, question_length)
    return question_length - span, question_length, False


def pool_scores(scores: torch.Tensor, kernel: int) -> torch.Tensor:
    """Max-pool the score vector in 1-D, preserving its length.

    Max pooling, not average: it pulls the contiguous neighbours of a high-vote
    token up so that kept regions stay clustered.
    """
    if kernel <= 1:
        return scores
    pooled = functional.max_pool1d(
        scores.view(1, 1, -1), kernel_size=kernel, stride=1, padding=kernel // 2
    )
    return pooled.view(-1)[: scores.shape[0]]


def set_attention_implementation(model: object, implementation: str) -> None:
    """Set attention implementation through the model setter or config fallback."""
    setter = getattr(model, "set_attn_implementation", None)
    if setter is not None:
        try:
            cast(Callable[[str], object], setter)(implementation)
            return
        except Exception:
            pass
    config = cast(_ModelWithConfig, model).config
    setattr(config, "_attn_implementation", implementation)  # noqa: B010


def snapshot_attention_implementations(
    *models: object,
) -> tuple[tuple[object, object], ...]:
    """Snapshot every model and sub-config implementation by object identity."""
    configs: list[object] = []
    seen: set[int] = set()

    def add_config(config: object | None) -> None:
        if config is not None and id(config) not in seen:
            seen.add(id(config))
            configs.append(config)
            for name in getattr(config, "sub_configs", ()):
                subconfig = getattr(config, name, None)
                if subconfig is not None:
                    add_config(subconfig)

    for model in models:
        add_config(getattr(model, "config", None))
        modules = getattr(model, "modules", None)
        if callable(modules):
            for module in cast(Callable[[], Iterable[object]], modules)():
                add_config(getattr(module, "config", None))
    return tuple((config, getattr(config, "_attn_implementation", None)) for config in configs)


def restore_attention_implementations(
    snapshot: tuple[tuple[object, object], ...],
) -> None:
    """Restore every snapshotted implementation without recursive clobbering."""
    for config, implementation in snapshot:
        values = vars(config)
        if hasattr(config, "_attn_implementation_internal"):
            values["_attn_implementation_internal"] = implementation
        else:
            values["_attn_implementation"] = implementation
        values.pop("_attn_was_changed", None)


def use_eager_attention(model: object) -> str:
    """Switch a model to eager attention and return its prior implementation."""
    config = cast(_ModelWithConfig, model).config
    previous = str(getattr(config, "_attn_implementation", "eager"))
    set_attention_implementation(model, "eager")
    return previous
