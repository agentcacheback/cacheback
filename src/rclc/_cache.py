"""Cache copying and decoder access shared by selection, reasoning and handoff."""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast

import torch

# Attention dispatch is process-global; use separate processes for concurrent forwards.
ATTENTION_LOCK = threading.Lock()

KVPair = tuple[torch.Tensor, torch.Tensor]


class _CacheLayer(Protocol):
    """Typed view of one layered Hugging Face cache entry."""

    keys: torch.Tensor
    values: torch.Tensor


def _clone_cache_value(value: object) -> object:
    """Clone cache state without requiring autograd tensors to be leaves."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        return {key: _clone_cache_value(item) for key, item in mapping.items()}
    if isinstance(value, list):
        items = cast(list[object], value)
        return [_clone_cache_value(item) for item in items]
    if isinstance(value, tuple):
        items = cast(tuple[object, ...], value)
        return tuple(_clone_cache_value(item) for item in items)
    if hasattr(value, "__dict__"):
        cloned = copy.copy(value)
        for name, item in vars(value).items():
            setattr(cloned, name, _clone_cache_value(item))
        return cloned
    return value


def clone_cache(past: object) -> object:
    """Return a disposable cache copy for a model forward."""
    try:
        return copy.deepcopy(past)
    except (RuntimeError, TypeError):
        return _clone_cache_value(past)


def cache_kv(past: object | None) -> list[KVPair]:
    """Return per-layer key/value tensors from any supported HF cache flavor."""
    if past is None:
        return []
    layers = getattr(past, "layers", None)
    if layers is not None:
        return [(layer.keys, layer.values) for layer in cast(Sequence[_CacheLayer], layers)]
    to_legacy = getattr(past, "to_legacy_cache", None)
    if to_legacy is not None:
        past = cast(Callable[[], object], to_legacy)()
    return list(cast(Sequence[KVPair], past))


def get_backbone(model: object) -> Any:
    """Return a model's decoder backbone, falling back to its `model` attribute."""
    get_decoder = getattr(model, "get_decoder", None)
    if callable(get_decoder):
        return get_decoder()
    return getattr(model, "model", model)
