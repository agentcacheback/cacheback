"""Gemma 4 rolling-cache construction shared by resident capture paths."""

from __future__ import annotations

from typing import Any

from transformers.cache_utils import Cache, DynamicCache, DynamicLayer, DynamicSlidingWindowLayer


class RollingHybridCache(DynamicCache):
    """A DynamicCache whose layers are built explicitly from layer types.

    ``DynamicCache(config=...)`` slices ``layer_types[:-num_kv_shared_layers]``, empty at the
    pinned zero on transformers 5.13.0, so it falls back to full retention; built here instead.
    """

    def __init__(self, decoder_config: Any) -> None:
        """Build full and sliding layer caches from the decoder schedule."""
        layer_types = list(decoder_config.layer_types)
        shared = int(getattr(decoder_config, "num_kv_shared_layers", 0) or 0)
        if shared:
            layer_types = layer_types[:-shared]
        layers: list[Any] = [
            DynamicSlidingWindowLayer(decoder_config)
            if layer_type == "sliding_attention"
            else DynamicLayer(decoder_config)
            for layer_type in layer_types
        ]
        Cache.__init__(self, layers=layers)


__all__ = ("RollingHybridCache",)
