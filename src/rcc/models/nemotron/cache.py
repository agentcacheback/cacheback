"""Native hybrid caches with independent FP32 recurrent state."""

from typing import Any

import torch


def new_cache(config: Any) -> Any:
    """Preserve SSM precision and measure tokens on an actual attention layer."""
    from transformers.cache_utils import DynamicCache, LinearAttentionLayer

    layer_types = tuple(config.layer_types)
    if "full_attention" not in layer_types:
        raise ValueError("Nemotron hybrid cache requires a full-attention layer")
    first_attention = layer_types.index("full_attention")

    class NativeCache(DynamicCache):
        def _attention_index(self, layer_idx: int) -> int:
            if 0 <= layer_idx < len(layer_types) and layer_types[layer_idx] != "full_attention":
                return first_attention
            return layer_idx

        def get_seq_length(self, layer_idx: int = 0) -> int:
            return super().get_seq_length(self._attention_index(layer_idx))

        def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
            return super().get_mask_sizes(query_length, self._attention_index(layer_idx))

    class FP32StateLayer(LinearAttentionLayer):
        def lazy_initialization(
            self, conv_states: Any = None, recurrent_states: Any = None
        ) -> None:
            super().lazy_initialization(conv_states=conv_states)
            if recurrent_states is not None:
                self.recurrent_states = torch.zeros_like(recurrent_states, dtype=torch.float32)
                self.is_recurrent_states_initialized = True

    cache = NativeCache(config=config)
    for index, layer in enumerate(cache.layers):
        if isinstance(layer, LinearAttentionLayer):
            cache.layers[index] = FP32StateLayer()
    return cache
