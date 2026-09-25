"""KVCache: the five-axis cache tensor plus its position map.

The data object every transform takes and returns. It holds plain tensors; the
two converters cross to Hugging Face caches in either layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

import torch

from rcc.rope import RopeParams

if TYPE_CHECKING:
    from rcc.meters import Meters

#: Per-layer keys/values as they cross the Hugging Face boundary: (K, V) with
#: each tensor shaped [batch, kv_heads, length, head_dim].
KVList = list[tuple[torch.Tensor, torch.Tensor]]


def _kv_to_list(past: Any) -> KVList:
    """Return per-layer (K, V) from any Hugging Face cache, or [] for None."""
    if past is None:
        return []
    layers = getattr(past, "layers", None)
    if layers is not None:
        return [(lyr.keys, lyr.values) for lyr in layers]
    to_legacy = getattr(past, "to_legacy_cache", None)
    if to_legacy is not None:
        return [(k, v) for (k, v) in to_legacy()]
    return [(k, v) for (k, v) in past]


def _list_to_kv(kv: KVList, like: Any = None) -> Any:
    """Rebuild a Hugging Face cache from per-layer (K, V), `like` giving the target class."""
    if like is not None:
        like_cls = cast(Any, type(like))
        if hasattr(like, "from_legacy_cache") and not hasattr(like, "layers"):
            return like_cls.from_legacy_cache(tuple(kv))
        if hasattr(like, "layers") and hasattr(like, "update"):
            cache: Any = like_cls()
            for i, (k, v) in enumerate(kv):
                cache.update(k, v, i)
            return cache
        return tuple(kv)
    try:
        from transformers import DynamicCache

        cache = DynamicCache()
        for i, (k, v) in enumerate(kv):
            cache.update(k, v, i)
        return cache
    except Exception:
        return tuple(kv)


class Axis(Enum):
    """The five axes of a KV cache; transforms on different axes compose."""

    LAYERS = "layers"
    KV_HEADS = "kv_heads"
    LENGTH = "length"
    HEAD_DIM = "head_dim"
    BITS = "bits"


@dataclass(frozen=True)
class KVCache:
    """A five-axis KV cache: keys/values of shape [layers, kv_heads, length, head_dim].

    Tensors always sit physically rotated at their slot index, slot i at position i.
    `positions` is provenance; `rope` holds the prefill rotary params, None if synthetic.
    """

    keys: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor
    rope: RopeParams | None = None

    def __post_init__(self) -> None:
        """Check that keys and values agree, positions covers length, and rope fits head_dim."""
        if self.keys.ndim != 4:
            raise ValueError(
                f"keys must be [layers, kv_heads, length, head_dim], got {self.keys.shape}"
            )
        if self.keys.shape != self.values.shape:
            raise ValueError(f"keys {self.keys.shape} and values {self.values.shape} disagree")
        if self.keys.dtype != self.values.dtype:
            raise ValueError(f"keys {self.keys.dtype} and values {self.values.dtype} disagree")
        if self.positions.ndim != 1 or self.positions.shape[0] != self.keys.shape[2]:
            raise ValueError(
                f"positions must be [length={self.keys.shape[2]}], got {self.positions.shape}"
            )
        if self.rope is not None:
            half = self.keys.shape[3] // 2
            if self.rope.inv_freq.ndim != 1 or self.rope.inv_freq.shape[0] != half:
                raise ValueError(
                    f"rope.inv_freq must be [head_dim/2={half}], got {self.rope.inv_freq.shape}"
                )

    @property
    def layers(self) -> int:
        """Return the layer count."""
        return int(self.keys.shape[0])

    @property
    def kv_heads(self) -> int:
        """Return the KV heads per layer."""
        return int(self.keys.shape[1])

    @property
    def length(self) -> int:
        """Return the number of kept token slots."""
        return int(self.keys.shape[2])

    @property
    def head_dim(self) -> int:
        """Return the per-head hidden dimension."""
        return int(self.keys.shape[3])

    @property
    def bits(self) -> int:
        """Return the bits per stored element, from the tensor dtype."""
        return self.keys.element_size() * 8

    def axis_size(self, axis: Axis) -> int:
        """Return the size of one axis."""
        return {
            Axis.LAYERS: self.layers,
            Axis.KV_HEADS: self.kv_heads,
            Axis.LENGTH: self.length,
            Axis.HEAD_DIM: self.head_dim,
            Axis.BITS: self.bits,
        }[axis]

    def meters(self) -> Meters:
        """Return the analytic byte counts for this cache: computed sizes, not measured RSS."""
        from rcc.meters import measure

        return measure(self)

    @classmethod
    def from_prefill(cls, model: Any, input_ids: torch.Tensor) -> KVCache:
        """Prefill a single sequence and capture its KV cache plus rotary params."""
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"from_prefill is single-sequence, got shape {tuple(input_ids.shape)}")
        length = int(input_ids.shape[1])
        position_ids = torch.arange(length, device=input_ids.device).unsqueeze(0)
        attention_mask = torch.ones(1, length, dtype=torch.long, device=input_ids.device)
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )
        kv = _kv_to_list(out.past_key_values)
        keys = torch.stack([k.squeeze(0) for (k, _) in kv], dim=0)
        values = torch.stack([v.squeeze(0) for (_, v) in kv], dim=0)
        rotary = model.model.rotary_emb
        rope = RopeParams(
            inv_freq=rotary.inv_freq.detach().clone(),
            attention_scaling=float(getattr(rotary, "attention_scaling", 1.0)),
        )
        return cls(keys=keys, values=values, positions=position_ids.squeeze(0), rope=rope)

    def to_hf_cache(self, like: Any = None) -> Any:
        """Rebuild a Hugging Face cache from this KVCache, re-adding the batch dim."""
        kv: KVList = [
            (self.keys[i].unsqueeze(0), self.values[i].unsqueeze(0)) for i in range(self.layers)
        ]
        return _list_to_kv(kv, like=like)
