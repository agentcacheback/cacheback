"""Hugging Face cache geometry used by latent rollout and composition.

Cuts and concatenation relocate the rotary keys and gather values unchanged;
split and pad preserve the positions rollout already wrote.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch

from rcc import RopeParams
from rcc.rope import apply_rope_k, rope_cos_sin, unapply_rope_k
from rcc.transforms.select.core.kernels import (
    KVPair as KVPair,
)
from rcc.transforms.select.core.kernels import (
    cache_kv as cache_kv,
)
from rcc.transforms.select.core.kernels import (
    cache_length as cache_length,
)
from rcc.transforms.select.core.kernels import (
    cache_logical_length as cache_logical_length,
)
from rcc.transforms.select.core.types import MutableHFCache


@dataclass
class BatchedPast:
    """A batched cache with interspersed padding and per-item live lengths.

    ``mask`` is one exactly at live columns. ``live`` records each row's batch-one
    cache length, so the next segment resumes at the same logical position.
    """

    past: Any
    mask: torch.Tensor
    live: list[int]


def compress_past(past: Any, keep: list[int], rope: RopeParams | None = None) -> Any:
    """Rebuild a cache from retained slots, preserving the given keep order.

    Without ``rope`` this is a position-preserving gather; with it, keys are
    relocated to a contiguous prefix. Empty and contiguous keeps skip the rotation.
    """
    from transformers import DynamicCache

    kv = cache_kv(past)
    cache = cast(MutableHFCache, DynamicCache())
    if not kv:
        return cache
    keep_list = list(keep)
    idx = torch.tensor(keep_list, dtype=torch.long)
    reloc: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    if rope is not None and keep_list != list(range(len(keep_list))):
        device = kv[0][0].device
        src = idx.to(device).unsqueeze(0)
        tgt = torch.arange(len(keep_list), device=device, dtype=torch.long).unsqueeze(0)
        src_cos, src_sin = rope_cos_sin(rope, src)
        tgt_cos, tgt_sin = rope_cos_sin(rope, tgt)
        reloc = (src_cos, src_sin, tgt_cos, tgt_sin)
    idx_by_device: dict[torch.device, torch.Tensor] = {}

    def idx_on(device: torch.device) -> torch.Tensor:
        shipped = idx_by_device.get(device)
        if shipped is None:
            shipped = idx.to(device)
            idx_by_device[device] = shipped
        return shipped

    for layer, (key, value) in enumerate(kv):
        new_key = key.index_select(2, idx_on(key.device))
        new_value = value.index_select(2, idx_on(value.device))
        if reloc is not None:
            src_cos, src_sin, tgt_cos, tgt_sin = reloc
            new_key = apply_rope_k(
                unapply_rope_k(new_key, src_cos, src_sin),
                tgt_cos,
                tgt_sin,
            )
        cache.update(new_key, new_value, layer)
    return cache


def concat_caches(caches: Sequence[Any], rope: RopeParams) -> Any:
    """Relocate and concatenate batch-one caches into one contiguous cache.

    Keys in later blocks are unrotated from their local range and rerotated at the
    running offset; the first block is not rotated, so one block is bit-identical.
    """
    from transformers import DynamicCache

    if not caches:
        raise ValueError("concat_caches needs at least one cache to concatenate")
    per_cache = [cache_kv(cache) for cache in caches]
    n_layers = len(per_cache[0])
    for kv in per_cache:
        if len(kv) != n_layers:
            raise ValueError("concat_caches requires the same per-layer count across caches")
    cache = cast(MutableHFCache, DynamicCache())
    if n_layers == 0:
        return cache
    lengths = [int(kv[0][0].shape[2]) for kv in per_cache]
    relocations: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None] = []
    offset = 0
    for index, length in enumerate(lengths):
        if offset == 0:
            relocations.append(None)
        else:
            device = per_cache[index][0][0].device
            source = torch.arange(length, device=device, dtype=torch.long).unsqueeze(0)
            target = torch.arange(
                offset,
                offset + length,
                device=device,
                dtype=torch.long,
            ).unsqueeze(0)
            src_cos, src_sin = rope_cos_sin(rope, source)
            tgt_cos, tgt_sin = rope_cos_sin(rope, target)
            relocations.append((src_cos, src_sin, tgt_cos, tgt_sin))
        offset += length
    for layer in range(n_layers):
        key_blocks: list[torch.Tensor] = []
        value_blocks: list[torch.Tensor] = []
        for index, kv in enumerate(per_cache):
            key, value = kv[layer]
            relocation = relocations[index]
            if relocation is None:
                key_blocks.append(key)
            else:
                src_cos, src_sin, tgt_cos, tgt_sin = relocation
                key_blocks.append(
                    apply_rope_k(
                        unapply_rope_k(key, src_cos, src_sin),
                        tgt_cos,
                        tgt_sin,
                    )
                )
            value_blocks.append(value)
        cache.update(torch.cat(key_blocks, dim=2), torch.cat(value_blocks, dim=2), layer)
    return cache


def split_past_rows(past: Any, live_mask: torch.Tensor) -> list[Any]:
    """Split a padded batched cache into exact live batch-one caches.

    Live columns are gathered in order and keep the rotations written at their true
    per-item positions, so no relocation occurs.
    """
    from transformers import DynamicCache

    kv = cache_kv(past)
    out: list[Any] = []
    for row in range(int(live_mask.shape[0])):
        index = live_mask[row].nonzero(as_tuple=True)[0]
        cache = cast(MutableHFCache, DynamicCache())
        for layer, (key, value) in enumerate(kv):
            cache.update(
                key[row : row + 1].index_select(2, index.to(key.device)),
                value[row : row + 1].index_select(2, index.to(value.device)),
                layer,
            )
        out.append(cache)
    return out


def compress_past_rows(
    past: Any,
    live_mask: torch.Tensor,
    keeps: Sequence[Sequence[int]],
    rope: RopeParams | None = None,
) -> list[Any]:
    """Cut retained logical positions directly from each padded cache row."""
    from transformers import DynamicCache

    if live_mask.dim() != 2:
        raise ValueError("live_mask must be shaped [batch, length]")
    batch, width = (int(size) for size in live_mask.shape)
    if len(keeps) != batch:
        raise ValueError("keeps must contain one logical index sequence per batch row")
    kv = cache_kv(past)
    if kv:
        if int(kv[0][0].shape[0]) != batch:
            raise ValueError("cache batch must match live_mask batch")
        if int(kv[0][0].shape[2]) != width:
            raise ValueError("cache width must match live_mask width")

    logical_keeps = [list(keep) for keep in keeps]
    live_indices = [live_mask[row].nonzero(as_tuple=True)[0] for row in range(batch)]
    for row, (keep, live_index) in enumerate(zip(logical_keeps, live_indices, strict=True)):
        live = int(live_index.numel())
        if any(index < 0 or index >= live for index in keep):
            raise IndexError(f"row {row} keep index is outside its {live} live positions")

    out: list[Any] = []
    for row, (keep, live_index) in enumerate(zip(logical_keeps, live_indices, strict=True)):
        cache = cast(MutableHFCache, DynamicCache())
        logical = torch.tensor(keep, dtype=torch.long)
        relocate = rope is not None and keep != list(range(len(keep)))
        devices = {tensor.device for pair in kv for tensor in pair}
        logical_by_device = {device: logical.to(device) for device in devices}
        physical_by_device = {
            device: live_index.to(device).index_select(0, logical_by_device[device])
            for device in devices
        }
        reloc_by_device: dict[
            torch.device,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        if relocate:
            assert rope is not None
            for device in {key.device for key, _value in kv}:
                source = logical_by_device[device].unsqueeze(0)
                target = torch.arange(len(keep), dtype=torch.long, device=device).unsqueeze(0)
                src_cos, src_sin = rope_cos_sin(rope, source)
                tgt_cos, tgt_sin = rope_cos_sin(rope, target)
                reloc_by_device[device] = (src_cos, src_sin, tgt_cos, tgt_sin)

        for layer, (key, value) in enumerate(kv):
            new_key = key[row : row + 1].index_select(2, physical_by_device[key.device])
            new_value = value[row : row + 1].index_select(2, physical_by_device[value.device])
            if relocate:
                src_cos, src_sin, tgt_cos, tgt_sin = reloc_by_device[key.device]
                new_key = apply_rope_k(
                    unapply_rope_k(new_key, src_cos, src_sin),
                    tgt_cos,
                    tgt_sin,
                )
            cache.update(new_key, new_value, layer)
        out.append(cache)
    return out


def pad_past_rows(caches: Sequence[Any]) -> BatchedPast:
    """Left-pad batch-one caches into one cache for a shared next segment.

    Live keys keep their rotations and end at the common right edge, so the next
    batched segment needs no relocation. Dense attention only; one copy per hop.
    """
    from transformers import DynamicCache

    if not caches:
        raise ValueError("padding needs at least one cache")
    per_item = [cache_kv(cache) for cache in caches]
    lengths = [int(cache_length(cache)) for cache in caches]
    batch = len(caches)
    width = max(lengths)
    device = per_item[0][0][0].device
    mask = torch.zeros(batch, width, dtype=torch.long, device=device)
    for row, length in enumerate(lengths):
        if length:
            mask[row, width - length :] = 1
    padded = cast(MutableHFCache, DynamicCache())
    for layer, (key_zero, _value_zero) in enumerate(per_item[0]):
        heads, dim = int(key_zero.shape[1]), int(key_zero.shape[3])
        keys = torch.zeros(
            batch,
            heads,
            width,
            dim,
            dtype=key_zero.dtype,
            device=key_zero.device,
        )
        values = torch.zeros(
            batch,
            heads,
            width,
            dim,
            dtype=key_zero.dtype,
            device=key_zero.device,
        )
        for row, length in enumerate(lengths):
            if length:
                key, value = per_item[row][layer]
                keys[row, :, width - length :, :] = key[0]
                values[row, :, width - length :, :] = value[0]
        padded.update(keys, values, layer)
    return BatchedPast(past=padded, mask=mask, live=list(lengths))


def model_rope(model: Any) -> RopeParams:
    """Capture a model's rotary parameters for off-model cache relocation."""
    get_decoder = getattr(model, "get_decoder", None)
    backbone: Any = get_decoder() if callable(get_decoder) else getattr(model, "model", model)
    rotary = backbone.rotary_emb
    return RopeParams(
        inv_freq=rotary.inv_freq.detach().clone(),
        attention_scaling=float(getattr(rotary, "attention_scaling", 1.0)),
    )


__all__ = [
    "BatchedPast",
    "KVPair",
    "cache_kv",
    "cache_length",
    "cache_logical_length",
    "compress_past",
    "compress_past_rows",
    "concat_caches",
    "model_rope",
    "pad_past_rows",
    "split_past_rows",
]
