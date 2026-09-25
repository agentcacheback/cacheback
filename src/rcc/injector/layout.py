"""Paged-KV geometry and the contiguous-to-paged bijection, with no engine involved.

The paged side is the FlashAttention GQA layout `[2, num_blocks, block_size,
kv_heads, head_dim]` with identity strides; the contiguous side is `KVCache`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Geometry:
    """The KV geometry one engine and one cache must agree on.

    `layers`, `kv_heads`, and `head_dim` describe the architecture, `block_size` is
    the receiver's paging grain; every field is read from the live engine.
    """

    layers: int
    kv_heads: int
    head_dim: int
    block_size: int

    def __post_init__(self) -> None:
        """Raise for a non-positive dimension."""
        for name in ("layers", "kv_heads", "head_dim", "block_size"):
            value = int(getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")

    def blocks_for(self, k: int) -> int:
        """Return the number of pages needed to hold `k` token slots."""
        if k < 1:
            raise ValueError(f"k must be positive, got {k}")
        return (k + self.block_size - 1) // self.block_size

    def page_shape(self, num_blocks: int) -> tuple[int, int, int, int, int]:
        """Return the per-layer paged shape `[2, num_blocks, block_size, kv_heads, head_dim]`."""
        return (2, num_blocks, self.block_size, self.kv_heads, self.head_dim)


def slot_mapping(block_ids: Sequence[int], k: int, block_size: int) -> torch.Tensor:
    """Map each token index `t` in `0..k-1` to its physical slot in a paged buffer.

    Token `t` lands at `block_ids[t // block_size] * block_size + t % block_size`.
    Block ids too few for `k`, or repeated and so not a bijection, raise.
    """
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    needed = (k + block_size - 1) // block_size
    if len(block_ids) < needed:
        raise ValueError(
            f"{len(block_ids)} block ids cannot cover k={k} at block_size={block_size} "
            f"({needed} needed)"
        )
    used = list(block_ids[:needed])
    lowest = min(used)
    if lowest < 0:
        raise ValueError(f"block id {lowest} is negative; physical page ids start at 0")
    if len(set(used)) != len(used):
        raise ValueError(
            f"block ids repeat in {used}: two token ranges would share one page, so the "
            "scatter would overwrite and the gather would return the same tokens twice"
        )
    t = torch.arange(k, dtype=torch.int64)
    ids = torch.tensor(used, dtype=torch.int64)
    return ids[t // block_size] * block_size + t % block_size


def _check_paged(paged: torch.Tensor, slots: torch.Tensor) -> None:
    """Raise for a paged buffer the flat-view scatter would miswrite or miss."""
    if paged.ndim != 5 or paged.shape[0] != 2:
        raise ValueError(
            f"paged must be [2, num_blocks, block_size, kv_heads, head_dim], got {paged.shape}"
        )
    if not paged.is_contiguous():
        raise ValueError(
            "paged buffer is not contiguous (HND layout?): a reshape scatter would "
            "write into a throwaway copy and silently leave every page unwritten"
        )
    capacity = int(paged.shape[1]) * int(paged.shape[2])
    lowest = int(slots.min())
    if lowest < 0:
        raise ValueError(
            f"slot {lowest} is negative: torch indexing would wrap it into the last page "
            "and silently write the wrong slot"
        )
    top = int(slots.max())
    if top >= capacity:
        raise ValueError(f"slot {top} is beyond the buffer capacity of {capacity} slots")


def pack_layer(
    paged: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, slots: torch.Tensor
) -> None:
    """Scatter one layer's contiguous K/V `[kv_heads, k, head_dim]` into a paged buffer.

    The destination is viewed as `[2, num_blocks * block_size, kv_heads * head_dim]`
    with `view`, never `reshape`, so a non-NHD buffer raises instead of copying.
    """
    _check_paged(paged, slots)
    k = int(slots.shape[0])
    expected = (int(paged.shape[3]), k, int(paged.shape[4]))
    if tuple(keys.shape) != expected or tuple(values.shape) != expected:
        raise ValueError(
            f"keys/values must be {expected}, got {tuple(keys.shape)} / {tuple(values.shape)}"
        )
    if keys.dtype != paged.dtype or values.dtype != paged.dtype:
        raise ValueError(
            f"dtype mismatch: paged {paged.dtype}, keys {keys.dtype}, values {values.dtype}"
        )
    src = torch.stack((keys.transpose(0, 1).reshape(k, -1), values.transpose(0, 1).reshape(k, -1)))
    dst = paged.view(2, int(paged.shape[1]) * int(paged.shape[2]), -1)
    dst[:, slots, :] = src


def unpack_layer(paged: torch.Tensor, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather one layer's K/V `[kv_heads, k, head_dim]` back out, inverse of `pack_layer`."""
    _check_paged(paged, slots)
    flat = paged.view(
        2, int(paged.shape[1]) * int(paged.shape[2]), int(paged.shape[3]), int(paged.shape[4])
    )
    picked = flat[:, slots].transpose(1, 2).contiguous()
    return picked[0], picked[1]


def pack_cache(
    paged_layers: Sequence[torch.Tensor],
    keys: torch.Tensor,
    values: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    """Scatter a full `[layers, kv_heads, k, head_dim]` cache, one paged buffer per layer.

    `paged_layers` is indexed by layer; the connector builds it by parsed layer index,
    not by dict order, which need not follow the layers.
    """
    if keys.shape != values.shape:
        raise ValueError(f"keys {keys.shape} and values {values.shape} disagree")
    if len(paged_layers) != int(keys.shape[0]):
        raise ValueError(
            f"{len(paged_layers)} paged buffers for a {int(keys.shape[0])} layer cache"
        )
    for layer, paged in enumerate(paged_layers):
        pack_layer(paged, keys[layer], values[layer], slots)


def unpack_cache(
    paged_layers: Sequence[torch.Tensor], slots: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather a full cache back into `[layers, kv_heads, k, head_dim]`, inverse of `pack_cache`."""
    pairs = [unpack_layer(paged, slots) for paged in paged_layers]
    keys = torch.stack([k for k, _ in pairs])
    values = torch.stack([v for _, v in pairs])
    return keys, values
