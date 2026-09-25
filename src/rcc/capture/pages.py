"""Pure dense-page artifacts and extraction for Qwen3 capture."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.self_attn\.attn$")


@dataclass(frozen=True)
class LayerKV:
    """One dense layer's device-resident cache and logical prompt interval."""

    layer_index: int
    layer_name: str
    keys: torch.Tensor
    values: torch.Tensor
    logical_length: int
    start_position: int


@dataclass(frozen=True)
class DenseKV:
    """All dense prompt caches captured after one completed prefill."""

    logical_length: int
    layers: tuple[LayerKV, ...]


@dataclass(frozen=True)
class LayerPages:
    layer_index: int
    layer_name: str
    group_index: int
    block_size: int
    kv_heads: int
    head_dim: int
    pages: torch.Tensor


class _ExtractionSpec(Protocol):
    """Structural request fields consumed by the pure page extractor."""

    num_prompt_tokens: int
    block_ids: tuple[tuple[int, ...], ...]


def _layer_index(name: str) -> int:
    match = _LAYER_RE.search(name)
    if match is None:
        raise ValueError(
            f"unrecognised Qwen3 cache layer {name!r}; expected '*.layers.N.self_attn.attn'"
        )
    return int(match.group(1))


def _dense_group_geometry(kv_cache_config: Any) -> tuple[Any, int, int, int]:
    """Return and validate the one registered dense group geometry."""
    groups: list[Any] = list(getattr(kv_cache_config, "kv_cache_groups", ()) or ())
    if len(groups) != 1:
        raise ValueError(
            f"Qwen3 capture requires exactly one dense KV cache group, got {len(groups)}"
        )
    group = groups[0]
    spec = group.kv_cache_spec
    sliding = getattr(spec, "sliding_window", None)
    if sliding is not None:
        raise ValueError(f"Qwen3 capture requires dense attention, got sliding_window={sliding}")
    block_size = int(getattr(spec, "block_size", 0) or 0)
    kv_heads = int(getattr(spec, "num_kv_heads", 0) or 0)
    head_dim = int(getattr(spec, "head_size", 0) or 0)
    if min(block_size, kv_heads, head_dim) <= 0:
        raise ValueError("Qwen3 KV cache group has invalid attention geometry")
    return group, block_size, kv_heads, head_dim


def _dense_group_pages(
    kv_caches: Mapping[str, torch.Tensor],
    group: Any,
    block_size: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[list[LayerPages], set[str]]:
    """Validate every layer tensor in the one dense cache group."""
    mapped: list[LayerPages] = []
    seen_names: set[str] = set()
    spec = group.kv_cache_spec
    for layer_name in group.layer_names:
        if layer_name in seen_names:
            raise ValueError(f"cache layer {layer_name!r} occurs more than once")
        seen_names.add(layer_name)
        pages = kv_caches.get(layer_name)
        if pages is None:
            raise ValueError(f"vLLM did not register pages for {layer_name!r}")
        expected_tail = (kv_heads, block_size, 2 * head_dim)
        packed = pages.ndim == 4 and tuple(pages.shape[1:]) == expected_tail
        legacy = (
            pages.ndim == 5
            and pages.shape[0] == 2
            and tuple(pages.shape[2:]) == (block_size, kv_heads, head_dim)
        )
        if not (packed or legacy):
            raise ValueError(
                f"{layer_name}: unsupported page shape {tuple(pages.shape)} for dense Qwen3"
            )
        declared_dtype = getattr(spec, "dtype", pages.dtype)
        if pages.dtype != declared_dtype or pages.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError(
                f"{layer_name} page dtype {pages.dtype} != group dtype {declared_dtype}"
            )
        mapped.append(
            LayerPages(
                layer_index=_layer_index(layer_name),
                layer_name=layer_name,
                group_index=0,
                block_size=block_size,
                kv_heads=kv_heads,
                head_dim=head_dim,
                pages=pages,
            )
        )
    return mapped, seen_names


def _ordered_dense_pages(
    kv_caches: Mapping[str, torch.Tensor],
    mapped: list[LayerPages],
    seen_names: set[str],
) -> tuple[LayerPages, ...]:
    """Reject unmapped or non-contiguous layers and return decoder order."""
    if set(kv_caches) != seen_names:
        extras = sorted(set(kv_caches) - seen_names)
        raise ValueError(f"unmapped vLLM KV cache layers: {extras}")
    mapped.sort(key=lambda layer: layer.layer_index)
    indices = [layer.layer_index for layer in mapped]
    if indices != list(range(len(mapped))):
        raise ValueError(f"Qwen3 layer indices {indices} are not contiguous from zero")
    return tuple(mapped)


def read_layer_pages(
    kv_caches: Mapping[str, torch.Tensor], kv_cache_config: Any
) -> tuple[LayerPages, ...]:
    """Validate Qwen3's single dense cache group and order its layers."""
    if not kv_caches:
        raise ValueError("vLLM registered no KV cache pages")
    group, block_size, kv_heads, head_dim = _dense_group_geometry(kv_cache_config)
    mapped, seen_names = _dense_group_pages(
        kv_caches,
        group,
        block_size,
        kv_heads,
        head_dim,
    )
    return _ordered_dense_pages(kv_caches, mapped, seen_names)


def _gather_layer(
    layer: LayerPages, chosen: Sequence[int], length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather logical positions from the two pinned FlashAttention page layouts."""
    index = torch.tensor(chosen, dtype=torch.long, device=layer.pages.device)
    if layer.pages.ndim == 5:
        picked = layer.pages.index_select(1, index)
        rows = picked.permute(0, 3, 1, 2, 4).reshape(2, layer.kv_heads, -1, layer.head_dim)
        return rows[0, :, :length].unsqueeze(0), rows[1, :, :length].unsqueeze(0)
    fused = layer.pages.index_select(0, index)
    fused = fused.permute(1, 0, 2, 3).reshape(layer.kv_heads, -1, 2 * layer.head_dim)
    fused = fused[:, :length].contiguous()
    return fused[..., : layer.head_dim].unsqueeze(0), fused[..., layer.head_dim :].unsqueeze(0)


def extract_dense_kv(layers: Sequence[LayerPages], spec: _ExtractionSpec) -> DenseKV:
    """Gather one request's dense pages into device-resident HF-shaped layer records."""
    length = int(spec.num_prompt_tokens)
    if length < 1:
        raise ValueError(f"prompt must be non-empty, got {length}")
    if len(spec.block_ids) != 1:
        raise RuntimeError(f"Qwen3 extraction received {len(spec.block_ids)} cache block groups")
    records: list[LayerKV] = []
    for layer in layers:
        end_block = (length + layer.block_size - 1) // layer.block_size
        group_blocks = spec.block_ids[layer.group_index]
        if len(group_blocks) < end_block:
            raise RuntimeError(
                f"{layer.layer_name}: block table has {len(group_blocks)} entries, "
                f"needs {end_block} for {length} tokens"
            )
        chosen = group_blocks[:end_block]
        if not chosen or 0 in chosen or len(set(chosen)) != len(chosen):
            raise RuntimeError(
                f"{layer.layer_name}: retained block ids {chosen} contain a null or duplicate"
            )
        capacity = layer.pages.shape[1] if layer.pages.ndim == 5 else layer.pages.shape[0]
        if min(chosen) < 0 or max(chosen) >= capacity:
            raise RuntimeError(f"{layer.layer_name}: retained block id is outside the page pool")
        keys, values = _gather_layer(layer, chosen, length)
        if not bool(torch.isfinite(keys).all()) or not bool(torch.isfinite(values).all()):
            raise RuntimeError(f"{layer.layer_name}: extracted cache contains a non-finite value")
        records.append(
            LayerKV(
                layer_index=layer.layer_index,
                layer_name=layer.layer_name,
                keys=keys,
                values=values,
                logical_length=length,
                start_position=0,
            )
        )
    return DenseKV(logical_length=length, layers=tuple(records))


__all__ = (
    "DenseKV",
    "LayerKV",
    "extract_dense_kv",
    "read_layer_pages",
)
