"""Pure dense-page artifacts and extraction for Qwen3 capture."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.self_attn\.attn$")
_PAGE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@dataclass(frozen=True)
class LayerKV:
    """One dense layer's device-resident prompt keys and values."""

    keys: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class DenseKV:
    """All dense prompt caches captured after one completed prefill."""

    logical_length: int
    layers: tuple[LayerKV, ...]


@dataclass(frozen=True)
class LayerPages:
    """One registered layer's page pool and its dense attention geometry."""

    layer_index: int
    layer_name: str
    block_size: int
    kv_heads: int
    head_dim: int
    pages: torch.Tensor


@dataclass(frozen=True)
class ExtractSpec:
    """A completed request and its ordered block table."""

    request_id: str
    num_prompt_tokens: int
    block_ids: tuple[int, ...]


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
        if pages.dtype not in _PAGE_DTYPES:
            raise ValueError(
                f"{layer_name}: unsupported page dtype {pages.dtype}; use cache_dtype='auto'"
            )
        declared_dtype = getattr(spec, "dtype", pages.dtype)
        if pages.dtype != declared_dtype:
            raise ValueError(
                f"{layer_name}: page dtype {pages.dtype} differs from group dtype {declared_dtype}"
            )
        mapped.append(
            LayerPages(
                layer_index=_layer_index(layer_name),
                layer_name=layer_name,
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
    layer: LayerPages, index: torch.Tensor, length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather logical positions from the two pinned FlashAttention page layouts."""
    heads, dim = layer.kv_heads, layer.head_dim
    if layer.pages.ndim == 5:
        rows = layer.pages.index_select(1, index).permute(0, 3, 1, 2, 4).reshape(2, heads, -1, dim)
        return rows[0, :, :length].unsqueeze(0), rows[1, :, :length].unsqueeze(0)
    picked = layer.pages.index_select(0, index)
    keys, values = (
        half.permute(1, 0, 2, 3).reshape(heads, -1, dim)[:, :length].unsqueeze(0)
        for half in (picked[..., :dim], picked[..., dim:])
    )
    return keys, values


def extract_dense_kv(layers: Sequence[LayerPages], spec: ExtractSpec) -> DenseKV:
    """Gather one request's dense pages into device-resident HF-shaped layer records."""
    length = spec.num_prompt_tokens
    first = layers[0]
    end_block = (length + first.block_size - 1) // first.block_size
    chosen = spec.block_ids[:end_block]
    if len(chosen) < end_block:
        raise RuntimeError(
            f"block table has {len(spec.block_ids)} entries, needs {end_block} for {length} tokens"
        )
    if min(chosen) <= 0 or len(set(chosen)) != len(chosen):
        raise RuntimeError(f"retained block ids {chosen} contain a null, negative or duplicate")
    for layer in layers:
        capacity = layer.pages.shape[1] if layer.pages.ndim == 5 else layer.pages.shape[0]
        if max(chosen) >= capacity:
            raise RuntimeError(f"{layer.layer_name}: retained block id is outside the page pool")
    index = torch.tensor(chosen, dtype=torch.long, device=first.pages.device)
    records = tuple(LayerKV(*_gather_layer(layer, index, length)) for layer in layers)
    finite = [torch.isfinite(t).all() for record in records for t in (record.keys, record.values)]
    if not bool(torch.stack(finite).all()):
        raise RuntimeError("extracted cache contains a non-finite value")
    return DenseKV(logical_length=length, layers=records)


__all__ = (
    "DenseKV",
    "ExtractSpec",
    "LayerKV",
    "LayerPages",
    "extract_dense_kv",
    "read_layer_pages",
)
