"""Pure dense-page artifacts and extraction for Ministral 3 capture.

Nothing here touches the engine; the capture bridge calls it with the tensors
vLLM registered.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.self_attn\.attn$")


@dataclass(frozen=True)
class MinistralLayerKV:
    """One dense layer's device-resident cache and logical prompt interval."""

    layer_index: int
    layer_name: str
    keys: torch.Tensor
    values: torch.Tensor
    logical_length: int
    start_position: int


@dataclass(frozen=True)
class MinistralDenseKV:
    """All dense prompt caches captured after one completed prefill."""

    logical_length: int
    layers: tuple[MinistralLayerKV, ...]


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
            f"unrecognised Ministral cache layer {name!r}; expected '*.layers.N.self_attn.attn'"
        )
    return int(match.group(1))


def _dense_group_geometry(kv_cache_config: Any) -> tuple[Any, int, int, int]:
    """Return and validate the one registered dense group geometry."""
    groups: list[Any] = list(getattr(kv_cache_config, "kv_cache_groups", ()) or ())
    if len(groups) != 1:
        raise ValueError(
            f"Ministral capture requires exactly one dense KV cache group, got {len(groups)}"
        )
    group = groups[0]
    spec = group.kv_cache_spec
    sliding = getattr(spec, "sliding_window", None)
    if sliding is not None:
        raise ValueError(
            f"Ministral capture requires dense attention, got sliding_window={sliding}"
        )
    block_size = int(getattr(spec, "block_size", 0) or 0)
    kv_heads = int(getattr(spec, "num_kv_heads", 0) or 0)
    head_dim = int(getattr(spec, "head_size", 0) or 0)
    if min(block_size, kv_heads, head_dim) <= 0:
        raise ValueError("Ministral KV cache group has invalid attention geometry")
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
        if pages.ndim != 4 or tuple(pages.shape[1:]) != expected_tail:
            raise ValueError(
                f"{layer_name} pages are {tuple(pages.shape)}, expected "
                f"[blocks, {kv_heads}, {block_size}, {2 * head_dim}]"
            )
        declared_dtype = getattr(spec, "dtype", pages.dtype)
        if pages.dtype != declared_dtype:
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
        raise ValueError(f"Ministral layer indices {indices} are not contiguous from zero")
    return tuple(mapped)


def read_layer_pages(
    kv_caches: Mapping[str, torch.Tensor], kv_cache_config: Any
) -> tuple[LayerPages, ...]:
    """Validate Ministral's single dense cache group and order its layers."""
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


def extract_dense_kv(layers: Sequence[LayerPages], spec: _ExtractionSpec) -> MinistralDenseKV:
    """Gather one request's dense pages into device-resident HF-shaped layer records.

    The gather is a fresh copy on the pages' device, never a view of the pool, and stays there:
    a host round trip would move about 7.6 GiB per worker each way at the 50K geometry.
    """
    length = int(spec.num_prompt_tokens)
    if length < 1:
        raise ValueError(f"prompt must be non-empty, got {length}")
    if len(spec.block_ids) != 1:
        raise RuntimeError(
            f"Ministral extraction received {len(spec.block_ids)} cache block groups"
        )
    records: list[MinistralLayerKV] = []
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
        if min(chosen) < 0 or max(chosen) >= int(layer.pages.shape[0]):
            raise RuntimeError(f"{layer.layer_name}: retained block id is outside the page pool")
        block_index = torch.tensor(chosen, dtype=torch.long, device=layer.pages.device)
        fused = layer.pages.index_select(0, block_index)
        fused = fused.permute(1, 0, 2, 3).reshape(
            layer.kv_heads,
            len(chosen) * layer.block_size,
            2 * layer.head_dim,
        )
        fused = fused[:, :length].contiguous()
        keys = fused[..., : layer.head_dim].unsqueeze(0)
        values = fused[..., layer.head_dim :].unsqueeze(0)
        if not bool(torch.isfinite(keys).all()) or not bool(torch.isfinite(values).all()):
            raise RuntimeError(f"{layer.layer_name}: extracted cache contains a non-finite value")
        records.append(
            MinistralLayerKV(
                layer_index=layer.layer_index,
                layer_name=layer.layer_name,
                keys=keys,
                values=values,
                logical_length=length,
                start_position=0,
            )
        )
    return MinistralDenseKV(logical_length=length, layers=tuple(records))


__all__ = (
    "MinistralDenseKV",
    "MinistralLayerKV",
    "extract_dense_kv",
    "read_layer_pages",
)
