"""Lazy vLLM 0.26 connector for Gemma 4's heterogeneous cache groups.

The planner and connector shell are shared in ``rcc.models.capture_seam``;
Gemma's cache groups, sliding schedule, and hybrid artifact stay here.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from rcc.models.capture_seam import (
    CaptureRegistry,
    ExtractSpec,
    build_capture_connector,
    validate_capture_runtime,
)

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.self_attn\.attn$")


@dataclass(frozen=True)
class GemmaLayerKV:
    """One device-resident HF-shaped layer cache and its logical interval."""

    layer_index: int
    layer_name: str
    layer_type: str
    keys: torch.Tensor
    values: torch.Tensor
    logical_length: int
    start_position: int


@dataclass(frozen=True)
class GemmaHybridKV:
    """All per-layer prompt caches captured after one completed prefill."""

    logical_length: int
    layers: tuple[GemmaLayerKV, ...]


@dataclass(frozen=True)
class _LayerPages:
    layer_index: int
    layer_name: str
    layer_type: str
    group_index: int
    block_size: int
    kv_heads: int
    head_dim: int
    sliding_window: int | None
    pages: torch.Tensor


def _layer_index(name: str) -> int:
    match = _LAYER_RE.search(name)
    if match is None:
        raise ValueError(
            f"unrecognised Gemma cache layer {name!r}; expected '*.layers.N.self_attn.attn'"
        )
    return int(match.group(1))


def _group_geometry(group: Any, group_index: int) -> tuple[Any, int, int, int, int | None, str]:
    """Validate and return one heterogeneous cache group's geometry."""
    spec = group.kv_cache_spec
    block_size = int(getattr(spec, "block_size", 0) or 0)
    kv_heads = int(getattr(spec, "num_kv_heads", 0) or 0)
    head_dim = int(getattr(spec, "head_size", 0) or 0)
    sliding = getattr(spec, "sliding_window", None)
    sliding_window = None if sliding is None else int(sliding)
    if min(block_size, kv_heads, head_dim) <= 0:
        raise ValueError(f"KV group {group_index} has invalid attention geometry")
    if sliding_window is not None and sliding_window < 2:
        raise ValueError(f"KV group {group_index} has sliding window {sliding_window}")
    layer_type = "sliding_attention" if sliding_window is not None else "full_attention"
    return spec, block_size, kv_heads, head_dim, sliding_window, layer_type


def _group_pages(
    kv_caches: Mapping[str, torch.Tensor],
    group: Any,
    group_index: int,
    seen_names: set[str],
) -> list[_LayerPages]:
    """Validate and map every layer tensor in one cache group."""
    spec, block_size, kv_heads, head_dim, sliding_window, layer_type = _group_geometry(
        group, group_index
    )
    mapped: list[_LayerPages] = []
    for layer_name in group.layer_names:
        if layer_name in seen_names:
            raise ValueError(f"cache layer {layer_name!r} occurs in multiple groups")
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
            _LayerPages(
                layer_index=_layer_index(layer_name),
                layer_name=layer_name,
                layer_type=layer_type,
                group_index=group_index,
                block_size=block_size,
                kv_heads=kv_heads,
                head_dim=head_dim,
                sliding_window=sliding_window,
                pages=pages,
            )
        )
    return mapped


def _ordered_hybrid_pages(
    kv_caches: Mapping[str, torch.Tensor],
    mapped: list[_LayerPages],
    seen_names: set[str],
) -> tuple[_LayerPages, ...]:
    """Reject unmapped or non-contiguous layers and return decoder order."""
    if set(kv_caches) != seen_names:
        extras = sorted(set(kv_caches) - seen_names)
        raise ValueError(f"unmapped vLLM KV cache layers: {extras}")
    mapped.sort(key=lambda layer: layer.layer_index)
    indices = [layer.layer_index for layer in mapped]
    if indices != list(range(len(mapped))):
        raise ValueError(f"Gemma layer indices {indices} are not contiguous from zero")
    return tuple(mapped)


def read_layer_pages(
    kv_caches: Mapping[str, torch.Tensor], kv_cache_config: Any
) -> tuple[_LayerPages, ...]:
    """Validate vLLM's group map and return layers in decoder order."""
    if not kv_caches:
        raise ValueError("vLLM registered no KV cache pages")
    mapped: list[_LayerPages] = []
    seen_names: set[str] = set()
    groups: list[Any] = list(getattr(kv_cache_config, "kv_cache_groups", ()) or ())
    if not groups:
        raise ValueError("vLLM registered no KV cache groups")
    for group_index, group in enumerate(groups):
        mapped.extend(_group_pages(kv_caches, group, group_index, seen_names))
    return _ordered_hybrid_pages(kv_caches, mapped, seen_names)


def extract_hybrid_kv(layers: Sequence[_LayerPages], spec: ExtractSpec) -> GemmaHybridKV:
    """Gather one request's live pages into device-resident layer records."""
    records: list[GemmaLayerKV] = []
    length = spec.num_prompt_tokens
    for layer in layers:
        start = 0 if layer.sliding_window is None else max(0, length - (layer.sliding_window - 1))
        first_block = start // layer.block_size
        end_block = (length + layer.block_size - 1) // layer.block_size
        group_blocks = spec.block_ids[layer.group_index]
        if len(group_blocks) < end_block:
            raise RuntimeError(
                f"{layer.layer_name}: block table has {len(group_blocks)} entries, "
                f"needs {end_block} for {length} tokens"
            )
        chosen = group_blocks[first_block:end_block]
        if not chosen or 0 in chosen or len(set(chosen)) != len(chosen):
            raise RuntimeError(
                f"{layer.layer_name}: retained block ids {chosen} contain a null or duplicate"
            )
        if min(chosen) < 0 or max(chosen) >= int(layer.pages.shape[0]):
            raise RuntimeError(f"{layer.layer_name}: retained block id is outside the page pool")
        block_index = torch.tensor(chosen, dtype=torch.long, device=layer.pages.device)
        fused = layer.pages.index_select(0, block_index)
        fused = fused.permute(1, 0, 2, 3).reshape(
            layer.kv_heads, len(chosen) * layer.block_size, 2 * layer.head_dim
        )
        offset = start - first_block * layer.block_size
        fused = fused[:, offset : offset + (length - start)].contiguous()
        keys = fused[..., : layer.head_dim].unsqueeze(0)
        values = fused[..., layer.head_dim :].unsqueeze(0)
        records.append(
            GemmaLayerKV(
                layer_index=layer.layer_index,
                layer_name=layer.layer_name,
                layer_type=layer.layer_type,
                keys=keys,
                values=values,
                logical_length=length,
                start_position=start,
            )
        )
    return GemmaHybridKV(logical_length=length, layers=tuple(records))


_REGISTRY = CaptureRegistry("Gemma")


def request_gemma_extraction(request_id: str) -> None:
    """Mark a fresh engine request for extraction before admission."""
    _REGISTRY.mark(request_id)


def take_gemma_extraction(request_id: str) -> GemmaHybridKV:
    """Consume one completed device artifact, failing if the connector never produced it."""
    artifact: GemmaHybridKV = _REGISTRY.take(request_id)
    return artifact


def discard_gemma_extraction(request_id: str) -> None:
    """Drop request-scoped marker/artifact state after success or failure."""
    _REGISTRY.release(request_id)


def _prepare_connector(envs: Any, vllm_config: Any, kv_cache_config: Any) -> int:
    """Validate the TP=1 preconditions and count Gemma's heterogeneous groups."""
    validate_capture_runtime("Gemma", envs, vllm_config)
    return len(kv_cache_config.kv_cache_groups)


_lazy_connector: Any = None


def __getattr__(name: str) -> Any:
    if name != "GemmaCaptureConnector":
        raise AttributeError(name)
    global _lazy_connector
    if _lazy_connector is None:
        _lazy_connector = build_capture_connector(
            name="GemmaCaptureConnector",
            summary="Read-only, HMA-aware extraction connector for one TP=1 Gemma engine.",
            registry=_REGISTRY,
            prepare=_prepare_connector,
            read_pages=read_layer_pages,
            extract=extract_hybrid_kv,
        )
    return _lazy_connector
