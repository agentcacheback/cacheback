"""The layer schedule and cache geometry a single-item capture works from.

The decoder's attention schedule comes from its config, and one callback's tensors
become a layer descriptor placed on the logical memory axis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from rcc.transforms.select.core.types import LayerDescriptor

if TYPE_CHECKING:
    from rcc.transforms.select.query_support.capture_bank import (
        LAYER_BANK_SCHEMA,
        LayerBank,
        LayerBankRow,
        LayerBankRowAccumulator,
    )

W16_WIDTH = 16

__all__ = [
    "LAYER_BANK_SCHEMA",
    "W16_WIDTH",
    "LayerBank",
    "LayerBankRow",
    "LayerBankRowAccumulator",
    "assert_fold_geometry",
    "descriptor",
    "expected_layers",
    "layer_types",
    "normal_layer_type",
    "scatter_memory",
    "w16_mean",
]


def __getattr__(name: str) -> object:
    """Re-export the layer bank names from `capture_bank` without importing it eagerly."""
    if name in {
        "LAYER_BANK_SCHEMA",
        "LayerBank",
        "LayerBankRow",
        "LayerBankRowAccumulator",
    }:
        from rcc.transforms.select.query_support import capture_bank

        return getattr(capture_bank, name)
    raise AttributeError(name)


def w16_mean(value: torch.Tensor) -> torch.Tensor:
    """Average the final axis in width-16 blocks, the last one short if it must be."""
    if value.ndim < 1:
        raise ValueError("W16 input needs a position axis")
    if value.shape[-1] == 0:
        return value.new_empty((*value.shape[:-1], 0), dtype=torch.float32)
    source = value.float()
    chunks = tuple(torch.split(source, W16_WIDTH, dim=-1))
    return torch.stack(
        [torch.mean(chunk, dim=-1) for chunk in chunks],
        dim=-1,
    )


def normal_layer_type(value: object) -> str:
    """Map a Transformers layer type to "sliding" or "global"."""
    return "sliding" if "sliding" in str(value) or "chunked" in str(value) else "global"


def layer_types(config: object, expected: int) -> tuple[str, ...]:
    """Return the decoder's attention schedule, one normalized type per layer."""
    configured = getattr(config, "layer_types", None)
    if configured is None:
        return tuple("global" for _ in range(expected))
    values = tuple(normal_layer_type(value) for value in configured)
    if len(values) != expected:
        raise RuntimeError(
            "decoder layer_types length does not match num_hidden_layers: "
            f"{len(values)} != {expected}"
        )
    return values


def expected_layers(config: object) -> int:
    """Return the decoder layer count the callback hit check expects."""
    value = getattr(config, "num_hidden_layers", None)
    if value is None:
        configured = getattr(config, "layer_types", None)
        value = len(configured) if configured is not None else 0
    count = int(value)
    if count < 1:
        raise RuntimeError("capture decoder has no expected attention layers")
    return count


def attention_indices(config: object) -> tuple[int, ...]:
    """Return the indices the capture expects callbacks from."""
    count = expected_layers(config)
    if getattr(config, "model_type", None) != "nemotron_h":
        return tuple(range(count))
    schedule = tuple(getattr(config, "layer_types", ()))
    if len(schedule) != count or any(
        kind not in {"full_attention", "linear_attention", "mlp"} for kind in schedule
    ):
        raise RuntimeError("Nemotron capture requires a complete dense hybrid schedule")
    indices = tuple(i for i, kind in enumerate(schedule) if kind == "full_attention")
    if not indices:
        raise RuntimeError("Nemotron capture needs at least one attention block")
    return indices


def descriptor(
    context: Any,
    module: object,
    key: torch.Tensor,
    query: torch.Tensor,
) -> LayerDescriptor:
    """Build one descriptor from the decoder module and callback tensors.

    The geometry assumes a dynamic or compacted cache: a StaticCache's padded buffer
    makes its physical key length longer than its live memory length.
    """
    layer_idx = int(getattr(module, "layer_idx", context.callback_hits))
    if layer_idx < 0 or layer_idx >= len(context.layer_types):
        raise RuntimeError(f"attention callback layer index {layer_idx} is outside the decoder")
    layer_type = context.layer_types[layer_idx]
    query_length = int(query.shape[-2])
    physical_key_length = int(key.shape[-2])
    physical_memory_length = max(0, physical_key_length - query_length)
    offset = max(0, context.mem_len - physical_memory_length)
    raw_window = getattr(module, "sliding_window", None)
    if raw_window is None and layer_type == "sliding":
        raw_window = getattr(getattr(module, "config", None), "sliding_window", None)
    window = int(raw_window) if raw_window is not None and layer_type == "sliding" else None
    return LayerDescriptor(
        layer_idx=layer_idx,
        layer_type=layer_type,
        logical_memory_length=context.mem_len,
        physical_key_length=physical_key_length,
        absolute_key_offset=offset,
        window=window,
        query_length=query_length,
    )


def assert_fold_geometry(
    descriptor: LayerDescriptor,
    *,
    include_layer: bool,
    scoring_policy: str,
) -> None:
    """Raise unless a shared per-layer fold can be scattered back.

    A scored layer's physical memory must cover the logical memory, so the scatter back
    is the identity, and it must be a global layer, which makes that reachable.
    """
    if include_layer != descriptor.is_global:
        raise RuntimeError(
            "capture scoring policy broke the scored-layer bank invariant: layer "
            f"{descriptor.layer_idx} of type {descriptor.layer_type!r} is "
            f"{'scored' if include_layer else 'unscored'} under scoring policy "
            f"{scoring_policy!r}; every scored layer must be a global layer"
        )
    if include_layer and descriptor.physical_memory_length < descriptor.logical_memory_length:
        raise RuntimeError(
            "capture scoring policy broke the bank scatter invariant: scored layer "
            f"{descriptor.layer_idx} holds {descriptor.physical_memory_length} physical "
            f"memory columns, below the {descriptor.logical_memory_length} logical "
            "columns it folds on, so its banked vectors could not scatter back"
        )


def scatter_memory(
    values: torch.Tensor,
    descriptor: LayerDescriptor,
    logical_length: int,
) -> torch.Tensor:
    """Place one layer's physical memory columns on the logical memory axis, zero elsewhere."""
    scattered = torch.zeros(
        *values.shape[:-1],
        logical_length,
        dtype=values.dtype,
        device=values.device,
    )
    physical_length = min(
        descriptor.physical_memory_length,
        max(0, logical_length - descriptor.absolute_key_offset),
    )
    if physical_length > 0:
        start = descriptor.absolute_key_offset
        scattered[..., start : start + physical_length] = values[..., :physical_length]
    return scattered
