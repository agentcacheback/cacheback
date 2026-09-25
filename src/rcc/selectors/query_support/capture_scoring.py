"""Turn a backend's prepared attention mask into the additive scoring bias.

Whatever mask form the model's backend passed is converted, and a causal
fallback is built when it passes no tensor at all.
"""

from __future__ import annotations

from typing import Any

import torch

from rcc.selectors.core.types import LayerDescriptor


def _fallback_attention_bias(
    context: Any,
    descriptor: LayerDescriptor,
    row_start: int,
    row_end: int,
    key_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build the causal fallback bias used when a backend passes no tensor mask."""
    physical_memory_length = descriptor.physical_memory_length
    memory_positions = torch.arange(
        descriptor.absolute_key_offset,
        descriptor.absolute_key_offset + physical_memory_length,
        device=device,
    )
    current_length = max(0, key_length - physical_memory_length)
    logical_memory_length = descriptor.logical_memory_length
    current_positions = torch.arange(
        logical_memory_length,
        logical_memory_length + current_length,
        device=device,
    )
    key_positions = torch.cat((memory_positions, current_positions))
    query_positions = torch.arange(
        logical_memory_length + context.lo + row_start,
        logical_memory_length + context.lo + row_end,
        device=device,
    )
    valid = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    if descriptor.window is not None:
        valid &= key_positions.unsqueeze(0) > query_positions.unsqueeze(1) - descriptor.window
    bias = torch.zeros(
        row_end - row_start,
        key_length,
        dtype=dtype,
        device=device,
    )
    return bias.masked_fill(~valid, torch.finfo(dtype).min).view(
        1, 1, row_end - row_start, key_length
    )


def score_attention_bias(
    context: Any,
    descriptor: LayerDescriptor,
    attention_mask: object,
    row_start: int,
    row_end: int,
    key_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return the additive bias for one layer's question rows."""
    if not isinstance(attention_mask, torch.Tensor):
        return _fallback_attention_bias(
            context,
            descriptor,
            row_start,
            row_end,
            key_length,
            dtype,
            device,
        )
    prepared = attention_mask.to(device=device)
    if prepared.ndim == 4:
        query_start = context.lo + row_start
        prepared = prepared[..., query_start : context.lo + row_end, :]
    elif prepared.ndim == 3:
        query_start = context.lo + row_start
        prepared = prepared[:, query_start : context.lo + row_end, :].unsqueeze(1)
    elif prepared.ndim == 2:
        padding = prepared[:, -key_length:].to(torch.bool)
        prepared = _fallback_attention_bias(
            context,
            descriptor,
            row_start,
            row_end,
            key_length,
            dtype,
            device,
        )
        return prepared.masked_fill(~padding[:, None, None, :], torch.finfo(dtype).min)
    else:
        return _fallback_attention_bias(
            context,
            descriptor,
            row_start,
            row_end,
            key_length,
            dtype,
            device,
        )
    if prepared.shape[-1] > key_length:
        prepared = prepared[..., -key_length:]
    if prepared.shape[-1] != key_length:
        return _fallback_attention_bias(
            context,
            descriptor,
            row_start,
            row_end,
            key_length,
            dtype,
            device,
        )
    if prepared.dtype == torch.bool:
        return torch.zeros_like(prepared, dtype=dtype).masked_fill(
            ~prepared,
            torch.finfo(dtype).min,
        )
    return prepared.to(dtype)
