"""The ChunkKV observation-window reduction, with the softmax taken in blocks.

One key-column block is folded at a time, so the row softmax comes from a running
maximum and exponential sum rather than a whole attention matrix.
"""

from __future__ import annotations

import torch
from torch import nn

from rcc.transforms.select.core.kernels import repeat_kv

__all__ = ["native_window_scores", "window_stream_scores"]


def window_stream_scores(
    weights: torch.Tensor,
    *,
    window_size: int,
    kv_groups: int,
    kernel_size: int = 5,
) -> torch.Tensor:
    """Reduce one layer's observation-window attention to one score per KV head."""
    if weights.ndim != 3:
        raise ValueError(
            f"weights must have shape [query heads, window rows, keys], got {tuple(weights.shape)}"
        )
    query_heads, rows, keys = (int(size) for size in weights.shape)
    if window_size < 1 or window_size >= keys:
        raise ValueError(f"window size {window_size} is outside the key length {keys}")
    if rows != window_size:
        raise ValueError(f"weights hold {rows} rows for an observation window of {window_size}")
    if kv_groups < 1 or query_heads % kv_groups:
        raise ValueError(f"{query_heads} query heads do not divide into groups of {kv_groups}")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd and positive, got {kernel_size}")
    scored = weights[:, :, :-window_size].to(torch.float32).mean(dim=-2)
    return _reduce_streams(
        scored.unsqueeze(0),
        kv_groups=kv_groups,
        window_size=window_size,
        kernel_size=kernel_size,
    )[0]


def _reduce_streams(
    scored: torch.Tensor,
    *,
    kv_groups: int,
    window_size: int,
    kernel_size: int,
) -> torch.Tensor:
    """Pool, average over each KV group, and pad the window, per query head."""
    batch, query_heads, prefix = (int(size) for size in scored.shape)
    pooled = nn.functional.avg_pool1d(
        scored, kernel_size=kernel_size, padding=kernel_size // 2, stride=1
    )
    grouped = pooled.view(batch, query_heads // kv_groups, kv_groups, prefix).mean(dim=2)
    return nn.functional.pad(grouped, (0, window_size), value=float(grouped.max()) + 1.0)


def _window_logits(
    query: torch.Tensor,
    key: torch.Tensor,
    start: int,
    end: int,
    scaling: float,
    kv_groups: int,
) -> torch.Tensor:
    """Return the observation-window logits against one key-column block."""
    columns = repeat_kv(key[:, :, start:end, :].to(query.dtype), kv_groups)
    return torch.matmul(query, columns.transpose(-2, -1)) * scaling


def _accumulate_softmax(
    running: torch.Tensor,
    total: torch.Tensor,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold one column block into the running rowwise maximum and exponential sum."""
    updated = torch.maximum(running, logits.amax(dim=-1, keepdim=True))
    rescale = torch.where(
        torch.isneginf(running), torch.zeros_like(running), torch.exp(running - updated)
    )
    return updated, total * rescale + torch.exp(logits - updated).sum(dim=-1, keepdim=True)


def native_window_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    scaling: float,
    window_size: int,
    kernel_size: int = 5,
    column_chunk: int = 4096,
) -> torch.Tensor:
    """Return one observation-window score per token."""
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query and key must both have shape [batch, heads, length, head_dim]")
    batch, query_heads, rows, head_dim = (int(size) for size in query.shape)
    kv_heads, key_length = int(key.shape[1]), int(key.shape[2])
    if int(key.shape[0]) != batch or int(key.shape[3]) != head_dim or query_heads % kv_heads:
        raise ValueError(
            f"query {tuple(query.shape)} and key {tuple(key.shape)} geometry do not align"
        )
    if rows != window_size:
        raise ValueError(f"query holds {rows} rows for an observation window of {window_size}")
    prefix = key_length - window_size
    if prefix < 1 or column_chunk < 1:
        raise ValueError(
            f"window {window_size} and column chunk {column_chunk} leave no scored key columns"
        )
    kv_groups = query_heads // kv_heads
    rows_query = query.to(torch.float32)
    blocks = [
        (start, min(start + column_chunk, prefix)) for start in range(0, prefix, column_chunk)
    ]
    shape = (batch, query_heads, window_size, 1)
    running = torch.full(shape, float("-inf"), dtype=torch.float32, device=query.device)
    total = torch.zeros_like(running)
    for start, end in blocks:
        running, total = _accumulate_softmax(
            running,
            total,
            _window_logits(rows_query, key, start, end, scaling, kv_groups),
        )
    index = torch.arange(window_size, device=query.device)
    causal = torch.where(
        index.unsqueeze(0) > index.unsqueeze(1),
        torch.full((), float("-inf"), dtype=torch.float32, device=query.device),
        torch.zeros((), dtype=torch.float32, device=query.device),
    )
    running, total = _accumulate_softmax(
        running,
        total,
        _window_logits(rows_query, key, prefix, key_length, scaling, kv_groups) + causal,
    )
    scored = torch.zeros(batch, query_heads, prefix, dtype=torch.float32, device=query.device)
    for start, end in blocks:
        logits = _window_logits(rows_query, key, start, end, scaling, kv_groups)
        scored[:, :, start:end] = (torch.exp(logits - running) / total).mean(dim=-2)
    return _reduce_streams(
        scored, kv_groups=kv_groups, window_size=window_size, kernel_size=kernel_size
    )
