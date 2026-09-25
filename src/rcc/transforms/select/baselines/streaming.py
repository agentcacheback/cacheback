"""StreamingLLM-style recency scoring: each position scored by how recent it is."""

from __future__ import annotations

import torch

__all__ = ["streaming_scores"]


def streaming_scores(length: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    """Return one float32 recency score per cache position."""
    if length < 0:
        raise ValueError(f"length must be non-negative, got {length}")
    return torch.arange(length, dtype=torch.float32, device=device)
