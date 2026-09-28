"""Selected sender rows, encoded densely or as token IDs plus continuous rows."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as functional

from rcc.selection import Selection

Representation = Literal["embeddings", "token_ids+continuous"]


@dataclass(frozen=True)
class Message:
    """A handoff payload; token ID -1 marks a continuous row at that position."""

    continuous_rows: torch.Tensor
    token_ids: torch.Tensor | None = None
    selection: Selection | None = None

    def __post_init__(self) -> None:
        """Reject malformed payloads before a receiver allocates or embeds them."""
        rows, ids = self.continuous_rows, self.token_ids
        if rows.ndim != 2 or not rows.is_floating_point() or rows.shape[1] == 0:
            raise ValueError("continuous_rows must be a floating [rows, hidden] tensor")
        if ids is None:
            if rows.shape[0] == 0:
                raise ValueError("a message must contain at least one position")
            return
        if ids.ndim != 1 or ids.dtype != torch.long or ids.numel() == 0:
            raise ValueError("token_ids must be a nonempty int64 vector")
        if bool((ids < -1).any()) or int((ids == -1).sum()) != rows.shape[0]:
            raise ValueError("each -1 token ID must identify exactly one continuous row")

    @property
    def positions(self) -> int:
        """Return the number of positions the receiver will prefill."""
        return self.continuous_rows.shape[0] if self.token_ids is None else self.token_ids.numel()

    @property
    def nbytes(self) -> int:
        """Count tensor payload bytes, excluding serialization metadata."""
        rows = self.continuous_rows
        ids = self.token_ids
        return rows.numel() * rows.element_size() + (
            0 if ids is None else ids.numel() * ids.element_size()
        )

    def materialize(self, embedding_weight: torch.Tensor) -> torch.Tensor:
        """Reconstruct [positions, hidden] rows with the receiver's matching embedding table."""
        if embedding_weight.ndim != 2 or not embedding_weight.is_floating_point():
            raise ValueError("embedding_weight must be a floating [vocab, hidden] tensor")
        if embedding_weight.shape[1] != self.continuous_rows.shape[1]:
            raise ValueError("sender and receiver embedding widths differ")
        rows = self.continuous_rows.to(embedding_weight)
        if self.token_ids is None:
            return rows
        ids = self.token_ids.to(embedding_weight.device)
        discrete = ids >= 0
        if bool((ids[discrete] >= embedding_weight.shape[0]).any()):
            raise ValueError("a token ID is outside the receiver's vocabulary")
        with torch.no_grad():
            result = embedding_weight.new_empty((self.positions, embedding_weight.shape[1]))
            result[discrete] = functional.embedding(ids[discrete], embedding_weight)
            result[~discrete] = rows
        return result


def encode_message(rows: torch.Tensor, token_ids: torch.Tensor | None) -> Message:
    """Encode selected rows densely, or experimentally as aligned IDs plus only continuous rows."""
    rows = rows.detach()
    if token_ids is None:
        return Message(rows.cpu().contiguous())
    ids = token_ids.detach().cpu().contiguous()
    warnings.warn(
        "token_ids+continuous is experimental; models beyond dense Qwen3 are untested end to end",
        UserWarning,
        stacklevel=2,
    )
    return Message(rows[(ids == -1).to(rows.device)].cpu().contiguous(), ids)
