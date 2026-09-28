"""Request-conditioned support statistics shared by CacheBack and QSnap.

The request rows are reduced in chunks, so the full softmax never materializes.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional

from rcc.selectors._capture import run_capture

ROW_CHUNK = 256
POOL_KERNEL = 7
# Keeps squared attention weights well above float32 underflow; it cancels in the ratio.
MOMENT_SCALE = 256.0


def _add(total: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
    """Accumulate into an optional running sum."""
    return value if total is None else total + value


def pool(values: torch.Tensor) -> torch.Tensor:
    """Max-pool the last axis per leading stream, keeping neighbours of high scores together."""
    length = values.shape[-1]
    pooled = functional.max_pool1d(
        values.reshape(-1, 1, length), POOL_KERNEL, stride=1, padding=POOL_KERNEL // 2
    )
    return pooled[..., :length].reshape(values.shape)


def _moment(values: torch.Tensor) -> torch.Tensor:
    """Return the scaled second moment of pooled attention."""
    return (values * MOMENT_SCALE).pow(2.0)


class _SupportFold:
    """Accumulate request-row votes and, for CacheBack, pooled energy and row energy."""

    def __init__(self, memory: int, rows: int, device: torch.device, *, moments: bool) -> None:
        """Prepare the causal mask over the request's own keys."""
        self.memory, self.rows, self.moments = memory, rows, moments
        self.future = torch.ones(rows, rows, dtype=torch.bool, device=device).triu(1)
        self.heads = 1
        self.votes: torch.Tensor | None = None
        self.energy: torch.Tensor | None = None
        self.row_energy: torch.Tensor | None = None

    def __call__(self, query: torch.Tensor, key: torch.Tensor, scaling: float) -> None:
        """Add one layer; energy squares the pooled row mean, row energy the pooled rows."""
        memory, rows = self.memory, self.rows
        if key.shape[2] != memory + rows or query.shape[2] != rows:
            raise RuntimeError("a scored layer must see the whole sender cache")
        heads, kv_heads = int(query.shape[1]), int(key.shape[1])
        groups = heads // kv_heads
        # Same expanded-key matmul as the validated capture, so scores stay bit-identical.
        key_t = key[:, :, None].expand(-1, -1, groups, -1, -1).flatten(1, 2).transpose(2, 3)
        rowsum: torch.Tensor | None = None
        row_energy: torch.Tensor | None = None
        for start in range(0, rows, ROW_CHUNK):
            end = min(start + ROW_CHUNK, rows)
            logits = (torch.matmul(query[:, :, start:end], key_t) * scaling)[0]
            logits[..., memory:].masked_fill_(self.future[start:end], float("-inf"))
            weights = torch.softmax(logits, dim=-1, dtype=torch.float32)[..., :memory]
            self.votes = _add(self.votes, weights.sum(dim=(0, 1)))
            if self.moments:
                grouped = weights.view(kv_heads, groups, end - start, memory).mean(dim=1)
                rowsum = _add(rowsum, grouped.sum(dim=1))
                row_energy = _add(row_energy, _moment(pool(grouped)).sum(dim=(0, 1)))
        self.heads = heads
        if rowsum is not None and row_energy is not None:
            self.energy = _add(self.energy, _moment(pool(rowsum / rows)).sum(dim=0))
            self.row_energy = _add(self.row_energy, row_energy / rows)

    def mean_votes(self) -> torch.Tensor:
        """Average the summed votes over heads and request rows."""
        if self.votes is None:
            raise RuntimeError("support capture scored no layer")
        return self.votes / (self.heads * self.rows)


def _capture(model: Any, past: Any, request_ids: torch.Tensor, *, moments: bool) -> _SupportFold:
    """Run the request over the whole sender cache and return the folded statistics."""
    memory = int(past.get_seq_length())
    fold = _SupportFold(memory, int(request_ids.shape[1]), request_ids.device, moments=moments)
    run_capture(model, past, fold, memory=memory, input_ids=request_ids)
    return fold


def mean_query_attention(model: Any, past: Any, request_ids: torch.Tensor) -> torch.Tensor:
    """Return request attention per sender position, row and head mean, summed over layers."""
    return _capture(model, past, request_ids, moments=False).mean_votes()


def support_moments(
    model: Any, past: Any, request_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mean votes, energy and row energy over the sender cache for one request."""
    fold = _capture(model, past, request_ids, moments=True)
    if fold.energy is None or fold.row_energy is None:
        raise RuntimeError("support capture collected no moments")
    return fold.mean_votes(), fold.energy, fold.row_energy
