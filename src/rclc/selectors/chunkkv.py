"""ChunkKV: request-agnostic chunk scores from the sender's own observation window.

The last ``window_size`` sender rows attend to the earlier positions. Each layer's attention
is averaged over window rows, average-pooled, averaged within KV groups, then averaged over
heads and global layers; chunks keep their mean score and the window is always kept.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as functional

from rclc.selectors._capture import run_capture
from rclc.selectors.cacheback import finite
from rclc.selectors.fixed_spans import keep_spans, selects, span_means

if TYPE_CHECKING:
    from rclc.transport import SenderState


class _WindowFold:
    """Accumulate observation-window attention over the scored prefix."""

    def __init__(self, prefix: int, window: int, kernel_size: int, device: torch.device) -> None:
        """Prepare the causal mask inside the observation window."""
        self.prefix, self.window, self.kernel_size = prefix, window, kernel_size
        self.future = torch.ones(window, window, dtype=torch.bool, device=device).triu(1)
        self.total: torch.Tensor | None = None
        self.layers = 0

    def __call__(self, query: torch.Tensor, key: torch.Tensor, scaling: float) -> None:
        """Add one layer's pooled window attention, one score per position."""
        heads, kv_heads = int(query.shape[1]), int(key.shape[1])
        groups = heads // kv_heads
        keys = key[0, :, None].expand(-1, groups, -1, -1).flatten(0, 1).float()
        logits = torch.matmul(query[0].float(), keys.transpose(1, 2)) * scaling
        logits[..., self.prefix :].masked_fill_(self.future, float("-inf"))
        scored = torch.softmax(logits, dim=-1)[..., : self.prefix].mean(dim=1)
        pooled = functional.avg_pool1d(
            scored[None], self.kernel_size, stride=1, padding=self.kernel_size // 2
        )[0]
        grouped = pooled.view(kv_heads, groups, self.prefix).mean(dim=1)
        window = (grouped.max() + 1.0).expand(kv_heads, self.window)
        layer = torch.cat((grouped, window), dim=1).mean(dim=0)
        self.total = layer if self.total is None else self.total + layer
        self.layers += 1


def chunkkv_scores(
    sender: SenderState, *, chunk_size: int = 16, window_size: int = 32, kernel_size: int = 5
) -> torch.Tensor:
    """Return one chunk-mean ChunkKV score per sender position, leaving the sender unchanged."""
    if type(window_size) is not int or type(kernel_size) is not int or type(chunk_size) is not int:
        raise ValueError("chunk_size, window_size and kernel_size must be integers")
    if chunk_size < 1 or kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("chunk_size must be positive and kernel_size odd and positive")
    rows = sender.input_embeds
    prefix = rows.shape[0] - window_size
    if window_size < 1 or prefix < 1:
        raise ValueError("ChunkKV needs a positive window_size shorter than the sender state")
    fold = _WindowFold(prefix, window_size, kernel_size, rows.device)
    with torch.inference_mode():
        run_capture(
            sender.model,
            sender.past_key_values,
            fold,
            memory=prefix,
            inputs_embeds=rows[prefix:][None],
        )
        if fold.total is None:
            raise RuntimeError("ChunkKV scored no layer")
        return finite(span_means(fold.total / fold.layers, chunk_size), "ChunkKV")


def chunkkv(
    sender: SenderState,
    request_ids: torch.Tensor,
    budget: int,
    *,
    chunk_size: int = 16,
    window_size: int = 32,
    kernel_size: int = 5,
) -> list[int]:
    """Select whole chunks by ChunkKV score; the request does not affect the selection."""
    del request_ids
    if not selects(sender, budget, chunk_size):
        return list(range(sender.input_embeds.shape[0]))
    scores = chunkkv_scores(
        sender, chunk_size=chunk_size, window_size=window_size, kernel_size=kernel_size
    )
    return keep_spans(sender, scores, budget, span_size=chunk_size)
