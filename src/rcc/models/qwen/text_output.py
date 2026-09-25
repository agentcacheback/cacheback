"""Completion data contracts shared by Qwen text generation and its backend."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class TextCompletion(Protocol):
    """Minimal vLLM completion fields retained for a worker report."""

    request_id: str
    prompt_token_ids: Sequence[int]
    text: str
    n_tokens: int
    token_ids: Sequence[int]
    finish_reason: str
    num_cached_tokens: int | None
    queued_ts: float | None
    scheduled_ts: float | None
    first_token_ts: float | None


@dataclass
class ContinuedCompletion:
    """One draw whose unclosed reasoning block was closed and decoded on."""

    request_id: str
    prompt_token_ids: Sequence[int]
    text: str
    n_tokens: int
    token_ids: Sequence[int]
    finish_reason: str
    num_cached_tokens: int | None
    queued_ts: float | None
    scheduled_ts: float | None
    first_token_ts: float | None
