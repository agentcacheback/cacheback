"""The item record one FanOutQA construction pass produces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rcc.benchmarks.fanoutqa.data import (
    Question,
)


@dataclass(frozen=True)
class ProbeItem:
    """One question with its per-worker private evidence shards."""

    qid: str
    question: str
    shards: tuple[tuple[int, ...], ...]
    question_obj: Question
    task_kind: str = "qa"
    #: Native sender prompts, carried as data inside the prepared bundle, or
    #: None on an artifact that carries none.
    native_prompt_artifacts: dict[str, Any] | None = None


__all__ = ("ProbeItem",)
