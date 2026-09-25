"""Durable value objects emitted by the Gemma resident capture phase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rcc.models.gemma.layout import ArmLayout


@dataclass(frozen=True)
class CaptureArtifact:
    """CPU-side capture products retained after the HF model is released."""

    qid: str
    keeps_by_arm: dict[str, tuple[tuple[int, ...], ...]]
    layouts_by_arm: dict[str, ArmLayout]
    rolled_by_worker: tuple[torch.Tensor, ...]
    selection_s_by_arm: dict[str, float]
    audition_s_by_ratio: dict[int, float]
    audition_replays_by_ratio: dict[int, int]
    reports: tuple[dict[str, Any], ...]
    report_failed: bool
    report_failure: str | None
    capture_s: float
    capture_s_by_worker: tuple[float, ...]
    report_generation_s: float
    embedding_digests: dict[str, Any]
    global_layers: tuple[int, ...]
    reloaded_captures: int
    reloaded_selections: int


__all__ = ("CaptureArtifact",)
