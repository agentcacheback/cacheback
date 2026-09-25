"""Route source-preparation identity for the FanOutQA panel."""

from __future__ import annotations

from dataclasses import dataclass

from rcc.models.route import RouteFamily


@dataclass(frozen=True)
class QwenSourceBuildConfig:
    """Fields required to rebuild prepared source truth offline."""

    sol_checkpoint: str
    sol_revision: str
    family: RouteFamily


def source_build_config(*, family: RouteFamily) -> QwenSourceBuildConfig:
    """Resolve the family's pinned receiver tokenizer and source geometry."""
    profile = family.profile
    return QwenSourceBuildConfig(profile.checkpoint, profile.revision, family)


__all__ = ("QwenSourceBuildConfig", "source_build_config")
