"""Download every pinned checkpoint once, before the seats open their engines."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, cast


def prefetch_checkpoints(
    pins: Iterable[tuple[str, str]],
    *,
    allow_patterns: Sequence[str] | None = None,
    repo_type: str = "model",
) -> None:
    """Snapshot each pinned repository into the local hub cache."""
    import huggingface_hub

    for repo, revision in pins:
        cast(Any, huggingface_hub).snapshot_download(
            repo_id=repo, revision=revision, allow_patterns=allow_patterns, repo_type=repo_type
        )


__all__ = ("prefetch_checkpoints",)
