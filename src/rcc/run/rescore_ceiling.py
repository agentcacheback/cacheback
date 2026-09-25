"""The report ceiling law every banked report is held to during a rescore.

`rcc.run.rescore` re-exports these three names, so every family's rescore path
takes them from one place.
"""

from __future__ import annotations

from collections.abc import Container, Mapping, Sequence
from typing import Any, cast

_REPORT_FINISH_REASONS = frozenset({"stop", "length"})


def injected_flags(row: Mapping[str, Any], *, label: str, family: str) -> list[bool] | None:
    """Return the banked per-worker injection flags, or None when a row has none."""
    raw = row.get("report_injected_by_worker")
    if raw is None:
        return None
    if not isinstance(raw, list) or any(
        not isinstance(flag, bool) for flag in cast(list[object], raw)
    ):
        raise RuntimeError(f"{label}: {family} report injection evidence is malformed")
    return cast(list[bool], raw)


def content_token_count(token_ids: Sequence[int], stop_token_ids: Container[int]) -> int:
    """Return how many banked ids precede the family's first stop token.

    Every family's rescore path decodes only this prefix, and the count is what
    :func:`validate_report_ceiling` reads, so the two cannot disagree.
    """
    for index, token in enumerate(token_ids):
        if token in stop_token_ids:
            return index
    return len(token_ids)


def validate_report_ceiling(
    content_counts: Sequence[int],
    finish_reasons: Sequence[str],
    *,
    ceiling: int,
    family: str,
    label: str,
    injected: Sequence[object] | None = None,
    closing_extra: int = 0,
) -> None:
    """Hold every banked per-worker report length to its ceiling.

    Stop-trimmed lengths: under a cap equal to the ceiling a natural draw's length and its
    finish reason agree; injected adds ``closing_extra``. See docs/running.md, Report ceiling.
    """
    if len(content_counts) != len(finish_reasons):
        raise RuntimeError(f"{label}: {family} report termination evidence is incomplete")
    raw_flags: list[object] = (
        list(injected) if injected is not None else [False] * len(content_counts)
    )
    if len(raw_flags) != len(content_counts) or any(
        not isinstance(flag, bool) for flag in raw_flags
    ):
        raise RuntimeError(f"{label}: {family} report injection evidence is incomplete")
    flags = [bool(flag) for flag in raw_flags]
    for count, reason, flag in zip(content_counts, finish_reasons, flags, strict=True):
        if reason not in _REPORT_FINISH_REASONS:
            raise RuntimeError(f"{label}: {family} report finish reason is unregistered")
        head_bound = ceiling if reason == "length" else ceiling - 1
        bound = head_bound + closing_extra if flag else ceiling
        if not 0 <= count <= bound:
            raise RuntimeError(f"{label}: {family} report exceeds the registered report ceiling")
        if flag and reason == "length" and count < ceiling:
            raise RuntimeError(f"{label}: {family} injected length head did not fill the ceiling")
        if not flag and (reason == "length") != (count == ceiling):
            raise RuntimeError(f"{label}: {family} length finish differs from the report ceiling")


__all__ = ("content_token_count", "injected_flags", "validate_report_ceiling")
