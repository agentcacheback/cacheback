"""Independent sender-native validation for banked Gemma text reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.models.gemma.text_closer import GEMMA_CLOSING_TOKEN_BUDGET
from rcc.models.gemma.text_codec import (
    GEMMA_STOP_TOKEN_IDS,
    GemmaVisibleReport,
    visible_report,
)
from rcc.models.gemma.text_contract import TEXT_REPORT_SCHEMA, text_sender_spec
from rcc.run.rescore import content_token_count, validate_report_ceiling
from rcc.topologies.fanout import FANOUT_M3


@dataclass
class _RawReport:
    """Raw-token completion reconstructed independently of banked backend text."""

    text: str
    n_tokens: int
    token_ids: Sequence[int]
    finish_reason: str


def _raw_token_rows(value: object, label: str) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label}: Gemma report token evidence is incomplete")
    raw_rows = cast(list[object], value)
    if len(raw_rows) != FANOUT_M3.workers_per_item:
        raise RuntimeError(f"{label}: Gemma report token evidence is incomplete")
    rows: list[tuple[int, ...]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, list):
            raise RuntimeError(f"{label}: Gemma report token ids are malformed")
        values = cast(list[object], raw_row)
        if any(type(token) is not int for token in values):
            raise RuntimeError(f"{label}: Gemma report token ids are malformed")
        tokens = cast(list[int], values)
        if any(token < 0 for token in tokens):
            raise RuntimeError(f"{label}: Gemma report token ids are malformed")
        rows.append(tuple(tokens))
    return tuple(rows)


def _string_rows(value: object, label: str, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label}: Gemma {field} evidence is incomplete")
    rows = cast(list[object], value)
    if len(rows) != FANOUT_M3.workers_per_item:
        raise RuntimeError(f"{label}: Gemma {field} evidence is incomplete")
    if any(not isinstance(row, str) for row in rows):
        raise RuntimeError(f"{label}: Gemma {field} evidence is incomplete")
    return tuple(cast(list[str], rows))


def validate_text_report_row(
    row: Mapping[str, Any],
    *,
    qid: str,
    semantic_arm: str,
    tokenizer: Any,
    expected_report_seeds: Sequence[int],
) -> None:
    """Reconstruct one three-worker report bundle through its sender codec."""
    label = f"{qid}/{semantic_arm}"
    sender = text_sender_spec(semantic_arm)
    if tokenizer is None:
        raise RuntimeError(f"{label}: Gemma sender-native tokenizer is unavailable")
    if (
        row.get("arm") != sender.semantic_arm
        or row.get("semantic_arm") != sender.semantic_arm
        or row.get("policy") != sender.policy
        or row.get("worker_checkpoint") != sender.checkpoint
        or row.get("worker_revision") != sender.revision
        or row.get("receiver_checkpoint") != sender.receiver_checkpoint
        or row.get("receiver_revision") != sender.receiver_revision
        or row.get("report_schema") != TEXT_REPORT_SCHEMA
        or row.get("workers") != FANOUT_M3.workers_per_item
    ):
        raise RuntimeError(f"{label}: Gemma sender identity differs from the registered arm")

    token_rows = _raw_token_rows(row.get("report_token_ids_by_worker"), label)
    finish_reasons = _string_rows(row.get("report_finish_reasons"), label, "finish-reason")
    _string_rows(row.get("reports"), label, "visible-report")
    _string_rows(row.get("report_raw_outputs"), label, "raw-output")
    # The banked counts bind the row fields; the ceiling binds the stop-trimmed
    # content, which is shorter by exactly the stop token on a stop finish.
    token_counts = [len(token_ids) for token_ids in token_rows]
    raw_injected = row.get("report_injected_by_worker")
    injected = None
    if raw_injected is not None:
        if not isinstance(raw_injected, list) or any(
            not isinstance(flag, bool) for flag in cast(list[object], raw_injected)
        ):
            raise RuntimeError(f"{label}: Gemma report injection evidence is malformed")
        injected = cast(list[bool], raw_injected)
    validate_report_ceiling(
        [content_token_count(token_ids, GEMMA_STOP_TOKEN_IDS) for token_ids in token_rows],
        finish_reasons,
        ceiling=FANOUTQA_NATURAL_DEV50.report_ceiling,
        family="Gemma",
        label=label,
        injected=injected,
        # One closer id plus the closing budget.
        closing_extra=1 + GEMMA_CLOSING_TOKEN_BUDGET,
    )
    reports: list[GemmaVisibleReport] = []
    for token_ids, finish_reason in zip(token_rows, finish_reasons, strict=True):
        raw_text = str(tokenizer.decode(token_ids, skip_special_tokens=True))
        reports.append(
            visible_report(
                tokenizer,
                _RawReport(
                    text=raw_text,
                    n_tokens=len(token_ids),
                    token_ids=token_ids,
                    finish_reason=finish_reason,
                ),
            )
        )

    closed = [report.thinking_closed for report in reports]
    failed = [worker for worker, report in enumerate(reports) if not report.accepted]
    expected: dict[str, object] = {
        "reports": [report.text for report in reports],
        "report_raw_outputs": [report.raw_text for report in reports],
        "report_tokens_by_worker": token_counts,
        "report_tokens": sum(token_counts),
        "report_thinking_closed": closed,
        "report_thinking_closed_rate": round(sum(closed) / len(closed), 4),
        "report_failed": bool(failed),
        "report_seeds": [int(seed) for seed in expected_report_seeds],
    }
    if any(row.get(field) != value for field, value in expected.items()):
        raise RuntimeError(f"{label}: Gemma reports differ from sender-native raw tokens")


__all__ = ("validate_text_report_row",)
