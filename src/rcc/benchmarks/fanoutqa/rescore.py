"""Rebuild every banked answer from its raw token ids and score it again.

One complete (item, arm, seed) grid is walked; a row whose banked fields differ
from what its tokens give, a repeated row, or a missing cell all raise.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from rcc.run.fleet.answer_closer import answer_hit_ceiling, validate_banked_answer

Decoder = Callable[[Sequence[int]], str]
Scorer = Callable[[Any, str], Mapping[str, float]]
ThinkingClosed = Callable[[Sequence[int]], bool]
ReportAccepted = Callable[[str], bool]


def _require_int_tokens(label: str, value: object) -> list[int]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label}: sampled token ids are malformed")
    tokens = cast(list[object], value)
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens):
        raise RuntimeError(f"{label}: sampled token ids are malformed")
    return [int(cast(int, token)) for token in tokens]


def _rescore_row(
    row: Mapping[str, Any],
    *,
    label: str,
    question: Any,
    decode: Decoder,
    score: Scorer,
    visible: Callable[[list[int]], list[int]],
    thinking_closed: ThinkingClosed,
    answer_ceiling: int,
    closer_ids: Sequence[int] = (),
) -> None:
    """Raise when one row's banked fields differ from what its raw tokens give."""
    if row.get("question") != question.question:
        raise RuntimeError(f"{label}: banked question text differs from the sealed panel")
    tokens = _require_int_tokens(label, row.get("sampled_tokens"))
    finish_reason = row.get("finish_reason")
    if finish_reason not in {"stop", "length"}:
        raise RuntimeError(f"{label}: finish reason is not a registered vLLM outcome")
    # The banked reason is the head's, and an injected row is the one case where
    # a row may exceed the sampler cap: head content, the injected closer, and
    # one continuation inside the closing budget.
    injected = row.get("answer_injected")
    budget = row.get("answer_closer_budget")
    if type(injected) is not bool or isinstance(budget, bool) or not isinstance(budget, int):
        raise RuntimeError(f"{label}: answer closer-injection evidence is missing or malformed")
    validate_banked_answer(
        tokens,
        injected=injected,
        finish_reason=str(finish_reason),
        closing_budget=budget,
        answer_ceiling=answer_ceiling,
        label=label,
        closer_ids=closer_ids,
    )
    if row.get("answer_decoded_tokens") != len(tokens):
        raise RuntimeError(f"{label}: banked decoded-token count differs from the raw ids")
    if row.get("sampled_text") != decode(tokens):
        raise RuntimeError(f"{label}: banked sampled text differs from the raw ids")
    visible_ids = visible(tokens)
    visible_text = decode(visible_ids)
    if row.get("visible_answer_text") != visible_text:
        raise RuntimeError(f"{label}: banked visible answer differs from the raw ids")
    task = score(question, visible_text)
    for name in ("loose", "strict", "n_leaves"):
        if row.get(name) != task[name]:
            raise RuntimeError(f"{label}: banked {name} differs from the independent rescore")
    if row.get("loose_match") != (task["loose"] > 0.0):
        raise RuntimeError(f"{label}: banked loose_match differs from the independent rescore")
    closed = thinking_closed(tokens)
    if row.get("thinking_closed") != closed:
        raise RuntimeError(f"{label}: banked thinking flag differs from the raw ids")
    # The head's reason, not the merged length: an injected row is longer than
    # the head the sampler capped.
    if row.get("answer_hit_ceiling") != answer_hit_ceiling(str(finish_reason)):
        raise RuntimeError(f"{label}: banked ceiling flag differs from the head finish reason")


def _check_text_reports(
    row: Mapping[str, Any], label: str, report_accepted: ReportAccepted
) -> None:
    """Raise when a text row's reports do not meet the family's acceptance rule.

    A cell whose reports failed carries ``report_failed`` and the failing workers,
    which are exempt; the flag may not be set on a cell whose reports all pass.
    """
    reports = row.get("reports")
    closed = row.get("report_thinking_closed")
    workers = int(row.get("workers") or 0)
    if not isinstance(reports, list) or not isinstance(closed, list):
        raise RuntimeError(f"{label}: banked text reports are not substantive and closed")
    report_rows = cast(list[object], reports)
    closed_rows = cast(list[object], closed)
    if len(report_rows) != workers or len(closed_rows) != workers:
        raise RuntimeError(f"{label}: banked text reports are not substantive and closed")
    accepted_rows = [
        isinstance(report, str) and report_accepted(report) and closed_rows[worker] is True
        for worker, report in enumerate(report_rows)
    ]
    if not row.get("report_failed"):
        if not all(accepted_rows):
            raise RuntimeError(f"{label}: banked text reports are not substantive and closed")
        return
    failed = [worker for worker, ok in enumerate(accepted_rows) if not ok]
    declared = row.get("report_failed_workers")
    if not failed or declared != failed:
        raise RuntimeError(f"{label}: quarantine flag differs from the banked reports")


def rescore_seed_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    question_ids: Sequence[str],
    arm_channels: Mapping[str, str],
    sample_tags: Sequence[str],
    answer_ceiling: int,
    questions_by_qid: Mapping[str, Any],
    decode: Decoder,
    score: Scorer,
    visible: Callable[[list[int]], list[int]],
    thinking_closed: ThinkingClosed,
    report_accepted: ReportAccepted,
    closer_ids: Sequence[int] = (),
) -> dict[str, int]:
    """Rebuild and rescore one complete per-seed result grid.

    ``closer_ids`` is the lane's injected answer delimiter; with it, an injected
    row's first-draw length is read as the index of the first closer id.
    """
    semantic_arms = tuple(arm_channels)
    text_arms = tuple(arm for arm, channel in arm_channels.items() if channel == "text")
    expected = {
        (qid, arm, seed_index)
        for qid in question_ids
        for arm in semantic_arms
        for seed_index in range(len(sample_tags))
    }
    observed: set[tuple[str, str, int]] = set()
    text_rows = 0
    for row in rows:
        qid = str(row.get("qid") or "")
        arm = str(row.get("arm") or "")
        seed_index = row.get("seed_index")
        label = f"{qid}/{arm}/s{seed_index}"
        if isinstance(seed_index, bool) or not isinstance(seed_index, int):
            raise RuntimeError(f"{label}: seed index is malformed")
        key = (qid, arm, seed_index)
        if key not in expected:
            raise RuntimeError(f"{label}: row is outside the resolved result grid")
        if key in observed:
            raise RuntimeError(f"{label}: duplicate result row")
        observed.add(key)
        question = questions_by_qid.get(qid)
        if question is None:
            raise RuntimeError(f"{label}: sealed panel carries no question object")
        _rescore_row(
            row,
            label=label,
            question=question,
            decode=decode,
            score=score,
            visible=visible,
            thinking_closed=thinking_closed,
            answer_ceiling=answer_ceiling,
            closer_ids=closer_ids,
        )
        if arm in text_arms:
            _check_text_reports(row, label, report_accepted)
            text_rows += 1
    missing = expected - observed
    if missing:
        raise RuntimeError(f"rescore grid is incomplete: {len(missing)} cells are missing")
    return {
        "rescored_rows": len(observed),
        "rescored_text_rows": text_rows,
        "items": len(question_ids),
        "arms": len(semantic_arms),
        "seeds": len(sample_tags),
    }


__all__ = (
    "Decoder",
    "ReportAccepted",
    "Scorer",
    "ThinkingClosed",
    "rescore_seed_rows",
)
