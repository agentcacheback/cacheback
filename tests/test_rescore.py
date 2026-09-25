"""The independent raw-token rescore of banked results."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import rcc.run.rescore as rescore_module
from rcc.models.gemma.contract import STOP_IDS
from rcc.models.gemma.mechanism import visible_answer_tokens
from rcc.models.gemma.text_contract import TEXT_REPORT_SCHEMA, TEXT_SENDERS, text_sender_spec
from rcc.models.ministral.results import rescore_report_row
from rcc.models.ministral.text import registered_text_senders as ministral_text_senders
from rcc.models.qwen import QWEN_FAMILY
from rcc.run.fleet.answer_closer import (
    ANSWER_CLOSER_MAX_IDS,
    ANSWER_CLOSING_TOKEN_BUDGET,
)
from rcc.run.plan import load_and_resolve
from rcc.run.rescore import rescore_gemma_rows

_ROOT = Path(__file__).resolve().parents[1]


_COMMIT = "0" * 40


_TEXT_ARMS = ("text_primary", "text_medium", "text_small")


@dataclass(frozen=True)
class _Question:
    question: str


def _plan() -> Any:
    return load_and_resolve(
        _ROOT / "configs" / "gemma-fanoutqa-natural-dev50.toml",
        git_commit=_COMMIT,
    )


def _decode(tokens: Any) -> str:
    return " ".join(str(int(token)) for token in tokens)


class _Tokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert not add_special_tokens
        return {"input_ids": [98 if text == "<|channel>" else 99]}

    def decode(self, tokens: Any, **_kwargs: Any) -> str:
        return _decode(tokens)


class _MinistralCodec:
    def __init__(self, semantic_arm: str) -> None:
        sender = next(row for row in ministral_text_senders() if row.semantic_arm == semantic_arm)
        self.checkpoint = sender.checkpoint
        self.revision = sender.revision

    def require_verified(self) -> None:
        return None


def _ministral_report_row(plan: Any, qid: str) -> dict[str, Any]:
    return {
        "qid": qid,
        "arm": "text_primary",
        "reports": ["alpha", "beta", "gamma"],
        "report_raw_outputs": ["alpha", "beta", "gamma"],
        "report_token_ids_by_worker": [[7, 8], [7, 8], [7, 8]],
        "report_visible_token_ids_by_worker": [[7], [7], [7]],
        "report_finish_reasons": ["stop", "stop", "stop"],
        "report_seeds": list(plan.benchmark.report_seeds(qid, "s0")),
        "report_seed_tag": "s0",
        "report_prompt_sha256": ["a" * 64] * 3,
        "prepared_prompt_fingerprint": "b" * 64,
        # The published clock is the shipped draw's span; the ladder columns
        # beside it are required, never defaulted, so a row that omits them is
        # refused rather than rescored as a clean single draw.
        "report_generation_s": 1.0,
        "draws_n": 1,
        "redraw_wall_s": 0.0,
        "injected": False,
        "report_injected_by_worker": [False, False, False],
    }


def _score(question: _Question, text: str) -> dict[str, float]:
    del question, text
    return {"loose": 0.5, "strict": 0.0, "n_leaves": 2.0}


def _rows(plan: Any, questions: dict[str, _Question]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qid in plan.selected_question_ids:
        for arm in (arm.arm_id for arm in plan.arms):
            for seed_index in range(3):
                tokens = [5, 6, 7, 1]
                row: dict[str, Any] = {
                    "qid": qid,
                    "arm": arm,
                    "seed_index": seed_index,
                    "question": questions[qid].question,
                    "sampled_tokens": list(tokens),
                    "sampled_text": _decode(tokens),
                    "visible_answer_text": _decode(tokens[:-1]),
                    "answer_decoded_tokens": len(tokens),
                    "finish_reason": "stop",
                    "loose": 0.5,
                    "strict": 0.0,
                    "n_leaves": 2.0,
                    "loose_match": True,
                    "thinking_closed": True,
                    "answer_hit_ceiling": False,
                    "answer_injected": False,
                    "answer_closer_budget": ANSWER_CLOSING_TOKEN_BUDGET,
                }
                if arm in _TEXT_ARMS:
                    sender = text_sender_spec(arm)
                    report_tokens = [[98, 40 + worker, 99, 10 + worker, 1] for worker in range(3)]
                    row.update(
                        {
                            "workers": 3,
                            "semantic_arm": arm,
                            "policy": sender.policy,
                            "worker_checkpoint": sender.checkpoint,
                            "worker_revision": sender.revision,
                            "receiver_checkpoint": TEXT_SENDERS[0].checkpoint,
                            "receiver_revision": TEXT_SENDERS[0].revision,
                            "report_schema": TEXT_REPORT_SCHEMA,
                            "reports": ["10", "11", "12"],
                            "report_raw_outputs": [_decode(tokens) for tokens in report_tokens],
                            "report_token_ids_by_worker": report_tokens,
                            "report_tokens_by_worker": [5, 5, 5],
                            "report_tokens": 15,
                            "report_finish_reasons": ["stop", "stop", "stop"],
                            "report_seed_tag": "s0",
                            "report_seeds": list(plan.benchmark.report_seeds(qid, "s0")),
                            "report_thinking_closed": [True, True, True],
                            "report_thinking_closed_rate": 1.0,
                            "report_failed": False,
                        }
                    )
                rows.append(row)
    return rows


def _rescore(
    rows: list[dict[str, Any]], plan: Any, questions: dict[str, _Question]
) -> dict[str, int]:
    return rescore_gemma_rows(
        rows,
        plan,
        questions_by_qid=questions,
        receiver_tokenizer=_Tokenizer(),
        report_tokenizers={arm: _Tokenizer() for arm in _TEXT_ARMS},
        score=_score,
    )


@pytest.mark.live
def test_faithful_rows_pass_the_full_grid_rescore() -> None:
    plan = _plan()
    assert len(plan.selected_question_ids) == 50
    questions = {qid: _Question(f"Q-{qid}") for qid in plan.selected_question_ids}
    faithful = _rows(plan, questions)
    summary = _rescore(faithful, plan, questions)
    assert summary == {
        "rescored_rows": 50 * 12 * 3,
        "rescored_text_rows": 50 * 3 * 3,
        "items": 50,
        "arms": 12,
        "seeds": 3,
    }

    # The boundary a raw-id ceiling check refused: a worker that ran to the
    # ceiling and then stopped banks the stop token too, so its vector is
    # exactly `ceiling` ids and its content ceiling-1. Such a row must rescore.
    ceiling = plan.benchmark.report_ceiling
    at_ceiling = [dict(row) for row in faithful]
    index = next(i for i, row in enumerate(at_ceiling) if row["arm"] in _TEXT_ARMS)
    token_rows = [[98, 40 + worker, 99, *([10 + worker] * (ceiling - 4)), 1] for worker in range(3)]
    at_ceiling[index] = {
        **at_ceiling[index],
        "report_token_ids_by_worker": token_rows,
        "report_raw_outputs": [_decode(tokens) for tokens in token_rows],
        "reports": [_decode([10 + worker] * (ceiling - 4)) for worker in range(3)],
        "report_tokens_by_worker": [ceiling] * 3,
        "report_tokens": 3 * ceiling,
    }
    assert _rescore(at_ceiling, plan, questions) == summary
    # Both directions of the biconditional, on content counts: a length finish
    # is exactly the ceiling and a stop finish is below it. An injected worker
    # banks one draw, so its bound adds the lane's closing extra.
    rescore_module.validate_report_ceiling(
        [ceiling + 10, ceiling - 1],
        ["length", "stop"],
        ceiling=ceiling,
        family="Qwen",
        label="injected",
        injected=[True, False],
        closing_extra=4097,
    )
    with pytest.raises(RuntimeError, match="exceeds the registered report ceiling"):
        rescore_module.validate_report_ceiling(
            [ceiling + 10], ["stop"], ceiling=ceiling, family="Qwen", label="natural-over"
        )
    with pytest.raises(RuntimeError, match="did not fill the ceiling"):
        rescore_module.validate_report_ceiling(
            [ceiling - 5],
            ["length"],
            ceiling=ceiling,
            family="Qwen",
            label="short-length",
            injected=[True],
            closing_extra=4097,
        )
    rescore_module.validate_report_ceiling(
        [ceiling - 1, ceiling, 0],
        ["stop", "length", "stop"],
        ceiling=ceiling,
        family="Gemma",
        label="probe",
    )

    # The answer side of the same protocol: an injected answer banks head,
    # closer and continuation as one draw, and the rescore reads the shipped
    # visible answer out of those merged ids.
    injected_ids = [98, 5, 99, 6, 1]
    visible = visible_answer_tokens(
        injected_ids,
        channel_open=98,
        channel_close=99,
        stop_ids=STOP_IDS,
    )
    assert list(visible) == [6]
    continued = [dict(row) for row in faithful]
    continued[5] = {
        **continued[5],
        "sampled_tokens": injected_ids,
        "sampled_text": _decode(injected_ids),
        "visible_answer_text": _decode(visible),
        "answer_decoded_tokens": len(injected_ids),
        "answer_injected": True,
    }
    assert _rescore(continued, plan, questions) == summary


@pytest.mark.live
def test_each_banked_answer_or_sender_report_divergence_is_refused() -> None:
    plan = _plan()
    questions = {qid: _Question(f"Q-{qid}") for qid in plan.selected_question_ids}
    faithful = _rows(plan, questions)
    for field, value, message in (
        ("visible_answer_text", "tampered", "banked visible answer differs"),
        ("loose", 0.9, "banked loose differs"),
        ("thinking_closed", False, "banked thinking flag differs"),
        ("answer_hit_ceiling", True, "banked ceiling flag differs"),
        ("sampled_text", "tampered", "banked sampled text differs"),
        ("answer_decoded_tokens", 99, "banked decoded-token count differs"),
    ):
        rows = copy.deepcopy(faithful)
        rows[5][field] = value
        with pytest.raises(RuntimeError, match=message):
            _rescore(rows, plan, questions)

    # The injection flag is banked at issuance: a row that lost it, malformed
    # it, or claims an unregistered closing budget is refused, and the flag
    # admits exactly one continuation of the closing budget.
    answer_ceiling = plan.benchmark.answer_ceiling
    over_natural = [7] * (answer_ceiling + 1)
    over_injected = [7] * (answer_ceiling + ANSWER_CLOSING_TOKEN_BUDGET + ANSWER_CLOSER_MAX_IDS + 1)
    for patch, message in (
        ({"answer_injected": None}, "closer-injection evidence is missing or malformed"),
        ({"answer_injected": "yes"}, "closer-injection evidence is missing or malformed"),
        ({"answer_closer_budget": 8192, "answer_injected": True}, "closing budget is unregistered"),
        ({"sampled_tokens": over_natural}, "longer than its registered decode allows"),
        # An over-long injected row: the closer sits after one head id, so the
        # tail is what the audit refuses.
        (
            {"sampled_tokens": [98, 99, *over_injected], "answer_injected": True},
            "continuation exceeds its closing budget",
        ),
        # A "length" head filled the ceiling, natural or injected: a short row
        # carrying that reason is refused before its ceiling flag is even read.
        ({"finish_reason": "length"}, "did not fill the ceiling"),
        # An injected row must carry the lane's closer; its head is the span
        # before the first closer id, and its tail at most one closing budget.
        ({"answer_injected": True}, "carries no closer id"),
        (
            {
                "finish_reason": "length",
                "answer_injected": True,
                "sampled_tokens": [98, 5, 99, 6, 1],
            },
            "did not fill the ceiling",
        ),
        (
            {"answer_injected": True, "sampled_tokens": [98, 5, 99, *([6] * 4_097), 1]},
            "continuation exceeds its closing budget",
        ),
    ):
        rows = copy.deepcopy(faithful)
        rows[5].update(patch)
        with pytest.raises(RuntimeError, match=message):
            _rescore(rows, plan, questions)

    text_index = next(index for index, row in enumerate(faithful) if row["arm"] in _TEXT_ARMS)
    ceiling = plan.benchmark.report_ceiling
    for field, value, message in (
        ("reports", ["tampered", "11", "12"], "differ from sender-native raw tokens"),
        (
            "report_token_ids_by_worker",
            [[10] * (ceiling + 1), [98, 41, 99, 11, 1], [98, 42, 99, 12, 1]],
            "exceeds the registered report ceiling",
        ),
        (
            "report_finish_reasons",
            ["length", "stop", "stop"],
            "length finish differs from the report ceiling",
        ),
        (
            "report_finish_reasons",
            ["truncated", "stop", "stop"],
            "Gemma report finish reason is unregistered",
        ),
        # The reverse direction: content that reaches the ceiling with no stop
        # token banked cannot honestly be reported as a stop finish, because
        # the stop token itself counts against the same max_tokens cap.
        (
            "report_token_ids_by_worker",
            [
                [98, 40, 99, *([10] * (ceiling - 3))],
                [98, 41, 99, 11, 1],
                [98, 42, 99, 12, 1],
            ],
            "Gemma length finish differs from the report ceiling",
        ),
        (
            "report_raw_outputs",
            ["tampered", "98 41 99 11 1", "98 42 99 12 1"],
            "differ from sender-native raw tokens",
        ),
        (
            "report_token_ids_by_worker",
            [[98, 40, 99, 77, 1], [98, 41, 99, 11, 1], [98, 42, 99, 12, 1]],
            "differ from sender-native raw tokens",
        ),
        ("worker_checkpoint", "google/not-gemma", "sender identity differs"),
    ):
        rows = copy.deepcopy(faithful)
        rows[text_index][field] = value
        with pytest.raises(RuntimeError, match=message):
            _rescore(rows, plan, questions)

    qid = plan.selected_question_ids[0]
    qwen_plan = load_and_resolve(
        _ROOT / "configs" / "qwen-fanoutqa-natural-dev50.toml", git_commit=_COMMIT
    )
    assert qwen_plan.benchmark.report_ceiling == ceiling
    for token_rows, reasons, message in (
        ([[1] * (ceiling + 1)] * 3, ["stop"] * 3, "Qwen report exceeds the registered report"),
        ([[5, 6, 1]] * 3, ["length", "stop", "stop"], "Qwen length finish differs"),
    ):
        with pytest.raises(RuntimeError, match=message):
            rescore_module._require_qwen_text_report(
                {
                    "report_token_ids_by_worker": token_rows,
                    "report_finish_reasons": reasons,
                    "report_seed_tag": "s0",
                },
                qwen_plan,
                qid=qid,
                arm="text_primary",
                tokenizer=_Tokenizer(),
                family=QWEN_FAMILY,
            )

    codec: Any = _MinistralCodec("text_primary")
    for token_rows, reasons, message in (
        ([[1] * (ceiling + 1)] * 3, ["stop"] * 3, "Ministral report exceeds the registered report"),
        ([[7, 8]] * 3, ["length", "stop", "stop"], "Ministral length finish differs"),
    ):
        ministral_row = _ministral_report_row(plan, qid)
        ministral_row["report_token_ids_by_worker"] = token_rows
        ministral_row["report_finish_reasons"] = reasons
        with pytest.raises(RuntimeError, match=message):
            rescore_report_row(ministral_row, codec)


@pytest.mark.live
def test_missing_cells_duplicates_and_hollow_reports_are_refused() -> None:
    plan = _plan()
    questions = {qid: _Question(f"Q-{qid}") for qid in plan.selected_question_ids}
    rows = _rows(plan, questions)
    with pytest.raises(RuntimeError, match="incomplete"):
        _rescore(rows[:-1], plan, questions)
    with pytest.raises(RuntimeError, match="duplicate result row"):
        _rescore([*rows, dict(rows[0])], plan, questions)
    hollow = [dict(row) for row in rows]
    for row in hollow:
        if row["arm"] == "text_primary":
            row["reports"] = ["<pad>", "beta", "gamma"]
    with pytest.raises(RuntimeError, match="substantive and closed"):
        _rescore(hollow, plan, questions)
