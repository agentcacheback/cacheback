"""CPU coverage for the Gemma three-sender text path."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest

import rcc.models.gemma.text_closer as text_closer_module
from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.models.gemma import text_reports
from rcc.models.gemma.rescore import validate_text_report_row
from rcc.models.gemma.text_bank import TextReportBank
from rcc.models.gemma.text_closer import GEMMA_CLOSING_TOKEN_BUDGET
from rcc.models.gemma.text_codec import (
    GEMMA_CHANNEL_CLOSE,
    GEMMA_CHANNEL_OPEN,
    GEMMA_STOP_TOKEN_IDS,
    GEMMA_THINKING_ON_PREFIX,
    GEMMA_THINKING_ON_SUFFIX,
    prompt_record,
    report_is_substantive,
)
from rcc.models.gemma.text_contract import (
    TEXT_REGISTRATION_FINGERPRINT,
    TEXT_REPORT_SCHEMA,
    registered_text_senders,
)
from rcc.models.gemma.text_reports import generate_report_bundle
from rcc.models.gemma.text_runtime import (
    bind_text_prompt_item,
    generate_resident_primary_reports,
)

_QID = "95de313fdfef9d01"
_SOURCE_IDENTITY = "source-manifest-fingerprint"
_PREPARED_SHA256 = "a" * 64
_PREPARED_MANIFEST = {
    "artifact_sha256": _PREPARED_SHA256,
    "source_manifest": {"fingerprint": _SOURCE_IDENTITY},
}
_SEEDS = {
    "s0": (290_991_591, 702_827_138, 995_222_892),
    "s1": (1_717_509_489, 1_591_966_228, 1_280_759_290),
    "s2": (2_136_362_187, 1_206_528_942, 1_432_323_136),
}


class _Tokenizer:
    all_special_ids: ClassVar[list[int]] = [1, 2, 50, 90, 91, 98, 105, 106, 107]

    def __call__(self, text: str, **_kwargs: Any) -> dict[str, list[int]]:
        special = {"<suppressed-258883>": 258_883, "<suppressed-258882>": 258_882}
        if text in special:
            return {"input_ids": [special[text]]}
        if text == "@@CONTENT@@":
            return {"input_ids": [700]}
        if text == GEMMA_CHANNEL_OPEN:
            return {"input_ids": [90]}
        if text == GEMMA_CHANNEL_CLOSE:
            return {"input_ids": [91]}
        if text.startswith("rendered:"):
            middle = [777] * (
                50_000 - len(GEMMA_THINKING_ON_PREFIX) - len(GEMMA_THINKING_ON_SUFFIX)
            )
            return {
                "input_ids": [
                    *GEMMA_THINKING_ON_PREFIX,
                    *middle,
                    *GEMMA_THINKING_ON_SUFFIX,
                ]
            }
        return {"input_ids": [701]}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        **_kwargs: Any,
    ) -> str | list[int]:
        content = messages[0]["content"]
        if tokenize:
            return [*GEMMA_THINKING_ON_PREFIX, 700, *GEMMA_THINKING_ON_SUFFIX]
        return f"rendered:{content}"

    @staticmethod
    def decode(ids: Any, **_kwargs: Any) -> str:
        values = tuple(int(value) for value in ids)
        if values in {(258_883,), (258_882,)}:
            return f"<suppressed-{values[0]}>"
        words = {40: "river", 41: "evidence", 42: "answer", 200: "private thought"}
        return " ".join(words.get(value, f"t{value}") for value in values)


class _Completion:
    """A backend row whose text is the detokenization of its own ids.

    The rescore validator rebuilds `report_raw_outputs` from the banked ids,
    so text unrelated to its ids could not round-trip through it.
    """

    def __init__(self, token_ids: tuple[int, ...], cached: int, reason: str = "stop") -> None:
        self.text = _Tokenizer.decode(token_ids)
        self.n_tokens = len(token_ids)
        self.token_ids = token_ids
        self.finish_reason = reason
        self.num_cached_tokens = cached
        self.queued_ts = 1.0
        self.scheduled_ts = 1.25
        self.first_token_ts = 1.5


# Worker shapes through the registered codec: 90 opens the channel, 91 closes it, 106
# ends the turn, and only ids outside both are visible; `starved` leaves no room under
# the report ceiling for the registered closing budget.
_CLEAN = (90, 200, 91, 91, 40, 106)
_HOLLOW = (90, 200, 91, 106)
_UNCLOSED = (40, 90, 200, 106)
_STARVED = (
    90,
    *([200] * (FANOUTQA_NATURAL_DEV50.report_ceiling - GEMMA_CLOSING_TOKEN_BUDGET)),
    106,
)
_TAG_FREE = (40, 106)
_CONTINUATION = (42, 106)


class _Engine:
    """A stub sender whose worker 1 takes the named shape on each draw in turn.

    A draw past the end of `flaws` is clean, so `_Engine()` is the healthy case
    and `_Engine("hollow", "hollow", "hollow")` exhausts the sealed ladder.
    """

    def __init__(
        self,
        *flaws: str,
        cached: int = 0,
        tag_free: bool = False,
        closer_reason: str = "stop",
    ) -> None:
        self.flaws = flaws
        self.cached = cached
        self.tag_free = tag_free
        self.closer_reason = closer_reason
        self.clock: _Clock | None = None
        self.calls: list[tuple[Any, tuple[int, ...]]] = []
        self.continuations: list[tuple[Any, tuple[int, ...], tuple[int, ...]]] = []

    def decode_token_ids_full(
        self,
        prompts: Any,
        sampling: Any,
        *,
        seeds: Any,
    ) -> list[_Completion]:
        if sampling.max_tokens == GEMMA_CLOSING_TOKEN_BUDGET:
            continued = list(prompts)
            # The continuation spends wall clock of its own, so taking a stamp
            # here is what lets the published span fail when the continuation
            # is charged outside the draw it belongs to.
            if self.clock is not None:
                self.clock.perf_counter()
            self.continuations.append(
                (sampling, tuple(seeds), tuple(len(row) for row in continued))
            )
            return [_Completion(_CONTINUATION, self.cached, self.closer_reason) for _ in continued]
        assert len(prompts) == 3
        self.calls.append((sampling, tuple(seeds)))
        flaw = self.flaws[len(self.calls) - 1] if len(self.calls) <= len(self.flaws) else ""
        shapes = {"hollow": _HOLLOW, "unclosed": _UNCLOSED, "starved": _STARVED}
        rows: list[_Completion] = []
        for worker in range(3):
            tokens = (*_CLEAN[:4], 40 + worker, _CLEAN[-1])
            if self.tag_free:
                tokens = _TAG_FREE
            elif worker == 1 and flaw:
                tokens = shapes[flaw]
            rows.append(_Completion(tokens, self.cached))
        return rows


def _panel_item(tokenizer: _Tokenizer) -> dict[str, Any]:
    """Return the item shape the panel loader builds, with nothing bound yet.

    ``memory_ids`` are empty: the prepared prompt is the only place the worker
    prompt can come from, so a codec that re-rendered would fail.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "report"}],
        tokenize=False,
    )
    prompt = tokenizer(cast(str, rendered))["input_ids"]
    return {
        "qid": _QID,
        "question": "Which river?",
        "memory_ids": [[], [], []],
        "prompt_ids": [prompt, prompt, prompt],
    }


def _bound_item(tokenizer: _Tokenizer, sender: Any) -> dict[str, Any]:
    """Bind one panel item through the exact binder a live arm process calls."""
    return bind_text_prompt_item(
        tokenizer,
        _panel_item(tokenizer),
        sender,
        prepared_manifest=_PREPARED_MANIFEST,
    )


def _assert_prepared_prompts_are_the_bound_prompts(tokenizer: _Tokenizer, sender: Any) -> None:
    """A raw panel item is refused, and binding it round-trips the prepared ids."""
    panel = _panel_item(tokenizer)
    assert not {"source_identity", "text_prompt_artifacts", "text_prepared_sha256_by_arm"} & set(
        panel
    )
    with pytest.raises(RuntimeError, match="prompt artifact mismatch"):
        prompt_record(tokenizer, panel, sender)

    bound = bind_text_prompt_item(
        tokenizer,
        panel,
        sender,
        prepared_manifest=_PREPARED_MANIFEST,
    )
    record = prompt_record(tokenizer, bound, sender)
    expected = tuple(tuple(row) for row in panel["prompt_ids"])
    assert record.prompt_ids == expected
    assert record.source_identity == _SOURCE_IDENTITY
    assert record.sender_prepared_sha256 == _PREPARED_SHA256
    assert bound["text_prepared_sha256_by_arm"] == {sender.semantic_arm: _PREPARED_SHA256}

    # The geometry is read off the prepared prompt, never rebuilt from tokens.
    short = {**panel, "prompt_ids": [row[:-1] for row in panel["prompt_ids"]]}
    with pytest.raises(RuntimeError, match="registered thinking geometry"):
        bind_text_prompt_item(
            tokenizer,
            short,
            sender,
            prepared_manifest=_PREPARED_MANIFEST,
        )


class _Clock:
    """A deterministic perf counter: two stamps per draw, in order."""

    def __init__(self, stamps: Sequence[float]) -> None:
        self.stamps = list(stamps)

    def perf_counter(self) -> float:
        return self.stamps.pop(0)


@contextmanager
def _clock(*stamps: float) -> Iterator[_Clock]:
    """Drive the report ladder on an exact clock, so spans are assertable."""
    fake = _Clock(stamps)
    real = text_reports.time
    text_reports.time = cast(Any, fake)
    try:
        yield fake
    finally:
        text_reports.time = real


def _rescore_row(bundle: dict[str, Any], sender: Any) -> dict[str, Any]:
    """Shape one banked bundle as the result row the independent audit reads."""
    carried = (
        "reports",
        "report_raw_outputs",
        "report_token_ids_by_worker",
        "report_tokens_by_worker",
        "report_tokens",
        "report_finish_reasons",
        "report_thinking_closed",
        "report_thinking_closed_rate",
        "report_failed",
        "report_seeds",
    )
    return {
        "arm": sender.semantic_arm,
        "semantic_arm": sender.semantic_arm,
        "policy": sender.policy,
        "worker_checkpoint": sender.checkpoint,
        "worker_revision": sender.revision,
        "receiver_checkpoint": sender.receiver_checkpoint,
        "receiver_revision": sender.receiver_revision,
        "report_schema": TEXT_REPORT_SCHEMA,
        "workers": 3,
        **{name: bundle[name] for name in carried},
    }


def _assert_rescores(bundle: dict[str, Any], tokenizer: _Tokenizer, sender: Any) -> None:
    """Round-trip one banked bundle through the real independent validator."""
    validate_text_report_row(
        _rescore_row(bundle, sender),
        qid=_QID,
        semantic_arm=sender.semantic_arm,
        tokenizer=tokenizer,
        expected_report_seeds=bundle["report_seeds"],
    )


def _bank(root: Path, arm: str, suffix: str = "") -> TextReportBank:
    return TextReportBank(
        root / f"{arm}{suffix}.jsonl",
        attempt_id="attempt-1",
        execution_identity={"arm": arm},
    )


def assert_gemma_text_contract(root: Path) -> None:
    """Exercise all senders, exact vLLM calls, binding, acceptance, and replay."""
    tokenizer = _Tokenizer()
    senders = registered_text_senders()
    assert TEXT_REGISTRATION_FINGERPRINT == (
        "a12b67c0ae20a6b0a4ae9238e650c89c19b9c641a9b69b38035e5989e1b7d888"
    )
    assert [(row.semantic_arm, row.policy, row.native_max_model_len) for row in senders] == [
        ("text_primary", "gemma4_12b_text", 262_144),
        ("text_medium", "gemma4_12b_calls_gemma4_e4b_text", 131_072),
        ("text_small", "gemma4_12b_calls_gemma4_e2b_text", 131_072),
    ]
    assert [row.revision for row in senders] == [
        "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    ]
    assert {row.receiver_checkpoint for row in senders} == {"google/gemma-4-12B-it"}

    for sender_index, sender in enumerate(senders):
        _assert_prepared_prompts_are_the_bound_prompts(tokenizer, sender)
        item = _bound_item(tokenizer, sender)
        # The first sender's opening draw banks a closed but empty report, so
        # the sealed ladder redraws it. That is the case the frozen timing rule
        # is about: two draws, one published span.
        redrawn = sender_index == 0
        engine = _Engine("hollow") if redrawn else _Engine()
        bank = _bank(root, sender.semantic_arm)
        stamps = (0.0, 3.0, 10.0, 15.0) if redrawn else (0.0, 2.0)
        with _clock(*stamps):
            bundle = generate_report_bundle(cast(Any, engine), tokenizer, item, sender, bank)
        expected_tag = "s1" if redrawn else "s0"
        assert engine.calls[-1][1] == _SEEDS[expected_tag]
        sampling = engine.calls[-1][0]
        assert (sampling.temperature, sampling.top_p, sampling.top_k) == (1.0, 0.95, 64)
        assert sampling.seed is None
        assert sampling.stop_token_ids == tuple(sorted(GEMMA_STOP_TOKEN_IDS)) == (1, 50, 106)
        assert sampling.bad_words == (
            ("<suppressed-258883>", "<suppressed-258882>")
            if sender.semantic_arm == "text_primary"
            else ()
        )
        assert bundle["schema"] == TEXT_REPORT_SCHEMA
        assert bundle["report_seed_tag"] == expected_tag
        assert bundle["report_seeds"] == list(_SEEDS[expected_tag])
        assert bundle["worker_prompt_tokens"] == [50_000, 50_000, 50_000]
        assert bundle["report_thinking_closed"] == [True, True, True]
        assert bundle["reports"] == ["river", "evidence", "answer"]
        assert bundle["report_tokens"] == 18
        assert bundle["report_visible_tokens_by_worker"] == [1, 1, 1]
        assert bundle["report_finish_reasons"] == ["stop", "stop", "stop"]
        # The published span is the shipped draw alone, and the draw that did
        # not ship is banked beside it rather than summed into it.
        assert bundle["draws_n"] == (2 if redrawn else 1)
        assert bundle["report_generation_s"] == (5.0 if redrawn else 2.0)
        assert bundle["redraw_wall_s"] == (3.0 if redrawn else 0.0)
        assert engine.continuations == [], "a closed draw is never continued"
        assert bundle["injected"] is False
        assert bundle["report_injected_by_worker"] == [False, False, False]
        _assert_rescores(bundle, tokenizer, sender)
        replay = _Engine()
        assert generate_report_bundle(cast(Any, replay), tokenizer, item, sender, bank) == bundle
        assert replay.calls == [], "complete durable report banks must not decode again"

    primary = senders[0]
    # The fused route hands the resident entry point the raw panel item and
    # lets it bind text_primary itself, so that exact shape is exercised here.
    resident = generate_resident_primary_reports(
        cast(Any, _Engine()),
        tokenizer,
        [_panel_item(tokenizer)],
        _bank(root, primary.semantic_arm, "-resident"),
        prepared_manifest=_PREPARED_MANIFEST,
    )
    assert set(resident) == {_QID}
    assert resident[_QID]["semantic_arm"] == "text_primary"
    assert resident[_QID]["source_identity"] == _SOURCE_IDENTITY
    assert resident[_QID]["sender_prepared_sha256"] == _PREPARED_SHA256

    with pytest.raises(RuntimeError, match="prefix caching contaminated"):
        generate_report_bundle(
            cast(Any, _Engine(cached=1)),
            tokenizer,
            _bound_item(tokenizer, primary),
            primary,
            _bank(root, primary.semantic_arm, "-cached"),
        )
    # The report ceiling bounds the head and the closing budget bounds the
    # continuation, so worker 1, which spent most of the ceiling thinking, is
    # continued once and its cell ships.
    starved = _Engine("starved")
    with _clock(0.0, 1.0, 5.0, 7.0):
        continued = generate_report_bundle(
            cast(Any, starved),
            tokenizer,
            _bound_item(tokenizer, primary),
            primary,
            _bank(root, primary.semantic_arm, "-starved"),
        )
    assert len(starved.calls) == 1, "a continued head does not redraw"
    assert len(starved.continuations) == 1, "the starved head is continued exactly once"
    assert continued["report_failed"] is False
    assert continued["report_injected_by_worker"] == [False, True, False]
    # Draw exhaustion quarantines the cell instead of killing the fleet. With
    # a closing budget wider than any engine context the closer is unaffordable,
    # so the sealed ladder, not an exception, disposes of the cell.
    saved_budget = text_closer_module.GEMMA_CLOSING_TOKEN_BUDGET
    text_closer_module.GEMMA_CLOSING_TOKEN_BUDGET = 10**9
    try:
        exhausted = _Engine("starved", "starved", "starved")
        with _clock(0.0, 1.0, 5.0, 7.0, 11.0, 14.0):
            flagged = generate_report_bundle(
                cast(Any, exhausted),
                tokenizer,
                _bound_item(tokenizer, primary),
                primary,
                _bank(root, primary.semantic_arm, "-exhausted"),
            )
    finally:
        text_closer_module.GEMMA_CLOSING_TOKEN_BUDGET = saved_budget
    assert len(exhausted.calls) == 3, "every sealed draw runs before quarantine"
    assert exhausted.continuations == [], "an unaffordable closer redraws, never raises"
    assert flagged["report_failed"] is True
    assert flagged["report_failed_workers"] == [1]
    assert flagged["report_thinking_closed"] == [True, False, True]
    assert flagged["report_thinking_closed_rate"] == round(2 / 3, 4)
    assert flagged["injected"] is False
    assert (flagged["draws_n"], flagged["report_generation_s"]) == (3, 3.0)
    assert flagged["redraw_wall_s"] == pytest.approx(3.0)
    assert "after 3 draws" in flagged["report_failure"]
    replay_flagged = _Engine("starved", "starved", "starved")
    assert (
        generate_report_bundle(
            cast(Any, replay_flagged),
            tokenizer,
            _bound_item(tokenizer, primary),
            primary,
            _bank(root, primary.semantic_arm, "-exhausted"),
        )
        == flagged
    )
    assert replay_flagged.calls == [], "quarantined durable banks must not decode again"
    _assert_closer_injection(root, tokenizer, primary)
    _assert_closer_is_inert_without_tags(root, tokenizer, primary)
    assert report_is_substantive("river before an unclosed channel")
    assert not report_is_substantive("<|turn><tool_call|>")


def _assert_closer_injection(root: Path, tokenizer: _Tokenizer, primary: Any) -> None:
    """The registered closer continues one draw once, inside that draw's clock."""
    engine = _Engine("unclosed")
    # Three stamps: the draw opens at 0.0, the continuation spends until 2.0,
    # and the draw closes at 4.0. A published span that stopped the clock
    # before the continuation would read 2.0 and fail the assertion below.
    with _clock(0.0, 2.0, 4.0) as clock:
        engine.clock = clock
        bundle = generate_report_bundle(
            cast(Any, engine),
            tokenizer,
            _bound_item(tokenizer, primary),
            primary,
            _bank(root, primary.semantic_arm, "-injected"),
        )
    assert len(engine.calls) == 1, "the continued draw ships instead of being redrawn"
    sampling, seeds, prompt_lengths = engine.continuations[0]
    # Only the output cap moves. Seed, seed tag, and every sampling field the
    # decode fingerprint covers are the head draw's own.
    assert sampling.max_tokens == GEMMA_CLOSING_TOKEN_BUDGET == 4096
    assert (sampling.temperature, sampling.top_p, sampling.top_k) == (1.0, 0.95, 64)
    assert sampling.stop_token_ids == tuple(sorted(GEMMA_STOP_TOKEN_IDS))
    assert sampling.bad_words == engine.calls[0][0].bad_words
    assert seeds == (_SEEDS["s0"][1],)
    # Prompt, then the head's content with its stop id dropped, then the closer.
    assert prompt_lengths == (50_000 + 3 + 1,)
    # The registered budget fits inside the report ceiling, which is what keeps
    # the merged row under the audit's hard bound.
    assert FANOUTQA_NATURAL_DEV50.report_ceiling > GEMMA_CLOSING_TOKEN_BUDGET
    assert bundle["injected"] is True
    assert bundle["report_injected_by_worker"] == [False, True, False]
    assert (bundle["draws_n"], bundle["redraw_wall_s"]) == (1, 0.0)
    assert bundle["report_generation_s"] == 4.0, "the continuation is inside the draw's span"
    assert bundle["report_token_ids_by_worker"][1] == [40, 90, 200, 91, 42, 106]
    assert bundle["report_thinking_closed"] == [True, True, True]
    assert bundle["reports"][1] == "river answer"
    assert bundle["report_failed"] is False
    _assert_rescores(bundle, tokenizer, primary)
    replay = _Engine("unclosed")
    assert (
        generate_report_bundle(
            cast(Any, replay),
            tokenizer,
            _bound_item(tokenizer, primary),
            primary,
            _bank(root, primary.semantic_arm, "-injected"),
        )
        == bundle
    )
    assert replay.calls == [] and replay.continuations == []
    _assert_truncated_continuation_banks_the_head_reason(root, tokenizer, primary)


def _assert_truncated_continuation_banks_the_head_reason(
    root: Path, tokenizer: _Tokenizer, primary: Any
) -> None:
    """A continuation cut off by the closing budget still banks the head's reason.

    It ends on "length" below the report ceiling, and `validate_report_ceiling` reads "length"
    as a claim that the content filled it, so the head's reason is the one describing the row.
    """
    engine = _Engine("unclosed", closer_reason="length")
    with _clock(0.0, 2.0, 4.0) as clock:
        engine.clock = clock
        bundle = generate_report_bundle(
            cast(Any, engine),
            tokenizer,
            _bound_item(tokenizer, primary),
            primary,
            _bank(root, primary.semantic_arm, "-truncated"),
        )
    assert len(engine.continuations) == 1
    assert bundle["report_injected_by_worker"] == [False, True, False]
    assert bundle["report_finish_reasons"] == ["stop", "stop", "stop"]
    assert bundle["report_generation_s"] == 4.0
    _assert_rescores(bundle, tokenizer, primary)


def _assert_closer_is_inert_without_tags(root: Path, tokenizer: _Tokenizer, primary: Any) -> None:
    """A report carrying no reasoning tags never reaches the closer at all.

    Gemma's trigger is the open delimiter, so a draw that never opened a
    channel costs one membership test and banks `injected` false everywhere.
    """
    engine = _Engine(tag_free=True)
    with _clock(0.0, 1.0, 5.0, 7.0, 11.0, 14.0):
        bundle = generate_report_bundle(
            cast(Any, engine),
            tokenizer,
            _bound_item(tokenizer, primary),
            primary,
            _bank(root, primary.semantic_arm, "-tagfree"),
        )
    assert engine.continuations == [], "a tag-free report can never trigger the closer"
    assert len(engine.calls) == 3
    assert bundle["injected"] is False
    assert bundle["report_injected_by_worker"] == [False, False, False]
    assert bundle["report_thinking_closed"] == [False, False, False]
    assert bundle["report_failed"] is True
    assert (bundle["draws_n"], bundle["report_generation_s"]) == (3, 3.0)
    assert bundle["redraw_wall_s"] == pytest.approx(3.0)
    _assert_rescores(bundle, tokenizer, primary)


__all__ = ("assert_gemma_text_contract",)
