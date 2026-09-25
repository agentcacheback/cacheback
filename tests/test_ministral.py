"""The Ministral split runtime: closer injection, the durable handoff, the adapter, the roles."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from tests.ministral_split_support import (
    _EMPTY_CONSTRUCTION_AUDIT,
    ANSWER_TOKENS,
    INJECTED_TOKENS,
    PRE_FIX_TOKENS,
    QIDS,
    SUPPORT_ARM,
    TEXT_ARM,
    Bank,
    _Engine,
    _Inner,
    bundle_from_raw,
    codec,
    drain_one,
    fixture_adapter,
    fixture_prompt_records,
    fixture_reports,
    fleet_config,
    items,
    producer,
    receiver,
    report_bundle,
    text_producer,
    word_codec,
)
from tests.natural_bundle import _one_item_config, bank_result_rows, rebanker, write_bundle

import rcc.models.ministral.text as ministral_text
import rcc.models.ministral.text_codec as ministral_text_codec
from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50, ministral_construction, ministral_data
from rcc.benchmarks.fanoutqa.ministral_construction import (
    parse_ministral_construction_manifest,
    validate_ministral_result_construction,
)
from rcc.benchmarks.fanoutqa.natural_panel import natural_worker_texts
from rcc.benchmarks.fanoutqa.panel import load_questions
from rcc.hardware.fleet import FleetPlacement
from rcc.hardware.placements import SEMANTIC_ARM_PLACEMENTS, bind_policy_placements, placement_table
from rcc.models.ministral import MINISTRAL, MINISTRAL_RUNTIME, ministral_semantic_bindings
from rcc.models.ministral.engine import MinistralEngineConfig
from rcc.models.ministral.engine_roles import (
    CAPTURE_ENGINE_IDENTITY,
    MINISTRAL_ENGINE_MAX_NUM_SEQS,
    PRODUCER_ROLE,
    RECEIVER_ROLE,
    SPLIT_RECEIVER_ENGINE_IDENTITY,
    registered_identity,
    role_engine_config,
    verify_engine_identity,
)
from rcc.models.ministral.handoff import (
    MINISTRAL_HANDOFF_ROWS,
    MINISTRAL_HANDOFF_SCHEMA,
    MinistralHandoffError,
    handoff_fingerprint,
    read_handoff,
    read_handoff_clocks,
    read_handoff_manifest,
    write_handoff,
    write_handoff_clocks,
)
from rcc.models.ministral.payload import (
    MINISTRAL_PAYLOAD_LAYOUT,
    MINISTRAL_PAYLOAD_SCHEMA,
    MINISTRAL_SELECTED_ORDER,
    MinistralFlatPayload,
    latent_plan_sha256,
    selected_indices_sha256,
    tensor_content_sha256,
)
from rcc.models.ministral.results import rescore_report_row
from rcc.models.ministral.runtime import (
    CAPTURE_CONNECTOR,
    observed_runtime_signature,
    runtime_fingerprint,
    validate_runtime_signature,
)
from rcc.models.ministral.text import (
    MINISTRAL_CLOSING_TOKEN_BUDGET,
    MinistralDecodeRequest,
    generate_report_bundle,
)
from rcc.models.ministral.text_codec import (
    MINISTRAL_TEXT_ARMS,
    MINISTRAL_THINK_CLOSE_ID,
    MINISTRAL_THINK_OPEN_ID,
    MinistralTextCodec,
    prompt_token_sha256,
    stop_trimmed,
)
from rcc.models.ministral.text_prompt import MinistralPromptRecord
from rcc.run import handoff_recall as recall
from rcc.run import plan as plan_module
from rcc.run.barrier import SPLIT_FLEET_ISOLATION_PROFILE, BarrierSpec
from rcc.run.contract import BankIdentity
from rcc.run.fleet.clocks import clock_record_path, read_bundle_clocks, write_bundle_clocks
from rcc.run.fleet.contract import ProducedArtifact, WorkItem
from rcc.run.fleet.latency import ITEM_LATENCY_FIELDS
from rcc.run.fleet.merge import read_arm_bank_rows
from rcc.run.fleet.runtime import FleetRunConfig, FleetWorkerHooks, run_worker
from rcc.run.fleet.schedule import SplitScheduleConfig, run_split_fleet
from rcc.run.ministral.prepare import prepared_panel_path
from rcc.run.ministral.split_adapter import MinistralSplitFleetAdapter
from rcc.run.ministral.split_build import handoff_root
from rcc.run.ministral.split_producer import handoff_manifest_path, payload_id
from rcc.run.ministral.split_receiver import (
    DIRECT_ARM,
    FUSED_ARM,
    TIMED_SAMPLE_TAG,
    MinistralFusedReceiver,
)
from rcc.run.ministral.split_text import (
    TEXT_BUNDLE_CLOCKS,
    MinistralTextHandoffError,
    read_text_bundle,
    text_bundle_path,
    write_text_bundle,
)

# --- The closer injection and the per-draw clock.

QID = FANOUTQA_NATURAL_DEV50.question_ids[0]


_STOP = 2


#: One naturally closed report: open, thought, close, answer, end of turn.
CLOSED = (MINISTRAL_THINK_OPEN_ID, 2_000, MINISTRAL_THINK_CLOSE_ID, 3_000, _STOP)


#: The head the protocol continues: exactly one opener, no closer, and the
#: model ended its turn inside the block.
ONE_OPEN_UNCLOSED = (MINISTRAL_THINK_OPEN_ID, 2_000, 3_000, _STOP)


#: The head the protocol leaves alone: two openers, no closer. Injecting a
#: closer here yields two opens and one close, which the reader still refuses,
#: so the draw would spend the whole closing budget and redraw anyway.
TWO_OPEN_UNCLOSED = (
    MINISTRAL_THINK_OPEN_ID,
    2_000,
    MINISTRAL_THINK_OPEN_ID,
    3_000,
    _STOP,
)


#: What the continuation decodes once the closer is injected: real answer
#: content, then the end of turn the head never reached.
CONTINUATION = (3_001, _STOP)


#: The five ordinary Tekken pieces that spell the literal closer.
LITERAL_CLOSE = (1_091, 1_047, 12_001, 57_142, 1_093)


#: The ordinary pieces a receiver or sender opens the block with.
LITERAL_OPEN = (1_091, 12_001, 57_142, 1_093)


_LITERAL_PIECES = {
    MINISTRAL_THINK_OPEN_ID: "",
    2_000: "reasoning",
    1_091: "[",
    1_047: "/",
    12_001: "TH",
    57_142: "INK",
    1_093: "]",
    3_000: "preserved answer",
}


class _LiteralInner:
    """Decode the real mixed special-opener/ordinary-closer token shape."""

    @staticmethod
    def decode(ids: list[int]) -> str:
        return "".join(_LITERAL_PIECES.get(int(token), "") for token in ids)


class _LiteralTokenizer:
    instruct_tokenizer = SimpleNamespace(tokenizer=_LiteralInner())

    @staticmethod
    def decode(ids: list[int]) -> str:
        return _LiteralInner.decode(ids)


def _literal_codec() -> MinistralTextCodec:
    """Use the production parser over a deterministic exact-piece decoder."""
    registered = codec(TEXT_ARM)
    result = object.__new__(MinistralTextCodec)
    object.__setattr__(result, "tokenizer", _LiteralTokenizer())
    object.__setattr__(result, "checkpoint", registered.checkpoint)
    object.__setattr__(result, "revision", registered.revision)
    object.__setattr__(result, "require_verified", lambda: None)
    return result


class _Result:
    """One raw completion in the shape the text contract reads."""

    # Declared, not inferred: the completion protocol reads these as mutable
    # attributes, so an inferred `tuple[int, ...]` would not satisfy it.
    request_id: str
    text: str
    token_ids: Sequence[int]
    n_tokens: int
    finish_reason: str
    num_cached_tokens: int | None
    queued_ts: float | None
    scheduled_ts: float | None
    first_token_ts: float | None

    def __init__(self, request_id: str, token_ids: Sequence[int]) -> None:
        """Return a well formed completion with the stamps the bundle prices."""
        self.request_id = request_id
        self.text = "not trusted"
        self.token_ids = tuple(token_ids)
        self.n_tokens = len(self.token_ids)
        self.finish_reason = "stop"
        self.num_cached_tokens = None
        self.queued_ts = 1.0
        self.scheduled_ts = 1.1
        self.first_token_ts = 1.2


class _ScriptedTextEngine:
    """A sender scripted per draw that records every continuation it serves."""

    def __init__(self, draws: Sequence[Sequence[Sequence[int]]]) -> None:
        """Bind one per-worker id roster for each draw of the seed ladder."""
        self.draws = [tuple(draw) for draw in draws]
        self.heads: list[tuple[MinistralDecodeRequest, ...]] = []
        self.continuations: list[tuple[tuple[tuple[int, ...], ...], Any]] = []

    def decode_token_ids_full(
        self,
        prompts: Sequence[Sequence[int]],
        requests: Sequence[MinistralDecodeRequest],
    ) -> tuple[_Result, ...]:
        """Serve one scripted head draw, or one closing continuation."""
        signed = tuple(requests)
        rows = tuple(tuple(int(token) for token in prompt) for prompt in prompts)
        assert len(rows) == len(signed)
        if any(request.decode.closing for request in signed):
            assert all(request.decode.closing for request in signed)
            self.continuations.append((rows, signed))
            return tuple(_Result(request.request_id, CONTINUATION) for request in signed)
        draw = self.draws[len(self.heads)]
        self.heads.append(signed)
        return tuple(_Result(request.request_id, draw[request.member_index]) for request in signed)


def _prompt_record(worker: int, report_codec: Any) -> MinistralPromptRecord:
    """Build one worker prompt record at the panel ceiling for the fixture sender."""
    body = tuple(
        100 + (index + worker) % 300
        for index in range(FANOUTQA_NATURAL_DEV50.worker_prompt_tokens - 5)
    )
    tokens = (1, 17, 18, 3, 4, *body)
    digest = prompt_token_sha256(tokens)
    return MinistralPromptRecord(
        qid=QID,
        semantic_arm=TEXT_ARM,
        worker=worker,
        source_identity=FANOUTQA_NATURAL_DEV50.source_logical_fingerprint,
        checkpoint=report_codec.checkpoint,
        revision=report_codec.revision,
        token_ids=tokens,
        token_sha256=digest,
        prepared_token_sha256=digest,
        prepared_fingerprint="d" * 64,
    )


def _clock(monkeypatch: pytest.MonkeyPatch, stamps: Sequence[float]) -> None:
    """Drive the report module's own clock from a scripted sequence."""
    values = iter(stamps)
    monkeypatch.setattr(
        "rcc.models.ministral.text.time.perf_counter",
        lambda: next(values),
    )


def test_the_closer_injection_and_the_per_draw_clock_run_the_shipped_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive the shipped generator over a two draw ladder with one injection.

    Draw one: worker 1 opens the block twice and never closes it, which is no trigger, so the
    ladder advances. Draw two: worker 1 opens once, the registered trigger.
    """
    report_codec = codec(TEXT_ARM)
    records = [_prompt_record(worker, report_codec) for worker in range(3)]
    engine = _ScriptedTextEngine(
        (
            (CLOSED, TWO_OPEN_UNCLOSED, CLOSED),
            (CLOSED, ONE_OPEN_UNCLOSED, CLOSED),
        )
    )
    # Two draws, two reads each, at different costs: the first draw takes 10 s
    # and the shipped one 30 s.
    _clock(monkeypatch, (0.0, 10.0, 100.0, 130.0))
    bundle = generate_report_bundle(engine, report_codec, records)

    # The trigger. Exactly one continuation ran, for exactly one worker, and
    # the multi-open draw of the first round was left alone.
    assert len(engine.heads) == 2
    assert len(engine.continuations) == 1
    prompts, continued = engine.continuations[0]
    assert len(continued) == 1
    assert continued[0].member_index == 1
    assert continued[0].sample_tag == bundle.seed_tag

    # The continuation prompt replaces the end of turn: worker turn, then the
    # head's content without its stop id, then the canonical closer.
    head = ONE_OPEN_UNCLOSED
    assert prompts[0] == (
        *records[1].token_ids,
        *stop_trimmed(head),
        MINISTRAL_THINK_CLOSE_ID,
    )
    assert prompts[0][-1] == MINISTRAL_THINK_CLOSE_ID
    assert _STOP not in prompts[0][len(records[1].token_ids) :]

    # The continuation decodes under the registered closing budget, never a
    # second full report ceiling, and every other sampling field is untouched.
    assert continued[0].decode.closing is True
    assert continued[0].decode.max_tokens == MINISTRAL_CLOSING_TOKEN_BUDGET == 4_096
    assert continued[0].decode.max_tokens != FANOUTQA_NATURAL_DEV50.report_ceiling
    assert continued[0].seed == FANOUTQA_NATURAL_DEV50.report_seeds(QID, bundle.seed_tag)[1]

    # The merge. The banked ids are head content, closer, continuation, and the
    # head's stop id is gone.
    assert bundle.injected_by_worker == (False, True, False)
    assert bundle.raw_token_ids_by_worker[1] == (
        *stop_trimmed(head),
        MINISTRAL_THINK_CLOSE_ID,
        *CONTINUATION,
    )
    assert bundle.reports[1] == "3001"
    assert not bundle.report_failed
    assert bundle.result_fields()["injected"] is True

    # The published clock is the shipped draw alone; the discarded draw is
    # banked beside it and never inside it.
    assert bundle.draws_n == 2
    assert bundle.generation_s == pytest.approx(30.0)
    assert bundle.redraw_wall_s == pytest.approx(10.0)
    assert bundle.generation_s != pytest.approx(40.0)

    # The model-real mixed encoding is already closed, not a continuation head.
    report_codec = _literal_codec()
    records = [_prompt_record(worker, report_codec) for worker in range(3)]
    literal_closed = (MINISTRAL_THINK_OPEN_ID, 2_000, *LITERAL_CLOSE, 3_000, _STOP)
    engine = _ScriptedTextEngine(((literal_closed,) * 3,))
    _clock(monkeypatch, (0.0, 1.0))

    bundle = generate_report_bundle(engine, report_codec, records)

    assert engine.continuations == []
    assert bundle.injected_by_worker == (False, False, False)
    assert bundle.reports == ("preserved answer",) * 3
    assert report_codec.visible_answer(literal_closed) == ((3_000,), "preserved answer", True)
    assert report_codec.visible_answer(CLOSED) == ((3_000,), "preserved answer", True)

    # A closer with no opened block, and two closers, remain invalid.
    assert report_codec.visible_answer((2_000, *LITERAL_CLOSE, 3_000, _STOP)) == ((), "", False)
    assert report_codec.visible_answer(
        (MINISTRAL_THINK_OPEN_ID, 2_000, *LITERAL_CLOSE, *LITERAL_CLOSE, 3_000, _STOP)
    ) == ((), "", False)

    # A block opened in the ordinary spelling at the start of the draw is a
    # real block: closed in either spelling it reads, unclosed it is continued.
    # A "[THINK]" inside a reserved block is prose, not a second opener.
    literal_opened = (*LITERAL_OPEN, 2_000, *LITERAL_CLOSE, 3_000, _STOP)
    assert report_codec.visible_answer(literal_opened) == ((3_000,), "preserved answer", True)
    assert report_codec.thinking_is_unclosed((*LITERAL_OPEN, 2_000, _STOP)) is True
    assert report_codec.visible_answer(
        (*LITERAL_OPEN, 2_000, MINISTRAL_THINK_CLOSE_ID, 3_000, _STOP)
    ) == ((3_000,), "preserved answer", True)
    assert report_codec.visible_answer(
        (MINISTRAL_THINK_OPEN_ID, *LITERAL_OPEN, 2_000, *LITERAL_CLOSE, 3_000, _STOP)
    ) == ((3_000,), "preserved answer", True)


def test_an_unaffordable_or_malformed_continuation_never_banks_a_bad_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A draw that cannot afford the closer is not continued; a broken one raises.

    Affordability is quiet: a worker that spent its ceiling thinking rejoins
    the seed ladder. A malformed continuation raises instead.
    """
    report_codec = codec(TEXT_ARM)
    records = [_prompt_record(worker, report_codec) for worker in range(3)]

    # A head that ran to the ceiling is continued: the ceiling bounds the head
    # and the closing budget bounds the continuation, so the banked report may
    # exceed the ceiling by the closer plus that budget.
    starved = (MINISTRAL_THINK_OPEN_ID, *([2_000] * (FANOUTQA_NATURAL_DEV50.report_ceiling - 1)))
    engine = _ScriptedTextEngine(((CLOSED, starved, CLOSED),))
    _clock(monkeypatch, (0.0, 1.0, 2.0, 3.0))
    bundle = generate_report_bundle(engine, report_codec, records)
    assert len(engine.continuations) == 1
    assert bundle.injected_by_worker == (False, True, False)
    assert bundle.report_failed_workers == ()
    assert bundle.draws_n == 1
    assert len(bundle.raw_token_ids_by_worker[1]) > FANOUTQA_NATURAL_DEV50.report_ceiling
    assert (
        len(bundle.raw_token_ids_by_worker[1])
        <= FANOUTQA_NATURAL_DEV50.report_ceiling
        + 1
        + ministral_text.MINISTRAL_CLOSING_TOKEN_BUDGET
        + 1
    )
    # Unaffordable means the engine context: with a closing budget wider than
    # the panel window the closer is refused quietly, the ladder runs to
    # exhaustion, and the worker is quarantined rather than raising.
    monkeypatch.setattr(
        ministral_text, "MINISTRAL_CLOSING_TOKEN_BUDGET", FANOUTQA_NATURAL_DEV50.max_model_len
    )
    engine = _ScriptedTextEngine(((CLOSED, starved, CLOSED),) * 3)
    _clock(monkeypatch, (0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
    bundle = generate_report_bundle(engine, report_codec, records)
    assert engine.continuations == []
    assert bundle.injected_by_worker == (False, False, False)
    assert bundle.draws_n == len(FANOUTQA_NATURAL_DEV50.sample_tags)
    assert bundle.report_failed_workers == (1,)
    monkeypatch.undo()

    class _BrokenEngine(_ScriptedTextEngine):
        def decode_token_ids_full(
            self,
            prompts: Sequence[Sequence[int]],
            requests: Sequence[MinistralDecodeRequest],
        ) -> tuple[_Result, ...]:
            served = super().decode_token_ids_full(prompts, requests)
            if any(request.decode.closing for request in requests):
                for row in served:
                    row.finish_reason = "abort"
            return served

    broken = _BrokenEngine(((CLOSED, ONE_OPEN_UNCLOSED, CLOSED),))
    _clock(monkeypatch, (0.0, 1.0))
    with pytest.raises(RuntimeError, match="invalid finish reason"):
        generate_report_bundle(broken, report_codec, records)


# --- The durable handoff: round trip and every refusal.

ARM = "latent_query_support_r2"


QID_HANDOFF = "fanoutqa-0007"


def _blocks(rows: torch.Tensor, rows_by_worker: tuple[int, ...]) -> list[torch.Tensor]:
    blocks: list[torch.Tensor] = []
    offset = 0
    for count in rows_by_worker:
        blocks.append(rows[offset : offset + count])
        offset += count
    return blocks


def _payload(
    *,
    semantic_arm: str = ARM,
    keeps: tuple[tuple[int, ...], ...] = ((0, 1, 2), (0, 3), (0, 1, 5, 7)),
    hidden: int = 8,
) -> MinistralFlatPayload:
    rows_by_worker = tuple(len(keep) for keep in keeps)
    total = sum(rows_by_worker)
    rows = (
        torch.arange(total * hidden, dtype=torch.float32).reshape(total, hidden).to(torch.bfloat16)
    )
    blocks = _blocks(rows, rows_by_worker)
    return MinistralFlatPayload(
        rows=rows,
        semantic_arm=semantic_arm,
        latent_plan_sha256=latent_plan_sha256(semantic_arm),
        keeps=keeps,
        rows_by_worker=rows_by_worker,
        worker_sha256=tuple(tensor_content_sha256(block) for block in blocks),
        selected_indices_sha256=selected_indices_sha256(keeps),
        tensor_sha256=tensor_content_sha256(rows),
    )


def _rewrite(manifest_path: Path, mutate: Any) -> None:
    """Apply a mutation to the manifest and re-seal its fingerprint."""
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    files: dict[str, Any] = manifest["files"]
    manifest["fingerprint"] = handoff_fingerprint(
        payload_id=manifest["payload_id"],
        files={name: entry["sha256"] for name, entry in files.items()},
        meta=manifest["meta"],
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def _rows_path(manifest_path: Path) -> Path:
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest_path.parent / manifest["files"][MINISTRAL_HANDOFF_ROWS]["path"]


def _fingerprint(manifest_path: Path) -> str:
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return str(manifest["fingerprint"])


def test_handoff_round_trip_rebuilds_the_signed_payload_and_its_producer_clocks(
    tmp_path: Path,
) -> None:
    payload = _payload()
    manifest_path = write_handoff(tmp_path, payload_id=QID_HANDOFF, payload=payload)
    assert manifest_path == tmp_path / f"{QID_HANDOFF}.handoff.json"
    assert _rows_path(manifest_path).is_file()

    manifest = read_handoff_manifest(manifest_path, payload_id=QID_HANDOFF)
    assert manifest["schema"] == MINISTRAL_HANDOFF_SCHEMA
    assert manifest["meta"]["payload_schema"] == MINISTRAL_PAYLOAD_SCHEMA
    assert manifest["meta"]["payload_layout"] == MINISTRAL_PAYLOAD_LAYOUT
    assert manifest["meta"]["selected_indices_order"] == MINISTRAL_SELECTED_ORDER
    assert manifest["meta"]["rows_shape"] == [9, 8]
    assert manifest["meta"]["rows_dtype"] == "torch.bfloat16"
    assert manifest["files"][MINISTRAL_HANDOFF_ROWS]["path"] == f"{QID_HANDOFF}/rows.pt"

    restored = read_handoff(manifest_path, payload_id=QID_HANDOFF)
    assert torch.equal(restored.rows, payload.rows)
    assert restored.rows.dtype == torch.bfloat16
    assert restored.rows.device.type == "cpu"
    fields = {field.name for field in dataclasses.fields(restored)} - {"rows"}
    assert {name: getattr(restored, name) for name in sorted(fields)} == {
        name: getattr(payload, name) for name in sorted(fields)
    }

    # The producer clocks ride a clock record bound to the manifest fingerprint,
    # because writing the manifest is the very thing being measured.
    write_handoff_clocks(
        manifest_path,
        payload_id=QID_HANDOFF,
        fingerprint=_fingerprint(manifest_path),
        spill_save_s=0.25,
        producer_s=1.5,
    )
    clocks = read_handoff_clocks(
        manifest_path,
        payload_id=QID_HANDOFF,
        fingerprint=_fingerprint(manifest_path),
    )
    assert (clocks.spill_save_s, clocks.producer_s) == (0.25, 1.5)
    with pytest.raises(MinistralHandoffError, match="clock record belongs to another manifest"):
        read_handoff_clocks(manifest_path, payload_id=QID_HANDOFF, fingerprint="0" * 64)
    with pytest.raises(MinistralHandoffError, match="producer clocks must be nonnegative"):
        write_handoff_clocks(
            manifest_path,
            payload_id=QID_HANDOFF,
            fingerprint=_fingerprint(manifest_path),
            spill_save_s=0.25,
            producer_s=-1.0,
        )
    body: dict[str, Any] = json.loads(clock_record_path(manifest_path).read_text("utf-8"))
    body["producer_s"] = "fast"
    clock_record_path(manifest_path).write_text(json.dumps(body, sort_keys=True), "utf-8")
    with pytest.raises(MinistralHandoffError, match="clock record entry producer_s is malformed"):
        read_handoff_clocks(
            manifest_path,
            payload_id=QID_HANDOFF,
            fingerprint=_fingerprint(manifest_path),
        )
    clock_record_path(manifest_path).unlink()
    with pytest.raises(MinistralHandoffError, match="handoff clock record is unreadable"):
        read_handoff_clocks(
            manifest_path,
            payload_id=QID_HANDOFF,
            fingerprint=_fingerprint(manifest_path),
        )

    # The fingerprint is deterministic and path free: the same payload written
    # from two roots is byte identical, and a different payload or id is not.
    second = write_handoff(tmp_path / "producer_b", payload_id=QID_HANDOFF, payload=payload)
    assert second.read_bytes() == manifest_path.read_bytes()
    assert write_handoff(tmp_path, payload_id=QID_HANDOFF, payload=payload).read_bytes() == (
        manifest_path.read_bytes()
    )
    other = _payload(keeps=((0, 1, 2), (0, 3), (0, 1, 5, 9)))
    with pytest.raises(
        MinistralHandoffError,
        match=f"{QID_HANDOFF}: a different handoff payload already holds this id",
    ):
        write_handoff(tmp_path, payload_id=QID_HANDOFF, payload=other)
    third = write_handoff(tmp_path / "producer_c", payload_id=QID_HANDOFF, payload=other)
    renamed = write_handoff(tmp_path / "producer_d", payload_id="fanoutqa-0008", payload=payload)
    assert len({_fingerprint(path) for path in (manifest_path, third, renamed)}) == 3


def test_handoff_refuses_every_torn_tampered_or_drifted_publication(tmp_path: Path) -> None:
    payload = _payload()
    other = _payload(keeps=((0, 1), (0, 3), (0, 1, 5, 7)))
    assert other.tensor_sha256 != payload.tensor_sha256

    flipped = write_handoff(tmp_path / "flipped", payload_id=QID_HANDOFF, payload=payload)
    rows_path = _rows_path(flipped)
    raw = bytearray(rows_path.read_bytes())
    raw[-1] ^= 0xFF
    rows_path.write_bytes(bytes(raw))
    with pytest.raises(MinistralHandoffError, match="handoff file digest differs"):
        read_handoff(flipped, payload_id=QID_HANDOFF)

    truncated = write_handoff(tmp_path / "truncated", payload_id=QID_HANDOFF, payload=payload)
    target = _rows_path(truncated)
    target.write_bytes(target.read_bytes()[:-32])
    with pytest.raises(MinistralHandoffError, match="handoff file digest differs"):
        read_handoff(truncated, payload_id=QID_HANDOFF)

    missing = write_handoff(tmp_path / "missing", payload_id=QID_HANDOFF, payload=payload)
    _rows_path(missing).unlink()
    with pytest.raises(
        MinistralHandoffError,
        match=f"{QID_HANDOFF}/{MINISTRAL_HANDOFF_ROWS}: handoff file is missing",
    ):
        read_handoff(missing, payload_id=QID_HANDOFF)

    torn = write_handoff(tmp_path / "torn", payload_id=QID_HANDOFF, payload=payload)
    torn.write_bytes(torn.read_bytes()[: len(torn.read_bytes()) // 2])
    with pytest.raises(MinistralHandoffError, match="handoff manifest is unreadable"):
        read_handoff(torn, payload_id=QID_HANDOFF)

    foreign = write_handoff(tmp_path / "foreign", payload_id=QID_HANDOFF, payload=payload)
    with pytest.raises(MinistralHandoffError, match="handoff manifest names another payload"):
        read_handoff(foreign, payload_id="fanoutqa-0009")
    _rewrite(foreign, lambda manifest: manifest.__setitem__("schema", "qwen-fleet-payload-v1"))
    with pytest.raises(MinistralHandoffError, match="handoff manifest schema differs"):
        read_handoff(foreign, payload_id=QID_HANDOFF)

    unsealed = write_handoff(tmp_path / "unsealed", payload_id=QID_HANDOFF, payload=payload)
    manifest: dict[str, Any] = json.loads(unsealed.read_text(encoding="utf-8"))
    manifest["meta"]["semantic_arm"] = "latent_q_snap_r2"
    unsealed.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(
        MinistralHandoffError, match="handoff manifest fingerprint differs from its content"
    ):
        read_handoff(unsealed, payload_id=QID_HANDOFF)

    for name, mutate, message in (
        ("dropped", lambda body: body["meta"].pop("tensor_sha256"), "metadata omits tensor_sha256"),
        ("no_files", lambda body: body["files"].pop(MINISTRAL_HANDOFF_ROWS), "file roster differs"),
        (
            "mistyped",
            lambda body: body["meta"].__setitem__("rows_by_worker", "3,2,4"),
            "metadata rows_by_worker is not integers",
        ),
        (
            "tensor",
            lambda body: body["meta"].__setitem__("tensor_sha256", other.tensor_sha256),
            "tensor digest differs",
        ),
        (
            "worker",
            lambda body: body["meta"].__setitem__(
                "worker_sha256", list(reversed(payload.worker_sha256))
            ),
            "worker digest differs",
        ),
        (
            "indices",
            lambda body: body["meta"].__setitem__(
                "selected_indices_sha256", other.selected_indices_sha256
            ),
            "selected-index digest differs",
        ),
    ):
        path = write_handoff(tmp_path / name, payload_id=QID_HANDOFF, payload=payload)
        _rewrite(path, mutate)
        with pytest.raises(MinistralHandoffError, match=message):
            read_handoff(path, payload_id=QID_HANDOFF)

    demoted = write_handoff(tmp_path / "demoted", payload_id=QID_HANDOFF, payload=payload)
    demoted_rows = _rows_path(demoted)
    torch.save(payload.rows.to(torch.float32), demoted_rows)
    _rewrite(
        demoted,
        lambda body: body["files"][MINISTRAL_HANDOFF_ROWS].__setitem__(
            "sha256", hashlib.sha256(demoted_rows.read_bytes()).hexdigest()
        ),
    )
    with pytest.raises(MinistralHandoffError, match="handoff rows are not rank-two bfloat16"):
        read_handoff(demoted, payload_id=QID_HANDOFF)


# --- The adapter surface and engine roles.

_ARM = "latent_query_support_r2"


def test_split_adapter_places_identifies_and_role_splits_every_registered_arm(
    tmp_path: Path,
) -> None:
    adapter = MinistralSplitFleetAdapter(
        fleet_config(tmp_path), authority_validator=lambda raw: dict(raw)
    )
    placements = adapter.placements.resolve(tmp_path, panel="production", source_commit="0" * 40)
    assert set(adapter.placements.arm_names) == set(placements)
    assert len(placements) == 12
    assert (placements["full"].producers, placements["full"].receivers) == (4, 4)
    bound = bind_policy_placements(
        ministral_semantic_bindings(),
        table=placement_table(adapter.config.benchmark_profile.topology_key),
    )
    assert placements == {
        arm: bound[policy] for policy, arm in ministral_semantic_bindings().items()
    }
    assert (
        placements["text_primary"].fused and placements["latent_query_support_r128"].producers == 5
    )

    placement = placements[_ARM]
    paths = adapter.bank_paths(tmp_path, panel="production", arm=_ARM, placement=placement)
    assert len(paths) == placement.workers == 8
    assert paths[0] == tmp_path / "arms" / _ARM / "workers" / "gpu0" / "raw.jsonl"

    identity = adapter.bank_identity(
        tmp_path,
        panel="production",
        arm=_ARM,
        source_commit="0" * 40,
        prepared_manifest={
            "artifact_sha256": "a" * 64,
            "construction_roster_fingerprint": "b" * 64,
        },
        placement=placement,
    )
    assert isinstance(identity, BankIdentity)
    assert identity.fields["execution_isolation_profile"] == SPLIT_FLEET_ISOLATION_PROFILE
    assert identity.fields["semantic_arm"] == _ARM
    assert identity.fields["placement"]["producers"] == 4  # four producer seats of eight
    assert identity.fields["decode_fingerprint"] == MINISTRAL.decode.identity_hash
    assert identity.fields["benchmark_profile"] == adapter.config.benchmark_profile.profile_id
    assert (
        identity.fields["source_identity"]
        == adapter.config.benchmark_profile.source_logical_fingerprint
    )
    assert (
        identity.fields["source_cache_fingerprint"]
        == adapter.config.benchmark_profile.source_logical_fingerprint
    )
    assert (
        identity.fields["source_manifest_sha256"]
        == adapter.config.benchmark_profile.source_manifest_sha256
    )
    assert identity.fields["model_id"] == MINISTRAL.model_id
    assert identity.fields["nonresident_runtime_authority"] == {"fixture": True}
    with pytest.raises(ValueError, match="unregistered Ministral split arm"):
        adapter.bank_identity(
            tmp_path,
            panel="production",
            arm="r2",
            source_commit="0" * 40,
            prepared_manifest={"artifact_sha256": "a" * 64},
            placement=placement,
        )
    assert adapter.registration()["isolation_profile"] == SPLIT_FLEET_ISOLATION_PROFILE
    assert adapter.sampling_contract()["sample_tags"] == ["s0", "s1", "s2"]
    # The sampling contract is read off the registered Ministral protocol, not
    # copied from another family's numbers.
    assert adapter.sampling_contract()["temperature"] == MINISTRAL.decode.temperature
    assert adapter.sampling_contract()["top_k"] is None

    # Only the one timed seed of a decoded cell is a completion. Its two peers
    # are banked before it, so counting them would let a crash in between leave
    # a two-of-three cell that a later read takes as done.
    assert TIMED_SAMPLE_TAG == "s0"
    assert {adapter.completion_key(row) for row in ({"kind": "phase"}, {"kind": "stage"})} == {None}
    assert (
        adapter.completion_key(
            {"kind": "result", "panel": "production", "qid": "q1", "sample_tag": "s0"}
        )
        is None
    )
    assert (
        adapter.completion_key(
            {
                "kind": "result",
                "panel": "production",
                "qid": "q1",
                "sample_tag": "s1",
                "cell": f"{_ARM}|s1",
            }
        )
        is None
    )
    assert adapter.completion_key(
        {
            "kind": "result",
            "panel": "production",
            "qid": "q1",
            "sample_tag": "s0",
            "cell": f"{_ARM}|s0",
        }
    ) == ("production", "q1", f"{_ARM}|s0")
    # The split report reads the cell ledger, so an empty root has no arm to
    # report over, and a row-selection policy is refused rather than ignored.
    with pytest.raises(RuntimeError, match="found no banked arm"):
        adapter.reports.build_report(
            tmp_path,
            source_commit="0" * 40,
            results_policy=None,
        )
    with pytest.raises(RuntimeError, match="selects by cell ledger"):
        adapter.reports.build_report(
            tmp_path,
            source_commit="0" * 40,
            results_policy=object(),
        )

    # The capture identity is hashed into every banked resident row, so these
    # four strings are pinned here byte for byte.
    assert registered_identity(PRODUCER_ROLE) is CAPTURE_ENGINE_IDENTITY
    assert CAPTURE_ENGINE_IDENTITY.engine_role == "receiver-capture-text-primary"
    assert dict(CAPTURE_ENGINE_IDENTITY.capture_connector or {}) == {
        "kv_connector": "MinistralCaptureConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": "rcc.models.ministral.capture_connector",
    }

    # The receiver is a second registered identity beside it: it injects
    # embedding rows and opens with no capture route at all.
    receiver = registered_identity(RECEIVER_ROLE)
    assert receiver is SPLIT_RECEIVER_ENGINE_IDENTITY
    assert receiver.kv_transfer_config is None and receiver.enable_prompt_embeds
    assert receiver.engine_role != CAPTURE_ENGINE_IDENTITY.engine_role
    assert (receiver.checkpoint, receiver.revision) == (MINISTRAL.checkpoint, MINISTRAL.revision)
    with pytest.raises(ValueError, match="unregistered Ministral engine role"):
        registered_identity("capture")

    # Only the transfer stanza separates the two roles; the checkpoint pin and
    # the injection route are identical, so the two halves decode alike.
    assert CAPTURE_ENGINE_IDENTITY.kv_transfer_config == tuple(sorted(CAPTURE_CONNECTOR.items()))
    assert {
        name: value
        for name, value in CAPTURE_ENGINE_IDENTITY.expectations().items()
        if name not in {"capture_connector", "engine_role"}
    } == {
        name: value
        for name, value in receiver.expectations().items()
        if name not in {"capture_connector", "engine_role"}
    }

    # Both roles are registered in the runtime validator, and nothing else is.
    signature = {
        "engine_role": receiver.engine_role,
        "enable_prompt_embeds": True,
        "capture_connector": None,
    }
    with pytest.raises(RuntimeError, match="observed runtime differs"):
        validate_runtime_signature(
            signature,
            active_checkpoint=receiver.checkpoint,
            active_revision=receiver.revision,
            engine_role=receiver.engine_role,
        )
    with pytest.raises(RuntimeError, match="unregistered Ministral engine role"):
        validate_runtime_signature(
            signature,
            active_checkpoint=receiver.checkpoint,
            active_revision=receiver.revision,
            engine_role="split-receiver",
        )

    with pytest.raises(RuntimeError, match="observed runtime differs"):
        verify_engine_identity(SimpleNamespace(runtime_signature=dict(signature)), receiver)
    with pytest.raises(RuntimeError, match="exposes no observed runtime signature"):
        verify_engine_identity(SimpleNamespace(), receiver)

    # The text wire format round trips one sealed bundle and refuses another
    # arm's, so a receiver cannot serve reports it did not claim.
    bundle = report_bundle(QIDS[0], "text_medium")
    path = write_text_bundle(text_bundle_path(tmp_path, QIDS[0], "text_medium"), bundle)
    assert read_text_bundle(path, qid=QIDS[0], semantic_arm="text_medium") == bundle
    with pytest.raises(MinistralTextHandoffError, match="report bundle identity differs"):
        read_text_bundle(path, qid=QIDS[0], semantic_arm="text_small")

    # A nonresident sender produces on another GPU, so the ladder and injection
    # evidence has to cross the wire with the reports. Without it a redrawn
    # cell would reach the bank looking like one clean natural draw.
    continued = bundle_from_raw(
        QIDS[1],
        "text_medium",
        (ANSWER_TOKENS, INJECTED_TOKENS, ANSWER_TOKENS),
        injected=(False, True, False),
        draws_n=2,
        redraw_wall_s=31.5,
        generation_s=12.25,
    )
    wire = write_text_bundle(text_bundle_path(tmp_path, QIDS[1], "text_medium"), continued)
    reloaded = read_text_bundle(wire, qid=QIDS[1], semantic_arm="text_medium")
    assert reloaded == continued
    assert (reloaded.draws_n, reloaded.redraw_wall_s) == (2, 31.5)
    assert reloaded.injected_by_worker == (False, True, False)
    assert reloaded.generation_s == 12.25

    # The clock record beside it prices the publish and binds that clock to the
    # exact bytes: another arm's record, a negative clock, a missing record, and a
    # bundle rewritten after publication all refuse.
    write_bundle_clocks(
        TEXT_BUNDLE_CLOCKS, path, qid=QIDS[0], semantic_arm="text_medium", spill_save_s=0.25
    )
    assert (
        read_bundle_clocks(TEXT_BUNDLE_CLOCKS, path, qid=QIDS[0], semantic_arm="text_medium")
        == 0.25
    )
    assert clock_record_path(path).name.endswith(".reports.json.clocks.json")
    with pytest.raises(MinistralTextHandoffError, match="bundle clock record names other bytes"):
        read_bundle_clocks(TEXT_BUNDLE_CLOCKS, path, qid=QIDS[0], semantic_arm="text_small")
    with pytest.raises(MinistralTextHandoffError, match="save clock must be nonnegative"):
        write_bundle_clocks(
            TEXT_BUNDLE_CLOCKS, path, qid=QIDS[0], semantic_arm="text_medium", spill_save_s=-0.1
        )
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(MinistralTextHandoffError, match="bundle clock record names other bytes"):
        read_bundle_clocks(TEXT_BUNDLE_CLOCKS, path, qid=QIDS[0], semantic_arm="text_medium")
    clock_record_path(path).unlink()
    with pytest.raises(MinistralTextHandoffError, match="bundle clock record is unreadable"):
        read_bundle_clocks(TEXT_BUNDLE_CLOCKS, path, qid=QIDS[0], semantic_arm="text_medium")


def test_producer_role_lowers_its_memory_budget_without_moving_identity() -> None:
    """The producer's memory budget moves; the signed identity does not.

    The arithmetic behind these two numbers is in docs/running.md, under
    "Ministral producer memory".
    """
    producer = registered_identity(PRODUCER_ROLE)
    receiver = registered_identity(RECEIVER_ROLE)
    assert (producer.gpu_memory_utilization, receiver.gpu_memory_utilization) == (0.72, 0.85)
    assert role_engine_config(producer).gpu_memory_utilization == 0.72
    assert role_engine_config(receiver).gpu_memory_utilization == 0.85
    # An explicit config still wins.
    override = MinistralEngineConfig(gpu_memory_utilization=0.5)
    assert role_engine_config(producer, override) is override
    # The other three config-derived kwargs reach the same vLLM constructor as
    # the utilization, so they are pinned here too.
    for role in (producer, receiver):
        resolved = role_engine_config(role)
        assert (resolved.enforce_eager, resolved.seed, resolved.disable_log_stats) == (
            False,
            0,
            False,
        )

    # `observed_runtime_signature` reads the engine kwargs only through the
    # role identity and otherwise carries the static registered flag roster,
    # so the utilization is in neither and cannot reach a banked row.
    def kwargs(utilization: float) -> dict[str, object]:
        return {
            "model": MINISTRAL.checkpoint,
            "revision": MINISTRAL.revision,
            "enable_prompt_embeds": True,
            "kv_transfer_config": dict(CAPTURE_CONNECTOR),
            "gpu_memory_utilization": utilization,
        }

    signatures = [
        observed_runtime_signature(kwargs(value), engine_role=CAPTURE_ENGINE_IDENTITY.engine_role)
        for value in (0.85, 0.72)
    ]
    assert signatures[0] == signatures[1]
    assert runtime_fingerprint(signatures[0]) == runtime_fingerprint(signatures[1])
    assert "gpu_memory_utilization" not in repr(signatures[0])
    # The flags that are signed did not move with it.
    assert dict(MINISTRAL_RUNTIME.engine_flags)["max_model_len"] == "180224"
    assert dict(MINISTRAL_RUNTIME.engine_flags)["max_num_seqs"] == "16"


# --- The offline producer and receiver.


@pytest.fixture(autouse=True)
def _offline_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    """Score offline: this suite covers plumbing, not FanOutQA answers."""
    monkeypatch.setattr(
        "rcc.run.ministral.split_receiver.score_text",
        lambda _question, _visible: {"loose": 0.5, "strict": 0.0, "n_leaves": 2.0},
    )


def _drive(
    root: Path,
    bank: Bank,
    placement: FleetPlacement,
    workers: list[Any],
    *,
    arm: str,
) -> None:
    """Run one arm's roles as threads through the shipped fleet runtime."""
    failures: list[BaseException] = []

    def one(index: int, worker: Any) -> None:
        append = cast(Callable[[dict[str, Any]], None], bank.append)
        hooks = FleetWorkerHooks(worker, bank.completed, append, append)
        try:
            run_worker(
                FleetRunConfig(
                    root=root / "arms" / arm,
                    arm=arm,
                    attempt_id="a1",
                    qids=QIDS,
                    placement=placement,
                    worker_index=index,
                    max_wall_s=60,
                    barrier_timeout_s=20,
                    claim_lease_s=20,
                    poll_s=0.001,
                ),
                hooks,
            )
        except BaseException as error:
            failures.append(error)

    threads = [
        threading.Thread(target=one, args=(index, worker)) for index, worker in enumerate(workers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert not [thread for thread in threads if thread.is_alive()]
    assert not failures, failures[0]


_BARRIER = BarrierSpec(
    family_label="ministral split",
    marker_schema="ministral-split-arm-barrier-marker-v1",
    barrier_dirname="arm-barrier",
)


def test_the_shared_driver_runs_one_latent_arm_then_merges_and_reports(tmp_path: Path) -> None:
    # One registered latent arm, at the node width one test process can drive.
    # What the driver decides is the isolation and the merge.
    placement = FleetPlacement(workers=2, producers=1, receivers=1)
    arms = {SUPPORT_ARM: placement}
    adapter = fixture_adapter(tmp_path, arms)
    report = run_split_fleet(
        adapter,
        SplitScheduleConfig(
            root=tmp_path,
            panel="production",
            source_commit="0" * 40,
            attempt_id="a1",
            barrier=_BARRIER,
            plan_fingerprint="plan",
            roster_fingerprint="roster",
            arms=tuple(arms),
            max_wall_s=60.0,
            barrier_timeout_s=20.0,
            claim_lease_s=20.0,
            poll_s=0.001,
            join_timeout_s=120.0,
        ),
    )

    assert report["arms"] == [SUPPORT_ARM]

    # The same shared merge and report the Gemma split run publishes, under
    # the Ministral schema and the registered semantic comparison.
    report = adapter.reports.build_report(tmp_path, source_commit="0" * 40)
    assert report["schema"] == "ministral-fanoutqa-split-report-v1"
    assert report["result_cells"] == len(QIDS)
    assert report["cell_width_by_arm"] == {SUPPORT_ARM: 3}
    grid = {row["arm"]: row for row in cast(list[dict[str, Any]], report["arm_grid"])}
    assert grid[SUPPORT_ARM]["n"] == len(QIDS)
    assert grid[SUPPORT_ARM]["fleet_producer_s_mean"] > 0.0
    comparisons = cast(list[dict[str, Any]], report["paired_comparisons"])
    assert comparisons == []
    banked_rows = [
        row
        for row in read_arm_bank_rows(tmp_path / "arms" / SUPPORT_ARM)
        if row.get("kind") == "result"
    ]
    assert banked_rows
    # The recall reader names the handoff fields as literals, so a producer
    # that renamed one would empty its draw comparison rather than fail it.
    # This arm is latent, so it carries none of the text channel's reports.
    cut = set(recall.HANDOFF_FIELDS) - {"reports", "cut_decoded_sha256"}
    assert all({"prepared_sha256", *cut} <= set(row) for row in banked_rows)
    assert all(row["source_cache_fingerprint"] for row in banked_rows)
    assert all(row["source_manifest_sha256"] for row in banked_rows)
    assert all(row["nonresident_runtime_authority"] == {"fixture": True} for row in banked_rows)
    assert {
        parse_ministral_construction_manifest(row["construction_manifest"]).semantic_arm
        for row in banked_rows
    } == {"text_primary"}
    assert all(
        validate_ministral_result_construction(
            row,
            qid=str(row["qid"]),
            source_arm="text_primary",
        )
        for row in banked_rows
    )


def test_split_arms_cross_the_queue_as_clock_bound_handoffs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for arm in (SUPPORT_ARM,):
        bank = Bank()
        root = tmp_path / arm
        _drive(
            root,
            bank,
            FleetPlacement(workers=2, producers=1, receivers=1),
            [producer(root, bank.append, arm=arm), receiver(arm, bank=bank.append)],
            arm=arm,
        )

        # Every item crossed the process boundary as a manifest the receiver
        # verified, and came back as the registered three seed rows.
        rows = bank.results()
        assert {row["scoring_version"] for row in rows} == {"fanoutqa-equivalence-v1"}
        assert {row["qid"] for row in rows} == set(QIDS)
        assert sorted(row["cell"] for row in rows if row["qid"] == QIDS[0]) == [
            f"{arm}|s{index}" for index in range(3)
        ]
        stages = {(row["stage"], row["qid"]) for row in bank.stages()}
        assert {("producer_end", QIDS[0]), ("payload_loaded", QIDS[0])} <= stages
        for qid in QIDS:
            manifest = handoff_manifest_path(handoff_root(root, arm), qid, arm)
            assert manifest.is_file() and clock_record_path(manifest).is_file()

        # The whole shared latency vocabulary is present and both spill clocks
        # are real: the producer measured its publish, the receiver its read.
        row = rows[0]
        assert not [name for name in ITEM_LATENCY_FIELDS if name not in row]
        assert row["spill_save_s"] > 0.0 and row["spill_load_s"] > 0.0
        clocks = json.loads(
            clock_record_path(
                handoff_manifest_path(handoff_root(root, arm), str(row["qid"]), arm)
            ).read_text(encoding="utf-8")
        )
        assert row["spill_save_s"] == clocks["spill_save_s"]
        assert row["producer_s"] == clocks["producer_s"]
        # The two harness clocks stay outside what the caller waited for.
        assert row["ttft_s"] == pytest.approx(
            row["producer_s"] + row["receiver_prepare_s"] + row["receiver_ttft_s"], abs=1e-4
        )
        assert row["arm"] == row["semantic_arm"] == arm
        assert row["payload_rows_by_worker"] == [417, 417, 417]
        assert len(row["payload_worker_sha256"]) == 3

    # The registered ladder binds the Query-Support arm to its shared placement.
    placements = bind_policy_placements(
        ministral_semantic_bindings(), table=SEMANTIC_ARM_PLACEMENTS
    )
    by_arm = {arm: placements[policy] for policy, arm in ministral_semantic_bindings().items()}
    assert by_arm[SUPPORT_ARM] == FleetPlacement(8, 5, 3)

    # A nonresident text arm crosses the same queue with a report bundle. Only
    # the two seams a live sender owns are faked, the sealed prompt rows and
    # the decode; everything from the bundle onward is the shipped wire.
    monkeypatch.setattr("rcc.run.ministral.split_producer.prompt_records", fixture_prompt_records)
    monkeypatch.setattr("rcc.run.ministral.split_producer.generate_report_bundle", fixture_reports)
    text_bank = Bank()
    text_root = tmp_path / TEXT_ARM
    _drive(
        text_root,
        text_bank,
        FleetPlacement(workers=2, producers=1, receivers=1),
        [text_producer(text_root, text_bank.append), receiver(TEXT_ARM, bank=text_bank.append)],
        arm=TEXT_ARM,
    )
    text_rows = text_bank.results()
    assert {row["qid"] for row in text_rows} == set(QIDS)
    assert all(row["reports"] and row["payload_layout"] is None for row in text_rows)
    bundles = {
        qid: text_bundle_path(handoff_root(text_root, TEXT_ARM), qid, TEXT_ARM) for qid in QIDS
    }
    published = {
        str(banked["qid"]): banked["spill_save_s"]
        for banked in text_bank.rows
        if banked.get("phase") == "ministral_split_reports"
    }
    for qid, bundle_path in bundles.items():
        clocks = json.loads(clock_record_path(bundle_path).read_text(encoding="utf-8"))
        served = next(row for row in text_rows if row["qid"] == qid and row["sample_tag"] == "s0")
        assert not [name for name in ITEM_LATENCY_FIELDS if name not in served]
        assert published[qid] == clocks["spill_save_s"] == served["spill_save_s"] > 0.0
        assert served["spill_load_s"] > 0.0
        assert served["producer_s"] == report_bundle(qid, TEXT_ARM).generation_s

    # A bundle rewritten between publication and claim differs from the digest
    # the record names, so the receiver refuses it rather than serving reports
    # nobody measured.
    rewritten = bundles[QIDS[0]]
    rewritten.write_bytes(rewritten.read_bytes() + b"\n")
    text_receiver = receiver(TEXT_ARM, bank=Bank().append)
    text_receiver.warm()
    with pytest.raises(MinistralTextHandoffError, match="bundle clock record names other bytes"):
        text_receiver.submit(WorkItem(QIDS[0], 0), ProducedArtifact(rewritten, 1.0))

    # A receiver refuses a payload whose bytes moved under it, and a record
    # that prices some other handoff.
    tampered_root = tmp_path / SUPPORT_ARM
    manifest = handoff_manifest_path(handoff_root(tampered_root, SUPPORT_ARM), QIDS[0], SUPPORT_ARM)
    body = json.loads(manifest.read_text(encoding="utf-8"))
    rows_path = manifest.parent / body["files"][MINISTRAL_HANDOFF_ROWS]["path"]
    raw = bytearray(rows_path.read_bytes())
    raw[-1] ^= 0x01
    rows_path.write_bytes(bytes(raw))
    fresh = receiver(SUPPORT_ARM, bank=Bank().append)
    fresh.warm()
    artifact = ProducedArtifact(manifest, 1.0)
    with pytest.raises(MinistralHandoffError, match="handoff file digest differs"):
        fresh.submit(WorkItem(QIDS[0], 0), artifact)
    # The direct arm never accepts a payload, whatever a ticket claims.
    fresh.arm = "issue_only"
    with pytest.raises(RuntimeError, match="admits no producer artifact"):
        fresh.submit(WorkItem(QIDS[0], 0), artifact)

    # A receiver banks the batched peers itself and hands the timed row back
    # to be banked last, so only that write completes the item: counting the
    # peers would let a crash leave a two-of-three cell that reads as done.
    crashed = Bank()
    live = receiver(SUPPORT_ARM, bank=crashed.append)
    live.warm()
    intact = handoff_manifest_path(handoff_root(tampered_root, SUPPORT_ARM), QIDS[1], SUPPORT_ARM)
    live.submit(WorkItem(QIDS[1], 1), ProducedArtifact(intact, 1.0))
    timed = drain_one(live)[0].result
    adapter = MinistralSplitFleetAdapter(fleet_config(tmp_path))
    assert len(crashed.results()) == 2
    assert {adapter.completion_key(row) for row in crashed.results()} == {None}
    assert crashed.completed() == set()
    assert adapter.completion_key(timed) == ("production", QIDS[1], f"{SUPPORT_ARM}|s0")
    crashed.append(timed)
    assert crashed.completed() == {QIDS[1]}


def test_direct_and_fused_arms_serve_with_nothing_on_the_wire(tmp_path: Path) -> None:
    # The direct arm gives every GPU to receivers; the fixture runs the same
    # shape at the width one test process can drive.
    placements = bind_policy_placements(
        ministral_semantic_bindings(), table=SEMANTIC_ARM_PLACEMENTS
    )
    assert placements["none"] == FleetPlacement(8, 0, 8)
    assert placements["ministral3_14b_text"].fused

    direct = Bank()
    _drive(
        tmp_path / "direct",
        direct,
        FleetPlacement(workers=2, producers=0, receivers=2),
        [receiver("issue_only", bank=direct.append) for _index in range(2)],
        arm="issue_only",
    )
    rows = direct.results()
    assert {row["qid"] for row in rows} == set(QIDS)
    # Nothing shipped, so all three producer-side clocks are honestly zero and
    # no producer stage was ever banked.
    assert all(
        row["spill_load_s"] == 0.0 and row["spill_save_s"] == 0.0 and row["producer_s"] == 0.0
        for row in rows
    )
    assert not any(row["stage"] == "producer_start" for row in direct.stages())
    assert all(row["payload_layout"] is None for row in rows)

    fused = Bank()
    generated: list[str] = []

    def reports(qid: str) -> Any:
        generated.append(qid)
        return report_bundle(qid, "text_primary")

    _drive(
        tmp_path / "fused",
        fused,
        FleetPlacement(workers=1, producers=0, receivers=1, fused=True),
        [receiver("text_primary", bank=fused.append, fused_reports=reports)],
        arm="text_primary",
    )
    assert sorted(generated) == sorted(QIDS)
    assert {row["qid"] for row in fused.results()} == set(QIDS)
    # Same model, same GPU: the producer and handoff boundaries are one
    # instant, and the runtime banks all four of them from the fused clock.
    stages = {(row["stage"], row["qid"]) for row in fused.stages()}
    for name in ("producer_end", "handoff_ready", "receiver_claim", "payload_loaded"):
        assert (name, QIDS[0]) in stages, name
    assert (
        len(
            {
                row["t_unix"]
                for row in fused.stages()
                if row["qid"] == QIDS[0] and row["stage"] in {"producer_end", "payload_loaded"}
            }
        )
        == 1
    )
    assert all(row["spill_load_s"] == 0.0 for row in fused.results())
    assert all(row["reports"] for row in fused.results())

    # Ministral redraws often, so the discarded draws bank beside
    # `report_generation_s`: the fused receiver charges producer_s with the
    # shipped span, not the wall it spent inside the generator.
    clocked = Bank()
    timed = receiver(
        FUSED_ARM,
        bank=clocked.append,
        fused_reports=lambda qid: bundle_from_raw(
            qid,
            FUSED_ARM,
            (ANSWER_TOKENS,) * 3,
            draws_n=2,
            redraw_wall_s=120.0,
            generation_s=37.5,
        ),
    )
    timed.warm()
    timed.submit(WorkItem(QIDS[0], 0), None)
    timed_row = drain_one(timed)[0].result
    assert timed_row["producer_s"] == 37.5
    assert (timed_row["draws_n"], timed_row["redraw_wall_s"]) == (2, 120.0)
    assert timed_row["report_generation_s"] == 37.5

    # Fusing is a property of one registered same-model arm, not a mode.
    with pytest.raises(ValueError, match="serves text_primary only"):
        MinistralFusedReceiver(
            reports=reports,
            arm="issue_only",
            items=items(),
            bank=fused.append,
            core=cast(Any, None),
            report_codecs={"text_primary": codec()},
        )


def test_closer_injection_round_trips_through_the_real_validator() -> None:
    """Injected, naturally closed, and quarantined bundles all rescore clean.

    A continued draw reaches the independent rescore as ordinary evidence:
    every published field is reproduced from the banked ids alone.
    """
    report_codec = codec(TEXT_ARM)
    qid = QIDS[0]

    def round_trip(bundle: Any) -> dict[str, Any]:
        row = {"qid": qid, "arm": TEXT_ARM, **bundle.result_fields()}
        rescore_report_row(row, report_codec)
        return row

    natural = round_trip(bundle_from_raw(qid, TEXT_ARM, (ANSWER_TOKENS,) * 3))
    assert natural["injected"] is False
    assert natural["report_injected_by_worker"] == [False] * 3
    assert natural["draws_n"] == 1

    # One injected worker beside two naturally closed ones. The injected row
    # carries a real post-thought answer, so it is accepted, not quarantined.
    mixed = round_trip(
        bundle_from_raw(
            qid,
            TEXT_ARM,
            (ANSWER_TOKENS, INJECTED_TOKENS, ANSWER_TOKENS),
            injected=(False, True, False),
            draws_n=2,
            redraw_wall_s=4.0,
        )
    )
    assert mixed["injected"] is True
    assert mixed["report_injected_by_worker"] == [False, True, False]
    assert mixed["reports"][1] == "3001"
    assert (mixed["draws_n"], mixed["redraw_wall_s"]) == (2, 4.0)

    # A cell the sealed ladder exhausted: quarantined workers carry empty
    # reports and still round trip, injected or not.
    quarantined = round_trip(
        bundle_from_raw(
            qid,
            TEXT_ARM,
            (ANSWER_TOKENS, INJECTED_TOKENS, ANSWER_TOKENS),
            injected=(False, True, False),
            failed=(1,),
            draws_n=3,
            redraw_wall_s=9.0,
        )
    )
    assert quarantined["report_failed"] is True
    assert quarantined["reports"][1] == ""
    assert quarantined["report_injected_by_worker"] == [False, True, False]

    # With the closer appended after the end of turn, every Ministral reader
    # cuts at the stop id and finds no closed channel at all, so such a draw
    # banks no report.
    assert codec().visible_report(INJECTED_TOKENS)[0] == (3_001,)
    with pytest.raises(RuntimeError, match="one closed thought channel"):
        codec().visible_report(PRE_FIX_TOKENS)
    # And the bundle refuses to bank that shape as an injected draw at all,
    # even quarantined, where no reader would look at the content.
    with pytest.raises(ValueError, match="replace the head's end of turn"):
        bundle_from_raw(
            qid,
            TEXT_ARM,
            (ANSWER_TOKENS, PRE_FIX_TOKENS, ANSWER_TOKENS),
            injected=(False, True, False),
            failed=(1,),
        )
    # And a worker flagged injected whose ids carry no closer at all.
    with pytest.raises(ValueError, match="bank its closing delimiter"):
        bundle_from_raw(
            qid,
            TEXT_ARM,
            (ANSWER_TOKENS, (34, 2_000, 3_000, 2), ANSWER_TOKENS),
            injected=(False, True, False),
            failed=(1,),
        )
    # The injection and ladder columns are producer provenance, so the rescore
    # cannot rebuild them; it enforces internal consistency instead. The two
    # injection columns must agree and a flagged worker's ids must show a closer.
    honest = bundle_from_raw(qid, TEXT_ARM, (ANSWER_TOKENS,) * 3)
    forged = {"qid": qid, "arm": TEXT_ARM, **honest.result_fields()}
    forged["report_injected_by_worker"] = [False, True, False]
    with pytest.raises(RuntimeError, match=r"rescore differs in \['injected'\]"):
        rescore_report_row(forged, report_codec)
    # Flipping both columns instead fails on the ids themselves: this worker
    # ended its turn unclosed and no closer was ever banked for it.
    unclosed = bundle_from_raw(
        qid,
        TEXT_ARM,
        (ANSWER_TOKENS, (34, 2_000, 3_000, 2), ANSWER_TOKENS),
        failed=(1,),
    )
    claimed = {"qid": qid, "arm": TEXT_ARM, **unclosed.result_fields()}
    claimed["injected"] = True
    claimed["report_injected_by_worker"] = [False, True, False]
    with pytest.raises(ValueError, match="bank its closing delimiter"):
        rescore_report_row(claimed, report_codec)


def test_the_receiver_admits_many_cells_under_the_registered_gate() -> None:
    """Admission symmetry: the claim cap is max_num_seqs, the gate owns memory.

    The decode seam streams, so several cells overlap in one engine batch and
    the AdmissionGate is what keeps them inside the KV pool.
    """
    banked = Bank()
    engine = _Engine()
    live = receiver(DIRECT_ARM, bank=banked.append, engine=engine)
    live.warm()
    assert live.claim_limit == MINISTRAL_ENGINE_MAX_NUM_SEQS == 16
    assert live.can_accept() and live.idle()

    warm_row = next(row for row in banked.rows if row.get("kind") == "phase")
    assert warm_row["claim_limit"] == 16
    assert warm_row["kv_capacity_tokens"] == int(1_000_000 * 0.95)

    # Both items are claimed before either decodes.
    for order, qid in enumerate(QIDS):
        assert live.can_accept()
        live.submit(WorkItem(qid, order), None)
    assert not live.idle()
    finished: list[Any] = []
    for _step in range(64):
        finished.extend(live.pump())
        if len(finished) == len(QIDS):
            break
    assert {completion.qid for completion in finished} == set(QIDS)
    assert live.idle() and live.can_accept()
    # The two cells really shared the batch rather than running end to end.
    assert engine.max_live > 1
    # Each cell banks its two batched peers here; the timed row goes back to
    # the runtime, which banks it last. Driving pump directly, as this test
    # does, therefore leaves the peers behind and no completed item.
    assert len(banked.results()) == 2 * len(QIDS)
    assert banked.completed() == set()
    assert [completion.result["sample_tag"] for completion in finished] == [TIMED_SAMPLE_TAG] * 2
    # The gate's reservation trace rides on the timed sample's admission.
    for completion in finished:
        assert completion.fields["reserved_tokens"] > 0
        assert "sibling_prepares" in completion.fields
        # The host cost of the claim cap is banked, not assumed: two cells were
        # claimed at once, so the peak exceeds one cell's own prompt.
        assert completion.fields["peak_resident_prompt_bytes"] > 0


# --- Offline handoff recall over one banked Ministral run.


#: What the Ministral builders write of `recall.HANDOFF_FIELDS`:
#: `cut_decoded_sha256` is the Gemma spelling of the shipped payload's digest.
_HANDOFF_FIELDS = tuple(name for name in recall.HANDOFF_FIELDS if name != "cut_decoded_sha256")


def _ministral_result_rows(root: Path, arm: str, **handoff: Any) -> Path:
    """Bank one arm's three seed rows under the Ministral cell spelling."""
    return bank_result_rows(
        root,
        arm,
        cells=[
            {"cell": f"{arm}|{tag}", "sample_tag": tag}
            for tag in FANOUTQA_NATURAL_DEV50.sample_tags
        ],
        written=_HANDOFF_FIELDS,
        **handoff,
    )


def test_the_ministral_reader_scores_the_keeps_one_banked_run_handed_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Ministral reader scores the decoded keeps, and refuses a moved handoff.

    One natural bundle, the shipped prompt-panel builder, a published handoff
    written by `write_handoff`, and bank rows carrying the producers' own
    fields: a latent arm whose keeps hold one reference's rows scores one leaf
    of two, the text arm scores the leaves its reports name, and the whole
    source scores both. A retouched keep hash, prepared digest, or banked
    payload digest is refused rather than rescored.
    """
    bundle = tmp_path / "bundle"
    profile = write_bundle(bundle)
    # Every Ministral panel check resolves its profile out of the registry by
    # source fingerprint or profile id, so the stand-in panel is registered the
    # way the sealed fifty-item one is.
    for module in (ministral_construction, ministral_data):
        monkeypatch.setattr(module, "REGISTERED_PROFILES", (profile,))
    monkeypatch.setattr(
        ministral_construction,
        "_REGISTERED_SOURCE_IDENTITIES",
        frozenset({profile.source_logical_fingerprint}),
    )
    # The freeze pin covers the shipped dev index, not a one-question stand-in.
    monkeypatch.setattr(
        ministral_data, "load_questions", lambda path: load_questions(path, verify_sha=False)
    )
    # The config names its benchmark by id, so the stand-in profile has to
    # answer to the sealed one's name while the plan resolves.
    monkeypatch.setitem(plan_module._BENCHMARKS, "fanoutqa-natural-dev50", profile)
    config = _one_item_config(tmp_path, "ministral-fanoutqa-natural-dev50.toml")
    plan = plan_module.load_and_resolve(config)
    root = Path(plan.output_uri)

    codecs = {arm: word_codec(arm) for arm in MINISTRAL_TEXT_ARMS}
    # The reader builds its decode seam from a tokenizer snapshot on disk, and
    # no CPU gate holds one; the fixture codecs answer for it.
    monkeypatch.setattr(
        ministral_text_codec, "load_registered_text_codec", lambda _path, arm: codecs[arm]
    )
    _record, texts = natural_worker_texts(bundle, profile=profile)["q1"]
    evidence = {arm: {"q1": [_Inner.encode(row) for row in texts]} for arm in MINISTRAL_TEXT_ARMS}
    construction = {
        arm: {
            "q1": ministral_construction.make_ministral_construction_manifest(
                qid="q1",
                semantic_arm=arm,
                retained_sources=tuple(
                    str(codecs[arm].tokenizer.decode(row)) for row in evidence[arm]["q1"]
                ),
                audit=_EMPTY_CONSTRUCTION_AUDIT,
                profile=profile,
            )
        }
        for arm in MINISTRAL_TEXT_ARMS
    }
    panel = ministral_data.prepare_ministral_prompt_panel(
        bundle,
        selected_qids=("q1",),
        codecs=codecs,
        evidence_ids_by_arm=evidence,
        construction_by_arm=construction,
        profile=profile,
    )
    ministral_data.write_ministral_prompt_panel(prepared_panel_path(root, 0, 1), panel)

    # Worker zero keeps the one row spelling the first river; the other two
    # workers keep nothing, so the handoff carries a single reference.
    prompts = panel.artifact("q1", "text_primary").prompt_ids_by_worker
    kept = tuple(codecs["text_primary"].tokenizer.decode(prompts[0]).split()).index("Alpha")
    keeps = ((kept,), (), ())
    rows = torch.zeros(1, 8, dtype=torch.bfloat16)
    payload = MinistralFlatPayload(
        rows=rows,
        semantic_arm=SUPPORT_ARM,
        latent_plan_sha256=latent_plan_sha256(SUPPORT_ARM),
        keeps=keeps,
        rows_by_worker=(1, 0, 0),
        worker_sha256=tuple(tensor_content_sha256(block) for block in (rows, rows[0:0], rows[0:0])),
        selected_indices_sha256=selected_indices_sha256(keeps),
        tensor_sha256=tensor_content_sha256(rows),
    )
    write_handoff(
        handoff_root(root, SUPPORT_ARM),
        payload_id=payload_id("q1", SUPPORT_ARM),
        payload=payload,
    )
    bank = _ministral_result_rows(
        root,
        SUPPORT_ARM,
        prepared_sha256=panel.fingerprint,
        selected_indices_sha256=payload.selected_indices_sha256,
        payload_sha256=payload.tensor_sha256,
    )
    _ministral_result_rows(
        root,
        "text_primary",
        prepared_sha256=panel.fingerprint,
        reports=["the Alpha river", "nothing locatable", "the Beta river"],
    )

    body = recall.handoff_recall(plan, root, bundle, tokenizer_snapshot=tmp_path)
    by_arm = {summary["arm"]: summary for summary in body["arms"]}
    assert body["model_id"] == "ministral3-14b"
    assert (body["source"]["hits"], body["source"]["leaves"]) == (2, 2)
    assert (by_arm["text_primary"]["hits"], by_arm["text_primary"]["leaves"]) == (2, 2)
    assert by_arm[SUPPORT_ARM] == {
        "arm": SUPPORT_ARM,
        "questions": 1,
        "hits": 1,
        "leaves": 2,
        "micro_recall_percent": 50.0,
        "macro_recall_percent": 50.0,
    }

    # Each banked digest signs one artifact the reader read back: the
    # manifest's keeps, the panel the prompts came from, the published tensor.
    rebank = rebanker(bank)
    rebank(selected_indices_sha256=selected_indices_sha256(((0,), (), ())))
    with pytest.raises(RuntimeError, match="banked selection hash"):
        recall.handoff_recall(plan, root, bundle, tokenizer_snapshot=tmp_path)
    rebank(prepared_sha256="f" * 64)
    with pytest.raises(RuntimeError, match="prepared_sha256"):
        recall.handoff_recall(plan, root, bundle, tokenizer_snapshot=tmp_path)
    rebank(payload_sha256=tensor_content_sha256(torch.ones(1, 8, dtype=torch.bfloat16)))
    with pytest.raises(RuntimeError, match="manifest differs from the banked payload"):
        recall.handoff_recall(plan, root, bundle, tokenizer_snapshot=tmp_path)
