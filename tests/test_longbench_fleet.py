"""The chain adapter and its seats, the terminal chain report, and the rescore rule."""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import transformers
from tests.engine_fakes import FakeEngine, embeds_request_factory, request_factory
from tests.longbench_coa_support import (
    CHAIN_FIXTURE,
    CHAIN_PROFILE,
    COMMIT,
    PROFILE_KEY,
    REDRAW_SCRIPT,
    AnswerTokenizer,
    ChainTracker,
    FakeTokenizer,
    bank_rows,
    chain_adapter,
    chain_context,
    chain_item,
    run_chain_receiver,
    seal_chain_panel,
    warm_chain_receiver,
)

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.fanoutqa.payload import read_payload
from rcc.benchmarks.longbench_v2 import LONGBENCH_COA_EASY50_BOUNDED, LONGBENCH_COA_EASY50_TEXT
from rcc.benchmarks.longbench_v2 import prepare as chain_prepare
from rcc.benchmarks.longbench_v2 import prompts as longbench_prompts
from rcc.benchmarks.longbench_v2.bootstrap import stratified_paired_bootstrap
from rcc.benchmarks.longbench_v2.geometry import REWRITE_WRAPPER_ALLOWANCE, rewrite_prompt_ceiling
from rcc.benchmarks.longbench_v2.registration import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.hardware.fleet import FleetPlacement
from rcc.hardware.placements import CHAIN_ARM_PLACEMENTS, placement_table
from rcc.hardware.qwen_manifest import placement_fingerprint, provisional_placements
from rcc.models.qwen import QWEN_FAMILY
from rcc.models.qwen import chain_text as qwen_chain_text
from rcc.models.qwen.chain_text import (
    QwenNotesChainBundle,
    generate_notes_chain,
    validate_notes_chain_fields,
)
from rcc.models.qwen.engine import EngineHandle
from rcc.models.qwen.prompts import manager_prompt_record
from rcc.models.qwen.receiver import QwenVisibleAnswer
from rcc.models.qwen.results import (
    build_result_row,
    headline_score_field,
    validate_and_rescore_result,
)
from rcc.models.route import tokenizer_decoder
from rcc.run import plan as plan_module
from rcc.run import rescore
from rcc.run.fanout_banks import bank_paths
from rcc.run.fleet.contract import WorkItem
from rcc.run.fleet.latency import ItemLatency
from rcc.run.fleet.merge import paired_deltas, score_grid, scored_fields
from rcc.run.fleet.report import add_paper_metrics
from rcc.run.plan import load_and_resolve
from rcc.run.qwen import adapter as qwen_adapter
from rcc.run.qwen import chain_adapter as chain_adapter_module
from rcc.run.qwen import chain_producers as chain_producers_module
from rcc.run.qwen import chain_report as chain_report_module
from rcc.run.qwen import report as report_module
from rcc.run.qwen.adapter import (
    FanoutQAAdapter,
    build_adapter,
    execution_profile,
    execution_qids,
    execution_roster_fingerprint,
)
from rcc.run.qwen.chain_adapter import (
    ChainAdapter,
    chain_placement_fingerprint,
    chain_roster_fingerprint,
)
from rcc.run.qwen.chain_producers import QWEN_CHAIN_REQUEST_DEADLINE_SECONDS
from rcc.run.qwen.chain_report import (
    chain_failures,
    chain_floors,
    chain_strata,
    failure_rate,
)
from rcc.run.qwen.report import build_qwen_report, read_qwen_result_rows, report_schema
from rcc.run.rescore import rescore_qwen_rows
from rcc.topologies.chain import HOPS
from rcc.topologies.chain.layout import CHAIN_TERMINAL_LAYOUT

# --- The adapter, the producer seats, the receiver, one result row.

_LATENT_ARM = "latent_query_support_rerank_r4"


_TEXT_ARM = "text_medium"


_FLOOR_ARM = "issue_only"


class _Sender:
    """Ship one scripted rewrite per draw and echo the prompt ids it consumed."""

    def __init__(self) -> None:
        """Bind the fake tokenizer at the width the seat renders through."""
        self.tokenizer = AnswerTokenizer(width=2)
        self.prompts: list[str] = []

    def decode_text_full(self, prompts: Sequence[str], requests: Sequence[Any]) -> list[Any]:
        """Answer the one rewrite request of one hop draw."""
        self.prompts.append(prompts[0])
        note = f"<think>x</think>notes after part {len(self.prompts)}"
        ids = (700_001, QWEN_FAMILY.stop_token_ids[0])
        return [
            SimpleNamespace(
                request_id=requests[0].request_id,
                prompt_token_ids=tuple(self.tokenizer.encode(prompts[0])),
                text=note,
                n_tokens=len(ids),
                token_ids=ids,
                finish_reason="stop",
                num_cached_tokens=0,
                queued_ts=None,
                scheduled_ts=None,
                first_token_ts=None,
            )
        ]

    def decode_token_ids_full(
        self, prompts: Sequence[Sequence[int]], requests: Sequence[Any]
    ) -> list[Any]:
        """Refuse: every scripted draw closes its own thinking block."""
        del prompts, requests
        raise AssertionError("the scripted chain sender closes every draw")

    def close(self) -> None:
        """Release nothing; the sender holds no engine."""

    def warmup_decode(self, prompt: str, *, seed: int) -> None:
        """Accept the seat's one warm decode."""
        del prompt, seed


def _seal_panel(root: Path) -> BenchmarkProfile:
    """Seal eight tiny chain items as this run's prepared production panel.

    The roster is the fixture's, the floor and the text arms beside the six
    rerank arms, so the latent, text, and floor seats all walk one panel.
    """
    items = tuple(chain_item(f"chain{index}", 65 + index) for index in range(8))
    qids = tuple(item.qid for item in items)
    profile = replace(
        CHAIN_FIXTURE,
        question_ids=qids,
        declared_passes=(),
    )
    seal_chain_panel(root, items=items, profile=profile)
    return profile


def _adapter(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ChainAdapter, tuple[str, ...]]:
    """Build the chain adapter and load the sealed roster through it."""
    profile = _seal_panel(root)
    adapter, manifest = chain_adapter(root, monkeypatch, profile=profile)
    assert manifest["item_count"] == 8
    assert adapter.data.qids == profile.question_ids
    # Two characters per row: at the ladder's smallest ratio of 4 the cut then
    # clears rows a four-wide tokenizer would leave protected. Every seat opens
    # its tokenizer lazily, so patching it here reaches all of them.
    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *_a, **_k: AnswerTokenizer(width=2)),
    )
    return adapter, adapter.data.qids


def test_the_chain_latent_arm_reaches_a_banked_row_through_its_own_seats(
    tmp_path: Path, tiny_model: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One item, one producer seat, four hops, one terminal block, one result row."""
    adapter, qids = _adapter(tmp_path, monkeypatch)
    qid = qids[0]
    from rcc.models.nemotron import NEMOTRON_FAMILY
    from rcc.models.nemotron.producer import NemotronProducer
    from rcc.models.qwen.backend import QwenVllmBackend

    monkeypatch.setattr(chain_producers_module, "VllmHybridPrefill", lambda llm, config: llm)
    native_backend = cast(
        QwenVllmBackend, SimpleNamespace(resident_model=lambda: tiny_model, raw_llm=object())
    )
    native = chain_producers_module.chain_engine_producer(native_backend, NEMOTRON_FAMILY)
    assert isinstance(native, NemotronProducer)
    assert native.model is tiny_model and native.latent_steps == 40
    assert native.realign is not None
    handle = EngineHandle(
        engine=FakeEngine(tiny_model),
        model=tiny_model,
        request_factory=request_factory,
        embeds_request_factory=embeds_request_factory,
    )
    monkeypatch.setattr(
        chain_producers_module,
        "build_qwen_backend",
        lambda *_a, **_k: SimpleNamespace(engine_handle=lambda: handle, close=lambda: None),
    )
    banked: list[dict[str, Any]] = []
    producer = cast(
        Any,
        adapter.build_worker(
            chain_context(adapter, tmp_path, arm=_LATENT_ARM, worker_index=0), bank_rows(banked)
        ),
    )
    # The seat names its own request deadline rather than inheriting the engine
    # producer default. Patching the constant to a different number shows the
    # seat reads it, rather than landing on a default that matches it.
    assert QWEN_CHAIN_REQUEST_DEADLINE_SECONDS == 600.0
    monkeypatch.setattr(chain_producers_module, "QWEN_CHAIN_REQUEST_DEADLINE_SECONDS", 1200.0)
    producer.warm()
    assert producer.producer._deadline_seconds == 1200.0
    artifact = producer.produce(WorkItem(qid, 0))
    producer.close()
    assert banked[0]["phase"] == "qwen_chain_latent_producer_warm"

    manifest = read_payload(artifact.path, qid=qid)
    meta = cast(dict[str, Any], manifest["meta"])
    assert meta["payload_layout"] == CHAIN_TERMINAL_LAYOUT
    assert len(meta["hop_budgets"]) == HOPS
    assert meta["kept_rows_by_hop"] == [len(keep) for keep in meta["keeps_by_worker"]]
    assert meta["latent_rows_by_worker"] == [meta["latent_tokens"]]
    rows = torch.load(
        Path(str(cast(dict[str, Any], manifest["files"])["embedding_rows"]["path"])),
        map_location="cpu",
        weights_only=True,
    )
    assert int(rows.shape[0]) == meta["latent_tokens"]
    assert set(cast(dict[str, Any], manifest["files"])) == {
        "embedding_rows",
        *(f"score_h{hop}" for hop in range(1, HOPS + 1)),
    }

    warm_chain_receiver(monkeypatch, tiny_model, ChainTracker(dict.fromkeys(qids, "B")))
    row = run_chain_receiver(adapter, tmp_path, arm=_LATENT_ARM, qid=qid, artifact=artifact)
    assert row["kind"] == "result"
    assert (row["correct"], row["answered"]) == (1.0, 1.0)
    assert row["gold"] == "B" and row["extracted_choice_by_sample"] == ["B"] * 3
    # A chain latent row records the four hop prompts the producer rendered,
    # not a fan-out roster of three.
    assert row["worker_prompt_tokens"] == list(meta["hop_prompt_rows"])
    assert row["producer_backend"] == "engine"
    assert row["latent_tokens"] == meta["latent_tokens"]
    assert row["hop_budgets"] == meta["hop_budgets"]
    assert row["producer_route"] == meta["producer_route"]

    # A seat's engine open retries a distributed-init port collision, sleeping
    # longer each time, and re-raises anything else at once.
    slept: list[float] = []
    calls = {"n": 0}

    def collide_twice() -> Any:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("The server socket has failed to listen: EADDRINUSE")
        return "engine"

    assert chain_producers_module._open_engine(collide_twice, sleep=slept.append) == "engine"
    assert calls["n"] == 3 and slept == [5.0, 10.0]

    def collide_always() -> Any:
        raise RuntimeError("EADDRINUSE, address already in use")

    with pytest.raises(RuntimeError, match="EADDRINUSE"):
        chain_producers_module._open_engine(collide_always, sleep=slept.append)
    assert slept == [5.0, 10.0, 5.0, 10.0]

    def other_failure() -> Any:
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        chain_producers_module._open_engine(other_failure, sleep=slept.append)
    assert slept == [5.0, 10.0, 5.0, 10.0]


def test_the_chain_text_arm_ships_one_ticket_and_banks_the_prompt_it_read(
    tmp_path: Path, tiny_model: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four rewrites on one sender seat, one ticket, one banked result row.

    The text chain's deliverable is a single note, so the manifest carries one
    report where FanOutQA carries three, beside each hop's prompt and part.
    """
    adapter, qids = _adapter(tmp_path, monkeypatch)
    qid = qids[1]
    sender = _Sender()
    monkeypatch.setattr(chain_producers_module, "build_qwen_backend", lambda *_a, **_k: sender)
    banked: list[dict[str, Any]] = []
    producer = cast(
        Any,
        adapter.build_worker(
            chain_context(adapter, tmp_path, arm=_TEXT_ARM, worker_index=0), bank_rows(banked)
        ),
    )
    producer.warm()
    artifact = producer.produce(WorkItem(qid, 0))
    producer.close()
    assert banked[0]["phase"] == "qwen_chain_text_producer_warm"
    # The seat reads its sender through the sender's tokenizer and the sealed
    # parts through the profile-pinned one, opened at warm beside it, so the
    # body a hop reads is the ledger's decode on every arm.
    assert producer.tokenizer_checkpoint == "Qwen/Qwen3-4B"
    assert producer.ledger_checkpoint == QWEN_FAMILY.profile.tokenizer
    assert producer.ledger_revision == QWEN_FAMILY.profile.tokenizer_revision

    manifest = read_payload(artifact.path, qid=qid)
    bundle = cast(dict[str, Any], manifest["report_bundle"])
    assert bundle["reports"] == ["notes after part 4"]
    assert bundle["notes_by_hop"] == [f"notes after part {hop}" for hop in range(1, HOPS + 1)]
    assert bundle["draws_by_hop"] == [1] * HOPS and not bundle["report_failed"]
    assert len(sender.prompts) == HOPS
    assert "notes after part 3" in sender.prompts[-1]
    # A text seat banks the prompt its sender consumed, hop by hop, and never
    # the hop prompt a latent seat prefills.
    assert bundle["report_prompt_tokens_by_hop"] == [
        len(sender.tokenizer.encode(prompt)) for prompt in sender.prompts
    ]

    warm_chain_receiver(monkeypatch, tiny_model, ChainTracker(dict.fromkeys(qids, "B")))
    row = run_chain_receiver(adapter, tmp_path, arm=_TEXT_ARM, qid=qid, artifact=artifact)
    assert row["kind"] == "result"
    assert (row["correct"], row["answered"]) == (1.0, 1.0)
    assert row["reports"] == ["notes after part 4"] and row["latent_tokens"] == 0
    # The receiver merges the manifest into the fields it seeded, so the
    # backend the manifest names reaches the bank.
    assert row["producer_backend"] == "vllm"
    # The row prices that count against the part each hop re-encoded, which
    # the seat banked beside it, and the registered rewrite context above it.
    assert row["worker_prompt_tokens"] == bundle["report_prompt_tokens_by_hop"]
    assert all(
        part
        <= count
        <= rewrite_prompt_ceiling(
            chunk, report_ceiling=adapter.profile.report_ceiling, wrapper=REWRITE_WRAPPER_ALLOWANCE
        )
        for part, count, chunk in zip(
            cast(list[int], row["chunk_text_tokens"]),
            cast(list[int], row["worker_prompt_tokens"]),
            adapter.data.item(qid).chunk_tokens,
            strict=True,
        )
    )


def test_the_chain_floor_arm_counts_the_hops_it_never_ran(
    tmp_path: Path, tiny_model: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The floor arm has no producer, so the receiver renders the hop prompts itself.

    The row still states how much source a four-hop run would have read, so
    the one arm that reads nothing still has a cost in the bank.
    """

    adapter, qids = _adapter(tmp_path, monkeypatch)
    qid = qids[2]
    warm_chain_receiver(monkeypatch, tiny_model, ChainTracker(dict.fromkeys(qids, "B")))
    row = run_chain_receiver(adapter, tmp_path, arm=_FLOOR_ARM, qid=qid, artifact=None)
    item = adapter.data.item(qid)
    assert row["latent_tokens"] == 0 and "reports" not in row
    assert row["producer_backend"] == "none"
    assert len(row["worker_prompt_tokens"]) == HOPS
    assert all(
        chunk <= count
        for chunk, count in zip(item.chunk_tokens, row["worker_prompt_tokens"], strict=True)
    )
    assert (row["correct"], row["answered"]) == (1.0, 1.0)

    # The floor arm answers from the registered no-context body, never from a
    # prompt whose text block is empty. It is the reference every latent ratio
    # is read against, so the prompt it reads is pinned here.
    tokenizer = AnswerTokenizer(width=2)
    thinking = QWEN_FAMILY.profile.decode.enable_thinking
    floor_text = longbench_prompts.no_context_prompt(
        tokenizer, item.question, item.choices, enable_thinking=thinking
    )
    assert "<text>" not in floor_text
    assert row["receiver_prompt_tokens"] == len(tokenizer.encode(floor_text))
    assert (
        manager_prompt_record(
            item, tokenizer, family=QWEN_FAMILY, profile=adapter.profile, channel="floor"
        ).text
        == floor_text
    )
    # The text channel reads the answer body with its notes in the text block,
    # and a floor arm handed notes is refused rather than answered.
    assert manager_prompt_record(
        item,
        tokenizer,
        family=QWEN_FAMILY,
        profile=adapter.profile,
        reports=("running notes",),
        channel="text",
    ).text == longbench_prompts.answer_prompt(
        tokenizer,
        item.question,
        item.choices,
        notes="running notes",
        enable_thinking=thinking,
    )
    with pytest.raises(ValueError, match="floor"):
        manager_prompt_record(
            item,
            tokenizer,
            family=QWEN_FAMILY,
            profile=adapter.profile,
            reports=("running notes",),
            channel="floor",
        )

    # The seams the runner calls on this adapter: the chain's own registry
    # table, the registration key FanOutQA publishes, the sampling identity,
    # and how a bank is addressed and locked.
    policy = QWEN_FAMILY.policy(_FLOOR_ARM)
    placement = placement_table(adapter.profile.topology_key)[_FLOOR_ARM]
    # An empty run root resolves the chain table the adapter registers, with
    # no file between.
    placements = adapter.placements.resolve(tmp_path, panel="production", source_commit=COMMIT)
    assert placements == adapter.placements.registered
    assert set(placements) == set(adapter.placements.arm_names)
    assert placements[policy] == placement == FleetPlacement(8, 0, 8)
    # The fixture's ten arms, split by split: the floor and the fused primary
    # sender on eight receivers, six chain latent seats at 4P+4R, and the two
    # smaller text senders at 6P+2R.
    assert sorted((split.producers, split.receivers) for split in placements.values()) == [
        (0, 8),
        (0, 8),
        *[(4, 4)] * 6,
        (6, 2),
        (6, 2),
    ]
    registration = adapter.registration()
    assert registration["benchmark"]["profile_id"] == adapter.profile.profile_id
    # The published roster is the topology's, not the lane's: a chain run and a
    # FanOutQA run on one family never share a roster fingerprint.
    lane_roster = execution_roster_fingerprint(QWEN_FAMILY)
    assert registration["execution_roster_fingerprint"] != lane_roster
    assert (
        build_adapter(QWEN_FAMILY, profile=FANOUTQA_NATURAL_DEV50).registration()[
            "execution_roster_fingerprint"
        ]
        == lane_roster
    )
    # Both chain identities follow the profile the adapter was built with, not
    # a module constant. The placement fingerprint signs the profile's own rows
    # alone, so a seat another profile never walks leaves it untouched.
    assert chain_placement_fingerprint(CHAIN_PROFILE) == "bd8644a995435a52"
    # The fixture registers ten arms, so it signs its own rows and its own roster.
    assert registration["execution_roster_fingerprint"] == chain_roster_fingerprint(
        QWEN_FAMILY, adapter.profile
    )
    assert registration["execution_roster_fingerprint"] != chain_roster_fingerprint(
        QWEN_FAMILY, CHAIN_PROFILE
    )
    # A topology with its own table signs its own roster, and one with no
    # registered table is refused by the key it named.
    assert chain_roster_fingerprint(QWEN_FAMILY, FANOUTQA_NATURAL_DEV50) not in {
        lane_roster,
        registration["execution_roster_fingerprint"],
    }
    with pytest.raises(ValueError, match="unregistered placement topology"):
        chain_roster_fingerprint(QWEN_FAMILY, replace(CHAIN_PROFILE, topology_key="chain-t5"))
    # Two of these seams are inherited from the fan-out adapter.
    assert ChainAdapter.bank_paths is FanoutQAAdapter.bank_paths
    assert ChainAdapter.completion_key is FanoutQAAdapter.completion_key
    contract = adapter.sampling_contract()
    assert contract["sample_tags"] == list(adapter.profile.sample_tags)
    assert contract["answer_ceiling"] == adapter.profile.answer_ceiling
    assert adapter.bank_paths(
        tmp_path, panel="production", arm=policy, placement=placement
    ) == bank_paths(tmp_path / "arms" / policy)
    assert adapter.completion_key(
        {"kind": "result", "panel": "production", "qid": qid, "arm": policy}
    ) == ("production", qid, policy)
    assert adapter.completion_key({"kind": "phase"}) is None
    monkeypatch.setenv("RCC_FANOUT_PLAN_FINGERPRINT", "a" * 64)
    identity = adapter.bank_identity(
        tmp_path,
        panel="production",
        arm=policy,
        source_commit=COMMIT,
        prepared_manifest={"artifact_sha256": "0" * 64},
        placement=placement,
    )
    # A bank names the registry rows it ran, read from the same table with no
    # run root in the call.
    assert identity.fields["placement_fingerprint"] == placement_fingerprint(
        QWEN_FAMILY, adapter.profile
    )


# --- The terminal report: score columns, floors, strata, and the bootstrap.

_FLOOR_POLICY = QWEN_FAMILY.policy(_FLOOR_ARM)


_LATENT_POLICY = QWEN_FAMILY.policy(_LATENT_ARM)


#: The eight fixture items' gold letters: B four times, so the majority floor
#: is a half and sits clear of the quarter chance floor.
_GOLDS = ("B", "B", "A", "B", "C", "D", "B", "A")


#: Six of eight right, then seven of eight, so no arm's macro is degenerate.
_FLOOR_ANSWERS = ("B", "A", "A", "B", "C", "A", "B", "A")


_LATENT_ANSWERS = ("B", "B", "A", "B", "C", "D", "A", "A")


_VALUES = {"q1": 1.0, "q2": 0.0, "q3": 1.0, "q4": 0.5, "q5": 0.25, "q6": 0.0}


def test_the_stratified_bootstrap_is_seed_reproducible_and_keeps_each_stratum_size() -> None:
    """One stratum draws flat; several keep each stratum's size. Draws and seed are local."""
    flat = dict.fromkeys(_VALUES, 3)
    once = stratified_paired_bootstrap(_VALUES, flat, draws=512, seed=11)
    assert once == stratified_paired_bootstrap(_VALUES, flat, draws=512, seed=11)
    assert (once["ci_low"], once["ci_high"]) != (
        stratified_paired_bootstrap(_VALUES, flat, draws=512, seed=12)["ci_low"],
        stratified_paired_bootstrap(_VALUES, flat, draws=512, seed=12)["ci_high"],
    )
    assert once["ci_low"] <= once["mean"] <= once["ci_high"]
    assert (once["draws"], once["seed"]) == (512, 11)
    assert once["n_by_stratum"] == {"3": 6}

    deltas = [_VALUES[qid] for qid in sorted(_VALUES)]
    assert once["mean"] == sum(deltas) / len(deltas)

    banded = {"q1": 3, "q2": 3, "q3": 4, "q4": 4, "q5": 5, "q6": 5}
    stratified = stratified_paired_bootstrap(_VALUES, banded, draws=512, seed=11)
    assert stratified["n_by_stratum"] == {"3": 2, "4": 2, "5": 2}
    assert stratified["mean"] == once["mean"]
    assert (stratified["ci_low"], stratified["ci_high"]) != (once["ci_low"], once["ci_high"])
    with pytest.raises(ValueError, match="q4"):
        stratified_paired_bootstrap(_VALUES, {**banded, "q4": None}, draws=8, seed=11)  # type: ignore[dict-item]


def test_the_shared_grid_keeps_its_default_columns_and_takes_the_chain_fields() -> None:
    """The default fields are the shipped ones; the chain's fields are the caller's."""
    rows = [
        {"qid": "q1", "policy": "a", "loose": 1.0, "strict": 0.0, "generated_tokens": 10.0},
        {"qid": "q2", "policy": "a", "loose": 0.0, "strict": 0.0, "generated_tokens": 20.0},
        {"qid": "q1", "policy": "b", "loose": 1.0, "strict": 1.0, "generated_tokens": 30.0},
        {"qid": "q2", "policy": "b", "loose": 0.5, "strict": 0.0, "generated_tokens": 40.0},
    ]
    grid = score_grid(
        rows, ("a", "b"), key="policy", mean_fields=("generated_tokens",), label="Qwen"
    )
    assert grid == [
        {
            "policy": "a",
            "n": 2,
            "loose": 0.5,
            "strict": 0.0,
            "loose_item_se": 0.5,
            "strict_item_se": 0.0,
            "generated_tokens": 15.0,
        },
        {
            "policy": "b",
            "n": 2,
            "loose": 0.75,
            "strict": 0.5,
            "loose_item_se": 0.25,
            "strict_item_se": 0.5,
            "generated_tokens": 35.0,
        },
    ]
    assert list(grid[0]) == [
        "policy",
        "n",
        "loose",
        "strict",
        "loose_item_se",
        "strict_item_se",
        "generated_tokens",
    ]
    assert paired_deltas(rows, (("a", "b"),), key="policy") == [
        {
            "reference_policy": "a",
            "candidate_policy": "b",
            "n_pairs": 2,
            "loose_delta": 0.25,
            "strict_delta": 0.5,
            "loose_delta_item_se": 0.25,
            "strict_delta_item_se": 0.5,
        }
    ]

    chain = [
        {"qid": "q1", "policy": "a", "correct": 1.0, "answered": 1.0},
        {"qid": "q2", "policy": "a", "correct": 0.0, "answered": 1.0},
        {"qid": "q1", "policy": "b", "correct": 1.0, "answered": 1.0},
        {"qid": "q2", "policy": "b", "correct": 1.0, "answered": 0.0},
    ]
    fields = ("correct", "answered")
    assert score_grid(
        chain, ("a",), key="policy", mean_fields=(), label="Qwen", score_fields=fields
    ) == [
        {
            "policy": "a",
            "n": 2,
            "correct": 0.5,
            "answered": 1.0,
            "correct_item_se": 0.5,
            "answered_item_se": 0.0,
        }
    ]
    assert paired_deltas(chain, (("a", "b"),), key="policy", score_fields=fields) == [
        {
            "reference_policy": "a",
            "candidate_policy": "b",
            "n_pairs": 2,
            "correct_delta": 0.5,
            "answered_delta": -0.5,
            "correct_delta_item_se": 0.5,
            "answered_delta_item_se": 0.5,
        }
    ]


def _panel(root: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Seal the whole easy fifty as tiny items under the fixture roster, then load eight.

    The fixture carries the floor beside the rerank ladder, which no sealed
    profile does, so it stands in under the rerank key for the test's length.
    """
    items = tuple(
        replace(
            chain_item(qid, 65 + index, stratum=3 + index % 3),
            gold=_GOLDS[index % len(_GOLDS)],
        )
        for index, qid in enumerate(CHAIN_FIXTURE.question_ids)
    )
    seal_chain_panel(root, items=items, profile=CHAIN_FIXTURE)
    monkeypatch.setitem(plan_module._BENCHMARKS, PROFILE_KEY, CHAIN_FIXTURE)
    adapter, _manifest = chain_adapter(root, monkeypatch, profile=CHAIN_FIXTURE, count=8)
    # Two characters per row: at the ladder's smallest ratio of 4 the cut then
    # clears rows a four-wide tokenizer would leave protected.
    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *_a, **_k: AnswerTokenizer(width=2)),
    )
    return adapter


def _chain_bank(
    root: Path, monkeypatch: pytest.MonkeyPatch, tiny_model: Any
) -> list[dict[str, Any]]:
    """Bank the floor arm and one latent arm over the eight fixture items."""
    adapter = _panel(root, monkeypatch)
    qids = adapter.data.qids
    handle = EngineHandle(
        engine=FakeEngine(tiny_model),
        model=tiny_model,
        request_factory=request_factory,
        embeds_request_factory=embeds_request_factory,
    )
    monkeypatch.setattr(
        chain_producers_module,
        "build_qwen_backend",
        lambda *_a, **_k: SimpleNamespace(engine_handle=lambda: handle, close=lambda: None),
    )
    banked: list[dict[str, Any]] = []
    producer = cast(
        Any,
        adapter.build_worker(
            chain_context(adapter, root, arm=_LATENT_ARM, worker_index=0), bank_rows(banked)
        ),
    )
    producer.warm()
    artifacts = {qid: producer.produce(WorkItem(qid, 0)) for qid in qids}
    producer.close()

    rows: list[dict[str, Any]] = []
    for arm, answers in ((_FLOOR_ARM, _FLOOR_ANSWERS), (_LATENT_ARM, _LATENT_ANSWERS)):
        warm_chain_receiver(
            monkeypatch, tiny_model, ChainTracker(dict(zip(qids, answers, strict=True)))
        )
        for qid in qids:
            row = run_chain_receiver(
                adapter,
                root,
                arm=arm,
                qid=qid,
                artifact=artifacts[qid] if arm == _LATENT_ARM else None,
            )
            # A live row takes its panel from the bank identity; this fixture
            # drives the seats directly, so it stamps the same value.
            rows.append({**row, "panel": "production"})
    return rows


def test_the_chain_report_publishes_floors_strata_and_a_stratified_bootstrap(
    tmp_path: Path, tiny_model: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chain's schema, columns, floors, strata, intervals, and lost parts."""
    monkeypatch.setenv("RCC_FANOUT_BENCHMARK", PROFILE_KEY)
    rows = _chain_bank(tmp_path, monkeypatch, tiny_model)
    # The sealed bundle is loaded and validated once per report. The counter is
    # installed on every chain module that could load it, so a second load
    # anywhere in the chain report path is caught here.
    loads: list[Path] = []

    def _counted(root: Path, **kwargs: Any) -> Any:
        loads.append(root)
        return chain_prepare.load_prepared_panel(root, **kwargs)

    for module in (chain_adapter_module, chain_report_module):
        monkeypatch.setattr(module, "load_prepared_panel", _counted, raising=False)
    report = build_qwen_report(tmp_path, rows=rows, source_commit=COMMIT, family=QWEN_FAMILY)
    assert loads == [tmp_path]
    assert report_schema(FANOUTQA_NATURAL_DEV50, "qwen") == "qwen-fanoutqa-report-v1"
    assert report["schema"] == "qwen-longbench-coa-report-v1"
    assert report["policies"] == [_FLOOR_POLICY, _LATENT_POLICY]

    # The scored columns are the profile's, and the two floors the chain reads
    # its macro against are published beside them.
    grid = {str(row["policy"]): row for row in cast(list[Any], report["policy_grid"])}
    assert (grid[_FLOOR_POLICY]["correct"], grid[_LATENT_POLICY]["correct"]) == (0.75, 0.875)
    assert grid[_FLOOR_POLICY]["answered"] == 1.0
    # The floor runs over the whole sealed panel, not this pass's eight items,
    # so every pass of one panel publishes the same number to beat, and the
    # report names the count it ran over.
    assert report["floors"] == {"chance": 0.25, "majority_floor": 0.52, "panel_items": 50}

    # Every arm is diagnosed on all three source-length bands the fixture spans.
    strata = cast(list[Any], report["strata"])
    assert len(strata) == 6
    assert {int(row["stratum"]) for row in strata} == {3, 4, 5}
    assert sum(int(row["n"]) for row in strata if row["policy"] == _FLOOR_POLICY) == 8

    # The floor arm is the chain's second reference, because it has no mean query attention
    # arm and the one arm that reads no source is what its macro must beat.
    comparisons = cast(list[Any], report["paired_comparisons"])
    assert [(row["reference_policy"], row["candidate_policy"]) for row in comparisons] == [
        (_FLOOR_POLICY, _LATENT_POLICY)
    ]
    assert comparisons[0]["correct_delta"] == 0.125

    # A chain law arm takes the two references and nothing else: there is no
    # coa mate at the same ratio to pair against. The rule reads the law off
    # the arm, so a bounded arm takes the same two pairs as a rerank arm.
    text_policy = QWEN_FAMILY.policy("text_primary")
    assert report_module._comparison_pairs(
        (text_policy, _LATENT_POLICY, _FLOOR_POLICY), QWEN_FAMILY, CHAIN_FIXTURE
    ) == (
        (text_policy, _LATENT_POLICY),
        (_FLOOR_POLICY, _LATENT_POLICY),
    )
    bounded = LONGBENCH_COA_EASY50_BOUNDED.arms[0]
    bounded_policy = QWEN_FAMILY.policy(bounded.arm_id)
    assert (bounded.budget_law, bounded.retention_ratio) == ("bounded", None)
    assert report_module._comparison_pairs(
        (text_policy, bounded_policy, _FLOOR_POLICY), QWEN_FAMILY, LONGBENCH_COA_EASY50_BOUNDED
    ) == (
        (text_policy, bounded_policy),
        (_FLOOR_POLICY, bounded_policy),
    )
    assert all((arm.budget_law is None) == (arm.channel != "latent") for arm in CHAIN_FIXTURE.arms)

    bootstrap = cast(dict[str, Any], report["bootstrap"])
    assert [row["policy"] for row in bootstrap["arms"]] == [_FLOOR_POLICY, _LATENT_POLICY]
    assert len(bootstrap["paired_comparisons"]) == 1
    for row in (*bootstrap["arms"], *bootstrap["paired_comparisons"]):
        assert (row["draws"], row["seed"]) == (BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
        assert row["ci_low"] <= row["mean"] <= row["ci_high"]
        assert row["n_by_stratum"] == {"3": 3, "4": 3, "5": 2}

    # A chain that lost a part is scored as it stands, and the report says how
    # often that happened. Only a text arm runs the rewrite hops, so the floor
    # arm and the latent arm this bank ran both publish zero.
    assert report["chain_failures"] == [
        {"policy": _FLOOR_POLICY, "failed_items": 0, "failed_qids": [], "failed_rate": 0.0},
        {"policy": _LATENT_POLICY, "failed_items": 0, "failed_qids": [], "failed_rate": 0.0},
    ]
    # The rate is the number the grid's quarantine_rate carries, over the same
    # rows and the same flag, so neither predicate can move alone.
    assert [row["quarantine_rate"] for row in report["policy_grid"]] == [
        row["failed_rate"] for row in report["chain_failures"]
    ]

    output = tmp_path / "report"
    published = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert published == report
    assert json.loads((output / "bootstrap.json").read_text(encoding="utf-8")) == bootstrap
    written = list(csv.DictReader((output / "strata.csv").read_text(encoding="utf-8").splitlines()))
    assert [row["stratum"] for row in written] == [str(row["stratum"]) for row in strata]
    # The failure count is a report.json section and writes no file of its own.
    assert sorted(path.name for path in output.iterdir()) == [
        "bootstrap.json",
        "paired_comparisons.csv",
        "policy_grid.csv",
        "report.json",
        "selected_results.jsonl",
        "strata.csv",
    ]
    # The chain keeps its original fields. FanOutQA's separate scoring-version
    # metadata must not label a chain report as a FanOutQA result.
    assert sorted(set(report) - {"bootstrap", "chain_failures", "floors", "strata"}) == [
        "decode_fingerprint",
        "decode_profile",
        "generation_censoring",
        "paired_comparisons",
        "panel",
        "policies",
        "policy_grid",
        "prepared_sha256",
        "qids",
        "result_rows",
        "sample_aggregation",
        "schema",
        "source_commit",
    ]

    # Offline there is no seat environment to read the benchmark from, so the
    # CLI names it and holds its own run id to it. Reporting a chain run as
    # FanOutQA would load the wrong panel, so the run id stem refuses first.
    stem = f"qwen-{CHAIN_FIXTURE.run_id_stem}-fixture"
    fanout_stem = f"qwen-{FANOUTQA_NATURAL_DEV50.run_id_stem}-other"
    assert CHAIN_FIXTURE.run_id_stem == CHAIN_PROFILE.run_id_stem
    assert report_module._cli_benchmark(stem, PROFILE_KEY) is CHAIN_FIXTURE
    assert report_module._cli_benchmark(stem, None) is CHAIN_FIXTURE
    monkeypatch.delenv("RCC_FANOUT_BENCHMARK")
    with pytest.raises(ValueError, match="RCC_FANOUT_BENCHMARK"):
        report_module._cli_benchmark(fanout_stem, None)
    fanout_key = FANOUTQA_NATURAL_DEV50.benchmark_key
    assert report_module._cli_benchmark(fanout_stem, fanout_key) is FANOUTQA_NATURAL_DEV50
    with pytest.raises(ValueError, match=FANOUTQA_NATURAL_DEV50.run_id_stem):
        report_module._cli_benchmark(stem, fanout_key)
    with pytest.raises(ValueError, match=CHAIN_FIXTURE.run_id_stem):
        report_module.main(
            [
                "--run-root",
                str(tmp_path / fanout_stem),
                "--run-id",
                fanout_stem,
                "--source-commit",
                COMMIT,
                "--benchmark",
                PROFILE_KEY,
            ]
        )

    # The report names its item range as a pair inside the panel, and neither
    # half of it stands alone.
    assert report_module.offline_item_range(CHAIN_FIXTURE, 0, 50) == (0, 50)
    assert report_module.offline_item_range(CHAIN_FIXTURE, 0, 15) == (0, 15)
    assert report_module.offline_item_range(CHAIN_FIXTURE, None, None) is None
    with pytest.raises(ValueError, match="outside the"):
        report_module.offline_item_range(CHAIN_FIXTURE, 0, 51)
    with pytest.raises(ValueError, match="item-start"):
        report_module.offline_item_range(CHAIN_FIXTURE, 0, None)

    # The named pass is published where every seam reads it, and only for the
    # length of that pass: a report that left its benchmark behind would rename
    # the next pass in the same process.
    for name in ("RCC_FANOUT_ITEM_START", "RCC_FANOUT_ITEM_COUNT"):
        monkeypatch.delenv(name, raising=False)
    before = dict(os.environ)
    with report_module.execution_scope(CHAIN_FIXTURE, (0, 50)):
        assert execution_profile() is CHAIN_FIXTURE
        assert execution_qids(profile=CHAIN_FIXTURE) == tuple(CHAIN_FIXTURE.question_ids[:50])
    assert dict(os.environ) == before

    production = f"qwen-{CHAIN_FIXTURE.run_id_stem}-production"
    seen: list[dict[str, str]] = []

    def _named(root: Path, *, family: Any) -> list[Any]:
        assert (root.name, family) == (production, QWEN_FAMILY)
        seen.append({name: value for name, value in os.environ.items() if name not in before})
        raise RuntimeError("the durable rows are pinned")

    monkeypatch.setattr(report_module, "read_qwen_result_rows", _named)
    argv = [
        "--run-root",
        str(tmp_path / production),
        "--run-id",
        production,
        "--source-commit",
        COMMIT,
        "--benchmark",
        PROFILE_KEY,
    ]
    with pytest.raises(RuntimeError, match="the durable rows are pinned"):
        report_module.main([*argv, "--item-start", "0", "--item-count", "50"])
    assert seen == [
        {
            "RCC_FANOUT_BENCHMARK": PROFILE_KEY,
            "RCC_FANOUT_ITEM_START": "0",
            "RCC_FANOUT_ITEM_COUNT": "50",
        }
    ]
    assert dict(os.environ) == before
    with pytest.raises(ValueError, match="outside the"):
        report_module.main([*argv, "--item-start", "0", "--item-count", "51"])


def test_the_report_seals_the_fanoutqa_pair_and_names_its_headline_score() -> None:
    """The scored columns, the headline score, the floor, and the strata roster.

    FanOutQA's `n_leaves` is a per-sample record, not a scored column, and a
    strata table short of a loaded item is refused by the items it misses.
    """
    assert FANOUTQA_NATURAL_DEV50.score_fields == ("loose", "strict", "n_leaves")
    assert scored_fields(FANOUTQA_NATURAL_DEV50.score_fields) == ("loose", "strict")
    assert CHAIN_PROFILE.score_fields == ("correct", "answered")
    assert scored_fields(CHAIN_PROFILE.score_fields) == CHAIN_PROFILE.score_fields
    assert headline_score_field(FANOUTQA_NATURAL_DEV50) == "loose"
    assert headline_score_field(CHAIN_PROFILE) == "correct"

    assert chain_floors(("A", "A", "B", "C")) == {
        "chance": 0.25,
        "majority_floor": 0.5,
        "panel_items": 4,
    }
    assert chain_floors(("A", "A"))["majority_floor"] == 1.0

    strata_by_qid = {"q1": 3, "q2": 3, "q3": 4}
    rows = [{"policy": "p", "qid": qid, "correct": 1.0} for qid in ("q1", "q2", "q3")]
    assert [row["n"] for row in chain_strata(rows, ("p",), strata_by_qid, "correct")] == [2, 1]
    with pytest.raises(RuntimeError, match="q2"):
        chain_strata(rows[:1] + rows[2:], ("p",), strata_by_qid, "correct")

    # A lost part is counted, not dropped: the failed rows stay in the macro
    # above, and the rate is what shows a text arm converging on the floor.
    failures = [
        {"policy": "text", "qid": "q3", "report_failed": False},
        {"policy": "text", "qid": "q2", "report_failed": True},
        {"policy": "text", "qid": "q1", "report_failed": True},
        *({"policy": "latent", "qid": qid} for qid in ("q1", "q2", "q3")),
    ]
    assert chain_failures(failures, ("text", "latent")) == [
        {"policy": "text", "failed_items": 2, "failed_qids": ["q1", "q2"], "failed_rate": 0.6667},
        {"policy": "latent", "failed_items": 0, "failed_qids": [], "failed_rate": 0.0},
    ]

    # `failed_rate` and the grid's `quarantine_rate` are the same share of the
    # same rows, so one helper forms both. Two float expressions over it first
    # disagree at 160 items, so the panels below run past the sizes in use.
    panels = [(items, lost) for items in range(1, 13) for lost in range(items + 1)]
    panels += [(160, lost) for lost in (1, 3, 11, 19, 21)]
    for items, lost in panels:
        rows = [
            {"policy": "text", "qid": f"q{index}", "report_failed": index < lost}
            for index in range(items)
        ]
        grid: list[dict[str, Any]] = [{"policy": "text"}]
        add_paper_metrics(grid, rows, key="policy")
        assert grid[0]["quarantine_rate"] == failure_rate(rows)
        assert chain_failures(rows, ("text",))[0]["failed_rate"] == failure_rate(rows)


def test_the_report_reader_classifies_channels_by_the_table_its_run_seated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The split reader is handed the placements of the topology it reads.

    The merge uses the placement table to decide which arm's channel is direct,
    so the table comes off the adapter of the benchmark this worker runs.
    """
    seen: list[Mapping[str, Any]] = []

    def _record(root: Path, placements: Mapping[str, Any], *, label: str) -> list[Any]:
        assert (root, label) == (tmp_path, QWEN_FAMILY.lane.capitalize())
        seen.append(placements)
        return []

    monkeypatch.setattr(report_module, "read_split_result_rows", _record)
    monkeypatch.setenv("RCC_FANOUT_BENCHMARK", PROFILE_KEY)
    assert read_qwen_result_rows(tmp_path, family=QWEN_FAMILY) == []
    monkeypatch.setenv("RCC_FANOUT_BENCHMARK", FANOUTQA_NATURAL_DEV50.benchmark_key)
    assert read_qwen_result_rows(tmp_path, family=QWEN_FAMILY) == []

    chain, fanout = seen
    # The chain table cut to the arms this profile registers: the lane also
    # implements the floor, the text arms, the fan-out ladder, and the bounded
    # ladder, which this profile does not run, so six rerank seats remain.
    registered = {arm.arm_id for arm in CHAIN_PROFILE.arms}
    assert chain == {
        arm.policy: CHAIN_ARM_PLACEMENTS[arm.semantic_arm]
        for arm in QWEN_FAMILY.profile.physical_arms
        if arm.semantic_arm in registered
    }
    assert len(chain) == 6 and all("rerank" in policy for policy in chain)
    assert fanout == provisional_placements(family=QWEN_FAMILY)
    # The fan-out table seats no rerank arm at all, so a chain row on this
    # profile could not be priced on the fan-out split even by accident.
    assert _LATENT_POLICY in chain and _LATENT_POLICY not in fanout


# --- The rescore rule: its ladder, its raw hop ids, its macro.

_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


#: Every row here is a text row, so the profile is the floor-and-text one; the
#: rerank config is loaded once below to show the command reads its profile.
_TEXT_CONFIG = "qwen-longbench-coa-easy50-text-n50.toml"


_RERANK_CONFIG = "qwen-longbench-coa-easy50-rerank-n50.toml"


_QID = LONGBENCH_COA_EASY50_TEXT.question_ids[0]


_ARM = "text_medium"


_ITEM = chain_item(_QID, 65)


_ANSWER = "The correct answer is (B)"


_RAW = f"<think>x</think>{_ANSWER}"


class _CodePointTokenizer(FakeTokenizer):
    """The shared chain fake whose decode inverts its own encode.

    The rescore reads a banked draw back from its ids alone, so the fake is a
    real inverse over one alphabet: one code point per character.
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Return one id per character."""
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, tokens: Any, **kwargs: Any) -> str:
        """Return the ids read back as their own characters."""
        del kwargs
        return "".join(chr(int(token)) for token in tokens)


class _ChainSender:
    """Replay one scripted rewrite per draw, banking the ids of its own text."""

    def __init__(self, script: tuple[str, ...]) -> None:
        """Script one draw per call and open the module clock at zero."""
        self.script = list(script)
        self.tokenizer = _CodePointTokenizer()
        self.prompts: list[str] = []
        self.clock = 0.0

    def decode_text_full(self, prompts: Any, requests: Any) -> list[Any]:
        """Answer the one rewrite request of one hop draw."""
        self.prompts.append(prompts[0])
        self.clock += 1.0
        text = str(self.script.pop(0))
        ids = (*(ord(char) for char in text), QWEN_FAMILY.stop_token_ids[0])
        return [
            SimpleNamespace(
                request_id=requests[0].request_id,
                prompt_token_ids=tuple(self.tokenizer.encode(prompts[0])),
                text=text,
                n_tokens=len(ids),
                token_ids=ids,
                finish_reason="stop",
                num_cached_tokens=0,
                queued_ts=0.0,
                scheduled_ts=1.0,
                first_token_ts=2.0,
            )
        ]

    def decode_token_ids_full(self, prompts: Any, requests: Any) -> list[Any]:
        """Refuse: every scripted draw closes its own thinking block."""
        del prompts, requests
        raise AssertionError("the scripted chain closes every draw")


def _bundle(monkeypatch: pytest.MonkeyPatch) -> tuple[QwenNotesChainBundle, _ChainSender]:
    """Rewrite the four hops on one scripted sender, with one redraw on hop two."""
    sender = _ChainSender(REDRAW_SCRIPT)
    monkeypatch.setattr(qwen_chain_text.time, "perf_counter", lambda: sender.clock)
    bundle = generate_notes_chain(
        cast(Any, sender),
        sender.tokenizer,
        _ITEM,
        qid=_QID,
        semantic_arm=_ARM,
        family=QWEN_FAMILY,
        decoder=tokenizer_decoder(sender.tokenizer),
        profile=LONGBENCH_COA_EASY50_TEXT,
        ledger_tokenizer=sender.tokenizer,
    )
    return bundle, sender


def _answers() -> list[QwenVisibleAnswer]:
    """Return the three independent receiver samples of one chain item."""
    ids = tuple(ord(char) for char in _RAW)
    return [
        QwenVisibleAnswer(
            tag=tag,
            seed=seed,
            backend_text=_RAW,
            raw_text=_RAW,
            visible_text=_ANSWER,
            token_ids=(*ids, QWEN_FAMILY.stop_token_ids[0]),
            visible_decode_token_ids=ids,
            finish_reason="stop",
            thinking_closed=True,
            num_cached_tokens=0,
        )
        for tag, seed in zip(
            LONGBENCH_COA_EASY50_TEXT.sample_tags,
            LONGBENCH_COA_EASY50_TEXT.answer_seeds(_QID, _ARM),
            strict=True,
        )
    ]


def _result_row(bundle: QwenNotesChainBundle) -> dict[str, Any]:
    """Bank one chain text result row over the four hops this bundle shipped."""
    fields = bundle.result_fields()
    latency = ItemLatency(
        producer_s=4.0,
        receiver_prepare_s=0.25,
        decode_s=1.0,
        decode_batched_s=0.5,
        receiver_ttft_s=0.4,
        generation_s=0.6,
        queued_offsets=(0.0, 1.0, 1.0),
        first_token_offsets=(0.4, 1.4, 1.4),
        finished_offsets=(1.0, 1.5, 1.5),
    )
    prompt = cast(
        Any,
        SimpleNamespace(
            prompt_rows=8,
            manager_tokens=8,
            payload_rows=0,
            payload_layout=None,
            payload_semantic_arm=None,
            payload_plan_sha256=None,
            payload_tensor_sha256=None,
        ),
    )
    return build_result_row(
        _ITEM,
        _ARM,
        _answers(),
        prompt,
        worker_prompt_tokens=cast(list[int], fields["report_prompt_tokens_by_hop"]),
        latency=latency,
        family=QWEN_FAMILY,
        channel_fields={key: value for key, value in fields.items() if key != "decode"},
        profile=LONGBENCH_COA_EASY50_TEXT,
    )


def _plan() -> Any:
    """Return a plan stand-in: the rescore rule reads the benchmark alone."""
    return cast(Any, SimpleNamespace(benchmark=LONGBENCH_COA_EASY50_TEXT))


def test_a_chain_text_row_rescores_from_its_raw_hop_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The four notes, their survivors, their seeds, and the ticket, from ids alone."""
    bundle, sender = _bundle(monkeypatch)
    fields = bundle.result_fields()
    assert len(cast(list[str], fields["notes_by_hop"])) == HOPS
    assert fields["draws_by_hop"] == [1, 2, 1, 1] and fields["draws_n"] == 5
    assert fields["notes_read_by_hop"] == ["", "note 1", "note 2", "note 3"]
    validate_notes_chain_fields(fields)
    for ladder_message, ladder_changes in (
        ("draw count differs from its hop ladder", {"draws_n": 6}),
        ("seed tag", {"report_seed_tags_by_hop": ["s2", "s1", "s0", "s0"]}),
        ("injection summary", {"injected": True}),
        ("notes each hop read", {"notes_read_by_hop": ["", "note 1", "", "note 3"]}),
        ("redraw wall clock differs", {"redraw_wall_s": 0.0}),
        ("one ticket", {"reports": ["note 3"]}),
    ):
        with pytest.raises(ValueError, match=ladder_message):
            validate_notes_chain_fields({**fields, **ladder_changes})

    row = _result_row(bundle)
    assert "scoring_version" not in row
    rescore._require_qwen_notes_chain(
        row,
        _plan(),
        qid=_QID,
        arm=_ARM,
        tokenizer=sender.tokenizer,
        family=QWEN_FAMILY,
    )
    # The row rule runs the chain's ladder check, so the whole result row
    # rescores where FanOutQA's three-worker rule would refuse it.
    rescored = validate_and_rescore_result(
        row, _ITEM, sender.tokenizer, family=QWEN_FAMILY, profile=LONGBENCH_COA_EASY50_TEXT
    )
    assert (rescored["correct"], rescored["answered"]) == (1.0, 1.0)
    assert rescored["reports"] == ["note 4"]

    seeds = cast(list[int], row["report_seeds"])
    for message, changes in (
        # The survivor moves with the note, so the raw ids, not the ladder,
        # refuse a note nobody could have written.
        (
            "notes_by_hop",
            {
                "notes_by_hop": ["note 1", "other", "note 3", "note 4"],
                "notes_read_by_hop": ["", "note 1", "other", "note 3"],
            },
        ),
        ("report_seeds", {"report_seeds": [*seeds[:2], seeds[2] + 1, seeds[3]]}),
        ("length finish", {"report_finish_reasons": ["stop", "stop", "stop", "length"]}),
    ):
        with pytest.raises(RuntimeError, match=message):
            rescore._require_qwen_notes_chain(
                {**row, **changes},
                _plan(),
                qid=_QID,
                arm=_ARM,
                tokenizer=sender.tokenizer,
                family=QWEN_FAMILY,
            )


def test_the_macro_gate_and_the_report_rule_read_the_registered_profile() -> None:
    """The macro runs over the profile's score fields; an unknown builder is refused."""
    policies = [LONGBENCH_COA_EASY50_TEXT.sealed_qwen_policy(arm) for arm in ("issue_only", _ARM)]
    rows = [{"qid": _QID, "arm": policy, "correct": 1.0, "answered": 1.0} for policy in policies]
    fields = LONGBENCH_COA_EASY50_TEXT.score_fields
    assert fields == ("correct", "answered")
    rescore._require_no_vote_macro(rows, (_QID,), policies, fields)
    with pytest.raises(RuntimeError, match="cannot form a correct macro"):
        rescore._require_no_vote_macro(rows, (_QID,), [*policies, "absent-policy"], fields)

    # FanOutQA registers `n_leaves` beside its scored pair as a per-sample
    # record of question width. A macro over a question-width count proves
    # nothing about a rescore, so the rule never asks these rows for one.
    assert FANOUTQA_NATURAL_DEV50.score_fields == ("loose", "strict", "n_leaves")
    scored = [{"qid": _QID, "arm": policy, "loose": 1.0, "strict": 1.0} for policy in policies]
    rescore._require_no_vote_macro(scored, (_QID,), policies, FANOUTQA_NATURAL_DEV50.score_fields)

    plan = load_and_resolve(_CONFIGS / _TEXT_CONFIG, git_commit="0" * 40)
    assert plan.benchmark is LONGBENCH_COA_EASY50_TEXT
    foreign = replace(
        plan, benchmark=replace(LONGBENCH_COA_EASY50_TEXT, prompt_builder="other-builder-v1")
    )
    with pytest.raises(RuntimeError, match="other-builder-v1"):
        rescore_qwen_rows(
            [],
            foreign,
            items_by_qid={},
            receiver_tokenizer=None,
            report_tokenizers={},
        )


def test_the_rescore_command_loads_its_panel_through_the_benchmarks_adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One loader seam, chosen by the profile the plan resolved."""
    plan = load_and_resolve(_CONFIGS / _RERANK_CONFIG, git_commit="0" * 40)
    assert plan.benchmark is CHAIN_PROFILE
    built: list[Any] = []
    named: list[dict[str, str]] = []
    before = dict(os.environ)

    class _Recording:
        """Record the profile the command built an adapter for, then stop."""

        def __init__(self, family: Any, *, profile: Any) -> None:
            built.append(profile)
            named.append({name: value for name, value in os.environ.items() if name not in before})

        @property
        def data(self) -> Any:
            raise RuntimeError("the prepared loader is pinned")

    monkeypatch.setattr(qwen_adapter, "build_adapter", _Recording)
    args = SimpleNamespace(runtime_bank=tmp_path, partition_runtime_bank=[])
    with pytest.raises(RuntimeError, match="the prepared loader is pinned"):
        rescore._rescore_qwen_command(
            cast(Any, args), plan, {"results": tmp_path / "selected_results.jsonl"}
        )
    assert built == [plan.benchmark]
    # A subsetting loader selects the pass by the benchmark and the item range
    # a seat reads from the environment. A rescore holds no seat environment,
    # so the plan's own pair is published for the read and taken back after.
    assert named == [
        {
            "RCC_FANOUT_BENCHMARK": plan.benchmark.benchmark_key,
            "RCC_FANOUT_ITEM_START": str(plan.item_start),
            "RCC_FANOUT_ITEM_COUNT": str(plan.item_count),
        }
    ]
    assert dict(os.environ) == before
