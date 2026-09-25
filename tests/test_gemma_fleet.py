"""Gemma on the split fleet: the adapter, the engine roles, the seats, and arm retry."""

from __future__ import annotations

import argparse
import json
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from tests.gemma_split_support import (
    _ARM,
    _ATTEMPT_ID,
    _FLOOR_ARM,
    _QIDS,
    _SOURCE_COMMIT,
    _SPLIT_PLACEMENT,
    _TABLE_SHAPE,
    _Bank,
    _drop_first_capture_row,
    _failing_capture,
    _prepared_text,
    _producer,
    _receiver,
    _rolled_tensor,
    _text_producer,
    _text_rows,
    _warm_real_split_receiver,
    _write_rolled,
    adapter,
    fixture_adapter,
    publish_table,
)
from tests.gemma_text_support import assert_gemma_text_contract
from tests.longbench_coa_support import FakeTokenizer
from tests.natural_bundle import _one_item_config, bank_result_rows, rebanker, write_bundle

import rcc.models.gemma.tokenizer as gemma_tokenizer
from rcc.benchmarks.fanoutqa import gemma_natural
from rcc.hardware.fleet import FleetPlacement
from rcc.hardware.placements import SEMANTIC_ARM_PLACEMENTS, bind_policy_placements
from rcc.models.gemma.capture_support import DeferredAudit, latent_roll_filename
from rcc.models.gemma.config import GemmaFleetConfig
from rcc.models.gemma.contract import LATENT_STEPS, MAX_NUM_SEQS
from rcc.models.gemma.engine_roles import (
    CAPTURE_ENGINE_IDENTITY,
    PRODUCER_ROLE,
    RECEIVER_ROLE,
    SPLIT_RECEIVER_ENGINE_IDENTITY,
    engine_config,
    registered_identity,
)
from rcc.models.gemma.handoff import SELECTION_KEY, GemmaHandoffError, handoff_manifest_path
from rcc.models.gemma.layout import flat_interleave_layout
from rcc.models.gemma.parity import selected_indices_sha256
from rcc.models.gemma.roster import gemma_semantic_bindings, gemma_v16_arm
from rcc.models.gemma.selection import SELECTION_SCHEMA, load_selection, payload_length
from rcc.run import handoff_recall as recall
from rcc.run import plan as plan_module
from rcc.run.barrier import SPLIT_FLEET_ISOLATION_PROFILE, BarrierSpec
from rcc.run.config import load_run_config
from rcc.run.contract import BankIdentity
from rcc.run.fleet.clocks import clock_record_path, read_bundle_clocks
from rcc.run.fleet.contract import WorkItem
from rcc.run.fleet.latency import ITEM_LATENCY_FIELDS
from rcc.run.fleet.merge import cell_ledger, merged_split_rows, read_arm_bank_rows
from rcc.run.fleet.report import collapse_cells
from rcc.run.fleet.runtime import FleetRunConfig, FleetWorkerHooks, run_worker
from rcc.run.fleet.schedule import (
    SplitScheduleConfig,
    driver_lock,
    run_in_threads,
    run_split_fleet,
    warmup_contract,
)
from rcc.run.gemma.split_adapter import GemmaSplitFleetAdapter, gemma_runtime_signature
from rcc.run.gemma.split_build import handoff_root
from rcc.run.gemma.split_receiver import GemmaFusedReceiver
from rcc.run.gemma.split_text import (
    TEXT_BUNDLE_CLOCKS,
    GemmaTextHandoffError,
    read_text_bundle,
    write_text_bundle,
)
from rcc.run.plan import ResolutionContext, resolve_plan
from rcc.run.runner import _completed_qids

# --- The adapter surface and engine roles.


def test_split_adapter_places_and_identifies_every_registered_arm(tmp_path: Path) -> None:
    from rcc.run.gemma.arm_subsets import TWELVE_ARMS, TWELVE_PROFILE

    adapter = GemmaSplitFleetAdapter(
        GemmaFleetConfig(
            results_base=tmp_path,
            source_cache=tmp_path / "source",
            item_offset=0,
            item_count=8,
            node_gpus=1,
            worker_index=0,
            attempt_id="a1",
            source_commit="0" * 40,
            unified_plan_fingerprint="9" * 64,
        ),
        arm_profile=TWELVE_PROFILE,
    )
    placements = adapter.placements.resolve(tmp_path, panel="production", source_commit="0" * 40)
    # The whole fleet is keyed by semantic arm: the placement map, the bank
    # paths, and the fixed identity every banked row carries.
    bindings = gemma_semantic_bindings()
    assert adapter.placements.arm_names == TWELVE_ARMS
    assert set(bindings.values()) == set(TWELVE_ARMS)
    assert set(adapter.placements.arm_names) == set(placements)
    assert len(placements) == 12
    assert placements["text_primary"].fused
    assert placements["latent_query_support_r128"].producers == 5
    assert placements["full"] == FleetPlacement(8, 4, 4)

    placement = placements[_ARM]
    paths = adapter.bank_paths(tmp_path, panel="production", arm=_ARM, placement=placement)
    assert len(paths) == placement.workers == 8
    assert paths[0] == tmp_path / "arms" / _ARM / "workers" / "gpu0" / "raw.jsonl"

    identity = adapter.bank_identity(
        tmp_path,
        panel="production",
        arm=_ARM,
        source_commit="0" * 40,
        prepared_manifest={"artifact_sha256": "a" * 64},
        placement=placement,
    )
    assert isinstance(identity, BankIdentity)
    assert identity.fields["execution_isolation_profile"] == SPLIT_FLEET_ISOLATION_PROFILE
    assert identity.fields["semantic_arm"] == _ARM
    assert identity.fields["placement"]["producers"] == 4  # four producer seats of eight
    assert identity.fields["source_cache_fingerprint"]
    assert identity.fields["unified_plan_fingerprint"] == "9" * 64
    assert identity.runtime_signature is not None
    assert identity.runtime_signature["auxiliary_packages"] == {
        "accelerate": "1.14.0",
        "tokenizers": "0.22.2",
    }
    assert identity.runtime_signature == gemma_runtime_signature()
    assert adapter.registration()["isolation_profile"] == SPLIT_FLEET_ISOLATION_PROFILE
    assert adapter.sampling_contract()["sample_tags"] == ["s0", "s1", "s2"]
    # Every split row is published under the shared panel, so a bank opened
    # under any other panel would refuse the first row it was handed.
    with pytest.raises(ValueError, match="publishes every row under"):
        adapter.bank_identity(
            tmp_path,
            panel="other",
            arm=_ARM,
            source_commit="0" * 40,
            prepared_manifest={"artifact_sha256": "a" * 64},
            placement=placement,
        )
    with pytest.raises(ValueError, match="not a registered Gemma semantic arm"):
        adapter.bank_identity(
            tmp_path,
            panel="production",
            arm="r2",
            source_commit="0" * 40,
            prepared_manifest={"artifact_sha256": "a" * 64},
            placement=placement,
        )

    # Only the timed s0 row of a decoded cell completes an item: its two
    # batched peers are banked first and complete nothing on their own.
    keys = {adapter.completion_key(row) for row in ({"kind": "phase"}, {"kind": "stage"})}
    assert keys == {None}
    cell = {"kind": "result", "panel": "production", "qid": "q1", "cell": "r2|s0"}
    assert adapter.completion_key({**cell, "seed_index": 0}) == ("production", "q1", "r2|s0")
    assert adapter.completion_key({**cell, "cell": "r2|s1", "seed_index": 1}) is None
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


def test_two_engine_identities_are_registered_and_only_one_carries_the_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    # The capture identity is hashed into every banked resident row, so these
    # three strings are pinned here byte for byte.
    assert registered_identity(PRODUCER_ROLE) is CAPTURE_ENGINE_IDENTITY
    assert CAPTURE_ENGINE_IDENTITY.expectations() == {
        "kv_connector": "GemmaCaptureConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": "rcc.models.gemma.capture_connector",
    }
    assert CAPTURE_ENGINE_IDENTITY.kv_transfer_config == (
        ("kv_connector", "GemmaCaptureConnector"),
        ("kv_role", "kv_both"),
        ("kv_connector_module_path", "rcc.models.gemma.capture_connector"),
    )
    # The receiver is a second registered identity beside it: it injects
    # prompt embeddings and opens with no capture route at all.
    receiver = registered_identity(RECEIVER_ROLE)
    assert receiver is SPLIT_RECEIVER_ENGINE_IDENTITY
    assert receiver.kv_transfer_config is None
    assert set(receiver.expectations().values()) == {None}
    with pytest.raises(ValueError, match="unregistered Gemma engine role"):
        registered_identity("capture")

    producer_config = engine_config(CAPTURE_ENGINE_IDENTITY)
    receiver_config = engine_config(receiver)
    assert producer_config.kv_transfer_config == CAPTURE_ENGINE_IDENTITY.kv_transfer_config
    assert receiver_config.kv_transfer_config is None
    # Only the transfer stanza separates the two roles; every other pinned
    # engine setting stays identical, so the two halves decode alike.
    assert {
        name: value for name, value in vars(producer_config).items() if name != "kv_transfer_config"
    } == {
        name: value for name, value in vars(receiver_config).items() if name != "kv_transfer_config"
    }


# --- The shipped config resolves to the registered arms and placements.


def test_a_relabelled_rerun_keeps_the_panel_and_takes_a_fresh_run_identity(
    tmp_path: Path,
) -> None:
    """A rerun of the twelve-arm profile repeats the measurement, not the run id."""
    from rcc.run.gemma.arm_subsets import TWELVE_ARMS, TWELVE_PROFILE
    from rcc.run.split_driver import _gemma_binding

    root = tmp_path
    config = load_run_config("configs/gemma-fanoutqa-natural-dev50.toml")
    context = ResolutionContext(git_commit="0" * 40)
    original = resolve_plan(config, context)
    repeat = resolve_plan(replace(config, run_label="gemma-subset-test"), context)
    assert tuple(arm.arm_id for arm in repeat.arms) == TWELVE_ARMS
    assert len(repeat.selected_question_ids) == 50
    assert repeat.run_id != original.run_id and repeat.output_uri != original.output_uri
    assert repeat.execution_identity_hash != original.execution_identity_hash
    fleet = GemmaFleetConfig(
        results_base=root,
        source_cache=root / "source",
        item_offset=0,
        item_count=50,
        node_gpus=8,
        worker_index=0,
        attempt_id="subset",
        source_commit=context.git_commit,
        benchmark_profile=repeat.benchmark,
        unified_plan_fingerprint=repeat.execution_identity_hash,
    )
    adapter = GemmaSplitFleetAdapter(fleet, arm_profile=TWELVE_PROFILE)
    selected = adapter.placements.resolve(
        root, panel="production", source_commit=context.git_commit
    )
    standard = GemmaSplitFleetAdapter(fleet).placements.resolve(
        root, panel="production", source_commit=context.git_commit
    )
    assert adapter.placements.arm_names == TWELVE_ARMS
    assert tuple(selected) == TWELVE_ARMS
    # The subset seats every published arm exactly where the default roster does,
    # and the full control at its own registered placement.
    assert all(selected[arm] == standard[arm] for arm in standard)
    assert selected["full"] == SEMANTIC_ARM_PLACEMENTS["full"]
    # The subset is registered for the natural panel on Gemma alone.
    for changed in (
        replace(config, model="ministral3-14b"),
        replace(config, benchmark="fanoutqa"),
    ):
        with pytest.raises(ValueError):
            resolve_plan(changed, context)
    args = argparse.Namespace(
        benchmark=repeat.benchmark.benchmark_key,
        source_cache=root / "source",
        results_root=root,
        item_offset=0,
        item_count=50,
        node_gpus=8,
        attempt_id="subset",
        plan_fingerprint=repeat.execution_identity_hash,
        arm=[],
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("RCC_GEMMA_M3_ARM_PROFILE", TWELVE_PROFILE)
        patch.setenv("RCC_GEMMA_M3_BENCHMARK_PROFILE", repeat.benchmark.profile_id)
        runtime, _, _, _ = _gemma_binding(args, context.git_commit)
        assert runtime.placements.arm_names == TWELVE_ARMS
        args.arm = ["latent_query_support_r8"]
        with pytest.raises(RuntimeError, match="all its arms"):
            _gemma_binding(args, context.git_commit)


# --- The single-rank executor and its one-shot engine shutdown.


def test_single_rank_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse shared stores and distributed peers instead of handing them port zero."""
    from rcc.models.gemma.executor import distributed_args

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(world_size=1, data_parallel_size=1),
        device_config=SimpleNamespace(device="cuda:0"),
    )
    assert distributed_args(config) == ("tcp://127.0.0.1:0", 0, 0)
    config.parallel_config.world_size = 2
    with pytest.raises(RuntimeError, match="single-rank"):
        distributed_args(config)
    config.parallel_config.world_size = 1
    monkeypatch.setenv("TORCHELASTIC_USE_AGENT_STORE", "True")
    with pytest.raises(RuntimeError, match="agent"):
        distributed_args(config)


def test_engine_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Close the real wrapper twice and require one explicit core shutdown."""
    from rcc.models.gemma import engine

    calls: list[str] = []
    wrapper = object.__new__(engine.DecodeEngine)
    wrapper._llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            engine_core=SimpleNamespace(shutdown=lambda: calls.append("stop"))
        )
    )
    monkeypatch.setattr(engine, "free_gpu_memory", lambda: None)
    wrapper.close()
    wrapper.close()
    assert calls == ["stop"]


_FUSED_PLACEMENT = FleetPlacement(workers=1, producers=0, receivers=1, fused=True)


_DIRECT_PLACEMENT = FleetPlacement(workers=2, producers=0, receivers=2)


@pytest.fixture(autouse=True)
def _offline_receiver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Score offline and embed at fixture width: this suite covers plumbing."""
    monkeypatch.setattr(
        "rcc.models.gemma.results.score_text",
        lambda _question, _visible: {"loose": 0.5, "strict": 0.25, "n_leaves": 2},
    )
    monkeypatch.setattr(
        "rcc.models.gemma.results.evidence_answerability_audit",
        lambda *_args, **_kwargs: {
            "n_locatable_in_full_source": 2,
            "n_survives_construction": 1,
        },
    )
    # The shipped receiver validates the published table against the pinned
    # production geometry; the fixture publishes the same mechanism at a width
    # a test can hold in memory.
    monkeypatch.setattr("rcc.models.gemma.receiver_core.EMBEDDING_SHAPE", _TABLE_SHAPE)


def _drive(
    root: Path,
    bank: _Bank,
    placement: FleetPlacement,
    workers: list[Any],
    *,
    arm: str = _ARM,
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
                    attempt_id=_ATTEMPT_ID,
                    qids=_QIDS,
                    placement=placement,
                    worker_index=index,
                    max_wall_s=30,
                    barrier_timeout_s=10,
                    claim_lease_s=10,
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
        thread.join(timeout=60)
    assert not [thread for thread in threads if thread.is_alive()]
    assert not failures, failures[0]


_BARRIER = BarrierSpec(
    family_label="gemma split",
    marker_schema="gemma-split-arm-barrier-marker-v1",
    barrier_dirname="arm-barrier",
)


def _schedule(root: Path, arms: dict[str, FleetPlacement], **overrides: Any) -> SplitScheduleConfig:
    fields: dict[str, Any] = {
        "root": root,
        "panel": "production",
        "source_commit": "0" * 40,
        "attempt_id": _ATTEMPT_ID,
        "barrier": _BARRIER,
        "plan_fingerprint": "plan",
        "roster_fingerprint": "roster",
        "arms": tuple(arms),
        "max_wall_s": 60.0,
        "barrier_timeout_s": 20.0,
        "claim_lease_s": 20.0,
        "poll_s": 0.001,
        "join_timeout_s": 120.0,
    }
    return SplitScheduleConfig(**{**fields, **overrides})


def test_the_shared_driver_runs_arms_in_order_then_merges_and_reports(tmp_path: Path) -> None:
    # A direct arm and a producer/receiver arm, at the node width one test
    # process can drive. The registered table is eight wide; what the driver
    # decides is the order and the isolation, not the width.
    arms = {_FLOOR_ARM: _DIRECT_PLACEMENT, _ARM: _SPLIT_PLACEMENT}
    adapter = fixture_adapter(tmp_path, arms)
    publish_table(tmp_path, _ARM, _SPLIT_PLACEMENT)
    summary = run_split_fleet(adapter, _schedule(tmp_path, arms))

    # Arms ran one after another, in the order the schedule named.
    assert summary["arms"] == [_FLOOR_ARM, _ARM]
    assert summary["attempt_id"] == _ATTEMPT_ID
    assert summary["panel"] == "production"
    assert summary["warmup_contract"] == warmup_contract()

    # The merge reports one cell per item and arm over the banked role rows.
    report = adapter.reports.build_report(tmp_path, source_commit="0" * 40)
    assert report["schema"] == "gemma-fanoutqa-split-report-v1"
    assert sorted(report["arms"]) == sorted(arms)
    assert report["result_cells"] == len(arms) * len(_QIDS)
    assert report["cell_width_by_arm"] == dict.fromkeys(arms, 3)
    assert report["dropped_partial_rows"] == {}
    grid = {row["arm"]: row for row in cast(list[dict[str, Any]], report["arm_grid"])}
    assert set(grid) == set(arms)
    assert grid[_ARM]["n"] == len(_QIDS)
    assert grid[_ARM]["latency_n"] == len(_QIDS)
    assert grid[_ARM]["policy_compute_tteoa_s_mean"] > 0.0
    # A direct arm prices no producer stage, so its mean is absent, never zero.
    assert grid[_FLOOR_ARM]["fleet_producer_s_mean"] is None
    assert grid[_ARM]["fleet_producer_s_mean"] > 0.0
    assert all(
        float(row["t_unix"]) > 1_000_000_000
        for arm in arms
        for row in read_arm_bank_rows(tmp_path / "arms" / arm)
        if row.get("kind") == "stage"
    )
    # The recall reader names the handoff fields as literals, so a producer
    # that renamed one would empty its draw comparison rather than fail it.
    # This run drives cut arms, which carry neither the text channel's reports
    # nor the Ministral spelling of the payload digest.
    banked = [
        row
        for arm in arms
        for row in read_arm_bank_rows(tmp_path / "arms" / arm)
        if row.get("kind") == "result"
    ]
    cut = set(recall.HANDOFF_FIELDS) - {"reports", "payload_sha256"}
    assert banked and all({"prepared_sha256", *cut} <= set(row) for row in banked)
    assert (tmp_path / "report" / "selected_results.jsonl").is_file()

    with driver_lock(tmp_path), pytest.raises(RuntimeError, match="already running"):
        run_split_fleet(adapter, _schedule(tmp_path, arms))


def test_latent_arms_cross_the_queue_from_capture_to_banked_seed_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warm_receiver = _warm_real_split_receiver(monkeypatch)
    warm_receiver.live = cast(Any, dict.fromkeys(("a", "b", "c")))
    assert warm_receiver.claim_limit == MAX_NUM_SEQS and warm_receiver.can_accept()
    # Query-Support ships the selected rows through the split queue.
    for arm in (_ARM,):
        root = tmp_path / arm
        bank = _Bank(root, arm=arm, placement=_SPLIT_PLACEMENT)
        _drive(
            root,
            bank,
            _SPLIT_PLACEMENT,
            [
                _producer(root, bank.append, arm=arm),
                _receiver(root, bank.append, arm=arm, placement=_SPLIT_PLACEMENT),
            ],
            arm=arm,
        )

        # Every item crossed the process boundary as a manifest the receiver
        # verified, and came back as the registered three seed rows, projected
        # onto the shared semantic identity the bank is keyed by.
        rows = bank.results()
        assert {row["scoring_version"] for row in rows} == {"fanoutqa-equivalence-v1"}
        assert {row["qid"] for row in rows} == set(_QIDS)
        v16_arm = gemma_semantic_bindings()
        physical = next(name for name, semantic in v16_arm.items() if semantic == arm)
        assert sorted(row["cell"] for row in rows if row["qid"] == _QIDS[0]) == [
            f"{physical}|s{index}" for index in range(3)
        ]
        assert all(row["arm"] == row["semantic_arm"] == arm for row in rows)
        assert {row["panel"] for row in rows} == {"production"}
        stages = {(row["stage"], row["qid"]) for row in bank.stages()}
        assert ("producer_end", _QIDS[0]) in stages
        assert ("payload_loaded", _QIDS[0]) in stages
        for qid in _QIDS:
            manifest = handoff_manifest_path(handoff_root(root, arm), qid)
            assert manifest.is_file() and clock_record_path(manifest).is_file()
            named = json.loads(manifest.read_text(encoding="utf-8"))["files"]
            assert SELECTION_KEY in named

        # The whole shared latency vocabulary is present, both spill clocks are
        # real, and the producer clock is the wall its own process measured.
        row = rows[0]
        assert not [name for name in ITEM_LATENCY_FIELDS if name not in row]
        assert row["spill_save_s"] > 0.0 and row["spill_load_s"] > 0.0
        clocks = json.loads(
            clock_record_path(
                handoff_manifest_path(handoff_root(root, arm), str(row["qid"]))
            ).read_text(encoding="utf-8")
        )
        assert row["spill_save_s"] == clocks["spill_save_s"]
        assert row["producer_s"] == clocks["producer_s"] > 0.0
        # The two harness clocks stay outside what the caller waited for.
        assert row["ttft_s"] == pytest.approx(
            row["producer_s"] + row["receiver_prepare_s"] + row["receiver_ttft_s"], abs=1e-4
        )
        assert row["ratio"] == 2
        assert row["payload_layout"] == "flat-interleave-v1"
        assert row["latent_steps"] == LATENT_STEPS
        assert row["answer_max_tokens"] == 24_000
        assert len(row["sampled_tokens"]) == 3
        assert row["handoff_prompt_tokens"] > LATENT_STEPS
        # Every producer row names the registered engine that made the cargo.
        capture_rows = [row for row in bank.rows if row.get("phase") == "gemma_split_capture"]
        assert capture_rows and all(row["engine_identity_sha256"] for row in capture_rows)
        clock_names = {"extract_s", "roll_s", "capture_query_s", "parity_s"}
        assert all(clock_names <= row.keys() for row in capture_rows)
        handoff_ready = {
            str(stage["qid"]): float(stage["t_unix"])
            for stage in bank.stages()
            if stage["stage"] == "handoff_ready"
        }
        for capture_row in capture_rows:
            bank_path = (
                handoff_root(root, arm) / "layer_banks" / str(capture_row["layer_bank_file"])
            )
            assert bank_path.stat().st_mtime >= handoff_ready[str(capture_row["qid"])]
        banked_capture_s = sum(
            float(capture_row["capture_s"])
            for capture_row in capture_rows
            if capture_row["qid"] == row["qid"]
        )
        assert row["producer_s"] == pytest.approx(banked_capture_s + row["selection_s"], abs=1e-3)

        # Only the timed s0 row completes an item. A crash between the two
        # peer banks and the s0 bank must leave the item to be decoded again.
        completion_key = adapter(root).completion_key
        peers = [row for row in rows if row["qid"] == _QIDS[0] and row["seed_index"] != 0]
        assert len(peers) == 2
        assert {completion_key(row) for row in peers} == {None}
        crashed = root / "crashed" / "raw.jsonl"
        crashed.parent.mkdir(parents=True)
        with crashed.open("w", encoding="utf-8") as handle:
            for peer in peers:
                handle.write(json.dumps(peer, sort_keys=True) + "\n")
        assert _completed_qids((crashed,), completion_key)() == set()
        timed = next(row for row in rows if row["qid"] == _QIDS[0] and row["seed_index"] == 0)
        with crashed.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(timed, sort_keys=True) + "\n")
        assert _completed_qids((crashed,), completion_key)() == {_QIDS[0]}

        _drop_first_capture_row(bank.path)
        with pytest.raises(RuntimeError, match="exactly one capture audit row"):
            fixture_adapter(root, {arm: _SPLIT_PLACEMENT}).reports.build_report(
                root,
                source_commit=_SOURCE_COMMIT,
            )

    failed_root = tmp_path / "failed-audit"
    publish_table(failed_root, _ARM, _SPLIT_PLACEMENT)

    def fail_audit(_audit: DeferredAudit) -> None:
        raise RuntimeError("deferred audit failed")

    monkeypatch.setattr("tests.gemma_split_support._fake_capture", _failing_capture)
    monkeypatch.setattr(DeferredAudit, "run", fail_audit)
    with pytest.raises(RuntimeError, match=r"deferred audit failed|FLEET_STOP"):
        run_split_fleet(
            fixture_adapter(failed_root, {_ARM: _SPLIT_PLACEMENT}),
            _schedule(failed_root, {_ARM: _SPLIT_PLACEMENT}),
        )
    assert (failed_root / "arms" / _ARM / "control" / f"FLEET_STOP.{_ATTEMPT_ID}").is_file()
    attempt_root = failed_root / "control" / "arm-barrier" / _ATTEMPT_ID
    assert not list(attempt_root.glob("*/arm_complete"))


def test_the_resident_text_sender_binds_its_prompts_and_holds_its_contract(
    tmp_path: Path,
) -> None:
    """The sender contract, its prompt-artifact binding, and the resident entry point."""
    assert_gemma_text_contract(tmp_path)


def test_text_arm_publishes_one_bundle_and_prices_its_own_spill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The text payload is the sender-contract test's subject; here it is the
    # bundle crossing processes.
    monkeypatch.setattr(
        "rcc.models.gemma.text_results.prepare_text_receiver",
        lambda tokenizer, weight, item, bundle: _prepared_text(str(item["qid"])),
    )
    monkeypatch.setattr(
        "rcc.models.gemma.text_results.build_text_result_rows",
        lambda **kwargs: _text_rows(kwargs),
    )
    arm = "text_medium"
    bank = _Bank(tmp_path, arm=arm, placement=_SPLIT_PLACEMENT)
    publish_table(tmp_path, arm, _SPLIT_PLACEMENT)
    producer, item = _text_producer(tmp_path, bank.append, arm=arm)
    producer.warm()
    artifact = producer.produce(WorkItem(str(item["qid"]), 0))
    producer.close()

    # One immutable per-item bundle, and a clock record bound to its exact bytes.
    bundle = read_text_bundle(artifact.path, qid=str(item["qid"]), semantic_arm=arm)
    assert bundle["reports"] == ["river", "evidence", "answer"]
    saved = read_bundle_clocks(
        TEXT_BUNDLE_CLOCKS, artifact.path, qid=str(item["qid"]), semantic_arm=arm
    )
    assert saved > 0.0 and saved == round(float(artifact.fields["spill_save_s"]), 4)

    receiver = _receiver(
        tmp_path,
        bank.append,
        arm=arm,
        placement=_SPLIT_PLACEMENT,
        items={str(item["qid"]): item},
    )
    receiver.warm()
    receiver.submit(WorkItem(str(item["qid"]), 0), artifact)
    completions: list[Any] = []
    for _turn in range(8):
        completions.extend(receiver.pump())
        if receiver.idle():
            break
    assert len(completions) == 1
    # The text arm prices its own disk exactly as the latent arm does: the
    # producer's measured write travels in the clock record, never as a zero.
    row = completions[0].result
    assert row["spill_save_s"] == saved and row["spill_load_s"] > 0.0
    assert row["arm"] == arm and row["panel"] == "production"

    # A bundle rewritten under its old clock record is another item's bytes.
    write_text_bundle(
        artifact.path,
        qid=str(item["qid"]),
        semantic_arm=arm,
        bundle={**bundle, "reports": ["other", "other", "other"]},
    )
    with pytest.raises(GemmaTextHandoffError, match="bundle clock record names other bytes"):
        read_bundle_clocks(
            TEXT_BUNDLE_CLOCKS, artifact.path, qid=str(item["qid"]), semantic_arm=arm
        )


def test_direct_arm_serves_the_whole_node_with_no_producer_and_no_handoff(
    tmp_path: Path,
) -> None:
    bank = _Bank(tmp_path, arm=_FLOOR_ARM, placement=_DIRECT_PLACEMENT)
    # Nobody produces on this arm, so the run publishes the shared table above
    # the arms before any receiver opens.
    publish_table(tmp_path, _FLOOR_ARM, _DIRECT_PLACEMENT)
    receivers = [
        _receiver(tmp_path, bank.append, arm=_FLOOR_ARM, placement=_DIRECT_PLACEMENT)
        for _index in range(2)
    ]
    # The registered direct placement gives every GPU to receivers; the
    # fixture runs the same shape at the width one test process can drive.
    assert bind_policy_placements(gemma_semantic_bindings(), table=SEMANTIC_ARM_PLACEMENTS)[
        "floor"
    ] == FleetPlacement(8, 0, 8)
    _drive(tmp_path, bank, _DIRECT_PLACEMENT, receivers, arm=_FLOOR_ARM)

    rows = bank.results()
    assert {row["qid"] for row in rows} == set(_QIDS)
    assert all(row["arm"] == _FLOOR_ARM and row["v16_arm"] == "floor" for row in rows)
    # Nothing shipped, so both spill clocks are honestly zero and no producer
    # stage was ever banked.
    assert all(row["spill_load_s"] == 0.0 and row["spill_save_s"] == 0.0 for row in rows)
    assert not any(row["stage"] == "producer_start" for row in bank.stages())


def test_fused_arm_produces_what_it_serves_and_collapses_the_handoff_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm = "text_primary"
    generated: list[str] = []
    # The sender contract the fused closure calls has its own test above; here
    # the subject is the fleet wiring.
    monkeypatch.setattr(
        "rcc.models.gemma.text_results.prepare_text_receiver",
        lambda tokenizer, weight, item, bundle: _prepared_text(str(item["qid"])),
    )
    monkeypatch.setattr(
        "rcc.models.gemma.text_results.build_text_result_rows",
        lambda **kwargs: _text_rows(kwargs),
    )

    def reports(item: dict[str, Any]) -> dict[str, Any]:
        generated.append(str(item["qid"]))
        return {"schema": "gemma4-fanoutqa-text-report-v1", "semantic_arm": arm}

    bank = _Bank(tmp_path, arm=arm, placement=_FUSED_PLACEMENT)
    publish_table(tmp_path, arm, _FUSED_PLACEMENT)
    receiver = _receiver(
        tmp_path,
        bank.append,
        arm=arm,
        placement=_FUSED_PLACEMENT,
        fused=True,
        reports=reports,
    )
    _drive(tmp_path, bank, _FUSED_PLACEMENT, [receiver], arm=arm)

    assert sorted(generated) == sorted(_QIDS)
    assert {row["qid"] for row in bank.results()} == set(_QIDS)
    # Same model, same GPU: the producer and handoff boundaries are one
    # instant, and the runtime banks all four of them from the fused clock.
    stages = {(row["stage"], row["qid"]) for row in bank.stages()}
    for name in ("producer_end", "handoff_ready", "receiver_claim", "payload_loaded"):
        assert (name, _QIDS[0]) in stages, name
    fused_times = {
        row["t_unix"]
        for row in bank.stages()
        if row["qid"] == _QIDS[0] and row["stage"] in {"producer_end", "payload_loaded"}
    }
    assert len(fused_times) == 1
    assert all(row["spill_load_s"] == 0.0 for row in bank.results())
    # Fusing is a property of one registered same-model arm, not a mode.
    with pytest.raises(ValueError, match="serves text_primary only"):
        GemmaFusedReceiver(
            reports=reports,
            arm=_FLOOR_ARM,
            v16_arm="floor",
            items={},
            bank=bank.append,
            tokenizer=None,
            shared_weight=tmp_path / "missing.bin",
            runtime_fingerprint="f" * 16,
            publication_id=_ATTEMPT_ID,
            engine=object(),
        )


def test_receiver_refuses_a_tampered_handoff_and_a_foreign_table(tmp_path: Path) -> None:
    bank = _Bank(tmp_path, arm=_ARM, placement=_SPLIT_PLACEMENT)
    producer = _producer(tmp_path, bank.append, arm=_ARM)
    producer.warm()
    artifact = producer.produce(WorkItem(_QIDS[0], 0))
    producer.close()
    receiver = _receiver(tmp_path, bank.append, arm=_ARM, placement=_SPLIT_PLACEMENT)
    receiver.warm()

    rolled = handoff_root(tmp_path, _ARM) / "rolled" / latent_roll_filename(_QIDS[0], 1)
    raw = bytearray(rolled.read_bytes())
    raw[-1] ^= 0x01
    rolled.write_bytes(bytes(raw))
    with pytest.raises(GemmaHandoffError, match="handoff file digest differs"):
        receiver.submit(WorkItem(_QIDS[0], 0), artifact)

    # A clock record that does not name this manifest prices another handoff.
    _write_rolled(rolled, 1, _rolled_tensor(1), qid=_QIDS[0])
    clocks = clock_record_path(artifact.path)
    published = clocks.read_bytes()
    body = json.loads(clocks.read_text(encoding="utf-8"))
    body["fingerprint"] = "0" * 64
    clocks.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(GemmaHandoffError, match="clock record belongs to another manifest"):
        receiver.submit(WorkItem(_QIDS[0], 0), artifact)

    # Cargo published beside another table is refused rather than banked with
    # two embedding digests nobody compared.
    clocks.write_bytes(published)
    receiver.embedding_digests = {**receiver.embedding_digests, "marker_sha256": "0" * 64}
    with pytest.raises(GemmaHandoffError, match="this receiver loaded"):
        receiver.submit(WorkItem(_QIDS[0], 0), artifact)

    # The direct arm never accepts a payload, whatever a ticket claims.
    direct_bank = _Bank(tmp_path, arm=_FLOOR_ARM, placement=_DIRECT_PLACEMENT)
    direct = _receiver(tmp_path, direct_bank.append, arm=_FLOOR_ARM, placement=_DIRECT_PLACEMENT)
    direct.warm()
    with pytest.raises(RuntimeError, match="admits no producer artifact"):
        direct.submit(WorkItem(_QIDS[0], 0), artifact)


# --- Retry after an interrupted arm.


def _schedule_retry(
    root: Path,
    attempt_id: str,
    arms: tuple[str, ...] = (_FLOOR_ARM, _ARM),
) -> SplitScheduleConfig:
    return SplitScheduleConfig(
        root=root,
        panel="production",
        source_commit="0" * 40,
        attempt_id=attempt_id,
        barrier=_BARRIER,
        plan_fingerprint="plan",
        roster_fingerprint="roster",
        arms=arms,
        max_wall_s=60.0,
        barrier_timeout_s=20.0,
        claim_lease_s=20.0,
        poll_s=0.001,
        join_timeout_s=120.0,
    )


def test_a_torn_cell_refuses_and_a_crashed_cell_is_decoded_again(tmp_path: Path) -> None:
    arms = {_ARM: _SPLIT_PLACEMENT}
    adapter = fixture_adapter(tmp_path, arms)
    publish_table(tmp_path, _ARM, _SPLIT_PLACEMENT)
    crashed = _Bank(tmp_path, arm=_ARM, placement=_SPLIT_PLACEMENT)
    peers = [
        {
            "kind": "result",
            "qid": _QIDS[0],
            "cell": f"r2|s{index}",
            "seed_index": index,
            "loose": 1.0,
            "strict": 1.0,
        }
        for index in (1, 2)
    ]
    for peer in peers:
        crashed.append(dict(peer))
    ledger = cell_ledger(
        crashed.rows,
        arm=_ARM,
        qids=_QIDS,
        completion_key=adapter.completion_key,
    )
    assert ledger.complete == () and ledger.pending == _QIDS

    # A stale stop file from the crashed attempt is cleared, not obeyed.
    stop = tmp_path / "arms" / _ARM / "control" / "FLEET_STOP"
    stop.parent.mkdir(parents=True, exist_ok=True)
    stop.write_text("stale\n", encoding="utf-8")

    def launch(bodies: Any, timeout: float) -> None:
        # The clear happens before the seats open, not after they close.
        assert not stop.exists()
        run_in_threads(bodies, join_timeout_s=timeout)

    summary = run_split_fleet(
        adapter,
        _schedule_retry(tmp_path, "gemma-split-relaunch", (_ARM,)),
        run_workers=launch,
    )
    assert summary["arms"] == [_ARM]
    assert summary["attempt_id"] == "gemma-split-relaunch"
    assert summary["panel"] == "production"
    report = adapter.reports.build_report(tmp_path, source_commit="0" * 40)
    assert report["result_cells"] == len(_QIDS)
    assert report["dropped_partial_rows"] == {_ARM: len(peers)}
    assert report["scored_rows"] == len(_QIDS) * 3
    merged, _ledgers = merged_split_rows(
        tmp_path,
        arms,
        label="gemma",
        qids=_QIDS,
        completion_key=adapter.completion_key,
    )
    cells = {row["qid"]: row for row in collapse_cells(merged, timed=adapter.completion_key)}
    crashed_cell = cells[_QIDS[0]]
    assert crashed_cell["scored_samples"] == 3
    assert (crashed_cell["loose"], crashed_cell["strict"]) == (0.5, 0.25)
    grid = {row["arm"]: row for row in cast(list[dict[str, Any]], report["arm_grid"])}
    assert (grid[_ARM]["loose"], grid[_ARM]["strict"]) == (0.5, 0.25)

    with pytest.raises(RuntimeError, match="attempt id was already used"):
        run_split_fleet(
            adapter,
            _schedule_retry(tmp_path, "gemma-split-relaunch", (_ARM,)),
        )
    crashed.append({**peers[0], "cell": "r2|s0", "seed_index": 0, "qid": "torn-item"})
    with pytest.raises(RuntimeError, match="the bank is torn"):
        adapter.reports.build_report(tmp_path, source_commit="0" * 40)


# --- Offline handoff recall over one banked Gemma run.

_CHAT_HEAD = "<bos><start_of_turn>user\n"
_CHAT_TAIL = "<end_of_turn>\n<start_of_turn>model\n"
#: What the Gemma builders write of `recall.HANDOFF_FIELDS`: `payload_sha256`
#: is the Ministral spelling of the published tensor's digest.
_HANDOFF_FIELDS = tuple(name for name in recall.HANDOFF_FIELDS if name != "payload_sha256")


class _NaturalTokenizer(FakeTokenizer):
    """The shared character fake wearing Gemma's turn, decoding back to text.

    The recall reader decodes the rows a selection kept, so the fake has to
    invert its own encode rather than name ids; the real turn is a string the
    template renders and the same string tokenized.
    """

    # The split helper looks for BOS in the rendered prefix; a character
    # alphabet creates no such id, so the fake declares it has none.
    bos_token_id = None

    def decode(self, tokens: Any, **kwargs: Any) -> str:
        del kwargs
        # The inverse of the width-one encode, `ord(character) % 997 + 1`,
        # which is one to one below U+03E5; the bundle is ASCII.
        return "".join(chr(int(token) - 1) for token in tokens)

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        rendered = f"{_CHAT_HEAD}{messages[0]['content']}{_CHAT_TAIL}"
        return self.encode(rendered) if kwargs.get("tokenize") else rendered


def _gemma_result_rows(root: Path, arm: str, **handoff: Any) -> Path:
    """Bank one arm's three seed rows under the Gemma cell spelling."""
    return bank_result_rows(
        root,
        arm,
        cells=[{"cell": f"{gemma_v16_arm(arm)}|s{index}"} for index in range(3)],
        written=_HANDOFF_FIELDS,
        **handoff,
    )


def test_the_gemma_reader_scores_the_keeps_one_banked_run_handed_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Gemma reader scores the decoded keeps, and refuses a moved handoff.

    One natural bundle, the shipped panel loader, and a bank whose rows and
    selection file carry the producers' own fields: a latent arm that kept the
    rows spelling one reference scores one leaf of two, the text arm scores the
    leaves its reports name, and the whole source scores both. Retouching the
    keeps or the prepared digest is refused rather than rescored.
    """
    bundle = tmp_path / "bundle"
    profile = write_bundle(bundle)
    tokenizer = _NaturalTokenizer(width=1)
    # The reader opens the shipped Gemma tokenizer, which no CPU gate holds.
    monkeypatch.setattr(gemma_tokenizer, "load_gemma4_tokenizer", lambda **_kwargs: tokenizer)
    # A one-question stand-in panel registers no construction pin, and the pin
    # is what the fifty sealed items are held to, not what this test covers.
    monkeypatch.setattr(gemma_natural, "require_registered_construction", lambda *_a, **_k: None)
    # The config names its benchmark by id, so the stand-in profile has to
    # answer to the sealed one's name while the plan resolves.
    monkeypatch.setitem(plan_module._BENCHMARKS, "fanoutqa-natural-dev50", profile)
    # The native subset is registered against the sealed fifty-item profile
    # alone, so the one-item stand-in runs the shared twelve-arm roster; the
    # recall reader reads only each arm's channel, which both rosters agree on.
    config = _one_item_config(
        tmp_path,
        "gemma-fanoutqa-natural-dev50.toml",
        (profile.arm_profiles[-1], profile.arm_profiles[1]),
    )
    plan = plan_module.load_and_resolve(config)
    root = Path(plan.output_uri)

    panel = gemma_natural.load_natural_worker_panel(
        bundle,
        tokenizer,
        offset=0,
        count=1,
        node_gpus=1,
        worker_index=0,
        profile=profile,
    )
    prompts = [ids.tolist() for ids in panel.items[0]["prompt_ids"]]
    # Worker zero keeps the rows that spell the first river and one rolled row
    # past its prompt: the cut runs over the rolled tail too, so a faithful
    # keep names a position the source ids never reach. The other two keep
    # nothing.
    start = tokenizer.decode(prompts[0]).index("Alpha")
    keeps = ((*range(start, start + 5), len(prompts[0]) + 2), (), ())
    latent_arm = "latent_query_support_r2"
    v16_arm = gemma_v16_arm(latent_arm)
    prompt_tokens = tuple(len(ids) for ids in prompts)
    layout = flat_interleave_layout(keeps, prompt_tokens)
    selection = handoff_root(root, latent_arm) / "selections" / "q1.json"
    selection.parent.mkdir(parents=True)
    selection.write_text(
        json.dumps(
            {
                "schema": SELECTION_SCHEMA,
                "qid": "q1",
                "keeps_by_arm": {v16_arm: [list(keep) for keep in keeps]},
                "layouts_by_arm": {
                    v16_arm: {
                        "span_order": [list(span) for span in layout.span_order],
                        "tier_start": layout.tier_start,
                        "cargo_first": layout.cargo_first,
                    }
                },
                "selection_s_by_arm": {v16_arm: 0.5},
                "selection_arms": [v16_arm],
            }
        ),
        encoding="utf-8",
    )
    # The fixture is a selection the shipped reader loads, not a shape only the
    # recall reader tolerates.
    lengths = tuple(payload_length(tokens) for tokens in prompt_tokens)
    stored = load_selection(selection, qid="q1", lengths=lengths, arms=(v16_arm,))
    assert stored.keeps_by_arm == {v16_arm: keeps}
    bank = _gemma_result_rows(
        root,
        latent_arm,
        prepared_sha256=panel.prepared_sha256,
        selected_indices_sha256=selected_indices_sha256(keeps),
    )
    _gemma_result_rows(
        root,
        "text_primary",
        prepared_sha256=panel.prepared_sha256,
        reports=["the Alpha river", "nothing locatable", "the Beta river"],
    )

    body = recall.handoff_recall(plan, root, bundle)
    by_arm = {summary["arm"]: summary for summary in body["arms"]}
    assert body["model_id"] == "gemma4-12b-it"
    assert (body["source"]["hits"], body["source"]["leaves"]) == (2, 2)
    assert (by_arm["text_primary"]["hits"], by_arm["text_primary"]["leaves"]) == (2, 2)
    assert by_arm[latent_arm] == {
        "arm": latent_arm,
        "questions": 1,
        "hits": 1,
        "leaves": 2,
        "micro_recall_percent": 50.0,
        "macro_recall_percent": 50.0,
    }

    # The banked keep hash signs the selection file, and the banked prepared
    # digest signs the panel the prompts were read from.
    rebank = rebanker(bank)
    rebank(selected_indices_sha256=selected_indices_sha256(((0,), (), ())))
    with pytest.raises(RuntimeError, match="banked selection hash"):
        recall.handoff_recall(plan, root, bundle)
    rebank(prepared_sha256="f" * 64)
    with pytest.raises(RuntimeError, match="prepared_sha256"):
        recall.handoff_recall(plan, root, bundle)
