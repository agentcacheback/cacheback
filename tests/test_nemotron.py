"""The Nemotron runtime: prompt sealing, receiver placement, the compiler, and registration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from tests.longbench_coa_support import FakeTokenizer, chain_item, raw_row

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.longbench_v2 import EASY50_ORDER
from rcc.benchmarks.longbench_v2.data import build_item
from rcc.benchmarks.longbench_v2.native_geometry import frozen_geometry
from rcc.benchmarks.longbench_v2.nemotron import (
    NEMOTRON_LONGBENCH_BOUNDED,
    NEMOTRON_LONGBENCH_RERANK,
    NEMOTRON_LONGBENCH_TEXT,
)
from rcc.benchmarks.longbench_v2.registration import build_config
from rcc.models.nemotron import NEMOTRON, NEMOTRON_FAMILY
from rcc.models.qwen.native_results import validate_native_report
from rcc.models.qwen.receiver import QwenVisibleAnswer, prepare_receiver_prompt
from rcc.models.qwen.results import build_result_row, validate_and_rescore_result
from rcc.models.qwen.text_decode import QwenDecodeSpec
from rcc.run.config import load_run_config
from rcc.run.fleet.latency import ItemLatency
from rcc.run.plan import ResolutionContext, load_and_resolve, resolve_plan
from rcc.topologies.chain import CHAIN_T4

# --- Toy tokenizers and sealed native prompt fixtures.


class _NativeTokenizer:
    """Small reversible vocabularies expose cross-sender prompt drift."""

    is_fast = True

    def __init__(self, offset: int) -> None:
        self.offset = offset

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        result: dict[str, Any] = {"input_ids": [ord(char) + self.offset for char in text]}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result

    def decode(self, ids: Any, **_kwargs: Any) -> str:
        return "".join(chr(int(token) - self.offset) for token in ids if token != 0)

    def apply_chat_template(self, messages: Any, **_kwargs: Any) -> str:
        # This toy renderer models only user/evidence bytes. The real Nano
        # template's system/flag controls are covered by the family contract;
        # this fixture isolates artifact mechanics and cross-tokenizer drift.
        body = "".join(
            f"<{row['role']}>{row['content']}" for row in messages if row["role"] == "user"
        )
        return body + "<assistant><think>"


def _native_prompt_case() -> tuple[Any, dict[str, Any]]:
    from rcc.benchmarks.fanoutqa.data import Question
    from rcc.benchmarks.fanoutqa.source_padding import ProbeItem
    from rcc.models.route import TEXT_SEMANTIC_ARMS

    tokenizers = {
        arm: _NativeTokenizer(index * 1000) for index, arm in enumerate(TEXT_SEMANTIC_ARMS, 1)
    }
    tokenizer = tokenizers["text_primary"]
    evidence = ("Alpha opened in 1972.", "Bravo owns station B.", "Charlie operates line C.")
    shards = tuple(tuple(tokenizer(text)["input_ids"]) for text in evidence)
    question = Question("native", "Which year?", ((1, 1, "Alpha"),), {"answer": "1972"})
    return ProbeItem("native", question.question, shards, question), tokenizers


def _native_artifact() -> tuple[Any, dict[str, Any], Any]:
    from rcc.models.qwen.native_prompts import build_native_prompt_artifact

    item, tokenizers = _native_prompt_case()
    artifact = build_native_prompt_artifact(
        item, tokenizers, construction_sha256="a" * 64, family=NEMOTRON_FAMILY
    )
    return item, tokenizers, replace(item, native_prompt_artifacts=artifact)


# --- Native prompt sealing and the hybrid continuation.


def test_nemotron_native_result_identity() -> None:
    """Keep report and answer decode and count fields tied to the sealed native prompts."""
    from rcc.models.qwen.native_results import validate_native_result

    _item, _tokenizers, prepared = _native_artifact()
    arm = "text_small"
    sender = prepared.native_prompt_artifacts["senders"][arm]
    profile = NEMOTRON_FAMILY.benchmark_profile(FANOUTQA_NATURAL_DEV50)
    counts = [row["prompt_tokens"] for row in sender["workers"]]
    fields = {
        "report_prompt_token_sha256": [row["prompt_token_sha256"] for row in sender["workers"]],
        "report_prompt_tokens_by_worker": counts,
        "report_prompt_unpadded_tokens_by_worker": [
            row["unpadded_prompt_tokens"] for row in sender["workers"]
        ],
        "report_prompt_policy": "natural-ceiling",
        "report_enable_thinking": True,
        "report_decode": QwenDecodeSpec("report", NEMOTRON_FAMILY.sender_family(arm)).to_dict(),
        "decode": {
            **NEMOTRON_FAMILY.profile.decode.to_dict(),
            "max_tokens": profile.answer_ceiling,
            "stop_token_ids": list(NEMOTRON_FAMILY.stop_token_ids),
        },
        "worker_prompt_tokens": counts,
        "aggregate_worker_prompt_tokens": sum(counts),
    }
    validate_native_result(prepared, fields, arm=arm, family=NEMOTRON_FAMILY)
    with pytest.raises(RuntimeError, match="worker prompt counts"):
        validate_native_result(
            prepared,
            {**fields, "worker_prompt_tokens": [*counts[:2], counts[2] - 1]},
            arm=arm,
            family=NEMOTRON_FAMILY,
        )


def test_nemotron_receiver_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the active Nano receiver's context reservation at its boundary."""
    from rcc.models.qwen import receiver

    profile = NEMOTRON_FAMILY.benchmark_profile(FANOUTQA_NATURAL_DEV50)
    primary = next(
        arm for arm in NEMOTRON_FAMILY.profile.physical_arms if arm.semantic_arm == "text_primary"
    )
    limit = min(
        profile.max_model_len,
        int(dict(NEMOTRON_FAMILY.profile.runtime.engine_flags)["max_model_len"]),
        primary.sender_native_max_model_len or 0,
    )
    closing = len(NEMOTRON_FAMILY.think_close_token_ids) + receiver.ANSWER_CLOSING_TOKEN_BUDGET
    manager = limit - profile.answer_ceiling - closing
    receiver.prepare_receiver_prompt(
        torch.zeros((1, 1)),
        [0] * manager,
        family=NEMOTRON_FAMILY,
        qid="budget",
        semantic_arm="text_small",
    )
    with pytest.raises(RuntimeError, match="receiver context"):
        receiver.prepare_receiver_prompt(
            torch.zeros((1, 1)),
            [0] * (manager + 1),
            family=NEMOTRON_FAMILY,
            qid="budget",
            semantic_arm="text_small",
        )
    monkeypatch.setattr(
        receiver, "token_embedding_rows", lambda *_a, **_k: torch.zeros((manager, 1))
    )


def test_nemotron_sealed_native_prompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Round-trip a sealed active-Nano artifact and reject one whose prompts are missing."""
    import pickle

    from rcc.benchmarks.fanoutqa import prepare
    from rcc.benchmarks.fanoutqa.source_audit import panel_prepared_paths
    from rcc.benchmarks.fanoutqa.source_build import append_construction_row, seal_prepared_panel
    from rcc.benchmarks.fanoutqa.source_planning import source_page_provenance
    from rcc.benchmarks.fanoutqa.source_topology import shard_fingerprints
    from rcc.models.qwen import native_prompts as native
    from rcc.run.io import sha256_file

    item, tokenizers = _native_prompt_case()
    profile = replace(FANOUTQA_NATURAL_DEV50, question_ids=(item.qid,))
    audit = {"audit_fingerprint": "native", "shards": shard_fingerprints((item,))}
    root = tmp_path / "nemotron"
    artifact, manifest_path, ledger = panel_prepared_paths(root, profile)
    append_construction_row(
        ledger,
        source_page_provenance(
            qid=item.qid,
            pageid=1,
            revid=1,
            raw_text="raw",
            canonical_text="canonical",
            actual_tokens=1,
            census_tokens=1,
        ),
    )
    kwargs = {
        "source_commit": "a" * 40,
        "profile": profile,
        "family": NEMOTRON_FAMILY,
    }
    with monkeypatch.context() as patch:
        patch.setattr(native, "_tokenizers", lambda *args, **kwargs: tokenizers)
        patch.setattr(prepare, "validate_source_audit", lambda *args, **kwargs: audit)
        seal_prepared_panel(root, items=(item,), source_audit=audit, **kwargs)
        loaded, manifest = prepare.load_prepared_panel(root, **kwargs)
        assert loaded[0].native_prompt_artifacts is not None
        native.native_worker_prompts(
            loaded[0], tokenizers["text_small"], semantic_arm="text_small", family=NEMOTRON_FAMILY
        )
        # The rebuild leaves a record; identical bytes reuse it, a foreign one rebuilds.
        record = artifact.with_name(f"{artifact.name}.native-prompts.json")
        assert json.loads(record.read_text())["schema"] == prepare.NATIVE_PROMPT_RECORD_SCHEMA
        rebuild, real = [], native.validate_native_prompts
        patch.setattr(native, "validate_native_prompts", lambda *a, **k: rebuild.append(a))
        prepare.load_prepared_panel(root, **kwargs)
        record.write_text('{"schema": "other"}')
        prepare.load_prepared_panel(root, **kwargs)
        assert len(rebuild) == 1 and json.loads(record.read_text())["schema"] != "other"
        patch.setattr(native, "validate_native_prompts", real)
        bundle = prepare._load_bundle(artifact)
        bundle["items"] = (replace(loaded[0], native_prompt_artifacts=None),)
        artifact.write_bytes(pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL))
        manifest["artifact_sha256"] = sha256_file(artifact)
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(RuntimeError, match="native prompt artifact differs"):
            prepare.load_prepared_panel(root, **kwargs)


# --- Receiver placement: the payload sits inside the user turn.


class _NativePromptTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) + 1 for char in text]

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        result: dict[str, object] = {"input_ids": self.encode(text)}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result

    def apply_chat_template(self, messages: list[dict[str, str]], **_kwargs: object) -> str:
        return "".join(
            f"<{row['role']}>{row['content']}" for row in messages if row["role"] == "user"
        )


def test_nemotron_in_turn_receiver_prompt() -> None:
    """Nemotron splices payload rows into the user turn; the other lanes prepend."""
    from rcc.models.qwen import QWEN_FAMILY
    from rcc.models.qwen.capture import (
        QwenFlatPayload,
        latent_plan_sha256,
        selected_indices_sha256,
        tensor_content_sha256,
    )
    from rcc.models.qwen.prompts import QwenManagerPromptRecord
    from rcc.models.qwen.receiver import token_embedding_rows

    assert QWEN_FAMILY.payload_in_user_turn is False
    assert NEMOTRON_FAMILY.payload_in_user_turn is True
    rows = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    arm = "latent_query_support_r2"
    keeps = ((0, 1), (0, 1), (0, 1))
    payload = QwenFlatPayload(
        rows=rows,
        semantic_arm=arm,
        latent_plan_sha256=latent_plan_sha256(arm, family=NEMOTRON_FAMILY),
        keeps=keeps,
        rows_by_worker=(2, 2, 2),
        selected_indices_sha256=selected_indices_sha256(keeps),
        tensor_sha256=tensor_content_sha256(rows),
        family=NEMOTRON_FAMILY,
    )
    weight = torch.arange(256 * 4, dtype=torch.float32).reshape(256, 4)
    manager = token_embedding_rows(weight, (65, 66))
    prepend = prepare_receiver_prompt(weight, (65, 66), payload=payload)
    torch.testing.assert_close(prepend.prompt_embeds, torch.cat((rows, manager)), atol=0, rtol=0)
    assert (prepend.manager_tokens, prepend.handoff_rows, prepend.prompt_rows) == (2, 6, 8)
    headers = ((7,), (8, 9), (10,))
    spliced = prepare_receiver_prompt(
        weight, (65, 66), payload=payload, payload_slot=1, payload_headers=headers
    )
    expected = torch.cat(
        (
            manager[:1],
            token_embedding_rows(weight, (7,)),
            rows[0:2],
            token_embedding_rows(weight, (8, 9)),
            rows[2:4],
            token_embedding_rows(weight, (10,)),
            rows[4:6],
            manager[1:],
        )
    )
    torch.testing.assert_close(spliced.prompt_embeds, expected, atol=0, rtol=0)
    assert (spliced.manager_tokens, spliced.handoff_rows, spliced.prompt_rows) == (6, 6, 12)
    assert spliced.payload_rows == 6 and spliced.payload_tensor_sha256 == payload.tensor_sha256
    with pytest.raises(ValueError, match="one header per worker"):
        prepare_receiver_prompt(weight, (65, 66), payload=payload, payload_slot=1)
    with pytest.raises(ValueError, match="inside the manager prompt"):
        prepare_receiver_prompt(
            weight, (65, 66), payload=payload, payload_slot=0, payload_headers=headers
        )
    with pytest.raises(ValueError, match="travel together"):
        QwenManagerPromptRecord(
            text="x",
            token_ids=(1, 2),
            question_token_ids=(2,),
            sha256=hashlib.sha256(b"x").hexdigest(),
            payload_slot=1,
        )
    record = QwenManagerPromptRecord(
        text="x",
        token_ids=(1, 2, 3),
        question_token_ids=(3,),
        sha256=hashlib.sha256(b"x").hexdigest(),
        payload_slot=1,
        payload_headers=headers,
    )
    assert record.payload_slot == 1


def test_nemotron_longbench_terminal_slot() -> None:
    """The native LongBench manager exposes one slot for the terminal chain block."""
    from rcc.benchmarks.longbench_v2.data import ChainItem
    from rcc.models.qwen.prompts import manager_prompt_record

    item = ChainItem(
        "native-slot",
        "Which year?",
        ("alpha", "beta", "gamma", "delta"),
        ((1,), (2,), (3,), (4,)),
        ("a", "b", "c", "d"),
        "source",
        4,
        0,
        "domain",
        "easy",
        "A",
    )
    record = manager_prompt_record(
        item,
        _NativePromptTokenizer(),
        family=NEMOTRON_FAMILY,
        profile=NEMOTRON_LONGBENCH_RERANK,
        channel="latent",
        payload_slot=True,
    )
    assert record.payload_slot is not None
    assert record.payload_headers is not None and len(record.payload_headers) == 1


# --- The compiler lifecycle and its environment.


def test_runtime_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require explicit model policy, repair only cleanup, and reject later drift."""
    from rcc.models.nemotron import runtime_policy as policy
    from rcc.models.qwen.backend import QwenBackendSettings, effective_engine_kwargs

    state = dict(policy.PRECISION_POLICY)
    compilation = SimpleNamespace(inductor_compile_config={"deterministic": True})
    module = SimpleNamespace(
        compilation_config=compilation,
        aot_compiled_fn=object(),
        compiled=True,
        was_aot_compile_fn_loaded_from_disk=False,
    )
    model = SimpleNamespace(named_modules=lambda: [("model", module)])
    config = SimpleNamespace(
        compilation_config=compilation,
        model_config=SimpleNamespace(enforce_eager=False),
    )
    llm = SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=config))
    with monkeypatch.context() as patch:
        for key, value in policy.ENVIRONMENT.items():
            patch.setenv(key, value)
        patch.setattr(policy, "precision", lambda: dict(state))
        patch.setattr(
            policy, "_restore_inductor", lambda: state.update(inductor_deterministic=True)
        )
        patch.setattr(policy, "_resident_model", lambda _: model)
        patch.setattr(
            policy,
            "_resolved_flags",
            lambda: {
                key: value for key, value in policy.ENVIRONMENT.items() if key.startswith("VLLM_")
            },
        )
        before = policy.before_engine()
        state["inductor_deterministic"] = False
        record = policy.after_engine(llm, before, capture=False)
        assert record["after_init_before_restore"]["inductor_deterministic"] is False
        assert record["after_restore"]["inductor_deterministic"] is True
        observed = policy.attest(llm, record, capture=False)
        assert observed["model_modules"]["model"]["has_aot_callable"] is True
        state["inductor_deterministic"] = False
        with pytest.raises(RuntimeError, match="precision drift") as failure:
            policy.attest(llm, record, capture=False)
        assert '"inductor_deterministic": {"actual": false, "expected": true}' in str(failure.value)
        assert state["inductor_deterministic"] is False
        state["inductor_deterministic"] = True
        state["matmul_allow_tf32"] = True
        with pytest.raises(RuntimeError, match="precision changed"):
            policy.after_engine(llm, before, capture=False)
        with pytest.raises(RuntimeError, match="initial precision"):
            policy.before_engine()
        state["matmul_allow_tf32"] = False
        compilation.inductor_compile_config = {"deterministic": False}
        with pytest.raises(RuntimeError, match="model compiler"):
            policy.after_engine(llm, before, capture=False)
        config.model_config.enforce_eager = True
        module.aot_compiled_fn = None
        compilation.inductor_compile_config = {"deterministic": True}
        eager = policy.after_engine(llm, before, capture=True)
        assert eager["mode"] == "eager_capture"
        compilation.inductor_compile_config = {"deterministic": True}
        config.model_config.enforce_eager = False
        module.aot_compiled_fn = object()
        _check_backend_lifecycle(patch, state, llm)
        patch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
        with pytest.raises(RuntimeError, match="environment"):
            policy.before_engine()

    for settings in (
        QwenBackendSettings(),
        QwenBackendSettings(receiver=True),
        QwenBackendSettings(capture=True),
    ):
        for arm in ("text_primary", "text_medium", "text_small"):
            sender = NEMOTRON_FAMILY.sender_family(arm).profile
            kwargs = effective_engine_kwargs(
                sender.checkpoint,
                sender.revision,
                settings=settings,
                family=NEMOTRON_FAMILY,
                tokenizer=sender.tokenizer,
                tokenizer_revision=sender.tokenizer_revision,
            )
            assert kwargs["compilation_config"] == {
                "inductor_compile_config": {"deterministic": True}
            }
            assert kwargs["max_num_seqs"] == (1 if settings.capture else 16)
            if settings.capture:
                assert kwargs["enforce_eager"] is True
                assert kwargs["max_num_batched_tokens"] == 4096


def _check_backend_lifecycle(
    patch: pytest.MonkeyPatch,
    state: dict[str, Any],
    llm: Any,
) -> None:
    """The factory applies the policy before vLLM and close releases on rejection."""
    from rcc.models.qwen import backend as shared
    from rcc.run.nemotron.compiler_policy import close_backend

    def construct(**kwargs: Any) -> Any:
        assert state["inductor_deterministic"] is True
        assert kwargs["compilation_config"]["inductor_compile_config"]["deterministic"] is True
        state["inductor_deterministic"] = False
        return llm

    original_import = shared.importlib.import_module
    patch.setattr(
        shared.importlib,
        "import_module",
        lambda name: SimpleNamespace(LLM=construct) if name == "vllm" else original_import(name),
    )
    patch.setattr(shared, "version", lambda _: NEMOTRON_FAMILY.profile.runtime.vllm)
    model = NEMOTRON_FAMILY.profile
    instance = shared.build_qwen_backend(
        model.checkpoint,
        model.revision,
        settings=shared.QwenBackendSettings(receiver=True),
        family=NEMOTRON_FAMILY,
        tokenizer=model.tokenizer,
        tokenizer_revision=model.tokenizer_revision,
    )
    assert state["inductor_deterministic"] is True
    records: list[dict[str, Any]] = []
    close_backend(instance, records.append, lane="nemotron", policy="r8", role="receiver")
    assert records[0]["phase"] == "nemotron_receiver_compiler_complete"
    assert instance._llm is None
    instance._llm = llm
    state["inductor_deterministic"] = False
    with pytest.raises(RuntimeError, match="precision drift"):
        close_backend(instance, records.append, lane="nemotron", policy="r8", role="receiver")
    assert instance._llm is None
    assert len(records) == 1
    state["inductor_deterministic"] = True


# --- The LongBench registration and native results.


def test_native_chain_plans(tmp_path: Path) -> None:
    """Bind four hops and the native roster while keeping the original model protocol."""
    del tmp_path
    configs = Path(__file__).resolve().parents[1] / "configs"
    rerank_config = configs / "nemotron-longbench-coa-easy50-rerank-n50.toml"
    paper = load_and_resolve(rerank_config, git_commit="0" * 40)
    text_paper = load_and_resolve(
        configs / "nemotron-longbench-coa-easy50-text-n50.toml", git_commit="0" * 40
    )
    bounded_paper = load_and_resolve(
        configs / "nemotron-longbench-coa-easy50-bounded-n50.toml", git_commit="0" * 40
    )
    # Three profiles read the easy fifty natively: six rerank arms, five bounded arms (no
    # 64K rung: one item's worst hop exceeds the window on this tokenizer), and the floor
    # with three text arms. A plan's roster is the lane cut to the arms its profile holds.
    for plan, profile, total in (
        (paper, NEMOTRON_LONGBENCH_RERANK, 900),
        (text_paper, NEMOTRON_LONGBENCH_TEXT, 600),
        (bounded_paper, NEMOTRON_LONGBENCH_BOUNDED, 750),
    ):
        assert plan.benchmark is profile and plan.topology is CHAIN_T4
        assert plan.model.model_id == NEMOTRON.model_id and plan.decode == NEMOTRON.decode
        registered = {arm.arm_id for arm in profile.arms}
        assert plan.physical_arms == tuple(
            arm for arm in NEMOTRON.physical_arms if arm.semantic_arm in registered
        )
        assert plan.expected_rows.total == total
        assert plan.execution_question_ids == EASY50_ORDER[: plan.item_count]
    assert paper.execution_question_ids == text_paper.execution_question_ids == EASY50_ORDER
    assert len(paper.arms) == 6 and len(text_paper.arms) == 4
    assert paper.benchmark.workers_per_item == text_paper.benchmark.workers_per_item == 4
    # The frozen geometry is per profile: the rerank block prices the rerank
    # ladder and the text block the sender context. Each block's item table is
    # keyed by qid, so coverage is by membership and count.
    geometry = frozen_geometry(paper.benchmark)
    assert set(geometry["items"]) == set(paper.benchmark.question_ids)
    assert len(geometry["items"]) == len(paper.benchmark.question_ids) == 50
    assert geometry["capture_window"] == 131_072
    assert set(geometry["rerank_ratios"]) == {str(ratio) for ratio in paper.benchmark.ratios}
    assert geometry["rerank_ratios"]["4"]["worst_hop_rows"] == 115_114
    assert not any(row["na_items"] for row in geometry["rerank_ratios"].values())
    assert "bounded_rows" not in geometry
    text_geometry = frozen_geometry(text_paper.benchmark)
    assert set(text_geometry["items"]) == set(text_paper.benchmark.question_ids)
    assert len(text_geometry["items"]) == 50
    assert "rerank_ratios" not in text_geometry and "bounded_rows" not in text_geometry
    assert max(text_geometry["text_sender_context"].values()) == max(
        geometry["text_sender_context"].values()
    )
    for profile in (paper.benchmark, text_paper.benchmark):
        config = build_config(profile)
        assert (config.tokenizer_checkpoint, config.tokenizer_revision) == (
            NEMOTRON.tokenizer,
            NEMOTRON.tokenizer_revision,
        )
    # Every native profile reads the 16K report ceiling; the sender's draw is
    # otherwise the FanOutQA one.
    for arm in ("text_primary", "text_medium", "text_small"):
        family = NEMOTRON_FAMILY.sender_family(arm)
        report = QwenDecodeSpec("report", family, profile=text_paper.benchmark).backend_sampling()
        fanout_report = QwenDecodeSpec(
            "report", family, profile=FANOUTQA_NATURAL_DEV50
        ).backend_sampling()
        assert report["max_tokens"] == 16_000
        assert {key: value for key, value in report.items() if key != "max_tokens"} == {
            key: value for key, value in fanout_report.items() if key != "max_tokens"
        }
    with pytest.raises(ValueError, match="cannot run the LongBench chain"):
        resolve_plan(
            replace(load_run_config(rerank_config), model="qwen3-8b"),
            ResolutionContext(git_commit="0" * 40),
        )
    _check_native_result()
    _check_native_tokenization()


def _check_native_tokenization() -> None:
    source_sha = hashlib.sha256(b"abcdefgh").hexdigest()
    ledger = [
        {
            "source_sha256": source_sha,
            "chunk_sha256": hashlib.sha256(chunk.encode()).hexdigest(),
            "char_start": i * 2,
            "char_end": (i + 1) * 2,
            "update": i + 1,
            "standalone_qwen_source_tokens": 999,
            "whole_token_end": (i + 1) * 2,
        }
        for i, chunk in enumerate(("ab", "cd", "ef", "gh"))
    ]
    item = build_item(
        raw_row("native-qid", "abcdefgh"),
        ledger,
        FakeTokenizer(width=1),
        stratum=3,
        expected_count_field=None,
    )
    assert item.chunk_tokens == (2, 2, 2, 2) and item.source_sha256 == source_sha


def _check_native_result() -> None:
    item = chain_item("native-result", 65)
    profile = replace(NEMOTRON_LONGBENCH_TEXT, question_ids=(item.qid,))
    family = NEMOTRON_FAMILY
    prompt = prepare_receiver_prompt(torch.ones(128, 4), (65, 66), family=family, profile=profile)
    raw = "<think>x</think>B"
    answers = tuple(
        QwenVisibleAnswer(tag, seed, raw, raw, "B", (100, 2), (100,), "stop", True, 0)
        for tag, seed in zip(
            profile.sample_tags, profile.answer_seeds(item.qid, "issue_only"), strict=True
        )
    )
    row = build_result_row(
        item,
        "issue_only",
        answers,
        prompt,
        worker_prompt_tokens=item.chunk_tokens,
        latency=ItemLatency(0, 0, 0, 0, 0, 0, (0, 0, 0), (0, 0, 0), (0, 0, 0)),
        family=family,
        profile=profile,
    )
    tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: raw)
    assert validate_and_rescore_result(row, item, tokenizer, family=family, profile=profile) == row
    with pytest.raises(RuntimeError, match="Native answer decode differs"):
        validate_and_rescore_result(
            {**row, "decode": {**row["decode"], "temperature": 0}},
            item,
            tokenizer,
            family=family,
            profile=profile,
        )
    for arm in ("text_primary", "text_medium", "text_small"):
        fields = {
            "report_decode": QwenDecodeSpec(
                "report", family.sender_family(arm), profile=profile
            ).to_dict()
        }
        validate_native_report(item, fields, arm=arm, family=family, profile=profile)
        with pytest.raises(RuntimeError, match="Native chain report decode drifted"):
            validate_native_report(item, {}, arm=arm, family=family, profile=profile)
