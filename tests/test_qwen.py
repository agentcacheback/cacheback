"""The Qwen runtime: its vLLM roles, fleet workers, report scoring versions, and result rows."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from tests.longbench_coa_support import CHAIN_FIXTURE

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.fanoutqa.panel import Question
from rcc.benchmarks.fanoutqa.scoring import DEFAULT_SCORER_VERSION
from rcc.benchmarks.fanoutqa.source_padding import ProbeItem
from rcc.benchmarks.longbench_v2.data import ChainItem
from rcc.benchmarks.longbench_v2.geometry import chain_wrapper_allowances, rewrite_prompt_ceiling
from rcc.models.nemotron import NEMOTRON_FAMILY
from rcc.models.qwen import QWEN, QWEN_FAMILY, backend
from rcc.models.qwen import text as qwen_text
from rcc.models.qwen.backend import QwenBackendSettings, QwenVllmBackend, build_qwen_backend
from rcc.models.qwen.capture import (
    QwenFlatPayload,
    chain_plan_sha256,
    latent_plan_sha256,
    selected_indices_sha256,
    tensor_content_sha256,
)
from rcc.models.qwen.engine import QWEN_ENGINE_ROUTE
from rcc.models.qwen.receiver import (
    prepare_receiver_prompt,
    receiver_batches,
    receiver_requests,
    reconstruct_visible_answers,
)
from rcc.models.qwen.results import build_result_row, validate_and_rescore_result
from rcc.models.qwen.text import QwenDecodeSpec
from rcc.models.route import RouteFamily, tokenizer_decoder
from rcc.run import rescore
from rcc.run.fleet import report as split_report
from rcc.run.fleet.answer_closer import ANSWER_CLOSING_TOKEN_BUDGET
from rcc.run.fleet.contract import WorkItem
from rcc.run.fleet.latency import (
    COMPUTE_TIMING_SCOPE,
    ITEM_LATENCY_FIELDS,
    ItemLatency,
)
from rcc.run.fleet.merge import SCORE_FIELDS, CellLedger
from rcc.run.qwen import producers as qwen_producers
from rcc.run.qwen import report as route_report
from rcc.run.qwen import worker as qwen_worker
from rcc.run.qwen.report import _policy_grid
from rcc.run.qwen.runtime import ENGINE_ROUTE
from rcc.topologies.chain.layout import CHAIN_TERMINAL_LAYOUT

# --- The pinned vLLM roles.


@pytest.fixture
def qwen_llm_kwargs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []

    class LLM:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    class KVTransferConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    monkeypatch.setattr(backend, "version", lambda _package: "0.11.1")
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=LLM))
    monkeypatch.setitem(
        sys.modules,
        "vllm.config",
        SimpleNamespace(KVTransferConfig=KVTransferConfig),
    )
    return captured


def test_qwen_capture_uses_the_50k_connector_geometry(
    qwen_llm_kwargs: list[dict[str, Any]],
) -> None:
    build_qwen_backend(
        "Qwen/Qwen3-8B",
        "revision",
        settings=QwenBackendSettings(capture=True),
        family=QWEN_FAMILY,
        tokenizer="Qwen/Qwen3-8B",
        tokenizer_revision="revision",
    )
    assert len(qwen_llm_kwargs) == 1
    kwargs = qwen_llm_kwargs[0]

    assert kwargs["max_model_len"] == 50_000
    assert kwargs["max_num_batched_tokens"] == 50_000
    assert kwargs["max_num_seqs"] == 16
    assert kwargs["gpu_memory_utilization"] == 0.40
    assert kwargs["disable_log_stats"] is True
    assert kwargs["enable_prompt_embeds"] is False
    assert kwargs["enable_prefix_caching"] is False
    assert kwargs["enforce_eager"] is True
    assert kwargs["disable_hybrid_kv_cache_manager"] is True
    capture_route = ENGINE_ROUTE["capture"]
    assert isinstance(capture_route, dict)
    assert kwargs["hf_overrides"] == capture_route["hf_overrides"]
    assert set(kwargs) == {
        "model",
        "revision",
        "tokenizer",
        "tokenizer_revision",
        "dtype",
        "gpu_memory_utilization",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "hf_overrides",
        "enable_prompt_embeds",
        "enable_prefix_caching",
        "enforce_eager",
        "disable_hybrid_kv_cache_manager",
        "disable_log_stats",
        "kv_transfer_config",
    }
    connector = kwargs["kv_transfer_config"]
    assert vars(connector) == {
        "kv_connector": "RCCConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": "rcc.injector.connector",
    }
    assert capture_route["max_model_len"] == 50_000
    assert capture_route["gpu_memory_utilization"] == 0.40
    assert capture_route["max_num_seqs"] == 16
    assert capture_route["enable_prompt_embeds"] is False
    assert capture_route["disable_log_stats"] is True


def test_qwen_serving_roles_keep_metrics_and_full_yarn_context(
    qwen_llm_kwargs: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for settings in (QwenBackendSettings(), QwenBackendSettings(receiver=True)):
        build_qwen_backend(
            "Qwen/Qwen3-8B",
            "revision",
            settings=settings,
            family=QWEN_FAMILY,
            tokenizer="Qwen/Qwen3-8B",
            tokenizer_revision="revision",
        )
        kwargs = qwen_llm_kwargs[-1]

        assert kwargs["max_model_len"] == 131_072
        assert kwargs["max_num_seqs"] == 16
        assert kwargs["gpu_memory_utilization"] == 0.80
        assert kwargs["disable_log_stats"] is False
        assert kwargs["enable_prompt_embeds"] is settings.receiver
        assert kwargs["hf_overrides"] == {
            "max_position_embeddings": 131_072,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32_768,
            },
        }

    pool_tokens = 337_808
    receiver_backend = SimpleNamespace(
        warmup_decode=lambda _prompt, *, seed: None,
        resident_model=lambda: SimpleNamespace(
            get_input_embeddings=lambda: SimpleNamespace(weight=torch.zeros(2, 2))
        ),
        kv_pool_tokens=lambda: pool_tokens,
    )
    tracked: dict[str, Any] = {}
    monkeypatch.setattr(qwen_worker, "build_qwen_backend", lambda *_a, **_k: receiver_backend)
    monkeypatch.setattr(qwen_worker, "VllmEngineHandle", lambda *_a, **_k: SimpleNamespace())
    monkeypatch.setattr(qwen_worker, "StreamDecoder", lambda *_a, **_k: SimpleNamespace())
    monkeypatch.setattr(
        qwen_worker,
        "ItemStreamTracker",
        lambda _decoder, gate: tracked.setdefault("gate", gate) or SimpleNamespace(),
    )
    policy = next(
        arm.policy for arm in QWEN.physical_arms if arm.semantic_arm == "latent_query_support_r2"
    )
    receiver = qwen_worker.QwenReceiver(
        policy=policy,
        tokenizer=SimpleNamespace(apply_chat_template=lambda *_a, **_k: "warm"),
        items={},
        arm_root=Path("."),
        bank=lambda _row: None,
        family=QWEN_FAMILY,
        profile=FANOUTQA_NATURAL_DEV50,
    )
    receiver.warm()
    receiver.pending = dict.fromkeys(("a", "b", "c"))  # type: ignore[assignment]
    assert receiver.claim_limit == qwen_worker.QWEN_ENGINE_MAX_NUM_SEQS
    assert receiver.can_accept() and pool_tokens // FANOUTQA_NATURAL_DEV50.max_model_len == 2
    gate = tracked["gate"]
    admitted = sum(gate.try_admit(f"q{index}", 1) for index in range(16))
    assert admitted < receiver.claim_limit and not gate.try_admit("full", 1)

    submitted: list[Any] = []

    class SamplingParams:
        def __init__(self, **values: object) -> None:
            self.values = values

    class Tokenizer:
        def __call__(self, prompt: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
            ids = [ord(char) for char in prompt]
            return {"input_ids": [128000, *ids] if add_special_tokens else ids}

    class LLM:
        def get_tokenizer(self) -> Tokenizer:
            return Tokenizer()

        def generate(self, prompts: list[Any], **_kwargs: object) -> list[Any]:
            submitted.extend(prompts)
            prompt_ids = list(prompts[0]["prompt_token_ids"])
            return [
                SimpleNamespace(
                    prompt_token_ids=prompt_ids,
                    outputs=(
                        SimpleNamespace(text="tail", token_ids=(41, 42), finish_reason="stop"),
                    ),
                    num_cached_tokens=0,
                    metrics=None,
                )
            ]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=SamplingParams))
    request = cast(
        Any,
        SimpleNamespace(
            request_id="continued-report",
            seed=17,
            decode=SimpleNamespace(backend_sampling=lambda: {"temperature": 0.6}),
        ),
    )
    backend_instance = QwenVllmBackend(LLM(), QwenBackendSettings(), QWEN_FAMILY)

    result = backend_instance.decode_token_ids_full(((7, 8, 151668),), (request,))[0]

    assert submitted == [{"prompt_token_ids": [7, 8, 151668]}]
    assert tuple(result.prompt_token_ids) == (7, 8, 151668)
    assert tuple(result.token_ids) == (41, 42)
    # A rendered chat is served as ids with no special token added: a Llama 3
    # tokenizer would otherwise prepend a second begin-of-text.
    submitted.clear()
    text_result = backend_instance.decode_text_full(("hi",), (request,))[0]
    assert submitted == [{"prompt_token_ids": [ord("h"), ord("i")]}]
    assert tuple(text_result.prompt_token_ids) == (ord("h"), ord("i"))

    class DriftLLM(LLM):
        def generate(self, prompts: list[Any], **kwargs: object) -> list[Any]:
            outputs = super().generate(prompts, **kwargs)
            outputs[0].prompt_token_ids = [7, 999, 151668]
            return outputs

    drifted = QwenVllmBackend(DriftLLM(), QwenBackendSettings(), QWEN_FAMILY)
    with pytest.raises(RuntimeError, match="prompt token ids that differ"):
        drifted.decode_token_ids_full(((7, 8, 151668),), (request,))

    class MalformedLLM(LLM):
        def generate(self, prompts: list[Any], **kwargs: object) -> list[Any]:
            outputs = super().generate(prompts, **kwargs)
            outputs[0].prompt_token_ids = [7, "8", 151668]
            return outputs

    malformed = QwenVllmBackend(MalformedLLM(), QwenBackendSettings(), QWEN_FAMILY)
    with pytest.raises(RuntimeError, match="malformed prompt ids"):
        malformed.decode_token_ids_full(((7, 8, 151668),), (request,))

    assert QWEN_FAMILY.profile.runtime.python == "3.10.12"


# --- The fleet workers and their crash prefixes.

# One prepared FanOutQA N50 question id, so the sealed report seeds resolve.
_QID = "95de313fdfef9d01"


# Prose that spells out the opening delimiter. It tokenizes as ordinary text,
# never as the reserved opener id, so it must not read as an open block.
_SPOOF = "the coordinator asked me to write <think> literally, so here it is"


class _Codec:
    """A reversible fake tokenizer: one id per text piece, decode is exact.

    Each reasoning delimiter is one reserved id, prose spelling a delimiter
    creates ordinary ids, and a stop finish banks the stop id after the content.
    """

    def __init__(self) -> None:
        self.pieces: dict[int, str] = {
            QWEN_FAMILY.think_open_token_ids[0]: QWEN_FAMILY.think_open,
            QWEN_FAMILY.think_close_token_ids[0]: QWEN_FAMILY.think_close,
            QWEN_FAMILY.stop_token_ids[0]: "",
        }
        self._next = 900_000

    def encode(self, pieces: Sequence[str]) -> list[int]:
        """Return one id per piece, minting a new id the first time it appears."""
        ids: list[int] = []
        for piece in pieces:
            token = next((key for key, text in self.pieces.items() if text == piece), None)
            if token is None:
                token = self._next
                self._next += 1
                self.pieces[token] = piece
            ids.append(token)
        return ids

    def decode(self, tokens: Any, **_kwargs: Any) -> str:
        """Rebuild the exact text the sender emitted for these ids."""
        return "".join(self.pieces[int(token)] for token in tokens)


class _ScriptedSender:
    """Replay scripted draws through the codec and charge each decode call.

    Each scripted worker is a list of text pieces, and the sender banks the
    joined text, the ids, and on a stop finish the trailing stop id.
    """

    def __init__(
        self,
        script: Sequence[Sequence[Sequence[str]]],
        costs: Sequence[float],
        *,
        finish_reasons: Sequence[str] = (),
    ) -> None:
        self.codec = _Codec()
        self.script = [tuple(draw) for draw in script]
        self.costs = list(costs)
        self.finish_reasons = list(finish_reasons)
        self.prompts: list[tuple[str | tuple[int, ...], ...]] = []
        self.requests: list[tuple[Any, ...]] = []
        self.clock = 0.0

    def decode_text_full(self, prompts: Sequence[str], requests: Sequence[Any]) -> list[Any]:
        """Answer one draw or one injected continuation and advance the clock."""
        return self._decode(tuple(prompts), requests)

    def decode_token_ids_full(
        self,
        prompts: Sequence[Sequence[int]],
        requests: Sequence[Any],
    ) -> list[Any]:
        """Answer a continuation from the exact caller-owned token ids."""
        return self._decode(tuple(tuple(prompt) for prompt in prompts), requests)

    def _decode(
        self,
        prompts: tuple[str | tuple[int, ...], ...],
        requests: Sequence[Any],
    ) -> list[Any]:
        """Replay one scripted batch while retaining its submitted prompt ids."""
        self.prompts.append(prompts)
        self.requests.append(tuple(requests))
        self.clock += self.costs.pop(0)
        draw = self.script.pop(0)
        reason = self.finish_reasons.pop(0) if self.finish_reasons else "stop"
        completions: list[Any] = []
        for prompt, request, pieces in zip(prompts, requests, draw, strict=True):
            ids = self.codec.encode(pieces)
            if reason == "stop":
                ids = [*ids, QWEN_FAMILY.stop_token_ids[0]]
            completions.append(
                SimpleNamespace(
                    request_id=request.request_id,
                    prompt_token_ids=tuple(prompt)
                    if not isinstance(prompt, str)
                    else tuple(self.codec.encode((prompt,))),
                    text="".join(pieces),
                    n_tokens=len(ids),
                    token_ids=tuple(ids),
                    finish_reason=reason,
                    num_cached_tokens=0,
                    queued_ts=None,
                    scheduled_ts=None,
                    first_token_ts=None,
                )
            )
        return completions


def _bundle(
    sender: _ScriptedSender,
    monkeypatch: pytest.MonkeyPatch,
    *,
    family: RouteFamily = QWEN_FAMILY,
) -> qwen_text.QwenReportBundle:
    monkeypatch.setattr(qwen_text.time, "perf_counter", lambda: sender.clock)
    return qwen_text.generate_report_bundle(
        cast(Any, sender),
        ("worker-0", "worker-1", "worker-2"),
        qid=_QID,
        semantic_arm="text_primary",
        family=family,
        decoder=tokenizer_decoder(sender.codec),
    )


def _rescore_report_row(
    bundle: qwen_text.QwenReportBundle,
    sender: _ScriptedSender,
    **overrides: object,
) -> None:
    """Run `_require_qwen_text_report`, the shipped validator, over one bundle.

    It rebuilds the outputs from the banked ids and refuses any field that
    differs. It reads only `plan.benchmark`, so a sealed profile stands in.
    """
    row = {"qid": _QID, "arm": "text_primary", **bundle.result_fields(), **overrides}
    rescore._require_qwen_text_report(
        row,
        cast(Any, SimpleNamespace(benchmark=FANOUTQA_NATURAL_DEV50)),
        qid=_QID,
        arm="text_primary",
        tokenizer=sender.codec,
        family=QWEN_FAMILY,
    )


def test_latent_retry_recaptures_despite_an_old_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arm = next(arm for arm in QWEN.physical_arms if arm.selector == "support")
    keeps = ((0,), (0,), (0,))
    scores = (torch.ones(50_040), torch.ones(50_040), torch.ones(50_040))
    calls: list[str] = []
    manifest_path = tmp_path / "payloads" / "qid.payload.json"

    def produce(*_args, **_kwargs):
        calls.append("capture")
        if len(calls) == 2:
            assert not manifest_path.exists()
        rows = torch.full((3, 4), len(calls), dtype=torch.bfloat16)
        return SimpleNamespace(
            payload=SimpleNamespace(rows=rows),
            selector_scores=scores,
            result_fields=lambda: {
                "producer_backend": "engine",
                "producer_route": QWEN_ENGINE_ROUTE,
                "payload_semantic_arm": arm.semantic_arm,
                "payload_plan_sha256": latent_plan_sha256(arm.semantic_arm, family=QWEN_FAMILY),
                "payload_tensor_sha256": tensor_content_sha256(rows),
                "selected_indices_sha256": selected_indices_sha256(keeps),
                "keeps_by_worker": [list(keep) for keep in keeps],
                "latent_rows_by_worker": [1, 1, 1],
                "selector_score_name": arm.selector,
                "selector_score_tensor_sha256": [tensor_content_sha256(score) for score in scores],
            },
        )

    monkeypatch.setattr(qwen_producers, "produce_latent_payload", produce)
    producer = qwen_worker.QwenLatentProducer(
        policy=arm.policy,
        tokenizer=SimpleNamespace(),
        items=cast(dict[str, ProbeItem], {"qid": object()}),
        payload_root=tmp_path / "payloads",
        bank=lambda _row: None,
        family=QWEN_FAMILY,
        profile=FANOUTQA_NATURAL_DEV50,
    )
    producer.producer = cast(Any, SimpleNamespace())
    item = WorkItem("qid", 0)

    first = producer.produce(item)
    rows_path = tmp_path / "payloads" / "rows" / "qid.pt"
    first_rows = rows_path.read_bytes()
    second = producer.produce(item)

    assert second.path == first.path
    assert rows_path.read_bytes() != first_rows
    assert calls == ["capture", "capture"]
    # The judger ablation replaces the question the selector scores with and
    # nothing else: the workers render the item's own question either way.
    from rcc.models.qwen import producer as qwen_producer

    real = ProbeItem(
        qid="qid", question="real?", shards=((1,), (2,), (3,)), question_obj=cast(Any, None)
    )
    seen: dict[str, str] = {}
    monkeypatch.setattr(
        qwen_producer,
        "worker_prompts",
        lambda item, *_a, **_k: (seen.__setitem__("worker", item.question), ())[1],
    )
    monkeypatch.setattr(
        qwen_producer,
        "manager_prompt_record",
        lambda item, *_a, **_k: (
            seen.__setitem__("judger", item.question),
            SimpleNamespace(token_ids=(1, 2), question_token_ids=(1,)),
        )[1],
    )

    class StopAtDeviceError(Exception):
        pass

    class Producer:
        @property
        def device(self) -> torch.device:
            raise StopAtDeviceError

    question = "Why was Le Chaton Fat denied a small-business loan?"
    for judger_question, expected in ((None, "real?"), (question, question)):
        with pytest.raises(StopAtDeviceError):
            qwen_producer.produce_latent_payload(
                cast(Any, Producer()),
                SimpleNamespace(),
                real,
                semantic_arm=arm.semantic_arm,
                family=QWEN_FAMILY,
                profile=replace(FANOUTQA_NATURAL_DEV50, judger_question=judger_question),
            )
        assert seen == {"worker": "real?", "judger": expected}


def test_the_report_codec_times_the_shipped_draw_and_injects_one_closer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The report clock and the closer injection, over the text ladder.

    The published span is the shipped draw alone, the discarded draws banked
    beside it. A draw ending unclosed is continued once, closer injected.
    """

    closed_zero = ("<think>", "t0", "</think>", "report zero")
    closed_one = ("<think>", "t1", "</think>", "report one")
    injected_sender = _ScriptedSender(
        script=(
            (closed_zero, closed_one, ("<think>", "t2 and the turn ended here")),
            (("report two",),),
        ),
        costs=(10.0, 3.0),
    )
    injected = _bundle(injected_sender, monkeypatch)

    assert injected.reports == ("report zero", "report one", "report two")
    assert injected.injected_by_worker == (False, False, True)
    assert injected.thinking_closed == (True, True, True)
    assert injected.raw_outputs[2] == "<think>t2 and the turn ended here</think>report two"
    # The continuation is the same draw: same prompt, same signed request and
    # seed, the closer injected as exact ids, its cost inside the published
    # span, and the one sampling field that moves is the registered cap.
    expected_continuation = tuple(
        injected_sender.codec.encode(
            ("worker-2", "<think>", "t2 and the turn ended here", "</think>")
        )
    )
    assert injected_sender.prompts[1] == (expected_continuation,)
    assert injected_sender.codec.encode(
        ("worker-2<think>t2 and the turn ended here</think>",)
    ) != list(expected_continuation)
    head_request = injected_sender.requests[0][2]
    tail_request = injected_sender.requests[1][0]
    assert tail_request.seed == head_request.seed
    assert tail_request.sample_tag == head_request.sample_tag
    assert head_request.decode.max_tokens == FANOUTQA_NATURAL_DEV50.report_ceiling
    assert tail_request.decode.max_tokens == QWEN_FAMILY.closing_token_budget
    assert injected.generation_s == pytest.approx(13.0)
    assert (injected.draws_n, injected.redraw_wall_s) == (1, 0.0)
    fields = injected.result_fields()
    assert fields["injected"] is True
    assert fields["report_injected_by_worker"] == [False, False, True]
    qwen_text.validate_report_ladder_fields(fields)
    with pytest.raises(ValueError, match="single-draw"):
        qwen_text.validate_report_ladder_fields({**fields, "redraw_wall_s": 1.0})
    with pytest.raises(ValueError, match="injection summary"):
        qwen_text.validate_report_ladder_fields({**fields, "injected": False})
    with pytest.raises(ValueError, match="draw count"):
        qwen_text.validate_report_ladder_fields({**fields, "draws_n": 2})
    with pytest.raises(ValueError, match="generation wall clock"):
        qwen_text.validate_report_ladder_fields({**fields, "report_generation_s": -1.0})

    # The rescore over the injected bundle: the banked ids carry the injected
    # closer and drop the head's stop id, and the banked finish reason is the
    # head's, so the ceiling rule holds over a continuation of its own.
    _rescore_report_row(injected, injected_sender)
    banked_ids = cast(list[list[int]], fields["report_token_ids_by_worker"])
    assert QWEN_FAMILY.think_close_token_ids[0] in banked_ids[2]
    assert QWEN_FAMILY.stop_token_ids[0] not in banked_ids[2][:-1]
    assert cast(list[str], fields["report_finish_reasons"])[2] == "stop"
    assert cast(list[int], fields["report_tokens_by_worker"])[2] == len(banked_ids[2])

    completion = cast(Any, SimpleNamespace)
    merged = qwen_text._merge_continuation(
        completion(
            request_id="head",
            prompt_token_ids=(11, 12),
            text="<think>head",
            n_tokens=2,
            token_ids=(QWEN_FAMILY.think_open_token_ids[0], 13),
            finish_reason="stop",
            num_cached_tokens=0,
            queued_ts=None,
            scheduled_ts=None,
            first_token_ts=None,
        ),
        completion(
            request_id="tail",
            prompt_token_ids=(11, 12, QWEN_FAMILY.think_open_token_ids[0], 13, 151668),
            text="tail",
            n_tokens=1,
            token_ids=(14,),
            finish_reason="stop",
            num_cached_tokens=0,
            queued_ts=None,
            scheduled_ts=None,
            first_token_ts=None,
        ),
        family=QWEN_FAMILY,
    )
    assert tuple(merged.prompt_token_ids) == (11, 12)

    ladder_sender = _ScriptedSender(
        script=(
            (
                closed_zero,
                (_SPOOF,),
                ("<think>", "t2", "</think>", "report two"),
            ),
            (
                ("<think>", "u0", "</think>", "second zero"),
                ("<think>", "u1", "</think>", "second one"),
                ("<think>", "u2", "</think>", "second two"),
            ),
        ),
        costs=(7.0, 11.0),
    )
    ladder = _bundle(ladder_sender, monkeypatch)

    assert ladder.injected_by_worker == (False, False, False)
    assert [len(prompts) for prompts in ladder_sender.prompts] == [3, 3]
    _rescore_report_row(ladder, ladder_sender)

    # The trigger reads the reserved delimiter ids, not the decoded string, the
    # same spoof-resistant predicate the Ministral codec applies. The worker
    # above quotes "<think>" in prose and is redrawn, never continued.
    spoof_ids = ladder_sender.codec.encode((_SPOOF,))
    decode = tokenizer_decoder(ladder_sender.codec)
    assert QWEN_FAMILY.think_open in ladder_sender.codec.decode(spoof_ids)
    assert QWEN_FAMILY.think_open_token_ids[0] not in spoof_ids
    assert QWEN_FAMILY.thinking_is_unclosed(spoof_ids, decode=decode) is False
    assert (
        QWEN_FAMILY.thinking_is_unclosed(
            ladder_sender.codec.encode(("<think>", "t1")), decode=decode
        )
        is True
    )
    assert (
        QWEN_FAMILY.thinking_is_unclosed(
            ladder_sender.codec.encode(("<think>", "t1", "</think>")), decode=decode
        )
        is False
    )
    assert ladder.seed_tag == qwen_text.QWEN_REPORT_SEED_TAGS[1]
    assert ladder.draws_n == 2
    assert ladder.generation_s == pytest.approx(11.0)
    assert ladder.redraw_wall_s == pytest.approx(7.0)
    ladder_fields = ladder.result_fields()
    assert ladder_fields["report_generation_s"] == pytest.approx(11.0)
    with pytest.raises(RuntimeError, match="draw count"):
        _rescore_report_row(ladder, ladder_sender, draws_n=3)
    # The sum over all draws is recoverable from the two banked fields.
    all_draws_s = float(cast(float, ladder_fields["redraw_wall_s"])) + float(
        cast(float, ladder_fields["report_generation_s"])
    )
    assert all_draws_s == pytest.approx(18.0)

    # Under the sealed geometry the continuation can never overflow the engine
    # context, even from a head that spent the whole report ceiling.
    assert (
        FANOUTQA_NATURAL_DEV50.worker_prompt_tokens
        + FANOUTQA_NATURAL_DEV50.report_ceiling
        + len(QWEN_FAMILY.think_close_token_ids)
        + QWEN_FAMILY.closing_token_budget
        <= FANOUTQA_NATURAL_DEV50.max_model_len
    )

    # A draw with no room left for the closing budget is not continued: the
    # report ceiling bounds the head and the closing budget the continuation.
    # Such a draw redraws and quarantines through the ladder, never raising.
    starved_family = replace(QWEN_FAMILY, closing_token_budget=FANOUTQA_NATURAL_DEV50.max_model_len)
    starved_sender = _ScriptedSender(
        script=tuple((closed_zero, closed_one, ("<think>", "never closes")) for _ in range(3)),
        costs=(1.0, 2.0, 4.0),
    )
    starved = _bundle(starved_sender, monkeypatch, family=starved_family)

    assert starved.injected_by_worker == (False, False, False)
    assert [len(prompts) for prompts in starved_sender.prompts] == [3, 3, 3]
    assert starved.report_failed_workers == (2,)
    assert starved.reports[2] == ""
    assert (starved.draws_n, starved.generation_s) == (3, pytest.approx(4.0))
    assert starved.redraw_wall_s == pytest.approx(3.0)
    _rescore_report_row(starved, starved_sender)

    grid_rows = [
        {
            "policy": "text",
            "loose": 1.0,
            "strict": 1.0,
            "generated_tokens": 1,
            "decode_s": value,
            "decode_batched_s": value,
            "ttft_s": value,
            "tteoa_s": value,
            "policy_compute_tteoa_s": value,
            "thinking_closed_rate": 1.0,
            **bundle_fields,
        }
        for value, bundle_fields in ((2.0, fields), (999.0, starved.result_fields()))
    ]

    grid = _policy_grid(grid_rows, ("text",), QWEN_FAMILY.lane.capitalize(), SCORE_FIELDS)
    assert grid[0]["latency_n"] == 1
    assert grid[0]["quarantine_rate"] == 0.5
    assert grid[0]["tteoa_s_p50"] == grid[0]["tteoa_s_p95"] == 2.0
    assert grid[0]["report_closure_rate"] == pytest.approx(5 / 6, abs=1e-4)
    assert grid[0]["injection_activation_rate"] == 0.5


# --- The report scoring versions.

Rows = list[dict[str, Any]]


RunReport = Callable[[Path, Rows], dict[str, Any]]


def _check_cohorts(root: Path, rows: Rows, run: RunReport, *, per_seed: bool) -> None:
    selectors = {
        "question": lambda row: row["qid"] == "q0",
        "arm": lambda row: row["arm"] == rows[0]["arm"],
    }
    if per_seed:
        selectors["sample"] = lambda row: row["sample_tag"] == "s1"
    for name, selected in selectors.items():
        mixed = [dict(row) for row in rows]
        for row in mixed:
            if selected(row):
                row["scoring_version"] = "fanoutqa-other-v1"
                row.update(loose=0.0, strict=0.0)
        output = root / name
        with pytest.raises(ValueError, match="mixed FanOutQA scoring versions"):
            run(output, mixed)
        assert not (output / "report").exists(), name
    invalid = (
        None,
        "unknown",
        "fanoutqa-string-proxy-loose-strict-v1",
        "fanoutqa-gold-patch-v1",
    )
    for index, version in enumerate(invalid):
        output = root / f"invalid-{index}"
        malformed = [{**row, "scoring_version": version} for row in rows]
        with pytest.raises(ValueError, match="scoring version"):
            run(output, malformed)
        assert not (output / "report").exists()
    output = root / "unversioned"
    unversioned = [
        {key: value for key, value in row.items() if key != "scoring_version"} for row in rows
    ]
    with pytest.raises(ValueError, match="scoring version"):
        run(output, unversioned)
    assert not (output / "report").exists()
    output = root / "uniform"
    report = run(output, [dict(row) for row in rows])
    assert report["scoring_version"] == DEFAULT_SCORER_VERSION
    assert (
        json.loads((output / "report" / "report.json").read_text())["scoring_version"]
        == DEFAULT_SCORER_VERSION
    )
    grid = report.get("arm_grid", report.get("policy_grid"))
    assert grid is not None, "the report carries neither an arm grid nor a policy grid"
    assert all(row["strict"] == row["loose"] == 1.0 for row in grid)


def _check_split(root: Path) -> None:
    arms = ("issue_only", "text_primary")
    rows = [
        {
            "qid": qid,
            "arm": arm,
            "panel": "production",
            "sample_tag": sample,
            "scoring_version": DEFAULT_SCORER_VERSION,
            "strict": 1.0,
            "loose": 1.0,
        }
        for qid in ("q0", "q1")
        for arm in arms
        for sample in ("s0", "s1", "s2")
    ]
    ledgers = {arm: CellLedger(arm, ("q0", "q1"), (), 0, 3) for arm in arms}
    spec = split_report.SplitReportSpec(
        "fixture-report", "fixture", {"scoring_version": DEFAULT_SCORER_VERSION}
    )

    def run(output: Path, cohort: Rows) -> dict[str, Any]:
        for arm in arms:
            (output / "arms" / arm / "workers").mkdir(parents=True)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(split_report, "_banked_qids", lambda *_args: ("q0", "q1"))
            patch.setattr(
                split_report, "merged_split_rows", lambda *_args, **_kw: (cohort, ledgers)
            )
            return split_report.build_split_report(
                output,
                spec,
                arms,
                source_commit="0" * 40,
                completion_key=lambda row: (
                    (row["qid"], row["arm"]) if row["sample_tag"] == "s0" else None
                ),
            )

    _check_cohorts(root, rows, run, per_seed=True)


def _check_route(root: Path, family: RouteFamily) -> None:
    policies = [arm.policy for arm in family.profile.physical_arms[:2]]
    rows = [
        {
            "qid": qid,
            "arm": policy,
            "policy": policy,
            "panel": "production",
            "scoring_version": DEFAULT_SCORER_VERSION,
            "strict": 1.0,
            "loose": 1.0,
            "generated_tokens": 1,
            "finish_reasons": ["stop"] * 3,
            "thinking_closed_by_sample": [True] * 3,
        }
        for qid in ("q0", "q1")
        for policy in policies
    ]
    items = [SimpleNamespace(qid=qid) for qid in ("q0", "q1")]
    from transformers import AutoTokenizer

    def run(output: Path, cohort: Rows) -> dict[str, Any]:
        # Isolate checkpoint IO and per-row token validation; keep the complete
        # report entry point, cohort guard, aggregation, and publication real.
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("RCC_FANOUT_BENCHMARK", FANOUTQA_NATURAL_DEV50.benchmark_key)
            patch.setattr(
                route_report,
                "load_prepared_panel",
                lambda *_args, **_kw: (items, {"artifact_sha256": "a" * 64}),
            )
            patch.setattr(AutoTokenizer, "from_pretrained", lambda *_args, **_kw: object())
            patch.setattr(
                route_report, "validate_and_rescore_result", lambda row, *_args, **_kw: row
            )
            return route_report.build_qwen_report(
                output,
                rows=cohort,
                source_commit="0" * 40,
                family=family,
            )

    _check_cohorts(root, rows, run, per_seed=False)


def test_report_scoring_versions(tmp_path: Path) -> None:
    """Reject mixed cohorts before aggregation and label homogeneous reports."""
    _check_split(tmp_path / "split-scoring")
    for family in (QWEN_FAMILY, NEMOTRON_FAMILY):
        _check_route(tmp_path / family.lane, family)


_CLOSER_IDS = QWEN_FAMILY.think_close_token_ids


def test_qwen_receiver_result_and_raw_token_rescore_contract():
    rows = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    semantic_arm = "latent_query_support_r2"
    payload = QwenFlatPayload(
        rows=rows,
        semantic_arm=semantic_arm,
        latent_plan_sha256=latent_plan_sha256(semantic_arm, family=QWEN_FAMILY),
        keeps=((0, 1), (0, 1), (0, 1)),
        rows_by_worker=(2, 2, 2),
        selected_indices_sha256=selected_indices_sha256(((0, 1), (0, 1), (0, 1))),
        tensor_sha256=tensor_content_sha256(rows),
        family=QWEN_FAMILY,
        profile=FANOUTQA_NATURAL_DEV50,
        layout=FANOUTQA_NATURAL_DEV50.payload_layout,
    )
    weight = torch.arange(256 * 4, dtype=torch.float32).reshape(256, 4)
    prompt = prepare_receiver_prompt(weight, (65, 66), payload=payload)
    assert prompt.payload_rows == prompt.handoff_rows == 6
    # The chain ships the last hop's block alone; the three preceding keeps
    # ride along as the provenance of the global re-cut that produced it, so the
    # rows here are the fourth keep's four rows and nothing else.
    chain_rows = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    chain_keeps = ((0,), (0,), (0,), (0, 1, 2, 3))
    chain_arm = f"latent_query_support_rerank_r{CHAIN_FIXTURE.ratios[0]}"
    chain_payload = QwenFlatPayload(
        rows=chain_rows,
        semantic_arm=chain_arm,
        latent_plan_sha256=chain_plan_sha256(chain_arm, family=QWEN_FAMILY, profile=CHAIN_FIXTURE),
        keeps=chain_keeps,
        rows_by_worker=(1, 1, 1, 4),
        selected_indices_sha256=selected_indices_sha256(chain_keeps),
        tensor_sha256=tensor_content_sha256(chain_rows),
        family=QWEN_FAMILY,
        profile=CHAIN_FIXTURE,
        layout=CHAIN_TERMINAL_LAYOUT,
    )
    chain_prompt = prepare_receiver_prompt(
        weight, (65, 66), payload=chain_payload, profile=CHAIN_FIXTURE
    )
    assert chain_prompt.payload_layout == CHAIN_FIXTURE.payload_layout
    # Window admission is the chain profile's on every family: the manager
    # rows, the handed-off rows, the answer ceiling and the closer fit inside
    # max_model_len, and one row past it is refused by item, arm and count.
    closing = len(_CLOSER_IDS) + ANSWER_CLOSING_TOKEN_BUDGET
    served = 2 + int(chain_rows.shape[0])
    narrow = replace(CHAIN_FIXTURE, max_model_len=served + CHAIN_FIXTURE.answer_ceiling + closing)
    assert (
        prepare_receiver_prompt(
            weight,
            (65, 66),
            payload=chain_payload,
            profile=narrow,
            family=QWEN_FAMILY,
            qid="window",
            semantic_arm=chain_arm,
        ).prompt_rows
        == served
    )
    with pytest.raises(RuntimeError, match=rf"window/{chain_arm}.*receiver context"):
        prepare_receiver_prompt(
            weight,
            (65, 66),
            payload=chain_payload,
            profile=replace(narrow, max_model_len=narrow.max_model_len - 1),
            family=QWEN_FAMILY,
            qid="window",
            semantic_arm=chain_arm,
        )
    with pytest.raises(ValueError, match="registered layout 'chain-terminal-v1'"):
        replace(chain_payload, layout=FANOUTQA_NATURAL_DEV50.payload_layout)
    with pytest.raises(ValueError, match="exactly 4 workers"):
        replace(chain_payload, keeps=chain_keeps[:3])
    qid = "b3b15d0277166b1d"
    requests = receiver_requests(
        qid, semantic_arm, prompt, family=QWEN_FAMILY, profile=FANOUTQA_NATURAL_DEV50
    )
    chain_requests = receiver_requests(
        CHAIN_FIXTURE.question_ids[0],
        chain_arm,
        chain_prompt,
        family=QWEN_FAMILY,
        profile=CHAIN_FIXTURE,
    )
    assert all(
        request.profile is request.decode.profile is CHAIN_FIXTURE for request in chain_requests
    )
    # A profile that registers another sample roster is refused by name.
    with pytest.raises(ValueError, match=CHAIN_FIXTURE.benchmark_key):
        receiver_requests(
            CHAIN_FIXTURE.question_ids[0],
            chain_arm,
            chain_prompt,
            family=QWEN_FAMILY,
            profile=replace(CHAIN_FIXTURE, sample_tags=("s0", "s1")),
        )
    with pytest.raises(ValueError, match="require the answer decode purpose"):
        replace(
            requests[0],
            decode=QwenDecodeSpec("report", QWEN_FAMILY, profile=FANOUTQA_NATURAL_DEV50),
            seed=1991879859,
        )
    # The request and its decode spec each name a benchmark; the seeds come
    # from one and the output ceiling from the other, so they must agree.
    with pytest.raises(ValueError, match="different benchmarks"):
        replace(
            chain_requests[0],
            decode=QwenDecodeSpec("answer", QWEN_FAMILY, profile=FANOUTQA_NATURAL_DEV50),
        )
    assert [len(batch) for batch in receiver_batches(requests)] == [1, 2]
    assert [request.seed for request in requests] == [1991879859, 1025933490, 32469775]
    assert all(
        request.sampling
        == {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 0.0,
            "max_tokens": 24_000,
            "stop_token_ids": [151645, 151643],
        }
        for request in requests
    )
    assert (
        len(
            receiver_requests(
                FANOUTQA_NATURAL_DEV50.question_ids[0],
                semantic_arm,
                prompt,
                family=QWEN_FAMILY,
                profile=FANOUTQA_NATURAL_DEV50,
            )
        )
        == 3
    )

    class _Tokenizer:
        @staticmethod
        def decode(token_ids, **_kwargs):
            # The reserved closer id renders as its delimiter, exactly as the
            # pinned tokenizer does, so an injected draw decodes the way the
            # visible-answer rule reads it.
            return "".join(
                "</think>" if token == _CLOSER_IDS[0] else chr(token) for token in token_ids
            )

    raw = "<think>x</think>answer"
    tokens = (*map(ord, raw), 151645, *map(ord, "ignored"))
    completions = tuple(
        SimpleNamespace(
            request_id=request.request_id,
            text=raw,
            token_ids=tokens,
            n_tokens=len(tokens),
            finish_reason="stop",
            num_cached_tokens=0,
            answer_injected=False,
        )
        for request in requests
    )
    answers = reconstruct_visible_answers(
        _Tokenizer(), requests, cast(Any, tuple(reversed(completions))), family=QWEN_FAMILY
    )
    assert [answer.visible_text for answer in answers] == ["answer"] * 3
    assert [answer.seed for answer in answers] == [1991879859, 1025933490, 32469775]
    assert all(answer.raw_text == raw for answer in answers)
    assert all(answer.token_ids[-7:] == tuple(map(ord, "ignored")) for answer in answers)
    item = ProbeItem(
        qid=qid,
        question="What is the answer?",
        shards=((1,), (2,), (3,)),
        question_obj=Question(
            qid=qid,
            question="What is the answer?",
            pages=((1, 1, "one"),),
            raw={"answer": ["answer"]},
        ),
    )
    latency = ItemLatency(
        producer_s=2.0,
        receiver_prepare_s=0.25,
        decode_s=1.0,
        decode_batched_s=0.5,
        receiver_ttft_s=0.4,
        generation_s=0.6,
        queued_offsets=(0.0, 1.0, 1.0),
        first_token_offsets=(0.4, 1.4, 1.4),
        finished_offsets=(1.0, 1.5, 1.5),
    )
    row = build_result_row(
        item,
        semantic_arm,
        answers,
        prompt,
        worker_prompt_tokens=(50_000, 50_000, 50_000),
        latency=latency,
        family=QWEN_FAMILY,
        profile=FANOUTQA_NATURAL_DEV50,
    )
    assert row["scoring_version"] == "fanoutqa-equivalence-v1"
    with pytest.raises(ValueError, match="unknown FanOutQA scoring version"):
        validate_and_rescore_result(
            {**row, "scoring_version": "unknown"},
            item,
            _Tokenizer(),
            family=QWEN_FAMILY,
            profile=FANOUTQA_NATURAL_DEV50,
        )
    assert row["sample_aggregation"].endswith("no_vote")
    assert row["loose"] == row["strict"] == 1.0
    # The scored keys are the profile's roster, not three names written out
    # here: a benchmark scoring different fields gets its own, in its order.
    assert [key for key in row if key in set(FANOUTQA_NATURAL_DEV50.score_fields)] == list(
        FANOUTQA_NATURAL_DEV50.score_fields
    )
    assert row["decode_tags"] == list(FANOUTQA_NATURAL_DEV50.sample_tags)
    # A benchmark whose scorer this builder does not implement is refused by
    # name, not by a missing key deep inside the mean over its fields.
    with pytest.raises(ValueError, match="registers unregistered-scorer-v1"):
        build_result_row(
            item,
            semantic_arm,
            answers,
            prompt,
            worker_prompt_tokens=(50_000, 50_000, 50_000),
            latency=latency,
            family=QWEN_FAMILY,
            profile=replace(FANOUTQA_NATURAL_DEV50, scorer="unregistered-scorer-v1"),
        )
    # The chain benchmark registers the official extraction, so the same
    # builder scores its row: one sample reads (B), one reads a bare C, and
    # one answers nothing at all.
    chain_qid = CHAIN_FIXTURE.question_ids[0]
    chain_item = ChainItem(
        qid=chain_qid,
        question="Which one?",
        choices=("first", "second", "third", "fourth"),
        chunks=((1, 2), (3, 4), (5, 6), (7, 8)),
        chunk_sha256=("0" * 64,) * 4,
        source_sha256="1" * 64,
        source_tokens=8,
        stratum=3,
        domain="Multi-Document QA",
        difficulty="easy",
        gold="B",
    )
    chain_answers = [
        replace(answer, tag=tag, seed=seed, visible_text=text, raw_text=f"<think>x</think>{text}")
        for answer, tag, seed, text in zip(
            answers,
            CHAIN_FIXTURE.sample_tags,
            CHAIN_FIXTURE.answer_seeds(chain_qid, "issue_only"),
            ("The correct answer is (B)", "The correct answer is C", "no answer"),
            strict=True,
        )
    ]
    chain_row = build_result_row(
        chain_item,
        "issue_only",
        chain_answers,
        chain_prompt,
        worker_prompt_tokens=chain_item.chunk_tokens,
        latency=latency,
        family=QWEN_FAMILY,
        profile=CHAIN_FIXTURE,
    )
    assert chain_row["correct"] == round(1 / 3, 4)
    assert chain_row["answered"] == round(2 / 3, 4)
    assert chain_row["extracted_choice_by_sample"] == ["B", "C", None]
    assert chain_row["gold"] == "B"
    assert (chain_row["stratum"], chain_row["domain"], chain_row["difficulty"]) == (
        3,
        "Multi-Document QA",
        "easy",
    )
    # A chain row banks one count per hop: the chunk's own tokens plus at most
    # the registered wrapper allowance. Three counts are not four hops.
    for counts in (chain_item.chunk_tokens[:3], (0, 0, 0, 0)):
        with pytest.raises(ValueError, match=CHAIN_FIXTURE.benchmark_key):
            build_result_row(
                chain_item,
                "issue_only",
                chain_answers,
                chain_prompt,
                worker_prompt_tokens=counts,
                latency=latency,
                family=QWEN_FAMILY,
                profile=CHAIN_FIXTURE,
            )
    # That band is the hop prompt's own geometry, so it serves every arm that
    # reads a hop prompt: the floor arm above and a latent arm here. Both
    # wrappers are the ones the profile's own prompt builder is priced on.
    hop_wrapper, rewrite_wrapper = chain_wrapper_allowances(CHAIN_FIXTURE)
    latent_arm = f"latent_query_support_rerank_r{CHAIN_FIXTURE.ratios[0]}"
    latent_row = build_result_row(
        chain_item,
        latent_arm,
        [
            replace(answer, seed=seed)
            for answer, seed in zip(
                chain_answers,
                CHAIN_FIXTURE.answer_seeds(chain_qid, latent_arm),
                strict=True,
            )
        ],
        chain_prompt,
        worker_prompt_tokens=tuple(count + hop_wrapper for count in chain_item.chunk_tokens),
        latency=latency,
        family=QWEN_FAMILY,
        profile=CHAIN_FIXTURE,
    )
    assert latent_row["worker_prompt_tokens"] == [
        count + hop_wrapper for count in chain_item.chunk_tokens
    ]
    # A text arm reads a rewrite prompt, not a hop prompt: it carries the
    # previous hop's notes up to the report ceiling and re-encodes the decoded
    # part instead of splicing the sealed ids.
    text_answers = [
        replace(answer, seed=seed)
        for answer, seed in zip(
            chain_answers,
            CHAIN_FIXTURE.answer_seeds(chain_qid, "text_primary"),
            strict=True,
        )
    ]
    parts = [count + 1 for count in chain_item.chunk_tokens]
    ceiling = [
        rewrite_prompt_ceiling(
            count, report_ceiling=CHAIN_FIXTURE.report_ceiling, wrapper=rewrite_wrapper
        )
        for count in chain_item.chunk_tokens
    ]

    def text_row(counts: Sequence[int], fields: Mapping[str, object] | None) -> dict[str, Any]:
        """Bank one text arm row under one prompt-token roster and one part roster."""
        return build_result_row(
            chain_item,
            "text_primary",
            text_answers,
            chain_prompt,
            worker_prompt_tokens=counts,
            latency=latency,
            family=QWEN_FAMILY,
            channel_fields=fields,
            profile=CHAIN_FIXTURE,
        )

    text_banked = text_row(ceiling, {"chunk_text_tokens": parts})
    assert text_banked["worker_prompt_tokens"] == ceiling
    assert (text_banked["correct"], text_banked["gold"]) == (round(1 / 3, 4), "B")
    # One token above the registered context, one below the part the seat
    # re-encoded, and a seat that banked no part at all: each refused by name.
    for counts, fields in (
        ([count + 1 for count in ceiling], {"chunk_text_tokens": parts}),
        ([count - 1 for count in parts], {"chunk_text_tokens": parts}),
        (ceiling, None),
    ):
        with pytest.raises(ValueError, match="text_primary"):
            text_row(counts, fields)
    # The producer wall sits inside the caller's wait, not beside it, so both
    # published clocks carry it; the harness spill read never enters either.
    assert set(ITEM_LATENCY_FIELDS) <= row.keys()
    assert row["ttft_s"] == 2.65
    assert row["tteoa_s"] == 3.25
    assert row["decode_s"] == 1.0
    assert row["decode_batched_s"] == 0.5
    assert row["queue_s"] is None
    assert row["decode_timing_sample"] == "s0"
    assert row["timing_scope"] == COMPUTE_TIMING_SCOPE
    assert row["decode_first_token_offset_s_by_sample"] == [0.4, 1.4, 1.4]
    with pytest.raises(ValueError, match="before s0 finished"):
        replace(latency, queued_offsets=(0.0, 0.5, 1.0))
    assert validate_and_rescore_result(
        row, item, _Tokenizer(), family=QWEN_FAMILY, profile=FANOUTQA_NATURAL_DEV50
    )["answer_texts"] == [
        "answer",
        "answer",
        "answer",
    ]
    tampered = {
        **row,
        "answers": ["tampered"] * 3,
        "answer": "tampered",
        "thinking_closed_by_sample": [False] * 3,
        "thinking_closed_rate": 0.0,
        "finish_reasons": ["length"] * 3,
        "finish_reason": "length",
        "length_finish_rate": 1.0,
        "generated_tokens_by_sample": [24_000] * 3,
        "generated_tokens": 24_000.0,
        "first_token_emitted_by_sample": [False] * 3,
    }
    with pytest.raises(RuntimeError, match="raw-token rescore differs"):
        validate_and_rescore_result(
            tampered, item, _Tokenizer(), family=QWEN_FAMILY, profile=FANOUTQA_NATURAL_DEV50
        )

    # The answer-side closer injection: head content, the appended closer, the
    # continuation, and the head's finish reason banked beside them. Here the
    # head ran to the 24,000-token ceiling thinking, so its reason is "length".
    head = (*map(ord, "<think>"), *([ord("x")] * (24_000 - 7)))
    injected_ids = (*head, *_CLOSER_IDS, *map(ord, "answer"), 151645)
    injected_answers = [
        replace(
            answer,
            raw_text=_Tokenizer.decode(injected_ids[:-1]),
            visible_text="answer",
            token_ids=injected_ids,
            visible_decode_token_ids=injected_ids[:-1],
            finish_reason="length",
            answer_injected=True,
        )
        for answer in answers
    ]
    injected_row = build_result_row(
        item,
        semantic_arm,
        injected_answers,
        prompt,
        worker_prompt_tokens=(50_000, 50_000, 50_000),
        latency=latency,
        family=QWEN_FAMILY,
        profile=FANOUTQA_NATURAL_DEV50,
    )
    assert injected_row["answer_injected_by_sample"] == [True] * 3
    assert injected_row["answer_head_finish_reasons"] == ["length"] * 3
    assert injected_row["answer_closer_budget"] == 4096
    rescored = validate_and_rescore_result(
        injected_row, item, _Tokenizer(), family=QWEN_FAMILY, profile=FANOUTQA_NATURAL_DEV50
    )
    # Closure is truthful after the merge, the head's reason survives, and the
    # censored answer scores.
    assert rescored["answer_texts"] == ["answer"] * 3
    assert rescored["thinking_closed_by_sample"] == [True] * 3
    assert rescored["finish_reasons"] == ["length"] * 3
    assert rescored["loose"] == 1.0
    short = (*map(ord, "<think>x"), *_CLOSER_IDS, *map(ord, "answer"), 151645)
    for patch, message in (
        # Without its flag the merged row is a natural draw past the ceiling.
        ({"answer_injected_by_sample": [False] * 3}, "longer than its registered decode"),
        ({"answer_head_finish_reasons": ["dropped"] * 3}, "head reason is unregistered"),
        ({"answer_injected_by_sample": [1, 1, 1]}, "injection evidence is malformed"),
        ({"answer_closer_budget": 10}, "closing budget is unregistered"),
        # A head banked as "length" filled the ceiling, so its merged row is at
        # least ceiling plus one closer id; a short row claiming it is refused.
        ({"generated_token_ids_by_sample": [list(short)] * 3}, "did not fill the ceiling"),
        # And the tail after the closer is at most one closing budget.
        (
            {
                "generated_token_ids_by_sample": [
                    [*head, *_CLOSER_IDS, *([ord("a")] * 4_097), 151645]
                ]
                * 3
            },
            "continuation exceeds its closing budget",
        ),
    ):
        with pytest.raises(RuntimeError, match=message):
            validate_and_rescore_result(
                {**injected_row, **patch},
                item,
                _Tokenizer(),
                family=QWEN_FAMILY,
                profile=FANOUTQA_NATURAL_DEV50,
            )

    completions[0].num_cached_tokens = 1
    with pytest.raises(RuntimeError, match="prefix caching contaminated Qwen answer"):
        reconstruct_visible_answers(
            _Tokenizer(), requests, cast(Any, completions), family=QWEN_FAMILY
        )
