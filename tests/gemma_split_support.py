"""Fixtures for the Gemma split fleet: engines, banks, producers, and receivers."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
from tests.gemma_text_support import _PREPARED_MANIFEST, _panel_item
from tests.gemma_text_support import _Engine as _TextReportEngine
from tests.gemma_text_support import _Tokenizer as _TextTokenizer

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.fanoutqa.gemma_data import RollFrame
from rcc.hardware.fleet import FleetPlacement
from rcc.models.gemma.capture_support import LATENT_ROLL_SCHEMA, DeferredAudit, latent_roll_filename
from rcc.models.gemma.capture_types import CaptureArtifact
from rcc.models.gemma.config import GemmaFleetConfig
from rcc.models.gemma.contract import (
    LATENT_REALIGN_ENABLED,
    LATENT_STEPS,
    MAX_NUM_SEQS,
    ROW_SCHEMA,
    WORKERS_PER_ITEM,
)
from rcc.models.gemma.embedding import publish_shared_embedding_once
from rcc.models.gemma.engine_contract import REGISTERED_TEMPLATE
from rcc.models.gemma.mechanism import layer_bank_filename
from rcc.models.gemma.receiver_core import GemmaReceiverCore
from rcc.models.gemma.roster import SHARED_RESULT_PANEL
from rcc.models.gemma.selection import load_or_bank_selection, payload_length
from rcc.run.banks import BankWriter
from rcc.run.contract import BankCallback, BankIdentity, WorkerContext
from rcc.run.fleet.admission import AdmissionGate
from rcc.run.fleet.latency import ItemLatency
from rcc.run.fleet.stream import ItemStreamTracker, StreamCompletion, StreamDecoder
from rcc.run.fleet.vllm import VllmEngineHandle
from rcc.run.gemma.split_adapter import GemmaSplitFleetAdapter, GemmaSplitPlacementAdapter
from rcc.run.gemma.split_build import build_producer, receiver_kwargs
from rcc.run.gemma.split_receiver import GemmaFusedReceiver, GemmaSplitReceiver
from rcc.run.gemma.split_report import gemma_split_report_adapter

_QID = "q1"
_FRAME_PREFIX = 3
_FRAME_SUFFIX = 2
# The rolled cargo validator takes the live model width, so the fixture may
# use a tiny hidden size instead of the 3840-wide production table.
_HIDDEN = 8
# The capture fields the receiver banks but cannot derive: the equivalence
# record, the architectural global-attention layer list, and the audition
# ledger.
_EMBEDDING_DIGESTS: dict[str, Any] = {
    "family": "gemma",
    "hidden_size": _HIDDEN,
    "max_abs_error": 0.0,
    "passed": True,
}
_GLOBAL_LAYERS = (5, 11, 17)
_REPLAYS: dict[int, int] = {}


def _rolled_tensor(worker: int, *, rows: int = LATENT_STEPS) -> torch.Tensor:
    values = torch.arange(rows * _HIDDEN, dtype=torch.float32) + float(worker)
    return values.reshape(rows, _HIDDEN).to(dtype=torch.bfloat16)


def _write_rolled(path: Path, worker: int, rolled: torch.Tensor, *, qid: str = _QID) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": LATENT_ROLL_SCHEMA,
            "qid": qid,
            "worker": worker,
            "latent_steps": LATENT_STEPS,
            "realign_enabled": LATENT_REALIGN_ENABLED,
            "frame_prefix_length": _FRAME_PREFIX,
            "frame_suffix_length": _FRAME_SUFFIX,
            "rolled": rolled,
        },
        path,
    )


_ARM = "latent_query_support_r2"
_FLOOR_ARM = "issue_only"
_ATTEMPT_ID = "gemma-split-fleet-fixture"
_SOURCE_COMMIT = "0" * 40
_ENGINE_IDENTITY_SHA = "d" * 64
_SPLIT_PLACEMENT = FleetPlacement(workers=2, producers=1, receivers=1)
_QIDS = FANOUTQA_NATURAL_DEV50.question_ids[:2]
_MEMORY_TOKENS = 6_000
_FRAME = RollFrame((11, 12, 13), (14, 15))
_PROMPT_TOKENS = len(_FRAME.prefix_ids) + _MEMORY_TOKENS + len(_FRAME.suffix_ids)
_PAYLOAD_TOKENS = payload_length(_PROMPT_TOKENS)
_TABLE_ROWS = 45_600
_TABLE_SHAPE = (_TABLE_ROWS, _HIDDEN)
_SENTINEL_ID = 7_777
_CHANNEL_OPEN = 98
_CHANNEL_CLOSE = 101
_DELIMITER_ID = 108
_SPECIAL_IDS = {
    "<|channel>": _CHANNEL_OPEN,
    "<channel|>": _CHANNEL_CLOSE,
    "\n\n": _DELIMITER_ID,
    "@@CONTENT@@": _SENTINEL_ID,
    "<audio|>": 258_883,
    "<image|>": 258_882,
}


def embedding_table(shape: tuple[int, int] = _TABLE_SHAPE) -> torch.Tensor:
    """A deterministic bfloat16 embedding table of one shape."""
    rows, hidden = shape
    values = torch.arange(rows * hidden, dtype=torch.float32) / float(rows)
    return values.reshape(shape).to(dtype=torch.bfloat16)


class _SplitTokenizer:
    bos_token_id = 2
    eos_token_id = 1

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        if text in _SPECIAL_IDS:
            return {"input_ids": [_SPECIAL_IDS[text]]}
        return {"input_ids": [10 + index for index, _word in enumerate(text.split())] or [10]}

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> list[int]:
        template = REGISTERED_TEMPLATE
        thinking = bool(kwargs.get("enable_thinking"))
        prefix = template.on_prefix if thinking else template.off_prefix
        suffix = template.on_suffix if thinking else template.off_suffix
        content = self(messages[0]["content"])["input_ids"]
        return [*prefix, *content, *suffix]

    @staticmethod
    def decode(ids: Any) -> str:
        values = [int(value) for value in ids]
        names = {value: name for name, value in _SPECIAL_IDS.items()}
        return " ".join(names.get(value, str(value)) for value in values)


class _Decoder:
    def __init__(self) -> None:
        self.live: list[tuple[float, Any]] = []
        self.now = time.monotonic()
        self.aborted = 0

    def clock(self) -> float:
        return self.now

    def submit(self, request: Any) -> None:
        self.now = max(self.now, time.monotonic())
        self.live.append((self.now, request))

    def in_flight(self) -> int:
        return len(self.live)

    def abort_all(self) -> list[str]:
        self.aborted += 1
        return []

    def poll(self) -> list[StreamCompletion]:
        if not self.live:
            return []
        submitted, request = self.live.pop(0)
        self.now = max(self.now, submitted) + 1.0
        return [
            StreamCompletion(
                request_id=request.request_id,
                qid=request.qid,
                tag=request.tag,
                text="answer",
                n_tokens=3,
                finish_reason="stop",
                submitted_at=submitted,
                first_token_at=submitted + 0.5,
                finished_at=self.now,
                num_cached_tokens=0,
                token_ids=(_CHANNEL_OPEN, 7, _CHANNEL_CLOSE),
            )
        ]


class _Handle:
    def sampling(self, sampling: Any, seed: int) -> Any:
        return SimpleNamespace(sampling=sampling, seed=seed)


class _SplitEngine:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class _SplitReceiver(GemmaSplitReceiver):
    def warm(self) -> None:
        self.engine_identity_sha256 = _ENGINE_IDENTITY_SHA
        self.decoder = cast(StreamDecoder, _Decoder())
        self.handle = cast(VllmEngineHandle, _Handle())
        self.kv_pool_tokens = 1_000_000
        self.claim_limit = MAX_NUM_SEQS
        self.tracker = ItemStreamTracker(
            self.decoder,
            AdmissionGate(
                self.kv_pool_tokens,
                answer_ceiling=24_000,
                max_in_flight=MAX_NUM_SEQS,
            ),
        )
        self._bind_decoder_clock()
        self.bank({"kind": "phase", "phase": "gemma_split_receiver_warm", "arm": self.arm})


def _warm_real_split_receiver(monkeypatch: Any) -> GemmaSplitReceiver:
    def warm(receiver: GemmaReceiverCore) -> None:
        receiver.engine_identity_sha256 = _ENGINE_IDENTITY_SHA
        receiver.model_load_s = receiver.warmup_s = 0.0
        receiver.kv_pool_tokens = 337_808
        receiver.decoder = cast(StreamDecoder, _Decoder())
        receiver.tracker = cast(Any, object())

    monkeypatch.setattr(GemmaReceiverCore, "warm", warm)
    receiver = GemmaSplitReceiver.__new__(GemmaSplitReceiver)
    receiver.arm, receiver.live = _ARM, {}
    receiver.bank = cast(Any, [].append)
    receiver.warm()
    return receiver


class _FusedReceiver(GemmaFusedReceiver, _SplitReceiver):
    pass


def _prepared_text(qid: str) -> Any:
    return SimpleNamespace(
        qid=qid,
        semantic_arm="text_primary",
        prompt=torch.zeros(8, _HIDDEN, dtype=torch.bfloat16),
        receiver_prepare_s=0.5,
    )


_TEXT_POLICIES = {
    "text_primary": "gemma4_12b_text",
    "text_medium": "gemma4_12b_calls_gemma4_e4b_text",
    "text_small": "gemma4_12b_calls_gemma4_e2b_text",
}


def _text_rows(kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    latency = ItemLatency(
        producer_s=1.0,
        receiver_prepare_s=0.5,
        decode_s=1.0,
        decode_batched_s=1.0,
        receiver_ttft_s=0.5,
        generation_s=0.5,
        queued_offsets=(0.0, 1.0, 1.0),
        first_token_offsets=(0.5, 1.5, 1.5),
        finished_offsets=(1.0, 2.0, 2.0),
    ).result_fields()
    arm = str(kwargs["arm"])
    qid = str(kwargs["item"]["qid"])
    return [
        {
            "kind": "result",
            "schema": ROW_SCHEMA,
            "qid": qid,
            "arm": arm,
            "policy": _TEXT_POLICIES[arm],
            "panel": "dev30",
            "cell": f"{arm}|s{index}",
            "seed_index": index,
            "seed": kwargs["seeds"][index],
            "sample_seeds": list(kwargs["seeds"]),
            **latency,
        }
        for index in range(3)
    ]


def _item(qid: str) -> dict[str, Any]:
    memory = tuple(torch.arange(_MEMORY_TOKENS, dtype=torch.long) + 10 for _ in range(3))
    return {
        "qid": qid,
        "question": SimpleNamespace(question="question"),
        "page_texts": ["page"],
        "memory_ids": memory,
        "prompt_ids": tuple(_FRAME.prompt(ids) for ids in memory),
        "frame": _FRAME,
        "request_ids": [30, 31],
        "panel": "dev30",
        "global_item_index": 1,
        "construction": {
            "n_locatable_in_full_source": 2,
            "n_survives_construction": 2,
            "n_removed_by_construction": 0,
        },
        "construction_complete": True,
        "construction_incomplete_reason": "",
        "dead_pages": [],
    }


def _scores(name: str) -> tuple[dict[str, torch.Tensor], ...]:
    return tuple(
        {name: torch.arange(_PAYLOAD_TOKENS, dtype=torch.float32) * (worker + 1)}
        for worker in range(WORKERS_PER_ITEM)
    )


@dataclass(frozen=True)
class _FakeAudit:
    bank_path: Path
    rolled_path: Path
    shard_log: Any
    qid: str
    worker: int
    row: dict[str, float]

    def run(self) -> None:
        started = time.perf_counter()
        self.bank_path.parent.mkdir(parents=True, exist_ok=True)
        self.bank_path.write_bytes(b"fixture layer bank")
        self.shard_log.bank_capture(
            {
                **self.row,
                "audit_s": round(time.perf_counter() - started, 4),
                "layer_bank_file": self.bank_path.name,
                "layer_bank_schema": "fixture",
                "latent_roll_file": self.rolled_path.name,
                "latent_roll_schema": LATENT_ROLL_SCHEMA,
            },
            qid=self.qid,
            worker=self.worker,
        )


def _fake_capture(
    items: tuple[dict[str, Any], ...],
    *,
    bridge: Any,
    artifact_root: Path,
    shard_log: Any,
    worker_index: int,
    runtime_fingerprint: str,
    publication_id: str,
    shared_weight: Path,
    selection_arms: tuple[str, ...],
    flat_interleave_arms: tuple[str, ...] = (),
) -> tuple[tuple[CaptureArtifact, ...], dict[str, str], tuple[_FakeAudit, ...]]:
    del bridge, worker_index, runtime_fingerprint, publication_id, shared_weight
    produced: list[CaptureArtifact] = []
    audits: list[_FakeAudit] = []
    for item in items:
        started = time.perf_counter()
        qid = str(item["qid"])
        cargo = tuple(_rolled_tensor(worker) for worker in range(WORKERS_PER_ITEM))
        rolled_paths: list[Path] = []
        for worker, tensor in enumerate(cargo):
            path = artifact_root / "rolled" / latent_roll_filename(qid, worker)
            _write_rolled(path, worker, tensor, qid=qid)
            rolled_paths.append(path)
        selection = load_or_bank_selection(
            item,
            _scores("support"),
            artifact_root / "selections" / f"{qid}.json",
            seat_keys_by_ratio={},
            ratio_cuts={},
            flat_interleave_arms=tuple(
                arm for arm in flat_interleave_arms if arm in selection_arms
            ),
            arms=selection_arms,
        )
        selection_s = sum(selection.selection_s_by_arm.values())
        capture_s = max(time.perf_counter() - started - selection_s, 0.0)
        worker_capture_s = capture_s / WORKERS_PER_ITEM
        clock_s = round(worker_capture_s / 4, 4)
        for worker, rolled_path in enumerate(rolled_paths):
            audits.append(
                _FakeAudit(
                    bank_path=artifact_root / "layer_banks" / layer_bank_filename(qid, worker),
                    rolled_path=rolled_path,
                    shard_log=shard_log,
                    qid=qid,
                    worker=worker,
                    row={
                        "capture_s": round(worker_capture_s, 4),
                        "extract_s": clock_s,
                        "roll_s": clock_s,
                        "capture_query_s": clock_s,
                        "parity_s": clock_s,
                    },
                )
            )
        produced.append(
            CaptureArtifact(
                qid=qid,
                keeps_by_arm=selection.keeps_by_arm,
                layouts_by_arm=selection.layouts_by_arm,
                rolled_by_worker=cargo,
                selection_s_by_arm=selection.selection_s_by_arm,
                audition_s_by_ratio={},
                audition_replays_by_ratio=dict(_REPLAYS),
                reports=(),
                report_failed=False,
                report_failure=None,
                capture_s=round(capture_s, 4),
                capture_s_by_worker=(round(worker_capture_s, 4),) * WORKERS_PER_ITEM,
                report_generation_s=0.0,
                embedding_digests=dict(_EMBEDDING_DIGESTS),
                global_layers=_GLOBAL_LAYERS,
                reloaded_captures=0,
                reloaded_selections=0,
            )
        )
    return tuple(produced), {"file_sha256": "e" * 64}, tuple(audits)


def _failing_capture(*args: Any, _capture: Any = _fake_capture, **kwargs: Any) -> Any:
    artifacts, embedding, _audits = _capture(*args, **kwargs)
    item = args[0][0]
    artifact_root = Path(kwargs["artifact_root"])
    audit = DeferredAudit(
        item=item,
        worker=0,
        bank=cast(Any, SimpleNamespace(schema="fixture")),
        scores={},
        rolled=artifacts[0].rolled_by_worker[0],
        bank_path=artifact_root / "layer_banks" / "deferred.pt",
        rolled_path=artifact_root / "rolled" / latent_roll_filename(str(item["qid"]), 0),
        frame=item["frame"],
        shard_log=kwargs["shard_log"],
        capture_s=0.0,
        include_qsnap=False,
        extract_s=0.0,
        roll_s=0.0,
        capture_query_s=0.0,
        parity_s=0.0,
    )
    return artifacts, embedding, (audit,)


def _drop_first_capture_row(path: Path) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows.pop(
        next(index for index, row in enumerate(rows) if row.get("phase") == "gemma_split_capture")
    )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class _SenderEngine(_TextReportEngine):
    def __init__(self) -> None:
        super().__init__()
        self.warmups = 0
        self.closed = 0

    def close(self) -> None:
        self.closed += 1

    def decode_token_ids_full(self, prompts: Any, sampling: Any, *, seeds: Any) -> list[Any]:
        if len(prompts) == 1:
            self.warmups += 1
            return []
        return super().decode_token_ids_full(prompts, sampling, seeds=seeds)


def adapter(root: Path) -> GemmaSplitFleetAdapter:
    return GemmaSplitFleetAdapter(
        GemmaFleetConfig(
            results_base=root,
            source_cache=root / "source",
            item_offset=0,
            item_count=len(_QIDS),
            node_gpus=1,
            worker_index=0,
            attempt_id=_ATTEMPT_ID,
            source_commit=_SOURCE_COMMIT,
            unified_plan_fingerprint="9" * 64,
        )
    )


class _FixturePlacements(GemmaSplitPlacementAdapter):
    def __init__(self, arms: tuple[str, ...], placements: dict[str, FleetPlacement]) -> None:
        self._arms = arms
        self._placements = placements

    @property
    def arm_names(self) -> tuple[str, ...]:
        return self._arms

    def resolve(
        self,
        root: Path,
        *,
        panel: str,
        source_commit: str,
    ) -> Mapping[str, FleetPlacement]:
        del root, panel, source_commit
        return dict(self._placements)


class _FixtureAdapter(GemmaSplitFleetAdapter):
    def __init__(self, root: Path, placements: dict[str, FleetPlacement]) -> None:
        super().__init__(adapter(root).config)
        self.data = _FixtureData()
        self.placements = _FixturePlacements(tuple(placements), dict(placements))
        self.reports = gemma_split_report_adapter(
            self.placements, self.completion_key, profile=self._profile
        )

    def build_worker(self, context: WorkerContext, bank: BankCallback) -> Any:
        panel = dict(zip(context.qids, context.items, strict=True))
        if context.placement.role(context.worker_index) == "producer":
            return _producer(
                context.root,
                bank,
                arm=context.arm,
                placement=context.placement,
                items=panel,
            )
        return _receiver(
            context.root,
            bank,
            arm=context.arm,
            placement=context.placement,
            items=panel,
            publish=False,
        )


class _FixtureData:
    @property
    def qids(self) -> tuple[str, ...]:
        return _QIDS

    def load_prepared(
        self,
        root: Path,
        *,
        panel: str,
        source_commit: str,
    ) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
        del root, panel, source_commit
        return tuple(_item(qid) for qid in _QIDS), {"artifact_sha256": "a" * 64}

    def item(self, qid: str) -> dict[str, Any]:
        return _item(qid)


def fixture_adapter(
    root: Path,
    placements: dict[str, FleetPlacement],
) -> GemmaSplitFleetAdapter:
    return _FixtureAdapter(root, placements)


def identity(root: Path, arm: str, placement: FleetPlacement) -> BankIdentity:
    return adapter(root).bank_identity(
        root,
        panel=SHARED_RESULT_PANEL,
        arm=arm,
        source_commit=_SOURCE_COMMIT,
        prepared_manifest={"artifact_sha256": "a" * 64},
        placement=placement,
    )


def _context(
    root: Path,
    arm: str,
    *,
    placement: FleetPlacement,
    worker_index: int,
    items: dict[str, dict[str, Any]],
) -> WorkerContext:
    return WorkerContext(
        root=root,
        panel=SHARED_RESULT_PANEL,
        arm=arm,
        worker_index=worker_index,
        attempt_id=_ATTEMPT_ID,
        source_commit=_SOURCE_COMMIT,
        qids=tuple(items),
        items=tuple(items.values()),
        prepared_manifest=_PREPARED_MANIFEST,
        placement=placement,
        identity=identity(root, arm, placement),
    )


def publish_table(root: Path, arm: str, placement: FleetPlacement) -> dict[str, str]:
    return publish_shared_embedding_once(
        root,
        embedding_table(),
        runtime_fingerprint=str(identity(root, arm, placement).fields["runtime_fingerprint"]),
        publication_id=_ATTEMPT_ID,
        expected_shape=_TABLE_SHAPE,
    )


def _publish(run_root: Path, weight: torch.Tensor, **kwargs: Any) -> dict[str, str]:
    return publish_shared_embedding_once(run_root, weight, expected_shape=_TABLE_SHAPE, **kwargs)


def _producer(
    root: Path,
    bank: Any,
    *,
    arm: str = _ARM,
    placement: FleetPlacement = _SPLIT_PLACEMENT,
    items: dict[str, dict[str, Any]] | None = None,
) -> Any:
    panel = {qid: _item(qid) for qid in _QIDS} if items is None else items
    return build_producer(
        _context(root, arm, placement=placement, worker_index=0, items=panel),
        bank,
        panel,
        open_engine=lambda role: _SplitEngine(),
        bridge_factory=lambda engine: SimpleNamespace(
            release_view=lambda: None,
            model=SimpleNamespace(
                get_input_embeddings=lambda: SimpleNamespace(weight=embedding_table())
            ),
        ),
        capture=cast(Any, _fake_capture),
        verify=lambda engine, contract: {"engine_role": contract.role},
        publish=_publish,
    )


def _text_producer(
    root: Path,
    bank: Any,
    *,
    arm: str,
    placement: FleetPlacement = _SPLIT_PLACEMENT,
) -> tuple[Any, dict[str, Any]]:
    tokenizer = _TextTokenizer()
    # The producer is handed the raw panel item, exactly as the fleet hands it
    # over, so its own prompt-artifact binding is what the test exercises.
    item = _panel_item(tokenizer)
    items = {str(item["qid"]): item}
    producer = build_producer(
        _context(root, arm, placement=placement, worker_index=0, items=items),
        bank,
        items,
        engine_builder=lambda _config: _SenderEngine(),
        tokenizer_loader=lambda **_kwargs: tokenizer,
    )
    return producer, item


def _receiver(
    root: Path,
    bank: Any,
    *,
    arm: str,
    placement: FleetPlacement,
    items: dict[str, dict[str, Any]] | None = None,
    fused: bool = False,
    reports: Any = None,
    publish: bool = True,
) -> Any:
    panel = {qid: _item(qid) for qid in _QIDS} if items is None else items
    if publish:
        publish_table(root, arm, placement)
    kwargs = receiver_kwargs(
        _context(root, arm, placement=placement, worker_index=placement.producers, items=panel),
        bank,
        panel,
        engine=_SplitEngine(),
        tokenizer=_SplitTokenizer(),
        model_load_s=0.0,
    )
    if fused:
        return _FusedReceiver(reports=reports, **kwargs)
    return _SplitReceiver(**kwargs)


class _Bank:
    def __init__(self, root: Path, *, arm: str, placement: FleetPlacement) -> None:
        self.identity = identity(root, arm, placement)
        self.completion_key = adapter(root).completion_key
        self.path = root / "arms" / arm / "workers" / "gpu0" / "raw.jsonl"
        self.writer = BankWriter(
            self.path,
            fixed_fields=self.identity.fields,
            attempt_id=_ATTEMPT_ID,
        )
        self.lock = threading.Lock()
        self.rows: list[dict[str, Any]] = []

    def append(self, row: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            durable = self.writer.append(row)
            self.rows.append(durable)
        return durable

    def results(self) -> list[dict[str, Any]]:
        with self.lock:
            return [row for row in self.rows if row.get("kind") == "result"]

    def completed(self) -> set[str]:
        return {str(row["qid"]) for row in self.results() if self.completion_key(row) is not None}

    def stages(self) -> list[dict[str, Any]]:
        with self.lock:
            return [row for row in self.rows if row.get("kind") == "stage"]
