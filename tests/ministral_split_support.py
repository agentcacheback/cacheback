"""Offline fixtures shared by the Ministral split-fleet tests.

The engines and the Tekken codec are fakes with deterministic completions, so
the selection, the payload, the manifest and the result rows stay shipped code.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import torch

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.fanoutqa.ministral_construction import (
    make_ministral_construction_manifest,
)
from rcc.benchmarks.fanoutqa.panel import Question
from rcc.hardware.fleet import FleetPlacement
from rcc.latent.rollout import Realign
from rcc.models.ministral import MINISTRAL
from rcc.models.ministral.capture import LATENT_STEPS, MinistralWorkerCapture
from rcc.models.ministral.engine_roles import SPLIT_RECEIVER_ENGINE_IDENTITY
from rcc.models.ministral.payload import token_embedding_rows
from rcc.models.ministral.receiver_core import MinistralReceiverCore
from rcc.models.ministral.text import MinistralDecodeSpec, MinistralReportBundle
from rcc.models.ministral.text_codec import (
    MINISTRAL_TEXT_ARMS,
    MINISTRAL_THINK_CLOSE_ID,
    MINISTRAL_THINK_OPEN_ID,
    MinistralTextCodec,
    stop_trimmed,
)
from rcc.run.ministral.config import MinistralFleetConfig
from rcc.run.ministral.split_adapter import (
    MinistralSplitFleetAdapter,
    MinistralSplitPlacementAdapter,
)
from rcc.run.ministral.split_build import handoff_root
from rcc.run.ministral.split_producer import MinistralLatentProducer, MinistralTextProducer
from rcc.run.ministral.split_receiver import (
    TIMED_SAMPLE_TAG,
    MinistralFusedReceiver,
    MinistralSplitReceiver,
)
from rcc.run.ministral.split_report import ministral_split_report_adapter
from rcc.transforms.select.query_support.capture_bank import LayerBank

HIDDEN = 8
VOCAB = 512
SUPPORT_ARM = "latent_query_support_r128"
#: The one nonresident text arm these fixtures drive: a second checkpoint on a
#: producer GPU, so its reports really do cross the wire.
TEXT_ARM = "text_medium"
QIDS = FANOUTQA_NATURAL_DEV50.question_ids[:2]
PLAN_FINGERPRINT = "9" * 64


#: The manifest-roster fingerprint of this fixture's three text arms over QIDS.
CONSTRUCTION_ROSTER_FINGERPRINT = "8223e49ca4abe3867e56ac39d41a52ac043ac92c24fa96fd88dd2d241455075d"
RUNTIME_FINGERPRINT = "7" * 64
#: One closed thought channel plus one substantive answer token, the exact
#: shape the shipped reader admits.
ANSWER_TOKENS = (MINISTRAL_THINK_OPEN_ID, 2_000, MINISTRAL_THINK_CLOSE_ID, 3_000, 2)
#: One head that ended its turn inside an unclosed block, continued once with
#: the canonical closer injected: head content, closer, continuation. The
#: head's own stop id is gone.
INJECTED_TOKENS = (MINISTRAL_THINK_OPEN_ID, 2_000, 3_000, MINISTRAL_THINK_CLOSE_ID, 3_001, 2)
#: The closer appended after the end of turn. Every Ministral reader cuts at
#: the stop id, so the closer and the continuation are invisible and such a
#: draw banks no report.
PRE_FIX_TOKENS = (MINISTRAL_THINK_OPEN_ID, 2_000, 3_000, 2, MINISTRAL_THINK_CLOSE_ID, 3_001)
_EMPTY_CONSTRUCTION_AUDIT = {
    "n_gold_leaves": 0,
    "n_question_given": 0,
    "n_off_question_leaves": 0,
    "n_locatable_in_full_source": 0,
    "n_survives_construction": 0,
    "n_removed_by_construction": 0,
    "n_absent_from_full_source": 0,
    "leaves": [],
}


def construction_manifests(qid: str) -> dict[str, Any]:
    """Return one valid prepared-source manifest for every sender lane."""
    return {
        arm: make_ministral_construction_manifest(
            qid=qid,
            semantic_arm=arm,
            retained_sources=("", "", ""),
            audit=_EMPTY_CONSTRUCTION_AUDIT,
        )
        for arm in MINISTRAL_TEXT_ARMS
    }


def embedding_weight() -> torch.Tensor:
    """The fixture input-embedding table both halves of the split read."""
    return (
        torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(VOCAB, HIDDEN).to(torch.bfloat16)
    )


class _Inner:
    """The Tekken inner tokenizer seam: one reversible token per word.

    The production locator finds the question by decoding the signed turn, so
    the fixture must round-trip words; ids stay under the fixture vocab.
    """

    _words: ClassVar[list[str]] = []

    @classmethod
    def encode(cls, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        del bos, eos
        tokens: list[int] = []
        for word in text.split():
            if word not in cls._words:
                cls._words.append(word)
            tokens.append(300 + cls._words.index(word))
        return tokens or [299]

    @classmethod
    def decode(cls, ids: Any) -> str:
        parts: list[str] = []
        for value in ids:
            token = int(value)
            if 300 <= token < 300 + len(cls._words):
                parts.append(cls._words[token - 300])
            else:
                parts.append(str(token))
        return " ".join(parts)


class _Tokenizer:
    """The fixture tokenizer: decode is join, and it owns the inner seam."""

    instruct_tokenizer = SimpleNamespace(tokenizer=_Inner())

    @staticmethod
    def decode(ids: Any) -> str:
        return " ".join(str(int(value)) for value in ids)


def _production_reader() -> MinistralTextCodec:
    """Return the shipped reader bound to the fixture token seam."""
    reader = object.__new__(MinistralTextCodec)
    object.__setattr__(reader, "tokenizer", _Tokenizer())
    return reader


_READER = _production_reader()


class _Codec:
    """A verified-looking stand-in for one registered Tekken sender codec."""

    def __init__(self, semantic_arm: str) -> None:
        """Bind the exact registered checkpoint pin for one sender lane."""
        arm = next(row for row in MINISTRAL.physical_arms if row.semantic_arm == semantic_arm)
        self.semantic_arm = semantic_arm
        self.checkpoint = str(arm.sender_checkpoint)
        self.revision = str(arm.sender_revision)
        self.tokenizer = _Tokenizer()

    def require_verified(self) -> None:
        """The fixture codec is verified by construction."""

    @staticmethod
    def encode_warmup_prompt() -> tuple[int, ...]:
        """Return a short discarded warmup turn."""
        return (40, 41, 42)

    def encode_worker_prompt(self, question: str, evidence_ids: Sequence[int]) -> tuple[int, ...]:
        """Frame one worker turn around verbatim evidence, as the codec does."""
        evidence = str(self.tokenizer.decode([int(token) for token in evidence_ids]))
        return tuple(_Inner.encode(f"Private evidence: {evidence} Question: {question}"))

    def encode_manager_prompt(
        self,
        *,
        qid: str,
        semantic_arm: str,
        source_identity: str,
        question: str,
        reports: Any = (),
        report_failed: bool = False,
        profile: Any = None,
    ) -> dict[str, object]:
        """Sign one coordinator turn that literally contains the question rows."""
        del source_identity, profile
        body = tuple(_Inner.encode(question))
        tokens = [40, 41, *body, 42, *(200 + len(tuple(reports)),)]
        return {
            "prompt_token_ids": tokens,
            "prompt_sha256": "a" * 64,
            "fingerprint": "b" * 64,
            "qid": qid,
            "semantic_arm": semantic_arm,
        }

    @staticmethod
    def thinking_is_unclosed(raw_tokens: Any) -> bool:
        """The shipped trigger, over the fixture's token seam."""
        return _READER.thinking_is_unclosed(raw_tokens)

    @staticmethod
    def visible_report(raw_tokens: Any) -> tuple[tuple[int, ...], str]:
        """The shipped report reader, over the fixture's token seam."""
        return _READER.visible_report(raw_tokens)

    @staticmethod
    def visible_answer(
        raw_tokens: Any, *, ended: bool = False
    ) -> tuple[tuple[int, ...], str, bool]:
        """The shipped answer reader, over the fixture's token seam."""
        return _READER.visible_answer(raw_tokens, ended=ended)


def codec(semantic_arm: str = "text_primary") -> MinistralTextCodec:
    """Return one fixture codec under the shipped codec type."""
    return cast(MinistralTextCodec, _Codec(semantic_arm))


class _WordTokenizer(_Tokenizer):
    """A token seam whose decode inverts its own encode, word for word."""

    @staticmethod
    def decode(ids: Any) -> str:
        """Return the words the ids stand for, so a kept row reads as text."""
        return _Inner.decode(ids)


def word_codec(semantic_arm: str = "text_primary") -> MinistralTextCodec:
    """Return one fixture codec that gives the words back on decode.

    The handoff-recall reader scores decoded prompt rows, so that lane needs a
    seam that round-trips text rather than one that names ids.
    """
    built = _Codec(semantic_arm)
    built.tokenizer = cast(Any, _WordTokenizer())
    return cast(MinistralTextCodec, built)


class _Metrics:
    """The vLLM per-request metrics object the latency vocabulary prices."""

    def __init__(self, stamp: float) -> None:
        """Stamp one sample instantly on the fixture's wall clock.

        The stage ledger compares these against real wall time, and the phase
        machine, not an offset baked in here, keeps s1/s2 strictly after s0.
        """
        self.queued_ts = stamp
        self.scheduled_ts = stamp
        self.first_token_ts = stamp
        self.last_token_ts = stamp


class _Output:
    """One finished vLLM-shaped output for one admitted request."""

    def __init__(self, request_id: str, stamp: float) -> None:
        """Return the completed answer rows under the engine's own shape."""
        self.request_id = request_id
        self.finished = True
        self.num_cached_tokens = None
        self.metrics = _Metrics(stamp)
        self.outputs = (
            SimpleNamespace(
                token_ids=list(ANSWER_TOKENS),
                text="not trusted",
                finish_reason="stop",
            ),
        )


class _Engine:
    """A stepping embedding-decode engine that never opens a model.

    It is its own stream handle: the shipped receiver drives ``add_request``
    and ``step`` through the shared decoder, so the fixture answers that seam.
    """

    def __init__(self) -> None:
        """Start the fixture clock and the closed-engine counter."""
        # The fleet stage ledger refuses a decode that finished before the
        # panel was submitted, so the fixture clock is wall time like the
        # engine clock it stands in for.
        self.now = time.time()
        self.closed = 0
        self.steps = 0
        self.max_live = 0
        self._live: list[str] = []

    @property
    def stream_handle(self) -> _Engine:
        """Return the add-request/step seam; the fixture is its own handle."""
        return self

    @staticmethod
    def stream_sampling(request: Any) -> Any:
        """Stand in for the validated vLLM sampling params of one request."""
        return SimpleNamespace(seed=request.seed)

    @staticmethod
    def embedding_prompt(rows: torch.Tensor, index: int = 0) -> dict[str, torch.Tensor]:
        """Apply the shipped prompt geometry and finiteness checks."""
        if rows.ndim != 2 or int(rows.shape[0]) < 1 or int(rows.shape[1]) != HIDDEN:
            raise ValueError(f"Ministral embedding prompt {index} has invalid geometry")
        if not bool(torch.isfinite(rows).all()):
            raise ValueError(f"Ministral embedding prompt {index} contains non-finite rows")
        return {"prompt_embeds": rows.to(dtype=torch.bfloat16, device="cpu")}

    def add_request(self, request_id: str, prompt: Any, sampling_params: Any) -> None:
        """Admit one embedding-row request exactly as the shipped route does."""
        del sampling_params
        assert "prompt_embeds" in prompt
        assert request_id not in self._live
        self._live.append(request_id)
        self.max_live = max(self.max_live, len(self._live))

    def step(self) -> tuple[_Output, ...]:
        """Finish every admitted request and advance the fixture clock."""
        self.steps += 1
        self.now = max(self.now + 1e-4, time.time())
        finished = tuple(_Output(request_id, self.now) for request_id in self._live)
        self._live.clear()
        return finished

    def has_unfinished_requests(self) -> bool:
        """Return whether the fixture still owns admitted work."""
        return bool(self._live)

    def abort_request(self, request_ids: list[str]) -> None:
        """Drop admitted requests, as the shipped teardown does."""
        self._live = [value for value in self._live if value not in set(request_ids)]

    @staticmethod
    def kv_pool_tokens() -> int:
        """Return a pool wide enough for any fixture prompt."""
        return 1_000_000

    def close(self) -> None:
        """Record the release the fleet runtime performs on every exit."""
        self.closed += 1


class _Core(MinistralReceiverCore):
    """The shipped receiver core with its engine and warmup injected."""

    def __init__(self, engine: _Engine, *, receiver_codec: MinistralTextCodec) -> None:
        """Bind the fixture engine seam without opening a model."""
        super().__init__(
            engine=engine,
            receiver_codec=receiver_codec,
            embedding_weight=embedding_weight(),
            execution_identity={
                "unified_plan_fingerprint": PLAN_FINGERPRINT,
                "runtime_fingerprint": RUNTIME_FINGERPRINT,
                "runtime_signature": {"engine_role": SPLIT_RECEIVER_ENGINE_IDENTITY.engine_role},
            },
            engine_contract=SPLIT_RECEIVER_ENGINE_IDENTITY,
        )

    def warm(self) -> None:
        """Skip the live identity check the fixture engine cannot satisfy."""
        self.warmup_s = 0.25
        self.kv_pool_tokens = 1_000_000
        # Opening the stream below builds the shipped claim cap, memory
        # reservation, and streaming decoder over the fixture's engine.
        self.open_stream()


def receiver(
    arm: str,
    *,
    bank: Any,
    engine: _Engine | None = None,
    fused_reports: Any = None,
) -> MinistralSplitReceiver:
    """Build one shipped split or fused receiver over the fixture seams."""
    receiver_codec = codec()
    kwargs: dict[str, Any] = {
        "arm": arm,
        "items": items(),
        "bank": bank,
        "core": _Core(engine or _Engine(), receiver_codec=receiver_codec),
        "report_codecs": {name: codec(name) for name in ("text_primary", "text_medium")},
    }
    if fused_reports is None:
        return MinistralSplitReceiver(**kwargs)
    return MinistralFusedReceiver(reports=fused_reports, **kwargs)


def drain_one(live: MinistralSplitReceiver, limit: int = 32) -> list[Any]:
    """Pump until one cell completes, as the fleet runtime's loop does.

    ``pump`` is one engine step, not one cell: the timed s0 sample finishes
    first and its peers are offered after it, so the caller keeps stepping.
    """
    for _step in range(limit):
        finished = live.pump()
        if finished:
            return list(finished)
    raise AssertionError("Ministral receiver completed no cell")


def bundle_from_raw(
    qid: str,
    semantic_arm: str,
    raw: tuple[tuple[int, ...], ...],
    *,
    injected: tuple[bool, ...] = (False, False, False),
    failed: tuple[int, ...] = (),
    draws_n: int = 1,
    redraw_wall_s: float = 0.0,
    generation_s: float = 1.5,
) -> MinistralReportBundle:
    """Build one shipped bundle from exactly these banked per-worker ids.

    Every derived field is computed the way the shipped reader computes it, so
    the independent rescore can reproduce a bundle built here.
    """
    read = tuple(
        ((), "") if worker in failed else _READER.visible_report(row)
        for worker, row in enumerate(raw)
    )
    visible = tuple(ids for ids, _text in read)
    return MinistralReportBundle(
        qid=qid,
        semantic_arm=semantic_arm,
        reports=tuple(text for _ids, text in read),
        raw_outputs=tuple(_Tokenizer.decode(stop_trimmed(row)) for row in raw),
        raw_token_ids_by_worker=raw,
        visible_token_ids_by_worker=visible,
        finish_reasons=("stop",) * 3,
        seeds=FANOUTQA_NATURAL_DEV50.report_seeds(qid, "s0"),
        seed_tag="s0",
        prompt_sha256=("c" * 64,) * 3,
        prepared_prompt_fingerprint="d" * 64,
        decode=MinistralDecodeSpec(),
        generation_s=generation_s,
        queue_s_mean=0.1,
        ttft_s_mean=0.2,
        report_failed_workers=failed,
        draws_n=draws_n,
        redraw_wall_s=redraw_wall_s,
        injected_by_worker=injected,
    )


def report_bundle(qid: str, semantic_arm: str) -> MinistralReportBundle:
    """Build one sealed three-worker report bundle for a text arm."""
    return bundle_from_raw(qid, semantic_arm, (ANSWER_TOKENS,) * 3)


def items(panel: str = "production") -> dict[str, dict[str, Any]]:
    """Return the fixture prepared panel every role of one arm reads."""
    return {
        qid: {
            "qid": qid,
            "panel": panel,
            "question": Question(qid=qid, question="which leaf", pages=(), raw={}),
            "prompt_ids": tuple(
                torch.tensor(_prompt_token_ids(worker), dtype=torch.long) for worker in range(3)
            ),
            # The real prepared artifact is three sealed Tekken prompt rows, which
            # only the live sender snapshot can create. These fixtures stop at the
            # sender seam, so the panel carries the identity that seam reads.
            "artifacts": {arm: {"qid": qid, "semantic_arm": arm} for arm in MINISTRAL_TEXT_ARMS},
            "construction": construction_manifests(qid),
        }
        for qid in QIDS
    }


def _prompt_token_ids(worker: int) -> list[int]:
    return [(index + worker) % 400 for index in range(FANOUTQA_NATURAL_DEV50.worker_prompt_tokens)]


def worker_capture(worker: int) -> MinistralWorkerCapture:
    """Build one complete worker capture artifact, prompt plus roll, without a model."""
    weight = embedding_weight()
    ids = _prompt_token_ids(worker)
    prompt_rows = token_embedding_rows(weight, ids)
    rolled = (
        torch.arange(LATENT_STEPS * HIDDEN, dtype=torch.float32)
        .reshape(LATENT_STEPS, HIDDEN)
        .to(torch.bfloat16)
        + worker
    )
    payload = len(ids) + LATENT_STEPS
    scores = {
        name: torch.linspace(0.0, 1.0, payload, dtype=torch.float32) + offset
        for offset, name in ((0.5, "support"),)
    }
    return MinistralWorkerCapture(
        scores=scores,
        layer_bank=cast(LayerBank, SimpleNamespace(scope="global-only")),
        embedding_rows=torch.cat((prompt_rows, rolled), dim=0),
        prompt_token_ids=tuple(ids),
        prompt_tokens=len(ids),
    )


def fake_capture(bridge: Any, item: Any, worker: int, realign: Any, frame: Any) -> Any:
    """Return the artifact the real capture returns, without a live engine."""
    del bridge, item, realign, frame
    return worker_capture(worker)


class _LatentProducer(MinistralLatentProducer):
    """The shipped latent producer with its capture engine injected."""

    def warm(self) -> None:
        """Open the fixture engine and skip the live identity check."""
        self.engine = SimpleNamespace(
            embedding_weight=embedding_weight(),
            capture_bridge=SimpleNamespace(),
            close=lambda: None,
        )
        self.realign = cast(Realign, SimpleNamespace(enabled=False))
        self.bank({"kind": "phase", "phase": "ministral_split_producer_warm", "arm": self.arm})


def producer(root: Path, bank: Any, *, arm: str = SUPPORT_ARM) -> MinistralLatentProducer:
    """Build one shipped latent producer publishing into the arm's handoff root."""
    return _LatentProducer(
        arm=arm,
        items=items(),
        handoff_root=handoff_root(root, arm),
        bank=bank,
        worker_index=0,
        codec=codec(),
        capture=fake_capture,
    )


class _TextProducer(MinistralTextProducer):
    """The shipped text producer with its sender checkpoint injected."""

    def warm(self) -> None:
        """Open the fixture sender and skip the live warmup decode."""
        self.engine = SimpleNamespace(close=lambda: None)
        self.bank({"kind": "phase", "phase": "ministral_split_text_producer_warm", "arm": self.arm})


def text_producer(root: Path, bank: Any, *, arm: str = TEXT_ARM) -> MinistralTextProducer:
    """Build one shipped text producer publishing into the arm's handoff root."""
    return _TextProducer(
        arm=arm,
        items=items(),
        handoff_root=handoff_root(root, arm),
        bank=bank,
        codec=codec(arm),
    )


def fixture_prompt_records(artifact: Any) -> Any:
    """Stand in for hydrating three sealed prompt rows off the panel."""
    return artifact


def fixture_reports(
    engine: Any,
    codec_: Any,
    records: Any,
    *,
    profile: Any = None,
) -> MinistralReportBundle:
    """Stand in for the sender decode, returning the arm's sealed bundle."""
    del engine, codec_, profile
    return report_bundle(str(records["qid"]), str(records["semantic_arm"]))


def fleet_config(tmp_path: Path) -> MinistralFleetConfig:
    """Return one valid split-fleet invocation over throwaway paths."""
    return MinistralFleetConfig(
        results_base=tmp_path,
        source_bundle=tmp_path / "source",
        prepared_panel=tmp_path / "prompts.json",
        tokenizer_snapshots={arm: tmp_path / arm for arm in MINISTRAL_TEXT_ARMS},
        item_count=15,
        node_gpus=1,
        worker_index=0,
        attempt_id="a1",
        source_commit="0" * 40,
        # The receiver core stamps this on every row it banks, and the durable
        # bank refuses a row that overrides a fixed identity field, so the
        # fixture run identity is the one the fixture receiver carries.
        unified_plan_fingerprint=PLAN_FINGERPRINT,
        nonresident_runtime_authority={"fixture": True},
    )


class Bank:
    """A thread-safe in-memory stand-in for one arm's durable worker bank."""

    def __init__(self) -> None:
        """Start an empty bank behind one writer lock."""
        self.lock = threading.Lock()
        self.rows: list[dict[str, Any]] = []

    def append(self, row: Any) -> dict[str, Any]:
        """Append one row and return it, exactly as the durable writer does."""
        with self.lock:
            self.rows.append(dict(row))
        return dict(row)

    def results(self) -> list[dict[str, Any]]:
        """Return every banked result row."""
        with self.lock:
            return [row for row in self.rows if row.get("kind") == "result"]

    def completed(self) -> set[str]:
        """Return the qids the bank holds the one completing timed row for.

        The rule the split adapter's ``completion_key`` applies: the batched
        peers bank first, and only the timed row completes the item.
        """
        return {
            str(row["qid"]) for row in self.results() if row.get("sample_tag") == TIMED_SAMPLE_TAG
        }

    def stages(self) -> list[dict[str, Any]]:
        """Return every banked stage-ledger row."""
        with self.lock:
            return [row for row in self.rows if row.get("kind") == "stage"]


class _FixturePlacements(MinistralSplitPlacementAdapter):
    """The registered arms at the node width one test process can drive."""

    def __init__(self, placements: dict[str, FleetPlacement]) -> None:
        """Bind the exact arm order and per-arm placement this fixture runs."""
        self._placements = dict(placements)

    @property
    def arm_names(self) -> tuple[str, ...]:
        """Return the fixture roster, in the order the driver runs it."""
        return tuple(self._placements)

    def resolve(
        self,
        root: Path,
        *,
        panel: str,
        source_commit: str,
    ) -> Mapping[str, FleetPlacement]:
        """Return the fixture placements for exactly the fixture roster."""
        del root, panel, source_commit
        return dict(self._placements)


class _FixtureData:
    """The fixture panel loader: the sealed head items, no prepared bundle."""

    @property
    def qids(self) -> tuple[str, ...]:
        """Return the fixture panel in its frozen order."""
        return QIDS

    def load_prepared(
        self,
        root: Path,
        *,
        panel: str,
        source_commit: str,
    ) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
        """Return the fixture items and the manifest the identity binds to."""
        del root, source_commit
        panel_items = items(panel)
        return tuple(panel_items.values()), {
            "artifact_sha256": PLAN_FINGERPRINT,
            "construction_roster_fingerprint": CONSTRUCTION_ROSTER_FINGERPRINT,
        }

    def item(self, qid: str) -> dict[str, Any]:
        """Return one fixture prepared item."""
        return items()[qid]


class _FixtureAdapter(MinistralSplitFleetAdapter):
    """The shipped adapter with the panel loader and the engines faked.

    A test process has no prepared bundle and no eight GPUs, so the panel and
    the seats come from the fixture builders and the rest stays shipped code.
    """

    def __init__(self, root: Path, placements: dict[str, FleetPlacement]) -> None:
        """Bind one fixture run root, arm order, and per-arm placement."""
        super().__init__(
            fleet_config(root),
            codec_loader=lambda _path, arm: codec(arm),
            authority_validator=lambda raw: dict(raw),
        )
        self.data = _FixtureData()
        self.placements = _FixturePlacements(placements)
        self.reports = ministral_split_report_adapter(
            self.placements,
            self.completion_key,
            unified_plan_fingerprint=self.config.unified_plan_fingerprint,
        )

    def build_worker(self, context: Any, bank: Any) -> Any:
        """Return the fixture producer or receiver for one physical seat."""
        if context.placement.role(context.worker_index) == "producer":
            return producer(context.root, bank, arm=context.arm)
        return receiver(context.arm, bank=bank)


def fixture_adapter(
    root: Path,
    placements: dict[str, FleetPlacement],
) -> MinistralSplitFleetAdapter:
    """Return the shipped Ministral adapter wired for one offline fixture run."""
    return _FixtureAdapter(root, placements)
