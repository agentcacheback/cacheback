"""The Ministral receiver mechanism shared by the resident and split routes.

It binds one warm decode engine and serves item-arm cells on it: the signed
manager turn, the rows prepended to it, and the pricing on engine stamps.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50, arms_with_full
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.ministral.engine_roles import (
    CAPTURE_ENGINE_IDENTITY,
    MINISTRAL_ENGINE_MAX_NUM_SEQS,
    MinistralEngineIdentity,
    verify_engine_identity,
)
from rcc.models.ministral.payload import MinistralFlatPayload
from rcc.models.ministral.receiver import (
    MinistralReceiverPrompt,
    MinistralReceiverRequest,
    ReceiverCompletion,
    ministral_answer_closer,
    prepare_receiver_prompt,
    receiver_requests,
    reconstruct_visible_answers,
    warm_embed_engine,
)
from rcc.models.ministral.results import build_result_rows
from rcc.models.ministral.text import MinistralReportBundle
from rcc.models.ministral.text_codec import MinistralTextCodec
from rcc.run.fleet.admission import AdmissionGate
from rcc.run.fleet.latency import ItemLatency
from rcc.run.fleet.stream import ItemStreamTracker, StreamCompletion, StreamDecoder, StreamRequest

Score = Callable[[str], Mapping[str, float]]


def _stamp(value: float | None, *, name: str, label: str) -> float:
    """Read one required engine stamp, refusing a missing or malformed clock.

    A stamp the engine never wrote is an unpriced sample, not a zero one:
    substituting 0.0 for a unix-epoch clock would bank a latency in the billions.
    """
    if value is None:
        raise RuntimeError(f"{label}: Ministral completion has no {name} stamp")
    if isinstance(value, bool) or not math.isfinite(value):
        raise RuntimeError(f"{label}: Ministral completion stamp {name} is malformed")
    return float(value)


def item_latency(
    qid: str,
    semantic_arm: str,
    completions: Sequence[ReceiverCompletion],
    *,
    producer_s: float,
    receiver_prepare_s: float,
    spill_load_s: float = 0.0,
    spill_save_s: float = 0.0,
) -> ItemLatency:
    """Price one cell on the engine's own clocks, timing only the s0 sample."""
    label = f"{qid}/{semantic_arm}"
    roster = tuple(completions)
    if len(roster) != len(FANOUTQA_NATURAL_DEV50.sample_tags):
        raise RuntimeError(f"{label}: Ministral latency needs one complete three-sample cell")
    tags = FANOUTQA_NATURAL_DEV50.sample_tags
    finished = tuple(
        _stamp(row.last_token_ts, name="last_token_ts", label=f"{label}/{tags[index]}")
        for index, row in enumerate(roster)
    )
    queued = tuple(
        _stamp(row.queued_ts, name="queued_ts", label=f"{label}/{tags[index]}")
        for index, row in enumerate(roster)
    )
    # The one legitimate fallback: a completion that emitted no token has no
    # first-token stamp, and its TTFT is its completion.
    first = tuple(
        finished[index]
        if row.first_token_ts is None
        else _stamp(row.first_token_ts, name="first_token_ts", label=f"{label}/{tags[index]}")
        for index, row in enumerate(roster)
    )
    origin = queued[0]

    def continuation(row: ReceiverCompletion, index: int, name: str) -> float | None:
        value = getattr(row, name, None)
        if value is None:
            return None
        return _stamp(value, name=name, label=f"{label}/{tags[index]}") - origin

    return ItemLatency(
        producer_s=producer_s,
        receiver_prepare_s=receiver_prepare_s,
        decode_s=finished[0] - queued[0],
        decode_batched_s=max(finished[1:]) - min(queued[1:]),
        receiver_ttft_s=first[0] - queued[0],
        generation_s=finished[0] - first[0],
        queued_offsets=tuple(value - origin for value in queued),
        first_token_offsets=tuple(value - origin for value in first),
        finished_offsets=tuple(value - origin for value in finished),
        spill_load_s=spill_load_s,
        spill_save_s=spill_save_s,
        continuation_submit_offsets=tuple(
            continuation(row, index, "continuation_queued_ts") for index, row in enumerate(roster)
        ),
        continuation_first_token_offsets=tuple(
            continuation(row, index, "continuation_first_token_ts")
            for index, row in enumerate(roster)
        ),
    )


@dataclass(frozen=True)
class MinistralReceiverCell:
    """One item-arm cell: everything a receiver needs for exactly three draws.

    A split receiver owns one arm, so the cell, not the item, is the unit the
    mechanism runs on.
    """

    qid: str
    semantic_arm: str
    manager_token_ids: tuple[int, ...]
    #: Producer wall this arm is charged for: capture plus selection on a
    #: latent arm, the sender's report generation on a text arm, and zero on an
    #: arm whose receiver reads no handoff at all.
    producer_s: float
    payload: MinistralFlatPayload | None = None
    report_bundle: MinistralReportBundle | None = None
    report_codec: MinistralTextCodec | None = None
    spill_load_s: float = 0.0
    spill_save_s: float = 0.0

    def __post_init__(self) -> None:
        """Refuse a cell whose arm, clocks, or manager turn are malformed."""
        if self.semantic_arm not in {arm.arm_id for arm in arms_with_full(FANOUTQA_NATURAL_DEV50)}:
            raise ValueError(f"unregistered Ministral semantic arm {self.semantic_arm!r}")
        if not self.manager_token_ids or any(
            type(token) is not int or token < 0 for token in self.manager_token_ids
        ):
            raise ValueError("Ministral manager prompt contains malformed token ids")
        if min(self.producer_s, self.spill_load_s, self.spill_save_s) < 0:
            raise ValueError("Ministral receiver cell clocks must be nonnegative")
        if self.payload is not None and self.payload.semantic_arm != self.semantic_arm:
            raise ValueError("Ministral latent payload is bound to another semantic arm")
        if self.report_bundle is not None and (
            (self.report_bundle.qid, self.report_bundle.semantic_arm)
            != (self.qid, self.semantic_arm)
        ):
            raise ValueError("Ministral report bundle is bound to another item or arm")


@dataclass
class _StreamedCompletion:
    """One streamed sample restated in the raw-completion vocabulary.

    The streaming decoder reports driver-clock stamps, but this lane prices a
    cell on the engine's own, carried in each completion's metrics object.
    """

    request_id: str
    text: str
    token_ids: Sequence[int]
    n_tokens: int
    finish_reason: str
    num_cached_tokens: int | None
    queued_ts: float | None
    first_token_ts: float | None
    last_token_ts: float | None
    continuation_queued_ts: float | None = None
    continuation_first_token_ts: float | None = None
    #: Carried through from the stream: set when the one closing continuation
    #: is issued, and banked by the rows as the activation rate.
    answer_injected: bool = False


def _streamed_completion(completion: StreamCompletion) -> _StreamedCompletion:
    """Restate one stream completion on the engine's own clocks."""
    metrics = completion.engine_metrics
    stamps: dict[str, float | None] = {}
    for name in (
        "queued_ts",
        "first_token_ts",
        "last_token_ts",
        "continuation_queued_ts",
        "continuation_first_token_ts",
    ):
        value = getattr(metrics, name, None)
        if value is None:
            stamps[name] = None
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"vLLM returned invalid {name} {value!r}")
        else:
            stamps[name] = float(value)
    return _StreamedCompletion(
        request_id=completion.request_id,
        text=completion.text,
        token_ids=completion.token_ids,
        n_tokens=completion.n_tokens,
        finish_reason=completion.finish_reason,
        num_cached_tokens=completion.num_cached_tokens,
        answer_injected=completion.answer_injected,
        **stamps,
    )


@dataclass(frozen=True)
class MinistralPreparedCell:
    """One claimed cell whose prompt is built and whose samples are signed."""

    cell: MinistralReceiverCell
    prompt: MinistralReceiverPrompt
    requests: tuple[MinistralReceiverRequest, ...]
    receiver_prepare_s: float


def manager_prompt_tokens(
    codec: MinistralTextCodec,
    *,
    qid: str,
    semantic_arm: str,
    question: str,
    reports: Sequence[str] = (),
    report_failed: bool = False,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> tuple[tuple[int, ...], dict[str, object]]:
    """Sign one coordinator turn and return its tokens and its identity record.

    Every Ministral route that needs a coordinator turn signs it here.
    """
    record = codec.encode_manager_prompt(
        qid=qid,
        semantic_arm=semantic_arm,
        source_identity=profile.source_logical_fingerprint,
        question=question,
        reports=reports,
        report_failed=report_failed,
        profile=profile,
    )
    raw = record.get("prompt_token_ids")
    if not isinstance(raw, list):
        raise RuntimeError(f"{qid}/{semantic_arm}: manager prompt token ids are malformed")
    values = cast(list[Any], raw)
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in values):
        raise RuntimeError(f"{qid}/{semantic_arm}: manager prompt token ids are malformed")
    tokens = tuple(int(token) for token in cast(list[int], values))
    if not tokens:
        raise RuntimeError(f"{qid}/{semantic_arm}: manager prompt is empty")
    return tokens, dict(record)


def question_token_ids(
    codec: MinistralTextCodec,
    manager_token_ids: Sequence[int],
    question: str,
) -> tuple[int, ...]:
    """Locate the question's exact in-context Tekken token rows.

    Byte-level BPE merges across context boundaries, so a standalone re-encoding
    need not appear verbatim; the span is located by character offset.
    """
    inner: Any = codec.tokenizer.instruct_tokenizer.tokenizer
    prompt = tuple(int(token) for token in manager_token_ids)
    decoded = str(inner.decode(list(prompt)))
    char_start = decoded.rfind(question)
    if char_start < 0:
        raise RuntimeError("Tekken could not locate the question in the manager prompt")
    char_end = char_start + len(question)
    lengths = [len(str(inner.decode(list(prompt[:count])))) for count in range(len(prompt) + 1)]
    start = max((count for count in range(len(prompt)) if lengths[count] <= char_start), default=0)
    stop = min(
        (count for count in range(1, len(prompt) + 1) if lengths[count] >= char_end),
        default=len(prompt),
    )
    # One-token widening absorbs replacement-character wobble at a prefix cut
    # inside a multi-byte character; the containment check still decides.
    for lo, hi in ((start, stop), (max(start - 1, 0), min(stop + 1, len(prompt)))):
        span = prompt[lo:hi]
        if span and question in str(inner.decode(list(span))):
            return span
    raise RuntimeError("Tekken could not locate the question in the manager prompt")


class MinistralReceiverCore:
    """One warm Ministral decode engine and the cells it serves on it."""

    def __init__(
        self,
        *,
        engine: Any,
        receiver_codec: MinistralTextCodec,
        embedding_weight: torch.Tensor,
        execution_identity: Mapping[str, Any],
        model_load_s: float = 0.0,
        engine_contract: MinistralEngineIdentity = CAPTURE_ENGINE_IDENTITY,
        profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
    ) -> None:
        """Bind one live engine, the registered 14B codec, and its table."""
        if embedding_weight.ndim != 2:
            raise ValueError("Ministral receiver embedding weight must be rank two")
        self.profile = profile
        self.engine: Any | None = engine
        self.receiver_codec = receiver_codec
        self.embedding_weight = embedding_weight
        self.execution_identity = dict(execution_identity)
        self.engine_contract = engine_contract
        self.engine_identity: dict[str, Any] = {}
        self.model_load_s = float(model_load_s)
        self.warmup_s = 0.0
        self.kv_pool_tokens = 0
        self.tracker: ItemStreamTracker | None = None
        #: One until the engine is warm and its pool is measured, then the
        #: registered engine sequence limit.
        self.claim_limit = 1
        self.capacity_tokens = 0

    def warm(self) -> None:
        """Fail-closed verify the role engine, warm it, and open the stream."""
        engine = self.engine
        if engine is None:
            raise RuntimeError("Ministral receiver engine was closed before warmup")
        self.engine_identity = verify_engine_identity(engine, self.engine_contract)
        # Warmed on the exact prompt-embeds route the receiver is later timed
        # on, so lazy allocation and graph capture are not charged to whichever
        # item arrives first.
        self.warmup_s = warm_embed_engine(
            engine,
            self.embedding_weight,
            self.receiver_codec.encode_warmup_prompt(),
            tag=f"ministral-{self.engine_contract.role}",
        )
        pool_tokens = engine.kv_pool_tokens()
        self.kv_pool_tokens = 0 if pool_tokens is None else int(pool_tokens)
        self.open_stream()

    def open_stream(self) -> None:
        """Bind the admission gate and the streaming decoder to this engine.

        The gate reserves prompt rows plus the full answer ceiling per live
        request and back-pressures by queueing, never by refusing a claim.
        """
        engine = self.engine
        if engine is None:
            raise RuntimeError("Ministral receiver engine is unavailable")
        if not self.kv_pool_tokens:
            raise RuntimeError("Ministral receiver KV pool capacity is unreadable")
        gate = AdmissionGate(
            self.kv_pool_tokens,
            answer_ceiling=self.profile.answer_ceiling,
            max_in_flight=MINISTRAL_ENGINE_MAX_NUM_SEQS,
        )
        # Wall clock, so every banked stamp lands in the fleet ledger's own
        # domain. The closer binds here rather than at warm time because the
        # continuation prompt is assembled through this engine's validator.
        self.tracker = ItemStreamTracker(
            StreamDecoder(
                engine.stream_handle,
                clock=time.time,
                closer=ministral_answer_closer(
                    self.receiver_codec,
                    self.embedding_weight,
                    embedding_prompt=engine.embedding_prompt,
                    profile=self.profile,
                ),
            ),
            gate,
        )
        self.claim_limit = MINISTRAL_ENGINE_MAX_NUM_SEQS
        self.capacity_tokens = gate.capacity_tokens

    def prepare(self, cell: MinistralReceiverCell) -> MinistralPreparedCell:
        """Build one cell's embedding prompt and sign its three samples."""
        started = time.perf_counter()
        prompt = prepare_receiver_prompt(
            self.embedding_weight,
            cell.manager_token_ids,
            payload=cell.payload,
        )
        receiver_prepare_s = time.perf_counter() - started
        return MinistralPreparedCell(
            cell=cell,
            prompt=prompt,
            requests=receiver_requests(cell.qid, cell.semantic_arm, prompt, profile=self.profile),
            receiver_prepare_s=receiver_prepare_s,
        )

    def offer(
        self,
        prepared: MinistralPreparedCell,
        batch: Sequence[MinistralReceiverRequest],
    ) -> None:
        """Queue one decode phase of a prepared cell behind the admission gate."""
        engine, tracker = self.engine, self.tracker
        if engine is None or tracker is None:
            raise RuntimeError("Ministral receiver stream is not open")
        # Validates rank, row count, hidden width, and finiteness before any of
        # it reaches the engine.
        prompt = engine.embedding_prompt(prepared.prompt.prompt_embeds)
        tracker.offer(
            prepared.cell.qid,
            [
                StreamRequest(
                    request_id=request.request_id,
                    prompt=prompt,
                    sampling=engine.stream_sampling(request),
                    prompt_tokens=prepared.prompt.prompt_rows,
                    qid=prepared.cell.qid,
                    tag=request.sample_tag,
                )
                for request in batch
            ],
        )

    def pump(self) -> list[tuple[str, float, list[StreamCompletion]]]:
        """Admit what fits, step the engine once, and return finished phases."""
        return [] if self.tracker is None else self.tracker.pump()

    def build_rows(
        self,
        prepared: MinistralPreparedCell,
        completions: Sequence[StreamCompletion],
        *,
        score: Score,
    ) -> tuple[dict[str, object], ...]:
        """Reconstruct and score one complete three-sample cell."""
        cell = prepared.cell
        by_id = {completion.request_id: completion for completion in completions}
        ordered = tuple(
            _streamed_completion(by_id[request.request_id]) for request in prepared.requests
        )
        answers = reconstruct_visible_answers(self.receiver_codec, prepared.requests, ordered)
        rows = build_result_rows(
            prepared.requests,
            answers,
            score=score,
            execution_identity=self.execution_identity,
            latency=item_latency(
                cell.qid,
                cell.semantic_arm,
                ordered,
                producer_s=cell.producer_s,
                receiver_prepare_s=prepared.receiver_prepare_s,
                spill_load_s=cell.spill_load_s,
                spill_save_s=cell.spill_save_s,
            ),
            report_bundle=cell.report_bundle,
            report_codec=cell.report_codec,
            profile=self.profile,
        )
        return self._verified_rows(cell, rows)

    def idle(self) -> bool:
        """Return whether the stream holds no queued or admitted request."""
        return self.tracker is None or self.tracker.idle()

    @property
    def admission_trace(self) -> Mapping[str, Mapping[str, Any]]:
        """Return the per-item reservation trace the gate recorded at admission."""
        return {} if self.tracker is None else self.tracker.admission_trace

    @staticmethod
    def _verified_rows(
        cell: MinistralReceiverCell,
        rows: tuple[dict[str, object], ...],
    ) -> tuple[dict[str, object], ...]:
        """Refuse a cell whose rows lost their registered arm and seed order."""
        observed = tuple((str(row.get("arm")), row.get("seed_index")) for row in rows)
        if observed != tuple((cell.semantic_arm, seed) for seed in range(3)):
            raise RuntimeError("Ministral execution rows differ from registered arm/seed order")
        return rows

    def close(self) -> None:
        """Release the receiver engine and its borrowed embedding view."""
        tracker, self.tracker = self.tracker, None
        if tracker is not None:
            tracker.decoder.abort_all()
        engine, self.engine = self.engine, None
        self.embedding_weight = torch.empty(0)
        if engine is not None:
            engine.close()


__all__ = (
    "MinistralReceiverCell",
    "MinistralReceiverCore",
    "Score",
    "item_latency",
    "manager_prompt_tokens",
    "question_token_ids",
)
