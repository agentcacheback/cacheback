"""Report generation for the Ministral text arms.

The Ministral text engine calls it with one item's three worker prompts.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, replace

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.ministral import MINISTRAL
from rcc.models.ministral.text_codec import (
    MINISTRAL_REPORT_SEED_TAGS,
    MINISTRAL_STOP_TOKEN_IDS,
    MINISTRAL_TEXT_ARMS,
    MINISTRAL_THINK_CLOSE_ID,
    MINISTRAL_THINK_OPEN_ID,
    MinistralTextCodec,
    load_registered_text_codec,
    stop_trimmed,
)
from rcc.models.ministral.text_decode import (
    FINISH_REASONS,
    MINISTRAL_CLOSING_TOKEN_BUDGET,
    MinistralDecodeRequest,
    MinistralDecodeSpec,
    TextCompletion,
    TextEngine,
)
from rcc.models.ministral.text_prompt import (
    MinistralPreparedPromptArtifact,
    MinistralPromptRecord,
    MinistralTextSender,
    bind_prompt_records,
    prepared_prompt_fingerprint,
    registered_text_senders,
)
from rcc.topologies.fanout import FANOUT_M3


def _validate_injected_ids(raw: Sequence[int]) -> None:
    """Refuse banked ids that cannot be the shipped text of a continued draw.

    A row claiming an injection carries the closer id with no stop token before
    it, so a reader cutting at the first stop id still sees the continuation.
    """
    tokens = tuple(int(token) for token in raw)
    if MINISTRAL_THINK_CLOSE_ID not in tokens:
        raise ValueError("Ministral closer injection must bank its closing delimiter")
    closer = tokens.index(MINISTRAL_THINK_CLOSE_ID)
    if any(token in MINISTRAL_STOP_TOKEN_IDS for token in tokens[:closer]):
        raise ValueError("Ministral closer injection must replace the head's end of turn")


@dataclass(frozen=True)
class MinistralReportBundle:
    """Three accepted reports with independent raw-token reconstruction."""

    qid: str
    semantic_arm: str
    reports: tuple[str, ...]
    raw_outputs: tuple[str, ...]
    raw_token_ids_by_worker: tuple[tuple[int, ...], ...]
    visible_token_ids_by_worker: tuple[tuple[int, ...], ...]
    finish_reasons: tuple[str, ...]
    seeds: tuple[int, ...]
    seed_tag: str
    prompt_sha256: tuple[str, ...]
    prepared_prompt_fingerprint: str
    decode: MinistralDecodeSpec
    generation_s: float
    queue_s_mean: float | None
    ttft_s_mean: float | None
    #: The draw count and the wall clock of the draws the redraw ladder
    #: discarded. Required rather than defaulted, so a redrawn cell cannot be
    #: banked as a single draw.
    draws_n: int
    redraw_wall_s: float
    injected_by_worker: tuple[bool, ...]
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50
    report_failed_workers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Validate one complete all-worker draw, quarantined workers included."""
        fields = (
            self.reports,
            self.raw_outputs,
            self.raw_token_ids_by_worker,
            self.visible_token_ids_by_worker,
            self.finish_reasons,
            self.seeds,
            self.prompt_sha256,
        )
        if self.semantic_arm not in MINISTRAL_TEXT_ARMS or any(
            len(field) != FANOUT_M3.workers_per_item for field in fields
        ):
            raise ValueError("Ministral report bundle is not one registered M=3 text lane")
        if self.seeds != self.profile.report_seeds(self.qid, self.seed_tag):
            raise ValueError("Ministral report bundle seeds differ from the sealed panel")
        failed = self.report_failed_workers
        if list(failed) != sorted(set(failed)) or any(
            worker not in range(FANOUT_M3.workers_per_item) for worker in failed
        ):
            raise ValueError("Ministral report bundle names unregistered quarantined workers")
        for worker, report in enumerate(self.reports):
            if worker in failed:
                if report:
                    raise ValueError("Ministral quarantined workers must carry empty reports")
            elif not report.strip():
                raise ValueError(
                    "Ministral reports require closed thinking and substantive content"
                )
        if len(self.prepared_prompt_fingerprint) != 64:
            raise ValueError("Ministral result lacks prepared prompt authority")
        self._validate_ladder()

    def _validate_ladder(self) -> None:
        """Validate the redraw ladder banking and the closer-injection evidence."""
        if not 1 <= self.draws_n <= len(MINISTRAL_REPORT_SEED_TAGS) or self.redraw_wall_s < 0.0:
            raise ValueError(
                "Ministral report bundle must bank one to "
                f"{len(MINISTRAL_REPORT_SEED_TAGS)} draws and no negative redraw"
            )
        if self.draws_n == 1 and self.redraw_wall_s != 0.0:
            raise ValueError("Ministral single-draw report bundle cannot carry redraw wall clock")
        if len(self.injected_by_worker) != FANOUT_M3.workers_per_item:
            raise ValueError("Ministral report bundle needs one injection flag per worker")
        for injected, raw in zip(
            self.injected_by_worker, self.raw_token_ids_by_worker, strict=True
        ):
            if injected:
                _validate_injected_ids(raw)

    @property
    def report_failed(self) -> bool:
        """Return whether draw exhaustion quarantined this cell."""
        return bool(self.report_failed_workers)

    def result_fields(self) -> dict[str, object]:
        """Return exact serializable report evidence for result rows."""
        sender = next(
            sender
            for sender in registered_text_senders()
            if sender.semantic_arm == self.semantic_arm
        )
        return {
            "qid": self.qid,
            "semantic_arm": self.semantic_arm,
            "policy": sender.policy,
            "worker_checkpoint": sender.checkpoint,
            "worker_revision": sender.revision,
            "worker_tokenizer": sender.checkpoint,
            "worker_tokenizer_revision": sender.revision,
            "receiver_checkpoint": sender.receiver_checkpoint,
            "receiver_revision": sender.receiver_revision,
            "workers": FANOUT_M3.workers_per_item,
            "reports": list(self.reports),
            "report_raw_outputs": list(self.raw_outputs),
            "report_tokens": sum(len(tokens) for tokens in self.raw_token_ids_by_worker),
            "report_tokens_by_worker": [len(tokens) for tokens in self.raw_token_ids_by_worker],
            "report_token_ids_by_worker": [list(tokens) for tokens in self.raw_token_ids_by_worker],
            "report_visible_token_ids_by_worker": [
                list(tokens) for tokens in self.visible_token_ids_by_worker
            ],
            "report_visible_tokens_by_worker": [
                len(tokens) for tokens in self.visible_token_ids_by_worker
            ],
            "report_generation_s": round(self.generation_s, 4),
            "draws_n": self.draws_n,
            "redraw_wall_s": round(self.redraw_wall_s, 4),
            "injected": any(self.injected_by_worker),
            "report_injected_by_worker": list(self.injected_by_worker),
            "report_seed_tag": self.seed_tag,
            "report_seeds": list(self.seeds),
            "report_thinking_closed": [
                MINISTRAL.decode.enable_thinking and worker not in self.report_failed_workers
                for worker in range(FANOUT_M3.workers_per_item)
            ],
            "report_thinking_closed_rate": round(
                1.0 - len(self.report_failed_workers) / FANOUT_M3.workers_per_item, 4
            ),
            "report_failed": self.report_failed,
            "report_failed_workers": list(self.report_failed_workers),
            "report_failure": (
                f"workers {list(self.report_failed_workers)} unclosed or empty after "
                f"{len(MINISTRAL_REPORT_SEED_TAGS)} sealed draws"
                if self.report_failed
                else None
            ),
            "report_enable_thinking": self.decode.to_dict()["enable_thinking"],
            "report_finish_reasons": list(self.finish_reasons),
            "report_prompt_sha256": list(self.prompt_sha256),
            "prepared_prompt_fingerprint": self.prepared_prompt_fingerprint,
            "report_queue_s_mean": self.queue_s_mean,
            "report_ttft_s_mean": self.ttft_s_mean,
            "report_decode": self.decode.to_dict(),
        }


@dataclass(frozen=True)
class _ReportDraw:
    results: tuple[TextCompletion, ...]
    visible_rows: tuple[tuple[int, ...], ...]
    reports: tuple[str, ...]
    seeds: tuple[int, ...]
    tag: str
    invalid: tuple[int, ...]
    elapsed: float
    injected: tuple[bool, ...]


@dataclass
class _ContinuedCompletion:
    """One draw whose unclosed reasoning block was closed and decoded on."""

    request_id: str
    text: str
    n_tokens: int
    # Sequence, not tuple: the completion protocol declares a mutable attribute.
    token_ids: Sequence[int]
    finish_reason: str
    num_cached_tokens: int | None
    queued_ts: float | None
    scheduled_ts: float | None
    first_token_ts: float | None


def _is_wellformed(result: TextCompletion) -> bool:
    """Return whether one completion's backend evidence is usable at all."""
    return result.finish_reason in FINISH_REASONS and int(result.n_tokens) == len(result.token_ids)


def _closer_is_affordable(result: TextCompletion, *, profile: BenchmarkProfile) -> bool:
    """Return whether one draw can be closed inside its registered budgets.

    Worker prompt plus head content, closer, and closing budget must fit the
    profile's max_model_len. The report ceiling bounds the head only.
    """
    content = len(stop_trimmed(result.token_ids))
    tail = 1 + MINISTRAL_CLOSING_TOKEN_BUDGET
    return profile.worker_prompt_tokens + content + tail <= profile.max_model_len


def _merge_continuation(
    codec: MinistralTextCodec,
    head: TextCompletion,
    tail: TextCompletion,
) -> _ContinuedCompletion:
    """Bank the head and its forced continuation as one draw.

    The banked ids are head content, the closer, then the continuation, with
    the head's stop id dropped; the banked finish reason is the head's.
    """
    ids = (
        *stop_trimmed(head.token_ids),
        MINISTRAL_THINK_CLOSE_ID,
        *(int(token) for token in tail.token_ids),
    )
    return _ContinuedCompletion(
        request_id=head.request_id,
        # Decoded from the merged ids rather than concatenated from backend
        # strings, so it equals what `reconstruct_raw_report` produces later.
        text=reconstruct_raw_report(codec, ids),
        n_tokens=len(ids),
        token_ids=ids,
        finish_reason=str(head.finish_reason),
        num_cached_tokens=head.num_cached_tokens,
        queued_ts=head.queued_ts,
        scheduled_ts=head.scheduled_ts,
        first_token_ts=head.first_token_ts,
    )


def _continue_unclosed(
    engine: TextEngine,
    codec: MinistralTextCodec,
    records: Sequence[MinistralPromptRecord],
    requests: Sequence[MinistralDecodeRequest],
    results: Sequence[TextCompletion],
    *,
    qid: str,
    profile: BenchmarkProfile,
) -> tuple[tuple[TextCompletion, ...], tuple[bool, ...]]:
    """Continue each unclosed draw exactly once with the closer injected.

    The continuation is the same draw, capped at the closing budget; its prompt
    is the worker turn, the head's stop-trimmed content, and the closer id.
    """
    unclosed = tuple(
        position
        for position, result in enumerate(results)
        if _is_wellformed(result)
        and codec.thinking_is_unclosed(result.token_ids)
        and _closer_is_affordable(result, profile=profile)
    )
    # The flag records that a continuation was issued, which is the quantity
    # the reported activation rate counts, so it is set here rather than after
    # a merge succeeds.
    injected = tuple(position in set(unclosed) for position in range(len(results)))
    if not unclosed:
        return tuple(results), injected
    retry = tuple(
        replace(requests[position], decode=replace(requests[position].decode, closing=True))
        for position in unclosed
    )
    prompts = tuple(
        (
            *records[position].token_ids,
            *stop_trimmed(results[position].token_ids),
            MINISTRAL_THINK_CLOSE_ID,
        )
        for position in unclosed
    )
    continued = _order_results(qid, retry, engine.decode_token_ids_full(prompts, retry))
    _validate_continuation(qid, continued, expected=len(unclosed))
    merged = list(results)
    for position, tail in zip(unclosed, continued, strict=True):
        merged[position] = _merge_continuation(codec, results[position], tail)
    return tuple(merged), injected


def _validate_continuation(
    qid: str,
    continued: Sequence[TextCompletion],
    *,
    expected: int,
) -> None:
    """Refuse a continuation whose evidence is unusable as banked identity.

    A failed head draw goes back to the redraw ladder, but a malformed
    continuation raises: merging it would publish unverifiable ids.
    """
    if len(continued) != expected:
        raise RuntimeError(f"{qid}: Ministral closer continuation roster is incomplete")
    if any(result.finish_reason not in FINISH_REASONS for result in continued):
        raise RuntimeError(
            f"{qid}: Ministral closer continuation returned an invalid finish reason"
        )
    if any(result.num_cached_tokens not in (None, 0) for result in continued):
        raise RuntimeError(f"{qid}: prefix caching contaminated a Ministral closer continuation")
    if any(len(result.token_ids) != int(result.n_tokens) for result in continued):
        raise RuntimeError(f"{qid}: Ministral closer continuation token counts differ from its ids")


def generate_report_bundle(
    engine: TextEngine,
    codec: MinistralTextCodec,
    records: Sequence[MinistralPromptRecord],
    *,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> MinistralReportBundle:
    """Decode three reports, retrying the whole three-worker draw on invalid output.

    `report_generation_s` is the span of the draw that shipped, the last one
    executed; the draws that did not ship bank as `draws_n` and `redraw_wall_s`.
    """
    codec.require_verified()
    qid, semantic_arm, ordered = _validated_prompt_batch(codec, records)
    decode = MinistralDecodeSpec()
    draw_elapsed: list[float] = []
    draw: _ReportDraw | None = None
    invalid: tuple[int, ...] = ()
    for tag in MINISTRAL_REPORT_SEED_TAGS:
        candidate = _decode_draw(engine, codec, ordered, qid, semantic_arm, tag, decode, profile)
        draw_elapsed.append(candidate.elapsed)
        invalid = candidate.invalid
        # Draw exhaustion quarantines the cell rather than raising: the final
        # draw banks as evidence and the receiver serves the base manager
        # prompt.
        draw = candidate
        if not invalid:
            break
    if draw is None:
        raise RuntimeError(f"{qid}: Ministral report draws produced no decode")
    return MinistralReportBundle(
        report_failed_workers=invalid,
        qid=qid,
        semantic_arm=semantic_arm,
        reports=draw.reports,
        raw_outputs=tuple(
            reconstruct_raw_report(codec, result.token_ids) for result in draw.results
        ),
        raw_token_ids_by_worker=tuple(
            tuple(int(token) for token in result.token_ids) for result in draw.results
        ),
        visible_token_ids_by_worker=draw.visible_rows,
        finish_reasons=tuple(str(result.finish_reason) for result in draw.results),
        seeds=draw.seeds,
        seed_tag=draw.tag,
        prompt_sha256=tuple(record.token_sha256 for record in ordered),
        prepared_prompt_fingerprint=ordered[0].prepared_fingerprint,
        decode=decode,
        generation_s=draw_elapsed[-1],
        draws_n=len(draw_elapsed),
        redraw_wall_s=sum(draw_elapsed[:-1]),
        injected_by_worker=draw.injected,
        queue_s_mean=_mean_elapsed(draw.results, later="scheduled_ts", earlier="queued_ts"),
        ttft_s_mean=_mean_elapsed(draw.results, later="first_token_ts", earlier="queued_ts"),
        profile=profile,
    )


def reconstruct_raw_report(
    codec: MinistralTextCodec,
    raw_tokens: Sequence[int],
) -> str:
    """Decode report ids up to the registered stop token, never backend text."""
    codec.require_verified()
    tokens = tuple(int(token) for token in raw_tokens)
    stop = next(
        (index for index, token in enumerate(tokens) if token in MINISTRAL_STOP_TOKEN_IDS),
        len(tokens),
    )
    return str(codec.tokenizer.decode(list(tokens[:stop])))


def _validated_prompt_batch(
    codec: MinistralTextCodec,
    records: Sequence[MinistralPromptRecord],
) -> tuple[str, str, tuple[MinistralPromptRecord, ...]]:
    if len(records) != FANOUT_M3.workers_per_item:
        raise ValueError("Ministral text generation requires exactly three prompt records")
    qids = {record.qid for record in records}
    arms = {record.semantic_arm for record in records}
    workers = {record.worker for record in records}
    if len(qids) != 1 or len(arms) != 1 or workers != set(range(FANOUT_M3.workers_per_item)):
        raise ValueError("Ministral prompt record batch mixes item, arm, or worker identity")
    qid = qids.pop()
    semantic_arm = arms.pop()
    if any(
        (record.checkpoint, record.revision) != (codec.checkpoint, codec.revision)
        for record in records
    ):
        raise RuntimeError("Ministral report codec differs from prompt sender identity")
    if len({record.prepared_fingerprint for record in records}) != 1:
        raise RuntimeError("Ministral report prompt records mix prepared authorities")
    return qid, semantic_arm, tuple(sorted(records, key=lambda record: record.worker))


def _decode_draw(
    engine: TextEngine,
    codec: MinistralTextCodec,
    records: Sequence[MinistralPromptRecord],
    qid: str,
    semantic_arm: str,
    tag: str,
    decode: MinistralDecodeSpec,
    profile: BenchmarkProfile,
) -> _ReportDraw:
    seeds = profile.report_seeds(qid, tag)
    requests = tuple(
        MinistralDecodeRequest(
            request_id=f"{qid}:{semantic_arm}:report:w{worker}:{tag}",
            qid=qid,
            semantic_arm=semantic_arm,
            sample_tag=tag,
            member_index=worker,
            seed=seed,
            decode=decode,
            profile=profile,
        )
        for worker, seed in enumerate(seeds)
    )
    started = time.perf_counter()
    returned = tuple(
        engine.decode_token_ids_full(tuple(record.token_ids for record in records), requests)
    )
    results = _order_results(qid, requests, returned)
    if any(result.num_cached_tokens not in (None, 0) for result in results):
        raise RuntimeError(f"{qid}: prefix caching contaminated Ministral reports")
    # Inside the draw's own clock: the continuation is part of the same draw,
    # so it belongs to the published span of the draw that ships.
    results, injected = _continue_unclosed(
        engine, codec, records, requests, results, qid=qid, profile=profile
    )
    elapsed = time.perf_counter() - started
    invalid: list[int] = []
    visible_rows: list[tuple[int, ...]] = []
    reports: list[str] = []
    for worker, result in enumerate(results):
        raw = tuple(int(token) for token in result.token_ids)
        if not _is_wellformed(result):
            invalid.append(worker)
            visible_rows.append(())
            reports.append("")
            continue
        try:
            visible, report = codec.visible_report(raw)
        except RuntimeError:
            invalid.append(worker)
            visible_rows.append(())
            reports.append("")
            continue
        if not report.strip():
            invalid.append(worker)
            visible_rows.append(())
            reports.append("")
            continue
        visible_rows.append(visible)
        reports.append(report)
    return _ReportDraw(
        results=results,
        visible_rows=tuple(visible_rows),
        reports=tuple(reports),
        seeds=seeds,
        tag=tag,
        invalid=tuple(invalid),
        elapsed=elapsed,
        injected=injected,
    )


def _order_results(
    qid: str,
    requests: Sequence[MinistralDecodeRequest],
    results: Sequence[TextCompletion],
) -> tuple[TextCompletion, ...]:
    """Map completions to signed requests and refuse missing, extra, or duplicate IDs."""
    expected = {request.request_id for request in requests}
    by_id: dict[str, TextCompletion] = {}
    for result in results:
        if not result.request_id or result.request_id in by_id:
            raise RuntimeError(f"{qid}: Ministral sender returned duplicate/empty request id")
        by_id[result.request_id] = result
    if set(by_id) != expected:
        raise RuntimeError(f"{qid}: Ministral sender request identities differ from the draw")
    return tuple(by_id[request.request_id] for request in requests)


def _mean_elapsed(results: Sequence[TextCompletion], *, later: str, earlier: str) -> float | None:
    values: list[float] = []
    for result in results:
        later_value = getattr(result, later)
        earlier_value = getattr(result, earlier)
        if later_value is not None and earlier_value is not None:
            values.append(float(later_value) - float(earlier_value))
    return round(sum(values) / len(values), 4) if values else None


__all__ = (
    "MINISTRAL_CLOSING_TOKEN_BUDGET",
    "MINISTRAL_REPORT_SEED_TAGS",
    "MINISTRAL_STOP_TOKEN_IDS",
    "MINISTRAL_TEXT_ARMS",
    "MINISTRAL_THINK_CLOSE_ID",
    "MINISTRAL_THINK_OPEN_ID",
    "MinistralDecodeRequest",
    "MinistralDecodeSpec",
    "MinistralPreparedPromptArtifact",
    "MinistralPromptRecord",
    "MinistralReportBundle",
    "MinistralTextCodec",
    "MinistralTextSender",
    "bind_prompt_records",
    "generate_report_bundle",
    "load_registered_text_codec",
    "prepared_prompt_fingerprint",
    "reconstruct_raw_report",
    "registered_text_senders",
)
