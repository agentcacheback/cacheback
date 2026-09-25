"""Route-native worker report codec and three-sender generation contract.

Every delimiter, stop id, and closing budget comes from the ``RouteFamily`` the
caller hands in; the benchmark-derived constants come from the profile.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, replace

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.fanoutqa.source_padding import ProbeItem
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.qwen.text_decode import (
    QWEN_REPORT_SEED_TAGS,
    QWEN_TEXT_ARMS,
    QwenDecodeRequest,
    QwenDecodeSpec,
    QwenReportBundle,
    TextEngine,
    validate_report_ladder_fields,
)
from rcc.models.qwen.text_output import ContinuedCompletion, TextCompletion
from rcc.models.route import Decoder, RouteFamily
from rcc.topologies.fanout import FANOUT_M3


@dataclass(frozen=True)
class QwenTextSender:
    """One registered text arm and its real sender checkpoint."""

    semantic_arm: str
    policy: str
    checkpoint: str
    revision: str


def registered_text_senders(*, family: RouteFamily) -> tuple[QwenTextSender, ...]:
    """Resolve the primary, medium, and small sender lanes from the registration."""
    by_arm = {arm.semantic_arm: arm for arm in family.profile.physical_arms}
    senders: list[QwenTextSender] = []
    for semantic_arm, policy in zip(QWEN_TEXT_ARMS, family.text_policies, strict=True):
        try:
            arm = by_arm[semantic_arm]
        except KeyError as exc:
            raise RuntimeError(f"Qwen registration lacks text arm {semantic_arm!r}") from exc
        if (
            arm.policy != policy
            or arm.sender_checkpoint is None
            or arm.sender_revision is None
            or arm.selector is not None
        ):
            raise RuntimeError(f"Qwen text arm {semantic_arm!r} differs from its sender contract")
        senders.append(
            QwenTextSender(
                semantic_arm=semantic_arm,
                policy=policy,
                checkpoint=arm.sender_checkpoint,
                revision=arm.sender_revision,
            )
        )
    return tuple(senders)


def format_report_payload(reports: Sequence[str]) -> str:
    """Format three visible reports exactly as the Qwen coordinator receives them."""
    if len(reports) != 3 or not all(report.strip() for report in reports):
        raise ValueError("Qwen report payload needs three nonempty reports")
    return "\n\n".join(
        f"[worker {worker} report]\n{report.strip()}" for worker, report in enumerate(reports)
    )


def report_seed(
    qid: str,
    worker: int,
    tag: str,
    *,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> int:
    """Return one registered per-item, per-worker Qwen report seed."""
    seeds = profile.report_seeds(qid, tag)
    if worker not in range(len(seeds)):
        raise ValueError("Qwen report worker is outside the canonical M=3 topology")
    return seeds[worker]


def mean_elapsed(results: Sequence[TextCompletion], *, later: str, earlier: str) -> float | None:
    """Return the mean of one backend interval over the draws that report it."""
    values: list[float] = []
    for result in results:
        later_value = getattr(result, later)
        earlier_value = getattr(result, earlier)
        if later_value is not None and earlier_value is not None:
            values.append(float(later_value) - float(earlier_value))
    return round(sum(values) / len(values), 4) if values else None


def ordered_results(
    requests: Sequence[QwenDecodeRequest],
    results: Sequence[TextCompletion],
) -> tuple[TextCompletion, ...]:
    """Bind asynchronous backend results to the exact signed worker requests."""
    request_ids = tuple(request.request_id for request in requests)
    by_id: dict[str, TextCompletion] = {}
    for result in results:
        if result.request_id not in request_ids or result.request_id in by_id:
            raise RuntimeError("Qwen text sender returned an unknown or duplicate request id")
        by_id[result.request_id] = result
    if set(by_id) != set(request_ids):
        raise RuntimeError("Qwen text sender returned an incomplete request roster")
    return tuple(by_id[request_id] for request_id in request_ids)


def validate_draw(qid: str, results: Sequence[TextCompletion]) -> None:
    """Refuse a draw whose backend evidence is unusable as banked identity."""
    if any(result.finish_reason not in {"stop", "length"} for result in results):
        raise RuntimeError(f"{qid}: Qwen text sender returned an invalid finish reason")
    if any(result.num_cached_tokens not in (None, 0) for result in results):
        raise RuntimeError(f"{qid}: prefix caching contaminated Qwen worker reports")
    if any(len(result.token_ids) != int(result.n_tokens) for result in results):
        raise RuntimeError(f"{qid}: Qwen report token counts differ from their raw ids")


def _content_len(token_ids: Sequence[int], *, family: RouteFamily) -> int:
    """Return how many banked ids precede the first stop id.

    This is the same count as `content_token_count` in the rescore module. The
    two cannot be one function: the model tree does not import the run tree.
    """
    stops = frozenset(family.stop_token_ids)
    for index, token in enumerate(token_ids):
        if token in stops:
            return index
    return len(token_ids)


def _closer_is_affordable(
    result: TextCompletion, *, family: RouteFamily, profile: BenchmarkProfile
) -> bool:
    """Check prompt, head, closer, and continuation against the engine context.

    The prompt is priced by the ids the engine reported it consumed, since a
    chain rewrite prompt carries the running notes above its chunk floor.
    """
    content = _content_len(result.token_ids, family=family)
    tail = len(family.think_close_token_ids) + family.closing_token_budget
    return len(result.prompt_token_ids) + content + tail <= profile.max_model_len


def _merge_continuation(
    head: TextCompletion, tail: TextCompletion, *, family: RouteFamily
) -> ContinuedCompletion:
    """Bank the head and its forced continuation as one draw.

    The banked ids are head content, the closer, then the continuation, with
    the head's stop id dropped; the banked finish reason is the head's.
    """
    content = _content_len(head.token_ids, family=family)
    ids = (
        tuple(int(token) for token in head.token_ids[:content])
        + family.think_close_token_ids
        + tuple(int(token) for token in tail.token_ids)
    )
    return ContinuedCompletion(
        request_id=head.request_id,
        prompt_token_ids=tuple(int(token) for token in head.prompt_token_ids),
        text=f"{head.text}{family.think_close}{tail.text}",
        n_tokens=len(ids),
        token_ids=ids,
        finish_reason=str(head.finish_reason),
        num_cached_tokens=head.num_cached_tokens,
        queued_ts=head.queued_ts,
        scheduled_ts=head.scheduled_ts,
        first_token_ts=head.first_token_ts,
    )


def continue_unclosed(
    engine: TextEngine,
    requests: Sequence[QwenDecodeRequest],
    results: Sequence[TextCompletion],
    *,
    qid: str,
    family: RouteFamily,
    decoder: Decoder,
    profile: BenchmarkProfile,
) -> tuple[tuple[TextCompletion, ...], tuple[bool, ...]]:
    """Continue each unclosed draw exactly once with the closer injected.

    The continuation is the same draw: every sampling field but the output cap
    is untouched, and the cap is the closing budget.
    """
    unclosed = tuple(
        position
        for position, result in enumerate(results)
        if family.thinking_is_unclosed(result.token_ids, decode=decoder)
        and _closer_is_affordable(result, family=family, profile=profile)
    )
    injected = tuple(position in set(unclosed) for position in range(len(results)))
    if not unclosed:
        return tuple(results), injected
    retry = tuple(
        replace(requests[position], decode=replace(requests[position].decode, closing=True))
        for position in unclosed
    )
    continuation_prompts = tuple(
        tuple(map(int, result.prompt_token_ids))
        + tuple(map(int, result.token_ids[: _content_len(result.token_ids, family=family)]))
        + family.think_close_token_ids
        for result in (results[position] for position in unclosed)
    )
    continued = ordered_results(
        retry,
        engine.decode_token_ids_full(continuation_prompts, retry),
    )
    validate_draw(qid, continued)
    merged = list(results)
    for position, tail in zip(unclosed, continued, strict=True):
        merged[position] = _merge_continuation(results[position], tail, family=family)
    return tuple(merged), injected


def generate_report_bundle(
    engine: TextEngine,
    prompts: Sequence[str],
    *,
    qid: str,
    semantic_arm: str,
    family: RouteFamily,
    decoder: Decoder,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
    prepared_item: ProbeItem | None = None,
) -> QwenReportBundle:
    """Decode three reports, advancing the seed ladder only while one is empty.

    `report_generation_s` is the span of the draw that shipped, the last one
    executed; the draws that did not ship bank as `draws_n` and `redraw_wall_s`.
    """
    profile = family.benchmark_profile(profile)
    if len(prompts) != FANOUT_M3.workers_per_item:
        raise ValueError("Qwen text generation needs exactly three worker prompts")
    if semantic_arm not in QWEN_TEXT_ARMS:
        raise ValueError("Qwen text generation requires one registered sender arm")
    native_sender = None
    if family.native_sender_prompts:
        from rcc.models.qwen.native_prompts import native_sender_artifact

        if prepared_item is None or prepared_item.qid != qid:
            raise RuntimeError(f"{qid}/{semantic_arm}: native generation requires its sealed item")
        native_sender = native_sender_artifact(
            prepared_item,
            semantic_arm=semantic_arm,
            family=family,
            prompts=prompts,
            profile=profile,
        )
    family = family.sender_family(semantic_arm)
    decode = QwenDecodeSpec("report", family, profile=profile)
    draw_elapsed: list[float] = []
    empty: list[int] = []
    results: Sequence[TextCompletion] = ()
    injected: tuple[bool, ...] = (False, False, False)
    seeds: tuple[int, ...] = ()
    reports: tuple[str, ...] = ()
    prompt_hashes: tuple[str, ...] = ()
    tag = QWEN_REPORT_SEED_TAGS[0]
    for tag in QWEN_REPORT_SEED_TAGS:
        seeds = profile.report_seeds(qid, tag)
        requests = tuple(
            QwenDecodeRequest(
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
        results = ordered_results(requests, engine.decode_text_full(prompts, requests))
        validate_draw(qid, results)
        if family.native_sender_prompts:
            from rcc.models.qwen.native_prompts import consumed_prompt_hashes

            assert native_sender is not None
            prompt_hashes = consumed_prompt_hashes(
                prompts, results, expected=native_sender["workers"], decoder=decoder
            )
        results, injected = continue_unclosed(
            engine,
            requests,
            results,
            qid=qid,
            family=family,
            decoder=decoder,
            profile=profile,
        )
        draw_elapsed.append(time.perf_counter() - started)
        reports = tuple(
            family.post_think_handoff(result.text, ended=str(result.finish_reason) == "stop")
            for result in results
        )
        empty = [worker for worker, report in enumerate(reports) if not report]
        # Draw exhaustion quarantines the cell rather than raising: the final
        # draw banks as evidence and the receiver serves the base manager
        # prompt.
        if not empty:
            break
    return QwenReportBundle(
        report_failed_workers=tuple(empty),
        reports=reports,
        raw_outputs=tuple(result.text for result in results),
        token_ids_by_worker=tuple(
            tuple(int(token) for token in result.token_ids) for result in results
        ),
        tokens_by_worker=tuple(int(result.n_tokens) for result in results),
        finish_reasons=tuple(str(result.finish_reason) for result in results),
        seeds=seeds,
        seed_tag=tag,
        thinking_closed=tuple(
            not decode.enable_thinking or family.think_close in result.text for result in results
        ),
        decode=decode,
        generation_s=draw_elapsed[-1],
        draws_n=len(draw_elapsed),
        redraw_wall_s=sum(draw_elapsed[:-1]),
        injected_by_worker=injected,
        queue_s_mean=mean_elapsed(results, later="scheduled_ts", earlier="queued_ts"),
        ttft_s_mean=mean_elapsed(results, later="first_token_ts", earlier="queued_ts"),
        prompt_token_sha256=prompt_hashes,
        prompt_tokens_by_worker=(
            tuple(len(result.prompt_token_ids) for result in results) if native_sender else ()
        ),
        prompt_unpadded_tokens_by_worker=(
            tuple(row["unpadded_prompt_tokens"] for row in native_sender["workers"])
            if native_sender
            else ()
        ),
        prompt_policy=str(native_sender["codec"]["prompt_policy"]) if native_sender else None,
    )


__all__ = (
    "QWEN_REPORT_SEED_TAGS",
    "QWEN_TEXT_ARMS",
    "QwenDecodeRequest",
    "QwenDecodeSpec",
    "QwenReportBundle",
    "QwenTextSender",
    "continue_unclosed",
    "format_report_payload",
    "generate_report_bundle",
    "mean_elapsed",
    "ordered_results",
    "registered_text_senders",
    "report_seed",
    "validate_draw",
    "validate_report_ladder_fields",
)
