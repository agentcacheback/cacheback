"""Ministral result-row construction and independent raw-token rescore."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from rcc.benchmarks.fanoutqa import (
    FANOUTQA_NATURAL_DEV50,
    SEALED_M3_PLAN_LABEL,
    shared_answer_seeds,
)
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.ministral import MINISTRAL
from rcc.models.ministral.receiver import (
    MinistralReceiverPrompt,
    MinistralReceiverRequest,
    MinistralVisibleAnswer,
)
from rcc.models.ministral.text import (
    MinistralDecodeSpec,
    MinistralReportBundle,
    reconstruct_raw_report,
    registered_text_senders,
)
from rcc.models.ministral.text_codec import (
    MINISTRAL_STOP_TOKEN_IDS,
    MINISTRAL_TEXT_ARMS,
    MINISTRAL_THINK_CLOSE_ID,
    MinistralTextCodec,
)
from rcc.models.ministral.text_decode import MINISTRAL_CLOSING_TOKEN_BUDGET
from rcc.run.fleet.answer_closer import (
    ANSWER_CLOSING_TOKEN_BUDGET,
    answer_hit_ceiling,
    validate_banked_answer,
)
from rcc.run.fleet.latency import ITEM_LATENCY_FIELDS, ItemLatency
from rcc.run.rescore import content_token_count, validate_report_ceiling

# The shared per-item latency vocabulary every family banks, plus the ladder
# and injection columns. A row of another schema is refused rather than
# back-filled: defaulting would relabel a three-draw cell as a single-draw one.
MINISTRAL_RESULT_SCHEMA = "ministral-fanoutqa-m3-fleet-row-v3"

Score = Callable[[str], Mapping[str, float]]


def _raw_tokens(label: str, value: object) -> tuple[int, ...]:
    """Require a nonnegative integer token vector without bool coercion."""
    if not isinstance(value, list):
        raise RuntimeError(f"{label}: sampled token ids are malformed")
    tokens = cast(list[object], value)
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens):
        raise RuntimeError(f"{label}: sampled token ids are malformed")
    return tuple(cast(list[int], tokens))


def _score_fields(score: Mapping[str, float], *, label: str) -> dict[str, float | bool]:
    """Validate and normalize the registered loose/strict score vocabulary."""
    if set(score) != {"loose", "strict", "n_leaves"}:
        raise RuntimeError(f"{label}: Ministral scorer returned an unknown field roster")
    loose = float(score["loose"])
    strict = float(score["strict"])
    leaves = float(score["n_leaves"])
    if (
        not all(math.isfinite(value) for value in (loose, strict, leaves))
        or not 0.0 <= loose <= 1.0
        or strict not in {0.0, 1.0}
        or leaves < 0.0
    ):
        raise RuntimeError(f"{label}: Ministral scorer returned invalid values")
    return {
        "loose": loose,
        "strict": strict,
        "n_leaves": leaves,
        "loose_match": loose > 0.0,
    }


def _payload_fields(prompt: MinistralReceiverPrompt) -> dict[str, object]:
    """Return signed flat payload identity, or explicit nulls for non-latent arms."""
    payload = prompt.payload
    return {
        "payload_layout": payload.layout if payload is not None else None,
        "latent_plan_sha256": payload.latent_plan_sha256 if payload is not None else None,
        "selected_indices_sha256": (
            payload.selected_indices_sha256 if payload is not None else None
        ),
        "selected_indices_order": (payload.selected_indices_order if payload is not None else None),
        "payload_sha256": payload.tensor_sha256 if payload is not None else None,
        "payload_rows_by_worker": list(payload.rows_by_worker) if payload is not None else None,
        "payload_worker_sha256": list(payload.worker_sha256) if payload is not None else None,
    }


def _execution_fields(identity: Mapping[str, Any]) -> dict[str, object]:
    """Require the resolved plan and runtime identities every row must carry."""
    plan = identity.get("unified_plan_fingerprint")
    runtime_fingerprint = identity.get("runtime_fingerprint")
    runtime_signature = identity.get("runtime_signature")
    if (
        not isinstance(plan, str)
        or len(plan) != 64
        or not isinstance(runtime_fingerprint, str)
        or len(runtime_fingerprint) != 64
        or not isinstance(runtime_signature, Mapping)
        or not runtime_signature
    ):
        raise ValueError("Ministral result rows require signed plan and runtime identity")
    signature = cast(Mapping[str, Any], runtime_signature)
    return {
        "unified_plan_fingerprint": plan,
        "runtime_fingerprint": runtime_fingerprint,
        "runtime_signature": dict(signature),
    }


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label}: Ministral report evidence is malformed")
    values = cast(list[object], value)
    if any(not isinstance(item, str) for item in values):
        raise RuntimeError(f"{label}: Ministral report evidence is malformed")
    return tuple(cast(list[str], values))


def _int_tuple(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label}: Ministral report evidence is malformed")
    values = cast(list[object], value)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in values):
        raise RuntimeError(f"{label}: Ministral report evidence is malformed")
    return tuple(cast(list[int], values))


def _bool_tuple(value: object, label: str) -> tuple[bool, ...]:
    """Require a real per-worker boolean roster, never a truthy integer."""
    if not isinstance(value, list):
        raise RuntimeError(f"{label}: Ministral report injection evidence is malformed")
    values = cast(list[object], value)
    if any(type(item) is not bool for item in values):
        raise RuntimeError(f"{label}: Ministral report injection evidence is malformed")
    return tuple(cast(list[bool], values))


def _token_rows(value: object, label: str) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label}: Ministral report token evidence is malformed")
    output: list[tuple[int, ...]] = []
    for row in cast(list[object], value):
        if not isinstance(row, list):
            raise RuntimeError(f"{label}: Ministral report token evidence is malformed")
        tokens = cast(list[object], row)
        if any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens
        ):
            raise RuntimeError(f"{label}: Ministral report token evidence is malformed")
        output.append(tuple(cast(list[int], tokens)))
    return tuple(output)


def _report_codec(codec: MinistralTextCodec, semantic_arm: str) -> None:
    codec.require_verified()
    sender = next(row for row in registered_text_senders() if row.semantic_arm == semantic_arm)
    if (codec.checkpoint, codec.revision) != (sender.checkpoint, sender.revision):
        raise RuntimeError("Ministral report rescore requires its sender-native codec")


def _validated_report_fields(
    bundle: MinistralReportBundle,
    codec: MinistralTextCodec,
) -> dict[str, object]:
    _report_codec(codec, bundle.semantic_arm)
    # Stop-trimmed content, exactly the prefix `reconstruct_raw_report` decodes:
    # the banked raw vector carries the stop token on a stop finish.
    validate_report_ceiling(
        [
            content_token_count(raw, frozenset(MINISTRAL_STOP_TOKEN_IDS))
            for raw in bundle.raw_token_ids_by_worker
        ],
        bundle.finish_reasons,
        ceiling=FANOUTQA_NATURAL_DEV50.report_ceiling,
        family="Ministral",
        label=f"{bundle.qid}/{bundle.semantic_arm}",
        injected=list(bundle.injected_by_worker),
        # One closer id plus the closing budget.
        closing_extra=1 + MINISTRAL_CLOSING_TOKEN_BUDGET,
    )
    raw_outputs: list[str] = []
    visible_rows: list[tuple[int, ...]] = []
    reports: list[str] = []
    for worker, raw in enumerate(bundle.raw_token_ids_by_worker):
        raw_outputs.append(reconstruct_raw_report(codec, raw))
        if worker in bundle.report_failed_workers:
            visible_rows.append(())
            reports.append("")
            continue
        visible, report = codec.visible_report(raw)
        visible_rows.append(visible)
        reports.append(report)
    if (
        bundle.raw_outputs != tuple(raw_outputs)
        or bundle.visible_token_ids_by_worker != tuple(visible_rows)
        or bundle.reports != tuple(reports)
    ):
        raise RuntimeError("Ministral report fields differ from sender-native raw-token rescore")
    return bundle.result_fields()


def _report_bundle_from_row(
    row: Mapping[str, Any],
    *,
    qid: str,
    semantic_arm: str,
    profile: BenchmarkProfile,
) -> MinistralReportBundle:
    label = f"{qid}/{semantic_arm}"
    seed_tag = row.get("report_seed_tag")
    prepared = row.get("prepared_prompt_fingerprint")
    if not isinstance(seed_tag, str) or not isinstance(prepared, str):
        raise RuntimeError(f"{label}: Ministral report identity is malformed")
    draws_n = row.get("draws_n")
    redraw_wall_s = row.get("redraw_wall_s")
    # The ladder and injection columns are required, never defaulted: the ids alone
    # cannot re-derive them. What is checked is that the two injection columns agree
    # with each other and with the banked ids.
    if (
        isinstance(draws_n, bool)
        or not isinstance(draws_n, int)
        or isinstance(redraw_wall_s, bool)
        or not isinstance(redraw_wall_s, (int, float))
    ):
        raise RuntimeError(f"{label}: Ministral report draw-ladder evidence is malformed")
    return MinistralReportBundle(
        qid=qid,
        semantic_arm=semantic_arm,
        reports=_string_tuple(row.get("reports"), label),
        raw_outputs=_string_tuple(row.get("report_raw_outputs"), label),
        raw_token_ids_by_worker=_token_rows(row.get("report_token_ids_by_worker"), label),
        visible_token_ids_by_worker=_token_rows(
            row.get("report_visible_token_ids_by_worker"), label
        ),
        finish_reasons=_string_tuple(row.get("report_finish_reasons"), label),
        seeds=_int_tuple(row.get("report_seeds"), label),
        seed_tag=seed_tag,
        prompt_sha256=_string_tuple(row.get("report_prompt_sha256"), label),
        prepared_prompt_fingerprint=prepared,
        decode=MinistralDecodeSpec(),
        generation_s=float(row.get("report_generation_s", -1.0)),
        queue_s_mean=cast(float | None, row.get("report_queue_s_mean")),
        ttft_s_mean=cast(float | None, row.get("report_ttft_s_mean")),
        profile=profile,
        report_failed_workers=_int_tuple(row.get("report_failed_workers", []), label),
        draws_n=int(draws_n),
        redraw_wall_s=float(redraw_wall_s),
        injected_by_worker=_bool_tuple(row.get("report_injected_by_worker"), label),
    )


def rescore_report_row(
    row: Mapping[str, Any],
    codec: MinistralTextCodec,
    *,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> None:
    """Reconstruct one text row's three worker reports with its sender codec."""
    qid, semantic_arm = str(row.get("qid") or ""), str(row.get("arm") or "")
    bundle = _report_bundle_from_row(
        row,
        qid=qid,
        semantic_arm=semantic_arm,
        profile=profile,
    )
    expected = _validated_report_fields(bundle, codec)
    drifted = [name for name, value in expected.items() if row.get(name) != value]
    if drifted:
        raise RuntimeError(f"{qid}/{semantic_arm}: report rescore differs in {drifted}")


def build_result_rows(
    requests: Sequence[MinistralReceiverRequest],
    answers: Sequence[MinistralVisibleAnswer],
    *,
    score: Score,
    execution_identity: Mapping[str, Any],
    latency: ItemLatency,
    report_bundle: MinistralReportBundle | None = None,
    report_codec: MinistralTextCodec | None = None,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> tuple[dict[str, object], ...]:
    """Build three independent rows; never vote or collapse the cell."""
    request_roster = tuple(requests)
    answer_roster = tuple(answers)
    if len(request_roster) != 3 or len(answer_roster) != 3:
        raise RuntimeError("Ministral result cell requires exactly three samples")
    if (
        len({request.qid for request in request_roster}) != 1
        or len({request.semantic_arm for request in request_roster}) != 1
    ):
        raise RuntimeError("Ministral result cell mixes item or arm identity")
    arm_by_name = {arm.semantic_arm: arm for arm in MINISTRAL.physical_arms}
    arm = arm_by_name[request_roster[0].semantic_arm]
    is_text = arm.semantic_arm in MINISTRAL_TEXT_ARMS
    if is_text != (report_bundle is not None and report_codec is not None):
        raise ValueError("Ministral text rows require exactly one report bundle and sender codec")
    report_fields: dict[str, object] = {}
    if report_bundle is not None and report_codec is not None:
        if (report_bundle.qid, report_bundle.semantic_arm) != (
            request_roster[0].qid,
            arm.semantic_arm,
        ):
            raise ValueError("Ministral report bundle differs from its receiver cell")
        report_fields = _validated_report_fields(report_bundle, report_codec)
    if len(latency.queued_offsets) != len(request_roster):
        raise ValueError("Ministral latency offsets differ from the three-sample roster")
    execution_fields = _execution_fields(execution_identity)
    latency_fields = latency.result_fields()
    rows: list[dict[str, object]] = []
    for request, answer in zip(request_roster, answer_roster, strict=True):
        label = f"{request.qid}/{request.semantic_arm}/{request.sample_tag}"
        if (
            answer.sample_tag,
            answer.sample_index,
            answer.seed,
        ) != (
            request.sample_tag,
            request.sample_index,
            request.seed,
        ):
            raise RuntimeError(f"{label}: reconstructed answer identity differs")
        scored = _score_fields(score(answer.visible_text), label=label)
        row: dict[str, object] = {
            "schema": MINISTRAL_RESULT_SCHEMA,
            "benchmark_profile": profile.profile_id,
            "source_identity": profile.source_logical_fingerprint,
            "model_id": MINISTRAL.model_id,
            "policy_family": "ministral3_14b",
            "qid": request.qid,
            "arm": request.semantic_arm,
            "semantic_arm": request.semantic_arm,
            "policy": arm.policy,
            "sample_tag": request.sample_tag,
            "seed_index": request.sample_index,
            "seed": request.seed,
            "worker_checkpoint": arm.sender_checkpoint,
            "worker_revision": arm.sender_revision,
            "receiver_checkpoint": MINISTRAL.checkpoint,
            "receiver_revision": MINISTRAL.revision,
            "sampled_tokens": list(answer.token_ids),
            "sampled_text": answer.raw_text,
            "backend_text": answer.backend_text,
            "visible_token_ids": list(answer.visible_token_ids),
            "visible_answer_text": answer.visible_text,
            "answer": answer.visible_text,
            "answer_decoded_tokens": len(answer.token_ids),
            "thinking_closed": answer.thinking_closed,
            "finish_reason": answer.finish_reason,
            # The banked reason is the head's on an injected draw, so it is
            # named apart from the merged evidence the reader reconstructs.
            "answer_injected": answer.answer_injected,
            "answer_injected_by_sample": [row.answer_injected for row in answer_roster],
            "answer_head_finish_reasons": [row.finish_reason for row in answer_roster],
            "answer_closer_budget": ANSWER_CLOSING_TOKEN_BUDGET,
            "num_cached_tokens": answer.num_cached_tokens,
            # The head's reason, never the merged length: an injected row is
            # longer than the head the sampler capped.
            "answer_hit_ceiling": answer_hit_ceiling(answer.finish_reason),
            "decode": request.decode.to_dict(),
            "decode_profile": MINISTRAL.decode.profile_id,
            "decode_fingerprint": MINISTRAL.decode.identity_hash,
            "temperature": MINISTRAL.decode.temperature,
            "top_p": MINISTRAL.decode.top_p,
            "top_k": MINISTRAL.decode.top_k,
            "presence_penalty": MINISTRAL.decode.presence_penalty,
            "enable_thinking": MINISTRAL.decode.enable_thinking,
            **execution_fields,
            **_payload_fields(request.prompt),
            **latency_fields,
            **scored,
        }
        collisions = [
            name for name, value in report_fields.items() if name in row and row[name] != value
        ]
        if collisions:
            raise RuntimeError(f"{label}: report identity differs in {collisions}")
        row.update(report_fields)
        rows.append(row)
    return tuple(rows)


def _rescore_report_evidence(
    roster: Sequence[Mapping[str, Any]],
    semantic_arm: str,
    report_codec: MinistralTextCodec | None,
    profile: BenchmarkProfile,
) -> None:
    if semantic_arm in MINISTRAL_TEXT_ARMS:
        if report_codec is None:
            raise RuntimeError("Ministral text rescore requires its sender-native report codec")
        for row in roster:
            rescore_report_row(row, report_codec, profile=profile)
    elif report_codec is not None:
        raise RuntimeError("Ministral non-text rescore cannot accept a report codec")


def _require_rescore_execution_identity(roster: Sequence[Mapping[str, Any]]) -> None:
    execution_identities = {
        (
            str(row.get("unified_plan_fingerprint") or ""),
            str(row.get("runtime_fingerprint") or ""),
            repr(row.get("runtime_signature")),
        )
        for row in roster
    }
    if len(execution_identities) != 1:
        raise RuntimeError("Ministral rescore cell mixes plan or runtime identity")
    _execution_fields(roster[0])


def _rescore_result_row(
    row: Mapping[str, Any],
    *,
    index: int,
    qid: str,
    semantic_arm: str,
    expected_seed: int,
    expected_decode: Mapping[str, object],
    codec: MinistralTextCodec,
    score: Score,
) -> None:
    label = f"{qid}/{semantic_arm}/s{index}"
    # The schema is checked on its own and the refusal names it: another
    # schema version differs in what its timing fields mean, which folded into
    # the grid check below would read as an anonymous identity drift.
    observed_schema = row.get("schema")
    if observed_schema != MINISTRAL_RESULT_SCHEMA:
        raise RuntimeError(
            f"{label}: result row schema is {observed_schema!r}, "
            f"expected {MINISTRAL_RESULT_SCHEMA!r}"
        )
    if (
        row.get("seed_index") != index
        or row.get("sample_tag") != FANOUTQA_NATURAL_DEV50.sample_tags[index]
        or row.get("seed") != expected_seed
    ):
        raise RuntimeError(f"{label}: result identity differs from the sealed grid")
    missing = [name for name in ITEM_LATENCY_FIELDS if name not in row]
    if missing:
        raise RuntimeError(f"{label}: result row omits banked latency fields {missing}")
    raw_tokens = _raw_tokens(label, row.get("sampled_tokens"))
    injected = row.get("answer_injected")
    budget = row.get("answer_closer_budget")
    if type(injected) is not bool or isinstance(budget, bool) or not isinstance(budget, int):
        raise RuntimeError(f"{label}: answer closer-injection evidence is missing or malformed")
    finish_reason = row.get("finish_reason")
    head_reasons = row.get("answer_head_finish_reasons")
    if (
        not isinstance(finish_reason, str)
        or not isinstance(head_reasons, list)
        or len(cast(list[object], head_reasons)) != len(FANOUTQA_NATURAL_DEV50.sample_tags)
        or cast(list[object], head_reasons)[index] != finish_reason
    ):
        raise RuntimeError(f"{label}: banked answer head finish reason differs from the row")
    validate_banked_answer(
        raw_tokens,
        injected=injected,
        finish_reason=finish_reason,
        closing_budget=budget,
        answer_ceiling=FANOUTQA_NATURAL_DEV50.answer_ceiling,
        label=label,
        closer_ids=(MINISTRAL_THINK_CLOSE_ID,),
    )
    # The codec reads the merged ids, so an injected draw round-trips here as a
    # natural one does: closure, the visible span, and the score are all
    # recomputed from the banked ids rather than read off the row.
    visible_ids, visible_text, thinking_closed = codec.visible_answer(
        raw_tokens, ended=finish_reason == "stop"
    )
    stop_index = next(
        (position for position, token in enumerate(raw_tokens) if token == 2),
        len(raw_tokens),
    )
    raw_text = str(codec.tokenizer.decode(list(raw_tokens[:stop_index])))
    expected_fields: dict[str, object] = {
        "sampled_text": raw_text,
        "visible_token_ids": list(visible_ids),
        "visible_answer_text": visible_text,
        "answer": visible_text,
        "answer_decoded_tokens": len(raw_tokens),
        "thinking_closed": thinking_closed,
        "answer_hit_ceiling": answer_hit_ceiling(finish_reason),
        "decode_profile": MINISTRAL.decode.profile_id,
        "decode_fingerprint": MINISTRAL.decode.identity_hash,
        "temperature": MINISTRAL.decode.temperature,
        "top_p": MINISTRAL.decode.top_p,
        "top_k": MINISTRAL.decode.top_k,
        "presence_penalty": MINISTRAL.decode.presence_penalty,
        "enable_thinking": MINISTRAL.decode.enable_thinking,
        "decode": dict(expected_decode),
        **_score_fields(score(visible_text), label=label),
    }
    if row.get("finish_reason") not in {"stop", "length"} or row.get("num_cached_tokens") not in (
        None,
        0,
    ):
        raise RuntimeError(f"{label}: receiver termination or cache evidence differs")
    drifted = [name for name, value in expected_fields.items() if row.get(name) != value]
    if drifted:
        raise RuntimeError(f"{label}: independent rescore differs in {drifted}")


def rescore_result_rows(
    rows: Sequence[Mapping[str, Any]],
    codec: MinistralTextCodec,
    *,
    score: Score,
    report_codec: MinistralTextCodec | None = None,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> dict[str, int]:
    """Independently reconstruct and rescore one complete item-arm cell."""
    codec.require_verified()
    if (codec.checkpoint, codec.revision) != (MINISTRAL.checkpoint, MINISTRAL.revision):
        raise RuntimeError("Ministral rescore requires the registered 14B receiver codec")
    roster = tuple(rows)
    if len(roster) != len(FANOUTQA_NATURAL_DEV50.sample_tags):
        raise RuntimeError("Ministral rescore requires one complete three-sample cell")
    identities = {(str(row.get("qid")), str(row.get("arm"))) for row in roster}
    if len(identities) != 1:
        raise RuntimeError("Ministral rescore cell mixes item or arm identity")
    qid, semantic_arm = next(iter(identities))
    _rescore_report_evidence(roster, semantic_arm, report_codec, profile)
    expected_seeds = shared_answer_seeds(profile, qid, semantic_arm)
    _require_rescore_execution_identity(roster)
    expected_decode = {
        "purpose": "answer",
        "decode_profile": MINISTRAL.decode.profile_id,
        "decode_fingerprint": MINISTRAL.decode.identity_hash,
        "benchmark_profile": SEALED_M3_PLAN_LABEL,
        "enable_thinking": MINISTRAL.decode.enable_thinking,
        "temperature": MINISTRAL.decode.temperature,
        "top_p": MINISTRAL.decode.top_p,
        "presence_penalty": MINISTRAL.decode.presence_penalty,
        "max_tokens": FANOUTQA_NATURAL_DEV50.answer_ceiling,
        "stop_token_ids": [2],
    }
    for index, row in enumerate(roster):
        _rescore_result_row(
            row,
            index=index,
            qid=qid,
            semantic_arm=semantic_arm,
            expected_seed=expected_seeds[index],
            expected_decode=expected_decode,
            codec=codec,
            score=score,
        )
    return {"rescored_rows": len(roster), "items": 1, "arms": 1, "seeds": len(roster)}


__all__ = (
    "MINISTRAL_RESULT_SCHEMA",
    "build_result_rows",
    "rescore_report_row",
    "rescore_result_rows",
)
