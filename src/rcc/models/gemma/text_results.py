"""Gemma 4 text receiver and result rows for the resident lifecycle.

E4B and E2B run as sequential native-context vLLM phases before the resident
engine opens; ``text_primary`` borrows that already-live 12B engine.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.fanoutqa.data import score_text
from rcc.benchmarks.fanoutqa.scoring import DEFAULT_SCORER_VERSION
from rcc.models.gemma import GEMMA
from rcc.models.gemma.contract import (
    ANSWER_MAX_NEW_TOKENS,
    RECEIVER_ENABLE_THINKING,
    ROW_SCHEMA,
    STOP_IDS,
    WORKERS_PER_ITEM,
)
from rcc.models.gemma.mechanism import encode, token_embeds, visible_answer_tokens
from rcc.models.gemma.prompts import manager_prompt
from rcc.models.gemma.text_codec import (
    receiver_turn_ids as text_receiver_turn_ids,
)
from rcc.models.gemma.text_codec import (
    report_is_substantive,
)
from rcc.models.gemma.text_contract import (
    TEXT_REPORT_SCHEMA,
    TEXT_SENDERS,
    TextSenderSpec,
    text_sender_spec,
)
from rcc.models.gemma.text_runtime import shared_sampling_spec
from rcc.run.fleet.answer_closer import ANSWER_CLOSING_TOKEN_BUDGET, answer_hit_ceiling
from rcc.run.fleet.latency import (
    COMPUTE_TIMING_SCOPE,
    FLEET_DECODE_SCHEDULE,
    continuation_offset_fields,
)
from rcc.run.fleet.stream import StreamCompletion

SHARED_DECODE_PRESENCE_PENALTY = GEMMA.decode.presence_penalty
SHARED_DECODE_TEMPERATURE = GEMMA.decode.temperature
SHARED_DECODE_TOP_K = GEMMA.decode.top_k
SHARED_DECODE_TOP_P = GEMMA.decode.top_p
SHARED_SEED = FANOUTQA_NATURAL_DEV50.report_seed_base

TEXT_RECEIVER_SCHEMA = "gemma4-fanoutqa-text-receiver-v1"
SHARED_SAMPLING_PROFILE = GEMMA.decode.profile_id


def format_report_payload(reports: Sequence[str]) -> str:
    """Format the ordered three-worker text handoff."""
    return "\n\n".join(
        f"[worker {worker} report]\n{report.strip()}" for worker, report in enumerate(reports)
    )


def question_text(item: Mapping[str, Any]) -> str:
    """Return one resident item's registered question text."""
    question = item.get("question")
    value = getattr(question, "question", question)
    if not isinstance(value, str) or not value:
        raise RuntimeError("Gemma report item has no question text")
    return value


def require_registered_thinking_turn(
    tokenizer: Any, sender: TextSenderSpec
) -> tuple[list[int], list[int]]:
    """Verify the sender's exact thinking-on template geometry."""
    prefix, suffix = text_receiver_turn_ids(tokenizer)
    if prefix != sender.thinking_on_prefix or suffix != sender.thinking_on_suffix:
        raise RuntimeError("Gemma text sender differs from the registered thinking turn")
    return list(prefix), list(suffix)


@dataclass(frozen=True)
class PreparedTextReceiver:
    """One 12B receiver prompt carrying a visible three-worker report payload."""

    schema: str
    qid: str
    semantic_arm: str
    policy: str
    prompt: torch.Tensor
    prompt_token_ids: tuple[int, ...]
    payload: str
    base_prompt_tokens: int
    receiver_prompt_tokens: int
    handoff_prompt_tokens: int
    text_wire_bytes: int
    receiver_prepare_s: float
    report_bundle: Mapping[str, Any]


def prepare_text_receiver(
    tokenizer: Any,
    embedding_weight: torch.Tensor,
    item: Mapping[str, Any],
    report_bundle: Mapping[str, Any],
) -> PreparedTextReceiver:
    """Build the exact report-payload manager prompt for the resident receiver."""
    started = time.perf_counter()
    arm = str(report_bundle.get("semantic_arm") or "")
    sender = text_sender_spec(arm)
    raw_reports: object = report_bundle.get("reports")
    reports = cast(list[object], raw_reports) if isinstance(raw_reports, list) else []
    quarantined = bool(report_bundle.get("report_failed"))
    if (
        report_bundle.get("schema") != TEXT_REPORT_SCHEMA
        or report_bundle.get("worker_checkpoint") != sender.checkpoint
        or report_bundle.get("worker_revision") != sender.revision
        or not isinstance(raw_reports, list)
        or len(reports) != WORKERS_PER_ITEM
        or not all(isinstance(row, str) for row in reports)
        or (
            not quarantined
            and (
                not all(report_is_substantive(cast(str, row)) for row in reports)
                or report_bundle.get("report_thinking_closed") != [True] * WORKERS_PER_ITEM
            )
        )
    ):
        raise RuntimeError(f"{item.get('qid')}/{arm}: text report bundle is incomplete")
    text_reports = [cast(str, row) for row in reports]
    # A quarantined cell serves the base manager prompt: a partial payload is
    # a different arm, not a degraded one, so no payload crosses at all.
    payload = "" if quarantined else format_report_payload(text_reports)
    question = question_text(item)
    base = manager_prompt(tokenizer, question)
    rendered = base if quarantined else manager_prompt(tokenizer, question, payload)
    prefix, suffix = require_registered_thinking_turn(tokenizer, TEXT_SENDERS[0])
    base_ids = tuple(encode(tokenizer, base))
    request_ids = tuple(encode(tokenizer, rendered))
    if (
        not request_ids
        or request_ids[: len(prefix)] != tuple(prefix)
        or request_ids[-len(suffix) :] != tuple(suffix)
    ):
        raise RuntimeError(f"{item.get('qid')}/{arm}: manager prompt is not thinking-on")
    if len(request_ids) + ANSWER_MAX_NEW_TOKENS > TEXT_SENDERS[0].native_max_model_len:
        raise RuntimeError(f"{item.get('qid')}/{arm}: receiver exceeds native 12B context")
    if not quarantined and len(request_ids) <= len(base_ids):
        raise RuntimeError(f"{item.get('qid')}/{arm}: report payload added no manager rows")
    prompt = token_embeds(
        embedding_weight,
        request_ids,
        dtype=torch.bfloat16,
    ).cpu()
    return PreparedTextReceiver(
        schema=TEXT_RECEIVER_SCHEMA,
        qid=str(item.get("qid") or ""),
        semantic_arm=arm,
        policy=sender.policy,
        prompt=prompt,
        prompt_token_ids=request_ids,
        payload=payload,
        base_prompt_tokens=len(base_ids),
        receiver_prompt_tokens=len(request_ids),
        handoff_prompt_tokens=len(request_ids) - len(base_ids),
        text_wire_bytes=len(payload.encode()),
        receiver_prepare_s=round(time.perf_counter() - started, 6),
        report_bundle=dict(report_bundle),
    )


def text_result_fields(prepared: PreparedTextReceiver) -> dict[str, Any]:
    """Map a text payload onto the shared result-row vocabulary."""
    bundle = prepared.report_bundle
    sampling = shared_sampling_spec()
    return {
        "channel": "text",
        "policy": prepared.policy,
        "worker_checkpoint": bundle["worker_checkpoint"],
        "worker_revision": bundle["worker_revision"],
        "receiver_checkpoint": TEXT_SENDERS[0].checkpoint,
        "receiver_revision": TEXT_SENDERS[0].revision,
        "receiver_format": "manager-prompt-with-worker-reports",
        "channel_block_kind": ("none" if bundle.get("report_failed") else "text_payload"),
        "base_prompt_tokens": prepared.base_prompt_tokens,
        "receiver_prompt_tokens": prepared.receiver_prompt_tokens,
        "handoff_prompt_tokens": prepared.handoff_prompt_tokens,
        "latent_tokens": 0,
        "text_wire_bytes": prepared.text_wire_bytes,
        "latent_wire_bytes": 0,
        "total_wire_bytes": prepared.text_wire_bytes,
        "reports": list(bundle["reports"]),
        "report_raw_outputs": list(bundle["report_raw_outputs"]),
        "report_tokens": int(bundle["report_tokens"]),
        "report_tokens_by_worker": list(bundle["report_tokens_by_worker"]),
        "report_token_ids_by_worker": list(bundle["report_token_ids_by_worker"]),
        "report_finish_reasons": list(bundle["report_finish_reasons"]),
        "report_seed_tag": bundle["report_seed_tag"],
        "report_seeds": list(bundle["report_seeds"]),
        "report_thinking_closed": list(bundle["report_thinking_closed"]),
        "report_thinking_closed_rate": float(bundle["report_thinking_closed_rate"]),
        "report_generation_s": float(bundle["report_generation_s"]),
        # The span above is the shipped draw only; these three record what it
        # excludes and whether the closer was injected.
        "draws_n": int(bundle["draws_n"]),
        "redraw_wall_s": float(bundle["redraw_wall_s"]),
        "injected": bool(bundle["injected"]),
        "report_injected_by_worker": list(cast(list[bool], bundle["report_injected_by_worker"])),
        "report_failed": bool(bundle.get("report_failed", False)),
        "report_failed_workers": list(cast(list[int], bundle.get("report_failed_workers", []))),
        "report_failure": bundle.get("report_failure"),
        "selected_indices_sha256": None,
        "selected_indices_order": None,
        "qwen_flat_payload_identity": None,
        "cut_decoded_sha256": None,
        "report_schema": TEXT_REPORT_SCHEMA,
        "receiver_prepare_s": prepared.receiver_prepare_s,
        "sampling_profile": sampling["profile"],
        "sampling_fingerprint": sampling["sampling_fingerprint"],
        "sampling_seed": sampling["seed"],
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "top_k": sampling["top_k"],
        "decode_temperature": sampling["temperature"],
        "decode_top_p": sampling["top_p"],
        "decode_top_k": sampling["top_k"],
        "decode_presence_penalty": sampling["presence_penalty"],
        "decode_repeats": len(sampling["sample_tags"]),
        "answer_max_tokens": sampling["answer_ceiling"],
        "answer_closer_budget": ANSWER_CLOSING_TOKEN_BUDGET,
        "decode_tags": list(sampling["sample_tags"]),
    }


def build_text_result_rows(
    *,
    tokenizer: Any,
    item: Mapping[str, Any],
    arm: str,
    prepared: PreparedTextReceiver,
    seeds: tuple[int, ...],
    completions: Sequence[StreamCompletion],
    timing_origin: float,
    channel_open: int,
    channel_close: int,
    engine_identity_sha256: str,
) -> list[dict[str, Any]]:
    """Build the three per-seed result rows for one text payload."""
    if arm != prepared.semantic_arm or len(seeds) != 3 or len(completions) != 3:
        raise RuntimeError(f"{item.get('qid')}/{arm}: text decode roster is invalid")
    ordered = sorted(completions, key=lambda row: int(row.tag[1:]))
    if [row.tag for row in ordered] != ["s0", "s1", "s2"]:
        raise RuntimeError(f"{item.get('qid')}/{arm}: text completion order drifted")
    for row in ordered:
        if (
            row.finish_reason not in {"stop", "length"}
            or row.n_tokens != len(row.token_ids)
            or row.num_cached_tokens not in {None, 0}
        ):
            raise RuntimeError(f"{item.get('qid')}/{arm}: text completion is invalid")
    token_rows = [list(row.token_ids) for row in ordered]
    sampled = [tokenizer.decode(tokens) for tokens in token_rows]
    visible_ids = [
        visible_answer_tokens(
            tokens,
            channel_open=channel_open,
            channel_close=channel_close,
            stop_ids=STOP_IDS,
        )
        for tokens in token_rows
    ]
    answers = [tokenizer.decode(tokens) for tokens in visible_ids]
    task_fields = [score_text(item["question"], answer) for answer in answers]
    first = [row.first_token_at or row.finished_at for row in ordered]
    if any(row.n_tokens and row.first_token_at is None for row in ordered):
        raise RuntimeError(f"{item.get('qid')}/{arm}: text completion has no TTFT clock")
    queued = [row.submitted_at - timing_origin for row in ordered]
    first_offsets = [value - timing_origin for value in first]
    finished = [row.finished_at - timing_origin for row in ordered]
    if queued[0] != 0.0 or any(value < finished[0] for value in queued[1:]):
        raise RuntimeError(f"{item.get('qid')}/{arm}: text s1/s2 preceded s0 completion")
    timed = ordered[0]
    decode_s = timed.finished_at - timed.submitted_at
    decode_batched_s = max(row.finished_at for row in ordered[1:]) - min(
        row.submitted_at for row in ordered[1:]
    )
    receiver_ttft_s = first[0] - timed.submitted_at
    generation_s = timed.finished_at - first[0]
    fields = text_result_fields(prepared)
    report_s = float(fields["report_generation_s"])
    common: dict[str, Any] = {
        "kind": "result",
        "schema": ROW_SCHEMA,
        "scoring_version": DEFAULT_SCORER_VERSION,
        "qid": str(item.get("qid") or ""),
        "question": getattr(item.get("question"), "question", item.get("question")),
        "arm": arm,
        "semantic_arm": arm,
        "construction_complete": bool(item["construction_complete"]),
        "construction_incomplete_reason": str(item["construction_incomplete_reason"]),
        "dead_pages": list(item["dead_pages"]),
        "answerability_group": (
            "construction_complete" if item["construction_complete"] else "construction_incomplete"
        ),
        "mode": "fanoutqa-m3",
        "panel": "production",
        "workers": WORKERS_PER_ITEM,
        "receiver_backend": "vllm",
        "decode_profile": GEMMA.decode.profile_id,
        "decode_fingerprint": GEMMA.decode.identity_hash,
        "receiver_enable_thinking": RECEIVER_ENABLE_THINKING,
        "enable_thinking": RECEIVER_ENABLE_THINKING,
        "receiver_rows": int(prepared.prompt.shape[0]),
        "decode_path": "vllm-public-enqueue-prompt-embeds-stream",
        "decode_order_index": 0,
        "decode_timing_sample": "s0",
        # The text lane bills what the latent lane bills: report generation,
        # receiver prepare, and the one timed decode. The scope on the row is
        # what tells a reader that from a fleet-scope number.
        "timing_scope": COMPUTE_TIMING_SCOPE,
        "fleet_decode_schedule": FLEET_DECODE_SCHEDULE,
        "sample_seeds": list(seeds),
        "generated_token_ids_by_sample": token_rows,
        "answers": sampled,
        "answer_texts": answers,
        "sample_task_fields": task_fields,
        "generated_tokens_by_sample": [len(tokens) for tokens in token_rows],
        "first_token_emitted_by_sample": [bool(tokens) for tokens in token_rows],
        # Read on the head's finish reason, never the merged length: an
        # injected row is longer than the head the sampler capped.
        "answer_hit_ceiling_by_sample": [answer_hit_ceiling(row.finish_reason) for row in ordered],
        "thinking_closed_by_sample": [
            channel_close in tokens or channel_open not in tokens for tokens in token_rows
        ],
        "finish_reasons": [row.finish_reason for row in ordered],
        "answer_injected_by_sample": [row.answer_injected for row in ordered],
        "answer_head_finish_reasons": [row.finish_reason for row in ordered],
        "num_cached_tokens_by_sample": [row.num_cached_tokens for row in ordered],
        "decode_queued_offset_s_by_sample": [round(value, 4) for value in queued],
        "decode_first_token_offset_s_by_sample": [round(value, 4) for value in first_offsets],
        "decode_finished_offset_s_by_sample": [round(value, 4) for value in finished],
        **continuation_offset_fields(ordered, timing_origin),
        "answer": sampled[0],
        "finish_reason": ordered[0].finish_reason,
        "first_token_emitted": bool(token_rows[0]),
        "generated_tokens": round(sum(map(len, token_rows)) / 3, 4),
        "thinking_closed_rate": round(
            sum(channel_close in tokens or channel_open not in tokens for tokens in token_rows) / 3,
            4,
        ),
        "length_finish_rate": round(sum(row.finish_reason == "length" for row in ordered) / 3, 4),
        "producer_s": report_s,
        # The same three columns the latent row carries, with the same meaning:
        # this lane writes nothing to disk between phases, and the streaming
        # path admits through the driver, so vLLM reports no scheduling stamp.
        "spill_load_s": 0.0,
        "spill_save_s": 0.0,
        "decode_s": round(decode_s, 4),
        "decode_batched_s": round(decode_batched_s, 4),
        "queue_s": None,
        "receiver_ttft_s": round(receiver_ttft_s, 4),
        "generation_s": round(generation_s, 4),
        "ttft_s": round(report_s + prepared.receiver_prepare_s + receiver_ttft_s, 4),
        "tteoa_s": round(report_s + prepared.receiver_prepare_s + decode_s, 4),
        "engine_identity_sha256": engine_identity_sha256,
        **fields,
    }
    from rcc.models.gemma.results import row_for_seed

    return [row_for_seed(common, index) for index in range(3)]
