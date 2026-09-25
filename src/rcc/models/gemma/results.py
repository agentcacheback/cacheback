"""Result construction for the all-thinking Gemma FanOutQA fleet lane."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, cast

import torch

from rcc.benchmarks.fanoutqa.data import evidence_answerability_audit, score_text
from rcc.benchmarks.fanoutqa.scoring import DEFAULT_SCORER_VERSION
from rcc.models.gemma import GEMMA
from rcc.models.gemma.capture_runtime import CaptureArtifact
from rcc.models.gemma.contract import (
    CHECKPOINT_ID,
    EMBEDDING_SHAPE,
    FLEET_RUNTIME,
    LATENT_STEPS,
    N_SEEDS,
    PRODUCER_BACKEND,
    RECEIVER_ENABLE_THINKING,
    RECEIVER_SLIDING_WINDOW,
    REGISTERED_ARM_POLICIES,
    REGISTERED_ARM_RATIOS,
    REGISTERED_ARM_SELECTORS,
    REGISTERED_CUT_ARMS,
    ROW_SCHEMA,
    SELECTOR_FAMILY,
    SPAN_WIDTH,
    STOP_IDS,
    SUPPORT_COMPOSITION,
    SUPPRESSED_TOKEN_IDS,
    WORKERS_PER_ITEM,
    sampling_contract,
)
from rcc.models.gemma.layout import (
    PAYLOAD_LAYOUT_BY_BLOCK_KIND as CUT_PAYLOAD_LAYOUT_BY_BLOCK_KIND,
)
from rcc.models.gemma.layout import (
    SOURCE_PAYLOAD_LAYOUT,
)
from rcc.models.gemma.mechanism import visible_answer_tokens
from rcc.run.fleet.answer_closer import ANSWER_CLOSING_TOKEN_BUDGET, answer_hit_ceiling
from rcc.run.fleet.latency import (
    COMPUTE_TIMING_SCOPE,
    FLEET_DECODE_SCHEDULE,
    continuation_offset_fields,
)
from rcc.run.fleet.stream import StreamCompletion
from rcc.run.gemma.cell_banks import decode_cell
from rcc.transforms.select.query_support.methods.compose import SUPPORT_DEFAULT_ALPHA
from rcc.transforms.select.query_support.methods.variants import VARIANT_FOLD_RULE


@dataclass(frozen=True)
class PreparedArm:
    """One normalized prompt plus separately auditable preparation clocks."""

    prompt: torch.Tensor
    keeps: tuple[tuple[int, ...], ...]
    segments: tuple[tuple[int, int, int], ...]
    retained_text: str
    block_kind: str
    handoff_rows: int
    base_prompt_rows: int
    selection_s: float
    receiver_prepare_s: float
    block_kinds: tuple[str, ...] = ()
    # False when the whole request fits one sliding window, so every row is
    # all-layer readable and the layout treatment is inert for this cell.
    window_binds: bool = False
    delimiter_rows: int = 0
    tier_spans: int = 0
    tier_rows: int = 0
    selected_indices_sha256: str | None = None
    qwen_flat_payload_identity: dict[str, object] | None = None


def _receiver_resident_bytes(rows: int) -> int:
    """Price the pinned 40 sliding plus 8 global Gemma KV prefix."""
    if rows < 0:
        raise ValueError("receiver rows cannot be negative")
    sliding = 40 * min(rows, 1_024) * (2 * 8 * 256 * 2)
    global_attention = 8 * rows * (2 * 1 * 512 * 2)
    return sliding + global_attention


def _channel(arm: str) -> str:
    if arm == "floor":
        return "none"
    prefix = f"{SELECTOR_FAMILY}_"
    policy = REGISTERED_ARM_POLICIES[arm]
    if not policy.startswith(prefix):
        raise RuntimeError(f"{arm}: policy does not carry the Gemma family prefix")
    return policy.removeprefix(prefix)


PAYLOAD_LAYOUT_BY_BLOCK_KIND = {
    **CUT_PAYLOAD_LAYOUT_BY_BLOCK_KIND,
    "worker_memory": SOURCE_PAYLOAD_LAYOUT,
    "none": SOURCE_PAYLOAD_LAYOUT,
}


def _payload_layout(block_kind: str) -> str:
    """The registered payload order the receiver actually built."""
    try:
        return PAYLOAD_LAYOUT_BY_BLOCK_KIND[block_kind]
    except KeyError as exc:
        raise RuntimeError(f"unregistered receiver block kind {block_kind!r}") from exc


def _keep_sha(keeps: tuple[tuple[int, ...], ...]) -> str:
    return "|".join(
        f"w{worker}:{hashlib.sha256(','.join(map(str, keep)).encode()).hexdigest()[:16]}"
        for worker, keep in enumerate(keeps)
    )


def _first(completion: StreamCompletion) -> float:
    if completion.n_tokens and completion.first_token_at is None:
        raise RuntimeError(f"{completion.request_id}: generated tokens have no TTFT clock")
    return completion.first_token_at or completion.finished_at


def _validate(completion: StreamCompletion) -> None:
    if completion.finish_reason not in {"stop", "length"}:
        raise RuntimeError(
            f"{completion.request_id}: invalid vLLM finish reason {completion.finish_reason!r}"
        )
    if completion.n_tokens != len(completion.token_ids):
        raise RuntimeError(f"{completion.request_id}: vLLM token count differs")
    if completion.num_cached_tokens not in {None, 0}:
        raise RuntimeError(f"{completion.request_id}: prefix cache contaminated timing")
    _first(completion)


def _producer_seconds(
    arm: str,
    capture: CaptureArtifact,
    prepared: PreparedArm,
    measured: float | None = None,
) -> float:
    """Producer work this live arm is charged for.

    ``measured`` is the split route's producer wall, priced in another process;
    a resident receiver passes nothing and adds the clocks it measured itself.
    """
    if arm not in REGISTERED_CUT_ARMS:
        return 0.0
    if measured is not None:
        return measured
    return capture.capture_s + prepared.selection_s


def _sample_fields(
    tokenizer: Any,
    item: dict[str, Any],
    completions: list[StreamCompletion],
    *,
    channel_open: int,
    channel_close: int,
    answer_ceiling: int,
) -> dict[str, list[Any]]:
    tokens_by_sample = [list(completion.token_ids) for completion in completions]
    sampled_texts = [tokenizer.decode(tokens) for tokens in tokens_by_sample]
    visible_ids = [
        visible_answer_tokens(
            tokens,
            channel_open=channel_open,
            channel_close=channel_close,
            stop_ids=STOP_IDS,
        )
        for tokens in tokens_by_sample
    ]
    answer_texts = [tokenizer.decode(tokens) for tokens in visible_ids]
    task_fields = [score_text(item["question"], answer) for answer in answer_texts]
    return {
        "generated_token_ids_by_sample": tokens_by_sample,
        "answers": sampled_texts,
        "answer_texts": answer_texts,
        "sample_task_fields": task_fields,
        "generated_tokens_by_sample": [len(tokens) for tokens in tokens_by_sample],
        "first_token_emitted_by_sample": [bool(tokens) for tokens in tokens_by_sample],
        # Read on the head's finish reason, never the merged length: an
        # injected row is longer than the head the sampler capped.
        "answer_hit_ceiling_by_sample": [
            answer_hit_ceiling(completion.finish_reason) for completion in completions
        ],
        "thinking_closed_by_sample": [
            channel_close in tokens or channel_open not in tokens for tokens in tokens_by_sample
        ],
        "finish_reasons": [completion.finish_reason for completion in completions],
        "answer_injected_by_sample": [completion.answer_injected for completion in completions],
        # The banked reason is the head's, named apart so a reader can find it
        # where a merged draw does not record its own termination.
        "answer_head_finish_reasons": [completion.finish_reason for completion in completions],
        "num_cached_tokens_by_sample": [completion.num_cached_tokens for completion in completions],
    }


def build_result_rows(
    *,
    tokenizer: Any,
    item: dict[str, Any],
    capture: CaptureArtifact,
    arm: str,
    prepared: PreparedArm,
    seeds: tuple[int, ...],
    completions: list[StreamCompletion],
    timing_origin: float,
    channel_open: int,
    channel_close: int,
    embedding_digests: dict[str, str],
    engine_identity_sha256: str,
    greedy_visible: str | None,
    greedy_loose: float | None,
    producer_s: float | None = None,
) -> list[dict[str, Any]]:
    """Build three per-seed rows and the aggregate timing fields they share.

    ``producer_s`` is the split route's producer wall, read from the record
    beside the handoff manifest; the resident route passes nothing.
    """
    if len(completions) != N_SEEDS or len(seeds) != N_SEEDS:
        raise RuntimeError(f"{item['qid']}/{arm}: decode did not return all registered samples")
    if [completion.tag for completion in completions] != [f"s{i}" for i in range(N_SEEDS)]:
        raise RuntimeError(f"{item['qid']}/{arm}: completion order differs from s0/s1/s2")
    for completion in completions:
        _validate(completion)
    sampling = cast(dict[str, Any], sampling_contract())
    sample = _sample_fields(
        tokenizer,
        item,
        completions,
        channel_open=channel_open,
        channel_close=channel_close,
        answer_ceiling=int(sampling["answer_ceiling"]),
    )
    timed = completions[0]
    rest = completions[1:]
    decode_s = timed.finished_at - timed.submitted_at
    decode_batched_s = max(row.finished_at for row in rest) - min(row.submitted_at for row in rest)
    receiver_ttft_s = _first(timed) - timed.submitted_at
    generation_s = timed.finished_at - _first(timed)
    queued_offsets = [row.submitted_at - timing_origin for row in completions]
    first_offsets = [_first(row) - timing_origin for row in completions]
    finished_offsets = [row.finished_at - timing_origin for row in completions]
    if queued_offsets[0] != 0.0 or any(
        queued < finished_offsets[0] for queued in queued_offsets[1:]
    ):
        raise RuntimeError(f"{item['qid']}/{arm}: s1/s2 started before s0 completed")
    continuation_offsets = continuation_offset_fields(completions, timing_origin)
    coverage = evidence_answerability_audit(
        item["question"],
        full_sources=item["page_texts"],
        retained_sources=[prepared.retained_text],
    )
    construction = item["construction"]
    if int(coverage["n_locatable_in_full_source"]) != int(
        construction["n_locatable_in_full_source"]
    ):
        raise RuntimeError(f"{item['qid']}/{arm}: answerability denominator drifted")
    # The worker prefills its whole framed prompt, so that is the denominator;
    # evidence volume is the shard alone, frame rows excluded.
    lengths = [int(ids.numel()) for ids in item["prompt_ids"]]
    evidence_lengths = [int(ids.numel()) for ids in item["memory_ids"]]
    kept_by_worker = [len(keep) for keep in prepared.keeps]
    charged_producer_s = _producer_seconds(arm, capture, prepared, producer_s)
    audition_s = 0.0
    audition_replays = 0
    selector = REGISTERED_ARM_SELECTORS.get(arm)
    # Delimiters are payload scaffolding, never kept rows, so a delimiter-free
    # qwen payload and a delimited tiered payload report the same kept count.
    kept_rows = prepared.handoff_rows - prepared.delimiter_rows
    carries_latent = arm != "floor"
    payload_identity = prepared.qwen_flat_payload_identity
    latent_tokens = kept_rows if carries_latent else 0
    rolled_tokens = WORKERS_PER_ITEM * LATENT_STEPS if carries_latent else 0
    handoff_prompt_tokens = prepared.prompt.shape[0] - prepared.base_prompt_rows
    if handoff_prompt_tokens != prepared.handoff_rows:
        raise RuntimeError(f"{item['qid']}/{arm}: handoff row accounting drifted")
    latent_wire_bytes = prepared.handoff_rows * EMBEDDING_SHAPE[1] * 2 if carries_latent else 0
    base_resident_bytes = _receiver_resident_bytes(prepared.base_prompt_rows)
    receiver_resident_bytes = _receiver_resident_bytes(int(prepared.prompt.shape[0]))
    timed_tokens = int(sample["generated_tokens_by_sample"][0])
    available_rows = sum(lengths) + rolled_tokens
    worker_prefill = 0 if arm == "floor" else available_rows
    total_prefill_tokens = worker_prefill + int(prepared.prompt.shape[0])
    common: dict[str, Any] = {
        "kind": "result",
        "schema": ROW_SCHEMA,
        "scoring_version": DEFAULT_SCORER_VERSION,
        "qid": item["qid"],
        "question": item["question"].question,
        "arm": arm,
        "mode": "fanoutqa-m3",
        "panel": str(item.get("panel", "production")),
        "ratio": REGISTERED_ARM_RATIOS.get(arm),
        "policy": REGISTERED_ARM_POLICIES[arm],
        "channel": _channel(arm),
        "policy_family": SELECTOR_FAMILY,
        "policy_registered": arm in REGISTERED_CUT_ARMS,
        "cut_arm": arm in REGISTERED_CUT_ARMS,
        "policy_selector": selector,
        "selector_composition": SUPPORT_COMPOSITION if selector == "support" else selector,
        "support_alpha": float(SUPPORT_DEFAULT_ALPHA) if selector == "support" else None,
        "variant_fold_rule": (
            VARIANT_FOLD_RULE.get(selector.removeprefix("support_"))
            if selector and selector != "support"
            else None
        ),
        "span_width": SPAN_WIDTH,
        "workers": WORKERS_PER_ITEM,
        "producer_backend": PRODUCER_BACKEND,
        "receiver_backend": "vllm",
        "decode_profile": GEMMA.decode.profile_id,
        "decode_fingerprint": GEMMA.decode.identity_hash,
        "worker_checkpoint": CHECKPOINT_ID,
        "receiver_checkpoint": CHECKPOINT_ID,
        "fleet_runtime": FLEET_RUNTIME,
        "receiver_format": "payload-then-chat-turn",
        "receiver_enable_thinking": RECEIVER_ENABLE_THINKING,
        "enable_thinking": RECEIVER_ENABLE_THINKING,
        "channel_blocks": len(prepared.segments),
        "channel_block_kind": prepared.block_kind,
        "payload_layout": _payload_layout(prepared.block_kind),
        "audition_s": round(audition_s, 4),
        "audition_replays": audition_replays,
        "receiver_block_kinds": list(prepared.block_kinds),
        "delimiter_rows": prepared.delimiter_rows,
        "tier_spans": prepared.tier_spans,
        "tier_rows": prepared.tier_rows,
        "receiver_sliding_window": RECEIVER_SLIDING_WINDOW,
        "window_binds": prepared.window_binds,
        "receiver_segments": [list(segment) for segment in prepared.segments],
        "receiver_rows": int(prepared.prompt.shape[0]),
        "base_prompt_tokens": prepared.base_prompt_rows,
        "receiver_prompt_tokens": int(prepared.prompt.shape[0]),
        "handoff_prompt_tokens": handoff_prompt_tokens,
        "latent_tokens": latent_tokens,
        "latent_wire_bytes": latent_wire_bytes,
        "total_wire_bytes": latent_wire_bytes,
        "handoff_resident_bytes": receiver_resident_bytes - base_resident_bytes,
        "receiver_resident_bytes": receiver_resident_bytes,
        "decode_path": "vllm-public-enqueue-prompt-embeds-stream",
        "decode_tags": [f"s{index}" for index in range(N_SEEDS)],
        "sample_seeds": list(seeds),
        "decode_order_index": 0,
        "decode_timing_sample": "s0",
        "fleet_decode_schedule": FLEET_DECODE_SCHEDULE,
        "timing_scope": COMPUTE_TIMING_SCOPE,
        "memory_tokens": sum(evidence_lengths),
        "memory_tokens_by_worker": evidence_lengths,
        "worker_prompt_tokens": lengths,
        "aggregate_worker_prompt_tokens": sum(lengths),
        "worker_evidence_payload_tokens": evidence_lengths,
        "aggregate_evidence_payload_tokens": sum(lengths),
        "latent_steps": LATENT_STEPS,
        "rolled_tokens": rolled_tokens,
        "kept_tokens": kept_rows,
        "kept_tokens_by_worker": kept_by_worker,
        "keep_fraction": kept_rows / available_rows,
        "full_latent_tokens": available_rows if carries_latent else 0,
        "achieved_ratio": (
            round(available_rows / latent_tokens, 4) if carries_latent and latent_tokens else None
        ),
        "keep_sha": _keep_sha(prepared.keeps) if carries_latent else "",
        "selected_indices_sha256": (prepared.selected_indices_sha256 if carries_latent else None),
        "selected_indices_order": "worker-major-ascending-v1" if carries_latent else None,
        "qwen_flat_payload_identity": payload_identity,
        "sink_kept": bool(sum(kept_by_worker)) and all(0 in keep for keep in prepared.keeps),
        "construction_locatable_leaves": int(construction["n_locatable_in_full_source"]),
        "construction_retained_leaves": int(construction["n_survives_construction"]),
        "construction_removed_leaves": int(construction["n_removed_by_construction"]),
        "construction_complete": bool(item["construction_complete"]),
        "construction_incomplete_reason": str(item["construction_incomplete_reason"]),
        "dead_pages": list(item["dead_pages"]),
        "answerability_group": (
            "construction_complete" if item["construction_complete"] else "construction_incomplete"
        ),
        "arm_retained_leaves": int(coverage["n_survives_construction"]),
        **sample,
        "answer": sample["answers"][0],
        "finish_reason": sample["finish_reasons"][0],
        "first_token_emitted": sample["first_token_emitted_by_sample"][0],
        "generated_tokens": round(
            sum(sample["generated_tokens_by_sample"]) / N_SEEDS,
            4,
        ),
        "thinking_closed_rate": round(
            sum(sample["thinking_closed_by_sample"]) / N_SEEDS,
            4,
        ),
        "length_finish_rate": round(
            sum(reason == "length" for reason in sample["finish_reasons"]) / N_SEEDS,
            4,
        ),
        "greedy_diagnostic_only": True,
        "greedy_visible_answer_text": greedy_visible,
        "greedy_loose_diagnostic": greedy_loose,
        # This lane registers no text arm; the two fields are still banked
        # because the shared merge reads report_failed on every row it folds.
        "report_failed": False,
        "report_failure": None,
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "top_k": sampling["top_k"],
        "decode_temperature": sampling["temperature"],
        "decode_top_p": sampling["top_p"],
        "decode_top_k": sampling["top_k"],
        "decode_presence_penalty": sampling["presence_penalty"],
        "decode_repeats": N_SEEDS,
        "decode_protocol": "gemma4_thinking_sampled",
        "answer_max_tokens": int(sampling["answer_ceiling"]),
        "answer_closer_budget": ANSWER_CLOSING_TOKEN_BUDGET,
        "stop_ids": sorted(STOP_IDS),
        "suppress_tokens": list(SUPPRESSED_TOKEN_IDS),
        "shared_embedding_file_sha256": embedding_digests["file_sha256"],
        "shared_embedding_tensor_sha256": embedding_digests["tensor_sha256"],
        "shared_embedding_marker_sha256": embedding_digests["marker_sha256"],
        "embedding_digests": capture.embedding_digests,
        "capture_s_by_worker": list(capture.capture_s_by_worker),
        "capture_s": capture.capture_s if arm in REGISTERED_CUT_ARMS else 0.0,
        "selection_s": round(prepared.selection_s, 4),
        "producer_s": round(charged_producer_s, 4),
        "latent_roll_s": (round(capture.capture_s, 4) if arm in REGISTERED_CUT_ARMS else 0.0),
        "full_materialize_s": 0.0,
        "vote_s": 0.0,
        "selector_scoring_peak_bytes": 0,
        "receiver_prepare_s": round(prepared.receiver_prepare_s, 6),
        "spill_load_s": 0.0,
        "spill_save_s": 0.0,
        "decode_s": round(decode_s, 4),
        "decode_batched_s": round(decode_batched_s, 4),
        "queue_s": None,
        "receiver_ttft_s": round(receiver_ttft_s, 4),
        "generation_s": round(generation_s, 4),
        "ttft_s": round(charged_producer_s + prepared.receiver_prepare_s + receiver_ttft_s, 4),
        "tteoa_s": round(charged_producer_s + prepared.receiver_prepare_s + decode_s, 4),
        "decode_queued_offset_s_by_sample": [round(value, 4) for value in queued_offsets],
        "decode_first_token_offset_s_by_sample": [round(value, 4) for value in first_offsets],
        "decode_finished_offset_s_by_sample": [round(value, 4) for value in finished_offsets],
        **continuation_offsets,
        "num_cached_tokens": timed.num_cached_tokens,
        "total_prefill_tokens": total_prefill_tokens,
        "total_decoded_tokens": timed_tokens,
        "total_tokens": total_prefill_tokens + timed_tokens,
        "payload_file_sha256": {},
        "payload_manifest_sha256": None,
        "cut_decoded_sha256": (
            payload_identity.get("payload_sha256") if payload_identity is not None else None
        ),
        "engine_identity_sha256": engine_identity_sha256,
        "global_attention_layers": list(capture.global_layers),
        "reloaded_captures": capture.reloaded_captures,
        "reloaded_selections": capture.reloaded_selections,
    }
    return [row_for_seed(common, index) for index in range(N_SEEDS)]


def row_for_seed(common: dict[str, Any], seed_index: int) -> dict[str, Any]:
    """Materialize one seed row from the aggregate payload banked on every row."""
    if not 0 <= seed_index < N_SEEDS:
        raise ValueError(f"seed index {seed_index} is outside the registered roster")
    required = {
        "sample_seeds",
        "generated_token_ids_by_sample",
        "answers",
        "answer_texts",
        "sample_task_fields",
        "generated_tokens_by_sample",
        "first_token_emitted_by_sample",
        "answer_hit_ceiling_by_sample",
        "thinking_closed_by_sample",
        "finish_reasons",
        "answer_injected_by_sample",
        "answer_head_finish_reasons",
        "num_cached_tokens_by_sample",
        "decode_queued_offset_s_by_sample",
        "decode_first_token_offset_s_by_sample",
        "decode_finished_offset_s_by_sample",
        "answer_continuation_submit_offset_s_by_sample",
        "answer_continuation_first_token_offset_s_by_sample",
    }
    if any(
        not isinstance(common.get(name), list) or len(common[name]) != N_SEEDS for name in required
    ):
        raise RuntimeError("banked Gemma aggregate sample vectors are incomplete")
    task = common["sample_task_fields"][seed_index]
    row = {
        key: value
        for key, value in common.items()
        if key not in {"attempt_id", "banked_at_unix", "cell"}
    }
    row.update(
        {
            "cell": decode_cell(str(common["arm"]), seed_index),
            "seed_index": seed_index,
            "seed": common["sample_seeds"][seed_index],
            "sampled_tokens": common["generated_token_ids_by_sample"][seed_index],
            "sampled_text": common["answers"][seed_index],
            "visible_answer_text": common["answer_texts"][seed_index],
            "answer_decoded_tokens": common["generated_tokens_by_sample"][seed_index],
            "answer_hit_ceiling": common["answer_hit_ceiling_by_sample"][seed_index],
            "thinking_closed": common["thinking_closed_by_sample"][seed_index],
            "finish_reason": common["finish_reasons"][seed_index],
            "answer_injected": common["answer_injected_by_sample"][seed_index],
            "first_token_emitted": common["first_token_emitted_by_sample"][seed_index],
            "num_cached_tokens": common["num_cached_tokens_by_sample"][seed_index],
            "loose": task["loose"],
            "strict": task["strict"],
            "n_leaves": task["n_leaves"],
            "loose_match": task["loose"] > 0.0,
            "decode_queued_offset_s": common["decode_queued_offset_s_by_sample"][seed_index],
            "decode_first_token_offset_s": common["decode_first_token_offset_s_by_sample"][
                seed_index
            ],
            "decode_finished_offset_s": common["decode_finished_offset_s_by_sample"][seed_index],
            "answer_continuation_submit_offset_s": common[
                "answer_continuation_submit_offset_s_by_sample"
            ][seed_index],
            "answer_continuation_first_token_offset_s": common[
                "answer_continuation_first_token_offset_s_by_sample"
            ][seed_index],
        }
    )
    return row


__all__ = (
    "FLEET_DECODE_SCHEDULE",
    "PreparedArm",
    "build_result_rows",
    "row_for_seed",
)
