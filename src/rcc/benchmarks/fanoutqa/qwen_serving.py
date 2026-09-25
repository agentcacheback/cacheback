"""The FanOutQA serving registration and the model-visible settings under it.

Their two fingerprints reproduce a profile's ``panel_registration_sha256`` and
``prepared_config_fingerprint``, over plain data, importing no engine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from typing import Any

from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.route import RouteFamily
from rcc.run import identity

SERVING_VERSION = "fanoutqa-m3-serving-v1"
SAMPLING_VERSION = "fanoutqa-fleet-sampling-v1"
FLEET_GPUS = 8
MANAGER_PROMPT_CEILING = 128
TEXT_WRAPPER_CEILING = 512
QUERY_COVERAGE_CHUNK_TOKENS = 256
#: The served rope geometry per lane, as (factor, original context, whether the
#: engine overrides the checkpoint). Qwen applies a YaRN override; Nemotron uses
#: its own native window.
_ROPE_GEOMETRY: dict[str, tuple[float, int, bool]] = {
    "qwen": (4.0, 32_768, True),
    "nemotron": (1.0, 131_072, False),
}


@dataclass(frozen=True)
class QwenPolicyConfig:
    """The model-visible settings the prepared bytes record.

    The fingerprint is taken over ``asdict`` with sorted keys, so declaration order
    does not matter but the field set does: adding or dropping a field moves it.
    """

    luna_checkpoint: str
    luna_revision: str
    sol_checkpoint: str
    sol_revision: str
    mini_checkpoint: str
    mini_revision: str
    mode: str
    data_source: str
    seed: int
    s_tokens: int
    latent_steps: int
    n_items: int
    shard_size: int
    span_size: int
    report_max_tokens: int
    answer_max_tokens: int
    worker_max_model_len: int
    receiver_max_model_len: int
    luna_gpu_memory_utilization: float
    sol_gpu_memory_utilization: float
    mini_gpu_memory_utilization: float
    latent_batch_rows: int
    context_padding: str
    page_allocation: str
    rope_factor: float
    rope_original_context: int
    rope_scaled_checkpoints: tuple[str, ...]
    report_style: str
    data_fetch_workers: int
    enable_thinking: bool
    decode_temperature: float
    decode_top_p: float
    decode_top_k: int | None
    decode_presence_penalty: float
    decode_repeats: int
    max_num_seqs: int
    producer_backend: str
    item_split: str
    workers_per_item: int


def _model_revisions(family: RouteFamily) -> dict[str, str]:
    return {checkpoint: revision for checkpoint, revision in family.profile.runtime.model_revisions}


def _checkpoints(family: RouteFamily) -> tuple[str, str, str]:
    """Return one family's pinned primary, medium, and small sender checkpoints."""
    ordered = tuple(checkpoint for checkpoint, _revision in family.profile.runtime.model_revisions)
    if len(ordered) != 3:
        raise RuntimeError(f"the sealed {family.lane} runtime must pin exactly three checkpoints")
    return ordered


def panel_policy_config(profile: BenchmarkProfile, *, family: RouteFamily) -> QwenPolicyConfig:
    """Build the model-visible settings for one panel of one profile."""
    profile = family.benchmark_profile(profile)
    sol, luna, mini = _checkpoints(family)
    revisions = _model_revisions(family)
    decode = family.profile.decode
    rope_factor, rope_original, rope_overridden = _ROPE_GEOMETRY[family.lane]
    items = len(profile.question_ids)
    config = QwenPolicyConfig(
        luna_checkpoint=luna,
        luna_revision=revisions[luna],
        sol_checkpoint=sol,
        sol_revision=revisions[sol],
        mini_checkpoint=mini,
        mini_revision=revisions[mini],
        mode=f"fanoutqa_m3_fleet_serving{items}",
        data_source="fanoutqa",
        seed=profile.report_seed_base,
        s_tokens=profile.worker_prompt_tokens,
        latent_steps=profile.latent_steps,
        n_items=items,
        # One item per service call, rather than a round-robin shard.
        shard_size=1,
        span_size=profile.span_width,
        report_max_tokens=profile.report_ceiling,
        answer_max_tokens=profile.answer_ceiling,
        worker_max_model_len=profile.max_model_len,
        receiver_max_model_len=profile.max_model_len,
        luna_gpu_memory_utilization=0.86,
        sol_gpu_memory_utilization=0.80,
        mini_gpu_memory_utilization=0.86,
        latent_batch_rows=2,
        context_padding="neutral",
        page_allocation="max_min_query",
        rope_factor=rope_factor,
        rope_original_context=rope_original,
        rope_scaled_checkpoints=(sol, luna, mini) if rope_overridden else (),
        report_style="natural",
        data_fetch_workers=FLEET_GPUS,
        enable_thinking=decode.enable_thinking,
        decode_temperature=decode.temperature,
        decode_top_p=decode.top_p,
        decode_top_k=decode.top_k,
        decode_presence_penalty=decode.presence_penalty,
        decode_repeats=len(profile.sample_tags),
        max_num_seqs=16,
        producer_backend="hf",
        item_split="heldout",
        workers_per_item=profile.workers_per_item,
    )
    if not profile.exact_prompts:
        # A natural panel is frozen as text and tokenized per family with no
        # filler, so it records no padding and its width is a ceiling.
        config = replace(config, context_padding="none", page_allocation="sealed_text")
    if family.native_sender_prompts:
        # A native sender records the serial engine roll and its serving
        # utilization; the per-role engine config holds everything else.
        config = replace(
            config,
            producer_backend="engine",
            luna_gpu_memory_utilization=0.80,
            mini_gpu_memory_utilization=0.80,
            latent_batch_rows=1,
        )
    return config


def policy_config_fingerprint(config: QwenPolicyConfig) -> str:
    """Fingerprint one set of model-visible settings."""
    return identity.fingerprint(asdict(config), identity.json_compact_legacy, digest_chars=16)


def sampling_contract(profile: BenchmarkProfile, *, family: RouteFamily) -> dict[str, Any]:
    """Return the decode settings every FanOutQA draw is made under."""
    profile = family.benchmark_profile(profile)
    decode = family.profile.decode
    return {
        "version": SAMPLING_VERSION,
        "models": _model_revisions(family),
        "sample_tags": list(profile.sample_tags),
        "enable_thinking": decode.enable_thinking,
        "temperature": decode.temperature,
        "top_p": decode.top_p,
        "top_k": decode.top_k,
        "presence_penalty": decode.presence_penalty,
        "answer_ceiling": profile.answer_ceiling,
        "report_ceiling": profile.report_ceiling,
        "max_model_len": profile.max_model_len,
        "max_num_seqs": 16,
    }


def sampling_fingerprint(profile: BenchmarkProfile, *, family: RouteFamily) -> str:
    """Return the full-length digest of the decode settings."""
    return identity.fingerprint(
        sampling_contract(profile, family=family), identity.json_compact_legacy
    )


def _policies(profile: BenchmarkProfile, family: RouteFamily) -> tuple[list[str], list[str]]:
    policies = [family.policy(arm.arm_id) for arm in profile.arms]
    support = [family.policy(arm.arm_id) for arm in profile.arms if arm.selector == "support"]
    return policies, support


def _display_names(profile: BenchmarkProfile, family: RouteFamily) -> dict[str, str]:
    fixed = {
        "issue_only": "fanout_issue_only",
        "text_primary": "fanout_text_8b_to_8b",
        "text_medium": "fanout_text_4b_to_8b",
        "text_small": "fanout_text_1p7b_to_8b",
    }
    if family.native_sender_prompts:
        fixed.update(text_medium="fanout_text_3b_to_8b", text_small="fanout_text_2b_to_8b")
    names: dict[str, str] = {}
    for arm in profile.arms:
        policy = family.policy(arm.arm_id)
        if arm.selector is None:
            names[policy] = fixed[arm.arm_id]
        elif arm.selector == "snap":
            names[policy] = f"fanout_latent_8b_r{arm.retention_ratio}"
        else:
            names[policy] = f"fanout_query_support_8b_r{arm.retention_ratio}"
    return names


def _ladder(profile: BenchmarkProfile, selector: str, family: RouteFamily) -> list[str]:
    return [family.policy(arm.arm_id) for arm in profile.arms if arm.selector == selector]


def _comparisons(profile: BenchmarkProfile, family: RouteFamily) -> list[list[str]]:
    text = family.policy("text_primary")
    snap = _ladder(profile, "snap", family)
    support = _ladder(profile, "support", family)
    pairs = [
        [family.policy("issue_only"), text],
        [family.policy("text_medium"), text],
        [family.policy("text_small"), text],
        *([text, policy] for policy in snap),
        *([left, right] for left, right in pairwise(snap)),
        *(([left, right] for left, right in zip(snap, support, strict=True)) if snap else ()),
        *([text, policy] for policy in support),
        *([left, right] for left, right in pairwise(support)),
    ]
    return pairs


def receiver_geometry(profile: BenchmarkProfile, ratio: int) -> dict[str, Any]:
    """Return the worst-case receiver window arithmetic for one latent ratio."""
    rows = profile.worker_prompt_tokens + profile.latent_steps
    latent = profile.workers_per_item * ((rows + ratio - 1) // ratio)
    required = latent + MANAGER_PROMPT_CEILING + profile.answer_ceiling
    return {
        "ratio": ratio,
        "latent_tokens": latent,
        "manager_prompt_tokens": MANAGER_PROMPT_CEILING,
        "answer_tokens": profile.answer_ceiling,
        "required_tokens": required,
        "max_model_len": profile.max_model_len,
        "headroom_tokens": profile.max_model_len - required,
    }


def span_tiling(profile: BenchmarkProfile) -> dict[str, Any]:
    """Return the fixed-width span tiling one worker roll is cut on."""
    rows = profile.worker_prompt_tokens + profile.latent_steps
    width = profile.span_width
    count = (rows + width - 1) // width
    return {
        "rows": rows,
        "span_size": width,
        "span_count": count,
        "full_span_width": width,
        "final_span_width": rows - width * (count - 1),
        "widest_unprotected_width": width,
    }


def text_geometry(profile: BenchmarkProfile) -> dict[str, int]:
    """Return the receiver window arithmetic of the text arms."""
    reports = profile.workers_per_item * profile.report_ceiling
    return {
        "reports": reports,
        "manager": MANAGER_PROMPT_CEILING,
        "wrapper": TEXT_WRAPPER_CEILING,
        "answer": profile.answer_ceiling,
        "required": reports
        + MANAGER_PROMPT_CEILING
        + TEXT_WRAPPER_CEILING
        + profile.answer_ceiling,
    }


_SOURCE_CONSTRUCTION = {
    "page_ownership": "frozen_census_longest_first",
    "within_worker_quota": "live_snapshot_max_min",
    "canonicalization": "drop_reference_and_navigation_appendices",
    "query_coverage_chunk_tokens": QUERY_COVERAGE_CHUNK_TOKENS,
    "head_quota_fraction": "1/2",
    "odd_head_rounding": "ceil",
    "tail_chunk_eligibility": "all_chunks_including_final_partial",
    "query_ranking": "direct_then_one_chunk_context_lexical_numeric_table",
    "materialization_order": "original_article_order",
}
_CLAIM = (
    "descriptive FanOutQA heldout scale run; proxy loose and strict scores "
    "are internal arm comparisons, not published FanOutQA accuracy"
)
_EXECUTION = (
    "arm-at-a-time dynamic FIFO on eight H100s; persistent producer, receiver, "
    "or fused engines; every arm consumes all eight GPUs"
)
_PAIRING = (
    "question id and registered per-policy logical sample tag across arm banks; "
    "GPU and attempt are recorded blocking factors"
)
_CENSORING = (
    "bank and score every registered sample once; never redraw or exclude an "
    "unclosed or length-finished answer; label the full run output-censored "
    "when any such sample exists and publish per-policy counts"
)


def serving_registration(profile: BenchmarkProfile, *, family: RouteFamily) -> dict[str, Any]:
    """Return the serving registration the prepared bytes carry."""
    profile = family.benchmark_profile(profile)
    policies, support = _policies(profile, family)
    ratios = list(profile.ratios)
    return {
        "version": SERVING_VERSION,
        "claim": _CLAIM,
        "items": list(profile.question_ids),
        "item_split": "heldout",
        "workers_per_item": profile.workers_per_item,
        "worker_prompt_tokens": profile.worker_prompt_tokens,
        "aggregate_worker_prompt_tokens": profile.workers_per_item * profile.worker_prompt_tokens,
        **({} if profile.exact_prompts else {"prompt_geometry": profile.prompt_geometry}),
        **(
            {
                "worker_prompt_tokens_scope": (
                    "primary/medium exact 50K; small per-worker T=max(50000,L), where L is the "
                    "complete native prompt before ballast; actual counts are banked; "
                    "small is an unequal token budget and realized evidence depth baseline"
                )
            }
            if family.native_sender_prompts
            else {}
        ),
        "source_construction": dict(_SOURCE_CONSTRUCTION),
        "gpus": FLEET_GPUS,
        "policies": policies,
        "support_policies": support,
        "selector_roster_rule": "Query-Support ratio arms execute unconditionally",
        "display_names": _display_names(profile, family),
        "comparisons": _comparisons(profile, family),
        "geometry": [receiver_geometry(profile, ratio) for ratio in ratios],
        "r1_infeasibility": receiver_geometry(profile, 1),
        "text_geometry": text_geometry(profile),
        "span_tiling": span_tiling(profile),
        "policy_config": asdict(panel_policy_config(profile, family=family)),
        "model_revisions": _model_revisions(family),
        "sampling": sampling_contract(profile, family=family),
        "sampling_fingerprint": sampling_fingerprint(profile, family=family),
        "scorer": "repository string-only FanOutQA proxy",
        "answer_censoring_rule": _CENSORING,
        "execution": _EXECUTION,
        "pairing": _PAIRING,
    }


def panel_registration_sha256(profile: BenchmarkProfile, *, family: RouteFamily) -> str:
    """Fingerprint the serving registration exactly as the prepared bytes carry it."""
    return identity.fingerprint(
        serving_registration(profile, family=family),
        identity.json_compact_legacy,
        digest_chars=16,
    )


__all__ = (
    "MANAGER_PROMPT_CEILING",
    "QwenPolicyConfig",
    "panel_policy_config",
    "panel_registration_sha256",
    "policy_config_fingerprint",
    "receiver_geometry",
    "sampling_contract",
    "sampling_fingerprint",
    "serving_registration",
    "span_tiling",
    "text_geometry",
)
