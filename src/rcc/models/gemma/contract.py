"""The registered Gemma 4 FanOutQA resident-model contract, validated at import."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from rcc.models.decode import decode_protocol
from rcc.models.gemma import GEMMA
from rcc.run import identity
from rcc.transforms.select.query_support.capture_bank_format import LAYER_BANK_SCHEMA
from rcc.transforms.select.query_support.methods.compose import (
    SUPPORT_DEFAULT_ALPHA,
    SUPPORT_DEFAULT_ORDER,
)

FLEET_RUNTIME = "generic-fleet-serving-v1"
PIPELINE_VERSION = "gemma-fanoutqa-m3-fleet-v17"
ROW_SCHEMA = "gemma-fanoutqa-m3-fleet-row-v10"
PRODUCER_BACKEND = "vllm"
RESIDENT_LIFETIME = (
    "vllm-open",
    "vllm-prefill",
    "side-capture",
    "select",
    "serve",
)
SELECTION_SCHEMA = "gemma-fanoutqa-m3-selection-v5"
MODEL_FAMILY = "gemma4"
SELECTOR_FAMILY = "gemma4_12b"
WORKERS_PER_ITEM = 3
EVIDENCE_TOKENS_PER_WORKER = 50_000
SPAN_WIDTH = 16
POOL_KERNEL = 7
# The registered cargo: 40 rolled latent thoughts per worker, norm-matched
# identity realign, and the tail protected in every cut.
LATENT_STEPS = 40
LATENT_REALIGN_ENABLED = False
# The worker report voice: the text arm renders it and the latent frame is
# built from the same brief and instruction.
REPORT_STYLE = "natural"
_ROLL_BRIEF = """You are one reasoning worker. Inspect only your
private evidence. Work through every relevant name, number, relationship, and
uncertainty that could help the coordinator answer the question. Return a complete
reasoning handoff, including intermediate reasoning and unresolved alternatives.
Do not claim access to any other worker's evidence."""
_ROLL_INSTRUCTION = "Return the complete reasoning handoff now."
# Question-conditioned cargo: the roll is framed in token space, so the rolled
# thoughts see the brief, the question, and the instruction while the packed
# evidence rows stay verbatim between the two frame halves.
LATENT_ROLL_PREFIX_TEMPLATE = f"{_ROLL_BRIEF}\n\nPrivate evidence:\n"
LATENT_ROLL_SUFFIX_TEMPLATE = f"\n\nQuestion: {{question}}\n\n{_ROLL_INSTRUCTION}"
# Live split payloads use the Qwen recipe: per-worker evidence followed by
# that worker's rolled tail, in source order, with no delimiter rows.
QWEN_PAYLOAD_LAYOUT = "flat-interleave-v1"
PAYLOAD_LAYOUT = QWEN_PAYLOAD_LAYOUT
SUPPORT_COMPOSITION = f"support-p{int(SUPPORT_DEFAULT_ORDER)}-a{int(SUPPORT_DEFAULT_ALPHA)}"
RECEIVER_SLIDING_WINDOW = 1_024
RATIOS = (2, 4, 8, 16, 32, 64, 128)
SELECTORS = ("support",)
BASELINE_ARMS = ("floor",)
FULL_ARM = "full"
POLICY_SUFFIXES = {"support": ""}


def _latent_arms() -> tuple[str, ...]:
    return tuple(
        f"r{ratio}{POLICY_SUFFIXES[selector]}" for selector in SELECTORS for ratio in RATIOS
    )


ARMS = (*BASELINE_ARMS, FULL_ARM, *_latent_arms())
ARM_SELECTORS = {
    f"r{ratio}{POLICY_SUFFIXES[selector]}": selector for selector in SELECTORS for ratio in RATIOS
}
ARM_RATIOS = {
    f"r{ratio}{POLICY_SUFFIXES[selector]}": ratio for selector in SELECTORS for ratio in RATIOS
}
LATENT_POLICIES = {
    arm: f"{SELECTOR_FAMILY}_r{ARM_RATIOS[arm]}_w{SPAN_WIDTH}_{selector}"
    for arm, selector in ARM_SELECTORS.items()
}
CUT_ARMS = tuple(LATENT_POLICIES)
ARM_POLICIES = {
    "floor": f"{SELECTOR_FAMILY}_floor",
    FULL_ARM: f"{SELECTOR_FAMILY}_full",
    **LATENT_POLICIES,
}
# Mean query attention remains available as an explicit shared mechanism, but is not in the
# default live execution roster above.
QSNAP_ARMS = tuple(f"qsnap_r{ratio}" for ratio in RATIOS)
QSNAP_ARM_RATIOS = {arm: ratio for arm, ratio in zip(QSNAP_ARMS, RATIOS, strict=True)}
QSNAP_ARM_SELECTORS = dict.fromkeys(QSNAP_ARMS, "snap")
QSNAP_ARM_POLICIES = {
    arm: f"{SELECTOR_FAMILY}_r{ratio}_w{SPAN_WIDTH}" for arm, ratio in QSNAP_ARM_RATIOS.items()
}
REGISTERED_ARM_POLICIES = {**ARM_POLICIES, **QSNAP_ARM_POLICIES}
REGISTERED_ARM_RATIOS = {**ARM_RATIOS, **QSNAP_ARM_RATIOS}
REGISTERED_ARM_SELECTORS = {**ARM_SELECTORS, **QSNAP_ARM_SELECTORS}
REGISTERED_CUT_ARMS = (*CUT_ARMS, *QSNAP_ARMS)
SAMPLE_TAGS = ("s0", "s1", "s2")
N_SEEDS = len(SAMPLE_TAGS)
ANSWER_MAX_NEW_TOKENS = 24_000
REPORT_MAX_NEW_TOKENS = ANSWER_MAX_NEW_TOKENS
REPORT_SEED_TAGS = ("s0", "s1", "s2")
RECEIVER_ENABLE_THINKING = GEMMA.decode.enable_thinking
REPORT_ENABLE_THINKING = GEMMA.decode.enable_thinking
STOP_IDS = frozenset({1, 50, 106})
SUPPRESSED_TOKEN_IDS = (258_883, 258_882)
# The checkpoint's trained context.
MAX_MODEL_LEN = 262_144
MAX_NUM_SEQS = 16
ENFORCE_EAGER = False
CHECKPOINT_ID = GEMMA.checkpoint
CHECKPOINT_REVISION = GEMMA.revision
EMBEDDING_SHAPE = (262_144, 3_840)
# Copied from the canonical JSON contract below. It is a literal, so a change
# to the contract has to move this value and any later drift fails loudly.
REGISTERED_THINKING_CONFIG_FINGERPRINT = "f85b44f1beca"


def scientific_contract() -> dict[str, object]:
    """Return the all-thinking Gemma scientific contract."""
    protocol = decode_protocol(MODEL_FAMILY)
    return {
        "stage": "fire",
        "checkpoint": CHECKPOINT_ID,
        "model_class": "Gemma4UnifiedForCausalLM",
        "dtype": "bfloat16",
        "revision": CHECKPOINT_REVISION,
        "attention_implementation": "sdpa",
        "S_tokens": EVIDENCE_TOKENS_PER_WORKER,
        "workers": WORKERS_PER_ITEM,
        "n_items": 30,
        "n_seeds": N_SEEDS,
        "selector_family": SELECTOR_FAMILY,
        "span_width": SPAN_WIDTH,
        "pool_kernel": POOL_KERNEL,
        "latent_steps": LATENT_STEPS,
        "latent_steps_scope": "per-worker",
        "latent_realign_enabled": LATENT_REALIGN_ENABLED,
        "latent_roll": "hf-backbone-norm-matched-identity-roll",
        "latent_roll_conditioning": "question-conditioned-token-space-frame",
        "latent_roll_frame": "chat-turn-prefix-then-memory-rows-verbatim-then-question-suffix",
        "latent_roll_prefix_template": LATENT_ROLL_PREFIX_TEMPLATE,
        "latent_roll_suffix_template": LATENT_ROLL_SUFFIX_TEMPLATE,
        "latent_roll_scored_rows": "full-framed-prompt-rows-then-rolled-tail-all-selectable",
        "report_style": REPORT_STYLE,
        "payload_layout": PAYLOAD_LAYOUT,
        "payload_layout_arms": "all-cut-arms",
        "qwen_payload_layout": QWEN_PAYLOAD_LAYOUT,
        "qwen_layout_rule": "per-worker-evidence-then-cargo-no-delimiters",
        "worker_geometry": "qwen-ballast-fixed-fraction-evidence-rolled-prompt-exact-50k",
        "probe": "qwen-coordinator-prompt-chat-templated",
        "selectable_domain": "full-framed-rolled-prompt-plus-tail-frame-rows-compete",
        "sink_pin": "scores0-pinned-to-max-before-cut",
        "receiver_wrapper": "payload-before-chat-turn",
        "cargo_delimited": False,
        "receiver_sliding_window": RECEIVER_SLIDING_WINDOW,
        "latent_ratios": RATIOS,
        "run_ratios": RATIOS,
        "collect_layer_bank": True,
        "bank_include_sliding": True,
        "layer_bank_schema": LAYER_BANK_SCHEMA,
        "selection_phase": "hf-before-release",
        "selection_artifact_schema": SELECTION_SCHEMA,
        "arms": ARMS,
        "fire_arms": ARMS,
        "arm_selectors": ARM_SELECTORS,
        "answer_max_new_tokens": ANSWER_MAX_NEW_TOKENS,
        "receiver_enable_thinking": RECEIVER_ENABLE_THINKING,
        "model_family": MODEL_FAMILY,
        "worker_frame_enable_thinking": REPORT_ENABLE_THINKING,
        "greedy_diagnostic_only": True,
        "row_schema": ROW_SCHEMA,
        "seeds": {"model": 5_201, "report": 5_202, "decode": 5_203, "fixture": 5_204},
        "sample_temperature": protocol.temperature,
        "sample_top_p": protocol.top_p,
        "sample_top_k": protocol.top_k,
    }


def scientific_config_fingerprint() -> str:
    """Return the exact ruled thinking-lane fingerprint used for logical seeds."""
    encoded = json.dumps(scientific_contract(), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def sampling_contract() -> dict[str, object]:
    """Return the registered model-visible Gemma receiver contract."""
    protocol = decode_protocol(MODEL_FAMILY)
    return {
        "version": "gemma-fanoutqa-m3-all-thinking-sampling-v3",
        "checkpoint": CHECKPOINT_ID,
        "revision": CHECKPOINT_REVISION,
        "sample_tags": list(SAMPLE_TAGS),
        "enable_thinking": RECEIVER_ENABLE_THINKING,
        "receiver_enable_thinking": RECEIVER_ENABLE_THINKING,
        "report_enable_thinking": REPORT_ENABLE_THINKING,
        "temperature": protocol.temperature,
        "top_p": protocol.top_p,
        "top_k": protocol.top_k,
        "presence_penalty": GEMMA.decode.presence_penalty,
        "answer_ceiling": ANSWER_MAX_NEW_TOKENS,
        "report_ceiling": REPORT_MAX_NEW_TOKENS,
        "stop_ids": sorted(STOP_IDS),
        "suppress_tokens": list(SUPPRESSED_TOKEN_IDS),
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": MAX_NUM_SEQS,
    }


def registration() -> dict[str, object]:
    """Return the immutable scientific and execution registration."""
    return {
        "version": PIPELINE_VERSION,
        "fleet_runtime": FLEET_RUNTIME,
        "producer_backend": PRODUCER_BACKEND,
        "producer_prefill": "vllm-engine-extracted-hybrid-cache",
        "producer_weight_view": "zero-copy-aliased-live-engine",
        "scientific_contract_role": "frozen-v15-logical-and-seed-oracle",
        "latent_recurrence": "native-embedding-mean-norm-every-step-v1",
        "latent_handoff_gain": 1.0,
        "latent_selection_support": "native-recurrence-own-support",
        "receiver_backend": "vllm",
        "enforce_eager": ENFORCE_EAGER,
        "lifetime": list(RESIDENT_LIFETIME),
        "checkpoint": CHECKPOINT_ID,
        "revision": CHECKPOINT_REVISION,
        "workers_per_item": WORKERS_PER_ITEM,
        "evidence_tokens_per_worker": EVIDENCE_TOKENS_PER_WORKER,
        "ratios": list(RATIOS),
        "selectors": list(SELECTORS),
        "arms": list(ARMS),
        "span_width": SPAN_WIDTH,
        "pool_kernel": POOL_KERNEL,
        "latent_steps": LATENT_STEPS,
        "scientific_config": scientific_contract(),
        "scientific_config_fingerprint": scientific_config_fingerprint(),
        "sampling": sampling_contract(),
    }


def registration_fingerprint() -> str:
    """Hash the registration under its pinned JSON encoding."""
    return identity.fingerprint(registration(), identity.json_compact_legacy)


def arm_seeds(qid: str, arm: str) -> tuple[int, ...]:
    """Return placement-independent seeds for one item-arm cell.

    Every control draws its ratio arm's seeds, so each ratio's variant set
    is draw-matched.
    """
    try:
        policy = ARM_POLICIES[arm]
    except KeyError as exc:
        raise ValueError(f"unregistered Gemma arm {arm!r}") from exc
    fingerprint = scientific_config_fingerprint()
    return tuple(
        identity.logical_seed((fingerprint, qid, policy, "answer", tag), 8) for tag in SAMPLE_TAGS
    )


def runtime_fingerprint(signature: Mapping[str, object]) -> str:
    """Bind this new route to its stack and registration."""
    body = {"runtime_signature": dict(signature), "registration": registration()}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def validate_contract() -> None:
    """Fail closed on any roster, grammar, or protocol drift."""
    if len(ARMS) != 9 or len(LATENT_POLICIES) != 7 or len(CUT_ARMS) != 7:
        raise RuntimeError(
            "Gemma fleet roster is not one baseline, one full, and seven latent arms"
        )
    if len(set(ARM_POLICIES.values())) != len(ARMS) or set(ARM_POLICIES) != set(ARMS):
        raise RuntimeError("Gemma fleet arms do not carry one distinct policy name each")
    if LATENT_STEPS != 40:
        raise RuntimeError("Gemma fleet latent cargo drifted from the registered 40-step roll")
    registered = {f"{SELECTOR_FAMILY}_r{ratio}_w{SPAN_WIDTH}_support" for ratio in RATIOS}
    if set(LATENT_POLICIES.values()) != registered:
        raise RuntimeError("Gemma fleet latent names differ from the family grammar")
    contract = sampling_contract()
    expected = (1.0, 0.95, 64, 0.0)
    observed = (
        contract["temperature"],
        contract["top_p"],
        contract["top_k"],
        contract["presence_penalty"],
    )
    if observed != expected:
        raise RuntimeError(f"Gemma decode registry drifted from {expected}: {observed}")
    if (
        ANSWER_MAX_NEW_TOKENS != 24_000
        or REPORT_MAX_NEW_TOKENS != ANSWER_MAX_NEW_TOKENS
        or len(SAMPLE_TAGS) != 3
        or REPORT_SEED_TAGS != ("s0", "s1", "s2")
        or contract["max_num_seqs"] != 16
    ):
        raise RuntimeError("Gemma fleet ceiling or sample count drifted")
    observed_fingerprint = scientific_config_fingerprint()
    if observed_fingerprint != REGISTERED_THINKING_CONFIG_FINGERPRINT:
        raise RuntimeError(
            "Gemma scientific contract drifted from the ruled thinking lane: "
            f"{observed_fingerprint} != {REGISTERED_THINKING_CONFIG_FINGERPRINT}"
        )


validate_contract()


__all__ = (
    "ANSWER_MAX_NEW_TOKENS",
    "ARMS",
    "ARM_POLICIES",
    "ARM_RATIOS",
    "ARM_SELECTORS",
    "BASELINE_ARMS",
    "CHECKPOINT_ID",
    "CHECKPOINT_REVISION",
    "CUT_ARMS",
    "EMBEDDING_SHAPE",
    "ENFORCE_EAGER",
    "EVIDENCE_TOKENS_PER_WORKER",
    "FLEET_RUNTIME",
    "FULL_ARM",
    "LATENT_POLICIES",
    "LATENT_ROLL_PREFIX_TEMPLATE",
    "LATENT_ROLL_SUFFIX_TEMPLATE",
    "MAX_MODEL_LEN",
    "MAX_NUM_SEQS",
    "N_SEEDS",
    "PIPELINE_VERSION",
    "POOL_KERNEL",
    "PRODUCER_BACKEND",
    "QSNAP_ARMS",
    "QSNAP_ARM_POLICIES",
    "QSNAP_ARM_RATIOS",
    "QWEN_PAYLOAD_LAYOUT",
    "RATIOS",
    "RECEIVER_ENABLE_THINKING",
    "REGISTERED_THINKING_CONFIG_FINGERPRINT",
    "REPORT_ENABLE_THINKING",
    "REPORT_MAX_NEW_TOKENS",
    "REPORT_SEED_TAGS",
    "REPORT_STYLE",
    "RESIDENT_LIFETIME",
    "ROW_SCHEMA",
    "SAMPLE_TAGS",
    "SELECTION_SCHEMA",
    "SELECTORS",
    "SELECTOR_FAMILY",
    "SPAN_WIDTH",
    "STOP_IDS",
    "SUPPORT_COMPOSITION",
    "SUPPRESSED_TOKEN_IDS",
    "WORKERS_PER_ITEM",
    "arm_seeds",
    "registration",
    "registration_fingerprint",
    "runtime_fingerprint",
    "sampling_contract",
    "scientific_config_fingerprint",
    "scientific_contract",
    "validate_contract",
)
