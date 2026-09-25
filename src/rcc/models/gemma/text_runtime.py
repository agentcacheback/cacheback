"""Gemma 4 text-arm runtime for the resident lifecycle.

E4B and E2B run as sequential native-context vLLM phases before the resident
engine opens; ``text_primary`` borrows that already-live 12B engine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict, cast

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.gemma import GEMMA
from rcc.models.gemma.contract import (
    ANSWER_MAX_NEW_TOKENS,
    MAX_NUM_SEQS,
)
from rcc.models.gemma.engine import EngineConfig
from rcc.models.gemma.text_bank import TextReportBankProtocol
from rcc.models.gemma.text_codec import bind_sender_prompt_artifact
from rcc.models.gemma.text_contract import (
    TEXT_SENDERS,
    TextSenderSpec,
)
from rcc.models.gemma.text_reports import generate_report_bundle

SHARED_DECODE_PRESENCE_PENALTY = GEMMA.decode.presence_penalty
SHARED_DECODE_TEMPERATURE = GEMMA.decode.temperature
_REGISTERED_TOP_K = GEMMA.decode.top_k
if _REGISTERED_TOP_K is None:
    raise RuntimeError("Gemma family-native decoding requires top-k 64")
SHARED_DECODE_TOP_K: int = _REGISTERED_TOP_K
SHARED_DECODE_TOP_P = GEMMA.decode.top_p
SHARED_SEED = FANOUTQA_NATURAL_DEV50.report_seed_base

TEXT_RECEIVER_SCHEMA = "gemma4-fanoutqa-text-receiver-v1"
SHARED_SAMPLING_PROFILE = GEMMA.decode.profile_id


class SharedSamplingSpec(TypedDict):
    """Family-native sampling values shared by Gemma text senders."""

    profile: str
    sampling_fingerprint: str
    seed: int
    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    answer_ceiling: int
    report_ceiling: int
    sample_tags: list[str]


def shared_sampling_spec() -> SharedSamplingSpec:
    """Return the family-native Gemma values used by every text arm."""
    return {
        "profile": SHARED_SAMPLING_PROFILE,
        "sampling_fingerprint": GEMMA.decode.identity_hash,
        "seed": SHARED_SEED,
        "temperature": SHARED_DECODE_TEMPERATURE,
        "top_p": SHARED_DECODE_TOP_P,
        "top_k": SHARED_DECODE_TOP_K,
        "presence_penalty": SHARED_DECODE_PRESENCE_PENALTY,
        "answer_ceiling": ANSWER_MAX_NEW_TOKENS,
        "report_ceiling": ANSWER_MAX_NEW_TOKENS,
        "sample_tags": ["s0", "s1", "s2"],
    }


def report_engine_config(sender: TextSenderSpec) -> EngineConfig:
    """Build a native-context sender config without YaRN or HF overrides."""
    if sender.resident:
        raise RuntimeError("text_primary must borrow the resident 12B engine")
    return EngineConfig(
        model=sender.checkpoint,
        revision=sender.revision,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=sender.native_max_model_len,
        max_num_seqs=MAX_NUM_SEQS,
        enforce_eager=False,
        enable_prompt_embeds=False,
        enable_prefix_caching=False,
        attention_config=(("backend", "FLASH_ATTN"), ("flash_attn_version", 4)),
        language_model_only=True,
        generation_config="auto",
        disable_log_stats=False,
    )


def bind_text_prompt_item(
    tokenizer: Any,
    item: Mapping[str, Any],
    sender: TextSenderSpec,
    *,
    prepared_manifest: Mapping[str, Any],
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> dict[str, Any]:
    """Attach one sender's source-bound prompt artifact to one panel item.

    Every Gemma text sender shares the panel's tokenizer bytes, so the prepared
    prompt is the one each sender sends; an item carries one arm's artifact.
    """
    manifest = cast(Mapping[str, Any], prepared_manifest.get("source_manifest") or {})
    source_identity = str(manifest.get("fingerprint") or "")
    prepared_sha256 = str(prepared_manifest.get("artifact_sha256") or "")
    artifact = bind_sender_prompt_artifact(
        tokenizer,
        item,
        sender,
        source_identity=source_identity,
        sender_prepared_sha256=prepared_sha256,
        profile=profile,
    )
    return {
        **item,
        "source_identity": source_identity,
        "text_prompt_artifacts": {sender.semantic_arm: artifact},
        "text_prepared_sha256_by_arm": {sender.semantic_arm: prepared_sha256},
    }


def generate_resident_primary_reports(
    engine: Any,
    tokenizer: Any,
    items: Sequence[Mapping[str, Any]],
    bank: TextReportBankProtocol,
    *,
    prepared_manifest: Mapping[str, Any],
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> dict[str, dict[str, Any]]:
    """Generate ``text_primary`` on the borrowed live 12B engine."""
    if engine is None:
        raise RuntimeError("resident text_primary phase received no 12B engine")
    sender = TEXT_SENDERS[0]
    bound = [
        bind_text_prompt_item(
            tokenizer,
            item,
            sender,
            prepared_manifest=prepared_manifest,
            profile=profile,
        )
        for item in items
    ]
    return {
        str(item["qid"]): generate_report_bundle(
            engine, tokenizer, item, sender, bank, profile=profile
        )
        for item in bound
    }
