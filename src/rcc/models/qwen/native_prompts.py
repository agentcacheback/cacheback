"""Native sender prompt artifacts inside the shared FanOutQA bundle.

Each sender's own worker prompts are rendered into the prepared item, rebuilt
from the primary evidence on load, and read back at generation time.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.fanoutqa.source_padding import ProbeItem
from rcc.benchmarks.fanoutqa.source_topology import token_sequence_sha256
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.qwen.native_prompt_rows import (
    PROMPT_POLICY,
    decode_ids,
    encode_text,
    natural_sender_worker,
    primary_evidence,
    prompt_target,
    render_prompt,
    sender_arm,
    text_hash,
    worker_evidence,
)
from rcc.models.qwen.text_output import TextCompletion
from rcc.models.route import TEXT_SEMANTIC_ARMS, Decoder, RouteFamily

# The schema name banked artifacts already carry; it stays as it is so those
# artifacts still read.
_SCHEMA = "granite-native-sender-prompts-v3"


def _codec_identity(family: RouteFamily, arm: str, profile: BenchmarkProfile) -> dict[str, Any]:
    sender = sender_arm(family, arm)
    codec = family.sender_family(arm)
    return {
        "arm": arm,
        "prompt_policy": PROMPT_POLICY,
        "checkpoint": sender.sender_checkpoint,
        "revision": sender.sender_revision,
        "tokenizer": sender.sender_tokenizer,
        "tokenizer_revision": sender.sender_tokenizer_revision,
        "runtime_profile": family.profile.runtime.runtime_identity_hash,
        "template_mode": {"native_thinking": True, "mechanism": "nemotron-native-template"},
        "system_prompt": codec.thinking_system_prompt,
        "assistant_prefill": codec.assistant_prefill,
        "low_effort": codec.low_effort,
        "prefilled_block": codec.prefilled_block,
        "stop_token_ids": codec.stop_token_ids,
        "think_open_token_ids": codec.think_open_token_ids,
        "think_close_token_ids": codec.think_close_token_ids,
        "literal_think_tags": codec.literal_think_tags,
    }


def codec_identities(
    family: RouteFamily, profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50
) -> dict[str, dict[str, Any]]:
    """Return every text sender's codec identity, the non-byte half of a prompt record."""
    return {arm: _codec_identity(family, arm, profile) for arm in TEXT_SEMANTIC_ARMS}


def build_native_prompt_artifact(
    item: ProbeItem,
    tokenizers: Mapping[str, Any],
    *,
    construction_sha256: str,
    family: RouteFamily,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> dict[str, Any]:
    """Preserve primary evidence under each sender's registered prompt rule."""
    if (
        not family.native_sender_prompts
        or len(item.shards) != 3
        or set(tokenizers) != set(TEXT_SEMANTIC_ARMS)
    ):
        raise RuntimeError(
            f"{item.qid}: Native native preparation lacks its exact sender/worker roster"
        )
    evidence = primary_evidence(item, tokenizers["text_primary"])
    senders: dict[str, Any] = {}
    for arm in TEXT_SEMANTIC_ARMS:
        tokenizer, codec = tokenizers[arm], family.sender_family(arm)
        native = tuple(encode_text(tokenizer, text) for text in evidence)
        if tuple(decode_ids(tokenizer, ids) for ids in native) != evidence:
            raise RuntimeError(
                f"{item.qid}/{arm}: tokenizer changed canonical retained evidence bytes"
            )
        workers: list[dict[str, Any]] = []
        for worker, ids in enumerate(native):
            workers.append(
                natural_sender_worker(
                    item,
                    tokenizer,
                    codec,
                    arm=arm,
                    worker=worker,
                    ids=ids,
                    evidence=evidence[worker],
                    family=family,
                    profile=profile,
                )
            )
        senders[arm] = {"codec": _codec_identity(family, arm, profile), "workers": workers}
    return {
        "schema": _SCHEMA,
        "model_id": family.model_id,
        "qid": item.qid,
        "question_sha256": text_hash(item.question),
        "primary_shard_sha256": tuple(token_sequence_sha256(ids) for ids in item.shards),
        "construction_sha256": construction_sha256,
        "evidence_sha256": tuple(text_hash(text) for text in evidence),
        "senders": senders,
    }


def _tokenizers(family: RouteFamily, *, local_files_only: bool) -> dict[str, Any]:
    from transformers import AutoTokenizer

    factory: Any = AutoTokenizer
    return {
        arm.semantic_arm: factory.from_pretrained(
            arm.sender_tokenizer,
            revision=arm.sender_tokenizer_revision,
            local_files_only=local_files_only,
            use_fast=True,
        )
        for arm in family.profile.physical_arms
        if arm.semantic_arm in TEXT_SEMANTIC_ARMS
    }


def attach_native_prompts(
    items: Sequence[ProbeItem],
    construction: Path,
    *,
    family: RouteFamily,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> tuple[ProbeItem, ...]:
    """Add one per-sender prompt record to each item before the artifact is written."""
    tokenizers = _tokenizers(family, local_files_only=False)
    digest = hashlib.sha256(construction.read_bytes()).hexdigest()
    return tuple(
        replace(
            item,
            native_prompt_artifacts=build_native_prompt_artifact(
                item,
                tokenizers,
                construction_sha256=digest,
                family=family,
                profile=profile,
            ),
        )
        for item in items
    )


def validate_native_prompts(
    items: Sequence[ProbeItem],
    construction: Path,
    *,
    family: RouteFamily,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> None:
    """Rebuild the native prompt records from the primary evidence at load."""
    tokenizers = _tokenizers(family, local_files_only=True)
    digest = hashlib.sha256(construction.read_bytes()).hexdigest()
    for item in items:
        expected = build_native_prompt_artifact(
            item,
            tokenizers,
            construction_sha256=digest,
            family=family,
            profile=profile,
        )
        if item.native_prompt_artifacts != expected:
            raise RuntimeError(
                f"{item.qid}: Native native prompt artifact differs from sealed evidence"
            )


def native_sender_artifact(
    item: ProbeItem,
    *,
    semantic_arm: str,
    family: RouteFamily,
    prompts: Sequence[str] | None = None,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> dict[str, Any]:
    """Read the sender artifact bound to this item and the current prompt policy."""
    artifact = item.native_prompt_artifacts
    if (
        artifact is None
        or artifact.get("schema") != _SCHEMA
        or artifact.get("model_id") != family.model_id
        or artifact.get("qid") != item.qid
        or artifact.get("question_sha256") != text_hash(item.question)
        or artifact.get("primary_shard_sha256")
        != tuple(token_sequence_sha256(ids) for ids in item.shards)
        or semantic_arm not in TEXT_SEMANTIC_ARMS
        or set(artifact.get("senders", {})) != set(TEXT_SEMANTIC_ARMS)
    ):
        raise RuntimeError(f"{item.qid}: missing or foreign Native native prompt artifact")
    sender = artifact["senders"][semantic_arm]
    if (
        sender.get("codec") != _codec_identity(family, semantic_arm, profile)
        or len(sender.get("workers", ())) != 3
    ):
        raise RuntimeError(
            f"{item.qid}/{semantic_arm}: Native sender codec or worker roster differs"
        )
    for worker, row in enumerate(sender["workers"]):
        target = prompt_target(
            row.get("unpadded_prompt_tokens"),
            arm=semantic_arm,
            family=family,
            label=f"{item.qid}/{semantic_arm}/w{worker}",
            profile=profile,
        )
        ids = row.get("prompt_token_ids", ())
        if (
            row.get("prompt_tokens") != target
            or len(ids) != target
            or token_sequence_sha256(ids) != row.get("prompt_token_sha256")
            or row.get("padding") is not None
        ):
            raise RuntimeError(f"{item.qid}/{semantic_arm}: native prompt count or hash drifted")
    if prompts is not None and tuple(map(text_hash, prompts)) != tuple(
        row["prompt_sha256"] for row in sender["workers"]
    ):
        raise RuntimeError(
            f"{item.qid}/{semantic_arm}: submitted native prompt differs from sealed artifact"
        )
    return sender


def native_prompt_counts(
    item: ProbeItem,
    *,
    arm: str,
    family: RouteFamily,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> tuple[int, ...]:
    """Return the banked sender prompt counts for one arm.

    The primary sender's counts also serve the latent and floor arms.
    """
    semantic = arm if arm in TEXT_SEMANTIC_ARMS else "text_primary"
    sender = native_sender_artifact(item, semantic_arm=semantic, family=family, profile=profile)
    return tuple(row["prompt_tokens"] for row in sender["workers"])


def native_worker_prompts(
    item: ProbeItem,
    tokenizer: Any,
    *,
    semantic_arm: str,
    family: RouteFamily,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> tuple[str, ...]:
    """Verify the native prompt bytes and retained evidence, and return the prompts."""
    sender = native_sender_artifact(item, semantic_arm=semantic_arm, family=family, profile=profile)
    artifact = item.native_prompt_artifacts
    assert artifact is not None
    codec = family.sender_family(semantic_arm)
    prompts: list[str] = []
    for worker, row in enumerate(sender["workers"]):
        rendered, kept = worker_evidence(row, worker)
        text = render_prompt(item, tokenizer, rendered, codec)
        ids = encode_text(tokenizer, text)
        retained = decode_ids(tokenizer, kept)
        if (
            ids != row["prompt_token_ids"]
            or len(ids) != row["prompt_tokens"]
            or token_sequence_sha256(ids) != row["prompt_token_sha256"]
            or text_hash(text) != row["prompt_sha256"]
            or text_hash(retained) != row["evidence_sha256"]
            or row["evidence_sha256"] != artifact["evidence_sha256"][worker]
            or len(
                encode_text(
                    tokenizer,
                    render_prompt(item, tokenizer, encode_text(tokenizer, retained), codec),
                )
            )
            != row["unpadded_prompt_tokens"]
        ):
            raise RuntimeError(f"{item.qid}/{semantic_arm}/w{worker}: native prompt bytes drifted")
        prompts.append(text)
    return tuple(prompts)


def consumed_prompt_hashes(
    prompts: Sequence[str],
    results: Sequence[TextCompletion],
    *,
    expected: Sequence[Mapping[str, Any]],
    decoder: Decoder,
) -> tuple[str, ...]:
    """Bind every backend draw to its exact native input before continuation."""
    for prompt, result, row in zip(prompts, results, expected, strict=True):
        if (
            tuple(result.prompt_token_ids) != tuple(row["prompt_token_ids"])
            or len(result.prompt_token_ids) != row["prompt_tokens"]
            or token_sequence_sha256(result.prompt_token_ids) != row["prompt_token_sha256"]
            or text_hash(prompt) != row["prompt_sha256"]
            or decoder(result.prompt_token_ids) != prompt
        ):
            raise RuntimeError("Native backend consumed a different native worker prompt")
    return tuple(token_sequence_sha256(result.prompt_token_ids) for result in results)
