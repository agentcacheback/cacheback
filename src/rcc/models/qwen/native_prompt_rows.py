"""Row-level rendering of native sender prompts under the natural ceiling."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.fanoutqa.prompts import worker_prompt
from rcc.benchmarks.fanoutqa.source_padding import ProbeItem
from rcc.benchmarks.fanoutqa.source_topology import token_sequence_sha256
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.protocol import PhysicalArm
from rcc.models.route import RouteFamily

NATIVE_REPORT_STYLE = "natural"
#: The one registered sender prompt rule: real content under a ceiling.
PROMPT_POLICY = "natural-ceiling"


def sender_arm(family: RouteFamily, arm: str) -> PhysicalArm:
    """Return one registered sender by name, never by roster position."""
    return next(value for value in family.profile.physical_arms if value.semantic_arm == arm)


def prompt_limit(
    *, arm: str, family: RouteFamily, profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50
) -> int:
    """Reserve the complete report head, native closer, and continuation."""
    sender = sender_arm(family, arm)
    codec = family.sender_family(arm)
    serving = int(dict(family.profile.runtime.engine_flags)["max_model_len"])
    profile = family.benchmark_profile(profile)
    context = min(sender.sender_native_max_model_len or 0, serving, profile.max_model_len)
    reserve = profile.report_ceiling + len(codec.think_close_token_ids) + codec.closing_token_budget
    return context - reserve


def prompt_target(
    length: int,
    *,
    arm: str,
    family: RouteFamily,
    label: str,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> int:
    """Return the prompt width this sender must render for one bare prompt length.

    Nothing is padded: the bare prompt is the prompt, and it must fit under the
    registered ceiling and the sender's own context reserve.
    """
    available = prompt_limit(arm=arm, family=family, profile=profile)
    ceiling = min(profile.worker_prompt_tokens, available)
    if type(length) is not int or not 0 < length <= ceiling:
        raise RuntimeError(f"{label}: native prompt {length} exceeds ceiling {ceiling}")
    return length


def text_hash(text: str) -> str:
    """Return the sha256 of one prompt or evidence string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def decode_ids(tokenizer: Any, ids: Sequence[int]) -> str:
    """Decode ids to text without dropping special tokens or cleaning spaces."""
    return str(
        tokenizer.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)
    )


def encode_text(tokenizer: Any, text: str) -> tuple[int, ...]:
    """Encode text to ids with no special tokens added."""
    return tuple(int(token) for token in tokenizer(text, add_special_tokens=False)["input_ids"])


def render_prompt(item: ProbeItem, tokenizer: Any, ids: Sequence[int], codec: RouteFamily) -> str:
    """Render one worker's natural-report prompt from its evidence ids under the sender codec."""
    return worker_prompt(
        tokenizer,
        item.question,
        ids,
        report_style=NATIVE_REPORT_STYLE,
        enable_thinking=codec.profile.decode.enable_thinking,
        system_prompt=codec.thinking_system_prompt,
        assistant_prefill=codec.assistant_prefill,
        low_effort=codec.low_effort,
    )


def worker_evidence(row: Mapping[str, Any], worker: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return one worker's (rendered evidence ids, retained evidence ids).

    A natural row renders and retains the same real evidence ids, so the pair is
    one sequence twice; a row carrying a padding record is refused.
    """
    if row.get("padding") is not None:
        raise RuntimeError(f"Native worker {worker}: padded evidence is no longer registered")
    ids = tuple(int(token) for token in row["evidence_ids"])
    return ids, ids


def natural_sender_worker(
    item: ProbeItem,
    tokenizer: Any,
    codec: RouteFamily,
    *,
    arm: str,
    worker: int,
    ids: Sequence[int],
    evidence: str,
    family: RouteFamily,
    profile: BenchmarkProfile,
) -> dict[str, Any]:
    """Render one sender's worker prompt from real evidence only, under the ceiling."""
    text = render_prompt(item, tokenizer, ids, codec)
    prompt_ids = encode_text(tokenizer, text)
    label = f"{item.qid}/{arm}/w{worker}"
    prompt_target(len(prompt_ids), arm=arm, family=family, label=label, profile=profile)
    return {
        "prompt_token_ids": prompt_ids,
        "prompt_token_sha256": token_sequence_sha256(prompt_ids),
        "prompt_sha256": text_hash(text),
        "prompt_tokens": len(prompt_ids),
        "unpadded_prompt_tokens": len(prompt_ids),
        "unpadded_prompt_sha256": text_hash(text),
        "evidence_sha256": text_hash(evidence),
        "evidence_ids": tuple(ids),
        "padding": None,
    }


def primary_evidence(item: ProbeItem, tokenizer: Any) -> tuple[str, ...]:
    """Return the three worker evidence texts the primary tokenizer produced.

    Every shard is the worker text tokenized once, so the text must round-trip
    through the tokenizer unchanged.
    """
    evidence = tuple(decode_ids(tokenizer, ids) for ids in item.shards)
    if tuple(encode_text(tokenizer, text) for text in evidence) != tuple(
        tuple(ids) for ids in item.shards
    ):
        raise RuntimeError(f"{item.qid}: natural evidence does not round-trip its tokenizer")
    return evidence
