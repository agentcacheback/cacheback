"""Native report and result identity at handoff, completion, and rescore."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.fanoutqa.source_padding import ProbeItem
from rcc.benchmarks.longbench_v2.data import ChainItem
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.qwen.native_prompt_rows import PROMPT_POLICY
from rcc.models.qwen.native_prompts import native_prompt_counts, native_sender_artifact
from rcc.models.route import TEXT_SEMANTIC_ARMS, RouteFamily


def validate_native_report(
    item: ProbeItem | ChainItem,
    fields: Mapping[str, Any],
    *,
    arm: str,
    family: RouteFamily,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> None:
    """Refuse a banked report from a different sender input or codec."""
    from rcc.models.qwen.text import QwenDecodeSpec

    if isinstance(item, ChainItem):
        expected_decode = QwenDecodeSpec("report", family.sender_family(arm), profile=profile)
        if fields.get("report_decode") != expected_decode.to_dict():
            raise RuntimeError(f"{item.qid}/{arm}: Native chain report decode drifted")
        return
    sender = native_sender_artifact(item, semantic_arm=arm, family=family, profile=profile)
    expected = {
        "report_prompt_token_sha256": [row["prompt_token_sha256"] for row in sender["workers"]],
        "report_prompt_tokens_by_worker": [row["prompt_tokens"] for row in sender["workers"]],
        "report_prompt_unpadded_tokens_by_worker": [
            row["unpadded_prompt_tokens"] for row in sender["workers"]
        ],
        "report_prompt_policy": PROMPT_POLICY,
        "report_enable_thinking": True,
        "report_decode": QwenDecodeSpec(
            "report", family.sender_family(arm), profile=profile
        ).to_dict(),
    }
    if any(fields.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"{item.qid}/{arm}: Native report native input or decode drifted")


def validate_native_result(
    item: ProbeItem | ChainItem,
    fields: Mapping[str, Any],
    *,
    arm: str,
    family: RouteFamily,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> None:
    """Bind the sender counts and the answer sampling in a banked native result."""
    if arm in TEXT_SEMANTIC_ARMS:
        validate_native_report(item, fields, arm=arm, family=family, profile=profile)
    profile = family.benchmark_profile(profile)
    if fields.get("decode") != {
        **family.profile.decode.to_dict(),
        "max_tokens": profile.answer_ceiling,
        "stop_token_ids": list(family.stop_token_ids),
    }:
        raise RuntimeError(f"{item.qid}/{arm}: Native answer decode differs")
    if isinstance(item, ChainItem):
        return
    counts = native_prompt_counts(item, arm=arm, family=family, profile=profile)
    if fields.get("worker_prompt_tokens") != list(counts) or fields.get(
        "aggregate_worker_prompt_tokens"
    ) != sum(counts):
        raise RuntimeError(f"{item.qid}/{arm}: native worker prompt counts differ")
