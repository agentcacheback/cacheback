"""Check a built FanOutQA panel against the widths its geometry assumes.

Every worker prompt must fit the profile's token width, the manager prompt its
ceiling, and every latent ratio must leave the receiver inside its window.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.fanoutqa.prompts import manager_prompt, worker_prompt
from rcc.benchmarks.fanoutqa.qwen_serving import MANAGER_PROMPT_CEILING
from rcc.benchmarks.fanoutqa.source_padding import ProbeItem
from rcc.benchmarks.protocol import BenchmarkProfile


def _prompt_tokens(tokenizer: Any, rendered: str) -> int:
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def _item_geometry(
    item: ProbeItem,
    tokenizer: Any,
    *,
    report_style: str,
    enable_thinking: bool,
    system_prompt: str | None,
    assistant_prefill: str | None,
    low_effort: bool,
    profile: BenchmarkProfile,
) -> list[int]:
    """Measure one item's rendered worker and manager prompts and check their widths."""
    lengths = [
        _prompt_tokens(
            tokenizer,
            worker_prompt(
                tokenizer,
                item.question,
                evidence,
                report_style=report_style,
                enable_thinking=enable_thinking,
                system_prompt=system_prompt,
                assistant_prefill=assistant_prefill,
                low_effort=low_effort,
            ),
        )
        for evidence in item.shards
    ]
    if not profile.admits_worker_prompt_tokens(lengths):
        raise RuntimeError(
            f"{item.qid}: worker prompts are not at most {profile.workers_per_item} "
            f"by {profile.worker_prompt_tokens} tokens ({lengths})"
        )
    manager_tokens = _prompt_tokens(
        tokenizer,
        manager_prompt(
            tokenizer,
            item.question,
            enable_thinking=enable_thinking,
            system_prompt=system_prompt,
            assistant_prefill=assistant_prefill,
            low_effort=low_effort,
        ),
    )
    if manager_tokens > MANAGER_PROMPT_CEILING:
        raise RuntimeError(
            f"{item.qid}: manager prompt {manager_tokens} exceeds ceiling {MANAGER_PROMPT_CEILING}"
        )
    steps = profile.latent_steps
    for ratio in profile.ratios:
        latent = sum((length + steps + ratio - 1) // ratio for length in lengths)
        need = latent + manager_tokens + profile.answer_ceiling
        if need > profile.max_model_len:
            raise RuntimeError(
                f"{item.qid}/r{ratio}: receiver needs {need} tokens, above {profile.max_model_len}"
            )
    return lengths


def actual_geometry_preflight(
    items: Sequence[ProbeItem],
    tokenizer: Any,
    *,
    report_style: str,
    enable_thinking: bool,
    system_prompt: str | None = None,
    assistant_prefill: str | None = None,
    low_effort: bool = False,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> dict[str, list[int]]:
    """Raise unless every rendered prompt and every selected row fits.

    Returns the measured per-worker prompt lengths by qid, so a caller does not
    have to tokenize the worker prompts a second time.
    """
    measured: dict[str, list[int]] = {}
    for item in items:
        measured[item.qid] = _item_geometry(
            item,
            tokenizer,
            report_style=report_style,
            enable_thinking=enable_thinking,
            system_prompt=system_prompt,
            assistant_prefill=assistant_prefill,
            low_effort=low_effort,
            profile=profile,
        )
    return measured


__all__ = ("actual_geometry_preflight",)
