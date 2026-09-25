"""The benchmark, range, and roster one route node resolves for itself.

Every seat reads the same answers here: which lane and benchmark the node runs,
and which question ids its item range holds, from the ``RCC_FANOUT_*`` variables.
"""

from __future__ import annotations

import os
from dataclasses import replace

from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.route import RouteFamily
from rcc.run.plan import registered_benchmark, route_family_for_lane

QWEN_ITEM_START, QWEN_ITEM_COUNT = 0, 15


def environment_family() -> RouteFamily:
    """Return the lane this node runs, Qwen by default."""
    return route_family_for_lane(os.environ.get("RCC_FANOUT_FAMILY", "qwen"))


def _execution_range() -> tuple[int, int]:
    """Read the item range the adapter and the node commands share."""
    raw_start = os.environ.get("RCC_FANOUT_ITEM_START", str(QWEN_ITEM_START))
    raw_count = os.environ.get("RCC_FANOUT_ITEM_COUNT", str(QWEN_ITEM_COUNT))
    if not raw_start.isdigit() or not raw_count.isdigit():
        raise ValueError("FanOutQA item start/count must be nonnegative decimal integers")
    return int(raw_start), int(raw_count)


def execution_profile() -> BenchmarkProfile:
    """Return the benchmark profile this node runs, judger ablation applied.

    ``RCC_FANOUT_JUDGER_QUESTION`` is the one word a launch adds for the
    ablation; every seat reads it here, so the producer, the bank identity,
    and the item-range guard all see the same profile.
    """
    family = environment_family()
    key = os.environ.get("RCC_FANOUT_BENCHMARK") or None
    if key is None:
        raise ValueError("RCC_FANOUT_BENCHMARK must name the benchmark this node runs")
    profile = family.benchmark_profile(registered_benchmark(key))
    question = os.environ.get("RCC_FANOUT_JUDGER_QUESTION")
    if question is None:
        return profile
    if not question.strip():
        raise ValueError("RCC_FANOUT_JUDGER_QUESTION must carry text when it is set")
    return replace(profile, judger_question=question)


def execution_qids(*, profile: BenchmarkProfile) -> tuple[str, ...]:
    """Return the item roster of this node's range on the given profile."""
    start, count = _execution_range()
    if start + count > len(profile.question_ids):
        raise ValueError(
            f"{profile.benchmark_key}: item range {start}:{count} exceeds the "
            f"{len(profile.question_ids)}-question panel"
        )
    return profile.question_ids[start : start + count]
