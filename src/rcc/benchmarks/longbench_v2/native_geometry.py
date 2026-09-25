"""The measured native request sizes of the four-hop panel, and the check on them.

Every hop, rewrite, and receiver prompt is rendered with the native tokenizers, and
the stored copy is read back only when its pinned digest agrees.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from rcc.benchmarks.longbench_v2.data import ChainItem
from rcc.benchmarks.longbench_v2.geometry import CLOSER_BUDGET
from rcc.benchmarks.longbench_v2.prompts import hop_prompt_ids, retention_prompt, rewrite_prompt
from rcc.benchmarks.longbench_v2.registration import bounded_rows, rerank_ratios
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.nemotron import NEMOTRON_FAMILY
from rcc.models.qwen.backend import QwenBackendSettings, effective_engine_kwargs
from rcc.models.qwen.prompts import manager_prompt_record
from rcc.models.route import TEXT_SEMANTIC_ARMS
from rcc.run.io import sha256_file
from rcc.topologies.chain import BOUNDED_BUDGET_LAW, RERANK_BUDGET_LAW, hop_budget

#: One stored measurement per native profile, keyed by its config key: the
#: counts of every item and the law tables its roster carries. The digest is
#: pinned beside the path, so a file that differs is refused, never read.
GEOMETRY_DIGESTS: dict[str, tuple[Path, str]] = {
    "longbench-v2-coa-nemotron-easy50-text-sealed": (
        Path(__file__).with_name("nemotron_easy50_text_geometry.json"),
        "6a2c58bdfe72ebf9dbf3679f2cdcfb01adc1ec2a3ad5803ff5ed62148aba19e1",
    ),
    "longbench-v2-coa-nemotron-easy50-rerank-sealed": (
        Path(__file__).with_name("nemotron_easy50_rerank_geometry.json"),
        "c8dfb3bce9e248c1df083f92192c5312769724a19544ee6e17227e869664cfc2",
    ),
    "longbench-v2-coa-nemotron-easy50-bounded-sealed": (
        Path(__file__).with_name("nemotron_easy50_bounded_geometry.json"),
        "83340a023351155ab4b010093db36fa9d9b3274189a74ec4905fad3c30ca28d9",
    ),
}


def geometry_digest(profile: BenchmarkProfile) -> tuple[Path, str]:
    """Return the stored measurement path and its pinned digest for one profile."""
    try:
        return GEOMETRY_DIGESTS[profile.benchmark_key]
    except KeyError:
        raise ValueError(f"{profile.benchmark_key}: no native geometry digest") from None


def frozen_geometry(profile: BenchmarkProfile) -> dict[str, Any]:
    """Read the stored native counts, raising unless their pinned digest agrees."""
    path, digest = geometry_digest(profile)
    if sha256_file(path) != digest:
        raise RuntimeError(f"native LongBench geometry digest drifted: {path.name}")
    return cast(dict[str, Any], json.loads(path.read_text()))


def _measure_item(
    item: ChainItem, tokenizers: Mapping[str, Any], profile: BenchmarkProfile
) -> dict[str, Any]:
    tokenizer = tokenizers["text_primary"]
    family = NEMOTRON_FAMILY
    prompts = [
        len(
            hop_prompt_ids(
                tokenizer,
                item.question,
                item.choices,
                chunk,
                hop,
                enable_thinking=True,
                family=family,
            )
        )
        for hop, chunk in enumerate(item.chunks, 1)
    ]
    record = manager_prompt_record(
        item, tokenizer, family=family, profile=profile, channel="latent", payload_slot=True
    )
    return {
        "chunks": list(item.chunk_tokens),
        "hop_prompts": prompts,
        "manager_rows": len(record.token_ids) + sum(map(len, record.payload_headers or ())),
        "retention_rows": len(
            retention_prompt(tokenizer, item.question, enable_thinking=True, family=family)[1]
        ),
        "text": {
            arm: _measure_text(item, tokenizer, tokenizers[arm], profile, arm)
            for arm in TEXT_SEMANTIC_ARMS
        },
    }


def _measure_text(
    item: ChainItem, ledger: Any, tokenizer: Any, profile: BenchmarkProfile, arm: str
) -> dict[str, list[int]]:
    chunks: list[int] = []
    prompts: list[int] = []
    for hop, chunk in enumerate(item.chunks, 1):
        body = str(
            ledger.decode(chunk, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        )
        chunks.append(len(tokenizer.encode(body, add_special_tokens=False)))
        prompt = rewrite_prompt(
            tokenizer,
            item.question,
            item.choices,
            notes="",
            chunk_text=body,
            hop=hop,
            enable_thinking=True,
            builder=profile.prompt_builder,
            family=NEMOTRON_FAMILY.sender_family(arm),
        )
        prompts.append(len(tokenizer.encode(prompt, add_special_tokens=False)))
    return {"chunks": chunks, "prompts": prompts}


def _law_geometry(
    rows: Sequence[dict[str, Any]],
    law: str,
    profile: BenchmarkProfile,
    capture_window: int,
    *,
    ratio: int | None = None,
    budget_rows: int | None = None,
) -> dict[str, int]:
    """Walk the four native hops under one law and price every request.

    A hop reads the carried rows, its own prompt, the latent rows, and the retention
    query. An item is N/A when a request or a budget cannot hold what the cut protects.
    """
    worst_hop = worst_terminal = na_items = 0
    for row in rows:
        carried = 0
        requests: list[int] = []
        latent_fits = True
        for prompt in row["hop_prompts"]:
            length = carried + prompt + profile.latent_steps
            requests.append(length + row["retention_rows"])
            carried = hop_budget(
                law, length, ratio=ratio, prefix_rows=carried, budget_rows=budget_rows
            )
            latent_fits = latent_fits and carried > profile.latent_steps
        terminal = (
            carried
            + row["manager_rows"]
            + profile.answer_ceiling
            + CLOSER_BUDGET
            + len(NEMOTRON_FAMILY.think_close_token_ids)
        )
        worst_hop, worst_terminal = max(worst_hop, *requests), max(worst_terminal, terminal)
        na_items += (
            max(requests) > capture_window or terminal > profile.max_model_len or not latent_fits
        )
    return {
        "worst_hop_rows": worst_hop,
        "worst_terminal_request": worst_terminal,
        "na_items": na_items,
    }


def measure_geometry(
    items: Sequence[ChainItem], tokenizers: Mapping[str, Any], profile: BenchmarkProfile
) -> dict[str, Any]:
    """Measure every native request and raise if an arm cannot run on this panel."""
    family = NEMOTRON_FAMILY
    kwargs = effective_engine_kwargs(
        family.profile.checkpoint,
        family.profile.revision,
        settings=QwenBackendSettings(capture=True, chain=True),
        family=family,
        tokenizer=family.profile.tokenizer,
        tokenizer_revision=family.profile.tokenizer_revision,
    )
    capture_window = int(cast(int, kwargs["max_model_len"]))
    measured = {item.qid: _measure_item(item, tokenizers, profile) for item in items}
    if tuple(measured) != profile.question_ids:
        raise ValueError("native geometry requires the complete registered question roster")
    rows = tuple(measured.values())
    tables: dict[str, dict[str, dict[str, int]]] = {}
    ratios = rerank_ratios(profile)
    if ratios:
        tables["rerank_ratios"] = {
            str(r): _law_geometry(rows, RERANK_BUDGET_LAW, profile, capture_window, ratio=r)
            for r in ratios
        }
    budgets = bounded_rows(profile)
    if budgets:
        tables["bounded_rows"] = {
            str(b): _law_geometry(rows, BOUNDED_BUDGET_LAW, profile, capture_window, budget_rows=b)
            for b in budgets
        }
    text = {
        arm: max(
            prompt
            + profile.report_ceiling * (1 + (hop > 0))
            + CLOSER_BUDGET
            + len(family.sender_family(arm).think_close_token_ids)
            for row in rows
            for hop, prompt in enumerate(row["text"][arm]["prompts"])
        )
        for arm in TEXT_SEMANTIC_ARMS
    }
    if max(text.values()) > profile.max_model_len:
        raise ValueError("native LongBench text requests exceed the registered context window")
    # A law table may carry N/A items rather than raising for the whole profile. The
    # 64K bounded rung does not fit one item on this tokenizer, so the native bounded
    # profile leaves that rung out; the run path refuses an over-window hop.
    return {
        "capture_window": capture_window,
        "receiver_window": profile.max_model_len,
        **tables,
        "text_sender_context": text,
        "hop_wrapper_allowance": max(
            p - c for row in rows for p, c in zip(row["hop_prompts"], row["chunks"], strict=True)
        ),
        "rewrite_wrapper_allowance": max(
            p - c
            for row in rows
            for arm in TEXT_SEMANTIC_ARMS
            for p, c in zip(row["text"][arm]["prompts"], row["text"][arm]["chunks"], strict=True)
        ),
        "items": measured,
    }
