"""How a worker's share of the cited pages is chosen and materialized.

Page ownership always comes from the frozen eligibility census. See
docs/benchmarks.md, Query-coverage spans.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from rcc.benchmarks.fanoutqa.data import (
    ShardPlan,
    max_min_page_tokens,
    plan_max_min_prefix_shards,
    plan_shards,
    shard_token_ids,
)
from rcc.run.io import is_sha256_hex

BankRow = Callable[[dict[str, Any]], None]
PagePlanner = Callable[..., ShardPlan]
QUERY_COVERAGE_CHUNK_TOKENS = 256
_QUERY_STOPWORDS = {
    "after",
    "and",
    "are",
    "before",
    "current",
    "did",
    "does",
    "five",
    "for",
    "from",
    "have",
    "how",
    "last",
    "many",
    "most",
    "that",
    "the",
    "their",
    "this",
    "top",
    "were",
    "what",
    "which",
    "who",
    "with",
    "year",
    "years",
}


def source_page_provenance(
    *,
    qid: str,
    pageid: int,
    revid: int,
    raw_text: str,
    canonical_text: str,
    actual_tokens: int,
    census_tokens: int,
) -> dict[str, Any]:
    """Return one cited revision's construction ledger row."""
    return {
        "kind": "source_page",
        "qid": qid,
        "pageid": pageid,
        "revid": revid,
        "raw_markdown_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "canonical_markdown_sha256": hashlib.sha256(canonical_text.encode("utf-8")).hexdigest(),
        "canonical_tokens": actual_tokens,
        "frozen_census_tokens": census_tokens,
        "census_delta_tokens": actual_tokens - census_tokens,
    }


def validated_source_ledger_sha256(
    path: Path,
    expected_pages: Sequence[tuple[str, int, int]],
) -> str:
    """Check every page provenance row and its order, then hash the whole ledger."""
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("FanOutQA construction ledger is not valid JSONL") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("FanOutQA construction ledger must contain JSON objects")
    source_rows = [row for row in rows if row.get("kind") == "source_page"]
    identities: list[tuple[str, int, int]] = []
    for row in source_rows:
        qid = row.get("qid")
        pageid = row.get("pageid")
        revid = row.get("revid")
        canonical_tokens = row.get("canonical_tokens")
        census_tokens = row.get("frozen_census_tokens")
        delta = row.get("census_delta_tokens")
        hashes = (row.get("raw_markdown_sha256"), row.get("canonical_markdown_sha256"))
        if (
            not isinstance(qid, str)
            or not qid
            or not isinstance(pageid, int)
            or isinstance(pageid, bool)
            or not isinstance(revid, int)
            or isinstance(revid, bool)
            or not isinstance(canonical_tokens, int)
            or isinstance(canonical_tokens, bool)
            or canonical_tokens <= 0
            or not isinstance(census_tokens, int)
            or isinstance(census_tokens, bool)
            or census_tokens <= 0
            or not isinstance(delta, int)
            or isinstance(delta, bool)
            or delta != canonical_tokens - census_tokens
            or any(not is_sha256_hex(value) for value in hashes)
        ):
            raise RuntimeError("FanOutQA construction ledger has invalid source provenance")
        identities.append((qid, pageid, revid))
    if tuple(identities) != tuple(expected_pages):
        raise RuntimeError("FanOutQA construction ledger has the wrong ordered source roster")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _query_terms(question: str) -> tuple[set[str], set[str]]:
    words = re.findall(r"[a-z0-9]+", question.lower())
    terms = {word for word in words if len(word) >= 3 and word not in _QUERY_STOPWORDS}
    numbers = {
        "".join(character for character in word if character.isdigit())
        for word in words
        if any(character.isdigit() for character in word)
    }
    return terms, numbers


def _query_chunks(
    tokenizer: Any, token_ids: Sequence[int], question: str
) -> tuple[list[list[int]], list[tuple[tuple[int, int, int, int], int]]]:
    chunks = [
        list(token_ids[start : start + QUERY_COVERAGE_CHUNK_TOKENS])
        for start in range(0, len(token_ids), QUERY_COVERAGE_CHUNK_TOKENS)
    ]
    terms, numbers = _query_terms(question)
    decoded = [str(tokenizer.decode(chunk, skip_special_tokens=False)).lower() for chunk in chunks]
    scored: list[tuple[tuple[int, int, int, int], int]] = []
    for index, text in enumerate(decoded):
        # Table rows and their headers cross the 256-token boundary, so a chunk is
        # ranked over its one-chunk neighborhood. No neighbor is selected automatically
        # and no answer field is read.
        context = " ".join(decoded[max(0, index - 1) : index + 2])
        direct_words = set(re.findall(r"[a-z0-9]+", text))
        context_words = re.findall(r"[a-z0-9]+", context)
        words = set(context_words)
        direct_numbers = {
            "".join(character for character in word if character.isdigit())
            for word in direct_words
            if any(character.isdigit() for character in word)
        }
        chunk_numbers = {
            "".join(character for character in word if character.isdigit())
            for word in words
            if any(character.isdigit() for character in word)
        }
        score = (
            10 * len(numbers & direct_numbers) + len(terms & direct_words),
            10 * len(numbers & chunk_numbers) + len(terms & words),
            sum(word in terms for word in context_words),
            text.count("|"),
        )
        scored.append((score, index))
    return chunks, scored


def select_query_covered_tokens(
    tokenizer: Any,
    token_ids: Sequence[int],
    kept: int,
    question: str,
) -> tuple[list[list[int]], int]:
    """Select `kept` tokens: the first half from the head, the rest by ranked chunk."""
    if kept < 1 or kept > len(token_ids):
        raise ValueError(f"kept must be in 1..{len(token_ids)}, got {kept}")
    if kept == len(token_ids):
        return [list(token_ids)], kept
    head_tokens = (kept + 1) // 2
    remaining = kept - head_tokens
    head = list(token_ids[:head_tokens])
    chunks, scored = _query_chunks(tokenizer, token_ids[head_tokens:], question)
    selected: dict[int, list[int]] = {}
    for _score, index in sorted(
        scored,
        key=lambda item: (item[0], -item[1]),
        reverse=True,
    ):
        take = min(remaining, len(chunks[index]))
        if take:
            selected[index] = list(chunks[index][:take])
            remaining -= take
        if remaining == 0:
            break
    if remaining:
        raise AssertionError("query coverage selected insufficient content capacity")
    selected_chunks = [head, *(selected[index] for index in sorted(selected))]
    if sum(map(len, selected_chunks)) != kept:
        raise AssertionError("query coverage did not materialize its exact content quota")
    return selected_chunks, kept


def reserved_query_segment_count(content_tokens: int, kept: int) -> int:
    """Return the separator count to reserve: the head plus the worst-case tail chunks."""
    if kept == content_tokens:
        return 1
    head_tokens = (kept + 1) // 2
    tail_budget = kept - head_tokens
    if tail_budget == 0:
        return 1
    # A high-ranked short final chunk can need one further selection to fill the
    # tail budget exactly. The bound is not capped by the available tail chunks:
    # staying monotone until the page is complete makes the interval search exact.
    selected_tail_chunks = (
        tail_budget + QUERY_COVERAGE_CHUNK_TOKENS - 1
    ) // QUERY_COVERAGE_CHUNK_TOKENS + 1
    return 1 + selected_tail_chunks


def plan_max_min_query_shards(
    page_tokens: dict[int, int],
    s_tokens: int,
    joiner_len: int = 0,
    *,
    assignment_tokens: dict[int, int] | None = None,
    workers: int = 2,
) -> ShardPlan:
    """Assign pages from the census, then max-min allocate query-covered quotas."""
    content = {pid: count for pid, count in page_tokens.items() if count > 0}
    assignment = (
        content
        if assignment_tokens is None
        else {pid: count for pid, count in assignment_tokens.items() if count > 0}
    )
    if set(content) != set(assignment):
        raise ValueError("live and assignment token ledgers must name the same nonempty pages")
    base = plan_max_min_prefix_shards(assignment, s_tokens, joiner_len, workers=workers)
    kept: dict[int, int] = {}
    chunk_counts: dict[int, int] = {}
    for shard_tuple in base.shards:
        shard = list(shard_tuple)
        page_joiners = max(0, len(shard) - 1) * joiner_len
        minimum_budget = len(shard)
        maximum_budget = s_tokens - page_joiners

        def candidate_for(budget: int, shard: list[int] = shard) -> dict[int, int]:
            return max_min_page_tokens(content, shard, budget)

        def feasible(
            budget: int,
            shard: list[int] = shard,
            page_joiners: int = page_joiners,
        ) -> bool:
            candidate = candidate_for(budget)
            internal_joiners = sum(
                max(0, reserved_query_segment_count(content[pid], candidate[pid]) - 1) * joiner_len
                for pid in shard
                if candidate[pid] < content[pid]
            )
            return sum(candidate.values()) + page_joiners + internal_joiners <= s_tokens

        # Separator demand is monotone between page-saturation points, but drops when a
        # page becomes complete, since the full article is one contiguous segment. Search
        # each monotone interval so a feasible full page cannot hide.
        boundaries = {minimum_budget, maximum_budget + 1}
        for pid in shard:
            if candidate_for(maximum_budget)[pid] < content[pid]:
                continue
            low = minimum_budget
            high = maximum_budget
            while low < high:
                middle = (low + high) // 2
                if candidate_for(middle)[pid] == content[pid]:
                    high = middle
                else:
                    low = middle + 1
            boundaries.add(low)

        best_budget: int | None = None
        ordered_boundaries = sorted(boundaries)
        for start, stop in pairwise(ordered_boundaries):
            low = start
            high = stop - 1
            interval_best: int | None = None
            while low <= high:
                budget = (low + high) // 2
                if feasible(budget):
                    interval_best = budget
                    low = budget + 1
                else:
                    high = budget - 1
            if interval_best is not None and (best_budget is None or interval_best > best_budget):
                best_budget = interval_best
        if best_budget is None:
            raise ValueError("source budget cannot hold one token from every page")
        allocation = candidate_for(best_budget)
        if not feasible(best_budget):
            raise AssertionError("query planner selected an infeasible allocation")
        kept.update(allocation)
        chunk_counts.update(
            {pid: (reserved_query_segment_count(content[pid], allocation[pid])) for pid in shard}
        )
    ledger = tuple(
        {
            "pageid": pid,
            "shard": index,
            "tokens": content[pid],
            "kept": kept[pid],
            "trimmed": content[pid] - kept[pid],
            "reserved_segments": chunk_counts[pid],
        }
        for index in range(workers)
        for pid in base.shards[index]
    )
    return ShardPlan(shards=base.shards, kept=kept, ledger=ledger)


def shard_query_token_ids(
    tokenizer: Any,
    question: str,
    plan: ShardPlan,
    page_ids: dict[int, list[int]],
    joiner: list[int],
    s_tokens: int,
) -> tuple[list[int], ...]:
    """Materialize head-plus-query coverage in original page order."""
    out: list[list[int]] = []
    for shard in plan.shards:
        ids: list[int] = []
        for page_index, pid in enumerate(shard):
            if page_index:
                ids.extend(joiner)
            chunks, _kept = select_query_covered_tokens(
                tokenizer,
                page_ids[pid],
                plan.kept[pid],
                question,
            )
            for chunk_index, chunk in enumerate(chunks):
                if chunk_index:
                    ids.extend(joiner)
                ids.extend(chunk)
        if len(ids) > s_tokens:
            raise AssertionError(
                f"materialized query shard {len(ids)} > {s_tokens}: planner "
                "did not reserve its internal joiners"
            )
        out.append(ids)
    return tuple(out)


_PAGE_ALLOCATORS: dict[str, PagePlanner] = {
    "longest_tail_trim": plan_shards,
    "max_min_prefix": plan_max_min_prefix_shards,
    "max_min_query": plan_max_min_query_shards,
}


def page_allocator(name: str) -> PagePlanner:
    """Return the planner one page-allocation policy names."""
    try:
        return _PAGE_ALLOCATORS[name]
    except KeyError as exc:
        raise ValueError(f"unknown page allocation {name!r}") from exc


def plan_source_shards(
    name: str,
    live_tokens: dict[int, int],
    census_tokens: dict[int, int],
    s_tokens: int,
    joiner_len: int,
    *,
    workers: int = 2,
) -> ShardPlan:
    """Plan one item's shards: census ownership, quotas bound to the live bytes.

    ``max_min_query`` allocates within the census-owned workers from the canonical live
    counts; the other two never record or request more tokens than the fetch produced.
    """
    if name == "max_min_query":
        return plan_max_min_query_shards(
            live_tokens,
            s_tokens,
            joiner_len,
            assignment_tokens=census_tokens,
            workers=workers,
        )
    live = {pid: count for pid, count in live_tokens.items() if count > 0}
    census = {pid: count for pid, count in census_tokens.items() if count > 0}
    if set(live) != set(census):
        raise ValueError("live and assignment token ledgers must name the same nonempty pages")
    planner = page_allocator(name)
    base = planner(census, s_tokens, joiner_len, workers=workers)
    # The census owns the split and the water level, but only the canonical live
    # bytes can be materialized, so the quota is bound to them and the ledger
    # never claims tokens the fetched page does not hold.
    kept = {pid: min(base.kept[pid], live[pid]) for pid in base.kept}
    ledger = tuple(
        {
            "pageid": pid,
            "shard": index,
            "tokens": live[pid],
            "kept": kept[pid],
            "trimmed": live[pid] - kept[pid],
        }
        for index in range(workers)
        for pid in base.shards[index]
    )
    return ShardPlan(shards=base.shards, kept=kept, ledger=ledger)


def materialize_shards(
    name: str,
    plan: ShardPlan,
    page_ids: dict[int, list[int]],
    joiner: list[int],
    s_tokens: int,
    *,
    tokenizer: Any | None = None,
    question: str | None = None,
) -> tuple[list[int], ...]:
    """Materialize one allocation with the token geometry its policy defines."""
    if name == "max_min_query":
        if tokenizer is None or question is None:
            raise ValueError("max_min_query materialization requires tokenizer and question")
        return shard_query_token_ids(tokenizer, question, plan, page_ids, joiner, s_tokens)
    if name in {"longest_tail_trim", "max_min_prefix"}:
        return shard_token_ids(plan, page_ids, joiner, s_tokens)
    raise ValueError(f"unknown page allocation {name!r}")
