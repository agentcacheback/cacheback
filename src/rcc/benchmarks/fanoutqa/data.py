"""FanOutQA evidence-provided loader: the frozen census, the shard split, the scorers.

``fanoutqa_freeze.json`` fixes the shard split and the census counts. The scorers are a string
proxy with no spaCy lemmatization and no model judge, so they never match published numbers.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rcc.benchmarks.fanoutqa.multihop import audit_dependency_shards
from rcc.benchmarks.fanoutqa.panel import (
    FANOUTQA_DATASET_REVISION,
    FANOUTQA_DEV_URL,
    Question,
    load_freeze,
)
from rcc.benchmarks.fanoutqa.panel import (
    load_questions as _load_questions,
)
from rcc.benchmarks.fanoutqa.scoring import (
    evidence_answerability_audit as _canonical_evidence_answerability_audit,
)
from rcc.benchmarks.fanoutqa.scoring import gold_leaves as _canonical_gold_leaves
from rcc.benchmarks.fanoutqa.scoring import score_text as _canonical_score_text


def load_questions(dev_json_path: str | Path, *, verify_sha: bool = True) -> list[Question]:
    """Load questions while preserving this module's freeze-injection seam."""
    return _load_questions(
        dev_json_path,
        verify_sha=verify_sha,
        freeze_loader=load_freeze,
    )


def real_fanout_structure_audit(
    question: Question,
    plan: ShardPlan,
    *,
    content_pageids: Sequence[int],
) -> dict[str, Any]:
    """Raise unless a constructed item is a disjoint fan-out."""
    decomposition = question.raw.get("decomposition")
    dependency_audit = audit_dependency_shards(decomposition, plan.shards)
    cited = set(dependency_audit.cited_pageids)
    declared = {pageid for pageid, _revid, _title in question.pages}
    content = {int(pageid) for pageid in content_pageids}
    worker_pages = tuple(map(set, plan.shards))
    problems: list[str] = []
    if dependency_audit.max_branch_width < 2:
        problems.append("official decomposition has no fan-out branch")
    if cited != declared:
        problems.append("recursive decomposition pages differ from Question.pages")
    if len(content) < len(worker_pages):
        problems.append("fewer content-bearing evidence pages than evidence workers")
    if any(not pages for pages in worker_pages):
        problems.append("one evidence worker has no page")
    flattened = [page for pages in worker_pages for page in pages]
    if len(flattened) != len(set(flattened)):
        problems.append("the evidence workers share a page")
    if set(flattened) != content:
        problems.append("worker-page union differs from content-bearing cited pages")
    if problems:
        raise RuntimeError(f"{question.qid}: not a valid real FanOutQA handoff: {problems}")
    return {
        "qid": question.qid,
        **dependency_audit.as_row(),
        "content_pages": len(content),
        "workers_per_item": len(worker_pages),
        "worker_page_counts": [len(pages) for pages in worker_pages],
        "worker_pages_disjoint": True,
        "worker_page_union_complete": True,
    }


def _sha_rank(value: object) -> str:
    """Return the sha256 hex digest used as the deterministic ordering tiebreak."""
    return hashlib.sha256(str(value).encode()).hexdigest()


@dataclass(frozen=True)
class ShardPlan:
    """The split decision for one item: page assignment plus trim ledger.

    ``shards`` is one tuple of pageids per worker in assignment order, ``kept`` how
    many tokens of each pageid survive, ``ledger`` the per-page rows.
    """

    shards: tuple[tuple[int, ...], ...]
    kept: dict[int, int]
    ledger: tuple[dict[str, int], ...]


def _balanced_page_shards(content: dict[int, int], workers: int) -> tuple[list[int], ...]:
    """Assign pages longest-first to `workers` capped, load-balanced shards."""
    if workers < 1:
        raise ValueError("shard planning needs at least one worker")
    if len(content) < workers:
        raise ValueError(f"shard planning needs >= {workers} content-bearing pages")
    cap = math.ceil(len(content) / workers)
    order = sorted(content, key=lambda page: (-content[page], _sha_rank(page)))
    shards = tuple([] for _worker in range(workers))
    loads = [0] * workers
    for page in order:
        open_workers = [worker for worker in range(workers) if len(shards[worker]) < cap]
        worker = min(open_workers, key=lambda index: (loads[index], index))
        shards[worker].append(page)
        loads[worker] += content[page]
    return shards


def plan_shards(
    page_tokens: dict[int, int],
    s_tokens: int,
    joiner_len: int = 0,
    *,
    workers: int = 2,
) -> ShardPlan:
    """Assign whole pages to `workers` shards and trim each shard to S tokens.

    Greedy longest-page-first with tail-first overflow trimming; the `longest_tail_trim`
    policy. See docs/benchmarks.md, The frozen census and the shard split.

    Args:
        page_tokens: Token count per pageid, from the frozen pipeline.
        s_tokens: The per-shard budget S.
        joiner_len: Token rows reserved between consecutive pages.
        workers: Number of disjoint evidence workers.

    Returns:
        The `ShardPlan`; both shards' kept totals are <= s_tokens.
    """
    content = {pid: n for pid, n in page_tokens.items() if n > 0}
    shards = _balanced_page_shards(content, workers)
    kept = dict(content)
    for shard in shards:
        # Reserve joiner room so that kept tokens plus joiners meet the budget.
        budget = s_tokens - max(0, len(shard) - 1) * joiner_len
        over = sum(content[page] for page in shard) - budget
        while over > 0:
            pid = max(shard, key=lambda p: (kept[p], _sha_rank(p)))
            cut = min(over, kept[pid] - 1)
            if cut <= 0:
                raise ValueError(
                    f"shard budget {budget} cannot hold {len(shard)} pages "
                    "at one token each; s_tokens is too small for this item"
                )
            kept[pid] -= cut
            over -= cut
    ledger = tuple(
        {
            "pageid": pid,
            "shard": i,
            "tokens": content[pid],
            "kept": kept[pid],
            "trimmed": content[pid] - kept[pid],
        }
        for i in range(workers)
        for pid in shards[i]
    )
    return ShardPlan(shards=tuple(map(tuple, shards)), kept=kept, ledger=ledger)


def max_min_page_tokens(content: dict[int, int], shard: list[int], budget: int) -> dict[int, int]:
    """Allocate an integer content budget lexicographically max-min fairly."""
    if budget < len(shard):
        raise ValueError(
            f"shard budget {budget} cannot hold {len(shard)} pages at one "
            "token each; s_tokens is too small for this item"
        )
    if sum(content[pid] for pid in shard) <= budget:
        return {pid: content[pid] for pid in shard}

    low = 1
    high = max(content[pid] for pid in shard)
    level = 1
    while low <= high:
        middle = (low + high) // 2
        used = sum(min(content[pid], middle) for pid in shard)
        if used <= budget:
            level = middle
            low = middle + 1
        else:
            high = middle - 1

    kept = {pid: min(content[pid], level) for pid in shard}
    remainder = budget - sum(kept.values())
    open_pages = sorted(
        (pid for pid in shard if kept[pid] < content[pid]),
        key=_sha_rank,
    )
    for pid in open_pages[:remainder]:
        kept[pid] += 1
    return kept


def plan_max_min_prefix_shards(
    page_tokens: dict[int, int],
    s_tokens: int,
    joiner_len: int = 0,
    *,
    workers: int = 2,
) -> ShardPlan:
    """Assign pages to `workers` shards and max-min allocate their prefix lengths.

    Same assignment as `plan_shards`; the `max_min_prefix` policy water-fills page prefixes
    instead of trimming. See docs/benchmarks.md, The frozen census and the shard split.
    """
    if joiner_len < 0:
        raise ValueError("joiner_len must be nonnegative")
    content = {pid: n for pid, n in page_tokens.items() if n > 0}
    shards = _balanced_page_shards(content, workers)

    kept: dict[int, int] = {}
    for shard in shards:
        content_budget = s_tokens - max(0, len(shard) - 1) * joiner_len
        kept.update(max_min_page_tokens(content, shard, content_budget))

    ledger = tuple(
        {
            "pageid": pid,
            "shard": i,
            "tokens": content[pid],
            "kept": kept[pid],
            "trimmed": content[pid] - kept[pid],
        }
        for i in range(workers)
        for pid in shards[i]
    )
    return ShardPlan(shards=tuple(map(tuple, shards)), kept=kept, ledger=ledger)


def shard_token_ids(
    plan: ShardPlan,
    page_ids: dict[int, list[int]],
    joiner: list[int],
    s_tokens: int,
) -> tuple[list[int], ...]:
    """Materialize all shards as token-id lists from per-page ids.

    Pages concatenate in assignment order, each trimmed tail-first to its `kept` count and
    separated by `joiner`; a joiner longer than the plan reserved raises only past `s_tokens`.

    Args:
        plan: The split decision from either planner above.
        page_ids: Token ids per pageid under the frozen pipeline.
        joiner: Ids inserted between consecutive pages.
        s_tokens: The hard per-shard budget the result never exceeds.

    Returns:
        One token-id list per worker, each of length <= s_tokens.
    """
    out: list[list[int]] = []
    for shard in plan.shards:
        ids: list[int] = []
        for k, pid in enumerate(shard):
            if k:
                ids.extend(joiner)
            ids.extend(page_ids[pid][: plan.kept[pid]])
        if len(ids) > s_tokens:
            raise AssertionError(
                f"materialized shard {len(ids)} > {s_tokens}: plan_shards was "
                "built with a smaller joiner_len than the joiner supplied"
            )
        out.append(ids)
    return tuple(out)


# Evaluation only below this fence: gold answers are read here and nowhere
# above, where the planners take only page-token geometry.

gold_leaves = _canonical_gold_leaves
evidence_answerability_audit = _canonical_evidence_answerability_audit
score_text = _canonical_score_text

__all__ = (
    "FANOUTQA_DATASET_REVISION",
    "FANOUTQA_DEV_URL",
    "Question",
    "ShardPlan",
    "evidence_answerability_audit",
    "gold_leaves",
    "load_freeze",
    "load_questions",
    "max_min_page_tokens",
    "plan_max_min_prefix_shards",
    "plan_shards",
    "real_fanout_structure_audit",
    "score_text",
    "shard_token_ids",
)
