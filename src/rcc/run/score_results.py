"""Score the banked answers of one FanOutQA run under a named reference set."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.fanoutqa.natural_panel import natural_questions
from rcc.benchmarks.fanoutqa.scoring import REFERENCE_SCORERS
from rcc.run.io import atomic_json, load_jsonl
from rcc.run.plan import ResolvedPlan

SCORE_RESULTS_SCHEMA = "rcc-reference-scores-v1"


def _answer_texts(row: Mapping[str, Any]) -> list[str]:
    """Return the visible answers one row banks: one per seed row, or every sample."""
    visible = row.get("visible_answer_text")
    if visible is not None:
        return [str(visible)]
    return [str(text) for text in row["answer_texts"]]


def score_results(
    plan: ResolvedPlan,
    rows: Sequence[Mapping[str, Any]],
    source_bundle: Path,
    *,
    references: str,
) -> dict[str, Any]:
    """Return per-arm loose and strict accuracy, sample-mean within item then macro over items."""
    if plan.benchmark.scorer != FANOUTQA_NATURAL_DEV50.scorer:
        raise RuntimeError("reference scoring is defined over the FanOutQA fan-out")
    score = REFERENCE_SCORERS[references]
    questions = natural_questions(source_bundle, profile=plan.benchmark)
    texts: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        if row.get("kind") != "result":
            continue
        texts[(str(row["qid"]), str(row.get("semantic_arm") or row["arm"]))].extend(
            _answer_texts(row)
        )
    items: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (qid, arm), answers in texts.items():
        scored = [score(questions[qid], answer) for answer in answers]
        items[arm][qid] = {
            field: mean(task[field] for task in scored) for field in ("loose", "strict")
        }
    arms: dict[str, dict[str, float]] = {}
    for arm, by_qid in sorted(items.items()):
        if set(by_qid) != set(plan.execution_question_ids):
            raise RuntimeError(f"{arm}: banked items differ from the plan's")
        arms[arm] = {
            "items": len(by_qid),
            **{
                field: mean(cell[field] for cell in by_qid.values())
                for field in ("loose", "strict")
            },
        }
    return {
        "schema": SCORE_RESULTS_SCHEMA,
        "run_id": plan.run_id,
        "model_id": plan.model.model_id,
        "references": references,
        "arms": arms,
    }


def run_score_command(args: Namespace, plan: ResolvedPlan) -> int:
    """Score ``--results`` under ``--references`` and write the summary to ``--out``."""
    if args.results is None or args.source_bundle is None or args.out is None:
        raise RuntimeError("score-results requires --results, --source-bundle, and --out")
    body = score_results(
        plan, load_jsonl(args.results), args.source_bundle, references=args.references
    )
    atomic_json(args.out, body)
    sys.stdout.write(json.dumps(body, sort_keys=True) + "\n")
    return 0


__all__ = ("SCORE_RESULTS_SCHEMA", "run_score_command", "score_results")
