"""Reference-answer coverage of the handoffs one FanOutQA run delivered.

A text handoff is the three banked worker reports. A latent handoff is the
source tokens the selection kept, decoded one contiguous run at a time with a
boundary between runs, workers, and reports, so a reference string never forms
across a dropped position. Generated latent rows carry no token and are left
out. Each item and arm counts once, not once per receiver draw.
"""

from __future__ import annotations

import hashlib
import json
import sys
from argparse import Namespace
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.fanoutqa.natural_panel import natural_questions
from rcc.benchmarks.fanoutqa.scoring_core import group_hits
from rcc.benchmarks.fanoutqa.scoring_equivalence import (
    equivalent_groups,
    normalize_equivalent_notation,
)
from rcc.models.qwen.capture import selected_indices_sha256
from rcc.run.io import atomic_json, load_jsonl
from rcc.run.plan import ROUTE_MODEL_IDS, ResolvedPlan, route_family
from rcc.run.rescore import route_prepared_items

HANDOFF_RECALL_SCHEMA = "rcc-handoff-recall-v1"
#: Separates decoded runs, workers, and reports; it normalizes to a word no
#: reference contains.
GAP = "\n RCCGAPBOUNDARY \n"
#: The handoff names the families' result builders write; every receiver draw
#: of one cell agrees on all of them.
HANDOFF_FIELDS = ("reports", "selected_indices_sha256", "payload_sha256", "cut_decoded_sha256")
Decode = Callable[[Sequence[int]], str]
Keeps = Sequence[Sequence[int]]
Row = Mapping[str, Any]


def retained_text(ids: Sequence[int], keep: Sequence[int], decode: Decode) -> str:
    """Decode one worker's kept source runs, boundary-separated, dropping latent rows."""
    if list(keep) != sorted(set(keep)) or (len(keep) > 0 and keep[0] < 0):
        raise ValueError("a keep set is ascending, unique, and nonnegative")
    runs: list[str] = []
    start = last = -2
    for index in keep:
        if index >= len(ids):
            break
        if index != last + 1:
            if start >= 0:
                runs.append(decode(ids[start : last + 1]))
            start = index
        last = index
    if start >= 0:
        runs.append(decode(ids[start : last + 1]))
    return GAP.join(runs)


def coverage(groups: Sequence[Sequence[str]], text: str) -> tuple[bool, ...]:
    """Return which reference groups the text carries an accepted form of."""
    return group_hits(tuple(tuple(group) for group in groups), normalize_equivalent_notation(text))


@dataclass(frozen=True)
class Handoffs:
    """One family's read of a run root: worker prompt ids and each row's keeps."""

    decode: Decode
    prompts: Mapping[str, tuple[Sequence[int], ...]]
    #: Raises when a banked row does not belong to the prompts read here.
    verify: Callable[[Row], None]
    keeps: Callable[[Row], Keeps]


def _require(row: Row, field: str, expected: object) -> None:
    if row.get(field) != expected:
        raise RuntimeError(f"{row['qid']}/{row['arm']}: banked {field} differs from the panel")


def _route(plan: ResolvedPlan, run_root: Path) -> Handoffs:
    from transformers import AutoTokenizer

    from rcc.models.qwen.prompts import worker_prompts

    family = route_family(plan.model.model_id)
    tokenizer: Any = AutoTokenizer.from_pretrained(
        family.profile.tokenizer, revision=family.profile.tokenizer_revision, local_files_only=True
    )
    prompts: dict[str, tuple[Sequence[int], ...]] = {}
    hashes: dict[str, list[str]] = {}
    items, manifest = route_prepared_items(plan, run_root)
    for item in items:
        if family.native_sender_prompts:
            workers = item.native_prompt_artifacts["senders"]["text_primary"]["workers"]
            ids = [list(worker["prompt_token_ids"]) for worker in workers]
            sha = [str(worker["prompt_sha256"]) for worker in workers]
        else:
            texts = worker_prompts(item, tokenizer, family=family, profile=plan.benchmark)
            ids = [list(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts]
            sha = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
        prompts[str(item.qid)] = tuple(ids)
        hashes[str(item.qid)] = sha

    def verify(row: Row) -> None:
        qid = str(row["qid"])
        _require(row, "prepared_sha256", str(manifest["artifact_sha256"]))
        _require(row, "worker_prompt_tokens", [len(ids) for ids in prompts[qid]])
        if "keeps_by_worker" in row:
            _require(row, "worker_prompt_sha256", hashes[qid])

    return Handoffs(
        lambda ids: str(tokenizer.decode(ids)), prompts, verify, lambda row: row["keeps_by_worker"]
    )


def _gemma(plan: ResolvedPlan, run_root: Path, bundle: Path) -> Handoffs:
    from rcc.benchmarks.fanoutqa.gemma_natural import load_natural_worker_panel
    from rcc.models.gemma.roster import gemma_v16_arm
    from rcc.models.gemma.tokenizer import load_gemma4_tokenizer
    from rcc.run.gemma.split_build import handoff_root

    tokenizer = load_gemma4_tokenizer()
    panel = load_natural_worker_panel(
        bundle,
        tokenizer,
        offset=plan.item_start,
        count=plan.item_count,
        node_gpus=1,
        worker_index=0,
        profile=plan.benchmark,
    )
    prompts = {
        str(item["qid"]): tuple(ids.tolist() for ids in item["prompt_ids"]) for item in panel.items
    }

    def keeps(row: Row) -> Keeps:
        arm = str(row["arm"])
        path = handoff_root(run_root, arm) / "selections" / f"{row['qid']}.json"
        selection = json.loads(path.read_text(encoding="utf-8"))
        return selection["keeps_by_arm"][gemma_v16_arm(arm)]

    return Handoffs(
        lambda ids: str(tokenizer.decode(ids)),
        prompts,
        lambda row: _require(row, "prepared_sha256", panel.prepared_sha256),
        keeps,
    )


def _ministral(plan: ResolvedPlan, run_root: Path, tokenizer_snapshot: Path | None) -> Handoffs:
    from rcc.benchmarks.fanoutqa.ministral_data import load_ministral_prompt_panel
    from rcc.models.ministral.text_codec import load_registered_text_codec
    from rcc.run.ministral.prepare import prepared_panel_path
    from rcc.run.ministral.split_build import handoff_root
    from rcc.run.ministral.split_producer import handoff_manifest_path

    if tokenizer_snapshot is None:
        raise RuntimeError("Ministral handoff recall requires --tokenizer-snapshot")
    codec = load_registered_text_codec(tokenizer_snapshot, "text_primary")
    panel = load_ministral_prompt_panel(
        prepared_panel_path(run_root, plan.item_start, plan.item_count)
    )
    prompts = {
        qid: tuple(panel.artifact(qid, "text_primary").prompt_ids_by_worker) for qid in panel.qids
    }

    def keeps(row: Row) -> Keeps:
        arm, qid = str(row["arm"]), str(row["qid"])
        path = handoff_manifest_path(handoff_root(run_root, arm), qid, arm)
        meta = json.loads(path.read_text(encoding="utf-8"))["meta"]
        if meta["tensor_sha256"] != row.get("payload_sha256"):
            raise RuntimeError(f"{qid}/{arm}: handoff manifest differs from the banked payload")
        return meta["keeps"]

    return Handoffs(
        lambda ids: str(codec.tokenizer.decode(ids)),
        prompts,
        lambda row: _require(row, "prepared_sha256", panel.fingerprint),
        keeps,
    )


def _cells(run_root: Path, seeds: int) -> dict[tuple[str, str], Row]:
    """Return one result row per item and arm.

    A lane that banks a row per receiver draw must bank every draw, all naming
    the same handoff.
    """
    rows: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for path in sorted(run_root.glob("arms/*/workers/gpu*/raw.jsonl")):
        for row in load_jsonl(path):
            if row.get("kind") == "result":
                rows[(str(row["qid"]), str(row.get("semantic_arm") or row["arm"]))].append(row)
    cells: dict[tuple[str, str], Row] = {}
    for (qid, arm), draws in rows.items():
        if len(draws) > 1 and {draw.get("seed_index") for draw in draws} != set(range(seeds)):
            raise RuntimeError(f"{qid}/{arm}: banked draws are not the registered roster")
        for field in HANDOFF_FIELDS:
            if len({json.dumps(draw.get(field), sort_keys=True) for draw in draws}) != 1:
                raise RuntimeError(f"{qid}/{arm}: receiver draws bank different handoffs")
        cells[(qid, arm)] = draws[0]
    return cells


def _summary(arm: str, found: Mapping[str, Sequence[bool]]) -> dict[str, Any]:
    hits = sum(sum(flags) for flags in found.values())
    leaves = sum(len(flags) for flags in found.values())
    return {
        "arm": arm,
        "questions": len(found),
        "hits": hits,
        "leaves": leaves,
        "micro_recall_percent": 100 * hits / leaves,
        "macro_recall_percent": 100 * mean(sum(flags) / len(flags) for flags in found.values()),
    }


def handoff_recall(
    plan: ResolvedPlan,
    run_root: Path,
    source_bundle: Path,
    *,
    tokenizer_snapshot: Path | None = None,
) -> dict[str, Any]:
    """Score every delivered handoff of one run against the panel's references."""
    if plan.benchmark.scorer != FANOUTQA_NATURAL_DEV50.scorer:
        raise RuntimeError("handoff recall is defined over the FanOutQA fan-out")
    if plan.model.model_id in ROUTE_MODEL_IDS:
        handoffs = _route(plan, run_root)
    elif plan.model.model_id == "gemma4-12b-it":
        handoffs = _gemma(plan, run_root, source_bundle)
    elif plan.model.model_id == "ministral3-14b":
        handoffs = _ministral(plan, run_root, tokenizer_snapshot)
    else:
        raise RuntimeError(f"no handoff reader for {plan.model.model_id}")
    questions = natural_questions(source_bundle, profile=plan.benchmark)
    groups = {qid: equivalent_groups(question) for qid, question in questions.items()}
    channels = {arm.arm_id: arm.channel for arm in plan.arms}
    cells = _cells(run_root, len(plan.benchmark.sample_tags))
    workers = plan.benchmark.workers_per_item
    arms: list[dict[str, Any]] = []
    for arm in sorted({arm for _, arm in cells}):
        if arm not in channels:
            raise RuntimeError(f"{arm}: banked arm is not on the plan")
        if channels[arm] == "floor":
            continue
        found: dict[str, tuple[bool, ...]] = {}
        for qid in plan.execution_question_ids:
            row = cells.get((qid, arm))
            if row is None:
                raise RuntimeError(f"{qid}/{arm}: no banked result row")
            handoffs.verify(row)
            if channels[arm] == "text":
                reports = row["reports"]
                if len(reports) != workers:
                    raise RuntimeError(f"{qid}/{arm}: {len(reports)} reports for {workers} workers")
                text = GAP.join(str(report) for report in reports)
            else:
                keeps = handoffs.keeps(row)
                if selected_indices_sha256(keeps) != row.get("selected_indices_sha256"):
                    raise RuntimeError(f"{qid}/{arm}: keeps differ from the banked selection hash")
                text = GAP.join(
                    retained_text(ids, keep, handoffs.decode)
                    for ids, keep in zip(handoffs.prompts[qid], keeps, strict=True)
                )
            found[qid] = coverage(groups[qid], text)
        arms.append(_summary(arm, found))
    source = {
        qid: coverage(groups[qid], GAP.join(handoffs.decode(ids) for ids in handoffs.prompts[qid]))
        for qid in plan.execution_question_ids
    }
    return {
        "schema": HANDOFF_RECALL_SCHEMA,
        "run_id": plan.run_id,
        "model_id": plan.model.model_id,
        "arms": arms,
        "source": _summary("source", source),
    }


def run_recall_command(args: Namespace, plan: ResolvedPlan) -> int:
    """Score the run under ``--run-root`` and write the summary to ``--out``."""
    if args.run_root is None or args.source_bundle is None or args.out is None:
        raise RuntimeError("handoff-recall requires --run-root, --source-bundle, and --out")
    body = handoff_recall(
        plan, args.run_root, args.source_bundle, tokenizer_snapshot=args.tokenizer_snapshot
    )
    atomic_json(args.out, body)
    sys.stdout.write(json.dumps(body, sort_keys=True) + "\n")
    return 0


__all__ = (
    "HANDOFF_FIELDS",
    "HANDOFF_RECALL_SCHEMA",
    "coverage",
    "handoff_recall",
    "retained_text",
    "run_recall_command",
)
