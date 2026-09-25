"""Independent, family-native rescore of banked result rows.

Each family reads its own banked raw token ids back through its own tokenizer,
rebuilds the answer and the reports, and refuses a row they do not reproduce.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol, cast

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.fanoutqa.scoring import scorer_for_rows
from rcc.benchmarks.longbench_v2.prompts import PROMPT_BUILDER, PROMPT_BUILDER_CONDENSE
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.route import RouteFamily
from rcc.run.fleet.merge import scored_fields
from rcc.run.io import atomic_json, load_jsonl
from rcc.run.plan import ROUTE_MODEL_IDS, ResolvedPlan, route_family
from rcc.run.rescore_ceiling import (
    content_token_count,
    injected_flags,
    validate_report_ceiling,
)
from rcc.topologies.chain import HOPS

RESCORE_SUMMARY_SCHEMA = "rcc-raw-token-rescore-summary-v1"
SAMPLE_AGGREGATION = "mean_within_item_arm_then_macro_over_items_no_vote"
FAMILY_RESCORE_RULES = {
    "qwen3-8b": "qwen-post-think-nonempty-v1",
    "nemotron-nano-12b-v2": "nemotron-native-closed-thinking-v1",
    "gemma4-12b-it": "gemma-closed-thinking-substantive-content-v1",
    "ministral3-14b": "ministral-after-last-closer-semantic-content-v2",
}


def expected_rescore_summary(plan: ResolvedPlan) -> dict[str, int]:
    """Return the row grid one rescore of this plan has to have covered."""
    text_arms = sum(arm.channel == "text" for arm in plan.arms)
    grid = plan.execution_expected_rows
    return {
        "rescored_rows": grid.total,
        "rescored_text_rows": grid.items * text_arms * grid.seeds,
        "items": grid.items,
        "arms": grid.arms,
        "seeds": grid.seeds,
    }


Score = Callable[[Any, str], Mapping[str, float]]


def _require_no_vote_macro(
    rows: Sequence[Mapping[str, Any]],
    question_ids: Sequence[str],
    arms: Sequence[str],
    score_fields: Sequence[str],
) -> None:
    """Run the seed-mean then item-macro path for every scored field.

    A profile may carry per-sample records beside its scores, and a macro over
    such a count would show nothing, so this runs over the scored columns alone.
    """
    from rcc.benchmarks.fanoutqa.aggregation import arm_macro

    for arm in arms:
        for field in scored_fields(score_fields):
            if arm_macro(rows, arm, field, question_ids) is None:
                raise RuntimeError(f"{arm}: independent rescore cannot form a {field} macro")


def rescore_gemma_rows(
    rows: Sequence[Mapping[str, Any]],
    plan: ResolvedPlan,
    *,
    questions_by_qid: Mapping[str, Any],
    receiver_tokenizer: Any,
    report_tokenizers: Mapping[str, Any],
    score: Score,
) -> dict[str, int]:
    """Rebuild Gemma answers and reports with their own tokenizers."""
    if plan.model.model_id != "gemma4-12b-it":
        raise RuntimeError("Gemma rescore cannot validate another model family")
    from rcc.benchmarks.fanoutqa.rescore import rescore_seed_rows
    from rcc.models.gemma.contract import STOP_IDS
    from rcc.models.gemma.mechanism import channel_token_ids, visible_answer_tokens
    from rcc.models.gemma.rescore import validate_text_report_row
    from rcc.models.gemma.text_codec import report_is_substantive

    text_arms = {arm.arm_id for arm in plan.arms if arm.channel == "text"}
    if set(report_tokenizers) != text_arms:
        raise RuntimeError("Gemma rescore requires all three sender-native tokenizers")
    channel_open, channel_close = channel_token_ids(receiver_tokenizer)
    summary = rescore_seed_rows(
        rows,
        question_ids=plan.execution_question_ids,
        arm_channels={arm.arm_id: arm.channel for arm in plan.arms},
        sample_tags=plan.benchmark.sample_tags,
        answer_ceiling=plan.benchmark.answer_ceiling,
        questions_by_qid=questions_by_qid,
        decode=lambda tokens: str(receiver_tokenizer.decode(tokens)),
        score=score,
        visible=lambda tokens: visible_answer_tokens(
            tokens,
            channel_open=channel_open,
            channel_close=channel_close,
            stop_ids=STOP_IDS,
        ),
        thinking_closed=lambda tokens: channel_close in tokens or channel_open not in tokens,
        report_accepted=report_is_substantive,
        closer_ids=(channel_close,),
    )
    for row in rows:
        arm = str(row.get("arm") or "")
        if arm not in text_arms:
            continue
        qid = str(row.get("qid") or "")
        tag = row.get("report_seed_tag")
        if not isinstance(tag, str) or tag not in plan.benchmark.sample_tags:
            raise RuntimeError(f"{qid}/{arm}: Gemma report seed tag is unregistered")
        validate_text_report_row(
            row,
            qid=qid,
            semantic_arm=arm,
            tokenizer=report_tokenizers[arm],
            expected_report_seeds=plan.benchmark.report_seeds(qid, tag),
        )
    _require_no_vote_macro(
        rows,
        plan.execution_question_ids,
        tuple(arm.arm_id for arm in plan.arms),
        plan.benchmark.score_fields,
    )
    return summary


def _qwen_report_outputs(
    row: Mapping[str, Any],
    *,
    label: str,
    tokenizer: Any,
    stop_token_ids: Sequence[int],
    field: str,
    count: int,
) -> tuple[list[str], list[int], list[int]]:
    """Decode one sender bundle directly from its raw ids.

    Returns the decoded raw outputs, the banked raw id counts, and the trimmed
    content counts the ceiling is held against, on the topology's own roster.
    """
    raw_token_rows = row.get(field)
    if tokenizer is None or not isinstance(raw_token_rows, list):
        raise RuntimeError(f"{label}: Qwen text report token evidence is incomplete")
    raw_token_roster = cast(list[object], raw_token_rows)
    if len(raw_token_roster) != count:
        raise RuntimeError(f"{label}: Qwen text report token evidence is incomplete")
    raw_outputs: list[str] = []
    token_counts: list[int] = []
    content_counts: list[int] = []
    for raw_ids in raw_token_roster:
        if not isinstance(raw_ids, list):
            raise RuntimeError(f"{label}: Qwen text report token ids are malformed")
        raw_id_values = cast(list[object], raw_ids)
        if any(type(token) is not int for token in raw_id_values):
            raise RuntimeError(f"{label}: Qwen text report token ids are malformed")
        ids = cast(list[int], raw_id_values)
        stop = content_token_count(ids, frozenset(stop_token_ids))
        raw_outputs.append(
            str(
                tokenizer.decode(
                    ids[:stop], skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
            )
        )
        token_counts.append(len(ids))
        content_counts.append(stop)
    return raw_outputs, token_counts, content_counts


def _qwen_report_reasons(row: Mapping[str, Any], *, label: str, count: int) -> list[str]:
    """Return the banked per-draw finish reasons, or refuse incomplete evidence."""
    banked = row.get("report_finish_reasons")
    reasons = cast(list[object], banked) if isinstance(banked, list) else []
    if len(reasons) != count or any(not isinstance(reason, str) for reason in reasons):
        raise RuntimeError(f"{label}: Qwen report termination evidence is incomplete")
    return cast(list[str], reasons)


def _require_qwen_text_report(
    row: Mapping[str, Any],
    plan: ResolvedPlan,
    *,
    qid: str,
    arm: str,
    tokenizer: Any,
    family: RouteFamily,
) -> None:
    """Apply the route's nonempty post-think report rule to the raw ids."""
    from rcc.models.qwen.text import (
        QWEN_REPORT_SEED_TAGS,
        validate_report_ladder_fields,
    )

    family = family.sender_family(arm)
    label = f"{qid}/{arm}"
    raw_outputs, token_counts, content_counts = _qwen_report_outputs(
        row,
        label=label,
        tokenizer=tokenizer,
        stop_token_ids=family.stop_token_ids,
        field="report_token_ids_by_worker",
        count=3,
    )
    raw_reasons = _qwen_report_reasons(row, label=label, count=3)
    validate_report_ceiling(
        content_counts,
        raw_reasons,
        ceiling=plan.benchmark.report_ceiling,
        family="Qwen",
        label=label,
        injected=injected_flags(row, label=label, family="Qwen"),
        closing_extra=len(family.think_close_token_ids) + family.closing_token_budget,
    )
    try:
        validate_report_ladder_fields(row)
    except ValueError as exc:
        raise RuntimeError(f"{label}: {exc}") from exc
    ended = [reason == "stop" for reason in raw_reasons]
    reports = [
        family.post_think_handoff(raw, ended=stop)
        for raw, stop in zip(raw_outputs, ended, strict=True)
    ]
    tag = row.get("report_seed_tag")
    if tag not in QWEN_REPORT_SEED_TAGS:
        raise RuntimeError(f"{label}: Qwen text reports differ from raw post-think ids")
    failed = [
        worker
        for worker, (raw, stop) in enumerate(zip(raw_outputs, ended, strict=True))
        if not family.report_is_accepted(raw, ended=stop)
    ]
    expected = {
        "report_raw_outputs": raw_outputs,
        "reports": reports,
        "report_thinking_closed": [family.think_close in raw for raw in raw_outputs],
        "report_tokens_by_worker": token_counts,
        "report_tokens": sum(token_counts),
        "report_failed": bool(failed),
        "report_failed_workers": failed,
        "report_seeds": list(plan.benchmark.report_seeds(qid, str(tag))),
    }
    if any(row.get(field) != value for field, value in expected.items()):
        raise RuntimeError(f"{label}: Qwen text reports differ from raw post-think ids")


def _require_qwen_notes_chain(
    row: Mapping[str, Any],
    plan: ResolvedPlan,
    *,
    qid: str,
    arm: str,
    tokenizer: Any,
    family: RouteFamily,
) -> None:
    """Rebuild one notes chain from its raw hop ids, refusing any difference.

    The rule is the route's, read one hop at a time. What the chain adds is its
    own roster of hops and the single note the receiver answers from.
    """
    from rcc.models.qwen.chain_text import validate_notes_chain_fields

    family = family.sender_family(arm)
    label = f"{qid}/{arm}"
    raw_outputs, token_counts, content_counts = _qwen_report_outputs(
        row,
        label=label,
        tokenizer=tokenizer,
        stop_token_ids=family.stop_token_ids,
        field="report_token_ids_by_hop",
        count=HOPS,
    )
    reasons = _qwen_report_reasons(row, label=label, count=HOPS)
    try:
        validate_notes_chain_fields(row)
    except ValueError as exc:
        raise RuntimeError(f"{label}: {exc}") from exc
    validate_report_ceiling(
        content_counts,
        reasons,
        ceiling=plan.benchmark.report_ceiling,
        family="Qwen",
        label=label,
        injected=cast(list[bool], row["report_injected_by_hop"]),
        closing_extra=len(family.think_close_token_ids) + family.closing_token_budget,
    )
    ended = [reason == "stop" for reason in reasons]
    notes = [
        family.post_think_handoff(raw, ended=stop)
        for raw, stop in zip(raw_outputs, ended, strict=True)
    ]
    carried = ""
    read: list[str] = []
    for note in notes:
        read.append(carried)
        carried = note or carried
    failed = [
        hop
        for hop, (raw, stop) in enumerate(zip(raw_outputs, ended, strict=True), start=1)
        if not family.report_is_accepted(raw, ended=stop)
    ]
    tags = cast(list[str], row["report_seed_tags_by_hop"])
    expected = {
        "report_raw_outputs": raw_outputs,
        "notes_by_hop": notes,
        "notes_read_by_hop": read,
        "reports": [notes[-1]],
        "report_thinking_closed": [
            not family.profile.decode.enable_thinking or family.think_close in raw
            for raw in raw_outputs
        ],
        "report_tokens_by_hop": token_counts,
        "report_tokens": sum(token_counts),
        "failed_hops": failed,
        "report_failed": bool(failed) or not notes[-1],
        "report_seeds": [
            plan.benchmark.report_seeds(qid, tag)[hop] for hop, tag in enumerate(tags)
        ],
    }
    differs = sorted(field for field, value in expected.items() if row.get(field) != value)
    if differs:
        raise RuntimeError(f"{label}: Qwen notes chain differs from its raw hop ids: {differs}")


class ReportRule(Protocol):
    """A benchmark's raw-id rule over the sender bundle its topology banks."""

    def __call__(
        self,
        row: Mapping[str, Any],
        plan: ResolvedPlan,
        *,
        qid: str,
        arm: str,
        tokenizer: Any,
        family: RouteFamily,
    ) -> None:
        """Refuse a banked sender bundle its own raw ids do not reproduce."""


#: One raw-id sender rule per prompt builder. A benchmark whose builder has no
#: rule is refused by that value before any row is read, rather than by a
#: missing key inside a roster of the wrong width.
_QWEN_REPORT_RULES: dict[str, ReportRule] = {
    FANOUTQA_NATURAL_DEV50.prompt_builder: _require_qwen_text_report,
    PROMPT_BUILDER: _require_qwen_notes_chain,
    PROMPT_BUILDER_CONDENSE: _require_qwen_notes_chain,
}


def _require_registered_report_rule(profile: BenchmarkProfile) -> ReportRule:
    """Return a benchmark's raw-id sender rule, or refuse its builder by name."""
    try:
        return _QWEN_REPORT_RULES[profile.prompt_builder]
    except KeyError:
        raise RuntimeError(
            f"the route rescore implements the {', '.join(_QWEN_REPORT_RULES)} prompt "
            f"builders; {profile.benchmark_key} registers {profile.prompt_builder}"
        ) from None


def rescore_qwen_rows(
    rows: Sequence[Mapping[str, Any]],
    plan: ResolvedPlan,
    *,
    items_by_qid: Mapping[str, Any],
    receiver_tokenizer: Any,
    report_tokenizers: Mapping[str, Any],
) -> dict[str, int]:
    """Rebuild route answers and text reports with each own tokenizer."""
    if plan.model.model_id not in ROUTE_MODEL_IDS:
        raise RuntimeError("Qwen rescore cannot validate another model family")
    from rcc.models.qwen.results import validate_and_rescore_result

    family = route_family(plan.model.model_id)
    if plan.benchmark != family.benchmark_profile(plan.benchmark):
        raise RuntimeError("route rescore plan has unresolved family generation ceilings")
    report_rule = _require_registered_report_rule(plan.benchmark)
    if plan.benchmark.scorer == FANOUTQA_NATURAL_DEV50.scorer:
        scorer_for_rows(rows)
    policy_by_arm = {arm.semantic_arm: arm.policy for arm in family.profile.physical_arms}
    expected = {(qid, arm.arm_id) for qid in plan.execution_question_ids for arm in plan.arms}
    observed: set[tuple[str, str]] = set()
    validated: list[Mapping[str, Any]] = []
    text_rows = 0
    execution_profile = replace(plan.benchmark, question_ids=plan.execution_question_ids)
    for row in rows:
        qid = str(row.get("qid") or "")
        arm = str(row.get("semantic_arm") or "")
        key = (qid, arm)
        if key not in expected or key in observed or row.get("policy") != policy_by_arm.get(arm):
            raise RuntimeError(f"{qid}/{arm}: Qwen result grid identity differs")
        item = items_by_qid.get(qid)
        if item is None:
            raise RuntimeError(f"{qid}/{arm}: sealed panel carries no prepared item")
        observed.add(key)
        validated.append(
            validate_and_rescore_result(
                row,
                item,
                receiver_tokenizer,
                family=family,
                profile=execution_profile,
            )
        )
        channel = next(profile.channel for profile in plan.arms if profile.arm_id == arm)
        if channel != "text":
            continue
        report_rule(
            row,
            plan,
            qid=qid,
            arm=arm,
            tokenizer=report_tokenizers.get(arm),
            family=family,
        )
        text_rows += plan.execution_expected_rows.seeds
    if observed != expected:
        raise RuntimeError(
            f"Qwen rescore grid is incomplete: {len(expected - observed)} cells missing"
        )
    physical_arms = tuple(policy_by_arm[arm.arm_id] for arm in plan.arms)
    _require_no_vote_macro(
        validated, plan.execution_question_ids, physical_arms, plan.benchmark.score_fields
    )
    return {
        "rescored_rows": len(observed) * plan.execution_expected_rows.seeds,
        "rescored_text_rows": text_rows,
        "items": plan.execution_expected_rows.items,
        "arms": plan.execution_expected_rows.arms,
        "seeds": plan.execution_expected_rows.seeds,
    }


def rescore_ministral_rows(
    rows: Sequence[Mapping[str, Any]],
    plan: ResolvedPlan,
    *,
    questions_by_qid: Mapping[str, Any],
    codec: Any,
    report_codecs: Mapping[str, Any],
    score: Score,
) -> dict[str, int]:
    """Rebuild Ministral rows through the sender and receiver codecs."""
    if plan.model.model_id != "ministral3-14b":
        raise RuntimeError("Ministral rescore cannot validate another model family")
    from rcc.models.ministral.results import rescore_result_rows

    by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("qid") or ""), str(row.get("arm") or ""))
        by_cell.setdefault(key, []).append(row)
    expected = {(qid, arm.arm_id) for qid in plan.execution_question_ids for arm in plan.arms}
    if set(by_cell) != expected:
        raise RuntimeError("Ministral rescore grid differs from the resolved result grid")
    text_arms = {arm.arm_id for arm in plan.arms if arm.channel == "text"}
    if set(report_codecs) != text_arms:
        raise RuntimeError("Ministral rescore requires all three sender-native report codecs")
    execution_profile = replace(plan.benchmark, question_ids=plan.execution_question_ids)
    for (qid, arm), cell in by_cell.items():
        question = questions_by_qid.get(qid)
        if question is None:
            raise RuntimeError(f"{qid}: sealed panel carries no question object")
        rescore_result_rows(
            cell,
            codec,
            score=lambda text, q=question: score(q, text),
            report_codec=report_codecs.get(arm),
            profile=execution_profile,
        )
    _require_no_vote_macro(
        rows,
        plan.execution_question_ids,
        tuple(arm.arm_id for arm in plan.arms),
        plan.benchmark.score_fields,
    )
    return expected_rescore_summary(plan)


def _named_paths(values: Sequence[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path or name in paths:
            raise RuntimeError("partition results must be unique partition=path pairs")
        paths[name] = Path(raw_path)
    return paths


def _result_sources(args: Namespace) -> dict[str, Path]:
    if args.results is not None:
        if args.partition_result:
            raise RuntimeError("rescore accepts either --results or split partition results")
        return {"results": args.results}
    sources = _named_paths(args.partition_result)
    if not sources:
        raise RuntimeError("rescore-results requires result rows")
    return sources


def shared_questions(source_bundle: Path, plan: ResolvedPlan) -> dict[str, Any]:
    """Load the panel's questions by qid from a validated source bundle."""
    from rcc.benchmarks.fanoutqa.data import load_questions
    from rcc.benchmarks.fanoutqa.source_bundle import validate_shared_source_bundle

    validate_shared_source_bundle(source_bundle, profile=plan.benchmark)
    by_qid = {
        question.qid: question
        for question in load_questions(source_bundle / "source_cache" / "fanout-final-dev.json")
    }
    missing = set(plan.execution_question_ids) - set(by_qid)
    if missing:
        raise RuntimeError(f"sealed source bundle omits panel questions: {sorted(missing)}")
    return {qid: by_qid[qid] for qid in plan.execution_question_ids}


def route_prepared_items(
    plan: ResolvedPlan, root: Path
) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
    """Load the prepared route panel one run wrote under its root, with its manifest."""
    from rcc.run.qwen.adapter import build_adapter
    from rcc.run.qwen.report import execution_scope

    family = route_family(plan.model.model_id)
    # The prepared loader is the topology's, read through the adapter seam. It
    # selects the pass by the benchmark and range a seat reads from the
    # environment, so the plan's own pair is published for this read alone.
    with execution_scope(plan.benchmark, (plan.item_start, plan.item_count)):
        return build_adapter(family, profile=plan.benchmark).data.load_prepared(
            root, panel="production", source_commit=plan.benchmark.source_commit
        )


def _rescore_qwen_command(
    args: Namespace,
    plan: ResolvedPlan,
    sources: Mapping[str, Path],
) -> dict[str, int]:
    from transformers import AutoTokenizer

    family = route_family(plan.model.model_id)
    profile = family.profile
    runtime_roots = (
        {"results": args.runtime_bank}
        if args.runtime_bank is not None
        else _named_paths(args.partition_runtime_bank)
    )
    if set(runtime_roots) != set(sources) or any(path is None for path in runtime_roots.values()):
        raise RuntimeError("Qwen rescore requires one prepared runtime bank per result source")
    items, _ = route_prepared_items(plan, cast(Path, runtime_roots[sorted(runtime_roots)[0]]))
    tokenizer_class: Any = AutoTokenizer
    receiver_tokenizer = tokenizer_class.from_pretrained(
        profile.tokenizer,
        revision=profile.tokenizer_revision,
        local_files_only=True,
    )
    text_arms = {spec.arm_id for spec in plan.arms if spec.channel == "text"}
    report_tokenizers: dict[str, Any] = {}
    for arm in profile.physical_arms:
        if arm.semantic_arm not in text_arms:
            continue
        if arm.sender_tokenizer is None or arm.sender_tokenizer_revision is None:
            raise RuntimeError(f"{arm.semantic_arm}: route text sender tokenizer is unregistered")
        report_tokenizers[arm.semantic_arm] = tokenizer_class.from_pretrained(
            arm.sender_tokenizer,
            revision=arm.sender_tokenizer_revision,
            local_files_only=True,
        )
    rows = [row for path in sources.values() for row in load_jsonl(path)]
    return rescore_qwen_rows(
        rows,
        plan,
        items_by_qid={item.qid: item for item in items},
        receiver_tokenizer=receiver_tokenizer,
        report_tokenizers=report_tokenizers,
    )


def run_rescore_command(args: Namespace, plan: ResolvedPlan) -> int:
    """Rescore the banked rows with the family's own reader and write a summary."""
    if args.rescore_out is None:
        raise RuntimeError("rescore-results requires --rescore-out")
    sources = _result_sources(args)
    rows = [row for path in sources.values() for row in load_jsonl(path)]
    if plan.model.model_id in ROUTE_MODEL_IDS:
        summary = _rescore_qwen_command(args, plan, sources)
    else:
        if args.source_bundle is None:
            raise RuntimeError("Gemma/Ministral rescore requires --source-bundle")
        score = scorer_for_rows(rows)
        questions = shared_questions(args.source_bundle, plan)
        if plan.model.model_id == "gemma4-12b-it":
            from rcc.models.gemma.text_contract import registered_text_senders
            from rcc.models.gemma.tokenizer import load_gemma4_tokenizer

            report_tokenizers = {
                sender.semantic_arm: load_gemma4_tokenizer(
                    checkpoint_id=sender.checkpoint,
                    revision=sender.revision,
                )
                for sender in registered_text_senders()
            }

            summary = rescore_gemma_rows(
                rows,
                plan,
                questions_by_qid=questions,
                receiver_tokenizer=report_tokenizers["text_primary"],
                report_tokenizers=report_tokenizers,
                score=score,
            )
        elif plan.model.model_id == "ministral3-14b":
            if args.tokenizer_snapshot is None:
                raise RuntimeError("Ministral rescore requires --tokenizer-snapshot")
            from rcc.models.ministral.text_codec import load_registered_text_codec

            text_arms = tuple(arm.arm_id for arm in plan.arms if arm.channel == "text")

            summary = rescore_ministral_rows(
                rows,
                plan,
                questions_by_qid=questions,
                codec=load_registered_text_codec(args.tokenizer_snapshot, "text_primary"),
                report_codecs={
                    arm: load_registered_text_codec(args.tokenizer_snapshot, arm)
                    for arm in text_arms
                },
                score=score,
            )
        else:
            raise RuntimeError(f"no raw-token rescore runner for {plan.model.model_id}")
    body: dict[str, Any] = {
        "schema": RESCORE_SUMMARY_SCHEMA,
        "run_id": plan.run_id,
        "model_id": plan.model.model_id,
        "visible_answer_rule": FAMILY_RESCORE_RULES[plan.model.model_id],
        "sample_aggregation": SAMPLE_AGGREGATION,
        "result_sources": sorted(sources),
        **summary,
    }
    atomic_json(args.rescore_out, body)
    sys.stdout.write(json.dumps(body, sort_keys=True) + "\n")
    return 0


__all__ = (
    "FAMILY_RESCORE_RULES",
    "RESCORE_SUMMARY_SCHEMA",
    "SAMPLE_AGGREGATION",
    "content_token_count",
    "rescore_gemma_rows",
    "rescore_ministral_rows",
    "rescore_qwen_rows",
    "route_prepared_items",
    "run_rescore_command",
    "shared_questions",
    "validate_report_ceiling",
)
