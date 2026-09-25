"""Offline raw-token validation and aggregation for one route lane's arms.

The banked rows are read through the shared split-bank merge, rescored from their
raw tokens, macroed over items, and written under ``report/``, once per profile.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.fanoutqa.prepare import load_prepared_panel
from rcc.benchmarks.fanoutqa.scoring import scoring_version_for_rows
from rcc.benchmarks.fanoutqa.source_audit import validate_source_commit
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.qwen.results import headline_score_field, validate_and_rescore_result
from rcc.models.route import RouteFamily
from rcc.run.fleet.merge import (
    SCORE_FIELDS,
    add_stage_means,
    paired_deltas,
    read_split_result_rows,
    score_grid,
    scored_fields,
)
from rcc.run.fleet.report import add_paper_metrics
from rcc.run.io import atomic_bytes
from rcc.run.plan import registered_benchmark
from rcc.run.qwen.adapter import build_adapter, environment_family, execution_profile
from rcc.run.qwen.chain_adapter import ChainDataAdapter
from rcc.run.qwen.chain_report import (
    chain_bootstrap,
    chain_failures,
    chain_floors,
    chain_strata,
    sealed_panel_golds,
)
from rcc.topologies.chain import CHAIN_TOPOLOGY_KEY

JsonRow = dict[str, Any]

#: The report schema stem each topology publishes under.
_REPORT_SCHEMAS = {
    FANOUTQA_NATURAL_DEV50.topology_key: "fanoutqa-report-v1",
    CHAIN_TOPOLOGY_KEY: "longbench-coa-report-v1",
}


def report_schema(profile: BenchmarkProfile, lane: str) -> str:
    """Return the report schema one lane publishes this benchmark under."""
    try:
        stem = _REPORT_SCHEMAS[profile.topology_key]
    except KeyError:
        raise ValueError(
            f"the {lane} report publishes the {', '.join(_REPORT_SCHEMAS)} topologies; "
            f"{profile.benchmark_key} registers {profile.topology_key!r}"
        ) from None
    return f"{lane}-{stem}"


def _load_panel(
    root: Path,
    *,
    panel: str,
    source_commit: str,
    profile: BenchmarkProfile,
    family: RouteFamily,
) -> tuple[tuple[Any, ...], Mapping[str, Any], tuple[Any, ...]]:
    """Load this topology's prepared items, or refuse the topology by name.

    Returns the items the run scored, the prepared manifest, and the whole fixed
    roster. FanOutQA fixes one panel per pass, so those two rosters are one.
    """
    if profile.topology_key == FANOUTQA_NATURAL_DEV50.topology_key:
        items, manifest = load_prepared_panel(
            root,
            source_commit=source_commit,
            profile=profile,
            family=family,
        )
        return items, manifest, items
    # `build_adapter` refuses an unknown topology, so the only remaining case
    # is the chain loader, whose fixed roster the lane's data protocol does not
    # declare because only a subsetting loader holds one.
    data = cast(ChainDataAdapter, build_adapter(family, profile=profile).data)
    items, manifest = data.load_prepared(root, panel=panel, source_commit=source_commit)
    return items, manifest, data.sealed_items


def read_qwen_result_rows(root: Path, *, family: RouteFamily) -> list[JsonRow]:
    """Read every complete result row and republish it on its arm's fleet clock.

    The read itself is the shared split-bank merge. The lane contributes its
    placement table, which belongs to the topology the arm was seated on.
    """
    adapter = build_adapter(family, profile=execution_profile())
    return read_split_result_rows(
        root,
        adapter.placements.registered,
        label=family.lane.capitalize(),
    )


#: The per-item means the policy grid publishes beside its scored columns.
QWEN_GRID_MEANS = ("generated_tokens",)


def _policy_grid(
    rows: Sequence[JsonRow],
    policies: Sequence[str],
    label: str,
    score_fields: Sequence[str] = SCORE_FIELDS,
) -> list[JsonRow]:
    grid = score_grid(
        rows,
        policies,
        key="policy",
        mean_fields=QWEN_GRID_MEANS,
        label=label,
        score_fields=score_fields,
    )
    add_paper_metrics(grid, rows, key="policy")
    return grid


def _comparison_pairs(
    policies: Sequence[str], family: RouteFamily, profile: BenchmarkProfile
) -> tuple[tuple[str, str], ...]:
    """Return each latent ratio's references, in reference order.

    The fused text arm is every benchmark's reference. The chain adds the floor
    arm as a second; a fan-out arm also takes its mean query attention mate at that ratio.
    """
    arms = family.profile.physical_arms
    by_semantic = {arm.policy: arm.semantic_arm for arm in arms}
    policy_of = {semantic: policy for policy, semantic in by_semantic.items()}
    references = [next(arm.policy for arm in arms if arm.semantic_arm == "text_primary")]
    if profile.topology_key == CHAIN_TOPOLOGY_KEY:
        references.append(next(arm.policy for arm in arms if arm.semantic_arm == "issue_only"))
    pairs: list[tuple[str, str]] = []
    roster = set(policies)
    for ratio in profile.ratios:
        snap = policy_of.get(f"latent_qsnap_r{ratio}")
        support = policy_of.get(f"latent_query_support_r{ratio}")
        if support is None or support not in roster:
            continue
        if snap is not None and snap in roster:
            pairs.append((snap, support))
        pairs.extend((reference, support) for reference in references if reference in roster)
    for arm in profile.arms:
        if arm.budget_law is None:
            continue
        lawful = policy_of[arm.arm_id]
        if lawful not in roster:
            continue
        pairs.extend((reference, lawful) for reference in references if reference in roster)
    return tuple(pairs)


def _paired(
    rows: Sequence[JsonRow], pairs: Sequence[tuple[str, str]], score_fields: Sequence[str]
) -> list[JsonRow]:
    return paired_deltas(rows, pairs, key="policy", score_fields=score_fields)


def _censoring(rows: Sequence[JsonRow], policies: Sequence[str]) -> dict[str, Any]:
    by_policy: dict[str, dict[str, int]] = {}
    for policy in policies:
        selected = [row for row in rows if row.get("policy") == policy]
        by_policy[policy] = {
            "samples": sum(len(cast(list[object], row["finish_reasons"])) for row in selected),
            "unclosed_samples": sum(
                not bool(value)
                for row in selected
                for value in cast(list[object], row["thinking_closed_by_sample"])
            ),
            "length_finished_samples": sum(
                value == "length"
                for row in selected
                for value in cast(list[object], row["finish_reasons"])
            ),
        }
    censored = any(
        counts["unclosed_samples"] or counts["length_finished_samples"]
        for counts in by_policy.values()
    )
    return {
        "status": "output_censored" if censored else "complete_within_caps",
        "rule": "retain_and_score_all_three_independent_samples_no_redraw",
        "policies": by_policy,
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        return b""
    buffer = io.StringIO()
    fields = tuple(rows[0])
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def _validate_grid(
    rows: Sequence[JsonRow], label: str
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    panels = {str(row.get("panel") or "") for row in rows}
    if len(panels) != 1 or "" in panels:
        raise RuntimeError(f"{label} report rows do not share one panel")
    panel = panels.pop()
    qids = tuple(dict.fromkeys(str(row.get("qid") or "") for row in rows))
    policies = tuple(dict.fromkeys(str(row.get("policy") or "") for row in rows))
    if "" in qids or "" in policies:
        raise RuntimeError(f"{label} report row lacks item or policy identity")
    observed = [(str(row["qid"]), str(row["policy"])) for row in rows]
    if len(observed) != len(set(observed)):
        raise RuntimeError(f"{label} report contains duplicate item-arm results")
    expected = {(qid, policy) for qid in qids for policy in policies}
    if set(observed) != expected:
        raise RuntimeError(f"{label} report item-arm grid is incomplete")
    return panel, qids, policies


def build_qwen_report(
    root: Path,
    *,
    rows: Sequence[JsonRow],
    source_commit: str,
    family: RouteFamily,
) -> dict[str, Any]:
    """Rescore raw tokens, macro over items, and write the report files."""
    label = family.lane.capitalize()
    if not rows:
        raise RuntimeError(f"{label} report has no selected result rows")
    panel, qids, policies = _validate_grid(rows, label)
    registered = execution_profile()
    scoring_fields = (
        {"scoring_version": scoring_version_for_rows(rows)}
        if registered.scorer == FANOUTQA_NATURAL_DEV50.scorer
        else {}
    )
    items, manifest, sealed = _load_panel(
        root,
        panel=panel,
        source_commit=validate_source_commit(source_commit),
        profile=registered,
        family=family,
    )
    by_qid = {item.qid: item for item in items}
    from transformers import AutoTokenizer

    tokenizer: Any = cast(Any, AutoTokenizer).from_pretrained(
        family.profile.tokenizer,
        revision=family.profile.tokenizer_revision,
        local_files_only=True,
    )
    profile = replace(registered, question_ids=qids)
    validated = [
        validate_and_rescore_result(
            row,
            by_qid[str(row["qid"])],
            tokenizer,
            profile=profile,
            family=family,
        )
        for row in rows
    ]
    score_fields = scored_fields(registered.score_fields)
    summary = _policy_grid(validated, policies, label, score_fields)
    add_stage_means(
        summary,
        [row for row in validated if not bool(row.get("report_failed", False))],
        key="policy",
    )
    pairs = _comparison_pairs(policies, family, registered)
    comparisons = _paired(validated, pairs, score_fields)
    if any(int(row["n"]) != len(qids) for row in summary):
        raise RuntimeError(f"{label} report policy macro is not complete on every item")
    if any(int(row["n_pairs"]) != len(qids) for row in comparisons):
        raise RuntimeError(f"{label} report comparison is not paired on every item")
    report = {
        "schema": report_schema(registered, family.lane),
        **scoring_fields,
        "source_commit": validate_source_commit(source_commit),
        "prepared_sha256": manifest["artifact_sha256"],
        "decode_profile": family.profile.decode.profile_id,
        "decode_fingerprint": family.profile.decode.identity_hash,
        "panel": panel,
        "qids": list(qids),
        "policies": list(policies),
        "result_rows": len(validated),
        "sample_aggregation": "mean_within_item_arm_then_macro_over_items_no_vote",
        "policy_grid": summary,
        "paired_comparisons": comparisons,
        "generation_censoring": _censoring(validated, policies),
    }
    output = root / "report"
    if registered.topology_key == CHAIN_TOPOLOGY_KEY:
        headline = headline_score_field(registered)
        strata_by_qid = {item.qid: int(item.stratum) for item in items}
        strata = chain_strata(validated, policies, strata_by_qid, headline)
        bootstrap = chain_bootstrap(validated, policies, pairs, strata_by_qid, headline)
        report["floors"] = chain_floors(sealed_panel_golds(sealed))
        report["strata"] = strata
        report["bootstrap"] = bootstrap
        # Failed chains stay in every table above, so this section is where a
        # reader learns how many of them there were.
        report["chain_failures"] = chain_failures(validated, policies)
        atomic_bytes(output / "strata.csv", _csv_bytes(strata))
        atomic_bytes(
            output / "bootstrap.json",
            (json.dumps(bootstrap, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(),
        )
    atomic_bytes(
        output / "selected_results.jsonl",
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in validated
        ).encode(),
    )
    atomic_bytes(output / "policy_grid.csv", _csv_bytes(summary))
    atomic_bytes(output / "paired_comparisons.csv", _csv_bytes(comparisons))
    atomic_bytes(
        output / "report.json",
        (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(),
    )
    return report


__all__ = (
    "build_qwen_report",
    "execution_scope",
    "offline_item_range",
    "read_qwen_result_rows",
    "report_schema",
)


def _cli_benchmark(run_id: str, named: str | None) -> BenchmarkProfile:
    """Resolve the benchmark one offline report reads, and hold its run id to it.

    An offline report runs outside the node environment, so it names its own
    benchmark, and the run id's own stem is checked against that name.
    """
    key = named or os.environ.get("RCC_FANOUT_BENCHMARK")
    if key is None:
        raise ValueError(
            "an offline report requires a benchmark: pass --benchmark or set "
            "RCC_FANOUT_BENCHMARK to a registered benchmark key"
        )
    profile = registered_benchmark(key)
    if f"-{profile.run_id_stem}-" not in run_id:
        raise ValueError(
            f"{run_id}: this report resolved {profile.benchmark_key}, whose run id stem "
            f"{profile.run_id_stem!r} the run id does not name"
        )
    return profile


def offline_item_range(
    profile: BenchmarkProfile, item_start: int | None, item_count: int | None
) -> tuple[int, int] | None:
    """Return the pass a command line named, or refuse a range outside the panel.

    A seat reads the pass bounds from the environment beside the benchmark. An
    offline caller names the pair instead, and it is held to the same panel.
    """
    if (item_start is None) != (item_count is None):
        raise ValueError("--item-start and --item-count are named together or not at all")
    if item_start is None or item_count is None:
        return None
    if item_start < 0 or item_count < 1 or item_start + item_count > len(profile.question_ids):
        raise ValueError(
            f"{profile.benchmark_key}: item range {item_start}:{item_count} is outside the "
            f"{len(profile.question_ids)}-question panel"
        )
    return item_start, item_count


@contextmanager
def execution_scope(
    profile: BenchmarkProfile | None, item_range: tuple[int, int] | None
) -> Generator[None]:
    """Publish one offline pass's benchmark and range, then put the process back.

    Everything an offline caller drives resolves the pass by reading the
    environment, so the names are published there and restored on the way out.
    """
    named: dict[str, str] = {}
    if profile is not None:
        named["RCC_FANOUT_BENCHMARK"] = profile.benchmark_key
    if item_range is not None:
        named["RCC_FANOUT_ITEM_START"] = str(item_range[0])
        named["RCC_FANOUT_ITEM_COUNT"] = str(item_range[1])
    restore = {name: os.environ.get(name) for name in named}
    os.environ.update(named)
    try:
        yield
    finally:
        for name, value in restore.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def main(argv: Sequence[str] | None = None) -> int:
    """Validate every banked route row and write the offline report."""
    family = environment_family()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--benchmark")
    # A panel also selects its pass by item range, so an offline report names
    # the range rather than inheriting a default.
    parser.add_argument("--item-start", type=int)
    parser.add_argument("--item-count", type=int)
    args = parser.parse_args(argv)
    if args.run_root.name != args.run_id:
        raise ValueError("run-root basename must equal run-id")
    benchmark = _cli_benchmark(args.run_id, args.benchmark)
    item_range = offline_item_range(benchmark, args.item_start, args.item_count)
    # Everything below resolves the pass the way a seat does, so a named
    # benchmark is published where they all read it, for this pass only.
    with execution_scope(benchmark if args.benchmark else None, item_range):
        rows = read_qwen_result_rows(args.run_root, family=family)
        report = build_qwen_report(
            args.run_root,
            rows=rows,
            source_commit=args.source_commit,
            family=family,
        )
    sys.stdout.write(json.dumps({"result_rows": report["result_rows"]}, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
