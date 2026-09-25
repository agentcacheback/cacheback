"""One offline report over the split bank layout, shared by every family.

The banks are keyed by semantic arm and the roster is the same for every family,
so the comparison follows from the roster rather than from the family.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

from rcc.benchmarks.fanoutqa.scoring import scoring_version_for_rows
from rcc.hardware.fleet import FleetPlacement
from rcc.hardware.placements import SEMANTIC_ARM_PLACEMENTS, placement_for_semantic_arm
from rcc.run.fleet.merge import (
    CellLedger,
    CompletionKey,
    JsonRow,
    add_stage_means,
    merged_split_rows,
    paired_deltas,
    read_arm_bank_rows,
    score_grid,
)
from rcc.run.io import atomic_bytes

#: The per-item means every split report publishes beside its scored columns.
#: All four are fleet-scope, because the merge republishes them.
SPLIT_GRID_MEANS = ("decode_s", "decode_batched_s", "ttft_s", "tteoa_s")

#: One item-arm cell is several independently scored samples. They are averaged
#: within the cell and then macro-ed over items; no sample ever votes.
SAMPLE_AGGREGATION = "mean_within_item_arm_then_macro_over_items_no_vote"

_SCORED = ("loose", "strict")


@dataclass(frozen=True)
class SplitReportSpec:
    """One family's binding of the shared split report."""

    schema: str
    label: str
    identity: Mapping[str, Any]


def comparison_pairs(arms: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Return the semantic comparisons present in one arm roster.

    The fused ``text_primary`` arm is the reference for every Query-Support
    ratio. A roster that also holds a mean query attention arm compares it at the same ratio.
    """
    roster = set(arms)
    pairs: list[tuple[str, str]] = []
    for arm in arms:
        ratio = arm.removeprefix("latent_query_support_r")
        if ratio == arm:
            continue
        snap = f"latent_qsnap_r{ratio}"
        if snap in roster:
            pairs.append((snap, arm))
        if "text_primary" in roster:
            pairs.append(("text_primary", arm))
    return tuple(pairs)


def collapse_cells(
    rows: Sequence[JsonRow],
    *,
    timed: CompletionKey,
    key: str = "arm",
) -> list[JsonRow]:
    """Return one row per item-arm cell: the timed row, scored over its samples.

    The timed row carries the fleet clock, so the cell is published as that row,
    with its scored fields replaced by the mean over the cell's samples.
    """
    cells: dict[tuple[str, str], list[JsonRow]] = {}
    for row in rows:
        cells.setdefault((str(row["qid"]), str(row[key])), []).append(row)
    collapsed: list[JsonRow] = []
    for (qid, group), banked in cells.items():
        priced = [row for row in banked if timed(row) is not None]
        if len(priced) != 1:
            raise RuntimeError(f"{group}/{qid}: cell does not hold exactly one timed sample")
        published = dict(priced[0])
        for field in _SCORED:
            values = [float(row[field]) for row in banked]
            published[field] = round(sum(values) / len(values), 6)
        closed = [row.get("thinking_closed") for row in banked]
        if closed and all(type(value) is bool for value in closed):
            published["thinking_closed_rate"] = round(
                sum(bool(value) for value in closed) / len(closed), 4
            )
        published["scored_samples"] = len(banked)
        collapsed.append(published)
    return collapsed


def _mean_present(rows: Sequence[JsonRow], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return round(sum(values) / len(values), 4) if values else None


def _nearest_rank(rows: Sequence[JsonRow], field: str, percentile: float) -> float | None:
    values = sorted(float(row[field]) for row in rows if row.get(field) is not None)
    return values[max(0, ceil(percentile * len(values)) - 1)] if values else None


def failure_rate(rows: Sequence[JsonRow]) -> float:
    """Return the share of one arm's rows whose sender report failed.

    The policy grid publishes it as ``quarantine_rate`` and the chain report as
    ``failed_rate``, from this one expression. An arm with no rows is refused.
    """
    if not rows:
        raise ValueError("a failure rate needs at least one result row")
    kept = sum(not bool(row.get("report_failed", False)) for row in rows)
    return round(1.0 - kept / len(rows), 4)


def add_paper_metrics(grid: Sequence[JsonRow], rows: Sequence[JsonRow], *, key: str) -> None:
    """Add the latency cohort and the reliability columns to one grid.

    The latency columns are formed over the rows whose sender report succeeded,
    and the reliability columns over every row of the arm.
    """
    for output in grid:
        selected = [row for row in rows if row.get(key) == output[key]]
        latency = [row for row in selected if not bool(row.get("report_failed", False))]
        output["latency_n"] = len(latency)
        output["quarantine_rate"] = failure_rate(selected)
        for field in SPLIT_GRID_MEANS:
            output[field] = _mean_present(latency, field)
        output["tteoa_s_p50"] = _nearest_rank(latency, "tteoa_s", 0.50)
        output["tteoa_s_p95"] = _nearest_rank(latency, "tteoa_s", 0.95)
        output["fleet_protocol_wall_s_p50"] = _nearest_rank(latency, "fleet_protocol_wall_s", 0.50)
        output["fleet_protocol_wall_s_p95"] = _nearest_rank(latency, "fleet_protocol_wall_s", 0.95)
        output["policy_compute_tteoa_s_mean"] = _mean_present(latency, "policy_compute_tteoa_s")
        output["policy_compute_tteoa_s_p50"] = _nearest_rank(
            latency, "policy_compute_tteoa_s", 0.50
        )
        output["policy_compute_tteoa_s_p95"] = _nearest_rank(
            latency, "policy_compute_tteoa_s", 0.95
        )
        output["answer_closure_rate"] = _mean_present(selected, "thinking_closed_rate")
        report_rates = [
            float(row["report_thinking_closed_rate"])
            if row.get("report_thinking_closed_rate") is not None
            else sum(bool(value) for value in row["report_thinking_closed"])
            / len(row["report_thinking_closed"])
            for row in selected
            if row.get("report_thinking_closed_rate") is not None
            or (
                isinstance(row.get("report_thinking_closed"), list)
                and row["report_thinking_closed"]
                and all(type(value) is bool for value in row["report_thinking_closed"])
            )
        ]
        output["report_closure_rate"] = (
            round(sum(report_rates) / len(report_rates), 4) if report_rates else None
        )
        output["draws_n_mean"] = _mean_present(selected, "draws_n")
        output["redraw_wall_s_mean"] = _mean_present(selected, "redraw_wall_s")
        output["injection_activation_rate"] = _mean_present(selected, "injected")


def _grid_identity(rows: Sequence[JsonRow], label: str) -> tuple[str, tuple[str, ...]]:
    panels = {str(row.get("panel") or "") for row in rows}
    if len(panels) != 1 or "" in panels:
        raise RuntimeError(f"{label} report rows do not share one panel")
    observed = [(str(row["qid"]), str(row["arm"])) for row in rows]
    if len(observed) != len(set(observed)):
        raise RuntimeError(f"{label} report contains duplicate item-arm cells")
    qids = tuple(dict.fromkeys(qid for qid, _arm in observed))
    arms = tuple(dict.fromkeys(arm for _qid, arm in observed))
    if set(observed) != {(qid, arm) for qid in qids for arm in arms}:
        raise RuntimeError(f"{label} report item-arm grid is incomplete")
    return panels.pop(), arms


def _refuse_pending(ledgers: Mapping[str, CellLedger], label: str) -> dict[str, int]:
    pending = {arm: len(ledger.pending) for arm, ledger in ledgers.items() if ledger.pending}
    if pending:
        arm = sorted(pending)[0]
        raise RuntimeError(
            f"{arm}: {pending[arm]} items are still undecoded; a split report is "
            "published over complete arms only"
        )
    return {arm: ledger.cell_width for arm, ledger in ledgers.items()}


class SplitReportAdapter:
    """Report over one family's split banks, from the spec that family supplies.

    Every family selects and refuses the same way, so that logic is here and a
    family's own module holds only the identity its rows are published under.
    """

    def __init__(
        self,
        spec: Callable[[], SplitReportSpec],
        placements: Any,
        completion_key: CompletionKey,
        bank_gate: Callable[[Path, Sequence[str], CompletionKey], None] | None = None,
    ) -> None:
        """Bind the family spec, its placement roster, and its completion rule."""
        self._spec = spec
        self._placements = placements
        self._completion_key = completion_key
        self._bank_gate = bank_gate

    def build_report(
        self,
        root: Path,
        *,
        source_commit: str,
        results_policy: Any = None,
        source: Any = None,
    ) -> Mapping[str, Any]:
        """Report over every banked arm, or refuse an unsupported selection.

        A split bank selects by cell ledger rather than by a row policy, so a
        caller passing a selection policy or an external row source is refused.
        """
        spec = self._spec()
        if results_policy is not None or source is not None:
            raise RuntimeError(
                f"the {spec.label} split report selects by cell ledger; it takes no "
                "row policy or external row source"
            )
        return build_split_report(
            root,
            spec,
            self._placements.arm_names,
            completion_key=self._completion_key,
            source_commit=source_commit,
            bank_gate=self._bank_gate,
        )


def build_split_report(
    root: Path,
    spec: SplitReportSpec,
    arms: Sequence[str],
    *,
    completion_key: CompletionKey,
    source_commit: str,
    bank_gate: Callable[[Path, Sequence[str], CompletionKey], None] | None = None,
) -> dict[str, Any]:
    """Merge, collapse, macro, and publish one split run's report.

    ``arms`` is the family's semantic roster; their placements come from the
    shared registry, which is what classifies a direct arm's missing producer.
    """
    banked = {
        arm: placement_for_semantic_arm(arm, table=SEMANTIC_ARM_PLACEMENTS)
        for arm in arms
        if (Path(root) / "arms" / arm / "workers").is_dir()
    }
    if not banked:
        raise RuntimeError(f"{spec.label} split report found no banked arm")
    rows, ledgers = merged_split_rows(
        root,
        banked,
        label=spec.label,
        qids=_banked_qids(root, banked),
        completion_key=completion_key,
    )
    scoring_version = scoring_version_for_rows(rows)
    widths = _refuse_pending(ledgers, spec.label)
    collapsed = collapse_cells(rows, timed=completion_key)
    panel, arms = _grid_identity(collapsed, spec.label)
    grid = score_grid(
        collapsed,
        arms,
        key="arm",
        mean_fields=(),
        label=spec.label,
    )
    add_paper_metrics(grid, collapsed, key="arm")
    add_stage_means(
        grid,
        [row for row in collapsed if not bool(row.get("report_failed", False))],
        key="arm",
    )
    comparisons = paired_deltas(collapsed, comparison_pairs(arms), key="arm")
    report = {
        "schema": spec.schema,
        **dict(spec.identity),
        "scoring_version": scoring_version,
        "source_commit": source_commit,
        "panel": panel,
        "arms": list(arms),
        "qids": sorted({str(row["qid"]) for row in collapsed}),
        "result_cells": len(collapsed),
        "scored_rows": len(rows),
        "cell_width_by_arm": {arm: widths[arm] for arm in arms},
        "dropped_partial_rows": {
            arm: ledgers[arm].dropped_rows for arm in arms if ledgers[arm].dropped_rows
        },
        "sample_aggregation": SAMPLE_AGGREGATION,
        "arm_grid": grid,
        "paired_comparisons": comparisons,
    }
    if bank_gate is not None:
        bank_gate(root, tuple(banked), completion_key)
    output = Path(root) / "report"
    atomic_bytes(
        output / "selected_results.jsonl",
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows).encode(),
    )
    atomic_bytes(
        output / "report.json",
        (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(),
    )
    return report


def _banked_qids(root: Path, placements: Mapping[str, FleetPlacement]) -> tuple[str, ...]:
    """Return every item any arm banked, in first-banked order.

    An offline report needs no panel loader: the grid it publishes has to be
    complete over the items the run banked, and an arm missing one is refused.
    """
    ordered: dict[str, None] = {}
    for arm in placements:
        for row in read_arm_bank_rows(Path(root) / "arms" / arm):
            if row.get("kind") == "result":
                ordered.setdefault(str(row.get("qid") or ""), None)
    ordered.pop("", None)
    return tuple(ordered)


__all__ = (
    "SAMPLE_AGGREGATION",
    "SPLIT_GRID_MEANS",
    "SplitReportAdapter",
    "SplitReportSpec",
    "add_paper_metrics",
    "build_split_report",
    "collapse_cells",
    "comparison_pairs",
)
