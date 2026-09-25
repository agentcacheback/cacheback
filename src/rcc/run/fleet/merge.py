"""Offline merge over the split-fleet bank layout, shared by every family.

A split arm banks one file per placement seat, because two processes appending
to one path can interleave. The combined view is produced here, offline.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from rcc.hardware.fleet import FleetPlacement
from rcc.run.banks import read_bank_rows
from rcc.run.fleet.latency import FLEET_STAGE_FIELDS, publish_fleet_latency
from rcc.run.fleet.ledger import result_attempt_rows

JsonRow = dict[str, Any]
#: The family's own rule for "this row completes its item", which is also the
#: rule for "this row is the timed sample": a family times exactly the sample
#: it completes an item with.
CompletionKey = Callable[[Mapping[str, Any]], object | None]


def split_bank_paths(arm_root: Path) -> tuple[Path, ...]:
    """Return every banked seat file of one arm, in seat order."""
    return tuple(sorted(Path(arm_root).glob("workers/gpu*/raw.jsonl")))


def worker_bank_paths(root: Path, *, arm: str, workers: int) -> tuple[Path, ...]:
    """Return the per-seat bank paths one arm's placement declares.

    This is the layout rather than a search: the runner opens exactly these
    files, in placement order, before any of them exists.
    """
    return tuple(
        Path(root) / "arms" / arm / "workers" / f"gpu{index}" / "raw.jsonl"
        for index in range(workers)
    )


def read_arm_bank_rows(arm_root: Path) -> list[JsonRow]:
    """Read one arm's whole bank, every seat, in seat order."""
    rows: list[JsonRow] = []
    for path in split_bank_paths(arm_root):
        rows.extend(read_bank_rows(path, repair_torn_tail=False))
    return rows


def is_direct(placement: FleetPlacement) -> bool:
    """Return whether an arm's receivers read no handoff at all."""
    return placement.producers == 0 and not placement.fused


def read_split_result_rows(
    root: Path,
    placements: Mapping[str, FleetPlacement],
    *,
    label: str,
    timed: CompletionKey | None = None,
) -> list[JsonRow]:
    """Read every banked arm under one run root, on its own fleet clock.

    ``timed`` names the one sample per item the stage ledger priced; a batched
    peer is kept only when it carries that timed row's ``attempt_id``.
    """
    rows: list[JsonRow] = []
    for arm_root in sorted((Path(root) / "arms").glob("*")):
        if not arm_root.is_dir():
            continue
        banked = read_arm_bank_rows(arm_root)
        results = [row for row in banked if row.get("kind") == "result"]
        if not results:
            continue
        arm = arm_root.name
        placement = placements.get(arm)
        if placement is None:
            raise RuntimeError(f"{arm}: banked arm is outside the registered {label} roster")
        priced = results if timed is None else [row for row in results if timed(row) is not None]
        # The stage ledger prices one attempt per item, so it is shown only the
        # timed rows: a cell's batched peers share the timed row's lineage and
        # would read as several results for one item.
        lineage = [row for row in banked if row.get("kind") != "result"] + priced
        rows.extend(
            publish_fleet_latency(
                priced,
                result_attempt_rows(lineage, arm),
                arm=arm,
                direct=is_direct(placement),
            )
        )
        if timed is not None:
            completed = {_cell_attempt(row) for row in priced}
            rows.extend(
                row for row in results if timed(row) is None and _cell_attempt(row) in completed
            )
    return rows


def _cell_attempt(row: Mapping[str, Any]) -> tuple[str, str]:
    """Return the item and attempt one banked row belongs to."""
    return str(row.get("qid") or ""), str(row.get("attempt_id") or "")


def add_stage_means(summary: Sequence[JsonRow], rows: Sequence[JsonRow], *, key: str) -> None:
    """Add each group's mean for every republished fleet stage column.

    A direct arm prices no producer, handoff, or receiver queue, so those columns
    arrive as ``None`` and are skipped; a group with none reports ``None``.
    """
    for output in summary:
        selected = [row for row in rows if row.get(key) == output[key]]
        for field in FLEET_STAGE_FIELDS:
            name = field if field.startswith("fleet_") else f"fleet_{field}"
            values = [float(row[name]) for row in selected if row.get(name) is not None]
            output[f"{name}_mean"] = round(fmean(values), 4) if values else None


def standard_error(values: Sequence[float]) -> float:
    """Return the item-level standard error, or zero for a single item."""
    return round(stdev(values) / math.sqrt(len(values)), 4) if len(values) > 1 else 0.0


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return round(sum(float(row[field]) for row in rows) / len(rows), 4)


#: The scored columns a caller gets when it names none, the FanOutQA pair. A
#: benchmark scoring anything else names its own fields from its profile.
SCORE_FIELDS = ("loose", "strict")

#: Per-sample fields a profile registers beside its scores that are records,
#: not scores: `n_leaves` is the width of the question a sample answered. No
#: mean, interval, or report column is formed over them.
RECORD_FIELDS = ("n_leaves",)


def scored_fields(registered: Sequence[str]) -> tuple[str, ...]:
    """Return the scored columns of one profile's registered score fields."""
    return tuple(field for field in registered if field not in RECORD_FIELDS)


def score_grid(
    rows: Sequence[JsonRow],
    groups: Sequence[str],
    *,
    key: str,
    mean_fields: Sequence[str],
    label: str,
    score_fields: Sequence[str] = SCORE_FIELDS,
) -> list[JsonRow]:
    """Macro each scored field over items, group by group, in group order."""
    output: list[JsonRow] = []
    for group in groups:
        selected = [row for row in rows if row.get(key) == group]
        if not selected:
            raise RuntimeError(f"{group}: {label} report has no selected rows")
        summary: JsonRow = {key: group, "n": len(selected)}
        for field in score_fields:
            summary[field] = _mean(selected, field)
        for field in score_fields:
            summary[f"{field}_item_se"] = standard_error([float(row[field]) for row in selected])
        for field in mean_fields:
            summary[field] = _mean(selected, field)
        output.append(summary)
    return output


def paired_deltas(
    rows: Sequence[JsonRow],
    pairs: Sequence[tuple[str, str]],
    *,
    key: str,
    score_fields: Sequence[str] = SCORE_FIELDS,
) -> list[JsonRow]:
    """Difference each candidate group against its reference, item by item."""
    by_key = {(str(row["qid"]), str(row[key])): row for row in rows}
    qids = sorted({qid for qid, _group in by_key})
    output: list[JsonRow] = []
    for reference, candidate in pairs:
        cells = [
            (by_key[(qid, reference)], by_key[(qid, candidate)])
            for qid in qids
            if (qid, reference) in by_key and (qid, candidate) in by_key
        ]
        deltas = {
            field: [
                float(candidate_row[field]) - float(reference_row[field])
                for reference_row, candidate_row in cells
            ]
            for field in score_fields
        }
        summary: JsonRow = {
            f"reference_{key}": reference,
            f"candidate_{key}": candidate,
            "n_pairs": len(cells),
        }
        for field in score_fields:
            summary[f"{field}_delta"] = round(sum(deltas[field]) / len(deltas[field]), 4)
        for field in score_fields:
            summary[f"{field}_delta_item_se"] = standard_error(deltas[field])
        output.append(summary)
    return output


@dataclass(frozen=True)
class CellLedger:
    """What one arm's banks say about every item's decode cell."""

    arm: str
    #: Items whose timed completion row is banked.
    complete: tuple[str, ...]
    #: Items with no complete cell: what still has to be decoded.
    pending: tuple[str, ...]
    #: How many rows belong to no complete cell and the merge drops.
    dropped_rows: int
    #: How many rows one whole cell holds, from the widest cell observed.
    cell_width: int


def _item_key(row: Mapping[str, Any], arm: str) -> str:
    """Return one row's item identity, refusing a row without a sample cell.

    A row's ``cell`` names one sample of one item-arm cell, so it is checked here
    but never grouped on: grouping by sample would make every cell one row wide.
    """
    qid, cell = str(row.get("qid") or ""), str(row.get("cell") or "")
    if not qid or not cell:
        raise RuntimeError(f"{arm}: banked result row lacks its item or cell identity")
    return qid


def cell_ledger(
    rows: Sequence[JsonRow],
    *,
    arm: str,
    qids: Sequence[str],
    completion_key: CompletionKey,
) -> CellLedger:
    """Split one arm's result rows into complete cells, orphans, and pending.

    A cell is complete when the family's completion key admits one of its rows.
    The timed row is banked last, so it is the only sound completion marker.
    """
    cells: dict[tuple[str, str], list[JsonRow]] = {}
    for row in rows:
        if row.get("kind") != "result":
            continue
        cells.setdefault((_item_key(row, arm), str(row.get("attempt_id") or "")), []).append(row)
    banked = [
        key for key, group in cells.items() if any(completion_key(row) is not None for row in group)
    ]
    done: dict[str, str] = {}
    for qid, attempt in banked:
        if qid in done:
            raise RuntimeError(f"{arm}/{qid}: two attempts banked a complete cell for one item")
        done[qid] = attempt
    width = max((len(cells[key]) for key in banked), default=0)
    torn = sorted(key[0] for key in banked if len(cells[key]) != width)
    if torn:
        raise RuntimeError(
            f"{arm}/{torn[0]}: cell banked its completion row without its whole "
            f"{width}-row width; the bank is torn, not merely incomplete"
        )
    return CellLedger(
        arm=arm,
        complete=tuple(done),
        pending=tuple(qid for qid in qids if qid not in done),
        dropped_rows=sum(len(group) for key, group in cells.items() if key not in set(banked)),
        cell_width=width,
    )


def arm_cell_ledgers(
    root: Path,
    arms: Sequence[str],
    *,
    qids: Sequence[str],
    completion_key: CompletionKey,
) -> dict[str, CellLedger]:
    """Return the cell ledger of every named arm under one run root."""
    ledgers: dict[str, CellLedger] = {}
    for arm in arms:
        arm_root = Path(root) / "arms" / arm
        ledger = cell_ledger(
            read_arm_bank_rows(arm_root),
            arm=arm,
            qids=qids,
            completion_key=completion_key,
        )
        ledgers[arm] = ledger
    return ledgers


def merged_split_rows(
    root: Path,
    placements: Mapping[str, FleetPlacement],
    *,
    label: str,
    qids: Sequence[str],
    completion_key: CompletionKey,
) -> tuple[list[JsonRow], dict[str, CellLedger]]:
    """Merge every arm's complete cells and return them beside the ledgers.

    The ledger is computed from the raw banks and the merge from the fleet-clock
    republication of those same banks, so one rule drops every orphaned peer.
    """
    ledgers = arm_cell_ledgers(
        root,
        tuple(placements),
        qids=qids,
        completion_key=completion_key,
    )
    keep = {(ledger.arm, qid) for ledger in ledgers.values() for qid in ledger.complete}
    published = read_split_result_rows(root, placements, label=label, timed=completion_key)
    merged = [row for row in published if (str(row.get("arm")), str(row.get("qid"))) in keep]
    return merged, ledgers


__all__ = (
    "RECORD_FIELDS",
    "SCORE_FIELDS",
    "CellLedger",
    "add_stage_means",
    "arm_cell_ledgers",
    "cell_ledger",
    "is_direct",
    "merged_split_rows",
    "paired_deltas",
    "read_arm_bank_rows",
    "read_split_result_rows",
    "score_grid",
    "scored_fields",
    "split_bank_paths",
    "standard_error",
)
