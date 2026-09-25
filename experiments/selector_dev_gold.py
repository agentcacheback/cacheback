"""Gold-evidence retention of one keep set, for the selector-development analysis.

`merged_gold_spans` merges located gold occurrences into position spans;
`gold_metrics` reports how much of that evidence a keep set retains.
"""

from __future__ import annotations

from bisect import bisect_left
from typing import Any

#: Span width the keep sets are scored on, in positions.
SPAN_SIZE = 16


def merged_gold_spans(rows: list[dict[str, Any]]) -> tuple[tuple[int, int], ...]:
    """Merge the official-page occurrences of every required leaf into spans."""
    spans = sorted(
        (int(span["start"]), int(span["end"]))
        for row in rows
        if row["required"]
        for span in row["occurrences"]
    )
    merged: list[tuple[int, int]] = []
    for start, stop in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
        else:
            merged.append((start, stop))
    return tuple(merged)


def _kept_in_range(keep: tuple[int, ...], start: int, stop: int) -> int:
    return bisect_left(keep, stop) - bisect_left(keep, start)


def gold_metrics(keep: tuple[int, ...], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure how much official evidence a keep set retains."""
    required = [row for row in rows if row["required"]]
    token_recalls = [
        max(
            _kept_in_range(keep, int(span["start"]), int(span["end"]))
            / (int(span["end"]) - int(span["start"]))
            for span in row["occurrences"]
        )
        for row in required
    ]
    leaf_retained = {
        int(row["leaf_index"]): any(
            _kept_in_range(keep, int(span["start"]), int(span["end"]))
            == int(span["end"]) - int(span["start"])
            for span in row["occurrences"]
        )
        for row in required
    }
    by_group: dict[int, list[bool]] = {}
    for row in required:
        for group in row["required_groups"]:
            group_id = int(group)
            retained_in_group = any(
                group_id in {int(value) for value in span["groups"]}
                and _kept_in_range(keep, int(span["start"]), int(span["end"]))
                == int(span["end"]) - int(span["start"])
                for span in row["occurrences"]
            )
            by_group.setdefault(group_id, []).append(retained_in_group)
    branch_recalls = [sum(values) / len(values) for values in by_group.values()]
    leaves_retained = sum(leaf_retained.values())
    islands_retained = sum(all(values) for values in by_group.values())
    ineligible = sum(
        row.get("mechanism_eligibility") not in {"eligible", "question_given"} for row in rows
    )
    mechanism_complete = ineligible == 0
    return {
        "official_evidence_eligible_leaves": len(required),
        "official_evidence_leaves_retained": leaves_retained,
        "official_evidence_leaf_recall": leaves_retained / len(required) if required else 0.0,
        "all_official_evidence_leaves_retained": leaves_retained == len(required),
        "mean_best_official_span_token_recall": (
            sum(token_recalls) / len(token_recalls) if token_recalls else 0.0
        ),
        "official_evidence_groups": len(by_group),
        "official_evidence_groups_retained": islands_retained,
        "official_evidence_group_recall": islands_retained / len(by_group) if by_group else 0.0,
        "all_official_evidence_groups_retained": islands_retained == len(by_group),
        "min_official_group_recall": min(branch_recalls) if branch_recalls else 0.0,
        "gold_mechanism_ineligible_leaves": ineligible,
        "gold_mechanism_complete": mechanism_complete,
        "gold_mechanism_identified": mechanism_complete and bool(required),
    }
