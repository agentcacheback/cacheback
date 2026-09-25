"""Average FanOutQA result rows over the receiver seeds of an item, then over items."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def item_arm_constant(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, dict[str, int]]:
    """Return one value per item and arm, raising when its seeds disagree."""
    cells: dict[tuple[str, str], set[int]] = {}
    for row in rows:
        cells.setdefault((str(row["qid"]), str(row["arm"])), set()).add(int(row[field]))
    constants: dict[str, dict[str, int]] = {}
    for (qid, arm), distinct in sorted(cells.items()):
        if len(distinct) != 1:
            raise RuntimeError(
                f"{qid}/{arm}: {field} disagrees across seeds ({sorted(distinct)}); "
                "it is a per-cell constant and must be identical in every seed row"
            )
        constants.setdefault(qid, {})[arm] = distinct.pop()
    return constants


def item_arm_means(
    rows: Iterable[Mapping[str, Any]], field: str = "loose"
) -> dict[str, dict[str, float]]:
    """Average one row field over the receiver seeds of each item and arm."""
    cells: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        cells.setdefault(str(row["qid"]), {}).setdefault(str(row["arm"]), []).append(
            float(row[field])
        )
    return {
        qid: {arm: sum(values) / len(values) for arm, values in by_arm.items()}
        for qid, by_arm in cells.items()
    }


def arm_macro(
    rows: Sequence[Mapping[str, Any]], arm: str, field: str, qids: Sequence[str]
) -> float | None:
    """Average over items, after averaging each item's receiver seeds."""
    per_item = item_arm_means(rows, field)
    values = [per_item[qid][arm] for qid in qids if qid in per_item and arm in per_item[qid]]
    if not values:
        return None
    return sum(values) / len(values)


__all__ = ("arm_macro", "item_arm_constant", "item_arm_means")
