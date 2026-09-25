"""The chain's own report sections: floors, strata, intervals, lost parts.

A FanOutQA report has none of these: a free-text proxy score has no chance floor
to beat, and its workers read the source in parallel rather than in a chain.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from rcc.benchmarks.longbench_v2.bootstrap import stratified_paired_bootstrap
from rcc.benchmarks.longbench_v2.data import ChainItem
from rcc.benchmarks.longbench_v2.registration import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED
from rcc.benchmarks.longbench_v2.scoring import CHOICES
from rcc.run.fleet.report import failure_rate

JsonRow = dict[str, Any]


def sealed_panel_golds(items: Sequence[ChainItem]) -> tuple[str, ...]:
    """Return the gold letter of every panel item, in or out of this pass.

    A floor is a property of the panel rather than of one pass, so it is read off
    the whole roster the adapter has already loaded and validated.
    """
    return tuple(str(item.gold) for item in items)


def chain_floors(golds: Sequence[str]) -> dict[str, Any]:
    """Return the two floors a multiple-choice macro is read against.

    Chance is one over the choice count; the majority floor is what always
    answering the panel's most common gold would score. Both run over the panel.
    """
    counted = Counter(str(gold) for gold in golds)
    if not counted:
        raise RuntimeError("the chain floors need the sealed panel's gold letters")
    return {
        "chance": round(1 / len(CHOICES), 4),
        "majority_floor": round(max(counted.values()) / sum(counted.values()), 4),
        "panel_items": sum(counted.values()),
    }


def chain_strata(
    rows: Sequence[JsonRow],
    policies: Sequence[str],
    strata_by_qid: Mapping[str, int],
    field: str,
) -> list[JsonRow]:
    """Macro the primary score inside each source-length band, arm by arm.

    A band short of a row would publish a macro over fewer items and read as a
    band effect, so an arm that misses any loaded item is refused by name.
    """
    loaded = set(strata_by_qid)
    output: list[JsonRow] = []
    for policy in policies:
        covered = {str(row["qid"]) for row in rows if row["policy"] == policy}
        missing = sorted(loaded - covered)
        if missing:
            raise RuntimeError(
                f"{policy}: report strata need every loaded item; missing {', '.join(missing)}"
            )
        for band in sorted(set(strata_by_qid.values())):
            selected = [
                row
                for row in rows
                if row["policy"] == policy and strata_by_qid[str(row["qid"])] == band
            ]
            output.append(
                {
                    "policy": policy,
                    "stratum": band,
                    "n": len(selected),
                    field: round(sum(float(row[field]) for row in selected) / len(selected), 4),
                }
            )
    return output


def chain_failures(rows: Sequence[JsonRow], policies: Sequence[str]) -> list[JsonRow]:
    """Count, per arm, the items whose chain lost a part, and publish the rate.

    A hop that ships an empty note hands the next hop nothing, so the arm carries
    less than the document. Such an item is scored as it stands and stays in.
    """
    output: list[JsonRow] = []
    for policy in policies:
        selected = [row for row in rows if row["policy"] == policy]
        failed = sorted(
            str(row["qid"]) for row in selected if bool(row.get("report_failed", False))
        )
        output.append(
            {
                "policy": policy,
                "failed_items": len(failed),
                "failed_qids": failed,
                "failed_rate": failure_rate(selected),
            }
        )
    return output


def chain_bootstrap(
    rows: Sequence[JsonRow],
    policies: Sequence[str],
    pairs: Sequence[tuple[str, str]],
    strata_by_qid: Mapping[str, int],
    field: str,
) -> dict[str, Any]:
    """Interval the per-arm macro and every paired delta, resampled within stratum."""
    by_key = {(str(row["qid"]), str(row["policy"])): float(row[field]) for row in rows}
    qids = sorted({qid for qid, _policy in by_key})
    return {
        "metric": field,
        "unit": "qid within stratum",
        "arms": [
            {
                "policy": policy,
                **stratified_paired_bootstrap(
                    {qid: by_key[(qid, policy)] for qid in qids},
                    strata_by_qid,
                    draws=BOOTSTRAP_DRAWS,
                    seed=BOOTSTRAP_SEED,
                ),
            }
            for policy in policies
        ],
        "paired_comparisons": [
            {
                "reference_policy": reference,
                "candidate_policy": candidate,
                **stratified_paired_bootstrap(
                    {qid: by_key[(qid, candidate)] - by_key[(qid, reference)] for qid in qids},
                    strata_by_qid,
                    draws=BOOTSTRAP_DRAWS,
                    seed=BOOTSTRAP_SEED,
                ),
            }
            for reference, candidate in pairs
        ],
    }


__all__ = (
    "chain_bootstrap",
    "chain_failures",
    "chain_floors",
    "chain_strata",
    "failure_rate",
    "sealed_panel_golds",
)
