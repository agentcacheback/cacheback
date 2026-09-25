"""The stage ledger: what each item's seats stamped, and the spans between.

Each seat banks one ``kind: "stage"`` row per boundary it crosses, and the
strict reader validates the lineage before decomposing it into spans.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

FLEET_STAGES = (
    "submit",
    "producer_claim",
    "producer_start",
    "producer_end",
    "handoff_ready",
    "receiver_claim",
    "payload_loaded",
    "admitted",
    "first_token",
    "decode_end",
)


def read_bank_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Read the complete JSONL rows of these banks, ignoring a torn final line."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            continue
        lines = path.read_bytes().decode("utf-8", errors="replace").split("\n")
        if (lines and lines[-1] == "") or lines:
            lines.pop()
        rows.extend(json.loads(line) for line in lines if line.strip())
    return rows


def banked_result_qids(rows: Sequence[dict[str, Any]], arm: str) -> set[str]:
    """Return the qids that have a banked result for one arm."""
    return {
        str(row["qid"]) for row in rows if row.get("kind") == "result" and row.get("arm") == arm
    }


def result_attempt_rows(
    rows: Sequence[dict[str, Any]],
    arm: str,
) -> list[dict[str, Any]]:
    """Keep the stage rows of the one banked result attempt per item."""
    attempts: dict[str, str] = {}
    for row in rows:
        if row.get("kind") != "result" or row.get("arm") != arm:
            continue
        qid = str(row.get("qid") or "")
        attempt = str(row.get("attempt_id") or "")
        if not qid or not attempt:
            raise RuntimeError(f"{arm}: result has no qid/attempt lineage")
        if qid in attempts:
            raise RuntimeError(f"{arm}/{qid}: multiple durable results have timing lineage")
        attempts[qid] = attempt
    return [
        row
        for row in rows
        if row.get("kind") != "stage"
        or row.get("arm") != arm
        or attempts.get(str(row.get("qid") or "")) == str(row.get("attempt_id") or "")
    ]


def _latest_stage_attempts(
    rows: Sequence[dict[str, Any]],
    arm: str,
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    by_attempt: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        if row.get("kind") != "stage" or row.get("arm") != arm:
            continue
        qid = str(row["qid"])
        stage = str(row["stage"])
        attempt = str(row.get("attempt_id", ""))
        stamp = float(row["t_unix"])
        stamps = by_attempt.setdefault((qid, attempt), {})
        stamps[stage] = max(stamp, stamps.get(stage, stamp))
    winners: dict[str, dict[str, float]] = {}
    winner_attempts: dict[str, str] = {}
    keys: dict[str, tuple[int, float]] = {}
    for (qid, attempt), stamps in by_attempt.items():
        key = (int("decode_end" in stamps), max(stamps.values()))
        if qid not in keys or key > keys[qid]:
            keys[qid] = key
            winners[qid] = stamps
            winner_attempts[qid] = attempt
    return winners, winner_attempts


def latest_stage_times(
    rows: Sequence[dict[str, Any]],
    arm: str,
) -> dict[str, dict[str, float]]:
    """Return one internally consistent stage attempt per item."""
    winners, _attempts = _latest_stage_attempts(rows, arm)
    return winners


def stage_decomposition(
    rows: Sequence[dict[str, Any]],
    arm: str,
) -> list[dict[str, Any]]:
    """Return the non-overlapping stage durations of each item."""

    def span(stamps: dict[str, float], later: str, earlier: str) -> float | None:
        if later not in stamps or earlier not in stamps:
            return None
        return round(stamps[later] - stamps[earlier], 4)

    output: list[dict[str, Any]] = []
    for qid, stamps in sorted(latest_stage_times(rows, arm).items()):
        output.append(
            {
                "qid": qid,
                "arm": arm,
                "producer_queue_s": span(stamps, "producer_claim", "submit"),
                "producer_s": span(stamps, "producer_end", "producer_start"),
                "handoff_s": span(stamps, "handoff_ready", "producer_end"),
                "receiver_queue_s": span(stamps, "receiver_claim", "handoff_ready"),
                "receiver_prepare_s": span(stamps, "payload_loaded", "receiver_claim"),
                "gate_wait_s": span(stamps, "admitted", "payload_loaded"),
                "decode_s": span(stamps, "decode_end", "admitted"),
                "fleet_ttft_s": span(stamps, "first_token", "submit"),
                "tteoa_s": span(stamps, "decode_end", "submit"),
                # The same span under the name the published rows use for it.
                # `decode_end` is stamped at the last scored sample, so this is
                # the protocol wall, not the `tteoa_s` headline (see latency.py).
                "protocol_wall_s": span(stamps, "decode_end", "submit"),
            }
        )
    return output


_PRODUCER_STAGES = frozenset({"producer_claim", "producer_start", "producer_end", "handoff_ready"})
_RECEIVER_STAGES = frozenset(
    {"receiver_claim", "payload_loaded", "admitted", "first_token", "decode_end"}
)


def strict_stage_decomposition(
    rows: Sequence[dict[str, Any]],
    arm: str,
    *,
    direct: bool = False,
) -> list[dict[str, Any]]:
    """Validate the fleet stage lineage, then return its decomposition."""
    accepted: dict[str, str] = {}
    for row in rows:
        if row.get("kind") != "result" or row.get("arm") != arm:
            continue
        qid = row.get("qid")
        attempt = row.get("attempt_id")
        if not isinstance(qid, str) or not qid or not isinstance(attempt, str) or not attempt:
            raise RuntimeError(f"{arm}: result has invalid timing lineage")
        if qid in accepted:
            raise RuntimeError(f"{arm}/{qid}: duplicate result timing lineage")
        accepted[qid] = attempt

    stages: dict[str, dict[str, float]] = {qid: {} for qid in accepted}
    stage_gpus: dict[str, dict[str, int]] = {qid: {} for qid in accepted}
    for row in rows:
        if row.get("kind") != "stage" or row.get("arm") != arm:
            continue
        qid = row.get("qid")
        attempt = row.get("attempt_id")
        stage = row.get("stage")
        role = row.get("role")
        stamp = row.get("t_unix")
        gpu = row.get("gpu_index")
        if not isinstance(qid, str) or accepted.get(qid) != attempt:
            raise RuntimeError(f"{arm}: stage is not bound to an accepted result attempt")
        if stage not in FLEET_STAGES or stage in stages[qid]:
            raise RuntimeError(f"{arm}/{qid}: duplicate or unknown fleet stage {stage!r}")
        if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
            raise RuntimeError(f"{arm}/{qid}: fleet stage has an invalid timestamp")
        numeric_stamp = float(stamp)
        if not math.isfinite(numeric_stamp):
            raise RuntimeError(f"{arm}/{qid}: fleet stage has a non-finite timestamp")
        if type(gpu) is not int or gpu < 0:
            raise RuntimeError(f"{arm}/{qid}: fleet stage has an invalid GPU")
        if stage in _PRODUCER_STAGES and role not in {"producer", "fused"}:
            raise RuntimeError(f"{arm}/{qid}: producer stage has role {role!r}")
        if stage in _RECEIVER_STAGES and role not in {"receiver", "fused"}:
            raise RuntimeError(f"{arm}/{qid}: receiver stage has role {role!r}")
        if stage == "submit" and role not in {"producer", "receiver", "fused"}:
            raise RuntimeError(f"{arm}/{qid}: submit stage has role {role!r}")
        if stage == "submit" and gpu != 0:
            raise RuntimeError(f"{arm}/{qid}: submit stage is not on coordinator gpu0")
        stages[qid][stage] = numeric_stamp
        stage_gpus[qid][stage] = gpu

    required = (
        ("submit", "receiver_claim", "payload_loaded", "admitted", "first_token", "decode_end")
        if direct
        else FLEET_STAGES
    )
    for qid, stamps in stages.items():
        if set(stamps) != set(required):
            missing = sorted(set(required) - set(stamps))
            raise RuntimeError(f"{arm}/{qid}: incomplete fleet stages; missing {missing}")
        ordered = [stamps[stage] for stage in required]
        if any(later < earlier for earlier, later in pairwise(ordered)):
            raise RuntimeError(f"{arm}/{qid}: fleet stages are not causally ordered")
        producer_gpus = {
            stage_gpus[qid][stage] for stage in _PRODUCER_STAGES if stage in stage_gpus[qid]
        }
        receiver_gpus = {
            stage_gpus[qid][stage] for stage in _RECEIVER_STAGES if stage in stage_gpus[qid]
        }
        if len(producer_gpus) > 1 or len(receiver_gpus) != 1:
            raise RuntimeError(f"{arm}/{qid}: fleet stage family changes GPU")
        if not direct and producer_gpus and receiver_gpus == producer_gpus:
            roles = {
                row.get("role")
                for row in rows
                if row.get("kind") == "stage"
                and row.get("arm") == arm
                and row.get("qid") == qid
                and row.get("attempt_id") == accepted[qid]
            }
            if roles != {"fused"}:
                raise RuntimeError(f"{arm}/{qid}: split stage families share one GPU")
    submit_by_attempt: dict[str, float] = {}
    for qid, attempt in accepted.items():
        submit = stages[qid]["submit"]
        if attempt in submit_by_attempt and submit_by_attempt[attempt] != submit:
            raise RuntimeError(f"{arm}/{attempt}: accepted items do not share one submit clock")
        submit_by_attempt[attempt] = submit

    decomposition = stage_decomposition(rows, arm)
    if len(decomposition) != len(accepted):
        raise RuntimeError(f"{arm}: fleet stage decomposition is incomplete")
    timing_fields = (
        ("fleet_ttft_s", "tteoa_s")
        if direct
        else (
            "producer_queue_s",
            "producer_s",
            "handoff_s",
            "receiver_queue_s",
            "receiver_prepare_s",
            "gate_wait_s",
            "decode_s",
            "fleet_ttft_s",
            "tteoa_s",
        )
    )
    if any(
        item[field] is None or not math.isfinite(float(item[field])) or float(item[field]) < 0
        for item in decomposition
        for field in timing_fields
    ):
        raise RuntimeError(f"{arm}: fleet stage decomposition is invalid")
    return decomposition
