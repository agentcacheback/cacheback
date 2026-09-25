"""The Gemma binding of the shared split-bank report.

The report itself is shared. Gemma contributes the schema, the identity fields,
and one check that every completed latent item banked a capture row per worker.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.gemma import GEMMA
from rcc.models.gemma.contract import (
    FLEET_RUNTIME,
    ROW_SCHEMA,
    WORKERS_PER_ITEM,
    registration_fingerprint,
    scientific_config_fingerprint,
)
from rcc.models.gemma.roster import gemma_v16_arm, shared_execution_roster
from rcc.models.gemma.schedule import TEXT_ARMS
from rcc.run.barrier import SPLIT_FLEET_ISOLATION_PROFILE
from rcc.run.fleet.merge import CompletionKey, read_arm_bank_rows
from rcc.run.fleet.report import SplitReportAdapter, SplitReportSpec

GEMMA_SPLIT_REPORT_SCHEMA = "gemma-fanoutqa-split-report-v1"
_CAPTURE_WORKERS = range(WORKERS_PER_ITEM)


def require_complete_capture_rows(
    root: Path,
    arms: Sequence[str],
    completion_key: CompletionKey,
) -> None:
    """Require one capture row per worker of every completed latent item."""
    for arm in arms:
        v16_arm = gemma_v16_arm(arm)
        if v16_arm == "floor" or v16_arm in TEXT_ARMS:
            continue
        rows = read_arm_bank_rows(Path(root) / "arms" / arm)
        complete = {
            (str(row.get("qid") or ""), str(row.get("attempt_id") or ""))
            for row in rows
            if completion_key(row) is not None
        }
        expected = Counter(
            (qid, attempt, worker) for qid, attempt in complete for worker in _CAPTURE_WORKERS
        )
        observed: Counter[tuple[str, str, int]] = Counter()
        for row in rows:
            pair = (str(row.get("qid") or ""), str(row.get("attempt_id") or ""))
            if row.get("phase") != "gemma_split_capture" or pair not in complete:
                continue
            worker = row.get("worker")
            if type(worker) is not int:
                raise RuntimeError(f"{arm}/{pair[0]}: capture audit worker is malformed")
            observed[(*pair, worker)] += 1
        if observed != expected:
            missing = sorted(expected - observed)[:5]
            extra = sorted(observed - expected)[:5]
            raise RuntimeError(
                f"{arm}: completed latent items require exactly one capture audit row "
                f"for each of workers 0, 1, and 2; missing (qid, attempt, worker) "
                f"{missing}, duplicated {extra}; a hole is not repairable under this "
                "run id, a fresh run id re-captures the item"
            )


def gemma_split_report_spec(profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50) -> SplitReportSpec:
    """Return the identity a Gemma split report is published under."""
    roster = shared_execution_roster(profile)
    roster.require_complete()
    return SplitReportSpec(
        schema=GEMMA_SPLIT_REPORT_SCHEMA,
        label="Gemma",
        identity={
            "row_schema": ROW_SCHEMA,
            "fleet_runtime": FLEET_RUNTIME,
            "isolation_profile": SPLIT_FLEET_ISOLATION_PROFILE,
            "config_fingerprint": scientific_config_fingerprint(),
            "route_registration_fingerprint": registration_fingerprint()[:16],
            "execution_roster_fingerprint": roster.fingerprint,
            "decode_profile": GEMMA.decode.profile_id,
            "decode_fingerprint": GEMMA.decode.identity_hash,
        },
    )


def gemma_split_report_adapter(
    placements: Any,
    completion_key: CompletionKey,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> SplitReportAdapter:
    """Return the shared split report bound to the Gemma identity."""
    return SplitReportAdapter(
        partial(gemma_split_report_spec, profile),
        placements,
        completion_key,
        require_complete_capture_rows,
    )


__all__ = (
    "GEMMA_SPLIT_REPORT_SCHEMA",
    "gemma_split_report_adapter",
    "gemma_split_report_spec",
    "require_complete_capture_rows",
)
