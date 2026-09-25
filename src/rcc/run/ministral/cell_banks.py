"""Attempt-aware banking for one Ministral worker shard.

The machinery is shared, in `rcc.run.runlog`. Ministral's own part is the cell
vocabulary, and the refusal of a bank with no full runtime-profile fingerprint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, ClassVar

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.models.ministral import MINISTRAL
from rcc.models.ministral.results import MINISTRAL_RESULT_SCHEMA
from rcc.run import runlog
from rcc.run.io import is_sha256_hex
from rcc.run.runlog import CellBankSpec, open_bank, round_robin_items

CELL_SCHEMA = "ministral-fanoutqa-m3-cell-v1"
_ARMS = tuple(arm.semantic_arm for arm in MINISTRAL.physical_arms)
_SEEDS = len(FANOUTQA_NATURAL_DEV50.sample_tags)
_WORKERS = FANOUTQA_NATURAL_DEV50.workers_per_item

if len(set(_ARMS)) != len(_ARMS) or _SEEDS != 3 or _WORKERS != 3:
    raise RuntimeError("Ministral cell registration differs from the exact M=3 arm grid")


def decode_cell(arm: str, seed_index: int) -> str:
    """Return the bank key of one arm and seed cell."""
    if arm not in _ARMS or "|" in arm:
        raise ValueError(f"unregistered Ministral arm {arm!r}")
    if isinstance(seed_index, bool) or seed_index not in range(_SEEDS):
        raise ValueError(f"Ministral seed index must be in [0, {_SEEDS}), got {seed_index}")
    return f"{arm}|s{seed_index}"


def report_producer(worker: int) -> str:
    """Return the bank key of one report worker."""
    if isinstance(worker, bool) or worker not in range(_WORKERS):
        raise ValueError(f"Ministral report worker must be in [0, {_WORKERS}), got {worker}")
    return f"w{worker}"


def capture_producer(worker: int) -> str:
    """Return the bank key of one capture worker."""
    if isinstance(worker, bool) or worker not in range(_WORKERS):
        raise ValueError(f"Ministral capture worker must be in [0, {_WORKERS}), got {worker}")
    return f"capture-w{worker}"


def bundle_producer(semantic_arm: str) -> str:
    """Return the bank key of one complete text-arm bundle."""
    text_arms = {arm.arm_id for arm in FANOUTQA_NATURAL_DEV50.arms if arm.channel == "text"}
    if semantic_arm not in text_arms:
        raise ValueError(f"unregistered Ministral text arm {semantic_arm!r}")
    return f"bundle:{semantic_arm}"


MINISTRAL_CELL_BANK = CellBankSpec(
    cell_schema=CELL_SCHEMA,
    launcher_label="Ministral M=3",
    lock_filename=".ministral-worker.lock",
    decode_cell=decode_cell,
    report_producer=report_producer,
    capture_producer=capture_producer,
)


def gpu_assignments(qids: Sequence[str], gpus: int) -> tuple[tuple[str, ...], ...]:
    """Assign whole items round-robin so matched arms stay on one GPU."""
    if isinstance(gpus, bool) or gpus < 1:
        raise ValueError(f"gpus must be at least 1, got {gpus}")
    normalized = tuple(str(qid) for qid in qids)
    if any(not qid for qid in normalized) or len(set(normalized)) != len(normalized):
        raise ValueError("Ministral GPU assignment qids must be nonempty and unique")
    return round_robin_items(normalized, gpus)


def worker_bank_lock(run_root: Path, worker_index: int) -> AbstractContextManager[None]:
    """Refuse two processes writing the same Ministral GPU bank."""
    if isinstance(worker_index, bool) or worker_index < 0:
        raise ValueError("Ministral worker index must be nonnegative")
    return runlog.worker_bank_lock(run_root, worker_index, MINISTRAL_CELL_BANK)


class ShardLog(runlog.ShardLog):
    """One GPU worker's append-only, attempt-aware bank."""

    spec: ClassVar[CellBankSpec] = MINISTRAL_CELL_BANK

    @classmethod
    def open(
        cls,
        run_root: Path,
        *,
        worker_index: int,
        attempt_id: str,
        fixed_fields: Mapping[str, Any],
    ) -> ShardLog:
        """Open one worker bank, repairing it, under the full runtime identity."""
        if not attempt_id:
            raise ValueError("Ministral shard attempt id must be nonempty")
        if isinstance(worker_index, bool) or worker_index < 0:
            raise ValueError("Ministral shard worker index must be nonnegative")
        if not is_sha256_hex(fixed_fields.get("runtime_profile_fingerprint")):
            raise ValueError("Ministral shard requires a full runtime-profile fingerprint")
        # The result-row schema is a fixed bank field, so a bank written under
        # another row schema refuses to reopen under this one. CELL_SCHEMA names
        # the cell shape; the row schema names what a banked clock means.
        return cls(
            log=open_bank(
                run_root,
                worker_index,
                {**dict(fixed_fields), "row_schema": MINISTRAL_RESULT_SCHEMA},
                MINISTRAL_CELL_BANK,
            ),
            attempt_id=attempt_id,
        )

    def banked_bundle(self, qid: str, semantic_arm: str) -> dict[str, Any] | None:
        """Return one complete sender-arm report bundle, if present."""
        return self.log.reports.get((str(qid), bundle_producer(semantic_arm)))

    def bank_decode(self, row: Mapping[str, Any], *, arm: str, seed_index: int) -> dict[str, Any]:
        """Append one decoded cell, fsynced and attempt-stamped."""
        decode_cell(arm, seed_index)  # validate the cell key before the row
        if row.get("arm") != arm or row.get("seed_index") != seed_index:
            raise ValueError("Ministral banked result differs from its cell identity")
        return super().bank_decode(row, arm=arm, seed_index=seed_index)

    def bank_bundle(
        self,
        row: Mapping[str, Any],
        *,
        qid: str,
        semantic_arm: str,
    ) -> dict[str, Any]:
        """Append one complete sender-arm report bundle."""
        return self.bank_producer(row, qid=qid, producer=bundle_producer(semantic_arm))

    def require_result_runtime(self, runtime_fingerprint: str) -> None:
        """Refuse receiver rows banked under another observed runtime digest."""
        if not is_sha256_hex(runtime_fingerprint):
            raise ValueError("Ministral observed runtime fingerprint is malformed")
        drifted = sorted(
            f"{qid}/{cell}"
            for (qid, cell), row in self.log.results.items()
            if row.get("runtime_fingerprint") != runtime_fingerprint
        )
        if drifted:
            raise RuntimeError(
                "Ministral banked receiver runtime differs from the live engine: "
                + ", ".join(drifted[:3])
            )


__all__ = (
    "CELL_SCHEMA",
    "ShardLog",
    "bundle_producer",
    "capture_producer",
    "decode_cell",
    "gpu_assignments",
    "report_producer",
    "worker_bank_lock",
)
