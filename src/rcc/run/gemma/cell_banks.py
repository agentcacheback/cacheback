"""Attempt-aware banking for one Gemma worker shard.

The machinery is shared, in `rcc.run.runlog`. Gemma's own part is the cell
vocabulary: its unit of work is (qid, arm, seed), since it decodes many seeds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, ClassVar

from rcc.run import runlog
from rcc.run.runlog import CellBankSpec, open_bank, round_robin_items

CELL_SCHEMA = "gemma-fanoutqa-m3-cell-v2"
"""Stamped on every banked row; rows under another schema are not reused."""


def decode_cell(arm: str, seed_index: int) -> str:
    """Return the bank key of one decoded cell, inside its qid.

    (qid, arm, seed) is the finest unit this run produces, and the bank keys on
    (qid, name), so the arm and the seed compose into that one name.
    """
    if not arm or "|" in arm:
        raise ValueError(f"arm {arm!r} cannot be part of a cell key")
    if seed_index < 0:
        raise ValueError(f"seed index must be nonnegative, got {seed_index}")
    return f"{arm}|s{seed_index}"


def report_producer(worker: int) -> str:
    """Return the bank key of one worker's text report, inside its qid."""
    if worker < 0:
        raise ValueError(f"worker must be nonnegative, got {worker}")
    return f"w{worker}"


def capture_producer(worker: int) -> str:
    """Return the bank key of one worker's capture timing, inside its qid."""
    if worker < 0:
        raise ValueError(f"worker must be nonnegative, got {worker}")
    return f"capture-w{worker}"


GEMMA_CELL_BANK = CellBankSpec(
    cell_schema=CELL_SCHEMA,
    launcher_label="gemma m3",
    lock_filename=".worker.lock",
    decode_cell=decode_cell,
    report_producer=report_producer,
    capture_producer=capture_producer,
)


def gpu_assignments(qids: Sequence[str], gpus: int) -> tuple[tuple[str, ...], ...]:
    """Assign whole items round-robin across a node's GPUs."""
    if gpus < 1:
        raise ValueError(f"gpus must be at least 1, got {gpus}")
    return round_robin_items(qids, gpus)


def worker_bank_lock(run_root: Path, worker_index: int) -> AbstractContextManager[None]:
    """Refuse two processes writing the same Gemma GPU bank concurrently."""
    return runlog.worker_bank_lock(run_root, worker_index, GEMMA_CELL_BANK)


class ShardLog(runlog.ShardLog):
    """One GPU worker's append-only, attempt-aware bank."""

    spec: ClassVar[CellBankSpec] = GEMMA_CELL_BANK

    @classmethod
    def open(
        cls,
        run_root: Path,
        *,
        worker_index: int,
        attempt_id: str,
        fixed_fields: Mapping[str, Any],
    ) -> ShardLog:
        """Open this worker's bank, repairing a torn tail."""
        return cls(
            log=open_bank(run_root, worker_index, fixed_fields, GEMMA_CELL_BANK),
            attempt_id=attempt_id,
        )
