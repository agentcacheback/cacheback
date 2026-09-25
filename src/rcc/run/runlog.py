"""Crash-safe JSONL banking: append-only rows with torn-tail repair.

Every row carries the bank's fixed identity fields and an attempt id, and the
in-memory index says which result and producer keys are already banked.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, cast

from rcc.run import identity

ResultKey = tuple[str, str]


def runtime_signature(model_revisions: dict[str, Any]) -> dict[str, Any]:
    """Return the hardware, software, and model identity of one timing run."""
    return identity.runtime_signature(model_revisions)


def runtime_fingerprint(signature: dict[str, Any]) -> str:
    """Return the short digest of one runtime signature."""
    return identity.fingerprint(signature, identity.runtime_legacy)


class DurableRunLog:
    """An append-only JSONL bank with torn-tail repair and fixed row identity."""

    def __init__(
        self,
        path: Path,
        *,
        result_field: str,
        fixed_fields: dict[str, Any],
    ) -> None:
        """Repair a torn tail, then load and index the rows already banked."""
        self.path = Path(path)
        self.result_field = result_field
        self.fixed_fields = dict(fixed_fields)
        self.done: set[ResultKey] = set()
        # A producer seat banks from two threads, its loop and the deferred
        # audit, so the append and the in-memory index move under one lock.
        self._lock = threading.Lock()
        self.results: dict[ResultKey, dict[str, Any]] = {}
        self.reports: dict[ResultKey, dict[str, Any]] = {}
        self.repaired_tail_bytes = self._repair_torn_tail()
        for row in self._read_rows():
            self._validate_identity(row)
            self._remember(row)

    def _repair_torn_tail(self) -> int:
        if not self.path.exists():
            return 0
        data = self.path.read_bytes()
        if not data or data.endswith(b"\n"):
            return 0
        boundary = data.rfind(b"\n") + 1
        repaired = len(data) - boundary
        with self.path.open("r+b") as handle:
            handle.truncate(boundary)
            handle.flush()
            os.fsync(handle.fileno())
        return repaired

    def _read_rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            try:
                decoded = cast(object, json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{self.path} has malformed complete JSON at line {line_number}"
                ) from exc
            if not isinstance(decoded, dict):
                raise RuntimeError(f"{self.path} line {line_number} is not a JSON object")
            row = cast(dict[str, Any], decoded)
            rows.append(row)
        return rows

    def _validate_identity(self, row: dict[str, Any]) -> None:
        for name, expected in self.fixed_fields.items():
            if row.get(name) != expected:
                raise RuntimeError(
                    f"{self.path} identity mismatch for {name}: {row.get(name)!r} != {expected!r}"
                )

    def _remember(self, row: dict[str, Any]) -> None:
        if row.get("kind") == "result":
            key = (str(row["qid"]), str(row[self.result_field]))
            self.done.add(key)
            self.results[key] = row
        elif row.get("kind") == "producer":
            key = (str(row["qid"]), str(row["producer"]))
            self.reports[key] = row

    def pending(self, items: Sequence[Any], names: Iterable[str]) -> dict[str, set[str]]:
        """Return the result names each item still owes."""
        expected_names = tuple(names)
        return {
            str(item.qid): {
                name for name in expected_names if (str(item.qid), name) not in self.done
            }
            for item in items
        }

    def bank(self, row: dict[str, Any], *, attempt_id: str) -> dict[str, Any]:
        """Append one identity-stamped row, fsync it, and index it."""
        for name, expected in self.fixed_fields.items():
            if name in row and row[name] != expected:
                raise ValueError(f"row overrides fixed field {name}")
        durable = {
            **row,
            **self.fixed_fields,
            "attempt_id": attempt_id,
            "banked_at_unix": time.time(),
        }
        encoded = json.dumps(durable, sort_keys=True, allow_nan=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._remember(durable)
        return durable

    def assert_complete(self, expected: set[ResultKey]) -> None:
        """Reread the bank from disk and require exactly this result key set."""
        durable: set[ResultKey] = set()
        for row in self._read_rows():
            self._validate_identity(row)
            if row.get("kind") == "result":
                durable.add((str(row["qid"]), str(row[self.result_field])))
        if durable != expected:
            missing = sorted(expected - durable)
            extra = sorted(durable - expected)
            raise RuntimeError(
                f"durable result key mismatch: missing={missing[:3]}, extra={extra[:3]}"
            )


_WORKER_DIR = "workers"
_BANK_FILENAME = "raw.jsonl"
_RESULT_FIELD = "cell"


@dataclass(frozen=True)
class CellBankSpec:
    """The per-family cell vocabulary this bank machinery is bound to."""

    cell_schema: str
    launcher_label: str
    lock_filename: str
    decode_cell: Callable[[str, int], str]
    report_producer: Callable[[int], str]
    capture_producer: Callable[[int], str]


def worker_bank_path(run_root: Path, worker_index: int) -> Path:
    """Return where one GPU worker appends its rows."""
    return Path(run_root) / _WORKER_DIR / f"gpu{worker_index}" / _BANK_FILENAME


def round_robin_items(qids: Sequence[str], gpus: int) -> tuple[tuple[str, ...], ...]:
    """Assign whole items round-robin across a node's GPUs.

    Items are assigned whole, so every arm of one item decodes on one device and
    a matched comparison never straddles GPUs under different memory pressure.
    """
    buckets: list[list[str]] = [[] for _ in range(gpus)]
    for index, qid in enumerate(qids):
        buckets[index % gpus].append(str(qid))
    return tuple(tuple(bucket) for bucket in buckets)


@contextmanager
def worker_bank_lock(run_root: Path, worker_index: int, spec: CellBankSpec) -> Generator[None]:
    """Refuse two processes writing the same GPU bank concurrently.

    The lock is non-blocking, so the second caller fails at once rather than
    queueing behind a process that may itself be shutting down.
    """
    lock_path = Path(run_root) / _WORKER_DIR / f"gpu{worker_index}" / spec.lock_filename
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"{spec.launcher_label} worker {worker_index} already has an active launcher"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def open_bank(
    run_root: Path,
    worker_index: int,
    fixed_fields: Mapping[str, Any],
    spec: CellBankSpec,
) -> DurableRunLog:
    """Open one worker's bank, repairing a torn tail."""
    return DurableRunLog(
        worker_bank_path(run_root, worker_index),
        result_field=_RESULT_FIELD,
        fixed_fields={**dict(fixed_fields), "cell_schema": spec.cell_schema},
    )


@dataclass(frozen=True)
class _Item:
    """The one attribute :meth:`DurableRunLog.pending` reads off an item."""

    qid: str


@dataclass
class ShardLog:
    """One GPU worker's durable, attempt-aware bank."""

    log: DurableRunLog
    attempt_id: str
    spec: ClassVar[CellBankSpec]

    @property
    def path(self) -> Path:
        """Return where this worker's rows are appended."""
        return self.log.path

    def pending_cells(self, qids: Sequence[str], cells: Iterable[str]) -> dict[str, set[str]]:
        """Return the cells each item still owes."""
        return self.log.pending([_Item(str(qid)) for qid in qids], cells)

    def banked_decode(self, qid: str, arm: str, seed_index: int) -> dict[str, Any] | None:
        """Return the banked row for one cell, or None when it is still to decode."""
        return self.log.results.get((str(qid), self.spec.decode_cell(arm, seed_index)))

    def banked_report(self, qid: str, worker: int) -> dict[str, Any] | None:
        """Return the banked text report for one worker, or None."""
        return self.log.reports.get((str(qid), self.spec.report_producer(worker)))

    def banked_capture(self, qid: str, worker: int) -> dict[str, Any] | None:
        """Return the banked capture timing row for one worker, or None."""
        return self.log.reports.get((str(qid), self.spec.capture_producer(worker)))

    def bank_decode(self, row: Mapping[str, Any], *, arm: str, seed_index: int) -> dict[str, Any]:
        """Append one decoded cell, fsynced, stamped with this attempt."""
        return self.log.bank(
            {**dict(row), "kind": "result", _RESULT_FIELD: self.spec.decode_cell(arm, seed_index)},
            attempt_id=self.attempt_id,
        )

    def bank_producer(self, row: Mapping[str, Any], *, qid: str, producer: str) -> dict[str, Any]:
        """Append one producer row under its own cell key."""
        return self.log.bank(
            {**dict(row), "kind": "producer", "qid": str(qid), "producer": producer},
            attempt_id=self.attempt_id,
        )

    def bank_report(self, row: Mapping[str, Any], *, qid: str, worker: int) -> dict[str, Any]:
        """Append one worker report, fsynced, stamped with this attempt."""
        return self.bank_producer(row, qid=qid, producer=self.spec.report_producer(worker))

    def bank_capture(self, row: Mapping[str, Any], *, qid: str, worker: int) -> dict[str, Any]:
        """Append one capture timing row, bound to its layer bank."""
        return self.bank_producer(row, qid=qid, producer=self.spec.capture_producer(worker))

    def banked_rows(self, qids: Sequence[str], cells: Sequence[str]) -> list[dict[str, Any]]:
        """Return every banked decode row for these items, in (item, cell) order.

        A run continuing an existing bank publishes these verbatim beside the
        cells it decodes itself, so the table is the same across launches.
        """
        rows: list[dict[str, Any]] = []
        for qid in qids:
            for cell in cells:
                row = self.log.results.get((str(qid), cell))
                if row is not None:
                    rows.append(row)
        return rows

    def assert_complete(self, qids: Sequence[str], cells: Sequence[str]) -> None:
        """Reread the bank and require exactly this cell set, from disk."""
        self.log.assert_complete({(str(qid), cell) for qid in qids for cell in cells})


__all__ = (
    "CellBankSpec",
    "DurableRunLog",
    "ResultKey",
    "ShardLog",
    "open_bank",
    "round_robin_items",
    "runtime_fingerprint",
    "runtime_signature",
    "worker_bank_lock",
)
