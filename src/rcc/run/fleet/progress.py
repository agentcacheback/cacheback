"""Per-seat progress records: what a seat is doing and when it last said so.

A hung seat raises nothing and publishes nothing, so each seat rewrites a small
JSON file whose mtime is its liveness and whose ``stage`` names its deadline.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# A tick is a liveness refresh, not a measurement. Five seconds is short
# against every deadline below and long enough that the write cost is
# invisible inside a decode or extraction span.
PROGRESS_TICK_SECONDS = 5.0

# Seconds a seat may publish nothing inside a stage before it reads as hung.
# The table is a deny-list: every stage is watched at the default unless named
# exempt below, since a receiver can sit a whole decode in `payload_loaded`.
SEAT_DEADLINE_DEFAULT = 2_700.0
SEAT_STAGE_DEADLINES: dict[str, float] = {
    "engine_open": 1_200.0,
    "producer_start": 900.0,
    "receiver_claim": 300.0,
}
# A seat in one of these is parked with nothing to advance: an idle receiver
# may sit out a whole arm, a seat at the barrier waits on its peers under the
# barrier's own timeout, and a finished seat holds no deadline at all.
SEAT_DEADLINE_EXEMPT = frozenset({"idle", "worker_ready", "closed", "failed"})


def seat_stage_deadline(stage: str) -> float:
    """Return the stall bound for one stage, in seconds, or 0.0 when exempt.

    A text producer publishes nothing while one blocking generate call runs its
    seed ladder, so ``producer_start`` reads ``FANOUT_PRODUCER_STALL_SECONDS``.
    """
    if not stage or stage in SEAT_DEADLINE_EXEMPT:
        return 0.0
    if stage == "producer_start":
        return float(os.environ.get("FANOUT_PRODUCER_STALL_SECONDS", SEAT_STAGE_DEADLINES[stage]))
    return SEAT_STAGE_DEADLINES.get(stage, SEAT_DEADLINE_DEFAULT)


_SEAT_OPEN_STAGE = "engine_open"
_LOCAL = threading.local()


class SeatProgress:
    """One seat's rewritable progress record under its arm control tree."""

    def __init__(
        self,
        path: Path,
        *,
        arm: str,
        worker_index: int,
        role: str,
        attempt_id: str,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> None:
        """Bind one seat's identity, record path, and injectable clocks."""
        self.path = Path(path)
        self.arm = arm
        self.worker_index = int(worker_index)
        self.role = role
        self.attempt_id = attempt_id
        self.pid = os.getpid()
        self._clock = clock
        self._wall = wall
        self._staging = self.path.with_name(f".{self.path.name}.{self.pid}.tmp")
        self._qid = "-"
        self._stage = _SEAT_OPEN_STAGE
        self._steps = 0
        self._last_write: float | None = None

    def record(self, stage: str, *, qid: str | None = None, steps: int | None = None) -> None:
        """Publish a stage transition, whatever the tick interval says."""
        self._stage = stage
        if qid is not None:
            self._qid = qid
        if steps is not None:
            self._steps = int(steps)
        self._write()

    def tick(self, steps: int, *, stage: str | None = None) -> None:
        """Refresh liveness and the step counter, at most once per interval."""
        now = self._clock()
        if self._last_write is not None and now - self._last_write < PROGRESS_TICK_SECONDS:
            return
        if stage is not None:
            self._stage = stage
        self._steps = int(steps)
        self._write()

    def _write(self) -> None:
        """Publish the record through a staged file and ``os.replace``, no fsync.

        An ``fsync`` would block on the very volume whose stalling is watched for.
        """
        payload: dict[str, Any] = {
            "arm": self.arm,
            "attempt_id": self.attempt_id,
            "pid": self.pid,
            "qid": self._qid,
            "role": self.role,
            "stage": self._stage,
            "steps": self._steps,
            "t_unix": round(self._wall(), 3),
            "worker_index": self.worker_index,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._staging.open("w", encoding="utf-8") as handle:
                handle.write(encoded)
            os.replace(self._staging, self.path)
        except OSError:
            # A progress write never fails the run. A record that cannot be
            # written reads as a stalled seat, which is the safe direction.
            return
        self._last_write = self._clock()


def seat_progress_path(root: Path, worker_index: int) -> Path:
    """Return one seat's record path under its arm root."""
    return Path(root) / "control" / "progress" / f"gpu{int(worker_index)}.json"


def install_seat_progress(progress: SeatProgress | None) -> None:
    """Install (or clear) the calling seat thread's progress writer.

    The holder is thread-local, which is correct whether a seat is one process
    or one thread, and never lets two seats publish over one record.
    """
    _LOCAL.progress = progress


def seat_progress() -> SeatProgress | None:
    """Return the calling seat thread's progress writer, if one is installed."""
    return getattr(_LOCAL, "progress", None)


def tick_seat_progress(steps: int) -> None:
    """Refresh the calling seat's liveness from inside an engine loop."""
    progress = seat_progress()
    if progress is not None:
        progress.tick(steps)


__all__ = (
    "PROGRESS_TICK_SECONDS",
    "SEAT_DEADLINE_DEFAULT",
    "SEAT_DEADLINE_EXEMPT",
    "SEAT_STAGE_DEADLINES",
    "SeatProgress",
    "install_seat_progress",
    "seat_progress",
    "seat_progress_path",
    "seat_stage_deadline",
    "tick_seat_progress",
)
