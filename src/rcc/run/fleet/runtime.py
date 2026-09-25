"""Single-node fleet lifecycle around benchmark-supplied adapters.

One seat warms its adapter, joins the shared batch-arrival clock, and then
serves its role. The per-arm runner calls this once per seat.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from rcc.hardware.fleet import FleetPlacement
from rcc.run.fleet.contract import (
    AsyncReceiver,
    FusedAdapter,
    ProducedArtifact,
    ReceiverCompletion,
    SplitProducer,
    WorkItem,
)
from rcc.run.fleet.progress import (
    SeatProgress,
    install_seat_progress,
    seat_progress_path,
)
from rcc.run.fleet.queue import (
    Ticket,
    arm_queue_roots,
    claim_next,
    count_ready_workers,
    publish_clock,
    publish_ticket,
    publish_worker_ready,
    queue_dirs,
    read_ticket,
    release_claim,
    requeue_claim,
    reset_stale_claims,
    ticket_filename,
    verify_ticket_payload,
)
from rcc.run.io import atomic_json, sha256_file


@dataclass(frozen=True)
class FleetRunConfig:
    """One seat's identity and lifecycle bounds for one arm attempt."""

    root: Path
    arm: str
    attempt_id: str
    qids: tuple[str, ...]
    placement: FleetPlacement
    worker_index: int
    max_wall_s: float = 14_400.0
    barrier_timeout_s: float = 3_600.0
    claim_lease_s: float = 7_200.0
    poll_s: float = 0.05

    def __post_init__(self) -> None:
        """Validate the run identity, placement, and timing bounds."""
        if not self.arm or not self.attempt_id or not self.qids:
            raise ValueError("fleet run needs an arm, attempt id, and nonempty panel")
        if len(set(self.qids)) != len(self.qids):
            raise ValueError("fleet panel qids must be unique")
        self.placement.role(self.worker_index)
        if min(self.max_wall_s, self.barrier_timeout_s, self.claim_lease_s, self.poll_s) <= 0:
            raise ValueError("fleet timeouts and poll interval must be positive")

    @property
    def items(self) -> tuple[WorkItem, ...]:
        """Return the frozen panel as ordered work items."""
        return tuple(WorkItem(qid, index) for index, qid in enumerate(self.qids))


@dataclass(frozen=True)
class FleetWorkerHooks:
    """The benchmark-owned operations the runtime calls."""

    adapter: SplitProducer | AsyncReceiver | FusedAdapter
    completed_qids: Callable[[], set[str]]
    # The bank writer returns the row it wrote. The runtime never reads that
    # row, but a hook typed as returning ``None`` would force every caller to
    # cast its own writer.
    bank_stage: Callable[[dict[str, Any]], Any]
    bank_result: Callable[[dict[str, Any]], Any]


def _atomic_marker(path: Path, payload: dict[str, Any]) -> None:
    atomic_json(path, payload)


def publish_fleet_stop(
    root: Path,
    *,
    attempt_id: str,
    worker_index: int,
    role: str,
    error: BaseException,
) -> None:
    """Raise one arm's fail-fast marker, including constructor failures."""
    control = Path(root) / "control"
    payload = {
        "attempt_id": attempt_id,
        "worker_index": worker_index,
        "role": role,
        "error": repr(error),
    }
    for stop in (control / f"FLEET_STOP.{attempt_id}", control / "FLEET_STOP"):
        if stop.is_file():
            continue
        try:
            _atomic_marker(stop, payload)
        except OSError:
            pass


class _WorkerRuntime:
    def __init__(
        self,
        config: FleetRunConfig,
        hooks: FleetWorkerHooks,
        *,
        progress: SeatProgress | None = None,
    ) -> None:
        self.config = config
        self.hooks = hooks
        self.role = config.placement.role(config.worker_index)
        self.started = time.monotonic()
        self.todo, self.ready = arm_queue_roots(config.root)
        self.barrier = config.root / "barriers" / config.attempt_id
        control = config.root / "control"
        self.stop = control / f"FLEET_STOP.{config.attempt_id}"
        self.legacy_stop = control / "FLEET_STOP"
        self.direct = config.placement.producers == 0 and not config.placement.fused
        self.claims: dict[str, Path] = {}
        self.last_refresh = 0.0
        self.last_recover = 0.0
        self.idle_polls = 0
        self.progress = progress or begin_worker_progress(config)
        install_seat_progress(self.progress)
        # A producer's post-publish audit runs on this one side thread, so the
        # seat can claim and prefill its next item meanwhile. At most one audit
        # is in flight: its bank tensors must not accumulate on the device.
        self._audit_pool: ThreadPoolExecutor | None = None
        self._audit: Future[None] | None = None

    def check(self) -> None:
        if self.stop.is_file() or self.legacy_stop.is_file():
            raise RuntimeError("FLEET_STOP is raised")
        if time.monotonic() - self.started > self.config.max_wall_s:
            raise TimeoutError(f"fleet arm exceeded {self.config.max_wall_s:g}s")

    def fail(self, error: BaseException) -> None:
        publish_fleet_stop(
            self.config.root,
            attempt_id=self.config.attempt_id,
            worker_index=self.config.worker_index,
            role=self.role,
            error=error,
        )

    def stage(self, qid: str, name: str, *, t_unix: float | None = None, **fields: Any) -> None:
        self.hooks.bank_stage(
            {
                "kind": "stage",
                "qid": qid,
                "arm": self.config.arm,
                "stage": name,
                "t_unix": time.time() if t_unix is None else float(t_unix),
                "gpu_index": self.config.worker_index,
                "role": self.role,
                **fields,
            }
        )
        # The banked row keeps the caller's ``t_unix``, often the moment the
        # measured work ended. The progress record keeps its own write clock,
        # because its reader asks how long ago this seat was last alive.
        self.progress.record(name, qid=qid)

    def sync(self) -> float:
        publish_worker_ready(self.barrier, f"gpu{self.config.worker_index}")
        self.progress.record("worker_ready")
        if self.config.worker_index == 0:
            deadline = time.monotonic() + self.config.barrier_timeout_s
            while count_ready_workers(self.barrier) < self.config.placement.workers:
                self.check()
                if time.monotonic() > deadline:
                    raise TimeoutError("fleet warm barrier timed out")
                time.sleep(self.config.poll_s)
            t0 = time.time()
            self._publish_panel(t0)
            publish_clock(
                self.barrier,
                expected_workers=self.config.placement.workers,
                t0=t0,
            )
            return t0
        deadline = time.monotonic() + self.config.barrier_timeout_s
        clock = self.barrier / "fleet_clock.json"
        while not clock.is_file():
            self.check()
            if time.monotonic() > deadline:
                raise TimeoutError("fleet clock wait timed out")
            time.sleep(self.config.poll_s)
        return float(json.loads(clock.read_text(encoding="utf-8"))["t0_unix"])

    def _publish_panel(self, t0: float) -> None:
        completed = self.hooks.completed_qids()
        for root in (self.todo, self.ready):
            reset_stale_claims(root, banked_qids=completed)
            ready, _, _ = queue_dirs(root)
            for stale in ready.glob("*.json"):
                stale.unlink(missing_ok=True)
        target = self.ready if self.direct else self.todo
        markers = self.config.root / "control" / "work_items"
        for item in self.config.items:
            if item.qid in completed:
                continue
            self.stage(item.qid, "submit", t_unix=t0)
            marker = markers / ticket_filename(item.qid, item.order_index)
            _atomic_marker(marker, {"qid": item.qid, "order_index": item.order_index})
            publish_ticket(
                target,
                Ticket(
                    qid=item.qid,
                    arm=self.config.arm,
                    order_index=item.order_index,
                    payload_path=str(marker),
                    payload_sha256=sha256_file(marker),
                    producer_gpu=-1,
                    produced_at_unix=t0,
                ),
            )

    def _all_complete(self) -> bool:
        return set(self.config.qids) <= self.hooks.completed_qids()

    def _refresh_claims(self) -> None:
        now = time.monotonic()
        if now - self.last_refresh < min(60.0, self.config.claim_lease_s / 2):
            return
        self.last_refresh = now
        for claim in tuple(self.claims.values()):
            try:
                os.utime(claim, None)
            except FileNotFoundError:
                pass

    def _recover(self, root: Path) -> None:
        now = time.monotonic()
        if now - self.last_recover < min(60.0, self.config.claim_lease_s / 2):
            return
        self.last_recover = now
        _, claimed, _ = queue_dirs(root)
        completed = self.hooks.completed_qids()
        for path in claimed.glob("*/*.json"):
            try:
                age = time.time() - path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age < self.config.claim_lease_s:
                continue
            try:
                qid = read_ticket(path).qid
            except FileNotFoundError:
                continue
            if qid in completed:
                path.unlink(missing_ok=True)
                continue
            try:
                requeue_claim(root, path, qid=qid)
            except FileNotFoundError:
                pass

    def _idle_tick(self) -> None:
        """Refresh liveness while this seat has nothing of its own to advance.

        Without this an idle receiver would freeze its own progress record for
        the length of an arm and read as stalled. ``idle`` carries no deadline.
        """
        self.idle_polls += 1
        self.progress.tick(self.idle_polls, stage="idle")

    def _drain_audit(self, *, wait: bool) -> None:
        """Surface a failed deferred audit on this thread, waiting if asked.

        The audit checks a published handoff against its capture, so its failure
        stops the fleet; the side thread only delays where it is raised.
        """
        if self._audit is None:
            return
        if wait or self._audit.done():
            future, self._audit = self._audit, None
            future.result()

    def _defer_audit(self, hook: Callable[[], None]) -> None:
        """Run one post-publish audit beside the next item, one at a time."""
        self._drain_audit(wait=True)
        if self._audit_pool is None:
            self._audit_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rcc-audit")
        self._audit = self._audit_pool.submit(hook)

    def close_audits(self) -> None:
        """Finish the in-flight audit and release its thread, raising on failure."""
        try:
            self._drain_audit(wait=True)
        finally:
            if self._audit_pool is not None:
                self._audit_pool.shutdown(wait=True)
                self._audit_pool = None

    def run_producer(self, adapter: SplitProducer) -> None:
        while True:
            self.check()
            self._drain_audit(wait=False)
            claimed = claim_next(self.todo, owner=f"gpu{self.config.worker_index}")
            if claimed is None:
                ready, claimed_dir, _ = queue_dirs(self.todo)
                if not any(ready.glob("*.json")) and not any(claimed_dir.glob("*/*.json")):
                    self.close_audits()
                    return
                self._recover(self.todo)
                self._idle_tick()
                time.sleep(self.config.poll_s)
                continue
            ticket, claim_path = claimed
            item = WorkItem(ticket.qid, ticket.order_index)
            self.stage(item.qid, "producer_claim")
            self.stage(item.qid, "producer_start")
            artifact = adapter.produce(item)
            if not artifact.path.is_file():
                raise FileNotFoundError(f"{item.qid}: producer artifact missing at {artifact.path}")
            self.stage(item.qid, "producer_end", t_unix=artifact.producer_finished_at)
            # The previous item's audit ran beside this capture and has to have
            # passed before this handoff becomes claimable; a failure stops the
            # fleet with nothing published.
            self._drain_audit(wait=True)
            publish_ticket(
                self.ready,
                Ticket(
                    qid=item.qid,
                    arm=self.config.arm,
                    order_index=item.order_index,
                    payload_path=str(artifact.path),
                    payload_sha256=sha256_file(artifact.path),
                    producer_gpu=self.config.worker_index,
                    produced_at_unix=artifact.producer_finished_at,
                    fields=dict(artifact.fields),
                ),
                # The receiver may claim immediately after the atomic rename,
                # so the boundary is banked while the ticket is still staged
                # and receiver_claim can never precede handoff_ready.
                before_publish=lambda qid=item.qid: self.stage(qid, "handoff_ready"),
            )
            release_claim(self.todo, claim_path, qid=item.qid)
            if artifact.after_publish is not None:
                # Off the seat's critical path: run inline, the audit would sit
                # between two items on every latent item of the arm.
                self._defer_audit(artifact.after_publish)
            artifact = None

    def _submit_receiver(self, adapter: AsyncReceiver, item: WorkItem, ticket: Ticket) -> None:
        verify_ticket_payload(ticket)
        artifact = None
        if ticket.producer_gpu >= 0:
            # The producer's own measurements travel on the ticket, so the
            # receiver sees the artifact it published, fields and all.
            artifact = ProducedArtifact(
                Path(ticket.payload_path),
                ticket.produced_at_unix,
                dict(ticket.fields),
            )
        adapter.submit(item, artifact)

    def _finish(
        self,
        completions: Sequence[ReceiverCompletion],
        *,
        fused: bool,
        source: Path,
    ) -> None:
        stage_source = cast(FusedAdapter, self.hooks.adapter) if fused else None
        for completion in completions:
            if completion.qid not in self.claims:
                raise RuntimeError(f"{completion.qid}: completion has no live claim")
            if fused:
                assert stage_source is not None
                times = stage_source.stage_times(completion.qid)
                for name in ("producer_end", "handoff_ready", "receiver_claim", "payload_loaded"):
                    self.stage(completion.qid, name, t_unix=times[name])
            self.stage(
                completion.qid,
                "admitted",
                t_unix=completion.admitted_at,
                **completion.fields,
            )
            if completion.first_token_at is not None:
                self.stage(completion.qid, "first_token", t_unix=completion.first_token_at)
            self.stage(completion.qid, "decode_end", t_unix=completion.finished_at)
            result = dict(completion.result)
            if result.get("kind") != "result" or str(result.get("qid")) != completion.qid:
                raise ValueError(f"{completion.qid}: adapter returned a malformed result row")
            self.hooks.bank_result(result)
            release_claim(source, self.claims.pop(completion.qid), qid=completion.qid)

    def run_receiver(self, adapter: AsyncReceiver, *, fused: bool) -> None:
        source = self.todo if fused else self.ready
        while True:
            self.check()
            self._refresh_claims()
            claimed = None
            if adapter.can_accept():
                claimed = claim_next(source, owner=f"gpu{self.config.worker_index}")
            if claimed is not None:
                ticket, claim_path = claimed
                item = WorkItem(ticket.qid, ticket.order_index)
                self.claims[item.qid] = claim_path
                if fused:
                    self.stage(item.qid, "producer_claim")
                    self.stage(item.qid, "producer_start")
                    adapter.submit(item, None)
                else:
                    self.stage(item.qid, "receiver_claim")
                    self._submit_receiver(adapter, item, ticket)
                    self.stage(item.qid, "payload_loaded")
            completions = adapter.pump()
            if completions:
                self._finish(completions, fused=fused, source=source)
            # `completed_qids` reads the banks from disk, so the two in-memory
            # predicates are asked first and a busy receiver never pays a bank
            # scan between engine steps.
            idle = adapter.idle()
            if idle and not self.claims and self._all_complete():
                return
            # A busy asynchronous adapter advances its decode engine even when
            # no request completed, so sleeping here would cap it at 1 / poll_s
            # steps per second. Poll only when there is no in-memory work.
            if idle and claimed is None and not completions:
                self._recover(source)
                self._idle_tick()
                time.sleep(self.config.poll_s)


def begin_worker_progress(config: FleetRunConfig) -> SeatProgress:
    """Start this seat's progress record before its adapter is constructed."""
    progress = SeatProgress(
        seat_progress_path(config.root, config.worker_index),
        arm=config.arm,
        worker_index=config.worker_index,
        role=config.placement.role(config.worker_index),
        attempt_id=config.attempt_id,
    )
    install_seat_progress(progress)
    progress.record("engine_open")
    return progress


def run_worker(
    config: FleetRunConfig,
    hooks: FleetWorkerHooks,
    *,
    progress: SeatProgress | None = None,
) -> float:
    """Warm one adapter, join the common clock, and serve its declared role.

    Returns the shared batch-arrival clock. A fault in any seat raises the arm's
    stop marker, so its peers fail fast.
    """
    runtime = _WorkerRuntime(config, hooks, progress=progress)
    failed = False
    try:
        hooks.adapter.warm()
        t0 = runtime.sync()
        if runtime.role == "producer":
            runtime.run_producer(hooks.adapter)  # type: ignore[arg-type]
        elif runtime.role == "fused":
            runtime.run_receiver(hooks.adapter, fused=True)  # type: ignore[arg-type]
        else:
            runtime.run_receiver(hooks.adapter, fused=False)  # type: ignore[arg-type]
        return t0
    except BaseException as error:
        failed = True
        runtime.fail(error)
        raise
    finally:
        runtime.progress.record("engine_close")
        try:
            _close_seat(runtime, hooks.adapter)
        except BaseException as error:
            failed = True
            runtime.fail(error)
            raise
        finally:
            # A seat holds no deadline only once engine teardown has returned.
            # Until then `engine_close` is what a supervisor reads.
            runtime.progress.record("failed" if failed else "closed")
            install_seat_progress(None)


def _close_seat(runtime: _WorkerRuntime, adapter: Any) -> None:
    """Finish the seat's audit, then close its engine, whatever the audit did.

    A producer that failed mid-arm may still hold a running audit, which finishes
    or fails here; its failure must not skip the adapter close.
    """
    audit_error: BaseException | None = None
    try:
        runtime.close_audits()
    except BaseException as error:
        audit_error = error
    adapter.close()
    if audit_error is not None:
        raise audit_error


def publish_worker_failure(config: FleetRunConfig, error: BaseException) -> None:
    """Publish an attempt-scoped stop for failures before ``run_worker``."""
    publish_fleet_stop(
        config.root,
        attempt_id=config.attempt_id,
        worker_index=config.worker_index,
        role=config.placement.role(config.worker_index),
        error=error,
    )


def clear_stale_stop_markers(root: Path, *, attempt_id: str | None = None) -> None:
    """Clear other attempts' stop markers, keeping a hand-written stop."""
    control = Path(root) / "control"
    if not control.is_dir():
        return
    for marker in control.glob("FLEET_STOP*"):
        if attempt_id is not None and marker.name == f"FLEET_STOP.{attempt_id}":
            continue
        if marker.name == "FLEET_STOP" and attempt_id is None:
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
        try:
            marker.unlink()
        except FileNotFoundError:
            pass
