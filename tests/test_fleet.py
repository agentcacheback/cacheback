"""The shared split-fleet runtime: the arm barrier, the seat runner, and the stream handle."""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from functools import partial
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import rcc.run.fleet.runtime as fleet_runtime
from rcc.hardware.fleet import FleetPlacement
from rcc.models.gemma.engine import Sampling
from rcc.models.nemotron import NEMOTRON_FAMILY
from rcc.models.qwen import QWEN_FAMILY
from rcc.models.qwen.text import QwenDecodeSpec
from rcc.run import banks, io
from rcc.run.barrier import (
    ARM_PHASES,
    BarrierRun,
    BarrierSpec,
    arm_marker_times,
    arm_phase_roster,
    arm_role_roster,
)
from rcc.run.fleet.contract import (
    ProducedArtifact,
    ReceiverCompletion,
    WorkItem,
)
from rcc.run.fleet.ledger import strict_stage_decomposition
from rcc.run.fleet.progress import (
    PROGRESS_TICK_SECONDS,
    SEAT_DEADLINE_DEFAULT,
    SEAT_DEADLINE_EXEMPT,
    SEAT_STAGE_DEADLINES,
    SeatProgress,
    install_seat_progress,
    seat_progress_path,
    seat_stage_deadline,
)
from rcc.run.fleet.queue import (
    Ticket,
    claim_next,
    publish_ticket,
    queue_dirs,
    reset_stale_claims,
    verify_ticket_payload,
)
from rcc.run.fleet.runtime import (
    FleetRunConfig,
    FleetWorkerHooks,
    clear_stale_stop_markers,
    publish_worker_failure,
    run_worker,
)
from rcc.run.fleet.schedule import run_in_processes, stop_seats_on_signal
from rcc.run.fleet.stream import StreamDecoder, StreamRequest
from rcc.run.fleet.vllm import VllmEngineHandle
from rcc.run.io import sha256_file
from rcc.run.runlog import DurableRunLog

# --- The role-aware arm barrier.

SPEC = BarrierSpec(
    family_label="split",
    marker_schema="rcc-split-arm-barrier-marker-v1",
    barrier_dirname="arm-barrier",
)


SPLIT = FleetPlacement(4, 1, 3)


FUSED = FleetPlacement(4, 0, 4, fused=True)


NO_PRODUCER = FleetPlacement(4, 0, 4)


def _run(root: Path, *, node_gpus: int = 4, attempt_id: str = "a1") -> BarrierRun:
    return BarrierRun(
        spec=SPEC,
        run_root=root,
        attempt_id=attempt_id,
        plan_fingerprint="plan",
        roster_fingerprint="roster",
        node_gpus=node_gpus,
    )


def _mark(
    run: BarrierRun,
    *,
    arm: str,
    phase: str,
    state: str,
    workers: list[int],
    at: float,
    marker_arm: str | None = None,
) -> None:
    for worker in workers:
        identity = run.arm_marker_identity(
            worker_index=worker, arm=marker_arm or arm, phase=phase, state=state
        )
        io.atomic_json(
            run.arm_phase_root(arm, phase) / state / f"gpu{worker}.json",
            {**identity, "utc_unix": at},
        )


def _active_workers(placement: FleetPlacement) -> tuple[list[int], list[int]]:
    """Return the producer and receiver marker rosters of one placement."""
    roster = arm_role_roster(placement)
    if placement.fused:
        return [], list(roster["fused"])
    return list(roster["producers"]), list(roster["receivers"])


def _drive_arm(
    run: BarrierRun,
    *,
    arm: str,
    placement: FleetPlacement,
    start: float,
    overlap: bool = True,
) -> float:
    """Publish one whole arm's markers and return the arm's finish time."""
    everyone = list(range(placement.workers))
    producers, receivers = _active_workers(placement)
    clock = start
    for state in ("entered", "finished"):
        _mark(run, arm=arm, phase="arm_open", state=state, workers=everyone, at=clock)
        clock += 1.0
    producer_start = clock
    if producers:
        _mark(
            run,
            arm=arm,
            phase="producers_active",
            state="entered",
            workers=producers,
            at=producer_start,
        )
    receiver_start = producer_start if overlap else producer_start + 10.0
    _mark(
        run,
        arm=arm,
        phase="receivers_active",
        state="entered",
        workers=receivers,
        at=receiver_start,
    )
    if producers:
        _mark(
            run,
            arm=arm,
            phase="producers_active",
            state="finished",
            workers=producers,
            at=producer_start + 5.0,
        )
    clock = receiver_start + 20.0
    _mark(
        run,
        arm=arm,
        phase="receivers_active",
        state="finished",
        workers=receivers,
        at=clock,
    )
    for state in ("entered", "finished"):
        clock += 1.0
        _mark(run, arm=arm, phase="arm_complete", state=state, workers=everyone, at=clock)
    return clock


def test_each_placement_publishes_its_own_role_roster_and_markers(tmp_path: Path) -> None:
    """FleetPlacement.role returns "fused" for every worker, so does the roster."""
    assert ARM_PHASES == (
        "arm_open",
        "producers_active",
        "receivers_active",
        "arm_complete",
    )
    assert arm_phase_roster(("floor", "r2")) == (
        "floor:arm_open",
        "floor:producers_active",
        "floor:receivers_active",
        "floor:arm_complete",
        "r2:arm_open",
        "r2:producers_active",
        "r2:receivers_active",
        "r2:arm_complete",
    )
    with pytest.raises(ValueError, match="unique"):
        arm_phase_roster(("r2", "r2"))
    with pytest.raises(ValueError, match="at least one arm"):
        arm_phase_roster(())

    # A split arm names its producers below its receivers, a producerless arm
    # names no producer at all, and a fused arm names one roster for both.
    assert arm_role_roster(SPLIT) == {"producers": (0,), "receivers": (1, 2, 3)}
    assert arm_role_roster(NO_PRODUCER) == {"producers": (), "receivers": (0, 1, 2, 3)}
    assert arm_role_roster(FUSED) == {"fused": (0, 1, 2, 3)}
    assert {FUSED.role(worker) for worker in range(FUSED.workers)} == {"fused"}

    # Every phase of a driven arm lands one marker per worker in its roster.
    run = _run(tmp_path)
    _drive_arm(run, arm="text_primary", placement=FUSED, start=100.0)
    for phase in ARM_PHASES:
        for state in ("entered", "finished"):
            times = arm_marker_times(run, arm="text_primary", phase=phase, state=state)
            assert set(times) == {"gpu0", "gpu1", "gpu2", "gpu3"} or times == {}
    opened = arm_marker_times(run, arm="text_primary", phase="arm_open", state="entered")
    closed = arm_marker_times(run, arm="text_primary", phase="arm_complete", state="finished")
    assert max(opened.values()) < min(closed.values())


_PROGRESS_FIELDS = frozenset(
    {
        "arm",
        "attempt_id",
        "pid",
        "qid",
        "role",
        "stage",
        "steps",
        "t_unix",
        "worker_index",
    }
)


class _SharedBank:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.results: dict[str, dict[str, Any]] = {}
        self.stages: list[dict[str, Any]] = []

    def completed(self) -> set[str]:
        with self.lock:
            return set(self.results)

    def bank_result(self, row: dict[str, Any]) -> None:
        with self.lock:
            qid = str(row["qid"])
            if qid in self.results:
                raise RuntimeError(f"duplicate result for {qid}")
            self.results[qid] = dict(row)

    def bank_stage(self, row: dict[str, Any]) -> None:
        with self.lock:
            self.stages.append(dict(row))


class _Producer:
    def __init__(
        self,
        root: Path,
        worker: int,
        *,
        audit_fail_qid: str | None = None,
        produce_fail_qid: str | None = None,
    ) -> None:
        self.root = root
        self.worker = worker
        self.warmed = False
        self.audit_fail_qid = audit_fail_qid
        self.produce_fail_qid = produce_fail_qid
        # (event, qid, thread name, monotonic stamp): the overlap evidence.
        self.trace: list[tuple[str, str, str, float]] = []

    def warm(self) -> None:
        self.warmed = True

    def _audit(self, qid: str) -> Callable[[], None]:
        def run() -> None:
            self.trace.append(
                ("audit_start", qid, threading.current_thread().name, time.monotonic())
            )
            # A bounded GIL-releasing wait stands in for the bank save, fsync,
            # and replay (disk and torch work that also release the GIL), so the
            # next produce can be observed starting while this one is running.
            threading.Event().wait(0.02)
            if qid == self.audit_fail_qid:
                # Fails the way a real replay does: after the work, while the
                # seat has already moved on to its next item.
                raise RuntimeError(f"{qid}: durable replay differs")
            self.trace.append(("audit_end", qid, threading.current_thread().name, time.monotonic()))

        return run

    def produce(self, item: WorkItem) -> ProducedArtifact:
        assert self.warmed
        self.trace.append(("produce", item.qid, threading.current_thread().name, time.monotonic()))
        if item.qid == self.produce_fail_qid:
            raise RuntimeError(f"{item.qid}: capture failed mid-arm")
        path = self.root / "payloads" / f"{item.qid}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"worker={self.worker};qid={item.qid}\n", encoding="utf-8")
        return ProducedArtifact(path, time.time(), after_publish=self._audit(item.qid))

    def close(self) -> None:
        self.trace.append(("close", "", threading.current_thread().name, time.monotonic()))


class _Receiver:
    def __init__(self) -> None:
        self.queue: list[tuple[WorkItem, ProducedArtifact | None]] = []

    def warm(self) -> None:
        pass

    def can_accept(self) -> bool:
        return len(self.queue) < 2

    def submit(self, item: WorkItem, artifact: ProducedArtifact | None) -> None:
        self.queue.append((item, artifact))

    def pump(self) -> list[ReceiverCompletion]:
        if not self.queue:
            return []
        item, artifact = self.queue.pop(0)
        if artifact is not None:
            assert item.qid in artifact.path.read_text(encoding="utf-8")
        now = time.time()
        return [
            ReceiverCompletion(
                qid=item.qid,
                result={"kind": "result", "qid": item.qid, "arm": "arm"},
                admitted_at=now,
                first_token_at=now,
                finished_at=now,
            )
        ]

    def idle(self) -> bool:
        return not self.queue

    def close(self) -> None:
        pass


class _MultiPumpReceiver(_Receiver):
    """Require several engine steps before one request completes."""

    def __init__(self, record: Path | None = None) -> None:
        super().__init__()
        self.remaining = 0
        self.record = record
        self.stage_during_decode: str | None = None
        self.stage_during_close: str | None = None

    def can_accept(self) -> bool:
        return not self.queue

    def submit(self, item: WorkItem, artifact: ProducedArtifact | None) -> None:
        super().submit(item, artifact)
        self.remaining = 5

    def pump(self) -> list[ReceiverCompletion]:
        if not self.queue:
            return []
        # Read the seat's own record from inside the decode, the only place
        # the watchdog's blind spot shows: by the time the loop returns the
        # later stages have overwritten whatever stage the seat was parked in.
        if self.record is not None and self.stage_during_decode is None:
            if self.record.is_file():
                self.stage_during_decode = str(
                    json.loads(self.record.read_text(encoding="utf-8"))["stage"]
                )
        self.remaining -= 1
        if self.remaining:
            return []
        return super().pump()

    def close(self) -> None:
        """Capture the supervised stage that covers engine teardown."""
        if self.record is not None and self.record.is_file():
            self.stage_during_close = str(
                json.loads(self.record.read_text(encoding="utf-8"))["stage"]
            )


def _drive(
    root: Path,
    bank: _SharedBank,
    placement: FleetPlacement,
    *,
    attempt: str,
    producers: dict[int, _Producer] | None = None,
    expect_failure: bool = False,
    audit_fail_qid: str | None = None,
    produce_fail_qid: str | None = None,
) -> None:
    failures: list[BaseException] = []

    def one(worker: int) -> None:
        role = placement.role(worker)
        if role == "producer":
            adapter: Any = _Producer(
                root,
                worker,
                audit_fail_qid=audit_fail_qid,
                produce_fail_qid=produce_fail_qid,
            )
            if producers is not None:
                producers[worker] = adapter
        else:
            adapter = _Receiver()
        hooks = FleetWorkerHooks(adapter, bank.completed, bank.bank_stage, bank.bank_result)
        try:
            run_worker(
                FleetRunConfig(
                    root=root,
                    arm="arm",
                    attempt_id=attempt,
                    qids=tuple(f"q{i}" for i in range(9)),
                    placement=placement,
                    worker_index=worker,
                    max_wall_s=10,
                    barrier_timeout_s=5,
                    claim_lease_s=2,
                    poll_s=0.001,
                ),
                hooks,
            )
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=one, args=(worker,)) for worker in range(placement.workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)
    assert not [thread for thread in threads if thread.is_alive()]
    if expect_failure:
        assert failures, "an audit failure must surface as a worker failure"
        return
    assert not failures


def test_generic_runtime_streams_split_pool_and_a_second_attempt_banks_nothing(
    tmp_path: Path,
):
    bank = _SharedBank()
    placement = FleetPlacement(workers=4, producers=2, receivers=2)
    producers: dict[int, _Producer] = {}
    _drive(tmp_path, bank, placement, attempt="a1", producers=producers)

    # The post-publish audit runs on the seat's side thread, one at a time,
    # while the seat produces its next item. That overlap is what an inline
    # audit costs per item, and every audit finishes before the seat closes.
    overlapped = 0
    for producer in producers.values():
        trace = producer.trace
        audits = [row for row in trace if row[0] == "audit_start"]
        assert audits and all(row[2].startswith("rcc-audit") for row in audits)
        assert sum(row[0] == "audit_end" for row in trace) == len(audits)
        produced = [row for row in trace if row[0] == "produce"]
        ended = {row[1]: row[3] for row in trace if row[0] == "audit_end"}
        for earlier, later in pairwise(produced):
            # The seat produced the next item before the previous audit ended.
            overlapped += ended[earlier[1]] > later[3]
    assert overlapped, "every audit finished before the next item was produced"

    assert set(bank.results) == {f"q{i}" for i in range(9)}
    assert {row["role"] for row in bank.stages} == {"producer", "receiver"}
    assert sum(row["stage"] == "submit" for row in bank.stages) == 9
    assert sum(row["stage"] == "producer_end" for row in bank.stages) == 9
    assert sum(row["stage"] == "decode_end" for row in bank.stages) == 9
    strict_stage_decomposition(
        [{**row, "attempt_id": "a1"} for row in (*bank.stages, *bank.results.values())],
        "arm",
    )

    sizes = (len(bank.results), len(bank.stages))
    _drive(tmp_path, bank, placement, attempt="a2")
    assert (len(bank.results), len(bank.stages)) == sizes

    # A deferred audit that fails is a fleet stop, raised on the seat before
    # its next handoff is published: the seat that produced q3 goes on to
    # capture one more item, and that item never becomes claimable.
    failing_root = tmp_path / "audit-fails"
    failing_root.mkdir()
    failing: dict[int, _Producer] = {}
    failing_bank = _SharedBank()
    _drive(
        failing_root,
        failing_bank,
        placement,
        attempt="a3",
        producers=failing,
        expect_failure=True,
        audit_fail_qid="q3",
    )
    (seat,) = [p for p in failing.values() if ("produce", "q3") in {r[:2] for r in p.trace}]
    produced = [row[1] for row in seat.trace if row[0] == "produce"]
    after = produced[produced.index("q3") + 1 :]
    assert len(after) <= 1
    published = {row["qid"] for row in failing_bank.stages if row["stage"] == "handoff_ready"}
    assert "q3" in published and not (set(after) & published)

    # A seat that fails mid-arm may still own a running audit. It ends in
    # run_worker's finally, before the adapter is closed and its engine torn
    # down from under the audit's tensors, so no audit is left without an end.
    crashing_root = tmp_path / "produce-fails"
    crashing_root.mkdir()
    crashing: dict[int, _Producer] = {}
    _drive(
        crashing_root,
        _SharedBank(),
        placement,
        attempt="a4",
        producers=crashing,
        expect_failure=True,
        produce_fail_qid="q7",
    )
    (crashed,) = [p for p in crashing.values() if ("produce", "q7") in {r[:2] for r in p.trace}]
    closed_at = next(row[3] for row in crashed.trace if row[0] == "close")
    starts = [row for row in crashed.trace if row[0] == "audit_start"]
    ends = [row for row in crashed.trace if row[0] == "audit_end"]
    assert starts and len(ends) == len(starts)
    assert all(row[3] <= closed_at for row in ends)

    # When that teardown audit itself fails, the seat still closes its engine:
    # one producer, q3's audit fails while q4's capture raises, so the audit
    # failure surfaces inside teardown without skipping the close.
    solo_root = tmp_path / "audit-fails-at-close"
    solo_root.mkdir()
    solo: dict[int, _Producer] = {}
    _drive(
        solo_root,
        _SharedBank(),
        FleetPlacement(workers=2, producers=1, receivers=1),
        attempt="a5",
        producers=solo,
        expect_failure=True,
        audit_fail_qid="q3",
        produce_fail_qid="q4",
    )
    (only,) = solo.values()
    assert ("produce", "q4") in {r[:2] for r in only.trace}
    assert any(row[0] == "close" for row in only.trace)

    # Every seat leaves one terminal record the supervisors can read, and no
    # staging file survives the run.
    for worker in range(4):
        record = json.loads(
            seat_progress_path(tmp_path, worker).read_text(encoding="utf-8"),
        )
        assert set(record) == _PROGRESS_FIELDS
        assert record["stage"] == "closed"
        assert record["arm"] == "arm"
        assert record["attempt_id"] == "a2"
        assert record["worker_index"] == worker
        assert record["role"] == placement.role(worker)
    directory = seat_progress_path(tmp_path, 0).parent
    assert sorted(path.name for path in directory.iterdir()) == [
        f"gpu{worker}.json" for worker in range(4)
    ]


def test_generic_runtime_supports_direct_receiver_fleet(tmp_path: Path):
    bank = _SharedBank()
    _drive(tmp_path, bank, FleetPlacement(workers=4, producers=0, receivers=4), attempt="a1")
    assert len(bank.results) == 9
    assert {row["role"] for row in bank.stages} == {"receiver"}
    assert not any(row["stage"] == "producer_start" for row in bank.stages)

    control = tmp_path / "control"
    control.mkdir(exist_ok=True)
    stale = control / "FLEET_STOP.old-attempt"
    current = control / "FLEET_STOP.current-attempt"
    manual = control / "FLEET_STOP"
    stale.write_text("stale", encoding="utf-8")
    current.write_text("current failure", encoding="utf-8")
    manual.write_text('{"worker_index": 0, "error": "old"}', encoding="utf-8")
    clear_stale_stop_markers(tmp_path, attempt_id="current-attempt")
    assert not stale.exists() and current.exists() and not manual.exists()
    current.unlink()
    manual.write_text("launcher fail-fast", encoding="utf-8")
    clear_stale_stop_markers(tmp_path)
    assert manual.exists()

    config = FleetRunConfig(
        root=tmp_path,
        arm="arm",
        attempt_id="attempt-1",
        qids=("q0",),
        placement=FleetPlacement(workers=1, producers=0, receivers=1),
        worker_index=0,
    )
    publish_worker_failure(config, RuntimeError("construction failed"))
    marker = control / "FLEET_STOP.attempt-1"
    assert '"attempt_id": "attempt-1"' in marker.read_text(encoding="utf-8")


class _HungEngine:
    """Step forever without ever finishing the live request."""

    def add_request(self, request_id: str, prompt: Any, sampling: Any) -> None:
        del request_id, prompt, sampling

    def step(self) -> list[Any]:
        return []

    def has_unfinished_requests(self) -> bool:
        return True


class _Ticks:
    """One fake clock standing in for both the monotonic and wall clocks."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def __call__(self) -> float:
        return self.now


def test_active_async_receiver_is_not_throttled_and_publishes_seat_liveness(
    tmp_path: Path,
    monkeypatch,
):
    """Two properties of the same loop: it never sleeps, and it stays legible.

    A busy receiver must step as fast as the engine allows and still leave a
    record, because a seat wedged inside ``engine.step()`` is otherwise silent.
    """
    clock = _Ticks()
    path = seat_progress_path(tmp_path, 3)
    progress = SeatProgress(
        path,
        arm="arm",
        worker_index=3,
        role="receiver",
        attempt_id="a1",
        clock=clock,
        wall=clock,
    )

    def read() -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    progress.record("engine_open")
    assert set(read()) == _PROGRESS_FIELDS
    assert read()["stage"] == "engine_open" and read()["steps"] == 0

    progress.tick(7)
    assert read()["steps"] == 0, "a tick inside the interval must not write"
    clock.advance(PROGRESS_TICK_SECONDS)
    progress.tick(7)
    assert read()["steps"] == 7 and read()["stage"] == "engine_open"

    # A stage transition always writes, whatever the throttle says, and the
    # deadline table bounds only the stages a seat can hang inside.
    progress.record("admitted", qid="q0")
    assert read()["stage"] == "admitted" and read()["qid"] == "q0"
    # The deadline table is a deny-list, not an allow-list: the stage a
    # receiver occupies for its whole decode is `payload_loaded`, so it takes
    # the default deadline rather than falling through to none.
    assert set(SEAT_STAGE_DEADLINES) == {"engine_open", "producer_start", "receiver_claim"}
    assert set(SEAT_DEADLINE_EXEMPT) == {"idle", "worker_ready", "closed", "failed"}
    assert seat_stage_deadline("payload_loaded") == SEAT_DEADLINE_DEFAULT > 0
    for stage in ("admitted", "first_token", "decode_end", "handoff_ready", "producer_end"):
        assert seat_stage_deadline(stage) == SEAT_DEADLINE_DEFAULT
    for stage in SEAT_DEADLINE_EXEMPT:
        assert seat_stage_deadline(stage) == 0.0
    assert seat_stage_deadline("receiver_claim") == 300.0

    install_seat_progress(progress)
    try:
        decoder = StreamDecoder(_HungEngine(), clock=clock, sleep=lambda _seconds: None)
        decoder.submit(StreamRequest("r0", "prompt", None, 4, "q0", "s0"))
        for _ in range(3):
            clock.advance(PROGRESS_TICK_SECONDS)
            assert decoder.poll() == []
    finally:
        install_seat_progress(None)
    assert read()["steps"] == 3 and read()["stage"] == "admitted"
    assert sorted(item.name for item in path.parent.iterdir()) == ["gpu3.json"]

    sleeps: list[float] = []
    monkeypatch.setattr(fleet_runtime.time, "sleep", sleeps.append)
    bank = _SharedBank()
    receiver = _MultiPumpReceiver(seat_progress_path(tmp_path / "receiver", 0))
    hooks = FleetWorkerHooks(
        receiver,
        bank.completed,
        bank.bank_stage,
        bank.bank_result,
    )
    run_worker(
        FleetRunConfig(
            root=tmp_path / "receiver",
            arm="arm",
            attempt_id="a1",
            qids=("q0",),
            placement=FleetPlacement(workers=1, producers=0, receivers=1),
            worker_index=0,
            max_wall_s=10,
            barrier_timeout_s=5,
            claim_lease_s=2,
            poll_s=0.05,
        ),
        hooks,
    )
    assert sleeps == []
    assert set(bank.results) == {"q0"}
    # The stage the seat actually occupies while its engine is stepping, and
    # therefore the stage a wedged decode would be found in. It must carry a
    # deadline, or the watchdog is blind to exactly the hang it is for.
    assert receiver.stage_during_decode == "payload_loaded"
    assert seat_stage_deadline(receiver.stage_during_decode) > 0
    assert receiver.stage_during_close == "engine_close"
    assert seat_stage_deadline(receiver.stage_during_close) > 0


# --- The vLLM stream handle.


class _Params:
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)


class _OutputKind:
    FINAL_ONLY = "final-only"
    CUMULATIVE = "cumulative"


class _Core:
    def __init__(self) -> None:
        self.live: list[str] = []
        self.abort_calls: list[tuple[list[str], bool]] = []
        self.private_add_calls: list[str] = []
        self.output_processor = SimpleNamespace(request_states={})

    def add_request(self, request_id: str, _prompt: Any, _params: Any) -> None:
        self.private_add_calls.append(request_id)
        self.live.append(request_id)

    def step(self) -> list[Any]:
        request_id = self.live.pop(0)
        output = SimpleNamespace(token_ids=[7], text="", finish_reason="stop")
        return [
            SimpleNamespace(
                request_id=request_id,
                outputs=[output],
                finished=True,
                num_cached_tokens=0,
                metrics=None,
            )
        ]

    def has_unfinished_requests(self) -> bool:
        return bool(self.live)

    def abort_request(self, request_ids: list[str], *, internal: bool = False) -> None:
        self.abort_calls.append((list(request_ids), internal))
        removed = set(request_ids)
        if internal:
            removed = {
                self.output_processor.request_states[request_id].external_req_id
                for request_id in request_ids
            }
        self.live = [request_id for request_id in self.live if request_id not in removed]


class _LLM:
    def __init__(self) -> None:
        self.llm_engine = _Core()
        self.next_id = 0
        self.enqueue_calls: list[tuple[Any, Any, bool]] = []

    def enqueue(
        self,
        prompts: Any,
        sampling_params: Any,
        use_tqdm: bool = True,
    ) -> list[str]:
        self.enqueue_calls.append((prompts, sampling_params, use_tqdm))
        output_id = str(self.next_id)
        request_id = f"{output_id}-deadbeef"
        self.next_id += 1
        self.llm_engine.output_processor.request_states[request_id] = SimpleNamespace(
            external_req_id=output_id,
            output_kind=_OutputKind.FINAL_ONLY,
        )
        self.llm_engine.live.append(output_id)
        return [request_id]


@pytest.fixture(autouse=True)
def fake_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=_Params))
    monkeypatch.setitem(
        sys.modules,
        "vllm.sampling_params",
        SimpleNamespace(RequestOutputKind=_OutputKind),
    )


def test_public_enqueue_keeps_logical_request_ids_and_translates_sampling() -> None:
    llm = _LLM()
    handle = VllmEngineHandle(
        SimpleNamespace(_llm=llm),
        require_public_enqueue=True,
        detokenize=False,
    )
    params = handle.sampling(
        Sampling(
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            max_tokens=24_000,
            stop_token_ids=(1, 50, 106),
            bad_words=("<audio|>", "<image|>"),
        ),
        123,
    )
    assert params.detokenize is False
    assert params.stop_token_ids == [1, 50, 106]
    assert params.bad_words == ["<audio|>", "<image|>"]
    handle.add_request("logical-s0", {"prompt_embeds": "tensor"}, params)
    assert not llm.llm_engine.private_add_calls
    assert llm.enqueue_calls[0][2] is False
    assert llm.enqueue_calls[0][0] == [{"prompt_embeds": "tensor"}]
    outputs = handle.step()
    assert [output.request_id for output in outputs] == ["logical-s0"]
    assert handle.has_unfinished_requests() is False

    # A engine that already served generate() calls keeps counting from there,
    # and the external output id still maps back to the caller's logical id.
    later = _LLM()
    later.next_id = 6
    later_handle = VllmEngineHandle(SimpleNamespace(_llm=later), require_public_enqueue=True)
    later_handle.add_request("qid|floor:s0", {"prompt_embeds": "tensor"}, _Params())
    state = later.llm_engine.output_processor.request_states["6-deadbeef"]
    assert state.output_kind == _OutputKind.CUMULATIVE
    assert [output.request_id for output in later_handle.step()] == ["qid|floor:s0"]


def test_qwen_private_path_does_not_import_public_enqueue_output_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "vllm.sampling_params")
    llm = _LLM()
    handle = VllmEngineHandle(SimpleNamespace(_llm=llm))
    handle.add_request("logical-s0", {"prompt_token_ids": [1]}, _Params())
    assert llm.llm_engine.private_add_calls == ["logical-s0"]


def test_qwen_sampling_does_not_require_optional_bad_words() -> None:
    handle = VllmEngineHandle(SimpleNamespace(_llm=_LLM()))
    params = handle.sampling(
        SimpleNamespace(**QwenDecodeSpec("answer", QWEN_FAMILY).backend_sampling()),
        123,
    )

    assert params.seed == 123
    assert not hasattr(params, "bad_words")
    # A family with top-k off (Llama 3.1 Nemotron) renders no top_k field at
    # all; the handle reads that as off instead of failing on the attribute.
    off = QwenDecodeSpec("answer", NEMOTRON_FAMILY).backend_sampling()
    assert "top_k" not in off
    for sampling in (SimpleNamespace(**off), SimpleNamespace(top_k=None, **off)):
        nemotron_params = handle.sampling(sampling, 7)
        assert not hasattr(nemotron_params, "top_k")
        assert nemotron_params.stop_token_ids == list(NEMOTRON_FAMILY.stop_token_ids)
    # Every route family, built the way the receiver seat builds it in
    # `rcc.run.qwen.worker._offer`. A top-k-on family already carries the
    # field, so the seat adds nothing: a second one is refused by the backend.
    for family in (QWEN_FAMILY, NEMOTRON_FAMILY):
        seat = SimpleNamespace(**QwenDecodeSpec("answer", family).backend_sampling())
        seat_params = handle.sampling(seat, 7)
        assert seat_params.stop_token_ids == list(family.stop_token_ids)
        assert hasattr(seat_params, "top_k") == (family is QWEN_FAMILY)


def test_public_abort_maps_logical_ids_and_passes_the_internal_flag() -> None:
    llm = _LLM()
    handle = VllmEngineHandle(
        SimpleNamespace(_llm=llm),
        require_public_enqueue=True,
    )
    handle.add_request("logical-s0", {"prompt_embeds": "tensor"}, _Params())
    handle.abort_request(["logical-s0"])
    assert llm.llm_engine.abort_calls == [(["0-deadbeef"], True)]
    assert handle.has_unfinished_requests() is False


def test_public_enqueue_signature_drift_is_refused() -> None:
    class Drifted:
        def __init__(self) -> None:
            self.llm_engine = _Core()

        def enqueue(self, prompt: Any) -> list[str]:
            del prompt
            return ["bad"]

    with pytest.raises(RuntimeError, match="signature drifted"):
        VllmEngineHandle(
            SimpleNamespace(_llm=Drifted()),
            require_public_enqueue=True,
        )


# --- Crash-state durability of the banks.


def _record_gpu(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
                "torch_threads": torch.get_num_threads(),
                "vllm_cache": os.environ["VLLM_CACHE_ROOT"],
                "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            }
        ),
        encoding="utf-8",
    )


def _fail_seat() -> None:
    raise RuntimeError("seat failed")


def _block_seat() -> None:
    while True:
        time.sleep(1)  # fast-ok: child blocks until the failing peer terminates it


class _LiveSeat:
    def __init__(self) -> None:
        self.alive = True
        self.terminated = False

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False

    def join(self, timeout: float) -> None:
        del timeout

    def kill(self) -> None:
        self.alive = False


def test_split_seats_are_spawned_on_distinct_gpus_and_fail_as_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RCC_LOCAL_RUN_DIR", str(tmp_path))
    # Measure the family default independently of inherited allocator configuration.
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    # The Qwen and Nemotron lanes get expandable allocator segments; a Gemma
    # or Ministral seat, or one that names no family, does not, because vLLM
    # refuses their capture connectors under that setting.
    monkeypatch.setenv("RCC_FANOUT_FAMILY", "qwen")
    paths = [tmp_path / f"gpu{index}" for index in range(2)]
    run_in_processes([partial(_record_gpu, path) for path in paths], join_timeout_s=60)
    assert [json.loads(path.read_text(encoding="utf-8")) for path in paths] == [
        {
            "gpu": "0",
            "torch_threads": 4,
            "alloc_conf": "expandable_segments:True",
            "vllm_cache": str(tmp_path / "vllm_cache_fleet_gpu0"),
        },
        {
            "gpu": "1",
            "torch_threads": 4,
            "alloc_conf": "expandable_segments:True",
            "vllm_cache": str(tmp_path / "vllm_cache_fleet_gpu1"),
        },
    ]
    monkeypatch.setenv("RCC_FANOUT_FAMILY", "gemma")
    run_in_processes([partial(_record_gpu, tmp_path / "gemma0")], join_timeout_s=60)
    assert json.loads((tmp_path / "gemma0").read_text(encoding="utf-8"))["alloc_conf"] is None
    with pytest.raises(RuntimeError, match="split-fleet seats failed"):
        run_in_processes([partial(_record_gpu, tmp_path / "ok"), _fail_seat], join_timeout_s=60)
    with pytest.raises(RuntimeError, match="split-fleet seats failed"):
        run_in_processes([_block_seat, _fail_seat], join_timeout_s=30)


def test_driver_signal_stops_seats_before_redelivery(monkeypatch: pytest.MonkeyPatch) -> None:
    seat = _LiveSeat()
    redelivered: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(os, "kill", lambda pid, number: redelivered.append((pid, number)))
    with stop_seats_on_signal([seat]):
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(signal.SIGTERM, None)
    assert seat.terminated
    assert redelivered == [(os.getpid(), signal.SIGTERM)]


def _ticket(tmp_path: Path, qid: str, order: int) -> Ticket:
    payload = tmp_path / "payloads" / f"{qid}.bin"
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_bytes(qid.encode())
    return Ticket(
        qid=qid,
        arm="fleet-arm",
        order_index=order,
        payload_path=str(payload),
        payload_sha256=sha256_file(payload),
        producer_gpu=0,
        produced_at_unix=1.0,
    )


@pytest.mark.parametrize("cut_kind", ("mid-json", "mid-line", "after-newline"))
def test_runlog_torn_tail_repair_keeps_complete_row_bytes(tmp_path: Path, cut_kind: str):
    """`DurableRunLog` discards the final incomplete JSONL row and nothing else."""
    path = tmp_path / "runlog.jsonl"
    log = DurableRunLog(path, result_field="arm", fixed_fields={"run_id": "run-a"})
    for qid in ("q1", "q2", "q3", "q4"):
        log.bank({"kind": "result", "qid": qid, "arm": "arm"}, attempt_id="a1")
    complete = path.read_bytes()
    lines = complete.splitlines(keepends=True)
    boundary = sum(len(line) for line in lines[:3])
    fourth = lines[3]
    cuts = {
        "mid-json": boundary + fourth.index(b'"qid"') + 1,
        "mid-line": boundary + len(fourth) // 2,
        "after-newline": boundary,
    }
    cut = cuts[cut_kind]
    path.write_bytes(complete[:cut])

    repaired = DurableRunLog(path, result_field="arm", fixed_fields={"run_id": "run-a"})

    assert path.read_bytes() == complete[:boundary]
    assert repaired.repaired_tail_bytes == (cut - boundary if cut > boundary else 0)
    assert set(repaired.results) == {(qid, "arm") for qid in ("q1", "q2", "q3")}


def test_fanout_torn_tail_repair_preserves_the_prior_row_bytes(tmp_path: Path):
    """`banks.repair_torn_tail` truncates a torn worker tail at the last newline."""
    path = tmp_path / "arm" / "workers" / "gpu0" / "raw.jsonl"
    path.parent.mkdir(parents=True)
    lines = [
        (json.dumps({"kind": "result", "qid": qid}, sort_keys=True) + "\n").encode()
        for qid in ("q1", "q2")
    ]
    path.write_bytes(lines[0] + lines[1][: len(lines[1]) // 2])

    banks.repair_torn_tail(path)

    assert path.read_bytes() == lines[0]


def test_atomic_bank_rewrite_ignores_crash_residue_beside_target(tmp_path: Path):
    """Readers inspect published bank paths only.

    `banks.atomic_rows` replaces the target despite a stale temp file beside it.
    """
    arm_root = tmp_path / "arm"
    target = arm_root / "workers" / "gpu0" / "raw.jsonl"
    first = {"kind": "result", "arm": "arm", "qid": "q1"}
    second = {"kind": "result", "arm": "arm", "qid": "q2"}
    target.parent.mkdir(parents=True)
    banks.atomic_rows(target, [first])
    residue = target.with_name(f".{target.name}.dead.repair")
    residue.write_text('{"kind":"result","arm":"arm","qid":"corrupt"}')

    assert banks.read_bank_rows(target) == [first]
    banks.atomic_rows(target, [first, second])

    assert banks.read_bank_rows(target) == [first, second]
    assert residue.is_file()


def test_stale_claim_reset_requeues_only_unbanked_claims(tmp_path: Path):
    """`reset_stale_claims` drops claims whose qids are banked and requeues the rest."""
    root = tmp_path / "queue"
    publish_ticket(root, _ticket(tmp_path, "done", 0))
    publish_ticket(root, _ticket(tmp_path, "lost", 1))
    first = claim_next(root, owner="dead-owner")
    second = claim_next(root, owner="dead-owner")
    assert first is not None and first[0].qid == "done"
    assert second is not None and second[0].qid == "lost"

    assert reset_stale_claims(root, banked_qids={"done"}) == ["lost"]
    ready, claimed, _ = queue_dirs(root)
    assert len(list(ready.iterdir())) == 1
    assert not list(claimed.joinpath("dead-owner").iterdir())
    recovered = claim_next(root, owner="replacement")
    assert recovered is not None and recovered[0].qid == "lost"
    assert claim_next(root, owner="replacement") is None


def test_verified_payload_read_refuses_a_flipped_byte(tmp_path: Path):
    """`verify_ticket_payload` refuses payload bytes that differ from the published digest."""
    ticket = _ticket(tmp_path, "item-a", 0)
    verify_ticket_payload(ticket)
    payload = Path(ticket.payload_path)
    corrupted = bytearray(payload.read_bytes())
    corrupted[0] ^= 1
    payload.write_bytes(corrupted)

    with pytest.raises(ValueError, match="torn handoff"):
        verify_ticket_payload(ticket)
