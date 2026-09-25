"""The arm-sequential driver for one split-fleet run.

One node runs one arm at a time with every role live together, as one process per
GPU on a node or as threads elsewhere. Arm open and close are node events.
"""

from __future__ import annotations

import fcntl
import multiprocessing
import os
import signal
import threading
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any

from rcc.hardware.fleet import FleetPlacement
from rcc.run import io
from rcc.run.barrier import (
    ARM_PHASES,
    STATES,
    BarrierRun,
    BarrierSpec,
)
from rcc.run.contract import FleetPipelineAdapter
from rcc.run.fleet.latency import WARMUP_BODY, WARMUP_MAX_TOKENS
from rcc.run.fleet.runtime import clear_stale_stop_markers, publish_fleet_stop
from rcc.run.runner import (
    RunnerConfig,
    freeze,
    run_arm,
    write_identity,
)

#: The warmup every family draws before any measured item, published once per
#: run so two launches of one run cannot be warmed differently.
WARMUP_CONTRACT_SCHEMA = "rcc-split-fleet-warmup-contract-v1"

#: The summary one driver invocation returns.
SPLIT_RUN_SCHEMA = "rcc-split-run-v1"

#: Seconds a killed seat is given to leave the process table before the driver
#: stops waiting on it.
_KILL_GRACE_S = 30.0

_LOCK_NAME = ".arm_driver.lock"
_WARMUP_NAME = "warmup_contract.json"

#: The signals a stop delivers to the driver, which the driver converts into a
#: takedown of its own seats.
_STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)

#: Every seat is spawned, never forked: a forked child of a process that has
#: touched the CUDA driver API inherits an unusable context.
_SEAT_START_METHOD = "spawn"
_LOCAL_RUN_ROOTS = ("RCC_LOCAL_RUN_DIR", "RCC_GEMMA_M3_ROOT", "RCC_MINISTRAL_M3_ROOT")

_ATTEMPT_ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


@dataclass(frozen=True)
class SplitScheduleConfig:
    """One driver invocation over one run root, panel, and arm roster.

    ``attempt_id`` names one launch and is never reused: a second launch under a
    spent id would republish the first launch's activity windows as its own.
    """

    root: Path
    panel: str
    source_commit: str
    attempt_id: str
    barrier: BarrierSpec
    plan_fingerprint: str
    roster_fingerprint: str
    #: The arms to run, in order. Empty means the adapter's whole roster.
    arms: tuple[str, ...] = ()
    max_wall_s: float = 14_400.0
    barrier_timeout_s: float = 3_600.0
    claim_lease_s: float = 7_200.0
    poll_s: float = 0.05
    join_timeout_s: float = 21_600.0

    def __post_init__(self) -> None:
        """Refuse a run with no identity or a duplicated arm."""
        if not self.attempt_id or not self.panel or not self.source_commit:
            raise ValueError("a split-fleet run needs a panel, a source commit, and an attempt")
        # The attempt id names a marker directory, so it has to be a plain
        # filename component.
        if set(self.attempt_id) - _ATTEMPT_ID_CHARS:
            raise ValueError("a split-fleet attempt id must be a plain identifier")
        if len(set(self.arms)) != len(self.arms):
            raise ValueError("a split-fleet arm roster must be unique")


def warmup_contract() -> dict[str, Any]:
    """Return the one warmup every family draws before its first measurement."""
    return {
        "schema": WARMUP_CONTRACT_SCHEMA,
        "body": WARMUP_BODY,
        "max_tokens": WARMUP_MAX_TOKENS,
    }


def publish_warmup_contract(root: Path) -> dict[str, Any]:
    """Publish this run's warmup contract, or refuse a drifted republication."""
    contract = warmup_contract()
    encoded = io.canonical_bytes(contract) + b"\n"
    path = Path(root) / "control" / _WARMUP_NAME
    if path.is_file():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"{path}: this run was warmed under another warmup contract")
        return contract
    io.atomic_bytes(path, encoded)
    return contract


@contextmanager
def driver_lock(root: Path) -> Generator[None]:
    """Hold the exclusive right to drive arms over one run root."""
    path = Path(root) / "control" / _LOCK_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"{root}: another split-fleet driver is already running") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def role_phase(placement: FleetPlacement, worker_index: int) -> str:
    """Return the arm phase one seat publishes its activity under.

    A fused seat runs one loop and publishes one activity window, the one the
    barrier reads as the arm's receiver side.
    """
    return ARM_PHASES[1] if placement.role(worker_index) == "producer" else ARM_PHASES[2]


def _mark(run: BarrierRun, *, arm: str, phase: str, state: str, worker_index: int) -> None:
    run.publish_marker(
        run.arm_phase_root(arm, phase) / state / f"gpu{worker_index}.json",
        run.arm_marker_identity(worker_index=worker_index, arm=arm, phase=phase, state=state),
    )


def _mark_fleet(run: BarrierRun, *, arm: str, phase: str, workers: int) -> None:
    """Publish one whole-node phase, entered and finished, for every seat."""
    for state in STATES:
        for worker_index in range(workers):
            _mark(run, arm=arm, phase=phase, state=state, worker_index=worker_index)


@contextmanager
def arm_activity(
    run: BarrierRun,
    *,
    arm: str,
    placement: FleetPlacement,
    worker_index: int,
) -> Generator[None]:
    """Bracket one seat's live work with its role activity markers."""
    phase = role_phase(placement, worker_index)
    _mark(run, arm=arm, phase=phase, state=STATES[0], worker_index=worker_index)
    yield
    _mark(run, arm=arm, phase=phase, state=STATES[1], worker_index=worker_index)


def run_in_threads(bodies: Sequence[Callable[[], None]], *, join_timeout_s: float) -> None:
    """Run one arm's seats concurrently as threads and raise the first failure.

    The thread sibling of :func:`run_in_processes`, for a test or a single-seat
    arm. The driver takes the runner as a seam, so either one binds.
    """
    failures: list[BaseException] = []
    lock = threading.Lock()

    def guarded(body: Callable[[], None]) -> None:
        try:
            body()
        except BaseException as error:
            with lock:
                failures.append(error)

    threads = [threading.Thread(target=guarded, args=(body,)) for body in bodies]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=join_timeout_s)
    alive = [thread for thread in threads if thread.is_alive()]
    if alive:
        raise TimeoutError(f"{len(alive)} split-fleet workers did not finish in {join_timeout_s}s")
    if failures:
        raise failures[0]


def _seat_process(body: Callable[[], None], worker_index: int) -> None:
    """Pin this child to its own GPU and a small CPU pool, then run its seat."""
    # Expandable segments stop the allocator holding fragments, but the Gemma
    # and Ministral connectors are refused under them.
    if os.environ.get("RCC_FANOUT_FAMILY") in ("qwen", "nemotron"):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if os.environ.get("RCC_FANOUT_FAMILY") == "nemotron":
        # The Nemotron engines run under an Inductor policy that torch reads at
        # import, so the seat sets it before its first torch import.
        from rcc.models.nemotron.runtime_policy import ENVIRONMENT

        for key, value in ENVIRONMENT.items():
            os.environ.setdefault(key, value)
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_index)
    cache_base = os.environ.get("VLLM_CACHE_ROOT")
    if cache_base is None:
        local_root = next(
            filter(None, map(os.environ.get, _LOCAL_RUN_ROOTS)),
            None,
        )
        if local_root is not None:
            cache_base = str(Path(local_root) / "vllm_cache_fleet")
    if cache_base is not None:
        os.environ["VLLM_CACHE_ROOT"] = f"{cache_base}_gpu{worker_index}"
    # The receiver's remaining CPU work is prompt assembly, which saturates
    # near four threads.
    torch.set_num_threads(4)
    body()


def terminate_seats(seats: Sequence[Any]) -> None:
    """Take every live seat out of the process table, politely then not."""
    for seat in seats:
        if seat.is_alive():
            seat.terminate()
    for seat in seats:
        seat.join(timeout=_KILL_GRACE_S)
    for seat in seats:
        if seat.is_alive():
            seat.kill()
            seat.join(timeout=_KILL_GRACE_S)


@contextmanager
def stop_seats_on_signal(seats: Sequence[Any]) -> Generator[None]:
    """Take the seats down with the driver when the node signals it.

    A stop signals the driver's pid alone, so without this its non-daemon children
    would keep appending. Handlers install only from the main thread.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous: dict[int, Any] = {}

    def handle(number: int, frame: Any) -> None:
        del frame
        terminate_seats(seats)
        signal.signal(number, previous[number])
        os.kill(os.getpid(), number)

    try:
        for number in _STOP_SIGNALS:
            previous[int(number)] = signal.signal(number, handle)
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _wait_for_seats(seats: Sequence[Any], *, deadline: float) -> None:
    """Join finished seats and stop peers as soon as one seat fails."""
    pending = set(seats)
    while pending and time.monotonic() < deadline:
        ready = set(
            wait(
                [seat.sentinel for seat in pending],
                timeout=max(0.0, deadline - time.monotonic()),
            )
        )
        if not ready:
            break
        for seat in tuple(pending):
            if seat.sentinel in ready:
                seat.join()
                pending.remove(seat)
        if any(seat.exitcode not in (None, 0) for seat in seats):
            terminate_seats(tuple(pending))
            break


def run_in_processes(bodies: Sequence[Callable[[], None]], *, join_timeout_s: float) -> None:
    """Run one arm's seats as one OS process per GPU seat, and raise on failure.

    Seats are spawned, not forked: a forked child of a CUDA-initialized parent
    dies on its first device call. A seat's index in ``bodies`` is its device.
    """
    context = multiprocessing.get_context(_SEAT_START_METHOD)
    seats = [
        context.Process(target=_seat_process, args=(body, index), daemon=False)
        for index, body in enumerate(bodies)
    ]
    started: list[Any] = []
    with stop_seats_on_signal(started):
        try:
            for seat in seats:
                seat.start()
                started.append(seat)
            deadline = time.monotonic() + join_timeout_s
            _wait_for_seats(started, deadline=deadline)
        except BaseException:
            terminate_seats(started)
            raise
        alive = [index for index, seat in enumerate(seats) if seat.is_alive()]
        for index in alive:
            seats[index].kill()
        for seat in seats:
            seat.join(timeout=_KILL_GRACE_S)
    if alive:
        raise TimeoutError(f"split-fleet seats {alive} did not finish in {join_timeout_s}s")
    failed = {index: seat.exitcode for index, seat in enumerate(seats) if seat.exitcode != 0}
    if failed:
        detail = ", ".join(f"gpu{index} exit {status}" for index, status in sorted(failed.items()))
        raise RuntimeError(f"split-fleet seats failed: {detail}")


WorkerRunner = Callable[[Sequence[Callable[[], None]], float], None]


def _barrier_run(config: SplitScheduleConfig, node_gpus: int) -> BarrierRun:
    return BarrierRun(
        spec=config.barrier,
        run_root=config.root,
        attempt_id=config.attempt_id,
        plan_fingerprint=config.plan_fingerprint,
        roster_fingerprint=config.roster_fingerprint,
        node_gpus=node_gpus,
    )


def refuse_spent_attempt(run: BarrierRun, root: Path) -> None:
    """Refuse a launch whose attempt id already left barrier markers here.

    A reused id would republish another launch's activity windows as this
    launch's own, so the merge could not tell the two apart.
    """
    spent = run.attempt_root.is_dir() and any(run.attempt_root.iterdir())
    if spent:
        raise RuntimeError(
            f"{run.attempt_id}: attempt id was already used; relaunch with a fresh attempt id"
        )


@dataclass(frozen=True)
class SeatBody:
    """One seat's whole duty, as a value a spawned child can be handed.

    Seats are spawned rather than forked, so everything the child needs crosses
    by pickle; a nested function would not survive, so this is data.
    """

    adapter: FleetPipelineAdapter
    config: SplitScheduleConfig
    run: BarrierRun
    placements: Mapping[str, FleetPlacement]
    arm: str
    worker_index: int

    def runner(self) -> RunnerConfig:
        """Return this seat's runner invocation over the shared schedule."""
        return RunnerConfig(
            root=self.config.root,
            panel=self.config.panel,
            source_commit=self.config.source_commit,
            worker_index=self.worker_index,
            arm=self.arm,
            attempt_id=self.config.attempt_id,
            write_identity=False,
            max_wall_s=self.config.max_wall_s,
            barrier_timeout_s=self.config.barrier_timeout_s,
            claim_lease_s=self.config.claim_lease_s,
            poll_s=self.config.poll_s,
        )

    def __call__(self) -> None:
        """Bracket this seat's arm activity and run its one arm."""
        placement = self.placements[self.arm]
        try:
            with arm_activity(
                self.run,
                arm=self.arm,
                placement=placement,
                worker_index=self.worker_index,
            ):
                run_arm(self.adapter, self.runner(), self.placements)
        except BaseException as error:
            publish_fleet_stop(
                self.config.root / "arms" / self.arm,
                attempt_id=self.config.attempt_id,
                worker_index=self.worker_index,
                role=placement.role(self.worker_index),
                error=error,
            )
            raise


def run_split_fleet(
    adapter: FleetPipelineAdapter,
    config: SplitScheduleConfig,
    *,
    run_workers: WorkerRunner | None = None,
    preflight: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    """Drive one attempt's arms in order and return the run summary.

    ``preflight`` is the family's one run-level publication, called once under the
    driver lock before the first arm opens, for receivers reading run-level bytes.
    """
    launch: WorkerRunner = run_workers or (
        lambda bodies, timeout: run_in_threads(bodies, join_timeout_s=timeout)
    )
    resolved = adapter.placements.resolve(
        config.root,
        panel=config.panel,
        source_commit=config.source_commit,
    )
    roster = config.arms or tuple(adapter.placements.arm_names)
    unknown = [arm for arm in roster if arm not in resolved]
    if unknown:
        raise ValueError(f"unregistered split-fleet arm {unknown[0]!r}")
    ordered = {arm: resolved[arm] for arm in roster}
    widths = {placement.workers for placement in ordered.values()}
    if len(widths) != 1:
        raise RuntimeError("one split-fleet run cannot mix node widths across its arms")
    run = _barrier_run(config, widths.pop())
    with driver_lock(config.root):
        refuse_spent_attempt(run, config.root)
        write_identity(
            adapter,
            config.root,
            panel=config.panel,
            source_commit=config.source_commit,
        )
        freeze(adapter, config.root, panel=config.panel, source_commit=config.source_commit)
        contract = publish_warmup_contract(config.root)
        if preflight is not None:
            preflight(config.root)
        for arm, placement in ordered.items():
            clear_stale_stop_markers(
                config.root / "arms" / arm,
                attempt_id=config.attempt_id,
            )
            _mark_fleet(run, arm=arm, phase=ARM_PHASES[0], workers=placement.workers)
            launch(
                [
                    SeatBody(
                        adapter=adapter,
                        config=config,
                        run=run,
                        placements=ordered,
                        arm=arm,
                        worker_index=worker_index,
                    )
                    for worker_index in range(placement.workers)
                ],
                config.join_timeout_s,
            )
            _mark_fleet(run, arm=arm, phase=ARM_PHASES[3], workers=placement.workers)
        return {
            "schema": SPLIT_RUN_SCHEMA,
            "panel": config.panel,
            "attempt_id": config.attempt_id,
            "source_commit": config.source_commit,
            "arms": list(ordered),
            "node_gpus": run.node_gpus,
            "warmup_contract": dict(contract),
        }


__all__ = (
    "SPLIT_RUN_SCHEMA",
    "WARMUP_CONTRACT_SCHEMA",
    "SeatBody",
    "SplitScheduleConfig",
    "arm_activity",
    "driver_lock",
    "publish_warmup_contract",
    "refuse_spent_attempt",
    "role_phase",
    "run_in_processes",
    "run_in_threads",
    "run_split_fleet",
    "stop_seats_on_signal",
    "terminate_seats",
    "warmup_contract",
)
