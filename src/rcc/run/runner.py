"""Run one arm of a benchmark adapter on the fleet runtime.

The runner resolves the arm's placement, opens its worker bank, builds the
seat's worker, and hands it to the fleet lifecycle, once per seat.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, cast

from rcc.hardware.fleet import FleetPlacement
from rcc.run import io
from rcc.run.banks import (
    BankWriter,
    IncrementalBankScanner,
    read_bank_rows,
    repair_torn_tail,
)
from rcc.run.contract import (
    CompletionKey,
    FleetPipelineAdapter,
    ReportSource,
    ResultPolicy,
    WorkerContext,
    WorkerLockProvider,
)
from rcc.run.fleet.progress import install_seat_progress
from rcc.run.fleet.runtime import (
    FleetRunConfig,
    FleetWorkerHooks,
    begin_worker_progress,
    publish_fleet_stop,
    publish_worker_failure,
    run_worker,
)


@dataclass(frozen=True)
class RunnerConfig:
    """One worker invocation over one panel and arm."""

    root: Path
    panel: str
    source_commit: str
    worker_index: int
    arm: str
    attempt_id: str
    write_identity: bool = True
    max_wall_s: float = 14_400.0
    barrier_timeout_s: float = 3_600.0
    claim_lease_s: float = 7_200.0
    poll_s: float = 0.05


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(cast(Any, value)).items()}
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, Any], value)
        return {str(key): _jsonable(item) for key, item in mapping.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in cast(list[Any] | tuple[Any, ...], value)]
    return value


def write_identity(
    adapter: FleetPipelineAdapter,
    root: Path,
    *,
    panel: str,
    source_commit: str,
) -> Path:
    """Publish the benchmark registration and sampling identity."""
    payload = {
        "panel": panel,
        "source_commit": source_commit,
        "arms": list(adapter.placements.arm_names),
        "registration": _jsonable(adapter.registration()),
        "sampling_contract": _jsonable(adapter.sampling_contract()),
    }
    target = root / "control" / "pipeline_identity.json"
    io.atomic_json(target, payload)
    return target


def freeze(
    adapter: FleetPipelineAdapter,
    root: Path,
    *,
    panel: str,
    source_commit: str,
) -> Path:
    """Resolve the adapter's placements and publish them."""
    placements = adapter.placements.resolve(
        root,
        panel=panel,
        source_commit=source_commit,
    )
    return _publish_freeze(
        adapter,
        root,
        panel=panel,
        source_commit=source_commit,
        placements=placements,
    )


def _publish_freeze(
    adapter: FleetPipelineAdapter,
    root: Path,
    *,
    panel: str,
    source_commit: str,
    placements: Mapping[str, FleetPlacement],
) -> Path:
    """Publish placements the caller has already resolved."""
    payload = {
        "panel": panel,
        "source_commit": source_commit,
        "arms": list(adapter.placements.arm_names),
        "placements": _jsonable(placements),
        "registration": _jsonable(adapter.registration()),
        "sampling_contract": _jsonable(adapter.sampling_contract()),
    }
    target = root / "control" / "pipeline_freeze.json"
    io.atomic_json(target, payload)
    return target


def _validate_existing(path: Path, fields: Mapping[str, Any]) -> None:
    if not path.is_file():
        return
    repair_torn_tail(path)
    for row in read_bank_rows(path, repair_torn_tail=False):
        for name, expected in fields.items():
            if row.get(name) != expected:
                raise RuntimeError(f"{path}: bank identity mismatch for {name}")


def _completed_qids(
    paths: tuple[Path, ...],
    completion_key: Callable[[Mapping[str, Any]], CompletionKey | None],
) -> Callable[[], set[str]]:
    """Return a callable naming the items whose cells are already complete."""

    def is_completion(row: Mapping[str, Any]) -> bool:
        return completion_key(row) is not None

    scanner_key = cast(Callable[[Mapping[str, Any]], str], completion_key)
    scanner = IncrementalBankScanner(paths, row_key=scanner_key, row_predicate=is_completion)
    completed_qids: set[str] = set()

    def completed() -> set[str]:
        for row in scanner.scan():
            qid = row.get("qid")
            if not isinstance(qid, str) or not qid:
                raise RuntimeError("completion row lacks a nonempty qid")
            completed_qids.add(qid)
        duplicates = sorted(str(key) for key, count in scanner.state.counts.items() if count != 1)
        if duplicates:
            raise RuntimeError(f"duplicate banked completions for {duplicates[:3]}")
        return set(completed_qids)

    return completed


def _bank_lock(
    adapter: FleetPipelineAdapter,
    root: Path,
    *,
    arm: str,
    worker_index: int,
) -> AbstractContextManager[None]:
    """Return the adapter's bank lock, or no lock when it offers none."""
    provider = getattr(adapter, "bank_lock", None)
    if not callable(provider):
        return nullcontext()
    return cast(WorkerLockProvider, adapter).bank_lock(
        root,
        arm=arm,
        worker_index=worker_index,
    )


def run_arm(
    adapter: FleetPipelineAdapter,
    config: RunnerConfig,
    placements: Mapping[str, FleetPlacement],
) -> float:
    """Run one arm through the fleet runtime and return its batch clock."""
    arm = config.arm
    loaded, manifest = adapter.data.load_prepared(
        config.root,
        panel=config.panel,
        source_commit=config.source_commit,
    )
    qids = tuple(adapter.data.qids)
    items = tuple(adapter.data.item(qid) for qid in qids)
    if len(items) != len(loaded):
        raise RuntimeError(f"{arm}: prepared item count differs from the adapter qid panel")
    try:
        placement = placements[arm]
    except KeyError as exc:
        raise RuntimeError(f"{arm}: placement map does not cover the registered arm") from exc
    placement.role(config.worker_index)
    identity = adapter.bank_identity(
        config.root,
        panel=config.panel,
        arm=arm,
        source_commit=config.source_commit,
        prepared_manifest=manifest,
        placement=placement,
    )
    paths = adapter.bank_paths(
        config.root,
        panel=config.panel,
        arm=arm,
        placement=placement,
    )
    if len(paths) != placement.workers:
        raise RuntimeError(f"{arm}: adapter returned the wrong number of bank paths")
    try:
        path = paths[config.worker_index]
    except IndexError as exc:
        raise RuntimeError(f"{arm}: adapter bank paths omit the worker index") from exc
    role = placement.role(config.worker_index)
    try:
        with _bank_lock(
            adapter,
            config.root,
            arm=arm,
            worker_index=config.worker_index,
        ):
            _validate_existing(path, identity.fields)
            bank = BankWriter(path, fixed_fields=identity.fields, attempt_id=config.attempt_id)
            completed = _completed_qids(paths, adapter.completion_key)
            context = WorkerContext(
                root=config.root,
                panel=config.panel,
                arm=arm,
                worker_index=config.worker_index,
                attempt_id=config.attempt_id,
                source_commit=config.source_commit,
                qids=qids,
                items=items,
                prepared_manifest=manifest,
                placement=placement,
                identity=identity,
            )
            if identity.runtime_signature is not None:
                bank.append(
                    {
                        "kind": "runtime",
                        "runtime_signature": identity.runtime_signature,
                        "role": role,
                    }
                )
            fleet_config = FleetRunConfig(
                root=config.root / "arms" / arm,
                arm=arm,
                attempt_id=config.attempt_id,
                qids=qids,
                placement=placement,
                worker_index=config.worker_index,
                max_wall_s=config.max_wall_s,
                barrier_timeout_s=config.barrier_timeout_s,
                claim_lease_s=config.claim_lease_s,
                poll_s=config.poll_s,
            )
            progress = begin_worker_progress(fleet_config)
            try:
                worker = adapter.build_worker(context, bank.append)
            except BaseException as error:
                publish_worker_failure(fleet_config, error)
                progress.record("failed")
                install_seat_progress(None)
                raise
            hooks = FleetWorkerHooks(
                adapter=worker,
                completed_qids=completed,
                bank_stage=bank.append,
                bank_result=bank.append,
            )
            clock = run_worker(
                fleet_config,
                hooks,
                progress=progress,
            )
            bank.append({"kind": "worker_complete", "role": role})
            return clock
    except BaseException as error:
        publish_fleet_stop(
            config.root / "arms" / arm,
            attempt_id=config.attempt_id,
            worker_index=config.worker_index,
            role=role,
            error=error,
        )
        raise


def run(adapter: FleetPipelineAdapter, config: RunnerConfig) -> tuple[float, ...]:
    """Resolve one arm, publish the run identity, and run its worker."""
    if config.arm not in adapter.placements.arm_names:
        raise ValueError(f"unregistered benchmark arm {config.arm!r}")
    placements = adapter.placements.resolve(
        config.root,
        panel=config.panel,
        source_commit=config.source_commit,
    )
    if config.write_identity:
        write_identity(
            adapter,
            config.root,
            panel=config.panel,
            source_commit=config.source_commit,
        )
        _publish_freeze(
            adapter,
            config.root,
            panel=config.panel,
            source_commit=config.source_commit,
            placements=placements,
        )
    return (run_arm(adapter, config, placements),)


def build_report(
    adapter: FleetPipelineAdapter,
    root: Path,
    *,
    source_commit: str,
    results_policy: ResultPolicy | None = None,
    source: ReportSource | None = None,
) -> Mapping[str, Any]:
    """Build this adapter's offline report over one run root."""
    return adapter.reports.build_report(
        root,
        source_commit=source_commit,
        results_policy=results_policy,
        source=source,
    )
