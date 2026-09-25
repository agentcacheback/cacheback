"""Typed seams between a benchmark adapter and the fleet pipeline runner.

A benchmark supplies its data, identity fields, result validation, and report;
`rcc.run.fleet` runs the execution lifecycle around them.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeAlias, TypeVar

from rcc.hardware.fleet import FleetPlacement
from rcc.run.fleet.contract import (
    AsyncReceiver,
    FusedAdapter,
    SplitProducer,
)

JsonRow: TypeAlias = dict[str, Any]
PreparedItem: TypeAlias = Any
BankCallback: TypeAlias = Callable[[Mapping[str, Any]], JsonRow]
FleetWorker: TypeAlias = SplitProducer | AsyncReceiver | FusedAdapter
PlacementMap: TypeAlias = Mapping[str, FleetPlacement]
ResultKey: TypeAlias = tuple[str, ...]
CompletionKey: TypeAlias = Hashable
ReportSource: TypeAlias = Sequence[JsonRow] | Mapping[str, Sequence[JsonRow]]


@dataclass(frozen=True)
class BankIdentity:
    """The fixed bank fields and runtime signature of one worker attempt."""

    fields: Mapping[str, Any]
    runtime_signature: Mapping[str, Any] | None = None


def placement_identity(placement: FleetPlacement) -> dict[str, Any]:
    """Return the placement fields every split family binds its rows to."""
    return {
        "workers": placement.workers,
        "producers": placement.producers,
        "receivers": placement.receivers,
        "fused": placement.fused,
    }


@dataclass(frozen=True)
class WorkerContext:
    """The inputs an adapter is given to construct one live worker."""

    root: Path
    panel: str
    arm: str
    worker_index: int
    attempt_id: str
    source_commit: str
    qids: tuple[str, ...]
    items: tuple[PreparedItem, ...]
    prepared_manifest: Mapping[str, Any]
    placement: FleetPlacement
    identity: BankIdentity


class SequenceResultPolicy(Protocol):
    """Selects a complete attempt for each item in a row sequence."""

    def select(
        self,
        rows: Sequence[JsonRow],
        qid: str,
    ) -> tuple[str, list[JsonRow]]:
        """Return the one complete attempt kept for an item."""
        ...


class BankResultPolicy(Protocol):
    """Selects reportable rows from banks keyed by their job identity."""

    def select(
        self,
        banks: Mapping[str, Sequence[JsonRow]],
    ) -> tuple[dict[ResultKey, JsonRow], list[str], dict[ResultKey, set[str]]]:
        """Return reportable rows and the attempts that were rejected."""
        ...


ResultPolicy: TypeAlias = SequenceResultPolicy | BankResultPolicy


class BenchmarkDataAdapter(Protocol):
    """Exposes a fixed, ordered panel and its prepared item lookup."""

    @property
    def qids(self) -> tuple[str, ...]:
        """Return the current panel's qids, in panel order."""
        ...

    def load_prepared(
        self,
        root: Path,
        *,
        panel: str,
        source_commit: str,
    ) -> tuple[tuple[PreparedItem, ...], Mapping[str, Any]]:
        """Load and validate prepared bytes for one panel."""
        ...

    def item(self, qid: str) -> PreparedItem:
        """Return one already-loaded prepared item by qid."""
        ...


class BenchmarkPlacementAdapter(Protocol):
    """Resolves the placements of the current panel and run."""

    @property
    def arm_names(self) -> tuple[str, ...]:
        """Return the arm roster, in execution order."""
        ...

    def resolve(
        self,
        root: Path,
        *,
        panel: str,
        source_commit: str,
    ) -> PlacementMap:
        """Return the panel's placements, including any manifest validation."""
        ...


class WorkerLockProvider(Protocol):
    """Provides an adapter's own exclusive worker-bank lock."""

    def bank_lock(
        self,
        root: Path,
        *,
        arm: str,
        worker_index: int,
    ) -> AbstractContextManager[None]:
        """Return the exclusive lock for one worker's bank."""
        ...


ReportSourceT = TypeVar("ReportSourceT", contravariant=True)
ReportPolicyT = TypeVar("ReportPolicyT", contravariant=True)


class BenchmarkReportAdapter(Protocol[ReportSourceT, ReportPolicyT]):
    """Runs one benchmark's report path for its own source and policy."""

    def build_report(
        self,
        root: Path,
        *,
        source_commit: str,
        results_policy: ReportPolicyT | None = None,
        source: ReportSourceT | None = None,
    ) -> Mapping[str, Any]:
        """Build a report after one result-selection pass."""
        ...


class FleetPipelineAdapter(Protocol):
    """Composes everything benchmark-specific that the runner calls."""

    @property
    def data(self) -> BenchmarkDataAdapter:
        """Return the data adapter."""
        ...

    @property
    def placements(self) -> BenchmarkPlacementAdapter:
        """Return the placement adapter."""
        ...

    @property
    def reports(self) -> BenchmarkReportAdapter[Any, Any]:
        """Return the report adapter."""
        ...

    def registration(self) -> Mapping[str, Any]:
        """Return the benchmark, model, and execution identities."""
        ...

    def sampling_contract(self) -> Mapping[str, Any]:
        """Return the model-visible sampling contract."""
        ...

    def bank_paths(
        self,
        root: Path,
        *,
        panel: str,
        arm: str,
        placement: FleetPlacement,
    ) -> tuple[Path, ...]:
        """Return ordered worker-bank paths for the current panel and arm."""
        ...

    def completion_key(self, row: Mapping[str, Any]) -> CompletionKey | None:
        """Return what this row completes, or ``None`` for other rows."""
        ...

    def bank_identity(
        self,
        root: Path,
        *,
        panel: str,
        arm: str,
        source_commit: str,
        prepared_manifest: Mapping[str, Any],
        placement: FleetPlacement,
    ) -> BankIdentity:
        """Return the fixed row identity and runtime signature for one seat."""
        ...

    def build_worker(
        self,
        context: WorkerContext,
        bank: BankCallback,
    ) -> FleetWorker:
        """Construct one worker, which banks through the supplied callback."""
        ...
