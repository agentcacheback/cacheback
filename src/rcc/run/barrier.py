"""Attempt-scoped arm-phase markers, shared by every family.

The seats of one node are separate processes, so the arm phases rendezvous
through files namespaced by attempt, under a directory a barrier spec names.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from rcc.hardware.fleet import FleetPlacement
from rcc.run import io

STATES = ("entered", "finished")
#: The four per-arm phases of the split fleet, in order.
ARM_PHASES = ("arm_open", "producers_active", "receivers_active", "arm_complete")
#: The isolation this profile names: arms stay strictly sequential on the node,
#: while producer and receiver activity inside one arm overlaps.
SPLIT_FLEET_ISOLATION_PROFILE = "split-fleet-arm-sequential-role-split-v1"
_SPLIT_ROLE_LABELS = ("fleet", "producer", "receiver", "fleet")
#: A fused arm has no producer phase at all: one role, one activity window.
_FUSED_ROLE_LABELS = ("fleet", "producer", "fused", "fleet")
_SAFE_ARM_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


@dataclass(frozen=True)
class BarrierSpec:
    """The per-family identity of one fleet's arm barrier."""

    family_label: str
    marker_schema: str
    barrier_dirname: str


def read_marker(path: Path, spec: BarrierSpec) -> dict[str, Any]:
    """Read one phase marker and validate its timestamp."""
    label = spec.family_label
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(f"{path}: malformed {label} barrier marker") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: {label} barrier marker is not an object")
    value = cast(dict[str, Any], value)
    timestamp = value.get("utc_unix")
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
        or not math.isfinite(float(timestamp))
    ):
        raise RuntimeError(f"{path}: {label} barrier timestamp is invalid")
    return value


@dataclass(frozen=True)
class BarrierRun:
    """One attempt's marker namespace and the identity its markers carry."""

    spec: BarrierSpec
    run_root: Path
    attempt_id: str
    plan_fingerprint: str
    roster_fingerprint: str
    node_gpus: int

    @property
    def attempt_root(self) -> Path:
        """Return the attempt-scoped barrier root."""
        return Path(self.run_root) / "control" / self.spec.barrier_dirname / self.attempt_id

    @property
    def expected_names(self) -> set[str]:
        """Return the marker filenames this fleet's GPUs publish."""
        return {f"gpu{index}.json" for index in range(self.node_gpus)}

    def arm_phase_root(self, arm: str, phase: str) -> Path:
        """Return the directory holding one arm phase's marker states.

        The path is keyed by the arm name alone, so a reader that rebuilds the
        placement mapping in another order still reads the markers the run wrote.
        """
        return self.attempt_root / arm_dirname(arm) / phase

    def _identity(self, worker_index: int, **scope: object) -> dict[str, object]:
        return {
            "schema": self.spec.marker_schema,
            "attempt_id": self.attempt_id,
            "plan_fingerprint": self.plan_fingerprint,
            "roster_fingerprint": self.roster_fingerprint,
            "node_gpus": self.node_gpus,
            "worker_index": worker_index,
            **scope,
        }

    def arm_marker_identity(
        self, *, worker_index: int, arm: str, phase: str, state: str
    ) -> dict[str, object]:
        """Return the identity every marker of one arm phase must carry.

        The arm name is part of the identity, not only of the path, so a marker
        published into the wrong arm's directory is refused by name.
        """
        return self._identity(worker_index, arm=arm, phase=phase, state=state)

    def verify_identity(self, path: Path, expected: Mapping[str, object]) -> dict[str, Any]:
        """Read one marker and refuse any identity outside the expected scope."""
        observed = read_marker(path, self.spec)
        drifted = sorted(name for name in expected if observed.get(name) != expected[name])
        if drifted:
            raise RuntimeError(
                f"{path}: {self.spec.family_label} barrier marker identity "
                f"drifted at {', '.join(drifted)}"
            )
        return observed

    def publish_marker(self, path: Path, identity: Mapping[str, object]) -> None:
        """Publish this worker's marker, or validate the one already there."""
        if not path.is_file():
            io.atomic_json(path, {**identity, "utc_unix": time.time()})
            return
        observed = read_marker(path, self.spec)
        if {name: observed.get(name) for name in identity} != identity:
            raise RuntimeError(f"{path}: {self.spec.family_label} barrier marker identity drifted")


def _marker_times(
    run: BarrierRun,
    *,
    phase_root: Path,
    state: str,
    identity: Callable[[int], dict[str, object]],
) -> dict[str, float]:
    paths = sorted((phase_root / state).glob("gpu*.json"))
    if not {path.name for path in paths} <= run.expected_names:
        raise RuntimeError(
            f"{phase_root}/{state}: marker outside {run.spec.family_label} GPU roster"
        )
    times: dict[str, float] = {}
    for path in paths:
        observed = run.verify_identity(path, identity(int(path.stem.removeprefix("gpu"))))
        times[path.stem] = float(observed["utc_unix"])
    return times


def arm_marker_times(run: BarrierRun, *, arm: str, phase: str, state: str) -> dict[str, float]:
    """Return every published marker time for one arm phase state."""
    return _marker_times(
        run,
        phase_root=run.arm_phase_root(arm, phase),
        state=state,
        identity=lambda worker: run.arm_marker_identity(
            worker_index=worker, arm=arm, phase=phase, state=state
        ),
    )


def arm_phase_roster(arms: Sequence[str]) -> tuple[str, ...]:
    """Return the split fleet's four-phase order for every arm, in arm order."""
    if not arms:
        raise ValueError("a split-fleet roster needs at least one arm")
    if len(set(arms)) != len(arms):
        raise ValueError("split-fleet arms must be unique")
    return tuple(f"{arm}:{phase}" for arm in arms for phase in ARM_PHASES)


def arm_dirname(arm: str) -> str:
    """Return the marker directory component for one arm, or refuse the name."""
    if not arm or arm.startswith(".") or not set(arm) <= _SAFE_ARM_CHARS:
        raise ValueError(f"unsafe split-fleet arm name {arm!r}")
    return arm


def arm_role_roster(placement: FleetPlacement) -> dict[str, tuple[int, ...]]:
    """Return the worker indices of one arm's placement, keyed by their role.

    The keys are the roles :meth:`FleetPlacement.role` hands out. A fused
    placement has one role and publishes one activity phase.
    """
    everyone = tuple(range(placement.workers))
    if placement.fused:
        return {"fused": everyone}
    return {
        "producers": everyone[: placement.producers],
        "receivers": everyone[placement.producers :],
    }


__all__ = (
    "ARM_PHASES",
    "SPLIT_FLEET_ISOLATION_PROFILE",
    "STATES",
    "BarrierRun",
    "BarrierSpec",
    "arm_dirname",
    "arm_marker_times",
    "arm_phase_roster",
    "arm_role_roster",
    "read_marker",
)
