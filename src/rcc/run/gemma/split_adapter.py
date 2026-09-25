"""The Gemma binding of the arm-at-a-time split-fleet runner.

Everything family-specific about a split Gemma arm lives here: the panel items,
each v16 route's placement, row identity, and the worker shape for a role.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from importlib.metadata import version
from pathlib import Path
from typing import Any

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.hardware.fleet import FleetPlacement
from rcc.hardware.placements import SEMANTIC_ARM_PLACEMENTS, bind_policy_placements
from rcc.models.gemma import GEMMA
from rcc.models.gemma.config import GemmaFleetConfig
from rcc.models.gemma.contract import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    FLEET_RUNTIME,
    ROW_SCHEMA,
    registration,
    registration_fingerprint,
    runtime_fingerprint,
    sampling_contract,
    scientific_config_fingerprint,
)
from rcc.models.gemma.roster import (
    SHARED_RESULT_PANEL,
    gemma_semantic_bindings,
    shared_execution_roster,
)
from rcc.models.gemma.tokenizer import load_gemma4_tokenizer
from rcc.run import identity as run_identity
from rcc.run.barrier import SPLIT_FLEET_ISOLATION_PROFILE, BarrierSpec
from rcc.run.contract import (
    BankCallback,
    BankIdentity,
    FleetWorker,
    WorkerContext,
    placement_identity,
)
from rcc.run.fleet.merge import worker_bank_paths
from rcc.run.gemma.arm_subsets import execution_arm_names
from rcc.run.gemma.cell_banks import worker_bank_lock
from rcc.run.gemma.split_report import gemma_split_report_adapter

SPLIT_PHASE_MARKER_SCHEMA = "gemma-split-arm-barrier-marker-v1"
#: The identity of the Gemma arm barrier the split driver publishes under. It
#: carries its own marker directory, so a split barrier marker is never read as
#: another route's, or the reverse.
GEMMA_SPLIT_BARRIER = BarrierSpec(
    family_label="Gemma split",
    marker_schema=SPLIT_PHASE_MARKER_SCHEMA,
    barrier_dirname="gemma_split_arm_barriers",
)


def gemma_runtime_signature() -> dict[str, Any]:
    """Return the runtime identity shared by Gemma markers and banks."""
    signature = run_identity.runtime_signature({CHECKPOINT_ID: CHECKPOINT_REVISION})
    signature["auxiliary_packages"] = {
        package: version(package) for package, _expected in GEMMA.runtime.auxiliary_packages
    }
    return signature


class GemmaSplitDataAdapter:
    """Load the shared held-out panel every split worker serves."""

    def __init__(self, config: GemmaFleetConfig, *, panel_loader: Any = None) -> None:
        """Bind one source bundle and item range."""
        if panel_loader is None:
            from rcc.benchmarks.fanoutqa.gemma_data import load_shared_worker_panel

            panel_loader = load_shared_worker_panel
        self.config = config
        self._panel_loader = panel_loader
        self._items: dict[str, Any] = {}
        self._qids: tuple[str, ...] = ()

    @property
    def qids(self) -> tuple[str, ...]:
        """Return the loaded panel in its frozen order."""
        return self._qids

    def load_prepared(
        self,
        root: Path,
        *,
        panel: str,
        source_commit: str,
    ) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
        """Load the whole panel once: a split arm shards by role, not by item."""
        del root, panel, source_commit
        loaded = self._panel_loader(
            self.config.source_cache,
            load_gemma4_tokenizer(checkpoint_id=CHECKPOINT_ID, revision=CHECKPOINT_REVISION),
            offset=self.config.item_offset,
            count=self.config.item_count,
            node_gpus=1,
            worker_index=0,
            profile=self.config.benchmark_profile,
        )
        self._qids = tuple(loaded.qids)
        self._items = {str(item["qid"]): item for item in loaded.items}
        return tuple(loaded.items), {
            "artifact_sha256": loaded.prepared_sha256,
            "source_manifest": dict(loaded.source_manifest),
        }

    def item(self, qid: str) -> Any:
        """Return one already-loaded prepared item."""
        try:
            return self._items[qid]
        except KeyError as exc:
            raise KeyError(f"Gemma qid {qid!r} is outside the loaded panel") from exc


class GemmaSplitPlacementAdapter:
    """Place every Gemma arm from the shared semantic table."""

    def __init__(
        self, profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50, *, arm_profile: str = ""
    ) -> None:
        """Bind the ordered execution roster of the selected panel."""
        self._profile = profile
        self._arms = execution_arm_names(profile, arm_profile)

    @property
    def arm_names(self) -> tuple[str, ...]:
        """Return the executable semantic roster, in benchmark order."""
        return self._arms

    def resolve(
        self,
        root: Path,
        *,
        panel: str,
        source_commit: str,
    ) -> Mapping[str, FleetPlacement]:
        """Return the split of every bound Gemma arm.

        The fleet is keyed by semantic arm end to end, so the placements the
        shared registry returns are re-keyed onto those semantic names.
        """
        del root, panel, source_commit
        bindings = gemma_semantic_bindings()
        placements = {
            bindings[v16_arm]: placement
            for v16_arm, placement in bind_policy_placements(
                bindings, table=SEMANTIC_ARM_PLACEMENTS
            ).items()
        }
        return {arm: placements[arm] for arm in self.arm_names}


class GemmaSplitFleetAdapter:
    """Compose the Gemma data, placement, report, and worker seams."""

    def __init__(
        self, config: GemmaFleetConfig, *, panel_loader: Any = None, arm_profile: str = ""
    ) -> None:
        """Create one adapter over one source configuration."""
        self.config = config
        self._profile = config.benchmark_profile
        self.data: Any = GemmaSplitDataAdapter(config, panel_loader=panel_loader)
        self.placements: Any = GemmaSplitPlacementAdapter(self._profile, arm_profile=arm_profile)
        self.reports: Any = gemma_split_report_adapter(
            self.placements, self.completion_key, profile=self._profile
        )

    def registration(self) -> Mapping[str, Any]:
        """Return benchmark, model, roster, and split-route identities."""
        roster = shared_execution_roster(self._profile)
        roster.require_complete()
        return {
            "benchmark": self.config.benchmark_profile.to_dict(),
            "model": GEMMA.to_dict(),
            "route": registration(),
            "execution_roster_fingerprint": roster.fingerprint,
            "isolation_profile": SPLIT_FLEET_ISOLATION_PROFILE,
        }

    def sampling_contract(self) -> Mapping[str, Any]:
        """Return the Gemma decode settings and sample identity."""
        return dict(sampling_contract())

    def bank_paths(
        self,
        root: Path,
        *,
        panel: str,
        arm: str,
        placement: FleetPlacement,
    ) -> tuple[Path, ...]:
        """Return the per-role worker banks for one arm."""
        del panel
        return worker_bank_paths(root, arm=arm, workers=placement.workers)

    def completion_key(self, row: Mapping[str, Any]) -> tuple[str, str, str] | None:
        """Return the panel, item, and cell one banked row completes.

        Only the timed s0 row completes an item. Its batched peers are banked
        first, so an item whose peers alone landed has its whole cell decoded again.
        """
        if row.get("kind") != "result" or row.get("seed_index") != 0:
            return None
        panel = str(row.get("panel") or "")
        qid = str(row.get("qid") or "")
        cell = str(row.get("cell") or "")
        return (panel, qid, cell) if panel and qid and cell else None

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
        """Bind every row to source, placement, runtime, decode, and roster.

        The identity is the shared semantic one every result row is projected
        onto, so a projected row cannot override a fixed field of its bank.
        """
        del root
        if arm not in gemma_semantic_bindings().values():
            raise ValueError(f"{arm!r} is not a registered Gemma semantic arm")
        if panel != SHARED_RESULT_PANEL:
            raise ValueError(
                f"the Gemma split route publishes every row under the "
                f"{SHARED_RESULT_PANEL!r} panel, not {panel!r}"
            )
        roster = shared_execution_roster(self._profile)
        plan_fingerprint = self.config.unified_plan_fingerprint
        if not plan_fingerprint:
            raise RuntimeError("Gemma split bank requires a plan fingerprint")
        signature = gemma_runtime_signature()
        prepared_sha256 = str(prepared_manifest["artifact_sha256"])
        return BankIdentity(
            fields={
                "schema": ROW_SCHEMA,
                "fleet_runtime": FLEET_RUNTIME,
                "panel": panel,
                "arm": arm,
                "semantic_arm": arm,
                "placement": placement_identity(placement),
                "execution_isolation_profile": SPLIT_FLEET_ISOLATION_PROFILE,
                "execution_roster_fingerprint": roster.fingerprint,
                "prepared_sha256": prepared_sha256,
                "source_commit": source_commit,
                "source_cache_fingerprint": (
                    self.config.benchmark_profile.source_logical_fingerprint
                ),
                "unified_plan_fingerprint": plan_fingerprint,
                "config_fingerprint": scientific_config_fingerprint(),
                "route_registration_fingerprint": registration_fingerprint()[:16],
                "runtime_fingerprint": runtime_fingerprint(signature),
                "runtime_signature": signature,
                "decode_profile": GEMMA.decode.profile_id,
                "decode_fingerprint": GEMMA.decode.identity_hash,
            },
            runtime_signature=signature,
        )

    def build_worker(self, context: WorkerContext, bank: BankCallback) -> FleetWorker:
        """Build the producer, receiver, or fused worker for one seat's role."""
        from rcc.run.gemma.split_build import build_split_worker

        return build_split_worker(self, context, bank)

    def bank_lock(
        self,
        root: Path,
        *,
        arm: str,
        worker_index: int,
    ) -> AbstractContextManager[None]:
        """Hold the exclusive writer lock for one worker bank."""
        return worker_bank_lock(root / "arms" / arm, worker_index)


def build_adapter(config: GemmaFleetConfig, *, arm_profile: str = "") -> GemmaSplitFleetAdapter:
    """Return one new Gemma split-fleet adapter."""
    return GemmaSplitFleetAdapter(config, arm_profile=arm_profile)


__all__ = (
    "GEMMA_SPLIT_BARRIER",
    "SPLIT_PHASE_MARKER_SCHEMA",
    "GemmaSplitDataAdapter",
    "GemmaSplitFleetAdapter",
    "GemmaSplitPlacementAdapter",
    "build_adapter",
    "gemma_runtime_signature",
    "gemma_split_report_adapter",
)
