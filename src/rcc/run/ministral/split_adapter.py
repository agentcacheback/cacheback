"""The Ministral binding of the arm-at-a-time split-fleet runner.

Everything family-specific about a split Ministral arm lives here: the panel
items, each policy's placement, row identity, and the worker shape for a role.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from rcc.benchmarks.fanoutqa.ministral_data import load_ministral_prompt_panel
from rcc.benchmarks.fanoutqa.panel import load_questions
from rcc.benchmarks.fanoutqa.source_bundle import validate_shared_source_bundle
from rcc.hardware.fleet import FleetPlacement
from rcc.hardware.placements import SEMANTIC_ARM_PLACEMENTS, bind_policy_placements
from rcc.models.ministral import (
    MINISTRAL,
    MINISTRAL_RUNTIME,
    ministral_semantic_bindings,
)
from rcc.models.ministral.payload import MINISTRAL_PAYLOAD_LAYOUT
from rcc.models.ministral.results import MINISTRAL_RESULT_SCHEMA
from rcc.models.ministral.runtime import validate_nonresident_runtime_authority
from rcc.models.ministral.text_codec import (
    MINISTRAL_TEXT_ARMS,
    MinistralTextCodec,
    load_registered_text_codec,
)
from rcc.run.barrier import SPLIT_FLEET_ISOLATION_PROFILE
from rcc.run.contract import (
    BankCallback,
    BankIdentity,
    FleetWorker,
    WorkerContext,
    placement_identity,
)
from rcc.run.fleet.merge import worker_bank_paths
from rcc.run.ministral.cell_banks import worker_bank_lock
from rcc.run.ministral.config import (
    MINISTRAL_FLEET_RUNTIME,
    MINISTRAL_SELECTION_ARTIFACT_PROFILE,
    MinistralFleetConfig,
)
from rcc.run.ministral.split_build import RECEIVER_CODEC_ARM, build_split_worker
from rcc.run.ministral.split_receiver import TIMED_SAMPLE_TAG
from rcc.run.ministral.split_report import ministral_split_report_adapter

_SEMANTIC_BY_POLICY = ministral_semantic_bindings()


@dataclass(frozen=True)
class MinistralSplitPanel:
    """One loaded panel: the items every role of one arm reads from."""

    qids: tuple[str, ...]
    items: tuple[dict[str, Any], ...]
    panel_fingerprint: str
    construction_roster_fingerprint: str


def load_split_panel(config: MinistralFleetConfig, *, panel: str) -> MinistralSplitPanel:
    """Load and revalidate the whole shared panel for one split arm.

    A split arm divides by role rather than by item, so every seat of the arm
    holds the same items and works on the ones the queue hands it.
    """
    profile = config.benchmark_profile
    source = validate_shared_source_bundle(config.source_bundle, profile=profile)
    if (
        source.get("fingerprint") != profile.source_logical_fingerprint
        or source.get("manifest_sha256") != profile.source_manifest_sha256
    ):
        raise RuntimeError("Ministral source bundle differs from the registered FanOutQA panel")
    prepared = load_ministral_prompt_panel(config.prepared_panel)
    if prepared.benchmark_profile != profile:
        raise RuntimeError("Ministral prepared panel names another benchmark profile")
    expected_qids = profile.question_ids[
        config.item_offset : config.item_offset + config.item_count
    ]
    expected_prepared = profile.question_ids if len(profile.question_ids) == 100 else expected_qids
    if prepared.qids != expected_prepared:
        raise RuntimeError("Ministral prepared panel differs from the registered item roster")
    questions = {
        question.qid: question
        for question in load_questions(
            config.source_bundle / "source_cache" / "fanout-final-dev.json"
        )
    }
    missing = [qid for qid in expected_qids if qid not in questions]
    if missing:
        raise RuntimeError(f"Ministral source omits selected questions {missing!r}")
    items = tuple(
        {
            "qid": qid,
            "panel": panel,
            "question": questions[qid],
            "prompt_ids": tuple(
                torch.tensor(list(row), dtype=torch.long)
                for row in prepared.artifact(qid, RECEIVER_CODEC_ARM).prompt_ids_by_worker
            ),
            "artifacts": {arm: prepared.artifact(qid, arm) for arm in MINISTRAL_TEXT_ARMS},
            "construction": {
                arm: prepared.construction_manifest(qid, arm) for arm in MINISTRAL_TEXT_ARMS
            },
        }
        for qid in expected_qids
    )
    return MinistralSplitPanel(
        qids=expected_qids,
        items=items,
        panel_fingerprint=prepared.fingerprint,
        construction_roster_fingerprint=prepared.construction_manifest_roster_fingerprint,
    )


class MinistralSplitDataAdapter:
    """Load the shared held-out panel every split worker serves."""

    def __init__(self, config: MinistralFleetConfig, *, panel_loader: Any = None) -> None:
        """Bind one prepared panel and source bundle."""
        self.config = config
        panel_loader = load_split_panel if panel_loader is None else panel_loader
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
        """Load the whole panel once: a split arm divides by role, not by item."""
        del root, source_commit
        loaded = self._panel_loader(self.config, panel=panel)
        self._qids = tuple(loaded.qids)
        self._items = {str(item["qid"]): item for item in loaded.items}
        return tuple(loaded.items), {
            "artifact_sha256": loaded.panel_fingerprint,
            "construction_roster_fingerprint": loaded.construction_roster_fingerprint,
        }

    def item(self, qid: str) -> Any:
        """Return one already-loaded prepared item."""
        try:
            return self._items[qid]
        except KeyError as exc:
            raise KeyError(f"Ministral qid {qid!r} is outside the loaded panel") from exc


class MinistralSplitPlacementAdapter:
    """Place every Ministral arm from the shared semantic table."""

    @property
    def arm_names(self) -> tuple[str, ...]:
        """Return the executable roster, in benchmark order."""
        return tuple(_SEMANTIC_BY_POLICY.values())

    def resolve(
        self,
        root: Path,
        *,
        panel: str,
        source_commit: str,
    ) -> Mapping[str, FleetPlacement]:
        """Return the split of every bound Ministral arm."""
        del root, panel, source_commit
        # Ministral policies and semantic arms are one-to-one, so the runner
        # is keyed by the semantic arm the placement table already names.
        return {
            semantic_arm: placement
            for policy, placement in bind_policy_placements(
                _SEMANTIC_BY_POLICY, table=SEMANTIC_ARM_PLACEMENTS
            ).items()
            for semantic_arm in (_SEMANTIC_BY_POLICY[policy],)
        }


class MinistralSplitFleetAdapter:
    """Compose the Ministral data, placement, report, and worker seams."""

    def __init__(
        self,
        config: MinistralFleetConfig,
        *,
        panel_loader: Any = None,
        codec_loader: Any = load_registered_text_codec,
        authority_validator: Any = validate_nonresident_runtime_authority,
    ) -> None:
        """Create one adapter over one source configuration."""
        self.config = config
        self._codec_loader = codec_loader
        self._authority_validator = authority_validator
        self._codecs: dict[str, MinistralTextCodec] | None = None
        self.data: Any = MinistralSplitDataAdapter(config, panel_loader=panel_loader)
        self.placements: Any = MinistralSplitPlacementAdapter()
        self.reports: Any = ministral_split_report_adapter(
            self.placements,
            self.completion_key,
            unified_plan_fingerprint=config.unified_plan_fingerprint,
        )

    @property
    def codecs(self) -> Mapping[str, MinistralTextCodec]:
        """Return the sender codecs, loaded once per process."""
        if self._codecs is None:
            self._codecs = {
                arm: self._codec_loader(Path(self.config.tokenizer_snapshots[arm]), arm)
                for arm in MINISTRAL_TEXT_ARMS
            }
        return self._codecs

    def registration(self) -> Mapping[str, Any]:
        """Return benchmark, model, roster, and split-route identities."""
        return {
            "benchmark": self.config.benchmark_profile.to_dict(),
            "model": MINISTRAL.to_dict(),
            "route": {
                "schema": MINISTRAL_RESULT_SCHEMA,
                "payload_layout": MINISTRAL_PAYLOAD_LAYOUT,
                "selection_artifact_profile": MINISTRAL_SELECTION_ARTIFACT_PROFILE,
                "runtime_profile_fingerprint": MINISTRAL_RUNTIME.runtime_identity_hash,
            },
            "isolation_profile": SPLIT_FLEET_ISOLATION_PROFILE,
        }

    def sampling_contract(self) -> Mapping[str, Any]:
        """Return the Ministral decode settings and sample identity."""
        return {
            **MINISTRAL.decode.to_dict(),
            "answer_ceiling": self.config.benchmark_profile.answer_ceiling,
            "report_ceiling": self.config.benchmark_profile.report_ceiling,
            "sample_tags": list(self.config.benchmark_profile.sample_tags),
        }

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
        """Return the panel, item, and cell one timed row completes.

        A Ministral cell is three rows banked at different moments, so only the
        timed row completes the item and a later attempt decodes the whole cell.
        """
        if row.get("kind") != "result" or row.get("sample_tag") != TIMED_SAMPLE_TAG:
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
        """Bind every row to source, placement, plan, decode, and roster."""
        del root
        if arm not in set(_SEMANTIC_BY_POLICY.values()):
            raise ValueError(f"unregistered Ministral split arm {arm!r}")
        raw_authority = self.config.nonresident_runtime_authority
        authority = (
            self._authority_validator(raw_authority) if isinstance(raw_authority, Mapping) else None
        )
        return BankIdentity(
            fields={
                "schema": MINISTRAL_RESULT_SCHEMA,
                "fleet_runtime": MINISTRAL_FLEET_RUNTIME,
                "panel": panel,
                "arm": arm,
                "semantic_arm": arm,
                "placement": placement_identity(placement),
                "execution_isolation_profile": SPLIT_FLEET_ISOLATION_PROFILE,
                "prepared_sha256": str(prepared_manifest["artifact_sha256"]),
                "prepared_panel_fingerprint": str(prepared_manifest["artifact_sha256"]),
                "construction_manifest_roster_fingerprint": str(
                    prepared_manifest["construction_roster_fingerprint"]
                ),
                "source_commit": source_commit,
                "benchmark_profile": self.config.benchmark_profile.profile_id,
                "source_identity": self.config.benchmark_profile.source_logical_fingerprint,
                "source_cache_fingerprint": (
                    self.config.benchmark_profile.source_logical_fingerprint
                ),
                "source_manifest_sha256": self.config.benchmark_profile.source_manifest_sha256,
                "model_id": MINISTRAL.model_id,
                "unified_plan_fingerprint": self.config.unified_plan_fingerprint,
                "nonresident_runtime_authority": authority,
                "runtime_profile_fingerprint": MINISTRAL_RUNTIME.runtime_identity_hash,
                "decode_profile": MINISTRAL.decode.profile_id,
                "decode_fingerprint": MINISTRAL.decode.identity_hash,
            }
        )

    def build_worker(self, context: WorkerContext, bank: BankCallback) -> FleetWorker:
        """Build the producer, receiver, or fused worker for one seat's role."""
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


def build_adapter(config: MinistralFleetConfig) -> MinistralSplitFleetAdapter:
    """Return one new Ministral split-fleet adapter."""
    return MinistralSplitFleetAdapter(config)


__all__ = (
    "MinistralSplitDataAdapter",
    "MinistralSplitFleetAdapter",
    "MinistralSplitPlacementAdapter",
    "build_adapter",
    "ministral_split_report_adapter",
)
