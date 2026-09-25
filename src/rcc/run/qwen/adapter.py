"""The route lane's FanOutQA adapter for the arm-at-a-time fleet runner.

It composes the seams the runner needs: the panel loader, the placement table,
the report, the bank identity, and the worker built for one seat.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.fanoutqa.prepare import load_prepared_panel
from rcc.benchmarks.fanoutqa.source_audit import validate_source_commit
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.hardware.fleet import FleetPlacement
from rcc.hardware.qwen_manifest import (
    placement_fingerprint,
    provisional_placements,
    registered_policies,
    route_placements,
)
from rcc.models.route import RouteFamily
from rcc.run.barrier import (
    SPLIT_FLEET_ISOLATION_PROFILE,
    BarrierSpec,
    arm_phase_roster,
)
from rcc.run.contract import (
    BankCallback,
    BankIdentity,
    BenchmarkDataAdapter,
    BenchmarkReportAdapter,
    FleetWorker,
    JsonRow,
    SequenceResultPolicy,
    WorkerContext,
)
from rcc.run.fanout_banks import worker_bank_lock
from rcc.run.io import canonical_sha
from rcc.run.plan import (
    route_family_for_lane,
)
from rcc.run.qwen.execution import (
    QWEN_ITEM_COUNT,
    QWEN_ITEM_START,
    environment_family,
    execution_profile,
    execution_qids,
)
from rcc.run.qwen.runtime import engine_route, runtime_identity_fields, runtime_route_field
from rcc.run.runlog import runtime_fingerprint
from rcc.run.runlog import runtime_signature as observed_runtime_signature
from rcc.topologies.chain import CHAIN_TOPOLOGY_KEY

if TYPE_CHECKING:
    from rcc.run.qwen.chain_adapter import ChainAdapter


def route_split_barrier(family: RouteFamily) -> BarrierSpec:
    """Return the split-arm barrier one lane publishes its markers under."""
    return BarrierSpec(
        family_label=f"{family.lane.capitalize()} split",
        marker_schema=f"{family.lane}-split-arm-barrier-marker-v1",
        barrier_dirname=f"{family.lane}_split_arm_barriers",
    )


def execution_roster_fingerprint(family: RouteFamily) -> str:
    """Sign one lane's arm roster and its GPU splits.

    The roster signed here is the lane baseline, the FanOutQA arms and their
    placements, since this value labels banks already written.
    """
    profile = family.profile
    arms = registered_policies(family, FANOUTQA_NATURAL_DEV50)
    physical = {arm.policy: arm for arm in profile.physical_arms}
    roster = tuple(physical[policy] for policy in arms)
    placements = provisional_placements(family=family)
    return canonical_sha(
        {
            "schema": f"{family.lane}-shared-split-execution-roster-v1",
            "model_id": profile.model_id,
            "decode_profile": profile.decode.profile_id,
            "decode_fingerprint": profile.decode.identity_hash,
            "sample_tags": list(FANOUTQA_NATURAL_DEV50.sample_tags),
            "arms": [
                {
                    "policy": arm.policy,
                    "semantic_arm": arm.semantic_arm,
                    "placement": {
                        "workers": placements[arm.policy].workers,
                        "producers": placements[arm.policy].producers,
                        "receivers": placements[arm.policy].receivers,
                        "fused": placements[arm.policy].fused,
                    },
                }
                for arm in roster
            ],
            "phases": list(arm_phase_roster(arms)),
        }
    )


def roster_model(family: RouteFamily, profile: BenchmarkProfile) -> dict[str, Any]:
    """Return the lane model identity, cut to the arms this profile names.

    A lane may implement arms a benchmark never names, so the model a run
    publishes names only the arms that run can walk.
    """
    registered = set(registered_policies(family, profile))
    model = dict(family.profile.to_dict())
    model["physical_arms"] = [
        arm.to_dict() for arm in family.profile.physical_arms if arm.policy in registered
    ]
    # The execution order is cut the same way, so a lane that gains an arm does
    # not move the identity of a run that never walks it.
    if "execution_order" in model:
        policy_of = {arm.semantic_arm: arm.policy for arm in family.profile.physical_arms}
        model["execution_order"] = [
            arm for arm in family.profile.execution_order if policy_of[arm] in registered
        ]
    return model


QWEN_FAMILY = route_family_for_lane("qwen")
QWEN_SPLIT_BARRIER = route_split_barrier(QWEN_FAMILY)
QWEN_EXECUTION_ROSTER_FINGERPRINT = execution_roster_fingerprint(QWEN_FAMILY)


def route_runtime_signature(
    family: RouteFamily, profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50
) -> dict[str, object]:
    """Return the runtime signature plus this benchmark's capture route."""
    runtime = family.profile.runtime
    signature = observed_runtime_signature(dict(runtime.model_revisions))
    signature["auxiliary_packages"] = {
        package: version(package) for package, _expected in runtime.auxiliary_packages
    }
    signature[runtime_route_field(family)] = engine_route(family, profile)
    return signature


class FanoutQADataAdapter:
    """Load the prepared panel and expose this pass's item subset."""

    def __init__(self, family: RouteFamily, *, profile: BenchmarkProfile) -> None:
        """Bind the family whose prepared panel this adapter loads."""
        self.family = family
        self.profile = family.benchmark_profile(profile)
        self._items: dict[str, Any] = {}
        self._qids: tuple[str, ...] = ()

    @property
    def qids(self) -> tuple[str, ...]:
        """Return the loaded execution subset, in panel order."""
        return self._qids

    def load_prepared(
        self,
        root: Path,
        *,
        panel: str,
        source_commit: str,
    ) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
        """Validate the whole prepared panel, then select this pass's range."""
        del panel
        items, manifest = load_prepared_panel(
            root,
            source_commit=source_commit,
            profile=self.profile,
            family=self.family,
        )
        by_qid = {item.qid: item for item in items}
        self._qids = execution_qids(profile=self.profile)
        self._items = {qid: by_qid[qid] for qid in self._qids}
        return tuple(self._items[qid] for qid in self._qids), manifest

    def item(self, qid: str) -> Any:
        """Return one already-validated prepared item."""
        try:
            return self._items[qid]
        except KeyError as exc:
            raise KeyError(f"FanOutQA qid {qid!r} is outside the loaded subset") from exc


class FanoutQAPlacementAdapter:
    """Resolve the split of every arm one benchmark seats."""

    def __init__(self, family: RouteFamily, *, profile: BenchmarkProfile) -> None:
        """Bind one route family's physical roster."""
        self.family = family
        self.profile = family.benchmark_profile(profile)

    @property
    def arm_names(self) -> tuple[str, ...]:
        """Return the physical route roster this benchmark names."""
        return registered_policies(self.family, self.profile)

    @property
    def registered(self) -> Mapping[str, FleetPlacement]:
        """Return this topology's placement table, cut to the profile's arms.

        A caller with no run root, such as the report reader, holds a bank to
        the split it ran by reading here.
        """
        return route_placements(self.family, self.profile)

    def resolve(
        self,
        root: Path,
        *,
        panel: str,
        source_commit: str,
    ) -> Mapping[str, FleetPlacement]:
        """Return one complete placement map, from the registry alone."""
        del root, panel, source_commit
        return self.registered


class FanoutQAReportAdapter:
    """Validate the banked rows and write the route report."""

    def __init__(self, family: RouteFamily) -> None:
        """Bind the family whose rows and report this adapter builds."""
        self.family = family

    def build_report(
        self,
        root: Path,
        *,
        source_commit: str,
        results_policy: SequenceResultPolicy | None = None,
        source: Sequence[JsonRow] | None = None,
    ) -> Mapping[str, Any]:
        """Read the banked rows, rescore raw ids, and write the route report."""
        from rcc.run.qwen.report import build_qwen_report, read_qwen_result_rows

        del results_policy
        rows = (
            list(source) if source is not None else read_qwen_result_rows(root, family=self.family)
        )
        return build_qwen_report(root, rows=rows, source_commit=source_commit, family=self.family)


class FanoutQAAdapter:
    """Compose the FanOutQA data, placement, report, and worker seams."""

    def __init__(self, family: RouteFamily, *, profile: BenchmarkProfile) -> None:
        """Create one adapter and its prepared-item lookup."""
        self.family = family
        self.profile = family.benchmark_profile(profile)
        self.data: BenchmarkDataAdapter = FanoutQADataAdapter(family, profile=profile)
        self.placements: FanoutQAPlacementAdapter = FanoutQAPlacementAdapter(
            family, profile=profile
        )
        self.reports: BenchmarkReportAdapter[Sequence[JsonRow], SequenceResultPolicy] = (
            FanoutQAReportAdapter(family)
        )

    def registration(self) -> Mapping[str, Any]:
        """Return benchmark, model, and topology identities together."""
        return {
            "benchmark": self.profile.to_dict(),
            "model": roster_model(self.family, self.profile),
            "execution_roster_fingerprint": execution_roster_fingerprint(self.family),
            "isolation_profile": SPLIT_FLEET_ISOLATION_PROFILE,
        }

    def sampling_contract(self) -> Mapping[str, Any]:
        """Return the decode settings and the sample identity of one arm."""
        return {
            **self.family.profile.decode.to_dict(),
            "sample_tags": list(self.profile.sample_tags),
            "answer_ceiling": self.profile.answer_ceiling,
            "report_ceiling": self.profile.report_ceiling,
            **(
                {}
                if not self.profile.report_presence_penalty
                else {"report_presence_penalty": self.profile.report_presence_penalty}
            ),
        }

    def bank_paths(
        self,
        root: Path,
        *,
        panel: str,
        arm: str,
        placement: FleetPlacement,
    ) -> tuple[Path, ...]:
        """Return ordered worker banks for one arm."""
        del panel
        return tuple(
            root / "arms" / arm / "workers" / f"gpu{index}" / "raw.jsonl"
            for index in range(placement.workers)
        )

    def completion_key(self, row: Mapping[str, Any]) -> tuple[str, str, str] | None:
        """Return the panel, item, and arm one banked row completes."""
        if row.get("kind") != "result":
            return None
        panel = str(row.get("panel") or "")
        qid = str(row.get("qid") or "")
        arm = str(row.get("arm") or row.get("policy") or "")
        return (panel, qid, arm) if panel and qid and arm else None

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
        """Bind every row to source, prepared bytes, placement, runtime, and decode."""
        del root, placement
        signature = route_runtime_signature(self.family, self.profile)
        return BankIdentity(
            fields={
                "fleet_runtime": "generic-fleet-serving-v1",
                "panel": panel,
                "policy": arm,
                "placement_fingerprint": placement_fingerprint(self.family, self.profile),
                "prepared_sha256": str(prepared_manifest["artifact_sha256"]),
                "source_commit": validate_source_commit(source_commit),
                "runtime_fingerprint": runtime_fingerprint(signature),
                "decode_profile": self.family.profile.decode.profile_id,
                "decode_fingerprint": self.family.profile.decode.identity_hash,
                **runtime_identity_fields(self.family),
                **(
                    {}
                    if self.profile.judger_question is None
                    else {"judger_question": self.profile.judger_question}
                ),
            },
            runtime_signature=signature,
        )

    def build_worker(self, context: WorkerContext, bank: BankCallback) -> FleetWorker:
        """Build the producer, receiver, or fused worker for one seat's role."""
        from transformers import AutoTokenizer

        from rcc.run.qwen.worker import QwenLatentProducer, QwenReceiver, QwenTextProducer

        profile = self.family.profile
        arm = next(arm for arm in profile.physical_arms if arm.policy == context.arm)
        role = context.placement.role(context.worker_index)
        items = dict(zip(context.qids, context.items, strict=True))
        arm_root = context.root / "arms" / context.arm
        if role in {"receiver", "fused"}:
            tokenizer: Any = cast(Any, AutoTokenizer).from_pretrained(
                profile.tokenizer,
                revision=profile.tokenizer_revision,
            )
            return QwenReceiver(
                policy=context.arm,
                tokenizer=tokenizer,
                items=items,
                arm_root=arm_root,
                bank=bank,
                family=self.family,
                profile=self.profile,
                fused=role == "fused",
            )
        if arm.selector in {"snap", "support"}:
            tokenizer = cast(Any, AutoTokenizer).from_pretrained(
                profile.tokenizer,
                revision=profile.tokenizer_revision,
            )
            return QwenLatentProducer(
                policy=context.arm,
                tokenizer=tokenizer,
                items=items,
                payload_root=arm_root / "payloads",
                bank=bank,
                family=self.family,
                profile=self.profile,
            )
        if arm.sender_checkpoint is None or arm.sender_revision is None:
            raise RuntimeError(f"{context.arm}: producer role has no sender checkpoint")
        tokenizer = cast(Any, AutoTokenizer).from_pretrained(
            arm.sender_tokenizer,
            revision=arm.sender_tokenizer_revision,
        )
        return QwenTextProducer(
            policy=context.arm,
            tokenizer=tokenizer,
            items=items,
            payload_root=arm_root / "payloads",
            bank=bank,
            family=self.family,
            profile=self.profile,
        )

    def bank_lock(
        self,
        root: Path,
        *,
        arm: str,
        worker_index: int,
    ) -> AbstractContextManager[None]:
        """Hold the exclusive writer lock for one worker bank."""
        return worker_bank_lock(root / "arms" / arm, worker_index)


def build_adapter(
    family: RouteFamily, *, profile: BenchmarkProfile
) -> FanoutQAAdapter | ChainAdapter:
    """Return the fleet adapter of the topology this benchmark names.

    The topology is the one seam a benchmark moves in the fleet, and a topology
    no adapter implements is refused here by name.
    """
    # Imported here: the chain adapter composes this module's FanOutQA seams.
    from rcc.run.qwen.chain_adapter import ChainAdapter

    topologies: dict[str, type[FanoutQAAdapter] | type[ChainAdapter]] = {
        FANOUTQA_NATURAL_DEV50.topology_key: FanoutQAAdapter,
        CHAIN_TOPOLOGY_KEY: ChainAdapter,
    }
    try:
        adapter = topologies[profile.topology_key]
    except KeyError:
        raise ValueError(
            f"the {family.lane} fleet runs the {', '.join(topologies)} topologies; "
            f"{profile.benchmark_key} registers {profile.topology_key!r}"
        ) from None
    return adapter(family, profile=profile)


__all__ = (
    "QWEN_EXECUTION_ROSTER_FINGERPRINT",
    "QWEN_FAMILY",
    "QWEN_ITEM_COUNT",
    "QWEN_ITEM_START",
    "QWEN_SPLIT_BARRIER",
    "FanoutQAAdapter",
    "FanoutQADataAdapter",
    "FanoutQAPlacementAdapter",
    "FanoutQAReportAdapter",
    "build_adapter",
    "environment_family",
    "execution_profile",
    "execution_qids",
    "execution_roster_fingerprint",
    "route_runtime_signature",
    "route_split_barrier",
)
