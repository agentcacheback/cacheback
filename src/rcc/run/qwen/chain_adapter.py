"""The LongBench chain's fleet adapter: its panel, its placements, its seats.

The chain reaches the shared runner through the same contract FanOutQA does and
extends the lane's own adapter, so what reads no topology is inherited.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from rcc.benchmarks.fanoutqa.prepare import prepared_source_commit
from rcc.benchmarks.fanoutqa.source_audit import validate_source_commit
from rcc.benchmarks.longbench_v2.data import ChainItem
from rcc.benchmarks.longbench_v2.prepare import load_prepared_panel
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.hardware.fleet import FleetPlacement
from rcc.hardware.placements import placement_table
from rcc.hardware.qwen_manifest import placement_fingerprint
from rcc.models.route import RouteFamily
from rcc.run.barrier import SPLIT_FLEET_ISOLATION_PROFILE
from rcc.run.contract import (
    BankCallback,
    BankIdentity,
    FleetWorker,
    WorkerContext,
    placement_identity,
)
from rcc.run.io import canonical_sha
from rcc.run.qwen.adapter import (
    FanoutQAAdapter,
    execution_qids,
    execution_roster_fingerprint,
    roster_model,
    route_runtime_signature,
)
from rcc.run.qwen.runtime import runtime_identity_fields
from rcc.run.runlog import runtime_fingerprint


def chain_placement_fingerprint(profile: BenchmarkProfile) -> str:
    """Sign the chain table's rows for the arms one profile declares.

    The table comes from :func:`placement_table`, and only the profile's own rows
    are signed, so a seat added for another profile leaves this identity intact.
    """
    table = placement_table(profile.topology_key)
    registered = {arm.arm_id for arm in profile.arms}
    missing = sorted(registered - set(table))
    if missing:
        raise KeyError(f"{profile.benchmark_key}: no chain placement for {missing}")
    return canonical_sha(
        {
            "schema": "longbench-v2-coa-provisional-placement-v2",
            "topology": profile.topology_key,
            "placements": {
                semantic_arm: placement_identity(placement)
                for semantic_arm, placement in sorted(table.items())
                if semantic_arm in registered
            },
        }
    )[:16]


def _roster_fingerprint(family: RouteFamily, profile: BenchmarkProfile, placements: str) -> str:
    return canonical_sha(
        {
            "schema": "longbench-v2-coa-execution-roster-v1",
            "topology": profile.topology_key,
            "lane_roster": execution_roster_fingerprint(family),
            "placements": placements,
        }
    )


def chain_roster_fingerprint(family: RouteFamily, profile: BenchmarkProfile) -> str:
    """Sign the lane's arm roster beside the profile's own table and topology.

    A chain seat and a fan-out seat divide a node's GPUs differently, so signing
    the lane roster alone would let both share one barrier identity.
    """
    return _roster_fingerprint(family, profile, chain_placement_fingerprint(profile))


class ChainDataAdapter:
    """Load the whole chain panel and expose this pass's execution subset."""

    def __init__(self, family: RouteFamily, *, profile: BenchmarkProfile) -> None:
        """Bind the family and the chain benchmark whose prepared bytes this loads."""
        self.family = family
        self.profile = family.benchmark_profile(profile)
        self._items: dict[str, ChainItem] = {}
        self._sealed: tuple[ChainItem, ...] = ()
        self._qids: tuple[str, ...] = ()

    @property
    def qids(self) -> tuple[str, ...]:
        """Return the loaded execution subset, in panel order."""
        return self._qids

    @property
    def sealed_items(self) -> tuple[ChainItem, ...]:
        """Return every validated item of the whole panel, not this pass's subset.

        A panel property such as a chance floor is read off the whole bundle,
        already loaded and validated here, rather than loading the bytes twice.
        """
        return self._sealed

    def load_prepared(
        self,
        root: Path,
        *,
        panel: str,
        source_commit: str,
    ) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
        """Validate the whole panel, then select this pass's items.

        The bytes are bound to the commit the panel was prepared at, which
        ``RCC_FANOUT_PREPARED_SOURCE_COMMIT`` names, not the run's own commit.
        """
        del panel
        items, manifest = load_prepared_panel(
            root,
            source_commit=prepared_source_commit(source_commit, profile=self.profile),
            profile=self.profile,
        )
        self._sealed = items
        by_qid = {item.qid: item for item in items}
        self._qids = execution_qids(profile=self.profile)
        self._items = {qid: by_qid[qid] for qid in self._qids}
        return tuple(self._items[qid] for qid in self._qids), manifest

    def item(self, qid: str) -> ChainItem:
        """Return one already-validated prepared chain item."""
        try:
            return self._items[qid]
        except KeyError:
            raise KeyError(f"LongBench chain qid {qid!r} is outside the loaded subset") from None


class ChainAdapter(FanoutQAAdapter):
    """Compose the chain's data, placements, and seats with the lane's report.

    Inherited from the lane: bank addressing, cell keying, locking, the sampling
    contract, the registry. The chain's own are loader, roster identity, seats.
    """

    def __init__(self, family: RouteFamily, *, profile: BenchmarkProfile) -> None:
        """Create one chain adapter and its prepared-item lookup."""
        super().__init__(family, profile=profile)
        self.data = ChainDataAdapter(family, profile=profile)

    def registration(self) -> Mapping[str, Any]:
        """Return benchmark, model, and topology identities together.

        The roster fingerprint is the chain's own, so a chain barrier is never a
        fan-out barrier; it is published under the key FanOutQA uses.
        """
        return {
            "benchmark": self.profile.to_dict(),
            "model": roster_model(self.family, self.profile),
            "execution_roster_fingerprint": chain_roster_fingerprint(self.family, self.profile),
            "isolation_profile": SPLIT_FLEET_ISOLATION_PROFILE,
        }

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
            },
            runtime_signature=signature,
        )

    def build_worker(self, context: WorkerContext, bank: BankCallback) -> FleetWorker:
        """Build the hop producer, the sender, or the receiver for one physical role."""
        from transformers import AutoTokenizer

        from rcc.run.qwen.chain_producers import QwenChainLatentProducer, QwenChainTextProducer
        from rcc.run.qwen.worker import QwenReceiver

        model = self.family.profile
        arm = next(arm for arm in model.physical_arms if arm.policy == context.arm)
        role = context.placement.role(context.worker_index)
        items = dict(zip(context.qids, context.items, strict=True))
        arm_root = context.root / "arms" / context.arm
        if role in {"receiver", "fused"}:
            tokenizer: Any = cast(Any, AutoTokenizer).from_pretrained(
                model.tokenizer,
                revision=model.tokenizer_revision,
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
                model.tokenizer,
                revision=model.tokenizer_revision,
            )
            return QwenChainLatentProducer(
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
        return QwenChainTextProducer(
            policy=context.arm,
            tokenizer=tokenizer,
            items=items,
            payload_root=arm_root / "payloads",
            bank=bank,
            family=self.family,
            profile=self.profile,
        )


__all__ = (
    "ChainAdapter",
    "ChainDataAdapter",
    "chain_placement_fingerprint",
    "chain_roster_fingerprint",
)
