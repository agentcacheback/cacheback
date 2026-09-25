"""The resolved run plan and the identities derived from it.

The plan carries the profiles, the arm roster, the item range, and the run id.
Its run fingerprint covers one pass; its execution fingerprint the whole bank.
"""

from __future__ import annotations

from dataclasses import dataclass

from rcc.benchmarks.protocol import ArmSpec, BenchmarkProfile
from rcc.hardware.metrics import ExpectedRows
from rcc.hardware.profile import HardwareProfile
from rcc.models.decode import DecodeProtocol
from rcc.models.protocol import ModelProfile, PhysicalArm
from rcc.run.io import canonical_sha
from rcc.topologies.protocol import TopologyProfile


@dataclass(frozen=True)
class ResolvedPlan:
    """One resolved run: its profiles, its arms, its item range, and its ids."""

    schema_version: int
    model: ModelProfile
    benchmark: BenchmarkProfile
    topology: TopologyProfile
    hardware: HardwareProfile
    arm_profile: str
    selector_profile: str
    support_composition: str
    text_profile: str
    isolation_profile: str
    item_start: int
    item_count: int
    selected_question_ids: tuple[str, ...]
    git_commit: str
    run_id: str
    output_uri: str
    arms: tuple[ArmSpec, ...]
    physical_arms: tuple[PhysicalArm, ...]
    expected_rows: ExpectedRows

    @property
    def scientific_identity_hash(self) -> str:
        """Return the benchmark identity, which no model takes part in."""
        return self.benchmark.scientific_identity_hash

    @property
    def decode(self) -> DecodeProtocol:
        """Return the model family's decode contract."""
        return self.model.decode

    @property
    def is_declared_pass(self) -> bool:
        """Return whether this plan is one declared pass over a shared bank."""
        return (self.item_start, self.item_count) in self.benchmark.declared_passes

    @property
    def execution_question_ids(self) -> tuple[str, ...]:
        """Return the items the offline validators read: every declared pass."""
        if not self.is_declared_pass:
            return self.selected_question_ids
        ids = self.benchmark.question_ids
        passes = self.benchmark.declared_passes
        return tuple(q for start, count in passes for q in ids[start : start + count])

    @property
    def execution_expected_rows(self) -> ExpectedRows:
        """Return the whole result grid of this execution tree."""
        if not self.is_declared_pass:
            return self.expected_rows
        return ExpectedRows(
            items=len(self.execution_question_ids),
            arms=len(self.arms),
            seeds=len(self.benchmark.sample_tags),
        )

    def _identity_dict(self) -> dict[str, object]:
        """Return the run inputs the plan fingerprint covers."""
        return {
            "schema_version": self.schema_version,
            "model": self.model.to_dict(),
            "decode": self.decode.to_dict(),
            "decode_profile": self.decode.profile_id,
            "decode_fingerprint": self.decode.identity_hash,
            "benchmark": self.benchmark.to_dict(),
            "topology": self.topology.to_dict(),
            "hardware": self.hardware.to_dict(),
            "arm_profile": self.arm_profile,
            "selector_profile": self.selector_profile,
            "support_composition": self.support_composition,
            "text_profile": self.text_profile,
            "isolation_profile": self.isolation_profile,
            "item_range": {"start": self.item_start, "count": self.item_count},
            "selected_question_ids": list(self.selected_question_ids),
            "git_commit": self.git_commit,
            "run_id": self.run_id,
            "output_uri": self.output_uri,
            "semantic_arms": [arm.to_dict() for arm in self.arms],
            "physical_arm_mapping": [arm.to_dict() for arm in self.physical_arms],
            "expected_rows": self.expected_rows.to_dict(),
            "scientific_fingerprint": self.scientific_identity_hash,
        }

    @property
    def run_identity_hash(self) -> str:
        """Return this pass's run identity, including its Git commit."""
        return canonical_sha(self._identity_dict())

    @property
    def execution_identity_hash(self) -> str:
        """Return the bank identity shared by every declared pass of one run.

        :attr:`run_identity_hash` stays pass-scoped and signs the exact range.
        This one covers all the declared passes, which append to one bank.
        """
        if not self.is_declared_pass:
            return self.run_identity_hash
        body = self._identity_dict()
        passes = self.benchmark.declared_passes
        body["item_range"] = {
            "passes": [{"start": start, "count": count} for start, count in passes]
        }
        body["selected_question_ids"] = list(self.execution_question_ids)
        body["expected_rows"] = self.execution_expected_rows.to_dict()
        return canonical_sha(body)

    def to_dict(self) -> dict[str, object]:
        """Return the whole plan body with its fingerprints."""
        return {
            **self._identity_dict(),
            "run_fingerprint": self.run_identity_hash,
            "execution_fingerprint": self.execution_identity_hash,
        }


__all__ = ("ResolvedPlan",)
