"""The Ministral split worker's invocation config and prompt records.

The config binds one invocation to its source bundle, prepared panel, tokenizer
snapshots, and run identity. The prompt records are rebuilt without re-encoding.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.ministral.text_prompt import (
    MinistralPreparedPromptArtifact,
    MinistralPromptRecord,
    registered_text_senders,
)
from rcc.run import io
from rcc.run.ministral.barrier import MINISTRAL_EXECUTION_ROSTER_FINGERPRINT

_TEXT_ARMS = tuple(sender.semantic_arm for sender in registered_text_senders())
_IDENTIFIER_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
MINISTRAL_SELECTION_ARTIFACT_PROFILE = "query-support-p2-a2-w16-exact-keeps-v1"
MINISTRAL_FLEET_RUNTIME = "generic-fleet-serving-v1"


@dataclass(frozen=True)
class MinistralFleetConfig:
    """One Ministral split-worker invocation and the identity its rows carry."""

    results_base: Path
    source_bundle: Path
    prepared_panel: Path
    tokenizer_snapshots: Mapping[str, Path]
    item_count: int
    node_gpus: int
    worker_index: int
    attempt_id: str
    source_commit: str
    unified_plan_fingerprint: str
    item_offset: int = 0
    benchmark_profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50
    barrier_timeout_s: float = 7_200.0
    barrier_poll_s: float = 0.05
    nonresident_runtime_authority: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        """Refuse an incomplete identity, placement, or tokenizer roster."""
        if self.node_gpus < 1 or not 0 <= self.worker_index < self.node_gpus:
            raise ValueError("Ministral worker index is outside the GPU roster")
        if not self.attempt_id or any(
            character not in _IDENTIFIER_CHARS for character in self.attempt_id
        ):
            raise ValueError("Ministral attempt id must be a plain identifier")
        if len(self.source_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.source_commit
        ):
            raise ValueError("Ministral source commit must be a full lowercase git SHA")
        if not io.is_sha256_hex(self.unified_plan_fingerprint):
            raise ValueError("Ministral plan fingerprint must be full SHA-256")
        if set(self.tokenizer_snapshots) != set(_TEXT_ARMS):
            raise ValueError("Ministral tokenizer snapshots differ from three sender lanes")
        if min(self.barrier_timeout_s, self.barrier_poll_s) <= 0:
            raise ValueError("Ministral barrier timing must be positive")

    @property
    def shard_tag(self) -> str:
        """Return this shard's tag: its item offset and item count."""
        return f"i{self.item_offset:03d}n{self.item_count:03d}"

    @property
    def run_root(self) -> Path:
        """Return the run directory, named by the roster fingerprint and shard tag."""
        prefix = MINISTRAL_EXECUTION_ROSTER_FINGERPRINT[:12]
        return Path(self.results_base) / f"fleet_{prefix}_{self.shard_tag}"

    @property
    def worker_root(self) -> Path:
        """Return this GPU worker's directory under the run root."""
        return self.run_root / "workers" / f"gpu{self.worker_index}"


def prompt_records(
    artifact: MinistralPreparedPromptArtifact,
) -> tuple[MinistralPromptRecord, ...]:
    """Rebuild the prepared token rows without re-encoding the source."""
    return tuple(
        MinistralPromptRecord(
            qid=artifact.qid,
            semantic_arm=artifact.semantic_arm,
            worker=worker,
            source_identity=artifact.source_identity,
            checkpoint=artifact.checkpoint,
            revision=artifact.revision,
            token_ids=tokens,
            token_sha256=artifact.prompt_sha256[worker],
            prepared_token_sha256=artifact.prompt_sha256[worker],
            prepared_fingerprint=artifact.fingerprint,
            profile=artifact.profile,
        )
        for worker, tokens in enumerate(artifact.prompt_ids_by_worker)
    )


__all__ = (
    "MINISTRAL_FLEET_RUNTIME",
    "MINISTRAL_SELECTION_ARTIFACT_PROFILE",
    "MinistralFleetConfig",
    "prompt_records",
)
