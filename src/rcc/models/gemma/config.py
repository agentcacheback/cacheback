"""Validated value object for one resident Gemma fleet worker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.gemma.contract import registration_fingerprint

_IDENTIFIER_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


@dataclass(frozen=True)
class GemmaFleetConfig:
    """One fixed-GPU invocation inside a Gemma fleet node."""

    results_base: Path
    source_cache: Path
    item_offset: int
    item_count: int
    node_gpus: int
    worker_index: int
    attempt_id: str
    source_commit: str
    benchmark_profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50
    unified_plan_fingerprint: str = ""

    def __post_init__(self) -> None:
        """Reject malformed source, placement, and attempt identities."""
        if len(self.source_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.source_commit
        ):
            raise ValueError("source_commit must be a full lowercase git SHA")
        if self.unified_plan_fingerprint and (
            len(self.unified_plan_fingerprint) != 64
            or any(
                character not in "0123456789abcdef" for character in self.unified_plan_fingerprint
            )
        ):
            raise ValueError("unified_plan_fingerprint must be a lowercase SHA-256")
        if not self.attempt_id or any(
            character not in _IDENTIFIER_CHARS for character in self.attempt_id
        ):
            raise ValueError("attempt_id must be a plain identifier")
        if self.item_offset < 0 or self.item_count < 1:
            raise ValueError("item offset must be nonnegative and item count positive")
        if self.node_gpus < 1:
            raise ValueError("node_gpus must be positive")
        if not 0 <= self.worker_index < self.node_gpus:
            raise ValueError("worker index is outside the node GPU roster")

    @property
    def shard_tag(self) -> str:
        """Return the stable item-slice tag used in artifact paths."""
        return f"i{self.item_offset:03d}n{self.item_count:03d}"

    @property
    def run_root(self) -> Path:
        """Return the registration- and slice-bound run directory."""
        short = registration_fingerprint()[:12]
        return self.results_base / f"fleet_{short}_{self.shard_tag}"

    @property
    def worker_root(self) -> Path:
        """Return this worker's durable directory."""
        return self.run_root / "workers" / f"gpu{self.worker_index}"


__all__ = ("GemmaFleetConfig",)
