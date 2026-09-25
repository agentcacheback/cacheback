"""How one arm's GPUs are split between producer and receiver seats on one node."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FleetRole = Literal["producer", "receiver", "fused"]


@dataclass(frozen=True)
class FleetPlacement:
    """One arm's GPU split on a single node."""

    workers: int
    producers: int
    receivers: int
    fused: bool = False

    def __post_init__(self) -> None:
        """Raise unless the roles account for every worker exactly once."""
        if self.workers < 1:
            raise ValueError("a fleet needs at least one worker")
        if self.fused:
            if self.producers or self.receivers != self.workers:
                raise ValueError("fused placement must be 0 producers and all workers fused")
            return
        if self.producers < 0 or self.receivers < 1:
            raise ValueError("split placement needs nonnegative producers and a receiver")
        if self.producers + self.receivers != self.workers:
            raise ValueError("producer and receiver pools must consume every worker")

    def role(self, worker_index: int) -> FleetRole:
        """Return the role one stable worker index holds."""
        if not 0 <= worker_index < self.workers:
            raise ValueError(f"worker {worker_index} is outside 0..{self.workers - 1}")
        if self.fused:
            return "fused"
        return "producer" if worker_index < self.producers else "receiver"


__all__ = ("FleetPlacement", "FleetRole")
