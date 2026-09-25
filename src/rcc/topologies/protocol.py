"""How one item is split across producer and receiver seats, before any placement."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class TopologyProfile:
    """One worker and receiver topology, independent of engine placement."""

    topology_id: str
    workers_per_item: int
    receiver_count: int
    worker_assignment: str
    isolation: str

    @property
    def topology_identity_hash(self) -> str:
        """Return this topology's identity string."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Return this topology as JSON-compatible fields."""
        return {
            "topology_id": self.topology_id,
            "workers_per_item": self.workers_per_item,
            "receiver_count": self.receiver_count,
            "worker_assignment": self.worker_assignment,
            "isolation": self.isolation,
        }
