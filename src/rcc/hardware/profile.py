"""The fixed hardware class a run is planned against, so its timings are attributable."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareProfile:
    """One fixed hardware class."""

    profile_id: str
    instance_type: str
    accelerators: int
    accelerator_name: str
    accelerator_memory_gib: int

    @property
    def hardware_identity_hash(self) -> str:
        """Return this hardware's identity string."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Return this hardware as JSON-compatible fields."""
        return {
            "profile_id": self.profile_id,
            "instance_type": self.instance_type,
            "accelerators": self.accelerators,
            "accelerator_name": self.accelerator_name,
            "accelerator_memory_gib": self.accelerator_memory_gib,
        }


P5_8XH100 = HardwareProfile(
    profile_id="aws-p5.48xlarge-8xh100-80gb-v1",
    instance_type="p5.48xlarge",
    accelerators=8,
    accelerator_name="NVIDIA H100",
    accelerator_memory_gib=80,
)

__all__ = ("P5_8XH100", "HardwareProfile")
