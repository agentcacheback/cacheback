"""The Ministral split-fleet barrier identity and execution roster.

The barrier machinery is shared, in `rcc.run.barrier`. Ministral's own part is
the marker schema, its directory, and the fingerprint signing the arm grid.
"""

from __future__ import annotations

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.models.ministral import MINISTRAL
from rcc.run import io
from rcc.run.barrier import BarrierSpec, arm_phase_roster

MINISTRAL_EXECUTION_ROSTER_SCHEMA = "ministral-shared-execution-roster-v1"
SPLIT_PHASE_MARKER_SCHEMA = "ministral-split-arm-barrier-marker-v1"
#: The identity of the Ministral arm barrier the split driver publishes under.
#: It carries its own marker directory, so a split barrier marker is never read
#: as another route's, or the reverse.
MINISTRAL_SPLIT_BARRIER = BarrierSpec(
    family_label="Ministral split",
    marker_schema=SPLIT_PHASE_MARKER_SCHEMA,
    barrier_dirname="ministral_split_arm_barriers",
)
_ARMS = tuple(arm.semantic_arm for arm in MINISTRAL.physical_arms)


def execution_roster_fingerprint() -> str:
    """Sign the physical arm, decode, seed, and arm-phase roster."""
    physical = tuple(arm.semantic_arm for arm in MINISTRAL.physical_arms)
    if physical != _ARMS or len(FANOUTQA_NATURAL_DEV50.sample_tags) != 3:
        raise RuntimeError("Ministral execution registration differs from the registered M=3 grid")
    body = {
        "schema": MINISTRAL_EXECUTION_ROSTER_SCHEMA,
        "model_id": MINISTRAL.model_id,
        "decode_profile": MINISTRAL.decode.profile_id,
        "decode_fingerprint": MINISTRAL.decode.identity_hash,
        "sample_tags": list(FANOUTQA_NATURAL_DEV50.sample_tags),
        "arms": [
            {
                "semantic_arm": arm.semantic_arm,
                "policy": arm.policy,
                "sender_checkpoint": arm.sender_checkpoint,
                "sender_revision": arm.sender_revision,
                "selector": arm.selector,
                "payload_layout": arm.payload_layout,
            }
            for arm in MINISTRAL.physical_arms
        ],
        "phases": list(arm_phase_roster(_ARMS)),
    }
    return io.canonical_sha(body)


MINISTRAL_EXECUTION_ROSTER_FINGERPRINT = execution_roster_fingerprint()


__all__ = (
    "MINISTRAL_EXECUTION_ROSTER_FINGERPRINT",
    "MINISTRAL_EXECUTION_ROSTER_SCHEMA",
    "MINISTRAL_SPLIT_BARRIER",
    "SPLIT_PHASE_MARKER_SCHEMA",
    "execution_roster_fingerprint",
)
