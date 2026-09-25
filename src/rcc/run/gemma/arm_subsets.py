"""The arm subsets a Gemma run may execute over the natural-dev50 panel."""

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.gemma.roster import shared_execution_roster

#: The published arms and the full-memory control in one fleet, so the whole
#: comparison runs on a single node.
TWELVE_PROFILE = "gemma-natural-dev50-12arm-native-v1"
TWELVE_ARMS = (*shared_execution_roster(FANOUTQA_NATURAL_DEV50).published_arms, "full")
EXECUTION_PROFILES: dict[str, tuple[str, ...]] = {TWELVE_PROFILE: TWELVE_ARMS}


def execution_arm_names(profile: BenchmarkProfile, arm_profile: str = "") -> tuple[str, ...]:
    """Return one named arm subset, or the whole published roster."""
    if not arm_profile:
        return shared_execution_roster(profile).published_arms
    if arm_profile not in EXECUTION_PROFILES or profile != FANOUTQA_NATURAL_DEV50:
        raise ValueError("Gemma execution subsets are registered only for natural-dev50")
    return EXECUTION_PROFILES[arm_profile]
