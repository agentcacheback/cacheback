"""The producer and receiver splits the route lanes seat their arms from.

The shared table is cut to the benchmark's own arm roster, and its fingerprint is
what a banked row names, so a run cannot claim a split the table does not hold.
"""

from __future__ import annotations

from typing import Any

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.hardware.fleet import FleetPlacement
from rcc.hardware.placements import bind_policy_placements, placement_table
from rcc.models.route import RouteFamily
from rcc.run.io import canonical_sha

#: Schema of the placement record one route run writes under ``control/``.
ROUTE_PLACEMENTS_VERSION = "route-registry-placements-v1"


def registered_policies(family: RouteFamily, profile: BenchmarkProfile) -> tuple[str, ...]:
    """Return the family policies one benchmark's arms bind, in family order."""
    registered = {arm.arm_id for arm in profile.arms}
    policies = family.production_policies
    return tuple(policy for policy in policies if family.semantic_arm(policy) in registered)


def route_semantic_bindings(family: RouteFamily) -> dict[str, str]:
    """Map every route policy name to its semantic arm in the shared table."""
    return {arm.policy: arm.semantic_arm for arm in family.profile.physical_arms}


def route_placements(family: RouteFamily, profile: BenchmarkProfile) -> dict[str, FleetPlacement]:
    """Return the split for every arm one benchmark seats.

    The bindings are cut to the profile's roster first: a lane arm this benchmark does
    not carry has no row, and binding it would raise for the whole table.
    """
    roster = set(registered_policies(family, profile))
    return bind_policy_placements(
        {
            policy: semantic_arm
            for policy, semantic_arm in route_semantic_bindings(family).items()
            if policy in roster
        },
        table=placement_table(profile.topology_key),
    )


def provisional_placements(*, family: RouteFamily) -> dict[str, FleetPlacement]:
    """Return the eight-GPU fan-out placement for every route arm."""
    return route_placements(family, FANOUTQA_NATURAL_DEV50)


def _placement_bodies(family: RouteFamily, profile: BenchmarkProfile) -> dict[str, dict[str, Any]]:
    """Return every resolved placement as the four fields a banked row binds to."""
    return {
        policy: {
            "workers": placement.workers,
            "producers": placement.producers,
            "receivers": placement.receivers,
            "fused": placement.fused,
        }
        for policy, placement in route_placements(family, profile).items()
    }


def placement_fingerprint(family: RouteFamily, profile: BenchmarkProfile) -> str:
    """Sign the splits one benchmark seats, so a banked row names the table it ran."""
    return canonical_sha(_placement_bodies(family, profile))[:16]


def placements_record(
    family: RouteFamily, profile: BenchmarkProfile, plan_fingerprint: str
) -> dict[str, Any]:
    """Return the placement record a route run writes beside its control files."""
    return {
        "version": ROUTE_PLACEMENTS_VERSION,
        "plan_fingerprint": plan_fingerprint,
        "placements": _placement_bodies(family, profile),
    }


__all__ = (
    "ROUTE_PLACEMENTS_VERSION",
    "placement_fingerprint",
    "placements_record",
    "provisional_placements",
    "registered_policies",
    "route_placements",
    "route_semantic_bindings",
)
