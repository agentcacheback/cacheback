"""How each arm splits one eight-worker node into producer and receiver seats.

Every item is submitted at once and the producer is serial, so latency is set by
the producer queue while the receivers batch. The table is keyed by semantic arm.
"""

from __future__ import annotations

from collections.abc import Mapping

from rcc.hardware.fleet import FleetPlacement
from rcc.topologies.chain import BOUNDED_ROW_BUDGETS

#: Every arm owns the whole node; only the split between roles moves.
SPLIT_FLEET_WORKERS = 8

#: Compression ratios of the latent arms, in ladder order.
LATENT_RATIOS: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128)

#: Producer counts for the Query-Support latent arms, ratio by ratio.
_LATENT_PRODUCERS: tuple[int, ...] = (4, 5, 5, 5, 5, 5, 5)

_TEXT_PLACEMENTS: dict[str, FleetPlacement] = {
    "issue_only": FleetPlacement(SPLIT_FLEET_WORKERS, 0, 8),
    "full": FleetPlacement(SPLIT_FLEET_WORKERS, 4, 4),
    "text_primary": FleetPlacement(SPLIT_FLEET_WORKERS, 0, 8, fused=True),
    "text_medium": FleetPlacement(SPLIT_FLEET_WORKERS, 6, 2),
    "text_small": FleetPlacement(SPLIT_FLEET_WORKERS, 6, 2),
}

_LATENT_PLACEMENTS: dict[str, FleetPlacement] = {
    f"latent_{family}_r{ratio}": FleetPlacement(
        SPLIT_FLEET_WORKERS, producers, SPLIT_FLEET_WORKERS - producers
    )
    for family in ("query_support",)
    for ratio, producers in zip(LATENT_RATIOS, _LATENT_PRODUCERS, strict=True)
}

SEMANTIC_ARM_PLACEMENTS: dict[str, FleetPlacement] = {
    **_TEXT_PLACEMENTS,
    **_LATENT_PLACEMENTS,
}

#: Producer count for the two smaller text writers, whose four hops of note
#: writing dominate the item. There is no `full` row: the uncut chain fits no
#: item on the served window.
_CHAIN_PRODUCERS = 6
#: Producer count for every chain latent arm, under either law: a latent
#: producer's four hops are short, so more producers would idle on the queue.
_CHAIN_LATENT_PRODUCERS = 4
CHAIN_ARM_PLACEMENTS: dict[str, FleetPlacement] = {
    "issue_only": FleetPlacement(SPLIT_FLEET_WORKERS, 0, 8),
    "text_primary": FleetPlacement(SPLIT_FLEET_WORKERS, 0, 8, fused=True),
    "text_medium": FleetPlacement(SPLIT_FLEET_WORKERS, _CHAIN_PRODUCERS, 2),
    "text_small": FleetPlacement(SPLIT_FLEET_WORKERS, _CHAIN_PRODUCERS, 2),
    # Every chain latent arm runs the same four hops on the same seat, so the
    # re-ranked and bounded arms share one split. There is no rerank r2 row and no
    # bounded row above 64K rows: neither sequence fits the served window.
    **{
        f"latent_query_support_rerank_r{ratio}": FleetPlacement(
            SPLIT_FLEET_WORKERS,
            _CHAIN_LATENT_PRODUCERS,
            SPLIT_FLEET_WORKERS - _CHAIN_LATENT_PRODUCERS,
        )
        for ratio in LATENT_RATIOS[1:]
    },
    **{
        f"latent_query_support_bounded_b{budget}": FleetPlacement(
            SPLIT_FLEET_WORKERS,
            _CHAIN_LATENT_PRODUCERS,
            SPLIT_FLEET_WORKERS - _CHAIN_LATENT_PRODUCERS,
        )
        for budget in BOUNDED_ROW_BUDGETS
    },
}

#: One placement table per topology key.
PLACEMENT_TABLES: dict[str, Mapping[str, FleetPlacement]] = {
    "fanout-m3": SEMANTIC_ARM_PLACEMENTS,
    "chain-t4": CHAIN_ARM_PLACEMENTS,
}


def placement_table(topology_key: str) -> Mapping[str, FleetPlacement]:
    """Return one topology's placement table, raising for an unknown key."""
    try:
        return PLACEMENT_TABLES[topology_key]
    except KeyError as exc:
        raise ValueError(f"unregistered placement topology {topology_key!r}") from exc


def placement_for_semantic_arm(
    semantic_arm: str, *, table: Mapping[str, FleetPlacement]
) -> FleetPlacement:
    """Return one semantic arm's placement, raising for an unknown arm."""
    try:
        return table[semantic_arm]
    except KeyError as exc:
        raise ValueError(f"unregistered semantic arm {semantic_arm!r}") from exc


def bind_policy_placements(
    bindings: Mapping[str, str], *, table: Mapping[str, FleetPlacement]
) -> dict[str, FleetPlacement]:
    """Bind one family's policy names onto the semantic arms.

    Every policy resolves through the shared table or the binding raises, so a family
    cannot introduce a placement of its own.
    """
    return {
        policy: placement_for_semantic_arm(semantic_arm, table=table)
        for policy, semantic_arm in bindings.items()
    }


__all__ = (
    "CHAIN_ARM_PLACEMENTS",
    "LATENT_RATIOS",
    "PLACEMENT_TABLES",
    "SEMANTIC_ARM_PLACEMENTS",
    "SPLIT_FLEET_WORKERS",
    "bind_policy_placements",
    "placement_for_semantic_arm",
    "placement_table",
)
