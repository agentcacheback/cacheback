"""The registered eight-GPU placement table."""

from __future__ import annotations

import pytest

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.hardware.fleet import FleetPlacement
from rcc.hardware.placements import (
    CHAIN_ARM_PLACEMENTS,
    LATENT_RATIOS,
    SEMANTIC_ARM_PLACEMENTS,
    SPLIT_FLEET_WORKERS,
    bind_policy_placements,
    placement_for_semantic_arm,
    placement_table,
)

PINNED: dict[str, FleetPlacement] = {
    "issue_only": FleetPlacement(8, 0, 8),
    "full": FleetPlacement(8, 4, 4),
    "text_primary": FleetPlacement(8, 0, 8, fused=True),
    "text_medium": FleetPlacement(8, 6, 2),
    "text_small": FleetPlacement(8, 6, 2),
    "latent_query_support_r2": FleetPlacement(8, 4, 4),
    "latent_query_support_r4": FleetPlacement(8, 5, 3),
    "latent_query_support_r8": FleetPlacement(8, 5, 3),
    "latent_query_support_r16": FleetPlacement(8, 5, 3),
    "latent_query_support_r32": FleetPlacement(8, 5, 3),
    "latent_query_support_r64": FleetPlacement(8, 5, 3),
    "latent_query_support_r128": FleetPlacement(8, 5, 3),
}


def test_every_registered_split_is_pinned_value_by_value() -> None:
    assert SPLIT_FLEET_WORKERS == 8
    assert len(PINNED) == 12
    assert SEMANTIC_ARM_PLACEMENTS == PINNED
    assert {
        arm: placement_for_semantic_arm(arm, table=SEMANTIC_ARM_PLACEMENTS) for arm in PINNED
    } == PINNED
    # The chain table is the second registered split and is pinned here too.
    # Every latent row of both laws is placed at 4P+4R, because under 6:2 the
    # producers idle two thirds of the time.
    chain_latent = {
        arm: placement
        for arm, placement in CHAIN_ARM_PLACEMENTS.items()
        if arm.startswith("latent_")
    }
    assert len(CHAIN_ARM_PLACEMENTS) == 16 and len(chain_latent) == 12
    assert set(chain_latent.values()) == {FleetPlacement(8, 4, 4)}
    assert {arm for arm in chain_latent if "rerank" in arm} == {
        f"latent_query_support_rerank_r{ratio}" for ratio in (4, 8, 16, 32, 64, 128)
    }
    assert {arm for arm in chain_latent if "bounded" in arm} == {
        f"latent_query_support_bounded_b{budget}"
        for budget in (65_536, 32_768, 16_384, 8_192, 4_096, 2_048)
    }
    assert set(CHAIN_ARM_PLACEMENTS) - set(chain_latent) == {
        "issue_only",
        "text_primary",
        "text_medium",
        "text_small",
    }


def test_the_latent_ladder_matches_the_sealed_panel_ratios() -> None:
    """Every rung of the placement ladder is a sealed panel ratio, and the reverse."""
    for ratio in LATENT_RATIOS:
        placement_for_semantic_arm(
            f"latent_query_support_r{ratio}",
            table=placement_table(FANOUTQA_NATURAL_DEV50.topology_key),
        )
    assert FANOUTQA_NATURAL_DEV50.ratios == LATENT_RATIOS


def test_a_policy_binding_resolves_through_the_shared_table_or_refuses() -> None:
    bound = bind_policy_placements(
        {"qwen3_8b_r2_w16_support": "latent_query_support_r2", "issue_only": "issue_only"},
        table=SEMANTIC_ARM_PLACEMENTS,
    )
    assert bound == {
        "qwen3_8b_r2_w16_support": FleetPlacement(8, 4, 4),
        "issue_only": FleetPlacement(8, 0, 8),
    }
    with pytest.raises(ValueError, match="unregistered semantic arm"):
        placement_for_semantic_arm("latent_query_support_r3", table=SEMANTIC_ARM_PLACEMENTS)
    with pytest.raises(ValueError, match="unregistered semantic arm"):
        bind_policy_placements(
            {"qwen3_8b_r3_w16_support": "latent_query_support_r3"}, table=SEMANTIC_ARM_PLACEMENTS
        )
