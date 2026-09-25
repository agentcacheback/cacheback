"""Resolve a route policy to its physical arm and its channel.

The arm name carries the channel, the way the placement table reads it, so no
benchmark roster is consulted: the lane seats arms a panel may not name.
"""

from __future__ import annotations

from typing import Literal, cast

from rcc.models.protocol import PhysicalArm
from rcc.models.route import RouteFamily

QwenChannel = Literal["floor", "text", "latent"]
ProducerBackend = Literal["none", "vllm", "engine"]


def physical_arm(policy: str, *, family: RouteFamily) -> PhysicalArm:
    """Return the physical arm one route policy names."""
    try:
        return next(arm for arm in family.profile.physical_arms if arm.policy == policy)
    except StopIteration:
        raise ValueError(f"unregistered {family.lane} policy {policy!r}") from None


def semantic_channel(arm: PhysicalArm) -> QwenChannel:
    """Return one physical arm's channel, read off its semantic arm name."""
    name = arm.semantic_arm
    if name == "issue_only":
        return "floor"
    if name.startswith("text_"):
        return "text"
    if name.startswith("latent_"):
        return "latent"
    raise ValueError(f"unregistered route semantic arm {name!r}")


def producer_backend(channel: QwenChannel) -> ProducerBackend:
    """Return the producer backend a row records for one channel."""
    return cast(ProducerBackend, {"floor": "none", "text": "vllm", "latent": "engine"}[channel])


__all__ = (
    "ProducerBackend",
    "QwenChannel",
    "physical_arm",
    "producer_backend",
    "semantic_channel",
)
