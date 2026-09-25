"""The FanOutQA fan-out topology: three workers per item, one receiver."""

from rcc.topologies.protocol import TopologyProfile

FANOUT_M3 = TopologyProfile(
    topology_id="fanout-m3-v1",
    workers_per_item=3,
    receiver_count=1,
    worker_assignment="whole-item-round-robin-matched-arms",
    isolation="one-item-arm-seed-cell; model-specific physical placement receipt",
)

__all__ = ("FANOUT_M3",)
