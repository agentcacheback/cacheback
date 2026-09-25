"""Byte accounting for one cache, and the per-axis compression ratio.

Wire bytes and resident bytes are the two meters computable from a cache, as
counted sizes rather than measured RSS. `Pipeline` checks them after each step.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING

from rcc.cache import Axis

if TYPE_CHECKING:
    from rcc.cache import KVCache


class Meter(Enum):
    """The four quantities a transform may declare that it moves."""

    WIRE_BYTES = "wire_bytes"
    RESIDENT_BYTES = "resident_bytes"
    TTFT = "ttft"
    ACCURACY = "accuracy"


#: The meters computable from a cache alone, which `Pipeline` checks each step.
COMPUTABLE_METERS: frozenset[Meter] = frozenset({Meter.WIRE_BYTES, Meter.RESIDENT_BYTES})


@dataclass(frozen=True)
class Meters:
    """The byte counts of one cache state."""

    wire_bytes: int
    resident_bytes: int


def measure(cache: KVCache) -> Meters:
    """Return the two cache-computable meters.

    `wire_bytes` is the K/V payload plus the position map, everything that crosses the network;
    `resident_bytes` is the payload alone, since the RoPE relocation on arrival consumes them.
    """
    payload = (cache.keys.numel() + cache.values.numel()) * cache.keys.element_size()
    positions = cache.positions.numel() * cache.positions.element_size()
    return Meters(wire_bytes=payload + positions, resident_bytes=payload)


@dataclass(frozen=True)
class FactoredRatio:
    """Per-axis compression ratios and their product, such as 12x = 3x(length) * 4x(bits)."""

    per_axis: Mapping[Axis, float]

    @property
    def product(self) -> float:
        """Return the product of the per-axis ratios."""
        out = 1.0
        for ratio in self.per_axis.values():
            out *= ratio
        return out

    def __str__(self) -> str:
        """Render as '12x = 3x(length) * 4x(bits)', or '1.0x (identity)' when empty."""
        if not self.per_axis:
            return "1.0x (identity)"
        parts = " * ".join(f"{r:g}x({a.value})" for a, r in self.per_axis.items())
        return f"{self.product:g}x = {parts}"


def factored(before: KVCache, after: KVCache) -> FactoredRatio:
    """Return the per-axis ratios between two cache states, leaving out unchanged axes."""
    per_axis = {
        axis: before.axis_size(axis) / after.axis_size(axis)
        for axis in Axis
        if before.axis_size(axis) != after.axis_size(axis)
    }
    return FactoredRatio(per_axis=MappingProxyType(per_axis))
