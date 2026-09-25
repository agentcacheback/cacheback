"""Pipeline: transforms applied in order, with each step's meters checked.

Each step's cache-computable meters are compared before and after, and a meter
that moved but was not declared by that transform raises `MeterViolation`.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from dataclasses import asdict

from rcc.cache import Axis, KVCache
from rcc.meters import COMPUTABLE_METERS, Meter, measure
from rcc.transform import Transform


class MeterViolation(RuntimeError):  # noqa: N818  public API name, kept without an Error suffix
    """A transform moved a meter it did not declare."""


class Pipeline:
    """An ordered sequence of transforms applied left to right."""

    def __init__(self, transforms: Sequence[Transform] = ()) -> None:
        """Build a pipeline; warns when two transforms share an axis.

        Two transforms on one axis are a comparison rather than a composition, so the
        constructor warns rather than blocks.
        """
        self._transforms: tuple[Transform, ...] = tuple(transforms)
        seen: set[Axis] = set()
        for t in self._transforms:
            if t.axis is None:
                continue
            if t.axis in seen:
                warnings.warn(
                    f"two transforms act on the same axis ({t.axis.value}); "
                    "same-axis methods are comparisons, not compositions",
                    stacklevel=2,
                )
            seen.add(t.axis)

    @property
    def transforms(self) -> tuple[Transform, ...]:
        """Return the composed transforms, in application order."""
        return self._transforms

    def __or__(self, other: Transform | Pipeline) -> Pipeline:
        """Extend the pipeline on the right."""
        tail = other.transforms if isinstance(other, Pipeline) else (other,)
        return Pipeline(self._transforms + tail)

    def __iter__(self) -> Iterator[Transform]:
        """Iterate the transforms in application order."""
        return iter(self._transforms)

    def __len__(self) -> int:
        """Return the number of composed transforms."""
        return len(self._transforms)

    def apply(self, cache: KVCache) -> KVCache:
        """Run every transform in order, checking each step's meters."""
        current = cache
        for t in self._transforms:
            before = asdict(measure(current))
            current = t.apply(current)
            after = asdict(measure(current))
            moved = {Meter(name) for name in before if before[name] != after[name]}
            undeclared = (moved & COMPUTABLE_METERS) - t.meters
            if undeclared:
                names = ", ".join(sorted(m.value for m in undeclared))
                raise MeterViolation(
                    f"{type(t).__name__} moved undeclared meter(s): {names}. "
                    f"Declared: {sorted(m.value for m in t.meters)}"
                )
        return current
