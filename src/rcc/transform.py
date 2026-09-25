"""Transform ABC: kind, meters, axis, apply, and the | composition operator.

A pure function over a cache, tagged with what it may change: `kind` gives the
family, `meters` what it may move, `axis` the cache axis it acts on or None.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

from rcc.cache import Axis, KVCache
from rcc.meters import Meter

if TYPE_CHECKING:
    from rcc.pipeline import Pipeline


class Kind(Enum):
    """The four transform families."""

    ARTIFACT = "artifact"
    WIRE = "wire"
    ACCESS = "access"
    WRAPPER = "wrapper"


class Transform(ABC):
    """A pure cache transform that declares what it moves; compose with `|`."""

    kind: ClassVar[Kind]
    meters: ClassVar[frozenset[Meter]]
    axis: ClassVar[Axis | None] = None

    @abstractmethod
    def apply(self, cache: KVCache) -> KVCache:
        """Return a new cache, leaving the input unchanged."""

    def __or__(self, other: Transform) -> Pipeline:
        """Compose left to right; two transforms on one axis warn."""
        from rcc.pipeline import Pipeline

        return Pipeline((self, other))
