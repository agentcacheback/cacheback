"""The fleet runtime's worker contract, independent of model and benchmark.

A seat is a producer, a receiver, or a fused adapter that produces what it then
serves. A producer hands over an artifact; a receiver returns one completion.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class WorkItem:
    """One benchmark item, in submission order."""

    qid: str
    order_index: int

    def __post_init__(self) -> None:
        """Validate the item identity and its submission order."""
        if not self.qid or self.order_index < 0:
            raise ValueError("a work item needs a qid and nonnegative order")


@dataclass(frozen=True)
class ProducedArtifact:
    """A manifest a producer published; the runtime hashes it before handoff."""

    path: Path
    producer_finished_at: float
    fields: dict[str, Any] = field(default_factory=dict[str, Any])
    after_publish: Callable[[], None] | None = None


@dataclass(frozen=True)
class ReceiverCompletion:
    """One bankable receiver result and its service-clock stamps."""

    qid: str
    result: dict[str, Any]
    admitted_at: float
    finished_at: float
    first_token_at: float | None = None
    fields: dict[str, Any] = field(default_factory=dict[str, Any])


class SplitProducer(Protocol):
    """A warm producer engine held by one worker process."""

    def warm(self) -> None:
        """Warm the persistent producer engine."""
        ...

    def produce(self, item: WorkItem) -> ProducedArtifact:
        """Produce one item's handoff artifact."""
        ...

    def close(self) -> None:
        """Close the producer engine."""
        ...


class AsyncReceiver(Protocol):
    """A warm receiver that can overlap several items on one engine."""

    def warm(self) -> None:
        """Warm the persistent receiver engine."""
        ...

    def can_accept(self) -> bool:
        """Return whether another item may be submitted."""
        ...

    def submit(self, item: WorkItem, artifact: ProducedArtifact | None) -> None:
        """Submit one item and its optional producer artifact."""
        ...

    def pump(self) -> Sequence[ReceiverCompletion]:
        """Step decoding and return the items that just completed."""
        ...

    def idle(self) -> bool:
        """Return whether the receiver has no pending work."""
        ...

    def close(self) -> None:
        """Close the receiver engine."""
        ...


class FusedAdapter(AsyncReceiver, Protocol):
    """A same-model seat whose ``submit`` produces before it serves."""

    def stage_times(self, qid: str) -> dict[str, float]:
        """Return the collapsed producer and handoff stage times of one item."""
        ...
