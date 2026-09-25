"""The item by arm by seed row count a complete run produces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpectedRows:
    """The expected item, arm, and seed counts of one run."""

    items: int
    arms: int
    seeds: int

    @property
    def total(self) -> int:
        """Return the total result-row count."""
        return self.items * self.arms * self.seeds

    def to_dict(self) -> dict[str, int]:
        """Return the three factors and their product."""
        return {"items": self.items, "arms": self.arms, "seeds": self.seeds, "total": self.total}


__all__ = ("ExpectedRows",)
