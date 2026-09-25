"""The request-keyed registry that carries staged caches to the worker connector.

A caller stages a cache under the engine request id before `add_request`. A load
does not consume the entry; it drops at the engine's request-finished signal.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StagedCache:
    """One staged payload: K/V `[layers, kv_heads, k, head_dim]` plus positions.

    Wire arrivals stage host tensors, in-process extractions stay on the pages'
    device, and a consumer moves an entry only when it is not already co-resident.
    """

    keys: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor

    def __post_init__(self) -> None:
        """Raise when the three tensors do not agree on shape."""
        if self.keys.ndim != 4:
            raise ValueError(f"keys must be [layers, kv_heads, k, head_dim], got {self.keys.shape}")
        if self.keys.shape != self.values.shape:
            raise ValueError(f"keys {self.keys.shape} and values {self.values.shape} disagree")
        if self.positions.ndim != 1 or self.positions.shape[0] != self.keys.shape[2]:
            raise ValueError(
                f"positions must be [k={self.keys.shape[2]}], got {self.positions.shape}"
            )


class StagingError(RuntimeError):
    """A staging protocol violation, for example a live duplicate request id."""


class StagingMissError(StagingError):
    """A load found no entry, so decoding would read unwritten pages."""


class StagingRegistry:
    """The pending map from engine request id to its staged cache."""

    def __init__(self) -> None:
        """Start empty, with no recorded load errors."""
        self._entries: dict[str, StagedCache] = {}
        self._load_errors: set[int] = set()

    def __len__(self) -> int:
        """Return the number of ids currently staged."""
        return len(self._entries)

    def __contains__(self, request_id: str) -> bool:
        """Return whether `request_id` is staged."""
        return request_id in self._entries

    def put(self, request_id: str, entry: StagedCache) -> None:
        """Stage `entry` under a fresh id, raising on a live duplicate id."""
        if request_id in self._entries:
            raise StagingError(f"request id {request_id!r} is already staged; live ids are unique")
        self._entries[request_id] = entry

    def load(self, request_id: str) -> StagedCache:
        """Return the staged entry without consuming it, so a preemption resume refills."""
        entry = self._entries.get(request_id)
        if entry is None:
            raise StagingMissError(
                f"no staged cache for request id {request_id!r}; "
                "refusing to decode from unwritten pages"
            )
        return entry

    def finish(self, request_id: str) -> bool:
        """Drop the entry on the engine's request-finished signal, False if already gone."""
        return self._entries.pop(request_id, None) is not None

    def record_load_error(self, block_ids: Iterable[int]) -> None:
        """Record physical block ids whose page write failed."""
        self._load_errors.update(block_ids)

    def take_load_errors(self) -> set[int]:
        """Drain and return the failed block ids."""
        errors = self._load_errors
        self._load_errors = set()
        return errors


#: The in-process registry the staging caller and the connector share.
PENDING = StagingRegistry()
