"""Protocols, typed state, and aliases shared by the query-support capture paths.

The protocols let the capture code type a cache, a backbone, or a padded batch
without importing the module that builds it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypeAlias

import torch

if TYPE_CHECKING:
    from rcc.selectors.query_support.methods.compose import SupportMomentBundle

__all__ = [
    "AttentionLayerObserver",
    "CaptureResult",
    "CaptureSpec",
    "CaptureStatistics",
    "DecoderBackbone",
    "EnergyAccumulator",
    "EnergyChunkState",
    "KVPair",
    "LayerDescriptor",
    "MutableHFCache",
    "SupportMomentState",
    "TokenIds",
]


def _tensor_map() -> dict[float, torch.Tensor]:
    """Create one typed moment map for a mutable capture state."""
    return {}


KVPair: TypeAlias = tuple[torch.Tensor, torch.Tensor]
"""A per-layer key/value tensor pair from a Hugging Face cache."""

TokenIds: TypeAlias = Sequence[int] | torch.Tensor
"""Token ids accepted by query-location helpers."""


@dataclass(frozen=True)
class CaptureStatistics:
    """The rank, nomination, and pre-pooling vectors from one capture."""

    rank_mean: torch.Tensor
    nomination_fraction: torch.Tensor
    raw_snap: torch.Tensor
    prepool_e2: torch.Tensor | None = None
    prepool_r2: torch.Tensor | None = None


@dataclass(frozen=True)
class CaptureResult:
    """The outputs of one query-support attention capture."""

    snap: torch.Tensor
    found: bool
    energy: torch.Tensor | None = None
    row_energy: torch.Tensor | None = None
    moments: SupportMomentBundle | None = None
    statistics: CaptureStatistics | None = None


@dataclass(frozen=True)
class CaptureSpec:
    """Which sufficient statistics a query-support capture accumulates."""

    collect_snap: bool = True
    energy_pool_kernel: int | None = None
    row_energy_pool_kernel: int | None = None
    support_orders: tuple[float, ...] = ()
    collect_statistics: bool = False


@dataclass(frozen=True)
class LayerDescriptor:
    """Describe one decoder layer's logical and physical attention geometry."""

    layer_idx: int
    layer_type: str
    logical_memory_length: int
    physical_key_length: int
    absolute_key_offset: int
    window: int | None
    query_length: int = 0

    @property
    def physical_memory_length(self) -> int:
        """Return the physical key length excluding the current query rows."""
        return max(0, self.physical_key_length - self.query_length)

    @property
    def is_global(self) -> bool:
        """Return whether this descriptor represents a full-attention layer."""
        return self.layer_type == "global"


class AttentionLayerObserver(Protocol):
    """Receives one layer's per-head memory attention during a capture."""

    def observe_layer(
        self,
        per_head_sum: torch.Tensor,
        *,
        query_rows: int,
        kv_groups: int,
    ) -> None:
        """Consume the per-head memory mass for one captured layer."""
        ...


@dataclass
class EnergyAccumulator:
    """The accumulators one query-support capture folds its layers into."""

    mem_len: int
    energy_pool_kernel: int | None = None
    row_energy_pool_kernel: int | None = None
    energy_sum: torch.Tensor | None = None
    row_energy_sum: torch.Tensor | None = None
    support_orders: tuple[float, ...] = ()
    support_e: dict[float, torch.Tensor] = field(default_factory=_tensor_map)
    support_eprime: dict[float, torch.Tensor] = field(default_factory=_tensor_map)
    support_r: dict[float, torch.Tensor] = field(default_factory=_tensor_map)
    support_einf: torch.Tensor | None = None
    support_rinf: torch.Tensor | None = None
    support_prepool_e2: torch.Tensor | None = None
    support_prepool_r2: torch.Tensor | None = None


@dataclass
class SupportMomentState:
    """The support moments accumulated across one row-chunked layer."""

    pooled_rowsum: torch.Tensor | None = None
    rowmax: torch.Tensor | None = None
    row_moments: dict[float, torch.Tensor] = field(default_factory=_tensor_map)
    prepool_row2: torch.Tensor | None = None


@dataclass
class EnergyChunkState:
    """The per-layer state of a reducer fold that walks the query rows in chunks."""

    grouped_rowsum: torch.Tensor | None = None
    row_sq_sum: torch.Tensor | None = None
    rows: int = 0
    support: SupportMomentState | None = None


class DecoderBackbone(Protocol):
    """The decoder-backbone call the cache-aware forwards make."""

    def __call__(self, **kwargs: object) -> object:
        """Run a decoder forward with keyword arguments."""
        ...


class MutableHFCache(Protocol):
    """The mutable Hugging Face cache surface the capture paths use."""

    def crop(self, max_length: int) -> None:
        """Trim the cache in place to ``max_length`` positions."""
        ...

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append one layer's key/value states and return the updated pair."""
        ...
