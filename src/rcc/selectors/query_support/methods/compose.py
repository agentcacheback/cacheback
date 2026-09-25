"""The Query-Support moments, the correction, and the score composed from them.

The score is Comm-Snap times the correction C_p raised to its dose.
The public transport uses the paper's order-2, dose-2 setting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

Order = float | Literal["inf"]

SUPPORT_MOMENT_SCALE = 256.0


def moment_power(values: torch.Tensor, order: float) -> torch.Tensor:
    """Scale one attention base by `SUPPORT_MOMENT_SCALE`, then raise it to `order`."""
    return (values * SUPPORT_MOMENT_SCALE).pow(order)


@dataclass(frozen=True)
class SupportMomentBundle:
    """The stream-pooled moment vectors from one query-aware forward."""

    snap: torch.Tensor
    e: dict[float, torch.Tensor]
    eprime: dict[float, torch.Tensor]
    r: dict[float, torch.Tensor]
    einf: torch.Tensor
    rinf: torch.Tensor

    def __post_init__(self) -> None:
        """Check that every vector shares one shape and device and is finite and nonnegative."""
        orders = set(self.e)
        if not orders or orders != set(self.eprime) or orders != set(self.r):
            raise ValueError("E, Eprime, and R must share at least one moment order")
        if any(order <= 1 or not math.isfinite(order) for order in orders):
            raise ValueError("finite support orders must be finite and greater than one")
        vectors = [self.snap, self.einf, self.rinf]
        vectors.extend(self.e[order] for order in sorted(orders))
        vectors.extend(self.eprime[order] for order in sorted(orders))
        vectors.extend(self.r[order] for order in sorted(orders))
        if any(vector.ndim != 1 for vector in vectors):
            raise ValueError("support moment vectors must be one-dimensional")
        if any(vector.shape != self.snap.shape for vector in vectors):
            raise ValueError("support moment vectors must share one length")
        if any(vector.device != self.snap.device for vector in vectors):
            raise ValueError("support moment vectors must share one device")
        if any(not bool(torch.isfinite(vector).all()) for vector in vectors):
            raise ValueError("support moment vectors must be finite")
        if any(bool((vector < 0).any()) for vector in vectors):
            raise ValueError("support moment vectors must be nonnegative")


def _ratio_power(
    numerator: torch.Tensor, denominator: torch.Tensor, exponent: float
) -> torch.Tensor:
    """Raise a positive ratio to a power, with one as the zero-over-zero limit."""
    numerator = numerator.float()
    denominator = denominator.float()
    tiny = torch.finfo(torch.float32).tiny
    both_zero = (numerator == 0) & (denominator == 0)
    powered = (numerator / denominator.clamp_min(tiny)).clamp_min(tiny).pow(exponent)
    return torch.where(both_zero, torch.ones_like(powered), powered)


def correction(bundle: SupportMomentBundle, order: Order) -> torch.Tensor:
    """Return C_p, including the exact global-max order-infinity endpoint."""
    if order == "inf":
        return _ratio_power(bundle.rinf, bundle.einf, 1.0)
    finite = float(order)
    if finite not in bundle.e:
        raise ValueError(f"moment order {finite:g} is absent from the capture")
    return _ratio_power(bundle.r[finite], bundle.e[finite], 1.0 / (finite - 1.0))


def compose_score(
    bundle: SupportMomentBundle,
    *,
    order: Order,
    alpha: float,
) -> torch.Tensor:
    """Return Comm-Snap times the correction raised to the dose alpha."""
    if alpha < 0 or not math.isfinite(alpha):
        raise ValueError("alpha must be finite and nonnegative")
    if alpha == 0.0:
        return bundle.snap
    return bundle.snap.float() * correction(bundle, order).pow(alpha)


def pin_sink(scores: torch.Tensor, n_sink: int) -> torch.Tensor:
    """Return a copy with the leading `n_sink` columns pinned to the vector maximum."""
    if n_sink < 0:
        raise ValueError(f"n_sink must be nonnegative, got {n_sink}")
    if n_sink == 0 or scores.numel() == 0:
        return scores
    pinned = scores.clone()
    pinned[: min(n_sink, int(scores.numel()))] = scores.max()
    return pinned
