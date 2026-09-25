"""The paired bootstrap over whole items, resampled within stratum.

Each resample is drawn inside one source-length band and keeps that band's size,
so no draw rebalances the panel across bands.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any


def nearest_rank(values: Sequence[float], percentile: float) -> float:
    """Return the nearest-rank percentile of a set of draws."""
    if not values:
        raise ValueError("a bootstrap percentile needs at least one draw")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _pools(
    values_by_qid: Mapping[str, float], strata_by_qid: Mapping[str, int]
) -> dict[int, list[float]]:
    """Split the values into one pool per stratum, in sorted item order."""
    if not values_by_qid:
        raise ValueError("a stratified bootstrap needs at least one item")
    pools: dict[int, list[float]] = {}
    for qid in sorted(values_by_qid):
        band = strata_by_qid.get(qid)
        if type(band) is not int:
            raise ValueError(f"{qid}: the stratified bootstrap has no stratum for this item")
        pools.setdefault(band, []).append(float(values_by_qid[qid]))
    return {band: pools[band] for band in sorted(pools)}


def stratified_paired_bootstrap(
    values_by_qid: Mapping[str, float],
    strata_by_qid: Mapping[str, int],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Resample whole items within stratum and return the equal-weight interval.

    Every item carries the same weight, so a draw is the sum over strata divided by the
    panel size; the interval is the 2.5 and 97.5 nearest-rank percentiles of the draws.
    """
    if draws < 1:
        raise ValueError("a stratified bootstrap needs at least one draw")
    pools = _pools(values_by_qid, strata_by_qid)
    total = len(values_by_qid)
    rng = random.Random(seed)
    drawn = [
        sum(sum(pool[rng.randrange(len(pool))] for _ in pool) for pool in pools.values()) / total
        for _ in range(draws)
    ]
    return {
        "mean": sum(sum(pool) for pool in pools.values()) / total,
        "ci_low": nearest_rank(drawn, 0.025),
        "ci_high": nearest_rank(drawn, 0.975),
        "draws": draws,
        "seed": seed,
        "n_by_stratum": {str(band): len(pool) for band, pool in pools.items()},
    }


__all__ = ("nearest_rank", "stratified_paired_bootstrap")
