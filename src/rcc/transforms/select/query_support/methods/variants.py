"""Two banked selector variants that also score the sliding layers.

``slidetail`` sums the vote of every layer covering a position, ``slidenorm``
takes the mean over them. See docs/selectors.md, Sliding-layer folds.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from rcc.transforms.select.core import kernels
from rcc.transforms.select.query_support.capture_bank_format import BANK_SCOPE_ALL_LAYER
from rcc.transforms.select.query_support.methods.compose import SupportMomentBundle, selector_score

if TYPE_CHECKING:
    from rcc.transforms.select.query_support.capture_bank import LayerBank, LayerBankRow

SLIDING_TAIL_VARIANT = "slidetail"
"""Grammar token for the sum fold over every covering layer."""

SLIDING_NORM_VARIANT = "slidenorm"
"""Grammar token for the mean fold over every covering layer."""

VARIANT_TOKENS = (SLIDING_TAIL_VARIANT, SLIDING_NORM_VARIANT)
"""Every variant token, in declaration order."""

VARIANT_FOLD_RULE = {
    SLIDING_TAIL_VARIANT: (
        "sum fold: every layer whose attention covers a position adds its vote, so a "
        "tail position can collect up to 48 votes against 8 for a deep position"
    ),
    SLIDING_NORM_VARIANT: (
        "mean fold: a position's vote is divided by the NUMBER of layers whose attention "
        "covers it (global layers cover everything, sliding layers cover their real "
        "window), which leaves the support correction untouched and rescales mean query attention "
        "alone; it is the recency-bias control for the sum fold"
    ),
}
"""The fold rule each variant applies, recorded alongside its scores."""

__all__ = [
    "SLIDING_NORM_VARIANT",
    "SLIDING_TAIL_VARIANT",
    "VARIANT_FOLD_RULE",
    "VARIANT_TOKENS",
    "coverage_counts",
    "variant_fold_score",
    "variant_score",
]


def _sliding_rows(rows: Sequence[LayerBankRow]) -> tuple[LayerBankRow, ...]:
    """Return the sliding rows, raising when the bank has none."""
    sliding = tuple(row for row in rows if not row.descriptor.is_global)
    if not sliding:
        raise RuntimeError(
            "selector variant needs sliding rows and the bank has none; capture with "
            "CaptureSpec(collect_layer_bank=True, bank_include_sliding=True) on a hybrid "
            "attention schedule"
        )
    return sliding


def coverage_counts(rows: Sequence[LayerBankRow]) -> torch.Tensor:
    """Return how many banked layers cover each logical memory position.

    A global row covers every position; a sliding row covers the span
    `capture_bank.covered_span` reports. See docs/selectors.md, Sliding-layer folds.
    """
    from rcc.transforms.select.query_support.capture_bank import covered_span

    if not rows:
        raise RuntimeError("coverage needs at least one banked row")
    logical = rows[0].descriptor.logical_memory_length
    counts = torch.zeros(logical, dtype=torch.float32)
    for row in rows:
        start, end = covered_span(row)
        if end > start:
            counts[start:end] += 1.0
    if not bool((counts > 0).all()):
        raise RuntimeError(
            "the banked layers do not cover every memory position, so the mean fold "
            "would divide by zero somewhere; the capture is incomplete"
        )
    return counts


def _summed_bundle(
    rows: Sequence[LayerBankRow],
    *,
    pool_kernel: int,
    support_orders: Sequence[float],
    counts: torch.Tensor | None = None,
) -> SupportMomentBundle:
    """Fold every row into one bundle, as the support composition does.

    `counts` divides the raw per-position sum before `pool_scores`, because a max pool
    and a division do not commute. See docs/selectors.md, Sliding-layer folds.
    """
    if not support_orders:
        raise RuntimeError("layer bank has no finite support moments")
    scale = float(rows[0].heads * max(1, rows[0].query_rows))
    snap = torch.zeros_like(rows[0].snap)
    for row in rows:
        for chunk in row.snap_chunks:
            snap = snap + chunk
    # Finite moments are per-stream sums and a hybrid schedule mixes stream
    # counts per layer, so each row is weighted to one stream population (see
    # `capture_bank.stream_weights`); on a uniform bank every weight is 1.0.
    from rcc.transforms.select.query_support.capture_bank import stream_weights

    weights = stream_weights(rows)
    totals: dict[str, dict[float, torch.Tensor]] = {"e": {}, "eprime": {}, "r": {}}
    for name, field in (("e", "support_e"), ("eprime", "support_eprime"), ("r", "support_r")):
        for order in support_orders:
            stacked = [getattr(row, field).get(order) for row in rows]
            if any(value is None for value in stacked):
                raise RuntimeError(f"layer bank is missing {field} order {order:g}")
            total = stacked[0].clone() if weights[0] == 1.0 else stacked[0] * weights[0]
            for value, weight in zip(stacked[1:], weights[1:], strict=True):
                total = total + (value if weight == 1.0 else value * weight)
            totals[name][order] = total
    einf, rinf = rows[0].support_einf, rows[0].support_rinf
    if einf is None or rinf is None:
        raise RuntimeError("layer bank is missing infinity support moments")
    for row in rows[1:]:
        if row.support_einf is None or row.support_rinf is None:
            raise RuntimeError("layer bank is missing infinity support moments")
        einf = torch.maximum(einf, row.support_einf)
        rinf = torch.maximum(rinf, row.support_rinf)
    raw_snap = snap / scale
    if counts is not None:
        # Before the pool; the two do not commute at a window boundary.
        raw_snap = raw_snap / counts
        totals = {
            name: {order: value / counts for order, value in mapping.items()}
            for name, mapping in totals.items()
        }
        einf, rinf = einf / counts, rinf / counts
    return SupportMomentBundle(
        snap=kernels.pool_scores(raw_snap, pool_kernel),
        e=totals["e"],
        eprime=totals["eprime"],
        r=totals["r"],
        einf=einf,
        rinf=rinf,
    )


def variant_fold_score(
    rows: Sequence[LayerBankRow],
    variant: str,
    selector: str,
    *,
    pool_kernel: int,
    support_orders: Sequence[float],
) -> torch.Tensor:
    """Score one fold over a row set, without the sliding-row requirement."""
    if variant not in VARIANT_TOKENS:
        raise ValueError(
            f"unknown selector variant {variant!r}; registered: {list(VARIANT_TOKENS)}"
        )
    counts = coverage_counts(rows) if variant == SLIDING_NORM_VARIANT else None
    bundle = _summed_bundle(
        rows, pool_kernel=pool_kernel, support_orders=support_orders, counts=counts
    )
    return selector_score(bundle, selector)


def variant_score(bank: LayerBank, variant: str, selector: str) -> torch.Tensor:
    """Score one registered variant from a bank that carries its sliding rows."""
    rows = bank.selected_rows(BANK_SCOPE_ALL_LAYER)
    _sliding_rows(rows)
    return variant_fold_score(
        rows,
        variant,
        selector,
        pool_kernel=bank.pool_kernel,
        support_orders=bank.support_orders,
    )
