"""Select: length-axis eviction that keeps a scored token subset and relocates it.

Kept keys and values are the model's own per-position KV, gathered with the sink at
0 and RoPE-relocated to a contiguous prefix, so slot i is always rotated to i.
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

import torch

from rcc.cache import Axis, KVCache
from rcc.meters import Meter
from rcc.rope import apply_rope_k, rope_cos_sin, unapply_rope_k
from rcc.transform import Kind, Transform
from rcc.transforms.select.spans import KeepSchedule, sink_within_budget


def select_keep(scores: torch.Tensor, budget: int, sink: tuple[int, ...] = (0,)) -> list[int]:
    """Top-`budget` positions by score with `sink` always kept, returned sorted ascending.

    Raises:
        ValueError: If any score is NaN, which has no rank: torch sorts it above
            +inf, so the cut would spend budget on it.
    """
    n = int(scores.shape[0])
    if scores.dim() > 1:
        # A column-shaped [n, 1] score vector is accepted; a genuinely
        # multi-column tensor is a caller error.
        scores = scores.reshape(-1)
        if int(scores.shape[0]) != n:
            raise ValueError(f"scores must be one vector, got {n} rows of several columns")
    if bool(torch.isnan(scores).any()):
        raise ValueError("scores contain NaN; a rank over NaN is meaningless")
    budget = max(1, min(budget, n))
    sink_keep = [s for s in sink if 0 <= s < n]
    sink_within_budget(set(sink_keep), budget)
    keep: set[int] = set(sink_keep)
    # One ranking op rather than one `float(scores[i])` per position, each of
    # which would be a kernel launch and a device sync on a CUDA score vector.
    # `stable=True` breaks ties by ascending index.
    order = torch.argsort(scores, descending=True, stable=True)
    ranked = [int(p) for p in cast(Any, order).tolist()]
    for p in ranked:
        if len(keep) >= budget:
            break
        keep.add(p)
    if len(keep) > budget:
        droppable = sorted((p for p in keep if p not in sink_keep), key=lambda i: float(scores[i]))
        for p in droppable:
            if len(keep) <= budget:
                break
            keep.discard(p)
    return sorted(keep)


class Select(Transform):
    """Keep a scored subset of the length axis, RoPE-relocated to a contiguous prefix."""

    kind: ClassVar[Kind] = Kind.ARTIFACT
    meters: ClassVar[frozenset[Meter]] = frozenset({Meter.WIRE_BYTES, Meter.RESIDENT_BYTES})
    axis: ClassVar[Axis | None] = Axis.LENGTH

    def __init__(
        self,
        scores: torch.Tensor,
        *,
        ratio: float | None = None,
        budget: int | None = None,
        sink: tuple[int, ...] = (0,),
        schedule: KeepSchedule = select_keep,
    ) -> None:
        """Configure selection by keep `ratio` (length / ratio) or absolute `budget`.

        `schedule` maps `(scores, budget, sink)` to the kept indices; the default is token
        top-k (`select_keep`), and `span_schedule(...)` keeps whole contiguous spans.
        """
        if (ratio is None) == (budget is None):
            raise ValueError("pass exactly one of ratio or budget")
        if ratio is not None and ratio <= 0:
            raise ValueError(f"ratio must be positive, got {ratio}")
        if budget is not None and budget < 1:
            raise ValueError(f"budget must be at least 1, got {budget}")
        self.scores = scores
        self.ratio = ratio
        self.budget = budget
        self.sink = sink
        self.schedule = schedule

    def _budget_for(self, length: int) -> int:
        """Resolve the absolute keep budget for a cache of the given length."""
        if self.budget is not None:
            return self.budget
        assert self.ratio is not None
        return max(1, round(length / self.ratio))

    def apply(self, cache: KVCache) -> KVCache:
        """Gather the kept KV, RoPE-relocate to 0..k-1, keep original indices as provenance."""
        if cache.rope is None:
            raise ValueError("Select needs rope params for relocation; cache.rope is None")
        if int(self.scores.shape[0]) != cache.length:
            raise ValueError(
                f"scores length {int(self.scores.shape[0])} != cache length {cache.length}"
            )
        keep = self.schedule(self.scores, self._budget_for(cache.length), self.sink)
        keep_idx = torch.tensor(keep, dtype=torch.long, device=cache.keys.device)
        new_keys = cache.keys.index_select(2, keep_idx)
        new_values = cache.values.index_select(2, keep_idx)
        provenance = cache.positions.index_select(0, keep_idx)
        # Slot i is always rotated at position i, because from_prefill uses
        # arange and every relocation re-targets a contiguous prefix, so the
        # un-rotation source is the kept slot index and not the provenance map.
        src_positions = keep_idx
        tgt_positions = torch.arange(len(keep), device=cache.keys.device, dtype=torch.long)
        if not torch.equal(src_positions, tgt_positions):
            src_cos, src_sin = rope_cos_sin(cache.rope, src_positions.unsqueeze(0))
            tgt_cos, tgt_sin = rope_cos_sin(cache.rope, tgt_positions.unsqueeze(0))
            new_keys = apply_rope_k(unapply_rope_k(new_keys, src_cos, src_sin), tgt_cos, tgt_sin)
        return KVCache(keys=new_keys, values=new_values, positions=provenance, rope=cache.rope)
