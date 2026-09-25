"""Score every selector of the roster over one captured latent state.

The engine prefills and rolls once, every selector reads that one cache, and
the cuts are taken afterwards at the same budgets.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from rcc.models.qwen.capture import qwen_w16_keep
from rcc.models.qwen.engine import EngineProducer
from rcc.run.selector_dev.contract import RATIOS, SELECTORS, SUPPORT_GRID, SUPPORT_ORDERS
from rcc.transforms.select.baselines.chunkkv import memory_chunkkv_scores
from rcc.transforms.select.baselines.h2o import memory_h2o_scores
from rcc.transforms.select.baselines.kvzip import memory_kvzip_scores
from rcc.transforms.select.baselines.streaming import streaming_scores
from rcc.transforms.select.query_support.methods.compose import _pin_sink, selector_score
from rcc.transforms.select.query_support.methods.scorers import memory_votes_with_support_moments


class DevProducer(EngineProducer):
    """Reuse the prefill and rollout, scoring every selector before the KV goes."""

    tokenizer: Any

    def capture_scores(
        self,
        past: Any,
        prompt_ids: torch.Tensor,
        judger_ids: torch.Tensor,
        judger_mask: torch.Tensor,
        question_ids: Sequence[int],
        *,
        consume_past: bool = False,
    ) -> tuple[dict[str, torch.Tensor], bool]:
        """Capture the support moments once, then every grid point and comparator."""
        if consume_past:
            raise ValueError("the dev roster reads one cache several times; it cannot be consumed")
        bundle, found = memory_votes_with_support_moments(
            self.model,
            past,
            judger_ids,
            judger_mask,
            question_ids,
            orders=SUPPORT_ORDERS,
            pool_kernel=7,
            n_sink=0,
        )
        scores = {
            "snap": _pin_sink(bundle.snap, 1),
            **{name: _pin_sink(selector_score(bundle, name), 1) for name in SUPPORT_GRID},
            "streaming": streaming_scores(int(bundle.snap.numel())),
        }
        for name, scorer in (
            ("chunkkv", memory_chunkkv_scores),
            ("h2o", memory_h2o_scores),
            ("kvzip", memory_kvzip_scores),
        ):
            scores[name] = scorer(self.model, past, prompt_ids.to(self.device), self.tokenizer)
        return {name: score.detach().float().cpu() for name, score in scores.items()}, bool(found)


def selections(scores: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    """Cut every selector at the same budgets, over its own score vector."""
    if set(scores) != set(SELECTORS):
        raise ValueError("every selector of the roster needs one score vector")
    length = int(scores["snap"].numel())
    if length <= 40 or any(
        score.shape != (length,) or not bool(torch.isfinite(score).all()) or bool((score < 0).any())
        for score in scores.values()
    ):
        raise ValueError("selector vector geometry or values differ")
    rows: list[dict[str, Any]] = [
        {"selector": "uncompressed", "ratio": 1, "keep": list(range(length))}
    ]
    for ratio in RATIOS:
        for name in SELECTORS:
            rows.append(
                {"selector": name, "ratio": ratio, "keep": list(qwen_w16_keep(scores[name], ratio))}
            )
    return rows
