"""Mean query attention evidence alignment from a capture-only selector-dev run.

Ranks each located gold token's mean query attention score among all rolled positions, averages
per task, and reports the median of those task means per model. Takes a RUN_ROOT.
The reported set is the recall table's: the development tasks whose required
evidence is fully located on the bound page.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import median

import torch

from experiments.selector_dev_gold import gold_metrics, merged_gold_spans

#: Where the tables are written; created on first use.
ROOT = Path(__file__).resolve().parents[1] / "output" / "selector-dev"
MODELS = ("qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-32b")


def task_mean_percentile(scores: torch.Tensor, spans: tuple[tuple[int, int], ...]) -> float:
    """Return the mean percentile rank of the gold tokens under one score vector."""
    ranks = torch.argsort(torch.argsort(scores.double(), stable=True), stable=True)
    percentile = ranks.double() / max(int(scores.numel()) - 1, 1)
    picked = torch.cat([percentile[start:stop] for start, stop in spans])
    return float(picked.mean())


def main(run_root: Path) -> None:
    """Write percentile, raw-score, and prompt-normalized medians per model."""
    ROOT.mkdir(parents=True, exist_ok=True)
    plan = json.loads((run_root / "plan.json").read_text())
    spans_by_qid: dict[str, tuple[tuple[int, int], ...]] = {}
    prompt_lengths: dict[str, int] = {}
    for item in plan["items"]:
        entry = json.loads((run_root / "inputs" / f"{item['qid']}.json").read_text())
        rows = entry["gold_leaf_token_spans"]
        # Eligibility is a property of the input, not of any score vector: the
        # same tasks the recall table reports.
        full = tuple(range(len(entry["prompt_token_ids"]) + 40))
        if gold_metrics(full, rows)["gold_mechanism_identified"]:
            spans_by_qid[item["qid"]] = merged_gold_spans(rows)
            prompt_lengths[item["qid"]] = len(entry["prompt_token_ids"])
    with (ROOT / "attention.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["model", "median_percentile", "items", "median_raw_score", "median_prompt_ratio"]
        )
        for model in MODELS:
            means: list[float] = []
            raw_means: list[float] = []
            prompt_ratios: list[float] = []
            for qid, spans in spans_by_qid.items():
                scores = torch.load(
                    run_root / "models" / model / qid / "scores.pt",
                    map_location="cpu",
                    weights_only=True,
                )["snap"]
                means.append(task_mean_percentile(scores, spans))
                picked = torch.cat([scores[start:stop].double() for start, stop in spans])
                assert bool(torch.isfinite(picked).all()) and bool((picked >= 0).all())
                raw_means.append(float(picked.mean()))
                prompt_length = prompt_lengths[qid]
                assert scores.numel() == prompt_length + 40
                assert all(1 <= start < stop <= prompt_length for start, stop in spans)
                # Position zero is pinned to the maximum, and the latent rows
                # are not prompt tokens.
                prompt_mean = float(scores[1:prompt_length].double().mean())
                assert prompt_mean > 0
                prompt_ratios.append(raw_means[-1] / prompt_mean)
            writer.writerow(
                [model, median(means), len(means), median(raw_means), median(prompt_ratios)]
            )
            print(model, f"median {median(means):.4f} over {len(means)} tasks")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
