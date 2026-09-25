"""Gold-evidence recall tables from a capture-only selector-dev run root.

Scores every recorded `keeps.json` against its input's located gold spans and
writes the figure package's recall files. Takes a RUN_ROOT. Nothing is decoded.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from experiments.selector_dev_gold import gold_metrics

#: Where the tables are written; created on first use.
ROOT = Path(__file__).resolve().parents[1] / "output" / "selector-dev"
MODELS = ("qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-32b")
RATIOS = (2, 4, 8, 16, 32, 64, 128)
PAPER_SELECTORS = ("snap", "support-p2-a2", "chunkkv", "h2o", "kvzip", "streaming")
LABELS = {
    "snap": "Mean query attention",
    "support-p2-a2": "CacheBack (ours)",
    "chunkkv": "ChunkKV",
    "h2o": "H2O",
    "kvzip": "KVzip",
    "streaming": "Streaming",
}


Cell = tuple[str, str, int]


def score_run(run_root: Path) -> tuple[dict[Cell, list[dict[str, Any]]], set[str]]:
    """Score every recorded keep set of the gold-located tasks, returning cells and task ids."""
    plan = json.loads((run_root / "plan.json").read_text())
    # Eligibility belongs to the input rather than to any keep set: a task
    # counts only when every gold leaf is located under the full keep.
    spans: dict[str, list[dict[str, Any]]] = {}
    for item in plan["items"]:
        qid = item["qid"]
        entry = json.loads((run_root / "inputs" / f"{qid}.json").read_text())
        rows = entry["gold_leaf_token_spans"]
        full = tuple(range(len(entry["prompt_token_ids"]) + 40))
        if gold_metrics(full, rows)["gold_mechanism_identified"]:
            spans[qid] = rows
    per_cell: dict[Cell, list[dict[str, Any]]] = defaultdict(list)
    for model in MODELS:
        for qid, rows in spans.items():
            cells = json.loads((run_root / "models" / model / qid / "keeps.json").read_text())
            for cell in cells:
                if cell["ratio"] != 1:
                    metrics = gold_metrics(tuple(cell["keep"]), rows)
                    per_cell[model, cell["selector"], int(cell["ratio"])].append(metrics)
    return per_cell, set(spans)


def write_csv(
    path: Path, per_cell: dict[Cell, list[dict[str, Any]]], selectors: tuple[str, ...], items: int
) -> None:
    """Write group and leaf recall per model, selector, and ratio."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "selector", "ratio", "items", "group_recall", "leaf_recall"])
        for model in MODELS:
            for selector in selectors:
                for ratio in RATIOS:
                    metrics = per_cell[model, selector, ratio]
                    assert len(metrics) == items, (model, selector, ratio)
                    writer.writerow(
                        [
                            model,
                            selector,
                            ratio,
                            len(metrics),
                            mean(m["official_evidence_group_recall"] for m in metrics),
                            mean(m["official_evidence_leaf_recall"] for m in metrics),
                        ]
                    )


def tabular(header: str, body: list[str], comment: list[str]) -> str:
    """Wrap rows in a booktabs tabular with a provenance comment."""
    lines = [
        *comment,
        "\\begin{tabular}{l" + "r" * len(RATIOS) + "}",
        "\\toprule",
        header + " & " + " & ".join(f"{r}$\\times$" for r in RATIOS) + " \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def main(run_root: Path) -> None:
    """Write recall.csv, recall-table.csv, recall-table.tex and the support grid."""
    ROOT.mkdir(parents=True, exist_ok=True)
    per_cell, identified = score_run(run_root)
    selectors = tuple(sorted({key[1] for key in per_cell}))
    write_csv(ROOT / "recall-table.csv", per_cell, selectors, len(identified))
    write_csv(ROOT / "recall.csv", per_cell, ("snap", "support-p2-a2"), len(identified))

    def pooled(selector: str, ratio: int) -> float:
        return mean(
            m["official_evidence_group_recall"]
            for model in MODELS
            for m in per_cell[model, selector, ratio]
        )

    best = {r: max(pooled(s, r) for s in PAPER_SELECTORS) for r in RATIOS}
    body = []
    for selector in PAPER_SELECTORS:
        cells = []
        for ratio in RATIOS:
            value = pooled(selector, ratio)
            text = f"{100 * value:.0f}"
            cells.append(f"\\textbf{{{text}}}" if value == best[ratio] else text)
        body.append(LABELS[selector] + " & " + " & ".join(cells) + " \\\\")
    comment = [
        "% Evidence recall (percent), mean over four Qwen3 sizes on the "
        f"{len(identified)} development tasks with located gold.",
    ]
    (ROOT / "recall-table.tex").write_text(tabular("Selector", body, comment), encoding="utf-8")

    grid = [s for s in selectors if s.startswith("support-")]
    with (ROOT / "support-grid-recall.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["selector", *[f"r{r}" for r in RATIOS]])
        for selector in ("snap", *grid):
            writer.writerow([selector, *[f"{pooled(selector, r):.4f}" for r in RATIOS]])
    body = [
        "Mean query attention & "
        + " & ".join(f"{100 * pooled('snap', r):.0f}" for r in RATIOS)
        + " \\\\"
    ]
    for selector in grid:
        order, alpha = selector.removeprefix("support-p").split("-a")
        order_text = order.replace("p", ".").replace("inf", "\\infty")
        label = f"({order_text}, {alpha.replace('p', '.')})"
        body.append(
            f"${label}$ & "
            + " & ".join(f"{100 * pooled(selector, r):.0f}" for r in RATIOS)
            + " \\\\"
        )
    comment = ["% Query-support grid: four-model mean evidence recall (percent)."]
    (ROOT / "support-grid-recall.tex").write_text(
        tabular("$(p, \\alpha)$", body, comment), encoding="utf-8"
    )
    print(f"{len(identified)} identified tasks; {len(selectors)} selectors; tables written")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
