"""Import inputs, capture, decode, and report the selector comparison."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from rcc.run.fleet.merge import paired_deltas
from rcc.run.io import atomic_json
from rcc.run.selector_dev.contract import (
    MODELS,
    RATIOS,
    SELECTORS,
    import_inputs,
    load_plan,
    read_json,
)

LOG = logging.getLogger(__name__)


def _source_commit() -> str:
    """Return the git commit of the checkout, or "unknown" outside a repository."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    """Run one named phase of the comparison."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("import", "check", "capture", "decode", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--bundle", type=Path, help="unpacked bundle holding prepared/ and the index"
    )
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument(
        "--source-commit", help="the 40-hex commit to pin when there is no checkout"
    )
    parser.add_argument("--model", choices=tuple(MODELS))
    parser.add_argument("--seat", type=int, default=0)
    args = parser.parse_args()
    if args.phase == "import":
        prepared, index = args.prepared, args.index
        if args.bundle is not None:
            prepared = prepared or args.bundle / "prepared"
            index = index or args.bundle / "fanout-final-dev.json"
        if prepared is None or index is None:
            parser.error("import requires --bundle, or --prepared and --index")
        commit = args.source_commit or _source_commit()
        if commit == "unknown":
            parser.error("no git checkout to read the source commit from; pass --source-commit")
        plan = import_inputs(args.root, prepared, index, commit=commit)
        LOG.info(plan["fingerprint"])
        return
    plan = load_plan(args.root)
    if args.phase == "check":
        LOG.info(
            f"30 inputs; {len(MODELS)} models; {len(SELECTORS)} selectors; {len(RATIOS)} ratios; "
            f"{plan['answer_cells']} answers"
        )
        LOG.info(f"plan={plan['fingerprint']} source_commit={plan['source_commit']}")
        return
    if args.phase in {"capture", "decode"}:
        if args.model is None:
            parser.error("capture/decode requires --model")
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from rcc.run.selector_dev.worker import work

        work(args.root, args.model, args.phase, seat=args.seat)
        return
    report(args.root)


def report(root: Path) -> None:
    """Validate the whole grid and write the question-paired results."""
    plan = load_plan(root)
    from rcc.run.selector_dev.worker import validate_answers

    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for entry in plan["items"]:
            qid = entry["qid"]
            item = read_json(root / "inputs" / f"{qid}.json")
            identity = {
                "plan": plan["fingerprint"],
                "model": model,
                "qid": qid,
                "prompt_sha256": item["prompt_sha256"],
            }
            answers = validate_answers(root / "models" / model / qid, identity)
            rows.extend({"model": model, "qid": qid, **row} for row in answers["rows"])
    if len(rows) != plan["answer_cells"]:
        raise RuntimeError("incomplete answer grid")
    summary: list[dict[str, Any]] = []
    for model, selector, ratio in sorted({(r["model"], r["selector"], r["ratio"]) for r in rows}):
        group = [
            r for r in rows if (r["model"], r["selector"], r["ratio"]) == (model, selector, ratio)
        ]
        summary.append(
            {
                "model": model,
                "selector": selector,
                "ratio": ratio,
                "n": len(group),
                "loose": sum(r["accuracy"]["loose"] for r in group) / len(group),
                "strict": sum(r["accuracy"]["strict"] for r in group) / len(group),
            }
        )
    paired: list[dict[str, Any]] = []
    pairs = [("snap", name) for name in SELECTORS if name != "snap"]
    pairs += [("uncompressed", name) for name in SELECTORS]
    for model in MODELS:
        for ratio in RATIOS:
            group = [
                {"qid": r["qid"], "selector": r["selector"], **r["accuracy"]}
                for r in rows
                if r["model"] == model and r["ratio"] in (1, ratio)
            ]
            comparisons = paired_deltas(group, pairs, key="selector")
            if any(row["n_pairs"] != 30 for row in comparisons):
                raise RuntimeError("comparison is not paired on all 30 questions")
            paired.extend({"model": model, "ratio": ratio, **row} for row in comparisons)
    atomic_json(
        root / "report.json",
        {
            "plan": plan["fingerprint"],
            "summary": summary,
            "paired_comparisons": paired,
            "inference": "descriptive; item SE; no adjusted superiority claim",
        },
    )
    LOG.info(f"validated {len(rows)} answers; report.json written")


if __name__ == "__main__":
    main()
