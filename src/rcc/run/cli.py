"""Resolve a run config into a plan, or read one run's banked rows back."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from rcc.benchmarks.fanoutqa.scoring import DEFAULT_SCORER_VERSION, REFERENCE_SCORERS
from rcc.run.io import atomic_json
from rcc.run.plan import ResolvedPlan, load_and_resolve


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("plan", "rescore-results", "score-results", "handoff-recall")
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--plan-out", required=True, type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--partition-result", action="append", default=[])
    parser.add_argument("--rescore-out", type=Path)
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--tokenizer-snapshot", type=Path)
    parser.add_argument("--runtime-bank", type=Path)
    parser.add_argument("--partition-runtime-bank", action="append", default=[])
    parser.add_argument("--run-root", type=Path)
    parser.add_argument(
        "--references", choices=sorted(REFERENCE_SCORERS), default=DEFAULT_SCORER_VERSION
    )
    parser.add_argument("--out", type=Path)
    return parser


def _emit(payload: Mapping[str, object]) -> int:
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve the plan, write its body out, then run the requested command."""
    args = _parser().parse_args(argv)
    plan: ResolvedPlan = load_and_resolve(args.config)
    atomic_json(args.plan_out, plan.to_dict())
    if args.command == "plan":
        return _emit(plan.to_dict())
    if args.command == "score-results":
        from rcc.run.score_results import run_score_command

        return run_score_command(args, plan)
    if args.command == "handoff-recall":
        from rcc.run.handoff_recall import run_recall_command

        return run_recall_command(args, plan)
    from rcc.run.rescore import run_rescore_command

    return run_rescore_command(args, plan)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        sys.stderr.write(f"FATAL: {exc}\n")
        raise SystemExit(2) from exc
