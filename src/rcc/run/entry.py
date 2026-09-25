"""Run one benchmark config end to end on one node.

Resolve the config, fetch the pinned tokenizer, build the panel, drive every arm
through the split-fleet driver, and publish the report. See docs/running.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rcc.run.checkpoints import prefetch_checkpoints
from rcc.run.io import atomic_json
from rcc.run.plan import ROUTE_MODEL_IDS, ResolvedPlan, load_and_resolve, route_family
from rcc.topologies.chain import CHAIN_TOPOLOGY_KEY

GEMMA_MODEL_ID = "gemma4-12b-it"
MINISTRAL_MODEL_ID = "ministral3-14b"


def _source_commit() -> str:
    from rcc.run.split_driver import _source_commit as driver_commit

    return driver_commit()


def _attempt_id() -> str:
    return time.strftime("a%Y%m%d%H%M%S", time.gmtime())


def _fanout_environment(plan: ResolvedPlan, lane: str) -> None:
    os.environ["RCC_FANOUT_FAMILY"] = lane
    os.environ["RCC_FANOUT_BENCHMARK"] = plan.benchmark.benchmark_key
    os.environ["RCC_FANOUT_ITEM_START"] = str(plan.item_start)
    os.environ["RCC_FANOUT_ITEM_COUNT"] = str(plan.item_count)
    os.environ["RCC_FANOUT_PLAN_FINGERPRINT"] = plan.execution_identity_hash
    # The one ablation word, published only when the plan names it. The split
    # driver and its seats inherit it from this process.
    os.environ.pop("RCC_FANOUT_JUDGER_QUESTION", None)
    if plan.benchmark.judger_question is not None:
        os.environ["RCC_FANOUT_JUDGER_QUESTION"] = plan.benchmark.judger_question


def _prepare_route_panel(
    plan: ResolvedPlan, run_root: Path, bundle: Path, *, source_commit: str
) -> None:
    family = route_family(plan.model.model_id)
    if plan.benchmark.topology_key == CHAIN_TOPOLOGY_KEY:
        from rcc.benchmarks.longbench_v2.build_cli import main as longbench_build

        status = longbench_build(
            [
                "prepare",
                "--profile",
                plan.benchmark.benchmark_key,
                "--bundle",
                str(bundle),
                "--run-root",
                str(run_root),
                "--source-commit",
                source_commit,
            ]
        )
        if status != 0:
            raise RuntimeError("the LongBench prepare step failed")
        return
    from rcc.benchmarks.fanoutqa.source_build import audit_source_panel, prepare_panel

    for build in (audit_source_panel, prepare_panel):
        build(
            run_root,
            source_commit=source_commit,
            profile=plan.benchmark,
            family=family,
            natural_bundle=bundle,
        )


def _write_placements(plan: ResolvedPlan, run_root: Path) -> None:
    """Record the splits this run seats, for offline inspection."""
    from rcc.hardware.qwen_manifest import placements_record

    atomic_json(
        run_root / "control" / "placements.json",
        placements_record(
            route_family(plan.model.model_id), plan.benchmark, plan.execution_identity_hash
        ),
    )


def _drive(
    plan: ResolvedPlan,
    *,
    lane: str,
    results_root: Path,
    attempt_id: str,
    arms: Sequence[str],
    extra: Sequence[str],
    report: Path,
) -> dict[str, Any]:
    from rcc.run.split_driver import main as split_driver

    common = [
        "--family",
        lane,
        "--results-root",
        str(results_root),
        "--plan-fingerprint",
        plan.execution_identity_hash,
        "--item-offset",
        str(plan.item_start),
        "--item-count",
        str(plan.item_count),
        "--attempt-id",
        attempt_id,
        "--benchmark",
        plan.benchmark.benchmark_key,
        *extra,
    ]
    for arm in arms:
        common += ["--arm", arm]
    if split_driver(["run", *common]) != 0:
        raise RuntimeError("the split-fleet run failed")
    if split_driver(["report", *common, "--report", str(report)]) != 0:
        raise RuntimeError("the split-fleet report failed")
    return json.loads(report.read_text(encoding="utf-8"))


def registered_checkpoints(model_id: str) -> tuple[tuple[str, str], ...]:
    """Return the pinned repositories and revisions one family's arms load."""
    if model_id in ROUTE_MODEL_IDS:
        return route_family(model_id).profile.runtime.model_revisions
    if model_id == GEMMA_MODEL_ID:
        from rcc.models.gemma import GEMMA

        return GEMMA.runtime.model_revisions
    if model_id == MINISTRAL_MODEL_ID:
        from rcc.models.ministral import MINISTRAL

        return MINISTRAL.runtime.model_revisions
    raise ValueError(f"model {model_id!r} has no registered runner")


def run(
    config: Path,
    bundle: Path,
    *,
    arms: Sequence[str] = (),
    attempt_id: str | None = None,
) -> dict[str, Any]:
    """Run every step of one config and return the published report."""
    plan = load_and_resolve(config)
    results_root = Path(plan.output_uri).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    # Every seat compiles into its own cache; eight seats sharing one corrupt it.
    os.environ.setdefault("VLLM_CACHE_ROOT", str(results_root / "vllm_cache"))
    source_commit = _source_commit()
    attempt = attempt_id or _attempt_id()
    report = results_root / "report.json"
    model_id = plan.model.model_id
    prefetch_checkpoints(registered_checkpoints(model_id))
    if model_id in ROUTE_MODEL_IDS:
        family = route_family(model_id)
        if family.lane == "nemotron":
            from rcc.models.nemotron.kernels import KERNEL_PINS, KERNEL_VARIANT

            prefetch_checkpoints(
                KERNEL_PINS, allow_patterns=(f"build/{KERNEL_VARIANT}/*",), repo_type="kernel"
            )
        _fanout_environment(plan, family.lane)
        _prepare_route_panel(plan, results_root, bundle, source_commit=source_commit)
        _write_placements(plan, results_root)
        return _drive(
            plan,
            lane=family.lane,
            results_root=results_root,
            attempt_id=attempt,
            arms=arms,
            extra=(),
            report=report,
        )
    if model_id == GEMMA_MODEL_ID:
        from rcc.run.gemma.arm_subsets import EXECUTION_PROFILES

        os.environ["RCC_GEMMA_M3_BENCHMARK_PROFILE"] = plan.benchmark.profile_id
        if plan.arm_profile in EXECUTION_PROFILES:
            os.environ["RCC_GEMMA_M3_ARM_PROFILE"] = plan.arm_profile
        return _drive(
            plan,
            lane="gemma",
            results_root=results_root,
            attempt_id=attempt,
            arms=arms,
            extra=("--source-cache", str(bundle)),
            report=report,
        )
    if model_id == MINISTRAL_MODEL_ID:
        from rcc.run.ministral.prepare import prepare_panel as prepare_ministral

        os.environ["RCC_MINISTRAL_M3_BENCHMARK_PROFILE"] = plan.benchmark.profile_id
        prepared = prepare_ministral(
            results_root,
            bundle,
            profile=plan.benchmark,
            item_offset=plan.item_start,
            item_count=plan.item_count,
        )
        return _drive(
            plan,
            lane="ministral",
            results_root=results_root,
            attempt_id=attempt,
            arms=arms,
            extra=("--source-bundle", str(bundle), "--prepared-panel", str(prepared["path"])),
            report=report,
        )
    raise ValueError(f"model {model_id!r} has no registered runner")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one config on this node and print the published report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--arm", action="append", default=[], help="run only these arms")
    parser.add_argument("--attempt-id")
    args = parser.parse_args(argv)
    report = run(args.config, args.bundle, arms=args.arm, attempt_id=args.attempt_id)
    sys.stdout.write(json.dumps(report, sort_keys=True, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        sys.stderr.write(f"FATAL: {exc}\n")
        raise SystemExit(2) from exc
