"""The node-side entry point that drives a split-fleet run.

One driver process per node opens the arms in order and spawns one worker
process per GPU seat. The driver process never initializes CUDA.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.run.barrier import BarrierSpec
from rcc.run.contract import FleetPipelineAdapter
from rcc.run.fleet.schedule import SplitScheduleConfig, run_in_processes, run_split_fleet
from rcc.run.io import atomic_json, is_sha256_hex, read_object

COMMANDS = ("run", "report")
#: Every lane. `gemma` and `ministral` have their own adapters; the rest are
#: the families that share the one capture route.
FAMILIES = ("gemma", "ministral", "nemotron", "qwen")
ROUTE_FAMILIES = ("nemotron", "qwen")

#: The panel name every family banks its split rows under.
SPLIT_PANEL = "production"

#: The record the spawned Gemma publication child leaves behind. The driver
#: never opens an engine, so this file is how it learns that the table was
#: published and under which runtime the publishing engine ran.
GEMMA_PREFLIGHT_RECORD = "gemma_shared_embedding_publication.json"
GEMMA_PREFLIGHT_SCHEMA = "gemma-split-run-shared-embedding-publication-v1"

_PreflightHook = Callable[[Path], None]


def run_detached_once(
    target: Callable[..., None], *args: Any, label: str, timeout_s: float
) -> None:
    """Run one GPU-touching duty in a spawned child with a hard timeout."""
    child = multiprocessing.get_context("spawn").Process(target=target, args=args, daemon=False)
    child.start()
    child.join(timeout=timeout_s)
    if child.is_alive():
        child.kill()
        child.join(timeout=30.0)
        raise TimeoutError(f"{label} did not finish in {timeout_s}s")
    if child.exitcode != 0:
        raise RuntimeError(f"{label} exited {child.exitcode}")


def _source_commit() -> str:
    root = Path(__file__).resolve().parents[3]
    try:
        return subprocess.run(
            ("git", "-C", str(root), "rev-parse", "HEAD"),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "0" * 40


def publish_gemma_table_in_child(root: str, publication_id: str) -> None:
    """Publish the run-level Gemma table in a pinned child and record it.

    This is the driver's only GPU-touching work, and it cannot run in the driver
    itself, which goes on to spawn one seat per card.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    from rcc.models.gemma.contract import runtime_fingerprint
    from rcc.run.gemma.split_adapter import gemma_runtime_signature
    from rcc.run.gemma.split_build import publish_run_shared_embedding

    fingerprint = runtime_fingerprint(gemma_runtime_signature())
    published = publish_run_shared_embedding(
        Path(root),
        runtime_fingerprint=fingerprint,
        publication_id=publication_id,
    )
    atomic_json(
        Path(root) / "control" / GEMMA_PREFLIGHT_RECORD,
        {
            "schema": GEMMA_PREFLIGHT_SCHEMA,
            "publication_id": publication_id,
            "runtime_fingerprint": fingerprint,
            "shared_embedding": dict(published),
        },
    )


def _benchmark_profile(args: argparse.Namespace) -> BenchmarkProfile:
    """Return the benchmark this node runs.

    ``--benchmark`` is a benchmark config key, and ``RCC_FANOUT_BENCHMARK`` is
    resolved the same way, so the driver and the seats never disagree.
    """
    from rcc.run.plan import environment_benchmark, registered_benchmark

    if getattr(args, "benchmark", None) is not None:
        return registered_benchmark(args.benchmark)
    named = environment_benchmark()
    if named is None:
        raise RuntimeError(
            "the split driver requires a benchmark: pass --benchmark or set "
            "RCC_FANOUT_BENCHMARK to a registered benchmark key"
        )
    return named


def _require_resident_topology(args: argparse.Namespace, lane: str) -> BenchmarkProfile:
    """Refuse, by name, a topology one resident lane cannot run.

    Only the route lanes take any benchmark. The Gemma and Ministral adapters
    render fan-out shard prompts, so a multi-hop benchmark is refused here.
    """
    from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50

    profile = _benchmark_profile(args)
    if profile.topology_key != FANOUTQA_NATURAL_DEV50.topology_key:
        raise RuntimeError(
            f"the {lane} split driver runs the {FANOUTQA_NATURAL_DEV50.topology_key} topology; "
            f"{profile.benchmark_key} registers {profile.topology_key}"
        )
    return profile


def _gemma_preflight(root: Path, publication_id: str) -> None:
    """Run the table publication in a child and refuse a missing record."""
    run_detached_once(
        publish_gemma_table_in_child,
        str(root),
        publication_id,
        label="the Gemma run-level shared embedding publication",
        timeout_s=1_200.0,
    )
    published = read_object(Path(root) / "control" / GEMMA_PREFLIGHT_RECORD)
    if (
        published.get("schema") != GEMMA_PREFLIGHT_SCHEMA
        or published.get("publication_id") != publication_id
    ):
        raise RuntimeError("the Gemma shared embedding publication does not name this attempt")


def _resident_profile(named: str | None) -> BenchmarkProfile:
    """Return a resident lane's benchmark profile: the named one, else the panel."""
    from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50, profile_by_id

    return profile_by_id(named) if named else FANOUTQA_NATURAL_DEV50


def _gemma_binding(
    args: argparse.Namespace, source_commit: str
) -> tuple[FleetPipelineAdapter, BarrierSpec, str, _PreflightHook]:
    """Bind the Gemma split adapter, barrier, roster, and table publication."""
    _require_resident_topology(args, "Gemma")

    from rcc.models.gemma.config import GemmaFleetConfig
    from rcc.models.gemma.roster import shared_execution_roster
    from rcc.run.gemma.split_adapter import GEMMA_SPLIT_BARRIER, build_adapter

    if args.source_cache is None:
        raise RuntimeError("the Gemma split driver requires --source-cache")
    profile = _resident_profile(os.environ.get("RCC_GEMMA_M3_BENCHMARK_PROFILE"))
    config = GemmaFleetConfig(
        results_base=Path(args.results_root),
        source_cache=Path(args.source_cache),
        item_offset=args.item_offset,
        item_count=args.item_count,
        node_gpus=args.node_gpus,
        # The driver is not a seat, and this index only names a directory.
        # Seat identity comes from the placement.
        worker_index=0,
        attempt_id=args.attempt_id,
        source_commit=source_commit,
        benchmark_profile=profile,
        unified_plan_fingerprint=args.plan_fingerprint,
    )
    publication_id = str(args.attempt_id)

    def preflight(root: Path) -> None:
        _gemma_preflight(root, publication_id)

    roster = shared_execution_roster(config.benchmark_profile)
    roster.require_complete()
    arm_profile = os.environ.get("RCC_GEMMA_M3_ARM_PROFILE", "")
    if arm_profile:
        from rcc.run.gemma.arm_subsets import execution_arm_names

        expected = execution_arm_names(profile, arm_profile)
        if args.arm and tuple(args.arm) != expected:
            raise RuntimeError("Gemma execution subset requires all its arms in registered order")
        adapter = build_adapter(config, arm_profile=arm_profile)
    else:
        adapter = build_adapter(config)
    return adapter, GEMMA_SPLIT_BARRIER, roster.fingerprint, preflight


def _tokenizer_snapshots() -> dict[str, Path]:
    """Return the Ministral sender tokenizer snapshots already on disk."""
    import importlib

    from rcc.models.ministral.text_codec import MINISTRAL_TEXT_CODEC_FILES
    from rcc.models.ministral.text_prompt import registered_text_senders

    hub: Any = importlib.import_module("huggingface_hub")
    return {
        sender.semantic_arm: Path(
            str(
                hub.snapshot_download(
                    repo_id=sender.checkpoint,
                    revision=sender.revision,
                    allow_patterns=list(MINISTRAL_TEXT_CODEC_FILES),
                    local_files_only=True,
                )
            )
        )
        for sender in registered_text_senders()
    }


def _ministral_binding(
    args: argparse.Namespace, source_commit: str
) -> tuple[FleetPipelineAdapter, BarrierSpec, str, _PreflightHook | None]:
    """Bind the Ministral split adapter, barrier, and execution roster."""
    _require_resident_topology(args, "Ministral")

    from rcc.run.ministral.barrier import (
        MINISTRAL_EXECUTION_ROSTER_FINGERPRINT,
        MINISTRAL_SPLIT_BARRIER,
    )
    from rcc.run.ministral.config import MinistralFleetConfig
    from rcc.run.ministral.split_adapter import build_adapter

    if args.source_bundle is None or args.prepared_panel is None:
        raise RuntimeError(
            "the Ministral split driver requires --source-bundle and --prepared-panel"
        )
    profile = _resident_profile(os.environ.get("RCC_MINISTRAL_M3_BENCHMARK_PROFILE"))
    config = MinistralFleetConfig(
        results_base=Path(args.results_root),
        source_bundle=Path(args.source_bundle),
        prepared_panel=Path(args.prepared_panel),
        tokenizer_snapshots=_tokenizer_snapshots(),
        item_offset=args.item_offset,
        item_count=args.item_count,
        node_gpus=args.node_gpus,
        worker_index=0,
        attempt_id=args.attempt_id,
        source_commit=source_commit,
        unified_plan_fingerprint=args.plan_fingerprint,
        benchmark_profile=profile,
    )
    # Ministral needs no run-level publication: every split arm has a producer,
    # and its payloads are per-arm bundles.
    return (
        build_adapter(config),
        MINISTRAL_SPLIT_BARRIER,
        MINISTRAL_EXECUTION_ROSTER_FINGERPRINT,
        None,
    )


def _route_binding(
    args: argparse.Namespace, source_commit: str
) -> tuple[FleetPipelineAdapter, BarrierSpec, str, _PreflightHook | None]:
    """Bind one route family to the shared driver and its splits."""
    del source_commit
    from rcc.run.plan import route_family_for_lane
    from rcc.run.qwen.adapter import (
        build_adapter,
        execution_profile,
        route_split_barrier,
    )

    family = route_family_for_lane(args.family)
    benchmark = _benchmark_profile(args)
    if args.item_offset < 0 or args.item_count < 1:
        raise RuntimeError("item range requires offset >= 0 and count >= 1")
    if args.item_offset + args.item_count > len(benchmark.question_ids):
        raise RuntimeError(
            f"item range exceeds the {len(benchmark.question_ids)}-question "
            f"{benchmark.benchmark_key} panel"
        )
    # The route data adapter reads these process-scoped values and the spawned
    # seats inherit them.
    os.environ["RCC_FANOUT_ITEM_START"] = str(args.item_offset)
    os.environ["RCC_FANOUT_ITEM_COUNT"] = str(args.item_count)
    os.environ["RCC_FANOUT_FAMILY"] = family.lane
    os.environ["RCC_FANOUT_BENCHMARK"] = benchmark.benchmark_key
    # The roster comes off the benchmark's own profile, so a chain node
    # publishes the chain roster rather than one its splits do not match.
    adapter = build_adapter(family, profile=execution_profile())
    return (
        adapter,
        route_split_barrier(family),
        str(adapter.registration()["execution_roster_fingerprint"]),
        None,
    )


def _run_root(adapter: FleetPipelineAdapter, *, fallback: Path | None = None) -> Path:
    """Return the family config's own run root for this invocation."""
    config: Any = getattr(adapter, "config", None)
    root: Any = getattr(config, "run_root", None)
    if root is None:
        root = fallback
    if not isinstance(root, Path):
        raise RuntimeError("the split-fleet adapter does not name a run root")
    return root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument(
        "--benchmark",
        help="the resolver's benchmark config key, when the environment does not name it",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--source-cache", type=Path)
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--prepared-panel", type=Path)
    parser.add_argument("--plan-fingerprint", required=True)
    parser.add_argument("--item-offset", type=int, default=0)
    parser.add_argument("--item-count", type=int, required=True)
    parser.add_argument("--node-gpus", type=int, default=8)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--arm", action="append", default=[])
    parser.add_argument("--max-wall-s", type=float, default=14_400.0)
    parser.add_argument("--join-timeout-s", type=float, default=21_600.0)
    return parser


def build_run(
    args: argparse.Namespace,
) -> tuple[
    FleetPipelineAdapter,
    SplitScheduleConfig,
    _PreflightHook | None,
]:
    """Resolve one command line into an adapter, a schedule, and a hook."""
    if not is_sha256_hex(args.plan_fingerprint):
        raise RuntimeError("--plan-fingerprint must be a full lowercase SHA-256")
    source_commit = _source_commit()
    binding = {
        **dict.fromkeys(ROUTE_FAMILIES, _route_binding),
        "gemma": _gemma_binding,
        "ministral": _ministral_binding,
    }[args.family]
    adapter, barrier, roster_fingerprint, preflight = binding(args, source_commit)
    fallback = Path(args.results_root) if args.family in ROUTE_FAMILIES else None
    schedule = SplitScheduleConfig(
        root=_run_root(adapter, fallback=fallback),
        panel=SPLIT_PANEL,
        source_commit=source_commit,
        attempt_id=args.attempt_id,
        barrier=barrier,
        plan_fingerprint=args.plan_fingerprint,
        roster_fingerprint=roster_fingerprint,
        arms=tuple(args.arm),
        max_wall_s=args.max_wall_s,
        join_timeout_s=args.join_timeout_s,
    )
    return adapter, schedule, preflight


def _drive(
    adapter: FleetPipelineAdapter,
    schedule: SplitScheduleConfig,
    preflight: _PreflightHook | None,
) -> dict[str, Any]:
    return run_split_fleet(
        adapter,
        schedule,
        run_workers=lambda bodies, timeout: run_in_processes(bodies, join_timeout_s=timeout),
        preflight=preflight,
    )


def _report(
    args: argparse.Namespace,
    adapter: FleetPipelineAdapter,
    schedule: SplitScheduleConfig,
) -> dict[str, Any]:
    """Merge this run's per-arm, per-seat banks and publish one report.

    The report is published by the node's one driver for the same reason the arm
    markers are: several seats racing one file would depend on who wrote last.
    """
    if args.report is None:
        raise RuntimeError("the split-fleet report command requires --report")
    reports: Any = adapter.reports
    body: dict[str, Any] = dict(
        reports.build_report(schedule.root, source_commit=schedule.source_commit)
    )
    atomic_json(Path(args.report), body)
    return body


def main(argv: Sequence[str] | None = None) -> int:
    """Drive one split-fleet attempt, or publish the report over its banks."""
    args = _parser().parse_args(argv)
    adapter, schedule, preflight = build_run(args)
    if args.command == "run":
        payload = _drive(adapter, schedule, preflight)
    else:
        payload = _report(args, adapter, schedule)
    sys.stdout.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
    return 0


__all__ = (
    "COMMANDS",
    "FAMILIES",
    "GEMMA_PREFLIGHT_RECORD",
    "GEMMA_PREFLIGHT_SCHEMA",
    "ROUTE_FAMILIES",
    "SPLIT_PANEL",
    "build_run",
    "main",
    "publish_gemma_table_in_child",
)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        sys.stderr.write(f"FATAL: {exc}\n")
        raise SystemExit(2) from exc
