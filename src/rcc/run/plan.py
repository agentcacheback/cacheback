"""Resolve a run config into one immutable plan.

The plan freezes the model, benchmark, topology, and hardware profiles, the arm
roster, the item range, and the run id and output prefix minted from them.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeVar

from rcc.benchmarks.fanoutqa import (
    FANOUTQA_NATURAL_DEV50,
    arms_with_full,
)
from rcc.benchmarks.fanoutqa import (
    REGISTERED_PROFILES as FANOUTQA_PROFILES,
)
from rcc.benchmarks.longbench_v2 import (
    LONGBENCH_COA_EASY50_BOUNDED,
    LONGBENCH_COA_EASY50_RERANK,
    LONGBENCH_COA_EASY50_TEXT,
)
from rcc.benchmarks.longbench_v2.nemotron import (
    NEMOTRON_LONGBENCH_BOUNDED,
    NEMOTRON_LONGBENCH_RERANK,
    NEMOTRON_LONGBENCH_TEXT,
)
from rcc.benchmarks.longbench_v2.panel import NEMOTRON_PANEL_KEYS
from rcc.benchmarks.longbench_v2.panel import TOKENIZER_REVISION as CHAIN_TOKENIZER_REVISION
from rcc.benchmarks.protocol import ArmSpec, BenchmarkProfile
from rcc.hardware.metrics import ExpectedRows
from rcc.hardware.profile import P5_8XH100, HardwareProfile
from rcc.models.gemma import GEMMA
from rcc.models.ministral import MINISTRAL
from rcc.models.nemotron import NEMOTRON, NEMOTRON_FAMILY
from rcc.models.protocol import ModelProfile
from rcc.models.qwen import QWEN, QWEN_FAMILY
from rcc.models.route import RouteFamily
from rcc.run.barrier import SPLIT_FLEET_ISOLATION_PROFILE
from rcc.run.config import RunConfig, load_run_config
from rcc.run.resolved import ResolvedPlan
from rcc.topologies.chain import CHAIN_T4, CHAIN_TOPOLOGY_KEY
from rcc.topologies.fanout import FANOUT_M3
from rcc.topologies.protocol import TopologyProfile
from rcc.transforms.select.query_support.methods.compose import (
    SUPPORT_DEFAULT_ALPHA,
    SUPPORT_DEFAULT_ORDER,
)

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_MODELS = {profile.model_id: profile for profile in (QWEN, GEMMA, MINISTRAL, NEMOTRON)}
#: The families sharing the one capture route under `rcc.models.qwen` and
#: `rcc.run.qwen`. Route code resolves its family here rather than reading
#: another family's constants.
_ROUTE_FAMILIES: dict[str, RouteFamily] = {
    QWEN.model_id: QWEN_FAMILY,
    NEMOTRON.model_id: NEMOTRON_FAMILY,
}
_ROUTE_LANES: dict[str, RouteFamily] = {family.lane: family for family in _ROUTE_FAMILIES.values()}
ROUTE_MODEL_IDS: frozenset[str] = frozenset(_ROUTE_FAMILIES)
ROUTE_LANES: frozenset[str] = frozenset(_ROUTE_LANES)
#: Every benchmark, keyed by the config word its profile declares.
_BENCHMARKS = {
    profile.benchmark_key: profile
    for profile in (
        *FANOUTQA_PROFILES,
        LONGBENCH_COA_EASY50_BOUNDED,
        LONGBENCH_COA_EASY50_RERANK,
        LONGBENCH_COA_EASY50_TEXT,
        NEMOTRON_LONGBENCH_TEXT,
        NEMOTRON_LONGBENCH_RERANK,
        NEMOTRON_LONGBENCH_BOUNDED,
    )
}
_TOPOLOGIES = {"fanout-m3": FANOUT_M3, "chain-t4": CHAIN_T4}
_SELECTOR_PROFILE = "fanoutqa-query-support-w16-v1"
_TEXT_PROFILE = "fanoutqa-primary-medium-small-v1"
# The selector profile names the method, not its correction strength. The (p, alpha) point
# comes from the selector registry and is hashed into the plan, so a build with
# another correction strength cannot create the same run identity.
_SUPPORT_COMPOSITION = f"support-p{SUPPORT_DEFAULT_ORDER:g}-a{SUPPORT_DEFAULT_ALPHA:g}"
# A split fleet divides one item range by role across the node, never by item
# across nodes.
_PRODUCTION_ISOLATION = SPLIT_FLEET_ISOLATION_PROFILE
_T = TypeVar("_T")


@dataclass(frozen=True)
class ResolutionContext:
    """The facts about the checkout that a plan is resolved against."""

    git_commit: str


def _registered(mapping: Mapping[str, _T], key: str, kind: str) -> _T:
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(f"unsupported {kind} {key!r}") from exc


def registered_benchmark(key: str) -> BenchmarkProfile:
    """Return the benchmark profile one config key names."""
    return _registered(_BENCHMARKS, key, "benchmark")


def environment_benchmark() -> BenchmarkProfile | None:
    """Return the benchmark this node was started for, when one was named.

    ``RCC_FANOUT_BENCHMARK`` carries a benchmark config key. The route lanes
    publish it in the environment; the resident lanes pass ``--benchmark``.
    """
    key = os.environ.get("RCC_FANOUT_BENCHMARK")
    return None if key is None else registered_benchmark(key)


def route_family(model_id: str) -> RouteFamily:
    """Return the route family one model id names."""
    return _registered(_ROUTE_FAMILIES, model_id, "route model")


def route_family_for_lane(lane: str) -> RouteFamily:
    """Return the route family one lane word names."""
    return _registered(_ROUTE_LANES, lane, "route lane")


def _run_identity(
    model_id: str,
    run_label: str,
    output_root: str,
    *,
    benchmark: BenchmarkProfile,
) -> tuple[str, str]:
    """Create the run id and output prefix, or refuse an unknown model.

    A route lane composes ``{lane}-{run_id_stem}-{label}`` under the benchmark's
    own output prefix, so two benchmarks on one lane cannot share a prefix.
    """
    if model_id in ROUTE_MODEL_IDS:
        run_id = f"{route_family(model_id).lane}-{benchmark.run_id_stem}-{run_label}"
        return run_id, f"{output_root}/{benchmark.output_prefix}/{run_id}"
    if model_id == "gemma4-12b-it":
        run_id = f"gemma-fanoutqa-m3-v1-{run_label}-fleet-v16"
        return run_id, f"{output_root}/gemma-fanoutqa-m3/{run_id}"
    if model_id == "ministral3-14b":
        run_id = f"ministral-fanoutqa-m3-v1-{run_label}-target-v1"
        return run_id, f"{output_root}/ministral-fanoutqa-m3/{run_id}"
    raise ValueError(f"model {model_id!r} has no registered run identity")


def _validate_config_header(config: RunConfig, context: ResolutionContext) -> None:
    """Validate the config fields that depend on nothing else."""
    if config.schema_version != 1:
        raise ValueError("only run config schema_version=1 is registered")
    if not _COMMIT.fullmatch(context.git_commit):
        raise ValueError("resolver Git commit must be exactly 40 lowercase hex characters")


def _validate_benchmark_model(
    config: RunConfig,
    model: ModelProfile,
    benchmark: BenchmarkProfile,
    topology: TopologyProfile,
) -> None:
    """Require the benchmark, its topology, and the model to be one valid set.

    This runs before the arm roster resolves, so a model that cannot run a
    benchmark at all is refused by the rule that excludes it.
    """
    if _registered(_TOPOLOGIES, benchmark.topology_key, "topology") is not topology:
        raise ValueError(
            f"benchmark {benchmark.benchmark_key!r} runs on topology "
            f"{benchmark.topology_key!r}, not {config.topology!r}"
        )
    native_chain = benchmark.benchmark_key in NEMOTRON_PANEL_KEYS
    expected_model = NEMOTRON if native_chain else QWEN
    expected_revision = NEMOTRON.tokenizer_revision if native_chain else CHAIN_TOKENIZER_REVISION
    if benchmark.topology_key == CHAIN_TOPOLOGY_KEY and (
        model.model_id != expected_model.model_id or model.tokenizer_revision != expected_revision
    ):
        raise ValueError(
            f"model {model.model_id!r} cannot run the LongBench chain: its chunk ledger is "
            f"encoded with {expected_model.tokenizer} at its registered revision and only that "
            "route lane replays it"
        )


def _validate_registered_contract(
    model: ModelProfile,
    benchmark: BenchmarkProfile,
    semantic_arms: tuple[ArmSpec, ...],
) -> None:
    """Require the shared arms to map, in order, onto this model's physical arms.

    A benchmark may name any subsequence of the arms the model implements,
    never more and never reordered.
    """
    semantic_names = tuple(arm.arm_id for arm in semantic_arms)
    sealed_arms = tuple(arm_id for arm_id, _policy in benchmark.sealed_qwen_policies)
    physical_arms = tuple(arm.semantic_arm for arm in model.physical_arms)
    benchmark_arms = tuple(arm.arm_id for arm in benchmark.arms)
    if len(set(benchmark_arms)) != len(benchmark_arms) or sealed_arms != benchmark_arms:
        raise ValueError("shared semantic arms do not map one-to-one to sealed Qwen policies")
    if tuple(arm for arm in physical_arms if arm in set(semantic_names)) != semantic_names:
        raise ValueError(f"model {model.model_id!r} has an incomplete or reordered arm mapping")


def _roster_model(model: ModelProfile, semantic_arms: tuple[ArmSpec, ...]) -> ModelProfile:
    """Cut a lane's physical roster and execution order to the arms this plan runs.

    A plan's identity is the roster it runs, not the whole roster of the lane.
    """
    names = {arm.arm_id for arm in semantic_arms}
    roster = tuple(arm for arm in model.physical_arms if arm.semantic_arm in names)
    if roster == model.physical_arms:
        return model
    order = tuple(arm for arm in model.execution_order if arm in names)
    return replace(model, physical_arms=roster, execution_order=order)


def _semantic_arms(
    config: RunConfig,
    model: ModelProfile,
    benchmark: BenchmarkProfile,
) -> tuple[ArmSpec, ...]:
    """Return the base arm roster, or that roster plus the full control.

    A benchmark names its arm profiles in order: the base roster first, then
    the full-control profiles the resident lanes run.
    """
    if config.arm_profile not in benchmark.arm_profiles:
        raise ValueError(
            f"unsupported arm profile {config.arm_profile!r} for benchmark "
            f"{benchmark.benchmark_key!r}"
        )
    base, *full = benchmark.arm_profiles
    from rcc.run.gemma.arm_subsets import EXECUTION_PROFILES

    if config.arm_profile in EXECUTION_PROFILES:
        if model.model_id != GEMMA.model_id or benchmark != FANOUTQA_NATURAL_DEV50:
            raise ValueError("Gemma execution subsets are registered only for natural-dev50")
        by_name = {arm.arm_id: arm for arm in arms_with_full(benchmark)}
        return tuple(by_name[name] for name in EXECUTION_PROFILES[config.arm_profile])
    if model.model_id == GEMMA.model_id and benchmark == FANOUTQA_NATURAL_DEV50:
        from rcc.models.gemma.roster import shared_execution_roster

        if config.arm_profile != base:
            raise ValueError("Gemma natural panel requires the eleven-arm profile")
        by_name = {arm.arm_id: arm for arm in benchmark.arms}
        return tuple(by_name[name] for name in shared_execution_roster(benchmark).published_arms)
    if model.model_id in ROUTE_MODEL_IDS:
        if config.arm_profile != base:
            lane = route_family(model.model_id).lane
            raise ValueError(f"{lane.capitalize()} does not support the full arm profile")
        return benchmark.arms
    if config.arm_profile not in full:
        raise ValueError(f"{model.model_id} requires the full arm profile")
    return arms_with_full(benchmark)


def _validate_item_range(
    config: RunConfig,
    model: ModelProfile,
    benchmark: BenchmarkProfile,
) -> int:
    """Validate the item range against the panel and return its exclusive stop."""
    if config.item_start < 0 or config.item_count < 1:
        raise ValueError("item range requires start >= 0 and count >= 1")
    stop = config.item_start + config.item_count
    if stop > len(benchmark.question_ids):
        raise ValueError(
            f"item range exceeds the sealed {len(benchmark.question_ids)}-question panel"
        )
    return stop


def resolve_plan(config: RunConfig, context: ResolutionContext) -> ResolvedPlan:
    """Resolve one config and its checkout commit into a frozen plan."""
    _validate_config_header(config, context)
    model = _registered(_MODELS, config.model, "model")
    benchmark = _registered(_BENCHMARKS, config.benchmark, "benchmark")
    if model.model_id in ROUTE_MODEL_IDS:
        benchmark = route_family(model.model_id).benchmark_profile(benchmark)
    topology = _registered(_TOPOLOGIES, config.topology, "topology")
    hardware = P5_8XH100
    assert isinstance(model, ModelProfile)
    assert isinstance(benchmark, BenchmarkProfile)
    assert isinstance(topology, TopologyProfile)
    assert isinstance(hardware, HardwareProfile)
    _validate_benchmark_model(config, model, benchmark, topology)
    semantic_arms = _semantic_arms(config, model, benchmark)
    if model.model_id == GEMMA.model_id and benchmark == FANOUTQA_NATURAL_DEV50:
        physical = {arm.semantic_arm: arm for arm in model.physical_arms}
        model = replace(model, physical_arms=tuple(physical[arm.arm_id] for arm in semantic_arms))
    _validate_registered_contract(model, benchmark, semantic_arms)
    model = _roster_model(model, semantic_arms)
    stop = _validate_item_range(config, model, benchmark)
    if config.judger_question is not None:
        # Only the route lanes' shared fan-out producer renders a judger: the
        # resident lanes carry none and the chain builds its own from retention
        # ids, so both are refused rather than banking an inert word. The
        # value moves the science and nothing else.
        if (
            model.model_id not in ROUTE_MODEL_IDS
            or benchmark.topology_key != FANOUTQA_NATURAL_DEV50.topology_key
        ):
            raise ValueError(
                "the judger question ablation is registered for the FanOutQA route lanes only"
            )
        benchmark = replace(benchmark, judger_question=config.judger_question)

    run_id, output_uri = _run_identity(
        model.model_id, config.run_label, config.output_root, benchmark=benchmark
    )
    return ResolvedPlan(
        schema_version=1,
        model=model,
        benchmark=benchmark,
        topology=topology,
        hardware=hardware,
        arm_profile=config.arm_profile,
        selector_profile=_SELECTOR_PROFILE,
        support_composition=_SUPPORT_COMPOSITION,
        text_profile=_TEXT_PROFILE,
        isolation_profile=_PRODUCTION_ISOLATION,
        item_start=config.item_start,
        item_count=config.item_count,
        selected_question_ids=benchmark.question_ids[config.item_start : stop],
        git_commit=context.git_commit,
        run_id=run_id,
        output_uri=output_uri,
        arms=semantic_arms,
        physical_arms=model.physical_arms,
        expected_rows=ExpectedRows(
            items=config.item_count,
            arms=len(semantic_arms),
            seeds=len(benchmark.sample_tags),
        ),
    )


def _git_commit() -> str:
    """Return the commit this code sits at, or forty zeros outside a checkout."""
    root = Path(__file__).resolve().parents[3]
    try:
        process = subprocess.run(
            ("git", "-C", str(root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "0" * 40
    return process.stdout.strip()


def load_and_resolve(
    path: str | Path,
    *,
    git_commit: str | None = None,
) -> ResolvedPlan:
    """Load TOML and resolve it against the commit this code is checked out at."""
    context = ResolutionContext(git_commit=git_commit or _git_commit())
    return resolve_plan(load_run_config(Path(path)), context)
