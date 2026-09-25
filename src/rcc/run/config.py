"""The TOML surface a run config is written in, parsed into one value object.

Everything else a run needs is resolved from these choices in `rcc.run.plan`.
"""

from __future__ import annotations

import importlib
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "model",
        "benchmark",
        "topology",
        "arm_profile",
        "items",
        "output",
        "ablation",
    }
)
_RUN_LABEL = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")


@dataclass(frozen=True)
class RunConfig:
    """The choices a run config may make; everything else is resolved."""

    schema_version: int
    model: str
    benchmark: str
    topology: str
    arm_profile: str
    item_start: int
    item_count: int
    run_label: str
    output_root: str
    #: The one registered ablation: the selector's judger question, replaced
    #: while the worker prompt and the final receiver prompt keep the real one.
    judger_question: str | None


def _load_toml(path: Path) -> Mapping[str, Any]:
    module_name = "tomllib" if sys.version_info >= (3, 11) else "tomli"
    try:
        parser: Any = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError("Python 3.10 requires tomli to read run configs") from exc
    with path.open("rb") as handle:
        value: object = parser.load(handle)
    if not isinstance(value, dict):
        raise ValueError("run config must decode to a TOML table")
    return cast(dict[str, Any], value)


def _table(payload: Mapping[str, Any], name: str, keys: frozenset[str]) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    table = cast(dict[str, Any], value)
    extras = sorted(set(table) - keys)
    if extras:
        raise ValueError(f"unsupported [{name}] keys: {extras}")
    return table


def _string(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty TOML string")
    return value


def _integer(payload: Mapping[str, Any], name: str) -> int:
    value = payload.get(name)
    if type(value) is not int:
        raise ValueError(f"{name} must be a TOML integer")
    return value


def load_run_config(path: str | Path) -> RunConfig:
    """Parse one config file, without resolving anything it names."""
    config_path = Path(path)
    payload = _load_toml(config_path)
    extras = sorted(set(payload) - _TOP_LEVEL_KEYS)
    if extras:
        raise ValueError(f"unsupported run config keys: {extras}")

    items = _table(payload, "items", frozenset({"start", "count"}))
    output = _table(payload, "output", frozenset({"run_label", "root"}))
    run_label = _string(output, "run_label")
    if not _RUN_LABEL.fullmatch(run_label):
        raise ValueError("output.run_label must be lowercase letters, digits, and hyphens")
    output_root = _string(output, "root").rstrip("/") or "/"
    judger_question: str | None = None
    if "ablation" in payload:
        ablation = _table(payload, "ablation", frozenset({"judger_question"}))
        judger_question = _string(ablation, "judger_question")
        if not judger_question.strip():
            raise ValueError("ablation.judger_question must carry text")

    return RunConfig(
        schema_version=_integer(payload, "schema_version"),
        model=_string(payload, "model"),
        benchmark=_string(payload, "benchmark"),
        topology=_string(payload, "topology"),
        arm_profile=_string(payload, "arm_profile"),
        item_start=_integer(items, "start"),
        item_count=_integer(items, "count"),
        run_label=run_label,
        output_root=output_root,
        judger_question=judger_question,
    )
