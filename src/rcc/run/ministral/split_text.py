"""The per-item wire format for a Ministral text arm's report handoff.

A shared appended bank cannot be a ticket payload, since a ticket carries the
digest of the exact file it names. A producer publishes a per-item bundle.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.ministral.text import MinistralDecodeSpec, MinistralReportBundle
from rcc.run.fleet.clocks import bundle_clock_spec
from rcc.run.io import atomic_bytes, canonical_bytes

# A bundle carries the redraw-ladder and closer-injection evidence beside the
# reports (draws_n, redraw_wall_s, injected_by_worker), and its generation_s is
# the span of the draw that shipped. Another schema is refused by identity.
TEXT_BUNDLE_SCHEMA = "ministral-split-text-bundle-v2"
TEXT_BUNDLE_CLOCKS_SCHEMA = "ministral-split-text-bundle-receipt-v1"

_ROWS = (
    "reports",
    "raw_outputs",
    "finish_reasons",
    "prompt_sha256",
)
_TOKEN_ROWS = ("raw_token_ids_by_worker", "visible_token_ids_by_worker")


class MinistralTextHandoffError(RuntimeError):
    """Raised when a text report bundle fails cross-process verification."""


def text_bundle_path(root: Path, qid: str, semantic_arm: str) -> Path:
    """Return where one item-arm report bundle is published."""
    return root / f"{qid}.{semantic_arm}.reports.json"


def write_text_bundle(path: Path, bundle: MinistralReportBundle) -> Path:
    """Publish one item's generated report bundle for its receiver."""
    body: dict[str, Any] = {
        "schema": TEXT_BUNDLE_SCHEMA,
        "qid": bundle.qid,
        "semantic_arm": bundle.semantic_arm,
        "seed_tag": bundle.seed_tag,
        "seeds": list(bundle.seeds),
        "prepared_prompt_fingerprint": bundle.prepared_prompt_fingerprint,
        # The published span is the shipped draw, and the ladder and injection
        # evidence travels with it. A field missing here is lost from the banked
        # row, and a cell that redrew would read as one clean natural draw.
        "generation_s": bundle.generation_s,
        "draws_n": bundle.draws_n,
        "redraw_wall_s": bundle.redraw_wall_s,
        "injected_by_worker": list(bundle.injected_by_worker),
        "queue_s_mean": bundle.queue_s_mean,
        "ttft_s_mean": bundle.ttft_s_mean,
        "report_failed_workers": list(bundle.report_failed_workers),
        **{name: list(getattr(bundle, name)) for name in _ROWS},
        **{name: [list(row) for row in getattr(bundle, name)] for name in _TOKEN_ROWS},
    }
    atomic_bytes(path, canonical_bytes(body) + b"\n")
    return path


def _flags(body: Mapping[str, Any], key: str, label: str) -> tuple[bool, ...]:
    """Read one per-worker boolean roster, refusing a truthy integer."""
    value = body.get(key)
    if not isinstance(value, list) or any(
        type(item) is not bool for item in cast(list[Any], value)
    ):
        raise MinistralTextHandoffError(
            f"{label}: report bundle draw-ladder evidence is malformed ({key})"
        )
    return tuple(cast(list[bool], value))


def _strings(body: Mapping[str, Any], key: str, label: str) -> tuple[str, ...]:
    value = body.get(key)
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in cast(list[Any], value)
    ):
        raise MinistralTextHandoffError(f"{label}: report bundle {key} is malformed")
    return tuple(cast(list[str], value))


def _ints(value: object, label: str, key: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in cast(list[Any], value)
    ):
        raise MinistralTextHandoffError(f"{label}: report bundle {key} is malformed")
    return tuple(cast(list[int], value))


def _rows(body: Mapping[str, Any], key: str, label: str) -> tuple[tuple[int, ...], ...]:
    value = body.get(key)
    if not isinstance(value, list):
        raise MinistralTextHandoffError(f"{label}: report bundle {key} is malformed")
    return tuple(_ints(row, label, key) for row in cast(list[Any], value))


def _clock(body: Mapping[str, Any], key: str, label: str, *, noun: str = "report bundle") -> float:
    value = body.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise MinistralTextHandoffError(f"{label}: {noun} {key} is malformed")
    return float(value)


def _optional_clock(body: Mapping[str, Any], key: str, label: str) -> float | None:
    """Read one clock the bundle is allowed to omit, refusing a malformed one."""
    return None if body.get(key) is None else _clock(body, key, label)


def read_text_bundle(
    path: Path, *, qid: str, semantic_arm: str, profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50
) -> MinistralReportBundle:
    """Read one published bundle, or refuse another item's or another arm's."""
    label = f"{qid}/{semantic_arm}"
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MinistralTextHandoffError(f"{label}: report bundle is unreadable") from error
    if not isinstance(decoded, dict):
        raise MinistralTextHandoffError(f"{label}: report bundle is malformed")
    body = cast(dict[str, Any], decoded)
    if (
        body.get("schema") != TEXT_BUNDLE_SCHEMA
        or body.get("qid") != qid
        or body.get("semantic_arm") != semantic_arm
    ):
        raise MinistralTextHandoffError(f"{label}: report bundle identity differs")
    seed_tag = body.get("seed_tag")
    fingerprint = body.get("prepared_prompt_fingerprint")
    generation_s = _clock(body, "generation_s", label)
    if not isinstance(seed_tag, str) or not isinstance(fingerprint, str):
        raise MinistralTextHandoffError(f"{label}: report bundle identity is malformed")
    # The ladder and injection evidence refuses with its own message, because
    # losing it would bank a redrawn, injected cell as one clean natural draw.
    draws_n = body.get("draws_n")
    if isinstance(draws_n, bool) or not isinstance(draws_n, int):
        raise MinistralTextHandoffError(f"{label}: report bundle draw-ladder evidence is malformed")
    try:
        return MinistralReportBundle(
            qid=qid,
            semantic_arm=semantic_arm,
            profile=profile,
            reports=_strings(body, "reports", label),
            raw_outputs=_strings(body, "raw_outputs", label),
            raw_token_ids_by_worker=_rows(body, "raw_token_ids_by_worker", label),
            visible_token_ids_by_worker=_rows(body, "visible_token_ids_by_worker", label),
            finish_reasons=_strings(body, "finish_reasons", label),
            seeds=_ints(body.get("seeds"), label, "seeds"),
            seed_tag=seed_tag,
            prompt_sha256=_strings(body, "prompt_sha256", label),
            prepared_prompt_fingerprint=fingerprint,
            decode=MinistralDecodeSpec(),
            generation_s=generation_s,
            draws_n=draws_n,
            redraw_wall_s=_clock(body, "redraw_wall_s", label, noun="report bundle draw-ladder"),
            injected_by_worker=_flags(body, "injected_by_worker", label),
            queue_s_mean=_optional_clock(body, "queue_s_mean", label),
            ttft_s_mean=_optional_clock(body, "ttft_s_mean", label),
            report_failed_workers=_ints(
                body.get("report_failed_workers", []), label, "report_failed_workers"
            ),
        )
    except ValueError as error:
        raise MinistralTextHandoffError(f"{label}: report bundle failed revalidation") from error


TEXT_BUNDLE_CLOCKS = bundle_clock_spec(
    schema=TEXT_BUNDLE_CLOCKS_SCHEMA,
    error=MinistralTextHandoffError,
)


__all__ = (
    "TEXT_BUNDLE_CLOCKS",
    "TEXT_BUNDLE_CLOCKS_SCHEMA",
    "TEXT_BUNDLE_SCHEMA",
    "MinistralTextHandoffError",
    "read_text_bundle",
    "text_bundle_path",
    "write_text_bundle",
)
