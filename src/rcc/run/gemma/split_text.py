"""The per-item wire format for a Gemma text arm's report handoff.

A shared appended bank cannot be a ticket payload, since a ticket carries the
digest of the exact file it names. A producer publishes a per-item bundle.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from rcc.run.fleet.clocks import bundle_clock_spec
from rcc.run.io import atomic_bytes, canonical_bytes

# A bundle carries the per-draw ladder (`draws_n`, `redraw_wall_s`) and the
# closer injection flags. The receiver reads those keys with no default, so a
# bundle of another schema is refused by name here.
TEXT_BUNDLE_SCHEMA = "gemma-split-text-bundle-v2"
TEXT_BUNDLE_CLOCKS_SCHEMA = "gemma-split-text-bundle-receipt-v1"


class GemmaTextHandoffError(RuntimeError):
    """Raised when a text report bundle fails cross-process verification."""


def text_bundle_path(root: Path, qid: str) -> Path:
    """Return where one item's report bundle is published."""
    return root / f"{qid}.reports.json"


def write_text_bundle(
    path: Path,
    *,
    qid: str,
    semantic_arm: str,
    bundle: Mapping[str, Any],
) -> Path:
    """Publish one item's generated report bundle for its receiver."""
    body = {
        "schema": TEXT_BUNDLE_SCHEMA,
        "qid": qid,
        "semantic_arm": semantic_arm,
        "bundle": dict(bundle),
    }
    atomic_bytes(path, canonical_bytes(body) + b"\n")
    return path


def read_text_bundle(path: Path, *, qid: str, semantic_arm: str) -> dict[str, Any]:
    """Read one published bundle, or refuse another item's or another arm's."""
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GemmaTextHandoffError(f"{qid}/{semantic_arm}: report bundle is unreadable") from exc
    if not isinstance(decoded, dict):
        raise GemmaTextHandoffError(f"{qid}/{semantic_arm}: report bundle is malformed")
    body = cast(dict[str, Any], decoded)
    if (
        body.get("schema") != TEXT_BUNDLE_SCHEMA
        or body.get("qid") != qid
        or body.get("semantic_arm") != semantic_arm
    ):
        raise GemmaTextHandoffError(f"{qid}/{semantic_arm}: report bundle identity differs")
    bundle = body.get("bundle")
    if not isinstance(bundle, dict):
        raise GemmaTextHandoffError(f"{qid}/{semantic_arm}: report bundle body is malformed")
    return cast(dict[str, Any], bundle)


TEXT_BUNDLE_CLOCKS = bundle_clock_spec(
    schema=TEXT_BUNDLE_CLOCKS_SCHEMA,
    error=GemmaTextHandoffError,
)


__all__ = (
    "TEXT_BUNDLE_CLOCKS",
    "TEXT_BUNDLE_CLOCKS_SCHEMA",
    "TEXT_BUNDLE_SCHEMA",
    "GemmaTextHandoffError",
    "read_text_bundle",
    "text_bundle_path",
    "write_text_bundle",
)
