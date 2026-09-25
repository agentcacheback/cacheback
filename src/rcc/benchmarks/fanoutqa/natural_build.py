"""Turn a natural FanOutQA bundle's frozen text into one family's tokens.

A natural profile never re-cuts a page, so building is validating the bundle,
tokenizing its worker texts, and recording the page provenance.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from rcc.benchmarks.fanoutqa.natural_panel import (
    natural_audit_body,
    natural_items,
    natural_source_rows,
    validate_natural_bundle,
)
from rcc.benchmarks.fanoutqa.preflight import actual_geometry_preflight
from rcc.benchmarks.fanoutqa.source_audit import panel_audit_paths, source_cache_root
from rcc.benchmarks.fanoutqa.source_build import (
    append_construction_row,
    build_config_fingerprint,
    family_tokenizer,
    refuse_foreign_bytes,
    source_identities,
)
from rcc.benchmarks.fanoutqa.source_padding import ProbeItem
from rcc.benchmarks.fanoutqa.source_planning import validated_source_ledger_sha256
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.route import RouteFamily
from rcc.run import identity
from rcc.run.io import atomic_json


def _bundle_root(natural_bundle: Path | None, profile: BenchmarkProfile) -> Path:
    if natural_bundle is None:
        raise ValueError(f"{profile.profile_id} builds from a natural bundle; none was given")
    validate_natural_bundle(natural_bundle, profile=profile)
    return Path(natural_bundle)


def natural_items_for(
    run_root: Path,
    natural_bundle: Path | None,
    *,
    profile: BenchmarkProfile,
    family: RouteFamily,
) -> tuple[tuple[ProbeItem, ...], list[dict[str, Any]]]:
    """Tokenize the bundle's natural texts for this family and stage the index."""
    bundle = _bundle_root(natural_bundle, profile)
    index = source_cache_root(run_root) / "fanout-final-dev.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    if not index.is_file():
        shutil.copyfile(bundle / "source_cache" / "fanout-final-dev.json", index)
    tokenizer = family_tokenizer(family)
    items = natural_items(bundle, tokenizer, profile=profile)
    rows = natural_source_rows(bundle, profile=profile)
    if [(row["qid"], row["pageid"], row["revid"]) for row in rows] != list(
        source_identities(items)
    ):
        raise RuntimeError("natural bundle page provenance differs from the pinned index order")
    actual_geometry_preflight(
        items,
        tokenizer,
        profile=family.benchmark_profile(profile),
        report_style="natural",
        enable_thinking=family.profile.decode.enable_thinking,
        system_prompt=family.thinking_system_prompt,
        assistant_prefill=family.assistant_prefill,
        low_effort=family.low_effort,
    )
    return items, rows


def audit_natural_panel(
    run_root: Path,
    natural_bundle: Path | None,
    *,
    source_commit: str,
    profile: BenchmarkProfile,
    family: RouteFamily,
) -> dict[str, Any]:
    """Write the source audit: page provenance, shard hashes, and worker widths."""
    ledger, manifest_path = panel_audit_paths(run_root, profile)
    refuse_foreign_bytes(manifest_path, profile=profile, family=family)
    ledger.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)
    items, rows = natural_items_for(run_root, natural_bundle, profile=profile, family=family)
    for row in rows:
        append_construction_row(ledger, row)
    body = natural_audit_body(
        source_commit=source_commit,
        config_fingerprint=build_config_fingerprint(profile=profile, family=family),
        items=items,
        construction_sha256=validated_source_ledger_sha256(ledger, source_identities(items)),
        profile=profile,
    )
    manifest = {
        **body,
        "audit_fingerprint": identity.fingerprint(body, identity.json_compact_legacy)[:16],
    }
    atomic_json(manifest_path, manifest)
    return manifest


__all__ = ("audit_natural_panel", "natural_items_for")
