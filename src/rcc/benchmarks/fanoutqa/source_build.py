"""The write halves of the FanOutQA source audit and prepared panel.

They produce exactly the bytes ``source_audit`` and ``prepare`` accept, under
``prepared/profiles/<profile_id>/``. The costly tokenizing lives in ``natural_build``.
"""

from __future__ import annotations

import json
import os
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from rcc.benchmarks.fanoutqa.panel_identity import panel_registration
from rcc.benchmarks.fanoutqa.prepare import (
    PREPARED_PANEL_SCHEMA,
    load_prepared_panel,
    prepared_source_commit,
)
from rcc.benchmarks.fanoutqa.qwen_serving import (
    panel_policy_config,
    policy_config_fingerprint,
)
from rcc.benchmarks.fanoutqa.source_audit import (
    panel_audit_paths,
    panel_prepared_paths,
    validate_source_audit,
    validate_source_commit,
)
from rcc.benchmarks.fanoutqa.source_padding import ProbeItem
from rcc.benchmarks.fanoutqa.source_planning import validated_source_ledger_sha256
from rcc.benchmarks.fanoutqa.source_topology import shard_fingerprints
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.qwen import QWEN_FAMILY
from rcc.models.qwen.source import source_build_config
from rcc.models.route import RouteFamily
from rcc.run import identity
from rcc.run.io import atomic_bytes, atomic_json, sha256_file


def build_config_fingerprint(
    *, profile: BenchmarkProfile, family: RouteFamily = QWEN_FAMILY
) -> str:
    return policy_config_fingerprint(panel_policy_config(profile, family=family))


def append_construction_row(path: Path, row: dict[str, Any]) -> None:
    """Append one sorted JSON provenance row and fsync the append-only ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def source_identities(items: Sequence[ProbeItem]) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (item.qid, pageid, revid)
        for item in items
        for pageid, revid, _title in item.question_obj.pages
    )


def refuse_foreign_bytes(
    path: Path, *, profile: BenchmarkProfile, family: RouteFamily = QWEN_FAMILY
) -> None:
    """Raise rather than unlink bytes a different profile wrote at this path.

    Rebuilding a torn tree unlinks it, which is safe only while the bytes belong to the
    profile being built; a readable manifest naming another profile stops the build.
    """
    if not path.is_file():
        return
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return
    if not isinstance(raw, dict):
        return
    body = cast(dict[str, Any], raw)
    expected = {
        "config_fingerprint": build_config_fingerprint(profile=profile, family=family),
    }
    drifted = sorted(
        field for field, value in expected.items() if field in body and body[field] != value
    )
    if drifted:
        raise RuntimeError(
            f"{path} was sealed by a different FanOutQA profile "
            f"({', '.join(drifted)} differ from {profile.profile_id}); this build refuses "
            "to overwrite it, so move or remove that tree deliberately"
        )


def family_tokenizer(family: RouteFamily) -> Any:
    import transformers

    config = source_build_config(family=family)
    tokenizer_class: Any = transformers.AutoTokenizer
    return tokenizer_class.from_pretrained(config.sol_checkpoint, revision=config.sol_revision)


def audit_source_panel(
    run_root: Path,
    *,
    source_commit: str,
    profile: BenchmarkProfile,
    family: RouteFamily = QWEN_FAMILY,
    natural_bundle: Path | None = None,
) -> dict[str, Any]:
    """Build or revalidate one profile's source audit, without any hardware."""
    from rcc.benchmarks.fanoutqa.natural_build import audit_natural_panel

    source_commit = prepared_source_commit(source_commit, profile=profile)
    ledger, manifest_path = panel_audit_paths(run_root, profile)
    if ledger.is_file() and manifest_path.is_file():
        # A node may arrive with the complete content-addressed prepared tree
        # already hydrated. Revalidate that tree rather than rebuilding the
        # expensive tokenizer audit on every worker.
        return validate_source_audit(
            run_root, source_commit=source_commit, profile=profile, family=family
        )
    return audit_natural_panel(
        run_root,
        natural_bundle,
        source_commit=source_commit,
        profile=profile,
        family=family,
    )


def prepared_fixed_fields(
    *,
    source_commit: str,
    source_audit: dict[str, Any],
    profile: BenchmarkProfile,
    family: RouteFamily = QWEN_FAMILY,
) -> dict[str, Any]:
    """Return the panel identity every construction row and the manifest carry."""
    registration = panel_registration(profile=profile, family=family)
    return {
        "prepared_schema": PREPARED_PANEL_SCHEMA,
        "panel_registration_sha": identity.fingerprint(
            registration,
            identity.json_compact_legacy,
            digest_chars=16,
        ),
        "config_fingerprint": build_config_fingerprint(profile=profile, family=family),
        "source_commit": validate_source_commit(source_commit),
        "source_audit_fingerprint": source_audit["audit_fingerprint"],
    }


def seal_prepared_panel(
    run_root: Path,
    *,
    source_commit: str,
    items: Sequence[ProbeItem],
    source_audit: dict[str, Any],
    profile: BenchmarkProfile,
    family: RouteFamily = QWEN_FAMILY,
) -> dict[str, Any]:
    """Write already-built items to the artifact and manifest, and hash the ledger."""
    artifact, manifest_path, construction = panel_prepared_paths(run_root, profile)
    qids = tuple(item.qid for item in items)
    if qids != profile.question_ids:
        raise RuntimeError("prepared FanOutQA order differs from its frozen panel")
    if shard_fingerprints(items) != source_audit["shards"]:
        raise RuntimeError("prepared FanOutQA shards differ from the sealed source audit")
    if not construction.is_file():
        raise RuntimeError("FanOutQA construction ledger was not published")
    construction_sha256 = validated_source_ledger_sha256(construction, source_identities(items))
    if family.native_sender_prompts:
        from rcc.models.qwen.native_prompts import attach_native_prompts

        items = attach_native_prompts(items, construction, family=family, profile=profile)
    bundle = {
        "schema": PREPARED_PANEL_SCHEMA,
        "registration": panel_registration(profile=profile, family=family),
        "construction_sha256": construction_sha256,
        "items": tuple(items),
    }
    atomic_bytes(artifact, pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL))
    manifest = {
        **prepared_fixed_fields(
            source_commit=source_commit,
            source_audit=source_audit,
            profile=profile,
            family=family,
        ),
        "artifact": artifact.name,
        "artifact_sha256": sha256_file(artifact),
        "construction": construction.name,
        "construction_sha256": construction_sha256,
        "item_count": len(items),
        "qids": list(qids),
    }
    atomic_json(manifest_path, manifest)
    return manifest


def prepare_panel(
    run_root: Path,
    *,
    source_commit: str,
    profile: BenchmarkProfile,
    family: RouteFamily = QWEN_FAMILY,
    natural_bundle: Path | None = None,
) -> dict[str, Any]:
    """Build, seal, and hash one FanOutQA panel, or revalidate one already written."""
    from rcc.benchmarks.fanoutqa.natural_build import natural_items_for

    source_commit = prepared_source_commit(source_commit, profile=profile)
    source_audit = validate_source_audit(
        run_root, source_commit=source_commit, profile=profile, family=family
    )
    artifact, manifest_path, construction = panel_prepared_paths(run_root, profile)
    if artifact.exists() and manifest_path.exists():
        _items, manifest = load_prepared_panel(
            run_root, source_commit=source_commit, profile=profile, family=family
        )
        return manifest
    refuse_foreign_bytes(manifest_path, profile=profile, family=family)
    artifact.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)
    # A failed prior construction can leave an append-only prefix without an
    # artifact. It is not provenance for this attempt, so rebuild the ledger
    # from empty and seal only the completed bytes.
    construction.unlink(missing_ok=True)
    fixed = prepared_fixed_fields(
        source_commit=source_commit,
        source_audit=source_audit,
        profile=profile,
        family=family,
    )
    items, rows = natural_items_for(run_root, natural_bundle, profile=profile, family=family)
    # No wall clock in the ledger: the frozen text and the pinned tokenizer fix
    # every byte, so a rebuild reproduces the artifact digest the profile pins.
    for row in rows:
        append_construction_row(construction, {**row, **fixed})
    return seal_prepared_panel(
        run_root,
        source_commit=source_commit,
        items=items,
        source_audit=source_audit,
        profile=profile,
        family=family,
    )


__all__ = (
    "append_construction_row",
    "audit_source_panel",
    "panel_registration",
    "prepare_panel",
    "prepared_fixed_fields",
    "seal_prepared_panel",
)
