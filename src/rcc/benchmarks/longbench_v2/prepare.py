"""Write and load the prepared chain panel and its source audit.

They live under ``prepared/profiles/<profile_id>/``, and a load accepts only bytes
whose manifest, registration digest, item roster, and chunk digests agree.
"""

from __future__ import annotations

import io
import json
import pickle
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, cast

from rcc.benchmarks.longbench_v2.chunking import SOURCE_UPDATES
from rcc.benchmarks.longbench_v2.data import ChainItem
from rcc.benchmarks.longbench_v2.registration import (
    build_config_fingerprint,
    panel_registration_sha256,
    registration_body,
)
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.run import identity
from rcc.run.io import atomic_bytes, atomic_json, sha256_file

PREPARED_SCHEMA = "longbench-v2-coa-prepared-v1"
AUDIT_SCHEMA = "longbench-v2-coa-source-audit-v1"
PANEL = "production"
_COMMIT_CHARS = set("0123456789abcdef")
_AUDIT_FIELDS = frozenset(
    {
        "schema",
        "panel",
        "source_commit",
        "config_fingerprint",
        "workers_per_item",
        "chunk_ceiling",
        "qids",
        "construction_sha256",
        "chunks",
        "strata",
        "audit_fingerprint",
    }
)


def validate_source_commit(value: str) -> str:
    """Return the value if it is a full lowercase Git commit hash, else raise."""
    if len(value) != 40 or set(value) - _COMMIT_CHARS:
        raise ValueError("source commit must be exactly 40 lowercase hex characters")
    return value


def prepared_root(run_root: Path, profile: BenchmarkProfile) -> Path:
    """Return the subtree one profile's built bytes own inside a run root."""
    return run_root / "prepared" / "profiles" / profile.profile_id


def audit_paths(run_root: Path, profile: BenchmarkProfile) -> tuple[Path, Path]:
    """Return the audit ledger and manifest paths."""
    root = prepared_root(run_root, profile) / "source_audit" / PANEL
    return root / "construction.jsonl", root / "audit.json"


def prepared_paths(run_root: Path, profile: BenchmarkProfile) -> tuple[Path, Path, Path]:
    """Return the artifact, manifest, and construction ledger paths."""
    root = prepared_root(run_root, profile)
    artifact = root / "items.pkl"
    return artifact, artifact.with_suffix(".manifest.json"), root / "construction.jsonl"


class _PreparedUnpickler(pickle.Unpickler):
    """An unpickler that resolves the prepared item dataclass and nothing else."""

    _CLASSES: ClassVar[dict[tuple[str, str], type[object]]] = {
        ("rcc.benchmarks.longbench_v2.data", "ChainItem"): ChainItem,
    }

    def find_class(self, module: str, name: str) -> Any:
        """Return the prepared item class, raising for any other global."""
        try:
            return self._CLASSES[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(
                f"prepared LongBench artifact names forbidden global {module}.{name}"
            ) from exc


def construction_rows(items: Sequence[ChainItem]) -> list[dict[str, Any]]:
    """Return one provenance row per item: source digest and four chunk digests."""
    return [
        {
            "kind": "chunk_plan",
            "qid": item.qid,
            "source_sha256": item.source_sha256,
            "source_tokens": item.source_tokens,
            "stratum": item.stratum,
            "chunks": [
                {"update": index + 1, "tokens": len(chunk), "sha256": sha}
                for index, (chunk, sha) in enumerate(
                    zip(item.chunks, item.chunk_sha256, strict=True)
                )
            ],
        }
        for item in items
    ]


def write_ledger(path: Path, rows: Sequence[dict[str, Any]]) -> str:
    """Write sorted JSON rows and return the ledger digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    atomic_bytes(path, text.encode("utf-8"))
    return sha256_file(path)


def _chunk_digests(items: Sequence[ChainItem]) -> list[dict[str, Any]]:
    return [
        {"qid": item.qid, "tokens": list(item.chunk_tokens), "sha256": list(item.chunk_sha256)}
        for item in items
    ]


def _validate_items(
    value: object, *, profile: BenchmarkProfile, audit: dict[str, Any]
) -> tuple[ChainItem, ...]:
    if type(value) is not tuple:
        raise RuntimeError("prepared LongBench item roster is not an exact tuple")
    roster = cast(tuple[object, ...], value)
    if len(roster) != len(profile.question_ids):
        raise RuntimeError("prepared LongBench artifact has the wrong item count")
    items: list[ChainItem] = []
    for expected_qid, item in zip(profile.question_ids, roster, strict=True):
        if type(item) is not ChainItem or item.qid != expected_qid:
            raise RuntimeError(f"{expected_qid}: prepared LongBench item is malformed")
        try:
            # Unpickling restores the fields without running the constructor,
            # so every field is rechecked here, the gold letter included.
            item.validate()
        except ValueError as exc:
            raise RuntimeError(f"{expected_qid}: prepared LongBench item is malformed") from exc
        if len(item.chunks) != profile.workers_per_item or any(
            len(chunk) > profile.worker_prompt_tokens for chunk in item.chunks
        ):
            raise RuntimeError(f"{expected_qid}: prepared LongBench chunks are malformed")
        items.append(item)
    if _chunk_digests(items) != audit["chunks"]:
        raise RuntimeError("prepared LongBench chunks differ from the sealed source audit")
    return tuple(items)


def seal_source_audit(
    run_root: Path,
    *,
    source_commit: str,
    items: Sequence[ChainItem],
    profile: BenchmarkProfile,
) -> dict[str, Any]:
    """Write the construction ledger and the audit manifest beside it."""
    if tuple(item.qid for item in items) != profile.question_ids:
        raise RuntimeError("LongBench source audit built the wrong panel")
    ledger, manifest_path = audit_paths(run_root, profile)
    construction_sha256 = write_ledger(ledger, construction_rows(items))
    strata = Counter(item.stratum for item in items)
    body = {
        "schema": AUDIT_SCHEMA,
        "panel": PANEL,
        "source_commit": validate_source_commit(source_commit),
        "config_fingerprint": build_config_fingerprint(profile),
        "workers_per_item": profile.workers_per_item,
        "chunk_ceiling": profile.worker_prompt_tokens,
        "qids": [item.qid for item in items],
        "construction_sha256": construction_sha256,
        "chunks": _chunk_digests(items),
        "strata": {str(band): strata[band] for band in sorted(strata)},
    }
    manifest = {
        **body,
        "audit_fingerprint": identity.fingerprint(body, identity.json_compact_legacy)[:16],
    }
    atomic_json(manifest_path, manifest)
    return manifest


def validate_source_audit(
    run_root: Path, *, source_commit: str, profile: BenchmarkProfile
) -> dict[str, Any]:
    """Raise unless the audit manifest and its ledger match this profile and each other."""
    ledger, manifest_path = audit_paths(run_root, profile)
    if not ledger.is_file() or not manifest_path.is_file():
        raise RuntimeError("LongBench source audit must run before panel preparation")
    raw: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("LongBench source audit is not an object")
    audit = cast(dict[str, Any], raw)
    if frozenset(audit) != _AUDIT_FIELDS:
        raise RuntimeError("LongBench source audit has an invalid schema")
    body = {key: value for key, value in audit.items() if key != "audit_fingerprint"}
    expected = {
        "schema": AUDIT_SCHEMA,
        "panel": PANEL,
        "source_commit": validate_source_commit(source_commit),
        "config_fingerprint": build_config_fingerprint(profile),
        "workers_per_item": profile.workers_per_item,
        "chunk_ceiling": profile.worker_prompt_tokens,
        "qids": list(profile.question_ids),
        "construction_sha256": sha256_file(ledger),
        "audit_fingerprint": identity.fingerprint(body, identity.json_compact_legacy)[:16],
    }
    for field, value in expected.items():
        if audit.get(field) != value:
            raise RuntimeError(f"LongBench source audit identity mismatch for {field}")
    chunks = audit.get("chunks")
    if not isinstance(chunks, list) or len(cast(list[object], chunks)) != len(profile.question_ids):
        raise RuntimeError("LongBench source audit chunk roster is incomplete")
    for row in cast(list[dict[str, Any]], chunks):
        if len(row.get("tokens", ())) != SOURCE_UPDATES or len(row.get("sha256", ())) != (
            SOURCE_UPDATES
        ):
            raise RuntimeError("LongBench source audit chunk row is malformed")
    return audit


def prepared_fixed_fields(
    *, source_commit: str, source_audit: dict[str, Any], profile: BenchmarkProfile
) -> dict[str, Any]:
    """Return the identity every construction row and the manifest carry."""
    return {
        "prepared_schema": PREPARED_SCHEMA,
        "panel": PANEL,
        "panel_registration_sha": panel_registration_sha256(profile),
        "config_fingerprint": build_config_fingerprint(profile),
        "source_commit": validate_source_commit(source_commit),
        "source_audit_fingerprint": source_audit["audit_fingerprint"],
    }


def seal_prepared_panel(
    run_root: Path,
    *,
    source_commit: str,
    items: Sequence[ChainItem],
    source_audit: dict[str, Any],
    profile: BenchmarkProfile,
) -> dict[str, Any]:
    """Write built items to the artifact and its manifest, and hash the ledger."""
    if tuple(item.qid for item in items) != profile.question_ids:
        raise RuntimeError("prepared LongBench order differs from its frozen panel")
    if _chunk_digests(items) != source_audit["chunks"]:
        raise RuntimeError("prepared LongBench chunks differ from the sealed source audit")
    artifact, manifest_path, construction = prepared_paths(run_root, profile)
    fixed = prepared_fixed_fields(
        source_commit=source_commit, source_audit=source_audit, profile=profile
    )
    construction_sha256 = write_ledger(
        construction, [{**row, **fixed} for row in construction_rows(items)]
    )
    bundle = {
        "schema": PREPARED_SCHEMA,
        "panel": PANEL,
        "registration": registration_body(profile),
        "construction_sha256": construction_sha256,
        "items": tuple(items),
    }
    atomic_bytes(artifact, pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL))
    manifest = {
        **fixed,
        "artifact": artifact.name,
        "artifact_sha256": sha256_file(artifact),
        "construction": construction.name,
        "construction_sha256": construction_sha256,
        "item_count": len(items),
        "qids": [item.qid for item in items],
    }
    atomic_json(manifest_path, manifest)
    return manifest


def load_prepared_panel(
    run_root: Path, *, source_commit: str, profile: BenchmarkProfile
) -> tuple[tuple[ChainItem, ...], dict[str, Any]]:
    """Load the prepared items, raising unless every digest matches this profile."""
    audit = validate_source_audit(run_root, source_commit=source_commit, profile=profile)
    artifact, manifest_path, construction = prepared_paths(run_root, profile)
    if not artifact.is_file() or not manifest_path.is_file() or not construction.is_file():
        raise RuntimeError("prepared LongBench artifact, manifest, and construction must exist")
    manifest: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        **prepared_fixed_fields(source_commit=source_commit, source_audit=audit, profile=profile),
        "artifact": artifact.name,
        "artifact_sha256": sha256_file(artifact),
        "construction": construction.name,
        "construction_sha256": sha256_file(construction),
        "item_count": len(profile.question_ids),
        "qids": list(profile.question_ids),
    }
    if manifest != expected:
        raise RuntimeError("prepared LongBench manifest does not match its panel or bytes")
    try:
        value = _PreparedUnpickler(io.BytesIO(artifact.read_bytes())).load()
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, TypeError) as exc:
        raise RuntimeError("prepared LongBench artifact is not an admitted bundle") from exc
    if not isinstance(value, dict):
        raise RuntimeError("prepared LongBench artifact is not an object")
    bundle = cast(dict[str, Any], value)
    if set(bundle) != {"schema", "panel", "registration", "construction_sha256", "items"}:
        raise RuntimeError("prepared LongBench artifact has an invalid schema")
    registration_value = bundle["registration"]
    registration = (
        cast(dict[str, Any], registration_value) if type(registration_value) is dict else None
    )
    if (
        bundle["schema"] != PREPARED_SCHEMA
        or bundle["panel"] != PANEL
        or registration is None
        or identity.fingerprint(registration, identity.json_compact_legacy, digest_chars=16)
        != panel_registration_sha256(profile)
        or bundle["construction_sha256"] != expected["construction_sha256"]
    ):
        raise RuntimeError("prepared LongBench artifact registration has drifted")
    return _validate_items(bundle["items"], profile=profile, audit=audit), expected


__all__ = (
    "AUDIT_SCHEMA",
    "PANEL",
    "PREPARED_SCHEMA",
    "audit_paths",
    "construction_rows",
    "load_prepared_panel",
    "prepared_fixed_fields",
    "prepared_paths",
    "prepared_root",
    "seal_prepared_panel",
    "seal_source_audit",
    "validate_source_audit",
    "validate_source_commit",
    "write_ledger",
)
