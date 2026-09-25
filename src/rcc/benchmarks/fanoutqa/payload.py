"""The manifest a producer publishes beside its payload artifacts.

Each artifact is named by digest and rehashed on read, so a consumer reads the
bytes the producer wrote or nothing.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from rcc.run.io import atomic_bytes, fsync_parent, sha256_file

PAYLOAD_SCHEMA = "fanoutqa-fleet-payload-v1"


def _manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    encoded = json.dumps(manifest, sort_keys=True, allow_nan=False) + "\n"
    return encoded.encode("utf-8")


def _publish_once(target: Path, payload: bytes) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with staging.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(staging, target)
        except FileExistsError:
            existing = target.read_bytes()
            if existing == payload:
                return target
            try:
                existing_value: object = json.loads(existing)
                new_value: object = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise RuntimeError(
                    f"{target}: payload already exists with different bytes"
                ) from None
            if not isinstance(existing_value, dict) or not isinstance(new_value, dict):
                raise RuntimeError(
                    f"{target}: payload already exists with different bytes"
                ) from None
            existing_manifest = cast(dict[str, Any], existing_value)
            new_manifest = cast(dict[str, Any], new_value)
            if (
                existing_manifest.get("schema") != new_manifest.get("schema")
                or existing_manifest.get("qid") != new_manifest.get("qid")
                or existing_manifest.get("files") != new_manifest.get("files")
            ):
                raise RuntimeError(
                    f"{target}: payload already exists with different bytes"
                ) from None
            atomic_bytes(target, payload)
            return target
        fsync_parent(target)
        return target
    finally:
        staging.unlink(missing_ok=True)


def write_payload(
    root: Path,
    *,
    qid: str,
    files: Mapping[str, Path],
    meta: Mapping[str, Any],
    report_bundle: Mapping[str, Any] | None = None,
) -> Path:
    """Write one payload manifest, naming each artifact by its digest."""
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"{qid}/{name}: payload file missing at {path}")
    manifest: dict[str, Any] = {
        "schema": PAYLOAD_SCHEMA,
        "qid": qid,
        "files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(files.items())
        },
        "meta": meta,
        "report_bundle": report_bundle,
    }
    target = root / f"{qid}.payload.json"
    return _publish_once(target, _manifest_bytes(manifest))


def read_payload(path: Path, *, qid: str) -> dict[str, Any]:
    """Read a payload manifest and rehash every artifact it names."""
    decoded: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError(f"{qid}: unknown or mismatched FanOut payload")
    manifest = cast(dict[str, Any], decoded)
    if manifest.get("schema") != PAYLOAD_SCHEMA or manifest.get("qid") != qid:
        raise RuntimeError(f"{qid}: unknown or mismatched FanOut payload")
    files_value = manifest.get("files")
    if not isinstance(files_value, dict):
        raise RuntimeError(f"{qid}: FanOut payload has no file map")
    files = cast(dict[str, Any], files_value)
    for name, raw_entry in files.items():
        if not isinstance(raw_entry, dict):
            raise RuntimeError(f"{qid}/{name}: payload file hash mismatch")
        entry = cast(dict[str, Any], raw_entry)
        file_path = Path(str(entry.get("path") or ""))
        digest = entry.get("sha256")
        if (
            not isinstance(digest, str)
            or not file_path.is_file()
            or sha256_file(file_path) != digest
        ):
            raise RuntimeError(f"{qid}/{name}: payload file hash mismatch")
    return manifest


__all__ = ("PAYLOAD_SCHEMA", "read_payload", "write_payload")
