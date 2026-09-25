"""Durable cross-process form of the Ministral flat latent handoff.

One payload's embedding rows and the manifest committing them are published
together, with the producer's clocks, unmeasurable downstream, beside it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch

from rcc.models.ministral.payload import (
    MinistralFlatPayload,
    selected_indices_sha256,
    tensor_content_sha256,
)
from rcc.run.fleet.clocks import (
    HandoffClocks,
    handoff_clock_spec,
    read_clock_record,
    write_clock_record,
)
from rcc.run.io import atomic_bytes, canonical_bytes, canonical_sha, fsync_parent, sha256_file

MINISTRAL_HANDOFF_SCHEMA = "ministral-split-handoff-v1"
MINISTRAL_HANDOFF_CLOCKS_SCHEMA = "ministral-split-handoff-receipt-v1"
MINISTRAL_HANDOFF_ROWS = "embedding_rows"
MINISTRAL_HANDOFF_SUFFIX = ".handoff.json"

_META_KEYS = (
    "payload_schema",
    "payload_layout",
    "selected_indices_order",
    "semantic_arm",
    "latent_plan_sha256",
    "keeps",
    "rows_by_worker",
    "worker_sha256",
    "selected_indices_sha256",
    "tensor_sha256",
    "rows_shape",
    "rows_dtype",
)


class MinistralHandoffError(RuntimeError):
    """A handoff manifest, or a file it names, failed cross-process verification."""


def _atomic_tensor(path: Path, tensor: torch.Tensor) -> None:
    """Write one tensor through a synced staging file and atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with staging.open("wb") as handle:
            torch.save(tensor.detach().cpu().contiguous(), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, path)
        fsync_parent(path)
    finally:
        staging.unlink(missing_ok=True)


def _require_payload_id(payload_id: str) -> str:
    if not payload_id or payload_id.startswith("."):
        raise ValueError("Ministral handoff payload id must be a nonempty plain name")
    if any(part in payload_id for part in ("/", "\\", "..", "\0")):
        raise ValueError(f"Ministral handoff payload id {payload_id!r} is not a plain name")
    return payload_id


def handoff_fingerprint(
    *,
    payload_id: str,
    files: Mapping[str, str],
    meta: Mapping[str, Any],
) -> str:
    """Return the path-free content fingerprint of one handoff manifest."""
    identity = {
        "schema": MINISTRAL_HANDOFF_SCHEMA,
        "payload_id": payload_id,
        "files": dict(sorted(files.items())),
        "meta": dict(meta),
    }
    return canonical_sha(identity)


def handoff_meta(payload: MinistralFlatPayload) -> dict[str, Any]:
    """Return every field needed to reconstruct one flat payload from disk."""
    return {
        "payload_schema": payload.schema,
        "payload_layout": payload.layout,
        "selected_indices_order": payload.selected_indices_order,
        "semantic_arm": payload.semantic_arm,
        "latent_plan_sha256": payload.latent_plan_sha256,
        "keeps": [list(keep) for keep in payload.keeps],
        "rows_by_worker": list(payload.rows_by_worker),
        "worker_sha256": list(payload.worker_sha256),
        "selected_indices_sha256": payload.selected_indices_sha256,
        "tensor_sha256": payload.tensor_sha256,
        "rows_shape": [int(size) for size in payload.rows.shape],
        "rows_dtype": str(payload.rows.dtype),
    }


def _refuse_divergent_republish(target: Path, payload_id: str, meta: Mapping[str, Any]) -> None:
    """Refuse a second publication of one payload id with another identity."""
    if not target.is_file():
        return
    try:
        decoded: object = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MinistralHandoffError(
            f"{payload_id}: an unreadable handoff manifest already holds this id"
        ) from error
    existing = cast(dict[str, Any], decoded) if isinstance(decoded, dict) else {}
    if existing.get("meta") != dict(meta):
        raise MinistralHandoffError(
            f"{payload_id}: a different handoff payload already holds this id"
        )


def write_handoff(root: Path, *, payload_id: str, payload: MinistralFlatPayload) -> Path:
    """Publish one embedding-row tensor and the manifest that commits it."""
    _require_payload_id(payload_id)
    relative = f"{payload_id}/rows.pt"
    rows_path = root / relative
    target = root / f"{payload_id}{MINISTRAL_HANDOFF_SUFFIX}"
    _refuse_divergent_republish(target, payload_id, handoff_meta(payload))
    _atomic_tensor(rows_path, payload.rows)
    digests = {MINISTRAL_HANDOFF_ROWS: sha256_file(rows_path)}
    meta = handoff_meta(payload)
    manifest = {
        "schema": MINISTRAL_HANDOFF_SCHEMA,
        "payload_id": payload_id,
        "files": {
            MINISTRAL_HANDOFF_ROWS: {
                "path": relative,
                "sha256": digests[MINISTRAL_HANDOFF_ROWS],
            }
        },
        "meta": meta,
        "fingerprint": handoff_fingerprint(payload_id=payload_id, files=digests, meta=meta),
    }
    atomic_bytes(target, canonical_bytes(manifest) + b"\n")
    return target


HANDOFF_CLOCKS = handoff_clock_spec(
    schema=MINISTRAL_HANDOFF_CLOCKS_SCHEMA,
    error=MinistralHandoffError,
)


def write_handoff_clocks(
    manifest_path: Path,
    *,
    payload_id: str,
    fingerprint: str,
    spill_save_s: float,
    producer_s: float,
) -> Path:
    """Publish the producer's measured clocks beside the manifest it wrote.

    Neither clock can travel inside the manifest: writing it is the thing being
    measured. Both bind to that manifest's fingerprint instead.
    """
    return write_clock_record(
        HANDOFF_CLOCKS,
        manifest_path,
        label=payload_id,
        identity={"payload_id": payload_id, "fingerprint": fingerprint},
        clocks={"spill_save_s": spill_save_s, "producer_s": producer_s},
    )


def read_handoff_clocks(
    manifest_path: Path,
    *,
    payload_id: str,
    fingerprint: str,
) -> HandoffClocks:
    """Return the producer's clocks, or refuse a clock record from another write."""
    clocks = read_clock_record(
        HANDOFF_CLOCKS,
        manifest_path,
        label=payload_id,
        identity={"payload_id": payload_id, "fingerprint": fingerprint},
    )
    return HandoffClocks(**clocks)


def _decode_manifest(path: Path, payload_id: str) -> dict[str, Any]:
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MinistralHandoffError(f"{payload_id}: handoff manifest is unreadable") from error
    if not isinstance(decoded, dict):
        raise MinistralHandoffError(f"{payload_id}: handoff manifest is not an object")
    manifest = cast(dict[str, Any], decoded)
    if manifest.get("schema") != MINISTRAL_HANDOFF_SCHEMA:
        raise MinistralHandoffError(f"{payload_id}: handoff manifest schema differs")
    if manifest.get("payload_id") != payload_id:
        raise MinistralHandoffError(f"{payload_id}: handoff manifest names another payload")
    return manifest


def _manifest_files(manifest: Mapping[str, Any], payload_id: str) -> dict[str, tuple[str, str]]:
    raw_files = manifest.get("files")
    if not isinstance(raw_files, dict):
        raise MinistralHandoffError(f"{payload_id}: handoff manifest has no file map")
    files = cast(dict[str, Any], raw_files)
    if set(files) != {MINISTRAL_HANDOFF_ROWS}:
        raise MinistralHandoffError(f"{payload_id}: handoff file roster differs")
    resolved: dict[str, tuple[str, str]] = {}
    for name, raw_entry in files.items():
        tag = f"{payload_id}/{name}"
        if not isinstance(raw_entry, dict):
            raise MinistralHandoffError(f"{tag}: handoff file entry is malformed")
        entry = cast(dict[str, Any], raw_entry)
        relative = entry.get("path")
        digest = entry.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str) or not relative:
            raise MinistralHandoffError(f"{tag}: handoff file entry is malformed")
        if relative.startswith("/") or ".." in relative:
            raise MinistralHandoffError(f"{tag}: handoff file path escapes root")
        resolved[name] = (relative, digest)
    return resolved


def _require_meta(manifest: Mapping[str, Any], payload_id: str) -> dict[str, Any]:
    raw_meta = manifest.get("meta")
    if not isinstance(raw_meta, dict):
        raise MinistralHandoffError(f"{payload_id}: handoff manifest has no metadata")
    meta = cast(dict[str, Any], raw_meta)
    missing = sorted(set(_META_KEYS) - set(meta))
    if missing:
        raise MinistralHandoffError(f"{payload_id}: handoff metadata omits {', '.join(missing)}")
    return meta


def _meta_str(meta: Mapping[str, Any], key: str, payload_id: str) -> str:
    value = meta.get(key)
    if not isinstance(value, str):
        raise MinistralHandoffError(f"{payload_id}: handoff metadata {key} is not a string")
    return value


def _meta_ints(meta: Mapping[str, Any], key: str, payload_id: str) -> tuple[int, ...]:
    value = meta.get(key)
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in cast(list[Any], value)
    ):
        raise MinistralHandoffError(f"{payload_id}: handoff metadata {key} is not integers")
    return tuple(int(item) for item in cast(list[int], value))


def _meta_strings(meta: Mapping[str, Any], key: str, payload_id: str) -> tuple[str, ...]:
    value = meta.get(key)
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in cast(list[Any], value)
    ):
        raise MinistralHandoffError(f"{payload_id}: handoff metadata {key} is not strings")
    return tuple(cast(list[str], value))


def _meta_keeps(meta: Mapping[str, Any], payload_id: str) -> tuple[tuple[int, ...], ...]:
    value = meta.get("keeps")
    if not isinstance(value, list):
        raise MinistralHandoffError(f"{payload_id}: handoff metadata keeps is not a list")
    keeps: list[tuple[int, ...]] = []
    for index, raw_keep in enumerate(cast(list[Any], value)):
        if not isinstance(raw_keep, list) or any(
            not isinstance(item, int) or isinstance(item, bool)
            for item in cast(list[Any], raw_keep)
        ):
            raise MinistralHandoffError(
                f"{payload_id}: handoff metadata keep {index} is not integers"
            )
        keeps.append(tuple(int(item) for item in cast(list[int], raw_keep)))
    return tuple(keeps)


def read_handoff_manifest(path: Path, *, payload_id: str) -> dict[str, Any]:
    """Read one manifest, verify its fingerprint, and verify every file digest."""
    manifest = _decode_manifest(path, payload_id)
    files = _manifest_files(manifest, payload_id)
    meta = _require_meta(manifest, payload_id)
    expected = handoff_fingerprint(
        payload_id=payload_id,
        files={name: digest for name, (_relative, digest) in files.items()},
        meta=meta,
    )
    if manifest.get("fingerprint") != expected:
        raise MinistralHandoffError(
            f"{payload_id}: handoff manifest fingerprint differs from its content"
        )
    for name, (relative, digest) in files.items():
        target = path.parent / relative
        if not target.is_file():
            raise MinistralHandoffError(f"{payload_id}/{name}: handoff file is missing")
        if sha256_file(target) != digest:
            raise MinistralHandoffError(f"{payload_id}/{name}: handoff file digest differs")
    return manifest


def _load_rows(path: Path, payload_id: str) -> torch.Tensor:
    try:
        loaded: object = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, EOFError, RuntimeError, TypeError, ValueError) as error:
        raise MinistralHandoffError(f"{payload_id}: handoff rows could not be loaded") from (error)
    if not isinstance(loaded, torch.Tensor):
        raise MinistralHandoffError(f"{payload_id}: handoff rows are not a tensor")
    return loaded


def _require_geometry(
    rows: torch.Tensor,
    *,
    payload_id: str,
    declared_shape: Sequence[int],
    declared_dtype: str,
    rows_by_worker: Sequence[int],
) -> None:
    if rows.ndim != 2 or rows.dtype != torch.bfloat16:
        raise MinistralHandoffError(f"{payload_id}: handoff rows are not rank-two bfloat16")
    if tuple(int(size) for size in rows.shape) != tuple(declared_shape):
        raise MinistralHandoffError(f"{payload_id}: handoff rows shape differs from its meta")
    if str(rows.dtype) != declared_dtype:
        raise MinistralHandoffError(f"{payload_id}: handoff rows dtype differs from its meta")
    if sum(rows_by_worker) != int(rows.shape[0]):
        raise MinistralHandoffError(f"{payload_id}: handoff worker geometry differs from rows")


def _require_payload_digests(
    rows: torch.Tensor,
    *,
    payload_id: str,
    rows_by_worker: Sequence[int],
    keeps: Sequence[Sequence[int]],
    worker_sha: Sequence[str],
    tensor_sha: str,
    indices_sha: str,
) -> None:
    if tensor_content_sha256(rows) != tensor_sha:
        raise MinistralHandoffError(f"{payload_id}: handoff tensor digest differs")
    blocks: list[torch.Tensor] = []
    offset = 0
    for row_count in rows_by_worker:
        blocks.append(rows[offset : offset + row_count])
        offset += row_count
    if tuple(tensor_content_sha256(block) for block in blocks) != tuple(worker_sha):
        raise MinistralHandoffError(f"{payload_id}: handoff worker digest differs")
    if selected_indices_sha256(keeps) != indices_sha:
        raise MinistralHandoffError(f"{payload_id}: handoff selected-index digest differs")


def read_handoff(path: Path, *, payload_id: str) -> MinistralFlatPayload:
    """Rebuild one verified flat payload from its durable manifest."""
    manifest = read_handoff_manifest(path, payload_id=payload_id)
    files = _manifest_files(manifest, payload_id)
    meta = _require_meta(manifest, payload_id)
    rows = _load_rows(path.parent / files[MINISTRAL_HANDOFF_ROWS][0], payload_id)
    rows_by_worker = _meta_ints(meta, "rows_by_worker", payload_id)
    _require_geometry(
        rows,
        payload_id=payload_id,
        declared_shape=_meta_ints(meta, "rows_shape", payload_id),
        declared_dtype=_meta_str(meta, "rows_dtype", payload_id),
        rows_by_worker=rows_by_worker,
    )
    keeps = _meta_keeps(meta, payload_id)
    worker_sha = _meta_strings(meta, "worker_sha256", payload_id)
    tensor_sha = _meta_str(meta, "tensor_sha256", payload_id)
    indices_sha = _meta_str(meta, "selected_indices_sha256", payload_id)
    _require_payload_digests(
        rows,
        payload_id=payload_id,
        rows_by_worker=rows_by_worker,
        keeps=keeps,
        worker_sha=worker_sha,
        tensor_sha=tensor_sha,
        indices_sha=indices_sha,
    )
    return MinistralFlatPayload(
        rows=rows,
        semantic_arm=_meta_str(meta, "semantic_arm", payload_id),
        latent_plan_sha256=_meta_str(meta, "latent_plan_sha256", payload_id),
        keeps=keeps,
        rows_by_worker=rows_by_worker,
        worker_sha256=worker_sha,
        selected_indices_sha256=indices_sha,
        tensor_sha256=tensor_sha,
        layout=_meta_str(meta, "payload_layout", payload_id),
        schema=_meta_str(meta, "payload_schema", payload_id),
        selected_indices_order=_meta_str(meta, "selected_indices_order", payload_id),
    )


__all__ = (
    "MINISTRAL_HANDOFF_CLOCKS_SCHEMA",
    "MINISTRAL_HANDOFF_ROWS",
    "MINISTRAL_HANDOFF_SCHEMA",
    "MINISTRAL_HANDOFF_SUFFIX",
    "MinistralHandoffError",
    "handoff_fingerprint",
    "handoff_meta",
    "read_handoff",
    "read_handoff_clocks",
    "read_handoff_manifest",
    "write_handoff",
    "write_handoff_clocks",
)
