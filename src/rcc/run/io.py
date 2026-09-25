"""Filesystem and digest primitives shared across the run package.

A file is published through a synced staging write and an atomic replace, and
read back with the path named in any failure.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, TypeGuard, cast

from rcc.run import identity

_HASH_CHUNK_BYTES = 1 << 20
_HEX_DIGITS = frozenset("0123456789abcdef")


def is_sha256_hex(value: object) -> TypeGuard[str]:
    """Return whether a value is a lowercase 64-character SHA-256 hex digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value)
    )


def fsync_parent(path: Path) -> None:
    """Persist the directory entry containing ``path``."""
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def atomic_bytes(path: Path, payload: bytes) -> None:
    """Write bytes through a synced staging file and atomic replacement.

    The staging name is unique per call rather than per process: two writes of
    one path from one process must not share a staging file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, staged = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    staging = Path(staged)
    try:
        with os.fdopen(handle_fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, path)
        fsync_parent(path)
    finally:
        staging.unlink(missing_ok=True)


def atomic_text(path: Path, text: str) -> None:
    """Write UTF-8 text through the atomic byte path."""
    atomic_bytes(path, text.encode("utf-8"))


def atomic_json(path: Path, payload: object) -> None:
    """Write indented, sorted JSON with a trailing newline and no NaN values."""
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    atomic_text(path, encoded)


def sha256_file(path: Path) -> str:
    """Return the streamed SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(payload: object) -> bytes:
    """Return canonical JSON bytes under the strict compact profile."""
    return identity.canonical_bytes(payload, identity.json_compact_strict)


def canonical_sha(payload: object) -> str:
    """Return the SHA-256 digest of those canonical JSON bytes."""
    return identity.fingerprint(payload, identity.json_compact_strict)


def read_object(path: Path) -> dict[str, Any]:
    """Read one JSON object, naming the file in any failure."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: expected one JSON object")
    return cast(dict[str, Any], value)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON objects, naming the file and line on error."""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"{path}:{line_number}: result row is not an object")
        rows.append(cast(dict[str, Any], value))
    return rows
