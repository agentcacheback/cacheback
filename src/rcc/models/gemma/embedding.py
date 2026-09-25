"""Shared embedding-table publication for the Gemma fleet lane.

One input-embedding table is published per run, as raw BF16 bytes beside an
identity marker, and every arm root links to those exact bytes.
"""

import json
import os
import shutil
import time
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import torch

from rcc.models.gemma.contract import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    EMBEDDING_SHAPE,
)
from rcc.models.gemma.mechanism import tensor_content_sha256
from rcc.run.io import sha256_file

_SHARED_EMBEDDING_SCHEMA = "fanoutqa-shared-embedding-v1"
#: Every route names the table at this fixed leaf of a bundle root, so a
#: publisher, a linker, and a receiver never disagree about where it lives.
SHARED_EMBEDDING_DIRECTORY = "shared"
SHARED_EMBEDDING_FILENAME = "input_embedding_weight.bin"
#: How often a waiting reader retries the publication it depends on.
_PUBLICATION_POLL_S = 0.25


def shared_embedding_path(root: Path) -> Path:
    """Return the shared table's fixed location under one bundle root."""
    return root / SHARED_EMBEDDING_DIRECTORY / SHARED_EMBEDDING_FILENAME


def embedding_marker_path(weight_path: Path) -> Path:
    """Durable location of the published table's identity marker."""
    return weight_path.with_suffix(".ready.json")


def embedding_validation_path(weight_path: Path, publication_id: str) -> Path:
    """Durable location of one publication's validation record."""
    tag = sha256(publication_id.encode()).hexdigest()[:16]
    return weight_path.with_name(f"input_embedding_weight.validated-{tag}.json")


def _fsync_parent(path: Path) -> None:
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def publish_shared_embedding(
    weight_path: Path,
    weight: torch.Tensor,
    *,
    checkpoint: str,
    revision: str,
    runtime_fingerprint: str,
    expected_shape: tuple[int, int],
) -> dict[str, str]:
    """Atomically publish exact raw BF16 table bytes and an identity marker."""
    value = weight.detach().cpu().contiguous()
    if value.dtype != torch.bfloat16 or tuple(value.shape) != expected_shape:
        raise RuntimeError("shared embedding table has the wrong dtype or shape")
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    staging = weight_path.with_name(f".{weight_path.name}.{os.getpid()}.tmp")
    value.view(torch.uint16).numpy().tofile(staging)
    with staging.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(staging, weight_path)
    _fsync_parent(weight_path)
    file_hash = sha256_file(weight_path)
    tensor_hash = tensor_content_sha256(value)
    marker: dict[str, object] = {
        "schema": _SHARED_EMBEDDING_SCHEMA,
        "checkpoint": checkpoint,
        "revision": revision,
        "runtime_fingerprint": runtime_fingerprint,
        "file_sha256": file_hash,
        "tensor_sha256": tensor_hash,
        "dtype": str(value.dtype),
        "shape": list(expected_shape),
        "bytes": weight_path.stat().st_size,
    }
    ready = embedding_marker_path(weight_path)
    marker_staging = ready.with_name(f".{ready.name}.{os.getpid()}.tmp")
    marker_staging.write_text(
        json.dumps(marker, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    with marker_staging.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(marker_staging, ready)
    _fsync_parent(ready)
    _load_shared_embedding_cached.cache_clear()
    return {
        "file_sha256": file_hash,
        "tensor_sha256": tensor_hash,
        "marker_sha256": sha256_file(ready),
    }


def publish_shared_embedding_validation(
    weight_path: Path, *, publication_id: str, digests: dict[str, str]
) -> None:
    """Publish the record that the table came from a live resident model."""
    target = embedding_validation_path(weight_path, publication_id)
    body = {
        "schema": "fanoutqa-shared-embedding-validation-v1",
        "publication_id": publication_id,
        "marker_sha256": digests["marker_sha256"],
        "tensor_sha256": digests["tensor_sha256"],
    }
    staging = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    staging.write_text(json.dumps(body, sort_keys=True, allow_nan=False) + "\n")
    with staging.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(staging, target)
    _fsync_parent(target)


def _require_embedding_validation(
    weight_path: Path, *, publication_id: str, digests: dict[str, str]
) -> None:
    target = embedding_validation_path(weight_path, publication_id)
    try:
        body = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("shared embedding validation record is not ready") from exc
    expected = {
        "schema": "fanoutqa-shared-embedding-validation-v1",
        "publication_id": publication_id,
        "marker_sha256": digests["marker_sha256"],
        "tensor_sha256": digests["tensor_sha256"],
    }
    if body != expected:
        raise RuntimeError("shared embedding validation record differs")


@lru_cache(maxsize=16)
def _load_shared_embedding_cached(
    weight_path: str,
    checkpoint: str,
    revision: str,
    runtime_fingerprint: str,
    expected_shape: tuple[int, int],
    weight_mtime_ns: int,
    weight_bytes: int,
    marker_mtime_ns: int,
    marker_bytes: int,
) -> tuple[torch.Tensor, dict[str, str]]:
    del weight_mtime_ns, weight_bytes, marker_mtime_ns, marker_bytes
    path = Path(weight_path)
    ready = embedding_marker_path(path)
    expected_bytes = expected_shape[0] * expected_shape[1] * 2
    try:
        raw_marker: object = json.loads(ready.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("shared embedding marker is missing or malformed") from exc
    if not isinstance(raw_marker, dict):
        raise RuntimeError("shared embedding marker is missing or malformed")
    marker = cast(dict[str, object], raw_marker)
    file_hash = marker.get("file_sha256")
    tensor_hash = marker.get("tensor_sha256")
    if not isinstance(file_hash, str) or not isinstance(tensor_hash, str):
        raise RuntimeError("shared embedding marker hashes are malformed")
    expected_marker: dict[str, object] = {
        "schema": _SHARED_EMBEDDING_SCHEMA,
        "checkpoint": checkpoint,
        "revision": revision,
        "runtime_fingerprint": runtime_fingerprint,
        "file_sha256": file_hash,
        "tensor_sha256": tensor_hash,
        "dtype": "torch.bfloat16",
        "shape": list(expected_shape),
        "bytes": expected_bytes,
    }
    if path.stat().st_size != expected_bytes or marker != expected_marker:
        raise RuntimeError("shared embedding marker identity differs")
    if sha256_file(path) != file_hash:
        raise RuntimeError("shared embedding file hash differs")
    value = torch.from_file(
        str(path),
        shared=False,
        size=expected_shape[0] * expected_shape[1],
        dtype=torch.bfloat16,
    ).reshape(expected_shape)
    if not bool(torch.isfinite(value).all()) or tensor_content_sha256(value) != tensor_hash:
        raise RuntimeError("shared embedding tensor content differs")
    return value, {
        "file_sha256": file_hash,
        "tensor_sha256": tensor_hash,
        "marker_sha256": sha256_file(ready),
    }


def load_validated_shared_embedding(
    weight_path: Path,
    *,
    checkpoint: str,
    revision: str,
    runtime_fingerprint: str,
    publication_id: str | None = None,
    expected_shape: tuple[int, int],
) -> tuple[torch.Tensor, dict[str, str]]:
    """Load only the exact table published for this checkpoint/runtime."""
    ready = embedding_marker_path(weight_path)
    try:
        weight_stat = weight_path.stat()
        marker_stat = ready.stat()
    except OSError as exc:
        raise RuntimeError("shared embedding table is not ready") from exc
    value, digests = _load_shared_embedding_cached(
        str(weight_path),
        checkpoint,
        revision,
        runtime_fingerprint,
        expected_shape,
        weight_stat.st_mtime_ns,
        weight_stat.st_size,
        marker_stat.st_mtime_ns,
        marker_stat.st_size,
    )
    if publication_id is not None:
        _require_embedding_validation(weight_path, publication_id=publication_id, digests=digests)
    return value, digests


def await_shared_embedding(
    weight_path: Path,
    *,
    checkpoint: str = CHECKPOINT_ID,
    revision: str = CHECKPOINT_REVISION,
    runtime_fingerprint: str,
    publication_id: str | None = None,
    expected_shape: tuple[int, int] = EMBEDDING_SHAPE,
    wait_s: float,
) -> tuple[torch.Tensor, dict[str, str]]:
    """Load the published table, waiting up to ``wait_s`` for its publisher.

    A split receiver opens beside its publisher, so the load waits rather than
    assuming. A nonpositive wait means the table is already there or refuses.
    """
    deadline = time.monotonic() + max(float(wait_s), 0.0)
    while True:
        try:
            return load_validated_shared_embedding(
                weight_path,
                checkpoint=checkpoint,
                revision=revision,
                runtime_fingerprint=runtime_fingerprint,
                publication_id=publication_id,
                expected_shape=expected_shape,
            )
        except RuntimeError as error:
            if time.monotonic() < deadline:
                time.sleep(_PUBLICATION_POLL_S)
                continue
            if wait_s <= 0:
                raise
            raise TimeoutError(
                f"the shared Gemma embedding table at {weight_path} was not "
                f"published within {wait_s:g}s"
            ) from error


def _table_content_sha256(weight: torch.Tensor, expected_shape: tuple[int, int]) -> str:
    value = weight.detach().cpu().contiguous()
    if value.dtype != torch.bfloat16 or tuple(value.shape) != expected_shape:
        raise RuntimeError("shared embedding table has the wrong dtype or shape")
    return tensor_content_sha256(value)


def publish_shared_embedding_once(
    run_root: Path,
    weight: torch.Tensor,
    *,
    runtime_fingerprint: str,
    publication_id: str,
    checkpoint: str = CHECKPOINT_ID,
    revision: str = CHECKPOINT_REVISION,
    expected_shape: tuple[int, int] = EMBEDDING_SHAPE,
) -> dict[str, str]:
    """Publish the run-level shared table once, or adopt the published one.

    Calling it again with the same table returns the same digests; a different
    table is refused, since banked rows were embedded from the bytes on disk.
    """
    path = shared_embedding_path(run_root)
    if not embedding_marker_path(path).is_file():
        digests = publish_shared_embedding(
            path,
            weight,
            checkpoint=checkpoint,
            revision=revision,
            runtime_fingerprint=runtime_fingerprint,
            expected_shape=expected_shape,
        )
        publish_shared_embedding_validation(path, publication_id=publication_id, digests=digests)
        return digests
    _table, digests = load_validated_shared_embedding(
        path,
        checkpoint=checkpoint,
        revision=revision,
        runtime_fingerprint=runtime_fingerprint,
        expected_shape=expected_shape,
    )
    if _table_content_sha256(weight, expected_shape) != digests["tensor_sha256"]:
        raise RuntimeError(
            "the run-level shared embedding table is already published as other bytes"
        )
    # A second attempt over the same table republishes nothing; it records its
    # own attempt against the published digests.
    publish_shared_embedding_validation(path, publication_id=publication_id, digests=digests)
    _require_embedding_validation(path, publication_id=publication_id, digests=digests)
    return digests


def _link_one(origin: Path, destination: Path) -> None:
    """Hardlink one published file into an arm root, or copy it, and verify it."""
    if not origin.is_file():
        raise RuntimeError(f"shared embedding publication is missing {origin.name}")
    digest = sha256_file(origin)
    if destination.is_file():
        if sha256_file(destination) != digest:
            raise RuntimeError(f"arm shared embedding {destination.name} differs from the run")
        return
    staging = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    staging.unlink(missing_ok=True)
    try:
        os.link(origin, staging)
    except OSError:
        # Another filesystem, so the bytes travel; the digest check below
        # holds either way and is what the receiver actually depends on.
        shutil.copyfile(origin, staging)
    os.replace(staging, destination)
    _fsync_parent(destination)
    if sha256_file(destination) != digest:
        raise RuntimeError(f"arm shared embedding {destination.name} did not link byte for byte")


def hardlink_shared_embedding(run_root: Path, arm_root: Path, *, publication_id: str) -> Path:
    """Link the run publication's three files into one arm's bundle root.

    A link across devices falls back to a copy; either way every linked file is
    digested against its source, and a divergent file is refused, not replaced.
    """
    source = shared_embedding_path(run_root)
    target = shared_embedding_path(arm_root)
    if source.resolve() == target.resolve():
        raise RuntimeError("the arm shared embedding root is the run publication itself")
    target.parent.mkdir(parents=True, exist_ok=True)
    _link_one(source, target)
    _link_one(embedding_marker_path(source), embedding_marker_path(target))
    _link_one(
        embedding_validation_path(source, publication_id),
        embedding_validation_path(target, publication_id),
    )
    return target


def publish_embedding(
    model: Any,
    weight_path: Path,
    *,
    worker_index: int,
    runtime_fingerprint: str,
    publication_id: str,
    wait_s: float = 3_600.0,
) -> dict[str, str]:
    """Publish worker zero's live table and make every peer validate its bytes."""
    if worker_index == 0:
        live = model.get_input_embeddings().weight.detach().cpu()
        digests = publish_shared_embedding(
            weight_path,
            live,
            checkpoint=CHECKPOINT_ID,
            revision=CHECKPOINT_REVISION,
            runtime_fingerprint=runtime_fingerprint,
            expected_shape=EMBEDDING_SHAPE,
        )
        publish_shared_embedding_validation(
            weight_path,
            publication_id=publication_id,
            digests=digests,
        )
        return digests
    _table, digests = await_shared_embedding(
        weight_path,
        runtime_fingerprint=runtime_fingerprint,
        publication_id=publication_id,
        wait_s=wait_s,
    )
    return digests


__all__ = (
    "SHARED_EMBEDDING_DIRECTORY",
    "SHARED_EMBEDDING_FILENAME",
    "await_shared_embedding",
    "embedding_marker_path",
    "embedding_validation_path",
    "hardlink_shared_embedding",
    "load_validated_shared_embedding",
    "publish_embedding",
    "publish_shared_embedding",
    "publish_shared_embedding_once",
    "publish_shared_embedding_validation",
    "shared_embedding_path",
)
