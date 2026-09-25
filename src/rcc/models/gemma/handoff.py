"""Cross-process handoff manifests for the split Gemma fleet lane.

One content-addressed manifest names the durable capture products, so another
process reads exactly the bytes the producer wrote. Paths are manifest-relative.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from rcc.models.gemma.capture_support import LATENT_ROLL_SCHEMA, load_latent_roll
from rcc.models.gemma.capture_types import CaptureArtifact
from rcc.models.gemma.contract import (
    ARMS,
    LATENT_REALIGN_ENABLED,
    LATENT_STEPS,
    SELECTION_SCHEMA,
    WORKERS_PER_ITEM,
)
from rcc.models.gemma.embedding import embedding_marker_path, embedding_validation_path
from rcc.models.gemma.selection import (
    QSNAP_SELECTION_SCHEMA,
    SelectionArtifact,
    load_qsnap_selection,
    load_selection,
)
from rcc.run.fleet.clocks import (
    HandoffClocks,
    handoff_clock_spec,
    read_clock_record,
    write_clock_record,
)
from rcc.run.io import atomic_bytes, canonical_bytes, canonical_sha, sha256_file

HANDOFF_SCHEMA = "gemma-split-handoff-v1"
HANDOFF_CLOCKS_SCHEMA = "gemma-split-handoff-receipt-v1"
SELECTION_KEY = "selection"
QSNAP_SELECTION_KEY = "qsnap_selection"
EMBEDDING_MARKER_KEY = "embedding_marker"
EMBEDDING_VALIDATION_KEY = "embedding_validation"


class GemmaHandoffError(RuntimeError):
    """A handoff manifest, or a file it names, failed cross-process verification."""


def rolled_key(worker: int) -> str:
    """Manifest key holding one worker's rolled latent cargo."""
    return f"rolled_w{worker}"


def handoff_manifest_path(root: Path, qid: str) -> Path:
    """Durable location of one item's handoff manifest at the artifact root."""
    return root / f"{qid}.handoff.json"


def file_roster(*, qsnap: bool = False) -> frozenset[str]:
    """Every key a complete Gemma handoff manifest names.

    A mean query attention selection banks in its own artifact, so a producer
    serving that arm names one more file than a CacheBack producer.
    """
    return frozenset(
        {
            *(rolled_key(worker) for worker in range(WORKERS_PER_ITEM)),
            SELECTION_KEY,
            EMBEDDING_MARKER_KEY,
            EMBEDDING_VALIDATION_KEY,
            *((QSNAP_SELECTION_KEY,) if qsnap else ()),
        }
    )


def manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    """Digest the manifest identity: schema, item, file digests, and geometry."""
    files = cast(Mapping[str, Mapping[str, Any]], manifest["files"])
    return canonical_sha(
        {
            "schema": manifest["schema"],
            "qid": manifest["qid"],
            "files": {name: entry["sha256"] for name, entry in files.items()},
            "meta": manifest["meta"],
        }
    )


def _selection_arms(qid: str, path: Path) -> tuple[str, ...]:
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GemmaHandoffError(f"{qid}/{SELECTION_KEY}: selection artifact is unreadable") from exc
    if not isinstance(decoded, dict):
        raise GemmaHandoffError(f"{qid}/{SELECTION_KEY}: selection artifact is malformed")
    payload = cast(dict[str, Any], decoded)
    if payload.get("schema") != SELECTION_SCHEMA or payload.get("qid") != qid:
        raise GemmaHandoffError(f"{qid}/{SELECTION_KEY}: selection artifact identity differs")
    arms = tuple(str(arm) for arm in payload.get("selection_arms", ARMS))
    if len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise GemmaHandoffError(f"{qid}/{SELECTION_KEY}: selection roster is not registered")
    return arms


def _relative_path(qid: str, name: str, anchor: Path, path: Path) -> str:
    """Record one named file relative to the manifest, or refuse to name it."""
    try:
        return path.resolve().relative_to(anchor).as_posix()
    except ValueError as exc:
        raise GemmaHandoffError(
            f"{qid}/{name}: handoff file {path} is outside the manifest root {anchor}"
        ) from exc


def write_handoff_manifest(
    manifest_path: Path,
    *,
    qid: str,
    rolled_paths: Sequence[Path],
    selection_path: Path,
    embedding_weight: Path,
    publication_id: str,
    hidden: int,
    frame_prefix_length: int,
    frame_suffix_length: int,
    lengths: Sequence[int],
    embedding_digests: Mapping[str, Any],
    global_layers: Sequence[int],
    audition_replays_by_ratio: Mapping[int, int],
    qsnap_selection_path: Path | None = None,
) -> Path:
    """Publish one item's manifest over the bytes the capture phase already wrote."""
    if len(rolled_paths) != WORKERS_PER_ITEM:
        raise GemmaHandoffError(
            f"{qid}: handoff names {len(rolled_paths)} rolled files, expected {WORKERS_PER_ITEM}"
        )
    if len(lengths) != WORKERS_PER_ITEM or any(int(value) <= 0 for value in lengths):
        raise GemmaHandoffError(
            f"{qid}: handoff payload lengths {tuple(lengths)} are not {WORKERS_PER_ITEM} positive"
        )
    sources: dict[str, Path] = {
        **{rolled_key(worker): Path(path) for worker, path in enumerate(rolled_paths)},
        SELECTION_KEY: Path(selection_path),
        EMBEDDING_MARKER_KEY: embedding_marker_path(Path(embedding_weight)),
        EMBEDDING_VALIDATION_KEY: embedding_validation_path(Path(embedding_weight), publication_id),
    }
    if qsnap_selection_path is not None:
        sources[QSNAP_SELECTION_KEY] = Path(qsnap_selection_path)
    for name, path in sorted(sources.items()):
        if not path.is_file():
            raise GemmaHandoffError(f"{qid}/{name}: handoff file is missing at {path}")
    anchor = manifest_path.parent.resolve()
    files = {
        name: {"path": _relative_path(qid, name, anchor, path), "sha256": sha256_file(path)}
        for name, path in sorted(sources.items())
    }
    meta: dict[str, Any] = {
        "selection_schema": SELECTION_SCHEMA,
        "selection_arms": list(_selection_arms(qid, sources[SELECTION_KEY])),
        "latent_roll_schema": LATENT_ROLL_SCHEMA,
        "latent_steps": LATENT_STEPS,
        "realign_enabled": LATENT_REALIGN_ENABLED,
        "workers": WORKERS_PER_ITEM,
        "hidden": int(hidden),
        "frame_prefix_length": int(frame_prefix_length),
        "frame_suffix_length": int(frame_suffix_length),
        "lengths": [int(value) for value in lengths],
        "embedding_digests": dict(embedding_digests),
        "global_layers": [int(layer) for layer in global_layers],
        "audition_replays_by_ratio": {
            str(int(ratio)): int(count)
            for ratio, count in sorted(audition_replays_by_ratio.items())
        },
        "embedding": {
            "weight_path": _relative_path(qid, "embedding_weight", anchor, Path(embedding_weight)),
            "publication_id": publication_id,
        },
        **(
            {"qsnap_selection_schema": QSNAP_SELECTION_SCHEMA}
            if qsnap_selection_path is not None
            else {}
        ),
    }
    manifest: dict[str, Any] = {
        "schema": HANDOFF_SCHEMA,
        "qid": qid,
        "files": files,
        "meta": meta,
    }
    manifest["fingerprint"] = manifest_fingerprint(manifest)
    atomic_bytes(manifest_path, canonical_bytes(manifest) + b"\n")
    return manifest_path


HANDOFF_CLOCKS = handoff_clock_spec(schema=HANDOFF_CLOCKS_SCHEMA, error=GemmaHandoffError)


def write_handoff_clocks(
    manifest_path: Path,
    *,
    qid: str,
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
        label=qid,
        identity={"qid": qid, "fingerprint": fingerprint},
        clocks={"spill_save_s": spill_save_s, "producer_s": producer_s},
    )


def read_handoff_clocks(manifest_path: Path, *, qid: str, fingerprint: str) -> HandoffClocks:
    """Return the producer's clocks, or refuse a clock record from another write."""
    clocks = read_clock_record(
        HANDOFF_CLOCKS,
        manifest_path,
        label=qid,
        identity={"qid": qid, "fingerprint": fingerprint},
    )
    return HandoffClocks(**clocks)


def _require_files(qid: str, manifest: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    raw = manifest.get("files")
    if not isinstance(raw, dict):
        raise GemmaHandoffError(f"{qid}: handoff manifest has no file map")
    entries = cast(dict[str, Any], raw)
    if frozenset(entries) not in (file_roster(), file_roster(qsnap=True)):
        raise GemmaHandoffError(f"{qid}: handoff manifest file roster differs: {sorted(entries)}")
    files: dict[str, dict[str, str]] = {}
    for name, entry in sorted(entries.items()):
        if not isinstance(entry, dict):
            raise GemmaHandoffError(f"{qid}/{name}: handoff file entry is malformed")
        body = cast(dict[str, Any], entry)
        path = body.get("path")
        digest = body.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str) or not path:
            raise GemmaHandoffError(f"{qid}/{name}: handoff file entry is malformed")
        if path.startswith("/") or ".." in Path(path).parts:
            raise GemmaHandoffError(f"{qid}/{name}: handoff file path escapes the manifest root")
        files[name] = {"path": path, "sha256": digest}
    return files


def _require_int_list(qid: str, meta: Mapping[str, Any], name: str, *, count: int | None) -> None:
    value = meta.get(name)
    if not isinstance(value, list) or (count is not None and len(cast(list[Any], value)) != count):
        raise GemmaHandoffError(f"{qid}: handoff geometry {name} is malformed")
    for item in cast(list[Any], value):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise GemmaHandoffError(f"{qid}: handoff geometry {name} is malformed")


def _require_carried(qid: str, meta: Mapping[str, Any]) -> None:
    """Refuse a manifest that omits or malforms a field the receiver banks."""
    _require_int_list(qid, meta, "lengths", count=WORKERS_PER_ITEM)
    if any(int(value) <= 0 for value in cast(list[int], meta["lengths"])):
        raise GemmaHandoffError(f"{qid}: handoff geometry lengths is malformed")
    _require_int_list(qid, meta, "global_layers", count=None)
    if not isinstance(meta.get("embedding_digests"), dict):
        raise GemmaHandoffError(f"{qid}: handoff embedding digests are malformed")
    replays = meta.get("audition_replays_by_ratio")
    if not isinstance(replays, dict):
        raise GemmaHandoffError(f"{qid}: handoff audition ledger is malformed")
    for ratio, count in cast(dict[str, Any], replays).items():
        # isdecimal, not isdigit: it is exactly the set int() accepts.
        if not ratio.isdecimal() or isinstance(count, bool) or not isinstance(count, int):
            raise GemmaHandoffError(f"{qid}: handoff audition ledger is malformed")
        if count < 0:
            raise GemmaHandoffError(f"{qid}: handoff audition ledger is malformed")


def _require_meta(qid: str, manifest: Mapping[str, Any], *, qsnap: bool) -> dict[str, Any]:
    raw = manifest.get("meta")
    if not isinstance(raw, dict):
        raise GemmaHandoffError(f"{qid}: handoff manifest has no geometry")
    meta = cast(dict[str, Any], raw)
    # The mean query attention schema and file are declared together, so a
    # manifest neither promises a missing selection nor carries an undeclared one.
    pinned = {
        "selection_schema": SELECTION_SCHEMA,
        "latent_roll_schema": LATENT_ROLL_SCHEMA,
        "latent_steps": LATENT_STEPS,
        "realign_enabled": LATENT_REALIGN_ENABLED,
        "workers": WORKERS_PER_ITEM,
        "qsnap_selection_schema": QSNAP_SELECTION_SCHEMA if qsnap else None,
    }
    for name, wanted in pinned.items():
        found = meta.get(name)
        if found != wanted:
            raise GemmaHandoffError(
                f"{qid}: handoff geometry {name}={found!r} is not the registered {wanted!r}"
            )
    for name in ("hidden", "frame_prefix_length", "frame_suffix_length"):
        value = meta.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GemmaHandoffError(f"{qid}: handoff geometry {name} is malformed")
    arms = meta.get("selection_arms")
    if not isinstance(arms, list) or any(arm not in ARMS for arm in cast(list[Any], arms)):
        raise GemmaHandoffError(f"{qid}: handoff selection roster is not registered")
    _require_carried(qid, meta)
    return meta


def read_handoff_manifest(manifest_path: Path, *, qid: str) -> dict[str, Any]:
    """Read one manifest and verify the digest of every file it names."""
    try:
        decoded: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GemmaHandoffError(f"{qid}: handoff manifest is unreadable") from exc
    if not isinstance(decoded, dict):
        raise GemmaHandoffError(f"{qid}: handoff manifest is malformed")
    manifest = cast(dict[str, Any], decoded)
    if manifest.get("schema") != HANDOFF_SCHEMA or manifest.get("qid") != qid:
        raise GemmaHandoffError(f"{qid}: handoff manifest identity differs")
    files = _require_files(qid, manifest)
    meta = _require_meta(qid, manifest, qsnap=QSNAP_SELECTION_KEY in files)
    verified: dict[str, Any] = {
        "schema": HANDOFF_SCHEMA,
        "qid": qid,
        "files": files,
        "meta": meta,
    }
    fingerprint = manifest_fingerprint(verified)
    if manifest.get("fingerprint") != fingerprint:
        raise GemmaHandoffError(f"{qid}: handoff manifest fingerprint differs")
    for name, entry in sorted(files.items()):
        path = manifest_path.parent / entry["path"]
        if not path.is_file():
            raise GemmaHandoffError(f"{qid}/{name}: handoff file is missing at {path}")
        if sha256_file(path) != entry["sha256"]:
            raise GemmaHandoffError(f"{qid}/{name}: handoff file digest differs")
    verified["fingerprint"] = fingerprint
    return verified


def _manifest_lengths(
    qid: str, meta: Mapping[str, Any], lengths: Sequence[int] | None
) -> tuple[int, ...]:
    """The payload geometry the manifest carries, cross-checked if one was passed."""
    carried = tuple(int(value) for value in cast(list[int], meta["lengths"]))
    if lengths is not None and tuple(int(value) for value in lengths) != carried:
        raise GemmaHandoffError(
            f"{qid}: handoff payload lengths {tuple(lengths)} differ from the manifest {carried}"
        )
    return carried


def _load_selections(
    root: Path,
    files: Mapping[str, Mapping[str, str]],
    meta: Mapping[str, Any],
    *,
    qid: str,
    lengths: tuple[int, ...],
) -> tuple[SelectionArtifact, SelectionArtifact | None]:
    """Load the v16 selection and, when the manifest names one, the mean query attention bank."""
    selection = load_selection(
        root / files[SELECTION_KEY]["path"],
        qid=qid,
        lengths=lengths,
        arms=tuple(str(arm) for arm in cast(list[Any], meta["selection_arms"])),
    )
    entry = files.get(QSNAP_SELECTION_KEY)
    if entry is None:
        return selection, None
    return selection, load_qsnap_selection(root / entry["path"], qid=qid)


def _require_shared_table(
    qid: str,
    files: Mapping[str, Mapping[str, str]],
    digests: Mapping[str, str] | None,
) -> None:
    """Refuse a bundle whose table is not the one the receiver embeds from."""
    if digests is None:
        return
    named = files[EMBEDDING_MARKER_KEY]["sha256"]
    if named != digests.get("marker_sha256"):
        raise GemmaHandoffError(
            f"{qid}: the handoff names shared embedding marker {named}, but this "
            f"receiver loaded {digests.get('marker_sha256')}"
        )


def load_capture_artifact(
    manifest_path: Path,
    *,
    qid: str,
    lengths: Sequence[int] | None = None,
    arm: str | None = None,
    shared_embedding_digests: Mapping[str, str] | None = None,
) -> CaptureArtifact:
    """Rebuild every receiver-banked capture field from a verified manifest.

    ``lengths`` and ``arm`` are cross-checks refused on mismatch, the digests
    bind the cargo to the receiver's own table, and producer clocks stay zero.
    """
    manifest = read_handoff_manifest(manifest_path, qid=qid)
    files = cast(dict[str, dict[str, str]], manifest["files"])
    meta = cast(dict[str, Any], manifest["meta"])
    _require_shared_table(qid, files, shared_embedding_digests)
    root = manifest_path.parent
    payload_lengths = _manifest_lengths(qid, meta, lengths)
    replays = {
        int(ratio): int(count)
        for ratio, count in cast(dict[str, int], meta["audition_replays_by_ratio"]).items()
    }
    try:
        selection, qsnap = _load_selections(
            root,
            files,
            meta,
            qid=qid,
            lengths=payload_lengths,
        )
        rolled = tuple(
            load_latent_roll(
                root / files[rolled_key(worker)]["path"],
                qid=qid,
                worker=worker,
                hidden=int(meta["hidden"]),
                frame_prefix_length=int(meta["frame_prefix_length"]),
                frame_suffix_length=int(meta["frame_suffix_length"]),
            )
            for worker in range(WORKERS_PER_ITEM)
        )
    except GemmaHandoffError:
        raise
    except RuntimeError as exc:
        raise GemmaHandoffError(str(exc)) from exc
    # The mean query attention bank adds its arms beside the v16 arms; an arm named by both
    # banks is refused rather than one bank's keeps overwriting the other's.
    if qsnap is not None:
        collisions = sorted(
            (set(selection.keeps_by_arm) & set(qsnap.keeps_by_arm))
            | (set(selection.layouts_by_arm) & set(qsnap.layouts_by_arm))
            | (set(selection.selection_s_by_arm) & set(qsnap.selection_s_by_arm))
        )
        if collisions:
            raise GemmaHandoffError(
                f"{qid}: the mean query attention and v16 selection banks both name {collisions}"
            )
    keeps_by_arm = {**selection.keeps_by_arm, **(qsnap.keeps_by_arm if qsnap else {})}
    if arm is not None and arm not in keeps_by_arm:
        raise GemmaHandoffError(
            f"{qid}/{arm}: the handoff manifest carries no selection for this arm"
        )
    return CaptureArtifact(
        qid=qid,
        keeps_by_arm=keeps_by_arm,
        layouts_by_arm={**selection.layouts_by_arm, **(qsnap.layouts_by_arm if qsnap else {})},
        rolled_by_worker=rolled,
        selection_s_by_arm={
            **selection.selection_s_by_arm,
            **(qsnap.selection_s_by_arm if qsnap else {}),
        },
        audition_s_by_ratio=dict.fromkeys(replays, 0.0),
        audition_replays_by_ratio=replays,
        reports=(),
        report_failed=False,
        report_failure=None,
        capture_s=0.0,
        capture_s_by_worker=(0.0,) * WORKERS_PER_ITEM,
        report_generation_s=0.0,
        embedding_digests=dict(cast(dict[str, Any], meta["embedding_digests"])),
        global_layers=tuple(int(layer) for layer in cast(list[int], meta["global_layers"])),
        reloaded_captures=WORKERS_PER_ITEM,
        reloaded_selections=1 + int(qsnap is not None),
    )


__all__ = (
    "EMBEDDING_MARKER_KEY",
    "EMBEDDING_VALIDATION_KEY",
    "HANDOFF_CLOCKS_SCHEMA",
    "HANDOFF_SCHEMA",
    "QSNAP_SELECTION_KEY",
    "SELECTION_KEY",
    "GemmaHandoffError",
    "file_roster",
    "handoff_manifest_path",
    "load_capture_artifact",
    "manifest_fingerprint",
    "read_handoff_clocks",
    "read_handoff_manifest",
    "rolled_key",
    "write_handoff_clocks",
    "write_handoff_manifest",
)
