"""Build and validate the LongBench v2 chain source bundle.

The manifest's ``logical_fingerprint`` is a function of the raw dataset digest, the
sorted question ids, and the context digests; validation rehashes all of it.
"""

from __future__ import annotations

import json
import subprocess
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from rcc.benchmarks.longbench_v2.data import RawRow, choices_of
from rcc.benchmarks.longbench_v2.panel import DATASET_ID, DATASET_REVISION, RAW_SOURCE_SHA256
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.run import identity
from rcc.run.io import atomic_bytes, sha256_file

SOURCE_BUNDLE_SCHEMA = "longbench-v2-coa-source-bundle-v1"
CONTEXTS_DIR = "contexts"
ITEMS_PATH = "items.json"
GOLD_PATH = "gold.json"
_ZSTD_LEVEL = "-19"
_DIGEST_FIELDS = (
    ("source_logical_fingerprint", "logical_fingerprint"),
    ("source_manifest_sha256", "manifest_sha256"),
    ("source_archive_sha256", "archive_sha256"),
)
# The manifest digest covers every file's bytes. The archive digest is recorded
# but not compared: two zstd builds compress the same tar to different bytes.
_COMPARED_FIELDS = _DIGEST_FIELDS[:2]


def bundle_name(fingerprint: str) -> str:
    """Return the archive stem one logical fingerprint names."""
    return f"longbench-v2-coa-source-{fingerprint[:16]}"


def context_relative_path(qid: str) -> str:
    """Return one context's path inside a bundle."""
    return f"{CONTEXTS_DIR}/{qid}.txt"


def logical_fingerprint(
    *, raw_sha256: str, qids: Sequence[str], roster: Sequence[tuple[str, str]]
) -> str:
    """Return the logical digest of one bundle's content."""
    body = {
        "schema_version": SOURCE_BUNDLE_SCHEMA,
        "raw_sha256": raw_sha256,
        "qids": sorted(qids),
        "roster": [[qid, sha] for qid, sha in sorted(roster)],
    }
    return identity.fingerprint(body, identity.json_compact_legacy)


def _manifest_sha256(manifest: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return identity.fingerprint(unsigned, identity.json_compact_legacy)


def _safe_relatives(relatives: Sequence[str]) -> list[str]:
    """Return the sorted paths, raising on an absolute, escaping, or repeated one."""
    seen: set[str] = set()
    for relative in relatives:
        if relative in seen or relative.startswith("/") or ".." in Path(relative).parts:
            raise RuntimeError(f"bundle file path is unsafe or duplicated: {relative}")
        seen.add(relative)
    return sorted(seen)


def _file_rows(root: Path, relatives: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in _safe_relatives(relatives):
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"{root}: bundle is missing {relative}")
        rows.append({"bytes": path.stat().st_size, "path": relative, "sha256": sha256_file(path)})
    return rows


def _write_archive(root: Path, name: str, relatives: Sequence[str]) -> str:
    tar_path = root / f".{name}.tar"
    archive = root / f"{name}.tar.zst"
    with tarfile.open(tar_path, "w") as handle:
        for relative in ["manifest.json", *sorted(relatives)]:
            info = handle.gettarinfo(str(root / relative), arcname=relative)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o644
            with (root / relative).open("rb") as payload:
                handle.addfile(info, payload)
    try:
        subprocess.run(
            ["zstd", "-q", _ZSTD_LEVEL, "-T1", "-f", "-o", str(archive), str(tar_path)],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("minting a source bundle requires the zstd command") from exc
    finally:
        tar_path.unlink(missing_ok=True)
    digest = sha256_file(archive)
    (root / "archive.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return digest


def _archive_name(root: Path) -> tuple[Path, str]:
    line = (root / "archive.sha256").read_text(encoding="utf-8").split()
    if len(line) != 2 or len(line[0]) != 64:
        raise RuntimeError(f"{root}: archive.sha256 is not a single sha256 digest line")
    return root / line[1], line[0]


def build_source_bundle(
    root: Path, rows: Mapping[str, Mapping[str, Any]], qids: Sequence[str]
) -> dict[str, Any]:
    """Write one panel's contexts, visible items, gold, manifest, and archive."""
    root.mkdir(parents=True, exist_ok=True)
    roster: list[tuple[str, str]] = []
    visible: list[dict[str, Any]] = []
    gold: dict[str, str] = {}
    for qid in qids:
        row = rows[qid]
        path = root / context_relative_path(qid)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_bytes(path, str(row["context"]).encode("utf-8"))
        roster.append((qid, sha256_file(path)))
        visible.append(
            {
                "qid": qid,
                "domain": str(row["domain"]),
                "sub_domain": str(row["sub_domain"]),
                "difficulty": str(row["difficulty"]),
                "length": str(row["length"]),
                "question": str(row["question"]),
                "choices": list(choices_of(row)),
            }
        )
        gold[qid] = str(row["answer"])
    atomic_bytes(root / ITEMS_PATH, (json.dumps(visible, sort_keys=True, indent=1) + "\n").encode())
    atomic_bytes(root / GOLD_PATH, (json.dumps(gold, sort_keys=True, indent=1) + "\n").encode())
    relatives = [ITEMS_PATH, GOLD_PATH, *(context_relative_path(qid) for qid in qids)]
    fingerprint = logical_fingerprint(raw_sha256=RAW_SOURCE_SHA256, qids=qids, roster=roster)
    manifest: dict[str, Any] = {
        "version": SOURCE_BUNDLE_SCHEMA,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "raw_sha256": RAW_SOURCE_SHA256,
        "logical_fingerprint": fingerprint,
        "qids": sorted(qids),
        "context_files": len(roster),
        "files": _file_rows(root, relatives),
    }
    manifest["manifest_sha256"] = _manifest_sha256(manifest)
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    archive_sha = _write_archive(root, bundle_name(fingerprint), relatives)
    return {**manifest, "archive_sha256": archive_sha}


def unminted_digests(profile: BenchmarkProfile, sentinel: str) -> tuple[str, ...]:
    """Return the profile's source digest fields still carrying the sentinel."""
    return tuple(name for name, _key in _DIGEST_FIELDS if getattr(profile, name) == sentinel)


def validate_source_bundle(
    root: Path, *, profile: BenchmarkProfile, sentinel: str
) -> dict[str, Any]:
    """Rehash every byte of a bundle and bind it to one registered profile.

    A digest the profile already pins must match; one still carrying the sentinel is
    listed by ``unminted_digests`` for the caller to paste into the profile.
    """
    raw: object = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"{root}: source manifest is not an object")
    manifest = cast(dict[str, Any], raw)
    rows = manifest.get("files")
    if manifest.get("version") != SOURCE_BUNDLE_SCHEMA or not isinstance(rows, list):
        raise RuntimeError(f"{root}: source manifest has an unknown schema")
    files = cast(list[dict[str, Any]], rows)
    observed = _file_rows(root, [str(row["path"]) for row in files])
    if observed != sorted(files, key=lambda row: str(row["path"])):
        raise RuntimeError(f"{root}: source bundle bytes drifted from their digests")
    prefix = f"{CONTEXTS_DIR}/"
    roster = [
        (str(row["path"])[len(prefix) :].removesuffix(".txt"), str(row["sha256"]))
        for row in observed
        if str(row["path"]).startswith(prefix)
    ]
    qids = [str(qid) for qid in cast(list[object], manifest.get("qids", []))]
    if qids != sorted(profile.question_ids) or [qid for qid, _sha in roster] != qids:
        raise RuntimeError(f"{root}: source bundle roster differs from {profile.profile_id}")
    archive, archive_sha = _archive_name(root)
    if not archive.is_file() or sha256_file(archive) != archive_sha:
        raise RuntimeError(f"{root}: source bundle archive does not match its digest")
    expected = {
        "manifest_sha256": _manifest_sha256(manifest),
        "logical_fingerprint": logical_fingerprint(
            raw_sha256=str(manifest.get("raw_sha256")), qids=qids, roster=roster
        ),
        "context_files": len(roster),
        "raw_sha256": RAW_SOURCE_SHA256,
    }
    validated = {**manifest, "archive_sha256": archive_sha}
    for field, value in expected.items():
        if validated.get(field) != value:
            raise RuntimeError(f"{root}: source bundle {field} does not match its bytes")
    minted = set(_COMPARED_FIELDS) - {
        (name, key) for name, key in _COMPARED_FIELDS if name in unminted_digests(profile, sentinel)
    }
    drifted = sorted(name for name, key in minted if getattr(profile, name) != validated[key])
    if drifted:
        raise RuntimeError(
            f"{root}: bundle differs from the sealed {profile.profile_id}: {drifted}"
        )
    return validated


def bundle_rows(root: Path) -> dict[str, RawRow]:
    """Rebuild raw-shaped rows from a validated bundle, the gold letters included."""
    visible: object = json.loads((root / ITEMS_PATH).read_text(encoding="utf-8"))
    gold: object = json.loads((root / GOLD_PATH).read_text(encoding="utf-8"))
    if not isinstance(visible, list) or not isinstance(gold, dict):
        raise RuntimeError(f"{root}: bundle items or gold are malformed")
    rows: dict[str, RawRow] = {}
    for raw_item in cast(list[dict[str, Any]], visible):
        qid = str(raw_item["qid"])
        choices = cast(list[str], raw_item["choices"])
        rows[qid] = {
            "_id": qid,
            "domain": raw_item["domain"],
            "sub_domain": raw_item["sub_domain"],
            "difficulty": raw_item["difficulty"],
            "length": raw_item["length"],
            "question": raw_item["question"],
            "choice_A": choices[0],
            "choice_B": choices[1],
            "choice_C": choices[2],
            "choice_D": choices[3],
            "answer": cast(dict[str, str], gold)[qid],
            "context": (root / context_relative_path(qid)).read_text(encoding="utf-8"),
        }
    return rows


__all__ = (
    "CONTEXTS_DIR",
    "GOLD_PATH",
    "ITEMS_PATH",
    "SOURCE_BUNDLE_SCHEMA",
    "build_source_bundle",
    "bundle_name",
    "bundle_rows",
    "context_relative_path",
    "logical_fingerprint",
    "unminted_digests",
    "validate_source_bundle",
)
