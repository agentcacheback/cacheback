"""The natural-content FanOutQA panel: frozen page text, one span set per worker.

Built once, offline, as text, and pinned three ways. See docs/benchmarks.md,
The natural bundle.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from rcc.benchmarks.fanoutqa.panel import Question, load_questions
from rcc.benchmarks.fanoutqa.source_padding import ProbeItem
from rcc.benchmarks.fanoutqa.source_topology import shard_fingerprints, validate_shard_fingerprints
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.run import identity
from rcc.run.io import is_sha256_hex, sha256_file

NATURAL_BUNDLE_SCHEMA = "fanoutqa-natural-dev50-source-v1"
NATURAL_PANEL_SCHEMA = "fanoutqa-natural-dev50-panel-v1"
NATURAL_SOURCE_AUDIT_SCHEMA = "fanoutqa-natural-source-audit-v1"
#: The ``panel`` field of the source audit. One panel is built, so it is a
#: constant rather than an argument, and it stays in the dict because every
#: profile's ``source_audit_fingerprint`` is taken over it.
NATURAL_SOURCE_AUDIT_PANEL = "production"
QUESTION_INDEX_PATH = "source_cache/fanout-final-dev.json"


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"natural panel {label} is missing or malformed") from exc
    if not isinstance(raw, dict):
        raise RuntimeError(f"natural panel {label} is not an object")
    return cast(dict[str, Any], raw)


def _manifest_sha256(manifest: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_files(root: Path, rows: object) -> set[str]:
    if not isinstance(rows, list):
        raise RuntimeError("natural panel manifest has no file roster")
    seen: set[str] = set()
    for raw_row in cast(list[object], rows):
        if not isinstance(raw_row, dict):
            raise RuntimeError("natural panel file row is malformed")
        row = cast(dict[str, object], raw_row)
        relative, size, digest = row.get("path"), row.get("bytes"), row.get("sha256")
        if (
            set(row) != {"bytes", "path", "sha256"}
            or not isinstance(relative, str)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or not isinstance(digest, str)
            or relative in seen
            or relative.startswith("/")
            or ".." in Path(relative).parts
        ):
            raise RuntimeError("natural panel file row is malformed")
        seen.add(relative)
        path = root / relative
        if not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest:
            raise RuntimeError(f"natural panel file drifted: {relative}")
    return seen


def selected_rows(panel: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the selected items of a panel manifest in roster order."""
    raw_items = panel.get("items")
    if not isinstance(raw_items, list):
        raise RuntimeError("natural panel manifest has no items")
    rows = [
        cast(dict[str, Any], row)
        for row in cast(list[object], raw_items)
        if isinstance(row, dict) and cast(dict[str, Any], row).get("role") == "selected"
    ]
    return sorted(rows, key=lambda row: int(row["rank"]))


def validate_natural_bundle(root: Path, *, profile: BenchmarkProfile) -> dict[str, Any]:
    """Rehash a natural bundle and check it against one profile's pins."""
    root = Path(root)
    manifest = _read_object(root / "manifest.json", "manifest")
    if (
        manifest.get("version") != NATURAL_BUNDLE_SCHEMA
        or manifest.get("manifest_sha256") != profile.source_manifest_sha256
        or _manifest_sha256(manifest) != profile.source_manifest_sha256
        or manifest.get("question_index_sha256") != profile.question_index_sha256
    ):
        raise RuntimeError(f"natural bundle differs from the {profile.profile_id} pins")
    seen = _validate_files(root, manifest.get("files"))
    if QUESTION_INDEX_PATH not in seen or "panel.json" not in seen:
        raise RuntimeError("natural bundle omits its question index or panel manifest")
    if sha256_file(root / "panel.json") != profile.source_logical_fingerprint:
        raise RuntimeError(f"natural panel manifest differs from the {profile.profile_id} pin")
    archive_digest = root / "archive.sha256"
    if archive_digest.is_file():
        line = archive_digest.read_text(encoding="utf-8").split()
        archive = root / line[1] if len(line) == 2 else None
        if (
            line[:1] != [profile.source_archive_sha256]
            or archive is None
            or (archive.is_file() and sha256_file(archive) != profile.source_archive_sha256)
        ):
            raise RuntimeError("natural bundle archive differs from its registered digest")
    panel = _read_object(root / "panel.json", "panel manifest")
    policy = panel.get("policy")
    if (
        panel.get("schema") != NATURAL_PANEL_SCHEMA
        or not isinstance(policy, dict)
        or cast(dict[str, Any], policy).get("padding") != "none"
        or cast(dict[str, Any], policy).get("workers_per_item") != profile.workers_per_item
    ):
        raise RuntimeError("natural panel manifest carries an unregistered construction")
    qids = tuple(str(row["qid"]) for row in selected_rows(panel))
    if qids != profile.question_ids:
        raise RuntimeError(f"natural panel roster differs from the {profile.profile_id} panel")
    return {
        "schema": NATURAL_BUNDLE_SCHEMA,
        "fingerprint": profile.source_logical_fingerprint,
        "manifest_sha256": profile.source_manifest_sha256,
        "question_index_sha256": profile.question_index_sha256,
        "qids": list(profile.question_ids),
    }


def natural_questions(root: Path, *, profile: BenchmarkProfile) -> dict[str, Question]:
    """Return the panel's questions from the pinned index, gold below the fence.

    The index is held to the profile's own pin rather than the freeze file's, so
    a natural panel built on another pinned index still validates.
    """
    index = Path(root) / QUESTION_INDEX_PATH
    if sha256_file(index) != profile.question_index_sha256:
        raise RuntimeError(
            f"natural bundle question index differs from the {profile.profile_id} pin"
        )
    questions = {
        question.qid: question
        for question in load_questions(index, verify_sha=False)
        if question.qid in set(profile.question_ids)
    }
    missing = [qid for qid in profile.question_ids if qid not in questions]
    if missing:
        raise RuntimeError(f"pinned question index omits natural panel items {missing[:3]}")
    return questions


def natural_worker_texts(
    root: Path, *, profile: BenchmarkProfile
) -> dict[str, tuple[dict[str, Any], tuple[str, ...]]]:
    """Return every selected item's record and its worker texts, digests checked."""
    root = Path(root)
    output: dict[str, tuple[dict[str, Any], tuple[str, ...]]] = {}
    for qid in profile.question_ids:
        item = _read_object(root / "items" / f"{qid}.json", f"item {qid}")
        workers = item.get("workers")
        if (
            item.get("qid") != qid
            or item.get("role") != "selected"
            or not isinstance(workers, list)
            or len(cast(list[object], workers)) != profile.workers_per_item
        ):
            raise RuntimeError(f"{qid}: natural panel item is malformed")
        texts: list[str] = []
        for index, raw_worker in enumerate(cast(list[object], workers)):
            worker = cast(dict[str, Any], raw_worker) if isinstance(raw_worker, dict) else {}
            text = worker.get("text")
            if (
                worker.get("worker") != index
                or not isinstance(text, str)
                or not text
                or hashlib.sha256(text.encode("utf-8")).hexdigest() != worker.get("text_sha256")
            ):
                raise RuntimeError(f"{qid}/w{index}: natural worker text drifted")
            texts.append(text)
        output[qid] = (item, tuple(texts))
    return output


def natural_page_texts(root: Path, *, profile: BenchmarkProfile) -> dict[int, str]:
    """Return the cleaned page text every worker span was cut from, by page id."""
    root = Path(root)
    texts: dict[int, str] = {}
    for qid, (record, _texts) in natural_worker_texts(root, profile=profile).items():
        for page in cast(list[dict[str, Any]], record["pages"]):
            pageid = int(page["pageid"])
            if pageid in texts:
                continue
            path = root / "canonical" / f"{pageid}.md"
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise RuntimeError(f"{qid}: natural bundle lacks canonical page {pageid}") from exc
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != page.get("canonical_sha256"):
                raise RuntimeError(f"{qid}: canonical page {pageid} drifted from its digest")
            texts[pageid] = text
    return texts


def natural_items(
    root: Path, tokenizer: Any, *, profile: BenchmarkProfile
) -> tuple[ProbeItem, ...]:
    """Tokenize the worker texts for one family, with no filler."""
    validate_natural_bundle(root, profile=profile)
    questions = natural_questions(root, profile=profile)
    items: list[ProbeItem] = []
    for qid, (_record, texts) in natural_worker_texts(root, profile=profile).items():
        question = questions[qid]
        shards = tuple(
            tuple(int(token) for token in tokenizer(text, add_special_tokens=False)["input_ids"])
            for text in texts
        )
        if any(not shard for shard in shards):
            raise RuntimeError(f"{qid}: natural worker text tokenized to nothing")
        items.append(
            ProbeItem(qid=qid, question=question.question, shards=shards, question_obj=question)
        )
    return tuple(items)


def natural_source_rows(root: Path, *, profile: BenchmarkProfile) -> list[dict[str, Any]]:
    """Return the page provenance rows a natural construction ledger carries."""
    rows: list[dict[str, Any]] = []
    for qid, (record, _texts) in natural_worker_texts(root, profile=profile).items():
        for page in cast(list[dict[str, Any]], record["pages"]):
            tokens = int(page["qwen_tokens"])
            if tokens <= 0 or not is_sha256_hex(str(page["canonical_sha256"])):
                raise RuntimeError(f"{qid}: natural page provenance is malformed")
            rows.append(
                {
                    "kind": "source_page",
                    "qid": qid,
                    "pageid": int(page["pageid"]),
                    "revid": int(page["revid"]),
                    "raw_markdown_sha256": str(page["raw_sha256"]),
                    "canonical_markdown_sha256": str(page["canonical_sha256"]),
                    "canonical_tokens": tokens,
                    "frozen_census_tokens": tokens,
                    "census_delta_tokens": 0,
                }
            )
    return rows


def natural_audit_body(
    *,
    source_commit: str,
    config_fingerprint: str,
    items: tuple[ProbeItem, ...],
    construction_sha256: str,
    profile: BenchmarkProfile,
) -> dict[str, Any]:
    """Assemble the source audit the loader revalidates field by field."""
    lengths = [[len(shard) for shard in item.shards] for item in items]
    if any(not profile.admits_worker_prompt_tokens(row) for row in lengths):
        raise RuntimeError("natural panel has a worker outside the registered ceiling")
    return {
        "schema": NATURAL_SOURCE_AUDIT_SCHEMA,
        "panel": NATURAL_SOURCE_AUDIT_PANEL,
        "source_commit": source_commit,
        "config_fingerprint": config_fingerprint,
        "workers_per_item": profile.workers_per_item,
        "worker_prompt_ceiling": profile.worker_prompt_tokens,
        "prompt_geometry": profile.prompt_geometry,
        "source_logical_fingerprint": profile.source_logical_fingerprint,
        "source_manifest_sha256": profile.source_manifest_sha256,
        "qids": [item.qid for item in items],
        "construction_sha256": construction_sha256,
        "shards": shard_fingerprints(items),
        "worker_tokens": lengths,
    }


def validate_natural_source_audit(
    audit: Mapping[str, Any],
    *,
    source_commit: str,
    config_fingerprint: str,
    ledger_sha256: str,
    profile: BenchmarkProfile,
) -> dict[str, Any]:
    """Raise unless a source audit names exactly this profile's bytes."""
    body = {key: value for key, value in audit.items() if key != "audit_fingerprint"}
    expected = {
        "schema": NATURAL_SOURCE_AUDIT_SCHEMA,
        "panel": NATURAL_SOURCE_AUDIT_PANEL,
        "source_commit": source_commit,
        "config_fingerprint": config_fingerprint,
        "workers_per_item": profile.workers_per_item,
        "worker_prompt_ceiling": profile.worker_prompt_tokens,
        "prompt_geometry": profile.prompt_geometry,
        "source_logical_fingerprint": profile.source_logical_fingerprint,
        "source_manifest_sha256": profile.source_manifest_sha256,
        "qids": list(profile.question_ids),
        "construction_sha256": ledger_sha256,
        "audit_fingerprint": identity.fingerprint(body, identity.json_compact_legacy)[:16],
    }
    if set(audit) != set(expected) | {"shards", "worker_tokens"}:
        raise RuntimeError("natural source audit has an invalid schema")
    for field, value in expected.items():
        if audit.get(field) != value:
            raise RuntimeError(f"natural source audit identity mismatch for {field}")
    validate_shard_fingerprints(audit.get("shards"), profile.question_ids)
    tokens = audit.get("worker_tokens")
    shards = cast(list[dict[str, Any]], audit["shards"])
    if (
        not isinstance(tokens, list)
        or len(cast(list[object], tokens)) != len(profile.question_ids)
        or any(
            not profile.admits_worker_prompt_tokens(row)
            or [worker["tokens"] for worker in shard["workers"]] != row
            for row, shard in zip(cast(list[list[int]], tokens), shards, strict=True)
        )
    ):
        raise RuntimeError("natural source audit worker widths differ from its shards")
    return dict(audit)


__all__ = (
    "NATURAL_BUNDLE_SCHEMA",
    "NATURAL_PANEL_SCHEMA",
    "NATURAL_SOURCE_AUDIT_SCHEMA",
    "natural_audit_body",
    "natural_items",
    "natural_page_texts",
    "natural_questions",
    "natural_source_rows",
    "natural_worker_texts",
    "selected_rows",
    "validate_natural_bundle",
    "validate_natural_source_audit",
)
