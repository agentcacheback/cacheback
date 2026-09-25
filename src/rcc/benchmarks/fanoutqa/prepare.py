"""Load a prepared FanOutQA panel, or raise.

The artifact is accepted only when its manifest, source audit, registration digest,
and item roster all agree with the profile asked for.
"""

from __future__ import annotations

import io
import json
import os
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, cast

from rcc.benchmarks.fanoutqa.panel import Question
from rcc.benchmarks.fanoutqa.panel_identity import panel_registration_sha
from rcc.benchmarks.fanoutqa.source_audit import (
    panel_config_fingerprint,
    panel_prepared_paths,
    validate_source_audit,
    validate_source_commit,
)
from rcc.benchmarks.fanoutqa.source_padding import ProbeItem
from rcc.benchmarks.fanoutqa.source_planning import validated_source_ledger_sha256
from rcc.benchmarks.fanoutqa.source_topology import shard_fingerprints
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.qwen import QWEN_FAMILY
from rcc.models.route import RouteFamily
from rcc.run import identity
from rcc.run.io import atomic_json, sha256_file

PREPARED_PANEL_SCHEMA = "fanoutqa-m3-serving-prepared-v2"
NATIVE_PROMPT_RECORD_SCHEMA = "fanoutqa-native-prompt-receipt-v1"
SEALED_PREPARED_SOURCE_COMMIT = "309fad8de6896d042fffbeed284e9384b6ade26f"


def prepared_source_commit(execution_commit: str, *, profile: BenchmarkProfile) -> str:
    """Return the commit the prepared bytes must carry.

    With no override that is the execution commit; an override is accepted only
    when it equals the profile's own source commit.
    """
    execution_commit = validate_source_commit(execution_commit)
    requested = os.environ.get("RCC_FANOUT_PREPARED_SOURCE_COMMIT")
    if requested is None:
        return execution_commit
    prepared = validate_source_commit(requested)
    if prepared != profile.source_commit:
        raise RuntimeError(
            f"prepared source commit differs from the sealed {profile.profile_id} run"
        )
    return prepared


class _PreparedPanelUnpickler(pickle.Unpickler):
    """An unpickler that resolves the two prepared dataclasses and nothing else."""

    _CLASSES: ClassVar[dict[tuple[str, str], type[object]]] = {
        ("rcc.benchmarks.fanoutqa.source_padding", "ProbeItem"): ProbeItem,
        ("rcc.benchmarks.fanoutqa.panel", "Question"): Question,
    }

    def find_class(self, module: str, name: str) -> Any:
        """Return one of the prepared dataclasses, raising for any other global."""
        try:
            return self._CLASSES[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(
                f"prepared FanOutQA artifact names forbidden global {module}.{name}"
            ) from exc


def _load_bundle(path: Path) -> dict[str, Any]:
    try:
        value = _PreparedPanelUnpickler(io.BytesIO(path.read_bytes())).load()
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, TypeError) as exc:
        raise RuntimeError("prepared FanOutQA artifact is not an admitted bundle") from exc
    if not isinstance(value, dict):
        raise RuntimeError("prepared FanOutQA artifact is not an object")
    return cast(dict[str, Any], value)


def _validate_question(question: Any, *, qid: str, text: str) -> Question:
    if (
        type(question) is not Question
        or question.qid != qid
        or question.question != text
        or type(question.raw) is not dict
        or type(question.pages) is not tuple
        or not question.pages
    ):
        raise RuntimeError(f"{qid}: prepared FanOutQA question object is malformed")
    for page in question.pages:
        if (
            type(page) is not tuple
            or len(page) != 3
            or type(page[0]) is not int
            or page[0] <= 0
            or type(page[1]) is not int
            or page[1] <= 0
            or type(page[2]) is not str
            or not page[2]
        ):
            raise RuntimeError(f"{qid}: prepared FanOutQA page identity is malformed")
    return question


def _validate_items(value: Any, *, profile: BenchmarkProfile) -> tuple[ProbeItem, ...]:
    if type(value) is not tuple:
        raise RuntimeError("prepared FanOutQA item roster is not an exact tuple")
    roster = cast(tuple[object, ...], value)
    expected_qids = profile.question_ids
    if len(roster) != len(expected_qids):
        raise RuntimeError("prepared FanOutQA artifact has the wrong item count")
    items: list[ProbeItem] = []
    for expected_qid, item in zip(expected_qids, roster, strict=True):
        if (
            type(item) is not ProbeItem
            or item.qid != expected_qid
            or type(item.question) is not str
            or not item.question
            or item.task_kind != "qa"
            or type(item.shards) is not tuple
            or len(item.shards) != profile.workers_per_item
        ):
            raise RuntimeError(f"{expected_qid}: prepared FanOutQA item is malformed")
        for shard in item.shards:
            if (
                type(shard) is not tuple
                or not shard
                or any(type(token) is not int or token < 0 for token in shard)
            ):
                raise RuntimeError(f"{expected_qid}: prepared FanOutQA token shard is malformed")
        _validate_question(item.question_obj, qid=item.qid, text=item.question)
        items.append(item)
    return tuple(items)


def load_prepared_panel(
    run_root: Path,
    *,
    source_commit: str,
    profile: BenchmarkProfile,
    family: RouteFamily = QWEN_FAMILY,
) -> tuple[tuple[ProbeItem, ...], dict[str, Any]]:
    """Load the prepared items, raising unless every digest matches this profile."""
    prepared_commit = prepared_source_commit(source_commit, profile=profile)
    audit = validate_source_audit(
        run_root,
        source_commit=prepared_commit,
        profile=profile,
        family=family,
    )
    artifact, manifest_path, construction = panel_prepared_paths(run_root, profile)
    if not artifact.is_file() or not manifest_path.is_file() or not construction.is_file():
        raise RuntimeError("prepared FanOutQA artifact, manifest, and construction must exist")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("prepared FanOutQA manifest is not valid JSON") from exc
    expected = {
        "prepared_schema": PREPARED_PANEL_SCHEMA,
        "panel_registration_sha": panel_registration_sha(profile=profile, family=family),
        "config_fingerprint": panel_config_fingerprint(profile=profile, family=family),
        "source_commit": prepared_commit,
        "source_audit_fingerprint": audit["audit_fingerprint"],
        "artifact": artifact.name,
        "artifact_sha256": sha256_file(artifact),
        "construction": construction.name,
        "construction_sha256": sha256_file(construction),
        "item_count": len(profile.question_ids),
        "qids": list(profile.question_ids),
    }
    if manifest != expected:
        raise RuntimeError("prepared FanOutQA manifest does not match its panel or bytes")
    bundle = _load_bundle(artifact)
    if set(bundle) != {"schema", "registration", "construction_sha256", "items"}:
        raise RuntimeError("prepared FanOutQA artifact has an invalid schema")
    registration_value = bundle["registration"]
    registration = (
        cast(dict[str, Any], registration_value) if type(registration_value) is dict else None
    )
    if (
        bundle["schema"] != PREPARED_PANEL_SCHEMA
        or registration is None
        or identity.fingerprint(
            registration,
            identity.json_compact_legacy,
            digest_chars=16,
        )
        != panel_registration_sha(profile=profile, family=family)
        or bundle["construction_sha256"] != manifest["construction_sha256"]
    ):
        raise RuntimeError("prepared FanOutQA artifact registration has drifted")
    items = _validate_items(bundle["items"], profile=profile)
    if shard_fingerprints(items) != audit["shards"]:
        raise RuntimeError("prepared FanOutQA artifact shards differ from the sealed source audit")
    source_identities = tuple(
        (item.qid, pageid, revid)
        for item in items
        for pageid, revid, _title in item.question_obj.pages
    )
    if (
        validated_source_ledger_sha256(construction, source_identities)
        != manifest["construction_sha256"]
    ):
        raise RuntimeError("prepared FanOutQA construction ledger hash has drifted")
    if family.native_sender_prompts:
        _validate_native_prompts_once(
            items,
            artifact=artifact,
            construction=construction,
            manifest=expected,
            family=family,
            profile=profile,
        )
    return items, manifest


def _validate_native_prompts_once(
    items: tuple[ProbeItem, ...],
    *,
    artifact: Path,
    construction: Path,
    manifest: Mapping[str, Any],
    family: RouteFamily,
    profile: BenchmarkProfile,
) -> None:
    """Recheck the native prompts once per set of bytes rather than once per process."""
    from rcc.models.qwen.native_prompts import codec_identities, validate_native_prompts

    record = artifact.with_name(f"{artifact.name}.native-prompts.json")
    expected = {
        "schema": NATIVE_PROMPT_RECORD_SCHEMA,
        "record_sha256": identity.fingerprint(
            {
                "artifact_sha256": manifest["artifact_sha256"],
                "construction_sha256": manifest["construction_sha256"],
                "panel_registration_sha": manifest["panel_registration_sha"],
                "model_id": family.model_id,
                "codecs": codec_identities(family, profile),
            },
            identity.json_compact_legacy,
        ),
    }
    try:
        if json.loads(record.read_text(encoding="utf-8")) == expected:
            return
    except (OSError, UnicodeError, json.JSONDecodeError):
        pass
    validate_native_prompts(items, construction, family=family, profile=profile)
    try:
        atomic_json(record, expected)
    except OSError:
        pass  # on a read-only tree the check simply runs on every load


__all__ = (
    "NATIVE_PROMPT_RECORD_SCHEMA",
    "PREPARED_PANEL_SCHEMA",
    "SEALED_PREPARED_SOURCE_COMMIT",
    "load_prepared_panel",
    "prepared_source_commit",
)
