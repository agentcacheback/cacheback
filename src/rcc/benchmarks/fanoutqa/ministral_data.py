"""The Ministral prompt panel: every item tokenized for all three sender lanes.

The panel is accepted on load only when its source, roster, geometry, and row
hashes all agree.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50, REGISTERED_PROFILES
from rcc.benchmarks.fanoutqa.ministral_construction import (
    MinistralConstructionManifest as _MinistralConstructionManifest,
)
from rcc.benchmarks.fanoutqa.ministral_construction import (
    canonical_ministral_qids as _canonical_ministral_qids,
)
from rcc.benchmarks.fanoutqa.ministral_construction import (
    ministral_construction_roster_fingerprint as _construction_roster_fingerprint,
)
from rcc.benchmarks.fanoutqa.ministral_construction import (
    parse_ministral_construction_manifest as _parse_construction_manifest,
)
from rcc.benchmarks.fanoutqa.panel import load_questions
from rcc.benchmarks.fanoutqa.source_bundle import validate_shared_source_bundle
from rcc.benchmarks.fanoutqa.source_pages import atomic_write_text
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.ministral.text_codec import (
    MinistralTextCodec,
    prompt_token_sha256,
)
from rcc.models.ministral.text_prompt import (
    MinistralPreparedPromptArtifact,
    MinistralTextSender,
    prepared_prompt_fingerprint,
    registered_text_senders,
)
from rcc.run.io import canonical_sha
from rcc.topologies.fanout import FANOUT_M3

MINISTRAL_PREPARED_PANEL_SCHEMA = "ministral3-fanoutqa-prepared-panel-v1"

EvidenceByArm = Mapping[str, Mapping[str, Sequence[Sequence[int]]]]
ConstructionByArm = Mapping[str, Mapping[str, _MinistralConstructionManifest]]


def _prompt_tokens(value: Sequence[object], *, label: str) -> tuple[int, ...]:
    tokens: list[int] = []
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(f"{label}: prompt token ids are malformed")
        tokens.append(raw)
    return tuple(tokens)


def _artifact(
    *,
    qid: str,
    semantic_arm: str,
    source_identity: str,
    checkpoint: str,
    revision: str,
    prompts: tuple[tuple[int, ...], ...],
    profile: BenchmarkProfile,
) -> MinistralPreparedPromptArtifact:
    hashes = tuple(prompt_token_sha256(prompt) for prompt in prompts)
    unsigned = SimpleNamespace(
        qid=qid,
        semantic_arm=semantic_arm,
        source_identity=source_identity,
        checkpoint=checkpoint,
        revision=revision,
        prompt_sha256=hashes,
    )
    fingerprint = prepared_prompt_fingerprint(cast(Any, unsigned))
    return MinistralPreparedPromptArtifact(
        qid=qid,
        semantic_arm=semantic_arm,
        source_identity=source_identity,
        checkpoint=checkpoint,
        revision=revision,
        prompt_ids_by_worker=prompts,
        prompt_sha256=hashes,
        fingerprint=fingerprint,
        profile=profile,
    )


def _artifact_row(artifact: MinistralPreparedPromptArtifact) -> dict[str, object]:
    return {
        "qid": artifact.qid,
        "semantic_arm": artifact.semantic_arm,
        "source_identity": artifact.source_identity,
        "checkpoint": artifact.checkpoint,
        "revision": artifact.revision,
        "prompt_ids_by_worker": [list(row) for row in artifact.prompt_ids_by_worker],
        "prompt_sha256": list(artifact.prompt_sha256),
        "fingerprint": artifact.fingerprint,
    }


def _panel_identity(
    qids: Sequence[str],
    artifacts: Sequence[MinistralPreparedPromptArtifact],
    construction_manifests: Sequence[_MinistralConstructionManifest],
    profile: BenchmarkProfile,
) -> dict[str, object]:
    return {
        "schema": MINISTRAL_PREPARED_PANEL_SCHEMA,
        "benchmark_profile": profile.profile_id,
        "source_identity": profile.source_logical_fingerprint,
        "source_manifest_sha256": profile.source_manifest_sha256,
        "qids": list(qids),
        "workers_per_item": FANOUT_M3.workers_per_item,
        "senders": [sender.to_dict() for sender in registered_text_senders()],
        "prompt_rows": len(qids) * len(registered_text_senders()) * FANOUT_M3.workers_per_item,
        "construction_manifest_roster_fingerprint": (
            _construction_roster_fingerprint(qids, construction_manifests)
        ),
        "cells": [
            {
                "qid": artifact.qid,
                "semantic_arm": artifact.semantic_arm,
                "prompt_sha256": list(artifact.prompt_sha256),
                "fingerprint": artifact.fingerprint,
            }
            for artifact in artifacts
        ],
        "construction_cells": [manifest.to_dict() for manifest in construction_manifests],
    }


@dataclass(frozen=True)
class MinistralPreparedPromptPanel:
    """Every selected item tokenized for all three sender lanes."""

    qids: tuple[str, ...]
    artifacts: tuple[MinistralPreparedPromptArtifact, ...]
    construction_manifests: tuple[_MinistralConstructionManifest, ...]
    fingerprint: str
    benchmark_profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50

    def __post_init__(self) -> None:
        """Raise on any difference in the source, pins, roster, geometry, or hashes."""
        qids = _canonical_ministral_qids(self.qids, profile=self.benchmark_profile)
        senders = registered_text_senders()
        expected = tuple((qid, sender.semantic_arm) for qid in qids for sender in senders)
        observed = tuple((artifact.qid, artifact.semantic_arm) for artifact in self.artifacts)
        if observed != expected:
            raise ValueError("Ministral prepared prompt cells differ from the selected roster")
        manifest_cells = tuple(
            (manifest.qid, manifest.semantic_arm) for manifest in self.construction_manifests
        )
        if manifest_cells != expected:
            raise ValueError("Ministral construction manifests differ from the selected roster")
        if any(
            artifact.source_identity != self.benchmark_profile.source_logical_fingerprint
            or manifest.source_identity != self.benchmark_profile.source_logical_fingerprint
            or len(artifact.prompt_ids_by_worker) != FANOUT_M3.workers_per_item
            or not self.benchmark_profile.admits_worker_prompt_tokens(
                [len(row) for row in artifact.prompt_ids_by_worker]
            )
            for artifact, manifest in zip(self.artifacts, self.construction_manifests, strict=True)
        ):
            raise ValueError("Ministral prepared prompt panel source or geometry drifted")
        measured = canonical_sha(
            _panel_identity(
                qids,
                self.artifacts,
                self.construction_manifests,
                self.benchmark_profile,
            )
        )
        if self.fingerprint != measured:
            raise ValueError("Ministral prepared prompt panel fingerprint drifted")

    @property
    def prompt_rows(self) -> int:
        """Return the row count: selected items by senders by workers."""
        return len(self.artifacts) * FANOUT_M3.workers_per_item

    @property
    def construction_manifest_roster_fingerprint(self) -> str:
        """Return the fingerprint over the ordered construction-manifest roster."""
        return _construction_roster_fingerprint(
            self.qids,
            self.construction_manifests,
        )

    def artifact(self, qid: str, semantic_arm: str) -> MinistralPreparedPromptArtifact:
        """Return one item and sender cell, raising when the panel has no such cell."""
        matches = [
            artifact
            for artifact in self.artifacts
            if (artifact.qid, artifact.semantic_arm) == (qid, semantic_arm)
        ]
        if len(matches) != 1:
            raise KeyError(f"unregistered Ministral prompt cell {qid}/{semantic_arm}")
        return matches[0]

    def construction_manifest(
        self,
        qid: str,
        semantic_arm: str,
    ) -> _MinistralConstructionManifest:
        """Return one item and sender cell's construction manifest."""
        matches = [
            manifest
            for manifest in self.construction_manifests
            if (manifest.qid, manifest.semantic_arm) == (qid, semantic_arm)
        ]
        if len(matches) != 1:
            raise KeyError(f"unregistered Ministral construction cell {qid}/{semantic_arm}")
        return matches[0]

    def to_dict(self) -> dict[str, object]:
        """Return the whole panel as JSON-compatible fields, prompt token ids included."""
        return {
            **_panel_identity(
                self.qids,
                self.artifacts,
                self.construction_manifests,
                self.benchmark_profile,
            ),
            "prompt_cells": [_artifact_row(artifact) for artifact in self.artifacts],
            "fingerprint": self.fingerprint,
        }


def _prepare_prompt_cell(
    *,
    qid: str,
    question: str,
    qids: Sequence[str],
    sender: MinistralTextSender,
    codec: MinistralTextCodec,
    evidence_by_qid: Mapping[str, Sequence[Sequence[int]]],
    manifests_by_qid: Mapping[str, _MinistralConstructionManifest],
    profile: BenchmarkProfile,
) -> tuple[MinistralPreparedPromptArtifact, _MinistralConstructionManifest]:
    codec.require_verified()
    if (codec.checkpoint, codec.revision) != (sender.checkpoint, sender.revision):
        raise RuntimeError(f"{sender.semantic_arm}: prompt codec pin drifted")
    if set(evidence_by_qid) != set(qids) or set(manifests_by_qid) != set(qids):
        raise ValueError(f"{sender.semantic_arm}: evidence question roster drifted")
    manifest = manifests_by_qid[qid]
    if (manifest.qid, manifest.semantic_arm) != (qid, sender.semantic_arm):
        raise ValueError(f"{qid}/{sender.semantic_arm}: construction manifest drifted")
    raw_rows = tuple(evidence_by_qid[qid])
    if len(raw_rows) != FANOUT_M3.workers_per_item:
        raise ValueError(f"{qid}/{sender.semantic_arm}: evidence does not cover M=3")
    retained_hashes = tuple(
        hashlib.sha256(str(codec.tokenizer.decode(list(row))).encode()).hexdigest()
        for row in raw_rows
    )
    if retained_hashes != manifest.retained_source_sha256:
        raise RuntimeError(f"{qid}/{sender.semantic_arm}: construction source hashes drifted")
    prompts = tuple(
        codec.encode_worker_prompt(
            question,
            _prompt_tokens(row, label=f"{qid}/{sender.semantic_arm}/w{worker}"),
        )
        for worker, row in enumerate(raw_rows)
    )
    if not profile.admits_worker_prompt_tokens([len(prompt) for prompt in prompts]):
        raise RuntimeError(
            f"{qid}/{sender.semantic_arm}: worker prompt widths differ from the registered "
            f"{profile.prompt_geometry} {profile.worker_prompt_tokens}"
        )
    artifact = _artifact(
        qid=qid,
        semantic_arm=sender.semantic_arm,
        source_identity=profile.source_logical_fingerprint,
        checkpoint=sender.checkpoint,
        revision=sender.revision,
        prompts=prompts,
        profile=profile,
    )
    return artifact, manifest


def prepare_ministral_prompt_panel(
    bundle_root: Path,
    *,
    selected_qids: Sequence[str],
    codecs: Mapping[str, MinistralTextCodec],
    evidence_ids_by_arm: EvidenceByArm,
    construction_by_arm: ConstructionByArm,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> MinistralPreparedPromptPanel:
    """Build every sender prompt, after rehashing the source bundle."""
    source = validate_shared_source_bundle(Path(bundle_root), profile=profile)
    qids = _canonical_ministral_qids(selected_qids, profile=profile)
    if (
        source.get("fingerprint") != profile.source_logical_fingerprint
        or source.get("manifest_sha256") != profile.source_manifest_sha256
        or tuple(map(str, cast(Sequence[object], source.get("qids") or ()))) != profile.question_ids
    ):
        raise RuntimeError("Ministral source bundle differs from the registered panel")
    questions = {
        question.qid: question.question
        for question in load_questions(Path(bundle_root) / "source_cache" / "fanout-final-dev.json")
    }
    if any(qid not in questions for qid in qids):
        raise RuntimeError("Ministral source bundle omits a selected question")
    senders = registered_text_senders()
    arm_names = {sender.semantic_arm for sender in senders}
    if (
        set(codecs) != arm_names
        or set(evidence_ids_by_arm) != arm_names
        or set(construction_by_arm) != arm_names
    ):
        raise ValueError("Ministral prompt preparation requires exactly three sender lanes")

    artifacts: list[MinistralPreparedPromptArtifact] = []
    manifests: list[_MinistralConstructionManifest] = []
    for qid in qids:
        for sender in senders:
            artifact, manifest = _prepare_prompt_cell(
                qid=qid,
                question=questions[qid],
                qids=qids,
                sender=sender,
                codec=codecs[sender.semantic_arm],
                evidence_by_qid=evidence_ids_by_arm[sender.semantic_arm],
                manifests_by_qid=construction_by_arm[sender.semantic_arm],
                profile=profile,
            )
            artifacts.append(artifact)
            manifests.append(manifest)
    fingerprint = canonical_sha(_panel_identity(qids, artifacts, manifests, profile))
    return MinistralPreparedPromptPanel(
        qids,
        tuple(artifacts),
        tuple(manifests),
        fingerprint,
        profile,
    )


def write_ministral_prompt_panel(path: Path, panel: MinistralPreparedPromptPanel) -> None:
    """Write one complete prompt panel atomically."""
    panel.__post_init__()
    encoded = json.dumps(panel.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, encoded + "\n")


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return cast(dict[str, object], value)


def _string_list(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} is not a string list")
    items = cast(list[object], value)
    if any(not isinstance(item, str) for item in items):
        raise RuntimeError(f"{label} is not a string list")
    return tuple(cast(list[str], items))


def _load_artifact(value: object, profile: BenchmarkProfile) -> MinistralPreparedPromptArtifact:
    row = _mapping(value, label="Ministral prompt cell")
    expected = {
        "qid",
        "semantic_arm",
        "source_identity",
        "checkpoint",
        "revision",
        "prompt_ids_by_worker",
        "prompt_sha256",
        "fingerprint",
    }
    if set(row) != expected or not isinstance(row["prompt_ids_by_worker"], list):
        raise RuntimeError("Ministral prompt cell fields differ from the schema")
    prompt_rows = cast(list[object], row["prompt_ids_by_worker"])
    prompts: list[tuple[int, ...]] = []
    for worker, raw in enumerate(prompt_rows):
        if not isinstance(raw, list):
            raise RuntimeError("Ministral prompt cell token rows are malformed")
        prompts.append(
            _prompt_tokens(cast(list[Any], raw), label=f"loaded Ministral prompt w{worker}")
        )
    string_fields = ("qid", "semantic_arm", "source_identity", "checkpoint", "revision")
    if any(not isinstance(row[name], str) for name in string_fields):
        raise RuntimeError("Ministral prompt cell identity fields are malformed")
    return MinistralPreparedPromptArtifact(
        qid=cast(str, row["qid"]),
        semantic_arm=cast(str, row["semantic_arm"]),
        source_identity=cast(str, row["source_identity"]),
        checkpoint=cast(str, row["checkpoint"]),
        revision=cast(str, row["revision"]),
        prompt_ids_by_worker=tuple(prompts),
        prompt_sha256=_string_list(row["prompt_sha256"], label="prompt_sha256"),
        fingerprint=cast(str, row["fingerprint"]),
        profile=profile,
    )


def load_ministral_prompt_panel(path: Path) -> MinistralPreparedPromptPanel:
    """Load one prepared prompt panel and rehash it."""
    try:
        decoded: object = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Ministral prepared prompt panel is unreadable") from exc
    payload = _mapping(decoded, label="Ministral prepared prompt panel")
    expected = {
        "schema",
        "benchmark_profile",
        "source_identity",
        "source_manifest_sha256",
        "qids",
        "workers_per_item",
        "senders",
        "prompt_rows",
        "construction_manifest_roster_fingerprint",
        "cells",
        "construction_cells",
        "prompt_cells",
        "fingerprint",
    }
    if set(payload) != expected:
        raise RuntimeError("Ministral prepared prompt panel fields differ from the schema")
    senders = [sender.to_dict() for sender in registered_text_senders()]
    profiles = {profile.profile_id: profile for profile in REGISTERED_PROFILES}
    profile = profiles.get(str(payload["benchmark_profile"]))
    if (
        payload["schema"] != MINISTRAL_PREPARED_PANEL_SCHEMA
        or profile is None
        or payload["source_identity"] != profile.source_logical_fingerprint
        or payload["source_manifest_sha256"] != profile.source_manifest_sha256
        or payload["workers_per_item"] != FANOUT_M3.workers_per_item
        or payload["senders"] != senders
        or not isinstance(payload["prompt_cells"], list)
        or not isinstance(payload["construction_cells"], list)
        or not isinstance(payload["fingerprint"], str)
    ):
        raise RuntimeError("Ministral prepared prompt panel identity or pins drifted")
    qids = _string_list(payload["qids"], label="Ministral panel qids")
    artifacts = tuple(
        _load_artifact(row, profile) for row in cast(list[object], payload["prompt_cells"])
    )
    manifests = tuple(
        _parse_construction_manifest(row)
        for row in cast(list[object], payload["construction_cells"])
    )
    assert profile is not None
    panel = MinistralPreparedPromptPanel(
        qids, artifacts, manifests, payload["fingerprint"], profile
    )
    identity = _panel_identity(
        panel.qids,
        panel.artifacts,
        panel.construction_manifests,
        panel.benchmark_profile,
    )
    if (
        payload["cells"] != identity["cells"]
        or payload["prompt_rows"] != panel.prompt_rows
        or payload["construction_manifest_roster_fingerprint"]
        != panel.construction_manifest_roster_fingerprint
    ):
        raise RuntimeError("Ministral prepared prompt panel signed roster drifted")
    return panel


__all__ = (
    "MINISTRAL_PREPARED_PANEL_SCHEMA",
    "MinistralPreparedPromptPanel",
    "load_ministral_prompt_panel",
    "prepare_ministral_prompt_panel",
    "write_ministral_prompt_panel",
)
