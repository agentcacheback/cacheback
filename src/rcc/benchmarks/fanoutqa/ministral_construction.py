"""The Ministral construction manifests: what one item's sender cell was built from.

A manifest carries the item's worker hashes and its post-packing answerability
audit under a fingerprint, so a result row binds back to its construction.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50, REGISTERED_PROFILES
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.ministral.text_prompt import registered_text_senders
from rcc.run.io import canonical_sha
from rcc.topologies.fanout import FANOUT_M3

MINISTRAL_CONSTRUCTION_MANIFEST_SCHEMA = "ministral3-source-construction-receipt-v1"
MINISTRAL_CONSTRUCTION_ROSTER_SCHEMA = "ministral3-construction-receipt-roster-v1"

_AUDIT_COUNT_FIELDS = {
    "n_gold_leaves",
    "n_question_given",
    "n_off_question_leaves",
    "n_locatable_in_full_source",
    "n_survives_construction",
    "n_removed_by_construction",
    "n_absent_from_full_source",
}
_AUDIT_LEAF_FIELDS = {
    "leaf",
    "in_question",
    "in_full_source",
    "in_retained_source",
    "status",
}


def _profile_for_source(source_identity: str) -> BenchmarkProfile:
    for profile in REGISTERED_PROFILES:
        if source_identity == profile.source_logical_fingerprint:
            return profile
    raise ValueError("Ministral construction manifest has an unregistered source identity")


def canonical_ministral_qids(
    values: Sequence[str],
    *,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> tuple[str, ...]:
    """Return the selected question ids, raising unless they follow the panel order."""
    qids = tuple(str(value) for value in values)
    if not qids or len(set(qids)) != len(qids):
        raise ValueError("Ministral prepared panel requires unique selected question ids")
    selected = set(qids)
    canonical = tuple(qid for qid in profile.question_ids if qid in selected)
    if canonical != qids:
        raise ValueError("Ministral prepared panel question ids differ from sealed order")
    return qids


def _validated_audit_leaf(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Ministral construction manifest leaf audit is malformed")
    row = cast(dict[str, object], value)
    if set(row) != _AUDIT_LEAF_FIELDS:
        raise ValueError("Ministral construction manifest leaf audit is malformed")
    if not isinstance(row["leaf"], str) or any(
        type(row[name]) is not bool
        for name in ("in_question", "in_full_source", "in_retained_source")
    ):
        raise ValueError("Ministral construction manifest leaf audit is malformed")
    if row["in_retained_source"] and not row["in_full_source"]:
        raise ValueError("Ministral construction manifest leaf source identity is impossible")
    expected = (
        "question_given"
        if row["in_question"]
        else "absent_from_full_source"
        if not row["in_full_source"]
        else "survives_construction"
        if row["in_retained_source"]
        else "removed_by_construction"
    )
    if row["status"] != expected:
        raise ValueError("Ministral construction manifest leaf status drifted")
    return row


def _derived_audit_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    off_question = [row for row in rows if not row["in_question"]]
    return {
        "n_gold_leaves": len(rows),
        "n_question_given": len(rows) - len(off_question),
        "n_off_question_leaves": len(off_question),
        "n_locatable_in_full_source": sum(bool(row["in_full_source"]) for row in off_question),
        "n_survives_construction": sum(
            row["status"] == "survives_construction" for row in off_question
        ),
        "n_removed_by_construction": sum(
            row["status"] == "removed_by_construction" for row in off_question
        ),
        "n_absent_from_full_source": sum(
            row["status"] == "absent_from_full_source" for row in off_question
        ),
    }


def _validate_construction_audit_shape(audit: Mapping[str, object]) -> None:
    if set(audit) != {*_AUDIT_COUNT_FIELDS, "leaves"} or any(
        isinstance(audit[name], bool) or not isinstance(audit[name], int)
        for name in _AUDIT_COUNT_FIELDS
    ):
        raise ValueError("Ministral construction manifest audit is malformed")
    leaves = audit["leaves"]
    if not isinstance(leaves, list):
        raise ValueError("Ministral construction manifest leaf audit is malformed")


def _validate_construction_audit(audit: Mapping[str, object]) -> None:
    _validate_construction_audit_shape(audit)
    leaves = cast(list[object], audit["leaves"])
    rows = tuple(_validated_audit_leaf(value) for value in leaves)
    if any(audit[name] != value for name, value in _derived_audit_counts(rows).items()):
        raise ValueError("Ministral construction manifest audit counts drifted")


@dataclass(frozen=True)
class MinistralConstructionManifest:
    """One item and sender arm's worker hashes and post-packing answerability audit."""

    qid: str
    semantic_arm: str
    retained_source_sha256: tuple[str, ...]
    audit: Mapping[str, object]
    fingerprint: str
    source_identity: str = FANOUTQA_NATURAL_DEV50.source_logical_fingerprint

    def __post_init__(self) -> None:
        """Raise for incomplete worker hashes, a malformed audit, or a stale fingerprint."""
        profile = _profile_for_source(self.source_identity)
        if self.qid not in profile.question_ids:
            raise ValueError("Ministral construction manifest has an unregistered question")
        if self.semantic_arm not in {sender.semantic_arm for sender in registered_text_senders()}:
            raise ValueError("Ministral construction manifest has an unregistered sender lane")
        if len(self.retained_source_sha256) != FANOUT_M3.workers_per_item or any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in self.retained_source_sha256
        ):
            raise ValueError("Ministral construction manifest lacks three full source hashes")
        _validate_construction_audit_shape(self.audit)
        if self.fingerprint != canonical_sha(self.unsigned_identity()):
            raise ValueError("Ministral construction manifest fingerprint drifted")
        _validate_construction_audit(self.audit)

    @property
    def construction_complete(self) -> bool:
        """Return whether packing kept every locatable off-question leaf."""
        return self.audit["n_removed_by_construction"] == 0

    def unsigned_identity(self) -> dict[str, object]:
        """Return the manifest fields the fingerprint is taken over."""
        return {
            "schema": MINISTRAL_CONSTRUCTION_MANIFEST_SCHEMA,
            "qid": self.qid,
            "semantic_arm": self.semantic_arm,
            "source_identity": self.source_identity,
            "page_allocation": "max_min_query",
            "retained_source_sha256": list(self.retained_source_sha256),
            "audit": dict(self.audit),
            "construction_complete": self.construction_complete,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the manifest as JSON-compatible fields, fingerprint included."""
        return {**self.unsigned_identity(), "fingerprint": self.fingerprint}


def make_ministral_construction_manifest(
    *,
    qid: str,
    semantic_arm: str,
    retained_sources: Sequence[str],
    audit: Mapping[str, object],
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> MinistralConstructionManifest:
    """Build one manifest from an item's retained sources and its answerability audit."""
    hashes = tuple(hashlib.sha256(value.encode()).hexdigest() for value in retained_sources)
    unsigned = {
        "schema": MINISTRAL_CONSTRUCTION_MANIFEST_SCHEMA,
        "qid": qid,
        "semantic_arm": semantic_arm,
        "source_identity": profile.source_logical_fingerprint,
        "page_allocation": "max_min_query",
        "retained_source_sha256": list(hashes),
        "audit": dict(audit),
        "construction_complete": audit.get("n_removed_by_construction") == 0,
    }
    return MinistralConstructionManifest(
        qid=qid,
        semantic_arm=semantic_arm,
        retained_source_sha256=hashes,
        audit=dict(audit),
        fingerprint=canonical_sha(unsigned),
        source_identity=profile.source_logical_fingerprint,
    )


def ministral_construction_roster_fingerprint(
    qids: Sequence[str],
    construction_manifests: Sequence[MinistralConstructionManifest],
) -> str:
    """Fingerprint one manifest per item and sender cell, in panel order."""
    if not construction_manifests:
        raise ValueError("Ministral construction manifest roster is empty")
    profile = _profile_for_source(construction_manifests[0].source_identity)
    if any(
        manifest.source_identity != profile.source_logical_fingerprint
        for manifest in construction_manifests
    ):
        raise ValueError("Ministral construction manifest roster mixes source identities")
    selected = canonical_ministral_qids(qids, profile=profile)
    expected = tuple(
        (qid, sender.semantic_arm) for qid in selected for sender in registered_text_senders()
    )
    observed = tuple((manifest.qid, manifest.semantic_arm) for manifest in construction_manifests)
    fingerprints = tuple(manifest.fingerprint for manifest in construction_manifests)
    if observed != expected:
        raise ValueError("Ministral construction manifest roster differs from canonical order")
    if len(set(fingerprints)) != len(expected):
        raise ValueError("Ministral construction manifest roster contains duplicate manifests")
    return canonical_sha(
        {
            "schema": MINISTRAL_CONSTRUCTION_ROSTER_SCHEMA,
            "cells": [
                {
                    "qid": manifest.qid,
                    "semantic_arm": manifest.semantic_arm,
                    "fingerprint": manifest.fingerprint,
                }
                for manifest in construction_manifests
            ],
        }
    )


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


#: The source fingerprint of every panel; a manifest naming any other source
#: raises.
_REGISTERED_SOURCE_IDENTITIES = frozenset(
    profile.source_logical_fingerprint for profile in REGISTERED_PROFILES
)


def parse_ministral_construction_manifest(value: object) -> MinistralConstructionManifest:
    """Parse one construction manifest and rehash its fingerprint."""
    row = _mapping(value, label="Ministral construction manifest")
    expected = {
        "schema",
        "qid",
        "semantic_arm",
        "source_identity",
        "page_allocation",
        "retained_source_sha256",
        "audit",
        "construction_complete",
        "fingerprint",
    }
    if set(row) != expected:
        raise RuntimeError("Ministral construction manifest fields differ from the schema")
    audit = _mapping(row["audit"], label="Ministral construction audit")
    if (
        row["schema"] != MINISTRAL_CONSTRUCTION_MANIFEST_SCHEMA
        or row["source_identity"] not in _REGISTERED_SOURCE_IDENTITIES
        or row["page_allocation"] != "max_min_query"
        or not isinstance(row["qid"], str)
        or not isinstance(row["semantic_arm"], str)
        or not isinstance(row["fingerprint"], str)
    ):
        raise RuntimeError("Ministral construction manifest identity drifted")
    manifest = MinistralConstructionManifest(
        qid=row["qid"],
        semantic_arm=row["semantic_arm"],
        retained_source_sha256=_string_list(
            row["retained_source_sha256"],
            label="retained_source_sha256",
        ),
        audit=dict(audit),
        fingerprint=row["fingerprint"],
        source_identity=cast(str, row["source_identity"]),
    )
    if row["construction_complete"] is not manifest.construction_complete:
        raise RuntimeError("Ministral construction-complete identity drifted")
    return manifest


def validate_ministral_result_construction(
    row: Mapping[str, object],
    *,
    qid: str,
    source_arm: str,
) -> MinistralConstructionManifest:
    """Raise unless one result row names the construction manifest it was built from."""
    manifest = parse_ministral_construction_manifest(row.get("construction_manifest"))
    expected_reason = "" if manifest.construction_complete else "leaves_removed_by_packing"
    if (manifest.qid, manifest.semantic_arm) != (qid, source_arm):
        raise RuntimeError("Ministral construction manifest cell identity drifted")
    if row.get("construction_manifest_fingerprint") != manifest.fingerprint:
        raise RuntimeError("Ministral construction manifest fingerprint is not cross-bound")
    if row.get("construction_complete") is not manifest.construction_complete:
        raise RuntimeError("Ministral result construction-complete identity drifted")
    if row.get("construction_incomplete_reason") != expected_reason:
        raise RuntimeError("Ministral result construction reason drifted")
    return manifest


__all__ = (
    "MINISTRAL_CONSTRUCTION_MANIFEST_SCHEMA",
    "MINISTRAL_CONSTRUCTION_ROSTER_SCHEMA",
    "MinistralConstructionManifest",
    "canonical_ministral_qids",
    "make_ministral_construction_manifest",
    "ministral_construction_roster_fingerprint",
    "parse_ministral_construction_manifest",
    "validate_ministral_result_construction",
)
