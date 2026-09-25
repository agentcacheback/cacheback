"""Where a FanOutQA panel's built bytes live, and the check that they are its own.

The source audit and its construction ledger are rehashed against the profile
asked for, and any difference raises.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

from rcc.benchmarks.fanoutqa.panel import load_questions
from rcc.benchmarks.fanoutqa.qwen_serving import panel_policy_config, policy_config_fingerprint
from rcc.benchmarks.fanoutqa.source_planning import validated_source_ledger_sha256
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.qwen import QWEN_FAMILY
from rcc.models.route import RouteFamily

_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def panel_config_fingerprint(
    *, profile: BenchmarkProfile, family: RouteFamily = QWEN_FAMILY
) -> str:
    """Return the policy-config fingerprint one panel's prepared bytes carry.

    Computed from the family's own registration, and on the Qwen lane compared with
    the profile's pinned value, so a registration change raises here.
    """
    computed = policy_config_fingerprint(panel_policy_config(profile, family=family))
    if family.lane == "qwen":
        sealed = profile.prepared_config_fingerprint
        if computed != sealed:
            raise RuntimeError("sealed Qwen policy-config fingerprint drifted")
    return computed


def prepared_root(run_root: Path, profile: BenchmarkProfile) -> Path:
    """Return the subtree one profile's built bytes own inside a run root.

    Each profile gets a subtree keyed by profile id, since two profiles sharing a run
    root would otherwise overwrite each other's bytes.
    """
    return run_root / "prepared" / "profiles" / profile.profile_id


def panel_audit_paths(run_root: Path, profile: BenchmarkProfile) -> tuple[Path, Path]:
    """Return one profile's source-audit ledger and manifest under a run root."""
    root = prepared_root(run_root, profile) / "source_audit"
    return root / "construction.jsonl", root / "audit.json"


def panel_prepared_paths(run_root: Path, profile: BenchmarkProfile) -> tuple[Path, Path, Path]:
    """Return one profile's prepared artifact, manifest, and construction ledger."""
    root = prepared_root(run_root, profile)
    artifact = root / "items.pkl"
    return artifact, artifact.with_suffix(".manifest.json"), root / "construction.jsonl"


def source_cache_root(run_root: Path) -> Path:
    """Return the profile-independent pinned page and question-index cache."""
    return run_root / "prepared" / "source_cache"


def validate_source_commit(value: str) -> str:
    """Return the value if it is a full lowercase Git commit hash, else raise."""
    if _COMMIT.fullmatch(value) is None:
        raise ValueError("FanOutQA source commit must be exactly 40 lowercase hex characters")
    return value


def expected_source_identities(
    run_root: Path,
    *,
    profile: BenchmarkProfile,
) -> tuple[tuple[str, int, int], ...]:
    """Return the (qid, pageid, revid) roster the pinned question index names."""
    dev_path = source_cache_root(run_root) / "fanout-final-dev.json"
    if not dev_path.is_file():
        raise RuntimeError("M=3 source audit is missing the pinned FanOutQA question index")
    # The profile pins the index it was built over, in place of the freeze file.
    pin = {"dev_json_sha256": profile.question_index_sha256}
    questions = {
        question.qid: question for question in load_questions(dev_path, freeze_loader=lambda: pin)
    }
    try:
        return tuple(
            (qid, pageid, revid)
            for qid in profile.question_ids
            for pageid, revid, _title in questions[qid].pages
        )
    except KeyError as exc:
        raise RuntimeError("M=3 source audit question index is missing a frozen item") from exc


def validate_source_audit(
    run_root: Path,
    *,
    source_commit: str,
    profile: BenchmarkProfile,
    family: RouteFamily = QWEN_FAMILY,
) -> dict[str, Any]:
    """Validate one profile's source audit under a run root."""
    from rcc.benchmarks.fanoutqa.natural_panel import validate_natural_source_audit

    ledger, path = panel_audit_paths(run_root, profile)
    if not ledger.is_file() or not path.is_file():
        raise RuntimeError("M=3 source audit must run before panel preparation")
    try:
        raw_audit: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("natural source audit bytes are not valid JSON") from exc
    if not isinstance(raw_audit, dict):
        raise RuntimeError("natural source audit must be a JSON object")
    ledger_sha256 = validated_source_ledger_sha256(
        ledger, expected_source_identities(run_root, profile=profile)
    )
    return validate_natural_source_audit(
        cast(dict[str, Any], raw_audit),
        source_commit=validate_source_commit(source_commit),
        config_fingerprint=panel_config_fingerprint(profile=profile, family=family),
        ledger_sha256=ledger_sha256,
        profile=profile,
    )


__all__ = (
    "expected_source_identities",
    "panel_audit_paths",
    "panel_config_fingerprint",
    "panel_prepared_paths",
    "prepared_root",
    "source_cache_root",
    "validate_source_audit",
    "validate_source_commit",
)
