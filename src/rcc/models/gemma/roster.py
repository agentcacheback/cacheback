"""The shared FanOutQA arm roster over the resident Gemma routes.

It binds each shared semantic arm to the physical route that runs it, and
translates between the two namings wherever a row or a path crosses.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from rcc.benchmarks.fanoutqa import (
    FANOUTQA_NATURAL_DEV50,
    arms_with_full,
    shared_answer_seeds,
)
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.gemma import GEMMA
from rcc.models.gemma.contract import REGISTERED_ARM_POLICIES
from rcc.run.barrier import SPLIT_FLEET_ISOLATION_PROFILE
from rcc.topologies.fanout.layout import FLAT_INTERLEAVE_LAYOUT

SHARED_SOURCE_VERSION = "fanoutqa-m3-source-cache-v1"
SHARED_EXECUTION_ROSTER_SCHEMA = "gemma-shared-fanoutqa-execution-roster-v6"
NATIVE_NATURAL_ARM_ORDER = (
    "latent_query_support_r8",
    "text_primary",
    "issue_only",
    "text_medium",
    "text_small",
    *(f"latent_query_support_r{ratio}" for ratio in (2, 4, 16, 32, 64, 128)),
)
SHARED_SELECTION_ARTIFACT_PROFILE = "execution-arm-keyed-no-unused-audition-v1"
#: Every shared-roster result row is banked under one panel name, whatever
#: physical panel the worker loaded its items from.
SHARED_RESULT_PANEL = "production"

_GEMMA_TEXT_SENDERS = {
    "text_primary": (
        "google/gemma-4-12B-it",
        "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
    ),
    "text_medium": (
        "google/gemma-4-E4B-it",
        "ee0ef6023621cff504d758262d4e04895a5af4a2",
    ),
    "text_small": (
        "google/gemma-4-E2B-it",
        "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    ),
}
_GEMMA_TEXT_NATIVE_MAX_MODEL_LEN = {
    "text_primary": 262_144,
    "text_medium": 131_072,
    "text_small": 131_072,
}


@dataclass(frozen=True)
class GemmaArmBinding:
    """One shared semantic arm bound to one exact resident v16 route."""

    semantic_arm: str
    v16_arm: str | None
    target_policy: str
    sender_checkpoint: str | None
    sender_revision: str | None
    mechanism_available: bool
    route_wired: bool
    blocker: str | None
    sender_tokenizer: str | None = None
    sender_tokenizer_revision: str | None = None
    sender_native_max_model_len: int | None = None
    sender_context_extension: str | None = None
    payload_layout: str | None = None


def arm_bindings() -> tuple[GemmaArmBinding, ...]:
    """Return the shared binding plus Gemma's full control arm."""
    bindings = [
        GemmaArmBinding(
            semantic_arm="issue_only",
            v16_arm="floor",
            target_policy="gemma4_12b_floor",
            sender_checkpoint=None,
            sender_revision=None,
            mechanism_available=True,
            route_wired=True,
            blocker=None,
        )
    ]
    bindings.extend(
        GemmaArmBinding(
            semantic_arm=arm,
            v16_arm=arm,
            target_policy={
                "text_primary": "gemma4_12b_text",
                "text_medium": "gemma4_12b_calls_gemma4_e4b_text",
                "text_small": "gemma4_12b_calls_gemma4_e2b_text",
            }[arm],
            sender_checkpoint=checkpoint,
            sender_revision=revision,
            mechanism_available=True,
            route_wired=True,
            blocker=None,
            sender_tokenizer=checkpoint,
            sender_tokenizer_revision=revision,
            sender_native_max_model_len=_GEMMA_TEXT_NATIVE_MAX_MODEL_LEN[arm],
            sender_context_extension="none",
        )
        for arm, (checkpoint, revision) in _GEMMA_TEXT_SENDERS.items()
    )
    bindings.extend(
        GemmaArmBinding(
            semantic_arm=f"latent_query_support_r{ratio}",
            v16_arm=f"r{ratio}",
            target_policy=f"gemma4_12b_r{ratio}_w16_support",
            sender_checkpoint=_GEMMA_TEXT_SENDERS["text_primary"][0],
            sender_revision=_GEMMA_TEXT_SENDERS["text_primary"][1],
            mechanism_available=True,
            route_wired=True,
            blocker=None,
            payload_layout=FLAT_INTERLEAVE_LAYOUT,
        )
        for ratio in FANOUTQA_NATURAL_DEV50.ratios
    )
    bindings.append(
        GemmaArmBinding(
            semantic_arm="full",
            v16_arm="full",
            target_policy="gemma4_12b_full",
            sender_checkpoint=_GEMMA_TEXT_SENDERS["text_primary"][0],
            sender_revision=_GEMMA_TEXT_SENDERS["text_primary"][1],
            mechanism_available=True,
            route_wired=True,
            blocker=None,
        )
    )
    result = tuple(bindings)
    expected = tuple(arm.arm_id for arm in arms_with_full(FANOUTQA_NATURAL_DEV50))
    if tuple(binding.semantic_arm for binding in result) != expected:
        raise RuntimeError("Gemma bindings drifted from the shared arm roster")
    return result


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class GemmaExecutionRoster:
    """The complete shared roster and its callable resident routes."""

    bindings: tuple[GemmaArmBinding, ...]
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50

    @property
    def wired(self) -> tuple[GemmaArmBinding, ...]:
        """Return arms whose registered physical route is callable."""
        return tuple(binding for binding in self.bindings if binding.route_wired)

    @property
    def execution_arms(self) -> tuple[str, ...]:
        """Return durable v16 cell names for all callable routes."""
        result = tuple(binding.v16_arm for binding in self.wired)
        if any(arm is None for arm in result):
            raise RuntimeError("wired Gemma binding has no physical v16 arm")
        return tuple(str(arm) for arm in result)

    @property
    def published_arms(self) -> tuple[str, ...]:
        """Return shared semantic row names paired with execution arms."""
        return tuple(binding.semantic_arm for binding in self.wired)

    @property
    def fingerprint(self) -> str:
        """Bind the full semantic-to-physical roster to one digest."""
        return _canonical_hash(
            {
                "schema": SHARED_EXECUTION_ROSTER_SCHEMA,
                "benchmark_profile": self.profile.profile_id,
                "scientific_fingerprint": self.profile.scientific_identity_hash,
                "source_fingerprint": self.profile.source_logical_fingerprint,
                "sampling_fingerprint": GEMMA.decode.identity_hash,
                "sample_tags": list(self.profile.sample_tags),
                "selection_artifact_profile": SHARED_SELECTION_ARTIFACT_PROFILE,
                "bindings": [
                    {
                        "semantic_arm": binding.semantic_arm,
                        "v16_arm": binding.v16_arm,
                        "target_policy": binding.target_policy,
                        "sender_checkpoint": binding.sender_checkpoint,
                        "sender_revision": binding.sender_revision,
                        "sender_tokenizer": binding.sender_tokenizer,
                        "sender_tokenizer_revision": binding.sender_tokenizer_revision,
                        "sender_native_max_model_len": binding.sender_native_max_model_len,
                        "sender_context_extension": binding.sender_context_extension,
                        "mechanism_available": binding.mechanism_available,
                        "route_wired": binding.route_wired,
                        "blocker": binding.blocker,
                        "payload_layout": binding.payload_layout,
                    }
                    for binding in self.bindings
                ],
            }
        )

    @property
    def identity_fields(self) -> dict[str, Any]:
        """Return fixed fields preventing cross-roster bank hydration."""
        return {
            "execution_roster_schema": SHARED_EXECUTION_ROSTER_SCHEMA,
            "execution_roster_fingerprint": self.fingerprint,
            "semantic_benchmark_profile": self.profile.profile_id,
            "semantic_scientific_fingerprint": self.profile.scientific_identity_hash,
            "semantic_sampling_profile": GEMMA.decode.profile_id,
            "semantic_sampling_fingerprint": GEMMA.decode.identity_hash,
            "semantic_answer_ceiling": self.profile.answer_ceiling,
            "semantic_report_ceiling": self.profile.report_ceiling,
            "execution_isolation_profile": SPLIT_FLEET_ISOLATION_PROFILE,
            "physical_latent_payload_layout": FLAT_INTERLEAVE_LAYOUT,
            "selection_artifact_profile": SHARED_SELECTION_ARTIFACT_PROFILE,
        }

    def require_complete(self) -> None:
        """Refuse all-arm execution while any binding is not callable."""
        missing = [binding.semantic_arm for binding in self.bindings if not binding.route_wired]
        if missing:
            raise RuntimeError("Gemma shared execution roster is incomplete: " + ", ".join(missing))


def shared_execution_roster(
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> GemmaExecutionRoster:
    """Return the complete, separately fingerprinted shared roster."""
    bindings = arm_bindings()
    if profile.profile_id == FANOUTQA_NATURAL_DEV50.profile_id:
        by_name = {binding.semantic_arm: binding for binding in bindings}
        bindings = tuple(by_name[name] for name in NATIVE_NATURAL_ARM_ORDER)
    return GemmaExecutionRoster(bindings, profile)


_SEMANTIC_BY_V16_ARM = {
    binding.v16_arm: binding.semantic_arm
    for binding in arm_bindings()
    if binding.v16_arm is not None
}
_BINDING_BY_V16_ARM = {
    binding.v16_arm: binding for binding in arm_bindings() if binding.v16_arm is not None
}
_V16_BY_SEMANTIC_ARM = {
    semantic_arm: str(v16_arm) for v16_arm, semantic_arm in _SEMANTIC_BY_V16_ARM.items()
}


def gemma_semantic_bindings() -> dict[str, str]:
    """Bind every executable Gemma v16 arm to its shared-registry semantic arm.

    The split fleet is placed by semantic arm, so this is where Gemma's physical
    roster meets :mod:`rcc.hardware.placements`.
    """
    bindings = {
        str(binding.v16_arm): binding.semantic_arm
        for binding in arm_bindings()
        if binding.route_wired and binding.v16_arm is not None
    }
    if len(bindings) != len(FANOUTQA_NATURAL_DEV50.arms) + 1:
        raise RuntimeError("Gemma placement binding does not cover the shared arm roster")
    return bindings


def gemma_v16_arm(semantic_arm: str) -> str:
    """Return the physical v16 route bound to one shared semantic arm.

    Placements, bank paths, and banked rows are keyed by semantic arm, while the
    capture, the selection, and the seed law are keyed by the physical route.
    """
    try:
        return _V16_BY_SEMANTIC_ARM[semantic_arm]
    except KeyError as exc:
        raise ValueError(f"Gemma semantic arm {semantic_arm!r} has no physical route") from exc


def flat_interleave_arm(arm: str) -> bool:
    """Return whether one physical arm ships the registered Qwen-flat payload."""
    binding = _BINDING_BY_V16_ARM.get(arm)
    return binding is not None and binding.payload_layout == FLAT_INTERLEAVE_LAYOUT


def shared_arm_seeds(
    qid: str,
    semantic_arm: str,
    *,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> tuple[int, ...]:
    """Return the registered logical draws for one semantic arm."""
    return shared_answer_seeds(profile, qid, semantic_arm)


def shared_seeds_for_v16_arm(
    qid: str,
    v16_arm: str,
    *,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> tuple[int, ...]:
    """Resolve shared draws for one registered resident arm."""
    try:
        semantic_arm = _SEMANTIC_BY_V16_ARM[v16_arm]
    except KeyError as exc:
        raise ValueError(f"Gemma v16 arm {v16_arm!r} has no shared semantic binding") from exc
    return shared_arm_seeds(qid, semantic_arm, profile=profile)


def project_shared_result_row(
    row: dict[str, Any],
    v16_arm: str,
    *,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> dict[str, Any]:
    """Project a resident v16 row onto its shared semantic identity."""
    try:
        binding = _BINDING_BY_V16_ARM[v16_arm]
    except KeyError as exc:
        raise ValueError(f"Gemma v16 arm {v16_arm!r} has no semantic row projection") from exc
    if not binding.route_wired:
        raise RuntimeError(f"Gemma arm {binding.semantic_arm!r} is not executable")
    if row.get("arm") != v16_arm:
        raise RuntimeError("Gemma result arm differs from the requested physical route")
    expected_policy = (
        binding.target_policy
        if v16_arm in _GEMMA_TEXT_SENDERS
        else REGISTERED_ARM_POLICIES[v16_arm]
    )
    if row.get("policy") != expected_policy or binding.target_policy != expected_policy:
        raise RuntimeError("Gemma physical policy differs from the shared target binding")
    if binding.payload_layout is not None and row.get("payload_layout") != binding.payload_layout:
        raise RuntimeError("Gemma latent result differs from the shared Qwen-flat payload layout")
    qid = str(row.get("qid") or "")
    if qid not in profile.question_ids:
        raise RuntimeError(f"Gemma result qid {qid!r} is outside the executing shared panel")
    if row.get("sample_seeds") != list(shared_seeds_for_v16_arm(qid, v16_arm, profile=profile)):
        raise RuntimeError(f"{qid}/{v16_arm}: result carries non-shared logical seeds")
    return {
        **row,
        "arm": binding.semantic_arm,
        "semantic_arm": binding.semantic_arm,
        "v16_arm": v16_arm,
        "v16_panel": row.get("panel"),
        "panel": SHARED_RESULT_PANEL,
    }


__all__ = (
    "SHARED_EXECUTION_ROSTER_SCHEMA",
    "SHARED_RESULT_PANEL",
    "SHARED_SELECTION_ARTIFACT_PROFILE",
    "SHARED_SOURCE_VERSION",
    "GemmaArmBinding",
    "GemmaExecutionRoster",
    "arm_bindings",
    "flat_interleave_arm",
    "gemma_semantic_bindings",
    "gemma_v16_arm",
    "project_shared_result_row",
    "shared_arm_seeds",
    "shared_execution_roster",
    "shared_seeds_for_v16_arm",
)
