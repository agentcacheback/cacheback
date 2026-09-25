"""Registered Ministral sender lanes and prepared prompt records.

One item's three worker prompts are re-encoded and bound to a prompt artifact
prepared elsewhere, and any difference is refused.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.ministral import MINISTRAL
from rcc.models.ministral.text_codec import (
    MINISTRAL_MAX_MODEL_LEN,
    MINISTRAL_TEXT_ARMS,
    MinistralTextCodec,
    prompt_token_sha256,
)
from rcc.topologies.fanout import FANOUT_M3

_BOS_TOKEN_ID = 1
_SYSTEM_OPEN_ID = 17
_SYSTEM_CLOSE_ID = 18
_PREPARED_PROMPT_SCHEMA = "ministral3-prepared-text-prompts-v1"


@dataclass(frozen=True)
class MinistralTextSender:
    """One 14B, 8B, or 3B sender targeting the registered 14B receiver."""

    semantic_arm: str
    policy: str
    checkpoint: str
    revision: str
    native_max_model_len: int
    resident: bool
    receiver_checkpoint: str
    receiver_revision: str

    def to_dict(self) -> dict[str, object]:
        """Return the complete sender-to-receiver lane identity."""
        return {
            "semantic_arm": self.semantic_arm,
            "policy": self.policy,
            "sender_checkpoint": self.checkpoint,
            "sender_revision": self.revision,
            "sender_tokenizer": self.checkpoint,
            "sender_tokenizer_revision": self.revision,
            "sender_native_max_model_len": self.native_max_model_len,
            "resident": self.resident,
            "receiver_checkpoint": self.receiver_checkpoint,
            "receiver_revision": self.receiver_revision,
        }


@dataclass(frozen=True)
class MinistralPromptRecord:
    """One worker prompt that exactly matched a separately prepared token artifact."""

    qid: str
    semantic_arm: str
    worker: int
    source_identity: str
    checkpoint: str
    revision: str
    token_ids: tuple[int, ...]
    token_sha256: str
    prepared_token_sha256: str
    prepared_fingerprint: str
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50

    def __post_init__(self) -> None:
        """Fail closed on source, sender, geometry, or prepared-artifact drift."""
        _require_prompt_record_identity(self)
        if not 0 < len(self.token_ids) <= self.profile.worker_prompt_tokens:
            raise ValueError("Ministral worker prompt does not have the registered geometry")
        if not self.token_ids or self.token_ids[0] != _BOS_TOKEN_ID:
            raise ValueError("Ministral worker prompt does not begin with Tekken BOS")
        if _SYSTEM_OPEN_ID not in self.token_ids or _SYSTEM_CLOSE_ID not in self.token_ids:
            raise ValueError("Ministral worker prompt omits the Reasoning system turn")
        measured = prompt_token_sha256(self.token_ids)
        if measured != self.token_sha256 or measured != self.prepared_token_sha256:
            raise ValueError("Ministral worker prompt differs from its prepared token artifact")
        if len(self.prepared_fingerprint) != 64:
            raise ValueError("Ministral prompt record lacks a prepared artifact fingerprint")


@dataclass(frozen=True)
class MinistralPreparedPromptArtifact:
    """One prepared three-worker prompt set for an exact item and sender."""

    qid: str
    semantic_arm: str
    source_identity: str
    checkpoint: str
    revision: str
    prompt_ids_by_worker: tuple[tuple[int, ...], ...]
    prompt_sha256: tuple[str, ...]
    fingerprint: str
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50

    def __post_init__(self) -> None:
        """Verify every prepared row and the artifact's canonical fingerprint."""
        sender = _sender(self.semantic_arm)
        if (
            self.qid not in self.profile.question_ids
            or self.source_identity != self.profile.source_logical_fingerprint
            or (self.checkpoint, self.revision) != (sender.checkpoint, sender.revision)
            or len(self.prompt_ids_by_worker) != FANOUT_M3.workers_per_item
            or not self.profile.admits_worker_prompt_tokens(
                [len(row) for row in self.prompt_ids_by_worker]
            )
        ):
            raise ValueError("Ministral prepared prompt artifact identity or geometry drifted")
        measured = tuple(prompt_token_sha256(row) for row in self.prompt_ids_by_worker)
        if measured != self.prompt_sha256:
            raise ValueError("Ministral prepared prompt row hashes drifted")
        if prepared_prompt_fingerprint(self) != self.fingerprint:
            raise ValueError("Ministral prepared prompt fingerprint drifted")


def registered_text_senders() -> tuple[MinistralTextSender, ...]:
    """Resolve the 14B, 8B, and 3B sender lanes into the 14B receiver."""
    by_arm = {arm.semantic_arm: arm for arm in MINISTRAL.physical_arms}
    senders: list[MinistralTextSender] = []
    for semantic_arm in MINISTRAL_TEXT_ARMS:
        arm = by_arm.get(semantic_arm)
        if (
            arm is None
            or arm.sender_checkpoint is None
            or arm.sender_revision is None
            or arm.sender_native_max_model_len is None
            or arm.selector is not None
        ):
            raise RuntimeError(f"Ministral text arm {semantic_arm!r} is incomplete")
        senders.append(
            MinistralTextSender(
                semantic_arm=semantic_arm,
                policy=arm.policy,
                checkpoint=arm.sender_checkpoint,
                revision=arm.sender_revision,
                native_max_model_len=arm.sender_native_max_model_len,
                resident=semantic_arm == "text_primary",
                receiver_checkpoint=MINISTRAL.checkpoint,
                receiver_revision=MINISTRAL.revision,
            )
        )
    return tuple(senders)


def bind_prompt_records(
    codec: MinistralTextCodec,
    *,
    qid: str,
    semantic_arm: str,
    question: str,
    evidence_ids_by_worker: Sequence[Sequence[int]],
    prepared: MinistralPreparedPromptArtifact,
    expected_prepared_fingerprint: str,
    source_identity: str,
) -> tuple[MinistralPromptRecord, ...]:
    """Re-encode three prompts and compare them to the prepared token artifact."""
    codec.require_verified()
    sender = _sender(semantic_arm)
    if (codec.checkpoint, codec.revision) != (sender.checkpoint, sender.revision):
        raise RuntimeError("Ministral prompt codec differs from its registered sender")
    if (
        prepared.qid,
        prepared.semantic_arm,
        prepared.source_identity,
        prepared.checkpoint,
        prepared.revision,
    ) != (
        qid,
        semantic_arm,
        source_identity,
        sender.checkpoint,
        sender.revision,
    ) or prepared.fingerprint != expected_prepared_fingerprint:
        raise RuntimeError("Ministral prepared prompt authority differs from resolved identity")
    if len(evidence_ids_by_worker) != FANOUT_M3.workers_per_item:
        raise ValueError("Ministral prompt binding requires exactly three evidence rows")
    records: list[MinistralPromptRecord] = []
    for worker, (evidence_ids, prepared_ids, prepared_sha256) in enumerate(
        zip(
            evidence_ids_by_worker,
            prepared.prompt_ids_by_worker,
            prepared.prompt_sha256,
            strict=True,
        )
    ):
        tokens = codec.encode_worker_prompt(question, evidence_ids)
        expected = tuple(int(token) for token in prepared_ids)
        if tokens != expected:
            raise RuntimeError(f"{qid}/{semantic_arm}/w{worker}: prepared prompt drifted")
        if len(tokens) + FANOUTQA_NATURAL_DEV50.report_ceiling > min(
            sender.native_max_model_len, MINISTRAL_MAX_MODEL_LEN
        ):
            raise RuntimeError("Ministral report exceeds the registered admission context")
        records.append(
            MinistralPromptRecord(
                qid=qid,
                semantic_arm=semantic_arm,
                worker=worker,
                source_identity=source_identity,
                checkpoint=sender.checkpoint,
                revision=sender.revision,
                token_ids=tokens,
                token_sha256=prompt_token_sha256(tokens),
                prepared_token_sha256=prepared_sha256,
                prepared_fingerprint=prepared.fingerprint,
                profile=prepared.profile,
            )
        )
    return tuple(records)


def prepared_prompt_fingerprint(artifact: MinistralPreparedPromptArtifact) -> str:
    """Fingerprint the independently prepared identity and all three row hashes."""
    payload = {
        "schema": _PREPARED_PROMPT_SCHEMA,
        "qid": artifact.qid,
        "semantic_arm": artifact.semantic_arm,
        "source_identity": artifact.source_identity,
        "checkpoint": artifact.checkpoint,
        "revision": artifact.revision,
        "prompt_sha256": list(artifact.prompt_sha256),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sender(semantic_arm: str) -> MinistralTextSender:
    sender = next(
        (
            candidate
            for candidate in registered_text_senders()
            if candidate.semantic_arm == semantic_arm
        ),
        None,
    )
    if sender is None:
        raise ValueError(f"unregistered Ministral text arm {semantic_arm!r}")
    return sender


def _require_prompt_record_identity(record: MinistralPromptRecord) -> None:
    if record.qid not in record.profile.question_ids:
        raise ValueError("Ministral prompt record has an unregistered question")
    if record.semantic_arm not in MINISTRAL_TEXT_ARMS:
        raise ValueError("Ministral prompt record has an unregistered text arm")
    if record.worker not in range(FANOUT_M3.workers_per_item):
        raise ValueError("Ministral prompt record lacks M=3 worker identity")
    if record.source_identity != record.profile.source_logical_fingerprint:
        raise ValueError("Ministral prompt record differs from the sealed source")
    sender = _sender(record.semantic_arm)
    if (record.checkpoint, record.revision) != (sender.checkpoint, sender.revision):
        raise ValueError("Ministral prompt record differs from its registered sender")


__all__ = (
    "MinistralPreparedPromptArtifact",
    "MinistralPromptRecord",
    "MinistralTextSender",
    "bind_prompt_records",
    "prepared_prompt_fingerprint",
    "registered_text_senders",
)
