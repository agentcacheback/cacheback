"""Dependency-clean model identities consumed by the run resolver."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from rcc.models.decode import DecodeProtocol


@dataclass(frozen=True)
class RuntimeProfile:
    """One immutable, registered software and engine environment."""

    profile_id: str
    python: str
    torch: str
    cuda: str
    transformers: str
    vllm: str
    model_revisions: tuple[tuple[str, str], ...]
    tokenizer_revisions: tuple[tuple[str, str], ...]
    engine_flags: tuple[tuple[str, str], ...]
    auxiliary_packages: tuple[tuple[str, str], ...] = ()

    @property
    def runtime_identity_hash(self) -> str:
        """Return the complete runtime-profile identity."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible runtime echo."""
        identity: dict[str, object] = {
            "profile_id": self.profile_id,
            "python": self.python,
            "torch": self.torch,
            "cuda": self.cuda,
            "transformers": self.transformers,
            "vllm": self.vllm,
            "model_revisions": {name: revision for name, revision in self.model_revisions},
            "tokenizer_revisions": {name: revision for name, revision in self.tokenizer_revisions},
            "engine_flags": {name: value for name, value in self.engine_flags},
        }
        if self.auxiliary_packages:
            identity["auxiliary_packages"] = {
                name: version for name, version in self.auxiliary_packages
            }
        return identity


@dataclass(frozen=True)
class PhysicalArm:
    """One model-specific implementation of a semantic scientific arm."""

    semantic_arm: str
    policy: str
    sender_checkpoint: str | None
    sender_revision: str | None
    sender_tokenizer: str | None
    sender_tokenizer_revision: str | None
    selector: str | None
    sender_native_max_model_len: int | None = None
    sender_context_extension: str | None = None
    payload_layout: str | None = None
    selector_recipe: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a complete physical arm identity."""
        identity: dict[str, object] = {
            "semantic_arm": self.semantic_arm,
            "policy": self.policy,
            "sender_checkpoint": self.sender_checkpoint,
            "sender_revision": self.sender_revision,
            "sender_tokenizer": self.sender_tokenizer,
            "sender_tokenizer_revision": self.sender_tokenizer_revision,
            "selector": self.selector,
        }
        if self.sender_native_max_model_len is not None:
            identity["sender_native_max_model_len"] = self.sender_native_max_model_len
        if self.sender_context_extension is not None:
            identity["sender_context_extension"] = self.sender_context_extension
        if self.payload_layout is not None:
            identity["payload_layout"] = self.payload_layout
        if self.selector_recipe is not None:
            identity["selector_recipe"] = self.selector_recipe
        return identity


@dataclass(frozen=True)
class ModelProfile:
    """Pinned model, tokenizer, runtime, lifecycle, and adapter registration."""

    model_id: str
    checkpoint: str
    revision: str
    tokenizer: str
    tokenizer_revision: str
    decode: DecodeProtocol
    runtime: RuntimeProfile
    lifecycle: str
    physical_arms: tuple[PhysicalArm, ...]
    execution_order: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible model identity."""
        identity: dict[str, object] = {
            "model_id": self.model_id,
            "checkpoint": self.checkpoint,
            "revision": self.revision,
            "tokenizer": self.tokenizer,
            "tokenizer_revision": self.tokenizer_revision,
            "decode": self.decode.to_dict(),
            "decode_profile": self.decode.profile_id,
            "decode_fingerprint": self.decode.identity_hash,
            "runtime": self.runtime.to_dict(),
            "lifecycle": self.lifecycle,
            "physical_arms": [arm.to_dict() for arm in self.physical_arms],
        }
        if self.execution_order:
            identity["execution_order"] = list(self.execution_order)
        return identity
