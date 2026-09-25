"""Registered Ministral engine identities for the producer and receiver roles.

A split fleet opens two engines: the producer prefills and captures through the KV transfer
connector; the receiver opens the zero-copy HF view for its embeddings but never that connector.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rcc.models.ministral import MINISTRAL
from rcc.models.ministral.engine import (
    MinistralEngine,
    MinistralEngineConfig,
    build_engine,
    build_split_receiver_engine,
)
from rcc.models.ministral.runtime import (
    CAPTURE_CONNECTOR,
    CAPTURE_ENGINE_ROLE,
    REGISTERED_ENGINE_FLAGS,
    SPLIT_RECEIVER_ENGINE_ROLE,
    validate_runtime_signature,
)

#: The registered engine sequence limit, and therefore the receiver claim cap.
#: The AdmissionGate beside it is the memory authority. Read from the signed
#: flag roster rather than restated, so the cap and the engine cannot disagree.
MINISTRAL_ENGINE_MAX_NUM_SEQS = int(REGISTERED_ENGINE_FLAGS["max_num_seqs"])

PRODUCER_ROLE = "producer"
RECEIVER_ROLE = "receiver"

#: The producer role's vLLM budget. Its capture side pass runs outside the vLLM
#: allocator on the same device, so the pool and that pass share the card. See
#: docs/running.md, Ministral producer memory.
PRODUCER_GPU_MEMORY_UTILIZATION = 0.72
#: The receiver decodes injected embedding rows and runs no side pass, so
#: nothing competes with its pool.
RECEIVER_GPU_MEMORY_UTILIZATION = 0.85


@dataclass(frozen=True)
class MinistralEngineIdentity:
    """One registered engine identity for one physical split-fleet role."""

    role: str
    engine_role: str
    checkpoint: str
    revision: str
    capture_connector: Mapping[str, str] | None
    enable_prompt_embeds: bool
    #: Operational rather than identity: it is absent from `expectations()`, the
    #: signed runtime-signature fields, so moving it changes no banked row.
    #: `max_model_len` and `max_num_seqs` are signed and must not move.
    gpu_memory_utilization: float

    @property
    def kv_transfer_config(self) -> tuple[tuple[str, str], ...] | None:
        """Return the vLLM transfer stanza, or ``None`` for a plain engine."""
        if self.capture_connector is None:
            return None
        return tuple(sorted(self.capture_connector.items()))

    def expectations(self) -> dict[str, Any]:
        """Return the signed runtime-signature fields this role must carry."""
        return {
            "engine_role": self.engine_role,
            "active_checkpoint": self.checkpoint,
            "active_revision": self.revision,
            "enable_prompt_embeds": self.enable_prompt_embeds,
            "capture_connector": (
                None if self.capture_connector is None else dict(self.capture_connector)
            ),
        }


#: The resident capture engine. Every banked Ministral row signs this role and
#: connector, so neither string moves.
CAPTURE_ENGINE_IDENTITY = MinistralEngineIdentity(
    role=PRODUCER_ROLE,
    engine_role=CAPTURE_ENGINE_ROLE,
    checkpoint=MINISTRAL.checkpoint,
    revision=MINISTRAL.revision,
    capture_connector=dict(CAPTURE_CONNECTOR),
    enable_prompt_embeds=True,
    gpu_memory_utilization=PRODUCER_GPU_MEMORY_UTILIZATION,
)

#: The split receiver engine. It injects embedding rows and never captures, so
#: it opens with no transfer connector and refuses one if it finds it.
SPLIT_RECEIVER_ENGINE_IDENTITY = MinistralEngineIdentity(
    role=RECEIVER_ROLE,
    engine_role=SPLIT_RECEIVER_ENGINE_ROLE,
    checkpoint=MINISTRAL.checkpoint,
    revision=MINISTRAL.revision,
    capture_connector=None,
    enable_prompt_embeds=True,
    gpu_memory_utilization=RECEIVER_GPU_MEMORY_UTILIZATION,
)

REGISTERED_ENGINE_IDENTITIES = {
    PRODUCER_ROLE: CAPTURE_ENGINE_IDENTITY,
    RECEIVER_ROLE: SPLIT_RECEIVER_ENGINE_IDENTITY,
}


def registered_identity(role: str) -> MinistralEngineIdentity:
    """Return one registered engine identity, or refuse an unknown role."""
    try:
        return REGISTERED_ENGINE_IDENTITIES[role]
    except KeyError as exc:
        raise ValueError(f"unregistered Ministral engine role {role!r}") from exc


def role_engine_config(
    identity: MinistralEngineIdentity,
    config: MinistralEngineConfig | None = None,
) -> MinistralEngineConfig:
    """Return this role's registered memory budget unless a caller overrides it."""
    if config is not None:
        return config
    return MinistralEngineConfig(gpu_memory_utilization=identity.gpu_memory_utilization)


def open_role_engine(
    role: str,
    config: MinistralEngineConfig | None = None,
) -> MinistralEngine:
    """Open the one engine registered for this role in this process."""
    identity = registered_identity(role)
    resolved = role_engine_config(identity, config)
    if identity.role == PRODUCER_ROLE:
        return build_engine(resolved)
    return build_split_receiver_engine(resolved)


def verify_engine_identity(
    engine: Any,
    identity: MinistralEngineIdentity,
) -> dict[str, Any]:
    """Refuse to serve unless the live engine is the one this role registered."""
    signature = getattr(engine, "runtime_signature", None)
    if not isinstance(signature, dict):
        raise RuntimeError("Ministral engine exposes no observed runtime signature")
    resolved: dict[str, Any] = dict(signature)  # pyright: ignore[reportUnknownArgumentType]
    validate_runtime_signature(
        resolved,
        active_checkpoint=identity.checkpoint,
        active_revision=identity.revision,
        engine_role=identity.engine_role,
    )
    for name, wanted in identity.expectations().items():
        if resolved.get(name) != wanted:
            raise RuntimeError(
                f"Ministral engine resolved {name}={resolved.get(name)!r}, expected {wanted!r}"
            )
    return resolved


__all__ = (
    "CAPTURE_ENGINE_IDENTITY",
    "MINISTRAL_ENGINE_MAX_NUM_SEQS",
    "PRODUCER_GPU_MEMORY_UTILIZATION",
    "PRODUCER_ROLE",
    "RECEIVER_GPU_MEMORY_UTILIZATION",
    "RECEIVER_ROLE",
    "REGISTERED_ENGINE_IDENTITIES",
    "SPLIT_RECEIVER_ENGINE_IDENTITY",
    "MinistralEngineIdentity",
    "open_role_engine",
    "registered_identity",
    "role_engine_config",
    "verify_engine_identity",
)
