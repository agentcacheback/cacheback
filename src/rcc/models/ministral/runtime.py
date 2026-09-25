"""Observed runtime identity for every registered Ministral engine role."""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping
from importlib import metadata
from typing import Any, cast

import torch

from rcc.hardware.profile import P5_8XH100
from rcc.models.ministral import MINISTRAL, MINISTRAL_RUNTIME

CAPTURE_CONNECTOR = {
    "kv_connector": "MinistralCaptureConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "rcc.models.ministral.capture_connector",
}
#: The resident capture role. Every banked Ministral row signs this exact
#: string inside its runtime signature, so it does not move.
CAPTURE_ENGINE_ROLE = "receiver-capture-text-primary"
#: The split receiver: it injects prompt embeddings and owns no capture route,
#: so it is a second registered role beside the first.
SPLIT_RECEIVER_ENGINE_ROLE = "split-receiver-prompt-embeds"
REGISTERED_ENGINE_FLAGS = dict(MINISTRAL_RUNTIME.engine_flags)
NONRESIDENT_RUNTIME_RECORD_SCHEMA = "ministral-nonresident-text-runtime-receipt-v1"
NONRESIDENT_RUNTIME_AUTHORITY_SCHEMA = "ministral-nonresident-text-runtime-authority-v1"
_NONRESIDENT_ARMS = ("text_medium", "text_small")


def _installed_version(package: str) -> str:
    """Return one installed distribution version without importing it."""
    distribution = package.partition("[")[0]
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _engine_role_identity(
    engine_kwargs: Mapping[str, object],
    *,
    engine_role: str,
) -> dict[str, object]:
    """Derive the signed role identity from the kwargs passed to vLLM."""
    checkpoint = engine_kwargs.get("model")
    revision = engine_kwargs.get("revision")
    prompt_embeds = engine_kwargs.get("enable_prompt_embeds")
    raw_connector = engine_kwargs.get("kv_transfer_config")
    if (
        not isinstance(checkpoint, str)
        or not checkpoint
        or not isinstance(revision, str)
        or not revision
        or type(prompt_embeds) is not bool
        or (raw_connector is not None and not isinstance(raw_connector, Mapping))
    ):
        raise RuntimeError("Ministral effective engine kwargs are malformed")
    connector = dict(cast(Mapping[str, object], raw_connector)) if raw_connector else None
    return {
        "active_checkpoint": checkpoint,
        "active_revision": revision,
        "engine_role": engine_role,
        "enable_prompt_embeds": prompt_embeds,
        "capture_connector": connector,
    }


def observed_runtime_signature(
    engine_kwargs: Mapping[str, object],
    *,
    engine_role: str,
) -> dict[str, object]:
    """Read the live stack and sign the effective engine role configuration."""
    gpu: dict[str, object] | None = None
    if torch.cuda.is_available():
        cuda: Any = torch.cuda
        properties = cuda.get_device_properties(0)
        gpu = {
            "name": str(properties.name),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_vram_bytes": int(properties.total_memory),
        }
    return {
        "profile_id": MINISTRAL_RUNTIME.profile_id,
        "gpu": gpu,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "transformers": _installed_version("transformers"),
        "vllm": _installed_version("vllm"),
        "auxiliary_packages": {
            package: _installed_version(package)
            for package, _expected in MINISTRAL_RUNTIME.auxiliary_packages
        },
        "model_revisions": dict(MINISTRAL_RUNTIME.model_revisions),
        "tokenizer_revisions": dict(MINISTRAL_RUNTIME.tokenizer_revisions),
        "engine_flags": dict(REGISTERED_ENGINE_FLAGS),
        **_engine_role_identity(engine_kwargs, engine_role=engine_role),
        "decode_profile": MINISTRAL.decode.profile_id,
        "decode_fingerprint": MINISTRAL.decode.identity_hash,
        "decode": MINISTRAL.decode.to_dict(),
    }


def _canonical_fingerprint(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    if len(digest) != 64:
        raise RuntimeError("Ministral identity fingerprint is not full SHA-256")
    return digest


def runtime_fingerprint(signature: Mapping[str, object]) -> str:
    """Return a full canonical SHA-256 over one observed runtime signature."""
    return _canonical_fingerprint(dict(signature))


def gpu_matches_registration(raw_gpu: object) -> bool:
    """Return whether one measured GPU is the registered p5 H100 class."""
    gpu = cast(Mapping[str, object], raw_gpu) if isinstance(raw_gpu, Mapping) else None
    if gpu is None:
        return False
    memory = gpu.get("total_vram_bytes")
    return (
        str(gpu.get("name", "")).startswith(P5_8XH100.accelerator_name)
        and gpu.get("compute_capability") == [9, 0]
        and isinstance(memory, int)
        and not isinstance(memory, bool)
        and memory >= (P5_8XH100.accelerator_memory_gib - 1) * (1 << 30)
    )


def validate_runtime_signature(
    signature: Mapping[str, object],
    *,
    active_checkpoint: str,
    active_revision: str,
    engine_role: str,
) -> None:
    """Reapply the complete registered stack, hardware, role, and decode contract."""
    if engine_role == CAPTURE_ENGINE_ROLE:
        enable_prompt_embeds = True
        capture_connector: dict[str, str] | None = dict(CAPTURE_CONNECTOR)
    elif engine_role == SPLIT_RECEIVER_ENGINE_ROLE:
        # The split receiver decodes prompt embeddings on its own GPU and has
        # no capture route at all, so the transfer stanza is absent rather than
        # emptied, and reads back as an absent field.
        enable_prompt_embeds = True
        capture_connector = None
    elif engine_role in {"text_primary", *_NONRESIDENT_ARMS}:
        enable_prompt_embeds = False
        capture_connector = None
    else:
        raise RuntimeError(f"unregistered Ministral engine role {engine_role!r}")
    expected_identity = {
        "profile_id": MINISTRAL_RUNTIME.profile_id,
        "model_revisions": dict(MINISTRAL_RUNTIME.model_revisions),
        "tokenizer_revisions": dict(MINISTRAL_RUNTIME.tokenizer_revisions),
        "engine_flags": dict(REGISTERED_ENGINE_FLAGS),
        "active_checkpoint": active_checkpoint,
        "active_revision": active_revision,
        "engine_role": engine_role,
        "enable_prompt_embeds": enable_prompt_embeds,
        "capture_connector": capture_connector,
        "decode_profile": MINISTRAL.decode.profile_id,
        "decode_fingerprint": MINISTRAL.decode.identity_hash,
        "decode": MINISTRAL.decode.to_dict(),
    }
    expected_stack = {
        "torch": MINISTRAL_RUNTIME.torch,
        "transformers": MINISTRAL_RUNTIME.transformers,
        "vllm": MINISTRAL_RUNTIME.vllm,
    }
    observed_stack = {
        name: str(signature.get(name, "")).split("+", 1)[0] for name in expected_stack
    }
    expected_fields = {
        *expected_identity,
        "gpu",
        "python",
        "torch",
        "torch_cuda",
        "transformers",
        "vllm",
        "auxiliary_packages",
    }
    python = ".".join(str(signature.get("python", "")).split(".")[:2])
    if (
        set(signature) != expected_fields
        or any(signature.get(name) != value for name, value in expected_identity.items())
        or observed_stack != expected_stack
        or python != MINISTRAL_RUNTIME.python
        or signature.get("torch_cuda") != MINISTRAL_RUNTIME.cuda
        or signature.get("auxiliary_packages") != dict(MINISTRAL_RUNTIME.auxiliary_packages)
        or not gpu_matches_registration(signature.get("gpu"))
    ):
        raise RuntimeError("Ministral observed runtime differs from the registered contract")


def _nonresident_registration(semantic_arm: str) -> tuple[str, str]:
    if semantic_arm not in _NONRESIDENT_ARMS:
        raise RuntimeError(f"unregistered nonresident Ministral sender {semantic_arm!r}")
    arm = next(
        (
            candidate
            for candidate in MINISTRAL.physical_arms
            if candidate.semantic_arm == semantic_arm
        ),
        None,
    )
    if arm is None or arm.sender_checkpoint is None or arm.sender_revision is None:
        raise RuntimeError(f"Ministral sender registration is incomplete for {semantic_arm}")
    return arm.sender_checkpoint, arm.sender_revision


def build_nonresident_runtime_record(
    *,
    semantic_arm: str,
    runtime_signature: Mapping[str, object],
    observed_runtime_fingerprint: str,
) -> dict[str, object]:
    """Validate and sign one exact 8B or 3B observed-runtime record."""
    checkpoint, revision = _nonresident_registration(semantic_arm)
    signature = dict(runtime_signature)
    validate_runtime_signature(
        signature,
        active_checkpoint=checkpoint,
        active_revision=revision,
        engine_role=semantic_arm,
    )
    measured = runtime_fingerprint(signature)
    if observed_runtime_fingerprint != measured:
        raise RuntimeError(f"{semantic_arm}: nonresident runtime fingerprint drifted")
    body: dict[str, object] = {
        "schema": NONRESIDENT_RUNTIME_RECORD_SCHEMA,
        "semantic_arm": semantic_arm,
        "checkpoint": checkpoint,
        "revision": revision,
        "runtime_profile_fingerprint": MINISTRAL_RUNTIME.runtime_identity_hash,
        "runtime_signature": signature,
        "runtime_fingerprint": measured,
    }
    return {**body, "record_fingerprint": _canonical_fingerprint(body)}


def validate_nonresident_runtime_record(
    record: Mapping[str, object],
    *,
    semantic_arm: str,
) -> dict[str, object]:
    """Revalidate one persisted record against its complete registered role."""
    expected_fields = {
        "schema",
        "semantic_arm",
        "checkpoint",
        "revision",
        "runtime_profile_fingerprint",
        "runtime_signature",
        "runtime_fingerprint",
        "record_fingerprint",
    }
    raw_signature = record.get("runtime_signature")
    if set(record) != expected_fields or not isinstance(raw_signature, Mapping):
        raise RuntimeError(f"{semantic_arm}: malformed nonresident runtime record")
    fingerprint = record.get("runtime_fingerprint")
    if not isinstance(fingerprint, str):
        raise RuntimeError(f"{semantic_arm}: malformed nonresident runtime fingerprint")
    canonical = build_nonresident_runtime_record(
        semantic_arm=semantic_arm,
        runtime_signature=cast(Mapping[str, object], raw_signature),
        observed_runtime_fingerprint=fingerprint,
    )
    if dict(record) != canonical:
        raise RuntimeError(f"{semantic_arm}: nonresident runtime record drifted")
    return canonical


def build_nonresident_runtime_authority(
    records: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Bind the complete ordered 8B and 3B record roster for every result row."""
    if set(records) != set(_NONRESIDENT_ARMS):
        raise RuntimeError("Ministral nonresident runtime record roster is incomplete")
    validated = [
        validate_nonresident_runtime_record(records[arm], semantic_arm=arm)
        for arm in _NONRESIDENT_ARMS
    ]
    body: dict[str, object] = {
        "schema": NONRESIDENT_RUNTIME_AUTHORITY_SCHEMA,
        "records": validated,
    }
    return {**body, "roster_fingerprint": _canonical_fingerprint(body)}


def validate_nonresident_runtime_authority(
    authority: Mapping[str, object],
) -> dict[str, object]:
    """Revalidate a row-level two-sender authority without external state."""
    raw_records = authority.get("records")
    if set(authority) != {"schema", "records", "roster_fingerprint"} or not isinstance(
        raw_records, list
    ):
        raise RuntimeError("Ministral nonresident runtime authority is malformed")
    records: dict[str, Mapping[str, object]] = {}
    for raw_record in cast(list[object], raw_records):
        if not isinstance(raw_record, Mapping):
            raise RuntimeError("Ministral nonresident runtime authority is malformed")
        record = cast(Mapping[str, object], raw_record)
        semantic_arm = record.get("semantic_arm")
        if not isinstance(semantic_arm, str) or semantic_arm in records:
            raise RuntimeError("Ministral nonresident runtime authority has duplicate senders")
        records[semantic_arm] = record
    canonical = build_nonresident_runtime_authority(records)
    if dict(authority) != canonical:
        raise RuntimeError("Ministral nonresident runtime authority drifted")
    return canonical


__all__ = (
    "CAPTURE_CONNECTOR",
    "CAPTURE_ENGINE_ROLE",
    "NONRESIDENT_RUNTIME_AUTHORITY_SCHEMA",
    "NONRESIDENT_RUNTIME_RECORD_SCHEMA",
    "REGISTERED_ENGINE_FLAGS",
    "SPLIT_RECEIVER_ENGINE_ROLE",
    "build_nonresident_runtime_authority",
    "build_nonresident_runtime_record",
    "gpu_matches_registration",
    "observed_runtime_signature",
    "runtime_fingerprint",
    "validate_nonresident_runtime_authority",
    "validate_nonresident_runtime_record",
    "validate_runtime_signature",
)
