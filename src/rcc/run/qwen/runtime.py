"""The engine identity a route lane's banked rows carry."""

from __future__ import annotations

import os

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.qwen.backend import QwenBackendSettings, effective_engine_kwargs
from rcc.models.qwen.engine import QWEN_CHAIN_ENGINE_ROUTE, QWEN_ENGINE_ROUTE
from rcc.models.route import RouteFamily
from rcc.run.io import is_sha256_hex
from rcc.run.plan import route_family_for_lane
from rcc.topologies.chain import CHAIN_TOPOLOGY_KEY


def _engine_identity(
    checkpoint: str,
    revision: str,
    *,
    settings: QwenBackendSettings,
    family: RouteFamily,
    tokenizer: str,
    tokenizer_revision: str,
) -> dict[str, object]:
    """Normalize the engine constructor kwargs, filling in omitted defaults."""
    identity = effective_engine_kwargs(
        checkpoint,
        revision,
        settings=settings,
        family=family,
        tokenizer=tokenizer,
        tokenizer_revision=tokenizer_revision,
    )
    identity.setdefault("max_num_batched_tokens", None)
    identity.setdefault("enforce_eager", False)
    identity.setdefault("disable_hybrid_kv_cache_manager", False)
    identity["seed"] = None
    identity["v1_multiprocessing"] = False
    identity["kv_transfer_config"] = identity.get("kv_transfer_config")
    return identity


#: The capture engine each topology opens, and the route name a row banks for
#: it. A benchmark moves nothing else in the identity below.
_CAPTURE_ROLES: dict[str, tuple[str, QwenBackendSettings]] = {
    FANOUTQA_NATURAL_DEV50.topology_key: (QWEN_ENGINE_ROUTE, QwenBackendSettings(capture=True)),
    CHAIN_TOPOLOGY_KEY: (
        QWEN_CHAIN_ENGINE_ROUTE,
        QwenBackendSettings(capture=True, chain=True),
    ),
}


def engine_route(
    family: RouteFamily, profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50
) -> dict[str, object]:
    """Return one lane's engine identity for the benchmark this node runs.

    Only the producer's capture engine differs by benchmark; the receiver and
    the senders are the same whatever the producer read.
    """
    try:
        route, capture = _CAPTURE_ROLES[profile.topology_key]
    except KeyError:
        raise ValueError(
            f"the {family.lane} engine route serves the {', '.join(_CAPTURE_ROLES)} "
            f"topologies; {profile.benchmark_key} registers {profile.topology_key!r}"
        ) from None
    model = family.profile
    return {
        "backend": "engine",
        "route": dict(model.runtime.engine_flags).get("producer_backend")
        if family.lane == "nemotron"
        else route,
        "capture": _engine_identity(
            model.checkpoint,
            model.revision,
            settings=capture,
            family=family,
            tokenizer=model.tokenizer,
            tokenizer_revision=model.tokenizer_revision,
        ),
        "receiver": _engine_identity(
            model.checkpoint,
            model.revision,
            settings=QwenBackendSettings(receiver=True),
            family=family,
            tokenizer=model.tokenizer,
            tokenizer_revision=model.tokenizer_revision,
        ),
        "text": {
            arm.semantic_arm: _engine_identity(
                arm.sender_checkpoint,
                arm.sender_revision,
                settings=QwenBackendSettings(
                    receiver=family.native_sender_prompts and arm.semantic_arm == "text_primary"
                ),
                family=family,
                tokenizer=arm.sender_tokenizer,
                tokenizer_revision=arm.sender_tokenizer_revision,
            )
            for arm in model.physical_arms
            if arm.semantic_arm.startswith("text_")
            and arm.sender_checkpoint is not None
            and arm.sender_revision is not None
            and arm.sender_tokenizer is not None
            and arm.sender_tokenizer_revision is not None
        },
    }


def runtime_route_field(family: RouteFamily) -> str:
    """Return the runtime-signature field one lane records its route under."""
    return f"rcc_{family.lane}_producer"


ENGINE_ROUTE = engine_route(route_family_for_lane("qwen"), FANOUTQA_NATURAL_DEV50)


def runtime_identity_fields(family: RouteFamily) -> dict[str, str]:
    """Return the plan fingerprint and decode identity a row carries."""
    fingerprint = os.environ.get("RCC_FANOUT_PLAN_FINGERPRINT")
    if fingerprint is None or not is_sha256_hex(fingerprint):
        raise RuntimeError(f"{family.lane} fleet requires its plan fingerprint")
    return {
        "decode_profile": family.profile.decode.profile_id,
        "decode_fingerprint": family.profile.decode.identity_hash,
        "unified_plan_fingerprint": fingerprint,
    }


__all__ = (
    "ENGINE_ROUTE",
    "engine_route",
    "runtime_identity_fields",
    "runtime_route_field",
)
