"""Registered Gemma engine identities for the producer and receiver roles.

A split fleet opens two engines: the producer prefills and captures through the
KV transfer connector, the receiver decodes prompt embeddings and cannot.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, cast

from rcc.models.gemma.contract import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    EMBEDDING_SHAPE,
    ENFORCE_EAGER,
    MAX_MODEL_LEN,
    MAX_NUM_SEQS,
)
from rcc.models.gemma.engine import EngineConfig, build_engine
from rcc.models.gemma.engine_contract import (
    FULL_VOCAB_LOGPROBS,
    REGISTERED_SPEC,
    VLLM_LOGPROBS_MODE,
    attention_layer_record,
    registered_attention_layers,
)

PRODUCER_ROLE = "producer"
RECEIVER_ROLE = "receiver"


@dataclass(frozen=True)
class GemmaEngineIdentity:
    """One registered KV-transfer identity for one engine role."""

    role: str
    kv_connector: str | None
    kv_role: str | None
    kv_connector_module_path: str | None

    @property
    def kv_transfer_config(self) -> tuple[tuple[str, Any], ...] | None:
        """Return the vLLM transfer stanza, or ``None`` for a plain engine."""
        if self.kv_connector is None:
            return None
        return (
            ("kv_connector", self.kv_connector),
            ("kv_role", self.kv_role),
            ("kv_connector_module_path", self.kv_connector_module_path),
        )

    def expectations(self) -> dict[str, Any]:
        """Return the resolved engine fields a live engine must match."""
        return {
            "kv_connector": self.kv_connector,
            "kv_role": self.kv_role,
            "kv_connector_module_path": self.kv_connector_module_path,
        }


#: The resident capture engine. Every banked Gemma row carries this string
#: set, so these three values do not move.
CAPTURE_ENGINE_IDENTITY = GemmaEngineIdentity(
    role=PRODUCER_ROLE,
    kv_connector="GemmaCaptureConnector",
    kv_role="kv_both",
    kv_connector_module_path="rcc.models.gemma.capture_connector",
)

#: The split receiver engine. It injects prompt embeddings and never captures,
#: so it opens with no transfer connector and refuses one if it finds it.
SPLIT_RECEIVER_ENGINE_IDENTITY = GemmaEngineIdentity(
    role=RECEIVER_ROLE,
    kv_connector=None,
    kv_role=None,
    kv_connector_module_path=None,
)

REGISTERED_ENGINE_IDENTITIES = {
    PRODUCER_ROLE: CAPTURE_ENGINE_IDENTITY,
    RECEIVER_ROLE: SPLIT_RECEIVER_ENGINE_IDENTITY,
}


def engine_config(identity: GemmaEngineIdentity) -> EngineConfig:
    """Return the exact bring-up settings for one registered engine role."""
    return EngineConfig(
        model=CHECKPOINT_ID,
        revision=CHECKPOINT_REVISION,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        enforce_eager=ENFORCE_EAGER,
        enable_prompt_embeds=True,
        enable_prefix_caching=False,
        disable_log_stats=False,
        attention_config=(("backend", "FLASH_ATTN"), ("flash_attn_version", 4)),
        kv_transfer_config=identity.kv_transfer_config,
        language_model_only=True,
        generation_config="auto",
        logprobs_mode=VLLM_LOGPROBS_MODE,
        max_logprobs=FULL_VOCAB_LOGPROBS,
    )


def registered_identity(role: str) -> GemmaEngineIdentity:
    """Return one registered engine identity, or refuse an unknown role."""
    try:
        return REGISTERED_ENGINE_IDENTITIES[role]
    except KeyError as exc:
        raise ValueError(f"unregistered Gemma engine role {role!r}") from exc


def open_role_engine(role: str) -> Any:
    """Open the one engine registered for this role in this process."""
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    return build_engine(engine_config(registered_identity(role)))


def _resolved_engine_fields(llm: Any) -> dict[str, Any]:
    model = llm.model_config
    attention = llm.llm_engine.vllm_config.attention_config
    transfer = llm.llm_engine.vllm_config.kv_transfer_config
    raw_layers = cast(object, llm.apply_model(attention_layer_record))
    if not isinstance(raw_layers, list):
        raise RuntimeError("Gemma attention record returned no worker list")
    layers = cast(list[object], raw_layers)
    if len(layers) != 1:
        raise RuntimeError(f"Gemma attention record returned {len(layers)} workers")
    return {
        "model": str(model.model),
        "architecture": str(model.architecture),
        "revision": str(model.revision),
        "dtype": str(model.dtype).removeprefix("torch."),
        "max_model_len": int(model.max_model_len),
        "max_logprobs": int(model.max_logprobs),
        "logprobs_mode": str(model.logprobs_mode),
        "vocab_size": int(model.get_vocab_size()),
        "hidden_size": int(model.get_inputs_embeds_size()),
        "enable_prompt_embeds": bool(model.enable_prompt_embeds),
        "enforce_eager": bool(model.enforce_eager),
        "generation_config": str(model.generation_config),
        "language_model_only": bool(model.get_multimodal_config().language_model_only),
        "enable_prefix_caching": bool(
            llm.llm_engine.vllm_config.cache_config.enable_prefix_caching
        ),
        "attention_backend": str(getattr(attention.backend, "name", attention.backend)),
        "flash_attn_version": attention.flash_attn_version,
        # A receiver engine opens with no transfer connector at all, so the
        # absent stanza reads as three absent fields and is checked against
        # the registered receiver identity like any other value.
        "kv_connector": None if transfer is None else str(transfer.kv_connector),
        "kv_role": None if transfer is None else str(transfer.kv_role),
        "kv_connector_module_path": (
            None if transfer is None else str(transfer.kv_connector_module_path)
        ),
        "tensor_parallel_size": int(
            llm.llm_engine.vllm_config.parallel_config.tensor_parallel_size
        ),
        "using_transformers_backend": bool(model.using_transformers_backend()),
        "attention_layers": layers[0],
    }


def verify_engine_identity(engine: Any, identity: GemmaEngineIdentity) -> dict[str, Any]:
    """Refuse to serve unless the live engine is the one this role registered."""
    if engine is None or engine._llm is None:
        raise RuntimeError("Gemma vLLM engine is not live")
    resolved = _resolved_engine_fields(engine._llm)
    expected: dict[str, Any] = {
        "model": CHECKPOINT_ID,
        "architecture": REGISTERED_SPEC.engine_architecture,
        "revision": CHECKPOINT_REVISION,
        "dtype": "bfloat16",
        "max_model_len": MAX_MODEL_LEN,
        "max_logprobs": FULL_VOCAB_LOGPROBS,
        "logprobs_mode": VLLM_LOGPROBS_MODE,
        "vocab_size": EMBEDDING_SHAPE[0],
        "hidden_size": EMBEDDING_SHAPE[1],
        "enable_prompt_embeds": True,
        "enforce_eager": ENFORCE_EAGER,
        "generation_config": "auto",
        "language_model_only": True,
        "enable_prefix_caching": False,
        "using_transformers_backend": False,
        "attention_backend": REGISTERED_SPEC.attention_backend,
        "flash_attn_version": REGISTERED_SPEC.flash_attn_version,
        **identity.expectations(),
        "tensor_parallel_size": 1,
        "attention_layers": registered_attention_layers(),
    }
    for name, wanted in expected.items():
        if resolved[name] != wanted:
            raise RuntimeError(
                f"Gemma vLLM resolved {name}={resolved[name]!r}, expected {wanted!r}"
            )
    return resolved


def build_capture_bridge(engine: Any) -> Any:
    """Borrow the live producer DecodeEngine through the zero-copy bridge."""
    llm = getattr(engine, "_llm", None)
    if llm is None:
        raise RuntimeError("Gemma capture cannot borrow a closed DecodeEngine")
    outer = llm.model_config.hf_config
    resolve_text = getattr(outer, "get_text_config", None)
    if not callable(resolve_text):
        raise RuntimeError("Gemma vLLM config cannot resolve its decoder text config")
    from rcc.models.gemma.capture_bridge import GemmaEngineCaptureBridge

    return GemmaEngineCaptureBridge.from_llm(llm, text_config=resolve_text(decoder=True))


__all__ = (
    "CAPTURE_ENGINE_IDENTITY",
    "PRODUCER_ROLE",
    "RECEIVER_ROLE",
    "REGISTERED_ENGINE_IDENTITIES",
    "SPLIT_RECEIVER_ENGINE_IDENTITY",
    "GemmaEngineIdentity",
    "build_capture_bridge",
    "engine_config",
    "open_role_engine",
    "registered_identity",
    "verify_engine_identity",
)
