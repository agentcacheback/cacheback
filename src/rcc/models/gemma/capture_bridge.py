"""Live-engine bridge for Gemma 4 capture on vLLM 0.26.

The resident vLLM model owns the weights: a meta-device HF Gemma skeleton takes
detached views, and the prompt pages are rebuilt as a rolling hybrid cache.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch

from rcc.models.capture_view import (
    EngineCaptureBridge,
    RequestFactory,
    alias_hf_model,
    check_qkv_splits,
    map_passthrough,
    required_tensor,
    resident_language_model,
    resident_named_tensors,
    shaped,
    vllm_request_factory,
)
from rcc.models.gemma.capture_connector import (
    GemmaHybridKV,
    discard_gemma_extraction,
    request_gemma_extraction,
    take_gemma_extraction,
)


def _required(named: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    return required_tensor(named, name, "Gemma")


def _layer_geometry(config: Any, layer: int) -> tuple[int, int, int]:
    layer_type = str(config.layer_types[layer])
    full = layer_type == "full_attention"
    if layer_type not in ("full_attention", "sliding_attention"):
        raise ValueError(f"Gemma layer {layer} has unsupported type {layer_type!r}")
    head_dim = int(config.global_head_dim if full else config.head_dim)
    kv_heads = int(
        config.num_global_key_value_heads
        if full and bool(getattr(config, "attention_k_eq_v", False))
        else config.num_key_value_heads
    )
    return int(config.num_attention_heads) * head_dim, kv_heads * head_dim, head_dim


def gemma4_view_state(named: Mapping[str, torch.Tensor], config: Any) -> dict[str, torch.Tensor]:
    """Map fused vLLM Gemma tensors to zero-copy HF state-dict views."""
    hidden = int(config.hidden_size)
    intermediate = int(config.intermediate_size)
    views: dict[str, torch.Tensor] = {}
    for layer in range(int(config.num_hidden_layers)):
        base = f"model.layers.{layer}"
        q_rows, kv_rows, head_dim = _layer_geometry(config, layer)
        qkv_name = f"{base}.self_attn.qkv_proj.weight"
        qkv = shaped(qkv_name, _required(named, qkv_name), (q_rows + 2 * kv_rows, hidden))
        views[f"{base}.self_attn.q_proj.weight"] = qkv[:q_rows]
        views[f"{base}.self_attn.k_proj.weight"] = qkv[q_rows : q_rows + kv_rows]
        full_k_eq_v = str(config.layer_types[layer]) == "full_attention" and bool(
            getattr(config, "attention_k_eq_v", False)
        )
        v_view = qkv[q_rows + kv_rows :]
        if full_k_eq_v:
            if not torch.equal(views[f"{base}.self_attn.k_proj.weight"], v_view):
                raise ValueError(f"{qkv_name} K/V slices differ on an attention_k_eq_v layer")
        else:
            views[f"{base}.self_attn.v_proj.weight"] = v_view

        gate_name = f"{base}.mlp.gate_up_proj.weight"
        gate_up = shaped(gate_name, _required(named, gate_name), (2 * intermediate, hidden))
        views[f"{base}.mlp.gate_proj.weight"] = gate_up[:intermediate]
        views[f"{base}.mlp.up_proj.weight"] = gate_up[intermediate:]
        map_passthrough(
            views,
            named,
            base,
            {
                "self_attn.o_proj.weight": (hidden, q_rows),
                "self_attn.q_norm.weight": (head_dim,),
                "self_attn.k_norm.weight": (head_dim,),
                "mlp.down_proj.weight": (hidden, intermediate),
                "input_layernorm.weight": (hidden,),
                "post_attention_layernorm.weight": (hidden,),
                "pre_feedforward_layernorm.weight": (hidden,),
                "post_feedforward_layernorm.weight": (hidden,),
                "layer_scalar": (1,),
            },
            "Gemma",
        )

    embed = _required(named, "model.embed_tokens.weight")
    expected_embed = (int(config.vocab_size), hidden)
    if tuple(embed.shape) != expected_embed:
        raise ValueError(
            f"model.embed_tokens.weight is {tuple(embed.shape)}, expected {expected_embed}"
        )
    views["model.embed_tokens.weight"] = embed
    views["model.norm.weight"] = shaped(
        "model.norm.weight", _required(named, "model.norm.weight"), (hidden,)
    )
    if not bool(getattr(config, "tie_word_embeddings", False)):
        raise ValueError("the Gemma capture bridge requires the registered tied embeddings")
    views["lm_head.weight"] = embed
    return views


def _materialize_nonpersistent_buffers(model: Any, device: torch.device) -> None:
    embed = model.model.embed_tokens
    embed.embed_scale = torch.tensor(embed.scalar_embed_scale, device=device)
    rotary = model.model.rotary_emb
    with torch.device("cpu"):
        rebuilt = type(rotary)(config=rotary.config)
    model.model.rotary_emb = rebuilt.to(device)


def gemma4_hf_view(
    named: Mapping[str, torch.Tensor], config: Any, *, generation_config: Any = None
) -> Any:
    """Build an eval-mode HF Gemma whose every parameter aliases vLLM storage."""
    from transformers import Gemma4UnifiedForCausalLM

    return alias_hf_model(
        named,
        gemma4_view_state(named, config),
        family="Gemma",
        skeleton=lambda: Gemma4UnifiedForCausalLM(config),
        materialize=_materialize_nonpersistent_buffers,
        generation_config=generation_config,
    )


def _resident_tensors(model: Any, config: Any) -> dict[str, torch.Tensor]:
    named = resident_named_tensors(model, "Gemma")
    check_qkv_splits(
        model,
        int(config.num_hidden_layers),
        lambda layer: _layer_geometry(config, layer)[:2],
    )
    return named


def rolling_cache_from_artifact(
    artifact: GemmaHybridKV,
    config: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    """Reconstruct the HF rolling cache from heterogeneous device layers.

    On the live route the artifact already sits on this device and dtype, so
    the layers adopt its own tensors; nothing writes into them in place.
    """
    from rcc.models.gemma.cache import RollingHybridCache

    cache = RollingHybridCache(config)
    schedule = list(config.layer_types)
    if len(artifact.layers) != len(cache.layers) or len(schedule) != len(cache.layers):
        raise RuntimeError(
            f"artifact/cache/config layer counts are {len(artifact.layers)}/"
            f"{len(cache.layers)}/{len(schedule)}"
        )
    for index, (record, target, layer_type) in enumerate(
        zip(artifact.layers, cache.layers, schedule, strict=True)
    ):
        target_layer = cast(Any, target)
        if record.layer_index != index or record.layer_type != str(layer_type):
            raise RuntimeError(
                f"artifact layer {record.layer_index}/{record.layer_type} does not match "
                f"config layer {index}/{layer_type}"
            )
        expected_start = (
            max(0, artifact.logical_length - (int(config.sliding_window) - 1))
            if layer_type == "sliding_attention"
            else 0
        )
        expected_rows = artifact.logical_length - expected_start
        if record.start_position != expected_start:
            raise RuntimeError(
                f"layer {index} starts at {record.start_position}, expected {expected_start}"
            )
        if record.keys.shape != record.values.shape or int(record.keys.shape[-2]) != expected_rows:
            raise RuntimeError(f"layer {index} extracted cache shape is inconsistent")
        if not bool(torch.isfinite(record.keys).all()) or not bool(
            torch.isfinite(record.values).all()
        ):
            raise RuntimeError(f"layer {index} extracted cache contains a non-finite value")
        keys = record.keys.to(device=device, dtype=dtype)
        values = record.values.to(device=device, dtype=dtype)
        target_layer.lazy_initialization(keys, values)
        target_layer.keys, target_layer.values = keys, values
        if layer_type == "sliding_attention":
            target_layer.cumulative_length = artifact.logical_length
    if cache.get_seq_length() != artifact.logical_length:
        raise RuntimeError(
            f"rebuilt cache length {cache.get_seq_length()} != {artifact.logical_length}"
        )
    return cache


class GemmaEngineCaptureBridge(EngineCaptureBridge):
    """Borrow one live vLLM engine for extraction and the HF side pass."""

    family = "Gemma"
    request_prefix = "gemma-capture"

    @classmethod
    def from_llm(
        cls,
        llm: Any,
        *,
        text_config: Any,
        generation_config: Any = None,
        request_factory: RequestFactory | None = None,
    ) -> GemmaEngineCaptureBridge:
        """Adopt a raw, in-process TP=1 vLLM 0.26 LLM without owning it."""
        resident = resident_language_model(llm, "Gemma")
        tensors = _resident_tensors(resident, text_config)
        model = gemma4_hf_view(tensors, text_config, generation_config=generation_config)
        tokenizer = llm.get_tokenizer()
        return cls(
            llm,
            model,
            tokenizer,
            request_factory=request_factory or vllm_request_factory,
        )

    def _mark(self, internal_id: str) -> None:
        request_gemma_extraction(internal_id)

    def _take(self, internal_id: str) -> Any:
        return take_gemma_extraction(internal_id)

    def _discard(self, internal_id: str) -> None:
        discard_gemma_extraction(internal_id)

    def _rebuild(
        self, artifact: Any, config: Any, *, device: torch.device, dtype: torch.dtype
    ) -> Any:
        return rolling_cache_from_artifact(artifact, config, device=device, dtype=dtype)


__all__ = (
    "GemmaEngineCaptureBridge",
    "gemma4_hf_view",
    "gemma4_view_state",
    "rolling_cache_from_artifact",
    "vllm_request_factory",
)
