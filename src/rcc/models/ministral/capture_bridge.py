"""Zero-copy resident-weight bridge for Ministral 3 on vLLM 0.26.

The live vLLM model owns the weights: a meta-device HF skeleton receives
detached views, and extracted KV pages are copied into a dense HF cache.
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
from rcc.models.ministral.capture_connector import (
    MinistralDenseKV,
    discard_ministral_extraction,
    request_ministral_extraction,
    take_ministral_extraction,
)


def _required(named: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    return required_tensor(named, name, "Ministral")


def _layer_geometry(config: Any) -> tuple[int, int]:
    head_dim = int(config.head_dim)
    q_rows = int(config.num_attention_heads) * head_dim
    kv_rows = int(config.num_key_value_heads) * head_dim
    if min(head_dim, q_rows, kv_rows) <= 0:
        raise ValueError("Ministral config has invalid attention geometry")
    return q_rows, kv_rows


def ministral3_view_state(
    named: Mapping[str, torch.Tensor], config: Any
) -> dict[str, torch.Tensor]:
    """Map fused resident tensors to zero-copy HF Ministral state views."""
    if getattr(config, "sliding_window", None) is not None:
        raise ValueError("Ministral capture requires the registered dense checkpoint")
    hidden = int(config.hidden_size)
    intermediate = int(config.intermediate_size)
    q_rows, kv_rows = _layer_geometry(config)
    views: dict[str, torch.Tensor] = {}
    for layer in range(int(config.num_hidden_layers)):
        base = f"model.layers.{layer}"
        qkv_name = f"{base}.self_attn.qkv_proj.weight"
        qkv = shaped(qkv_name, _required(named, qkv_name), (q_rows + 2 * kv_rows, hidden))
        views[f"{base}.self_attn.q_proj.weight"] = qkv[:q_rows]
        views[f"{base}.self_attn.k_proj.weight"] = qkv[q_rows : q_rows + kv_rows]
        views[f"{base}.self_attn.v_proj.weight"] = qkv[q_rows + kv_rows :]

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
                "mlp.down_proj.weight": (hidden, intermediate),
                "input_layernorm.weight": (hidden,),
                "post_attention_layernorm.weight": (hidden,),
            },
            "Ministral",
        )

    expected_embed = (int(config.vocab_size), hidden)
    embed = _required(named, "model.embed_tokens.weight")
    if tuple(embed.shape) != expected_embed:
        raise ValueError(
            f"model.embed_tokens.weight is {tuple(embed.shape)}, expected {expected_embed}"
        )
    norm = shaped("model.norm.weight", _required(named, "model.norm.weight"), (hidden,))
    if bool(getattr(config, "tie_word_embeddings", False)):
        raise ValueError("the registered Ministral 14B capture weights are untied")
    lm_head = shaped("lm_head.weight", _required(named, "lm_head.weight"), expected_embed)
    if lm_head.untyped_storage().data_ptr() == embed.untyped_storage().data_ptr():
        raise ValueError("the registered Ministral 14B lm_head must have separate storage")
    views["model.embed_tokens.weight"] = embed
    views["model.norm.weight"] = norm
    views["lm_head.weight"] = lm_head
    return views


def _materialize_nonpersistent_buffers(model: Any, device: torch.device) -> None:
    rotary = model.model.rotary_emb
    with torch.device("cpu"):
        rebuilt = type(rotary)(config=rotary.config)
    model.model.rotary_emb = rebuilt.to(device)


def ministral3_hf_view(
    named: Mapping[str, torch.Tensor], config: Any, *, generation_config: Any = None
) -> Any:
    """Build an eval-mode HF Ministral whose parameters alias engine storage."""
    from transformers import Ministral3ForCausalLM

    return alias_hf_model(
        named,
        ministral3_view_state(named, config),
        family="Ministral",
        skeleton=lambda: Ministral3ForCausalLM(config),
        materialize=_materialize_nonpersistent_buffers,
        generation_config=generation_config,
    )


def _resident_tensors(model: Any, config: Any) -> dict[str, torch.Tensor]:
    named = resident_named_tensors(model, "Ministral")
    rows = _layer_geometry(config)
    check_qkv_splits(model, int(config.num_hidden_layers), lambda _layer: rows)
    return named


def dense_cache_from_artifact(
    artifact: MinistralDenseKV,
    config: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    """Bind extracted prompt KV into a dense HF cache for the latent roll.

    The extraction already lives on the capture device, so the moves below are
    no-ops and the dense cache is the gathered tensor, not a second copy.
    """
    from transformers import Cache
    from transformers.cache_utils import DynamicLayer

    count = int(config.num_hidden_layers)
    cache = Cache(layers=[DynamicLayer() for _ in range(count)])
    if len(artifact.layers) != count:
        raise RuntimeError(f"artifact/config layer counts are {len(artifact.layers)}/{count}")
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(config.head_dim)
    expected_shape = (1, kv_heads, artifact.logical_length, head_dim)
    for index, (record, target) in enumerate(zip(artifact.layers, cache.layers, strict=True)):
        target_layer = cast(Any, target)
        if record.layer_index != index:
            raise RuntimeError(
                f"artifact layer {record.layer_index} does not match config layer {index}"
            )
        if record.logical_length != artifact.logical_length or record.start_position != 0:
            raise RuntimeError(f"layer {index} is not a full dense prompt cache")
        if record.keys.shape != record.values.shape or tuple(record.keys.shape) != expected_shape:
            raise RuntimeError(f"layer {index} extracted cache shape is inconsistent")
        if not bool(torch.isfinite(record.keys).all()) or not bool(
            torch.isfinite(record.values).all()
        ):
            raise RuntimeError(f"layer {index} extracted cache contains a non-finite value")
        keys = record.keys.to(device=device, dtype=dtype)
        values = record.values.to(device=device, dtype=dtype)
        target_layer.lazy_initialization(keys, values)
        target_layer.keys, target_layer.values = keys, values
    if cache.get_seq_length() != artifact.logical_length:
        raise RuntimeError(
            f"rebuilt cache length {cache.get_seq_length()} != {artifact.logical_length}"
        )
    return cache


class MinistralEngineCaptureBridge(EngineCaptureBridge):
    """Borrow one live engine for dense prefill extraction and HF rollout."""

    family = "Ministral"
    request_prefix = "ministral-capture"

    @classmethod
    def from_llm(
        cls,
        llm: Any,
        *,
        text_config: Any,
        generation_config: Any = None,
        request_factory: RequestFactory | None = None,
    ) -> MinistralEngineCaptureBridge:
        """Adopt a raw in-process TP=1 vLLM 0.26 LLM without owning it."""
        resident = resident_language_model(llm, "Ministral")
        tensors = _resident_tensors(resident, text_config)
        model = ministral3_hf_view(
            tensors,
            text_config,
            generation_config=generation_config,
        )
        return cls(
            llm,
            model,
            llm.get_tokenizer(),
            request_factory=request_factory or vllm_request_factory,
        )

    def _mark(self, internal_id: str) -> None:
        request_ministral_extraction(internal_id)

    def _take(self, internal_id: str) -> Any:
        return take_ministral_extraction(internal_id)

    def _discard(self, internal_id: str) -> None:
        discard_ministral_extraction(internal_id)

    def _rebuild(
        self, artifact: Any, config: Any, *, device: torch.device, dtype: torch.dtype
    ) -> Any:
        return dense_cache_from_artifact(artifact, config, device=device, dtype=dtype)


__all__ = (
    "MinistralEngineCaptureBridge",
    "dense_cache_from_artifact",
    "ministral3_hf_view",
    "ministral3_view_state",
    "vllm_request_factory",
)
