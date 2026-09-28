"""An HF-shaped decoder whose parameters are views into the engine's tensors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def _require(named: Mapping[str, torch.Tensor], name: str, rows: int | None = None) -> torch.Tensor:
    """Return one engine tensor, detached, checking a fused matrix's stacked rows."""
    tensor = named.get(name)
    if tensor is None:
        raise ValueError(f"engine tensors lack {name!r}; the fusion map cannot alias")
    if rows is not None and (tensor.ndim != 2 or int(tensor.shape[0]) != rows):
        raise ValueError(
            f"{name} has shape {tuple(tensor.shape)}, expected {rows} rows; "
            "the fusion map does not match this engine"
        )
    return tensor.detach()


def fused_view_state(named: Mapping[str, torch.Tensor], config: Any) -> dict[str, torch.Tensor]:
    """Map vllm-fused tensors to an HF state dict of zero-copy views."""
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    q_rows = int(config.num_attention_heads) * head_dim
    kv_rows = int(config.num_key_value_heads) * head_dim
    intermediate = int(config.intermediate_size)
    views: dict[str, torch.Tensor] = {}
    for layer in range(int(config.num_hidden_layers)):
        base = f"model.layers.{layer}"
        qkv = _require(named, f"{base}.self_attn.qkv_proj.weight", q_rows + 2 * kv_rows)
        views[f"{base}.self_attn.q_proj.weight"] = qkv[:q_rows]
        views[f"{base}.self_attn.k_proj.weight"] = qkv[q_rows : q_rows + kv_rows]
        views[f"{base}.self_attn.v_proj.weight"] = qkv[q_rows + kv_rows :]
        gate_up = _require(named, f"{base}.mlp.gate_up_proj.weight", 2 * intermediate)
        views[f"{base}.mlp.gate_proj.weight"] = gate_up[:intermediate]
        views[f"{base}.mlp.up_proj.weight"] = gate_up[intermediate:]
        for name in (
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ):
            views[f"{base}.{name}"] = _require(named, f"{base}.{name}")
    embed = _require(named, "model.embed_tokens.weight")
    views["model.embed_tokens.weight"] = embed
    views["model.norm.weight"] = _require(named, "model.norm.weight")
    if bool(getattr(config, "tie_word_embeddings", False)):
        views["lm_head.weight"] = embed
    else:
        views["lm_head.weight"] = _require(named, "lm_head.weight")
    return views


def _materialize_rotary(model: Any, device: torch.device) -> None:
    """Rebuild the rotary module the way from_pretrained does: on the host, then moved."""
    rotary = model.model.rotary_emb
    # Build on the host to preserve the pretrained model's rotary rounding.
    with torch.device("cpu"):
        rebuilt = type(rotary)(rotary.config)
    rebuilt = rebuilt.to(device)
    rebuilt.original_inv_freq = rebuilt.inv_freq
    model.model.rotary_emb = rebuilt


def hf_view_model(named: Mapping[str, torch.Tensor], config: Any) -> Any:
    """Build an eval-mode Qwen3 view without copying or modifying the engine's weights."""
    from transformers import Qwen3ForCausalLM

    views = fused_view_state(named, config)
    device = views["model.embed_tokens.weight"].device
    with torch.device("meta"):
        model: Any = Qwen3ForCausalLM(config)
    model.load_state_dict(views, assign=True, strict=True)
    _materialize_rotary(model, device)
    model.requires_grad_(False)
    for name, buffer in model.named_buffers():
        if buffer.device.type == "meta":
            raise ValueError(f"buffer {name} was not materialised")
    return model.eval()
