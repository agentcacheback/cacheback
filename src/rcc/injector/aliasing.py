"""An HF-shaped decoder whose parameters are views into the engine's tensors.

The named transformers class is built without a second copy of the weights: the
fused splits are dim 0 slices and the skeleton is built on the meta device.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

#: The transformers classes this fusion map covers.
ALIASED_ARCHITECTURES = ("Qwen3ForCausalLM",)


def _require(named: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    """Return one fused engine tensor, detached, naming it if it is absent.

    `load_state_dict(assign=True)` installs Parameters by identity, so a detached view
    is handed over instead and the engine's own Parameter objects are left alone.
    """
    tensor = named.get(name)
    if tensor is None:
        raise ValueError(f"engine tensors lack {name!r}; the fusion map cannot alias")
    return tensor.detach()


def _check_rows(name: str, tensor: torch.Tensor, rows: int) -> torch.Tensor:
    """Check that the fused stacking produced the expected rows for this split."""
    if tensor.ndim != 2 or int(tensor.shape[0]) != rows:
        raise ValueError(
            f"{name} has shape {tuple(tensor.shape)}, expected {rows} rows; "
            "the fusion map does not match this engine"
        )
    return tensor


def _require_architecture(architecture: str) -> None:
    """Raise for an architecture this fusion map does not cover."""
    if architecture not in ALIASED_ARCHITECTURES:
        raise ValueError(
            f"aliasing has no fusion map for {architecture!r}; "
            f"the admitted architectures are {list(ALIASED_ARCHITECTURES)}"
        )


def fused_view_state(
    named: Mapping[str, torch.Tensor], config: Any, *, architecture: str
) -> dict[str, torch.Tensor]:
    """Map vllm-fused tensors to an HF state dict of zero-copy views.

    Splits q, k, v along dim 0 of `qkv_proj` and gate then up in `gate_up_proj`, the
    vllm 0.11.1 stacking order. A tied config ships no lm_head and aliases embeddings.
    """
    _require_architecture(architecture)
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    q_rows = int(config.num_attention_heads) * head_dim
    kv_rows = int(config.num_key_value_heads) * head_dim
    intermediate = int(config.intermediate_size)
    views: dict[str, torch.Tensor] = {}
    for layer in range(int(config.num_hidden_layers)):
        base = f"model.layers.{layer}"
        qkv = _check_rows(
            f"{base}.self_attn.qkv_proj.weight",
            _require(named, f"{base}.self_attn.qkv_proj.weight"),
            q_rows + 2 * kv_rows,
        )
        views[f"{base}.self_attn.q_proj.weight"] = qkv[:q_rows]
        views[f"{base}.self_attn.k_proj.weight"] = qkv[q_rows : q_rows + kv_rows]
        views[f"{base}.self_attn.v_proj.weight"] = qkv[q_rows + kv_rows :]
        gate_up = _check_rows(
            f"{base}.mlp.gate_up_proj.weight",
            _require(named, f"{base}.mlp.gate_up_proj.weight"),
            2 * intermediate,
        )
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
    """Rebuild the rotary module the way from_pretrained does: on the host, then moved.

    `inv_freq` is computed with a `pow` whose host and accelerator kernels disagree by
    1 to 2 ulp, so building on the accelerator would differ on every rotated key.
    """
    rotary = model.model.rotary_emb
    # The constructor takes no device argument, so without this context the
    # table lands on the caller's ambient factory device, which on an
    # accelerator reintroduces the divergence described above.
    with torch.device("cpu"):
        rebuilt = type(rotary)(rotary.config)
    rebuilt = rebuilt.to(device)
    rebuilt.original_inv_freq = rebuilt.inv_freq
    model.model.rotary_emb = rebuilt


def hf_view_model(
    named: Mapping[str, torch.Tensor],
    config: Any,
    *,
    generation_config: Any = None,
    architecture: str,
) -> Any:
    """Build an eval-mode `architecture` whose parameters alias `named`.

    The skeleton is built on the meta device, the detached views are assigned in, and
    every parameter ends `requires_grad` False so no stray backward copies the model.

    Args:
        named: The engine's fused parameter mapping.
        config: The model config the skeleton is built from.
        generation_config: The checkpoint's, needed whenever these decodes are compared
            against a from_pretrained model; the eos set and sampling defaults differ.
        architecture: The transformers class name to build.
    """
    import transformers

    _require_architecture(architecture)
    model_class: Any = getattr(transformers, architecture)
    # The engine's flags as found, before any aliasing: vllm 0.11.1 builds RMSNorm
    # weights as plain Parameters with requires_grad True while its linear and
    # embedding weights force it False, so the check below compares against these.
    grad_flags = {name: tensor.requires_grad for name, tensor in named.items()}
    views = fused_view_state(named, config, architecture=architecture)
    device = views["model.embed_tokens.weight"].device
    with torch.device("meta"):
        model = model_class(config)
    model.load_state_dict(views, assign=True, strict=True)
    _materialize_rotary(model, device)
    model.requires_grad_(False)
    source_ids = {id(tensor) for tensor in named.values()}
    for name, parameter in model.named_parameters():
        if parameter.device.type == "meta":
            raise ValueError(f"parameter {name} was not materialised by the fusion map")
        if id(parameter) in source_ids:
            raise ValueError(f"parameter {name} aliases an engine object by identity")
    for name, buffer in model.named_buffers():
        if buffer.device.type == "meta":
            raise ValueError(f"buffer {name} was not materialised")
    for name, tensor in named.items():
        if tensor.requires_grad != grad_flags[name]:
            raise ValueError(f"engine tensor {name} had requires_grad flipped; aliasing must not")
    if generation_config is not None:
        model.generation_config = generation_config
    return model.eval()
