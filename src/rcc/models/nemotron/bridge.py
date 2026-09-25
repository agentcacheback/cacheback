"""TP1 native Nemotron view of the vLLM 0.26 resident checkpoint."""

from collections.abc import Mapping
from typing import Any

import torch


def resident_model(llm: Any) -> Any:
    """Reach the in-process TP1 model, refusing remote or sharded execution."""
    engine = llm.llm_engine
    executor = getattr(engine, "model_executor", None)
    driver = getattr(executor, "driver_worker", None)
    runner = getattr(driver, "model_runner", None)
    model = getattr(runner, "model", None)
    if model is None:
        raise RuntimeError("Nemotron requires in-process vLLM 0.26 driver_worker.model_runner")
    return model


def view_state(
    named: Mapping[str, torch.Tensor], target: Mapping[str, torch.Tensor], config: Any
) -> dict[str, torch.Tensor]:
    """Alias TP1 projections and invert vLLM's small negative-exponential A buffer."""
    q_rows = int(config.num_attention_heads) * int(config.head_dim)
    kv_rows = int(config.num_key_value_heads) * int(config.head_dim)
    offsets = {
        "q": (0, q_rows),
        "k": (q_rows, q_rows + kv_rows),
        "v": (q_rows + kv_rows, q_rows + 2 * kv_rows),
    }
    state: dict[str, torch.Tensor] = {}
    for name, expected in target.items():
        source = name.replace("model.embeddings.", "model.embed_tokens.")
        if source.endswith(".A_log"):
            value = named[source.removesuffix("_log")].detach()
            if not bool((value < 0).all()) or not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"{source}: vLLM A must be finite and negative")
            value = (-value.float()).log()
        elif any(f".{part}_proj." in source for part in offsets):
            part = next(part for part in offsets if f".{part}_proj." in source)
            packed = named[source.replace(f".{part}_proj.", ".qkv_proj.")].detach()
            lo, hi = offsets[part]
            value = packed[lo:hi]
        else:
            value = named[source].detach()
        if value.shape != expected.shape:
            raise RuntimeError(f"{name}: resident tensor shape or finite-value check failed")
        state[name] = value
    return state


def native_view(llm: Any) -> Any:
    """Build an eval-only native skeleton with large weights sharing engine storage."""
    from transformers import NemotronHConfig, NemotronHForCausalLM

    factory: Any = NemotronHConfig
    native_config = factory.from_dict(llm.model_config.hf_config.to_dict())
    native_config._attn_implementation = "sdpa"
    native_config.use_mamba_kernels = False
    native_config.dtype = torch.bfloat16
    with torch.device("meta"):
        model = NemotronHForCausalLM(native_config)
    named = dict(resident_model(llm).named_parameters())
    state = view_state(named, model.state_dict(), native_config)
    model.load_state_dict(state, assign=True, strict=True)
    model.eval()
    # Detached Parameter wrappers own autograd flags independently of vLLM.
    model.requires_grad_(False)
    if next(model.parameters()).device.type == "cuda":
        from rcc.models.nemotron.kernels import enable_native_kernels

        enable_native_kernels(model)
    if any(p.device.type == "meta" for p in model.parameters()):
        raise RuntimeError("Nemotron native view retained an unmaterialized parameter")
    return model
