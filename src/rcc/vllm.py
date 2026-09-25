"""Adapt existing vLLM Qwen3 prompt state; no request submission or generation."""

from __future__ import annotations

import copy
from typing import Any, cast

import torch

from rcc.capture.aliasing import hf_view_model
from rcc.capture.connector import discard_capture as discard_capture
from rcc.capture.connector import get_capture, require_runtime
from rcc.capture.connector import request_capture as request_capture
from rcc.transport import SenderState


def sender_from_vllm(
    llm: Any,
    request_id: str,
    token_ids: torch.Tensor,
    *,
    inherited_positions: int = 0,
    allow_unstable: bool = False,
) -> SenderState:
    """Consume a captured prompt, sharing the existing engine's weights without rerunning it."""
    from transformers import DynamicCache

    require_runtime(allow_unstable)
    engine = llm.llm_engine
    config = copy.deepcopy(engine.vllm_config.model_config.hf_config)
    if config.model_type != "qwen3":
        raise ValueError("sender_from_vllm currently supports dense Qwen3 only")
    if token_ids.ndim != 1 or token_ids.dtype != torch.long or token_ids.numel() == 0:
        raise ValueError("token_ids must be the exact nonempty int64 prompt vector")
    if bool(((token_ids < 0) | (token_ids >= config.vocab_size)).any()):
        raise ValueError("token_ids contain an ID outside the sender vocabulary")
    runner = getattr(engine.model_executor.driver_worker, "model_runner", None)
    if runner is None:
        raise RuntimeError("sender capture needs the in-process engine with multiprocessing=0")
    resident = runner.model
    expected = [config.num_attention_heads * config.head_dim] + [
        config.num_key_value_heads * config.head_dim
    ] * 2
    for layer in resident.model.layers:
        if list(layer.self_attn.qkv_proj.output_sizes) != expected:
            raise ValueError("resident QKV shards do not match the unsharded Qwen3 config")
    config._attn_implementation = "sdpa"
    model = hf_view_model(dict(resident.named_parameters()), config)
    artifact = get_capture(request_id)
    if (
        artifact.logical_length != token_ids.numel()
        or len(artifact.layers) != config.num_hidden_layers
    ):
        raise ValueError("captured prompt length or layer count differs from the sender")
    past = cast(Any, DynamicCache())
    for index, record in enumerate(artifact.layers):
        past.update(record.keys, record.values, index)
    with torch.inference_mode():
        rows = model.get_input_embeddings()(
            token_ids.to(model.get_input_embeddings().weight.device)
        )
    state = SenderState(
        model,
        past,
        rows,
        tokenizer=llm.get_tokenizer(),
        token_ids=token_ids,
        inherited_positions=inherited_positions,
    )
    state.validate()
    discard_capture(request_id)
    return state
