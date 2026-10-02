"""Bind the paper's resident vLLM prefill, shared-weight selection and input-row handoff."""

from __future__ import annotations

import copy
import importlib
import time
import uuid
from typing import Any, cast

import torch

from rclc.capture.aliasing import hf_view_model
from rclc.capture.connector import discard_capture as discard_capture
from rclc.capture.connector import get_capture, require_runtime, validate_capture_config
from rclc.capture.connector import request_capture as request_capture
from rclc.transport import SenderState


def require_idle(llm: Any) -> None:
    """Keep shared-weight HF work outside the engine's active scheduling loop."""
    if llm.llm_engine.has_unfinished_requests():
        raise RuntimeError(
            "keep the vLLM engine idle while binding, transferring or updating state"
        )


def model_from_vllm(llm: Any, *, allow_unstable: bool = False) -> Any:
    """Expose the resident dense Qwen3 weights without loading a second checkpoint."""
    require_runtime(allow_unstable)
    require_idle(llm)
    engine = llm.llm_engine
    config = copy.deepcopy(engine.vllm_config.model_config.hf_config)
    if config.model_type != "qwen3":
        raise ValueError("the vLLM adapter currently supports dense Qwen3 only")
    core = getattr(getattr(engine, "engine_core", None), "engine_core", None)
    executor = getattr(core, "model_executor", None)
    worker = getattr(executor, "driver_worker", None)
    worker = getattr(worker, "worker", worker)
    runner = getattr(worker, "model_runner", None)
    if runner is None:
        raise RuntimeError("sender capture needs the in-process engine with multiprocessing=0")
    config._attn_implementation = "sdpa"
    return hf_view_model(dict(runner.model.named_parameters()), config)


def _check_prompt(model: Any, token_ids: torch.Tensor, input_embeds: torch.Tensor | None) -> None:
    """Reject a malformed prompt before the engine or the capture registry sees it."""
    if token_ids.ndim != 1 or token_ids.dtype != torch.long or token_ids.numel() == 0:
        raise ValueError("token_ids must be the exact nonempty int64 prompt vector")
    lowest = 0 if input_embeds is None else -1
    if bool(((token_ids < lowest) | (token_ids >= model.config.vocab_size)).any()):
        raise ValueError("token_ids contain an ID outside the sender vocabulary")
    if input_embeds is not None and (
        input_embeds.ndim != 2 or input_embeds.shape[0] != token_ids.numel()
    ):
        raise ValueError("input_embeds must hold one [hidden] row per token ID")


def state_from_capture(
    model: Any,
    tokenizer: Any,
    request_id: str,
    token_ids: torch.Tensor,
    *,
    inherited_positions: int = 0,
    input_embeds: torch.Tensor | None = None,
) -> SenderState:
    """Consume a validated captured prompt using an existing shared-weight model view."""
    from transformers import DynamicCache

    _check_prompt(model, token_ids, input_embeds)
    artifact = get_capture(request_id)
    if (
        artifact.logical_length != token_ids.numel()
        or len(artifact.layers) != model.config.num_hidden_layers
    ):
        raise ValueError("captured prompt length or layer count differs from the sender")
    past = cast(Any, DynamicCache())
    for index, record in enumerate(artifact.layers):
        past.update(record.keys, record.values, index)
    with torch.inference_mode():
        rows = input_embeds
        if rows is None:
            rows = model.get_input_embeddings()(
                token_ids.to(model.get_input_embeddings().weight.device)
            )
    state = SenderState(
        model,
        past,
        rows,
        tokenizer=tokenizer,
        token_ids=token_ids,
        inherited_positions=inherited_positions,
    )
    state.validate()
    discard_capture(request_id)
    return state


def sender_from_vllm(
    llm: Any,
    request_id: str,
    token_ids: torch.Tensor,
    *,
    inherited_positions: int = 0,
    allow_unstable: bool = False,
) -> SenderState:
    """Consume an existing engine capture without submitting or rerunning its prompt."""
    model = model_from_vllm(llm, allow_unstable=allow_unstable)
    return state_from_capture(
        model, llm.get_tokenizer(), request_id, token_ids, inherited_positions=inherited_positions
    )


def validate_binding_config(llm: Any) -> Any:
    """Check the loaded engine settings without allocating a model view or running it."""
    if not hasattr(llm, "llm_engine"):
        raise ValueError("pass an existing vLLM LLM, or use backend='hf' for an HF model")
    config = llm.llm_engine.vllm_config
    if not config.model_config.enable_prompt_embeds:
        raise ValueError("create the vLLM engine with enable_prompt_embeds=True before binding")
    validate_capture_config(config)
    capture = config.kv_transfer_config
    if capture is None or capture.kv_connector != "RCCCaptureConnector":
        raise ValueError("create the vLLM engine with RCCCaptureConnector before binding")
    return config


def binding_model(
    llm: Any,
    tokenizer: Any,
    context_limit: int | None,
    allow_unstable: bool,
) -> tuple[Any, Any, int]:
    """Validate unified binding requirements and honor the engine's actual context limit."""
    config = validate_binding_config(llm)
    model = model_from_vllm(llm, allow_unstable=allow_unstable)
    limit = min(config.model_config.max_model_len, model.config.max_position_embeddings)
    if context_limit is not None and context_limit > limit:
        raise ValueError("context_limit exceeds the vLLM engine's configured context limit")
    native_tokenizer = llm.get_tokenizer()
    if tokenizer is not None and tokenizer.get_vocab() != native_tokenizer.get_vocab():
        raise ValueError("use the vLLM engine's tokenizer when binding")
    return model, native_tokenizer, limit if context_limit is None else context_limit


def generation_inputs(rows: torch.Tensor, max_new_tokens: int) -> dict[str, Any]:
    """Build native LLM.generate arguments without submitting a request."""
    sampling_params = importlib.import_module("vllm").SamplingParams

    return {
        "prompts": {"prompt_embeds": rows.detach().cpu()},
        "sampling_params": sampling_params(max_tokens=max_new_tokens),
        "use_tqdm": False,
    }


def completion_ids(generation: Any) -> torch.Tensor:
    """Read one finished vLLM completion; streaming fragments and multiple outputs are ambiguous."""
    if isinstance(generation, (list, tuple)):
        batch = cast(list[Any] | tuple[Any, ...], generation)
        if len(batch) != 1:
            raise ValueError("provide exactly one vLLM request output")
        generation = batch[0]
    outputs = getattr(generation, "outputs", ())
    if not getattr(generation, "finished", False) or len(outputs) != 1:
        raise ValueError("provide one finished vLLM completion with n=1")
    ids = outputs[0].token_ids
    if not ids or not all(type(token) is int for token in ids):
        raise ValueError("vLLM completion must contain integer token IDs")
    return torch.tensor([ids], dtype=torch.long)


@torch.inference_mode()
def prefill_state(
    llm: Any,
    model: Any,
    tokenizer: Any,
    token_ids: torch.Tensor,
    *,
    input_embeds: torch.Tensor | None = None,
    inherited_positions: int = 0,
) -> SenderState:
    """Run the paper's one-token capture request; only prompt rows enter sender state."""
    sampling_params = importlib.import_module("vllm").SamplingParams

    require_idle(llm)
    _check_prompt(model, token_ids, input_embeds)
    if token_ids.numel() + 1 > llm.llm_engine.vllm_config.model_config.max_model_len:
        raise ValueError("sender prompt needs space for the vLLM capture request's one token")
    prompt: dict[str, Any] = (
        {"prompt_token_ids": cast(Any, token_ids).tolist()}
        if input_embeds is None
        else {"prompt_embeds": input_embeds.detach().cpu()}
    )
    engine = llm.llm_engine
    external_id = f"rcc-{uuid.uuid4().hex}"
    returned_id = engine.add_request(
        external_id, prompt, sampling_params(max_tokens=1, temperature=0, ignore_eos=True)
    )
    capture_id = returned_id if isinstance(returned_id, str) else external_id
    finished = False
    try:
        request_capture(capture_id)
        deadline = time.monotonic() + 600
        while not finished and engine.has_unfinished_requests():
            for output in engine.step():
                if output.request_id in (external_id, capture_id):
                    if getattr(output, "num_cached_tokens", None) not in (None, 0):
                        raise RuntimeError("prefix caching contaminated the capture request")
                    finished = bool(output.finished)
            if time.monotonic() > deadline:
                raise RuntimeError("vLLM capture exceeded 600 seconds")
        if not finished:
            raise RuntimeError("vLLM capture request did not finish")
        return state_from_capture(
            model,
            tokenizer,
            capture_id,
            token_ids,
            input_embeds=input_embeds,
            inherited_positions=inherited_positions,
        )
    finally:
        try:
            if not finished:
                engine.abort_request([external_id])
        finally:
            discard_capture(capture_id)
