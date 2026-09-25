"""Open one pinned engine for a capture or a decode seat."""

from __future__ import annotations

import importlib
from importlib.metadata import version
from typing import Any, cast

import torch
from transformers import AutoTokenizer, GenerationConfig

from rcc.models.qwen import QWEN_FAMILY
from rcc.models.qwen.backend import QwenBackendSettings, QwenVllmBackend, effective_engine_kwargs
from rcc.run.selector_dev.capture import DevProducer
from rcc.run.selector_dev.contract import CELLS, MODELS


def open_engine(model: str, *, capture: bool) -> tuple[Any, Any, dict[str, Any]]:
    """Open one single-GPU engine, after checking the runtime it needs."""
    if version("vllm") != "0.11.1" or version("transformers") != "4.57.1":
        raise RuntimeError("dev execution requires vLLM 0.11.1 and transformers 4.57.1")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("each dev process must own exactly one visible CUDA GPU")
    device: Any = cast(Any, torch.cuda).get_device_properties(0)
    if not any(name in device.name for name in ("H100", "H200")):
        raise RuntimeError("dev execution requires H100 or H200")
    if model == "qwen3-32b" and device.total_memory < 128 * 1024**3:
        raise RuntimeError("32B TP1 capture requires H200; the H100 TP2 bridge is not admitted")
    repository, revision = MODELS[model]
    tokenizer: Any = cast(Any, AutoTokenizer).from_pretrained(
        repository, revision=revision, local_files_only=True
    )
    generation: Any = cast(Any, GenerationConfig).from_pretrained(
        repository, revision=revision, local_files_only=True
    )
    if tokenizer.eos_token_id != 151645 or generation.eos_token_id != [151645, 151643]:
        raise RuntimeError("tokenizer or generation stops differ from frozen Qwen dev")
    settings = QwenBackendSettings(capture=capture, receiver=not capture)
    kwargs = effective_engine_kwargs(
        repository,
        revision,
        settings=settings,
        family=QWEN_FAMILY,
        tokenizer=repository,
        tokenizer_revision=revision,
    )
    # The larger checkpoint needs more room for resident weights.
    kwargs["gpu_memory_utilization"] = (
        0.60 if capture and model == "qwen3-32b" else (0.40 if capture else 0.80)
    )
    kwargs["max_num_seqs"] = 6 if capture else CELLS
    if not capture:
        kwargs["max_model_len"] = 56_000
    runtime: dict[str, Any] = {
        "gpu": device.name,
        "gpu_memory_bytes": device.total_memory,
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "vllm": version("vllm"),
        "engine_kwargs": kwargs.copy(),
    }
    vllm: Any = importlib.import_module("vllm")
    if "kv_transfer_config" in kwargs:
        config: Any = importlib.import_module("vllm.config")
        kwargs["kv_transfer_config"] = config.KVTransferConfig(
            **cast(dict[str, Any], kwargs["kv_transfer_config"])
        )
    llm = vllm.LLM(**kwargs)
    if capture:
        backend = QwenVllmBackend(llm, settings, QWEN_FAMILY)
        producer = DevProducer(backend.engine_handle())
        producer.tokenizer = tokenizer
        return producer, tokenizer, runtime
    return llm, tokenizer, runtime
