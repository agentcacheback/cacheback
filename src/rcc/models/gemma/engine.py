"""vLLM 0.26 engine adapter for the resident Gemma decode channels."""

from __future__ import annotations

import gc
import importlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from types import TracebackType
from typing import Any

import torch

from rcc.models.gemma.executor import executor_class

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class EngineConfig:
    """Exact bring-up settings forwarded to one vLLM engine."""

    model: str
    revision: str | None = None
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.70
    max_model_len: int = 16_384
    max_num_seqs: int = 64
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    enable_prompt_embeds: bool = True
    seed: int = 0
    enable_sleep_mode: bool = False
    disable_log_stats: bool = True
    enable_prefix_caching: bool | None = None
    attention_config: tuple[tuple[str, Any], ...] | None = None
    kv_transfer_config: tuple[tuple[str, Any], ...] | None = None
    language_model_only: bool | None = None
    generation_config: str | None = None
    logprobs_mode: str | None = None
    max_logprobs: int | None = None


@dataclass(frozen=True)
class Sampling:
    """Sampling fields accepted by the resident vLLM seam."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    presence_penalty: float = 0.0
    max_tokens: int = 256
    seed: int | None = None
    stop_token_ids: tuple[int, ...] | None = None
    bad_words: tuple[str, ...] | None = None


@dataclass(frozen=True)
class DecodeResult:
    """Raw decode output and engine monotonic timestamps."""

    text: str
    finish_reason: str
    n_tokens: int
    token_ids: tuple[int, ...] = ()
    queued_ts: float | None = None
    scheduled_ts: float | None = None
    first_token_ts: float | None = None
    last_token_ts: float | None = None
    num_cached_tokens: int | None = None


def _require_vllm() -> tuple[Any, Any]:
    """Import vLLM lazily so CPU validation stays importable."""
    try:
        module: Any = importlib.import_module("vllm")
    except ImportError as exc:
        raise ImportError("resident Gemma requires vllm==0.26.0") from exc
    return module.LLM, module.SamplingParams


def _chunks(items: Sequence[Any], size: int | None) -> list[Sequence[Any]]:
    if size is None:
        return [items]
    if size <= 0:
        raise ValueError("batch size must be positive")
    return [items[start : start + size] for start in range(0, len(items), size)]


def _request_timestamps(output: Any) -> dict[str, float | None]:
    stats = getattr(output, "metrics", None)
    return {
        name: float(value) if (value := getattr(stats, name, None)) else None
        for name in ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts")
    }


def _check_submission_order(outputs: Sequence[Any], expected: int) -> None:
    if len(outputs) != expected:
        raise RuntimeError(f"generate returned {len(outputs)} outputs for {expected} requests")
    raw = [getattr(output, "request_id", None) for output in outputs]
    if any(value is None for value in raw):
        return
    try:
        request_ids = [int(str(value)) for value in raw]
    except (TypeError, ValueError) as exc:
        raise RuntimeError("vLLM request ids no longer preserve submission order") from exc
    if any(later <= earlier for earlier, later in pairwise(request_ids)):
        raise RuntimeError("vLLM outputs arrived out of submission order")


def _pool_tokens(llm: Any) -> int | None:
    engine = getattr(llm, "llm_engine", None)
    for holder in (getattr(engine, "vllm_config", None), engine):
        cache = getattr(holder, "cache_config", None)
        blocks = getattr(cache, "num_gpu_blocks", None)
        size = getattr(cache, "block_size", None)
        if isinstance(blocks, int) and isinstance(size, int) and blocks > 0 and size > 0:
            return blocks * size
    return None


def free_gpu_memory(settle_seconds: float = 5.0) -> tuple[float, float] | None:
    """Release Python and CUDA caches and report available device memory."""
    gc.collect()
    if not torch.cuda.is_available():
        return None
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(settle_seconds)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    gib = 1024**3
    return free_bytes / gib, total_bytes / gib


class DecodeEngine:
    """One sole-owner vLLM engine for token and embedding requests."""

    def __init__(self, config: EngineConfig) -> None:
        """Bring up the pinned engine and retain its exact configuration."""
        llm_cls, sampling_params = _require_vllm()
        kwargs: dict[str, Any] = {}
        if config.tensor_parallel_size == 1:
            kwargs["distributed_executor_backend"] = executor_class()
        for name in (
            "revision",
            "enable_prefix_caching",
            "language_model_only",
            "generation_config",
            "logprobs_mode",
            "max_logprobs",
        ):
            value = getattr(config, name)
            if value is not None:
                kwargs[name] = value
        if config.attention_config is not None:
            kwargs["attention_config"] = dict(config.attention_config)
        if config.kv_transfer_config is not None:
            module: Any = importlib.import_module("vllm.config")
            kwargs["kv_transfer_config"] = module.KVTransferConfig(
                **dict(config.kv_transfer_config)
            )
        self._llm: Any | None = llm_cls(
            model=config.model,
            dtype=config.dtype,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            max_num_seqs=config.max_num_seqs,
            tensor_parallel_size=config.tensor_parallel_size,
            enforce_eager=config.enforce_eager,
            enable_prompt_embeds=config.enable_prompt_embeds,
            seed=config.seed,
            enable_sleep_mode=config.enable_sleep_mode,
            disable_log_stats=config.disable_log_stats,
            **kwargs,
        )
        self._SamplingParams = sampling_params
        self._dtype = _DTYPE_MAP[config.dtype]
        self._config = config

    @property
    def config(self) -> EngineConfig:
        """Return the exact engine configuration."""
        return self._config

    @property
    def raw_llm(self) -> Any:
        """Return the live in-process LLM for resident capture."""
        if self._llm is None:
            raise RuntimeError("engine has been closed")
        return self._llm

    def _generate_full(
        self,
        requests: list[Any],
        sampling: Sampling,
        seeds: Sequence[int] | None,
        batch_size: int | None,
    ) -> list[DecodeResult]:
        if self._llm is None:
            raise RuntimeError("engine has been closed")
        resolved = [sampling.seed] * len(requests) if seeds is None else list(seeds)
        if len(resolved) != len(requests):
            raise ValueError("seed roster differs from request roster")
        base: dict[str, Any] = {
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
        }
        if sampling.top_k > 0:
            base["top_k"] = sampling.top_k
        if sampling.presence_penalty != 0:
            base["presence_penalty"] = sampling.presence_penalty
        if sampling.stop_token_ids is not None:
            base["stop_token_ids"] = list(sampling.stop_token_ids)
        if sampling.bad_words is not None:
            base["bad_words"] = list(sampling.bad_words)
        results: list[DecodeResult] = []
        offset = 0
        for chunk in _chunks(requests, batch_size):
            params = [
                self._SamplingParams(**base, seed=seed)
                for seed in resolved[offset : offset + len(chunk)]
            ]
            offset += len(chunk)
            outputs = self._llm.generate(chunk, sampling_params=params)
            _check_submission_order(outputs, len(chunk))
            for output in outputs:
                first = output.outputs[0]
                cached = getattr(output, "num_cached_tokens", None)
                results.append(
                    DecodeResult(
                        text=str(first.text),
                        finish_reason=str(first.finish_reason),
                        n_tokens=len(first.token_ids),
                        token_ids=tuple(map(int, first.token_ids)),
                        num_cached_tokens=int(cached) if cached is not None else None,
                        **_request_timestamps(output),
                    )
                )
        return results

    def decode_token_ids_full(
        self,
        prompts: Sequence[Sequence[int]],
        sampling: Sampling,
        *,
        seeds: Sequence[int] | None = None,
        batch_size: int | None = None,
    ) -> list[DecodeResult]:
        """Decode caller-tokenized prompts with raw token readout."""
        requests = [{"prompt_token_ids": list(ids)} for ids in prompts]
        return self._generate_full(requests, sampling, seeds, batch_size)

    def decode_embeds_full(
        self,
        embeds: Sequence[torch.Tensor],
        sampling: Sampling,
        *,
        seeds: Sequence[int] | None = None,
        batch_size: int | None = None,
    ) -> list[DecodeResult]:
        """Decode rank-two embedding rows with raw token readout."""
        if not self._config.enable_prompt_embeds:
            raise ValueError("engine was built without prompt embedding support")
        if any(value.ndim != 2 for value in embeds):
            raise ValueError("embedding requests must be [rows, hidden]")
        requests = [
            {"prompt_embeds": value.to(dtype=self._dtype, device="cpu")} for value in embeds
        ]
        return self._generate_full(requests, sampling, seeds, batch_size)

    def kv_pool_tokens(self) -> int | None:
        """Return the live KV pool capacity in tokens when readable."""
        if self._llm is None:
            raise RuntimeError("engine has been closed")
        return _pool_tokens(self._llm)

    def close(self) -> None:
        """Release the sole vLLM handle and its GPU allocations."""
        llm = self._llm
        self._llm = None
        try:
            if llm is not None:
                llm.llm_engine.engine_core.shutdown()
        finally:
            del llm
            free_gpu_memory()

    def __enter__(self) -> DecodeEngine:
        """Return the live engine."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the engine when leaving its context."""
        self.close()


def build_engine(config: EngineConfig) -> DecodeEngine:
    """Bring up one resident decode engine without hidden defaults."""
    return DecodeEngine(config)


__all__ = (
    "DecodeEngine",
    "DecodeResult",
    "EngineConfig",
    "Sampling",
    "build_engine",
    "free_gpu_memory",
)
