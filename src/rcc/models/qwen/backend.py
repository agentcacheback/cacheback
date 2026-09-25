"""Pinned vLLM 0.11.1 construction and decode seams for Qwen FanOutQA.

One engine opens in the role the backend settings name, and its decode seams
keep the prompt and completion ids the caller submitted.
"""

from __future__ import annotations

import gc
import importlib
import os
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, cast

import torch

from rcc.models.qwen.engine import (
    QWEN_CAPTURE_WINDOWS,
    QWEN_ENGINE_GPU_MEMORY_UTILIZATION,
    QWEN_ENGINE_MAX_MODEL_LEN,
    QWEN_ENGINE_MAX_NUM_SEQS,
    EngineHandle,
    QwenEngineSettings,
    aliased_from_engine,
)
from rcc.models.qwen.engine import QWEN_VLLM_VERSION as QWEN_VLLM_VERSION
from rcc.models.qwen.engine_routes import (
    QWEN_CHAIN_ENGINE_MAX_MODEL_LEN,
    QWEN_CHAIN_ONE_STEP_PREFILL_GPU_MEMORY_UTILIZATION,
)
from rcc.models.qwen.text import QwenDecodeRequest
from rcc.models.qwen.text_output import TextCompletion
from rcc.models.route import RouteFamily
from rcc.run.fleet.latency import WARMUP_MAX_TOKENS


@dataclass(frozen=True)
class QwenBackendSettings:
    """The only live Qwen engine roles admitted by the fleet worker.

    ``chain`` is a capture role, with prompt embeddings on because a hop past
    the first prefills rows that were never tokens.
    """

    capture: bool = False
    receiver: bool = False
    #: The producer window; the registered 50,000 unless a natural panel is
    #: served.
    capture_max_model_len: int = QWEN_ENGINE_MAX_MODEL_LEN
    chain: bool = False
    #: Diagnostic only: open the chain engine with the prefill in one step. No
    #: result row reads its output.
    chain_one_step_prefill: bool = False

    def __post_init__(self) -> None:
        """Reject a mixed capture/receiver engine, and unregistered settings."""
        if self.capture and self.receiver:
            raise ValueError("Qwen capture and receiver require separate engine roles")
        if self.capture_max_model_len not in QWEN_CAPTURE_WINDOWS:
            raise ValueError("Qwen capture window is not a registered capture window")
        if self.chain and not self.capture:
            raise ValueError("the Qwen chain engine role is a capture role; pass capture=True")
        if self.chain_one_step_prefill and not self.chain:
            raise ValueError("one-step prefill is a chain diagnostic; pass chain=True")


def _chain_capture_overrides(
    *, family: RouteFamily, one_step_prefill: bool = False
) -> dict[str, object]:
    """Return the chain capture tuple, read off the settings that validate it.

    The connector fields arrive as ordered pairs and become the mapping the
    vLLM constructor takes.
    """
    overrides = QwenEngineSettings.for_chain().engine_overrides(family=family)
    overrides["kv_transfer_config"] = dict(
        cast(tuple[tuple[str, str], ...], overrides["kv_transfer_config"])
    )
    if one_step_prefill:
        # The whole prompt in one prefill step, at the memory share that leaves
        # room for it. Applied after the validated read.
        overrides["max_num_batched_tokens"] = QWEN_CHAIN_ENGINE_MAX_MODEL_LEN
        overrides["gpu_memory_utilization"] = QWEN_CHAIN_ONE_STEP_PREFILL_GPU_MEMORY_UTILIZATION
    return overrides


def effective_engine_kwargs(
    checkpoint: str,
    revision: str,
    *,
    settings: QwenBackendSettings,
    family: RouteFamily,
    tokenizer: str,
    tokenizer_revision: str,
) -> dict[str, object]:
    """Return the one source of truth for every vLLM constructor field.

    The tokenizer is passed apart from the checkpoint because a sender arm may
    be served by a checkpoint whose chat template is not the family's.
    """
    kwargs: dict[str, object] = {
        "model": checkpoint,
        "revision": revision,
        "tokenizer": tokenizer,
        "tokenizer_revision": tokenizer_revision,
        "dtype": "bfloat16",
        "gpu_memory_utilization": (
            QWEN_ENGINE_GPU_MEMORY_UTILIZATION if settings.capture else 0.80
        ),
        "max_model_len": (
            settings.capture_max_model_len
            if settings.capture
            else int(dict(family.profile.runtime.engine_flags)["max_model_len"])
        ),
        "max_num_seqs": QWEN_ENGINE_MAX_NUM_SEQS,
        "enable_prompt_embeds": settings.receiver,
        "enable_prefix_caching": False,
        "disable_log_stats": settings.capture,
        "hf_overrides": dict(family.hf_overrides),
    }
    if settings.capture:
        kwargs.update(
            {
                "max_num_batched_tokens": settings.capture_max_model_len,
                "enforce_eager": True,
                "disable_hybrid_kv_cache_manager": True,
                "kv_transfer_config": {
                    "kv_connector": "RCCConnector",
                    "kv_role": "kv_both",
                    "kv_connector_module_path": "rcc.injector.connector",
                },
            }
        )
    if settings.capture and not settings.chain:
        # Validate the registered fan-out tuple and the window this seat
        # serves; the kwargs above spell the same values.
        QwenEngineSettings.for_fanoutqa()
        QwenEngineSettings.for_fanoutqa(settings.capture_max_model_len)
    if settings.chain:
        kwargs.update(
            _chain_capture_overrides(
                family=family, one_step_prefill=settings.chain_one_step_prefill
            )
        )
    if family.lane == "nemotron":
        from rcc.models.nemotron.backend import engine_kwargs

        return engine_kwargs(
            kwargs,
            capture=settings.capture,
            chain=settings.chain,
            one_step_prefill=settings.chain_one_step_prefill,
        )
    return kwargs


@dataclass
class _Completion:
    request_id: str
    prompt_token_ids: Sequence[int]
    text: str
    n_tokens: int
    token_ids: Sequence[int]
    finish_reason: str
    num_cached_tokens: int | None
    queued_ts: float | None
    scheduled_ts: float | None
    first_token_ts: float | None


def _metrics(output: Any) -> dict[str, float | None]:
    metrics = getattr(output, "metrics", None)

    def value(name: str) -> float | None:
        raw = getattr(metrics, name, None)
        return float(raw) if raw is not None else None

    # vLLM 0.11.1 attaches V1 ``RequestStateStats`` (queued_ts, scheduled_ts,
    # first_token_ts, one monotonic clock). The older V0 names bank None, which
    # would leave the report queue and TTFT means empty.
    return {
        "queued_ts": value("queued_ts"),
        "scheduled_ts": value("scheduled_ts"),
        "first_token_ts": value("first_token_ts"),
    }


class QwenVllmBackend:
    """One resident Qwen engine with signed synchronous and streaming seams."""

    def __init__(self, llm: Any, settings: QwenBackendSettings, family: RouteFamily) -> None:
        """Bind one already-constructed pinned engine to its declared role."""
        self._llm: Any | None = llm
        self.settings = settings
        self.family = family

    @property
    def raw_llm(self) -> Any:
        """Return the live in-process vLLM object."""
        if self._llm is None:
            raise RuntimeError("Qwen vLLM backend is closed")
        return self._llm

    def engine_handle(self) -> EngineHandle:
        """Expose the capture engine and its zero-copy HF-shaped weight view."""
        if not self.settings.capture:
            raise RuntimeError("only a capture backend exposes an EngineProducer handle")
        llm = self.raw_llm
        config = llm.model_config.hf_config
        return EngineHandle(
            engine=llm.llm_engine,
            model=aliased_from_engine(llm, config=config, family=self.family),
        )

    def resident_model(self) -> Any:
        """Return a zero-copy HF-shaped view for local receiver embeddings."""
        llm = self.raw_llm
        return aliased_from_engine(llm, config=llm.model_config.hf_config, family=self.family)

    def decode_text_full(
        self,
        prompts: Sequence[str],
        requests: Sequence[QwenDecodeRequest],
    ) -> Sequence[TextCompletion]:
        """Decode signed requests and retain backend request identity and raw ids.

        The rendered chat already carries its template's special tokens, so it
        is encoded with none added: a string would gain a second BOS on Llama 3.
        """
        tokenizer = self.raw_llm.get_tokenizer()
        encoded = [
            [int(token) for token in tokenizer(prompt, add_special_tokens=False)["input_ids"]]
            for prompt in prompts
        ]
        return self.decode_token_ids_full(encoded, requests)

    def decode_token_ids_full(
        self,
        prompts: Sequence[Sequence[int]],
        requests: Sequence[QwenDecodeRequest],
    ) -> Sequence[TextCompletion]:
        """Decode exact caller-owned prompt ids without retokenizing them."""
        prepared: list[dict[str, list[int]]] = []
        submitted: list[tuple[int, ...]] = []
        for index, prompt in enumerate(prompts):
            tokens = [int(token) for token in prompt]
            if not tokens or any(token < 0 for token in tokens):
                raise ValueError(f"Qwen token prompt {index} is empty or malformed")
            prepared.append({"prompt_token_ids": tokens})
            submitted.append(tuple(tokens))
        completed = self._decode_full(prepared, requests)
        if any(
            tuple(result.prompt_token_ids) != prompt
            for result, prompt in zip(completed, submitted, strict=True)
        ):
            raise RuntimeError("Qwen vLLM returned prompt token ids that differ from submission")
        return completed

    def _decode_full(
        self,
        prompts: Sequence[Any],
        requests: Sequence[QwenDecodeRequest],
    ) -> Sequence[TextCompletion]:
        """Decode prepared vLLM prompts and retain prompt and completion ids."""
        if len(prompts) != len(requests):
            raise ValueError("Qwen prompt/request rosters differ")
        vllm = importlib.import_module("vllm")
        params: list[Any] = []
        for request in requests:
            values = dict(request.decode.backend_sampling())
            params.append(cast(Any, vllm).SamplingParams(**values, seed=request.seed))
        outputs = self.raw_llm.generate(list(prompts), sampling_params=params, use_tqdm=False)
        if len(outputs) != len(requests):
            raise RuntimeError("Qwen vLLM returned an incomplete request roster")
        completed: list[_Completion] = []
        for request, output in zip(requests, outputs, strict=True):
            choices = getattr(output, "outputs", ())
            if len(choices) != 1:
                raise RuntimeError(f"{request.request_id}: Qwen vLLM choice roster differs")
            choice = choices[0]
            prompt_token_ids = getattr(output, "prompt_token_ids", None)
            if not isinstance(prompt_token_ids, list) or not prompt_token_ids:
                raise RuntimeError(f"{request.request_id}: Qwen vLLM omitted prompt token ids")
            if any(type(token) is not int for token in cast(list[object], prompt_token_ids)):
                raise RuntimeError(f"{request.request_id}: Qwen vLLM returned malformed prompt ids")
            validated_prompt_ids = cast(list[int], prompt_token_ids)
            cached = getattr(output, "num_cached_tokens", None)
            completed.append(
                _Completion(
                    request_id=request.request_id,
                    prompt_token_ids=tuple(validated_prompt_ids),
                    text=str(choice.text),
                    n_tokens=len(choice.token_ids),
                    token_ids=tuple(int(token) for token in choice.token_ids),
                    finish_reason=str(choice.finish_reason),
                    num_cached_tokens=int(cached) if cached is not None else None,
                    **_metrics(output),
                )
            )
        return completed

    def warmup_decode(self, prompt: str, *, seed: int) -> int:
        """Run one discarded two-token generation and return its token count.

        The first request through a fresh engine pays for lazy allocation and
        graph capture. Native senders warm up under their registered sampling.
        """
        vllm = importlib.import_module("vllm")
        sampling = (
            self.family.profile.decode.backend_sampling()
            if self.family.native_sender_prompts
            else {"temperature": 0.0}
        )
        params = cast(Any, vllm).SamplingParams(
            **sampling,
            max_tokens=WARMUP_MAX_TOKENS,
            seed=seed,
        )
        outputs = self.raw_llm.generate([prompt], sampling_params=[params], use_tqdm=False)
        choices = getattr(outputs[0], "outputs", ()) if len(outputs) == 1 else ()
        if len(choices) != 1:
            raise RuntimeError("Qwen warmup decode returned an incomplete roster")
        return len(choices[0].token_ids)

    def kv_pool_tokens(self) -> int | None:
        """Return vLLM's live GPU-block capacity when the pinned route exposes it."""
        engine = getattr(self.raw_llm, "llm_engine", None)
        for holder in (getattr(engine, "vllm_config", None), engine):
            cache = getattr(holder, "cache_config", None)
            blocks = getattr(cache, "num_gpu_blocks", None)
            block_size = getattr(cache, "block_size", None)
            if (
                isinstance(blocks, int)
                and isinstance(block_size, int)
                and blocks > 0
                and block_size > 0
            ):
                return blocks * block_size
        return None

    def close(self) -> None:
        """Release the one resident engine and return cached allocator pages."""
        self._llm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_qwen_backend(
    checkpoint: str,
    revision: str,
    *,
    settings: QwenBackendSettings,
    family: RouteFamily,
    tokenizer: str,
    tokenizer_revision: str,
) -> QwenVllmBackend:
    """Construct one exact engine role without importing vLLM before execution."""
    observed = version("vllm")
    expected = family.profile.runtime.vllm
    if observed != expected:
        raise RuntimeError(f"{family.lane} execution requires vllm {expected}, found {observed}")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    policy_before: dict[str, Any] = {}
    if family.lane == "nemotron":
        from rcc.models.nemotron.runtime_policy import before_engine

        policy_before = before_engine()
    vllm = importlib.import_module("vllm")
    kwargs = effective_engine_kwargs(
        checkpoint,
        revision,
        settings=settings,
        family=family,
        tokenizer=tokenizer,
        tokenizer_revision=tokenizer_revision,
    )
    if "kv_transfer_config" in kwargs:
        config = importlib.import_module("vllm.config")
        transfer = cast(dict[str, object], kwargs.pop("kv_transfer_config"))
        kwargs["kv_transfer_config"] = cast(Any, config).KVTransferConfig(**transfer)
    llm = cast(Any, vllm).LLM(**kwargs)
    if family.lane == "nemotron":
        from rcc.models.nemotron.backend import NemotronBackend

        return NemotronBackend(llm, settings, family, policy_before=policy_before)
    return QwenVllmBackend(llm, settings, family)


__all__ = (
    "QwenBackendSettings",
    "QwenVllmBackend",
    "build_qwen_backend",
    "effective_engine_kwargs",
)
