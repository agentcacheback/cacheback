"""Resident Qwen capture adapter for the pinned vLLM 0.11.1 route.

One prefill is driven through the in-process engine, its staged prompt pages are
read, the latent steps are rolled on them, and the rolled cache is scored.
"""

from __future__ import annotations

import importlib
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, cast

import torch

from rcc import KVCache
from rcc.injector.aliasing import hf_view_model
from rcc.injector.extraction import EXTRACTED, REQUESTED, request_extraction
from rcc.injector.staging import StagedCache
from rcc.latent.cache import model_rope
from rcc.latent.rollout import build_realign, latent_rollout
from rcc.models.qwen.capture import QWEN_LATENT_STEPS, capture_qwen_scores
from rcc.models.qwen.engine_routes import (
    QWEN_CAPTURE_WINDOWS,
    QWEN_CHAIN_ENGINE_GPU_MEMORY_UTILIZATION,
    QWEN_CHAIN_ENGINE_MAX_BATCHED_TOKENS,
    QWEN_CHAIN_ENGINE_MAX_MODEL_LEN,
    QWEN_CHAIN_ENGINE_ROUTE,
    QWEN_ENGINE_GPU_MEMORY_UTILIZATION,
    QWEN_ENGINE_MAX_BATCHED_TOKENS,
    QWEN_ENGINE_MAX_MODEL_LEN,
    QWEN_ENGINE_MAX_NUM_SEQS,
    QWEN_ENGINE_ROUTE,
    QWEN_NATURAL_ENGINE_MAX_MODEL_LEN,
    QwenEngineSettings,
)
from rcc.models.route import RouteFamily
from rcc.run.fleet.progress import tick_seat_progress

RequestFactory = Callable[[str, list[int]], tuple[Any, Any]]
EmbedsRequestFactory = Callable[[str, torch.Tensor], tuple[Any, Any]]

QWEN_VLLM_VERSION = "0.11.1"
_ENGINE_CHAIN = ("llm_engine", "engine_core", "engine_core", "model_executor", "driver_worker")


def _require_vllm_pin() -> None:
    observed = version("vllm")
    if observed != QWEN_VLLM_VERSION:
        raise RuntimeError(f"Qwen capture requires vllm {QWEN_VLLM_VERSION}, found {observed}")


def vllm_request_factory(request_id: str, token_ids: list[int]) -> tuple[Any, Any]:
    """Build the one-token greedy request used only to capture prompt pages."""
    del request_id
    _require_vllm_pin()
    vllm = importlib.import_module("vllm")
    inputs = importlib.import_module("vllm.inputs")
    sampling_params = cast(Any, vllm).SamplingParams
    tokens_prompt = cast(Any, inputs).TokensPrompt

    return (
        tokens_prompt(prompt_token_ids=[int(token) for token in token_ids]),
        sampling_params(temperature=0.0, max_tokens=1, ignore_eos=True),
    )


def vllm_embeds_request_factory(request_id: str, rows: torch.Tensor) -> tuple[Any, Any]:
    """Build the one-token greedy request that prefills from embedding rows.

    The connector reads ``prompt_embeds`` off a request whose token ids are
    None, so extraction keys on the request id exactly as it does for tokens.
    """
    del request_id
    _require_vllm_pin()
    vllm = importlib.import_module("vllm")
    sampling_params = cast(Any, vllm).SamplingParams

    return (
        {"prompt_embeds": rows.detach().to("cpu", torch.bfloat16)},
        sampling_params(temperature=0.0, max_tokens=1, ignore_eos=True),
    )


def engine_named_parameters(llm: Any, config: Any) -> dict[str, torch.Tensor]:
    """Reach and validate vLLM 0.11.1's resident fused Qwen tensors."""
    _require_vllm_pin()
    handle = llm
    for step in _ENGINE_CHAIN:
        handle = getattr(handle, step, None)
        if handle is None:
            raise RuntimeError(
                f"engine reach failed at {step!r}; the in-process vLLM chain changed"
            )
    model = None
    for tail in (("worker", "model_runner", "model"), ("model_runner", "model")):
        candidate = handle
        for step in tail:
            candidate = getattr(candidate, step, None)
            if candidate is None:
                break
        if candidate is not None:
            model = candidate
            break
    if model is None:
        raise RuntimeError("engine reach failed after driver_worker; no model_runner.model")
    params = dict(model.named_parameters())
    witness = "model.layers.0.self_attn.qkv_proj.weight"
    if witness not in params:
        raise RuntimeError(f"engine parameters lack {witness!r}; not the pinned Qwen layout")
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    expected = [
        int(config.num_attention_heads) * head_dim,
        int(config.num_key_value_heads) * head_dim,
        int(config.num_key_value_heads) * head_dim,
    ]
    declared = list(model.model.layers[0].self_attn.qkv_proj.output_sizes)
    if declared != expected:
        raise RuntimeError(
            f"engine QKV shards {declared} disagree with config arithmetic {expected}"
        )
    return params


def aliased_from_engine(
    llm: Any, *, config: Any, family: RouteFamily, generation_config: Any = None
) -> Any:
    """Build a zero-copy HF-shaped view over vLLM's sole resident weights."""
    return hf_view_model(
        engine_named_parameters(llm, config),
        config,
        generation_config=generation_config,
        architecture=family.hf_architecture,
    )


@dataclass(frozen=True)
class EngineHandle:
    """The sole live engine and its zero-copy HF-shaped weight view."""

    engine: Any
    model: Any
    request_factory: RequestFactory | None = None
    embeds_request_factory: EmbedsRequestFactory | None = None


@dataclass(frozen=True)
class WorkerProduct:
    """One exact prompt-plus-40 roll and its fused production capture."""

    embeds: torch.Tensor
    scores: dict[str, torch.Tensor]
    length: int
    question_found: bool
    extract_s: float
    roll_s: float
    capture_s: float
    capture_peak_bytes: int
    mamba_signals: dict[int, dict[str, torch.Tensor]] | None = None


@dataclass(frozen=True)
class HopProduct:
    """One chain hop: a worker product plus the geometry it was built from."""

    embeds: torch.Tensor
    scores: dict[str, torch.Tensor]
    length: int
    question_found: bool
    extract_s: float
    roll_s: float
    capture_s: float
    capture_peak_bytes: int
    prefix_rows: int
    prompt_rows: int


def _as_worker(product: HopProduct) -> WorkerProduct:
    """Drop the hop geometry so the FanOutQA path keeps its public type."""
    return WorkerProduct(
        embeds=product.embeds,
        scores=product.scores,
        length=product.length,
        question_found=product.question_found,
        extract_s=product.extract_s,
        roll_s=product.roll_s,
        capture_s=product.capture_s,
        capture_peak_bytes=product.capture_peak_bytes,
    )


class EngineProducer:
    """Single-owner vLLM prefill, extraction, roll, and capture adapter."""

    def __init__(
        self,
        handle: EngineHandle,
        *,
        latent_steps: int = QWEN_LATENT_STEPS,
        deadline_seconds: float = 600.0,
    ) -> None:
        """Bind one engine owner and the exact side-pass roll geometry."""
        if latent_steps < 0:
            raise ValueError("latent_steps must be nonnegative")
        if deadline_seconds <= 0:
            raise ValueError("deadline_seconds must be positive")
        self._engine = handle.engine
        self._model = handle.model
        self._request_factory = handle.request_factory or vllm_request_factory
        self._embeds_request_factory = handle.embeds_request_factory or vllm_embeds_request_factory
        self._latent_steps = latent_steps
        self._deadline_seconds = deadline_seconds
        self.admission_failures = 0

    @property
    def device(self) -> torch.device:
        """Return the device used by the resident side pass."""
        return next(self._model.parameters()).device

    @property
    def model(self) -> Any:
        """Return the zero-copy HF-shaped resident view."""
        return self._model

    def _abort(self, request_id: str) -> None:
        abort = getattr(self._engine, "abort_request", None)
        if abort is None:
            return
        try:
            abort([request_id])
        except Exception:
            pass

    @staticmethod
    def _own_request_finished(output: Any, request_id: str) -> bool:
        """Validate cache evidence and report whether this request finished."""
        if output.request_id != request_id:
            return False
        if getattr(output, "num_cached_tokens", None) not in (None, 0):
            raise RuntimeError(f"{request_id}: prefix caching contaminated Qwen capture")
        return bool(getattr(output, "finished", False))

    def _drive(self, request_id: str, prompt: Any, sampling: Any) -> None:
        try:
            self._engine.add_request(request_id, prompt, sampling)
        except Exception:
            self.admission_failures += 1
            raise
        finished = False
        elapsed = 0.0
        steps = 0
        started = time.perf_counter()
        try:
            while not finished and self._engine.has_unfinished_requests():
                steps += 1
                for output in self._engine.step():
                    if self._own_request_finished(output, request_id):
                        finished = True
                elapsed = time.perf_counter() - started
                # An in-process deadline cannot survive a stuck step, so this
                # loop records its own progress. The tick is throttled to one
                # staged write every five seconds.
                tick_seat_progress(steps)
                if not finished and elapsed > self._deadline_seconds:
                    break
        except Exception:
            self._abort(request_id)
            raise
        if finished:
            return
        self._abort(request_id)
        if elapsed > self._deadline_seconds:
            raise RuntimeError(
                f"engine request {request_id!r} did not finish within "
                f"{self._deadline_seconds:.1f}s (waited {elapsed:.1f}s)"
            )
        raise RuntimeError(f"engine request {request_id!r} never finished")

    def _extract(self, build: Callable[[str], tuple[Any, Any]]) -> tuple[StagedCache, float]:
        """Drive one capture request and read the pages it staged."""
        request_id = f"produce-{uuid.uuid4().hex}"
        request_extraction(request_id)
        started = time.perf_counter()
        try:
            prompt, sampling = build(request_id)
            self._drive(request_id, prompt, sampling)
            entry = EXTRACTED.load(request_id)
        finally:
            EXTRACTED.finish(request_id)
            REQUESTED.discard(request_id)
        return entry, time.perf_counter() - started

    def extract(self, token_ids: Sequence[int]) -> tuple[StagedCache, float]:
        """Prefill token ids and read their prompt pages without exporting foreign KV.

        The pages are gathered on the engine's device and stay there, so the
        clock covers the prefill and the gather and no host round trip.
        """
        if not token_ids:
            raise ValueError("an extraction prefill needs at least one token")
        ids = [int(token) for token in token_ids]
        return self._extract(lambda request_id: self._request_factory(request_id, ids))

    def extract_embeds(self, rows: torch.Tensor) -> tuple[StagedCache, float]:
        """Prefill embedding rows and read their prompt pages, as `extract` does.

        A chain hop past the first carries rows that were never tokens, so only
        the request shape differs from the token path.
        """
        if rows.ndim != 2 or int(rows.shape[0]) < 1:
            raise ValueError("an embeddings prefill needs rows shaped [N, D] with N >= 1")
        return self._extract(lambda request_id: self._embeds_request_factory(request_id, rows))

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up the resident input embeddings of one id block."""
        with torch.no_grad():
            return self._model.get_input_embeddings()(token_ids.to(self.device))

    def cache_from_extraction(self, entry: StagedCache) -> KVCache:
        """Bind an extracted prompt to the resident model device and dtype.

        The copy is a no-op when the entry is already on that device and dtype;
        the cast covers an entry staged elsewhere.
        """
        parameter = next(self._model.parameters())
        return KVCache(
            keys=entry.keys.to(device=parameter.device, dtype=parameter.dtype),
            values=entry.values.to(device=parameter.device, dtype=parameter.dtype),
            positions=entry.positions.to(parameter.device),
            rope=model_rope(self._model),
        )

    def capture_scores(
        self,
        past: Any,
        prompt_ids: torch.Tensor,
        judger_ids: torch.Tensor,
        judger_mask: torch.Tensor,
        question_ids: Sequence[int],
        *,
        consume_past: bool = False,
    ) -> tuple[dict[str, torch.Tensor], bool]:
        """Score the live rolled cache before producer ownership releases it."""
        return capture_qwen_scores(
            self._model, past, judger_ids, judger_mask, question_ids, consume_past=consume_past
        )

    def produce_worker(
        self,
        prompt_ids: torch.Tensor,
        *,
        judger_ids: torch.Tensor,
        judger_mask: torch.Tensor,
        question_ids: Sequence[int],
    ) -> WorkerProduct:
        """Produce embedding rows and scores, dropping full KV before returning."""
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] < 2:
            raise ValueError("prompt_ids must be [1, L] with L >= 2")
        return _as_worker(
            self.produce_hop(
                None,
                prompt_ids,
                judger_ids=judger_ids,
                judger_mask=judger_mask,
                question_ids=question_ids,
            )
        )

    def _prefill_hop(
        self,
        prefix_rows: torch.Tensor | None,
        head_ids: torch.Tensor,
    ) -> tuple[StagedCache, float, torch.Tensor | None, int]:
        """Prefill one hop and report its staged pages, rows, and prefix width.

        Hop one submits token ids and needs no rows, so it returns None and
        embeds its prompt later, inside the roll clock.
        """
        if prefix_rows is None:
            prefill = cast(Any, head_ids[0].detach().cpu())
            entry, extract_s = self.extract(cast(list[int], prefill.tolist()))
            return entry, extract_s, None, 0
        if prefix_rows.ndim != 3 or prefix_rows.shape[0] != 1:
            raise ValueError("prefix_rows must be [1, P, D]")
        head_rows = self._embed(head_ids)
        head_rows = torch.cat((prefix_rows.to(head_rows), head_rows), dim=1)
        entry, extract_s = self.extract_embeds(head_rows[0])
        return entry, extract_s, head_rows, int(prefix_rows.shape[1])

    def produce_hop(
        self,
        prefix_rows: torch.Tensor | None,
        tail_ids: torch.Tensor,
        *,
        judger_ids: torch.Tensor,
        judger_mask: torch.Tensor,
        question_ids: Sequence[int],
        consume_past: bool = False,
    ) -> HopProduct:
        """Produce one hop over a carried prefix, this hop's prompt, and the roll.

        Hop one carries no prefix and prefills the prompt minus its held-back
        seed as ids; a later hop prefills carried rows and prompt as embeddings.
        """
        if tail_ids.ndim != 2 or tail_ids.shape[0] != 1 or tail_ids.shape[1] < 2:
            raise ValueError("tail_ids must be [1, L] with L >= 2")
        prompt_rows = int(tail_ids.shape[1])
        head_ids = tail_ids[:, :-1]
        entry, extract_s, head_rows, prefix_count = self._prefill_hop(prefix_rows, head_ids)
        cache = self.cache_from_extraction(entry)
        if cache.length != prefix_count + prompt_rows - 1:
            raise RuntimeError(
                "Qwen hop extraction length differs from prefix plus held-back-seed geometry"
            )
        measure_cuda = self.device.type == "cuda"
        if measure_cuda:
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        # The HF cache views the staged tensors and the roll's first step
        # replaces every layer view with a grown copy, so dropping the staged
        # handles here lets that step free the gathered copy.
        past = cache.to_hf_cache()
        del cache, entry
        rolled = latent_rollout(
            self._model,
            tail_ids[:, -1:].to(self.device),
            latent_steps=self._latent_steps,
            realign=build_realign(self._model, enabled=False),
            past=past,
            record_embeds=True,
        )
        if rolled.embeds is None:
            raise RuntimeError("Qwen rollout did not record its seed and latent rows")
        if head_rows is None:
            head_rows = self._embed(head_ids)
        embeds = torch.cat((head_rows, rolled.embeds), dim=1)
        if measure_cuda:
            torch.cuda.synchronize(self.device)
        roll_s = time.perf_counter() - started
        base_bytes = 0
        if measure_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            base_bytes = torch.cuda.memory_allocated(self.device)
        started = time.perf_counter()
        # The rolled length is read before the capture: a consumed cache comes
        # back with the judger rows appended, and the geometry is the roll's.
        observed = int(rolled.past.get_seq_length())
        scores, found = self.capture_scores(
            rolled.past,
            tail_ids,
            judger_ids,
            judger_mask,
            question_ids,
            consume_past=consume_past,
        )
        if measure_cuda:
            torch.cuda.synchronize(self.device)
        capture_s = time.perf_counter() - started
        peak = int(torch.cuda.max_memory_allocated(self.device) - base_bytes) if measure_cuda else 0
        expected = prefix_count + prompt_rows + self._latent_steps
        if observed != expected or int(embeds.shape[1]) != expected:
            raise RuntimeError(
                "Qwen roll differs from prefix plus prompt plus latent-step geometry"
            )
        return HopProduct(
            embeds=embeds,
            scores=scores,
            length=observed,
            question_found=found,
            extract_s=extract_s,
            roll_s=roll_s,
            capture_s=capture_s,
            capture_peak_bytes=peak,
            prefix_rows=prefix_count,
            prompt_rows=prompt_rows,
        )


_installed: EngineHandle | None = None


def install_engine_handle(handle: EngineHandle) -> None:
    """Install one process-wide resident handle and refuse co-tenancy."""
    global _installed
    if _installed is not None:
        raise RuntimeError("a Qwen engine handle is already installed; clear it first")
    _installed = handle


def installed_producer() -> EngineProducer:
    """Build the exact 40-step production adapter around the resident engine."""
    if _installed is None:
        raise RuntimeError("no Qwen engine handle is installed; call install_engine_handle first")
    return EngineProducer(_installed, latent_steps=QWEN_LATENT_STEPS)


__all__ = (
    "QWEN_CAPTURE_WINDOWS",
    "QWEN_CHAIN_ENGINE_GPU_MEMORY_UTILIZATION",
    "QWEN_CHAIN_ENGINE_MAX_BATCHED_TOKENS",
    "QWEN_CHAIN_ENGINE_MAX_MODEL_LEN",
    "QWEN_CHAIN_ENGINE_ROUTE",
    "QWEN_ENGINE_GPU_MEMORY_UTILIZATION",
    "QWEN_ENGINE_MAX_BATCHED_TOKENS",
    "QWEN_ENGINE_MAX_MODEL_LEN",
    "QWEN_ENGINE_MAX_NUM_SEQS",
    "QWEN_ENGINE_ROUTE",
    "QWEN_NATURAL_ENGINE_MAX_MODEL_LEN",
    "QWEN_VLLM_VERSION",
    "EmbedsRequestFactory",
    "EngineHandle",
    "EngineProducer",
    "HopProduct",
    "QwenEngineSettings",
    "WorkerProduct",
    "aliased_from_engine",
    "engine_named_parameters",
    "install_engine_handle",
    "installed_producer",
    "vllm_embeds_request_factory",
    "vllm_request_factory",
)
