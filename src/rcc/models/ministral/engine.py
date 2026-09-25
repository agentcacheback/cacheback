"""Pinned vLLM 0.26 engine for resident Ministral decode and capture.

One set of weights serves the token, embedding, and capture channels in any of
the three registered roles.
"""

from __future__ import annotations

import gc
import importlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

import torch

from rcc.models.ministral import MINISTRAL
from rcc.models.ministral.capture_bridge import MinistralEngineCaptureBridge
from rcc.models.ministral.runtime import (
    CAPTURE_CONNECTOR as _CAPTURE_CONNECTOR,
)
from rcc.models.ministral.runtime import (
    CAPTURE_ENGINE_ROLE as _CAPTURE_ENGINE_ROLE,
)
from rcc.models.ministral.runtime import (
    REGISTERED_ENGINE_FLAGS as _REGISTERED_FLAGS,
)
from rcc.models.ministral.runtime import (
    SPLIT_RECEIVER_ENGINE_ROLE as _SPLIT_RECEIVER_ENGINE_ROLE,
)
from rcc.models.ministral.runtime import (
    observed_runtime_signature as _observed_runtime_signature,
)
from rcc.models.ministral.runtime import runtime_fingerprint, validate_runtime_signature
from rcc.run.fleet.vllm import VllmEngineHandle

if TYPE_CHECKING:
    from rcc.models.ministral.receiver import (
        MinistralReceiverRequest,
        MinistralWarmupRequest,
    )
    from rcc.models.ministral.text import MinistralDecodeRequest

_SAMPLING_FIELDS = frozenset(
    {"temperature", "top_p", "presence_penalty", "max_tokens", "stop_token_ids"}
)


@dataclass(frozen=True)
class MinistralEngineConfig:
    """Operational settings outside the registered engine identity."""

    gpu_memory_utilization: float = 0.85
    enforce_eager: bool = False
    seed: int = 0
    disable_log_stats: bool = False

    def __post_init__(self) -> None:
        """Reject invalid operational limits before vLLM owns GPU memory."""
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must lie in (0, 1]")
        if self.seed < 0:
            raise ValueError("engine seed must be nonnegative")


@dataclass(frozen=True)
class MinistralDecodeResult:
    """One raw completion with logical identity and engine timing evidence."""

    request_id: str
    text: str
    finish_reason: str
    n_tokens: int
    token_ids: tuple[int, ...]
    num_cached_tokens: int | None
    queued_ts: float | None
    scheduled_ts: float | None
    first_token_ts: float | None
    last_token_ts: float | None


def _require_vllm() -> tuple[Any, Any, Any, Any]:
    """Import only the pinned vLLM interfaces after runtime verification."""
    try:
        module: Any = importlib.import_module("vllm")
        config_module: Any = importlib.import_module("vllm.config")
        inputs_module: Any = importlib.import_module("vllm.inputs")
    except ImportError as exc:
        raise ImportError("resident Ministral requires vllm==0.26.0") from exc
    return (
        module.LLM,
        module.SamplingParams,
        config_module.KVTransferConfig,
        inputs_module.TokensPrompt,
    )


def _request_timestamps(output: Any) -> dict[str, float | None]:
    """Read all monotonic request timestamps without replacing zero values."""
    metrics = getattr(output, "metrics", None)
    timestamps: dict[str, float | None] = {}
    for name in ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts"):
        value = getattr(metrics, name, None)
        if value is None:
            timestamps[name] = None
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"vLLM returned invalid {name} {value!r}")
        else:
            timestamps[name] = float(value)
    return timestamps


def _pool_tokens(llm: Any) -> int | None:
    """Read the resident KV-pool token capacity when vLLM exposes it."""
    engine = getattr(llm, "llm_engine", None)
    for holder in (getattr(engine, "vllm_config", None), engine):
        cache = getattr(holder, "cache_config", None)
        blocks = getattr(cache, "num_gpu_blocks", None)
        block_size = getattr(cache, "block_size", None)
        if (
            isinstance(blocks, int)
            and not isinstance(blocks, bool)
            and isinstance(block_size, int)
            and not isinstance(block_size, bool)
            and blocks > 0
            and block_size > 0
        ):
            return blocks * block_size
    return None


def _sampling(request: Any) -> dict[str, object]:
    """Validate one request against the sole registered Ministral decode."""
    decode: Any = getattr(request, "decode", None)
    backend_sampling = getattr(decode, "backend_sampling", None)
    to_dict = getattr(decode, "to_dict", None)
    if not callable(backend_sampling) or not callable(to_dict):
        raise TypeError("Ministral request lacks its registered decode specification")
    raw_sampling = backend_sampling()
    raw_identity = to_dict()
    if not isinstance(raw_sampling, Mapping) or not isinstance(raw_identity, Mapping):
        raise TypeError("Ministral decode specification returned a malformed mapping")
    values = dict(cast(Mapping[str, object], raw_sampling))
    identity = cast(Mapping[str, object], raw_identity)
    if frozenset(values) != _SAMPLING_FIELDS:
        raise RuntimeError(f"Ministral sampling fields drifted: {sorted(values)}")
    expected_family = {
        "temperature": MINISTRAL.decode.temperature,
        "top_p": MINISTRAL.decode.top_p,
        "presence_penalty": MINISTRAL.decode.presence_penalty,
    }
    if any(values[name] != expected for name, expected in expected_family.items()):
        raise RuntimeError("Ministral request sampling differs from family-native registration")
    if "top_k" in values or identity.get("top_k") is not None:
        raise RuntimeError("Ministral request enabled top-k")
    if (
        identity.get("decode_profile") != MINISTRAL.decode.profile_id
        or identity.get("decode_fingerprint") != MINISTRAL.decode.identity_hash
        or identity.get("enable_thinking") is not True
    ):
        raise RuntimeError("Ministral request decode identity differs from registration")
    max_tokens = values["max_tokens"]
    stops = values["stop_token_ids"]
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
        or not isinstance(stops, list)
        or not stops
    ):
        raise RuntimeError("Ministral request has invalid decode bounds")
    stop_tokens = cast(list[object], stops)
    if any(isinstance(token, bool) or not isinstance(token, int) for token in stop_tokens):
        raise RuntimeError("Ministral request has invalid decode bounds")
    return values


def _text_config(llm: Any) -> Any:
    """Resolve the decoder configuration from the checkpoint's own HF config."""
    model_config = getattr(llm, "model_config", None)
    outer = getattr(model_config, "hf_config", None)
    if outer is None:
        raise RuntimeError("Ministral vLLM engine omitted its HF configuration")
    resolve_text = getattr(outer, "get_text_config", None)
    if callable(resolve_text):
        synthesized = resolve_text(decoder=True)
    else:
        synthesized = getattr(outer, "text_config", None)
    if synthesized is None:
        raise RuntimeError("Ministral vLLM config cannot resolve its decoder text config")
    # vLLM synthesizes a bare config from the mistral-format checkpoint and
    # drops attributes the HF modeling reads, so the HF view is configured from
    # the checkpoint's own HF config and the dimensions are checked to agree.
    transformers = importlib.import_module("transformers")
    typed_model_config = cast(Any, model_config)
    genuine = transformers.AutoConfig.from_pretrained(
        typed_model_config.model, revision=typed_model_config.revision
    ).get_text_config(decoder=True)
    for name in (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
    ):
        if hasattr(synthesized, name) and getattr(synthesized, name) != getattr(genuine, name):
            raise RuntimeError(f"Ministral decoder config drifted on {name}")
    return genuine


def _completion_tokens(completion: Any, request_id: str) -> tuple[int, ...]:
    """Validate and freeze one raw vLLM token-id vector."""
    raw_tokens = getattr(completion, "token_ids", None)
    if not isinstance(raw_tokens, list):
        raise RuntimeError(f"Ministral request {request_id!r} returned malformed token ids")
    values = cast(list[object], raw_tokens)
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in values):
        raise RuntimeError(f"Ministral request {request_id!r} returned malformed token ids")
    return tuple(cast(list[int], raw_tokens))


def _completion_result(output: Any, request_id: str) -> MinistralDecodeResult:
    """Convert one finished vLLM output without losing raw evidence."""
    candidates = cast(Sequence[Any], getattr(output, "outputs", ()))
    if len(candidates) != 1:
        raise RuntimeError(f"Ministral request {request_id!r} returned {len(candidates)} choices")
    completion = candidates[0]
    finish_reason = getattr(completion, "finish_reason", None)
    if not isinstance(finish_reason, str) or not finish_reason:
        raise RuntimeError(f"Ministral request {request_id!r} omitted its finish reason")
    cached = getattr(output, "num_cached_tokens", None)
    if cached is not None and (
        isinstance(cached, bool) or not isinstance(cached, int) or cached < 0
    ):
        raise RuntimeError(f"Ministral request {request_id!r} returned invalid cache evidence")
    token_ids = _completion_tokens(completion, request_id)
    return MinistralDecodeResult(
        request_id=request_id,
        text=str(getattr(completion, "text", "")),
        finish_reason=finish_reason,
        n_tokens=len(token_ids),
        token_ids=token_ids,
        num_cached_tokens=cached,
        **_request_timestamps(output),
    )


def _text_sender_pin(semantic_arm: str) -> tuple[str, str]:
    """Return one exact registered text sender checkpoint and revision."""
    arms = {arm.semantic_arm: arm for arm in MINISTRAL.physical_arms}
    arm = arms.get(semantic_arm)
    if (
        semantic_arm not in {"text_primary", "text_medium", "text_small"}
        or arm is None
        or arm.sender_checkpoint is None
        or arm.sender_revision is None
    ):
        raise ValueError(f"unregistered Ministral text sender {semantic_arm!r}")
    return arm.sender_checkpoint, arm.sender_revision


class MinistralEngine:
    """One vLLM weight owner serving token, embedding, and capture channels."""

    def __init__(
        self,
        config: MinistralEngineConfig | None = None,
        *,
        text_sender_arm: str | None = None,
        split_receiver: bool = False,
    ) -> None:
        """Open the registered capture, split-receiver, or text-sender engine."""
        if split_receiver and text_sender_arm is not None:
            raise ValueError("a Ministral split receiver is never a text sender engine")
        self._config = config or MinistralEngineConfig()
        self._text_sender_arm = text_sender_arm
        checkpoint, revision = (
            (MINISTRAL.checkpoint, MINISTRAL.revision)
            if text_sender_arm is None
            else _text_sender_pin(text_sender_arm)
        )
        current_v1 = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
        if current_v1 not in {None, "0"}:
            raise RuntimeError("Ministral requires VLLM_ENABLE_V1_MULTIPROCESSING=0")
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        # A split receiver injects prompt embeddings and captures nothing, so
        # it is the one 14B role that opens without the transfer stanza.
        capture = text_sender_arm is None and not split_receiver
        engine_role = text_sender_arm or (
            _SPLIT_RECEIVER_ENGINE_ROLE if split_receiver else _CAPTURE_ENGINE_ROLE
        )
        engine_kwargs: dict[str, object] = {
            "model": checkpoint,
            "revision": revision,
            "tokenizer": checkpoint,
            "tokenizer_revision": revision,
            "dtype": _REGISTERED_FLAGS["dtype"],
            "tokenizer_mode": _REGISTERED_FLAGS["tokenizer_mode"],
            "config_format": _REGISTERED_FLAGS["config_format"],
            "load_format": _REGISTERED_FLAGS["load_format"],
            "language_model_only": True,
            "gpu_memory_utilization": self._config.gpu_memory_utilization,
            "max_model_len": int(_REGISTERED_FLAGS["max_model_len"]),
            "max_num_seqs": int(_REGISTERED_FLAGS["max_num_seqs"]),
            "tensor_parallel_size": int(_REGISTERED_FLAGS["tensor_parallel_size"]),
            "enforce_eager": self._config.enforce_eager,
            "enable_prompt_embeds": text_sender_arm is None,
            "enable_prefix_caching": False,
            "seed": self._config.seed,
            "disable_log_stats": self._config.disable_log_stats,
        }
        if capture:
            engine_kwargs["kv_transfer_config"] = dict(_CAPTURE_CONNECTOR)
        self._runtime_signature = _observed_runtime_signature(
            engine_kwargs,
            engine_role=engine_role,
        )
        validate_runtime_signature(
            self._runtime_signature,
            active_checkpoint=checkpoint,
            active_revision=revision,
            engine_role=engine_role,
        )
        self._runtime_fingerprint = runtime_fingerprint(self._runtime_signature)
        llm_cls, sampling_params, transfer_cls, tokens_prompt = _require_vllm()
        self._SamplingParams: Any = sampling_params
        self._TokensPrompt: Any = tokens_prompt
        if capture:
            engine_kwargs["kv_transfer_config"] = transfer_cls(**_CAPTURE_CONNECTOR)
        self._llm: Any | None = llm_cls(
            **engine_kwargs,
        )
        self._handle: VllmEngineHandle | None = None
        self._bridge: MinistralEngineCaptureBridge | None = None
        try:
            self._handle = VllmEngineHandle(self, require_public_enqueue=True)
            if text_sender_arm is None:
                self._bridge = MinistralEngineCaptureBridge.from_llm(
                    self.raw_llm,
                    text_config=_text_config(self.raw_llm),
                )
        except Exception:
            self.close()
            raise

    @property
    def config(self) -> MinistralEngineConfig:
        """Return operational settings that do not alter engine identity."""
        return self._config

    @property
    def raw_llm(self) -> Any:
        """Return the live in-process vLLM owner."""
        if self._llm is None:
            raise RuntimeError("Ministral engine has been closed")
        return self._llm

    @property
    def capture_bridge(self) -> MinistralEngineCaptureBridge:
        """Return the capture bridge borrowing this engine's resident weights."""
        if self._bridge is None:
            raise RuntimeError("Ministral capture view is unavailable")
        return self._bridge

    @property
    def resident_model(self) -> Any:
        """Return the zero-copy HF-shaped view over resident vLLM weights."""
        return self.capture_bridge.model

    @property
    def embedding_weight(self) -> torch.Tensor:
        """Return the resident input-embedding tensor used for receiver rows."""
        weight = self.resident_model.get_input_embeddings().weight
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise RuntimeError("Ministral resident embedding weight is malformed")
        return weight

    @property
    def runtime_signature(self) -> dict[str, object]:
        """Return a copy of the verified observed runtime signature."""
        return cast(dict[str, object], json.loads(json.dumps(self._runtime_signature)))

    @property
    def runtime_fingerprint(self) -> str:
        """Return the full 64-character runtime SHA-256."""
        return self._runtime_fingerprint

    @property
    def execution_identity(self) -> dict[str, object]:
        """Return result-row fields for this exact resident runtime."""
        return {
            "runtime_signature": self.runtime_signature,
            "runtime_fingerprint": self.runtime_fingerprint,
        }

    def _decode(
        self,
        prompts: Sequence[Any],
        requests: Sequence[
            MinistralDecodeRequest | MinistralReceiverRequest | MinistralWarmupRequest
        ],
    ) -> list[MinistralDecodeResult]:
        """Admit one independent request roster and drain it to raw completions."""
        handle = self._handle
        if handle is None or self._llm is None:
            raise RuntimeError("Ministral engine has been closed")
        roster = tuple(requests)
        prompt_roster = tuple(prompts)
        if not roster or len(prompt_roster) != len(roster):
            raise ValueError("Ministral prompt and request rosters must be equally nonempty")
        request_ids = tuple(request.request_id for request in roster)
        if any(not request_id for request_id in request_ids) or len(set(request_ids)) != len(
            request_ids
        ):
            raise ValueError("Ministral request ids must be nonempty and unique")
        if handle.has_unfinished_requests():
            raise RuntimeError("Ministral engine has unrelated unfinished requests")
        results: dict[str, MinistralDecodeResult] = {}
        live: set[str] = set()
        try:
            self._admit(handle, prompt_roster, roster, live)
            self._drain(handle, live, results)
            if live or set(results) != set(request_ids):
                raise RuntimeError(
                    f"Ministral decode ended with missing logical ids {sorted(live)!r}"
                )
        except Exception:
            handle.abort_request(sorted(live))
            raise
        return [results[request_id] for request_id in request_ids]

    def _admit(
        self,
        handle: VllmEngineHandle,
        prompts: Sequence[Any],
        requests: Sequence[
            MinistralDecodeRequest | MinistralReceiverRequest | MinistralWarmupRequest
        ],
        live: set[str],
    ) -> None:
        """Admit one validated logical roster through vLLM's public queue."""
        for prompt, request in zip(prompts, requests, strict=True):
            seed = request.seed
            if isinstance(seed, bool) or seed < 0:
                raise ValueError(f"Ministral request {request.request_id!r} has invalid seed")
            params = self._SamplingParams(**_sampling(request), seed=seed)
            handle.add_request(request.request_id, prompt, params)
            live.add(request.request_id)

    @staticmethod
    def _drain(
        handle: VllmEngineHandle,
        live: set[str],
        results: dict[str, MinistralDecodeResult],
    ) -> None:
        """Drain cumulative output while retaining only each finished request."""
        while handle.has_unfinished_requests():
            for output in cast(Sequence[Any], handle.step()):
                request_id = str(getattr(output, "request_id", ""))
                if request_id not in live:
                    raise RuntimeError(f"vLLM returned unknown logical id {request_id!r}")
                if not bool(getattr(output, "finished", False)):
                    continue
                results[request_id] = _completion_result(output, request_id)
                live.remove(request_id)

    def decode_token_ids_full(
        self,
        prompts: Sequence[Sequence[int]],
        requests: Sequence[
            MinistralDecodeRequest | MinistralReceiverRequest | MinistralWarmupRequest
        ],
    ) -> list[MinistralDecodeResult]:
        """Decode direct mistral-common token ids with logical request identity."""
        prepared: list[Any] = []
        for index, prompt in enumerate(prompts):
            tokens = list(prompt)
            if not tokens or any(isinstance(token, bool) or token < 0 for token in tokens):
                raise ValueError(f"Ministral token prompt {index} is empty or malformed")
            prepared.append(self._TokensPrompt(prompt_token_ids=tokens))
        return self._decode(prepared, requests)

    def decode_embeds_full(
        self,
        embeds: Sequence[torch.Tensor],
        requests: Sequence[MinistralReceiverRequest],
    ) -> list[MinistralDecodeResult]:
        """Decode receiver embedding rows while retaining raw token evidence."""
        prepared = [self.embedding_prompt(value, index) for index, value in enumerate(embeds)]
        return self._decode(prepared, requests)

    def embedding_prompt(self, rows: torch.Tensor, index: int = 0) -> dict[str, torch.Tensor]:
        """Validate one embedding prompt and render it for the engine.

        Both admission routes pass through here, so the geometry and finiteness
        checks apply to the streaming receiver too.
        """
        width = int(self.embedding_weight.shape[1])
        if rows.ndim != 2 or int(rows.shape[0]) < 1 or int(rows.shape[1]) != width:
            raise ValueError(f"Ministral embedding prompt {index} has invalid geometry")
        if not bool(torch.isfinite(rows).all()):
            raise ValueError(f"Ministral embedding prompt {index} contains non-finite rows")
        return {"prompt_embeds": rows.to(dtype=torch.bfloat16, device="cpu")}

    @property
    def stream_handle(self) -> VllmEngineHandle:
        """Return the public add-request/step seam for streaming admission.

        The blocking `_decode` above owns the whole engine for one roster, so a
        fleet receiver serving several cells drives this handle instead.
        """
        handle = self._handle
        if handle is None:
            raise RuntimeError("Ministral engine has been closed")
        return handle

    def stream_sampling(
        self,
        request: MinistralDecodeRequest | MinistralReceiverRequest | MinistralWarmupRequest,
    ) -> Any:
        """Build vLLM sampling for one signed request, through the sole validator."""
        seed = request.seed
        if isinstance(seed, bool) or seed < 0:
            raise ValueError(f"Ministral request {request.request_id!r} has invalid seed")
        return self._SamplingParams(**_sampling(request), seed=seed)

    def kv_pool_tokens(self) -> int | None:
        """Return the live KV-pool token capacity when readable."""
        return _pool_tokens(self.raw_llm)

    def close(self) -> None:
        """Release the aliased view before dropping the sole vLLM owner."""
        bridge, self._bridge = self._bridge, None
        if bridge is not None:
            bridge.release_view()
        self._handle = None
        self._llm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> MinistralEngine:
        """Return the live resident engine."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the engine when leaving its resident lifetime."""
        self.close()


def build_engine(config: MinistralEngineConfig | None = None) -> MinistralEngine:
    """Open one exact registered Ministral engine without model fallbacks."""
    return MinistralEngine(config)


def build_split_receiver_engine(config: MinistralEngineConfig | None = None) -> MinistralEngine:
    """Open the registered 14B receiver that injects rows and never captures.

    The HF-shaped weight view still opens, because the receiver reads the
    resident input-embedding table; what it does not own is the KV connector.
    """
    return MinistralEngine(config, split_receiver=True)


def build_text_engine(
    semantic_arm: str,
    config: MinistralEngineConfig | None = None,
) -> MinistralEngine:
    """Open one registered 14B, 8B, or 3B token-only sender engine."""
    return MinistralEngine(config, text_sender_arm=semantic_arm)


__all__ = (
    "MinistralDecodeResult",
    "MinistralEngine",
    "MinistralEngineConfig",
    "build_engine",
    "build_split_receiver_engine",
    "build_text_engine",
    "runtime_fingerprint",
)
