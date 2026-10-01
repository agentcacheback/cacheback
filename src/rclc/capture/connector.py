"""Read-only Qwen3 capture for vLLM 0.11.1 and opt-in experimental 0.26.0."""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import torch

from rclc.capture.pages import DenseKV, ExtractSpec, LayerPages, extract_dense_kv, read_layer_pages


@dataclass
class _Capture:
    num_prompt_tokens: int
    blocks: list[int]
    seen: int = 0


class _Planner:
    """Accumulate the one cache group's block table across chunked prefill."""

    def __init__(self) -> None:
        self._captures: dict[str, _Capture] = {}

    def tracks(self, request_id: str) -> bool:
        return request_id in self._captures

    def _blocks(self, groups: Sequence[Iterable[int]]) -> list[int]:
        if len(groups) != 1:
            raise RuntimeError(f"vLLM supplied {len(groups)} block groups, expected 1")
        return [int(block) for block in groups[0]]

    def start(
        self,
        request_id: str,
        num_prompt_tokens: int,
        block_ids: Sequence[Iterable[int]],
    ) -> None:
        if num_prompt_tokens < 1:
            raise ValueError(f"prompt must be non-empty, got {num_prompt_tokens}")
        self._captures[request_id] = _Capture(
            num_prompt_tokens=num_prompt_tokens,
            blocks=self._blocks(block_ids),
        )

    def extend(self, request_id: str, block_ids: Sequence[Iterable[int]]) -> None:
        capture = self._captures.get(request_id)
        if capture is None:
            return
        capture.blocks.extend(self._blocks(block_ids))

    def reset(self, request_id: str, block_ids: Sequence[Iterable[int]]) -> None:
        capture = self._captures.get(request_id)
        if capture is None:
            return
        capture.blocks = self._blocks(block_ids)
        capture.seen = 0

    def discard(self, request_id: str) -> None:
        self._captures.pop(request_id, None)

    def advance(self, scheduled_tokens: Mapping[str, int]) -> list[ExtractSpec]:
        completed: list[ExtractSpec] = []
        for request_id, count in scheduled_tokens.items():
            capture = self._captures.get(request_id)
            if capture is None:
                continue
            capture.seen += int(count)
            if capture.seen > capture.num_prompt_tokens:
                raise RuntimeError(
                    f"prompt token overshoot for {request_id!r}: {capture.seen} "
                    f"against {capture.num_prompt_tokens}"
                )
            if capture.seen == capture.num_prompt_tokens:
                completed.append(
                    ExtractSpec(
                        request_id=request_id,
                        num_prompt_tokens=capture.num_prompt_tokens,
                        block_ids=tuple(capture.blocks),
                    )
                )
                self.discard(request_id)
        return completed


_REQUESTED: set[str] = set()
_EXTRACTED: dict[str, DenseKV] = {}


def request_capture(request_id: str) -> None:
    """Mark the supplied ID on 0.11.1 or returned internal ID on 0.26.0 before stepping."""
    if request_id in _REQUESTED or request_id in _EXTRACTED:
        raise RuntimeError(f"Qwen3 extraction request id {request_id!r} is already live")
    _REQUESTED.add(request_id)


def get_capture(request_id: str) -> DenseKV:
    """Read a completed artifact; the adapter releases it after successful validation."""
    try:
        return _EXTRACTED[request_id]
    except KeyError as error:
        raise RuntimeError(f"Qwen3 extraction {request_id!r} produced no artifact") from error


def discard_capture(request_id: str) -> None:
    """Drop request-scoped marker and artifact state."""
    _REQUESTED.discard(request_id)
    _EXTRACTED.pop(request_id, None)


_lazy_connector: Any = None


def require_runtime(allow_unstable: bool = False) -> str:
    """Keep the paper's Qwen pin as default and require opt-in for the newer adapter."""
    installed = version("vllm")
    if installed == "0.11.1":
        return installed
    if installed != "0.26.0" or not allow_unstable:
        raise RuntimeError("use vllm==0.11.1, or opt into unstable 0.26.0 with allow_unstable=True")
    warnings.warn(
        "vLLM 0.26.0 is opt-in via allow_unstable; 0.11.1 remains the default",
        UserWarning,
        stacklevel=2,
    )
    return installed


def __getattr__(name: str) -> Any:
    if name != "RCCCaptureConnector":
        raise AttributeError(name)
    global _lazy_connector
    if _lazy_connector is None:
        _lazy_connector = _build_connector()
    return _lazy_connector


def validate_capture_config(config: Any) -> None:
    """Require the engine settings shared by binding and capture."""
    parallel = config.parallel_config
    if (parallel.tensor_parallel_size, getattr(parallel, "pipeline_parallel_size", 1)) != (1, 1):
        raise ValueError("vLLM capture requires tensor and pipeline parallel sizes of one")
    if getattr(getattr(config, "model_config", None), "quantization", None) is not None:
        raise ValueError("vLLM capture requires unquantized weights")
    if config.cache_config.enable_prefix_caching:
        raise ValueError("vLLM capture requires enable_prefix_caching=False")
    if "fp8" in str(config.cache_config.cache_dtype):
        raise ValueError(
            "vLLM capture needs unquantized KV pages; use cache_dtype='auto', got "
            f"{config.cache_config.cache_dtype!r}"
        )


def _prompt_length(request: Any) -> int:
    """Count tokens or continuous rows as in the paper's capture connector."""
    ids = getattr(request, "prompt_token_ids", None)
    rows = getattr(request, "prompt_embeds", None)
    if ids is not None and (getattr(request, "prompt_is_token_ids", True) or rows is None):
        return len(ids)
    return 0 if rows is None else int(rows.shape[0])


def _start_scheduled_requests(planner: _Planner, scheduler_output: Any) -> None:
    """Admit marked new requests into the extraction planner."""
    for new_request in getattr(scheduler_output, "scheduled_new_reqs", ()):
        request_id = str(new_request.req_id)
        if request_id not in _REQUESTED:
            continue
        _REQUESTED.discard(request_id)
        planner.start(
            request_id,
            _prompt_length(new_request),
            new_request.block_ids,
        )


def _update_cached_requests(planner: _Planner, scheduler_output: Any) -> None:
    """Apply chunk-extension or resume block tables to tracked requests."""
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    if cached is None:
        return
    resumed = {str(value) for value in (cached.resumed_req_ids or ())}
    entries = list(cached.new_block_ids or ())
    for position, raw_id in enumerate(cached.req_ids or ()):
        request_id = str(raw_id)
        if not planner.tracks(request_id):
            continue
        entry = entries[position] if position < len(entries) else None
        if entry is None:
            continue
        if request_id in resumed:
            planner.reset(request_id, entry)
        else:
            planner.extend(request_id, entry)


def _scheduled_specs(planner: _Planner, scheduler_output: Any) -> list[ExtractSpec]:
    """Return prompt-complete extraction specs for this scheduler step."""
    _start_scheduled_requests(planner, scheduler_output)
    _update_cached_requests(planner, scheduler_output)
    progress = {
        str(request_id): int(count)
        for request_id, count in scheduler_output.num_scheduled_tokens.items()
    }
    return planner.advance(progress)


def _save_extractions(metadata: Any, layers: tuple[LayerPages, ...]) -> None:
    """Materialize every completed spec into request-scoped device tensors."""
    specs = list(getattr(metadata, "specs", ()) or ())
    if specs and not layers:
        raise RuntimeError("Qwen3 extraction ran before vLLM registered its pages")
    for spec in specs:
        if spec.request_id in _EXTRACTED:
            raise RuntimeError(f"duplicate Qwen3 artifact for {spec.request_id!r}")
        _EXTRACTED[spec.request_id] = extract_dense_kv(layers, spec)


class _NoLoadHooks:
    """Connector hooks that a read-only extractor leaves inert."""

    def get_num_new_matched_tokens(self, request: Any, count: int) -> tuple[int, bool]:
        return 0, False

    def update_state_after_alloc(self, request: Any, blocks: Any, num_external_tokens: int) -> None:
        return None

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        return None

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(
        self, layer_name: str, kv_layer: Any, attn_metadata: Any, **kwargs: Any
    ) -> None:
        return None


def _build_connector() -> Any:
    """Build the external vLLM connector only when its factory requests it."""
    base_module = importlib.import_module("vllm.distributed.kv_transfer.kv_connector.v1.base")
    envs = importlib.import_module("vllm.envs")
    base = base_module.KVConnectorBase_V1
    supports_hma = base_module.SupportsHMA
    metadata_base = base_module.KVConnectorMetadata

    class Metadata(metadata_base):
        def __init__(self, specs: list[ExtractSpec]) -> None:
            super().__init__()  # pyright: ignore[reportUnknownMemberType]
            self.specs = specs

    class RCCCaptureConnector(_NoLoadHooks, base, supports_hma):
        """Read-only extractor for one in-process TP=1 Qwen3 engine."""

        def __init__(self, vllm_config: Any, role: Any, kv_cache_config: Any) -> None:
            extra = vllm_config.kv_transfer_config.kv_connector_extra_config
            require_runtime(bool(extra.get("allow_unstable", False)))
            base.__init__(self, vllm_config, role, kv_cache_config)
            self._config = kv_cache_config
            if bool(getattr(envs, "VLLM_ENABLE_V1_MULTIPROCESSING", True)):
                raise RuntimeError("Qwen3 capture requires VLLM_ENABLE_V1_MULTIPROCESSING=0")
            validate_capture_config(vllm_config)
            self._planner = _Planner()
            self._layers: tuple[LayerPages, ...] = ()

        def build_connector_meta(self, scheduler_output: Any) -> Any:
            return Metadata(_scheduled_specs(self._planner, scheduler_output))

        def request_finished(self, request: Any, block_ids: Any) -> tuple[bool, Any]:
            request_id = str(request.request_id)
            self._planner.discard(request_id)
            _REQUESTED.discard(request_id)
            return False, None

        def request_finished_all_groups(
            self, request: Any, block_ids: tuple[list[int], ...]
        ) -> tuple[bool, Any]:
            return self.request_finished(request, block_ids)

        def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
            self._layers = read_layer_pages(kv_caches, self._config)

        def wait_for_save(self) -> None:
            _save_extractions(self._get_connector_metadata(), self._layers)

    return RCCCaptureConnector


__all__ = (
    "discard_capture",
    "get_capture",
    "request_capture",
    "require_runtime",
    "validate_capture_config",
)
