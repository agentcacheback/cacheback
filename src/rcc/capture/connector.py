"""Read-only Qwen3 capture for vLLM 0.11.1 and opt-in experimental 0.26.0."""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import torch

from rcc.capture.pages import (
    DenseKV,
    LayerKV,
    LayerPages,
    extract_dense_kv,
    read_layer_pages,
)


@dataclass(frozen=True)
class _ExtractSpec:
    request_id: str
    num_prompt_tokens: int
    block_ids: tuple[tuple[int, ...], ...]


@dataclass
class _Capture:
    num_prompt_tokens: int
    blocks: list[list[int]]
    seen: int = 0


class _Planner:
    """Accumulate the one cache group's block table across chunked prefill."""

    def __init__(self, group_count: int) -> None:
        self._group_count = group_count
        self._captures: dict[str, _Capture] = {}

    def tracks(self, request_id: str) -> bool:
        return request_id in self._captures

    def _groups(self, raw: Sequence[Iterable[int]]) -> list[list[int]]:
        groups = [[int(block) for block in group] for group in raw]
        if len(groups) != self._group_count:
            raise RuntimeError(
                f"vLLM supplied {len(groups)} block groups, expected {self._group_count}"
            )
        return groups

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
            blocks=self._groups(block_ids),
        )

    def extend(self, request_id: str, block_ids: Sequence[Iterable[int]]) -> None:
        capture = self._captures.get(request_id)
        if capture is None:
            return
        for current, appended in zip(capture.blocks, self._groups(block_ids), strict=True):
            current.extend(appended)

    def reset(self, request_id: str, block_ids: Sequence[Iterable[int]]) -> None:
        capture = self._captures.get(request_id)
        if capture is None:
            return
        capture.blocks = self._groups(block_ids)
        capture.seen = 0

    def discard(self, request_id: str) -> None:
        self._captures.pop(request_id, None)

    def advance(self, scheduled_tokens: Mapping[str, int]) -> list[_ExtractSpec]:
        completed: list[_ExtractSpec] = []
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
                    _ExtractSpec(
                        request_id=request_id,
                        num_prompt_tokens=capture.num_prompt_tokens,
                        block_ids=tuple(tuple(group) for group in capture.blocks),
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
        "vLLM 0.26.0 support is unstable and lacks live GPU testing", UserWarning, stacklevel=2
    )
    return installed


def __getattr__(name: str) -> Any:
    if name != "RCCCaptureConnector":
        raise AttributeError(name)
    global _lazy_connector
    if _lazy_connector is None:
        _lazy_connector = _build_connector()
    return _lazy_connector


def _validate_connector_runtime(envs: Any, vllm_config: Any, kv_cache_config: Any) -> int:
    """Validate the in-process TP=1 connector preconditions."""
    if bool(getattr(envs, "VLLM_ENABLE_V1_MULTIPROCESSING", True)):
        raise RuntimeError("Qwen3 capture requires VLLM_ENABLE_V1_MULTIPROCESSING=0")
    if bool(vllm_config.cache_config.enable_prefix_caching):
        raise RuntimeError("Qwen3 capture requires enable_prefix_caching=False")
    tp = int(vllm_config.parallel_config.tensor_parallel_size)
    pp = int(getattr(vllm_config.parallel_config, "pipeline_parallel_size", 1))
    if (tp, pp) != (1, 1):
        raise RuntimeError("Qwen3 capture requires tensor_parallel_size=pipeline_parallel_size=1")
    model_config = getattr(vllm_config, "model_config", None)
    quantization = getattr(model_config, "quantization", None)
    if quantization is not None:
        raise RuntimeError("Qwen3 capture requires unquantized model weights")
    groups = list(getattr(kv_cache_config, "kv_cache_groups", ()) or ())
    if len(groups) != 1:
        raise RuntimeError(f"Qwen3 capture requires exactly one KV cache group, got {len(groups)}")
    return len(groups)


def _start_scheduled_requests(planner: _Planner, scheduler_output: Any) -> None:
    """Admit marked new requests into the extraction planner."""
    for new_request in getattr(scheduler_output, "scheduled_new_reqs", ()):
        request_id = str(new_request.req_id)
        if request_id not in _REQUESTED:
            continue
        _REQUESTED.discard(request_id)
        planner.start(
            request_id,
            len(new_request.prompt_token_ids or ()),
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


def _scheduled_specs(planner: _Planner, scheduler_output: Any) -> list[_ExtractSpec]:
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


def _get_num_new_matched_tokens(self: Any, request: Any, count: int) -> tuple[int, bool]:
    return 0, False


def _update_state_after_alloc(
    self: Any,
    request: Any,
    blocks: Any,
    num_external_tokens: int,
) -> None:
    return None


def _start_load_kv(self: Any, forward_context: Any, **kwargs: Any) -> None:
    return None


def _wait_for_layer_load(self: Any, layer_name: str) -> None:
    return None


def _save_kv_layer(
    self: Any,
    layer_name: str,
    kv_layer: Any,
    attn_metadata: Any,
    **kwargs: Any,
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
        def __init__(self, specs: list[_ExtractSpec]) -> None:
            super().__init__()  # pyright: ignore[reportUnknownMemberType]
            self.specs = specs

    class RCCCaptureConnector(base, supports_hma):
        """Read-only extractor for one in-process TP=1 Qwen3 engine."""

        def __init__(self, vllm_config: Any, role: Any, kv_cache_config: Any) -> None:
            extra = vllm_config.kv_transfer_config.kv_connector_extra_config
            require_runtime(bool(extra.get("allow_unstable", False)))
            super().__init__(  # pyright: ignore[reportUnknownMemberType]
                vllm_config, role, kv_cache_config
            )
            self._config = kv_cache_config
            self._planner = _Planner(
                _validate_connector_runtime(envs, vllm_config, kv_cache_config)
            )
            self._layers: tuple[LayerPages, ...] = ()

        get_num_new_matched_tokens = _get_num_new_matched_tokens
        update_state_after_alloc = _update_state_after_alloc

        def build_connector_meta(self, scheduler_output: Any) -> Any:
            return Metadata(_scheduled_specs(self._planner, scheduler_output))

        def request_finished(self, request: Any, block_ids: Any) -> tuple[bool, Any]:
            self._finish(str(request.request_id))
            return False, None

        def request_finished_all_groups(
            self, request: Any, block_ids: tuple[list[int], ...]
        ) -> tuple[bool, Any]:
            self._finish(str(request.request_id))
            return False, None

        def _finish(self, request_id: str) -> None:
            self._planner.discard(request_id)
            _REQUESTED.discard(request_id)

        def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
            self._layers = read_layer_pages(kv_caches, self._config)

        start_load_kv = _start_load_kv
        wait_for_layer_load = _wait_for_layer_load
        save_kv_layer = _save_kv_layer

        def wait_for_save(self) -> None:
            _save_extractions(self._get_connector_metadata(), self._layers)

    return RCCCaptureConnector


__all__ = (
    "DenseKV",
    "LayerKV",
    "discard_capture",
    "extract_dense_kv",
    "get_capture",
    "read_layer_pages",
    "request_capture",
)
