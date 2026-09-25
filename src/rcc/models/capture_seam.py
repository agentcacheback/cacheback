"""Shared vLLM 0.26 capture seam for the single-engine receiver families.

Gemma 4 and Ministral 3 share the KV-transfer protocol: mark a request before
admission, accumulate its block table across chunked prefill, gather it once.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch


@dataclass(frozen=True)
class ExtractSpec:
    """One prompt-complete request and the block table holding its pages."""

    request_id: str
    num_prompt_tokens: int
    block_ids: tuple[tuple[int, ...], ...]


@dataclass
class _Capture:
    num_prompt_tokens: int
    blocks: list[list[int]]
    seen: int = 0


class _Planner:
    """Accumulate every cache group's block table across chunked prefill."""

    def __init__(self, group_count: int) -> None:
        self._group_count = group_count
        self._captures: dict[str, _Capture] = {}

    def tracks(self, request_id: str) -> bool:
        """Report whether this request is still being accumulated."""
        return request_id in self._captures

    def _groups(self, raw: Sequence[Iterable[int]]) -> list[list[int]]:
        groups = [[int(block) for block in group] for group in raw]
        if len(groups) != self._group_count:
            raise RuntimeError(
                f"vLLM supplied {len(groups)} block groups, expected {self._group_count}"
            )
        return groups

    def start(
        self, request_id: str, num_prompt_tokens: int, block_ids: Sequence[Iterable[int]]
    ) -> None:
        """Begin accumulating one admitted request's prompt block table."""
        if num_prompt_tokens < 1:
            raise ValueError(f"prompt must be non-empty, got {num_prompt_tokens}")
        self._captures[request_id] = _Capture(
            num_prompt_tokens=num_prompt_tokens,
            blocks=self._groups(block_ids),
        )

    def extend(self, request_id: str, block_ids: Sequence[Iterable[int]]) -> None:
        """Append a chunk's newly allocated blocks to every cache group."""
        capture = self._captures.get(request_id)
        if capture is None:
            return
        for current, appended in zip(capture.blocks, self._groups(block_ids), strict=True):
            current.extend(appended)

    def reset(self, request_id: str, block_ids: Sequence[Iterable[int]]) -> None:
        """Replace a resumed request's block table and rewind its progress."""
        capture = self._captures.get(request_id)
        if capture is None:
            return
        capture.blocks = self._groups(block_ids)
        capture.seen = 0

    def discard(self, request_id: str) -> None:
        """Forget one request, whether it completed or failed."""
        self._captures.pop(request_id, None)

    def advance(self, scheduled_tokens: Mapping[str, int]) -> list[ExtractSpec]:
        """Credit this step's scheduled tokens and return completed prompts."""
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
                        block_ids=tuple(tuple(group) for group in capture.blocks),
                    )
                )
                self.discard(request_id)
        return completed


class CaptureRegistry:
    """Request-scoped marker and artifact state for one family's connector."""

    def __init__(self, family: str) -> None:
        """Hold the marker and artifact tables for one named model family."""
        self.family = family
        self._requested: set[str] = set()
        self._extracted: dict[str, Any] = {}

    def mark(self, request_id: str) -> None:
        """Mark a fresh engine request for extraction before admission."""
        if request_id in self._requested or request_id in self._extracted:
            raise RuntimeError(
                f"{self.family} extraction request id {request_id!r} is already live"
            )
        self._requested.add(request_id)

    def is_marked(self, request_id: str) -> bool:
        """Report whether this request is still waiting to be admitted."""
        return request_id in self._requested

    def unmark(self, request_id: str) -> None:
        """Drop only the admission marker, leaving any artifact in place."""
        self._requested.discard(request_id)

    def store(self, request_id: str, build: Callable[[], Any]) -> None:
        """Materialize one artifact, refusing a second for the same request."""
        if request_id in self._extracted:
            raise RuntimeError(f"duplicate {self.family} artifact for {request_id!r}")
        self._extracted[request_id] = build()

    def take(self, request_id: str) -> Any:
        """Consume one completed artifact, failing if none was produced."""
        try:
            return self._extracted.pop(request_id)
        except KeyError as error:
            raise RuntimeError(
                f"{self.family} extraction {request_id!r} produced no artifact"
            ) from error

    def release(self, request_id: str) -> None:
        """Drop request-scoped marker and artifact state after success or failure."""
        self._requested.discard(request_id)
        self._extracted.pop(request_id, None)


def validate_capture_runtime(family: str, envs: Any, vllm_config: Any) -> None:
    """Validate the in-process TP=1 connector preconditions both families share."""
    if bool(getattr(envs, "VLLM_ENABLE_V1_MULTIPROCESSING", True)):
        raise RuntimeError(f"{family} capture requires VLLM_ENABLE_V1_MULTIPROCESSING=0")
    if bool(vllm_config.cache_config.enable_prefix_caching):
        raise RuntimeError(f"{family} capture requires enable_prefix_caching=False")
    tp = int(vllm_config.parallel_config.tensor_parallel_size)
    if tp != 1:
        raise RuntimeError(f"{family} capture requires tensor_parallel_size=1, got {tp}")


def _start_scheduled_requests(
    planner: _Planner, registry: CaptureRegistry, scheduler_output: Any
) -> None:
    """Admit marked new requests into the extraction planner."""
    for new_request in getattr(scheduler_output, "scheduled_new_reqs", ()):
        request_id = str(new_request.req_id)
        if not registry.is_marked(request_id):
            continue
        registry.unmark(request_id)
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


def _scheduled_specs(
    planner: _Planner, registry: CaptureRegistry, scheduler_output: Any
) -> list[ExtractSpec]:
    """Return prompt-complete extraction specs for this scheduler step."""
    _start_scheduled_requests(planner, registry, scheduler_output)
    _update_cached_requests(planner, scheduler_output)
    progress = {
        str(request_id): int(count)
        for request_id, count in scheduler_output.num_scheduled_tokens.items()
    }
    return planner.advance(progress)


def _save_extractions(
    registry: CaptureRegistry,
    metadata: Any,
    layers: tuple[Any, ...],
    extract: Callable[[Any, Any], Any],
) -> None:
    """Materialize every completed spec into request-scoped artifacts."""
    specs = list(getattr(metadata, "specs", ()) or ())
    if specs and not layers:
        raise RuntimeError(f"{registry.family} extraction ran before vLLM registered its pages")
    for spec in specs:
        registry.store(spec.request_id, partial(extract, layers, spec))


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


def build_capture_connector(
    *,
    name: str,
    summary: str,
    registry: CaptureRegistry,
    prepare: Callable[[Any, Any, Any], int],
    read_pages: Callable[[Mapping[str, torch.Tensor], Any], tuple[Any, ...]],
    extract: Callable[[Any, Any], Any],
) -> Any:
    """Build the external vLLM 0.26 connector only when its factory asks.

    ``prepare`` returns the number of KV cache groups the planner tracks,
    ``read_pages`` maps vLLM tensors to layer records, ``extract`` gathers one.
    """
    base_module = importlib.import_module("vllm.distributed.kv_transfer.kv_connector.v1.base")
    envs = importlib.import_module("vllm.envs")
    base: Any = base_module.KVConnectorBase_V1
    supports_hma: Any = base_module.SupportsHMA
    metadata_base: Any = base_module.KVConnectorMetadata
    base_init: Callable[..., None] = base.__init__
    metadata_init: Callable[..., None] = metadata_base.__init__

    class Metadata(metadata_base):
        def __init__(self, specs: list[ExtractSpec]) -> None:
            metadata_init(self)
            self.specs = specs

    class CaptureConnector(base, supports_hma):
        def __init__(self, vllm_config: Any, role: Any, kv_cache_config: Any) -> None:
            base_init(self, vllm_config, role, kv_cache_config)
            self._config = kv_cache_config
            self._planner = _Planner(prepare(envs, vllm_config, kv_cache_config))
            self._layers: tuple[Any, ...] = ()

        get_num_new_matched_tokens = _get_num_new_matched_tokens
        update_state_after_alloc = _update_state_after_alloc

        def build_connector_meta(self, scheduler_output: Any) -> Any:
            return Metadata(_scheduled_specs(self._planner, registry, scheduler_output))

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
            registry.unmark(request_id)

        def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
            self._layers = read_pages(kv_caches, self._config)

        start_load_kv = _start_load_kv
        wait_for_layer_load = _wait_for_layer_load
        save_kv_layer = _save_kv_layer

        def wait_for_save(self) -> None:
            _save_extractions(registry, self._get_connector_metadata(), self._layers, extract)

    CaptureConnector.__name__ = name
    CaptureConnector.__qualname__ = name
    CaptureConnector.__doc__ = summary
    return CaptureConnector


__all__ = (
    "CaptureRegistry",
    "ExtractSpec",
    "build_capture_connector",
    "validate_capture_runtime",
)
