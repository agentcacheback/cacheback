"""Lazy vLLM 0.26 request planner for dense Ministral 3 extraction.

The planner and connector shell are shared in ``rcc.models.capture_seam``; what
stays here is the one Ministral precondition, exactly one dense KV cache group.
"""

from __future__ import annotations

from typing import Any

from rcc.models.capture_seam import (
    CaptureRegistry,
    build_capture_connector,
    validate_capture_runtime,
)
from rcc.models.ministral.capture_pages import (
    MinistralDenseKV,
    MinistralLayerKV,
    extract_dense_kv,
    read_layer_pages,
)

_REGISTRY = CaptureRegistry("Ministral")


def request_ministral_extraction(request_id: str) -> None:
    """Mark a fresh engine request for extraction before admission."""
    _REGISTRY.mark(request_id)


def take_ministral_extraction(request_id: str) -> MinistralDenseKV:
    """Consume one completed artifact, failing if none was produced."""
    artifact: MinistralDenseKV = _REGISTRY.take(request_id)
    return artifact


def discard_ministral_extraction(request_id: str) -> None:
    """Drop request-scoped marker and artifact state."""
    _REGISTRY.release(request_id)


def _prepare_connector(envs: Any, vllm_config: Any, kv_cache_config: Any) -> int:
    """Validate the TP=1 preconditions and the single dense cache group."""
    validate_capture_runtime("Ministral", envs, vllm_config)
    groups = list(getattr(kv_cache_config, "kv_cache_groups", ()) or ())
    if len(groups) != 1:
        raise RuntimeError(
            f"Ministral capture requires exactly one KV cache group, got {len(groups)}"
        )
    return len(groups)


_lazy_connector: Any = None


def __getattr__(name: str) -> Any:
    if name != "MinistralCaptureConnector":
        raise AttributeError(name)
    global _lazy_connector
    if _lazy_connector is None:
        _lazy_connector = build_capture_connector(
            name="MinistralCaptureConnector",
            summary="Read-only extractor for one in-process TP=1 Ministral engine.",
            registry=_REGISTRY,
            prepare=_prepare_connector,
            read_pages=read_layer_pages,
            extract=extract_dense_kv,
        )
    return _lazy_connector


__all__ = (
    "MinistralDenseKV",
    "MinistralLayerKV",
    "discard_ministral_extraction",
    "extract_dense_kv",
    "read_layer_pages",
    "request_ministral_extraction",
    "take_ministral_extraction",
)
