"""The query-support capture: score a cache by what the question attends to."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rcc.transforms.select.core.types import (
        AttentionLayerObserver,
        BatchedPastLike,
        CaptureResult,
        CaptureSpec,
        CaptureStatistics,
    )
    from rcc.transforms.select.query_support.methods.batch import (
        capture_query_support_batched_from_past,
    )
    from rcc.transforms.select.query_support.methods.single import capture_query_support

__all__ = [
    "AttentionLayerObserver",
    "BatchedPastLike",
    "CaptureResult",
    "CaptureSpec",
    "CaptureStatistics",
    "capture_query_support",
    "capture_query_support_batched_from_past",
]


def __getattr__(name: str) -> Any:
    """Resolve the capture entry points without importing them at package import."""
    if name == "capture_query_support":
        from rcc.transforms.select.query_support.methods.single import capture_query_support

        return capture_query_support
    if name == "capture_query_support_batched_from_past":
        from rcc.transforms.select.query_support.methods.batch import (
            capture_query_support_batched_from_past,
        )

        return capture_query_support_batched_from_past
    if name in {
        "AttentionLayerObserver",
        "BatchedPastLike",
        "CaptureResult",
        "CaptureSpec",
        "CaptureStatistics",
    }:
        from rcc.transforms.select.core import types

        return getattr(types, name)
    raise AttributeError(name)
