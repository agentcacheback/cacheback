"""Checks over one assembled Gemma receiver request.

Only the last RECEIVER_SLIDING_WINDOW rows are all-layer readable, so the question turn after
a cut payload must fit it; payload rows outside it are the item's length, banked as `window_binds`.
"""

from __future__ import annotations

from typing import Any

from rcc.models.gemma.contract import (
    RECEIVER_ENABLE_THINKING,
    RECEIVER_SLIDING_WINDOW,
)
from rcc.models.gemma.engine_contract import REGISTERED_SPEC
from rcc.models.gemma.mechanism import receiver_turn_ids


def require_registered_thinking_turn(tokenizer: Any) -> tuple[list[int], list[int]]:
    """Return the exact thinking-on turn the registered template spec admits."""
    off = receiver_turn_ids(tokenizer, enable_thinking=False)
    on = receiver_turn_ids(tokenizer, enable_thinking=RECEIVER_ENABLE_THINKING)
    expected_off = (
        list(REGISTERED_SPEC.template.off_prefix),
        list(REGISTERED_SPEC.template.off_suffix),
    )
    expected_on = (
        list(REGISTERED_SPEC.template.on_prefix),
        list(REGISTERED_SPEC.template.on_suffix),
    )
    if not RECEIVER_ENABLE_THINKING or off != expected_off or on != expected_on or on == off:
        raise RuntimeError("Gemma fleet chat turn differs from the registered thinking wrap")
    return on


def require_window_fit(*, tag: str, tier_rows: int, trailing_rows: int) -> None:
    """Refuse a request whose question turn does not fit the sliding window."""
    window_rows = tier_rows + trailing_rows
    if window_rows > RECEIVER_SLIDING_WINDOW:
        raise RuntimeError(
            f"{tag}: the question turn needs {window_rows} rows, exceeding the "
            f"{RECEIVER_SLIDING_WINDOW}-row sliding window"
        )


__all__ = (
    "require_registered_thinking_turn",
    "require_window_fit",
)
