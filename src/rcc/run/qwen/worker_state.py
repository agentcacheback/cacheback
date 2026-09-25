"""The route receiver's per-item state, and the readers around it."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from rcc.benchmarks.longbench_v2.data import ChainItem
from rcc.models.qwen.receiver import QwenReceiverPrompt, QwenReceiverRequest
from rcc.models.qwen.text import QwenReportBundle
from rcc.run.contract import PreparedItem
from rcc.run.fleet.stream import StreamCompletion


def report_dict(bundle: QwenReportBundle) -> dict[str, object]:
    """Return the banked fields of one sender report bundle."""
    return bundle.result_fields()


def chain_item(item: PreparedItem) -> ChainItem:
    """Return one chain item, refusing anything else by the type that arrived."""
    if not isinstance(item, ChainItem):
        raise RuntimeError(f"the chain receiver needs a chain item, not {type(item).__name__}")
    return item


@dataclass
class Pending:
    """One submitted item's prompt, requests, and clocks until it completes."""

    item: PreparedItem
    prompt: QwenReceiverPrompt
    requests: tuple[QwenReceiverRequest, ...]
    stream_prompt: dict[str, object]
    worker_prompt_tokens: tuple[int, ...]
    channel_fields: dict[str, object]
    receiver_prepare_s: float
    spill_load_s: float
    phase: int = 0
    admitted_at: float | None = None
    admission_fields: dict[str, int] = field(default_factory=lambda: {})
    completions: list[StreamCompletion] = field(default_factory=lambda: [])
    #: When this item's own preparation began, on the process clock.
    submitted_mono: float = 0.0


@dataclass
class VisibleCompletion:
    """One finished sample as the receiver reads it, with its injection flag."""

    request_id: str
    text: str
    token_ids: Sequence[int]
    n_tokens: int
    finish_reason: str
    num_cached_tokens: int | None
    #: Set when the closing continuation is issued. The activation rate is
    #: formed over this flag, and a row that loses it is refused by the rescore
    #: for exceeding the ceiling.
    answer_injected: bool
