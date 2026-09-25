"""Registered Qwen worker and receiver prompt records.

One prompt is rendered per private shard, and the receiver prompt is rendered
for whichever benchmark the profile registers.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.fanoutqa import prompts as fanoutqa_prompts
from rcc.benchmarks.fanoutqa.source_padding import ProbeItem
from rcc.benchmarks.longbench_v2 import prompts as longbench_prompts
from rcc.benchmarks.longbench_v2.data import ChainItem
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.qwen.text import format_report_payload
from rcc.models.route import RouteFamily
from rcc.run.qwen.worker_contract import QwenChannel
from rcc.topologies.fanout import FANOUT_M3

_REGISTERED_PROMPT_BUILDERS: dict[str, ModuleType] = {
    FANOUTQA_NATURAL_DEV50.prompt_builder: fanoutqa_prompts,
    longbench_prompts.PROMPT_BUILDER: longbench_prompts,
    longbench_prompts.PROMPT_BUILDER_CONDENSE: longbench_prompts,
}


def _require_registered_prompt_builder(profile: BenchmarkProfile) -> ModuleType:
    """Return the benchmark's prompt module or refuse its builder by name."""
    try:
        return _REGISTERED_PROMPT_BUILDERS[profile.prompt_builder]
    except KeyError:
        raise ValueError(
            f"Qwen prompts have no registered builder {profile.prompt_builder!r} "
            f"for {profile.benchmark_key}"
        ) from None


@dataclass(frozen=True)
class QwenManagerPromptRecord:
    """Manager prompt tokens and exact question rows used by capture/receiver.

    ``payload_slot`` is the token index where a family seats payload rows in
    the user turn; it and ``payload_headers`` are None for the prepend layout.
    """

    text: str
    token_ids: tuple[int, ...]
    question_token_ids: tuple[int, ...]
    sha256: str
    payload_slot: int | None = None
    payload_headers: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        """Require nonempty prompt rows, their exact text hash, and a sane slot."""
        if not self.text or not self.token_ids or not self.question_token_ids:
            raise ValueError("Qwen manager prompt record cannot be empty")
        if self.sha256 != hashlib.sha256(self.text.encode()).hexdigest():
            raise ValueError("Qwen manager prompt record hash differs")
        if (self.payload_slot is None) != (self.payload_headers is None):
            raise ValueError("Qwen manager payload slot and headers travel together")
        if self.payload_slot is not None:
            if not 0 < self.payload_slot < len(self.token_ids):
                raise ValueError("Qwen manager payload slot must fall inside the prompt")
            headers = self.payload_headers or ()
            if not all(headers) or len(headers) not in (1, FANOUT_M3.workers_per_item):
                raise ValueError("Qwen manager payload headers must cover the payload blocks")


def worker_prompts(
    item: ProbeItem,
    tokenizer: Any,
    *,
    family: RouteFamily,
    semantic_arm: str = "text_primary",
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> tuple[str, ...]:
    """Render the exact natural-report prompt for each private shard."""
    builder = _require_registered_prompt_builder(profile)
    if builder is not fanoutqa_prompts:
        raise ValueError(f"Qwen worker prompt roster does not implement {profile.prompt_builder!r}")
    if family.native_sender_prompts:
        from rcc.models.qwen.native_prompts import native_worker_prompts

        return native_worker_prompts(
            item, tokenizer, semantic_arm=semantic_arm, family=family, profile=profile
        )
    prompts = tuple(
        builder.worker_prompt(
            tokenizer,
            item.question,
            evidence,
            report_style="natural",
            enable_thinking=family.profile.decode.enable_thinking,
            system_prompt=family.thinking_system_prompt,
            assistant_prefill=family.assistant_prefill,
            low_effort=family.low_effort,
        )
        for evidence in item.shards
    )
    counts = tuple(
        len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) for prompt in prompts
    )
    if len(prompts) != 3 or not profile.admits_worker_prompt_tokens(counts):
        raise RuntimeError(
            f"{item.qid}: Qwen worker prompt geometry {counts} differs from the registered "
            f"{profile.prompt_geometry} width {profile.worker_prompt_tokens}"
        )
    return prompts


PAYLOAD_SLOT_MARKER = "RCCPAYLOADSLOT"  # rendered into the prompt; keep the spelling


def _encode_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    return tuple(int(token) for token in tokenizer(text, add_special_tokens=False)["input_ids"])


def _fanoutqa_manager_prompt(
    item: ProbeItem | ChainItem,
    tokenizer: Any,
    *,
    family: RouteFamily,
    reports: Sequence[str],
    payload_slot: bool,
) -> tuple[str, str, tuple[int, ...], int | None, tuple[tuple[int, ...], ...] | None]:
    """Render the FanOutQA manager prompt, with the worker-report slot open on request."""
    if not isinstance(item, ProbeItem):
        raise TypeError("FanOutQA prompt builder requires a prepared ProbeItem")
    question = item.question

    def render(report_payload: str) -> str:
        return fanoutqa_prompts.manager_prompt(
            tokenizer,
            question,
            report_payload,
            enable_thinking=family.profile.decode.enable_thinking,
            system_prompt=family.thinking_system_prompt,
            assistant_prefill=family.assistant_prefill,
            low_effort=family.low_effort,
        )

    if not payload_slot:
        text = render(format_report_payload(reports) if reports else "")
        return question, text, _encode_ids(tokenizer, text), None, None
    rendered = render(PAYLOAD_SLOT_MARKER)
    before, marker, after = rendered.partition(PAYLOAD_SLOT_MARKER)
    if not marker or PAYLOAD_SLOT_MARKER in after or not before or not after:
        raise ValueError("Qwen manager prompt payload slot did not render exactly once")
    ids_before = _encode_ids(tokenizer, before)
    headers = tuple(
        _encode_ids(
            tokenizer,
            f"[worker {worker} report]\n" if worker == 0 else f"\n\n[worker {worker} report]\n",
        )
        for worker in range(FANOUT_M3.workers_per_item)
    )
    return (
        question,
        before + after,
        ids_before + _encode_ids(tokenizer, after),
        len(ids_before),
        headers,
    )


def _longbench_manager_prompt(
    item: ProbeItem | ChainItem,
    tokenizer: Any,
    *,
    family: RouteFamily,
    channel: QwenChannel,
    reports: Sequence[str],
    payload_slot: bool,
) -> tuple[str, str, tuple[int, ...], int | None, tuple[tuple[int, ...], ...] | None]:
    """Render the LongBench answering body, or the no-context body for the floor arm."""
    if not isinstance(item, ChainItem):
        raise TypeError("LongBench prompt builder requires a prepared ChainItem")
    if payload_slot and (channel != "latent" or not family.payload_in_user_turn):
        raise ValueError("LongBench payload slot requires the native latent family")
    if len(reports) > 1:
        raise ValueError("LongBench receiver prompt accepts at most one rewritten note")
    # The official template strips the question, so the rows are located
    # against the stripped text the builder rendered, never the raw panel
    # string: a question with edge whitespace would otherwise abort the item.
    question = item.question.strip()
    if channel == "floor":
        if reports:
            raise ValueError(
                f"the LongBench floor channel reads no notes; {len(reports)} were handed to it"
            )
        text = longbench_prompts.no_context_prompt(
            tokenizer,
            question,
            item.choices,
            enable_thinking=family.profile.decode.enable_thinking,
            family=family,
        )
    else:
        text = longbench_prompts.answer_prompt(
            tokenizer,
            question,
            item.choices,
            notes=reports[0] if reports else "",
            enable_thinking=family.profile.decode.enable_thinking,
            latent=channel == "latent",
            family=family,
            payload_slot=payload_slot,
        )
    if not payload_slot:
        return question, text, _encode_ids(tokenizer, text), None, None
    marker = longbench_prompts.PAYLOAD_SLOT_MARKER
    if text.count(marker) != 1:
        raise RuntimeError("LongBench native prompt did not preserve one payload slot")
    before, after = text.split(marker)
    # The two sides are tokenized apart, the way the FanOutQA slot is: encoding
    # the joined text would merge the seam into one token, and the receiver
    # splices at the slot index of the ids it is handed.
    ids_before = _encode_ids(tokenizer, before)
    token_ids = ids_before + _encode_ids(tokenizer, after)
    headers = ((_encode_ids(tokenizer, "[terminal payload]\n")),)
    return question, before + after, token_ids, len(ids_before), headers


def manager_prompt_record(
    item: ProbeItem | ChainItem,
    tokenizer: Any,
    *,
    family: RouteFamily,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
    channel: QwenChannel = "text",
    reports: Sequence[str] = (),
    payload_slot: bool = False,
) -> QwenManagerPromptRecord:
    """Render one exact receiver prompt and locate its question token rows.

    With ``payload_slot`` the slot is left open and each side tokenized apart,
    so the receiver can splice rows there. ``channel`` picks a LongBench body.
    """
    if payload_slot and reports:
        raise ValueError("Qwen manager prompt cannot carry both text reports and a payload slot")
    builder = _require_registered_prompt_builder(profile)
    slot: int | None = None
    headers: tuple[tuple[int, ...], ...] | None = None
    if builder is fanoutqa_prompts:
        question, text, token_ids, slot, headers = _fanoutqa_manager_prompt(
            item, tokenizer, family=family, reports=reports, payload_slot=payload_slot
        )
    elif builder is longbench_prompts:
        question, text, token_ids, slot, headers = _longbench_manager_prompt(
            item,
            tokenizer,
            family=family,
            channel=channel,
            reports=reports,
            payload_slot=payload_slot,
        )
    else:
        raise ValueError(
            f"Qwen receiver prompt roster does not implement {profile.prompt_builder!r}"
        )
    return QwenManagerPromptRecord(
        text=text,
        token_ids=token_ids,
        question_token_ids=fanoutqa_prompts.question_token_ids(tokenizer, text, question),
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        payload_slot=slot,
        payload_headers=headers,
    )


__all__ = (
    "PAYLOAD_SLOT_MARKER",
    "QwenManagerPromptRecord",
    "manager_prompt_record",
    "worker_prompts",
)
