"""Asynchronous decode schedule shared by resident Gemma payload types."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch

from rcc.models.gemma.capture_runtime import CaptureArtifact
from rcc.models.gemma.contract import MAX_MODEL_LEN, STOP_IDS
from rcc.models.gemma.results import build_result_rows
from rcc.models.gemma.text_closer import block_is_unclosed
from rcc.run.fleet.answer_closer import AnswerCloser
from rcc.run.fleet.stream import StreamCompletion, StreamRequest

TEXT_ARMS = ("text_primary", "text_medium", "text_small")


def gemma_answer_closer(
    embed: Callable[[Sequence[int]], torch.Tensor],
    *,
    channel_open: int,
    channel_close: int,
    answer_ceiling: int,
) -> AnswerCloser:
    """Bind the registered answer-side closer to this receiver's channel pair.

    The trigger is `block_is_unclosed`, and the continuation prompt is gathered
    through the shared embedding table on the receiver's own device.
    """

    def continuation_prompt(
        prompt: Any,
        content: tuple[int, ...],
        closer_ids: tuple[int, ...],
    ) -> dict[str, Any]:
        rows = cast(torch.Tensor, cast(Mapping[str, Any], prompt)["prompt_embeds"])
        gathered = embed((*content, *closer_ids))
        return {"prompt_embeds": torch.cat((rows, gathered.to(rows)), dim=0)}

    def unclosed(token_ids: Sequence[int]) -> bool:
        return block_is_unclosed(token_ids, channel_open=channel_open, channel_close=channel_close)

    return AnswerCloser(
        closer_ids=(channel_close,),
        stop_ids=tuple(sorted(STOP_IDS)),
        answer_ceiling=answer_ceiling,
        max_model_len=MAX_MODEL_LEN,
        continuation_prompt=continuation_prompt,
        unclosed=unclosed,
    )


def build_cell_rows(receiver: Any, state: PendingArm) -> list[dict[str, Any]]:
    """Build one item-arm cell's per-seed rows for whichever route banks them."""
    completions = sorted(state.completions or (), key=lambda row: int(row.tag[1:]))
    if state.timing_origin is None:
        raise RuntimeError(f"{state.item['qid']}/{state.arm}: timing origin is missing")
    if state.arm in TEXT_ARMS:
        from rcc.models.gemma.text_results import build_text_result_rows

        candidates = build_text_result_rows(
            tokenizer=receiver.tokenizer,
            item=state.item,
            arm=state.arm,
            prepared=state.prepared,
            seeds=state.seeds,
            completions=completions,
            timing_origin=state.timing_origin,
            channel_open=receiver.channel_open,
            channel_close=receiver.channel_close,
            engine_identity_sha256=receiver.engine_identity_sha256,
        )
    else:
        candidates = build_result_rows(
            tokenizer=receiver.tokenizer,
            item=state.item,
            capture=state.capture,
            arm=state.arm,
            prepared=state.prepared,
            seeds=state.seeds,
            completions=completions,
            timing_origin=state.timing_origin,
            channel_open=receiver.channel_open,
            channel_close=receiver.channel_close,
            embedding_digests=receiver.embedding_digests,
            engine_identity_sha256=receiver.engine_identity_sha256,
            greedy_visible=state.greedy_visible,
            greedy_loose=state.greedy_loose,
            producer_s=state.producer_s,
        )
    if receiver.result_row_adapter is not None:
        candidates = [receiver.result_row_adapter(dict(row), state.arm) for row in candidates]
    return list(candidates)


@dataclass
class PendingArm:
    """One item-arm cell in flight: its prompt, its draws, and its completions."""

    item: dict[str, Any]
    capture: CaptureArtifact
    arm: str
    prepared: Any
    seeds: tuple[int, ...]
    greedy_visible: str | None
    greedy_loose: float | None
    #: The producer wall the split route read beside the handoff manifest;
    #: ``None`` on the resident route, which measured the same work itself.
    producer_s: float | None = None
    phase: int = 0
    timing_origin: float | None = None
    completions: list[StreamCompletion] | None = None


class GemmaReceiverScheduleMixin:
    """The s0 then s1/s2 decode schedule over resident Gemma requests."""

    @staticmethod
    def _job_id(qid: str, arm: str) -> str:
        return f"{qid}|{arm}"

    def _offer(self: Any, job_id: str, state: PendingArm, indices: tuple[int, ...]) -> None:
        if self.tracker is None:
            raise RuntimeError("Gemma vLLM stream tracker is not ready")
        requests = [
            StreamRequest(
                request_id=f"{job_id}:s{index}",
                prompt={"prompt_embeds": state.prepared.prompt},
                sampling=self._sample_params(state.seeds[index]),
                prompt_tokens=int(state.prepared.prompt.shape[0]),
                qid=job_id,
                tag=f"s{index}",
            )
            for index in indices
        ]
        self.tracker.offer(job_id, requests)


__all__ = (
    "TEXT_ARMS",
    "GemmaReceiverScheduleMixin",
    "PendingArm",
    "build_cell_rows",
    "gemma_answer_closer",
)
