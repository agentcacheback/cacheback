"""The registered closer-injection decode protocol for the Gemma text lane.

A draw ending inside an unclosed reasoning block is continued exactly once with
the closing delimiter injected, on the draw's own clock and closing budget.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.gemma.text_codec import (
    GEMMA_CHANNEL_CLOSE,
    GEMMA_CHANNEL_OPEN,
    GEMMA_STOP_TOKEN_IDS,
    single_token_id,
)
from rcc.models.gemma.text_contract import (
    Sampling,
    TextCompletion,
    TextEngine,
    TextSenderSpec,
)
from rcc.run.rescore import content_token_count

# How far a draw may decode after its closer is injected. Like the seed
# ladder, it sits outside `text_registration` and does not move the text
# registration fingerprint.
GEMMA_CLOSING_TOKEN_BUDGET = 4096
_FINISH_REASONS = frozenset({"stop", "length"})


@dataclass
class ContinuedCompletion:
    """One draw whose unclosed reasoning block was closed and decoded on.

    The banked ids are head content, the closer, then the continuation, with
    the head's stop id dropped; the banked finish reason is the head's.
    """

    text: str
    n_tokens: int
    token_ids: Sequence[int]
    finish_reason: str
    num_cached_tokens: int | None
    queued_ts: float | None
    scheduled_ts: float | None
    first_token_ts: float | None


def block_is_unclosed(token_ids: Sequence[int], *, channel_open: int, channel_close: int) -> bool:
    """Return whether a draw ended inside a block it opened but never closed.

    A draw that never opens a channel is not continued: the protocol closes an
    open block and never manufactures one.
    """
    ids = tuple(int(token) for token in token_ids)
    return channel_open in ids and channel_close not in ids


def closer_is_affordable(
    token_ids: Sequence[int],
    *,
    prompt_tokens: int,
    closer_tokens: int,
    sender: TextSenderSpec,
    profile: BenchmarkProfile,
) -> bool:
    """Return whether one draw can be closed inside its registered budgets.

    Prompt plus head content, closer, and closing budget must fit the tighter
    of `max_model_len` and the sender's window. The ceiling bounds the head.
    """
    content = content_token_count(token_ids, GEMMA_STOP_TOKEN_IDS)
    tail = closer_tokens + GEMMA_CLOSING_TOKEN_BUDGET
    context = min(sender.native_max_model_len, profile.max_model_len)
    return prompt_tokens + content + tail <= context


def merge_continuation(
    tokenizer: Any,
    head: TextCompletion,
    tail: TextCompletion,
    *,
    closer_ids: Sequence[int],
) -> ContinuedCompletion:
    """Bank one head and its forced continuation as a single draw.

    The banked text is detokenized from the merged ids rather than string
    concatenated, which is what the rescore recomputes.
    """
    content = content_token_count(head.token_ids, GEMMA_STOP_TOKEN_IDS)
    ids = (
        *(int(token) for token in head.token_ids[:content]),
        *(int(token) for token in closer_ids),
        *(int(token) for token in tail.token_ids),
    )
    return ContinuedCompletion(
        text=str(tokenizer.decode(ids, skip_special_tokens=True)),
        n_tokens=len(ids),
        token_ids=ids,
        finish_reason=str(head.finish_reason),
        num_cached_tokens=head.num_cached_tokens,
        queued_ts=head.queued_ts,
        scheduled_ts=head.scheduled_ts,
        first_token_ts=head.first_token_ts,
    )


def continue_unclosed(
    engine: TextEngine,
    tokenizer: Any,
    prompts: Sequence[Sequence[int]],
    sampling: Sampling,
    results: Sequence[TextCompletion],
    *,
    seeds: Sequence[int],
    qid: str,
    sender: TextSenderSpec,
    profile: BenchmarkProfile,
) -> tuple[tuple[TextCompletion, ...], tuple[bool, ...]]:
    """Continue every unclosed draw exactly once with the closer injected.

    The continuation is the same draw: seed, seed tag, and every sampling field
    but the output cap are untouched, and the cap is the closing budget.
    """
    channel_open = single_token_id(tokenizer, GEMMA_CHANNEL_OPEN)
    channel_close = single_token_id(tokenizer, GEMMA_CHANNEL_CLOSE)
    closer_ids = (channel_close,)
    unclosed = tuple(
        position
        for position, result in enumerate(results)
        if block_is_unclosed(
            result.token_ids, channel_open=channel_open, channel_close=channel_close
        )
        and closer_is_affordable(
            result.token_ids,
            prompt_tokens=len(prompts[position]),
            closer_tokens=len(closer_ids),
            sender=sender,
            profile=profile,
        )
    )
    injected = tuple(position in set(unclosed) for position in range(len(results)))
    if not unclosed:
        return tuple(results), injected
    continued = tuple(
        engine.decode_token_ids_full(
            [
                (
                    *prompts[position],
                    *(
                        int(token)
                        for token in results[position].token_ids[
                            : content_token_count(results[position].token_ids, GEMMA_STOP_TOKEN_IDS)
                        ]
                    ),
                    *closer_ids,
                )
                for position in unclosed
            ],
            replace(sampling, max_tokens=GEMMA_CLOSING_TOKEN_BUDGET),
            seeds=[seeds[position] for position in unclosed],
        )
    )
    _validate_continuation(qid, continued, expected=len(unclosed))
    merged = list(results)
    for position, tail in zip(unclosed, continued, strict=True):
        merged[position] = merge_continuation(
            tokenizer, results[position], tail, closer_ids=closer_ids
        )
    return tuple(merged), injected


def _validate_continuation(qid: str, continued: Sequence[TextCompletion], *, expected: int) -> None:
    """Refuse a continuation whose evidence is unusable as banked identity."""
    if len(continued) != expected:
        raise RuntimeError(f"{qid}: Gemma closer continuation roster is incomplete")
    if any(result.finish_reason not in _FINISH_REASONS for result in continued):
        raise RuntimeError(f"{qid}: Gemma closer continuation returned an invalid finish reason")
    if any(result.num_cached_tokens not in (None, 0) for result in continued):
        raise RuntimeError(f"{qid}: prefix caching contaminated a Gemma closer continuation")
    if any(len(result.token_ids) != int(result.n_tokens) for result in continued):
        raise RuntimeError(f"{qid}: Gemma closer continuation token counts differ from its ids")


__all__ = (
    "GEMMA_CLOSING_TOKEN_BUDGET",
    "ContinuedCompletion",
    "block_is_unclosed",
    "closer_is_affordable",
    "continue_unclosed",
    "merge_continuation",
)
