"""Answer-side closer injection, shared by the receiver lanes.

A draw that ends inside an unclosed reasoning block is truncated at its stop
token and continued once under a closing budget, with the same sampling and seed.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

#: How far an answer may decode after its closer is injected, in tokens.
ANSWER_CLOSING_TOKEN_BUDGET = 4096

#: The longest closer a lane may inject, in token ids: one reserved delimiter
#: id, or the ids of a tag spelled out as ordinary text. Three is the longest,
#: the ids of ``</think>`` spelled out under the Nemotron tokenizer.
ANSWER_CLOSER_MAX_IDS = 3

FINISH_REASONS = frozenset({"stop", "length"})


def capped_sampling(sampling: Any, max_tokens: int) -> Any:
    """Return the head's own sampling with only its output cap replaced.

    Temperature, top-p, top-k, penalties and the seed carry over unchanged, so
    the continuation is the same draw. The backend's own ``clone`` is preferred.
    """
    cloner = getattr(sampling, "clone", None)
    continued: Any = cloner() if callable(cloner) else copy.copy(sampling)
    try:
        continued.max_tokens = int(max_tokens)
    except (AttributeError, TypeError) as error:
        raise RuntimeError("the answer closer cannot cap this lane's sampling object") from error
    return continued


@dataclass(frozen=True)
class AnswerCloser:
    """One lane's answer-side closer injection policy.

    ``unclosed`` is the lane's own trigger and the complement of what its reader
    accepts: a reserved-delimiter lane tests token ids, a text lane the decode.
    """

    closer_ids: tuple[int, ...]
    stop_ids: tuple[int, ...]
    answer_ceiling: int
    max_model_len: int
    #: (prompt, head content ids, closer ids) -> the continuation's prompt, in
    #: whatever shape the lane's engine takes. The lane builds it because only
    #: it can gather its own embedding table on its own receiver device.
    continuation_prompt: Callable[[Any, tuple[int, ...], tuple[int, ...]], Any]
    unclosed: Callable[[Sequence[int]], bool]
    closing_budget: int = ANSWER_CLOSING_TOKEN_BUDGET
    continuation_sampling: Callable[[Any, int], Any] = capped_sampling

    def __post_init__(self) -> None:
        """Refuse a policy whose delimiters or budgets cannot bank an answer."""
        if not self.closer_ids or len(self.closer_ids) > ANSWER_CLOSER_MAX_IDS:
            raise ValueError(
                f"the answer closer injects one to {ANSWER_CLOSER_MAX_IDS} delimiter ids"
            )
        if min(self.closing_budget, self.answer_ceiling, self.max_model_len) <= 0:
            raise ValueError("answer closer budgets must be positive")

    @property
    def continuation_tokens(self) -> int:
        """How many output tokens one continuation can add: closer plus budget."""
        return len(self.closer_ids) + self.closing_budget

    def content_length(self, token_ids: Sequence[int]) -> int:
        """Return how many banked ids precede this lane's first stop token."""
        for index, token in enumerate(token_ids):
            if int(token) in self.stop_ids:
                return index
        return len(tuple(token_ids))

    def head_content(self, token_ids: Sequence[int]) -> tuple[int, ...]:
        """Return the head without its stop id: what the closer replaces."""
        ids = tuple(int(token) for token in token_ids)
        return ids[: self.content_length(ids)]

    def is_unclosed(self, token_ids: Sequence[int]) -> bool:
        """Return whether this draw ended inside a block it never closed."""
        return bool(self.unclosed(token_ids))

    def refusal(self, token_ids: Sequence[int], *, prompt_tokens: int) -> str | None:
        """Return why this head cannot afford its closer, or None when it can.

        The prompt, head, closer, and closing budget must all fit the engine
        context; a head that does not fit is left unclosed rather than redrawn.
        """
        tail = self.continuation_tokens
        content = self.content_length(token_ids)
        if prompt_tokens + content + tail > self.max_model_len:
            return "max_model_len"
        return None

    def merged_ids(self, head_ids: Sequence[int], tail_ids: Sequence[int]) -> tuple[int, ...]:
        """Bank head content, the injected closer, and the continuation as one."""
        return (
            *self.head_content(head_ids),
            *self.closer_ids,
            *(int(token) for token in tail_ids),
        )


def validate_banked_answer(
    token_ids: Sequence[int],
    *,
    injected: bool,
    finish_reason: str,
    closing_budget: int,
    answer_ceiling: int,
    label: str,
    closer_ids: Sequence[int] = (),
) -> None:
    """Refuse a banked answer whose length its own protocol cannot explain.

    A natural answer is at most the ceiling; an injected one is head content plus
    closer plus at most one closing budget, located by ``closer_ids`` when given.
    """
    if finish_reason not in FINISH_REASONS:
        raise RuntimeError(f"{label}: banked answer head finish reason is unregistered")
    ids = tuple(int(token) for token in token_ids)
    length = len(ids)
    if not injected:
        if length > answer_ceiling:
            raise RuntimeError(
                f"{label}: banked answer is longer than its registered decode allows"
            )
        if finish_reason == "length" and length != answer_ceiling:
            raise RuntimeError(f"{label}: banked answer length finish did not fill the ceiling")
        return
    if closing_budget != ANSWER_CLOSING_TOKEN_BUDGET:
        raise RuntimeError(f"{label}: banked answer closing budget is unregistered")
    head_bound = answer_ceiling if finish_reason == "length" else answer_ceiling - 1
    if closer_ids:
        _validate_injected_split(
            ids,
            closer_ids=closer_ids,
            head_bound=head_bound,
            filled=finish_reason == "length",
            answer_ceiling=answer_ceiling,
            closing_budget=closing_budget,
            label=label,
        )
        return
    if length > head_bound + closing_budget + ANSWER_CLOSER_MAX_IDS:
        raise RuntimeError(f"{label}: banked answer is longer than its registered decode allows")
    if finish_reason == "length" and length < answer_ceiling + 1:
        raise RuntimeError(f"{label}: banked injected length head did not fill the ceiling")


def _validate_injected_split(
    ids: tuple[int, ...],
    *,
    closer_ids: Sequence[int],
    head_bound: int,
    filled: bool,
    answer_ceiling: int,
    closing_budget: int,
    label: str,
) -> None:
    """Hold the head and the tail of an injected row to their own bounds.

    The closer is matched as one contiguous id sequence, so a lone prefix of a
    spelled-out tag inside the head is not read as a closer.
    """
    closer = tuple(int(token) for token in closer_ids)
    head_length = next(
        (
            index
            for index in range(len(ids) - len(closer) + 1)
            if ids[index : index + len(closer)] == closer
        ),
        None,
    )
    if head_length is None:
        raise RuntimeError(f"{label}: banked injected answer carries no closer id")
    if head_length > head_bound:
        raise RuntimeError(f"{label}: banked answer head is longer than the ceiling allows")
    if filled and head_length != answer_ceiling:
        raise RuntimeError(f"{label}: banked injected length head did not fill the ceiling")
    if len(ids) - head_length - len(closer) > closing_budget:
        raise RuntimeError(f"{label}: banked answer continuation exceeds its closing budget")


def answer_hit_ceiling(finish_reason: str) -> bool:
    """Return whether the head ran to the sampler cap.

    A merged injected row is longer than its head, so a length test on the banked
    ids would flag a head that stopped short; the head's own reason is banked.
    """
    return finish_reason == "length"


__all__ = (
    "ANSWER_CLOSER_MAX_IDS",
    "ANSWER_CLOSING_TOKEN_BUDGET",
    "FINISH_REASONS",
    "AnswerCloser",
    "answer_hit_ceiling",
    "capped_sampling",
    "validate_banked_answer",
)
