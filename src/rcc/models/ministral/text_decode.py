"""The one registered Ministral decode contract and its engine seams.

This is the sampling half of the text lane: the request, its sampling, and what
an engine must offer. The generation half is :mod:`rcc.models.ministral.text`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50, SEALED_M3_PLAN_LABEL
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.ministral import MINISTRAL
from rcc.models.ministral.text_codec import (
    MINISTRAL_STOP_TOKEN_IDS,
    MINISTRAL_TEXT_ARMS,
)

FINISH_REASONS = frozenset({"stop", "length"})
#: How far a draw may decode after its closer is injected. Like the seed
#: ladder, it is not part of the decode or runtime fingerprint.
MINISTRAL_CLOSING_TOKEN_BUDGET = 4096


class TextCompletion(Protocol):
    """Raw vLLM completion fields retained for independent reconstruction."""

    request_id: str
    text: str
    n_tokens: int
    token_ids: Sequence[int]
    finish_reason: str
    num_cached_tokens: int | None
    queued_ts: float | None
    scheduled_ts: float | None
    first_token_ts: float | None


@dataclass(frozen=True)
class MinistralDecodeSpec:
    """The sole family-native sampling contract for a worker report."""

    #: Set only on the one continuation of a draw whose closer was injected.
    #: Every sampling field except the output cap is untouched, so the
    #: continuation runs the same registered decode.
    closing: bool = False

    def __post_init__(self) -> None:
        """Refuse a registration that lost thinking or enabled top-k."""
        if not MINISTRAL.decode.enable_thinking or MINISTRAL.decode.top_k is not None:
            raise RuntimeError("Ministral Reasoning decode must keep thinking on and top-k off")

    @property
    def max_tokens(self) -> int:
        """Return the registered report ceiling, or the closing budget.

        The ceiling bounds the head and the closing budget the continuation, so an
        injected report may reach the ceiling plus one closer id plus that budget.
        """
        return (
            MINISTRAL_CLOSING_TOKEN_BUDGET
            if self.closing
            else FANOUTQA_NATURAL_DEV50.report_ceiling
        )

    def backend_sampling(self) -> dict[str, object]:
        """Render the vLLM sampling fields, omitting top-k."""
        return {
            **MINISTRAL.decode.backend_sampling(),
            "presence_penalty": MINISTRAL.decode.presence_penalty,
            "max_tokens": self.max_tokens,
            "stop_token_ids": list(MINISTRAL_STOP_TOKEN_IDS),
        }

    def to_dict(self) -> dict[str, object]:
        """Bind the family decode, the panel, and the report ceiling in one record."""
        return {
            "purpose": "report",
            "decode_profile": MINISTRAL.decode.profile_id,
            "decode_fingerprint": MINISTRAL.decode.identity_hash,
            "benchmark_profile": SEALED_M3_PLAN_LABEL,
            "enable_thinking": MINISTRAL.decode.enable_thinking,
            **self.backend_sampling(),
        }


@dataclass(frozen=True)
class MinistralDecodeRequest:
    """One report request whose seed comes only from the registered panel."""

    request_id: str
    qid: str
    semantic_arm: str
    sample_tag: str
    member_index: int
    seed: int
    decode: MinistralDecodeSpec
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50

    def __post_init__(self) -> None:
        """Reject caller-minted questions, tags, worker indices, and seeds."""
        if not self.request_id:
            raise ValueError("Ministral decode request id must be nonempty")
        if self.semantic_arm not in MINISTRAL_TEXT_ARMS:
            raise ValueError("Ministral report request has an unregistered text arm")
        expected = self.profile.report_seeds(self.qid, self.sample_tag)
        if (
            self.member_index not in range(len(expected))
            or self.seed != expected[self.member_index]
        ):
            raise ValueError("Ministral decode request seed differs from sealed panel identity")


class TextEngine(Protocol):
    """Engine seam for all three token-ID-native Ministral text senders."""

    def decode_token_ids_full(
        self,
        prompts: Sequence[Sequence[int]],
        requests: Sequence[MinistralDecodeRequest],
    ) -> Sequence[TextCompletion]:
        """Decode one independent report per signed request."""
        ...


__all__ = (
    "FINISH_REASONS",
    "MINISTRAL_CLOSING_TOKEN_BUDGET",
    "MinistralDecodeRequest",
    "MinistralDecodeSpec",
    "TextCompletion",
    "TextEngine",
)
