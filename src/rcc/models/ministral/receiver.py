"""Ministral embedding-row receiver and raw-token answer reconstruction.

The payload rows are prepended to the locally embedded manager turn, and every
answer is rebuilt from its raw token ids.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import torch

from rcc.benchmarks.fanoutqa import (
    FANOUTQA_NATURAL_DEV50,
    SEALED_M3_PLAN_LABEL,
    arms_with_full,
    shared_answer_seeds,
)
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.ministral import MINISTRAL
from rcc.models.ministral.payload import (
    MINISTRAL_PAYLOAD_LAYOUT,
    MinistralFlatPayload,
    tensor_content_sha256,
    token_embedding_rows,
)
from rcc.models.ministral.text_codec import (
    MINISTRAL_STOP_TOKEN_IDS,
    MINISTRAL_THINK_CLOSE_ID,
    MinistralTextCodec,
)
from rcc.run.fleet.answer_closer import AnswerCloser
from rcc.run.fleet.latency import WARMUP_MAX_TOKENS, warmup_seed
from rcc.topologies.fanout import FANOUT_M3

MINISTRAL_RECEIVER_SAMPLE_TAGS = FANOUTQA_NATURAL_DEV50.sample_tags
#: The receiver admission window, read off the one registered engine flag so a
#: served window and a checked window cannot differ. It is the engine context,
#: not the panel cap: it bounds what the receiver may be asked to run.
MINISTRAL_RECEIVER_MAX_MODEL_LEN = int(dict(MINISTRAL.runtime.engine_flags)["max_model_len"])
MINISTRAL_RECEIVER_DECODE_SCHEDULE = (
    "s0 admitted first; s1/s2 offered together after s0 completion; "
    "requests from other items may overlap under token admission"
)


class ReceiverCompletion(Protocol):
    """Raw completion fields retained for independent reconstruction."""

    request_id: str
    text: str
    token_ids: Sequence[int]
    n_tokens: int
    finish_reason: str
    num_cached_tokens: int | None
    queued_ts: float | None
    first_token_ts: float | None
    last_token_ts: float | None
    #: The answer closer's one continuation, on the engine clock; None on a
    #: sample that was not continued.
    continuation_queued_ts: float | None
    continuation_first_token_ts: float | None
    #: Whether the answer closer continued this draw; banked per sample and
    #: reported as the activation rate.
    answer_injected: bool


@dataclass(frozen=True)
class MinistralWarmupDecodeSpec:
    """Family-native sampling truncated to one discarded warmup completion."""

    def backend_sampling(self) -> dict[str, object]:
        """Render the registered vLLM fields at the warmup output ceiling."""
        return {
            **MINISTRAL.decode.backend_sampling(),
            "presence_penalty": MINISTRAL.decode.presence_penalty,
            "max_tokens": WARMUP_MAX_TOKENS,
            "stop_token_ids": list(MINISTRAL_STOP_TOKEN_IDS),
        }

    def to_dict(self) -> dict[str, object]:
        """Bind the family decode identity under an explicit warmup purpose."""
        return {
            "purpose": "warmup",
            "decode_profile": MINISTRAL.decode.profile_id,
            "decode_fingerprint": MINISTRAL.decode.identity_hash,
            "benchmark_profile": SEALED_M3_PLAN_LABEL,
            "enable_thinking": MINISTRAL.decode.enable_thinking,
            **self.backend_sampling(),
        }


@dataclass(frozen=True)
class MinistralWarmupRequest:
    """One discarded warmup request, seeded outside every panel seed law."""

    request_id: str
    seed: int
    decode: MinistralWarmupDecodeSpec = field(default_factory=MinistralWarmupDecodeSpec)

    def __post_init__(self) -> None:
        """Require a nonempty identity and a nonnegative discarded seed."""
        if not self.request_id or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("Ministral warmup request identity or seed is invalid")


class TokenWarmableEngine(Protocol):
    """The token-id decode seam a sender lane drives before it is measured."""

    def decode_token_ids_full(
        self,
        prompts: Sequence[Sequence[int]],
        requests: Sequence[Any],
    ) -> Sequence[Any]:
        """Decode direct token prompts for signed request identities."""
        ...


class EmbedWarmableEngine(Protocol):
    """The embedding-row decode seam the receiver lane is measured on."""

    def decode_embeds_full(
        self,
        embeds: Sequence[torch.Tensor],
        requests: Sequence[Any],
    ) -> Sequence[Any]:
        """Decode receiver embedding rows for signed request identities."""
        ...


def _warmup_request(tag: str) -> MinistralWarmupRequest:
    return MinistralWarmupRequest(request_id=f"warmup:{tag}", seed=warmup_seed(tag))


def warm_token_engine(engine: TokenWarmableEngine, token_ids: Sequence[int], *, tag: str) -> float:
    """Decode once and discard it, returning the measured warmup seconds.

    The first request through a fresh engine pays for lazy allocation, graph
    capture, and autotuning, which would otherwise land on the first item.
    """
    started = time.perf_counter()
    engine.decode_token_ids_full([list(token_ids)], [_warmup_request(tag)])
    return time.perf_counter() - started


def warm_embed_engine(
    engine: EmbedWarmableEngine,
    embedding_weight: torch.Tensor,
    token_ids: Sequence[int],
    *,
    tag: str,
) -> float:
    """Warm the exact prompt-embeds route the receiver is later measured on."""
    started = time.perf_counter()
    engine.decode_embeds_full(
        [token_embedding_rows(embedding_weight, token_ids)],
        [_warmup_request(tag)],
    )
    return time.perf_counter() - started


@dataclass(frozen=True)
class MinistralAnswerDecodeSpec:
    """The sole family-native answer sampling contract."""

    def __post_init__(self) -> None:
        """Refuse a registration that lost reasoning or enabled top-k."""
        if not MINISTRAL.decode.enable_thinking or MINISTRAL.decode.top_k is not None:
            raise RuntimeError("Ministral receiver requires thinking on and top-k off")

    def backend_sampling(self) -> dict[str, object]:
        """Render the vLLM fields, omitting top-k."""
        return {
            **MINISTRAL.decode.backend_sampling(),
            "presence_penalty": MINISTRAL.decode.presence_penalty,
            "max_tokens": FANOUTQA_NATURAL_DEV50.answer_ceiling,
            "stop_token_ids": list(MINISTRAL_STOP_TOKEN_IDS),
        }

    def to_dict(self) -> dict[str, object]:
        """Bind the family decode profile and answer ceiling."""
        return {
            "purpose": "answer",
            "decode_profile": MINISTRAL.decode.profile_id,
            "decode_fingerprint": MINISTRAL.decode.identity_hash,
            "benchmark_profile": SEALED_M3_PLAN_LABEL,
            "enable_thinking": MINISTRAL.decode.enable_thinking,
            **self.backend_sampling(),
        }


@dataclass(frozen=True)
class MinistralReceiverPrompt:
    """Local manager rows optionally preceded by one signed latent payload."""

    prompt_embeds: torch.Tensor
    manager_tokens: int
    handoff_rows: int
    prompt_rows: int
    payload: MinistralFlatPayload | None

    def __post_init__(self) -> None:
        """Validate row geometry and reject every foreign-KV representation."""
        if self.prompt_embeds.ndim != 2 or self.prompt_embeds.dtype != torch.bfloat16:
            raise ValueError("Ministral receiver prompt must be rank-two bfloat16 rows")
        if self.prompt_rows != int(self.prompt_embeds.shape[0]):
            raise ValueError("Ministral receiver prompt row count differs")
        if self.prompt_rows != self.manager_tokens + self.handoff_rows:
            raise ValueError("Ministral receiver prompt geometry differs")
        if self.payload is None:
            if self.handoff_rows != 0:
                raise ValueError("Ministral non-latent prompt cannot claim handoff rows")
        else:
            if self.payload.layout != MINISTRAL_PAYLOAD_LAYOUT:
                raise ValueError("Ministral receiver requires flat-interleave-v1")
            if self.handoff_rows != int(self.payload.rows.shape[0]):
                raise ValueError("Ministral receiver payload geometry differs")
            if not torch.equal(self.prompt_embeds[: self.handoff_rows], self.payload.rows):
                raise ValueError("Ministral receiver observed different handoff rows")
            if tensor_content_sha256(self.prompt_embeds[: self.handoff_rows]) != (
                self.payload.tensor_sha256
            ):
                raise ValueError("Ministral receiver payload hash differs")


@dataclass(frozen=True)
class MinistralReceiverRequest:
    """One independently seeded receiver request in an item-arm cell."""

    request_id: str
    qid: str
    semantic_arm: str
    sample_tag: str
    sample_index: int
    seed: int
    prompt: MinistralReceiverPrompt
    decode: MinistralAnswerDecodeSpec
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50

    def __post_init__(self) -> None:
        """Bind item, arm, sample tag, seed, and payload identity."""
        if not self.request_id:
            raise ValueError("Ministral receiver request id must be nonempty")
        if self.qid not in self.profile.question_ids:
            raise ValueError("Ministral receiver request has an unregistered question")
        arms = {arm.arm_id: arm for arm in arms_with_full(self.profile)}
        arm = arms.get(self.semantic_arm)
        if arm is None:
            raise ValueError("Ministral receiver request has an unregistered arm")
        if (
            self.sample_index not in range(len(MINISTRAL_RECEIVER_SAMPLE_TAGS))
            or self.sample_tag != MINISTRAL_RECEIVER_SAMPLE_TAGS[self.sample_index]
            or self.seed
            != shared_answer_seeds(self.profile, self.qid, self.semantic_arm)[self.sample_index]
        ):
            raise ValueError("Ministral receiver seed differs from the sealed all-arm grid")
        if arm.channel in {"latent", "full"}:
            if self.prompt.payload is None or self.prompt.payload.semantic_arm != self.semantic_arm:
                raise ValueError("Ministral latent receiver request lacks its signed payload")
        elif self.prompt.payload is not None:
            raise ValueError("Ministral floor and text requests cannot receive latent rows")

    @property
    def sampling(self) -> dict[str, object]:
        """Return the effective family-native backend fields."""
        return self.decode.backend_sampling()


@dataclass(frozen=True)
class MinistralVisibleAnswer:
    """One answer reconstructed independently from raw token ids."""

    sample_tag: str
    sample_index: int
    seed: int
    backend_text: str
    raw_text: str
    visible_text: str
    token_ids: tuple[int, ...]
    visible_token_ids: tuple[int, ...]
    finish_reason: str
    thinking_closed: bool
    num_cached_tokens: int | None
    answer_injected: bool = False


def ministral_answer_closer(
    codec: MinistralTextCodec,
    embedding_weight: torch.Tensor,
    *,
    embedding_prompt: Callable[[torch.Tensor], Any],
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> AnswerCloser:
    """Bind the registered answer-side closer to the 14B codec and table.

    The trigger is the codec's own unclosed predicate, not the token-id one:
    Tekken encodes `[THINK]` both as a reserved id and as ordinary text.
    """

    def continuation_prompt(
        prompt: Any,
        content: tuple[int, ...],
        closer_ids: tuple[int, ...],
    ) -> Any:
        rows = cast(torch.Tensor, cast(Mapping[str, Any], prompt)["prompt_embeds"])
        gathered = token_embedding_rows(embedding_weight, (*content, *closer_ids))
        return embedding_prompt(torch.cat((rows, gathered.to(rows)), dim=0))

    return AnswerCloser(
        closer_ids=(MINISTRAL_THINK_CLOSE_ID,),
        stop_ids=tuple(MINISTRAL_STOP_TOKEN_IDS),
        answer_ceiling=profile.answer_ceiling,
        max_model_len=MINISTRAL_RECEIVER_MAX_MODEL_LEN,
        unclosed=codec.thinking_is_unclosed,
        continuation_prompt=continuation_prompt,
    )


def prepare_receiver_prompt(
    embedding_weight: torch.Tensor,
    manager_token_ids: Sequence[int],
    *,
    payload: MinistralFlatPayload | None = None,
) -> MinistralReceiverPrompt:
    """Prepend optional embedding cargo to locally embedded manager tokens."""
    manager = token_embedding_rows(embedding_weight, manager_token_ids)
    if payload is None:
        rows = manager
        handoff_rows = 0
    else:
        if int(payload.rows.shape[1]) != int(manager.shape[1]):
            raise ValueError("Ministral payload and manager embedding widths differ")
        rows = torch.cat((payload.rows, manager), dim=0)
        handoff_rows = int(payload.rows.shape[0])
    if (
        int(rows.shape[0]) + FANOUTQA_NATURAL_DEV50.answer_ceiling
        > MINISTRAL_RECEIVER_MAX_MODEL_LEN
    ):
        raise ValueError("Ministral receiver exceeds the registered engine context")
    return MinistralReceiverPrompt(
        prompt_embeds=rows,
        manager_tokens=len(tuple(manager_token_ids)),
        handoff_rows=handoff_rows,
        prompt_rows=int(rows.shape[0]),
        payload=payload,
    )


def receiver_requests(
    qid: str,
    semantic_arm: str,
    prompt: MinistralReceiverPrompt,
    *,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> tuple[MinistralReceiverRequest, ...]:
    """Create exactly three registered requests with no seed or decode override."""
    seeds = shared_answer_seeds(profile, qid, semantic_arm)
    if len(seeds) != FANOUT_M3.workers_per_item or len(set(seeds)) != len(seeds):
        raise RuntimeError("Ministral receiver seeds differ from canonical M=3")
    decode = MinistralAnswerDecodeSpec()
    return tuple(
        MinistralReceiverRequest(
            request_id=f"{qid}:{semantic_arm}:answer:{tag}",
            qid=qid,
            semantic_arm=semantic_arm,
            sample_tag=tag,
            sample_index=index,
            seed=seed,
            prompt=prompt,
            decode=decode,
            profile=profile,
        )
        for index, (tag, seed) in enumerate(zip(MINISTRAL_RECEIVER_SAMPLE_TAGS, seeds, strict=True))
    )


def receiver_batches(
    requests: Sequence[MinistralReceiverRequest],
) -> tuple[tuple[MinistralReceiverRequest, ...], tuple[MinistralReceiverRequest, ...]]:
    """Return the timed-first sample followed by the paired throughput samples."""
    roster = tuple(requests)
    if tuple(request.sample_tag for request in roster) != MINISTRAL_RECEIVER_SAMPLE_TAGS:
        raise ValueError("Ministral receiver requests must be ordered s0, s1, s2")
    return roster[:1], roster[1:]


def reconstruct_visible_answers(
    codec: MinistralTextCodec,
    requests: Sequence[MinistralReceiverRequest],
    completions: Sequence[ReceiverCompletion],
) -> tuple[MinistralVisibleAnswer, ...]:
    """Decode raw ids independently and retain all three samples without voting."""
    codec.require_verified()
    if (codec.checkpoint, codec.revision) != (MINISTRAL.checkpoint, MINISTRAL.revision):
        raise RuntimeError("Ministral answers require the registered 14B receiver codec")
    request_roster = tuple(requests)
    if len(request_roster) != FANOUT_M3.workers_per_item:
        raise RuntimeError("Ministral receiver cell must contain exactly three requests")
    by_id: dict[str, ReceiverCompletion] = {}
    for completion in completions:
        if not completion.request_id or completion.request_id in by_id:
            raise RuntimeError("Ministral receiver returned a duplicate or empty request id")
        by_id[completion.request_id] = completion
    if set(by_id) != {request.request_id for request in request_roster}:
        raise RuntimeError("Ministral receiver completion identities differ from the cell")
    answers: list[MinistralVisibleAnswer] = []
    for request in request_roster:
        completion = by_id[request.request_id]
        tokens = tuple(int(token) for token in completion.token_ids)
        if completion.finish_reason not in {"stop", "length"}:
            raise RuntimeError(f"{completion.request_id}: invalid receiver finish reason")
        if len(tokens) != int(completion.n_tokens):
            raise RuntimeError(f"{completion.request_id}: raw receiver token count differs")
        if completion.num_cached_tokens not in (None, 0):
            raise RuntimeError(
                f"{completion.request_id}: prefix caching contaminated Ministral answer"
            )
        stop = next(
            (index for index, token in enumerate(tokens) if token in MINISTRAL_STOP_TOKEN_IDS),
            len(tokens),
        )
        raw_ids = tokens[:stop]
        visible_ids, visible_text, thinking_closed = codec.visible_answer(
            tokens, ended=completion.finish_reason == "stop"
        )
        answers.append(
            MinistralVisibleAnswer(
                sample_tag=request.sample_tag,
                sample_index=request.sample_index,
                seed=request.seed,
                backend_text=str(completion.text),
                raw_text=str(codec.tokenizer.decode(list(raw_ids))),
                visible_text=visible_text,
                token_ids=tokens,
                visible_token_ids=visible_ids,
                finish_reason=str(completion.finish_reason),
                thinking_closed=thinking_closed,
                num_cached_tokens=completion.num_cached_tokens,
                answer_injected=bool(completion.answer_injected),
            )
        )
    return tuple(answers)


__all__ = (
    "MINISTRAL_RECEIVER_DECODE_SCHEDULE",
    "MINISTRAL_RECEIVER_MAX_MODEL_LEN",
    "MINISTRAL_RECEIVER_SAMPLE_TAGS",
    "MinistralAnswerDecodeSpec",
    "MinistralReceiverPrompt",
    "MinistralReceiverRequest",
    "MinistralVisibleAnswer",
    "MinistralWarmupDecodeSpec",
    "MinistralWarmupRequest",
    "ministral_answer_closer",
    "prepare_receiver_prompt",
    "receiver_batches",
    "receiver_requests",
    "reconstruct_visible_answers",
    "warm_embed_engine",
    "warm_token_engine",
)
