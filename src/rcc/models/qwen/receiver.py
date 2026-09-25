"""Embedding-row Qwen receiver and raw-token visible-answer reconstruction.

One receiver prompt is built from the handed-off payload rows and the locally
embedded manager turn, and every answer is rebuilt from its raw token ids.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.qwen.capture import QwenFlatPayload, tensor_content_sha256
from rcc.models.qwen.text import QwenDecodeRequest, QwenDecodeSpec
from rcc.models.route import RouteFamily, tokenizer_decoder
from rcc.run.fleet.answer_closer import ANSWER_CLOSING_TOKEN_BUDGET, AnswerCloser

#: The route's receiver schedule: s0 timed first, then s1 and s2 paired.
QWEN_RECEIVER_SAMPLE_TAGS = ("s0", "s1", "s2")
QWEN_RECEIVER_DECODE_SCHEDULE = (
    "s0 admitted first; s1/s2 offered together after s0 completion; "
    "requests from other items may overlap under token admission"
)


class ReceiverCompletion(Protocol):
    """Minimal streamed completion fields retained by the Qwen receiver."""

    request_id: str
    text: str
    token_ids: Sequence[int]
    n_tokens: int
    finish_reason: str
    num_cached_tokens: int | None
    #: Whether the answer closer continued this draw; banked per sample and
    #: reported as the activation rate.
    answer_injected: bool


@dataclass(frozen=True)
class QwenReceiverPrompt:
    """One prompt-embedding request made only from handed-off and local rows."""

    prompt_embeds: torch.Tensor
    manager_tokens: int
    #: Rows served beyond the manager prompt.
    handoff_rows: int
    #: Rows the payload handed off, the lane's `latent_tokens`. A layout that
    #: drops each worker's begin-of-sequence row serves fewer rows than it
    #: received; the handoff cost is what crossed the wire.
    payload_rows: int
    prompt_rows: int
    payload_layout: str | None
    payload_semantic_arm: str | None
    payload_plan_sha256: str | None
    payload_tensor_sha256: str | None
    profile: BenchmarkProfile

    def __post_init__(self) -> None:
        """Validate the receiver's complete prompt-row geometry."""
        if self.prompt_embeds.ndim != 2:
            raise ValueError("Qwen receiver prompt must be [rows, hidden]")
        if self.prompt_rows != int(self.prompt_embeds.shape[0]):
            raise ValueError("Qwen receiver prompt row count differs")
        if self.prompt_rows != self.manager_tokens + self.handoff_rows:
            raise ValueError("Qwen receiver prompt geometry differs")
        if not 0 <= self.payload_rows - self.handoff_rows <= self.profile.workers_per_item:
            raise ValueError("Qwen receiver served rows differ from the handed-off payload")
        if bool(self.payload_rows) != bool(self.handoff_rows):
            raise ValueError("Qwen receiver payload rows and served rows disagree")
        if self.handoff_rows:
            if (
                self.payload_layout != self.profile.payload_layout
                or self.payload_semantic_arm is None
                or self.payload_plan_sha256 is None
                or self.payload_tensor_sha256 is None
            ):
                raise ValueError("latent Qwen prompts require a signed flat embedding payload")
        elif any(
            field is not None
            for field in (
                self.payload_layout,
                self.payload_semantic_arm,
                self.payload_plan_sha256,
                self.payload_tensor_sha256,
            )
        ):
            raise ValueError("non-latent Qwen prompts cannot claim a payload")


@dataclass(frozen=True)
class QwenReceiverRequest:
    """One of the three independent receiver samples for an item-arm cell."""

    request_id: str
    qid: str
    semantic_arm: str
    tag: str
    seed: int
    prompt: QwenReceiverPrompt
    decode: QwenDecodeSpec
    profile: BenchmarkProfile

    def __post_init__(self) -> None:
        """Bind this request to the sealed answer seed, decode, and latent plan."""
        if self.decode.purpose != "answer":
            raise ValueError("Qwen receiver requests require the answer decode purpose")
        try:
            member_index = QWEN_RECEIVER_SAMPLE_TAGS.index(self.tag)
        except ValueError as exc:
            raise ValueError("Qwen receiver request has an unregistered sample tag") from exc
        QwenDecodeRequest(
            request_id=self.request_id,
            qid=self.qid,
            semantic_arm=self.semantic_arm,
            sample_tag=self.tag,
            member_index=member_index,
            seed=self.seed,
            decode=self.decode,
            profile=self.profile,
        )
        arm = next(arm for arm in self.profile.arms if arm.arm_id == self.semantic_arm)
        if arm.channel == "latent":
            if self.prompt.payload_semantic_arm != self.semantic_arm:
                raise ValueError("Qwen receiver request and latent payload arm differ")
        elif self.prompt.handoff_rows:
            raise ValueError("Qwen text and floor arms cannot receive latent embedding cargo")

    @property
    def sampling(self) -> dict[str, object]:
        """Render the effective family-native fields from the signed decode request."""
        return self.decode.backend_sampling()


@dataclass(frozen=True)
class QwenVisibleAnswer:
    """Independent answer reconstruction from one completion's raw token ids."""

    tag: str
    seed: int
    backend_text: str
    raw_text: str
    visible_text: str
    token_ids: tuple[int, ...]
    visible_decode_token_ids: tuple[int, ...]
    finish_reason: str
    thinking_closed: bool
    num_cached_tokens: int | None
    answer_injected: bool = False


def qwen_answer_closer(
    embedding_weight: torch.Tensor,
    tokenizer: Any,
    *,
    family: RouteFamily,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> AnswerCloser:
    """Bind the registered answer-side closer to this receiver's own table.

    The continuation prompt is built from ids: a token prompt grows by them, an embeds prompt
    gathers them through the manager turn's own table. The tokenizer is here for the trigger.
    """
    profile = family.benchmark_profile(profile)
    decoder = tokenizer_decoder(tokenizer)

    def continuation_prompt(
        prompt: Any,
        content: tuple[int, ...],
        closer_ids: tuple[int, ...],
    ) -> dict[str, object]:
        rows = cast(Mapping[str, Any], prompt)
        ids = (*content, *closer_ids)
        if "prompt_token_ids" in rows:
            return {"prompt_token_ids": [*cast(Sequence[int], rows["prompt_token_ids"]), *ids]}
        embeds = cast(torch.Tensor, rows["prompt_embeds"])
        gathered = token_embedding_rows(embedding_weight, ids, dtype=embeds.dtype)
        return {"prompt_embeds": torch.cat((embeds, gathered.to(embeds)), dim=0)}

    return AnswerCloser(
        closer_ids=family.think_close_token_ids,
        stop_ids=family.stop_token_ids,
        answer_ceiling=profile.answer_ceiling,
        max_model_len=profile.max_model_len,
        continuation_prompt=continuation_prompt,
        # The family's own report trigger (opened and never closed), so the
        # answer and report closers inside this route read one rule.
        unclosed=lambda ids: family.thinking_is_unclosed(ids, decode=decoder),
    )


def token_embedding_rows(
    embedding_weight: torch.Tensor,
    token_ids: Sequence[int],
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Map local token ids through Qwen's identity-normalized input table."""
    if embedding_weight.ndim != 2:
        raise ValueError("Qwen input embedding weight must be [vocab, hidden]")
    ids = tuple(int(token) for token in token_ids)
    if not ids:
        raise ValueError("Qwen manager prompt must contain at least one token")
    if min(ids) < 0 or max(ids) >= int(embedding_weight.shape[0]):
        raise ValueError("Qwen manager token id is outside the input embedding table")
    index = torch.tensor(ids, dtype=torch.long, device=embedding_weight.device)
    return embedding_weight.index_select(0, index).detach().to(device="cpu", dtype=dtype)


def prepare_receiver_prompt(
    embedding_weight: torch.Tensor,
    manager_token_ids: Sequence[int],
    *,
    payload: QwenFlatPayload | None = None,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
    family: RouteFamily | None = None,
    qid: str = "unknown",
    semantic_arm: str = "unknown",
    payload_slot: int | None = None,
    payload_headers: Sequence[Sequence[int]] | None = None,
) -> QwenReceiverPrompt:
    """Combine optional embedding cargo with the locally embedded manager prompt.

    Without a slot the cargo precedes the manager prompt; with ``payload_slot``
    each worker's rows are spliced into the user turn behind its header rows.
    """
    header_rows = sum(len(header) for header in payload_headers) if payload_headers else 0
    effective_profile = family.benchmark_profile(profile) if family is not None else profile
    if family is not None and family.native_sender_prompts:
        effective_profile = family.benchmark_profile(profile)
        handoff = int(payload.rows.shape[0]) if payload is not None else 0
        closing = len(family.think_close_token_ids) + ANSWER_CLOSING_TOKEN_BUDGET
        required = len(manager_token_ids) + header_rows + handoff + effective_profile.answer_ceiling
        required += closing
        primary = next(
            arm for arm in family.profile.physical_arms if arm.semantic_arm == "text_primary"
        )
        limit = min(
            effective_profile.max_model_len,
            int(dict(family.profile.runtime.engine_flags)["max_model_len"]),
            primary.sender_native_max_model_len or 0,
        )
        if required > limit:
            raise RuntimeError(
                f"{qid}/{semantic_arm}: Native receiver context needs "
                f"{len(manager_token_ids)} manager + {handoff} handoff + "
                f"{effective_profile.answer_ceiling} answer + {closing} closer/continuation rows, "
                f"exceeding {limit}"
            )
    elif family is not None and profile.topology_key != FANOUTQA_NATURAL_DEV50.topology_key:
        # A benchmark of another topology holds every family to the profile's
        # own window: manager rows, handed-off rows, the answer ceiling, and the
        # closer must sit inside max_model_len. The chain terminal is the worst.
        if payload is not None and payload.profile.profile_id != profile.profile_id:
            raise ValueError("Qwen receiver and payload benchmark profiles differ")
        handoff = int(payload.rows.shape[0]) if payload is not None else 0
        closing = len(family.think_close_token_ids) + ANSWER_CLOSING_TOKEN_BUDGET
        required = len(manager_token_ids) + header_rows + handoff + effective_profile.answer_ceiling
        required += closing
        limit = effective_profile.max_model_len
        if required > limit:
            raise RuntimeError(
                f"{qid}/{semantic_arm}: {family.lane} receiver context needs "
                f"{len(manager_token_ids)} manager + {handoff} handoff + "
                f"{effective_profile.answer_ceiling} answer + {closing} closer/continuation rows, "
                f"exceeding {limit}"
            )
    manager = token_embedding_rows(embedding_weight, manager_token_ids)
    if payload is None:
        rows = manager
        handoff_rows = 0
        payload_rows = 0
        layout = None
        payload_arm = None
        plan_sha = None
        payload_sha = None
    else:
        if int(payload.rows.shape[1]) != int(manager.shape[1]):
            raise ValueError("Qwen payload and manager embedding widths differ")
        # The embedding-row payload is the only cross-model object accepted
        # here; no cache type has a receiver entry point.
        layout = payload.layout
        payload_arm = payload.semantic_arm
        plan_sha = payload.latent_plan_sha256
        payload_sha = payload.tensor_sha256
        if payload_sha != tensor_content_sha256(payload.rows):
            raise ValueError("Qwen receiver observed different handed-off embedding rows")
        payload_rows = int(payload.rows.shape[0])
        if payload_slot is None:
            rows = torch.cat((payload.rows.to(manager), manager), dim=0)
        else:
            rows = _splice_payload(
                embedding_weight, manager, payload, payload_slot, payload_headers
            )
        handoff_rows = payload_rows
    return QwenReceiverPrompt(
        prompt_embeds=rows,
        manager_tokens=len(tuple(manager_token_ids)) + header_rows,
        handoff_rows=handoff_rows,
        payload_rows=payload_rows,
        prompt_rows=int(rows.shape[0]),
        payload_layout=layout,
        payload_semantic_arm=payload_arm,
        payload_plan_sha256=plan_sha,
        payload_tensor_sha256=payload_sha,
        profile=profile,
    )


def _splice_payload(
    embedding_weight: torch.Tensor,
    manager: torch.Tensor,
    payload: QwenFlatPayload,
    slot: int,
    headers: Sequence[Sequence[int]] | None,
) -> torch.Tensor:
    """Seat each worker's rows behind its header inside the manager prompt."""
    counts = tuple(payload.rows_by_worker)
    if payload.layout == "chain-terminal-v1" and sum(counts) != int(payload.rows.shape[0]):
        counts = (int(payload.rows.shape[0]),)
    if headers is None or len(headers) != len(counts):
        raise ValueError("Qwen in-turn payload needs one header per worker block")
    if sum(counts) != int(payload.rows.shape[0]):
        raise ValueError("Qwen payload worker geometry does not cover its rows")
    if not 0 < slot < int(manager.shape[0]):
        raise ValueError("Qwen payload slot must fall inside the manager prompt")
    parts = [manager[:slot]]
    offset = 0
    for header, count in zip(headers, counts, strict=True):
        parts.append(token_embedding_rows(embedding_weight, header))
        parts.append(payload.rows[offset : offset + count].to(manager))
        offset += count
    parts.append(manager[slot:])
    return torch.cat(parts, dim=0)


def receiver_requests(
    qid: str,
    semantic_arm: str,
    prompt: QwenReceiverPrompt,
    *,
    family: RouteFamily,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> tuple[QwenReceiverRequest, ...]:
    """Derive exactly three signed samples without seed or sampling overrides."""
    if prompt.profile.profile_id != profile.profile_id:
        raise ValueError("Qwen receiver request and prompt benchmark profiles differ")
    if profile.sample_tags != QWEN_RECEIVER_SAMPLE_TAGS:
        raise ValueError(
            f"Qwen receiver schedule {QWEN_RECEIVER_SAMPLE_TAGS!r} does not implement "
            f"{profile.benchmark_key} sample tags {profile.sample_tags!r}"
        )
    # One registered seed per schedule tag. That the three are distinct is a
    # property of the profile, checked when the panel is prepared.
    seed_roster = profile.answer_seeds(qid, semantic_arm)
    decode = QwenDecodeSpec("answer", family, profile=profile)
    return tuple(
        QwenReceiverRequest(
            request_id=f"{qid}:{tag}",
            qid=qid,
            semantic_arm=semantic_arm,
            tag=tag,
            seed=seed,
            prompt=prompt,
            decode=decode,
            profile=profile,
        )
        for tag, seed in zip(QWEN_RECEIVER_SAMPLE_TAGS, seed_roster, strict=True)
    )


def receiver_batches(
    requests: Sequence[QwenReceiverRequest],
) -> tuple[tuple[QwenReceiverRequest, ...], tuple[QwenReceiverRequest, ...]]:
    """Return the timed-first batch and the paired-throughput batch."""
    roster = tuple(requests)
    if tuple(request.tag for request in roster) != QWEN_RECEIVER_SAMPLE_TAGS:
        raise ValueError("Qwen receiver request tags must be ordered s0, s1, s2")
    return roster[:1], roster[1:]


def reconstruct_visible_answers(
    tokenizer: Any,
    requests: Sequence[QwenReceiverRequest],
    completions: Sequence[ReceiverCompletion],
    *,
    family: RouteFamily,
) -> tuple[QwenVisibleAnswer, ...]:
    """Decode raw tokens independently and retain every sample without voting."""
    request_roster = tuple(requests)
    completion_roster = tuple(completions)
    if len(request_roster) != 3 or len(completion_roster) != 3:
        raise RuntimeError("Qwen receiver result cell must contain exactly three samples")
    expected_ids = {request.request_id for request in request_roster}
    by_id: dict[str, ReceiverCompletion] = {}
    for completion in completion_roster:
        if completion.request_id not in expected_ids or completion.request_id in by_id:
            raise RuntimeError("Qwen receiver returned an unknown or duplicate request id")
        by_id[completion.request_id] = completion
    if set(by_id) != expected_ids:
        raise RuntimeError("Qwen receiver returned an incomplete request roster")
    answers: list[QwenVisibleAnswer] = []
    for request in request_roster:
        completion = by_id[request.request_id]
        if completion.finish_reason not in {"stop", "length"}:
            raise RuntimeError(f"{completion.request_id}: invalid receiver finish reason")
        token_ids = tuple(int(token) for token in completion.token_ids)
        if len(token_ids) != int(completion.n_tokens):
            raise RuntimeError(f"{completion.request_id}: token count differs from raw ids")
        if completion.num_cached_tokens not in (None, 0):
            raise RuntimeError(f"{completion.request_id}: prefix caching contaminated Qwen answer")
        stop = next(
            (index for index, token in enumerate(token_ids) if token in family.stop_token_ids),
            len(token_ids),
        )
        visible_token_ids = token_ids[:stop]
        raw = str(
            tokenizer.decode(
                list(visible_token_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
        # Reconstructed from the raw token ids rather than the backend's own
        # decoded text. The backend readout is banked beside it so validation
        # can tell benign tokenizer cleanup from a real mismatch.
        answers.append(
            QwenVisibleAnswer(
                tag=request.tag,
                seed=request.seed,
                backend_text=completion.text,
                raw_text=raw,
                visible_text=family.post_think_report(raw),
                token_ids=token_ids,
                visible_decode_token_ids=visible_token_ids,
                finish_reason=completion.finish_reason,
                # The flag reads the reconstruction of the banked ids, so an
                # injected closer closes the block here as a natural one does,
                # and a continuation that reopened the block does not.
                thinking_closed=family.think_close in raw,
                num_cached_tokens=completion.num_cached_tokens,
                answer_injected=bool(completion.answer_injected),
            )
        )
    return tuple(answers)


__all__ = (
    "QWEN_RECEIVER_DECODE_SCHEDULE",
    "QWEN_RECEIVER_SAMPLE_TAGS",
    "QwenReceiverPrompt",
    "QwenReceiverRequest",
    "QwenVisibleAnswer",
    "prepare_receiver_prompt",
    "qwen_answer_closer",
    "receiver_batches",
    "receiver_requests",
    "reconstruct_visible_answers",
    "token_embedding_rows",
)
