"""Token-exact Gemma prompt and visible-report codecs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.decode import decode_protocol
from rcc.topologies.fanout import FANOUT_M3

GEMMA_STOP_TOKEN_IDS = frozenset({1, 50, 106})
#: The worker prompt width the Gemma text registration is signed under. The
#: value is sealed into `TEXT_REGISTRATION_FINGERPRINT`, so it is written here
#: rather than read off a panel.
GEMMA_WORKER_PROMPT_TOKENS = 50_000
GEMMA_THINKING_ON_PREFIX = (2, 105, 9731, 107, 98, 107, 106, 107, 105, 2364, 107)
GEMMA_THINKING_ON_SUFFIX = (106, 107, 105, 4368, 107)
GEMMA_TOKENIZER_JSON_SHA256 = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
GEMMA_CHANNEL_OPEN = "<|channel>"
GEMMA_CHANNEL_CLOSE = "<channel|>"
_SCAFFOLD_TEXT = re.compile(r"<\|?[^<>]*\|?>")


class PromptSender(Protocol):
    """Registered sender identity required by prompt artifacts."""

    @property
    def semantic_arm(self) -> str: ...

    @property
    def checkpoint(self) -> str: ...

    @property
    def revision(self) -> str: ...

    @property
    def native_max_model_len(self) -> int: ...

    @property
    def tokenizer_json_sha256(self) -> str: ...

    @property
    def chat_template_blob(self) -> str: ...


class RawCompletion(Protocol):
    """Raw token fields required by independent report reconstruction."""

    text: str
    n_tokens: int
    token_ids: Sequence[int]
    finish_reason: str


@dataclass(frozen=True)
class GemmaPromptRecord:
    """One source-bound, sender-tokenized prompt record for three workers."""

    qid: str
    semantic_arm: str
    checkpoint: str
    revision: str
    source_identity: str
    sender_prepared_sha256: str
    tokenizer_json_sha256: str
    chat_template_blob: str
    prompt_ids: tuple[tuple[int, ...], ...]
    prompt_sha256: tuple[str, ...]


@dataclass(frozen=True)
class GemmaVisibleReport:
    """One report reconstructed from raw token ids, independent of backend text."""

    text: str
    visible_token_ids: tuple[int, ...]
    raw_token_ids: tuple[int, ...]
    raw_text: str
    finish_reason: str
    thinking_closed: bool

    @property
    def accepted(self) -> bool:
        """Return Gemma's closed-thinking plus substantive-content decision."""
        return self.thinking_closed and report_is_substantive(self.text)


def encode(tokenizer: Any, text: str) -> list[int]:
    """Encode without adding tokens outside the checkpoint chat template."""
    return [int(value) for value in tokenizer(text, add_special_tokens=False)["input_ids"]]


def single_token_id(tokenizer: Any, text: str) -> int:
    """Resolve one scaffold literal to its exact id, refusing a split token."""
    ids = encode(tokenizer, text)
    if len(ids) != 1:
        raise RuntimeError(f"expected {text!r} to be one token, got {ids}")
    return ids[0]


def receiver_turn_ids(tokenizer: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split the real thinking-on chat template around one sentinel."""
    sentinel = "@@CONTENT@@"
    rendered = cast(
        object,
        tokenizer.apply_chat_template(
            [{"role": "user", "content": sentinel}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors=None,
            enable_thinking=decode_protocol("gemma4").enable_thinking,
        ),
    )
    if isinstance(rendered, Mapping):
        rendered = cast(Mapping[str, object], rendered).get("input_ids")
    if not isinstance(rendered, list):
        raise RuntimeError("Gemma thinking template did not return token ids")
    ids = cast(list[object], rendered)
    if ids and isinstance(ids[0], list):
        ids = cast(list[object], ids[0])
    token_ids: list[int] = []
    for value in ids:
        if not isinstance(value, int):
            raise RuntimeError("Gemma thinking template returned non-integer token ids")
        token_ids.append(value)
    needle = encode(tokenizer, sentinel)
    hits = [
        start
        for start in range(len(token_ids) - len(needle) + 1)
        if token_ids[start : start + len(needle)] == needle
    ]
    if len(hits) != 1:
        raise RuntimeError("Gemma thinking template sentinel must occur exactly once")
    prefix = tuple(token_ids[: hits[0]])
    suffix = tuple(token_ids[hits[0] + len(needle) :])
    if prefix != GEMMA_THINKING_ON_PREFIX or suffix != GEMMA_THINKING_ON_SUFFIX:
        raise RuntimeError("Gemma text sender differs from the registered thinking turn")
    return prefix, suffix


def _prompt_sha256(prompt: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(list(prompt), separators=(",", ":")).encode()).hexdigest()


def _prepared_prompt_ids(
    tokenizer: Any,
    item: Mapping[str, Any],
    sender: PromptSender,
    profile: BenchmarkProfile,
) -> tuple[tuple[int, ...], ...]:
    """Validate the panel's prepared worker prompts against the sender geometry.

    The prepared prompt is the roll frame the panel packed in token space, so
    it is checked, never rebuilt: decode and re-encode is not a round trip.
    """
    _prefix, suffix = receiver_turn_ids(tokenizer)
    qid = str(item.get("qid") or "")
    question = item.get("question")
    question = getattr(question, "question", question)
    if not qid or not isinstance(question, str) or not question:
        raise RuntimeError("Gemma report item has no registered question")
    profile.report_seeds(qid, profile.sample_tags[0])
    prompt_rows = cast(Sequence[Sequence[Any]], item.get("prompt_ids") or ())
    prompts = tuple(tuple(int(value) for value in row) for row in prompt_rows)
    if len(prompts) != FANOUT_M3.workers_per_item:
        raise RuntimeError(f"{qid}: report item does not carry three workers")
    if not profile.admits_worker_prompt_tokens([len(prompt) for prompt in prompts]) or any(
        prompt[-len(suffix) :] != suffix for prompt in prompts
    ):
        raise RuntimeError("Gemma report prompt differs from the registered thinking geometry")
    if any(
        len(prompt) + FANOUTQA_NATURAL_DEV50.report_ceiling > sender.native_max_model_len
        for prompt in prompts
    ):
        raise RuntimeError(f"{qid}/{sender.semantic_arm}: report exceeds sender native context")
    return prompts


def bind_sender_prompt_artifact(
    tokenizer: Any,
    item: Mapping[str, Any],
    sender: PromptSender,
    *,
    source_identity: str,
    sender_prepared_sha256: str,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> dict[str, object]:
    """Bind exact prompt tokens to source, preparation, and tokenizer identities."""
    if not source_identity or len(sender_prepared_sha256) != 64:
        raise ValueError("Gemma prompt artifact requires source and prepared identities")
    prompts = _prepared_prompt_ids(tokenizer, item, sender, profile)
    return {
        "qid": str(item.get("qid") or ""),
        "semantic_arm": sender.semantic_arm,
        "checkpoint": sender.checkpoint,
        "revision": sender.revision,
        "source_identity": source_identity,
        "sender_prepared_sha256": sender_prepared_sha256,
        "tokenizer_json_sha256": sender.tokenizer_json_sha256,
        "chat_template_blob": sender.chat_template_blob,
        "prompt_ids": [list(prompt) for prompt in prompts],
        "prompt_sha256": [_prompt_sha256(prompt) for prompt in prompts],
    }


def prompt_record(
    tokenizer: Any,
    item: Mapping[str, Any],
    sender: PromptSender,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> GemmaPromptRecord:
    """Validate and hydrate one bound sender prompt artifact."""
    artifacts_value = item.get("text_prompt_artifacts")
    artifact_value = (
        cast(Mapping[str, object], artifacts_value).get(sender.semantic_arm)
        if isinstance(artifacts_value, Mapping)
        else None
    )
    prepared_value = item.get("text_prepared_sha256_by_arm")
    prepared_identity = (
        cast(Mapping[str, object], prepared_value).get(sender.semantic_arm)
        if isinstance(prepared_value, Mapping)
        else None
    )
    source_identity = item.get("source_identity")
    if not isinstance(artifact_value, Mapping):
        raise RuntimeError(f"{item.get('qid')}/{sender.semantic_arm}: prompt artifact mismatch")
    artifact = cast(Mapping[str, object], artifact_value)
    if (
        artifact.get("qid") != item.get("qid")
        or artifact.get("semantic_arm") != sender.semantic_arm
        or artifact.get("checkpoint") != sender.checkpoint
        or artifact.get("revision") != sender.revision
        or artifact.get("source_identity") != source_identity
        or artifact.get("sender_prepared_sha256") != prepared_identity
        or artifact.get("tokenizer_json_sha256") != sender.tokenizer_json_sha256
        or artifact.get("chat_template_blob") != sender.chat_template_blob
    ):
        raise RuntimeError(f"{item.get('qid')}/{sender.semantic_arm}: prompt artifact mismatch")
    prompt_rows = cast(Sequence[Sequence[Any]], artifact.get("prompt_ids") or ())
    prompts = tuple(tuple(int(value) for value in row) for row in prompt_rows)
    hashes = tuple(_prompt_sha256(prompt) for prompt in prompts)
    _prefix, suffix = receiver_turn_ids(tokenizer)
    if (
        len(prompts) != FANOUT_M3.workers_per_item
        or not profile.admits_worker_prompt_tokens([len(prompt) for prompt in prompts])
        or any(prompt[-len(suffix) :] != suffix for prompt in prompts)
        or any(
            len(prompt) + FANOUTQA_NATURAL_DEV50.report_ceiling > sender.native_max_model_len
            for prompt in prompts
        )
        or tuple(cast(Sequence[object], artifact.get("prompt_sha256") or ())) != hashes
        or not isinstance(source_identity, str)
        or not isinstance(prepared_identity, str)
    ):
        raise RuntimeError(f"{item.get('qid')}/{sender.semantic_arm}: sender prompts invalid")
    return GemmaPromptRecord(
        qid=str(item.get("qid") or ""),
        semantic_arm=sender.semantic_arm,
        checkpoint=sender.checkpoint,
        revision=sender.revision,
        source_identity=source_identity,
        sender_prepared_sha256=prepared_identity,
        tokenizer_json_sha256=sender.tokenizer_json_sha256,
        chat_template_blob=sender.chat_template_blob,
        prompt_ids=prompts,
        prompt_sha256=hashes,
    )


def report_is_substantive(text: str) -> bool:
    """Reject empty, whitespace-only, and Gemma scaffold-only reports."""
    return bool(_SCAFFOLD_TEXT.sub(" ", text).strip())


def _visible_tokens(tokenizer: Any, tokens: Sequence[int]) -> tuple[int, ...]:
    channel_open = single_token_id(tokenizer, GEMMA_CHANNEL_OPEN)
    channel_close = single_token_id(tokenizer, GEMMA_CHANNEL_CLOSE)
    visible: list[int] = []
    in_channel = False
    for value in tokens:
        token = int(value)
        if token in GEMMA_STOP_TOKEN_IDS:
            break
        if token == channel_open:
            in_channel = True
            continue
        if token == channel_close:
            in_channel = False
            continue
        if not in_channel:
            visible.append(token)
    special = {int(value) for value in getattr(tokenizer, "all_special_ids", ()) or ()}
    return tuple(token for token in visible if token not in special)


def visible_report(tokenizer: Any, completion: RawCompletion) -> GemmaVisibleReport:
    """Reconstruct visible content and first-open-followed-by-close status."""
    if completion.finish_reason not in {"stop", "length"}:
        raise RuntimeError(f"invalid Gemma report finish reason {completion.finish_reason!r}")
    raw = tuple(int(value) for value in completion.token_ids)
    if completion.n_tokens != len(raw):
        raise RuntimeError("Gemma report token count differs from its raw token ids")
    channel_open = single_token_id(tokenizer, GEMMA_CHANNEL_OPEN)
    channel_close = single_token_id(tokenizer, GEMMA_CHANNEL_CLOSE)
    try:
        thought_open = raw.index(channel_open)
        thought_close = raw.index(channel_close, thought_open + 1)
    except ValueError:
        thought_open = -1
        thought_close = -1
    visible = _visible_tokens(tokenizer, raw)
    return GemmaVisibleReport(
        text=str(tokenizer.decode(visible)).strip() if visible else "",
        visible_token_ids=visible,
        raw_token_ids=raw,
        raw_text=str(completion.text),
        finish_reason=str(completion.finish_reason),
        thinking_closed=0 <= thought_open < thought_close,
    )


__all__ = (
    "GEMMA_CHANNEL_CLOSE",
    "GEMMA_CHANNEL_OPEN",
    "GEMMA_STOP_TOKEN_IDS",
    "GEMMA_THINKING_ON_PREFIX",
    "GEMMA_THINKING_ON_SUFFIX",
    "GEMMA_TOKENIZER_JSON_SHA256",
    "GEMMA_WORKER_PROMPT_TOKENS",
    "GemmaPromptRecord",
    "GemmaVisibleReport",
    "bind_sender_prompt_artifact",
    "encode",
    "prompt_record",
    "receiver_turn_ids",
    "report_is_substantive",
    "single_token_id",
    "visible_report",
)
