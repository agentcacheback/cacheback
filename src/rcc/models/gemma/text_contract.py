"""Pinned Gemma sender, sampling, and report-registration contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.gemma import GEMMA
from rcc.models.gemma.text_bank import TEXT_REPORT_BANK_SCHEMA
from rcc.models.gemma.text_codec import (
    GEMMA_STOP_TOKEN_IDS,
    GEMMA_THINKING_ON_PREFIX,
    GEMMA_THINKING_ON_SUFFIX,
    GEMMA_TOKENIZER_JSON_SHA256,
    GEMMA_WORKER_PROMPT_TOKENS,
)
from rcc.run import identity
from rcc.topologies.fanout import FANOUT_M3

TEXT_REPORT_SCHEMA = "gemma4-fanoutqa-text-report-v1"
REPORT_BACKEND = "vllm"
REPORT_DECODE_PATH = "vllm-batched-prompt-token-ids"
GEMMA_REPORT_SEED_TAGS = FANOUTQA_NATURAL_DEV50.sample_tags
GEMMA_TEXT_ARMS = tuple(arm.arm_id for arm in FANOUTQA_NATURAL_DEV50.arms if arm.channel == "text")


@dataclass(frozen=True)
class Sampling:
    """One sampling value accepted by the vLLM decode seam."""

    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    max_tokens: int
    stop_token_ids: tuple[int, ...]
    bad_words: tuple[str, ...]
    seed: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Return exact backend fields and family-native identity."""
        return {
            "profile": GEMMA.decode.profile_id,
            "sampling_fingerprint": GEMMA.decode.identity_hash,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "presence_penalty": self.presence_penalty,
            "max_tokens": self.max_tokens,
            "stop_token_ids": list(self.stop_token_ids),
            "bad_words": list(self.bad_words),
            "seed": self.seed,
            "enable_thinking": GEMMA.decode.enable_thinking,
        }


@dataclass(frozen=True)
class TextSenderSpec:
    """One exact Gemma report sender feeding the resident 12B receiver."""

    semantic_arm: str
    policy: str
    checkpoint: str
    revision: str
    native_max_model_len: int
    suppressed_token_ids: tuple[int, ...]
    resident: bool
    chat_template_blob: str
    tokenizer_json_sha256: str = GEMMA_TOKENIZER_JSON_SHA256
    thinking_on_prefix: tuple[int, ...] = GEMMA_THINKING_ON_PREFIX
    thinking_on_suffix: tuple[int, ...] = GEMMA_THINKING_ON_SUFFIX

    @property
    def receiver_checkpoint(self) -> str:
        """Return the common resident receiver checkpoint."""
        return GEMMA.checkpoint

    @property
    def receiver_revision(self) -> str:
        """Return the common resident receiver revision."""
        return GEMMA.revision

    def to_dict(self) -> dict[str, object]:
        """Return the pinned sender and tokenizer identity."""
        return {
            "semantic_arm": self.semantic_arm,
            "policy": self.policy,
            "checkpoint": self.checkpoint,
            "revision": self.revision,
            "tokenizer": self.checkpoint,
            "tokenizer_revision": self.revision,
            "native_max_model_len": self.native_max_model_len,
            "context_extension": "none",
            "suppressed_token_ids": list(self.suppressed_token_ids),
            "resident": self.resident,
            "chat_template_blob": self.chat_template_blob,
            "tokenizer_json_sha256": self.tokenizer_json_sha256,
            "thinking_on_prefix": list(self.thinking_on_prefix),
            "thinking_on_suffix": list(self.thinking_on_suffix),
        }


TEXT_SENDERS = (
    TextSenderSpec(
        "text_primary",
        "gemma4_12b_text",
        "google/gemma-4-12B-it",
        "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        262_144,
        (258_883, 258_882),
        True,
        "4741bf6e4132ba23a5537f9d6e74e9a6d613d7cd",
    ),
    TextSenderSpec(
        "text_medium",
        "gemma4_12b_calls_gemma4_e4b_text",
        "google/gemma-4-E4B-it",
        "ee0ef6023621cff504d758262d4e04895a5af4a2",
        131_072,
        (),
        False,
        "fbe3b59b625cd1b8850ea592d4203df6ec04684b",
    ),
    TextSenderSpec(
        "text_small",
        "gemma4_12b_calls_gemma4_e2b_text",
        "google/gemma-4-E2B-it",
        "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        131_072,
        (),
        False,
        "fbe3b59b625cd1b8850ea592d4203df6ec04684b",
    ),
)
_SENDERS_BY_ARM = {sender.semantic_arm: sender for sender in TEXT_SENDERS}


def registered_text_senders() -> tuple[TextSenderSpec, ...]:
    """Validate all three senders against the model physical-arm registry."""
    physical = {arm.semantic_arm: arm for arm in GEMMA.physical_arms}
    for sender in TEXT_SENDERS:
        arm = physical.get(sender.semantic_arm)
        if arm is None or (
            arm.policy != sender.policy
            or arm.sender_checkpoint != sender.checkpoint
            or arm.sender_revision != sender.revision
            or arm.sender_native_max_model_len != sender.native_max_model_len
            or arm.sender_context_extension != "none"
            or arm.selector is not None
        ):
            raise RuntimeError(f"Gemma text arm {sender.semantic_arm!r} differs from registry")
    return TEXT_SENDERS


def text_sender_spec(semantic_arm: str) -> TextSenderSpec:
    """Return one registered sender without accepting aliases."""
    registered_text_senders()
    try:
        return _SENDERS_BY_ARM[semantic_arm]
    except KeyError as exc:
        raise ValueError(f"unregistered Gemma text arm {semantic_arm!r}") from exc


def text_registration() -> dict[str, object]:
    """Return the registration payload the text fingerprint is taken over."""
    return {
        "schema": TEXT_REPORT_SCHEMA,
        "senders": [sender.to_dict() for sender in TEXT_SENDERS],
        "workers_per_item": FANOUT_M3.workers_per_item,
        "worker_prompt_tokens": GEMMA_WORKER_PROMPT_TOKENS,
        "report_style": "natural",
        "report_enable_thinking": GEMMA.decode.enable_thinking,
        "report_ceiling": FANOUTQA_NATURAL_DEV50.report_ceiling,
        "report_seed_tags": list(GEMMA_REPORT_SEED_TAGS),
        "report_seed_rule": "crc32(20260811|qid|report|worker|tag)",
        "report_redraw_scope": "all-workers",
        "sampling_profile": GEMMA.decode.profile_id,
        "sampling_fingerprint": GEMMA.decode.identity_hash,
        "sampling": {
            "temperature": GEMMA.decode.temperature,
            "top_p": GEMMA.decode.top_p,
            "top_k": GEMMA.decode.top_k,
            "presence_penalty": GEMMA.decode.presence_penalty,
        },
        "payload_format": "handoff_channels.format_report_payload",
        "receiver_prompt": "handoff_prompts.manager_prompt",
        "receiver_checkpoint": TEXT_SENDERS[0].checkpoint,
        "receiver_revision": TEXT_SENDERS[0].revision,
    }


TEXT_REGISTRATION_FINGERPRINT = identity.fingerprint(
    text_registration(), identity.json_compact_legacy
)


class TextCompletion(Protocol):
    """Raw completion fields returned by the vLLM adapter."""

    text: str
    n_tokens: int
    token_ids: Sequence[int]
    finish_reason: str
    num_cached_tokens: int | None
    queued_ts: float | None
    scheduled_ts: float | None
    first_token_ts: float | None


class TextEngine(Protocol):
    """The vLLM token-id decode seam the resident lifecycle drives."""

    def decode_token_ids_full(
        self,
        prompts: Sequence[Sequence[int]],
        sampling: Sampling,
        *,
        seeds: Sequence[int],
    ) -> Sequence[TextCompletion]:
        """Decode pre-tokenized prompts under shared sampling and explicit seeds."""
        ...


def report_seed(
    qid: str,
    worker: int,
    tag: str,
    *,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> int:
    """Return one registered report seed."""
    seeds = profile.report_seeds(qid, tag)
    if worker not in range(len(seeds)):
        raise ValueError("Gemma report worker is outside M=3")
    return seeds[worker]


def bad_words(tokenizer: Any, sender: TextSenderSpec) -> tuple[str, ...]:
    """Resolve checkpoint suppression IDs through their pinned tokenizer."""
    words: list[str] = []
    for token in sender.suppressed_token_ids:
        value = str(tokenizer.decode([token]))
        encoded = [int(row) for row in tokenizer(value, add_special_tokens=False)["input_ids"]]
        if encoded != [token]:
            raise RuntimeError(f"suppressed token {token} does not round-trip")
        words.append(value)
    return tuple(words)


def report_sampling(tokenizer: Any, sender: TextSenderSpec) -> Sampling:
    """Build Gemma-native sampling, including 12B token suppression."""
    top_k = GEMMA.decode.top_k
    if top_k is None:
        raise RuntimeError("Gemma family-native decoding requires top-k 64")
    return Sampling(
        temperature=GEMMA.decode.temperature,
        top_p=GEMMA.decode.top_p,
        top_k=top_k,
        presence_penalty=GEMMA.decode.presence_penalty,
        max_tokens=FANOUTQA_NATURAL_DEV50.report_ceiling,
        stop_token_ids=tuple(sorted(GEMMA_STOP_TOKEN_IDS)),
        bad_words=bad_words(tokenizer, sender),
    )


__all__ = (
    "GEMMA_REPORT_SEED_TAGS",
    "GEMMA_TEXT_ARMS",
    "REPORT_BACKEND",
    "REPORT_DECODE_PATH",
    "TEXT_REGISTRATION_FINGERPRINT",
    "TEXT_REPORT_BANK_SCHEMA",
    "TEXT_REPORT_SCHEMA",
    "TEXT_SENDERS",
    "Sampling",
    "TextCompletion",
    "TextEngine",
    "TextSenderSpec",
    "bad_words",
    "registered_text_senders",
    "report_sampling",
    "report_seed",
    "text_registration",
    "text_sender_spec",
)
