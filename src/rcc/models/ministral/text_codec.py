"""Pinned mistral-common Tekken prompt and visible-report codec.

It encodes the worker, manager, and warmup turns and reads the visible content
back out of a draw's token ids; the Ministral text and receiver paths call it.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50, arms_with_full
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.ministral import MINISTRAL
from rcc.run.fleet.latency import WARMUP_BODY

MINISTRAL_REPORT_SEED_TAGS = FANOUTQA_NATURAL_DEV50.sample_tags
MINISTRAL_TEXT_ARMS = tuple(
    arm.arm_id for arm in FANOUTQA_NATURAL_DEV50.arms if arm.channel == "text"
)
MINISTRAL_STOP_TOKEN_IDS = (2,)
MINISTRAL_THINK_OPEN_ID = 34
MINISTRAL_THINK_CLOSE_ID = 35
MINISTRAL_SPECIAL_TOKEN_COUNT = 1_000
MINISTRAL_MAX_MODEL_LEN = FANOUTQA_NATURAL_DEV50.max_model_len

_BOS_TOKEN_ID = 1
_PAD_TOKEN_ID = 11
_SYSTEM_OPEN_ID = 17
_SYSTEM_CLOSE_ID = 18
_INSTRUCTION_OPEN_ID = 3
_INSTRUCTION_CLOSE_ID = 4
_TEKKEN_VERSION = "v13"
_TEKKEN_VOCAB_SIZE = 131_072
_MISTRAL_COMMON_VERSION = "1.11.7"
_PINNED_FILE_DIGESTS = {
    "tekken.json": "e29d19ea32eb7e26e6c0572d57cb7f9eca0f4420e0e0fe6ae1cf3be94da1c0d6",
    "chat_template.jinja": "6b5044075f09f4daa57beebe2d989d9fbe67dea351f220e1a84e19ddc893f2c2",
    "SYSTEM_PROMPT.txt": "aba1efaff0bdc73f4a864139e2c07dce8fc1e3df7b5d5c2b2623df6816697156",
    "generation_config.json": "e0923390059f84a9180b00e5501778acc45ea9856cd7f2fd68208b360927c677",
}
MINISTRAL_TEXT_CODEC_FILES = tuple(_PINNED_FILE_DIGESTS)
#: Appended to the pinned system prompt on the runtime-signed manager and
#: judger turns. The model can end a turn inside an unclosed [THINK] block, and
#: this sentence asks it to close the block.
SYSTEM_PROMPT_ADDENDUM = (
    "It is imperative to close the [THINK] tag with a [/THINK] once you are "
    "ready to present the answer to the user."
)
_ADDENDUM_SUFFIX = f"\n{SYSTEM_PROMPT_ADDENDUM}\n"
_VERIFIED_TOKENIZERS: dict[
    int,
    tuple[weakref.ReferenceType[Any], tuple[tuple[str, str], ...], str, str],
] = {}
_NATURAL_WORKER_BRIEF = """You are one reasoning worker. Inspect only your
private evidence. Work through every relevant name, number, relationship, and
uncertainty that could help the coordinator answer the question. Return a complete
reasoning handoff, including intermediate reasoning and unresolved alternatives.
Do not claim access to any other worker's evidence."""
_NATURAL_WORKER_INSTRUCTION = "Return the complete reasoning handoff now."
_MANAGER_BRIEF = "You are the coordinator. Answer from the supplied handoff; invent no evidence."


@dataclass(frozen=True)
class _TagScan:
    """One draw's thinking delimiters, counted in both Tekken spellings."""

    reserved_opens: int
    literal_opened: bool
    reserved_closes: int
    literal_closes: int

    @property
    def opened(self) -> bool:
        """Whether exactly one block opened: one reserved id, or a leading spelling."""
        return self.reserved_opens == 1 or (self.reserved_opens == 0 and self.literal_opened)

    @property
    def closers(self) -> int:
        """Closers in either spelling, wherever they appear."""
        return self.reserved_closes + self.literal_closes

    @property
    def blockless(self) -> bool:
        """Whether the draw carries no delimiter at all: no opener, no closer."""
        return self.reserved_opens == 0 and not self.literal_opened and self.closers == 0


@dataclass(frozen=True)
class MinistralTextCodec:
    """A codec whose constructor rechecks every pinned identity."""

    tokenizer: Any
    system_prompt: str
    checkpoint: str
    revision: str
    file_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        """Refuse an unregistered sender or an incomplete or spoofed file digest."""
        if (self.checkpoint, self.revision) != _registered_sender_identity(self.checkpoint):
            raise ValueError("Ministral codec checkpoint/revision is not a registered text sender")
        if self.file_sha256 != tuple(_PINNED_FILE_DIGESTS.items()):
            raise ValueError("Ministral codec does not carry the exact pinned file digest roster")
        if not self.system_prompt.endswith(_ADDENDUM_SUFFIX):
            raise ValueError("Ministral codec system prompt lacks the registered reinforcement")
        base_prompt = self.system_prompt[: -len(_ADDENDUM_SUFFIX)]
        observed_system = hashlib.sha256(base_prompt.encode()).hexdigest()
        if observed_system != _PINNED_FILE_DIGESTS["SYSTEM_PROMPT.txt"]:
            raise ValueError("Ministral codec system prompt bytes differ from the pin")
        _require_tokenizer_provenance(
            self.tokenizer,
            self.file_sha256,
            self.checkpoint,
            self.revision,
        )

    def require_verified(self) -> None:
        """Recheck the pinned identity digest at the result-generation boundary."""
        self.__post_init__()

    def encode_warmup_prompt(self) -> tuple[int, ...]:
        """Encode the discarded warmup turn through this sender's own codec.

        It carries no item identity but keeps the registered Reasoning framing,
        so the engine is warmed on the shape it will serve.
        """
        return self._encode_reasoning_prompt(WARMUP_BODY)

    def encode_worker_prompt(self, question: str, evidence_ids: Sequence[int]) -> tuple[int, ...]:
        """Encode the natural worker turn through mistral-common, never HF fallback.

        A worker turn carries the pinned file prompt alone; the reinforcement
        rides only the manager and judger turns.
        """
        if not question or not evidence_ids:
            raise ValueError("Ministral worker prompt requires question and private evidence")
        evidence = str(self.tokenizer.decode([int(token) for token in evidence_ids]))
        body = (
            f"{_NATURAL_WORKER_BRIEF}\n\nPrivate evidence:\n{evidence}\n\n"
            f"Question: {question}\n\n{_NATURAL_WORKER_INSTRUCTION}"
        )
        return self._encode_reasoning_prompt(body, system_prompt=self.base_system_prompt)

    def encode_manager_prompt(
        self,
        *,
        qid: str,
        semantic_arm: str,
        source_identity: str,
        question: str,
        reports: Sequence[str] = (),
        report_failed: bool = False,
        profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
    ) -> dict[str, object]:
        """Encode and sign one real Tekken coordinator turn for the 14B receiver."""
        self.require_verified()
        if (self.checkpoint, self.revision) != (MINISTRAL.checkpoint, MINISTRAL.revision):
            raise RuntimeError("Ministral manager prompts require the registered 14B codec")
        arms = {arm.arm_id: arm for arm in arms_with_full(profile)}
        arm = arms.get(semantic_arm)
        normalized = tuple(report.strip() for report in reports)
        if (
            qid not in profile.question_ids
            or source_identity != profile.source_logical_fingerprint
            or not question
        ):
            raise ValueError("Ministral manager prompt differs from the sealed source panel")
        # A quarantined text cell serves the base turn with no payload; the
        # flag is legal only on a text arm and only with zero reports.
        if arm is None or (report_failed and (arm.channel != "text" or normalized)):
            raise ValueError("Ministral manager report handoff differs from the arm channel")
        if not report_failed and (arm.channel == "text") != bool(normalized):
            raise ValueError("Ministral manager report handoff differs from the arm channel")
        if normalized and (len(normalized) != 3 or not all(normalized)):
            raise ValueError("Ministral text manager requires three substantive reports")
        handoff = "\n\n".join(
            f"[worker {worker} report]\n{report}" for worker, report in enumerate(normalized)
        )
        report_section = f"\n\nWorker reports:\n{handoff}" if handoff else ""
        body = (
            f"{_MANAGER_BRIEF}{report_section}\n\nQuestion: {question}\n\n"
            "Give the answer directly and include every requested leaf."
        )
        tokens = self._encode_reasoning_prompt(body)
        if len(tokens) + FANOUTQA_NATURAL_DEV50.answer_ceiling > MINISTRAL_MAX_MODEL_LEN:
            raise RuntimeError("Ministral manager prompt exceeds receiver admission context")
        identity: dict[str, object] = {
            "schema": "ministral3-manager-prompt-v1",
            "qid": qid,
            "semantic_arm": semantic_arm,
            "source_identity": source_identity,
            "receiver_checkpoint": self.checkpoint,
            "receiver_revision": self.revision,
            "report_failed": bool(report_failed),
            "report_sha256": hashlib.sha256(handoff.encode()).hexdigest(),
            "prompt_token_ids": list(tokens),
            "prompt_sha256": prompt_token_sha256(tokens),
        }
        identity["fingerprint"] = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return identity

    @property
    def base_system_prompt(self) -> str:
        """Return the pinned file prompt without the runtime reinforcement."""
        return self.system_prompt[: -len(_ADDENDUM_SUFFIX)]

    def _encode_reasoning_prompt(
        self, body: str, *, system_prompt: str | None = None
    ) -> tuple[int, ...]:
        """Encode one system-plus-user Reasoning turn through mistral-common."""
        from mistral_common.protocol.instruct.request import ChatCompletionRequest

        request_class: Any = ChatCompletionRequest
        request = request_class.from_openai(
            messages=[
                {"role": "system", "content": system_prompt or self.system_prompt},
                {"role": "user", "content": body},
            ]
        )
        encoded = self.tokenizer.encode_chat_completion(request)
        tokens = tuple(int(token) for token in encoded.tokens)
        required = (
            _BOS_TOKEN_ID,
            _SYSTEM_OPEN_ID,
            _SYSTEM_CLOSE_ID,
            _INSTRUCTION_OPEN_ID,
            _INSTRUCTION_CLOSE_ID,
        )
        try:
            positions = tuple(tokens.index(token) for token in required)
        except ValueError as exc:
            raise RuntimeError("mistral-common omitted a registered prompt control token") from exc
        if positions != tuple(sorted(positions)):
            raise RuntimeError("mistral-common emitted the Reasoning turn out of order")
        return tokens

    def _scan(self, tokens: Sequence[int]) -> _TagScan:
        """Read one draw's delimiters in both spellings.

        Tekken carries them as reserved ids and as ordinary text: an opener
        counts at the reserved id or at the draw's start, a closer anywhere.
        """
        values = list(tokens)
        inner: Any = self.tokenizer.instruct_tokenizer.tokenizer
        ordinary_runs: list[list[int]] = []
        for token in values:
            if token < MINISTRAL_SPECIAL_TOKEN_COUNT:
                if ordinary_runs and ordinary_runs[-1]:
                    ordinary_runs.append([])
            else:
                if not ordinary_runs:
                    ordinary_runs.append([])
                ordinary_runs[-1].append(token)
        ordinary = tuple(str(inner.decode(run)) for run in ordinary_runs if run)
        leading: list[int] = []
        for token in values:
            if token < MINISTRAL_SPECIAL_TOKEN_COUNT:
                break
            leading.append(token)
        literal_opened = bool(leading) and (
            str(inner.decode(leading)).lstrip().startswith("[THINK]")
        )
        return _TagScan(
            reserved_opens=values.count(MINISTRAL_THINK_OPEN_ID),
            literal_opened=literal_opened,
            reserved_closes=values.count(MINISTRAL_THINK_CLOSE_ID),
            literal_closes=sum(text.count("[/THINK]") for text in ordinary),
        )

    def thinking_is_unclosed(self, raw_tokens: Sequence[int]) -> bool:
        """Return whether a draw ended inside a block it opened but never closed.

        This triggers closer injection: exactly one opened block and no closer
        in either spelling. Two opened blocks go to the redraw ladder instead.
        """
        scan = self._scan(stop_trimmed(raw_tokens))
        return scan.opened and scan.closers == 0

    def _whole(self, tokens: Sequence[int]) -> tuple[tuple[int, ...], str, bool]:
        """Read an opened, never closed draw whole; its closure flag stays False."""
        inner: Any = self.tokenizer.instruct_tokenizer.tokenizer
        visible = tuple(token for token in tokens if token >= MINISTRAL_SPECIAL_TOKEN_COUNT)
        text = str(inner.decode(list(visible))).strip()
        return (visible, text, False) if visible and text else ((), "", False)

    def _after_last_closer(self, tokens: Sequence[int]) -> tuple[tuple[int, ...], str, bool]:
        """Read the ordinary content after the last closer of either spelling."""
        inner: Any = self.tokenizer.instruct_tokenizer.tokenizer
        values = list(tokens)
        last_reserved = max(
            (index for index, token in enumerate(values) if token == MINISTRAL_THINK_CLOSE_ID),
            default=-1,
        )
        tail = [
            token for token in values[last_reserved + 1 :] if token >= MINISTRAL_SPECIAL_TOKEN_COUNT
        ]
        text = str(inner.decode(tail))
        literal = text.rfind("[/THINK]")
        if literal < 0:
            return (tuple(tail), text.strip(), True) if tail and text.strip() else ((), "", False)
        char_end = literal + len("[/THINK]")
        lo, hi = 0, len(tail)
        while lo < hi:
            middle = (lo + hi) // 2
            if len(str(inner.decode(tail[:middle]))) < char_end:
                lo = middle + 1
            else:
                hi = middle
        visible = tuple(tail[lo:])
        content = text[char_end:].strip()
        return (visible, content, True) if visible and content else ((), "", False)

    def _thought_split(
        self, raw_tokens: Sequence[int], *, ended: bool = False
    ) -> tuple[tuple[int, ...], str, bool]:
        """Split one opened block at one unambiguous closer, either spelling.

        Exactly one closer of either spelling ends the block, and the content
        follows it. See docs/running.md, Ministral thought-block reading.
        """
        inner: Any = self.tokenizer.instruct_tokenizer.tokenizer
        tokens = list(stop_trimmed(raw_tokens))
        decoded = str(inner.decode(tokens))
        scan = self._scan(tokens)
        if scan.blockless:
            visible = tuple(token for token in tokens if token >= MINISTRAL_SPECIAL_TOKEN_COUNT)
            return (visible, decoded.strip(), True) if visible else ((), "", False)
        if ended and scan.closers:
            return self._after_last_closer(tokens)
        if ended:
            return self._whole(tokens)
        if not scan.opened or scan.closers != 1:
            return (), "", False
        # A literal opener sits at the very start by construction, so its
        # position is -1 in token space and the leading character in text.
        opened_at = tokens.index(MINISTRAL_THINK_OPEN_ID) if scan.reserved_opens == 1 else -1
        if scan.reserved_closes:
            closed_at = tokens.index(MINISTRAL_THINK_CLOSE_ID)
            if opened_at >= closed_at:
                return (), "", False
            boundary = closed_at + 1
            text = str(inner.decode(tokens[boundary:])).strip()
        else:
            close_char = decoded.find("[/THINK]")
            open_char = (
                len(str(inner.decode(tokens[:opened_at])))
                if opened_at >= 0
                else decoded.find("[THINK]")
            )
            if close_char < open_char:
                return (), "", False
            char_end = close_char + len("[/THINK]")
            lo, hi = opened_at + 1, len(tokens)
            while lo < hi:
                middle = (lo + hi) // 2
                if len(str(inner.decode(tokens[:middle]))) < char_end:
                    lo = middle + 1
                else:
                    hi = middle
            boundary = lo
            text = decoded[char_end:].strip()
        visible = tuple(
            token for token in tokens[boundary:] if token >= MINISTRAL_SPECIAL_TOKEN_COUNT
        )
        return visible, text, True

    def visible_report(self, raw_tokens: Sequence[int]) -> tuple[tuple[int, ...], str]:
        """Reconstruct post-thought content independently from raw token IDs."""
        visible, text, closed = self._thought_split(raw_tokens)
        if not closed:
            raise RuntimeError("Ministral report does not contain one closed thought channel")
        if not visible or not text:
            raise RuntimeError("Ministral report has no substantive post-thought content")
        return visible, text

    def visible_answer(
        self, raw_tokens: Sequence[int], *, ended: bool = False
    ) -> tuple[tuple[int, ...], str, bool]:
        """Reconstruct a receiver answer; an unclosed thought channel is empty.

        An answer is never redrawn: an unclosed trace carries no answer and
        scores zero. ``ended`` admits the after-last-closer reading.
        """
        return self._thought_split(raw_tokens, ended=ended)


def load_registered_text_codec(snapshot: Path, semantic_arm: str) -> MinistralTextCodec:
    """Load and measure the real Tekken codec for one pinned sender snapshot."""
    checkpoint, revision = _sender_identity_for_arm(semantic_arm)
    if importlib.metadata.version("mistral-common") != _MISTRAL_COMMON_VERSION:
        raise RuntimeError("mistral-common version differs from the registered runtime")
    measured: list[tuple[str, str]] = []
    for name, expected in _PINNED_FILE_DIGESTS.items():
        path = snapshot / name
        if not path.is_file():
            raise RuntimeError(f"Ministral sender snapshot is missing {name}")
        observed = _sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"Ministral {name} digest drifted to {observed}")
        measured.append((name, observed))
    _validate_tekken(snapshot / "tekken.json")
    _validate_generation_config(snapshot / "generation_config.json")

    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tokenizer_class: Any = MistralTokenizer
    tokenizer = tokenizer_class.from_file(snapshot / "tekken.json")
    inner = tokenizer.instruct_tokenizer.tokenizer
    geometry = (int(inner.bos_id), int(inner.eos_id), int(inner.n_words))
    if geometry != (_BOS_TOKEN_ID, MINISTRAL_STOP_TOKEN_IDS[0], _TEKKEN_VOCAB_SIZE):
        raise RuntimeError(f"loaded Tekken geometry differs from the pin: {geometry}")
    measured_tuple = tuple(measured)
    _register_tokenizer(tokenizer, measured_tuple, checkpoint, revision)
    try:
        return MinistralTextCodec(
            tokenizer=tokenizer,
            system_prompt=(
                (snapshot / "SYSTEM_PROMPT.txt").read_text(encoding="utf-8") + _ADDENDUM_SUFFIX
            ),
            checkpoint=checkpoint,
            revision=revision,
            file_sha256=measured_tuple,
        )
    except Exception:
        _VERIFIED_TOKENIZERS.pop(id(tokenizer), None)
        raise


def stop_trimmed(raw_tokens: Sequence[int]) -> tuple[int, ...]:
    """Return the banked ids that precede the first registered stop token.

    A draw that ended its turn banks the stop id itself, and the closer
    injection replaces exactly this boundary.
    """
    tokens = [int(token) for token in raw_tokens]
    for index, token in enumerate(tokens):
        if token in MINISTRAL_STOP_TOKEN_IDS:
            return tuple(tokens[:index])
    return tuple(tokens)


def prompt_token_sha256(tokens: Sequence[int]) -> str:
    """Return the canonical digest used by prepared prompt artifacts and records."""
    return hashlib.sha256(json.dumps(list(tokens), separators=(",", ":")).encode()).hexdigest()


def _registered_sender_identity(checkpoint: str) -> tuple[str, str]:
    matches = [
        (arm.sender_checkpoint, arm.sender_revision)
        for arm in MINISTRAL.physical_arms
        if arm.semantic_arm in MINISTRAL_TEXT_ARMS and arm.sender_checkpoint == checkpoint
    ]
    if len(matches) != 1 or matches[0][0] is None or matches[0][1] is None:
        raise ValueError(f"unregistered Ministral text checkpoint {checkpoint!r}")
    return matches[0][0], matches[0][1]


def _sender_identity_for_arm(semantic_arm: str) -> tuple[str, str]:
    matches = [
        (arm.sender_checkpoint, arm.sender_revision)
        for arm in MINISTRAL.physical_arms
        if arm.semantic_arm == semantic_arm and semantic_arm in MINISTRAL_TEXT_ARMS
    ]
    if len(matches) != 1 or matches[0][0] is None or matches[0][1] is None:
        raise ValueError(f"unregistered Ministral text arm {semantic_arm!r}")
    return matches[0][0], matches[0][1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_tekken(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload["config"]
    specials = payload["special_tokens"]
    if (
        config["version"] != _TEKKEN_VERSION
        or int(config["num_vocab_tokens"]) != _TEKKEN_VOCAB_SIZE
        or len(specials) != MINISTRAL_SPECIAL_TOKEN_COUNT
    ):
        raise RuntimeError("Ministral Tekken vocabulary geometry drifted")
    ids = {str(row["token_str"]): int(row["rank"]) for row in specials}
    expected = {
        "<s>": _BOS_TOKEN_ID,
        "</s>": MINISTRAL_STOP_TOKEN_IDS[0],
        "<pad>": _PAD_TOKEN_ID,
        "[THINK]": MINISTRAL_THINK_OPEN_ID,
        "[/THINK]": MINISTRAL_THINK_CLOSE_ID,
    }
    if any(ids.get(token) != expected_id for token, expected_id in expected.items()):
        raise RuntimeError("Ministral Tekken control-token geometry drifted")


def _validate_generation_config(path: Path) -> None:
    payload: Mapping[str, object] = json.loads(path.read_text(encoding="utf-8"))
    observed = tuple(
        _required_int(payload, key) for key in ("bos_token_id", "eos_token_id", "pad_token_id")
    )
    if observed != (_BOS_TOKEN_ID, MINISTRAL_STOP_TOKEN_IDS[0], _PAD_TOKEN_ID):
        raise RuntimeError(f"Ministral generation controls drifted to {observed}")


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if type(value) is not int:
        raise RuntimeError(f"Ministral generation config lacks integer {key}")
    return value


def _register_tokenizer(
    tokenizer: Any,
    file_sha256: tuple[tuple[str, str], ...],
    checkpoint: str,
    revision: str,
) -> None:
    """Bind one measured live tokenizer object to its exact snapshot digest."""
    _require_tokenizer_implementation(tokenizer)
    object_id = id(tokenizer)

    def remove_if_current(dead_ref: weakref.ReferenceType[Any]) -> None:
        current = _VERIFIED_TOKENIZERS.get(object_id)
        if current is not None and current[0] is dead_ref:
            _VERIFIED_TOKENIZERS.pop(object_id, None)

    tokenizer_ref = weakref.ref(tokenizer, remove_if_current)
    _VERIFIED_TOKENIZERS[object_id] = (
        tokenizer_ref,
        file_sha256,
        checkpoint,
        revision,
    )


def _require_tokenizer_provenance(
    tokenizer: Any,
    file_sha256: tuple[tuple[str, str], ...],
    checkpoint: str,
    revision: str,
) -> None:
    """Require this exact object to have been loaded from this measured snapshot."""
    _require_tokenizer_implementation(tokenizer)
    registered = _VERIFIED_TOKENIZERS.get(id(tokenizer))
    if (
        registered is None
        or registered[0]() is not tokenizer
        or registered[1:] != (file_sha256, checkpoint, revision)
    ):
        raise ValueError("Ministral codec tokenizer lacks exact measured-object provenance")


def _require_tokenizer_implementation(tokenizer: Any) -> None:
    """Prove the live object is the pinned mistral-common V13 Tekken implementation."""
    if importlib.metadata.version("mistral-common") != _MISTRAL_COMMON_VERSION:
        raise ValueError("Ministral codec mistral-common version differs from the pin")
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    if type(tokenizer) is not MistralTokenizer:
        raise ValueError("Ministral codec tokenizer is not the pinned MistralTokenizer")
    tokenizer_any = cast(Any, tokenizer)
    instruct: Any = tokenizer_any.instruct_tokenizer
    inner = instruct.tokenizer
    implementations = (
        (type(instruct).__module__, type(instruct).__qualname__),
        (type(inner).__module__, type(inner).__qualname__),
    )
    if implementations != (
        ("mistral_common.tokens.tokenizers.instruct", "InstructTokenizerV13"),
        ("mistral_common.tokens.tokenizers.tekken", "Tekkenizer"),
    ):
        raise ValueError("Ministral codec tokenizer implementation differs from V13 Tekken")
    geometry = (int(inner.bos_id), int(inner.eos_id), int(inner.n_words))
    if geometry != (_BOS_TOKEN_ID, MINISTRAL_STOP_TOKEN_IDS[0], _TEKKEN_VOCAB_SIZE):
        raise ValueError(f"Ministral codec live Tekken geometry drifted to {geometry}")


__all__ = (
    "MINISTRAL_MAX_MODEL_LEN",
    "MINISTRAL_REPORT_SEED_TAGS",
    "MINISTRAL_STOP_TOKEN_IDS",
    "MINISTRAL_TEXT_ARMS",
    "MINISTRAL_TEXT_CODEC_FILES",
    "MINISTRAL_THINK_CLOSE_ID",
    "MINISTRAL_THINK_OPEN_ID",
    "MinistralTextCodec",
    "load_registered_text_codec",
    "prompt_token_sha256",
    "stop_trimmed",
)
