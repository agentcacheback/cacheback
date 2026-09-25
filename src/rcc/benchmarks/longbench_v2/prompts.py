"""The LongBench v2 chain prompt bodies: the floor, the hops, the rewrites, the receiver.

The no-context floor is the official ``0shot_no_context`` template byte for byte, and
every body renders from the question and the choices alone over fixed placeholders.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from rcc.benchmarks.fanoutqa.prompts import question_token_ids, render_chat
from rcc.models.route import RouteFamily

#: The official question, choices, and format lines every answer body ends with.
_ANSWER_TAIL = (
    "What is the correct answer to this question: $Q$\n"
    "Choices:\n(A) $C_A$\n(B) $C_B$\n(C) $C_C$\n(D) $C_D$\n\n"
    'Format your response as follows: "The correct answer is (insert answer here)".'
)
#: The official ``0shot`` body, the reference the answer bodies derive from.
ZERO_SHOT_TEMPLATE = (
    "Please read the following text and answer the question below.\n\n"
    "<text>\n$DOC$\n</text>\n\n" + _ANSWER_TAIL
)
#: The receiver brief: the answer comes from what the chain kept. The text
#: channel reads the notes inside the official text block; the latent channel
#: reads rows that precede the turn, so its body has no text block.
ANSWER_TEMPLATE = (
    "The notes a chain of readers kept from the document are below. Answer from "
    "them, not from what you already know.\n\n"
    "<text>\n$DOC$\n</text>\n\n" + _ANSWER_TAIL
)
LATENT_ANSWER_TEMPLATE = (
    "The passages a chain of readers kept from the document precede this message. "
    "Answer from them, not from what you already know.\n\n" + _ANSWER_TAIL
)
ZERO_SHOT_TEMPLATE_SHA256 = "68a162252bc9ff71d5d7abca3d69bb31aac3c35f832d657a2866f2018b8a6950"
NO_CONTEXT_TEMPLATE = (
    "What is the correct answer to this question: $Q$\n"
    "Choices:\n(A) $C_A$\n(B) $C_B$\n(C) $C_C$\n(D) $C_D$\n\n"
    "What is the single, most likely answer choice? Format your response as follows: "
    '"The correct answer is (insert answer here)".'
)
NO_CONTEXT_TEMPLATE_SHA256 = "125fb44183af01ffc623df19392b80a287db62af3e2a2805e5b2ff9c7b2bee30"
#: The worker brief a hop reads: the parts are read in order, the rows an
#: earlier hop kept precede this hop's prompt, and the worker notes what helps.
HOP_BRIEF = (
    "You are one worker in a chain reading a long document in order. This is part "
    "{hop} of {hops}{carried}. Note anything in your part that helps answer the "
    "question below. Do not answer it."
)
HOP_CARRIED = "; what the earlier parts kept precedes this message"
REWRITE_BRIEF = (
    "You are one reading worker in a chain of {hops}. You receive the running notes "
    "from the earlier parts of a text and one new part. Rewrite the notes so that they "
    "hold everything read so far that could help answer the question below: names, "
    "numbers, events, relationships, and open alternatives. Return only the rewritten "
    "notes, not a final answer."
)
#: The condensing brief: it names a length, forbids copying, and asks for the
#: question's own facts in the writer's words. Under the brief above a writer
#: may instead copy its part verbatim up to the notes ceiling.
REWRITE_BRIEF_CONDENSE = (
    "You are one reading worker in a chain of {hops}. You receive the running notes "
    "from the earlier parts of a text and one new part. Rewrite the notes into a compact "
    "summary of everything read so far that bears on the question below: the names, "
    "numbers, dates, events, relationships, and open alternatives it turns on, each "
    "stated once in your own words. Do not copy passages, code, tables, or data from "
    "the part; state what they say and where they sit. Drop what the question does not "
    "need. Keep the notes under {words} words. Return only the rewritten notes, not a "
    "final answer."
)
CONDENSE_WORDS = 8_000
#: The two LongBench prompt builders: the same official templates and hop brief,
#: and one rewrite brief each. A profile names one, and every rewrite and every
#: rewrite digest is rendered under that builder.
PROMPT_BUILDER = "longbench-v2-official-0shot-v1"
PROMPT_BUILDER_CONDENSE = "longbench-v2-official-0shot-condense-v1"
PROMPT_BUILDERS = frozenset({PROMPT_BUILDER, PROMPT_BUILDER_CONDENSE})
NO_NOTES = "(no earlier parts)"
HOPS = 4
_DOC_SLOT = "<<RCC_DOC_SLOT>>"  # the document placeholder, split on after rendering
PAYLOAD_SLOT_MARKER = _DOC_SLOT
#: Placeholders the digest helpers render the bodies over: the official
#: template's own markers, so a digest covers every fixed byte and no item's
#: own text.
_SEAL_QUESTION = "$Q$"
_SEAL_CHOICES = ("$C_A$", "$C_B$", "$C_C$", "$C_D$")
_SEAL_NOTES = "$NOTES$"


def _family_chat_kwargs(family: RouteFamily | None) -> dict[str, Any]:
    """Return one route family's chat controls, or nothing when no family is given."""
    if family is None:
        return {}
    return {
        "system_prompt": family.thinking_system_prompt,
        "assistant_prefill": family.assistant_prefill,
        "low_effort": family.low_effort,
    }


def prompt_sha256(text: str) -> str:
    """Return the sha256 hex digest of one prompt body."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate(question: str, choices: Sequence[str], hop: int | None = None) -> None:
    if not question.strip():
        raise ValueError("a prompt needs a nonempty question")
    if len(choices) != 4 or any(not choice.strip() for choice in choices):
        raise ValueError("a prompt needs exactly four nonempty choices")
    if hop is not None and not 1 <= hop <= HOPS:
        raise ValueError(f"hop must be between 1 and {HOPS}, got {hop}")


def fill_template(template: str, question: str, choices: Sequence[str], doc: str) -> str:
    """Substitute the official placeholders, stripping question and choices."""
    _validate(question, choices)
    body = template.replace("$Q$", question.strip())
    for letter, choice in zip("ABCD", choices, strict=True):
        body = body.replace(f"$C_{letter}$", choice.strip())
    # The document slot goes last, so a marker inside the notes is never
    # substituted by a later pass.
    return body.replace("$DOC$", doc)


def question_choices_text(question: str, choices: Sequence[str]) -> str:
    """Return the measurement string: the question, then the four choices.

    The geometry allowances count this string; the ``Choices:`` header and the
    rest of the official body belong to the template allowance.
    """
    _validate(question, choices)
    lines = [question.strip()]
    lines.extend(
        f"({letter}) {choice.strip()}" for letter, choice in zip("ABCD", choices, strict=True)
    )
    return "\n".join(lines)


def question_block(question: str, choices: Sequence[str]) -> str:
    """Return the question and choices in the official layout, no instructions."""
    _validate(question, choices)
    lines = [question.strip(), "Choices:"]
    lines.extend(
        f"({letter}) {choice.strip()}" for letter, choice in zip("ABCD", choices, strict=True)
    )
    return "\n".join(lines)


def answer_prompt(
    tokenizer: Any,
    question: str,
    choices: Sequence[str],
    *,
    notes: str = "",
    enable_thinking: bool,
    latent: bool = False,
    family: RouteFamily | None = None,
    payload_slot: bool = False,
) -> str:
    """Render the receiver's answer prompt: notes in the slot, or the latent body."""
    if latent and notes and not payload_slot:
        raise ValueError("the latent receiver reads rows before the turn, never notes")
    template = LATENT_ANSWER_TEMPLATE if latent else ANSWER_TEMPLATE
    if payload_slot:
        if not latent or family is None or not family.payload_in_user_turn:
            raise ValueError("payload slot requires a native latent family")
        body = template.replace("precede this message. ", f"precede this message.\n{_DOC_SLOT}\n\n")
        return render_chat(
            tokenizer,
            fill_template(body, question, choices, ""),
            enable_thinking=enable_thinking,
            **_family_chat_kwargs(family),
        )
    return render_chat(
        tokenizer,
        fill_template(template, question, choices, notes),
        enable_thinking=enable_thinking,
        **_family_chat_kwargs(family),
    )


def no_context_prompt(
    tokenizer: Any,
    question: str,
    choices: Sequence[str],
    *,
    enable_thinking: bool,
    family: RouteFamily | None = None,
) -> str:
    """Render the official no-context floor prompt."""
    return render_chat(
        tokenizer,
        fill_template(NO_CONTEXT_TEMPLATE, question, choices, ""),
        enable_thinking=enable_thinking,
        **_family_chat_kwargs(family),
    )


def hop_prompt_text(question: str, choices: Sequence[str], hop: int) -> str:
    """Return one hop's body with the document slot still open."""
    _validate(question, choices, hop)
    carried = "" if hop == 1 else HOP_CARRIED
    brief = HOP_BRIEF.format(hop=hop, hops=HOPS, carried=carried)
    return (
        brief
        + "\n\nQuestion and choices:\n"
        + question_block(question, choices)
        + "\n\nPassage:\n"
        + _DOC_SLOT
    )


def hop_prompt_ids(
    tokenizer: Any,
    question: str,
    choices: Sequence[str],
    chunk_ids: Sequence[int],
    hop: int,
    *,
    enable_thinking: bool,
    family: RouteFamily | None = None,
) -> tuple[int, ...]:
    """Return one hop's prompt ids: rendered prefix, the chunk, rendered suffix.

    The body is split at the document slot so the ledger's standalone chunk ids sit
    inside the user turn with no second tokenization, at the count the geometry prices.
    """
    if not chunk_ids:
        raise ValueError("a hop needs at least one chunk token")
    rendered = render_chat(
        tokenizer,
        hop_prompt_text(question, choices, hop),
        enable_thinking=enable_thinking,
        **_family_chat_kwargs(family),
    )
    if rendered.count(_DOC_SLOT) != 1:
        raise RuntimeError("the chat codec did not keep exactly one document slot")
    prefix, suffix = rendered.split(_DOC_SLOT)
    encode = tokenizer.encode
    ids = [int(token) for token in encode(prefix, add_special_tokens=False)]
    ids.extend(int(token) for token in chunk_ids)
    ids.extend(int(token) for token in encode(suffix, add_special_tokens=False))
    return tuple(ids)


def hop_bodies_sha256() -> str:
    """Digest every hop body over the placeholders, wrapper included."""
    return prompt_sha256(
        "\n".join(hop_prompt_text(_SEAL_QUESTION, _SEAL_CHOICES, hop) for hop in range(1, HOPS + 1))
    )


def answer_bodies_sha256() -> str:
    """Digest the text and the latent answer bodies over the placeholders."""
    return prompt_sha256(
        fill_template(ANSWER_TEMPLATE, _SEAL_QUESTION, _SEAL_CHOICES, _SEAL_NOTES)
        + "\n"
        + fill_template(LATENT_ANSWER_TEMPLATE, _SEAL_QUESTION, _SEAL_CHOICES, "")
    )


def rewrite_bodies_sha256(builder: str) -> str:
    """Digest one builder's rewrite bodies over the placeholders, hop one without notes."""
    return prompt_sha256(
        "\n".join(
            rewrite_prompt_text(
                _SEAL_QUESTION,
                _SEAL_CHOICES,
                notes="" if hop == 1 else _SEAL_NOTES,
                chunk_text=_DOC_SLOT,
                hop=hop,
                builder=builder,
            )
            for hop in range(1, HOPS + 1)
        )
    )


def prompt_token_counts(
    tokenizer: Any,
    question: str,
    choices: Sequence[str],
    *,
    enable_thinking: bool,
    builder: str,
    family: RouteFamily | None = None,
) -> dict[str, int]:
    """Count the rendered wrapper tokens the geometry allowances must cover."""

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    def wrapper(text: str) -> int:
        rendered = render_chat(
            tokenizer, text, enable_thinking=enable_thinking, **_family_chat_kwargs(family)
        )
        prefix, suffix = rendered.split(_DOC_SLOT)
        return count(prefix) + count(suffix)

    hop_wrappers: list[int] = []
    rewrite_wrappers: list[int] = []
    for hop in range(1, HOPS + 1):
        hop_wrappers.append(wrapper(hop_prompt_text(question, choices, hop)))
        rewrite_wrappers.append(
            wrapper(
                rewrite_prompt_text(
                    question,
                    choices,
                    notes="",
                    chunk_text=_DOC_SLOT,
                    hop=hop,
                    builder=builder,
                )
            )
        )
    return {
        "question_choices": count(question_choices_text(question, choices)),
        "hop_wrapper": max(hop_wrappers),
        "rewrite_wrapper": max(rewrite_wrappers),
        "answer_prompt": max(
            count(
                answer_prompt(
                    tokenizer,
                    question,
                    choices,
                    enable_thinking=enable_thinking,
                    family=family,
                )
            ),
            count(
                answer_prompt(
                    tokenizer,
                    question,
                    choices,
                    enable_thinking=enable_thinking,
                    latent=True,
                    family=family,
                )
            ),
        ),
        "no_context_prompt": count(
            no_context_prompt(
                tokenizer,
                question,
                choices,
                enable_thinking=enable_thinking,
                family=family,
            )
        ),
    }


def retention_prompt(
    tokenizer: Any,
    question: str,
    *,
    enable_thinking: bool,
    family: RouteFamily | None = None,
) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    """Render the question-only retention prompt and locate the question rows."""
    stripped = question.strip()
    if not stripped:
        raise ValueError("a retention prompt needs a nonempty question")
    text = render_chat(
        tokenizer, stripped, enable_thinking=enable_thinking, **_family_chat_kwargs(family)
    )
    ids = tuple(int(token) for token in tokenizer(text, add_special_tokens=False)["input_ids"])
    return text, ids, question_token_ids(tokenizer, text, stripped)


def rewrite_brief(builder: str) -> str:
    """Return the rewrite brief one prompt builder renders, raising for any other."""
    if builder == PROMPT_BUILDER:
        return REWRITE_BRIEF.format(hops=HOPS)
    if builder == PROMPT_BUILDER_CONDENSE:
        return REWRITE_BRIEF_CONDENSE.format(hops=HOPS, words=f"{CONDENSE_WORDS:,}")
    raise ValueError(f"no rewrite brief is registered for prompt builder {builder!r}")


def rewrite_prompt_text(
    question: str,
    choices: Sequence[str],
    *,
    notes: str,
    chunk_text: str,
    hop: int,
    builder: str,
) -> str:
    """Return one rewrite body: the builder's brief, the question, the notes, one part."""
    _validate(question, choices, hop)
    if not chunk_text:
        raise ValueError("a rewrite needs a nonempty part")
    return (
        rewrite_brief(builder)
        + "\n\nQuestion and choices:\n"
        + question_block(question, choices)
        + "\n\nNotes from the earlier parts:\n"
        + (notes if notes else NO_NOTES)
        + f"\n\nPart {hop} of {HOPS}:\n"
        + chunk_text
        + "\n\nReturn the rewritten notes now."
    )


def rewrite_prompt(
    tokenizer: Any,
    question: str,
    choices: Sequence[str],
    *,
    notes: str,
    chunk_text: str,
    hop: int,
    enable_thinking: bool,
    builder: str,
    family: RouteFamily | None = None,
) -> str:
    """Render one text-chain rewrite: the notes so far plus one new part."""
    body = rewrite_prompt_text(
        question, choices, notes=notes, chunk_text=chunk_text, hop=hop, builder=builder
    )
    return render_chat(
        tokenizer, body, enable_thinking=enable_thinking, **_family_chat_kwargs(family)
    )


__all__ = (
    "ANSWER_TEMPLATE",
    "CONDENSE_WORDS",
    "HOPS",
    "HOP_BRIEF",
    "HOP_CARRIED",
    "LATENT_ANSWER_TEMPLATE",
    "NO_CONTEXT_TEMPLATE",
    "NO_CONTEXT_TEMPLATE_SHA256",
    "NO_NOTES",
    "PAYLOAD_SLOT_MARKER",
    "PROMPT_BUILDER",
    "PROMPT_BUILDERS",
    "PROMPT_BUILDER_CONDENSE",
    "REWRITE_BRIEF",
    "REWRITE_BRIEF_CONDENSE",
    "ZERO_SHOT_TEMPLATE",
    "ZERO_SHOT_TEMPLATE_SHA256",
    "answer_bodies_sha256",
    "answer_prompt",
    "fill_template",
    "hop_bodies_sha256",
    "hop_prompt_ids",
    "hop_prompt_text",
    "no_context_prompt",
    "prompt_sha256",
    "prompt_token_counts",
    "question_block",
    "question_choices_text",
    "retention_prompt",
    "rewrite_bodies_sha256",
    "rewrite_brief",
    "rewrite_prompt",
    "rewrite_prompt_text",
)
