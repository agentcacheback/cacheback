"""Prompt construction shared by the handoff benchmarks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rcc.models.gemma import GEMMA

_WORKER_BRIEF = """You are one evidence worker. Inspect only your private
evidence. Preserve names, numbers, relationships, and uncertainty that could help
the coordinator answer the question. Return a compact evidence report, not a final
answer."""
_NATURAL_WORKER_BRIEF = """You are one reasoning worker. Inspect only your
private evidence. Work through every relevant name, number, relationship, and
uncertainty that could help the coordinator answer the question. Return a complete
reasoning handoff, including intermediate reasoning and unresolved alternatives.
Do not claim access to any other worker's evidence."""
_REPORT_STYLES = {
    "compact": (_WORKER_BRIEF, "Return the compact report now."),
    "natural": (_NATURAL_WORKER_BRIEF, "Return the complete reasoning handoff now."),
}


def report_style_parts(report_style: str) -> tuple[str, str]:
    """Return one report style's worker brief and its closing instruction."""
    try:
        return _REPORT_STYLES[report_style]
    except KeyError as exc:
        raise ValueError(f"unknown report style {report_style!r}") from exc


def render_chat(tokenizer: Any, body: str) -> str:
    """Render a Gemma chat prompt under the registered thinking regime."""
    messages = [{"role": "user", "content": body}]
    try:
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=GEMMA.decode.enable_thinking,
            )
        )
    except TypeError as exc:
        raise RuntimeError(
            "registered Gemma chat codec does not accept the required thinking flag"
        ) from exc


def worker_prompt(
    tokenizer: Any,
    question: str,
    evidence_ids: Sequence[int],
    *,
    report_style: str = "compact",
) -> str:
    """Render the evidence-bearing prompt shared by text and latent workers."""
    brief, instruction = report_style_parts(report_style)
    evidence = tokenizer.decode(evidence_ids, skip_special_tokens=False)
    body = f"{brief}\n\nPrivate evidence:\n{evidence}\n\nQuestion: {question}\n\n{instruction}"
    return render_chat(tokenizer, body)


def manager_prompt(
    tokenizer: Any,
    question: str,
    report_payload: str = "",
) -> str:
    """Render a coordinator prompt with an optional text handoff."""
    reports = f"\n\nWorker reports:\n{report_payload}" if report_payload else ""
    body = (
        "You are the coordinator. Answer from the supplied handoff and do not "
        "invent missing evidence."
        f"{reports}\n\nQuestion: {question}\n\n"
        "Give the answer directly and include every requested leaf."
    )
    return render_chat(tokenizer, body)


def question_token_ids(tokenizer: Any, rendered_prompt: str, question: str) -> list[int]:
    """Extract the question's exact in-context token rows."""
    char_start = rendered_prompt.rfind(question)
    if char_start < 0:
        raise ValueError("question text is absent from the rendered manager prompt")
    char_end = char_start + len(question)
    encoded = tokenizer(
        rendered_prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = list(encoded["input_ids"])
    offsets = encoded.get("offset_mapping")
    if offsets is not None:
        positions = [
            index
            for index, (start, end) in enumerate(offsets)
            if int(end) > char_start and int(start) < char_end
        ]
        if positions:
            return [int(token_id) for token_id in ids[positions[0] : positions[-1] + 1]]
    for candidate in (question, " " + question, question + "\n", " " + question + "\n"):
        needle = tokenizer(candidate, add_special_tokens=False)["input_ids"]
        for start in range(len(ids) - len(needle) + 1):
            if ids[start : start + len(needle)] == needle:
                return [int(token_id) for token_id in needle]
    raise ValueError("tokenizer could not locate question rows in the rendered prompt")
