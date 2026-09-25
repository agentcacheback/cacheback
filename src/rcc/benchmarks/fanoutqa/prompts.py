"""The FanOutQA worker and receiver prompt bodies, for any family's chat template.

A worker prompt frames that worker's private evidence with one report style's
brief; the receiver prompt carries the question and any visible reports.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

_COMPACT_BRIEF = """You are one evidence worker. Inspect only your private
evidence. Preserve names, numbers, relationships, and uncertainty that could help
the coordinator answer the question. Return a compact evidence report, not a final
answer."""
# The line wrapping below is part of the panel identity: every prepared panel
# tokenizes these bytes, so reflowing the string shifts every worker prompt.
_NATURAL_BRIEF = """You are one reasoning worker. Inspect only your
private evidence. Work through every relevant name, number, relationship, and
uncertainty that could help the coordinator answer the question. Return a complete
reasoning handoff, including intermediate reasoning and unresolved alternatives.
Do not claim access to any other worker's evidence."""
_REPORT_STYLES = {
    "compact": (_COMPACT_BRIEF, "Return the compact report now."),
    "natural": (_NATURAL_BRIEF, "Return the complete reasoning handoff now."),
}


def report_style_parts(report_style: str) -> tuple[str, str]:
    """Return one report style's worker brief and closing instruction."""
    try:
        return _REPORT_STYLES[report_style]
    except KeyError as exc:
        raise ValueError(f"unknown report style {report_style!r}") from exc


def render_chat(
    tokenizer: Any,
    body: str,
    *,
    enable_thinking: bool,
    system_prompt: str | None = None,
    assistant_prefill: str | None = None,
    low_effort: bool = False,
) -> str:
    """Render a chat through a family codec that supports the thinking flag.

    A family whose template has no thinking flag switches it on with a system message instead;
    the flag is passed anyway and ignored. ``assistant_prefill`` follows the generation prompt.
    """
    messages = [{"role": "user", "content": body}]
    if system_prompt is not None:
        messages = [{"role": "system", "content": system_prompt}, *messages]
    # Passed only when it is on, so a family that does not take it renders
    # through the same call as before.
    effort = {"low_effort": True} if low_effort else {}
    try:
        rendered = str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                **effort,
            )
        )
    except TypeError as exc:
        raise RuntimeError(
            "registered family chat codec does not accept the required thinking flag"
        ) from exc
    return rendered + (assistant_prefill or "")


def worker_body_parts(question: str, report_style: str) -> tuple[str, str]:
    """Return the two halves of the worker body, around its private evidence."""
    brief, instruction = report_style_parts(report_style)
    return f"{brief}\n\nPrivate evidence:\n", f"\n\nQuestion: {question}\n\n{instruction}"


def worker_prompt(
    tokenizer: Any,
    question: str,
    evidence_ids: Sequence[int],
    *,
    report_style: str,
    enable_thinking: bool,
    system_prompt: str | None = None,
    assistant_prefill: str | None = None,
    low_effort: bool = False,
) -> str:
    """Render one evidence-bearing worker prompt."""
    prefix, suffix = worker_body_parts(question, report_style)
    evidence = tokenizer.decode(
        tuple(int(token) for token in evidence_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    body = prefix + evidence + suffix
    return render_chat(
        tokenizer,
        body,
        enable_thinking=enable_thinking,
        system_prompt=system_prompt,
        assistant_prefill=assistant_prefill,
        low_effort=low_effort,
    )


def manager_prompt(
    tokenizer: Any,
    question: str,
    report_payload: str = "",
    *,
    enable_thinking: bool,
    system_prompt: str | None = None,
    assistant_prefill: str | None = None,
    low_effort: bool = False,
) -> str:
    """Render the receiver prompt with an optional visible-report handoff."""
    reports = f"\n\nWorker reports:\n{report_payload}" if report_payload else ""
    body = (
        "You are the coordinator. Answer from the supplied handoff and do not "
        "invent missing evidence."
        f"{reports}\n\nQuestion: {question}\n\n"
        "Give the answer directly and include every requested leaf."
    )
    return render_chat(
        tokenizer,
        body,
        enable_thinking=enable_thinking,
        system_prompt=system_prompt,
        assistant_prefill=assistant_prefill,
        low_effort=low_effort,
    )


def question_token_ids(tokenizer: Any, rendered_prompt: str, question: str) -> tuple[int, ...]:
    """Return the question's token rows inside a rendered prompt."""
    char_start = rendered_prompt.rfind(question)
    if char_start < 0:
        raise ValueError("question text is absent from the rendered manager prompt")
    char_end = char_start + len(question)
    encoded = tokenizer(
        rendered_prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = [int(token) for token in encoded["input_ids"]]
    offsets = encoded.get("offset_mapping")
    if offsets is not None:
        positions = [
            index
            for index, (start, end) in enumerate(offsets)
            if int(end) > char_start and int(start) < char_end
        ]
        if positions:
            return tuple(ids[positions[0] : positions[-1] + 1])
    for candidate in (question, " " + question, question + "\n", " " + question + "\n"):
        needle = [
            int(token) for token in tokenizer(candidate, add_special_tokens=False)["input_ids"]
        ]
        for start in range(len(ids) - len(needle) + 1):
            if ids[start : start + len(needle)] == needle:
                return tuple(needle)
    raise ValueError("tokenizer could not locate question rows in the rendered prompt")


__all__ = (
    "manager_prompt",
    "question_token_ids",
    "render_chat",
    "report_style_parts",
    "worker_prompt",
)
