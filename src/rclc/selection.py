"""Opt-in source-position records and a standalone highlighted selection view."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from html import escape
from itertools import groupby
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

import torch

if TYPE_CHECKING:
    from rclc.transport import SenderState

Kind = Literal["token", "latent", "continuous", "unknown"]


class Span(TypedDict):
    """A contiguous source range with matching selection status and row kind."""

    start: int
    stop: int
    kept: bool
    kind: Kind
    text: str | None


@dataclass(frozen=True)
class Selection:
    """Local debug metadata, separate from the tensor payload and its byte count."""

    source_positions: int
    budget: int
    indices: tuple[int, ...]
    spans: tuple[Span, ...]
    added_latent_positions: int

    def inspect(self) -> dict[str, Any]:
        """Return a detached, JSON-serializable copy of the recorded selection."""
        return asdict(self)


def _kind(position: int, length: int, latent_steps: int, ids: list[int] | None) -> Kind:
    if position >= length - latent_steps:
        return "latent"
    if ids is None:
        return "unknown"
    return "continuous" if ids[position] == -1 else "token"


def _decode_span(tokenizer: Any, ids: list[int], start: int, stop: int) -> str:
    options = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    text = cast(str, tokenizer.decode(ids[start:stop], **options))
    if start and ids[start - 1] >= 0:
        prefix = cast(str, tokenizer.decode(ids[start - 1 : start], **options))
        combined = cast(str, tokenizer.decode(ids[start - 1 : stop], **options))
        if combined.startswith(prefix):
            text = combined[len(prefix) :]
    return text


def record_selection(
    state: SenderState, indices: torch.Tensor, budget: int, added: int
) -> Selection:
    """Snapshot source spans without retaining model tensors, caches or the tokenizer."""
    length = len(state.input_embeds)
    kept = tuple(cast(list[int], cast(Any, indices).tolist()))
    selected = set(kept)
    ids = None if state.token_ids is None else cast(list[int], cast(Any, state.token_ids).tolist())
    groups = groupby(
        range(length),
        key=lambda i: (i in selected, _kind(i, length, state.latent_steps, ids)),
    )
    spans: list[Span] = []
    for (is_kept, kind), positions in groups:
        run = list(positions)
        start, stop = run[0], run[-1] + 1
        text = None
        if kind == "token" and ids is not None and state.tokenizer is not None:
            text = _decode_span(state.tokenizer, ids, start, stop)
        spans.append(Span(start=start, stop=stop, kept=is_kept, kind=kind, text=text))
    return Selection(length, budget, kept, tuple(spans), added)


def _span_html(span: Span) -> str:
    status = "kept" if span["kept"] else "omitted"
    title = f"{status}: positions [{span['start']}, {span['stop']})"
    text = span["text"]
    if text is None:
        text = f" [{span['kind']}: {span['stop'] - span['start']} positions] "
    tag = "mark" if span["kept"] else "span"
    return f'<{tag} title="{escape(title, quote=True)}">{escape(text)}</{tag}>'


def selection_html(selections: Sequence[Selection | None], request: str) -> str:
    """Render recorded sources as escaped HTML, with retained spans highlighted in green."""
    sections: list[str] = []
    for index, selection in enumerate(selections):
        if selection is None:
            raise ValueError("selection was not recorded; transfer with record_selection=True")
        added = selection.added_latent_positions
        summary = (
            f"Sender {index + 1}: {len(selection.indices)} / {selection.source_positions} "
            f"source positions kept; budget {selection.budget}"
        )
        if added:
            summary += f"; {added} latent positions added after selection"
        spans = "".join(_span_html(span) for span in selection.spans)
        sections.append(f"<section><h2>{summary}</h2><pre>{spans}</pre></section>")
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Transfer selection</title><style>"
        ".rclc-selection{font:16px/1.6 system-ui,sans-serif;max-width:900px;"
        "margin:32px auto;padding:20px;color:#202620;background:#fff}"
        ".rclc-selection h2{font-size:16px}.rclc-selection p{overflow-wrap:anywhere}"
        ".rclc-selection pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}"
        ".rclc-selection mark{background:#c9f0cb;color:#18391c;text-decoration:underline}"
        '.rclc-selection section{margin-top:28px}</style><body><div class="rclc-selection">'
        f"<p><strong>Request:</strong> {escape(request)}</p>"
        "<p>Green underlined spans were kept. Hover for source positions.</p>"
        + "".join(sections)
        + "</div></body></html>"
    )
