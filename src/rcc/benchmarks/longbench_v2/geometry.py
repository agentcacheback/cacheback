"""Chain geometry: what fits the served window under each budget law.

A hop's sequence is the carried rows plus the chunk, the wrapper allowance, and the
forty latent rows; the allowances below are ceilings measured on the panel.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rcc.benchmarks.longbench_v2.prompts import PROMPT_BUILDER, PROMPT_BUILDER_CONDENSE
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.topologies.chain import LATENT_STEPS, hop_budget

WINDOW = 131_072
#: Longest question plus choices on the panel, in Qwen tokens: the question,
#: a newline, and the four ``(X) choice`` lines (``prompts.question_choices_text``).
QUESTION_CHOICES_MAX_TOKENS = 883
HOP_TEMPLATE_ALLOWANCE = 80
HOP_WRAPPER_ALLOWANCE = QUESTION_CHOICES_MAX_TOKENS + HOP_TEMPLATE_ALLOWANCE
#: The rewrite brief, the notes header, the part header, and the closing line
#: around one rewrite prompt, sized to the wider of the two wrappers with slack.
REWRITE_TEMPLATE_ALLOWANCE = 160
REWRITE_WRAPPER_ALLOWANCE = QUESTION_CHOICES_MAX_TOKENS + REWRITE_TEMPLATE_ALLOWANCE
#: The condensing brief is wider than the shared brief's allowance, so it is
#: priced on its own with the same kind of slack. A profile is priced on its
#: builder's allowance.
REWRITE_TEMPLATE_ALLOWANCE_CONDENSE = 240
REWRITE_WRAPPER_ALLOWANCE_CONDENSE = (
    QUESTION_CHOICES_MAX_TOKENS + REWRITE_TEMPLATE_ALLOWANCE_CONDENSE
)
#: The receiver brief and the official question, choices, and format lines around
#: one answer prompt, sized to the wider of the two bodies on the panel with slack.
ANSWER_TEMPLATE_ALLOWANCE = 80
ANSWER_PROMPT_ALLOWANCE = QUESTION_CHOICES_MAX_TOKENS + ANSWER_TEMPLATE_ALLOWANCE
#: The floor arm reads the no-context body, the same question and choices with no
#: text block, so it is priced on the answer prompt's template allowance.
NO_CONTEXT_PROMPT_ALLOWANCE = QUESTION_CHOICES_MAX_TOKENS + ANSWER_TEMPLATE_ALLOWANCE
CLOSER_BUDGET = 4_096
#: How a ``geometry.ratios`` table prices a cut: the plain per-hop re-cut
#: ``ceil(rows / ratio)`` over the whole sequence. It enters the registration
#: body; no arm runs it.
RATIO_TABLE_LAW = "coa"
EXECUTABLE = "executable"
NA_CONTEXT = "N/A_context"


@dataclass(frozen=True)
class ChainGeometry:
    """One item's hop sequences, carried rows, and terminal request."""

    hop_rows: tuple[int, ...]
    carried_rows: tuple[int, ...]
    terminal_request: int
    window: int
    #: Every hop's budget holds the sink and the forty latent rows it protects.
    latent_fits: bool = True

    @property
    def worst_hop(self) -> int:
        """Return the longest hop sequence."""
        return max(self.hop_rows)

    @property
    def terminal_rows(self) -> int:
        """Return the rows the receiver receives."""
        return self.carried_rows[-1]

    @property
    def executable(self) -> bool:
        """Return whether every hop, its protected tail, and the terminal request fit."""
        return (
            self.worst_hop <= self.window
            and self.terminal_request <= self.window
            and self.latent_fits
        )

    @property
    def status(self) -> str:
        """Return the status word for this geometry."""
        return EXECUTABLE if self.executable else NA_CONTEXT


def chain_geometry(
    chunk_tokens: Sequence[int],
    *,
    law: str,
    answer_ceiling: int,
    ratio: int | None = None,
    schedule: str | None = None,
    cumulative_source_tokens: Sequence[int] | None = None,
    budget_rows: int | None = None,
    window: int = WINDOW,
) -> ChainGeometry:
    """Walk the four hops under one budget law and price the terminal request."""
    if not chunk_tokens or any(count < 1 for count in chunk_tokens):
        raise ValueError("chain geometry needs positive chunk token counts")
    if cumulative_source_tokens is not None and len(cumulative_source_tokens) != len(chunk_tokens):
        raise ValueError("cumulative source tokens must be given per chunk")
    carried = 0
    latent_fits = True
    hop_rows: list[int] = []
    carried_rows: list[int] = []
    for index, count in enumerate(chunk_tokens):
        rows = carried + count + HOP_WRAPPER_ALLOWANCE + LATENT_STEPS
        hop_rows.append(rows)
        cumulative = None if cumulative_source_tokens is None else cumulative_source_tokens[index]
        # ``rerank`` spends the prefix on the count and protects none of it,
        # so every carried row is re-scored inside the result.
        if law == RATIO_TABLE_LAW:
            if ratio is None:
                raise ValueError("the ratio table needs a ratio")
            carried = math.ceil(rows / ratio)
        else:
            carried = hop_budget(
                law,
                rows,
                ratio=ratio,
                schedule=schedule,
                cumulative_source_tokens=cumulative,
                prefix_rows=carried,
                budget_rows=budget_rows,
            )
        # The cut protects the sink and the latent tail inside the rows that
        # compete, so a budget too small to hold them makes the item N/A.
        latent_fits = latent_fits and carried > LATENT_STEPS
        carried_rows.append(carried)
    terminal = carried + ANSWER_PROMPT_ALLOWANCE + answer_ceiling + CLOSER_BUDGET
    return ChainGeometry(
        hop_rows=tuple(hop_rows),
        carried_rows=tuple(carried_rows),
        terminal_request=terminal,
        window=window,
        latent_fits=latent_fits,
    )


def direct_request(source_tokens: int, *, answer_ceiling: int) -> int:
    """Return the one-shot request of the whole source in one prompt."""
    return source_tokens + ANSWER_PROMPT_ALLOWANCE + answer_ceiling + CLOSER_BUDGET


def rewrite_wrapper_allowance(builder: str) -> int:
    """Return the rewrite wrapper allowance one prompt builder is priced on."""
    if builder == PROMPT_BUILDER:
        return REWRITE_WRAPPER_ALLOWANCE
    if builder == PROMPT_BUILDER_CONDENSE:
        return REWRITE_WRAPPER_ALLOWANCE_CONDENSE
    raise ValueError(f"no rewrite wrapper allowance is registered for builder {builder!r}")


def text_sender_prompt(chunk_tokens: int, *, report_ceiling: int, wrapper: int) -> int:
    """Return the sender prompt ceiling: the notes, the chunk, the rewrite wrapper."""
    return report_ceiling + chunk_tokens + wrapper


def chain_wrapper_allowances(profile: BenchmarkProfile) -> tuple[int, int]:
    """Read hop and rewrite bounds from the tokenizer that prepared this panel."""
    from rcc.benchmarks.longbench_v2.panel import NEMOTRON_PANEL_KEYS

    if profile.benchmark_key in NEMOTRON_PANEL_KEYS:
        from rcc.benchmarks.longbench_v2.native_geometry import frozen_geometry

        measured = frozen_geometry(profile)
        return int(measured["hop_wrapper_allowance"]), int(measured["rewrite_wrapper_allowance"])
    return HOP_WRAPPER_ALLOWANCE, rewrite_wrapper_allowance(profile.prompt_builder)


def rewrite_prompt_ceiling(chunk_tokens: int, *, report_ceiling: int, wrapper: int) -> int:
    """Return one rendered rewrite prompt's ceiling.

    A rewrite prompt carries the previous hop's notes, at most the report ceiling, one
    part, and the rewrite wrapper the prepared build measures on the panel.
    """
    return report_ceiling + chunk_tokens + wrapper


def text_sender_context(chunk_tokens: Sequence[int], *, report_ceiling: int, wrapper: int) -> int:
    """Return the worst rewrite context: notes, chunk, wrapper, report, closer."""
    return (
        text_sender_prompt(max(chunk_tokens), report_ceiling=report_ceiling, wrapper=wrapper)
        + report_ceiling
        + CLOSER_BUDGET
    )


def ratio_table(
    chunk_tokens_by_qid: Mapping[str, Sequence[int]],
    ratios: Sequence[int],
    *,
    answer_ceiling: int,
    law: str = RATIO_TABLE_LAW,
    window: int = WINDOW,
) -> dict[int, dict[str, int]]:
    """Return the worst hop, the worst terminal request, and N/A items per ratio.

    An empty ladder returns an empty table: a roster of floor and text arms cuts
    nothing, so it is priced on the controls alone.
    """
    table: dict[int, dict[str, int]] = {}
    for ratio in ratios:
        worst_hop = worst_terminal = na_items = 0
        for chunk_tokens in chunk_tokens_by_qid.values():
            geometry = chain_geometry(
                chunk_tokens, law=law, ratio=ratio, answer_ceiling=answer_ceiling, window=window
            )
            worst_hop = max(worst_hop, geometry.worst_hop)
            worst_terminal = max(worst_terminal, geometry.terminal_request)
            na_items += not geometry.executable
        table[ratio] = {
            "worst_hop_rows": worst_hop,
            "worst_terminal_request": worst_terminal,
            "na_items": na_items,
        }
    return table


def bounded_table(
    chunk_tokens_by_qid: Mapping[str, Sequence[int]],
    budgets: Sequence[int],
    *,
    answer_ceiling: int,
    window: int = WINDOW,
) -> dict[int, dict[str, int]]:
    """Return the worst hop, the worst terminal request, and N/A items per row budget."""
    table: dict[int, dict[str, int]] = {}
    for budget_rows in budgets:
        worst_hop = worst_terminal = na_items = 0
        for chunk_tokens in chunk_tokens_by_qid.values():
            geometry = chain_geometry(
                chunk_tokens,
                law="bounded",
                budget_rows=budget_rows,
                answer_ceiling=answer_ceiling,
                window=window,
            )
            worst_hop = max(worst_hop, geometry.worst_hop)
            worst_terminal = max(worst_terminal, geometry.terminal_request)
            na_items += not geometry.executable
        table[budget_rows] = {
            "worst_hop_rows": worst_hop,
            "worst_terminal_request": worst_terminal,
            "na_items": na_items,
        }
    return table


def control_fits(
    chunk_tokens_by_qid: Mapping[str, Sequence[int]],
    source_tokens_by_qid: Mapping[str, int],
    *,
    answer_ceiling: int,
    window: int = WINDOW,
) -> dict[str, int]:
    """Count the items the uncut chain and the one-shot prompt would fit."""
    full = sum(
        chain_geometry(
            chunk_tokens, law="full", answer_ceiling=answer_ceiling, window=window
        ).executable
        for chunk_tokens in chunk_tokens_by_qid.values()
    )
    direct = sum(
        direct_request(tokens, answer_ceiling=answer_ceiling) <= window
        for tokens in source_tokens_by_qid.values()
    )
    return {"full_fits": full, "direct_fits": direct}


__all__ = (
    "ANSWER_PROMPT_ALLOWANCE",
    "ANSWER_TEMPLATE_ALLOWANCE",
    "CLOSER_BUDGET",
    "EXECUTABLE",
    "HOP_TEMPLATE_ALLOWANCE",
    "HOP_WRAPPER_ALLOWANCE",
    "NA_CONTEXT",
    "NO_CONTEXT_PROMPT_ALLOWANCE",
    "QUESTION_CHOICES_MAX_TOKENS",
    "RATIO_TABLE_LAW",
    "REWRITE_TEMPLATE_ALLOWANCE",
    "REWRITE_TEMPLATE_ALLOWANCE_CONDENSE",
    "REWRITE_WRAPPER_ALLOWANCE",
    "REWRITE_WRAPPER_ALLOWANCE_CONDENSE",
    "WINDOW",
    "ChainGeometry",
    "bounded_table",
    "chain_geometry",
    "control_fits",
    "direct_request",
    "ratio_table",
    "rewrite_prompt_ceiling",
    "rewrite_wrapper_allowance",
    "text_sender_context",
    "text_sender_prompt",
)
