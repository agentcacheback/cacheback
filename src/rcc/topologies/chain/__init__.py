"""The four-hop sequential chain topology and its resident-budget laws.

One item walks its source in four ordered chunks on one producer seat, and after
every hop the whole sequence is re-ranked with only the sink and latent rows held.
"""

from __future__ import annotations

import math

from rcc.topologies.protocol import TopologyProfile

#: The topology key every chain profile carries and every chain seat checks.
CHAIN_TOPOLOGY_KEY = "chain-t4"
CHAIN_T4 = TopologyProfile(
    topology_id="chain-t4-v1",
    workers_per_item=4,
    receiver_count=1,
    worker_assignment="sequential-hop-chain-global-recut",
    isolation="one-item-arm-seed-cell; one producer seat owns all four hops; "
    "model-specific physical placement receipt",
)

#: Hops per item; equal to the topology's worker count.
HOPS = CHAIN_T4.workers_per_item
#: Strict span width of every cut on this topology.
SPAN_WIDTH = 16
#: Latent rows rolled after every hop; the current roll is always protected.
LATENT_STEPS = 40
#: Resident-schedule unit: one full source chunk at its ceiling, in tokens.
SCHEDULE_UNIT_TOKENS = 64_000
#: Base resident rows for the looped-agent schedules.
KAPPA = 2_048
#: The bounded confirmatory reference schedule.
CAP_ROWS = 65_536
SCHEDULE_NAMES = ("constant", "log", "sqrt", "linear", "cap_65536")
BUDGET_LAWS = ("rerank", "bounded", "la", "full")
#: The re-ranked handoff law: the carried count plus the new rows cut, with
#: no protected prefix, so every carried row is re-scored against the rows the
#: hop just read.
RERANK_BUDGET_LAW = "rerank"
#: The bounded law: one fixed carried row count, whatever the document.
BOUNDED_BUDGET_LAW = "bounded"
#: The bounded ladder, largest first: powers of two from the largest fixed memory
#: the 131,072 window admits beside one raw Easy50 chunk (61,466 rows on the Qwen
#: tokenizer) down to 2,048, the rung under the smallest text note.
BOUNDED_ROW_BUDGETS = (65_536, 32_768, 16_384, 8_192, 4_096, 2_048)


def rerank_budget(rows: int, prefix_rows: int, ratio: int) -> int:
    """Return the re-ranked handoff budget: the carried count plus the new rows cut.

    ``prefix_rows + ceil((rows - prefix_rows) / ratio)``. The prefix sets the count
    only: the cut protects no carried row, so any of them may lose its seat.
    """
    if rows < 1:
        raise ValueError("a hop sequence has at least one row")
    if ratio < 1:
        raise ValueError("retention ratio must be positive")
    if not 0 <= prefix_rows < rows:
        raise ValueError("the carried prefix is a proper part of the hop sequence")
    return prefix_rows + math.ceil((rows - prefix_rows) / ratio)


def bounded_budget(rows: int, budget_rows: int) -> int:
    """Return the bounded budget: every row until the cap binds, then the cap.

    ``min(budget_rows, rows)``, so early hops on a short document compress nothing and,
    once the cap binds, exactly ``budget_rows`` survive, chosen from the whole sequence.
    """
    if rows < 1:
        raise ValueError("a hop sequence has at least one row")
    if budget_rows < 1:
        raise ValueError("the bounded budget must be positive")
    return min(budget_rows, rows)


def schedule_rows(name: str, cumulative_source_tokens: int) -> int:
    """Return the resident rows one schedule requests at one stream size.

    With ``z = max(1, S_t / 64,000)`` the growth laws are ``kappa``,
    ``kappa (1 + ln z)``, ``kappa (1 + 2 (sqrt z - 1))``, and ``kappa z``, span-floored.
    """
    if cumulative_source_tokens < 1:
        raise ValueError("cumulative source tokens must be positive")
    if name == "cap_65536":
        return CAP_ROWS
    z = max(1.0, cumulative_source_tokens / SCHEDULE_UNIT_TOKENS)
    if name == "constant":
        raw = float(KAPPA)
    elif name == "log":
        raw = KAPPA * (1.0 + math.log(z))
    elif name == "sqrt":
        raw = KAPPA * (1.0 + 2.0 * (math.sqrt(z) - 1.0))
    elif name == "linear":
        raw = KAPPA * z
    else:
        raise ValueError(f"unregistered chain schedule {name!r}")
    return math.floor(raw / SPAN_WIDTH) * SPAN_WIDTH


def la_budget(name: str, cumulative_source_tokens: int, rows: int) -> int:
    """Return the looped-agent budget: the schedule clamped to the candidate rows."""
    if rows < 1:
        raise ValueError("a hop sequence has at least one row")
    return min(schedule_rows(name, cumulative_source_tokens), rows)


def hop_budget(
    law: str,
    rows: int,
    *,
    ratio: int | None = None,
    schedule: str | None = None,
    cumulative_source_tokens: int | None = None,
    prefix_rows: int | None = None,
    budget_rows: int | None = None,
) -> int:
    """Resolve one hop's budget from a law name and that law's own parameter."""
    if law == RERANK_BUDGET_LAW:
        if ratio is None:
            raise ValueError("the rerank law needs a ratio")
        return rerank_budget(rows, 0 if prefix_rows is None else prefix_rows, ratio)
    if law == BOUNDED_BUDGET_LAW:
        if budget_rows is None:
            raise ValueError("the bounded law needs a row budget")
        return bounded_budget(rows, budget_rows)
    if law == "la":
        if schedule is None or cumulative_source_tokens is None:
            raise ValueError("the la law needs a schedule and the cumulative source tokens")
        return la_budget(schedule, cumulative_source_tokens, rows)
    if law == "full":
        if rows < 1:
            raise ValueError("a hop sequence has at least one row")
        return rows
    raise ValueError(f"unregistered chain budget law {law!r}")


__all__ = (
    "BOUNDED_BUDGET_LAW",
    "BOUNDED_ROW_BUDGETS",
    "BUDGET_LAWS",
    "CAP_ROWS",
    "CHAIN_T4",
    "CHAIN_TOPOLOGY_KEY",
    "HOPS",
    "KAPPA",
    "LATENT_STEPS",
    "RERANK_BUDGET_LAW",
    "SCHEDULE_NAMES",
    "SCHEDULE_UNIT_TOKENS",
    "SPAN_WIDTH",
    "bounded_budget",
    "hop_budget",
    "la_budget",
    "rerank_budget",
    "schedule_rows",
)
