"""The four-hop chain loop, its provenance ledger, and the terminal payload.

One producer seat walks one item's four ordered chunks, re-cutting the whole
hop sequence after every hop under the arm's budget law (docs/benchmarks.md).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from rcc.benchmarks.longbench_v2.data import ChainItem
from rcc.benchmarks.longbench_v2.prompts import (
    PROMPT_BUILDERS,
    hop_prompt_ids,
    prompt_sha256,
    retention_prompt,
)
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.nemotron import PRODUCER_ROUTE as NEMOTRON_PRODUCER_ROUTE
from rcc.models.nemotron.producer import NemotronProducer
from rcc.models.qwen.capture import (
    QWEN_CAPTURE_SELECTORS,
    QwenFlatPayload,
    chain_plan_sha256,
    qwen_w16_keep,
    registered_selection,
    selected_indices_sha256,
    tensor_content_sha256,
)
from rcc.models.qwen.engine import QWEN_CHAIN_ENGINE_ROUTE, EngineProducer, HopProduct
from rcc.models.route import RouteFamily
from rcc.topologies.chain import (
    BUDGET_LAWS,
    HOPS,
    LATENT_STEPS,
    hop_budget,
)
from rcc.topologies.chain.layout import CHAIN_TERMINAL_LAYOUT
from rcc.transforms.select.query_support.methods.compose import SUPPORT_DEFAULT_ALPHA

#: What the reference leg runs instead of the engine: one plain HF forward per
#: hop, then the same roll. It is a route name because a banked row records
#: which path made it.
QWEN_CHAIN_REFERENCE_ROUTE = "hf-forward-latent-roll-v1"
#: The hop paths this loop can walk, and the backend and route each one
#: records. A production names the path it ran.
CHAIN_HOP_PATHS: dict[str, tuple[str, str]] = {
    "engine": ("engine", QWEN_CHAIN_ENGINE_ROUTE),
    "nemotron-engine": ("engine", NEMOTRON_PRODUCER_ROUTE),
    "hf-reference": ("hf", QWEN_CHAIN_REFERENCE_ROUTE),
}


def chain_hop_path(family: RouteFamily) -> str:
    """Return the engine path identity for one family's chain producer."""
    if family.lane == "nemotron":
        route = str(dict(family.profile.runtime.engine_flags)["producer_backend"])
        if route != NEMOTRON_PRODUCER_ROUTE:
            raise ValueError(f"unregistered Nemotron chain producer route {route!r}")
        return "nemotron-engine"
    return "engine"


def chain_hop_backend(path: str | tuple[str, str]) -> tuple[str, str]:
    """Return the backend and route fields for a recorded chain path."""
    if path in CHAIN_HOP_PATHS:
        return CHAIN_HOP_PATHS[path]
    if isinstance(path, tuple) and path in CHAIN_HOP_PATHS.values():
        return path
    raise ValueError(f"unregistered chain hop path {path!r}")


class HopFn(Protocol):
    """One hop of prefill, roll, and capture, whatever path performs it.

    Naming the hop as a parameter lets the reference leg drive the same loop
    over a plain HF forward, so the two paths differ only in the prefill.
    """

    def __call__(
        self,
        prefix_rows: torch.Tensor | None,
        tail_ids: torch.Tensor,
        *,
        judger_ids: torch.Tensor,
        judger_mask: torch.Tensor,
        question_ids: Sequence[int],
        consume_past: bool = False,
    ) -> HopProduct:
        """Return one hop's rows, selector scores, geometry, and timings."""
        ...


@dataclass(frozen=True)
class HopRecord:
    """One hop's geometry, its cut, and the origin of every row it kept."""

    hop: int
    prefix_rows: int
    prompt_rows: int
    rows_pre_cut: int
    budget: int
    keep: tuple[int, ...]
    origins: tuple[tuple[int, int], ...]
    prompt_sha256: str
    extract_s: float
    roll_s: float
    capture_s: float
    capture_peak_bytes: int

    def __post_init__(self) -> None:
        """Validate one hop record against the geometry it claims."""
        if not 1 <= self.hop <= HOPS:
            raise ValueError(f"a chain hop is between 1 and {HOPS}, got {self.hop}")
        if self.prefix_rows < 0 or self.prompt_rows < 1:
            raise ValueError("a chain hop needs a nonnegative prefix and a nonempty prompt")
        if self.rows_pre_cut != self.prefix_rows + self.prompt_rows + LATENT_STEPS:
            raise ValueError("a chain hop sequence is its prefix, its prompt, and the roll")
        if len(self.keep) != self.budget:
            raise ValueError("a chain hop keeps exactly its budget")
        if len(self.origins) != len(self.keep):
            raise ValueError("a chain hop names the origin of every row it kept")
        if any(not 1 <= born <= self.hop for born, _ in self.origins):
            raise ValueError("a kept row is born in this hop or an earlier one")
        if any(value < 0 for value in (self.extract_s, self.roll_s, self.capture_s)):
            raise ValueError("chain hop timings must be nonnegative")
        if self.capture_peak_bytes < 0:
            raise ValueError("chain hop peak allocation must be nonnegative")


@dataclass(frozen=True)
class ChainProduction:
    """One terminal payload, its four hop records, and their measurements."""

    qid: str
    semantic_arm: str
    payload: QwenFlatPayload
    selector_score_name: str
    selector_scores: tuple[torch.Tensor, ...]
    hops: tuple[HopRecord, ...]
    retention_prompt_sha256: str
    hop_path: str
    budget_law: str

    def __post_init__(self) -> None:
        """Validate the payload identity against the ledger that produced it."""
        if self.budget_law not in BUDGET_LAWS:
            raise ValueError(f"a chain production runs a registered law, not {self.budget_law!r}")
        chain_hop_backend(self.hop_path)
        if self.payload.semantic_arm != self.semantic_arm:
            raise ValueError("chain production and payload semantic arms differ")
        if self.payload.layout != CHAIN_TERMINAL_LAYOUT:
            raise ValueError(f"a chain production ships layout {CHAIN_TERMINAL_LAYOUT!r}")
        if self.selector_score_name not in QWEN_CAPTURE_SELECTORS:
            raise ValueError("chain production has an unregistered selector score")
        if len(self.hops) != HOPS or [hop.hop for hop in self.hops] != list(range(1, HOPS + 1)):
            raise ValueError(f"a chain production walks hops 1 to {HOPS} in order")
        if self.payload.keeps != tuple(hop.keep for hop in self.hops):
            raise ValueError("chain payload keeps differ from the ledger's cuts")
        if len(self.selector_scores) != HOPS or any(
            score.shape != (hop.rows_pre_cut,) or not bool(score.isfinite().all())
            for score, hop in zip(self.selector_scores, self.hops, strict=True)
        ):
            raise ValueError("a chain production retains one finite selector vector per hop")

    @property
    def extract_s(self) -> float:
        """Return the extraction seconds of all four hops."""
        return sum(hop.extract_s for hop in self.hops)

    @property
    def roll_s(self) -> float:
        """Return the roll seconds of all four hops."""
        return sum(hop.roll_s for hop in self.hops)

    @property
    def capture_s(self) -> float:
        """Return the selector-capture seconds of all four hops."""
        return sum(hop.capture_s for hop in self.hops)

    def result_fields(self) -> dict[str, object]:
        """Return payload lineage, the hop ledger, and producer measurements.

        The FanOutQA field names keep their meaning here, so one bank reads
        both topologies; what each hop kept banks as ``kept_rows_by_hop``.
        """
        backend, route = chain_hop_backend(self.hop_path)
        shipped = int(self.payload.rows.shape[0])
        fields: dict[str, object] = {
            "producer_backend": backend,
            "producer_route": route,
            "budget_law": self.budget_law,
            "payload_layout": self.payload.layout,
            "payload_semantic_arm": self.payload.semantic_arm,
            "payload_plan_sha256": self.payload.latent_plan_sha256,
            "payload_tensor_sha256": self.payload.tensor_sha256,
            "selected_indices_sha256": self.payload.selected_indices_sha256,
            "selector_score_name": self.selector_score_name,
            "selector_score_tensor_sha256": [
                tensor_content_sha256(score) for score in self.selector_scores
            ],
            "keeps_by_worker": [list(keep) for keep in self.payload.keeps],
            "kept_rows_by_hop": list(self.payload.rows_by_worker),
            "latent_rows_by_worker": [shipped],
            "latent_tokens": shipped,
            "hop_prompt_sha256": [hop.prompt_sha256 for hop in self.hops],
            "retention_prompt_sha256": self.retention_prompt_sha256,
            "hop_prefix_rows": [hop.prefix_rows for hop in self.hops],
            "hop_prompt_rows": [hop.prompt_rows for hop in self.hops],
            "hop_rows_pre_cut": [hop.rows_pre_cut for hop in self.hops],
            "hop_budgets": [hop.budget for hop in self.hops],
            "terminal_origins": [[born, position] for born, position in self.hops[-1].origins],
            "hop_extract_s": [round(hop.extract_s, 4) for hop in self.hops],
            "hop_latent_roll_s": [round(hop.roll_s, 4) for hop in self.hops],
            "hop_selector_capture_s": [round(hop.capture_s, 4) for hop in self.hops],
            "extract_s": round(self.extract_s, 4),
            "latent_roll_s": round(self.roll_s, 4),
            "selector_capture_s": round(self.capture_s, 4),
            "selector_capture_peak_bytes": max(hop.capture_peak_bytes for hop in self.hops),
        }
        if self.semantic_arm.startswith("latent_query_support_"):
            fields["support_alpha"] = SUPPORT_DEFAULT_ALPHA
        return fields


def _hop_prompt_sha256(prompt_ids: Sequence[int]) -> str:
    """Hash one hop's prompt ids, which are the only form the prompt has.

    A hop prompt is two rendered halves around the chunk's standalone ids, so it
    exists as ids and never as one string.
    """
    encoded = json.dumps([int(token) for token in prompt_ids], separators=(",", ":"))
    return prompt_sha256(encoded)


def _require_chain_profile(profile: BenchmarkProfile) -> None:
    """Refuse, by registered name, a profile this loop cannot walk."""
    if profile.prompt_builder not in PROMPT_BUILDERS:
        raise ValueError(
            f"Qwen chain producer has no registered prompt builder "
            f"{profile.prompt_builder!r} for {profile.benchmark_key}"
        )
    if profile.payload_layout != CHAIN_TERMINAL_LAYOUT:
        raise ValueError(
            f"{profile.benchmark_key} registers payload layout "
            f"{profile.payload_layout!r}, not {CHAIN_TERMINAL_LAYOUT!r}"
        )
    if profile.workers_per_item != HOPS:
        raise ValueError(
            f"{profile.benchmark_key} registers {profile.workers_per_item} workers "
            f"per item, not the {HOPS} hops of the chain"
        )
    if profile.latent_steps != LATENT_STEPS:
        raise ValueError(
            f"{profile.benchmark_key} registers a {profile.latent_steps}-step roll, "
            f"not the {LATENT_STEPS} rows the chain cut protects"
        )


def produce_chain_payload(
    producer: EngineProducer | NemotronProducer,
    tokenizer: Any,
    item: ChainItem,
    *,
    semantic_arm: str,
    family: RouteFamily,
    profile: BenchmarkProfile,
    hop_fn: HopFn | None = None,
) -> ChainProduction:
    """Walk one item's four hops, re-cutting under the arm's law, and ship the last block.

    ``hop_fn`` defaults to the producer's own engine hop; the reference leg
    passes a plain HF pass instead.
    """
    _require_chain_profile(profile)
    run_hop = producer.produce_hop if hop_fn is None else hop_fn
    hop_path = chain_hop_path(family) if hop_fn is None else "hf-reference"
    selector, ratio, budget_law, budget_rows = registered_selection(semantic_arm, profile=profile)
    if budget_law is None:
        raise ValueError(f"{semantic_arm!r} names no chain budget law")
    # The carried rows compete under every law; the laws differ only in the
    # count `hop_budget` resolves.
    plan_sha256 = chain_plan_sha256(semantic_arm, family=family, profile=profile)
    if len(item.chunks) != HOPS:
        raise RuntimeError(
            f"{item.qid}/{semantic_arm}: the prepared item carries "
            f"{len(item.chunks)} chunks, not the {HOPS} the chain walks"
        )
    enable_thinking = family.profile.decode.enable_thinking
    retention_text, retention_ids, question_ids = retention_prompt(
        tokenizer, item.question, enable_thinking=enable_thinking, family=family
    )
    judger = torch.tensor((retention_ids,), dtype=torch.long, device=producer.device)
    judger_mask = torch.ones_like(judger)

    carried: torch.Tensor | None = None
    origins: tuple[tuple[int, int], ...] = ()
    records: list[HopRecord] = []
    scores_by_hop: list[torch.Tensor] = []
    for hop in range(1, HOPS + 1):
        prompt_ids = hop_prompt_ids(
            tokenizer,
            item.question,
            item.choices,
            item.chunks[hop - 1],
            hop,
            enable_thinking=enable_thinking,
            family=family,
        )
        prefix_rows = 0 if carried is None else int(carried.shape[1])
        rows_pre_cut = prefix_rows + len(prompt_ids) + profile.latent_steps
        if rows_pre_cut > profile.max_model_len:
            raise RuntimeError(
                f"{item.qid}/{semantic_arm}/hop {hop}: {rows_pre_cut} rows "
                f"exceed the {profile.max_model_len} window"
            )
        # The hop's cache is dropped after the cut, so the scorer takes it over
        # rather than copying it.
        product = run_hop(
            carried,
            torch.tensor((prompt_ids,), dtype=torch.long),
            judger_ids=judger,
            judger_mask=judger_mask,
            question_ids=question_ids,
            consume_past=True,
        )
        rows = product.embeds[0]
        if int(rows.shape[0]) != rows_pre_cut or product.prefix_rows != prefix_rows:
            raise RuntimeError(
                f"{item.qid}/{semantic_arm}/hop {hop}: the engine rolled "
                f"{int(rows.shape[0])} rows over a {product.prefix_rows}-row prefix, "
                f"not the planned {rows_pre_cut} over {prefix_rows}"
            )
        if not product.question_found:
            raise RuntimeError(
                f"{item.qid}/{semantic_arm}/hop {hop}: the retention prompt "
                f"did not locate the question in its {len(retention_ids)} rows"
            )
        budget = hop_budget(
            budget_law, rows_pre_cut, ratio=ratio, prefix_rows=prefix_rows, budget_rows=budget_rows
        )
        # The cut checks its own budget; this refusal restates that failure
        # with the item, the arm, and the hop the run stopped on.
        try:
            keep = qwen_w16_keep(
                product.scores[selector],
                ratio,
                latent_steps=profile.latent_steps,
                budget=budget,
            )
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"{item.qid}/{semantic_arm}/hop {hop}: the cut over {rows_pre_cut} rows "
                f"did not return the {budget} rows its budget allows: {exc}"
            ) from exc
        # The carried origins lead this hop's sequence in order, so a position
        # below the prefix names the row that survived every earlier cut.
        born = origins + tuple((hop, position) for position in range(prefix_rows, rows_pre_cut))
        origins = tuple(born[position] for position in keep)
        records.append(
            HopRecord(
                hop=hop,
                prefix_rows=prefix_rows,
                prompt_rows=len(prompt_ids),
                rows_pre_cut=rows_pre_cut,
                budget=budget,
                keep=keep,
                origins=origins,
                prompt_sha256=_hop_prompt_sha256(prompt_ids),
                extract_s=product.extract_s,
                roll_s=product.roll_s,
                capture_s=product.capture_s,
                capture_peak_bytes=product.capture_peak_bytes,
            )
        )
        scores_by_hop.append(product.scores[selector].detach().float().cpu())
        carried = rows[list(keep)].detach().unsqueeze(0)

    assert carried is not None
    terminal = carried[0].detach().cpu().to(torch.bfloat16)
    keeps = tuple(record.keep for record in records)
    payload = QwenFlatPayload(
        rows=terminal,
        semantic_arm=semantic_arm,
        latent_plan_sha256=plan_sha256,
        keeps=keeps,
        rows_by_worker=tuple(len(keep) for keep in keeps),
        selected_indices_sha256=selected_indices_sha256(keeps),
        tensor_sha256=tensor_content_sha256(terminal),
        family=family,
        profile=profile,
        layout=profile.payload_layout,
    )
    return ChainProduction(
        qid=item.qid,
        semantic_arm=semantic_arm,
        payload=payload,
        selector_score_name=selector,
        selector_scores=tuple(scores_by_hop),
        hops=tuple(records),
        retention_prompt_sha256=prompt_sha256(retention_text),
        hop_path=hop_path,
        budget_law=budget_law,
    )


__all__ = (
    "CHAIN_HOP_PATHS",
    "QWEN_CHAIN_REFERENCE_ROUTE",
    "ChainProduction",
    "HopFn",
    "HopRecord",
    "produce_chain_payload",
)
