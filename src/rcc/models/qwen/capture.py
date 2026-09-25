"""Qwen capture, p2-a2 selection, and flat embedding payloads.

The FanOutQA and chain producers score a rolled cache here, cut it, sign the
recipe a payload names, and materialize the payload rows.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50, SEALED_M3_PLAN_LABEL
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.route import RouteFamily
from rcc.models.selection import ScoreAdapter
from rcc.topologies.chain import CHAIN_T4, HOPS
from rcc.topologies.chain.layout import CHAIN_TERMINAL_LAYOUT
from rcc.topologies.fanout import FANOUT_M3
from rcc.topologies.fanout.layout import (
    FLAT_INTERLEAVE_LAYOUT,
    SPAN_WIDTH,
    flat_interleave_layout,
    interleaved_blocks,
)
from rcc.transforms.select.query_support.fixed_spans import fixed_span_keep
from rcc.transforms.select.query_support.methods.compose import (
    SUPPORT_DEFAULT_ALPHA,
    SUPPORT_DEFAULT_ORDER,
    _pin_sink,
    selector_score,
)
from rcc.transforms.select.query_support.methods.scorers import (
    memory_votes_with_support_moments,
)

QWEN_LATENT_STEPS = FANOUTQA_NATURAL_DEV50.latent_steps
QWEN_SPAN_WIDTH = SPAN_WIDTH
QWEN_WORKERS_PER_ITEM = FANOUT_M3.workers_per_item
QWEN_RATIOS = FANOUTQA_NATURAL_DEV50.ratios
QWEN_SUPPORT_SELECTOR = "support-p2-a2"
QWEN_PAYLOAD_LAYOUT = FLAT_INTERLEAVE_LAYOUT
QWEN_CAPTURE_SELECTORS = ("snap", "support")

if FANOUTQA_NATURAL_DEV50.span_width != QWEN_SPAN_WIDTH:
    raise RuntimeError("Qwen benchmark and canonical fan-out topology disagree on W16")


def tensor_content_sha256(tensor: torch.Tensor) -> str:
    """Hash a tensor's dtype, shape, and contiguous decoded value bytes."""
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def selected_indices_sha256(keeps: Sequence[Sequence[int]]) -> str:
    """Hash the complete ordered keep roster, worker by worker."""
    normalized = tuple(tuple(int(position) for position in keep) for keep in keeps)
    return hashlib.sha256(json.dumps(normalized, separators=(",", ":")).encode("ascii")).hexdigest()


def latent_plan_sha256(semantic_arm: str, *, family: RouteFamily) -> str:
    """Sign one registered latent arm and its canonical fan-out recipe.

    The schema string names the layout, which both families share; the family
    itself enters through `model_id` and the physical arm.
    """
    benchmark_arms = {arm.arm_id: arm for arm in FANOUTQA_NATURAL_DEV50.arms}
    physical_arms = {arm.semantic_arm: arm for arm in family.profile.physical_arms}
    try:
        benchmark_arm = benchmark_arms[semantic_arm]
        physical_arm = physical_arms[semantic_arm]
    except KeyError as exc:
        raise ValueError(f"unregistered Qwen semantic arm {semantic_arm!r}") from exc
    if (
        benchmark_arm.channel != "latent"
        or benchmark_arm.selector not in QWEN_CAPTURE_SELECTORS
        or benchmark_arm.retention_ratio not in QWEN_RATIOS
        or physical_arm.selector != benchmark_arm.selector
        or physical_arm.payload_layout != FLAT_INTERLEAVE_LAYOUT
    ):
        raise ValueError(f"Qwen arm {semantic_arm!r} is not a registered flat latent arm")
    identity = {
        "schema": "qwen-flat-interleave-plan-v1",
        "benchmark_profile": SEALED_M3_PLAN_LABEL,
        "model_id": family.model_id,
        "topology": FANOUT_M3.to_dict(),
        "arm": benchmark_arm.to_dict(),
        "physical_arm": physical_arm.to_dict(),
        "latent_steps": FANOUTQA_NATURAL_DEV50.latent_steps,
        "span_width": SPAN_WIDTH,
        "support_composition": QWEN_SUPPORT_SELECTOR,
        "payload_layout": FLAT_INTERLEAVE_LAYOUT,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def chain_plan_sha256(semantic_arm: str, *, family: RouteFamily, profile: BenchmarkProfile) -> str:
    """Sign one registered latent arm and the chain's own terminal recipe.

    The chain ships one hop's block after four cuts, so it carries its own
    schema and topology, and its signed body carries ``budget_law``.
    """
    if profile.payload_layout != CHAIN_TERMINAL_LAYOUT:
        raise ValueError(
            f"{profile.benchmark_key} registers payload layout "
            f"{profile.payload_layout!r}, not the {CHAIN_TERMINAL_LAYOUT!r} this plan signs"
        )
    benchmark_arms = {arm.arm_id: arm for arm in profile.arms}
    physical_arms = {arm.semantic_arm: arm for arm in family.profile.physical_arms}
    # Two rosters: the benchmark registers the arm and the lane registers the
    # model that implements it, so each gap is named for what is missing.
    try:
        benchmark_arm = benchmark_arms[semantic_arm]
    except KeyError as exc:
        raise ValueError(
            f"{profile.benchmark_key} registers no chain latent arm {semantic_arm!r}"
        ) from exc
    try:
        physical_arm = physical_arms[semantic_arm]
    except KeyError as exc:
        raise ValueError(
            f"the {family.lane} lane implements no physical arm {semantic_arm!r}"
        ) from exc
    # A chain latent arm names its law and carries exactly one count: a ratio
    # under the rerank law, a row budget under the bounded law. `profile.ratios`
    # is not consulted here, because a bounded profile registers no ratio.
    if (
        benchmark_arm.channel != "latent"
        or benchmark_arm.selector not in QWEN_CAPTURE_SELECTORS
        or benchmark_arm.budget_law is None
        or (benchmark_arm.retention_ratio is None) == (benchmark_arm.budget_rows is None)
        or physical_arm.selector != benchmark_arm.selector
    ):
        raise ValueError(f"Qwen arm {semantic_arm!r} is not a registered chain latent arm")
    # Two layouts sit in this body: `payload_layout` is the chain's, the recipe
    # this digest signs, and `physical_payload_layout` is the lane's own fan-out
    # layout, named apart so neither is read as the other.
    identity = {
        "schema": "qwen-chain-terminal-plan-v1",
        "benchmark_profile": profile.profile_id,
        "topology": CHAIN_T4.to_dict(),
        "arm": benchmark_arm.to_dict(),
        "physical_arm": physical_arm.to_dict(),
        "physical_payload_layout": physical_arm.payload_layout,
        "latent_steps": profile.latent_steps,
        "span_width": profile.span_width,
        "support_composition": QWEN_SUPPORT_SELECTOR,
        "payload_layout": CHAIN_TERMINAL_LAYOUT,
        "hops": HOPS,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def payload_plan_sha256(
    semantic_arm: str, *, family: RouteFamily, profile: BenchmarkProfile
) -> str:
    """Return the plan digest of the recipe the profile's layout registers.

    One place decides which recipe signs a payload, so the producer and a
    reader re-deriving a banked digest resolve it the same way.
    """
    if profile.payload_layout == CHAIN_TERMINAL_LAYOUT:
        return chain_plan_sha256(semantic_arm, family=family, profile=profile)
    return latent_plan_sha256(semantic_arm, family=family)


def registered_selection(
    semantic_arm: str, *, profile: BenchmarkProfile
) -> tuple[str, int | None, str | None, int | None]:
    """Resolve one latent arm's selector, ratio, budget law, and bounded row count.

    A chain arm names its law and carries one count, a ratio or a row budget. A
    fan-out arm names no law and its one cut is ``ceil(rows / ratio)``.
    """
    arm = next((arm for arm in profile.arms if arm.arm_id == semantic_arm), None)
    if arm is None or arm.channel != "latent" or arm.selector not in QWEN_CAPTURE_SELECTORS:
        raise ValueError(f"{profile.benchmark_key} registers no latent arm {semantic_arm!r}")
    if arm.budget_law is None:
        if profile.payload_layout == CHAIN_TERMINAL_LAYOUT:
            raise ValueError(f"{profile.benchmark_key}: {semantic_arm!r} names no chain budget law")
        if arm.retention_ratio is None:
            raise ValueError(f"{semantic_arm!r} carries no ratio")
        return arm.selector, arm.retention_ratio, None, None
    if (arm.retention_ratio is None) == (arm.budget_rows is None):
        raise ValueError(f"{semantic_arm!r} carries exactly one of a ratio and a row budget")
    return arm.selector, arm.retention_ratio, arm.budget_law, arm.budget_rows


def _latent_selection(semantic_arm: str, *, family: RouteFamily) -> tuple[str, int]:
    """Resolve selector and ratio from the signed benchmark arm, never caller fields."""
    latent_plan_sha256(semantic_arm, family=family)
    arm = next(arm for arm in FANOUTQA_NATURAL_DEV50.arms if arm.arm_id == semantic_arm)
    assert arm.selector is not None
    assert arm.retention_ratio is not None
    return arm.selector, arm.retention_ratio


def _protected_positions(length: int, latent_steps: int) -> tuple[int, ...]:
    """Return the Qwen cargo law: sink zero plus the complete rolled tail."""
    if length < 1:
        return ()
    if latent_steps < 0:
        raise ValueError("latent_steps must be nonnegative")
    return (0, *range(max(1, length - latent_steps), length))


def capture_qwen_scores(
    model: object,
    past: Any,
    judger_ids: torch.Tensor,
    judger_mask: torch.Tensor,
    question_ids: Sequence[int],
    *,
    consume_past: bool = False,
) -> tuple[dict[str, torch.Tensor], bool]:
    """Capture mean query attention and shipped p2-a2 Query-Support in one fused pass."""
    moments, found = memory_votes_with_support_moments(
        model,
        past,
        judger_ids,
        judger_mask,
        question_ids,
        orders=(SUPPORT_DEFAULT_ORDER,),
        n_sink=0,
        consume_past=consume_past,
    )
    if SUPPORT_DEFAULT_ORDER != 2.0 or SUPPORT_DEFAULT_ALPHA != 2.0:
        raise RuntimeError("Qwen production requires the registered support-p2-a2 selector")
    scores = {
        "snap": _pin_sink(moments.snap, 1).detach().float().cpu(),
        "support": _pin_sink(selector_score(moments, QWEN_SUPPORT_SELECTOR), 1)
        .detach()
        .float()
        .cpu(),
    }
    return scores, bool(found)


def qwen_w16_keep(
    scores: torch.Tensor,
    ratio: int | None,
    *,
    latent_steps: int = QWEN_LATENT_STEPS,
    budget: int | None = None,
) -> tuple[int, ...]:
    """Return the exact fixed-W16 keep with sink and all latent rows protected.

    Every scored row competes; only the sink at zero and the current latent
    rows are protected. ``budget`` is total rows, else ``ceil(rows / ratio)``.
    """
    if ratio is not None and ratio not in QWEN_RATIOS:
        raise ValueError(f"Qwen ratio must be one of {QWEN_RATIOS}, got {ratio}")
    if scores.ndim != 1:
        raise ValueError("Qwen selector scores must be one-dimensional")
    length = int(scores.numel())
    protected = _protected_positions(length, latent_steps)
    if budget is None:
        if ratio is None:
            raise ValueError("a keep with no budget needs a ratio")
        expected = math.ceil(length / ratio)
    else:
        expected = budget
    if not 1 <= expected <= length:
        raise ValueError("the requested keep is a nonempty part of the scored rows")
    # `locbench_rn_keep` is this call at the ratio's own count, so the grid, the
    # ranking, and the partial-span rule are the registered ones either way.
    keep = fixed_span_keep(scores, expected, protected, span_size=QWEN_SPAN_WIDTH)
    if len(keep) != expected or not set(protected).issubset(keep):
        raise RuntimeError("Qwen W16 selection changed its exact budget or protected rows")
    return tuple(keep)


@dataclass(frozen=True)
class QwenWorkerCapture:
    """One rolled worker's embedding rows and canonical selector vectors."""

    embedding_rows: torch.Tensor
    scores: Mapping[str, torch.Tensor]
    question_found: bool
    prompt_token_ids: tuple[int, ...]
    latent_steps: int = QWEN_LATENT_STEPS

    #: Optional independently replayable family-specific scoring components.
    selector_components: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Validate one complete worker capture."""
        rows = self.embedding_rows
        if rows.ndim != 3 or rows.shape[0] != 1:
            raise ValueError("Qwen worker embeddings must be [1, length, hidden]")
        if set(self.scores) != set(QWEN_CAPTURE_SELECTORS):
            raise ValueError("Qwen capture must contain exactly snap and support scores")
        length = int(rows.shape[1])
        if any(score.shape != (length,) for score in self.scores.values()):
            raise ValueError("Qwen selector vectors must cover every rolled embedding row")
        if not self.question_found:
            raise ValueError("Qwen capture did not locate the question in the manager prompt")
        if self.latent_steps < 0 or len(self.prompt_token_ids) + self.latent_steps != length:
            raise ValueError("Qwen capture prompt plus latent rows differ from its geometry")

    @property
    def length(self) -> int:
        """Return the rolled row count."""
        return int(self.embedding_rows.shape[1])

    def select(self, selector: str, ratio: int, adapter: ScoreAdapter | None) -> tuple[int, ...]:
        """Apply the registered family recipe with its complete scoring evidence."""
        if adapter is None:
            return qwen_w16_keep(self.scores[selector], ratio, latent_steps=self.latent_steps)
        bank = self.selector_components
        if bank is None or bank.shape != adapter.shape(self.length):
            raise ValueError("family selection requires its registered component bank")
        adapter.validate(bank, self.scores[selector])
        return adapter.keep(bank, ratio)


@dataclass(frozen=True)
class QwenFlatPayload:
    """Embedding-only payload in its benchmark's worker/source order."""

    rows: torch.Tensor
    semantic_arm: str
    latent_plan_sha256: str
    keeps: tuple[tuple[int, ...], ...]
    rows_by_worker: tuple[int, ...]
    selected_indices_sha256: str
    tensor_sha256: str
    family: RouteFamily
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50
    layout: str = QWEN_PAYLOAD_LAYOUT

    def __post_init__(self) -> None:
        """Validate the payload identity and geometry against the profile it names."""
        if self.layout != self.profile.payload_layout:
            raise ValueError(
                f"Qwen payload must use registered layout {self.profile.payload_layout!r}"
            )
        expected_plan = payload_plan_sha256(
            self.semantic_arm, family=self.family, profile=self.profile
        )
        if self.latent_plan_sha256 != expected_plan:
            raise ValueError("Qwen payload latent-plan signature differs")
        if self.rows.ndim != 2:
            raise ValueError("Qwen payload rows must be [length, hidden]")
        self._validate_geometry()
        if self.tensor_sha256 != tensor_content_sha256(self.rows):
            raise ValueError("Qwen payload tensor hash differs")
        if self.selected_indices_sha256 != selected_indices_sha256(self.keeps):
            raise ValueError("Qwen payload selected-index hash differs")

    def _validate_geometry(self) -> None:
        """Validate the keeps and the row count under the layout that names them.

        A flat payload is every worker's rows concatenated; a chain payload is
        the last hop's block alone, with the earlier keeps as provenance.
        """
        if len(self.keeps) != self.profile.workers_per_item:
            raise ValueError(
                f"Qwen payload must cover exactly {self.profile.workers_per_item} workers"
            )
        invalid_keep = any(
            tuple(sorted(set(keep))) != keep or any(position < 0 for position in keep)
            for keep in self.keeps
        )
        if invalid_keep:
            raise ValueError("Qwen payload keeps must be sorted, unique, and nonnegative")
        if self.rows_by_worker != tuple(len(keep) for keep in self.keeps):
            raise ValueError("Qwen payload worker geometry differs from its keeps")
        if self.layout == CHAIN_TERMINAL_LAYOUT:
            if self.rows_by_worker[-1] != int(self.rows.shape[0]):
                raise ValueError("chain terminal payload rows differ from the last hop's keep")
        elif sum(self.rows_by_worker) != int(self.rows.shape[0]):
            raise ValueError("Qwen payload row count differs from its worker geometry")


def build_flat_payload(
    captures: Sequence[QwenWorkerCapture],
    *,
    semantic_arm: str,
    embed: Callable[[list[int]], torch.Tensor],
    family: RouteFamily,
) -> QwenFlatPayload:
    """Materialize the canonical three-worker flat-interleave payload."""
    if len(captures) != QWEN_WORKERS_PER_ITEM:
        raise ValueError("Qwen flat payload needs exactly three worker captures")
    selector, ratio = _latent_selection(semantic_arm, family=family)
    if any(capture.latent_steps != FANOUTQA_NATURAL_DEV50.latent_steps for capture in captures):
        raise ValueError("Qwen production payload requires the registered 40-step roll")
    adapter = family.score_adapter(semantic_arm)
    keep_roster = tuple(capture.select(selector, ratio, adapter) for capture in captures)
    prompt_ids = tuple(
        torch.tensor(capture.prompt_token_ids, dtype=torch.long) for capture in captures
    )
    prompt_lengths = tuple(len(capture.prompt_token_ids) for capture in captures)
    rolled = tuple(
        capture.embedding_rows[0, prompt_length:].detach()
        for capture, prompt_length in zip(captures, prompt_lengths, strict=True)
    )
    blocks, _kinds, stats = interleaved_blocks(
        prompt_ids,
        flat_interleave_layout(keep_roster, prompt_lengths),
        keep_roster,
        rolled,
        embed,
    )
    if (
        len(blocks) != QWEN_WORKERS_PER_ITEM
        or stats["delimiter_rows"] != 0
        or stats["tier_rows"] != 0
    ):
        raise RuntimeError("canonical Qwen flat-interleave materialization changed")
    widths = {int(block.shape[1]) for block in blocks}
    if len(widths) != 1:
        raise ValueError("Qwen worker embedding widths differ")
    if tuple(int(block.shape[0]) for block in blocks) != tuple(map(len, keep_roster)):
        raise RuntimeError("canonical Qwen flat-interleave rows differ from the exact keeps")
    rows = torch.cat([block.detach().cpu() for block in blocks], dim=0).to(torch.bfloat16)
    return QwenFlatPayload(
        rows=rows,
        semantic_arm=semantic_arm,
        latent_plan_sha256=latent_plan_sha256(semantic_arm, family=family),
        keeps=keep_roster,
        rows_by_worker=tuple(len(keep) for keep in keep_roster),
        selected_indices_sha256=selected_indices_sha256(keep_roster),
        tensor_sha256=tensor_content_sha256(rows),
        family=family,
        profile=FANOUTQA_NATURAL_DEV50,
        layout=FANOUTQA_NATURAL_DEV50.payload_layout,
    )


__all__ = (
    "QWEN_CAPTURE_SELECTORS",
    "QWEN_LATENT_STEPS",
    "QWEN_PAYLOAD_LAYOUT",
    "QWEN_RATIOS",
    "QWEN_SPAN_WIDTH",
    "QWEN_SUPPORT_SELECTOR",
    "QWEN_WORKERS_PER_ITEM",
    "QwenFlatPayload",
    "QwenWorkerCapture",
    "build_flat_payload",
    "capture_qwen_scores",
    "chain_plan_sha256",
    "latent_plan_sha256",
    "payload_plan_sha256",
    "qwen_w16_keep",
    "registered_selection",
    "selected_indices_sha256",
    "tensor_content_sha256",
)
