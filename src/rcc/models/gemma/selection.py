"""HF-phase selection artifacts for the Gemma FanOutQA fleet lane.

The capture path derives them while the HF view is live and publishes them to
disk; the handoff path reloads them and checks they replay.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from rcc.models.gemma.contract import (
    ARM_POLICIES,
    ARM_RATIOS,
    ARM_SELECTORS,
    ARMS,
    CUT_ARMS,
    LATENT_POLICIES,
    LATENT_STEPS,
    QSNAP_ARM_POLICIES,
    QSNAP_ARM_RATIOS,
    QSNAP_ARMS,
    RATIOS,
    SELECTION_SCHEMA,
    SELECTOR_FAMILY,
    SPAN_WIDTH,
    WORKERS_PER_ITEM,
)
from rcc.models.gemma.layout import (
    ArmLayout,
    flat_interleave_layout,
    validate_layout,
)
from rcc.models.gemma.mechanism import protected_positions
from rcc.run import io
from rcc.transforms.select.query_support.fixed_spans import fixed_span_keep, locbench_rn_keep

SeatKeys = Mapping[tuple[int, int], tuple[float, float]]
SeatKeysByRatio = Mapping[int, SeatKeys]
Keeps = tuple[tuple[int, ...], ...]
QSNAP_SELECTION_SCHEMA = "gemma-shared-fanoutqa-qsnap-selection-v1"


@dataclass(frozen=True)
class SelectionArtifact:
    """Durable per-arm keeps, tiered layouts, and original HF-phase clocks."""

    keeps_by_arm: dict[str, Keeps]
    layouts_by_arm: dict[str, ArmLayout]
    selection_s_by_arm: dict[str, float]
    reloaded: bool


@dataclass(frozen=True)
class RatioCut:
    """One ratio's keep and the clock every arm that ships it pays."""

    keeps: Keeps
    selection_s: float


def payload_length(prompt_tokens: int) -> int:
    """One worker's scored payload: its whole framed prompt plus the latent tail.

    The cut runs over every rolled cache row, so frame rows compete with
    evidence rows and can ship.
    """
    return prompt_tokens + LATENT_STEPS


def worker_keep(arm: str, scores: torch.Tensor, payload: int) -> list[int]:
    """Return one worker's registered keep over the payload length.

    The sink and the rolled tail are always protected; on a worker too
    short for the exact budget, protection outranks the ratio.
    """
    if arm == "floor":
        return []
    if arm == "full":
        return list(range(payload))
    cut_arm = arm
    expected_policy = (
        f"{SELECTOR_FAMILY}_r{ARM_RATIOS[cut_arm]}_w{SPAN_WIDTH}_{ARM_SELECTORS[cut_arm]}"
    )
    if expected_policy != ARM_POLICIES[cut_arm]:
        raise RuntimeError(f"{cut_arm}: registered policy name drifted")
    protected = protected_positions(payload, LATENT_STEPS)
    budget = -(-payload // ARM_RATIOS[cut_arm])
    if budget < len(protected):
        return fixed_span_keep(scores, len(protected), protected, span_size=SPAN_WIDTH)
    # The registered keep law: a strict fixed-width span grid, the same one the
    # mean query attention path below runs.
    return locbench_rn_keep(
        scores,
        float(ARM_RATIOS[cut_arm]),
        protected=protected,
        span_size=SPAN_WIDTH,
    )


def qsnap_worker_keep(arm: str, scores: torch.Tensor, payload: int) -> list[int]:
    """Run the canonical Gemma mean query attention W16 keep for an opt-in shared arm."""
    try:
        ratio = QSNAP_ARM_RATIOS[arm]
        registered_policy = QSNAP_ARM_POLICIES[arm]
    except KeyError as exc:
        raise ValueError(f"{arm}: not a registered Gemma mean query attention arm") from exc
    if scores.shape != (payload,):
        raise RuntimeError(f"{arm}: mean query attention score length differs from the payload")
    expected_policy = f"{SELECTOR_FAMILY}_r{ratio}_w{SPAN_WIDTH}"
    if expected_policy != registered_policy:
        raise RuntimeError(f"{arm}: registered mean query attention policy name drifted")
    return locbench_rn_keep(
        scores,
        float(ratio),
        protected=protected_positions(payload, LATENT_STEPS),
        span_size=SPAN_WIDTH,
    )


def _derive_qsnap_selection(
    item: dict[str, Any],
    scores_by_worker: Sequence[dict[str, torch.Tensor]],
    *,
    flat_interleave: bool = True,
) -> SelectionArtifact:
    """Derive the opt-in mean query attention ladder from canonical captured snap."""
    qid = str(item["qid"])
    if len(scores_by_worker) != WORKERS_PER_ITEM or any(
        set(scores) != {"snap"} for scores in scores_by_worker
    ):
        raise RuntimeError(f"{qid}: mean query attention score worker roster is incomplete")
    lengths = tuple(payload_length(int(ids.numel())) for ids in item["prompt_ids"])
    prompt_tokens = tuple(int(ids.numel()) for ids in item["prompt_ids"])
    keeps_by_arm: dict[str, Keeps] = {}
    layouts_by_arm: dict[str, ArmLayout] = {}
    clocks: dict[str, float] = {}
    for arm in QSNAP_ARMS:
        started = time.perf_counter()
        keeps = tuple(
            tuple(qsnap_worker_keep(arm, scores_by_worker[worker]["snap"], length))
            for worker, length in enumerate(lengths)
        )
        if not flat_interleave:
            raise RuntimeError(
                f"{qid}/{arm}: mean query attention selection requires flat-interleave"
            )
        layout = flat_interleave_layout(keeps, prompt_tokens)
        try:
            validate_layout(layout, keeps, prompt_tokens)
        except ValueError as exc:
            raise RuntimeError(f"{qid}/{arm}: {exc}") from exc
        keeps_by_arm[arm] = keeps
        layouts_by_arm[arm] = layout
        clocks[arm] = round(time.perf_counter() - started, 6)
    return SelectionArtifact(keeps_by_arm, layouts_by_arm, clocks, False)


def load_qsnap_selection(path: Path, *, qid: str) -> SelectionArtifact:
    """Load one banked mean query attention artifact and re-run its roster validation."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload["schema"] != QSNAP_SELECTION_SCHEMA
            or payload["qid"] != qid
            or payload["arm_policies"] != QSNAP_ARM_POLICIES
        ):
            raise RuntimeError("mean query attention selection identity differs")
        keeps = {
            str(arm): tuple(tuple(int(value) for value in worker) for worker in workers)
            for arm, workers in payload["keeps_by_arm"].items()
        }
        layouts = {
            str(arm): ArmLayout(
                span_order=tuple((int(worker), int(block)) for worker, block in body["span_order"]),
                tier_start=int(body["tier_start"]),
                cargo_first=bool(body["cargo_first"]),
            )
            for arm, body in payload["layouts_by_arm"].items()
        }
        clocks = {str(arm): float(value) for arm, value in payload["selection_s_by_arm"].items()}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{qid}: mean query attention selection artifact is malformed") from exc
    if (
        set(keeps) != set(QSNAP_ARMS)
        or set(layouts) != set(QSNAP_ARMS)
        or set(clocks) != set(QSNAP_ARMS)
    ):
        raise RuntimeError(
            f"{qid}: mean query attention selection artifact has the wrong arm roster"
        )
    if any(clock < 0 for clock in clocks.values()):
        raise RuntimeError(f"{qid}: mean query attention selection artifact has an invalid clock")
    return SelectionArtifact(keeps, layouts, clocks, True)


def load_or_bank_qsnap_selection(
    item: dict[str, Any],
    scores_by_worker: Sequence[dict[str, torch.Tensor]],
    path: Path,
    *,
    flat_interleave: bool = True,
) -> SelectionArtifact:
    """Bank exact mean query attention indices/layouts without changing the live selection bank."""
    qid = str(item["qid"])
    derived = _derive_qsnap_selection(item, scores_by_worker, flat_interleave=flat_interleave)
    reloaded = path.is_file()
    if not reloaded:
        io.atomic_json(
            path,
            {
                "schema": QSNAP_SELECTION_SCHEMA,
                "qid": qid,
                "arm_policies": QSNAP_ARM_POLICIES,
                "keeps_by_arm": {
                    arm: [list(worker) for worker in derived.keeps_by_arm[arm]]
                    for arm in QSNAP_ARMS
                },
                "layouts_by_arm": {
                    arm: {
                        "span_order": [
                            list(span) for span in derived.layouts_by_arm[arm].span_order
                        ],
                        "tier_start": derived.layouts_by_arm[arm].tier_start,
                        "cargo_first": derived.layouts_by_arm[arm].cargo_first,
                    }
                    for arm in QSNAP_ARMS
                },
                "selection_s_by_arm": derived.selection_s_by_arm,
            },
        )
    stored = load_qsnap_selection(path, qid=qid)
    if stored.keeps_by_arm != derived.keeps_by_arm:
        raise RuntimeError(
            f"{qid}: durable mean query attention indices differ from canonical replay"
        )
    if stored.layouts_by_arm != derived.layouts_by_arm:
        raise RuntimeError(
            f"{qid}: durable mean query attention layout differs from canonical replay"
        )
    return SelectionArtifact(
        stored.keeps_by_arm,
        stored.layouts_by_arm,
        stored.selection_s_by_arm,
        reloaded,
    )


def cut_ratio(
    ratio: int,
    scores_by_worker: Sequence[dict[str, torch.Tensor]],
    lengths: Sequence[int],
) -> RatioCut:
    """Run one ratio's cut once, on its own clock, for every arm that ships it."""
    arm = f"r{ratio}"
    if arm not in CUT_ARMS or ARM_RATIOS[arm] != ratio:
        raise ValueError(f"{arm}: not a registered ratio arm")
    selector = ARM_SELECTORS[arm]
    started = time.perf_counter()
    keeps = tuple(
        tuple(worker_keep(arm, scores_by_worker[worker][selector], length))
        for worker, length in enumerate(lengths)
    )
    return RatioCut(keeps, round(time.perf_counter() - started, 6))


def _validate(
    qid: str,
    lengths: Sequence[int],
    keeps_by_arm: dict[str, Keeps],
    layouts_by_arm: dict[str, ArmLayout],
    selection_s_by_arm: dict[str, float],
    *,
    arms: Sequence[str] = ARMS,
) -> None:
    selected = tuple(arms)
    if len(set(selected)) != len(selected) or any(arm not in ARMS for arm in selected):
        raise RuntimeError(f"{qid}: selection roster must name unique registered arms")
    if set(keeps_by_arm) != set(selected) or set(selection_s_by_arm) != set(selected):
        raise RuntimeError(f"{qid}: selection artifact does not cover its execution arms")
    expected_layouts = {arm for arm in selected if arm in CUT_ARMS}
    if set(layouts_by_arm) != expected_layouts:
        raise RuntimeError(f"{qid}: layout roster differs from the cut arms")
    prompt_tokens = tuple(length - LATENT_STEPS for length in lengths)
    for arm, layout in layouts_by_arm.items():
        try:
            validate_layout(layout, keeps_by_arm[arm], prompt_tokens)
        except ValueError as exc:
            raise RuntimeError(f"{qid}/{arm}: {exc}") from exc
    if len(lengths) != WORKERS_PER_ITEM:
        raise RuntimeError(f"{qid}: selection artifact does not cover three workers")
    for arm in selected:
        workers = keeps_by_arm[arm]
        if len(workers) != WORKERS_PER_ITEM:
            raise RuntimeError(f"{qid}/{arm}: selection worker roster is incomplete")
        for worker, (keep, length) in enumerate(zip(workers, lengths, strict=True)):
            if tuple(sorted(set(keep))) != keep:
                raise RuntimeError(f"{qid}/{arm}/w{worker}: keep is not sorted and unique")
            if any(position < 0 or position >= length for position in keep):
                raise RuntimeError(f"{qid}/{arm}/w{worker}: keep is outside the worker payload")
            if arm == "floor" and keep:
                raise RuntimeError(f"{qid}/{arm}/w{worker}: baseline keep is not empty")
        clock = selection_s_by_arm[arm]
        if isinstance(clock, bool) or clock < 0:
            raise RuntimeError(f"{qid}/{arm}: selection clock is invalid")
        if arm not in CUT_ARMS and float(clock) != 0.0:
            raise RuntimeError(f"{qid}/{arm}: baseline carries a selection clock")


def _arm_layout_for(
    arm: str,
    keeps: Keeps,
    score_by_worker: tuple[torch.Tensor, ...],
    prompt_tokens: tuple[int, ...],
    seat_keys_by_ratio: SeatKeysByRatio,
    *,
    flat_interleave: bool = False,
) -> ArmLayout:
    """Return the live split payload's source-order flat layout."""
    del score_by_worker, seat_keys_by_ratio
    if not flat_interleave:
        raise RuntimeError(f"{arm}: live Gemma selection requires flat-interleave")
    return flat_interleave_layout(keeps, prompt_tokens)


def derive_selection(
    item: dict[str, Any],
    scores_by_worker: Sequence[dict[str, torch.Tensor]],
    *,
    seat_keys_by_ratio: SeatKeysByRatio,
    ratio_cuts: Mapping[int, RatioCut],
    flat_interleave_arms: Sequence[str] = (),
    arms: Sequence[str] = ARMS,
) -> SelectionArtifact:
    """Run every live selector while the HF capture phase is live."""
    qid = str(item["qid"])
    selected = tuple(arms)
    if len(set(selected)) != len(selected) or any(arm not in ARMS for arm in selected):
        raise RuntimeError(f"{qid}: selection roster must name unique registered arms")
    lengths = tuple(payload_length(int(ids.numel())) for ids in item["prompt_ids"])
    if len(scores_by_worker) != WORKERS_PER_ITEM:
        raise RuntimeError(f"{qid}: score worker roster is incomplete")
    flat_arms = tuple(flat_interleave_arms) or tuple(
        arm for arm in selected if arm in LATENT_POLICIES
    )
    if len(set(flat_arms)) != len(flat_arms) or any(
        arm not in LATENT_POLICIES or arm not in selected for arm in flat_arms
    ):
        raise RuntimeError(f"{qid}: Qwen-flat overrides must name selected latent base arms")
    cuts = {f"r{ratio}": cut for ratio, cut in ratio_cuts.items() if ratio in RATIOS}
    prompt_tokens = tuple(int(ids.numel()) for ids in item["prompt_ids"])
    keeps_by_arm: dict[str, Keeps] = {}
    layouts_by_arm: dict[str, ArmLayout] = {}
    selection_s_by_arm: dict[str, float] = {}
    for arm in selected:
        base = arm
        if arm in CUT_ARMS and base not in cuts:
            cuts[base] = cut_ratio(ARM_RATIOS[base], scores_by_worker, lengths)
        # The shared cut carries its own clock, so the arm clock starts here.
        started = time.perf_counter()
        if arm in CUT_ARMS:
            keeps = cuts[base].keeps
            cut_s = cuts[base].selection_s
            layouts_by_arm[arm] = _arm_layout_for(
                arm,
                keeps,
                tuple(
                    scores_by_worker[worker][ARM_SELECTORS[arm]] for worker in range(len(lengths))
                ),
                prompt_tokens,
                seat_keys_by_ratio,
                flat_interleave=arm in flat_arms,
            )
        else:
            keeps = tuple(tuple(worker_keep(arm, torch.empty(0), length)) for length in lengths)
            cut_s = 0.0
        keeps_by_arm[arm] = keeps
        selection_s_by_arm[arm] = (
            round(cut_s + time.perf_counter() - started, 6) if arm in CUT_ARMS else 0.0
        )
    _validate(
        qid,
        lengths,
        keeps_by_arm,
        layouts_by_arm,
        selection_s_by_arm,
        arms=selected,
    )
    return SelectionArtifact(keeps_by_arm, layouts_by_arm, selection_s_by_arm, False)


def load_selection(
    path: Path,
    *,
    qid: str,
    lengths: Sequence[int],
    arms: Sequence[str] = ARMS,
) -> SelectionArtifact:
    """Load one banked selection and re-run every registered keep validation."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        selected = tuple(arms)
        stored_arms = tuple(payload.get("selection_arms", ARMS))
        if (
            payload["schema"] != SELECTION_SCHEMA
            or payload["qid"] != qid
            or stored_arms != selected
        ):
            raise RuntimeError("selection identity differs")
        keeps = {
            str(arm): tuple(tuple(int(value) for value in worker) for worker in workers)
            for arm, workers in payload["keeps_by_arm"].items()
        }
        layouts = {
            str(arm): ArmLayout(
                span_order=tuple((int(worker), int(block)) for worker, block in body["span_order"]),
                tier_start=int(body["tier_start"]),
                cargo_first=bool(body["cargo_first"]),
            )
            for arm, body in payload["layouts_by_arm"].items()
        }
        clocks = {str(arm): float(value) for arm, value in payload["selection_s_by_arm"].items()}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{qid}: selection artifact is malformed") from exc
    _validate(qid, lengths, keeps, layouts, clocks, arms=selected)
    return SelectionArtifact(keeps, layouts, clocks, True)


def load_or_bank_selection(
    item: dict[str, Any],
    scores_by_worker: Sequence[dict[str, torch.Tensor]],
    path: Path,
    *,
    seat_keys_by_ratio: SeatKeysByRatio,
    ratio_cuts: Mapping[int, RatioCut],
    flat_interleave_arms: Sequence[str] = (),
    arms: Sequence[str] = ARMS,
) -> SelectionArtifact:
    """Bank new keeps or restore their original clocks and verify exact replay."""
    qid = str(item["qid"])
    selected = tuple(arms)
    lengths = tuple(payload_length(int(ids.numel())) for ids in item["prompt_ids"])
    derived = derive_selection(
        item,
        scores_by_worker,
        seat_keys_by_ratio=seat_keys_by_ratio,
        ratio_cuts=ratio_cuts,
        flat_interleave_arms=flat_interleave_arms,
        arms=selected,
    )
    if path.is_file():
        stored = load_selection(path, qid=qid, lengths=lengths, arms=selected)
        if stored.keeps_by_arm != derived.keeps_by_arm:
            raise RuntimeError(f"{qid}: durable selection differs from HF replay")
        if stored.layouts_by_arm != derived.layouts_by_arm:
            raise RuntimeError(f"{qid}: durable layout differs from HF replay")
        return stored
    payload: dict[str, object] = {
        "schema": SELECTION_SCHEMA,
        "qid": qid,
        "keeps_by_arm": {
            arm: [list(worker) for worker in derived.keeps_by_arm[arm]] for arm in selected
        },
        "layouts_by_arm": {
            arm: {
                "span_order": [list(span) for span in layout.span_order],
                "tier_start": layout.tier_start,
                "cargo_first": layout.cargo_first,
            }
            for arm, layout in derived.layouts_by_arm.items()
        },
        "selection_s_by_arm": derived.selection_s_by_arm,
    }
    if selected != ARMS:
        payload["selection_arms"] = list(selected)
    io.atomic_json(path, payload)
    stored = load_selection(path, qid=qid, lengths=lengths, arms=selected)
    if stored.keeps_by_arm != derived.keeps_by_arm:
        raise RuntimeError(f"{qid}: newly banked selection differs from HF selection")
    if stored.layouts_by_arm != derived.layouts_by_arm:
        raise RuntimeError(f"{qid}: newly banked layout differs from HF layout")
    return SelectionArtifact(
        stored.keeps_by_arm,
        stored.layouts_by_arm,
        stored.selection_s_by_arm,
        False,
    )


__all__ = (
    "Keeps",
    "RatioCut",
    "SeatKeys",
    "SeatKeysByRatio",
    "SelectionArtifact",
    "cut_ratio",
    "derive_selection",
    "load_or_bank_qsnap_selection",
    "load_or_bank_selection",
    "load_qsnap_selection",
    "load_selection",
    "payload_length",
    "qsnap_worker_keep",
    "worker_keep",
)
