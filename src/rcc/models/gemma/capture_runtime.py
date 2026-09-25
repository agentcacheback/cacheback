"""Resident-engine capture phase for the Gemma 4 FanOutQA fleet lane."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import torch

from rcc.benchmarks.fanoutqa.gemma_data import RollFrame
from rcc.latent.rollout import Realign, latent_rollout
from rcc.models.gemma.capture_support import (
    LATENT_ROLL_SCHEMA,
    DeferredAudit,
    ResidentCaptureBridge,
    atomic_torch_save,
    durable_qsnap_score,
    durable_support_score,
    latent_roll_filename,
    load_rolled,
    pinned_rows,
    replay_worker,
    require_bank_geometry,
    validate_scores,
)
from rcc.models.gemma.capture_types import CaptureArtifact
from rcc.models.gemma.contract import (
    ARM_POLICIES,
    ARM_SELECTORS,
    ARMS,
    LATENT_POLICIES,
    LATENT_REALIGN_ENABLED,
    LATENT_STEPS,
    POOL_KERNEL,
    QSNAP_ARMS,
    SELECTORS,
)
from rcc.models.gemma.contract import (
    EMBEDDING_SHAPE as EMBEDDING_SHAPE,
)
from rcc.models.gemma.embedding import publish_embedding
from rcc.models.gemma.mechanism import (
    layer_bank_filename,
    require_rolling_cache,
    token_digests,
)
from rcc.models.gemma.recurrence import build_native_realign
from rcc.models.gemma.selection import (
    load_or_bank_qsnap_selection,
    load_or_bank_selection,
    payload_length,
)
from rcc.run.gemma.cell_banks import ShardLog
from rcc.transforms.select.core import kernels
from rcc.transforms.select.core.types import CaptureSpec
from rcc.transforms.select.query_support import capture_query_support
from rcc.transforms.select.query_support.methods.compose import SUPPORT_DEFAULT_ORDER


def _global_layer_indices(model: Any) -> tuple[int, ...]:
    config = cast(Any, kernels.get_backbone(model)).config
    schedule = getattr(config, "layer_types", None)
    if schedule is None:
        return tuple(range(int(config.num_hidden_layers)))
    return tuple(index for index, value in enumerate(schedule) if "sliding" not in str(value))


def _stage_clock(device: torch.device) -> Callable[[], float]:
    """Return a clock that fences the device first, as the Qwen engine does."""
    sync = device.type == "cuda"

    def now() -> float:
        if sync:
            torch.cuda.synchronize(device)
        return time.perf_counter()

    return now


def _capture_worker(
    bridge: ResidentCaptureBridge,
    item: dict[str, Any],
    worker: int,
    realign: Realign,
    frame: RollFrame,
    *,
    include_qsnap: bool = False,
) -> dict[str, Any]:
    model = bridge.model
    device = next(model.parameters()).device
    memory_ids = item["memory_ids"][worker].to(device=device)
    judger_ids = item["judger_ids"].to(device=device)
    question_ids = item["question_ids"].to(device=device)
    prompt_ids = item["prompt_ids"][worker].to(device=device)
    if not torch.equal(prompt_ids, frame.prompt(memory_ids)):
        raise RuntimeError(f"{item['qid']}/w{worker}: the item's framed prompt is not this frame")
    prefix_length = int(prompt_ids.numel()) - 1
    now = _stage_clock(device)
    started = now()
    past = bridge.extract(prompt_ids[:-1].tolist())
    extract_s = now() - started
    extracted_length = int(past.get_seq_length())
    if extracted_length != prefix_length:
        raise RuntimeError(
            f"{item['qid']}/w{worker}: extracted cache has {extracted_length} "
            f"rows, expected {prefix_length}"
        )
    started = now()
    roll = latent_rollout(
        model,
        prompt_ids[-1:].unsqueeze(0),
        latent_steps=LATENT_STEPS,
        realign=realign,
        past=past,
        record_embeds=True,
    )
    past = roll.past
    require_rolling_cache(past, model)
    roll_s = now() - started
    if roll.embeds is None:
        raise RuntimeError(f"{item['qid']}/w{worker}: latent roll recorded no embeddings")
    rolled = roll.embeds[0, -LATENT_STEPS:].detach().to(dtype=torch.bfloat16).cpu()
    # Live-model width so the tiny-model audit runs this path unmodified.
    hidden = int(model.get_input_embeddings().weight.shape[1])
    if rolled.shape != (LATENT_STEPS, hidden):
        raise RuntimeError(f"{item['qid']}/w{worker}: rolled tail shape is {tuple(rolled.shape)}")
    started = now()
    result = capture_query_support(
        model,
        past,
        judger_ids,
        torch.ones_like(judger_ids),
        question_ids,
        pool_kernel=POOL_KERNEL,
        spec=CaptureSpec(
            energy_pool_kernel=POOL_KERNEL,
            row_energy_pool_kernel=POOL_KERNEL,
            support_orders=(float(SUPPORT_DEFAULT_ORDER),),
            collect_statistics=True,
            collect_layer_bank=True,
            bank_include_sliding=True,
        ),
    )
    tag = f"{item['qid']}/w{worker}"
    if not result.found or result.moments is None or result.layer_bank is None:
        raise RuntimeError(f"{tag}: Query-Support capture is incomplete")
    bank = result.layer_bank
    if not bank.selected_rows():
        raise RuntimeError(f"{tag}: layer bank contains no selected rows")
    require_bank_geometry(tag, bank)
    capture_query_s = now() - started
    payload = payload_length(int(prompt_ids.numel()))
    started = now()
    scores = {
        "support": durable_support_score(
            tag,
            bank,
            result.moments,
            device=device,
            payload=payload,
        )
    }
    if include_qsnap:
        scores["snap"] = durable_qsnap_score(
            tag,
            bank,
            result.snap,
            payload=payload,
        )
    for arm, selector in ARM_SELECTORS.items():
        if selector in scores:
            continue
        scores[selector] = pinned_rows(
            tag,
            bank.replay_score(ARM_POLICIES[arm]).float().cpu(),
            payload,
        )
    expected_selectors: set[str] = set(SELECTORS)
    if include_qsnap:
        expected_selectors.add("snap")
    if set(scores) != expected_selectors:
        raise RuntimeError(f"{tag}: capture produced selector roster {sorted(scores)}")
    validate_scores(tag, scores, payload, expected_selectors=expected_selectors)
    parity_s = now() - started
    return {
        **scores,
        "_layer_bank": bank,
        "_rolled": rolled,
        "extract_s": extract_s,
        "roll_s": roll_s,
        "capture_query_s": capture_query_s,
        "parity_s": parity_s,
    }


def _capture_or_replay_worker(
    bridge: ResidentCaptureBridge,
    item: dict[str, Any],
    worker: int,
    *,
    banks: Path,
    shard_log: ShardLog,
    realign: Realign,
    frame: RollFrame,
    include_qsnap: bool = False,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, float, bool, DeferredAudit | None]:
    """Return scores and cargo with the original live clock, never a replay clock."""
    model = bridge.model
    banked = banks / layer_bank_filename(item["qid"], worker)
    rolled_path = banks.parent / "rolled" / latent_roll_filename(item["qid"], worker)
    timing = shard_log.banked_capture(item["qid"], worker)
    if timing is not None and not banked.is_file():
        raise RuntimeError(f"{item['qid']}/w{worker}: capture timing exists without its layer bank")
    if banked.is_file() and timing is not None:
        if timing.get("layer_bank_file") != banked.name:
            raise RuntimeError(f"{item['qid']}/w{worker}: capture timing names another layer bank")
        if timing.get("latent_roll_file") != rolled_path.name or not rolled_path.is_file():
            raise RuntimeError(f"{item['qid']}/w{worker}: capture timing has no latent roll")
        capture_s = timing.get("capture_s")
        if isinstance(capture_s, bool) or not isinstance(capture_s, (int, float)):
            raise RuntimeError(f"{item['qid']}/w{worker}: capture timing is malformed")
        if float(capture_s) < 0:
            raise RuntimeError(f"{item['qid']}/w{worker}: capture timing is negative")
        hidden = int(model.get_input_embeddings().weight.shape[1])
        rolled = load_rolled(
            rolled_path,
            qid=item["qid"],
            worker=worker,
            hidden=hidden,
            frame=frame,
        )
        return (
            replay_worker(
                item,
                worker,
                banked,
                **({"include_qsnap": True} if include_qsnap else {}),
            ),
            rolled,
            float(capture_s),
            True,
            None,
        )
    started = time.perf_counter()
    captured = _capture_worker(
        bridge,
        item,
        worker,
        realign,
        frame,
        **({"include_qsnap": True} if include_qsnap else {}),
    )
    capture_s = round(time.perf_counter() - started, 4)
    bank = captured.pop("_layer_bank")
    rolled = cast(torch.Tensor, captured.pop("_rolled"))
    clocks = {
        name: round(float(captured.pop(name)), 4)
        for name in ("extract_s", "roll_s", "capture_query_s", "parity_s")
    }
    if not rolled_path.is_file():
        atomic_torch_save(
            {
                "schema": LATENT_ROLL_SCHEMA,
                "qid": item["qid"],
                "worker": worker,
                "latent_steps": LATENT_STEPS,
                "realign_enabled": LATENT_REALIGN_ENABLED,
                "frame_prefix_length": len(frame.prefix_ids),
                "frame_suffix_length": len(frame.suffix_ids),
                "rolled": rolled,
            },
            rolled_path,
        )
    audit = DeferredAudit(
        item=item,
        worker=worker,
        bank=bank,
        scores=captured,
        rolled=rolled,
        bank_path=banked,
        rolled_path=rolled_path,
        frame=frame,
        shard_log=shard_log,
        capture_s=capture_s,
        include_qsnap=include_qsnap,
        **clocks,
    )
    return captured, rolled, capture_s, False, audit


def capture_panel(
    items: tuple[dict[str, Any], ...],
    *,
    bridge: ResidentCaptureBridge,
    artifact_root: Path,
    shard_log: ShardLog,
    worker_index: int,
    runtime_fingerprint: str,
    publication_id: str,
    shared_weight: Path,
    include_qsnap: bool = False,
    flat_interleave_arms: Sequence[str] = (),
    selection_arms: Sequence[str] = ARMS,
) -> tuple[tuple[CaptureArtifact, ...], dict[str, str], tuple[DeferredAudit, ...]]:
    """Capture every assigned item through the borrowed resident engine view."""
    selected = tuple(selection_arms)
    if len(set(selected)) != len(selected) or any(arm not in ARMS for arm in selected):
        raise RuntimeError("capture selection roster must name unique registered v16 arms")
    flat_arms = tuple(flat_interleave_arms)
    allowed_flat_arms = set(LATENT_POLICIES) | set(QSNAP_ARMS)
    if len(set(flat_arms)) != len(flat_arms) or any(
        arm not in allowed_flat_arms for arm in flat_arms
    ):
        raise RuntimeError("Qwen-flat capture overrides must name unique latent base arms")
    if set(flat_arms) & set(QSNAP_ARMS) and not include_qsnap:
        raise RuntimeError(
            "Qwen-flat mean query attention layout requires mean query attention capture"
        )
    selected_flat_arms = tuple(arm for arm in flat_arms if arm in LATENT_POLICIES)
    if not set(selected_flat_arms).issubset(selected):
        raise RuntimeError("Qwen-flat Query-Support arms must be in the selection roster")
    model = bridge.model
    embedding = publish_embedding(
        model,
        shared_weight,
        worker_index=worker_index,
        runtime_fingerprint=runtime_fingerprint,
        publication_id=publication_id,
    )
    realign = build_native_realign(model)
    artifacts: list[CaptureArtifact] = []
    audits: list[DeferredAudit] = []
    for item in items:
        frame = item["frame"]
        scores_by_worker: list[dict[str, torch.Tensor]] = []
        rolled_by_worker: list[torch.Tensor] = []
        capture_s_by_worker: list[float] = []
        banks = artifact_root / "layer_banks"
        reloaded_captures = 0
        for worker in range(3):
            scores, rolled, capture_s, reloaded, audit = _capture_or_replay_worker(
                bridge,
                item,
                worker,
                banks=banks,
                shard_log=shard_log,
                realign=realign,
                frame=frame,
                **({"include_qsnap": True} if include_qsnap else {}),
            )
            scores_by_worker.append(scores)
            rolled_by_worker.append(rolled)
            capture_s_by_worker.append(capture_s)
            reloaded_captures += int(reloaded)
            if audit is not None:
                audits.append(audit)
        audition_s_by_ratio: dict[int, float] = {}
        audition_replays_by_ratio: dict[int, int] = {}
        selection = load_or_bank_selection(
            item,
            scores_by_worker,
            artifact_root / "selections" / f"{item['qid']}.json",
            seat_keys_by_ratio={},
            ratio_cuts={},
            flat_interleave_arms=selected_flat_arms,
            arms=selected,
        )
        qsnap_selection = (
            load_or_bank_qsnap_selection(
                item,
                tuple({"snap": scores["snap"]} for scores in scores_by_worker),
                artifact_root / "qsnap_selections" / f"{item['qid']}.json",
                flat_interleave=bool(set(flat_arms) & set(QSNAP_ARMS)),
            )
            if include_qsnap
            else None
        )
        digests = token_digests(model, torch.tensor(item["request_ids"], dtype=torch.long))
        artifacts.append(
            CaptureArtifact(
                qid=item["qid"],
                keeps_by_arm={
                    **selection.keeps_by_arm,
                    **(qsnap_selection.keeps_by_arm if qsnap_selection is not None else {}),
                },
                layouts_by_arm={
                    **selection.layouts_by_arm,
                    **(qsnap_selection.layouts_by_arm if qsnap_selection is not None else {}),
                },
                rolled_by_worker=tuple(rolled_by_worker),
                selection_s_by_arm={
                    **selection.selection_s_by_arm,
                    **(qsnap_selection.selection_s_by_arm if qsnap_selection is not None else {}),
                },
                audition_s_by_ratio=audition_s_by_ratio,
                audition_replays_by_ratio=audition_replays_by_ratio,
                reports=(),
                report_failed=False,
                report_failure=None,
                capture_s=round(sum(capture_s_by_worker), 4),
                capture_s_by_worker=tuple(round(value, 4) for value in capture_s_by_worker),
                report_generation_s=0.0,
                embedding_digests=digests,
                global_layers=_global_layer_indices(model),
                reloaded_captures=reloaded_captures,
                reloaded_selections=int(selection.reloaded)
                + int(qsnap_selection.reloaded if qsnap_selection is not None else False),
            )
        )
    return tuple(artifacts), embedding, tuple(audits)


__all__ = (
    "LATENT_ROLL_SCHEMA",
    "CaptureArtifact",
    "ResidentCaptureBridge",
    "capture_panel",
    "latent_roll_filename",
    "publish_embedding",
)
