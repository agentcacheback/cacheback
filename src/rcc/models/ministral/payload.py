"""Ministral W16 selection and embedding-only flat payloads."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from rcc.benchmarks.fanoutqa import (
    FANOUTQA_NATURAL_DEV50,
    SEALED_M3_PLAN_LABEL,
    arms_with_full,
)
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.ministral import MINISTRAL, MINISTRAL_RATIOS
from rcc.models.ministral.capture import LATENT_STEPS, MinistralWorkerCapture
from rcc.topologies.fanout import FANOUT_M3
from rcc.topologies.fanout.layout import (
    FLAT_INTERLEAVE_LAYOUT,
    SPAN_WIDTH,
    flat_interleave_layout,
    interleaved_blocks,
)
from rcc.transforms.select.query_support.fixed_spans import locbench_rn_keep

MINISTRAL_PAYLOAD_LAYOUT = FLAT_INTERLEAVE_LAYOUT
MINISTRAL_PAYLOAD_SCHEMA = "ministral-flat-payload-identity-v1"
MINISTRAL_SELECTED_ORDER = "worker-major-ascending-v1"


def tensor_content_sha256(tensor: torch.Tensor) -> str:
    """Hash one tensor's dtype, shape, and contiguous value bytes."""
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def selected_indices_sha256(keeps: Sequence[Sequence[int]]) -> str:
    """Hash the complete ordered keep roster without truncation."""
    normalized = tuple(tuple(int(position) for position in keep) for keep in keeps)
    encoded = json.dumps(normalized, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _latent_arm(semantic_arm: str) -> tuple[str, int]:
    """Resolve selector and ratio only from the registered semantic arm."""
    benchmark = next(
        (arm for arm in arms_with_full(FANOUTQA_NATURAL_DEV50) if arm.arm_id == semantic_arm),
        None,
    )
    physical = next(
        (arm for arm in MINISTRAL.physical_arms if arm.semantic_arm == semantic_arm),
        None,
    )
    if benchmark is None or physical is None or physical.payload_layout != MINISTRAL_PAYLOAD_LAYOUT:
        raise ValueError(f"unregistered Ministral flat payload arm {semantic_arm!r}")
    if semantic_arm == "full" and benchmark.channel in {"latent", "full"}:
        if benchmark.selector is not None or benchmark.retention_ratio is not None:
            raise ValueError("Ministral full arm must not carry selector or retention metadata")
        if physical.selector is not None:
            raise ValueError("Ministral full arm must not carry a selector")
        return "full", 1
    if (
        benchmark.channel != "latent"
        or benchmark.selector != "support"
        or benchmark.retention_ratio not in MINISTRAL_RATIOS
        or physical.selector != benchmark.selector
    ):
        raise ValueError(f"unregistered Ministral flat payload arm {semantic_arm!r}")
    return benchmark.selector, benchmark.retention_ratio


def latent_plan_sha256(semantic_arm: str) -> str:
    """Sign one Ministral latent arm and the canonical payload recipe."""
    _latent_arm(semantic_arm)
    benchmark = next(
        arm for arm in arms_with_full(FANOUTQA_NATURAL_DEV50) if arm.arm_id == semantic_arm
    )
    physical = next(arm for arm in MINISTRAL.physical_arms if arm.semantic_arm == semantic_arm)
    identity = {
        "schema": "ministral-flat-interleave-plan-v1",
        "benchmark_profile": SEALED_M3_PLAN_LABEL,
        "model_id": MINISTRAL.model_id,
        "topology": FANOUT_M3.to_dict(),
        "arm": benchmark.to_dict(),
        "physical_arm": physical.to_dict(),
        "latent_steps": LATENT_STEPS,
        "span_width": SPAN_WIDTH,
        "support_composition": "support-p2-a2",
        "payload_layout": MINISTRAL_PAYLOAD_LAYOUT,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def protected_positions(length: int, latent_steps: int = LATENT_STEPS) -> tuple[int, ...]:
    """Return sink zero plus every row in the protected latent tail."""
    if length < 1:
        raise ValueError("Ministral selection requires a nonempty payload")
    if latent_steps < 0 or latent_steps >= length:
        raise ValueError("Ministral latent tail is outside the payload")
    return (0, *range(max(1, length - latent_steps), length))


def ministral_w16_keep(
    scores: torch.Tensor,
    ratio: int,
    *,
    latent_steps: int = LATENT_STEPS,
) -> tuple[int, ...]:
    """Return exact-ratio W16 keeps with sink and all 40 latent rows protected."""
    if ratio not in MINISTRAL_RATIOS:
        raise ValueError(f"Ministral ratio must be one of {MINISTRAL_RATIOS}, got {ratio}")
    if scores.ndim != 1 or not bool(torch.isfinite(scores).all()):
        raise ValueError("Ministral selector scores must be one-dimensional and finite")
    protected = protected_positions(int(scores.numel()), latent_steps)
    budget = math.ceil(int(scores.numel()) / ratio)
    if budget < len(protected):
        raise ValueError("Ministral exact-ratio budget cannot contain its protected rows")
    keep = tuple(
        locbench_rn_keep(
            scores,
            float(ratio),
            protected=protected,
            span_size=SPAN_WIDTH,
        )
    )
    if len(keep) != budget or tuple(sorted(set(keep))) != keep:
        raise RuntimeError("Ministral W16 selection changed its exact budget or ordering")
    if not set(protected).issubset(keep):
        raise RuntimeError("Ministral W16 selection dropped protected sink or latent rows")
    return keep


def token_embedding_rows(
    embedding_weight: torch.Tensor,
    token_ids: Sequence[int],
) -> torch.Tensor:
    """Map Tekken ids through Ministral's measured identity-normalized table."""
    if embedding_weight.ndim != 2:
        raise ValueError("Ministral input embedding weight must be [vocab, hidden]")
    ids = tuple(int(token) for token in token_ids)
    if not ids:
        raise ValueError("Ministral embedding lookup requires at least one token")
    if min(ids) < 0 or max(ids) >= int(embedding_weight.shape[0]):
        raise ValueError("Ministral token id is outside the input embedding table")
    index = torch.tensor(ids, dtype=torch.long, device=embedding_weight.device)
    # The measured Ministral inputs_embeds convention has scale 1.0.
    return embedding_weight.index_select(0, index).detach().to(device="cpu", dtype=torch.bfloat16)


@dataclass(frozen=True)
class MinistralFlatPayload:
    """Signed embedding rows in flat worker and source order."""

    rows: torch.Tensor
    semantic_arm: str
    latent_plan_sha256: str
    keeps: tuple[tuple[int, ...], ...]
    rows_by_worker: tuple[int, ...]
    worker_sha256: tuple[str, ...]
    selected_indices_sha256: str
    tensor_sha256: str
    layout: str = MINISTRAL_PAYLOAD_LAYOUT
    schema: str = MINISTRAL_PAYLOAD_SCHEMA
    selected_indices_order: str = MINISTRAL_SELECTED_ORDER

    def __post_init__(self) -> None:
        """Validate every materialized row and identity field."""
        if (
            self.schema != MINISTRAL_PAYLOAD_SCHEMA
            or self.layout != MINISTRAL_PAYLOAD_LAYOUT
            or self.selected_indices_order != MINISTRAL_SELECTED_ORDER
        ):
            raise ValueError("Ministral flat payload schema or layout differs")
        if self.latent_plan_sha256 != latent_plan_sha256(self.semantic_arm):
            raise ValueError("Ministral payload latent-plan signature differs")
        if self.rows.ndim != 2 or self.rows.dtype != torch.bfloat16:
            raise ValueError("Ministral payload must contain rank-two bfloat16 rows")
        if len(self.keeps) != FANOUT_M3.workers_per_item:
            raise ValueError("Ministral payload must cover exactly three workers")
        if any(
            tuple(sorted(set(keep))) != keep or any(position < 0 for position in keep)
            for keep in self.keeps
        ):
            raise ValueError("Ministral payload keeps must be sorted, unique, and nonnegative")
        if self.rows_by_worker != tuple(len(keep) for keep in self.keeps):
            raise ValueError("Ministral payload worker geometry differs from its keeps")
        if sum(self.rows_by_worker) != int(self.rows.shape[0]):
            raise ValueError("Ministral payload row count differs from its worker geometry")
        blocks: list[torch.Tensor] = []
        offset = 0
        for row_count in self.rows_by_worker:
            blocks.append(self.rows[offset : offset + row_count])
            offset += row_count
        if self.worker_sha256 != tuple(tensor_content_sha256(block) for block in blocks):
            raise ValueError("Ministral payload worker hashes differ")
        if self.tensor_sha256 != tensor_content_sha256(self.rows):
            raise ValueError("Ministral payload tensor hash differs")
        if self.selected_indices_sha256 != selected_indices_sha256(self.keeps):
            raise ValueError("Ministral payload selected-index hash differs")


def build_flat_payload(
    captures: Sequence[MinistralWorkerCapture],
    *,
    semantic_arm: str,
    embedding_weight: torch.Tensor,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> MinistralFlatPayload:
    """Materialize the three-worker W16 flat-interleave embedding payload."""
    roster = tuple(captures)
    if len(roster) != FANOUT_M3.workers_per_item:
        raise ValueError("Ministral flat payload needs exactly three worker captures")
    selector, ratio = _latent_arm(semantic_arm)
    if not profile.admits_worker_prompt_tokens([capture.prompt_tokens for capture in roster]):
        raise ValueError(
            "Ministral production payload worker prompts differ from the registered "
            f"{profile.prompt_geometry} {profile.worker_prompt_tokens}"
        )
    if selector == "full":
        keeps = tuple(tuple(range(int(capture.embedding_rows.shape[0]))) for capture in roster)
    else:
        keeps = tuple(ministral_w16_keep(capture.scores[selector], ratio) for capture in roster)
    prompt_ids = tuple(torch.tensor(capture.prompt_token_ids) for capture in roster)
    prompt_lengths = tuple(capture.prompt_tokens for capture in roster)
    rolled = tuple(capture.rolled for capture in roster)
    blocks, _kinds, stats = interleaved_blocks(
        prompt_ids,
        flat_interleave_layout(keeps, prompt_lengths),
        keeps,
        rolled,
        lambda ids: token_embedding_rows(embedding_weight, ids),
    )
    if stats != {
        "delimiter_rows": 0,
        "tier_spans": 0,
        "tier_rows": 0,
        "kept_evidence_rows": sum(len(keep) - LATENT_STEPS for keep in keeps),
    }:
        raise RuntimeError("Ministral flat-interleave materialization changed")
    for capture, keep, block in zip(roster, keeps, blocks, strict=True):
        expected = capture.embedding_rows.index_select(0, torch.tensor(keep))
        if not torch.equal(block.cpu().to(torch.bfloat16), expected):
            raise RuntimeError("Ministral payload rows differ from the captured selected rows")
    materialized = tuple(block.detach().cpu().to(torch.bfloat16) for block in blocks)
    rows = torch.cat(materialized, dim=0)
    return MinistralFlatPayload(
        rows=rows,
        semantic_arm=semantic_arm,
        latent_plan_sha256=latent_plan_sha256(semantic_arm),
        keeps=keeps,
        rows_by_worker=tuple(len(keep) for keep in keeps),
        worker_sha256=tuple(tensor_content_sha256(block) for block in materialized),
        selected_indices_sha256=selected_indices_sha256(keeps),
        tensor_sha256=tensor_content_sha256(rows),
    )


__all__ = (
    "MINISTRAL_PAYLOAD_LAYOUT",
    "MINISTRAL_PAYLOAD_SCHEMA",
    "MINISTRAL_SELECTED_ORDER",
    "MinistralFlatPayload",
    "build_flat_payload",
    "latent_plan_sha256",
    "ministral_w16_keep",
    "protected_positions",
    "selected_indices_sha256",
    "tensor_content_sha256",
    "token_embedding_rows",
)
