"""What the Gemma lane's capture and receiver halves share about a panel.

Which items a GPU owns, the receiver's replay frame, the coordinator prompt split,
and the per-item digest a panel load is checked against.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from rcc.benchmarks.fanoutqa.data import load_freeze
from rcc.benchmarks.fanoutqa.panel_pins import require_registered_identity
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.gemma.contract import (
    LATENT_ROLL_PREFIX_TEMPLATE,
    LATENT_ROLL_SUFFIX_TEMPLATE,
    RECEIVER_ENABLE_THINKING,
    REPORT_ENABLE_THINKING,
)
from rcc.models.gemma.mechanism import encode, receiver_turn_ids
from rcc.models.gemma.prompts import manager_prompt, question_token_ids
from rcc.run import identity
from rcc.run.gemma.cell_banks import gpu_assignments


@dataclass(frozen=True)
class WorkerPanel:
    """One GPU's source panel and its source identity."""

    qids: tuple[str, ...]
    items: tuple[dict[str, Any], ...]
    source_manifest: dict[str, Any]
    prepared_sha256: str


@dataclass(frozen=True)
class RollFrame:
    """The question-conditioned worker prompt around verbatim evidence rows."""

    prefix_ids: tuple[int, ...]
    suffix_ids: tuple[int, ...]

    def prompt(self, memory_ids: torch.Tensor) -> torch.Tensor:
        """Return the framed rollout prompt for one worker's evidence rows."""
        device = memory_ids.device
        return torch.cat(
            (
                torch.tensor(self.prefix_ids, dtype=torch.long, device=device),
                memory_ids,
                torch.tensor(self.suffix_ids, dtype=torch.long, device=device),
            )
        )


def build_roll_frame(tokenizer: Any, question: str) -> RollFrame:
    """Encode the roll frame templates inside the model's chat turn."""
    turn_prefix, turn_suffix = receiver_turn_ids(
        tokenizer,
        enable_thinking=REPORT_ENABLE_THINKING,
    )
    head = encode(tokenizer, LATENT_ROLL_PREFIX_TEMPLATE)
    tail = encode(tokenizer, LATENT_ROLL_SUFFIX_TEMPLATE.format(question=question))
    if not head or not tail:
        raise RuntimeError("the latent roll frame templates encoded to no tokens")
    return RollFrame(tuple(turn_prefix) + tuple(head), tuple(tail) + tuple(turn_suffix))


def coordinator_ids(tokenizer: Any, question: str) -> tuple[list[int], list[int], list[int]]:
    """Return the coordinator prompt as probe rows, body rows, and question rows."""
    rendered = manager_prompt(tokenizer, question)
    judger_ids = encode(tokenizer, rendered)
    prefix, suffix = receiver_turn_ids(tokenizer, enable_thinking=RECEIVER_ENABLE_THINKING)
    cut = len(judger_ids) - len(suffix)
    if judger_ids[: len(prefix)] != prefix or judger_ids[cut:] != suffix or cut <= len(prefix):
        raise RuntimeError("the rendered coordinator prompt does not carry the registered turn")
    body_ids = judger_ids[len(prefix) : cut]
    question_ids = question_token_ids(tokenizer, rendered, question)
    if not question_ids:
        raise RuntimeError("the coordinator prompt exposes no question rows")
    return judger_ids, body_ids, question_ids


def panel_qids(
    *,
    item_split: str,
    offset: int,
    count: int,
    profile: BenchmarkProfile,
) -> tuple[str, ...]:
    """Return one explicitly selected frozen panel slice."""
    if offset < 0 or count < 1:
        raise ValueError("offset must be nonnegative and count positive")
    if item_split == "dev":
        values = tuple(str(qid) for qid in load_freeze()["dev_ids"])
    elif item_split == "heldout":
        values = profile.question_ids
    else:
        raise ValueError(f"unsupported FanOutQA item split {item_split!r}")
    selected = values[offset : offset + count]
    if len(selected) != count:
        raise ValueError(f"{item_split} slice {offset}:{offset + count} exceeds {len(values)} ids")
    return selected


def worker_qids(
    *,
    offset: int,
    count: int,
    node_gpus: int,
    worker_index: int,
    profile: BenchmarkProfile,
    item_split: str = "dev",
) -> tuple[str, ...]:
    """Return the items one GPU owns under the whole-item round-robin assignment."""
    if not 0 <= worker_index < node_gpus:
        raise ValueError("worker index is outside the node GPU roster")
    selected = gpu_assignments(
        panel_qids(item_split=item_split, offset=offset, count=count, profile=profile), node_gpus
    )[worker_index]
    if not selected:
        raise RuntimeError(f"gpu{worker_index} owns no items")
    return selected


def _prepared_digest(items: list[dict[str, Any]], source_fingerprint: str) -> str:
    body = {
        "source_fingerprint": source_fingerprint,
        "items": [
            {
                "qid": item["qid"],
                # The framed prompt carries the evidence rows verbatim, so one
                # digest covers the evidence, the frame, and the packing.
                "prompt_ids": [ids.tolist() for ids in item["prompt_ids"]],
                "request_ids": item["request_ids"],
                "dead_pages": item["dead_pages"],
                "global_item_index": item["global_item_index"],
            }
            for item in items
        ],
    }
    return identity.fingerprint(body, identity.json_compact_legacy)


def item_construction_sha256(item: Mapping[str, Any]) -> str:
    """Digest one reconstructed item, independent of how the panel is split."""
    body = {
        "qid": str(item["qid"]),
        "prompt_ids": [ids.tolist() for ids in item["prompt_ids"]],
        "request_ids": list(item["request_ids"]),
        "dead_pages": list(item["dead_pages"]),
    }
    return identity.fingerprint(body, identity.json_compact_legacy)


def require_registered_construction(
    panel_id: str,
    items: Sequence[Mapping[str, Any]],
    *,
    required: bool,
) -> None:
    """Raise when a Gemma panel's reconstruction differs from its pinned identity."""
    require_registered_identity(
        panel_id,
        "gemma",
        {str(item["qid"]): item_construction_sha256(item) for item in items},
        required=required,
    )


def load_shared_worker_panel(
    bundle_root: Path,
    tokenizer: Any,
    *,
    offset: int,
    count: int,
    node_gpus: int,
    worker_index: int,
    profile: BenchmarkProfile,
) -> WorkerPanel:
    """Construct this worker's Gemma-tokenized items from the natural bundle."""
    from rcc.benchmarks.fanoutqa.gemma_natural import load_natural_worker_panel

    return load_natural_worker_panel(
        Path(bundle_root),
        tokenizer,
        offset=offset,
        count=count,
        node_gpus=node_gpus,
        worker_index=worker_index,
        profile=profile,
    )


__all__ = (
    "RollFrame",
    "WorkerPanel",
    "build_roll_frame",
    "coordinator_ids",
    "item_construction_sha256",
    "load_shared_worker_panel",
    "panel_qids",
    "require_registered_construction",
    "worker_qids",
)
