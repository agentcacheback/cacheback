"""Durable score-replay support for resident Gemma capture."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch

from rcc.benchmarks.fanoutqa.gemma_data import RollFrame
from rcc.models.gemma.contract import (
    ARM_POLICIES,
    ARM_SELECTORS,
    LATENT_POLICIES,
    LATENT_REALIGN_ENABLED,
    LATENT_STEPS,
    QSNAP_ARMS,
    SELECTORS,
    SUPPORT_COMPOSITION,
)
from rcc.models.gemma.mechanism import pin_selector_sink
from rcc.models.gemma.selection import payload_length, qsnap_worker_keep, worker_keep
from rcc.run.gemma.cell_banks import ShardLog
from rcc.transforms.select.query_support.capture_bank import (
    BANK_SCOPE_ALL_LAYER,
    BANK_SCOPE_GLOBAL_ONLY,
    LayerBank,
)
from rcc.transforms.select.query_support.methods.compose import (
    SupportMomentBundle,
    selector_score,
)

LATENT_ROLL_SCHEMA = "gemma-fanoutqa-m3-latent-roll-native-norm-v3"


class ResidentCaptureBridge(Protocol):
    """Borrowed zero-copy view over the one live Gemma vLLM engine."""

    model: Any
    tokenizer: Any

    def extract(self, token_ids: Sequence[int]) -> Any:
        """Return the engine-prefilled hybrid cache for these token ids."""
        ...

    def release_view(self) -> None:
        """Drop wrapper references without closing the borrowed engine."""
        ...


@dataclass(frozen=True, eq=False)
class DeferredAudit:
    """Verify and bank one live capture after its handoff is published."""

    item: dict[str, Any]
    worker: int
    bank: LayerBank
    scores: dict[str, torch.Tensor]
    rolled: torch.Tensor
    bank_path: Path
    rolled_path: Path
    frame: RollFrame
    shard_log: ShardLog
    capture_s: float
    include_qsnap: bool
    extract_s: float
    roll_s: float
    capture_query_s: float
    parity_s: float

    def run(self) -> None:
        """Persist and replay the capture, then bank its audit wall.

        Runs on the seat's side thread beside the next capture; everything it
        holds is host memory, and every banked row is a frozen float32 clone.
        """
        started = time.perf_counter()
        if not self.bank_path.is_file():
            atomic_torch_save(self.bank.to_payload(), self.bank_path)
        reloaded_rolled = load_rolled(
            self.rolled_path,
            qid=self.item["qid"],
            worker=self.worker,
            hidden=int(self.rolled.shape[1]),
            frame=self.frame,
        )
        if not torch.equal(reloaded_rolled, self.rolled):
            raise RuntimeError(
                f"{self.item['qid']}/w{self.worker}: durable latent roll replay differs"
            )
        reloaded = replay_worker(
            self.item,
            self.worker,
            self.bank_path,
            **({"include_qsnap": True} if self.include_qsnap else {}),
        )
        for selector in (*SELECTORS, *(("snap",) if self.include_qsnap else ())):
            if not torch.equal(reloaded[selector], self.scores[selector]):
                raise RuntimeError(
                    f"{self.item['qid']}/w{self.worker}: durable {selector} replay differs"
                )
        audit_s = round(time.perf_counter() - started, 4)
        self.shard_log.bank_capture(
            {
                "capture_s": self.capture_s,
                "extract_s": self.extract_s,
                "roll_s": self.roll_s,
                "capture_query_s": self.capture_query_s,
                "parity_s": self.parity_s,
                "audit_s": audit_s,
                "layer_bank_file": self.bank_path.name,
                "layer_bank_schema": str(self.bank.schema),
                "latent_roll_file": self.rolled_path.name,
                "latent_roll_schema": LATENT_ROLL_SCHEMA,
            },
            qid=self.item["qid"],
            worker=self.worker,
        )


def latent_roll_filename(qid: str, worker: int) -> str:
    """Durable name of one worker's rolled latent-thought embeddings."""
    return f"{qid}_w{worker}_latent_roll.pt"


def load_latent_roll(
    path: Path,
    *,
    qid: str,
    worker: int,
    hidden: int,
    frame_prefix_length: int,
    frame_suffix_length: int,
) -> torch.Tensor:
    """Load one worker's rolled cargo and refuse any identity or shape drift."""
    raw_payload: object = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw_payload, dict):
        raise RuntimeError(f"{qid}/w{worker}: latent roll artifact identity differs")
    payload = cast(dict[str, object], raw_payload)
    if (
        payload.get("schema") != LATENT_ROLL_SCHEMA
        or payload.get("qid") != qid
        or payload.get("worker") != worker
        or payload.get("latent_steps") != LATENT_STEPS
        or payload.get("realign_enabled") != LATENT_REALIGN_ENABLED
        or payload.get("frame_prefix_length") != frame_prefix_length
        or payload.get("frame_suffix_length") != frame_suffix_length
    ):
        raise RuntimeError(f"{qid}/w{worker}: latent roll artifact identity differs")
    rolled: object = payload.get("rolled")
    if (
        not isinstance(rolled, torch.Tensor)
        or rolled.dtype != torch.bfloat16
        or rolled.shape != (LATENT_STEPS, hidden)
    ):
        raise RuntimeError(f"{qid}/w{worker}: latent roll artifact tensor is malformed")
    return rolled


def pinned_rows(tag: str, values: torch.Tensor, payload: int) -> torch.Tensor:
    """Validate a framed score and apply the Qwen pre-cut sink pin."""
    if values.shape != (payload,):
        raise RuntimeError(
            f"{tag}: framed score covers {tuple(values.shape)}, not {payload} cache rows"
        )
    return pin_selector_sink(values)


def durable_support_score(
    tag: str,
    bank: LayerBank,
    live_moments: SupportMomentBundle,
    *,
    device: torch.device,
    payload: int,
) -> torch.Tensor:
    """Return CPU bank replay after exact device and registered-keep parity."""
    live = selector_score(live_moments, SUPPORT_COMPOSITION).float()
    device_replay = selector_score(
        bank.replay_moments(scope=BANK_SCOPE_GLOBAL_ONLY, device=device),
        SUPPORT_COMPOSITION,
    ).float()
    if not torch.equal(device_replay, live):
        raise RuntimeError(f"{tag}: same-device global layer replay differs from live support")
    banked = bank.replay_score(ARM_POLICIES["r2"], scope=BANK_SCOPE_GLOBAL_ONLY).float().cpu()
    durable = pinned_rows(tag, banked, payload)
    live_cpu = pinned_rows(tag, live.cpu(), payload)
    for arm in LATENT_POLICIES:
        if worker_keep(arm, live_cpu, payload) != worker_keep(arm, durable, payload):
            raise RuntimeError(f"{tag}: durable support replay changes the {arm} keep")
    return durable


def durable_qsnap_score(
    tag: str,
    bank: LayerBank,
    live: torch.Tensor,
    *,
    payload: int,
) -> torch.Tensor:
    """Replay canonical global-layer mean query attention and prove every shared W16 keep."""
    durable = pinned_rows(
        tag,
        bank.replay_score("snap", scope=BANK_SCOPE_GLOBAL_ONLY).float().cpu(),
        payload,
    )
    live_cpu = pinned_rows(tag, live.float().cpu(), payload)
    if not torch.equal(durable, live_cpu):
        raise RuntimeError(f"{tag}: durable mean query attention replay differs from live capture")
    for arm in QSNAP_ARMS:
        if qsnap_worker_keep(arm, live_cpu, payload) != qsnap_worker_keep(arm, durable, payload):
            raise RuntimeError(f"{tag}: durable mean query attention replay changes the {arm} keep")
    return durable


def require_bank_geometry(tag: str, bank: LayerBank) -> None:
    """Require the sliding-layer geometry needed for offline recuts."""
    rows = bank.selected_rows(BANK_SCOPE_ALL_LAYER)
    sliding = [row for row in rows if not row.descriptor.is_global]
    if not sliding:
        raise RuntimeError(f"{tag}: layer bank has no sliding rows for offline recuts")
    windows = {row.descriptor.window for row in sliding}
    if any(window is None or window < 32 for window in windows):
        rendered = sorted(str(window) for window in windows)
        raise RuntimeError(f"{tag}: sliding layer windows are invalid: {rendered}")


def validate_scores(
    tag: str,
    scores: dict[str, torch.Tensor],
    payload: int,
    *,
    expected_selectors: set[str] | None = None,
) -> None:
    """Require the registered selector roster and finite payload scores."""
    expected = set(SELECTORS) if expected_selectors is None else expected_selectors
    if set(scores) != expected:
        raise RuntimeError(f"{tag}: capture produced selector roster {sorted(scores)}")
    for selector, values in scores.items():
        if values.shape != (payload,):
            raise RuntimeError(f"{tag}: {selector} score length is invalid")
        if not bool(torch.isfinite(values).all()) or not bool((values >= 0).all()):
            raise RuntimeError(f"{tag}: {selector} scores are not finite and nonnegative")
    duplicates = [
        selector
        for selector, values in scores.items()
        if selector not in {"support", "snap"} and torch.equal(values, scores["support"])
    ]
    if duplicates:
        raise RuntimeError(f"{tag}: variant scores duplicate support: {duplicates}")


def replay_worker(
    item: dict[str, Any],
    worker: int,
    path: Path,
    *,
    include_qsnap: bool = False,
) -> dict[str, torch.Tensor]:
    """Replay one worker's selector scores from its durable layer bank."""
    tag = f"{item['qid']}/w{worker}"
    payload = payload_length(int(item["prompt_ids"][worker].numel()))
    bank = LayerBank.load(path)
    require_bank_geometry(tag, bank)
    scores = {
        "support": pinned_rows(
            tag,
            bank.replay_score(ARM_POLICIES["r2"], scope=BANK_SCOPE_GLOBAL_ONLY).float().cpu(),
            payload,
        )
    }
    if include_qsnap:
        scores["snap"] = pinned_rows(
            tag,
            bank.replay_score("snap", scope=BANK_SCOPE_GLOBAL_ONLY).float().cpu(),
            payload,
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
    validate_scores(tag, scores, payload, expected_selectors=expected_selectors)
    return scores


def atomic_torch_save(value: Any, path: Path) -> None:
    """Fsync and atomically replace one torch payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, staging)
    with staging.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(staging, path)


def load_rolled(
    path: Path,
    *,
    qid: str,
    worker: int,
    hidden: int,
    frame: RollFrame,
) -> torch.Tensor:
    """Load one rolled payload using its frame identity."""
    return load_latent_roll(
        path,
        qid=qid,
        worker=worker,
        hidden=hidden,
        frame_prefix_length=len(frame.prefix_ids),
        frame_suffix_length=len(frame.suffix_ids),
    )


__all__ = (
    "LATENT_ROLL_SCHEMA",
    "DeferredAudit",
    "ResidentCaptureBridge",
    "atomic_torch_save",
    "durable_qsnap_score",
    "durable_support_score",
    "latent_roll_filename",
    "load_latent_roll",
    "load_rolled",
    "pinned_rows",
    "replay_worker",
    "require_bank_geometry",
    "validate_scores",
)
