"""Persistent Ministral producers for the split-fleet runtime.

A producer process holds one GPU for the whole arm and publishes either the flat
payload with its manifest or one report bundle, releasing its weights at close.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.latent.rollout import Realign, build_realign
from rcc.models.ministral.capture import MinistralWorkerCapture, capture_worker
from rcc.models.ministral.engine import build_text_engine
from rcc.models.ministral.engine_roles import (
    CAPTURE_ENGINE_IDENTITY,
    PRODUCER_ROLE,
    open_role_engine,
    verify_engine_identity,
)
from rcc.models.ministral.handoff import (
    MINISTRAL_HANDOFF_SUFFIX,
    write_handoff,
    write_handoff_clocks,
)
from rcc.models.ministral.payload import build_flat_payload
from rcc.models.ministral.receiver import warm_token_engine
from rcc.models.ministral.receiver_core import manager_prompt_tokens, question_token_ids
from rcc.models.ministral.text import generate_report_bundle
from rcc.models.ministral.text_codec import MINISTRAL_TEXT_ARMS, MinistralTextCodec
from rcc.run.fleet.clocks import write_bundle_clocks
from rcc.run.fleet.contract import ProducedArtifact, WorkItem
from rcc.run.ministral.config import prompt_records
from rcc.run.ministral.split_text import (
    TEXT_BUNDLE_CLOCKS,
    text_bundle_path,
    write_text_bundle,
)
from rcc.topologies.fanout import FANOUT_M3

Bank = Callable[[Mapping[str, Any]], dict[str, Any]]

#: The capture judger turn: the manager prompt with no reports in it, the
#: issue-only turn taken before selection.
CAPTURE_JUDGER_ARM = "issue_only"


def payload_id(qid: str, semantic_arm: str) -> str:
    """Return the identity of one item's payload on one arm."""
    return f"{qid}.{semantic_arm}"


def handoff_manifest_path(root: Path, qid: str, semantic_arm: str) -> Path:
    """Return where one item-arm handoff manifest is published."""
    return root / f"{payload_id(qid, semantic_arm)}{MINISTRAL_HANDOFF_SUFFIX}"


class _IdentityFrame:
    """Return the already-signed worker token row without reframing it."""

    @staticmethod
    def prompt(memory_ids: torch.Tensor) -> torch.Tensor:
        """Return the prepared row unchanged: it is already the framed prompt."""
        return memory_ids


class MinistralLatentProducer:
    """Keep the capture engine warm and publish one handoff per claimed item."""

    def __init__(
        self,
        *,
        arm: str,
        items: Mapping[str, dict[str, Any]],
        handoff_root: Path,
        bank: Bank,
        worker_index: int,
        codec: MinistralTextCodec,
        profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
        open_engine: Callable[[str], Any] = open_role_engine,
        capture: Callable[..., MinistralWorkerCapture] = capture_worker,
    ) -> None:
        """Bind one latent arm to its item panel and artifact root."""
        if arm in MINISTRAL_TEXT_ARMS or arm == CAPTURE_JUDGER_ARM:
            raise ValueError(f"{arm!r}: the Ministral latent producer serves latent arms only")
        self.arm = arm
        self.items = dict(items)
        self.handoff_root = handoff_root
        self.bank = bank
        self.worker_index = worker_index
        self.codec = codec
        self.profile = profile
        self._open_engine = open_engine
        self._capture = capture
        self.engine: Any | None = None
        self.realign: Realign | None = None

    def warm(self) -> None:
        """Open the producer engine and check that it is the capture engine."""
        started = time.perf_counter()
        engine = self._open_engine(PRODUCER_ROLE)
        load_s = time.perf_counter() - started
        try:
            verify_engine_identity(engine, CAPTURE_ENGINE_IDENTITY)
            self.realign = build_realign(engine.resident_model, enabled=False)
        except Exception:
            engine.close()
            raise
        self.engine = engine
        self.bank(
            {
                "kind": "phase",
                "phase": "ministral_split_producer_warm",
                "arm": self.arm,
                "engine_role": CAPTURE_ENGINE_IDENTITY.engine_role,
                "capture_connector": dict(CAPTURE_ENGINE_IDENTITY.capture_connector or {}),
                "runtime_fingerprint": str(engine.runtime_fingerprint),
                "model_load_s": round(load_s, 4),
            }
        )

    def _captures(self, qid: str) -> tuple[MinistralWorkerCapture, ...]:
        engine, realign = self.engine, self.realign
        if engine is None or realign is None:
            raise RuntimeError("Ministral split producer is not warm")
        source = self.items[qid]
        question = str(source["question"].question)
        manager, _record = manager_prompt_tokens(
            self.codec,
            qid=qid,
            semantic_arm=CAPTURE_JUDGER_ARM,
            question=question,
            profile=self.profile,
        )
        prompts = tuple(source["prompt_ids"])
        item: dict[str, object] = {
            "qid": qid,
            "memory_ids": prompts,
            "prompt_ids": prompts,
            "judger_ids": torch.tensor([list(manager)], dtype=torch.long),
            "question_ids": torch.tensor(
                list(question_token_ids(self.codec, manager, question)),
                dtype=torch.long,
            ),
        }
        frame = _IdentityFrame()
        return tuple(
            self._capture(engine.capture_bridge, item, worker, realign, frame)
            for worker in range(FANOUT_M3.workers_per_item)
        )

    def produce(self, item: WorkItem) -> ProducedArtifact:
        """Capture, select, and publish one item's handoff manifest."""
        engine = self.engine
        if engine is None:
            raise RuntimeError("Ministral split producer is not warm")
        capture_started = time.perf_counter()
        captures = self._captures(item.qid)
        capture_s = time.perf_counter() - capture_started
        select_started = time.perf_counter()
        payload = build_flat_payload(
            captures,
            semantic_arm=self.arm,
            embedding_weight=engine.embedding_weight,
            profile=self.profile,
        )
        select_s = time.perf_counter() - select_started
        # The cost of publishing: writing the tensor, hashing it, and closing
        # the manifest. A sender on a wire would not pay it, so it is banked
        # beside the caller clocks rather than inside them.
        spill_started = time.perf_counter()
        manifest_path = write_handoff(
            self.handoff_root,
            payload_id=payload_id(item.qid, self.arm),
            payload=payload,
        )
        spill_save_s = time.perf_counter() - spill_started
        body: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
        write_handoff_clocks(
            manifest_path,
            payload_id=payload_id(item.qid, self.arm),
            fingerprint=str(body["fingerprint"]),
            spill_save_s=spill_save_s,
            producer_s=capture_s + select_s,
        )
        self.bank(
            {
                "kind": "producer",
                "phase": "ministral_split_capture",
                "arm": self.arm,
                "qid": item.qid,
                "worker": self.worker_index,
                "capture_s": round(capture_s, 4),
                "selection_s": round(select_s, 4),
                "spill_save_s": round(spill_save_s, 4),
                "handoff_rows": int(payload.rows.shape[0]),
                "payload_tensor_sha256": payload.tensor_sha256,
            }
        )
        return ProducedArtifact(manifest_path, time.time(), {"spill_save_s": spill_save_s})

    def close(self) -> None:
        """Release the capture view and the producer engine."""
        engine, self.engine = self.engine, None
        self.realign = None
        if engine is not None:
            engine.close()


class MinistralTextProducer:
    """Keep one non-resident sender warm and publish its report bundles."""

    def __init__(
        self,
        *,
        arm: str,
        items: Mapping[str, dict[str, Any]],
        handoff_root: Path,
        bank: Bank,
        codec: MinistralTextCodec,
        profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
        engine_builder: Callable[[str], Any] = build_text_engine,
    ) -> None:
        """Bind one non-resident text sender to its item panel."""
        if arm not in {"text_medium", "text_small"}:
            raise ValueError(f"{arm!r}: the resident text sender runs on the fused route")
        self.arm = arm
        self.items = dict(items)
        self.profile = profile
        self.handoff_root = handoff_root
        self.bank = bank
        self.codec = codec
        self._engine_builder = engine_builder
        self.engine: Any | None = None

    def warm(self) -> None:
        """Open the sender checkpoint and decode once before any measurement."""
        started = time.perf_counter()
        engine = self._engine_builder(self.arm)
        load_s = time.perf_counter() - started
        try:
            warmup_s = warm_token_engine(
                engine,
                self.codec.encode_warmup_prompt(),
                tag=f"ministral-{self.arm}",
            )
        except Exception:
            engine.close()
            raise
        self.engine = engine
        self.bank(
            {
                "kind": "phase",
                "phase": "ministral_split_text_producer_warm",
                "arm": self.arm,
                "engine_role": self.arm,
                "runtime_fingerprint": str(engine.runtime_fingerprint),
                "model_load_s": round(load_s, 4),
                "warmup_s": round(warmup_s, 4),
            }
        )

    def produce(self, item: WorkItem) -> ProducedArtifact:
        """Generate one item's three worker reports and publish the bundle."""
        engine = self.engine
        if engine is None:
            raise RuntimeError("Ministral split text producer is not warm")
        source = self.items[item.qid]
        bundle = generate_report_bundle(
            engine,
            self.codec,
            prompt_records(source["artifacts"][self.arm]),
            profile=self.profile,
        )
        # The same publish clock the latent arm banks, so a text row and a
        # latent row price the crossing the same way.
        spill_started = time.perf_counter()
        path = write_text_bundle(
            text_bundle_path(self.handoff_root, item.qid, self.arm),
            bundle,
        )
        spill_save_s = time.perf_counter() - spill_started
        write_bundle_clocks(
            TEXT_BUNDLE_CLOCKS,
            path,
            qid=item.qid,
            semantic_arm=self.arm,
            spill_save_s=spill_save_s,
        )
        self.bank(
            {
                "kind": "producer",
                "phase": "ministral_split_reports",
                "arm": self.arm,
                "qid": item.qid,
                "generation_s": round(bundle.generation_s, 4),
                "spill_save_s": round(spill_save_s, 4),
                "seed_tag": bundle.seed_tag,
            }
        )
        return ProducedArtifact(path, time.time(), {"spill_save_s": spill_save_s})

    def close(self) -> None:
        """Release the sender checkpoint before the next arm."""
        engine, self.engine = self.engine, None
        if engine is not None:
            engine.close()


__all__ = (
    "CAPTURE_JUDGER_ARM",
    "MinistralLatentProducer",
    "MinistralTextProducer",
    "handoff_manifest_path",
    "payload_id",
)
