"""The route lane's producer seats for the fan-out topology.

The text seat publishes one report bundle per claimed item; the latent seat
publishes the selected embedding rows with one selector vector per worker.
"""

from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.fanoutqa.payload import write_payload
from rcc.benchmarks.fanoutqa.prompts import render_chat
from rcc.benchmarks.fanoutqa.source_padding import ProbeItem
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.nemotron.prefill import VllmHybridPrefill
from rcc.models.nemotron.producer import NemotronProducer
from rcc.models.qwen.backend import QwenBackendSettings, QwenVllmBackend, build_qwen_backend
from rcc.models.qwen.engine import (
    QWEN_ENGINE_ROUTE,
    QWEN_NATURAL_ENGINE_MAX_MODEL_LEN,
    EngineProducer,
)
from rcc.models.qwen.producer import produce_latent_payload
from rcc.models.qwen.prompts import worker_prompts
from rcc.models.qwen.text import QwenReportBundle, generate_report_bundle
from rcc.models.route import Decoder, RouteFamily, tokenizer_decoder
from rcc.run.fleet.contract import ProducedArtifact, WorkItem
from rcc.run.fleet.latency import WARMUP_BODY, warmup_seed
from rcc.run.nemotron.compiler_policy import close_backend, policy_fields
from rcc.run.qwen.worker_contract import physical_arm
from rcc.topologies.fanout import FANOUT_M3

Bank = Any


def _seat_profile(profile: BenchmarkProfile | None) -> BenchmarkProfile:
    """Return the seat's benchmark: the one it was given, else the node's own."""
    if profile is not None:
        return profile
    from rcc.run.qwen.adapter import execution_profile

    return execution_profile()


def _require_fanout_benchmark(profile: BenchmarkProfile, *, seat: str) -> None:
    """Refuse, by name, a benchmark these seats cannot serve.

    These seats render the fan-out shard prompts and write one score per fan-out
    worker, so another layout would name files the payload never wrote.
    """
    if (
        profile.payload_layout != FANOUTQA_NATURAL_DEV50.payload_layout
        or profile.workers_per_item != FANOUT_M3.workers_per_item
    ):
        raise ValueError(
            f"the FanOutQA {seat} producer serves layout "
            f"{FANOUTQA_NATURAL_DEV50.payload_layout!r} over "
            f"{FANOUT_M3.workers_per_item} workers; {profile.benchmark_key} registers "
            f"{profile.payload_layout!r} over {profile.workers_per_item}"
        )


def save_tensor(path: Path, tensor: torch.Tensor) -> None:
    """Write one tensor to its path through a staging file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(tensor.detach().cpu(), staging)
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def _report_dict(bundle: QwenReportBundle) -> dict[str, object]:
    return bundle.result_fields()


class QwenTextProducer:
    """Keep one sender resident and publish one report bundle per item."""

    def __init__(
        self,
        *,
        policy: str,
        tokenizer: Any,
        items: dict[str, ProbeItem],
        payload_root: Path,
        bank: Bank,
        family: RouteFamily,
        profile: BenchmarkProfile | None = None,
    ) -> None:
        """Bind one text arm, the benchmark it serves, and its item lookup."""
        profile = _seat_profile(profile)
        _require_fanout_benchmark(profile, seat="text")
        arm = physical_arm(policy, family=family)
        if arm.semantic_arm not in {"text_primary", "text_medium", "text_small"}:
            raise ValueError(f"{policy}: not a route text producer")
        if (
            arm.sender_checkpoint is None
            or arm.sender_revision is None
            or arm.sender_tokenizer is None
            or arm.sender_tokenizer_revision is None
        ):
            raise RuntimeError(f"{policy}: route sender registration is incomplete")
        self.family = family
        self.decoder: Decoder = tokenizer_decoder(tokenizer)
        self.policy = policy
        self.semantic_arm = arm.semantic_arm
        self.checkpoint = arm.sender_checkpoint
        self.revision = arm.sender_revision
        # The sender renders with its own tokenizer and codec.
        self.tokenizer_checkpoint = arm.sender_tokenizer
        self.tokenizer_revision = arm.sender_tokenizer_revision
        self.tokenizer = tokenizer
        self.items = items
        self.profile = replace(family.benchmark_profile(profile), question_ids=tuple(items))
        self.payload_root = payload_root
        self.bank = bank
        self.backend: QwenVllmBackend | None = None

    def warm(self) -> None:
        """Open the sender checkpoint and decode once before any measurement."""
        started = time.perf_counter()
        self.backend = build_qwen_backend(
            self.checkpoint,
            self.revision,
            settings=QwenBackendSettings(),
            family=self.family,
            tokenizer=self.tokenizer_checkpoint,
            tokenizer_revision=self.tokenizer_revision,
        )
        load_s = time.perf_counter() - started
        warm_started = time.perf_counter()
        codec = self.family.sender_family(self.semantic_arm)
        self.backend.warmup_decode(
            render_chat(
                self.tokenizer,
                WARMUP_BODY,
                enable_thinking=self.family.profile.decode.enable_thinking,
                system_prompt=codec.thinking_system_prompt,
                assistant_prefill=codec.assistant_prefill,
                low_effort=codec.low_effort,
            ),
            seed=warmup_seed(self.policy, "text_producer"),
        )
        self.bank(
            {
                "kind": "phase",
                "phase": f"{self.family.lane}_text_producer_warm",
                **policy_fields(self.backend, self.family.lane),
                "policy": self.policy,
                "checkpoint": self.checkpoint,
                "model_load_s": round(load_s, 4),
                "warmup_s": round(time.perf_counter() - warm_started, 4),
            }
        )

    def produce(self, item: WorkItem) -> ProducedArtifact:
        """Generate one item's worker reports and publish the bundle."""
        if self.backend is None:
            raise RuntimeError("route text producer is not warm")
        probe = self.items[item.qid]
        bundle = generate_report_bundle(
            self.backend,
            worker_prompts(
                probe,
                self.tokenizer,
                semantic_arm=self.semantic_arm,
                family=self.family,
                profile=self.profile,
            ),
            qid=item.qid,
            semantic_arm=self.semantic_arm,
            family=self.family,
            decoder=self.decoder,
            profile=self.profile,
            prepared_item=probe,
        )
        path = write_payload(
            self.payload_root,
            qid=item.qid,
            files={},
            meta={
                "family": self.family.model_id,
                "semantic_arm": self.semantic_arm,
                "policy": self.policy,
                "checkpoint": self.checkpoint,
                "revision": self.revision,
            },
            report_bundle=_report_dict(bundle),
        )
        return ProducedArtifact(path, time.time())

    def close(self) -> None:
        """Release the text sender."""
        if self.backend is not None:
            close_backend(
                self.backend,
                self.bank,
                lane=self.family.lane,
                policy=self.policy,
                role="text_producer",
            )
            self.backend = None


class QwenLatentProducer:
    """Keep the capture engine resident and publish rows and selector scores."""

    def __init__(
        self,
        *,
        policy: str,
        tokenizer: Any,
        items: dict[str, ProbeItem],
        payload_root: Path,
        bank: Bank,
        family: RouteFamily,
        profile: BenchmarkProfile | None = None,
    ) -> None:
        """Bind one latent arm, the benchmark it serves, and its payload root."""
        profile = _seat_profile(profile)
        _require_fanout_benchmark(profile, seat="latent")
        arm = physical_arm(policy, family=family)
        if arm.selector not in {"snap", "support"}:
            raise ValueError(f"{policy}: not a route latent producer")
        self.family = family
        self.policy = policy
        self.semantic_arm = arm.semantic_arm
        self.selector = arm.selector
        self.tokenizer = tokenizer
        self.items = items
        self.profile = replace(family.benchmark_profile(profile), question_ids=tuple(items))
        self.payload_root = payload_root
        self.bank = bank
        self.backend: QwenVllmBackend | None = None
        self.producer: EngineProducer | NemotronProducer | None = None

    def warm(self) -> None:
        """Open the capture engine and the side view onto its resident model."""
        adapter = self.family.score_adapter(self.semantic_arm)
        if adapter is not None:
            adapter.validate_tokenizer(self.tokenizer)
        started = time.perf_counter()
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        profile = self.family.profile
        capture_window = QWEN_NATURAL_ENGINE_MAX_MODEL_LEN
        self.backend = build_qwen_backend(
            profile.checkpoint,
            profile.revision,
            settings=QwenBackendSettings(capture=True, capture_max_model_len=capture_window),
            family=self.family,
            tokenizer=profile.tokenizer,
            tokenizer_revision=profile.tokenizer_revision,
        )
        if self.family.lane == "nemotron":
            model = self.backend.resident_model()
            self.producer = NemotronProducer(
                model, VllmHybridPrefill(self.backend.raw_llm, model.config)
            )
        else:
            self.producer = EngineProducer(
                self.backend.engine_handle(),
                latent_steps=self.profile.latent_steps,
            )
        self.bank(
            {
                "kind": "phase",
                "phase": f"{self.family.lane}_latent_producer_warm",
                **policy_fields(self.backend, self.family.lane),
                "policy": self.policy,
                "checkpoint": profile.checkpoint,
                "producer_backend": "engine",
                "producer_route": dict(profile.runtime.engine_flags).get("producer_backend")
                if self.family.lane == "nemotron"
                else QWEN_ENGINE_ROUTE,
                # The window the producer engine actually opened, banked here
                # because the route identity carries only its label.
                "capture_max_model_len": capture_window,
                "model_load_s": round(time.perf_counter() - started, 4),
            }
        )

    def _expected_paths(self, qid: str) -> dict[str, Path]:
        return {
            "embedding_rows": self.payload_root / "rows" / f"{qid}.pt",
            **{
                f"score_w{worker}": self.payload_root / "selector_scores" / f"{qid}_w{worker}.pt"
                for worker in range(self.profile.workers_per_item)
            },
        }

    def produce(self, item: WorkItem) -> ProducedArtifact:
        """Capture, select, and atomically replace one queued embedding payload."""
        if self.producer is None:
            raise RuntimeError("route latent producer is not warm")
        (self.payload_root / f"{item.qid}.payload.json").unlink(missing_ok=True)
        production = produce_latent_payload(
            self.producer,
            self.tokenizer,
            self.items[item.qid],
            semantic_arm=self.semantic_arm,
            family=self.family,
            profile=self.profile,
        )
        paths = self._expected_paths(item.qid)
        save_tensor(paths["embedding_rows"], production.payload.rows)
        for worker, score in enumerate(production.selector_scores):
            save_tensor(paths[f"score_w{worker}"], score)
        meta = production.result_fields()
        meta.update(
            {
                "family": self.family.model_id,
                "policy": self.policy,
                "semantic_arm": self.semantic_arm,
            }
        )
        path = write_payload(
            self.payload_root,
            qid=item.qid,
            files=paths,
            meta=meta,
        )
        return ProducedArtifact(path, time.time())

    def close(self) -> None:
        """Release the capture engine and the view aliased onto it."""
        self.producer = None
        if self.backend is not None:
            close_backend(
                self.backend,
                self.bank,
                lane=self.family.lane,
                policy=self.policy,
                role="latent_producer",
            )
            self.backend = None


__all__ = ("QwenLatentProducer", "QwenTextProducer", "save_tensor")
