"""The chain's two producer seats: hops of capture, or hops of rewriting.

One seat walks every hop of one item, which is the difference from the fan-out
seats beside them. The text seat publishes the last note as the one ticket.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from rcc.benchmarks.fanoutqa.payload import write_payload
from rcc.benchmarks.fanoutqa.prompts import render_chat
from rcc.benchmarks.longbench_v2.data import ChainItem
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.nemotron.prefill import VllmHybridPrefill
from rcc.models.nemotron.producer import NemotronProducer
from rcc.models.qwen.backend import QwenBackendSettings, QwenVllmBackend, build_qwen_backend
from rcc.models.qwen.chain import chain_hop_backend, chain_hop_path, produce_chain_payload
from rcc.models.qwen.chain_text import generate_notes_chain
from rcc.models.qwen.engine import EngineProducer
from rcc.models.route import Decoder, RouteFamily, tokenizer_decoder
from rcc.run.fleet.contract import ProducedArtifact, WorkItem
from rcc.run.fleet.latency import WARMUP_BODY, warmup_seed
from rcc.run.nemotron.compiler_policy import close_backend, policy_fields
from rcc.run.qwen.producers import save_tensor
from rcc.run.qwen.worker_contract import physical_arm
from rcc.topologies.chain import CHAIN_TOPOLOGY_KEY, HOPS

#: A seat's engine binds a distributed-init port when it opens. Every seat of an
#: arm opens at once and the port is drawn at random, so warming retries an
#: EADDRINUSE open this many times and re-raises everything else.
_ENGINE_OPEN_ATTEMPTS = 3
_ENGINE_OPEN_RETRY_SECONDS = 5.0


def _open_engine(
    build: Callable[[], QwenVllmBackend], *, sleep: Callable[[float], None] = time.sleep
) -> QwenVllmBackend:
    """Open one engine, retrying a startup port collision and nothing else."""
    for attempt in range(1, _ENGINE_OPEN_ATTEMPTS + 1):
        try:
            return build()
        except Exception as exc:
            if "EADDRINUSE" not in str(exc) or attempt == _ENGINE_OPEN_ATTEMPTS:
                raise
            sleep(_ENGINE_OPEN_RETRY_SECONDS * attempt)
    raise RuntimeError("engine open attempts exhausted")


Bank = Any

#: Seconds one chain request may spend inside the engine before the seat
#: abandons it. It is longer than the ``EngineProducer`` default because one
#: hop prefills far more rows than one fan-out worker does.
QWEN_CHAIN_REQUEST_DEADLINE_SECONDS = 600.0


def chain_engine_producer(
    backend: QwenVllmBackend, family: RouteFamily, *, latent_steps: int = 40
) -> EngineProducer | NemotronProducer:
    """Bind the chain to its family's resident capture and continuation."""
    if family.lane == "nemotron":
        model = backend.resident_model()
        return NemotronProducer(
            model, VllmHybridPrefill(backend.raw_llm, model.config), latent_steps=latent_steps
        )
    return EngineProducer(
        backend.engine_handle(),
        latent_steps=latent_steps,
        deadline_seconds=QWEN_CHAIN_REQUEST_DEADLINE_SECONDS,
    )


def _require_chain_benchmark(profile: BenchmarkProfile, *, seat: str) -> None:
    """Refuse, by name, a benchmark these seats do not walk."""
    if profile.topology_key != CHAIN_TOPOLOGY_KEY:
        raise ValueError(
            f"the chain {seat} producer walks the {CHAIN_TOPOLOGY_KEY} topology; "
            f"{profile.benchmark_key} registers {profile.topology_key!r}"
        )


class QwenChainTextProducer:
    """Keep one sender resident and publish one item's notes chain."""

    def __init__(
        self,
        *,
        policy: str,
        tokenizer: Any,
        items: dict[str, ChainItem],
        payload_root: Path,
        bank: Bank,
        family: RouteFamily,
        profile: BenchmarkProfile,
    ) -> None:
        """Bind one text arm, the chain it rewrites, and its item lookup."""
        _require_chain_benchmark(profile, seat="text")
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
        self.tokenizer_checkpoint = arm.sender_tokenizer
        self.tokenizer_revision = arm.sender_tokenizer_revision
        self.tokenizer = tokenizer
        # The source parts were cut with the profile's own tokenizer, so that
        # is the tokenizer that reads them back, whatever sender this arm
        # opened. The seat holds both, and the sender's renders the rewrite.
        self.ledger_checkpoint = family.profile.tokenizer
        self.ledger_revision = family.profile.tokenizer_revision
        self.ledger_tokenizer: Any | None = None
        self.items = items
        self.profile = replace(family.benchmark_profile(profile), question_ids=tuple(items))
        self.payload_root = payload_root
        self.bank = bank
        self.backend: QwenVllmBackend | None = None

    def warm(self) -> None:
        """Open the sender checkpoint and decode once before any measurement."""
        from transformers import AutoTokenizer

        started = time.perf_counter()
        self.ledger_tokenizer = cast(Any, AutoTokenizer).from_pretrained(
            self.ledger_checkpoint, revision=self.ledger_revision
        )
        self.backend = _open_engine(
            lambda: build_qwen_backend(
                self.checkpoint,
                self.revision,
                settings=QwenBackendSettings(),
                family=self.family,
                tokenizer=self.tokenizer_checkpoint,
                tokenizer_revision=self.tokenizer_revision,
            )
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
                "phase": "nemotron_text_producer_warm"
                if self.family.lane == "nemotron"
                else f"{self.family.lane}_chain_text_producer_warm",
                **policy_fields(self.backend, self.family.lane),
                "policy": self.policy,
                "checkpoint": self.checkpoint,
                "model_load_s": round(load_s, 4),
                "warmup_s": round(time.perf_counter() - warm_started, 4),
            }
        )

    def produce(self, item: WorkItem) -> ProducedArtifact:
        """Rewrite the running notes across the hops and publish the ticket."""
        if self.backend is None or self.ledger_tokenizer is None:
            raise RuntimeError("chain text producer is not warm")
        bundle = generate_notes_chain(
            self.backend,
            self.tokenizer,
            self.items[item.qid],
            qid=item.qid,
            semantic_arm=self.semantic_arm,
            family=self.family,
            decoder=self.decoder,
            profile=self.profile,
            ledger_tokenizer=self.ledger_tokenizer,
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
            report_bundle=bundle.result_fields(),
        )
        return ProducedArtifact(path, time.time())

    def close(self) -> None:
        """Release the text sender."""
        self.ledger_tokenizer = None
        if self.backend is not None:
            close_backend(
                self.backend,
                self.bank,
                lane=self.family.lane,
                policy=self.policy,
                role="text_producer",
            )
            self.backend = None


class QwenChainLatentProducer:
    """Keep the chain capture engine resident and publish one terminal block."""

    def __init__(
        self,
        *,
        policy: str,
        tokenizer: Any,
        items: dict[str, ChainItem],
        payload_root: Path,
        bank: Bank,
        family: RouteFamily,
        profile: BenchmarkProfile,
    ) -> None:
        """Bind one latent arm, the chain it walks, and its payload root."""
        _require_chain_benchmark(profile, seat="latent")
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
        """Open the chain capture engine: prompt embeddings at the served window."""
        started = time.perf_counter()
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        profile = self.family.profile
        self.backend = _open_engine(
            lambda: build_qwen_backend(
                profile.checkpoint,
                profile.revision,
                settings=QwenBackendSettings(capture=True, chain=True),
                family=self.family,
                tokenizer=profile.tokenizer,
                tokenizer_revision=profile.tokenizer_revision,
            )
        )
        self.producer = chain_engine_producer(
            self.backend, self.family, latent_steps=self.profile.latent_steps
        )
        self.bank(
            {
                "kind": "phase",
                "phase": "nemotron_latent_producer_warm"
                if self.family.lane == "nemotron"
                else f"{self.family.lane}_chain_latent_producer_warm",
                **policy_fields(self.backend, self.family.lane),
                "policy": self.policy,
                "checkpoint": profile.checkpoint,
                "producer_backend": "engine",
                "producer_route": chain_hop_backend(chain_hop_path(self.family))[1],
                "model_load_s": round(time.perf_counter() - started, 4),
            }
        )

    def _expected_paths(self, qid: str) -> dict[str, Path]:
        """Return the rows this item ships and one selector vector per hop."""
        return {
            "embedding_rows": self.payload_root / "rows" / f"{qid}.pt",
            **{
                f"score_h{hop}": self.payload_root / "selector_scores" / f"{qid}_h{hop}.pt"
                for hop in range(1, HOPS + 1)
            },
        }

    def produce(self, item: WorkItem) -> ProducedArtifact:
        """Walk the hops and atomically replace one queued terminal payload."""
        if self.producer is None:
            raise RuntimeError("chain latent producer is not warm")
        (self.payload_root / f"{item.qid}.payload.json").unlink(missing_ok=True)
        production = produce_chain_payload(
            self.producer,
            self.tokenizer,
            self.items[item.qid],
            semantic_arm=self.semantic_arm,
            family=self.family,
            profile=self.profile,
        )
        paths = self._expected_paths(item.qid)
        save_tensor(paths["embedding_rows"], production.payload.rows)
        for hop, score in enumerate(production.selector_scores, start=1):
            save_tensor(paths[f"score_h{hop}"], score)
        meta = production.result_fields()
        meta.update(
            {
                "family": self.family.model_id,
                "policy": self.policy,
                "semantic_arm": self.semantic_arm,
            }
        )
        path = write_payload(self.payload_root, qid=item.qid, files=paths, meta=meta)
        return ProducedArtifact(path, time.time())

    def close(self) -> None:
        """Release the chain capture engine and the view aliased onto it."""
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


__all__ = (
    "QWEN_CHAIN_REQUEST_DEADLINE_SECONDS",
    "QwenChainLatentProducer",
    "QwenChainTextProducer",
)
