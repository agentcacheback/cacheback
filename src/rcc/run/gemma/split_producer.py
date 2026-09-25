"""Persistent Gemma producers for the split-fleet runtime.

A producer process holds one GPU for the whole arm and publishes one handoff
manifest or report bundle per claimed item. Neither decodes an answer.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.gemma.capture_runtime import capture_panel
from rcc.models.gemma.capture_support import DeferredAudit, latent_roll_filename
from rcc.models.gemma.capture_types import CaptureArtifact
from rcc.models.gemma.contract import ARMS, QSNAP_ARMS, STOP_IDS, WORKERS_PER_ITEM
from rcc.models.gemma.embedding import (
    hardlink_shared_embedding,
    publish_shared_embedding_once,
    shared_embedding_path,
)
from rcc.models.gemma.engine import Sampling, build_engine, free_gpu_memory
from rcc.models.gemma.engine_roles import (
    CAPTURE_ENGINE_IDENTITY,
    PRODUCER_ROLE,
    build_capture_bridge,
    open_role_engine,
    verify_engine_identity,
)
from rcc.models.gemma.handoff import (
    handoff_manifest_path,
    write_handoff_clocks,
    write_handoff_manifest,
)
from rcc.models.gemma.mechanism import encode
from rcc.models.gemma.prompts import render_chat
from rcc.models.gemma.roster import flat_interleave_arm
from rcc.models.gemma.selection import payload_length
from rcc.models.gemma.text_bank import TextReportBankProtocol
from rcc.models.gemma.text_contract import bad_words, text_sender_spec
from rcc.models.gemma.text_reports import generate_report_bundle
from rcc.models.gemma.text_runtime import bind_text_prompt_item, report_engine_config
from rcc.models.gemma.tokenizer import load_gemma4_tokenizer
from rcc.run.fleet.clocks import write_bundle_clocks
from rcc.run.fleet.contract import ProducedArtifact, WorkItem
from rcc.run.fleet.latency import WARMUP_BODY, WARMUP_MAX_TOKENS, warmup_seed
from rcc.run.gemma.split_text import TEXT_BUNDLE_CLOCKS, text_bundle_path, write_text_bundle
from rcc.run.io import canonical_sha

Bank = Callable[[Mapping[str, Any]], dict[str, Any]]

#: ``capture_panel`` publishes the shared table from worker index 0 and validates
#: it from every other index. A split producer publishes the run-level table once
#: while warming, so its per-item captures take the validating path.
_CAPTURE_VALIDATES_ONLY = 1


class CaptureLedger:
    """The capture-clock sink a split producer offers to ``capture_panel``.

    The resident route hands capture a shard log to read a banked capture back
    from. A split producer works per ticket, so every claim captures once.
    """

    def __init__(self, bank: Bank, *, arm: str, engine_identity_sha256: str) -> None:
        """Bind one arm's fleet bank and the digest of the engine writing to it."""
        self.bank = bank
        self.arm = arm
        self.engine_identity_sha256 = engine_identity_sha256

    def banked_capture(self, qid: str, worker: int) -> None:
        """Return no banked capture: a claimed ticket is always new work."""
        del qid, worker
        return None

    def bank_capture(self, row: Mapping[str, Any], *, qid: str, worker: int) -> dict[str, Any]:
        """Record one worker's live capture clock on the arm bank."""
        return self.bank(
            {
                **dict(row),
                "kind": "producer",
                "phase": "gemma_split_capture",
                "arm": self.arm,
                "qid": qid,
                "worker": worker,
                # The bank records which engine produced these bytes, not only
                # that some producer did.
                "engine_identity_sha256": self.engine_identity_sha256,
            }
        )


class GemmaLatentProducer:
    """Keep the capture engine warm and publish one handoff per claimed item."""

    def __init__(
        self,
        *,
        arm: str,
        v16_arm: str,
        items: Mapping[str, dict[str, Any]],
        run_root: Path,
        handoff_root: Path,
        bank: Bank,
        runtime_fingerprint: str,
        publication_id: str,
        open_engine: Callable[[str], Any] = open_role_engine,
        bridge_factory: Callable[[Any], Any] = build_capture_bridge,
        capture: Callable[
            ..., tuple[tuple[CaptureArtifact, ...], dict[str, str], tuple[DeferredAudit, ...]]
        ] = capture_panel,
        verify: Callable[..., dict[str, Any]] = verify_engine_identity,
        publish: Callable[..., dict[str, str]] = publish_shared_embedding_once,
    ) -> None:
        """Bind one latent arm to its item panel and artifact root.

        ``arm`` is the shared semantic identity every banked row carries;
        ``v16_arm`` is the physical Gemma route the capture selects for.
        """
        if v16_arm not in ARMS and v16_arm not in QSNAP_ARMS:
            raise ValueError(
                f"{v16_arm!r}: not a registered Gemma v16 or mean query attention selection arm"
            )
        self.arm = arm
        self.v16_arm = v16_arm
        # A mean query attention arm banks its own selection beside the v16 one, so the
        # capture runs the mean query attention ladder with an empty v16 roster; the manifest
        # then names both banks and the receiver merges them.
        self.qsnap = v16_arm in QSNAP_ARMS
        self.capture_kwargs: dict[str, Any] = {
            "selection_arms": () if self.qsnap else (v16_arm,),
            **({"include_qsnap": True} if self.qsnap else {}),
            **({"flat_interleave_arms": (v16_arm,)} if flat_interleave_arm(v16_arm) else {}),
        }
        self.items = dict(items)
        self.run_root = run_root
        self.handoff_root = handoff_root
        self.bank = bank
        self.runtime_fingerprint = runtime_fingerprint
        self.publication_id = publication_id
        # The run publishes one table above the arms, and the arm bundle holds
        # the linked copy the manifest names, so a handoff stays relocatable.
        self.run_weight = shared_embedding_path(run_root)
        self.shared_weight = shared_embedding_path(handoff_root)
        self._open_engine = open_engine
        self._bridge_factory = bridge_factory
        self._capture = capture
        self._verify = verify
        self._publish = publish
        self.engine: Any | None = None
        self.bridge: Any | None = None
        self.engine_identity_sha256 = ""
        self.embedding_digests: dict[str, str] = {}

    def warm(self) -> None:
        """Open the producer engine, check its identity, and publish the table."""
        started = time.perf_counter()
        engine = self._open_engine(PRODUCER_ROLE)
        load_s = time.perf_counter() - started
        bridge: Any | None = None
        try:
            identity = self._verify(engine, CAPTURE_ENGINE_IDENTITY)
            bridge = self._bridge_factory(engine)
            self._publish_shared_table(bridge)
        except Exception:
            if bridge is not None:
                bridge.release_view()
            engine.close()
            raise
        self.engine = engine
        self.bridge = bridge
        self.engine_identity_sha256 = canonical_sha(identity)
        self.bank(
            {
                "kind": "phase",
                "phase": "gemma_split_producer_warm",
                "arm": self.arm,
                "engine_role": PRODUCER_ROLE,
                "engine_identity_sha256": self.engine_identity_sha256,
                "kv_connector": CAPTURE_ENGINE_IDENTITY.kv_connector,
                "kv_connector_module_path": CAPTURE_ENGINE_IDENTITY.kv_connector_module_path,
                "shared_embedding_marker_sha256": self.embedding_digests["marker_sha256"],
                "model_load_s": round(load_s, 4),
            }
        )

    def _publish_shared_table(self, bridge: Any) -> None:
        """Publish the run table once, then link it into this arm's bundle.

        Every arm of the run embeds from one table, so the publication sits above
        the arms and is idempotent. The arm bundle then hardlinks those files.
        """
        self.embedding_digests = self._publish(
            self.run_root,
            bridge.model.get_input_embeddings().weight,
            runtime_fingerprint=self.runtime_fingerprint,
            publication_id=self.publication_id,
        )
        hardlink_shared_embedding(
            self.run_root,
            self.handoff_root,
            publication_id=self.publication_id,
        )

    def produce(self, item: WorkItem) -> ProducedArtifact:
        """Capture, select, and publish one item's handoff manifest."""
        if self.bridge is None:
            raise RuntimeError("Gemma split producer is not warm")
        source = self.items[item.qid]
        # The whole wall a caller waits on before anything can ship: prefill,
        # side capture, roll, and the ranked cut for this arm.
        capture_started = time.perf_counter()
        artifacts, _embedding, audits = self._capture(
            (source,),
            bridge=self.bridge,
            artifact_root=self.handoff_root,
            shard_log=CaptureLedger(
                self.bank,
                arm=self.arm,
                engine_identity_sha256=self.engine_identity_sha256,
            ),
            worker_index=_CAPTURE_VALIDATES_ONLY,
            runtime_fingerprint=self.runtime_fingerprint,
            publication_id=self.publication_id,
            shared_weight=self.run_weight,
            **self.capture_kwargs,
        )
        producer_s = time.perf_counter() - capture_started
        capture = artifacts[0]
        manifest_path = handoff_manifest_path(self.handoff_root, item.qid)
        frame = source["frame"]
        # The cost of publishing: hashing every named file and writing the
        # manifest. A sender on a wire would not pay it, so it is banked beside
        # the caller clocks rather than inside them.
        started = time.perf_counter()
        write_handoff_manifest(
            manifest_path,
            qid=item.qid,
            rolled_paths=tuple(
                self.handoff_root / "rolled" / latent_roll_filename(item.qid, worker)
                for worker in range(WORKERS_PER_ITEM)
            ),
            selection_path=self.handoff_root / "selections" / f"{item.qid}.json",
            qsnap_selection_path=(
                self.handoff_root / "qsnap_selections" / f"{item.qid}.json" if self.qsnap else None
            ),
            embedding_weight=self.shared_weight,
            publication_id=self.publication_id,
            hidden=int(capture.rolled_by_worker[0].shape[1]),
            frame_prefix_length=len(frame.prefix_ids),
            frame_suffix_length=len(frame.suffix_ids),
            lengths=tuple(payload_length(int(ids.numel())) for ids in source["prompt_ids"]),
            embedding_digests=capture.embedding_digests,
            global_layers=capture.global_layers,
            audition_replays_by_ratio=capture.audition_replays_by_ratio,
        )
        spill_save_s = time.perf_counter() - started
        body: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
        write_handoff_clocks(
            manifest_path,
            qid=item.qid,
            fingerprint=str(body["fingerprint"]),
            spill_save_s=spill_save_s,
            producer_s=producer_s,
        )

        pending = list(audits)

        def after_publish() -> None:
            while pending:
                pending.pop(0).run()

        return ProducedArtifact(
            manifest_path,
            time.time(),
            {"spill_save_s": spill_save_s},
            after_publish=after_publish,
        )

    def close(self) -> None:
        """Release the capture view and the producer engine."""
        bridge, self.bridge = self.bridge, None
        engine, self.engine = self.engine, None
        try:
            if bridge is not None:
                bridge.release_view()
        finally:
            if engine is not None:
                engine.close()


class GemmaTextProducer:
    """Keep one text sender warm and publish one report bundle per item."""

    def __init__(
        self,
        *,
        arm: str,
        items: Mapping[str, dict[str, Any]],
        handoff_root: Path,
        bank: Bank,
        report_bank: TextReportBankProtocol,
        prepared_manifest: Mapping[str, Any],
        profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
        engine_builder: Callable[[Any], Any] = build_engine,
        tokenizer_loader: Callable[..., Any] = load_gemma4_tokenizer,
    ) -> None:
        """Bind one non-resident text sender to its item panel."""
        self.sender = text_sender_spec(arm)
        if self.sender.resident:
            raise ValueError(f"{arm}: the resident text sender runs on the fused route")
        self.arm = arm
        self.items = dict(items)
        self.prepared_manifest = dict(prepared_manifest)
        self.profile = profile
        self.handoff_root = handoff_root
        self.bank = bank
        self.report_bank = report_bank
        self._engine_builder = engine_builder
        self._tokenizer_loader = tokenizer_loader
        self.engine: Any | None = None
        self.tokenizer: Any | None = None

    def warm(self) -> None:
        """Open the sender checkpoint and decode once before any measurement."""
        sender = self.sender
        self.tokenizer = self._tokenizer_loader(
            checkpoint_id=sender.checkpoint,
            revision=sender.revision,
        )
        started = time.perf_counter()
        engine = self._engine_builder(report_engine_config(sender))
        self.engine = engine
        load_s = time.perf_counter() - started
        warm_started = time.perf_counter()
        engine.decode_token_ids_full(
            [encode(self.tokenizer, render_chat(self.tokenizer, WARMUP_BODY))],
            Sampling(
                temperature=0.0,
                max_tokens=WARMUP_MAX_TOKENS,
                stop_token_ids=tuple(sorted(STOP_IDS)),
                bad_words=bad_words(self.tokenizer, sender),
            ),
            seeds=[warmup_seed(sender.policy, "text_producer")],
        )
        self.bank(
            {
                "kind": "phase",
                "phase": "gemma_split_text_producer_warm",
                "arm": self.arm,
                "checkpoint": sender.checkpoint,
                "revision": sender.revision,
                "model_load_s": round(load_s, 4),
                "warmup_s": round(time.perf_counter() - warm_started, 4),
            }
        )

    def produce(self, item: WorkItem) -> ProducedArtifact:
        """Generate one item's three worker reports and publish the bundle."""
        if self.engine is None or self.tokenizer is None:
            raise RuntimeError("Gemma split text producer is not warm")
        bound = bind_text_prompt_item(
            self.tokenizer,
            self.items[item.qid],
            self.sender,
            prepared_manifest=self.prepared_manifest,
            profile=self.profile,
        )
        bundle = generate_report_bundle(
            self.engine,
            self.tokenizer,
            bound,
            self.sender,
            self.report_bank,
            profile=self.profile,
        )
        # The publish is the cost a wire would have paid, so it is measured
        # here and banked beside the caller clocks rather than inside them.
        started = time.perf_counter()
        path = write_text_bundle(
            text_bundle_path(self.handoff_root, item.qid),
            qid=item.qid,
            semantic_arm=self.arm,
            bundle=bundle,
        )
        spill_save_s = time.perf_counter() - started
        write_bundle_clocks(
            TEXT_BUNDLE_CLOCKS,
            path,
            qid=item.qid,
            semantic_arm=self.arm,
            spill_save_s=spill_save_s,
        )
        return ProducedArtifact(path, time.time(), {"spill_save_s": spill_save_s})

    def close(self) -> None:
        """Release the sender checkpoint and its GPU memory."""
        engine, self.engine = self.engine, None
        self.tokenizer = None
        if engine is not None:
            engine.close()
            free_gpu_memory(settle_seconds=1.0)


__all__ = ("CaptureLedger", "GemmaLatentProducer", "GemmaTextProducer")
