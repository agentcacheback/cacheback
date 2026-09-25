"""Per-ticket Gemma receiver for the split-fleet runtime.

One process, one GPU, one arm. Items arrive as claimed tickets rather than as a
panel, so this receiver holds no capture state of its own.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rcc.models.gemma.capture_types import CaptureArtifact
from rcc.models.gemma.contract import MAX_NUM_SEQS, WORKERS_PER_ITEM
from rcc.models.gemma.engine_roles import SPLIT_RECEIVER_ENGINE_IDENTITY
from rcc.models.gemma.handoff import load_capture_artifact, read_handoff_clocks
from rcc.models.gemma.receiver_core import GemmaReceiverCore
from rcc.models.gemma.schedule import (
    TEXT_ARMS,
    GemmaReceiverScheduleMixin,
    PendingArm,
    build_cell_rows,
)
from rcc.run.fleet.clocks import HandoffClocks, read_bundle_clocks
from rcc.run.fleet.contract import ProducedArtifact, ReceiverCompletion, WorkItem
from rcc.run.fleet.latency import ITEM_LATENCY_FIELDS
from rcc.run.gemma.split_text import TEXT_BUNDLE_CLOCKS, read_text_bundle

Bank = Callable[[Mapping[str, Any]], dict[str, Any]]


@dataclass
class _Spill:
    """The two local disk clocks charged to one item, never to the caller."""

    load_s: float = 0.0
    save_s: float = 0.0


@dataclass
class _Handoff:
    """What one claimed ticket's published bytes cost, and what they carry."""

    capture: CaptureArtifact
    spill: _Spill
    #: The producer's own capture-and-select wall, ``None`` when no producer
    #: ran at all: the direct arm ships nothing and the fused arm is timed by
    #: the receiver that produced it.
    producer_s: float | None = None


@dataclass
class _Live:
    """One claimed item in flight on this receiver."""

    state: PendingArm
    spill: _Spill
    admitted_at: float | None = None
    stage_times: dict[str, float] = field(default_factory=dict[str, float])
    #: When this item's own preparation began, on the process clock, so the
    #: sibling preparations that stalled its decode can be counted at the end.
    submitted_mono: float = 0.0


def issue_only_capture(qid: str) -> CaptureArtifact:
    """Return the empty capture the direct arm serves: nothing crosses the wire."""
    return CaptureArtifact(
        qid=qid,
        keeps_by_arm={"floor": ((),) * WORKERS_PER_ITEM},
        layouts_by_arm={},
        rolled_by_worker=(),
        selection_s_by_arm={"floor": 0.0},
        audition_s_by_ratio={},
        audition_replays_by_ratio={},
        reports=(),
        report_failed=False,
        report_failure=None,
        capture_s=0.0,
        capture_s_by_worker=(0.0,) * WORKERS_PER_ITEM,
        report_generation_s=0.0,
        embedding_digests={},
        global_layers=(),
        reloaded_captures=0,
        reloaded_selections=0,
    )


def load_handoff(
    manifest_path: Path,
    *,
    qid: str,
    arm: str,
    shared_embedding_digests: Mapping[str, str],
) -> tuple[CaptureArtifact, HandoffClocks]:
    """Load one verified handoff and the producer clocks bound to it.

    ``load_capture_artifact`` re-derives the manifest fingerprint and refuses any
    drift; the arm and table digest given here refuse a foreign selection by name.
    """
    artifact = load_capture_artifact(
        manifest_path,
        qid=qid,
        arm=arm,
        shared_embedding_digests=shared_embedding_digests,
    )
    body: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    clocks = read_handoff_clocks(manifest_path, qid=qid, fingerprint=str(body["fingerprint"]))
    return artifact, clocks


class GemmaSplitReceiver(GemmaReceiverCore, GemmaReceiverScheduleMixin):
    """One warm receiver engine serving one arm from the fleet ticket queue."""

    def __init__(
        self,
        *,
        arm: str,
        v16_arm: str,
        items: Mapping[str, dict[str, Any]],
        bank: Bank,
        tokenizer: Any,
        shared_weight: Path,
        runtime_fingerprint: str,
        publication_id: str,
        engine: Any,
        model_load_s: float = 0.0,
        seed_resolver: Any = None,
        flat_interleave_arms: tuple[str, ...] = (),
        result_row_adapter: Callable[[dict[str, Any], str], dict[str, Any]] | None = None,
        shared_weight_wait_s: float = 0.0,
    ) -> None:
        """Bind one arm, its prepared items, and the receiver-owned engine.

        ``arm`` is the shared semantic identity every banked row carries;
        ``v16_arm`` is the physical Gemma route whose selection this assembles.
        """
        self.arm = arm
        self.v16_arm = v16_arm
        self.items = dict(items)
        self.bank = bank
        self.result_row_adapter = result_row_adapter
        self.text_report_bundles: dict[str, dict[str, dict[str, Any]]] = {}
        self.live: dict[str, _Live] = {}
        self.claim_limit = 1
        # Every preparation this receiver ran, as (start, wall) on the process
        # clock. A claim prepares between two engine steps, so the items
        # already decoding pay for it, and the count and wall are banked.
        self._prepares: list[tuple[float, float]] = []
        self._decoder_unix_offset: float | None = None
        core: dict[str, Any] = {} if seed_resolver is None else {"seed_resolver": seed_resolver}
        super().__init__(
            tokenizer=tokenizer,
            shared_weight=shared_weight,
            runtime_fingerprint=runtime_fingerprint,
            publication_id=publication_id,
            engine=engine,
            model_load_s=model_load_s,
            flat_interleave_arms=flat_interleave_arms,
            engine_contract=SPLIT_RECEIVER_ENGINE_IDENTITY,
            shared_weight_wait_s=shared_weight_wait_s,
            **core,
        )

    def warm(self) -> None:
        """Warm the receiver engine and bank the phase row."""
        super().warm()
        self._bind_decoder_clock()
        self.claim_limit = MAX_NUM_SEQS
        self.bank(
            {
                "kind": "phase",
                "phase": "gemma_split_receiver_warm",
                "arm": self.arm,
                "engine_role": SPLIT_RECEIVER_ENGINE_IDENTITY.role,
                "engine_identity_sha256": self.engine_identity_sha256,
                "model_load_s": round(self.model_load_s, 4),
                "warmup_s": round(self.warmup_s, 4),
                "kv_pool_tokens": self.kv_pool_tokens,
                "claim_limit": self.claim_limit,
            }
        )

    def _bind_decoder_clock(self) -> None:
        """Bind the decoder's monotonic clock to the fleet's Unix clock."""
        if self.decoder is None:
            raise RuntimeError("Gemma split receiver has no warm decoder")
        self._decoder_unix_offset = time.time() - self.decoder.clock()

    def _completion_unix(self, value: float) -> float:
        """Project one decoder timestamp into the fleet's clock domain."""
        if self._decoder_unix_offset is None:
            raise RuntimeError("Gemma split receiver decoder clock is not bound")
        return value + self._decoder_unix_offset

    def can_accept(self) -> bool:
        """Return whether one more ticket may enter receiver preparation."""
        return self.tracker is not None and len(self.live) < self.claim_limit

    def _payload(self, qid: str, artifact: ProducedArtifact | None) -> _Handoff:
        """Read whatever this arm's producer published, on the spill clock."""
        if self.v16_arm == "floor":
            if artifact is not None:
                raise RuntimeError(f"{qid}/{self.arm}: the direct arm admits no producer artifact")
            return _Handoff(issue_only_capture(qid), _Spill())
        if artifact is None:
            raise RuntimeError(f"{qid}/{self.arm}: receiver claimed a ticket with no handoff")
        started = time.perf_counter()
        if self.v16_arm in TEXT_ARMS:
            bundle = read_text_bundle(artifact.path, qid=qid, semantic_arm=self.arm)
            save_s = read_bundle_clocks(
                TEXT_BUNDLE_CLOCKS, artifact.path, qid=qid, semantic_arm=self.arm
            )
            spill = _Spill(load_s=time.perf_counter() - started, save_s=save_s)
            self.text_report_bundles.setdefault(self.v16_arm, {})[qid] = bundle
            return _Handoff(issue_only_capture(qid), spill)
        capture, clocks = load_handoff(
            artifact.path,
            qid=qid,
            arm=self.v16_arm,
            shared_embedding_digests=self.embedding_digests,
        )
        return _Handoff(
            capture,
            _Spill(load_s=time.perf_counter() - started, save_s=clocks.spill_save_s),
            producer_s=clocks.producer_s,
        )

    def submit(self, item: WorkItem, artifact: ProducedArtifact | None) -> None:
        """Read the handoff, assemble the payload, and offer the timed draw."""
        if self.tracker is None:
            raise RuntimeError("Gemma split receiver is not warm")
        started = time.perf_counter()
        source = self.items[item.qid]
        handoff = self._payload(item.qid, artifact)
        capture = handoff.capture
        state = PendingArm(
            item=source,
            capture=capture,
            arm=self.v16_arm,
            prepared=self._prepare(source, capture, self.v16_arm),
            seeds=self._arm_seeds(item.qid, self.v16_arm),
            greedy_visible=None,
            greedy_loose=None,
            producer_s=handoff.producer_s,
            completions=[],
        )
        job_id = self._job_id(item.qid, self.v16_arm)
        self.live[job_id] = _Live(state=state, spill=handoff.spill, submitted_mono=started)
        self._prepares.append((started, time.perf_counter() - started))
        self._offer(job_id, state, (0,))

    def _sibling_prepares(self, since: float) -> dict[str, Any]:
        """Count and price the preparations that ran while one item was live."""
        now = time.perf_counter()
        walls = [wall for began, wall in self._prepares if since < began < now]
        oldest = min((live.submitted_mono for live in self.live.values()), default=now)
        self._prepares = [entry for entry in self._prepares if entry[0] >= oldest]
        return {"sibling_prepares": len(walls), "sibling_prepare_s": round(sum(walls), 4)}

    def pump(self) -> list[ReceiverCompletion]:
        """Step decode and surface finished cells: s0 alone, then s1 and s2."""
        if self.tracker is None:
            return []
        finished: list[ReceiverCompletion] = []
        for job_id, admitted_at, completions in self.tracker.pump():
            live = self.live[job_id]
            state = live.state
            if state.phase == 0:
                if len(completions) != 1 or completions[0].tag != "s0":
                    raise RuntimeError(f"{job_id}: first decode phase is not exactly s0")
                live.admitted_at = admitted_at
                state.timing_origin = completions[0].submitted_at
                state.completions = list(completions)
                state.phase = 1
                self._offer(job_id, state, (1, 2))
                continue
            state.completions = [*(state.completions or ()), *completions]
            finished.append(self._complete(job_id))
        return finished

    def _complete(self, job_id: str) -> ReceiverCompletion:
        live = self.live.pop(job_id)
        state = live.state
        rows = [_with_spill(row, live.spill) for row in build_cell_rows(self, state)]
        self.text_report_bundles.get(self.v16_arm, {}).pop(str(state.item["qid"]), None)
        # The two batched peers are complete rows of the same cell and are
        # banked here; the timed row the fleet clock prices is returned.
        for row in rows[1:]:
            self.bank(row)
        completions = sorted(state.completions or (), key=lambda row: int(row.tag[1:]))
        timed = completions[0]
        return ReceiverCompletion(
            qid=str(state.item["qid"]),
            result=rows[0],
            admitted_at=self._completion_unix(live.admitted_at or timed.submitted_at),
            first_token_at=self._completion_unix(timed.first_token_at or timed.finished_at),
            finished_at=self._completion_unix(max(row.finished_at for row in completions)),
            fields={
                **(dict(self.tracker.admission_trace.get(job_id, {})) if self.tracker else {}),
                **self._sibling_prepares(live.submitted_mono),
            },
        )

    def idle(self) -> bool:
        """Return whether no claimed item remains queued or decoding."""
        return not self.live and (self.tracker is None or self.tracker.idle())


class GemmaFusedReceiver(GemmaSplitReceiver):
    """The same-model text arm: this GPU generates the reports it then serves.

    ``text_primary`` is Gemma calling Gemma, so each receiver produces its own
    claimed item before serving it and the producer stage collapses to an instant.
    """

    def __init__(self, *, reports: Callable[[dict[str, Any]], dict[str, Any]], **kwargs: Any):
        """Bind the in-process report generator for the fused arm."""
        if kwargs.get("arm") != "text_primary" or kwargs.get("v16_arm") != "text_primary":
            raise ValueError("the fused Gemma route serves text_primary only")
        self._reports = reports
        self._stage_times: dict[str, dict[str, float]] = {}
        super().__init__(**kwargs)

    def can_accept(self) -> bool:
        """Take one item at a time: producing blocks this engine anyway."""
        return self.tracker is not None and not self.live and self.tracker.idle()

    def _payload(self, qid: str, artifact: ProducedArtifact | None) -> _Handoff:
        if artifact is not None:
            raise RuntimeError(f"{qid}/{self.arm}: the fused arm admits no producer artifact")
        self.text_report_bundles.setdefault(self.v16_arm, {})[qid] = self._reports(self.items[qid])
        boundary = time.time()
        self._stage_times[qid] = dict.fromkeys(
            ("producer_end", "handoff_ready", "receiver_claim", "payload_loaded"), boundary
        )
        return _Handoff(issue_only_capture(qid), _Spill())

    def stage_times(self, qid: str) -> dict[str, float]:
        """Return the collapsed fused producer and handoff stage clocks."""
        return self._stage_times.pop(qid)


def _with_spill(row: dict[str, Any], spill: _Spill) -> dict[str, Any]:
    """Add the two local disk clocks to a row that carries the rest."""
    missing = [name for name in ITEM_LATENCY_FIELDS if name not in row]
    if missing:
        raise RuntimeError(f"Gemma split row is missing latency fields {missing!r}")
    return {
        **row,
        "spill_load_s": round(spill.load_s, 4),
        "spill_save_s": round(spill.save_s, 4),
    }


__all__ = (
    "GemmaFusedReceiver",
    "GemmaSplitReceiver",
    "issue_only_capture",
    "load_handoff",
)
