"""Per-ticket Ministral receiver for the split-fleet runtime.

One process, one GPU, one arm. Items arrive as claimed tickets rather than as a
panel, and decode runs in two phases, so an item never competes with its peers.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.fanoutqa.ministral_construction import MinistralConstructionManifest
from rcc.benchmarks.fanoutqa.scoring import DEFAULT_SCORER_VERSION, score_text
from rcc.models.ministral.handoff import read_handoff, read_handoff_clocks
from rcc.models.ministral.payload import MinistralFlatPayload
from rcc.models.ministral.receiver import receiver_batches
from rcc.models.ministral.receiver_core import (
    MinistralPreparedCell,
    MinistralReceiverCell,
    MinistralReceiverCore,
    manager_prompt_tokens,
)
from rcc.models.ministral.text import MinistralReportBundle
from rcc.models.ministral.text_codec import MINISTRAL_TEXT_ARMS, MinistralTextCodec
from rcc.run.fleet.clocks import read_bundle_clocks
from rcc.run.fleet.contract import ProducedArtifact, ReceiverCompletion, WorkItem
from rcc.run.fleet.latency import ITEM_LATENCY_FIELDS
from rcc.run.fleet.stream import StreamCompletion
from rcc.run.ministral.split_producer import payload_id
from rcc.run.ministral.split_text import TEXT_BUNDLE_CLOCKS, read_text_bundle

Bank = Callable[[Mapping[str, Any]], dict[str, Any]]

#: The direct arm: no producers, receivers only, nothing on the wire.
DIRECT_ARM = "issue_only"
#: The fused arm: one GPU produces the reports it then serves.
FUSED_ARM = "text_primary"
#: The sample the fleet clock prices, and the only row whose bank completes a
#: cell. Its two batched peers are banked before it, so a crash between those
#: writes leaves the item open rather than two thirds complete.
TIMED_SAMPLE_TAG = FANOUTQA_NATURAL_DEV50.sample_tags[0]


@dataclass
class _Live:
    """One claimed cell moving through the two decode phases."""

    prepared: MinistralPreparedCell
    submitted_mono: float
    #: 0 while the timed s0 sample decodes alone, 1 once s1 and s2 are offered.
    phase: int = 0
    admitted_at: float | None = None
    admission_fields: dict[str, Any] = field(default_factory=dict[str, Any])
    completions: list[StreamCompletion] = field(default_factory=list[StreamCompletion])


def load_handoff(
    manifest_path: Path,
    *,
    qid: str,
    semantic_arm: str,
) -> tuple[MinistralFlatPayload, float, float]:
    """Load one verified payload and the producer clocks bound to it.

    ``read_handoff`` re-derives the manifest fingerprint and verifies every file
    digest, so the clock record beside it holds the clocks for exactly these bytes.
    """
    identity = payload_id(qid, semantic_arm)
    payload = read_handoff(manifest_path, payload_id=identity)
    body: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    clocks = read_handoff_clocks(
        manifest_path,
        payload_id=identity,
        fingerprint=str(body["fingerprint"]),
    )
    return payload, clocks.spill_save_s, clocks.producer_s


class MinistralSplitReceiver:
    """One warm receiver engine serving one arm from the fleet ticket queue."""

    def __init__(
        self,
        *,
        arm: str,
        items: Mapping[str, dict[str, Any]],
        bank: Bank,
        core: MinistralReceiverCore,
        report_codecs: Mapping[str, MinistralTextCodec] | None = None,
    ) -> None:
        """Bind one arm, its prepared items, and the receiver-owned core."""
        self.arm = arm
        self.items = dict(items)
        self.bank = bank
        self.core = core
        self.report_codecs = dict(report_codecs or {})
        self.live: dict[str, _Live] = {}
        self._prepares: list[tuple[float, float]] = []
        self._peak_prompt_bytes = 0

    def warm(self) -> None:
        """Warm the receiver engine and bank the phase row."""
        self.core.warm()
        self.bank(
            {
                "kind": "phase",
                "phase": "ministral_split_receiver_warm",
                "arm": self.arm,
                "engine_role": self.core.engine_contract.engine_role,
                "model_load_s": round(self.core.model_load_s, 4),
                "warmup_s": round(self.core.warmup_s, 4),
                "kv_pool_tokens": self.core.kv_pool_tokens,
                "kv_capacity_tokens": self.core.capacity_tokens,
                "claim_limit": self.claim_limit,
            }
        )

    @property
    def claim_limit(self) -> int:
        """Return the claim cap: the engine's own sequence limit."""
        return self.core.claim_limit

    def can_accept(self) -> bool:
        """Return whether one more ticket may enter receiver preparation.

        Claims are counted, not tokens. The core holds the memory limit and
        back-pressures by queueing, so a claim past the KV pool is delayed.
        """
        return len(self.live) < self.claim_limit

    def _payload(
        self,
        qid: str,
        artifact: ProducedArtifact | None,
    ) -> tuple[MinistralFlatPayload | None, MinistralReportBundle | None, float, float, float]:
        """Read whatever this arm's producer published, on the spill clock."""
        if self.arm == DIRECT_ARM:
            if artifact is not None:
                raise RuntimeError(f"{qid}/{self.arm}: the direct arm admits no producer artifact")
            return None, None, 0.0, 0.0, 0.0
        if artifact is None:
            raise RuntimeError(f"{qid}/{self.arm}: receiver claimed a ticket with no handoff")
        started = time.perf_counter()
        if self.arm in MINISTRAL_TEXT_ARMS:
            bundle = read_text_bundle(
                artifact.path, qid=qid, semantic_arm=self.arm, profile=self.core.profile
            )
            # Reading the clock record rehashes the bundle, so it both yields the
            # producer's clock and refuses a rewritten bundle.
            save_s = read_bundle_clocks(
                TEXT_BUNDLE_CLOCKS, artifact.path, qid=qid, semantic_arm=self.arm
            )
            load_s = time.perf_counter() - started
            return None, bundle, load_s, save_s, bundle.generation_s
        payload, save_s, producer_s = load_handoff(
            artifact.path,
            qid=qid,
            semantic_arm=self.arm,
        )
        return payload, None, time.perf_counter() - started, save_s, producer_s

    def _cell(
        self,
        qid: str,
        artifact: ProducedArtifact | None,
    ) -> MinistralReceiverCell:
        payload, bundle, load_s, save_s, producer_s = self._payload(qid, artifact)
        source = self.items[qid]
        manager, _record = manager_prompt_tokens(
            self.core.receiver_codec,
            qid=qid,
            semantic_arm=self.arm,
            question=str(source["question"].question),
            reports=() if bundle is None or bundle.report_failed else bundle.reports,
            report_failed=bundle is not None and bundle.report_failed,
            profile=self.core.profile,
        )
        return MinistralReceiverCell(
            qid=qid,
            semantic_arm=self.arm,
            manager_token_ids=manager,
            producer_s=producer_s,
            payload=payload,
            report_bundle=bundle,
            report_codec=self.report_codecs.get(self.arm),
            spill_load_s=load_s,
            spill_save_s=save_s,
        )

    def submit(self, item: WorkItem, artifact: ProducedArtifact | None) -> None:
        """Read the handoff, sign the manager turn, and offer the timed sample."""
        if item.qid in self.live:
            raise RuntimeError(f"{item.qid}/{self.arm}: receiver already holds this item")
        submitted = time.perf_counter()
        prepared = self.core.prepare(self._cell(item.qid, artifact))
        self.live[item.qid] = _Live(prepared=prepared, submitted_mono=submitted)
        self._prepares.append((submitted, time.perf_counter() - submitted))
        self._peak_prompt_bytes = max(self._peak_prompt_bytes, self._resident_prompt_bytes())
        first, _rest = receiver_batches(prepared.requests)
        self.core.offer(prepared, first)

    def pump(self) -> list[ReceiverCompletion]:
        """Step the engine once and surface every cell that just completed."""
        finished: list[ReceiverCompletion] = []
        for qid, admitted_at, completions in self.core.pump():
            live = self.live[qid]
            if live.phase == 0:
                if len(completions) != 1 or completions[0].tag != TIMED_SAMPLE_TAG:
                    raise RuntimeError(f"{qid}/{self.arm}: first decode phase is not exactly s0")
                # The timed sample entering the batch is the item's service
                # start. Its trace is copied here because offering s1 and s2
                # below overwrites the tracker's entry for this same qid.
                live.admitted_at = admitted_at
                live.admission_fields = dict(self.core.admission_trace.get(qid, {}))
                live.completions.extend(completions)
                live.phase = 1
                _first, rest = receiver_batches(live.prepared.requests)
                self.core.offer(live.prepared, rest)
                continue
            live.completions.extend(completions)
            finished.append(self._complete(qid))
        return finished

    def _complete(self, qid: str) -> ReceiverCompletion:
        """Build, bank, and price one finished three-sample cell."""
        live = self.live.pop(qid)
        question = self.items[qid]["question"]
        rows = self.core.build_rows(
            live.prepared,
            live.completions,
            score=lambda answer: score_text(question, answer),
        )
        banked = [
            self._bank_row({**row, "scoring_version": DEFAULT_SCORER_VERSION}) for row in rows
        ]
        timed_row = banked[0]
        if timed_row["sample_tag"] != TIMED_SAMPLE_TAG:
            raise RuntimeError(f"{qid}/{self.arm}: the priced Ministral seed is not the timed one")
        # The two batched peers are banked here; the runtime banks the timed
        # row after them, and only that write completes the item.
        for row in banked[1:]:
            self.bank(row)
        timed = live.completions[0]
        return ReceiverCompletion(
            qid=qid,
            result=timed_row,
            admitted_at=live.admitted_at or timed.submitted_at,
            first_token_at=timed.first_token_at or timed.finished_at,
            finished_at=max(completion.finished_at for completion in live.completions),
            fields={**live.admission_fields, **self._sibling_prepares(live.submitted_mono)},
        )

    def _resident_prompt_bytes(self) -> int:
        """Return host bytes held by every claimed cell's embedding prompt.

        One dense prompt is held per claimed cell, the largest object either side
        of the wire. The core's memory limit bounds device KV, not this.
        """
        return sum(
            live.prepared.prompt.prompt_embeds.element_size()
            * live.prepared.prompt.prompt_embeds.nelement()
            for live in self.live.values()
        )

    def _sibling_prepares(self, since: float) -> dict[str, Any]:
        """Count and price the preparations that ran while one item was live.

        Preparation runs between engine steps, so a second claim's prompt build
        stalls the first claim's decode; that stall is banked per item.
        """
        now = time.perf_counter()
        walls = [wall for began, wall in self._prepares if since < began < now]
        oldest = min((live.submitted_mono for live in self.live.values()), default=now)
        self._prepares = [entry for entry in self._prepares if entry[0] >= oldest]
        return {
            "sibling_prepares": len(walls),
            "sibling_prepare_s": round(sum(walls), 4),
            "peak_resident_prompt_bytes": self._peak_prompt_bytes,
        }

    def _bank_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Project one built result row onto the fleet bank's row identity."""
        missing = [name for name in ITEM_LATENCY_FIELDS if name not in row]
        if missing:
            raise RuntimeError(f"Ministral split row is missing latency fields {missing!r}")
        qid = str(row["qid"])
        source_arm = self.arm if self.arm in MINISTRAL_TEXT_ARMS else FUSED_ARM
        construction = self.items[qid]["construction"].get(source_arm)
        if not isinstance(construction, MinistralConstructionManifest):
            raise RuntimeError(
                f"{qid}/{self.arm}: prepared panel lacks the {source_arm} construction manifest"
            )
        return {
            **dict(row),
            "kind": "result",
            "panel": str(self.items[qid]["panel"]),
            "cell": f"{self.arm}|{row['sample_tag']}",
            "construction_manifest": construction.to_dict(),
            "construction_manifest_fingerprint": construction.fingerprint,
            "construction_complete": construction.construction_complete,
            "construction_incomplete_reason": (
                "" if construction.construction_complete else "leaves_removed_by_packing"
            ),
        }

    def idle(self) -> bool:
        """Return whether no claimed item remains prepared or decoding."""
        return not self.live and self.core.idle()

    def close(self) -> None:
        """Release the receiver engine."""
        self.core.close()


class MinistralFusedReceiver(MinistralSplitReceiver):
    """The same-model text arm: this GPU generates the reports it then serves.

    ``text_primary`` is Ministral calling Ministral, so each receiver produces its
    own claimed item before serving it and the producer stage collapses.
    """

    def __init__(
        self,
        *,
        reports: Callable[[str], MinistralReportBundle],
        **kwargs: Any,
    ) -> None:
        """Bind the in-process report generator for the fused arm."""
        if kwargs.get("arm") != FUSED_ARM:
            raise ValueError("the fused Ministral route serves text_primary only")
        self._reports = reports
        self._stage_times: dict[str, dict[str, float]] = {}
        super().__init__(**kwargs)

    def can_accept(self) -> bool:
        """Take one cell at a time: this arm generates its own reports.

        The fused arm decodes its worker turns inside ``submit``, on the engine it
        then serves from, and that generation blocks the pump.
        """
        return not self.live and self.core.idle()

    def _cell(
        self,
        qid: str,
        artifact: ProducedArtifact | None,
    ) -> MinistralReceiverCell:
        if artifact is not None:
            raise RuntimeError(f"{qid}/{self.arm}: the fused arm admits no producer artifact")
        bundle = self._reports(qid)
        boundary = time.time()
        self._stage_times[qid] = dict.fromkeys(
            ("producer_end", "handoff_ready", "receiver_claim", "payload_loaded"), boundary
        )
        source = self.items[qid]
        manager, _record = manager_prompt_tokens(
            self.core.receiver_codec,
            qid=qid,
            semantic_arm=self.arm,
            question=str(source["question"].question),
            reports=() if bundle.report_failed else bundle.reports,
            report_failed=bundle.report_failed,
            profile=self.core.profile,
        )
        return MinistralReceiverCell(
            qid=qid,
            semantic_arm=self.arm,
            manager_token_ids=manager,
            # The producer clock is the span of the draw that shipped, which is
            # what the bundle carries. A wall clock would also charge this arm
            # for discarded draws; those stay as `draws_n` and `redraw_wall_s`.
            producer_s=bundle.generation_s,
            report_bundle=bundle,
            report_codec=self.report_codecs.get(self.arm),
        )

    def stage_times(self, qid: str) -> dict[str, float]:
        """Return the collapsed fused producer and handoff stage clocks."""
        return self._stage_times.pop(qid)


__all__ = (
    "DIRECT_ARM",
    "FUSED_ARM",
    "TIMED_SAMPLE_TAG",
    "MinistralFusedReceiver",
    "MinistralSplitReceiver",
    "load_handoff",
)
