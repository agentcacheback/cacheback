"""The route lane's receiver seat for the fleet runtime.

It holds one warm receiver engine, takes claimed items from the fleet queue,
assembles each prompt from its handoff, streams the samples, and banks a row.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch

from rcc.benchmarks.fanoutqa.payload import read_payload
from rcc.benchmarks.fanoutqa.prompts import render_chat
from rcc.benchmarks.fanoutqa.source_padding import ProbeItem
from rcc.benchmarks.longbench_v2.data import ChainItem
from rcc.benchmarks.longbench_v2.prompts import PROMPT_BUILDERS, hop_prompt_ids
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.qwen.backend import (
    QwenBackendSettings,
    QwenVllmBackend,
    build_qwen_backend,
)
from rcc.models.qwen.capture import QwenFlatPayload
from rcc.models.qwen.chain_text import generate_notes_chain
from rcc.models.qwen.engine import QWEN_ENGINE_MAX_NUM_SEQS
from rcc.models.qwen.prompts import manager_prompt_record, worker_prompts
from rcc.models.qwen.receiver import (
    QwenReceiverRequest,
    prepare_receiver_prompt,
    qwen_answer_closer,
    receiver_batches,
    receiver_requests,
    reconstruct_visible_answers,
)
from rcc.models.qwen.results import build_result_row
from rcc.models.qwen.text import generate_report_bundle
from rcc.models.route import Decoder, RouteFamily, tokenizer_decoder
from rcc.run.contract import PreparedItem
from rcc.run.fleet.admission import AdmissionGate
from rcc.run.fleet.contract import ProducedArtifact, ReceiverCompletion, WorkItem
from rcc.run.fleet.latency import WARMUP_BODY, ItemLatency, warmup_seed
from rcc.run.fleet.stream import ItemStreamTracker, StreamDecoder, StreamRequest
from rcc.run.fleet.vllm import VllmEngineHandle
from rcc.run.io import sha256_file
from rcc.run.nemotron.compiler_policy import close_backend, policy_fields
from rcc.run.qwen.producers import QwenLatentProducer, QwenTextProducer
from rcc.run.qwen.worker_contract import (
    QwenChannel,
)
from rcc.run.qwen.worker_contract import (
    physical_arm as _arm,
)
from rcc.run.qwen.worker_contract import (
    producer_backend as _producer_backend,
)
from rcc.run.qwen.worker_contract import (
    semantic_channel as _semantic_channel,
)
from rcc.run.qwen.worker_state import Pending, VisibleCompletion, chain_item, report_dict

Bank = Any


class QwenReceiver:
    """Stream one item's independent answer samples through a warm receiver."""

    def __init__(
        self,
        *,
        policy: str,
        tokenizer: Any,
        items: Mapping[str, PreparedItem],
        arm_root: Path,
        bank: Bank,
        family: RouteFamily,
        profile: BenchmarkProfile | None = None,
        fused: bool = False,
    ) -> None:
        """Bind one physical arm, the benchmark it serves, and its item lookup."""
        arm = _arm(policy, family=family)
        self.family = family
        self.decoder: Decoder = tokenizer_decoder(tokenizer)
        self.policy = policy
        self.semantic_arm = arm.semantic_arm
        self.channel: QwenChannel = _semantic_channel(arm)
        self.tokenizer = tokenizer
        self.items = dict(items)
        if profile is None:
            from rcc.run.qwen.adapter import execution_profile

            profile = execution_profile()
        self.profile = replace(family.benchmark_profile(profile), question_ids=tuple(items))
        self.arm_root = arm_root
        self.bank = bank
        self.fused = fused
        self.backend: QwenVllmBackend | None = None
        self.handle: VllmEngineHandle | None = None
        self.tracker: ItemStreamTracker | None = None
        self.embedding_weight: torch.Tensor | None = None
        self.pending: dict[str, Pending] = {}
        self._stage_times: dict[str, dict[str, float]] = {}
        # Every preparation this receiver ran, as (start, wall) on the process
        # clock. A claim prepares between two engine steps, so the items
        # already decoding pay for it, and the count and wall are banked.
        self._prepares: list[tuple[float, float]] = []
        self.claim_limit = 1

    def warm(self) -> None:
        """Open the receiver engine, decode one warmup, and bank the phase row."""
        started = time.perf_counter()
        profile = self.family.profile
        self.backend = build_qwen_backend(
            profile.checkpoint,
            profile.revision,
            settings=QwenBackendSettings(receiver=True),
            family=self.family,
            tokenizer=profile.tokenizer,
            tokenizer_revision=profile.tokenizer_revision,
        )
        load_s = time.perf_counter() - started
        # Before the fleet barrier, so this cost sits outside the service
        # window of whichever item is admitted first.
        warm_started = time.perf_counter()
        self.backend.warmup_decode(
            render_chat(
                self.tokenizer,
                WARMUP_BODY,
                enable_thinking=profile.decode.enable_thinking,
                system_prompt=self.family.thinking_system_prompt,
                assistant_prefill=self.family.assistant_prefill,
                low_effort=self.family.low_effort,
            ),
            seed=warmup_seed(self.policy, "receiver"),
        )
        warmup_s = time.perf_counter() - warm_started
        resident = self.backend.resident_model()
        embedding_weight = cast(torch.Tensor, resident.get_input_embeddings().weight)
        self.embedding_weight = embedding_weight
        self.handle = VllmEngineHandle(
            self.backend, require_public_enqueue=self.family.lane == "nemotron"
        )
        pool_tokens = self.backend.kv_pool_tokens()
        if pool_tokens is None:
            raise RuntimeError("route vLLM KV pool capacity is unreadable")
        gate = AdmissionGate(
            pool_tokens,
            answer_ceiling=self.profile.answer_ceiling,
            max_in_flight=16,
        )
        # The answer-side closer is bound on every lane. It cannot fire on a
        # draw that already carries a closer id, so it is inert on a lane whose
        # answers always close.
        self.tracker = ItemStreamTracker(
            StreamDecoder(
                self.handle,
                clock=time.time,
                closer=qwen_answer_closer(
                    embedding_weight,
                    self.tokenizer,
                    family=self.family,
                    profile=self.profile,
                ),
            ),
            gate,
        )
        # The engine's own sequence limit is the claim cap, and the gate above
        # holds the memory limit. A cap of pool // max_model_len instead would
        # reserve a whole worst-case context per claim.
        self.claim_limit = QWEN_ENGINE_MAX_NUM_SEQS
        self.bank(
            {
                "kind": "phase",
                "phase": f"{self.family.lane}_receiver_warm",
                **policy_fields(self.backend, self.family.lane),
                "policy": self.policy,
                "checkpoint": profile.checkpoint,
                "model_load_s": round(load_s, 4),
                "warmup_s": round(warmup_s, 4),
                "kv_pool_tokens": pool_tokens,
                "kv_capacity_tokens": gate.capacity_tokens,
                "claim_limit": self.claim_limit,
            }
        )

    def can_accept(self) -> bool:
        """Return whether another item may enter receiver preparation."""
        if self.tracker is None:
            return False
        if self.fused:
            return not self.pending and self.tracker.idle()
        return len(self.pending) < self.claim_limit

    def _text_bundle(self, item: PreparedItem) -> dict[str, object]:
        """Run this benchmark's own sender pass on the receiver's engine.

        A fused seat is the sender and the receiver at once, so it runs whatever
        the benchmark's prompt builder names.
        """
        if self.backend is None:
            raise RuntimeError("route receiver is not warm")
        if self.profile.prompt_builder in PROMPT_BUILDERS:
            return generate_notes_chain(
                self.backend,
                self.tokenizer,
                chain_item(item),
                qid=item.qid,
                semantic_arm="text_primary",
                family=self.family,
                decoder=self.decoder,
                profile=self.profile,
                # A fused seat renders through the profile's own tokenizer.
                ledger_tokenizer=self.tokenizer,
            ).result_fields()
        bundle = generate_report_bundle(
            self.backend,
            worker_prompts(item, self.tokenizer, family=self.family, profile=self.profile),
            qid=item.qid,
            semantic_arm="text_primary",
            family=self.family,
            decoder=self.decoder,
            profile=self.profile,
            prepared_item=item,
        )
        return report_dict(bundle)

    def _chain_prompt_counts(
        self, item: ChainItem, channel_fields: Mapping[str, object]
    ) -> tuple[int, ...]:
        """Return the prompt sizes this arm's hop seats consumed.

        Each channel records its own hops. The floor arm has no producer, so the
        receiver renders the hop prompts the item would have been read through.
        """
        banked = {"text": "report_prompt_tokens_by_hop", "latent": "hop_prompt_rows"}.get(
            self.channel
        )
        if banked is None:
            return tuple(
                len(
                    hop_prompt_ids(
                        self.tokenizer,
                        item.question,
                        item.choices,
                        item.chunks[hop - 1],
                        hop,
                        enable_thinking=self.family.profile.decode.enable_thinking,
                        family=self.family,
                    )
                )
                for hop in range(1, self.profile.workers_per_item + 1)
            )
        counts = channel_fields.get(banked)
        if not isinstance(counts, list):
            raise RuntimeError(
                f"{item.qid}/{self.semantic_arm}: the chain producer banked no {banked}"
            )
        return tuple(int(count) for count in cast(list[Any], counts))

    def _worker_prompt_record(
        self, item: PreparedItem, channel_fields: Mapping[str, object]
    ) -> tuple[int, ...]:
        """Return the per-worker prompt sizes this item's producers consumed."""
        if self.profile.prompt_builder in PROMPT_BUILDERS:
            return self._chain_prompt_counts(chain_item(item), channel_fields)
        if self.family.native_sender_prompts:
            from rcc.models.qwen.native_prompts import native_prompt_counts

            return native_prompt_counts(
                cast(ProbeItem, item),
                arm=self.semantic_arm,
                family=self.family,
                profile=self.profile,
            )
        return tuple(
            len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
            for text in worker_prompts(
                item, self.tokenizer, family=self.family, profile=self.profile
            )
        )

    def submit(self, item: WorkItem, artifact: ProducedArtifact | None) -> None:
        """Validate the handoff, prepare the prompt, and offer the timed sample."""
        if self.tracker is None or self.handle is None or self.embedding_weight is None:
            raise RuntimeError("route receiver is not warm")
        submitted = time.perf_counter()
        prepared = self.items[item.qid]
        report_bundle: dict[str, object] | None = None
        payload: QwenFlatPayload | None = None
        channel = self.channel
        channel_fields: dict[str, object] = {"producer_backend": _producer_backend(channel)}
        spill_load_s = 0.0
        if self.fused:
            report_bundle = self._text_bundle(prepared)
            boundary = time.time()
            self._stage_times[prepared.qid] = {
                name: boundary for name in ("producer_end", "handoff_ready", "receiver_claim")
            }
        elif artifact is not None:
            manifest = read_payload(artifact.path, qid=prepared.qid)
            raw_meta = manifest.get("meta")
            if not isinstance(raw_meta, dict):
                raise RuntimeError(f"{prepared.qid}: Qwen payload metadata is malformed")
            meta = cast(dict[str, object], raw_meta)
            if (
                meta.get("family") != self.family.model_id
                or meta.get("semantic_arm") != self.semantic_arm
            ):
                raise RuntimeError(f"{prepared.qid}: route payload arm identity differs")
            raw_bundle = manifest.get("report_bundle")
            if raw_bundle is not None:
                if not isinstance(raw_bundle, dict):
                    raise RuntimeError(f"{prepared.qid}: Qwen report bundle is malformed")
                report_bundle = cast(dict[str, object], raw_bundle)
            files = cast(dict[str, dict[str, object]], manifest["files"])
            if "embedding_rows" in files:
                started = time.perf_counter()
                rows = torch.load(
                    Path(str(files["embedding_rows"]["path"])),
                    map_location="cpu",
                    weights_only=True,
                )
                spill_load_s = time.perf_counter() - started
                if not isinstance(rows, torch.Tensor):
                    raise RuntimeError(f"{prepared.qid}: Qwen payload rows are not a tensor")
                keeps = tuple(
                    tuple(map(int, keep)) for keep in cast(list[list[int]], meta["keeps_by_worker"])
                )
                # `latent_rows_by_worker` is the blocks the receiver is handed,
                # three on the fan-out and one on the chain, so it is checked
                # against the file it describes, not read as worker geometry.
                shipped = tuple(map(int, cast(list[int], meta["latent_rows_by_worker"])))
                if sum(shipped) != int(rows.shape[0]):
                    raise RuntimeError(
                        f"{prepared.qid}: Qwen payload manifest ships {sum(shipped)} rows "
                        f"where its file carries {int(rows.shape[0])}"
                    )
                payload = QwenFlatPayload(
                    rows=rows,
                    semantic_arm=self.semantic_arm,
                    latent_plan_sha256=str(meta["payload_plan_sha256"]),
                    keeps=keeps,
                    rows_by_worker=tuple(len(keep) for keep in keeps),
                    selected_indices_sha256=str(meta["selected_indices_sha256"]),
                    tensor_sha256=str(meta["payload_tensor_sha256"]),
                    family=self.family,
                    profile=self.profile,
                    layout=str(meta["payload_layout"]),
                )
            # The manifest merges into the fields the seat seeded rather than
            # replacing them, so a row keeps the channel's own backend unless
            # the manifest names one itself.
            dropped = {
                "family",
                "semantic_arm",
                "policy",
                "arm",
                "channel",
                "decode",
                "decode_profile",
                "decode_fingerprint",
                "latent_tokens",
                "payload_layout",
                "payload_semantic_arm",
                "payload_plan_sha256",
                "payload_tensor_sha256",
            }
            channel_fields.update({k: v for k, v in meta.items() if k not in dropped})
            channel_fields["payload_file_sha256"] = {
                name: str(entry["sha256"]) for name, entry in files.items()
            }
            channel_fields["payload_manifest_sha256"] = sha256_file(artifact.path)
        if self.family.native_sender_prompts and report_bundle is not None:
            from rcc.models.qwen.native_results import validate_native_report

            validate_native_report(
                prepared,
                report_bundle,
                arm=self.semantic_arm,
                family=self.family,
                profile=self.profile,
            )
        reports = (
            cast(list[str], report_bundle["reports"])
            if report_bundle is not None and not report_bundle.get("report_failed")
            else []
        )
        if report_bundle is not None:
            channel_fields.update(
                {key: value for key, value in report_bundle.items() if key != "decode"}
            )
        in_turn = payload is not None and self.family.payload_in_user_turn
        record = manager_prompt_record(
            prepared,
            self.tokenizer,
            family=self.family,
            profile=self.profile,
            reports=reports,
            channel=channel,
            payload_slot=in_turn,
        )
        started = time.perf_counter()
        prompt = prepare_receiver_prompt(
            self.embedding_weight,
            record.token_ids,
            payload=payload,
            profile=self.profile,
            family=self.family,
            qid=prepared.qid,
            semantic_arm=self.semantic_arm,
            payload_slot=record.payload_slot,
            payload_headers=record.payload_headers,
        )
        if payload is not None:
            channel_fields["receiver_placement"] = "user-turn" if in_turn else "prepend"
        receiver_prepare_s = time.perf_counter() - started
        stream_prompt: dict[str, object] = (
            {"prompt_embeds": prompt.prompt_embeds.to(device="cpu", dtype=torch.bfloat16)}
            if payload is not None
            else {"prompt_token_ids": list(record.token_ids)}
        )
        requests = receiver_requests(
            prepared.qid,
            self.semantic_arm,
            prompt,
            family=self.family,
            profile=self.profile,
        )
        worker_counts = self._worker_prompt_record(prepared, channel_fields)
        self.pending[prepared.qid] = Pending(
            item=prepared,
            prompt=prompt,
            requests=requests,
            stream_prompt=stream_prompt,
            worker_prompt_tokens=worker_counts,
            channel_fields=channel_fields,
            receiver_prepare_s=receiver_prepare_s,
            spill_load_s=spill_load_s,
            submitted_mono=submitted,
        )
        self._prepares.append((submitted, time.perf_counter() - submitted))
        if self.fused:
            self._stage_times[prepared.qid]["payload_loaded"] = time.time()
        first, _rest = receiver_batches(requests)
        self._offer(prepared.qid, first)

    def _offer(self, qid: str, requests: tuple[QwenReceiverRequest, ...]) -> None:
        assert self.tracker is not None and self.handle is not None
        state = self.pending[qid]
        stream: list[StreamRequest] = []
        for request in requests:
            # The handle reads a missing top_k as off, so the seat passes the
            # family's own sampling fields and adds none of its own.
            sampling = SimpleNamespace(**request.sampling)
            stream.append(
                StreamRequest(
                    request_id=request.request_id,
                    prompt=state.stream_prompt,
                    sampling=self.handle.sampling(sampling, request.seed),
                    prompt_tokens=state.prompt.prompt_rows,
                    qid=qid,
                    tag=request.tag,
                )
            )
        self.tracker.offer(qid, stream)

    def pump(self) -> list[ReceiverCompletion]:
        """Step decode and surface the result row of each completed cell."""
        if self.tracker is None:
            return []
        finished: list[ReceiverCompletion] = []
        for qid, admitted_at, completions in self.tracker.pump():
            state = self.pending[qid]
            if state.phase == 0:
                state.admitted_at = admitted_at
                state.admission_fields = dict(self.tracker.admission_trace.get(qid, {}))
                state.completions.extend(completions)
                state.phase = 1
                _first, rest = receiver_batches(state.requests)
                self._offer(qid, rest)
                continue
            state.completions.extend(completions)
            finished.append(self._finalize(qid))
        return finished

    def _producer_seconds(self, channel_fields: Mapping[str, object]) -> float:
        """Return the producer seconds this arm's own channel performed."""
        channel = self.channel
        if channel == "latent":
            names = ("extract_s", "latent_roll_s", "selector_capture_s")
        elif channel == "text":
            names = ("report_generation_s",)
        else:
            return 0.0
        return sum(float(cast(float, channel_fields.get(name, 0.0))) for name in names)

    def _latency(self, state: Pending) -> ItemLatency:
        """Order the cell by sample tag and price it on one shared origin."""
        by_tag = {completion.tag: completion for completion in state.completions}
        tags = tuple(request.tag for request in state.requests)
        if set(by_tag) != set(tags) or len(by_tag) != len(state.completions):
            raise RuntimeError(f"{state.item.qid}: route receiver cell sample roster differs")
        ordered = tuple(by_tag[tag] for tag in tags)
        timed, rest = ordered[0], ordered[1:]
        origin = timed.submitted_at
        return ItemLatency(
            producer_s=self._producer_seconds(state.channel_fields),
            receiver_prepare_s=state.receiver_prepare_s,
            decode_s=timed.finished_at - timed.submitted_at,
            decode_batched_s=(
                max(completion.finished_at for completion in rest)
                - min(completion.submitted_at for completion in rest)
                if rest
                else 0.0
            ),
            receiver_ttft_s=(timed.first_token_at or timed.finished_at) - timed.submitted_at,
            generation_s=timed.finished_at - (timed.first_token_at or timed.finished_at),
            queued_offsets=tuple(completion.submitted_at - origin for completion in ordered),
            first_token_offsets=tuple(
                (completion.first_token_at or completion.finished_at) - origin
                for completion in ordered
            ),
            finished_offsets=tuple(completion.finished_at - origin for completion in ordered),
            spill_load_s=state.spill_load_s,
            continuation_submit_offsets=tuple(
                None
                if completion.continuation_submitted_at is None
                else completion.continuation_submitted_at - origin
                for completion in ordered
            ),
            continuation_first_token_offsets=tuple(
                None
                if completion.continuation_first_token_at is None
                else completion.continuation_first_token_at - origin
                for completion in ordered
            ),
        )

    def _sibling_prepares(self, since: float) -> dict[str, Any]:
        """Count and price the preparations that ran while one item was live."""
        now = time.perf_counter()
        walls = [wall for began, wall in self._prepares if since < began < now]
        oldest = min((state.submitted_mono for state in self.pending.values()), default=now)
        self._prepares = [entry for entry in self._prepares if entry[0] >= oldest]
        return {"sibling_prepares": len(walls), "sibling_prepare_s": round(sum(walls), 4)}

    def _finalize(self, qid: str) -> ReceiverCompletion:
        state = self.pending.pop(qid)
        readouts = tuple(
            VisibleCompletion(
                request_id=completion.request_id,
                text=completion.text,
                token_ids=completion.token_ids,
                n_tokens=completion.n_tokens,
                finish_reason=completion.finish_reason,
                num_cached_tokens=completion.num_cached_tokens,
                answer_injected=completion.answer_injected,
            )
            for completion in state.completions
        )
        visible = reconstruct_visible_answers(
            self.tokenizer, state.requests, readouts, family=self.family
        )
        first = state.completions[0]
        row = build_result_row(
            state.item,
            self.semantic_arm,
            visible,
            state.prompt,
            worker_prompt_tokens=state.worker_prompt_tokens,
            latency=self._latency(state),
            family=self.family,
            channel_fields=state.channel_fields,
            profile=self.profile,
        )
        return ReceiverCompletion(
            qid=qid,
            result=row,
            admitted_at=state.admitted_at or first.submitted_at,
            first_token_at=first.first_token_at or first.finished_at,
            finished_at=max(completion.finished_at for completion in state.completions),
            fields={**state.admission_fields, **self._sibling_prepares(state.submitted_mono)},
        )

    def stage_times(self, qid: str) -> dict[str, float]:
        """Return the collapsed fused producer and handoff stage clocks."""
        return self._stage_times.pop(qid)

    def idle(self) -> bool:
        """Return whether no receiver item remains queued or live."""
        return not self.pending and (self.tracker is None or self.tracker.idle())

    def close(self) -> None:
        """Abort live requests and release the receiver engine."""
        if self.tracker is not None:
            self.tracker.decoder.abort_all()
        self.tracker = None
        self.handle = None
        self.embedding_weight = None
        if self.backend is not None:
            close_backend(
                self.backend,
                self.bank,
                lane=self.family.lane,
                policy=self.policy,
                role="receiver",
            )
            self.backend = None


__all__ = ("QwenLatentProducer", "QwenReceiver", "QwenTextProducer")
