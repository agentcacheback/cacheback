"""Build the one Gemma worker a split-fleet seat calls for.

The placement decides the role and the arm decides the channel, and those two
together decide which of the four worker shapes this process is.
"""

from __future__ import annotations

import time
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.gemma.contract import CHECKPOINT_ID, CHECKPOINT_REVISION
from rcc.models.gemma.embedding import publish_shared_embedding_once, shared_embedding_path
from rcc.models.gemma.engine_roles import (
    PRODUCER_ROLE,
    RECEIVER_ROLE,
    build_capture_bridge,
    open_role_engine,
)
from rcc.models.gemma.roster import (
    flat_interleave_arm,
    gemma_v16_arm,
    project_shared_result_row,
    shared_seeds_for_v16_arm,
)
from rcc.models.gemma.schedule import TEXT_ARMS
from rcc.models.gemma.text_bank import TextReportBank
from rcc.models.gemma.tokenizer import load_gemma4_tokenizer
from rcc.run.contract import BankCallback, FleetWorker, WorkerContext
from rcc.run.gemma.split_producer import GemmaLatentProducer, GemmaTextProducer
from rcc.run.gemma.split_receiver import GemmaFusedReceiver, GemmaSplitReceiver

#: Where one arm's producers publish, and its receivers read, the bytes that
#: cross the process boundary: the linked shared table, the rolled rows, the
#: banked selection, and the manifest naming all three.
HANDOFF_DIRECTORY = "handoff"

#: How long a receiver waits for the run-level table its producers publish.
SHARED_EMBEDDING_WAIT_S = 3_600.0


def handoff_root(root: Path, arm: str) -> Path:
    """Return the cross-process bundle root for one arm."""
    return root / "arms" / arm / HANDOFF_DIRECTORY


def publish_run_shared_embedding(
    root: Path,
    *,
    runtime_fingerprint: str,
    publication_id: str,
    open_engine: Any = open_role_engine,
    bridge_factory: Any = build_capture_bridge,
    publish: Any = publish_shared_embedding_once,
) -> dict[str, str]:
    """Publish the run-level shared embedding table before any arm opens.

    An arm with no producer has no capture engine, so the driver opens one above
    the arms; every later producer adopts exactly these bytes or refuses.
    """
    engine = open_engine(PRODUCER_ROLE)
    bridge = None
    try:
        bridge = bridge_factory(engine)
        digests: dict[str, str] = publish(
            Path(root),
            bridge.model.get_input_embeddings().weight,
            runtime_fingerprint=runtime_fingerprint,
            publication_id=publication_id,
        )
    finally:
        if bridge is not None:
            bridge.release_view()
        engine.close()
    return digests


def _text_report_bank(context: WorkerContext, arm: str) -> TextReportBank:
    return TextReportBank(
        context.root / "arms" / arm / "text_reports.jsonl",
        attempt_id=context.attempt_id,
        execution_identity={
            "arm": arm,
            "panel": context.panel,
            "source_commit": context.source_commit,
        },
    )


def build_producer(
    context: WorkerContext,
    bank: BankCallback,
    items: dict[str, Any],
    *,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
    **seams: Any,
) -> FleetWorker:
    """Return the text or latent producer this arm calls for.

    ``seams`` are the producer's engine, bridge, capture, and publish callables.
    The fleet passes none of them; a caller can pass fakes to run offline.
    """
    arm = context.arm
    v16_arm = gemma_v16_arm(arm)
    root = handoff_root(context.root, arm)
    if v16_arm in TEXT_ARMS:
        return GemmaTextProducer(
            arm=arm,
            items=items,
            handoff_root=root,
            bank=bank,
            report_bank=_text_report_bank(context, arm),
            prepared_manifest=context.prepared_manifest,
            profile=profile,
            **seams,
        )
    return GemmaLatentProducer(
        arm=arm,
        v16_arm=v16_arm,
        items=items,
        run_root=context.root,
        handoff_root=root,
        bank=bank,
        runtime_fingerprint=str(context.identity.fields["runtime_fingerprint"]),
        publication_id=context.attempt_id,
        **seams,
    )


def receiver_kwargs(
    context: WorkerContext,
    bank: BankCallback,
    items: dict[str, Any],
    *,
    engine: Any,
    tokenizer: Any,
    model_load_s: float,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> dict[str, Any]:
    """Return every receiver argument this arm and run root imply.

    The table is the run-level publication, not an arm-local copy: an arm bundle
    only hardlinks it, so its handoff manifests stay relocatable.
    """
    v16_arm = gemma_v16_arm(context.arm)
    return {
        "arm": context.arm,
        "v16_arm": v16_arm,
        "items": items,
        "bank": bank,
        "tokenizer": tokenizer,
        "shared_weight": shared_embedding_path(context.root),
        "shared_weight_wait_s": SHARED_EMBEDDING_WAIT_S,
        "runtime_fingerprint": str(context.identity.fields["runtime_fingerprint"]),
        "publication_id": context.attempt_id,
        "engine": engine,
        "model_load_s": model_load_s,
        "seed_resolver": partial(shared_seeds_for_v16_arm, profile=profile),
        "flat_interleave_arms": (v16_arm,) if flat_interleave_arm(v16_arm) else (),
        "result_row_adapter": partial(project_shared_result_row, profile=profile),
    }


def _build_receiver(
    context: WorkerContext,
    bank: BankCallback,
    items: dict[str, Any],
    profile: BenchmarkProfile,
) -> Any:
    started = time.perf_counter()
    engine = open_role_engine(RECEIVER_ROLE)
    return receiver_kwargs(
        context,
        bank,
        items,
        engine=engine,
        tokenizer=load_gemma4_tokenizer(
            checkpoint_id=CHECKPOINT_ID,
            revision=CHECKPOINT_REVISION,
        ),
        model_load_s=time.perf_counter() - started,
        profile=profile,
    )


def build_split_worker(
    adapter: Any,
    context: WorkerContext,
    bank: BankCallback,
) -> FleetWorker:
    """Return the worker this arm and this seat's role call for."""
    items = {qid: item for qid, item in zip(context.qids, context.items, strict=True)}
    profile = replace(adapter.config.benchmark_profile, question_ids=context.qids)
    role = context.placement.role(context.worker_index)
    if role == "producer":
        return build_producer(context, bank, items, profile=profile)
    kwargs = _build_receiver(context, bank, items, profile)
    if role != "fused":
        return GemmaSplitReceiver(**kwargs)
    receiver: GemmaFusedReceiver

    def reports(item: dict[str, Any]) -> dict[str, Any]:
        from rcc.models.gemma.text_runtime import generate_resident_primary_reports

        return generate_resident_primary_reports(
            receiver.engine,
            receiver.tokenizer,
            [item],
            _text_report_bank(context, context.arm),
            prepared_manifest=context.prepared_manifest,
            profile=profile,
        )[str(item["qid"])]

    receiver = GemmaFusedReceiver(reports=reports, **kwargs)
    return receiver


__all__ = (
    "HANDOFF_DIRECTORY",
    "SHARED_EMBEDDING_WAIT_S",
    "build_producer",
    "build_split_worker",
    "handoff_root",
    "publish_run_shared_embedding",
    "receiver_kwargs",
)
