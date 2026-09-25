"""Build the one Ministral worker a split-fleet seat calls for.

The placement decides the role and the arm decides the channel, and those two
together decide which of the four worker shapes this process is.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.ministral.engine_roles import (
    RECEIVER_ROLE,
    SPLIT_RECEIVER_ENGINE_IDENTITY,
    open_role_engine,
)
from rcc.models.ministral.receiver_core import MinistralReceiverCore
from rcc.models.ministral.text import MinistralReportBundle, generate_report_bundle
from rcc.models.ministral.text_codec import MINISTRAL_TEXT_ARMS, MinistralTextCodec
from rcc.run.contract import BankCallback, FleetWorker, WorkerContext
from rcc.run.ministral.config import prompt_records
from rcc.run.ministral.split_producer import (
    MinistralLatentProducer,
    MinistralTextProducer,
)
from rcc.run.ministral.split_receiver import (
    FUSED_ARM,
    MinistralFusedReceiver,
    MinistralSplitReceiver,
)

#: Where one arm's producers publish, and its receivers read, the bytes that
#: cross the process boundary: the flat payload, the report bundle, and the
#: manifest and clock record that name them.
HANDOFF_DIRECTORY = "handoff"

#: Every manager turn and every answer is decoded by the receiver model, so the
#: receiver-side codec is this one arm's codec whichever arm is running.
RECEIVER_CODEC_ARM = FUSED_ARM


def handoff_root(root: Path, arm: str) -> Path:
    """Return the cross-process bundle root for one arm."""
    return root / "arms" / arm / HANDOFF_DIRECTORY


def _build_producer(
    context: WorkerContext,
    bank: BankCallback,
    items: dict[str, Any],
    codecs: Mapping[str, MinistralTextCodec],
    profile: BenchmarkProfile,
) -> FleetWorker:
    arm = context.arm
    root = handoff_root(context.root, arm)
    if arm in MINISTRAL_TEXT_ARMS:
        return MinistralTextProducer(
            arm=arm,
            items=items,
            handoff_root=root,
            bank=bank,
            codec=codecs[arm],
            profile=profile,
        )
    return MinistralLatentProducer(
        arm=arm,
        items=items,
        handoff_root=root,
        bank=bank,
        worker_index=context.worker_index,
        codec=codecs[RECEIVER_CODEC_ARM],
        profile=profile,
    )


def _build_core(
    context: WorkerContext,
    codecs: Mapping[str, MinistralTextCodec],
    profile: BenchmarkProfile,
) -> MinistralReceiverCore:
    started = time.perf_counter()
    engine = open_role_engine(RECEIVER_ROLE)
    model_load_s = time.perf_counter() - started
    return MinistralReceiverCore(
        engine=engine,
        receiver_codec=codecs[RECEIVER_CODEC_ARM],
        embedding_weight=engine.embedding_weight,
        execution_identity={
            "unified_plan_fingerprint": str(context.identity.fields["unified_plan_fingerprint"]),
            **engine.execution_identity,
        },
        model_load_s=model_load_s,
        engine_contract=SPLIT_RECEIVER_ENGINE_IDENTITY,
        profile=profile,
    )


def build_split_worker(
    adapter: Any,
    context: WorkerContext,
    bank: BankCallback,
) -> FleetWorker:
    """Return the worker this arm and this seat's role call for."""
    items = {qid: item for qid, item in zip(context.qids, context.items, strict=True)}
    codecs: Mapping[str, MinistralTextCodec] = adapter.codecs
    profile = replace(adapter.config.benchmark_profile, question_ids=context.qids)
    role = context.placement.role(context.worker_index)
    if role == "producer":
        return _build_producer(context, bank, items, codecs, profile)
    core = _build_core(context, codecs, profile)
    kwargs: dict[str, Any] = {
        "arm": context.arm,
        "items": items,
        "bank": bank,
        "core": core,
        "report_codecs": {arm: codecs[arm] for arm in MINISTRAL_TEXT_ARMS},
    }
    if role != "fused":
        if context.arm == FUSED_ARM:
            raise RuntimeError("text_primary is the fused Ministral arm and has no split receiver")
        return MinistralSplitReceiver(**kwargs)

    def reports(qid: str) -> MinistralReportBundle:
        engine = core.engine
        if engine is None:
            raise RuntimeError("Ministral fused receiver engine is unavailable")
        return generate_report_bundle(
            engine,
            codecs[FUSED_ARM],
            prompt_records(items[qid]["artifacts"][FUSED_ARM]),
            profile=profile,
        )

    return MinistralFusedReceiver(reports=reports, **kwargs)


__all__ = (
    "HANDOFF_DIRECTORY",
    "RECEIVER_CODEC_ARM",
    "build_split_worker",
    "handoff_root",
)
