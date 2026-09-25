"""A natural profile's frozen worker texts, tokenized and framed for Gemma.

Called at panel load; nothing here re-cuts a page or pads a prompt to a width.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from rcc.benchmarks.fanoutqa.gemma_data import (
    WorkerPanel,
    _prepared_digest,  # pyright: ignore[reportPrivateUsage]
    build_roll_frame,
    coordinator_ids,
    require_registered_construction,
    worker_qids,
)
from rcc.benchmarks.fanoutqa.natural_panel import (
    natural_page_texts,
    natural_questions,
    natural_worker_texts,
    validate_natural_bundle,
)
from rcc.benchmarks.fanoutqa.panel import Question
from rcc.benchmarks.fanoutqa.scoring import evidence_answerability_audit
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.gemma.mechanism import encode
from rcc.models.gemma.source_geometry import ItemSource, WorkerSource


def build_natural_item(
    tokenizer: Any,
    question: Question,
    texts: tuple[str, ...],
    page_texts: dict[int, str],
    pageids_by_worker: tuple[tuple[int, ...], ...],
) -> dict[str, Any]:
    """Frame one item's worker texts on the Gemma tokenizer."""
    shards = tuple(tuple(int(token) for token in encode(tokenizer, text)) for text in texts)
    if any(not shard for shard in shards):
        raise RuntimeError(f"{question.qid}: a natural worker text tokenized to nothing")
    workers = tuple(
        WorkerSource(worker=index, memory_ids=shard, pageids=pageids)
        for index, (shard, pageids) in enumerate(zip(shards, pageids_by_worker, strict=True))
    )
    source = ItemSource(
        qid=question.qid,
        workers=workers,
        dead_pages=(),
        ledger=(),
        structure={"reasoning_shape": "natural_sealed_text"},
    )
    full_sources = [page_texts[pageid] for pageid, _revid, _title in question.pages]
    construction = evidence_answerability_audit(
        question, full_sources=full_sources, retained_sources=list(texts)
    )
    frame = build_roll_frame(tokenizer, question.question)
    memory_ids = [torch.tensor(list(shard), dtype=torch.long) for shard in shards]
    prompt_ids = [frame.prompt(ids) for ids in memory_ids]
    context_rows = [
        {
            "worker": index,
            "original_evidence_tokens": len(shard),
            "original_prompt_tokens": int(prompt.shape[0]),
            "final_evidence_payload_tokens": len(shard),
            "final_prompt_tokens": int(prompt.shape[0]),
            "evidence_start": len(frame.prefix_ids),
            "evidence_end": len(frame.prefix_ids) + len(shard),
            "padding_payload_tokens": 0,
            "padding_correction_tokens": 0,
            "padding_source": "none",
        }
        for index, (shard, prompt) in enumerate(zip(shards, prompt_ids, strict=True))
    ]
    judger_ids, request_ids, question_ids = coordinator_ids(tokenizer, question.question)
    removed = int(construction["n_removed_by_construction"]) > 0
    return {
        "qid": question.qid,
        "question": question,
        "source": source,
        "memory_ids": memory_ids,
        "prompt_ids": prompt_ids,
        "frame": frame,
        "context_fill": context_rows,
        "question_ids": torch.tensor(question_ids, dtype=torch.long),
        "judger_ids": torch.tensor([judger_ids], dtype=torch.long),
        "request_ids": request_ids,
        "page_texts": full_sources,
        "construction": construction,
        "dead_pages": [],
        "construction_complete": not removed,
        "construction_incomplete_reason": "leaves_removed_by_packing" if removed else "",
    }


def load_natural_worker_panel(
    bundle_root: Path,
    tokenizer: Any,
    *,
    offset: int,
    count: int,
    node_gpus: int,
    worker_index: int,
    profile: BenchmarkProfile,
) -> WorkerPanel:
    """Construct the items one GPU owns from the validated natural bundle."""
    root = Path(bundle_root)
    manifest = validate_natural_bundle(root, profile=profile)
    qids = worker_qids(
        offset=offset,
        count=count,
        node_gpus=node_gpus,
        worker_index=worker_index,
        item_split="heldout",
        profile=profile,
    )
    questions = natural_questions(root, profile=profile)
    records = natural_worker_texts(root, profile=profile)
    page_texts = natural_page_texts(root, profile=profile)
    items: list[dict[str, Any]] = []
    for qid in qids:
        record, texts = records[qid]
        pageids_by_worker = tuple(
            tuple(int(page["pageid"]) for page in worker["pages"]) for worker in record["workers"]
        )
        item = build_natural_item(tokenizer, questions[qid], texts, page_texts, pageids_by_worker)
        item["global_item_index"] = profile.question_ids.index(qid)
        item["panel"] = "production"
        items.append(item)
    require_registered_construction(profile.profile_id, items, required=True)
    return WorkerPanel(
        qids=qids,
        items=tuple(items),
        source_manifest=dict(manifest),
        prepared_sha256=_prepared_digest(items, str(manifest["fingerprint"])),
    )


__all__ = ("build_natural_item", "load_natural_worker_panel")
