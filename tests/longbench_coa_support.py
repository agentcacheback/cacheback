"""Fakes shared by the LongBench chain tests; not a test module."""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import transformers

from conftest import TINY_VOCAB
from rcc.benchmarks.longbench_v2 import LONGBENCH_COA_EASY50_RERANK, LONGBENCH_COA_EASY50_TEXT
from rcc.benchmarks.longbench_v2.data import ChainItem
from rcc.benchmarks.longbench_v2.panel import EASY50_RERANK_KEY
from rcc.benchmarks.longbench_v2.prepare import seal_prepared_panel, seal_source_audit
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.hardware.placements import placement_table
from rcc.models.qwen import QWEN_FAMILY
from rcc.run.contract import BankCallback, BankIdentity, WorkerContext
from rcc.run.fleet.contract import ProducedArtifact, WorkItem
from rcc.run.qwen import worker as qwen_worker
from rcc.run.qwen.adapter import build_adapter
from rcc.run.qwen.chain_adapter import ChainAdapter

COMMIT = "a" * 40
CHOICES = ("alpha", "beta", "gamma", "delta")
#: The chain profile the shared fakes walk: the re-ranked handoff ladder over
#: the easy fifty. Tests that also need the floor and the text arms use
#: ``CHAIN_FIXTURE``, an unsealed roster under a fixture-only arm profile id.
PROFILE_KEY = EASY50_RERANK_KEY
CHAIN_PROFILE = LONGBENCH_COA_EASY50_RERANK
CHAIN_FIXTURE = dataclasses.replace(
    LONGBENCH_COA_EASY50_RERANK,
    arms=(*LONGBENCH_COA_EASY50_TEXT.arms, *LONGBENCH_COA_EASY50_RERANK.arms),
    sealed_qwen_policies=(
        *LONGBENCH_COA_EASY50_TEXT.sealed_qwen_policies,
        *LONGBENCH_COA_EASY50_RERANK.sealed_qwen_policies,
    ),
    arm_profiles=("test-chain-fixture-10arm",),
)
#: What one scripted sample emits: a thinking block, then the official phrasing
#: the LongBench extraction reads its letter out of.
CHAIN_ANSWER = "<think>x</think>The correct answer is ({letter})"


class FakeTokenizer:
    """Fixed-width character tokenizer with offsets and a chat template."""

    def __init__(self, width: int = 1) -> None:
        self.width = width

    def _spans(self, text: str) -> list[tuple[int, int]]:
        return [(i, min(i + self.width, len(text))) for i in range(0, len(text), self.width)]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [sum(map(ord, text[s:e])) % 997 + 1 for s, e in self._spans(text)]

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        out: dict[str, Any] = {"input_ids": self.encode(text)}
        if kwargs.get("return_offsets_mapping"):
            out["offset_mapping"] = self._spans(text)
        return out

    def decode(self, tokens: Any, **kwargs: Any) -> str:
        """Return a readable stand-in for the ids.

        The encoder folds characters into one id, so no exact inverse exists;
        the chain only needs a nonempty, deterministic body per chunk.
        """
        del kwargs
        return " ".join(f"t{int(token)}" for token in tokens)

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Render the Qwen shape, including the thinking switch the real one honours.

        The real template closes an empty thinking block when the switch is
        off, so the fake does the same rather than render both alike.
        """
        head = f"<|im_start|>user\n{messages[-1]['content']}<|im_end|>\n<|im_start|>assistant\n"
        if kwargs.get("enable_thinking", True):
            return head
        return f"{head}<think>\n\n</think>\n\n"


class TinyChainTokenizer(FakeTokenizer):
    """The shared chain fake folded into the tiny model's 512-id vocabulary.

    The shared fake creates ids up to 997, past the tiny embedding table;
    folding keeps every offset and rendering identical, narrowing the alphabet.
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Return the shared fake's ids, folded into the tiny vocabulary."""
        return [token % TINY_VOCAB for token in super().encode(text, add_special_tokens)]


#: Five scripted draws over four hops: hop two's first draw ships an empty note.
REDRAW_SCRIPT = (
    "<think>x</think>note 1",
    "<think>x</think>",
    "<think>x</think>note 2",
    "<think>x</think>note 3",
    "<think>x</think>note 4",
)


def chain_item(qid: str, seed: int, stratum: int = 3) -> ChainItem:
    """Build one small valid item whose chunks are consecutive ids.

    The four chunks have four different lengths, so a producer that read the
    wrong part of the source cannot still render the right geometry.
    """
    chunks = tuple(tuple(range(seed + hop, seed + hop + 4 + hop)) for hop in range(4))
    return ChainItem(
        qid=qid,
        question=f"question {qid}",
        choices=CHOICES,
        chunks=chunks,
        chunk_sha256=tuple(hashlib.sha256(str(chunk).encode()).hexdigest() for chunk in chunks),
        source_sha256=hashlib.sha256(qid.encode()).hexdigest(),
        source_tokens=sum(len(chunk) for chunk in chunks),
        stratum=stratum,
        domain="Single-Document QA",
        difficulty="hard",
        gold="B",
    )


def raw_row(qid: str, context: str, answer: str = "C") -> dict[str, Any]:
    """Build one raw-shaped row."""
    return {
        "_id": qid,
        "domain": "d",
        "sub_domain": "s",
        "difficulty": "easy",
        "length": "long",
        "question": f"Q {qid}?",
        "choice_A": "a",
        "choice_B": "b",
        "choice_C": "c",
        "choice_D": "d",
        "answer": answer,
        "context": context,
    }


class AnswerTokenizer(TinyChainTokenizer):
    """The shared chain fake with a character decode, so a banked answer reads back.

    The receiver reconstructs every answer from raw ids, so the fake inverts
    its own alphabet; every id a chain test decodes is a printable code point.
    """

    def decode(self, tokens: Any, **kwargs: Any) -> str:
        """Return the ids read as code points."""
        del kwargs
        return "".join(chr(int(token)) for token in tokens)


class ChainTracker:
    """Finish every offered request on the next pump, one scripted letter per item.

    The letter is keyed by item so each one scores on its own; a tracker that
    answers every item alike can only publish a macro of one value.
    """

    def __init__(self, letters: Mapping[str, str]) -> None:
        """Open an idle tracker that answers each item with its scripted letter."""
        self.letters = dict(letters)
        self.decoder = SimpleNamespace(abort_all=lambda: None)
        self.admission_trace: dict[str, dict[str, int]] = {}
        self.clock = 0.0
        self._ready: list[tuple[str, float, tuple[Any, ...]]] = []

    def offer(self, qid: str, requests: Sequence[Any]) -> None:
        """Complete the offered batch, one second apart, half a second each."""
        raw = CHAIN_ANSWER.format(letter=self.letters[qid])
        completions: list[Any] = []
        for request in requests:
            self.clock += 1.0
            ids = (*(ord(char) for char in raw), QWEN_FAMILY.stop_token_ids[0])
            completions.append(
                SimpleNamespace(
                    request_id=request.request_id,
                    tag=request.tag,
                    text=raw,
                    token_ids=ids,
                    n_tokens=len(ids),
                    finish_reason="stop",
                    num_cached_tokens=0,
                    answer_injected=False,
                    submitted_at=self.clock,
                    first_token_at=self.clock + 0.1,
                    finished_at=self.clock + 0.5,
                    continuation_submitted_at=None,
                    continuation_first_token_at=None,
                )
            )
        self._ready.append((qid, self.clock, tuple(completions)))

    def pump(self) -> list[tuple[str, float, tuple[Any, ...]]]:
        """Surface every batch offered since the last pump."""
        ready, self._ready = self._ready, []
        return ready

    def idle(self) -> bool:
        """Return whether nothing is waiting to be surfaced."""
        return not self._ready


def bank_rows(rows: list[dict[str, Any]]) -> BankCallback:
    """Return a bank callback that keeps every row a seat hands it."""

    def record(row: Mapping[str, Any]) -> dict[str, Any]:
        rows.append(dict(row))
        return rows[-1]

    return record


def seal_chain_panel(root: Path, *, items: Sequence[ChainItem], profile: BenchmarkProfile) -> None:
    """Seal one roster of chain items as a run root's prepared production panel."""
    audit = seal_source_audit(root, source_commit=COMMIT, items=items, profile=profile)
    seal_prepared_panel(
        root, source_commit=COMMIT, items=items, source_audit=audit, profile=profile
    )


def chain_adapter(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    profile: BenchmarkProfile,
    count: int | None = None,
) -> tuple[ChainAdapter, Mapping[str, Any]]:
    """Build the chain adapter over a sealed panel and load one item range of it.

    The range is the whole sealed roster unless a caller names a shorter one,
    the way the driver names the range every seat then reads.
    """
    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *_a, **_k: AnswerTokenizer(width=4)),
    )
    monkeypatch.setenv("RCC_FANOUT_ITEM_START", "0")
    monkeypatch.setenv(
        "RCC_FANOUT_ITEM_COUNT", str(len(profile.question_ids) if count is None else count)
    )
    adapter = build_adapter(QWEN_FAMILY, profile=profile)
    assert isinstance(adapter, ChainAdapter)
    _items, manifest = adapter.data.load_prepared(root, panel="production", source_commit=COMMIT)
    return adapter, manifest


def chain_context(
    adapter: ChainAdapter, root: Path, *, arm: str, worker_index: int
) -> WorkerContext:
    """Bind one seat of one arm to the loaded panel."""
    qids = adapter.data.qids
    placement = placement_table(adapter.profile.topology_key)[arm]
    return WorkerContext(
        root=root,
        panel="production",
        arm=QWEN_FAMILY.policy(arm),
        worker_index=worker_index,
        attempt_id="attempt-test",
        source_commit=COMMIT,
        qids=qids,
        items=tuple(adapter.data.item(qid) for qid in qids),
        prepared_manifest={"artifact_sha256": "0" * 64},
        placement=placement,
        identity=BankIdentity(fields={}),
    )


def warm_chain_receiver(
    monkeypatch: pytest.MonkeyPatch, tiny_model: Any, tracker: ChainTracker
) -> None:
    """Stand in the resident receiver engine, its handle, and its stream decoder."""
    backend = SimpleNamespace(
        warmup_decode=lambda _prompt, *, seed: None,
        resident_model=lambda: tiny_model,
        kv_pool_tokens=lambda: 337_808,
        close=lambda: None,
    )
    monkeypatch.setattr(qwen_worker, "build_qwen_backend", lambda *_a, **_k: backend)
    monkeypatch.setattr(
        qwen_worker,
        "VllmEngineHandle",
        lambda *_a, **_k: SimpleNamespace(sampling=lambda params, seed: (params, seed)),
    )
    monkeypatch.setattr(qwen_worker, "StreamDecoder", lambda *_a, **_k: SimpleNamespace())
    monkeypatch.setattr(qwen_worker, "ItemStreamTracker", lambda *_a, **_k: tracker)


def run_chain_receiver(
    adapter: ChainAdapter,
    root: Path,
    *,
    arm: str,
    qid: str,
    artifact: ProducedArtifact | None,
) -> dict[str, Any]:
    """Drive one receiver seat through both offered batches and bank its row."""
    banked: list[dict[str, Any]] = []
    context = chain_context(adapter, root, arm=arm, worker_index=7)
    receiver = cast(Any, adapter.build_worker(context, bank_rows(banked)))
    receiver.warm()
    receiver.submit(WorkItem(qid, 0), artifact)
    assert receiver.pump() == []
    completions = receiver.pump()
    assert len(completions) == 1
    receiver.close()
    return cast(dict[str, Any], completions[0].result)
