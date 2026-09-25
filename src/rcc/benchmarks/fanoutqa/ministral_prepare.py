"""Build the Ministral worker prompts from a FanOutQA panel's frozen text.

The worker texts are tokenized on the Tekken tokenizer; the per-item digest of
that tokenized evidence is what a panel load is checked against.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from rcc.benchmarks.fanoutqa.data import Question, load_questions
from rcc.benchmarks.fanoutqa.ministral_construction import (
    MinistralConstructionManifest,
    make_ministral_construction_manifest,
)
from rcc.benchmarks.fanoutqa.ministral_data import (
    MinistralPreparedPromptPanel,
    prepare_ministral_prompt_panel,
)
from rcc.benchmarks.fanoutqa.panel_pins import require_registered_identity
from rcc.benchmarks.fanoutqa.scoring import evidence_answerability_audit
from rcc.benchmarks.fanoutqa.source_bundle import validate_shared_source_bundle
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.ministral.text_codec import (
    MINISTRAL_TEXT_ARMS,
    MinistralTextCodec,
)
from rcc.run import identity


class _TekkenAdapter:
    """The tokenizer calls the shared source planner makes."""

    def __init__(self, codec: MinistralTextCodec) -> None:
        self._codec = codec
        self._inner: Any = codec.tokenizer.instruct_tokenizer.tokenizer

    def __call__(
        self,
        text: str | Sequence[str],
        *,
        add_special_tokens: bool = False,
    ) -> dict[str, object]:
        if add_special_tokens:
            raise ValueError("Ministral source tokenization never adds control tokens")

        def encode(value: str) -> list[int]:
            return [int(token) for token in self._inner.encode(value, bos=False, eos=False)]

        if isinstance(text, str):
            return {"input_ids": encode(text)}
        return {"input_ids": [encode(value) for value in text]}

    def decode(
        self,
        tokens: Sequence[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        if skip_special_tokens:
            raise ValueError("Ministral source planning retains exact token bytes")
        return str(self._codec.tokenizer.decode([int(token) for token in tokens]))


def _selected_questions(
    bundle_root: Path,
    qids: Sequence[str],
    *,
    profile: BenchmarkProfile,
) -> tuple[Question, ...]:
    selected = tuple(str(qid) for qid in qids)
    canonical = tuple(qid for qid in profile.question_ids if qid in set(selected))
    if selected != canonical:
        raise ValueError("Ministral preparation qids differ from sealed panel order")
    indexed = {
        question.qid: question
        for question in load_questions(bundle_root / "source_cache" / "fanout-final-dev.json")
    }
    if any(qid not in indexed for qid in selected):
        raise RuntimeError("sealed source bundle omits a selected Ministral question")
    return tuple(indexed[qid] for qid in selected)


def item_construction_sha256(qid: str, evidence: Sequence[Sequence[int]]) -> str:
    """Digest one item's reconstructed Tekken evidence, in worker order."""
    body = {"qid": str(qid), "evidence_ids": [[int(token) for token in row] for row in evidence]}
    return identity.fingerprint(body, identity.json_compact_legacy)


def _natural_evidence_by_qid(
    bundle_root: Path,
    questions: Sequence[Question],
    codec: MinistralTextCodec,
    *,
    profile: BenchmarkProfile,
) -> tuple[
    dict[str, tuple[tuple[int, ...], ...]],
    dict[str, tuple[tuple[str, ...], Mapping[str, object]]],
]:
    """Tokenize the natural worker texts on Tekken, with no filler."""
    from rcc.benchmarks.fanoutqa.natural_panel import natural_page_texts, natural_worker_texts

    tokenizer = _TekkenAdapter(codec)
    records = natural_worker_texts(bundle_root, profile=profile)
    page_texts = natural_page_texts(bundle_root, profile=profile)
    output: dict[str, tuple[tuple[int, ...], ...]] = {}
    construction: dict[str, tuple[tuple[str, ...], Mapping[str, object]]] = {}
    for question in questions:
        _record, texts = records[question.qid]
        rows = tuple(
            tuple(int(token) for token in cast(list[int], tokenizer(text)["input_ids"]))
            for text in texts
        )
        if any(not row for row in rows):
            raise RuntimeError(f"{question.qid}: natural worker text tokenized to nothing")
        audit = evidence_answerability_audit(
            question,
            full_sources=[page_texts[pageid] for pageid, _revid, _title in question.pages],
            retained_sources=texts,
        )
        construction[question.qid] = (texts, audit)
        output[question.qid] = rows
    return output, construction


def build_prepared_panel(
    bundle_root: Path,
    *,
    selected_qids: Sequence[str],
    codecs: Mapping[str, MinistralTextCodec],
    profile: BenchmarkProfile,
) -> MinistralPreparedPromptPanel:
    """Build all three sender lanes from the bundle's frozen source bytes."""
    validate_shared_source_bundle(bundle_root, profile=profile)
    if set(codecs) != set(MINISTRAL_TEXT_ARMS):
        raise ValueError("Ministral source preparation requires all three sender codecs")
    for codec in codecs.values():
        codec.require_verified()
    file_rosters = {codec.file_sha256 for codec in codecs.values()}
    if len(file_rosters) != 1:
        raise RuntimeError("Ministral sender tokenizers do not share the sealed Tekken bytes")
    questions = _selected_questions(bundle_root, selected_qids, profile=profile)
    primary = codecs["text_primary"]
    evidence, construction_inputs = _natural_evidence_by_qid(
        bundle_root, questions, primary, profile=profile
    )
    # The panel's own fingerprint shows it is internally consistent, not that it
    # is this panel, so the reconstructed items are compared with the pinned
    # identity here, at load, before any weight is placed.
    require_registered_identity(
        profile.profile_id,
        "ministral",
        {qid: item_construction_sha256(qid, rows) for qid, rows in evidence.items()},
        required=True,
    )
    construction: dict[str, dict[str, MinistralConstructionManifest]] = {}
    for arm in MINISTRAL_TEXT_ARMS:
        construction[arm] = {
            qid: make_ministral_construction_manifest(
                qid=qid,
                semantic_arm=arm,
                retained_sources=retained_sources,
                audit=audit,
                profile=profile,
            )
            for qid, (retained_sources, audit) in construction_inputs.items()
        }
    return prepare_ministral_prompt_panel(
        bundle_root,
        selected_qids=selected_qids,
        codecs=codecs,
        evidence_ids_by_arm={arm: evidence for arm in MINISTRAL_TEXT_ARMS},
        construction_by_arm=construction,
        profile=profile,
    )


__all__ = ("build_prepared_panel", "item_construction_sha256")
