"""Raw LongBench v2 rows, the prepared chain item, and item construction.

The prompt builders take a question and its choices rather than an item, so the
gold letter reaches no rendered byte; the scorer reads it after the answer lands.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from rcc.benchmarks.longbench_v2.chunking import SOURCE_UPDATES, ledger_by_qid
from rcc.benchmarks.longbench_v2.panel import RAW_ROWS, RAW_SOURCE_BYTES, RAW_SOURCE_SHA256
from rcc.benchmarks.longbench_v2.scoring import CHOICES
from rcc.run.io import sha256_file

RAW_KEYS = frozenset(
    {
        "_id",
        "domain",
        "sub_domain",
        "difficulty",
        "length",
        "question",
        "choice_A",
        "choice_B",
        "choice_C",
        "choice_D",
        "answer",
        "context",
    }
)
RawRow = dict[str, Any]


@dataclass(frozen=True)
class ChainItem:
    """One question, its four source chunks as token ids, and its gold letter."""

    qid: str
    question: str
    choices: tuple[str, str, str, str]
    chunks: tuple[tuple[int, ...], ...]
    chunk_sha256: tuple[str, ...]
    source_sha256: str
    source_tokens: int
    stratum: int
    domain: str
    difficulty: str
    gold: str
    task_kind: str = "mcq"

    def __post_init__(self) -> None:
        """Check the item at construction."""
        self.validate()

    def validate(self) -> None:
        """Raise for a malformed item; the loader calls this again after unpickling."""
        if type(self.qid) is not str or not self.qid:
            raise ValueError("a chain item needs a question id")
        if type(self.question) is not str or not self.question.strip():
            raise ValueError(f"{self.qid}: a chain item needs a question")
        choices = self.choices
        if (
            type(choices) is not tuple
            or len(choices) != 4
            or any(type(choice) is not str or not choice.strip() for choice in choices)
        ):
            raise ValueError(f"{self.qid}: a chain item needs four nonempty choices")
        chunks = self.chunks
        if type(chunks) is not tuple or len(chunks) != SOURCE_UPDATES:
            raise ValueError(f"{self.qid}: a chain item needs exactly four chunks")
        if type(self.chunk_sha256) is not tuple or len(self.chunk_sha256) != SOURCE_UPDATES:
            raise ValueError(f"{self.qid}: a chain item needs four chunk digests")
        if any(
            type(chunk) is not tuple
            or not chunk
            or any(type(token) is not int or token < 0 for token in chunk)
            for chunk in chunks
        ):
            raise ValueError(f"{self.qid}: every chunk needs nonnegative token ids")
        if self.gold not in CHOICES:
            raise ValueError(f"{self.qid}: gold answer must be one of {CHOICES}")
        if self.task_kind != "mcq":
            raise ValueError(f"{self.qid}: chain items are multiple choice")

    @property
    def chunk_tokens(self) -> tuple[int, ...]:
        """Return the standalone token count of every chunk."""
        return tuple(len(chunk) for chunk in self.chunks)


def load_raw_rows(path: Path) -> tuple[RawRow, ...]:
    """Load the raw dataset file, raising unless its size and digest are the pinned ones."""
    if path.stat().st_size != RAW_SOURCE_BYTES:
        raise RuntimeError(f"{path}: raw file size differs from the pinned dataset")
    if sha256_file(path) != RAW_SOURCE_SHA256:
        raise RuntimeError(f"{path}: raw file digest differs from the pinned dataset")
    with path.open(encoding="utf-8") as handle:
        value: object = json.load(handle)
    if not isinstance(value, list) or len(cast(list[object], value)) != RAW_ROWS:
        raise RuntimeError(f"{path}: raw file does not hold {RAW_ROWS} rows")
    rows: list[RawRow] = []
    for raw in cast(list[object], value):
        if not isinstance(raw, dict) or frozenset(cast(dict[str, object], raw)) != RAW_KEYS:
            raise RuntimeError(f"{path}: raw row has an unknown key set")
        rows.append(cast(RawRow, raw))
    return tuple(rows)


def rows_by_qid(rows: Sequence[Mapping[str, Any]]) -> dict[str, RawRow]:
    """Return the raw rows keyed by question id."""
    indexed = {str(row["_id"]): dict(row) for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("raw rows repeat a question id")
    return indexed


def choices_of(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Return the four choices of one raw row in letter order."""
    return (
        str(row["choice_A"]),
        str(row["choice_B"]),
        str(row["choice_C"]),
        str(row["choice_D"]),
    )


def _check_chunk_token_count(
    qid: str, chunk_row: Mapping[str, Any], ids: Sequence[int], expected_count_field: str | None
) -> None:
    expected = (
        len(ids)
        if expected_count_field is None
        else int(chunk_row.get(expected_count_field, len(ids)))
    )
    if len(ids) != expected:
        raise RuntimeError(f"{qid}: chunk {chunk_row['update']} token count drifted")


def build_item(
    row: Mapping[str, Any],
    chunk_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    stratum: int,
    expected_count_field: str | None = "standalone_qwen_source_tokens",
) -> ChainItem:
    """Cut one raw row at its ledger offsets and tokenize every chunk on its own."""
    context = str(row["context"])
    source_sha = hashlib.sha256(context.encode("utf-8")).hexdigest()
    chunks: list[tuple[int, ...]] = []
    shas: list[str] = []
    for chunk_row in chunk_rows:
        if str(chunk_row["source_sha256"]) != source_sha:
            raise RuntimeError(f"{row['_id']}: ledger names a different source")
        text = context[int(chunk_row["char_start"]) : int(chunk_row["char_end"])]
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if sha != str(chunk_row["chunk_sha256"]):
            raise RuntimeError(f"{row['_id']}: chunk {chunk_row['update']} bytes drifted")
        ids = tuple(int(token) for token in tokenizer.encode(text, add_special_tokens=False))
        _check_chunk_token_count(str(row["_id"]), chunk_row, ids, expected_count_field)
        chunks.append(ids)
        shas.append(sha)
    return ChainItem(
        qid=str(row["_id"]),
        question=str(row["question"]),
        choices=choices_of(row),
        chunks=tuple(chunks),
        chunk_sha256=tuple(shas),
        source_sha256=source_sha,
        source_tokens=int(chunk_rows[-1]["whole_token_end"]),
        stratum=stratum,
        domain=str(row["domain"]),
        difficulty=str(row["difficulty"]),
        gold=str(row["answer"]),
    )


def build_items(
    rows: Mapping[str, Mapping[str, Any]],
    ledger_rows: Sequence[Mapping[str, Any]],
    strata: Mapping[str, int],
    qids: Sequence[str],
    tokenizer: Any,
    *,
    expected_count_field: str | None = "standalone_qwen_source_tokens",
) -> tuple[ChainItem, ...]:
    """Build the prepared items in the panel's execution order."""
    ledger = ledger_by_qid(ledger_rows)
    items: list[ChainItem] = []
    for qid in qids:
        try:
            row, chunk_rows = rows[qid], ledger[qid]
        except KeyError as exc:
            raise RuntimeError(f"{qid}: raw rows or ledger omit a panel item") from exc
        items.append(
            build_item(
                row,
                chunk_rows,
                tokenizer,
                stratum=strata[qid],
                expected_count_field=expected_count_field,
            )
        )
    return tuple(items)


__all__ = (
    "RAW_KEYS",
    "ChainItem",
    "RawRow",
    "build_item",
    "build_items",
    "choices_of",
    "load_raw_rows",
    "rows_by_qid",
)
