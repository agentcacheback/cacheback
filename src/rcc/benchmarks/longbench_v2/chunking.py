"""Cut one context into four contiguous raw substrings at whole-token quartiles.

The substrings concatenate back to the source byte for byte and are retokenized on
their own. See docs/benchmarks.md, Chunk boundaries.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

SOURCE_UPDATES = 4
SOURCE_CHUNK_CEILING = 64_000
LEDGER_FIELDS = (
    "balance_delta_tokens",
    "boundary_kind",
    "byte_end",
    "byte_start",
    "char_end",
    "char_start",
    "chunk_sha256",
    "cumulative_whole_source_tokens",
    "qid",
    "sample_order",
    "source_sha256",
    "source_tokens",
    "standalone_qwen_source_tokens",
    "target_whole_source_tokens",
    "terminal",
    "update",
    "whole_token_end",
    "whole_token_start",
)
ChunkRow = dict[str, Any]


def balanced_four_chunks(context: str, tokenizer: Any) -> tuple[list[str], list[ChunkRow]]:
    """Return the four raw substrings and one ledger row per chunk.

    The rows carry every field but the three the caller owns: ``qid``,
    ``sample_order``, and ``source_tokens``.
    """
    encoded = tokenizer(
        context,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    source_tokens = len(encoded["input_ids"])
    if source_tokens == 0 or len(offsets) != source_tokens:
        raise ValueError("source token offsets did not close")
    token_boundaries = [0]
    for update in range(1, SOURCE_UPDATES):
        token_boundaries.append(math.floor(update * source_tokens / SOURCE_UPDATES))
    token_boundaries.append(source_tokens)
    boundaries = [0]
    for token_boundary in token_boundaries[1:-1]:
        char_offset = offsets[token_boundary - 1][1]
        if not boundaries[-1] < char_offset < len(context):
            raise ValueError("fixed-four source boundary is not strictly increasing")
        boundaries.append(char_offset)
    boundaries.append(len(context))
    chunks = [context[start:end] for start, end in pairwise(boundaries)]
    if len(chunks) != SOURCE_UPDATES or any(not chunk for chunk in chunks):
        raise ValueError("fixed-four partition must have four nonempty chunks")
    if "".join(chunks) != context:
        raise ValueError("fixed-four chunks do not reconstruct the source")
    source_sha = hashlib.sha256(context.encode("utf-8")).hexdigest()
    rows: list[ChunkRow] = []
    for update, ((start, end), chunk) in enumerate(
        zip(pairwise(boundaries), chunks, strict=True), start=1
    ):
        standalone = len(tokenizer.encode(chunk, add_special_tokens=False))
        if standalone > SOURCE_CHUNK_CEILING:
            raise ValueError(f"fixed-four chunk exceeds {SOURCE_CHUNK_CEILING}: {standalone}")
        whole_start = token_boundaries[update - 1]
        whole_end = token_boundaries[update]
        rows.append(
            {
                "update": update,
                "char_start": start,
                "char_end": end,
                "byte_start": len(context[:start].encode("utf-8")),
                "byte_end": len(context[:end].encode("utf-8")),
                "whole_token_start": whole_start,
                "whole_token_end": whole_end,
                "cumulative_whole_source_tokens": whole_end,
                "standalone_qwen_source_tokens": standalone,
                "target_whole_source_tokens": math.floor(update * source_tokens / SOURCE_UPDATES),
                "balance_delta_tokens": (whole_end - whole_start) - source_tokens / SOURCE_UPDATES,
                "boundary_kind": "qwen_token" if update < SOURCE_UPDATES else "terminal",
                "terminal": update == SOURCE_UPDATES,
                "source_sha256": source_sha,
                "chunk_sha256": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
            }
        )
    return chunks, rows


def write_chunk_ledger(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write one canonical sorted JSON line per chunk row."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def read_chunk_ledger(path: Path) -> tuple[ChunkRow, ...]:
    """Read a ledger's rows in file order."""
    rows: list[ChunkRow] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value: object = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}: ledger row is not an object")
            rows.append(cast(ChunkRow, value))
    return tuple(rows)


def ledger_by_qid(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[ChunkRow, ...]]:
    """Group ledger rows per question in update order, raising on a gap."""
    grouped: dict[str, list[ChunkRow]] = {}
    for row in rows:
        grouped.setdefault(str(row["qid"]), []).append(dict(row))
    out: dict[str, tuple[ChunkRow, ...]] = {}
    for qid, chunk_rows in grouped.items():
        chunk_rows.sort(key=lambda row: int(row["update"]))
        if [int(row["update"]) for row in chunk_rows] != list(range(1, SOURCE_UPDATES + 1)):
            raise RuntimeError(f"{qid}: ledger does not hold exactly four ordered updates")
        out[qid] = tuple(chunk_rows)
    return out


def chunk_token_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[int, ...]]:
    """Return every question's four standalone chunk token counts."""
    return {
        qid: tuple(int(row["standalone_qwen_source_tokens"]) for row in chunk_rows)
        for qid, chunk_rows in ledger_by_qid(rows).items()
    }


def _validate_chunk_row(
    qid: str, row: Mapping[str, Any], previous_byte: int, previous_token: int
) -> tuple[int, int]:
    """Check one chunk row against its predecessor and return its end coordinates."""
    update = int(row["update"])
    if int(row["byte_start"]) != previous_byte or int(row["whole_token_start"]) != previous_token:
        raise RuntimeError(f"{qid}: chunk {update} is not contiguous")
    if int(row["byte_end"]) <= int(row["byte_start"]):
        raise RuntimeError(f"{qid}: chunk {update} is empty")
    if int(row["standalone_qwen_source_tokens"]) > SOURCE_CHUNK_CEILING:
        raise RuntimeError(f"{qid}: chunk {update} exceeds the ceiling")
    if bool(row["terminal"]) != (update == SOURCE_UPDATES):
        raise RuntimeError(f"{qid}: chunk {update} terminal flag is wrong")
    if int(row["cumulative_whole_source_tokens"]) != int(row["whole_token_end"]):
        raise RuntimeError(f"{qid}: chunk {update} cumulative count is wrong")
    source_tokens = int(row["source_tokens"])
    target = math.floor(update * source_tokens / SOURCE_UPDATES)
    if int(row["whole_token_end"]) != target or int(row["target_whole_source_tokens"]) != target:
        raise RuntimeError(f"{qid}: chunk {update} is not cut at the whole-token quartile")
    delta = (target - previous_token) - source_tokens / SOURCE_UPDATES
    if float(row["balance_delta_tokens"]) != delta:
        raise RuntimeError(f"{qid}: chunk {update} balance delta is wrong")
    return int(row["byte_end"]), int(row["whole_token_end"])


def validate_ledger(rows: Sequence[Mapping[str, Any]], qids: Sequence[str]) -> None:
    """Raise unless every question has four contiguous chunks covering its source."""
    if any(tuple(sorted(row)) != LEDGER_FIELDS for row in rows):
        raise RuntimeError("ledger rows differ from the sealed field layout")
    grouped = ledger_by_qid(rows)
    if tuple(grouped) != tuple(qids):
        raise RuntimeError("ledger question order differs from the panel")
    for qid, chunk_rows in grouped.items():
        if len({str(row["source_sha256"]) for row in chunk_rows}) != 1:
            raise RuntimeError(f"{qid}: chunks name more than one source")
        previous = (0, 0)
        for row in chunk_rows:
            previous = _validate_chunk_row(qid, row, *previous)
        if previous[1] != int(chunk_rows[0]["source_tokens"]):
            raise RuntimeError(f"{qid}: chunks do not cover the whole source")


__all__ = (
    "LEDGER_FIELDS",
    "SOURCE_CHUNK_CEILING",
    "SOURCE_UPDATES",
    "ChunkRow",
    "balanced_four_chunks",
    "chunk_token_counts",
    "ledger_by_qid",
    "read_chunk_ledger",
    "validate_ledger",
    "write_chunk_ledger",
)
