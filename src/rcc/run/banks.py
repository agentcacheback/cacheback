"""Benchmark-neutral JSONL bank storage: append, read back, repair a torn tail.

Nothing here decides which attempt is reportable, and a row's identity is its
canonical sorted compact JSON encoding. The family shard logs build on it.
"""

from __future__ import annotations

import inspect
import json
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from rcc.run import io

JsonRow = dict[str, Any]
RowKey = str
RowKeyFn = Callable[[Mapping[str, Any]], RowKey]
RowPredicate = Callable[[Mapping[str, Any]], bool]
AttemptCallback = Callable[..., str]

DEFAULT_BANK_NAME = "raw.jsonl"
DEFAULT_RESULT_KIND = "result"
DEFAULT_POLICY_FIELD = "arm"


def canonical_row_key(row: Mapping[str, Any]) -> RowKey:
    """Return the canonical JSON identity of one row."""
    return json.dumps(dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_line(row: Mapping[str, Any]) -> bytes:
    encoded = json.dumps(dict(row), sort_keys=True, allow_nan=False)
    return encoded.encode("utf-8") + b"\n"


def _complete_payload(payload: bytes) -> tuple[bytes, int]:
    if not payload or payload.endswith(b"\n"):
        return payload, 0
    boundary = payload.rfind(b"\n") + 1
    return payload[:boundary], len(payload) - boundary


@dataclass(frozen=True)
class _Record:
    row: JsonRow | None
    raw: bytes


@dataclass(frozen=True)
class _ParsedBank:
    records: tuple[_Record, ...]
    complete_payload: bytes
    torn_tail_bytes: int


class BankFormatError(RuntimeError):
    """Raised when a complete bank line is not a JSON object."""


def read_json(path: Path, detail: str) -> Any:
    """Read one JSON document and report failures with the source path first."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: {detail}") from exc


def require(condition: bool, message: str) -> None:
    """Raise ValueError with the supplied detail unless the condition holds."""
    if not condition:
        raise ValueError(message)


def _decode_line(
    path: Path,
    line: bytes,
    line_number: int,
    *,
    reject_blank_lines: bool = False,
) -> JsonRow | None:
    if not line.strip():
        if reject_blank_lines:
            raise BankFormatError(f"{path}: malformed JSON at line {line_number}")
        return None
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BankFormatError(f"{path}: malformed UTF-8 at line {line_number}") from exc
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BankFormatError(f"{path}: malformed JSON at line {line_number}") from exc
    if not isinstance(value, dict):
        raise BankFormatError(f"{path}: line {line_number} is not a JSON object")
    return cast(JsonRow, value)


def _parse_bank(path: Path, *, reject_blank_lines: bool = False) -> _ParsedBank:
    payload = path.read_bytes()
    complete, torn_tail_bytes = _complete_payload(payload)
    records: list[_Record] = []
    for line_number, line in enumerate(complete.splitlines(keepends=True), start=1):
        records.append(
            _Record(
                _decode_line(
                    path,
                    line,
                    line_number,
                    reject_blank_lines=reject_blank_lines,
                ),
                line,
            )
        )
    return _ParsedBank(tuple(records), complete, torn_tail_bytes)


def read_bank_rows(
    path: Path,
    *,
    repair_torn_tail: bool = True,
    reject_torn_tail: bool = False,
    reject_blank_lines: bool = False,
) -> list[JsonRow]:
    """Read complete object rows, rejecting malformed complete lines.

    A trailing fragment with no newline is a torn row, discarded and repaired in
    place unless ``repair_torn_tail`` is false; a malformed complete line is not.
    """
    parsed = _parse_bank(path, reject_blank_lines=reject_blank_lines)
    if reject_torn_tail and parsed.torn_tail_bytes:
        line_number = parsed.complete_payload.count(b"\n") + 1
        raise BankFormatError(f"{path}: malformed JSON at line {line_number}")
    if repair_torn_tail and parsed.torn_tail_bytes:
        io.atomic_bytes(path, parsed.complete_payload)
    return [record.row for record in parsed.records if record.row is not None]


def repair_torn_tail(path: Path) -> int:
    """Atomically discard only the final unterminated byte fragment."""
    parsed = _parse_bank(path)
    if not parsed.torn_tail_bytes:
        return 0
    io.atomic_bytes(path, parsed.complete_payload)
    return parsed.torn_tail_bytes


def atomic_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Publish JSONL rows through the shared atomic writer."""
    io.atomic_bytes(path, b"".join(_json_line(row) for row in rows))


class BankWriter:
    """Append identity-stamped JSONL rows, with one fsync per row."""

    def __init__(
        self,
        path: Path,
        fixed_fields: Mapping[str, Any] | None = None,
        attempt_id: str | AttemptCallback | None = None,
        *,
        attempt_callback: AttemptCallback | None = None,
    ) -> None:
        """Bind one bank path, its fixed fields, and its attempt id source."""
        if attempt_id is not None and attempt_callback is not None:
            raise ValueError("provide one attempt id source")
        self.path = Path(path)
        self.fixed_fields = dict(fixed_fields or {})
        self.attempt_source = attempt_callback if attempt_callback is not None else attempt_id
        # A fleet producer seat appends from two threads (its loop's stage rows
        # and the deferred audit's capture rows), so every append is serialized.
        self._lock = threading.Lock()

    def _attempt_id(self, row: Mapping[str, Any], explicit: str | None) -> str:
        if explicit is not None:
            return explicit
        source = self.attempt_source
        if source is None:
            raise ValueError("an attempt id or callback is required")
        if callable(source):
            try:
                parameters = inspect.signature(source).parameters
            except (TypeError, ValueError):
                parameters = {"row": object()}
            value = source() if not parameters else source(row)
        else:
            value = source
        return str(value)

    def append(
        self,
        row: Mapping[str, Any],
        *,
        attempt_id: str | None = None,
    ) -> JsonRow:
        """Append one row under the bank's fixed identity and fsync it."""
        for name, expected in self.fixed_fields.items():
            if name in row and row[name] != expected:
                raise ValueError(f"row overrides fixed field {name}")
        durable = {
            **dict(row),
            **self.fixed_fields,
            "attempt_id": self._attempt_id(row, attempt_id),
            "banked_at_unix": time.time(),
        }
        encoded = json.dumps(durable, sort_keys=True, allow_nan=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return durable


@dataclass
class BankScanState:
    """One scan's per-file cursors and observed row-key counts."""

    offsets: dict[Path, int] = field(default_factory=lambda: {})
    counts: dict[RowKey, int] = field(default_factory=lambda: {})


class IncrementalBankScanner:
    """Scan only newly newline-terminated rows from a set of bank paths."""

    def __init__(
        self,
        paths: Iterable[Path],
        *,
        row_key: RowKeyFn = canonical_row_key,
        row_predicate: RowPredicate | None = None,
    ) -> None:
        """Set up cursors for the given bank paths and row predicate."""
        self.paths = tuple(Path(path) for path in paths)
        self.row_key = row_key
        self.row_predicate = row_predicate or _is_result_row
        self.state = BankScanState()

    def scan(self) -> tuple[JsonRow, ...]:
        """Return newly complete rows and leave any torn suffix unread."""
        new_rows: list[JsonRow] = []
        for path in self.paths:
            if not path.is_file():
                continue
            offset = self.state.offsets.get(path, 0)
            size = path.stat().st_size
            if size < offset:
                offset = 0
            with path.open("rb") as handle:
                handle.seek(offset)
                chunk = handle.read()
            complete = chunk.rfind(b"\n") + 1
            if not complete:
                continue
            for line_number, line in enumerate(
                chunk[:complete].splitlines(keepends=True),
                start=1,
            ):
                row = _decode_line(path, line, line_number)
                if row is None or not self.row_predicate(row):
                    continue
                new_rows.append(row)
                key = self.row_key(row)
                self.state.counts[key] = self.state.counts.get(key, 0) + 1
            self.state.offsets[path] = offset + complete
        return tuple(new_rows)

    def keys(self) -> set[RowKey]:
        """Return all row keys observed by this scanner so far."""
        return set(self.state.counts)

    def __call__(self) -> set[RowKey]:
        """Scan the new rows and return every key observed so far."""
        self.scan()
        return self.keys()


def _is_result_row(row: Mapping[str, Any]) -> bool:
    return row.get("kind") == DEFAULT_RESULT_KIND
