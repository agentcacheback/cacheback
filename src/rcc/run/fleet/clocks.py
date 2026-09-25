"""The producer-clock record every family and channel writes.

A producer cannot price its own publish inside the bytes it publishes, so every
channel writes a ``.clocks.json`` sidecar bound to the manifest or file digest.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from rcc.run.io import atomic_bytes, canonical_bytes, sha256_file


@dataclass(frozen=True)
class ClockRecordSpec:
    """One channel's clock-record identity: its schema, clocks, and refusals."""

    schema: str
    noun: str
    error: type[RuntimeError]
    clocks: tuple[str, ...]
    negative_refusal: str
    identity_refusal: str
    #: When set, the record also carries the digest of the file it sits beside,
    #: so a payload rewritten after publication carries no record for its new
    #: content. A manifest-bound channel passes ``identity`` and leaves this None.
    digest_field: str | None = None

    def __post_init__(self) -> None:
        """Refuse a spec that prices nothing or names no schema."""
        if not self.schema or not self.clocks:
            raise ValueError("a clock-record spec needs a schema and at least one clock")


def clock_record_path(path: Path) -> Path:
    """Return where the clock record belonging to one published file sits."""
    return path.with_name(f"{path.name}.clocks.json")


def write_clock_record(
    spec: ClockRecordSpec,
    path: Path,
    *,
    label: str,
    identity: Mapping[str, str],
    clocks: Mapping[str, float],
) -> Path:
    """Publish one producer's measured clocks beside the bytes it wrote."""
    if set(clocks) != set(spec.clocks):
        raise ValueError(f"{label}: {spec.noun} clock roster differs")
    if any(clocks[name] < 0 for name in spec.clocks):
        raise spec.error(f"{label}: {spec.negative_refusal}")
    body: dict[str, Any] = {"schema": spec.schema, **identity}
    if spec.digest_field is not None:
        body[spec.digest_field] = sha256_file(path)
    for name in spec.clocks:
        body[name] = round(float(clocks[name]), 4)
    target = clock_record_path(path)
    atomic_bytes(target, canonical_bytes(body) + b"\n")
    return target


def read_clock_record(
    spec: ClockRecordSpec,
    path: Path,
    *,
    label: str,
    identity: Mapping[str, str],
) -> dict[str, float]:
    """Return one record's clocks, or refuse a record written for other bytes."""
    target = clock_record_path(path)
    try:
        decoded: object = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise spec.error(f"{label}: {spec.noun} is unreadable") from error
    if not isinstance(decoded, dict):
        raise spec.error(f"{label}: {spec.noun} is malformed")
    body = cast(dict[str, Any], decoded)
    expected = {"schema": spec.schema, **identity}
    if spec.digest_field is not None:
        expected[spec.digest_field] = sha256_file(path)
    if any(body.get(name) != value for name, value in expected.items()):
        raise spec.error(f"{label}: {spec.identity_refusal}")
    clocks: dict[str, float] = {}
    for name in spec.clocks:
        value = body.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise spec.error(f"{label}: {spec.noun} entry {name} is malformed")
        clocks[name] = float(value)
    return clocks


@dataclass(frozen=True)
class HandoffClocks:
    """The producer-side clocks a receiver cannot measure for itself."""

    spill_save_s: float
    producer_s: float


def handoff_clock_spec(*, schema: str, error: type[RuntimeError]) -> ClockRecordSpec:
    """Return one family's binding of the manifest-bound latent handoff clocks."""
    return ClockRecordSpec(
        schema=schema,
        noun="handoff clock record",
        error=error,
        clocks=("spill_save_s", "producer_s"),
        negative_refusal="handoff producer clocks must be nonnegative",
        identity_refusal="handoff clock record belongs to another manifest",
    )


def bundle_clock_spec(*, schema: str, error: type[RuntimeError]) -> ClockRecordSpec:
    """Return one family's binding of the digest-bound text bundle clock.

    A text bundle has no manifest, so its clock binds to the bundle's own digest
    and a bundle rewritten after publication carries no record for its content.
    """
    return ClockRecordSpec(
        schema=schema,
        noun="bundle clock record",
        error=error,
        clocks=("spill_save_s",),
        negative_refusal="bundle save clock must be nonnegative",
        identity_refusal="bundle clock record names other bytes",
        digest_field="bundle_sha256",
    )


def write_bundle_clocks(
    spec: ClockRecordSpec,
    path: Path,
    *,
    qid: str,
    semantic_arm: str,
    spill_save_s: float,
) -> Path:
    """Publish the text producer's measured save cost beside the bundle."""
    return write_clock_record(
        spec,
        path,
        label=f"{qid}/{semantic_arm}",
        identity={"qid": qid, "semantic_arm": semantic_arm},
        clocks={"spill_save_s": spill_save_s},
    )


def read_bundle_clocks(
    spec: ClockRecordSpec,
    path: Path,
    *,
    qid: str,
    semantic_arm: str,
) -> float:
    """Return the producer save cost, or refuse a clock record for other bytes."""
    clocks = read_clock_record(
        spec,
        path,
        label=f"{qid}/{semantic_arm}",
        identity={"qid": qid, "semantic_arm": semantic_arm},
    )
    return clocks["spill_save_s"]


__all__ = (
    "ClockRecordSpec",
    "HandoffClocks",
    "bundle_clock_spec",
    "clock_record_path",
    "handoff_clock_spec",
    "read_bundle_clocks",
    "read_clock_record",
    "write_bundle_clocks",
    "write_clock_record",
)
