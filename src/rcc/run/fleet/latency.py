"""Family-neutral per-item latency fields and the fleet-scope stamp rewrite.

Every family banks the same compute-scope clocks under the same names, and the
merge republishes the ledger's fleet-scope numbers: what the caller waited for.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from rcc.run.fleet.ledger import strict_stage_decomposition

COMPUTE_TIMING_SCOPE = (
    "end to end compute for the single timed sample: producing the handoff, "
    "preparing the receiver request, and decoding. Engines are already resident, "
    "so model load is excluded and banked on phase rows; the harness spill round "
    "trip is excluded and banked as spill_save_s and spill_load_s, since a real "
    "sender and receiver use the wire and not local disk; wire transit itself is "
    "priced in bytes, not in seconds"
)

FLEET_TIMING_SCOPE = (
    "fleet batch arrival through this item's s0 first token when emitted / s0 "
    "completion; producer, queue, handoff, admission, and receiver contention "
    "included; the final scored-sample completion is published beside it as "
    "fleet_protocol_wall_s"
)

FLEET_DECODE_SCHEDULE = (
    "s0 admitted first; s1/s2 offered together after s0 completion; "
    "requests from other items may overlap under token admission"
)

#: The per-item latency vocabulary every family banks. A row kind that carries
#: one of these carries all of them, so a cross-family read never compares a
#: present field against an absent one.
ITEM_LATENCY_FIELDS = (
    "decode_timing_sample",
    "timing_scope",
    "fleet_decode_schedule",
    "producer_s",
    "receiver_prepare_s",
    "spill_load_s",
    "spill_save_s",
    "decode_s",
    "decode_batched_s",
    "queue_s",
    "receiver_ttft_s",
    "generation_s",
    "ttft_s",
    "tteoa_s",
    "decode_queued_offset_s_by_sample",
    "decode_first_token_offset_s_by_sample",
    "decode_finished_offset_s_by_sample",
    "answer_continuation_submit_offset_s_by_sample",
    "answer_continuation_first_token_offset_s_by_sample",
)

#: The stage spans the ledger decomposes; every one is republished under a
#: ``fleet_`` prefix so a fleet number can never be read as a compute number.
FLEET_STAGE_FIELDS = (
    "producer_queue_s",
    "producer_s",
    "handoff_s",
    "receiver_queue_s",
    "receiver_prepare_s",
    "gate_wait_s",
    "decode_s",
    "fleet_ttft_s",
    "tteoa_s",
    "protocol_wall_s",
)

#: Result-row keys for the same two decode clocks under an older bank format.
#: A row carrying either of them, or carrying no ``timing_scope``, is not on
#: the latency schema and is refused.
LEGACY_DECODE_FIELDS = ("timed_decode_s", "batched_decode_s")

WARMUP_BODY = "Reply with OK."
WARMUP_MAX_TOKENS = 2


def warmup_seed(*parts: object) -> int:
    """Derive one warmup seed that sits outside every panel seed grid.

    The warmup completion is discarded, so it carries no panel identity. The
    derivation is stable, so warming twice warms the same way twice.
    """
    joined = "|".join(str(part) for part in ("rcc-fleet-warmup", *parts))
    return zlib.crc32(joined.encode()) & 0x7FFFFFFF


@dataclass(frozen=True)
class ItemLatency:
    """One item's compute-scope clocks, banked the same way by every family.

    The offsets are per-sample from one shared timing origin, so a reader can
    rebuild the decode schedule. Only s0 is a service time.
    """

    producer_s: float
    receiver_prepare_s: float
    decode_s: float
    decode_batched_s: float
    receiver_ttft_s: float
    generation_s: float
    queued_offsets: tuple[float, ...]
    first_token_offsets: tuple[float, ...]
    finished_offsets: tuple[float, ...]
    spill_load_s: float = 0.0
    spill_save_s: float = 0.0
    #: The answer-side closer's one continuation, per sample, on the same origin:
    #: submitted and first token, or None when the sample was not continued. The
    #: span is the continuation's re-prefill of prompt plus head, prefix cache off.
    continuation_submit_offsets: tuple[float | None, ...] | None = None
    continuation_first_token_offsets: tuple[float | None, ...] | None = None

    def __post_init__(self) -> None:
        """Refuse negative clocks, ragged offsets, or an overlapped s0."""
        _check_continuation_offsets(
            self.continuation_submit_offsets,
            self.continuation_first_token_offsets,
            width=len(self.queued_offsets),
        )
        if any(
            value < 0
            for value in (
                self.producer_s,
                self.receiver_prepare_s,
                self.decode_s,
                self.decode_batched_s,
                self.receiver_ttft_s,
                self.generation_s,
                self.spill_load_s,
                self.spill_save_s,
            )
        ):
            raise ValueError("item latency clocks must be nonnegative")
        widths = {
            len(self.queued_offsets),
            len(self.first_token_offsets),
            len(self.finished_offsets),
        }
        if len(widths) != 1 or widths == {0}:
            raise ValueError("item latency sample offsets must share one nonempty width")
        if self.queued_offsets[0] != 0.0:
            raise ValueError("item latency offsets must start at the timed sample")
        if any(queued < self.finished_offsets[0] for queued in self.queued_offsets[1:]):
            raise ValueError("item latency shows a batched sample starting before s0 finished")

    def result_fields(self) -> dict[str, Any]:
        """Return the whole latency vocabulary for one result row."""
        return {
            "decode_timing_sample": "s0",
            "timing_scope": COMPUTE_TIMING_SCOPE,
            "fleet_decode_schedule": FLEET_DECODE_SCHEDULE,
            "producer_s": round(self.producer_s, 4),
            # Tiny requests and high-ratio arms can prepare in under 0.1 ms;
            # microsecond resolution keeps real work from banking as zero.
            "receiver_prepare_s": round(self.receiver_prepare_s, 6),
            # The local disk round trip, both halves. It sits outside every
            # caller clock below, and is banked so it can be added back.
            "spill_load_s": round(self.spill_load_s, 4),
            "spill_save_s": round(self.spill_save_s, 4),
            "decode_s": round(self.decode_s, 4),
            "decode_batched_s": round(self.decode_batched_s, 4),
            # The streaming lane admits through the driver, so the engine
            # reports no separate scheduling stamp to price a queue with.
            "queue_s": None,
            "receiver_ttft_s": round(self.receiver_ttft_s, 4),
            "generation_s": round(self.generation_s, 4),
            # What the caller waits for: the workers finish and the handoff is
            # prepared before the receiver can emit a token, so the producer
            # wall is inside TTFT rather than beside it.
            "ttft_s": round(self.producer_s + self.receiver_prepare_s + self.receiver_ttft_s, 4),
            "tteoa_s": round(self.producer_s + self.receiver_prepare_s + self.decode_s, 4),
            "decode_queued_offset_s_by_sample": [round(value, 4) for value in self.queued_offsets],
            "decode_first_token_offset_s_by_sample": [
                round(value, 4) for value in self.first_token_offsets
            ],
            "decode_finished_offset_s_by_sample": [
                round(value, 4) for value in self.finished_offsets
            ],
            "answer_continuation_submit_offset_s_by_sample": _rounded_or_none(
                self.continuation_submit_offsets, width=len(self.queued_offsets)
            ),
            "answer_continuation_first_token_offset_s_by_sample": _rounded_or_none(
                self.continuation_first_token_offsets, width=len(self.queued_offsets)
            ),
        }


def _rounded_or_none(offsets: tuple[float | None, ...] | None, *, width: int) -> list[float | None]:
    """Round banked optional offsets; an unbanked vector is one None per sample."""
    if offsets is None:
        return [None] * width
    return [None if value is None else round(value, 4) for value in offsets]


def _check_continuation_offsets(
    submits: tuple[float | None, ...] | None,
    firsts: tuple[float | None, ...] | None,
    *,
    width: int,
) -> None:
    """Refuse continuation stamps banked half, ragged, or out of order."""
    if (submits is None) != (firsts is None):
        raise ValueError("continuation offsets must be banked as a pair")
    if submits is None or firsts is None:
        return
    if len(submits) != width or len(firsts) != width:
        raise ValueError("continuation offsets must match the sample width")
    for submit, first in zip(submits, firsts, strict=True):
        if (submit is None) != (first is None):
            raise ValueError("a continuation banks both its submit and first-token stamps")
        if submit is not None and first is not None and not 0.0 <= submit <= first:
            raise ValueError("a continuation cannot emit before it was submitted")


def continuation_offsets(
    completions: Sequence[Any], origin: float
) -> tuple[tuple[float | None, ...], tuple[float | None, ...]]:
    """Read each sample's continuation stamps off its completion, on one origin.

    A completion that was never continued carries None for both. A lane pricing
    on engine clocks reads the same two stamps off its own metrics.
    """

    def offset(value: float | None) -> float | None:
        return None if value is None else float(value) - origin

    return (
        tuple(offset(getattr(row, "continuation_submitted_at", None)) for row in completions),
        tuple(offset(getattr(row, "continuation_first_token_at", None)) for row in completions),
    )


def continuation_offset_fields(completions: Sequence[Any], origin: float) -> dict[str, Any]:
    """Return the two continuation vectors, for a lane that builds its own row."""
    submits, firsts = continuation_offsets(completions, origin)
    _check_continuation_offsets(submits, firsts, width=len(tuple(completions)))
    return {
        "answer_continuation_submit_offset_s_by_sample": _rounded_or_none(
            submits, width=len(submits)
        ),
        "answer_continuation_first_token_offset_s_by_sample": _rounded_or_none(
            firsts, width=len(firsts)
        ),
    }


def publish_fleet_latency(
    rows: Sequence[Mapping[str, Any]],
    timing_rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    direct: bool = False,
) -> list[dict[str, Any]]:
    """Republish one arm's rows on the fleet clock and retain the compute clock.

    The compute-scope numbers stay under ``policy_compute_*`` and ``ttft_s`` and
    ``tteoa_s`` become stage-ledger numbers. See docs/running.md, Fleet latency.
    """
    stages = strict_stage_decomposition([dict(row) for row in timing_rows], arm, direct=direct)
    stage_by_qid = {str(item["qid"]): item for item in stages}
    qids = [str(row.get("qid") or "") for row in rows]
    if not all(qids) or len(set(qids)) != len(qids):
        raise RuntimeError(f"{arm}: fleet latency rows have duplicate or empty item identity")
    if set(qids) != set(stage_by_qid):
        raise RuntimeError(f"{arm}: incomplete fleet stage decomposition")
    published: list[dict[str, Any]] = []
    for row in rows:
        qid = str(row["qid"])
        if "timing_scope" not in row or any(name in row for name in LEGACY_DECODE_FIELDS):
            raise RuntimeError(f"{arm}/{qid}: row is not on the latency schema; re-bank it")
        if row.get("timing_scope") != COMPUTE_TIMING_SCOPE:
            raise RuntimeError(f"{arm}/{qid}: row is not on the compute timing scope")
        if row.get("ttft_s") is None or row.get("tteoa_s") is None:
            raise RuntimeError(f"{arm}/{qid}: row lacks the compute-scope caller clocks")
        stages = stage_by_qid[qid]
        output = dict(row)
        output["policy_compute_ttft_s"] = row["ttft_s"]
        output["policy_compute_tteoa_s"] = row["tteoa_s"]
        output["ttft_s"] = stages["fleet_ttft_s"]
        output["tteoa_s"] = stages["tteoa_s"]
        output["timing_scope"] = FLEET_TIMING_SCOPE
        for field in FLEET_STAGE_FIELDS:
            name = field if field.startswith("fleet_") else f"fleet_{field}"
            output[name] = stages[field]
        headline, wall = _s0_headline(row, stages, arm=arm, qid=qid)
        output["tteoa_s"] = headline
        output["fleet_tteoa_s"] = headline
        output["fleet_protocol_wall_s"] = wall
        published.append(output)
    return published


def _s0_headline(
    row: Mapping[str, Any], stages: Mapping[str, Any], *, arm: str, qid: str
) -> tuple[float, float]:
    """Return one row's (s0-completion headline, last-sample protocol wall).

    The headline is batch arrival to ``admitted`` plus the s0 finish offset banked
    at index 0; a row whose s0 offset lies past its ``decode_end`` is refused.
    """
    raw_offsets = row.get("decode_finished_offset_s_by_sample")
    offsets: list[object] = (
        list(cast(list[object], raw_offsets)) if isinstance(raw_offsets, list) else []
    )
    numeric = [
        value
        for value in offsets
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not offsets or len(numeric) != len(offsets):
        raise RuntimeError(f"{arm}/{qid}: row lacks the per-sample finish offsets")
    raw_wall = stages.get("tteoa_s")
    raw_decode = stages.get("decode_s")
    if raw_wall is None or raw_decode is None:
        raise RuntimeError(f"{arm}/{qid}: fleet lineage lacks the admitted or decode_end span")
    s0_finish = float(numeric[0])
    wall = float(raw_wall)
    decode_span = float(raw_decode)
    if s0_finish < 0.0 or s0_finish > decode_span + 0.01:
        raise RuntimeError(f"{arm}/{qid}: s0 finished after the last sample's decode_end")
    return round(wall - decode_span + s0_finish, 4), wall


__all__ = (
    "COMPUTE_TIMING_SCOPE",
    "FLEET_DECODE_SCHEDULE",
    "FLEET_STAGE_FIELDS",
    "FLEET_TIMING_SCOPE",
    "ITEM_LATENCY_FIELDS",
    "LEGACY_DECODE_FIELDS",
    "WARMUP_BODY",
    "WARMUP_MAX_TOKENS",
    "ItemLatency",
    "publish_fleet_latency",
    "warmup_seed",
)
