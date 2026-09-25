"""Atomic filesystem tickets for the producer-to-receiver fleet handoff.

One arm has two queues under its run root, and a ticket is claimed out of one by
atomic rename. Every claim event is appended to the queue's claim ledger.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from rcc.run.io import sha256_file as _sha256_file

_READY = "ready"
_CLAIMED = "claimed"
_TMP = "tmp"
_WORKERS = "workers"
_CLOCK = "fleet_clock.json"
_CLAIMS = "claims.jsonl"
_QUEUES = "queues"
_MAX_ORDER_INDEX = 10**20 - 1
_CLAIM_FIELDS = frozenset(("event", "ticket", "owner", "qid", "at_unix"))
_CLAIM_EVENTS = frozenset(("claim", "release", "requeue"))
#: The two queues one arm holds below its run root: ``todo`` carries the panel to
#: whoever drains it, ``ready`` carries producer output to the receivers. A direct
#: arm publishes its panel straight into ``ready`` and leaves ``todo`` empty.
QUEUE_NAMES = ("todo", "ready")


def ticket_filename(qid: str, order_index: int) -> str:
    """Return a sortable, opaque filename that does not spell out the item id."""
    if not 0 <= order_index <= _MAX_ORDER_INDEX:
        raise ValueError("ticket order_index is outside the 20-digit queue range")
    digest = hashlib.sha256(qid.encode("utf-8")).hexdigest()
    return f"{order_index:020d}__{digest}.json"


@dataclass(frozen=True)
class Ticket:
    """One queued item: its identity, its payload on disk, and its stamps.

    ``fields`` carries whatever the producer measured about the artifact. A
    receiver in another process cannot recover those clocks from the payload.
    """

    qid: str
    arm: str
    order_index: int
    payload_path: str
    payload_sha256: str
    producer_gpu: int
    produced_at_unix: float
    fields: dict[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        """Validate the ticket identity and queue order."""
        if self.order_index < 0:
            raise ValueError(f"{self.qid}: order_index must be non-negative")
        if not self.qid or not self.arm:
            raise ValueError("a ticket needs a qid and an arm")

    def filename(self) -> str:
        """Return the lexically sortable name that keeps FIFO submission order."""
        return ticket_filename(self.qid, self.order_index)


def arm_queue_roots(run_root: Path) -> tuple[Path, Path]:
    """Return one arm's ``todo`` and ``ready`` queue roots, in that order."""
    queues = Path(run_root) / _QUEUES
    return queues / QUEUE_NAMES[0], queues / QUEUE_NAMES[1]


def queue_dirs(root: Path) -> tuple[Path, Path, Path]:
    """Create and return a queue's ready, claimed, and staging directories."""
    ready, claimed, tmp = root / _READY, root / _CLAIMED, root / _TMP
    for path in (ready, claimed, tmp):
        path.mkdir(parents=True, exist_ok=True)
    return ready, claimed, tmp


def _durable_write(
    staging: Path,
    target: Path,
    text: str,
    *,
    before_publish: Callable[[], None] | None = None,
) -> None:
    with staging.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if before_publish is not None:
            before_publish()
        os.replace(staging, target)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    directory = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def publish_ticket(
    root: Path,
    ticket: Ticket,
    *,
    before_publish: Callable[[], None] | None = None,
) -> Path:
    """Stage the ticket, run the optional hook, then publish it to ``ready``."""
    ready, _, tmp = queue_dirs(root)
    target = ready / ticket.filename()
    staging = tmp / f"{ticket.filename()}.{os.getpid()}"
    _durable_write(
        staging,
        target,
        json.dumps(asdict(ticket), sort_keys=True) + "\n",
        before_publish=before_publish,
    )
    return target


def read_ticket(path: Path) -> Ticket:
    """Read one ticket from a claimed or ready queue path."""
    return Ticket(**json.loads(path.read_text(encoding="utf-8")))


def _record_claim_event(root: Path, *, event: str, ticket: str, owner: str, qid: str) -> None:
    """Append one line to a queue's append-only claim ledger.

    A claim file is deleted as soon as its work is banked, so this ledger is what
    outlives the claim and shows each ticket was claimed exactly once.
    """
    line = (
        json.dumps(
            {
                "event": event,
                "ticket": ticket,
                "owner": owner,
                "qid": qid,
                "at_unix": time.time(),
            },
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if event not in _CLAIM_EVENTS:
        raise ValueError(f"unknown claim-ledger event {event!r}")
    handle = os.open(root / _CLAIMS, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        size = os.lseek(handle, 0, os.SEEK_END)
        os.lseek(handle, 0, os.SEEK_SET)
        payload = bytearray()
        while len(payload) < size:
            chunk = os.read(handle, size - len(payload))
            if not chunk:
                break
            payload.extend(chunk)
        boundary = bytes(payload).rfind(b"\n") + 1
        if boundary != size:
            os.ftruncate(handle, boundary)
        remaining = memoryview(line)
        while remaining:
            written = os.write(handle, remaining)
            if written <= 0:
                raise OSError("claim-ledger append made no progress")
            remaining = remaining[written:]
        os.fsync(handle)
    finally:
        os.close(handle)


def read_claim_ledger(root: Path) -> list[dict[str, Any]]:
    """Read one queue's claim ledger, oldest event first."""
    ledger = Path(root) / _CLAIMS
    if not ledger.is_file():
        return []
    payload = ledger.read_bytes()
    boundary = payload.rfind(b"\n") + 1
    events: list[dict[str, Any]] = []
    for number, raw in enumerate(payload[:boundary].splitlines(), 1):
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"{ledger}:{number}: invalid claim-ledger line") from exc
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{ledger}:{number}: invalid claim-ledger line") from exc
        if not isinstance(value, dict) or not set(cast(dict[str, Any], value)) >= _CLAIM_FIELDS:
            raise RuntimeError(f"{ledger}:{number}: incomplete claim-ledger line")
        event = cast(dict[str, Any], value)
        if event.get("event") not in _CLAIM_EVENTS:
            raise RuntimeError(f"{ledger}:{number}: unknown claim-ledger event")
        events.append(event)
    return events


def _claim_balance(root: Path, ticket: str) -> int:
    return sum(
        1 if event["event"] == "claim" else -1
        for event in read_claim_ledger(root)
        if event["ticket"] == ticket
    )


def claim_next(root: Path, *, owner: str) -> tuple[Ticket, Path] | None:
    """Claim the oldest ready ticket by same-filesystem atomic rename."""
    ready, claimed, _ = queue_dirs(root)
    own = claimed / owner
    own.mkdir(parents=True, exist_ok=True)
    for entry in sorted(ready.iterdir()):
        if not entry.name.endswith(".json"):
            continue
        target = own / entry.name
        if target.exists():
            continue
        try:
            os.rename(entry, target)
        except (FileNotFoundError, PermissionError):
            continue
        # Publication mtime may be old; leases age from the successful claim.
        os.utime(target, None)
        ticket = read_ticket(target)
        other_claim = any(path != target for path in claimed.glob(f"*/{target.name}"))
        if not other_claim and _claim_balance(root, target.name) > 0:
            # An open ledger entry with no claim file left on disk is an
            # orphan: this atomic claim proves the old claim file is gone, so
            # close the entry before recording this one.
            _record_claim_event(
                root,
                event="release",
                ticket=target.name,
                owner=owner,
                qid=ticket.qid,
            )
        _record_claim_event(root, event="claim", ticket=target.name, owner=owner, qid=ticket.qid)
        return ticket, target
    return None


def requeue_claim(root: Path, claim: Path, *, qid: str) -> None:
    """Return one claimed ticket to ``ready`` and record the release."""
    ready, _, _ = queue_dirs(root)
    owner = claim.parent.name
    os.replace(claim, ready / claim.name)
    _record_claim_event(root, event="requeue", ticket=claim.name, owner=owner, qid=qid)


def release_claim(root: Path, claim: Path, *, qid: str) -> None:
    """Close finished work in the ledger, then remove its claim file."""
    owner = claim.parent.name
    _record_claim_event(root, event="release", ticket=claim.name, owner=owner, qid=qid)
    claim.unlink(missing_ok=True)


def verify_ticket_payload(ticket: Ticket) -> None:
    """Refuse a payload that is missing or whose digest differs."""
    path = Path(ticket.payload_path)
    if not path.is_file():
        raise FileNotFoundError(f"{ticket.qid}: payload missing at {path}")
    actual = _sha256_file(path)
    if actual != ticket.payload_sha256:
        raise ValueError(
            f"{ticket.qid}: payload hash {actual[:12]} does not match the "
            f"published {ticket.payload_sha256[:12]}; refusing a torn handoff"
        )


def reset_stale_claims(root: Path, *, banked_qids: set[str]) -> list[str]:
    """Delete banked claims and return every unbanked claim to ``ready``."""
    _, claimed, _ = queue_dirs(root)
    balances = {path.name: _claim_balance(root, path.name) for path in claimed.glob("*/*.json")}
    requeued: list[str] = []
    for claim in sorted(claimed.glob("*/*.json")):
        try:
            ticket = read_ticket(claim)
        except FileNotFoundError:
            continue
        if balances.get(claim.name, 0) <= 0:
            claim.unlink(missing_ok=True)
            continue
        if ticket.qid in banked_qids:
            release_claim(root, claim, qid=ticket.qid)
            continue
        requeue_claim(root, claim, qid=ticket.qid)
        requeued.append(ticket.qid)
    return requeued


def publish_worker_ready(root: Path, worker: str) -> Path:
    """Publish one warm-worker marker for an attempt-scoped barrier."""
    workers = root / _WORKERS
    workers.mkdir(parents=True, exist_ok=True)
    marker = workers / f"{worker}.ready"
    staging = marker.with_suffix(f".tmp{os.getpid()}")
    _durable_write(staging, marker, f"{time.time()}\n")
    return marker


def count_ready_workers(root: Path) -> int:
    """Count the warm-worker markers of one attempt-scoped barrier."""
    workers = root / _WORKERS
    if not workers.is_dir():
        return 0
    return sum(1 for entry in workers.iterdir() if entry.suffix == ".ready")


def publish_clock(root: Path, *, expected_workers: int, t0: float | None = None) -> float:
    """Publish the common batch-arrival clock after every worker is warm."""
    ready = count_ready_workers(root)
    if ready < expected_workers:
        raise RuntimeError(f"only {ready} of {expected_workers} workers are warm; no clock yet")
    t0 = time.time() if t0 is None else float(t0)
    staging = root / f"{_CLOCK}.tmp{os.getpid()}"
    target = root / _CLOCK
    _durable_write(staging, target, json.dumps({"t0_unix": t0, "workers": ready}) + "\n")
    return t0
