"""Hash the worker shards of a prepared item and check the roster they form."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any, Protocol, cast

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50


class ShardedItem(Protocol):
    """The prepared item fields the shard fingerprint reads."""

    @property
    def qid(self) -> str:
        """Return the source item id."""
        ...

    @property
    def shards(self) -> tuple[tuple[int, ...], ...]:
        """Return this item's per-worker token shards."""
        ...


def token_sequence_sha256(tokens: Sequence[int]) -> str:
    """Hash one token sequence in a platform-independent encoding."""
    digest = hashlib.sha256()
    for token in tokens:
        if type(token) is not int or not 0 <= token < 2**32:
            raise RuntimeError("invalid prepared token id")
        digest.update(token.to_bytes(4, "big"))
    return digest.hexdigest()


def shard_fingerprints(items: Sequence[ShardedItem]) -> list[dict[str, Any]]:
    """Hash every worker token sequence without serializing the tokens."""
    output: list[dict[str, Any]] = []
    for item in items:
        workers: list[dict[str, Any]] = []
        for worker, shard in enumerate(item.shards):
            try:
                sha256 = token_sequence_sha256(shard)
            except RuntimeError as exc:
                raise RuntimeError(f"{item.qid}/w{worker}: invalid prepared token id") from exc
            workers.append({"worker": worker, "tokens": len(shard), "sha256": sha256})
        output.append({"qid": item.qid, "workers": workers})
    return output


def validate_shard_fingerprints(value: Any, qids: tuple[str, ...]) -> None:
    """Raise unless the roster names every item in order with its full worker set."""
    workers_per_item = FANOUTQA_NATURAL_DEV50.workers_per_item
    if not isinstance(value, list):
        raise RuntimeError("M=3 source audit has malformed exact shard fingerprints")
    untyped_rows = cast(list[object], value)
    if any(not isinstance(row, dict) for row in untyped_rows):
        raise RuntimeError("M=3 source audit has malformed exact shard fingerprints")
    rows = cast(list[dict[str, Any]], untyped_rows)
    if tuple(row.get("qid") for row in rows) != qids or any(
        set(row) != {"qid", "workers"}
        or not isinstance(row["workers"], list)
        or len(cast(list[object], row["workers"])) != workers_per_item
        or any(
            set(worker) != {"worker", "tokens", "sha256"}
            or type(worker["worker"]) is not int
            or worker["worker"] != index
            or type(worker["tokens"]) is not int
            or worker["tokens"] <= 0
            or not isinstance(worker["sha256"], str)
            or len(worker["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in worker["sha256"])
            for index, worker in enumerate(cast(list[dict[str, Any]], row["workers"]))
        )
        for row in rows
    ):
        raise RuntimeError("M=3 source audit has malformed exact shard fingerprints")


__all__ = (
    "shard_fingerprints",
    "token_sequence_sha256",
    "validate_shard_fingerprints",
)
