"""Three-worker source construction and receiver assembly for the Gemma probe.

Construction runs the registered page assignment and packing; assembly joins
the kept sets and the question turn into one request. Nothing touches the model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

WORKERS_PER_ITEM = 3
"""The registered fan-out width this probe runs at."""

PAGE_ALLOCATION = "max_min_query"
"""The registered source-construction policy: fixed assignment, live quotas."""


@dataclass(frozen=True)
class WorkerSource:
    """One worker's packed evidence and the pages it owns."""

    worker: int
    memory_ids: tuple[int, ...]
    pageids: tuple[int, ...]

    def __post_init__(self) -> None:
        """Refuse a worker that owns nothing or carries nothing."""
        if self.worker < 0:
            raise ValueError(f"worker index must be nonnegative, got {self.worker}")
        if not self.memory_ids:
            raise RuntimeError(f"worker {self.worker} packed no evidence tokens")
        if not self.pageids:
            raise RuntimeError(f"worker {self.worker} owns no pages")


@dataclass(frozen=True)
class ItemSource:
    """One item's three workers plus the construction record they came from."""

    qid: str
    workers: tuple[WorkerSource, ...]
    dead_pages: tuple[int, ...]
    ledger: tuple[Mapping[str, Any], ...]
    structure: Mapping[str, Any]

    @property
    def memory_tokens(self) -> tuple[int, ...]:
        """Each worker's packed memory length, in worker order."""
        return tuple(len(worker.memory_ids) for worker in self.workers)

    @property
    def aggregate_memory_tokens(self) -> int:
        """The total rows a full-arm receiver prefill carries."""
        return sum(self.memory_tokens)


@dataclass(frozen=True)
class ReceiverAssembly:
    """One receiver request and the row provenance of its memory block."""

    rows: torch.Tensor
    segments: tuple[tuple[int, int, int], ...]
    prefix_rows: int
    latent_rows: int
    prompt_rows: int
    suffix_rows: int

    @property
    def total_rows(self) -> int:
        """The row count the receiver actually prefills."""
        return int(self.rows.shape[0])


def assemble_receiver_request(
    *,
    turn_prefix: torch.Tensor,
    memory_blocks: Sequence[torch.Tensor],
    prompt: torch.Tensor,
    turn_suffix: torch.Tensor,
) -> ReceiverAssembly:
    """Concatenate the kept sets and the question in the Qwen wrapper order.

    Memory blocks come first, so the payload precedes the chat turn; `segments`
    records them in absolute coordinates, `receiver_block_kinds` their workers.
    """
    if turn_prefix.ndim != 2 or prompt.ndim != 2 or turn_suffix.ndim != 2:
        raise ValueError("receiver assembly needs two-dimensional [rows, hidden] blocks")
    width = int(prompt.shape[1])
    parts: list[torch.Tensor] = []
    segments: list[tuple[int, int, int]] = []
    cursor = 0
    latent_rows = 0
    for index, block in enumerate(memory_blocks):
        if block.ndim != 2 or int(block.shape[1]) != width:
            raise ValueError(f"memory block {index} is {tuple(block.shape)}, not [rows, {width}]")
        count = int(block.shape[0])
        if count == 0:
            raise RuntimeError(
                f"memory block {index} contributed no rows; an empty block is a "
                "construction failure, not an arm with less memory"
            )
        parts.append(block.to(dtype=prompt.dtype))
        segments.append((index, cursor, cursor + count))
        cursor += count
        latent_rows += count
    parts.append(turn_prefix.to(dtype=prompt.dtype))
    parts.append(prompt)
    parts.append(turn_suffix.to(dtype=prompt.dtype))
    rows = torch.cat(parts, dim=0)
    assembly = ReceiverAssembly(
        rows=rows,
        segments=tuple(segments),
        prefix_rows=int(turn_prefix.shape[0]),
        latent_rows=latent_rows,
        prompt_rows=int(prompt.shape[0]),
        suffix_rows=int(turn_suffix.shape[0]),
    )
    require_assembly_geometry(assembly)
    return assembly


def require_assembly_geometry(assembly: ReceiverAssembly) -> None:
    """Refuse an assembly whose row accounting does not add up.

    The failure it catches is silent: a request that lost or duplicated one
    worker's rows still decodes and still scores.
    """
    expected = (
        assembly.prefix_rows + assembly.latent_rows + assembly.prompt_rows + assembly.suffix_rows
    )
    if assembly.total_rows != expected:
        raise RuntimeError(
            f"receiver request holds {assembly.total_rows} rows, not the "
            f"{expected} its own prefix/memory/prompt/suffix accounting declares"
        )
    if [index for index, _start, _end in assembly.segments] != list(range(len(assembly.segments))):
        raise RuntimeError("receiver memory segments are not in block order")
    # The payload opens the request, so the first memory row is row zero and
    # the chat turn begins exactly where the memory ends.
    cursor = 0
    for index, start, end in assembly.segments:
        if start != cursor or end <= start:
            raise RuntimeError(
                f"memory block {index} segment [{start}, {end}) is not contiguous with the "
                f"block that precedes it at {cursor}"
            )
        cursor = end
    if cursor != assembly.latent_rows:
        raise RuntimeError("receiver memory segments do not cover the declared memory block")
