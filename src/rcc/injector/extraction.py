"""Read a request's prompt KV back out of the engine's pages, the mirror of injection.

The scheduler role accumulates the block list and emits an order only on the step whose
scheduled tokens finish the prompt; a first-schedule order would ship only the first chunk.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import torch

from rcc.injector.layout import Geometry, slot_mapping, unpack_cache
from rcc.injector.staging import StagedCache, StagingRegistry


@dataclass(frozen=True)
class ExtractSpec:
    """One completed extraction order: identity, prompt length, the full block list."""

    request_id: str
    num_prompt_tokens: int
    block_ids: tuple[int, ...]


@dataclass
class _Capture:
    """One tracked request's accumulating state, private to the planner."""

    num_prompt_tokens: int
    blocks: list[int] = field(default_factory=lambda: list[int]())
    seen: int = 0


class ExtractionPlanner:
    """Scheduler-side extraction state: block accumulation and completion.

    Per step: `start`, `reset` for a preemption resume (a replacement block list and
    zeroed counts), `extend`, then `advance`, which emits one spec at completion.
    """

    def __init__(self) -> None:
        """Start with nothing tracked."""
        self._captures: dict[str, _Capture] = {}

    def tracks(self, request_id: str) -> bool:
        """Return whether this request's extraction is still being assembled."""
        return request_id in self._captures

    def start(self, request_id: str, num_prompt_tokens: int, block_ids: Iterable[int]) -> None:
        """Begin tracking at the request's first schedule."""
        if num_prompt_tokens < 1:
            raise ValueError(f"prompt must be non-empty, got {num_prompt_tokens}")
        self._captures[request_id] = _Capture(
            num_prompt_tokens=num_prompt_tokens,
            blocks=[int(block) for block in block_ids],
        )

    def extend(self, request_id: str, block_ids: Iterable[int]) -> None:
        """Append a later chunk's newly allocated blocks."""
        capture = self._captures.get(request_id)
        if capture is None:
            return
        capture.blocks.extend(int(block) for block in block_ids)

    def reset(self, request_id: str, block_ids: Iterable[int]) -> None:
        """Replace state on a preemption resume: fresh blocks, count restarts at zero."""
        capture = self._captures.get(request_id)
        if capture is None:
            return
        capture.blocks = [int(block) for block in block_ids]
        capture.seen = 0

    def discard(self, request_id: str) -> None:
        """Stop tracking a request, whether it finished or was aborted."""
        self._captures.pop(request_id, None)

    def advance(self, scheduled_tokens: Mapping[str, int]) -> list[ExtractSpec]:
        """Account one step's scheduled tokens; return the orders completed by it."""
        completed: list[ExtractSpec] = []
        for request_id, count in scheduled_tokens.items():
            capture = self._captures.get(request_id)
            if capture is None:
                continue
            seen = capture.seen + int(count)
            if seen > capture.num_prompt_tokens:
                raise ValueError(
                    f"prompt token overshoot for {request_id!r}: {seen} scheduled against "
                    f"{capture.num_prompt_tokens} prompt tokens; the accounting is broken"
                )
            capture.seen = seen
            if seen == capture.num_prompt_tokens:
                completed.append(
                    ExtractSpec(
                        request_id=request_id,
                        num_prompt_tokens=capture.num_prompt_tokens,
                        block_ids=tuple(capture.blocks),
                    )
                )
                self.discard(request_id)
        return completed


def extract_request(
    layer_tensors: list[torch.Tensor],
    spec: ExtractSpec,
    registry: StagingRegistry,
    geometry: Geometry,
) -> None:
    """Gather one completed request's prompt KV into the registry, reading only.

    The gathered tensors are fresh copies on the pages' device, so the pages are
    neither aliased nor mutated. Positions are `arange(num_prompt_tokens)`.
    """
    slots = slot_mapping(spec.block_ids, spec.num_prompt_tokens, geometry.block_size)
    keys, values = unpack_cache(layer_tensors, slots)
    registry.put(
        spec.request_id,
        StagedCache(
            keys=keys,
            values=values,
            positions=torch.arange(spec.num_prompt_tokens, dtype=torch.int64, device=keys.device),
        ),
    )


#: The in-process registry completed extractions land in. The caller takes each
#: entry from here and finishes it once consumed, so no device memory stays
#: pinned across requests.
EXTRACTED = StagingRegistry()

#: Request ids marked for extraction. The scheduler role consumes an id on the
#: request's first schedule.
REQUESTED: set[str] = set()


def request_extraction(request_id: str) -> None:
    """Mark a request id for prompt-KV extraction, before `add_request` submits it."""
    REQUESTED.add(request_id)
