"""Streaming engine driver and item-level token admission.

The decoder drives an add-request/step engine and applies the lane's answer
closer to a draw that ended unclosed. The tracker offers items in FIFO order.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from rcc.run.fleet.admission import AdmissionGate
from rcc.run.fleet.answer_closer import FINISH_REASONS, AnswerCloser
from rcc.run.fleet.progress import tick_seat_progress

_POLL_SECONDS = 0.01
#: Appended to the head's request id for its one closing continuation. The
#: merged completion is published under the head's id, so the item tracker and
#: the admission gate never see this one.
_CLOSER_SUFFIX = "#closer"


@dataclass(frozen=True)
class StreamRequest:
    """One engine request with benchmark identity and KV geometry."""

    request_id: str
    prompt: Any
    sampling: Any
    prompt_tokens: int
    qid: str
    tag: str

    def __post_init__(self) -> None:
        """Validate the request's prompt-token geometry."""
        if self.prompt_tokens <= 0:
            raise ValueError(f"{self.request_id}: prompt_tokens must be positive")


@dataclass(frozen=True)
class StreamCompletion:
    """One finished request with driver-clock and engine readout fields."""

    request_id: str
    qid: str
    tag: str
    text: str
    n_tokens: int
    finish_reason: str
    submitted_at: float
    first_token_at: float | None
    finished_at: float
    num_cached_tokens: int | None
    token_ids: tuple[int, ...] = ()
    engine_metrics: Any = None
    #: Set when the closing continuation is issued, not when it is merged, so
    #: the activation rate counts what was acted on rather than what completed.
    answer_injected: bool = False
    #: The continuation's own outcome, kept beside the head's rather than
    #: replacing it: a continuation that spent the whole closing budget ends on
    #: "length" while the head ended on "stop".
    continuation_finish_reason: str | None = None
    #: Why an unclosed draw was not continued. The head is published unclosed.
    closer_refusal: str | None = None
    #: The continuation's own driver-clock stamps. With prefix caching off the
    #: continuation re-prefills the prompt and the head before its first token,
    #: so the span between these two is a cost of the injection itself.
    continuation_submitted_at: float | None = None
    continuation_first_token_at: float | None = None


@dataclass(frozen=True)
class _MergedMetrics:
    """Engine stamps for a continued draw: head start, continuation end.

    The continuation's own queued and first-token stamps ride along, for the
    lane that prices its cells on engine clocks rather than driver clocks.
    """

    queued_ts: float | None
    scheduled_ts: float | None
    first_token_ts: float | None
    last_token_ts: float | None
    continuation_queued_ts: float | None = None
    continuation_first_token_ts: float | None = None


def _merged_metrics(head: Any, tail: Any) -> Any:
    if head is None and tail is None:
        return None
    return _MergedMetrics(
        queued_ts=getattr(head, "queued_ts", None),
        scheduled_ts=getattr(head, "scheduled_ts", None),
        first_token_ts=getattr(head, "first_token_ts", None),
        last_token_ts=getattr(tail, "last_token_ts", None),
        continuation_queued_ts=getattr(tail, "queued_ts", None),
        continuation_first_token_ts=getattr(tail, "first_token_ts", None),
    )


@dataclass(frozen=True)
class _Readout:
    n_tokens: int
    text: str
    finish_reason: str
    num_cached_tokens: int | None
    token_ids: tuple[int, ...] = ()
    finished: bool = False
    engine_metrics: Any = None


def _completion_of(output: Any) -> _Readout:
    completions = getattr(output, "outputs", None)
    if not completions:
        return _Readout(0, "", "", None, finished=bool(getattr(output, "finished", False)))
    head = completions[0]
    token_ids = getattr(head, "token_ids", None) or ()
    cached = getattr(output, "num_cached_tokens", None)
    return _Readout(
        n_tokens=len(list(token_ids)),
        text=str(getattr(head, "text", "") or ""),
        finish_reason=str(getattr(head, "finish_reason", None) or ""),
        num_cached_tokens=int(cached) if cached is not None else None,
        token_ids=tuple(map(int, token_ids)),
        finished=bool(getattr(output, "finished", False)),
        engine_metrics=getattr(output, "metrics", None),
    )


@dataclass(frozen=True)
class _Closing:
    """One head awaiting its single closing continuation."""

    request: StreamRequest
    head: _Readout
    submitted_at: float
    first_token_at: float | None


class StreamDecoder:
    """Drive an add-request/step engine and surface each completion immediately."""

    def __init__(
        self,
        engine: Any,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        closer: AnswerCloser | None = None,
        on_continue: Callable[[str, int], None] | None = None,
    ) -> None:
        """Bind one stepping engine to injectable clocks and its closer policy.

        ``on_continue`` is called with the head's request id and the tokens its
        continuation may add, so the tail is booked on the head's reservation.
        """
        self.engine = engine
        self.clock = clock
        self.sleep = sleep
        self.closer = closer
        self.on_continue = on_continue
        self._live: dict[str, StreamRequest] = {}
        self._submitted_at: dict[str, float] = {}
        self._first_token_at: dict[str, float] = {}
        self._closing: dict[str, _Closing] = {}
        self.steps = 0
        self.continuations = 0

    def in_flight(self) -> int:
        """Return the number of requests still owned by the engine."""
        return len(self._live)

    def submit(self, request: StreamRequest) -> None:
        """Submit without waiting for decode completion."""
        if request.request_id in self._live:
            raise ValueError(f"request {request.request_id!r} is already live")
        self._live[request.request_id] = request
        self._submitted_at[request.request_id] = self.clock()
        try:
            self.engine.add_request(request.request_id, request.prompt, request.sampling)
        except Exception:
            del self._live[request.request_id]
            del self._submitted_at[request.request_id]
            raise

    def poll(self) -> list[StreamCompletion]:
        """Step once and return every request that finished during the step."""
        if not self._live:
            return []
        self.steps += 1
        outputs = self.engine.step()
        finished: list[StreamCompletion] = []
        progressed = False
        for output in outputs:
            request_id = str(getattr(output, "request_id", ""))
            request = self._live.get(request_id)
            if request is None:
                continue
            progressed = True
            readout = _completion_of(output)
            if readout.n_tokens and request_id not in self._first_token_at:
                self._first_token_at[request_id] = self.clock()
            if readout.finished:
                finished.extend(self._settle(request, readout))
        if self._live and not self.engine.has_unfinished_requests():
            finished.extend(self._drop_live())
        elif not progressed:
            self.sleep(_POLL_SECONDS)
        # After every clock read above, so no published submit, first-token, or
        # finish timestamp carries the cost of writing the progress record.
        tick_seat_progress(self.steps)
        return finished

    def abort_all(self) -> list[str]:
        """Abort every live request and return their ids."""
        live_ids = list(self._live)
        aborter = getattr(self.engine, "abort_request", None)
        if aborter is not None and live_ids:
            try:
                aborter(live_ids)
            except Exception:
                pass
        self._live.clear()
        self._closing.clear()
        return live_ids

    def _drop_live(self) -> list[StreamCompletion]:
        """Settle every request the engine no longer knows about, rather than spin."""
        dropped: list[StreamCompletion] = []
        for request_id in list(self._live):
            readout = _Readout(0, "", "dropped", None, finished=True)
            dropped.extend(self._settle(self._live[request_id], readout))
        return dropped

    def _settle(self, request: StreamRequest, readout: _Readout) -> list[StreamCompletion]:
        """Publish one finished request, or hold it for its closing continuation.

        Exactly one of three things happens to a finished head, in this order: a
        continuation is merged, an affordable unclosed head is held, or it ships.
        """
        closing = self._closing.pop(request.request_id, None)
        if closing is not None:
            return [self._merge(closing, readout)]
        closer = self.closer
        if (
            closer is None
            or readout.finish_reason not in FINISH_REASONS
            or not closer.is_unclosed(readout.token_ids)
        ):
            return [self._finish(request, readout)]
        refusal = closer.refusal(readout.token_ids, prompt_tokens=request.prompt_tokens)
        if refusal is not None:
            return [self._finish(request, readout, closer_refusal=refusal)]
        self._continue(closer, request, readout)
        return []

    def _continue(self, closer: AnswerCloser, request: StreamRequest, head: _Readout) -> None:
        """Issue the one closing continuation for an unclosed head.

        The head's reservation is extended rather than released, since the ceiling
        bounds the head alone and the continuation decodes past what it booked.
        """
        if self.on_continue is not None:
            self.on_continue(request.request_id, closer.continuation_tokens)
        content = closer.head_content(head.token_ids)
        continuation = replace(
            request,
            request_id=f"{request.request_id}{_CLOSER_SUFFIX}",
            prompt=closer.continuation_prompt(request.prompt, content, closer.closer_ids),
            sampling=closer.continuation_sampling(request.sampling, closer.closing_budget),
            prompt_tokens=request.prompt_tokens + len(content) + len(closer.closer_ids),
        )
        del self._live[request.request_id]
        self._closing[continuation.request_id] = _Closing(
            # Only the identity is kept: the head's prompt is a large host
            # tensor and the continuation already owns its own copy.
            request=replace(request, prompt=None, sampling=None),
            head=head,
            submitted_at=self._submitted_at.pop(request.request_id),
            first_token_at=self._first_token_at.pop(request.request_id, None),
        )
        self.continuations += 1
        try:
            self.submit(continuation)
        except Exception:
            del self._closing[continuation.request_id]
            raise

    def _merge(self, closing: _Closing, tail: _Readout) -> StreamCompletion:
        """Bank head, injected closer, and continuation as one draw.

        This raises where the head path would not: a continuation answered with
        something unbankable would merge unverifiable ids into a published answer.
        """
        closer = self.closer
        assert closer is not None
        request = closing.request
        label = request.request_id
        continuation_id = f"{label}{_CLOSER_SUFFIX}"
        self._live.pop(continuation_id, None)
        continuation_submitted_at = self._submitted_at.pop(continuation_id, None)
        continuation_first_token_at = self._first_token_at.pop(continuation_id, None)
        if tail.finish_reason == "dropped":
            # The engine forgot the continuation. It is published under the
            # head's id as "dropped" for the lane to refuse, with the injection
            # already counted kept visible on the row.
            return StreamCompletion(
                request_id=label,
                qid=request.qid,
                tag=request.tag,
                text=closing.head.text,
                n_tokens=0,
                finish_reason="dropped",
                submitted_at=closing.submitted_at,
                first_token_at=closing.first_token_at,
                finished_at=self.clock(),
                num_cached_tokens=None,
                token_ids=(),
                engine_metrics=None,
                answer_injected=True,
                continuation_finish_reason="dropped",
                continuation_submitted_at=continuation_submitted_at,
            )
        if tail.finish_reason not in FINISH_REASONS:
            raise RuntimeError(f"{label}: answer closer continuation finish reason is invalid")
        if tail.num_cached_tokens not in (None, 0):
            raise RuntimeError(f"{label}: prefix caching contaminated an answer continuation")
        if len(tail.token_ids) != int(tail.n_tokens):
            raise RuntimeError(f"{label}: answer closer continuation token count differs from ids")
        if len(tail.token_ids) > closer.closing_budget:
            raise RuntimeError(f"{label}: answer closer continuation exceeded its closing budget")
        ids = closer.merged_ids(closing.head.token_ids, tail.token_ids)
        return StreamCompletion(
            request_id=label,
            qid=request.qid,
            tag=request.tag,
            # The backend readout, concatenated. Each lane reconstructs the
            # text it reads from the merged ids beside this field.
            text=f"{closing.head.text}{tail.text}",
            n_tokens=len(ids),
            finish_reason=closing.head.finish_reason,
            submitted_at=closing.submitted_at,
            first_token_at=closing.first_token_at,
            finished_at=self.clock(),
            num_cached_tokens=closing.head.num_cached_tokens,
            token_ids=ids,
            engine_metrics=_merged_metrics(closing.head.engine_metrics, tail.engine_metrics),
            answer_injected=True,
            continuation_finish_reason=tail.finish_reason,
            continuation_submitted_at=continuation_submitted_at,
            continuation_first_token_at=continuation_first_token_at,
        )

    def _finish(
        self,
        request: StreamRequest,
        readout: _Readout,
        *,
        closer_refusal: str | None = None,
    ) -> StreamCompletion:
        del self._live[request.request_id]
        return StreamCompletion(
            request_id=request.request_id,
            qid=request.qid,
            tag=request.tag,
            text=readout.text,
            n_tokens=readout.n_tokens,
            finish_reason=readout.finish_reason,
            submitted_at=self._submitted_at.pop(request.request_id),
            first_token_at=self._first_token_at.pop(request.request_id, None),
            finished_at=self.clock(),
            num_cached_tokens=readout.num_cached_tokens,
            token_ids=readout.token_ids,
            engine_metrics=readout.engine_metrics,
            closer_refusal=closer_refusal,
        )


@dataclass
class _ItemState:
    requests: list[StreamRequest]
    next_request: int = 0
    completions: dict[str, StreamCompletion] = field(default_factory=dict[str, StreamCompletion])
    admitted_at: float | None = None


class ItemStreamTracker:
    """Offer FIFO items and group streamed sample completions back by item."""

    def __init__(self, decoder: StreamDecoder, gate: AdmissionGate) -> None:
        """Bind a stream decoder to its shared admission gate.

        The decoder's one closing continuation is booked on the head's reservation
        the moment it is issued, so the gate accounts for every live token.
        """
        self.decoder = decoder
        self.gate = gate
        decoder.on_continue = gate.extend
        self._queue: list[str] = []
        self._items: dict[str, _ItemState] = {}
        self.admission_trace: dict[str, dict[str, int]] = {}

    def offer(self, qid: str, requests: list[StreamRequest]) -> None:
        """Queue one item's ordered receiver requests."""
        if not requests:
            raise ValueError(f"{qid}: an item needs at least one request")
        if qid in self._items:
            raise ValueError(f"item {qid!r} is already tracked")
        self._items[qid] = _ItemState(requests=list(requests))
        self._queue.append(qid)

    def pump(self) -> list[tuple[str, float, list[StreamCompletion]]]:
        """Admit what fits, step once, and return newly completed items."""
        self._admit()
        finished_items: list[tuple[str, float, list[StreamCompletion]]] = []
        for completion in self.decoder.poll():
            state = self._items.get(completion.qid)
            self.gate.release(completion.request_id)
            if state is None:
                continue
            state.completions[completion.request_id] = completion
            if len(state.completions) == len(state.requests):
                ordered = [state.completions[req.request_id] for req in state.requests]
                assert state.admitted_at is not None
                finished_items.append((completion.qid, state.admitted_at, ordered))
                del self._items[completion.qid]
        return finished_items

    def _admit(self) -> None:
        while self._queue:
            state = self._items[self._queue[0]]
            while state.next_request < len(state.requests):
                request = state.requests[state.next_request]
                if not self.gate.try_admit(request.request_id, request.prompt_tokens):
                    return
                try:
                    self.decoder.submit(request)
                except Exception:
                    self.gate.release(request.request_id)
                    raise
                if state.admitted_at is None:
                    state.admitted_at = self.decoder.clock()
                    self.admission_trace[request.qid] = {
                        "reserved_tokens": self.gate.reserved_tokens,
                        "in_flight": self.decoder.in_flight(),
                    }
                state.next_request += 1
            self._queue.pop(0)

    def tracked(self) -> int:
        """Return the number of items awaiting full completion."""
        return len(self._items)

    def idle(self) -> bool:
        """Return whether no queued or admitted items remain."""
        return not self._queue and not self._items
