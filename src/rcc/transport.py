"""Receiver-conditioned state delivery without model loading or receiver generation."""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import torch

from rcc._async import run_in_worker
from rcc.latent import selected_state
from rcc.message import Message, Representation, encode_message
from rcc.reasoning import Reasoning, ReasoningContext, apply_reasoning, resolve_options
from rcc.selection import record_selection as capture_selection
from rcc.selection import selection_html
from rcc.selectors import Selector, cacheback, chunkkv, qsnap

if TYPE_CHECKING:
    from rcc.hf import Agent, HFReceiver, PreparedRequest

# Snapshots and selection serialize; use separate processes for model isolation.
_PREPARATION_LOCK = threading.Lock()


def validate_model(model: Any) -> None:
    """Reject unsupported model families and training mode before adapter work."""
    if getattr(model.config, "model_type", None) != "qwen3":
        raise ValueError("only dense Qwen3 models are supported")
    if model.training:
        raise ValueError("put the existing model in eval mode before binding or transferring it")


def token_ids(model: Any, tokenizer: Any, value: object, name: str) -> torch.Tensor:
    """Tokenize text or check caller IDs as nonempty in-vocabulary int64 [1, tokens]."""
    ids = value
    if isinstance(value, str):
        if tokenizer is None:
            raise ValueError(f"a text {name} needs a tokenizer; bind one or pass token IDs")
        ids = tokenizer(value, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if not isinstance(ids, torch.Tensor) or (
        ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0 or ids.dtype != torch.long
    ):
        raise ValueError(f"{name} must be text or nonempty int64 IDs shaped [1, tokens]")
    weight = model.get_input_embeddings().weight
    if bool(((ids < 0) | (ids >= weight.shape[0])).any()):
        raise ValueError(f"{name} contains an ID outside the model vocabulary")
    return ids.to(weight.device)


@dataclass
class SenderState:
    """A view of an existing Qwen3 sender, including the exact input rows behind its cache."""

    model: Any
    past_key_values: Any
    input_embeds: torch.Tensor
    tokenizer: Any = None
    token_ids: torch.Tensor | None = None
    latent_steps: int = 0
    inherited_positions: int = 0

    def validate(self) -> None:
        """Reject stale, unsupported or inconsistent state before capture."""
        validate_rows(self)
        if (
            self.past_key_values is None
            or self.past_key_values.get_seq_length() != self.input_embeds.shape[0]
        ):
            raise ValueError("sender cache length differs from its recorded input rows")

    def request_ids(self, request: object) -> torch.Tensor:
        """Tokenize a textual request or validate the caller's existing request IDs."""
        if isinstance(request, str) and not request.strip():
            raise ValueError("a text request must be a nonempty string")
        return token_ids(self.model, self.tokenizer, request, "request")


def validate_rows(state: SenderState) -> None:
    """Check a state's model, input rows, counts and token IDs; its cache is checked separately."""
    validate_model(state.model)
    rows, latent, inherited = state.input_embeds, state.latent_steps, state.inherited_positions
    weight = state.model.get_input_embeddings().weight
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] != weight.shape[1]:
        raise ValueError("input_embeds must be the sender's exact [positions, hidden] input rows")
    if rows.dtype != weight.dtype or rows.device != weight.device:
        raise ValueError("input_embeds must have the sender embedding table's dtype and device")
    if (
        type(latent) is not int
        or not 0 <= latent < rows.shape[0]
        or type(inherited) is not int
        or not 0 <= inherited <= rows.shape[0] - latent
    ):
        raise ValueError(
            "latent_steps and inherited_positions must be nonnegative integer counts; "
            "the inherited prefix cannot overlap the current latent tail"
        )
    ids = state.token_ids
    if ids is None:
        return
    if ids.ndim != 1 or ids.dtype != torch.long or ids.numel() != rows.shape[0]:
        raise ValueError("token_ids must be int64 IDs aligned to input_embeds, with -1 for latents")
    invalid_id = bool(((ids < -1) | (ids >= weight.shape[0])).any())
    if invalid_id or (latent and not bool((ids[-latent:] == -1).all())):
        raise ValueError("token_ids must be valid source IDs, with -1 for the latent tail")


def _selection_indices(
    selected: object, length: int, budget: int, device: torch.device
) -> torch.Tensor:
    """Validate selector output and place unique positions in source order."""
    if isinstance(selected, torch.Tensor):
        if selected.ndim != 1 or selected.dtype not in (torch.int32, torch.int64):
            raise ValueError("selector must return a one-dimensional int32 or int64 tensor")
    else:
        if not isinstance(selected, Sequence):
            raise ValueError("selector must return a sequence of integer positions")
        positions = cast(Sequence[object], selected)
        if not all(type(index) is int for index in positions):
            raise ValueError("selector must return a sequence of integer positions")
        selected = torch.tensor(positions, dtype=torch.long)
    if not 0 < selected.numel() <= budget:
        raise ValueError("selector must return at least one position without exceeding its budget")
    indices = selected.detach().to(device=device, dtype=torch.long).sort().values
    if bool(((indices < 0) | (indices >= length)).any()):
        raise ValueError("selector returned a position outside the sender state")
    if bool((indices[1:] == indices[:-1]).any()):
        raise ValueError("selector returned duplicate positions")
    return indices


def _select_rows(
    sender: SenderState,
    request_ids: torch.Tensor,
    selection_budget: int,
    representation: Representation,
    selector: Selector,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Select and validate input rows without holding a lock across caller awaits."""
    with _PREPARATION_LOCK:
        selected = selector(sender, request_ids, selection_budget)
        with torch.inference_mode():
            indices = _selection_indices(
                selected, sender.input_embeds.shape[0], selection_budget, sender.input_embeds.device
            )
            rows = sender.input_embeds.index_select(0, indices)
            ids = sender.token_ids
            selected_ids = None if ids is None else ids.index_select(0, indices.to(ids.device))
            if representation == "token_ids+continuous" and selected_ids is not None:
                discrete = selected_ids >= 0
                weight = sender.model.get_input_embeddings().weight
                token_rows = weight.index_select(0, selected_ids[discrete].to(weight.device))
                if not torch.equal(rows[discrete.to(rows.device)], token_rows):
                    raise ValueError(
                        "token_ids do not reproduce the sender's selected input embeddings"
                    )
        return rows, selected_ids, indices


async def _prepare_message(
    sender: SenderState,
    request_ids: torch.Tensor,
    budget: int,
    representation: Representation,
    selector: Selector,
    reserve: int,
    reasoning: Reasoning | None,
    request: torch.Tensor | None,
    record_selection: bool,
) -> Message:
    """Select within the budget, then await optional reasoning over the selection."""
    rows, selected_ids, indices = await run_in_worker(
        _select_rows, sender, request_ids, budget - reserve, representation, selector
    )
    if reasoning is not None:
        state = await run_in_worker(selected_state, sender, rows, selected_ids)
        grown = await apply_reasoning(
            state, reasoning, reserve, request, request_positions=request_ids.shape[1]
        )
        rows, selected_ids = grown.input_embeds, grown.token_ids

    def encode() -> Message:
        mixed = representation == "token_ids+continuous"
        message = encode_message(rows, selected_ids if mixed else None)
        if not record_selection:
            return message
        added = len(rows) - len(indices)
        return replace(message, selection=capture_selection(sender, indices, budget, added))

    return await run_in_worker(encode)


@dataclass(frozen=True)
class Delivery:
    """One receiver request and its selected messages, in sender-list order."""

    request: str | torch.Tensor
    messages: tuple[Message, ...]

    def __post_init__(self) -> None:
        """Reject an empty delivery before a receiver queues it."""
        if not self.messages:
            raise ValueError("a delivery must contain at least one message")

    def selection_html(self) -> str:
        """Render all recorded sender selections without materializing the payload."""
        request = self.request if isinstance(self.request, str) else "Supplied as token IDs"
        return selection_html([message.selection for message in self.messages], request)

    def for_hf(self, embedding_weight: torch.Tensor) -> dict[str, torch.Tensor]:
        """Prepare the handoff rows for Hugging Face; the agent owns its prompt and continuation."""
        rows = torch.cat([m.materialize(embedding_weight) for m in self.messages]).unsqueeze(0)
        return {
            "inputs_embeds": rows,
            "attention_mask": torch.ones(rows.shape[:2], dtype=torch.long, device=rows.device),
        }

    def for_vllm(self, embedding_weight: torch.Tensor) -> dict[str, torch.Tensor]:
        """Prepare the CPU prompt-embedding input for vLLM receivers."""
        weight = embedding_weight
        rows = [
            m.continuous_rows.to(device="cpu", dtype=weight.dtype)
            if m.token_ids is None
            and weight.is_floating_point()
            and weight.shape[1:] == m.continuous_rows.shape[1:]
            else m.materialize(weight).cpu()
            for m in self.messages
        ]
        return {"prompt_embeds": torch.cat(rows).detach()}


Receiver = Callable[[Delivery], object]


def _check_size(ratio: int | None, budget: int | None) -> None:
    """Validate the relative or bounded handoff size before any model work."""
    if ratio is not None and budget is not None:
        raise ValueError("choose ratio for relative compression or budget for a bounded message")
    if budget is not None and (type(budget) is not int or budget < 1):
        raise ValueError("budget must be a positive integer position cap")
    if ratio is not None and (type(ratio) is not int or ratio < 1):
        raise ValueError("ratio must be a positive integer compression factor")


def _message_budget(
    sender: SenderState, ratio: int | None, budget: int | None, reserve: int
) -> int:
    """Apply the relative handoff (default ratio 4) or the bounded position count to one sender."""
    length = sender.input_embeds.shape[0] + reserve
    if budget is not None:
        return min(budget, length)
    step = 4 if ratio is None else ratio
    inherited = sender.inherited_positions
    return inherited + (length - inherited + step - 1) // step


def _receiver_plans(
    senders: list[SenderState], receivers: list[Receiver], requests: list[str | torch.Tensor]
) -> list[list[PreparedRequest | None]]:
    """Snapshot configured prompts and validate every receiver before selecting anything."""
    from rcc.hf import HFReceiver

    for receiver in receivers:
        if isinstance(receiver, HFReceiver):
            for sender in senders:
                receiver.validate_sender(sender)
    return [
        [
            receiver.prepare_request(request) if isinstance(receiver, HFReceiver) else None
            for receiver in receivers
        ]
        for request in requests
    ]


def _fit_budgets(
    caps: list[int], floors: list[int], plans: list[PreparedRequest | None]
) -> list[int]:
    """Share available receiver space after reserving each sender's mandatory positions."""
    if any(cap < floor for cap, floor in zip(caps, floors, strict=True)):
        raise ValueError(
            "budget must cover each selector's required positions and reserved reasoning rows"
        )
    limits = [plan.available_positions for plan in plans if plan is not None]
    available = min(limits, default=sum(caps))
    if sum(caps) <= available:
        return caps
    counts = floors.copy()
    if sum(counts) > available:
        raise ValueError(
            f"receiver has {available} handoff positions but needs at least {sum(counts)}; "
            "shorten its history, reserve fewer output tokens, use fewer senders or reason less"
        )
    remaining = available - sum(counts)
    active: list[int] = [i for i, cap in enumerate(caps) if counts[i] < cap]
    while remaining and active:
        share = max(1, remaining // len(active))
        for index in active:
            take = min(share, caps[index] - counts[index], remaining)
            counts[index] += take
            remaining -= take
        active = [i for i in active if counts[i] < caps[i]]
    return counts


def transfer_sync(
    sender: SenderState | Agent | Sequence[SenderState | Agent],
    receiver: Receiver | Sequence[Receiver],
    request: str | torch.Tensor | Sequence[str | torch.Tensor],
    *,
    ratio: int | None = None,
    budget: int | None = None,
    selector: Selector = cacheback,
    representation: Representation = "embeddings",
    reasoning: Reasoning | None = None,
    reasoning_context: ReasoningContext | None = None,
    reasoning_budget: int | None = None,
    record_selection: bool = False,
) -> None:
    """Run the async handoff to completion from code without a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("use await rcc.transfer(...) inside a running event loop")
    return asyncio.run(
        transfer(
            sender,
            receiver,
            request,
            ratio=ratio,
            budget=budget,
            selector=selector,
            representation=representation,
            reasoning=reasoning,
            reasoning_context=reasoning_context,
            reasoning_budget=reasoning_budget,
            record_selection=record_selection,
        )
    )


async def transfer(
    sender: SenderState | Agent | Sequence[SenderState | Agent],
    receiver: Receiver | Sequence[Receiver],
    request: str | torch.Tensor | Sequence[str | torch.Tensor],
    *,
    ratio: int | None = None,
    budget: int | None = None,
    selector: Selector = cacheback,
    representation: Representation = "embeddings",
    reasoning: Reasoning | None = None,
    reasoning_context: ReasoningContext | None = None,
    reasoning_budget: int | None = None,
    record_selection: bool = False,
) -> None:
    """Prepare off the event loop, then deliver in order, awaiting async receivers."""
    context = resolve_options(reasoning, reasoning_budget, reasoning_context)
    _check_size(ratio, budget)
    senders, receivers, requests, plans = await run_in_worker(
        _snapshot_transfer, sender, receiver, request, selector, representation
    )
    deliveries = await _deliveries(
        senders,
        requests,
        plans,
        ratio,
        budget,
        selector,
        representation,
        reasoning,
        context,
        reasoning_budget,
        record_selection,
    )
    for delivery, planned in zip(deliveries, plans, strict=True):
        for deliver, prepared in zip(receivers, planned, strict=True):
            if prepared is not None:
                cast("HFReceiver", deliver).enqueue(delivery, prepared)
            else:
                result = deliver(delivery)
                if inspect.isawaitable(result):
                    await result


def _snapshot_transfer(
    sender: SenderState | Agent | Sequence[SenderState | Agent],
    receiver: Receiver | Sequence[Receiver],
    request: str | torch.Tensor | Sequence[str | torch.Tensor],
    selector: Selector,
    representation: Representation,
) -> tuple[
    list[SenderState],
    list[Receiver],
    list[str | torch.Tensor],
    list[list[PreparedRequest | None]],
]:
    """Snapshot participating state and receiver prompts before reasoning or selection."""
    from rcc.hf import Agent

    with _PREPARATION_LOCK:
        sources = [sender] if isinstance(sender, (SenderState, Agent)) else list(sender)
        senders = [
            source.sender_state() if isinstance(source, Agent) else source for source in sources
        ]
        receivers = [receiver] if callable(receiver) else list(receiver)
        requests = [request] if isinstance(request, (str, torch.Tensor)) else list(request)
        _validate_inputs(senders, receivers, requests, selector, representation)
        plans = _receiver_plans(senders, receivers, requests)
        return senders, receivers, requests, plans


def _is_builtin_selector(selector: Selector) -> bool:
    """Apply the protected-position floor only to the built-in selectors and their partials."""
    return (selector.func if isinstance(selector, partial) else selector) in (
        cacheback,
        qsnap,
        chunkkv,
    )


async def _deliveries(
    senders: list[SenderState],
    requests: list[str | torch.Tensor],
    plans: list[list[PreparedRequest | None]],
    ratio: int | None,
    budget: int | None,
    selector: Selector,
    representation: Representation,
    reasoning: Reasoning | None,
    context: ReasoningContext,
    maximum: int | None,
    record_selection: bool,
) -> list[Delivery]:
    """Fit actual pre-selection output or reserve bounded post-selection output."""
    selected = reasoning is not None and context in ("selected", "selected_with_request")
    full = reasoning is not None and not selected
    reserve = (maximum or 0) if selected else 0
    protected = _is_builtin_selector(selector)

    def fit(states: list[SenderState], planned: list[PreparedRequest | None]) -> list[int]:
        caps = [_message_budget(s, ratio, budget, reserve) for s in states]
        floors = [(s.latent_steps if protected else 0) + reserve + 1 for s in states]
        return _fit_budgets(caps, floors, planned)

    fitted = [] if full else [fit(senders, planned) for planned in plans]
    ids = await run_in_worker(lambda: [[s.request_ids(q) for s in senders] for q in requests])
    if reasoning is not None and context == "full":
        senders = [
            await apply_reasoning(
                s, reasoning, maximum, request_positions=max(q[i].shape[1] for q in ids)
            )
            for i, s in enumerate(senders)
        ]
    conditioned = context.endswith("_with_request")
    deliveries: list[Delivery] = []
    for index, (query, tokens, planned) in enumerate(zip(requests, ids, plans, strict=True)):
        states = senders
        if reasoning is not None and context == "full_with_request":
            states = [
                await apply_reasoning(s, reasoning, maximum, q, request_positions=q.shape[1])
                for s, q in zip(senders, tokens, strict=True)
            ]
        counts = fit(states, planned) if full else fitted[index]
        messages = [
            await _prepare_message(
                s,
                q,
                count,
                representation,
                selector,
                reserve,
                reasoning if selected else None,
                q if conditioned else None,
                record_selection,
            )
            for s, q, count in zip(states, tokens, counts, strict=True)
        ]
        deliveries.append(Delivery(query, tuple(messages)))
    return deliveries


def _validate_inputs(
    senders: list[SenderState],
    receivers: list[Receiver],
    requests: list[str | torch.Tensor],
    selector: Selector,
    representation: Representation,
) -> None:
    """Validate routing, encoding and existing state before doing model work."""
    if not senders or not all(type(state) is SenderState for state in senders):
        raise ValueError("sender must contain at least one SenderState")
    if not receivers or not all(callable(deliver) for deliver in receivers):
        raise ValueError("receiver must contain at least one delivery callback")
    if not callable(selector):
        raise ValueError("selector must be a callable")
    if not requests:
        raise ValueError("provide at least one request")
    if representation not in ("embeddings", "token_ids+continuous"):
        raise ValueError(f"unknown message representation: {representation}")
    for state in senders:
        state.validate()
        if representation == "token_ids+continuous" and state.token_ids is None:
            raise ValueError("mixed representation requires every sender's aligned token_ids")
