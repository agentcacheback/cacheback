"""Receiver-conditioned state delivery without model loading or receiver generation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch

from rcc.latent import LatentContext, rollout, selected_state, validate_steps
from rcc.message import Message, Representation, encode_message
from rcc.selectors import Selector, cacheback


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
        if getattr(self.model.config, "model_type", None) != "qwen3":
            raise ValueError("this sender adapter supports dense Qwen3; other families use paper")
        rows = self.input_embeds
        weight = self.model.get_input_embeddings().weight
        if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] != weight.shape[1]:
            raise ValueError(
                "input_embeds must be the sender's exact [positions, hidden] input rows"
            )
        if rows.dtype != weight.dtype or rows.device != weight.device:
            raise ValueError("input_embeds must have the sender embedding table's dtype and device")
        if self.past_key_values is None or self.past_key_values.get_seq_length() != rows.shape[0]:
            raise ValueError("sender cache length differs from its recorded input rows")
        if self.model.training:
            raise ValueError("put the existing sender model in eval mode before transferring state")
        if (
            type(self.latent_steps) is not int
            or not 0 <= self.latent_steps < rows.shape[0]
            or type(self.inherited_positions) is not int
            or not 0 <= self.inherited_positions <= rows.shape[0] - self.latent_steps
        ):
            raise ValueError(
                "latent_steps and inherited_positions must be nonnegative integer counts; "
                "the inherited prefix cannot overlap the current latent tail"
            )
        if self.token_ids is not None:
            ids = self.token_ids
            if ids.ndim != 1 or ids.dtype != torch.long or ids.numel() != rows.shape[0]:
                raise ValueError(
                    "token_ids must be int64 IDs aligned to input_embeds, with -1 for latents"
                )
            invalid_id = bool(((ids < -1) | (ids >= weight.shape[0])).any())
            invalid_tail = self.latent_steps and not bool((ids[-self.latent_steps :] == -1).all())
            if invalid_id or invalid_tail:
                raise ValueError("token_ids must be valid source IDs, with -1 for the latent tail")

    def request_ids(self, request: object) -> torch.Tensor:
        """Tokenize a textual request or validate the caller's existing request IDs."""
        if isinstance(request, str):
            if not request.strip() or self.tokenizer is None:
                raise ValueError(
                    "a text request needs a nonempty string and the sender's tokenizer"
                )
            ids = cast(
                torch.Tensor,
                self.tokenizer(request, add_special_tokens=False, return_tensors="pt")["input_ids"],
            )
        else:
            ids = request
        if not isinstance(ids, torch.Tensor):
            raise ValueError("each request must be a string or an int64 token-ID tensor")
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0 or ids.dtype != torch.long:
            raise ValueError("request IDs must be a nonempty int64 tensor shaped [1, tokens]")
        if bool(((ids < 0) | (ids >= self.model.get_input_embeddings().weight.shape[0])).any()):
            raise ValueError("request contains an ID outside the sender's vocabulary")
        return ids.to(self.input_embeds.device)


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


def _prepare_message(
    sender: SenderState,
    request_ids: torch.Tensor,
    budget: int,
    representation: Representation,
    selector: Selector,
    latent_steps: int,
    latent_context: LatentContext,
) -> Message:
    """Run the chosen selector, validate its positions, then encode the selected input rows."""
    latent_request = (
        request_ids if latent_context in ("full_with_request", "selected_with_request") else None
    )
    from_selected = latent_context in ("selected", "selected_with_request")
    if latent_steps and not from_selected:
        sender = rollout(sender, steps=latent_steps, request=latent_request)
    selection_budget = budget - latent_steps if from_selected else budget
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
        if latent_steps and from_selected:
            grown = rollout(
                selected_state(sender, rows, selected_ids),
                steps=latent_steps,
                request=latent_request,
            )
            rows, selected_ids = grown.input_embeds, grown.token_ids
        return encode_message(rows, token_ids=selected_ids, representation=representation)


@dataclass(frozen=True)
class Delivery:
    """One receiver request and its selected messages, in sender-list order."""

    request: str | torch.Tensor
    messages: tuple[Message, ...]

    def _rows(self, embedding_weight: torch.Tensor) -> torch.Tensor:
        if not self.messages:
            raise ValueError("a delivery must contain at least one message")
        return torch.cat([message.materialize(embedding_weight) for message in self.messages])

    def for_hf(self, embedding_weight: torch.Tensor) -> dict[str, torch.Tensor]:
        """Prepare the handoff rows for Hugging Face; the agent owns its prompt and continuation."""
        rows = self._rows(embedding_weight).unsqueeze(0)
        return {
            "inputs_embeds": rows,
            "attention_mask": torch.ones(rows.shape[:2], dtype=torch.long, device=rows.device),
        }

    def for_vllm(self, embedding_weight: torch.Tensor) -> dict[str, torch.Tensor]:
        """Prepare the CPU prompt-embedding input used by vLLM receivers in the paper."""
        return {"prompt_embeds": self._rows(embedding_weight).detach().cpu()}


Receiver = Callable[[Delivery], object]


def _message_budget(
    sender: SenderState, ratio: int | None, budget: int | None, latent_steps: int
) -> int:
    """Apply the paper's relative handoff or bounded position count to one sender."""
    if ratio is not None and budget is not None:
        raise ValueError("choose ratio for relative compression or budget for a bounded message")
    length = sender.input_embeds.shape[0] + latent_steps
    if budget is not None:
        if type(budget) is not int or budget < 1:
            raise ValueError("budget must be a positive integer position cap")
        count = min(budget, length)
    else:
        ratio = 4 if ratio is None else ratio
        if type(ratio) is not int or ratio < 1:
            raise ValueError("ratio must be a positive integer compression factor")
        inherited = sender.inherited_positions
        count = inherited + (length - inherited + ratio - 1) // ratio
    return count


def transfer(
    sender: SenderState | Sequence[SenderState],
    receiver: Receiver | Sequence[Receiver],
    request: str | torch.Tensor | Sequence[str | torch.Tensor],
    *,
    ratio: int | None = None,
    budget: int | None = None,
    selector: Selector = cacheback,
    representation: Representation = "embeddings",
    latent_steps: int = 0,
    latent_context: LatentContext = "full",
) -> None:
    """Deliver every request to every receiver; default to CacheBack with relative r4 budgets."""
    senders = [sender] if isinstance(sender, SenderState) else list(sender)
    receivers = [receiver] if callable(receiver) else list(receiver)
    requests = [request] if isinstance(request, (str, torch.Tensor)) else list(request)
    _validate_inputs(senders, receivers, requests, selector, representation)
    _validate_latent_options(latent_steps, latent_context)
    budgets = [_message_budget(state, ratio, budget, latent_steps) for state in senders]
    if latent_context in ("selected", "selected_with_request") and any(
        count <= latent_steps for count in budgets
    ):
        raise ValueError("budget must leave room for selected context and the new latent rows")
    ids_by_request = [[state.request_ids(query) for state in senders] for query in requests]
    if latent_steps and latent_context == "full":
        senders = [rollout(state, steps=latent_steps) for state in senders]
        latent_steps = 0
    deliveries = [
        Delivery(
            query,
            tuple(
                _prepare_message(
                    state,
                    ids,
                    count,
                    representation,
                    selector,
                    latent_steps,
                    latent_context,
                )
                for state, ids, count in zip(senders, request_ids, budgets, strict=True)
            ),
        )
        for query, request_ids in zip(requests, ids_by_request, strict=True)
    ]
    for delivery in deliveries:
        for deliver in receivers:
            deliver(delivery)


def _validate_latent_options(steps: int, context: LatentContext) -> None:
    """Reject unknown latent settings even when additional rollout is disabled."""
    validate_steps(steps)
    if context not in ("full", "full_with_request", "selected", "selected_with_request"):
        raise ValueError(
            "latent_context must be full, full_with_request, selected or selected_with_request"
        )


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
