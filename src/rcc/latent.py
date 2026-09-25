"""Opt-in continuous rollout using the paper's Qwen norm-matching rule."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, cast

import torch

from rcc.selectors.core.kernels import ATTENTION_LOCK, clone_cache, get_backbone

if TYPE_CHECKING:
    from rcc.transport import SenderState

LatentContext = Literal["full", "full_with_request", "selected", "selected_with_request"]


def validate_steps(steps: int) -> None:
    """Require an explicit nonnegative count, excluding booleans."""
    if type(steps) is not int or steps < 0:
        raise ValueError("latent steps must be a nonnegative integer")


def _forward(model: Any, rows: torch.Tensor, past: Any, start: int) -> Any:
    """Append exact input rows at dense positions to a private cache."""
    stop = start + rows.shape[1]
    with ATTENTION_LOCK:
        return get_backbone(model)(
            inputs_embeds=rows,
            past_key_values=past,
            position_ids=torch.arange(start, stop, device=rows.device)[None],
            attention_mask=torch.ones((1, stop), dtype=torch.long, device=rows.device),
            use_cache=True,
            return_dict=True,
        )


@torch.inference_mode()
def rollout(
    state: SenderState, *, steps: int, request: str | torch.Tensor | None = None
) -> SenderState:
    """Return extended state without changing the agent; optional request rows are not retained."""
    validate_steps(steps)
    state.validate()
    ids = None if request is None else state.request_ids(request)
    if steps == 0:
        return state
    length = state.input_embeds.shape[0]
    request_length = 0 if ids is None else ids.shape[1]
    if length + request_length + steps > state.model.config.max_position_embeddings:
        raise ValueError("latent rollout exceeds the model's context limit")
    past = cast(Any, clone_cache(state.past_key_values))
    if ids is None:
        past.crop(length - 1)
        seed = state.input_embeds[-1:][None]
        start = length - 1
    else:
        seed = state.model.get_input_embeddings()(ids)
        start = length
    output = _forward(state.model, seed, past, start)
    weight = state.model.get_input_embeddings().weight.detach()
    target_norm = cast(
        torch.Tensor,
        torch.linalg.vector_norm(weight, dim=-1, dtype=torch.float32).mean(),  # pyright: ignore[reportUnknownMemberType]
    )
    thoughts: list[torch.Tensor] = []
    for index in range(steps):
        hidden = cast(torch.Tensor, output.last_hidden_state[:, -1:])
        norm = cast(torch.Tensor, hidden.norm(dim=-1, keepdim=True).clamp_min(1e-6))  # pyright: ignore[reportUnknownMemberType]
        thought = hidden / norm * target_norm.to(hidden)
        thoughts.append(thought)
        output = _forward(
            state.model, thought, output.past_key_values, length + request_length + index
        )
    added = torch.cat(thoughts, dim=1)
    if ids is not None:
        # Replay only the latent rows so the returned cache excludes conditioning tokens.
        del output, past
        output = _forward(state.model, added, clone_cache(state.past_key_values), length)
    token_ids = state.token_ids
    if token_ids is not None:
        token_ids = torch.cat((token_ids, token_ids.new_full((steps,), -1)))
    return replace(
        state,
        past_key_values=output.past_key_values,
        input_embeds=torch.cat((state.input_embeds, added[0])),
        token_ids=token_ids,
        latent_steps=state.latent_steps + steps,
    )


@torch.inference_mode()
def selected_state(
    state: SenderState, rows: torch.Tensor, token_ids: torch.Tensor | None
) -> SenderState:
    """Prefill selected rows as a fresh rollout prompt at dense positions."""
    if rows.shape[0] > state.model.config.max_position_embeddings:
        raise ValueError("selected context exceeds the model's context limit")
    output = _forward(state.model, rows[None], None, 0)
    return replace(
        state,
        past_key_values=output.past_key_values,
        input_embeds=rows,
        token_ids=token_ids,
        latent_steps=0,
        inherited_positions=0,
    )
