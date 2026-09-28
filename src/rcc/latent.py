"""Opt-in continuous rollout using the paper's Qwen norm-matching rule."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import torch

from rcc._async import run_in_worker
from rcc._cache import ATTENTION_LOCK, cache_kv, clone_cache, get_backbone

if TYPE_CHECKING:
    from rcc.transport import SenderState


def _validate_count(value: int, name: str) -> None:
    """Require an explicit nonnegative count, excluding booleans."""
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _target_norm(weight: torch.Tensor) -> torch.Tensor:
    """Return the mean embedding norm; recomputed each call so weight updates are seen."""
    return cast(
        torch.Tensor,
        torch.linalg.vector_norm(weight.detach(), dim=-1, dtype=torch.float32).mean(),  # pyright: ignore[reportUnknownMemberType]
    )


def forward_rows(model: Any, rows: torch.Tensor, past: Any, start: int) -> Any:
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


def _seed_output(state: SenderState, ids: torch.Tensor | None) -> Any:
    """Read the rollout seed without replacing any existing KV positions."""
    length = state.input_embeds.shape[0]
    past = cast(Any, clone_cache(state.past_key_values))
    if ids is not None:
        return forward_rows(state.model, state.model.get_input_embeddings()(ids), past, length)
    past.crop(length - 1)
    output = forward_rows(state.model, state.input_embeds[-1:][None], past, length - 1)
    # Reuse the original seed KV; recomputing one token can round differently.
    for original, updated in zip(
        cache_kv(state.past_key_values), cache_kv(output.past_key_values), strict=True
    ):
        for source, target in zip(original, updated, strict=True):
            target[:, :, -1:].copy_(source[:, :, -1:])
    return output


@torch.inference_mode()
def latent_mass_sync(
    state: SenderState,
    *,
    steps: int,
    request: str | torch.Tensor | None = None,
    max_positions: int | None = None,
) -> SenderState:
    """Return extended state without changing the agent; optional request rows are not retained."""
    _validate_count(steps, "latent steps")
    if max_positions is not None:
        _validate_count(max_positions, "max_positions")
        if steps > max_positions:
            raise ValueError("latent steps exceed max_positions")
    state.validate()
    ids = None if request is None else state.request_ids(request)
    if steps == 0:
        return state
    length = state.input_embeds.shape[0]
    request_length = 0 if ids is None else ids.shape[1]
    if length + request_length + steps > state.model.config.max_position_embeddings:
        raise ValueError("latent rollout exceeds the model's context limit")
    output = _seed_output(state, ids)
    target_norm = _target_norm(state.model.get_input_embeddings().weight).to(
        output.last_hidden_state
    )
    thoughts: list[torch.Tensor] = []
    for index in range(steps):
        hidden = cast(torch.Tensor, output.last_hidden_state[:, -1:])
        norm = cast(torch.Tensor, hidden.norm(dim=-1, keepdim=True).clamp_min(1e-6))  # pyright: ignore[reportUnknownMemberType]
        thought = hidden / norm * target_norm
        thoughts.append(thought)
        output = forward_rows(
            state.model, thought, output.past_key_values, length + request_length + index
        )
    added = torch.cat(thoughts, dim=1)
    if ids is not None:
        # Replay only the latent rows so the returned cache excludes conditioning tokens.
        del output
        output = forward_rows(state.model, added, clone_cache(state.past_key_values), length)
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


async def latent_mass(
    state: SenderState,
    *,
    steps: int,
    request: str | torch.Tensor | None = None,
    max_positions: int | None = None,
) -> SenderState:
    """Await a private latent rollout without blocking the caller's event loop."""
    return await run_in_worker(
        latent_mass_sync, state, steps=steps, request=request, max_positions=max_positions
    )


@torch.inference_mode()
def selected_state(
    state: SenderState, rows: torch.Tensor, token_ids: torch.Tensor | None
) -> SenderState:
    """Prefill selected rows as a fresh rollout prompt at dense positions."""
    if rows.shape[0] > state.model.config.max_position_embeddings:
        raise ValueError("selected context exceeds the model's context limit")
    output = forward_rows(state.model, rows[None], None, 0)
    return replace(
        state,
        past_key_values=output.past_key_values,
        input_embeds=rows,
        token_ids=token_ids,
        latent_steps=0,
        inherited_positions=0,
    )
