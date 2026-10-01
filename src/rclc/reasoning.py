"""Callable reasoning methods with bounded output and private sender state."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Literal, Protocol

import torch

from rclc._async import run_in_worker
from rclc._cache import cache_kv, clone_cache
from rclc.latent import latent_mass, latent_mass_sync

if TYPE_CHECKING:
    from rclc.transport import SenderState

ReasoningContext = Literal["full", "full_with_request", "selected", "selected_with_request"]
# Default assert_close tolerances, so recomputed prefixes may differ by kernel roundoff.
_PREFIX_TOLERANCE = {
    torch.float16: (1e-3, 1e-5),
    torch.bfloat16: (1.6e-2, 1e-5),
    torch.float32: (1.3e-6, 1e-5),
}


class Reasoning(Protocol):
    """Append continuous input rows using an algorithm-specific stopping rule."""

    def __call__(
        self, state: SenderState, *, max_positions: int, request: torch.Tensor | None = None
    ) -> SenderState | Awaitable[SenderState]:
        """Return consistent extended state within the supplied new-position allowance."""
        ...


def resolve_options(
    reasoning: Reasoning | None, budget: int | None, context: ReasoningContext | None
) -> ReasoningContext:
    """Reject ambiguous configuration before invoking a method."""
    if reasoning is None and (budget is not None or context is not None):
        raise ValueError("reasoning_budget and reasoning_context require a reasoning callable")
    context = "full" if context is None else context
    if context not in ("full", "full_with_request", "selected", "selected_with_request"):
        raise ValueError(
            "reasoning_context must be full, full_with_request, selected or selected_with_request"
        )
    if reasoning is not None and not callable(reasoning):
        raise ValueError("reasoning must be a callable")
    if budget is not None and (type(budget) is not int or budget < 0):
        raise ValueError("reasoning_budget must be a nonnegative position cap")
    if reasoning is not None and context.startswith("selected") and budget is None:
        raise ValueError("selected reasoning requires reasoning_budget to reserve output space")
    return context


def _validate_result(state: SenderState, result: SenderState, maximum: int) -> None:
    """Check append-only rows, metadata and cache geometry at the callable boundary."""
    from rclc.transport import SenderState

    if type(result) is not SenderState:
        raise ValueError("reasoning must return a SenderState")
    result.validate()
    length = state.input_embeds.shape[0]
    added = result.input_embeds.shape[0] - length
    if not 0 <= added <= maximum:
        raise ValueError("reasoning exceeded max_positions or removed source rows")
    if (
        result.model is not state.model
        or result.tokenizer is not state.tokenizer
        or result.inherited_positions != state.inherited_positions
        or result.latent_steps != state.latent_steps + added
        or not torch.equal(result.input_embeds[:length], state.input_embeds)
        or not bool(torch.isfinite(result.input_embeds[length:]).all())
    ):
        raise ValueError("reasoning must preserve the sender prefix and append finite latent rows")
    if state.token_ids is not None:
        if result.token_ids is None or not torch.equal(result.token_ids[:length], state.token_ids):
            raise ValueError("reasoning must preserve source token IDs")
    _validate_cache(state, result)


def _validate_cache(state: SenderState, result: SenderState) -> None:
    """Allow normal recomputation roundoff while rejecting changed cache prefixes."""
    length = len(state.input_embeds)
    before, after = cache_kv(state.past_key_values), cache_kv(result.past_key_values)
    if len(before) != len(after):
        raise ValueError("reasoning changed the cache layer count")
    valid: list[torch.Tensor] = []
    for original, updated in zip(before, after, strict=True):
        for old, new in zip(original, updated, strict=True):
            expected = (*old.shape[:2], len(result.input_embeds), old.shape[-1])
            if new.shape != expected or new.device != old.device or new.dtype != old.dtype:
                raise ValueError("reasoning must retain the cache geometry")
            rtol, atol = _PREFIX_TOLERANCE.get(old.dtype, (1e-7, 1e-7))
            prefix = torch.isclose(new[:, :, :length], old, rtol=rtol, atol=atol).all()
            valid.append(prefix & torch.isfinite(new[:, :, length:]).all())
    if valid and not bool(torch.stack([check.to(valid[0].device) for check in valid]).all()):
        raise ValueError("reasoning must retain the cache prefix and append finite KV rows")


async def apply_reasoning(
    state: SenderState,
    method: Reasoning,
    maximum: int | None,
    request: torch.Tensor | None = None,
    *,
    request_positions: int = 0,
) -> SenderState:
    """Run trusted caller code on private tensors; validate it before selection or delivery."""
    capacity = (
        int(state.model.config.max_position_embeddings)
        - len(state.input_embeds)
        - request_positions
    )
    allowance = capacity if maximum is None else min(maximum, capacity)
    if allowance < 0:
        raise ValueError("reasoning context and request exceed the model context limit")
    target: object = method
    builtin = (target.func if isinstance(target, partial) else target) in (
        latent_mass,
        latent_mass_sync,
    )
    private = state if builtin else await run_in_worker(_private_state, state)
    request = None if request is None else request.clone()
    if inspect.iscoroutinefunction(method) or inspect.iscoroutinefunction(method.__call__):
        result = method(private, max_positions=allowance, request=request)
    else:
        result = await run_in_worker(method, private, max_positions=allowance, request=request)
    if inspect.isawaitable(result):
        result = await result
    if not builtin:
        await run_in_worker(_validate_result, state, result, allowance)
    return result


def _private_state(state: SenderState) -> SenderState:
    """Give a custom reasoning method its own rows and cache to extend."""
    return replace(
        state,
        input_embeds=state.input_embeds.clone(),
        past_key_values=clone_cache(state.past_key_values),
        token_ids=None if state.token_ids is None else state.token_ids.clone(),
    )
