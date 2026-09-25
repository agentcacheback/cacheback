"""Continuous latent-thought rollout and norm-matched realignment.

For ``latent_steps`` the last hidden state is re-fed after a closed-form
least-squares map into input-embedding space, then matched to the mean row norm.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch

from rcc.latent.cache import BatchedPast, cache_logical_length
from rcc.transforms.select.core.kernels import get_backbone


class _RolloutOutput(Protocol):
    """The decoder output fields latent rollout reads."""

    past_key_values: Any
    last_hidden_state: torch.Tensor


@dataclass(frozen=True)
class Realign:
    """An output-to-input embedding map with a target row norm."""

    matrix: torch.Tensor
    target_norm: torch.Tensor
    enabled: bool


@dataclass
class RolloutOut:
    """A grown cache and, when requested, the embeddings fed to it."""

    past: Any
    embeds: torch.Tensor | None


def build_realign(model: Any, *, enabled: bool, reg: float = 1e-5) -> Realign:
    """Solve the norm-matched output-to-input embedding map for a model."""
    e_in_raw = model.get_input_embeddings().weight.detach()
    dim = e_in_raw.shape[1]
    target_norm = cast(
        torch.Tensor,
        torch.linalg.vector_norm(  # pyright: ignore[reportUnknownMemberType]
            e_in_raw,
            dim=-1,
            dtype=torch.float32,
        ).mean(),  # pyright: ignore[reportUnknownMemberType]
    )
    if enabled:
        e_in = e_in_raw.float()
        out_emb = model.get_output_embeddings()
        e_out = e_in if out_emb is None else out_emb.weight.detach().float()
        gram = e_out.t() @ e_out + reg * torch.eye(
            dim,
            dtype=e_out.dtype,
            device=e_out.device,
        )
        rhs = e_out.t() @ e_in
        matrix = cast(
            torch.Tensor,
            torch.linalg.solve(gram, rhs),  # pyright: ignore[reportUnknownMemberType]
        )
    else:
        matrix = torch.eye(dim, dtype=torch.float32, device=e_in_raw.device)
    return Realign(matrix=matrix, target_norm=target_norm, enabled=enabled)


def apply_realign(hidden: torch.Tensor, realign: Realign) -> torch.Tensor:
    """Map hidden rows through the realign matrix and match their norms."""
    mapped = hidden @ realign.matrix.to(hidden)
    norms = cast(
        torch.Tensor,
        mapped.norm(dim=-1, keepdim=True).clamp_min(  # pyright: ignore[reportUnknownMemberType]
            1e-6
        ),
    )
    return mapped / norms * realign.target_norm.to(hidden)


def latent_rollout(
    model: Any,
    input_ids: torch.Tensor,
    *,
    latent_steps: int,
    realign: Realign,
    past: Any = None,
    attention_mask: torch.Tensor | None = None,
    record_embeds: bool = False,
) -> RolloutOut:
    """Prefill a batch-one prompt and append norm-matched latent thoughts.

    Existing cache position comes from its logical cursor, not the physical layer-zero
    axis, which preserves Gemma's mixed sliding and global cache geometry.
    """
    if input_ids.shape[0] != 1:
        raise ValueError(
            "latent_rollout is batch-1 only (right padding corrupts the latent seed); "
            "run a left-padded batch as separate items instead"
        )
    device = input_ids.device
    prompt_len = int(input_ids.shape[1])
    past_len = cache_logical_length(past)
    total = past_len + prompt_len
    position_ids = torch.arange(past_len, total, device=device).unsqueeze(0)
    if attention_mask is None:
        mask = torch.ones(1, total, dtype=torch.long, device=device)
    else:
        pad = torch.ones(1, past_len, dtype=attention_mask.dtype, device=device)
        mask = torch.cat([pad, attention_mask], dim=1)
    backbone = get_backbone(model)
    with torch.no_grad():
        out = cast(
            _RolloutOutput,
            backbone(
                input_ids=input_ids,
                past_key_values=past,
                position_ids=position_ids,
                attention_mask=mask,
                use_cache=True,
                return_dict=True,
            ),
        )
    past = out.past_key_values
    collected: list[torch.Tensor] = []
    if record_embeds:
        with torch.no_grad():
            collected.append(model.get_input_embeddings()(input_ids))
    last_hidden = out.last_hidden_state[:, -1:, :]
    cur = total
    for _ in range(latent_steps):
        embed = apply_realign(last_hidden, realign)
        if record_embeds:
            collected.append(embed)
        position_ids = torch.tensor([[cur]], device=device)
        mask = torch.ones(1, cur + 1, dtype=torch.long, device=device)
        with torch.no_grad():
            out = cast(
                _RolloutOutput,
                backbone(
                    inputs_embeds=embed,
                    past_key_values=past,
                    position_ids=position_ids,
                    attention_mask=mask,
                    use_cache=True,
                    return_dict=True,
                ),
            )
        past = out.past_key_values
        last_hidden = out.last_hidden_state[:, -1:, :]
        cur += 1
    embeds = torch.cat(collected, dim=1) if record_embeds else None
    return RolloutOut(past=past, embeds=embeds)


def latent_rollout_batched(
    model: Any,
    prompts: Sequence[torch.Tensor],
    *,
    latent_steps: int,
    realign: Realign,
    state: BatchedPast | None = None,
    record_embeds: bool = False,
) -> tuple[BatchedPast, list[torch.Tensor] | None]:
    """Roll one left-padded sender segment for several items in one forward.

    Every item's live rows end at the segment edge and positions advance from its own
    live count. Full non-windowed attention only: pad columns would shift a window.
    """
    if not prompts:
        raise ValueError("batched rollout needs at least one prompt")
    for ids in prompts:
        if ids.dim() != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
            raise ValueError("each prompt must be a [1, L] id tensor with L >= 1")
    batch = len(prompts)
    device = prompts[0].device
    lengths = [int(prompt.shape[1]) for prompt in prompts]
    width = max(lengths)
    if state is None:
        past: Any = None
        previous_mask = torch.zeros(batch, 0, dtype=torch.long, device=device)
        live = [0] * batch
    else:
        if int(state.mask.shape[0]) != batch:
            raise ValueError("state batch size must match the prompt count")
        past, previous_mask, live = state.past, state.mask, list(state.live)
    ids = torch.zeros(batch, width, dtype=torch.long, device=device)
    segment_mask = torch.zeros(batch, width, dtype=torch.long, device=device)
    positions = torch.zeros(batch, width, dtype=torch.long, device=device)
    for row, prompt in enumerate(prompts):
        pad = width - lengths[row]
        ids[row, pad:] = prompt[0]
        segment_mask[row, pad:] = 1
        positions[row, pad:] = torch.arange(
            live[row],
            live[row] + lengths[row],
            device=device,
        )
    mask = torch.cat([previous_mask, segment_mask], dim=1)
    backbone = get_backbone(model)
    with torch.no_grad():
        out = cast(
            _RolloutOutput,
            backbone(
                input_ids=ids,
                past_key_values=past,
                position_ids=positions,
                attention_mask=mask,
                use_cache=True,
                return_dict=True,
            ),
        )
    past = out.past_key_values
    collected: list[list[torch.Tensor]] = [[] for _ in range(batch)]
    if record_embeds:
        with torch.no_grad():
            for row, prompt in enumerate(prompts):
                collected[row].append(model.get_input_embeddings()(prompt))
    last_hidden = out.last_hidden_state[:, -1:, :]
    for step in range(latent_steps):
        embed = apply_realign(last_hidden, realign)
        if record_embeds:
            for row in range(batch):
                collected[row].append(embed[row : row + 1])
        positions = torch.tensor(
            [[live[row] + lengths[row] + step] for row in range(batch)],
            device=device,
        )
        mask = torch.cat(
            [mask, torch.ones(batch, 1, dtype=mask.dtype, device=device)],
            dim=1,
        )
        with torch.no_grad():
            out = cast(
                _RolloutOutput,
                backbone(
                    inputs_embeds=embed,
                    past_key_values=past,
                    position_ids=positions,
                    attention_mask=mask,
                    use_cache=True,
                    return_dict=True,
                ),
            )
        past = out.past_key_values
        last_hidden = out.last_hidden_state[:, -1:, :]
    grown = [live[row] + lengths[row] + latent_steps for row in range(batch)]
    embeds = [torch.cat(parts, dim=1) for parts in collected] if record_embeds else None
    return BatchedPast(past=past, mask=mask, live=grown), embeds


__all__ = [
    "Realign",
    "RolloutOut",
    "apply_realign",
    "build_realign",
    "latent_rollout",
    "latent_rollout_batched",
]
