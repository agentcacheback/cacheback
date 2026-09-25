"""Embedding-fed latent rollout: an inherited handoff at fresh dense positions.

The prefix embeddings are prefilled before the current prompt and its latent tail,
under the same realignment and norm matching as the token-fed rollout.
"""

from __future__ import annotations

from typing import Any, Protocol, cast

import torch

from rcc.latent.cache import cache_length
from rcc.latent.rollout import Realign, RolloutOut, apply_realign
from rcc.transforms.select.core.kernels import get_backbone


class _RolloutOutput(Protocol):
    """The decoder output fields the embedding-fed rollout reads."""

    past_key_values: Any
    last_hidden_state: torch.Tensor


def latent_rollout_embeds(
    model: Any,
    prefix_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    latent_steps: int,
    realign: Realign,
    record_embeds: bool = False,
) -> RolloutOut:
    """Prefill inherited embeds plus prompt ids, then append latent thoughts.

    ``prefix_embeds`` occupies fresh positions ``0..P-1`` and the prompt follows it
    directly. Batch-one, since right padding would replace a shorter latent seed.
    """
    if prefix_embeds.dim() != 3 or prefix_embeds.shape[0] != 1:
        raise ValueError(f"prefix_embeds must be [1, P, D], got {tuple(prefix_embeds.shape)}")
    if input_ids.dim() != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError(
            "latent_rollout_embeds is batch-1 with a non-empty prompt "
            f"(got ids shape {tuple(input_ids.shape)})"
        )
    device = input_ids.device
    with torch.no_grad():
        prompt_embeds = model.get_input_embeddings()(input_ids)
    seed = torch.cat([prefix_embeds.to(prompt_embeds), prompt_embeds], dim=1)
    total = int(seed.shape[1])
    position_ids = torch.arange(total, device=device).unsqueeze(0)
    mask = torch.ones(1, total, dtype=torch.long, device=device)
    backbone = get_backbone(model)
    with torch.no_grad():
        out = cast(
            _RolloutOutput,
            backbone(
                inputs_embeds=seed,
                past_key_values=None,
                position_ids=position_ids,
                attention_mask=mask,
                use_cache=True,
                return_dict=True,
            ),
        )
    past = out.past_key_values
    collected = [seed] if record_embeds else []
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
    if cache_length(past) != total + latent_steps:
        raise AssertionError("embeds rollout must grow the cache by prefix + prompt + thoughts")
    embeds = torch.cat(collected, dim=1) if record_embeds else None
    return RolloutOut(past=past, embeds=embeds)


__all__ = ["latent_rollout_embeds"]
