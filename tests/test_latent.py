"""The latent rollout: the embeds-prefix path every family's producer rolls."""

import pytest
import torch
from tests.conftest import random_ids

from rcc.latent.cache import cache_kv, cache_length
from rcc.latent.embeds import latent_rollout_embeds
from rcc.latent.rollout import build_realign, latent_rollout


def _embed(model, ids):
    with torch.no_grad():
        return model.get_input_embeddings()(ids)


def _greedy_decode(model, probe_ids, probe_mask, past, *, max_new_tokens):
    """Greedy-decode `max_new_tokens` ids on top of `past`, space-joined."""
    past_length = cache_length(past)
    total = past_length + int(probe_ids.shape[1])
    mask = torch.cat(
        [torch.ones(1, past_length, dtype=probe_mask.dtype, device=probe_ids.device), probe_mask],
        dim=1,
    )
    kwargs = {
        "input_ids": probe_ids,
        "attention_mask": mask,
        "past_key_values": past,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    with torch.no_grad():
        try:
            out = model.generate(
                **kwargs,
                cache_position=torch.arange(past_length, total, device=probe_ids.device),
            )
        except (TypeError, ValueError):
            out = model.generate(**kwargs)
    return " ".join(str(int(token)) for token in out[0, int(probe_ids.shape[1]) :])


def test_matches_ids_rollout_when_prefix_is_embedded_ids(tiny_model):
    prefix_ids = random_ids(14, seed=31)
    prompt_ids = random_ids(9, seed=32)
    realign = build_realign(tiny_model, enabled=False)
    ref = latent_rollout(
        tiny_model,
        torch.cat([prefix_ids, prompt_ids], dim=1),
        latent_steps=3,
        realign=realign,
        record_embeds=True,
    )
    got = latent_rollout_embeds(
        tiny_model,
        _embed(tiny_model, prefix_ids),
        prompt_ids,
        latent_steps=3,
        realign=realign,
        record_embeds=True,
    )
    assert cache_length(got.past) == cache_length(ref.past) == 14 + 9 + 3
    for (k0, v0), (k1, v1) in zip(cache_kv(ref.past), cache_kv(got.past), strict=True):
        assert torch.allclose(k0, k1, atol=1e-6)
        assert torch.allclose(v0, v1, atol=1e-6)
    assert ref.embeds is not None and got.embeds is not None
    assert torch.allclose(ref.embeds, got.embeds, atol=1e-6)


def test_kept_subset_prefix_rolls_at_dense_positions(tiny_model):
    """A kept-subset handoff occupies dense fresh positions.

    The cache equals an ids rollout over the same surviving rows, not over the
    original gapped sequence, because the embeds channel has no relocation.
    """
    source_ids = random_ids(20, seed=33)
    keep = [0, 2, 3, 9, 10, 11, 17]
    prompt_ids = random_ids(6, seed=34)
    realign = build_realign(tiny_model, enabled=False)
    kept_embeds = _embed(tiny_model, source_ids)[:, keep, :]
    got = latent_rollout_embeds(
        tiny_model, kept_embeds, prompt_ids, latent_steps=2, realign=realign
    )
    ref = latent_rollout(
        tiny_model,
        torch.cat([source_ids[:, keep], prompt_ids], dim=1),
        latent_steps=2,
        realign=realign,
    )
    for (k0, v0), (k1, v1) in zip(cache_kv(ref.past), cache_kv(got.past), strict=True):
        assert torch.allclose(k0, k1, atol=1e-6)
        assert torch.allclose(v0, v1, atol=1e-6)


def test_empty_prefix_reduces_to_plain_rollout(tiny_model):
    prompt_ids = random_ids(11, seed=35)
    realign = build_realign(tiny_model, enabled=False)
    empty = torch.zeros(1, 0, _embed(tiny_model, prompt_ids).shape[-1])
    got = latent_rollout_embeds(tiny_model, empty, prompt_ids, latent_steps=2, realign=realign)
    ref = latent_rollout(tiny_model, prompt_ids, latent_steps=2, realign=realign)
    for (k0, v0), (k1, v1) in zip(cache_kv(ref.past), cache_kv(got.past), strict=True):
        assert torch.allclose(k0, k1, atol=1e-6)
        assert torch.allclose(v0, v1, atol=1e-6)


def test_produced_cache_decodes_and_thought_embeds_recorded(tiny_model):
    prefix_ids = random_ids(8, seed=36)
    prompt_ids = random_ids(5, seed=37)
    realign = build_realign(tiny_model, enabled=False)
    out = latent_rollout_embeds(
        tiny_model,
        _embed(tiny_model, prefix_ids),
        prompt_ids,
        latent_steps=3,
        realign=realign,
        record_embeds=True,
    )
    assert out.embeds is not None and out.embeds.shape[1] == 8 + 5 + 3
    assert not torch.allclose(out.embeds[:, -1, :], _embed(tiny_model, prompt_ids)[:, -1, :]), (
        "thought rows must be realigned hidden states, not prompt embeddings"
    )
    probe = random_ids(4, seed=38)
    mask = torch.ones(1, 4, dtype=torch.long)
    decoded = _greedy_decode(tiny_model, probe, mask, out.past, max_new_tokens=6)
    assert len(decoded.split()) == 6


def test_rejects_batched_or_empty_prompts(tiny_model):
    realign = build_realign(tiny_model, enabled=False)
    good_prefix = _embed(tiny_model, random_ids(4, seed=39))
    with pytest.raises(ValueError, match="batch-1"):
        latent_rollout_embeds(
            tiny_model,
            good_prefix,
            torch.zeros(1, 0, dtype=torch.long),
            latent_steps=1,
            realign=realign,
        )
    with pytest.raises(ValueError, match="prefix_embeds"):
        latent_rollout_embeds(
            tiny_model,
            good_prefix[0],
            random_ids(3, seed=40),
            latent_steps=1,
            realign=realign,
        )
