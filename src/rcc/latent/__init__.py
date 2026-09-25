"""Continuous latent rollout and the cache geometry it uses."""

from rcc.latent.cache import (
    BatchedPast,
    cache_kv,
    cache_length,
    cache_logical_length,
    compress_past,
    compress_past_rows,
    concat_caches,
    model_rope,
    pad_past_rows,
    split_past_rows,
)
from rcc.latent.embeds import latent_rollout_embeds
from rcc.latent.rollout import (
    Realign,
    RolloutOut,
    apply_realign,
    build_realign,
    latent_rollout,
    latent_rollout_batched,
)

__all__ = [
    "BatchedPast",
    "Realign",
    "RolloutOut",
    "apply_realign",
    "build_realign",
    "cache_kv",
    "cache_length",
    "cache_logical_length",
    "compress_past",
    "compress_past_rows",
    "concat_caches",
    "latent_rollout",
    "latent_rollout_batched",
    "latent_rollout_embeds",
    "model_rope",
    "pad_past_rows",
    "split_past_rows",
]
