"""Gemma latent updates normalized in the native token-embedding space."""

from __future__ import annotations

import math
from typing import Any

import torch

from rcc.latent.rollout import Realign, build_realign
from rcc.models.gemma.contract import LATENT_REALIGN_ENABLED


def build_native_realign(model: Any) -> Realign:
    """Match every recurrent input to Gemma's scaled embedding mean norm."""
    base = build_realign(model, enabled=LATENT_REALIGN_ENABLED)
    embedding = model.get_input_embeddings()
    # Gemma casts its scale to the table dtype before multiplying token rows.
    # Derive that value from the live model; do not hard-code sqrt(hidden).
    scale = float(embedding.embed_scale.to(embedding.weight.dtype))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Gemma native embedding scale must be finite and positive")
    if not bool(torch.isfinite(base.target_norm)) or float(base.target_norm) <= 0:
        raise ValueError("Gemma raw embedding mean norm must be finite and positive")
    return Realign(base.matrix, base.target_norm * scale, base.enabled)
