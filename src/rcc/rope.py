"""RoPE relocation math, computed from stored rotary params (no live model handle).

A KV block captured at absolute positions p is moved to positions q by undoing
the source rotation and applying the target one, from `RopeParams` alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RopeParams:
    """The rotary parameters needed to recompute cos and sin without the model."""

    inv_freq: torch.Tensor
    attention_scaling: float = 1.0


def rope_cos_sin(
    rope_params: RopeParams, position_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin), each [batch, seq, head_dim], for the given positions."""
    if position_ids.ndim != 2:
        raise ValueError(f"position_ids must be [batch, seq], got {tuple(position_ids.shape)}")
    inv_freq = rope_params.inv_freq.to(position_ids.device)
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * rope_params.attention_scaling
    sin = emb.sin() * rope_params.attention_scaling
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dim by half: [x1, x2] -> [-x2, x1]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_k(k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to keys k [batch, kv_heads, seq, head_dim] with cos/sin [batch, seq, head_dim]."""
    cos = cos.unsqueeze(1).to(k.dtype)
    sin = sin.unsqueeze(1).to(k.dtype)
    return k * cos + _rotate_half(k) * sin


def unapply_rope_k(k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Invert apply_rope_k exactly, including any attention_scaling baked into cos/sin.

    Negating the sine alone gives `s * Rot(-theta)`, which composes to `s^2 * I`;
    dividing by `cos^2 + sin^2` restores the true inverse for every rope scaling type.
    """
    # Normalizing cos and sin, shaped [batch, seq, head_dim], rather than the
    # result costs kv_heads times fewer divisions.
    scale = cos * cos + sin * sin
    cos = (cos / scale).unsqueeze(1).to(k.dtype)
    sin = (sin / scale).unsqueeze(1).to(k.dtype)
    return k * cos - _rotate_half(k) * sin
