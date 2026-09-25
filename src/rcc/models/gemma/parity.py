"""Exact identity helpers for Gemma's materialized Qwen-flat handoff."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import torch

from rcc.models.gemma.mechanism import tensor_content_sha256

QWEN_FLAT_PAYLOAD_IDENTITY_SCHEMA = "gemma-qwen-flat-payload-identity-v1"


def selected_indices_sha256(keeps: Sequence[Sequence[int]]) -> str:
    """Hash the complete worker-major selected-index order without truncation."""
    payload = [[int(position) for position in worker] for worker in keeps]
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def qwen_flat_payload_identity(blocks: Sequence[torch.Tensor]) -> dict[str, object]:
    """Hash the exact BF16 worker blocks and their materialized concatenation."""
    materialized = tuple(block.detach().cpu().contiguous() for block in blocks)
    if len(materialized) != 3:
        raise RuntimeError("Gemma Qwen-flat payload must contain exactly three worker blocks")
    if any(block.ndim != 2 or block.dtype != torch.bfloat16 for block in materialized):
        raise RuntimeError("Gemma Qwen-flat payload blocks must be rank-two bfloat16 tensors")
    hidden = {int(block.shape[1]) for block in materialized}
    if len(hidden) != 1:
        raise RuntimeError("Gemma Qwen-flat payload worker widths differ")
    combined = torch.cat(materialized, dim=0)
    return {
        "schema": QWEN_FLAT_PAYLOAD_IDENTITY_SCHEMA,
        "dtype": "bfloat16",
        "rows_by_worker": [int(block.shape[0]) for block in materialized],
        "worker_sha256": [tensor_content_sha256(block) for block in materialized],
        "payload_sha256": tensor_content_sha256(combined),
    }


__all__ = (
    "QWEN_FLAT_PAYLOAD_IDENTITY_SCHEMA",
    "qwen_flat_payload_identity",
    "selected_indices_sha256",
)
