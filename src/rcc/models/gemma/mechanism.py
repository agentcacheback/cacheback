"""Gemma token, embedding, and resident-mechanism helpers.

The Gemma capture, selection, and receiver paths share them.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch


def encode(tokenizer: Any, text: str) -> list[int]:
    """Encode a text fragment without adding special tokens."""
    return [int(value) for value in tokenizer(text, add_special_tokens=False)["input_ids"]]


def single_token_id(tokenizer: Any, text: str) -> int:
    """Return the sole token id for a registered one-token fragment."""
    ids = encode(tokenizer, text)
    if len(ids) != 1:
        raise RuntimeError(f"expected {text!r} to be one token, got {ids}")
    return ids[0]


def channel_token_ids(tokenizer: Any) -> tuple[int, int]:
    """Return Gemma's thought-channel delimiter ids."""
    return single_token_id(tokenizer, "<|channel>"), single_token_id(tokenizer, "<channel|>")


def receiver_turn_ids(
    tokenizer: Any, *, enable_thinking: bool = False
) -> tuple[list[int], list[int]]:
    """Split the checkpoint's real chat template around a unique sentinel."""
    sentinel_text = "@@CONTENT@@"
    raw_ids = cast(
        object,
        tokenizer.apply_chat_template(
            [{"role": "user", "content": sentinel_text}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors=None,
            enable_thinking=enable_thinking,
        ),
    )
    if isinstance(raw_ids, Mapping):
        raw_ids = cast(Mapping[str, object], raw_ids).get("input_ids")
    if not isinstance(raw_ids, list):
        raise RuntimeError("chat template did not return token ids")
    values = cast(list[object], raw_ids)
    if values and isinstance(values[0], list):
        values = cast(list[object], values[0])
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        raise RuntimeError("chat template returned malformed token ids")
    ids = [cast(int, value) for value in values]
    sentinel = encode(tokenizer, sentinel_text)
    hits = [
        start
        for start in range(len(ids) - len(sentinel) + 1)
        if ids[start : start + len(sentinel)] == sentinel
    ]
    if len(hits) != 1:
        raise RuntimeError(f"chat template sentinel matched {len(hits)} times, expected 1")
    prefix, suffix = ids[: hits[0]], ids[hits[0] + len(sentinel) :]
    if tokenizer.bos_token_id is not None and tokenizer.bos_token_id not in prefix:
        raise RuntimeError("chat template prefix does not carry the BOS token")
    if not prefix or not suffix:
        raise RuntimeError("chat template split produced an empty prefix or suffix")
    return [int(value) for value in prefix], [int(value) for value in suffix]


def visible_answer_tokens(
    tokens: list[int],
    *,
    channel_open: int | None = None,
    channel_close: int | None = None,
    stop_ids: frozenset[int] = frozenset(),
) -> list[int]:
    """Reconstruct independently visible answer ids from raw decoder ids."""
    visible: list[int] = []
    in_channel = False
    for token in tokens:
        if token in stop_ids:
            break
        if channel_open is not None and token == channel_open:
            in_channel = True
            continue
        if channel_close is not None and token == channel_close:
            in_channel = False
            continue
        if not in_channel:
            visible.append(token)
    return visible


def protected_positions(length: int, protected_tail: int) -> tuple[int, ...]:
    """Protect sink row zero and every final latent-thought row."""
    tail_start = max(1, length - protected_tail)
    return (0, *range(tail_start, length))


def pin_selector_sink(scores: torch.Tensor) -> torch.Tensor:
    """Pin the sink column to the maximum score and return CPU float32."""
    values = scores.float()
    if values.numel() == 0:
        return values.cpu()
    pinned = values.clone()
    pinned[0] = values.max()
    return pinned.cpu()


def layer_bank_filename(qid: str, worker: int) -> str:
    """Name one durable per-worker all-layer capture bank."""
    if worker < 0:
        raise ValueError(f"worker must be nonnegative, got {worker}")
    return f"{qid}_w{worker}_layer_bank.pt"


def tensor_content_sha256(value: torch.Tensor) -> str:
    """Hash exact tensor metadata and contiguous bytes."""
    tensor = value.detach().cpu().contiguous()
    header = f"{tensor.dtype}|{tuple(tensor.shape)}|".encode()
    raw = tensor.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + raw).hexdigest()


def token_embeds(
    embedding_weight: torch.Tensor,
    token_ids: Sequence[int],
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Map ids through Gemma's measured sqrt(hidden) input-embedding scale."""
    if embedding_weight.ndim != 2:
        raise ValueError("embedding_weight must be [vocab, hidden]")
    index = torch.tensor(list(token_ids), dtype=torch.long, device=embedding_weight.device)
    selected = embedding_weight.index_select(0, index)
    scale = torch.tensor(
        math.sqrt(int(selected.shape[-1])),
        dtype=selected.dtype,
        device=selected.device,
    )
    return (selected * scale).to(device="cpu", dtype=dtype)


def token_digests(model: Any, token_ids: torch.Tensor) -> dict[str, Any]:
    """Prove assembled token rows match the model's Gemma input convention.

    Both sides run one op order on one device, so `bitwise_identical` is the
    normal case. See docs/running.md, Gemma embedding tolerance.
    """
    embedding = model.get_input_embeddings()
    weight = embedding.weight.detach()
    if weight.ndim != 2:
        raise RuntimeError("Gemma input embeddings must expose a [vocab, hidden] weight")
    ids = token_ids.to(device=weight.device, dtype=torch.long).reshape(-1)
    assembled = token_embeds(weight, [int(value) for value in ids], dtype=torch.bfloat16)
    with torch.no_grad():
        expected = embedding(ids).detach().to(device="cpu", dtype=torch.bfloat16)
    if assembled.shape != expected.shape:
        raise RuntimeError(
            f"G0c embedding digest shape differs: {tuple(assembled.shape)} "
            f"against the module's {tuple(expected.shape)}"
        )
    identical = bool(torch.equal(assembled, expected))
    delta = (assembled.float() - expected.float()).abs()
    max_abs_error = float(delta.max()) if delta.numel() else 0.0
    # |assembled - expected| <= atol + rtol * |expected|, both read off the
    # shipped dtype: rtol is bfloat16's own relative spacing (one ULP) and
    # atol its smallest normal, never a float32-grade constant on bf16 rows.
    rtol = float(torch.finfo(torch.bfloat16).eps)
    atol = float(torch.finfo(torch.bfloat16).tiny)
    # The banked margin is what decides: elementwise residue past its own
    # bound, positive only when an element exceeds one ULP of its magnitude.
    bound = atol + rtol * expected.float().abs()
    worst_margin = float((delta - bound).max()) if delta.numel() else 0.0
    passed = identical or worst_margin <= 0.0
    raw_norm = float(weight.index_select(0, ids).float().square().sum().sqrt())
    scale = 1.0 / raw_norm if raw_norm else 0.0
    observed_ratio = float(assembled.float().square().sum().sqrt()) * scale
    expected_ratio = float(expected.float().square().sum().sqrt()) * scale
    digests = {
        "family": "gemma",
        "hidden_size": int(weight.shape[-1]),
        "dtype": str(assembled.dtype),
        "expected_ratio": expected_ratio,
        "observed_ratio": observed_ratio,
        "relative_error": abs(observed_ratio - expected_ratio) / max(expected_ratio, 1e-12),
        "max_abs_error": max_abs_error,
        "worst_ulp_margin": worst_margin,
        "bitwise_identical": identical,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"G0c embedding digest check failed: {digests}")
    return digests


def require_rolling_cache(cache: Any, model: Any) -> None:
    """Refuse a capture cache that is not the exact hybrid rolling form."""
    from rcc.models.gemma.cache import RollingHybridCache

    if not isinstance(cache, RollingHybridCache):
        raise RuntimeError("Gemma capture did not retain a RollingHybridCache")
    schedule = list(model.config.layer_types)
    if len(cache.layers) != len(schedule):
        raise RuntimeError("Gemma rolling cache layer roster differs from the decoder")


__all__ = (
    "channel_token_ids",
    "encode",
    "layer_bank_filename",
    "pin_selector_sink",
    "protected_positions",
    "receiver_turn_ids",
    "require_rolling_cache",
    "tensor_content_sha256",
    "token_digests",
    "token_embeds",
    "visible_answer_tokens",
)
