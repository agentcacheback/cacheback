"""The batched query-support capture: one scored forward over a padded batch.

The registered attention function returns the model's own output and, on the side,
reduces each item's question rows. The caller owns the rollout and the padding.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch

from rcc.transforms.select.core import kernels
from rcc.transforms.select.core.types import (
    BatchedPastLike,
    EnergyAccumulator,
    TokenIds,
)
from rcc.transforms.select.query_support import capture_single
from rcc.transforms.select.query_support.methods import reducers

_CAPTURE_IMPL = "rcc_vote_capture_batched"

_flex_compiled: Any = None

try:
    # torch >= 2.5. Without these the Flex path is unavailable and the capture
    # runs dense.
    from torch.nn.attention.flex_attention import create_block_mask as _create_block_mask
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention
except Exception:  # pragma: no cover - depends on the installed torch
    _create_block_mask = None
    _flex_attention = None


class _ModelWithConfig(Protocol):
    """The model surface needed to restore an attention implementation."""

    config: object


@dataclass
class BatchedCaptureContext:
    """The state one batched capture shares with its attention function."""

    los: list[int]
    his: list[int]
    width: int
    bias: torch.Tensor | None
    votes: list[torch.Tensor | None]
    live_indices: list[torch.Tensor]
    energies: list[EnergyAccumulator] | None = None
    heads: int = 1
    use_flex: bool = False
    block_mask: Any = None
    span_biases: list[torch.Tensor] | None = None


@dataclass(frozen=True)
class BatchedCaptureItem:
    """One item's capture state, unpooled, for the caller to convert."""

    mem_len: int
    found: bool
    live_indices: torch.Tensor
    votes: torch.Tensor | None
    heads: int
    question_rows: int
    energy: EnergyAccumulator | None = None


_context: BatchedCaptureContext | None = None
_registered = False

__all__ = [
    "BatchedCaptureContext",
    "BatchedCaptureItem",
    "batched_bias",
    "capture_batched",
    "current_context",
    "flex_mask_fn",
    "span_bias_item",
]


def current_context() -> BatchedCaptureContext | None:
    """Return the batched capture context that is active, or None."""
    return _context


def _flex() -> Any:
    """Return the compiled FlexAttention entry point, compiled once on first use."""
    global _flex_compiled
    if _flex_compiled is None:
        compile_fn = cast(Any, torch).compile
        _flex_compiled = compile_fn(_flex_attention, dynamic=True)
    return _flex_compiled


def batched_bias(
    col_mask: torch.Tensor,
    q_row_mask: torch.Tensor,
    width: int,
    q_width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build the additive pad-plus-causal output mask for one padded batch."""
    neg = torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=device)
    zero = torch.zeros((), dtype=dtype, device=device)
    mask = torch.zeros(col_mask.shape[0], 1, q_width, width + q_width, dtype=dtype, device=device)
    mask[:, 0, :, :width] = torch.where((col_mask == 0).unsqueeze(1), neg, zero)
    rows = torch.arange(q_width, device=device).unsqueeze(1)
    cols = torch.arange(q_width, device=device).unsqueeze(0)
    causal = (cols > rows).unsqueeze(0)
    pad_row = (q_row_mask == 0).unsqueeze(1)
    mask[:, 0, :, width:] = torch.where(causal | pad_row, neg, zero)
    return mask


def flex_mask_fn(col_ok: torch.Tensor, row_ok: torch.Tensor, width: int) -> Any:
    """Return the FlexAttention predicate that matches :func:`batched_bias`."""

    def mask_mod(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        del h
        in_mem = kv_idx < width
        mem_ok = in_mem & col_ok[b, torch.clamp(kv_idx, max=width - 1)]
        wcol = torch.clamp(kv_idx - width, min=0)
        win_ok = (~in_mem) & row_ok[b, wcol] & (kv_idx - width <= q_idx)
        return mem_ok | win_ok

    return mask_mod


def span_bias_item(
    col_row: torch.Tensor,
    row_ok_row: torch.Tensor,
    width: int,
    lo: int,
    hi: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build one item's additive bias over its question rows, for the Flex path."""
    q_width = int(row_ok_row.shape[0])
    neg = torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=device)
    zero = torch.zeros((), dtype=dtype, device=device)
    bias = torch.zeros(hi - lo, width + q_width, dtype=dtype, device=device)
    bias[:, :width] = torch.where((col_row == 0).unsqueeze(0), neg, zero)
    rows = torch.arange(lo, hi, device=device).unsqueeze(1)
    cols = torch.arange(q_width, device=device).unsqueeze(0)
    pad_col = (row_ok_row == 0).unsqueeze(0)
    bias[:, width:] = torch.where((cols > rows) | pad_col, neg, zero)
    return bias.view(1, 1, hi - lo, width + q_width)


def _capture_batched_attn(
    module: object,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: object,
) -> tuple[torch.Tensor, None]:
    """Return the attention output and reduce each item's question rows on the side."""
    del attention_mask, dropout, kwargs
    context = _context
    if context is None:
        raise RuntimeError("batched vote capture invoked with no active context")
    repeats = int(getattr(module, "num_key_value_groups", 1))
    full_key = kernels.repeat_kv(key, repeats)
    full_value = kernels.repeat_kv(value, repeats)
    if context.use_flex:
        output = _flex()(
            query,
            full_key,
            full_value,
            block_mask=context.block_mask,
            scale=scaling,
        )
    else:
        if context.bias is None:
            raise RuntimeError("dense batched capture has no output attention bias")
        output = torch.nn.functional.scaled_dot_product_attention(
            query,
            full_key,
            full_value,
            attn_mask=context.bias.to(query.dtype),
            dropout_p=0.0,
            scale=scaling,
        )
    output = output.transpose(1, 2).contiguous()
    for index, (lo, hi) in enumerate(zip(context.los, context.his, strict=True)):
        if hi <= lo:
            continue
        if context.span_biases is not None:
            span_bias = context.span_biases[index]
        else:
            if context.bias is None:
                raise RuntimeError("dense batched capture has no skinny attention bias")
            span_bias = context.bias[index : index + 1, :, lo:hi, :]
        question = query[index : index + 1, :, lo:hi, :]
        logits = (
            torch.matmul(
                question,
                full_key[index : index + 1].transpose(2, 3),
            )
            * scaling
        )
        logits = logits + span_bias.to(logits.dtype)
        weights = torch.softmax(logits, dim=-1, dtype=torch.float32)
        contribution = weights[0, :, :, : context.width].sum(dim=(0, 1))
        prior = context.votes[index]
        context.votes[index] = contribution if prior is None else prior + contribution
        if context.energies is not None:
            logical_memory = weights[0, :, :, : context.width].index_select(
                2,
                context.live_indices[index],
            )
            reducers.accumulate_capture_energies(
                context.energies[index],
                logical_memory,
                kv_groups=repeats,
            )
    context.heads = int(query.shape[1])
    return output, None


def _register_capture() -> None:
    """Register the batched capture attention function with Transformers, once."""
    global _registered
    if _registered:
        return
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    register = getattr(ALL_ATTENTION_FUNCTIONS, "register", None)
    if not callable(register):
        raise RuntimeError("Transformers attention registry has no register method")
    register_fn = cast(Callable[[str, Callable[..., object]], object], register)
    register_fn(_CAPTURE_IMPL, _capture_batched_attn)
    _registered = True


def _validate_prebuilt(padded: BatchedPastLike, batch: int) -> None:
    """Raise for a padded cache whose mask, live counts, and width disagree."""
    if padded.mask.dim() != 2 or int(padded.mask.shape[0]) != batch:
        raise ValueError("BatchedPast mask batch must match the judger batch")
    if len(padded.live) != batch:
        raise ValueError("BatchedPast live counts must match the judger batch")
    width = kernels.cache_length(padded.past)
    if int(padded.mask.shape[1]) != width:
        raise ValueError("BatchedPast mask width must match its cache width")
    # Only on CPU: reading the row sums of a device tensor would synchronize on
    # every capture.
    if padded.mask.device.type == "cpu":
        mask_values = padded.mask.to(torch.long).sum(dim=1)
        tolist = cast(Callable[[], list[int]], cast(Any, mask_values).tolist)
        mask_live = tolist()
        if mask_live != [int(value) for value in padded.live]:
            raise ValueError("BatchedPast live counts must equal its mask row sums")


def _model_dtype(model: object) -> torch.dtype:
    """Return the dtype of the model's first parameter."""
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return torch.float32
    iterator = cast(Iterator[torch.Tensor], parameters())
    parameter = next(iterator, None)
    return parameter.dtype if parameter is not None else torch.float32


def capture_batched(
    model: object,
    padded: BatchedPastLike,
    judger_ids: Sequence[torch.Tensor],
    judger_masks: Sequence[torch.Tensor],
    question_ids: Sequence[TokenIds],
    *,
    pool_kernel: int,
    use_flex: bool = False,
    energy_pool_kernel: int | None = None,
    row_energy_pool_kernel: int | None = None,
    restore: bool = True,
    validate: bool = True,
) -> tuple[BatchedCaptureItem, ...]:
    """Run one capture forward over a prebuilt padded batch."""
    global _context
    if _context is not None or capture_single.current_context() is not None:
        raise RuntimeError("only one query-support capture may run per process")
    if kernels.has_layer_aware_attention(model):
        raise RuntimeError(
            "batched query-support capture is deferred for hybrid attention; use capture_single"
        )
    reducers.validate_pool_kernel(pool_kernel)
    if energy_pool_kernel is not None:
        reducers.validate_pool_kernel(energy_pool_kernel)
    if row_energy_pool_kernel is not None:
        reducers.validate_pool_kernel(row_energy_pool_kernel)
    batch = len(judger_ids)
    if batch == 0:
        raise ValueError("batched vote needs at least one cache")
    if len(judger_masks) != batch or len(question_ids) != batch:
        raise ValueError("judger ids, masks, and questions must have equal batch size")
    if validate:
        _validate_prebuilt(padded, batch)
    device = judger_ids[0].device
    dtype = _model_dtype(model)
    mem_lens = [int(length) for length in padded.live]
    spans = [kernels.locate_question(judger_ids[j], question_ids[j]) for j in range(batch)]
    los = [lo for lo, _hi, _found in spans]
    his = [hi for _lo, hi, _found in spans]
    founds = [found for _lo, _hi, found in spans]
    q_lens = [int(judger_ids[j].shape[1]) for j in range(batch)]
    q_width = max(q_lens)
    col_mask = padded.mask
    width = int(col_mask.shape[1])
    live_indices = [col_mask[j].nonzero(as_tuple=True)[0] for j in range(batch)]
    energies = (
        [
            EnergyAccumulator(
                mem_len=mem_lens[j],
                energy_pool_kernel=energy_pool_kernel,
                row_energy_pool_kernel=row_energy_pool_kernel,
            )
            for j in range(batch)
        ]
        if energy_pool_kernel is not None or row_energy_pool_kernel is not None
        else None
    )
    mask_dtype = judger_masks[0].dtype
    q_ids = torch.zeros(batch, q_width, dtype=torch.long, device=device)
    q_row_mask = torch.zeros(batch, q_width, dtype=mask_dtype, device=device)
    for index in range(batch):
        if judger_ids[index].dim() != 2 or int(judger_ids[index].shape[0]) != 1:
            raise ValueError("each judger prompt must be shaped [1, length]")
        if tuple(judger_masks[index].shape) != tuple(judger_ids[index].shape):
            raise ValueError("each judger mask must match its prompt shape")
        q_ids[index, : q_lens[index]] = judger_ids[index][0]
        q_row_mask[index, : q_lens[index]] = judger_masks[index][0]
    mem_lens_tensor = torch.tensor(mem_lens, device=device)
    if use_flex:
        if _create_block_mask is None:
            raise RuntimeError("FlexAttention block-mask support is unavailable")
        mask_mod = flex_mask_fn(col_mask.to(torch.bool), q_row_mask.to(torch.bool), width)
        block_mask = _create_block_mask(
            mask_mod,
            batch,
            None,
            q_width,
            width + q_width,
            device=device,
        )
        span_biases = [
            span_bias_item(
                col_mask[index],
                q_row_mask[index],
                width,
                los[index],
                his[index],
                dtype,
                device,
            )
            for index in range(batch)
        ]
        context = BatchedCaptureContext(
            los=los,
            his=his,
            width=width,
            bias=None,
            votes=[None] * batch,
            live_indices=live_indices,
            energies=energies,
            use_flex=True,
            block_mask=block_mask,
            span_biases=span_biases,
        )
    else:
        bias = batched_bias(col_mask, q_row_mask, width, q_width, dtype, device)
        context = BatchedCaptureContext(
            los=los,
            his=his,
            width=width,
            bias=bias,
            votes=[None] * batch,
            live_indices=live_indices,
            energies=energies,
        )
    backbone = kernels.get_backbone(model)
    _register_capture()
    config = cast(_ModelWithConfig, model).config
    previous = str(getattr(config, "_attn_implementation", "sdpa"))
    try:
        kernels.set_attention_implementation(model, _CAPTURE_IMPL)
        _context = context
        position_ids = mem_lens_tensor.unsqueeze(1) + torch.arange(q_width, device=device)
        with torch.no_grad():
            backbone(
                input_ids=q_ids,
                past_key_values=padded.past,
                position_ids=position_ids,
                attention_mask=torch.cat([col_mask, q_row_mask], dim=1),
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
    finally:
        _context = None
        try:
            kernels.set_attention_implementation(model, previous)
        finally:
            if restore and kernels.cache_length(padded.past) != width:
                padded.past.crop(width)
    return tuple(
        BatchedCaptureItem(
            mem_len=mem_lens[index],
            found=founds[index],
            live_indices=live_indices[index],
            votes=context.votes[index],
            heads=context.heads,
            question_rows=max(1, his[index] - los[index]),
            energy=energies[index] if energies is not None else None,
        )
        for index in range(batch)
    )
