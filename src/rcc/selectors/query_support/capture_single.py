"""The single-item query-support capture: one scored forward over a cache.

A registered attention function reduces the question rows in chunks on the side, so
the full softmax never materializes. Only one capture may be active per process.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from importlib import import_module
from typing import cast

import torch

from rcc.selectors.core import kernels
from rcc.selectors.core.types import (
    AttentionLayerObserver,
    CaptureResult,
    CaptureSpec,
    EnergyAccumulator,
    EnergyChunkState,
    LayerDescriptor,
    MutableHFCache,
    TokenIds,
)
from rcc.selectors.query_support import capture_layers
from rcc.selectors.query_support.capture_scoring import score_attention_bias
from rcc.selectors.query_support.methods import reducers

CAPTURE_IMPL = "rcc_vote_capture"
"""The name this capture's attention function is registered under."""

VOTE_ROW_CHUNK = 256
"""Maximum question-row width the capture softmax materializes at once."""

__all__ = [
    "CAPTURE_IMPL",
    "VOTE_ROW_CHUNK",
    "CaptureContext",
    "current_context",
]


def _int_list() -> list[int]:
    """Create a typed callback-index list for the dataclass default."""
    return []


def _descriptor_list() -> list[LayerDescriptor]:
    """Create a typed descriptor list for the dataclass default."""
    return []


@dataclass
class CaptureContext(EnergyAccumulator):
    """The state one capture shares with its attention function."""

    lo: int = 0
    hi: int = 0
    bias: torch.Tensor | None = None
    bias_span: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    votes: torch.Tensor | None = None
    heads: int = 1
    lower_right: bool = False
    collect_votes: bool = True
    collect_statistics: bool = False
    head_stats: bool = False
    observer: AttentionLayerObserver | None = None
    rank_sum: torch.Tensor | None = None
    nomination_sum: torch.Tensor | None = None
    pairs: int = 0
    original_impl: str = "sdpa"
    original_attention: Callable[..., object] | None = None
    original_mask_factory: Callable[..., object] | None = None
    callback_hits: int = 0
    callback_layer_indices: list[int] = field(default_factory=_int_list)
    layer_descriptors: list[LayerDescriptor] = field(default_factory=_descriptor_list)
    layer_types: tuple[str, ...] = ()
    scoring_policy: str = "all_layers"


_context: CaptureContext | None = None
_registered = False


def current_context() -> CaptureContext | None:
    """Return the capture context that is active, or None."""
    return _context


def __getattr__(name: str) -> object:
    """Expose the private context under its alternate module-level name."""
    if name == "_CTX":
        return _context
    raise AttributeError(name)


def _span_bias(
    lo: int,
    hi: int,
    mem_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build only the causal bias rows the question-row softmax needs."""
    span = hi - lo
    key_length = mem_len + hi
    bias = torch.zeros(span, key_length, dtype=dtype, device=device)
    if span > 0 and hi > 0:
        query_positions = torch.arange(lo, hi, device=device).unsqueeze(1)
        key_positions = torch.arange(hi, device=device).unsqueeze(0)
        bias[:, mem_len:] = torch.where(
            key_positions > query_positions,
            torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=device),
            torch.zeros((), dtype=dtype, device=device),
        )
    return bias.view(1, 1, span, key_length)


def _register_capture() -> None:
    """Register the capture attention and mask functions with Transformers, once."""
    global _registered
    if _registered:
        return
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    attention_register = getattr(ALL_ATTENTION_FUNCTIONS, "register", None)
    mask_register = getattr(ALL_MASK_ATTENTION_FUNCTIONS, "register", None)
    if not callable(attention_register) or not callable(mask_register):
        raise RuntimeError("Transformers attention registry has no register method")
    register_attention = cast(Callable[[str, Callable[..., object]], object], attention_register)
    register_mask = cast(Callable[[str, Callable[..., object]], object], mask_register)
    register_attention(CAPTURE_IMPL, _vote_capture_attn)
    register_mask(CAPTURE_IMPL, _capture_mask_factory)
    _registered = True


def _capture_mask_factory(*args: object, **kwargs: object) -> object:
    """Delegate mask construction to the backend the capture replaced."""
    context = _context
    if context is None or context.original_mask_factory is None:
        raise RuntimeError("capture mask factory invoked with no active capture context")
    return context.original_mask_factory(*args, **kwargs)


def _decoder_config(backbone: object) -> object:
    """Return the config owned by the decoder that dispatches attention."""
    config = getattr(backbone, "config", None)
    if config is None:
        raise RuntimeError("capture decoder has no config")
    return config


def _resolve_original_backend(
    implementation: str,
) -> tuple[Callable[..., object] | None, Callable[..., object]]:
    """Resolve the attention and mask callables in use before the config swap."""
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    try:
        mask_factory = cast(Callable[..., object], ALL_MASK_ATTENTION_FUNCTIONS[implementation])
    except KeyError as exc:
        raise RuntimeError(
            f"capture cannot delegate mask implementation {implementation!r}"
        ) from exc
    if implementation == "eager":
        return None, mask_factory
    try:
        attention = cast(Callable[..., object], ALL_ATTENTION_FUNCTIONS[implementation])
    except KeyError as exc:
        raise RuntimeError(
            f"capture cannot delegate attention implementation {implementation!r}"
        ) from exc
    return attention, mask_factory


def _eager_attention_forward(module: object) -> Callable[..., object]:
    """Resolve a model family's eager attention implementation at call time."""
    family = import_module(module.__class__.__module__)
    eager = getattr(family, "eager_attention_forward", None)
    if not callable(eager):
        eager = getattr(module, "eager_attention_forward", None)
    if not callable(eager):
        raise RuntimeError(
            f"eager attention implementation is unavailable for {module.__class__.__module__}"
        )
    return eager


def _delegate_attention(
    context: CaptureContext,
    module: object,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: object,
    scaling: float,
    dropout: float,
    kwargs: dict[str, object],
) -> tuple[torch.Tensor, object]:
    """Run the original attention backend, leaving the scoring to the caller."""
    if context.original_impl == "eager":
        attention = _eager_attention_forward(module)
    elif context.original_attention is not None:
        attention = context.original_attention
    else:
        raise RuntimeError("capture context has no original attention callable")
    result = attention(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        **kwargs,
    )
    return cast(tuple[torch.Tensor, object], result)


def _vote_capture_attn(
    module: object,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: object,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: object,
) -> tuple[torch.Tensor, None]:
    """Return the model's own attention output and reduce the question rows on the side."""
    context = _context
    if context is None:
        raise RuntimeError("vote capture invoked with no active capture context")
    descriptor = capture_layers.descriptor(context, module, key, query)
    context.callback_hits += 1
    context.callback_layer_indices.append(descriptor.layer_idx)
    context.layer_descriptors.append(descriptor)
    output, _attention_weights = _delegate_attention(
        context,
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout,
        kwargs,
    )
    repeats = int(getattr(module, "num_key_value_groups", 1))
    full_key = kernels.repeat_kv(key, repeats)
    if context.hi <= context.lo:
        return output, None

    rows = context.hi - context.lo
    include_layer = context.scoring_policy == "all_layers" or descriptor.is_global
    needs_statistics = include_layer and (
        context.collect_statistics or context.head_stats or context.observer is not None
    )
    if not include_layer:
        return output, None
    energy_state = EnergyChunkState()
    per_head_sum: torch.Tensor | None = None
    needs_reducers = include_layer and (
        context.energy_pool_kernel is not None
        or context.row_energy_pool_kernel is not None
        or bool(context.support_orders)
    )
    layer_energies: EnergyAccumulator | None = None
    if needs_reducers:
        capture_layers.assert_fold_geometry(
            descriptor,
            include_layer=include_layer,
            scoring_policy=context.scoring_policy,
        )
        layer_energies = EnergyAccumulator(
            mem_len=context.mem_len if include_layer else descriptor.physical_memory_length,
            energy_pool_kernel=context.energy_pool_kernel,
            row_energy_pool_kernel=context.row_energy_pool_kernel,
            support_orders=context.support_orders,
        )
    for row_start in range(0, rows, VOTE_ROW_CHUNK):
        row_end = min(row_start + VOTE_ROW_CHUNK, rows)
        question = query[:, :, context.lo + row_start : context.lo + row_end, :]
        logits = torch.matmul(question, full_key.transpose(2, 3)) * scaling
        if context.scoring_policy == "all_layers":
            score_bias = context.bias_span[..., row_start:row_end, :].to(logits.dtype)
        else:
            score_bias = score_attention_bias(
                context,
                descriptor,
                attention_mask,
                row_start,
                row_end,
                int(full_key.shape[-2]),
                logits.dtype,
                logits.device,
            )
        logits = logits + score_bias
        weights = torch.softmax(logits, dim=-1, dtype=torch.float32)
        physical_memory_length = descriptor.physical_memory_length
        contribution = weights[0, :, :, :physical_memory_length].sum(dim=(0, 1))
        contribution = capture_layers.scatter_memory(
            contribution, descriptor, descriptor.logical_memory_length
        )
        if context.collect_votes and include_layer:
            context.votes = contribution if context.votes is None else context.votes + contribution
        if include_layer:
            context.heads = int(weights.shape[1])
        if layer_energies is not None:
            reducers.fold_capture_energies(
                layer_energies,
                weights[0],
                kv_groups=repeats,
                state=energy_state,
            )
        if needs_statistics and descriptor.logical_memory_length > 0:
            head_chunk = weights[0].sum(dim=1)[:, :physical_memory_length]
            head_chunk = capture_layers.scatter_memory(
                head_chunk, descriptor, descriptor.logical_memory_length
            )
            per_head_sum = head_chunk if per_head_sum is None else per_head_sum + head_chunk
    if layer_energies is not None:
        reducers.finish_capture_energies(layer_energies, energy_state)
        if needs_reducers:
            reducers.merge_layer_energies(context, layer_energies)
    if per_head_sum is not None:
        if context.observer is not None and include_layer:
            context.observer.observe_layer(
                per_head_sum,
                query_rows=rows,
                kv_groups=repeats,
            )
        if context.head_stats and include_layer:
            head_count, columns = (int(size) for size in per_head_sum.shape)
            order = per_head_sum.argsort(dim=-1)
            grid = torch.arange(columns, dtype=torch.float32, device=per_head_sum.device)
            ranks = torch.empty_like(per_head_sum)
            ranks.scatter_(-1, order, grid.repeat(head_count, 1))
            ranks = ranks / float(max(1, columns - 1))
            nomination = torch.zeros_like(per_head_sum)
            top_k = max(1, columns // 20)
            nomination.scatter_(-1, per_head_sum.topk(top_k, dim=-1).indices, 1.0)
            rank_sum, nomination_sum = ranks.sum(dim=0), nomination.sum(dim=0)
            context.rank_sum = rank_sum if context.rank_sum is None else context.rank_sum + rank_sum
            context.nomination_sum = (
                nomination_sum
                if context.nomination_sum is None
                else context.nomination_sum + nomination_sum
            )
            context.pairs += head_count
    return output, None


def _validate_spec(pool_kernel: int, spec: CaptureSpec) -> tuple[float, ...]:
    """Check the pooling kernels and support orders, returning the sorted orders."""
    reducers.validate_pool_kernel(pool_kernel)
    if spec.energy_pool_kernel is not None:
        reducers.validate_pool_kernel(spec.energy_pool_kernel)
    if spec.row_energy_pool_kernel is not None:
        reducers.validate_pool_kernel(spec.row_energy_pool_kernel)
    orders = tuple(float(order) for order in spec.support_orders)
    if len(set(orders)) != len(orders) or any(
        order <= 1 or not math.isfinite(order) for order in orders
    ):
        raise ValueError("support orders must be unique finite values greater than one")
    if orders and (
        spec.energy_pool_kernel is None or spec.row_energy_pool_kernel != spec.energy_pool_kernel
    ):
        raise ValueError("generalized support capture requires one shared pooling kernel")
    return tuple(sorted(orders))


def _model_dtype(model: object) -> torch.dtype:
    """Read the first parameter dtype at the Transformers boundary."""
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return torch.float32
    iterator = cast(Iterator[torch.Tensor], parameters())
    parameter = next(iterator, None)
    return parameter.dtype if parameter is not None else torch.float32


def _result(
    context: CaptureContext,
    *,
    pool_kernel: int,
    spec: CaptureSpec,
    found: bool,
    device: torch.device,
) -> CaptureResult:
    """Convert the mutable capture state into its frozen result."""
    from rcc.selectors.query_support.methods.scorers import finish_capture_result

    return finish_capture_result(
        context,
        pool_kernel=pool_kernel,
        spec=spec,
        found=found,
        device=device,
    )


def _assert_text_decoder_inputs(model: object, judger_ids: torch.Tensor) -> None:
    """Raise for multimodal placeholder ids or a loaded vision or audio tower."""
    config = getattr(model, "config", None)
    forbidden = {
        int(value)
        for name in (
            "image_token_id",
            "image_token_index",
            "video_token_id",
            "audio_token_id",
            "boi_token_id",
            "boi_token_index",
            "eoi_token_id",
            "eoi_token_index",
            "boa_token_id",
        )
        if (value := getattr(config, name, None)) is not None
    }
    if forbidden and bool(
        torch.isin(judger_ids, torch.tensor(tuple(forbidden), device=judger_ids.device)).any()
    ):
        raise ValueError("query-support capture accepts text-decoder inputs only")
    owners = (model, getattr(model, "model", None))
    tower_names = ("vision_tower", "audio_tower", "embed_vision", "embed_audio")
    loaded = tuple(
        name
        for owner in owners
        if owner is not None
        for name in tower_names
        if getattr(owner, name, None) is not None
    )
    if loaded:
        raise ValueError(
            "query-support capture requires a text decoder with no loaded vision/audio tower"
        )


def capture_context(
    model: object,
    past: MutableHFCache | None,
    judger_ids: torch.Tensor,
    judger_mask: torch.Tensor,
    question_ids: TokenIds,
    *,
    pool_kernel: int,
    spec: CaptureSpec,
    observer: AttentionLayerObserver | None,
    lower_right: bool,
    collect_votes: bool | None = None,
    consume_past: bool = False,
) -> tuple[CaptureResult, CaptureContext]:
    """Run one capture forward and return its result and the context it filled.

    The forward appends judger rows to the cache it reads, so by default it reads a copy: a
    second whole cache, 19 GiB per 128K rows. ``consume_past`` hands it over, no copy instead.
    """
    global _context
    if _context is not None:
        raise RuntimeError("only one query-support capture may run per process")
    orders = _validate_spec(pool_kernel, spec)
    memory_length = kernels.cache_logical_length(past)
    device = judger_ids.device
    query_length = int(judger_ids.shape[1])
    if memory_length == 0 or query_length == 0:
        context = CaptureContext(
            mem_len=memory_length,
            support_orders=orders,
            bias_span=torch.zeros(0, dtype=torch.float32, device=device),
            observer=observer,
        )
        return _result(
            context, pool_kernel=pool_kernel, spec=spec, found=False, device=device
        ), context

    lo, hi, found = kernels.locate_question(judger_ids, question_ids)
    lo = max(0, min(lo, query_length))
    hi = max(lo + 1, min(hi, query_length))
    backbone = kernels.get_backbone(model)
    config = _decoder_config(backbone)
    expected_layers = capture_layers.expected_layers(config)
    layer_types = capture_layers.layer_types(config, expected_layers)
    scoring_policy = "global_layers" if kernels.has_layer_aware_attention(model) else "all_layers"
    if scoring_policy == "global_layers" and not any(
        layer_type == "global" for layer_type in layer_types
    ):
        raise RuntimeError("global_layers scoring policy requires at least one global layer")
    original_impl = str(getattr(config, "_attn_implementation", "sdpa"))
    original_attention, original_mask_factory = _resolve_original_backend(original_impl)
    _assert_text_decoder_inputs(model, judger_ids)
    _register_capture()
    dtype = _model_dtype(model)
    use_lower_right = lower_right and (
        kernels.lower_right_causal_bias(hi, memory_length + hi) is not None
    )
    if use_lower_right:
        bias = None
        bias_span = _span_bias(lo, hi, memory_length, dtype, device)
    else:
        bias = kernels.causal_bias(hi, memory_length, dtype, device)
        bias_span = bias[:, :, lo:hi, :]
    context = CaptureContext(
        mem_len=memory_length,
        energy_pool_kernel=spec.energy_pool_kernel,
        row_energy_pool_kernel=spec.row_energy_pool_kernel,
        support_orders=orders,
        lo=lo,
        hi=hi,
        bias=bias,
        bias_span=bias_span,
        lower_right=use_lower_right,
        collect_votes=(
            spec.collect_snap or spec.collect_statistics or observer is not None
            if collect_votes is None
            else collect_votes
        ),
        collect_statistics=spec.collect_statistics,
        head_stats=spec.collect_statistics,
        observer=observer,
        original_impl=original_impl,
        original_attention=original_attention,
        original_mask_factory=original_mask_factory,
        layer_types=layer_types,
        scoring_policy=scoring_policy,
    )
    capture_past = cast(MutableHFCache, past if consume_past else kernels.clone_cache(past))
    attention_snapshot = kernels.snapshot_attention_implementations(model, backbone)
    try:
        kernels.set_attention_implementation(model, CAPTURE_IMPL)
        if str(getattr(config, "_attn_implementation", "")) != CAPTURE_IMPL:
            kernels.set_attention_implementation(backbone, CAPTURE_IMPL)
        if str(getattr(config, "_attn_implementation", "")) != CAPTURE_IMPL:
            raise RuntimeError(
                "attention implementation swap was refused: "
                f"decoder config stayed {getattr(config, '_attn_implementation', None)!r}"
            )
        _context = context
        base = torch.ones(1, memory_length, dtype=judger_mask.dtype, device=device)
        with torch.no_grad():
            backbone(
                input_ids=judger_ids[:, :hi],
                past_key_values=capture_past,
                position_ids=torch.arange(
                    memory_length, memory_length + hi, device=device
                ).unsqueeze(0),
                attention_mask=torch.cat([base, judger_mask[:, :hi]], dim=1),
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
        expected_indices = capture_layers.attention_indices(config)
        if context.callback_hits != len(expected_indices):
            raise RuntimeError(
                "capture attention callback hit count mismatch: "
                f"{context.callback_hits} != {len(expected_indices)}"
            )
        if set(context.callback_layer_indices) != set(expected_indices):
            raise RuntimeError(
                "capture attention callback layer indices were incomplete: "
                f"{context.callback_layer_indices!r}"
            )
    finally:
        _context = None
        try:
            kernels.set_attention_implementation(model, original_impl)
            if str(getattr(config, "_attn_implementation", "")) != original_impl:
                kernels.set_attention_implementation(backbone, original_impl)
        finally:
            kernels.restore_attention_implementations(attention_snapshot)
    return _result(context, pool_kernel=pool_kernel, spec=spec, found=found, device=device), context
