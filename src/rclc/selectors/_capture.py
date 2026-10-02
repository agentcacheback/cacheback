"""Attention capture shared by the built-in selectors.

One forward runs over a private copy of the sender cache. A registered attention function
returns the model's own attention output and hands each global layer's query and key to a
selector-specific fold, so selectors never change the sender or its attention backend.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Protocol, cast

import torch

from rclc._cache import ATTENTION_LOCK, clone_cache, get_backbone

CAPTURE_IMPL = "rclc_vote_capture"

Attention = Callable[..., object]


class Fold(Protocol):
    """Reduce one global layer's rotated query and cached key into selector state."""

    def __call__(self, query: torch.Tensor, key: torch.Tensor, scaling: float) -> None:
        """Fold one layer; query is ``[1, heads, rows, dim]`` and key covers memory plus rows."""
        ...


@dataclass
class _Capture:
    """The state one capture shares with its attention function."""

    fold: Fold
    layer_types: tuple[str, ...]
    attention: Attention | None
    mask_factory: Attention
    layers: list[int] = field(default_factory=lambda: list[int]())


_active: _Capture | None = None
_registered = False


def _register() -> None:
    """Register the capture attention and mask functions with Transformers, once."""
    global _registered
    if _registered:
        return
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    for registry, function in (
        (ALL_ATTENTION_FUNCTIONS, _capture_attention),
        (ALL_MASK_ATTENTION_FUNCTIONS, _capture_mask),
    ):
        register = cast(Callable[[str, Attention], None], getattr(registry, "register"))  # noqa: B009
        register(CAPTURE_IMPL, function)
    _registered = True


def _capture_mask(*args: object, **kwargs: object) -> object:
    """Delegate mask construction to the backend the capture replaced."""
    if _active is None:
        raise RuntimeError("selector capture mask ran with no active capture")
    return _active.mask_factory(*args, **kwargs)


def _original_backend(implementation: str) -> tuple[Attention | None, Attention]:
    """Resolve the attention and mask callables in use before the swap; eager resolves later."""
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    try:
        mask = cast(Attention, ALL_MASK_ATTENTION_FUNCTIONS[implementation])
        if implementation == "eager":
            return None, mask
        return cast(Attention, ALL_ATTENTION_FUNCTIONS[implementation]), mask
    except KeyError as exc:
        raise RuntimeError(f"capture cannot delegate attention {implementation!r}") from exc


def _eager_attention(module: object) -> Attention:
    """Resolve a model family's eager attention implementation at call time."""
    eager = getattr(import_module(module.__class__.__module__), "eager_attention_forward", None)
    if not callable(eager):
        raise RuntimeError(f"no eager attention in {module.__class__.__module__}")
    return eager


def _capture_attention(
    module: object,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: object,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: object,
) -> tuple[torch.Tensor, None]:
    """Return the model's own attention output and fold global layers aside."""
    capture = _active
    if capture is None:
        raise RuntimeError("selector capture attention ran with no active capture")
    layer = int(getattr(module, "layer_idx", len(capture.layers)))
    capture.layers.append(layer)
    attention = capture.attention or _eager_attention(module)
    result = attention(
        module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs
    )
    if capture.layer_types[layer] == "global":
        capture.fold(query, key, scaling)
    return cast(tuple[torch.Tensor, object], result)[0], None


def set_attention_implementation(model: object, implementation: str) -> None:
    """Set attention implementation through the model setter or config fallback."""
    setter = getattr(model, "set_attn_implementation", None)
    if setter is not None:
        try:
            cast(Callable[[str], object], setter)(implementation)
            return
        except Exception:
            pass
    setattr(getattr(model, "config"), "_attn_implementation", implementation)  # noqa: B009, B010


def _snapshot_implementations(*models: object) -> tuple[tuple[object, object], ...]:
    """Snapshot every model and sub-config implementation by object identity."""
    configs: dict[int, object] = {}

    def add(config: object | None) -> None:
        if config is not None and id(config) not in configs:
            configs[id(config)] = config
            for name in getattr(config, "sub_configs", ()):
                add(getattr(config, name, None))

    for model in models:
        add(getattr(model, "config", None))
        modules = getattr(model, "modules", None)
        if callable(modules):
            for module in cast(Callable[[], Iterable[object]], modules)():
                add(getattr(module, "config", None))
    return tuple((c, getattr(c, "_attn_implementation", None)) for c in configs.values())


def _restore_implementations(snapshot: tuple[tuple[object, object], ...]) -> None:
    """Restore every snapshotted implementation without recursive clobbering."""
    for config, implementation in snapshot:
        values = vars(config)
        if hasattr(config, "_attn_implementation_internal"):
            values["_attn_implementation_internal"] = implementation
        else:
            values["_attn_implementation"] = implementation
        values.pop("_attn_was_changed", None)


def run_capture(
    model: Any,
    past: Any,
    fold: Fold,
    *,
    memory: int,
    input_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
) -> None:
    """Forward ``[1, rows]`` inputs after the first ``memory`` cached positions of a copy."""
    global _active
    backbone = get_backbone(model)
    config = backbone.config
    configured = getattr(config, "layer_types", None) or ("full",) * config.num_hidden_layers
    layer_types = tuple(
        "sliding" if "sliding" in str(kind) or "chunked" in str(kind) else "global"
        for kind in configured
    )
    if "global" not in layer_types:
        raise RuntimeError("selector capture needs at least one global attention layer")
    implementation = str(getattr(config, "_attn_implementation", "sdpa"))
    attention, mask_factory = _original_backend(implementation)
    _register()
    inputs = input_ids if input_ids is not None else inputs_embeds
    if inputs is None:
        raise ValueError("run_capture needs input_ids or inputs_embeds")
    rows, device = int(inputs.shape[1]), inputs.device
    capture = _Capture(fold, layer_types, attention, mask_factory)
    with ATTENTION_LOCK, torch.inference_mode():
        private: Any = clone_cache(past)
        if int(private.get_seq_length()) != memory:
            private.crop(memory)
        snapshot = _snapshot_implementations(model, backbone)
        try:
            set_attention_implementation(model, CAPTURE_IMPL)
            if config._attn_implementation != CAPTURE_IMPL:
                set_attention_implementation(backbone, CAPTURE_IMPL)
            if config._attn_implementation != CAPTURE_IMPL:
                raise RuntimeError(f"attention swap refused: {config._attn_implementation!r}")
            _active = capture
            backbone(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                past_key_values=private,
                position_ids=torch.arange(memory, memory + rows, device=device)[None],
                attention_mask=torch.ones(1, memory + rows, dtype=torch.long, device=device),
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
        finally:
            _active = None
            try:
                set_attention_implementation(model, implementation)
                if config._attn_implementation != implementation:
                    set_attention_implementation(backbone, implementation)
            finally:
                _restore_implementations(snapshot)
    if sorted(capture.layers) != list(range(len(layer_types))):
        raise RuntimeError(f"selector capture saw layers {capture.layers}, not every layer")
