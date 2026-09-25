"""The vLLM 0.26 engine and tokenizer records for resident Gemma."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from rcc.models.gemma import GEMMA

VLLM_LOGPROBS_MODE = "processed_logits"
FULL_VOCAB_LOGPROBS = -1
VLLM_ENGINE_ARG_NAMES = frozenset(
    {
        "model",
        "revision",
        "dtype",
        "attention_config",
        "language_model_only",
        "enable_prompt_embeds",
        "enable_prefix_caching",
        "enforce_eager",
        "generation_config",
        "logprobs_mode",
        "max_logprobs",
        "max_model_len",
        "gpu_memory_utilization",
    }
)
REGISTERED_LAYER_TYPES = (("sliding_attention",) * 5 + ("full_attention",)) * 8
_FLASH_BACKEND_CLASS = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"


class ThinkingWrapError(Exception):
    """A registered resident-engine invariant did not hold."""


def require(holds: bool, message: str) -> None:
    """Refuse the engine unless one exact registered condition holds."""
    if not holds:
        raise ThinkingWrapError(message)


@dataclass(frozen=True)
class StackPins:
    """The sole runtime stack accepted by the resident mechanism."""

    vllm: str
    transformers: str
    torch: str


@dataclass(frozen=True)
class TemplateShape:
    """Measured thinking-off/on Gemma chat-template token geometry."""

    off_prefix: tuple[int, ...]
    off_suffix: tuple[int, ...]
    on_prefix: tuple[int, ...]
    on_suffix: tuple[int, ...]
    pre_closed_thought: tuple[int, ...]
    user_turn_open: tuple[int, ...]
    think_control_text: str = "<|think|>"


@dataclass(frozen=True)
class ThinkingWrapSpec:
    """The fields one live receiver engine is checked against."""

    checkpoint_id: str
    checkpoint_revision: str
    stack: StackPins
    template: TemplateShape
    suppressed_token_ids: tuple[int, ...] = (258_883, 258_882)
    suppressed_words: tuple[str, ...] = ("<audio|>", "<image|>")
    stop_ids: frozenset[int] = frozenset({1, 106, 50})
    engine_architecture: str = "Gemma4UnifiedForConditionalGeneration"
    # vllm names this concept in two shapes. This row is compared against the
    # enum name (AttentionBackendEnum.FLASH_ATTN.name); the per-layer rows are
    # compared against the class qualname, which they derive for themselves.
    attention_backend: str = "FLASH_ATTN"
    flash_attn_version: int = 4


REGISTERED_TEMPLATE = TemplateShape(
    off_prefix=(2, 105, 2364, 107),
    off_suffix=(106, 107, 105, 4368, 107, 100, 45518, 107, 101),
    on_prefix=(2, 105, 9731, 107, 98, 107, 106, 107, 105, 2364, 107),
    on_suffix=(106, 107, 105, 4368, 107),
    pre_closed_thought=(100, 45518, 107, 101),
    user_turn_open=(105, 2364, 107),
)
REGISTERED_SPEC = ThinkingWrapSpec(
    checkpoint_id=GEMMA.checkpoint,
    checkpoint_revision=GEMMA.revision,
    stack=StackPins(vllm="0.26.0", transformers="5.13.0", torch="2.11.0"),
    template=REGISTERED_TEMPLATE,
)


def registered_attention_layers() -> list[dict[str, Any]]:
    """Return the exact 48-layer hybrid FlashAttention record."""
    return [
        {
            "layer": index,
            "kind": kind,
            "head_size": 512 if kind == "full_attention" else 256,
            "sliding_window": None if kind == "full_attention" else 1024,
            "backend": _FLASH_BACKEND_CLASS.rsplit(".", 1)[1],
            "backend_class": _FLASH_BACKEND_CLASS,
        }
        for index, kind in enumerate(REGISTERED_LAYER_TYPES)
    ]


def attention_layer_record(model: Any) -> list[dict[str, Any]]:
    """Read the live attention backend constructed in each decoder layer."""
    import re

    rows: list[dict[str, Any]] = []
    for _name, module in model.named_modules():
        read_backend = getattr(module, "get_attn_backend", None)
        if not callable(read_backend) or not hasattr(module, "head_size"):
            continue
        match = re.fullmatch(
            r".*\.layers\.(\d+)\.self_attn\.attn",
            str(getattr(module, "layer_name", "")),
        )
        backend: Any = read_backend()
        backend_module, backend_name = backend.full_cls_name()
        sliding_window: object = getattr(module, "sliding_window", None)
        if isinstance(sliding_window, tuple):
            values = cast(tuple[object, ...], sliding_window)
            first = values[0] if values else None
            sliding_window = int(first) if isinstance(first, int) else None
        rows.append(
            {
                "layer": int(match.group(1)) if match else -1,
                "kind": ("sliding_attention" if sliding_window is not None else "full_attention"),
                "head_size": int(module.head_size),
                "sliding_window": sliding_window,
                "backend": str(backend_name),
                "backend_class": f"{backend_module}.{backend_name}",
            }
        )
    return sorted(rows, key=lambda row: int(row["layer"]))


def suppression_bad_words(
    tokenizer: Any,
    spec: ThinkingWrapSpec = REGISTERED_SPEC,
) -> tuple[str, ...]:
    """Derive and verify the two checkpoint suppression strings."""
    words: list[str] = []
    for token in spec.suppressed_token_ids:
        value = str(tokenizer.decode([token]))
        encoded = [int(row) for row in tokenizer(value, add_special_tokens=False)["input_ids"]]
        require(encoded == [token], f"suppressed id {token} does not round-trip")
        words.append(value)
    derived = tuple(words)
    require(derived == spec.suppressed_words, "Gemma suppression strings drifted")
    return derived


__all__ = (
    "FULL_VOCAB_LOGPROBS",
    "REGISTERED_LAYER_TYPES",
    "REGISTERED_SPEC",
    "REGISTERED_TEMPLATE",
    "VLLM_ENGINE_ARG_NAMES",
    "VLLM_LOGPROBS_MODE",
    "attention_layer_record",
    "registered_attention_layers",
    "suppression_bad_words",
)
