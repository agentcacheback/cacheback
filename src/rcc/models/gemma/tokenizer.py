"""Pinned Gemma tokenizer loader used before resident engine bring-up."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Any

from rcc.models.gemma import GEMMA
from rcc.models.gemma.text_codec import GEMMA_TOKENIZER_JSON_SHA256

GEMMA_TOKENIZER_PATTERNS = ("tokenizer*", "*.model", "*.jinja", "special_tokens_map.json")


def load_gemma4_tokenizer(
    *,
    checkpoint_id: str = GEMMA.checkpoint,
    revision: str = GEMMA.revision,
) -> Any:
    """Snapshot and load only the exact registered tokenizer files."""
    hub: Any = importlib.import_module("huggingface_hub")
    transformers: Any = importlib.import_module("transformers")
    local_dir = str(
        hub.snapshot_download(
            checkpoint_id,
            revision=revision,
            allow_patterns=list(GEMMA_TOKENIZER_PATTERNS),
        )
    )
    # Every Gemma-family sender reuses the panel's 12B prompt_ids verbatim, so
    # this snapshot's tokenizer.json must be byte-identical to the registered
    # one; a drifted vocabulary would prefill foreign ids silently.
    digest = hashlib.sha256((Path(local_dir) / "tokenizer.json").read_bytes()).hexdigest()
    if digest != GEMMA_TOKENIZER_JSON_SHA256:
        raise RuntimeError(
            f"{checkpoint_id}@{revision}: tokenizer.json sha256 {digest} differs from the "
            f"registered {GEMMA_TOKENIZER_JSON_SHA256}; the panel prompt_ids are not valid "
            "for this tokenizer"
        )
    tokenizer: Any = transformers.AutoTokenizer.from_pretrained(local_dir)
    if getattr(tokenizer, "chat_template", None) is None:
        raise RuntimeError(
            "tokenizer loaded without a chat template; chat_template.jinja is missing"
        )
    return tokenizer


__all__ = ("GEMMA_TOKENIZER_PATTERNS", "load_gemma4_tokenizer")
