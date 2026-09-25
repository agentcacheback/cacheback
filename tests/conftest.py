from typing import Any

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from rcc import KVCache

TINY_VOCAB = 512


def make_cache(
    layers: int = 2,
    kv_heads: int = 2,
    length: int = 8,
    head_dim: int = 4,
    dtype: torch.dtype = torch.float16,
    seed: int = 0,
) -> KVCache:
    gen = torch.Generator().manual_seed(seed)
    shape = (layers, kv_heads, length, head_dim)
    return KVCache(
        keys=torch.randn(shape, generator=gen).to(dtype),
        values=torch.randn(shape, generator=gen).to(dtype),
        positions=torch.arange(length, dtype=torch.int64),
    )


def random_ids(length: int = 12, seed: int = 0, vocab: int = TINY_VOCAB) -> torch.Tensor:
    """A [1, length] batch of random token ids for the tiny model (no tokenizer needed)."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (1, length), generator=gen)


def prefill(model: Any, length: int = 12) -> Any:
    """Prefill one tiny model over deterministic ids and return its cache."""
    ids = random_ids(length=length, seed=410)
    with torch.no_grad():
        return model(input_ids=ids, use_cache=True, return_dict=True).past_key_values


def tiny_config(attn_implementation: str = "eager") -> Qwen3Config:
    """The one tiny Qwen3 config; tests that need a non-eager model vary only the impl."""
    return Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=TINY_VOCAB,
        max_position_embeddings=256,
        attn_implementation=attn_implementation,
    )


@pytest.fixture(scope="session")
def tiny_model() -> Qwen3ForCausalLM:
    """A tiny seeded Qwen3 on CPU with eager attention."""
    torch.manual_seed(0)
    return Qwen3ForCausalLM(tiny_config()).eval()
