"""Native models and distinct sender states for complete offline handoff scenarios."""

from typing import Any

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from rcc import SenderState


def tiny_config(attn_implementation: str = "sdpa") -> Qwen3Config:
    return Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=512,
        max_position_embeddings=512,
        attn_implementation=attn_implementation,
    )


@pytest.fixture
def model() -> Any:
    with torch.random.fork_rng():
        torch.manual_seed(17)
        return Qwen3ForCausalLM(tiny_config()).eval()


@pytest.fixture
def senders(model: Any) -> list[SenderState]:
    vocabulary = {
        word: i
        for i, word in enumerate(
            ["[UNK]", "Who", "owns", "Cedar", "When", "Birch", "launches", "?"]
        )
    }
    unknown = "[UNK]"
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token=unknown))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token=unknown)
    states = []
    with torch.inference_mode():
        for offset, length in ((10, 80), (200, 96)):
            ids = torch.arange(offset, offset + length)[None]
            rows = model.get_input_embeddings()(ids)
            out = model.model(inputs_embeds=rows, use_cache=True)
            # The existing agent owns its two latent steps and records the actual input rows.
            for _ in range(2):
                thought = out.last_hidden_state[:, -1:]
                rows = torch.cat((rows, thought), dim=1)
                out = model.model(
                    inputs_embeds=thought, past_key_values=out.past_key_values, use_cache=True
                )
            states.append(
                SenderState(
                    model,
                    out.past_key_values,
                    rows[0],
                    tokenizer=tokenizer,
                    token_ids=torch.cat((ids[0], torch.tensor([-1, -1]))),
                    latent_steps=2,
                )
            )
    return states
