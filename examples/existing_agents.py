"""Reuse an agent's cached conversation and keep the receiver's own instructions."""

import asyncio
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rcc import bind, transfer


async def handoff(model: Any, tokenizer: Any) -> str:
    """Run the agents' native loops around a single RCC handoff."""
    history = [
        {"role": "system", "content": "You track project decisions."},
        {"role": "user", "content": "Cedar launches October 12. Maya Chen owns the release."},
        {"role": "assistant", "content": "Recorded Cedar's owner and launch date."},
        {
            "role": "user",
            "content": "Update: Cedar now launches October 19. The owner is unchanged.",
        },
        {"role": "assistant", "content": "Recorded the revised launch date."},
    ]
    ids = tokenizer.apply_chat_template(history, return_tensors="pt", return_dict=True)[
        "input_ids"
    ].to(model.device)
    # This prefill belongs to the existing sender loop; RCC reuses its exact cache.
    with torch.inference_mode():
        cache = model.model(input_ids=ids, use_cache=True).past_key_values
    sender = bind(model, tokenizer, prompt=ids, past_key_values=cache, backend="hf")
    receiver = bind(
        model,
        tokenizer,
        messages=[
            {
                "role": "system",
                "content": "Use the preceding project notes. Answer in one sentence.",
            }
        ],
        max_new_tokens=64,
        context_limit=256,
    )
    # Keep at most half of this short conversation, shrinking further only if needed.
    await transfer(sender, receiver, "When does Cedar launch now?", ratio=2)
    inputs = receiver.pop()
    output = model.generate(
        **inputs,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )
    receiver.update(inputs=inputs, generation=output)
    await receiver.append("\nCalendar check: the October 19 launch is confirmed.\n")
    await transfer(receiver, sender, "When is the confirmed launch?", ratio=2)
    return str(tokenizer.decode(output.sequences[0], skip_special_tokens=True))


def main() -> None:
    """Run on CPU with Qwen3-0.6B; download weights only on first use."""
    torch.set_num_threads(4)
    checkpoint = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, attn_implementation="sdpa").eval()
    print(asyncio.run(handoff(model, tokenizer)))


if __name__ == "__main__":
    main()
