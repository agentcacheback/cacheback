"""Run a two-sender Qwen handoff on CPU; downloads Qwen3-0.6B on first use."""

import asyncio

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rcc import bind, transfer


async def main() -> None:
    """Bind existing models, deliver state, then generate through native Hugging Face."""
    torch.set_num_threads(4)
    checkpoint = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, attn_implementation="sdpa").eval()
    senders = [
        bind(model, tokenizer, prompt=document, backend="hf")
        for document in (
            "Cedar is owned by Maya Chen and launches on October 12. " * 8,
            "Birch is owned by Luis Ortega and launches on November 8. " * 8,
        )
    ]
    receiver = bind(model, tokenizer, backend="hf", max_new_tokens=64)
    requests = ["Who owns Cedar?", "When does Birch launch?"]

    await transfer(senders, receiver, requests)

    for request in requests:
        inputs = receiver.pop()
        output = model.generate(
            **inputs,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
        receiver.update(inputs=inputs, generation=output)
        print(f"{request} {tokenizer.decode(output.sequences[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    asyncio.run(main())
