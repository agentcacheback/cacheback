"""Run an HF state handoff on CPU; downloads Qwen3-0.6B on first use."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rcc import Delivery, SenderState, transfer


def main() -> None:
    """Let native agent code prefill, transfer state, then continue at each receiver."""
    torch.set_num_threads(4)
    checkpoint = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, attn_implementation="sdpa").eval()
    documents = [
        "Cedar is owned by Maya Chen and launches on October 12. " * 8,
        "Birch is owned by Luis Ortega and launches on November 8. " * 8,
    ]
    senders = []
    with torch.inference_mode():
        for document in documents:
            ids = tokenizer(document, return_tensors="pt")["input_ids"]
            output = model.model(input_ids=ids, use_cache=True)
            senders.append(
                SenderState(
                    model,
                    output.past_key_values,
                    model.get_input_embeddings()(ids)[0],
                    tokenizer=tokenizer,
                    token_ids=ids[0],
                )
            )
    first: list[Delivery] = []
    second: list[Delivery] = []
    requests = ["Who owns Cedar?", "When does Birch launch?"]
    original_lengths = [state.past_key_values.get_seq_length() for state in senders]

    transfer(senders, [first.append, second.append], requests)

    assert len(first) == len(second) == 2
    assert original_lengths == [state.past_key_values.get_seq_length() for state in senders]
    with torch.inference_mode():
        for index, inbox in enumerate((first, second), 1):
            for delivery in inbox:
                inputs = delivery.for_hf(model.get_input_embeddings().weight)
                received = model.model(**inputs, use_cache=True)
                query = tokenizer(f"\n{delivery.request}\nAnswer:", return_tensors="pt")
                output = model(
                    input_ids=query["input_ids"],
                    past_key_values=received.past_key_values,
                    use_cache=True,
                )
                tokens = []
                for _ in range(8):
                    token = output.logits[:, -1:].argmax(-1)
                    tokens.append(int(token.item()))
                    output = model(
                        input_ids=token, past_key_values=output.past_key_values, use_cache=True
                    )
                print(f"Receiver {index}, {delivery.request}: {tokenizer.decode(tokens)!r}")
    print("Four deliveries and receiver continuations complete.")


if __name__ == "__main__":
    main()
