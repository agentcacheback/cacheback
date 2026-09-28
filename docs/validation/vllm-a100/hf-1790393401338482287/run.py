import json
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from rcc import bind, transfer

run = Path(sys.argv[1])
config = json.loads((run / "config.json").read_text())
device = config["device"]
torch.set_num_threads(4)
dtype = torch.float32 if device == "cpu" else (
    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)
started = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
model = AutoModelForCausalLM.from_pretrained(
    config["model"], revision=config["revision"], torch_dtype=dtype,
    attn_implementation="sdpa",
).to(device).eval()
if device == "cuda":
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
load_seconds = time.perf_counter() - started


def clock():
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def save():
    if device == "cuda":
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    (run / "results.json").write_text(json.dumps(result, indent=2) + "\n")


documents = [
    "Cedar is owned by Maya Chen and launches on October 12. " * 8,
    "Birch is owned by Luis Ortega and launches on November 8. " * 8,
]
requests = ["Who owns Cedar?", "When does Birch launch?"]
result = {
    "complete": False, "config": config, "torch": torch.__version__,
    "transformers": transformers.__version__, "dtype": str(dtype),
    "resolved_model_revision": getattr(model.config, "_commit_hash", None),
    "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
    "model_load_seconds": load_seconds, "documents": documents, "cases": [],
}
save()
started = clock()
senders = [bind(model, tokenizer, prompt=document, backend="hf") for document in documents]
result["sender_prefill_seconds"] = clock() - started
result["sender_positions"] = [agent.sender_state().input_embeds.shape[0] for agent in senders]
receiver = bind(model, tokenizer, backend="hf", max_new_tokens=config["max_new_tokens"])
started = clock()
transfer(senders, receiver, requests, ratio=config["ratio"])
result["transfer_seconds"] = clock() - started
assert len(receiver) == len(requests)
assert result["sender_positions"] == [
    s.sender_state().past_key_values.get_seq_length() for s in senders
]
save()
for request, expected in zip(requests, ["Maya Chen", "November 8"], strict=True):
    started = clock()
    inputs = receiver.pop()
    preparation_seconds = clock() - started
    started = clock()
    tokens = model.generate(
        **inputs, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    seconds = clock() - started
    text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    result["cases"].append({
        "request": request, "answer": text, "token_ids": tokens[0].tolist(),
        "receiver_input_positions": inputs["inputs_embeds"].shape[1],
        "input_preparation_seconds": preparation_seconds,
        "receiver_prefill_and_generation_seconds": seconds,
        "expected_text": expected, "contains_expected_text": expected.lower() in text.lower(),
    })
    save()
    print(f"{request}\n{text}\nReceiver: {seconds:.3f}s", flush=True)
assert not receiver
result["complete"] = True
save()
print(f"Saved {run / 'results.json'}", flush=True)
