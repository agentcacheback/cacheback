import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_XET"] = "1"
import torch
import rcc
from vllm import LLM

folder = Path(sys.argv[1])
pin = sys.argv[2]
config = json.loads((folder.parent / "config.json").read_text())
folder.mkdir()
report = {"complete": False, "config": config, "gpu": torch.cuda.get_device_name(),
          "versions": {p: importlib.metadata.version(p) for p in ("vllm", "torch", "transformers", "rclc")},
          "calls": []}

def save():
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    (folder / "result.json").write_text(json.dumps(report, indent=2) + "\n")

def measured(fn, name):
    def call(*args, **kwargs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = fn(*args, **kwargs)
        torch.cuda.synchronize()
        entry = {"phase": name, "seconds": time.perf_counter() - started}
        if name == "generate":
            prompts = kwargs.get("prompts", args[1] if len(args) > 1 else [])
            if isinstance(prompts, dict):
                prompts = [prompts]
            entry["input_positions"] = [len(p.get("prompt_embeds", p.get("prompt_token_ids", []))) for p in prompts]
            entry["output_ids"] = [o.outputs[0].token_ids for o in output]
        report["calls"].append(entry)
        save()
        return output
    return call

LLM.generate = measured(LLM.generate, "generate")
rcc.transfer = measured(rcc.transfer, "transfer")
from rcc import vllm
vllm.prefill_state = measured(vllm.prefill_state, "capture")
from examples.vllm_transfer import run
save()
try:
    tokens = run(allow_unstable=pin == "0.26.0", model=config["model"], revision=config["revision"])
    assert len(tokens) == 68 and all(len(t) == 4 for t in tokens)
    report["output_ids"] = tokens
    report["complete"] = True
finally:
    save()
print(json.dumps({"complete": report["complete"], "calls": len(report["calls"]),
                  "peak_allocated_bytes": report["peak_allocated_bytes"]}))
