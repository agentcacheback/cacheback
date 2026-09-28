"""Hand off between two existing agents using vLLM 0.11.1 on one CUDA GPU."""

import asyncio
import os
from typing import Any

import rcc


async def handoff(llm: Any) -> str:
    """Transfer notes, generate natively and record the receiver's continuation."""
    sender = rcc.bind(
        llm,
        backend="vllm",
        prompt="Cedar is owned by Maya Chen and launches October 12. " * 8,
    )
    receiver = rcc.bind(llm, backend="vllm", max_new_tokens=64)
    await rcc.transfer(sender, receiver, "Who owns Cedar?")
    inputs = receiver.pop()
    inputs["sampling_params"].temperature = 0
    output = llm.generate(**inputs)
    receiver.update(inputs=inputs, generation=output)
    return str(output[0].outputs[0].text)


def main() -> None:
    """Create the caller-owned engine with capture enabled, then run the handoff."""
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    from vllm import LLM

    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        tensor_parallel_size=1,
        enable_prefix_caching=False,
        enforce_eager=True,
        enable_prompt_embeds=True,
        gpu_memory_utilization=0.45,
        kv_transfer_config={
            "kv_connector": "RCCCaptureConnector",
            "kv_connector_module_path": "rcc.capture.connector",
            "kv_role": "kv_both",
        },
    )
    print(asyncio.run(handoff(llm)))


if __name__ == "__main__":
    main()
