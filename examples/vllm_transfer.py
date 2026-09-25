"""Real Qwen sender capture, broadcast handoff and vLLM receiver continuation."""

import argparse
import os
from typing import Any

import torch

from rcc import Delivery, SenderState, rollout, transfer
from rcc.capture.connector import require_runtime
from rcc.latent import LatentContext
from rcc.selectors.core.kernels import cache_kv
from rcc.vllm import discard_capture, request_capture, sender_from_vllm


@torch.inference_mode()
def _latent_prompts(
    state: SenderState, inboxes: list[list[Delivery]]
) -> list[dict[str, torch.Tensor]]:
    """Prepare both receivers, with two local latent steps on the second receiver."""
    model = state.model
    weight = model.get_input_embeddings().weight
    prompts = []
    for receiver_index, inbox in enumerate(inboxes):
        for delivery in inbox:
            query = model.get_input_embeddings()(state.request_ids(delivery.request))[0]
            rows = torch.cat((delivery.for_vllm(weight)["prompt_embeds"].to(weight), query))
            if receiver_index == 1:
                output = model.model(inputs_embeds=rows[None], use_cache=True)
                receiver = SenderState(model, output.past_key_values, rows)
                rows = rollout(receiver, steps=2).input_embeds
            assert torch.isfinite(rows).all(), "receiver prompt contains nonfinite rows"
            prompts.append({"prompt_embeds": rows.cpu()})
    return prompts


def _check_latents(
    llm: Any, sampling: Any, senders: list[SenderState], requests: list[str | torch.Tensor]
) -> list[list[int]]:
    """Check every latent mode and budget through dense/mixed delivery and real decoding."""
    snapshots = [[(k.clone(), v.clone()) for k, v in cache_kv(s.past_key_values)] for s in senders]
    contexts: tuple[LatentContext, ...] = (
        "full",
        "full_with_request",
        "selected",
        "selected_with_request",
    )
    results = []
    for context in contexts:
        for bounded in (False, True):
            outputs, encoded_prompts = [], []
            for representation in ("embeddings", "token_ids+continuous"):
                inboxes: list[list[Delivery]] = [[], []]
                transfer(
                    senders,
                    [inbox.append for inbox in inboxes],
                    requests,
                    latent_steps=2,
                    latent_context=context,
                    representation=representation,
                    **({"budget": 32} if bounded else {"ratio": 4}),
                )
                assert [len(inbox) for inbox in inboxes] == [2, 2]
                assert all(a is b for a, b in zip(*inboxes, strict=True))
                for delivery in inboxes[0]:
                    for state, message in zip(senders, delivery.messages, strict=True):
                        length = state.input_embeds.shape[0] + 2
                        inherited = state.inherited_positions
                        count = (
                            min(32, length)
                            if bounded
                            else inherited + (length - inherited + 3) // 4
                        )
                        assert message.positions == count
                        if message.token_ids is not None:
                            assert bool((message.token_ids[-2:] == -1).all())
                prompts = _latent_prompts(senders[0], inboxes)
                output = llm.generate(prompts, sampling, use_tqdm=False)
                tokens = [item.outputs[0].token_ids for item in output]
                assert len(tokens) == 4 and all(len(sequence) == 4 for sequence in tokens)
                encoded_prompts.append(prompts)
                outputs.append(tokens)
                results.extend(tokens)
            assert all(
                torch.equal(dense["prompt_embeds"], mixed["prompt_embeds"])
                for dense, mixed in zip(*encoded_prompts, strict=True)
            ), f"{context}: dense and mixed receiver inputs differ"
            assert outputs[0] == outputs[1], f"{context}: dense and mixed continuations differ"
    for state, snapshot in zip(senders, snapshots, strict=True):
        for actual, expected in zip(cache_kv(state.past_key_values), snapshot, strict=True):
            assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    return results


def run(*, allow_unstable: bool = False) -> list[list[int]]:
    """Check native-token parity and every latent handoff mode in one engine session."""
    pin = require_runtime(allow_unstable)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        tensor_parallel_size=1,
        dtype="bfloat16",
        enable_prefix_caching=False,
        enforce_eager=True,
        enable_prompt_embeds=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=64,
        max_model_len=2048,
        gpu_memory_utilization=0.45,
        max_num_seqs=4,
        kv_transfer_config={
            "kv_connector": "RCCCaptureConnector",
            "kv_role": "kv_both",
            "kv_connector_module_path": "rcc.capture.connector",
            "kv_connector_extra_config": {"allow_unstable": allow_unstable},
        },
        **({"attention_backend": "FLASH_ATTN"} if pin == "0.26.0" else {}),
    )
    tokenizer = llm.get_tokenizer()
    documents = [
        "Cedar is owned by Maya Chen and launches October 12. " * 12,
        "Birch is owned by Luis Ortega and launches November 8. " * 12,
    ]
    senders = []
    for index, document in enumerate(documents):
        ids = tokenizer.encode(document)
        external_id = f"sender-{index}"
        capture_id = external_id
        try:
            if pin == "0.11.1":
                request_capture(capture_id)
            internal_id = llm.llm_engine.add_request(
                external_id,
                {"prompt_token_ids": ids},
                SamplingParams(max_tokens=1, temperature=0, ignore_eos=True),
            )
            if pin == "0.26.0":
                capture_id = internal_id
                request_capture(capture_id)
            while llm.llm_engine.has_unfinished_requests():
                llm.llm_engine.step()
            senders.append(
                sender_from_vllm(
                    llm,
                    capture_id,
                    torch.tensor(ids, dtype=torch.long),
                    allow_unstable=allow_unstable,
                )
            )
        finally:
            discard_capture(capture_id)
    inboxes: list[list[Delivery]] = [[], []]
    transfer(
        senders,
        [inbox.append for inbox in inboxes],
        ["Who owns Cedar?", "When does Birch launch?"],
        ratio=4,
    )
    assert [len(inbox) for inbox in inboxes] == [2, 2]
    weight = senders[0].model.get_input_embeddings().weight
    prompts, native_prompts = [], []
    for inbox in inboxes:
        for delivery in inbox:
            query_ids = tokenizer.encode(delivery.request, add_special_tokens=False)
            with torch.inference_mode():
                query_rows = torch.nn.functional.embedding(
                    torch.tensor(query_ids, device=weight.device), weight
                ).cpu()
            rows = delivery.for_vllm(weight)["prompt_embeds"]
            prompts.append({"prompt_embeds": torch.cat((rows, query_rows))})
            reference_ids = []
            for state, message in zip(senders, delivery.messages, strict=True):
                source = state.input_embeds.cpu()
                for row in message.materialize(weight).cpu():
                    matches = (source == row).all(dim=1).nonzero()
                    assert matches.numel(), "handoff contains a row outside its sender"
                    reference_ids.append(int(state.token_ids[int(matches[0].item())]))
            native_prompts.append({"prompt_token_ids": reference_ids + query_ids})
    sampling = SamplingParams(max_tokens=4, temperature=0, ignore_eos=True)
    actual = [item.outputs[0].token_ids for item in llm.generate(prompts, sampling, use_tqdm=False)]
    reference = [
        item.outputs[0].token_ids for item in llm.generate(native_prompts, sampling, use_tqdm=False)
    ]
    assert actual == reference, "transport and native token-input continuations differ"
    assert len(actual) == 4 and all(len(tokens) == 4 for tokens in actual)
    return actual + _check_latents(
        llm, sampling, senders, ["Who owns Cedar?", "When does Birch launch?"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-unstable", action="store_true")
    options = parser.parse_args()
    print(f"{len(run(allow_unstable=options.allow_unstable))} receiver continuations passed.")
