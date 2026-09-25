"""Check both engine layouts and the adapter with real tiny Qwen3 weights and caches."""

from types import SimpleNamespace as Namespace
from typing import Any

import pytest
import torch
from tests.conftest import tiny_config
from transformers import Qwen3ForCausalLM

from rcc import transfer
from rcc.capture import connector
from rcc.selectors.core.kernels import cache_kv
from rcc.vllm import sender_from_vllm


def test_vllm_capture_layouts_aliasing_and_explicit_unstable_opt_in(monkeypatch: Any) -> None:
    with torch.random.fork_rng():
        torch.manual_seed(19)
        model = Qwen3ForCausalLM(tiny_config("sdpa")).eval()
    config = model.config
    ids = torch.arange(18, dtype=torch.long)
    with torch.inference_mode():
        original = model.model(input_ids=ids[None], use_cache=True).past_key_values
    named = dict(model.named_parameters())

    @torch.inference_mode()
    def generate(prompts: list[dict[str, torch.Tensor]], sampling: Any, **kwargs: Any) -> list[Any]:
        results = []
        for prompt in prompts:
            output = model(inputs_embeds=prompt["prompt_embeds"][None], use_cache=True)
            tokens = []
            for _ in range(4):
                token = output.logits[:, -1:].argmax(-1)
                tokens.append(int(token.item()))
                output = model(
                    input_ids=token, past_key_values=output.past_key_values, use_cache=True
                )
            results.append(Namespace(outputs=[Namespace(token_ids=tokens)]))
        return results

    for i in range(config.num_hidden_layers):
        base = f"model.layers.{i}"
        named[f"{base}.self_attn.qkv_proj.weight"] = torch.cat(
            [named.pop(f"{base}.self_attn.{part}_proj.weight") for part in ("q", "k", "v")]
        )
        named[f"{base}.mlp.gate_up_proj.weight"] = torch.cat(
            [named.pop(f"{base}.mlp.{part}_proj.weight") for part in ("gate", "up")]
        )
    splits = [config.num_attention_heads * config.head_dim] + [
        config.num_key_value_heads * config.head_dim
    ] * 2
    resident = Namespace(
        named_parameters=lambda: named.items(),
        model=Namespace(
            layers=[
                Namespace(self_attn=Namespace(qkv_proj=Namespace(output_sizes=splits)))
                for _ in cache_kv(original)
            ]
        ),
    )
    llm = Namespace(
        llm_engine=Namespace(
            vllm_config=Namespace(model_config=Namespace(hf_config=config)),
            model_executor=Namespace(
                driver_worker=Namespace(model_runner=Namespace(model=resident))
            ),
        ),
        get_tokenizer=lambda: None,
        generate=generate,
    )
    layer_names = [f"model.layers.{i}.self_attn.attn" for i in range(config.num_hidden_layers)]
    group = Namespace(
        layer_names=layer_names,
        kv_cache_spec=Namespace(
            block_size=16,
            num_kv_heads=config.num_key_value_heads,
            head_size=config.head_dim,
            dtype=torch.float32,
        ),
    )
    cache_config = Namespace(kv_cache_groups=[group])

    class Base:
        def __init__(self, *args: Any) -> None:
            self.metadata = None

        def _get_connector_metadata(self) -> Any:
            return self.metadata

    class HMA:
        pass

    base_module = Namespace(KVConnectorBase_V1=Base, SupportsHMA=HMA, KVConnectorMetadata=object)
    import_module = connector.importlib.import_module
    monkeypatch.setattr(
        connector.importlib,
        "import_module",
        lambda name, *args, **kwargs: (
            Namespace(VLLM_ENABLE_V1_MULTIPROCESSING=False)
            if name == "vllm.envs"
            else base_module
            if name == "vllm.distributed.kv_transfer.kv_connector.v1.base"
            else import_module(name, *args, **kwargs)
        ),
    )
    for pin in ("0.11.1", "0.26.0"):
        monkeypatch.setattr(connector, "version", lambda name, pin=pin: pin)
        if pin == "0.26.0":
            with pytest.raises(RuntimeError, match="allow_unstable"):
                sender_from_vllm(llm, "missing", ids)
        runtime = Namespace(
            cache_config=Namespace(enable_prefix_caching=False),
            parallel_config=Namespace(tensor_parallel_size=1),
            kv_transfer_config=Namespace(
                kv_connector_extra_config={"allow_unstable": pin == "0.26.0"}
            ),
        )
        cls = connector._build_connector()
        if pin == "0.26.0":
            with pytest.warns(UserWarning, match="unstable"):
                capture = cls(runtime, None, cache_config)
        else:
            capture = cls(runtime, None, cache_config)
        pages = {}
        for name, (key, value) in zip(layer_names, cache_kv(original), strict=True):
            # Engine pages are physically shuffled; each version has its own tensor layout.
            packed = torch.zeros(3, config.num_key_value_heads, 16, 2 * config.head_dim)
            rows = torch.cat((key, value), dim=-1)[0]
            packed[2] = rows[:, :16]
            packed[1, :, :2] = rows[:, 16:]
            if pin == "0.11.1":
                packed = (
                    torch.stack(packed.split(config.head_dim, dim=-1))
                    .permute(0, 1, 3, 2, 4)
                    .contiguous()
                )
            else:
                packed = packed.transpose(1, 2).contiguous().transpose(1, 2)
            pages[name] = packed
        untouched = {name: tensor.clone() for name, tensor in pages.items()}
        capture.register_kv_caches(pages)
        connector.request_capture(pin)
        new = Namespace(req_id=pin, prompt_token_ids=ids.tolist(), block_ids=([2],))
        step = Namespace(
            scheduled_new_reqs=[new], scheduled_cached_reqs=None, num_scheduled_tokens={pin: 16}
        )
        capture.metadata = capture.build_connector_meta(step)
        capture.wait_for_save()
        assert pin not in connector._EXTRACTED
        step.scheduled_new_reqs = []
        step.scheduled_cached_reqs = Namespace(
            req_ids=[pin], resumed_req_ids=set(), new_block_ids=[([1],)]
        )
        step.num_scheduled_tokens = {pin: 2}
        capture.metadata = capture.build_connector_meta(step)
        capture.wait_for_save()
        with pytest.raises(ValueError, match="length"):
            if pin == "0.26.0":
                with pytest.warns(UserWarning):
                    sender_from_vllm(llm, pin, ids[:-1], allow_unstable=True)
            else:
                sender_from_vllm(llm, pin, ids[:-1])
        assert pin in connector._EXTRACTED
        if pin == "0.26.0":
            with pytest.warns(UserWarning, match="unstable"):
                state = sender_from_vllm(llm, pin, ids, inherited_positions=3, allow_unstable=True)
        else:
            state = sender_from_vllm(llm, pin, ids, inherited_positions=3)
        assert state.inherited_positions == 3
        assert pin not in connector._EXTRACTED
        assert (
            state.model.get_input_embeddings().weight.data_ptr()
            == model.get_input_embeddings().weight.data_ptr()
        )
        assert state.model.get_input_embeddings().weight is not model.get_input_embeddings().weight
        for actual, expected in zip(
            cache_kv(state.past_key_values), cache_kv(original), strict=True
        ):
            assert all(torch.equal(a, e) for a, e in zip(actual, expected, strict=True))
        inbox = []
        transfer(state, inbox.append, torch.tensor([[20, 21]]), budget=8)
        assert len(inbox) == 1 and inbox[0].messages[0].positions == 8
        assert all(torch.equal(pages[name], untouched[name]) for name in pages)
        assert model.get_input_embeddings().weight.requires_grad
        from examples.vllm_transfer import _check_latents

        with pytest.warns(UserWarning, match="untested"):
            continuations = _check_latents(
                llm, None, [state, state], [torch.tensor([[20, 21]]), torch.tensor([[22, 23]])]
            )
        assert len(continuations) == 64
    planner = connector._Planner(1)
    planner.start("resume", 18, ([1],))
    assert planner.advance({"resume": 4}) == []
    planner.reset("resume", ([2, 1],))
    assert planner.advance({"resume": 18})[0].block_ids == ((2, 1),)
