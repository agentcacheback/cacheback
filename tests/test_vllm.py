"""Check both engine layouts and the adapter with real tiny Qwen3 weights and caches."""

import asyncio
from functools import partial
from types import SimpleNamespace as Namespace
from typing import Any

import pytest
import torch
from tests.conftest import tiny_config
from transformers import Qwen3ForCausalLM

from rclc import latent_mass, transfer_sync
from rclc._cache import cache_kv
from rclc.capture import connector
from rclc.vllm import binding_model, prefill_state, sender_from_vllm


def test_vllm_capture_layouts_aliasing_and_explicit_unstable_opt_in(
    monkeypatch: Any, senders: list[Any]
) -> None:
    with torch.random.fork_rng():
        torch.manual_seed(19)
        model = Qwen3ForCausalLM(tiny_config("sdpa")).eval()
    config = model.config
    ids = torch.arange(18, dtype=torch.long)
    with torch.inference_mode():
        original = model.model(input_ids=ids[None], use_cache=True).past_key_values
    named = dict(model.named_parameters())

    @torch.inference_mode()
    def generate(prompts: Any, sampling_params: Any = None, **kwargs: Any) -> list[Any]:
        if isinstance(prompts, dict):
            prompts = [prompts]
        results = []
        for prompt in prompts:
            output = model(inputs_embeds=prompt["prompt_embeds"][None], use_cache=True)
            tokens = []
            for _ in range(getattr(sampling_params, "max_tokens", 4)):
                token = output.logits[:, -1:].argmax(-1)
                tokens.append(int(token.item()))
                output = model(
                    input_ids=token, past_key_values=output.past_key_values, use_cache=True
                )
            results.append(Namespace(finished=True, outputs=[Namespace(token_ids=tokens)]))
        return results

    for i in range(config.num_hidden_layers):
        base = f"model.layers.{i}"
        named[f"{base}.self_attn.qkv_proj.weight"] = torch.cat(
            [named.pop(f"{base}.self_attn.{part}_proj.weight") for part in ("q", "k", "v")]
        )
        named[f"{base}.mlp.gate_up_proj.weight"] = torch.cat(
            [named.pop(f"{base}.mlp.{part}_proj.weight") for part in ("gate", "up")]
        )
    resident = Namespace(named_parameters=lambda: named.items())
    llm = Namespace(
        llm_engine=Namespace(
            vllm_config=Namespace(
                model_config=Namespace(
                    hf_config=config,
                    max_model_len=512,
                    enable_prompt_embeds=True,
                    quantization=None,
                ),
                parallel_config=Namespace(tensor_parallel_size=1, pipeline_parallel_size=1),
                cache_config=Namespace(enable_prefix_caching=False, cache_dtype="auto"),
                kv_transfer_config=Namespace(kv_connector="RCCCaptureConnector"),
            ),
            engine_core=Namespace(
                engine_core=Namespace(
                    model_executor=Namespace(
                        driver_worker=Namespace(
                            worker=Namespace(model_runner=Namespace(model=resident))
                        )
                    )
                )
            ),
            has_unfinished_requests=lambda: False,
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
            else Namespace(SamplingParams=Namespace)
            if name == "vllm"
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
            cache_config=Namespace(enable_prefix_caching=False, cache_dtype="auto"),
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
        engine_config = llm.llm_engine.vllm_config
        engine_config.kv_transfer_config.kv_connector_extra_config = {
            "allow_unstable": pin == "0.26.0"
        }
        for node, field, value, message in (
            (engine_config.parallel_config, "tensor_parallel_size", 2, "parallel"),
            (engine_config.parallel_config, "pipeline_parallel_size", 2, "parallel"),
            (engine_config.model_config, "quantization", "fp8", "unquantized"),
            (engine_config.cache_config, "enable_prefix_caching", True, "prefix_caching"),
            (engine_config.cache_config, "cache_dtype", "fp8", "cache_dtype='auto'"),
        ):
            with monkeypatch.context() as patch:
                patch.setattr(node, field, value)
                with pytest.raises(ValueError, match=message):
                    binding_model(llm, None, None, pin == "0.26.0")
                with pytest.raises(ValueError, match=message):
                    cls(engine_config, None, cache_config)
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
        for dtype, message in (
            (torch.float8_e4m3fn, "unsupported page dtype"),
            (torch.float16, "differs from group dtype"),
        ):
            with pytest.raises(ValueError, match=message):
                capture.register_kv_caches({name: page.to(dtype) for name, page in pages.items()})
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
        for finish in (capture.request_finished, capture.request_finished_all_groups):
            connector.request_capture("unfinished")
            capture._planner.start("unfinished", 18, ([2],))
            for request_id in (pin, "unfinished"):
                assert finish(Namespace(request_id=request_id), ([2],)) == (False, None)
            assert not capture._planner.tracks("unfinished")
            assert "unfinished" not in connector._REQUESTED
            assert pin in connector._EXTRACTED
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
        transfer_sync(state, inbox.append, torch.tensor([[20, 21]]), budget=8)
        assert len(inbox) == 1 and inbox[0].messages[0].positions == 8
        assert all(torch.equal(pages[name], untouched[name]) for name in pages)
        assert model.get_input_embeddings().weight.requires_grad
        from scripts.validate_vllm import _check_latents

        with pytest.warns(UserWarning, match="untested"):
            continuations = asyncio.run(
                _check_latents(
                    llm, None, [state, state], [torch.tensor([[20, 21]]), torch.tensor([[22, 23]])]
                )
            )
        assert len(continuations) == 64
        engine = llm.llm_engine
        pending: list[tuple[str, dict[str, Any]]] = []
        captured_rows: list[torch.Tensor] = []

        def add_request(
            request_id: str,
            prompt: dict[str, Any],
            sampling: Any,
            pin: str = pin,
            pending: Any = pending,
        ) -> str | None:
            assert sampling.max_tokens == 1 and sampling.temperature == 0 and sampling.ignore_eos
            internal_id = request_id + "-internal" if pin == "0.26.0" else request_id
            pending.append((internal_id, prompt))
            return internal_id if pin == "0.26.0" else None

        @torch.inference_mode()
        def step(
            pin: str = pin,
            pending: Any = pending,
            captured_rows: Any = captured_rows,
            capture: Any = capture,
        ) -> list[Any]:
            request_id, prompt = pending.pop()
            token_ids = prompt.get("prompt_token_ids")
            rows = prompt.get("prompt_embeds")
            if rows is None:
                rows = model.get_input_embeddings()(torch.tensor(token_ids))
            captured_rows.append(rows.clone())
            past = model(inputs_embeds=rows[None], use_cache=True).past_key_values
            length = rows.shape[0]
            blocks = (length + 15) // 16
            live_pages = {}
            for name, (key, value) in zip(layer_names, cache_kv(past), strict=True):
                combined = torch.zeros(
                    (blocks + 1) * 16, config.num_key_value_heads, 2 * config.head_dim
                )
                combined[16 : 16 + length] = torch.cat((key, value), dim=-1)[0].transpose(0, 1)
                packed = combined.reshape(blocks + 1, 16, config.num_key_value_heads, -1)
                live_pages[name] = (
                    torch.stack(packed.split(config.head_dim, dim=-1)).contiguous()
                    if pin == "0.11.1"
                    else packed.transpose(1, 2)
                )
            capture.register_kv_caches(live_pages)
            request = Namespace(
                req_id=request_id,
                prompt_token_ids=token_ids,
                prompt_embeds=rows if token_ids is None else None,
                prompt_is_token_ids=token_ids is not None,
                block_ids=(list(range(1, blocks + 1)),),
            )
            scheduled = Namespace(
                scheduled_new_reqs=[request],
                scheduled_cached_reqs=None,
                num_scheduled_tokens={request_id: length},
            )
            capture.metadata = capture.build_connector_meta(scheduled)
            capture.wait_for_save()
            return [Namespace(request_id=request_id, finished=True, num_cached_tokens=0)]

        engine.add_request = add_request
        engine.step = step
        engine.has_unfinished_requests = lambda pending=pending: bool(pending)
        engine.abort_request = lambda ids, pending=pending: pending.clear()
        with monkeypatch.context() as patch:
            patch.setattr(engine, "add_request", lambda *args: pytest.fail("submitted bad prompt"))
            for bad_ids, rows, message in (
                (ids + config.vocab_size, None, "vocabulary"),
                (ids - 1, None, "vocabulary"),
                (ids, torch.zeros(len(ids) - 1, config.hidden_size), "one"),
            ):
                with pytest.raises(ValueError, match=message):
                    prefill_state(llm, state.model, None, bad_ids, input_embeds=rows)
        from scripts.validate_vllm import _check_bind

        if pin == "0.26.0":
            with pytest.warns(UserWarning, match="unstable"):
                returned = asyncio.run(_check_bind(llm, ids, allow_unstable=True))
        else:
            returned = asyncio.run(_check_bind(llm, ids))
        with torch.inference_mode():
            expected = model(
                inputs_embeds=returned.input_embeds[None], use_cache=True
            ).past_key_values
        for actual, reference in zip(
            cache_kv(returned.past_key_values), cache_kv(expected), strict=True
        ):
            torch.testing.assert_close(actual, reference)
        assert len(captured_rows) == 5
        assert torch.equal(captured_rows[0], model.get_input_embeddings()(ids))
        assert not connector._REQUESTED and not connector._EXTRACTED
        from rclc import bind

        tokenizer = senders[0].tokenizer
        tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}?"
        llm.get_tokenizer = lambda tokenizer=tokenizer: tokenizer
        history = [{"role": "user", "content": "Cedar launches " * 10}]
        if pin == "0.26.0":
            with pytest.warns(UserWarning, match="unstable"):
                agent = bind(llm, backend="vllm", messages=history, allow_unstable=True)
        else:
            agent = bind(llm, backend="vllm", messages=history)
        assert len(captured_rows) == 5
        first = agent.sender_state()
        assert agent.sender_state() is first and len(captured_rows) == 6
        history.append({"role": "assistant", "content": "November"})
        changed = agent.sender_state()
        assert (
            len(captured_rows) == 7 and changed.input_embeds.shape[0] > first.input_embeds.shape[0]
        )
        external_id = "already-captured"
        internal_id = engine.add_request(
            external_id,
            {"prompt_token_ids": ids.tolist()},
            Namespace(max_tokens=1, temperature=0, ignore_eos=True),
        )
        capture_id = internal_id or external_id
        connector.request_capture(capture_id)
        engine.step()
        with monkeypatch.context() as patch:
            patch.setattr(
                engine, "add_request", lambda *args: pytest.fail("replayed a live capture")
            )
            agent.update(ids[None], request_id=capture_id)
        assert len(captured_rows) == 8 and agent.sender_state().token_ids.tolist() == ids.tolist()
        assert not connector._REQUESTED and not connector._EXTRACTED
        before = agent.sender_state()
        aborted = []

        def abort(request_ids: list[str], aborted: Any = aborted, pending: Any = pending) -> None:
            aborted.extend(request_ids)
            pending.clear()

        with monkeypatch.context() as patch:
            patch.setattr(engine, "abort_request", abort)
            patch.setattr(
                engine, "step", lambda: (_ for _ in ()).throw(RuntimeError("step failed"))
            )
            with pytest.raises(RuntimeError, match="step failed"):
                agent.update(ids[None])
        assert len(aborted) == 1 and not aborted[0].endswith("-internal")
        assert agent.sender_state() is before and not pending
        assert not connector._REQUESTED and not connector._EXTRACTED
        with monkeypatch.context() as patch:
            patch.setattr(engine, "has_unfinished_requests", lambda: True)
            with pytest.raises(RuntimeError, match="idle"):
                bind(llm, backend="vllm", allow_unstable=pin == "0.26.0")
        with monkeypatch.context() as patch:
            patch.setattr(engine.vllm_config.model_config, "enable_prompt_embeds", False)
            with pytest.raises(ValueError, match="enable_prompt_embeds"):
                bind(llm, backend="vllm")
            from rclc import check

            assert any("enable_prompt_embeds" in issue for issue in check(agent)["issues"])

        receiver = bind(llm, backend="vllm", max_new_tokens=4, allow_unstable=pin == "0.26.0")
        for source, target in ((agent, receiver), (receiver, agent)):
            source_state = source.sender_state()
            length = len(source_state.input_embeds)
            transfer_sync(
                source, target, "Who owns Cedar ?", ratio=1, reasoning=partial(latent_mass, steps=2)
            )
            inputs = target.pop(max_new_tokens=4)
            with pytest.raises(ValueError, match="discard_pending"):
                target.pop()
            output = llm.generate(**inputs)
            target.update(inputs=inputs, generation=output)
            continued = target.sender_state()
            assert continued.inherited_positions == length + 2
            assert source_state.past_key_values.get_seq_length() == length
            asyncio.run(target.append(torch.tensor([[30, 31]])))
            appended = target.sender_state()
            assert torch.equal(appended.input_embeds[:-2], continued.input_embeds)
            assert appended.inherited_positions == continued.inherited_positions
            assert appended.token_ids[-2:].tolist() == [30, 31]
            assert continued.past_key_values.get_seq_length() == len(continued.input_embeds)
            with torch.inference_mode():
                expected = model(
                    inputs_embeds=appended.input_embeds[None], use_cache=True
                ).past_key_values
            for actual, reference in zip(
                cache_kv(appended.past_key_values), cache_kv(expected), strict=True
            ):
                torch.testing.assert_close(actual, reference)
        assert not connector._REQUESTED and not connector._EXTRACTED

    planner = connector._Planner()
    with pytest.raises(RuntimeError, match="expected 1"):
        planner.start("multiple-groups", 18, ([1], [2]))
    planner.start("resume", 18, ([1],))
    assert planner.advance({"resume": 4}) == []
    planner.reset("resume", ([2, 1],))
    assert planner.advance({"resume": 18})[0].block_ids == (2, 1)
