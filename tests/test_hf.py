"""A complete native Qwen handoff through the convenience bindings."""

import asyncio
import copy
from functools import partial
from typing import Any

import pytest
import torch
from examples.existing_agents import handoff

from rclc import (
    Agent,
    Delivery,
    HFReceiver,
    SenderState,
    bind,
    latent_mass,
    sender_from_hf,
    transfer_sync,
)


@torch.inference_mode()
def test_hf_agents_transfer_and_generate(senders: list[SenderState], monkeypatch: Any) -> None:
    model, tokenizer = senders[0].model, senders[0].tokenizer
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}?"
    ids = torch.arange(10, 90)[None]
    sender = sender_from_hf(model, tokenizer, ids)
    with monkeypatch.context() as patch:
        patch.setattr(model.model, "forward", lambda **kw: pytest.fail("recomputed cached prompt"))
        bound = sender_from_hf(model, tokenizer, ids, past_key_values=sender.past_key_values)
    assert bound.past_key_values is sender.past_key_values
    other = sender_from_hf(model, tokenizer, "Birch launches " * 32)
    receivers = [HFReceiver(model, tokenizer), HFReceiver(copy.deepcopy(model), tokenizer)]
    reference: list[Delivery] = []
    requests = ["Who owns Cedar ?", "When Birch launches ?"]
    with monkeypatch.context() as patch:
        patch.setattr(model, "generate", lambda **kw: pytest.fail("transfer generated an answer"))
        transfer_sync(
            [bound, other],
            [*receivers, reference.append],
            requests,
            reasoning=partial(latent_mass, steps=2),
        )
    assert all(len(receiver) == 2 for receiver in receivers)
    assert sender.past_key_values.get_seq_length() == ids.shape[1]
    weight = model.get_input_embeddings().weight
    for delivery in reference:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": delivery.request}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
        expected = torch.cat(
            (
                delivery.for_hf(weight)["inputs_embeds"],
                model.get_input_embeddings()(prompt_ids),
            ),
            dim=1,
        )
        reference_tokens = model.generate(
            inputs_embeds=expected,
            attention_mask=torch.ones(expected.shape[:2], dtype=torch.long),
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=0,
        )
        for receiver in receivers:
            with pytest.raises(ValueError, match="no context space"):
                receiver.pop(prompt=torch.ones((1, 512), dtype=torch.long))
            with pytest.raises(ValueError, match="output tokens"):
                receiver.pop(max_new_tokens=512)
            with pytest.raises(ValueError, match="positive integer"):
                receiver.pop(max_new_tokens=0)
            with monkeypatch.context() as patch:
                patch.setattr(tokenizer, "chat_template", None)
                inputs = receiver.pop(max_new_tokens=3)
            assert inputs["max_new_tokens"] == 3
            assert torch.equal(inputs["inputs_embeds"], expected)
            actual = receiver.model.generate(**inputs, do_sample=False, pad_token_id=0)
            assert torch.equal(actual, reference_tokens)
    assert all(len(receiver) == 0 for receiver in receivers)
    with pytest.raises(IndexError, match="no queued handoff"):
        receivers[0].pop()
    transfer_sync(bound, receivers[0], ids[:, :2], ratio=1)
    inputs = receivers[0].pop(prompt=ids[:, :3])
    assert torch.equal(inputs["inputs_embeds"][:, -3:], model.get_input_embeddings()(ids[:, :3]))
    for invalid in (torch.ones(3), torch.tensor([[512]]), torch.empty((1, 0), dtype=torch.long)):
        with pytest.raises(ValueError):
            sender_from_hf(model, tokenizer, invalid)
    with pytest.raises(ValueError, match="cache length"):
        sender_from_hf(model, tokenizer, ids[:, :4], past_key_values=sender.past_key_values)
    incompatible = receivers[1]
    with monkeypatch.context() as patch:
        patch.setattr(incompatible.model.config, "_name_or_path", "different-checkpoint")
        patch.setattr(model.config, "_name_or_path", "original-checkpoint")
        with pytest.raises(ValueError, match="checkpoint names"):
            transfer_sync(bound, [reference.append, incompatible], requests)
    with monkeypatch.context() as patch:
        patch.setattr(incompatible.model.config, "hidden_size", 128)
        with pytest.raises(ValueError, match="checkpoint architecture"):
            transfer_sync(bound, incompatible, requests)
    changed_tokenizer = copy.deepcopy(tokenizer)
    changed_tokenizer.add_tokens(["different-token"])
    with pytest.raises(ValueError, match="tokenizer vocabulary"):
        transfer_sync(bound, HFReceiver(model, changed_tokenizer), requests)
    assert len(reference) == 2 and len(incompatible) == 0
    assert isinstance(asyncio.run(handoff(model, tokenizer)), str)
    model.train()
    with pytest.raises(ValueError, match="eval mode"):
        HFReceiver(model, tokenizer)
    model.eval()
    with monkeypatch.context() as patch:
        patch.setattr(model.config, "model_type", "llama")
        with pytest.raises(ValueError, match="dense Qwen3"):
            sender_from_hf(model, tokenizer, ids)


@torch.inference_mode()
def test_existing_agent_history_and_receiver_budget(
    senders: list[SenderState], monkeypatch: Any
) -> None:
    from rclc._cache import cache_kv

    model, tokenizer = senders[0].model, senders[0].tokenizer
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}"
        "{% if add_generation_prompt %}assistant{% endif %}"
    )
    generation = model.generate(
        input_ids=torch.arange(10, 40)[None],
        max_new_tokens=4,
        do_sample=False,
        pad_token_id=0,
        return_dict_in_generate=True,
    )
    cached = generation.past_key_values.get_seq_length()
    before = [(k.clone(), v.clone()) for k, v in cache_kv(generation.past_key_values)]
    calls = []
    native_forward = model.model.forward

    def traced_forward(**kwargs: Any) -> Any:
        calls.append(kwargs["inputs_embeds"].shape[1])
        return native_forward(**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(model.model, "forward", traced_forward)
        bound = sender_from_hf(model, tokenizer, generation=generation)
    assert calls == [generation.sequences.shape[1] - cached] == [1]
    assert generation.past_key_values.get_seq_length() == cached
    fresh = model.model(input_ids=generation.sequences, use_cache=True).past_key_values
    for actual, expected in zip(cache_kv(bound.past_key_values), cache_kv(fresh), strict=True):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(cache_kv(generation.past_key_values), before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="choose generation"):
        sender_from_hf(model, tokenizer, generation.sequences, generation=generation)
    with pytest.raises(ValueError, match="generation result"):
        sender_from_hf(model, tokenizer, generation=torch.ones(3))

    beam_output = model.generate(
        input_ids=torch.arange(10, 20)[None],
        max_new_tokens=2,
        num_beams=2,
        do_sample=False,
        pad_token_id=0,
        return_dict_in_generate=True,
    )
    with pytest.raises(ValueError, match="non-beam"):
        sender_from_hf(model, tokenizer, generation=beam_output)
    extended_ids = torch.cat((generation.sequences, torch.tensor([[20, 21, 22]])), dim=1)
    calls.clear()
    with monkeypatch.context() as patch:
        patch.setattr(model.model, "forward", traced_forward)
        updated = sender_from_hf(
            model, tokenizer, extended_ids, past_key_values=bound.past_key_values
        )
    assert calls == [3]
    assert bound.past_key_values.get_seq_length() == generation.sequences.shape[1]
    fresh = model.model(input_ids=extended_ids, use_cache=True).past_key_values
    for actual, expected in zip(cache_kv(updated.past_key_values), cache_kv(fresh), strict=True):
        torch.testing.assert_close(actual, expected)

    history = [{"role": "system", "content": "Cedar"}, {"role": "user", "content": "owns"}]
    chat_sender = sender_from_hf(model, tokenizer, history)
    chat_ids = tokenizer.apply_chat_template(history, return_tensors="pt", return_dict=True)[
        "input_ids"
    ]
    assert torch.equal(chat_sender.token_ids, chat_ids[0])
    short = sender_from_hf(model, tokenizer, torch.tensor([[10, 11]]))
    states = [bound, senders[0], short]
    requests = ["Who owns Cedar ?", "When Birch launches now ?"]
    histories = [history, [{"role": "system", "content": "Birch launches"}]]
    prompts = [
        [
            tokenizer.apply_chat_template(
                [*h, {"role": "user", "content": q}],
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )["input_ids"]
            for q in requests
        ]
        for h in histories
    ]
    receivers = [
        HFReceiver(
            model,
            tokenizer,
            messages=h,
            max_new_tokens=3,
            context_limit=ids[0].shape[1] + 3 + space,
        )
        for h, ids, space in zip(histories, prompts, [25, 20], strict=True)
    ]
    recorded: list[Delivery] = []
    with monkeypatch.context() as patch:
        patch.setattr(model, "generate", lambda **kw: pytest.fail("transfer generated"))
        transfer_sync(states, [recorded.append, *receivers], requests, ratio=1)
    assert [[m.positions for m in d.messages] for d in recorded] == [[8, 10, 2], [8, 9, 2]]
    assert all(len(r) == 2 for r in receivers)
    for delivery in recorded:
        torch.testing.assert_close(
            delivery.messages[1].continuous_rows[-2:], senders[0].input_embeds[-2:]
        )
    history.append({"role": "user", "content": "Cedar " * 512})
    for request_index, delivery in enumerate(recorded):
        for index, receiver in enumerate(receivers):
            with pytest.raises(ValueError, match="no context space"):
                receiver.pop(max_new_tokens=512)
            expected = torch.cat(
                (
                    delivery.for_hf(model.get_input_embeddings().weight)["inputs_embeds"],
                    model.get_input_embeddings()(prompts[index][request_index]),
                ),
                dim=1,
            )
            inputs = receiver.pop()
            assert inputs["max_new_tokens"] == 3
            assert inputs["inputs_embeds"].shape[1] + 3 <= receiver.context_limit
            torch.testing.assert_close(inputs["inputs_embeds"], expected, rtol=0, atol=0)
            actual = model.generate(**inputs, do_sample=False, pad_token_id=0)
            reference = model.generate(
                inputs_embeds=expected,
                attention_mask=torch.ones(expected.shape[:2], dtype=torch.long),
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=0,
            )
            assert torch.equal(actual, reference)
    with pytest.raises(ValueError, match="handoff positions"):
        transfer_sync(
            states,
            [recorded.append, receivers[0]],
            requests,
            selector=lambda *args: pytest.fail("selection before failed budget planning"),
        )
    assert len(recorded) == 2 and not any(len(r) for r in receivers)
    history.pop()
    receiver = receivers[1]
    transfer_sync(short, [receiver, recorded.append], requests[0], budget=1)
    assert recorded[-1].messages[0].positions == 1
    receiver.pop()
    receiver.context_limit = prompts[1][0].shape[1] + 3 + 4
    with pytest.raises(ValueError, match="needs at least 5"):
        transfer_sync(states, receiver, requests[0], ratio=1)
    assert len(receiver) == 0
    for arguments in ({"max_new_tokens": 0}, {"context_limit": True}, {"context_limit": 513}):
        with pytest.raises(ValueError):
            HFReceiver(model, tokenizer, **arguments)
    receiver.context_limit = prompts[1][0].shape[1] + 3 + 12
    for context in ("full", "full_with_request", "selected", "selected_with_request"):
        transfer_sync(
            states,
            [receiver, recorded.append],
            requests[0],
            ratio=1,
            reasoning=partial(latent_mass, steps=2),
            reasoning_context=context,
            reasoning_budget=2,
        )
        counts = [message.positions for message in recorded[-1].messages]
        assert sum(counts) == 12 and all(
            c >= floor for c, floor in zip(counts, [3, 5, 3], strict=True)
        )
        assert receiver.pop()["inputs_embeds"].shape[1] + 3 == receiver.context_limit
    bound.inherited_positions = 10
    roomy = HFReceiver(model, tokenizer, max_new_tokens=3)
    transfer_sync(bound, [roomy, recorded.append], requests[0])
    assert recorded[-1].messages[0].positions == 10 + (bound.input_embeds.shape[0] - 10 + 3) // 4
    roomy.pop()
    from rclc.selectors import cacheback

    for selector in (cacheback, partial(cacheback, span_size=4)):
        with pytest.raises(ValueError, match="required positions"):
            transfer_sync(senders[0], roomy, requests[0], budget=1, selector=selector)
    for context in (None, "full", "selected"):
        transfer_sync(
            senders[0],
            [roomy, recorded.append],
            requests[0],
            budget=3 if context == "selected" else 1,
            selector=lambda state, request, budget: [1],
            reasoning=partial(latent_mass, steps=2) if context else None,
            reasoning_context=context,
            reasoning_budget=2 if context else None,
        )
        assert recorded[-1].messages[0].positions == (3 if context == "selected" else 1)
        inputs = roomy.pop()
        assert inputs["max_new_tokens"] == 3
        model.generate(**inputs, do_sample=False, pad_token_id=0)

    x_history = [{"role": "user", "content": "Cedar owns " * 12}]
    with monkeypatch.context() as patch:
        patch.setattr(model.model, "forward", lambda **kw: pytest.fail("bind ran the agent"))
        agent_x = bind(model, tokenizer, messages=x_history, backend="hf", max_new_tokens=3)
        agent_y = bind(model, tokenizer, messages=histories[1], backend="hf", max_new_tokens=3)
    assert type(agent_x) is type(agent_y) is Agent
    original = agent_x.sender_state()
    x_history.append({"role": "assistant", "content": "Cedar launches"})
    calls.clear()
    with monkeypatch.context() as patch:
        patch.setattr(model.model, "forward", traced_forward)
        extended = agent_x.sender_state()
    assert calls == [extended.input_embeds.shape[0] - original.input_embeds.shape[0]]
    assert original.past_key_values.get_seq_length() == original.input_embeds.shape[0]
    x_history[0]["content"] = "Birch launches " * 12
    calls.clear()
    with monkeypatch.context() as patch:
        patch.setattr(model.model, "forward", traced_forward)
        edited = agent_x.sender_state()
    assert calls == [edited.input_embeds.shape[0]]
    transfer_sync([agent_x, short], [agent_y, recorded.append], requests[0])
    received_positions = sum(m.positions for m in recorded[-1].messages)
    inputs = agent_y.pop()
    with pytest.raises(ValueError, match=r"agent\.update"):
        transfer_sync(agent_y, agent_x, requests[0])
    with pytest.raises(ValueError, match=r"agent\.update"):
        asyncio.run(agent_y.append("Cedar"))
    output = model.generate(**inputs, do_sample=False, pad_token_id=0, return_dict_in_generate=True)
    with pytest.raises(ValueError, match="pop_result"):
        agent_y.update(generation=output)
    with pytest.raises(ValueError, match="most recent pop"):
        agent_y.update(generation=output, inputs=dict(inputs))
    cache_length = output.past_key_values.get_seq_length()
    agent_y.update(generation=output, inputs=inputs)
    y_state = agent_y.sender_state()
    assert y_state.inherited_positions == received_positions
    assert output.past_key_values.get_seq_length() == cache_length
    expected_rows = torch.cat(
        (inputs["inputs_embeds"][0], model.get_input_embeddings()(output.sequences)[0])
    )
    torch.testing.assert_close(y_state.input_embeds, expected_rows, rtol=0, atol=0)
    prefix = inputs["inputs_embeds"].shape[1]
    assert bool((y_state.token_ids[:received_positions] == -1).all())
    assert torch.equal(y_state.token_ids[received_positions:prefix], prompts[1][0][0])
    assert torch.equal(y_state.token_ids[prefix:], output.sequences[0])
    fresh = model.model(inputs_embeds=expected_rows[None], use_cache=True).past_key_values
    for actual, expected in zip(cache_kv(y_state.past_key_values), cache_kv(fresh), strict=True):
        torch.testing.assert_close(actual, expected)
    for suffix in ("Cedar launches", torch.tensor([[30, 31]])):
        previous = agent_y.sender_state()
        asyncio.run(agent_y.append(suffix))
        extended = agent_y.sender_state()
        assert extended.inherited_positions == received_positions
        assert extended.latent_steps == 0
        assert torch.equal(extended.input_embeds[:-2], previous.input_embeds)
        assert previous.past_key_values.get_seq_length() == len(previous.input_embeds)
        fresh = model.model(inputs_embeds=extended.input_embeds[None], use_cache=True)
        for actual, expected in zip(
            cache_kv(extended.past_key_values), cache_kv(fresh.past_key_values), strict=True
        ):
            torch.testing.assert_close(actual, expected)
    for bad_suffix in (torch.tensor([[512]]), torch.ones((1, 512), dtype=torch.long)):
        with pytest.raises(ValueError):
            asyncio.run(agent_y.append(bad_suffix))
        assert agent_y.sender_state() is extended
    expected_rows = extended.input_embeds
    with pytest.warns(UserWarning, match="experimental"):
        transfer_sync(
            agent_y, [agent_x, recorded.append], requests[1], representation="token_ids+continuous"
        )
    assert (
        recorded[-1].messages[0].positions
        == received_positions + (expected_rows.shape[0] - received_positions + 3) // 4
    )
    inputs_x = agent_x.pop()
    output_x = model.generate(
        **inputs_x, do_sample=False, pad_token_id=0, return_dict_in_generate=True
    )
    agent_x.update(generation=output_x, inputs=inputs_x)
    transfer_sync(agent_x, agent_y, requests[0])
    assert len(agent_y) == 1
    agent_x.update()
    assert agent_x.sender_state().inherited_positions == 0
    transfer_sync(agent_x, agent_y, requests[1])
    first = agent_y.pop()
    with pytest.raises(ValueError, match="discard_pending"):
        agent_y.pop()
    with pytest.raises(ValueError, match="no context space"):
        agent_y.pop(discard_pending=True, max_new_tokens=512)
    assert len(agent_y) == 1 and agent_y._pending_inputs is first
    second = agent_y.pop(discard_pending=True)
    assert not agent_y and agent_y._pending_inputs is second
    output = model.generate(**second, pad_token_id=0, return_dict_in_generate=True)
    with pytest.raises(ValueError, match="most recent pop"):
        agent_y.update(inputs=first, generation=output)
    agent_y.update(inputs=second, generation=output)
    assert agent_y.sender_state().inherited_positions > 0
    with pytest.raises(ValueError, match="backend='hf'"):
        bind(model, tokenizer, backend="vllm")
    from rclc import vllm

    with monkeypatch.context() as patch:
        patch.setattr(vllm, "binding_model", lambda llm, tok, *args: (model, tok, 512))
        with pytest.raises(ValueError, match="exact captured prompt"):
            bind(object(), tokenizer, backend="vllm", request_id="captured")
    with pytest.raises(ValueError, match="agent has no state"):
        transfer_sync(bind(model, tokenizer), recorded.append, requests[0])
    with pytest.raises(ValueError, match="needs a tokenizer"):
        bind(model).update("Cedar launches")
    explicit = bind(model, tokenizer, generation=generation)
    assert torch.equal(explicit.sender_state().token_ids, generation.sequences[0])
