"""Optional latent rollout through selection, both payloads and native receiver continuation."""

from dataclasses import replace
from itertools import product
from typing import Any

import pytest
import torch
import torch.nn.functional as functional

from rcc import Delivery, SenderState, rollout, transfer
from rcc.selectors import cacheback
from rcc.selectors.core.kernels import cache_kv


@torch.inference_mode()
def _reference(state: SenderState, steps: int, request: torch.Tensor | None = None) -> SenderState:
    model = state.model
    rows = state.input_embeds[None]
    prefix = (
        rows if request is None else torch.cat((rows, model.get_input_embeddings()(request)), 1)
    )
    target = model.get_input_embeddings().weight.float().norm(dim=-1).mean()
    thoughts = []
    for _ in range(steps):
        hidden = model.model(inputs_embeds=prefix, use_cache=False).last_hidden_state[:, -1:]
        thought = functional.normalize(hidden, dim=-1, eps=1e-6) * target.to(hidden)
        thoughts.append(thought)
        prefix = torch.cat((prefix, thought), 1)
    rows = torch.cat((rows, *thoughts), 1)
    output = model.model(inputs_embeds=rows, use_cache=True)
    ids = state.token_ids
    if ids is not None:
        ids = torch.cat((ids, ids.new_full((steps,), -1)))
    return replace(
        state,
        input_embeds=rows[0],
        past_key_values=output.past_key_values,
        token_ids=ids,
        latent_steps=state.latent_steps + steps,
    )


@torch.inference_mode()
def _selected(state: SenderState, query: torch.Tensor, budget: int) -> SenderState:
    keep = cacheback(state, query, budget)
    rows = state.input_embeds[keep]
    output = state.model.model(inputs_embeds=rows[None], use_cache=True)
    return replace(
        state,
        input_embeds=rows,
        past_key_values=output.past_key_values,
        token_ids=state.token_ids[keep],
        latent_steps=0,
        inherited_positions=0,
    )


def test_latent_handoffs_and_receiver_rollout_match_full_forward_reference(
    senders: list[SenderState],
    monkeypatch: Any,
) -> None:
    snapshots = [[(k.clone(), v.clone()) for k, v in cache_kv(s.past_key_values)] for s in senders]
    queries = ["Who owns Cedar ?", "When Birch launches ?"]
    steps = 3
    for context, bounded in product(
        ("full", "full_with_request", "selected", "selected_with_request"), (False, True)
    ):
        inboxes: list[list[Delivery]] = [[], []]
        options = {"budget": 35} if bounded else {}
        latent_options = {} if context == "full" else {"latent_context": context}
        transfer(
            senders,
            [inbox.append for inbox in inboxes],
            queries,
            latent_steps=steps,
            **latent_options,
            **options,
        )
        mixed: list[Delivery] = []
        with pytest.warns(UserWarning, match="untested"):
            transfer(
                senders,
                mixed.append,
                queries,
                representation="token_ids+continuous",
                latent_steps=steps,
                latent_context=context,
                **options,
            )
        assert inboxes[0] == inboxes[1]
        assert len(mixed) == len(inboxes[0]) == 2
        for index, delivery in enumerate(inboxes[0]):
            expected = []
            for state, message in zip(senders, delivery.messages, strict=True):
                query = state.request_ids(delivery.request)
                prompt = (
                    query if context in ("full_with_request", "selected_with_request") else None
                )
                count = 35 if bounded else (state.input_embeds.shape[0] + steps + 3) // 4
                if context in ("full", "full_with_request"):
                    grown = _reference(state, steps, prompt)
                    wanted = grown.input_embeds[cacheback(grown, query, count)]
                else:
                    selected = _selected(state, query, count - steps)
                    wanted = _reference(selected, steps, prompt).input_embeds
                assert message.positions == count
                torch.testing.assert_close(message.continuous_rows, wanted, rtol=1e-5, atol=1e-6)
                expected.append(wanted)
            model = senders[0].model
            weight = model.get_input_embeddings().weight
            received = delivery.for_hf(weight)["inputs_embeds"]
            mixed_rows = mixed[index].for_hf(weight)["inputs_embeds"]
            assert torch.equal(received, mixed_rows)
            assert all(
                bool((message.token_ids[-steps:] == -1).all()) for message in mixed[index].messages
            )
            with torch.inference_mode():
                query = senders[0].request_ids(delivery.request)
                cached = model.model(inputs_embeds=received, use_cache=True)
                actual = model(input_ids=query, past_key_values=cached.past_key_values)
                joined = torch.cat((*expected, model.get_input_embeddings()(query)[0]))[None]
                native = model(inputs_embeds=joined)
                torch.testing.assert_close(
                    actual.logits[:, -1], native.logits[:, -1], rtol=1e-5, atol=1e-6
                )

    state = replace(senders[0], inherited_positions=20)
    for context in ("full", "selected"):
        inherited: list[Delivery] = []
        transfer(state, inherited.append, queries[0], latent_steps=steps, latent_context=context)
        assert inherited[0].messages[0].positions == 37
    for request in (None, state.request_ids(queries[0])):
        grown = rollout(state, steps=steps, request=request)
        expected_state = _reference(state, steps, request)
        assert grown.inherited_positions == 20 and grown.latent_steps == 5
        assert grown.input_embeds.shape[0] == state.input_embeds.shape[0] + steps
        torch.testing.assert_close(
            grown.input_embeds, expected_state.input_embeds, rtol=1e-5, atol=1e-6
        )
        for actual, expected_kv in zip(
            cache_kv(grown.past_key_values), cache_kv(expected_state.past_key_values), strict=True
        ):
            for actual_tensor, expected_tensor in zip(actual, expected_kv, strict=True):
                torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-5, atol=1e-6)

    # The receiver owns its prompt and can use the same helper after delivery.
    rows = inboxes[0][0].for_hf(state.model.get_input_embeddings().weight)["inputs_embeds"]
    with torch.inference_mode():
        output = state.model.model(inputs_embeds=rows, use_cache=True)
    receiver = SenderState(state.model, output.past_key_values, rows[0], tokenizer=state.tokenizer)
    receiver = rollout(receiver, steps=steps, request=queries[0])
    with torch.inference_mode():
        query = state.request_ids(queries[1])
        actual = state.model(input_ids=query, past_key_values=receiver.past_key_values)
        native = state.model(
            inputs_embeds=torch.cat(
                (receiver.input_embeds, state.model.get_input_embeddings()(query)[0])
            )[None]
        )
        torch.testing.assert_close(actual.logits[:, -1], native.logits[:, -1], rtol=1e-5, atol=1e-6)

    import rcc.transport as transport

    observed = []

    def record(state: SenderState, **kwargs: Any) -> SenderState:
        observed.append(state)
        return rollout(state, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(transport, "rollout", record)
        transfer(senders, [].append, queries, latent_steps=steps)
    assert len(observed) == 2 and all(a is b for a, b in zip(observed, senders, strict=True))

    for state, snapshot in zip(senders, snapshots, strict=True):
        for actual, expected_kv in zip(cache_kv(state.past_key_values), snapshot, strict=True):
            assert all(torch.equal(a, b) for a, b in zip(actual, expected_kv, strict=True))


def test_latent_options_fail_before_delivery_and_zero_preserves_existing_path(
    senders: list[SenderState],
    monkeypatch: Any,
) -> None:
    state = senders[0]
    before = [(k.clone(), v.clone()) for k, v in cache_kv(state.past_key_values)]
    query = "Who owns Cedar ?"
    inbox: list[Delivery] = []
    for options in (
        {"latent_steps": -1},
        {"latent_steps": True},
        {"latent_steps": 1.5},
        {"latent_context": "after"},
        {"latent_context": "prompt_and_request"},
        {"latent_steps": 35, "latent_context": "selected", "budget": 35},
        {"latent_steps": 440, "ratio": 1},
    ):
        with pytest.raises(ValueError):
            transfer(senders, inbox.append, query, **options)
    assert inbox == []
    for steps in (-1, True, 1.5, 440):
        with pytest.raises(ValueError):
            rollout(state, steps=steps)
    with pytest.raises(ValueError):
        rollout(state, steps=1, request=torch.ones((1, 440), dtype=torch.long))

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("rollout failed")

    with monkeypatch.context() as patch:
        patch.setattr(state.model.model, "forward", fail)
        assert rollout(state, steps=0) is state
        transfer(state, inbox.append, query, ratio=1, latent_steps=0, latent_context="selected")
        with pytest.raises(RuntimeError, match="rollout failed"):
            transfer(state, inbox.append, query, ratio=1, latent_steps=1)
    assert len(inbox) == 1
    assert torch.equal(inbox[0].messages[0].continuous_rows, state.input_embeds)

    original = state.model.model.forward
    calls = 0

    def interrupt(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("rollout interrupted")
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(state.model.model, "forward", interrupt)
        with pytest.raises(RuntimeError, match="rollout interrupted"):
            transfer(state, inbox.append, query, latent_steps=2, latent_context="full_with_request")
    assert len(inbox) == 1
    for actual, expected in zip(cache_kv(state.past_key_values), before, strict=True):
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))

    def underfill(sender: SenderState, ids: torch.Tensor, budget: int) -> list[int]:
        assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
        return [0, sender.input_embeds.shape[0] - 1]

    transfer(
        state,
        inbox.append,
        query,
        budget=10,
        selector=underfill,
        latent_steps=2,
        latent_context="selected",
    )
    assert inbox[-1].messages[0].positions == 4
