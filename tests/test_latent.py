"""Optional latent rollout through selection, both payloads and native receiver continuation."""

import copy
from dataclasses import replace
from functools import partial
from itertools import product
from typing import Any

import pytest
import torch
import torch.nn.functional as functional

from rclc import Delivery, SenderState, latent_mass_sync, transfer_sync
from rclc._cache import cache_kv
from rclc.selectors import cacheback


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
) -> None:
    _check_callable_journey(senders)
    snapshots = [[(k.clone(), v.clone()) for k, v in cache_kv(s.past_key_values)] for s in senders]
    queries = ["Who owns Cedar ?", "When Birch launches ?"]
    steps = 3
    for context, bounded in product(
        ("full", "full_with_request", "selected", "selected_with_request"), (False, True)
    ):
        inboxes: list[list[Delivery]] = [[], []]
        options = {"budget": 35} if bounded else {}
        options |= _latent(steps, context)
        transfer_sync(senders, [inbox.append for inbox in inboxes], queries, **options)
        mixed: list[Delivery] = []
        with pytest.warns(UserWarning, match="untested"):
            transfer_sync(
                senders,
                mixed.append,
                queries,
                representation="token_ids+continuous",
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
        transfer_sync(state, inherited.append, queries[0], **_latent(steps, context))
        assert inherited[0].messages[0].positions == 37
    for request in (None, state.request_ids(queries[0])):
        grown = latent_mass_sync(state, steps=steps, request=request)
        for original, updated in zip(
            cache_kv(state.past_key_values), cache_kv(grown.past_key_values), strict=True
        ):
            assert all(
                torch.equal(a, b[:, :, : len(state.input_embeds)])
                for a, b in zip(original, updated, strict=True)
            )
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
    receiver = latent_mass_sync(receiver, steps=steps, request=queries[0])
    with torch.inference_mode():
        query = state.request_ids(queries[1])
        actual = state.model(input_ids=query, past_key_values=receiver.past_key_values)
        native = state.model(
            inputs_embeds=torch.cat(
                (receiver.input_embeds, state.model.get_input_embeddings()(query)[0])
            )[None]
        )
        torch.testing.assert_close(actual.logits[:, -1], native.logits[:, -1], rtol=1e-5, atol=1e-6)

    observed = []

    def record(state: SenderState, **kwargs: Any) -> SenderState:
        observed.append(state)
        return latent_mass_sync(state, steps=steps, **kwargs)

    transfer_sync(senders, [].append, queries, reasoning=record)
    assert len(observed) == 2
    assert all(
        torch.equal(a.input_embeds, b.input_embeds) for a, b in zip(observed, senders, strict=True)
    )

    for state, snapshot in zip(senders, snapshots, strict=True):
        for actual, expected_kv in zip(cache_kv(state.past_key_values), snapshot, strict=True):
            assert all(torch.equal(a, b) for a, b in zip(actual, expected_kv, strict=True))


def test_latent_options_fail_before_delivery_and_zero_preserves_existing_path(
    senders: list[SenderState],
    monkeypatch: Any,
) -> None:
    state = senders[0]
    _check_callable_failures(state)
    before = [(k.clone(), v.clone()) for k, v in cache_kv(state.past_key_values)]
    query = "Who owns Cedar ?"
    inbox: list[Delivery] = []
    for options in (
        _latent(-1),
        _latent(True),
        _latent(1.5),
        {"reasoning_context": "selected"},
        {"reasoning": partial(latent_mass_sync, steps=1), "reasoning_context": "after"},
        {
            "reasoning": partial(latent_mass_sync, steps=1),
            "reasoning_context": "prompt_and_request",
        },
        {**_latent(35, "selected"), "budget": 35},
        {**_latent(440), "ratio": 1},
    ):
        with pytest.raises(ValueError):
            transfer_sync(senders, inbox.append, query, **options)
    assert inbox == []
    for steps in (-1, True, 1.5, 440):
        with pytest.raises(ValueError):
            latent_mass_sync(state, steps=steps)
    with pytest.raises(ValueError, match="max_positions must be"):
        latent_mass_sync(state, steps=0, max_positions=True)
    with pytest.raises(ValueError):
        latent_mass_sync(state, steps=1, request=torch.ones((1, 440), dtype=torch.long))

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("rollout failed")

    with monkeypatch.context() as patch:
        patch.setattr(state.model.model, "forward", fail)
        assert latent_mass_sync(state, steps=0) is state
        transfer_sync(state, inbox.append, query, ratio=1)
        with pytest.raises(RuntimeError, match="rollout failed"):
            transfer_sync(state, inbox.append, query, ratio=1, **_latent(1))
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
            transfer_sync(state, inbox.append, query, **_latent(2, "full_with_request"))
    assert len(inbox) == 1
    for actual, expected in zip(cache_kv(state.past_key_values), before, strict=True):
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))

    def underfill(sender: SenderState, ids: torch.Tensor, budget: int) -> list[int]:
        assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
        return [0, sender.input_embeds.shape[0] - 1]

    transfer_sync(
        state,
        inbox.append,
        query,
        budget=10,
        selector=underfill,
        **_latent(2, "selected"),
    )
    assert inbox[-1].messages[0].positions == 4


def _adaptive_mean(
    state: SenderState,
    *,
    max_positions: int,
    request: torch.Tensor | None = None,
    tolerance: float = 0.01,
) -> SenderState:
    from rclc.latent import forward_rows

    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    limit = 2 if request is None else int(request[0, 0]) % 4
    thoughts = []
    previous = None
    for _ in range(min(limit, max_positions)):
        thought = state.input_embeds.mean(0, keepdim=True)
        thoughts.append(thought)
        if previous is not None and float((thought - previous).norm()) < tolerance:
            break
        previous = thought
    if not thoughts:
        return state
    added = torch.cat(thoughts)
    with torch.inference_mode():
        out = forward_rows(state.model, added[None], state.past_key_values, len(state.input_embeds))
    return replace(
        state,
        input_embeds=torch.cat((state.input_embeds, added)),
        past_key_values=out.past_key_values,
        token_ids=torch.cat((state.token_ids, state.token_ids.new_full((len(added),), -1))),
        latent_steps=state.latent_steps + len(added),
    )


@torch.inference_mode()
def _check_bfloat16_callable(state: SenderState) -> None:
    model = copy.deepcopy(state.model).to(torch.bfloat16)
    rows = state.input_embeds.to(torch.bfloat16)
    output = model.model(inputs_embeds=rows[None], use_cache=True)
    sender = replace(state, model=model, input_embeds=rows, past_key_values=output.past_key_values)
    # Cached values from another kernel can differ by one representable value.
    key = cache_kv(sender.past_key_values)[0][0][:, :, -1:, :1]
    key.copy_(torch.nextafter(key, torch.full_like(key, float("inf"))))

    def custom(current: SenderState, **kwargs: Any) -> SenderState:
        grown = latent_mass_sync(current, steps=2, **kwargs)
        for original, updated in zip(
            cache_kv(current.past_key_values), cache_kv(grown.past_key_values), strict=True
        ):
            assert all(
                torch.equal(a, b[:, :, : len(current.input_embeds)])
                for a, b in zip(original, updated, strict=True)
            )
        return grown

    for context in ("full", "full_with_request", "selected", "selected_with_request"):
        actual: list[Delivery] = []
        expected: list[Delivery] = []
        options = {"reasoning_context": context, "reasoning_budget": 2}
        transfer_sync(sender, actual.append, "Who owns Cedar ?", reasoning=custom, **options)
        transfer_sync(
            sender,
            expected.append,
            "Who owns Cedar ?",
            reasoning=partial(latent_mass_sync, steps=2),
            **options,
        )
        torch.testing.assert_close(
            actual[0].messages[0].continuous_rows,
            expected[0].messages[0].continuous_rows,
            rtol=0,
            atol=0,
        )


def _latent(steps: Any, context: str = "full") -> dict[str, Any]:
    return {
        "reasoning": partial(latent_mass_sync, steps=steps),
        "reasoning_context": context,
        "reasoning_budget": steps if type(steps) is int and steps >= 0 else None,
    }


def _check_callable_journey(senders: list[SenderState]) -> None:
    from rclc import HFReceiver, bind

    queries = ["Who owns Cedar ?", "When Birch launches ?"]
    state = senders[0]
    _check_bfloat16_callable(state)
    queries = [state.request_ids(q) for q in queries]
    weight = state.model.get_input_embeddings().weight
    fifty: list[Delivery] = []
    transfer_sync(
        state, fifty.append, queries[0], reasoning=partial(latent_mass_sync, steps=50), budget=128
    )
    assert fifty[0].messages[0].positions == 128

    calls = []

    def custom(sender: SenderState, **kwargs: Any) -> SenderState:
        result = _adaptive_mean(sender, tolerance=0.001, **kwargs)
        calls.append((sender, kwargs, result))
        return result

    for context, bounded in product(
        ("full", "full_with_request", "selected", "selected_with_request"), (False, True)
    ):
        calls.clear()
        recorded: list[Delivery] = []
        receiver = bind(state.model, state.tokenizer, max_new_tokens=2)
        options = {"budget": 15} if bounded else {}
        with pytest.warns(UserWarning, match="untested"):
            transfer_sync(
                senders,
                [receiver, recorded.append],
                queries,
                reasoning=custom,
                reasoning_context=context,
                reasoning_budget=4,
                representation="token_ids+continuous",
                **options,
            )
        assert len(calls) == (2 if context == "full" else 4)
        assert len(recorded) == len(receiver) == 2
        for index, delivery in enumerate(recorded):
            for sender_index, message in enumerate(delivery.messages):
                before, kwargs, grown = calls[
                    sender_index if context == "full" else index * 2 + sender_index
                ]
                assert (
                    0
                    <= len(grown.input_embeds) - len(before.input_embeds)
                    <= kwargs["max_positions"]
                )
                assert (kwargs["request"] is not None) == context.endswith("with_request")
                if context.startswith("selected"):
                    expected = grown.input_embeds
                else:
                    count = 15 if bounded else (len(grown.input_embeds) + 3) // 4
                    expected = grown.input_embeds[
                        cacheback(grown, grown.request_ids(delivery.request), count)
                    ]
                torch.testing.assert_close(message.materialize(weight), expected)
                fresh = state.model.model(inputs_embeds=grown.input_embeds[None], use_cache=True)
                for a, b in zip(
                    cache_kv(grown.past_key_values), cache_kv(fresh.past_key_values), strict=True
                ):
                    torch.testing.assert_close(a, b)
            inputs = receiver.pop(discard_pending=True)
            query = state.request_ids(delivery.request)
            expected_rows = torch.cat(
                (delivery.for_hf(weight)["inputs_embeds"], weight[query]), dim=1
            )
            torch.testing.assert_close(inputs["inputs_embeds"], expected_rows)
            actual = state.model.generate(**inputs, do_sample=False, pad_token_id=0)
            expected_ids = state.model.generate(
                inputs_embeds=expected_rows, max_new_tokens=2, do_sample=False, pad_token_id=0
            )
            assert torch.equal(actual, expected_ids)

    constrained = HFReceiver(state.model, state.tokenizer, max_new_tokens=2, context_limit=18)
    recorded = []
    transfer_sync(
        senders,
        [constrained, recorded.append],
        queries,
        reasoning=partial(_adaptive_mean, tolerance=0.01),
        ratio=1,
    )
    assert all(sum(m.positions for m in d.messages) <= 12 for d in recorded)
    assert len(constrained) == 2


def _check_callable_failures(state: SenderState) -> None:
    inbox: list[Delivery] = []
    query = "Who owns Cedar ?"
    before = [(k.clone(), v.clone()) for k, v in cache_kv(state.past_key_values)]
    rows, ids = state.input_embeds.clone(), state.token_ids.clone()
    for options in (
        {"reasoning": "latent_mass_sync"},
        {"reasoning": _adaptive_mean, "reasoning_context": "selected"},
        {"reasoning": _adaptive_mean, "reasoning_budget": -1},
        {"reasoning": _adaptive_mean, "reasoning_budget": True},
        {"reasoning_budget": 3},
        {"reasoning_context": "", "reasoning": _adaptive_mean},
        {"reasoning": partial(latent_mass_sync, steps=3), "reasoning_budget": 2},
    ):
        with pytest.raises(ValueError):
            transfer_sync(state, inbox.append, query, **options)

    def broken(sender: SenderState, *, defect: str, **kwargs: Any) -> SenderState:
        if defect == "raise":
            sender.input_embeds.zero_()
            sender.token_ids.zero_()
            with torch.inference_mode():
                cache_kv(sender.past_key_values)[0][0].zero_()
            raise RuntimeError("custom failure")
        if defect == "type":
            return None
        if defect == "prefix":
            sender.input_embeds[0].zero_()
        elif defect == "ids":
            sender.token_ids[0] = 0
        elif defect == "cache":
            with torch.inference_mode():
                cache_kv(sender.past_key_values)[0][0].zero_()
        elif defect == "nan":
            sender.input_embeds[0, 0] = float("nan")
        elif defect == "overflow":
            return latent_mass_sync(sender, steps=3)
        elif defect == "metadata":
            sender.inherited_positions += 1
        return sender

    for defect in ("raise", "type", "prefix", "ids", "cache", "nan", "overflow", "metadata"):
        with pytest.raises((ValueError, RuntimeError)):
            transfer_sync(
                state,
                inbox.append,
                query,
                reasoning=partial(broken, defect=defect),
                reasoning_budget=2,
            )
        assert inbox == []
        assert torch.equal(state.input_embeds, rows) and torch.equal(state.token_ids, ids)
        for actual, expected in zip(cache_kv(state.past_key_values), before, strict=True):
            assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))

    calls = 0

    def fail_later(sender: SenderState, **kwargs: Any) -> SenderState:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("later request failed")
        return _adaptive_mean(sender, **kwargs)

    with pytest.raises(RuntimeError, match="later request failed"):
        transfer_sync(
            state,
            inbox.append,
            [query, query],
            reasoning=fail_later,
            reasoning_context="full_with_request",
        )
    assert inbox == []
