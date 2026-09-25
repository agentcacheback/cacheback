"""Public handoffs with native capture, independent attention checks and receiver continuation."""

import copy
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from typing import Any

import pytest
import torch
import torch.nn.functional as functional

from rcc import Delivery, SenderState, transfer
from rcc.selectors import cacheback
from rcc.selectors.core import kernels
from rcc.selectors.fixed_spans import fixed_span_keep
from rcc.selectors.query_support import capture_single
from rcc.selectors.query_support.methods import reducers
from rcc.selectors.query_support.methods.compose import compose_score
from rcc.selectors.query_support.methods.scorers import memory_votes_with_support_moments


def _snapshot(state: SenderState) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(k.clone(), v.clone()) for k, v in kernels.cache_kv(state.past_key_values)]


def _assert_unchanged(state: SenderState, before: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    for actual, expected in zip(kernels.cache_kv(state.past_key_values), before, strict=True):
        assert all(torch.equal(a, e) for a, e in zip(actual, expected, strict=True))
    assert state.model.config._attn_implementation == "sdpa"
    assert capture_single.current_context() is None


@torch.inference_mode()
def _decode(
    model: Any, inputs: dict[str, torch.Tensor], query: torch.Tensor, *, split: bool
) -> tuple[torch.Tensor, list[int]]:
    if split:
        prefix = model.model(**inputs, use_cache=True)
        out = model(input_ids=query, past_key_values=prefix.past_key_values, use_cache=True)
    else:
        rows = torch.cat((inputs["inputs_embeds"], model.get_input_embeddings()(query)), dim=1)
        out = model(inputs_embeds=rows, use_cache=True)
    first_logits = out.logits[:, -1].clone()
    tokens = []
    for _ in range(4):
        token = out.logits[:, -1:].argmax(-1)
        tokens.append(int(token.item()))
        out = model(input_ids=token, past_key_values=out.past_key_values, use_cache=True)
    return first_logits, tokens


@torch.inference_mode()
def _eager_scores(state: SenderState, query: torch.Tensor) -> torch.Tensor:
    model = state.model
    length = state.input_embeds.shape[0]
    previous = kernels.use_eager_attention(model)
    try:
        output = model.model(
            input_ids=query,
            past_key_values=copy.deepcopy(state.past_key_values),
            output_attentions=True,
            use_cache=True,
        )
    finally:
        kernels.set_attention_implementation(model, previous)
    snap, energy, row_energy = (torch.zeros(length) for _ in range(3))
    for attention in output.attentions:
        weights = attention[0, :, :, :length].float()
        snap += weights.mean(dim=(0, 1))
        grouped = weights.reshape(
            model.config.num_key_value_heads, -1, query.shape[1], length
        ).mean(1)
        pooled_mean = functional.max_pool1d(grouped.mean(1)[None], 7, stride=1, padding=3)[0]
        pooled_rows = functional.max_pool1d(grouped, 7, stride=1, padding=3)
        energy += pooled_mean.square().sum(0)
        row_energy += pooled_rows.square().mean(1).sum(0)
    pooled_snap = functional.max_pool1d(snap[None, None], 7, stride=1, padding=3)[0, 0]
    return pooled_snap * (row_energy / energy).square()


@pytest.mark.parametrize("budget", [35, 1024], ids=["compressed", "identity"])
def test_native_handoff_matches_independent_selection_and_receiver_decode(
    senders: list[SenderState], budget: int, monkeypatch: Any
) -> None:
    state = senders[0]
    request = "Who owns Cedar ? " * (65 if budget == 35 else 1)
    query = state.request_ids(request)
    snapshot = _snapshot(state)
    inbox: list[Delivery] = []
    monkeypatch.setattr(
        state.model, "generate", lambda *a, **kw: pytest.fail("transfer generated an answer")
    )
    transfer(state, inbox.append, request, budget=budget)
    assert len(inbox) == 1 and inbox[0].request == request
    expected_scores = _eager_scores(state, query).clone()
    moments, found = memory_votes_with_support_moments(
        state.model,
        state.past_key_values,
        query,
        torch.ones_like(query),
        query[0],
        orders=(2.0,),
        n_sink=0,
    )
    assert found
    torch.testing.assert_close(
        compose_score(moments, order=2.0, alpha=2.0), expected_scores, rtol=2e-5, atol=1e-7
    )
    expected_scores[0] = expected_scores.max()
    keep = fixed_span_keep(expected_scores, budget, (0, 80, 81), span_size=16)
    expected_rows = state.input_embeds[keep]
    payload = inbox[0].for_hf(state.model.get_input_embeddings().weight)
    assert torch.equal(payload["inputs_embeds"][0], expected_rows)
    assert inbox[0].messages[0].positions == min(budget, 82)
    assert {0, 80, 81}.issubset(keep)
    receiver = copy.deepcopy(state.model)
    actual = _decode(receiver, payload, query, split=True)
    reference = _decode(receiver, {"inputs_embeds": expected_rows[None]}, query, split=False)
    torch.testing.assert_close(actual[0], reference[0], rtol=1e-5, atol=1e-6)
    assert actual[1] == reference[1]
    relative: list[Delivery] = []
    ratio, count = (3, 28) if budget == 35 else (1, 82)
    transfer(state, relative.append, request, ratio=ratio)
    relative_keep = fixed_span_keep(expected_scores, count, (0, 80, 81), span_size=16)
    relative_rows = relative[0].for_hf(receiver.get_input_embeddings().weight)["inputs_embeds"]
    assert torch.equal(relative_rows[0], state.input_embeds[relative_keep])
    narrow: list[Delivery] = []
    transfer(state, narrow.append, request, ratio=ratio, selector=partial(cacheback, span_size=4))
    narrow_keep = fixed_span_keep(expected_scores, count, (0, 80, 81), span_size=4)
    narrow_payload = narrow[0].for_hf(receiver.get_input_embeddings().weight)
    assert torch.equal(narrow_payload["inputs_embeds"][0], state.input_embeds[narrow_keep])
    actual = _decode(receiver, narrow_payload, query, split=True)
    reference = _decode(
        receiver, {"inputs_embeds": state.input_embeds[narrow_keep][None]}, query, split=False
    )
    torch.testing.assert_close(actual[0], reference[0], rtol=1e-5, atol=1e-6)
    assert actual[1] == reference[1]
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="span_size"):
            transfer(
                state,
                narrow.append,
                request,
                budget=1024,
                selector=partial(cacheback, span_size=invalid),
            )
        assert len(narrow) == 1
    _assert_unchanged(state, snapshot)


def test_distinct_senders_broadcast_every_request_to_independent_receivers(
    senders: list[SenderState],
) -> None:
    queries = ["Who owns Cedar ?", "When Birch launches ?"]
    inboxes: list[list[Delivery]] = [[], []]
    before = [_snapshot(state) for state in senders]
    calls = []

    def observe(state: SenderState, ids: torch.Tensor, budget: int) -> list[int]:
        calls.append(ids.clone())
        return cacheback(state, ids, budget)

    transfer(senders, [inbox.append for inbox in inboxes], queries, selector=observe)
    assert len(calls) == 4
    assert all([item.request for item in inbox] == queries for inbox in inboxes)
    continuations = []
    for inbox in inboxes:
        receiver = copy.deepcopy(senders[0].model)
        weight = receiver.get_input_embeddings().weight
        outputs = []
        for delivery in inbox:
            assert len(delivery.messages) == 2
            assert [message.positions for message in delivery.messages] == [21, 25]
            rows = [message.materialize(weight) for message in delivery.messages]
            assert not torch.equal(rows[0], rows[1])
            assert all(
                torch.equal(row[-2:], state.input_embeds[-2:])
                for row, state in zip(rows, senders, strict=True)
            )
            query = senders[0].request_ids(delivery.request)
            payload = delivery.for_hf(weight)
            actual = _decode(receiver, payload, query, split=True)
            reference = _decode(
                receiver, {"inputs_embeds": torch.cat(rows)[None]}, query, split=False
            )
            torch.testing.assert_close(actual[0], reference[0], rtol=1e-5, atol=1e-6)
            assert actual[1] == reference[1]
            outputs.append(actual[1])
        continuations.append(outputs)
    assert continuations[0] == continuations[1]
    for state, snapshot in zip(senders, before, strict=True):
        _assert_unchanged(state, snapshot)


def test_custom_selector_broadcasts_both_encodings_and_receivers_continue(
    senders: list[SenderState], monkeypatch: Any
) -> None:
    before = [_snapshot(state) for state in senders]
    calls = []
    monkeypatch.setattr(
        capture_single,
        "capture_context",
        lambda *a, **kw: pytest.fail("custom selection invoked CacheBack capture"),
    )

    def recent(state: SenderState, ids: torch.Tensor, budget: int) -> Any:
        assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
        assert ids.shape in ((1, 4), (1, 440)) and ids.dtype == torch.long
        calls.append(budget)
        length = state.input_embeds.shape[0]
        chosen = tuple(range(length - 1, length - min(3, budget) - 1, -1))
        return torch.tensor(chosen, dtype=torch.int32) if state is senders[0] else chosen

    for representation, options in (("embeddings", {}), ("token_ids+continuous", {"budget": 1})):
        inboxes: list[list[Delivery]] = [[], []]
        warning = pytest.warns(UserWarning, match="untested") if options else nullcontext()
        with warning:
            transfer(
                senders,
                [inbox.append for inbox in inboxes],
                ["Who owns Cedar ?", "When Birch launches ? " * 110],
                selector=recent,
                representation=representation,
                **options,
            )
        count = 1 if options else 3
        expected = torch.cat([state.input_embeds[-count:] for state in senders])[None]
        for inbox in inboxes:
            receiver = copy.deepcopy(senders[0].model)
            assert len(inbox) == 2
            for delivery in inbox:
                assert [message.positions for message in delivery.messages] == [count, count]
                payload = delivery.for_hf(receiver.get_input_embeddings().weight)
                assert torch.equal(payload["inputs_embeds"], expected)
                query = senders[0].request_ids(delivery.request)
                actual = _decode(receiver, payload, query, split=True)
                reference = _decode(receiver, {"inputs_embeds": expected}, query, split=False)
                torch.testing.assert_close(actual[0], reference[0], rtol=1e-5, atol=1e-6)
                assert actual[1] == reference[1]
    assert calls == [21, 25, 21, 25, 1, 1, 1, 1]
    with pytest.raises(ValueError, match="context limit"):
        transfer(senders, [].append, "When Birch launches ? " * 110)
    for state, snapshot in zip(senders, before, strict=True):
        _assert_unchanged(state, snapshot)


def test_relative_and_bounded_chain_budgets_survive_native_receiver_continuation(
    senders: list[SenderState],
) -> None:
    model = senders[0].model
    weight = model.get_input_embeddings().weight
    query = senders[0].request_ids("Who owns Cedar ?")
    for options, counts in (({}, [21, 28, 35, 42]), ({"budget": 100}, [82, 100, 100, 100])):
        state = senders[0]
        for count in counts:
            before = _snapshot(state)
            inbox: list[Delivery] = []
            transfer(state, inbox.append, query, **options)
            assert inbox[0].messages[0].positions == count
            scores = _eager_scores(state, query).clone()
            scores[0] = scores.max()
            length = state.input_embeds.shape[0]
            keep = fixed_span_keep(scores, count, (0, length - 2, length - 1), span_size=16)
            payload = inbox[0].for_hf(weight)
            assert torch.equal(payload["inputs_embeds"][0], state.input_embeds[keep])
            actual = _decode(model, payload, query, split=True)
            reference = _decode(model, payload, query, split=False)
            torch.testing.assert_close(actual[0], reference[0], rtol=1e-5, atol=1e-6)
            assert actual[1] == reference[1]
            _assert_unchanged(state, before)
            with torch.inference_mode():
                rows = torch.cat(
                    (
                        payload["inputs_embeds"],
                        model.get_input_embeddings()(torch.arange(23)[None]),
                    ),
                    dim=1,
                )
                out = model.model(inputs_embeds=rows, use_cache=True)
                for _ in range(2):
                    thought = out.last_hidden_state[:, -1:]
                    rows = torch.cat((rows, thought), dim=1)
                    out = model.model(
                        inputs_embeds=thought, past_key_values=out.past_key_values, use_cache=True
                    )
            state = SenderState(
                model, out.past_key_values, rows[0], latent_steps=2, inherited_positions=count
            )


def test_mixed_ids_and_continuous_rows_preserve_receiver_continuation(
    senders: list[SenderState],
) -> None:
    for state in senders:
        state.token_ids = state.token_ids.clone()
        state.token_ids[0] = -1
    dense: list[Delivery] = []
    mixed: list[Delivery] = []
    query = "Who owns Cedar ?"
    transfer(senders, dense.append, query, budget=35)
    with pytest.warns(UserWarning, match="untested"):
        transfer(senders, mixed.append, query, budget=35, representation="token_ids+continuous")
    model = senders[0].model
    weight = model.get_input_embeddings().weight
    assert torch.equal(
        dense[0].for_vllm(weight)["prompt_embeds"], mixed[0].for_vllm(weight)["prompt_embeds"]
    )
    for left, right in zip(dense[0].messages, mixed[0].messages, strict=True):
        assert right.nbytes < left.nbytes and right.positions == left.positions
        assert right.continuous_rows.shape[0] == 3
        assert right.token_ids[0] == -1 and bool((right.token_ids[1:-2] >= 0).all())
    query_ids = senders[0].request_ids(query)
    actual = _decode(model, mixed[0].for_hf(weight), query_ids, split=True)
    reference = _decode(model, dense[0].for_hf(weight), query_ids, split=True)
    assert torch.equal(actual[0], reference[0]) and actual[1] == reference[1]


def test_failed_capture_restores_state_and_delivery_failures_propagate(
    senders: list[SenderState], monkeypatch: Any
) -> None:
    before = [_snapshot(state) for state in senders]
    inbox: list[Delivery] = []

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("injected failure")

    with monkeypatch.context() as patch:
        patch.setattr(reducers, "fold_capture_energies", fail)
        with pytest.raises(RuntimeError, match="injected failure"):
            transfer(senders, inbox.append, "Who owns Cedar ?", budget=35)
    assert inbox == []
    for state, snapshot in zip(senders, before, strict=True):
        _assert_unchanged(state, snapshot)
    transfer(senders, inbox.append, "Who owns Cedar ?", budget=35)
    assert len(inbox) == 1
    with pytest.raises(RuntimeError, match="injected failure"):
        transfer(senders, [inbox.append, fail], "Who owns Cedar ?", budget=35)
    assert len(inbox) == 2
    for invalid in (
        None,
        [],
        [0, 0],
        [-1],
        [senders[1].input_embeds.shape[0]],
        [True],
        [1.0],
        torch.tensor([0.0]),
        torch.tensor([[0]]),
        list(range(36)),
    ):

        def bad_second(
            state: SenderState, ids: torch.Tensor, budget: int, bad: Any = invalid
        ) -> Any:
            return [0] if state is senders[0] else bad

        with pytest.raises(ValueError, match="selector"):
            transfer(senders, inbox.append, "Who owns Cedar ?", budget=35, selector=bad_second)
        assert len(inbox) == 2
    with pytest.raises(ValueError, match="callable"):
        transfer(senders, inbox.append, "Who owns Cedar ?", selector="cacheback")


@pytest.mark.parametrize("bad", ["protected-budget", "stale-cache", "wrong-token-ids"])
def test_invalid_state_is_rejected_before_delivery(senders: list[SenderState], bad: str) -> None:
    state = senders[0]
    budget = 35
    if bad == "protected-budget":
        budget = 2
    elif bad == "stale-cache":
        state = replace(state, input_embeds=state.input_embeds[:-1], token_ids=state.token_ids[:-1])
    else:
        state = replace(
            state, token_ids=torch.cat((torch.zeros(80, dtype=torch.long), torch.tensor([-1, -1])))
        )
    inbox: list[Delivery] = []
    if bad == "protected-budget":
        for options in (
            {"ratio": 100},
            {"ratio": 4, "budget": 35},
            {"ratio": 0},
            {"ratio": True},
            {"ratio": 1.5},
            {"budget": -1},
        ):
            with pytest.raises(ValueError):
                transfer(state, inbox.append, "Who owns Cedar ?", **options)
        for inherited in (-1, True, 81):
            with pytest.raises(ValueError):
                transfer(
                    replace(state, inherited_positions=inherited),
                    inbox.append,
                    "Who owns Cedar ?",
                )
    with pytest.raises(ValueError):
        transfer(
            state,
            inbox.append,
            "Who owns Cedar ?",
            budget=budget,
            representation="token_ids+continuous",
        )
    assert inbox == []
