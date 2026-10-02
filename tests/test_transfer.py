"""Public handoffs with native capture, independent attention checks and receiver continuation."""

import copy
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from typing import Any

import pytest
import torch
import torch.nn.functional as functional

from rclc import Delivery, SenderState, transfer_sync, transport
from rclc._cache import cache_kv
from rclc.selectors import _capture, _support, cacheback, chunkkv, qsnap
from rclc.selectors.cacheback import cacheback_scores
from rclc.selectors.chunkkv import chunkkv_scores
from rclc.selectors.fixed_spans import fixed_span_keep
from rclc.selectors.qsnap import qsnap_scores


def _snapshot(state: SenderState) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(k.clone(), v.clone()) for k, v in cache_kv(state.past_key_values)]


def _assert_unchanged(state: SenderState, before: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    for actual, expected in zip(cache_kv(state.past_key_values), before, strict=True):
        assert all(torch.equal(a, e) for a, e in zip(actual, expected, strict=True))
    assert state.model.config._attn_implementation == "sdpa"
    assert _capture._active is None


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
def _eager_scores(
    state: SenderState, query: torch.Tensor, *, snap_only: bool = False
) -> torch.Tensor:
    model = state.model
    length = state.input_embeds.shape[0]
    previous = model.config._attn_implementation
    _capture.set_attention_implementation(model, "eager")
    try:
        output = model.model(
            input_ids=query,
            past_key_values=copy.deepcopy(state.past_key_values),
            output_attentions=True,
            use_cache=True,
        )
    finally:
        _capture.set_attention_implementation(model, previous)
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
    if snap_only:
        return pooled_snap
    return pooled_snap * (row_energy / energy).square()


@torch.inference_mode()
def _eager_chunkkv(state: SenderState, window: int, chunk: int) -> torch.Tensor:
    model, rows = state.model, state.input_embeds
    prefix = rows.shape[0] - window
    past = copy.deepcopy(state.past_key_values)
    past.crop(prefix)
    previous = model.config._attn_implementation
    _capture.set_attention_implementation(model, "eager")
    try:
        output = model.model(
            inputs_embeds=rows[prefix:][None], past_key_values=past, output_attentions=True
        )
    finally:
        _capture.set_attention_implementation(model, previous)
    layers = []
    for attention in output.attentions:
        scored = attention[0, :, :, :prefix].float().mean(1)
        pooled = functional.avg_pool1d(scored[None], 5, stride=1, padding=2)[0]
        grouped = pooled.reshape(model.config.num_key_value_heads, -1, prefix).mean(1)
        layers.append(functional.pad(grouped, (0, window), value=float(grouped.max()) + 1))
    scores = torch.stack(layers).mean(dim=(0, 1))
    means = [scores[i : i + chunk].mean() for i in range(0, len(scores), chunk)]
    return torch.cat([m.expand(min(chunk, len(scores) - i * chunk)) for i, m in enumerate(means)])


def test_qsnap_and_chunkkv_match_eager_references_and_swap_into_transfer(
    senders: list[SenderState],
) -> None:
    state = senders[0]
    before = _snapshot(state)
    query = state.request_ids("Who owns Cedar ?")
    with torch.inference_mode():
        eager = _eager_scores(state, query, snap_only=True)
    torch.testing.assert_close(qsnap_scores(state, query), eager, rtol=2e-5, atol=1e-7)
    expected_chunks = _eager_chunkkv(state, 8, 4)
    actual_chunks = chunkkv_scores(state, chunk_size=4, window_size=8)
    torch.testing.assert_close(actual_chunks, expected_chunks, rtol=1e-5, atol=1e-7)
    for selector, scores, span in (
        (qsnap, eager, 16),
        (partial(chunkkv, chunk_size=4, window_size=8), expected_chunks, 4),
    ):
        pinned = scores.clone()
        pinned[0] = pinned.max()
        keep = fixed_span_keep(pinned, 28, (0, 80, 81), span_size=span)
        inbox: list[Delivery] = []
        transfer_sync(state, inbox.append, "Who owns Cedar ?", budget=28, selector=selector)
        rows = inbox[0].for_hf(state.model.get_input_embeddings().weight)["inputs_embeds"][0]
        assert torch.equal(rows, state.input_embeds[keep])
    assert set(range(74, 82)).issubset(fixed_span_keep(expected_chunks, 28, (0,), span_size=4))
    with pytest.raises(ValueError, match="window_size"):
        chunkkv_scores(state, window_size=82)
    builtins = (qsnap, partial(chunkkv, window_size=8), cacheback)
    assert all(transport._is_builtin_selector(selector) for selector in builtins)
    assert not transport._is_builtin_selector(lambda *args: [0])
    _assert_unchanged(state, before)


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
    transfer_sync(state, inbox.append, request, budget=budget, record_selection=True)
    assert len(inbox) == 1 and inbox[0].request == request
    expected_scores = _eager_scores(state, query).clone()
    torch.testing.assert_close(
        cacheback_scores(state, query), expected_scores, rtol=2e-5, atol=1e-7
    )
    expected_scores[0] = expected_scores.max()
    keep = fixed_span_keep(expected_scores, budget, (0, 80, 81), span_size=16)
    expected_rows = state.input_embeds[keep]
    selection = inbox[0].messages[0].selection
    assert selection is not None and selection.indices == tuple(keep)
    assert selection.source_positions == 82 and selection.budget == min(budget, 82)
    assert selection.added_latent_positions == 0
    assert selection.spans[-1]["kind"] == "latent" and selection.spans[-1]["kept"]
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
    transfer_sync(state, relative.append, request, ratio=ratio)
    assert relative[0].messages[0].selection is None
    relative_keep = fixed_span_keep(expected_scores, count, (0, 80, 81), span_size=16)
    relative_rows = relative[0].for_hf(receiver.get_input_embeddings().weight)["inputs_embeds"]
    assert torch.equal(relative_rows[0], state.input_embeds[relative_keep])
    narrow: list[Delivery] = []
    transfer_sync(
        state, narrow.append, request, ratio=ratio, selector=partial(cacheback, span_size=4)
    )
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
            transfer_sync(
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

    transfer_sync(senders, [inbox.append for inbox in inboxes], queries, selector=observe)
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
        _capture,
        "_register",
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
            transfer_sync(
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
        transfer_sync(senders, [].append, "When Birch launches ? " * 110)
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
            transfer_sync(state, inbox.append, query, **options)
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
    transfer_sync(senders, dense.append, query, budget=35)
    with pytest.warns(UserWarning, match="untested"):
        transfer_sync(
            senders, mixed.append, query, budget=35, representation="token_ids+continuous"
        )
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
        patch.setattr(_support._SupportFold, "__call__", fail)
        with pytest.raises(RuntimeError, match="injected failure"):
            transfer_sync(senders, inbox.append, "Who owns Cedar ?", budget=35)
    assert inbox == []
    for state, snapshot in zip(senders, before, strict=True):
        _assert_unchanged(state, snapshot)
    transfer_sync(senders, inbox.append, "Who owns Cedar ?", budget=35)
    assert len(inbox) == 1
    with pytest.raises(RuntimeError, match="injected failure"):
        transfer_sync(senders, [inbox.append, fail], "Who owns Cedar ?", budget=35)
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
            transfer_sync(senders, inbox.append, "Who owns Cedar ?", budget=35, selector=bad_second)
        assert len(inbox) == 2
    with pytest.raises(ValueError, match="callable"):
        transfer_sync(senders, inbox.append, "Who owns Cedar ?", selector="cacheback")


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
                transfer_sync(state, inbox.append, "Who owns Cedar ?", **options)
        for inherited in (-1, True, 81):
            with pytest.raises(ValueError):
                transfer_sync(
                    replace(state, inherited_positions=inherited),
                    inbox.append,
                    "Who owns Cedar ?",
                )
    with pytest.raises(ValueError):
        transfer_sync(
            state,
            inbox.append,
            "Who owns Cedar ?",
            budget=budget,
            representation="token_ids+continuous",
        )
    assert inbox == []
