"""An async fan-in journey with native continuation, failure and cancellation."""

import asyncio
import threading
from functools import partial
from typing import Any

import pytest
import torch

import rclc
from rclc._cache import cache_kv
from rclc.selectors import cacheback


def test_async_handoffs_keep_loop_live_and_cancel_without_delivery(
    senders: list[rclc.SenderState],
    monkeypatch: Any,
) -> None:
    model, tokenizer = senders[0].model, senders[0].tokenizer
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}?"
    snapshots = [[(k.clone(), v.clone()) for k, v in cache_kv(s.past_key_values)] for s in senders]
    requests = ["Who owns Cedar ?", "When Birch launches ?"]
    reference: list[rclc.Delivery] = []
    rclc.transfer_sync(
        senders, reference.append, requests, selector=partial(cacheback, span_size=4)
    )

    async def journey() -> None:
        loop = asyncio.get_running_loop()
        main_thread = threading.get_ident()
        receiver = rclc.bind(model, tokenizer, max_new_tokens=2)
        async_received: list[rclc.Delivery] = []
        sync_received: list[rclc.Delivery] = []

        async def receive(delivery: rclc.Delivery) -> None:
            assert threading.get_ident() == main_thread
            await asyncio.sleep(0)
            async_received.append(delivery)

        await rclc.transfer(
            senders,
            [receiver, sync_received.append, receive],
            requests,
            selector=partial(cacheback, span_size=4),
        )
        assert len(receiver) == len(reference) == len(async_received) == len(sync_received) == 2
        for actual, expected in zip(async_received, reference, strict=True):
            assert actual.request == expected.request
            for message, other in zip(actual.messages, expected.messages, strict=True):
                torch.testing.assert_close(message.continuous_rows, other.continuous_rows)
            inputs = receiver.pop()
            output = await asyncio.to_thread(
                model.generate,
                **inputs,
                do_sample=False,
                pad_token_id=0,
                return_dict_in_generate=True,
            )
            receiver.update(inputs=inputs, generation=output)
        returned: list[rclc.Delivery] = []
        await rclc.transfer(receiver, returned.append, requests[0])
        assert returned[0].messages[0].positions > 0

        started = asyncio.Event()
        release = threading.Event()
        active = 0
        calls = 0

        def paused_selector(state: rclc.SenderState, ids: torch.Tensor, budget: int) -> list[int]:
            nonlocal active, calls
            assert threading.get_ident() != main_thread
            active += 1
            assert active == 1
            calls += 1
            loop.call_soon_threadsafe(started.set)
            try:
                assert release.wait(10), "event loop did not release selection"
                return cacheback(state, ids, budget, span_size=4)
            finally:
                active -= 1

        first = asyncio.create_task(
            rclc.transfer(senders, receive, requests, selector=paused_selector)
        )
        await asyncio.wait_for(started.wait(), 10)
        second = asyncio.create_task(
            rclc.transfer(senders, receive, requests, selector=paused_selector)
        )
        await asyncio.sleep(0)
        assert not first.done() and active == 1
        release.set()
        await asyncio.gather(first, second)
        assert calls == 8 and active == 0 and len(async_received) == 6

        started.clear()
        release.clear()
        before = len(async_received)
        cancelled = asyncio.create_task(
            rclc.transfer(senders[0], [receiver, receive], requests[0], selector=paused_selector)
        )
        await asyncio.wait_for(started.wait(), 10)
        cancelled.cancel()
        await asyncio.sleep(0)
        assert not cancelled.done() and active == 1
        cancelled.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert active == 0 and len(receiver) == 0 and len(async_received) == before

        from rclc import hf

        state = receiver.sender_state()
        started.clear()
        release.clear()
        forward = hf.forward_rows

        def paused_append(*args: Any, **kwargs: Any) -> Any:
            loop.call_soon_threadsafe(started.set)
            assert release.wait(10), "event loop blocked during append"
            return forward(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(hf, "forward_rows", paused_append)
            task = asyncio.create_task(receiver.append("Cedar launches"))
            await asyncio.wait_for(started.wait(), 10)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert receiver.sender_state() is state
        assert state.past_key_values.get_seq_length() == len(state.input_embeds)
        await receiver.append("Cedar launches")
        assert torch.equal(receiver.sender_state().input_embeds[:-2], state.input_embeds)

        def fail(*args: Any) -> list[int]:
            raise ValueError("selection failed")

        with pytest.raises(ValueError, match="selection failed"):
            await rclc.transfer(senders, [receiver, receive], requests, selector=fail)
        assert len(receiver) == 0 and len(async_received) == before

        async def failed_receiver(delivery: rclc.Delivery) -> None:
            raise RuntimeError("receiver failed")

        with pytest.raises(RuntimeError, match="receiver failed"):
            await rclc.transfer(senders, [receive, failed_receiver], requests[0])
        assert len(async_received) == before + 1
        await rclc.transfer(senders, receiver, requests[0])
        assert len(receiver) == 1
        with pytest.raises(RuntimeError, match=r"await rclc\.transfer"):
            rclc.transfer_sync(senders, receive, requests[0])

    asyncio.run(journey())
    for sender, snapshot in zip(senders, snapshots, strict=True):
        for actual, expected in zip(cache_kv(sender.past_key_values), snapshot, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert model.config._attn_implementation == "sdpa"


def test_async_thoughts_support_custom_methods_and_cancellation(
    senders: list[rclc.SenderState],
    monkeypatch: Any,
) -> None:
    from rclc import latent

    state = senders[0]
    query = "Who owns Cedar ?"
    snapshots = [(k.clone(), v.clone()) for k, v in cache_kv(state.past_key_values)]
    rows = state.input_embeds.clone()

    async def journey() -> None:
        loop = asyncio.get_running_loop()
        direct = await rclc.latent_mass(state, steps=2, request=query)
        reference = rclc.latent_mass_sync(state, steps=2, request=query)
        torch.testing.assert_close(direct.input_embeds, reference.input_embeds, rtol=0, atol=0)
        for actual, expected in zip(
            cache_kv(direct.past_key_values), cache_kv(reference.past_key_values), strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert await rclc.latent_mass(state, steps=0) is state
        with pytest.raises(ValueError, match="max_positions"):
            await rclc.latent_mass(state, steps=3, max_positions=2)

        async def custom(current: rclc.SenderState, **kwargs: Any) -> rclc.SenderState:
            assert asyncio.get_running_loop() is loop
            await asyncio.sleep(0)
            return await rclc.latent_mass(current, steps=2, **kwargs)

        for context in ("full", "full_with_request", "selected", "selected_with_request"):
            actual: list[rclc.Delivery] = []
            expected: list[rclc.Delivery] = []
            options = dict(reasoning_context=context, reasoning_budget=2, budget=24)
            await rclc.transfer(senders, actual.append, [query, query], reasoning=custom, **options)
            await rclc.transfer(
                senders,
                expected.append,
                [query, query],
                reasoning=partial(rclc.latent_mass_sync, steps=2),
                **options,
            )
            assert len(actual) == len(expected) == 2
            for delivery, other in zip(actual, expected, strict=True):
                for message, baseline in zip(delivery.messages, other.messages, strict=True):
                    torch.testing.assert_close(message.continuous_rows, baseline.continuous_rows)
                inputs = delivery.for_hf(state.model.get_input_embeddings().weight)
                output = state.model.generate(**inputs, max_new_tokens=2, do_sample=False)
                assert output.shape == (1, 2)

        started, stopped = asyncio.Event(), asyncio.Event()
        waiting = asyncio.Event()
        inbox: list[rclc.Delivery] = []

        async def wait_for_tool(current: rclc.SenderState, **kwargs: Any) -> rclc.SenderState:
            current.input_embeds.zero_()
            started.set()
            try:
                await waiting.wait()
                return current
            finally:
                stopped.set()

        task = asyncio.create_task(
            rclc.transfer(state, inbox.append, query, reasoning=wait_for_tool)
        )
        await asyncio.wait_for(started.wait(), 10)
        independent: list[rclc.Delivery] = []
        await asyncio.wait_for(rclc.transfer(state, independent.append, query), 10)
        assert independent and not inbox
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stopped.is_set() and not inbox
        torch.testing.assert_close(state.input_embeds, rows, rtol=0, atol=0)

        waiting.set()
        with pytest.raises(ValueError, match="preserve the sender prefix"):
            await rclc.transfer(state, inbox.append, query, reasoning=wait_for_tool)
        assert not inbox

        started.clear()
        release = threading.Event()
        forward = latent.forward_rows

        def paused_forward(*args: Any, **kwargs: Any) -> Any:
            loop.call_soon_threadsafe(started.set)
            assert release.wait(10), "event loop blocked during latent rollout"
            return forward(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(latent, "forward_rows", paused_forward)
            task = asyncio.create_task(rclc.latent_mass(state, steps=2))
            await asyncio.wait_for(started.wait(), 10)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert (await rclc.latent_mass(state, steps=1)).latent_steps == state.latent_steps + 1

    asyncio.run(journey())
    torch.testing.assert_close(state.input_embeds, rows, rtol=0, atol=0)
    for actual, expected in zip(cache_kv(state.past_key_values), snapshots, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
