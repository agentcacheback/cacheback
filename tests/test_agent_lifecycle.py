"""Continue, inspect, save and restore a bound agent without losing continuous state."""

import asyncio
import copy
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import rclc
from rclc import checkpoint
from rclc._cache import cache_kv


def test_saved_agent_continues_and_transfers(
    senders: list[rclc.SenderState], tmp_path: Path, monkeypatch: Any
) -> None:
    model, tokenizer = senders[0].model, senders[0].tokenizer
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}?"
    path = tmp_path / "agent.safetensors"

    def generate(inputs: dict[str, Any]) -> Any:
        return model.generate(
            **inputs, do_sample=False, pad_token_id=0, return_dict_in_generate=True
        )

    async def journey() -> None:
        receiver = rclc.bind(model, tokenizer, max_new_tokens=2)
        inbox: list[rclc.Delivery] = []
        await rclc.transfer(senders, [receiver, inbox.append], "Who owns Cedar ?")
        with monkeypatch.context() as patch:
            patch.setattr(model.model, "forward", lambda **kw: pytest.fail("inspection forwarded"))
            report = receiver.inspect()
            assert rclc.check(receiver)["ok"]
        assert report["queued_requests"] == len(receiver) == 1
        delivery = report["deliveries"][0]
        assert delivery["senders"] == 2
        assert delivery["retained_positions"] == [m.positions for m in inbox[0].messages]
        assert delivery["payload_bytes"] == sum(m.nbytes for m in inbox[0].messages)
        positions = sum(delivery["retained_positions"]) + delivery["prompt_positions"]
        assert delivery["remaining_positions"] == 512 - positions - 2
        assert json.loads(json.dumps(report)) == report
        with pytest.raises(ValueError, match="bound agent"):
            rclc.check(receiver, backend="vllm")
        with pytest.raises(ValueError, match="queued"):
            await receiver.save(path)
        inputs = receiver.pop()
        with pytest.raises(ValueError, match="pending"):
            await receiver.load(path)
        receiver.update(inputs=inputs, generation=generate(inputs))
        await receiver.append("Cedar launches")
        state = receiver.sender_state()
        with pytest.raises(ValueError, match="context space"):
            await receiver.inputs(max_new_tokens=512)
        assert not receiver.inspect()["pending_generation"]
        await receiver.save(path)

        original_file = path.read_bytes()
        with monkeypatch.context() as patch:

            def failed_write(*args: Any, **kwargs: Any) -> None:
                raise OSError("disk full")

            patch.setattr(checkpoint, "save_file", failed_write)
            with pytest.raises(OSError, match="disk full"):
                await receiver.save(path)
        assert path.read_bytes() == original_file and list(tmp_path.iterdir()) == [path]
        saved = load_file(path)
        assert saved["rows"].device.type == "cpu" and "past_key_values" not in saved
        restored = rclc.bind(copy.deepcopy(model), tokenizer, max_new_tokens=2)
        await restored.load(path)
        loaded = restored.sender_state()
        assert loaded.inherited_positions == state.inherited_positions
        assert torch.equal(loaded.input_embeds, state.input_embeds)
        assert torch.equal(loaded.token_ids, state.token_ids)
        fresh = model.model(inputs_embeds=state.input_embeds[None], use_cache=True)
        for actual, expected in zip(
            cache_kv(loaded.past_key_values), cache_kv(fresh.past_key_values), strict=True
        ):
            torch.testing.assert_close(actual, expected)
        left, right = await receiver.inputs(), await restored.inputs()
        assert torch.equal(left["inputs_embeds"], right["inputs_embeds"])
        assert restored.inspect()["pending_generation"] and len(restored) == 0
        with pytest.raises(ValueError, match=r"agent\.update"):
            await restored.inputs()
        before = generate(left)
        after = restored.model.generate(
            **right, do_sample=False, pad_token_id=0, return_dict_in_generate=True
        )
        assert torch.equal(before.sequences, after.sequences)
        receiver.update(inputs=left, generation=before)
        restored.update(inputs=right, generation=after)
        assert restored.sender_state().inherited_positions == state.inherited_positions
        with monkeypatch.context() as patch:
            patch.setattr(checkpoint, "_hash_weights", lambda _: pytest.fail("rehashed weights"))
            await restored.save(path)
        await rclc.transfer(restored, receiver, "When Birch launches ?")
        assert len(receiver) == 1

        corrupted = load_file(path)
        corrupted["rows"][0, 0] = float("nan")
        bad_path = tmp_path / "bad.safetensors"
        with safe_open(path, framework="pt") as archive:
            metadata = archive.metadata()
        save_file(corrupted, bad_path, metadata=metadata)
        retained = restored.sender_state()
        with monkeypatch.context() as patch:
            patch.setattr(restored.model.model, "forward", lambda **kw: pytest.fail("bad prefill"))
            with pytest.raises(ValueError, match="nonfinite"):
                await restored.load(bad_path)
        assert restored.sender_state() is retained
        edited = copy.deepcopy(model)
        with torch.no_grad():
            edited.get_input_embeddings().weight[0, 0].add_(1)
        changed = rclc.bind(edited, tokenizer, prompt=torch.tensor([[10, 11]]))
        retained = changed.sender_state()
        with pytest.raises(ValueError, match="identical weights"):
            await changed.load(path)
        assert changed.sender_state() is retained
        with torch.no_grad():
            restored.model.get_input_embeddings().weight[0, 0].add_(1)
        with pytest.raises(ValueError, match="identical weights"):
            await restored.load(path)

    asyncio.run(journey())
    command = subprocess.run(
        [sys.executable, "-m", "rclc", "doctor"], capture_output=True, text=True, check=True
    )
    assert json.loads(command.stdout)["scope"] == "environment"
    failed_command = subprocess.run(
        [sys.executable, "-m", "rclc", "doctor", "--backend", "vllm"],
        env={**os.environ, "VLLM_ENABLE_V1_MULTIPROCESSING": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed_command.returncode == 1
    assert any("MULTIPROCESSING" in issue for issue in json.loads(failed_command.stdout)["issues"])
    from rclc import diagnostics

    with monkeypatch.context() as patch:
        patch.setattr(diagnostics.torch.cuda, "is_available", lambda: False)
        patch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
        versions = {"torch": "2.4.0", "transformers": "4.57.1", "vllm": "0.11.1"}
        patch.setattr(diagnostics, "_version", versions.get)
        report = rclc.check(backend="vllm")
    assert not report["ok"] and any("CUDA" in issue for issue in report["issues"])
    assert any("torch==2.9.0" in issue for issue in report["issues"])


def test_cancelled_restore_keeps_current_state(
    senders: list[rclc.SenderState], tmp_path: Path, monkeypatch: Any
) -> None:

    model, tokenizer = senders[0].model, senders[0].tokenizer

    async def journey() -> None:
        agent = rclc.bind(model, tokenizer, prompt=torch.tensor([[10, 11, 12]]))
        path = tmp_path / "saved.safetensors"
        await agent.save(path)
        await agent.append(torch.tensor([[13]]))
        current = agent.sender_state()
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        forward = checkpoint.forward_rows

        def paused(*args: Any, **kwargs: Any) -> Any:
            loop.call_soon_threadsafe(started.set)
            assert release.wait(10)
            return forward(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(checkpoint, "forward_rows", paused)
            task = asyncio.create_task(agent.load(path))
            await asyncio.wait_for(started.wait(), 10)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert agent.sender_state() is current
        await agent.load(path)
        assert len(agent.sender_state().input_embeds) == 3

    asyncio.run(journey())
