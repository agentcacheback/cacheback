"""Inspect real transfers without changing their payloads or native continuations."""

import asyncio
import json
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import pytest
import torch

import rcc


def test_recorded_selection_survives_fan_in_reasoning_and_native_generation(
    senders: list[rcc.SenderState], tmp_path: Path, monkeypatch: Any
) -> None:
    model, tokenizer = senders[0].model, senders[0].tokenizer
    tokenizer.add_tokens(["<script>alert(1)</script>"])
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}?"
    source = rcc.sender_from_hf(
        model, tokenizer, "Who owns Cedar ? Who owns <script>alert(1)</script> ?"
    )
    sources = [source, rcc.sender_from_hf(model, tokenizer, "When Birch launches ? " * 3)]

    def choose(state: rcc.SenderState, query: torch.Tensor, budget: int) -> list[int]:
        return [len(state.input_embeds) - 1, 6, 2, 0]

    async def journey() -> None:
        for context in (None, "full", "full_with_request", "selected", "selected_with_request"):
            options = (
                {}
                if context is None
                else {
                    "reasoning": partial(rcc.latent_mass, steps=2),
                    "reasoning_context": context,
                    "reasoning_budget": 2,
                }
            )
            for representation in ("embeddings", "token_ids+continuous"):
                receiver = rcc.bind(model, tokenizer, max_new_tokens=2)
                baseline: list[rcc.Delivery] = []
                recorded: list[rcc.Delivery] = []
                warning = (
                    pytest.warns(UserWarning) if representation != "embeddings" else nullcontext()
                )
                with warning:
                    await rcc.transfer(
                        sources,
                        baseline.append,
                        ["Who owns Cedar ?", "When Birch launches ?"],
                        budget=8,
                        selector=choose,
                        representation=representation,
                        **options,
                    )
                    await rcc.transfer(
                        sources,
                        [receiver, recorded.append],
                        ["Who owns Cedar ?", "When Birch launches ?"],
                        budget=8,
                        selector=choose,
                        representation=representation,
                        record_selection=True,
                        **options,
                    )
                with monkeypatch.context() as patch:
                    patch.setattr(
                        model.model, "forward", lambda **kw: pytest.fail("inspection forwarded")
                    )
                    report = receiver.inspect(selection=True)
                    rendered = receiver.selection_html()
                    assert receiver.selection_html(1) == recorded[1].selection_html()
                assert len(receiver) == 2 and "selection" not in receiver.inspect()["deliveries"][0]
                json.dumps(report)
                report["deliveries"][0]["selection"][0]["spans"][0]["text"] = "changed"
                assert receiver.selection_html() == rendered
                assert "<script>" not in rendered and "&lt;script&gt;" in rendered
                assert "Request:</strong> Who owns Cedar ?" in rendered
                assert "Sender 1" in rendered and "Sender 2" in rendered and "<mark" in rendered
                (tmp_path / "selection.html").write_text(rendered)
                for original, delivery in zip(baseline, recorded, strict=True):
                    with pytest.raises(ValueError, match="record_selection=True"):
                        original.selection_html()
                    for source_state, left, right in zip(
                        sources, original.messages, delivery.messages, strict=True
                    ):
                        assert left.selection is None and left.nbytes == right.nbytes
                        weight = model.get_input_embeddings().weight
                        assert torch.equal(left.materialize(weight), right.materialize(weight))
                        selection = right.selection
                        assert selection is not None and selection.budget == 8
                        pre = context in ("full", "full_with_request")
                        post = context in ("selected", "selected_with_request")
                        assert (
                            selection.source_positions == len(source_state.input_embeds) + 2 * pre
                        )
                        assert selection.added_latent_positions == 2 * post
                        assert selection.indices == (0, 2, 6, selection.source_positions - 1)
                        if not pre:
                            text = "".join(span["text"] or "" for span in selection.spans)
                            assert text == tokenizer.decode(source_state.token_ids)
                        assert (
                            sum(span["stop"] - span["start"] for span in selection.spans)
                            == selection.source_positions
                        )
                        assert selection.spans[-1]["kind"] == ("latent" if pre else "token")
                        assert (
                            right.positions
                            == len(selection.indices) + selection.added_latent_positions
                        )
                peer = rcc.bind(model, tokenizer, max_new_tokens=2)
                peer(baseline[0])
                left, right = peer.pop(), receiver.pop()
                a = model.generate(**left, do_sample=False, pad_token_id=0)
                b = model.generate(**right, do_sample=False, pad_token_id=0)
                assert torch.equal(a, b)
                assert len(receiver) == 1
                with pytest.raises(ValueError, match="request_index"):
                    receiver.selection_html(1)
        unknown: list[rcc.Delivery] = []
        state = replace(senders[0], token_ids=None)
        await rcc.transfer(
            state, unknown.append, "Who owns Cedar ?", selector=choose, record_selection=True
        )
        selection = unknown[0].messages[0].selection
        assert selection is not None and selection.spans[0]["kind"] == "unknown"
        assert selection.spans[-1]["kind"] == "latent"
        assert all(span["text"] is None for span in selection.spans)
        ids = senders[0].token_ids.clone()
        ids[0] = -1
        continuous: list[rcc.Delivery] = []
        await rcc.transfer(
            replace(senders[0], token_ids=ids),
            continuous.append,
            source.request_ids("Who owns Cedar ?"),
            selector=choose,
            record_selection=True,
        )
        assert continuous[0].messages[0].selection.spans[0]["kind"] == "continuous"
        assert "Supplied as token IDs" in continuous[0].selection_html()

    asyncio.run(journey())
