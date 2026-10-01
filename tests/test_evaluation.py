"""The small selector comparison runs through native decoding and produces reusable results."""

import asyncio
import json
from pathlib import Path

import pytest
from examples.evaluate_selectors import evaluate, load_cases, recent

from rclc import SenderState
from rclc.selectors import cacheback, chunkkv, qsnap


def test_selector_comparison(senders: list[SenderState], tmp_path: Path) -> None:
    state = senders[0]
    tokenizer = state.tokenizer
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}?"
    path = tmp_path / "cases.jsonl"
    case = {
        "documents": ["Who owns Cedar ? " * 12, "When Birch launches ? " * 12],
        "question": "Who owns Cedar ?",
        "answers": ["Cedar"],
    }
    path.write_text(json.dumps(case) + "\n")
    cases = load_cases(path)
    selectors = {"cacheback": cacheback, "recent": recent, "qsnap": qsnap, "chunkkv": chunkkv}
    results = asyncio.run(
        evaluate(
            state.model,
            tokenizer,
            cases,
            selectors,
            budget=16,
            max_new_tokens=2,
        )
    )
    assert [r["selector"] for r in results] == list(selectors)
    assert all(r["retained_positions"] == [16, 16] for r in results)
    assert all(r["payload_bytes"] == 32 * 64 * 4 for r in results)
    assert all(r["transfer_seconds"] > 0 and r["generation_seconds"] > 0 for r in results)
    assert all(isinstance(r["correct"], bool) for r in results)
    case["answers"] = [results[0]["answer"]]
    repeated = asyncio.run(
        evaluate(
            state.model,
            tokenizer,
            [case],
            {"cacheback": cacheback},
            budget=16,
            max_new_tokens=2,
        )
    )
    assert repeated[0]["correct"]
    assert json.loads(json.dumps(results)) == results
    path.write_text('{"question":"Cedar?","documents":[],"answers":["Cedar"]}')
    with pytest.raises(ValueError, match="documents"):
        load_cases(path)
