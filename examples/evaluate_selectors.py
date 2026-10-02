"""Compare CacheBack and a selector callable on a small local JSONL question set."""

import argparse
import asyncio
import importlib
import json
import re
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import rclc
from rclc.selectors import Selector, cacheback, chunkkv, qsnap


def recent(state: rclc.SenderState, request_ids: torch.Tensor, budget: int) -> list[int]:
    """Keep the most recent positions as a simple comparison."""
    return list(range(len(state.input_embeds) - budget, len(state.input_embeds)))


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Read documents, question and acceptable answers from each nonblank JSONL line."""
    cases = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        if not isinstance(case, dict) or not isinstance(case.get("question"), str):
            raise ValueError(f"line {number}: provide a question string")
        for field in ("documents", "answers"):
            values = case.get(field)
            if (
                not isinstance(values, list)
                or not values
                or not all(isinstance(value, str) and value.strip() for value in values)
            ):
                raise ValueError(f"line {number}: {field} must be a nonempty list of strings")
        if not case["question"].strip():
            raise ValueError(f"line {number}: question must be nonempty")
        cases.append(case)
    if not cases:
        raise ValueError("provide at least one evaluation case")
    return cases


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text.casefold()).split())


def _synchronize(model: Any) -> None:
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)


async def evaluate(
    model: Any,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    selectors: dict[str, Selector],
    *,
    ratio: int | None = None,
    budget: int | None = None,
    max_new_tokens: int = 64,
) -> list[dict[str, Any]]:
    """Compare identical cached sources, requests and generation settings for each selector."""
    results = []
    for index, case in enumerate(cases):
        senders = [
            rclc.sender_from_hf(model, tokenizer, document) for document in case["documents"]
        ]
        for name, selector in selectors.items():
            receiver = rclc.bind(
                model,
                tokenizer,
                max_new_tokens=max_new_tokens,
                messages=[{"role": "system", "content": "Answer with only the short answer."}],
            )
            deliveries: list[rclc.Delivery] = []
            _synchronize(model)
            started = time.perf_counter()
            await rclc.transfer(
                senders,
                [receiver, deliveries.append],
                case["question"],
                selector=selector,
                ratio=ratio,
                budget=budget,
            )
            _synchronize(model)
            selected = time.perf_counter()
            inputs = receiver.pop()
            output = model.generate(
                **inputs,
                do_sample=False,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            _synchronize(model)
            generated = time.perf_counter()
            receiver.update(inputs=inputs, generation=output)
            answer = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
            results.append(
                {
                    "case": index,
                    "selector": name,
                    "question": case["question"],
                    "answer": answer,
                    "expected": case["answers"],
                    "correct": _normalise(answer) in {_normalise(a) for a in case["answers"]},
                    "source_positions": [len(s.input_embeds) for s in senders],
                    "retained_positions": [m.positions for m in deliveries[0].messages],
                    "payload_bytes": sum(m.nbytes for m in deliveries[0].messages),
                    "transfer_seconds": selected - started,
                    "generation_seconds": generated - selected,
                }
            )
    return results


def main() -> None:
    """Load one dense Qwen3 model and write the raw comparison results as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path(__file__).with_name("selector_cases.jsonl")
    )
    parser.add_argument(
        "--selector",
        default="recent",
        help="'recent', 'qsnap', 'chunkkv' or module:callable to compare with CacheBack",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    caps = parser.add_mutually_exclusive_group()
    caps.add_argument("--ratio", type=int)
    caps.add_argument("--budget", type=int)
    options = parser.parse_args()
    builtin: dict[str, Any] = {"recent": recent, "qsnap": qsnap, "chunkkv": chunkkv}
    selector: Any = builtin.get(options.selector)
    if selector is None:
        module, _, name = options.selector.rpartition(":")
        if not module:
            parser.error("--selector must be 'recent', 'qsnap', 'chunkkv' or module:callable")
        selector = getattr(importlib.import_module(module), name)
    if not callable(selector):
        parser.error("--selector must resolve to a callable")
    if selector is cacheback:
        parser.error("--selector must differ from CacheBack, which is always included")
    selectors: dict[str, Selector] = {"cacheback": cacheback, options.selector: selector}
    cases = load_cases(options.data)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(options.model, revision=options.revision)
    model = (
        AutoModelForCausalLM.from_pretrained(
            options.model,
            revision=options.revision,
            attn_implementation="sdpa",
        )
        .to(options.device)
        .eval()
    )
    results = asyncio.run(
        evaluate(
            model,
            tokenizer,
            cases,
            selectors,
            ratio=options.ratio,
            budget=options.budget,
            max_new_tokens=options.max_new_tokens,
        )
    )
    print(
        json.dumps(
            {
                "model": options.model,
                "revision": getattr(model.config, "_commit_hash", None),
                "device": options.device,
                "dtype": str(model.dtype),
                "torch": torch.__version__,
                "transformers": version("transformers"),
                "ratio": options.ratio or (4 if options.budget is None else None),
                "budget": options.budget,
                "max_new_tokens": options.max_new_tokens,
                "scores": {
                    name: {
                        "correct": sum(r["correct"] for r in results if r["selector"] == name),
                        "total": len(cases),
                    }
                    for name in selectors
                },
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
