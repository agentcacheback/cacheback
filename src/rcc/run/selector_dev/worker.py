"""One seat's capture or decode work over the questions it claims.

Each question is written once, atomically, so the two phases share the same
per-question bank and a seat that stops leaves nothing half written.
"""

from __future__ import annotations

import importlib
import io
import logging
import time
from pathlib import Path
from typing import Any

import torch

from rcc.benchmarks.fanoutqa.panel import Question
from rcc.benchmarks.fanoutqa.prompts import manager_prompt, question_token_ids
from rcc.benchmarks.fanoutqa.scoring import DEFAULT_SCORER_VERSION, score_text
from rcc.models.qwen import QWEN_FAMILY
from rcc.run.io import atomic_bytes, atomic_json, canonical_sha, sha256_file
from rcc.run.selector_dev.capture import selections
from rcc.run.selector_dev.contract import CELLS, MAX_ANSWER_TOKENS, load_plan, read_json
from rcc.run.selector_dev.engine import open_engine

LOG = logging.getLogger(__name__)
_CAPTURE_FILES = ("capture.json", "embeddings.pt")


def _tensor_file(path: Path, payload: Any) -> None:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    atomic_bytes(path, buffer.getvalue())


def _load_bank(base: Path, identity: dict[str, Any]) -> dict[str, Any]:
    banked = read_json(base / "capture.json")
    if (
        banked["identity"] != identity
        or sha256_file(base / "scores.pt") != banked["scores_sha256"]
        or sha256_file(base / "keeps.json") != banked["keeps_sha256"]
    ):
        raise ValueError("capture bank identity or bytes differ")
    return banked


def claim(root: Path, phase: str, model: str, qid: str) -> bool:
    """Claim one question for this phase, or return False if a seat holds it."""
    try:
        (root / "claims" / phase / model / qid).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return False
    return True


def complete(base: Path, phase: str) -> bool:
    """Return whether this phase has nothing left to do on one question."""
    if (base / "answers.json").exists():
        return True
    return phase == "capture" and all((base / name).exists() for name in _CAPTURE_FILES)


def work(root: Path, model: str, phase: str, *, seat: int) -> None:
    """Work through this model's queue, one claimed question at a time.

    Every seat of a model reads the same plan order and claims the next unclaimed
    question, so the seats balance themselves.
    """
    plan = load_plan(root)
    qids = [row["qid"] for row in plan["items"]]
    engine, tokenizer, runtime = open_engine(model, capture=phase == "capture")
    atomic_json(root / "runtime" / f"{model}-{phase}-{seat}.json", runtime)
    for qid in qids:
        if not claim(root, phase, model, qid):
            continue
        started = time.monotonic()
        item = read_json(root / "inputs" / f"{qid}.json")
        base = root / "models" / model / qid
        identity = {
            "plan": plan["fingerprint"],
            "model": model,
            "qid": qid,
            "prompt_sha256": item["prompt_sha256"],
        }
        answers = base / "answers.json"
        if answers.exists():
            validate_answers(base, identity)
            continue
        if phase == "capture":
            _capture(engine, tokenizer, base, identity, item, runtime, started)
        else:
            _decode(engine, tokenizer, base, identity, item, runtime)
        LOG.info(f"{phase} {model} {qid} complete")


def _capture(
    engine: Any,
    tokenizer: Any,
    base: Path,
    identity: dict[str, Any],
    item: dict[str, Any],
    runtime: dict[str, Any],
    started: float,
) -> None:
    if (base / "capture.json").exists():
        banked = _load_bank(base, identity)
        if (base / "embeddings.pt").exists():
            if sha256_file(base / "embeddings.pt") != banked["embeddings_sha256"]:
                raise ValueError("transient embeddings changed")
            return
        # A copied root omits the transient embeddings, so capture again.
        (base / "capture.json").unlink()
    ids = torch.tensor([item["prompt_token_ids"]], dtype=torch.long)
    prompt = manager_prompt(tokenizer, item["question"], enable_thinking=True)
    judger_ids = torch.tensor(
        [tokenizer(prompt, add_special_tokens=False)["input_ids"]],
        dtype=torch.long,
        device=engine.device,
    )
    product = engine.produce_worker(
        ids,
        judger_ids=judger_ids,
        judger_mask=torch.ones_like(judger_ids),
        question_ids=question_token_ids(tokenizer, prompt, item["question"]),
    )
    if not product.question_found or product.length != len(item["prompt_token_ids"]) + 40:
        raise RuntimeError("capture omitted question rows or changed rolled geometry")
    with torch.no_grad():
        judger = engine.model.get_input_embeddings()(judger_ids)[0].cpu()
    _tensor_file(
        base / "embeddings.pt",
        {
            "memory": product.embeds[0].detach().cpu(),
            "judger": judger,
        },
    )
    _tensor_file(base / "scores.pt", product.scores)
    atomic_json(base / "keeps.json", selections(product.scores))
    atomic_json(
        base / "capture.json",
        {
            "identity": identity,
            "runtime": canonical_sha(runtime),
            "embeddings_sha256": sha256_file(base / "embeddings.pt"),
            "scores_sha256": sha256_file(base / "scores.pt"),
            "keeps_sha256": sha256_file(base / "keeps.json"),
            "prompt_tokens": len(item["prompt_token_ids"]),
            "source_tokens": item["source_composition"]["original_evidence_tokens"],
            "latent_steps": 40,
            "padding_tokens": 0,
            "capture_seconds": time.monotonic() - started,
        },
    )
    del product, judger


def _decode(
    engine: Any,
    tokenizer: Any,
    base: Path,
    identity: dict[str, Any],
    item: dict[str, Any],
    runtime: dict[str, Any],
) -> None:
    vllm: Any = importlib.import_module("vllm")
    decode_started = time.monotonic()

    banked = _load_bank(base, identity)
    if sha256_file(base / "embeddings.pt") != banked["embeddings_sha256"]:
        raise ValueError("transient embedding bytes differ")
    tensors = torch.load(base / "embeddings.pt", map_location="cpu", weights_only=True)
    memory, judger = tensors["memory"], tensors["judger"]
    cells = read_json(base / "keeps.json")
    sampling = vllm.SamplingParams(
        temperature=0.0, seed=0, max_tokens=MAX_ANSWER_TOKENS, stop_token_ids=[151645, 151643]
    )
    rows: list[dict[str, Any]] = []
    # All cells of a question go in one submission, scheduled inside the KV budget, so
    # a short answer frees its seat instead of a fixed batch's longest. These draws are
    # greedy and not batch invariant either way.
    requests = [
        {"prompt_embeds": torch.cat((memory[cell["keep"]], judger), dim=0)} for cell in cells
    ]
    outputs = engine.generate(requests, sampling, use_tqdm=False)
    if len(outputs) != len(cells):
        raise RuntimeError("vLLM returned a different batch roster")
    for cell, output in zip(cells, outputs, strict=True):
        if len(output.outputs) != 1:
            raise RuntimeError("greedy dev requires exactly one answer per cell")
        answer = output.outputs[0]
        ids = list(answer.token_ids)
        if (
            answer.finish_reason not in {"stop", "length"}
            or len(ids) > MAX_ANSWER_TOKENS
            or (answer.finish_reason == "length" and len(ids) != MAX_ANSWER_TOKENS)
        ):
            raise RuntimeError("malformed generation termination")
        text = tokenizer.decode(ids, skip_special_tokens=True)
        rows.append(
            {
                "selector": cell["selector"],
                "ratio": cell["ratio"],
                "keep_sha256": canonical_sha(cell["keep"]),
                "transmitted_rows": len(cell["keep"]),
                "answer_token_ids": ids,
                "answer_tokens": len(ids),
                "answer_text": text,
                "finish_reason": answer.finish_reason,
                "scoring_version": DEFAULT_SCORER_VERSION,
                "accuracy": score_text(
                    Question(
                        qid=item["qid"], question=item["question"], pages=(), raw=item["gold"]
                    ),
                    QWEN_FAMILY.post_think_report(text),
                ),
            }
        )
    body: dict[str, Any] = {
        "identity": identity,
        "runtime": canonical_sha(runtime),
        "capture_sha256": sha256_file(base / "capture.json"),
        "decode_seconds": time.monotonic() - decode_started,
        "rows": rows,
    }
    atomic_json(base / "answers.json", {**body, "fingerprint": canonical_sha(body)})
    validate_answers(base, identity)
    (base / "embeddings.pt").unlink()


def validate_answers(base: Path, identity: dict[str, Any]) -> dict[str, Any]:
    """Require complete answers, bound to the capture and keeps they came from."""
    _load_bank(base, identity)
    answers = read_json(base / "answers.json")
    body: dict[str, Any] = {key: value for key, value in answers.items() if key != "fingerprint"}
    cells = read_json(base / "keeps.json")
    if (
        answers["identity"] != identity
        or answers["fingerprint"] != canonical_sha(body)
        or answers["capture_sha256"] != sha256_file(base / "capture.json")
        or len(answers["rows"]) != CELLS
        or len(cells) != CELLS
    ):
        raise ValueError("answer bank identity or completeness differs")
    for row, cell in zip(answers["rows"], cells, strict=True):
        if (
            row["selector"] != cell["selector"]
            or row["ratio"] != cell["ratio"]
            or row["keep_sha256"] != canonical_sha(cell["keep"])
            or row["transmitted_rows"] != len(cell["keep"])
        ):
            raise ValueError("answer selection digest differs")
    return answers
