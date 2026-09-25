"""The comparison's fixed identities, and the import of its inputs.

``SCIENCE`` pins what the run holds constant, ``SELECTORS`` and ``RATIOS`` the
grid it walks. Importing the inputs writes a plan the later phases are held to.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from rcc.benchmarks.fanoutqa.panel import load_freeze, load_questions
from rcc.run.io import atomic_json, canonical_sha, sha256_file

MODELS = {
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    "qwen3-4b": ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
    "qwen3-8b": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
    "qwen3-32b": ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137"),
}
# The roster: mean query attention, the query-support (p, alpha) grid over five finite orders
# and the p=inf endpoint, and four comparators. Every point is scored from one
# capture, and nothing is decoded unless the decode phase is asked for.
SUPPORT_GRID = tuple(
    f"support-p{order}-a{alpha}"
    for order in ("1p5", "2", "3", "4", "8")
    for alpha in ("0p25", "0p5", "1", "2", "4")
) + tuple(f"support-pinf-a{alpha}" for alpha in ("1", "2", "4", "8"))
SUPPORT_ORDERS = (1.5, 2.0, 3.0, 4.0, 8.0)
SELECTORS = ("snap", *SUPPORT_GRID, "chunkkv", "h2o", "kvzip", "streaming")
RATIOS = (2, 4, 8, 16, 32, 64, 128)
#: Answer cells per model and question: one uncompressed baseline plus every
#: selector at every ratio.
CELLS = 1 + len(SELECTORS) * len(RATIOS)
LATENT_STEPS = 40
MAX_ANSWER_TOKENS = 4096
#: Digest of the released capture bundle, `selector-dev-v8.tar.zst`.
BUNDLE_SHA256 = "9c2e240877b3b1f9d8d86d2fdc3579467b5c504c7a859439e717064d19010eb4"
SCIENCE = {
    "schema": "unpadded-selector-dev-v1",
    "models": MODELS,
    "selectors": SELECTORS,
    "ratios": RATIOS,
    "latent_steps": LATENT_STEPS,
    "span_width": 16,
    "snap_support_sink_score": "production-position-zero-max",
    "workers_per_question": 1,
    "prompt_tokens_max": 50_000,
    "padding": False,
    "temperature": 0.0,
    "seed": 0,
    "samples": 1,
    "answer_tokens_max": MAX_ANSWER_TOKENS,
    "stop_token_ids": (151645, 151643),
    "scoring": "post-think-closed-string-proxy",
    "producer": "vllm-prefill-hf-alias-roll-support-grid-scores",
    "receiver": "vllm-prompt-embeds",
    "timings": "budgeting-only-not-latency-results",
}


def read_json(path: Path) -> Any:
    """Read one banked JSON artifact."""
    return json.loads(path.read_text())


def token_hash(ids: list[int]) -> str:
    """Hash one token id list the way the input bank does, comma-separated."""
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def import_inputs(root: Path, prepared: Path, index: Path, *, commit: str) -> dict[str, Any]:
    """Import the audited inputs into an immutable plan, refusing any drift."""
    from transformers import AutoTokenizer

    if (root / "plan.json").exists():
        raise ValueError("use a fresh run directory; an existing plan is immutable")
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("a full source commit is required")
    qids = list(load_freeze()["dev_ids"])
    questions = {q.qid: q for q in load_questions(index)}
    manifest = read_json(prepared / "manifest.json")
    if manifest["qids"] != qids or len(manifest["items"]) != 30:
        raise ValueError("prepared input is not the frozen dev30 roster")
    tokenizers: list[Any] = [
        cast(Any, AutoTokenizer).from_pretrained(repo, revision=revision, local_files_only=True)
        for repo, revision in MODELS.values()
    ]
    rows: list[dict[str, Any]] = []
    for qid, entry in zip(qids, manifest["items"], strict=True):
        path = prepared / "items" / f"{qid}.json"
        if entry["qid"] != qid or sha256_file(path) != entry["sha256"]:
            raise ValueError("prepared item bytes differ from their audit manifest")
        item = read_json(path)
        ids = item["prompt_token_ids"]
        composition = item["source_composition"]
        if (
            item["qid"] != qid
            or item["question"] != questions[qid].question
            or not 0 < len(ids) <= 50_000
            or any(type(token) is not int or token < 0 for token in ids)
            or composition.get("padding_source") != "none"
            or composition.get("padding_payload_tokens") != 0
            or composition.get("padding_correction_tokens") != 0
            or composition["final_prompt_tokens"] != len(ids)
            or item["prompt_sha256"] != token_hash(ids)
        ):
            raise ValueError(f"{qid}: input is not an audited unpadded prompt")
        for tokenizer in tokenizers:
            if tokenizer(item["prompt_text"], add_special_tokens=False)["input_ids"] != ids:
                raise ValueError(f"{qid}: model tokenizer changes the prompt")
        item["gold"] = questions[qid].raw
        output = root / "inputs" / f"{qid}.json"
        atomic_json(output, item)
        rows.append({"qid": qid, "sha256": sha256_file(output)})
    plan: dict[str, Any] = {
        "run_id": root.name,
        "science": SCIENCE,
        "source_commit": commit,
        "items": rows,
        "prepared_manifest_sha256": sha256_file(prepared / "manifest.json"),
        "question_index_sha256": sha256_file(index),
        "answer_cells": len(MODELS) * len(qids) * CELLS,
    }
    plan["fingerprint"] = canonical_sha(plan)
    atomic_json(root / "plan.json", plan)
    return plan


def load_plan(root: Path) -> dict[str, Any]:
    """Rehash the plan and every input before any work or report."""
    plan = read_json(root / "plan.json")
    body = {key: value for key, value in plan.items() if key != "fingerprint"}
    if (
        canonical_sha(body) != plan["fingerprint"]
        or canonical_sha(plan["science"]) != canonical_sha(SCIENCE)
        or [row["qid"] for row in plan["items"]] != list(load_freeze()["dev_ids"])
    ):
        raise ValueError("dev plan identity differs")
    for row in plan["items"]:
        if sha256_file(root / "inputs" / f"{row['qid']}.json") != row["sha256"]:
            raise ValueError("dev input bytes changed")
    return plan
