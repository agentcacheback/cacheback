"""A one-question natural FanOutQA bundle written under a temporary directory.

The stand-in profile carries digests computed from exactly the bytes written, so
the validator runs the same checks it runs on the shipped bundle. The offline
handoff-recall lanes read one such bundle per family, so the run root each of
them stages around it is built here too.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.fanoutqa.natural_panel import (
    NATURAL_BUNDLE_SCHEMA,
    NATURAL_PANEL_SCHEMA,
)

_DEV_INDEX = [
    {
        "id": "q1",
        "question": "Which two rivers?",
        "answer": ["Alpha", "Beta"],
        "categories": [],
        "decomposition": [
            {
                "question": "first",
                "evidence": {"pageid": 11, "revid": 101, "title": "Alpha", "url": "u"},
                "decomposition": [],
            },
            {
                "question": "second",
                "evidence": {"pageid": 12, "revid": 102, "title": "Beta", "url": "u"},
                "decomposition": [],
            },
            {
                "question": "third",
                "evidence": {"pageid": 13, "revid": 103, "title": "Gamma", "url": "u"},
                "decomposition": [],
            },
        ],
    }
]


class CharTokenizer:
    """One token per character: geometry checks need lengths, not a vocabulary."""

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [ord(character) for character in text]}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_bundle(root: Path) -> Any:
    """Create a one-item natural bundle and the stand-in profile that pins it."""
    pages = {11: "Alpha river text. " * 4, 12: "Beta river text. " * 4, 13: "Gamma text. " * 4}
    texts = (pages[11].strip(), pages[12].strip(), pages[13].strip())
    (root / "canonical").mkdir(parents=True)
    for pageid, text in pages.items():
        (root / "canonical" / f"{pageid}.md").write_text(text, encoding="utf-8")
    (root / "source_cache").mkdir()
    index = root / "source_cache" / "fanout-final-dev.json"
    index.write_text(json.dumps(_DEV_INDEX), encoding="utf-8")
    (root / "items").mkdir()
    item = {
        "qid": "q1",
        "rank": 0,
        "role": "selected",
        "question": "Which two rivers?",
        "pages": [
            {
                "pageid": pageid,
                "revid": 100 + pageid - 10,
                "title": "t",
                "raw_sha256": "a" * 64,
                "canonical_sha256": _sha(text),
                "qwen_tokens": len(text),
            }
            for pageid, text in pages.items()
        ],
        "workers": [
            {
                "worker": index,
                "pages": [{"pageid": 11 + index}],
                "text": text,
                "text_sha256": _sha(text),
            }
            for index, text in enumerate(texts)
        ],
    }
    (root / "items" / "q1.json").write_text(json.dumps(item), encoding="utf-8")
    panel = {
        "schema": NATURAL_PANEL_SCHEMA,
        "policy": {"padding": "none", "workers_per_item": 3},
        "items": [{"qid": "q1", "rank": 0, "role": "selected"}],
    }
    (root / "panel.json").write_text(json.dumps(panel), encoding="utf-8")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "bytes": path.stat().st_size,
                    "path": str(path.relative_to(root)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    manifest: dict[str, Any] = {
        "version": NATURAL_BUNDLE_SCHEMA,
        "question_index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        "files": files,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dataclasses.replace(
        FANOUTQA_NATURAL_DEV50,
        question_ids=("q1",),
        source_logical_fingerprint=hashlib.sha256((root / "panel.json").read_bytes()).hexdigest(),
        source_manifest_sha256=manifest["manifest_sha256"],
        question_index_sha256=manifest["question_index_sha256"],
        worker_prompt_tokens=200,
    )


def _one_item_config(tmp_path: Path, config_name: str, *swaps: tuple[str, str]) -> Path:
    """Copy one shipped config under ``tmp_path``, narrowed to its first item.

    Extra ``swaps`` are applied first, for a lane whose sealed roster does not
    survive the narrowing.
    """
    text = (Path("configs") / config_name).read_text(encoding="utf-8")
    for old, new in swaps:
        text = text.replace(old, new)
    path = tmp_path / config_name
    path.write_text(
        text.replace("count = 50", "count = 1").replace('root = "runs"', f'root = "{tmp_path}"'),
        encoding="utf-8",
    )
    return path


def bank_result_rows(
    root: Path,
    arm: str,
    *,
    cells: Sequence[Mapping[str, Any]],
    written: Sequence[str],
    prepared_sha256: str,
    **handoff: Any,
) -> Path:
    """Bank one arm's seed rows the way its family's result builder writes them.

    The identity fields mirror each split adapter's ``bank_identity``;
    ``written`` names the handoff fields that family emits, each banked empty
    unless the caller hands over the digest its channel carries.
    """
    path = root / "arms" / arm / "workers" / "gpu0" / "raw.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "kind": "result",
            "qid": "q1",
            "arm": arm,
            "semantic_arm": arm,
            "panel": "production",
            "seed_index": index,
            "prepared_sha256": prepared_sha256,
            **dict.fromkeys(written),
            **cell,
            **handoff,
        }
        for index, cell in enumerate(cells)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def rebanker(path: Path) -> Callable[..., None]:
    """Return a writer that reseats one bank with named fields retouched.

    Every call starts from the rows as banked, so one injected field never
    carries into the next.
    """
    banked = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def rebank(**fields: Any) -> None:
        rows = [{**row, **fields} for row in banked]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    return rebank
