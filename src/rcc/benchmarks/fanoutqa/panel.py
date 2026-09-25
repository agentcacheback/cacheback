"""The FanOutQA census freeze and the questions it admits.

The freeze fixes the eligible items and the panel identities; the loader parses
the official dev bytes into questions with their cited pages.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_FREEZE_PATH = Path(__file__).parent / "fanoutqa_freeze.json"
FANOUTQA_DATASET_REVISION = "48da441031b16f83b2c71d478c45150c7dece613"
FANOUTQA_DEV_URL = (
    "https://raw.githubusercontent.com/zhudotexe/fanoutqa/"
    f"{FANOUTQA_DATASET_REVISION}/fanoutqa/data/fanout-final-dev.json"
)


def load_freeze() -> dict[str, Any]:
    """Return the census freeze and the panel identities it fixes."""
    with _FREEZE_PATH.open(encoding="utf-8") as handle:
        freeze: dict[str, Any] = json.load(handle)
    return freeze


@dataclass(frozen=True)
class Question:
    """One question: its visible text, its cited pages, and its raw record."""

    qid: str
    question: str
    pages: tuple[tuple[int, int, str], ...]
    raw: dict[str, Any]


def _walk_evidence(decomposition: Any) -> Any:
    """Yield the cited evidence recursively, nested fan-out branches included."""
    for sub in decomposition or ():
        evidence = sub.get("evidence")
        if evidence:
            yield evidence
        yield from _walk_evidence(sub.get("decomposition"))


def load_questions(
    dev_json_path: str | Path,
    *,
    verify_sha: bool = True,
    freeze_loader: Callable[[], Mapping[str, Any]] | None = None,
) -> list[Question]:
    """Parse the official dev bytes into questions and their cited pages."""
    raw_bytes = Path(dev_json_path).read_bytes()
    if verify_sha:
        observed = hashlib.sha256(raw_bytes).hexdigest()
        expected = (freeze_loader or load_freeze)()["dev_json_sha256"]
        if observed != expected:
            raise ValueError(
                f"dev json sha {observed[:16]} != frozen {expected[:16]}; refusing an "
                "unfrozen dataset (eligibility is pinned to these bytes)"
            )
    output: list[Question] = []
    for record in json.loads(raw_bytes):
        seen: dict[int, tuple[int, int, str]] = {}
        for evidence in _walk_evidence(record.get("decomposition")):
            pageid = int(evidence["pageid"])
            if pageid not in seen:
                seen[pageid] = (
                    pageid,
                    int(evidence["revid"]),
                    str(evidence["title"]),
                )
        output.append(
            Question(
                qid=str(record["id"]),
                question=str(record["question"]),
                pages=tuple(seen.values()),
                raw=record,
            )
        )
    return output


__all__ = (
    "FANOUTQA_DATASET_REVISION",
    "FANOUTQA_DEV_URL",
    "Question",
    "load_freeze",
    "load_questions",
)
