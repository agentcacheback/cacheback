"""The official LongBench v2 answer extraction and the chain's aggregation.

Extraction is the official rule character for character; no answer, an unclosed
answer, and a wrong letter all score zero, and only the visible answer is read.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

CHOICES = ("A", "B", "C", "D")
CHANCE = 0.25
EXTRACTION_RULE = "longbench-v2-official-extraction-strip-star-paren-then-bare-v1"
_PAREN = re.compile(r"The correct answer is \(([A-D])\)")
_BARE = re.compile(r"The correct answer is ([A-D])")


def extract_choice(text: str) -> str | None:
    """Return the letter the official rule reads from one visible answer."""
    response = text.replace("*", "")
    match = _PAREN.search(response) or _BARE.search(response)
    return match.group(1) if match else None


def score_choice(gold: str, text: str) -> dict[str, float]:
    """Score one sample: exact letter match under the official extraction."""
    if gold not in CHOICES:
        raise ValueError(f"gold answer must be one of {CHOICES}, got {gold!r}")
    predicted = extract_choice(text)
    return {
        "correct": 1.0 if predicted == gold else 0.0,
        "answered": 1.0 if predicted is not None else 0.0,
    }


def item_mean(samples: Sequence[Mapping[str, float]], field: str = "correct") -> float:
    """Return the mean of one field over an item's samples."""
    if not samples:
        raise ValueError("an item needs at least one scored sample")
    return round(sum(float(sample[field]) for sample in samples) / len(samples), 4)


def macro_mean(item_scores: Mapping[str, float]) -> float:
    """Return the mean over items, each item weighted equally."""
    if not item_scores:
        raise ValueError("a macro mean needs at least one item")
    return round(sum(item_scores.values()) / len(item_scores), 4)


def stratum_means(item_scores: Mapping[str, float], strata: Mapping[str, int]) -> dict[int, float]:
    """Return the mean inside each stratum, keyed by band."""
    by_band: dict[int, list[float]] = {}
    for qid, score in item_scores.items():
        by_band.setdefault(strata[qid], []).append(score)
    return {band: round(sum(values) / len(values), 4) for band, values in sorted(by_band.items())}


def majority_floor(golds: Iterable[str]) -> float:
    """Return the score of always guessing the most common gold letter."""
    counts = Counter(golds)
    if not counts:
        raise ValueError("the majority floor needs at least one gold answer")
    return round(max(counts.values()) / sum(counts.values()), 4)


__all__ = (
    "CHANCE",
    "CHOICES",
    "EXTRACTION_RULE",
    "extract_choice",
    "item_mean",
    "macro_mean",
    "majority_floor",
    "score_choice",
    "stratum_means",
)
