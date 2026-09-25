"""FanOutQA scoring under one policy, ``fanoutqa-equivalence-v1``.

A row or bank naming any other version raises, so one bank never mixes two
metrics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from rcc.benchmarks.fanoutqa.scoring_core import (
    LeafGroups,
    ScoredQuestion,
    evidence_answerability_audit,
    gold_leaf_groups,
    gold_leaves,
    load_gold_patch,
    score_groups,
    validate_gold_patch,
)
from rcc.benchmarks.fanoutqa.scoring_equivalence import SCORER_VERSION, score_text_equivalent

DEFAULT_SCORER_VERSION = SCORER_VERSION
Scorer = Callable[[ScoredQuestion, str], dict[str, float]]


def score_text(question: ScoredQuestion, text: str) -> dict[str, float]:
    """Score one answer under ``fanoutqa-equivalence-v1``."""
    return score_text_equivalent(question, text)


def score_text_original(question: ScoredQuestion, text: str) -> dict[str, float]:
    """Score one answer against the dataset's references as shipped, one leaf per group."""
    return score_groups(tuple((leaf,) for leaf in gold_leaves(question)), text)


#: The reference sets an answer bank can be scored under, by name.
REFERENCE_SCORERS: dict[str, Scorer] = {
    DEFAULT_SCORER_VERSION: score_text,
    "original": score_text_original,
}


def scorer_for_version(version: str) -> Scorer:
    """Raise for any scoring version other than ``fanoutqa-equivalence-v1``."""
    if version != DEFAULT_SCORER_VERSION:
        raise ValueError(f"unknown FanOutQA scoring version: {version!r}")
    return score_text


def scoring_version_for_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    """Raise unless every row names ``fanoutqa-equivalence-v1``."""
    versions = [row.get("scoring_version") for row in rows]
    if not versions or any(not isinstance(version, str) for version in versions):
        raise ValueError("missing rows or malformed FanOutQA scoring version")
    if len(set(versions)) != 1:
        raise ValueError("mixed FanOutQA scoring versions in one result bank")
    version = str(versions[0])
    scorer_for_version(version)
    return version


def scorer_for_rows(rows: Sequence[Mapping[str, Any]]) -> Scorer:
    """Return the scorer, after checking the bank's scoring version."""
    return scorer_for_version(scoring_version_for_rows(rows))


__all__ = (
    "DEFAULT_SCORER_VERSION",
    "REFERENCE_SCORERS",
    "LeafGroups",
    "ScoredQuestion",
    "Scorer",
    "evidence_answerability_audit",
    "gold_leaf_groups",
    "gold_leaves",
    "load_gold_patch",
    "score_groups",
    "score_text",
    "score_text_original",
    "scorer_for_rows",
    "scorer_for_version",
    "scoring_version_for_rows",
    "validate_gold_patch",
)
