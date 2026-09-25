"""Score a FanOutQA answer under the representation rules of equivalence-v1.

The rules canonicalize how a value is written, such as an integer decimal or a
squared metric unit, without rounding it, so two spellings of one answer match.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from rcc.benchmarks.fanoutqa.scoring_core import (
    LeafGroups,
    ScoredQuestion,
    gold_leaf_groups,
    gold_leaves,
    score_groups,
)

SCORER_VERSION = "fanoutqa-equivalence-v1"
_ALIASES_PATH = Path(__file__).with_name("equivalence_aliases.json")
_INTEGER_DECIMAL = re.compile(r"(?<![\w.])(\d+)\.0+(?!\w|\.\w)")
_SQUARE_UNIT = re.compile(r"(?<![^\W\d])(mm|cm|km|m)²(?!\w)")


def normalize_equivalent_notation(text: str) -> str:
    """Canonicalize integer decimals and squared metric units, without rounding."""
    text = _INTEGER_DECIMAL.sub(r"\1", text.lower())
    return _SQUARE_UNIT.sub(r"\g<1>2", text)


@dataclass(frozen=True)
class _Alias:
    index: int
    original: str
    alternatives: tuple[str, ...]


def _read_alias(entry: dict[str, Any]) -> tuple[str, _Alias]:
    qid, index = entry.get("qid"), entry.get("leaf_index")
    original, alternatives = entry.get("original"), entry.get("alternatives")
    if not isinstance(qid, str) or not qid.strip() or type(index) is not int or index < 0:
        raise ValueError("malformed equivalence alias reference")
    if not isinstance(original, str) or not original.strip():
        raise ValueError("empty equivalence alias reference")
    if not isinstance(alternatives, list) or not alternatives:
        raise ValueError("equivalence alternatives must be a nonempty list")
    values = cast(list[object], alternatives)
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError("equivalence alternatives must be nonblank strings")
    return qid, _Alias(index, original, tuple(cast(list[str], values)))


@lru_cache(maxsize=1)
def _aliases() -> dict[str, tuple[_Alias, ...]]:
    payload = cast(dict[str, Any], json.loads(_ALIASES_PATH.read_text()))
    if payload.get("schema") != "fanoutqa-equivalence-aliases-v1":
        raise ValueError("unknown equivalence alias schema")
    if payload.get("policy") != "complete-two-person-shared-surname-v1":
        raise ValueError("unknown equivalence alias policy")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("equivalence entries must be a list of records")
    records = cast(list[object], entries)
    if not all(isinstance(entry, dict) for entry in records):
        raise ValueError("equivalence entries must be a list of records")
    result: dict[str, tuple[_Alias, ...]] = {}
    for entry in records:
        qid, new_alias = _read_alias(cast(dict[str, Any], entry))
        if any(alias.index == new_alias.index for alias in result.get(qid, ())):
            raise ValueError("duplicate equivalence alias reference")
        result[qid] = (*result.get(qid, ()), new_alias)
    return result


def equivalent_groups(question: ScoredQuestion) -> LeafGroups:
    """Return the patched gold groups with the audited aliases and notation applied."""
    groups = list(gold_leaf_groups(question))
    leaves = gold_leaves(question)
    for alias in _aliases().get(str(question.raw.get("id")), ()):
        if alias.index >= len(leaves) or leaves[alias.index] != alias.original:
            raise ValueError("equivalence alias reference differs from the question")
        original = (alias.original,)
        if groups.count(original) != 1:
            raise ValueError("equivalence alias reference conflicts with corrected gold")
        groups[groups.index(original)] = (alias.original, *alias.alternatives)
    return tuple(tuple(normalize_equivalent_notation(leaf) for leaf in group) for group in groups)


def score_text_equivalent(question: ScoredQuestion, text: str) -> dict[str, float]:
    """Score one answer against the patched gold under the equivalence-v1 rules."""
    return score_groups(equivalent_groups(question), normalize_equivalent_notation(text))
