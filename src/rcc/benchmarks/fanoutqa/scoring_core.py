"""The FanOutQA gold leaves, the string matching over them, and the reference patch.

A gold answer flattens to the official reference chain, every comparison runs under
one simplified normalization, and the patch corrects page-contradicted leaves.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, cast

_PATCH_PATH = Path(__file__).parent / "gold_patch.json"
GOLD_PATCH_SCHEMA = "fanoutqa-gold-patch-v1"
#: One patched reference: the dataset string it replaces and the groups that
#: stand in for it. Each group matches through at least one of its alternatives.
LeafGroups = tuple[tuple[str, ...], ...]


class ScoredQuestion(Protocol):
    """The question fields the scorers read."""

    @property
    def question(self) -> str:
        """Return the visible question text."""
        ...

    @property
    def raw(self) -> Mapping[str, Any]:
        """Return the raw source record containing the gold answer."""
        ...


def gold_leaves(question: ScoredQuestion) -> tuple[str, ...]:
    """Flatten one gold answer into the official reference strings."""

    def flat(value: Any) -> list[str]:
        if isinstance(value, bool):
            return ["yes" if value else "no"]
        if isinstance(value, dict):
            mapping = cast(dict[Any, Any], value)
            keys = [str(key) for key in mapping]
            return keys + [item for child in mapping.values() for item in flat(child)]
        if isinstance(value, (list, tuple)):
            sequence = cast(list[Any] | tuple[Any, ...], value)
            return [item for child in sequence for item in flat(child)]
        return [str(value)]

    return tuple(flat(question.raw.get("answer")))


def _norm(text: str) -> str:
    """Normalize one string: the simplified form every comparison here uses."""
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9. ]", " ", text)).strip()


def _leaf_forms(leaf: str) -> tuple[str, ...]:
    """Return normalized string and numeric forms for one gold leaf."""
    forms = {_norm(leaf)}
    cleaned = leaf.replace(",", "")
    try:
        number = float(cleaned)
    except ValueError:
        pass
    else:
        forms.add(_norm(cleaned))
        if number == int(number):
            forms.add(_norm(str(int(number))))
    return tuple(form for form in forms if form)


def evidence_answerability_audit(
    question: ScoredQuestion,
    *,
    full_sources: Sequence[str],
    retained_sources: Sequence[str],
) -> dict[str, Any]:
    """Count how many gold leaves a constructed item still contains."""

    def contains(haystack: str, leaf: str) -> bool:
        return any(f" {form} " in haystack for form in _leaf_forms(leaf))

    full_text = "\n".join(str(source) for source in full_sources)
    retained_text = "\n".join(str(source) for source in retained_sources)
    question_hay = f" {_norm(question.question)} "
    full_hay = f" {_norm(full_text)} "
    retained_hay = f" {_norm(retained_text)} "
    rows: list[dict[str, Any]] = []
    for leaf in gold_leaves(question):
        in_question = contains(question_hay, leaf)
        in_full = contains(full_hay, leaf)
        in_retained = contains(retained_hay, leaf)
        if in_question:
            status = "question_given"
        elif not in_full:
            status = "absent_from_full_source"
        elif in_retained:
            status = "survives_construction"
        else:
            status = "removed_by_construction"
        rows.append(
            {
                "leaf": leaf,
                "in_question": in_question,
                "in_full_source": in_full,
                "in_retained_source": in_retained,
                "status": status,
            }
        )
    off_question = [row for row in rows if not row["in_question"]]
    return {
        "n_gold_leaves": len(rows),
        "n_question_given": len(rows) - len(off_question),
        "n_off_question_leaves": len(off_question),
        "n_locatable_in_full_source": sum(bool(row["in_full_source"]) for row in off_question),
        "n_survives_construction": sum(
            row["status"] == "survives_construction" for row in off_question
        ),
        "n_removed_by_construction": sum(
            row["status"] == "removed_by_construction" for row in off_question
        ),
        "n_absent_from_full_source": sum(
            row["status"] == "absent_from_full_source" for row in off_question
        ),
        "leaves": rows,
    }


def group_hits(groups: LeafGroups, text: str) -> tuple[bool, ...]:
    """Return, per group, whether any alternative's form appears in the text."""
    haystack = f" {_norm(text)} "
    return tuple(
        any(f" {form} " in haystack for leaf in group for form in _leaf_forms(leaf))
        for group in groups
    )


def score_groups(groups: LeafGroups, text: str) -> dict[str, float]:
    """Score one answer against leaf groups: any alternative satisfies a group."""
    if not groups:
        return {"loose": 0.0, "strict": 0.0, "n_leaves": 0.0}
    hits = sum(group_hits(groups, text))
    return {
        "loose": hits / len(groups),
        "strict": 1.0 if hits == len(groups) else 0.0,
        "n_leaves": float(len(groups)),
    }


@lru_cache(maxsize=1)
def load_gold_patch() -> dict[str, dict[int, tuple[str, LeafGroups]]]:
    """Return the pinned-page reference patch as {qid: {leaf index: (original, groups)}}."""
    payload = cast(dict[str, Any], json.loads(_PATCH_PATH.read_text(encoding="utf-8")))
    if payload.get("schema") != GOLD_PATCH_SCHEMA:
        raise RuntimeError("gold patch schema is not the registered one")
    patch: dict[str, dict[int, tuple[str, LeafGroups]]] = {}
    for raw_entry in cast(list[dict[str, Any]], payload["entries"]):
        qid = str(raw_entry["qid"])
        index = int(raw_entry["leaf_index"])
        groups = tuple(
            tuple(str(leaf) for leaf in cast(list[Any], group))
            for group in cast(list[Any], raw_entry["groups"])
        )
        if any(not group or any(not leaf for leaf in group) for group in groups):
            raise RuntimeError(f"gold patch {qid}[{index}] has an empty group or alternative")
        if bool(groups) == (raw_entry.get("kind") == "drop"):
            raise RuntimeError(f"gold patch {qid}[{index}]: only a drop entry may have no groups")
        if index in patch.setdefault(qid, {}):
            raise RuntimeError(f"gold patch {qid}[{index}] is listed twice")
        patch[qid][index] = (str(raw_entry["original"]), groups)
    return patch


def gold_leaf_groups(question: ScoredQuestion) -> LeafGroups:
    """Flatten the gold answer with the pinned-page patch applied by question id."""
    patch = load_gold_patch().get(str(question.raw.get("id")), {})
    groups: list[tuple[str, ...]] = []
    for index, leaf in enumerate(gold_leaves(question)):
        entry = patch.get(index)
        if entry is None:
            groups.append((leaf,))
            continue
        original, replacement = entry
        if original != leaf:
            raise RuntimeError(f"gold patch names {original!r} where the dataset has {leaf!r}")
        groups.extend(replacement)
    return tuple(groups)


def validate_gold_patch(questions: Sequence[ScoredQuestion]) -> None:
    """Raise for a patch entry naming an absent question or a leaf it does not have."""
    by_id = {str(question.raw.get("id")): question for question in questions}
    for qid, entries in load_gold_patch().items():
        question = by_id.get(qid)
        if question is None:
            raise RuntimeError(f"gold patch names {qid}, which is not on this panel")
        leaves = gold_leaves(question)
        for index, (original, _groups) in entries.items():
            if index >= len(leaves) or leaves[index] != original:
                raise RuntimeError(
                    f"gold patch {qid}[{index}] does not name a leaf of that question"
                )


__all__ = (
    "GOLD_PATCH_SCHEMA",
    "LeafGroups",
    "ScoredQuestion",
    "evidence_answerability_audit",
    "gold_leaf_groups",
    "gold_leaves",
    "load_gold_patch",
    "score_groups",
    "validate_gold_patch",
)
