"""Answer extraction, SQuAD-style F1 and exact match, and the QA decode prompt.

Use the chat template with its explicit "Answer:" directive; a raw completion
prompt scores far worse.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Iterable
from typing import Any

_ANSWER_PATTERN = re.compile(r"\banswer\s*[:\-]\s*(.+?)(?:\n|$)", re.IGNORECASE)
_TRAILING_PUNCT = ".,;:!?\"')]}*`>"
_LEADING_PUNCT = "*`<\"'([{"
_ARTICLES_RE = re.compile(r"\b(?:a|an|the)\b", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def extract_answer(generation: str | None) -> str | None:
    """Pull the answer span out of a raw model generation, or None if extraction fails.

    The answer is the first non-empty line, since `qa_prompt` primes the reply with
    `Answer: `. A later `answer:` marker would let a hallucinated turn overwrite it.
    """
    if not generation:
        return None
    text = generation.strip()
    if not text:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    matches = list(_ANSWER_PATTERN.finditer(lines[0]))
    if matches:
        span = matches[-1].group(1).strip()
        return _strip_wrappers(span) or None
    return _strip_wrappers(lines[0]) or None


def _strip_wrappers(s: str) -> str:
    """Strip leading wrapper chars and trailing punctuation from a span, to a fixed point."""
    prev = None
    while s != prev:
        prev = s
        s = s.strip().lstrip(_LEADING_PUNCT).rstrip(_TRAILING_PUNCT)
    return s


def normalize_answer(s: str | None) -> str:
    """Normalize SQuAD style: lowercase, strip articles and punctuation, collapse whitespace."""
    if s is None:
        return ""
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def _tokenize(s: str) -> list[str]:
    """Whitespace-tokenize an already-normalized string."""
    return s.split() if s else []


def token_f1(prediction: str, gold: str) -> float:
    """Return the SQuAD token-level F1 between two already-normalized strings."""
    pred_tokens = _tokenize(prediction)
    gold_tokens = _tokenize(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str) -> float:
    """Return 1.0 if two already-normalized strings are equal, else 0.0."""
    return float(prediction == gold)


def score_prediction(
    prediction: str | None, gold: str, aliases: Iterable[str] | None = None
) -> tuple[float, float]:
    """Score one prediction against a gold answer and optional aliases, as (best_f1, best_em)."""
    pred_norm = normalize_answer(prediction)
    candidates = [normalize_answer(gold)] + [normalize_answer(a) for a in (aliases or [])]
    candidates = [c for c in candidates if c]
    if not candidates:
        return (1.0, 1.0) if not pred_norm else (0.0, 0.0)
    best_f1 = max(token_f1(pred_norm, c) for c in candidates)
    best_em = max(exact_match(pred_norm, c) for c in candidates)
    return best_f1, best_em


def qa_prompt(tokenizer: Any, question: str) -> str:
    """Chat-template the question with enable_thinking False and an Answer: directive."""
    q = question or ""
    content = (
        f"Question: {q}\n\nAnswer the question using only the information in the context. "
        "Reply with just the answer, prefixed by 'Answer:'. Keep it to a few words."
    )
    msg = [{"role": "user", "content": content}]
    for kw in ({"enable_thinking": False}, {}):
        try:
            return (
                tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, **kw)
                + "Answer: "
            )
        except Exception:
            continue
    return f"Question: {q}\nAnswer:"
