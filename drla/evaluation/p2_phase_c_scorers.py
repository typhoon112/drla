"""Offline scorer helpers for P2 Phase C benchmark validation.

These helpers are intentionally small and model-free.  They score already
parsed final answers for Phase C protocol validation; they do not run agents,
inspect held-out generations, or depend on SwanLab.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any


_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


@dataclass(frozen=True)
class QAScore:
    exact_match: float
    alias_match: float
    token_f1: float
    primary_score: float
    normalized_prediction: str
    normalized_gold: str
    matched_alias: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StructuredScore:
    exact_match: float
    primary_score: float
    canonical_prediction: str
    canonical_gold: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_qa_text(text: Any) -> str:
    """Normalize text for QA exact match and token-F1 scoring."""

    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    tokens = [token for token in text.split() if token not in _ARTICLES]
    return " ".join(tokens)


def qa_token_f1(prediction: Any, gold: Any) -> float:
    pred_tokens = normalize_qa_text(prediction).split()
    gold_tokens = normalize_qa_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(overlap.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_qa_answer(prediction: Any, gold: Any, aliases: list[Any] | None = None) -> QAScore:
    normalized_prediction = normalize_qa_text(prediction)
    normalized_gold = normalize_qa_text(gold)
    exact = float(normalized_prediction == normalized_gold)
    matched_alias = None
    alias_match = 0.0
    for alias in aliases or []:
        normalized_alias = normalize_qa_text(alias)
        if normalized_alias and normalized_prediction == normalized_alias:
            alias_match = 1.0
            matched_alias = normalized_alias
            break
    primary = max(exact, alias_match)
    return QAScore(
        exact_match=exact,
        alias_match=alias_match,
        token_f1=qa_token_f1(prediction, gold),
        primary_score=primary,
        normalized_prediction=normalized_prediction,
        normalized_gold=normalized_gold,
        matched_alias=matched_alias,
    )


def score_structured_exact(
    prediction: Any,
    gold: Any,
    *,
    sort_lists: bool = False,
) -> StructuredScore:
    canonical_prediction = canonical_json(prediction, sort_lists=sort_lists)
    canonical_gold = canonical_json(gold, sort_lists=sort_lists)
    exact = float(canonical_prediction == canonical_gold)
    return StructuredScore(
        exact_match=exact,
        primary_score=exact,
        canonical_prediction=canonical_prediction,
        canonical_gold=canonical_gold,
    )


def canonical_json(value: Any, *, sort_lists: bool = False) -> str:
    return json.dumps(
        canonicalize(value, sort_lists=sort_lists),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonicalize(value: Any, *, sort_lists: bool = False) -> Any:
    if isinstance(value, dict):
        return {str(key): canonicalize(val, sort_lists=sort_lists) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        items = [canonicalize(item, sort_lists=sort_lists) for item in value]
        if sort_lists or isinstance(value, set):
            return sorted(items, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))
        return items
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value.strip())
    return value
