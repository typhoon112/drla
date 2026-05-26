"""Answer extraction and normalization for GSM8K-style math answers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction


_HASH_ANSWER_RE = re.compile(r"####\s*([^\n\r]+)")
_PHRASE_ANSWER_RE = re.compile(
    r"(?:answer\s+is|final\s+answer\s*(?:is|:)?|therefore\s*,?\s*)\s*([^\n\r.]+)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?%?"
)


@dataclass(frozen=True)
class JudgeResult:
    """Structured result returned by the math answer judge."""

    correct: bool
    pred_norm: str | None
    gold_norm: str
    answer_found: bool


def extract_answer_text(text: str) -> str | None:
    """Extract the most likely final answer span from generated text."""
    if not text:
        return None

    hash_match = _HASH_ANSWER_RE.search(text)
    if hash_match:
        return hash_match.group(1).strip()

    phrase_matches = list(_PHRASE_ANSWER_RE.finditer(text))
    if phrase_matches:
        return phrase_matches[-1].group(1).strip()

    number_matches = list(_NUMBER_RE.finditer(text))
    if number_matches:
        return number_matches[-1].group(0).strip()

    return None


def normalize_answer(value: str | int | float | Decimal | None) -> str | None:
    """Normalize integers, decimals, fractions, and percents for exact matching."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    text = text.replace("$", "")
    text = text.replace(",", "")
    text = text.strip()
    text = _strip_wrapping_text(text)
    if not text:
        return None

    percent = text.endswith("%")
    if percent:
        text = text[:-1].strip()

    try:
        if "/" in text:
            left, right = [part.strip() for part in text.split("/", 1)]
            number = Decimal(left) / Decimal(right)
        else:
            number = Decimal(text)
    except (InvalidOperation, ZeroDivisionError, ValueError):
        return _normalize_symbolic(text)

    if percent:
        number = number / Decimal("100")
    return _format_decimal(number)


def judge(pred_text: str, gold_answer: str | int | float | Decimal) -> dict[str, object]:
    """Judge a generated answer against a normalized gold answer."""
    pred_answer = extract_answer_text(pred_text)
    pred_norm = normalize_answer(pred_answer)
    gold_norm = normalize_answer(gold_answer)
    if gold_norm is None:
        raise ValueError(f"Gold answer could not be normalized: {gold_answer!r}")
    return JudgeResult(
        correct=pred_norm == gold_norm,
        pred_norm=pred_norm,
        gold_norm=gold_norm,
        answer_found=pred_norm is not None,
    ).__dict__


def _strip_wrapping_text(text: str) -> str:
    number_match = _NUMBER_RE.search(text)
    if number_match:
        return number_match.group(0).strip()
    return text.strip().strip(".。!！")


def _normalize_symbolic(text: str) -> str | None:
    normalized = text.lower().strip().strip(".。!！")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized or None


def _format_decimal(number: Decimal) -> str:
    if number == number.to_integral_value():
        return format(number.to_integral_value(), "f")
    normalized = number.normalize()
    text = format(normalized, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def decimal_to_fraction_string(value: str) -> str | None:
    """Return a compact fraction string for diagnostics."""
    normalized = normalize_answer(value)
    if normalized is None:
        return None
    return str(Fraction(Decimal(normalized)))

