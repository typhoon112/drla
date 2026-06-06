"""Build candidate answer sets for nonheldout Phase A answer selection.

This script is local-only. It extracts candidate answer strings from evidence
text already present in a Phase C manifest, audits whether the offline gold
answer is covered by those evidence-derived candidates, and writes candidate
sets for later answer-selection experiments.

It never injects the gold answer as a candidate. Gold/aliases are used only for
offline labels and coverage metrics.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST_JSON = (
    "/data1/luyifei/drla/outputs/p2_phase_c_manifests/"
    "musique_interface_train_manifest_10000_seed20260606/manifest.json"
)
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p2_phase_a_candidate_answers/"
    "musique_train_candidate_answers_10000_seed20260606_20260606"
)

CONNECTOR_WORDS = {
    "and",
    "of",
    "the",
    "in",
    "for",
    "to",
    "de",
    "del",
    "la",
    "le",
    "van",
    "von",
    "da",
    "di",
    "du",
    "al",
    "bin",
}
ORDINAL_WORDS = {
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
}
NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
}
MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)


def main() -> None:
    summary = build_candidate_sets(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples.")
    parser.add_argument("--max-candidates", type=int, default=96)
    parser.add_argument("--topk-report", default="16,32,64,96")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_candidate_sets(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
    samples = list(manifest.get("samples", []))
    if args.max_samples:
        samples = samples[: args.max_samples]
    topks = sorted({int(item) for item in args.topk_report.split(",") if item.strip()})

    rows = []
    metrics = []
    rule_counts = Counter()
    for sample in samples:
        row = build_sample_candidates(sample, max_candidates=args.max_candidates)
        rows.append(row)
        rule_counts.update(candidate["rule"] for candidate in row["candidates"])
        metrics.append(row["audit"])

    candidates_path = output_dir / "candidates.jsonl"
    write_jsonl(candidates_path, rows)
    summary = summarize(args, rows, metrics, rule_counts, topks)
    summary_path = output_dir / "summary.json"
    metrics_path = output_dir / "metrics.jsonl"
    summary["summary_json"] = str(summary_path)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(summary["metrics"], ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["metrics_jsonl"] = str(metrics_path)
    return summary


def build_sample_candidates(sample: dict[str, Any], *, max_candidates: int) -> dict[str, Any]:
    full_evidence = str(sample.get("metadata", {}).get("full_info_observation", ""))
    evidence_items = parse_evidence_items(full_evidence)
    gold = str(sample.get("scoring", {}).get("gold_answer", "")).strip()
    aliases = [str(alias).strip() for alias in sample.get("scoring", {}).get("answer_aliases", []) or []]
    gold_forms = [gold, *aliases]
    gold_norms = {normalize_qa(text) for text in gold_forms if normalize_qa(text)}

    candidates_by_norm: dict[str, dict[str, Any]] = {}
    for item_index, item in enumerate(evidence_items):
        for candidate in extract_candidates_from_item(item, item_index):
            norm = normalize_qa(candidate["text"])
            if not norm or len(norm) < 2:
                continue
            if norm in candidates_by_norm:
                candidates_by_norm[norm]["occurrences"] += 1
                if item["kind"] == "support":
                    candidates_by_norm[norm]["has_support_occurrence"] = True
                continue
            candidate["candidate_id"] = f"cand_{len(candidates_by_norm):03d}"
            candidate["normalized"] = norm
            candidate["occurrences"] = 1
            candidate["is_gold_or_alias"] = norm in gold_norms
            candidate["has_support_occurrence"] = item["kind"] == "support"
            candidates_by_norm[norm] = candidate

    candidates = sorted(candidates_by_norm.values(), key=candidate_sort_key)
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank
    truncated = len(candidates) > max_candidates
    kept = candidates[:max_candidates]
    for rank, candidate in enumerate(kept, start=1):
        candidate["rank"] = rank

    gold_ranks = [candidate["rank"] for candidate in kept if candidate["is_gold_or_alias"]]
    all_gold_ranks = [candidate["rank"] for candidate in candidates if candidate["is_gold_or_alias"]]
    audit = {
        "num_candidates_total": len(candidates),
        "num_candidates_kept": len(kept),
        "truncated": truncated,
        "gold_covered_total": bool(all_gold_ranks),
        "gold_covered_kept": bool(gold_ranks),
        "gold_best_rank_total": min(all_gold_ranks) if all_gold_ranks else None,
        "gold_best_rank_kept": min(gold_ranks) if gold_ranks else None,
        "num_support_candidates_kept": sum(1 for c in kept if c["has_support_occurrence"]),
        "num_distractor_only_candidates_kept": sum(1 for c in kept if not c["has_support_occurrence"]),
    }
    return {
        "sample_id": sample["sample_id"],
        "split": sample.get("split", ""),
        "task_name": sample.get("task_name", ""),
        "question": sample.get("question", ""),
        "gold_answer": gold,
        "answer_aliases": aliases,
        "candidates": kept,
        "audit": audit,
        "execution_boundary": [
            "candidate strings are extracted from evidence text only",
            "gold/aliases used only for offline labels and coverage audit",
            "do not expose is_gold_or_alias or gold ranks as online inputs",
        ],
    }


def parse_evidence_items(full_evidence: str) -> list[dict[str, str]]:
    items = []
    pattern = re.compile(r"^\[(?P<idx>\d+)\]\s*\((?P<kind>support|distractor)\)\s*(?P<body>.*)$", re.I)
    for line in full_evidence.splitlines():
        line = line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            body = match.group("body").strip()
            title, text = split_title_text(body)
            items.append(
                {
                    "index": match.group("idx"),
                    "kind": match.group("kind").lower(),
                    "title": title,
                    "text": text,
                    "raw": line,
                }
            )
        else:
            title, text = split_title_text(line)
            items.append({"index": "", "kind": "unknown", "title": title, "text": text, "raw": line})
    return items


def split_title_text(body: str) -> tuple[str, str]:
    if ":" not in body:
        return "", body.strip()
    title, text = body.split(":", 1)
    return title.strip(), text.strip()


def extract_candidates_from_item(item: dict[str, str], item_index: int) -> list[dict[str, Any]]:
    candidates = []
    source_text = " ".join(part for part in [item.get("title", ""), item.get("text", "")] if part)
    if item.get("title"):
        candidates.append(make_candidate(item["title"], "title", item, item_index))
    for text, rule in extract_surface_spans(source_text):
        candidates.append(make_candidate(text, rule, item, item_index))
    return candidates


def extract_surface_spans(text: str) -> list[tuple[str, str]]:
    spans: list[tuple[str, str]] = []
    for match in re.finditer(r'"([^"\\n]{2,80})"|“([^”\\n]{2,80})”|\'([^\'\\n]{2,80})\'', text):
        value = next(group for group in match.groups() if group)
        spans.append((clean_candidate(value), "quoted_span"))
    for match in re.finditer(
        rf"\b(?:{MONTHS})\s+\d{{1,2}}(?:,\s*\d{{3,4}})?\b",
        text,
    ):
        spans.append((clean_candidate(match.group(0)), "date_phrase"))
    for match in re.finditer(
        r"\b(?:early|mid|late)[-\s]?\d{1,2}(?:st|nd|rd|th)[-\s]+century\b|\b\d{1,2}(?:st|nd|rd|th)[-\s]+century\b",
        text,
        flags=re.I,
    ):
        spans.append((clean_candidate(match.group(0)), "century_phrase"))
    for match in re.finditer(
        r"\b(?:spring|summer|fall|autumn|winter)\s+of\s+\d{3,4}\b",
        text,
        flags=re.I,
    ):
        spans.append((clean_candidate(match.group(0)), "season_year_phrase"))
    for match in re.finditer(
        r"\b(?:over|about|around|approximately|more than|at least|nearly)?\s*\d{1,4}(?:[,.]\d{3})*(?:\.\d+)?\s+(?:million|billion|thousand|percent|people|adherents)\b",
        text,
        flags=re.I,
    ):
        spans.append((clean_candidate(match.group(0)), "quantity_phrase"))
    for match in re.finditer(r"\b\d{1,4}(?:[,.]\d{3})*(?:\.\d+)?\b|\b\d{3,4}s\b", text):
        spans.append((clean_candidate(match.group(0)), "number_or_year"))
    for match in re.finditer(r"\bseason\s+(?:\d+|[a-z]+)\b", text, flags=re.I):
        spans.append((clean_candidate(match.group(0)), "season_phrase"))
    for match in re.finditer(r"\b(?:" + "|".join(sorted(NUMBER_WORDS)) + r")\b", text, flags=re.I):
        spans.append((clean_candidate(match.group(0)), "number_word"))
    for match in re.finditer(r"\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿ-]+\s+language\b", text):
        spans.append((clean_candidate(match.group(0)), "language_phrase"))
    for span, rule in capitalized_spans(text):
        spans.append((span, rule))
    return [(value, rule) for value, rule in spans if value]


def capitalized_spans(text: str) -> list[tuple[str, str]]:
    # Punctuation boundaries matter: without them, "Woodrow Wilson (December 28"
    # becomes one huge name-like span and floods the candidate list with noise.
    segments = re.split(r"[\n\r\t,;:()\[\]{}]|--| - ", text)
    spans: list[tuple[str, str]] = []
    for segment in segments:
        tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9][A-Za-zÀ-ÖØ-öø-ÿ0-9'’.-]*", segment)
        current: list[str] = []
        for token in tokens:
            stripped = token.strip(".")
            lowered = stripped.lower()
            starts_like_name = bool(stripped) and (
                stripped[0].isupper()
                or stripped.isupper()
                or lowered in ORDINAL_WORDS
                or re.fullmatch(r"\d+(?:st|nd|rd|th)?", lowered)
            )
            if starts_like_name or (current and lowered in CONNECTOR_WORDS):
                current.append(stripped)
                continue
            flush_capitalized(current, spans)
            current = []
        flush_capitalized(current, spans)
    return spans


def flush_capitalized(tokens: list[str], spans: list[tuple[str, str]]) -> None:
    if not tokens:
        return
    while tokens and tokens[-1].lower() in CONNECTOR_WORDS:
        tokens.pop()
    if not tokens:
        return
    text = clean_candidate(" ".join(tokens[:10]))
    if text and len(text) <= 120:
        spans.append((text, "capitalized_full_span"))
    for token in tokens:
        lowered = token.lower()
        if lowered not in CONNECTOR_WORDS and not re.fullmatch(r"\d+(?:st|nd|rd|th)?", lowered):
            single = clean_candidate(token)
            if single and len(single) <= 40:
                spans.append((single, "capitalized_single"))
    if len(tokens) <= 2:
        return
    # Add a small number of shorter contiguous subspans for names embedded in
    # longer title-like phrases, but keep them behind full spans in ranking.
    for size in (2, 3, 4):
        if size > len(tokens):
            continue
        for start in range(0, len(tokens) - size + 1):
            chunk = tokens[start : start + size]
            if any(tok.lower() not in CONNECTOR_WORDS for tok in chunk):
                sub = clean_candidate(" ".join(chunk))
                if sub and len(sub) <= 80:
                    spans.append((sub, "capitalized_subspan"))


def make_candidate(text: str, rule: str, item: dict[str, str], item_index: int) -> dict[str, Any]:
    return {
        "text": clean_candidate(text),
        "rule": rule,
        "evidence_kind": item.get("kind", ""),
        "evidence_index": item.get("index", ""),
        "item_index": item_index,
        "source_title": item.get("title", ""),
    }


def clean_candidate(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip(" \t\n\r.,;:()[]{}")
    return text


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    support_rank = 0 if candidate.get("has_support_occurrence") else 1
    rule_priority = {
        "title": 0,
        "season_phrase": 1,
        "date_phrase": 2,
        "century_phrase": 3,
        "season_year_phrase": 4,
        "quantity_phrase": 5,
        "language_phrase": 6,
        "number_word": 7,
        "number_or_year": 8,
        "quoted_span": 9,
        "capitalized_full_span": 10,
        "capitalized_subspan": 11,
        "capitalized_single": 12,
    }.get(str(candidate.get("rule", "")), 9)
    text_len = len(str(candidate.get("text", "")).split())
    return (
        support_rank,
        rule_priority,
        text_len,
        int(candidate.get("item_index", 9999)),
        str(candidate.get("normalized", "")),
    )


def summarize(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    rule_counts: Counter[str],
    topks: list[int],
) -> dict[str, Any]:
    n = len(rows)
    topk_metrics = {}
    for topk in topks:
        topk_metrics[f"gold_covered_top{topk}"] = mean(
            [
                bool(metric["gold_best_rank_total"] and metric["gold_best_rank_total"] <= topk)
                for metric in metrics
            ]
        )
    values_total = [metric["num_candidates_total"] for metric in metrics]
    values_kept = [metric["num_candidates_kept"] for metric in metrics]
    summary_metrics = {
        "num_samples": n,
        "gold_covered_total": mean([metric["gold_covered_total"] for metric in metrics]),
        "gold_covered_kept": mean([metric["gold_covered_kept"] for metric in metrics]),
        "truncated_rate": mean([metric["truncated"] for metric in metrics]),
        "candidate_total_mean": mean(values_total),
        "candidate_kept_mean": mean(values_kept),
        "candidate_total_p50": percentile(values_total, 0.50),
        "candidate_total_p90": percentile(values_total, 0.90),
        "candidate_total_p95": percentile(values_total, 0.95),
        "candidate_total_max": max(values_total) if values_total else 0,
        **topk_metrics,
    }
    return {
        "created_at": int(time.time()),
        "status": "pass",
        "manifest_json": args.manifest_json,
        "output_dir": args.output_dir,
        "candidates_jsonl": str(Path(args.output_dir) / "candidates.jsonl"),
        "max_candidates": args.max_candidates,
        "topk_report": topks,
        "metrics": summary_metrics,
        "rule_counts": dict(sorted(rule_counts.items())),
        "execution_boundary": [
            "local-only candidate construction and coverage audit",
            "no model generation",
            "no optimizer or backward",
            "no SwanLab run",
            "candidate strings are evidence-derived; gold is not injected as a candidate",
            "gold/aliases used only for offline label and coverage metrics",
        ],
    }


def mean(values: list[Any]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def normalize_qa(text: Any) -> str:
    text = "" if text is None else str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [token for token in text.split() if token not in {"a", "an", "the"}]
    return " ".join(tokens)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
