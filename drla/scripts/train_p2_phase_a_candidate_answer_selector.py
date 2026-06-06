"""Train a local candidate-answer selector for Phase A diagnostics.

This is not a deep-learning training script. It uses scikit-learn logistic
regression over evidence-derived candidate features to test whether the current
candidate protocol contains enough signal to select the gold answer. It never
uses held-out data, never calls a model, and never creates SwanLab runs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.evaluation.p2_phase_c_scorers import score_qa_answer


DEFAULT_TRAIN_CANDIDATES_JSONL = (
    "/data1/luyifei/drla/outputs/p2_phase_a_candidate_answers/"
    "musique_train_candidate_answers_10000_seed20260606_20260606/candidates.jsonl"
)
DEFAULT_EVAL_CANDIDATES_JSONL = (
    "/data1/luyifei/drla/outputs/p2_phase_a_candidate_answers/"
    "musique_calibration_candidate_answers_200_seed20260606_20260606/candidates.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p2_phase_a_candidate_selectors/"
    "musique_candidate_selector_train10000_eval_calib200_20260606"
)


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-candidates-jsonl", default=DEFAULT_TRAIN_CANDIDATES_JSONL)
    parser.add_argument("--eval-candidates-jsonl", default=DEFAULT_EVAL_CANDIDATES_JSONL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-candidates-per-sample", type=int, default=128)
    parser.add_argument("--negative-subsample-per-sample", type=int, default=48)
    parser.add_argument("--model-type", choices=["logistic", "hist_gbdt"], default="logistic")
    parser.add_argument("--random-seed", type=int, default=20260606)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(Path(args.train_candidates_jsonl))
    eval_rows = read_jsonl(Path(args.eval_candidates_jsonl))
    if args.max_train_samples:
        train_rows = train_rows[: args.max_train_samples]

    train_x, train_y, train_meta = featurize_rows(
        train_rows,
        max_candidates_per_sample=args.max_candidates_per_sample,
        negative_subsample_per_sample=args.negative_subsample_per_sample,
        seed=args.random_seed,
    )
    eval_x, eval_y, eval_meta = featurize_rows(
        eval_rows,
        max_candidates_per_sample=args.max_candidates_per_sample,
        negative_subsample_per_sample=0,
        seed=args.random_seed,
    )
    if not any(train_y):
        raise ValueError("training candidates contain no positive examples")

    model = make_model(args)
    model.fit(train_x, np.asarray(train_y, dtype=np.int64))
    eval_scores = model.predict_proba(eval_x)[:, 1]
    train_scores = model.predict_proba(train_x)[:, 1]
    predictions = select_by_sample(eval_meta, eval_scores)

    predictions_path = output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    metrics = compute_metrics(predictions, eval_y, eval_scores, train_y, train_scores)
    summary = {
        "created_at": int(time.time()),
        "status": "pass",
        "train_candidates_jsonl": args.train_candidates_jsonl,
        "eval_candidates_jsonl": args.eval_candidates_jsonl,
        "output_dir": str(output_dir),
        "predictions_jsonl": str(predictions_path),
        "num_train_samples": len(train_rows),
        "num_eval_samples": len(eval_rows),
        "num_train_candidate_rows": len(train_y),
        "num_eval_candidate_rows": len(eval_y),
        "max_candidates_per_sample": args.max_candidates_per_sample,
        "negative_subsample_per_sample": args.negative_subsample_per_sample,
        "model_type": args.model_type,
        "metrics": metrics,
        "feature_space_size": len(model.named_steps["vectorizer"].vocabulary_),
        "execution_boundary": [
            "local-only sklearn candidate selector diagnostic",
            "no deep-learning optimizer/backward",
            "no model generation",
            "no SwanLab run",
            "no held-out data",
            "gold labels used only for supervised candidate-label training and offline scoring",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.jsonl").write_text(
        json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def make_model(args: argparse.Namespace) -> Pipeline:
    if args.model_type == "logistic":
        return Pipeline(
            [
                ("vectorizer", DictVectorizer(sparse=True)),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=500,
                        solver="liblinear",
                        random_state=args.random_seed,
                    ),
                ),
            ]
        )
    if args.model_type == "hist_gbdt":
        return Pipeline(
            [
                ("vectorizer", DictVectorizer(sparse=False)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        class_weight="balanced",
                        learning_rate=0.06,
                        max_iter=160,
                        max_leaf_nodes=31,
                        l2_regularization=0.05,
                        random_state=args.random_seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"unknown model_type: {args.model_type}")


def featurize_rows(
    rows: list[dict[str, Any]],
    *,
    max_candidates_per_sample: int,
    negative_subsample_per_sample: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[int], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    features: list[dict[str, Any]] = []
    labels: list[int] = []
    meta: list[dict[str, Any]] = []
    for row in rows:
        candidates = list(row.get("candidates", []))[:max_candidates_per_sample]
        positives = [candidate for candidate in candidates if candidate.get("is_gold_or_alias")]
        negatives = [candidate for candidate in candidates if not candidate.get("is_gold_or_alias")]
        if negative_subsample_per_sample and len(negatives) > negative_subsample_per_sample:
            indices = rng.choice(len(negatives), size=negative_subsample_per_sample, replace=False)
            negatives = [negatives[int(index)] for index in sorted(indices)]
        selected = positives + negatives
        for candidate in selected:
            features.append(candidate_features(row, candidate))
            labels.append(1 if candidate.get("is_gold_or_alias") else 0)
            meta.append(
                {
                    "sample_id": row["sample_id"],
                    "question": row.get("question", ""),
                    "gold_answer": row.get("gold_answer", ""),
                    "answer_aliases": row.get("answer_aliases", []) or [],
                    "candidate": candidate,
                    "candidate_label": int(bool(candidate.get("is_gold_or_alias"))),
                    "oracle_gold_covered_kept": bool(row.get("audit", {}).get("gold_covered_kept")),
                    "oracle_gold_best_rank_kept": row.get("audit", {}).get("gold_best_rank_kept"),
                }
            )
    return features, labels, meta


def candidate_features(row: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    text = str(candidate.get("text", ""))
    question = str(row.get("question", ""))
    qtype = question_type(question)
    rule = str(candidate.get("rule", ""))
    candidate_tokens = token_set(text)
    question_tokens = token_set(question)
    overlap = candidate_tokens & question_tokens
    rank = max(1, int(candidate.get("rank", 999)))
    token_count = max(1, len(candidate_tokens))
    is_numeric = any(char.isdigit() for char in text)
    is_date_like = rule in {"date_phrase", "century_phrase", "season_year_phrase"} or (
        is_numeric and any(word in text.lower() for word in ["century", "season"])
    )
    is_quantity_like = rule in {"quantity_phrase", "number_word"} or (
        is_numeric and any(word in text.lower() for word in ["million", "billion", "thousand", "people"])
    )
    is_language_like = rule == "language_phrase" or "language" in text.lower()
    return {
        "bias": 1.0,
        "rule=" + rule: 1.0,
        "qtype=" + qtype: 1.0,
        "rule_x_qtype=" + rule + "::" + qtype: 1.0,
        "evidence_kind=" + str(candidate.get("evidence_kind", "")): 1.0,
        "has_support_occurrence": float(bool(candidate.get("has_support_occurrence"))),
        "is_title": float(candidate.get("rule") == "title"),
        "is_numeric": float(is_numeric),
        "is_date_like": float(is_date_like),
        "is_quantity_like": float(is_quantity_like),
        "is_language_like": float(is_language_like),
        "qtype_numeric_match": float(qtype in {"when", "how_many"} and is_numeric),
        "qtype_date_match": float(qtype == "when" and is_date_like),
        "qtype_quantity_match": float(qtype == "how_many" and is_quantity_like),
        "qtype_language_match": float(qtype == "language" and is_language_like),
        "qtype_entity_match": float(qtype in {"who", "where", "what_entity"} and not is_numeric),
        "is_short": float(token_count <= 4),
        "rank_log": math.log(rank + 1.0),
        "rank_inverse": 1.0 / rank,
        "candidate_token_count": float(token_count),
        "candidate_char_count_log": math.log(len(text) + 1.0),
        "occurrences_log": math.log(float(candidate.get("occurrences", 1)) + 1.0),
        "item_index": float(candidate.get("item_index", 0)),
        "question_overlap_count": float(len(overlap)),
        "question_overlap_fraction": float(len(overlap) / token_count),
        "candidate_in_question": float(bool(candidate_tokens and candidate_tokens <= question_tokens)),
        "candidate_not_in_question": float(not bool(candidate_tokens and candidate_tokens <= question_tokens)),
        "source_title_overlap": float(len(token_set(candidate.get("source_title", "")) & question_tokens)),
    }


def question_type(question: str) -> str:
    q = question.lower()
    if "how many" in q or "how much" in q:
        return "how_many"
    if "what language" in q or "which language" in q:
        return "language"
    if q.startswith("when") or " what year" in q or " what season" in q:
        return "when"
    if q.startswith("where") or "what city" in q or "what country" in q or "what place" in q:
        return "where"
    if q.startswith("who") or "which person" in q:
        return "who"
    if q.startswith("what") or q.startswith("which"):
        return "what_entity"
    return "other"


def select_by_sample(meta: list[dict[str, Any]], scores: np.ndarray) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = {}
    for item, score in zip(meta, scores):
        grouped.setdefault(str(item["sample_id"]), []).append((item, float(score)))
    predictions = []
    for sample_id, items in grouped.items():
        best_item, best_score = max(items, key=lambda pair: pair[1])
        candidate = best_item["candidate"]
        score = score_qa_answer(
            candidate.get("text", ""),
            best_item.get("gold_answer", ""),
            best_item.get("answer_aliases", []),
        ).to_dict()
        predictions.append(
            {
                "sample_id": sample_id,
                "prediction": candidate.get("text", ""),
                "selector_score": best_score,
                "candidate_rank": candidate.get("rank"),
                "candidate_rule": candidate.get("rule"),
                "candidate_evidence_kind": candidate.get("evidence_kind"),
                "candidate_is_gold_or_alias": int(bool(candidate.get("is_gold_or_alias"))),
                "oracle_gold_covered_kept": best_item.get("oracle_gold_covered_kept"),
                "oracle_gold_best_rank_kept": best_item.get("oracle_gold_best_rank_kept"),
                "gold_answer": best_item.get("gold_answer", ""),
                "score": score,
                "primary_score": score["primary_score"],
                "token_f1": score["token_f1"],
                "exact_match": score["exact_match"],
            }
        )
    predictions.sort(key=lambda row: row["sample_id"])
    return predictions


def compute_metrics(
    predictions: list[dict[str, Any]],
    eval_y: list[int],
    eval_scores: np.ndarray,
    train_y: list[int],
    train_scores: np.ndarray,
) -> dict[str, Any]:
    def safe_auc(labels: list[int], scores: np.ndarray) -> float | None:
        return float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else None

    def safe_ap(labels: list[int], scores: np.ndarray) -> float | None:
        return float(average_precision_score(labels, scores)) if len(set(labels)) > 1 else None

    primary = [float(row["primary_score"]) for row in predictions]
    token_f1 = [float(row["token_f1"]) for row in predictions]
    selected_rules = Counter(str(row.get("candidate_rule", "")) for row in predictions)
    return {
        "train_candidate_auc": safe_auc(train_y, train_scores),
        "train_candidate_average_precision": safe_ap(train_y, train_scores),
        "eval_candidate_auc": safe_auc(eval_y, eval_scores),
        "eval_candidate_average_precision": safe_ap(eval_y, eval_scores),
        "eval_selected_primary": mean(primary),
        "eval_selected_token_f1": mean(token_f1),
        "eval_oracle_coverage_kept": mean([row["oracle_gold_covered_kept"] for row in predictions]),
        "eval_selected_given_covered": mean(
            [
                row["primary_score"]
                for row in predictions
                if row.get("oracle_gold_covered_kept")
            ]
        ),
        "eval_num_predictions": len(predictions),
        "selected_rule_counts": dict(sorted(selected_rules.items())),
    }


def token_set(text: Any) -> set[str]:
    text = "" if text is None else str(text).lower()
    return {token for token in __import__("re").sub(r"[^a-z0-9]+", " ", text).split() if token}


def mean(values: list[Any]) -> float:
    values = list(values)
    return sum(float(value) for value in values) / len(values) if values else 0.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_no}")
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
