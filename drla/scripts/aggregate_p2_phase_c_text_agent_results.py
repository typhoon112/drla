"""Aggregate Phase C text-agent runs and apply admission gates.

This local-only script reads ``run_p2_phase_c_text_agents.py`` generations and
computes condition metrics, paired deltas, bootstrap confidence intervals, and
the documented Phase C benchmark-admission gates.  It does not run models,
train adapters, inspect held-out generations, or create SwanLab runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import tempfile
import time
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = "/data1/luyifei/drla/outputs/p2_phase_c_text_agent_aggregates/aggregate_20260601"

REQUIRED_CONDITIONS = [
    "single_q_only",
    "single_full_info",
    "textmas_matched",
    "textmas_no_message",
    "textmas_shuffled_message",
    "textmas_wrong_evidence_or_wrong_shard",
]

PAIRED_COMPARISONS = [
    ("single_full_info", "single_q_only", "full_info_vs_question_only"),
    ("textmas_matched", "textmas_no_message", "matched_vs_no_message"),
    ("textmas_matched", "textmas_shuffled_message", "matched_vs_shuffled"),
    ("textmas_matched", "textmas_wrong_evidence_or_wrong_shard", "matched_vs_wrong_evidence"),
]


def main() -> None:
    args = parse_args()
    summary = run_selfcheck(args) if args.selfcheck else aggregate(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations-jsonl", default="")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-full-info-primary", type=float, default=0.2)
    parser.add_argument("--min-parseable-rate", type=float, default=0.95)
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260601)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if not args.selfcheck and not args.generations_jsonl:
        raise ValueError("Pass --generations-jsonl, or use --selfcheck")
    if args.bootstrap_iters <= 0:
        raise ValueError("--bootstrap-iters must be positive")
    return args


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(Path(args.generations_jsonl))
    condition_metrics = condition_summary(rows)
    paired_metrics = paired_summary(
        rows,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_seed=args.bootstrap_seed,
    )
    gate = evaluate_gate(condition_metrics, paired_metrics, args)

    condition_csv = output_dir / "condition_metrics.csv"
    paired_csv = output_dir / "paired_comparisons.csv"
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    write_condition_csv(condition_csv, condition_metrics)
    write_paired_csv(paired_csv, paired_metrics)
    metrics = {
        "num_rows": len(rows),
        "num_conditions": len(condition_metrics),
        "admitted": int(gate["admitted"]),
        "num_failed_gates": len(gate["failed_gates"]),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "created_at": int(time.time()),
        "status": "pass",
        "admitted": gate["admitted"],
        "gate": gate,
        "generations_jsonl": args.generations_jsonl,
        "condition_metrics": condition_metrics,
        "paired_metrics": paired_metrics,
        "condition_metrics_csv": str(condition_csv),
        "paired_comparisons_csv": str(paired_csv),
        "metrics_jsonl": str(metrics_path),
        "execution_boundary": [
            "local-only Phase C text-agent result aggregation",
            "no model generation",
            "no optimizer or backward",
            "no SwanLab run",
            "gold/scorer fields are read only from completed offline-scored generations",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_json"] = str(summary_path)
    return summary


def run_selfcheck(args: argparse.Namespace) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        generations_path = Path(tmpdir) / "toy_generations.jsonl"
        write_jsonl(generations_path, make_toy_generations())
        self_args = argparse.Namespace(
            **{
                **vars(args),
                "generations_jsonl": str(generations_path),
                "output_dir": args.output_dir,
                "overwrite": True,
            }
        )
        summary = aggregate(self_args)
    if not summary["admitted"]:
        raise AssertionError(f"Phase C aggregate self-check expected admitted=true: {summary['gate']}")
    summary["selfcheck"] = {
        "status": "pass",
        "meaning": "Toy scored generations exercised aggregation/gate logic only; not an experiment.",
    }
    return summary


def condition_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("condition", "")), []).append(row)
    metrics = {}
    for condition, group in sorted(grouped.items()):
        predictions = [str(row.get("prediction", "")).strip() for row in group]
        metrics[condition] = {
            "num_rows": len(group),
            "primary_score_mean": mean([float(row.get("primary_score", 0.0)) for row in group]),
            "token_f1_mean": mean([float(row.get("token_f1", 0.0)) for row in group]),
            "exact_match_mean": mean([float(row.get("exact_match", 0.0)) for row in group]),
            "parseable_rate": mean([float(bool(pred)) for pred in predictions]),
            "nonempty_rate": mean([float(bool(pred)) for pred in predictions]),
            "unknown_rate": mean([float(pred.lower() in {"unknown", "i don't know", "not enough information"}) for pred in predictions]),
        }
    return metrics


def paired_summary(rows: list[dict[str, Any]], *, bootstrap_iters: int, bootstrap_seed: int) -> dict[str, dict[str, Any]]:
    by_sample_condition: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        by_sample_condition[(str(row.get("sample_id", "")), str(row.get("condition", "")))] = row
    metrics = {}
    for condition_a, condition_b, name in PAIRED_COMPARISONS:
        deltas = []
        for sample_id in sorted({key[0] for key in by_sample_condition}):
            row_a = by_sample_condition.get((sample_id, condition_a))
            row_b = by_sample_condition.get((sample_id, condition_b))
            if row_a is None or row_b is None:
                continue
            deltas.append(float(row_a.get("primary_score", 0.0)) - float(row_b.get("primary_score", 0.0)))
        ci = bootstrap_ci_lower(deltas, bootstrap_iters=bootstrap_iters, seed=bootstrap_seed)
        metrics[name] = {
            "condition_a": condition_a,
            "condition_b": condition_b,
            "num_pairs": len(deltas),
            "primary_delta_mean": mean(deltas),
            "primary_delta_ci_lower": ci["ci_lower"],
            "primary_delta_ci_upper": ci["ci_upper"],
        }
    return metrics


def evaluate_gate(
    condition_metrics: dict[str, dict[str, Any]],
    paired_metrics: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    failed = []
    for condition in REQUIRED_CONDITIONS:
        if condition not in condition_metrics:
            failed.append({"gate": "required_condition_present", "condition": condition, "value": None})
    if failed:
        return {"admitted": False, "failed_gates": failed}

    full_info = condition_metrics["single_full_info"]
    matched = condition_metrics["textmas_matched"]
    if full_info["primary_score_mean"] < args.min_full_info_primary:
        failed.append(
            {
                "gate": "single_full_info_above_floor",
                "condition": "single_full_info",
                "value": full_info["primary_score_mean"],
                "threshold": args.min_full_info_primary,
            }
        )
    for condition in ["single_full_info", "textmas_matched"]:
        parseable = condition_metrics[condition]["parseable_rate"]
        if parseable < args.min_parseable_rate:
            failed.append(
                {
                    "gate": "parseable_rate",
                    "condition": condition,
                    "value": parseable,
                    "threshold": args.min_parseable_rate,
                }
            )
    for comparison in [
        "full_info_vs_question_only",
        "matched_vs_no_message",
        "matched_vs_shuffled",
        "matched_vs_wrong_evidence",
    ]:
        metric = paired_metrics.get(comparison, {})
        if metric.get("num_pairs", 0) == 0:
            failed.append({"gate": "paired_comparison_available", "comparison": comparison, "value": 0})
            continue
        if metric.get("primary_delta_ci_lower", 0.0) <= 0.0:
            failed.append(
                {
                    "gate": "paired_ci_lower_positive",
                    "comparison": comparison,
                    "value": metric.get("primary_delta_ci_lower", 0.0),
                    "delta_mean": metric.get("primary_delta_mean", 0.0),
                }
            )
    return {
        "admitted": not failed,
        "failed_gates": failed,
        "thresholds": {
            "min_full_info_primary": args.min_full_info_primary,
            "min_parseable_rate": args.min_parseable_rate,
            "paired_delta_ci_lower_must_be_gt": 0.0,
        },
    }


def bootstrap_ci_lower(deltas: list[float], *, bootstrap_iters: int, seed: int) -> dict[str, float]:
    if not deltas:
        return {"ci_lower": 0.0, "ci_upper": 0.0}
    if len(deltas) == 1:
        return {"ci_lower": deltas[0], "ci_upper": deltas[0]}
    rng = random.Random(seed)
    means = []
    n = len(deltas)
    for _ in range(bootstrap_iters):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))
    means.sort()
    lower_idx = int(0.025 * (len(means) - 1))
    upper_idx = int(0.975 * (len(means) - 1))
    return {"ci_lower": means[lower_idx], "ci_upper": means[upper_idx]}


def write_condition_csv(path: Path, metrics: dict[str, dict[str, Any]]) -> None:
    fieldnames = [
        "condition",
        "num_rows",
        "primary_score_mean",
        "token_f1_mean",
        "exact_match_mean",
        "parseable_rate",
        "nonempty_rate",
        "unknown_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for condition, values in sorted(metrics.items()):
            writer.writerow({"condition": condition, **values})


def write_paired_csv(path: Path, metrics: dict[str, dict[str, Any]]) -> None:
    fieldnames = [
        "comparison",
        "condition_a",
        "condition_b",
        "num_pairs",
        "primary_delta_mean",
        "primary_delta_ci_lower",
        "primary_delta_ci_upper",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for comparison, values in sorted(metrics.items()):
            writer.writerow({"comparison": comparison, **values})


def make_toy_generations() -> list[dict[str, Any]]:
    rows = []
    conditions_to_score = {
        "single_q_only": 0.0,
        "single_full_info": 1.0,
        "textmas_matched": 1.0,
        "textmas_no_message": 0.0,
        "textmas_shuffled_message": 0.0,
        "textmas_wrong_evidence_or_wrong_shard": 0.0,
        "textmas_compressed_state": 1.0,
    }
    for sample_index in range(12):
        sample_id = f"toy_{sample_index:03d}"
        for condition, score in conditions_to_score.items():
            rows.append(
                {
                    "row_id": f"{sample_id}::{condition}",
                    "sample_id": sample_id,
                    "condition": condition,
                    "prediction": "correct" if score else "wrong",
                    "primary_score": score,
                    "token_f1": score,
                    "exact_match": score,
                }
            )
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
