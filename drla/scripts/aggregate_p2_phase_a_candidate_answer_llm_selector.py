"""Aggregate Phase A LLM candidate-answer selector shards.

This is a local-only post-processor. It merges prediction JSONL files from
semantic candidate selector shards, deduplicates by sample id, recomputes
offline metrics, and writes a reproducible aggregate artifact. It never runs
models, never trains, never reads held-out data by itself, and never logs to
SwanLab.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p2_phase_a_candidate_selectors/"
    "musique_candidate_selector_qwen3_8b_fp8_train1000_top128_aggregate_20260606"
)


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Prediction JSONL files or directories containing predictions.jsonl.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = resolve_input_paths(args.inputs)
    merged = merge_predictions(input_paths)
    predictions_path = output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, merged)
    metrics = compute_metrics(merged)
    summary = {
        "created_at": int(time.time()),
        "status": "pass",
        "input_paths": [str(path) for path in input_paths],
        "output_dir": str(output_dir),
        "predictions_jsonl": str(predictions_path),
        "num_input_files": len(input_paths),
        "num_predictions": len(merged),
        "metrics": metrics,
        "execution_boundary": [
            "local-only semantic selector shard aggregate",
            "no model generation",
            "no training",
            "no SwanLab run",
            "gold labels used only because shard predictions already contain offline scores",
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


def resolve_input_paths(inputs: list[str]) -> list[Path]:
    paths = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            path = path / "predictions.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


def merge_predictions(paths: list[Path]) -> list[dict[str, Any]]:
    by_sample: dict[str, dict[str, Any]] = {}
    duplicates = Counter()
    for path in paths:
        for row in read_jsonl(path):
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                raise ValueError(f"missing sample_id in {path}")
            if sample_id in by_sample:
                duplicates[sample_id] += 1
                continue
            row["source_predictions_jsonl"] = str(path)
            by_sample[sample_id] = row
    merged = list(by_sample.values())
    merged.sort(key=lambda row: str(row.get("sample_id", "")))
    if duplicates:
        duplicate_report = ", ".join(f"{key}:{value}" for key, value in duplicates.most_common(10))
        raise ValueError(f"duplicate sample_ids across shards: {duplicate_report}")
    return merged


def compute_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [float(row.get("primary_score", 0.0)) for row in predictions]
    token_f1 = [float(row.get("token_f1", 0.0)) for row in predictions]
    exact = [float(row.get("exact_match", 0.0)) for row in predictions]
    covered_rows = [row for row in predictions if row.get("oracle_gold_covered_kept")]
    candidate_match_rows = [
        row
        for row in predictions
        if float((row.get("matched_candidate") or {}).get("primary_score", 0.0)) > 0.0
        or float((row.get("matched_candidate") or {}).get("token_f1", 0.0)) >= 0.99
    ]
    return {
        "selected_primary": mean(primary),
        "selected_token_f1": mean(token_f1),
        "selected_exact_match": mean(exact),
        "oracle_coverage_kept": mean([row.get("oracle_gold_covered_kept") for row in predictions]),
        "selected_given_covered": mean([row.get("primary_score", 0.0) for row in covered_rows]),
        "candidate_exact_or_high_f1_rate": len(candidate_match_rows) / len(predictions) if predictions else 0.0,
        "num_predictions": len(predictions),
        "num_oracle_covered": len(covered_rows),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
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


def mean(values: list[Any]) -> float:
    values = list(values)
    return sum(float(value) for value in values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
