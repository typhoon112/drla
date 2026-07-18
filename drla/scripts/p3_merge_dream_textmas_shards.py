"""Merge P3 Dream TextMAS shard outputs.

This local-only utility merges completed ``generations.jsonl`` files from
Dream TextMAS shards, validates duplicate row ids and condition coverage, and
writes a merged run root that can be passed to the existing Phase C aggregator.
It does not run models, train, or create SwanLab runs.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    summary = merge(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-rows", type=int, default=0)
    parser.add_argument("--expected-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def merge(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    input_summaries = []
    for input_dir_text in args.input_dir:
        input_dir = Path(input_dir_text)
        generations_path = input_dir / "generations.jsonl"
        summary_path = input_dir / "summary.json"
        shard_rows = read_jsonl(generations_path)
        rows.extend(shard_rows)
        input_summaries.append(
            {
                "input_dir": str(input_dir),
                "generations_jsonl": str(generations_path),
                "summary_json": str(summary_path) if summary_path.exists() else "",
                "num_rows": len(shard_rows),
            }
        )

    duplicate_row_ids = sorted(row_id for row_id, count in Counter(str(row.get("row_id", "")) for row in rows).items() if count > 1)
    rows_sorted = sorted(rows, key=lambda row: (str(row.get("sample_id", "")), condition_order(str(row.get("condition", "")))))
    sample_ids = sorted({str(row.get("sample_id", "")) for row in rows_sorted})
    condition_counts = Counter(str(row.get("condition", "")) for row in rows_sorted)
    status_errors = [row for row in rows_sorted if row.get("status") == "error"]

    failures = []
    if duplicate_row_ids:
        failures.append(f"duplicate_row_ids={len(duplicate_row_ids)}")
    if args.expected_rows and len(rows_sorted) != args.expected_rows:
        failures.append(f"expected_rows={args.expected_rows}, actual={len(rows_sorted)}")
    if args.expected_samples and len(sample_ids) != args.expected_samples:
        failures.append(f"expected_samples={args.expected_samples}, actual={len(sample_ids)}")
    if status_errors:
        failures.append(f"row_status_errors={len(status_errors)}")

    generations_path = output_dir / "generations.jsonl"
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(generations_path, rows_sorted)
    metrics = {
        "status_pass": int(not failures),
        "num_rows": len(rows_sorted),
        "num_samples": len(sample_ids),
        "num_duplicate_row_ids": len(duplicate_row_ids),
        "num_status_errors": len(status_errors),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "created_at": int(time.time()),
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "input_summaries": input_summaries,
        "generations_jsonl": str(generations_path),
        "metrics_jsonl": str(metrics_path),
        "num_rows": len(rows_sorted),
        "num_samples": len(sample_ids),
        "condition_counts": dict(sorted(condition_counts.items())),
        "duplicate_row_ids_preview": duplicate_row_ids[:20],
        "execution_boundary": [
            "local-only P3 Dream shard merge",
            "no model generation",
            "no optimizer or backward",
            "no SwanLab run",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_json"] = str(summary_path)
    if failures:
        raise RuntimeError("; ".join(failures))
    return summary


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Expected object at {path}:{line_no}")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def condition_order(condition: str) -> int:
    order = {
        "single_q_only": 0,
        "single_full_info": 1,
        "textmas_matched": 2,
        "textmas_no_message": 3,
        "textmas_shuffled_message": 4,
        "textmas_wrong_evidence_or_wrong_shard": 5,
        "textmas_compressed_state": 6,
    }
    return order.get(condition, 100)


if __name__ == "__main__":
    main()
