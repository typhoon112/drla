"""Merge Phase A teacher evidence-agent message shards.

This local-only helper combines ``build_p2_phase_a_teacher_agent_messages.py``
shards, validates duplicate row/sample keys, and writes a single
``generations.jsonl`` compatible with ``build_p2_phase_a_cola_interface_sft.py``.
It does not run models, train adapters, inspect held-out data, or create
SwanLab runs.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p2_phase_a_teacher_agent_messages/"
    "merged_teacher_agent_messages_20260605"
)


def main() -> None:
    summary = merge_shards(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-glob", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-rows", type=int, default=0)
    parser.add_argument("--expected-agent-messages-per-row", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def merge_shards(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_resolved = output_dir.resolve()
    shard_dirs = [
        Path(path)
        for path in sorted(glob.glob(args.shard_glob))
        if Path(path).is_dir() and Path(path).resolve() != output_resolved
    ]
    if not shard_dirs:
        raise FileNotFoundError(f"no shard directories matched: {args.shard_glob}")

    rows: list[dict[str, Any]] = []
    shard_summaries = []
    for shard_dir in shard_dirs:
        generations_path = shard_dir / "generations.jsonl"
        summary_path = shard_dir / "summary.json"
        if not generations_path.exists():
            raise FileNotFoundError(generations_path)
        shard_rows = read_jsonl(generations_path)
        rows.extend(shard_rows)
        shard_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        shard_summaries.append(
            {
                "shard_dir": str(shard_dir),
                "num_generations": len(shard_rows),
                "summary_status": shard_summary.get("status", ""),
                "summary_num_generations": shard_summary.get("num_generations"),
                "summary_num_rows_requested": shard_summary.get("num_rows_requested"),
            }
        )

    duplicate_row_ids = duplicate_values(str(row.get("row_id", "")) for row in rows)
    duplicate_sample_conditions = duplicate_values(
        f"{row.get('sample_id', '')}::{row.get('condition', '')}" for row in rows
    )
    if duplicate_row_ids:
        raise RuntimeError(f"duplicate row_id values: {duplicate_row_ids[:5]}")
    if duplicate_sample_conditions:
        raise RuntimeError(f"duplicate sample/condition values: {duplicate_sample_conditions[:5]}")
    if args.expected_rows and len(rows) != args.expected_rows:
        raise RuntimeError(f"expected {args.expected_rows} rows, found {len(rows)}")

    rows.sort(key=lambda row: (str(row.get("sample_id", "")), str(row.get("condition", ""))))
    message_lengths = [
        len(str(message.get("message", "")).strip())
        for row in rows
        for message in row.get("agent_messages", [])
        if isinstance(message, dict)
    ]
    row_agent_counts = Counter(len(row.get("agent_messages", [])) for row in rows)
    if args.expected_agent_messages_per_row:
        bad_counts = {
            str(count): value
            for count, value in sorted(row_agent_counts.items())
            if count != args.expected_agent_messages_per_row
        }
        if bad_counts:
            raise RuntimeError(f"unexpected agent_messages count distribution: {bad_counts}")

    generations_path = output_dir / "generations.jsonl"
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(generations_path, rows)
    metrics = {
        "num_rows": len(rows),
        "num_shards": len(shard_dirs),
        "num_agent_messages": len(message_lengths),
        "nonempty_agent_message_rate": (
            sum(1 for length in message_lengths if length > 0) / len(message_lengths)
            if message_lengths
            else 0.0
        ),
        "mean_agent_message_chars": sum(message_lengths) / len(message_lengths) if message_lengths else 0.0,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "created_at": int(time.time()),
        "status": "pass",
        "shard_glob": args.shard_glob,
        "output_dir": str(output_dir),
        "generations_jsonl": str(generations_path),
        "metrics_jsonl": str(metrics_path),
        "num_rows": len(rows),
        "num_shards": len(shard_dirs),
        "shards": shard_summaries,
        "condition_counts": dict(sorted(Counter(str(row.get("condition", "")) for row in rows).items())),
        "split_counts": dict(sorted(Counter(str(row.get("split", "")) for row in rows).items())),
        "row_agent_message_count_distribution": dict(sorted(row_agent_counts.items())),
        "metrics": metrics,
        "execution_boundary": [
            "local-only Phase A teacher-message shard merge",
            "no model generation",
            "no optimizer or backward",
            "no SwanLab run",
            "does not inspect held-out generations or tune prompts",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_json"] = str(summary_path)
    return summary


def duplicate_values(values: Any) -> list[str]:
    counts = Counter(value for value in values if value)
    return sorted(value for value, count in counts.items() if count > 1)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"expected JSON object at {path}:{line_no}")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
