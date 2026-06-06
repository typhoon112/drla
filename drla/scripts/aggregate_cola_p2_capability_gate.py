"""Aggregate Cola P2 capability-gate summaries.

This is a local-only post-processing script.  It reads completed
``run_cola_p2_capability_gate.py`` output directories and writes a compact CSV
plus a JSON summary for document updates and paper-table triage.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import time
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = "/data1/luyifei/drla/outputs/p2_capability_gate/aggregate_20260601"


def main() -> None:
    summary = aggregate_capability_gate(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", action="append", default=[])
    parser.add_argument("--glob", default="")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if not args.summary_json and not args.glob:
        raise ValueError("Pass at least one --summary-json or --glob")
    return args


def aggregate_capability_gate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = collect_summary_paths(args.summary_json, args.glob)
    mode_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []

    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        config = summary.get("config", {})
        task_summaries = summary.get("task_summaries", [])
        for row in task_summaries:
            mode_rows.append(
                {
                    "task": row["task"],
                    "mode": row["mode"],
                    "prompt_variant": config.get("prompt_variant", ""),
                    "single_prompt_variant": config.get("single_prompt_variant", ""),
                    "role_prompt_variant": config.get("role_prompt_variant", ""),
                    "answer_type": row["answer_type"],
                    "num_samples": row["num_samples"],
                    "accuracy": row["accuracy"],
                    "score_mean": row["score_mean"],
                    "random_floor": row["random_floor"],
                    "nonempty_rate": row["nonempty_rate"],
                    "parseable_rate": row["parseable_rate"],
                    "meets_format_gate": row["meets_format_gate"],
                    "meets_accuracy_gate": row["meets_accuracy_gate"],
                    "requires_execution_gate": row["requires_execution_gate"],
                    "gate_pass": row["gate_pass"],
                    "admitted_for_main": row["admitted_for_main"],
                    "summary_json": str(path),
                    "generations_jsonl": summary.get("generations_jsonl", ""),
                    "metrics_jsonl": summary.get("metrics_jsonl", ""),
                    "elapsed_seconds": summary.get("elapsed_seconds", 0.0),
                }
            )

    by_task: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in mode_rows:
        by_task.setdefault(
            (
                row["task"],
                row.get("prompt_variant", ""),
                row.get("single_prompt_variant", ""),
                row.get("role_prompt_variant", ""),
            ),
            [],
        ).append(row)
    for (task, prompt_variant, single_prompt_variant, role_prompt_variant), rows in sorted(
        by_task.items()
    ):
        modes = {row["mode"]: row for row in rows}
        single = modes.get("single", {})
        role = modes.get("role_textmas", {})
        admitted = bool(
            single.get("gate_pass", False)
            and role.get("gate_pass", False)
            and single.get("admitted_for_main", False)
            and role.get("admitted_for_main", False)
        )
        task_rows.append(
            {
                "task": task,
                "prompt_variant": prompt_variant,
                "single_prompt_variant": single_prompt_variant,
                "role_prompt_variant": role_prompt_variant,
                "answer_type": rows[0].get("answer_type", ""),
                "num_samples": rows[0].get("num_samples", 0),
                "single_accuracy": single.get("accuracy", ""),
                "single_parseable_rate": single.get("parseable_rate", ""),
                "single_gate_pass": single.get("gate_pass", False),
                "role_textmas_accuracy": role.get("accuracy", ""),
                "role_textmas_parseable_rate": role.get("parseable_rate", ""),
                "role_textmas_gate_pass": role.get("gate_pass", False),
                "admitted_for_main": admitted,
            }
        )

    mode_csv = output_dir / "mode_gate_summary.csv"
    task_csv = output_dir / "task_gate_summary.csv"
    write_csv(mode_csv, mode_rows)
    write_csv(task_csv, task_rows)
    aggregate = {
        "created_at": int(time.time()),
        "num_summary_json": len(paths),
        "num_mode_rows": len(mode_rows),
        "num_tasks": len(task_rows),
        "admitted_tasks": [row["task"] for row in task_rows if row["admitted_for_main"]],
        "mode_gate_summary_csv": str(mode_csv),
        "task_gate_summary_csv": str(task_csv),
        "summary_jsons": [str(path) for path in paths],
        "task_rows": task_rows,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    aggregate["summary_json"] = str(summary_path)
    return aggregate


def collect_summary_paths(raw_paths: list[str], pattern: str) -> list[Path]:
    paths = [Path(path) for path in raw_paths]
    if pattern:
        paths.extend(Path(path) for path in sorted(glob.glob(pattern)))
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not resolved.exists():
            raise FileNotFoundError(f"summary_json does not exist: {resolved}")
        seen.add(resolved)
        unique.append(resolved)
    return unique


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
