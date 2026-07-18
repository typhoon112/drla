"""Merge P3 Dream trace shard outputs with call-id remapping.

Each trace shard starts call ids at ``call_000001``. This merger rewrites trace
call ids and generation ``trace_call_ids`` with a shard prefix so downstream D4
frontier building can map each generation row to the correct solver trace.
It is local-only and does not run models, train, or create SwanLab runs.
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
    parser.add_argument("--expected-traces", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def merge(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_generations: list[dict[str, Any]] = []
    all_traces: list[dict[str, Any]] = []
    all_call_metrics: list[dict[str, Any]] = []
    all_selected_rows: list[dict[str, Any]] = []
    input_summaries: list[dict[str, Any]] = []
    manifest_jsons: set[str] = set()
    online_inputs_jsonls: set[str] = set()
    model_paths: set[str] = set()
    for shard_index, input_dir_text in enumerate(args.input_dir):
        input_dir = Path(input_dir_text)
        shard_prefix = f"shard{shard_index:03d}"
        generations = read_jsonl(input_dir / "generations.jsonl")
        traces = read_jsonl(input_dir / "traces.jsonl")
        call_metrics = read_jsonl(input_dir / "dream_trace_call_metrics.jsonl")
        selected_rows = read_jsonl(input_dir / "selected_rows.jsonl") if (input_dir / "selected_rows.jsonl").exists() else []
        shard_summary = read_json(input_dir / "summary.json") if (input_dir / "summary.json").exists() else {}
        if shard_summary.get("manifest_json"):
            manifest_jsons.add(str(shard_summary["manifest_json"]))
        if shard_summary.get("online_inputs_jsonl"):
            online_inputs_jsonls.add(str(shard_summary["online_inputs_jsonl"]))
        if shard_summary.get("model_path"):
            model_paths.add(str(shard_summary["model_path"]))
        call_map = {str(trace.get("call_id", "")): f"{shard_prefix}_{trace.get('call_id', '')}" for trace in traces}
        for trace in traces:
            old_call_id = str(trace.get("call_id", ""))
            trace["original_call_id"] = old_call_id
            trace["call_id"] = call_map[old_call_id]
            trace["source_shard_index"] = shard_index
            trace["source_shard_dir"] = str(input_dir)
        for metric in call_metrics:
            old_call_id = str(metric.get("call_id", ""))
            metric["original_call_id"] = old_call_id
            metric["call_id"] = call_map.get(old_call_id, f"{shard_prefix}_{old_call_id}")
            metric["source_shard_index"] = shard_index
        for row in generations:
            row["trace_call_ids_original"] = list(row.get("trace_call_ids", []))
            row["trace_call_ids"] = [call_map.get(str(call_id), f"{shard_prefix}_{call_id}") for call_id in row.get("trace_call_ids", [])]
            row["source_shard_index"] = shard_index
            row["source_shard_dir"] = str(input_dir)
        all_generations.extend(generations)
        all_traces.extend(traces)
        all_call_metrics.extend(call_metrics)
        all_selected_rows.extend(selected_rows)
        input_summaries.append(
            {
                "input_dir": str(input_dir),
                "summary_json": str(input_dir / "summary.json") if (input_dir / "summary.json").exists() else "",
                "num_generations": len(generations),
                "num_traces": len(traces),
                "num_call_metrics": len(call_metrics),
                "num_selected_rows": len(selected_rows),
            }
        )

    generations_sorted = sorted(
        all_generations,
        key=lambda row: (
            int(row.get("source_shard_index", 0)),
            int(row.get("row_index", 0)),
            str(row.get("row_id", "")),
        ),
    )
    traces_sorted = sorted(
        all_traces,
        key=lambda row: (int(row.get("source_shard_index", 0)), str(row.get("call_id", ""))),
    )
    call_metrics_sorted = sorted(
        all_call_metrics,
        key=lambda row: (int(row.get("source_shard_index", 0)), str(row.get("call_id", ""))),
    )
    selected_rows_sorted = sorted(
        all_selected_rows,
        key=lambda row: (str(row.get("sample_id", "")), condition_order(str(row.get("condition", "")))),
    )
    sample_ids = sorted({str(row.get("sample_id", "")) for row in generations_sorted})
    duplicate_row_ids = sorted(
        row_id for row_id, count in Counter(str(row.get("row_id", "")) for row in generations_sorted).items() if count > 1
    )
    duplicate_call_ids = sorted(
        call_id for call_id, count in Counter(str(row.get("call_id", "")) for row in traces_sorted).items() if count > 1
    )
    status_errors = [row for row in generations_sorted if row.get("status") == "error"]
    missing_trace_ids = sorted(
        {
            str(call_id)
            for row in generations_sorted
            for call_id in row.get("trace_call_ids", [])
            if str(call_id) not in {str(trace.get("call_id", "")) for trace in traces_sorted}
        }
    )

    failures = []
    if duplicate_row_ids:
        failures.append(f"duplicate_row_ids={len(duplicate_row_ids)}")
    if duplicate_call_ids:
        failures.append(f"duplicate_call_ids={len(duplicate_call_ids)}")
    if status_errors:
        failures.append(f"row_status_errors={len(status_errors)}")
    if missing_trace_ids:
        failures.append(f"missing_trace_ids={len(missing_trace_ids)}")
    if args.expected_rows and len(generations_sorted) != args.expected_rows:
        failures.append(f"expected_rows={args.expected_rows}, actual={len(generations_sorted)}")
    if args.expected_samples and len(sample_ids) != args.expected_samples:
        failures.append(f"expected_samples={args.expected_samples}, actual={len(sample_ids)}")
    if args.expected_traces and len(traces_sorted) != args.expected_traces:
        failures.append(f"expected_traces={args.expected_traces}, actual={len(traces_sorted)}")
    if len(manifest_jsons) > 1:
        failures.append(f"multiple_manifest_jsons={len(manifest_jsons)}")
    if len(online_inputs_jsonls) > 1:
        failures.append(f"multiple_online_inputs_jsonls={len(online_inputs_jsonls)}")

    write_jsonl(output_dir / "generations.jsonl", generations_sorted)
    write_jsonl(output_dir / "traces.jsonl", traces_sorted)
    write_jsonl(output_dir / "dream_trace_call_metrics.jsonl", call_metrics_sorted)
    write_jsonl(output_dir / "selected_rows.jsonl", selected_rows_sorted)
    metrics = {
        "status_pass": int(not failures),
        "num_rows": len(generations_sorted),
        "num_samples": len(sample_ids),
        "num_traces": len(traces_sorted),
        "num_call_metrics": len(call_metrics_sorted),
        "num_duplicate_row_ids": len(duplicate_row_ids),
        "num_duplicate_call_ids": len(duplicate_call_ids),
        "num_status_errors": len(status_errors),
        "num_missing_trace_ids": len(missing_trace_ids),
    }
    (output_dir / "metrics.jsonl").write_text(
        json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "created_at": int(time.time()),
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "input_summaries": input_summaries,
        "manifest_json": next(iter(manifest_jsons), ""),
        "online_inputs_jsonl": next(iter(online_inputs_jsonls), ""),
        "model_path": next(iter(model_paths), ""),
        "generations_jsonl": str(output_dir / "generations.jsonl"),
        "traces_jsonl": str(output_dir / "traces.jsonl"),
        "dream_trace_call_metrics_jsonl": str(output_dir / "dream_trace_call_metrics.jsonl"),
        "selected_rows_jsonl": str(output_dir / "selected_rows.jsonl"),
        "metrics_jsonl": str(output_dir / "metrics.jsonl"),
        "num_rows": len(generations_sorted),
        "num_samples": len(sample_ids),
        "num_traces": len(traces_sorted),
        "condition_counts": dict(sorted(Counter(str(row.get("condition", "")) for row in generations_sorted).items())),
        "duplicate_row_ids_preview": duplicate_row_ids[:20],
        "duplicate_call_ids_preview": duplicate_call_ids[:20],
        "missing_trace_ids_preview": missing_trace_ids[:20],
        "execution_boundary": [
            "local-only P3 Dream trace shard merge",
            "no model generation",
            "no optimizer or backward",
            "no SwanLab run",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_json"] = str(output_dir / "summary.json")
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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object in {path}")
    return value


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
