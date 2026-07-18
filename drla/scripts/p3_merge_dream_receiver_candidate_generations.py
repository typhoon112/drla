"""Merge local-only D7 receiver-generation shards for V6 candidate pools."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.scripts.p3_run_dream_latent_prefix_eval import aggregate  # noqa: E402
from drla.scripts.run_p2_phase_c_text_agents import read_jsonl, write_jsonl  # noqa: E402


DEFAULT_SOURCE_DIRS = [
    "/data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/dream_layer_receiver_eval_v1_best50_20260607",
    "/data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/dream_layer_receiver_eval_v1_best200_candidates_shard050_20260607",
    "/data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/dream_layer_receiver_eval_v1_best200_candidates_shard100_20260607",
    "/data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/dream_layer_receiver_eval_v1_best200_candidates_shard150_20260607",
]
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/"
    "dream_layer_receiver_eval_v1_best200_candidates_merged_20260607"
)
DEFAULT_CONDITIONS = [
    "no_message",
    "layer_receiver_matched",
    "layer_receiver_shuffled_row",
    "layer_receiver_agent_swap",
    "layer_receiver_zero",
]
FORBIDDEN_PAYLOAD_KEYS = {
    "agent_private_observations",
    "gold_answer",
    "answer_aliases",
    "online_input_fields",
}


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dirs", nargs="+", default=DEFAULT_SOURCE_DIRS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-rows", type=int, default=200)
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    source_summaries = []
    for source_dir_text in args.source_dirs:
        source_dir = Path(source_dir_text)
        generations_path = source_dir / "generations.jsonl"
        if not generations_path.exists():
            raise FileNotFoundError(generations_path)
        source_rows = read_jsonl(generations_path)
        rows.extend(source_rows)
        source_summaries.append(
            {
                "source_dir": str(source_dir),
                "generations_jsonl": str(generations_path),
                "num_generations": len(source_rows),
                "summary_json": str(source_dir / "summary.json") if (source_dir / "summary.json").exists() else None,
            }
        )

    validation = validate(rows, expected_rows=args.expected_rows, expected_conditions=expected_conditions)
    sorted_rows = sorted(rows, key=lambda row: (str(row.get("row_id", "")), expected_conditions.index(str(row.get("condition", "")))))
    generations_out = output_dir / "generations.jsonl"
    write_jsonl(generations_out, sorted_rows)
    metrics = aggregate(sorted_rows)
    (output_dir / "metrics.jsonl").write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "created_at": int(time.time()),
        "status": "pass",
        "output_dir": str(output_dir),
        "source_summaries": source_summaries,
        "validation": validation,
        "metrics": metrics,
        "artifacts": {
            "generations_jsonl": str(generations_out),
            "metrics_jsonl": str(output_dir / "metrics.jsonl"),
            "summary_json": str(output_dir / "summary.json"),
        },
        "execution_boundary": [
            "local-only receiver-generation shard merge",
            "no optimizer, backward, or weight update",
            "no SwanLab run",
            "candidate rows are receiver-generated predictions only",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def validate(rows: list[dict[str, Any]], *, expected_rows: int, expected_conditions: list[str]) -> dict[str, Any]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    duplicates = []
    forbidden_hits = []
    for row in rows:
        key = (str(row.get("row_id", "")), str(row.get("condition", "")))
        if key in by_key:
            duplicates.append({"row_id": key[0], "condition": key[1]})
        by_key[key] = row
        hits = sorted(FORBIDDEN_PAYLOAD_KEYS.intersection(row.keys()))
        if hits:
            forbidden_hits.append({"row_id": key[0], "condition": key[1], "keys": hits})
    row_ids = sorted({key[0] for key in by_key})
    conditions = sorted({key[1] for key in by_key})
    missing = []
    for row_id in row_ids:
        for condition in expected_conditions:
            if (row_id, condition) not in by_key:
                missing.append({"row_id": row_id, "condition": condition})
    status = "pass"
    errors = []
    if len(row_ids) != expected_rows:
        status = "fail"
        errors.append(f"expected {expected_rows} unique row_ids, got {len(row_ids)}")
    if conditions != sorted(expected_conditions):
        status = "fail"
        errors.append(f"conditions mismatch: expected {sorted(expected_conditions)}, got {conditions}")
    if duplicates:
        status = "fail"
        errors.append(f"duplicate row/condition pairs: {len(duplicates)}")
    if missing:
        status = "fail"
        errors.append(f"missing row/condition pairs: {len(missing)}")
    if forbidden_hits:
        status = "fail"
        errors.append(f"forbidden payload keys found: {len(forbidden_hits)}")
    if status != "pass":
        raise ValueError({"errors": errors, "duplicates": duplicates[:5], "missing": missing[:5], "forbidden_hits": forbidden_hits[:5]})
    return {
        "status": status,
        "num_generations": len(rows),
        "num_unique_rows": len(row_ids),
        "num_conditions": len(conditions),
        "conditions": conditions,
        "num_duplicates": len(duplicates),
        "num_missing": len(missing),
        "num_forbidden_payload_hits": len(forbidden_hits),
    }


if __name__ == "__main__":
    main()
