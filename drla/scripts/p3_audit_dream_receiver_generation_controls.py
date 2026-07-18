"""Audit P3 Dream receiver generation controls with paired row-level metrics.

This local-only auditor reads one or more receiver generation run directories
and recomputes condition means, paired matched-vs-control deltas, bootstrap CIs,
and correct-row overlaps. It does not train, call models, or create SwanLab runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_receiver_control_audits/"
    "dream_receiver_generation_control_audit_20260617"
)


def main() -> None:
    summary = run_audit(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, help="Receiver eval run directory; may be repeated.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hard-controls", default="no_message,zero,shuffled_row")
    parser.add_argument("--diagnostic-controls", default="agent_swap")
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260617)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    hard_controls = parse_list(args.hard_controls)
    diagnostic_controls = parse_list(args.diagnostic_controls)
    run_summaries: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []

    for run_dir_text in args.run_dir:
        run_dir = Path(run_dir_text)
        rows = read_jsonl(run_dir / "generations.jsonl")
        run_name = run_dir.name
        canonical_rows = [canonicalize_record(row) for row in rows]
        matched_condition = find_condition(canonical_rows, "matched")
        if matched_condition is None:
            raise ValueError(f"no matched condition found in {run_dir}")
        available = sorted({row["canonical_condition"] for row in canonical_rows})
        by_condition = group_by_condition(canonical_rows)
        by_row = group_by_row(canonical_rows)

        run_condition_rows = [
            summarize_condition(run_name, condition, condition_items)
            for condition, condition_items in sorted(by_condition.items())
        ]
        condition_rows.extend(run_condition_rows)

        controls = [control for control in [*hard_controls, *diagnostic_controls] if control in by_condition]
        run_paired_rows = [
            paired_comparison(
                run_name=run_name,
                matched_condition=matched_condition,
                control_condition=control,
                rows_by_id=by_row,
                bootstrap_iters=args.bootstrap_iters,
                bootstrap_seed=args.bootstrap_seed + index,
                control_role="hard" if control in hard_controls else "diagnostic",
            )
            for index, control in enumerate(controls)
        ]
        paired_rows.extend(run_paired_rows)

        matched_correct = correct_rows(by_condition.get(matched_condition, []))
        run_overlap_rows = [
            overlap_summary(run_name, matched_condition, control, matched_correct, correct_rows(by_condition.get(control, [])))
            for control in controls
        ]
        overlap_rows.extend(run_overlap_rows)

        hard_results = [row for row in run_paired_rows if row["control_role"] == "hard"]
        hard_gate_pass = bool(hard_results) and all(
            float(row["primary_delta_mean"]) > 0.0 and float(row["primary_delta_ci_lower"]) > 0.0
            for row in hard_results
        )
        run_summaries.append(
            {
                "run_name": run_name,
                "run_dir": str(run_dir),
                "num_generations": len(canonical_rows),
                "num_unique_rows": len(by_row),
                "available_conditions": available,
                "matched_condition": matched_condition,
                "hard_controls": hard_controls,
                "diagnostic_controls": diagnostic_controls,
                "hard_gate_pass": hard_gate_pass,
                "condition_metrics": run_condition_rows,
                "paired_comparisons": run_paired_rows,
                "overlap_metrics": run_overlap_rows,
            }
        )

    condition_csv = output_dir / "condition_metrics.csv"
    paired_csv = output_dir / "paired_comparisons.csv"
    overlap_csv = output_dir / "row_overlap.csv"
    write_csv(condition_csv, condition_rows)
    write_csv(paired_csv, paired_rows)
    write_csv(overlap_csv, overlap_rows)
    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as handle:
        for item in run_summaries:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "created_at": int(time.time()),
        "status": "pass",
        "output_dir": str(output_dir),
        "num_runs": len(run_summaries),
        "hard_controls": hard_controls,
        "diagnostic_controls": diagnostic_controls,
        "bootstrap_iters": args.bootstrap_iters,
        "run_summaries": run_summaries,
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "metrics_jsonl": str(metrics_path),
            "condition_metrics_csv": str(condition_csv),
            "paired_comparisons_csv": str(paired_csv),
            "row_overlap_csv": str(overlap_csv),
        },
        "execution_boundary": [
            "local-only P3 Dream receiver generation-control audit",
            "no model loading or generation",
            "no optimizer, backward, or weight update",
            "no SwanLab run",
            "gold/scorer outputs are read only from existing offline eval records",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def canonicalize_record(row: dict[str, Any]) -> dict[str, Any]:
    condition = str(row.get("condition", ""))
    canonical = canonical_condition(condition)
    return {
        **row,
        "canonical_condition": canonical,
        "primary_score": float(row.get("primary_score", 0.0) or 0.0),
        "token_f1": float(row.get("token_f1", 0.0) or 0.0),
        "exact_match": float(row.get("exact_match", 0.0) or 0.0),
        "row_id": str(row.get("row_id", "")),
    }


def canonical_condition(condition: str) -> str:
    if condition == "no_message" or condition.endswith("_no_message"):
        return "no_message"
    if "agent_swap" in condition:
        return "agent_swap"
    if "shuffled" in condition:
        return "shuffled_row"
    if "zero" in condition:
        return "zero"
    if "matched" in condition:
        return "matched"
    return condition


def find_condition(rows: list[dict[str, Any]], canonical: str) -> str | None:
    for row in rows:
        if row["canonical_condition"] == canonical:
            return canonical
    return None


def group_by_condition(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["canonical_condition"], []).append(row)
    return grouped


def group_by_row(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        row_id = row["row_id"]
        condition = row["canonical_condition"]
        if condition in grouped.setdefault(row_id, {}):
            raise ValueError(f"duplicate row/condition pair: row_id={row_id}, condition={condition}")
        grouped[row_id][condition] = row
    return grouped


def summarize_condition(run_name: str, condition: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "condition": condition,
        "num_rows": len(rows),
        "primary_score_mean": mean([row["primary_score"] for row in rows]),
        "token_f1_mean": mean([row["token_f1"] for row in rows]),
        "exact_match_mean": mean([row["exact_match"] for row in rows]),
        "num_primary_correct": len(correct_rows(rows)),
    }


def paired_comparison(
    *,
    run_name: str,
    matched_condition: str,
    control_condition: str,
    rows_by_id: dict[str, dict[str, dict[str, Any]]],
    bootstrap_iters: int,
    bootstrap_seed: int,
    control_role: str,
) -> dict[str, Any]:
    primary_deltas: list[float] = []
    token_deltas: list[float] = []
    wins = losses = ties = 0
    for row_conditions in rows_by_id.values():
        matched = row_conditions.get(matched_condition)
        control = row_conditions.get(control_condition)
        if matched is None or control is None:
            continue
        primary_delta = matched["primary_score"] - control["primary_score"]
        token_delta = matched["token_f1"] - control["token_f1"]
        primary_deltas.append(primary_delta)
        token_deltas.append(token_delta)
        if primary_delta > 0:
            wins += 1
        elif primary_delta < 0:
            losses += 1
        else:
            ties += 1
    ci = bootstrap_ci(primary_deltas, bootstrap_iters=bootstrap_iters, seed=bootstrap_seed)
    token_ci = bootstrap_ci(token_deltas, bootstrap_iters=bootstrap_iters, seed=bootstrap_seed + 7919)
    return {
        "run_name": run_name,
        "comparison": f"{matched_condition}_minus_{control_condition}",
        "control_condition": control_condition,
        "control_role": control_role,
        "num_paired": len(primary_deltas),
        "primary_delta_mean": mean(primary_deltas),
        "primary_delta_ci_lower": ci["lower"],
        "primary_delta_ci_upper": ci["upper"],
        "token_f1_delta_mean": mean(token_deltas),
        "token_f1_delta_ci_lower": token_ci["lower"],
        "token_f1_delta_ci_upper": token_ci["upper"],
        "primary_win_count": wins,
        "primary_loss_count": losses,
        "primary_tie_count": ties,
    }


def overlap_summary(
    run_name: str,
    matched_condition: str,
    control_condition: str,
    matched_correct: set[str],
    control_correct: set[str],
) -> dict[str, Any]:
    overlap = matched_correct & control_correct
    return {
        "run_name": run_name,
        "comparison": f"{matched_condition}_vs_{control_condition}",
        "control_condition": control_condition,
        "matched_correct_count": len(matched_correct),
        "control_correct_count": len(control_correct),
        "overlap_correct_count": len(overlap),
        "matched_unique_correct_count": len(matched_correct - control_correct),
        "control_unique_correct_count": len(control_correct - matched_correct),
        "overlap_correct_rows": "|".join(sorted(overlap)[:20]),
        "matched_unique_correct_rows": "|".join(sorted(matched_correct - control_correct)[:20]),
    }


def correct_rows(rows: list[dict[str, Any]]) -> set[str]:
    return {row["row_id"] for row in rows if float(row.get("primary_score", 0.0)) > 0.0}


def bootstrap_ci(values: list[float], *, bootstrap_iters: int, seed: int) -> dict[str, float]:
    if not values:
        return {"lower": 0.0, "upper": 0.0}
    if bootstrap_iters <= 0:
        value = mean(values)
        return {"lower": value, "upper": value}
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(bootstrap_iters):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lower_index = int(0.025 * (len(means) - 1))
    upper_index = int(0.975 * (len(means) - 1))
    return {"lower": means[lower_index], "upper": means[upper_index]}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
