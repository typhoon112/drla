"""Aggregate LOTO learned action-to-halt gate summaries."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


POLICIES = [
    "action",
    "halt_original",
    "causal_always_defer_after_action",
    "causal_oracle_defer_after_action",
    "source_valid_safety_selected_gate",
    "source_valid_cost_selected_gate",
    "source_valid_cost_limited_selected_gate",
    "target_valid_safety_selected_gate",
    "target_valid_cost_selected_gate",
    "target_valid_cost_limited_selected_gate",
    "best_test_gate_by_loss",
    "best_test_gate_under_action_plus_0p10_blocks",
]

WEIGHTED_FIELDS = [
    "accuracy",
    "fixed_final_accuracy",
    "prediction_stability_accuracy",
    "accuracy_drop_vs_final",
    "accuracy_drop_vs_prediction_stability",
    "avg_blocks",
    "fixed_final_avg_blocks",
    "block_saving_vs_final",
    "block_saving_fraction_vs_final",
    "prediction_mismatch_rate_vs_final",
    "prediction_mismatch_rate_vs_prediction_stability",
    "defer_rate",
]

COUNT_FIELDS = [
    "losses_vs_final",
    "losses_vs_prediction_stability",
    "gains_vs_final",
    "gains_vs_prediction_stability",
    "prediction_mismatch_vs_final",
    "prediction_mismatch_vs_prediction_stability",
    "defer_count",
    "rescued_action_losses",
    "introduced_losses_vs_action",
]


@dataclass(frozen=True)
class ActionHaltGateAggregateConfig:
    summary_glob: str = (
        "/data1/luyifei/drla/outputs/cola_action_halt_gate/"
        "official8_full_b64_bs12_loto_*_seed20260526_besteval/summary.json"
    )
    output_dir: str = (
        "/data1/luyifei/drla/outputs/cola_experiment_summaries/"
        "official8_full_b64_bs12_p1_learned_action_halt_gate_loto_seed20260526"
    )


def aggregate_action_halt_gate(config: ActionHaltGateAggregateConfig) -> dict[str, Any]:
    summaries = load_summaries(config.summary_glob)
    if not summaries:
        raise FileNotFoundError(f"no summaries matched {config.summary_glob}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_rows = build_task_rows(summaries)
    aggregate = aggregate_policies(summaries)
    write_csv(output_dir / "task_policy_summary.csv", task_rows)
    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "num_summaries": len(summaries),
        "num_samples": sum(int(item["test_policies"]["action"]["num_samples"]) for item in summaries),
        "aggregate": aggregate,
        "task_policy_summary_csv": str(output_dir / "task_policy_summary.csv"),
        "task_summaries": [
            {
                "task": item["config"]["heldout_task"],
                "summary_path": item["_summary_path"],
                "swanlab_run_id": item.get("swanlab_run_id"),
                "best_step": item.get("best_step"),
                "best_metric": item.get("best_metric"),
                "train_positive_rate": item.get("data_summary", {}).get("train_positive_rate"),
                "valid_positive_rate": item.get("data_summary", {}).get("valid_positive_rate"),
                "test_positive_rate": item.get("data_summary", {}).get("test_positive_rate"),
            }
            for item in summaries
        ],
        "readout": build_readout(aggregate),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def load_summaries(pattern: str) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted(Path("/").glob(pattern.lstrip("/"))):
        item = json.loads(path.read_text(encoding="utf-8"))
        item["_summary_path"] = str(path)
        summaries.append(item)
    summaries.sort(key=lambda item: item["config"]["heldout_task"])
    return summaries


def build_task_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in summaries:
        task = item["config"]["heldout_task"]
        test_policies = materialize_test_policies(item)
        for policy in POLICIES:
            metrics = test_policies.get(policy)
            if metrics:
                rows.append(
                    {
                        "task": task,
                        "policy": policy,
                        "summary_path": item["_summary_path"],
                        "swanlab_run_id": item.get("swanlab_run_id"),
                        **metrics,
                    }
                )
    return rows


def aggregate_policies(summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    aggregate: dict[str, dict[str, Any]] = {}
    total_samples = sum(int(item["test_policies"]["action"]["num_samples"]) for item in summaries)
    for policy in POLICIES:
        present = 0
        weighted = {field: 0.0 for field in WEIGHTED_FIELDS}
        counts = {field: 0.0 for field in COUNT_FIELDS}
        threshold_by_task = {}
        for item in summaries:
            metrics = materialize_test_policies(item).get(policy)
            if not metrics:
                continue
            n = int(metrics["num_samples"])
            present += n
            if "threshold" in metrics:
                threshold_by_task[item["config"]["heldout_task"]] = metrics["threshold"]
            for field in WEIGHTED_FIELDS:
                if field in metrics:
                    weighted[field] += float(metrics[field]) * n
            for field in COUNT_FIELDS:
                if field in metrics:
                    counts[field] += float(metrics[field])
        if not present:
            continue
        aggregate[policy] = {
            "num_samples": present,
            "covers_all_samples": present == total_samples,
            **{field: weighted[field] / present for field in WEIGHTED_FIELDS if weighted[field] != 0.0},
            **{field: counts[field] for field in COUNT_FIELDS if counts[field] != 0.0},
        }
        if threshold_by_task:
            aggregate[policy]["threshold_by_task"] = threshold_by_task
    return aggregate


def materialize_test_policies(item: dict[str, Any]) -> dict[str, Any]:
    """Backfill newer cost-limited selections from legacy sweep CSV artifacts."""
    policies = dict(item["test_policies"])
    summary_dir = Path(item["_summary_path"]).parent
    test_rows = read_csv_rows(summary_dir / "test_threshold_sweep.csv")
    if test_rows:
        maybe_add_cost_limited_policy(
            policies,
            name="source_valid_cost_limited_selected_gate",
            validation_rows=read_csv_rows(summary_dir / "valid_threshold_sweep.csv"),
            test_rows=test_rows,
        )
        maybe_add_cost_limited_policy(
            policies,
            name="target_valid_cost_limited_selected_gate",
            validation_rows=read_csv_rows(summary_dir / "target_valid_threshold_sweep.csv"),
            test_rows=test_rows,
        )
    return policies


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [coerce_csv_row(row) for row in csv.DictReader(f)]


def coerce_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    coerced: dict[str, Any] = {}
    for key, value in row.items():
        if value in {None, ""}:
            coerced[key] = value
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            coerced[key] = value
            continue
        coerced[key] = int(numeric) if numeric.is_integer() else numeric
    return coerced


def maybe_add_cost_limited_policy(
    policies: dict[str, Any],
    *,
    name: str,
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> None:
    if name in policies or not validation_rows or not test_rows:
        return
    action_avg_blocks = estimate_action_avg_blocks(validation_rows)
    selected = select_cost_limited_row(validation_rows, action_avg_blocks=action_avg_blocks)
    policies[name] = matching_threshold_row(test_rows, selected["threshold"])


def estimate_action_avg_blocks(rows: list[dict[str, Any]]) -> float:
    no_defer = min(
        rows,
        key=lambda row: (
            int(float(row.get("defer_count", 0))),
            -float(row.get("threshold", 0.0)),
        ),
    )
    return float(no_defer["avg_blocks"])


def select_cost_limited_row(rows: list[dict[str, Any]], *, action_avg_blocks: float) -> dict[str, Any]:
    eligible = [row for row in rows if float(row["avg_blocks"]) <= action_avg_blocks + 0.10]
    return select_safety_row(eligible or rows)


def select_safety_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        rows,
        key=lambda row: (
            int(float(row["losses_vs_final"])),
            int(float(row["prediction_mismatch_vs_final"])),
            float(row["avg_blocks"]),
            -int(float(row.get("rescued_action_losses", 0))),
            float(row["threshold"]),
        ),
    )


def matching_threshold_row(rows: list[dict[str, Any]], threshold: str | float) -> dict[str, Any]:
    target = float(threshold)
    for row in rows:
        if float(row["threshold"]) == target:
            return row
    raise KeyError(threshold)


def build_readout(aggregate: dict[str, dict[str, Any]]) -> list[str]:
    lines = []
    for policy in [
        "action",
        "halt_original",
        "source_valid_cost_selected_gate",
        "source_valid_cost_limited_selected_gate",
        "source_valid_safety_selected_gate",
        "causal_always_defer_after_action",
        "best_test_gate_under_action_plus_0p10_blocks",
    ]:
        metrics = aggregate.get(policy)
        if not metrics:
            continue
        lines.append(
            f"{policy}: {metrics.get('losses_vs_final', 0):.0f} losses, "
            f"{metrics.get('prediction_mismatch_vs_final', 0):.0f} mismatches, "
            f"{metrics.get('avg_blocks', 0):.3f}/4 blocks."
        )
    return lines


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> ActionHaltGateAggregateConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-glob", default=ActionHaltGateAggregateConfig.summary_glob)
    parser.add_argument("--output-dir", default=ActionHaltGateAggregateConfig.output_dir)
    args = parser.parse_args()
    return ActionHaltGateAggregateConfig(summary_glob=args.summary_glob, output_dir=args.output_dir)


def main() -> None:
    print(json.dumps(aggregate_action_halt_gate(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
