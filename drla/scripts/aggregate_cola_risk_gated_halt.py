"""Aggregate Cola continuation-risk gated halt evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RiskGatedAggregateConfig:
    cross_task_eval_root: str = (
        "/data1/luyifei/drla/outputs/cola_risk_gated_halt/"
        "cross_task_b64_process_no_task_seed20260524"
    )
    same_task_summary_path: str = (
        "/data1/luyifei/drla/outputs/cola_risk_gated_halt/"
        "official8_b64_process_no_task_seed20260524/summary.json"
    )
    output_dir: str = (
        "/data1/luyifei/drla/outputs/cola_experiment_summaries/"
        "official8_b64_risk_gated_halt_20260524"
    )


POLICY_KEYS = [
    "fixed_final",
    "prediction_stability",
    "early_or_stability",
    "calibrated_risk_gated",
    "oracle_prefix_gated",
]

WEIGHTED_FIELDS = [
    "accuracy",
    "avg_blocks",
    "halted_before_final_rate",
    "oracle_match_rate",
    "before_prediction_stability_rate",
]

COUNT_FIELDS = [
    "loss_count_vs_final",
    "gain_count_vs_final",
    "loss_count_vs_prediction_stability",
    "gain_count_vs_prediction_stability",
    "prefix_skip_count",
    "shape_guard_skip_count",
    "fragment_guard_skip_count",
    "uncertainty_guard_skip_count",
    "single_choice_guard_skip_count",
]


def aggregate_risk_gated_halt(config: RiskGatedAggregateConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cross_task_summaries = load_cross_task_summaries(Path(config.cross_task_eval_root))
    if not cross_task_summaries:
        raise FileNotFoundError(f"no risk-gated summary.json files under {config.cross_task_eval_root}")

    same_task_summary = load_optional_summary(config.same_task_summary_path)
    rows = build_task_rows(cross_task_summaries)
    aggregate = aggregate_policies(cross_task_summaries)
    threshold_rows, threshold_frontier = aggregate_threshold_sweep(cross_task_summaries)

    task_csv = output_dir / "cross_task_policy_summary.csv"
    threshold_csv = output_dir / "aggregate_risk_threshold_sweep.csv"
    write_csv(task_csv, rows)
    write_csv(threshold_csv, threshold_rows)

    summary = {
        "created_at": int(time.time()),
        "route": "official8 b64 process_no_task continuation-risk gated halt",
        "config": asdict(config),
        "num_task_summaries": len(cross_task_summaries),
        "num_samples": sum(item["num_samples"] for item in cross_task_summaries),
        "aggregate": aggregate,
        "aggregate_threshold_sweep": {
            "csv": str(threshold_csv),
            "frontier": threshold_frontier,
        },
        "task_csv": str(task_csv),
        "task_rows": rows,
        "same_task_summary": summarize_same_task(same_task_summary),
        "readout": build_readout(aggregate, same_task_summary, rows),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def load_cross_task_summaries(root: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted(root.glob("leave_*_out_eval_*_all/summary.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        item["_summary_path"] = str(path)
        summaries.append(item)
    return summaries


def load_optional_summary(path: str) -> dict[str, Any] | None:
    summary_path = Path(path)
    if not summary_path.exists():
        return None
    item = json.loads(summary_path.read_text(encoding="utf-8"))
    item["_summary_path"] = str(summary_path)
    return item


def build_task_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in summaries:
        row: dict[str, Any] = {
            "task": item["eval_tasks"],
            "num_samples": item["num_samples"],
            "summary_path": item["_summary_path"],
            "swanlab_run_id": item.get("swanlab_run_id"),
            "readiness_threshold": item.get("readiness_threshold"),
            "risk_threshold": item.get("calibrated_risk_gated", {}).get("risk_threshold"),
            "single_choice_guard_scope": item.get("calibrated_risk_gated", {}).get(
                "single_choice_guard_scope"
            ),
        }
        for policy in POLICY_KEYS:
            metrics = item.get(policy, {})
            for field, value in metrics.items():
                row[f"{policy}_{field}"] = value
        row["risk_gated_accuracy_delta_vs_prediction_stability"] = (
            row.get("calibrated_risk_gated_accuracy", 0.0)
            - row.get("prediction_stability_accuracy", 0.0)
        )
        row["risk_gated_block_saving_vs_prediction_stability"] = (
            row.get("prediction_stability_avg_blocks", 0.0)
            - row.get("calibrated_risk_gated_avg_blocks", 0.0)
        )
        rows.append(row)
    return rows


def aggregate_policies(summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    total_samples = sum(int(item["num_samples"]) for item in summaries)
    aggregate: dict[str, dict[str, Any]] = {}
    for policy in POLICY_KEYS:
        present_samples = 0
        weighted = {field: 0.0 for field in WEIGHTED_FIELDS}
        counts = {field: 0.0 for field in COUNT_FIELDS}
        threshold_by_task = {}
        for item in summaries:
            metrics = item.get(policy)
            if not metrics:
                continue
            num_samples = int(item["num_samples"])
            present_samples += num_samples
            threshold = metrics.get("risk_threshold")
            if threshold is not None:
                threshold_by_task[item["eval_tasks"]] = threshold
            for field in WEIGHTED_FIELDS:
                if field in metrics:
                    weighted[field] += float(metrics[field]) * num_samples
            for field in COUNT_FIELDS:
                if field in metrics:
                    counts[field] += float(metrics[field])
        if not present_samples:
            continue
        policy_summary = {
            "num_samples": present_samples,
            "covers_all_samples": present_samples == total_samples,
            **{field: weighted[field] / present_samples for field in WEIGHTED_FIELDS if weighted[field] != 0.0},
            **{field: counts[field] for field in COUNT_FIELDS if counts[field] != 0.0},
        }
        if threshold_by_task:
            policy_summary["threshold_by_task"] = threshold_by_task
        aggregate[policy] = policy_summary

    fixed = aggregate.get("fixed_final", {})
    stability = aggregate.get("prediction_stability", {})
    fixed_acc = float(fixed.get("accuracy", 0.0))
    fixed_blocks = float(fixed.get("avg_blocks", 0.0))
    stability_acc = float(stability.get("accuracy", 0.0))
    stability_blocks = float(stability.get("avg_blocks", 0.0))
    for metrics in aggregate.values():
        metrics["accuracy_delta_vs_fixed_final"] = float(metrics.get("accuracy", 0.0)) - fixed_acc
        metrics["accuracy_delta_vs_prediction_stability"] = (
            float(metrics.get("accuracy", 0.0)) - stability_acc
        )
        metrics["block_saving_vs_fixed_final"] = fixed_blocks - float(metrics.get("avg_blocks", 0.0))
        metrics["block_saving_vs_prediction_stability"] = (
            stability_blocks - float(metrics.get("avg_blocks", 0.0))
        )
        metrics["block_saving_rate_vs_fixed_final"] = (
            metrics["block_saving_vs_fixed_final"] / fixed_blocks if fixed_blocks else 0.0
        )
    return aggregate


def aggregate_threshold_sweep(summaries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_threshold: dict[tuple[str, str, str, str, str], dict[str, float]] = {}
    total_samples = sum(int(item["num_samples"]) for item in summaries)
    stability_accuracy = weighted_policy_metric(summaries, "prediction_stability", "accuracy")
    for item in summaries:
        sweep_path = Path(item["risk_threshold_sweep_csv"])
        with sweep_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                threshold = f"{round(float(row['risk_threshold']), 10):.10g}"
                readiness_threshold = f"{round(float(row.get('readiness_threshold') or 0.0), 10):.10g}"
                entropy_max = row.get("entropy_max") or ""
                top_prob_min = row.get("top_prob_min") or ""
                single_choice_guard_scope = row.get("single_choice_guard_scope") or "fixed"
                key = (threshold, readiness_threshold, entropy_max, top_prob_min, single_choice_guard_scope)
                bucket = by_threshold.setdefault(
                    key,
                    {
                        "num_samples": 0.0,
                        "accuracy": 0.0,
                        "avg_blocks": 0.0,
                        "halted_before_final_rate": 0.0,
                        "before_prediction_stability_rate": 0.0,
                        "prefix_skip_count": 0.0,
                        "shape_guard_skip_count": 0.0,
                        "fragment_guard_skip_count": 0.0,
                        "uncertainty_guard_skip_count": 0.0,
                        "single_choice_guard_skip_count": 0.0,
                        "loss_count_vs_final": 0.0,
                        "gain_count_vs_final": 0.0,
                        "loss_count_vs_prediction_stability": 0.0,
                        "gain_count_vs_prediction_stability": 0.0,
                    },
                )
                num_samples = int(item["num_samples"])
                bucket["num_samples"] += num_samples
                for field in [
                    "accuracy",
                    "avg_blocks",
                    "halted_before_final_rate",
                    "before_prediction_stability_rate",
                ]:
                    bucket[field] += float(row[field]) * num_samples
                bucket["prefix_skip_count"] += float(row["prefix_skip_count"])
                bucket["shape_guard_skip_count"] += float(row.get("shape_guard_skip_count", 0.0))
                bucket["fragment_guard_skip_count"] += float(row.get("fragment_guard_skip_count", 0.0))
                bucket["uncertainty_guard_skip_count"] += float(row.get("uncertainty_guard_skip_count", 0.0))
                bucket["single_choice_guard_skip_count"] += float(
                    row.get("single_choice_guard_skip_count", 0.0)
                )
                bucket["loss_count_vs_final"] += float(row.get("loss_count_vs_final", 0.0))
                bucket["gain_count_vs_final"] += float(row.get("gain_count_vs_final", 0.0))
                bucket["loss_count_vs_prediction_stability"] += float(
                    row.get("loss_count_vs_prediction_stability", 0.0)
                )
                bucket["gain_count_vs_prediction_stability"] += float(
                    row.get("gain_count_vs_prediction_stability", 0.0)
                )

    rows = []
    for key, totals in sorted(by_threshold.items()):
        threshold, readiness_threshold, entropy_max, top_prob_min, single_choice_guard_scope = key
        num_samples = totals["num_samples"]
        if num_samples != total_samples:
            continue
        row = {
            "risk_threshold": float(threshold),
            "readiness_threshold": float(readiness_threshold),
            "entropy_max": entropy_max or None,
            "top_prob_min": top_prob_min or None,
            "single_choice_guard_scope": single_choice_guard_scope,
            "num_samples": int(num_samples),
            "accuracy": totals["accuracy"] / num_samples,
            "avg_blocks": totals["avg_blocks"] / num_samples,
            "halted_before_final_rate": totals["halted_before_final_rate"] / num_samples,
            "before_prediction_stability_rate": totals["before_prediction_stability_rate"] / num_samples,
            "prefix_skip_count": totals["prefix_skip_count"],
            "shape_guard_skip_count": totals["shape_guard_skip_count"],
            "fragment_guard_skip_count": totals["fragment_guard_skip_count"],
            "uncertainty_guard_skip_count": totals["uncertainty_guard_skip_count"],
            "single_choice_guard_skip_count": totals["single_choice_guard_skip_count"],
            "loss_count_vs_final": totals["loss_count_vs_final"],
            "gain_count_vs_final": totals["gain_count_vs_final"],
            "loss_count_vs_prediction_stability": totals["loss_count_vs_prediction_stability"],
            "gain_count_vs_prediction_stability": totals["gain_count_vs_prediction_stability"],
        }
        row["accuracy_drop_vs_prediction_stability"] = stability_accuracy - row["accuracy"]
        rows.append(row)

    frontier = {
        "best_accuracy": max(rows, key=lambda row: (row["accuracy"], -row["avg_blocks"])) if rows else None,
        "best_cost_drop_tolerance_0p00": choose_best_cost(rows, max_drop=0.0),
        "best_cost_drop_tolerance_0p001": choose_best_cost(rows, max_drop=0.001),
        "best_cost_drop_tolerance_0p01": choose_best_cost(rows, max_drop=0.01),
        "best_cost_zero_loss_vs_prediction_stability": choose_best_cost(
            [
                row
                for row in rows
                if row.get("loss_count_vs_prediction_stability", 0.0) == 0.0
            ],
            max_drop=0.0,
        ),
    }
    return rows, frontier


def weighted_policy_metric(summaries: list[dict[str, Any]], policy: str, metric: str) -> float:
    numerator = 0.0
    denominator = 0
    for item in summaries:
        if policy not in item or metric not in item[policy]:
            continue
        num_samples = int(item["num_samples"])
        numerator += float(item[policy][metric]) * num_samples
        denominator += num_samples
    return numerator / denominator if denominator else 0.0


def choose_best_cost(rows: list[dict[str, Any]], max_drop: float) -> dict[str, Any] | None:
    candidates = [row for row in rows if row["accuracy_drop_vs_prediction_stability"] <= max_drop + 1e-12]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (row["avg_blocks"], -row["accuracy"]))


def summarize_same_task(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        "summary_path": summary.get("_summary_path"),
        "swanlab_run_id": summary.get("swanlab_run_id"),
        "num_samples": summary.get("num_samples"),
        "fixed_final": summary.get("fixed_final"),
        "prediction_stability": summary.get("prediction_stability"),
        "early_or_stability": summary.get("early_or_stability"),
        "calibrated_risk_gated": summary.get("calibrated_risk_gated"),
        "oracle_prefix_gated": summary.get("oracle_prefix_gated"),
    }


def build_readout(
    aggregate: dict[str, dict[str, Any]],
    same_task_summary: dict[str, Any] | None,
    task_rows: list[dict[str, Any]],
) -> list[str]:
    lines = []
    risk = aggregate["calibrated_risk_gated"]
    stability = aggregate["prediction_stability"]
    oracle = aggregate["oracle_prefix_gated"]
    early = aggregate["early_or_stability"]
    lines.append(
        "Cross-task risk-gated halt improves over aggressive readiness OR: "
        f"{risk['accuracy']:.4f} accuracy at {risk['avg_blocks']:.3f}/4 blocks, "
        f"versus OR {early['accuracy']:.4f} at {early['avg_blocks']:.3f}/4."
    )
    lines.append(
        "Compared with prediction-stability, risk-gating saves "
        f"{risk['block_saving_vs_prediction_stability']:.3f} blocks/sample but changes accuracy by "
        f"{risk['accuracy_delta_vs_prediction_stability']:.4f}."
    )
    if risk.get("shape_guard_skip_count"):
        lines.append(
            "The contentful-prediction guard blocked "
            f"{risk['shape_guard_skip_count']:.0f} punctuation-only readiness halts across the aggregate."
        )
    lines.append(
        "The remaining gap to the oracle prefix gate is mostly calibration: "
        f"risk-gated {risk['avg_blocks']:.3f}/4 vs oracle {oracle['avg_blocks']:.3f}/4 blocks, "
        f"with oracle accuracy {oracle['accuracy']:.4f}."
    )
    if same_task_summary is not None:
        same_risk = same_task_summary["calibrated_risk_gated"]
        same_stability = same_task_summary["prediction_stability"]
        lines.append(
            "Same-task risk-gating already reaches the prediction-stability accuracy "
            f"({same_risk['accuracy']:.4f}) while reducing average blocks from "
            f"{same_stability['avg_blocks']:.3f} to {same_risk['avg_blocks']:.3f}."
        )
    risky_tasks = [
        row["task"]
        for row in task_rows
        if row["risk_gated_accuracy_delta_vs_prediction_stability"] < 0.0
    ]
    if risky_tasks:
        lines.append(
            "Residual task-level losses remain the safety checks for the next calibration pass: "
            + ", ".join(risky_tasks)
            + "."
        )
    return lines


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> RiskGatedAggregateConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-task-eval-root", default=RiskGatedAggregateConfig.cross_task_eval_root)
    parser.add_argument("--same-task-summary-path", default=RiskGatedAggregateConfig.same_task_summary_path)
    parser.add_argument("--output-dir", default=RiskGatedAggregateConfig.output_dir)
    args = parser.parse_args()
    return RiskGatedAggregateConfig(
        cross_task_eval_root=args.cross_task_eval_root,
        same_task_summary_path=args.same_task_summary_path,
        output_dir=args.output_dir,
    )


def main() -> None:
    summary = aggregate_risk_gated_halt(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
