"""Aggregate sample-level Cola halt decision diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HaltDecisionAggregateConfig:
    analysis_root: str = "/data1/luyifei/drla/outputs/cola_halt_decision_analysis/cross_task_b64_guarded_20260524"
    output_dir: str = "/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_b64_halt_decision_analysis_20260524"


def aggregate_halt_decision_analysis(config: HaltDecisionAggregateConfig) -> dict[str, Any]:
    analysis_root = Path(config.analysis_root)
    summaries = load_summaries(analysis_root)
    if not summaries:
        raise FileNotFoundError(f"no decision analysis summaries under {analysis_root}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_rows, aggregate_policy_summary = aggregate_policies(summaries)
    bin_rows = aggregate_bins(summaries)

    policy_csv = output_dir / "policy_comparison.csv"
    bins_csv = output_dir / "readiness_bins.csv"
    write_csv(policy_csv, policy_rows)
    write_csv(bins_csv, bin_rows)

    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "num_task_summaries": len(summaries),
        "num_samples": sum(item["num_samples"] for item in summaries),
        "aggregate_policy_summary": aggregate_policy_summary,
        "policy_comparison_csv": str(policy_csv),
        "readiness_bins_csv": str(bins_csv),
        "task_summaries": [
            {
                "task": item["eval_tasks"],
                "num_samples": item["num_samples"],
                "summary_path": item["_summary_path"],
                "policy_comparison_csv": item["policy_comparison_csv"],
                "readiness_bins_csv": item["readiness_bins_csv"],
                "halt_decisions_jsonl": item["halt_decisions_jsonl"],
            }
            for item in summaries
        ],
        "readout": build_readout(aggregate_policy_summary),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def load_summaries(root: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted(root.glob("leave_*_out_eval_*_all/summary.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        item["_summary_path"] = str(path)
        summaries.append(item)
    return summaries


def aggregate_policies(summaries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    policy_names = sorted({name for item in summaries for name in item["policy_summary"]})
    policy_rows: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, Any]] = {}
    total_samples = sum(item["num_samples"] for item in summaries)

    weighted_metric_names = [
        "accuracy",
        "avg_blocks",
        "avg_saved_blocks_vs_final",
        "block_saving_vs_final",
        "before_prediction_stability_rate",
    ]
    count_metric_names = [
        "lost_final_correct_count",
        "gained_over_final_count",
        "pre_stability_loss_count",
        "prefix_final_correct_loss_count",
        "pre_stability_prefix_loss_count",
    ]
    for name in policy_names:
        threshold_by_task = {}
        totals = {metric: 0.0 for metric in weighted_metric_names + count_metric_names}
        present_samples = 0
        for item in summaries:
            metrics = item["policy_summary"].get(name)
            if not metrics:
                continue
            task = item["eval_tasks"]
            n = item["num_samples"]
            present_samples += n
            threshold_by_task[task] = metrics.get("threshold")
            for metric in weighted_metric_names:
                totals[metric] += float(metrics[metric]) * n
            for metric in count_metric_names:
                totals[metric] += float(metrics[metric])
            policy_rows.append(
                {
                    "task": task,
                    "policy": name,
                    "num_samples": n,
                    **metrics,
                }
            )
        if present_samples:
            aggregate[name] = {
                "num_samples": present_samples,
                "threshold_by_task": threshold_by_task,
                **{metric: totals[metric] / present_samples for metric in weighted_metric_names},
                **{metric: totals[metric] for metric in count_metric_names},
                "covers_all_samples": present_samples == total_samples,
            }
    return policy_rows, aggregate


def aggregate_bins(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bucket: dict[tuple[str, str], dict[str, float]] = {}
    rate_fields = [
        "mean_readiness_prob",
        "current_correct_rate",
        "final_correct_rate",
        "oracle_ready_rate",
        "future_gain_correct_rate",
    ]
    for item in summaries:
        with Path(item["readiness_bins_csv"]).open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                key = (row["scope"], row["prob_bin"])
                count = int(row["count"])
                target = bucket.setdefault(key, {"count": 0.0, **{field: 0.0 for field in rate_fields}})
                target["count"] += count
                for field in rate_fields:
                    target[field] += float(row[field]) * count
    result = []
    for (scope, prob_bin), values in sorted(bucket.items()):
        count = max(values["count"], 1.0)
        result.append(
            {
                "scope": scope,
                "prob_bin": prob_bin,
                "count": int(values["count"]),
                **{field: values[field] / count for field in rate_fields},
            }
        )
    return result


def build_readout(aggregate: dict[str, dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    adaptive = aggregate.get("calibrated_adaptive")
    early_or = aggregate.get("calibrated_early_or_stability")
    guarded = aggregate.get("calibrated_stability_guarded")
    stability = aggregate.get("prediction_stability")
    conservative_or = aggregate.get("early_or_stability_t0p75")
    if stability:
        lines.append(
            "Prediction-stability is the conservative baseline: "
            f"{stability['accuracy']:.4f} accuracy at {stability['avg_blocks']:.3f}/4 blocks."
        )
    if guarded:
        lines.append(
            "The true stability-gated guard matches prediction-stability: "
            f"{guarded['accuracy']:.4f} accuracy, {guarded['avg_blocks']:.3f}/4 blocks, "
            f"{guarded['lost_final_correct_count']:.0f} final-correct losses."
        )
    if adaptive:
        lines.append(
            "Calibrated adaptive thresholding saves more blocks but loses final-correct samples: "
            f"{adaptive['accuracy']:.4f} accuracy, {adaptive['avg_blocks']:.3f}/4 blocks, "
            f"{adaptive['lost_final_correct_count']:.0f} losses; "
            f"{adaptive['pre_stability_loss_count']:.0f} occur before prediction stability."
        )
    if early_or:
        lines.append(
            "Aggressive readiness OR stability has the same failure mode: "
            f"{early_or['accuracy']:.4f} accuracy, {early_or['avg_blocks']:.3f}/4 blocks, "
            f"{early_or['pre_stability_loss_count']:.0f} pre-stability losses, "
            f"{early_or['prefix_final_correct_loss_count']:.0f} prefix losses."
        )
    if conservative_or:
        lines.append(
            "A conservative OR threshold 0.75 preserves accuracy with almost no extra saving: "
            f"{conservative_or['accuracy']:.4f} accuracy at {conservative_or['avg_blocks']:.3f}/4 blocks."
        )
    return lines


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> HaltDecisionAggregateConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", default=HaltDecisionAggregateConfig.analysis_root)
    parser.add_argument("--output-dir", default=HaltDecisionAggregateConfig.output_dir)
    args = parser.parse_args()
    return HaltDecisionAggregateConfig(
        analysis_root=args.analysis_root,
        output_dir=args.output_dir,
    )


def main() -> None:
    summary = aggregate_halt_decision_analysis(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
