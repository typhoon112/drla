"""Aggregate leave-one-task-out LatentHaltStudent evaluations.

The aggregate intentionally reports both counts and rates. A single lost
sample on a large task is a boundary event, not automatically evidence of a
systematic failure mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


TASK_ORDER = [
    "lambada",
    "mmlu",
    "obqa",
    "hellaswag",
    "race",
    "siqa",
    "squad",
    "story_cloze",
]

LOW_LOSS_RATE = 0.00025
LOW_MISMATCH_RATE = 0.001
MODERATE_MISMATCH_RATE = 0.01


@dataclass(frozen=True)
class LatentHaltStudentLotoAggregateConfig:
    eval_roots: list[str] = field(default_factory=list)
    output_dir: str = (
        "/data1/luyifei/drla/outputs/cola_experiment_summaries/"
        "official8_full_b64_bs12_latent_halt_student_loto_20260525"
    )
    strict_eval_local_only: bool = True


def aggregate_latent_halt_student_loto(
    config: LatentHaltStudentLotoAggregateConfig,
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = load_summaries([Path(root) for root in config.eval_roots])
    if not summaries:
        raise FileNotFoundError(f"no summary.json files under eval roots: {config.eval_roots}")

    if config.strict_eval_local_only:
        check_eval_local_only(summaries)

    task_rows = [build_task_row(item) for item in summaries]
    task_rows.sort(key=lambda row: (int(row["seed"]), task_sort_key(str(row["task"]))))

    seed_rows = aggregate_by_seed(task_rows)
    per_task_rows = aggregate_by_task(task_rows)
    aggregate = aggregate_rows(task_rows)

    task_csv = output_dir / "task_summary.csv"
    seed_csv = output_dir / "seed_summary.csv"
    per_task_csv = output_dir / "cross_seed_task_summary.csv"
    write_csv(task_csv, task_rows)
    write_csv(seed_csv, seed_rows)
    write_csv(per_task_csv, per_task_rows)

    summary = {
        "created_at": int(time.time()),
        "route": "official8 full b64 bs12 LatentHaltStudent-v1 leave-one-task-out",
        "config": asdict(config),
        "num_eval_summaries": len(summaries),
        "num_samples": int(sum(row["num_samples"] for row in task_rows)),
        "rate_policy": {
            "low_loss_rate_max": LOW_LOSS_RATE,
            "low_mismatch_rate_max": LOW_MISMATCH_RATE,
            "moderate_mismatch_rate_max": MODERATE_MISMATCH_RATE,
            "interpretation": (
                "Loss counts are interpreted with loss_count / num_samples, "
                "mismatch_count / num_samples, and cross-seed recurrence."
            ),
        },
        "aggregate": aggregate,
        "seed_rows": seed_rows,
        "task_rows": task_rows,
        "cross_seed_task_rows": per_task_rows,
        "task_csv": str(task_csv),
        "seed_csv": str(seed_csv),
        "cross_seed_task_csv": str(per_task_csv),
        "readout": build_readout(aggregate, seed_rows, per_task_rows),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def load_summaries(eval_roots: list[Path]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for root in eval_roots:
        for path in sorted(root.glob("leave_*_out_eval_*_all/summary.json")):
            item = json.loads(path.read_text(encoding="utf-8"))
            item["_summary_path"] = str(path)
            item["_eval_root"] = str(root)
            item["_seed"] = infer_seed(root, item)
            summaries.append(item)
    return summaries


def infer_seed(root: Path, item: dict[str, Any]) -> int:
    for text in [
        str(root),
        str(item.get("output_dir", "")),
        str(item.get("checkpoint", "")),
        str(item.get("train_config", {}).get("output_dir", "")),
        str(item.get("train_config", {}).get("experiment_name", "")),
    ]:
        match = re.search(r"seed(\d+)", text)
        if match:
            return int(match.group(1))
    raise ValueError(f"cannot infer seed for {item.get('_summary_path')}")


def check_eval_local_only(summaries: list[dict[str, Any]]) -> None:
    offenders = []
    for item in summaries:
        if item.get("swanlab_mode") != "disabled" or item.get("swanlab_run_id") is not None:
            offenders.append(item["_summary_path"])
    if offenders:
        joined = "\n".join(offenders)
        raise ValueError(f"non-training eval summaries must be local-only:\n{joined}")


def build_task_row(item: dict[str, Any]) -> dict[str, Any]:
    selected = item["selected_eval"]
    final = item["eval_baselines"]["fixed_final"]
    stability = item["eval_baselines"]["prediction_stability"]
    num_samples = int(selected["num_samples"])
    loss_ps = int(selected.get("losses_vs_prediction_stability", 0))
    loss_final = int(selected.get("losses_vs_final", 0))
    mismatch_ps = int(selected.get("prediction_mismatch_vs_prediction_stability", 0))
    mismatch_final = int(selected.get("prediction_mismatch_vs_final", 0))

    loss_rate_ps = rate(loss_ps, num_samples)
    mismatch_rate_ps = rate(mismatch_ps, num_samples)
    row = {
        "seed": int(item["_seed"]),
        "task": item["eval_tasks"],
        "num_samples": num_samples,
        "summary_path": item["_summary_path"],
        "swanlab_mode": item.get("swanlab_mode"),
        "swanlab_run_id": item.get("swanlab_run_id"),
        "selection_note": selected.get("selection_note"),
        "readiness_threshold": selected.get("readiness_threshold"),
        "risk_threshold": selected.get("risk_threshold"),
        "contentful_threshold": selected.get("contentful_threshold"),
        "completion_risk_threshold": selected.get("completion_risk_threshold"),
        "selected_accuracy": float(selected["accuracy"]),
        "fixed_final_accuracy": float(final["accuracy"]),
        "prediction_stability_accuracy": float(stability["accuracy"]),
        "selected_avg_blocks": float(selected["avg_blocks"]),
        "fixed_final_avg_blocks": float(final["avg_blocks"]),
        "prediction_stability_avg_blocks": float(stability["avg_blocks"]),
        "selected_block_saving_vs_final": float(selected["block_saving_vs_final"]),
        "selected_block_saving_vs_prediction_stability": (
            float(stability["avg_blocks"]) - float(selected["avg_blocks"])
        ),
        "selected_accuracy_delta_vs_final": float(selected["accuracy"]) - float(final["accuracy"]),
        "selected_accuracy_delta_vs_prediction_stability": (
            float(selected["accuracy"]) - float(stability["accuracy"])
        ),
        "losses_vs_final": loss_final,
        "losses_vs_prediction_stability": loss_ps,
        "loss_rate_vs_final": rate(loss_final, num_samples),
        "loss_rate_vs_prediction_stability": loss_rate_ps,
        "loss_rate_vs_prediction_stability_wilson95_upper": wilson_upper(loss_ps, num_samples),
        "gains_vs_final": int(selected.get("gains_vs_final", 0)),
        "gains_vs_prediction_stability": int(selected.get("gains_vs_prediction_stability", 0)),
        "prediction_mismatch_vs_final": mismatch_final,
        "prediction_mismatch_vs_prediction_stability": mismatch_ps,
        "prediction_mismatch_rate_vs_final": rate(mismatch_final, num_samples),
        "prediction_mismatch_rate_vs_prediction_stability": mismatch_rate_ps,
        "risk_bucket": classify_risk(loss_rate_ps, mismatch_rate_ps, loss_ps, mismatch_ps),
    }
    return row


def rate(count: int | float, total: int | float) -> float:
    return float(count) / float(total) if total else 0.0


def wilson_upper(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total) / denom
    return min(1.0, center + margin)


def classify_risk(
    loss_rate: float,
    mismatch_rate: float,
    loss_count: int,
    mismatch_count: int,
) -> str:
    if loss_count == 0 and mismatch_count == 0:
        return "zero_loss_zero_mismatch"
    if loss_rate <= LOW_LOSS_RATE and mismatch_rate <= LOW_MISMATCH_RATE:
        return "low_rate_boundary"
    if loss_rate <= LOW_LOSS_RATE and mismatch_rate <= MODERATE_MISMATCH_RATE:
        return "mismatch_watch"
    return "systematic_risk"


def aggregate_by_seed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"seed": int(seed), **aggregate_rows(seed_rows)}
        for seed, seed_rows in sorted(group_by(rows, "seed").items(), key=lambda item: int(item[0]))
    ]


def aggregate_by_task(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for task, task_rows in sorted(group_by(rows, "task").items(), key=lambda item: task_sort_key(str(item[0]))):
        aggregate = aggregate_rows(task_rows)
        loss_rates = [float(row["loss_rate_vs_prediction_stability"]) for row in task_rows]
        mismatch_rates = [
            float(row["prediction_mismatch_rate_vs_prediction_stability"]) for row in task_rows
        ]
        aggregate.update(
            {
                "task": task,
                "num_seeds": len(task_rows),
                "loss_rate_mean_across_seeds": statistics.mean(loss_rates),
                "loss_rate_std_across_seeds": safe_stdev(loss_rates),
                "mismatch_rate_mean_across_seeds": statistics.mean(mismatch_rates),
                "mismatch_rate_std_across_seeds": safe_stdev(mismatch_rates),
                "seeds_with_loss": sum(int(row["losses_vs_prediction_stability"] > 0) for row in task_rows),
                "seeds_with_mismatch": sum(
                    int(row["prediction_mismatch_vs_prediction_stability"] > 0)
                    for row in task_rows
                ),
                "risk_bucket_by_rate": classify_risk(
                    float(aggregate["loss_rate_vs_prediction_stability"]),
                    float(aggregate["prediction_mismatch_rate_vs_prediction_stability"]),
                    int(aggregate["losses_vs_prediction_stability"]),
                    int(aggregate["prediction_mismatch_vs_prediction_stability"]),
                ),
                "risk_buckets_by_seed": {
                    str(row["seed"]): row["risk_bucket"] for row in sorted(task_rows, key=lambda row: row["seed"])
                },
            }
        )
        result.append(aggregate)
    return result


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = int(sum(int(row["num_samples"]) for row in rows))
    weighted_fields = [
        "selected_accuracy",
        "fixed_final_accuracy",
        "prediction_stability_accuracy",
        "selected_avg_blocks",
        "fixed_final_avg_blocks",
        "prediction_stability_avg_blocks",
        "selected_block_saving_vs_final",
        "selected_block_saving_vs_prediction_stability",
    ]
    count_fields = [
        "losses_vs_final",
        "losses_vs_prediction_stability",
        "gains_vs_final",
        "gains_vs_prediction_stability",
        "prediction_mismatch_vs_final",
        "prediction_mismatch_vs_prediction_stability",
    ]
    aggregate: dict[str, Any] = {"num_samples": total}
    for field_name in weighted_fields:
        aggregate[field_name] = weighted_mean(rows, field_name, total)
    for field_name in count_fields:
        aggregate[field_name] = int(sum(int(row[field_name]) for row in rows))

    aggregate["selected_accuracy_delta_vs_final"] = (
        float(aggregate["selected_accuracy"]) - float(aggregate["fixed_final_accuracy"])
    )
    aggregate["selected_accuracy_delta_vs_prediction_stability"] = (
        float(aggregate["selected_accuracy"]) - float(aggregate["prediction_stability_accuracy"])
    )
    aggregate["loss_rate_vs_final"] = rate(aggregate["losses_vs_final"], total)
    aggregate["loss_rate_vs_prediction_stability"] = rate(
        aggregate["losses_vs_prediction_stability"],
        total,
    )
    aggregate["loss_rate_vs_prediction_stability_wilson95_upper"] = wilson_upper(
        aggregate["losses_vs_prediction_stability"],
        total,
    )
    aggregate["prediction_mismatch_rate_vs_final"] = rate(
        aggregate["prediction_mismatch_vs_final"],
        total,
    )
    aggregate["prediction_mismatch_rate_vs_prediction_stability"] = rate(
        aggregate["prediction_mismatch_vs_prediction_stability"],
        total,
    )
    aggregate["risk_bucket_by_rate"] = classify_risk(
        aggregate["loss_rate_vs_prediction_stability"],
        aggregate["prediction_mismatch_rate_vs_prediction_stability"],
        aggregate["losses_vs_prediction_stability"],
        aggregate["prediction_mismatch_vs_prediction_stability"],
    )
    return aggregate


def weighted_mean(rows: list[dict[str, Any]], field_name: str, total: int) -> float:
    if total <= 0:
        return 0.0
    return sum(float(row[field_name]) * int(row["num_samples"]) for row in rows) / total


def safe_stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def group_by(rows: list[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(row)
    return grouped


def task_sort_key(task: str) -> int:
    try:
        return TASK_ORDER.index(task)
    except ValueError:
        return len(TASK_ORDER)


def build_readout(
    aggregate: dict[str, Any],
    seed_rows: list[dict[str, Any]],
    per_task_rows: list[dict[str, Any]],
) -> list[str]:
    lines = [
        (
            "Overall selected policy: "
            f"{aggregate['selected_accuracy']:.4f} accuracy, "
            f"{aggregate['selected_avg_blocks']:.3f}/4 blocks, "
            f"{aggregate['losses_vs_prediction_stability']} losses "
            f"({aggregate['loss_rate_vs_prediction_stability']:.4%}) and "
            f"{aggregate['prediction_mismatch_vs_prediction_stability']} text mismatches "
            f"({aggregate['prediction_mismatch_rate_vs_prediction_stability']:.4%}) "
            "against prediction-stability."
        )
    ]
    for row in seed_rows:
        lines.append(
            f"Seed {row['seed']}: {row['losses_vs_prediction_stability']} losses "
            f"({row['loss_rate_vs_prediction_stability']:.4%}), "
            f"{row['prediction_mismatch_vs_prediction_stability']} mismatches "
            f"({row['prediction_mismatch_rate_vs_prediction_stability']:.4%}), "
            f"{row['selected_avg_blocks']:.3f}/4 blocks."
        )
    risky = [
        row
        for row in per_task_rows
        if row["risk_bucket_by_rate"] in {"mismatch_watch", "systematic_risk"}
    ]
    if risky:
        lines.append(
            "Tasks needing caution by rate: "
            + ", ".join(
                f"{row['task']}={row['risk_bucket_by_rate']} "
                f"(loss {row['loss_rate_vs_prediction_stability']:.4%}, "
                f"mismatch {row['prediction_mismatch_rate_vs_prediction_stability']:.4%})"
                for row in risky
            )
            + "."
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


def parse_args() -> LatentHaltStudentLotoAggregateConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", action="append", required=True)
    parser.add_argument(
        "--output-dir",
        default=LatentHaltStudentLotoAggregateConfig.output_dir,
    )
    parser.add_argument(
        "--allow-cloud-eval-summary",
        action="store_true",
        help="Allow historical eval summaries that were uploaded to SwanLab.",
    )
    args = parser.parse_args()
    return LatentHaltStudentLotoAggregateConfig(
        eval_roots=args.eval_root,
        output_dir=args.output_dir,
        strict_eval_local_only=not args.allow_cloud_eval_summary,
    )


def main() -> None:
    summary = aggregate_latent_halt_student_loto(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
