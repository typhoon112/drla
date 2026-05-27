"""Aggregate LatentHaltStudent leave-one-task-out evals with calibration subseeds.

This is a local-only analysis utility.  It combines existing evaluation
summaries and never launches training or SwanLab logging.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


TASK_ORDER = ["lambada", "mmlu", "obqa", "hellaswag", "race", "siqa", "squad", "story_cloze"]


@dataclass(frozen=True)
class SubseedLotoAggregateConfig:
    eval_roots: list[str] = field(default_factory=list)
    output_dir: str = (
        "/data1/luyifei/drla/outputs/cola_experiment_summaries/"
        "official8_full_b64_bs12_latent_halt_student_subseed_loto"
    )
    strict_eval_local_only: bool = True


def main() -> None:
    summary = aggregate_subseed_loto(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def aggregate_subseed_loto(config: SubseedLotoAggregateConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = load_summaries([Path(root) for root in config.eval_roots])
    if not summaries:
        raise FileNotFoundError(f"no summary.json files found under {config.eval_roots}")
    if config.strict_eval_local_only:
        offenders = [
            item["_summary_path"]
            for item in summaries
            if item.get("swanlab_mode") != "disabled" or item.get("swanlab_run_id") is not None
        ]
        if offenders:
            raise ValueError("eval summaries must be local-only:\n" + "\n".join(offenders))

    rows = [build_row(item) for item in summaries]
    rows.sort(key=lambda row: (row["seed"], row["subseed"], task_sort_key(row["task"])))
    seed_rows = aggregate_groups(rows, "seed")
    subseed_rows = aggregate_groups(rows, "subseed")
    task_rows = aggregate_groups(rows, "task")
    seed_task_rows = aggregate_groups(rows, "seed_task", key_fn=lambda row: f"{row['seed']}::{row['task']}")
    aggregate = aggregate_rows(rows)

    write_csv(output_dir / "eval_summary_rows.csv", rows)
    write_csv(output_dir / "seed_summary.csv", seed_rows)
    write_csv(output_dir / "subseed_summary.csv", subseed_rows)
    write_csv(output_dir / "task_summary.csv", task_rows)
    write_csv(output_dir / "seed_task_summary.csv", seed_task_rows)

    summary = {
        "created_at": int(time.time()),
        "route": "official8 full b64 bs12 LatentHaltStudent-v1 LOTO target-calibration subseed aggregate",
        "config": asdict(config),
        "num_eval_summaries": len(rows),
        "num_samples_repeated": int(sum(row["num_samples"] for row in rows)),
        "aggregate": aggregate,
        "seed_rows": seed_rows,
        "subseed_rows": subseed_rows,
        "task_rows": task_rows,
        "seed_task_rows": seed_task_rows,
        "artifacts": {
            "eval_summary_rows_csv": str(output_dir / "eval_summary_rows.csv"),
            "seed_summary_csv": str(output_dir / "seed_summary.csv"),
            "subseed_summary_csv": str(output_dir / "subseed_summary.csv"),
            "task_summary_csv": str(output_dir / "task_summary.csv"),
            "seed_task_summary_csv": str(output_dir / "seed_task_summary.csv"),
        },
        "readout": build_readout(aggregate, task_rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> SubseedLotoAggregateConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-legacy-swanlab-eval", action="store_true")
    args = parser.parse_args()
    return SubseedLotoAggregateConfig(
        eval_roots=args.eval_root,
        output_dir=args.output_dir,
        strict_eval_local_only=not args.allow_legacy_swanlab_eval,
    )


def load_summaries(eval_roots: list[Path]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    patterns = [
        "subseed*/leave_*_out_eval_*_test/summary.json",
        "subseed*/leave_*_out_eval_*_all/summary.json",
        "leave_*_out_eval_*_test/summary.json",
        "leave_*_out_eval_*_all/summary.json",
    ]
    for root in eval_roots:
        for pattern in patterns:
            for path in sorted(root.glob(pattern)):
                item = json.loads(path.read_text(encoding="utf-8"))
                item["_summary_path"] = str(path)
                item["_eval_root"] = str(root)
                item["_seed"] = infer_seed(root, item)
                item["_subseed"] = infer_subseed(path, item)
                summaries.append(item)
    return summaries


def infer_seed(root: Path, item: dict[str, Any]) -> int:
    texts = [
        str(root),
        str(item.get("output_dir", "")),
        str(item.get("checkpoint", "")),
        str(item.get("train_config", {}).get("output_dir", "")),
        str(item.get("train_config", {}).get("experiment_name", "")),
    ]
    for text in texts:
        match = re.search(r"seed(\d+)", text)
        if match:
            return int(match.group(1))
    raise ValueError(f"cannot infer seed for {item.get('_summary_path')}")


def infer_subseed(path: Path, item: dict[str, Any]) -> str:
    for part in path.parts:
        if part.startswith("subseed"):
            return part.replace("subseed", "")
    policy = item.get("calibration_policy", {})
    return str(policy.get("calibration_subsample_seed", "none"))


def build_row(item: dict[str, Any]) -> dict[str, Any]:
    selected = item["selected_eval"]
    valid_selected = item.get("selected_valid", {})
    final = item["eval_baselines"]["fixed_final"]
    stability = item["eval_baselines"]["prediction_stability"]
    num_samples = int(selected["num_samples"])
    task = str(item["eval_tasks"])
    row = {
        "seed": int(item["_seed"]),
        "subseed": str(item["_subseed"]),
        "task": task,
        "num_samples": num_samples,
        "summary_path": item["_summary_path"],
        "swanlab_mode": item.get("swanlab_mode"),
        "split_seed": item.get("split_seed"),
        "selection_note": selected.get("selection_note"),
        "readiness_threshold": selected.get("readiness_threshold"),
        "risk_threshold": selected.get("risk_threshold"),
        "completion_risk_threshold": selected.get("completion_risk_threshold"),
        "contentful_threshold": selected.get("contentful_threshold"),
        "answer_identity_stability_threshold": selected.get("answer_identity_stability_threshold"),
        "selected_accuracy": float(selected["accuracy"]),
        "fixed_final_accuracy": float(final["accuracy"]),
        "prediction_stability_accuracy": float(stability["accuracy"]),
        "selected_avg_blocks": float(selected["avg_blocks"]),
        "fixed_final_avg_blocks": float(final["avg_blocks"]),
        "prediction_stability_avg_blocks": float(stability["avg_blocks"]),
        "losses_vs_final": int(selected.get("losses_vs_final", 0)),
        "losses_vs_prediction_stability": int(selected.get("losses_vs_prediction_stability", 0)),
        "gains_vs_final": int(selected.get("gains_vs_final", 0)),
        "gains_vs_prediction_stability": int(selected.get("gains_vs_prediction_stability", 0)),
        "mismatches_vs_final": int(selected.get("prediction_mismatch_vs_final", 0)),
        "mismatches_vs_prediction_stability": int(
            selected.get("prediction_mismatch_vs_prediction_stability", 0)
        ),
        "calibration_loss_risk_target": valid_selected.get("calibration_loss_risk_target"),
        "calibration_mismatch_risk_target": valid_selected.get("calibration_mismatch_risk_target"),
        "calibration_loss_risk_satisfied": valid_selected.get("calibration_loss_risk_satisfied"),
        "calibration_mismatch_risk_satisfied": valid_selected.get("calibration_mismatch_risk_satisfied"),
        "calibration_loss_upper_max": valid_selected.get("loss_upper_max"),
        "calibration_mismatch_upper_max": valid_selected.get("mismatch_upper_max"),
        "eval_loss_upper_max": selected.get("loss_upper_max"),
        "eval_mismatch_upper_max": selected.get("mismatch_upper_max"),
    }
    row["loss_rate_vs_prediction_stability"] = rate(row["losses_vs_prediction_stability"], num_samples)
    row["mismatch_rate_vs_prediction_stability"] = rate(
        row["mismatches_vs_prediction_stability"],
        num_samples,
    )
    row["block_saving_vs_prediction_stability"] = (
        row["prediction_stability_avg_blocks"] - row["selected_avg_blocks"]
    )
    return row


def aggregate_groups(
    rows: list[dict[str, Any]],
    group_name: str,
    key_fn: Any | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(key_fn(row) if key_fn else row[group_name])
        grouped.setdefault(key, []).append(row)
    output = []
    for key, items in grouped.items():
        aggregate = aggregate_rows(items)
        aggregate[group_name] = key
        if group_name == "task":
            aggregate["task_sort"] = task_sort_key(key)
        output.append(aggregate)
    return sorted(output, key=group_sort_key)


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(row["num_samples"]) for row in rows)
    if total <= 0:
        raise ValueError("cannot aggregate zero samples")
    loss_ps = sum(int(row["losses_vs_prediction_stability"]) for row in rows)
    mismatch_ps = sum(int(row["mismatches_vs_prediction_stability"]) for row in rows)
    loss_final = sum(int(row["losses_vs_final"]) for row in rows)
    mismatch_final = sum(int(row["mismatches_vs_final"]) for row in rows)
    return {
        "num_eval_summaries": len(rows),
        "num_samples_repeated": int(total),
        "accuracy_micro": weighted_mean(rows, "selected_accuracy", total),
        "fixed_accuracy_micro": weighted_mean(rows, "fixed_final_accuracy", total),
        "prediction_stability_accuracy_micro": weighted_mean(rows, "prediction_stability_accuracy", total),
        "avg_blocks_micro": weighted_mean(rows, "selected_avg_blocks", total),
        "fixed_avg_blocks_micro": weighted_mean(rows, "fixed_final_avg_blocks", total),
        "prediction_stability_avg_blocks_micro": weighted_mean(
            rows,
            "prediction_stability_avg_blocks",
            total,
        ),
        "block_saving_vs_prediction_stability_micro": weighted_mean(
            rows,
            "block_saving_vs_prediction_stability",
            total,
        ),
        "losses_vs_prediction_stability_total": int(loss_ps),
        "losses_vs_prediction_stability_rate": rate(loss_ps, total),
        "losses_vs_final_total": int(loss_final),
        "losses_vs_final_rate": rate(loss_final, total),
        "mismatches_vs_prediction_stability_total": int(mismatch_ps),
        "mismatches_vs_prediction_stability_rate": rate(mismatch_ps, total),
        "mismatches_vs_final_total": int(mismatch_final),
        "mismatches_vs_final_rate": rate(mismatch_final, total),
        "selected_avg_blocks_mean_unweighted": statistics.mean(float(row["selected_avg_blocks"]) for row in rows),
    }


def weighted_mean(rows: list[dict[str, Any]], key: str, total: int) -> float:
    return float(sum(float(row[key]) * int(row["num_samples"]) for row in rows) / total)


def rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def task_sort_key(task: str) -> int:
    return TASK_ORDER.index(task) if task in TASK_ORDER else len(TASK_ORDER)


def group_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    if "seed" in row:
        return (0, int(row["seed"]))
    if "subseed" in row:
        return (1, str(row["subseed"]))
    if "task_sort" in row:
        return (2, int(row["task_sort"]))
    if "seed_task" in row:
        seed, task = str(row["seed_task"]).split("::", 1)
        return (3, int(seed), task_sort_key(task))
    return (9, "")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_readout(aggregate: dict[str, Any], task_rows: list[dict[str, Any]]) -> dict[str, Any]:
    loss_tasks = [
        {
            "task": row["task"],
            "losses": row["losses_vs_prediction_stability_total"],
            "mismatches": row["mismatches_vs_prediction_stability_total"],
            "avg_blocks": row["avg_blocks_micro"],
        }
        for row in task_rows
        if int(row["losses_vs_prediction_stability_total"]) > 0
    ]
    return {
        "headline": (
            f"{aggregate['losses_vs_prediction_stability_total']} losses, "
            f"{aggregate['mismatches_vs_prediction_stability_total']} mismatches, "
            f"{aggregate['avg_blocks_micro']:.3f}/4 blocks"
        ),
        "loss_tasks": loss_tasks,
    }


if __name__ == "__main__":
    main()
