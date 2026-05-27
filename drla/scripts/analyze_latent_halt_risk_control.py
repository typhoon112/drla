"""Risk-control audit for LatentHaltStudent threshold sweeps.

This script is local-only: it does not train a model and must not create
SwanLab runs. It reads existing leave-one-task-out evaluation summaries,
selects thresholds on validation sweeps using Wilson upper bounds for observed
loss/mismatch rates, then reports the held-out evaluation trade-off.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


THRESHOLD_SUFFIX = "_threshold"


@dataclass(frozen=True)
class LatentHaltRiskControlConfig:
    eval_root_globs: list[str]
    output_dir: str = (
        "/data1/luyifei/drla/outputs/cola_experiment_summaries/"
        "official8_full_b64_bs12_p1_latent_halt_risk_control_20260527"
    )
    loss_risk_targets: str = "0.00025,0.0005,0.001,0.002"
    mismatch_risk_targets: str = "0.001,0.005,0.01"
    z_value: float = 1.96
    allow_legacy_swanlab_eval: bool = False


def analyze_latent_halt_risk_control(config: LatentHaltRiskControlConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = load_summaries(config.eval_root_globs, allow_legacy_swanlab_eval=config.allow_legacy_swanlab_eval)
    if not summaries:
        raise FileNotFoundError(f"no summary.json files matched: {config.eval_root_globs}")

    loss_targets = parse_float_list(config.loss_risk_targets)
    mismatch_targets = parse_float_list(config.mismatch_risk_targets)
    task_rows: list[dict[str, Any]] = []
    policy_rows: dict[str, list[dict[str, Any]]] = {}

    for summary in summaries:
        valid_sweep = read_csv_dicts(Path(summary["artifacts"]["threshold_sweep_valid"]))
        eval_key = "threshold_sweep_eval" if "threshold_sweep_eval" in summary["artifacts"] else "threshold_sweep_test"
        eval_sweep = read_csv_dicts(Path(summary["artifacts"][eval_key]))
        eval_by_threshold = {threshold_key(row): row for row in eval_sweep}
        task = infer_single_task(summary.get("eval_tasks") or summary.get("eval_task") or "")
        seed = infer_seed(summary)

        candidates = build_policy_candidates(valid_sweep, loss_targets, mismatch_targets, config.z_value)
        for policy_name, valid_row in candidates.items():
            selected: dict[str, Any] | None = None
            if valid_row is not None:
                selected = eval_by_threshold.get(threshold_key(valid_row))
                if selected is None:
                    raise KeyError(f"missing eval threshold for {summary['_summary_path']}: {threshold_key(valid_row)}")
            row = build_task_policy_row(summary, task, seed, policy_name, valid_row, selected, config.z_value)
            task_rows.append(row)
            policy_rows.setdefault(policy_name, []).append(row)

    aggregate_rows = [aggregate_policy(name, rows) for name, rows in sorted(policy_rows.items())]
    seed_rows = aggregate_by_seed(task_rows)
    task_csv = output_dir / "task_policy_summary.csv"
    policy_csv = output_dir / "policy_summary.csv"
    seed_csv = output_dir / "seed_policy_summary.csv"
    write_csv(task_csv, task_rows)
    write_csv(policy_csv, aggregate_rows)
    write_csv(seed_csv, seed_rows)

    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "num_eval_summaries": len(summaries),
        "num_policy_rows": len(task_rows),
        "method": {
            "name": "Wilson95 validation risk-control audit",
            "description": (
                "For each fold, select the lowest validation avg_blocks threshold "
                "whose Wilson upper bound satisfies the requested loss/mismatch risk. "
                "Selection uses validation sweeps only; held-out eval rows are used "
                "only after threshold selection."
            ),
            "references": [
                "Learn-then-Test risk-control framing: https://arxiv.org/abs/2110.01052",
                "Selective prediction risk/coverage framing: https://arxiv.org/abs/1901.09192",
                "Early-exit threshold calibration motivation: https://arxiv.org/abs/1709.01686",
            ],
        },
        "aggregate": {row["policy"]: row for row in aggregate_rows},
        "seed_rows": seed_rows,
        "task_rows": task_rows,
        "policy_csv": str(policy_csv),
        "task_csv": str(task_csv),
        "seed_csv": str(seed_csv),
        "readout": build_readout(aggregate_rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def load_summaries(patterns: list[str], *, allow_legacy_swanlab_eval: bool) -> list[dict[str, Any]]:
    paths: list[Path] = []
    for pattern in patterns:
        root_matches = [Path(item) for item in glob.glob(pattern)]
        for root in root_matches:
            if root.is_file() and root.name == "summary.json":
                paths.append(root)
            else:
                paths.extend(sorted(root.glob("leave_*_out_eval_*_all/summary.json")))
    unique_paths = sorted(set(paths))
    summaries = []
    for path in unique_paths:
        item = json.loads(path.read_text(encoding="utf-8"))
        if (
            not allow_legacy_swanlab_eval
            and (item.get("swanlab_mode") != "disabled" or item.get("swanlab_run_id") is not None)
        ):
            raise ValueError(f"risk-control audit expects local-only eval summary: {path}")
        item["_summary_path"] = str(path)
        summaries.append(item)
    return summaries


def build_policy_candidates(
    valid_sweep: list[dict[str, Any]],
    loss_targets: list[float],
    mismatch_targets: list[float],
    z_value: float,
) -> dict[str, dict[str, Any] | None]:
    candidates: dict[str, dict[str, Any] | None] = {
        "empirical_min_blocks": select_min_blocks(valid_sweep, lambda _row: True),
        "empirical_zero_loss_min_blocks": select_min_blocks(
            valid_sweep,
            lambda row: int_float(row, "losses_vs_prediction_stability") == 0,
        ),
    }
    for loss_target in loss_targets:
        policy = f"wilson_loss_le_{slug_float(loss_target)}"
        candidates[policy] = select_min_blocks(
            valid_sweep,
            lambda row, target=loss_target: risk_bounds(row, z_value)["loss_upper"] <= target,
        )
        for mismatch_target in mismatch_targets:
            policy = f"wilson_loss_le_{slug_float(loss_target)}_mismatch_le_{slug_float(mismatch_target)}"
            candidates[policy] = select_min_blocks(
                valid_sweep,
                lambda row, lt=loss_target, mt=mismatch_target: (
                    risk_bounds(row, z_value)["loss_upper"] <= lt
                    and risk_bounds(row, z_value)["mismatch_upper"] <= mt
                ),
            )
    return candidates


def select_min_blocks(
    rows: list[dict[str, Any]],
    predicate,
) -> dict[str, Any] | None:
    eligible = [row for row in rows if predicate(row)]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            as_float(row, "avg_blocks"),
            int_float(row, "losses_vs_prediction_stability"),
            int_float(row, "prediction_mismatch_vs_prediction_stability"),
            -as_float(row, "accuracy"),
        ),
    )


def build_task_policy_row(
    summary: dict[str, Any],
    task: str,
    seed: int,
    policy: str,
    valid_row: dict[str, Any] | None,
    eval_row: dict[str, Any] | None,
    z_value: float,
) -> dict[str, Any]:
    base = {
        "policy": policy,
        "seed": seed,
        "task": task,
        "summary_path": summary["_summary_path"],
        "selected": valid_row is not None and eval_row is not None,
    }
    if valid_row is None or eval_row is None:
        return {
            **base,
            "num_samples": 0,
            "eval_accuracy": None,
            "eval_avg_blocks": None,
            "eval_losses_vs_prediction_stability": None,
            "eval_mismatches_vs_prediction_stability": None,
            "valid_loss_upper": None,
            "valid_mismatch_upper": None,
            "threshold_key": "",
        }
    bounds = risk_bounds(valid_row, z_value)
    return {
        **base,
        "threshold_key": json.dumps(threshold_key(valid_row), ensure_ascii=False, sort_keys=True),
        "num_samples": int_float(eval_row, "num_samples"),
        "eval_accuracy": as_float(eval_row, "accuracy"),
        "eval_fixed_final_accuracy": as_float(eval_row, "fixed_final_accuracy"),
        "eval_prediction_stability_accuracy": as_float(eval_row, "prediction_stability_accuracy"),
        "eval_avg_blocks": as_float(eval_row, "avg_blocks"),
        "eval_losses_vs_prediction_stability": int_float(eval_row, "losses_vs_prediction_stability"),
        "eval_gains_vs_prediction_stability": int_float(eval_row, "gains_vs_prediction_stability"),
        "eval_mismatches_vs_prediction_stability": int_float(
            eval_row,
            "prediction_mismatch_vs_prediction_stability",
        ),
        "valid_num_samples": int_float(valid_row, "num_samples"),
        "valid_losses_vs_prediction_stability": int_float(valid_row, "losses_vs_prediction_stability"),
        "valid_mismatches_vs_prediction_stability": int_float(
            valid_row,
            "prediction_mismatch_vs_prediction_stability",
        ),
        "valid_loss_rate": bounds["loss_rate"],
        "valid_loss_upper": bounds["loss_upper"],
        "valid_mismatch_rate": bounds["mismatch_rate"],
        "valid_mismatch_upper": bounds["mismatch_upper"],
        "valid_avg_blocks": as_float(valid_row, "avg_blocks"),
    }


def aggregate_policy(policy: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected_rows = [row for row in rows if row["selected"]]
    total = sum(int(row["num_samples"]) for row in selected_rows)
    complete = len(selected_rows) == len(rows)
    result = {
        "policy": policy,
        "complete": complete,
        "selected_folds": len(selected_rows),
        "total_folds": len(rows),
        "num_samples": total,
    }
    if not complete or total == 0:
        return result
    losses = sum(int(row["eval_losses_vs_prediction_stability"]) for row in selected_rows)
    gains = sum(int(row["eval_gains_vs_prediction_stability"]) for row in selected_rows)
    mismatches = sum(int(row["eval_mismatches_vs_prediction_stability"]) for row in selected_rows)
    result.update(
        {
            "eval_accuracy": weighted_mean(selected_rows, "eval_accuracy"),
            "eval_fixed_final_accuracy": weighted_mean(selected_rows, "eval_fixed_final_accuracy"),
            "eval_prediction_stability_accuracy": weighted_mean(selected_rows, "eval_prediction_stability_accuracy"),
            "eval_avg_blocks": weighted_mean(selected_rows, "eval_avg_blocks"),
            "eval_losses_vs_prediction_stability": losses,
            "eval_gains_vs_prediction_stability": gains,
            "eval_loss_rate_vs_prediction_stability": losses / total,
            "eval_loss_wilson95_upper": wilson_upper(losses, total),
            "eval_mismatches_vs_prediction_stability": mismatches,
            "eval_mismatch_rate_vs_prediction_stability": mismatches / total,
            "eval_mismatch_wilson95_upper": wilson_upper(mismatches, total),
        }
    )
    return result


def aggregate_by_seed(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in task_rows:
        groups.setdefault((row["policy"], int(row["seed"])), []).append(row)
    results = []
    for (policy, seed), rows in sorted(groups.items()):
        aggregate = aggregate_policy(policy, rows)
        aggregate["seed"] = seed
        results.append(aggregate)
    return results


def build_readout(aggregate_rows: list[dict[str, Any]]) -> list[str]:
    lines = []
    for row in sorted(aggregate_rows, key=lambda item: (not item.get("complete", False), str(item["policy"]))):
        if not row.get("complete"):
            lines.append(
                f"{row['policy']}: no complete official8 selection "
                f"({row['selected_folds']}/{row['total_folds']} folds)"
            )
            continue
        lines.append(
            f"{row['policy']}: losses={row['eval_losses_vs_prediction_stability']}, "
            f"mismatches={row['eval_mismatches_vs_prediction_stability']}, "
            f"avg_blocks={row['eval_avg_blocks']:.3f}/4, "
            f"loss_wilson95={row['eval_loss_wilson95_upper']:.6f}"
        )
    return lines


def risk_bounds(row: dict[str, Any], z_value: float) -> dict[str, float]:
    total = int_float(row, "num_samples")
    losses = int_float(row, "losses_vs_prediction_stability")
    mismatches = int_float(row, "prediction_mismatch_vs_prediction_stability")
    return {
        "loss_rate": losses / total if total else 0.0,
        "loss_upper": wilson_upper(losses, total, z=z_value),
        "mismatch_rate": mismatches / total if total else 0.0,
        "mismatch_upper": wilson_upper(mismatches, total, z=z_value),
    }


def wilson_upper(successes: int | float, total: int | float, z: float = 1.96) -> float:
    total = int(total)
    successes = int(successes)
    if total <= 0:
        return 0.0
    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total) / denom
    return min(1.0, center + margin)


def threshold_key(row: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((key, str(row[key])) for key in row if key.endswith(THRESHOLD_SUFFIX)))


def weighted_mean(rows: list[dict[str, Any]], key: str) -> float:
    total = sum(int(row["num_samples"]) for row in rows)
    return sum(float(row[key]) * int(row["num_samples"]) for row in rows) / total if total else math.nan


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def read_csv_dicts(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, 0)
    if value in {"", None}:
        return 0.0
    return float(value)


def int_float(row: dict[str, Any], key: str) -> int:
    return int(round(as_float(row, key)))


def infer_seed(summary: dict[str, Any]) -> int:
    for text in [
        summary.get("_summary_path", ""),
        summary.get("output_dir", ""),
        summary.get("checkpoint", ""),
        summary.get("train_config", {}).get("output_dir", ""),
        summary.get("train_config", {}).get("experiment_name", ""),
    ]:
        match = re.search(r"seed(\d+)", str(text))
        if match:
            return int(match.group(1))
    raise ValueError(f"cannot infer seed for {summary.get('_summary_path')}")


def infer_single_task(value: str) -> str:
    tasks = [item.strip() for item in str(value).split(",") if item.strip()]
    if len(tasks) == 1:
        return tasks[0]
    return str(value)


def slug_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def parse_args() -> LatentHaltRiskControlConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root-glob", action="append", required=True)
    parser.add_argument("--output-dir", default=LatentHaltRiskControlConfig.output_dir)
    parser.add_argument("--loss-risk-targets", default=LatentHaltRiskControlConfig.loss_risk_targets)
    parser.add_argument("--mismatch-risk-targets", default=LatentHaltRiskControlConfig.mismatch_risk_targets)
    parser.add_argument("--z-value", type=float, default=LatentHaltRiskControlConfig.z_value)
    parser.add_argument(
        "--allow-legacy-swanlab-eval",
        action="store_true",
        help="Read historical eval summaries that were logged to SwanLab before the local-only eval rule.",
    )
    args = parser.parse_args()
    return LatentHaltRiskControlConfig(
        eval_root_globs=args.eval_root_glob,
        output_dir=args.output_dir,
        loss_risk_targets=args.loss_risk_targets,
        mismatch_risk_targets=args.mismatch_risk_targets,
        z_value=args.z_value,
        allow_legacy_swanlab_eval=args.allow_legacy_swanlab_eval,
    )


def main() -> None:
    summary = analyze_latent_halt_risk_control(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
