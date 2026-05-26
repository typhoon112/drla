"""Evaluate official Cola benchmark outputs with the official scorer."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from drla.tracking import finish_experiment, init_experiment, log_metrics
from drla.tracking import require_swanlab_disabled_for_non_training


OFFICIAL_COLA_TASKS = [
    "lambada",
    "mmlu",
    "obqa",
    "hellaswag",
    "race",
    "siqa",
    "squad",
    "story_cloze",
]


@dataclass(frozen=True)
class ColaBenchmarkEvalConfig:
    eval_root: str = "/data1/luyifei/Cola-DLM/code/eval_output"
    summary_json: str = "/data1/luyifei/drla/outputs/cola_official_benchmarks/summary.json"
    summary_csv: str | None = None
    acc_calc_script: str = "/data1/luyifei/Cola-DLM/code/scripts/acc_calc.py"
    skip_acc_calc: bool = False
    swanlab_mode: str = "disabled"
    experiment_name: str = "official-cola-benchmark-eval"


def evaluate_cola_benchmarks(config: ColaBenchmarkEvalConfig) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        config.swanlab_mode,
        script_kind="official Cola benchmark evaluation",
    )
    eval_root = Path(config.eval_root)
    if not eval_root.exists():
        raise FileNotFoundError(f"Cola benchmark eval root does not exist: {eval_root}")

    summary_csv = Path(config.summary_csv) if config.summary_csv else eval_root / "accuracy_summary.csv"
    if not config.skip_acc_calc:
        run_official_acc_calc(config, summary_csv)
    if not summary_csv.exists():
        raise FileNotFoundError(f"Cola benchmark summary CSV does not exist: {summary_csv}")

    results_by_alias = parse_summary_csv(summary_csv)
    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "official_tasks": OFFICIAL_COLA_TASKS,
        "summary_csv": str(summary_csv),
        "results_by_alias": results_by_alias,
    }
    summary_path = Path(config.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    summary["swanlab_run_id"] = None
    if config.swanlab_mode != "disabled":
        run = init_experiment(
            stage="cola-benchmark",
            experiment_name=config.experiment_name,
            description="Official Cola 8-task benchmark evaluation using scripts/acc_calc.py.",
            config={
                **asdict(config),
                "official_tasks": OFFICIAL_COLA_TASKS,
                "summary_csv": str(summary_csv),
            },
            mode=config.swanlab_mode,
            tags=["cola", "official-benchmark", "baseline"],
        )
        try:
            log_metrics(flatten_for_swanlab(results_by_alias), prefix="valid")
            summary["swanlab_run_id"] = getattr(run, "id", None)
        finally:
            finish_experiment()
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def run_official_acc_calc(config: ColaBenchmarkEvalConfig, summary_csv: Path) -> None:
    script = Path(config.acc_calc_script)
    if not script.exists():
        raise FileNotFoundError(f"Official Cola acc_calc.py not found: {script}")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(script), config.eval_root, str(summary_csv)],
        check=True,
    )


def parse_summary_csv(path: Path) -> dict[str, dict[str, float | None]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows or len(rows[0]) < 2:
        raise ValueError(f"Unexpected Cola benchmark summary CSV format: {path}")
    aliases = rows[0][1:]
    results = {alias: {} for alias in aliases}
    for row in rows[1:]:
        if not row:
            continue
        task = row[0]
        for alias, value in zip(aliases, row[1:]):
            results[alias][task] = float(value) if value.strip() else None
    return results


def flatten_for_swanlab(results_by_alias: dict[str, dict[str, float | None]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for alias, task_metrics in results_by_alias.items():
        safe_alias = alias.replace("/", "_")
        for task, value in task_metrics.items():
            if value is not None:
                metrics[f"{safe_alias}/{task}_accuracy_pct"] = float(value)
    return metrics


def parse_args() -> ColaBenchmarkEvalConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", default=ColaBenchmarkEvalConfig.eval_root)
    parser.add_argument("--summary-json", default=ColaBenchmarkEvalConfig.summary_json)
    parser.add_argument("--summary-csv", default=ColaBenchmarkEvalConfig.summary_csv)
    parser.add_argument("--acc-calc-script", default=ColaBenchmarkEvalConfig.acc_calc_script)
    parser.add_argument("--skip-acc-calc", action="store_true")
    parser.add_argument("--swanlab-mode", default=ColaBenchmarkEvalConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=ColaBenchmarkEvalConfig.experiment_name)
    args = parser.parse_args()
    return ColaBenchmarkEvalConfig(
        eval_root=args.eval_root,
        summary_json=args.summary_json,
        summary_csv=args.summary_csv,
        acc_calc_script=args.acc_calc_script,
        skip_acc_calc=args.skip_acc_calc,
        swanlab_mode=args.swanlab_mode,
        experiment_name=args.experiment_name,
    )


def main() -> None:
    summary = evaluate_cola_benchmarks(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
