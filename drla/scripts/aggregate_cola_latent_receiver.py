"""Aggregate P2 latent receiver compatibility runs.

This script is local-only.  It reads training/eval summaries and creates a
compact comparison table for receiver ablations.  It does not train models and
must not use SwanLab.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from drla.tracking import require_swanlab_disabled_for_non_training


@dataclass(frozen=True)
class LatentReceiverAggregateConfig:
    summary_jsons: list[str] = field(default_factory=list)
    run_roots: list[str] = field(default_factory=list)
    output_dir: str = "/data1/luyifei/drla/outputs/cola_latent_receiver/p2c_receiver_compat_aggregate"
    swanlab_mode: str = "disabled"


def main() -> None:
    summary = aggregate_latent_receiver(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> LatentReceiverAggregateConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", action="append", default=[])
    parser.add_argument("--run-root", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--swanlab-mode", default="disabled")
    args = parser.parse_args()
    return LatentReceiverAggregateConfig(
        summary_jsons=args.summary_json,
        run_roots=args.run_root,
        output_dir=args.output_dir,
        swanlab_mode=args.swanlab_mode,
    )


def aggregate_latent_receiver(config: LatentReceiverAggregateConfig) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        config.swanlab_mode,
        script_kind="P2 latent receiver aggregation",
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_paths = discover_summary_paths(config)
    if not summary_paths:
        raise FileNotFoundError("no receiver summary.json files found")

    rows = []
    for path in summary_paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        config_row = summary.get("config", {})
        train_config = summary.get("train_config", {})
        source_config = train_config or config_row
        test = summary.get("test_metrics") or summary.get("metrics", {})
        valid = summary.get("last_valid_metrics", {})
        rows.append(
            {
                "input_mode": source_config.get("input_mode", ""),
                "summary_kind": "eval" if "metrics" in summary else "train",
                "summary_json": str(path),
                "swanlab_run_id": summary.get("swanlab_run_id", ""),
                "best_step": summary.get("best_step", ""),
                "best_metric": summary.get("best_metric", ""),
                "test_mean_control_auroc": test.get("mean_control_auroc", ""),
                "test_mean_control_gap": test.get("mean_control_gap", ""),
                "test_compatibility_auroc": test.get("compatibility_auroc", ""),
                "test_compatibility_auprc": test.get("compatibility_auprc", ""),
                "test_accuracy": test.get("compatibility_accuracy", ""),
                "test_positive_rate": test.get("positive_rate", ""),
                "test_target_positive_rate": test.get("target_positive_rate", ""),
                "test_metadata_only_auroc": test.get("metadata_only_auroc", ""),
                "test_shuffle_auroc": test.get("shuffle_auroc", ""),
                "test_cross_task_auroc": test.get("cross_task_auroc", ""),
                "test_wrong_block_auroc": test.get("wrong_block_auroc", ""),
                "test_noise_auroc": test.get("noise_auroc", ""),
                "test_rotation_auroc": test.get("rotation_auroc", ""),
                "valid_mean_control_auroc": valid.get("mean_control_auroc", ""),
            },
        )
    rows.sort(key=lambda row: str(row["input_mode"]))
    best_by_mean = max(rows, key=lambda row: float(row["test_mean_control_auroc"]))
    csv_path = output_dir / "receiver_ablation_summary.csv"
    write_csv(csv_path, rows)
    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "num_runs": len(rows),
        "best_by_test_mean_control_auroc": best_by_mean,
        "runs": rows,
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "receiver_ablation_summary_csv": str(csv_path),
        },
        "interpretation": (
            "This aggregate compares decoder-free receiver compatibility ablations. "
            "It is P2-C readability evidence only; it does not prove downstream task utility "
            "or superiority over text handoff."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def discover_summary_paths(config: LatentReceiverAggregateConfig) -> list[Path]:
    paths = [Path(path) for path in config.summary_jsons]
    for root in config.run_roots:
        root_path = Path(root)
        if root_path.is_file():
            paths.append(root_path)
        else:
            paths.extend(sorted(root_path.glob("*/summary.json")))
    seen = set()
    result = []
    for path in paths:
        resolved = str(path)
        if resolved not in seen:
            seen.add(resolved)
            result.append(path)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
