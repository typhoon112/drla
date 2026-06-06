"""Rescore existing official8 native single generations.

This is local-only post-processing for ``run_cola_p2_official8_native_single_gate``.
It is useful when scoring logic changes but generation should not be rerun.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from drla.scripts.run_cola_p2_official8_native_single_gate import (
    build_task_summaries,
    score_official8,
    write_metrics_jsonl,
    write_task_summary_csv,
)
from drla.tracking import require_swanlab_disabled_for_non_training


@dataclass(frozen=True)
class RescoreOfficial8NativeConfig:
    input_summary_json: str
    output_dir: str
    swanlab_mode: str = "disabled"
    min_nonempty_rate: float = 0.95
    min_parseable_rate: float = 0.90
    min_accuracy_margin: float = 0.02
    max_samples_per_task: int = 0


def main() -> None:
    summary = rescore(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> RescoreOfficial8NativeConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-summary-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--swanlab-mode", default="disabled")
    args = parser.parse_args()
    source_summary = json.loads(Path(args.input_summary_json).read_text(encoding="utf-8"))
    source_config = source_summary.get("config", {})
    return RescoreOfficial8NativeConfig(
        input_summary_json=args.input_summary_json,
        output_dir=args.output_dir,
        swanlab_mode=args.swanlab_mode,
        min_nonempty_rate=float(source_config.get("min_nonempty_rate", 0.95)),
        min_parseable_rate=float(source_config.get("min_parseable_rate", 0.90)),
        min_accuracy_margin=float(source_config.get("min_accuracy_margin", 0.02)),
        max_samples_per_task=int(source_config.get("max_samples_per_task", 0)),
    )


def rescore(config: RescoreOfficial8NativeConfig) -> dict:
    require_swanlab_disabled_for_non_training(
        config.swanlab_mode,
        script_kind="rescore Cola P2 official8 native single gate",
    )
    source_summary = json.loads(Path(config.input_summary_json).read_text(encoding="utf-8"))
    generations_path = Path(source_summary["generations_jsonl"])
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rescored_records = []
    with generations_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            task = str(record.get("source_task") or str(record.get("task", "")).replace("official8_", ""))
            prompt = {
                "ground_truth": record.get("ground_truth", ""),
                "choices": record.get("choices", []),
            }
            record["score"] = score_official8(task, str(record.get("generate", "")), prompt)
            rescored_records.append(record)

    generations_out = output_dir / "generations.jsonl"
    with generations_out.open("w", encoding="utf-8") as f:
        for record in rescored_records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    summaries = build_task_summaries(rescored_records, config=config)
    task_summary_path = output_dir / "task_summary.csv"
    metrics_path = output_dir / "metrics.jsonl"
    write_task_summary_csv(task_summary_path, summaries)
    write_metrics_jsonl(metrics_path, summaries)
    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "source_summary_json": config.input_summary_json,
        "source_generations_jsonl": str(generations_path),
        "generations_jsonl": str(generations_out),
        "metrics_jsonl": str(metrics_path),
        "task_summary_csv": str(task_summary_path),
        "num_generation_records": len(rescored_records),
        "is_smoke": config.max_samples_per_task > 0,
        "task_summaries": summaries,
        "admitted_tasks": [] if config.max_samples_per_task else [
            row["task"] for row in summaries if row["gate_pass"]
        ],
        "notes": [
            "Local-only rescore; no generation and no SwanLab run.",
            "Uses official acc_calc-style first-segment preprocessing before scoring.",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    main()
