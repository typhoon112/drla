"""Legacy GSM8K-style evaluation for Cola JSONL outputs.

Use eval_cola_benchmarks.py for the official Cola 8-task benchmark suite.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from drla.data.answer_judge import judge
from drla.tracking import finish_experiment, init_experiment, log_metrics


@dataclass(frozen=True)
class ColaEvalConfig:
    output_jsonl: str
    summary_json: str
    swanlab_mode: str = "cloud"
    experiment_name: str = "cola-dlm-gsm8k-diagnostic"


def evaluate_cola_outputs(config: ColaEvalConfig) -> dict[str, Any]:
    rows = read_jsonl(Path(config.output_jsonl))
    judged = []
    for row in rows:
        gold = str(row.get("ground_truth") or row.get("answer") or "")
        pred = str(row.get("generate") or "")
        result = judge(pred, gold)
        judged.append({"id": row.get("id"), "prediction": pred, "gold": gold, **result})

    count = len(judged)
    correct = sum(int(item["correct"]) for item in judged)
    answer_found = sum(int(item["answer_found"]) for item in judged)
    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "num_samples": count,
        "accuracy": correct / max(count, 1),
        "answer_found_rate": answer_found / max(count, 1),
        "correct": correct,
        "answer_found": answer_found,
        "examples": judged[:10],
    }
    summary_path = Path(config.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    run = init_experiment(
        stage="cola-eval",
        experiment_name=config.experiment_name,
        description="Legacy GSM8K-style diagnostic; not valid for official Cola benchmarks.",
        config=asdict(config),
        mode=config.swanlab_mode,
        tags=["cola", "gsm8k", "diagnostic"],
    )
    try:
        log_metrics(
            {
                "accuracy": summary["accuracy"],
                "answer_found_rate": summary["answer_found_rate"],
                "num_samples": summary["num_samples"],
            },
            prefix="valid",
        )
    finally:
        finish_experiment()
    summary["swanlab_run_id"] = getattr(run, "id", None)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_args() -> ColaEvalConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--swanlab-mode", default=ColaEvalConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=ColaEvalConfig.experiment_name)
    args = parser.parse_args()
    return ColaEvalConfig(
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        swanlab_mode=args.swanlab_mode,
        experiment_name=args.experiment_name,
    )


def main() -> None:
    summary = evaluate_cola_outputs(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
