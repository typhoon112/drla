"""Build oracle readiness frontier labels from Cola block traces.

Gold answers and the official scorer are used only here to create offline
labels/metrics. They must not be fed as inference-time halt features.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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


CHOICE_TASKS = {"mmlu", "obqa", "race", "siqa"}


@dataclass(frozen=True)
class ReadinessFrontierConfig:
    trace_root: str = "/data1/luyifei/drla/outputs/cola_block_traces/tasks_official8_1000_b20_t16_seed66_20260524"
    data_dir: str = "/data1/luyifei/Cola-DLM/code/generate_task_data"
    output_dir: str = "/data1/luyifei/drla/outputs/cola_readiness_frontiers/official8_1000_b20_t16_seed66_20260524"
    acc_calc_script: str = "/data1/luyifei/Cola-DLM/code/scripts/acc_calc.py"
    tasks: str = ",".join(OFFICIAL_COLA_TASKS)
    max_samples: int = 1000
    stability_window: int = 2
    min_block_number: int = 1
    swanlab_mode: str = "disabled"
    experiment_name: str = "official8-readiness-frontier"


def build_readiness_frontiers(config: ReadinessFrontierConfig) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        config.swanlab_mode,
        script_kind="readiness frontier building",
    )
    if config.stability_window <= 0:
        raise ValueError("stability_window must be positive")
    if config.min_block_number <= 0:
        raise ValueError("min_block_number must be positive")

    tasks = parse_tasks(config.tasks)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    scorer = load_official_scorer(Path(config.acc_calc_script))

    run = None
    if config.swanlab_mode != "disabled":
        run = init_experiment(
            stage="cola-readiness-frontier",
            experiment_name=config.experiment_name,
            description="Oracle readiness frontier labels from official Cola block traces.",
            config=asdict(config),
            mode=config.swanlab_mode,
            tags=["cola", "official-benchmark", "readiness", "oracle-frontier"],
        )

    summary: dict[str, Any] = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "tasks": {},
        "swanlab_run_id": getattr(run, "id", None) if run is not None else None,
    }

    try:
        with metrics_path.open("w", encoding="utf-8") as metrics_f:
            for task in tasks:
                task_summary = build_task_frontier(
                    task=task,
                    config=config,
                    scorer=scorer,
                    output_dir=output_dir,
                )
                summary["tasks"][task] = task_summary
                metrics = task_metrics(task_summary)
                if run is not None:
                    log_metrics(metrics, prefix=f"frontier/{task}")
                metrics_f.write(
                    json.dumps(
                        {
                            "created_at": int(time.time()),
                            "task": task,
                            "swanlab_run_id": getattr(run, "id", None),
                            "metrics": metrics,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                metrics_f.flush()

            aggregate = aggregate_metrics(summary["tasks"])
            summary["aggregate"] = aggregate
            if run is not None:
                log_metrics(aggregate, prefix="frontier/all")

        summary["metrics_jsonl"] = str(metrics_path)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary
    finally:
        if run is not None:
            finish_experiment()


def build_task_frontier(
    *,
    task: str,
    config: ReadinessFrontierConfig,
    scorer: Any,
    output_dir: Path,
) -> dict[str, Any]:
    trace_path = Path(config.trace_root) / f"{task}_traces.jsonl"
    if not trace_path.exists():
        raise FileNotFoundError(f"missing trace file for {task}: {trace_path}")

    references = load_references(
        Path(config.data_dir) / f"{task}.jsonl",
        max_samples=config.max_samples,
    )
    traces_by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(trace_path):
        sample_key = str(row["sample_id"])
        traces_by_sample.setdefault(sample_key, []).append(row)

    labels_path = output_dir / f"{task}_readiness_labels.jsonl"
    task_rows = 0
    final_correct = 0
    ready_samples = 0
    frontier_sum = 0
    early_ready_samples = 0
    block_correct_counts: dict[int, int] = {}
    block_total_counts: dict[int, int] = {}

    with labels_path.open("w", encoding="utf-8") as f:
        for sample_key, rows in sorted(traces_by_sample.items(), key=sort_sample_key):
            reference = references.get(sample_key)
            if reference is None:
                raise KeyError(f"trace sample_id {sample_key!r} missing from {task} references")
            rows.sort(key=lambda item: int(item["block_index"]))
            scored_rows = [
                {
                    **row,
                    **score_trace_row(
                        task=task,
                        row=row,
                        reference=reference,
                        scorer=scorer,
                    ),
                }
                for row in rows
            ]
            frontier_index = earliest_ready_block(
                scored_rows,
                stability_window=config.stability_window,
                min_block_number=config.min_block_number,
            )
            final_row = scored_rows[-1]
            final_correct += int(final_row["official_correct"])
            if frontier_index is not None:
                ready_samples += 1
                frontier_sum += frontier_index + 1
                early_ready_samples += int(frontier_index < int(final_row["block_index"]))

            final_score = float(final_row["official_score"])
            final_correct_value = int(final_row["official_correct"])
            for row in scored_rows:
                block_index = int(row["block_index"])
                block_total_counts[block_index] = block_total_counts.get(block_index, 0) + 1
                block_correct_counts[block_index] = block_correct_counts.get(block_index, 0) + int(
                    row["official_correct"]
                )
                row["earliest_ready_block_index"] = frontier_index
                row["oracle_ready"] = frontier_index is not None and block_index == frontier_index
                row["is_at_or_after_oracle_frontier"] = frontier_index is not None and block_index >= frontier_index
                row["future_gain_score"] = final_score - float(row["official_score"])
                row["future_gain_correct"] = final_correct_value - int(row["official_correct"])
                row["label_source"] = "official_acc_calc_rules_offline_only"
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                task_rows += 1

    sample_count = len(traces_by_sample)
    block_accuracy = {
        str(block): block_correct_counts[block] / max(block_total_counts[block], 1)
        for block in sorted(block_total_counts)
    }
    return {
        "task": task,
        "trace_jsonl": str(trace_path),
        "labels_jsonl": str(labels_path),
        "num_samples": sample_count,
        "num_label_rows": task_rows,
        "final_correct": final_correct,
        "final_accuracy": final_correct / max(sample_count, 1),
        "ready_samples": ready_samples,
        "ready_coverage": ready_samples / max(sample_count, 1),
        "early_ready_samples": early_ready_samples,
        "early_ready_rate": early_ready_samples / max(sample_count, 1),
        "avg_frontier_block_number": frontier_sum / max(ready_samples, 1) if ready_samples else None,
        "block_accuracy": block_accuracy,
        "stability_window": config.stability_window,
        "min_block_number": config.min_block_number,
    }


def score_trace_row(
    *,
    task: str,
    row: dict[str, Any],
    reference: dict[str, Any],
    scorer: Any,
) -> dict[str, Any]:
    generation = scorer.process_line({"generate": row.get("decode_text_so_far", "")}).get("generate", "")
    ground_truth = reference.get("ground_truth", reference.get("answer", ""))
    choices = reference.get("choices", [])

    if task == "lambada":
        prediction = scorer.get_first_word(generation)
        target = scorer.get_first_word(ground_truth)
        score = 1.0 if prediction == target else 0.0
    elif task in CHOICE_TASKS:
        prediction = scorer.extract_mmlu_choice_letter(generation, choices)
        target = scorer.extract_gt_mmlu_choice_letter(ground_truth, choices)
        if prediction and target and prediction == target:
            score = 1.0
        else:
            score = scorer.calculate_similarity(generation, ground_truth)
    else:
        prediction = scorer.extract_answer_segment(generation)
        target = scorer.extract_answer_segment(ground_truth)
        score = scorer.calculate_similarity(generation, ground_truth)

    return {
        "official_processed_generation": generation,
        "official_score": float(score),
        "official_correct": bool(score >= 1.0),
        "scored_prediction": prediction,
        "scored_target": target,
        "ground_truth": ground_truth,
        "choices": choices,
    }


def earliest_ready_block(
    rows: list[dict[str, Any]],
    *,
    stability_window: int,
    min_block_number: int,
) -> int | None:
    for index, row in enumerate(rows):
        if int(row["block_number"]) < min_block_number:
            continue
        if not row["official_correct"]:
            continue
        window = rows[index : min(len(rows), index + stability_window)]
        prediction = row.get("scored_prediction")
        if all(item["official_correct"] and item.get("scored_prediction") == prediction for item in window):
            return int(row["block_index"])
    return None


def load_references(path: Path, *, max_samples: int) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    if max_samples > 0:
        rows = rows[:max_samples]
    return {str(row["id"]): row for row in rows}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_tasks(value: str) -> list[str]:
    tasks = [task.strip() for task in value.split(",") if task.strip()]
    invalid = [task for task in tasks if task not in OFFICIAL_COLA_TASKS]
    if invalid:
        raise ValueError(f"unknown Cola tasks: {invalid}")
    return tasks


def sort_sample_key(item: tuple[str, Any]) -> tuple[int, str]:
    key = item[0]
    try:
        return 0, f"{int(key):020d}"
    except ValueError:
        return 1, key


def task_metrics(summary: dict[str, Any]) -> dict[str, float]:
    metrics = {
        "num_samples": float(summary["num_samples"]),
        "num_label_rows": float(summary["num_label_rows"]),
        "final_accuracy": float(summary["final_accuracy"]),
        "ready_coverage": float(summary["ready_coverage"]),
        "early_ready_rate": float(summary["early_ready_rate"]),
    }
    if summary["avg_frontier_block_number"] is not None:
        metrics["avg_frontier_block_number"] = float(summary["avg_frontier_block_number"])
    for block, accuracy in summary["block_accuracy"].items():
        metrics[f"block_{block}_accuracy"] = float(accuracy)
    return metrics


def aggregate_metrics(tasks: dict[str, dict[str, Any]]) -> dict[str, float]:
    total_samples = sum(task["num_samples"] for task in tasks.values())
    total_rows = sum(task["num_label_rows"] for task in tasks.values())
    total_final_correct = sum(task["final_correct"] for task in tasks.values())
    total_ready = sum(task["ready_samples"] for task in tasks.values())
    total_early_ready = sum(task["early_ready_samples"] for task in tasks.values())
    return {
        "num_samples": float(total_samples),
        "num_label_rows": float(total_rows),
        "final_accuracy": total_final_correct / max(total_samples, 1),
        "ready_coverage": total_ready / max(total_samples, 1),
        "early_ready_rate": total_early_ready / max(total_samples, 1),
    }


def load_official_scorer(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("cola_official_acc_calc", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> ReadinessFrontierConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", default=ReadinessFrontierConfig.trace_root)
    parser.add_argument("--data-dir", default=ReadinessFrontierConfig.data_dir)
    parser.add_argument("--output-dir", default=ReadinessFrontierConfig.output_dir)
    parser.add_argument("--acc-calc-script", default=ReadinessFrontierConfig.acc_calc_script)
    parser.add_argument("--tasks", default=ReadinessFrontierConfig.tasks)
    parser.add_argument("--max-samples", type=int, default=ReadinessFrontierConfig.max_samples)
    parser.add_argument("--stability-window", type=int, default=ReadinessFrontierConfig.stability_window)
    parser.add_argument("--min-block-number", type=int, default=ReadinessFrontierConfig.min_block_number)
    parser.add_argument("--swanlab-mode", default=ReadinessFrontierConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=ReadinessFrontierConfig.experiment_name)
    args = parser.parse_args()
    return ReadinessFrontierConfig(
        trace_root=args.trace_root,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        acc_calc_script=args.acc_calc_script,
        tasks=args.tasks,
        max_samples=args.max_samples,
        stability_window=args.stability_window,
        min_block_number=args.min_block_number,
        swanlab_mode=args.swanlab_mode,
        experiment_name=args.experiment_name,
    )


def main() -> None:
    summary = build_readiness_frontiers(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
