"""Prepare P2 candidate benchmarks for Cola MAS capability gates.

This script does not train a model and does not log to SwanLab.  It creates a
single normalized JSONL schema for the benchmarks that can plausibly support
role-conditioned Agent-A -> Agent-B experiments.  Datasets that are unavailable
or gated are reported in the manifest instead of being silently substituted.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


TASK_ORDER = [
    "arc_easy",
    "arc_challenge",
    "gsm8k",
    "mbppplus",
    "humanevalplus",
    "gpqa_diamond",
    "medqa",
]

DEFAULT_TASKS = [
    "arc_easy",
    "arc_challenge",
    "gsm8k",
    "mbppplus",
    "humanevalplus",
]


@dataclass(frozen=True)
class PrepareP2CandidateBenchmarksConfig:
    output_dir: str = "/data1/luyifei/drla/outputs/p2_capability_gate/data"
    tasks: str = ",".join(DEFAULT_TASKS)
    max_samples_per_task: int = 0
    sample_strategy: str = "head"
    seed: int = 20260601
    overwrite: bool = False


def main() -> None:
    summary = prepare_p2_candidate_benchmarks(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> PrepareP2CandidateBenchmarksConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=PrepareP2CandidateBenchmarksConfig.output_dir)
    parser.add_argument("--tasks", default=PrepareP2CandidateBenchmarksConfig.tasks)
    parser.add_argument("--max-samples-per-task", type=int, default=0)
    parser.add_argument("--sample-strategy", choices=["head", "random"], default="head")
    parser.add_argument("--seed", type=int, default=PrepareP2CandidateBenchmarksConfig.seed)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_samples_per_task < 0:
        raise ValueError("--max-samples-per-task must be non-negative")
    tasks = normalize_tasks(args.tasks)
    return PrepareP2CandidateBenchmarksConfig(
        output_dir=args.output_dir,
        tasks=",".join(tasks),
        max_samples_per_task=args.max_samples_per_task,
        sample_strategy=args.sample_strategy,
        seed=args.seed,
        overwrite=args.overwrite,
    )


def prepare_p2_candidate_benchmarks(
    config: PrepareP2CandidateBenchmarksConfig,
) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required. Activate the project conda "
            "environment before running this script."
        ) from exc

    output_dir = Path(config.output_dir)
    tasks_dir = output_dir / "tasks"
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    tasks_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.seed)
    task_names = normalize_tasks(config.tasks)
    all_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    builders: dict[str, Callable[[Any], tuple[list[dict[str, Any]], dict[str, Any]]]] = {
        "arc_easy": lambda loader: build_arc_rows(
            loader,
            task="arc_easy",
            dataset_name="allenai/ai2_arc",
            config_name="ARC-Easy",
            split="validation",
        ),
        "arc_challenge": lambda loader: build_arc_rows(
            loader,
            task="arc_challenge",
            dataset_name="allenai/ai2_arc",
            config_name="ARC-Challenge",
            split="validation",
        ),
        "gsm8k": build_gsm8k_rows,
        "mbppplus": build_mbppplus_rows,
        "humanevalplus": build_humanevalplus_rows,
        "gpqa_diamond": build_gpqa_diamond_rows,
        "medqa": build_medqa_rows,
    }

    for task in task_names:
        task_manifest: dict[str, Any] = {
            "task": task,
            "status": "pending",
            "num_rows_raw": 0,
            "num_rows_written": 0,
        }
        if task not in builders:
            task_manifest.update({"status": "unknown_task", "reason": "No builder is registered."})
            manifest.append(task_manifest)
            continue

        try:
            rows, source = builders[task](load_dataset)
            task_manifest.update(source)
            task_manifest["num_rows_raw"] = len(rows)
            rows = sample_rows(
                rows,
                max_samples=config.max_samples_per_task,
                strategy=config.sample_strategy,
                rng=rng,
            )
            task_manifest["num_rows_written"] = len(rows)
            task_manifest["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 - manifest must record external dataset failures.
            rows = []
            task_manifest.update(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                }
            )

        task_path = tasks_dir / f"{task}.jsonl"
        write_jsonl(task_path, rows)
        all_rows.extend(rows)
        task_manifest["jsonl"] = str(task_path)
        manifest.append(task_manifest)

    combined_path = output_dir / "p2_candidate_benchmarks.jsonl"
    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"
    write_jsonl(combined_path, all_rows)

    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "tasks": task_names,
        "num_rows": len(all_rows),
        "num_tasks_ok": sum(1 for item in manifest if item["status"] == "ok"),
        "num_tasks_unprepared": sum(1 for item in manifest if item["status"] == "unprepared"),
        "num_tasks_error": sum(1 for item in manifest if item["status"] == "error"),
        "combined_jsonl": str(combined_path),
        "tasks_dir": str(tasks_dir),
        "manifest_json": str(manifest_path),
        "is_smoke": config.max_samples_per_task > 0,
        "manifest": manifest,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_arc_rows(
    load_dataset: Any,
    *,
    task: str,
    dataset_name: str,
    config_name: str,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = load_dataset(dataset_name, config_name, split=split)
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(dataset):
        choices = normalize_choice_list(item["choices"])
        answer = normalize_choice_label(item.get("answerKey", ""), choices)
        rows.append(
            {
                "id": f"{task}:{item.get('id', idx)}",
                "task": task,
                "source_id": item.get("id", str(idx)),
                "source_dataset": dataset_name,
                "source_config": config_name,
                "split": split,
                "question": item.get("question", ""),
                "choices": choices,
                "answer": answer,
                "ground_truth": answer,
                "answer_text": choice_text_for_label(choices, answer),
                "answer_type": "multiple_choice",
                "benchmark_family": "p2_capability_gate",
            }
        )
    return rows, {"source_dataset": dataset_name, "source_config": config_name, "split": split}


def build_gsm8k_rows(load_dataset: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_name = "openai/gsm8k"
    config_name = "main"
    split = "test"
    dataset = load_dataset(dataset_name, config_name, split=split)
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(dataset):
        raw_answer = item.get("answer", "")
        final_answer = extract_gsm8k_final_answer(raw_answer)
        rows.append(
            {
                "id": f"gsm8k:{idx}",
                "task": "gsm8k",
                "source_id": str(idx),
                "source_dataset": dataset_name,
                "source_config": config_name,
                "split": split,
                "question": item.get("question", ""),
                "choices": [],
                "answer": final_answer,
                "ground_truth": final_answer,
                "raw_answer": raw_answer,
                "answer_type": "numeric",
                "benchmark_family": "p2_capability_gate",
            }
        )
    return rows, {"source_dataset": dataset_name, "source_config": config_name, "split": split}


def build_mbppplus_rows(load_dataset: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return build_evalplus_rows(
        load_dataset,
        task="mbppplus",
        dataset_name="evalplus/mbppplus",
        split="test",
    )


def build_humanevalplus_rows(load_dataset: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return build_evalplus_rows(
        load_dataset,
        task="humanevalplus",
        dataset_name="evalplus/humanevalplus",
        split="test",
    )


def build_gpqa_diamond_rows(load_dataset: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_name = "hendrydong/gpqa_diamond_mc"
    split = "test"
    dataset = load_dataset(dataset_name, split=split)
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(dataset):
        question, choices = parse_gpqa_mc_problem(str(item.get("problem", "")))
        answer = extract_boxed_choice(str(item.get("solution", "")))
        rows.append(
            {
                "id": f"gpqa_diamond:{idx}",
                "task": "gpqa_diamond",
                "source_id": str(idx),
                "source_dataset": dataset_name,
                "source_config": None,
                "split": split,
                "question": question,
                "choices": choices,
                "answer": answer,
                "ground_truth": answer,
                "answer_text": choice_text_for_label(choices, answer),
                "domain": item.get("domain", ""),
                "answer_type": "multiple_choice",
                "benchmark_family": "p2_capability_gate",
            }
        )
    return rows, {"source_dataset": dataset_name, "source_config": None, "split": split}


def build_medqa_rows(load_dataset: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_name = "GBaker/MedQA-USMLE-4-options"
    split = "test"
    dataset = load_dataset(dataset_name, split=split)
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(dataset):
        options = item.get("options", {})
        choices = [
            {"label": str(label), "text": str(text)}
            for label, text in sorted(options.items(), key=lambda pair: pair[0])
        ]
        answer = str(item.get("answer_idx", "")).strip()
        rows.append(
            {
                "id": f"medqa:{idx}",
                "task": "medqa",
                "source_id": str(idx),
                "source_dataset": dataset_name,
                "source_config": None,
                "split": split,
                "question": item.get("question", ""),
                "choices": choices,
                "answer": answer,
                "ground_truth": answer,
                "answer_text": choice_text_for_label(choices, answer),
                "meta_info": item.get("meta_info", ""),
                "answer_type": "multiple_choice",
                "benchmark_family": "p2_capability_gate",
            }
        )
    return rows, {"source_dataset": dataset_name, "source_config": None, "split": split}


def build_evalplus_rows(
    load_dataset: Any,
    *,
    task: str,
    dataset_name: str,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        dataset = load_dataset(dataset_name, split=split)
        source_config = None
    except Exception:
        dataset = load_dataset(dataset_name, "default", split=split)
        source_config = "default"
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(dataset):
        source_id = str(item.get("task_id", idx))
        tests = item.get("test") or item.get("tests") or ""
        rows.append(
            {
                "id": f"{task}:{source_id}",
                "task": task,
                "source_id": source_id,
                "source_dataset": dataset_name,
                "source_config": source_config,
                "split": split,
                "question": item.get("prompt", ""),
                "choices": [],
                "answer": item.get("canonical_solution", item.get("code", "")),
                "ground_truth": item.get("canonical_solution", item.get("code", "")),
                "entry_point": item.get("entry_point", ""),
                "test": tests,
                "answer_type": "code",
                "benchmark_family": "p2_capability_gate",
                "requires_execution_gate": True,
            }
        )
    return rows, {"source_dataset": dataset_name, "source_config": source_config, "split": split}


def parse_gpqa_mc_problem(problem: str) -> tuple[str, list[dict[str, str]]]:
    instruction_marker = "Please write your final answer"
    problem = problem.split(instruction_marker, 1)[0].strip()
    matches = list(
        re.finditer(
            r"(?m)^\(([A-E])\)\s*(.+?)(?=^\([A-E]\)\s*|\Z)",
            problem,
            flags=re.DOTALL,
        )
    )
    if not matches:
        return problem, []
    question = problem[: matches[0].start()].strip()
    choices = [
        {"label": match.group(1), "text": re.sub(r"\s+", " ", match.group(2)).strip()}
        for match in matches
    ]
    return question, choices


def extract_boxed_choice(solution: str) -> str:
    match = re.search(r"\\boxed\{([A-E])\}", solution)
    if match:
        return match.group(1)
    match = re.search(r"\b([A-E])\b", solution)
    return match.group(1) if match else solution.strip()


def normalize_tasks(tasks: str) -> list[str]:
    values = [task.strip() for task in tasks.split(",") if task.strip()]
    unknown = sorted(set(values) - set(TASK_ORDER))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}; known tasks: {TASK_ORDER}")
    return values


def normalize_choice_list(raw_choices: Any) -> list[dict[str, str]]:
    labels = list(raw_choices.get("label", []))
    texts = list(raw_choices.get("text", []))
    if not labels:
        labels = [chr(ord("A") + idx) for idx in range(len(texts))]
    choices: list[dict[str, str]] = []
    for idx, text in enumerate(texts):
        label = str(labels[idx] if idx < len(labels) else chr(ord("A") + idx)).strip()
        choices.append({"label": label, "text": str(text)})
    return choices


def normalize_choice_label(answer: Any, choices: list[dict[str, str]]) -> str:
    answer_text = str(answer).strip()
    labels = {choice["label"]: choice["label"] for choice in choices}
    if answer_text in labels:
        return labels[answer_text]
    if answer_text.isdigit():
        index = int(answer_text) - 1
        if 0 <= index < len(choices):
            return choices[index]["label"]
    for choice in choices:
        if normalize_text(choice["text"]) == normalize_text(answer_text):
            return choice["label"]
    return answer_text


def choice_text_for_label(choices: list[dict[str, str]], label: str) -> str:
    for choice in choices:
        if choice["label"] == label:
            return choice["text"]
    return ""


def extract_gsm8k_final_answer(raw_answer: str) -> str:
    marker = "####"
    if marker in raw_answer:
        return raw_answer.rsplit(marker, 1)[1].strip()
    numbers = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", raw_answer)
    return numbers[-1].replace(",", "") if numbers else raw_answer.strip()


def sample_rows(
    rows: list[dict[str, Any]],
    *,
    max_samples: int,
    strategy: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if max_samples <= 0 or len(rows) <= max_samples:
        return rows
    if strategy == "head":
        return rows[:max_samples]
    indices = sorted(rng.sample(range(len(rows)), max_samples))
    return [rows[idx] for idx in indices]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
