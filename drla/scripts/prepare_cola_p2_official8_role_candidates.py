"""Prepare official8-compatible P2 role-gate candidate data.

This script is safe Branch-B preparation.  It only converts existing official
CoLA benchmark JSONL files into the normalized schema consumed by
``run_cola_p2_capability_gate.py``.  It does not train, evaluate, inspect
held-out data, or replace any benchmark.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = "/data1/luyifei/Cola-DLM/code/generate_task_data"
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p2_benchmark_redesign/"
    "official8_role_candidates_20260601"
)
DEFAULT_TASKS = "obqa,mmlu,race,hellaswag,siqa,story_cloze"
KNOWN_TASKS = [
    "obqa",
    "mmlu",
    "race",
    "hellaswag",
    "siqa",
    "story_cloze",
    "squad",
    "lambada",
]


@dataclass(frozen=True)
class Official8RoleCandidateConfig:
    input_dir: str = DEFAULT_INPUT_DIR
    output_dir: str = DEFAULT_OUTPUT_DIR
    tasks: str = DEFAULT_TASKS
    max_samples_per_task: int = 0
    sample_strategy: str = "head"
    seed: int = 20260601
    overwrite: bool = False


def main() -> None:
    summary = prepare_official8_role_candidates(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> Official8RoleCandidateConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=Official8RoleCandidateConfig.input_dir)
    parser.add_argument("--output-dir", default=Official8RoleCandidateConfig.output_dir)
    parser.add_argument("--tasks", default=Official8RoleCandidateConfig.tasks)
    parser.add_argument("--max-samples-per-task", type=int, default=0)
    parser.add_argument("--sample-strategy", choices=["head", "random"], default="head")
    parser.add_argument("--seed", type=int, default=Official8RoleCandidateConfig.seed)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_samples_per_task < 0:
        raise ValueError("--max-samples-per-task must be non-negative")
    return Official8RoleCandidateConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tasks=",".join(normalize_tasks(args.tasks)),
        max_samples_per_task=args.max_samples_per_task,
        sample_strategy=args.sample_strategy,
        seed=args.seed,
        overwrite=args.overwrite,
    )


def prepare_official8_role_candidates(
    config: Official8RoleCandidateConfig,
) -> dict[str, Any]:
    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    tasks_dir = output_dir / "tasks"
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    tasks_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.seed)
    task_names = normalize_tasks(config.tasks)
    all_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for task in task_names:
        input_path = input_dir / f"{task}.jsonl"
        task_manifest: dict[str, Any] = {
            "task": task,
            "input_jsonl": str(input_path),
            "status": "pending",
            "num_rows_raw": 0,
            "num_rows_written": 0,
        }
        if not input_path.exists():
            task_manifest.update({"status": "missing_input"})
            manifest.append(task_manifest)
            write_jsonl(tasks_dir / f"{task}.jsonl", [])
            continue

        raw_rows = read_jsonl(input_path)
        task_manifest["num_rows_raw"] = len(raw_rows)
        rows = [convert_official8_row(task, row, idx) for idx, row in enumerate(raw_rows)]
        rows = [row for row in rows if row is not None]
        rows = sample_rows(
            rows,
            max_samples=config.max_samples_per_task,
            strategy=config.sample_strategy,
            rng=rng,
        )
        task_manifest["num_rows_written"] = len(rows)
        task_manifest["status"] = "ok"
        task_manifest["answer_type"] = rows[0]["answer_type"] if rows else ""
        task_manifest["jsonl"] = str(tasks_dir / f"{task}.jsonl")
        write_jsonl(tasks_dir / f"{task}.jsonl", rows)
        all_rows.extend(rows)
        manifest.append(task_manifest)

    combined_path = output_dir / "p2_official8_role_candidates.jsonl"
    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"
    write_jsonl(combined_path, all_rows)
    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "tasks": task_names,
        "num_rows": len(all_rows),
        "num_tasks_ok": sum(1 for item in manifest if item["status"] == "ok"),
        "combined_jsonl": str(combined_path),
        "tasks_dir": str(tasks_dir),
        "manifest_json": str(manifest_path),
        "is_smoke": config.max_samples_per_task > 0,
        "notes": [
            "Branch-B preparation only; no model generation or held-out evaluation.",
            "Default tasks are official8-compatible MCQ/choice tasks for role-gate triage.",
            "LAMBADA and SQuAD are supported but intentionally excluded from the default task list.",
        ],
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


def convert_official8_row(task: str, row: dict[str, Any], index: int) -> dict[str, Any] | None:
    choices = row.get("choices", [])
    if choices:
        norm_choices = normalize_choices(choices)
        answer_label = answer_to_label(row.get("ground_truth", row.get("answer", "")), norm_choices)
        question = official8_question(task, row)
        return {
            "id": f"official8_{task}:{row.get('id', index)}",
            "task": f"official8_{task}",
            "source_task": task,
            "source_id": str(row.get("source_id", row.get("id", index))),
            "source_dataset": "official_cola_generate_task_data",
            "source_jsonl": task,
            "split": "official_cola_prepared",
            "question": question,
            "choices": norm_choices,
            "answer": answer_label,
            "ground_truth": answer_label,
            "answer_text": choice_text_for_label(norm_choices, answer_label),
            "answer_type": "multiple_choice",
            "benchmark_family": "p2_official8_role_redesign",
        }
    if task == "squad":
        question = official8_question(task, row)
        answer = str(row.get("ground_truth", row.get("answer", ""))).strip()
        return {
            "id": f"official8_{task}:{row.get('id', index)}",
            "task": f"official8_{task}",
            "source_task": task,
            "source_id": str(row.get("id", index)),
            "source_dataset": "official_cola_generate_task_data",
            "source_jsonl": task,
            "split": "official_cola_prepared",
            "question": question,
            "choices": [],
            "answer": answer,
            "ground_truth": answer,
            "answer_type": "short_answer",
            "benchmark_family": "p2_official8_role_redesign",
        }
    if task == "lambada":
        question = official8_question(task, row)
        answer = str(row.get("ground_truth", row.get("answer", ""))).strip()
        return {
            "id": f"official8_{task}:{row.get('id', index)}",
            "task": f"official8_{task}",
            "source_task": task,
            "source_id": str(row.get("id", index)),
            "source_dataset": "official_cola_generate_task_data",
            "source_jsonl": task,
            "split": "official_cola_prepared",
            "question": question,
            "choices": [],
            "answer": answer,
            "ground_truth": answer,
            "answer_type": "short_answer",
            "benchmark_family": "p2_official8_role_redesign",
        }
    return None


def official8_question(task: str, row: dict[str, Any]) -> str:
    if task == "siqa" and row.get("context"):
        return f"Context: {row.get('context', '')}\nQuestion: {row.get('question', '')}"
    if task == "squad" and row.get("context"):
        return f"Context: {row.get('context', '')}\nQuestion: {row.get('question', '')}"
    return str(row.get("question", ""))


def normalize_choices(raw_choices: Any) -> list[dict[str, str]]:
    if isinstance(raw_choices, dict):
        labels = list(raw_choices.get("label", []))
        texts = list(raw_choices.get("text", []))
    else:
        labels = []
        texts = list(raw_choices)
    if not labels:
        labels = [chr(ord("A") + idx) for idx in range(len(texts))]
    choices = []
    for idx, text in enumerate(texts):
        label = str(labels[idx] if idx < len(labels) else chr(ord("A") + idx)).strip()
        choices.append({"label": label, "text": str(text)})
    return choices


def answer_to_label(answer: Any, choices: list[dict[str, str]]) -> str:
    answer_text = str(answer).strip()
    labels = {choice["label"]: choice["label"] for choice in choices}
    if answer_text in labels:
        return answer_text
    for choice in choices:
        if normalize_text(choice["text"]) == normalize_text(answer_text):
            return choice["label"]
    return answer_text


def choice_text_for_label(choices: list[dict[str, str]], label: str) -> str:
    for choice in choices:
        if choice["label"] == label:
            return choice["text"]
    return ""


def normalize_tasks(tasks: str) -> list[str]:
    values = [task.strip() for task in tasks.split(",") if task.strip()]
    unknown = sorted(set(values) - set(KNOWN_TASKS))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}; known tasks: {KNOWN_TASKS}")
    return values


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
    return " ".join(str(text).strip().lower().split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
