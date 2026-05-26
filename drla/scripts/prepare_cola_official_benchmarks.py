"""Prepare official Cola 8-task benchmark JSONL data."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from datasets import load_dataset


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
class ColaBenchmarkDataConfig:
    output_dir: str = "/data1/luyifei/Cola-DLM/code/generate_task_data"
    max_samples: int | None = None
    overwrite: bool = False


def prepare_cola_official_benchmarks(config: ColaBenchmarkDataConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_builders: dict[str, Callable[[int | None], Iterable[dict[str, Any]]]] = {
        "lambada": build_lambada,
        "mmlu": build_mmlu,
        "obqa": build_obqa,
        "hellaswag": build_hellaswag,
        "race": build_race,
        "siqa": build_siqa,
        "squad": build_squad,
        "story_cloze": build_story_cloze,
    }

    task_summaries: dict[str, Any] = {}
    for task in OFFICIAL_COLA_TASKS:
        output_path = output_dir / f"{task}.jsonl"
        if output_path.exists() and not config.overwrite:
            task_summaries[task] = {
                "output_jsonl": str(output_path),
                "num_samples": count_jsonl(output_path),
                "status": "exists",
            }
            continue
        count = write_jsonl(output_path, task_builders[task](config.max_samples))
        task_summaries[task] = {
            "output_jsonl": str(output_path),
            "num_samples": count,
            "status": "written",
        }

    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "tasks": OFFICIAL_COLA_TASKS,
        "task_summaries": task_summaries,
        "sources": source_manifest(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_lambada(max_samples: int | None) -> Iterable[dict[str, Any]]:
    ds = load_dataset("EleutherAI/lambada_openai", split="test")
    for idx, row in enumerate(limit_dataset(ds, max_samples)):
        question, answer = split_lambada_text(row["text"])
        yield {
            "id": idx + 1,
            "question": question,
            "answer": answer,
            "ground_truth": answer,
        }


def build_mmlu(max_samples: int | None) -> Iterable[dict[str, Any]]:
    ds = load_dataset("cais/mmlu", "all", split="test")
    for idx, row in enumerate(limit_dataset(ds, max_samples)):
        choices = list(row["choices"])
        answer = choices[int(row["answer"])]
        yield {
            "id": idx,
            "question": row["question"],
            "answer": answer,
            "ground_truth": answer,
            "choices": choices,
            "subject": row.get("subject"),
        }


def build_obqa(max_samples: int | None) -> Iterable[dict[str, Any]]:
    ds = load_dataset("allenai/openbookqa", "main", split="test")
    for idx, row in enumerate(limit_dataset(ds, max_samples)):
        labels = list(row["choices"]["label"])
        choices = list(row["choices"]["text"])
        answer = choices[labels.index(row["answerKey"])]
        yield {
            "id": idx,
            "source_id": row["id"],
            "question": row["question_stem"],
            "answer": answer,
            "ground_truth": answer,
            "choices": choices,
        }


def build_hellaswag(max_samples: int | None) -> Iterable[dict[str, Any]]:
    ds = load_dataset("Rowan/hellaswag", split="validation")
    for idx, row in enumerate(limit_dataset(ds, max_samples)):
        choices = list(row["endings"])
        answer = choices[int(row["label"])]
        yield {
            "id": idx,
            "source_id": row["source_id"],
            "question": row["ctx"],
            "answer": answer,
            "ground_truth": answer,
            "choices": choices,
        }


def build_race(max_samples: int | None) -> Iterable[dict[str, Any]]:
    ds = load_dataset("ehovy/race", "all", split="validation")
    for idx, row in enumerate(limit_dataset(ds, max_samples)):
        choices = list(row["options"])
        answer = choices[ord(row["answer"]) - ord("A")]
        yield {
            "id": idx,
            "source_id": row["example_id"],
            "question": f"{row['article']} {row['question']}",
            "answer": answer,
            "ground_truth": answer,
            "choices": choices,
        }


def build_siqa(max_samples: int | None) -> Iterable[dict[str, Any]]:
    ds = load_dataset("baber/social_i_qa", split="validation")
    for idx, row in enumerate(limit_dataset(ds, max_samples)):
        choices = [row["answerA"], row["answerB"], row["answerC"]]
        answer = choices[int(row["label"]) - 1]
        yield {
            "id": idx,
            "context": row["context"],
            "question": row["question"],
            "answer": answer,
            "ground_truth": answer,
            "choices": choices,
        }


def build_squad(max_samples: int | None) -> Iterable[dict[str, Any]]:
    ds = load_dataset("squad", split="validation")
    for row in limit_dataset(ds, max_samples):
        answers = list(row["answers"]["text"])
        answer = answers[0] if answers else ""
        yield {
            "id": row["id"],
            "title": row.get("title"),
            "context": row["context"],
            "question": row["question"],
            "answer": answer,
            "ground_truth": answer,
        }


def build_story_cloze(max_samples: int | None) -> Iterable[dict[str, Any]]:
    ds = load_dataset("MoE-UNC/story_cloze", split="validation")
    for idx, row in enumerate(limit_dataset(ds, max_samples)):
        choices = [row["sentence_quiz1"], row["sentence_quiz2"]]
        answer = choices[int(row["answer_right_ending"]) - 1]
        story = " ".join(
            [
                row["input_sentence_1"],
                row["input_sentence_2"],
                row["input_sentence_3"],
                row["input_sentence_4"],
            ]
        )
        yield {
            "id": idx,
            "source_id": row["story_id"],
            "question": story,
            "answer": answer,
            "ground_truth": answer,
            "choices": choices,
        }


def split_lambada_text(text: str) -> tuple[str, str]:
    stripped = text.rstrip()
    prefix, sep, last = stripped.rpartition(" ")
    if not sep or not last:
        raise ValueError("LAMBADA text must contain at least two whitespace-separated tokens")
    return prefix, last


def limit_dataset(dataset: Any, max_samples: int | None) -> Iterable[dict[str, Any]]:
    limit = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    for index in range(limit):
        yield dataset[index]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def source_manifest() -> dict[str, dict[str, str]]:
    return {
        "lambada": {"dataset": "EleutherAI/lambada_openai", "split": "test"},
        "mmlu": {"dataset": "cais/mmlu", "config": "all", "split": "test"},
        "obqa": {"dataset": "allenai/openbookqa", "config": "main", "split": "test"},
        "hellaswag": {"dataset": "Rowan/hellaswag", "split": "validation"},
        "race": {"dataset": "ehovy/race", "config": "all", "split": "validation"},
        "siqa": {
            "dataset": "baber/social_i_qa",
            "split": "validation",
            "note": "Parquet mirror of Social IQa used because the legacy allenai/social_i_qa script loader is unsupported by this datasets version.",
        },
        "squad": {"dataset": "squad", "split": "validation"},
        "story_cloze": {"dataset": "MoE-UNC/story_cloze", "split": "validation"},
    }


def parse_args() -> ColaBenchmarkDataConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=ColaBenchmarkDataConfig.output_dir)
    parser.add_argument("--max-samples", type=int, default=ColaBenchmarkDataConfig.max_samples)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    return ColaBenchmarkDataConfig(
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
    )


def main() -> None:
    summary = prepare_cola_official_benchmarks(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
