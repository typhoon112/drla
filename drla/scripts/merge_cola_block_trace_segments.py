"""Merge Cola block trace segments into one score-ready trace root."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


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
class MergeTraceSegmentsConfig:
    output_dir: str = "/data1/luyifei/drla/outputs/cola_block_traces/tasks_official8_full_b64_t16_seed66_20260524_merged"
    data_dir: str = "/data1/luyifei/Cola-DLM/code/generate_task_data"
    tasks: str = ",".join(OFFICIAL_COLA_TASKS)
    task_root: tuple[str, ...] = ()
    max_samples: int = 0


def merge_trace_segments(config: MergeTraceSegmentsConfig) -> dict[str, Any]:
    tasks = parse_tasks(config.tasks)
    roots_by_task = parse_task_roots(config.task_root)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_summaries = {}
    for task in tasks:
        roots = roots_by_task.get(task)
        if not roots:
            raise ValueError(f"missing --task-root for {task}")
        task_summaries[task] = merge_task(
            task=task,
            roots=[Path(root) for root in roots],
            output_dir=output_dir,
            data_dir=Path(config.data_dir),
            max_samples=config.max_samples,
        )

    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "tasks": task_summaries,
        "aggregate": {
            "num_tasks": len(task_summaries),
            "num_generation_rows": sum(item["num_generation_rows"] for item in task_summaries.values()),
            "num_trace_rows": sum(item["num_trace_rows"] for item in task_summaries.values()),
            "ok": all(item["ok"] for item in task_summaries.values()),
        },
    }
    summary_path = output_dir / "merge_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def merge_task(
    *,
    task: str,
    roots: list[Path],
    output_dir: Path,
    data_dir: Path,
    max_samples: int,
) -> dict[str, Any]:
    expected_ids = load_expected_ids(data_dir / f"{task}.jsonl", max_samples=max_samples)
    generation_rows = []
    trace_rows = []
    segment_summaries = []
    for root in roots:
        generation_path = root / f"{task}.jsonl"
        trace_path = root / f"{task}_traces.jsonl"
        if not generation_path.exists():
            raise FileNotFoundError(f"missing generation segment: {generation_path}")
        if not trace_path.exists():
            raise FileNotFoundError(f"missing trace segment: {trace_path}")
        segment_generation_rows = read_jsonl(generation_path)
        segment_trace_rows = read_jsonl(trace_path)
        generation_rows.extend(segment_generation_rows)
        trace_rows.extend(segment_trace_rows)
        segment_summaries.append(
            {
                "root": str(root),
                "generation_jsonl": str(generation_path),
                "trace_jsonl": str(trace_path),
                "num_generation_rows": len(segment_generation_rows),
                "num_trace_rows": len(segment_trace_rows),
            }
        )

    generation_ids = [str(row["id"]) for row in generation_rows]
    trace_ids = sorted({str(row["sample_id"]) for row in trace_rows}, key=sort_sample_key)
    expected_ids = expected_ids[: len(generation_ids)] if max_samples else expected_ids
    ok = generation_ids == expected_ids and trace_ids == sorted(expected_ids, key=sort_sample_key)
    if not ok:
        missing = sorted(set(expected_ids) - set(generation_ids), key=sort_sample_key)[:20]
        extra = sorted(set(generation_ids) - set(expected_ids), key=sort_sample_key)[:20]
        duplicate_count = len(generation_ids) - len(set(generation_ids))
        raise ValueError(
            f"{task} merge integrity failed: rows={len(generation_ids)} expected={len(expected_ids)} "
            f"duplicates={duplicate_count} missing_head={missing} extra_head={extra}"
        )

    generation_output = output_dir / f"{task}.jsonl"
    trace_output = output_dir / f"{task}_traces.jsonl"
    write_jsonl(generation_output, generation_rows)
    write_jsonl(trace_output, trace_rows)

    summary = {
        "ok": ok,
        "task": task,
        "segments": segment_summaries,
        "generation_jsonl": str(generation_output),
        "trace_jsonl": str(trace_output),
        "num_generation_rows": len(generation_rows),
        "num_trace_rows": len(trace_rows),
        "num_expected_rows": len(expected_ids),
        "first_id": generation_ids[0] if generation_ids else None,
        "last_id": generation_ids[-1] if generation_ids else None,
    }
    (output_dir / f"{task}_merge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_task_roots(items: tuple[str, ...]) -> dict[str, list[str]]:
    roots: dict[str, list[str]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--task-root must look like task=/path, got {item!r}")
        task, root = item.split("=", 1)
        if task not in OFFICIAL_COLA_TASKS:
            raise ValueError(f"unknown task in --task-root: {task}")
        roots.setdefault(task, []).append(root)
    return roots


def load_expected_ids(path: Path, *, max_samples: int) -> list[str]:
    rows = read_jsonl(path)
    if max_samples:
        rows = rows[:max_samples]
    return [str(row.get("id", idx)) for idx, row in enumerate(rows)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_tasks(value: str) -> list[str]:
    tasks = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [task for task in tasks if task not in OFFICIAL_COLA_TASKS]
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}")
    return tasks


def sort_sample_key(value: str) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def parse_args() -> MergeTraceSegmentsConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=MergeTraceSegmentsConfig.output_dir)
    parser.add_argument("--data-dir", default=MergeTraceSegmentsConfig.data_dir)
    parser.add_argument("--tasks", default=MergeTraceSegmentsConfig.tasks)
    parser.add_argument("--task-root", action="append", default=[])
    parser.add_argument("--max-samples", type=int, default=MergeTraceSegmentsConfig.max_samples)
    args = parser.parse_args()
    return MergeTraceSegmentsConfig(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        tasks=args.tasks,
        task_root=tuple(args.task_root),
        max_samples=args.max_samples,
    )


def main() -> None:
    summary = merge_trace_segments(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
