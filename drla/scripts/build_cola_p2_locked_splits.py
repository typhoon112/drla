"""Build locked calibration / held-out splits for P2 capability repair.

The split is deterministic and local-only.  It is intended to prevent prompt
and protocol repair from repeatedly tuning on the same rows that will later be
used for held-out capability gates and P2 main tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_INPUT_JSONL = (
    "/data1/luyifei/drla/outputs/p2_capability_gate/data_20260601/"
    "p2_candidate_benchmarks.jsonl"
)


@dataclass(frozen=True)
class P2LockedSplitConfig:
    input_jsonl: str = DEFAULT_INPUT_JSONL
    output_dir: str = "/data1/luyifei/drla/outputs/p2_capability_gate/locked_splits_20260601"
    split_seed: int = 20260602
    calibration_fraction: float = 0.2
    min_calibration_per_task: int = 20
    max_calibration_per_task: int = 300
    overwrite: bool = False


def main() -> None:
    summary = build_locked_splits(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> P2LockedSplitConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", default=P2LockedSplitConfig.input_jsonl)
    parser.add_argument("--output-dir", default=P2LockedSplitConfig.output_dir)
    parser.add_argument("--split-seed", type=int, default=P2LockedSplitConfig.split_seed)
    parser.add_argument("--calibration-fraction", type=float, default=P2LockedSplitConfig.calibration_fraction)
    parser.add_argument("--min-calibration-per-task", type=int, default=P2LockedSplitConfig.min_calibration_per_task)
    parser.add_argument("--max-calibration-per-task", type=int, default=P2LockedSplitConfig.max_calibration_per_task)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 < args.calibration_fraction < 1:
        raise ValueError("--calibration-fraction must be in (0, 1)")
    if args.min_calibration_per_task < 0 or args.max_calibration_per_task <= 0:
        raise ValueError("calibration size bounds must be non-negative/positive")
    if args.min_calibration_per_task > args.max_calibration_per_task:
        raise ValueError("--min-calibration-per-task cannot exceed --max-calibration-per-task")
    return P2LockedSplitConfig(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        split_seed=args.split_seed,
        calibration_fraction=args.calibration_fraction,
        min_calibration_per_task=args.min_calibration_per_task,
        max_calibration_per_task=args.max_calibration_per_task,
        overwrite=args.overwrite,
    )


def build_locked_splits(config: P2LockedSplitConfig) -> dict[str, Any]:
    input_path = Path(config.input_jsonl)
    rows = read_jsonl(input_path)
    if not rows:
        raise ValueError(f"input_jsonl is empty: {input_path}")
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir = output_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task"])].append(row)

    calibration_rows: list[dict[str, Any]] = []
    heldout_rows: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    task_summaries: list[dict[str, Any]] = []

    for task in sorted(by_task):
        task_rows = sorted(by_task[task], key=lambda item: stable_row_key(item))
        selected_ids = select_calibration_ids(task_rows, task=task, config=config)
        task_calibration = []
        task_heldout = []
        for row in task_rows:
            split = "calibration" if row["id"] in selected_ids else "heldout"
            annotated = {**row, "p2_split": split, "p2_split_seed": config.split_seed}
            assignments.append(
                {
                    "id": row["id"],
                    "task": task,
                    "split": split,
                    "answer_type": row.get("answer_type", ""),
                    "ground_truth": row.get("ground_truth", ""),
                    "stable_key": stable_row_key(row),
                }
            )
            if split == "calibration":
                calibration_rows.append(annotated)
                task_calibration.append(annotated)
            else:
                heldout_rows.append(annotated)
                task_heldout.append(annotated)
        write_jsonl(tasks_dir / f"{task}_calibration.jsonl", task_calibration)
        write_jsonl(tasks_dir / f"{task}_heldout.jsonl", task_heldout)
        task_summaries.append(
            {
                "task": task,
                "answer_type": task_rows[0].get("answer_type", ""),
                "num_total": len(task_rows),
                "num_calibration": len(task_calibration),
                "num_heldout": len(task_heldout),
                "calibration_fraction_actual": len(task_calibration) / len(task_rows),
                "stratified_by_ground_truth": task_rows[0].get("answer_type") == "multiple_choice",
            }
        )

    calibration_path = output_dir / "calibration.jsonl"
    heldout_path = output_dir / "heldout.jsonl"
    assignment_path = output_dir / "split_assignments.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(calibration_path, calibration_rows)
    write_jsonl(heldout_path, heldout_rows)
    write_jsonl(assignment_path, assignments)
    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "input_jsonl": str(input_path),
        "input_sha256": sha256_file(input_path),
        "num_total": len(rows),
        "num_calibration": len(calibration_rows),
        "num_heldout": len(heldout_rows),
        "calibration_jsonl": str(calibration_path),
        "heldout_jsonl": str(heldout_path),
        "split_assignments_jsonl": str(assignment_path),
        "tasks_dir": str(tasks_dir),
        "task_summaries": task_summaries,
        "usage_rules": [
            "Use calibration split for prompt/protocol repair and parser audits.",
            "Use heldout split only for locked capability gate and P2 main-table evaluation.",
            "Do not inspect heldout sample-level generations while repairing prompts.",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def select_calibration_ids(
    rows: list[dict[str, Any]],
    *,
    task: str,
    config: P2LockedSplitConfig,
) -> set[str]:
    target = int(round(len(rows) * config.calibration_fraction))
    target = max(config.min_calibration_per_task, target)
    target = min(config.max_calibration_per_task, target)
    target = min(max(1, target), len(rows) - 1)
    if rows[0].get("answer_type") == "multiple_choice":
        return stratified_calibration_ids(rows, target=target, task=task, config=config)
    rng = random.Random(split_rng_seed(config.split_seed, task, "unstratified"))
    shuffled = list(rows)
    rng.shuffle(shuffled)
    return {row["id"] for row in shuffled[:target]}


def stratified_calibration_ids(
    rows: list[dict[str, Any]],
    *,
    target: int,
    task: str,
    config: P2LockedSplitConfig,
) -> set[str]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[str(row.get("ground_truth", ""))].append(row)
    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for label in sorted(by_label):
        group = list(by_label[label])
        rng = random.Random(split_rng_seed(config.split_seed, task, f"label:{label}"))
        rng.shuffle(group)
        quota = max(1, int(round(target * len(group) / len(rows))))
        quota = min(quota, len(group))
        selected.extend(group[:quota])
        leftovers.extend(group[quota:])
    if len(selected) > target:
        rng = random.Random(split_rng_seed(config.split_seed, task, "trim"))
        rng.shuffle(selected)
        selected = selected[:target]
    elif len(selected) < target:
        rng = random.Random(split_rng_seed(config.split_seed, task, "fill"))
        rng.shuffle(leftovers)
        selected.extend(leftovers[: target - len(selected)])
    return {row["id"] for row in selected}


def split_rng_seed(seed: int, task: str, salt: str) -> int:
    payload = f"{seed}:{task}:{salt}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def stable_row_key(row: dict[str, Any]) -> str:
    return f"{row.get('task', '')}:{row.get('id', '')}:{row.get('source_id', '')}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
