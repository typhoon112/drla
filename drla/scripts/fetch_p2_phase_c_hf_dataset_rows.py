"""Fetch deterministic HuggingFace dataset rows through ``datasets``.

This is a local-only data-preparation fallback for Phase C/Phase A source rows.
It uses ``datasets.load_dataset`` instead of the HuggingFace datasets-server
``/rows`` endpoint, which can be rate-limited on large pulls.  It writes the
same core artifacts as ``fetch_p2_phase_c_hf_rows.py`` and does not run models,
train adapters, inspect held-out generations, or create SwanLab runs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = "/data1/luyifei/drla/outputs/p2_phase_c_data_source_audits/hf_dataset_rows_20260606"


def main() -> None:
    summary = fetch_rows(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--split", required=True)
    parser.add_argument("--num-rows", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_rows <= 0:
        raise ValueError("--num-rows must be positive")
    if args.num_blocks <= 0:
        raise ValueError("--num-blocks must be positive")
    return args


def fetch_rows(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    config = args.config or None
    dataset = load_dataset(args.dataset, config, split=args.split)
    total_rows = len(dataset)
    target_rows = min(args.num_rows, total_rows)
    block_ranges = choose_block_ranges(
        total_rows=total_rows,
        target_rows=target_rows,
        num_blocks=min(args.num_blocks, target_rows),
        seed=args.seed,
    )
    selected_indices: list[int] = []
    for start, length in block_ranges:
        selected_indices.extend(range(start, start + length))
    selected_indices = sorted(dict.fromkeys(selected_indices))[:target_rows]
    rows = [dataset[int(index)] for index in selected_indices]

    rows_path = output_dir / "rows.jsonl"
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    fetch_manifest_path = output_dir / "fetch_manifest.json"
    write_jsonl(rows_path, rows)
    fetch_manifest = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "backend": "datasets.load_dataset",
        "total_rows": total_rows,
        "requested_rows": args.num_rows,
        "written_rows": len(rows),
        "seed": args.seed,
        "num_blocks": args.num_blocks,
        "block_ranges": [{"start": start, "length": length} for start, length in block_ranges],
        "selected_indices_min": min(selected_indices) if selected_indices else None,
        "selected_indices_max": max(selected_indices) if selected_indices else None,
    }
    fetch_manifest_path.write_text(
        json.dumps(fetch_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics = {
        "num_rows_total": total_rows,
        "num_rows_written": len(rows),
        "status_pass": int(len(rows) == target_rows),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "created_at": int(time.time()),
        "status": "pass" if len(rows) == target_rows else "fail",
        "rows_jsonl": str(rows_path),
        "fetch_manifest_json": str(fetch_manifest_path),
        "metrics_jsonl": str(metrics_path),
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "num_rows_total": total_rows,
        "num_rows_written": len(rows),
        "block_ranges": fetch_manifest["block_ranges"],
        "execution_boundary": [
            "local-only HuggingFace datasets.load_dataset row fetch",
            "downloads or reads selected public source rows for manifest drafting",
            "does not construct benchmark manifests",
            "does not run model generation",
            "does not run optimizer or backward",
            "does not create SwanLab runs",
            "does not inspect held-out generations or tune prompts",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_json"] = str(summary_path)
    if summary["status"] != "pass":
        raise ValueError(f"expected {target_rows} rows, wrote {len(rows)}")
    return summary


def choose_block_ranges(total_rows: int, target_rows: int, num_blocks: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    per_block = int(math.ceil(target_rows / num_blocks))
    ranges = []
    segment = max(total_rows // num_blocks, 1)
    remaining = target_rows
    for block_idx in range(num_blocks):
        length = min(per_block, remaining)
        remaining -= length
        segment_start = block_idx * segment
        segment_end = total_rows if block_idx == num_blocks - 1 else min((block_idx + 1) * segment, total_rows)
        max_start = max(segment_start, segment_end - length)
        start = rng.randint(segment_start, max_start) if max_start > segment_start else segment_start
        ranges.append((start, length))
        if remaining <= 0:
            break
    return ranges


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
