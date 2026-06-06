"""Fetch deterministic HuggingFace datasets-server rows for Phase C prep.

This is a data-preparation utility for Phase C manifest drafting.  It downloads
selected rows from the public HuggingFace datasets-server ``/rows`` endpoint and
writes local JSONL plus fetch metadata.  It does not construct benchmarks, run
models, train adapters, inspect held-out generations, or create SwanLab runs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = "/data1/luyifei/drla/outputs/p2_phase_c_data_source_audits/hf_rows_20260601"
BASE_URL = "https://datasets-server.huggingface.co/rows"


def main() -> None:
    summary = fetch_rows(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--num-rows", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_rows <= 0:
        raise ValueError("--num-rows must be positive")
    if args.num_blocks <= 0:
        raise ValueError("--num-blocks must be positive")
    if args.page_size <= 0 or args.page_size > 100:
        raise ValueError("--page-size must be in 1..100 for datasets-server rows endpoint")
    return args


def fetch_rows(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    first_page = fetch_page(args.dataset, args.config, args.split, offset=0, length=1)
    total_rows = int(first_page.get("num_rows_total", 0))
    if total_rows <= 0:
        raise ValueError("datasets-server did not report a positive num_rows_total")
    target_rows = min(args.num_rows, total_rows)
    block_ranges = choose_block_ranges(
        total_rows=total_rows,
        target_rows=target_rows,
        num_blocks=min(args.num_blocks, target_rows),
        seed=args.seed,
    )

    rows_by_index: dict[int, dict[str, Any]] = {}
    pages_fetched = []
    for start, length in block_ranges:
        for offset in range(start, start + length, args.page_size):
            page_length = min(args.page_size, start + length - offset)
            page = fetch_page(args.dataset, args.config, args.split, offset=offset, length=page_length)
            page_rows = page.get("rows", [])
            if not isinstance(page_rows, list):
                raise ValueError(f"rows endpoint returned non-list rows at offset {offset}")
            for item in page_rows:
                if not isinstance(item, dict) or not isinstance(item.get("row"), dict):
                    continue
                row_idx = item.get("row_idx", item.get("index"))
                if not isinstance(row_idx, int):
                    row_idx = offset + len(rows_by_index)
                rows_by_index[row_idx] = item["row"]
            pages_fetched.append({"offset": offset, "length": page_length, "returned": len(page_rows)})

    selected_indices = sorted(rows_by_index)[:target_rows]
    rows = [rows_by_index[index] for index in selected_indices]
    rows_path = output_dir / "rows.jsonl"
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    fetch_manifest_path = output_dir / "fetch_manifest.json"
    with rows_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    fetch_manifest = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "endpoint": BASE_URL,
        "total_rows": total_rows,
        "requested_rows": args.num_rows,
        "written_rows": len(rows),
        "seed": args.seed,
        "num_blocks": args.num_blocks,
        "page_size": args.page_size,
        "block_ranges": [{"start": start, "length": length} for start, length in block_ranges],
        "selected_indices_min": min(selected_indices) if selected_indices else None,
        "selected_indices_max": max(selected_indices) if selected_indices else None,
        "pages_fetched": pages_fetched,
    }
    fetch_manifest_path.write_text(
        json.dumps(fetch_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics = {
        "num_rows_total": total_rows,
        "num_rows_written": len(rows),
        "num_pages_fetched": len(pages_fetched),
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
            "local-only HuggingFace datasets-server row fetch",
            "downloads selected public source rows for manifest drafting",
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


def fetch_page(dataset: str, config: str, split: str, offset: int, length: int) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"{BASE_URL}?{query}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


if __name__ == "__main__":
    main()
