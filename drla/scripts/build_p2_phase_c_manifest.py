"""Build a P2 Phase C manifest from normalized sample records.

This is a local-only packaging script.  It does not download data, synthesize
benchmark samples, run models, inspect hidden held-out outputs, train adapters,
or create SwanLab runs.  Each input JSONL row must already be a complete
``p2_phase_c_manifest_v0`` sample object.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.scripts.validate_p2_phase_c_manifest import validate_manifest


DEFAULT_OUTPUT_DIR = "/data1/luyifei/drla/outputs/p2_phase_c_manifests/manifest_20260601"


def main() -> None:
    summary = build_manifest(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--protocol-version", required=True)
    parser.add_argument("--created-at-utc", default="2026-06-01T00:00:00Z")
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    records_path = Path(args.records_jsonl)
    records = read_jsonl(records_path)
    if not records:
        raise ValueError(f"records_jsonl is empty: {records_path}")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    families = sorted({str(record.get("family", "")) for record in records if record.get("family")})
    manifest = {
        "manifest_version": "p2_phase_c_manifest_v0",
        "protocol_version": args.protocol_version,
        "created_at_utc": args.created_at_utc,
        "families": families,
        "samples": records,
    }
    if args.split_seed is not None:
        manifest["split_seed"] = args.split_seed

    manifest_path = output_dir / "manifest.json"
    samples_path = output_dir / "samples.jsonl"
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(samples_path, records)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    validation_dir = output_dir / "manifest_audit"
    validation_summary = validate_manifest(
        argparse.Namespace(
            manifest_json=str(manifest_path),
            output_dir=str(validation_dir),
            overwrite=True,
        )
    )
    split_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    for record in records:
        split = str(record.get("split", ""))
        task = str(record.get("task_name", ""))
        split_counts[split] = split_counts.get(split, 0) + 1
        task_counts[task] = task_counts.get(task, 0) + 1

    metrics = {
        "num_samples": len(records),
        "num_families": len(families),
        "validation_errors": validation_summary["num_errors"],
        "validation_warnings": validation_summary["num_warnings"],
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "created_at": int(time.time()),
        "status": "pass" if validation_summary["status"] == "pass" else "fail",
        "records_jsonl": str(records_path),
        "manifest_json": str(manifest_path),
        "samples_jsonl": str(samples_path),
        "metrics_jsonl": str(metrics_path),
        "manifest_audit_summary_json": validation_summary["summary_json"],
        "num_samples": len(records),
        "families": families,
        "split_counts": split_counts,
        "task_counts": task_counts,
        "execution_boundary": [
            "local-only manifest packaging",
            "input rows must already be complete sample records",
            "no data download or benchmark synthesis",
            "no model generation",
            "no optimizer or backward",
            "no SwanLab run",
            "no held-out inspection beyond manifest-level split counts",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_json"] = str(summary_path)
    return summary


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
