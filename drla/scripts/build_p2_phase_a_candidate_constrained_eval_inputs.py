"""Add evidence-derived candidate answers to Phase C eval online inputs.

This local-only builder is for candidate-constrained CoLA adapter diagnostics.
It copies an existing online-input package and attaches ``candidate_answers``
to rows that expose ``full_evidence``. Candidate texts come from the existing
evidence-derived candidate extractor. Gold labels, alias flags, normalized gold
matches, and scorer outputs are never copied into online inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.scripts.audit_p2_phase_c_run_leakage import audit_rows, write_audit_summary


DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p2_phase_c_control_inputs/"
    "candidate_constrained_eval_inputs_20260606"
)
SAFE_CANDIDATE_FIELDS = ["text", "rank", "rule", "source_title", "evidence_index", "evidence_kind"]


def main() -> None:
    summary = build_inputs(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--online-inputs-jsonl", required=True)
    parser.add_argument("--candidates-jsonl", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument(
        "--conditions",
        default="single_full_info",
        help="Comma-separated conditions that should receive candidate_answers.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_inputs(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_candidates < 1:
        raise ValueError("--max-candidates must be positive")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
    rows = read_jsonl(Path(args.online_inputs_jsonl))
    candidates_by_sample = {
        str(row.get("sample_id", "")): row for row in read_jsonl(Path(args.candidates_jsonl))
    }
    attach_conditions = {condition.strip() for condition in args.conditions.split(",") if condition.strip()}
    if not attach_conditions:
        raise ValueError("--conditions must contain at least one condition")

    out_rows = []
    counts: Counter[str] = Counter()
    for row in rows:
        copied = json.loads(json.dumps(row, ensure_ascii=False))
        condition = str(copied.get("condition", ""))
        fields = copied.setdefault("online_input_fields", {})
        if condition in attach_conditions and str(fields.get("full_evidence", "")).strip():
            candidate_row = candidates_by_sample.get(str(copied.get("sample_id", "")))
            if candidate_row:
                candidates = [
                    sanitize_candidate(candidate)
                    for candidate in list(candidate_row.get("candidates", []))[: args.max_candidates]
                    if isinstance(candidate, dict) and str(candidate.get("text", "")).strip()
                ]
                if candidates:
                    fields["candidate_answers"] = candidates
                    counts["rows_with_candidates"] += 1
                    counts[f"{condition}_with_candidates"] += 1
                    counts["candidate_items_attached"] += len(candidates)
                else:
                    counts["rows_without_nonempty_candidates"] += 1
            else:
                counts["rows_missing_candidate_record"] += 1
        out_rows.append(copied)
        counts[f"condition_{condition}"] += 1

    online_inputs_path = output_dir / "online_inputs.jsonl"
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(online_inputs_path, out_rows)
    leakage_summary = write_audit_summary(
        output_dir=output_dir / "leakage_audit",
        overwrite=True,
        manifest_json=args.manifest_json,
        generations_jsonl=str(online_inputs_path),
        audit=audit_rows(manifest, out_rows),
    )
    metrics = {
        "num_rows": len(out_rows),
        "max_candidates": args.max_candidates,
        "rows_with_candidates": counts["rows_with_candidates"],
        "candidate_items_attached": counts["candidate_items_attached"],
        "leakage_errors": leakage_summary["num_errors"],
        "leakage_warnings": leakage_summary["num_warnings"],
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "created_at": int(time.time()),
        "status": "pass" if leakage_summary["status"] == "pass" else "fail",
        "manifest_json": args.manifest_json,
        "source_online_inputs_jsonl": args.online_inputs_jsonl,
        "candidates_jsonl": args.candidates_jsonl,
        "online_inputs_jsonl": str(online_inputs_path),
        "metrics_jsonl": str(metrics_path),
        "leakage_audit_summary_json": leakage_summary["summary_json"],
        "output_dir": str(output_dir),
        "max_candidates": args.max_candidates,
        "attach_conditions": sorted(attach_conditions),
        "counts": dict(sorted(counts.items())),
        "metrics": metrics,
        "execution_boundary": [
            "local-only eval-input construction",
            "no model generation",
            "no optimizer or backward",
            "no SwanLab run",
            "candidate_answers contain evidence-derived text/metadata only",
            "no gold labels, alias flags, teacher correctness, or scorer output in online inputs",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_json"] = str(summary_path)
    if summary["status"] != "pass":
        raise ValueError(f"leakage audit failed; see {leakage_summary['summary_json']}")
    return summary


def sanitize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        field: candidate.get(field, "")
        for field in SAFE_CANDIDATE_FIELDS
        if str(candidate.get(field, "")).strip()
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_no}")
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
