"""Build support-only solver controls for Phase A diagnostic eval.

This is local-only data preparation. It strips distractor lines from existing
Phase C ``single_full_info`` online inputs, keeps ``single_q_only`` controls,
and writes a diagnostic control package. It must not be used as a locked gate or
as held-out repair data; its purpose is to test whether CoLA solver adapters can
extract answers when distractor evidence is removed.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_ONLINE_INPUTS_JSONL = (
    "/data1/luyifei/drla/outputs/p2_phase_c_control_inputs/"
    "musique_calibration_controls_200_seed20260601_v1_strict_wrong/online_inputs.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p2_phase_c_control_inputs/"
    "musique_calibration_solver_support_only_diag_200_seed20260606"
)


def main() -> None:
    summary = build_controls(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-inputs-jsonl", default=DEFAULT_ONLINE_INPUTS_JSONL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples in source controls.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_controls(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = read_jsonl(Path(args.online_inputs_jsonl))
    by_sample: dict[str, dict[str, dict[str, Any]]] = {}
    for row in source_rows:
        condition = str(row.get("condition", ""))
        if condition not in {"single_q_only", "single_full_info"}:
            continue
        by_sample.setdefault(str(row["sample_id"]), {})[condition] = row

    rows: list[dict[str, Any]] = []
    drops = Counter()
    for sample_id, grouped in by_sample.items():
        if args.max_samples and len(rows) >= args.max_samples * 2:
            break
        q_row = grouped.get("single_q_only")
        full_row = grouped.get("single_full_info")
        if q_row is None or full_row is None:
            drops["missing_q_or_full_row"] += 1
            continue
        support_evidence = support_only_evidence(
            str(full_row.get("online_input_fields", {}).get("full_evidence", ""))
        )
        if not support_evidence:
            drops["missing_support_evidence"] += 1
            continue
        rows.append(rewrite_row(q_row, sample_id=sample_id, suffix="q_only", support_evidence=""))
        rows.append(
            rewrite_row(
                full_row,
                sample_id=sample_id,
                suffix="support_only",
                support_evidence=support_evidence,
            )
        )

    output_jsonl = output_dir / "online_inputs.jsonl"
    write_jsonl(output_jsonl, rows)
    condition_counts = Counter(str(row["condition"]) for row in rows)
    summary = {
        "created_at": int(time.time()),
        "status": "pass",
        "source_online_inputs_jsonl": args.online_inputs_jsonl,
        "output_dir": str(output_dir),
        "online_inputs_jsonl": str(output_jsonl),
        "num_rows": len(rows),
        "num_samples": len({row["sample_id"] for row in rows}),
        "condition_counts": dict(sorted(condition_counts.items())),
        "drops": dict(sorted(drops.items())),
        "diagnostic_context_mode": "support_only_solver_single_full_info",
        "execution_boundary": [
            "local-only diagnostic control construction",
            "no model generation",
            "no optimizer or backward",
            "no SwanLab run",
            "calibration/train-style controls only; do not use for held-out repair",
            "not a locked Phase C or Phase A gate",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.jsonl").write_text(
        json.dumps(
            {
                "num_rows": len(rows),
                "num_samples": summary["num_samples"],
                **{f"condition_count/{k}": v for k, v in condition_counts.items()},
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def support_only_evidence(full_evidence: str) -> str:
    lines = []
    for line in full_evidence.splitlines():
        if re.search(r"\(\s*support\s*\)", line, flags=re.IGNORECASE):
            lines.append(line.strip())
    return "\n".join(line for line in lines if line)


def rewrite_row(row: dict[str, Any], *, sample_id: str, suffix: str, support_evidence: str) -> dict[str, Any]:
    rewritten = json.loads(json.dumps(row))
    rewritten["row_id"] = f"{row['row_id']}__support_only_diag_{suffix}"
    fields = rewritten.setdefault("online_input_fields", {})
    fields["diagnostic_context_mode"] = "support_only_solver"
    if rewritten.get("condition") == "single_full_info":
        fields["full_evidence"] = support_evidence
    fields["support_only_diag_source_sample_id"] = sample_id
    fields["support_only_diag_note"] = (
        "Diagnostic only: full_evidence contains support-labelled lines and omits distractors."
    )
    return rewritten


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{line_no}")
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
