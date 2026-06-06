"""Audit P2 Phase C run artifacts for online-input leakage.

This is a local-only artifact checker.  It reads a Phase C manifest and future
``generations.jsonl`` rows, then verifies that online input fields do not
contain explicit gold/scorer/full-evidence fields in conditions where they are
forbidden.  It does not run models, train adapters, inspect hidden held-out
generations beyond provided artifact rows, or create SwanLab runs.
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

from drla.evaluation.p2_phase_c_scorers import normalize_qa_text


DEFAULT_OUTPUT_DIR = "/data1/luyifei/drla/outputs/p2_phase_c_leakage_audits/run_leakage_audit_20260601"

EXPLICIT_GOLD_KEYS = {
    "answer",
    "answer_aliases",
    "correct_answer",
    "gold",
    "gold_answer",
    "ground_truth",
    "label",
    "oracle_answer",
    "target",
}

EXPLICIT_SCORER_KEYS = {
    "official_correctness",
    "official_score",
    "score",
    "scored_prediction",
    "scorer_output",
    "selected_prediction",
}

FULL_EVIDENCE_KEYS = {
    "all_evidence",
    "all_private_observations",
    "full_context",
    "full_evidence",
    "full_evidence_union",
}

FULL_EVIDENCE_ALLOWED_CONDITIONS = {
    "single_full_info",
}

CONTROL_CONDITIONS = {
    "textmas_shuffled_message",
    "textmas_wrong_evidence_or_wrong_shard",
}


def main() -> None:
    args = parse_args()
    summary = run_selfcheck(args) if args.selfcheck else audit_leakage(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", default="")
    parser.add_argument("--generations-jsonl", default="")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if not args.selfcheck and (not args.manifest_json or not args.generations_jsonl):
        raise ValueError("Pass --manifest-json and --generations-jsonl, or use --selfcheck")
    return args


def audit_leakage(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
    rows = read_jsonl(Path(args.generations_jsonl))
    return write_audit_summary(
        output_dir=Path(args.output_dir),
        overwrite=args.overwrite,
        manifest_json=args.manifest_json,
        generations_jsonl=args.generations_jsonl,
        audit=audit_rows(manifest, rows),
    )


def run_selfcheck(args: argparse.Namespace) -> dict[str, Any]:
    manifest = {
        "samples": [
            {
                "sample_id": "safe_sample",
                "split": "calibration",
                "question": "Which color is implied?",
                "public_context": "",
                "agent_views": [
                    {
                        "agent_id": "agent_a",
                        "private_observation": "The clue refers to a clear daytime sky.",
                    }
                ],
                "scoring": {
                    "gold_answer": "blue",
                    "answer_aliases": ["the color blue"],
                },
            },
            {
                "sample_id": "heldout_sample",
                "split": "heldout",
                "question": "Which color is implied?",
                "public_context": "",
                "agent_views": [
                    {
                        "agent_id": "agent_a",
                        "private_observation": "The clue refers to ripe strawberries.",
                    }
                ],
                "scoring": {
                    "gold_answer": "red",
                    "answer_aliases": ["the color red"],
                },
            },
        ]
    }
    safe_rows = [
        {
            "sample_id": "safe_sample",
            "condition": "textmas_matched",
            "online_input_fields": {
                "question": "Which color is implied?",
                "agent_a_private_observation": "The clue refers to a clear daytime sky.",
            },
        },
        {
            "sample_id": "safe_sample",
            "condition": "textmas_shuffled_message",
            "control_source_sample_id": "other_sample",
            "online_input_fields": {
                "question": "Which color is implied?",
                "message": "A shuffled message from another sample.",
            },
        },
    ]
    unsafe_rows = [
        {
            "sample_id": "safe_sample",
            "condition": "textmas_matched",
            "online_input_fields": {
                "question": "Which color is implied?",
                "gold_answer": "blue",
            },
        },
        {
            "sample_id": "heldout_sample",
            "condition": "textmas_matched",
            "used_for_prompt_repair": True,
            "online_input_fields": {
                "question": "Which color is implied?",
            },
        },
    ]
    safe_audit = audit_rows(manifest, safe_rows)
    unsafe_audit = audit_rows(manifest, unsafe_rows)
    checks = [
        ("safe_rows_pass", safe_audit["num_errors"] == 0),
        ("unsafe_rows_fail", unsafe_audit["num_errors"] >= 2),
    ]
    failed = [name for name, passed in checks if not passed]
    audit = {
        "status": "pass" if not failed else "fail",
        "num_rows": len(safe_rows) + len(unsafe_rows),
        "num_errors": len(failed),
        "num_warnings": 0,
        "checks": [{"name": name, "passed": passed} for name, passed in checks],
        "errors": [{"path": "selfcheck", "message": f"failed check: {name}"} for name in failed],
        "warnings": [],
        "safe_audit": safe_audit,
        "unsafe_audit": unsafe_audit,
    }
    summary = write_audit_summary(
        output_dir=Path(args.output_dir),
        overwrite=args.overwrite,
        manifest_json="<selfcheck>",
        generations_jsonl="<selfcheck>",
        audit=audit,
    )
    if failed:
        raise AssertionError(f"P2 Phase C leakage self-check failed: {failed}")
    return summary


def audit_rows(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = {str(sample.get("sample_id", "")): sample for sample in manifest.get("samples", [])}
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    condition_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for index, row in enumerate(rows):
        path = f"generations[{index}]"
        if not isinstance(row, dict):
            errors.append({"path": path, "message": "generation row must be an object"})
            continue
        sample_id = str(row.get("sample_id", ""))
        condition = str(row.get("condition", ""))
        condition_counts[condition] = condition_counts.get(condition, 0) + 1
        sample = samples.get(sample_id)
        if sample is None:
            errors.append({"path": f"{path}.sample_id", "message": f"unknown sample_id: {sample_id}"})
            continue
        split = str(row.get("split", sample.get("split", "")))
        split_counts[split] = split_counts.get(split, 0) + 1
        fields = row.get("online_input_fields")
        if not isinstance(fields, dict):
            errors.append(
                {
                    "path": f"{path}.online_input_fields",
                    "message": "online_input_fields object is required for leakage audit",
                }
            )
            continue
        audit_explicit_fields(fields, path, condition, errors)
        audit_control_source(row, path, sample_id, condition, errors)
        audit_heldout_usage(row, path, sample, split, errors)
        audit_content_warnings(fields, path, sample, warnings)
    return {
        "status": "pass" if not errors else "fail",
        "num_rows": len(rows),
        "num_errors": len(errors),
        "num_warnings": len(warnings),
        "condition_counts": condition_counts,
        "split_counts": split_counts,
        "errors": errors,
        "warnings": warnings,
    }


def audit_explicit_fields(
    fields: dict[str, Any],
    path: str,
    condition: str,
    errors: list[dict[str, Any]],
) -> None:
    flat_keys = {key.lower() for key in flatten_field_keys(fields)}
    for key in sorted(flat_keys & EXPLICIT_GOLD_KEYS):
        errors.append({"path": f"{path}.online_input_fields.{key}", "message": "explicit gold field in online input"})
    for key in sorted(flat_keys & EXPLICIT_SCORER_KEYS):
        errors.append(
            {"path": f"{path}.online_input_fields.{key}", "message": "explicit scorer/eval field in online input"}
        )
    if condition not in FULL_EVIDENCE_ALLOWED_CONDITIONS:
        for key in sorted(flat_keys & FULL_EVIDENCE_KEYS):
            errors.append(
                {
                    "path": f"{path}.online_input_fields.{key}",
                    "message": f"full evidence field is forbidden for condition {condition}",
                }
            )


def audit_control_source(
    row: dict[str, Any],
    path: str,
    sample_id: str,
    condition: str,
    errors: list[dict[str, Any]],
) -> None:
    if condition not in CONTROL_CONDITIONS:
        return
    control_id = row.get("control_source_sample_id") or row.get("wrong_evidence_sample_id")
    if not control_id:
        errors.append(
            {
                "path": f"{path}.control_source_sample_id",
                "message": f"{condition} requires a non-self control source sample id",
            }
        )
    elif str(control_id) == sample_id:
        errors.append(
            {
                "path": f"{path}.control_source_sample_id",
                "message": f"{condition} control source must not equal sample_id",
            }
        )


def audit_heldout_usage(
    row: dict[str, Any],
    path: str,
    sample: dict[str, Any],
    split: str,
    errors: list[dict[str, Any]],
) -> None:
    sample_split = str(sample.get("split", split))
    if (split == "heldout" or sample_split == "heldout") and row.get("used_for_prompt_repair") is True:
        errors.append(
            {
                "path": f"{path}.used_for_prompt_repair",
                "message": "heldout row is marked as used for prompt repair",
            }
        )


def audit_content_warnings(
    fields: dict[str, Any],
    path: str,
    sample: dict[str, Any],
    warnings: list[dict[str, Any]],
) -> None:
    text = normalize_qa_text(json.dumps(fields, ensure_ascii=False, sort_keys=True))
    scoring = sample.get("scoring", {})
    terms = [scoring.get("gold_answer", "")]
    terms.extend(scoring.get("answer_aliases", []) or [])
    for term in terms:
        normalized = normalize_qa_text(term)
        if normalized and normalized in text:
            warnings.append(
                {
                    "path": f"{path}.online_input_fields",
                    "message": "gold/alias string appears in online input text; verify this is evidence, not leakage",
                }
            )
            break


def flatten_field_keys(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        keys: list[str] = []
        for key, item in value.items():
            key_text = str(key)
            full_key = f"{prefix}.{key_text}" if prefix else key_text
            keys.append(key_text)
            keys.append(full_key)
            keys.extend(flatten_field_keys(item, full_key))
        return keys
    if isinstance(value, list):
        keys = []
        for index, item in enumerate(value):
            keys.extend(flatten_field_keys(item, f"{prefix}[{index}]"))
        return keys
    return []


def write_audit_summary(
    *,
    output_dir: Path,
    overwrite: bool,
    manifest_json: str,
    generations_jsonl: str,
    audit: dict[str, Any],
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    audit_path = output_dir / "leakage_audit.json"
    summary_path = output_dir / "summary.json"
    metrics = {
        "num_rows": audit["num_rows"],
        "num_errors": audit["num_errors"],
        "num_warnings": audit["num_warnings"],
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "created_at": int(time.time()),
        "status": audit["status"],
        "manifest_json": manifest_json,
        "generations_jsonl": generations_jsonl,
        "num_rows": audit["num_rows"],
        "num_errors": audit["num_errors"],
        "num_warnings": audit["num_warnings"],
        "metrics_jsonl": str(metrics_path),
        "leakage_audit_json": str(audit_path),
        "execution_boundary": [
            "local-only leakage audit",
            "no model generation",
            "no optimizer or backward",
            "no SwanLab run",
            "no hidden held-out inspection beyond provided artifact rows",
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


if __name__ == "__main__":
    main()
