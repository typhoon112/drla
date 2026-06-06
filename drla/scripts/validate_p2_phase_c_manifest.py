"""Validate a P2 Phase C benchmark manifest.

This is a local-only hygiene script.  It does not run models, inspect held-out
generations, train adapters, or create SwanLab runs.  It validates that a
candidate Phase C manifest follows the documented communication benchmark
contract before any Branch C execution is allowed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = "/data1/luyifei/drla/outputs/p2_phase_c_manifest_audits/manifest_audit_20260601"

REQUIRED_BASELINES = {
    "single_q_only",
    "single_full_info",
    "textmas_matched",
    "textmas_no_message",
    "textmas_shuffled_message",
    "textmas_wrong_evidence_or_wrong_shard",
}

ALLOWED_FAMILIES = {
    "evidence_split_qa",
    "distributed_state_synthesis",
    "code_workflow",
}

ALLOWED_SPLITS = {
    "calibration",
    "heldout",
    "train",
    "valid",
    "test",
}

ALLOWED_SCORE_TYPES = {
    "exact_match",
    "normalized_f1",
    "multiple_choice",
    "unit_test",
    "custom",
}


def main() -> None:
    summary = validate_manifest(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest_json)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    require_object(manifest, "manifest", errors)
    samples = manifest.get("samples", []) if isinstance(manifest, dict) else []
    if not isinstance(samples, list) or not samples:
        errors.append({"path": "samples", "message": "samples must be a non-empty list"})
        samples = []

    sample_ids: set[str] = set()
    split_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    baseline_missing_counts: dict[str, int] = {}

    validate_top_level(manifest, errors, warnings)
    for index, sample in enumerate(samples):
        path = f"samples[{index}]"
        if not isinstance(sample, dict):
            errors.append({"path": path, "message": "sample must be an object"})
            continue
        sample_id = str(sample.get("sample_id", ""))
        if not sample_id:
            errors.append({"path": f"{path}.sample_id", "message": "sample_id is required"})
        elif sample_id in sample_ids:
            errors.append({"path": f"{path}.sample_id", "message": f"duplicate sample_id: {sample_id}"})
        sample_ids.add(sample_id)
        family = str(sample.get("family", ""))
        split = str(sample.get("split", ""))
        task_name = str(sample.get("task_name", ""))
        family_counts[family] = family_counts.get(family, 0) + 1
        split_counts[split] = split_counts.get(split, 0) + 1
        task_counts[task_name] = task_counts.get(task_name, 0) + 1
        validate_sample(sample, path, errors, warnings)
        missing = sorted(REQUIRED_BASELINES - set(sample.get("baselines_required", [])))
        for baseline in missing:
            baseline_missing_counts[baseline] = baseline_missing_counts.get(baseline, 0) + 1

    if "calibration" not in split_counts:
        warnings.append(
            {
                "path": "samples",
                "message": "no calibration split found; prompt/protocol repair requires calibration rows",
            }
        )
    if "heldout" not in split_counts and "test" not in split_counts:
        warnings.append(
            {
                "path": "samples",
                "message": "no heldout/test split found; paper-level reporting requires locked evaluation rows",
            }
        )

    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    metrics = {
        "num_samples": len(samples),
        "num_errors": len(errors),
        "num_warnings": len(warnings),
        "num_unique_sample_ids": len(sample_ids),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "created_at": int(time.time()),
        "manifest_json": str(manifest_path),
        "status": "pass" if not errors else "fail",
        "num_samples": len(samples),
        "num_errors": len(errors),
        "num_warnings": len(warnings),
        "split_counts": split_counts,
        "family_counts": family_counts,
        "task_counts": task_counts,
        "baseline_missing_counts": baseline_missing_counts,
        "errors": errors,
        "warnings": warnings,
        "metrics_jsonl": str(metrics_path),
        "execution_boundary": [
            "local-only manifest validation",
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


def validate_top_level(manifest: Any, errors: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> None:
    if not isinstance(manifest, dict):
        return
    required = ["manifest_version", "protocol_version", "created_at_utc", "families", "samples"]
    for key in required:
        if key not in manifest:
            errors.append({"path": key, "message": "top-level field is required"})
    if manifest.get("manifest_version") != "p2_phase_c_manifest_v0":
        errors.append(
            {
                "path": "manifest_version",
                "message": "manifest_version must be p2_phase_c_manifest_v0",
            }
        )
    families = manifest.get("families", [])
    if not isinstance(families, list) or not families:
        errors.append({"path": "families", "message": "families must be a non-empty list"})
    else:
        for family in families:
            if family not in ALLOWED_FAMILIES:
                errors.append({"path": "families", "message": f"unknown family: {family}"})
    if "split_seed" not in manifest:
        warnings.append({"path": "split_seed", "message": "split_seed is recommended for reproducibility"})


def validate_sample(
    sample: dict[str, Any],
    path: str,
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    required = [
        "sample_id",
        "family",
        "task_name",
        "split",
        "source",
        "question",
        "agent_views",
        "scoring",
        "leakage_audit",
        "baselines_required",
    ]
    for key in required:
        if key not in sample:
            errors.append({"path": f"{path}.{key}", "message": "field is required"})
    if sample.get("family") not in ALLOWED_FAMILIES:
        errors.append({"path": f"{path}.family", "message": f"unknown family: {sample.get('family')}"})
    if sample.get("split") not in ALLOWED_SPLITS:
        errors.append({"path": f"{path}.split", "message": f"unknown split: {sample.get('split')}"})
    validate_source(sample.get("source"), f"{path}.source", errors)
    validate_agent_views(sample, path, errors, warnings)
    validate_scoring(sample, path, errors, warnings)
    validate_leakage(sample.get("leakage_audit"), f"{path}.leakage_audit", errors)
    baselines = sample.get("baselines_required", [])
    if not isinstance(baselines, list):
        errors.append({"path": f"{path}.baselines_required", "message": "must be a list"})
    else:
        missing = sorted(REQUIRED_BASELINES - set(baselines))
        if missing:
            errors.append(
                {
                    "path": f"{path}.baselines_required",
                    "message": f"missing required baselines: {', '.join(missing)}",
                }
            )


def validate_source(source: Any, path: str, errors: list[dict[str, Any]]) -> None:
    if not isinstance(source, dict):
        errors.append({"path": path, "message": "source must be an object"})
        return
    for key in ("name", "version"):
        if not source.get(key):
            errors.append({"path": f"{path}.{key}", "message": "field is required"})


def validate_agent_views(
    sample: dict[str, Any],
    path: str,
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    views = sample.get("agent_views", [])
    if not isinstance(views, list) or not views:
        errors.append({"path": f"{path}.agent_views", "message": "agent_views must be a non-empty list"})
        return
    agent_ids: set[str] = set()
    gold_answer_text = normalized_text(sample.get("scoring", {}).get("gold_answer", ""))
    full_context_text = normalized_text(sample.get("public_context", ""))
    for index, view in enumerate(views):
        view_path = f"{path}.agent_views[{index}]"
        if not isinstance(view, dict):
            errors.append({"path": view_path, "message": "agent view must be an object"})
            continue
        for key in ("agent_id", "role", "private_observation", "allowed_output_contract", "forbidden_fields"):
            if key not in view:
                errors.append({"path": f"{view_path}.{key}", "message": "field is required"})
        agent_id = str(view.get("agent_id", ""))
        if agent_id in agent_ids:
            errors.append({"path": f"{view_path}.agent_id", "message": f"duplicate agent_id: {agent_id}"})
        agent_ids.add(agent_id)
        forbidden = view.get("forbidden_fields", [])
        if not isinstance(forbidden, list):
            errors.append({"path": f"{view_path}.forbidden_fields", "message": "must be a list"})
        elif "gold_answer" not in forbidden:
            errors.append({"path": f"{view_path}.forbidden_fields", "message": "must include gold_answer"})
        private_text = normalized_text(view.get("private_observation", ""))
        if gold_answer_text and gold_answer_text in private_text:
            warnings.append(
                {
                    "path": f"{view_path}.private_observation",
                    "message": "gold answer string appears in private observation; verify this is not leakage",
                }
            )
        if full_context_text and private_text and private_text == full_context_text:
            warnings.append(
                {
                    "path": f"{view_path}.private_observation",
                    "message": "private observation equals public_context; partial-information value may be weak",
                }
            )


def validate_scoring(
    sample: dict[str, Any],
    path: str,
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    scoring = sample.get("scoring")
    if not isinstance(scoring, dict):
        errors.append({"path": f"{path}.scoring", "message": "scoring must be an object"})
        return
    score_type = scoring.get("type")
    if score_type not in ALLOWED_SCORE_TYPES:
        errors.append({"path": f"{path}.scoring.type", "message": f"unknown score type: {score_type}"})
    if "gold_answer" not in scoring:
        errors.append({"path": f"{path}.scoring.gold_answer", "message": "field is required"})
    if score_type == "unit_test" and not scoring.get("unit_tests"):
        warnings.append({"path": f"{path}.scoring.unit_tests", "message": "unit_test scorer should include tests"})
    if score_type == "custom" and not scoring.get("custom_scorer"):
        errors.append({"path": f"{path}.scoring.custom_scorer", "message": "custom scorer path/name is required"})


def validate_leakage(leakage: Any, path: str, errors: list[dict[str, Any]]) -> None:
    if not isinstance(leakage, dict):
        errors.append({"path": path, "message": "leakage_audit must be an object"})
        return
    false_fields = [
        "gold_in_online_prompt",
        "scorer_output_in_online_prompt",
        "full_evidence_available_to_split_agent",
        "heldout_used_for_prompt_repair",
    ]
    for field in false_fields:
        if leakage.get(field) is not False:
            errors.append({"path": f"{path}.{field}", "message": "must be false"})


def require_object(value: Any, path: str, errors: list[dict[str, Any]]) -> None:
    if not isinstance(value, dict):
        errors.append({"path": path, "message": "must be an object"})


def normalized_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return " ".join(value.lower().strip().split())


if __name__ == "__main__":
    main()
