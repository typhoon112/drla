"""Freeze the P3 Dream TextMAS protocol after calibration gate admission.

This local-only utility records the exact prompt/parser/control/model/generation
configuration and source artifact hashes that are allowed for the one-shot
held-out evaluation. It does not run models, train, or create SwanLab runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any


def main() -> None:
    summary = freeze(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-run-summary-json", required=True)
    parser.add_argument("--calibration-aggregate-summary-json", required=True)
    parser.add_argument("--calibration-leakage-summary-json", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--online-inputs-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_summary = read_json(Path(args.calibration_run_summary_json))
    config_source_summary = resolve_config_source_summary(run_summary)
    aggregate_summary = read_json(Path(args.calibration_aggregate_summary_json))
    leakage_summary = read_json(Path(args.calibration_leakage_summary_json))

    failures = []
    if aggregate_summary.get("status") != "pass" or not aggregate_summary.get("admitted"):
        failures.append("calibration aggregate did not admit protocol")
    if aggregate_summary.get("gate", {}).get("failed_gates"):
        failures.append("calibration aggregate has failed gates")
    if leakage_summary.get("status") != "pass" or int(leakage_summary.get("num_errors", 0)) != 0:
        failures.append("leakage audit did not pass with zero errors")
    run_config = config_source_summary.get("run_config", {})
    generation_config = {
        key: run_config.get(key)
        for key in [
            "dtype",
            "max_tokens",
            "dream_steps",
            "temperature",
            "top_p",
            "alg",
            "alg_temp",
            "max_context_tokens",
        ]
        if key in run_config
    }

    protocol_lock = {
        "created_at": int(time.time()),
        "status": "locked" if not failures else "failed",
        "failures": failures,
        "model": config_source_summary.get("model", ""),
        "model_path": config_source_summary.get("model_path", ""),
        "provider": config_source_summary.get("provider", ""),
        "generation_config": generation_config,
        "shard_run_config_example": run_config,
        "parser": {
            "prediction_extraction_mode": config_source_summary.get("run_config", {}).get(
                "prediction_extraction_mode", ""
            ),
            "scorer": "drla.evaluation.p2_phase_c_scorers.score_qa_answer",
        },
        "calibration_artifacts": {
            "run_summary_json": args.calibration_run_summary_json,
            "aggregate_summary_json": args.calibration_aggregate_summary_json,
            "leakage_summary_json": args.calibration_leakage_summary_json,
            "generations_jsonl": run_summary.get("generations_jsonl", ""),
            "config_source_summary_json": config_source_summary.get("summary_json", args.calibration_run_summary_json),
            "aggregate_metrics_jsonl": aggregate_summary.get("metrics_jsonl", ""),
            "leakage_audit_json": leakage_summary.get("leakage_audit_json", ""),
        },
        "source_inputs": {
            "manifest_json": args.manifest_json,
            "online_inputs_jsonl": args.online_inputs_jsonl,
            "manifest_sha256": sha256(Path(args.manifest_json)),
            "online_inputs_sha256": sha256(Path(args.online_inputs_jsonl)),
        },
        "calibration_metrics": {
            "condition_metrics": aggregate_summary.get("condition_metrics", {}),
            "paired_metrics": aggregate_summary.get("paired_metrics", {}),
            "num_leakage_errors": leakage_summary.get("num_errors", None),
            "num_leakage_warnings": leakage_summary.get("num_warnings", None),
        },
        "heldout_rules": [
            "Use the same model_path, provider, generation_config, parser, scorer, and control conditions.",
            "Do not modify prompt, parser, thresholds, row filtering, or control definitions after this lock.",
            "Held-out evaluation is one-shot and cannot be used for prompt repair or adapter selection.",
            "Evaluation and aggregation stay local-only; no SwanLab run because there is no training.",
        ],
        "execution_boundary": [
            "local-only P3 protocol freeze",
            "no model generation",
            "no optimizer or backward",
            "no SwanLab run",
        ],
    }
    protocol_lock_path = output_dir / "protocol_lock.json"
    summary_path = output_dir / "summary.json"
    metrics_path = output_dir / "metrics.jsonl"
    protocol_lock_path.write_text(
        json.dumps(protocol_lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics = {
        "status_locked": int(not failures),
        "num_failures": len(failures),
        "manifest_sha256_prefix": protocol_lock["source_inputs"]["manifest_sha256"][:12],
        "online_inputs_sha256_prefix": protocol_lock["source_inputs"]["online_inputs_sha256"][:12],
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "created_at": protocol_lock["created_at"],
        "status": "pass" if not failures else "fail",
        "protocol_lock_json": str(protocol_lock_path),
        "metrics_jsonl": str(metrics_path),
        "failures": failures,
        "model_path": protocol_lock["model_path"],
        "generation_config": protocol_lock["generation_config"],
        "shard_run_config_example": protocol_lock["shard_run_config_example"],
        "heldout_rules": protocol_lock["heldout_rules"],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_json"] = str(summary_path)
    if failures:
        raise RuntimeError("; ".join(failures))
    return summary


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object in {path}")
    return value


def resolve_config_source_summary(run_summary: dict[str, Any]) -> dict[str, Any]:
    if run_summary.get("run_config") and run_summary.get("model_path"):
        return run_summary
    for item in run_summary.get("input_summaries", []):
        if not isinstance(item, dict):
            continue
        summary_json = item.get("summary_json", "")
        if not summary_json:
            continue
        candidate = read_json(Path(summary_json))
        if candidate.get("run_config") and candidate.get("model_path"):
            candidate["summary_json"] = summary_json
            return candidate
    return run_summary


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
