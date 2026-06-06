"""Build Phase C Single/TextMAS/control online-input packages.

This local-only script reads a validated P2 Phase C manifest and materializes
the online inputs for the required baseline/control conditions.  It does not
call any model, train adapters, inspect held-out generations, or create SwanLab
runs.  The output JSONL is intentionally compatible with
``audit_p2_phase_c_run_leakage.py`` so protocol leakage can be checked before
any capable text-agent run.
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

from drla.scripts.audit_p2_phase_c_run_leakage import audit_rows, write_audit_summary


DEFAULT_OUTPUT_DIR = "/data1/luyifei/drla/outputs/p2_phase_c_control_inputs/control_inputs_20260601"

CONDITIONS = [
    "single_q_only",
    "single_full_info",
    "textmas_matched",
    "textmas_no_message",
    "textmas_shuffled_message",
    "textmas_wrong_evidence_or_wrong_shard",
    "textmas_compressed_state",
]


def main() -> None:
    summary = build_control_inputs(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt-contract-version", default="p2_phase_c_evidence_split_v1_strict_wrong_evidence")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_control_inputs(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest_json)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples", [])
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"manifest has no samples: {manifest_path}")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    control_map = build_control_map(samples)
    rows = []
    for sample in samples:
        for condition in CONDITIONS:
            rows.append(make_condition_row(sample, condition, control_map, args.prompt_contract_version))

    online_inputs_path = output_dir / "online_inputs.jsonl"
    prompts_path = output_dir / "prompt_preview.jsonl"
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(online_inputs_path, rows)
    write_prompt_preview(prompts_path, rows)

    leakage_summary = write_audit_summary(
        output_dir=output_dir / "leakage_audit",
        overwrite=True,
        manifest_json=str(manifest_path),
        generations_jsonl=str(online_inputs_path),
        audit=audit_rows(manifest, rows),
    )
    condition_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for row in rows:
        condition_counts[row["condition"]] = condition_counts.get(row["condition"], 0) + 1
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    metrics = {
        "num_samples": len(samples),
        "num_rows": len(rows),
        "num_conditions": len(condition_counts),
        "leakage_errors": leakage_summary["num_errors"],
        "leakage_warnings": leakage_summary["num_warnings"],
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "created_at": int(time.time()),
        "status": "pass" if leakage_summary["status"] == "pass" else "fail",
        "manifest_json": str(manifest_path),
        "online_inputs_jsonl": str(online_inputs_path),
        "prompt_preview_jsonl": str(prompts_path),
        "metrics_jsonl": str(metrics_path),
        "leakage_audit_summary_json": leakage_summary["summary_json"],
        "num_samples": len(samples),
        "num_rows": len(rows),
        "condition_counts": condition_counts,
        "split_counts": split_counts,
        "prompt_contract_version": args.prompt_contract_version,
        "execution_boundary": [
            "local-only online-input/control package construction",
            "no model generation",
            "no optimizer or backward",
            "no SwanLab run",
            "no held-out prompt repair",
            "leakage audit runs on constructed online_input_fields",
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


def build_control_map(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for sample in samples:
        key = (str(sample.get("task_name", "")), str(sample.get("split", "")))
        buckets.setdefault(key, []).append(sample)
    control_map: dict[str, dict[str, Any]] = {}
    for bucket_samples in buckets.values():
        if len(bucket_samples) < 2:
            continue
        ordered = sorted(bucket_samples, key=lambda item: str(item.get("sample_id", "")))
        for index, sample in enumerate(ordered):
            control_map[str(sample["sample_id"])] = ordered[(index + 1) % len(ordered)]
    return control_map


def make_condition_row(
    sample: dict[str, Any],
    condition: str,
    control_map: dict[str, dict[str, Any]],
    prompt_contract_version: str,
) -> dict[str, Any]:
    sample_id = str(sample["sample_id"])
    control_sample = control_map.get(sample_id)
    if condition in {"textmas_shuffled_message", "textmas_wrong_evidence_or_wrong_shard"} and control_sample is None:
        raise ValueError(f"condition {condition} requires at least two samples in same task/split")
    row = {
        "row_id": f"{sample_id}::{condition}",
        "sample_id": sample_id,
        "split": sample.get("split", ""),
        "task_name": sample.get("task_name", ""),
        "condition": condition,
        "prompt_contract_version": prompt_contract_version,
        "used_for_prompt_repair": False,
        "online_input_fields": build_online_fields(sample, condition, control_sample),
        "prompt_messages": build_prompt_messages(sample, condition, control_sample),
        "expected_output_contract": expected_output_contract(condition),
    }
    if control_sample is not None and condition == "textmas_shuffled_message":
        row["control_source_sample_id"] = control_sample["sample_id"]
    if control_sample is not None and condition == "textmas_wrong_evidence_or_wrong_shard":
        row["wrong_evidence_sample_id"] = control_sample["sample_id"]
    return row


def build_online_fields(
    sample: dict[str, Any],
    condition: str,
    control_sample: dict[str, Any] | None,
) -> dict[str, Any]:
    question = sample["question"]
    public_context = sample.get("public_context", "")
    if condition == "single_q_only":
        return {"question": question, "public_context": public_context}
    if condition == "single_full_info":
        return {
            "question": question,
            "public_context": public_context,
            "full_evidence": full_evidence(sample),
        }
    if condition == "textmas_matched":
        return {
            "question": question,
            "public_context": public_context,
            "agent_private_observations": agent_private_observations(sample),
            "solver_message_contract": "Use generated messages from the same sample's evidence agents.",
        }
    if condition == "textmas_no_message":
        return {
            "question": question,
            "public_context": public_context,
            "upstream_messages": [],
        }
    if condition == "textmas_shuffled_message":
        assert control_sample is not None
        return {
            "question": question,
            "public_context": public_context,
            "shuffled_message_source_sample_id": control_sample["sample_id"],
            "solver_message_contract": "Use generated messages from the control sample, not this sample.",
        }
    if condition == "textmas_wrong_evidence_or_wrong_shard":
        assert control_sample is not None
        control_views = agent_private_observations(control_sample)
        wrong_views = [
            {**view, "control_source_sample_id": control_sample["sample_id"]}
            for view in control_views
        ]
        return {
            "question": question,
            "public_context": public_context,
            "agent_private_observations": wrong_views,
            "wrong_evidence_source_sample_id": control_sample["sample_id"],
            "solver_message_contract": "All evidence agents receive irrelevant private shards from the control sample.",
        }
    if condition == "textmas_compressed_state":
        return {
            "question": question,
            "public_context": public_context,
            "agent_private_observations": agent_private_observations(sample),
            "agent_output_schema": {
                "useful_facts": "array of concise evidence facts",
                "uncertainty": "short uncertainty note",
                "answer_hint": "optional, only if directly supported by the private shard",
            },
        }
    raise ValueError(f"unknown condition: {condition}")


def build_prompt_messages(
    sample: dict[str, Any],
    condition: str,
    control_sample: dict[str, Any] | None,
) -> list[dict[str, str]]:
    fields = build_online_fields(sample, condition, control_sample)
    system = (
        "You are participating in an evidence-split multi-agent QA protocol. "
        "Use only the online input fields shown in the user message. Do not use "
        "gold answers or scorer outputs."
    )
    user = json.dumps(fields, ensure_ascii=False, indent=2, sort_keys=True)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def expected_output_contract(condition: str) -> str:
    if condition.startswith("single"):
        return "Return the final answer only, with minimal supporting rationale if needed."
    if condition == "textmas_compressed_state":
        return "Evidence agents must emit compact typed state; final solver returns the final answer."
    return "Evidence agents emit messages; final solver returns the final answer."


def agent_private_observations(sample: dict[str, Any]) -> list[dict[str, str]]:
    observations = []
    for view in sample.get("agent_views", []):
        observations.append(
            {
                "agent_id": str(view.get("agent_id", "")),
                "role": str(view.get("role", "")),
                "private_observation": str(view.get("private_observation", "")),
                "allowed_output_contract": str(view.get("allowed_output_contract", "")),
            }
        )
    return observations


def full_evidence(sample: dict[str, Any]) -> str:
    metadata = sample.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("full_info_observation"):
        return str(metadata["full_info_observation"])
    return "\n".join(view["private_observation"] for view in agent_private_observations(sample))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_prompt_preview(path: Path, rows: list[dict[str, Any]], max_rows: int = 20) -> None:
    preview_rows = [
        {
            "row_id": row["row_id"],
            "sample_id": row["sample_id"],
            "condition": row["condition"],
            "prompt_messages": row["prompt_messages"],
        }
        for row in rows[:max_rows]
    ]
    write_jsonl(path, preview_rows)


if __name__ == "__main__":
    main()
