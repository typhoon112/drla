"""Audit static Dream TextMAS prompt lengths.

This local-only preflight uses the Dream tokenizer to measure prompt lengths
that are known before generation. It catches static overflows such as
``single_full_info`` full-evidence prompts and evidence-agent private-observation
prompts. It does not claim to cover solver prompts that include model-generated
upstream messages, because those lengths are dynamic.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.scripts.run_p2_phase_c_text_agents import (  # noqa: E402
    make_agent_messages,
    make_solver_messages,
    read_jsonl,
    sample_agent_observations,
)


DEFAULT_MODEL_PATH = "/data1/luyifei/drla/models/Dream-v0-Instruct-7B"


def main() -> None:
    summary = run_audit(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--online-inputs-jsonl", required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-context-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
    samples = {str(sample["sample_id"]): sample for sample in manifest.get("samples", [])}
    rows = read_jsonl(Path(args.online_inputs_jsonl))

    call_rows: list[dict[str, Any]] = []
    for row in rows:
        call_rows.extend(static_calls_for_row(row, samples))

    measured = []
    for item in call_rows:
        messages = item.pop("messages")
        input_tokens = count_chat_tokens(tokenizer, messages)
        overflow_margin = input_tokens + args.max_tokens - args.max_context_tokens
        item.update(
            {
                "input_tokens": input_tokens,
                "total_with_max_tokens": input_tokens + args.max_tokens,
                "overflow": overflow_margin > 0,
                "overflow_margin": overflow_margin,
            }
        )
        measured.append(item)

    overflows = [row for row in measured if row["overflow"]]
    measured.sort(key=lambda row: int(row["input_tokens"]), reverse=True)
    overflows.sort(key=lambda row: int(row["overflow_margin"]), reverse=True)

    calls_csv = output_dir / "static_context_calls.csv"
    overflows_jsonl = output_dir / "static_context_overflows.jsonl"
    summary_json = output_dir / "summary.json"
    write_csv(calls_csv, measured)
    with overflows_jsonl.open("w", encoding="utf-8") as handle:
        for row in overflows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    overflow_by_condition = Counter(row["condition"] for row in overflows)
    overflow_by_call_kind = Counter(row["call_kind"] for row in overflows)
    summary = {
        "created_at": int(time.time()),
        "status": "pass" if not overflows else "warn",
        "manifest_json": args.manifest_json,
        "online_inputs_jsonl": args.online_inputs_jsonl,
        "model_path": args.model_path,
        "max_context_tokens": args.max_context_tokens,
        "max_tokens": args.max_tokens,
        "num_rows": len(rows),
        "num_static_calls": len(measured),
        "num_static_overflows": len(overflows),
        "max_input_tokens": max([int(row["input_tokens"]) for row in measured] or [0]),
        "max_total_with_max_tokens": max([int(row["total_with_max_tokens"]) for row in measured] or [0]),
        "overflow_by_condition": dict(sorted(overflow_by_condition.items())),
        "overflow_by_call_kind": dict(sorted(overflow_by_call_kind.items())),
        "top_static_calls": measured[:20],
        "overflow_preview": overflows[:50],
        "calls_csv": str(calls_csv),
        "overflows_jsonl": str(overflows_jsonl),
        "scope_note": (
            "Static preflight only. Solver prompts containing model-generated upstream "
            "messages are dynamic and are not fully covered by this audit."
        ),
        "execution_boundary": [
            "local-only tokenizer length audit",
            "no model generation",
            "no optimizer or backward",
            "no SwanLab run",
        ],
    }
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_json"] = str(summary_json)
    return summary


def static_calls_for_row(row: dict[str, Any], samples: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    condition = str(row.get("condition", ""))
    fields = row.get("online_input_fields", {})
    base = {
        "row_id": row.get("row_id", ""),
        "sample_id": row.get("sample_id", ""),
        "condition": condition,
    }
    calls: list[dict[str, Any]] = []
    if condition in {"single_q_only", "single_full_info", "textmas_no_message"}:
        calls.append({**base, "call_kind": "solver_static", "messages": make_solver_messages(fields, [])})
    elif condition in {"textmas_matched", "textmas_compressed_state", "textmas_wrong_evidence_or_wrong_shard"}:
        compressed = condition == "textmas_compressed_state"
        for idx, observation in enumerate(fields.get("agent_private_observations", [])):
            calls.append(
                {
                    **base,
                    "call_kind": "agent_static",
                    "agent_observation_index": idx,
                    "messages": make_agent_messages(observation, compressed=compressed),
                }
            )
        calls.append({**base, "call_kind": "solver_base_without_dynamic_upstream", "messages": make_solver_messages(fields, [])})
    elif condition == "textmas_shuffled_message":
        control_id = str(row.get("control_source_sample_id", ""))
        control_sample = samples.get(control_id)
        if control_sample is not None:
            for idx, observation in enumerate(sample_agent_observations(control_sample)):
                calls.append(
                    {
                        **base,
                        "call_kind": "agent_static_shuffled_source",
                        "agent_observation_index": idx,
                        "control_source_sample_id": control_id,
                        "messages": make_agent_messages(observation, compressed=False),
                    }
                )
        calls.append({**base, "call_kind": "solver_base_without_dynamic_upstream", "messages": make_solver_messages(fields, [])})
    return calls


def count_chat_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    )
    return int(inputs.input_ids.shape[-1])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "row_id",
        "sample_id",
        "condition",
        "call_kind",
        "agent_observation_index",
        "control_source_sample_id",
        "input_tokens",
        "total_with_max_tokens",
        "overflow",
        "overflow_margin",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


if __name__ == "__main__":
    main()
