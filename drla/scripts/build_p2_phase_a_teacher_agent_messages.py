"""Collect teacher evidence-agent messages for Phase A CoLA interface SFT.

This is a local-only data generation utility.  It reads Phase C manifest/control
inputs, calls a configured teacher only for evidence-agent messages, and writes a
``generations.jsonl`` compatible with ``build_p2_phase_a_cola_interface_sft.py``.

It intentionally does not call the final solver, score answers, run optimizer or
backward, create SwanLab runs, inspect held-out generations, or tune prompts.
The intended use is train/calibration source splits only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.scripts.run_p2_phase_c_text_agents import (
    DEFAULT_COLA_CODE_PATH,
    DEFAULT_COLA_DIT_PATH,
    DEFAULT_COLA_TOKENIZER_PATH,
    DEFAULT_COLA_VAE_PATH,
    append_jsonl,
    filter_rows,
    get_or_make_agent_messages,
    make_provider,
    read_jsonl,
    restore_agent_cache,
    write_jsonl,
)


DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p2_phase_a_teacher_agent_messages/"
    "teacher_agent_messages_20260605"
)
ALLOWED_SOURCE_SPLITS = {"calibration", "calib", "train"}
SUPPORTED_CONDITIONS = {"textmas_matched", "textmas_compressed_state"}


def main() -> None:
    summary = collect_teacher_messages(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--online-inputs-jsonl", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--provider",
        default="local_transformers",
        choices=["openai_compatible", "local_transformers", "cola_dlm"],
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", ""))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--local-model-path", default="/data1/luyifei/drla/models/Qwen3-8B-FP8")
    parser.add_argument("--local-device-map", default="auto")
    parser.add_argument("--local-dtype", default="auto")
    parser.add_argument("--local-enable-thinking", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--cola-dit-path", default=DEFAULT_COLA_DIT_PATH)
    parser.add_argument("--cola-dit-lora-path", default="")
    parser.add_argument("--cola-agent-dit-lora-path", default="")
    parser.add_argument("--cola-solver-dit-lora-path", default="")
    parser.add_argument("--cola-vae-path", default=DEFAULT_COLA_VAE_PATH)
    parser.add_argument("--cola-tokenizer-path", default=DEFAULT_COLA_TOKENIZER_PATH)
    parser.add_argument("--cola-code-path", default=DEFAULT_COLA_CODE_PATH)
    parser.add_argument("--cola-device", default="auto")
    parser.add_argument(
        "--cola-prompt-style",
        choices=["chat_join", "plain_qa_v1", "squad_template_v1"],
        default="chat_join",
    )
    parser.add_argument("--cola-timestep-num", type=int, default=16)
    parser.add_argument("--cola-guidance-scale", type=float, default=7.0)
    parser.add_argument("--cola-noise-seed", default="66")
    parser.add_argument("--cola-top-k", type=int, default=50)
    parser.add_argument("--cola-top-p", type=float, default=0.9)
    parser.add_argument("--cola-repetition-penalty", type=float, default=1.1)
    parser.add_argument("--cola-pad-token-id", type=int, default=100277)
    parser.add_argument("--cola-eos-token-id", type=int, default=100257)
    parser.add_argument("--cola-im-end-token-id", type=int, default=100265)
    parser.add_argument("--conditions", default="textmas_matched")
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all filtered rows.")
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def collect_teacher_messages(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite and not args.resume:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite or --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
    samples = {str(sample["sample_id"]): sample for sample in manifest.get("samples", [])}
    assert_allowed_source_splits(samples.values())

    rows = read_jsonl(Path(args.online_inputs_jsonl))
    rows = filter_rows(rows, args.conditions, args.max_rows, args.row_offset)
    assert_supported_conditions(rows)

    generations_path = output_dir / "generations.jsonl"
    generations = (
        read_jsonl(generations_path, allow_truncated_final_line=args.resume)
        if args.resume and generations_path.exists()
        else []
    )
    if args.resume:
        write_jsonl(generations_path, generations)
    else:
        generations_path.write_text("", encoding="utf-8")

    provider = make_provider(args)
    restore_agent_cache(provider, generations)
    completed = {str(row.get("row_id", "")) for row in generations}
    for row in rows:
        row_id = str(row.get("row_id", ""))
        if row_id in completed:
            continue
        sample_id = str(row.get("sample_id", ""))
        if sample_id not in samples:
            raise ValueError(f"unknown sample_id in online inputs: {sample_id}")
        result = collect_row(row, provider, args)
        result["row_index"] = len(generations) + 1
        append_jsonl(generations_path, result)
        generations.append(result)

    return write_summary(args, output_dir, manifest, rows, generations)


def collect_row(row: dict[str, Any], provider: Any, args: argparse.Namespace) -> dict[str, Any]:
    condition = str(row.get("condition", ""))
    fields = row.get("online_input_fields", {})
    compressed = condition == "textmas_compressed_state"
    agent_messages = get_or_make_agent_messages(
        cache_key=f"{row['sample_id']}::{condition}",
        observations=fields.get("agent_private_observations", []),
        provider=provider,
        args=args,
        compressed=compressed,
    )
    return {
        "row_id": row["row_id"],
        "sample_id": row["sample_id"],
        "task_name": row.get("task_name", ""),
        "split": row.get("split", ""),
        "condition": condition,
        "model": args.model,
        "provider": provider.name,
        "online_input_fields": fields,
        "agent_messages": agent_messages,
        "raw_final_output": "",
        "prediction": "",
        "score": {},
        "primary_score": 0.0,
        "token_f1": 0.0,
        "exact_match": 0.0,
        "prompt_contract_version": row.get("prompt_contract_version", ""),
        "collector_note": "teacher evidence-agent messages only; final solver not called",
    }


def write_summary(
    args: argparse.Namespace,
    output_dir: Path,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    generations: list[dict[str, Any]],
) -> dict[str, Any]:
    generations_path = output_dir / "generations.jsonl"
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    condition_counts = Counter(str(row.get("condition", "")) for row in generations)
    split_counts = Counter(str(row.get("split", "")) for row in generations)
    message_lengths = [
        len(str(message.get("message", "")).strip())
        for row in generations
        for message in row.get("agent_messages", [])
        if isinstance(message, dict)
    ]
    nonempty_messages = sum(1 for length in message_lengths if length > 0)
    metrics = {
        "num_rows_requested": len(rows),
        "num_generations": len(generations),
        "num_agent_messages": len(message_lengths),
        "nonempty_agent_message_rate": nonempty_messages / len(message_lengths) if message_lengths else 0.0,
        "mean_agent_message_chars": sum(message_lengths) / len(message_lengths) if message_lengths else 0.0,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "created_at": int(time.time()),
        "status": "pass" if len(generations) == len(rows) else "partial",
        "provider": args.provider,
        "model": args.model,
        "manifest_json": args.manifest_json,
        "online_inputs_jsonl": args.online_inputs_jsonl,
        "generations_jsonl": str(generations_path),
        "metrics_jsonl": str(metrics_path),
        "num_manifest_samples": len(manifest.get("samples", [])),
        "num_rows_requested": len(rows),
        "num_generations": len(generations),
        "condition_counts": dict(sorted(condition_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "metrics": metrics,
        "execution_boundary": [
            "local-only Phase A teacher evidence-agent message collection",
            "source split must be calibration/calib/train",
            "no final solver calls",
            "no offline scoring",
            "no optimizer or backward",
            "no SwanLab run",
            "no held-out generations or prompt repair",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_json"] = str(summary_path)
    return summary


def assert_allowed_source_splits(samples: Any) -> None:
    bad = [
        str(sample.get("sample_id", ""))
        for sample in samples
        if str(sample.get("split", "")).lower() not in ALLOWED_SOURCE_SPLITS
    ]
    if bad:
        raise ValueError(f"held-out/test/valid samples are forbidden for Phase A teacher collection: {bad[:5]}")


def assert_supported_conditions(rows: list[dict[str, Any]]) -> None:
    bad = sorted({str(row.get("condition", "")) for row in rows} - SUPPORTED_CONDITIONS)
    if bad:
        raise ValueError(f"unsupported teacher-message conditions: {bad}; allowed={sorted(SUPPORTED_CONDITIONS)}")


if __name__ == "__main__":
    main()
