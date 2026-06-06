"""Preflight a Phase C capable-text-agent run before calling any model.

This local-only script checks manifest/control-input consistency, estimates
solver and evidence-agent chat calls, reports environment readiness for an
OpenAI-compatible endpoint, and writes a run plan.  It does not call models,
train adapters, inspect held-out generations, or create SwanLab runs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = "/data1/luyifei/drla/outputs/p2_phase_c_text_agent_preflights/preflight_20260601"
DEFAULT_COLA_DIT_PATH = "/data1/luyifei/drla/models/Cola-DLM/cola_dlm/cola_dit"
DEFAULT_COLA_VAE_PATH = "/data1/luyifei/drla/models/Cola-DLM/cola_dlm/cola_vae"
DEFAULT_COLA_TOKENIZER_PATH = "/data1/luyifei/drla/models/Cola-DLM/tokenizer.json"
DEFAULT_COLA_CODE_PATH = "/data1/luyifei/Cola-DLM/code"


def main() -> None:
    summary = preflight(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--online-inputs-jsonl", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--conditions", default="")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument(
        "--prediction-extraction-mode",
        choices=["default", "first_segment"],
        default="default",
    )
    parser.add_argument(
        "--provider",
        default="openai_compatible",
        choices=["openai_compatible", "local_transformers", "cola_dlm"],
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", ""))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument(
        "--local-model-path",
        default="/data1/luyifei/drla/models/Qwen3-4B-Instruct-2507-git",
        help="Local HuggingFace model path for --provider local_transformers.",
    )
    parser.add_argument("--cola-dit-path", default=DEFAULT_COLA_DIT_PATH)
    parser.add_argument("--cola-dit-lora-path", default="")
    parser.add_argument("--cola-agent-dit-lora-path", default="")
    parser.add_argument("--cola-solver-dit-lora-path", default="")
    parser.add_argument("--cola-vae-path", default=DEFAULT_COLA_VAE_PATH)
    parser.add_argument("--cola-tokenizer-path", default=DEFAULT_COLA_TOKENIZER_PATH)
    parser.add_argument("--cola-code-path", default=DEFAULT_COLA_CODE_PATH)
    parser.add_argument(
        "--cola-prompt-style",
        choices=["chat_join", "plain_qa_v1", "squad_template_v1"],
        default="chat_join",
    )
    parser.add_argument("--cola-noise-seed", default="66")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
    samples = {str(sample["sample_id"]): sample for sample in manifest.get("samples", [])}
    rows = filter_rows(read_jsonl(Path(args.online_inputs_jsonl)), args.conditions, args.max_rows, args.row_offset)
    check_rows_known(rows, samples)
    condition_counts = Counter(str(row.get("condition", "")) for row in rows)
    call_estimate = estimate_calls(rows, samples)
    env = {
        "provider": args.provider,
        "api_key_env": args.api_key_env,
        "api_key_set": bool(os.environ.get(args.api_key_env)),
        "model": args.model or "",
        "model_set": bool(args.model),
        "base_url": args.base_url,
        "local_model_path": args.local_model_path,
        "local_model_path_exists": Path(args.local_model_path).exists(),
        "prediction_extraction_mode": args.prediction_extraction_mode,
        "cola_dit_path": args.cola_dit_path,
        "cola_dit_path_exists": Path(args.cola_dit_path).exists(),
        "cola_dit_lora_path": args.cola_dit_lora_path,
        "cola_dit_lora_path_exists": bool(args.cola_dit_lora_path) and Path(args.cola_dit_lora_path).exists(),
        "cola_agent_dit_lora_path": args.cola_agent_dit_lora_path,
        "cola_agent_dit_lora_path_exists": bool(args.cola_agent_dit_lora_path)
        and Path(args.cola_agent_dit_lora_path).exists(),
        "cola_solver_dit_lora_path": args.cola_solver_dit_lora_path,
        "cola_solver_dit_lora_path_exists": bool(args.cola_solver_dit_lora_path)
        and Path(args.cola_solver_dit_lora_path).exists(),
        "cola_vae_path": args.cola_vae_path,
        "cola_vae_path_exists": Path(args.cola_vae_path).exists(),
        "cola_tokenizer_path": args.cola_tokenizer_path,
        "cola_tokenizer_path_exists": Path(args.cola_tokenizer_path).exists(),
        "cola_code_path": args.cola_code_path,
        "cola_code_path_exists": Path(args.cola_code_path).exists(),
        "cola_prompt_style": args.cola_prompt_style,
        "cola_noise_seed": args.cola_noise_seed,
    }
    ready = env_ready(env)
    provider_args = provider_command_args(args)
    command_template = {
        "run": (
            "source /data1/luyifei/drla/scripts/activate_conda.sh && "
            "python drla/scripts/run_p2_phase_c_text_agents.py "
            f"--manifest-json {args.manifest_json} "
            f"--online-inputs-jsonl {args.online_inputs_jsonl} "
            f"{provider_args} "
            f"--prediction-extraction-mode {args.prediction_extraction_mode} "
            f"--row-offset {args.row_offset} "
            "--output-dir <output_dir> "
            "--resume"
        ),
        "aggregate": (
            "source /data1/luyifei/drla/scripts/activate_conda.sh && "
            "python drla/scripts/aggregate_p2_phase_c_text_agent_results.py "
            "--generations-jsonl <output_dir>/generations.jsonl "
            "--output-dir <aggregate_output_dir> "
            "--overwrite"
        ),
    }
    metrics = {
        "num_rows": len(rows),
        "estimated_total_chat_calls": call_estimate["estimated_total_chat_calls"],
        "estimated_solver_calls": call_estimate["solver_calls"],
        "estimated_agent_calls": call_estimate["agent_calls"],
        "env_ready": int(ready),
    }
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    run_plan_path = output_dir / "run_plan.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "created_at": int(time.time()),
        "status": "pass" if rows else "fail",
        "ready_to_run_model": ready,
        "missing_requirements": missing_requirements(env),
        "manifest_json": args.manifest_json,
        "online_inputs_jsonl": args.online_inputs_jsonl,
        "num_samples": len(samples),
        "num_rows": len(rows),
        "condition_counts": dict(sorted(condition_counts.items())),
        "call_estimate": call_estimate,
        "env": env,
        "command_template": command_template,
        "metrics_jsonl": str(metrics_path),
        "run_plan_json": str(run_plan_path),
        "execution_boundary": [
            "local-only Phase C run preflight",
            "no model generation",
            "no optimizer or backward",
            "no SwanLab run",
            "no held-out prompt repair",
        ],
    }
    run_plan_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_json"] = str(summary_path)
    if not rows:
        raise ValueError("No rows selected for preflight")
    return summary


def estimate_calls(rows: list[dict[str, Any]], samples: dict[str, dict[str, Any]]) -> dict[str, Any]:
    solver_calls = len(rows)
    agent_cache: dict[str, int] = {}
    for row in rows:
        condition = str(row.get("condition", ""))
        fields = row.get("online_input_fields", {})
        if condition in {"textmas_matched", "textmas_compressed_state"}:
            key = f"{row['sample_id']}::{condition}"
            agent_cache.setdefault(key, len(fields.get("agent_private_observations", [])))
        elif condition == "textmas_shuffled_message":
            control_id = str(row.get("control_source_sample_id", fields.get("shuffled_message_source_sample_id", "")))
            control_sample = samples.get(control_id, {})
            key = f"{control_id}::textmas_matched"
            agent_cache.setdefault(key, len(control_sample.get("agent_views", [])))
        elif condition == "textmas_wrong_evidence_or_wrong_shard":
            key = f"{row['row_id']}::wrong_evidence"
            agent_cache.setdefault(key, len(fields.get("agent_private_observations", [])))
    agent_calls = sum(agent_cache.values())
    return {
        "solver_calls": solver_calls,
        "agent_calls": agent_calls,
        "unique_agent_cache_keys": len(agent_cache),
        "estimated_total_chat_calls": solver_calls + agent_calls,
    }


def missing_requirements(env: dict[str, Any]) -> list[str]:
    missing = []
    if env["provider"] == "openai_compatible":
        if not env["api_key_set"]:
            missing.append(env["api_key_env"])
        if not env["model_set"]:
            missing.append("OPENAI_MODEL or --model")
    elif env["provider"] == "local_transformers":
        if not env["local_model_path_exists"]:
            missing.append("existing --local-model-path")
    elif env["provider"] == "cola_dlm":
        if not env["cola_dit_path_exists"]:
            missing.append("existing --cola-dit-path")
        if env["cola_dit_lora_path"] and not env["cola_dit_lora_path_exists"]:
            missing.append("existing --cola-dit-lora-path")
        if env["cola_agent_dit_lora_path"] and not env["cola_agent_dit_lora_path_exists"]:
            missing.append("existing --cola-agent-dit-lora-path")
        if env["cola_solver_dit_lora_path"] and not env["cola_solver_dit_lora_path_exists"]:
            missing.append("existing --cola-solver-dit-lora-path")
        if not env["cola_vae_path_exists"]:
            missing.append("existing --cola-vae-path")
        if not env["cola_tokenizer_path_exists"]:
            missing.append("existing --cola-tokenizer-path")
        if not env["cola_code_path_exists"]:
            missing.append("existing --cola-code-path")
    else:
        missing.append(f"known provider, got {env['provider']}")
    return missing


def env_ready(env: dict[str, Any]) -> bool:
    return not missing_requirements(env)


def provider_command_args(args: argparse.Namespace) -> str:
    if args.provider == "openai_compatible":
        return "--provider openai_compatible"
    if args.provider == "local_transformers":
        return f"--provider local_transformers --local-model-path {args.local_model_path}"
    if args.provider == "cola_dlm":
        return (
            "--provider cola_dlm "
            f"--cola-dit-path {args.cola_dit_path} "
            f"{'--cola-dit-lora-path ' + args.cola_dit_lora_path + ' ' if args.cola_dit_lora_path else ''}"
            f"{'--cola-agent-dit-lora-path ' + args.cola_agent_dit_lora_path + ' ' if args.cola_agent_dit_lora_path else ''}"
            f"{'--cola-solver-dit-lora-path ' + args.cola_solver_dit_lora_path + ' ' if args.cola_solver_dit_lora_path else ''}"
            f"--cola-vae-path {args.cola_vae_path} "
            f"--cola-tokenizer-path {args.cola_tokenizer_path} "
            f"--cola-code-path {args.cola_code_path} "
            f"--cola-prompt-style {args.cola_prompt_style} "
            f"--cola-noise-seed {args.cola_noise_seed}"
        )
    raise ValueError(f"unknown provider: {args.provider}")


def check_rows_known(rows: list[dict[str, Any]], samples: dict[str, dict[str, Any]]) -> None:
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if sample_id not in samples:
            raise ValueError(f"unknown sample_id in online inputs: {sample_id}")


def filter_rows(rows: list[dict[str, Any]], conditions: str, max_rows: int, row_offset: int = 0) -> list[dict[str, Any]]:
    if conditions:
        allowed = {condition.strip() for condition in conditions.split(",") if condition.strip()}
        rows = [row for row in rows if row.get("condition") in allowed]
    if row_offset:
        if row_offset < 0:
            raise ValueError("--row-offset must be non-negative")
        rows = rows[row_offset:]
    if max_rows:
        rows = rows[:max_rows]
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(row)
    return rows


if __name__ == "__main__":
    main()
