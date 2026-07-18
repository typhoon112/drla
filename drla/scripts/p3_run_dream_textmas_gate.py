"""Run P3 Dream TextMAS capability-gate rows.

This local-only script consumes the locked MuSiQue evidence-split online inputs
and evaluates Dream-v0-Instruct-7B with the same Agent A -> Solver text protocol
used in Phase C. It uses Dream's ``diffusion_generate`` directly, not an
autoregressive ``generate`` fallback. It does not train, inspect held-out for
prompt repair, or create SwanLab runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.scripts.run_p2_phase_c_text_agents import (  # noqa: E402
    aggregate_condition_metrics,
    append_jsonl,
    filter_rows,
    read_jsonl,
    restore_agent_cache,
    run_condition,
    write_condition_csv,
    write_jsonl,
)


DEFAULT_MANIFEST_JSON = (
    "/data1/luyifei/drla/outputs/p2_phase_c_manifests/"
    "musique_calibration_manifest_200_seed20260601/manifest.json"
)
DEFAULT_ONLINE_INPUTS_JSONL = (
    "/data1/luyifei/drla/outputs/p2_phase_c_control_inputs/"
    "musique_calibration_controls_200_seed20260601_v1_strict_wrong/online_inputs.jsonl"
)
DEFAULT_MODEL_PATH = "/data1/luyifei/drla/models/Dream-v0-Instruct-7B"
DEFAULT_OUTPUT_ROOT = "/data1/luyifei/drla/outputs/p3_dream_textmas_runs"


def main() -> None:
    summary = run_eval(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--online-inputs-jsonl", default=DEFAULT_ONLINE_INPUTS_JSONL)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model", default="Dream-v0-Instruct-7B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--dream-steps", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--alg", default="entropy")
    parser.add_argument("--alg-temp", type=float, default=0.0)
    parser.add_argument("--max-context-tokens", type=int, default=2048)
    parser.add_argument(
        "--prediction-extraction-mode",
        choices=["default", "first_segment"],
        default="first_segment",
    )
    parser.add_argument("--conditions", default="")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-on-row-error", action="store_true")
    parser.add_argument("--call-log-limit", type=int, default=200)
    return parser.parse_args()


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite and not args.resume:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite or --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    generations_path = output_dir / "generations.jsonl"
    manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
    samples = {str(sample["sample_id"]): sample for sample in manifest.get("samples", [])}
    rows = filter_rows(read_jsonl(Path(args.online_inputs_jsonl)), args.conditions, 0, args.row_offset)
    rows = filter_by_sample_count(rows, args.max_samples)
    if args.max_rows:
        rows = rows[: args.max_rows]
    if not rows:
        raise ValueError("No rows selected")

    provider = DreamDLMProvider(args)
    generations = (
        read_jsonl(generations_path, allow_truncated_final_line=args.resume)
        if args.resume and generations_path.exists()
        else []
    )
    if args.resume:
        write_jsonl(generations_path, generations)
    else:
        generations_path.write_text("", encoding="utf-8")
    restore_agent_cache(provider, generations)
    completed = {str(row.get("row_id", "")) for row in generations}

    errors = []
    for row in rows:
        row_id = str(row.get("row_id", ""))
        if row_id in completed:
            continue
        sample = samples.get(str(row.get("sample_id", "")))
        if sample is None:
            raise ValueError(f"unknown sample_id in online inputs: {row.get('sample_id')}")
        try:
            result = run_condition(row, sample, samples, provider, args)
            result["row_index"] = len(generations) + 1
            result["dream_call_count_so_far"] = provider.call_index
            result["dream_recent_call_metrics"] = provider.recent_call_metrics()
        except Exception as exc:  # Keep smoke artifacts inspectable.
            if args.fail_on_row_error:
                raise
            result = error_result(row, args, provider, exc, len(generations) + 1)
            errors.append(result)
        append_jsonl(generations_path, result)
        generations.append(result)

    return write_outputs(args, output_dir, manifest, rows, generations, errors, provider)


def default_output_dir(args: argparse.Namespace) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    suffix = "dream_textmas_gate"
    if args.max_rows:
        suffix += f"_rows{args.max_rows}"
    if args.max_samples:
        suffix += f"_samples{args.max_samples}"
    if args.conditions:
        suffix += "_" + args.conditions.replace(",", "_")
    return Path(DEFAULT_OUTPUT_ROOT) / f"{suffix}_{stamp}"


class DreamDLMProvider:
    name = "dream_dlm"

    def __init__(self, args: argparse.Namespace) -> None:
        self.agent_cache: dict[str, list[dict[str, str]]] = {}
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("DreamDLMProvider requires CUDA when --device starts with cuda")
        self.device = torch.device(args.device)
        self.dtype = resolve_dtype(args.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device).eval()
        self.call_index = 0
        self.call_metrics: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, str]], args: argparse.Namespace) -> str:
        self.call_index += 1
        call_start = time.time()
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        input_ids = inputs.input_ids.to(device=self.device)
        attention_mask = inputs.attention_mask.to(device=self.device)
        input_tokens = int(input_ids.shape[-1])
        if input_tokens + args.max_tokens > args.max_context_tokens:
            raise ValueError(
                f"Dream context exceeds {args.max_context_tokens}: input_tokens={input_tokens}, "
                f"max_tokens={args.max_tokens}"
            )
        torch.cuda.reset_peak_memory_stats(self.device) if self.device.type == "cuda" else None
        with torch.no_grad():
            output = self.model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_tokens,
                output_history=False,
                return_dict_in_generate=True,
                steps=args.dream_steps,
                temperature=args.temperature,
                top_p=args.top_p,
                alg=args.alg,
                alg_temp=args.alg_temp,
            )
        sequences = getattr(output, "sequences", output)
        generated = sequences[0, input_tokens:].detach().cpu().tolist()
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        peak_memory = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
        self.call_metrics.append(
            {
                "call_index": self.call_index,
                "input_tokens": input_tokens,
                "max_tokens": args.max_tokens,
                "dream_steps": args.dream_steps,
                "elapsed_seconds": round(time.time() - call_start, 3),
                "peak_memory_gib": round(peak_memory / 1024**3, 3),
                "num_messages": len(messages),
                "output_chars": len(text),
            }
        )
        return text

    def recent_call_metrics(self) -> list[dict[str, Any]]:
        return self.call_metrics[-8:]


def resolve_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def error_result(
    row: dict[str, Any],
    args: argparse.Namespace,
    provider: DreamDLMProvider,
    exc: Exception,
    row_index: int,
) -> dict[str, Any]:
    return {
        "row_id": row.get("row_id", ""),
        "sample_id": row.get("sample_id", ""),
        "task_name": row.get("task_name", ""),
        "split": row.get("split", ""),
        "condition": row.get("condition", ""),
        "model": args.model,
        "provider": provider.name,
        "row_index": row_index,
        "status": "error",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "online_input_fields": row.get("online_input_fields", {}),
        "agent_messages": [],
        "raw_final_output": "",
        "prediction": "",
        "score": {"primary_score": 0.0, "token_f1": 0.0, "exact_match": 0.0},
        "primary_score": 0.0,
        "token_f1": 0.0,
        "exact_match": 0.0,
        "prompt_contract_version": row.get("prompt_contract_version", ""),
        "dream_call_count_so_far": provider.call_index,
        "dream_recent_call_metrics": provider.recent_call_metrics(),
    }


def write_outputs(
    args: argparse.Namespace,
    output_dir: Path,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    provider: DreamDLMProvider,
) -> dict[str, Any]:
    generations_path = output_dir / "generations.jsonl"
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    condition_csv_path = output_dir / "condition_metrics.csv"
    call_metrics_path = output_dir / "dream_call_metrics.jsonl"
    write_jsonl(generations_path, generations)
    condition_metrics = aggregate_condition_metrics(generations)
    write_condition_csv(condition_csv_path, condition_metrics)
    with call_metrics_path.open("w", encoding="utf-8") as handle:
        for item in provider.call_metrics:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    metrics = {
        "num_rows": len(rows),
        "num_generations": len(generations),
        "num_errors": len([row for row in generations if row.get("status") == "error"]),
        "mean_primary_score": mean([float(row.get("primary_score", 0.0)) for row in generations]),
        "mean_token_f1": mean([float(row.get("token_f1", 0.0)) for row in generations]),
        "dream_call_count": provider.call_index,
        "max_input_tokens": max([int(item.get("input_tokens", 0)) for item in provider.call_metrics] or [0]),
        "max_peak_memory_gib": max([float(item.get("peak_memory_gib", 0.0)) for item in provider.call_metrics] or [0.0]),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "created_at": int(time.time()),
        "status": "pass" if not errors else "warn",
        "provider": "dream_dlm",
        "model": args.model,
        "model_path": args.model_path,
        "manifest_json": args.manifest_json,
        "online_inputs_jsonl": args.online_inputs_jsonl,
        "generations_jsonl": str(generations_path),
        "metrics_jsonl": str(metrics_path),
        "condition_metrics_csv": str(condition_csv_path),
        "dream_call_metrics_jsonl": str(call_metrics_path),
        "num_manifest_samples": len(manifest.get("samples", [])),
        "num_rows": len(rows),
        "num_errors": metrics["num_errors"],
        "condition_metrics": condition_metrics,
        "run_config": {
            "device": args.device,
            "dtype": args.dtype,
            "max_tokens": args.max_tokens,
            "dream_steps": args.dream_steps,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "alg": args.alg,
            "alg_temp": args.alg_temp,
            "max_context_tokens": args.max_context_tokens,
            "conditions": args.conditions,
            "row_offset": args.row_offset,
            "max_rows": args.max_rows,
            "max_samples": args.max_samples,
            "prediction_extraction_mode": args.prediction_extraction_mode,
        },
        "metrics": metrics,
        "errors_preview": errors[:20],
        "execution_boundary": [
            "local-only P3 Dream TextMAS capability-gate evaluation",
            "Dream diffusion_generate inference only",
            "no optimizer or backward",
            "no SwanLab run",
            "gold/scorer fields used only for offline scoring after generation",
            "held-out must not be used for prompt repair or threshold tuning",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_json"] = str(summary_path)
    return summary


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def filter_by_sample_count(rows: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    if max_samples <= 0:
        return rows
    kept_sample_ids: list[str] = []
    kept = set()
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if sample_id not in kept:
            if len(kept_sample_ids) >= max_samples:
                break
            kept.add(sample_id)
            kept_sample_ids.append(sample_id)
    allowed = set(kept_sample_ids)
    return [row for row in rows if str(row.get("sample_id", "")) in allowed]


if __name__ == "__main__":
    main()
