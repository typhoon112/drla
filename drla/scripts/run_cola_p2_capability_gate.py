"""Run Cola P2 benchmark and role-MAS capability gates.

This is a local-only generation/evaluation script.  It must not create SwanLab
runs because it has no training loop.  The goal is to decide which candidate
benchmarks are suitable for later text-vs-latent Agent-A -> Agent-B comparison,
not to claim a new state of the art.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import statistics
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch

from drla.scripts.collect_cola_block_traces import (
    ColaBlockTraceConfig,
    load_cola_symbols,
    read_jsonl,
    resolve_device,
    set_seed,
)
from drla.tracking import require_swanlab_disabled_for_non_training


DEFAULT_INPUT_JSONL = (
    "/data1/luyifei/drla/outputs/p2_capability_gate/data/"
    "p2_candidate_benchmarks.jsonl"
)

MODES = ["single", "role_textmas"]
ROLE_ORDER = ["planner", "critic", "refiner", "solver"]
PROMPT_VARIANTS = [
    "generic_v1",
    "cola_fewshot_v1",
    "answer_state_v1",
    "answer_state_structured_v1",
    "role_plan_ignore_v1",
]


@dataclass(frozen=True)
class ColaP2CapabilityGateConfig:
    input_jsonl: str = DEFAULT_INPUT_JSONL
    output_dir: str = "/data1/luyifei/drla/outputs/p2_capability_gate/eval"
    dit_path: str = ColaBlockTraceConfig.dit_path
    vae_path: str = ColaBlockTraceConfig.vae_path
    tokenizer_path: str = ColaBlockTraceConfig.tokenizer_path
    cola_code_path: str = "/data1/luyifei/Cola-DLM/code"
    tasks: str = ""
    modes: str = ",".join(MODES)
    batch_size: int = 4
    max_samples_per_task: int = 0
    seed: int = 20260601
    max_new_tokens: int = 64
    role_max_new_tokens: int = 64
    timestep_num: int = 16
    guidance_scale: float = 7.0
    temperature: float = 0.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    pad_token_id: int = 100277
    eos_token_id: int | None = 100257
    im_end_token_id: int | None = 100265
    device: str = "auto"
    swanlab_mode: str = "disabled"
    min_nonempty_rate: float = 0.95
    min_parseable_rate: float = 0.90
    min_accuracy_margin: float = 0.02
    enable_code_execution: bool = False
    code_timeout_seconds: float = 5.0
    prompt_variant: str = "generic_v1"
    single_prompt_variant: str = ""
    role_prompt_variant: str = ""


def main() -> None:
    summary = run_cola_p2_capability_gate(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> ColaP2CapabilityGateConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", default=ColaP2CapabilityGateConfig.input_jsonl)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dit-path", default=ColaP2CapabilityGateConfig.dit_path)
    parser.add_argument("--vae-path", default=ColaP2CapabilityGateConfig.vae_path)
    parser.add_argument("--tokenizer-path", default=ColaP2CapabilityGateConfig.tokenizer_path)
    parser.add_argument("--cola-code-path", default=ColaP2CapabilityGateConfig.cola_code_path)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--modes", default=ColaP2CapabilityGateConfig.modes)
    parser.add_argument("--batch-size", type=int, default=ColaP2CapabilityGateConfig.batch_size)
    parser.add_argument("--max-samples-per-task", type=int, default=0)
    parser.add_argument("--seed", type=int, default=ColaP2CapabilityGateConfig.seed)
    parser.add_argument("--max-new-tokens", type=int, default=ColaP2CapabilityGateConfig.max_new_tokens)
    parser.add_argument("--role-max-new-tokens", type=int, default=ColaP2CapabilityGateConfig.role_max_new_tokens)
    parser.add_argument("--timestep-num", type=int, default=ColaP2CapabilityGateConfig.timestep_num)
    parser.add_argument("--guidance-scale", type=float, default=ColaP2CapabilityGateConfig.guidance_scale)
    parser.add_argument("--temperature", type=float, default=ColaP2CapabilityGateConfig.temperature)
    parser.add_argument("--top-k", type=int, default=ColaP2CapabilityGateConfig.top_k)
    parser.add_argument("--top-p", type=float, default=ColaP2CapabilityGateConfig.top_p)
    parser.add_argument("--repetition-penalty", type=float, default=ColaP2CapabilityGateConfig.repetition_penalty)
    parser.add_argument("--pad-token-id", type=int, default=ColaP2CapabilityGateConfig.pad_token_id)
    parser.add_argument("--eos-token-id", type=int, default=ColaP2CapabilityGateConfig.eos_token_id)
    parser.add_argument("--im-end-token-id", type=int, default=ColaP2CapabilityGateConfig.im_end_token_id)
    parser.add_argument("--device", default=ColaP2CapabilityGateConfig.device)
    parser.add_argument("--swanlab-mode", default=ColaP2CapabilityGateConfig.swanlab_mode)
    parser.add_argument("--min-nonempty-rate", type=float, default=ColaP2CapabilityGateConfig.min_nonempty_rate)
    parser.add_argument("--min-parseable-rate", type=float, default=ColaP2CapabilityGateConfig.min_parseable_rate)
    parser.add_argument("--min-accuracy-margin", type=float, default=ColaP2CapabilityGateConfig.min_accuracy_margin)
    parser.add_argument("--enable-code-execution", action="store_true")
    parser.add_argument("--code-timeout-seconds", type=float, default=ColaP2CapabilityGateConfig.code_timeout_seconds)
    parser.add_argument("--prompt-variant", choices=PROMPT_VARIANTS, default=ColaP2CapabilityGateConfig.prompt_variant)
    parser.add_argument(
        "--single-prompt-variant",
        choices=PROMPT_VARIANTS,
        default="",
        help="Optional override for single-solver prompts; defaults to --prompt-variant.",
    )
    parser.add_argument(
        "--role-prompt-variant",
        choices=PROMPT_VARIANTS,
        default="",
        help="Optional override for Role TextMAS prompts; defaults to --prompt-variant.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_samples_per_task < 0:
        raise ValueError("--max-samples-per-task must be non-negative")
    if args.min_nonempty_rate < 0 or args.min_parseable_rate < 0:
        raise ValueError("rate thresholds must be non-negative")
    if args.code_timeout_seconds <= 0:
        raise ValueError("--code-timeout-seconds must be positive")
    modes = normalize_modes(args.modes)
    return ColaP2CapabilityGateConfig(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        dit_path=args.dit_path,
        vae_path=args.vae_path,
        tokenizer_path=args.tokenizer_path,
        cola_code_path=args.cola_code_path,
        tasks=args.tasks,
        modes=",".join(modes),
        batch_size=args.batch_size,
        max_samples_per_task=args.max_samples_per_task,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        role_max_new_tokens=args.role_max_new_tokens,
        timestep_num=args.timestep_num,
        guidance_scale=args.guidance_scale,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=args.pad_token_id,
        eos_token_id=args.eos_token_id,
        im_end_token_id=args.im_end_token_id,
        device=args.device,
        swanlab_mode=args.swanlab_mode,
        min_nonempty_rate=args.min_nonempty_rate,
        min_parseable_rate=args.min_parseable_rate,
        min_accuracy_margin=args.min_accuracy_margin,
        enable_code_execution=args.enable_code_execution,
        code_timeout_seconds=args.code_timeout_seconds,
        prompt_variant=args.prompt_variant,
        single_prompt_variant=args.single_prompt_variant,
        role_prompt_variant=args.role_prompt_variant,
    )


def run_cola_p2_capability_gate(config: ColaP2CapabilityGateConfig) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        config.swanlab_mode,
        script_kind="Cola P2 benchmark capability gate",
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = select_rows(
        read_jsonl(Path(config.input_jsonl)),
        tasks=config.tasks,
        max_samples_per_task=config.max_samples_per_task,
    )
    if not rows:
        raise ValueError("No rows selected for capability gate")

    set_seed(config.seed)
    add_cola_code_path(config.cola_code_path)
    device = resolve_device(config.device)
    if device.type != "cuda":
        raise RuntimeError(
            "Official Cola generation uses CUDA autocast and should run on GPU. "
            f"Resolved device: {device}."
        )

    cola = load_cola_symbols()
    from cola_dlm import generate_task_repaint_inference

    tokenizer = cola["Tokenizer"].from_file(config.tokenizer_path)
    dit = cola["ColaDiTModel"].from_pretrained(config.dit_path).to(device).eval()
    vae = cola["ColaTextVAEModel"].from_pretrained(config.vae_path).to(device).eval()

    modes = normalize_modes(config.modes)
    generations_path = output_dir / "generations.jsonl"
    metrics_path = output_dir / "metrics.jsonl"
    task_summary_path = output_dir / "task_summary.csv"
    summary_path = output_dir / "summary.json"

    all_records: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    started = time.time()

    with generations_path.open("w", encoding="utf-8") as gen_f, metrics_path.open(
        "w", encoding="utf-8"
    ) as metrics_f:
        if "single" in modes:
            single_records = run_single_mode(
                rows=rows,
                config=config,
                dit=dit,
                vae=vae,
                tokenizer=tokenizer,
                generate_task_repaint_inference=generate_task_repaint_inference,
                device=device,
            )
            write_records(gen_f, single_records)
            all_records.extend(single_records)
            metrics_rows.extend(write_incremental_metrics(metrics_f, single_records, mode="single"))

        if "role_textmas" in modes:
            role_records = run_role_textmas_mode(
                rows=rows,
                config=config,
                dit=dit,
                vae=vae,
                tokenizer=tokenizer,
                generate_task_repaint_inference=generate_task_repaint_inference,
                device=device,
            )
            write_records(gen_f, role_records)
            all_records.extend(role_records)
            metrics_rows.extend(
                write_incremental_metrics(metrics_f, role_records, mode="role_textmas")
            )

    task_summaries = build_task_summaries(all_records, config=config)
    write_task_summary_csv(task_summary_path, task_summaries)
    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "input_jsonl": config.input_jsonl,
        "output_dir": str(output_dir),
        "generations_jsonl": str(generations_path),
        "metrics_jsonl": str(metrics_path),
        "task_summary_csv": str(task_summary_path),
        "num_input_rows": len(rows),
        "num_generation_records": len(all_records),
        "elapsed_seconds": time.time() - started,
        "is_smoke": config.max_samples_per_task > 0,
        "task_summaries": task_summaries,
        "admitted_tasks": [
            item["task"]
            for item in task_summaries
            if item["mode"] == "single" and item["admitted_for_main"]
        ],
        "notes": [
            "Pure generation/evaluation run; SwanLab must remain disabled.",
            (
                "Code tasks use execution tests when --enable-code-execution is set; "
                "otherwise they remain syntax-only pre-gates."
            ),
            "Smoke runs never admit tasks for main P2 tables.",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def run_single_mode(
    *,
    rows: list[dict[str, Any]],
    config: ColaP2CapabilityGateConfig,
    dit: Any,
    vae: Any,
    tokenizer: Any,
    generate_task_repaint_inference: Any,
    device: torch.device,
) -> list[dict[str, Any]]:
    prompts = []
    prompt_variant = resolve_single_prompt_variant(config)
    for row in rows:
        prompts.append(to_generation_prompt(row, build_single_solver_prompt(row, prompt_variant)))
    generated = generate_batches(
        prompts=prompts,
        config=config,
        dit=dit,
        vae=vae,
        tokenizer=tokenizer,
        generate_task_repaint_inference=generate_task_repaint_inference,
        device=device,
        max_new_tokens=config.max_new_tokens,
    )
    records: list[dict[str, Any]] = []
    row_by_id = {row["id"]: row for row in rows}
    for out in generated:
        row = row_by_id[str(out.get("id"))]
        text = str(out.get("generate", ""))
        score = score_candidate(row, text, config=config)
        records.append(
            {
                "id": row["id"],
                "task": row["task"],
                "mode": "single",
                "answer_type": row.get("answer_type"),
                "prompt": out.get("prompt", ""),
                "prompt_variant": prompt_variant,
                "generate": text,
                "role_outputs": {},
                "score": score,
            }
        )
    return records


def run_role_textmas_mode(
    *,
    rows: list[dict[str, Any]],
    config: ColaP2CapabilityGateConfig,
    dit: Any,
    vae: Any,
    tokenizer: Any,
    generate_task_repaint_inference: Any,
    device: torch.device,
) -> list[dict[str, Any]]:
    role_outputs: dict[str, dict[str, str]] = {row["id"]: {} for row in rows}
    prompt_variant = resolve_role_prompt_variant(config)
    for role in ROLE_ORDER:
        prompts = []
        for row in rows:
            prompts.append(
                to_generation_prompt(
                    row,
                    build_role_prompt(row, role_outputs[row["id"]], role, prompt_variant),
                )
            )
        generated = generate_batches(
            prompts=prompts,
            config=config,
            dit=dit,
            vae=vae,
            tokenizer=tokenizer,
            generate_task_repaint_inference=generate_task_repaint_inference,
            device=device,
            max_new_tokens=config.role_max_new_tokens,
        )
        for out in generated:
            row_id = str(out.get("id"))
            role_outputs[row_id][role] = str(out.get("generate", ""))

    records: list[dict[str, Any]] = []
    row_by_id = {row["id"]: row for row in rows}
    for row_id, outputs in role_outputs.items():
        row = row_by_id[row_id]
        text = outputs.get("solver", "")
        score = score_candidate(row, text, config=config)
        records.append(
            {
                "id": row_id,
                "task": row["task"],
                "mode": "role_textmas",
                "answer_type": row.get("answer_type"),
                "prompt": build_role_prompt(row, outputs, "solver", prompt_variant),
                "prompt_variant": prompt_variant,
                "generate": text,
                "role_outputs": outputs,
                "score": score,
            }
        )
    return records


def generate_batches(
    *,
    prompts: list[dict[str, Any]],
    config: ColaP2CapabilityGateConfig,
    dit: Any,
    vae: Any,
    tokenizer: Any,
    generate_task_repaint_inference: Any,
    device: torch.device,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(prompts), config.batch_size):
            batch = prompts[start : start + config.batch_size]
            outputs.extend(
                generate_task_repaint_inference(
                    dit=dit,
                    vae=vae,
                    tokenizer=tokenizer,
                    prompts=batch,
                    task_name="p2_generic",
                    device=device,
                    timestep_num=config.timestep_num,
                    guidance_scale=config.guidance_scale,
                    max_new_tokens=max_new_tokens,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    repetition_penalty=config.repetition_penalty,
                    pad_token_id=config.pad_token_id,
                    eos_token_id=config.eos_token_id,
                    im_end_token_id=config.im_end_token_id,
                    is_sft=False,
                )
            )
    return outputs


def to_generation_prompt(row: dict[str, Any], prompt: str) -> dict[str, Any]:
    return {
        "id": row["id"],
        "question": prompt,
        "answer": row.get("ground_truth", row.get("answer", "")),
        "ground_truth": row.get("ground_truth", row.get("answer", "")),
        "choices": row.get("choices", []),
    }


def build_single_solver_prompt(row: dict[str, Any], prompt_variant: str) -> str:
    if prompt_variant == "cola_fewshot_v1":
        return build_cola_fewshot_single_prompt(row)
    if prompt_variant in {"answer_state_v1", "answer_state_structured_v1"}:
        return build_answer_state_single_prompt(row)
    task = row.get("task", "")
    body = format_task_body(row)
    if row.get("answer_type") == "multiple_choice":
        instruction = "Choose the best option. End with a final line starting with Answer: followed by one option letter."
    elif row.get("answer_type") == "numeric":
        instruction = "Solve the problem. End with a final line starting with Answer: followed by the number."
    elif row.get("answer_type") == "code":
        instruction = "Complete the Python code. Return code only."
    else:
        instruction = "Answer the task."
    return f"Task: {task}\n{body}\n\n{instruction}\nAnswer:"


def resolve_single_prompt_variant(config: ColaP2CapabilityGateConfig) -> str:
    return config.single_prompt_variant or config.prompt_variant


def resolve_role_prompt_variant(config: ColaP2CapabilityGateConfig) -> str:
    return config.role_prompt_variant or config.prompt_variant


def build_role_prompt(
    row: dict[str, Any],
    role_outputs: dict[str, str],
    role: str,
    prompt_variant: str = "generic_v1",
) -> str:
    if prompt_variant == "cola_fewshot_v1":
        return build_cola_fewshot_role_prompt(row, role_outputs, role)
    if prompt_variant == "answer_state_v1":
        return build_answer_state_role_prompt(row, role_outputs, role)
    if prompt_variant == "answer_state_structured_v1":
        return build_answer_state_structured_role_prompt(row, role_outputs, role)
    if prompt_variant == "role_plan_ignore_v1":
        return build_role_plan_ignore_prompt(row, role_outputs, role)
    body = format_task_body(row)
    if role == "planner":
        return (
            f"{body}\n\nRole: Planner.\n"
            "Write a concise plan or key evidence. Do not give a final answer unless unavoidable.\n"
            "Planner:"
        )
    if role == "critic":
        return (
            f"{body}\n\nPlanner message:\n{role_outputs.get('planner', '')}\n\n"
            "Role: Critic.\nPoint out mistakes, missing evidence, and what the solver should verify.\n"
            "Critic:"
        )
    if role == "refiner":
        return (
            f"{body}\n\nPlanner message:\n{role_outputs.get('planner', '')}\n\n"
            f"Critic message:\n{role_outputs.get('critic', '')}\n\n"
            "Role: Refiner.\nProduce the corrected reasoning state for the solver.\n"
            "Refiner:"
        )
    if row.get("answer_type") == "multiple_choice":
        answer_instruction = "Use the refined state and end with a final line starting with Answer: followed by one option letter."
    elif row.get("answer_type") == "numeric":
        answer_instruction = "Use the refined state and end with a final line starting with Answer: followed by the number."
    elif row.get("answer_type") == "code":
        answer_instruction = "Use the refined state and return code only."
    else:
        answer_instruction = "Use the refined state and answer directly."
    return (
        f"{body}\n\nRefined state:\n{role_outputs.get('refiner', '')}\n\n"
        f"Role: Solver.\n{answer_instruction}\nAnswer:"
    )


def build_answer_state_single_prompt(row: dict[str, Any]) -> str:
    body = format_task_body(row)
    answer_type = row.get("answer_type")
    if answer_type == "multiple_choice":
        return (
            f"{body}\n\n"
            "Choose the best option. Output only the final answer line.\n"
            "Format: Answer: option letter\n"
            "Answer:"
        )
    if answer_type == "numeric":
        return (
            f"{body}\n\n"
            "Solve the problem. Output only the final answer line.\n"
            "Format: Answer: number\n"
            "Answer:"
        )
    if answer_type == "code":
        return (
            f"{body}\n\n"
            "Complete the Python function. Return code only.\n"
        )
    return f"{body}\n\nOutput only the final answer.\nAnswer:"


def build_role_plan_ignore_prompt(
    row: dict[str, Any],
    role_outputs: dict[str, str],
    role: str,
) -> str:
    body = format_task_body(row)
    if role in {"planner", "critic", "refiner"}:
        return build_role_prompt(row, role_outputs, role, "generic_v1")
    refined_state = compact_role_state(role_outputs.get("refiner", ""))
    if row.get("answer_type") == "multiple_choice":
        answer_instruction = (
            "The upstream state may be noisy or irrelevant. Ignore it if it is not helpful. "
            "Choose the best option and output only: Answer: <letter>."
        )
    elif row.get("answer_type") == "numeric":
        answer_instruction = (
            "The upstream state may be noisy or irrelevant. Ignore it if it is not helpful. "
            "Solve the problem and output only: Answer: <number>."
        )
    elif row.get("answer_type") == "code":
        answer_instruction = (
            "The upstream state may be noisy or irrelevant. Ignore it if it is not helpful. "
            "Return code only."
        )
    else:
        answer_instruction = (
            "The upstream state may be noisy or irrelevant. Ignore it if it is not helpful. "
            "Answer directly."
        )
    return (
        f"{body}\n\n"
        f"Upstream state for reference:\n{refined_state}\n\n"
        f"Role: Solver.\n{answer_instruction}\nAnswer:"
    )


def build_answer_state_role_prompt(
    row: dict[str, Any],
    role_outputs: dict[str, str],
    role: str,
) -> str:
    body = format_task_body(row)
    target = answer_state_target(row)
    planner_state = compact_role_state(role_outputs.get("planner", ""))
    critic_state = compact_role_state(role_outputs.get("critic", ""))
    refiner_state = compact_role_state(role_outputs.get("refiner", ""))
    if role == "planner":
        return (
            f"{body}\n\n"
            "Role: Planner.\n"
            "Return a compact candidate answer state, not full reasoning.\n"
            f"Candidate: {target} or unknown\n"
            "Evidence: short phrase\n"
            "Planner state:"
        )
    if role == "critic":
        return (
            f"{body}\n\n"
            f"Planner state:\n{planner_state}\n\n"
            "Role: Critic.\n"
            "Check the candidate and return a corrected compact answer state.\n"
            f"Candidate: {target} or unknown\n"
            "Issue: short correction or none\n"
            "Critic state:"
        )
    if role == "refiner":
        return (
            f"{body}\n\n"
            f"Planner state:\n{planner_state}\n\n"
            f"Critic state:\n{critic_state}\n\n"
            "Role: Refiner.\n"
            "Merge the states into one compact answer state for the solver.\n"
            f"Candidate: {target}\n"
            "Evidence: short phrase\n"
            "Refined state:"
        )
    if row.get("answer_type") == "code":
        return (
            f"{body}\n\n"
            f"Upstream answer state:\n{refiner_state}\n\n"
            "Role: Solver.\n"
            "Use the upstream state as advice and return code only.\n"
        )
    return (
        f"{body}\n\n"
        f"Upstream answer state:\n{refiner_state}\n\n"
        "Role: Solver.\n"
        "Use the upstream state as advice. Output only the final answer line.\n"
        "Format: Answer: final value\n"
        "Answer:"
    )


def build_answer_state_structured_role_prompt(
    row: dict[str, Any],
    role_outputs: dict[str, str],
    role: str,
) -> str:
    body = format_task_body(row)
    target = answer_state_target(row)
    planner_state = structured_answer_state(row, role_outputs.get("planner", ""))
    critic_state = structured_answer_state(row, role_outputs.get("critic", ""))
    refiner_state = structured_answer_state(row, role_outputs.get("refiner", ""))
    if role == "planner":
        return (
            f"{body}\n\n"
            "Role: Planner.\n"
            "Return only a compact state.\n"
            f"Candidate: {target} or unknown\n"
            "Planner state:"
        )
    if role == "critic":
        return (
            f"{body}\n\n"
            f"Planner parsed state:\n{planner_state}\n\n"
            "Role: Critic.\n"
            "Correct the candidate if needed. Return only a compact state.\n"
            f"Candidate: {target} or unknown\n"
            "Critic state:"
        )
    if role == "refiner":
        return (
            f"{body}\n\n"
            f"Planner parsed state:\n{planner_state}\n\n"
            f"Critic parsed state:\n{critic_state}\n\n"
            "Role: Refiner.\n"
            "Select the best candidate. Return only a compact state.\n"
            f"Candidate: {target} or unknown\n"
            "Refined state:"
        )
    if row.get("answer_type") == "code":
        return (
            f"{body}\n\n"
            f"Refiner parsed state:\n{refiner_state}\n\n"
            "Role: Solver.\n"
            "Use the parsed upstream state as advice and return code only.\n"
        )
    return (
        f"{body}\n\n"
        f"Refiner parsed state:\n{refiner_state}\n\n"
        "Role: Solver.\n"
        "Use the parsed upstream state as advice. Output only the final answer line.\n"
        "Format: Answer: final value\n"
        "Answer:"
    )


def answer_state_target(row: dict[str, Any]) -> str:
    answer_type = row.get("answer_type")
    if answer_type == "multiple_choice":
        return "one option letter"
    if answer_type == "numeric":
        return "number"
    if answer_type == "code":
        return "code"
    return "answer"


def compact_role_state(text: str, *, max_chars: int = 600) -> str:
    value = re.sub(r"\s+", " ", str(text).strip())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def structured_answer_state(row: dict[str, Any], text: str) -> str:
    answer_type = row.get("answer_type")
    if answer_type == "multiple_choice":
        candidate = extract_choice_prediction(text, row.get("choices", [])) or "unknown"
    elif answer_type == "numeric":
        candidate = normalize_number(extract_last_number(text) or "") or "unknown"
    elif answer_type == "code":
        candidate = "code-draft" if text.strip() else "unknown"
    else:
        candidate = compact_role_state(text, max_chars=120) or "unknown"
    return f"Candidate: {candidate}"


def build_cola_fewshot_single_prompt(row: dict[str, Any]) -> str:
    answer_type = row.get("answer_type")
    body = format_task_body(row)
    if answer_type == "multiple_choice":
        return build_mcq_text_fewshot(row)
    if answer_type == "numeric":
        return (
            "Question: Janet has 3 apples and buys 4 more. How many apples does she have?\n"
            "Answer: 7\n\n"
            "Question: A box has 12 pencils. 5 are removed. How many remain?\n"
            "Answer: 7\n\n"
            f"{body}\n"
            "Answer:"
        )
    if answer_type == "code":
        return build_code_completion_prompt(row)
    return f"{body}\nAnswer:"


def build_cola_fewshot_role_prompt(
    row: dict[str, Any],
    role_outputs: dict[str, str],
    role: str,
) -> str:
    body = format_task_body(row)
    if role == "planner":
        return f"{body}\n\nRelevant information:"
    if role == "critic":
        return (
            f"{body}\n\n"
            f"Proposed information:\n{role_outputs.get('planner', '')}\n\n"
            "Correction:"
        )
    if role == "refiner":
        return (
            f"{body}\n\n"
            f"Information:\n{role_outputs.get('planner', '')}\n\n"
            f"Correction:\n{role_outputs.get('critic', '')}\n\n"
            "Useful final hint:"
        )
    solver_context = role_outputs.get("refiner", "")
    if row.get("answer_type") == "multiple_choice":
        return build_mcq_text_fewshot(row, extra_context=solver_context)
    if row.get("answer_type") == "numeric":
        return (
            "Question: Janet has 3 apples and buys 4 more. How many apples does she have?\n"
            "Answer: 7\n\n"
            f"{body}\n"
            f"Useful hint: {solver_context}\n"
            "Answer:"
        )
    if row.get("answer_type") == "code":
        return build_code_completion_prompt(row, extra_context=solver_context)
    return f"{body}\nUseful hint: {solver_context}\nAnswer:"


def build_mcq_text_fewshot(row: dict[str, Any], extra_context: str = "") -> str:
    current_choices = "\n".join(
        f"({choice['label']}) {choice['text']}" for choice in row.get("choices", [])
    )
    context_line = f"Useful hint: {extra_context}\n" if extra_context.strip() else ""
    return (
        "Question: Which gas do plants absorb from the air during photosynthesis?\n"
        "(A) Oxygen\n"
        "(B) Carbon dioxide\n"
        "(C) Nitrogen\n"
        "(D) Hydrogen\n"
        "Answer: Carbon dioxide\n\n"
        "Question: Which tool is best for tightening a screw?\n"
        "(A) spoon\n"
        "(B) hammer\n"
        "(C) screwdriver\n"
        "(D) paintbrush\n"
        "Answer: screwdriver\n\n"
        f"Question: {row.get('question', '')}\n"
        f"{current_choices}\n"
        f"{context_line}"
        "Answer:"
    )


def build_code_completion_prompt(row: dict[str, Any], extra_context: str = "") -> str:
    question = str(row.get("question", "")).rstrip()
    context_line = f"# Hint: {extra_context}\n" if extra_context.strip() else ""
    if re.search(r"(?m)^\s*def\s+\w+\s*\(", question):
        return question + "\n" + context_line
    return (
        "# Write a Python function for the following task.\n"
        f"# Task: {question}\n"
        f"{context_line}"
    )


def format_task_body(row: dict[str, Any]) -> str:
    answer_type = row.get("answer_type")
    question = str(row.get("question", ""))
    if answer_type == "multiple_choice":
        options = "\n".join(
            f"({choice['label']}) {choice['text']}" for choice in row.get("choices", [])
        )
        return f"Question:\n{question}\n\nOptions:\n{options}"
    if answer_type == "code":
        entry = row.get("entry_point")
        suffix = f"\nRequired entry point: {entry}" if entry else ""
        return f"Programming task:\n{question}{suffix}"
    return f"Question:\n{question}"


def score_candidate(
    row: dict[str, Any],
    text: str,
    *,
    config: ColaP2CapabilityGateConfig | None = None,
) -> dict[str, Any]:
    answer_type = row.get("answer_type")
    nonempty = bool(text.strip())
    if answer_type == "multiple_choice":
        return score_multiple_choice(row, text, nonempty=nonempty)
    if answer_type == "numeric":
        return score_numeric(row, text, nonempty=nonempty)
    if answer_type == "code":
        return score_code(row, text, nonempty=nonempty, config=config)
    return {
        "nonempty": nonempty,
        "parseable": nonempty,
        "prediction": text.strip(),
        "target": row.get("ground_truth", ""),
        "score": 0.0,
        "correct": False,
        "metric_kind": "unknown",
    }


def score_multiple_choice(row: dict[str, Any], text: str, *, nonempty: bool) -> dict[str, Any]:
    choices = row.get("choices", [])
    target = str(row.get("ground_truth", row.get("answer", ""))).strip()
    prediction = extract_choice_prediction(text, choices)
    correct = bool(prediction and target and prediction == target)
    return {
        "nonempty": nonempty,
        "parseable": bool(prediction),
        "prediction": prediction,
        "target": target,
        "score": 1.0 if correct else 0.0,
        "correct": correct,
        "metric_kind": "accuracy",
        "random_floor": 1.0 / max(len(choices), 1),
    }


def score_numeric(row: dict[str, Any], text: str, *, nonempty: bool) -> dict[str, Any]:
    target = normalize_number(str(row.get("ground_truth", row.get("answer", ""))))
    prediction = normalize_number(extract_last_number(text) or "")
    correct = bool(prediction and target and numbers_equal(prediction, target))
    return {
        "nonempty": nonempty,
        "parseable": bool(prediction),
        "prediction": prediction,
        "target": target,
        "score": 1.0 if correct else 0.0,
        "correct": correct,
        "metric_kind": "numeric_exact",
        "random_floor": 0.0,
    }


def score_code(
    row: dict[str, Any],
    text: str,
    *,
    nonempty: bool,
    config: ColaP2CapabilityGateConfig | None,
) -> dict[str, Any]:
    completion = extract_code(text)
    code = compose_code_candidate(row, completion)
    syntax_ok = False
    entry_point_present = False
    error = ""
    try:
        ast.parse(code)
        syntax_ok = bool(code.strip())
    except SyntaxError as exc:
        error = str(exc)
    entry_point = str(row.get("entry_point", "")).strip()
    if syntax_ok and entry_point:
        entry_point_present = bool(re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", code))
    execution_enabled = bool(config and config.enable_code_execution)
    execution_result = {
        "enabled": execution_enabled,
        "passed": False,
        "timed_out": False,
        "returncode": None,
        "stderr_tail": "",
    }
    if syntax_ok and execution_enabled:
        execution_result = run_code_execution_gate(
            row,
            code,
            timeout_seconds=config.code_timeout_seconds if config else 5.0,
        )
    if execution_enabled:
        score = float(bool(execution_result["passed"]))
        metric_kind = "code_execution"
        correct = bool(execution_result["passed"])
    else:
        score = float(syntax_ok and (entry_point_present or not entry_point))
        metric_kind = "code_syntax_only"
        correct = False
    return {
        "nonempty": nonempty,
        "parseable": syntax_ok,
        "prediction": "",
        "target": row.get("entry_point", ""),
        "score": score,
        "correct": correct,
        "metric_kind": metric_kind,
        "requires_execution_gate": not execution_enabled,
        "execution": execution_result,
        "entry_point_present": entry_point_present,
        "syntax_error": error,
        "random_floor": 0.0,
    }


def extract_choice_prediction(text: str, choices: list[dict[str, str]]) -> str:
    labels = [str(choice["label"]).strip() for choice in choices]
    label_set = set(labels)
    answer_segment = text
    marker_match = list(re.finditer(r"(?i)\banswer\s*[:：]\s*", text))
    if marker_match:
        answer_segment = text[marker_match[-1].end() :]
        answer_segment = strip_answer_placeholder(answer_segment)
    leading = re.match(r"^\s*\(?([A-E])\)?(?:[\s).,:;-]|$)", answer_segment)
    if leading and leading.group(1).upper() in label_set:
        return leading.group(1).upper()
    for pattern in [
        r"\(([A-E])\)",
        r"(?<![A-Za-z])([A-E])(?![A-Za-z])",
        r"(?i)\boption\s+([A-E])\b",
        r"(?i)\bchoice\s+([A-E])\b",
    ]:
        for match in re.finditer(pattern, answer_segment):
            value = match.group(1).upper()
            if value in label_set:
                return value
    normalized = normalize_text(answer_segment)
    sorted_choices = sorted(choices, key=lambda item: len(item["text"]), reverse=True)
    for choice in sorted_choices:
        if normalize_text(choice["text"]) and normalize_text(choice["text"]) in normalized:
            return str(choice["label"]).strip()
    return ""


def strip_answer_placeholder(answer_segment: str) -> str:
    value = answer_segment.lstrip()
    if not value.startswith("<"):
        return answer_segment
    match = re.match(r"^<([^>]*)>\s*", value)
    if not match:
        return answer_segment
    placeholder = match.group(1).lower()
    if "option" not in placeholder and "letter" not in placeholder and "answer" not in placeholder:
        return answer_segment
    return value[match.end() :]


def extract_last_number(text: str) -> str:
    numbers = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text)
    return numbers[-1] if numbers else ""


def normalize_number(value: str) -> str:
    return value.strip().replace(",", "")


def numbers_equal(lhs: str, rhs: str) -> bool:
    try:
        return math.isclose(float(lhs), float(rhs), rel_tol=1e-9, abs_tol=1e-9)
    except ValueError:
        return lhs == rhs


def extract_code(text: str) -> str:
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text.rstrip("\n")


def compose_code_candidate(row: dict[str, Any], completion: str) -> str:
    question = str(row.get("question", "")).rstrip()
    entry_point = str(row.get("entry_point", "")).strip()
    if entry_point and re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", completion):
        return completion
    if entry_point and re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", question):
        return question + "\n" + completion
    return completion


def run_code_execution_gate(
    row: dict[str, Any],
    code: str,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    test = str(row.get("test", ""))
    entry_point = str(row.get("entry_point", "")).strip()
    script = code.rstrip() + "\n\n" + test.strip() + "\n"
    if entry_point and re.search(r"\bdef\s+check\s*\(\s*candidate\s*\)", test):
        script += f"\ncheck({entry_point})\n"
    with TemporaryDirectory(prefix="drla_p2_code_gate_") as tmp:
        script_path = Path(tmp) / "candidate_test.py"
        script_path.write_text(script, encoding="utf-8")
        try:
            result = subprocess.run(
                ["python", str(script_path)],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            return {
                "enabled": True,
                "passed": result.returncode == 0,
                "timed_out": False,
                "returncode": result.returncode,
                "stderr_tail": result.stderr[-1000:],
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "enabled": True,
                "passed": False,
                "timed_out": True,
                "returncode": None,
                "stderr_tail": str(exc)[-1000:],
            }


def build_task_summaries(
    records: list[dict[str, Any]],
    *,
    config: ColaP2CapabilityGateConfig,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["task"], record["mode"])].append(record)
    summaries: list[dict[str, Any]] = []
    for (task, mode), items in sorted(grouped.items()):
        answer_type = str(items[0].get("answer_type", ""))
        scores = [float(item["score"]["score"]) for item in items]
        correct = [1.0 if item["score"].get("correct") else 0.0 for item in items]
        nonempty = [1.0 if item["score"].get("nonempty") else 0.0 for item in items]
        parseable = [1.0 if item["score"].get("parseable") else 0.0 for item in items]
        floors = [float(item["score"].get("random_floor", 0.0)) for item in items]
        random_floor = statistics.mean(floors) if floors else 0.0
        accuracy = statistics.mean(correct) if correct else 0.0
        score_mean = statistics.mean(scores) if scores else 0.0
        nonempty_rate = statistics.mean(nonempty) if nonempty else 0.0
        parseable_rate = statistics.mean(parseable) if parseable else 0.0
        meets_format_gate = (
            nonempty_rate >= config.min_nonempty_rate
            and parseable_rate >= config.min_parseable_rate
        )
        requires_execution_gate = any(
            bool(item["score"].get("requires_execution_gate", False)) for item in items
        )
        meets_accuracy_gate = accuracy >= random_floor + config.min_accuracy_margin
        summaries.append(
            {
                "task": task,
                "mode": mode,
                "answer_type": answer_type,
                "num_samples": len(items),
                "nonempty_rate": nonempty_rate,
                "parseable_rate": parseable_rate,
                "accuracy": accuracy,
                "score_mean": score_mean,
                "random_floor": random_floor,
                "meets_format_gate": meets_format_gate,
                "meets_accuracy_gate": meets_accuracy_gate,
                "requires_execution_gate": requires_execution_gate,
                "gate_pass": bool(
                    meets_format_gate and meets_accuracy_gate and not requires_execution_gate
                ),
                "admitted_for_main": False,
            }
        )
    apply_task_admission(summaries, config=config)
    return summaries


def apply_task_admission(
    summaries: list[dict[str, Any]],
    *,
    config: ColaP2CapabilityGateConfig,
) -> None:
    requested_modes = set(normalize_modes(config.modes))
    if config.max_samples_per_task or not {"single", "role_textmas"}.issubset(requested_modes):
        return
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in summaries:
        by_task[item["task"]].append(item)
    for task_items in by_task.values():
        by_mode = {item["mode"]: item for item in task_items}
        admitted = all(
            by_mode.get(mode, {}).get("gate_pass", False)
            for mode in ["single", "role_textmas"]
        )
        for item in task_items:
            item["admitted_for_main"] = bool(admitted)


def write_records(file_obj: Any, records: list[dict[str, Any]]) -> None:
    for record in records:
        file_obj.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    file_obj.flush()


def write_incremental_metrics(
    file_obj: Any,
    records: list[dict[str, Any]],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    rows = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[record["task"]].append(record)
    for task, items in sorted(by_task.items()):
        row = {
            "created_at": int(time.time()),
            "step": len(items),
            "mode": mode,
            "task": task,
            "num_samples": len(items),
            "accuracy": statistics.mean(
                [1.0 if item["score"].get("correct") else 0.0 for item in items]
            ),
            "score_mean": statistics.mean([float(item["score"].get("score", 0.0)) for item in items]),
            "nonempty_rate": statistics.mean(
                [1.0 if item["score"].get("nonempty") else 0.0 for item in items]
            ),
            "parseable_rate": statistics.mean(
                [1.0 if item["score"].get("parseable") else 0.0 for item in items]
            ),
        }
        file_obj.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        rows.append(row)
    file_obj.flush()
    return rows


def write_task_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fieldnames = [
        "task",
        "mode",
        "answer_type",
        "num_samples",
        "nonempty_rate",
        "parseable_rate",
        "accuracy",
        "score_mean",
        "random_floor",
        "meets_format_gate",
        "meets_accuracy_gate",
        "requires_execution_gate",
        "gate_pass",
        "admitted_for_main",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def select_rows(
    rows: list[dict[str, Any]],
    *,
    tasks: str,
    max_samples_per_task: int,
) -> list[dict[str, Any]]:
    allowed = {task.strip() for task in tasks.split(",") if task.strip()}
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task = str(row.get("task", ""))
        if allowed and task not in allowed:
            continue
        by_task[task].append(row)
    selected: list[dict[str, Any]] = []
    for task in sorted(by_task):
        task_rows = by_task[task]
        if max_samples_per_task:
            task_rows = task_rows[:max_samples_per_task]
        selected.extend(task_rows)
    return selected


def normalize_modes(modes: str) -> list[str]:
    values = [mode.strip() for mode in modes.split(",") if mode.strip()]
    unknown = sorted(set(values) - set(MODES))
    if unknown:
        raise ValueError(f"Unknown modes: {unknown}; known modes: {MODES}")
    return values


def normalize_text(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text).strip().lower())
    return re.sub(r"[^a-z0-9 ._+-]+", "", value)


def add_cola_code_path(path: str) -> None:
    import sys

    if path and path not in sys.path:
        sys.path.insert(0, path)


if __name__ == "__main__":
    main()
