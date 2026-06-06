"""Run native official8 single-solver calibration for P2 Branch B.

This local-only audit uses Cola's original ``apply_prompt_template`` task
interfaces and acc_calc-style scoring.  It is meant to check whether the
normalized P2 gate is misaligned with the official Cola benchmark interface
before any Role TextMAS, held-out, or latent-vs-text experiment is attempted.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from drla.scripts.collect_cola_block_traces import (
    ColaBlockTraceConfig,
    load_cola_symbols,
    read_jsonl,
    resolve_device,
    set_seed,
)
from drla.tracking import require_swanlab_disabled_for_non_training


DEFAULT_INPUT_JSONL = (
    "/data1/luyifei/drla/outputs/p2_benchmark_redesign/"
    "official8_role_candidates_splits_seed20260603_20260601/calibration.jsonl"
)
DEFAULT_RAW_TASK_DIR = "/data1/luyifei/Cola-DLM/code/generate_task_data"
DEFAULT_COLA_CODE_PATH = "/data1/luyifei/Cola-DLM/code"
OFFICIAL8_TASKS = ["obqa", "mmlu", "race", "hellaswag", "siqa", "story_cloze"]


@dataclass(frozen=True)
class Official8NativeSingleGateConfig:
    input_jsonl: str = DEFAULT_INPUT_JSONL
    output_dir: str = (
        "/data1/luyifei/drla/outputs/p2_benchmark_redesign/"
        "eval_calibration_official8_native_single_20260601"
    )
    raw_task_dir: str = DEFAULT_RAW_TASK_DIR
    cola_code_path: str = DEFAULT_COLA_CODE_PATH
    dit_path: str = ColaBlockTraceConfig.dit_path
    vae_path: str = ColaBlockTraceConfig.vae_path
    tokenizer_path: str = ColaBlockTraceConfig.tokenizer_path
    tasks: str = ",".join(OFFICIAL8_TASKS)
    batch_size: int = 8
    max_samples_per_task: int = 0
    seed: int = 20260601
    per_sample_noise_seed: int = 66
    max_new_tokens: int = 32
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
    overwrite: bool = False


def main() -> None:
    summary = run_native_single_gate(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> Official8NativeSingleGateConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", default=Official8NativeSingleGateConfig.input_jsonl)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-task-dir", default=Official8NativeSingleGateConfig.raw_task_dir)
    parser.add_argument("--cola-code-path", default=Official8NativeSingleGateConfig.cola_code_path)
    parser.add_argument("--dit-path", default=Official8NativeSingleGateConfig.dit_path)
    parser.add_argument("--vae-path", default=Official8NativeSingleGateConfig.vae_path)
    parser.add_argument("--tokenizer-path", default=Official8NativeSingleGateConfig.tokenizer_path)
    parser.add_argument("--tasks", default=Official8NativeSingleGateConfig.tasks)
    parser.add_argument("--batch-size", type=int, default=Official8NativeSingleGateConfig.batch_size)
    parser.add_argument("--max-samples-per-task", type=int, default=0)
    parser.add_argument("--seed", type=int, default=Official8NativeSingleGateConfig.seed)
    parser.add_argument(
        "--per-sample-noise-seed",
        type=int,
        default=Official8NativeSingleGateConfig.per_sample_noise_seed,
    )
    parser.add_argument("--max-new-tokens", type=int, default=Official8NativeSingleGateConfig.max_new_tokens)
    parser.add_argument("--timestep-num", type=int, default=Official8NativeSingleGateConfig.timestep_num)
    parser.add_argument("--guidance-scale", type=float, default=Official8NativeSingleGateConfig.guidance_scale)
    parser.add_argument("--temperature", type=float, default=Official8NativeSingleGateConfig.temperature)
    parser.add_argument("--top-k", type=int, default=Official8NativeSingleGateConfig.top_k)
    parser.add_argument("--top-p", type=float, default=Official8NativeSingleGateConfig.top_p)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=Official8NativeSingleGateConfig.repetition_penalty,
    )
    parser.add_argument("--pad-token-id", type=int, default=Official8NativeSingleGateConfig.pad_token_id)
    parser.add_argument("--eos-token-id", type=int, default=Official8NativeSingleGateConfig.eos_token_id)
    parser.add_argument("--im-end-token-id", type=int, default=Official8NativeSingleGateConfig.im_end_token_id)
    parser.add_argument("--device", default=Official8NativeSingleGateConfig.device)
    parser.add_argument("--swanlab-mode", default=Official8NativeSingleGateConfig.swanlab_mode)
    parser.add_argument("--min-nonempty-rate", type=float, default=Official8NativeSingleGateConfig.min_nonempty_rate)
    parser.add_argument("--min-parseable-rate", type=float, default=Official8NativeSingleGateConfig.min_parseable_rate)
    parser.add_argument("--min-accuracy-margin", type=float, default=Official8NativeSingleGateConfig.min_accuracy_margin)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_samples_per_task < 0:
        raise ValueError("--max-samples-per-task must be non-negative")
    return Official8NativeSingleGateConfig(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        raw_task_dir=args.raw_task_dir,
        cola_code_path=args.cola_code_path,
        dit_path=args.dit_path,
        vae_path=args.vae_path,
        tokenizer_path=args.tokenizer_path,
        tasks=",".join(normalize_tasks(args.tasks)),
        batch_size=args.batch_size,
        max_samples_per_task=args.max_samples_per_task,
        seed=args.seed,
        per_sample_noise_seed=args.per_sample_noise_seed,
        max_new_tokens=args.max_new_tokens,
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
        overwrite=args.overwrite,
    )


def run_native_single_gate(config: Official8NativeSingleGateConfig) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        config.swanlab_mode,
        script_kind="Cola P2 official8 native single gate",
    )
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = select_rows(read_jsonl(Path(config.input_jsonl)), config=config)
    if not rows:
        raise ValueError("No rows selected")
    raw_by_task = load_raw_task_rows(Path(config.raw_task_dir), normalize_tasks(config.tasks))
    prompts_by_task = build_native_prompts_by_task(rows, raw_by_task)

    set_seed(config.seed)
    add_cola_code_path(config.cola_code_path)
    os.environ["COLA_INFER_PER_SAMPLE_NOISE_SEED"] = str(config.per_sample_noise_seed)
    device = resolve_device(config.device)
    if device.type != "cuda":
        raise RuntimeError(f"Official Cola native generation requires CUDA. Resolved: {device}")

    cola = load_cola_symbols()
    from cola_dlm import generate_task_repaint_inference

    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    dit = cola["ColaDiTModel"].from_pretrained(config.dit_path).to(device).eval()
    vae = cola["ColaTextVAEModel"].from_pretrained(config.vae_path).to(device).eval()

    generations_path = output_dir / "generations.jsonl"
    metrics_path = output_dir / "metrics.jsonl"
    task_summary_path = output_dir / "task_summary.csv"
    summary_path = output_dir / "summary.json"
    started = time.time()
    all_records: list[dict[str, Any]] = []

    with generations_path.open("w", encoding="utf-8") as gen_f:
        for task in sorted(prompts_by_task):
            prompts = prompts_by_task[task]
            task_records = run_task_generation(
                task=task,
                prompts=prompts,
                config=config,
                dit=dit,
                vae=vae,
                tokenizer=tokenizer,
                generate_task_repaint_inference=generate_task_repaint_inference,
                device=device,
            )
            for record in task_records:
                gen_f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            all_records.extend(task_records)

    summaries = build_task_summaries(all_records, config=config)
    write_task_summary_csv(task_summary_path, summaries)
    write_metrics_jsonl(metrics_path, summaries)
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
        "task_summaries": summaries,
        "admitted_tasks": [] if config.max_samples_per_task else [
            row["task"] for row in summaries if row["gate_pass"]
        ],
        "notes": [
            "Pure official8 native single-solver calibration; SwanLab disabled.",
            "Uses Cola apply_prompt_template through native task_name grouping.",
            "Uses acc_calc-style choice-text/similarity scoring; no Role TextMAS or held-out.",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def run_task_generation(
    *,
    task: str,
    prompts: list[dict[str, Any]],
    config: Official8NativeSingleGateConfig,
    dit: Any,
    vae: Any,
    tokenizer: Tokenizer,
    generate_task_repaint_inference: Any,
    device: torch.device,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(prompts), config.batch_size):
            batch = prompts[start : start + config.batch_size]
            generated = generate_task_repaint_inference(
                dit=dit,
                vae=vae,
                tokenizer=tokenizer,
                prompts=batch,
                task_name=task,
                device=device,
                timestep_num=config.timestep_num,
                guidance_scale=config.guidance_scale,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                pad_token_id=config.pad_token_id,
                eos_token_id=config.eos_token_id,
                im_end_token_id=config.im_end_token_id,
                is_sft=False,
            )
            for prompt, out in zip(batch, generated):
                text = str(out.get("generate", ""))
                score = score_official8(task, text, prompt)
                records.append(
                    {
                        "id": prompt["p2_id"],
                        "raw_id": prompt["id"],
                        "task": f"official8_{task}",
                        "source_task": task,
                        "mode": "official8_native_single",
                        "prompt": out.get("prompt", ""),
                        "generate": text,
                        "ground_truth": prompt.get("ground_truth", prompt.get("answer", "")),
                        "choices": prompt.get("choices", []),
                        "score": score,
                    }
                )
    return records


def build_native_prompts_by_task(
    rows: list[dict[str, Any]],
    raw_by_task: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task = str(row.get("source_task") or str(row.get("task", "")).replace("official8_", ""))
        raw_id = str(row.get("id", "")).split(":")[-1]
        raw = raw_by_task[task][raw_id]
        prompt = {
            "id": raw["id"],
            "p2_id": row["id"],
            "question": raw.get("question", ""),
            "answer": raw.get("answer", raw.get("ground_truth", "")),
            "ground_truth": raw.get("ground_truth", raw.get("answer", "")),
            "choices": raw.get("choices", []),
        }
        if "context" in raw:
            prompt["context"] = raw["context"]
        by_task[task].append(prompt)
    return by_task


def load_raw_task_rows(raw_dir: Path, tasks: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    raw_by_task = {}
    for task in tasks:
        path = raw_dir / f"{task}.jsonl"
        task_rows = {}
        for row in read_jsonl(path):
            task_rows[str(row["id"])] = row
        raw_by_task[task] = task_rows
    return raw_by_task


def select_rows(rows: list[dict[str, Any]], *, config: Official8NativeSingleGateConfig) -> list[dict[str, Any]]:
    allowed = {f"official8_{task}" for task in normalize_tasks(config.tasks)}
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task = str(row.get("task", ""))
        if task in allowed:
            by_task[task].append(row)
    selected = []
    for task in sorted(by_task):
        task_rows = by_task[task]
        if config.max_samples_per_task:
            task_rows = task_rows[: config.max_samples_per_task]
        selected.extend(task_rows)
    return selected


def score_official8(task: str, text: str, prompt: dict[str, Any]) -> dict[str, Any]:
    eval_text = official_eval_first_segment(text)
    nonempty = bool(text.strip())
    gt = str(prompt.get("ground_truth", prompt.get("answer", "")))
    choices = prompt.get("choices", [])
    if task in {"mmlu", "obqa", "race", "siqa"}:
        pred_choice = extract_choice_letter_or_text(eval_text, choices)
        gt_choice = extract_choice_letter_or_text(gt, choices)
        choice_correct = bool(pred_choice and gt_choice and pred_choice == gt_choice)
        sim_score = calculate_similarity(eval_text, gt)
        score = 1.0 if choice_correct else sim_score
        correct = score >= 1.0
        return {
            "nonempty": nonempty,
            "parseable": bool(pred_choice),
            "prediction": pred_choice,
            "target": gt_choice,
            "evaluated_text": eval_text,
            "similarity_score": round(sim_score, 4),
            "score": 1.0 if correct else 0.0,
            "correct": correct,
            "metric_kind": "official_choice_or_exact_similarity",
            "random_floor": 1.0 / max(len(choices), 1),
        }
    sim_score = calculate_similarity(eval_text, gt)
    correct = sim_score >= 1.0
    return {
        "nonempty": nonempty,
        "parseable": nonempty,
        "prediction": eval_text.strip(),
        "target": gt,
        "evaluated_text": eval_text,
        "similarity_score": round(sim_score, 4),
        "score": 1.0 if correct else 0.0,
        "correct": correct,
        "metric_kind": "official_exact_similarity",
        "random_floor": 1.0 / max(len(choices), 1) if choices else 0.0,
    }


def official_eval_first_segment(text: str) -> str:
    content = str(text)
    current_pos = 0
    while True:
        idx = content.find("\n", current_pos)
        if idx == -1:
            return content.strip()
        part1 = content[:idx]
        if any(ch.isalnum() or ch == "_" for ch in part1):
            return part1.strip()
        current_pos = idx + 1


def build_task_summaries(
    records: list[dict[str, Any]],
    *,
    config: Official8NativeSingleGateConfig,
) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[record["task"]].append(record)
    rows = []
    for task, items in sorted(by_task.items()):
        scores = [float(item["score"]["score"]) for item in items]
        parseable = [bool(item["score"].get("parseable", False)) for item in items]
        nonempty = [bool(item["score"].get("nonempty", False)) for item in items]
        floors = [float(item["score"].get("random_floor", 0.0)) for item in items]
        accuracy = sum(scores) / len(scores)
        parseable_rate = sum(parseable) / len(parseable)
        nonempty_rate = sum(nonempty) / len(nonempty)
        random_floor = sum(floors) / len(floors)
        meets_format_gate = (
            nonempty_rate >= config.min_nonempty_rate
            and parseable_rate >= config.min_parseable_rate
        )
        meets_accuracy_gate = accuracy >= random_floor + config.min_accuracy_margin
        rows.append(
            {
                "task": task,
                "mode": "official8_native_single",
                "num_samples": len(items),
                "accuracy": accuracy,
                "score_mean": accuracy,
                "random_floor": random_floor,
                "nonempty_rate": nonempty_rate,
                "parseable_rate": parseable_rate,
                "meets_format_gate": meets_format_gate,
                "meets_accuracy_gate": meets_accuracy_gate,
                "gate_pass": meets_format_gate and meets_accuracy_gate,
            }
        )
    return rows


def write_task_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_tasks(tasks: str) -> list[str]:
    values = [task.strip().replace("official8_", "") for task in tasks.split(",") if task.strip()]
    unknown = sorted(set(values) - set(OFFICIAL8_TASKS))
    if unknown:
        raise ValueError(f"Unknown official8 task(s): {unknown}; known={OFFICIAL8_TASKS}")
    return values


def normalize_text(text: str) -> str:
    import re

    value = str(text).lower().strip()
    value = re.sub(r"[^\w\s]", "", value)
    return " ".join(value.split())


def calculate_similarity(text1: str, text2: str) -> float:
    import difflib

    norm_t1 = normalize_text(text1)
    norm_t2 = normalize_text(text2)
    if not norm_t1 and not norm_t2:
        return 1.0
    return difflib.SequenceMatcher(None, norm_t1, norm_t2).ratio()


def extract_answer_segment(text: str) -> str:
    import re

    raw = str(text).strip()
    if not raw:
        return ""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in reversed(lines):
        m = re.search(r"(?i)\b(?:final\s+answer|answer)\b\s*(?:is|=|:|：)?\s*(.+)$", line)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return raw


def extract_choice_letter(text: str, max_choices: int) -> str:
    import re

    raw = str(text).strip()
    if not raw:
        return ""
    m = re.fullmatch(r"[\(\[]?\s*([A-Za-z])\s*[\)\]]?\.?", raw)
    if m:
        letter = m.group(1).upper()
        if 0 <= (ord(letter) - 65) < max_choices:
            return letter
    keyword_pattern = re.compile(
        r"(?i)\b(?:final\s+answer|answer|option|choice)\b\s*(?:is|=|:|：)?\s*[\(\[]?\s*([A-Za-z])\s*[\)\]]?(?=\s|$|[.,;:!?])"
    )
    matches = keyword_pattern.findall(raw)
    if matches:
        letter = matches[-1].upper()
        if 0 <= (ord(letter) - 65) < max_choices:
            return letter
    if len(raw) <= 40:
        bracket_matches = re.findall(r"[\(\[]\s*([A-Za-z])\s*[\)\]]", raw)
        if bracket_matches:
            letter = bracket_matches[-1].upper()
            if 0 <= (ord(letter) - 65) < max_choices:
                return letter
    return ""


def extract_choice_letter_or_text(text: str, choices: list[str]) -> str:
    max_choices = min(len(choices), 26)
    if max_choices == 0:
        return ""
    for candidate in [extract_answer_segment(text), str(text)]:
        letter = extract_choice_letter(candidate, max_choices)
        if letter:
            return letter
        matched = match_choice_by_text(candidate, choices)
        if matched:
            return matched
    return ""


def match_choice_by_text(text: str, choices: list[str]) -> str:
    import re

    norm_text = normalize_text(text)
    if not norm_text:
        return ""
    for idx, choice in enumerate(choices):
        if normalize_text(choice) == norm_text:
            return chr(65 + idx)
    cleaned = re.sub(
        r"(?i)^(the\s+)?(correct\s+)?(final\s+)?(answer|option|choice)\b\s*(is|=|:|：)?\s*",
        "",
        str(text),
    ).strip()
    norm_cleaned = normalize_text(cleaned)
    if not norm_cleaned:
        return ""
    for idx, choice in enumerate(choices):
        if normalize_text(choice) == norm_cleaned:
            return chr(65 + idx)
    contained = [
        idx
        for idx, choice in enumerate(choices)
        if normalize_text(choice) and normalize_text(choice) in norm_cleaned
    ]
    if len(contained) == 1:
        return chr(65 + contained[0])
    best_idx, best_score = -1, 0.0
    for idx, choice in enumerate(choices):
        score = calculate_similarity(cleaned, choice)
        if score > best_score:
            best_idx = idx
            best_score = score
    if best_idx >= 0 and best_score >= 0.9:
        return chr(65 + best_idx)
    return ""


def add_cola_code_path(path: str) -> None:
    import sys

    if path not in sys.path:
        sys.path.insert(0, path)


if __name__ == "__main__":
    main()
