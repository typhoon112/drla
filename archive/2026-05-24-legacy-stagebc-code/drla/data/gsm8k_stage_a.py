"""Stage A data preparation and block statistics for GSM8K."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from transformers import AutoTokenizer

from drla.data.answer_judge import extract_answer_text, judge, normalize_answer
from drla.tracking import finish_experiment, init_experiment, log_metrics


DEFAULT_TOKENIZER = "Qwen/Qwen3-4B-Instruct-2507"


@dataclass(frozen=True)
class StageAConfig:
    dataset_name: str = "gsm8k"
    dataset_config: str = "main"
    tokenizer_name: str = DEFAULT_TOKENIZER
    block_size: int = 16
    patch_size: int = 1
    b_max: int = 32
    output_dir: str = "/data1/luyifei/drla/data/stage_a"
    sample_limit: int | None = None
    splits: tuple[str, ...] = ("train", "test")
    seed: int = 42
    swanlab_mode: str = "disabled"


def build_stage_a(config: StageAConfig) -> dict[str, Any]:
    """Create normalized GSM8K JSONL files, token files, and summary statistics."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, trust_remote_code=True)
    dataset = load_dataset(config.dataset_name, config.dataset_config)
    all_stats: dict[str, Any] = {}

    init_experiment(
        stage="stage-a",
        experiment_name=f"stage-a-gsm8k-bs{config.block_size}-limit{config.sample_limit or 'all'}",
        config=asdict(config),
        mode=config.swanlab_mode,
    )
    try:
        for split in config.splits:
            split_data = dataset[split]
            if config.sample_limit is not None:
                split_data = split_data.select(range(min(config.sample_limit, len(split_data))))

            samples = [_convert_raw_sample(row, split=split, index=i) for i, row in enumerate(split_data)]
            tokenized = [
                _tokenize_sample(
                    sample,
                    tokenizer=tokenizer,
                    block_size=config.block_size,
                    patch_size=config.patch_size,
                    b_max=config.b_max,
                )
                for sample in samples
            ]

            _write_jsonl(output_dir / f"gsm8k_{split}.jsonl", samples)
            _write_jsonl(output_dir / f"gsm8k_{split}.tokenized.jsonl", tokenized)

            split_stats = _summarize_split(samples, tokenized, b_max=config.b_max)
            all_stats[split] = split_stats
            log_metrics(_flatten_metrics(split_stats), prefix=split)

        judge_stats = _run_judge_self_check()
        all_stats["judge_self_check"] = judge_stats
        all_stats["config"] = asdict(config)
        all_stats["tokenizer_vocab_size"] = len(tokenizer)
        all_stats["created_at"] = int(time.time())
        all_stats["output_dir"] = str(output_dir)

        _write_json(output_dir / "summary.json", all_stats)
        log_metrics(_flatten_metrics({"judge_self_check": judge_stats}), prefix="stage_a")
    finally:
        finish_experiment()

    return all_stats


def _convert_raw_sample(row: dict[str, str], *, split: str, index: int) -> dict[str, Any]:
    answer_text = row["answer"]
    final_answer = extract_answer_text(answer_text)
    answer_norm = normalize_answer(final_answer)
    if answer_norm is None:
        raise ValueError(f"Could not normalize GSM8K answer at {split}[{index}]: {answer_text!r}")
    return {
        "id": f"gsm8k_{split}_{index:06d}",
        "question": row["question"],
        "solution_trace": answer_text,
        "final_answer": answer_norm,
        "target_text": answer_text,
        "source": "gsm8k",
        "metadata": {
            "teacher_model": None,
            "split": split,
            "raw_final_answer": final_answer,
        },
    }


def _tokenize_sample(
    sample: dict[str, Any],
    *,
    tokenizer: Any,
    block_size: int,
    patch_size: int,
    b_max: int,
) -> dict[str, Any]:
    question_ids = tokenizer.encode(sample["question"], add_special_tokens=False)
    target_ids = tokenizer.encode(sample["target_text"], add_special_tokens=False)
    block_capacity = block_size * patch_size
    b_star = max(1, math.ceil(len(target_ids) / block_capacity))
    block_mask = [1 if i < min(b_star, b_max) else 0 for i in range(b_max)]
    noop_mask = [0 if i < min(b_star, b_max) else 1 for i in range(b_max)]
    return {
        "id": sample["id"],
        "question_ids": question_ids,
        "target_ids": target_ids,
        "answer_norm": sample["final_answer"],
        "target_len": len(target_ids),
        "question_len": len(question_ids),
        "B_star": b_star,
        "B_max": b_max,
        "truncated_by_B_max": b_star > b_max,
        "block_mask": block_mask,
        "noop_mask": noop_mask,
    }


def _summarize_split(
    samples: list[dict[str, Any]], tokenized: list[dict[str, Any]], *, b_max: int
) -> dict[str, Any]:
    target_lens = [item["target_len"] for item in tokenized]
    question_lens = [item["question_len"] for item in tokenized]
    b_stars = [item["B_star"] for item in tokenized]
    judged = [judge(sample["target_text"], sample["final_answer"]) for sample in samples]
    answer_found_rate = sum(result["answer_found"] for result in judged) / len(judged)
    judge_acc = sum(result["correct"] for result in judged) / len(judged)
    b_hist = Counter(min(item["B_star"], b_max) for item in tokenized)
    return {
        "num_samples": len(samples),
        "judge_acc_on_gold_target": judge_acc,
        "answer_found_rate_on_gold_target": answer_found_rate,
        "target_len": _describe_numbers(target_lens),
        "question_len": _describe_numbers(question_lens),
        "B_star": _describe_numbers(b_stars),
        "truncated_by_B_max_count": sum(item["truncated_by_B_max"] for item in tokenized),
        "truncated_by_B_max_rate": sum(item["truncated_by_B_max"] for item in tokenized)
        / len(tokenized),
        "B_star_clipped_histogram": dict(sorted(b_hist.items())),
    }


def _run_judge_self_check() -> dict[str, Any]:
    cases = [
        ("We compute it. #### 42", "42", True),
        ("The answer is 1,234.", "1234", True),
        ("Final answer: 50%", "0.5", True),
        ("After simplification, #### 3/4", "0.75", True),
        ("No numeric answer here", "7", False),
    ]
    results = [judge(pred, gold) for pred, gold, _ in cases]
    passed = [result["correct"] == expected for result, (_, _, expected) in zip(results, cases)]
    return {
        "num_cases": len(cases),
        "passed": sum(passed),
        "pass_rate": sum(passed) / len(passed),
        "failures": [
            {"case": cases[i][0], "gold": cases[i][1], "result": results[i]}
            for i, ok in enumerate(passed)
            if not ok
        ],
    }


def _describe_numbers(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {}
    sorted_values = sorted(values)
    return {
        "min": sorted_values[0],
        "p25": _percentile(sorted_values, 0.25),
        "median": statistics.median(sorted_values),
        "mean": statistics.fmean(sorted_values),
        "p75": _percentile(sorted_values, 0.75),
        "p90": _percentile(sorted_values, 0.90),
        "p95": _percentile(sorted_values, 0.95),
        "max": sorted_values[-1],
    }


def _percentile(sorted_values: list[int], q: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    index = (len(sorted_values) - 1) * q
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return float(sorted_values[lo])
    weight = index - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def _flatten_metrics(value: dict[str, Any], *, prefix: str = "") -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, bool):
            result[name] = int(item)
        elif isinstance(item, (int, float)):
            result[name] = item
        elif isinstance(item, dict):
            result.update(_flatten_metrics(item, prefix=name))
    return result


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def parse_args() -> StageAConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=StageAConfig.dataset_name)
    parser.add_argument("--dataset-config", default=StageAConfig.dataset_config)
    parser.add_argument("--tokenizer-name", default=StageAConfig.tokenizer_name)
    parser.add_argument("--block-size", type=int, default=StageAConfig.block_size)
    parser.add_argument("--patch-size", type=int, default=StageAConfig.patch_size)
    parser.add_argument("--b-max", type=int, default=StageAConfig.b_max)
    parser.add_argument("--output-dir", default=StageAConfig.output_dir)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--splits", nargs="+", default=list(StageAConfig.splits))
    parser.add_argument("--seed", type=int, default=StageAConfig.seed)
    parser.add_argument("--swanlab-mode", default=StageAConfig.swanlab_mode)
    args = parser.parse_args()
    return StageAConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        tokenizer_name=args.tokenizer_name,
        block_size=args.block_size,
        patch_size=args.patch_size,
        b_max=args.b_max,
        output_dir=args.output_dir,
        sample_limit=args.sample_limit,
        splits=tuple(args.splits),
        seed=args.seed,
        swanlab_mode=args.swanlab_mode,
    )


def main() -> None:
    summary = build_stage_a(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
