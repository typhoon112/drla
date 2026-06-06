"""Aggregate corrected P2-D Agent-B channel-equivalent evaluation outputs.

This is a local-only post-hoc scorer. It reads ``generations.jsonl`` produced
by ``run_cola_agent_b_channel_eval.py``, scores only the final Agent-B outputs
with the official Cola scorer rules, and reports paired comparisons between
the matched latent channel and text/none/corrupted channels.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from drla.scripts.audit_cola_sequential_latent_mas import (
    load_official_scorer,
    score_text_with_official_rules,
)
from drla.tracking import require_swanlab_disabled_for_non_training


@dataclass(frozen=True)
class ChannelEvalAggregateConfig:
    eval_root: str
    output_dir: str = ""
    acc_calc_script: str = "/data1/luyifei/Cola-DLM/code/scripts/acc_calc.py"
    baseline_channel: str = "latent_matched"
    text_channel: str = "text"
    none_channel: str = "none"
    text_competitive_score_tolerance: float = 0.01
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 20260531
    swanlab_mode: str = "disabled"


def main() -> None:
    summary = aggregate_channel_eval(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> ChannelEvalAggregateConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--acc-calc-script", default=ChannelEvalAggregateConfig.acc_calc_script)
    parser.add_argument("--baseline-channel", default=ChannelEvalAggregateConfig.baseline_channel)
    parser.add_argument("--text-channel", default=ChannelEvalAggregateConfig.text_channel)
    parser.add_argument("--none-channel", default=ChannelEvalAggregateConfig.none_channel)
    parser.add_argument(
        "--text-competitive-score-tolerance",
        type=float,
        default=ChannelEvalAggregateConfig.text_competitive_score_tolerance,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=ChannelEvalAggregateConfig.bootstrap_samples)
    parser.add_argument("--bootstrap-seed", type=int, default=ChannelEvalAggregateConfig.bootstrap_seed)
    parser.add_argument("--swanlab-mode", default=ChannelEvalAggregateConfig.swanlab_mode)
    args = parser.parse_args()
    if args.text_competitive_score_tolerance < 0:
        raise ValueError("text competitive tolerance must be non-negative")
    if args.bootstrap_samples < 0:
        raise ValueError("bootstrap samples must be non-negative")
    return ChannelEvalAggregateConfig(
        eval_root=args.eval_root,
        output_dir=args.output_dir,
        acc_calc_script=args.acc_calc_script,
        baseline_channel=args.baseline_channel,
        text_channel=args.text_channel,
        none_channel=args.none_channel,
        text_competitive_score_tolerance=args.text_competitive_score_tolerance,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        swanlab_mode=args.swanlab_mode,
    )


def aggregate_channel_eval(config: ChannelEvalAggregateConfig) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        config.swanlab_mode,
        script_kind="P2-D Agent-B channel-equivalent aggregate",
    )
    eval_root = Path(config.eval_root)
    generations_path = eval_root / "generations.jsonl"
    if not generations_path.exists():
        raise FileNotFoundError(generations_path)
    output_dir = Path(config.output_dir) if config.output_dir else eval_root / "channel_eval_aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_scores_path = output_dir / "sample_scores.jsonl"
    channel_summary_path = output_dir / "channel_score_summary.csv"
    task_channel_summary_path = output_dir / "task_channel_score_summary.csv"
    paired_summary_path = output_dir / "paired_channel_comparison.csv"
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"

    scorer = load_official_scorer(Path(config.acc_calc_script))
    generations = read_jsonl(generations_path)
    if not generations:
        raise ValueError(f"no generations in {generations_path}")
    scored_rows = [score_generation(row, scorer) for row in generations]
    write_jsonl(sample_scores_path, scored_rows)

    channel_rows = aggregate_rows(scored_rows, group_keys=["channel"])
    task_channel_rows = aggregate_rows(scored_rows, group_keys=["task", "channel"])
    paired_rows = build_paired_rows(scored_rows, baseline_channel=config.baseline_channel, config=config)
    write_csv(channel_summary_path, channel_rows)
    write_csv(task_channel_summary_path, task_channel_rows)
    write_csv(paired_summary_path, paired_rows)

    decision = build_decision_summary(channel_rows, config)
    with metrics_path.open("w", encoding="utf-8") as handle:
        for row in channel_rows:
            handle.write(
                json.dumps(
                    {"created_at": int(time.time()), "kind": "channel", "metrics": row},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
            )
        for row in paired_rows:
            handle.write(
                json.dumps(
                    {"created_at": int(time.time()), "kind": "paired", "metrics": row},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
            )
        handle.write(
            json.dumps(
                {"created_at": int(time.time()), "kind": "decision_rule", "metrics": decision},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
        )

    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "num_generations": len(scored_rows),
        "channels": sorted({str(row["channel"]) for row in scored_rows}),
        "channel_score_summary": channel_rows,
        "paired_channel_comparison": paired_rows,
        "decision_rule": decision,
        "artifacts": {
            "summary_json": str(summary_path),
            "metrics_jsonl": str(metrics_path),
            "sample_scores_jsonl": str(sample_scores_path),
            "channel_score_summary_csv": str(channel_summary_path),
            "task_channel_score_summary_csv": str(task_channel_summary_path),
            "paired_channel_comparison_csv": str(paired_summary_path),
        },
        "interpretation": (
            "Post-hoc official scoring for corrected P2-D Agent-B channels. "
            "Scorer/gold labels are used only after final Agent-B generation."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def score_generation(row: dict[str, Any], scorer: Any) -> dict[str, Any]:
    score = score_text_with_official_rules(
        task=str(row["task"]),
        text=str(row.get("generate", "")),
        ground_truth=row.get("ground_truth", ""),
        choices=row.get("choices", []),
        scorer=scorer,
    )
    return {
        **row,
        "official_processed_generation": score["processed_generation"],
        "official_prediction": score["prediction"],
        "official_target": score["target"],
        "official_score": float(score["score"]),
        "official_correct": int(bool(score["correct"])),
    }


def aggregate_rows(rows: list[dict[str, Any]], *, group_keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key, "") for key in group_keys)].append(row)
    output = []
    for key, bucket in sorted(buckets.items(), key=lambda item: tuple(str(part) for part in item[0])):
        item = {group_key: key[index] for index, group_key in enumerate(group_keys)}
        item.update(summarize_bucket(bucket))
        output.append(item)
    return output


def summarize_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "count": count,
        "accuracy": mean(row.get("official_correct", 0) for row in rows),
        "mean_score": mean(row.get("official_score", 0.0) for row in rows),
        "nonempty_rate": mean(1 if str(row.get("generate", "")).strip() else 0 for row in rows),
        "avg_total_blocks": mean(row.get("total_blocks", 0) for row in rows),
        "avg_replay_blocks": mean(row.get("replay_blocks_consumed", 0) for row in rows),
        "avg_receiver_blocks": mean(row.get("receiver_blocks_generated", 0) for row in rows),
        "avg_latent_elements_received": mean(row.get("latent_elements_received", 0) for row in rows),
        "avg_text_message_tokens_received": mean(row.get("text_message_tokens_received", 0) for row in rows),
        "avg_text_message_chars_received": mean(row.get("text_message_chars_received", 0) for row in rows),
        "avg_generated_chars": mean(len(str(row.get("generate", ""))) for row in rows),
        "stop_token_rate": mean(1 if row.get("stop_reason") == "stop_token" else 0 for row in rows),
    }


def build_paired_rows(
    rows: list[dict[str, Any]],
    *,
    baseline_channel: str,
    config: ChannelEvalAggregateConfig,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        pair_key = (str(row.get("sample_key", "")), str(row.get("message_index", "")))
        grouped[pair_key][str(row["channel"])] = row

    pair_buckets: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for channels in grouped.values():
        baseline = channels.get(baseline_channel)
        if baseline is None:
            continue
        for channel, row in channels.items():
            if channel == baseline_channel:
                continue
            pair_buckets[channel].append((baseline, row))

    output = []
    for channel, pairs in sorted(pair_buckets.items()):
        output.append(summarize_pairs(baseline_channel, channel, pairs, config))
    return output


def summarize_pairs(
    baseline_channel: str,
    compare_channel: str,
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    config: ChannelEvalAggregateConfig,
) -> dict[str, Any]:
    score_deltas = [float(base["official_score"]) - float(other["official_score"]) for base, other in pairs]
    acc_deltas = [int(base["official_correct"]) - int(other["official_correct"]) for base, other in pairs]
    score_ci = bootstrap_mean_ci(
        score_deltas,
        samples=config.bootstrap_samples,
        seed=config.bootstrap_seed + stable_channel_offset(compare_channel),
    )
    acc_ci = bootstrap_mean_ci(
        acc_deltas,
        samples=config.bootstrap_samples,
        seed=config.bootstrap_seed + stable_channel_offset(compare_channel) + 100_000,
    )
    return {
        "baseline_channel": baseline_channel,
        "compare_channel": compare_channel,
        "count": len(pairs),
        "baseline_accuracy": mean(base["official_correct"] for base, _ in pairs),
        "compare_accuracy": mean(other["official_correct"] for _, other in pairs),
        "accuracy_delta_baseline_minus_compare": mean(acc_deltas),
        "baseline_mean_score": mean(base["official_score"] for base, _ in pairs),
        "compare_mean_score": mean(other["official_score"] for _, other in pairs),
        "score_delta_baseline_minus_compare": mean(score_deltas),
        "score_delta_ci95_low": score_ci[0],
        "score_delta_ci95_high": score_ci[1],
        "baseline_score_wins": sum(1 for value in score_deltas if value > 0),
        "baseline_score_losses": sum(1 for value in score_deltas if value < 0),
        "baseline_score_ties": sum(1 for value in score_deltas if value == 0),
        "baseline_accuracy_wins": sum(1 for value in acc_deltas if value > 0),
        "baseline_accuracy_losses": sum(1 for value in acc_deltas if value < 0),
        "baseline_accuracy_ties": sum(1 for value in acc_deltas if value == 0),
        "accuracy_delta_ci95_low": acc_ci[0],
        "accuracy_delta_ci95_high": acc_ci[1],
    }


def build_decision_summary(
    channel_rows: list[dict[str, Any]],
    config: ChannelEvalAggregateConfig,
) -> dict[str, Any]:
    by_channel = {str(row["channel"]): row for row in channel_rows}
    baseline = by_channel.get(config.baseline_channel)
    if baseline is None:
        return {"status": "missing_baseline", "baseline_channel": config.baseline_channel}

    corrupt_rows = [
        row
        for channel, row in by_channel.items()
        if channel.startswith("latent_") and channel != config.baseline_channel
    ]
    none = by_channel.get(config.none_channel)
    text = by_channel.get(config.text_channel)
    baseline_score = float(baseline["mean_score"])
    baseline_acc = float(baseline["accuracy"])
    text_score = None if text is None else float(text["mean_score"])
    return {
        "baseline_channel": config.baseline_channel,
        "matched_mean_score": baseline_score,
        "matched_accuracy": baseline_acc,
        "matched_beats_all_corrupt_controls_by_score": bool(
            corrupt_rows and all(baseline_score > float(row["mean_score"]) for row in corrupt_rows)
        ),
        "matched_beats_none_by_score": None if none is None else baseline_score > float(none["mean_score"]),
        "matched_beats_none_by_accuracy": None if none is None else baseline_acc > float(none["accuracy"]),
        "matched_competitive_with_text_by_score": (
            None
            if text_score is None
            else baseline_score + config.text_competitive_score_tolerance >= text_score
        ),
        "matched_beats_text_by_score": None if text_score is None else baseline_score > text_score,
        "matched_beats_text_by_accuracy": None if text is None else baseline_acc > float(text["accuracy"]),
        "text_competitive_score_tolerance": config.text_competitive_score_tolerance,
    }


def mean(values: Any) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)


def bootstrap_mean_ci(values: list[float], *, samples: int, seed: int) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if samples <= 0:
        point = mean(values)
        return (point, point)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(samples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    low_idx = max(0, min(len(means) - 1, int(0.025 * samples)))
    high_idx = max(0, min(len(means) - 1, int(0.975 * samples) - 1))
    return (means[low_idx], means[high_idx])


def stable_channel_offset(channel: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(channel))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
