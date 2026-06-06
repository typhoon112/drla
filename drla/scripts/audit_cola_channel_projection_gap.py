"""Audit the decoder semantic-projection gap in P2-D channel evaluation.

This is a local-only analysis.  It pairs a decode-and-emit ``latent_matched``
Agent-B channel with the receiver-native cache-only
``latent_matched_cache_only`` channel on the same samples.

Important: if the decode-and-emit input was generated with the historical
``legacy_all_visible`` scorer scope, this audit measures a replay-output /
scorer-leak gap, not valid Agent-B communication.  Use it to diagnose how much
of the old gain came from A replay tokens being visible in final ``generate``.

The audit does not train, does not call SwanLab, and does not use scorer output
as any online Agent-A/B input.  It only reads post-hoc ``sample_scores.jsonl``
files that have already been officially scored.
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

from drla.tracking import require_swanlab_disabled_for_non_training


@dataclass(frozen=True)
class ProjectionGapAuditConfig:
    decode_emit_scores: str
    cache_only_scores: str
    output_dir: str
    decode_emit_channel: str = "latent_matched"
    cache_only_channel: str = "latent_matched_cache_only"
    text_channel: str = "text"
    none_channel: str = "none"
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 20260531
    swanlab_mode: str = "disabled"


def main() -> None:
    summary = audit_projection_gap(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> ProjectionGapAuditConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-emit-scores", required=True)
    parser.add_argument("--cache-only-scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--decode-emit-channel", default=ProjectionGapAuditConfig.decode_emit_channel)
    parser.add_argument("--cache-only-channel", default=ProjectionGapAuditConfig.cache_only_channel)
    parser.add_argument("--text-channel", default=ProjectionGapAuditConfig.text_channel)
    parser.add_argument("--none-channel", default=ProjectionGapAuditConfig.none_channel)
    parser.add_argument("--bootstrap-samples", type=int, default=ProjectionGapAuditConfig.bootstrap_samples)
    parser.add_argument("--bootstrap-seed", type=int, default=ProjectionGapAuditConfig.bootstrap_seed)
    parser.add_argument("--swanlab-mode", default=ProjectionGapAuditConfig.swanlab_mode)
    args = parser.parse_args()
    if args.bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    return ProjectionGapAuditConfig(
        decode_emit_scores=args.decode_emit_scores,
        cache_only_scores=args.cache_only_scores,
        output_dir=args.output_dir,
        decode_emit_channel=args.decode_emit_channel,
        cache_only_channel=args.cache_only_channel,
        text_channel=args.text_channel,
        none_channel=args.none_channel,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        swanlab_mode=args.swanlab_mode,
    )


def audit_projection_gap(config: ProjectionGapAuditConfig) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        config.swanlab_mode,
        script_kind="P2-D decoder semantic-projection gap audit",
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    decode_rows = read_jsonl(Path(config.decode_emit_scores))
    cache_rows = read_jsonl(Path(config.cache_only_scores))
    if not decode_rows:
        raise ValueError(f"no decode-and-emit rows: {config.decode_emit_scores}")
    if not cache_rows:
        raise ValueError(f"no cache-only rows: {config.cache_only_scores}")

    decode_by_key = index_rows(decode_rows)
    cache_by_key = index_rows(cache_rows)
    pair_rows = build_pair_rows(decode_by_key, cache_by_key, config)
    if not pair_rows:
        raise ValueError("no paired decode/cache rows found")

    overall_rows = [summarize_projection_bucket("all", pair_rows, config)]
    task_rows = [
        summarize_projection_bucket(task, rows, config)
        for task, rows in sorted(group_by(pair_rows, "task").items())
    ]
    paired_rows = build_paired_comparison_rows(pair_rows, config)

    pairs_path = output_dir / "projection_gap_pairs.jsonl"
    overall_path = output_dir / "projection_gap_summary.csv"
    task_path = output_dir / "task_projection_gap_summary.csv"
    paired_path = output_dir / "paired_projection_gap_comparison.csv"
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(pairs_path, pair_rows)
    write_csv(overall_path, overall_rows)
    write_csv(task_path, task_rows)
    write_csv(paired_path, paired_rows)
    write_metrics(metrics_path, overall_rows, task_rows, paired_rows)

    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "num_decode_emit_rows": len(decode_rows),
        "num_cache_only_rows": len(cache_rows),
        "num_paired_samples": len(pair_rows),
        "summary": overall_rows[0],
        "task_summary": task_rows,
        "paired_comparisons": paired_rows,
        "artifacts": {
            "summary_json": str(summary_path),
            "metrics_jsonl": str(metrics_path),
            "projection_gap_pairs_jsonl": str(pairs_path),
            "projection_gap_summary_csv": str(overall_path),
            "task_projection_gap_summary_csv": str(task_path),
            "paired_projection_gap_comparison_csv": str(paired_path),
        },
        "interpretation": (
            "Measures the gap between decode-and-emit and cache-only matched "
            "latent channels. If the decode-and-emit run used legacy all-visible "
            "scoring, a large positive gap means A replay tokens were directly "
            "visible to the scorer and the result should be treated as a "
            "decodability/replay-output diagnostic, not as Agent-B communication."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_pair_rows(
    decode_by_key: dict[tuple[str, str], dict[str, dict[str, Any]]],
    cache_by_key: dict[tuple[str, str], dict[str, dict[str, Any]]],
    config: ProjectionGapAuditConfig,
) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(set(decode_by_key) & set(cache_by_key)):
        decode_channels = decode_by_key[key]
        cache_channels = cache_by_key[key]
        decode = decode_channels.get(config.decode_emit_channel)
        cache = cache_channels.get(config.cache_only_channel)
        if decode is None or cache is None:
            continue
        text = decode_channels.get(config.text_channel) or cache_channels.get(config.text_channel)
        none = decode_channels.get(config.none_channel) or cache_channels.get(config.none_channel)
        cache_corrupt = {
            channel: row
            for channel, row in cache_channels.items()
            if channel.startswith("latent_") and channel != config.cache_only_channel
        }
        rows.append(make_pair_row(key, decode, cache, text, none, cache_corrupt))
    return rows


def make_pair_row(
    key: tuple[str, str],
    decode: dict[str, Any],
    cache: dict[str, Any],
    text: dict[str, Any] | None,
    none: dict[str, Any] | None,
    cache_corrupt: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    decode_score = row_score(decode)
    cache_score = row_score(cache)
    text_score = None if text is None else row_score(text)
    none_score = None if none is None else row_score(none)
    decode_correct = row_correct(decode)
    cache_correct = row_correct(cache)
    text_correct = None if text is None else row_correct(text)
    none_correct = None if none is None else row_correct(none)
    corrupt_scores = [row_score(row) for row in cache_corrupt.values()]
    corrupt_corrects = [row_correct(row) for row in cache_corrupt.values()]
    return {
        "sample_key": key[0],
        "message_index": key[1],
        "task": str(decode.get("task", cache.get("task", ""))),
        "handoff_depth": decode.get("handoff_depth", cache.get("handoff_depth", "")),
        "decode_emit_score": decode_score,
        "cache_only_score": cache_score,
        "projection_score_gain": decode_score - cache_score,
        "decode_emit_correct": decode_correct,
        "cache_only_correct": cache_correct,
        "projection_accuracy_gain": decode_correct - cache_correct,
        "decode_emit_prediction": decode.get("official_prediction", ""),
        "cache_only_prediction": cache.get("official_prediction", ""),
        "same_prediction": int(str(decode.get("official_prediction", "")) == str(cache.get("official_prediction", ""))),
        "text_score": text_score,
        "none_score": none_score,
        "text_correct": text_correct,
        "none_correct": none_correct,
        "decode_emit_score_minus_text": none_if_missing(text_score, decode_score),
        "cache_only_score_minus_text": none_if_missing(text_score, cache_score),
        "decode_emit_score_minus_none": none_if_missing(none_score, decode_score),
        "cache_only_score_minus_none": none_if_missing(none_score, cache_score),
        "decode_emit_accuracy_minus_text": none_if_missing(text_correct, decode_correct),
        "cache_only_accuracy_minus_text": none_if_missing(text_correct, cache_correct),
        "decode_emit_accuracy_minus_none": none_if_missing(none_correct, decode_correct),
        "cache_only_accuracy_minus_none": none_if_missing(none_correct, cache_correct),
        "cache_corrupt_mean_score": mean(corrupt_scores),
        "cache_corrupt_mean_accuracy": mean(corrupt_corrects),
        "cache_only_score_minus_corrupt_mean": cache_score - mean(corrupt_scores),
        "cache_only_accuracy_minus_corrupt_mean": cache_correct - mean(corrupt_corrects),
        "decode_emit_generated_chars": len(str(decode.get("generate", ""))),
        "cache_only_generated_chars": len(str(cache.get("generate", ""))),
        "decode_emit_replay_blocks_consumed": number_or_zero(decode.get("replay_blocks_consumed", 0)),
        "cache_only_replay_blocks_consumed": number_or_zero(cache.get("replay_blocks_consumed", 0)),
        "cache_only_replay_blocks_decoded_to_text": number_or_zero(cache.get("replay_blocks_decoded_to_text", 0)),
        "cache_only_replay_decode_mode": cache.get("replay_decode_mode", ""),
        "decode_emit_latent_elements_received": number_or_zero(decode.get("latent_elements_received", 0)),
        "cache_only_latent_elements_received": number_or_zero(cache.get("latent_elements_received", 0)),
    }


def summarize_projection_bucket(label: str, rows: list[dict[str, Any]], config: ProjectionGapAuditConfig) -> dict[str, Any]:
    score_gains = [float(row["projection_score_gain"]) for row in rows]
    acc_gains = [float(row["projection_accuracy_gain"]) for row in rows]
    score_ci = bootstrap_mean_ci(
        score_gains,
        samples=config.bootstrap_samples,
        seed=config.bootstrap_seed + stable_offset(label),
    )
    acc_ci = bootstrap_mean_ci(
        acc_gains,
        samples=config.bootstrap_samples,
        seed=config.bootstrap_seed + stable_offset(label) + 100_000,
    )
    return {
        "group": label,
        "count": len(rows),
        "decode_emit_accuracy": mean(row["decode_emit_correct"] for row in rows),
        "cache_only_accuracy": mean(row["cache_only_correct"] for row in rows),
        "projection_accuracy_gain": mean(acc_gains),
        "projection_accuracy_gain_ci95_low": acc_ci[0],
        "projection_accuracy_gain_ci95_high": acc_ci[1],
        "decode_emit_mean_score": mean(row["decode_emit_score"] for row in rows),
        "cache_only_mean_score": mean(row["cache_only_score"] for row in rows),
        "projection_score_gain": mean(score_gains),
        "projection_score_gain_ci95_low": score_ci[0],
        "projection_score_gain_ci95_high": score_ci[1],
        "decode_emit_score_wins": sum(1 for value in score_gains if value > 0),
        "decode_emit_score_losses": sum(1 for value in score_gains if value < 0),
        "decode_emit_score_ties": sum(1 for value in score_gains if value == 0),
        "decode_emit_accuracy_wins": sum(1 for value in acc_gains if value > 0),
        "decode_emit_accuracy_losses": sum(1 for value in acc_gains if value < 0),
        "decode_emit_accuracy_ties": sum(1 for value in acc_gains if value == 0),
        "same_prediction_rate": mean(row["same_prediction"] for row in rows),
        "decode_emit_score_minus_text": mean_present(row["decode_emit_score_minus_text"] for row in rows),
        "cache_only_score_minus_text": mean_present(row["cache_only_score_minus_text"] for row in rows),
        "decode_emit_score_minus_none": mean_present(row["decode_emit_score_minus_none"] for row in rows),
        "cache_only_score_minus_none": mean_present(row["cache_only_score_minus_none"] for row in rows),
        "cache_only_score_minus_corrupt_mean": mean(row["cache_only_score_minus_corrupt_mean"] for row in rows),
        "decode_emit_avg_generated_chars": mean(row["decode_emit_generated_chars"] for row in rows),
        "cache_only_avg_generated_chars": mean(row["cache_only_generated_chars"] for row in rows),
        "cache_only_avg_replay_blocks_decoded_to_text": mean(
            row["cache_only_replay_blocks_decoded_to_text"] for row in rows
        ),
    }


def build_paired_comparison_rows(rows: list[dict[str, Any]], config: ProjectionGapAuditConfig) -> list[dict[str, Any]]:
    specs = [
        ("decode_emit", "cache_only", "decode_emit_score", "cache_only_score", "decode_emit_correct", "cache_only_correct"),
        ("decode_emit", "text", "decode_emit_score", "text_score", "decode_emit_correct", "text_correct"),
        ("cache_only", "text", "cache_only_score", "text_score", "cache_only_correct", "text_correct"),
        ("decode_emit", "none", "decode_emit_score", "none_score", "decode_emit_correct", "none_correct"),
        ("cache_only", "none", "cache_only_score", "none_score", "cache_only_correct", "none_correct"),
    ]
    output = []
    for base_name, compare_name, base_score_key, compare_score_key, base_acc_key, compare_acc_key in specs:
        pairs = [
            row
            for row in rows
            if row.get(base_score_key) is not None and row.get(compare_score_key) is not None
        ]
        if not pairs:
            continue
        output.append(
            summarize_named_pairs(
                base_name=base_name,
                compare_name=compare_name,
                rows=pairs,
                base_score_key=base_score_key,
                compare_score_key=compare_score_key,
                base_acc_key=base_acc_key,
                compare_acc_key=compare_acc_key,
                config=config,
            ),
        )
    return output


def summarize_named_pairs(
    *,
    base_name: str,
    compare_name: str,
    rows: list[dict[str, Any]],
    base_score_key: str,
    compare_score_key: str,
    base_acc_key: str,
    compare_acc_key: str,
    config: ProjectionGapAuditConfig,
) -> dict[str, Any]:
    score_deltas = [float(row[base_score_key]) - float(row[compare_score_key]) for row in rows]
    acc_deltas = [float(row[base_acc_key]) - float(row[compare_acc_key]) for row in rows]
    score_ci = bootstrap_mean_ci(
        score_deltas,
        samples=config.bootstrap_samples,
        seed=config.bootstrap_seed + stable_offset(base_name + compare_name),
    )
    acc_ci = bootstrap_mean_ci(
        acc_deltas,
        samples=config.bootstrap_samples,
        seed=config.bootstrap_seed + stable_offset(base_name + compare_name) + 100_000,
    )
    return {
        "baseline": base_name,
        "compare": compare_name,
        "count": len(rows),
        "baseline_accuracy": mean(row[base_acc_key] for row in rows),
        "compare_accuracy": mean(row[compare_acc_key] for row in rows),
        "accuracy_delta_baseline_minus_compare": mean(acc_deltas),
        "accuracy_delta_ci95_low": acc_ci[0],
        "accuracy_delta_ci95_high": acc_ci[1],
        "baseline_mean_score": mean(row[base_score_key] for row in rows),
        "compare_mean_score": mean(row[compare_score_key] for row in rows),
        "score_delta_baseline_minus_compare": mean(score_deltas),
        "score_delta_ci95_low": score_ci[0],
        "score_delta_ci95_high": score_ci[1],
        "baseline_score_wins": sum(1 for value in score_deltas if value > 0),
        "baseline_score_losses": sum(1 for value in score_deltas if value < 0),
        "baseline_score_ties": sum(1 for value in score_deltas if value == 0),
        "baseline_accuracy_wins": sum(1 for value in acc_deltas if value > 0),
        "baseline_accuracy_losses": sum(1 for value in acc_deltas if value < 0),
        "baseline_accuracy_ties": sum(1 for value in acc_deltas if value == 0),
    }


def index_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    indexed: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row.get("sample_key", "")), str(row.get("message_index", "")))
        channel = str(row.get("channel", ""))
        indexed[key][channel] = row
    return dict(indexed)


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    return dict(grouped)


def row_score(row: dict[str, Any]) -> float:
    return float(row.get("official_score", 0.0))


def row_correct(row: dict[str, Any]) -> int:
    return int(row.get("official_correct", 0))


def none_if_missing(compare: float | int | None, base: float | int) -> float | None:
    if compare is None:
        return None
    return float(base) - float(compare)


def number_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def mean(values: Any) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)


def mean_present(values: Any) -> float | None:
    materialized = [float(value) for value in values if value is not None]
    if not materialized:
        return None
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


def stable_offset(label: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(label))


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


def write_metrics(
    path: Path,
    overall_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
) -> None:
    created_at = int(time.time())
    with path.open("w", encoding="utf-8") as handle:
        for row in overall_rows:
            handle.write(json.dumps({"created_at": created_at, "kind": "overall", "metrics": row}) + "\n")
        for row in task_rows:
            handle.write(json.dumps({"created_at": created_at, "kind": "task", "metrics": row}) + "\n")
        for row in paired_rows:
            handle.write(json.dumps({"created_at": created_at, "kind": "paired", "metrics": row}) + "\n")


if __name__ == "__main__":
    main()
