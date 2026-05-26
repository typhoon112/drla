"""Sample-level diagnostics for Cola readiness / halt policies."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from drla.scripts.eval_cola_adaptive_halt import (
    choose_row,
    group_rows,
    normalize_inputs,
    predict_readiness,
    resolve_split_indices,
)
from drla.scripts.train_cola_readiness_model import (
    ReadinessModel,
    ReadinessTrainConfig,
    build_tensors,
    load_training_rows,
    resolve_device,
)


@dataclass(frozen=True)
class HaltDecisionAnalysisConfig:
    eval_summary_path: str | None = None
    checkpoint_path: str | None = None
    output_dir: str = "/data1/luyifei/drla/outputs/cola_halt_decision_analysis/debug"
    eval_labels_dir: str | None = None
    eval_tasks: str | None = None
    split: str = "test"
    adaptive_threshold: float | None = None
    early_or_stability_threshold: float | None = None
    guarded_threshold: float | None = None
    extra_early_or_thresholds: str = ""
    batch_size: int = 512
    device: str = "auto"
    max_loss_examples: int = 20


def analyze_halt_decisions(config: HaltDecisionAnalysisConfig) -> dict[str, Any]:
    config = hydrate_from_eval_summary(config)
    if not config.checkpoint_path:
        raise ValueError("checkpoint_path is required, either directly or via --eval-summary-path")

    checkpoint = torch.load(config.checkpoint_path, map_location="cpu")
    train_config = ReadinessTrainConfig(**checkpoint["config"])
    feature_fields = checkpoint.get("feature_fields") or checkpoint.get("metadata", {}).get("feature_fields")
    eval_config = replace(
        train_config,
        labels_dir=config.eval_labels_dir or train_config.labels_dir,
        tasks=config.eval_tasks or train_config.tasks,
    )

    rows = load_training_rows(eval_config)
    tensors, metadata = build_tensors(rows, eval_config, feature_fields=feature_fields)
    eval_indices = resolve_split_indices(metadata["sample_keys"], eval_config, config.split)

    latent, features = normalize_inputs(tensors, checkpoint["norm_stats"])
    task_onehot = tensors["task_onehot"]
    device = resolve_device(config.device)
    model = ReadinessModel(
        latent_dim=latent.shape[1],
        feature_dim=features.shape[1],
        task_dim=task_onehot.shape[1],
        hidden_dim=train_config.hidden_dim,
        dropout=train_config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    probs = predict_readiness(
        model=model,
        latent=latent[eval_indices],
        features=features[eval_indices],
        task_onehot=task_onehot[eval_indices],
        device=device,
        batch_size=config.batch_size,
    )
    eval_rows = materialize_diagnostic_rows(rows, metadata["sample_keys"], eval_indices, probs)
    grouped = group_rows(eval_rows)
    policy_specs = build_policy_specs(config)
    decision_records = [
        build_decision_record(sample_key, sample_rows, policy_specs)
        for sample_key, sample_rows in sorted(grouped.items())
    ]

    policy_summary = summarize_policies(decision_records, policy_specs)
    bin_rows = build_readiness_bins(grouped)
    loss_examples = collect_loss_examples(decision_records, policy_specs, config.max_loss_examples)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = output_dir / "halt_decisions.jsonl"
    policy_csv_path = output_dir / "policy_comparison.csv"
    bins_csv_path = output_dir / "readiness_bins.csv"
    summary_path = output_dir / "summary.json"

    with decisions_path.open("w", encoding="utf-8") as f:
        for record in decision_records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    write_policy_csv(policy_csv_path, policy_summary)
    write_bins_csv(bins_csv_path, bin_rows)

    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "eval_tasks": eval_config.tasks,
        "split": config.split,
        "num_samples": len(decision_records),
        "policy_summary": policy_summary,
        "readiness_bins_csv": str(bins_csv_path),
        "policy_comparison_csv": str(policy_csv_path),
        "halt_decisions_jsonl": str(decisions_path),
        "loss_examples": loss_examples,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def hydrate_from_eval_summary(config: HaltDecisionAnalysisConfig) -> HaltDecisionAnalysisConfig:
    if not config.eval_summary_path:
        return config
    summary = json.loads(Path(config.eval_summary_path).read_text(encoding="utf-8"))
    summary_config = summary.get("config", {})

    def keep_or_summary(value: Any, key: str) -> Any:
        return value if value is not None else summary_config.get(key)

    return replace(
        config,
        checkpoint_path=keep_or_summary(config.checkpoint_path, "checkpoint_path") or summary.get("checkpoint_path"),
        eval_labels_dir=keep_or_summary(config.eval_labels_dir, "eval_labels_dir"),
        eval_tasks=keep_or_summary(config.eval_tasks, "eval_tasks") or summary.get("eval_tasks"),
        split=config.split if config.split != HaltDecisionAnalysisConfig.split else summary.get("split", config.split),
        adaptive_threshold=(
            config.adaptive_threshold
            if config.adaptive_threshold is not None
            else summary.get("calibrated_global_threshold", {}).get("threshold")
        ),
        early_or_stability_threshold=(
            config.early_or_stability_threshold
            if config.early_or_stability_threshold is not None
            else summary.get("calibrated_early_or_stability_threshold", {}).get("threshold")
        ),
        guarded_threshold=(
            config.guarded_threshold
            if config.guarded_threshold is not None
            else summary.get("calibrated_guarded_global_threshold", {}).get("threshold")
        ),
    )


def materialize_diagnostic_rows(
    rows: list[dict[str, Any]],
    sample_keys: list[str],
    indices: list[int],
    probs: torch.Tensor,
) -> list[dict[str, Any]]:
    eval_rows: list[dict[str, Any]] = []
    for local_idx, row_idx in enumerate(indices):
        row = rows[row_idx]
        eval_rows.append(
            {
                "task": row["task"],
                "sample_id": row["sample_id"],
                "sample_key": sample_keys[row_idx],
                "block_index": int(row["block_index"]),
                "block_number": int(row["block_number"]),
                "readiness_prob": float(probs[local_idx].item()),
                "official_correct": bool(row["official_correct"]),
                "official_score": row.get("official_score"),
                "oracle_ready": bool(row["oracle_ready"]),
                "is_at_or_after_oracle_frontier": bool(row["is_at_or_after_oracle_frontier"]),
                "earliest_ready_block_index": row["earliest_ready_block_index"],
                "future_gain_correct": int(row.get("future_gain_correct") or 0),
                "contains_eos": bool(row.get("contains_eos")),
                "contains_im_end": bool(row.get("contains_im_end")),
                "contains_stop": bool(row.get("contains_stop")),
                "scored_prediction": row.get("scored_prediction"),
                "scored_target": row.get("scored_target"),
                "official_processed_generation": row.get("official_processed_generation"),
                "token_entropy_mean": row.get("token_entropy_mean"),
                "token_top_prob_mean": row.get("token_top_prob_mean"),
                "same_text_streak": row.get("same_text_streak"),
            }
        )
    return eval_rows


def build_policy_specs(config: HaltDecisionAnalysisConfig) -> list[dict[str, Any]]:
    specs = [
        {"name": "fixed_b1", "policy": "fixed", "fixed_block_number": 1, "threshold": None},
        {"name": "final", "policy": "final", "fixed_block_number": None, "threshold": None},
        {
            "name": "prediction_stability",
            "policy": "prediction_stability",
            "fixed_block_number": None,
            "threshold": None,
        },
    ]
    if config.adaptive_threshold is not None:
        specs.append(
            {
                "name": "calibrated_adaptive",
                "policy": "adaptive",
                "fixed_block_number": None,
                "threshold": float(config.adaptive_threshold),
            }
        )
    if config.early_or_stability_threshold is not None:
        specs.append(
            {
                "name": "calibrated_early_or_stability",
                "policy": "adaptive_or_prediction_stability",
                "fixed_block_number": None,
                "threshold": float(config.early_or_stability_threshold),
            }
        )
    if config.guarded_threshold is not None:
        specs.append(
            {
                "name": "calibrated_stability_guarded",
                "policy": "stability_guarded_adaptive",
                "fixed_block_number": None,
                "threshold": float(config.guarded_threshold),
            }
        )
    for threshold in parse_float_list(config.extra_early_or_thresholds):
        specs.append(
            {
                "name": f"early_or_stability_t{threshold:g}".replace(".", "p"),
                "policy": "adaptive_or_prediction_stability",
                "fixed_block_number": None,
                "threshold": threshold,
            }
        )
    return specs


def build_decision_record(
    sample_key: str,
    rows: list[dict[str, Any]],
    policy_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    stability_row = choose_row(rows, policy="prediction_stability", fixed_block_number=None, threshold=None)
    final_row = rows[-1]
    oracle_index = final_row.get("earliest_ready_block_index")
    policies = {}
    for spec in policy_specs:
        chosen = choose_row(
            rows,
            policy=spec["policy"],
            fixed_block_number=spec["fixed_block_number"],
            threshold=spec["threshold"],
        )
        policies[spec["name"]] = policy_decision(chosen, final_row, stability_row, spec)

    return {
        "sample_key": sample_key,
        "task": final_row["task"],
        "sample_id": final_row["sample_id"],
        "scored_target": final_row.get("scored_target"),
        "final": row_snapshot(final_row),
        "prediction_stability_block_number": int(stability_row["block_number"]),
        "prediction_stability_correct": bool(stability_row["official_correct"]),
        "earliest_ready_block_number": None if oracle_index is None else int(oracle_index) + 1,
        "policies": policies,
        "blocks": [row_snapshot(row) for row in rows],
    }


def policy_decision(
    chosen: dict[str, Any],
    final_row: dict[str, Any],
    stability_row: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    final_correct = bool(final_row["official_correct"])
    chosen_correct = bool(chosen["official_correct"])
    prefix_of_final = is_strict_prefix(chosen.get("scored_prediction"), final_row.get("scored_prediction"))
    return {
        "policy": spec["policy"],
        "threshold": spec["threshold"],
        "block_number": int(chosen["block_number"]),
        "block_index": int(chosen["block_index"]),
        "readiness_prob": float(chosen["readiness_prob"]),
        "official_correct": chosen_correct,
        "official_score": chosen.get("official_score"),
        "scored_prediction": chosen.get("scored_prediction"),
        "saved_blocks_vs_final": int(final_row["block_number"]) - int(chosen["block_number"]),
        "before_prediction_stability": int(chosen["block_index"]) < int(stability_row["block_index"]),
        "correctness_delta_vs_final": int(chosen_correct) - int(final_correct),
        "lost_final_correct": final_correct and not chosen_correct,
        "gained_over_final": chosen_correct and not final_correct,
        "chosen_is_strict_prefix_of_final": prefix_of_final,
        "prefix_final_correct_loss": final_correct and not chosen_correct and prefix_of_final,
    }


def row_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "block_number": int(row["block_number"]),
        "block_index": int(row["block_index"]),
        "readiness_prob": float(row["readiness_prob"]),
        "official_correct": bool(row["official_correct"]),
        "official_score": row.get("official_score"),
        "oracle_ready": bool(row.get("oracle_ready")),
        "is_at_or_after_oracle_frontier": bool(row.get("is_at_or_after_oracle_frontier")),
        "future_gain_correct": int(row.get("future_gain_correct") or 0),
        "scored_prediction": row.get("scored_prediction"),
        "contains_stop": bool(row.get("contains_stop")),
        "token_entropy_mean": row.get("token_entropy_mean"),
        "token_top_prob_mean": row.get("token_top_prob_mean"),
        "same_text_streak": row.get("same_text_streak"),
    }


def summarize_policies(
    decision_records: list[dict[str, Any]],
    policy_specs: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    n = max(len(decision_records), 1)
    for spec in policy_specs:
        name = spec["name"]
        decisions = [record["policies"][name] for record in decision_records]
        correct = sum(int(item["official_correct"]) for item in decisions)
        total_blocks = sum(int(item["block_number"]) for item in decisions)
        saved_blocks = sum(int(item["saved_blocks_vs_final"]) for item in decisions)
        losses = sum(int(item["lost_final_correct"]) for item in decisions)
        gains = sum(int(item["gained_over_final"]) for item in decisions)
        before_stability = sum(int(item["before_prediction_stability"]) for item in decisions)
        pre_stability_losses = sum(
            int(item["before_prediction_stability"] and item["lost_final_correct"])
            for item in decisions
        )
        prefix_losses = sum(int(item["prefix_final_correct_loss"]) for item in decisions)
        pre_stability_prefix_losses = sum(
            int(item["before_prediction_stability"] and item["prefix_final_correct_loss"])
            for item in decisions
        )
        summary[name] = {
            "threshold": float(spec["threshold"]) if spec["threshold"] is not None else None,
            "accuracy": correct / n,
            "avg_blocks": total_blocks / n,
            "avg_saved_blocks_vs_final": saved_blocks / n,
            "block_saving_vs_final": saved_blocks / max(4 * n, 1),
            "lost_final_correct_count": float(losses),
            "gained_over_final_count": float(gains),
            "before_prediction_stability_rate": before_stability / n,
            "pre_stability_loss_count": float(pre_stability_losses),
            "prefix_final_correct_loss_count": float(prefix_losses),
            "pre_stability_prefix_loss_count": float(pre_stability_prefix_losses),
        }
    return summary


def build_readiness_bins(grouped: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int], dict[str, float]] = {}
    for rows in grouped.values():
        stability_row = choose_row(rows, policy="prediction_stability", fixed_block_number=None, threshold=None)
        final_correct = bool(rows[-1]["official_correct"])
        for row in rows:
            bin_idx = min(int(float(row["readiness_prob"]) * 10), 9)
            for scope in scopes_for_row(row, stability_row):
                bucket = buckets.setdefault(
                    (scope, bin_idx),
                    {
                        "count": 0.0,
                        "current_correct": 0.0,
                        "final_correct": 0.0,
                        "oracle_ready": 0.0,
                        "future_gain_correct": 0.0,
                        "readiness_prob_sum": 0.0,
                    },
                )
                bucket["count"] += 1
                bucket["current_correct"] += int(bool(row["official_correct"]))
                bucket["final_correct"] += int(final_correct)
                bucket["oracle_ready"] += int(bool(row["is_at_or_after_oracle_frontier"]))
                bucket["future_gain_correct"] += int(row.get("future_gain_correct") or 0)
                bucket["readiness_prob_sum"] += float(row["readiness_prob"])

    rows_out: list[dict[str, Any]] = []
    for (scope, bin_idx), bucket in sorted(buckets.items()):
        count = max(bucket["count"], 1.0)
        rows_out.append(
            {
                "scope": scope,
                "prob_bin": f"{bin_idx / 10:.1f}-{(bin_idx + 1) / 10:.1f}",
                "count": int(bucket["count"]),
                "mean_readiness_prob": bucket["readiness_prob_sum"] / count,
                "current_correct_rate": bucket["current_correct"] / count,
                "final_correct_rate": bucket["final_correct"] / count,
                "oracle_ready_rate": bucket["oracle_ready"] / count,
                "future_gain_correct_rate": bucket["future_gain_correct"] / count,
            }
        )
    return rows_out


def scopes_for_row(row: dict[str, Any], stability_row: dict[str, Any]) -> list[str]:
    scopes = ["all"]
    if int(row["block_index"]) < int(stability_row["block_index"]):
        scopes.append("pre_prediction_stability")
    else:
        scopes.append("at_or_after_prediction_stability")
    return scopes


def collect_loss_examples(
    decision_records: list[dict[str, Any]],
    policy_specs: list[dict[str, Any]],
    max_examples: int,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for spec in policy_specs:
        name = spec["name"]
        examples = []
        for record in decision_records:
            decision = record["policies"][name]
            if not decision["lost_final_correct"]:
                continue
            examples.append(
                {
                    "sample_key": record["sample_key"],
                    "task": record["task"],
                    "sample_id": record["sample_id"],
                    "target": record.get("scored_target"),
                    "chosen_block_number": decision["block_number"],
                    "chosen_readiness_prob": decision["readiness_prob"],
                    "chosen_prediction": decision.get("scored_prediction"),
                    "final_prediction": record["final"].get("scored_prediction"),
                    "prediction_stability_block_number": record["prediction_stability_block_number"],
                    "before_prediction_stability": decision["before_prediction_stability"],
                }
            )
        examples.sort(key=lambda item: (not item["before_prediction_stability"], item["chosen_block_number"]))
        result[name] = examples[:max_examples]
    return result


def write_policy_csv(path: Path, policy_summary: dict[str, dict[str, float]]) -> None:
    fieldnames = [
        "policy",
        "threshold",
        "accuracy",
        "avg_blocks",
        "avg_saved_blocks_vs_final",
        "block_saving_vs_final",
        "lost_final_correct_count",
        "gained_over_final_count",
        "before_prediction_stability_rate",
        "pre_stability_loss_count",
        "prefix_final_correct_loss_count",
        "pre_stability_prefix_loss_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for policy, metrics in policy_summary.items():
            writer.writerow({"policy": policy, **metrics})


def write_bins_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "scope",
        "prob_bin",
        "count",
        "mean_readiness_prob",
        "current_correct_rate",
        "final_correct_rate",
        "oracle_ready_rate",
        "future_gain_correct_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float_list(value: str) -> list[float]:
    if not value.strip():
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def is_strict_prefix(value: Any, final_value: Any) -> bool:
    value_text = normalize_for_prefix(value)
    final_text = normalize_for_prefix(final_value)
    return bool(value_text and final_text and value_text != final_text and final_text.startswith(value_text))


def normalize_for_prefix(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def parse_args() -> HaltDecisionAnalysisConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-summary-path", default=HaltDecisionAnalysisConfig.eval_summary_path)
    parser.add_argument("--checkpoint-path", default=HaltDecisionAnalysisConfig.checkpoint_path)
    parser.add_argument("--output-dir", default=HaltDecisionAnalysisConfig.output_dir)
    parser.add_argument("--eval-labels-dir", default=HaltDecisionAnalysisConfig.eval_labels_dir)
    parser.add_argument("--eval-tasks", default=HaltDecisionAnalysisConfig.eval_tasks)
    parser.add_argument("--split", default=HaltDecisionAnalysisConfig.split)
    parser.add_argument("--adaptive-threshold", type=float, default=HaltDecisionAnalysisConfig.adaptive_threshold)
    parser.add_argument(
        "--early-or-stability-threshold",
        type=float,
        default=HaltDecisionAnalysisConfig.early_or_stability_threshold,
    )
    parser.add_argument("--guarded-threshold", type=float, default=HaltDecisionAnalysisConfig.guarded_threshold)
    parser.add_argument("--extra-early-or-thresholds", default=HaltDecisionAnalysisConfig.extra_early_or_thresholds)
    parser.add_argument("--batch-size", type=int, default=HaltDecisionAnalysisConfig.batch_size)
    parser.add_argument("--device", default=HaltDecisionAnalysisConfig.device)
    parser.add_argument("--max-loss-examples", type=int, default=HaltDecisionAnalysisConfig.max_loss_examples)
    args = parser.parse_args()
    return HaltDecisionAnalysisConfig(
        eval_summary_path=args.eval_summary_path,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        eval_labels_dir=args.eval_labels_dir,
        eval_tasks=args.eval_tasks,
        split=args.split,
        adaptive_threshold=args.adaptive_threshold,
        early_or_stability_threshold=args.early_or_stability_threshold,
        guarded_threshold=args.guarded_threshold,
        extra_early_or_thresholds=args.extra_early_or_thresholds,
        batch_size=args.batch_size,
        device=args.device,
        max_loss_examples=args.max_loss_examples,
    )


def main() -> None:
    summary = analyze_halt_decisions(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
