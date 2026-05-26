"""Evaluate adaptive halt policies from a trained Cola readiness model."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from drla.scripts.train_cola_readiness_model import (
    ReadinessModel,
    ReadinessTrainConfig,
    build_tensors,
    load_training_rows,
    resolve_device,
    split_indices,
)
from drla.tracking import finish_experiment, init_experiment, log_metrics
from drla.tracking import require_swanlab_disabled_for_non_training


@dataclass(frozen=True)
class AdaptiveHaltEvalConfig:
    checkpoint_path: str = "/data1/luyifei/drla/outputs/cola_readiness_model/official8_1000_b20_t16_seed66_20260524/checkpoints/best_checkpoint.pt"
    output_dir: str = "/data1/luyifei/drla/outputs/cola_adaptive_halt/official8_1000_b20_t16_seed66_20260524"
    eval_labels_dir: str | None = None
    eval_tasks: str | None = None
    calibration_tasks: str | None = None
    split: str = "test"
    calibration_split: str = "valid"
    threshold_start: float = 0.0
    threshold_end: float = 1.0
    threshold_step: float = 0.05
    accuracy_drop_tolerance: float = 0.01
    batch_size: int = 512
    device: str = "auto"
    swanlab_mode: str = "disabled"
    experiment_name: str = "official8-adaptive-halt-eval"


def evaluate_adaptive_halt(config: AdaptiveHaltEvalConfig) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        config.swanlab_mode,
        script_kind="adaptive halt evaluation",
    )
    checkpoint = torch.load(config.checkpoint_path, map_location="cpu")
    train_config = ReadinessTrainConfig(**checkpoint["config"])
    feature_fields = checkpoint.get("feature_fields") or checkpoint.get("metadata", {}).get("feature_fields")
    eval_config = replace(
        train_config,
        labels_dir=config.eval_labels_dir or train_config.labels_dir,
        tasks=config.eval_tasks or train_config.tasks,
    )
    calibration_config = replace(
        train_config,
        labels_dir=config.eval_labels_dir or train_config.labels_dir,
        tasks=config.calibration_tasks or train_config.tasks,
    )
    rows = load_training_rows(eval_config)
    tensors, metadata = build_tensors(rows, eval_config, feature_fields=feature_fields)
    eval_indices = resolve_split_indices(metadata["sample_keys"], eval_config, config.split)
    calibration_rows_source = rows
    calibration_metadata = metadata
    calibration_tensors = tensors
    if calibration_config.tasks != eval_config.tasks or calibration_config.labels_dir != eval_config.labels_dir:
        calibration_rows_source = load_training_rows(calibration_config)
        calibration_tensors, calibration_metadata = build_tensors(
            calibration_rows_source,
            calibration_config,
            feature_fields=feature_fields,
        )
    calibration_indices = resolve_split_indices(
        calibration_metadata["sample_keys"],
        calibration_config,
        config.calibration_split,
    )

    norm_stats = checkpoint["norm_stats"]
    latent, features = normalize_inputs(tensors, norm_stats)
    task_onehot = tensors["task_onehot"]
    calibration_latent, calibration_features = normalize_inputs(calibration_tensors, norm_stats)
    calibration_task_onehot = calibration_tensors["task_onehot"]

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
    calibration_probs = predict_readiness(
        model=model,
        latent=calibration_latent[calibration_indices],
        features=calibration_features[calibration_indices],
        task_onehot=calibration_task_onehot[calibration_indices],
        device=device,
        batch_size=config.batch_size,
    )

    eval_rows = materialize_eval_rows(rows, metadata["sample_keys"], eval_indices, probs)
    calibration_rows = materialize_eval_rows(
        calibration_rows_source,
        calibration_metadata["sample_keys"],
        calibration_indices,
        calibration_probs,
    )

    grouped = group_rows(eval_rows)
    calibration_grouped = group_rows(calibration_rows)
    fixed_b1 = evaluate_policy(grouped, policy="fixed", fixed_block_number=1)
    fixed_final = evaluate_policy(grouped, policy="final")
    oracle = evaluate_policy(grouped, policy="oracle")
    eos_only = evaluate_policy(grouped, policy="eos_only")
    prediction_stability = evaluate_policy(grouped, policy="prediction_stability")
    calibration_prediction_stability = evaluate_policy(calibration_grouped, policy="prediction_stability")
    thresholds = make_thresholds(config.threshold_start, config.threshold_end, config.threshold_step)
    sweep = [
        {
            "threshold": threshold,
            **evaluate_policy(grouped, policy="adaptive", threshold=threshold),
        }
        for threshold in thresholds
    ]
    best_accuracy = max(sweep, key=lambda row: (row["accuracy"], -row["avg_blocks"]))
    target_accuracy = fixed_final["accuracy"] - config.accuracy_drop_tolerance
    eligible = [row for row in sweep if row["accuracy"] >= target_accuracy]
    best_cost = min(eligible, key=lambda row: (row["avg_blocks"], -row["accuracy"])) if eligible else best_accuracy
    calibration_sweep = [
        {
            "threshold": threshold,
            **evaluate_policy(calibration_grouped, policy="adaptive", threshold=threshold),
        }
        for threshold in thresholds
    ]
    calibration_fixed_final = evaluate_policy(calibration_grouped, policy="final")
    calibration_target = calibration_fixed_final["accuracy"] - config.accuracy_drop_tolerance
    calibration_eligible = [row for row in calibration_sweep if row["accuracy"] >= calibration_target]
    calibrated_best_cost = (
        min(calibration_eligible, key=lambda row: (row["avg_blocks"], -row["accuracy"]))
        if calibration_eligible
        else max(calibration_sweep, key=lambda row: (row["accuracy"], -row["avg_blocks"]))
    )
    calibrated_global = {
        "threshold": calibrated_best_cost["threshold"],
        **evaluate_policy(grouped, policy="adaptive", threshold=calibrated_best_cost["threshold"]),
    }
    early_or_stability_sweep = [
        {
            "threshold": threshold,
            **evaluate_policy(grouped, policy="adaptive_or_prediction_stability", threshold=threshold),
        }
        for threshold in thresholds
    ]
    calibration_early_or_stability_sweep = [
        {
            "threshold": threshold,
            **evaluate_policy(
                calibration_grouped,
                policy="adaptive_or_prediction_stability",
                threshold=threshold,
            ),
        }
        for threshold in thresholds
    ]
    stability_guarded_sweep = [
        {
            "threshold": threshold,
            **evaluate_policy(grouped, policy="stability_guarded_adaptive", threshold=threshold),
        }
        for threshold in thresholds
    ]
    calibration_stability_guarded_sweep = [
        {
            "threshold": threshold,
            **evaluate_policy(
                calibration_grouped,
                policy="stability_guarded_adaptive",
                threshold=threshold,
            ),
        }
        for threshold in thresholds
    ]
    stability_target = calibration_prediction_stability["accuracy"] - config.accuracy_drop_tolerance
    early_or_eligible = [
        row for row in calibration_early_or_stability_sweep if row["accuracy"] >= stability_target
    ]
    early_or_calibrated_best = (
        min(early_or_eligible, key=lambda row: (row["avg_blocks"], -row["accuracy"]))
        if early_or_eligible
        else max(calibration_early_or_stability_sweep, key=lambda row: (row["accuracy"], -row["avg_blocks"]))
    )
    stability_guarded_eligible = [
        row for row in calibration_stability_guarded_sweep if row["accuracy"] >= stability_target
    ]
    stability_guarded_calibrated_best = (
        min(stability_guarded_eligible, key=lambda row: (row["avg_blocks"], -row["accuracy"]))
        if stability_guarded_eligible
        else max(calibration_stability_guarded_sweep, key=lambda row: (row["accuracy"], -row["avg_blocks"]))
    )
    calibrated_early_or_stability = {
        "threshold": early_or_calibrated_best["threshold"],
        **evaluate_policy(
            grouped,
            policy="adaptive_or_prediction_stability",
            threshold=early_or_calibrated_best["threshold"],
        ),
    }
    calibrated_guarded_global = {
        "threshold": stability_guarded_calibrated_best["threshold"],
        **evaluate_policy(
            grouped,
            policy="stability_guarded_adaptive",
            threshold=stability_guarded_calibrated_best["threshold"],
        ),
    }
    prediction_stability_guarded = {
        "threshold": 0.0,
        **evaluate_policy(
            grouped,
            policy="stability_guarded_adaptive",
            threshold=0.0,
        ),
    }
    calibration_prediction_stability_guarded = {
        "threshold": 0.0,
        **evaluate_policy(
            calibration_grouped,
            policy="stability_guarded_adaptive",
            threshold=0.0,
        ),
    }
    if (
        abs(prediction_stability_guarded["accuracy"] - prediction_stability["accuracy"]) > 1e-12
        or abs(prediction_stability_guarded["avg_blocks"] - prediction_stability["avg_blocks"]) > 1e-12
    ):
        raise RuntimeError(
            "stability_guarded_adaptive at threshold 0.0 must match prediction_stability"
        )
    if (
        abs(
            calibration_prediction_stability_guarded["accuracy"]
            - calibration_prediction_stability["accuracy"]
        )
        > 1e-12
        or abs(
            calibration_prediction_stability_guarded["avg_blocks"]
            - calibration_prediction_stability["avg_blocks"]
        )
        > 1e-12
    ):
        raise RuntimeError(
            "calibration stability_guarded_adaptive at threshold 0.0 must match prediction_stability"
        )
    per_task_thresholds = calibrate_per_task_thresholds(
        calibration_grouped,
        thresholds=thresholds,
        accuracy_drop_tolerance=config.accuracy_drop_tolerance,
    )
    calibrated_per_task = evaluate_per_task_thresholds(
        grouped,
        per_task_thresholds,
        fallback_threshold=calibrated_best_cost["threshold"],
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_csv = output_dir / "threshold_sweep.csv"
    guarded_sweep_csv = output_dir / "guarded_threshold_sweep.csv"
    early_or_stability_sweep_csv = output_dir / "early_or_stability_threshold_sweep.csv"
    write_sweep_csv(sweep_csv, sweep)
    write_sweep_csv(guarded_sweep_csv, stability_guarded_sweep)
    write_sweep_csv(early_or_stability_sweep_csv, early_or_stability_sweep)
    summary_path = output_dir / "summary.json"
    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "checkpoint_path": config.checkpoint_path,
        "eval_tasks": eval_config.tasks,
        "calibration_tasks": calibration_config.tasks,
        "split": config.split,
        "calibration_split": config.calibration_split,
        "num_samples": len(grouped),
        "fixed_b1": fixed_b1,
        "fixed_final": fixed_final,
        "oracle": oracle,
        "eos_only": eos_only,
        "prediction_stability": prediction_stability,
        "best_accuracy_threshold": best_accuracy,
        "best_cost_threshold": best_cost,
        "calibrated_global_threshold": calibrated_global,
        "calibrated_early_or_stability_threshold": calibrated_early_or_stability,
        "calibrated_guarded_global_threshold": calibrated_guarded_global,
        "prediction_stability_guarded_threshold0": prediction_stability_guarded,
        "guarded_policy_name": "stability_guarded_adaptive",
        "early_or_stability_policy_name": "adaptive_or_prediction_stability",
        "calibrated_per_task_threshold": calibrated_per_task,
        "per_task_thresholds": per_task_thresholds,
        "threshold_sweep_csv": str(sweep_csv),
        "guarded_threshold_sweep_csv": str(guarded_sweep_csv),
        "early_or_stability_threshold_sweep_csv": str(early_or_stability_sweep_csv),
    }

    summary["swanlab_run_id"] = None
    if config.swanlab_mode != "disabled":
        run = init_experiment(
            stage="cola-adaptive-halt",
            experiment_name=config.experiment_name,
            description="Adaptive halt accuracy-cost frontier from the trained readiness model.",
            config=asdict(config),
            mode=config.swanlab_mode,
            tags=["cola", "official-benchmark", "readiness", "adaptive-halt"],
        )
        try:
            metrics = flatten_summary_metrics(summary)
            log_metrics(metrics, prefix="halt")
            summary["swanlab_run_id"] = getattr(run, "id", None)
        finally:
            finish_experiment()

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


@torch.no_grad()
def predict_readiness(
    *,
    model: ReadinessModel,
    latent: torch.Tensor,
    features: torch.Tensor,
    task_onehot: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, latent.shape[0], batch_size):
        end = start + batch_size
        outputs = model(
            latent[start:end].to(device),
            features[start:end].to(device),
            task_onehot[start:end].to(device),
        )
        chunks.append(torch.sigmoid(outputs["readiness_logits"]).cpu())
    return torch.cat(chunks)


def materialize_eval_rows(
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
                "oracle_ready": bool(row["oracle_ready"]),
                "is_at_or_after_oracle_frontier": bool(row["is_at_or_after_oracle_frontier"]),
                "earliest_ready_block_index": row["earliest_ready_block_index"],
                "contains_eos": bool(row.get("contains_eos")),
                "contains_im_end": bool(row.get("contains_im_end")),
                "contains_stop": bool(row.get("contains_stop")),
                "scored_prediction": row.get("scored_prediction"),
            }
        )
    return eval_rows


def normalize_inputs(
    tensors: dict[str, torch.Tensor],
    norm_stats: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    latent = (tensors["latent"] - norm_stats["latent_mean"]) / norm_stats["latent_std"]
    features = (tensors["features"] - norm_stats["feature_mean"]) / norm_stats["feature_std"]
    return latent, features


def resolve_split_indices(
    sample_keys: list[str],
    train_config: ReadinessTrainConfig,
    split: str,
) -> list[int]:
    if split == "all":
        return list(range(len(sample_keys)))
    splits = split_indices(sample_keys, train_config)
    if split not in splits:
        raise ValueError(f"split must be one of {sorted(splits) + ['all']}")
    return splits[split]


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["sample_key"], []).append(row)
    for sample_rows in grouped.values():
        sample_rows.sort(key=lambda row: row["block_index"])
    return grouped


def evaluate_policy(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    policy: str,
    fixed_block_number: int | None = None,
    threshold: float | None = None,
) -> dict[str, float]:
    correct = 0
    total_blocks = 0
    halted_before_final = 0
    oracle_match = 0
    for rows in grouped.values():
        chosen = choose_row(rows, policy=policy, fixed_block_number=fixed_block_number, threshold=threshold)
        final_block = rows[-1]["block_index"]
        correct += int(chosen["official_correct"])
        total_blocks += int(chosen["block_number"])
        halted_before_final += int(int(chosen["block_index"]) < int(final_block))
        oracle_index = chosen.get("earliest_ready_block_index")
        oracle_match += int(oracle_index is not None and int(chosen["block_index"]) == int(oracle_index))
    n = max(len(grouped), 1)
    return {
        "accuracy": correct / n,
        "avg_blocks": total_blocks / n,
        "halted_before_final_rate": halted_before_final / n,
        "oracle_match_rate": oracle_match / n,
    }


def choose_row(
    rows: list[dict[str, Any]],
    *,
    policy: str,
    fixed_block_number: int | None,
    threshold: float | None,
) -> dict[str, Any]:
    if policy == "fixed":
        assert fixed_block_number is not None
        for row in rows:
            if int(row["block_number"]) >= fixed_block_number:
                return row
        return rows[-1]
    if policy == "final":
        return rows[-1]
    if policy == "oracle":
        for row in rows:
            oracle_index = row.get("earliest_ready_block_index")
            if oracle_index is not None and int(row["block_index"]) == int(oracle_index):
                return row
        return rows[-1]
    if policy == "eos_only":
        for row in rows:
            if row.get("contains_stop") or row.get("contains_eos") or row.get("contains_im_end"):
                return row
        return rows[-1]
    if policy == "prediction_stability":
        annotate_prediction_stability(rows)
        for row in rows:
            if prediction_stability_reached(row):
                return row
        return rows[-1]
    if policy == "adaptive_or_prediction_stability":
        assert threshold is not None
        annotate_prediction_stability(rows)
        for row in rows:
            if float(row["readiness_prob"]) >= threshold:
                return row
            if prediction_stability_reached(row):
                return row
        return rows[-1]
    if policy == "stability_guarded_adaptive":
        assert threshold is not None
        annotate_prediction_stability(rows)
        for row in rows:
            if prediction_stability_reached(row) and float(row["readiness_prob"]) >= threshold:
                return row
        return rows[-1]
    if policy == "adaptive":
        assert threshold is not None
        for row in rows:
            if float(row["readiness_prob"]) >= threshold:
                return row
        return rows[-1]
    raise ValueError(f"unknown policy: {policy}")


def prediction_stability_reached(row: dict[str, Any]) -> bool:
    cached = row.get("_prediction_stability_reached")
    if cached is not None:
        return bool(cached)
    raise RuntimeError("prediction stability state was not initialized")


def annotate_prediction_stability(rows: list[dict[str, Any]]) -> None:
    previous_prediction: str | None = None
    stable_streak = 0
    for row in rows:
        prediction = normalize_prediction(row.get("scored_prediction"))
        if not prediction:
            previous_prediction = None
            stable_streak = 0
            row["_prediction_stability_reached"] = False
            continue
        if prediction == previous_prediction:
            stable_streak += 1
        else:
            previous_prediction = prediction
            stable_streak = 1
        row["_prediction_stability_reached"] = stable_streak >= 2


def normalize_prediction(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def calibrate_per_task_thresholds(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    thresholds: list[float],
    accuracy_drop_tolerance: float,
) -> dict[str, float]:
    by_task: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for key, rows in grouped.items():
        by_task.setdefault(rows[0]["task"], {})[key] = rows
    result: dict[str, float] = {}
    for task, task_grouped in by_task.items():
        fixed_final = evaluate_policy(task_grouped, policy="final")
        target = fixed_final["accuracy"] - accuracy_drop_tolerance
        sweep = [
            {
                "threshold": threshold,
                **evaluate_policy(task_grouped, policy="adaptive", threshold=threshold),
            }
            for threshold in thresholds
        ]
        eligible = [row for row in sweep if row["accuracy"] >= target]
        chosen = min(eligible, key=lambda row: (row["avg_blocks"], -row["accuracy"])) if eligible else max(
            sweep,
            key=lambda row: (row["accuracy"], -row["avg_blocks"]),
        )
        result[task] = float(chosen["threshold"])
    return result


def evaluate_per_task_thresholds(
    grouped: dict[str, list[dict[str, Any]]],
    thresholds: dict[str, float],
    *,
    fallback_threshold: float,
) -> dict[str, float]:
    correct = 0
    total_blocks = 0
    halted_before_final = 0
    oracle_match = 0
    for rows in grouped.values():
        threshold = thresholds.get(rows[0]["task"], fallback_threshold)
        chosen = choose_row(rows, policy="adaptive", fixed_block_number=None, threshold=threshold)
        final_block = rows[-1]["block_index"]
        correct += int(chosen["official_correct"])
        total_blocks += int(chosen["block_number"])
        halted_before_final += int(int(chosen["block_index"]) < int(final_block))
        oracle_index = chosen.get("earliest_ready_block_index")
        oracle_match += int(oracle_index is not None and int(chosen["block_index"]) == int(oracle_index))
    n = max(len(grouped), 1)
    return {
        "accuracy": correct / n,
        "avg_blocks": total_blocks / n,
        "halted_before_final_rate": halted_before_final / n,
        "oracle_match_rate": oracle_match / n,
    }


def make_thresholds(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("threshold_step must be positive")
    thresholds: list[float] = []
    value = start
    while value <= end + 1e-9:
        thresholds.append(round(value, 6))
        value += step
    return thresholds


def write_sweep_csv(path: Path, sweep: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold",
                "accuracy",
                "avg_blocks",
                "halted_before_final_rate",
                "oracle_match_rate",
            ],
        )
        writer.writeheader()
        for row in sweep:
            writer.writerow(row)


def flatten_summary_metrics(summary: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for prefix in [
        "fixed_b1",
        "fixed_final",
        "oracle",
        "eos_only",
        "prediction_stability",
        "best_accuracy_threshold",
        "best_cost_threshold",
        "calibrated_global_threshold",
        "calibrated_early_or_stability_threshold",
        "calibrated_guarded_global_threshold",
        "prediction_stability_guarded_threshold0",
        "calibrated_per_task_threshold",
    ]:
        for key, value in summary[prefix].items():
            metrics[f"{prefix}/{key}"] = float(value)
    return metrics


def parse_args() -> AdaptiveHaltEvalConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", default=AdaptiveHaltEvalConfig.checkpoint_path)
    parser.add_argument("--output-dir", default=AdaptiveHaltEvalConfig.output_dir)
    parser.add_argument("--eval-labels-dir", default=AdaptiveHaltEvalConfig.eval_labels_dir)
    parser.add_argument("--eval-tasks", default=AdaptiveHaltEvalConfig.eval_tasks)
    parser.add_argument("--calibration-tasks", default=AdaptiveHaltEvalConfig.calibration_tasks)
    parser.add_argument("--split", default=AdaptiveHaltEvalConfig.split)
    parser.add_argument("--calibration-split", default=AdaptiveHaltEvalConfig.calibration_split)
    parser.add_argument("--threshold-start", type=float, default=AdaptiveHaltEvalConfig.threshold_start)
    parser.add_argument("--threshold-end", type=float, default=AdaptiveHaltEvalConfig.threshold_end)
    parser.add_argument("--threshold-step", type=float, default=AdaptiveHaltEvalConfig.threshold_step)
    parser.add_argument(
        "--accuracy-drop-tolerance",
        type=float,
        default=AdaptiveHaltEvalConfig.accuracy_drop_tolerance,
    )
    parser.add_argument("--batch-size", type=int, default=AdaptiveHaltEvalConfig.batch_size)
    parser.add_argument("--device", default=AdaptiveHaltEvalConfig.device)
    parser.add_argument("--swanlab-mode", default=AdaptiveHaltEvalConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=AdaptiveHaltEvalConfig.experiment_name)
    args = parser.parse_args()
    return AdaptiveHaltEvalConfig(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        eval_labels_dir=args.eval_labels_dir,
        eval_tasks=args.eval_tasks,
        calibration_tasks=args.calibration_tasks,
        split=args.split,
        calibration_split=args.calibration_split,
        threshold_start=args.threshold_start,
        threshold_end=args.threshold_end,
        threshold_step=args.threshold_step,
        accuracy_drop_tolerance=args.accuracy_drop_tolerance,
        batch_size=args.batch_size,
        device=args.device,
        swanlab_mode=args.swanlab_mode,
        experiment_name=args.experiment_name,
    )


def main() -> None:
    summary = evaluate_adaptive_halt(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
