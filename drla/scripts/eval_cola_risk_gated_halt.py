"""Evaluate continuation-risk gated Cola halt policies."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any

import torch

from drla.scripts.eval_cola_adaptive_halt import (
    choose_row,
    evaluate_policy,
    group_rows,
    normalize_inputs,
    predict_readiness,
    resolve_split_indices,
    annotate_prediction_stability,
    prediction_stability_reached,
)
from drla.scripts.train_cola_continuation_risk_model import (
    ContinuationRiskModel,
    ContinuationRiskTrainConfig,
    row_features as risk_row_features,
)
from drla.scripts.train_cola_readiness_model import (
    OFFICIAL_COLA_TASKS,
    ReadinessModel,
    ReadinessTrainConfig,
    build_tensors,
    load_training_rows,
    resolve_device,
)
from drla.tracking import finish_experiment, init_experiment, log_metrics
from drla.tracking import require_swanlab_disabled_for_non_training


@dataclass(frozen=True)
class RiskGatedHaltEvalConfig:
    eval_summary_path: str | None = None
    readiness_checkpoint_path: str | None = None
    risk_checkpoint_path: str = "/data1/luyifei/drla/outputs/cola_continuation_risk_model/official8_b64_process_no_task_seed20260524/checkpoints/best_checkpoint.pt"
    output_dir: str = "/data1/luyifei/drla/outputs/cola_risk_gated_halt/debug"
    eval_labels_dir: str | None = None
    eval_tasks: str | None = None
    calibration_tasks: str | None = None
    split: str = "test"
    calibration_split: str = "valid"
    readiness_threshold: float | None = None
    readiness_threshold_values: str = ""
    risk_threshold_start: float = 0.0
    risk_threshold_end: float = 1.0
    risk_threshold_step: float = 0.01
    risk_threshold_selection_mode: str = "min_blocks"
    require_contentful_prediction: bool = False
    require_fragment_complete_prediction: bool = False
    require_stable_single_choice: bool = False
    stable_single_choice_max_block: int | None = None
    stable_single_choice_guard_scopes: str = ""
    entropy_max_values: str = ""
    top_prob_min_values: str = ""
    accuracy_drop_tolerance: float = 0.0
    require_zero_calibration_loss: bool = False
    batch_size: int = 512
    device: str = "auto"
    swanlab_mode: str = "disabled"
    experiment_name: str = "official8-risk-gated-halt-eval"


def evaluate_risk_gated_halt(config: RiskGatedHaltEvalConfig) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        config.swanlab_mode,
        script_kind="risk-gated halt evaluation",
    )
    config = hydrate_from_eval_summary(config)
    if not config.readiness_checkpoint_path:
        raise ValueError("readiness_checkpoint_path is required")
    if config.readiness_threshold is None:
        raise ValueError("readiness_threshold is required")

    readiness_checkpoint = torch.load(config.readiness_checkpoint_path, map_location="cpu")
    readiness_config = ReadinessTrainConfig(**readiness_checkpoint["config"])
    feature_fields = readiness_checkpoint.get("feature_fields") or readiness_checkpoint.get("metadata", {}).get(
        "feature_fields"
    )
    eval_config = replace(
        readiness_config,
        labels_dir=config.eval_labels_dir or readiness_config.labels_dir,
        tasks=config.eval_tasks or readiness_config.tasks,
    )
    calibration_config = replace(
        readiness_config,
        labels_dir=config.eval_labels_dir or readiness_config.labels_dir,
        tasks=config.calibration_tasks or readiness_config.tasks,
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

    device = resolve_device(config.device)
    readiness_probs = predict_readiness_for_rows(
        checkpoint=readiness_checkpoint,
        train_config=readiness_config,
        tensors=tensors,
        indices=eval_indices,
        device=device,
        batch_size=config.batch_size,
    )
    calibration_readiness_probs = predict_readiness_for_rows(
        checkpoint=readiness_checkpoint,
        train_config=readiness_config,
        tensors=calibration_tensors,
        indices=calibration_indices,
        device=device,
        batch_size=config.batch_size,
    )

    risk_checkpoint = torch.load(config.risk_checkpoint_path, map_location="cpu")
    risk_probs = predict_risk_for_rows(
        checkpoint=risk_checkpoint,
        rows=rows,
        indices=eval_indices,
        device=device,
        batch_size=config.batch_size,
    )
    calibration_risk_probs = predict_risk_for_rows(
        checkpoint=risk_checkpoint,
        rows=calibration_rows_source,
        indices=calibration_indices,
        device=device,
        batch_size=config.batch_size,
    )

    eval_rows = materialize_rows(
        rows,
        metadata["sample_keys"],
        eval_indices,
        readiness_probs,
        risk_probs,
    )
    calibration_rows = materialize_rows(
        calibration_rows_source,
        calibration_metadata["sample_keys"],
        calibration_indices,
        calibration_readiness_probs,
        calibration_risk_probs,
    )
    grouped = group_rows(eval_rows)
    calibration_grouped = group_rows(calibration_rows)

    fixed_final = evaluate_policy(grouped, policy="final")
    prediction_stability = evaluate_policy(grouped, policy="prediction_stability")
    early_or = evaluate_policy(
        grouped,
        policy="adaptive_or_prediction_stability",
        threshold=config.readiness_threshold,
    )
    oracle_prefix_gated = evaluate_risk_gated_policy(
        grouped,
        readiness_threshold=config.readiness_threshold,
        risk_threshold=1.0,
        use_oracle_prefix=True,
        require_contentful_prediction=False,
        require_fragment_complete_prediction=False,
        require_stable_single_choice=False,
        stable_single_choice_max_block=None,
    )
    calibration_prediction_stability = evaluate_policy(calibration_grouped, policy="prediction_stability")
    risk_thresholds = make_thresholds(
        config.risk_threshold_start,
        config.risk_threshold_end,
        config.risk_threshold_step,
    )
    readiness_thresholds = parse_float_values(config.readiness_threshold_values) or [config.readiness_threshold]
    entropy_max_values = parse_optional_float_values(config.entropy_max_values)
    top_prob_min_values = parse_optional_float_values(config.top_prob_min_values)
    single_choice_guard_scopes = parse_single_choice_guard_scopes(
        config.stable_single_choice_guard_scopes,
        require_stable_single_choice=config.require_stable_single_choice,
        stable_single_choice_max_block=config.stable_single_choice_max_block,
    )
    sweep = build_policy_sweep(
        grouped,
        risk_thresholds=risk_thresholds,
        readiness_thresholds=readiness_thresholds,
        entropy_max_values=entropy_max_values,
        top_prob_min_values=top_prob_min_values,
        require_contentful_prediction=config.require_contentful_prediction,
        require_fragment_complete_prediction=config.require_fragment_complete_prediction,
        single_choice_guard_scopes=single_choice_guard_scopes,
    )
    calibration_sweep = build_policy_sweep(
        calibration_grouped,
        risk_thresholds=risk_thresholds,
        readiness_thresholds=readiness_thresholds,
        entropy_max_values=entropy_max_values,
        top_prob_min_values=top_prob_min_values,
        require_contentful_prediction=config.require_contentful_prediction,
        require_fragment_complete_prediction=config.require_fragment_complete_prediction,
        single_choice_guard_scopes=single_choice_guard_scopes,
    )
    calibration_target = calibration_prediction_stability["accuracy"] - config.accuracy_drop_tolerance
    calibrated_choice = select_risk_threshold(
        calibration_sweep=calibration_sweep,
        calibration_target=calibration_target,
        prediction_stability_avg_blocks=calibration_prediction_stability["avg_blocks"],
        mode=config.risk_threshold_selection_mode,
        require_zero_calibration_loss=config.require_zero_calibration_loss,
    )
    selected_require_stable_single_choice, selected_stable_single_choice_max_block = (
        single_choice_scope_to_params(calibrated_choice["single_choice_guard_scope"])
    )
    calibrated_risk_gated = {
        "risk_threshold": calibrated_choice["risk_threshold"],
        "readiness_threshold": calibrated_choice["readiness_threshold"],
        "entropy_max": calibrated_choice["entropy_max"],
        "top_prob_min": calibrated_choice["top_prob_min"],
        "single_choice_guard_scope": calibrated_choice["single_choice_guard_scope"],
        "require_stable_single_choice": selected_require_stable_single_choice,
        "stable_single_choice_max_block": selected_stable_single_choice_max_block,
        **evaluate_risk_gated_policy(
            grouped,
            readiness_threshold=calibrated_choice["readiness_threshold"],
            risk_threshold=calibrated_choice["risk_threshold"],
            entropy_max=calibrated_choice["entropy_max"],
            top_prob_min=calibrated_choice["top_prob_min"],
            require_contentful_prediction=config.require_contentful_prediction,
            require_fragment_complete_prediction=config.require_fragment_complete_prediction,
            require_stable_single_choice=selected_require_stable_single_choice,
            stable_single_choice_max_block=selected_stable_single_choice_max_block,
        ),
    }

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_csv = output_dir / "risk_threshold_sweep.csv"
    calibration_sweep_csv = output_dir / "calibration_risk_threshold_sweep.csv"
    write_sweep_csv(sweep_csv, sweep)
    write_sweep_csv(calibration_sweep_csv, calibration_sweep)
    summary_path = output_dir / "summary.json"
    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "eval_tasks": eval_config.tasks,
        "calibration_tasks": calibration_config.tasks,
        "split": config.split,
        "calibration_split": config.calibration_split,
        "num_samples": len(grouped),
        "readiness_threshold": calibrated_choice["readiness_threshold"],
        "base_readiness_threshold": config.readiness_threshold,
        "readiness_threshold_values": readiness_thresholds,
        "entropy_max_values": encode_optional_float_values(entropy_max_values),
        "top_prob_min_values": encode_optional_float_values(top_prob_min_values),
        "stable_single_choice_guard_scopes": single_choice_guard_scopes,
        "fixed_final": fixed_final,
        "prediction_stability": prediction_stability,
        "early_or_stability": early_or,
        "oracle_prefix_gated": oracle_prefix_gated,
        "calibrated_validation_risk_gated": calibrated_choice,
        "calibrated_risk_gated": calibrated_risk_gated,
        "risk_threshold_sweep_csv": str(sweep_csv),
        "calibration_risk_threshold_sweep_csv": str(calibration_sweep_csv),
    }

    summary["swanlab_run_id"] = None
    if config.swanlab_mode != "disabled":
        run = init_experiment(
            stage="cola-risk-gated-halt",
            experiment_name=config.experiment_name,
            description="Continuation-risk gated readiness halt evaluation.",
            config=asdict(config),
            mode=config.swanlab_mode,
            tags=["cola", "official-benchmark", "halt", "continuation-risk"],
        )
        try:
            log_metrics(flatten_metrics(summary), prefix="risk_halt")
            summary["swanlab_run_id"] = getattr(run, "id", None)
        finally:
            finish_experiment()

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def hydrate_from_eval_summary(config: RiskGatedHaltEvalConfig) -> RiskGatedHaltEvalConfig:
    if not config.eval_summary_path:
        return config
    summary = json.loads(Path(config.eval_summary_path).read_text(encoding="utf-8"))
    summary_config = summary.get("config", {})

    defaults = RiskGatedHaltEvalConfig()

    def keep_or_summary(value: Any, key: str, default: Any = None) -> Any:
        if value is not None and value != default:
            return value
        return summary_config.get(key)

    readiness_threshold = config.readiness_threshold
    if readiness_threshold is None:
        readiness_threshold = summary.get("readiness_threshold")
        if readiness_threshold is None:
            readiness_threshold = summary.get("calibrated_early_or_stability_threshold", {}).get("threshold")
    return replace(
        config,
        readiness_checkpoint_path=(
            keep_or_summary(
                config.readiness_checkpoint_path,
                "readiness_checkpoint_path",
                defaults.readiness_checkpoint_path,
            )
            or keep_or_summary(config.readiness_checkpoint_path, "checkpoint_path", defaults.readiness_checkpoint_path)
            or summary.get("checkpoint_path")
        ),
        risk_checkpoint_path=(
            keep_or_summary(config.risk_checkpoint_path, "risk_checkpoint_path", defaults.risk_checkpoint_path)
            or config.risk_checkpoint_path
        ),
        eval_labels_dir=keep_or_summary(config.eval_labels_dir, "eval_labels_dir", defaults.eval_labels_dir),
        eval_tasks=keep_or_summary(config.eval_tasks, "eval_tasks", defaults.eval_tasks) or summary.get("eval_tasks"),
        calibration_tasks=keep_or_summary(
            config.calibration_tasks,
            "calibration_tasks",
            defaults.calibration_tasks,
        ),
        split=config.split if config.split != RiskGatedHaltEvalConfig.split else summary.get("split", config.split),
        calibration_split=(
            config.calibration_split
            if config.calibration_split != RiskGatedHaltEvalConfig.calibration_split
            else summary.get("calibration_split", config.calibration_split)
        ),
        readiness_threshold=readiness_threshold,
    )


def predict_readiness_for_rows(
    *,
    checkpoint: dict[str, Any],
    train_config: ReadinessTrainConfig,
    tensors: dict[str, torch.Tensor],
    indices: list[int],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    latent, features = normalize_inputs(tensors, checkpoint["norm_stats"])
    task_onehot = tensors["task_onehot"]
    model = ReadinessModel(
        latent_dim=latent.shape[1],
        feature_dim=features.shape[1],
        task_dim=task_onehot.shape[1],
        hidden_dim=train_config.hidden_dim,
        dropout=train_config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return predict_readiness(
        model=model,
        latent=latent[indices],
        features=features[indices],
        task_onehot=task_onehot[indices],
        device=device,
        batch_size=batch_size,
    )


@torch.no_grad()
def predict_risk_for_rows(
    *,
    checkpoint: dict[str, Any],
    rows: list[dict[str, Any]],
    indices: list[int],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    risk_config = ContinuationRiskTrainConfig(**checkpoint["config"])
    risk_feature_fields = checkpoint.get("feature_fields") or checkpoint.get("metadata", {}).get("feature_fields")
    features = torch.tensor(
        [risk_row_features(rows[idx], feature_fields=risk_feature_fields) for idx in indices],
        dtype=torch.float32,
    )
    task_to_idx = {task: idx for idx, task in enumerate(OFFICIAL_COLA_TASKS)}
    task_onehot_rows: list[list[float]] = []
    for idx in indices:
        task_vec = [0.0] * len(OFFICIAL_COLA_TASKS)
        if risk_config.signal_mode == "process":
            task_vec[task_to_idx[rows[idx]["task"]]] = 1.0
        task_onehot_rows.append(task_vec)
    task_onehot = torch.tensor(task_onehot_rows, dtype=torch.float32)
    norm_stats = checkpoint["norm_stats"]
    features = (features - norm_stats["feature_mean"]) / norm_stats["feature_std"]
    model = ContinuationRiskModel(
        feature_dim=features.shape[1],
        task_dim=task_onehot.shape[1],
        hidden_dim=risk_config.hidden_dim,
        dropout=risk_config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    chunks: list[torch.Tensor] = []
    for start in range(0, features.shape[0], batch_size):
        end = start + batch_size
        logits = model(features[start:end].to(device), task_onehot[start:end].to(device))
        chunks.append(torch.sigmoid(logits).cpu())
    return torch.cat(chunks)


def materialize_rows(
    rows: list[dict[str, Any]],
    sample_keys: list[str],
    indices: list[int],
    readiness_probs: torch.Tensor,
    risk_probs: torch.Tensor,
) -> list[dict[str, Any]]:
    materialized = []
    for local_idx, row_idx in enumerate(indices):
        row = rows[row_idx]
        materialized.append(
            {
                "task": row["task"],
                "sample_id": row["sample_id"],
                "sample_key": sample_keys[row_idx],
                "block_index": int(row["block_index"]),
                "block_number": int(row["block_number"]),
                "readiness_prob": float(readiness_probs[local_idx].item()),
                "continuation_risk_prob": float(risk_probs[local_idx].item()),
                "official_correct": bool(row["official_correct"]),
                "oracle_ready": bool(row["oracle_ready"]),
                "is_at_or_after_oracle_frontier": bool(row["is_at_or_after_oracle_frontier"]),
                "earliest_ready_block_index": row["earliest_ready_block_index"],
                "contains_eos": bool(row.get("contains_eos")),
                "contains_im_end": bool(row.get("contains_im_end")),
                "contains_stop": bool(row.get("contains_stop")),
                "scored_prediction": row.get("scored_prediction"),
                "scored_target": row.get("scored_target"),
                "official_processed_generation": row.get("official_processed_generation"),
                "decode_text_so_far": row.get("decode_text_so_far"),
                "token_entropy_mean": row.get("token_entropy_mean"),
                "token_top_prob_mean": row.get("token_top_prob_mean"),
                "same_text_streak": row.get("same_text_streak"),
                "answer_changed": row.get("answer_changed"),
                "scored_prediction_same_streak": row.get("scored_prediction_same_streak"),
                "processed_generation_same_streak": row.get("processed_generation_same_streak"),
            }
        )
    return materialized


def evaluate_risk_gated_policy(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    readiness_threshold: float,
    risk_threshold: float,
    entropy_max: float | None = None,
    top_prob_min: float | None = None,
    use_oracle_prefix: bool = False,
    require_contentful_prediction: bool = False,
    require_fragment_complete_prediction: bool = False,
    require_stable_single_choice: bool = False,
    stable_single_choice_max_block: int | None = None,
) -> dict[str, float]:
    correct = 0
    total_blocks = 0
    halted_before_final = 0
    before_stability = 0
    loss_vs_final = 0
    gain_vs_final = 0
    loss_vs_prediction_stability = 0
    gain_vs_prediction_stability = 0
    prefix_skips = 0
    shape_guard_skips = 0
    fragment_guard_skips = 0
    uncertainty_guard_skips = 0
    single_choice_guard_skips = 0
    for rows in grouped.values():
        chosen, skips, shape_skips, fragment_skips, uncertainty_skips, single_choice_skips = choose_risk_gated_row(
            rows,
            readiness_threshold=readiness_threshold,
            risk_threshold=risk_threshold,
            entropy_max=entropy_max,
            top_prob_min=top_prob_min,
            use_oracle_prefix=use_oracle_prefix,
            require_contentful_prediction=require_contentful_prediction,
            require_fragment_complete_prediction=require_fragment_complete_prediction,
            require_stable_single_choice=require_stable_single_choice,
            stable_single_choice_max_block=stable_single_choice_max_block,
        )
        prefix_skips += skips
        shape_guard_skips += shape_skips
        fragment_guard_skips += fragment_skips
        uncertainty_guard_skips += uncertainty_skips
        single_choice_guard_skips += single_choice_skips
        final_block = rows[-1]["block_index"]
        stability_row = choose_row(rows, policy="prediction_stability", fixed_block_number=None, threshold=None)
        chosen_correct = bool(chosen["official_correct"])
        final_correct = bool(rows[-1]["official_correct"])
        stability_correct = bool(stability_row["official_correct"])
        correct += int(chosen_correct)
        total_blocks += int(chosen["block_number"])
        halted_before_final += int(int(chosen["block_index"]) < int(final_block))
        before_stability += int(int(chosen["block_index"]) < int(stability_row["block_index"]))
        loss_vs_final += int(final_correct and not chosen_correct)
        gain_vs_final += int(chosen_correct and not final_correct)
        loss_vs_prediction_stability += int(stability_correct and not chosen_correct)
        gain_vs_prediction_stability += int(chosen_correct and not stability_correct)
    n = max(len(grouped), 1)
    return {
        "accuracy": correct / n,
        "avg_blocks": total_blocks / n,
        "halted_before_final_rate": halted_before_final / n,
        "before_prediction_stability_rate": before_stability / n,
        "loss_count_vs_final": float(loss_vs_final),
        "gain_count_vs_final": float(gain_vs_final),
        "loss_count_vs_prediction_stability": float(loss_vs_prediction_stability),
        "gain_count_vs_prediction_stability": float(gain_vs_prediction_stability),
        "prefix_skip_count": float(prefix_skips),
        "shape_guard_skip_count": float(shape_guard_skips),
        "fragment_guard_skip_count": float(fragment_guard_skips),
        "uncertainty_guard_skip_count": float(uncertainty_guard_skips),
        "single_choice_guard_skip_count": float(single_choice_guard_skips),
    }


def choose_risk_gated_row(
    rows: list[dict[str, Any]],
    *,
    readiness_threshold: float,
    risk_threshold: float,
    entropy_max: float | None,
    top_prob_min: float | None,
    use_oracle_prefix: bool,
    require_contentful_prediction: bool = False,
    require_fragment_complete_prediction: bool = False,
    require_stable_single_choice: bool = False,
    stable_single_choice_max_block: int | None = None,
) -> tuple[dict[str, Any], int, int, int, int, int]:
    annotate_prediction_stability(rows)
    final_prediction = normalize_text(rows[-1].get("scored_prediction"))
    prefix_skips = 0
    shape_guard_skips = 0
    fragment_guard_skips = 0
    uncertainty_guard_skips = 0
    single_choice_guard_skips = 0
    for row in rows:
        if float(row["readiness_prob"]) >= readiness_threshold:
            risk_blocks = float(row["continuation_risk_prob"]) >= risk_threshold
            if use_oracle_prefix:
                risk_blocks = is_strict_prefix(normalize_text(row.get("scored_prediction")), final_prediction)
            if require_contentful_prediction and not is_contentful_prediction(row.get("scored_prediction")):
                risk_blocks = True
                shape_guard_skips += 1
            if require_fragment_complete_prediction and is_incomplete_prediction_fragment(row):
                risk_blocks = True
                fragment_guard_skips += 1
            if not uncertainty_guard_passes(row, entropy_max=entropy_max, top_prob_min=top_prob_min):
                risk_blocks = True
                uncertainty_guard_skips += 1
            if should_guard_single_choice(
                row,
                require_stable_single_choice=require_stable_single_choice,
                stable_single_choice_max_block=stable_single_choice_max_block,
            ):
                if not prediction_stability_reached(row):
                    risk_blocks = True
                    single_choice_guard_skips += 1
            if risk_blocks:
                prefix_skips += 1
            else:
                return (
                    row,
                    prefix_skips,
                    shape_guard_skips,
                    fragment_guard_skips,
                    uncertainty_guard_skips,
                    single_choice_guard_skips,
                )
        if prediction_stability_reached(row):
            return (
                row,
                prefix_skips,
                shape_guard_skips,
                fragment_guard_skips,
                uncertainty_guard_skips,
                single_choice_guard_skips,
            )
    return (
        rows[-1],
        prefix_skips,
        shape_guard_skips,
        fragment_guard_skips,
        uncertainty_guard_skips,
        single_choice_guard_skips,
    )


def build_policy_sweep(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    risk_thresholds: list[float],
    readiness_thresholds: list[float],
    entropy_max_values: list[float | None],
    top_prob_min_values: list[float | None],
    require_contentful_prediction: bool,
    require_fragment_complete_prediction: bool,
    single_choice_guard_scopes: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for risk_threshold, readiness_threshold, entropy_max, top_prob_min, single_choice_guard_scope in product(
        risk_thresholds,
        readiness_thresholds,
        entropy_max_values,
        top_prob_min_values,
        single_choice_guard_scopes,
    ):
        require_stable_single_choice, stable_single_choice_max_block = single_choice_scope_to_params(
            single_choice_guard_scope
        )
        rows.append(
            {
                "risk_threshold": risk_threshold,
                "readiness_threshold": readiness_threshold,
                "entropy_max": entropy_max,
                "top_prob_min": top_prob_min,
                "single_choice_guard_scope": single_choice_guard_scope,
                "require_stable_single_choice": require_stable_single_choice,
                "stable_single_choice_max_block": stable_single_choice_max_block,
                **evaluate_risk_gated_policy(
                    grouped,
                    readiness_threshold=readiness_threshold,
                    risk_threshold=risk_threshold,
                    entropy_max=entropy_max,
                    top_prob_min=top_prob_min,
                    require_contentful_prediction=require_contentful_prediction,
                    require_fragment_complete_prediction=require_fragment_complete_prediction,
                    require_stable_single_choice=require_stable_single_choice,
                    stable_single_choice_max_block=stable_single_choice_max_block,
                ),
            }
        )
    return rows


def make_thresholds(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("risk_threshold_step must be positive")
    thresholds = []
    value = start
    while value <= end + 1e-9:
        thresholds.append(round(value, 6))
        value += step
    return thresholds


def parse_optional_float_values(raw: str) -> list[float | None]:
    if not raw:
        return [None]
    values: list[float | None] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() in {"none", "null", "off"}:
            values.append(None)
        else:
            values.append(float(item))
    return values or [None]


def parse_float_values(raw: str) -> list[float]:
    if not raw:
        return []
    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def encode_optional_float_values(values: list[float | None]) -> list[str]:
    return ["none" if value is None else str(value) for value in values]


def parse_single_choice_guard_scopes(
    raw: str,
    *,
    require_stable_single_choice: bool,
    stable_single_choice_max_block: int | None,
) -> list[str]:
    if not raw:
        return [
            default_single_choice_guard_scope(
                require_stable_single_choice=require_stable_single_choice,
                stable_single_choice_max_block=stable_single_choice_max_block,
            )
        ]
    scopes = []
    for item in raw.split(","):
        scope = normalize_single_choice_guard_scope(item)
        if scope not in scopes:
            scopes.append(scope)
    return scopes or ["off"]


def default_single_choice_guard_scope(
    *,
    require_stable_single_choice: bool,
    stable_single_choice_max_block: int | None,
) -> str:
    if not require_stable_single_choice:
        return "off"
    if stable_single_choice_max_block is None:
        return "all"
    return str(stable_single_choice_max_block)


def normalize_single_choice_guard_scope(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"", "off", "false", "no", "none", "null"}:
        return "off"
    if text in {"all", "true", "yes"}:
        return "all"
    block = int(text)
    if block < 1:
        return "off"
    return str(block)


def single_choice_scope_to_params(scope: Any) -> tuple[bool, int | None]:
    normalized = normalize_single_choice_guard_scope(scope)
    if normalized == "off":
        return False, None
    if normalized == "all":
        return True, None
    return True, int(normalized)


def select_risk_threshold(
    *,
    calibration_sweep: list[dict[str, Any]],
    calibration_target: float,
    prediction_stability_avg_blocks: float,
    mode: str,
    require_zero_calibration_loss: bool,
) -> dict[str, Any]:
    eligible = [row for row in calibration_sweep if row["accuracy"] >= calibration_target]
    if require_zero_calibration_loss:
        zero_loss_rows = [row for row in eligible if row["loss_count_vs_prediction_stability"] == 0.0]
        if zero_loss_rows:
            eligible = zero_loss_rows
    if not eligible:
        return max(calibration_sweep, key=lambda row: (row["accuracy"], -row["avg_blocks"]))
    if mode == "min_blocks":
        return min(eligible, key=lambda row: (row["avg_blocks"], -row["accuracy"]))
    if mode == "first_saving":
        saving_rows = [
            row
            for row in eligible
            if row["avg_blocks"] < prediction_stability_avg_blocks - 1e-12
        ]
        if saving_rows:
            return min(
                saving_rows,
                key=lambda row: (row["risk_threshold"], -row["readiness_threshold"], row["avg_blocks"]),
            )
        return min(
            eligible,
            key=lambda row: (row["risk_threshold"], -row["readiness_threshold"], row["avg_blocks"]),
        )
    raise ValueError(
        "risk_threshold_selection_mode must be one of: min_blocks, first_saving"
    )


def write_sweep_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "risk_threshold",
        "readiness_threshold",
        "entropy_max",
        "top_prob_min",
        "single_choice_guard_scope",
        "require_stable_single_choice",
        "stable_single_choice_max_block",
        "accuracy",
        "avg_blocks",
        "halted_before_final_rate",
        "before_prediction_stability_rate",
        "loss_count_vs_final",
        "gain_count_vs_final",
        "loss_count_vs_prediction_stability",
        "gain_count_vs_prediction_stability",
        "prefix_skip_count",
        "shape_guard_skip_count",
        "fragment_guard_skip_count",
        "uncertainty_guard_skip_count",
        "single_choice_guard_skip_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_metrics(summary: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for prefix in [
        "fixed_final",
        "prediction_stability",
        "early_or_stability",
        "oracle_prefix_gated",
        "calibrated_validation_risk_gated",
        "calibrated_risk_gated",
    ]:
        for key, value in summary[prefix].items():
            if value is None:
                continue
            if not isinstance(value, (int, float, bool)):
                continue
            metrics[f"{prefix}/{key}"] = float(value)
    metrics["readiness_threshold"] = float(summary["readiness_threshold"])
    return metrics


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def is_strict_prefix(value: str, final_value: str) -> bool:
    return bool(value and final_value and value != final_value and final_value.startswith(value))


def is_contentful_prediction(value: Any) -> bool:
    text = normalize_text(value)
    return any(ch.isalnum() for ch in text)


def is_incomplete_prediction_fragment(row: dict[str, Any]) -> bool:
    if prediction_stability_reached(row):
        return False
    text = normalize_text(row.get("scored_prediction"))
    if not text or is_single_choice_prediction(text):
        return False
    raw_text = str(row.get("scored_prediction") or "").strip()
    tokens = text.split()
    last_token = tokens[-1] if tokens else ""
    if "�" in raw_text:
        return True
    if has_trailing_continuation_marker(text):
        return True
    if has_open_answer_marker(text):
        return True
    if is_bare_currency_amount(text):
        return True
    if is_numeric_range_prefix(text):
        return True
    if ends_with_connector_token(text):
        return True
    if has_likely_truncated_hyphen_token(text):
        return True
    if re.search(r"\d\.$", text):
        return True
    if text.isdigit():
        return True
    if re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]?", text):
        return True
    if re.fullmatch(r"[A-Za-z]{1,3}\.", text):
        return True
    if re.fullmatch(r"[A-Za-z]\.", last_token):
        return True
    if re.fullmatch(r"\w", text, flags=re.UNICODE):
        return True
    if len(tokens) == 1 and "-" in text and re.search(r"[A-Za-z]$", text):
        return True
    if len(tokens) == 1 and text.isalpha():
        return True
    return is_unstable_short_phrase(text)


def has_trailing_continuation_marker(text: str) -> bool:
    if text.endswith((",", ":", ";", "-", "–", "—", "/", "&", "+")):
        return True
    return bool(re.search(r"(?:^|\s)\[(?:step|substeps|title)\]$", text))


def has_open_answer_marker(text: str) -> bool:
    if text.endswith(("(", "[", "{", '"', "'")):
        return True
    return any(text.count(open_ch) > text.count(close_ch) for open_ch, close_ch in [("(", ")"), ("[", "]"), ("{", "}")])


def is_bare_currency_amount(text: str) -> bool:
    return bool(re.fullmatch(r"[$£€¥]\s*\d[\d,]*(?:\.\d+)?", text))


def is_numeric_range_prefix(text: str) -> bool:
    if re.fullmatch(r"\d[\d,]*[,/]", text):
        return True
    return bool(re.fullmatch(r"\d{2,4}\s*[-–—]\s*\d{1,3}", text))


def ends_with_connector_token(text: str) -> bool:
    return bool(
        re.search(
            r"(?:^|\s)(?:&|\+|and|or|of|the|a|an|to|in|on|for|with|by|from|v|vs|versus)$",
            text,
        )
    )


def has_likely_truncated_hyphen_token(text: str) -> bool:
    tokens = text.split()
    if not tokens:
        return False
    last_token = tokens[-1].strip("\"'.,;:!?()[]{}")
    if last_token.endswith(("-", "–", "—")):
        return True
    match = re.search(r"[A-Za-z]+[-–—]([A-Za-z]{1,6})$", last_token)
    return bool(match)


def is_unstable_short_phrase(text: str) -> bool:
    tokens = text.split()
    if not tokens or text.endswith((".", "!", "?", '"', "'")):
        return False
    if len(tokens) <= 2:
        return True
    return len(tokens) <= 3 and tokens[0] in {"the", "a", "an"}


def is_single_choice_prediction(value: Any) -> bool:
    text = normalize_text(value).upper()
    return len(text) == 1 and text in {"A", "B", "C", "D", "E"}


def should_guard_single_choice(
    row: dict[str, Any],
    *,
    require_stable_single_choice: bool,
    stable_single_choice_max_block: int | None,
) -> bool:
    if not require_stable_single_choice:
        return False
    if not is_single_choice_prediction(row.get("scored_prediction")):
        return False
    if stable_single_choice_max_block is None:
        return True
    return int(row["block_number"]) <= stable_single_choice_max_block


def uncertainty_guard_passes(
    row: dict[str, Any],
    *,
    entropy_max: float | None,
    top_prob_min: float | None,
) -> bool:
    if entropy_max is not None and safe_float(row.get("token_entropy_mean")) > entropy_max:
        return False
    if top_prob_min is not None and safe_float(row.get("token_top_prob_mean")) < top_prob_min:
        return False
    return True


def safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def parse_args() -> RiskGatedHaltEvalConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-summary-path", default=RiskGatedHaltEvalConfig.eval_summary_path)
    parser.add_argument("--readiness-checkpoint-path", default=RiskGatedHaltEvalConfig.readiness_checkpoint_path)
    parser.add_argument("--risk-checkpoint-path", default=RiskGatedHaltEvalConfig.risk_checkpoint_path)
    parser.add_argument("--output-dir", default=RiskGatedHaltEvalConfig.output_dir)
    parser.add_argument("--eval-labels-dir", default=RiskGatedHaltEvalConfig.eval_labels_dir)
    parser.add_argument("--eval-tasks", default=RiskGatedHaltEvalConfig.eval_tasks)
    parser.add_argument("--calibration-tasks", default=RiskGatedHaltEvalConfig.calibration_tasks)
    parser.add_argument("--split", default=RiskGatedHaltEvalConfig.split)
    parser.add_argument("--calibration-split", default=RiskGatedHaltEvalConfig.calibration_split)
    parser.add_argument("--readiness-threshold", type=float, default=RiskGatedHaltEvalConfig.readiness_threshold)
    parser.add_argument(
        "--readiness-threshold-values",
        default=RiskGatedHaltEvalConfig.readiness_threshold_values,
        help="Comma-separated validation-swept readiness thresholds. Defaults to the hydrated threshold.",
    )
    parser.add_argument("--risk-threshold-start", type=float, default=RiskGatedHaltEvalConfig.risk_threshold_start)
    parser.add_argument("--risk-threshold-end", type=float, default=RiskGatedHaltEvalConfig.risk_threshold_end)
    parser.add_argument("--risk-threshold-step", type=float, default=RiskGatedHaltEvalConfig.risk_threshold_step)
    parser.add_argument(
        "--risk-threshold-selection-mode",
        default=RiskGatedHaltEvalConfig.risk_threshold_selection_mode,
        choices=["min_blocks", "first_saving"],
    )
    parser.add_argument(
        "--require-contentful-prediction",
        action="store_true",
        default=RiskGatedHaltEvalConfig.require_contentful_prediction,
    )
    parser.add_argument(
        "--require-fragment-complete-prediction",
        action="store_true",
        default=RiskGatedHaltEvalConfig.require_fragment_complete_prediction,
        help="Guard early non-stable answer fragments such as decimal prefixes, initials, and short bare words.",
    )
    parser.add_argument(
        "--require-stable-single-choice",
        action="store_true",
        default=RiskGatedHaltEvalConfig.require_stable_single_choice,
    )
    parser.add_argument(
        "--stable-single-choice-max-block",
        type=int,
        default=RiskGatedHaltEvalConfig.stable_single_choice_max_block,
    )
    parser.add_argument(
        "--stable-single-choice-guard-scopes",
        default=RiskGatedHaltEvalConfig.stable_single_choice_guard_scopes,
        help="Comma-separated validation-swept single-choice guard scopes: off,1,2,...,all.",
    )
    parser.add_argument("--entropy-max-values", default=RiskGatedHaltEvalConfig.entropy_max_values)
    parser.add_argument("--top-prob-min-values", default=RiskGatedHaltEvalConfig.top_prob_min_values)
    parser.add_argument(
        "--accuracy-drop-tolerance",
        type=float,
        default=RiskGatedHaltEvalConfig.accuracy_drop_tolerance,
    )
    parser.add_argument(
        "--require-zero-calibration-loss",
        action="store_true",
        default=RiskGatedHaltEvalConfig.require_zero_calibration_loss,
        help="Only select calibration rows with zero losses versus prediction-stability when available.",
    )
    parser.add_argument("--batch-size", type=int, default=RiskGatedHaltEvalConfig.batch_size)
    parser.add_argument("--device", default=RiskGatedHaltEvalConfig.device)
    parser.add_argument("--swanlab-mode", default=RiskGatedHaltEvalConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=RiskGatedHaltEvalConfig.experiment_name)
    args = parser.parse_args()
    return RiskGatedHaltEvalConfig(
        eval_summary_path=args.eval_summary_path,
        readiness_checkpoint_path=args.readiness_checkpoint_path,
        risk_checkpoint_path=args.risk_checkpoint_path,
        output_dir=args.output_dir,
        eval_labels_dir=args.eval_labels_dir,
        eval_tasks=args.eval_tasks,
        calibration_tasks=args.calibration_tasks,
        split=args.split,
        calibration_split=args.calibration_split,
        readiness_threshold=args.readiness_threshold,
        readiness_threshold_values=args.readiness_threshold_values,
        risk_threshold_start=args.risk_threshold_start,
        risk_threshold_end=args.risk_threshold_end,
        risk_threshold_step=args.risk_threshold_step,
        risk_threshold_selection_mode=args.risk_threshold_selection_mode,
        require_contentful_prediction=args.require_contentful_prediction,
        require_fragment_complete_prediction=args.require_fragment_complete_prediction,
        require_stable_single_choice=args.require_stable_single_choice,
        stable_single_choice_max_block=args.stable_single_choice_max_block,
        stable_single_choice_guard_scopes=args.stable_single_choice_guard_scopes,
        entropy_max_values=args.entropy_max_values,
        top_prob_min_values=args.top_prob_min_values,
        accuracy_drop_tolerance=args.accuracy_drop_tolerance,
        require_zero_calibration_loss=args.require_zero_calibration_loss,
        batch_size=args.batch_size,
        device=args.device,
        swanlab_mode=args.swanlab_mode,
        experiment_name=args.experiment_name,
    )


def main() -> None:
    summary = evaluate_risk_gated_halt(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
