"""Evaluate a LatentHaltStudent-v1 checkpoint as a student-only halt policy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from drla.scripts.train_cola_continuation_risk_model import prediction_stability_reference
from drla.scripts.train_cola_latent_halt_student import (
    PROCESS_FEATURE_FIELDS,
    LatentHaltStudent,
    LatentHaltStudentTrainConfig,
    binary_targets_for_config,
    build_student_tensors,
    normalize_text,
    split_indices,
)
from drla.scripts.train_cola_readiness_model import (
    ReadinessTrainConfig,
    add_derived_stability_features,
    load_training_rows,
    parse_tasks,
    resolve_device,
    stable_uniform,
)
from drla.tracking import finish_experiment, init_experiment, log_metrics
from drla.tracking import require_swanlab_disabled_for_non_training


def parse_float_grid(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--labels-dir", default=None)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--calibration-tasks", default=None)
    parser.add_argument("--eval-tasks", default=None)
    parser.add_argument("--eval-split", choices=["valid", "test", "all"], default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--readiness-thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--risk-thresholds", default="0.01,0.05,0.1,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--correctness-thresholds", default="")
    parser.add_argument("--completion-risk-thresholds", default="0.01,0.05,0.1,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--empty-answer-risk-thresholds", default="0.005,0.01,0.02,0.05,0.075,0.1,0.15,0.2,0.3,0.4,0.6,0.8,1.0")
    parser.add_argument("--answer-format-risk-thresholds", default="0.005,0.01,0.02,0.05,0.075,0.1,0.15,0.2,0.3,0.4,0.6,0.8,1.0")
    parser.add_argument("--answer-identity-stability-thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--contentful-thresholds", default="0.0,0.5,0.7,0.9")
    parser.add_argument("--accuracy-drop-tolerance", type=float, default=0.0)
    parser.add_argument("--require-zero-calibration-loss", action="store_true")
    parser.add_argument("--require-zero-calibration-mismatch", action="store_true")
    parser.add_argument("--max-calibration-mismatches", type=int, default=None)
    parser.add_argument("--max-calibration-mismatch-rate", type=float, default=None)
    parser.add_argument("--max-calibration-samples-per-task", type=int, default=None)
    parser.add_argument("--calibration-subsample-seed", default="20260525")
    parser.add_argument("--calibration-scope", choices=["pooled", "per_task"], default="pooled")
    parser.add_argument("--calibration-boundary-risk-penalty", type=float, default=0.0)
    parser.add_argument("--swanlab-mode", default="disabled")
    parser.add_argument("--experiment-name", default="official8-latent-halt-student-eval")
    return parser.parse_args()


def main() -> None:
    summary = evaluate_latent_halt_student(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def evaluate_latent_halt_student(args: argparse.Namespace) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        args.swanlab_mode,
        script_kind="LatentHaltStudent evaluation",
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_config = LatentHaltStudentTrainConfig(**checkpoint["config"])
    if args.labels_dir:
        train_config = replace(train_config, labels_dir=args.labels_dir)
    if args.tasks:
        parse_tasks(args.tasks)
        train_config = replace(train_config, tasks=args.tasks)
    calibration_tasks = args.calibration_tasks or train_config.tasks
    eval_tasks = args.eval_tasks or train_config.tasks
    parse_tasks(calibration_tasks)
    parse_tasks(eval_tasks)
    calibration_config = replace(train_config, tasks=calibration_tasks)
    eval_config = replace(train_config, tasks=eval_tasks)
    cross_eval = calibration_tasks != eval_tasks or args.eval_split == "all"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"

    device = resolve_device(args.device)
    calibration = prepare_bundle(calibration_config, checkpoint)
    model = load_model(checkpoint, calibration_config, calibration["tensors"], calibration["metadata"], device)
    calibration_scores = score_rows(
        model,
        calibration["tensors"],
        calibration["norm_process"],
        args.batch_size,
        device,
    )
    calibration_split_names = index_to_split(calibration["splits"], len(calibration["rows"]))
    calibration_score_path = output_dir / (
        "student_scores_calibration.jsonl" if cross_eval else "student_scores.jsonl"
    )
    write_scores(calibration_score_path, calibration["rows"], calibration_split_names, calibration_scores)

    if cross_eval:
        evaluation = prepare_bundle(eval_config, checkpoint)
        eval_scores = score_rows(
            model,
            evaluation["tensors"],
            evaluation["norm_process"],
            args.batch_size,
            device,
        )
        eval_split_names = index_to_split(evaluation["splits"], len(evaluation["rows"]))
        if args.eval_split == "all":
            eval_indices = list(range(len(evaluation["rows"])))
            eval_split_label = "all"
        else:
            eval_indices = evaluation["splits"][args.eval_split]
            eval_split_label = args.eval_split
        write_scores(output_dir / "student_scores_eval.jsonl", evaluation["rows"], eval_split_names, eval_scores)
    else:
        evaluation = calibration
        eval_scores = calibration_scores
        eval_indices = calibration["splits"][args.eval_split]
        eval_split_label = args.eval_split

    thresholds = {
        "readiness": parse_float_grid(args.readiness_thresholds),
        "risk": parse_float_grid(args.risk_thresholds),
        "contentful": parse_float_grid(args.contentful_thresholds),
    }
    if args.correctness_thresholds:
        thresholds["correctness"] = parse_float_grid(args.correctness_thresholds)
    if train_config.use_completion_risk:
        thresholds["completion_risk"] = parse_float_grid(args.completion_risk_thresholds)
    if train_config.use_empty_answer_risk:
        thresholds["empty_answer_risk"] = parse_float_grid(args.empty_answer_risk_thresholds)
    if train_config.use_answer_format_risk:
        thresholds["answer_format_risk"] = parse_float_grid(args.answer_format_risk_thresholds)
    if train_config.use_answer_identity_stability:
        thresholds["answer_identity_stability"] = parse_float_grid(args.answer_identity_stability_thresholds)
    calibration_valid_indices = calibration["splits"]["valid"]
    calibration_valid_sample_count_before = unique_sample_count(calibration["sample_keys"], calibration_valid_indices)
    if args.max_calibration_samples_per_task is not None:
        calibration_valid_indices = limit_indices_by_samples_per_task(
            calibration["rows"],
            calibration["sample_keys"],
            calibration_valid_indices,
            max_samples_per_task=args.max_calibration_samples_per_task,
            seed=args.calibration_subsample_seed,
        )
    calibration_valid_sample_count_after = unique_sample_count(calibration["sample_keys"], calibration_valid_indices)

    valid_sweep = sweep_thresholds(
        calibration["rows"],
        calibration["sample_keys"],
        calibration_scores,
        calibration_valid_indices,
        thresholds,
    )
    eval_sweep = sweep_thresholds(
        evaluation["rows"],
        evaluation["sample_keys"],
        eval_scores,
        eval_indices,
        thresholds,
    )
    write_csv(output_dir / "threshold_sweep_valid.csv", valid_sweep)
    write_csv(output_dir / ("threshold_sweep_eval.csv" if cross_eval else "threshold_sweep_test.csv"), eval_sweep)

    valid_baselines = baseline_metrics(calibration["rows"], calibration["sample_keys"], calibration_valid_indices)
    eval_baselines = baseline_metrics(evaluation["rows"], evaluation["sample_keys"], eval_indices)
    valid_task_sweeps = None
    valid_task_baselines = None
    if args.calibration_scope == "per_task":
        valid_task_sweeps = sweep_thresholds_by_task(
            calibration["rows"],
            calibration["sample_keys"],
            calibration_scores,
            calibration_valid_indices,
            thresholds,
        )
        valid_task_baselines = {
            task: baseline_metrics(calibration["rows"], calibration["sample_keys"], task_indices)
            for task, task_indices in group_indices_by_task(calibration["rows"], calibration_valid_indices).items()
        }
    selected_valid = select_valid_threshold(
        valid_sweep,
        fixed_final_accuracy=valid_baselines["fixed_final"]["accuracy"],
        prediction_stability_accuracy=valid_baselines["prediction_stability"]["accuracy"],
        tolerance=args.accuracy_drop_tolerance,
        require_zero_calibration_loss=args.require_zero_calibration_loss,
        require_zero_calibration_mismatch=args.require_zero_calibration_mismatch,
        max_calibration_mismatches=args.max_calibration_mismatches,
        max_calibration_mismatch_rate=args.max_calibration_mismatch_rate,
        boundary_risk_penalty=args.calibration_boundary_risk_penalty,
        calibration_scope=args.calibration_scope,
        task_sweeps=valid_task_sweeps,
        task_baselines=valid_task_baselines,
    )
    selected_eval = find_matching_row(eval_sweep, selected_valid)
    valid_decisions = policy_decisions(
        calibration["rows"],
        calibration["sample_keys"],
        calibration_scores,
        calibration_valid_indices,
        selected_valid,
    )
    eval_decisions = policy_decisions(
        evaluation["rows"],
        evaluation["sample_keys"],
        eval_scores,
        eval_indices,
        selected_valid,
    )
    write_jsonl(output_dir / "halt_decisions_valid.jsonl", valid_decisions)
    write_jsonl(output_dir / ("halt_decisions_eval.jsonl" if cross_eval else "halt_decisions_test.jsonl"), eval_decisions)

    swanlab_run_id = None
    if args.swanlab_mode != "disabled":
        run = init_experiment(
            stage="cola-latent-halt-student-eval",
            experiment_name=args.experiment_name,
            description="Student-only LatentHaltStudent-v1 threshold calibration and accuracy-cost evaluation.",
            config={
                "checkpoint": args.checkpoint,
                "output_dir": args.output_dir,
                "train_config": asdict(train_config),
                "calibration_tasks": calibration_tasks,
                "eval_tasks": eval_tasks,
                "eval_split": eval_split_label,
                "thresholds": thresholds,
                "accuracy_drop_tolerance": args.accuracy_drop_tolerance,
                "require_zero_calibration_loss": args.require_zero_calibration_loss,
                "require_zero_calibration_mismatch": args.require_zero_calibration_mismatch,
                "calibration_scope": args.calibration_scope,
                "online_input_policy": checkpoint["metadata"]["online_input_policy"],
                "decision_policy": (
                    "earliest block with student readiness >= threshold, "
                    "student prediction_change <= risk threshold, student contentful >= threshold, "
                    "optional correctness >= threshold, "
                    "optional completion_risk <= threshold, "
                    "optional empty_answer_risk <= threshold, "
                    "optional answer_format_risk <= threshold, "
                    "and optional answer_identity_stability >= threshold; else final block"
                ),
            },
            mode=args.swanlab_mode,
            tags=["cola", "official-benchmark", "latent-halt-student", "student-only-eval"],
        )
        try:
            log_metrics(flatten_metrics("valid_fixed_final", valid_baselines["fixed_final"]))
            log_metrics(flatten_metrics("valid_prediction_stability", valid_baselines["prediction_stability"]))
            log_metrics(flatten_metrics("valid_student", selected_valid))
            log_metrics(flatten_metrics("eval_fixed_final", eval_baselines["fixed_final"]))
            log_metrics(flatten_metrics("eval_prediction_stability", eval_baselines["prediction_stability"]))
            log_metrics(flatten_metrics("eval_student", selected_eval))
            swanlab_run_id = getattr(run, "id", None)
        finally:
            finish_experiment()

    artifacts = {
        "student_scores_calibration": str(calibration_score_path),
        "threshold_sweep_valid": str(output_dir / "threshold_sweep_valid.csv"),
        "halt_decisions_valid": str(output_dir / "halt_decisions_valid.jsonl"),
    }
    if cross_eval:
        artifacts.update(
            {
                "student_scores_eval": str(output_dir / "student_scores_eval.jsonl"),
                "threshold_sweep_eval": str(output_dir / "threshold_sweep_eval.csv"),
                "halt_decisions_eval": str(output_dir / "halt_decisions_eval.jsonl"),
            }
        )
    else:
        artifacts.update(
            {
                "student_scores": str(calibration_score_path),
                "threshold_sweep_test": str(output_dir / "threshold_sweep_test.csv"),
                "halt_decisions_test": str(output_dir / "halt_decisions_test.jsonl"),
            }
        )

    summary = {
        "created_at": int(time.time()),
        "checkpoint": args.checkpoint,
        "output_dir": args.output_dir,
        "swanlab_mode": args.swanlab_mode,
        "swanlab_run_id": swanlab_run_id,
        "train_config": asdict(train_config),
        "calibration_tasks": calibration_tasks,
        "eval_tasks": eval_tasks,
        "eval_split": eval_split_label,
        "online_input_policy": checkpoint["metadata"]["online_input_policy"],
        "decision_policy": (
            "student-only threshold policy over readiness, prediction_change risk, contentful, "
            "optional correctness, "
            "optional completion_risk, optional empty_answer_risk, optional answer_format_risk, "
            "and optional answer_identity_stability probabilities"
        ),
        "thresholds": thresholds,
        "calibration_policy": {
            "accuracy_drop_tolerance": args.accuracy_drop_tolerance,
            "require_zero_calibration_loss": args.require_zero_calibration_loss,
            "require_zero_calibration_mismatch": args.require_zero_calibration_mismatch,
            "max_calibration_mismatches": args.max_calibration_mismatches,
            "max_calibration_mismatch_rate": args.max_calibration_mismatch_rate,
            "max_calibration_samples_per_task": args.max_calibration_samples_per_task,
            "calibration_subsample_seed": args.calibration_subsample_seed,
            "calibration_boundary_risk_penalty": args.calibration_boundary_risk_penalty,
            "valid_rows_before_subsample": len(calibration["splits"]["valid"]),
            "valid_rows_after_subsample": len(calibration_valid_indices),
            "valid_samples_before_subsample": calibration_valid_sample_count_before,
            "valid_samples_after_subsample": calibration_valid_sample_count_after,
            "calibration_scope": args.calibration_scope,
        },
        "valid_baselines": valid_baselines,
        "test_baselines": eval_baselines,
        "eval_baselines": eval_baselines,
        "selected_valid": selected_valid,
        "selected_test": selected_eval,
        "selected_eval": selected_eval,
        "artifacts": artifacts,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def prepare_bundle(
    config: LatentHaltStudentTrainConfig,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    rows = load_training_rows(
        ReadinessTrainConfig(labels_dir=config.labels_dir, tasks=config.tasks, seed=config.seed)
    )
    ordered_rows, ordered_sample_keys = ordered_student_rows(rows)
    tensor_config = replace(config, readiness_target_mode="oracle_frontier", teacher_decisions_jsonl="")
    tensors, metadata = build_student_tensors(rows, tensor_config)
    if ordered_sample_keys != metadata["sample_keys"]:
        raise RuntimeError("ordered row metadata does not match tensor order")
    splits = split_indices(metadata["sample_keys"], config)
    norm_process = normalize_process_features(tensors, checkpoint["norm_stats"])
    return {
        "rows": ordered_rows,
        "sample_keys": ordered_sample_keys,
        "tensors": tensors,
        "metadata": metadata,
        "splits": splits,
        "norm_process": norm_process,
    }


def load_model(
    checkpoint: dict[str, Any],
    config: LatentHaltStudentTrainConfig,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    device: torch.device,
) -> LatentHaltStudent:
    model = LatentHaltStudent(
        latent_dim=tensors["latent_blocks"].shape[-1],
        process_dim=tensors["process_features"].shape[-1],
        max_blocks=tensors["latent_blocks"].shape[1],
        block_size=tensors["latent_blocks"].shape[2],
        d_model=config.d_model,
        attention_heads=config.attention_heads,
        inter_layers=config.inter_layers,
        dropout=config.dropout,
        pooling_mode=config.pooling_mode,
        task_conditioning=config.task_conditioning,
        process_interaction_mode=config.process_interaction_mode,
        readout_context_mode=config.readout_context_mode,
        task_count=len(metadata["task_to_idx"]),
        binary_targets=binary_targets_for_config(config),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def ordered_student_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    add_derived_stability_features(rows)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(f"{row['task']}::{row['sample_id']}", []).append(row)
    ordered_rows: list[dict[str, Any]] = []
    sample_keys: list[str] = []
    for sample_key, sample_rows in grouped.items():
        sample_rows.sort(key=lambda item: int(item["block_index"]))
        for row in sample_rows:
            ordered_rows.append(row)
            sample_keys.append(sample_key)
    return ordered_rows, sample_keys


def normalize_process_features(tensors: dict[str, torch.Tensor], norm_stats: dict[str, torch.Tensor]) -> torch.Tensor:
    mean = norm_stats["process_mean"].view(1, 1, -1)
    std = norm_stats["process_std"].view(1, 1, -1).clamp_min(1e-6)
    norm = (tensors["process_features"] - mean) / std
    return norm.masked_fill(~tensors["block_mask"].unsqueeze(-1), 0.0)


@torch.no_grad()
def score_rows(
    model: LatentHaltStudent,
    tensors: dict[str, torch.Tensor],
    norm_process: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    dataset = TensorDataset(
        tensors["latent_blocks"],
        norm_process,
        tensors["block_mask"],
        tensors["task_idx"],
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    binary_targets = [name for name in model.query_names if name != "future_gain"]
    outputs: dict[str, list[torch.Tensor]] = {name: [] for name in [*binary_targets, "future_gain"]}
    for latent_blocks, process_features, block_mask, task_idx in loader:
        result = model(
            latent_blocks.to(device),
            process_features.to(device),
            block_mask.to(device),
            task_idx.to(device),
        )
        for name in binary_targets:
            outputs[name].append(torch.sigmoid(result[name]).cpu())
        outputs["future_gain"].append(result["future_gain"].cpu())
    return {name: torch.cat(values) for name, values in outputs.items()}


def index_to_split(splits: dict[str, list[int]], row_count: int) -> list[str]:
    result = ["unknown"] * row_count
    for split, indices in splits.items():
        for idx in indices:
            result[idx] = split
    return result


def unique_sample_count(sample_keys: list[str], indices: list[int]) -> int:
    return len({sample_keys[idx] for idx in indices})


def limit_indices_by_samples_per_task(
    rows: list[dict[str, Any]],
    sample_keys: list[str],
    indices: list[int],
    *,
    max_samples_per_task: int,
    seed: str,
) -> list[int]:
    if max_samples_per_task < 1:
        raise ValueError("max_calibration_samples_per_task must be >= 1")
    by_task: dict[str, set[str]] = {}
    for idx in indices:
        by_task.setdefault(str(rows[idx]["task"]), set()).add(sample_keys[idx])
    selected_keys: set[str] = set()
    for task, keys in by_task.items():
        ordered = sorted(keys, key=lambda key: stable_uniform(f"{seed}:{task}:{key}"))
        selected_keys.update(ordered[:max_samples_per_task])
    return [idx for idx in indices if sample_keys[idx] in selected_keys]


def write_scores(path: Path, rows: list[dict[str, Any]], splits: list[str], scores: dict[str, torch.Tensor]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            payload = {
                "task": row["task"],
                "sample_id": row["sample_id"],
                "block_number": row["block_number"],
                "split": splits[idx],
                "official_correct": bool(row.get("official_correct")),
                "scored_prediction": row.get("scored_prediction"),
            }
            for name, tensor in scores.items():
                payload[f"student_{name}"] = float(tensor[idx].item())
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def sweep_thresholds(
    rows: list[dict[str, Any]],
    sample_keys: list[str],
    scores: dict[str, torch.Tensor],
    indices: list[int],
    thresholds: dict[str, list[float]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    arrays = build_policy_arrays(rows, sample_keys, scores, indices)
    correctness_grid = thresholds.get("correctness", [None])
    completion_grid = thresholds.get("completion_risk", [None])
    empty_answer_grid = thresholds.get("empty_answer_risk", [None])
    answer_format_grid = thresholds.get("answer_format_risk", [None])
    answer_identity_grid = thresholds.get("answer_identity_stability", [None])
    for readiness_threshold in thresholds["readiness"]:
        for risk_threshold in thresholds["risk"]:
            for contentful_threshold in thresholds["contentful"]:
                for correctness_threshold in correctness_grid:
                    for completion_risk_threshold in completion_grid:
                        for empty_answer_risk_threshold in empty_answer_grid:
                            for answer_format_risk_threshold in answer_format_grid:
                                for answer_identity_threshold in answer_identity_grid:
                                    policy_thresholds = {
                                        "readiness_threshold": readiness_threshold,
                                        "risk_threshold": risk_threshold,
                                        "contentful_threshold": contentful_threshold,
                                    }
                                    if correctness_threshold is not None:
                                        policy_thresholds["correctness_threshold"] = correctness_threshold
                                    if completion_risk_threshold is not None:
                                        policy_thresholds["completion_risk_threshold"] = completion_risk_threshold
                                    if empty_answer_risk_threshold is not None:
                                        policy_thresholds["empty_answer_risk_threshold"] = empty_answer_risk_threshold
                                    if answer_format_risk_threshold is not None:
                                        policy_thresholds["answer_format_risk_threshold"] = answer_format_risk_threshold
                                    if answer_identity_threshold is not None:
                                        policy_thresholds["answer_identity_stability_threshold"] = answer_identity_threshold
                                    result.append(policy_metrics_from_arrays(arrays, policy_thresholds))
    return result


def sweep_thresholds_by_task(
    rows: list[dict[str, Any]],
    sample_keys: list[str],
    scores: dict[str, torch.Tensor],
    indices: list[int],
    thresholds: dict[str, list[float]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        task: sweep_thresholds(rows, sample_keys, scores, task_indices, thresholds)
        for task, task_indices in group_indices_by_task(rows, indices).items()
    }


def build_policy_arrays(
    rows: list[dict[str, Any]],
    sample_keys: list[str],
    scores: dict[str, torch.Tensor],
    indices: list[int],
) -> dict[str, Any]:
    grouped = sorted_group_indices(rows, sample_keys, indices)
    sample_count = len(grouped)
    max_len = max(len(row_indices) for _, row_indices in grouped)
    mask = torch.zeros(sample_count, max_len, dtype=torch.bool)
    block_number = torch.zeros(sample_count, max_len, dtype=torch.float32)
    correct = torch.zeros(sample_count, max_len, dtype=torch.bool)
    prediction_id = torch.zeros(sample_count, max_len, dtype=torch.long)
    final_pos = torch.zeros(sample_count, dtype=torch.long)
    stable_pos = torch.zeros(sample_count, dtype=torch.long)
    score_mats = {
        name: torch.zeros(sample_count, max_len, dtype=torch.float32)
        for name in [
            "readiness",
            "correctness",
            "prediction_change",
            "contentful",
            "completion_risk",
            "empty_answer_risk",
            "answer_format_risk",
            "answer_identity_stability",
        ]
        if name in scores
    }
    prediction_to_id: dict[str, int] = {"": 0}

    def encode_prediction(value: Any) -> int:
        prediction = normalize_text(value)
        if prediction not in prediction_to_id:
            prediction_to_id[prediction] = len(prediction_to_id)
        return prediction_to_id[prediction]

    for sample_pos, (_, row_indices) in enumerate(grouped):
        stable_idx = prediction_stability_index(rows, row_indices)
        stable_pos[sample_pos] = row_indices.index(stable_idx)
        final_pos[sample_pos] = len(row_indices) - 1
        for block_pos, row_idx in enumerate(row_indices):
            row = rows[row_idx]
            mask[sample_pos, block_pos] = True
            block_number[sample_pos, block_pos] = float(row["block_number"])
            correct[sample_pos, block_pos] = bool(row.get("official_correct"))
            prediction_id[sample_pos, block_pos] = encode_prediction(row.get("scored_prediction"))
            for name, tensor in score_mats.items():
                tensor[sample_pos, block_pos] = float(scores[name][row_idx].item())

    return {
        "mask": mask,
        "block_number": block_number,
        "correct": correct,
        "prediction_id": prediction_id,
        "final_pos": final_pos,
        "stable_pos": stable_pos,
        "scores": score_mats,
    }


def policy_metrics_from_arrays(arrays: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    readiness_threshold = float(thresholds["readiness_threshold"])
    risk_threshold = float(thresholds["risk_threshold"])
    contentful_threshold = float(thresholds["contentful_threshold"])
    correctness_threshold = thresholds.get("correctness_threshold")
    completion_risk_threshold = thresholds.get("completion_risk_threshold")
    empty_answer_risk_threshold = thresholds.get("empty_answer_risk_threshold")
    answer_format_risk_threshold = thresholds.get("answer_format_risk_threshold")
    answer_identity_stability_threshold = thresholds.get("answer_identity_stability_threshold")
    scores = arrays["scores"]
    candidates = (
        arrays["mask"]
        & (scores["readiness"] >= readiness_threshold)
        & (scores["prediction_change"] <= risk_threshold)
        & (scores["contentful"] >= contentful_threshold)
    )
    if correctness_threshold is not None and "correctness" in scores:
        correctness_threshold = float(correctness_threshold)
        candidates = candidates & (scores["correctness"] >= correctness_threshold)
    else:
        correctness_threshold = None
    if completion_risk_threshold is not None and "completion_risk" in scores:
        completion_risk_threshold = float(completion_risk_threshold)
        candidates = candidates & (scores["completion_risk"] <= completion_risk_threshold)
    else:
        completion_risk_threshold = None
    if empty_answer_risk_threshold is not None and "empty_answer_risk" in scores:
        empty_answer_risk_threshold = float(empty_answer_risk_threshold)
        candidates = candidates & (scores["empty_answer_risk"] <= empty_answer_risk_threshold)
    else:
        empty_answer_risk_threshold = None
    if answer_format_risk_threshold is not None and "answer_format_risk" in scores:
        answer_format_risk_threshold = float(answer_format_risk_threshold)
        candidates = candidates & (scores["answer_format_risk"] <= answer_format_risk_threshold)
    else:
        answer_format_risk_threshold = None
    if answer_identity_stability_threshold is not None and "answer_identity_stability" in scores:
        answer_identity_stability_threshold = float(answer_identity_stability_threshold)
        candidates = candidates & (scores["answer_identity_stability"] >= answer_identity_stability_threshold)
    else:
        answer_identity_stability_threshold = None

    has_candidate = candidates.any(dim=1)
    first_candidate = torch.argmax(candidates.to(torch.int64), dim=1)
    selected_pos = torch.where(has_candidate, first_candidate, arrays["final_pos"])
    row_idx = torch.arange(selected_pos.numel())
    final_pos = arrays["final_pos"]
    stable_pos = arrays["stable_pos"]

    selected_correct = arrays["correct"][row_idx, selected_pos]
    final_correct = arrays["correct"][row_idx, final_pos]
    stable_correct = arrays["correct"][row_idx, stable_pos]
    selected_prediction = arrays["prediction_id"][row_idx, selected_pos]
    final_prediction = arrays["prediction_id"][row_idx, final_pos]
    stable_prediction = arrays["prediction_id"][row_idx, stable_pos]
    selected_blocks = arrays["block_number"][row_idx, selected_pos]
    final_blocks = arrays["block_number"][row_idx, final_pos]
    count = int(selected_pos.numel())
    selected_correct_count = int(selected_correct.sum().item())
    final_correct_count = int(final_correct.sum().item())
    stable_correct_count = int(stable_correct.sum().item())
    accuracy = selected_correct_count / count
    final_accuracy = final_correct_count / count
    stable_accuracy = stable_correct_count / count
    avg_blocks = float(selected_blocks.mean().item())
    fixed_final_avg_blocks = float(final_blocks.mean().item())
    loss_vs_final = final_correct & ~selected_correct
    gain_vs_final = selected_correct & ~final_correct
    loss_vs_stability = stable_correct & ~selected_correct
    gain_vs_stability = selected_correct & ~stable_correct
    mismatch_vs_final = selected_prediction != final_prediction
    mismatch_vs_stability = selected_prediction != stable_prediction
    mismatch_final_count = int(mismatch_vs_final.sum().item())
    mismatch_stability_count = int(mismatch_vs_stability.sum().item())
    return {
        "readiness_threshold": readiness_threshold,
        "risk_threshold": risk_threshold,
        "contentful_threshold": contentful_threshold,
        "correctness_threshold": correctness_threshold,
        "completion_risk_threshold": completion_risk_threshold,
        "empty_answer_risk_threshold": empty_answer_risk_threshold,
        "answer_format_risk_threshold": answer_format_risk_threshold,
        "answer_identity_stability_threshold": answer_identity_stability_threshold,
        "boundary_risk_slack": boundary_risk_slack(
            {
                "risk_threshold": risk_threshold,
                "completion_risk_threshold": completion_risk_threshold,
                "empty_answer_risk_threshold": empty_answer_risk_threshold,
                "answer_format_risk_threshold": answer_format_risk_threshold,
                "answer_identity_stability_threshold": answer_identity_stability_threshold,
            }
        ),
        "num_samples": count,
        "accuracy": accuracy,
        "fixed_final_accuracy": final_accuracy,
        "prediction_stability_accuracy": stable_accuracy,
        "accuracy_drop_vs_final": final_accuracy - accuracy,
        "accuracy_drop_vs_prediction_stability": stable_accuracy - accuracy,
        "avg_blocks": avg_blocks,
        "fixed_final_avg_blocks": fixed_final_avg_blocks,
        "block_saving_vs_final": fixed_final_avg_blocks - avg_blocks,
        "block_saving_fraction_vs_final": (fixed_final_avg_blocks - avg_blocks) / max(fixed_final_avg_blocks, 1e-9),
        "losses_vs_final": int(loss_vs_final.sum().item()),
        "gains_vs_final": int(gain_vs_final.sum().item()),
        "losses_vs_prediction_stability": int(loss_vs_stability.sum().item()),
        "gains_vs_prediction_stability": int(gain_vs_stability.sum().item()),
        "prediction_mismatch_vs_final": mismatch_final_count,
        "prediction_mismatch_rate_vs_final": mismatch_final_count / count,
        "prediction_mismatch_vs_prediction_stability": mismatch_stability_count,
        "prediction_mismatch_rate_vs_prediction_stability": mismatch_stability_count / count,
    }


def baseline_metrics(
    rows: list[dict[str, Any]],
    sample_keys: list[str],
    indices: list[int],
) -> dict[str, dict[str, Any]]:
    fixed_decisions = []
    stable_decisions = []
    for sample_key, row_indices in sorted_group_indices(rows, sample_keys, indices):
        final_idx = row_indices[-1]
        stable_idx = prediction_stability_index(rows, row_indices)
        fixed_decisions.append(decision_payload(rows, sample_key, final_idx, final_idx, stable_idx))
        stable_decisions.append(decision_payload(rows, sample_key, stable_idx, final_idx, stable_idx))
    return {
        "fixed_final": policy_metrics(fixed_decisions),
        "prediction_stability": policy_metrics(stable_decisions),
    }


def policy_decisions(
    rows: list[dict[str, Any]],
    sample_keys: list[str],
    scores: dict[str, torch.Tensor],
    indices: list[int],
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped = sorted_group_indices(rows, sample_keys, indices)
    return policy_decisions_for_groups(rows, grouped, scores, thresholds)


def policy_decisions_for_groups(
    rows: list[dict[str, Any]],
    grouped: list[tuple[str, list[int]]],
    scores: dict[str, torch.Tensor],
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    readiness_threshold = float(thresholds["readiness_threshold"])
    risk_threshold = float(thresholds["risk_threshold"])
    contentful_threshold = float(thresholds["contentful_threshold"])
    correctness_threshold = thresholds.get("correctness_threshold")
    completion_risk_threshold = thresholds.get("completion_risk_threshold")
    empty_answer_risk_threshold = thresholds.get("empty_answer_risk_threshold")
    answer_format_risk_threshold = thresholds.get("answer_format_risk_threshold")
    answer_identity_stability_threshold = thresholds.get("answer_identity_stability_threshold")
    use_correctness = correctness_threshold is not None and "correctness" in scores
    use_completion_risk = completion_risk_threshold is not None and "completion_risk" in scores
    use_empty_answer_risk = empty_answer_risk_threshold is not None and "empty_answer_risk" in scores
    use_answer_format_risk = answer_format_risk_threshold is not None and "answer_format_risk" in scores
    use_answer_identity_stability = (
        answer_identity_stability_threshold is not None and "answer_identity_stability" in scores
    )
    if correctness_threshold is not None:
        correctness_threshold = float(correctness_threshold)
    if completion_risk_threshold is not None:
        completion_risk_threshold = float(completion_risk_threshold)
    if empty_answer_risk_threshold is not None:
        empty_answer_risk_threshold = float(empty_answer_risk_threshold)
    if answer_format_risk_threshold is not None:
        answer_format_risk_threshold = float(answer_format_risk_threshold)
    if answer_identity_stability_threshold is not None:
        answer_identity_stability_threshold = float(answer_identity_stability_threshold)
    for sample_key, row_indices in grouped:
        final_idx = row_indices[-1]
        stable_idx = prediction_stability_index(rows, row_indices)
        selected_idx = final_idx
        for idx in row_indices:
            if (
                float(scores["readiness"][idx].item()) >= readiness_threshold
                and float(scores["prediction_change"][idx].item()) <= risk_threshold
                and float(scores["contentful"][idx].item()) >= contentful_threshold
                and (not use_correctness or float(scores["correctness"][idx].item()) >= correctness_threshold)
                and (
                    not use_completion_risk
                    or float(scores["completion_risk"][idx].item()) <= completion_risk_threshold
                )
                and (
                    not use_empty_answer_risk
                    or float(scores["empty_answer_risk"][idx].item()) <= empty_answer_risk_threshold
                )
                and (
                    not use_answer_format_risk
                    or float(scores["answer_format_risk"][idx].item()) <= answer_format_risk_threshold
                )
                and (
                    not use_answer_identity_stability
                    or float(scores["answer_identity_stability"][idx].item()) >= answer_identity_stability_threshold
                )
            ):
                selected_idx = idx
                break
        payload = decision_payload(rows, sample_key, selected_idx, final_idx, stable_idx)
        payload.update(
            {
                "readiness_threshold": readiness_threshold,
                "risk_threshold": risk_threshold,
                "contentful_threshold": contentful_threshold,
                "correctness_threshold": correctness_threshold if use_correctness else None,
                "completion_risk_threshold": completion_risk_threshold if use_completion_risk else None,
                "empty_answer_risk_threshold": empty_answer_risk_threshold if use_empty_answer_risk else None,
                "answer_format_risk_threshold": answer_format_risk_threshold if use_answer_format_risk else None,
                "answer_identity_stability_threshold": (
                    answer_identity_stability_threshold if use_answer_identity_stability else None
                ),
                "student_readiness": float(scores["readiness"][selected_idx].item()),
                "student_prediction_change": float(scores["prediction_change"][selected_idx].item()),
                "student_contentful": float(scores["contentful"][selected_idx].item()),
                "student_correctness": float(scores["correctness"][selected_idx].item()),
                "student_future_gain": float(scores["future_gain"][selected_idx].item()),
            }
        )
        if use_completion_risk:
            payload["student_completion_risk"] = float(scores["completion_risk"][selected_idx].item())
        if use_empty_answer_risk:
            payload["student_empty_answer_risk"] = float(scores["empty_answer_risk"][selected_idx].item())
        if use_answer_format_risk:
            payload["student_answer_format_risk"] = float(scores["answer_format_risk"][selected_idx].item())
        if use_answer_identity_stability:
            payload["student_answer_identity_stability"] = float(
                scores["answer_identity_stability"][selected_idx].item()
            )
        decisions.append(payload)
    return decisions


def group_indices(sample_keys: list[str], indices: list[int]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for idx in indices:
        grouped.setdefault(sample_keys[idx], []).append(idx)
    return grouped


def sorted_group_indices(
    rows: list[dict[str, Any]],
    sample_keys: list[str],
    indices: list[int],
) -> list[tuple[str, list[int]]]:
    grouped = group_indices(sample_keys, indices)
    result = []
    for sample_key, row_indices in grouped.items():
        row_indices.sort(key=lambda idx: int(rows[idx]["block_index"]))
        result.append((sample_key, row_indices))
    return result


def group_indices_by_task(rows: list[dict[str, Any]], indices: list[int]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for idx in indices:
        grouped.setdefault(str(rows[idx]["task"]), []).append(idx)
    return grouped


def prediction_stability_index(rows: list[dict[str, Any]], row_indices: list[int]) -> int:
    previous = ""
    streak = 0
    for idx in row_indices:
        prediction = normalize_text(rows[idx].get("scored_prediction"))
        if prediction and prediction == previous:
            streak += 1
        else:
            previous = prediction
            streak = 1
        if prediction and streak >= 2:
            return idx
    return row_indices[-1]


def decision_payload(
    rows: list[dict[str, Any]],
    sample_key: str,
    selected_idx: int,
    final_idx: int,
    stable_idx: int,
) -> dict[str, Any]:
    selected = rows[selected_idx]
    final = rows[final_idx]
    stable = rows[stable_idx]
    selected_correct = bool(selected.get("official_correct"))
    final_correct = bool(final.get("official_correct"))
    stable_correct = bool(stable.get("official_correct"))
    return {
        "sample_key": sample_key,
        "task": selected["task"],
        "sample_id": selected["sample_id"],
        "selected_block": int(selected["block_number"]),
        "final_block": int(final["block_number"]),
        "prediction_stability_block": int(stable["block_number"]),
        "selected_correct": selected_correct,
        "final_correct": final_correct,
        "prediction_stability_correct": stable_correct,
        "loss_vs_final": bool(final_correct and not selected_correct),
        "gain_vs_final": bool(selected_correct and not final_correct),
        "loss_vs_prediction_stability": bool(stable_correct and not selected_correct),
        "gain_vs_prediction_stability": bool(selected_correct and not stable_correct),
        "selected_prediction": selected.get("scored_prediction"),
        "final_prediction": final.get("scored_prediction"),
        "prediction_stability_prediction": stable.get("scored_prediction"),
    }


def policy_metrics(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(decisions)
    if count == 0:
        raise ValueError("empty decision set")
    accuracy = sum(1 for item in decisions if item["selected_correct"]) / count
    final_accuracy = sum(1 for item in decisions if item["final_correct"]) / count
    stable_accuracy = sum(1 for item in decisions if item["prediction_stability_correct"]) / count
    avg_blocks = sum(float(item["selected_block"]) for item in decisions) / count
    final_blocks = sum(float(item["final_block"]) for item in decisions) / count
    prediction_mismatch_vs_final = sum(
        1
        for item in decisions
        if normalize_text(item.get("selected_prediction")) != normalize_text(item.get("final_prediction"))
    )
    prediction_mismatch_vs_stability = sum(
        1
        for item in decisions
        if normalize_text(item.get("selected_prediction"))
        != normalize_text(item.get("prediction_stability_prediction"))
    )
    return {
        "readiness_threshold": decisions[0].get("readiness_threshold"),
        "risk_threshold": decisions[0].get("risk_threshold"),
        "contentful_threshold": decisions[0].get("contentful_threshold"),
        "correctness_threshold": decisions[0].get("correctness_threshold"),
        "completion_risk_threshold": decisions[0].get("completion_risk_threshold"),
        "empty_answer_risk_threshold": decisions[0].get("empty_answer_risk_threshold"),
        "answer_format_risk_threshold": decisions[0].get("answer_format_risk_threshold"),
        "answer_identity_stability_threshold": decisions[0].get("answer_identity_stability_threshold"),
        "num_samples": count,
        "accuracy": accuracy,
        "fixed_final_accuracy": final_accuracy,
        "prediction_stability_accuracy": stable_accuracy,
        "accuracy_drop_vs_final": final_accuracy - accuracy,
        "accuracy_drop_vs_prediction_stability": stable_accuracy - accuracy,
        "avg_blocks": avg_blocks,
        "fixed_final_avg_blocks": final_blocks,
        "block_saving_vs_final": final_blocks - avg_blocks,
        "block_saving_fraction_vs_final": (final_blocks - avg_blocks) / max(final_blocks, 1e-9),
        "losses_vs_final": sum(1 for item in decisions if item["loss_vs_final"]),
        "gains_vs_final": sum(1 for item in decisions if item["gain_vs_final"]),
        "losses_vs_prediction_stability": sum(
            1 for item in decisions if item["loss_vs_prediction_stability"]
        ),
        "gains_vs_prediction_stability": sum(
            1 for item in decisions if item["gain_vs_prediction_stability"]
        ),
        "prediction_mismatch_vs_final": prediction_mismatch_vs_final,
        "prediction_mismatch_rate_vs_final": prediction_mismatch_vs_final / count,
        "prediction_mismatch_vs_prediction_stability": prediction_mismatch_vs_stability,
        "prediction_mismatch_rate_vs_prediction_stability": prediction_mismatch_vs_stability / count,
    }


def select_valid_threshold(
    sweep: list[dict[str, Any]],
    *,
    fixed_final_accuracy: float,
    prediction_stability_accuracy: float,
    tolerance: float,
    require_zero_calibration_loss: bool,
    require_zero_calibration_mismatch: bool,
    max_calibration_mismatches: int | None,
    max_calibration_mismatch_rate: float | None,
    boundary_risk_penalty: float = 0.0,
    calibration_scope: str = "pooled",
    task_sweeps: dict[str, list[dict[str, Any]]] | None = None,
    task_baselines: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if boundary_risk_penalty < 0:
        raise ValueError("calibration_boundary_risk_penalty must be >= 0")
    task_lookup = build_task_threshold_lookup(task_sweeps) if task_sweeps else {}
    score_label = "min_selection_score" if boundary_risk_penalty > 0 else "min_blocks"

    def selection_key(row: dict[str, Any]) -> tuple[float, float, float]:
        slack = boundary_risk_slack(row)
        score = row["avg_blocks"] + boundary_risk_penalty * slack
        return (score, row["avg_blocks"], -row["accuracy"])

    def finalize_selected(row: dict[str, Any], note: str) -> dict[str, Any]:
        selected = row
        selected["selection_note"] = note
        selected["calibration_scope"] = calibration_scope
        selected["calibration_boundary_risk_penalty"] = boundary_risk_penalty
        selected["calibration_selection_score"] = (
            selected["avg_blocks"] + boundary_risk_penalty * boundary_risk_slack(selected)
        )
        return selected

    def row_meets_mismatch_cap(row: dict[str, Any]) -> bool:
        if max_calibration_mismatches is not None and (
            row["prediction_mismatch_vs_final"] > max_calibration_mismatches
            or row["prediction_mismatch_vs_prediction_stability"] > max_calibration_mismatches
        ):
            return False
        if max_calibration_mismatch_rate is not None and (
            row["prediction_mismatch_rate_vs_final"] > max_calibration_mismatch_rate
            or row["prediction_mismatch_rate_vs_prediction_stability"] > max_calibration_mismatch_rate
        ):
            return False
        return True

    def row_meets_constraints(row: dict[str, Any], *, require_loss_zero: bool, require_mismatch_zero: bool) -> bool:
        if row["accuracy"] < fixed_final_accuracy - tolerance:
            return False
        if row["accuracy"] < prediction_stability_accuracy - tolerance:
            return False
        if require_loss_zero and (row["losses_vs_final"] != 0 or row["losses_vs_prediction_stability"] != 0):
            return False
        if require_mismatch_zero and (
            row["prediction_mismatch_vs_final"] != 0
            or row["prediction_mismatch_vs_prediction_stability"] != 0
        ):
            return False
        if not row_meets_mismatch_cap(row):
            return False
        if calibration_scope != "per_task":
            return True
        if not task_baselines or not task_lookup:
            raise ValueError("per_task calibration requires task sweeps and task baselines")
        key = threshold_key(row)
        for task, baselines in task_baselines.items():
            task_row = task_lookup[task].get(key)
            if task_row is None:
                raise RuntimeError(f"threshold {key} missing from task sweep for {task}")
            if task_row["accuracy"] < baselines["fixed_final"]["accuracy"] - tolerance:
                return False
            if task_row["accuracy"] < baselines["prediction_stability"]["accuracy"] - tolerance:
                return False
            if require_loss_zero and (
                task_row["losses_vs_final"] != 0 or task_row["losses_vs_prediction_stability"] != 0
            ):
                return False
            if require_mismatch_zero and (
                task_row["prediction_mismatch_vs_final"] != 0
                or task_row["prediction_mismatch_vs_prediction_stability"] != 0
            ):
                return False
            if not row_meets_mismatch_cap(task_row):
                return False
        return True

    candidates = [
        row
        for row in sweep
        if row_meets_constraints(row, require_loss_zero=False, require_mismatch_zero=False)
    ]
    if require_zero_calibration_loss:
        zero_loss = [
            row
            for row in sweep
            if row_meets_constraints(row, require_loss_zero=True, require_mismatch_zero=False)
        ]
        if require_zero_calibration_mismatch:
            zero_loss_zero_mismatch = [
                row
                for row in sweep
                if row_meets_constraints(row, require_loss_zero=True, require_mismatch_zero=True)
            ]
            if zero_loss_zero_mismatch:
                return finalize_selected(
                    min(zero_loss_zero_mismatch, key=selection_key),
                    f"{calibration_scope}_zero_loss_zero_mismatch_{score_label}",
                )
        if zero_loss:
            if max_calibration_mismatches is not None or max_calibration_mismatch_rate is not None:
                note = f"{calibration_scope}_zero_loss_mismatch_capped_{score_label}"
            else:
                note = f"{calibration_scope}_zero_loss_{score_label}"
            return finalize_selected(min(zero_loss, key=selection_key), note)
    if candidates:
        return finalize_selected(
            min(candidates, key=selection_key),
            f"{calibration_scope}_accuracy_tolerance_{score_label}",
        )
    selected = max(sweep, key=lambda row: (row["accuracy"], -row["avg_blocks"]))
    return finalize_selected(selected, f"{calibration_scope}_no_tolerance_candidate_max_accuracy")


def boundary_risk_slack(row: dict[str, Any]) -> float:
    """How permissive the risk-side gates are; lower means more conservative."""
    slack = 0.0
    for field in (
        "risk_threshold",
        "completion_risk_threshold",
        "empty_answer_risk_threshold",
        "answer_format_risk_threshold",
    ):
        value = row.get(field)
        if value is not None:
            slack += float(value)
    answer_identity_threshold = row.get("answer_identity_stability_threshold")
    if answer_identity_threshold is not None:
        slack += 1.0 - float(answer_identity_threshold)
    return slack


def threshold_key(row: dict[str, Any]) -> tuple[float, ...]:
    fields = ["readiness_threshold", "risk_threshold", "contentful_threshold"]
    if row.get("correctness_threshold") is not None:
        fields.append("correctness_threshold")
    if row.get("completion_risk_threshold") is not None:
        fields.append("completion_risk_threshold")
    if row.get("empty_answer_risk_threshold") is not None:
        fields.append("empty_answer_risk_threshold")
    if row.get("answer_format_risk_threshold") is not None:
        fields.append("answer_format_risk_threshold")
    if row.get("answer_identity_stability_threshold") is not None:
        fields.append("answer_identity_stability_threshold")
    return tuple(float(row[field]) for field in fields)


def build_task_threshold_lookup(
    task_sweeps: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, dict[tuple[float, ...], dict[str, Any]]]:
    if task_sweeps is None:
        return {}
    return {
        task: {threshold_key(row): row for row in rows}
        for task, rows in task_sweeps.items()
    }


def find_matching_row(sweep: list[dict[str, Any]], selected: dict[str, Any]) -> dict[str, Any]:
    for row in sweep:
        if threshold_key(row) == threshold_key(selected):
            result = dict(row)
            result["selection_note"] = selected.get("selection_note")
            result["calibration_scope"] = selected.get("calibration_scope")
            result["calibration_boundary_risk_penalty"] = selected.get("calibration_boundary_risk_penalty")
            result["calibration_selection_score"] = selected.get("calibration_selection_score")
            return result
    raise RuntimeError("selected threshold not found in test sweep")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            result[f"{prefix}_{key}"] = float(value)
        elif isinstance(value, (int, float)) and value is not None and not math.isnan(float(value)):
            result[f"{prefix}_{key}"] = float(value)
    return result


if __name__ == "__main__":
    main()
