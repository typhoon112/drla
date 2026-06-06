"""Train a latent-student action-to-halt gate for Cola P1.

The gate is a second-stage rare-event model.  It sees only the aggressive
action student's scores at the block where action would stop, plus the block
position.  It learns whether continuing with the strict halt policy after that
block would improve correctness.  Decoder text and official correctness are
used only as offline labels/metrics, never as online inference features.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from drla.scripts.aggregate_cola_action_halt_gate import (
    ActionHaltGateConfig,
    build_cases,
    find_eval_pairs,
    oracle_causal_metrics,
    policy_metrics,
    summarize_decisions,
)
from drla.scripts.train_cola_readiness_model import (
    OFFICIAL_COLA_TASKS,
    binary_auprc,
    binary_auroc,
    device_metadata,
    parse_tasks,
    pos_weight,
    require_cuda_training,
    resolve_device,
)
from drla.tracking import finish_experiment, init_experiment, log_metrics


FEATURE_NAMES = [
    "selected_block_fraction",
    "remaining_block_fraction",
    "selected_block_is_1",
    "selected_block_is_2",
    "selected_block_is_3",
    "student_readiness",
    "student_prediction_change",
    "student_contentful",
    "student_correctness",
    "student_future_gain",
    "student_completion_risk",
]


@dataclass(frozen=True)
class ActionHaltGateTrainConfig:
    action_root_glob: str = ActionHaltGateConfig.action_root_glob
    halt_root_glob: str = ActionHaltGateConfig.halt_root_glob
    output_dir: str = (
        "/data1/luyifei/drla/outputs/cola_action_halt_gate/"
        "official8_full_b64_bs12_loto_lambada_seed20260526"
    )
    heldout_task: str = "lambada"
    seed: int = 20260526
    train_split: str = "train"
    valid_split: str = "valid"
    eval_split: str = "test"
    validation_source: str = "train_stratified"
    internal_valid_fraction: float = 0.1
    boundary_valid_positive_fraction: float = 0.5
    boundary_valid_mismatch_fraction: float = 0.2
    boundary_valid_negative_fraction: float = 0.05
    batch_size: int = 4096
    epochs: int = 20
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    dropout: float = 0.1
    objective: str = "binary_bce"
    rescue_loss_weight: float = 800.0
    false_defer_block_weight: float = 2.0
    utility_correct_reward: float = 1.0
    utility_early_wrong_penalty: float = 10.0
    utility_block_cost: float = 0.25
    utility_mismatch_penalty: float = 0.1
    utility_weight_scale: float = 20.0
    utility_temperature: float = 1.0
    checkpoint_selection: str = "auprc"
    valid_interval: int = 10
    num_workers: int = 0
    device: str = "auto"
    swanlab_mode: str = "cloud"
    experiment_name: str = "official8-action-halt-gate-loto-lambada"
    dry_run: bool = False


class ActionHaltGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, output_dim: int = 1):
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.net(x)
        if self.output_dim == 1:
            return output.squeeze(-1)
        return output


def train_action_halt_gate(config: ActionHaltGateTrainConfig) -> dict[str, Any]:
    validate_config(config)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"

    pairs = find_eval_pairs(
        ActionHaltGateConfig(
            action_root_glob=config.action_root_glob,
            halt_root_glob=config.halt_root_glob,
            output_dir=config.output_dir,
        )
    )
    train_pairs = [pair for pair in pairs if pair["task"] != config.heldout_task]
    eval_pairs = [pair for pair in pairs if pair["task"] == config.heldout_task]
    if not train_pairs or not eval_pairs:
        raise ValueError("matched train/eval pairs are empty")

    source_train_cases = load_cases(train_pairs, split=config.train_split)
    source_official_valid_cases = load_cases(train_pairs, split=config.valid_split)
    if config.validation_source == "train_stratified":
        train_cases, valid_cases = split_cases_by_group(
            source_train_cases,
            valid_fraction=config.internal_valid_fraction,
            seed=config.seed,
        )
    elif config.validation_source == "train_boundary_stratified":
        train_cases, valid_cases = split_cases_by_boundary_group(source_train_cases, config)
    else:
        train_cases = source_train_cases
        valid_cases = source_official_valid_cases
    target_valid_cases = load_cases(eval_pairs, split=config.valid_split)
    test_cases = load_cases(eval_pairs, split=config.eval_split)

    train_bundle = build_feature_bundle(train_cases, config)
    valid_bundle = build_feature_bundle(valid_cases, config)
    target_valid_bundle = build_feature_bundle(target_valid_cases, config)
    source_official_valid_bundle = build_feature_bundle(source_official_valid_cases, config)
    test_bundle = build_feature_bundle(test_cases, config)
    norm_stats = fit_norm(train_bundle["x"])
    train_x = normalize(train_bundle["x"], norm_stats)
    valid_x = normalize(valid_bundle["x"], norm_stats)
    target_valid_x = normalize(target_valid_bundle["x"], norm_stats)
    source_official_valid_x = normalize(source_official_valid_bundle["x"], norm_stats)
    test_x = normalize(test_bundle["x"], norm_stats)

    data_summary = {
        "num_pairs": len(pairs),
        "num_train_pairs": len(train_pairs),
        "num_eval_pairs": len(eval_pairs),
        "num_train_cases": len(train_cases),
        "num_valid_cases": len(valid_cases),
        "num_source_official_valid_cases": len(source_official_valid_cases),
        "num_target_valid_cases": len(target_valid_cases),
        "num_test_cases": len(test_cases),
        "train_positive_rate": positive_rate(train_bundle["y"]),
        "valid_positive_rate": positive_rate(valid_bundle["y"]),
        "source_official_valid_positive_rate": positive_rate(source_official_valid_bundle["y"]),
        "target_valid_positive_rate": positive_rate(target_valid_bundle["y"]),
        "test_positive_rate": positive_rate(test_bundle["y"]),
        "train_mean_sample_weight": float(train_bundle["sample_weight"].mean().item()),
        "valid_mean_sample_weight": float(valid_bundle["sample_weight"].mean().item()),
        "train_utility_delta_mean": float(train_bundle["utility_target"].mean().item()),
        "train_utility_delta_min": float(train_bundle["utility_target"].min().item()),
        "train_utility_delta_max": float(train_bundle["utility_target"].max().item()),
        "valid_utility_delta_mean": float(valid_bundle["utility_target"].mean().item()),
        "valid_utility_delta_min": float(valid_bundle["utility_target"].min().item()),
        "valid_utility_delta_max": float(valid_bundle["utility_target"].max().item()),
        "train_rescue_rate": positive_rate(train_bundle["rescue_target"]),
        "train_defer_mismatch_rescue_rate": positive_rate(train_bundle["mismatch_rescue_target"]),
        "train_defer_harm_rate": positive_rate(train_bundle["harm_target"]),
        "train_defer_mismatch_rate": positive_rate(train_bundle["mismatch_target"]),
        "train_defer_cost_mean": float(train_bundle["cost_target"].mean().item()),
        "valid_rescue_rate": positive_rate(valid_bundle["rescue_target"]),
        "valid_defer_mismatch_rescue_rate": positive_rate(valid_bundle["mismatch_rescue_target"]),
        "valid_defer_harm_rate": positive_rate(valid_bundle["harm_target"]),
        "valid_defer_mismatch_rate": positive_rate(valid_bundle["mismatch_target"]),
        "valid_defer_cost_mean": float(valid_bundle["cost_target"].mean().item()),
        "source_train_group_summary": summarize_case_groups(source_train_cases),
        "train_group_summary": summarize_case_groups(train_cases),
        "valid_group_summary": summarize_case_groups(valid_cases),
    }
    if config.dry_run:
        summary = {
            "created_at": int(time.time()),
            "config": asdict(config),
            "feature_names": FEATURE_NAMES,
            "data_summary": data_summary,
            "online_input_policy": online_input_policy(),
            "note": "dry-run only; no optimizer was created and no SwanLab run was started",
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary

    device = resolve_device(config.device)
    require_cuda_training(device, "train_cola_action_halt_gate.py")
    device_info = device_metadata(device)
    if config.objective == "binary_bce":
        train_dataset = TensorDataset(train_x, train_bundle["y"])
    elif config.objective == "cost_weighted_bce":
        train_dataset = TensorDataset(train_x, train_bundle["y"], train_bundle["sample_weight"])
    elif config.objective in {"utility_mse", "utility_pairwise", "utility_soft_bce"}:
        train_dataset = TensorDataset(
            train_x,
            train_bundle["y"],
            train_bundle["utility_target"],
            train_bundle["utility_weight"],
        )
    elif config.objective == "decomposed_expected_utility":
        train_dataset = TensorDataset(
            train_x,
            train_bundle["y"],
            train_bundle["rescue_target"],
            train_bundle["mismatch_rescue_target"],
            train_bundle["harm_target"],
            train_bundle["mismatch_target"],
            train_bundle["cost_target"],
        )
    else:
        raise ValueError(f"unknown objective: {config.objective}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
    )
    valid_loader = DataLoader(TensorDataset(valid_x, valid_bundle["y"]), batch_size=config.batch_size)
    model = ActionHaltGate(
        len(FEATURE_NAMES),
        config.hidden_dim,
        config.dropout,
        output_dim=objective_output_dim(config),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    class_pos_weight = pos_weight(train_bundle["y"]).to(device) if config.objective == "binary_bce" else None
    decomposed_pos_weights = {
        "rescue": pos_weight(train_bundle["rescue_target"]).to(device),
        "mismatch_rescue": pos_weight(train_bundle["mismatch_rescue_target"]).to(device),
        "harm": pos_weight(train_bundle["harm_target"]).to(device),
        "mismatch": pos_weight(train_bundle["mismatch_target"]).to(device),
    }

    run = init_experiment(
        stage="cola-action-halt-gate",
        experiment_name=config.experiment_name,
        description="Rare-event latent-student action-to-halt gate trained without decoder inference inputs.",
        config={
            **asdict(config),
            "device_info": device_info,
            "feature_names": FEATURE_NAMES,
            "data_summary": data_summary,
            "online_input_policy": online_input_policy(),
            "label_policy": label_policy(),
        },
        mode=config.swanlab_mode,
        tags=["cola", "official-benchmark", "latent-halt-student", "action-halt-gate"],
    )

    best_metric = -math.inf
    best_step = 0
    global_step = 0
    start_time = time.time()

    try:
        with metrics_path.open("w", encoding="utf-8") as metrics_f:
            for epoch in range(config.epochs):
                model.train()
                for batch in train_loader:
                    global_step += 1
                    x = batch[0].to(device)
                    y = batch[1].to(device)
                    logits = model(x)
                    if config.objective == "binary_bce":
                        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=class_pos_weight)
                    elif config.objective == "cost_weighted_bce":
                        sample_weight = batch[2].to(device)
                        loss_raw = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
                        loss = (loss_raw * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
                    elif config.objective == "utility_mse":
                        utility_target = batch[2].to(device)
                        utility_weight = batch[3].to(device)
                        loss_raw = F.smooth_l1_loss(logits, utility_target, reduction="none")
                        loss = (loss_raw * utility_weight).sum() / utility_weight.sum().clamp_min(1e-6)
                    elif config.objective == "utility_pairwise":
                        utility_target = batch[2].to(device)
                        utility_weight = batch[3].to(device)
                        target_sign = torch.where(utility_target > 0.0, 1.0, -1.0)
                        loss_raw = F.softplus(-target_sign * logits)
                        loss = (loss_raw * utility_weight).sum() / utility_weight.sum().clamp_min(1e-6)
                    elif config.objective == "utility_soft_bce":
                        utility_target = batch[2].to(device)
                        utility_weight = batch[3].to(device)
                        soft_target = torch.sigmoid(utility_target / config.utility_temperature)
                        loss_raw = F.binary_cross_entropy_with_logits(logits, soft_target, reduction="none")
                        loss = (loss_raw * utility_weight).sum() / utility_weight.sum().clamp_min(1e-6)
                    elif config.objective == "decomposed_expected_utility":
                        rescue_target = batch[2].to(device)
                        mismatch_rescue_target = batch[3].to(device)
                        harm_target = batch[4].to(device)
                        mismatch_target = batch[5].to(device)
                        cost_target = batch[6].to(device)
                        if logits.ndim != 2 or logits.shape[1] != 5:
                            raise ValueError("decomposed_expected_utility requires five output heads")
                        rescue_loss = F.binary_cross_entropy_with_logits(
                            logits[:, 0],
                            rescue_target,
                            pos_weight=decomposed_pos_weights["rescue"],
                        )
                        mismatch_rescue_loss = F.binary_cross_entropy_with_logits(
                            logits[:, 1],
                            mismatch_rescue_target,
                            pos_weight=decomposed_pos_weights["mismatch_rescue"],
                        )
                        harm_loss = F.binary_cross_entropy_with_logits(
                            logits[:, 2],
                            harm_target,
                            pos_weight=decomposed_pos_weights["harm"],
                        )
                        mismatch_loss = F.binary_cross_entropy_with_logits(
                            logits[:, 3],
                            mismatch_target,
                            pos_weight=decomposed_pos_weights["mismatch"],
                        )
                        cost_loss = F.smooth_l1_loss(torch.sigmoid(logits[:, 4]), cost_target)
                        loss = (
                            rescue_loss
                            + mismatch_rescue_loss
                            + harm_loss
                            + mismatch_loss
                            + config.utility_block_cost * cost_loss
                        )
                    else:
                        raise ValueError(f"unknown objective: {config.objective}")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    train_metrics = {
                        "loss": float(loss.item()),
                        "epoch": float(epoch),
                        "positive_rate": data_summary["train_positive_rate"],
                    }
                    log_metrics(train_metrics, step=global_step, prefix="train")
                    write_metric(metrics_f, "train", global_step, train_metrics)

                    if global_step % config.valid_interval == 0:
                        valid_metrics = evaluate_binary(model, valid_loader, device)
                        log_metrics(valid_metrics, step=global_step, prefix="valid")
                        write_metric(metrics_f, "valid", global_step, valid_metrics)
                        selected = select_checkpoint_metric(
                            valid_metrics,
                            model=model,
                            x=valid_x,
                            cases=valid_cases,
                            device=device,
                            config=config,
                        )
                        if selected > best_metric:
                            best_metric = selected
                            best_step = global_step
                            save_checkpoint(
                                checkpoint_dir / "best_checkpoint.pt",
                                model=model,
                                optimizer=optimizer,
                                config=config,
                                norm_stats=norm_stats,
                                step=global_step,
                                metric=best_metric,
                            )

            last_valid_metrics = evaluate_binary(model, valid_loader, device)
            last_metric = select_checkpoint_metric(
                last_valid_metrics,
                model=model,
                x=valid_x,
                cases=valid_cases,
                device=device,
                config=config,
            )
            save_checkpoint(
                checkpoint_dir / "last_checkpoint.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                norm_stats=norm_stats,
                step=global_step,
                metric=last_metric,
            )
            if last_metric > best_metric:
                best_metric = last_metric
                best_step = global_step
                save_checkpoint(
                    checkpoint_dir / "best_checkpoint.pt",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    norm_stats=norm_stats,
                    step=global_step,
                    metric=best_metric,
                )

            best_checkpoint = torch.load(checkpoint_dir / "best_checkpoint.pt", map_location="cpu")
            model.load_state_dict(best_checkpoint["model_state"])
            best_step = int(best_checkpoint["step"])
            best_metric = float(best_checkpoint["metric"])

            final_valid_metrics = evaluate_binary(model, valid_loader, device)
            source_official_valid_metrics = evaluate_binary_tensor(
                model,
                source_official_valid_x,
                source_official_valid_bundle["y"],
                device,
            )
            target_valid_metrics = evaluate_binary_tensor(model, target_valid_x, target_valid_bundle["y"], device)
            test_binary_metrics = evaluate_binary_tensor(model, test_x, test_bundle["y"], device)
            valid_scores = predict_policy_scores(model, valid_x, device, config)
            target_valid_scores = predict_policy_scores(model, target_valid_x, device, config)
            test_scores = predict_policy_scores(model, test_x, device, config)
            valid_thresholds = policy_thresholds(valid_scores, config)
            target_valid_thresholds = policy_thresholds(target_valid_scores, config)
            test_thresholds = policy_thresholds(test_scores, config)
            test_apply_thresholds = sorted(set(test_thresholds + valid_thresholds + target_valid_thresholds))
            valid_sweep = sweep_gate_thresholds(valid_cases, valid_scores, thresholds=valid_thresholds)
            target_valid_sweep = sweep_gate_thresholds(
                target_valid_cases,
                target_valid_scores,
                thresholds=target_valid_thresholds,
            )
            test_sweep = sweep_gate_thresholds(test_cases, test_scores, thresholds=test_apply_thresholds)
            selected_source_safety = select_policy_threshold(valid_sweep, mode="safety")
            selected_source_cost = select_policy_threshold(valid_sweep, mode="cost")
            selected_source_cost_limited = select_policy_threshold(
                valid_sweep,
                mode="test_loss_cost_limited",
                action_avg_blocks=policy_metrics(valid_cases, "action")["avg_blocks"],
            )
            selected_target_safety = select_policy_threshold(target_valid_sweep, mode="safety")
            selected_target_cost = select_policy_threshold(target_valid_sweep, mode="cost")
            selected_target_cost_limited = select_policy_threshold(
                target_valid_sweep,
                mode="test_loss_cost_limited",
                action_avg_blocks=policy_metrics(target_valid_cases, "action")["avg_blocks"],
            )
            selected_source_safety_test = matching_threshold(test_sweep, selected_source_safety["threshold"])
            selected_source_cost_test = matching_threshold(test_sweep, selected_source_cost["threshold"])
            selected_source_cost_limited_test = matching_threshold(
                test_sweep,
                selected_source_cost_limited["threshold"],
            )
            selected_target_safety_test = matching_threshold(test_sweep, selected_target_safety["threshold"])
            selected_target_cost_test = matching_threshold(test_sweep, selected_target_cost["threshold"])
            selected_target_cost_limited_test = matching_threshold(
                test_sweep,
                selected_target_cost_limited["threshold"],
            )

            log_metrics(final_valid_metrics, step=global_step, prefix="valid")
            log_metrics(last_valid_metrics, step=global_step, prefix="last_valid")
            log_metrics(source_official_valid_metrics, step=global_step, prefix="source_official_valid")
            log_metrics(target_valid_metrics, step=global_step, prefix="target_valid")
            log_metrics(test_binary_metrics, step=global_step, prefix="test")
            log_metrics(flatten_policy("test_source_safety_gate", selected_source_safety_test), step=global_step)
            log_metrics(flatten_policy("test_source_cost_gate", selected_source_cost_test), step=global_step)
            log_metrics(
                flatten_policy("test_source_cost_limited_gate", selected_source_cost_limited_test),
                step=global_step,
            )
            write_metric(metrics_f, "valid", global_step, final_valid_metrics)
            write_metric(metrics_f, "last_valid", global_step, last_valid_metrics)
            write_metric(metrics_f, "source_official_valid", global_step, source_official_valid_metrics)
            write_metric(metrics_f, "target_valid", global_step, target_valid_metrics)
            write_metric(metrics_f, "test", global_step, test_binary_metrics)
            write_metric(metrics_f, "test_source_safety_gate", global_step, selected_source_safety_test)
            write_metric(metrics_f, "test_source_cost_gate", global_step, selected_source_cost_test)
            write_metric(metrics_f, "test_source_cost_limited_gate", global_step, selected_source_cost_limited_test)
        write_csv(output_dir / "valid_threshold_sweep.csv", valid_sweep)
        write_csv(output_dir / "target_valid_threshold_sweep.csv", target_valid_sweep)
        write_csv(output_dir / "test_threshold_sweep.csv", test_sweep)
        write_predictions(output_dir / "test_gate_predictions.jsonl", test_cases, test_scores, test_bundle["y"])

        summary = {
            "created_at": int(time.time()),
            "config": asdict(config),
            "device_info": device_info,
            "feature_names": FEATURE_NAMES,
            "policy_score_semantics": policy_score_semantics(config),
            "online_input_policy": online_input_policy(),
            "label_policy": label_policy(),
            "forbidden_inference_inputs": forbidden_inputs(),
            "data_summary": data_summary,
            "best_step": best_step,
            "best_metric_name": f"valid/{config.checkpoint_selection}",
            "best_metric": best_metric,
            "best_valid_metrics": final_valid_metrics,
            "last_valid_metrics": last_valid_metrics,
            "source_official_valid_metrics": source_official_valid_metrics,
            "target_valid_metrics": target_valid_metrics,
            "test_binary_metrics": test_binary_metrics,
            "test_policies": {
                "action": policy_metrics(test_cases, "action"),
                "halt_original": policy_metrics(test_cases, "halt_original"),
                "causal_always_defer_after_action": policy_metrics(test_cases, "fallback"),
                "causal_oracle_defer_after_action": oracle_causal_metrics(test_cases),
                "source_valid_safety_selected_gate": selected_source_safety_test,
                "source_valid_cost_selected_gate": selected_source_cost_test,
                "source_valid_cost_limited_selected_gate": selected_source_cost_limited_test,
                "target_valid_safety_selected_gate": selected_target_safety_test,
                "target_valid_cost_selected_gate": selected_target_cost_test,
                "target_valid_cost_limited_selected_gate": selected_target_cost_limited_test,
                "best_test_gate_by_loss": select_policy_threshold(test_sweep, mode="test_loss"),
                "best_test_gate_under_action_plus_0p10_blocks": select_policy_threshold(
                    test_sweep,
                    mode="test_loss_cost_limited",
                    action_avg_blocks=policy_metrics(test_cases, "action")["avg_blocks"],
                ),
            },
            "artifacts": {
                "metrics_jsonl": str(metrics_path),
                "best_checkpoint": str(checkpoint_dir / "best_checkpoint.pt"),
                "last_checkpoint": str(checkpoint_dir / "last_checkpoint.pt"),
                "valid_threshold_sweep": str(output_dir / "valid_threshold_sweep.csv"),
                "target_valid_threshold_sweep": str(output_dir / "target_valid_threshold_sweep.csv"),
                "test_threshold_sweep": str(output_dir / "test_threshold_sweep.csv"),
                "test_gate_predictions": str(output_dir / "test_gate_predictions.jsonl"),
            },
            "elapsed_seconds": time.time() - start_time,
            "swanlab_run_id": getattr(run, "id", None),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary
    finally:
        finish_experiment()


def load_cases(pairs: list[dict[str, Any]], *, split: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for pair in pairs:
        cases.extend(build_cases(pair, split=split))
    return cases


def split_cases_by_group(
    cases: list[dict[str, Any]],
    *,
    valid_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("internal_valid_fraction must be between 0 and 1")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for case in cases:
        groups.setdefault((case["task"], case["sample_key"]), []).append(case)

    positive_keys = []
    negative_keys = []
    for key, group_cases in groups.items():
        if any(case_label(case) > 0.5 for case in group_cases):
            positive_keys.append(key)
        else:
            negative_keys.append(key)

    generator = torch.Generator().manual_seed(seed)
    valid_keys = set(sample_keys(positive_keys, valid_fraction, generator))
    valid_keys.update(sample_keys(negative_keys, valid_fraction, generator))
    train_cases = []
    valid_cases = []
    for case in cases:
        target = valid_cases if (case["task"], case["sample_key"]) in valid_keys else train_cases
        target.append(case)
    return train_cases, valid_cases


def split_cases_by_boundary_group(
    cases: list[dict[str, Any]],
    config: ActionHaltGateTrainConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for case in cases:
        groups.setdefault((case["task"], case["sample_key"]), []).append(case)

    positive_keys = []
    boundary_negative_keys = []
    ordinary_negative_keys = []
    for key, group_cases in groups.items():
        if any(case_label(case) > 0.5 for case in group_cases):
            positive_keys.append(key)
        elif any(case_is_boundary_negative(case) for case in group_cases):
            boundary_negative_keys.append(key)
        else:
            ordinary_negative_keys.append(key)

    generator = torch.Generator().manual_seed(config.seed)
    valid_keys = set(sample_keys(positive_keys, config.boundary_valid_positive_fraction, generator))
    valid_keys.update(sample_keys(boundary_negative_keys, config.boundary_valid_mismatch_fraction, generator))
    valid_keys.update(sample_keys(ordinary_negative_keys, config.boundary_valid_negative_fraction, generator))

    train_cases = []
    valid_cases = []
    for case in cases:
        target = valid_cases if (case["task"], case["sample_key"]) in valid_keys else train_cases
        target.append(case)
    return train_cases, valid_cases


def sample_keys(
    keys: list[tuple[str, str]],
    fraction: float,
    generator: torch.Generator,
) -> list[tuple[str, str]]:
    if not keys or fraction <= 0.0:
        return []
    count = max(1, int(round(len(keys) * fraction)))
    count = min(count, len(keys))
    order = torch.randperm(len(keys), generator=generator).tolist()
    return [keys[idx] for idx in order[:count]]


def summarize_case_groups(cases: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for case in cases:
        groups.setdefault((case["task"], case["sample_key"]), []).append(case)
    positive = 0
    boundary_negative = 0
    ordinary_negative = 0
    for group_cases in groups.values():
        if any(case_label(case) > 0.5 for case in group_cases):
            positive += 1
        elif any(case_is_boundary_negative(case) for case in group_cases):
            boundary_negative += 1
        else:
            ordinary_negative += 1
    total = max(len(groups), 1)
    return {
        "num_groups": len(groups),
        "positive_groups": positive,
        "positive_group_rate": positive / total,
        "boundary_negative_groups": boundary_negative,
        "boundary_negative_group_rate": boundary_negative / total,
        "ordinary_negative_groups": ordinary_negative,
        "ordinary_negative_group_rate": ordinary_negative / total,
    }


def build_feature_bundle(
    cases: list[dict[str, Any]],
    config: ActionHaltGateTrainConfig,
) -> dict[str, torch.Tensor]:
    features = [case_features(case) for case in cases]
    labels = [case_label(case) for case in cases]
    sample_weights = [case_sample_weight(case, config) for case in cases]
    utility_targets = [case_utility_delta(case, config) for case in cases]
    utility_weights = [1.0 + config.utility_weight_scale * abs(value) for value in utility_targets]
    rescue_targets = [case_label(case) for case in cases]
    mismatch_rescue_targets = [case_defer_mismatch_rescue_label(case) for case in cases]
    harm_targets = [case_defer_harm_label(case) for case in cases]
    mismatch_targets = [case_defer_mismatch_label(case) for case in cases]
    cost_targets = [case_defer_cost(case) for case in cases]
    return {
        "x": torch.tensor(features, dtype=torch.float32),
        "y": torch.tensor(labels, dtype=torch.float32),
        "sample_weight": torch.tensor(sample_weights, dtype=torch.float32),
        "utility_target": torch.tensor(utility_targets, dtype=torch.float32),
        "utility_weight": torch.tensor(utility_weights, dtype=torch.float32),
        "rescue_target": torch.tensor(rescue_targets, dtype=torch.float32),
        "mismatch_rescue_target": torch.tensor(mismatch_rescue_targets, dtype=torch.float32),
        "harm_target": torch.tensor(harm_targets, dtype=torch.float32),
        "mismatch_target": torch.tensor(mismatch_targets, dtype=torch.float32),
        "cost_target": torch.tensor(cost_targets, dtype=torch.float32),
    }


def case_features(case: dict[str, Any]) -> list[float]:
    action = case["action"]
    selected_block = float(action["selected_block"])
    final_block = max(float(action["final_block"]), 1.0)
    return [
        selected_block / final_block,
        (final_block - selected_block) / final_block,
        1.0 if int(selected_block) == 1 else 0.0,
        1.0 if int(selected_block) == 2 else 0.0,
        1.0 if int(selected_block) == 3 else 0.0,
        safe_float(action.get("student_readiness")),
        safe_float(action.get("student_prediction_change")),
        safe_float(action.get("student_contentful")),
        safe_float(action.get("student_correctness")),
        safe_float(action.get("student_future_gain")),
        safe_float(action.get("student_completion_risk")),
    ]


def case_label(case: dict[str, Any]) -> float:
    action_correct = bool(case["action"]["selected_correct"])
    fallback_correct = bool(case["fallback"]["selected_correct"])
    return 1.0 if fallback_correct and not action_correct else 0.0


def case_defer_harm_label(case: dict[str, Any]) -> float:
    action_safe = not bool(case["action"]["loss_vs_final"])
    fallback_loses = bool(case["fallback"]["loss_vs_final"])
    return 1.0 if action_safe and fallback_loses else 0.0


def case_defer_mismatch_rescue_label(case: dict[str, Any]) -> float:
    action_mismatches = bool(case["action"].get("_prediction_mismatch_vs_final"))
    fallback_matches = not bool(case["fallback"].get("_prediction_mismatch_vs_final"))
    return 1.0 if action_mismatches and fallback_matches else 0.0


def case_defer_mismatch_label(case: dict[str, Any]) -> float:
    action_matches = not bool(case["action"].get("_prediction_mismatch_vs_final"))
    fallback_mismatches = bool(case["fallback"].get("_prediction_mismatch_vs_final"))
    return 1.0 if action_matches and fallback_mismatches else 0.0


def case_defer_cost(case: dict[str, Any]) -> float:
    action = case["action"]
    fallback = case["fallback"]
    final_block = max(float(action["final_block"]), 1.0)
    return max(0.0, float(fallback["selected_block"]) - float(action["selected_block"])) / final_block


def case_is_boundary_negative(case: dict[str, Any]) -> bool:
    action = case["action"]
    fallback = case["fallback"]
    if bool(action["loss_vs_final"]) or bool(fallback["loss_vs_final"]):
        return True
    if bool(action.get("_prediction_mismatch_vs_final")) or bool(fallback.get("_prediction_mismatch_vs_final")):
        return True
    return bool(action.get("_prediction_mismatch_vs_prediction_stability")) or bool(
        fallback.get("_prediction_mismatch_vs_prediction_stability")
    )


def case_sample_weight(case: dict[str, Any], config: ActionHaltGateTrainConfig) -> float:
    if config.objective == "binary_bce":
        return 1.0
    label = case_label(case)
    if label > 0.5:
        return config.rescue_loss_weight
    action = case["action"]
    fallback = case["fallback"]
    return 1.0 + config.false_defer_block_weight * case_defer_cost({"action": action, "fallback": fallback})


def case_utility_delta(case: dict[str, Any], config: ActionHaltGateTrainConfig) -> float:
    return decision_utility(case["fallback"], config) - decision_utility(case["action"], config)


def decision_utility(decision: dict[str, Any], config: ActionHaltGateTrainConfig) -> float:
    final_block = max(float(decision["final_block"]), 1.0)
    selected_block = float(decision["selected_block"])
    utility = config.utility_correct_reward if bool(decision["selected_correct"]) else 0.0
    utility -= config.utility_block_cost * (selected_block / final_block)
    if bool(decision["loss_vs_final"]):
        utility -= config.utility_early_wrong_penalty
    if bool(decision.get("_prediction_mismatch_vs_final")):
        utility -= config.utility_mismatch_penalty
    return utility


def safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def positive_rate(y: torch.Tensor) -> float:
    if y.numel() == 0:
        return 0.0
    return float(y.float().mean().item())


def fit_norm(x: torch.Tensor) -> dict[str, torch.Tensor]:
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    return {"mean": mean, "std": std}


def normalize(x: torch.Tensor, norm_stats: dict[str, torch.Tensor]) -> torch.Tensor:
    return (x - norm_stats["mean"]) / norm_stats["std"]


def objective_output_dim(config: ActionHaltGateTrainConfig) -> int:
    return 5 if config.objective == "decomposed_expected_utility" else 1


def primary_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2:
        return logits[:, 0]
    return logits


def decomposed_expected_utility_score(
    logits: torch.Tensor,
    config: ActionHaltGateTrainConfig,
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != 5:
        raise ValueError("decomposed_expected_utility requires five output heads")
    rescue_prob = torch.sigmoid(logits[:, 0])
    mismatch_rescue_prob = torch.sigmoid(logits[:, 1])
    harm_prob = torch.sigmoid(logits[:, 2])
    mismatch_prob = torch.sigmoid(logits[:, 3])
    extra_block_cost = torch.sigmoid(logits[:, 4])
    correctness_swing = config.utility_correct_reward + config.utility_early_wrong_penalty
    return (
        rescue_prob * correctness_swing
        + mismatch_rescue_prob * config.utility_mismatch_penalty
        - harm_prob * correctness_swing
        - mismatch_prob * config.utility_mismatch_penalty
        - extra_block_cost * config.utility_block_cost
    )


@torch.no_grad()
def evaluate_binary(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    logits_list = []
    target_list = []
    model.eval()
    for x, y in loader:
        logits_list.append(model(x.to(device)).cpu())
        target_list.append(y.cpu())
    logits = torch.cat(logits_list)
    target = torch.cat(target_list)
    return binary_metrics(logits, target)


@torch.no_grad()
def evaluate_binary_tensor(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    logits = model(x.to(device)).cpu()
    return binary_metrics(logits, y.cpu())


def binary_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    logits = primary_logits(logits)
    prob = torch.sigmoid(logits)
    pred = prob >= 0.5
    target_bool = target.bool()
    tp = int((pred & target_bool).sum().item())
    fp = int((pred & ~target_bool).sum().item())
    fn = int((~pred & target_bool).sum().item())
    tn = int((~pred & ~target_bool).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "loss": float(F.binary_cross_entropy_with_logits(logits, target).item()),
        "auroc": binary_auroc(prob, target),
        "auprc": binary_auprc(prob, target),
        "accuracy": float((pred.float() == target).float().mean().item()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_rate": positive_rate(target),
        "predicted_positive_rate": float(pred.float().mean().item()),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


@torch.no_grad()
def predict_probs(model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    probs = []
    for start in range(0, x.shape[0], 65536):
        probs.append(torch.sigmoid(model(x[start : start + 65536].to(device))).cpu())
    return torch.cat(probs)


@torch.no_grad()
def predict_policy_scores(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    config: ActionHaltGateTrainConfig,
) -> torch.Tensor:
    model.eval()
    scores = []
    for start in range(0, x.shape[0], 65536):
        logits = model(x[start : start + 65536].to(device)).cpu()
        if config.objective in {"utility_mse", "utility_pairwise"}:
            scores.append(logits)
        elif config.objective == "decomposed_expected_utility":
            scores.append(decomposed_expected_utility_score(logits, config))
        else:
            scores.append(torch.sigmoid(logits))
    return torch.cat(scores)


def policy_thresholds(scores: torch.Tensor, config: ActionHaltGateTrainConfig) -> list[float]:
    if config.objective not in {"utility_mse", "utility_pairwise", "decomposed_expected_utility"}:
        return sorted({0.0, 0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0})
    finite_scores = scores[torch.isfinite(scores)]
    if finite_scores.numel() == 0:
        return [0.0]
    quantiles = torch.tensor(
        [0.0, 0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
        dtype=finite_scores.dtype,
    )
    values = torch.quantile(finite_scores.float(), quantiles.float()).tolist()
    values.extend([0.0, float(finite_scores.min().item()) - 1e-6, float(finite_scores.max().item()) + 1e-6])
    return sorted({round(float(value), 8) for value in values})


def sweep_gate_thresholds(
    cases: list[dict[str, Any]],
    scores: torch.Tensor,
    *,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    rows = []
    for threshold in thresholds:
        rows.append(metrics_for_threshold(cases, scores, threshold))
    return rows


def metrics_for_threshold(cases: list[dict[str, Any]], scores: torch.Tensor, threshold: float) -> dict[str, Any]:
    decisions = []
    defer_count = 0
    rescued_action_losses = 0
    introduced_losses = 0
    for case, score in zip(cases, scores.tolist(), strict=True):
        can_defer = int(case["action"]["selected_block"]) < int(case["action"]["final_block"])
        use_fallback = can_defer and score >= threshold
        if use_fallback:
            defer_count += 1
        decision = case["fallback"] if use_fallback else case["action"]
        if bool(case["action"]["loss_vs_final"]) and not bool(decision["loss_vs_final"]):
            rescued_action_losses += 1
        if not bool(case["action"]["loss_vs_final"]) and bool(decision["loss_vs_final"]):
            introduced_losses += 1
        decisions.append(decision)
    metrics = summarize_decisions(decisions)
    metrics.update(
        {
            "threshold": threshold,
            "defer_count": defer_count,
            "defer_rate": defer_count / max(len(cases), 1),
            "rescued_action_losses": rescued_action_losses,
            "introduced_losses_vs_action": introduced_losses,
        }
    )
    return metrics


def select_policy_threshold(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    action_avg_blocks: float | None = None,
) -> dict[str, Any]:
    if mode == "safety":
        return min(
            rows,
            key=lambda row: (
                int(row["losses_vs_final"]),
                int(row["prediction_mismatch_vs_final"]),
                float(row["avg_blocks"]),
                -int(row["rescued_action_losses"]),
                float(row["threshold"]),
            ),
        )
    if mode == "cost":
        return min(
            rows,
            key=lambda row: (
                int(row["losses_vs_final"]),
                float(row["avg_blocks"]),
                int(row["prediction_mismatch_vs_final"]),
                -int(row["rescued_action_losses"]),
                float(row["threshold"]),
            ),
        )
    if mode == "test_loss":
        return select_policy_threshold(rows, mode="safety")
    if mode == "test_loss_cost_limited":
        if action_avg_blocks is None:
            raise ValueError("action_avg_blocks is required")
        eligible = [row for row in rows if float(row["avg_blocks"]) <= action_avg_blocks + 0.10]
        return select_policy_threshold(eligible or rows, mode="safety")
    raise ValueError(f"unknown mode: {mode}")


def matching_threshold(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    for row in rows:
        if float(row["threshold"]) == float(threshold):
            return row
    raise KeyError(threshold)


def select_checkpoint_metric(
    metrics: dict[str, float],
    *,
    model: nn.Module,
    x: torch.Tensor,
    cases: list[dict[str, Any]],
    device: torch.device,
    config: ActionHaltGateTrainConfig,
) -> float:
    if config.checkpoint_selection in {"auprc", "f1", "auroc"}:
        value = metrics.get(config.checkpoint_selection)
        if value is None or math.isnan(value):
            return -math.inf
        return float(value)
    scores = predict_policy_scores(model, x, device, config)
    sweep = sweep_gate_thresholds(cases, scores, thresholds=policy_thresholds(scores, config))
    action_avg_blocks = policy_metrics(cases, "action")["avg_blocks"]
    if config.checkpoint_selection == "policy_cost_limited":
        selected = select_policy_threshold(
            sweep,
            mode="test_loss_cost_limited",
            action_avg_blocks=action_avg_blocks,
        )
    elif config.checkpoint_selection == "policy_safety":
        selected = select_policy_threshold(sweep, mode="safety")
    else:
        raise ValueError("unknown checkpoint_selection")
    return -(
        10000.0 * float(selected.get("losses_vs_final", 0.0))
        + float(selected.get("prediction_mismatch_vs_final", 0.0))
        + 0.01 * float(selected.get("avg_blocks", 0.0))
    )


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ActionHaltGateTrainConfig,
    norm_stats: dict[str, torch.Tensor],
    step: int,
    metric: float,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(config),
            "feature_names": FEATURE_NAMES,
            "norm_stats": norm_stats,
            "step": step,
            "metric": metric,
        },
        path,
    )


def write_metric(f, split: str, step: int, metrics: dict[str, Any]) -> None:
    f.write(json.dumps({"split": split, "step": step, **metrics}, ensure_ascii=False, sort_keys=True) + "\n")
    f.flush()


def flatten_policy(prefix: str, metrics: dict[str, Any]) -> dict[str, float]:
    return {
        f"{prefix}/{key}": float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(
    path: Path,
    cases: list[dict[str, Any]],
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        for case, score, label in zip(cases, scores.tolist(), labels.tolist(), strict=True):
            action = case["action"]
            record = {
                "task": case["task"],
                "seed": case["seed"],
                "subseed": case["subseed"],
                "sample_key": case["sample_key"],
                "gate_score": float(score),
                "target_defer": float(label),
                "action_selected_block": action["selected_block"],
                "action_selected_correct": action["selected_correct"],
                "fallback_selected_correct": case["fallback"]["selected_correct"],
                "action_loss_vs_final": action["loss_vs_final"],
                "fallback_loss_vs_final": case["fallback"]["loss_vs_final"],
            }
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def online_input_policy() -> str:
    return (
        "Only action selected block number/fraction and action student latent-head scores "
        "(readiness, prediction_change, contentful, correctness, future_gain, completion_risk) "
        "are used as inference features."
    )


def policy_score_semantics(config: ActionHaltGateTrainConfig) -> str:
    if config.objective == "utility_mse":
        return "raw predicted fallback-minus-action utility delta; defer when score >= selected threshold"
    if config.objective == "utility_pairwise":
        return "raw accept-vs-defer utility ranking score; defer when score >= selected threshold"
    if config.objective == "utility_soft_bce":
        return "predicted soft utility advantage probability; defer when probability >= selected threshold"
    if config.objective == "decomposed_expected_utility":
        return (
            "raw expected defer utility from five learned heads: rescue_loss_prob, mismatch_rescue_prob, "
            "introduced_loss_prob, introduced_mismatch_prob, and extra_block_cost; defer when score >= selected threshold"
        )
    return "predicted defer probability; defer when probability >= selected threshold"


def label_policy() -> str:
    return (
        "target_defer=1 iff causal fallback after action's proposed stop block is correct "
        "and action's selected block is not correct; all decoded text/scorer/correctness fields "
        "are offline labels or metrics only."
    )


def forbidden_inputs() -> list[str]:
    return [
        "decoded selected_prediction",
        "decoded final_prediction",
        "prediction_stability_prediction",
        "official correctness at inference",
        "gold answers",
        "task scorer outputs at inference",
        "future block outputs before they are generated",
        "task id or task-hardcoded routing",
    ]


def validate_config(config: ActionHaltGateTrainConfig) -> None:
    parse_tasks(config.heldout_task)
    if config.heldout_task not in OFFICIAL_COLA_TASKS:
        raise ValueError("heldout_task must be one official Cola task")
    if config.valid_interval > 10:
        raise ValueError("valid_interval must be <= 10 steps")
    if config.validation_source not in {"train_stratified", "train_boundary_stratified", "official_valid"}:
        raise ValueError("validation_source must be train_stratified, train_boundary_stratified, or official_valid")
    for name, value in {
        "boundary_valid_positive_fraction": config.boundary_valid_positive_fraction,
        "boundary_valid_mismatch_fraction": config.boundary_valid_mismatch_fraction,
        "boundary_valid_negative_fraction": config.boundary_valid_negative_fraction,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if config.objective not in {
        "binary_bce",
        "cost_weighted_bce",
        "utility_mse",
        "utility_pairwise",
        "utility_soft_bce",
        "decomposed_expected_utility",
    }:
        raise ValueError(
            "objective must be binary_bce, cost_weighted_bce, utility_mse, utility_pairwise, "
            "utility_soft_bce, or decomposed_expected_utility"
        )
    if config.utility_temperature <= 0.0:
        raise ValueError("utility_temperature must be positive")
    if config.checkpoint_selection not in {"auprc", "auroc", "f1", "policy_cost_limited", "policy_safety"}:
        raise ValueError("unknown checkpoint_selection")
    if not config.dry_run and config.swanlab_mode != "cloud":
        raise ValueError("all deep-learning training experiments must use SwanLab cloud")


def parse_args() -> ActionHaltGateTrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-root-glob", default=ActionHaltGateTrainConfig.action_root_glob)
    parser.add_argument("--halt-root-glob", default=ActionHaltGateTrainConfig.halt_root_glob)
    parser.add_argument("--output-dir", default=ActionHaltGateTrainConfig.output_dir)
    parser.add_argument("--heldout-task", default=ActionHaltGateTrainConfig.heldout_task)
    parser.add_argument("--seed", type=int, default=ActionHaltGateTrainConfig.seed)
    parser.add_argument("--train-split", default=ActionHaltGateTrainConfig.train_split)
    parser.add_argument("--valid-split", default=ActionHaltGateTrainConfig.valid_split)
    parser.add_argument("--eval-split", default=ActionHaltGateTrainConfig.eval_split)
    parser.add_argument("--validation-source", default=ActionHaltGateTrainConfig.validation_source)
    parser.add_argument("--internal-valid-fraction", type=float, default=ActionHaltGateTrainConfig.internal_valid_fraction)
    parser.add_argument(
        "--boundary-valid-positive-fraction",
        type=float,
        default=ActionHaltGateTrainConfig.boundary_valid_positive_fraction,
    )
    parser.add_argument(
        "--boundary-valid-mismatch-fraction",
        type=float,
        default=ActionHaltGateTrainConfig.boundary_valid_mismatch_fraction,
    )
    parser.add_argument(
        "--boundary-valid-negative-fraction",
        type=float,
        default=ActionHaltGateTrainConfig.boundary_valid_negative_fraction,
    )
    parser.add_argument("--batch-size", type=int, default=ActionHaltGateTrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=ActionHaltGateTrainConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=ActionHaltGateTrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=ActionHaltGateTrainConfig.weight_decay)
    parser.add_argument("--hidden-dim", type=int, default=ActionHaltGateTrainConfig.hidden_dim)
    parser.add_argument("--dropout", type=float, default=ActionHaltGateTrainConfig.dropout)
    parser.add_argument("--objective", default=ActionHaltGateTrainConfig.objective)
    parser.add_argument("--rescue-loss-weight", type=float, default=ActionHaltGateTrainConfig.rescue_loss_weight)
    parser.add_argument(
        "--false-defer-block-weight",
        type=float,
        default=ActionHaltGateTrainConfig.false_defer_block_weight,
    )
    parser.add_argument("--checkpoint-selection", default=ActionHaltGateTrainConfig.checkpoint_selection)
    parser.add_argument("--utility-correct-reward", type=float, default=ActionHaltGateTrainConfig.utility_correct_reward)
    parser.add_argument(
        "--utility-early-wrong-penalty",
        type=float,
        default=ActionHaltGateTrainConfig.utility_early_wrong_penalty,
    )
    parser.add_argument("--utility-block-cost", type=float, default=ActionHaltGateTrainConfig.utility_block_cost)
    parser.add_argument(
        "--utility-mismatch-penalty",
        type=float,
        default=ActionHaltGateTrainConfig.utility_mismatch_penalty,
    )
    parser.add_argument("--utility-weight-scale", type=float, default=ActionHaltGateTrainConfig.utility_weight_scale)
    parser.add_argument("--utility-temperature", type=float, default=ActionHaltGateTrainConfig.utility_temperature)
    parser.add_argument("--valid-interval", type=int, default=ActionHaltGateTrainConfig.valid_interval)
    parser.add_argument("--num-workers", type=int, default=ActionHaltGateTrainConfig.num_workers)
    parser.add_argument("--device", default=ActionHaltGateTrainConfig.device)
    parser.add_argument("--swanlab-mode", default=ActionHaltGateTrainConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=ActionHaltGateTrainConfig.experiment_name)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return ActionHaltGateTrainConfig(
        action_root_glob=args.action_root_glob,
        halt_root_glob=args.halt_root_glob,
        output_dir=args.output_dir,
        heldout_task=args.heldout_task,
        seed=args.seed,
        train_split=args.train_split,
        valid_split=args.valid_split,
        eval_split=args.eval_split,
        validation_source=args.validation_source,
        internal_valid_fraction=args.internal_valid_fraction,
        boundary_valid_positive_fraction=args.boundary_valid_positive_fraction,
        boundary_valid_mismatch_fraction=args.boundary_valid_mismatch_fraction,
        boundary_valid_negative_fraction=args.boundary_valid_negative_fraction,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        objective=args.objective,
        rescue_loss_weight=args.rescue_loss_weight,
        false_defer_block_weight=args.false_defer_block_weight,
        utility_correct_reward=args.utility_correct_reward,
        utility_early_wrong_penalty=args.utility_early_wrong_penalty,
        utility_block_cost=args.utility_block_cost,
        utility_mismatch_penalty=args.utility_mismatch_penalty,
        utility_weight_scale=args.utility_weight_scale,
        utility_temperature=args.utility_temperature,
        checkpoint_selection=args.checkpoint_selection,
        valid_interval=args.valid_interval,
        num_workers=args.num_workers,
        device=args.device,
        swanlab_mode=args.swanlab_mode,
        experiment_name=args.experiment_name,
        dry_run=args.dry_run,
    )


def main() -> None:
    summary = train_action_halt_gate(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
