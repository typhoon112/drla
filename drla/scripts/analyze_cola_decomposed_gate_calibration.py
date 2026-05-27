"""Analyze calibrated selectors for decomposed action-to-halt gates.

This script is local-only.  It does not train a model or create SwanLab runs.
It reloads existing decomposed_expected_utility checkpoints, sweeps utility
weights and constrained head thresholds on source validation cases, diagnoses
per-head reliability and rare-event capture, and applies the selected policies
to the held-out test split.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch

from drla.scripts.aggregate_cola_action_halt_gate import (
    ActionHaltGateConfig,
    find_eval_pairs,
    policy_metrics,
)
from drla.scripts.aggregate_cola_action_halt_gate_loto import aggregate_policies
from drla.scripts.train_cola_action_halt_gate import (
    FEATURE_NAMES,
    ActionHaltGate,
    ActionHaltGateTrainConfig,
    build_feature_bundle,
    load_cases,
    normalize,
    split_cases_by_boundary_group,
    split_cases_by_group,
)
from drla.scripts.train_cola_readiness_model import resolve_device


DEFAULT_SUMMARY_GLOB = (
    "/data1/luyifei/drla/outputs/cola_action_halt_gate/"
    "official8_full_b64_bs12_loto_*_seed20260527_decomposed_expectedutility_boundaryvalid_policycost_besteval/"
    "summary.json"
)
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/cola_experiment_summaries/"
    "official8_full_b64_bs12_p1_decomposed_expectedutility_calibrated_partial4_seed20260527"
)
QUANTILES = [0.0, 0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
CONSTRAINED_QUANTILES = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
HEAD_TARGETS = [
    ("rescue_loss", "rescue_target"),
    ("mismatch_rescue", "mismatch_rescue_target"),
    ("introduced_loss", "harm_target"),
    ("introduced_mismatch", "mismatch_target"),
]


def analyze_decomposed_gate_calibration(
    *,
    summary_glob: str,
    output_dir: str,
    device_arg: str,
) -> dict[str, Any]:
    summary_paths = sorted(Path("/").glob(summary_glob.lstrip("/")))
    if not summary_paths:
        raise FileNotFoundError(f"no summaries matched {summary_glob}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_arg)
    task_summaries = []
    for summary_path in summary_paths:
        task_summary = analyze_task(summary_path, output_path, device)
        task_summaries.append(task_summary)
    aggregate = aggregate_policies(task_summaries)
    summary = {
        "created_at": int(time.time()),
        "config": {
            "summary_glob": summary_glob,
            "output_dir": output_dir,
            "device": str(device),
            "weight_grid": weight_grid(),
        },
        "num_summaries": len(task_summaries),
        "num_samples": sum(int(item["test_policies"]["action"]["num_samples"]) for item in task_summaries),
        "aggregate": aggregate,
        "task_summaries": [
            {
                "task": item["config"]["heldout_task"],
                "summary_path": item["_summary_path"],
                "source_summary_path": item["source_summary_path"],
                "swanlab_run_id": item.get("swanlab_run_id"),
                "selected_weights": item["selected_weights"],
                "data_summary": item["data_summary"],
                "head_diagnostics": item["head_diagnostics"],
            }
            for item in task_summaries
        ],
        "readout": build_readout(aggregate),
        "head_diagnostic_readout": build_head_diagnostic_readout(task_summaries),
        "note": (
            "Local-only selector analysis over frozen decomposed heads; no optimizer was created "
            "and no SwanLab run was started."
        ),
    }
    (output_path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_task_csv(output_path / "task_policy_summary.csv", task_summaries)
    return summary


def analyze_task(summary_path: Path, output_dir: Path, device: torch.device) -> dict[str, Any]:
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = config_from_summary(source_summary)
    print(f"analyzing {config.heldout_task}", flush=True)
    if config.objective != "decomposed_expected_utility":
        raise ValueError(f"{summary_path} is not a decomposed_expected_utility run")
    valid_cases, test_cases = load_valid_and_test_cases(config)
    checkpoint_path = Path(source_summary["artifacts"]["best_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = ActionHaltGate(
        len(FEATURE_NAMES),
        config.hidden_dim,
        config.dropout,
        output_dim=5,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    valid_bundle = build_feature_bundle(valid_cases, config)
    test_bundle = build_feature_bundle(test_cases, config)
    valid_x = normalize(valid_bundle["x"], checkpoint["norm_stats"])
    test_x = normalize(test_bundle["x"], checkpoint["norm_stats"])
    valid_probs = predict_head_probs(model, valid_x, device)
    test_probs = predict_head_probs(model, test_x, device)
    valid_stats = build_case_stats(valid_cases)
    test_stats = build_case_stats(test_cases)

    action_valid_blocks = policy_metrics(valid_cases, "action")["avg_blocks"]
    action_test_blocks = policy_metrics(test_cases, "action")["avg_blocks"]
    selection = select_calibrated_policies(
        valid_cases=valid_cases,
        test_cases=test_cases,
        valid_stats=valid_stats,
        test_stats=test_stats,
        valid_probs=valid_probs,
        test_probs=test_probs,
        action_valid_blocks=action_valid_blocks,
        action_test_blocks=action_test_blocks,
    )
    policies = {
        "action": policy_metrics(test_cases, "action"),
        "halt_original": policy_metrics(test_cases, "halt_original"),
        "causal_always_defer_after_action": policy_metrics(test_cases, "fallback"),
        **selection["test_policies"],
    }
    task_output = {
        "created_at": int(time.time()),
        "config": {"heldout_task": config.heldout_task},
        "source_summary_path": str(summary_path),
        "swanlab_run_id": source_summary.get("swanlab_run_id"),
        "checkpoint_path": str(checkpoint_path),
        "data_summary": {
            "num_valid_cases": len(valid_cases),
            "num_test_cases": len(test_cases),
            "action_valid_avg_blocks": action_valid_blocks,
            "action_test_avg_blocks": action_test_blocks,
        },
        "head_diagnostics": {
            "valid": head_diagnostics(valid_probs, valid_bundle),
            "test": head_diagnostics(test_probs, test_bundle),
        },
        "selected_weights": selection["selected_weights"],
        "test_policies": policies,
    }
    task_path = output_dir / f"{config.heldout_task}_summary.json"
    task_path.write_text(
        json.dumps(task_output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    task_output["_summary_path"] = str(task_path)
    return task_output


def config_from_summary(summary: dict[str, Any]) -> ActionHaltGateTrainConfig:
    valid_fields = {field.name for field in fields(ActionHaltGateTrainConfig)}
    values = {key: value for key, value in summary["config"].items() if key in valid_fields}
    return ActionHaltGateTrainConfig(**values)


def load_valid_and_test_cases(config: ActionHaltGateTrainConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = find_eval_pairs(
        ActionHaltGateConfig(
            action_root_glob=config.action_root_glob,
            halt_root_glob=config.halt_root_glob,
            output_dir=config.output_dir,
        )
    )
    train_pairs = [pair for pair in pairs if pair["task"] != config.heldout_task]
    eval_pairs = [pair for pair in pairs if pair["task"] == config.heldout_task]
    source_train_cases = load_cases(train_pairs, split=config.train_split)
    if config.validation_source == "train_boundary_stratified":
        _, valid_cases = split_cases_by_boundary_group(source_train_cases, config)
    elif config.validation_source == "train_stratified":
        _, valid_cases = split_cases_by_group(
            source_train_cases,
            valid_fraction=config.internal_valid_fraction,
            seed=config.seed,
        )
    else:
        valid_cases = load_cases(train_pairs, split=config.valid_split)
    test_cases = load_cases(eval_pairs, split=config.eval_split)
    return valid_cases, test_cases


@torch.no_grad()
def predict_head_probs(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    chunks = []
    for start in range(0, x.shape[0], 65536):
        logits = model(x[start : start + 65536].to(device)).cpu()
        if logits.ndim != 2 or logits.shape[1] != 5:
            raise ValueError("expected five decomposed output heads")
        chunks.append(torch.sigmoid(logits))
    return torch.cat(chunks, dim=0)


def select_calibrated_policies(
    *,
    valid_cases: list[dict[str, Any]],
    test_cases: list[dict[str, Any]],
    valid_stats: dict[str, torch.Tensor],
    test_stats: dict[str, torch.Tensor],
    valid_probs: torch.Tensor,
    test_probs: torch.Tensor,
    action_valid_blocks: float,
    action_test_blocks: float,
) -> dict[str, Any]:
    all_rows = []
    for weights in weight_grid():
        valid_scores = score_from_weights(valid_probs, weights)
        test_scores = score_from_weights(test_probs, weights)
        for threshold in score_thresholds(valid_scores):
            valid_row = metrics_for_threshold_fast(valid_stats, valid_scores, threshold)
            test_row = metrics_for_threshold_fast(test_stats, test_scores, threshold)
            valid_row.update(weights)
            test_row.update(weights)
            all_rows.append({"valid": valid_row, "test": test_row})
    selected = {
        "source_valid_safety_selected_gate": select_pair(all_rows, mode="safety", action_avg_blocks=action_valid_blocks),
        "source_valid_cost_selected_gate": select_pair(all_rows, mode="cost", action_avg_blocks=action_valid_blocks),
        "source_valid_cost_limited_selected_gate": select_pair(
            all_rows,
            mode="cost_limited",
            action_avg_blocks=action_valid_blocks,
        ),
        "best_test_gate_by_loss": select_pair_by_test(all_rows, mode="safety", action_avg_blocks=action_test_blocks),
        "best_test_gate_under_action_plus_0p10_blocks": select_pair_by_test(
            all_rows,
            mode="cost_limited",
            action_avg_blocks=action_test_blocks,
        ),
    }
    selected.update(
        select_constrained_head_policies(
            valid_stats=valid_stats,
            test_stats=test_stats,
            valid_probs=valid_probs,
            test_probs=test_probs,
            action_valid_blocks=action_valid_blocks,
            action_test_blocks=action_test_blocks,
        )
    )
    return {
        "test_policies": {name: pair["test"] for name, pair in selected.items()},
        "selected_weights": {name: weights_from_row(pair["valid"]) for name, pair in selected.items()},
    }


def build_case_stats(cases: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    action = [case["action"] for case in cases]
    fallback = [case["fallback"] for case in cases]
    return {
        "action_selected_block": tensor_field(action, "selected_block"),
        "fallback_selected_block": tensor_field(fallback, "selected_block"),
        "final_block": tensor_field(action, "final_block"),
        "action_selected_correct": bool_tensor_field(action, "selected_correct"),
        "fallback_selected_correct": bool_tensor_field(fallback, "selected_correct"),
        "final_correct": bool_tensor_field(action, "final_correct"),
        "prediction_stability_correct": bool_tensor_field(action, "prediction_stability_correct"),
        "action_loss_vs_final": bool_tensor_field(action, "loss_vs_final"),
        "fallback_loss_vs_final": bool_tensor_field(fallback, "loss_vs_final"),
        "action_gain_vs_final": bool_tensor_field(action, "gain_vs_final"),
        "fallback_gain_vs_final": bool_tensor_field(fallback, "gain_vs_final"),
        "action_loss_vs_prediction_stability": bool_tensor_field(action, "loss_vs_prediction_stability"),
        "fallback_loss_vs_prediction_stability": bool_tensor_field(fallback, "loss_vs_prediction_stability"),
        "action_gain_vs_prediction_stability": bool_tensor_field(action, "gain_vs_prediction_stability"),
        "fallback_gain_vs_prediction_stability": bool_tensor_field(fallback, "gain_vs_prediction_stability"),
        "action_prediction_mismatch_vs_final": bool_tensor_field(action, "_prediction_mismatch_vs_final"),
        "fallback_prediction_mismatch_vs_final": bool_tensor_field(fallback, "_prediction_mismatch_vs_final"),
        "action_prediction_mismatch_vs_prediction_stability": bool_tensor_field(
            action,
            "_prediction_mismatch_vs_prediction_stability",
        ),
        "fallback_prediction_mismatch_vs_prediction_stability": bool_tensor_field(
            fallback,
            "_prediction_mismatch_vs_prediction_stability",
        ),
    }


def tensor_field(items: list[dict[str, Any]], key: str) -> torch.Tensor:
    return torch.tensor([float(item[key]) for item in items], dtype=torch.float32)


def bool_tensor_field(items: list[dict[str, Any]], key: str) -> torch.Tensor:
    return torch.tensor([bool(item[key]) for item in items], dtype=torch.bool)


def head_diagnostics(probs: torch.Tensor, bundle: dict[str, torch.Tensor]) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for head_idx, (head_name, target_name) in enumerate(HEAD_TARGETS):
        target = bundle[target_name].float()
        prob = probs[:, head_idx].float()
        diagnostics[head_name] = binary_head_diagnostics(prob, target)
    diagnostics["extra_block_cost"] = cost_head_diagnostics(probs[:, 4].float(), bundle["cost_target"].float())
    return diagnostics


def select_constrained_head_policies(
    *,
    valid_stats: dict[str, torch.Tensor],
    test_stats: dict[str, torch.Tensor],
    valid_probs: torch.Tensor,
    test_probs: torch.Tensor,
    action_valid_blocks: float,
    action_test_blocks: float,
) -> dict[str, dict[str, dict[str, Any]]]:
    rows = []
    for rescue_threshold in coarse_thresholds(valid_probs[:, 0], high_is_positive=True):
        for mismatch_threshold in coarse_thresholds(valid_probs[:, 1], high_is_positive=True):
            for cost_threshold in coarse_thresholds(valid_probs[:, 4], high_is_positive=False):
                valid_mask = constrained_defer_mask(
                    valid_stats,
                    valid_probs,
                    rescue_threshold=rescue_threshold,
                    mismatch_threshold=mismatch_threshold,
                    cost_threshold=cost_threshold,
                )
                test_mask = constrained_defer_mask(
                    test_stats,
                    test_probs,
                    rescue_threshold=rescue_threshold,
                    mismatch_threshold=mismatch_threshold,
                    cost_threshold=cost_threshold,
                )
                valid_row = metrics_for_defer_mask(valid_stats, valid_mask)
                test_row = metrics_for_defer_mask(test_stats, test_mask)
                params = {
                    "rescue_threshold": rescue_threshold,
                    "mismatch_threshold": mismatch_threshold,
                    "cost_threshold": cost_threshold,
                }
                valid_row.update(params)
                test_row.update(params)
                rows.append({"valid": valid_row, "test": test_row})
    return {
        "source_valid_constrained_safety_gate": select_pair(rows, mode="safety", action_avg_blocks=action_valid_blocks),
        "source_valid_constrained_cost_limited_gate": select_pair(
            rows,
            mode="cost_limited",
            action_avg_blocks=action_valid_blocks,
        ),
        "best_test_constrained_under_action_plus_0p10_blocks": select_pair_by_test(
            rows,
            mode="cost_limited",
            action_avg_blocks=action_test_blocks,
        ),
    }


def coarse_thresholds(values: torch.Tensor, *, high_is_positive: bool) -> list[float]:
    finite = values[torch.isfinite(values)].float()
    if finite.numel() == 0:
        return [0.0]
    quantiles = torch.tensor(CONSTRAINED_QUANTILES, dtype=torch.float32)
    thresholds = torch.quantile(finite, quantiles).tolist()
    thresholds.append(0.5)
    if high_is_positive:
        thresholds.append(float(finite.min().item()) - 1e-6)
        thresholds.append(float(finite.max().item()) + 1e-6)
    else:
        thresholds.append(float(finite.max().item()) + 1e-6)
    return sorted({round(float(value), 8) for value in thresholds})


def constrained_defer_mask(
    stats: dict[str, torch.Tensor],
    probs: torch.Tensor,
    *,
    rescue_threshold: float,
    mismatch_threshold: float,
    cost_threshold: float,
) -> torch.Tensor:
    can_defer = stats["action_selected_block"] < stats["final_block"]
    wants_defer = (probs[:, 0] >= rescue_threshold) | (probs[:, 1] >= mismatch_threshold)
    predicted_affordable = probs[:, 4] <= cost_threshold
    return can_defer & wants_defer & predicted_affordable


def binary_head_diagnostics(prob: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    target = target.float()
    pred = (prob >= 0.5).float()
    positives = int(target.sum().item())
    tp = int(((pred == 1) & (target == 1)).sum().item())
    fp = int(((pred == 1) & (target == 0)).sum().item())
    fn = int(((pred == 0) & (target == 1)).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "num_samples": int(target.numel()),
        "positive_count": positives,
        "positive_rate": positives / max(int(target.numel()), 1),
        "mean_pred": float(prob.mean().item()),
        "brier": float(torch.mean((prob - target) ** 2).item()),
        "ece_10": calibration_error(prob, target, n_bins=10),
        "mce_10": calibration_error(prob, target, n_bins=10, maximum=True),
        "auroc": binary_auroc_local(prob, target),
        "auprc": binary_auprc_local(prob, target),
        "threshold_0p5_precision": precision,
        "threshold_0p5_recall": recall,
        "threshold_0p5_predicted_positive_rate": float(pred.mean().item()),
        "top_fraction_capture": top_fraction_capture(prob, target),
    }


def cost_head_diagnostics(prob: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    diff = prob - target
    centered_prob = prob - prob.mean()
    centered_target = target - target.mean()
    denom = torch.sqrt(torch.sum(centered_prob**2) * torch.sum(centered_target**2)).clamp_min(1e-12)
    return {
        "num_samples": int(target.numel()),
        "mean_target": float(target.mean().item()),
        "mean_pred": float(prob.mean().item()),
        "mae": float(torch.mean(torch.abs(diff)).item()),
        "rmse": float(torch.sqrt(torch.mean(diff**2)).item()),
        "pearson": float((torch.sum(centered_prob * centered_target) / denom).item()),
    }


def calibration_error(prob: torch.Tensor, target: torch.Tensor, *, n_bins: int, maximum: bool = False) -> float:
    prob = prob.float()
    target = target.float()
    total = max(int(prob.numel()), 1)
    gaps = []
    weighted = 0.0
    for index in range(n_bins):
        lower = index / n_bins
        upper = (index + 1) / n_bins
        if index == 0:
            mask = (prob >= lower) & (prob <= upper)
        else:
            mask = (prob > lower) & (prob <= upper)
        count = int(mask.sum().item())
        if count == 0:
            continue
        gap = abs(float(prob[mask].mean().item()) - float(target[mask].mean().item()))
        gaps.append(gap)
        weighted += gap * count / total
    if maximum:
        return max(gaps) if gaps else 0.0
    return weighted


def top_fraction_capture(prob: torch.Tensor, target: torch.Tensor) -> dict[str, dict[str, Any]]:
    positives = int(target.sum().item())
    result = {}
    if prob.numel() == 0:
        return result
    order = torch.argsort(prob, descending=True)
    sorted_target = target[order].float()
    sorted_prob = prob[order].float()
    for fraction in [0.001, 0.005, 0.01, 0.05, 0.10]:
        k = max(1, int(round(float(prob.numel()) * fraction)))
        hits = int(sorted_target[:k].sum().item())
        result[f"top_{fraction:g}"] = {
            "k": k,
            "positive_hits": hits,
            "recall": hits / max(positives, 1),
            "precision": hits / k,
            "min_score": float(sorted_prob[k - 1].item()),
        }
    return result


def binary_auroc_local(prob: torch.Tensor, target: torch.Tensor) -> float:
    target = target.float()
    positives = int(target.sum().item())
    negatives = int(target.numel() - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(prob)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(prob) + 1, dtype=torch.float32)
    pos_rank_sum = ranks[target.bool()].sum()
    auc = (pos_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return float(auc.item())


def binary_auprc_local(prob: torch.Tensor, target: torch.Tensor) -> float:
    target = target.float()
    positives = target.sum()
    if positives <= 0:
        return float("nan")
    order = torch.argsort(prob, descending=True)
    sorted_target = target[order]
    tp = torch.cumsum(sorted_target, dim=0)
    fp = torch.cumsum(1 - sorted_target, dim=0)
    precision = tp / (tp + fp).clamp_min(1e-6)
    recall = tp / positives
    recall_prev = torch.cat([torch.zeros(1), recall[:-1]])
    area = torch.sum((recall - recall_prev) * precision)
    return float(area.item())


def metrics_for_threshold_fast(
    stats: dict[str, torch.Tensor],
    scores: torch.Tensor,
    threshold: float,
) -> dict[str, Any]:
    can_defer = stats["action_selected_block"] < stats["final_block"]
    use_fallback = can_defer & (scores >= threshold)
    row = metrics_for_defer_mask(stats, use_fallback)
    row["threshold"] = threshold
    return row


def metrics_for_defer_mask(
    stats: dict[str, torch.Tensor],
    use_fallback: torch.Tensor,
) -> dict[str, Any]:
    n = int(use_fallback.numel())
    selected_block = torch.where(use_fallback, stats["fallback_selected_block"], stats["action_selected_block"])
    selected_correct = torch.where(
        use_fallback,
        stats["fallback_selected_correct"],
        stats["action_selected_correct"],
    )
    loss_final = torch.where(use_fallback, stats["fallback_loss_vs_final"], stats["action_loss_vs_final"])
    gain_final = torch.where(use_fallback, stats["fallback_gain_vs_final"], stats["action_gain_vs_final"])
    loss_stability = torch.where(
        use_fallback,
        stats["fallback_loss_vs_prediction_stability"],
        stats["action_loss_vs_prediction_stability"],
    )
    gain_stability = torch.where(
        use_fallback,
        stats["fallback_gain_vs_prediction_stability"],
        stats["action_gain_vs_prediction_stability"],
    )
    mismatch_final = torch.where(
        use_fallback,
        stats["fallback_prediction_mismatch_vs_final"],
        stats["action_prediction_mismatch_vs_final"],
    )
    mismatch_stability = torch.where(
        use_fallback,
        stats["fallback_prediction_mismatch_vs_prediction_stability"],
        stats["action_prediction_mismatch_vs_prediction_stability"],
    )
    final_blocks = stats["final_block"].float().mean().item()
    avg_blocks = selected_block.float().mean().item()
    accuracy = selected_correct.float().mean().item()
    final_accuracy = stats["final_correct"].float().mean().item()
    stability_accuracy = stats["prediction_stability_correct"].float().mean().item()
    defer_count = int(use_fallback.sum().item())
    rescued_action_losses = int((stats["action_loss_vs_final"] & ~loss_final).sum().item())
    introduced_losses = int((~stats["action_loss_vs_final"] & loss_final).sum().item())
    mismatch_final_count = int(mismatch_final.sum().item())
    mismatch_stability_count = int(mismatch_stability.sum().item())
    return {
        "num_samples": n,
        "accuracy": accuracy,
        "fixed_final_accuracy": final_accuracy,
        "prediction_stability_accuracy": stability_accuracy,
        "accuracy_drop_vs_final": final_accuracy - accuracy,
        "accuracy_drop_vs_prediction_stability": stability_accuracy - accuracy,
        "avg_blocks": avg_blocks,
        "fixed_final_avg_blocks": final_blocks,
        "block_saving_vs_final": final_blocks - avg_blocks,
        "block_saving_fraction_vs_final": (final_blocks - avg_blocks) / max(final_blocks, 1e-9),
        "losses_vs_final": int(loss_final.sum().item()),
        "gains_vs_final": int(gain_final.sum().item()),
        "losses_vs_prediction_stability": int(loss_stability.sum().item()),
        "gains_vs_prediction_stability": int(gain_stability.sum().item()),
        "prediction_mismatch_vs_final": mismatch_final_count,
        "prediction_mismatch_rate_vs_final": mismatch_final_count / max(n, 1),
        "prediction_mismatch_vs_prediction_stability": mismatch_stability_count,
        "prediction_mismatch_rate_vs_prediction_stability": mismatch_stability_count / max(n, 1),
        "defer_count": defer_count,
        "defer_rate": defer_count / max(n, 1),
        "rescued_action_losses": rescued_action_losses,
        "introduced_losses_vs_action": introduced_losses,
    }


def weight_grid() -> list[dict[str, float]]:
    weights = []
    for correctness_weight in [4.0, 8.0, 11.0, 16.0]:
        for mismatch_weight in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]:
            for cost_weight in [0.05, 0.1, 0.25, 0.5, 1.0]:
                weights.append(
                    {
                        "correctness_weight": correctness_weight,
                        "mismatch_weight": mismatch_weight,
                        "cost_weight": cost_weight,
                    }
                )
    return weights


def score_from_weights(probs: torch.Tensor, weights: dict[str, float]) -> torch.Tensor:
    correctness = weights["correctness_weight"]
    mismatch = weights["mismatch_weight"]
    cost = weights["cost_weight"]
    return (
        probs[:, 0] * correctness
        + probs[:, 1] * mismatch
        - probs[:, 2] * correctness
        - probs[:, 3] * mismatch
        - probs[:, 4] * cost
    )


def score_thresholds(scores: torch.Tensor) -> list[float]:
    finite_scores = scores[torch.isfinite(scores)]
    if finite_scores.numel() == 0:
        return [0.0]
    quantiles = torch.tensor(QUANTILES, dtype=torch.float32)
    values = torch.quantile(finite_scores.float(), quantiles).tolist()
    values.extend([0.0, float(finite_scores.min().item()) - 1e-6, float(finite_scores.max().item()) + 1e-6])
    return sorted({round(float(value), 8) for value in values})


def select_pair(
    pairs: list[dict[str, dict[str, Any]]],
    *,
    mode: str,
    action_avg_blocks: float,
) -> dict[str, dict[str, Any]]:
    eligible = pairs
    if mode == "cost_limited":
        eligible = [
            pair
            for pair in pairs
            if float(pair["valid"]["avg_blocks"]) <= action_avg_blocks + 0.10
        ]
    return min(eligible or pairs, key=lambda pair: rank_row(pair["valid"], mode="safety" if mode == "cost_limited" else mode))


def select_pair_by_test(
    pairs: list[dict[str, dict[str, Any]]],
    *,
    mode: str,
    action_avg_blocks: float,
) -> dict[str, dict[str, Any]]:
    eligible = pairs
    if mode == "cost_limited":
        eligible = [
            pair
            for pair in pairs
            if float(pair["test"]["avg_blocks"]) <= action_avg_blocks + 0.10
        ]
    return min(eligible or pairs, key=lambda pair: rank_row(pair["test"], mode="safety" if mode == "cost_limited" else mode))


def rank_row(row: dict[str, Any], *, mode: str) -> tuple[float, ...]:
    if mode == "safety":
        return (
            float(row.get("losses_vs_final", 0.0)),
            float(row.get("prediction_mismatch_vs_final", 0.0)),
            float(row.get("avg_blocks", 0.0)),
            -float(row.get("rescued_action_losses", 0.0)),
            float(row.get("threshold", 0.0)),
        )
    if mode == "cost":
        return (
            float(row.get("losses_vs_final", 0.0)),
            float(row.get("avg_blocks", 0.0)),
            float(row.get("prediction_mismatch_vs_final", 0.0)),
            -float(row.get("rescued_action_losses", 0.0)),
            float(row.get("threshold", 0.0)),
        )
    raise ValueError(f"unknown mode: {mode}")


def weights_from_row(row: dict[str, Any]) -> dict[str, float]:
    keys = [
        "correctness_weight",
        "mismatch_weight",
        "cost_weight",
        "threshold",
        "rescue_threshold",
        "mismatch_threshold",
        "cost_threshold",
    ]
    return {key: float(row[key]) for key in keys if key in row}


def build_readout(aggregate: dict[str, dict[str, Any]]) -> list[str]:
    rows = []
    for policy in [
        "action",
        "source_valid_cost_selected_gate",
        "source_valid_cost_limited_selected_gate",
        "source_valid_safety_selected_gate",
        "source_valid_constrained_cost_limited_gate",
        "source_valid_constrained_safety_gate",
        "best_test_gate_under_action_plus_0p10_blocks",
        "best_test_constrained_under_action_plus_0p10_blocks",
    ]:
        metrics = aggregate.get(policy)
        if not metrics:
            continue
        rows.append(
            (
                f"{policy}: {int(metrics.get('losses_vs_prediction_stability', 0))} losses, "
                f"{int(metrics.get('prediction_mismatch_vs_prediction_stability', 0))} mismatches, "
                f"{float(metrics.get('avg_blocks', 0.0)):.3f}/4 blocks."
            )
        )
    return rows


def build_head_diagnostic_readout(task_summaries: list[dict[str, Any]]) -> list[str]:
    rows = []
    for item in task_summaries:
        task = item["config"]["heldout_task"]
        test_heads = item["head_diagnostics"]["test"]
        rescue = test_heads["rescue_loss"]
        mismatch = test_heads["mismatch_rescue"]
        cost = test_heads["extra_block_cost"]
        rows.append(
            (
                f"{task}: rescue positives={rescue['positive_count']} "
                f"auprc={format_float(rescue['auprc'])} top5%recall="
                f"{rescue['top_fraction_capture']['top_0.05']['recall']:.3f}; "
                f"mismatch positives={mismatch['positive_count']} "
                f"auprc={format_float(mismatch['auprc'])} top5%recall="
                f"{mismatch['top_fraction_capture']['top_0.05']['recall']:.3f}; "
                f"cost_mae={cost['mae']:.3f} cost_r={cost['pearson']:.3f}."
            )
        )
    return rows


def format_float(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value:.4f}"


def write_task_csv(path: Path, task_summaries: list[dict[str, Any]]) -> None:
    import csv

    rows = []
    for item in task_summaries:
        task = item["config"]["heldout_task"]
        for policy, metrics in item["test_policies"].items():
            rows.append({"task": task, "policy": policy, **metrics})
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-glob", default=DEFAULT_SUMMARY_GLOB)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze_decomposed_gate_calibration(
        summary_glob=args.summary_glob,
        output_dir=args.output_dir,
        device_arg=args.device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
