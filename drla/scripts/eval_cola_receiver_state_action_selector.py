"""Evaluate a receiver-state action selector checkpoint locally.

This evaluator restores the locked split, rebuilds structured receiver-state
features, loads a trained selector, and selects risk/coverage thresholds on the
valid split before reporting held-out test metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from drla.scripts.train_cola_receiver_state_action_selector import (
    ReceiverStateActionSelector,
    ReceiverStateActionSelectorConfig,
    build_state_action_data,
    evaluate_outputs,
    mean,
)
from drla.scripts.train_cola_readiness_model import resolve_device
from drla.tracking import require_swanlab_disabled_for_non_training


def main() -> None:
    summary = eval_action_selector(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-precision-values", default="0.55,0.60,0.65")
    parser.add_argument("--coverage-values", default="0.10,0.25,0.50")
    parser.add_argument("--swanlab-mode", default="disabled")
    return vars(parser.parse_args())


def eval_action_selector(args: dict[str, Any]) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        str(args["swanlab_mode"]),
        script_kind="P2-E receiver-state action selector evaluator",
    )
    output_dir = Path(args["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args["checkpoint"], map_location="cpu")
    train_config = ReceiverStateActionSelectorConfig(**checkpoint["config"])
    eval_config = replace(train_config, device=str(args["device"]), swanlab_mode="disabled")
    data = build_state_action_data(eval_config)
    datasets = make_eval_datasets(data, checkpoint["norm_stats"])
    device = resolve_device(str(args["device"]))
    model = ReceiverStateActionSelector(
        input_dim=checkpoint["metadata"]["input_dim"],
        hidden_dim=train_config.hidden_dim,
        dropout=train_config.dropout,
        sender_count=checkpoint["metadata"]["sender_count"],
        residual_fuser_logits=train_config.sender_output_mode == "residual_fuser",
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    outputs = {}
    aggregate = {}
    for split in ["valid", "test"]:
        loader = DataLoader(datasets[split], batch_size=int(args["batch_size"]), shuffle=False, num_workers=0)
        outputs[split] = collect_outputs(model, loader, device)
        aggregate[split] = evaluate_outputs(
            outputs[split]["sender_logits"],
            outputs[split]["any_prob"],
            outputs[split]["best_score_pred"],
            data[split],
        )

    policy_rows = evaluate_valid_selected_policies(
        valid_outputs=outputs["valid"],
        test_outputs=outputs["test"],
        valid_data=data["valid"],
        test_data=data["test"],
        target_precision_values=parse_float_list(str(args["target_precision_values"])),
        coverage_values=parse_float_list(str(args["coverage_values"])),
    )
    predictions = build_predictions(outputs["test"], data["test"])
    artifacts = write_outputs(output_dir, args, aggregate, policy_rows, predictions)
    summary = {
        "created_at": int(time.time()),
        "checkpoint": str(args["checkpoint"]),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_metric": checkpoint.get("metric"),
        "config": args,
        "train_config": asdict(train_config),
        "split_sizes": {split: int(data[split]["features"].shape[0]) for split in ["train", "valid", "test"]},
        "aggregate": aggregate,
        "policy_metrics": policy_rows,
        "artifacts": artifacts,
        "interpretation": (
            "Local-only locked evaluation. Thresholds are selected on valid and "
            "reported on held-out test. Training-script self-gate metrics should "
            "not be used as risk-control evidence."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def make_eval_datasets(data: dict[str, Any], norm_stats: dict[str, torch.Tensor]) -> dict[str, TensorDataset]:
    mean_value = norm_stats["feature_mean"]
    std_value = norm_stats["feature_std"].clamp_min(1e-6)
    out = {}
    for split in ["train", "valid", "test"]:
        features = (data[split]["features"] - mean_value) / std_value
        out[split] = TensorDataset(features, data[split]["labels"], data[split]["target_scores"], data[split]["residual_logits"])
    return out


@torch.no_grad()
def collect_outputs(model: ReceiverStateActionSelector, loader: DataLoader, device: torch.device) -> dict[str, torch.Tensor]:
    sender_logits = []
    any_probs = []
    score_preds = []
    for batch in loader:
        batch = [item.to(device) for item in batch]
        out = model(batch[0], batch[3])
        sender_logits.append(out["sender_logits"].cpu())
        any_probs.append(torch.sigmoid(out["any_logit"]).cpu())
        score_preds.append(torch.sigmoid(out["best_score_logit"]).cpu())
    return {
        "sender_logits": torch.cat(sender_logits),
        "any_prob": torch.cat(any_probs),
        "best_score_pred": torch.cat(score_preds),
    }


def evaluate_valid_selected_policies(
    *,
    valid_outputs: dict[str, torch.Tensor],
    test_outputs: dict[str, torch.Tensor],
    valid_data: dict[str, Any],
    test_data: dict[str, Any],
    target_precision_values: list[float],
    coverage_values: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in target_precision_values:
        threshold = choose_threshold_for_precision(valid_outputs["any_prob"], any_target(valid_data), target)
        rows.append(
            {
                "selection_mode": "target_any_precision",
                "target_value": target,
                "threshold": threshold,
                **prefixed("valid", gate_metrics(valid_outputs, valid_data, threshold)),
                **prefixed("test", gate_metrics(test_outputs, test_data, threshold)),
            }
        )
    for coverage in coverage_values:
        threshold = choose_threshold_for_coverage(valid_outputs["any_prob"], coverage)
        rows.append(
            {
                "selection_mode": "target_coverage",
                "target_value": coverage,
                "threshold": threshold,
                **prefixed("valid", gate_metrics(valid_outputs, valid_data, threshold)),
                **prefixed("test", gate_metrics(test_outputs, test_data, threshold)),
            }
        )
    return rows


def gate_metrics(outputs: dict[str, torch.Tensor], split_data: dict[str, Any], threshold: float) -> dict[str, float]:
    labels = split_data["labels"]
    target_scores = split_data["target_scores"]
    rows = split_data["baseline_rows"]
    selected = outputs["sender_logits"].argmax(dim=1)
    row_ids = torch.arange(labels.shape[0])
    selected_correct = labels[row_ids, selected].float()
    selected_score = target_scores[row_ids, selected].float()
    target = any_target(split_data)
    mask = outputs["any_prob"] >= threshold
    selected_rows = int(mask.sum().item())
    fallback_first_acc = []
    fallback_first_score = []
    text_mixed_acc = []
    text_mixed_score = []
    for idx, row in enumerate(rows):
        if bool(mask[idx].item()):
            fallback_first_acc.append(float(selected_correct[idx].item()))
            fallback_first_score.append(float(selected_score[idx].item()))
            text_mixed_acc.append(float(selected_correct[idx].item()))
            text_mixed_score.append(float(selected_score[idx].item()))
        else:
            fallback_first_acc.append(float(row["first_correct"]))
            fallback_first_score.append(float(row["first_score"]))
            text_mixed_acc.append(float(row["text_majority_correct"]))
            text_mixed_score.append(float(row["text_majority_score"]))
    if selected_rows:
        accepted_acc = float(selected_correct[mask].mean().item())
        accepted_score = float(selected_score[mask].mean().item())
        any_precision = float(target[mask].mean().item())
    else:
        accepted_acc = 0.0
        accepted_score = 0.0
        any_precision = 0.0
    return {
        "selected_count": float(selected_rows),
        "coverage": selected_rows / len(rows) if rows else 0.0,
        "any_precision": any_precision,
        "accepted_selector_accuracy": accepted_acc,
        "accepted_selector_score": accepted_score,
        "fallback_first_accuracy": mean(fallback_first_acc),
        "fallback_first_score": mean(fallback_first_score),
        "text_mixed_fallback_accuracy": mean(text_mixed_acc),
        "text_mixed_fallback_score": mean(text_mixed_score),
    }


def build_predictions(outputs: dict[str, torch.Tensor], split_data: dict[str, Any]) -> list[dict[str, Any]]:
    labels = split_data["labels"]
    target_scores = split_data["target_scores"]
    selected = outputs["sender_logits"].argmax(dim=1)
    probs = torch.softmax(outputs["sender_logits"], dim=1)
    rows = []
    for idx, row in enumerate(split_data["baseline_rows"]):
        selected_idx = int(selected[idx].item())
        rows.append(
            {
                **row,
                "selector_selected_index": selected_idx,
                "selector_selected_correct": float(labels[idx, selected_idx].item()),
                "selector_selected_score": float(target_scores[idx, selected_idx].item()),
                "selector_any_prob": float(outputs["any_prob"][idx].item()),
                "selector_best_score_pred": float(outputs["best_score_pred"][idx].item()),
                "selector_sender_probs": [float(value) for value in probs[idx].tolist()],
            }
        )
    return rows


def any_target(split_data: dict[str, Any]) -> torch.Tensor:
    return (split_data["labels"].sum(dim=1) > 0).float()


def choose_threshold_for_precision(prob: torch.Tensor, target: torch.Tensor, target_precision: float) -> float:
    thresholds = sorted({float(value) for value in prob.tolist()}, reverse=True)
    best = 1.0
    best_coverage = -1.0
    for threshold in thresholds:
        mask = prob >= threshold
        if int(mask.sum().item()) == 0:
            continue
        precision = float(target[mask].mean().item())
        coverage = float(mask.float().mean().item())
        if precision >= target_precision and coverage > best_coverage:
            best = threshold
            best_coverage = coverage
    return float(best)


def choose_threshold_for_coverage(prob: torch.Tensor, target_coverage: float) -> float:
    values = sorted((float(value) for value in prob.tolist()), reverse=True)
    if not values:
        return 1.0
    k = max(1, min(len(values), int(round(target_coverage * len(values)))))
    return float(values[k - 1])


def write_outputs(
    output_dir: Path,
    args: dict[str, Any],
    aggregate: dict[str, dict[str, float]],
    policy_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, str]:
    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as handle:
        now = int(time.time())
        for split, metrics in aggregate.items():
            handle.write(json.dumps({"created_at": now, "kind": "aggregate", "split": split, "metrics": metrics}, sort_keys=True) + "\n")
        for row in policy_rows:
            handle.write(json.dumps({"created_at": now, "kind": "policy", "metrics": row}, sort_keys=True) + "\n")
    policy_path = output_dir / "policy_metrics.csv"
    write_csv(policy_path, policy_rows)
    predictions_path = output_dir / "test_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(args, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "summary_json": str(output_dir / "summary.json"),
        "metrics_jsonl": str(metrics_path),
        "policy_metrics_csv": str(policy_path),
        "test_predictions_jsonl": str(predictions_path),
        "config_json": str(config_path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    main()
