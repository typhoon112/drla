"""Evaluate, calibrate, and ablate a P2-E latent-state verifier checkpoint.

This script is local-only.  It does not update model weights or log to SwanLab.
It restores the locked packet split, evaluates the verifier with optional input
ablations, and fits small post-hoc calibrators on the valid split before
reporting held-out test calibration.
"""

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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from drla.scripts.audit_cola_hierarchical_aggregation_potential import build_groups
from drla.scripts.audit_cola_sequential_latent_mas import load_official_scorer
from drla.scripts.train_cola_hierarchical_latent_fuser import build_tensors, make_datasets, read_jsonl, split_groups
from drla.scripts.train_cola_hierarchical_state_verifier import (
    HierarchicalStateVerifier,
    HierarchicalStateVerifierConfig,
    auroc,
    pearson,
    regression_and_binary_metrics,
)
from drla.scripts.train_cola_readiness_model import resolve_device
from drla.tracking import require_swanlab_disabled_for_non_training


ABLATIONS = [
    "full",
    "zero_latent",
    "zero_process",
    "zero_certificate",
    "zero_latent_process",
    "zero_latent_certificate",
    "zero_process_certificate",
]


def main() -> None:
    summary = eval_hierarchical_state_verifier(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--swanlab-mode", default="disabled")
    return vars(parser.parse_args())


def eval_hierarchical_state_verifier(args: dict[str, Any]) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        str(args["swanlab_mode"]),
        script_kind="P2-E hierarchical state verifier evaluator/calibrator",
    )
    output_dir = Path(args["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args["checkpoint"], map_location="cpu")
    train_config = HierarchicalStateVerifierConfig(**checkpoint["config"])
    eval_config = replace(train_config, swanlab_mode="disabled")
    scorer = load_official_scorer(Path(eval_config.acc_calc_script))
    packets = read_jsonl(Path(eval_config.packets_jsonl))
    groups = build_groups(packets, eval_config, scorer)
    if eval_config.max_groups:
        groups = groups[: eval_config.max_groups]
    splits = split_groups(groups, eval_config)
    tensors_by_split, metadata = build_tensors(groups, splits, eval_config)
    train_ds, valid_ds, test_ds, _norm_stats = make_datasets(tensors_by_split)
    loaders = {
        "valid": DataLoader(valid_ds, batch_size=int(args["batch_size"]), shuffle=False, num_workers=0),
        "test": DataLoader(test_ds, batch_size=int(args["batch_size"]), shuffle=False, num_workers=0),
    }
    device = resolve_device(str(args["device"]))
    model = build_model(checkpoint, train_config, metadata, device)

    raw_eval: dict[str, dict[str, Any]] = {}
    for split in ["valid", "test"]:
        raw_eval[split] = {}
        for ablation in ABLATIONS:
            outputs = collect_outputs(model, loaders[split], tensors_by_split[split], device, ablation)
            raw_eval[split][ablation] = outputs_to_metrics(outputs)

    valid_full = collect_outputs(model, loaders["valid"], tensors_by_split["valid"], device, "full")
    test_full = collect_outputs(model, loaders["test"], tensors_by_split["test"], device, "full")
    any_calibrator = fit_platt(valid_full["any_logit"], valid_full["any_target"])
    score_calibrator = fit_affine(valid_full["score_pred"], valid_full["best_score"])
    calibrated = {
        "valid": calibrated_metrics(valid_full, any_calibrator, score_calibrator),
        "test": calibrated_metrics(test_full, any_calibrator, score_calibrator),
        "calibrators": {
            "any_platt": any_calibrator,
            "score_affine": score_calibrator,
        },
    }
    threshold_report = threshold_calibration_report(valid_full, test_full)

    artifacts = write_outputs(output_dir, args, raw_eval, calibrated, threshold_report)
    summary = {
        "created_at": int(time.time()),
        "checkpoint": str(args["checkpoint"]),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_metric": checkpoint.get("metric"),
        "config": args,
        "train_config": asdict(train_config),
        "split_sizes": {split: len(indices) for split, indices in splits.items()},
        "raw_eval": raw_eval,
        "calibrated": calibrated,
        "threshold_report": threshold_report,
        "artifacts": artifacts,
        "interpretation": (
            "Local-only calibration and input ablation for the P2-E latent-state verifier. "
            "Ablations zero inputs at inference time; calibration is fitted on valid and "
            "reported on held-out test."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_model(
    checkpoint: dict[str, Any],
    config: HierarchicalStateVerifierConfig,
    metadata: dict[str, Any],
    device: torch.device,
) -> HierarchicalStateVerifier:
    model = HierarchicalStateVerifier(
        latent_dim=metadata["latent_dim"],
        process_dim=metadata["process_dim"],
        certificate_dim=metadata["certificate_dim"],
        max_blocks=metadata["max_blocks"],
        block_size=metadata["block_size"],
        sender_count=metadata["sender_count"],
        task_count=len(metadata["task_to_idx"]),
        d_model=config.d_model,
        attention_heads=config.attention_heads,
        inter_layers=config.inter_layers,
        sender_layers=config.sender_layers,
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def collect_outputs(
    model: HierarchicalStateVerifier,
    loader: DataLoader,
    tensors: dict[str, torch.Tensor],
    device: torch.device,
    ablation: str,
) -> dict[str, torch.Tensor]:
    any_logits = []
    score_preds = []
    for batch in loader:
        batch = [item.to(device) for item in batch]
        batch = apply_ablation(batch, ablation)
        outputs = model(*batch[:5])
        any_logits.append(outputs["any_correct_logit"].cpu())
        score_preds.append(torch.sigmoid(outputs["best_score_logit"]).cpu())
    any_logit = torch.cat(any_logits)
    score_pred = torch.cat(score_preds)
    labels = tensors["labels"]
    target_scores = tensors["target_scores"]
    return {
        "any_logit": any_logit,
        "any_prob": torch.sigmoid(any_logit),
        "score_pred": score_pred,
        "any_target": (labels.sum(dim=1) > 0).float(),
        "best_score": target_scores.max(dim=1).values.float(),
    }


def apply_ablation(batch: list[torch.Tensor], ablation: str) -> list[torch.Tensor]:
    out = list(batch)
    if "zero_latent" in ablation:
        out[0] = torch.zeros_like(out[0])
    if "zero_process" in ablation:
        out[1] = torch.zeros_like(out[1])
    if "zero_certificate" in ablation:
        out[3] = torch.zeros_like(out[3])
    return out


def outputs_to_metrics(outputs: dict[str, torch.Tensor]) -> dict[str, float]:
    metrics = regression_and_binary_metrics(
        outputs["score_pred"],
        outputs["best_score"],
        outputs["any_prob"],
        outputs["any_target"],
    )
    metrics["ece_10"] = ece(outputs["any_prob"], outputs["any_target"], bins=10)
    metrics["nll"] = float(F.binary_cross_entropy(outputs["any_prob"].clamp(1e-6, 1 - 1e-6), outputs["any_target"]).item())
    return metrics


def fit_platt(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    scale = torch.tensor(1.0, requires_grad=True)
    bias = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.LBFGS([scale, bias], lr=0.1, max_iter=100)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(scale * logits + bias, target)
        loss.backward()
        return loss

    optimizer.step(closure)
    return {"scale": float(scale.detach().item()), "bias": float(bias.detach().item())}


def fit_affine(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    x = pred.float()
    y = target.float()
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if float(denom) <= 1e-12:
        slope = torch.tensor(1.0)
    else:
        slope = ((x - x_mean) * (y - y_mean)).sum() / denom
    intercept = y_mean - slope * x_mean
    return {"slope": float(slope.item()), "intercept": float(intercept.item())}


def calibrated_metrics(
    outputs: dict[str, torch.Tensor],
    any_calibrator: dict[str, float],
    score_calibrator: dict[str, float],
) -> dict[str, float]:
    any_prob = torch.sigmoid(any_calibrator["scale"] * outputs["any_logit"] + any_calibrator["bias"])
    score_pred = (score_calibrator["slope"] * outputs["score_pred"] + score_calibrator["intercept"]).clamp(0.0, 1.0)
    metrics = regression_and_binary_metrics(score_pred, outputs["best_score"], any_prob, outputs["any_target"])
    metrics["ece_10"] = ece(any_prob, outputs["any_target"], bins=10)
    metrics["nll"] = float(F.binary_cross_entropy(any_prob.clamp(1e-6, 1 - 1e-6), outputs["any_target"]).item())
    return metrics


def threshold_calibration_report(valid: dict[str, torch.Tensor], test: dict[str, torch.Tensor]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for target_precision in [0.6, 0.7, 0.8]:
        threshold = choose_threshold_for_precision(valid["any_prob"], valid["any_target"], target_precision)
        report[f"precision_{target_precision:.1f}"] = {
            "threshold": threshold,
            "valid": threshold_metrics(valid["any_prob"], valid["any_target"], threshold),
            "test": threshold_metrics(test["any_prob"], test["any_target"], threshold),
        }
    return report


def choose_threshold_for_precision(prob: torch.Tensor, target: torch.Tensor, target_precision: float) -> float:
    thresholds = sorted({float(value) for value in prob.tolist()}, reverse=True)
    best = 1.0
    best_recall = -1.0
    for threshold in thresholds:
        metrics = threshold_metrics(prob, target, threshold)
        if metrics["selected_count"] > 0 and metrics["precision"] >= target_precision and metrics["recall"] > best_recall:
            best = threshold
            best_recall = metrics["recall"]
    return float(best)


def threshold_metrics(prob: torch.Tensor, target: torch.Tensor, threshold: float) -> dict[str, float]:
    selected = prob >= threshold
    correct = target == 1
    selected_count = int(selected.sum().item())
    true_positive = int((selected & correct).sum().item())
    precision = true_positive / selected_count if selected_count else 0.0
    recall = true_positive / int(correct.sum().item()) if int(correct.sum().item()) else 0.0
    coverage = selected_count / len(prob) if len(prob) else 0.0
    return {
        "selected_count": float(selected_count),
        "precision": float(precision),
        "recall": float(recall),
        "coverage": float(coverage),
    }


def ece(prob: torch.Tensor, target: torch.Tensor, *, bins: int) -> float:
    total = float(len(prob))
    if total == 0:
        return 0.0
    value = 0.0
    for idx in range(bins):
        low = idx / bins
        high = (idx + 1) / bins
        if idx == bins - 1:
            mask = (prob >= low) & (prob <= high)
        else:
            mask = (prob >= low) & (prob < high)
        if int(mask.sum()) == 0:
            continue
        conf = float(prob[mask].mean().item())
        acc = float(target[mask].mean().item())
        value += float(mask.float().mean().item()) * abs(conf - acc)
    return float(value)


def write_outputs(
    output_dir: Path,
    args: dict[str, Any],
    raw_eval: dict[str, dict[str, Any]],
    calibrated: dict[str, Any],
    threshold_report: dict[str, Any],
) -> dict[str, str]:
    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as handle:
        now = int(time.time())
        for split, ablations in raw_eval.items():
            for ablation, metrics in ablations.items():
                handle.write(json.dumps({"created_at": now, "split": split, "ablation": ablation, "metrics": metrics}, sort_keys=True) + "\n")
        for split in ["valid", "test"]:
            handle.write(json.dumps({"created_at": now, "split": split, "ablation": "full_calibrated", "metrics": calibrated[split]}, sort_keys=True) + "\n")
    ablation_csv = output_dir / "ablation_metrics.csv"
    rows = []
    for split, ablations in raw_eval.items():
        for ablation, metrics in ablations.items():
            rows.append({"split": split, "ablation": ablation, **metrics})
    write_csv(ablation_csv, rows)
    calibration_path = output_dir / "calibration_report.json"
    calibration_path.write_text(
        json.dumps({"calibrated": calibrated, "threshold_report": threshold_report}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "summary_json": str(output_dir / "summary.json"),
        "metrics_jsonl": str(metrics_path),
        "ablation_metrics_csv": str(ablation_csv),
        "calibration_report_json": str(calibration_path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
