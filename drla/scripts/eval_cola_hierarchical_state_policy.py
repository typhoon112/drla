"""Evaluate a calibrated P2-E latent-state verifier as receiver policy.

This script is local-only. It does not train or update model weights. It
combines:

* a group-level latent-state utility verifier
* a sender-choice latent fuser
* text-majority, task-prior, and global-prior controls

The goal is to test whether the verifier's calibrated decoder-free state can
serve as an online receiver decision signal before agent-to-agent latent
communication claims are strengthened.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from drla.scripts.audit_cola_hierarchical_aggregation_potential import build_groups, choose_text_majority_selected
from drla.scripts.audit_cola_sequential_latent_mas import load_official_scorer, score_text_with_official_rules
from drla.scripts.train_cola_hierarchical_latent_fuser import (
    HierarchicalLatentFuser,
    HierarchicalLatentFuserConfig,
    build_tensors,
    make_datasets,
    read_jsonl,
    split_groups,
)
from drla.scripts.train_cola_hierarchical_state_verifier import (
    HierarchicalStateVerifier,
    HierarchicalStateVerifierConfig,
)
from drla.scripts.train_cola_readiness_model import resolve_device
from drla.tracking import require_swanlab_disabled_for_non_training


def main() -> None:
    summary = eval_state_policy(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-checkpoint",
        default=(
            "/data1/luyifei/drla/outputs/cola_hierarchical_state_verifier/"
            "p2e_state_verifier_full_seed20260529_20260529/checkpoints/best_checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--fuser-checkpoint",
        default=(
            "/data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/"
            "p2e_hierarchical_fuser_score_full_seed20260529_20260529/checkpoints/best_checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--calibration-report",
        default=(
            "/data1/luyifei/drla/outputs/cola_hierarchical_state_verifier/"
            "p2e_state_verifier_full_seed20260529_20260529_calibration_ablation/"
            "calibration_report.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-precision-values", default="0.55,0.60,0.65")
    parser.add_argument("--coverage-values", default="0.10,0.25,0.50")
    parser.add_argument("--swanlab-mode", default="disabled")
    return vars(parser.parse_args())


def eval_state_policy(args: dict[str, Any]) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        str(args["swanlab_mode"]),
        script_kind="P2-E calibrated state-policy evaluator",
    )
    output_dir = Path(args["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    state_checkpoint = torch.load(args["state_checkpoint"], map_location="cpu")
    fuser_checkpoint = torch.load(args["fuser_checkpoint"], map_location="cpu")
    state_config = replace(HierarchicalStateVerifierConfig(**state_checkpoint["config"]), swanlab_mode="disabled")
    fuser_config = replace(HierarchicalLatentFuserConfig(**fuser_checkpoint["config"]), swanlab_mode="disabled")
    validate_compatible_configs(state_config, fuser_config)

    scorer = load_official_scorer(Path(state_config.acc_calc_script))
    packets = read_jsonl(Path(state_config.packets_jsonl))
    groups = build_groups(packets, state_config, scorer)
    if state_config.max_groups:
        groups = groups[: state_config.max_groups]
    splits = split_groups(groups, state_config)
    tensors_by_split, metadata = build_tensors(groups, splits, state_config)
    train_ds, valid_ds, test_ds, _norm_stats = make_datasets(tensors_by_split)
    datasets = {"train": train_ds, "valid": valid_ds, "test": test_ds}

    device = resolve_device(str(args["device"]))
    state_model = build_state_model(state_checkpoint, state_config, metadata, device)
    fuser_model = build_fuser_model(fuser_checkpoint, fuser_config, metadata, device)
    calibrators = load_calibrators(Path(args["calibration_report"]))

    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ["train", "valid", "test"]:
        loader = DataLoader(datasets[split], batch_size=int(args["batch_size"]), shuffle=False, num_workers=0)
        model_outputs = collect_outputs(state_model, fuser_model, loader, device, calibrators)
        rows_by_split[split] = build_rows(
            split=split,
            groups=groups,
            indices=splits[split],
            tensors=tensors_by_split[split],
            outputs=model_outputs,
            scorer=scorer,
        )

    priors = fit_priors(rows_by_split["train"])
    for split in ["valid", "test"]:
        attach_priors(rows_by_split[split], priors)

    target_precision_values = parse_float_list(str(args["target_precision_values"]))
    coverage_values = parse_float_list(str(args["coverage_values"]))
    always = {split: aggregate_always(rows) for split, rows in rows_by_split.items() if split in {"valid", "test"}}
    policy_rows = evaluate_policies(
        valid_rows=rows_by_split["valid"],
        test_rows=rows_by_split["test"],
        target_precision_values=target_precision_values,
        coverage_values=coverage_values,
    )
    artifacts = write_outputs(output_dir, args, rows_by_split, always, policy_rows, priors)
    summary = {
        "created_at": int(time.time()),
        "state_checkpoint": str(args["state_checkpoint"]),
        "fuser_checkpoint": str(args["fuser_checkpoint"]),
        "calibration_report": str(args["calibration_report"]),
        "state_checkpoint_step": state_checkpoint.get("step"),
        "fuser_checkpoint_step": fuser_checkpoint.get("step"),
        "config": args,
        "state_train_config": asdict(state_config),
        "fuser_train_config": asdict(fuser_config),
        "split_sizes": {split: len(indices) for split, indices in splits.items()},
        "always": always,
        "policy_metrics": policy_rows,
        "priors": priors,
        "artifacts": artifacts,
        "interpretation": (
            "Local-only receiver-state policy audit. Thresholds are selected on "
            "valid and reported on held-out test. Decoded text and official "
            "scores are used only as offline evaluation labels; online policy "
            "signals are latent-state verifier outputs or prior controls."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def validate_compatible_configs(
    state_config: HierarchicalStateVerifierConfig,
    fuser_config: HierarchicalLatentFuserConfig,
) -> None:
    checks = {
        "packets_jsonl": (state_config.packets_jsonl, fuser_config.packets_jsonl),
        "seed": (state_config.seed, fuser_config.seed),
        "train_ratio": (state_config.train_ratio, fuser_config.train_ratio),
        "valid_ratio": (state_config.valid_ratio, fuser_config.valid_ratio),
        "max_groups": (state_config.max_groups, fuser_config.max_groups),
    }
    mismatches = {name: values for name, values in checks.items() if values[0] != values[1]}
    if mismatches:
        raise ValueError(f"state/fuser configs use incompatible locked splits: {mismatches}")


def build_state_model(
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


def build_fuser_model(
    checkpoint: dict[str, Any],
    config: HierarchicalLatentFuserConfig,
    metadata: dict[str, Any],
    device: torch.device,
) -> HierarchicalLatentFuser:
    model = HierarchicalLatentFuser(
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


def load_calibrators(path: Path) -> dict[str, dict[str, float]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    calibrators = report["calibrated"]["calibrators"]
    return {
        "any_platt": {
            "scale": float(calibrators["any_platt"]["scale"]),
            "bias": float(calibrators["any_platt"]["bias"]),
        },
        "score_affine": {
            "slope": float(calibrators["score_affine"]["slope"]),
            "intercept": float(calibrators["score_affine"]["intercept"]),
        },
    }


@torch.no_grad()
def collect_outputs(
    state_model: HierarchicalStateVerifier,
    fuser_model: HierarchicalLatentFuser,
    loader: DataLoader,
    device: torch.device,
    calibrators: dict[str, dict[str, float]],
) -> dict[str, torch.Tensor]:
    state_logits = []
    state_scores = []
    fuser_logits = []
    for batch in loader:
        batch = [item.to(device) for item in batch]
        state_out = state_model(*batch[:5])
        fuser_out = fuser_model(*batch[:5])
        state_logits.append(state_out["any_correct_logit"].detach().cpu())
        state_scores.append(torch.sigmoid(state_out["best_score_logit"]).detach().cpu())
        fuser_logits.append(fuser_out.detach().cpu())
    any_logit = torch.cat(state_logits)
    raw_score = torch.cat(state_scores)
    any_cal = calibrators["any_platt"]
    score_cal = calibrators["score_affine"]
    return {
        "state_raw_any_prob": torch.sigmoid(any_logit),
        "state_any_prob": torch.sigmoid(any_cal["scale"] * any_logit + any_cal["bias"]),
        "state_raw_best_score_pred": raw_score,
        "state_best_score_pred": (score_cal["slope"] * raw_score + score_cal["intercept"]).clamp(0.0, 1.0),
        "fuser_logits": torch.cat(fuser_logits),
    }


def build_rows(
    *,
    split: str,
    groups: list[dict[str, Any]],
    indices: list[int],
    tensors: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    scorer: Any,
) -> list[dict[str, Any]]:
    labels = tensors["labels"]
    target_scores = tensors["target_scores"]
    fuser_selected = outputs["fuser_logits"].argmax(dim=1)
    fuser_conf = torch.softmax(outputs["fuser_logits"], dim=1).max(dim=1).values
    rows = []
    for local_idx, group_idx in enumerate(indices):
        group = groups[group_idx]
        selected_idx = int(fuser_selected[local_idx].item())
        fuser_member = group["members"][selected_idx]
        text_choice = choose_text_majority_selected(group["members"])
        text_score = score_text_with_official_rules(
            task=str(group["task"]),
            text=str(text_choice["prediction"]),
            ground_truth=group["ground_truth"],
            choices=group["choices"],
            scorer=scorer,
        )
        rows.append(
            {
                "split": split,
                "task": group["task"],
                "sample_key": group["sample_key"],
                "sample_id": group["sample_id"],
                "fuser_selected_index": selected_idx,
                "fuser_selected_seed": fuser_member["sender_seed"],
                "fuser_confidence": float(fuser_conf[local_idx].item()),
                "fuser_correct": float(labels[local_idx, selected_idx].item()),
                "fuser_score": float(target_scores[local_idx, selected_idx].item()),
                "first_correct": float(labels[local_idx, 0].item()),
                "first_score": float(target_scores[local_idx, 0].item()),
                "text_majority_correct": float(bool(text_score["correct"])),
                "text_majority_score": float(text_score["score"]),
                "oracle_any": float(labels[local_idx].sum().item() > 0),
                "oracle_best_score": float(target_scores[local_idx].max().item()),
                "num_correct_senders": int(labels[local_idx].sum().item()),
                "state_raw_any_prob": float(outputs["state_raw_any_prob"][local_idx].item()),
                "state_any_prob": float(outputs["state_any_prob"][local_idx].item()),
                "state_raw_best_score_pred": float(outputs["state_raw_best_score_pred"][local_idx].item()),
                "state_best_score_pred": float(outputs["state_best_score_pred"][local_idx].item()),
            }
        )
    return rows


def fit_priors(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        task_rows[str(row["task"])].append(row)
    global_prior = {
        "any": mean(row["oracle_any"] for row in train_rows),
        "best_score": mean(row["oracle_best_score"] for row in train_rows),
    }
    task_priors = {}
    for task, rows in sorted(task_rows.items()):
        task_priors[task] = {
            "count": len(rows),
            "any": mean(row["oracle_any"] for row in rows),
            "best_score": mean(row["oracle_best_score"] for row in rows),
        }
    return {"train_global": global_prior, "train_task": task_priors}


def attach_priors(rows: list[dict[str, Any]], priors: dict[str, Any]) -> None:
    global_prior = priors["train_global"]
    task_priors = priors["train_task"]
    for row in rows:
        task_prior = task_priors.get(str(row["task"]), global_prior)
        row["train_task_prior_any"] = float(task_prior["any"])
        row["train_task_prior_score"] = float(task_prior["best_score"])
        row["train_global_prior_any"] = float(global_prior["any"])
        row["train_global_prior_score"] = float(global_prior["best_score"])


def evaluate_policies(
    *,
    valid_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    target_precision_values: list[float],
    coverage_values: list[float],
) -> list[dict[str, Any]]:
    signal_specs = [
        ("state_any_prob", "latent_state"),
        ("state_best_score_pred", "latent_state"),
        ("fuser_confidence", "latent_fuser_confidence"),
        ("train_task_prior_any", "task_prior"),
        ("train_task_prior_score", "task_prior"),
        ("train_global_prior_any", "global_prior"),
        ("train_global_prior_score", "global_prior"),
    ]
    rows: list[dict[str, Any]] = []
    for signal, signal_type in signal_specs:
        for target in target_precision_values:
            threshold = choose_threshold_for_precision(valid_rows, signal, "oracle_any", target)
            rows.append(
                {
                    "selection_mode": "target_any_precision",
                    "target_value": target,
                    "signal": signal,
                    "signal_type": signal_type,
                    "threshold": threshold,
                    **prefixed("valid", gate_metrics(valid_rows, signal, threshold)),
                    **prefixed("test", gate_metrics(test_rows, signal, threshold)),
                }
            )
            fuser_threshold = choose_threshold_for_precision(valid_rows, signal, "fuser_correct", target)
            rows.append(
                {
                    "selection_mode": "target_fuser_precision",
                    "target_value": target,
                    "signal": signal,
                    "signal_type": signal_type,
                    "threshold": fuser_threshold,
                    **prefixed("valid", gate_metrics(valid_rows, signal, fuser_threshold)),
                    **prefixed("test", gate_metrics(test_rows, signal, fuser_threshold)),
                }
            )
        for coverage in coverage_values:
            threshold = choose_threshold_for_coverage(valid_rows, signal, coverage)
            rows.append(
                {
                    "selection_mode": "target_coverage",
                    "target_value": coverage,
                    "signal": signal,
                    "signal_type": signal_type,
                    "threshold": threshold,
                    **prefixed("valid", gate_metrics(valid_rows, signal, threshold)),
                    **prefixed("test", gate_metrics(test_rows, signal, threshold)),
                }
            )
    return rows


def choose_threshold_for_precision(rows: list[dict[str, Any]], signal: str, target_name: str, target_precision: float) -> float:
    thresholds = sorted({float(row[signal]) for row in rows}, reverse=True)
    best_threshold = 1.0
    best_coverage = -1.0
    for threshold in thresholds:
        selected = [row for row in rows if float(row[signal]) >= threshold]
        if not selected:
            continue
        precision = mean(row[target_name] for row in selected)
        coverage = len(selected) / len(rows)
        if precision >= target_precision and coverage > best_coverage:
            best_threshold = threshold
            best_coverage = coverage
    return float(best_threshold)


def choose_threshold_for_coverage(rows: list[dict[str, Any]], signal: str, target_coverage: float) -> float:
    values = sorted((float(row[signal]) for row in rows), reverse=True)
    if not values:
        return 1.0
    k = max(1, min(len(values), int(round(target_coverage * len(values)))))
    return float(values[k - 1])


def gate_metrics(rows: list[dict[str, Any]], signal: str, threshold: float) -> dict[str, float]:
    selected = [row for row in rows if float(row[signal]) >= threshold]
    selected_count = len(selected)
    positives = [row for row in rows if float(row["oracle_any"]) > 0]
    recall_denom = len(positives)
    return {
        "selected_count": float(selected_count),
        "coverage": selected_count / len(rows) if rows else 0.0,
        "any_precision": mean(row["oracle_any"] for row in selected),
        "any_recall": (
            sum(float(row["oracle_any"]) for row in selected) / recall_denom
            if recall_denom
            else 0.0
        ),
        "accepted_fuser_accuracy": mean(row["fuser_correct"] for row in selected),
        "accepted_fuser_score": mean(row["fuser_score"] for row in selected),
        "accepted_first_accuracy": mean(row["first_correct"] for row in selected),
        "accepted_first_score": mean(row["first_score"] for row in selected),
        "accepted_text_accuracy": mean(row["text_majority_correct"] for row in selected),
        "accepted_text_score": mean(row["text_majority_score"] for row in selected),
        "accepted_oracle_best_score": mean(row["oracle_best_score"] for row in selected),
        "fallback_first_accuracy": mean(
            row["fuser_correct"] if float(row[signal]) >= threshold else row["first_correct"]
            for row in rows
        ),
        "fallback_first_score": mean(
            row["fuser_score"] if float(row[signal]) >= threshold else row["first_score"]
            for row in rows
        ),
        "text_mixed_fallback_accuracy": mean(
            row["fuser_correct"] if float(row[signal]) >= threshold else row["text_majority_correct"]
            for row in rows
        ),
        "text_mixed_fallback_score": mean(
            row["fuser_score"] if float(row[signal]) >= threshold else row["text_majority_score"]
            for row in rows
        ),
    }


def aggregate_always(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "count": float(len(rows)),
        "fuser_accuracy": mean(row["fuser_correct"] for row in rows),
        "fuser_score": mean(row["fuser_score"] for row in rows),
        "first_accuracy": mean(row["first_correct"] for row in rows),
        "first_score": mean(row["first_score"] for row in rows),
        "text_majority_accuracy": mean(row["text_majority_correct"] for row in rows),
        "text_majority_score": mean(row["text_majority_score"] for row in rows),
        "oracle_any_accuracy": mean(row["oracle_any"] for row in rows),
        "oracle_best_score": mean(row["oracle_best_score"] for row in rows),
        "state_any_prob_mean": mean(row["state_any_prob"] for row in rows),
        "state_best_score_pred_mean": mean(row["state_best_score_pred"] for row in rows),
    }


def write_outputs(
    output_dir: Path,
    args: dict[str, Any],
    rows_by_split: dict[str, list[dict[str, Any]]],
    always: dict[str, dict[str, float]],
    policy_rows: list[dict[str, Any]],
    priors: dict[str, Any],
) -> dict[str, str]:
    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as handle:
        now = int(time.time())
        handle.write(json.dumps({"created_at": now, "kind": "always", "metrics": always}, sort_keys=True) + "\n")
        for row in policy_rows:
            handle.write(json.dumps({"created_at": now, "kind": "policy", "metrics": row}, sort_keys=True) + "\n")

    policy_csv = output_dir / "policy_metrics.csv"
    write_csv(policy_csv, policy_rows)

    states_path = output_dir / "per_group_states.jsonl"
    with states_path.open("w", encoding="utf-8") as handle:
        for split in ["valid", "test"]:
            for row in rows_by_split[split]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    priors_path = output_dir / "train_priors.json"
    priors_path.write_text(json.dumps(priors, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(args, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "summary_json": str(output_dir / "summary.json"),
        "metrics_jsonl": str(metrics_path),
        "policy_metrics_csv": str(policy_csv),
        "per_group_states_jsonl": str(states_path),
        "train_priors_json": str(priors_path),
        "config_json": str(config_path),
    }


def prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def mean(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


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
