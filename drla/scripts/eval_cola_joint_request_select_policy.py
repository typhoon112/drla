"""Evaluate a P2-E joint request/select checkpoint locally.

The trainer writes full prediction tables only for the validation-selected best
checkpoint.  This evaluator reconstructs the same locked split for any saved
checkpoint, including ``last_checkpoint.pt``, and applies request thresholds
selected on valid to held-out test.  It has no optimizer/backward path and must
stay local-only with SwanLab disabled.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from drla.scripts.audit_cola_hierarchical_aggregation_potential import build_groups
from drla.scripts.audit_cola_sequential_latent_mas import load_official_scorer
from drla.scripts.train_cola_hierarchical_latent_fuser import (
    HierarchicalLatentFuserConfig,
    build_tensors,
    read_jsonl,
    split_groups,
)
from drla.scripts.train_cola_joint_request_select_policy import (
    JointRequestSelectConfig,
    JointRequestSelectPolicy,
    aggregate,
    evaluate_threshold_policies,
    make_joint_datasets,
    parse_float_list,
    score_predictions_from_model,
    write_csv,
    write_jsonl,
)
from drla.scripts.train_cola_readiness_model import resolve_device
from drla.scripts.train_cola_request_more_policy import compute_fuser_refs, normalize_tensors
from drla.tracking import require_swanlab_disabled_for_non_training


def main() -> None:
    summary = eval_joint_request_select(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-splits", default="valid,test")
    parser.add_argument("--target-request-rates", default="")
    parser.add_argument("--target-helpful-precisions", default="")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--swanlab-mode", default="disabled")
    return vars(parser.parse_args())


def eval_joint_request_select(args: dict[str, Any]) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        str(args["swanlab_mode"]),
        script_kind="P2-E joint request/select checkpoint evaluator",
    )
    output_dir = Path(args["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_config = JointRequestSelectConfig(**checkpoint["config"])
    eval_config = replace(train_config, swanlab_mode="disabled")
    eval_splits = parse_split_names(str(args["eval_splits"]))
    device = resolve_device(str(args["device"]))

    scorer = load_official_scorer(Path(eval_config.acc_calc_script))
    fuser_checkpoint = torch.load(eval_config.fuser_checkpoint, map_location="cpu")
    fuser_train_config = HierarchicalLatentFuserConfig(**fuser_checkpoint["config"])
    data_config = replace(
        fuser_train_config,
        packets_jsonl=eval_config.packets_jsonl,
        output_dir=eval_config.output_dir,
        data_root=eval_config.data_root,
        acc_calc_script=eval_config.acc_calc_script,
        seed=eval_config.seed,
        train_ratio=eval_config.train_ratio,
        valid_ratio=eval_config.valid_ratio,
        max_groups=eval_config.max_groups,
        max_cached_shards=eval_config.max_cached_shards,
        swanlab_mode="disabled",
    )
    groups = build_groups(read_jsonl(Path(eval_config.packets_jsonl)), data_config, scorer)
    if eval_config.max_groups:
        groups = groups[: eval_config.max_groups]
    splits = split_groups(groups, data_config)
    tensors_by_split, metadata = build_tensors(groups, splits, data_config)
    normalized = normalize_tensors(tensors_by_split, fuser_checkpoint["norm_stats"])
    fuser_refs = compute_fuser_refs(
        fuser_checkpoint=fuser_checkpoint,
        fuser_config=fuser_train_config,
        metadata=metadata,
        normalized=normalized,
        tensors_by_split=tensors_by_split,
        device=device,
        batch_size=int(args["batch_size"]),
    )
    datasets = dict(zip(("train", "valid", "test"), make_joint_datasets(normalized, tensors_by_split), strict=True))

    model = JointRequestSelectPolicy(
        latent_dim=metadata["latent_dim"],
        process_dim=metadata["process_dim"],
        certificate_dim=metadata["certificate_dim"],
        max_blocks=metadata["max_blocks"],
        block_size=metadata["block_size"],
        sender_count=metadata["sender_count"],
        task_count=len(metadata["task_to_idx"]),
        d_model=eval_config.d_model,
        attention_heads=eval_config.attention_heads,
        inter_layers=eval_config.inter_layers,
        request_sender_layers=eval_config.request_sender_layers,
        selector_sender_layers=eval_config.selector_sender_layers,
        dropout=eval_config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions: dict[str, list[dict[str, Any]]] = {}
    split_metrics = []
    per_task_rows = []
    for split in eval_splits:
        loader = DataLoader(
            datasets[split],
            batch_size=int(args["batch_size"]),
            shuffle=False,
            num_workers=0,
        )
        rows = score_predictions_from_model(
            model,
            loader,
            tensors_by_split[split],
            groups,
            splits[split],
            fuser_refs[split],
            scorer,
            device,
            eval_config,
        )
        predictions[split] = rows
        write_jsonl(output_dir / f"{split}_predictions.jsonl", rows)
        metrics = aggregate(rows)
        split_metrics.append({"split": split, **metrics})
        for task in sorted({row["task"] for row in rows}):
            task_rows = [row for row in rows if row["task"] == task]
            per_task_rows.append({"split": split, **aggregate(task_rows, task=task)})

    request_rates = parse_float_list(str(args["target_request_rates"]) or eval_config.target_request_rates)
    helpful_precisions = parse_float_list(
        str(args["target_helpful_precisions"]) or eval_config.target_helpful_precisions
    )
    policy_rows: list[dict[str, Any]] = []
    if "valid" in predictions and "test" in predictions:
        policy_rows = evaluate_threshold_policies(
            valid_predictions=predictions["valid"],
            test_predictions=predictions["test"],
            request_rates=request_rates,
            helpful_precisions=helpful_precisions,
        )
        write_csv(output_dir / "policy_metrics.csv", policy_rows)

    write_csv(output_dir / "split_metrics.csv", split_metrics)
    write_csv(output_dir / "per_task_metrics.csv", per_task_rows)
    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as handle:
        for row in split_metrics:
            handle.write(json.dumps({"created_at": int(time.time()), "kind": "split", "metrics": row}, sort_keys=True) + "\n")
        for row in policy_rows:
            handle.write(json.dumps({"created_at": int(time.time()), "kind": "policy", "metrics": row}, sort_keys=True) + "\n")

    artifacts = {
        "summary_json": str(output_dir / "summary.json"),
        "metrics_jsonl": str(metrics_path),
        "split_metrics_csv": str(output_dir / "split_metrics.csv"),
        "per_task_metrics_csv": str(output_dir / "per_task_metrics.csv"),
    }
    if policy_rows:
        artifacts["policy_metrics_csv"] = str(output_dir / "policy_metrics.csv")
    for split in eval_splits:
        artifacts[f"{split}_predictions_jsonl"] = str(output_dir / f"{split}_predictions.jsonl")

    summary = {
        "created_at": int(time.time()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_metric": checkpoint.get("metric"),
        "config": args,
        "train_config": asdict(train_config),
        "fuser_checkpoint": eval_config.fuser_checkpoint,
        "fuser_train_config": asdict(fuser_train_config),
        "split_sizes": {split: len(splits[split]) for split in eval_splits},
        "metrics": {row["split"]: row for row in split_metrics},
        "best_policy_by_test_score": best_policy(policy_rows, "test_model_after_request_score"),
        "best_policy_by_test_accuracy": best_policy(policy_rows, "test_model_after_request_accuracy"),
        "artifacts": artifacts,
        "interpretation": (
            "Local-only checkpoint evaluator for the P2-E joint request/select "
            "policy. Request thresholds are selected on valid when both valid "
            "and test predictions are requested."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def best_policy(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: float(row[key]))


def parse_split_names(value: str) -> list[str]:
    splits = [part.strip() for part in value.split(",") if part.strip()]
    valid = {"train", "valid", "test"}
    unknown = [split for split in splits if split not in valid]
    if unknown:
        raise ValueError(f"unknown eval split(s): {unknown}")
    if not splits:
        raise ValueError("at least one eval split is required")
    return splits


if __name__ == "__main__":
    main()
