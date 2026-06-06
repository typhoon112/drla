"""Evaluate a P2-E hierarchical latent fuser checkpoint locally."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from drla.scripts.audit_cola_hierarchical_aggregation_potential import (
    build_groups,
    choose_text_majority_selected,
)
from drla.scripts.audit_cola_sequential_latent_mas import load_official_scorer, score_text_with_official_rules
from drla.scripts.train_cola_hierarchical_latent_fuser import (
    HierarchicalLatentFuser,
    HierarchicalLatentFuserConfig,
    build_tensors,
    make_datasets,
    read_jsonl,
    split_groups,
)
from drla.scripts.train_cola_readiness_model import resolve_device
from drla.tracking import require_swanlab_disabled_for_non_training


def main() -> None:
    summary = eval_hierarchical_latent_fuser(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--swanlab-mode", default="disabled")
    return vars(parser.parse_args())


def eval_hierarchical_latent_fuser(args: dict[str, Any]) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        str(args["swanlab_mode"]),
        script_kind="P2-E hierarchical latent fuser checkpoint evaluator",
    )
    output_dir = Path(args["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args["checkpoint"], map_location="cpu")
    train_config = HierarchicalLatentFuserConfig(**checkpoint["config"])
    eval_config = replace(train_config, swanlab_mode="disabled")
    scorer = load_official_scorer(Path(eval_config.acc_calc_script))
    packets = read_jsonl(Path(eval_config.packets_jsonl))
    groups = build_groups(packets, eval_config, scorer)
    if eval_config.max_groups:
        groups = groups[: eval_config.max_groups]
    splits = split_groups(groups, eval_config)
    tensors_by_split, metadata = build_tensors(groups, splits, eval_config)
    train_ds, valid_ds, test_ds, _norm_stats = make_datasets(tensors_by_split, eval_config)
    datasets = {"train": train_ds, "valid": valid_ds, "test": test_ds}
    split = str(args["eval_split"])
    dataset = datasets[split]
    loader = DataLoader(dataset, batch_size=int(args["batch_size"]), shuffle=False, num_workers=0)

    device = resolve_device(str(args["device"]))
    model = HierarchicalLatentFuser(
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
        sender_layers=eval_config.sender_layers,
        dropout=eval_config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logits = []
    with torch.no_grad():
        for batch in loader:
            batch = [item.to(device) for item in batch]
            logits.append(model(*batch[:5]).cpu())
    logits_tensor = torch.cat(logits, dim=0)
    metrics, per_task, predictions = score_predictions(
        logits_tensor,
        tensors_by_split[split],
        groups,
        splits[split],
        scorer,
    )
    artifacts = write_outputs(output_dir, args, metrics, per_task, predictions)
    summary = {
        "created_at": int(time.time()),
        "checkpoint": str(args["checkpoint"]),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_metric": checkpoint.get("metric"),
        "eval_split": split,
        "config": args,
        "train_config": asdict(train_config),
        "split_size": len(splits[split]),
        "metrics": metrics,
        "per_task": per_task,
        "artifacts": artifacts,
        "interpretation": (
            "Local-only best-checkpoint evaluation for the P2-E latent fuser. "
            "Decoded answers are used only to score the sender selected by the "
            "decoder-free fuser."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def score_predictions(
    logits: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    groups: list[dict[str, Any]],
    indices: list[int],
    scorer: Any,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    labels = tensors["labels"]
    target_scores = tensors["target_scores"]
    selected = logits.argmax(dim=1)
    predictions = []
    for local_idx, group_index in enumerate(indices):
        group = groups[group_index]
        chosen_member = group["members"][int(selected[local_idx])]
        chosen_score = score_text_with_official_rules(
            task=str(group["task"]),
            text=str(chosen_member["selected_prediction"]),
            ground_truth=group["ground_truth"],
            choices=group["choices"],
            scorer=scorer,
        )
        text_choice = choose_text_majority_selected(group["members"])
        text_score = score_text_with_official_rules(
            task=str(group["task"]),
            text=str(text_choice["prediction"]),
            ground_truth=group["ground_truth"],
            choices=group["choices"],
            scorer=scorer,
        )
        first_member = group["members"][0]
        predictions.append(
            {
                "task": group["task"],
                "sample_key": group["sample_key"],
                "sample_id": group["sample_id"],
                "model_selected_index": int(selected[local_idx]),
                "model_selected_seed": chosen_member["sender_seed"],
                "model_correct": int(bool(chosen_score["correct"])),
                "model_score": float(chosen_score["score"]),
                "single_sender_first_correct": int(bool(first_member["selected_correct"])),
                "single_sender_first_score": float(first_member.get("selected_score", float(bool(first_member["selected_correct"])))),
                "text_majority_correct": int(bool(text_score["correct"])),
                "text_majority_score": float(text_score["score"]),
                "oracle_any_selected_correct": int(bool(labels[local_idx].sum().item() > 0)),
                "oracle_best_selected_score": float(target_scores[local_idx].max().item()),
                "num_selected_correct_senders": int(labels[local_idx].sum().item()),
            }
        )
    metrics = aggregate(predictions)
    per_task = [aggregate([row for row in predictions if row["task"] == task], task=task) for task in sorted({row["task"] for row in predictions})]
    return metrics, per_task, predictions


def aggregate(rows: list[dict[str, Any]], *, task: str = "all") -> dict[str, float]:
    count = len(rows)
    any_rows = [row for row in rows if row["oracle_any_selected_correct"]]
    return {
        "task": task,
        "count": float(count),
        "model_selected_accuracy": mean(row["model_correct"] for row in rows),
        "model_mean_official_score": mean(row["model_score"] for row in rows),
        "single_sender_first_accuracy": mean(row["single_sender_first_correct"] for row in rows),
        "single_sender_first_mean_official_score": mean(row["single_sender_first_score"] for row in rows),
        "text_majority_selected_accuracy": mean(row["text_majority_correct"] for row in rows),
        "text_majority_mean_official_score": mean(row["text_majority_score"] for row in rows),
        "oracle_any_selected_accuracy": mean(row["oracle_any_selected_correct"] for row in rows),
        "oracle_best_selected_mean_official_score": mean(row["oracle_best_selected_score"] for row in rows),
        "model_selects_correct_when_any_correct": mean(row["model_correct"] for row in any_rows),
    }


def write_outputs(
    output_dir: Path,
    args: dict[str, Any],
    metrics: dict[str, Any],
    per_task: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, str]:
    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"created_at": int(time.time()), "split": args["eval_split"], "metrics": metrics}, sort_keys=True) + "\n")
    per_task_path = output_dir / "per_task_metrics.csv"
    write_csv(per_task_path, per_task)
    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "summary_json": str(output_dir / "summary.json"),
        "metrics_jsonl": str(metrics_path),
        "per_task_metrics_csv": str(per_task_path),
        "predictions_jsonl": str(predictions_path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


if __name__ == "__main__":
    main()
