"""Train a non-gold continuation-risk model for Cola halt calibration.

The default target is whether the current task-scored prediction is a strict
prefix of the final task-scored prediction in the same block rollout. The
prediction_change target instead predicts whether the current prediction will
change before the rollout reaches its prediction-stability reference. These are
not gold correctness labels; they are process labels for continuation risk.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from drla.scripts.train_cola_readiness_model import (
    OFFICIAL_COLA_TASKS,
    ReadinessTrainConfig,
    add_derived_stability_features,
    binary_accuracy,
    binary_auprc,
    binary_auroc,
    load_training_rows,
    parse_tasks,
    pos_weight,
    resolve_device,
    stable_uniform,
)
from drla.tracking import finish_experiment, init_experiment, log_metrics


RISK_FEATURE_FIELDS = [
    "block_number",
    "max_block_budget",
    "remaining_blocks",
    "block_fraction",
    "token_entropy_mean",
    "token_top_prob_mean",
    "eos_prob_max",
    "im_end_prob_max",
    "stop_prob_max",
    "stop_prob_margin_vs_non_stop",
    "contains_eos",
    "contains_im_end",
    "contains_stop",
    "answer_text_nonempty",
    "answer_changed",
    "same_text_streak",
    "scored_prediction_nonempty",
    "scored_prediction_changed",
    "scored_prediction_same_streak",
    "processed_generation_changed",
    "processed_generation_same_streak",
    "already_stopped_before_block",
    "prediction_char_len",
    "prediction_word_count",
    "prediction_ends_alnum",
    "prediction_ends_terminal_punct",
    "prediction_ends_mid_punct",
    "prediction_contains_space",
    "prediction_is_numericish",
    "prediction_ends_decimal_point",
    "prediction_ends_single_letter_period",
    "prediction_has_unbalanced_quote",
    "prediction_has_unbalanced_bracket",
    "prediction_last_token_char_len",
    "prediction_last_token_is_short",
    "processed_char_len",
    "processed_word_count",
    "decode_char_len",
]


@dataclass(frozen=True)
class ContinuationRiskTrainConfig:
    labels_dir: str = "/data1/luyifei/drla/outputs/cola_readiness_frontiers/official8_1000_b64_t16_seed66_20260524"
    output_dir: str = "/data1/luyifei/drla/outputs/cola_continuation_risk_model/official8_b64_process_no_task_seed20260524"
    tasks: str = ",".join(OFFICIAL_COLA_TASKS)
    seed: int = 20260524
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    batch_size: int = 512
    epochs: int = 40
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    hidden_dim: int = 96
    signal_mode: str = "process_no_task"
    target_mode: str = "strict_prefix"
    valid_interval: int = 50
    num_workers: int = 0
    device: str = "auto"
    swanlab_mode: str = "cloud"
    experiment_name: str = "official8-continuation-risk-model"


class ContinuationRiskModel(nn.Module):
    def __init__(self, feature_dim: int, task_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.net = nn.Sequential(
            nn.Linear(feature_dim + task_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, task_onehot: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([self.feature_norm(features), task_onehot], dim=-1)).squeeze(-1)


def train_continuation_risk_model(config: ContinuationRiskTrainConfig) -> dict[str, Any]:
    if config.signal_mode not in {"process", "process_no_task"}:
        raise ValueError("signal_mode must be one of: process, process_no_task")
    if config.target_mode not in {"strict_prefix", "prediction_change"}:
        raise ValueError("target_mode must be one of: strict_prefix, prediction_change")
    if config.valid_interval > 100:
        raise ValueError("valid_interval must be <= 100 steps")
    if not 0 < config.train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0 <= config.valid_ratio < 1:
        raise ValueError("valid_ratio must be in [0, 1)")
    if config.train_ratio + config.valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio must leave a test split")

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"

    rows = load_training_rows(
        ReadinessTrainConfig(labels_dir=config.labels_dir, tasks=config.tasks, seed=config.seed)
    )
    tensors, metadata = build_risk_tensors(rows, config)
    splits = split_indices(metadata["sample_keys"], config)
    train_data, valid_data, test_data, norm_stats = make_split_datasets(tensors, splits)

    device = resolve_device(config.device)
    model = ContinuationRiskModel(
        feature_dim=train_data.tensors[0].shape[1],
        task_dim=train_data.tensors[1].shape[1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    risk_pos_weight = pos_weight(train_data.tensors[2]).to(device)

    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
    )
    valid_loader = DataLoader(valid_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    run = init_experiment(
        stage="cola-continuation-risk",
        experiment_name=config.experiment_name,
        description="Non-gold continuation/prefix-risk model for Cola halt calibration.",
        config={
            **asdict(config),
            "feature_fields": RISK_FEATURE_FIELDS,
            "target_mode": config.target_mode,
            "num_rows": len(rows),
            "split_sizes": {name: len(indices) for name, indices in splits.items()},
            "positive_rate": float(tensors["y_risk"].mean().item()),
        },
        mode=config.swanlab_mode,
        tags=["cola", "official-benchmark", "halt", "continuation-risk"],
    )

    best_metric = -math.inf
    best_step = 0
    global_step = 0
    history: list[dict[str, Any]] = []
    start_time = time.time()

    try:
        with metrics_path.open("w", encoding="utf-8") as metrics_f:
            for epoch in range(config.epochs):
                model.train()
                for batch in train_loader:
                    global_step += 1
                    features, task_onehot, y_risk = [item.to(device) for item in batch]
                    logits = model(features, task_onehot)
                    loss = F.binary_cross_entropy_with_logits(logits, y_risk, pos_weight=risk_pos_weight)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    with torch.no_grad():
                        prob = torch.sigmoid(logits)
                        train_metrics = {
                            "loss": float(loss.item()),
                            "risk_accuracy": binary_accuracy(prob.cpu(), y_risk.cpu()),
                            "epoch": float(epoch),
                        }
                    log_metrics(train_metrics, step=global_step, prefix="train")
                    write_metric(metrics_f, "train", global_step, train_metrics)

                    if global_step % config.valid_interval == 0:
                        valid_metrics = evaluate(model, valid_loader, device)
                        log_metrics(valid_metrics, step=global_step, prefix="valid")
                        write_metric(metrics_f, "valid", global_step, valid_metrics)
                        selected = select_metric(valid_metrics)
                        if selected > best_metric:
                            best_metric = selected
                            best_step = global_step
                            save_checkpoint(
                                checkpoint_dir / "best_checkpoint.pt",
                                model=model,
                                optimizer=optimizer,
                                config=config,
                                norm_stats=norm_stats,
                                metadata=metadata,
                                step=global_step,
                                metric=best_metric,
                            )
                        history.append({"step": global_step, "valid": valid_metrics})

            valid_metrics = evaluate(model, valid_loader, device)
            test_metrics = evaluate(model, test_loader, device)
            log_metrics(valid_metrics, step=global_step, prefix="valid")
            log_metrics(test_metrics, step=global_step, prefix="test")
            write_metric(metrics_f, "valid", global_step, valid_metrics)
            write_metric(metrics_f, "test", global_step, test_metrics)
            selected = select_metric(valid_metrics)
            if selected > best_metric:
                best_metric = selected
                best_step = global_step
                save_checkpoint(
                    checkpoint_dir / "best_checkpoint.pt",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    norm_stats=norm_stats,
                    metadata=metadata,
                    step=global_step,
                    metric=best_metric,
                )
            save_checkpoint(
                checkpoint_dir / "last_checkpoint.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                norm_stats=norm_stats,
                metadata=metadata,
                step=global_step,
                metric=selected,
            )

        summary = {
            "created_at": int(time.time()),
            "config": asdict(config),
            "feature_fields": metadata["feature_fields"],
            "signal_mode": config.signal_mode,
            "target_mode": config.target_mode,
            "num_rows": len(rows),
            "split_sizes": {name: len(indices) for name, indices in splits.items()},
            "positive_rate": float(tensors["y_risk"].mean().item()),
            "best_step": best_step,
            "best_metric_name": "valid/risk_auroc",
            "best_metric": best_metric,
            "last_valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
            "metrics_jsonl": str(metrics_path),
            "best_checkpoint": str(checkpoint_dir / "best_checkpoint.pt"),
            "last_checkpoint": str(checkpoint_dir / "last_checkpoint.pt"),
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


def build_risk_tensors(
    rows: list[dict[str, Any]],
    config: ContinuationRiskTrainConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    add_derived_stability_features(rows)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(f"{row['task']}::{row['sample_id']}", []).append(row)
    for sample_rows in grouped.values():
        sample_rows.sort(key=lambda item: int(item["block_index"]))

    features: list[list[float]] = []
    task_onehots: list[list[float]] = []
    labels: list[float] = []
    sample_keys: list[str] = []
    task_to_idx = {task: idx for idx, task in enumerate(OFFICIAL_COLA_TASKS)}

    for sample_key, sample_rows in grouped.items():
        final_prediction = normalize_text(sample_rows[-1].get("scored_prediction"))
        stability_prediction = prediction_stability_reference(sample_rows)
        for row in sample_rows:
            features.append(row_features(row))
            task_vec = [0.0] * len(OFFICIAL_COLA_TASKS)
            if config.signal_mode == "process":
                task_vec[task_to_idx[row["task"]]] = 1.0
            task_onehots.append(task_vec)
            prediction = normalize_text(row.get("scored_prediction"))
            labels.append(
                target_label(
                    prediction=prediction,
                    final_prediction=final_prediction,
                    stability_prediction=stability_prediction,
                    target_mode=config.target_mode,
                )
            )
            sample_keys.append(sample_key)

    tensors = {
        "features": torch.tensor(features, dtype=torch.float32),
        "task_onehot": torch.tensor(task_onehots, dtype=torch.float32),
        "y_risk": torch.tensor(labels, dtype=torch.float32),
    }
    metadata = {
        "sample_keys": sample_keys,
        "task_to_idx": task_to_idx,
        "feature_fields": RISK_FEATURE_FIELDS,
        "signal_mode": config.signal_mode,
        "target_mode": config.target_mode,
    }
    return tensors, metadata


def target_label(
    *,
    prediction: str,
    final_prediction: str,
    stability_prediction: str,
    target_mode: str,
) -> float:
    if target_mode == "strict_prefix":
        return float(is_strict_prefix(prediction, final_prediction))
    if target_mode == "prediction_change":
        return float(bool(prediction or stability_prediction) and prediction != stability_prediction)
    raise ValueError(f"unknown target_mode: {target_mode}")


def prediction_stability_reference(sample_rows: list[dict[str, Any]]) -> str:
    previous = ""
    streak = 0
    for row in sample_rows:
        prediction = normalize_text(row.get("scored_prediction"))
        if prediction and prediction == previous:
            streak += 1
        else:
            streak = 1
            previous = prediction
        if prediction and streak >= 2:
            return prediction
    return normalize_text(sample_rows[-1].get("scored_prediction"))


def row_features(row: dict[str, Any], feature_fields: list[str] | None = None) -> list[float]:
    values = row_feature_values(row)
    fields = feature_fields or RISK_FEATURE_FIELDS
    return [values.get(field, 0.0) for field in fields]


def row_feature_values(row: dict[str, Any]) -> dict[str, float]:
    prediction = normalize_text(row.get("scored_prediction"))
    processed = normalize_text(row.get("official_processed_generation"))
    decoded = str(row.get("decode_text_so_far") or "")
    block_number = float(row["block_number"])
    max_budget = float(row["max_block_budget"])
    prediction_tokens = prediction.split()
    last_token = prediction_tokens[-1] if prediction_tokens else ""
    values = {
        "block_number": block_number,
        "max_block_budget": max_budget,
        "remaining_blocks": max_budget - block_number,
        "block_fraction": block_number / max(max_budget, 1.0),
        "token_entropy_mean": safe_float(row.get("token_entropy_mean")),
        "token_top_prob_mean": safe_float(row.get("token_top_prob_mean")),
        "eos_prob_max": safe_float(row.get("eos_prob_max")),
        "im_end_prob_max": safe_float(row.get("im_end_prob_max")),
        "stop_prob_max": safe_float(row.get("stop_prob_max")),
        "stop_prob_margin_vs_non_stop": safe_float(row.get("stop_prob_margin_vs_non_stop")),
        "contains_eos": bool_float(row.get("contains_eos")),
        "contains_im_end": bool_float(row.get("contains_im_end")),
        "contains_stop": bool_float(row.get("contains_stop")),
        "answer_text_nonempty": bool_float(row.get("answer_text_nonempty")),
        "answer_changed": bool_float(row.get("answer_changed")),
        "same_text_streak": safe_float(row.get("same_text_streak")),
        "scored_prediction_nonempty": bool_float(prediction),
        "scored_prediction_changed": bool_float(row.get("scored_prediction_changed")),
        "scored_prediction_same_streak": safe_float(row.get("scored_prediction_same_streak")),
        "processed_generation_changed": bool_float(row.get("processed_generation_changed")),
        "processed_generation_same_streak": safe_float(row.get("processed_generation_same_streak")),
        "already_stopped_before_block": bool_float(row.get("already_stopped_before_block")),
        "prediction_char_len": float(len(prediction)),
        "prediction_word_count": float(len(prediction_tokens)),
        "prediction_ends_alnum": bool_float(prediction[-1:].isalnum()),
        "prediction_ends_terminal_punct": bool_float(prediction.endswith((".", "!", "?", "\"", "'"))),
        "prediction_ends_mid_punct": bool_float(prediction.endswith((",", ":", ";", "-", "–", "—", "/"))),
        "prediction_contains_space": bool_float(" " in prediction),
        "prediction_is_numericish": bool_float(any(ch.isdigit() for ch in prediction)),
        "prediction_ends_decimal_point": bool_float(is_decimal_prefix(prediction)),
        "prediction_ends_single_letter_period": bool_float(is_single_letter_period_suffix(prediction)),
        "prediction_has_unbalanced_quote": bool_float(has_unbalanced_quote(prediction)),
        "prediction_has_unbalanced_bracket": bool_float(has_unbalanced_bracket(prediction)),
        "prediction_last_token_char_len": float(len(last_token)),
        "prediction_last_token_is_short": bool_float(0 < len(last_token.strip("\"'.,;:!?()[]{}")) <= 2),
        "processed_char_len": float(len(processed)),
        "processed_word_count": float(len(processed.split())),
        "decode_char_len": float(len(decoded)),
    }
    return values


def split_indices(sample_keys: list[str], config: ContinuationRiskTrainConfig) -> dict[str, list[int]]:
    splits = {"train": [], "valid": [], "test": []}
    for idx, key in enumerate(sample_keys):
        value = stable_uniform(f"{config.seed}:{key}")
        if value < config.train_ratio:
            splits["train"].append(idx)
        elif value < config.train_ratio + config.valid_ratio:
            splits["valid"].append(idx)
        else:
            splits["test"].append(idx)
    for name, indices in splits.items():
        if not indices:
            raise ValueError(f"empty split: {name}")
    return splits


def make_split_datasets(
    tensors: dict[str, torch.Tensor],
    splits: dict[str, list[int]],
) -> tuple[TensorDataset, TensorDataset, TensorDataset, dict[str, torch.Tensor]]:
    train_idx = torch.tensor(splits["train"], dtype=torch.long)
    feature_mean = tensors["features"][train_idx].mean(dim=0, keepdim=True)
    feature_std = tensors["features"][train_idx].std(dim=0, keepdim=True).clamp_min(1e-6)
    norm_features = (tensors["features"] - feature_mean) / feature_std
    norm_stats = {"feature_mean": feature_mean, "feature_std": feature_std}

    def dataset(indices: list[int]) -> TensorDataset:
        idx = torch.tensor(indices, dtype=torch.long)
        return TensorDataset(norm_features[idx], tensors["task_onehot"][idx], tensors["y_risk"][idx])

    return dataset(splits["train"]), dataset(splits["valid"]), dataset(splits["test"]), norm_stats


@torch.no_grad()
def evaluate(model: ContinuationRiskModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    probs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    losses: list[float] = []
    for batch in loader:
        features, task_onehot, y_risk = [item.to(device) for item in batch]
        logits = model(features, task_onehot)
        prob = torch.sigmoid(logits)
        loss = F.binary_cross_entropy(prob, y_risk)
        losses.append(float(loss.item()))
        probs.append(prob.cpu())
        targets.append(y_risk.cpu())
    y = torch.cat(targets)
    p = torch.cat(probs)
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "risk_accuracy": binary_accuracy(p, y),
        "risk_auroc": binary_auroc(p, y),
        "risk_auprc": binary_auprc(p, y),
        "risk_brier": float(torch.mean((p - y) ** 2).item()),
        "positive_rate": float(y.mean().item()),
    }


def save_checkpoint(
    path: Path,
    *,
    model: ContinuationRiskModel,
    optimizer: torch.optim.Optimizer,
    config: ContinuationRiskTrainConfig,
    norm_stats: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    step: int,
    metric: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "feature_fields": metadata["feature_fields"],
            "norm_stats": norm_stats,
            "metadata": {key: value for key, value in metadata.items() if key != "sample_keys"},
            "step": step,
            "metric": metric,
        },
        path,
    )


def write_metric(metrics_f: Any, split: str, step: int, metrics: dict[str, float]) -> None:
    metrics_f.write(
        json.dumps(
            {"created_at": int(time.time()), "split": split, "step": step, "metrics": metrics},
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )
    metrics_f.flush()


def select_metric(metrics: dict[str, float]) -> float:
    value = metrics.get("risk_auroc", float("nan"))
    if math.isnan(value):
        return -metrics["loss"]
    return value


def safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def bool_float(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def is_strict_prefix(value: str, final_value: str) -> bool:
    return bool(value and final_value and value != final_value and final_value.startswith(value))


def is_decimal_prefix(value: str) -> bool:
    return bool(re.search(r"\d+\.$", value))


def is_single_letter_period_suffix(value: str) -> bool:
    return bool(re.search(r"(?:^|\s)[a-z]\.$", value))


def has_unbalanced_quote(value: str) -> bool:
    return value.count('"') % 2 == 1 or value.count("'") % 2 == 1


def has_unbalanced_bracket(value: str) -> bool:
    pairs = [("(", ")"), ("[", "]"), ("{", "}")]
    return any(value.count(open_ch) != value.count(close_ch) for open_ch, close_ch in pairs)


def parse_args() -> ContinuationRiskTrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", default=ContinuationRiskTrainConfig.labels_dir)
    parser.add_argument("--output-dir", default=ContinuationRiskTrainConfig.output_dir)
    parser.add_argument("--tasks", default=ContinuationRiskTrainConfig.tasks)
    parser.add_argument("--seed", type=int, default=ContinuationRiskTrainConfig.seed)
    parser.add_argument("--train-ratio", type=float, default=ContinuationRiskTrainConfig.train_ratio)
    parser.add_argument("--valid-ratio", type=float, default=ContinuationRiskTrainConfig.valid_ratio)
    parser.add_argument("--batch-size", type=int, default=ContinuationRiskTrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=ContinuationRiskTrainConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=ContinuationRiskTrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=ContinuationRiskTrainConfig.weight_decay)
    parser.add_argument("--dropout", type=float, default=ContinuationRiskTrainConfig.dropout)
    parser.add_argument("--hidden-dim", type=int, default=ContinuationRiskTrainConfig.hidden_dim)
    parser.add_argument("--signal-mode", default=ContinuationRiskTrainConfig.signal_mode)
    parser.add_argument("--target-mode", default=ContinuationRiskTrainConfig.target_mode)
    parser.add_argument("--valid-interval", type=int, default=ContinuationRiskTrainConfig.valid_interval)
    parser.add_argument("--num-workers", type=int, default=ContinuationRiskTrainConfig.num_workers)
    parser.add_argument("--device", default=ContinuationRiskTrainConfig.device)
    parser.add_argument("--swanlab-mode", default=ContinuationRiskTrainConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=ContinuationRiskTrainConfig.experiment_name)
    args = parser.parse_args()
    parse_tasks(args.tasks)
    return ContinuationRiskTrainConfig(
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        tasks=args.tasks,
        seed=args.seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
        signal_mode=args.signal_mode,
        target_mode=args.target_mode,
        valid_interval=args.valid_interval,
        num_workers=args.num_workers,
        device=args.device,
        swanlab_mode=args.swanlab_mode,
        experiment_name=args.experiment_name,
    )


def main() -> None:
    summary = train_continuation_risk_model(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
