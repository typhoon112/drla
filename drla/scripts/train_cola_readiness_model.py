"""Train a multi-signal Cola readiness / halt model.

This is not a raw-latent binary classifier. It combines the raw latent block
with latent trajectory stats, decoder probe stats, EOS/im_end signals, text
stability signals, and task identity. Gold-derived fields are used only as
offline labels.
"""

from __future__ import annotations

import argparse
import hashlib
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

from drla.tracking import finish_experiment, init_experiment, log_metrics


OFFICIAL_COLA_TASKS = [
    "lambada",
    "mmlu",
    "obqa",
    "hellaswag",
    "race",
    "siqa",
    "squad",
    "story_cloze",
]


FEATURE_FIELDS = [
    "block_number",
    "max_block_budget",
    "latent_norm_mean",
    "latent_norm_std",
    "latent_delta_norm",
    "latent_delta_missing",
    "latent_cosine_to_prev",
    "latent_cosine_missing",
    "denoise_drift_norm_mean",
    "token_entropy_mean",
    "token_top_prob_mean",
    "eos_prob_max",
    "im_end_prob_max",
    "stop_prob_max",
    "stop_prob_margin_vs_non_stop",
    "answer_text_nonempty",
    "answer_changed",
    "same_text_streak",
    "scored_prediction_nonempty",
    "scored_prediction_changed",
    "scored_prediction_same_streak",
    "processed_generation_changed",
    "processed_generation_same_streak",
    "already_stopped_before_block",
    "contains_eos",
    "contains_im_end",
    "contains_stop",
]


@dataclass(frozen=True)
class ReadinessTrainConfig:
    labels_dir: str = "/data1/luyifei/drla/outputs/cola_readiness_frontiers/official8_1000_b20_t16_seed66_20260524"
    output_dir: str = "/data1/luyifei/drla/outputs/cola_readiness_model/official8_1000_b20_t16_seed66_20260524"
    tasks: str = ",".join(OFFICIAL_COLA_TASKS)
    seed: int = 20260524
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    batch_size: int = 256
    epochs: int = 40
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    hidden_dim: int = 192
    signal_mode: str = "full"
    valid_interval: int = 50
    num_workers: int = 0
    device: str = "auto"
    swanlab_mode: str = "cloud"
    experiment_name: str = "official8-readiness-model"


class ReadinessModel(nn.Module):
    def __init__(self, latent_dim: int, feature_dim: int, task_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim + task_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        joint_dim = hidden_dim + hidden_dim // 2
        self.joint = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.readiness_head = nn.Linear(hidden_dim, 1)
        self.correctness_head = nn.Linear(hidden_dim, 1)
        self.future_gain_head = nn.Linear(hidden_dim, 1)

    def forward(self, latent: torch.Tensor, features: torch.Tensor, task_onehot: torch.Tensor) -> dict[str, torch.Tensor]:
        latent_h = self.latent_encoder(self.latent_norm(latent))
        feature_h = self.feature_encoder(torch.cat([self.feature_norm(features), task_onehot], dim=-1))
        hidden = self.joint(torch.cat([latent_h, feature_h], dim=-1))
        return {
            "readiness_logits": self.readiness_head(hidden).squeeze(-1),
            "correctness_logits": self.correctness_head(hidden).squeeze(-1),
            "future_gain": self.future_gain_head(hidden).squeeze(-1),
        }


def train_readiness_model(config: ReadinessTrainConfig) -> dict[str, Any]:
    if config.signal_mode not in {"full", "process_only", "process_no_task", "latent_only"}:
        raise ValueError("signal_mode must be one of: full, process_only, process_no_task, latent_only")
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

    rows = load_training_rows(config)
    tensors, metadata = build_tensors(rows, config)
    splits = split_indices(metadata["sample_keys"], config)
    train_data, valid_data, test_data, norm_stats = make_split_datasets(tensors, splits)

    device = resolve_device(config.device)
    require_cuda_training(device, "train_cola_readiness_model.py")
    device_info = device_metadata(device)
    model = ReadinessModel(
        latent_dim=train_data.tensors[0].shape[1],
        feature_dim=train_data.tensors[1].shape[1],
        task_dim=train_data.tensors[2].shape[1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)

    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
    )
    valid_loader = DataLoader(valid_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    readiness_pos_weight = pos_weight(train_data.tensors[3]).to(device)
    correctness_pos_weight = pos_weight(train_data.tensors[4]).to(device)

    run = init_experiment(
        stage="cola-readiness-model",
        experiment_name=config.experiment_name,
        description="Multi-signal readiness/halt model over official Cola block traces.",
        config={
            **asdict(config),
            "device_info": device_info,
            "feature_fields": FEATURE_FIELDS,
            "signal_mode": config.signal_mode,
            "num_rows": len(rows),
            "split_sizes": {name: len(indices) for name, indices in splits.items()},
        },
        mode=config.swanlab_mode,
        tags=["cola", "official-benchmark", "readiness", "halt-model"],
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
                    batch = [item.to(device) for item in batch]
                    loss, batch_metrics = compute_loss(
                        model,
                        batch,
                        readiness_pos_weight=readiness_pos_weight,
                        correctness_pos_weight=correctness_pos_weight,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    train_metrics = {
                        "loss": float(loss.item()),
                        **{f"train_{key}": value for key, value in batch_metrics.items()},
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
            "device_info": device_info,
            "feature_fields": metadata["feature_fields"],
            "signal_mode": config.signal_mode,
            "num_rows": len(rows),
            "split_sizes": {name: len(indices) for name, indices in splits.items()},
            "best_step": best_step,
            "best_metric_name": "valid/readiness_auroc",
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


def compute_loss(
    model: ReadinessModel,
    batch: list[torch.Tensor],
    *,
    readiness_pos_weight: torch.Tensor,
    correctness_pos_weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    latent, features, task_onehot, y_ready, y_correct, y_future = batch
    outputs = model(latent, features, task_onehot)
    readiness_loss = F.binary_cross_entropy_with_logits(
        outputs["readiness_logits"],
        y_ready,
        pos_weight=readiness_pos_weight,
    )
    correctness_loss = F.binary_cross_entropy_with_logits(
        outputs["correctness_logits"],
        y_correct,
        pos_weight=correctness_pos_weight,
    )
    future_loss = F.mse_loss(outputs["future_gain"], y_future)
    loss = readiness_loss + 0.5 * correctness_loss + 0.25 * future_loss
    with torch.no_grad():
        readiness_prob = torch.sigmoid(outputs["readiness_logits"])
        correctness_prob = torch.sigmoid(outputs["correctness_logits"])
        metrics = {
            "readiness_loss": float(readiness_loss.item()),
            "correctness_loss": float(correctness_loss.item()),
            "future_loss": float(future_loss.item()),
            "readiness_accuracy": binary_accuracy(readiness_prob, y_ready),
            "correctness_accuracy": binary_accuracy(correctness_prob, y_correct),
        }
    return loss, metrics


@torch.no_grad()
def evaluate(model: ReadinessModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    readiness_probs: list[torch.Tensor] = []
    correctness_probs: list[torch.Tensor] = []
    future_preds: list[torch.Tensor] = []
    readiness_targets: list[torch.Tensor] = []
    correctness_targets: list[torch.Tensor] = []
    future_targets: list[torch.Tensor] = []
    losses: list[float] = []

    for batch in loader:
        batch = [item.to(device) for item in batch]
        latent, features, task_onehot, y_ready, y_correct, y_future = batch
        outputs = model(latent, features, task_onehot)
        readiness = torch.sigmoid(outputs["readiness_logits"])
        correctness = torch.sigmoid(outputs["correctness_logits"])
        loss = (
            F.binary_cross_entropy(readiness, y_ready)
            + 0.5 * F.binary_cross_entropy(correctness, y_correct)
            + 0.25 * F.mse_loss(outputs["future_gain"], y_future)
        )
        losses.append(float(loss.item()))
        readiness_probs.append(readiness.cpu())
        correctness_probs.append(correctness.cpu())
        future_preds.append(outputs["future_gain"].cpu())
        readiness_targets.append(y_ready.cpu())
        correctness_targets.append(y_correct.cpu())
        future_targets.append(y_future.cpu())

    y_ready = torch.cat(readiness_targets)
    p_ready = torch.cat(readiness_probs)
    y_correct = torch.cat(correctness_targets)
    p_correct = torch.cat(correctness_probs)
    y_future = torch.cat(future_targets)
    p_future = torch.cat(future_preds)
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "readiness_accuracy": binary_accuracy(p_ready, y_ready),
        "readiness_auroc": binary_auroc(p_ready, y_ready),
        "readiness_auprc": binary_auprc(p_ready, y_ready),
        "readiness_brier": float(torch.mean((p_ready - y_ready) ** 2).item()),
        "correctness_accuracy": binary_accuracy(p_correct, y_correct),
        "correctness_auroc": binary_auroc(p_correct, y_correct),
        "future_gain_mse": float(torch.mean((p_future - y_future) ** 2).item()),
    }


def load_training_rows(config: ReadinessTrainConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in parse_tasks(config.tasks):
        path = Path(config.labels_dir) / f"{task}_readiness_labels.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing labels: {path}")
        with path.open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def build_tensors(
    rows: list[dict[str, Any]],
    config: ReadinessTrainConfig,
    feature_fields: list[str] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    feature_fields = feature_fields or FEATURE_FIELDS
    add_derived_stability_features(rows)
    latent_cache: dict[str, torch.Tensor] = {}
    needs_latent = config.signal_mode in {"full", "latent_only"}
    latents: list[torch.Tensor] = []
    features: list[list[float]] = []
    task_onehots: list[list[float]] = []
    y_ready: list[float] = []
    y_correct: list[float] = []
    y_future: list[float] = []
    sample_keys: list[str] = []
    task_to_idx = {task: idx for idx, task in enumerate(OFFICIAL_COLA_TASKS)}

    for row in rows:
        if needs_latent:
            latent_path = row["latent_batch_path"]
            if latent_path not in latent_cache:
                latent_cache[latent_path] = torch.load(latent_path, map_location="cpu")["latent_blocks"]
            latent_block = latent_cache[latent_path][
                int(row["latent_batch_sample_index"]),
                int(row["latent_batch_block_index"]),
            ].float()
            latents.append(latent_block.flatten())
        else:
            latents.append(torch.zeros(1, dtype=torch.float32))
        features.append(row_features(row, feature_fields))
        task_vec = [0.0] * len(OFFICIAL_COLA_TASKS)
        task_vec[task_to_idx[row["task"]]] = 1.0
        task_onehots.append(task_vec)
        y_ready.append(float(row["is_at_or_after_oracle_frontier"]))
        y_correct.append(float(row["official_correct"]))
        y_future.append(float(row["future_gain_correct"]))
        sample_keys.append(f"{row['task']}::{row['sample_id']}")

    tensors = {
        "latent": torch.stack(latents),
        "features": torch.tensor(features, dtype=torch.float32),
        "task_onehot": torch.tensor(task_onehots, dtype=torch.float32),
        "y_ready": torch.tensor(y_ready, dtype=torch.float32),
        "y_correct": torch.tensor(y_correct, dtype=torch.float32),
        "y_future": torch.tensor(y_future, dtype=torch.float32),
    }
    metadata = {
        "sample_keys": sample_keys,
        "task_to_idx": task_to_idx,
        "feature_fields": feature_fields,
        "signal_mode": config.signal_mode,
    }
    apply_signal_mode(tensors, config.signal_mode)
    return tensors, metadata


def apply_signal_mode(tensors: dict[str, torch.Tensor], signal_mode: str) -> None:
    row_count = tensors["latent"].shape[0]
    if signal_mode == "full":
        return
    if signal_mode in {"process_only", "process_no_task"}:
        tensors["latent"] = torch.zeros(row_count, 1, dtype=torch.float32)
    if signal_mode == "latent_only":
        tensors["features"] = torch.zeros(row_count, 1, dtype=torch.float32)
        tensors["task_onehot"] = torch.zeros_like(tensors["task_onehot"])
    if signal_mode == "process_no_task":
        tensors["task_onehot"] = torch.zeros_like(tensors["task_onehot"])


def add_derived_stability_features(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(f"{row['task']}::{row['sample_id']}", []).append(row)

    for sample_rows in grouped.values():
        sample_rows.sort(key=lambda item: int(item["block_index"]))
        previous_prediction: str | None = None
        prediction_streak = 0
        previous_processed: str | None = None
        processed_streak = 0
        for row in sample_rows:
            prediction = normalize_text(row.get("scored_prediction"))
            if prediction:
                prediction_changed = previous_prediction is not None and prediction != previous_prediction
                prediction_streak = prediction_streak + 1 if prediction == previous_prediction else 1
                previous_prediction = prediction
            else:
                prediction_changed = False
                prediction_streak = 0
                previous_prediction = None

            processed = normalize_text(row.get("official_processed_generation"))
            if processed:
                processed_changed = previous_processed is not None and processed != previous_processed
                processed_streak = processed_streak + 1 if processed == previous_processed else 1
                previous_processed = processed
            else:
                processed_changed = False
                processed_streak = 0
                previous_processed = None

            row["scored_prediction_nonempty"] = bool(prediction)
            row["scored_prediction_changed"] = prediction_changed
            row["scored_prediction_same_streak"] = prediction_streak
            row["processed_generation_changed"] = processed_changed
            row["processed_generation_same_streak"] = processed_streak


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def row_features(row: dict[str, Any], feature_fields: list[str]) -> list[float]:
    delta = row.get("latent_delta_norm")
    cosine = row.get("latent_cosine_to_prev")
    values = {
        "block_number": float(row["block_number"]),
        "max_block_budget": float(row["max_block_budget"]),
        "latent_norm_mean": safe_float(row.get("latent_norm_mean")),
        "latent_norm_std": safe_float(row.get("latent_norm_std")),
        "latent_delta_norm": safe_float(delta),
        "latent_delta_missing": 1.0 if delta is None else 0.0,
        "latent_cosine_to_prev": safe_float(cosine),
        "latent_cosine_missing": 1.0 if cosine is None else 0.0,
        "denoise_drift_norm_mean": safe_float(row.get("denoise_drift_norm_mean")),
        "token_entropy_mean": safe_float(row.get("token_entropy_mean")),
        "token_top_prob_mean": safe_float(row.get("token_top_prob_mean")),
        "eos_prob_max": safe_float(row.get("eos_prob_max")),
        "im_end_prob_max": safe_float(row.get("im_end_prob_max")),
        "stop_prob_max": safe_float(row.get("stop_prob_max")),
        "stop_prob_margin_vs_non_stop": safe_float(row.get("stop_prob_margin_vs_non_stop")),
        "answer_text_nonempty": bool_float(row.get("answer_text_nonempty")),
        "answer_changed": bool_float(row.get("answer_changed")),
        "same_text_streak": safe_float(row.get("same_text_streak")),
        "scored_prediction_nonempty": bool_float(row.get("scored_prediction_nonempty")),
        "scored_prediction_changed": bool_float(row.get("scored_prediction_changed")),
        "scored_prediction_same_streak": safe_float(row.get("scored_prediction_same_streak")),
        "processed_generation_changed": bool_float(row.get("processed_generation_changed")),
        "processed_generation_same_streak": safe_float(row.get("processed_generation_same_streak")),
        "already_stopped_before_block": bool_float(row.get("already_stopped_before_block")),
        "contains_eos": bool_float(row.get("contains_eos")),
        "contains_im_end": bool_float(row.get("contains_im_end")),
        "contains_stop": bool_float(row.get("contains_stop")),
    }
    return [values[field] for field in feature_fields]


def make_split_datasets(
    tensors: dict[str, torch.Tensor],
    splits: dict[str, list[int]],
) -> tuple[TensorDataset, TensorDataset, TensorDataset, dict[str, torch.Tensor]]:
    train_idx = torch.tensor(splits["train"], dtype=torch.long)
    latent_mean = tensors["latent"][train_idx].mean(dim=0, keepdim=True)
    latent_std = tensors["latent"][train_idx].std(dim=0, keepdim=True).clamp_min(1e-6)
    feature_mean = tensors["features"][train_idx].mean(dim=0, keepdim=True)
    feature_std = tensors["features"][train_idx].std(dim=0, keepdim=True).clamp_min(1e-6)

    norm_latent = (tensors["latent"] - latent_mean) / latent_std
    norm_features = (tensors["features"] - feature_mean) / feature_std
    norm_stats = {
        "latent_mean": latent_mean,
        "latent_std": latent_std,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }

    def dataset(indices: list[int]) -> TensorDataset:
        idx = torch.tensor(indices, dtype=torch.long)
        return TensorDataset(
            norm_latent[idx],
            norm_features[idx],
            tensors["task_onehot"][idx],
            tensors["y_ready"][idx],
            tensors["y_correct"][idx],
            tensors["y_future"][idx],
        )

    return dataset(splits["train"]), dataset(splits["valid"]), dataset(splits["test"]), norm_stats


def split_indices(sample_keys: list[str], config: ReadinessTrainConfig) -> dict[str, list[int]]:
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


def stable_uniform(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def save_checkpoint(
    path: Path,
    *,
    model: ReadinessModel,
    optimizer: torch.optim.Optimizer,
    config: ReadinessTrainConfig,
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
            {
                "created_at": int(time.time()),
                "split": split,
                "step": step,
                "metrics": metrics,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )
    metrics_f.flush()


def select_metric(metrics: dict[str, float]) -> float:
    value = metrics.get("readiness_auroc", float("nan"))
    if math.isnan(value):
        return -metrics["loss"]
    return value


def pos_weight(target: torch.Tensor) -> torch.Tensor:
    positives = target.sum().clamp_min(1.0)
    negatives = (target.numel() - target.sum()).clamp_min(1.0)
    return negatives / positives


def binary_accuracy(prob: torch.Tensor, target: torch.Tensor) -> float:
    return float(((prob >= 0.5).float() == target).float().mean().item())


def binary_auroc(prob: torch.Tensor, target: torch.Tensor) -> float:
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


def binary_auprc(prob: torch.Tensor, target: torch.Tensor) -> float:
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


def safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def bool_float(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def parse_tasks(value: str) -> list[str]:
    tasks = [task.strip() for task in value.split(",") if task.strip()]
    invalid = [task for task in tasks if task not in OFFICIAL_COLA_TASKS]
    if invalid:
        raise ValueError(f"unknown Cola tasks: {invalid}")
    return tasks


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def require_cuda_training(device: torch.device, script_name: str) -> None:
    if device.type != "cuda":
        raise RuntimeError(
            f"{script_name} is a deep-learning training script and must run on CUDA/GPU; "
            f"resolved device is {device}. Pass --device cuda or fix the CUDA environment."
        )


def device_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "requested_resolved_device": str(device),
        "device_type": device.type,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        metadata["cuda_device_index"] = int(index)
        metadata["cuda_device_name"] = torch.cuda.get_device_name(index)
    return metadata


def parse_args() -> ReadinessTrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", default=ReadinessTrainConfig.labels_dir)
    parser.add_argument("--output-dir", default=ReadinessTrainConfig.output_dir)
    parser.add_argument("--tasks", default=ReadinessTrainConfig.tasks)
    parser.add_argument("--seed", type=int, default=ReadinessTrainConfig.seed)
    parser.add_argument("--train-ratio", type=float, default=ReadinessTrainConfig.train_ratio)
    parser.add_argument("--valid-ratio", type=float, default=ReadinessTrainConfig.valid_ratio)
    parser.add_argument("--batch-size", type=int, default=ReadinessTrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=ReadinessTrainConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=ReadinessTrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=ReadinessTrainConfig.weight_decay)
    parser.add_argument("--dropout", type=float, default=ReadinessTrainConfig.dropout)
    parser.add_argument("--hidden-dim", type=int, default=ReadinessTrainConfig.hidden_dim)
    parser.add_argument("--signal-mode", default=ReadinessTrainConfig.signal_mode)
    parser.add_argument("--valid-interval", type=int, default=ReadinessTrainConfig.valid_interval)
    parser.add_argument("--num-workers", type=int, default=ReadinessTrainConfig.num_workers)
    parser.add_argument("--device", default=ReadinessTrainConfig.device)
    parser.add_argument("--swanlab-mode", default=ReadinessTrainConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=ReadinessTrainConfig.experiment_name)
    args = parser.parse_args()
    return ReadinessTrainConfig(
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
        valid_interval=args.valid_interval,
        num_workers=args.num_workers,
        device=args.device,
        swanlab_mode=args.swanlab_mode,
        experiment_name=args.experiment_name,
    )


def main() -> None:
    summary = train_readiness_model(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
