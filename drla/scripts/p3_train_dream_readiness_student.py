"""Train P3 DreamStepReadinessStudent from offline frontier labels.

This is a deep-learning training script. It trains a causal trajectory student
over decoder-free Dream step features (hidden summaries, logit summaries, mask
and process features). Decoded text, gold answers, scorer outputs, and oracle
labels are used only as supervision/evaluation targets and are never model
inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.scripts.train_cola_readiness_model import device_metadata, require_cuda_training, resolve_device  # noqa: E402
from drla.tracking import finish_experiment, init_experiment, log_metrics  # noqa: E402


DEFAULT_FRONTIER_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_readiness_frontiers/"
    "musique_calibration_trace_full200_steps64_stride4_hidden_summary_frontier_20260606"
)
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_readiness_students/"
    "dream_step_readiness_student_v1_full200_seed20260606"
)

FEATURE_NAMES = [
    "step_norm",
    "event_index_norm",
    "has_hidden_summary",
    "has_logit_stats",
    "num_mask_tokens_norm",
    "changed_tokens_norm",
    "top1_prob_mean",
    "top2_margin_mean",
    "entropy_norm",
    "hidden_mean",
    "hidden_std",
    "hidden_abs_mean",
    "hidden_l2_mean_norm",
    "hidden_last_token_l2_norm",
    "condition_single_full_info",
    "condition_textmas_matched",
]


@dataclass(frozen=True)
class TrainConfig:
    frontier_dir: str = DEFAULT_FRONTIER_DIR
    output_dir: str = DEFAULT_OUTPUT_DIR
    device: str = "auto"
    seed: int = 20260606
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    batch_size: int = 32
    epochs: int = 40
    max_train_steps: int = 0
    valid_interval: int = 10
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    future_gain_loss_weight: float = 0.25
    prediction_change_loss_weight: float = 0.5
    final_match_loss_weight: float = 0.5
    swanlab_mode: str = "cloud"
    experiment_name: str = "p3-dream-step-readiness-student-v1-full200"


class FrontierSequenceDataset(Dataset):
    def __init__(self, sequences: list[dict[str, Any]], feature_stats: dict[str, list[float]]) -> None:
        self.sequences = sequences
        self.feature_mean = torch.tensor(feature_stats["mean"], dtype=torch.float32)
        self.feature_std = torch.tensor(feature_stats["std"], dtype=torch.float32).clamp_min(1e-6)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.sequences[index]
        features = (torch.tensor(item["features"], dtype=torch.float32) - self.feature_mean) / self.feature_std
        return {
            "features": features,
            "ready": torch.tensor(item["ready"], dtype=torch.float32),
            "future_gain": torch.tensor(item["future_gain"], dtype=torch.float32),
            "prediction_change": torch.tensor(item["prediction_change"], dtype=torch.float32),
            "final_match": torch.tensor(item["final_match"], dtype=torch.float32),
            "row_id": item["row_id"],
            "sample_id": item["sample_id"],
            "condition": item["condition"],
        }


class DreamStepReadinessStudent(nn.Module):
    def __init__(self, feature_dim: int, d_model: int, num_layers: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.ready_head = nn.Linear(d_model, 1)
        self.future_gain_head = nn.Linear(d_model, 1)
        self.prediction_change_head = nn.Linear(d_model, 1)
        self.final_match_head = nn.Linear(d_model, 1)

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.input(features)
        seq_len = hidden.shape[1]
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=hidden.device),
            diagonal=1,
        )
        encoded = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)
        encoded = self.norm(encoded)
        return {
            "ready_logit": self.ready_head(encoded).squeeze(-1),
            "future_gain": self.future_gain_head(encoded).squeeze(-1),
            "prediction_change_logit": self.prediction_change_head(encoded).squeeze(-1),
            "final_match_logit": self.final_match_head(encoded).squeeze(-1),
        }


def main() -> None:
    summary = train(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-dir", default=TrainConfig.frontier_dir)
    parser.add_argument("--output-dir", default=TrainConfig.output_dir)
    parser.add_argument("--device", default=TrainConfig.device)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--train-ratio", type=float, default=TrainConfig.train_ratio)
    parser.add_argument("--valid-ratio", type=float, default=TrainConfig.valid_ratio)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--max-train-steps", type=int, default=TrainConfig.max_train_steps)
    parser.add_argument("--valid-interval", type=int, default=TrainConfig.valid_interval)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=TrainConfig.grad_clip_norm)
    parser.add_argument("--d-model", type=int, default=TrainConfig.d_model)
    parser.add_argument("--num-layers", type=int, default=TrainConfig.num_layers)
    parser.add_argument("--num-heads", type=int, default=TrainConfig.num_heads)
    parser.add_argument("--dropout", type=float, default=TrainConfig.dropout)
    parser.add_argument("--future-gain-loss-weight", type=float, default=TrainConfig.future_gain_loss_weight)
    parser.add_argument("--prediction-change-loss-weight", type=float, default=TrainConfig.prediction_change_loss_weight)
    parser.add_argument("--final-match-loss-weight", type=float, default=TrainConfig.final_match_loss_weight)
    parser.add_argument("--swanlab-mode", default=TrainConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=TrainConfig.experiment_name)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def train(config: TrainConfig) -> dict[str, Any]:
    if config.swanlab_mode != "cloud":
        raise ValueError("D5 training must use SwanLab cloud; pass --swanlab-mode cloud")
    if config.valid_interval > 10:
        raise ValueError("valid_interval must be <= 10 for current P3 training")
    set_seed(config.seed)
    device = resolve_device(config.device)
    require_cuda_training(device, "p3_train_dream_readiness_student.py")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences, metadata = load_sequences(Path(config.frontier_dir))
    splits = split_by_sample(sequences, config.seed, config.train_ratio, config.valid_ratio)
    feature_stats = compute_feature_stats(splits["train"])
    datasets = {name: FrontierSequenceDataset(items, feature_stats) for name, items in splits.items()}
    loaders = {
        name: DataLoader(dataset, batch_size=config.batch_size, shuffle=(name == "train"), collate_fn=collate_batch)
        for name, dataset in datasets.items()
    }
    ready_pos_weight = compute_pos_weight(splits["train"], "ready")
    model = DreamStepReadinessStudent(
        feature_dim=len(FEATURE_NAMES),
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    run = init_experiment(
        stage="p3-dream-readiness-student",
        config={
            **asdict(config),
            **device_metadata(device),
            "feature_names": FEATURE_NAMES,
            "ready_pos_weight": ready_pos_weight,
            "split_sizes": {name: len(items) for name, items in splits.items()},
            "metadata": metadata,
        },
        experiment_name=config.experiment_name,
        tags=["dream", "p3", "readiness-student", "latent-halt", "swanlab-cloud"],
        mode=config.swanlab_mode,
    )

    metrics_path = output_dir / "metrics.jsonl"
    metrics_f = metrics_path.open("w", encoding="utf-8")
    best_metric = float("-inf")
    best_step = 0
    global_step = 0
    final_valid_metrics: dict[str, float] = {}
    final_test_metrics: dict[str, float] = {}
    try:
        for epoch in range(config.epochs):
            model.train()
            for batch in loaders["train"]:
                global_step += 1
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch["features"], batch["padding_mask"])
                loss, train_metrics = compute_loss(outputs, batch, config, ready_pos_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                optimizer.step()
                train_metrics = {"loss": float(loss.detach().item()), **train_metrics, "epoch": float(epoch)}
                write_metrics(metrics_f, "train", global_step, train_metrics)
                log_metrics(train_metrics, step=global_step, prefix="train")
                if global_step % config.valid_interval == 0:
                    valid_metrics = evaluate(model, loaders["valid"], device, config, ready_pos_weight)
                    write_metrics(metrics_f, "valid", global_step, valid_metrics)
                    log_metrics(valid_metrics, step=global_step, prefix="valid")
                    if valid_metrics["selection_metric"] > best_metric:
                        best_metric = valid_metrics["selection_metric"]
                        best_step = global_step
                        save_checkpoint(output_dir / "best_checkpoint.pt", model, optimizer, config, metadata, feature_stats, best_step, best_metric)
                if config.max_train_steps and global_step >= config.max_train_steps:
                    break
            if config.max_train_steps and global_step >= config.max_train_steps:
                break
        final_valid_metrics = evaluate(model, loaders["valid"], device, config, ready_pos_weight)
        final_test_metrics = evaluate(model, loaders["test"], device, config, ready_pos_weight)
        write_metrics(metrics_f, "valid", global_step, final_valid_metrics)
        write_metrics(metrics_f, "test", global_step, final_test_metrics)
        log_metrics(final_valid_metrics, step=global_step, prefix="valid")
        log_metrics(final_test_metrics, step=global_step, prefix="test")
        if final_valid_metrics["selection_metric"] > best_metric:
            best_metric = final_valid_metrics["selection_metric"]
            best_step = global_step
            save_checkpoint(output_dir / "best_checkpoint.pt", model, optimizer, config, metadata, feature_stats, best_step, best_metric)
        save_checkpoint(output_dir / "last_checkpoint.pt", model, optimizer, config, metadata, feature_stats, global_step, final_valid_metrics["selection_metric"])
    finally:
        metrics_f.close()
        finish_experiment()

    best_valid_metrics = evaluate_checkpoint(output_dir / "best_checkpoint.pt", loaders["valid"], device, config, ready_pos_weight)
    best_test_metrics = evaluate_checkpoint(output_dir / "best_checkpoint.pt", loaders["test"], device, config, ready_pos_weight)
    summary = {
        "created_at": int(time.time()),
        "status": "pass",
        "config": asdict(config),
        "swanlab_run_id": getattr(run, "id", None),
        "metadata": metadata,
        "feature_names": FEATURE_NAMES,
        "feature_stats": feature_stats,
        "split_sizes": {name: len(items) for name, items in splits.items()},
        "global_step": global_step,
        "best_step": best_step,
        "best_valid_selection_metric": best_metric,
        "final_valid_metrics": final_valid_metrics,
        "final_test_metrics": final_test_metrics,
        "best_valid_metrics": best_valid_metrics,
        "best_test_metrics": best_test_metrics,
        "artifacts": {
            "metrics_jsonl": str(metrics_path),
            "best_checkpoint": str(output_dir / "best_checkpoint.pt"),
            "last_checkpoint": str(output_dir / "last_checkpoint.pt"),
            "summary_json": str(output_dir / "summary.json"),
        },
        "execution_boundary": [
            "P3 D5 deep-learning training",
            "CUDA/GPU required",
            "SwanLab cloud required",
            "decoded/gold/scorer fields are supervision only",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def load_sequences(frontier_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = read_jsonl(frontier_dir / "frontier_events.jsonl")
    summary = json.loads((frontier_dir / "summary.json").read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[str(event["row_id"])].append(event)
    sequences = []
    for row_id, row_events in grouped.items():
        row_events = sorted(row_events, key=lambda item: int(item.get("trace_event_index", 0)))
        first = row_events[0]
        features = [event_features(event) for event in row_events]
        sequences.append(
            {
                "row_id": row_id,
                "sample_id": first.get("sample_id", ""),
                "condition": first.get("condition", ""),
                "features": features,
                "ready": [float(event.get("answer_ready_correct_and_final_stable", False)) for event in row_events],
                "future_gain": [float(event.get("future_gain_vs_final", 0.0)) for event in row_events],
                "prediction_change": [float(event.get("prediction_changes_next_event", False)) for event in row_events],
                "final_match": [float(event.get("prediction_matches_final", False)) for event in row_events],
            }
        )
    metadata = {
        "frontier_dir": str(frontier_dir),
        "frontier_summary": summary,
        "num_sequences": len(sequences),
        "num_events": len(events),
        "condition_counts": dict(Counter(item["condition"] for item in sequences)),
    }
    return sequences, metadata


def event_features(event: dict[str, Any]) -> list[float]:
    condition = str(event.get("condition", ""))
    step = float(event.get("step") or 0.0)
    event_index = float(event.get("trace_event_index") or 0.0)
    num_mask = float(event.get("num_mask_tokens") or 0.0)
    changed = float(event.get("changed_suffix_tokens_vs_prev_hook") or 0.0)
    entropy = float(event.get("entropy_mean") or 0.0)
    hidden = event.get("hidden_summary") or {}
    return [
        step / 64.0,
        event_index / 32.0,
        float(bool(event.get("has_hidden_summary", False))),
        float(bool(event.get("has_logit_stats", False))),
        num_mask / 128.0,
        changed / 128.0,
        float(event.get("top1_prob_mean") or 0.0),
        float(event.get("top2_margin_mean") or 0.0),
        entropy / 12.0,
        float(hidden.get("mean") or 0.0),
        float(hidden.get("std") or 0.0),
        float(hidden.get("abs_mean") or 0.0),
        float(hidden.get("l2_mean") or 0.0) / 512.0,
        float(hidden.get("last_token_l2") or 0.0) / 512.0,
        float(condition == "single_full_info"),
        float(condition == "textmas_matched"),
    ]


def split_by_sample(
    sequences: list[dict[str, Any]],
    seed: int,
    train_ratio: float,
    valid_ratio: float,
) -> dict[str, list[dict[str, Any]]]:
    sample_ids = sorted({str(item["sample_id"]) for item in sequences})
    rng = random.Random(seed)
    rng.shuffle(sample_ids)
    n_train = max(1, int(len(sample_ids) * train_ratio))
    n_valid = max(1, int(len(sample_ids) * valid_ratio))
    train_ids = set(sample_ids[:n_train])
    valid_ids = set(sample_ids[n_train : n_train + n_valid])
    test_ids = set(sample_ids[n_train + n_valid :])
    if not test_ids:
        test_ids = set(sample_ids[-n_valid:])
        valid_ids = set(sample_ids[n_train : -n_valid])
    splits = {"train": [], "valid": [], "test": []}
    for item in sequences:
        sample_id = str(item["sample_id"])
        if sample_id in train_ids:
            splits["train"].append(item)
        elif sample_id in valid_ids:
            splits["valid"].append(item)
        elif sample_id in test_ids:
            splits["test"].append(item)
    return splits


def compute_feature_stats(sequences: list[dict[str, Any]]) -> dict[str, list[float]]:
    values = torch.tensor([features for item in sequences for features in item["features"]], dtype=torch.float32)
    return {
        "mean": values.mean(dim=0).tolist(),
        "std": values.std(dim=0, unbiased=False).clamp_min(1e-6).tolist(),
    }


def compute_pos_weight(sequences: list[dict[str, Any]], key: str) -> float:
    labels = [float(label) for item in sequences for label in item[key]]
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives <= 0:
        return 1.0
    return float(negatives / positives)


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = torch.tensor([item["features"].shape[0] for item in items], dtype=torch.long)
    features = pad_sequence([item["features"] for item in items], batch_first=True)
    ready = pad_sequence([item["ready"] for item in items], batch_first=True)
    future_gain = pad_sequence([item["future_gain"] for item in items], batch_first=True)
    prediction_change = pad_sequence([item["prediction_change"] for item in items], batch_first=True)
    final_match = pad_sequence([item["final_match"] for item in items], batch_first=True)
    mask = torch.arange(features.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    return {
        "features": features,
        "ready": ready,
        "future_gain": future_gain,
        "prediction_change": prediction_change,
        "final_match": final_match,
        "mask": mask,
        "padding_mask": ~mask,
        "lengths": lengths,
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: TrainConfig,
    ready_pos_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["mask"]
    ready_loss = masked_bce(outputs["ready_logit"], batch["ready"], mask, pos_weight=ready_pos_weight)
    change_loss = masked_bce(outputs["prediction_change_logit"], batch["prediction_change"], mask)
    final_match_loss = masked_bce(outputs["final_match_logit"], batch["final_match"], mask)
    future_gain_loss = masked_mse(outputs["future_gain"], batch["future_gain"], mask)
    loss = (
        ready_loss
        + config.prediction_change_loss_weight * change_loss
        + config.final_match_loss_weight * final_match_loss
        + config.future_gain_loss_weight * future_gain_loss
    )
    return loss, {
        "ready_bce": float(ready_loss.detach().item()),
        "prediction_change_bce": float(change_loss.detach().item()),
        "final_match_bce": float(final_match_loss.detach().item()),
        "future_gain_mse": float(future_gain_loss.detach().item()),
    }


def masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, pos_weight: float = 1.0) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=torch.tensor(pos_weight, device=logits.device),
    )
    return (loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = (pred - target).pow(2)
    return (loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(
    model: DreamStepReadinessStudent,
    loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
    ready_pos_weight: float,
) -> dict[str, float]:
    model.eval()
    losses = []
    ready_scores = []
    ready_labels = []
    change_scores = []
    change_labels = []
    final_match_scores = []
    final_match_labels = []
    future_pred = []
    future_labels = []
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(batch["features"], batch["padding_mask"])
        loss, _ = compute_loss(outputs, batch, config, ready_pos_weight)
        losses.append(float(loss.item()))
        mask = batch["mask"]
        ready_scores.extend(torch.sigmoid(outputs["ready_logit"])[mask].detach().cpu().tolist())
        ready_labels.extend(batch["ready"][mask].detach().cpu().tolist())
        change_scores.extend(torch.sigmoid(outputs["prediction_change_logit"])[mask].detach().cpu().tolist())
        change_labels.extend(batch["prediction_change"][mask].detach().cpu().tolist())
        final_match_scores.extend(torch.sigmoid(outputs["final_match_logit"])[mask].detach().cpu().tolist())
        final_match_labels.extend(batch["final_match"][mask].detach().cpu().tolist())
        future_pred.extend(outputs["future_gain"][mask].detach().cpu().tolist())
        future_labels.extend(batch["future_gain"][mask].detach().cpu().tolist())
    ready_auroc = binary_auroc(ready_scores, ready_labels)
    change_auroc = binary_auroc(change_scores, change_labels)
    final_match_auroc = binary_auroc(final_match_scores, final_match_labels)
    future_mse = mean([(p - y) ** 2 for p, y in zip(future_pred, future_labels)])
    metrics = {
        "loss": mean(losses),
        "ready_auroc": ready_auroc,
        "ready_accuracy_at_05": binary_accuracy(ready_scores, ready_labels, 0.5),
        "ready_brier": mean([(s - y) ** 2 for s, y in zip(ready_scores, ready_labels)]),
        "prediction_change_auroc": change_auroc,
        "final_match_auroc": final_match_auroc,
        "future_gain_mse": future_mse,
        "num_events": float(len(ready_labels)),
        "ready_positive_rate": mean(ready_labels),
    }
    metrics["selection_metric"] = ready_auroc + 0.25 * final_match_auroc + 0.25 * change_auroc - 0.1 * future_mse
    return metrics


def evaluate_checkpoint(
    path: Path,
    loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
    ready_pos_weight: float,
) -> dict[str, float]:
    checkpoint = torch.load(path, map_location=device)
    model = DreamStepReadinessStudent(
        feature_dim=len(FEATURE_NAMES),
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return evaluate(model, loader, device, config, ready_pos_weight)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    metadata: dict[str, Any],
    feature_stats: dict[str, list[float]],
    step: int,
    selection_metric: float,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(config),
            "metadata": metadata,
            "feature_names": FEATURE_NAMES,
            "feature_stats": feature_stats,
            "step": step,
            "selection_metric": selection_metric,
        },
        path,
    )


def binary_auroc(scores: list[float], labels: list[float]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    pos = sum(1 for _, label in pairs if label > 0.5)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return 0.5
    rank_sum = 0.0
    for rank, (_, label) in enumerate(pairs, start=1):
        if label > 0.5:
            rank_sum += rank
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def binary_accuracy(scores: list[float], labels: list[float], threshold: float) -> float:
    if not labels:
        return 0.0
    correct = sum((score >= threshold) == (label > 0.5) for score, label in zip(scores, labels))
    return correct / len(labels)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_metrics(handle, phase: str, step: int, metrics: dict[str, float]) -> None:
    handle.write(json.dumps({"phase": phase, "step": step, **metrics}, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


if __name__ == "__main__":
    main()
