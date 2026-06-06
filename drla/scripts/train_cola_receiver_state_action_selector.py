"""Train a structured receiver-state action selector for P2-E.

The model consumes only structured decoder-free receiver state:

* calibrated latent-state verifier outputs
* sender-choice fuser logits/probabilities/confidence
* train-split task/global priors

It does not consume raw decoded predictions, token ids, gold answers, or scorer
outputs as online inputs. Decoded/scored answers are used only as offline
supervision labels and evaluation references.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from drla.scripts.audit_cola_hierarchical_aggregation_potential import build_groups
from drla.scripts.audit_cola_sequential_latent_mas import load_official_scorer
from drla.scripts.eval_cola_hierarchical_state_policy import (
    attach_priors,
    build_fuser_model,
    build_rows,
    build_state_model,
    collect_outputs,
    fit_priors,
    load_calibrators,
    mean,
    validate_compatible_configs,
)
from drla.scripts.train_cola_hierarchical_latent_fuser import (
    HierarchicalLatentFuserConfig,
    build_tensors,
    make_datasets,
    read_jsonl,
    split_groups,
)
from drla.scripts.train_cola_hierarchical_state_verifier import HierarchicalStateVerifierConfig
from drla.scripts.train_cola_readiness_model import device_metadata, require_cuda_training, resolve_device
from drla.tracking import finish_experiment, init_experiment, log_metrics


@dataclass(frozen=True)
class ReceiverStateActionSelectorConfig:
    state_checkpoint: str = (
        "/data1/luyifei/drla/outputs/cola_hierarchical_state_verifier/"
        "p2e_state_verifier_full_seed20260529_20260529/checkpoints/best_checkpoint.pt"
    )
    fuser_checkpoint: str = (
        "/data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/"
        "p2e_hierarchical_fuser_score_full_seed20260529_20260529/checkpoints/best_checkpoint.pt"
    )
    calibration_report: str = (
        "/data1/luyifei/drla/outputs/cola_hierarchical_state_verifier/"
        "p2e_state_verifier_full_seed20260529_20260529_calibration_ablation/"
        "calibration_report.json"
    )
    output_dir: str = (
        "/data1/luyifei/drla/outputs/cola_receiver_state_action_selector/"
        "p2e_receiver_state_action_selector_v1"
    )
    feature_mode: str = "state_fuser_prior"
    seed: int = 20260529
    batch_size: int = 256
    epochs: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    dropout: float = 0.1
    sender_output_mode: str = "direct"
    sender_loss_weight: float = 1.0
    pointwise_loss_weight: float = 0.5
    any_loss_weight: float = 0.5
    score_loss_weight: float = 0.5
    valid_interval: int = 10
    num_workers: int = 0
    device: str = "auto"
    swanlab_mode: str = "cloud"
    experiment_name: str = "p2e-receiver-state-action-selector-v1"


class ReceiverStateActionSelector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        sender_count: int,
        *,
        residual_fuser_logits: bool,
    ) -> None:
        super().__init__()
        self.residual_fuser_logits = residual_fuser_logits
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.sender_head = nn.Linear(hidden_dim, sender_count)
        self.any_head = nn.Linear(hidden_dim, 1)
        self.best_score_head = nn.Linear(hidden_dim, 1)
        if residual_fuser_logits:
            nn.init.zeros_(self.sender_head.weight)
            nn.init.zeros_(self.sender_head.bias)

    def forward(self, features: torch.Tensor, residual_logits: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        hidden = self.net(features)
        sender_logits = self.sender_head(hidden)
        if self.residual_fuser_logits:
            if residual_logits is None:
                raise ValueError("residual_fuser_logits mode requires residual logits")
            sender_logits = sender_logits + residual_logits
        return {
            "sender_logits": sender_logits,
            "any_logit": self.any_head(hidden).squeeze(-1),
            "best_score_logit": self.best_score_head(hidden).squeeze(-1),
        }


def main() -> None:
    summary = train_action_selector(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> ReceiverStateActionSelectorConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-checkpoint", default=ReceiverStateActionSelectorConfig.state_checkpoint)
    parser.add_argument("--fuser-checkpoint", default=ReceiverStateActionSelectorConfig.fuser_checkpoint)
    parser.add_argument("--calibration-report", default=ReceiverStateActionSelectorConfig.calibration_report)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--feature-mode",
        choices=["state_only", "fuser_only", "prior_only", "state_fuser", "state_prior", "state_fuser_prior"],
        default=ReceiverStateActionSelectorConfig.feature_mode,
    )
    parser.add_argument("--seed", type=int, default=ReceiverStateActionSelectorConfig.seed)
    parser.add_argument("--batch-size", type=int, default=ReceiverStateActionSelectorConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=ReceiverStateActionSelectorConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=ReceiverStateActionSelectorConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=ReceiverStateActionSelectorConfig.weight_decay)
    parser.add_argument("--hidden-dim", type=int, default=ReceiverStateActionSelectorConfig.hidden_dim)
    parser.add_argument("--dropout", type=float, default=ReceiverStateActionSelectorConfig.dropout)
    parser.add_argument(
        "--sender-output-mode",
        choices=["direct", "residual_fuser"],
        default=ReceiverStateActionSelectorConfig.sender_output_mode,
    )
    parser.add_argument("--sender-loss-weight", type=float, default=ReceiverStateActionSelectorConfig.sender_loss_weight)
    parser.add_argument("--pointwise-loss-weight", type=float, default=ReceiverStateActionSelectorConfig.pointwise_loss_weight)
    parser.add_argument("--any-loss-weight", type=float, default=ReceiverStateActionSelectorConfig.any_loss_weight)
    parser.add_argument("--score-loss-weight", type=float, default=ReceiverStateActionSelectorConfig.score_loss_weight)
    parser.add_argument("--valid-interval", type=int, default=ReceiverStateActionSelectorConfig.valid_interval)
    parser.add_argument("--num-workers", type=int, default=ReceiverStateActionSelectorConfig.num_workers)
    parser.add_argument("--device", default=ReceiverStateActionSelectorConfig.device)
    parser.add_argument("--swanlab-mode", default=ReceiverStateActionSelectorConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=ReceiverStateActionSelectorConfig.experiment_name)
    args = parser.parse_args()
    return ReceiverStateActionSelectorConfig(
        state_checkpoint=args.state_checkpoint,
        fuser_checkpoint=args.fuser_checkpoint,
        calibration_report=args.calibration_report,
        output_dir=args.output_dir,
        feature_mode=args.feature_mode,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        sender_output_mode=args.sender_output_mode,
        sender_loss_weight=args.sender_loss_weight,
        pointwise_loss_weight=args.pointwise_loss_weight,
        any_loss_weight=args.any_loss_weight,
        score_loss_weight=args.score_loss_weight,
        valid_interval=args.valid_interval,
        num_workers=args.num_workers,
        device=args.device,
        swanlab_mode=args.swanlab_mode,
        experiment_name=args.experiment_name,
    )


def train_action_selector(config: ReceiverStateActionSelectorConfig) -> dict[str, Any]:
    validate_config(config)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    data = build_state_action_data(config)
    train_ds, valid_ds, test_ds, norm_stats = make_feature_datasets(data)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    device = resolve_device(config.device)
    require_cuda_training(device, "train_cola_receiver_state_action_selector.py")
    model = ReceiverStateActionSelector(
        input_dim=data["metadata"]["input_dim"],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        sender_count=data["metadata"]["sender_count"],
        residual_fuser_logits=config.sender_output_mode == "residual_fuser",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    any_pos = (data["train"]["labels"].sum(dim=1) > 0).float().sum().clamp_min(1.0)
    any_neg = data["train"]["labels"].shape[0] - any_pos
    any_pos_weight = (any_neg / any_pos).to(device)

    run = init_experiment(
        stage="p2e-receiver-state-action-selector",
        config={**asdict(config), **device_metadata(device), "any_pos_weight": float(any_pos_weight.detach().cpu())},
        experiment_name=config.experiment_name,
        tags=["cola", "official-benchmark", "p2e", "receiver-state", "action-selector"],
        mode=config.swanlab_mode,
    )

    best_metric = float("-inf")
    best_step = 0
    global_step = 0
    metrics_f = metrics_path.open("w", encoding="utf-8")
    try:
        for _epoch in range(config.epochs):
            model.train()
            for batch in train_loader:
                global_step += 1
                batch = [item.to(device) for item in batch]
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch[0], batch[3])
                loss, train_metrics = compute_loss(
                    outputs=outputs,
                    labels=batch[1],
                    target_scores=batch[2],
                    any_pos_weight=any_pos_weight,
                    config=config,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_metrics = {"loss": float(loss.detach().item()), **train_metrics}
                write_metrics(metrics_f, "train", global_step, train_metrics)
                log_metrics(train_metrics, step=global_step, prefix="train")
                if global_step % config.valid_interval == 0:
                    valid_metrics = evaluate(model, valid_loader, data["valid"], device)
                    write_metrics(metrics_f, "valid", global_step, valid_metrics)
                    log_metrics(valid_metrics, step=global_step, prefix="valid")
                    current = selection_metric(valid_metrics)
                    if current > best_metric:
                        best_metric = current
                        best_step = global_step
                        save_checkpoint(checkpoint_dir / "best_checkpoint.pt", model, optimizer, config, data["metadata"], norm_stats, best_step, best_metric)

        valid_metrics = evaluate(model, valid_loader, data["valid"], device)
        test_metrics = evaluate(model, test_loader, data["test"], device)
        write_metrics(metrics_f, "valid", global_step, valid_metrics)
        write_metrics(metrics_f, "test", global_step, test_metrics)
        log_metrics(valid_metrics, step=global_step, prefix="valid")
        log_metrics(test_metrics, step=global_step, prefix="test")
        current = selection_metric(valid_metrics)
        if current > best_metric:
            best_metric = current
            best_step = global_step
            save_checkpoint(checkpoint_dir / "best_checkpoint.pt", model, optimizer, config, data["metadata"], norm_stats, best_step, best_metric)
        save_checkpoint(checkpoint_dir / "last_checkpoint.pt", model, optimizer, config, data["metadata"], norm_stats, global_step, current)
    finally:
        metrics_f.close()
        finish_experiment()

    best_checkpoint = torch.load(checkpoint_dir / "best_checkpoint.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    best_valid_metrics = evaluate(model, valid_loader, data["valid"], device)
    best_test_metrics = evaluate(model, test_loader, data["test"], device)
    write_predictions(output_dir / "test_predictions.jsonl", model, test_loader, data["test"], device)

    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "swanlab_run_id": getattr(run, "id", None),
        "split_sizes": {split: int(data[split]["features"].shape[0]) for split in ["train", "valid", "test"]},
        "metadata": data["metadata"],
        "best_step": best_step,
        "best_valid_selection_metric": best_metric,
        "final_valid_metrics": valid_metrics,
        "final_test_metrics": test_metrics,
        "best_valid_metrics": best_valid_metrics,
        "best_test_metrics": best_test_metrics,
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "metrics_jsonl": str(metrics_path),
            "best_checkpoint": str(checkpoint_dir / "best_checkpoint.pt"),
            "last_checkpoint": str(checkpoint_dir / "last_checkpoint.pt"),
            "test_predictions_jsonl": str(output_dir / "test_predictions.jsonl"),
        },
        "interpretation": (
            "Structured receiver-state action selector. Online inputs are "
            "calibrated state/fuser/prior features; decoded/scored answers are "
            "offline labels only."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_state_action_data(config: ReceiverStateActionSelectorConfig) -> dict[str, Any]:
    state_checkpoint = torch.load(config.state_checkpoint, map_location="cpu")
    fuser_checkpoint = torch.load(config.fuser_checkpoint, map_location="cpu")
    state_config = HierarchicalStateVerifierConfig(**state_checkpoint["config"])
    fuser_config = HierarchicalLatentFuserConfig(**fuser_checkpoint["config"])
    validate_compatible_configs(state_config, fuser_config)
    scorer = load_official_scorer(Path(state_config.acc_calc_script))
    packets = read_jsonl(Path(state_config.packets_jsonl))
    groups = build_groups(packets, state_config, scorer)
    if state_config.max_groups:
        groups = groups[: state_config.max_groups]
    splits = split_groups(groups, state_config)
    tensors_by_split, metadata = build_tensors(groups, splits, state_config)
    train_ds, valid_ds, test_ds, _ = make_datasets(tensors_by_split)
    raw_datasets = {"train": train_ds, "valid": valid_ds, "test": test_ds}

    device = resolve_device(config.device)
    state_model = build_state_model(state_checkpoint, state_config, metadata, device)
    fuser_model = build_fuser_model(fuser_checkpoint, fuser_config, metadata, device)
    calibrators = load_calibrators(Path(config.calibration_report))

    rows_by_split = {}
    outputs_by_split = {}
    for split in ["train", "valid", "test"]:
        loader = DataLoader(raw_datasets[split], batch_size=config.batch_size, shuffle=False, num_workers=0)
        outputs = collect_outputs(state_model, fuser_model, loader, device, calibrators)
        outputs_by_split[split] = outputs
        rows_by_split[split] = build_rows(
            split=split,
            groups=groups,
            indices=splits[split],
            tensors=tensors_by_split[split],
            outputs=outputs,
            scorer=scorer,
        )
    priors = fit_priors(rows_by_split["train"])
    for split in ["train", "valid", "test"]:
        attach_priors(rows_by_split[split], priors)

    feature_names = feature_names_for_mode(config.feature_mode, sender_count=metadata["sender_count"])
    data: dict[str, Any] = {}
    for split in ["train", "valid", "test"]:
        data[split] = {
            "features": build_feature_tensor(rows_by_split[split], outputs_by_split[split], config.feature_mode),
            "residual_logits": outputs_by_split[split]["fuser_logits"].float(),
            "labels": tensors_by_split[split]["labels"].float(),
            "target_scores": tensors_by_split[split]["target_scores"].float(),
            "baseline_rows": rows_by_split[split],
        }
    data["metadata"] = {
        **metadata,
        "input_dim": len(feature_names),
        "feature_names": feature_names,
        "feature_mode": config.feature_mode,
        "state_checkpoint": config.state_checkpoint,
        "fuser_checkpoint": config.fuser_checkpoint,
        "calibration_report": config.calibration_report,
    }
    return data


def feature_names_for_mode(feature_mode: str, *, sender_count: int) -> list[str]:
    state = [
        "state_any_prob",
        "state_best_score_pred",
        "state_raw_any_prob",
        "state_raw_best_score_pred",
    ]
    fuser = [f"fuser_logit_{idx}" for idx in range(sender_count)]
    fuser += [f"fuser_prob_{idx}" for idx in range(sender_count)]
    fuser += ["fuser_confidence", "fuser_margin", "fuser_entropy"]
    prior = ["train_task_prior_any", "train_task_prior_score", "train_global_prior_any", "train_global_prior_score"]
    names: list[str] = []
    if "state" in feature_mode:
        names += state
    if "fuser" in feature_mode:
        names += fuser
    if "prior" in feature_mode:
        names += prior
    return names


def build_feature_tensor(rows: list[dict[str, Any]], outputs: dict[str, torch.Tensor], feature_mode: str) -> torch.Tensor:
    logits = outputs["fuser_logits"].float()
    probs = torch.softmax(logits, dim=1)
    sorted_probs = probs.sort(dim=1, descending=True).values
    confidence = sorted_probs[:, 0]
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1)
    columns: list[torch.Tensor] = []
    if "state" in feature_mode:
        columns.extend(
            [
                torch.tensor([row["state_any_prob"] for row in rows], dtype=torch.float32),
                torch.tensor([row["state_best_score_pred"] for row in rows], dtype=torch.float32),
                torch.tensor([row["state_raw_any_prob"] for row in rows], dtype=torch.float32),
                torch.tensor([row["state_raw_best_score_pred"] for row in rows], dtype=torch.float32),
            ]
        )
    if "fuser" in feature_mode:
        columns.extend([logits[:, idx] for idx in range(logits.shape[1])])
        columns.extend([probs[:, idx] for idx in range(probs.shape[1])])
        columns.extend([confidence, margin, entropy])
    if "prior" in feature_mode:
        columns.extend(
            [
                torch.tensor([row["train_task_prior_any"] for row in rows], dtype=torch.float32),
                torch.tensor([row["train_task_prior_score"] for row in rows], dtype=torch.float32),
                torch.tensor([row["train_global_prior_any"] for row in rows], dtype=torch.float32),
                torch.tensor([row["train_global_prior_score"] for row in rows], dtype=torch.float32),
            ]
        )
    if not columns:
        raise ValueError(f"feature mode {feature_mode} produced no features")
    return torch.stack(columns, dim=1).float()


def make_feature_datasets(data: dict[str, Any]) -> tuple[TensorDataset, TensorDataset, TensorDataset, dict[str, torch.Tensor]]:
    train_features = data["train"]["features"]
    mean_value = train_features.mean(dim=0, keepdim=True)
    std_value = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    norm_stats = {"feature_mean": mean_value, "feature_std": std_value}

    def dataset(split: str) -> TensorDataset:
        features = (data[split]["features"] - mean_value) / std_value
        return TensorDataset(features, data[split]["labels"], data[split]["target_scores"], data[split]["residual_logits"])

    return dataset("train"), dataset("valid"), dataset("test"), norm_stats


def compute_loss(
    *,
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    target_scores: torch.Tensor,
    any_pos_weight: torch.Tensor,
    config: ReceiverStateActionSelectorConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    sender_logits = outputs["sender_logits"]
    sender_prob = torch.softmax(sender_logits, dim=1)
    score_sum = target_scores.sum(dim=1)
    has_signal = score_sum > 1e-8
    if has_signal.any():
        target_dist = target_scores[has_signal] / score_sum[has_signal].unsqueeze(1).clamp_min(1e-8)
        rank_loss = -(target_dist * F.log_softmax(sender_logits[has_signal], dim=1)).sum(dim=1).mean()
    else:
        rank_loss = torch.zeros((), device=sender_logits.device)
    pointwise_loss = (torch.sigmoid(sender_logits) - target_scores).square().mean()
    any_target = (labels.sum(dim=1) > 0).float()
    any_loss = F.binary_cross_entropy_with_logits(outputs["any_logit"], any_target, pos_weight=any_pos_weight)
    best_score = target_scores.max(dim=1).values
    score_loss = (torch.sigmoid(outputs["best_score_logit"]) - best_score).square().mean()
    loss = (
        config.sender_loss_weight * rank_loss
        + config.pointwise_loss_weight * pointwise_loss
        + config.any_loss_weight * any_loss
        + config.score_loss_weight * score_loss
    )
    with torch.no_grad():
        selected = sender_prob.argmax(dim=1)
        row_ids = torch.arange(labels.shape[0], device=labels.device)
        selected_acc = labels[row_ids, selected].float().mean()
        selected_score = target_scores[row_ids, selected].float().mean()
    return loss, {
        "rank_loss": float(rank_loss.detach().item()),
        "pointwise_loss": float(pointwise_loss.detach().item()),
        "any_loss": float(any_loss.detach().item()),
        "score_loss": float(score_loss.detach().item()),
        "batch_selected_accuracy": float(selected_acc.detach().item()),
        "batch_selected_score": float(selected_score.detach().item()),
    }


@torch.no_grad()
def evaluate(
    model: ReceiverStateActionSelector,
    loader: DataLoader,
    split_data: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    sender_logits = []
    any_logits = []
    score_logits = []
    for batch in loader:
        batch = [item.to(device) for item in batch]
        outputs = model(batch[0], batch[3])
        sender_logits.append(outputs["sender_logits"].cpu())
        any_logits.append(outputs["any_logit"].cpu())
        score_logits.append(outputs["best_score_logit"].cpu())
    sender_logits_t = torch.cat(sender_logits)
    any_prob = torch.sigmoid(torch.cat(any_logits))
    best_score_pred = torch.sigmoid(torch.cat(score_logits))
    return evaluate_outputs(sender_logits_t, any_prob, best_score_pred, split_data)


def evaluate_outputs(
    sender_logits: torch.Tensor,
    any_prob: torch.Tensor,
    best_score_pred: torch.Tensor,
    split_data: dict[str, Any],
) -> dict[str, float]:
    labels = split_data["labels"]
    target_scores = split_data["target_scores"]
    rows = split_data["baseline_rows"]
    selected = sender_logits.argmax(dim=1)
    row_ids = torch.arange(labels.shape[0])
    selected_correct = labels[row_ids, selected].float()
    selected_score = target_scores[row_ids, selected].float()
    any_target = (labels.sum(dim=1) > 0).float()
    best_score = target_scores.max(dim=1).values.float()
    metrics = {
        "model_selected_accuracy": float(selected_correct.mean().item()),
        "model_mean_score": float(selected_score.mean().item()),
        "model_selects_correct_when_any_correct": masked_mean(selected_correct, any_target.bool()),
        "any_brier": float((any_prob - any_target).square().mean().item()),
        "any_prob_mean": float(any_prob.mean().item()),
        "any_target_mean": float(any_target.mean().item()),
        "best_score_rmse": float((best_score_pred - best_score).square().mean().sqrt().item()),
        "best_score_pred_mean": float(best_score_pred.mean().item()),
        "best_score_target_mean": float(best_score.mean().item()),
        "first_accuracy": mean(row["first_correct"] for row in rows),
        "first_score": mean(row["first_score"] for row in rows),
        "fuser_accuracy": mean(row["fuser_correct"] for row in rows),
        "fuser_score": mean(row["fuser_score"] for row in rows),
        "text_majority_accuracy": mean(row["text_majority_correct"] for row in rows),
        "text_majority_score": mean(row["text_majority_score"] for row in rows),
        "oracle_any_accuracy": mean(row["oracle_any"] for row in rows),
        "oracle_best_score": mean(row["oracle_best_score"] for row in rows),
        "num_groups": float(len(rows)),
    }
    metrics.update(gated_metrics(sender_logits, any_prob, split_data, target_precision=0.60))
    return metrics


def gated_metrics(
    sender_logits: torch.Tensor,
    any_prob: torch.Tensor,
    split_data: dict[str, Any],
    *,
    target_precision: float,
) -> dict[str, float]:
    labels = split_data["labels"]
    target_scores = split_data["target_scores"]
    rows = split_data["baseline_rows"]
    selected = sender_logits.argmax(dim=1)
    row_ids = torch.arange(labels.shape[0])
    selected_correct = labels[row_ids, selected].float()
    selected_score = target_scores[row_ids, selected].float()
    any_target = (labels.sum(dim=1) > 0).float()
    threshold = choose_threshold_for_precision(any_prob, any_target, target_precision)
    mask = any_prob >= threshold
    if int(mask.sum().item()) == 0:
        accepted_accuracy = 0.0
        accepted_score = 0.0
        any_precision = 0.0
    else:
        accepted_accuracy = float(selected_correct[mask].mean().item())
        accepted_score = float(selected_score[mask].mean().item())
        any_precision = float(any_target[mask].mean().item())
    fallback_first_acc = []
    fallback_first_score = []
    for idx, row in enumerate(rows):
        if bool(mask[idx].item()):
            fallback_first_acc.append(float(selected_correct[idx].item()))
            fallback_first_score.append(float(selected_score[idx].item()))
        else:
            fallback_first_acc.append(float(row["first_correct"]))
            fallback_first_score.append(float(row["first_score"]))
    return {
        "valid_like_gate_target_precision": float(target_precision),
        "self_gate_threshold": float(threshold),
        "self_gate_coverage": float(mask.float().mean().item()),
        "self_gate_any_precision": any_precision,
        "self_gate_accepted_accuracy": accepted_accuracy,
        "self_gate_accepted_score": accepted_score,
        "self_gate_fallback_first_accuracy": mean(fallback_first_acc),
        "self_gate_fallback_first_score": mean(fallback_first_score),
    }


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


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum().item()) == 0:
        return 0.0
    return float(values[mask].mean().item())


def selection_metric(metrics: dict[str, float]) -> float:
    return float(metrics["model_mean_score"])


def save_checkpoint(
    path: Path,
    model: ReceiverStateActionSelector,
    optimizer: torch.optim.Optimizer,
    config: ReceiverStateActionSelectorConfig,
    metadata: dict[str, Any],
    norm_stats: dict[str, torch.Tensor],
    step: int,
    metric: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "metadata": metadata,
            "norm_stats": norm_stats,
            "step": step,
            "metric": metric,
        },
        path,
    )


def write_predictions(
    path: Path,
    model: ReceiverStateActionSelector,
    loader: DataLoader,
    split_data: dict[str, Any],
    device: torch.device,
) -> None:
    model.eval()
    rows = split_data["baseline_rows"]
    labels = split_data["labels"]
    target_scores = split_data["target_scores"]
    offset = 0
    with path.open("w", encoding="utf-8") as handle, torch.no_grad():
        for batch in loader:
            batch_size = batch[0].shape[0]
            batch = [item.to(device) for item in batch]
            outputs = model(batch[0], batch[3])
            sender_prob = torch.softmax(outputs["sender_logits"], dim=1).cpu()
            selected = sender_prob.argmax(dim=1)
            any_prob = torch.sigmoid(outputs["any_logit"]).cpu()
            score_pred = torch.sigmoid(outputs["best_score_logit"]).cpu()
            for idx in range(batch_size):
                row_idx = offset + idx
                selected_idx = int(selected[idx].item())
                record = {
                    **rows[row_idx],
                    "selector_selected_index": selected_idx,
                    "selector_selected_correct": float(labels[row_idx, selected_idx].item()),
                    "selector_selected_score": float(target_scores[row_idx, selected_idx].item()),
                    "selector_any_prob": float(any_prob[idx].item()),
                    "selector_best_score_pred": float(score_pred[idx].item()),
                    "selector_sender_probs": [float(value) for value in sender_prob[idx].tolist()],
                }
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            offset += batch_size


def write_metrics(handle: Any, split: str, step: int, metrics: dict[str, float]) -> None:
    handle.write(json.dumps({"created_at": int(time.time()), "split": split, "step": step, "metrics": metrics}, sort_keys=True) + "\n")
    handle.flush()


def validate_config(config: ReceiverStateActionSelectorConfig) -> None:
    if config.valid_interval > 10:
        raise ValueError("valid_interval must be <= 10")
    if config.swanlab_mode != "cloud":
        raise ValueError("training must use SwanLab cloud; pass --swanlab-mode cloud")
    if config.feature_mode not in {"state_only", "fuser_only", "prior_only", "state_fuser", "state_prior", "state_fuser_prior"}:
        raise ValueError("unsupported feature_mode")
    if config.sender_output_mode not in {"direct", "residual_fuser"}:
        raise ValueError("unsupported sender_output_mode")
    if config.sender_output_mode == "residual_fuser" and "fuser" not in config.feature_mode:
        raise ValueError("residual_fuser sender output requires a feature_mode containing fuser")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")


if __name__ == "__main__":
    main()
