"""Train a P2-E multi-sender latent-state utility verifier.

Unlike the sender-choice fuser, this model predicts group-level receiver state:

* whether any sender packet in the group contains an exact-correct selected answer
* the best selected official score available among the sender packets

The verifier reads only sanitized latent/process/certificate fields online.
Decoded answers and official scores are offline supervision labels only.
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
from torch.utils.data import DataLoader

from drla.scripts.audit_cola_hierarchical_aggregation_potential import build_groups
from drla.scripts.audit_cola_sequential_latent_mas import load_official_scorer
from drla.scripts.train_cola_hierarchical_latent_fuser import (
    HierarchicalLatentFuser,
    build_tensors,
    make_datasets,
    read_jsonl,
    split_groups,
)
from drla.scripts.train_cola_readiness_model import device_metadata, require_cuda_training, resolve_device
from drla.tracking import finish_experiment, init_experiment, log_metrics


@dataclass(frozen=True)
class HierarchicalStateVerifierConfig:
    packets_jsonl: str = (
        "/data1/luyifei/drla/outputs/cola_agent_latent_comm/"
        "p2_agent_latent_comm_v2_locked_seed66_67_68_split20260601_20260529/"
        "agent_latent_comm_packets_test.jsonl"
    )
    output_dir: str = (
        "/data1/luyifei/drla/outputs/cola_hierarchical_state_verifier/"
        "p2e_hierarchical_state_verifier_v1"
    )
    data_root: str = "/data1/luyifei/Cola-DLM/code/generate_task_data"
    acc_calc_script: str = "/data1/luyifei/Cola-DLM/code/scripts/acc_calc.py"
    seed: int = 20260529
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    max_groups: int = 0
    batch_size: int = 256
    epochs: int = 24
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    d_model: int = 128
    attention_heads: int = 4
    inter_layers: int = 2
    sender_layers: int = 2
    any_loss_weight: float = 1.0
    score_loss_weight: float = 1.0
    valid_interval: int = 10
    max_cached_shards: int = 1024
    num_workers: int = 0
    device: str = "auto"
    swanlab_mode: str = "cloud"
    experiment_name: str = "p2e-hierarchical-state-verifier-v1"


class HierarchicalStateVerifier(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        process_dim: int,
        certificate_dim: int,
        max_blocks: int,
        block_size: int,
        sender_count: int,
        task_count: int,
        d_model: int,
        attention_heads: int,
        inter_layers: int,
        sender_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = HierarchicalLatentFuser(
            latent_dim=latent_dim,
            process_dim=process_dim,
            certificate_dim=certificate_dim,
            max_blocks=max_blocks,
            block_size=block_size,
            sender_count=sender_count,
            task_count=task_count,
            d_model=d_model,
            attention_heads=attention_heads,
            inter_layers=inter_layers,
            sender_layers=sender_layers,
            dropout=dropout,
        )
        self.pool = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.any_correct_head = nn.Linear(d_model, 1)
        self.best_score_head = nn.Linear(d_model, 1)

    def forward(
        self,
        latent_blocks: torch.Tensor,
        process_features: torch.Tensor,
        block_mask: torch.Tensor,
        certificate_features: torch.Tensor,
        task_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        sender_state = self.backbone.encode_senders(
            latent_blocks=latent_blocks,
            process_features=process_features,
            block_mask=block_mask,
            certificate_features=certificate_features,
            task_idx=task_idx,
        )
        pooled = self.pool(torch.cat([sender_state.mean(dim=1), sender_state.max(dim=1).values], dim=-1))
        return {
            "any_correct_logit": self.any_correct_head(pooled).squeeze(-1),
            "best_score_logit": self.best_score_head(pooled).squeeze(-1),
        }


def main() -> None:
    summary = train_hierarchical_state_verifier(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> HierarchicalStateVerifierConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets-jsonl", default=HierarchicalStateVerifierConfig.packets_jsonl)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default=HierarchicalStateVerifierConfig.data_root)
    parser.add_argument("--acc-calc-script", default=HierarchicalStateVerifierConfig.acc_calc_script)
    parser.add_argument("--seed", type=int, default=HierarchicalStateVerifierConfig.seed)
    parser.add_argument("--train-ratio", type=float, default=HierarchicalStateVerifierConfig.train_ratio)
    parser.add_argument("--valid-ratio", type=float, default=HierarchicalStateVerifierConfig.valid_ratio)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=HierarchicalStateVerifierConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=HierarchicalStateVerifierConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=HierarchicalStateVerifierConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=HierarchicalStateVerifierConfig.weight_decay)
    parser.add_argument("--dropout", type=float, default=HierarchicalStateVerifierConfig.dropout)
    parser.add_argument("--d-model", type=int, default=HierarchicalStateVerifierConfig.d_model)
    parser.add_argument("--attention-heads", type=int, default=HierarchicalStateVerifierConfig.attention_heads)
    parser.add_argument("--inter-layers", type=int, default=HierarchicalStateVerifierConfig.inter_layers)
    parser.add_argument("--sender-layers", type=int, default=HierarchicalStateVerifierConfig.sender_layers)
    parser.add_argument("--any-loss-weight", type=float, default=HierarchicalStateVerifierConfig.any_loss_weight)
    parser.add_argument("--score-loss-weight", type=float, default=HierarchicalStateVerifierConfig.score_loss_weight)
    parser.add_argument("--valid-interval", type=int, default=HierarchicalStateVerifierConfig.valid_interval)
    parser.add_argument("--max-cached-shards", type=int, default=HierarchicalStateVerifierConfig.max_cached_shards)
    parser.add_argument("--num-workers", type=int, default=HierarchicalStateVerifierConfig.num_workers)
    parser.add_argument("--device", default=HierarchicalStateVerifierConfig.device)
    parser.add_argument("--swanlab-mode", default=HierarchicalStateVerifierConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=HierarchicalStateVerifierConfig.experiment_name)
    args = parser.parse_args()
    return HierarchicalStateVerifierConfig(
        packets_jsonl=args.packets_jsonl,
        output_dir=args.output_dir,
        data_root=args.data_root,
        acc_calc_script=args.acc_calc_script,
        seed=args.seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        max_groups=args.max_groups,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        d_model=args.d_model,
        attention_heads=args.attention_heads,
        inter_layers=args.inter_layers,
        sender_layers=args.sender_layers,
        any_loss_weight=args.any_loss_weight,
        score_loss_weight=args.score_loss_weight,
        valid_interval=args.valid_interval,
        max_cached_shards=args.max_cached_shards,
        num_workers=args.num_workers,
        device=args.device,
        swanlab_mode=args.swanlab_mode,
        experiment_name=args.experiment_name,
    )


def train_hierarchical_state_verifier(config: HierarchicalStateVerifierConfig) -> dict[str, Any]:
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
    scorer = load_official_scorer(Path(config.acc_calc_script))
    packets = read_jsonl(Path(config.packets_jsonl))
    groups = build_groups(packets, config, scorer)
    if config.max_groups:
        groups = groups[: config.max_groups]
    splits = split_groups(groups, config)
    tensors_by_split, metadata = build_tensors(groups, splits, config)
    train_ds, valid_ds, test_ds, norm_stats = make_datasets(tensors_by_split)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    device = resolve_device(config.device)
    require_cuda_training(device, "train_cola_hierarchical_state_verifier.py")
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_labels = tensors_by_split["train"]["labels"]
    any_pos = (train_labels.sum(dim=1) > 0).float().sum().clamp_min(1.0)
    any_neg = train_labels.shape[0] - any_pos
    any_pos_weight = (any_neg / any_pos).to(device)
    run = init_experiment(
        stage="p2e-hierarchical-state-verifier",
        config={**asdict(config), **device_metadata(device), "any_pos_weight": float(any_pos_weight.detach().cpu())},
        experiment_name=config.experiment_name,
        tags=["cola", "official-benchmark", "p2e", "hierarchical", "latent-state-verifier"],
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
                outputs = model(*batch[:5])
                loss, train_metrics = compute_loss(
                    outputs=outputs,
                    labels=batch[5],
                    target_scores=batch[6],
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
                    valid_metrics = evaluate(model, valid_loader, tensors_by_split["valid"], device)
                    write_metrics(metrics_f, "valid", global_step, valid_metrics)
                    log_metrics(valid_metrics, step=global_step, prefix="valid")
                    current = valid_metrics["selection_metric"]
                    if current > best_metric:
                        best_metric = current
                        best_step = global_step
                        save_checkpoint(checkpoint_dir / "best_checkpoint.pt", model, optimizer, config, metadata, norm_stats, best_step, best_metric)

        valid_metrics = evaluate(model, valid_loader, tensors_by_split["valid"], device)
        test_metrics = evaluate(model, test_loader, tensors_by_split["test"], device)
        write_metrics(metrics_f, "valid", global_step, valid_metrics)
        write_metrics(metrics_f, "test", global_step, test_metrics)
        log_metrics(valid_metrics, step=global_step, prefix="valid")
        log_metrics(test_metrics, step=global_step, prefix="test")
        if valid_metrics["selection_metric"] > best_metric:
            best_metric = valid_metrics["selection_metric"]
            best_step = global_step
            save_checkpoint(checkpoint_dir / "best_checkpoint.pt", model, optimizer, config, metadata, norm_stats, best_step, best_metric)
        save_checkpoint(checkpoint_dir / "last_checkpoint.pt", model, optimizer, config, metadata, norm_stats, global_step, valid_metrics["selection_metric"])
    finally:
        metrics_f.close()
        finish_experiment()

    best_valid_metrics, best_test_metrics = evaluate_best_checkpoint(
        checkpoint_dir / "best_checkpoint.pt",
        config,
        metadata,
        valid_loader,
        test_loader,
        tensors_by_split,
        device,
    )
    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "swanlab_run_id": getattr(run, "id", None),
        "num_groups": len(groups),
        "split_sizes": {name: len(indices) for name, indices in splits.items()},
        "metadata": metadata,
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
        },
        "interpretation": (
            "P2-E latent-state utility verifier. The model predicts group-level "
            "answer-readiness utility from decoder-free multi-sender latent packets."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def validate_config(config: HierarchicalStateVerifierConfig) -> None:
    if config.valid_interval > 10:
        raise ValueError("valid_interval must be <= 10 steps")
    if config.swanlab_mode != "cloud":
        raise ValueError("training must use SwanLab cloud; pass --swanlab-mode cloud")
    if not 0.0 < config.train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0.0 <= config.valid_ratio < 1.0:
        raise ValueError("valid_ratio must be in [0, 1)")
    if config.train_ratio + config.valid_ratio >= 1.0:
        raise ValueError("train_ratio + valid_ratio must be < 1")


def compute_loss(
    *,
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    target_scores: torch.Tensor,
    any_pos_weight: torch.Tensor,
    config: HierarchicalStateVerifierConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    any_target = (labels.sum(dim=1) > 0).float()
    best_score = target_scores.max(dim=1).values
    any_loss = F.binary_cross_entropy_with_logits(outputs["any_correct_logit"], any_target, pos_weight=any_pos_weight)
    score_pred = torch.sigmoid(outputs["best_score_logit"])
    score_loss = F.mse_loss(score_pred, best_score)
    loss = config.any_loss_weight * any_loss + config.score_loss_weight * score_loss
    with torch.no_grad():
        any_prob = torch.sigmoid(outputs["any_correct_logit"])
        metrics = regression_and_binary_metrics(score_pred.detach().cpu(), best_score.detach().cpu(), any_prob.detach().cpu(), any_target.detach().cpu())
    return loss, {"any_loss": float(any_loss.detach().item()), "score_loss": float(score_loss.detach().item()), **metrics}


@torch.no_grad()
def evaluate(
    model: HierarchicalStateVerifier,
    loader: DataLoader,
    tensors: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    score_preds = []
    any_probs = []
    for batch in loader:
        batch = [item.to(device) for item in batch]
        outputs = model(*batch[:5])
        score_preds.append(torch.sigmoid(outputs["best_score_logit"]).cpu())
        any_probs.append(torch.sigmoid(outputs["any_correct_logit"]).cpu())
    score_pred = torch.cat(score_preds)
    any_prob = torch.cat(any_probs)
    labels = tensors["labels"]
    target_scores = tensors["target_scores"]
    any_target = (labels.sum(dim=1) > 0).float()
    best_score = target_scores.max(dim=1).values
    metrics = regression_and_binary_metrics(score_pred, best_score, any_prob, any_target)
    metrics["num_groups"] = float(labels.shape[0])
    metrics["selection_metric"] = metrics["any_auroc"] + (1.0 - metrics["best_score_rmse"])
    return metrics


def regression_and_binary_metrics(
    score_pred: torch.Tensor,
    best_score: torch.Tensor,
    any_prob: torch.Tensor,
    any_target: torch.Tensor,
) -> dict[str, float]:
    error = score_pred - best_score
    return {
        "best_score_mae": float(error.abs().mean().item()),
        "best_score_rmse": float(torch.sqrt((error.square()).mean()).item()),
        "best_score_corr": pearson(score_pred, best_score),
        "best_score_pred_mean": float(score_pred.mean().item()),
        "best_score_target_mean": float(best_score.mean().item()),
        "any_brier": float((any_prob - any_target).square().mean().item()),
        "any_accuracy_at_0_5": float(((any_prob >= 0.5).float() == any_target).float().mean().item()),
        "any_auroc": auroc(any_prob, any_target),
        "any_prob_mean": float(any_prob.mean().item()),
        "any_target_mean": float(any_target.mean().item()),
    }


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    vx = x - x.mean()
    vy = y - y.mean()
    denom = vx.norm() * vy.norm()
    if float(denom) <= 0:
        return 0.0
    return float((vx @ vy / denom).item())


def auroc(prob: torch.Tensor, target: torch.Tensor) -> float:
    prob = prob.float()
    target = target.float()
    pos = target == 1
    neg = target == 0
    if int(pos.sum()) == 0 or int(neg.sum()) == 0:
        return 0.5
    order = torch.argsort(prob)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(prob) + 1, dtype=torch.float32)
    pos_ranks = ranks[pos].sum()
    n_pos = float(pos.sum())
    n_neg = float(neg.sum())
    return float(((pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item())


def evaluate_best_checkpoint(
    path: Path,
    config: HierarchicalStateVerifierConfig,
    metadata: dict[str, Any],
    valid_loader: DataLoader,
    test_loader: DataLoader,
    tensors_by_split: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    checkpoint = torch.load(path, map_location="cpu")
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
    return (
        evaluate(model, valid_loader, tensors_by_split["valid"], device),
        evaluate(model, test_loader, tensors_by_split["test"], device),
    )


def save_checkpoint(
    path: Path,
    model: HierarchicalStateVerifier,
    optimizer: torch.optim.Optimizer,
    config: HierarchicalStateVerifierConfig,
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


def write_metrics(handle: Any, split: str, step: int, metrics: dict[str, float]) -> None:
    handle.write(json.dumps({"created_at": int(time.time()), "split": split, "step": step, "metrics": metrics}, sort_keys=True) + "\n")
    handle.flush()


if __name__ == "__main__":
    main()
