"""Train a joint P2-E request-and-select latent policy.

The model has two online stages:

1. request head: sees only the first sender packet and decides whether to
   request more latent evidence.
2. select head: if more evidence is requested, sees first + requested sender
   packets and selects the answer packet.

Training combines supervised request/selection losses with a differentiable
budgeted utility term.  Decoded answers and official scores are labels and
evaluation references only; they are not model inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from drla.scripts.audit_cola_hierarchical_aggregation_potential import (
    build_groups,
    choose_text_majority_selected,
)
from drla.scripts.audit_cola_sequential_latent_mas import load_official_scorer, score_text_with_official_rules
from drla.scripts.train_cola_hierarchical_latent_fuser import (
    HierarchicalLatentFuser,
    HierarchicalLatentFuserConfig,
    build_tensors,
    read_jsonl,
    split_groups,
)
from drla.scripts.train_cola_post_request_selector import (
    AnchorAwarePostRequestSelector,
    compute_pairwise_loss,
    select_index,
)
from drla.scripts.train_cola_readiness_model import device_metadata, require_cuda_training, resolve_device
from drla.scripts.train_cola_request_more_policy import (
    FirstSenderRequestPolicy,
    compute_fuser_refs,
    normalize_tensors,
)
from drla.tracking import finish_experiment, init_experiment, log_metrics


@dataclass(frozen=True)
class JointRequestSelectConfig:
    packets_jsonl: str = (
        "/data1/luyifei/drla/outputs/cola_agent_latent_comm/"
        "p2_agent_latent_comm_v2_locked_seed66_67_68_split20260601_20260529/"
        "agent_latent_comm_packets_test.jsonl"
    )
    output_dir: str = (
        "/data1/luyifei/drla/outputs/cola_joint_request_select_policy/"
        "p2e_joint_request_select_policy_v1"
    )
    fuser_checkpoint: str = (
        "/data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/"
        "p2e_hierarchical_fuser_score_full_seed20260529_20260529/checkpoints/"
        "best_checkpoint.pt"
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
    request_sender_layers: int = 1
    selector_sender_layers: int = 2
    listwise_loss_weight: float = 1.0
    score_loss_weight: float = 1.0
    pairwise_loss_weight: float = 0.5
    gain_loss_weight: float = 0.5
    request_loss_weight: float = 1.0
    utility_loss_weight: float = 1.0
    request_gain_loss_weight: float = 0.5
    request_cost: float = 0.0
    listwise_temperature: float = 0.25
    selection_output: str = "rank"
    target_request_rates: str = "0.10,0.25,0.50"
    target_helpful_precisions: str = "0.50,0.60,0.70"
    checkpoint_selection_mode: str = "target025_request_prob"
    checkpoint_selection_metric: str = "model_after_request_score"
    checkpoint_selection_signals: str = "request_prob,request_gain_pred"
    checkpoint_selection_request_rates: str = ""
    valid_interval: int = 10
    max_cached_shards: int = 1024
    num_workers: int = 0
    device: str = "auto"
    swanlab_mode: str = "cloud"
    experiment_name: str = "p2e-joint-request-select-policy-v1"


class JointRequestSelectPolicy(nn.Module):
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
        request_sender_layers: int,
        selector_sender_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.request_model = FirstSenderRequestPolicy(
            latent_dim=latent_dim,
            process_dim=process_dim,
            certificate_dim=certificate_dim,
            max_blocks=max_blocks,
            block_size=block_size,
            task_count=task_count,
            d_model=d_model,
            attention_heads=attention_heads,
            inter_layers=inter_layers,
            sender_layers=request_sender_layers,
            dropout=dropout,
        )
        self.selector_model = AnchorAwarePostRequestSelector(
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
            sender_layers=selector_sender_layers,
            dropout=dropout,
        )

    def forward(
        self,
        latent_blocks: torch.Tensor,
        process_features: torch.Tensor,
        block_mask: torch.Tensor,
        certificate_features: torch.Tensor,
        task_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        request_outputs = self.request_model(
            latent_blocks[:, :1],
            process_features[:, :1],
            block_mask[:, :1],
            certificate_features[:, :1],
            task_idx,
        )
        selector_outputs = self.selector_model(
            latent_blocks,
            process_features,
            block_mask,
            certificate_features,
            task_idx,
        )
        return {
            "request_logit": request_outputs["helpful_logit"],
            "request_gain_pred": request_outputs["gain_pred"],
            **selector_outputs,
        }


def main() -> None:
    summary = train_joint_request_select(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> JointRequestSelectConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets-jsonl", default=JointRequestSelectConfig.packets_jsonl)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fuser-checkpoint", default=JointRequestSelectConfig.fuser_checkpoint)
    parser.add_argument("--data-root", default=JointRequestSelectConfig.data_root)
    parser.add_argument("--acc-calc-script", default=JointRequestSelectConfig.acc_calc_script)
    parser.add_argument("--seed", type=int, default=JointRequestSelectConfig.seed)
    parser.add_argument("--train-ratio", type=float, default=JointRequestSelectConfig.train_ratio)
    parser.add_argument("--valid-ratio", type=float, default=JointRequestSelectConfig.valid_ratio)
    parser.add_argument("--max-groups", type=int, default=JointRequestSelectConfig.max_groups)
    parser.add_argument("--batch-size", type=int, default=JointRequestSelectConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=JointRequestSelectConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=JointRequestSelectConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=JointRequestSelectConfig.weight_decay)
    parser.add_argument("--dropout", type=float, default=JointRequestSelectConfig.dropout)
    parser.add_argument("--d-model", type=int, default=JointRequestSelectConfig.d_model)
    parser.add_argument("--attention-heads", type=int, default=JointRequestSelectConfig.attention_heads)
    parser.add_argument("--inter-layers", type=int, default=JointRequestSelectConfig.inter_layers)
    parser.add_argument("--request-sender-layers", type=int, default=JointRequestSelectConfig.request_sender_layers)
    parser.add_argument("--selector-sender-layers", type=int, default=JointRequestSelectConfig.selector_sender_layers)
    parser.add_argument("--listwise-loss-weight", type=float, default=JointRequestSelectConfig.listwise_loss_weight)
    parser.add_argument("--score-loss-weight", type=float, default=JointRequestSelectConfig.score_loss_weight)
    parser.add_argument("--pairwise-loss-weight", type=float, default=JointRequestSelectConfig.pairwise_loss_weight)
    parser.add_argument("--gain-loss-weight", type=float, default=JointRequestSelectConfig.gain_loss_weight)
    parser.add_argument("--request-loss-weight", type=float, default=JointRequestSelectConfig.request_loss_weight)
    parser.add_argument("--utility-loss-weight", type=float, default=JointRequestSelectConfig.utility_loss_weight)
    parser.add_argument("--request-gain-loss-weight", type=float, default=JointRequestSelectConfig.request_gain_loss_weight)
    parser.add_argument("--request-cost", type=float, default=JointRequestSelectConfig.request_cost)
    parser.add_argument("--listwise-temperature", type=float, default=JointRequestSelectConfig.listwise_temperature)
    parser.add_argument("--selection-output", choices=["score", "rank"], default=JointRequestSelectConfig.selection_output)
    parser.add_argument("--target-request-rates", default=JointRequestSelectConfig.target_request_rates)
    parser.add_argument("--target-helpful-precisions", default=JointRequestSelectConfig.target_helpful_precisions)
    parser.add_argument("--checkpoint-selection-mode", default=JointRequestSelectConfig.checkpoint_selection_mode)
    parser.add_argument("--checkpoint-selection-metric", default=JointRequestSelectConfig.checkpoint_selection_metric)
    parser.add_argument("--checkpoint-selection-signals", default=JointRequestSelectConfig.checkpoint_selection_signals)
    parser.add_argument("--checkpoint-selection-request-rates", default=JointRequestSelectConfig.checkpoint_selection_request_rates)
    parser.add_argument("--valid-interval", type=int, default=JointRequestSelectConfig.valid_interval)
    parser.add_argument("--max-cached-shards", type=int, default=JointRequestSelectConfig.max_cached_shards)
    parser.add_argument("--num-workers", type=int, default=JointRequestSelectConfig.num_workers)
    parser.add_argument("--device", default=JointRequestSelectConfig.device)
    parser.add_argument("--swanlab-mode", default=JointRequestSelectConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=JointRequestSelectConfig.experiment_name)
    args = parser.parse_args()
    return JointRequestSelectConfig(**vars(args))


def train_joint_request_select(config: JointRequestSelectConfig) -> dict[str, Any]:
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

    device = resolve_device(config.device)
    require_cuda_training(device, "train_cola_joint_request_select_policy.py")
    scorer = load_official_scorer(Path(config.acc_calc_script))
    fuser_checkpoint = torch.load(config.fuser_checkpoint, map_location="cpu")
    fuser_train_config = HierarchicalLatentFuserConfig(**fuser_checkpoint["config"])
    data_config = replace(
        fuser_train_config,
        packets_jsonl=config.packets_jsonl,
        output_dir=config.output_dir,
        data_root=config.data_root,
        acc_calc_script=config.acc_calc_script,
        seed=config.seed,
        train_ratio=config.train_ratio,
        valid_ratio=config.valid_ratio,
        max_groups=config.max_groups,
        max_cached_shards=config.max_cached_shards,
        swanlab_mode="disabled",
    )
    groups = build_groups(read_jsonl(Path(config.packets_jsonl)), data_config, scorer)
    if config.max_groups:
        groups = groups[: config.max_groups]
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
        batch_size=config.batch_size,
    )
    train_ds, valid_ds, test_ds = make_joint_datasets(normalized, tensors_by_split)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = JointRequestSelectPolicy(
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
        request_sender_layers=config.request_sender_layers,
        selector_sender_layers=config.selector_sender_layers,
        dropout=config.dropout,
    ).to(device)
    request_labels = (tensors_by_split["train"]["target_scores"].max(dim=1).values > tensors_by_split["train"]["target_scores"][:, 0] + 1e-12).float()
    pos = request_labels.sum().clamp_min(1.0)
    neg = request_labels.shape[0] - pos
    request_pos_weight = (neg / pos).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    run = init_experiment(
        stage="p2e-joint-request-select-policy",
        config={**asdict(config), **device_metadata(device), "request_pos_weight": float(request_pos_weight.detach().cpu())},
        experiment_name=config.experiment_name,
        tags=["cola", "official-benchmark", "p2e", "joint-request-select", "latent-policy"],
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
                    request_pos_weight=request_pos_weight,
                    config=config,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_metrics = {"loss": float(loss.detach().item()), **train_metrics}
                write_metrics(metrics_f, "train", global_step, train_metrics)
                log_metrics(train_metrics, step=global_step, prefix="train")
                if global_step % config.valid_interval == 0:
                    valid_metrics = evaluate_model(
                        model,
                        valid_loader,
                        tensors_by_split["valid"],
                        groups,
                        splits["valid"],
                        fuser_refs["valid"],
                        scorer,
                        device,
                        config,
                    )
                    write_metrics(metrics_f, "valid", global_step, valid_metrics)
                    log_metrics(numeric_metrics(valid_metrics), step=global_step, prefix="valid")
                    current = valid_metrics["selection_metric"]
                    if current > best_metric:
                        best_metric = current
                        best_step = global_step
                        save_checkpoint(checkpoint_dir / "best_checkpoint.pt", model, optimizer, config, metadata, best_step, best_metric)

        valid_metrics = evaluate_model(
            model,
            valid_loader,
            tensors_by_split["valid"],
            groups,
            splits["valid"],
            fuser_refs["valid"],
            scorer,
            device,
            config,
        )
        test_metrics = evaluate_model(
            model,
            test_loader,
            tensors_by_split["test"],
            groups,
            splits["test"],
            fuser_refs["test"],
            scorer,
            device,
            config,
        )
        write_metrics(metrics_f, "valid", global_step, valid_metrics)
        write_metrics(metrics_f, "test", global_step, test_metrics)
        log_metrics(numeric_metrics(valid_metrics), step=global_step, prefix="valid")
        log_metrics(numeric_metrics(test_metrics), step=global_step, prefix="test")
        if valid_metrics["selection_metric"] > best_metric:
            best_metric = valid_metrics["selection_metric"]
            best_step = global_step
            save_checkpoint(checkpoint_dir / "best_checkpoint.pt", model, optimizer, config, metadata, best_step, best_metric)
        save_checkpoint(checkpoint_dir / "last_checkpoint.pt", model, optimizer, config, metadata, global_step, valid_metrics["selection_metric"])
    finally:
        metrics_f.close()
        finish_experiment()

    best_valid_metrics, best_test_metrics, policy_rows, predictions = evaluate_best_checkpoint(
        checkpoint_dir / "best_checkpoint.pt",
        config,
        metadata,
        valid_loader,
        test_loader,
        tensors_by_split,
        groups,
        splits,
        fuser_refs,
        scorer,
        device,
    )
    artifacts = write_outputs(output_dir, config, policy_rows, predictions)
    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "fuser_train_config": asdict(fuser_train_config),
        "swanlab_run_id": getattr(run, "id", None),
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
            **artifacts,
        },
        "interpretation": (
            "P2-E joint request-and-select policy. The request head sees only "
            "the first sender; the selector head sees requested sender packets "
            "only in the after-request branch."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def validate_config(config: JointRequestSelectConfig) -> None:
    if config.valid_interval > 10:
        raise ValueError("valid_interval must be <= 10 steps")
    if config.swanlab_mode != "cloud":
        raise ValueError("training must use SwanLab cloud; pass --swanlab-mode cloud")
    if config.selection_output not in {"score", "rank"}:
        raise ValueError("selection_output must be score or rank")
    if config.checkpoint_selection_mode not in {"target025_request_prob", "valid_rate_frontier", "always_request_score"}:
        raise ValueError("checkpoint_selection_mode must be target025_request_prob, valid_rate_frontier, or always_request_score")
    if config.checkpoint_selection_metric not in {"model_after_request_score", "model_after_request_accuracy"}:
        raise ValueError("checkpoint_selection_metric must be model_after_request_score or model_after_request_accuracy")
    unknown_signals = [signal for signal in parse_str_list(config.checkpoint_selection_signals) if signal not in {"request_prob", "request_gain_pred"}]
    if unknown_signals:
        raise ValueError(f"unknown checkpoint selection signal(s): {unknown_signals}")
    if config.listwise_temperature <= 0:
        raise ValueError("listwise_temperature must be positive")
    if not 0.0 < config.train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0.0 <= config.valid_ratio < 1.0:
        raise ValueError("valid_ratio must be in [0, 1)")
    if config.train_ratio + config.valid_ratio >= 1.0:
        raise ValueError("train_ratio + valid_ratio must be < 1")


def make_joint_datasets(
    normalized: dict[str, dict[str, torch.Tensor]],
    tensors_by_split: dict[str, dict[str, torch.Tensor]],
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    def dataset(split: str) -> TensorDataset:
        tensors = normalized[split]
        raw = tensors_by_split[split]
        return TensorDataset(
            tensors["latent_blocks"],
            tensors["process_features"],
            tensors["block_mask"],
            tensors["certificate_features"],
            tensors["task_idx"],
            raw["labels"],
            raw["target_scores"],
        )

    return dataset("train"), dataset("valid"), dataset("test")


def compute_loss(
    *,
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    target_scores: torch.Tensor,
    request_pos_weight: torch.Tensor,
    config: JointRequestSelectConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    score_loss = F.mse_loss(outputs["score_pred"], target_scores)
    target_dist = F.softmax(target_scores / config.listwise_temperature, dim=1)
    listwise_loss = -(target_dist * F.log_softmax(outputs["rank_logits"], dim=1)).sum(dim=1).mean()
    pairwise_loss = compute_pairwise_loss(outputs["rank_logits"], target_scores)
    gain_target = target_scores - target_scores[:, :1]
    gain_loss = F.mse_loss(outputs["gain_pred"], gain_target)

    oracle_gain = target_scores.max(dim=1).values - target_scores[:, 0]
    request_target = (oracle_gain > 1e-12).float()
    request_prob = torch.sigmoid(outputs["request_logit"])
    request_loss = F.binary_cross_entropy_with_logits(
        outputs["request_logit"],
        request_target,
        pos_weight=request_pos_weight,
    )
    request_gain_loss = F.mse_loss(outputs["request_gain_pred"], oracle_gain)
    selector_prob = F.softmax(outputs["rank_logits"], dim=1)
    expected_selected_score = (selector_prob * target_scores).sum(dim=1)
    first_score = target_scores[:, 0]
    expected_budgeted_score = first_score + request_prob * (expected_selected_score - first_score - config.request_cost)
    utility_loss = -expected_budgeted_score.mean()
    loss = (
        config.score_loss_weight * score_loss
        + config.listwise_loss_weight * listwise_loss
        + config.pairwise_loss_weight * pairwise_loss
        + config.gain_loss_weight * gain_loss
        + config.request_loss_weight * request_loss
        + config.request_gain_loss_weight * request_gain_loss
        + config.utility_loss_weight * utility_loss
    )
    with torch.no_grad():
        selected = select_index(outputs, config.selection_output)
        row_ids = torch.arange(target_scores.shape[0], device=target_scores.device)
        selected_score = target_scores[row_ids, selected].mean()
        selected_correct = labels[row_ids, selected].mean()
    return loss, {
        "score_loss": float(score_loss.detach().item()),
        "listwise_loss": float(listwise_loss.detach().item()),
        "pairwise_loss": float(pairwise_loss.detach().item()),
        "gain_loss": float(gain_loss.detach().item()),
        "request_loss": float(request_loss.detach().item()),
        "request_gain_loss": float(request_gain_loss.detach().item()),
        "utility_loss": float(utility_loss.detach().item()),
        "expected_budgeted_score": float(expected_budgeted_score.detach().mean().item()),
        "request_prob_mean": float(request_prob.detach().mean().item()),
        "batch_selected_score": float(selected_score.detach().item()),
        "batch_selected_accuracy": float(selected_correct.detach().item()),
    }


@torch.no_grad()
def evaluate_model(
    model: JointRequestSelectPolicy,
    loader: DataLoader,
    tensors: dict[str, torch.Tensor],
    groups: list[dict[str, Any]],
    indices: list[int],
    fuser_refs: dict[str, torch.Tensor],
    scorer: Any,
    device: torch.device,
    config: JointRequestSelectConfig,
) -> dict[str, float]:
    predictions = score_predictions_from_model(model, loader, tensors, groups, indices, fuser_refs, scorer, device, config)
    base = aggregate(predictions)
    threshold = choose_threshold_for_rate(predictions, "request_prob", 0.25)
    target025 = request_policy_metrics(predictions, "request_prob", threshold)
    base.update({f"target025_{key}": value for key, value in target025.items()})
    base["target025_threshold"] = threshold
    frontier = choose_valid_rate_frontier(predictions, config)
    base.update({f"frontier_{key}": value for key, value in frontier.items()})
    base["selection_metric"] = select_checkpoint_metric(base, target025, frontier, config)
    return base


def choose_valid_rate_frontier(
    predictions: list[dict[str, Any]],
    config: JointRequestSelectConfig,
) -> dict[str, float | str]:
    rates = parse_float_list(config.checkpoint_selection_request_rates or config.target_request_rates)
    signals = parse_str_list(config.checkpoint_selection_signals)
    rows: list[dict[str, float | str]] = []
    for signal in signals:
        for rate in rates:
            threshold = choose_threshold_for_rate(predictions, signal, rate)
            metrics = request_policy_metrics(predictions, signal, threshold)
            rows.append(
                {
                    "selection_mode": "target_request_rate",
                    "target_value": float(rate),
                    "signal": signal,
                    "threshold": threshold,
                    **metrics,
                }
            )
    if not rows:
        return {
            "selection_mode": "none",
            "target_value": 0.0,
            "signal": "none",
            "threshold": 1.0,
            "model_after_request_score": 0.0,
            "model_after_request_accuracy": 0.0,
        }
    return max(
        rows,
        key=lambda row: (
            float(row[config.checkpoint_selection_metric]),
            float(row["model_score_gain_vs_first"]),
            -float(row["avg_sender_budget"]),
        ),
    )


def select_checkpoint_metric(
    base_metrics: dict[str, float],
    target025: dict[str, float],
    frontier: dict[str, float | str],
    config: JointRequestSelectConfig,
) -> float:
    if config.checkpoint_selection_mode == "target025_request_prob":
        return float(target025[config.checkpoint_selection_metric])
    if config.checkpoint_selection_mode == "valid_rate_frontier":
        return float(frontier[config.checkpoint_selection_metric])
    return float(base_metrics["always_request_model_score"])


@torch.no_grad()
def score_predictions_from_model(
    model: JointRequestSelectPolicy,
    loader: DataLoader,
    tensors: dict[str, torch.Tensor],
    groups: list[dict[str, Any]],
    indices: list[int],
    fuser_refs: dict[str, torch.Tensor],
    scorer: Any,
    device: torch.device,
    config: JointRequestSelectConfig,
) -> list[dict[str, Any]]:
    model.eval()
    outputs = {"rank_logits": [], "score_pred": [], "gain_pred": [], "request_logit": [], "request_gain_pred": []}
    for batch in loader:
        batch = [item.to(device) for item in batch]
        batch_outputs = model(*batch[:5])
        for key in outputs:
            outputs[key].append(batch_outputs[key].cpu())
    cat_outputs = {key: torch.cat(value, dim=0) for key, value in outputs.items()}
    selected = select_index(cat_outputs, config.selection_output)
    request_prob = torch.sigmoid(cat_outputs["request_logit"])
    labels = tensors["labels"]
    target_scores = tensors["target_scores"]
    predictions = []
    for local_idx, group_index in enumerate(indices):
        group = groups[group_index]
        first_member = group["members"][0]
        text_choice = choose_text_majority_selected(group["members"])
        text_score = score_text_with_official_rules(
            task=str(group["task"]),
            text=str(text_choice["prediction"]),
            ground_truth=group["ground_truth"],
            choices=group["choices"],
            scorer=scorer,
        )
        selected_idx = int(selected[local_idx])
        predictions.append(
            {
                "task": group["task"],
                "sample_key": group["sample_key"],
                "sample_id": group["sample_id"],
                "model_selected_index": selected_idx,
                "model_correct": float(labels[local_idx, selected_idx].item()),
                "model_score": float(target_scores[local_idx, selected_idx].item()),
                "request_prob": float(request_prob[local_idx].item()),
                "request_gain_pred": float(cat_outputs["request_gain_pred"][local_idx].item()),
                "first_correct": float(bool(first_member["selected_correct"])),
                "first_score": float(first_member.get("selected_score", float(bool(first_member["selected_correct"])))),
                "fuser_correct": float(fuser_refs["fuser_correct"][local_idx].item()),
                "fuser_score": float(fuser_refs["fuser_score"][local_idx].item()),
                "text_majority_correct": float(bool(text_score["correct"])),
                "text_majority_score": float(text_score["score"]),
                "oracle_any_correct": float(labels[local_idx].sum().item() > 0),
                "oracle_best_score": float(target_scores[local_idx].max().item()),
                "oracle_request_helpful": float(target_scores[local_idx].max().item() > target_scores[local_idx, 0].item() + 1e-12),
                "model_request_helpful": float(target_scores[local_idx, selected_idx].item() > target_scores[local_idx, 0].item() + 1e-12),
            }
        )
    return predictions


def aggregate(rows: list[dict[str, Any]], *, task: str = "all") -> dict[str, float]:
    if not rows:
        return {"task": task, "count": 0.0, "selection_metric": 0.0}
    metrics = {
        "task": task,
        "count": float(len(rows)),
        "always_request_model_accuracy": mean(row["model_correct"] for row in rows),
        "always_request_model_score": mean(row["model_score"] for row in rows),
        "request_prob_mean": mean(row["request_prob"] for row in rows),
        "first_accuracy": mean(row["first_correct"] for row in rows),
        "first_score": mean(row["first_score"] for row in rows),
        "fuser_accuracy": mean(row["fuser_correct"] for row in rows),
        "fuser_score": mean(row["fuser_score"] for row in rows),
        "text_majority_accuracy": mean(row["text_majority_correct"] for row in rows),
        "text_majority_score": mean(row["text_majority_score"] for row in rows),
        "oracle_any_accuracy": mean(row["oracle_any_correct"] for row in rows),
        "oracle_best_score": mean(row["oracle_best_score"] for row in rows),
    }
    metrics["model_fraction_of_oracle_score_gap_closed"] = safe_gap_fraction(
        metrics["always_request_model_score"],
        metrics["first_score"],
        metrics["oracle_best_score"],
    )
    metrics["selection_metric"] = metrics["always_request_model_score"]
    return metrics


def evaluate_threshold_policies(
    *,
    valid_predictions: list[dict[str, Any]],
    test_predictions: list[dict[str, Any]],
    request_rates: list[float],
    helpful_precisions: list[float],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for signal in ["request_prob", "request_gain_pred"]:
        for rate in request_rates:
            threshold = choose_threshold_for_rate(valid_predictions, signal, rate)
            rows.append(policy_row("target_request_rate", rate, signal, threshold, valid_predictions, test_predictions))
        for precision in helpful_precisions:
            threshold = choose_threshold_for_helpful_precision(valid_predictions, signal, precision)
            rows.append(policy_row("target_helpful_precision", precision, signal, threshold, valid_predictions, test_predictions))
    return rows


def policy_row(
    mode: str,
    target: float,
    signal: str,
    threshold: float,
    valid_predictions: list[dict[str, Any]],
    test_predictions: list[dict[str, Any]],
) -> dict[str, float | str]:
    return {
        "selection_mode": mode,
        "target_value": target,
        "signal": signal,
        "threshold": threshold,
        **prefixed("valid", request_policy_metrics(valid_predictions, signal, threshold)),
        **prefixed("test", request_policy_metrics(test_predictions, signal, threshold)),
    }


def request_policy_metrics(rows: list[dict[str, Any]], signal: str, threshold: float) -> dict[str, float]:
    requested = [float(row[signal]) >= threshold for row in rows]
    model_correct = [row["model_correct"] if req else row["first_correct"] for row, req in zip(rows, requested)]
    model_score = [row["model_score"] if req else row["first_score"] for row, req in zip(rows, requested)]
    oracle_correct = [row["oracle_any_correct"] if req else row["first_correct"] for row, req in zip(rows, requested)]
    oracle_score = [row["oracle_best_score"] if req else row["first_score"] for row, req in zip(rows, requested)]
    requested_rows = [row for row, req in zip(rows, requested) if req]
    return {
        "request_rate": mean(float(req) for req in requested),
        "avg_sender_budget": 1.0 + 2.0 * mean(float(req) for req in requested),
        "model_helpful_precision": mean(row["model_request_helpful"] for row in requested_rows),
        "oracle_helpful_precision": mean(row["oracle_request_helpful"] for row in requested_rows),
        "model_after_request_accuracy": mean(model_correct),
        "model_after_request_score": mean(model_score),
        "oracle_after_request_accuracy": mean(oracle_correct),
        "oracle_after_request_score": mean(oracle_score),
        "model_score_gain_vs_first": mean(score - row["first_score"] for score, row in zip(model_score, rows)),
        "oracle_score_gain_vs_first": mean(score - row["first_score"] for score, row in zip(oracle_score, rows)),
    }


def choose_threshold_for_rate(rows: list[dict[str, Any]], signal: str, target_rate: float) -> float:
    values = sorted(float(row[signal]) for row in rows)[::-1]
    if not values:
        return 1.0
    k = max(1, min(len(values), int(round(target_rate * len(values)))))
    return float(values[k - 1])


def choose_threshold_for_helpful_precision(rows: list[dict[str, Any]], signal: str, target_precision: float) -> float:
    thresholds = sorted({float(row[signal]) for row in rows}, reverse=True)
    best_threshold = 1.0
    best_rate = -1.0
    for threshold in thresholds:
        requested = [row for row in rows if float(row[signal]) >= threshold]
        if not requested:
            continue
        precision = mean(row["model_request_helpful"] for row in requested)
        rate = len(requested) / len(rows)
        if precision >= target_precision and rate > best_rate:
            best_threshold = float(threshold)
            best_rate = rate
    return best_threshold


def choose_policy_row(rows: list[dict[str, float | str]], mode: str, target: float, signal: str) -> dict[str, float | str] | None:
    for row in rows:
        if (
            row["selection_mode"] == mode
            and row["signal"] == signal
            and abs(float(row["target_value"]) - target) < 1e-9
        ):
            return row
    return None


def evaluate_best_checkpoint(
    path: Path,
    config: JointRequestSelectConfig,
    metadata: dict[str, Any],
    valid_loader: DataLoader,
    test_loader: DataLoader,
    tensors_by_split: dict[str, dict[str, torch.Tensor]],
    groups: list[dict[str, Any]],
    splits: dict[str, list[int]],
    fuser_refs: dict[str, dict[str, torch.Tensor]],
    scorer: Any,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float], list[dict[str, float | str]], dict[str, list[dict[str, Any]]]]:
    checkpoint = torch.load(path, map_location="cpu")
    model = JointRequestSelectPolicy(
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
        request_sender_layers=config.request_sender_layers,
        selector_sender_layers=config.selector_sender_layers,
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    valid_predictions = score_predictions_from_model(
        model, valid_loader, tensors_by_split["valid"], groups, splits["valid"], fuser_refs["valid"], scorer, device, config
    )
    test_predictions = score_predictions_from_model(
        model, test_loader, tensors_by_split["test"], groups, splits["test"], fuser_refs["test"], scorer, device, config
    )
    policy_rows = evaluate_threshold_policies(
        valid_predictions=valid_predictions,
        test_predictions=test_predictions,
        request_rates=parse_float_list(config.target_request_rates),
        helpful_precisions=parse_float_list(config.target_helpful_precisions),
    )
    return aggregate(valid_predictions), aggregate(test_predictions), policy_rows, {"valid": valid_predictions, "test": test_predictions}


def write_outputs(
    output_dir: Path,
    config: JointRequestSelectConfig,
    policy_rows: list[dict[str, float | str]],
    predictions: dict[str, list[dict[str, Any]]],
) -> dict[str, str]:
    valid_predictions_path = output_dir / "valid_predictions.jsonl"
    test_predictions_path = output_dir / "test_predictions.jsonl"
    policy_path = output_dir / "policy_metrics.csv"
    per_task_path = output_dir / "test_per_task_metrics.csv"
    write_jsonl(valid_predictions_path, predictions["valid"])
    write_jsonl(test_predictions_path, predictions["test"])
    write_csv(policy_path, policy_rows)
    per_task = [
        aggregate([row for row in predictions["test"] if row["task"] == task], task=task)
        for task in sorted({row["task"] for row in predictions["test"]})
    ]
    write_csv(per_task_path, per_task)
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "valid_predictions_jsonl": str(valid_predictions_path),
        "test_predictions_jsonl": str(test_predictions_path),
        "policy_metrics_csv": str(policy_path),
        "test_per_task_metrics_csv": str(per_task_path),
        "config_json": str(config_path),
    }


def save_checkpoint(
    path: Path,
    model: JointRequestSelectPolicy,
    optimizer: torch.optim.Optimizer,
    config: JointRequestSelectConfig,
    metadata: dict[str, Any],
    step: int,
    metric: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "metadata": metadata,
            "step": step,
            "metric": metric,
        },
        path,
    )


def write_metrics(handle: Any, split: str, step: int, metrics: dict[str, float]) -> None:
    handle.write(json.dumps({"created_at": int(time.time()), "split": split, "step": step, "metrics": metrics}, sort_keys=True) + "\n")
    handle.flush()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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


def parse_str_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def safe_gap_fraction(model_score: float, base_score: float, oracle_score: float) -> float:
    denom = oracle_score - base_score
    if denom <= 0:
        return 0.0
    return float((model_score - base_score) / denom)


def numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items() if isinstance(value, int | float)}


def mean(values: Any) -> float:
    values = [float(value) for value in values]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


if __name__ == "__main__":
    main()
