"""Train a P2-E request-more-latent policy.

The policy sees only the first sender's sanitized latent packet and decides
whether the receiver should request the remaining sender packets.  If it
requests more evidence, evaluation reports two after-request views:

* oracle-after-request: offline upper bound using the best sender score
* fuser-after-request: practical decoder-free fuser v2 selection

Decoded answers and official scores are labels/evaluation references only.
They are not model inputs.
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
    make_datasets,
    read_jsonl,
    split_groups,
)
from drla.scripts.train_cola_hierarchical_state_verifier import auroc, pearson
from drla.scripts.train_cola_readiness_model import device_metadata, require_cuda_training, resolve_device
from drla.tracking import finish_experiment, init_experiment, log_metrics


@dataclass(frozen=True)
class RequestMorePolicyConfig:
    packets_jsonl: str = (
        "/data1/luyifei/drla/outputs/cola_agent_latent_comm/"
        "p2_agent_latent_comm_v2_locked_seed66_67_68_split20260601_20260529/"
        "agent_latent_comm_packets_test.jsonl"
    )
    output_dir: str = (
        "/data1/luyifei/drla/outputs/cola_request_more_policy/"
        "p2e_request_more_policy_v1"
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
    sender_layers: int = 1
    target_mode: str = "fuser_gain"
    gain_loss_weight: float = 1.0
    valid_interval: int = 10
    max_cached_shards: int = 1024
    num_workers: int = 0
    target_request_rates: str = "0.10,0.25,0.50"
    target_helpful_precisions: str = "0.50,0.60,0.70"
    device: str = "auto"
    swanlab_mode: str = "cloud"
    experiment_name: str = "p2e-request-more-policy-v1"


class FirstSenderRequestPolicy(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        process_dim: int,
        certificate_dim: int,
        max_blocks: int,
        block_size: int,
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
            sender_count=1,
            task_count=task_count,
            d_model=d_model,
            attention_heads=attention_heads,
            inter_layers=inter_layers,
            sender_layers=sender_layers,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.helpful_head = nn.Linear(d_model, 1)
        self.gain_head = nn.Linear(d_model, 1)

    def forward(
        self,
        latent_blocks: torch.Tensor,
        process_features: torch.Tensor,
        block_mask: torch.Tensor,
        certificate_features: torch.Tensor,
        task_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        state = self.backbone.encode_senders(
            latent_blocks=latent_blocks,
            process_features=process_features,
            block_mask=block_mask,
            certificate_features=certificate_features,
            task_idx=task_idx,
        ).squeeze(1)
        hidden = self.head(state)
        return {
            "helpful_logit": self.helpful_head(hidden).squeeze(-1),
            "gain_pred": self.gain_head(hidden).squeeze(-1),
        }


def main() -> None:
    summary = train_request_more_policy(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> RequestMorePolicyConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets-jsonl", default=RequestMorePolicyConfig.packets_jsonl)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fuser-checkpoint", default=RequestMorePolicyConfig.fuser_checkpoint)
    parser.add_argument("--data-root", default=RequestMorePolicyConfig.data_root)
    parser.add_argument("--acc-calc-script", default=RequestMorePolicyConfig.acc_calc_script)
    parser.add_argument("--seed", type=int, default=RequestMorePolicyConfig.seed)
    parser.add_argument("--train-ratio", type=float, default=RequestMorePolicyConfig.train_ratio)
    parser.add_argument("--valid-ratio", type=float, default=RequestMorePolicyConfig.valid_ratio)
    parser.add_argument("--max-groups", type=int, default=RequestMorePolicyConfig.max_groups)
    parser.add_argument("--batch-size", type=int, default=RequestMorePolicyConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=RequestMorePolicyConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=RequestMorePolicyConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=RequestMorePolicyConfig.weight_decay)
    parser.add_argument("--dropout", type=float, default=RequestMorePolicyConfig.dropout)
    parser.add_argument("--d-model", type=int, default=RequestMorePolicyConfig.d_model)
    parser.add_argument("--attention-heads", type=int, default=RequestMorePolicyConfig.attention_heads)
    parser.add_argument("--inter-layers", type=int, default=RequestMorePolicyConfig.inter_layers)
    parser.add_argument("--sender-layers", type=int, default=RequestMorePolicyConfig.sender_layers)
    parser.add_argument("--target-mode", choices=["oracle_gain", "fuser_gain"], default=RequestMorePolicyConfig.target_mode)
    parser.add_argument("--gain-loss-weight", type=float, default=RequestMorePolicyConfig.gain_loss_weight)
    parser.add_argument("--valid-interval", type=int, default=RequestMorePolicyConfig.valid_interval)
    parser.add_argument("--max-cached-shards", type=int, default=RequestMorePolicyConfig.max_cached_shards)
    parser.add_argument("--num-workers", type=int, default=RequestMorePolicyConfig.num_workers)
    parser.add_argument("--target-request-rates", default=RequestMorePolicyConfig.target_request_rates)
    parser.add_argument("--target-helpful-precisions", default=RequestMorePolicyConfig.target_helpful_precisions)
    parser.add_argument("--device", default=RequestMorePolicyConfig.device)
    parser.add_argument("--swanlab-mode", default=RequestMorePolicyConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=RequestMorePolicyConfig.experiment_name)
    args = parser.parse_args()
    return RequestMorePolicyConfig(
        packets_jsonl=args.packets_jsonl,
        output_dir=args.output_dir,
        fuser_checkpoint=args.fuser_checkpoint,
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
        target_mode=args.target_mode,
        gain_loss_weight=args.gain_loss_weight,
        valid_interval=args.valid_interval,
        max_cached_shards=args.max_cached_shards,
        num_workers=args.num_workers,
        target_request_rates=args.target_request_rates,
        target_helpful_precisions=args.target_helpful_precisions,
        device=args.device,
        swanlab_mode=args.swanlab_mode,
        experiment_name=args.experiment_name,
    )


def train_request_more_policy(config: RequestMorePolicyConfig) -> dict[str, Any]:
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
    require_cuda_training(device, "train_cola_request_more_policy.py")
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
    eval_refs = build_eval_refs(groups, splits, tensors_by_split, fuser_refs, scorer)
    train_ds, valid_ds, test_ds = make_request_datasets(normalized, eval_refs, config)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = FirstSenderRequestPolicy(
        latent_dim=metadata["latent_dim"],
        process_dim=metadata["process_dim"],
        certificate_dim=metadata["certificate_dim"],
        max_blocks=metadata["max_blocks"],
        block_size=metadata["block_size"],
        task_count=len(metadata["task_to_idx"]),
        d_model=config.d_model,
        attention_heads=config.attention_heads,
        inter_layers=config.inter_layers,
        sender_layers=config.sender_layers,
        dropout=config.dropout,
    ).to(device)
    train_helpful = eval_refs["train"][f"{config.target_mode}_helpful"]
    pos = train_helpful.sum().clamp_min(1.0)
    neg = train_helpful.shape[0] - pos
    pos_weight = (neg / pos).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    run = init_experiment(
        stage="p2e-request-more-policy",
        config={
            **asdict(config),
            **device_metadata(device),
            "helpful_pos_weight": float(pos_weight.detach().cpu()),
        },
        experiment_name=config.experiment_name,
        tags=["cola", "official-benchmark", "p2e", "request-more-latent", "latent-policy"],
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
                    helpful_target=batch[5],
                    gain_target=batch[6],
                    pos_weight=pos_weight,
                    gain_loss_weight=config.gain_loss_weight,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_metrics = {"loss": float(loss.detach().item()), **train_metrics}
                write_metrics(metrics_f, "train", global_step, train_metrics)
                log_metrics(train_metrics, step=global_step, prefix="train")
                if global_step % config.valid_interval == 0:
                    valid_metrics = evaluate_model(model, valid_loader, eval_refs["valid"], device, config)
                    write_metrics(metrics_f, "valid", global_step, valid_metrics)
                    log_metrics(valid_metrics, step=global_step, prefix="valid")
                    current = valid_metrics["selection_metric"]
                    if current > best_metric:
                        best_metric = current
                        best_step = global_step
                        save_checkpoint(checkpoint_dir / "best_checkpoint.pt", model, optimizer, config, metadata, best_step, best_metric)

        valid_metrics = evaluate_model(model, valid_loader, eval_refs["valid"], device, config)
        test_metrics = evaluate_model(model, test_loader, eval_refs["test"], device, config)
        write_metrics(metrics_f, "valid", global_step, valid_metrics)
        write_metrics(metrics_f, "test", global_step, test_metrics)
        log_metrics(valid_metrics, step=global_step, prefix="valid")
        log_metrics(test_metrics, step=global_step, prefix="test")
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
        eval_refs,
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
        "baseline_metrics": {
            "valid": baseline_metrics(eval_refs["valid"]),
            "test": baseline_metrics(eval_refs["test"]),
        },
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "metrics_jsonl": str(metrics_path),
            "best_checkpoint": str(checkpoint_dir / "best_checkpoint.pt"),
            "last_checkpoint": str(checkpoint_dir / "last_checkpoint.pt"),
            **artifacts,
        },
        "interpretation": (
            "P2-E learned request-more-latent policy. The policy sees only the "
            "first sender latent packet and decides whether to request additional "
            "sender evidence. Fuser/oracle/text scores are offline evaluation views."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def validate_config(config: RequestMorePolicyConfig) -> None:
    if config.valid_interval > 10:
        raise ValueError("valid_interval must be <= 10 steps")
    if config.swanlab_mode != "cloud":
        raise ValueError("training must use SwanLab cloud; pass --swanlab-mode cloud")
    if config.target_mode not in {"oracle_gain", "fuser_gain"}:
        raise ValueError("target_mode must be oracle_gain or fuser_gain")
    if not 0.0 < config.train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0.0 <= config.valid_ratio < 1.0:
        raise ValueError("valid_ratio must be in [0, 1)")
    if config.train_ratio + config.valid_ratio >= 1.0:
        raise ValueError("train_ratio + valid_ratio must be < 1")


def normalize_tensors(
    tensors_by_split: dict[str, dict[str, torch.Tensor]],
    norm_stats: dict[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor]]:
    normalized = {}
    process_mean = norm_stats["process_mean"].view(1, 1, 1, -1)
    process_std = norm_stats["process_std"].view(1, 1, 1, -1).clamp_min(1e-6)
    cert_mean = norm_stats["certificate_mean"].view(1, 1, -1)
    cert_std = norm_stats["certificate_std"].view(1, 1, -1).clamp_min(1e-6)
    for split, tensors in tensors_by_split.items():
        process = (tensors["process_features"] - process_mean) / process_std
        process = process.masked_fill(~tensors["block_mask"].unsqueeze(-1), 0.0)
        cert = (tensors["certificate_features"] - cert_mean) / cert_std
        normalized[split] = {
            **tensors,
            "process_features": process,
            "certificate_features": cert,
        }
    return normalized


@torch.no_grad()
def compute_fuser_refs(
    *,
    fuser_checkpoint: dict[str, Any],
    fuser_config: HierarchicalLatentFuserConfig,
    metadata: dict[str, Any],
    normalized: dict[str, dict[str, torch.Tensor]],
    tensors_by_split: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, torch.Tensor]]:
    model = HierarchicalLatentFuser(
        latent_dim=metadata["latent_dim"],
        process_dim=metadata["process_dim"],
        certificate_dim=metadata["certificate_dim"],
        max_blocks=metadata["max_blocks"],
        block_size=metadata["block_size"],
        sender_count=metadata["sender_count"],
        task_count=len(metadata["task_to_idx"]),
        d_model=fuser_config.d_model,
        attention_heads=fuser_config.attention_heads,
        inter_layers=fuser_config.inter_layers,
        sender_layers=fuser_config.sender_layers,
        dropout=fuser_config.dropout,
    ).to(device)
    model.load_state_dict(fuser_checkpoint["model_state_dict"])
    model.eval()
    refs = {}
    for split, tensors in normalized.items():
        dataset = TensorDataset(
            tensors["latent_blocks"],
            tensors["process_features"],
            tensors["block_mask"],
            tensors["certificate_features"],
            tensors["task_idx"],
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        logits = []
        for batch in loader:
            batch = [item.to(device) for item in batch]
            logits.append(model(*batch).cpu())
        logits_tensor = torch.cat(logits, dim=0)
        selected = logits_tensor.argmax(dim=1)
        row_ids = torch.arange(selected.shape[0])
        labels = tensors_by_split[split]["labels"]
        scores = tensors_by_split[split]["target_scores"]
        refs[split] = {
            "fuser_selected_idx": selected,
            "fuser_correct": labels[row_ids, selected].float(),
            "fuser_score": scores[row_ids, selected].float(),
            "fuser_confidence": torch.softmax(logits_tensor, dim=1).max(dim=1).values,
        }
    return refs


def build_eval_refs(
    groups: list[dict[str, Any]],
    splits: dict[str, list[int]],
    tensors_by_split: dict[str, dict[str, torch.Tensor]],
    fuser_refs: dict[str, dict[str, torch.Tensor]],
    scorer: Any,
) -> dict[str, dict[str, torch.Tensor]]:
    refs = {}
    for split, indices in splits.items():
        labels = tensors_by_split[split]["labels"].float()
        scores = tensors_by_split[split]["target_scores"].float()
        first_correct = labels[:, 0]
        first_score = scores[:, 0]
        oracle_correct = (labels.sum(dim=1) > 0).float()
        oracle_score = scores.max(dim=1).values
        text_correct = []
        text_score = []
        for group_index in indices:
            group = groups[group_index]
            text_choice = choose_text_majority_selected(group["members"])
            score = score_text_with_official_rules(
                task=str(group["task"]),
                text=str(text_choice["prediction"]),
                ground_truth=group["ground_truth"],
                choices=group["choices"],
                scorer=scorer,
            )
            text_correct.append(float(bool(score["correct"])))
            text_score.append(float(score["score"]))
        fuser_score = fuser_refs[split]["fuser_score"]
        oracle_gain = oracle_score - first_score
        fuser_gain = fuser_score - first_score
        refs[split] = {
            "first_correct": first_correct,
            "first_score": first_score,
            "oracle_correct": oracle_correct,
            "oracle_score": oracle_score,
            "fuser_correct": fuser_refs[split]["fuser_correct"],
            "fuser_score": fuser_score,
            "fuser_confidence": fuser_refs[split]["fuser_confidence"],
            "text_correct": torch.tensor(text_correct, dtype=torch.float32),
            "text_score": torch.tensor(text_score, dtype=torch.float32),
            "oracle_gain": oracle_gain,
            "oracle_gain_helpful": (oracle_gain > 1e-12).float(),
            "fuser_gain": fuser_gain,
            "fuser_gain_helpful": (fuser_gain > 1e-12).float(),
        }
    return refs


def make_request_datasets(
    normalized: dict[str, dict[str, torch.Tensor]],
    eval_refs: dict[str, dict[str, torch.Tensor]],
    config: RequestMorePolicyConfig,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    def dataset(split: str) -> TensorDataset:
        tensors = normalized[split]
        refs = eval_refs[split]
        return TensorDataset(
            tensors["latent_blocks"][:, :1],
            tensors["process_features"][:, :1],
            tensors["block_mask"][:, :1],
            tensors["certificate_features"][:, :1],
            tensors["task_idx"],
            refs[f"{config.target_mode}_helpful"],
            refs[config.target_mode],
        )

    return dataset("train"), dataset("valid"), dataset("test")


def compute_loss(
    *,
    outputs: dict[str, torch.Tensor],
    helpful_target: torch.Tensor,
    gain_target: torch.Tensor,
    pos_weight: torch.Tensor,
    gain_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    helpful_loss = F.binary_cross_entropy_with_logits(outputs["helpful_logit"], helpful_target, pos_weight=pos_weight)
    gain_loss = F.mse_loss(outputs["gain_pred"], gain_target)
    loss = helpful_loss + gain_loss_weight * gain_loss
    with torch.no_grad():
        prob = torch.sigmoid(outputs["helpful_logit"])
        metrics = binary_gain_metrics(prob.detach().cpu(), outputs["gain_pred"].detach().cpu(), helpful_target.detach().cpu(), gain_target.detach().cpu())
    return loss, {"helpful_loss": float(helpful_loss.detach().item()), "gain_loss": float(gain_loss.detach().item()), **metrics}


@torch.no_grad()
def evaluate_model(
    model: FirstSenderRequestPolicy,
    loader: DataLoader,
    refs: dict[str, torch.Tensor],
    device: torch.device,
    config: RequestMorePolicyConfig,
) -> dict[str, float]:
    model.eval()
    probs = []
    gains = []
    for batch in loader:
        batch = [item.to(device) for item in batch]
        outputs = model(*batch[:5])
        probs.append(torch.sigmoid(outputs["helpful_logit"]).cpu())
        gains.append(outputs["gain_pred"].cpu())
    prob = torch.cat(probs)
    gain_pred = torch.cat(gains)
    helpful = refs[f"{config.target_mode}_helpful"]
    gain_target = refs[config.target_mode]
    metrics = binary_gain_metrics(prob, gain_pred, helpful, gain_target)
    metrics.update(baseline_metrics(refs))
    metrics["selection_metric"] = metrics["helpful_auroc"] + max(0.0, metrics["gain_corr"])
    return metrics


def binary_gain_metrics(
    prob: torch.Tensor,
    gain_pred: torch.Tensor,
    helpful: torch.Tensor,
    gain_target: torch.Tensor,
) -> dict[str, float]:
    error = gain_pred - gain_target
    return {
        "helpful_auroc": auroc(prob, helpful),
        "helpful_brier": float((prob - helpful).square().mean().item()),
        "helpful_prob_mean": float(prob.mean().item()),
        "helpful_target_mean": float(helpful.mean().item()),
        "gain_mae": float(error.abs().mean().item()),
        "gain_rmse": float(torch.sqrt(error.square().mean()).item()),
        "gain_corr": pearson(gain_pred, gain_target),
        "gain_pred_mean": float(gain_pred.mean().item()),
        "gain_target_mean": float(gain_target.mean().item()),
    }


def baseline_metrics(refs: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        "first_accuracy": tensor_mean(refs["first_correct"]),
        "first_score": tensor_mean(refs["first_score"]),
        "always_request_oracle_accuracy": tensor_mean(refs["oracle_correct"]),
        "always_request_oracle_score": tensor_mean(refs["oracle_score"]),
        "always_request_fuser_accuracy": tensor_mean(refs["fuser_correct"]),
        "always_request_fuser_score": tensor_mean(refs["fuser_score"]),
        "text_majority_accuracy": tensor_mean(refs["text_correct"]),
        "text_majority_score": tensor_mean(refs["text_score"]),
        "oracle_request_helpful_rate": tensor_mean(refs["oracle_gain_helpful"]),
        "fuser_request_helpful_rate": tensor_mean(refs["fuser_gain_helpful"]),
    }


def evaluate_best_checkpoint(
    path: Path,
    config: RequestMorePolicyConfig,
    metadata: dict[str, Any],
    valid_loader: DataLoader,
    test_loader: DataLoader,
    eval_refs: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float], list[dict[str, float | str]], dict[str, str]]:
    checkpoint = torch.load(path, map_location="cpu")
    model = FirstSenderRequestPolicy(
        latent_dim=metadata["latent_dim"],
        process_dim=metadata["process_dim"],
        certificate_dim=metadata["certificate_dim"],
        max_blocks=metadata["max_blocks"],
        block_size=metadata["block_size"],
        task_count=len(metadata["task_to_idx"]),
        d_model=config.d_model,
        attention_heads=config.attention_heads,
        inter_layers=config.inter_layers,
        sender_layers=config.sender_layers,
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_valid = evaluate_model(model, valid_loader, eval_refs["valid"], device, config)
    best_test = evaluate_model(model, test_loader, eval_refs["test"], device, config)
    valid_pred = predict(model, valid_loader, device)
    test_pred = predict(model, test_loader, device)
    policy_rows = evaluate_request_policies(
        valid_refs=eval_refs["valid"],
        test_refs=eval_refs["test"],
        valid_pred=valid_pred,
        test_pred=test_pred,
        request_rates=parse_float_list(config.target_request_rates),
        helpful_precisions=parse_float_list(config.target_helpful_precisions),
        target_mode=config.target_mode,
    )
    predictions = {
        "valid_predictions_jsonl": "",
        "test_predictions_jsonl": "",
        "_valid_prob": valid_pred["prob"],
        "_valid_gain_pred": valid_pred["gain_pred"],
        "_test_prob": test_pred["prob"],
        "_test_gain_pred": test_pred["gain_pred"],
    }
    return best_valid, best_test, policy_rows, predictions


@torch.no_grad()
def predict(model: FirstSenderRequestPolicy, loader: DataLoader, device: torch.device) -> dict[str, torch.Tensor]:
    model.eval()
    probs = []
    gains = []
    for batch in loader:
        batch = [item.to(device) for item in batch]
        outputs = model(*batch[:5])
        probs.append(torch.sigmoid(outputs["helpful_logit"]).cpu())
        gains.append(outputs["gain_pred"].cpu())
    return {"prob": torch.cat(probs), "gain_pred": torch.cat(gains)}


def evaluate_request_policies(
    *,
    valid_refs: dict[str, torch.Tensor],
    test_refs: dict[str, torch.Tensor],
    valid_pred: dict[str, torch.Tensor],
    test_pred: dict[str, torch.Tensor],
    request_rates: list[float],
    helpful_precisions: list[float],
    target_mode: str,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for signal_name in ["prob", "gain_pred"]:
        for rate in request_rates:
            threshold = choose_threshold_for_rate(valid_pred[signal_name], rate)
            rows.append(policy_row("target_request_rate", rate, signal_name, threshold, valid_refs, test_refs, valid_pred, test_pred, target_mode))
        for precision in helpful_precisions:
            threshold = choose_threshold_for_helpful_precision(valid_pred[signal_name], valid_refs[f"{target_mode}_helpful"], precision)
            rows.append(policy_row("target_helpful_precision", precision, signal_name, threshold, valid_refs, test_refs, valid_pred, test_pred, target_mode))
    return rows


def policy_row(
    mode: str,
    target: float,
    signal_name: str,
    threshold: float,
    valid_refs: dict[str, torch.Tensor],
    test_refs: dict[str, torch.Tensor],
    valid_pred: dict[str, torch.Tensor],
    test_pred: dict[str, torch.Tensor],
    target_mode: str,
) -> dict[str, float | str]:
    return {
        "selection_mode": mode,
        "target_value": target,
        "signal": signal_name,
        "threshold": threshold,
        **prefixed("valid", request_policy_metrics(valid_refs, valid_pred[signal_name], threshold, target_mode)),
        **prefixed("test", request_policy_metrics(test_refs, test_pred[signal_name], threshold, target_mode)),
    }


def request_policy_metrics(
    refs: dict[str, torch.Tensor],
    signal: torch.Tensor,
    threshold: float,
    target_mode: str,
) -> dict[str, float]:
    request = signal >= threshold
    oracle_correct = torch.where(request, refs["oracle_correct"], refs["first_correct"])
    oracle_score = torch.where(request, refs["oracle_score"], refs["first_score"])
    fuser_correct = torch.where(request, refs["fuser_correct"], refs["first_correct"])
    fuser_score = torch.where(request, refs["fuser_score"], refs["first_score"])
    requested = request.bool()
    helpful = refs[f"{target_mode}_helpful"]
    return {
        "request_rate": tensor_mean(request.float()),
        "avg_sender_budget": 1.0 + 2.0 * tensor_mean(request.float()),
        "helpful_precision": tensor_mean(helpful[requested]) if requested.any() else 0.0,
        "oracle_after_request_accuracy": tensor_mean(oracle_correct),
        "oracle_after_request_score": tensor_mean(oracle_score),
        "fuser_after_request_accuracy": tensor_mean(fuser_correct),
        "fuser_after_request_score": tensor_mean(fuser_score),
        "mean_score_gain_vs_first_oracle": tensor_mean(oracle_score - refs["first_score"]),
        "mean_score_gain_vs_first_fuser": tensor_mean(fuser_score - refs["first_score"]),
    }


def choose_threshold_for_rate(signal: torch.Tensor, target_rate: float) -> float:
    values = torch.sort(signal.float(), descending=True).values
    if values.numel() == 0:
        return 1.0
    k = max(1, min(values.numel(), int(round(target_rate * values.numel()))))
    return float(values[k - 1].item())


def choose_threshold_for_helpful_precision(signal: torch.Tensor, helpful: torch.Tensor, target_precision: float) -> float:
    thresholds = torch.sort(torch.unique(signal.float()), descending=True).values
    best_threshold = 1.0
    best_rate = -1.0
    for threshold in thresholds:
        request = signal >= threshold
        if not request.any():
            continue
        precision = tensor_mean(helpful[request])
        rate = tensor_mean(request.float())
        if precision >= target_precision and rate > best_rate:
            best_threshold = float(threshold.item())
            best_rate = rate
    return best_threshold


def write_outputs(
    output_dir: Path,
    config: RequestMorePolicyConfig,
    policy_rows: list[dict[str, float | str]],
    predictions: dict[str, Any],
) -> dict[str, str]:
    policy_path = output_dir / "policy_metrics.csv"
    write_csv(policy_path, policy_rows)
    valid_predictions_path = output_dir / "valid_predictions.jsonl"
    test_predictions_path = output_dir / "test_predictions.jsonl"
    write_predictions(valid_predictions_path, predictions["_valid_prob"], predictions["_valid_gain_pred"])
    write_predictions(test_predictions_path, predictions["_test_prob"], predictions["_test_gain_pred"])
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "policy_metrics_csv": str(policy_path),
        "valid_predictions_jsonl": str(valid_predictions_path),
        "test_predictions_jsonl": str(test_predictions_path),
        "config_json": str(config_path),
    }


def write_predictions(path: Path, prob: torch.Tensor, gain_pred: torch.Tensor) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for idx, (p, g) in enumerate(zip(prob.tolist(), gain_pred.tolist())):
            handle.write(json.dumps({"index": idx, "request_prob": float(p), "gain_pred": float(g)}, sort_keys=True) + "\n")


def save_checkpoint(
    path: Path,
    model: FirstSenderRequestPolicy,
    optimizer: torch.optim.Optimizer,
    config: RequestMorePolicyConfig,
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


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
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


def tensor_mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.float().mean().item())


if __name__ == "__main__":
    main()
