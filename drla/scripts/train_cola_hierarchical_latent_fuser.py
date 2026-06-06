"""Train a P2-E hierarchical latent fuser over three locked sender packets.

The fuser reads only sanitized latent/process/certificate packet fields from
the three same-sample senders and learns to select the sender with the best
offline answer utility.  Decoded answers and official scores are
labels/evaluation references only; they are not model inputs.
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

from drla.scripts.audit_cola_agent_latent_packet_distribution import ShardCache, load_packet_blocks
from drla.scripts.audit_cola_hierarchical_aggregation_potential import (
    build_groups,
    choose_text_majority_selected,
)
from drla.scripts.audit_cola_sequential_latent_mas import load_official_scorer, score_text_with_official_rules
from drla.scripts.train_cola_latent_receiver import (
    OFFICIAL_COLA_TASKS,
    certificate_feature_dim,
    certificate_feature_values,
    packet_block_mask,
    packet_process_tensor,
    stable_uniform,
)
from drla.scripts.train_cola_readiness_model import device_metadata, require_cuda_training, resolve_device
from drla.tracking import finish_experiment, init_experiment, log_metrics


EXACT_UTILITY_TASKS = {"lambada", "mmlu", "obqa", "race", "siqa"}


@dataclass(frozen=True)
class HierarchicalLatentFuserConfig:
    packets_jsonl: str = (
        "/data1/luyifei/drla/outputs/cola_agent_latent_comm/"
        "p2_agent_latent_comm_v2_locked_seed66_67_68_split20260601_20260529/"
        "agent_latent_comm_packets_test.jsonl"
    )
    output_dir: str = (
        "/data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/"
        "p2e_hierarchical_latent_fuser_v1"
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
    target_mode: str = "correct"
    task_loss_weighting: str = "none"
    rank_loss_weight: float = 0.5
    valid_interval: int = 10
    max_cached_shards: int = 1024
    num_workers: int = 0
    device: str = "auto"
    swanlab_mode: str = "cloud"
    experiment_name: str = "p2e-hierarchical-latent-fuser-v1"


class HierarchicalLatentFuser(nn.Module):
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
        if d_model % attention_heads != 0:
            raise ValueError("d_model must be divisible by attention_heads")
        self.max_blocks = max_blocks
        self.block_size = block_size
        self.sender_count = sender_count
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.latent_adapter = nn.Linear(latent_dim, d_model)
        self.slot_pos = nn.Embedding(block_size, d_model)
        self.block_pos = nn.Embedding(max_blocks, d_model)
        self.intra_block = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=attention_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=inter_layers,
        )
        self.inter_block = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=attention_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=inter_layers,
        )
        self.latent_query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.latent_attn = nn.MultiheadAttention(d_model, attention_heads, dropout=dropout, batch_first=True)
        self.process_mlp = nn.Sequential(
            nn.LayerNorm(process_dim),
            nn.Linear(process_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.certificate_mlp = nn.Sequential(
            nn.LayerNorm(certificate_dim),
            nn.Linear(certificate_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.task_embedding = nn.Embedding(task_count, d_model)
        self.sender_pos = nn.Embedding(sender_count, d_model)
        self.sender_in = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.sender_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=attention_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=sender_layers,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        latent_blocks: torch.Tensor,
        process_features: torch.Tensor,
        block_mask: torch.Tensor,
        certificate_features: torch.Tensor,
        task_idx: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(
            self.encode_senders(
                latent_blocks=latent_blocks,
                process_features=process_features,
                block_mask=block_mask,
                certificate_features=certificate_features,
                task_idx=task_idx,
            )
        ).squeeze(-1)

    def encode_senders(
        self,
        *,
        latent_blocks: torch.Tensor,
        process_features: torch.Tensor,
        block_mask: torch.Tensor,
        certificate_features: torch.Tensor,
        task_idx: torch.Tensor,
    ) -> torch.Tensor:
        batch, senders, max_blocks, block_size, latent_dim = latent_blocks.shape
        flat_latent = latent_blocks.reshape(batch * senders, max_blocks, block_size, latent_dim)
        flat_process = process_features.reshape(batch * senders, max_blocks, -1)
        flat_mask = block_mask.reshape(batch * senders, max_blocks)
        flat_cert = certificate_features.reshape(batch * senders, -1)
        latent_state = self.encode_latent(flat_latent, flat_mask)
        process_state = masked_mean(self.process_mlp(flat_process), flat_mask)
        cert_state = self.certificate_mlp(flat_cert)
        task_state = self.task_embedding(task_idx).unsqueeze(1).expand(batch, senders, -1).reshape(batch * senders, -1)
        sender_state = self.sender_in(torch.cat([latent_state, process_state, cert_state, task_state], dim=-1))
        sender_state = sender_state.reshape(batch, senders, -1)
        sender_ids = torch.arange(senders, device=sender_state.device).view(1, senders)
        sender_state = sender_state + self.sender_pos(sender_ids)
        sender_state = self.sender_encoder(sender_state)
        return sender_state

    def encode_latent(self, latent_blocks: torch.Tensor, block_mask: torch.Tensor) -> torch.Tensor:
        batch, max_blocks, block_size, _ = latent_blocks.shape
        device = latent_blocks.device
        slot_pos = self.slot_pos(torch.arange(block_size, device=device)).view(1, 1, block_size, -1)
        block_pos = self.block_pos(torch.arange(max_blocks, device=device)).view(1, max_blocks, 1, -1)
        tokens = self.latent_adapter(self.latent_norm(latent_blocks)) + slot_pos + block_pos
        encoded = self.intra_block(tokens.reshape(batch * max_blocks, block_size, -1))
        block_summary = encoded.reshape(batch, max_blocks, block_size, -1).mean(dim=2)
        block_summary = self.inter_block(block_summary, src_key_padding_mask=~block_mask.bool())
        query = self.latent_query.unsqueeze(0).expand(batch, -1, -1)
        pooled, _ = self.latent_attn(
            query,
            block_summary,
            block_summary,
            key_padding_mask=~block_mask.bool(),
            need_weights=False,
        )
        return pooled.squeeze(1)


def main() -> None:
    summary = train_hierarchical_latent_fuser(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> HierarchicalLatentFuserConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets-jsonl", default=HierarchicalLatentFuserConfig.packets_jsonl)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default=HierarchicalLatentFuserConfig.data_root)
    parser.add_argument("--acc-calc-script", default=HierarchicalLatentFuserConfig.acc_calc_script)
    parser.add_argument("--seed", type=int, default=HierarchicalLatentFuserConfig.seed)
    parser.add_argument("--train-ratio", type=float, default=HierarchicalLatentFuserConfig.train_ratio)
    parser.add_argument("--valid-ratio", type=float, default=HierarchicalLatentFuserConfig.valid_ratio)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=HierarchicalLatentFuserConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=HierarchicalLatentFuserConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=HierarchicalLatentFuserConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=HierarchicalLatentFuserConfig.weight_decay)
    parser.add_argument("--dropout", type=float, default=HierarchicalLatentFuserConfig.dropout)
    parser.add_argument("--d-model", type=int, default=HierarchicalLatentFuserConfig.d_model)
    parser.add_argument("--attention-heads", type=int, default=HierarchicalLatentFuserConfig.attention_heads)
    parser.add_argument("--inter-layers", type=int, default=HierarchicalLatentFuserConfig.inter_layers)
    parser.add_argument("--sender-layers", type=int, default=HierarchicalLatentFuserConfig.sender_layers)
    parser.add_argument(
        "--target-mode",
        choices=["correct", "score", "task_aware_score"],
        default=HierarchicalLatentFuserConfig.target_mode,
    )
    parser.add_argument("--task-loss-weighting", choices=["none", "balanced"], default=HierarchicalLatentFuserConfig.task_loss_weighting)
    parser.add_argument("--rank-loss-weight", type=float, default=HierarchicalLatentFuserConfig.rank_loss_weight)
    parser.add_argument("--valid-interval", type=int, default=HierarchicalLatentFuserConfig.valid_interval)
    parser.add_argument("--max-cached-shards", type=int, default=HierarchicalLatentFuserConfig.max_cached_shards)
    parser.add_argument("--num-workers", type=int, default=HierarchicalLatentFuserConfig.num_workers)
    parser.add_argument("--device", default=HierarchicalLatentFuserConfig.device)
    parser.add_argument("--swanlab-mode", default=HierarchicalLatentFuserConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=HierarchicalLatentFuserConfig.experiment_name)
    args = parser.parse_args()
    return HierarchicalLatentFuserConfig(
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
        target_mode=args.target_mode,
        task_loss_weighting=args.task_loss_weighting,
        rank_loss_weight=args.rank_loss_weight,
        valid_interval=args.valid_interval,
        max_cached_shards=args.max_cached_shards,
        num_workers=args.num_workers,
        device=args.device,
        swanlab_mode=args.swanlab_mode,
        experiment_name=args.experiment_name,
    )


def train_hierarchical_latent_fuser(config: HierarchicalLatentFuserConfig) -> dict[str, Any]:
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
    train_ds, valid_ds, test_ds, norm_stats = make_datasets(tensors_by_split, config)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    device = resolve_device(config.device)
    require_cuda_training(device, "train_cola_hierarchical_latent_fuser.py")
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    labels = tensors_by_split["train"]["labels"]
    pos = labels.sum().clamp_min(1.0)
    neg = labels.numel() - pos
    pos_weight = (neg / pos).to(device)
    run = init_experiment(
        stage="p2e-hierarchical-latent-fuser",
        config={**asdict(config), **device_metadata(device), "pos_weight": float(pos_weight.detach().cpu())},
        experiment_name=config.experiment_name,
        tags=["cola", "official-benchmark", "p2e", "hierarchical", "latent-fuser"],
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
                logits = model(*batch[:5])
                loss, train_metrics = compute_loss(
                    logits=logits,
                    labels=batch[5],
                    target_scores=batch[6],
                    train_targets=batch[7],
                    group_weight=batch[8],
                    pos_weight=pos_weight,
                    rank_loss_weight=config.rank_loss_weight,
                    target_mode=config.target_mode,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_metrics = {"loss": float(loss.detach().item()), **train_metrics}
                write_metrics(metrics_f, "train", global_step, train_metrics)
                log_metrics(train_metrics, step=global_step, prefix="train")
                if global_step % config.valid_interval == 0:
                    valid_metrics = evaluate(model, valid_loader, tensors_by_split["valid"], groups, splits["valid"], device, scorer)
                    write_metrics(metrics_f, "valid", global_step, valid_metrics)
                    log_metrics(valid_metrics, step=global_step, prefix="valid")
                    current = selection_metric(valid_metrics, config.target_mode)
                    if current > best_metric:
                        best_metric = current
                        best_step = global_step
                        save_checkpoint(checkpoint_dir / "best_checkpoint.pt", model, optimizer, config, metadata, norm_stats, best_step, best_metric)

        valid_metrics = evaluate(model, valid_loader, tensors_by_split["valid"], groups, splits["valid"], device, scorer)
        test_metrics = evaluate(model, test_loader, tensors_by_split["test"], groups, splits["test"], device, scorer)
        write_metrics(metrics_f, "valid", global_step, valid_metrics)
        write_metrics(metrics_f, "test", global_step, test_metrics)
        log_metrics(valid_metrics, step=global_step, prefix="valid")
        log_metrics(test_metrics, step=global_step, prefix="test")
        if selection_metric(valid_metrics, config.target_mode) > best_metric:
            best_metric = selection_metric(valid_metrics, config.target_mode)
            best_step = global_step
            save_checkpoint(checkpoint_dir / "best_checkpoint.pt", model, optimizer, config, metadata, norm_stats, best_step, best_metric)
        save_checkpoint(checkpoint_dir / "last_checkpoint.pt", model, optimizer, config, metadata, norm_stats, global_step, selection_metric(valid_metrics, config.target_mode))
    finally:
        metrics_f.close()
        finish_experiment()

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
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "metrics_jsonl": str(metrics_path),
            "best_checkpoint": str(checkpoint_dir / "best_checkpoint.pt"),
            "last_checkpoint": str(checkpoint_dir / "last_checkpoint.pt"),
        },
        "interpretation": (
            "P2-E hierarchical latent fuser. The model selects among same-sample latent "
            "sender packets using decoder-free inputs; decoded selected answers are used "
            "only to evaluate the selected sender."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def validate_config(config: HierarchicalLatentFuserConfig) -> None:
    if config.valid_interval > 10:
        raise ValueError("valid_interval must be <= 10 steps")
    if config.swanlab_mode != "cloud":
        raise ValueError("training must use SwanLab cloud; pass --swanlab-mode cloud")
    if config.target_mode not in {"correct", "score", "task_aware_score"}:
        raise ValueError("target_mode must be correct, score, or task_aware_score")
    if config.task_loss_weighting not in {"none", "balanced"}:
        raise ValueError("task_loss_weighting must be none or balanced")
    if not 0.0 < config.train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0.0 <= config.valid_ratio < 1.0:
        raise ValueError("valid_ratio must be in [0, 1)")
    if config.train_ratio + config.valid_ratio >= 1.0:
        raise ValueError("train_ratio + valid_ratio must be < 1")


def split_groups(groups: list[dict[str, Any]], config: HierarchicalLatentFuserConfig) -> dict[str, list[int]]:
    splits = {"train": [], "valid": [], "test": []}
    for index, group in enumerate(groups):
        value = stable_uniform(f"{config.seed}:{group['sample_key']}")
        if value < config.train_ratio:
            splits["train"].append(index)
        elif value < config.train_ratio + config.valid_ratio:
            splits["valid"].append(index)
        else:
            splits["test"].append(index)
    for split, indices in splits.items():
        if not indices:
            raise ValueError(f"empty split: {split}")
    return splits


def build_tensors(
    groups: list[dict[str, Any]],
    splits: dict[str, list[int]],
    config: HierarchicalLatentFuserConfig,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    first_packet = groups[0]["members"][0]["packet"]
    first_block = first_packet["latent_memory"]["blocks"][0]
    max_blocks = max(int(member["packet"]["agent_a"]["max_block_budget"]) for group in groups for member in group["members"])
    sender_count = max(len(group["members"]) for group in groups)
    block_size = int(first_block["latent_ref"]["shape"][0])
    latent_dim = int(first_block["latent_ref"]["shape"][1])
    process_dim = 11
    cert_dim = certificate_feature_dim()
    task_to_idx = {task: idx for idx, task in enumerate(OFFICIAL_COLA_TASKS)}
    shard_cache = ShardCache(config.max_cached_shards)
    tensors = {}
    for split, indices in splits.items():
        tensors[split] = build_split_tensors(
            [groups[index] for index in indices],
            task_to_idx=task_to_idx,
            max_blocks=max_blocks,
            sender_count=sender_count,
            block_size=block_size,
            latent_dim=latent_dim,
            process_dim=process_dim,
            cert_dim=cert_dim,
            shard_cache=shard_cache,
        )
    metadata = {
        "max_blocks": max_blocks,
        "sender_count": sender_count,
        "block_size": block_size,
        "latent_dim": latent_dim,
        "process_dim": process_dim,
        "certificate_dim": cert_dim,
        "task_to_idx": task_to_idx,
    }
    return tensors, metadata


def build_split_tensors(
    groups: list[dict[str, Any]],
    *,
    task_to_idx: dict[str, int],
    max_blocks: int,
    sender_count: int,
    block_size: int,
    latent_dim: int,
    process_dim: int,
    cert_dim: int,
    shard_cache: ShardCache,
) -> dict[str, torch.Tensor]:
    count = len(groups)
    latent = torch.zeros(count, sender_count, max_blocks, block_size, latent_dim, dtype=torch.float32)
    process = torch.zeros(count, sender_count, max_blocks, process_dim, dtype=torch.float32)
    mask = torch.zeros(count, sender_count, max_blocks, dtype=torch.bool)
    cert = torch.zeros(count, sender_count, cert_dim, dtype=torch.float32)
    task_idx = torch.zeros(count, dtype=torch.long)
    labels = torch.zeros(count, sender_count, dtype=torch.float32)
    target_scores = torch.zeros(count, sender_count, dtype=torch.float32)
    train_targets = torch.zeros(count, sender_count, dtype=torch.float32)
    for i, group in enumerate(groups):
        task = str(group["task"])
        task_idx[i] = task_to_idx[task]
        for j, member in enumerate(group["members"]):
            packet = member["packet"]
            for block_idx, block in enumerate(load_packet_blocks(packet, shard_cache)):
                latent[i, j, block_idx] = block
            process[i, j] = packet_process_tensor(packet, max_blocks, process_dim)
            mask[i, j] = packet_block_mask(packet, max_blocks)
            cert[i, j] = torch.tensor(certificate_feature_values(packet), dtype=torch.float32)
            labels[i, j] = float(bool(member["selected_correct"]))
            target_scores[i, j] = float(member.get("selected_score", labels[i, j].item()))
            train_targets[i, j] = labels[i, j] if task in EXACT_UTILITY_TASKS else target_scores[i, j]
    return {
        "latent_blocks": latent,
        "process_features": process,
        "block_mask": mask,
        "certificate_features": cert,
        "task_idx": task_idx,
        "labels": labels,
        "target_scores": target_scores,
        "train_targets": train_targets,
    }


def make_datasets(
    tensors_by_split: dict[str, dict[str, torch.Tensor]],
    config: HierarchicalLatentFuserConfig | None = None,
) -> tuple[TensorDataset, TensorDataset, TensorDataset, dict[str, torch.Tensor]]:
    train = tensors_by_split["train"]
    active_process = train["process_features"][train["block_mask"]]
    process_mean = active_process.mean(dim=0, keepdim=True)
    process_std = active_process.std(dim=0, keepdim=True).clamp_min(1e-6)
    flat_cert = train["certificate_features"].reshape(-1, train["certificate_features"].shape[-1])
    cert_mean = flat_cert.mean(dim=0, keepdim=True)
    cert_std = flat_cert.std(dim=0, keepdim=True).clamp_min(1e-6)
    norm_stats = {
        "process_mean": process_mean,
        "process_std": process_std,
        "certificate_mean": cert_mean,
        "certificate_std": cert_std,
    }
    task_weights = torch.ones(len(OFFICIAL_COLA_TASKS), dtype=torch.float32)
    if config is not None and config.task_loss_weighting == "balanced":
        counts = torch.bincount(train["task_idx"], minlength=len(OFFICIAL_COLA_TASKS)).float()
        nonzero = counts > 0
        task_weights[nonzero] = counts[nonzero].sum() / (nonzero.float().sum() * counts[nonzero])
    norm_stats["task_loss_weights"] = task_weights

    def dataset(split: str) -> TensorDataset:
        tensors = tensors_by_split[split]
        process = (tensors["process_features"] - process_mean.view(1, 1, 1, -1)) / process_std.view(1, 1, 1, -1)
        process = process.masked_fill(~tensors["block_mask"].unsqueeze(-1), 0.0)
        cert = (tensors["certificate_features"] - cert_mean.view(1, 1, -1)) / cert_std.view(1, 1, -1)
        return TensorDataset(
            tensors["latent_blocks"],
            process,
            tensors["block_mask"],
            cert,
            tensors["task_idx"],
            tensors["labels"],
            tensors["target_scores"],
            tensors["train_targets"],
            task_weights[tensors["task_idx"]],
        )

    return dataset("train"), dataset("valid"), dataset("test"), norm_stats


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_scores: torch.Tensor,
    train_targets: torch.Tensor,
    group_weight: torch.Tensor,
    pos_weight: torch.Tensor,
    rank_loss_weight: float,
    target_mode: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    if target_mode in {"score", "task_aware_score"}:
        target_values = train_targets if target_mode == "task_aware_score" else target_scores
        pointwise_per_group = (torch.sigmoid(logits) - target_values).square().mean(dim=1)
        pointwise_loss = weighted_mean(pointwise_per_group, group_weight)
        ranking_values = target_values
    else:
        pointwise = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none")
        pointwise_loss = weighted_mean(pointwise.mean(dim=1), group_weight)
        ranking_values = labels
    has_signal = ranking_values.sum(dim=1) > 0
    if has_signal.any():
        positive = ranking_values[has_signal]
        target = positive / positive.sum(dim=1, keepdim=True).clamp_min(1e-6)
        rank_per_group = -(target * F.log_softmax(logits[has_signal], dim=1)).sum(dim=1)
        rank_loss = weighted_mean(rank_per_group, group_weight[has_signal])
    else:
        rank_loss = torch.zeros((), device=logits.device)
    loss = pointwise_loss + rank_loss_weight * rank_loss
    with torch.no_grad():
        selected = logits.argmax(dim=1)
        selected_correct = labels[torch.arange(labels.shape[0], device=labels.device), selected].mean()
        selected_score = target_scores[torch.arange(target_scores.shape[0], device=target_scores.device), selected].mean()
        selected_train_target = train_targets[torch.arange(train_targets.shape[0], device=train_targets.device), selected].mean()
        any_correct = (labels.sum(dim=1) > 0).float().mean()
        best_score = target_scores.max(dim=1).values.mean()
        best_train_target = train_targets.max(dim=1).values.mean()
    return loss, {
        "pointwise_loss": float(pointwise_loss.detach().item()),
        "rank_loss": float(rank_loss.detach().item()),
        "batch_model_selected_accuracy": float(selected_correct.detach().item()),
        "batch_model_selected_score": float(selected_score.detach().item()),
        "batch_model_selected_train_target": float(selected_train_target.detach().item()),
        "batch_oracle_any_selected_accuracy": float(any_correct.detach().item()),
        "batch_oracle_best_selected_score": float(best_score.detach().item()),
        "batch_oracle_best_train_target": float(best_train_target.detach().item()),
        "batch_group_weight_mean": float(group_weight.mean().detach().item()),
    }


@torch.no_grad()
def evaluate(
    model: HierarchicalLatentFuser,
    loader: DataLoader,
    tensors: dict[str, torch.Tensor],
    groups: list[dict[str, Any]],
    indices: list[int],
    device: torch.device,
    scorer: Any,
) -> dict[str, float]:
    model.eval()
    logits_all = []
    for batch in loader:
        batch = [item.to(device) for item in batch]
        logits_all.append(model(*batch[:5]).cpu())
    logits = torch.cat(logits_all, dim=0)
    labels = tensors["labels"]
    target_scores = tensors["target_scores"]
    selected = logits.argmax(dim=1)
    row_ids = torch.arange(labels.shape[0])
    model_correct = labels[row_ids, selected].float()
    selected_target_score = target_scores[row_ids, selected].float()
    first_correct = labels[:, 0].float()
    first_score = target_scores[:, 0].float()
    oracle_any = (labels.sum(dim=1) > 0).float()
    oracle_best_score = target_scores.max(dim=1).values.float()
    text_majority_correct = []
    text_majority_scores = []
    model_scores = []
    for local_idx, group_index in enumerate(indices):
        group = groups[group_index]
        member = group["members"][int(selected[local_idx])]
        score = score_text_with_official_rules(
            task=str(group["task"]),
            text=str(member["selected_prediction"]),
            ground_truth=group["ground_truth"],
            choices=group["choices"],
            scorer=scorer,
        )
        model_scores.append(float(score["score"]))
        chosen = choose_text_majority_selected(group["members"])
        text_score = score_text_with_official_rules(
            task=str(group["task"]),
            text=str(chosen["prediction"]),
            ground_truth=group["ground_truth"],
            choices=group["choices"],
            scorer=scorer,
        )
        text_majority_correct.append(float(bool(text_score["correct"])))
        text_majority_scores.append(float(text_score["score"]))
    text_majority = torch.tensor(text_majority_correct, dtype=torch.float32)
    text_majority_score = torch.tensor(text_majority_scores, dtype=torch.float32)
    model_score = torch.tensor(model_scores, dtype=torch.float32)
    any_mask = oracle_any.bool()
    selected_in_any = model_correct[any_mask].mean() if any_mask.any() else torch.tensor(0.0)
    return {
        "model_selected_accuracy": float(model_correct.mean().item()),
        "model_mean_official_score": float(model_score.mean().item()),
        "model_mean_target_score": float(selected_target_score.mean().item()),
        "single_sender_first_accuracy": float(first_correct.mean().item()),
        "single_sender_first_mean_official_score": float(first_score.mean().item()),
        "text_majority_selected_accuracy": float(text_majority.mean().item()),
        "text_majority_mean_official_score": float(text_majority_score.mean().item()),
        "oracle_any_selected_accuracy": float(oracle_any.mean().item()),
        "oracle_best_selected_mean_official_score": float(oracle_best_score.mean().item()),
        "model_fraction_of_oracle_gap_closed": safe_gap_fraction(model_correct.mean(), first_correct.mean(), oracle_any.mean()),
        "model_fraction_of_oracle_score_gap_closed": safe_gap_fraction(model_score.mean(), first_score.mean(), oracle_best_score.mean()),
        "model_selects_correct_when_any_correct": float(selected_in_any.item()),
        "num_groups": float(labels.shape[0]),
    }


def safe_gap_fraction(model_acc: torch.Tensor, base_acc: torch.Tensor, oracle_acc: torch.Tensor) -> float:
    denom = float((oracle_acc - base_acc).item())
    if denom <= 0:
        return 0.0
    return float(((model_acc - base_acc) / (oracle_acc - base_acc)).item())


def selection_metric(metrics: dict[str, float], target_mode: str) -> float:
    if target_mode in {"score", "task_aware_score"}:
        return float(metrics["model_mean_official_score"])
    return float(metrics["model_selected_accuracy"])


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float().unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def save_checkpoint(
    path: Path,
    model: HierarchicalLatentFuser,
    optimizer: torch.optim.Optimizer,
    config: HierarchicalLatentFuserConfig,
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    main()
