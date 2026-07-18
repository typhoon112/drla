"""Train D7 Dream latent fuser with contrastive receiver alignment.

V1 MSE prefix distillation can learn an average solver state. This V2 objective
instead aligns upstream evidence-agent packets with the same row's solver latent
state via symmetric InfoNCE. It is a deep-learning training script and must use
CUDA + SwanLab cloud.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.scripts.p3_train_dream_latent_fuser import (  # noqa: E402
    DEFAULT_PACKET_DIR,
    DEFAULT_TRACE_DIR,
    load_rows,
    load_tensor,
    select_evenly_spaced,
    split_rows,
)
from drla.scripts.train_cola_readiness_model import device_metadata, require_cuda_training, resolve_device  # noqa: E402
from drla.tracking import finish_experiment, init_experiment, log_metrics  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_latent_fusers/"
    "dream_latent_fuser_v2_contrastive_textmas_matched200_seed20260606"
)


@dataclass(frozen=True)
class ContrastiveConfig:
    packet_dir: str = DEFAULT_PACKET_DIR
    trace_dir: str = DEFAULT_TRACE_DIR
    output_dir: str = DEFAULT_OUTPUT_DIR
    device: str = "auto"
    seed: int = 20260606
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    batch_size: int = 16
    epochs: int = 80
    max_train_steps: int = 0
    valid_interval: int = 10
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    input_tokens_per_agent: int = 32
    target_tokens: int = 32
    hidden_size: int = 3584
    d_model: int = 256
    embed_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    temperature: float = 0.07
    swanlab_mode: str = "cloud"
    experiment_name: str = "p3-dream-latent-fuser-v2-contrastive-textmas-matched200"


class ContrastiveLatentDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], config: ContrastiveConfig) -> None:
        self.rows = rows
        self.config = config

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        agent_tensors = []
        for agent_id in ["agent_a", "agent_b"]:
            tensor = load_tensor(row["agent_hidden_refs"][agent_id])
            agent_tensors.append(select_evenly_spaced(tensor, self.config.input_tokens_per_agent))
        target = select_evenly_spaced(load_tensor(row["solver_hidden_ref"]), self.config.target_tokens)
        return {
            "packets": torch.stack(agent_tensors, dim=0).to(torch.float32),
            "target": target.to(torch.float32),
            "sample_id": row["sample_id"],
            "row_id": row["row_id"],
        }


class PacketEncoder(nn.Module):
    def __init__(self, config: ContrastiveConfig) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(config.hidden_size)
        self.input_proj = nn.Linear(config.hidden_size, config.d_model)
        self.agent_embed = nn.Embedding(2, config.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 2 * config.input_tokens_per_agent, config.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.query = nn.Parameter(torch.randn(1, config.d_model) / math.sqrt(config.d_model))
        self.attn = nn.MultiheadAttention(config.d_model, config.num_heads, dropout=config.dropout, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, config.embed_dim))

    def forward(self, packets: torch.Tensor) -> torch.Tensor:
        batch, num_agents, tokens, hidden = packets.shape
        x = self.input_proj(self.input_norm(packets.reshape(batch, num_agents * tokens, hidden)))
        agent_ids = torch.arange(num_agents, device=packets.device).repeat_interleave(tokens)
        x = x + self.agent_embed(agent_ids).unsqueeze(0) + self.pos_embed[:, : num_agents * tokens]
        memory = self.encoder(x)
        query = self.query.unsqueeze(0).expand(batch, -1, -1)
        pooled, _ = self.attn(query, memory, memory, need_weights=False)
        return F.normalize(self.out(pooled[:, 0]), dim=-1)


class TargetEncoder(nn.Module):
    def __init__(self, config: ContrastiveConfig) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(config.hidden_size)
        self.input_proj = nn.Linear(config.hidden_size, config.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.target_tokens, config.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.out = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, config.embed_dim))

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(self.input_norm(target)) + self.pos_embed[:, : target.shape[1]]
        encoded = self.encoder(x)
        return F.normalize(self.out(encoded.mean(dim=1)), dim=-1)


class ContrastiveLatentFuser(nn.Module):
    def __init__(self, config: ContrastiveConfig) -> None:
        super().__init__()
        self.packet_encoder = PacketEncoder(config)
        self.target_encoder = TargetEncoder(config)
        self.temperature = config.temperature

    def forward(self, packets: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        packet_emb = self.packet_encoder(packets)
        target_emb = self.target_encoder(target)
        logits = packet_emb @ target_emb.T / self.temperature
        return packet_emb, target_emb, logits


def main() -> None:
    summary = train(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> ContrastiveConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", default=ContrastiveConfig.packet_dir)
    parser.add_argument("--trace-dir", default=ContrastiveConfig.trace_dir)
    parser.add_argument("--output-dir", default=ContrastiveConfig.output_dir)
    parser.add_argument("--device", default=ContrastiveConfig.device)
    parser.add_argument("--seed", type=int, default=ContrastiveConfig.seed)
    parser.add_argument("--train-ratio", type=float, default=ContrastiveConfig.train_ratio)
    parser.add_argument("--valid-ratio", type=float, default=ContrastiveConfig.valid_ratio)
    parser.add_argument("--batch-size", type=int, default=ContrastiveConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=ContrastiveConfig.epochs)
    parser.add_argument("--max-train-steps", type=int, default=ContrastiveConfig.max_train_steps)
    parser.add_argument("--valid-interval", type=int, default=ContrastiveConfig.valid_interval)
    parser.add_argument("--learning-rate", type=float, default=ContrastiveConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=ContrastiveConfig.weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=ContrastiveConfig.grad_clip_norm)
    parser.add_argument("--input-tokens-per-agent", type=int, default=ContrastiveConfig.input_tokens_per_agent)
    parser.add_argument("--target-tokens", type=int, default=ContrastiveConfig.target_tokens)
    parser.add_argument("--hidden-size", type=int, default=ContrastiveConfig.hidden_size)
    parser.add_argument("--d-model", type=int, default=ContrastiveConfig.d_model)
    parser.add_argument("--embed-dim", type=int, default=ContrastiveConfig.embed_dim)
    parser.add_argument("--num-layers", type=int, default=ContrastiveConfig.num_layers)
    parser.add_argument("--num-heads", type=int, default=ContrastiveConfig.num_heads)
    parser.add_argument("--dropout", type=float, default=ContrastiveConfig.dropout)
    parser.add_argument("--temperature", type=float, default=ContrastiveConfig.temperature)
    parser.add_argument("--swanlab-mode", default=ContrastiveConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=ContrastiveConfig.experiment_name)
    return ContrastiveConfig(**vars(parser.parse_args()))


def train(config: ContrastiveConfig) -> dict[str, Any]:
    if config.swanlab_mode != "cloud":
        raise ValueError("D7 contrastive fuser training must use SwanLab cloud")
    if config.valid_interval > 10:
        raise ValueError("valid_interval must be <= 10 for current P3 training")
    set_seed(config.seed)
    device = resolve_device(config.device)
    require_cuda_training(device, "p3_train_dream_latent_fuser_contrastive.py")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, metadata = load_rows_for_config(config)
    splits = split_rows(rows, config.seed, config.train_ratio, config.valid_ratio)
    datasets = {name: ContrastiveLatentDataset(items, config) for name, items in splits.items()}
    loaders = {
        name: DataLoader(dataset, batch_size=config.batch_size, shuffle=(name == "train"), collate_fn=collate_batch)
        for name, dataset in datasets.items()
    }
    model = ContrastiveLatentFuser(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    run = init_experiment(
        stage="p3-dream-latent-fuser-contrastive",
        config={
            **asdict(config),
            **device_metadata(device),
            "metadata": metadata,
            "split_sizes": {name: len(items) for name, items in splits.items()},
        },
        experiment_name=config.experiment_name,
        tags=["dream", "p3", "latentmas", "latent-fuser", "contrastive", "swanlab-cloud"],
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
                _, _, logits = model(batch["packets"], batch["target"])
                loss, train_metrics = contrastive_loss(logits)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                optimizer.step()
                train_metrics = {"loss": float(loss.detach().item()), **train_metrics, "epoch": float(epoch)}
                write_metrics(metrics_f, "train", global_step, train_metrics)
                log_metrics(train_metrics, step=global_step, prefix="train")
                if global_step % config.valid_interval == 0:
                    valid_metrics = evaluate(model, loaders["valid"], device)
                    write_metrics(metrics_f, "valid", global_step, valid_metrics)
                    log_metrics(valid_metrics, step=global_step, prefix="valid")
                    if valid_metrics["selection_metric"] > best_metric:
                        best_metric = valid_metrics["selection_metric"]
                        best_step = global_step
                        save_checkpoint(output_dir / "best_checkpoint.pt", model, optimizer, config, metadata, best_step, best_metric)
                if config.max_train_steps and global_step >= config.max_train_steps:
                    break
            if config.max_train_steps and global_step >= config.max_train_steps:
                break
        final_valid_metrics = evaluate(model, loaders["valid"], device)
        final_test_metrics = evaluate(model, loaders["test"], device)
        write_metrics(metrics_f, "valid", global_step, final_valid_metrics)
        write_metrics(metrics_f, "test", global_step, final_test_metrics)
        log_metrics(final_valid_metrics, step=global_step, prefix="valid")
        log_metrics(final_test_metrics, step=global_step, prefix="test")
        if final_valid_metrics["selection_metric"] > best_metric:
            best_metric = final_valid_metrics["selection_metric"]
            best_step = global_step
            save_checkpoint(output_dir / "best_checkpoint.pt", model, optimizer, config, metadata, best_step, best_metric)
        save_checkpoint(output_dir / "last_checkpoint.pt", model, optimizer, config, metadata, global_step, final_valid_metrics["selection_metric"])
    finally:
        metrics_f.close()
        finish_experiment()

    best_valid_metrics = evaluate_checkpoint(output_dir / "best_checkpoint.pt", loaders["valid"], device, config)
    best_test_metrics = evaluate_checkpoint(output_dir / "best_checkpoint.pt", loaders["test"], device, config)
    summary = {
        "created_at": int(time.time()),
        "status": "pass",
        "config": asdict(config),
        "swanlab_run_id": getattr(run, "id", None),
        "metadata": metadata,
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
            "P3 D7 contrastive latent fuser deep-learning training",
            "CUDA/GPU required",
            "SwanLab cloud required",
            "no decoded text/gold/scorer inputs",
            "targets are solver latent tensors from the matched trace",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def load_rows_for_config(config: ContrastiveConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, metadata = load_rows(SimpleNamespace(packet_dir=config.packet_dir, trace_dir=config.trace_dir))
    metadata["objective"] = "symmetric InfoNCE packet-to-solver latent alignment"
    metadata["condition_counts"] = dict(Counter(row["condition"] for row in rows))
    return rows, metadata


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "packets": torch.stack([item["packets"] for item in items], dim=0),
        "target": torch.stack([item["target"] for item in items], dim=0),
    }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def contrastive_loss(logits: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    loss = 0.5 * (loss_i2t + loss_t2i)
    top1_i2t = (logits.argmax(dim=1) == labels).float().mean()
    top1_t2i = (logits.argmax(dim=0) == labels).float().mean()
    diag = logits.diag()
    off_diag = logits[~torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)]
    return loss, {
        "loss_i2t": float(loss_i2t.detach().item()),
        "loss_t2i": float(loss_t2i.detach().item()),
        "top1_i2t": float(top1_i2t.detach().item()),
        "top1_t2i": float(top1_t2i.detach().item()),
        "diag_logit_mean": float(diag.detach().mean().item()),
        "offdiag_logit_mean": float(off_diag.detach().mean().item()) if off_diag.numel() else 0.0,
        "diag_minus_offdiag": float((diag.mean() - off_diag.mean()).detach().item()) if off_diag.numel() else 0.0,
    }


@torch.no_grad()
def evaluate(model: ContrastiveLatentFuser, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses = []
    top1_i2t = []
    top1_t2i = []
    diag_margin = []
    for batch in loader:
        batch = move_batch(batch, device)
        _, _, logits = model(batch["packets"], batch["target"])
        loss, metrics = contrastive_loss(logits)
        losses.append(float(loss.item()))
        top1_i2t.append(metrics["top1_i2t"])
        top1_t2i.append(metrics["top1_t2i"])
        diag_margin.append(metrics["diag_minus_offdiag"])
    metrics = {
        "loss": mean(losses),
        "top1_i2t": mean(top1_i2t),
        "top1_t2i": mean(top1_t2i),
        "diag_minus_offdiag": mean(diag_margin),
    }
    metrics["selection_metric"] = metrics["top1_i2t"] + metrics["top1_t2i"] + 0.01 * metrics["diag_minus_offdiag"] - 0.01 * metrics["loss"]
    return metrics


def evaluate_checkpoint(path: Path, loader: DataLoader, device: torch.device, config: ContrastiveConfig) -> dict[str, float]:
    checkpoint = torch.load(path, map_location=device)
    model = ContrastiveLatentFuser(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return evaluate(model, loader, device)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ContrastiveConfig,
    metadata: dict[str, Any],
    step: int,
    selection_metric: float,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(config),
            "metadata": metadata,
            "step": step,
            "selection_metric": selection_metric,
        },
        path,
    )


def write_metrics(handle, phase: str, step: int, metrics: dict[str, float]) -> None:
    handle.write(json.dumps({"phase": phase, "step": step, **metrics}, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
