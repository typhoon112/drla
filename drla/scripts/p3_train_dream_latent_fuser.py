"""Train D7 Dream latent packet fuser by solver-state distillation.

This is a deep-learning training script. It learns a small receiver/fuser that
maps upstream evidence-agent latent packets to a solver latent prefix. The
Dream model is not called here; decoded text, gold answers, and scorer outputs
are not inputs or targets. Validation includes a shuffled-packet corruption
control to check whether the fuser uses packet identity rather than only
learning an average solver state.
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
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.scripts.train_cola_readiness_model import device_metadata, require_cuda_training, resolve_device  # noqa: E402
from drla.tracking import finish_experiment, init_experiment, log_metrics  # noqa: E402


DEFAULT_PACKET_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_latent_packets/"
    "dream_textmas_matched200_agent_ab_suffix_tensor_packets_v1_20260606"
)
DEFAULT_TRACE_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_traces/"
    "musique_calibration_trace_textmas_matched200_steps64_stride4_hidden_tensor_merged_20260606"
)
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_latent_fusers/"
    "dream_latent_fuser_v1_textmas_matched200_seed20260606"
)


@dataclass(frozen=True)
class FuserConfig:
    packet_dir: str = DEFAULT_PACKET_DIR
    trace_dir: str = DEFAULT_TRACE_DIR
    output_dir: str = DEFAULT_OUTPUT_DIR
    device: str = "auto"
    seed: int = 20260606
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    batch_size: int = 8
    epochs: int = 40
    max_train_steps: int = 0
    valid_interval: int = 10
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    input_tokens_per_agent: int = 32
    prefix_len: int = 16
    hidden_size: int = 3584
    d_model: int = 256
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    cosine_loss_weight: float = 0.25
    swanlab_mode: str = "cloud"
    experiment_name: str = "p3-dream-latent-fuser-v1-textmas-matched200"


class LatentFuserDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], config: FuserConfig) -> None:
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
        target = select_evenly_spaced(load_tensor(row["solver_hidden_ref"]), self.config.prefix_len)
        return {
            "inputs": torch.stack(agent_tensors, dim=0).to(torch.float32),
            "target": target.to(torch.float32),
            "sample_id": row["sample_id"],
            "row_id": row["row_id"],
        }


class DreamLatentFuser(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        d_model: int,
        input_tokens_per_agent: int,
        prefix_len: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden_size)
        self.input_proj = nn.Linear(hidden_size, d_model)
        self.agent_embed = nn.Embedding(2, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 2 * input_tokens_per_agent, d_model))
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
        self.query = nn.Parameter(torch.randn(prefix_len, d_model) / math.sqrt(d_model))
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, num_agents, tokens, hidden = inputs.shape
        x = self.input_proj(self.input_norm(inputs.reshape(batch, num_agents * tokens, hidden)))
        agent_ids = torch.arange(num_agents, device=inputs.device).repeat_interleave(tokens)
        x = x + self.agent_embed(agent_ids).unsqueeze(0) + self.pos_embed[:, : num_agents * tokens]
        memory = self.encoder(x)
        queries = self.query.unsqueeze(0).expand(batch, -1, -1)
        fused, _ = self.cross_attn(queries, memory, memory, need_weights=False)
        return self.output(fused)


def main() -> None:
    summary = train(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> FuserConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", default=FuserConfig.packet_dir)
    parser.add_argument("--trace-dir", default=FuserConfig.trace_dir)
    parser.add_argument("--output-dir", default=FuserConfig.output_dir)
    parser.add_argument("--device", default=FuserConfig.device)
    parser.add_argument("--seed", type=int, default=FuserConfig.seed)
    parser.add_argument("--train-ratio", type=float, default=FuserConfig.train_ratio)
    parser.add_argument("--valid-ratio", type=float, default=FuserConfig.valid_ratio)
    parser.add_argument("--batch-size", type=int, default=FuserConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=FuserConfig.epochs)
    parser.add_argument("--max-train-steps", type=int, default=FuserConfig.max_train_steps)
    parser.add_argument("--valid-interval", type=int, default=FuserConfig.valid_interval)
    parser.add_argument("--learning-rate", type=float, default=FuserConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=FuserConfig.weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=FuserConfig.grad_clip_norm)
    parser.add_argument("--input-tokens-per-agent", type=int, default=FuserConfig.input_tokens_per_agent)
    parser.add_argument("--prefix-len", type=int, default=FuserConfig.prefix_len)
    parser.add_argument("--hidden-size", type=int, default=FuserConfig.hidden_size)
    parser.add_argument("--d-model", type=int, default=FuserConfig.d_model)
    parser.add_argument("--num-layers", type=int, default=FuserConfig.num_layers)
    parser.add_argument("--num-heads", type=int, default=FuserConfig.num_heads)
    parser.add_argument("--dropout", type=float, default=FuserConfig.dropout)
    parser.add_argument("--cosine-loss-weight", type=float, default=FuserConfig.cosine_loss_weight)
    parser.add_argument("--swanlab-mode", default=FuserConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=FuserConfig.experiment_name)
    return FuserConfig(**vars(parser.parse_args()))


def train(config: FuserConfig) -> dict[str, Any]:
    if config.swanlab_mode != "cloud":
        raise ValueError("D7 latent fuser training must use SwanLab cloud")
    if config.valid_interval > 10:
        raise ValueError("valid_interval must be <= 10 for current P3 training")
    set_seed(config.seed)
    device = resolve_device(config.device)
    require_cuda_training(device, "p3_train_dream_latent_fuser.py")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, metadata = load_rows(config)
    splits = split_rows(rows, config.seed, config.train_ratio, config.valid_ratio)
    datasets = {name: LatentFuserDataset(items, config) for name, items in splits.items()}
    loaders = {
        name: DataLoader(dataset, batch_size=config.batch_size, shuffle=(name == "train"), collate_fn=collate_batch)
        for name, dataset in datasets.items()
    }
    model = DreamLatentFuser(
        hidden_size=config.hidden_size,
        d_model=config.d_model,
        input_tokens_per_agent=config.input_tokens_per_agent,
        prefix_len=config.prefix_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    run = init_experiment(
        stage="p3-dream-latent-fuser",
        config={
            **asdict(config),
            **device_metadata(device),
            "metadata": metadata,
            "split_sizes": {name: len(items) for name, items in splits.items()},
        },
        experiment_name=config.experiment_name,
        tags=["dream", "p3", "latentmas", "latent-fuser", "swanlab-cloud"],
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
                pred = model(batch["inputs"])
                loss, train_metrics = fuser_loss(pred, batch["target"], config)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                optimizer.step()
                train_metrics = {"loss": float(loss.detach().item()), **train_metrics, "epoch": float(epoch)}
                write_metrics(metrics_f, "train", global_step, train_metrics)
                log_metrics(train_metrics, step=global_step, prefix="train")
                if global_step % config.valid_interval == 0:
                    valid_metrics = evaluate(model, loaders["valid"], device, config)
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
        final_valid_metrics = evaluate(model, loaders["valid"], device, config)
        final_test_metrics = evaluate(model, loaders["test"], device, config)
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
            "P3 D7 latent fuser deep-learning training",
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


def load_rows(config: FuserConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    packet_dir = Path(config.packet_dir)
    trace_dir = Path(config.trace_dir)
    packets = read_jsonl(packet_dir / "packets.jsonl")
    groups = read_jsonl(packet_dir / "packet_groups.jsonl")
    traces = read_jsonl(trace_dir / "traces.jsonl")
    packet_by_id = {str(packet["packet_id"]): packet for packet in packets}
    traces_by_call = {str(trace["call_id"]): trace for trace in traces}
    rows = []
    missing = []
    for group in groups:
        packet_ids = group.get("packet_ids_by_agent", {})
        agent_refs = {}
        for agent_id in ["agent_a", "agent_b"]:
            packet = packet_by_id.get(str(packet_ids.get(agent_id, "")))
            if packet is None:
                missing.append({"row_id": group.get("row_id", ""), "reason": f"missing_{agent_id}_packet"})
                continue
            agent_refs[agent_id] = packet["hidden_ref"]
        solver_trace = traces_by_call.get(str(group.get("solver_call_id", "")))
        if solver_trace is None:
            missing.append({"row_id": group.get("row_id", ""), "reason": "missing_solver_trace"})
            continue
        solver_ref = last_hidden_ref(solver_trace)
        if not solver_ref:
            missing.append({"row_id": group.get("row_id", ""), "reason": "missing_solver_hidden_ref"})
            continue
        if set(agent_refs) == {"agent_a", "agent_b"}:
            rows.append(
                {
                    "row_id": group.get("row_id", ""),
                    "sample_id": group.get("sample_id", ""),
                    "condition": group.get("condition", ""),
                    "agent_hidden_refs": agent_refs,
                    "solver_hidden_ref": solver_ref,
                }
            )
    if missing:
        raise ValueError(f"cannot build D7 fuser rows; missing={missing[:5]}")
    metadata = {
        "packet_dir": config.packet_dir,
        "trace_dir": config.trace_dir,
        "num_rows": len(rows),
        "condition_counts": dict(Counter(row["condition"] for row in rows)),
        "target_selection": "last solver hidden_ref event",
        "input_selection": "D6 packet selected hidden_ref for agent_a and agent_b",
    }
    return rows, metadata


def last_hidden_ref(trace: dict[str, Any]) -> str:
    events = sorted(trace.get("step_summaries", []) or [], key=lambda item: int(item.get("trace_event_index", 0)))
    for event in reversed(events):
        ref = str(event.get("hidden_ref", ""))
        if ref:
            return ref
    return ""


def split_rows(rows: list[dict[str, Any]], seed: int, train_ratio: float, valid_ratio: float) -> dict[str, list[dict[str, Any]]]:
    sample_ids = sorted({str(row["sample_id"]) for row in rows})
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
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in train_ids:
            splits["train"].append(row)
        elif sample_id in valid_ids:
            splits["valid"].append(row)
        elif sample_id in test_ids:
            splits["test"].append(row)
    return splits


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "inputs": torch.stack([item["inputs"] for item in items], dim=0),
        "target": torch.stack([item["target"] for item in items], dim=0),
    }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def fuser_loss(pred: torch.Tensor, target: torch.Tensor, config: FuserConfig) -> tuple[torch.Tensor, dict[str, float]]:
    mse = F.mse_loss(pred, target)
    cosine = F.cosine_similarity(pred.flatten(1), target.flatten(1), dim=-1).mean()
    cosine_loss = 1.0 - cosine
    loss = mse + config.cosine_loss_weight * cosine_loss
    return loss, {
        "mse": float(mse.detach().item()),
        "cosine": float(cosine.detach().item()),
        "cosine_loss": float(cosine_loss.detach().item()),
    }


@torch.no_grad()
def evaluate(model: DreamLatentFuser, loader: DataLoader, device: torch.device, config: FuserConfig) -> dict[str, float]:
    model.eval()
    matched_mse = []
    matched_cosine = []
    shuffled_mse = []
    shuffled_cosine = []
    losses = []
    for batch in loader:
        batch = move_batch(batch, device)
        pred = model(batch["inputs"])
        loss, metrics = fuser_loss(pred, batch["target"], config)
        losses.append(float(loss.item()))
        matched_mse.extend(per_example_mse(pred, batch["target"]))
        matched_cosine.extend(per_example_cosine(pred, batch["target"]))
        shuffled_inputs = batch["inputs"].roll(shifts=1, dims=0) if batch["inputs"].shape[0] > 1 else batch["inputs"].flip(dims=[1])
        shuffled_pred = model(shuffled_inputs)
        shuffled_mse.extend(per_example_mse(shuffled_pred, batch["target"]))
        shuffled_cosine.extend(per_example_cosine(shuffled_pred, batch["target"]))
    metrics = {
        "loss": mean(losses),
        "matched_mse": mean(matched_mse),
        "matched_cosine": mean(matched_cosine),
        "shuffled_mse": mean(shuffled_mse),
        "shuffled_cosine": mean(shuffled_cosine),
        "corruption_mse_margin": mean(shuffled_mse) - mean(matched_mse),
        "corruption_cosine_margin": mean(matched_cosine) - mean(shuffled_cosine),
        "num_rows": float(len(matched_mse)),
    }
    metrics["selection_metric"] = -metrics["matched_mse"] + 0.1 * metrics["corruption_mse_margin"] + 0.01 * metrics["matched_cosine"]
    return metrics


def evaluate_checkpoint(path: Path, loader: DataLoader, device: torch.device, config: FuserConfig) -> dict[str, float]:
    checkpoint = torch.load(path, map_location=device)
    model = DreamLatentFuser(
        hidden_size=config.hidden_size,
        d_model=config.d_model,
        input_tokens_per_agent=config.input_tokens_per_agent,
        prefix_len=config.prefix_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return evaluate(model, loader, device, config)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: FuserConfig,
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


def load_tensor(path_text: str) -> torch.Tensor:
    obj = torch.load(path_text, map_location="cpu")
    tensor = obj["tensor"] if isinstance(obj, dict) and "tensor" in obj else obj
    if not torch.is_tensor(tensor):
        raise TypeError(f"hidden ref did not contain a tensor: {path_text}")
    return tensor


def select_evenly_spaced(tensor: torch.Tensor, length: int) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError(f"expected [tokens, hidden], got {tuple(tensor.shape)}")
    if tensor.shape[0] == length:
        return tensor
    indices = torch.linspace(0, tensor.shape[0] - 1, length).round().long()
    return tensor.index_select(0, indices)


def per_example_mse(pred: torch.Tensor, target: torch.Tensor) -> list[float]:
    return (pred - target).pow(2).flatten(1).mean(dim=1).detach().cpu().tolist()


def per_example_cosine(pred: torch.Tensor, target: torch.Tensor) -> list[float]:
    return F.cosine_similarity(pred.flatten(1), target.flatten(1), dim=-1).detach().cpu().tolist()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
