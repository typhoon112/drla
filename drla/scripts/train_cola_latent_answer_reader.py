"""Train a P2 latent-to-answer-state reader.

This is a communication diagnostic, not an official Cola benchmark finetune.
The model reads only sanitized latent packet fields and learns to retrieve the
P1 teacher's selected answer text from a candidate answer-text pool.  Evaluation
scores the retrieved text with the official Cola scorer, so the result is
directly comparable to text_selected on the same packet split.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from drla.scripts.audit_cola_agent_latent_packet_distribution import ShardCache, load_packet_blocks
from drla.scripts.audit_cola_sequential_latent_mas import load_official_scorer, score_text_with_official_rules
from drla.scripts.run_cola_sequential_latent_mas import DecisionSampleIdCache, TaskDataCache
from drla.scripts.train_cola_latent_receiver import (
    OFFICIAL_COLA_TASKS,
    packet_block_mask,
    packet_process_tensor,
    stable_uniform,
)
from drla.scripts.train_cola_readiness_model import device_metadata, require_cuda_training, resolve_device
from drla.tracking import finish_experiment, init_experiment, log_metrics


@dataclass(frozen=True)
class LatentAnswerReaderConfig:
    packets_jsonl: str = (
        "/data1/luyifei/drla/outputs/cola_agent_latent_comm/"
        "p2_agent_latent_comm_v2_locked_seed66_67_68_split20260601_20260529/"
        "agent_latent_comm_packets_test.jsonl"
    )
    output_dir: str = (
        "/data1/luyifei/drla/outputs/cola_latent_answer_reader/"
        "p2_latent_answer_reader_v1"
    )
    data_root: str = "/data1/luyifei/Cola-DLM/code/generate_task_data"
    acc_calc_script: str = "/data1/luyifei/Cola-DLM/code/scripts/acc_calc.py"
    seed: int = 20260529
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    max_packets: int = 0
    batch_size: int = 256
    epochs: int = 12
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    d_model: int = 128
    attention_heads: int = 4
    inter_layers: int = 2
    text_max_bytes: int = 128
    temperature: float = 0.07
    valid_interval: int = 10
    max_cached_shards: int = 1024
    num_workers: int = 0
    device: str = "auto"
    swanlab_mode: str = "cloud"
    experiment_name: str = "p2-latent-answer-reader-v1"


class LatentAnswerReaderModel(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        process_dim: int,
        max_blocks: int,
        block_size: int,
        task_count: int,
        d_model: int,
        attention_heads: int,
        inter_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % attention_heads != 0:
            raise ValueError("d_model must be divisible by attention_heads")
        self.max_blocks = max_blocks
        self.block_size = block_size
        self.temperature = 1.0
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.latent_adapter = nn.Linear(latent_dim, d_model)
        self.slot_pos = nn.Embedding(block_size, d_model)
        self.block_pos = nn.Embedding(max_blocks, d_model)
        self.latent_encoder = nn.TransformerEncoder(
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
        self.process_mlp = nn.Sequential(
            nn.LayerNorm(process_dim),
            nn.Linear(process_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.task_embedding = nn.Embedding(task_count, d_model)
        self.text_embedding = nn.Embedding(257, d_model, padding_idx=0)
        self.text_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.latent_out = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.text_out = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def encode_latent(
        self,
        latent_blocks: torch.Tensor,
        process_features: torch.Tensor,
        block_mask: torch.Tensor,
        task_idx: torch.Tensor,
    ) -> torch.Tensor:
        batch, max_blocks, block_size, _ = latent_blocks.shape
        x = self.latent_adapter(self.latent_norm(latent_blocks))
        block_ids = torch.arange(max_blocks, device=x.device).view(1, max_blocks, 1)
        slot_ids = torch.arange(block_size, device=x.device).view(1, 1, block_size)
        x = x + self.block_pos(block_ids) + self.slot_pos(slot_ids)
        x = x.view(batch, max_blocks * block_size, -1)
        token_mask = block_mask.unsqueeze(-1).expand(batch, max_blocks, block_size).reshape(batch, -1)
        x = self.latent_encoder(x, src_key_padding_mask=~token_mask)
        denom = token_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        latent_state = (x * token_mask.unsqueeze(-1).float()).sum(dim=1) / denom
        process_masked = process_features.masked_fill(~block_mask.unsqueeze(-1), 0.0)
        process_state = process_masked.sum(dim=1) / block_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        process_state = self.process_mlp(process_state)
        task_state = self.task_embedding(task_idx)
        return F.normalize(self.latent_out(torch.cat([latent_state, process_state, task_state], dim=-1)), dim=-1)

    def encode_text(self, text_bytes: torch.Tensor, task_idx: torch.Tensor) -> torch.Tensor:
        mask = text_bytes != 0
        embedded = self.text_embedding(text_bytes)
        denom = mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        text_state = (embedded * mask.unsqueeze(-1).float()).sum(dim=1) / denom
        text_state = self.text_mlp(text_state)
        task_state = self.task_embedding(task_idx)
        return F.normalize(self.text_out(torch.cat([text_state, task_state], dim=-1)), dim=-1)

    def forward(
        self,
        latent_blocks: torch.Tensor,
        process_features: torch.Tensor,
        block_mask: torch.Tensor,
        task_idx: torch.Tensor,
        text_bytes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.encode_latent(latent_blocks, process_features, block_mask, task_idx),
            self.encode_text(text_bytes, task_idx),
        )


def main() -> None:
    summary = train_latent_answer_reader(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> LatentAnswerReaderConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets-jsonl", default=LatentAnswerReaderConfig.packets_jsonl)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default=LatentAnswerReaderConfig.data_root)
    parser.add_argument("--acc-calc-script", default=LatentAnswerReaderConfig.acc_calc_script)
    parser.add_argument("--seed", type=int, default=LatentAnswerReaderConfig.seed)
    parser.add_argument("--train-ratio", type=float, default=LatentAnswerReaderConfig.train_ratio)
    parser.add_argument("--valid-ratio", type=float, default=LatentAnswerReaderConfig.valid_ratio)
    parser.add_argument("--max-packets", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=LatentAnswerReaderConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=LatentAnswerReaderConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=LatentAnswerReaderConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=LatentAnswerReaderConfig.weight_decay)
    parser.add_argument("--dropout", type=float, default=LatentAnswerReaderConfig.dropout)
    parser.add_argument("--d-model", type=int, default=LatentAnswerReaderConfig.d_model)
    parser.add_argument("--attention-heads", type=int, default=LatentAnswerReaderConfig.attention_heads)
    parser.add_argument("--inter-layers", type=int, default=LatentAnswerReaderConfig.inter_layers)
    parser.add_argument("--text-max-bytes", type=int, default=LatentAnswerReaderConfig.text_max_bytes)
    parser.add_argument("--temperature", type=float, default=LatentAnswerReaderConfig.temperature)
    parser.add_argument("--valid-interval", type=int, default=LatentAnswerReaderConfig.valid_interval)
    parser.add_argument("--max-cached-shards", type=int, default=LatentAnswerReaderConfig.max_cached_shards)
    parser.add_argument("--num-workers", type=int, default=LatentAnswerReaderConfig.num_workers)
    parser.add_argument("--device", default=LatentAnswerReaderConfig.device)
    parser.add_argument("--swanlab-mode", default=LatentAnswerReaderConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=LatentAnswerReaderConfig.experiment_name)
    args = parser.parse_args()
    return LatentAnswerReaderConfig(
        packets_jsonl=args.packets_jsonl,
        output_dir=args.output_dir,
        data_root=args.data_root,
        acc_calc_script=args.acc_calc_script,
        seed=args.seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        max_packets=args.max_packets,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        d_model=args.d_model,
        attention_heads=args.attention_heads,
        inter_layers=args.inter_layers,
        text_max_bytes=args.text_max_bytes,
        temperature=args.temperature,
        valid_interval=args.valid_interval,
        max_cached_shards=args.max_cached_shards,
        num_workers=args.num_workers,
        device=args.device,
        swanlab_mode=args.swanlab_mode,
        experiment_name=args.experiment_name,
    )


def train_latent_answer_reader(config: LatentAnswerReaderConfig) -> dict[str, Any]:
    validate_config(config)
    torch.manual_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    scorer = load_official_scorer(Path(config.acc_calc_script))
    packets = load_packets(Path(config.packets_jsonl), config.max_packets)
    examples = build_examples(packets, config, scorer)
    splits = split_indices(examples, config)
    tensors_by_split, metadata = build_tensors(examples, splits, config)
    train_ds, valid_ds, test_ds, norm_stats = make_datasets(tensors_by_split)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    device = resolve_device(config.device)
    require_cuda_training(device, "train_cola_latent_answer_reader.py")
    model = LatentAnswerReaderModel(
        latent_dim=metadata["latent_dim"],
        process_dim=metadata["process_dim"],
        max_blocks=metadata["max_blocks"],
        block_size=metadata["block_size"],
        task_count=len(metadata["task_to_idx"]),
        d_model=config.d_model,
        attention_heads=config.attention_heads,
        inter_layers=config.inter_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    run = init_experiment(
        stage="p2-latent-answer-reader",
        config={**asdict(config), **device_metadata(device)},
        experiment_name=config.experiment_name,
        tags=["cola", "official-benchmark", "latent-answer-reader", "contrastive", "p2"],
        mode=config.swanlab_mode,
    )
    best_metric = float("-inf")
    best_step = 0
    global_step = 0
    metrics_f = metrics_path.open("w", encoding="utf-8")
    try:
        for epoch in range(config.epochs):
            model.train()
            for batch in train_loader:
                global_step += 1
                batch = [item.to(device) for item in batch]
                optimizer.zero_grad(set_to_none=True)
                loss, train_metrics = compute_loss(model, batch, config.temperature)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_metrics = {"loss": float(loss.detach().item()), **train_metrics}
                write_metrics(metrics_f, "train", global_step, train_metrics)
                log_metrics(train_metrics, step=global_step, prefix="train")
                if global_step % config.valid_interval == 0:
                    valid_metrics = evaluate_reader(model, valid_loader, tensors_by_split["valid"], examples, splits["valid"], scorer, device, config)
                    write_metrics(metrics_f, "valid", global_step, valid_metrics)
                    log_metrics(valid_metrics, step=global_step, prefix="valid")
                    current = valid_metrics["official_top1_accuracy"]
                    if current > best_metric:
                        best_metric = current
                        best_step = global_step
                        save_checkpoint(checkpoint_dir / "best_checkpoint.pt", model, optimizer, config, metadata, norm_stats, best_step, best_metric)
        valid_metrics = evaluate_reader(model, valid_loader, tensors_by_split["valid"], examples, splits["valid"], scorer, device, config)
        test_metrics = evaluate_reader(model, test_loader, tensors_by_split["test"], examples, splits["test"], scorer, device, config)
        write_metrics(metrics_f, "valid", global_step, valid_metrics)
        write_metrics(metrics_f, "test", global_step, test_metrics)
        log_metrics(valid_metrics, step=global_step, prefix="valid")
        log_metrics(test_metrics, step=global_step, prefix="test")
        if valid_metrics["official_top1_accuracy"] > best_metric:
            best_metric = valid_metrics["official_top1_accuracy"]
            best_step = global_step
            save_checkpoint(checkpoint_dir / "best_checkpoint.pt", model, optimizer, config, metadata, norm_stats, best_step, best_metric)
        save_checkpoint(checkpoint_dir / "last_checkpoint.pt", model, optimizer, config, metadata, norm_stats, global_step, valid_metrics["official_top1_accuracy"])
    finally:
        metrics_f.close()
        finish_experiment()

    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "swanlab_run_id": getattr(run, "id", None),
        "num_examples": len(examples),
        "split_sizes": {name: len(indices) for name, indices in splits.items()},
        "metadata": metadata,
        "best_step": best_step,
        "best_valid_official_top1_accuracy": best_metric,
        "final_valid_metrics": valid_metrics,
        "final_test_metrics": test_metrics,
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "metrics_jsonl": str(metrics_path),
            "best_checkpoint": str(checkpoint_dir / "best_checkpoint.pt"),
            "last_checkpoint": str(checkpoint_dir / "last_checkpoint.pt"),
        },
        "interpretation": (
            "P2 latent-answer reader diagnostic. The model retrieves a P1 teacher answer text "
            "from latent packet inputs; it is not an official benchmark finetune and does not "
            "prove text-channel superiority."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def validate_config(config: LatentAnswerReaderConfig) -> None:
    if config.valid_interval > 10:
        raise ValueError("valid_interval must be <= 10 steps")
    if config.swanlab_mode != "cloud":
        raise ValueError("training must use SwanLab cloud; pass --swanlab-mode cloud")
    if config.max_packets < 0:
        raise ValueError("max_packets must be non-negative")
    if not 0.0 < config.train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0.0 <= config.valid_ratio < 1.0:
        raise ValueError("valid_ratio must be in [0, 1)")
    if config.train_ratio + config.valid_ratio >= 1.0:
        raise ValueError("train_ratio + valid_ratio must be < 1")


def load_packets(path: Path, max_packets: int) -> list[dict[str, Any]]:
    packets = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                packets.append(json.loads(line))
                if max_packets and len(packets) >= max_packets:
                    break
    if not packets:
        raise ValueError("no packets loaded")
    return packets


def build_examples(packets: list[dict[str, Any]], config: LatentAnswerReaderConfig, scorer: Any) -> list[dict[str, Any]]:
    decision_cache: dict[str, dict[str, dict[str, Any]]] = {}
    data_cache = TaskDataCache(Path(config.data_root))
    sample_cache = DecisionSampleIdCache()
    examples = []
    for packet in packets:
        halt_path = str(packet["audit_refs"]["halt_decisions_jsonl"])
        if halt_path not in decision_cache:
            decision_cache[halt_path] = {row["sample_key"]: row for row in read_jsonl(Path(halt_path))}
        decision = decision_cache[halt_path][str(packet["sample_key"])]
        task = str(packet["task"])
        sample_id = sample_cache.resolve(packet)
        raw_item = data_cache.get(task, sample_id)
        selected_prediction = str(decision.get("selected_prediction", ""))
        selected_score = score_text_with_official_rules(
            task=task,
            text=selected_prediction,
            ground_truth=raw_item.get("ground_truth", raw_item.get("answer", "")),
            choices=raw_item.get("choices", []),
            scorer=scorer,
        )
        examples.append(
            {
                "packet": packet,
                "task": task,
                "sample_key": str(packet["sample_key"]),
                "sample_id": sample_id,
                "selected_prediction": selected_prediction,
                "answer_key": f"{task}::{normalize_answer(selected_prediction)}",
                "selected_correct": int(bool(selected_score["correct"])),
                "ground_truth": raw_item.get("ground_truth", raw_item.get("answer", "")),
                "choices": raw_item.get("choices", []),
            }
        )
    return examples


def split_indices(examples: list[dict[str, Any]], config: LatentAnswerReaderConfig) -> dict[str, list[int]]:
    splits = {"train": [], "valid": [], "test": []}
    for index, example in enumerate(examples):
        value = stable_uniform(f"{config.seed}:{example['sample_key']}")
        if value < config.train_ratio:
            splits["train"].append(index)
        elif value < config.train_ratio + config.valid_ratio:
            splits["valid"].append(index)
        else:
            splits["test"].append(index)
    for name, indices in splits.items():
        if not indices:
            raise ValueError(f"empty split: {name}")
    return splits


def build_tensors(
    examples: list[dict[str, Any]],
    splits: dict[str, list[int]],
    config: LatentAnswerReaderConfig,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    first_block = examples[0]["packet"]["latent_memory"]["blocks"][0]
    max_blocks = max(int(example["packet"]["agent_a"]["max_block_budget"]) for example in examples)
    block_size = int(first_block["latent_ref"]["shape"][0])
    latent_dim = int(first_block["latent_ref"]["shape"][1])
    process_dim = 11
    task_to_idx = {task: idx for idx, task in enumerate(OFFICIAL_COLA_TASKS)}
    key_to_idx = {key: idx for idx, key in enumerate(sorted({example["answer_key"] for example in examples}))}
    tensors_by_split = {}
    shard_cache = ShardCache(config.max_cached_shards)
    for split_name, indices in splits.items():
        tensors_by_split[split_name] = build_split_tensors(
            examples=[examples[index] for index in indices],
            task_to_idx=task_to_idx,
            key_to_idx=key_to_idx,
            max_blocks=max_blocks,
            block_size=block_size,
            latent_dim=latent_dim,
            process_dim=process_dim,
            text_max_bytes=config.text_max_bytes,
            shard_cache=shard_cache,
        )
    metadata = {
        "task_to_idx": task_to_idx,
        "answer_key_count": len(key_to_idx),
        "max_blocks": max_blocks,
        "block_size": block_size,
        "latent_dim": latent_dim,
        "process_dim": process_dim,
        "text_max_bytes": config.text_max_bytes,
        "online_input_policy": (
            "Latent side uses only packet latent tensors, process features, and task id. "
            "Teacher selected_prediction is the target text state, not an online input."
        ),
    }
    return tensors_by_split, metadata


def build_split_tensors(
    *,
    examples: list[dict[str, Any]],
    task_to_idx: dict[str, int],
    key_to_idx: dict[str, int],
    max_blocks: int,
    block_size: int,
    latent_dim: int,
    process_dim: int,
    text_max_bytes: int,
    shard_cache: ShardCache,
) -> dict[str, torch.Tensor]:
    count = len(examples)
    latent_blocks = torch.zeros(count, max_blocks, block_size, latent_dim, dtype=torch.float32)
    process_features = torch.zeros(count, max_blocks, process_dim, dtype=torch.float32)
    block_mask = torch.zeros(count, max_blocks, dtype=torch.bool)
    task_idx = torch.zeros(count, dtype=torch.long)
    text_bytes = torch.zeros(count, text_max_bytes, dtype=torch.long)
    answer_key_idx = torch.zeros(count, dtype=torch.long)
    selected_correct = torch.zeros(count, dtype=torch.float32)
    for index, example in enumerate(examples):
        packet = example["packet"]
        for block_offset, block in enumerate(load_packet_blocks(packet, shard_cache)):
            latent_blocks[index, block_offset] = block
        process_features[index] = packet_process_tensor(packet, max_blocks, process_dim)
        block_mask[index] = packet_block_mask(packet, max_blocks)
        task_idx[index] = task_to_idx[str(example["task"])]
        text_bytes[index] = encode_bytes(str(example["selected_prediction"]), text_max_bytes)
        answer_key_idx[index] = key_to_idx[str(example["answer_key"])]
        selected_correct[index] = float(example["selected_correct"])
    return {
        "latent_blocks": latent_blocks,
        "process_features": process_features,
        "block_mask": block_mask,
        "task_idx": task_idx,
        "text_bytes": text_bytes,
        "answer_key_idx": answer_key_idx,
        "selected_correct": selected_correct,
    }


def make_datasets(tensors_by_split: dict[str, dict[str, torch.Tensor]]) -> tuple[TensorDataset, TensorDataset, TensorDataset, dict[str, torch.Tensor]]:
    train = tensors_by_split["train"]
    active = train["process_features"][train["block_mask"]]
    process_mean = active.mean(dim=0, keepdim=True)
    process_std = active.std(dim=0, keepdim=True).clamp_min(1e-6)
    norm_stats = {"process_mean": process_mean, "process_std": process_std}

    def dataset(name: str) -> TensorDataset:
        tensors = tensors_by_split[name]
        process = (tensors["process_features"] - process_mean.view(1, 1, -1)) / process_std.view(1, 1, -1)
        process = process.masked_fill(~tensors["block_mask"].unsqueeze(-1), 0.0)
        return TensorDataset(
            tensors["latent_blocks"],
            process,
            tensors["block_mask"],
            tensors["task_idx"],
            tensors["text_bytes"],
            tensors["answer_key_idx"],
        )

    return dataset("train"), dataset("valid"), dataset("test"), norm_stats


def compute_loss(
    model: LatentAnswerReaderModel,
    batch: list[torch.Tensor],
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    latent_blocks, process_features, block_mask, task_idx, text_bytes, answer_key_idx = batch
    latent_emb, text_emb = model(latent_blocks, process_features, block_mask, task_idx, text_bytes)
    logits = latent_emb @ text_emb.T / temperature
    positives = answer_key_idx[:, None] == answer_key_idx[None, :]
    loss_i = multi_positive_ce(logits, positives)
    loss_t = multi_positive_ce(logits.T, positives.T)
    loss = 0.5 * (loss_i + loss_t)
    with torch.no_grad():
        top1 = logits.argmax(dim=1)
        top1_match = positives[torch.arange(positives.shape[0], device=logits.device), top1].float().mean()
    return loss, {"batch_answer_key_top1": float(top1_match.item())}


def multi_positive_ce(logits: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=1)
    target = positives.float()
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0)
    return -(target * log_prob).sum(dim=1).mean()


@torch.no_grad()
def evaluate_reader(
    model: LatentAnswerReaderModel,
    loader: DataLoader,
    tensors: dict[str, torch.Tensor],
    examples: list[dict[str, Any]],
    split_indices: list[int],
    scorer: Any,
    device: torch.device,
    config: LatentAnswerReaderConfig,
) -> dict[str, float]:
    model.eval()
    latent_embs = []
    text_embs = []
    for batch in loader:
        batch = [item.to(device) for item in batch]
        latent_blocks, process_features, block_mask, task_idx, text_bytes, _ = batch
        latent_embs.append(model.encode_latent(latent_blocks, process_features, block_mask, task_idx).cpu())
        text_embs.append(model.encode_text(text_bytes, task_idx).cpu())
    latent = torch.cat(latent_embs)
    text = torch.cat(text_embs)
    scores = latent @ text.T / config.temperature
    top1 = scores.argmax(dim=1)
    key_idx = tensors["answer_key_idx"]
    text_match = (key_idx[top1] == key_idx).float()
    selected_correct = tensors["selected_correct"]
    official_correct = []
    for row_idx, candidate_idx in enumerate(top1.tolist()):
        example = examples[split_indices[row_idx]]
        candidate = examples[split_indices[candidate_idx]]
        score = score_text_with_official_rules(
            task=str(example["task"]),
            text=str(candidate["selected_prediction"]),
            ground_truth=example["ground_truth"],
            choices=example["choices"],
            scorer=scorer,
        )
        official_correct.append(float(score["correct"]))
    official = torch.tensor(official_correct, dtype=torch.float32)
    return {
        "answer_key_top1": float(text_match.mean().item()),
        "official_top1_accuracy": float(official.mean().item()),
        "selected_reference_accuracy": float(selected_correct.mean().item()),
        "retrieval_gap_vs_selected": float(official.mean().item() - selected_correct.mean().item()),
        "num_examples": float(len(split_indices)),
    }


def save_checkpoint(
    path: Path,
    model: LatentAnswerReaderModel,
    optimizer: torch.optim.Optimizer,
    config: LatentAnswerReaderConfig,
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


def encode_bytes(text: str, max_len: int) -> torch.Tensor:
    raw = text.encode("utf-8", errors="ignore")[:max_len]
    values = [byte + 1 for byte in raw]
    values += [0] * (max_len - len(values))
    return torch.tensor(values, dtype=torch.long)


def normalize_answer(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    main()
