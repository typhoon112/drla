"""Train D7 V4 Dream layer-conditioned latent receiver.

This trainer freezes Dream and learns lightweight cross-attention adapters
inside selected Dream layers. The adapters condition generated/masked positions
on D6 agent_a/agent_b latent packets. Unlike shallow input soft-prefixing, this
injects the packet signal at native hidden-state layers during Dream denoising.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.scripts.p3_train_dream_latent_fuser import load_tensor, select_evenly_spaced, split_rows  # noqa: E402
from drla.scripts.p3_train_dream_soft_prefix_adapter import (  # noqa: E402
    DEFAULT_MANIFEST_JSON,
    DEFAULT_MODEL_PATH,
    DEFAULT_ONLINE_INPUTS_JSONL,
    DEFAULT_PACKET_DIR,
    load_training_rows,
    resolve_mask_token_id,
)
from drla.scripts.run_p2_phase_c_text_agents import make_solver_messages  # noqa: E402
from drla.scripts.train_cola_readiness_model import device_metadata, require_cuda_training, resolve_device  # noqa: E402
from drla.tracking import finish_experiment, init_experiment, log_metrics  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_layer_receivers/"
    "dream_layer_receiver_v1_textmas_matched200_seed20260607"
)


@dataclass(frozen=True)
class LayerReceiverConfig:
    manifest_json: str = DEFAULT_MANIFEST_JSON
    online_inputs_jsonl: str = DEFAULT_ONLINE_INPUTS_JSONL
    packet_dir: str = DEFAULT_PACKET_DIR
    model_path: str = DEFAULT_MODEL_PATH
    output_dir: str = DEFAULT_OUTPUT_DIR
    device: str = "auto"
    dtype: str = "bfloat16"
    seed: int = 20260607
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    batch_size: int = 1
    epochs: int = 3
    max_train_steps: int = 0
    valid_interval: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    input_tokens_per_agent: int = 32
    max_target_tokens: int = 32
    hidden_size: int = 3584
    d_model: int = 256
    num_memory_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    selected_layers: str = "7,14,21,27"
    swanlab_mode: str = "cloud"
    experiment_name: str = "p3-dream-layer-receiver-v1-textmas-matched200"


class LayerReceiverDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, config: LayerReceiverConfig) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        packets = load_row_packets(row, self.config)
        prompt_ids = self.tokenizer.apply_chat_template(
            make_solver_messages(row["online_input_fields"], upstream_messages=[]),
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).input_ids[0]
        target_ids = self.tokenizer(
            f"Final answer: {row['gold_answer']}",
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0][: self.config.max_target_tokens]
        if target_ids.numel() == 0:
            raise ValueError(f"empty target ids for row_id={row['row_id']}")
        return {
            "packets": packets,
            "prompt_ids": prompt_ids.to(torch.long),
            "target_ids": target_ids.to(torch.long),
            "row_id": row["row_id"],
            "sample_id": row["sample_id"],
        }


class PacketMemoryEncoder(nn.Module):
    def __init__(self, config: LayerReceiverConfig) -> None:
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
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_memory_layers)

    def forward(self, packets: torch.Tensor) -> torch.Tensor:
        batch, num_agents, tokens, hidden = packets.shape
        x = self.input_proj(self.input_norm(packets.reshape(batch, num_agents * tokens, hidden)))
        agent_ids = torch.arange(num_agents, device=packets.device).repeat_interleave(tokens)
        x = x + self.agent_embed(agent_ids).unsqueeze(0) + self.pos_embed[:, : num_agents * tokens]
        return self.encoder(x)


class LayerCrossAttentionConditioner(nn.Module):
    def __init__(self, config: LayerReceiverConfig) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(config.hidden_size)
        self.query_proj = nn.Linear(config.hidden_size, config.d_model)
        self.packet_norm = nn.LayerNorm(config.d_model)
        self.cross_attn = nn.MultiheadAttention(config.d_model, config.num_heads, dropout=config.dropout, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, config.hidden_size))
        self.dropout = nn.Dropout(config.dropout)
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, hidden_states: torch.Tensor, memory: torch.Tensor, condition_mask: torch.Tensor) -> torch.Tensor:
        hidden_float = hidden_states.to(torch.float32)
        memory_float = memory.to(torch.float32)
        query = self.query_proj(self.query_norm(hidden_float))
        packet_memory = self.packet_norm(memory_float)
        conditioned, _ = self.cross_attn(query, packet_memory, packet_memory, need_weights=False)
        delta = self.dropout(self.out(conditioned))
        gate = torch.sigmoid(self.gate_logit).to(dtype=hidden_states.dtype)
        return hidden_states + gate * delta.to(dtype=hidden_states.dtype) * condition_mask.to(dtype=hidden_states.dtype)


class DreamLayerConditionedReceiver(nn.Module):
    def __init__(self, config: LayerReceiverConfig) -> None:
        super().__init__()
        self.config = config
        self.input_tokens_per_agent = config.input_tokens_per_agent
        self.selected_layers = parse_selected_layers(config.selected_layers)
        self.memory_encoder = PacketMemoryEncoder(config)
        self.conditioners = nn.ModuleDict(
            {str(layer_idx): LayerCrossAttentionConditioner(config) for layer_idx in self.selected_layers}
        )

    def forward_logits(
        self,
        dream: Any,
        input_ids: torch.Tensor,
        packets: torch.Tensor,
        *,
        condition_start: int,
    ) -> torch.Tensor:
        memory = self.memory_encoder(packets)
        hidden_states = dream.get_input_embeddings()(input_ids)
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        position_embeddings = dream.model.rotary_emb(hidden_states, position_ids)
        condition_mask = torch.zeros(hidden_states.shape[:2], device=hidden_states.device, dtype=hidden_states.dtype)
        condition_mask[:, condition_start:] = 1.0
        condition_mask = condition_mask.unsqueeze(-1)
        for layer_idx, decoder_layer in enumerate(dream.model.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
            layer_key = str(layer_idx)
            if layer_key in self.conditioners:
                hidden_states = self.conditioners[layer_key](hidden_states, memory, condition_mask)
        hidden_states = dream.model.norm(hidden_states)
        return dream.lm_head(hidden_states)


def main() -> None:
    summary = train(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> LayerReceiverConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", default=LayerReceiverConfig.manifest_json)
    parser.add_argument("--online-inputs-jsonl", default=LayerReceiverConfig.online_inputs_jsonl)
    parser.add_argument("--packet-dir", default=LayerReceiverConfig.packet_dir)
    parser.add_argument("--model-path", default=LayerReceiverConfig.model_path)
    parser.add_argument("--output-dir", default=LayerReceiverConfig.output_dir)
    parser.add_argument("--device", default=LayerReceiverConfig.device)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=LayerReceiverConfig.dtype)
    parser.add_argument("--seed", type=int, default=LayerReceiverConfig.seed)
    parser.add_argument("--train-ratio", type=float, default=LayerReceiverConfig.train_ratio)
    parser.add_argument("--valid-ratio", type=float, default=LayerReceiverConfig.valid_ratio)
    parser.add_argument("--batch-size", type=int, default=LayerReceiverConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=LayerReceiverConfig.epochs)
    parser.add_argument("--max-train-steps", type=int, default=LayerReceiverConfig.max_train_steps)
    parser.add_argument("--valid-interval", type=int, default=LayerReceiverConfig.valid_interval)
    parser.add_argument("--learning-rate", type=float, default=LayerReceiverConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=LayerReceiverConfig.weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=LayerReceiverConfig.grad_clip_norm)
    parser.add_argument("--input-tokens-per-agent", type=int, default=LayerReceiverConfig.input_tokens_per_agent)
    parser.add_argument("--max-target-tokens", type=int, default=LayerReceiverConfig.max_target_tokens)
    parser.add_argument("--hidden-size", type=int, default=LayerReceiverConfig.hidden_size)
    parser.add_argument("--d-model", type=int, default=LayerReceiverConfig.d_model)
    parser.add_argument("--num-memory-layers", type=int, default=LayerReceiverConfig.num_memory_layers)
    parser.add_argument("--num-heads", type=int, default=LayerReceiverConfig.num_heads)
    parser.add_argument("--dropout", type=float, default=LayerReceiverConfig.dropout)
    parser.add_argument("--selected-layers", default=LayerReceiverConfig.selected_layers)
    parser.add_argument("--swanlab-mode", default=LayerReceiverConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=LayerReceiverConfig.experiment_name)
    return LayerReceiverConfig(**vars(parser.parse_args()))


def train(config: LayerReceiverConfig) -> dict[str, Any]:
    if config.swanlab_mode != "cloud":
        raise ValueError("D7 layer-conditioned receiver training must use SwanLab cloud")
    if config.valid_interval > 10:
        raise ValueError("valid_interval must be <= 10 for current P3 training")
    if config.batch_size != 1:
        raise ValueError("This script currently requires --batch-size 1 because prompts have variable lengths")
    set_seed(config.seed)
    device = resolve_device(config.device)
    require_cuda_training(device, "p3_train_dream_layer_conditioned_receiver.py")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[config.dtype]
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    dream = AutoModel.from_pretrained(config.model_path, torch_dtype=model_dtype, trust_remote_code=True).to(device).eval()
    for param in dream.parameters():
        param.requires_grad_(False)
    rows, metadata = load_training_rows(config)
    metadata["selected_layers"] = parse_selected_layers(config.selected_layers)
    splits = split_rows(rows, config.seed, config.train_ratio, config.valid_ratio)
    datasets = {name: LayerReceiverDataset(items, tokenizer, config) for name, items in splits.items()}
    loaders = {
        name: DataLoader(dataset, batch_size=1, shuffle=(name == "train"), collate_fn=collate_one)
        for name, dataset in datasets.items()
    }
    receiver = DreamLayerConditionedReceiver(config).to(device)
    optimizer = torch.optim.AdamW(receiver.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    run = init_experiment(
        stage="p3-dream-layer-conditioned-receiver",
        config={
            **asdict(config),
            **device_metadata(device),
            "metadata": metadata,
            "split_sizes": {name: len(items) for name, items in splits.items()},
        },
        experiment_name=config.experiment_name,
        tags=["dream", "p3", "latentmas", "layer-receiver", "cross-attention", "swanlab-cloud"],
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
            receiver.train()
            for batch in loaders["train"]:
                global_step += 1
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss, train_metrics = receiver_loss(receiver, dream, batch, tokenizer)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(receiver.parameters(), config.grad_clip_norm)
                optimizer.step()
                train_metrics = {"loss": float(loss.detach().item()), **train_metrics, "epoch": float(epoch)}
                write_metrics(metrics_f, "train", global_step, train_metrics)
                log_metrics(train_metrics, step=global_step, prefix="train")
                if global_step % config.valid_interval == 0:
                    valid_metrics = evaluate_dataset(receiver, dream, datasets["valid"], tokenizer, device)
                    write_metrics(metrics_f, "valid", global_step, valid_metrics)
                    log_metrics(valid_metrics, step=global_step, prefix="valid")
                    if valid_metrics["selection_metric"] > best_metric:
                        best_metric = valid_metrics["selection_metric"]
                        best_step = global_step
                        save_checkpoint(output_dir / "best_checkpoint.pt", receiver, optimizer, config, metadata, best_step, best_metric)
                if config.max_train_steps and global_step >= config.max_train_steps:
                    break
            if config.max_train_steps and global_step >= config.max_train_steps:
                break
        final_valid_metrics = evaluate_dataset(receiver, dream, datasets["valid"], tokenizer, device)
        final_test_metrics = evaluate_dataset(receiver, dream, datasets["test"], tokenizer, device)
        write_metrics(metrics_f, "valid", global_step, final_valid_metrics)
        write_metrics(metrics_f, "test", global_step, final_test_metrics)
        log_metrics(final_valid_metrics, step=global_step, prefix="valid")
        log_metrics(final_test_metrics, step=global_step, prefix="test")
        if final_valid_metrics["selection_metric"] > best_metric:
            best_metric = final_valid_metrics["selection_metric"]
            best_step = global_step
            save_checkpoint(output_dir / "best_checkpoint.pt", receiver, optimizer, config, metadata, best_step, best_metric)
        save_checkpoint(output_dir / "last_checkpoint.pt", receiver, optimizer, config, metadata, global_step, final_valid_metrics["selection_metric"])
    finally:
        metrics_f.close()
        finish_experiment()

    best_valid_metrics = evaluate_checkpoint(output_dir / "best_checkpoint.pt", dream, datasets["valid"], tokenizer, device)
    best_test_metrics = evaluate_checkpoint(output_dir / "best_checkpoint.pt", dream, datasets["test"], tokenizer, device)
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
            "P3 D7 V4 layer-conditioned receiver deep-learning training",
            "CUDA/GPU required",
            "SwanLab cloud required",
            "Dream frozen; only packet memory encoder and layer conditioners update",
            "adapter inputs are D6 agent_a/agent_b suffix tensors",
            "gold answers are supervised loss targets only, not runtime inputs",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def receiver_loss(
    receiver: DreamLayerConditionedReceiver,
    dream: Any,
    batch: dict[str, torch.Tensor],
    tokenizer: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    prompt_ids = batch["prompt_ids"]
    target_ids = batch["target_ids"]
    mask_token_id = resolve_mask_token_id(dream, tokenizer)
    masked_target = torch.full_like(target_ids, mask_token_id)
    input_ids = torch.cat([prompt_ids, masked_target], dim=1)
    prompt_len = prompt_ids.shape[1]
    logits = receiver.forward_logits(dream, input_ids, batch["packets"], condition_start=prompt_len)
    shifted = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    target_logits = shifted[:, prompt_len : prompt_len + target_ids.shape[1], :]
    loss = F.cross_entropy(target_logits.reshape(-1, target_logits.shape[-1]).float(), target_ids.reshape(-1))
    with torch.no_grad():
        pred = target_logits.argmax(dim=-1)
        token_acc = (pred == target_ids).float().mean()
        first_acc = (pred[:, :1] == target_ids[:, :1]).float().mean()
    return loss, {
        "ce": float(loss.detach().item()),
        "token_accuracy": float(token_acc.detach().item()),
        "first_token_accuracy": float(first_acc.detach().item()),
        "target_tokens": float(target_ids.numel()),
        "prompt_tokens": float(prompt_ids.numel()),
    }


@torch.no_grad()
def evaluate_dataset(
    receiver: DreamLayerConditionedReceiver,
    dream: Any,
    dataset: LayerReceiverDataset,
    tokenizer: Any,
    device: torch.device,
) -> dict[str, float]:
    receiver.eval()
    matched = []
    zero = []
    agent_swap = []
    shuffled = []
    token_acc = []
    first_acc = []
    for idx in range(len(dataset)):
        batch = move_batch(collate_one([dataset[idx]]), device)
        loss, metrics = receiver_loss(receiver, dream, batch, tokenizer)
        matched.append(float(loss.item()))
        token_acc.append(metrics["token_accuracy"])
        first_acc.append(metrics["first_token_accuracy"])

        zero_batch = {**batch, "packets": torch.zeros_like(batch["packets"])}
        zero_loss, _ = receiver_loss(receiver, dream, zero_batch, tokenizer)
        zero.append(float(zero_loss.item()))

        swap_batch = {**batch, "packets": batch["packets"].flip(dims=[1])}
        swap_loss, _ = receiver_loss(receiver, dream, swap_batch, tokenizer)
        agent_swap.append(float(swap_loss.item()))

        shuffled_item = dataset[(idx + 1) % len(dataset)]
        shuffled_batch = {**batch, "packets": shuffled_item["packets"].unsqueeze(0).to(device)}
        shuffled_loss, _ = receiver_loss(receiver, dream, shuffled_batch, tokenizer)
        shuffled.append(float(shuffled_loss.item()))
    metrics_out = {
        "matched_ce": mean(matched),
        "zero_ce": mean(zero),
        "shuffled_row_ce": mean(shuffled),
        "agent_swap_ce": mean(agent_swap),
        "zero_ce_margin": mean(zero) - mean(matched),
        "shuffled_row_ce_margin": mean(shuffled) - mean(matched),
        "agent_swap_ce_margin": mean(agent_swap) - mean(matched),
        "token_accuracy": mean(token_acc),
        "first_token_accuracy": mean(first_acc),
        "num_rows": float(len(matched)),
    }
    metrics_out["selection_metric"] = (
        -metrics_out["matched_ce"]
        + 0.05 * metrics_out["zero_ce_margin"]
        + 0.05 * metrics_out["shuffled_row_ce_margin"]
        + 0.02 * metrics_out["agent_swap_ce_margin"]
    )
    return metrics_out


def evaluate_checkpoint(
    path: Path,
    dream: Any,
    dataset: LayerReceiverDataset,
    tokenizer: Any,
    device: torch.device,
) -> dict[str, float]:
    checkpoint = torch.load(path, map_location=device)
    config = LayerReceiverConfig(**checkpoint["config"])
    receiver = DreamLayerConditionedReceiver(config).to(device)
    receiver.load_state_dict(checkpoint["model_state"])
    return evaluate_dataset(receiver, dream, dataset, tokenizer, device)


def save_checkpoint(
    path: Path,
    receiver: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: LayerReceiverConfig,
    metadata: dict[str, Any],
    step: int,
    selection_metric: float,
) -> None:
    torch.save(
        {
            "model_state": receiver.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(config),
            "metadata": metadata,
            "step": step,
            "selection_metric": selection_metric,
        },
        path,
    )


def load_row_packets(row: dict[str, Any], config: LayerReceiverConfig) -> torch.Tensor:
    tensors = []
    for agent_id in ["agent_a", "agent_b"]:
        tensor = load_tensor(row["agent_hidden_refs"][agent_id])
        tensors.append(select_evenly_spaced(tensor, config.input_tokens_per_agent))
    return torch.stack(tensors, dim=0).to(torch.float32)


def collate_one(items: list[dict[str, Any]]) -> dict[str, Any]:
    if len(items) != 1:
        raise ValueError("collate_one requires batch_size=1")
    item = items[0]
    return {
        "packets": item["packets"].unsqueeze(0),
        "prompt_ids": item["prompt_ids"].unsqueeze(0),
        "target_ids": item["target_ids"].unsqueeze(0),
    }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def parse_selected_layers(text: str) -> list[int]:
    layers = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not layers:
        raise ValueError("selected_layers must not be empty")
    return layers


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
