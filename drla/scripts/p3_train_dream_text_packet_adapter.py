"""Train a text-encoded packet adapter for the Dream latent receiver.

D7.5 showed that Dream last-layer hidden states from Agent text messages cannot
be directly consumed by the V7 receiver. This trainer freezes Dream and the V7
receiver, then learns a lightweight adapter that maps text-encoded Agent packets
into the receiver's expected packet space.
"""

from __future__ import annotations

import argparse
import json
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

from drla.scripts.p3_collect_dream_step_traces import find_last_layer_module  # noqa: E402
from drla.scripts.p3_run_dream_text_encoded_packet_eval import (  # noqa: E402
    DEFAULT_TEXTMAS_GENERATIONS,
    encode_agent_messages,
    load_textmas_messages,
)
from drla.scripts.p3_train_dream_layer_conditioned_receiver import (  # noqa: E402
    DreamLayerConditionedReceiver,
    LayerReceiverConfig,
    collate_one,
    load_row_packets,
    load_training_rows,
    move_batch,
    receiver_loss,
    split_rows,
    write_metrics,
)
from drla.scripts.train_cola_readiness_model import device_metadata, require_cuda_training, resolve_device  # noqa: E402
from drla.tracking import finish_experiment, init_experiment, log_metrics  # noqa: E402


DEFAULT_RECEIVER_CHECKPOINT = (
    "/data1/luyifei/drla/outputs/p3_dream_layer_receivers/"
    "dream_layer_receiver_v7_v4init_zeroshuf_textmas_matched200_seed20260607_20260607/"
    "best_checkpoint.pt"
)
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_text_packet_adapters/"
    "dream_text_packet_adapter_v1_v7_seed20260617"
)


@dataclass(frozen=True)
class TextPacketAdapterConfig:
    receiver_checkpoint: str = DEFAULT_RECEIVER_CHECKPOINT
    textmas_generations_jsonl: str = DEFAULT_TEXTMAS_GENERATIONS
    output_dir: str = DEFAULT_OUTPUT_DIR
    device: str = "auto"
    dtype: str = "bfloat16"
    seed: int = 20260617
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    batch_size: int = 1
    epochs: int = 5
    max_train_steps: int = 0
    valid_interval: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    input_tokens_per_agent: int = 32
    hidden_size: int = 3584
    d_model: int = 256
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    corruption_margin: float = 0.2
    corruption_loss_weight: float = 0.1
    distill_loss_weight: float = 0.0
    swanlab_mode: str = "cloud"
    experiment_name: str = "p3-dream-text-packet-adapter-v1"


class TextPacketAdapterDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        text_packets: dict[str, torch.Tensor],
        tokenizer: Any,
        receiver_config: LayerReceiverConfig,
    ) -> None:
        self.rows = rows
        self.text_packets = text_packets
        self.tokenizer = tokenizer
        self.receiver_config = receiver_config

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        row_id = str(row["row_id"])
        # Reuse the receiver dataset prompt construction through a tiny local import
        # to keep runtime prompts exactly aligned with D7.
        from drla.scripts.run_p2_phase_c_text_agents import make_solver_messages  # noqa: PLC0415

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
        ).input_ids[0][: self.receiver_config.max_target_tokens]
        if target_ids.numel() == 0:
            raise ValueError(f"empty target ids for row_id={row_id}")
        return {
            "text_packets": self.text_packets[row_id].to(torch.float32),
            "target_packets": load_row_packets(row, self.receiver_config),
            "prompt_ids": prompt_ids.to(torch.long),
            "target_ids": target_ids.to(torch.long),
            "row_id": row_id,
            "sample_id": row["sample_id"],
        }


class TextPacketAdapter(nn.Module):
    def __init__(self, config: TextPacketAdapterConfig) -> None:
        super().__init__()
        self.input_tokens_per_agent = config.input_tokens_per_agent
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
        self.output = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.hidden_size),
        )
        self.residual_gate_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, text_packets: torch.Tensor) -> torch.Tensor:
        batch, num_agents, tokens, hidden = text_packets.shape
        x = self.input_proj(self.input_norm(text_packets.reshape(batch, num_agents * tokens, hidden)))
        agent_ids = torch.arange(num_agents, device=text_packets.device).repeat_interleave(tokens)
        x = x + self.agent_embed(agent_ids).unsqueeze(0) + self.pos_embed[:, : num_agents * tokens]
        delta = self.output(self.encoder(x)).reshape(batch, num_agents, tokens, hidden)
        gate = torch.sigmoid(self.residual_gate_logit)
        return text_packets + gate * delta


def main() -> None:
    summary = train(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> TextPacketAdapterConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    for field_name, field_def in TextPacketAdapterConfig.__dataclass_fields__.items():
        default = field_def.default
        arg_name = "--" + field_name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(arg_name, action="store_true", default=default)
        else:
            parser.add_argument(arg_name, type=type(default), default=default)
    return TextPacketAdapterConfig(**vars(parser.parse_args()))


def train(config: TextPacketAdapterConfig) -> dict[str, Any]:
    if config.swanlab_mode != "cloud":
        raise ValueError("text packet adapter training must use SwanLab cloud")
    if config.valid_interval > 10:
        raise ValueError("valid_interval must be <= 10 for current P3 training")
    if config.batch_size != 1:
        raise ValueError("This trainer currently requires --batch-size 1")
    set_seed(config.seed)
    rng = random.Random(config.seed + 19)
    device = resolve_device(config.device)
    require_cuda_training(device, "p3_train_dream_text_packet_adapter.py")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    receiver_checkpoint = torch.load(config.receiver_checkpoint, map_location=device)
    receiver_config = LayerReceiverConfig(**receiver_checkpoint["config"])
    model_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[config.dtype]
    tokenizer = AutoTokenizer.from_pretrained(receiver_config.model_path, trust_remote_code=True)
    dream = AutoModel.from_pretrained(receiver_config.model_path, torch_dtype=model_dtype, trust_remote_code=True).to(device).eval()
    for param in dream.parameters():
        param.requires_grad_(False)
    receiver = DreamLayerConditionedReceiver(receiver_config).to(device).eval()
    receiver.load_state_dict(receiver_checkpoint["model_state"])
    for param in receiver.parameters():
        param.requires_grad_(False)
    _, last_layer = find_last_layer_module(dream)
    if last_layer is None:
        raise RuntimeError("Could not find Dream last layer for text packet encoding")

    rows, metadata = load_training_rows(receiver_config)
    text_packets = build_text_packet_cache(rows, config, dream, tokenizer, last_layer, receiver_config, device)
    splits = split_rows(rows, config.seed, config.train_ratio, config.valid_ratio)
    datasets = {
        name: TextPacketAdapterDataset(items, text_packets, tokenizer, receiver_config)
        for name, items in splits.items()
    }
    loaders = {
        name: DataLoader(dataset, batch_size=1, shuffle=(name == "train"), collate_fn=collate_one_text_adapter)
        for name, dataset in datasets.items()
    }
    adapter = TextPacketAdapter(config).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    run = init_experiment(
        stage="p3-dream-text-packet-adapter",
        config={
            **asdict(config),
            **device_metadata(device),
            "metadata": {
                **metadata,
                "receiver_checkpoint": config.receiver_checkpoint,
                "textmas_generations_jsonl": config.textmas_generations_jsonl,
                "text_packet_rows": len(text_packets),
                "objective": "frozen Dream + frozen V7 receiver CE with zero/shuffled controls",
            },
            "split_sizes": {name: len(items) for name, items in splits.items()},
        },
        experiment_name=config.experiment_name,
        tags=["dream", "p3", "latentmas", "text-packet-adapter", "swanlab-cloud"],
        mode=config.swanlab_mode,
    )

    metrics_path = output_dir / "metrics.jsonl"
    metrics_f = metrics_path.open("w", encoding="utf-8")
    best_metric = float("-inf")
    best_step = 0
    global_step = 0
    try:
        for epoch in range(config.epochs):
            adapter.train()
            for batch in loaders["train"]:
                global_step += 1
                batch = move_batch(batch, device)
                shuffled_batch = move_batch(collate_one_text_adapter([datasets["train"][rng.randrange(len(datasets["train"]))]]), device)
                optimizer.zero_grad(set_to_none=True)
                loss, train_metrics = adapter_loss(adapter, receiver, dream, batch, shuffled_batch, tokenizer, config)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), config.grad_clip_norm)
                optimizer.step()
                train_metrics = {"loss": float(loss.detach().item()), **train_metrics, "epoch": float(epoch)}
                write_metrics(metrics_f, "train", global_step, train_metrics)
                log_metrics(train_metrics, step=global_step, prefix="train")
                if global_step % config.valid_interval == 0:
                    valid_metrics = evaluate(adapter, receiver, dream, datasets["valid"], tokenizer, device, config)
                    write_metrics(metrics_f, "valid", global_step, valid_metrics)
                    log_metrics(valid_metrics, step=global_step, prefix="valid")
                    if valid_metrics["selection_metric"] > best_metric:
                        best_metric = valid_metrics["selection_metric"]
                        best_step = global_step
                        save_checkpoint(output_dir / "best_checkpoint.pt", adapter, optimizer, config, best_step, best_metric)
                if config.max_train_steps and global_step >= config.max_train_steps:
                    break
            if config.max_train_steps and global_step >= config.max_train_steps:
                break
        final_valid_metrics = evaluate(adapter, receiver, dream, datasets["valid"], tokenizer, device, config)
        final_test_metrics = evaluate(adapter, receiver, dream, datasets["test"], tokenizer, device, config)
        write_metrics(metrics_f, "valid", global_step, final_valid_metrics)
        write_metrics(metrics_f, "test", global_step, final_test_metrics)
        log_metrics(final_valid_metrics, step=global_step, prefix="valid")
        log_metrics(final_test_metrics, step=global_step, prefix="test")
        if final_valid_metrics["selection_metric"] > best_metric:
            best_metric = final_valid_metrics["selection_metric"]
            best_step = global_step
            save_checkpoint(output_dir / "best_checkpoint.pt", adapter, optimizer, config, best_step, best_metric)
        save_checkpoint(output_dir / "last_checkpoint.pt", adapter, optimizer, config, global_step, final_valid_metrics["selection_metric"])
    finally:
        metrics_f.close()
        finish_experiment()

    best_valid_metrics = evaluate_checkpoint(output_dir / "best_checkpoint.pt", receiver, dream, datasets["valid"], tokenizer, device)
    best_test_metrics = evaluate_checkpoint(output_dir / "best_checkpoint.pt", receiver, dream, datasets["test"], tokenizer, device)
    summary = {
        "created_at": int(time.time()),
        "status": "pass",
        "config": asdict(config),
        "swanlab_run_id": getattr(run, "id", None),
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
            "P3 D7.6 text-packet adapter deep-learning training",
            "CUDA/GPU required",
            "SwanLab cloud required",
            "Dream and V7 receiver are frozen",
            "Agent text messages are encoded into continuous packets only",
            "gold answers are supervised loss targets only",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def build_text_packet_cache(
    rows: list[dict[str, Any]],
    config: TextPacketAdapterConfig,
    dream: Any,
    tokenizer: Any,
    last_layer: nn.Module,
    receiver_config: LayerReceiverConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    textmas_by_row = load_textmas_messages(Path(config.textmas_generations_jsonl))
    cache = {}
    for row in rows:
        row_id = str(row["row_id"])
        messages = textmas_by_row.get(row_id)
        if messages is None:
            raise ValueError(f"missing TextMAS agent messages for row_id={row_id}")
        cache[row_id] = encode_agent_messages(messages, dream, tokenizer, last_layer, receiver_config, device).squeeze(0).cpu()
    return cache


def adapter_loss(
    adapter: TextPacketAdapter,
    receiver: DreamLayerConditionedReceiver,
    dream: Any,
    batch: dict[str, torch.Tensor],
    shuffled_batch: dict[str, torch.Tensor],
    tokenizer: Any,
    config: TextPacketAdapterConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    matched_packets = adapter(batch["text_packets"])
    matched_batch = {**batch, "packets": matched_packets}
    matched_loss, matched_metrics = receiver_loss(receiver, dream, matched_batch, tokenizer)
    zero_batch = {**batch, "packets": torch.zeros_like(matched_packets)}
    zero_loss = receiver_loss(receiver, dream, zero_batch, tokenizer)[0]
    shuffled_packets = adapter(shuffled_batch["text_packets"])
    shuffled_loss = receiver_loss(receiver, dream, {**batch, "packets": shuffled_packets}, tokenizer)[0]
    zero_margin = F.relu(matched_loss.detach() + config.corruption_margin - zero_loss)
    shuffled_margin = F.relu(matched_loss.detach() + config.corruption_margin - shuffled_loss)
    corruption_loss = (zero_margin + shuffled_margin) / 2
    distill_loss = F.mse_loss(matched_packets, batch["target_packets"]) if config.distill_loss_weight else matched_loss.new_zeros(())
    loss = matched_loss + config.corruption_loss_weight * corruption_loss + config.distill_loss_weight * distill_loss
    return loss, {
        "matched_ce": float(matched_loss.detach().item()),
        "zero_ce": float(zero_loss.detach().item()),
        "shuffled_row_ce": float(shuffled_loss.detach().item()),
        "zero_ce_margin": float((zero_loss.detach() - matched_loss.detach()).item()),
        "shuffled_row_ce_margin": float((shuffled_loss.detach() - matched_loss.detach()).item()),
        "zero_margin_violation": float(zero_margin.detach().item()),
        "shuffled_row_margin_violation": float(shuffled_margin.detach().item()),
        "corruption_loss": float(corruption_loss.detach().item()),
        "distill_loss": float(distill_loss.detach().item()),
        "token_accuracy": matched_metrics["token_accuracy"],
        "first_token_accuracy": matched_metrics["first_token_accuracy"],
    }


@torch.no_grad()
def evaluate(
    adapter: TextPacketAdapter,
    receiver: DreamLayerConditionedReceiver,
    dream: Any,
    dataset: TextPacketAdapterDataset,
    tokenizer: Any,
    device: torch.device,
    config: TextPacketAdapterConfig,
) -> dict[str, float]:
    adapter.eval()
    matched = []
    zero = []
    shuffled = []
    token_acc = []
    first_acc = []
    for idx in range(len(dataset)):
        batch = move_batch(collate_one_text_adapter([dataset[idx]]), device)
        matched_packets = adapter(batch["text_packets"])
        loss, metrics = receiver_loss(receiver, dream, {**batch, "packets": matched_packets}, tokenizer)
        matched.append(float(loss.item()))
        token_acc.append(metrics["token_accuracy"])
        first_acc.append(metrics["first_token_accuracy"])
        zero_loss = receiver_loss(receiver, dream, {**batch, "packets": torch.zeros_like(matched_packets)}, tokenizer)[0]
        zero.append(float(zero_loss.item()))
        shuffled_item = move_batch(collate_one_text_adapter([dataset[(idx + 1) % len(dataset)]]), device)
        shuffled_packets = adapter(shuffled_item["text_packets"])
        shuffled_loss = receiver_loss(receiver, dream, {**batch, "packets": shuffled_packets}, tokenizer)[0]
        shuffled.append(float(shuffled_loss.item()))
    out = {
        "matched_ce": mean(matched),
        "zero_ce": mean(zero),
        "shuffled_row_ce": mean(shuffled),
        "zero_ce_margin": mean(zero) - mean(matched),
        "shuffled_row_ce_margin": mean(shuffled) - mean(matched),
        "token_accuracy": mean(token_acc),
        "first_token_accuracy": mean(first_acc),
        "num_rows": float(len(matched)),
    }
    out["selection_metric"] = -out["matched_ce"] + 0.05 * out["zero_ce_margin"] + 0.05 * out["shuffled_row_ce_margin"]
    return out


def evaluate_checkpoint(
    path: Path,
    receiver: DreamLayerConditionedReceiver,
    dream: Any,
    dataset: TextPacketAdapterDataset,
    tokenizer: Any,
    device: torch.device,
) -> dict[str, float]:
    checkpoint = torch.load(path, map_location=device)
    config = TextPacketAdapterConfig(**checkpoint["config"])
    adapter = TextPacketAdapter(config).to(device)
    adapter.load_state_dict(checkpoint["model_state"])
    return evaluate(adapter, receiver, dream, dataset, tokenizer, device, config)


def save_checkpoint(
    path: Path,
    adapter: TextPacketAdapter,
    optimizer: torch.optim.Optimizer,
    config: TextPacketAdapterConfig,
    step: int,
    selection_metric: float,
) -> None:
    torch.save(
        {
            "model_state": adapter.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(config),
            "step": step,
            "selection_metric": selection_metric,
        },
        path,
    )


def collate_one_text_adapter(items: list[dict[str, Any]]) -> dict[str, Any]:
    if len(items) != 1:
        raise ValueError("collate_one_text_adapter requires batch_size=1")
    item = items[0]
    return {
        "text_packets": item["text_packets"].unsqueeze(0),
        "target_packets": item["target_packets"].unsqueeze(0),
        "prompt_ids": item["prompt_ids"].unsqueeze(0),
        "target_ids": item["target_ids"].unsqueeze(0),
    }


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
