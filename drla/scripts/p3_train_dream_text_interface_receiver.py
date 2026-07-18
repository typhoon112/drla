"""Train D7.10 Dream text-interface-aligned latent receiver.

The receiver maps D6 agent latent packets into continuous virtual message tokens
at AgentB's solver interface. Dream stays frozen. Decoded TextMAS Agent messages
are used only during training to define teacher hidden/logit targets; online
student inputs remain the no-message solver prompt plus latent packets.
"""

from __future__ import annotations

import argparse
import json
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
from drla.scripts.run_p2_phase_c_text_agents import make_solver_messages, read_jsonl  # noqa: E402
from drla.scripts.train_cola_readiness_model import device_metadata, require_cuda_training, resolve_device  # noqa: E402
from drla.tracking import finish_experiment, init_experiment, log_metrics  # noqa: E402


DEFAULT_TRACE_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_traces/"
    "musique_calibration_trace_textmas_matched200_steps64_stride4_hidden_tensor_merged_20260606"
)
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_text_interface_receivers/"
    "dream_text_interface_receiver_d710_textmas_matched200_seed20260617"
)


@dataclass(frozen=True)
class TextInterfaceReceiverConfig:
    manifest_json: str = DEFAULT_MANIFEST_JSON
    online_inputs_jsonl: str = DEFAULT_ONLINE_INPUTS_JSONL
    packet_dir: str = DEFAULT_PACKET_DIR
    trace_dir: str = DEFAULT_TRACE_DIR
    model_path: str = DEFAULT_MODEL_PATH
    init_checkpoint: str = ""
    output_dir: str = DEFAULT_OUTPUT_DIR
    device: str = "auto"
    dtype: str = "bfloat16"
    seed: int = 20260617
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    batch_size: int = 1
    epochs: int = 3
    max_train_steps: int = 0
    valid_interval: int = 10
    learning_rate: float = 8e-5
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    input_tokens_per_agent: int = 32
    prefix_len: int = 96
    max_target_tokens: int = 32
    hidden_size: int = 3584
    d_model: int = 1024
    num_memory_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.05
    hidden_align_weight: float = 0.5
    input_align_weight: float = 0.05
    teacher_kl_weight: float = 0.1
    corruption_weight: float = 0.25
    corruption_margin: float = 0.2
    corrupt_unlikelihood_weight: float = 0.0
    logit_contrast_weight: float = 0.0
    hidden_contrast_weight: float = 0.0
    contrast_temperature: float = 0.5
    negative_loss_warmup_steps: int = 0
    selection_token_accuracy_weight: float = 0.0
    selection_margin_target: float = 0.0
    selection_margin_overflow_penalty: float = 0.0
    teacher_temperature: float = 2.0
    swanlab_mode: str = "cloud"
    experiment_name: str = "p3-dream-text-interface-receiver-d710"


class TextInterfaceReceiverDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, config: TextInterfaceReceiverConfig) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        packets = load_packets(row, self.config)
        no_message_ids = self.tokenizer.apply_chat_template(
            make_solver_messages(row["online_input_fields"], upstream_messages=[]),
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).input_ids[0]
        textmas_ids = self.tokenizer.apply_chat_template(
            make_solver_messages(row["online_input_fields"], upstream_messages=row["agent_messages"]),
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
            "prompt_ids": no_message_ids.to(torch.long),
            "textmas_prompt_ids": textmas_ids.to(torch.long),
            "target_ids": target_ids.to(torch.long),
            "row_id": row["row_id"],
            "sample_id": row["sample_id"],
        }


class PacketToVirtualMessageReceiver(nn.Module):
    def __init__(self, config: TextInterfaceReceiverConfig) -> None:
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.hidden_size)
        self.input_proj = nn.Linear(config.hidden_size, config.d_model)
        self.agent_embed = nn.Embedding(2, config.d_model)
        self.packet_pos_embed = nn.Parameter(torch.zeros(1, 2 * config.input_tokens_per_agent, config.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.memory_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_memory_layers)
        self.prefix_queries = nn.Parameter(torch.randn(1, config.prefix_len, config.d_model) * 0.02)
        self.query_pos_embed = nn.Parameter(torch.zeros(1, config.prefix_len, config.d_model))
        self.query_attn = nn.MultiheadAttention(config.d_model, config.num_heads, dropout=config.dropout, batch_first=True)
        self.out = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.hidden_size),
        )
        self.output_norm = nn.LayerNorm(config.hidden_size)

    def encode_memory(self, packets: torch.Tensor) -> torch.Tensor:
        batch, num_agents, tokens, _hidden = packets.shape
        x = self.input_proj(self.input_norm(packets.reshape(batch, num_agents * tokens, -1)))
        agent_ids = torch.arange(num_agents, device=packets.device).repeat_interleave(tokens)
        x = x + self.agent_embed(agent_ids).unsqueeze(0) + self.packet_pos_embed[:, : num_agents * tokens]
        return self.memory_encoder(x)

    def forward(self, packets: torch.Tensor) -> torch.Tensor:
        memory = self.encode_memory(packets)
        queries = self.prefix_queries.expand(packets.shape[0], -1, -1) + self.query_pos_embed
        prefix, _ = self.query_attn(queries, memory, memory, need_weights=False)
        return self.output_norm(self.out(prefix))


def main() -> None:
    summary = train(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> TextInterfaceReceiverConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", default=TextInterfaceReceiverConfig.manifest_json)
    parser.add_argument("--online-inputs-jsonl", default=TextInterfaceReceiverConfig.online_inputs_jsonl)
    parser.add_argument("--packet-dir", default=TextInterfaceReceiverConfig.packet_dir)
    parser.add_argument("--trace-dir", default=TextInterfaceReceiverConfig.trace_dir)
    parser.add_argument("--model-path", default=TextInterfaceReceiverConfig.model_path)
    parser.add_argument("--init-checkpoint", default=TextInterfaceReceiverConfig.init_checkpoint)
    parser.add_argument("--output-dir", default=TextInterfaceReceiverConfig.output_dir)
    parser.add_argument("--device", default=TextInterfaceReceiverConfig.device)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=TextInterfaceReceiverConfig.dtype)
    parser.add_argument("--seed", type=int, default=TextInterfaceReceiverConfig.seed)
    parser.add_argument("--train-ratio", type=float, default=TextInterfaceReceiverConfig.train_ratio)
    parser.add_argument("--valid-ratio", type=float, default=TextInterfaceReceiverConfig.valid_ratio)
    parser.add_argument("--batch-size", type=int, default=TextInterfaceReceiverConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TextInterfaceReceiverConfig.epochs)
    parser.add_argument("--max-train-steps", type=int, default=TextInterfaceReceiverConfig.max_train_steps)
    parser.add_argument("--valid-interval", type=int, default=TextInterfaceReceiverConfig.valid_interval)
    parser.add_argument("--learning-rate", type=float, default=TextInterfaceReceiverConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TextInterfaceReceiverConfig.weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=TextInterfaceReceiverConfig.grad_clip_norm)
    parser.add_argument("--input-tokens-per-agent", type=int, default=TextInterfaceReceiverConfig.input_tokens_per_agent)
    parser.add_argument("--prefix-len", type=int, default=TextInterfaceReceiverConfig.prefix_len)
    parser.add_argument("--max-target-tokens", type=int, default=TextInterfaceReceiverConfig.max_target_tokens)
    parser.add_argument("--hidden-size", type=int, default=TextInterfaceReceiverConfig.hidden_size)
    parser.add_argument("--d-model", type=int, default=TextInterfaceReceiverConfig.d_model)
    parser.add_argument("--num-memory-layers", type=int, default=TextInterfaceReceiverConfig.num_memory_layers)
    parser.add_argument("--num-heads", type=int, default=TextInterfaceReceiverConfig.num_heads)
    parser.add_argument("--dropout", type=float, default=TextInterfaceReceiverConfig.dropout)
    parser.add_argument("--hidden-align-weight", type=float, default=TextInterfaceReceiverConfig.hidden_align_weight)
    parser.add_argument("--input-align-weight", type=float, default=TextInterfaceReceiverConfig.input_align_weight)
    parser.add_argument("--teacher-kl-weight", type=float, default=TextInterfaceReceiverConfig.teacher_kl_weight)
    parser.add_argument("--corruption-weight", type=float, default=TextInterfaceReceiverConfig.corruption_weight)
    parser.add_argument("--corruption-margin", type=float, default=TextInterfaceReceiverConfig.corruption_margin)
    parser.add_argument("--corrupt-unlikelihood-weight", type=float, default=TextInterfaceReceiverConfig.corrupt_unlikelihood_weight)
    parser.add_argument("--logit-contrast-weight", type=float, default=TextInterfaceReceiverConfig.logit_contrast_weight)
    parser.add_argument("--hidden-contrast-weight", type=float, default=TextInterfaceReceiverConfig.hidden_contrast_weight)
    parser.add_argument("--contrast-temperature", type=float, default=TextInterfaceReceiverConfig.contrast_temperature)
    parser.add_argument("--negative-loss-warmup-steps", type=int, default=TextInterfaceReceiverConfig.negative_loss_warmup_steps)
    parser.add_argument("--selection-token-accuracy-weight", type=float, default=TextInterfaceReceiverConfig.selection_token_accuracy_weight)
    parser.add_argument("--selection-margin-target", type=float, default=TextInterfaceReceiverConfig.selection_margin_target)
    parser.add_argument(
        "--selection-margin-overflow-penalty",
        type=float,
        default=TextInterfaceReceiverConfig.selection_margin_overflow_penalty,
    )
    parser.add_argument("--teacher-temperature", type=float, default=TextInterfaceReceiverConfig.teacher_temperature)
    parser.add_argument("--swanlab-mode", default=TextInterfaceReceiverConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=TextInterfaceReceiverConfig.experiment_name)
    return TextInterfaceReceiverConfig(**vars(parser.parse_args()))


def train(config: TextInterfaceReceiverConfig) -> dict[str, Any]:
    if config.swanlab_mode != "cloud":
        raise ValueError("all P3 deep-learning training must use SwanLab cloud")
    if config.valid_interval > 10:
        raise ValueError("valid_interval must be <= 10 for current P3 training")
    if config.batch_size != 1:
        raise ValueError("batch_size=1 is required because prompts have variable lengths")
    device = resolve_device(config.device)
    require_cuda_training(device, Path(__file__).name)
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output_dir is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[config.dtype]
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    dream = AutoModel.from_pretrained(config.model_path, torch_dtype=dtype, trust_remote_code=True).to(device).eval()
    for param in dream.parameters():
        param.requires_grad_(False)
    rows, metadata = load_text_interface_rows(config)
    splits = split_rows(rows, config.seed, config.train_ratio, config.valid_ratio)
    datasets = {name: TextInterfaceReceiverDataset(part, tokenizer, config) for name, part in splits.items()}
    loaders = {
        name: DataLoader(dataset, batch_size=1, shuffle=(name == "train"), collate_fn=collate_one)
        for name, dataset in datasets.items()
    }

    receiver = PacketToVirtualMessageReceiver(config).to(device)
    if config.init_checkpoint:
        init_checkpoint = torch.load(config.init_checkpoint, map_location=device)
        receiver.load_state_dict(init_checkpoint["model_state"])
    optimizer = torch.optim.AdamW(receiver.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    metrics_path = output_dir / "metrics.jsonl"
    metrics_handle = metrics_path.open("w", encoding="utf-8")
    run = init_experiment(
        stage="p3-dream-text-interface-receiver",
        experiment_name=config.experiment_name,
        config={
            **asdict(config),
            **device_metadata(device),
            "metadata": metadata,
            "split_sizes": {name: len(items) for name, items in splits.items()},
        },
        tags=["dream", "p3", "latentmas", "text-interface-receiver", "swanlab-cloud"],
        mode=config.swanlab_mode,
    )

    best_metric = float("-inf")
    best_step = 0
    global_step = 0
    started = time.time()
    try:
        for _epoch in range(config.epochs):
            for batch in loaders["train"]:
                receiver.train()
                batch = move_batch(batch, device)
                shuffled_item = datasets["train"][(global_step + 1) % len(datasets["train"])]
                shuffled_packets = shuffled_item["packets"].unsqueeze(0).to(device)
                loss, train_metrics = receiver_loss(
                    receiver=receiver,
                    dream=dream,
                    batch=batch,
                    shuffled_packets=shuffled_packets,
                    tokenizer=tokenizer,
                    config=config,
                    global_step=global_step,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(receiver.parameters(), config.grad_clip_norm)
                optimizer.step()
                global_step += 1
                write_metrics(metrics_handle, "train", global_step, train_metrics | {"loss": float(loss.detach().item())})
                log_metrics({"train/" + key: value for key, value in train_metrics.items()} | {"train/loss": float(loss.detach().item())}, step=global_step)
                if global_step % config.valid_interval == 0:
                    valid_metrics = evaluate_dataset(receiver, dream, datasets["valid"], tokenizer, device, config)
                    write_metrics(metrics_handle, "valid", global_step, valid_metrics)
                    log_metrics({"valid/" + key: value for key, value in valid_metrics.items()}, step=global_step)
                    selection_metric = valid_metrics["selection_metric"]
                    if selection_metric > best_metric:
                        best_metric = selection_metric
                        best_step = global_step
                        save_checkpoint(output_dir / "best_checkpoint.pt", receiver, optimizer, config, metadata, best_step, best_metric)
                if config.max_train_steps and global_step >= config.max_train_steps:
                    break
            if config.max_train_steps and global_step >= config.max_train_steps:
                break
        save_checkpoint(output_dir / "last_checkpoint.pt", receiver, optimizer, config, metadata, global_step, best_metric)
        if not (output_dir / "best_checkpoint.pt").exists():
            best_step = global_step
            best_metric = evaluate_dataset(receiver, dream, datasets["valid"], tokenizer, device, config)["selection_metric"]
            save_checkpoint(output_dir / "best_checkpoint.pt", receiver, optimizer, config, metadata, best_step, best_metric)
    finally:
        metrics_handle.close()
        finish_experiment()

    best_valid_metrics = evaluate_checkpoint(output_dir / "best_checkpoint.pt", dream, datasets["valid"], tokenizer, device, config)
    best_test_metrics = evaluate_checkpoint(output_dir / "best_checkpoint.pt", dream, datasets["test"], tokenizer, device, config)
    summary = {
        "created_at": int(time.time()),
        "status": "pass",
        "output_dir": str(output_dir),
        "global_step": global_step,
        "best_step": best_step,
        "best_selection_metric": best_metric,
        "swanlab_run_id": getattr(run, "id", None),
        "swanlab_url": getattr(run, "url", None),
        "config": asdict(config),
        "metadata": metadata,
        "split_sizes": {name: len(part) for name, part in splits.items()},
        "best_valid_metrics": best_valid_metrics,
        "best_test_metrics": best_test_metrics,
        "artifacts": {
            "metrics_jsonl": str(metrics_path),
            "best_checkpoint": str(output_dir / "best_checkpoint.pt"),
            "last_checkpoint": str(output_dir / "last_checkpoint.pt"),
            "summary_json": str(output_dir / "summary.json"),
        },
        "elapsed_seconds": round(time.time() - started, 3),
        "execution_boundary": [
            "P3 text-interface receiver deep-learning training",
            "CUDA/GPU required",
            "SwanLab cloud required",
            "Dream frozen; only packet-to-virtual-message receiver updates",
            "online student prompt contains no decoded Agent messages",
            "TextMAS decoded Agent messages are training-only teacher context",
            "gold answers are supervised loss targets only, not runtime inputs",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def receiver_loss(
    *,
    receiver: PacketToVirtualMessageReceiver,
    dream: Any,
    batch: dict[str, torch.Tensor],
    shuffled_packets: torch.Tensor,
    tokenizer: Any,
    config: TextInterfaceReceiverConfig,
    global_step: int | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    negative_scale = negative_loss_scale(config, global_step)
    matched = receiver_forward(receiver, dream, batch, tokenizer, config, packets=batch["packets"])
    target_ids = batch["target_ids"]
    matched_ce = F.cross_entropy(matched["target_logits"].reshape(-1, matched["target_logits"].shape[-1]).float(), target_ids.reshape(-1))
    teacher = textmas_teacher(dream, batch, tokenizer, config)
    hidden_mse, hidden_cosine_loss = prefix_hidden_alignment(matched["prefix_final_hidden"], teacher["target_hidden"])
    input_mse = F.mse_loss(matched["prefix_embeds"].float(), teacher["target_input_embeds"].float())
    teacher_kl = kl_to_teacher(matched["target_logits"], teacher["target_logits"], config.teacher_temperature)
    zero = receiver_forward(receiver, dream, batch, tokenizer, config, packets=torch.zeros_like(batch["packets"]))
    swap = receiver_forward(receiver, dream, batch, tokenizer, config, packets=batch["packets"].flip(dims=[1]))
    shuffled = receiver_forward(receiver, dream, batch, tokenizer, config, packets=shuffled_packets)
    zero_ce = target_ce(zero["target_logits"], target_ids)
    swap_ce = target_ce(swap["target_logits"], target_ids)
    shuffled_ce = target_ce(shuffled["target_logits"], target_ids)
    corruption_loss = (
        F.relu(matched_ce.detach() + config.corruption_margin - zero_ce)
        + F.relu(matched_ce.detach() + config.corruption_margin - swap_ce)
        + F.relu(matched_ce.detach() + config.corruption_margin - shuffled_ce)
    ) / 3.0
    corrupt_unlikelihood = (
        target_unlikelihood(zero["target_logits"], target_ids)
        + target_unlikelihood(swap["target_logits"], target_ids)
        + target_unlikelihood(shuffled["target_logits"], target_ids)
    ) / 3.0
    logit_contrast = ce_contrast_loss(matched_ce, [zero_ce, swap_ce, shuffled_ce], config.contrast_temperature)
    hidden_contrast = hidden_contrast_loss(
        matched["prefix_final_hidden"],
        [zero["prefix_final_hidden"], swap["prefix_final_hidden"], shuffled["prefix_final_hidden"]],
        teacher["target_hidden"],
        config.contrast_temperature,
    )
    loss = (
        matched_ce
        + config.hidden_align_weight * (hidden_mse + hidden_cosine_loss)
        + config.input_align_weight * input_mse
        + config.teacher_kl_weight * teacher_kl
        + negative_scale * config.corruption_weight * corruption_loss
        + negative_scale * config.corrupt_unlikelihood_weight * corrupt_unlikelihood
        + negative_scale * config.logit_contrast_weight * logit_contrast
        + negative_scale * config.hidden_contrast_weight * hidden_contrast
    )
    with torch.no_grad():
        pred = matched["target_logits"].argmax(dim=-1)
        token_acc = (pred == target_ids).float().mean()
        first_acc = (pred[:, :1] == target_ids[:, :1]).float().mean()
    return loss, {
        "matched_ce": float(matched_ce.detach().item()),
        "hidden_mse": float(hidden_mse.detach().item()),
        "hidden_cosine_loss": float(hidden_cosine_loss.detach().item()),
        "input_mse": float(input_mse.detach().item()),
        "teacher_kl": float(teacher_kl.detach().item()),
        "zero_ce": float(zero_ce.detach().item()),
        "agent_swap_ce": float(swap_ce.detach().item()),
        "shuffled_row_ce": float(shuffled_ce.detach().item()),
        "zero_ce_margin": float((zero_ce.detach() - matched_ce.detach()).item()),
        "agent_swap_ce_margin": float((swap_ce.detach() - matched_ce.detach()).item()),
        "shuffled_row_ce_margin": float((shuffled_ce.detach() - matched_ce.detach()).item()),
        "corruption_loss": float(corruption_loss.detach().item()),
        "corrupt_unlikelihood": float(corrupt_unlikelihood.detach().item()),
        "logit_contrast_loss": float(logit_contrast.detach().item()),
        "hidden_contrast_loss": float(hidden_contrast.detach().item()),
        "negative_loss_scale": float(negative_scale),
        "token_accuracy": float(token_acc.detach().item()),
        "first_token_accuracy": float(first_acc.detach().item()),
    }


@torch.no_grad()
def evaluate_dataset(
    receiver: PacketToVirtualMessageReceiver,
    dream: Any,
    dataset: TextInterfaceReceiverDataset,
    tokenizer: Any,
    device: torch.device,
    config: TextInterfaceReceiverConfig,
) -> dict[str, float]:
    receiver.eval()
    metrics: dict[str, list[float]] = {}
    for idx in range(len(dataset)):
        batch = move_batch(collate_one([dataset[idx]]), device)
        shuffled = dataset[(idx + 1) % len(dataset)]["packets"].unsqueeze(0).to(device)
        _loss, item = receiver_loss(
            receiver=receiver,
            dream=dream,
            batch=batch,
            shuffled_packets=shuffled,
            tokenizer=tokenizer,
            config=config,
            global_step=None,
        )
        for key, value in item.items():
            metrics.setdefault(key, []).append(float(value))
    out = {key: mean(value) for key, value in metrics.items()}
    zero_margin = selection_margin_score(out["zero_ce_margin"], config)
    shuffled_margin = selection_margin_score(out["shuffled_row_ce_margin"], config)
    agent_swap_margin = selection_margin_score(out["agent_swap_ce_margin"], config)
    out["selection_metric"] = (
        -out["matched_ce"]
        - 0.25 * out["hidden_cosine_loss"]
        + config.selection_token_accuracy_weight * out["token_accuracy"]
        + 0.10 * zero_margin
        + 0.10 * shuffled_margin
        + 0.05 * agent_swap_margin
        - 0.05 * out.get("logit_contrast_loss", 0.0)
        - 0.05 * out.get("hidden_contrast_loss", 0.0)
    )
    out["num_rows"] = float(len(dataset))
    return out


def negative_loss_scale(config: TextInterfaceReceiverConfig, global_step: int | None) -> float:
    if global_step is None or config.negative_loss_warmup_steps <= 0:
        return 1.0
    return min(1.0, max(0.0, float(global_step + 1) / float(config.negative_loss_warmup_steps)))


def selection_margin_score(margin: float, config: TextInterfaceReceiverConfig) -> float:
    if config.selection_margin_target <= 0:
        return margin
    capped = min(margin, config.selection_margin_target)
    overflow = max(0.0, margin - config.selection_margin_target)
    return capped - config.selection_margin_overflow_penalty * overflow


def receiver_forward(
    receiver: PacketToVirtualMessageReceiver,
    dream: Any,
    batch: dict[str, torch.Tensor],
    tokenizer: Any,
    config: TextInterfaceReceiverConfig,
    *,
    packets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    mask_token_id = resolve_mask_token_id(dream, tokenizer)
    prompt_ids = batch["prompt_ids"]
    target_ids = batch["target_ids"]
    masked_target = torch.full_like(target_ids, mask_token_id)
    input_ids = torch.cat([prompt_ids, masked_target], dim=1)
    prefix_embeds = receiver(packets)
    prompt_len = prompt_ids.shape[1]
    target_len = target_ids.shape[1]
    raw_logits, full_hidden = forward_with_prefix(dream, input_ids, prefix_embeds, insert_at=prompt_len)
    shifted = torch.cat([raw_logits[:, :1], raw_logits[:, :-1]], dim=1)
    target_start = prompt_len + config.prefix_len
    return {
        "target_logits": shifted[:, target_start : target_start + target_len, :],
        "prefix_final_hidden": full_hidden[:, prompt_len:target_start, :],
        "prefix_embeds": prefix_embeds,
    }


def target_ce(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), target_ids.reshape(-1))


def target_unlikelihood(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits.float(), dim=-1)
    target_probs = probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    return -torch.log1p(-target_probs.clamp(max=1.0 - 1e-6)).mean()


def ce_contrast_loss(matched_ce: torch.Tensor, corrupt_ces: list[torch.Tensor], temperature: float) -> torch.Tensor:
    scores = torch.stack([-matched_ce, *[-item for item in corrupt_ces]], dim=0).unsqueeze(0)
    scores = scores / max(temperature, 1e-6)
    target = torch.zeros(1, dtype=torch.long, device=scores.device)
    return F.cross_entropy(scores, target)


def hidden_contrast_loss(
    matched_hidden: torch.Tensor,
    corrupt_hiddens: list[torch.Tensor],
    teacher_hidden: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    def score(hidden: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(
            hidden.float().reshape(-1, hidden.shape[-1]),
            teacher_hidden.float().reshape(-1, teacher_hidden.shape[-1]),
            dim=-1,
        ).mean()

    scores = torch.stack([score(matched_hidden), *[score(hidden) for hidden in corrupt_hiddens]], dim=0).unsqueeze(0)
    target = torch.zeros(1, dtype=torch.long, device=scores.device)
    return F.cross_entropy(scores / max(temperature, 1e-6), target)


@torch.no_grad()
def textmas_teacher(dream: Any, batch: dict[str, torch.Tensor], tokenizer: Any, config: TextInterfaceReceiverConfig) -> dict[str, torch.Tensor]:
    mask_token_id = resolve_mask_token_id(dream, tokenizer)
    textmas_ids = batch["textmas_prompt_ids"]
    target_ids = batch["target_ids"]
    masked_target = torch.full_like(target_ids, mask_token_id)
    input_ids = torch.cat([textmas_ids, masked_target], dim=1)
    outputs = dream(input_ids, output_hidden_states=True)
    raw_logits = outputs.logits
    shifted = torch.cat([raw_logits[:, :1], raw_logits[:, :-1]], dim=1)
    prompt_len = textmas_ids.shape[1]
    target_len = target_ids.shape[1]
    hidden = outputs.hidden_states[-1][:, :prompt_len, :]
    input_embeds = dream.get_input_embeddings()(textmas_ids)
    return {
        "target_logits": shifted[:, prompt_len : prompt_len + target_len, :],
        "target_hidden": take_last_or_pad(hidden, config.prefix_len),
        "target_input_embeds": take_last_or_pad(input_embeds, config.prefix_len),
    }


def forward_with_prefix(dream: Any, input_ids: torch.Tensor, prefix_embeds: torch.Tensor, *, insert_at: int) -> tuple[torch.Tensor, torch.Tensor]:
    token_embeds = dream.get_input_embeddings()(input_ids)
    hidden_states = torch.cat([token_embeds[:, :insert_at], prefix_embeds.to(dtype=token_embeds.dtype), token_embeds[:, insert_at:]], dim=1)
    seq_len = hidden_states.shape[1]
    position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
    position_embeddings = dream.model.rotary_emb(hidden_states, position_ids)
    for decoder_layer in dream.model.layers:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=position_embeddings,
        )[0]
    hidden_states = dream.model.norm(hidden_states)
    return dream.lm_head(hidden_states), hidden_states


def prefix_hidden_alignment(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred_f = pred.float()
    target_f = target.float()
    mse = F.mse_loss(pred_f, target_f)
    cosine_loss = 1.0 - F.cosine_similarity(pred_f.reshape(-1, pred_f.shape[-1]), target_f.reshape(-1, target_f.shape[-1]), dim=-1).mean()
    return mse, cosine_loss


def kl_to_teacher(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    student_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_p = F.softmax(teacher_logits.float() / temperature, dim=-1)
    return F.kl_div(student_logp, teacher_p, reduction="batchmean") * (temperature**2)


def take_last_or_pad(tensor: torch.Tensor, length: int) -> torch.Tensor:
    if tensor.shape[1] >= length:
        return tensor[:, -length:, :]
    pad = tensor[:, :1, :].expand(-1, length - tensor.shape[1], -1)
    return torch.cat([pad, tensor], dim=1)


def load_text_interface_rows(config: TextInterfaceReceiverConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_rows, metadata = load_training_rows(config)  # type: ignore[arg-type]
    trace_rows = {str(row.get("row_id", "")): row for row in read_jsonl(Path(config.trace_dir) / "generations.jsonl")}
    rows = []
    missing = []
    for row in base_rows:
        trace = trace_rows.get(row["row_id"])
        if trace is None:
            missing.append({"row_id": row["row_id"], "reason": "missing_trace_generation"})
            continue
        agent_messages = trace.get("agent_messages", [])
        if len(agent_messages) < 2:
            missing.append({"row_id": row["row_id"], "reason": "missing_agent_messages"})
            continue
        rows.append({**row, "agent_messages": agent_messages})
    if missing:
        raise ValueError(f"cannot build D7.10 rows; missing={missing[:5]}")
    metadata = dict(metadata)
    metadata.update(
        {
            "trace_dir": config.trace_dir,
            "num_rows_with_textmas_teacher_messages": len(rows),
            "teacher_context": "TextMAS decoded Agent messages used only for hidden/logit distillation",
            "student_runtime_prompt": "no-message solver prompt plus D6 latent packets",
        }
    )
    return rows, metadata


def load_packets(row: dict[str, Any], config: TextInterfaceReceiverConfig) -> torch.Tensor:
    tensors = []
    for agent_id in ["agent_a", "agent_b"]:
        tensor = load_tensor(row["agent_hidden_refs"][agent_id])
        tensors.append(select_evenly_spaced(tensor, config.input_tokens_per_agent))
    return torch.stack(tensors, dim=0).to(torch.float32)


def evaluate_checkpoint(
    path: Path,
    dream: Any,
    dataset: TextInterfaceReceiverDataset,
    tokenizer: Any,
    device: torch.device,
    runtime_config: TextInterfaceReceiverConfig,
) -> dict[str, float]:
    checkpoint = torch.load(path, map_location=device)
    config = TextInterfaceReceiverConfig(**checkpoint["config"])
    receiver = PacketToVirtualMessageReceiver(config).to(device)
    receiver.load_state_dict(checkpoint["model_state"])
    return evaluate_dataset(receiver, dream, dataset, tokenizer, device, runtime_config)


def save_checkpoint(
    path: Path,
    receiver: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TextInterfaceReceiverConfig,
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


def collate_one(items: list[dict[str, Any]]) -> dict[str, Any]:
    if len(items) != 1:
        raise ValueError("collate_one requires batch_size=1")
    item = items[0]
    return {
        "packets": item["packets"].unsqueeze(0),
        "prompt_ids": item["prompt_ids"].unsqueeze(0),
        "textmas_prompt_ids": item["textmas_prompt_ids"].unsqueeze(0),
        "target_ids": item["target_ids"].unsqueeze(0),
    }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def write_metrics(handle: Any, phase: str, step: int, metrics: dict[str, float]) -> None:
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
