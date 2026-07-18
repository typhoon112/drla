"""Train D7 Dream embedding-space soft-prefix adapter.

This deep-learning training script maps D6 upstream evidence-agent suffix
tensors into Dream input-embedding soft prefixes. Dream is frozen, but gradients
flow through Dream into the adapter. The online receiver prompt is the
no-message solver prompt; gold answers are used only as supervised training
targets and never as runtime inputs.
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
from transformers import AutoModel, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.scripts.p3_train_dream_latent_fuser import (  # noqa: E402
    DEFAULT_PACKET_DIR,
    load_tensor,
    select_evenly_spaced,
    split_rows,
)
from drla.scripts.run_p2_phase_c_text_agents import make_solver_messages, read_jsonl  # noqa: E402
from drla.scripts.train_cola_readiness_model import device_metadata, require_cuda_training, resolve_device  # noqa: E402
from drla.tracking import finish_experiment, init_experiment, log_metrics  # noqa: E402


DEFAULT_MANIFEST_JSON = (
    "/data1/luyifei/drla/outputs/p2_phase_c_manifests/"
    "musique_calibration_manifest_200_seed20260601/manifest.json"
)
DEFAULT_ONLINE_INPUTS_JSONL = (
    "/data1/luyifei/drla/outputs/p2_phase_c_control_inputs/"
    "musique_calibration_controls_200_seed20260601_v1_strict_wrong/online_inputs.jsonl"
)
DEFAULT_MODEL_PATH = "/data1/luyifei/drla/models/Dream-v0-Instruct-7B"
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_soft_prefix_adapters/"
    "dream_soft_prefix_adapter_v1_textmas_matched200_seed20260607"
)


@dataclass(frozen=True)
class SoftPrefixConfig:
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
    prefix_len: int = 16
    max_target_tokens: int = 32
    hidden_size: int = 3584
    d_model: int = 256
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    swanlab_mode: str = "cloud"
    experiment_name: str = "p3-dream-soft-prefix-adapter-v1-textmas-matched200"


class SoftPrefixDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, config: SoftPrefixConfig) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        agent_tensors = []
        for agent_id in ["agent_a", "agent_b"]:
            tensor = load_tensor(row["agent_hidden_refs"][agent_id])
            agent_tensors.append(select_evenly_spaced(tensor, self.config.input_tokens_per_agent))
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
            "packets": torch.stack(agent_tensors, dim=0).to(torch.float32),
            "prompt_ids": prompt_ids.to(torch.long),
            "target_ids": target_ids.to(torch.long),
            "sample_id": row["sample_id"],
            "row_id": row["row_id"],
        }


class DreamSoftPrefixAdapter(nn.Module):
    def __init__(self, config: SoftPrefixConfig, embed_size: int) -> None:
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
        self.query = nn.Parameter(torch.randn(config.prefix_len, config.d_model) / math.sqrt(config.d_model))
        self.cross_attn = nn.MultiheadAttention(config.d_model, config.num_heads, dropout=config.dropout, batch_first=True)
        self.output = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, embed_size),
        )

    def forward(self, packets: torch.Tensor) -> torch.Tensor:
        batch, num_agents, tokens, hidden = packets.shape
        if num_agents != 2:
            raise ValueError(f"expected 2 agents, got {num_agents}")
        x = self.input_proj(self.input_norm(packets.reshape(batch, num_agents * tokens, hidden)))
        agent_ids = torch.arange(num_agents, device=packets.device).repeat_interleave(tokens)
        x = x + self.agent_embed(agent_ids).unsqueeze(0) + self.pos_embed[:, : num_agents * tokens]
        memory = self.encoder(x)
        queries = self.query.unsqueeze(0).expand(batch, -1, -1)
        prefix, _ = self.cross_attn(queries, memory, memory, need_weights=False)
        return self.output(prefix)


def main() -> None:
    summary = train(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> SoftPrefixConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", default=SoftPrefixConfig.manifest_json)
    parser.add_argument("--online-inputs-jsonl", default=SoftPrefixConfig.online_inputs_jsonl)
    parser.add_argument("--packet-dir", default=SoftPrefixConfig.packet_dir)
    parser.add_argument("--model-path", default=SoftPrefixConfig.model_path)
    parser.add_argument("--output-dir", default=SoftPrefixConfig.output_dir)
    parser.add_argument("--device", default=SoftPrefixConfig.device)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=SoftPrefixConfig.dtype)
    parser.add_argument("--seed", type=int, default=SoftPrefixConfig.seed)
    parser.add_argument("--train-ratio", type=float, default=SoftPrefixConfig.train_ratio)
    parser.add_argument("--valid-ratio", type=float, default=SoftPrefixConfig.valid_ratio)
    parser.add_argument("--batch-size", type=int, default=SoftPrefixConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=SoftPrefixConfig.epochs)
    parser.add_argument("--max-train-steps", type=int, default=SoftPrefixConfig.max_train_steps)
    parser.add_argument("--valid-interval", type=int, default=SoftPrefixConfig.valid_interval)
    parser.add_argument("--learning-rate", type=float, default=SoftPrefixConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=SoftPrefixConfig.weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=SoftPrefixConfig.grad_clip_norm)
    parser.add_argument("--input-tokens-per-agent", type=int, default=SoftPrefixConfig.input_tokens_per_agent)
    parser.add_argument("--prefix-len", type=int, default=SoftPrefixConfig.prefix_len)
    parser.add_argument("--max-target-tokens", type=int, default=SoftPrefixConfig.max_target_tokens)
    parser.add_argument("--hidden-size", type=int, default=SoftPrefixConfig.hidden_size)
    parser.add_argument("--d-model", type=int, default=SoftPrefixConfig.d_model)
    parser.add_argument("--num-layers", type=int, default=SoftPrefixConfig.num_layers)
    parser.add_argument("--num-heads", type=int, default=SoftPrefixConfig.num_heads)
    parser.add_argument("--dropout", type=float, default=SoftPrefixConfig.dropout)
    parser.add_argument("--swanlab-mode", default=SoftPrefixConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=SoftPrefixConfig.experiment_name)
    return SoftPrefixConfig(**vars(parser.parse_args()))


def train(config: SoftPrefixConfig) -> dict[str, Any]:
    if config.swanlab_mode != "cloud":
        raise ValueError("D7 soft-prefix adapter training must use SwanLab cloud")
    if config.valid_interval > 10:
        raise ValueError("valid_interval must be <= 10 for current P3 training")
    if config.batch_size != 1:
        raise ValueError("This script currently requires --batch-size 1 because prompts have variable lengths")
    set_seed(config.seed)
    device = resolve_device(config.device)
    require_cuda_training(device, "p3_train_dream_soft_prefix_adapter.py")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[config.dtype]
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    dream = AutoModel.from_pretrained(config.model_path, torch_dtype=model_dtype, trust_remote_code=True).to(device).eval()
    for param in dream.parameters():
        param.requires_grad_(False)
    embed_size = int(dream.get_input_embeddings().embedding_dim)
    if embed_size != config.hidden_size:
        raise ValueError(f"embed_size {embed_size} != configured hidden_size {config.hidden_size}")

    rows, metadata = load_training_rows(config)
    splits = split_rows(rows, config.seed, config.train_ratio, config.valid_ratio)
    datasets = {name: SoftPrefixDataset(items, tokenizer, config) for name, items in splits.items()}
    loaders = {
        name: DataLoader(dataset, batch_size=1, shuffle=(name == "train"), collate_fn=collate_one)
        for name, dataset in datasets.items()
    }
    adapter = DreamSoftPrefixAdapter(config, embed_size=embed_size).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    run = init_experiment(
        stage="p3-dream-soft-prefix-adapter",
        config={
            **asdict(config),
            **device_metadata(device),
            "metadata": metadata,
            "split_sizes": {name: len(items) for name, items in splits.items()},
        },
        experiment_name=config.experiment_name,
        tags=["dream", "p3", "latentmas", "soft-prefix", "receiver", "swanlab-cloud"],
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
            adapter.train()
            for batch in loaders["train"]:
                global_step += 1
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                prefix = adapter(batch["packets"])
                loss, train_metrics = soft_prefix_loss(dream, batch, prefix, tokenizer, model_dtype)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), config.grad_clip_norm)
                optimizer.step()
                train_metrics = {"loss": float(loss.detach().item()), **train_metrics, "epoch": float(epoch)}
                write_metrics(metrics_f, "train", global_step, train_metrics)
                log_metrics(train_metrics, step=global_step, prefix="train")
                if global_step % config.valid_interval == 0:
                    valid_metrics = evaluate(adapter, dream, loaders["valid"], tokenizer, device, model_dtype)
                    write_metrics(metrics_f, "valid", global_step, valid_metrics)
                    log_metrics(valid_metrics, step=global_step, prefix="valid")
                    if valid_metrics["selection_metric"] > best_metric:
                        best_metric = valid_metrics["selection_metric"]
                        best_step = global_step
                        save_checkpoint(output_dir / "best_checkpoint.pt", adapter, optimizer, config, metadata, best_step, best_metric)
                if config.max_train_steps and global_step >= config.max_train_steps:
                    break
            if config.max_train_steps and global_step >= config.max_train_steps:
                break
        final_valid_metrics = evaluate(adapter, dream, loaders["valid"], tokenizer, device, model_dtype)
        final_test_metrics = evaluate(adapter, dream, loaders["test"], tokenizer, device, model_dtype)
        write_metrics(metrics_f, "valid", global_step, final_valid_metrics)
        write_metrics(metrics_f, "test", global_step, final_test_metrics)
        log_metrics(final_valid_metrics, step=global_step, prefix="valid")
        log_metrics(final_test_metrics, step=global_step, prefix="test")
        if final_valid_metrics["selection_metric"] > best_metric:
            best_metric = final_valid_metrics["selection_metric"]
            best_step = global_step
            save_checkpoint(output_dir / "best_checkpoint.pt", adapter, optimizer, config, metadata, best_step, best_metric)
        save_checkpoint(output_dir / "last_checkpoint.pt", adapter, optimizer, config, metadata, global_step, final_valid_metrics["selection_metric"])
    finally:
        metrics_f.close()
        finish_experiment()

    best_valid_metrics = evaluate_checkpoint(output_dir / "best_checkpoint.pt", dream, loaders["valid"], tokenizer, device, model_dtype)
    best_test_metrics = evaluate_checkpoint(output_dir / "best_checkpoint.pt", dream, loaders["test"], tokenizer, device, model_dtype)
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
            "P3 D7 soft-prefix adapter deep-learning training",
            "CUDA/GPU required",
            "SwanLab cloud required",
            "Dream frozen; only adapter weights update",
            "adapter inputs are D6 agent_a/agent_b suffix tensors",
            "gold answers are supervised loss targets only, not runtime inputs",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def load_training_rows(config: SoftPrefixConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(Path(config.manifest_json).read_text(encoding="utf-8"))
    samples = {str(sample["sample_id"]): sample for sample in manifest.get("samples", [])}
    online_by_row = {
        str(row["row_id"]): row
        for row in read_jsonl(Path(config.online_inputs_jsonl))
        if row.get("condition") == "textmas_matched"
    }
    packet_dir = Path(config.packet_dir)
    packets = {str(packet["packet_id"]): packet for packet in read_jsonl(packet_dir / "packets.jsonl")}
    groups = read_jsonl(packet_dir / "packet_groups.jsonl")
    rows = []
    missing = []
    for group in groups:
        row_id = str(group.get("row_id", ""))
        online = online_by_row.get(row_id)
        sample = samples.get(str(group.get("sample_id", "")))
        if online is None or sample is None:
            missing.append({"row_id": row_id, "reason": "missing_online_or_sample"})
            continue
        packet_ids = group.get("packet_ids_by_agent", {})
        agent_refs = {}
        for agent_id in ["agent_a", "agent_b"]:
            packet = packets.get(str(packet_ids.get(agent_id, "")))
            if packet is None:
                missing.append({"row_id": row_id, "reason": f"missing_{agent_id}_packet"})
                continue
            agent_refs[agent_id] = packet["hidden_ref"]
        scoring = sample.get("scoring", {})
        gold_answer = str(scoring.get("gold_answer", "")).strip()
        if set(agent_refs) == {"agent_a", "agent_b"} and gold_answer:
            rows.append(
                {
                    "row_id": row_id,
                    "sample_id": str(group.get("sample_id", "")),
                    "condition": str(group.get("condition", "")),
                    "agent_hidden_refs": agent_refs,
                    "online_input_fields": online.get("online_input_fields", {}),
                    "gold_answer": gold_answer,
                }
            )
    if missing:
        raise ValueError(f"cannot build soft-prefix rows; missing={missing[:5]}")
    metadata = {
        "manifest_json": config.manifest_json,
        "online_inputs_jsonl": config.online_inputs_jsonl,
        "packet_dir": config.packet_dir,
        "num_rows": len(rows),
        "condition_counts": dict(Counter(row["condition"] for row in rows)),
        "input_selection": "D6 packet selected hidden_ref for agent_a and agent_b",
        "target": "gold answer tokens for supervised receiver training",
        "runtime_prompt": "no-message solver prompt",
    }
    return rows, metadata


def soft_prefix_loss(
    dream: Any,
    batch: dict[str, torch.Tensor],
    prefix_embeds: torch.Tensor,
    tokenizer: Any,
    model_dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, float]]:
    prompt_ids = batch["prompt_ids"]
    target_ids = batch["target_ids"]
    mask_token_id = resolve_mask_token_id(dream, tokenizer)
    masked_target = torch.full_like(target_ids, mask_token_id)
    input_ids = torch.cat([prompt_ids, masked_target], dim=1)
    token_embeds = dream.get_input_embeddings()(input_ids)
    combined = torch.cat([prefix_embeds.to(dtype=model_dtype), token_embeds], dim=1)
    logits = dream(inputs_embeds=combined).logits
    shifted = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    prefix_len = prefix_embeds.shape[1]
    prompt_len = prompt_ids.shape[1]
    target_logits = shifted[:, prefix_len + prompt_len : prefix_len + prompt_len + target_ids.shape[1], :]
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
def evaluate(
    adapter: DreamSoftPrefixAdapter,
    dream: Any,
    loader: DataLoader,
    tokenizer: Any,
    device: torch.device,
    model_dtype: torch.dtype,
) -> dict[str, float]:
    adapter.eval()
    matched = []
    zero = []
    shuffled = []
    token_acc = []
    first_acc = []
    for batch in loader:
        batch = move_batch(batch, device)
        prefix = adapter(batch["packets"])
        loss, metrics = soft_prefix_loss(dream, batch, prefix, tokenizer, model_dtype)
        matched.append(float(loss.item()))
        token_acc.append(metrics["token_accuracy"])
        first_acc.append(metrics["first_token_accuracy"])
        zero_prefix = torch.zeros_like(prefix)
        zero_loss, _ = soft_prefix_loss(dream, batch, zero_prefix, tokenizer, model_dtype)
        zero.append(float(zero_loss.item()))
        shuffled_packets = batch["packets"].flip(dims=[1])
        shuffled_prefix = adapter(shuffled_packets)
        shuffled_loss, _ = soft_prefix_loss(dream, batch, shuffled_prefix, tokenizer, model_dtype)
        shuffled.append(float(shuffled_loss.item()))
    metrics = {
        "matched_ce": mean(matched),
        "zero_ce": mean(zero),
        "agent_swap_ce": mean(shuffled),
        "zero_ce_margin": mean(zero) - mean(matched),
        "agent_swap_ce_margin": mean(shuffled) - mean(matched),
        "token_accuracy": mean(token_acc),
        "first_token_accuracy": mean(first_acc),
        "num_rows": float(len(matched)),
    }
    metrics["selection_metric"] = -metrics["matched_ce"] + 0.05 * metrics["zero_ce_margin"] + 0.05 * metrics["agent_swap_ce_margin"]
    return metrics


def evaluate_checkpoint(
    path: Path,
    dream: Any,
    loader: DataLoader,
    tokenizer: Any,
    device: torch.device,
    model_dtype: torch.dtype,
) -> dict[str, float]:
    checkpoint = torch.load(path, map_location=device)
    config = SoftPrefixConfig(**checkpoint["config"])
    adapter = DreamSoftPrefixAdapter(config, embed_size=int(dream.get_input_embeddings().embedding_dim)).to(device)
    adapter.load_state_dict(checkpoint["model_state"])
    return evaluate(adapter, dream, loader, tokenizer, device, model_dtype)


def save_checkpoint(
    path: Path,
    adapter: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: SoftPrefixConfig,
    metadata: dict[str, Any],
    step: int,
    selection_metric: float,
) -> None:
    torch.save(
        {
            "model_state": adapter.state_dict(),
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
        "target_ids": item["target_ids"].unsqueeze(0),
    }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def resolve_mask_token_id(model: Any, tokenizer: Any) -> int:
    for value in [
        getattr(getattr(model, "generation_config", None), "mask_token_id", None),
        getattr(getattr(model, "config", None), "mask_token_id", None),
        getattr(tokenizer, "mask_token_id", None),
    ]:
        if value is not None:
            return int(value)
    token_id = tokenizer.convert_tokens_to_ids("<|mask|>")
    if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
        return int(token_id)
    raise ValueError("Cannot resolve Dream mask_token_id")


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
