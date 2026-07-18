"""Train D7 V6 receiver-side answer reranker.

The candidate pool is built from receiver-generated answers, not from private
evidence text or gold labels. Gold/scorer is used only offline to label the
best candidate. The reranker consumes D6 latent packets plus candidate answer
text and is trained to select the best candidate while keeping matched packet
scores above zero/shuffled/agent-swap controls.
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

from drla.evaluation.p2_phase_c_scorers import score_qa_answer  # noqa: E402
from drla.scripts.p3_train_dream_layer_conditioned_receiver import (  # noqa: E402
    LayerReceiverConfig,
    load_row_packets,
    load_training_rows,
    split_rows,
)
from drla.scripts.p3_train_dream_soft_prefix_adapter import DEFAULT_MODEL_PATH  # noqa: E402
from drla.scripts.run_p2_phase_c_text_agents import read_jsonl  # noqa: E402
from drla.scripts.train_cola_readiness_model import device_metadata, require_cuda_training, resolve_device  # noqa: E402
from drla.tracking import finish_experiment, init_experiment, log_metrics  # noqa: E402


DEFAULT_CANDIDATE_GENERATIONS = (
    "/data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/"
    "dream_layer_receiver_eval_v1_best200_candidates_20260607/generations.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_answer_rerankers/"
    "dream_receiver_answer_reranker_v1_seed20260607"
)


@dataclass(frozen=True)
class RerankerConfig:
    candidate_generations_jsonl: str = DEFAULT_CANDIDATE_GENERATIONS
    model_path: str = DEFAULT_MODEL_PATH
    output_dir: str = DEFAULT_OUTPUT_DIR
    device: str = "auto"
    dtype: str = "bfloat16"
    seed: int = 20260607
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    batch_size: int = 1
    epochs: int = 20
    max_train_steps: int = 0
    valid_interval: int = 10
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    input_tokens_per_agent: int = 32
    max_candidate_tokens: int = 16
    hidden_size: int = 3584
    d_model: int = 256
    num_heads: int = 4
    dropout: float = 0.1
    control_margin: float = 0.5
    control_loss_weight: float = 0.25
    swanlab_mode: str = "cloud"
    experiment_name: str = "p3-dream-receiver-answer-reranker-v1"


class RerankerDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], config: RerankerConfig) -> None:
        self.rows = rows
        self.config = config
        self.layer_config = LayerReceiverConfig(input_tokens_per_agent=config.input_tokens_per_agent)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "packets": load_row_packets(row, self.layer_config),
            "row": row,
        }


class PacketEncoder(nn.Module):
    def __init__(self, config: RerankerConfig) -> None:
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
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.out = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, config.d_model))

    def forward(self, packets: torch.Tensor) -> torch.Tensor:
        batch, num_agents, tokens, hidden = packets.shape
        x = self.input_proj(self.input_norm(packets.reshape(batch, num_agents * tokens, hidden)))
        agent_ids = torch.arange(num_agents, device=packets.device).repeat_interleave(tokens)
        x = x + self.agent_embed(agent_ids).unsqueeze(0) + self.pos_embed[:, : num_agents * tokens]
        return self.out(self.encoder(x).mean(dim=1))


class AnswerReranker(nn.Module):
    def __init__(self, config: RerankerConfig) -> None:
        super().__init__()
        self.packet_encoder = PacketEncoder(config)
        self.candidate_proj = nn.Sequential(nn.LayerNorm(config.hidden_size), nn.Linear(config.hidden_size, config.d_model))
        self.scorer = nn.Sequential(
            nn.LayerNorm(config.d_model * 4),
            nn.Linear(config.d_model * 4, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )

    def forward(self, packets: torch.Tensor, candidate_embeds: torch.Tensor) -> torch.Tensor:
        packet = self.packet_encoder(packets)
        candidates = self.candidate_proj(candidate_embeds)
        packet_expanded = packet.unsqueeze(1).expand(-1, candidates.shape[1], -1)
        features = torch.cat(
            [
                packet_expanded,
                candidates,
                packet_expanded * candidates,
                torch.abs(packet_expanded - candidates),
            ],
            dim=-1,
        )
        return self.scorer(features).squeeze(-1)


def main() -> None:
    summary = train(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> RerankerConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-generations-jsonl", default=RerankerConfig.candidate_generations_jsonl)
    parser.add_argument("--model-path", default=RerankerConfig.model_path)
    parser.add_argument("--output-dir", default=RerankerConfig.output_dir)
    parser.add_argument("--device", default=RerankerConfig.device)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=RerankerConfig.dtype)
    parser.add_argument("--seed", type=int, default=RerankerConfig.seed)
    parser.add_argument("--train-ratio", type=float, default=RerankerConfig.train_ratio)
    parser.add_argument("--valid-ratio", type=float, default=RerankerConfig.valid_ratio)
    parser.add_argument("--batch-size", type=int, default=RerankerConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=RerankerConfig.epochs)
    parser.add_argument("--max-train-steps", type=int, default=RerankerConfig.max_train_steps)
    parser.add_argument("--valid-interval", type=int, default=RerankerConfig.valid_interval)
    parser.add_argument("--learning-rate", type=float, default=RerankerConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=RerankerConfig.weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=RerankerConfig.grad_clip_norm)
    parser.add_argument("--input-tokens-per-agent", type=int, default=RerankerConfig.input_tokens_per_agent)
    parser.add_argument("--max-candidate-tokens", type=int, default=RerankerConfig.max_candidate_tokens)
    parser.add_argument("--hidden-size", type=int, default=RerankerConfig.hidden_size)
    parser.add_argument("--d-model", type=int, default=RerankerConfig.d_model)
    parser.add_argument("--num-heads", type=int, default=RerankerConfig.num_heads)
    parser.add_argument("--dropout", type=float, default=RerankerConfig.dropout)
    parser.add_argument("--control-margin", type=float, default=RerankerConfig.control_margin)
    parser.add_argument("--control-loss-weight", type=float, default=RerankerConfig.control_loss_weight)
    parser.add_argument("--swanlab-mode", default=RerankerConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=RerankerConfig.experiment_name)
    return RerankerConfig(**vars(parser.parse_args()))


def train(config: RerankerConfig) -> dict[str, Any]:
    if config.swanlab_mode != "cloud":
        raise ValueError("D7 answer reranker training must use SwanLab cloud")
    if config.valid_interval > 10:
        raise ValueError("valid_interval must be <= 10 for current P3 training")
    if config.batch_size != 1:
        raise ValueError("This script currently requires --batch-size 1")
    set_seed(config.seed)
    rng = random.Random(config.seed + 31)
    device = resolve_device(config.device)
    require_cuda_training(device, "p3_train_dream_receiver_answer_reranker.py")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    model_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[config.dtype]
    dream = AutoModel.from_pretrained(config.model_path, torch_dtype=model_dtype, trust_remote_code=True).to(device).eval()
    for param in dream.parameters():
        param.requires_grad_(False)

    rows, metadata = build_rows(config)
    splits = split_rows(rows, config.seed, config.train_ratio, config.valid_ratio)
    datasets = {name: RerankerDataset(items, config) for name, items in splits.items()}
    loaders = {name: DataLoader(dataset, batch_size=1, shuffle=(name == "train"), collate_fn=collate_one) for name, dataset in datasets.items()}
    reranker = AnswerReranker(config).to(device)
    optimizer = torch.optim.AdamW(reranker.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    run = init_experiment(
        stage="p3-dream-receiver-answer-reranker",
        config={
            **asdict(config),
            **device_metadata(device),
            "metadata": metadata,
            "split_sizes": {name: len(items) for name, items in splits.items()},
        },
        experiment_name=config.experiment_name,
        tags=["dream", "p3", "latentmas", "answer-reranker", "swanlab-cloud"],
        mode=config.swanlab_mode,
    )

    metrics_path = output_dir / "metrics.jsonl"
    metrics_f = metrics_path.open("w", encoding="utf-8")
    best_metric = float("-inf")
    best_step = 0
    global_step = 0
    try:
        for epoch in range(config.epochs):
            reranker.train()
            for batch in loaders["train"]:
                global_step += 1
                row = batch["row"]
                packets = batch["packets"].to(device)
                candidate_embeds = embed_candidates(dream, tokenizer, row["candidates"], config, device)
                labels = torch.tensor([row["target_index"]], device=device)
                optimizer.zero_grad(set_to_none=True)
                logits = reranker(packets, candidate_embeds)
                ce = F.cross_entropy(logits, labels)
                corrupt_loss, corrupt_metrics = corruption_margin_loss(
                    reranker, dream, tokenizer, row, packets, candidate_embeds, logits, config, rng, datasets["train"], device
                )
                loss = ce + config.control_loss_weight * corrupt_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(reranker.parameters(), config.grad_clip_norm)
                optimizer.step()
                train_metrics = {
                    "loss": float(loss.detach().item()),
                    "ce": float(ce.detach().item()),
                    "corruption_loss": float(corrupt_loss.detach().item()),
                    "epoch": float(epoch),
                    **corrupt_metrics,
                }
                write_metrics(metrics_f, "train", global_step, train_metrics)
                log_metrics(train_metrics, step=global_step, prefix="train")
                if global_step % config.valid_interval == 0:
                    valid_metrics = evaluate(reranker, dream, tokenizer, datasets["valid"], config, device)
                    write_metrics(metrics_f, "valid", global_step, valid_metrics)
                    log_metrics(valid_metrics, step=global_step, prefix="valid")
                    if valid_metrics["selection_metric"] > best_metric:
                        best_metric = valid_metrics["selection_metric"]
                        best_step = global_step
                        save_checkpoint(output_dir / "best_checkpoint.pt", reranker, optimizer, config, metadata, best_step, best_metric)
                if config.max_train_steps and global_step >= config.max_train_steps:
                    break
            if config.max_train_steps and global_step >= config.max_train_steps:
                break
        final_valid_metrics = evaluate(reranker, dream, tokenizer, datasets["valid"], config, device)
        final_test_metrics = evaluate(reranker, dream, tokenizer, datasets["test"], config, device)
        write_metrics(metrics_f, "valid", global_step, final_valid_metrics)
        write_metrics(metrics_f, "test", global_step, final_test_metrics)
        log_metrics(final_valid_metrics, step=global_step, prefix="valid")
        log_metrics(final_test_metrics, step=global_step, prefix="test")
        if final_valid_metrics["selection_metric"] > best_metric:
            best_metric = final_valid_metrics["selection_metric"]
            best_step = global_step
            save_checkpoint(output_dir / "best_checkpoint.pt", reranker, optimizer, config, metadata, best_step, best_metric)
        save_checkpoint(output_dir / "last_checkpoint.pt", reranker, optimizer, config, metadata, global_step, final_valid_metrics["selection_metric"])
    finally:
        metrics_f.close()
        finish_experiment()

    best_valid_metrics = evaluate_checkpoint(output_dir / "best_checkpoint.pt", dream, tokenizer, datasets["valid"], device)
    best_test_metrics = evaluate_checkpoint(output_dir / "best_checkpoint.pt", dream, tokenizer, datasets["test"], device)
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
            "P3 D7 V6 answer-reranker deep-learning training",
            "CUDA/GPU required",
            "SwanLab cloud required",
            "candidate pool comes from receiver generations, not private evidence text",
            "gold/scorer used only for offline target labels and metrics",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def build_rows(config: RerankerConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_config = LayerReceiverConfig(input_tokens_per_agent=config.input_tokens_per_agent)
    base_rows, base_meta = load_training_rows(base_config)
    base_by_row = {str(row["row_id"]): row for row in base_rows}
    aliases_by_sample = load_aliases_by_sample(Path(base_config.manifest_json))
    generated = read_jsonl(Path(config.candidate_generations_jsonl))
    by_row: dict[str, list[dict[str, Any]]] = {}
    for item in generated:
        by_row.setdefault(str(item["row_id"]), []).append(item)
    rows = []
    oracle_primary = []
    oracle_token_f1 = []
    candidate_sizes = []
    for row_id, items in sorted(by_row.items()):
        base = base_by_row.get(row_id)
        if base is None:
            continue
        candidates = unique_candidates([str(item.get("prediction", "")).strip() for item in items])
        if not candidates:
            continue
        aliases = aliases_by_sample.get(str(base.get("sample_id", "")), [])
        scored = [
            score_qa_answer(candidate, base["gold_answer"], aliases).to_dict()
            for candidate in candidates
        ]
        utilities = [float(score["primary_score"]) + 0.1 * float(score["token_f1"]) for score in scored]
        target_index = max(range(len(candidates)), key=lambda idx: utilities[idx])
        row = {
            **base,
            "candidates": candidates,
            "candidate_scores": scored,
            "target_index": target_index,
            "oracle_primary": max(float(score["primary_score"]) for score in scored),
            "oracle_token_f1": max(float(score["token_f1"]) for score in scored),
        }
        rows.append(row)
        oracle_primary.append(row["oracle_primary"])
        oracle_token_f1.append(row["oracle_token_f1"])
        candidate_sizes.append(len(candidates))
    metadata = {
        **base_meta,
        "candidate_generations_jsonl": config.candidate_generations_jsonl,
        "num_candidate_rows": len(rows),
        "candidate_source": "receiver-generated predictions only",
        "candidate_label_alias_source": str(base_config.manifest_json),
        "oracle_primary_mean": mean(oracle_primary),
        "oracle_token_f1_mean": mean(oracle_token_f1),
        "candidate_size_mean": mean(candidate_sizes),
        "candidate_condition_counts": dict(Counter(str(item.get("condition", "")) for item in generated)),
    }
    if not rows:
        raise ValueError("no reranker rows built from candidate generations")
    return rows, metadata


def load_aliases_by_sample(manifest_path: Path) -> dict[str, list[str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aliases: dict[str, list[str]] = {}
    for sample in manifest.get("samples", []):
        scoring = sample.get("scoring", {}) or {}
        values = scoring.get("answer_aliases", []) or []
        aliases[str(sample.get("sample_id", ""))] = [str(value) for value in values if str(value).strip()]
    return aliases


def unique_candidates(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        cleaned = " ".join(value.split()).strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def embed_candidates(
    dream: Any,
    tokenizer: Any,
    candidates: list[str],
    config: RerankerConfig,
    device: torch.device,
) -> torch.Tensor:
    embeds = []
    embedding = dream.get_input_embeddings()
    with torch.no_grad():
        for candidate in candidates:
            ids = tokenizer(candidate, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            ids = ids[:, : config.max_candidate_tokens]
            if ids.numel() == 0:
                ids = torch.tensor([[getattr(tokenizer, "eos_token_id", 0) or 0]], device=device)
            embeds.append(embedding(ids).float().mean(dim=1).squeeze(0))
    return torch.stack(embeds, dim=0).unsqueeze(0)


def corruption_margin_loss(
    reranker: AnswerReranker,
    dream: Any,
    tokenizer: Any,
    row: dict[str, Any],
    packets: torch.Tensor,
    candidate_embeds: torch.Tensor,
    matched_logits: torch.Tensor,
    config: RerankerConfig,
    rng: random.Random,
    train_dataset: RerankerDataset,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    target = int(row["target_index"])
    matched_score = matched_logits[0, target].detach()
    zero_logits = reranker(torch.zeros_like(packets), candidate_embeds)
    swap_logits = reranker(packets.flip(dims=[1]), candidate_embeds)
    shuffled_item = train_dataset[rng.randrange(len(train_dataset))]
    shuffled_packets = shuffled_item["packets"].unsqueeze(0).to(device)
    shuffled_logits = reranker(shuffled_packets, candidate_embeds)
    losses = {
        "zero": F.relu(zero_logits[0, target] + config.control_margin - matched_score),
        "agent_swap": F.relu(swap_logits[0, target] + config.control_margin - matched_score),
        "shuffled_row": F.relu(shuffled_logits[0, target] + config.control_margin - matched_score),
    }
    loss = sum(losses.values()) / len(losses)
    metrics = {}
    for name, value in losses.items():
        metrics[f"{name}_margin_violation"] = float(value.detach().item())
    return loss, metrics


@torch.no_grad()
def evaluate(
    reranker: AnswerReranker,
    dream: Any,
    tokenizer: Any,
    dataset: RerankerDataset,
    config: RerankerConfig,
    device: torch.device,
) -> dict[str, float]:
    reranker.eval()
    conditions = ["matched", "zero", "agent_swap", "shuffled_row"]
    primary: dict[str, list[float]] = {name: [] for name in conditions}
    token_f1: dict[str, list[float]] = {name: [] for name in conditions}
    selected_is_target: dict[str, list[float]] = {name: [] for name in conditions}
    oracle_primary = []
    oracle_token_f1 = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        row = item["row"]
        packets = item["packets"].unsqueeze(0).to(device)
        candidate_embeds = embed_candidates(dream, tokenizer, row["candidates"], config, device)
        packet_variants = {
            "matched": packets,
            "zero": torch.zeros_like(packets),
            "agent_swap": packets.flip(dims=[1]),
            "shuffled_row": dataset[(idx + 1) % len(dataset)]["packets"].unsqueeze(0).to(device),
        }
        oracle_primary.append(float(row["oracle_primary"]))
        oracle_token_f1.append(float(row["oracle_token_f1"]))
        for name, variant in packet_variants.items():
            logits = reranker(variant, candidate_embeds)
            selected = int(logits.argmax(dim=-1).item())
            score = row["candidate_scores"][selected]
            primary[name].append(float(score["primary_score"]))
            token_f1[name].append(float(score["token_f1"]))
            selected_is_target[name].append(float(selected == int(row["target_index"])))
    metrics = {
        "num_rows": float(len(dataset)),
        "oracle_primary": mean(oracle_primary),
        "oracle_token_f1": mean(oracle_token_f1),
    }
    for name in conditions:
        metrics[f"{name}_primary"] = mean(primary[name])
        metrics[f"{name}_token_f1"] = mean(token_f1[name])
        metrics[f"{name}_target_acc"] = mean(selected_is_target[name])
    metrics["matched_minus_zero_primary"] = metrics["matched_primary"] - metrics["zero_primary"]
    metrics["matched_minus_shuffled_primary"] = metrics["matched_primary"] - metrics["shuffled_row_primary"]
    metrics["matched_minus_agent_swap_primary"] = metrics["matched_primary"] - metrics["agent_swap_primary"]
    metrics["selection_metric"] = (
        metrics["matched_primary"]
        + 0.1 * metrics["matched_token_f1"]
        + metrics["matched_minus_zero_primary"]
        + metrics["matched_minus_shuffled_primary"]
        + 0.5 * metrics["matched_minus_agent_swap_primary"]
    )
    return metrics


def evaluate_checkpoint(path: Path, dream: Any, tokenizer: Any, dataset: RerankerDataset, device: torch.device) -> dict[str, float]:
    checkpoint = torch.load(path, map_location=device)
    config = RerankerConfig(**checkpoint["config"])
    reranker = AnswerReranker(config).to(device)
    reranker.load_state_dict(checkpoint["model_state"])
    return evaluate(reranker, dream, tokenizer, dataset, config, device)


def save_checkpoint(
    path: Path,
    reranker: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: RerankerConfig,
    metadata: dict[str, Any],
    step: int,
    selection_metric: float,
) -> None:
    torch.save(
        {
            "model_state": reranker.state_dict(),
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
    return {"packets": items[0]["packets"].unsqueeze(0), "row": items[0]["row"]}


def write_metrics(handle, phase: str, step: int, metrics: dict[str, float]) -> None:
    handle.write(json.dumps({"phase": phase, "step": step, **metrics}, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def mean(values: list[float] | list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
