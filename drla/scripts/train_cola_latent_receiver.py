"""Train P2 single-handoff latent receiver compatibility model.

P2-C's sparse accept/defer target is not suitable as the main proof of
receiver readability on the locked packet set.  This training script instead
uses a balanced decoder-free compatibility objective: matched packet payloads
are positives, while metadata-only, shuffled, cross-task, wrong-block, noised,
and rotated payloads are negatives.  The receiver never receives decoded text,
gold answers, scorer outputs, or the control type.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from drla.scripts.audit_cola_agent_latent_packet_distribution import (
    DEFAULT_CONTROL_TYPES,
    ShardCache,
    build_control_blocks,
    build_packet_indexes,
    load_packet_blocks,
    normalize_control_types,
)
from drla.scripts.train_cola_readiness_model import (
    binary_accuracy,
    binary_auprc,
    binary_auroc,
    device_metadata,
    pos_weight,
    require_cuda_training,
    resolve_device,
)
from drla.tracking import finish_experiment, init_experiment, log_metrics


OFFICIAL_COLA_TASKS = [
    "lambada",
    "mmlu",
    "obqa",
    "hellaswag",
    "race",
    "siqa",
    "squad",
    "story_cloze",
]

PROCESS_FEATURE_FIELDS = [
    "block_number",
    "max_block_budget",
    "remaining_blocks",
    "block_fraction",
    "latent_norm_mean",
    "latent_norm_std",
    "latent_delta_norm",
    "latent_delta_missing",
    "latent_cosine_to_prev",
    "latent_cosine_missing",
    "denoise_drift_norm_mean",
]

READINESS_SCORE_FIELDS = [
    "readiness",
    "correctness",
    "future_gain",
    "decoder_stop",
    "prediction_change",
    "completion_risk",
    "contentful",
    "answer_identity_stability",
]

READINESS_MARGIN_FIELDS = [
    "readiness",
    "prediction_change",
    "completion_risk",
    "contentful",
    "answer_identity_stability",
]

READINESS_THRESHOLD_FIELDS = [
    "readiness",
    "prediction_change",
    "completion_risk",
    "contentful",
    "answer_identity_stability",
    "correctness",
    "decoder_stop",
    "empty_answer_risk",
    "answer_format_risk",
]

RISK_CERTIFICATE_FIELDS = [
    "calibration_joint_risk_satisfied",
    "loss_risk_satisfied",
    "mismatch_risk_satisfied",
    "loss_upper_max",
    "mismatch_upper_max",
    "loss_risk_target",
    "mismatch_risk_target",
    "risk_bound_z",
]

INPUT_MODES = {
    "envelope_only",
    "process_only",
    "certificate_only",
    "latent_only",
    "latent_process",
    "latent_process_certificate",
    "latent_process_certificate_no_task",
}

SELECTION_METRICS = {
    "compatibility_auroc",
    "compatibility_auprc",
    "compatibility_mean_control_auroc",
    "compatibility_mean_control_gap",
}


@dataclass(frozen=True)
class LatentReceiverTrainConfig:
    packets_jsonl: str = (
        "/data1/luyifei/drla/outputs/cola_agent_latent_comm/"
        "p2_agent_latent_comm_v2_locked_seed66_67_68_split20260601_20260529/"
        "agent_latent_comm_packets_test.jsonl"
    )
    output_dir: str = (
        "/data1/luyifei/drla/outputs/cola_latent_receiver/"
        "p2c_receiver_compatibility_v1"
    )
    input_mode: str = "latent_process_certificate"
    control_types: str = ",".join(DEFAULT_CONTROL_TYPES)
    seed: int = 20260529
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    max_packets: int = 0
    batch_size: int = 512
    epochs: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    d_model: int = 64
    attention_heads: int = 4
    inter_layers: int = 2
    valid_interval: int = 10
    selection_metric: str = "compatibility_mean_control_auroc"
    max_cached_shards: int = 1024
    noise_std: float = 1.0
    num_workers: int = 0
    device: str = "auto"
    swanlab_mode: str = "cloud"
    experiment_name: str = "p2c-latent-receiver-compatibility-v1"


class LatentReceiverCompatibilityModel(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        process_dim: int,
        envelope_dim: int,
        certificate_dim: int,
        max_blocks: int,
        block_size: int,
        task_count: int,
        d_model: int,
        attention_heads: int,
        inter_layers: int,
        dropout: float,
        input_mode: str,
    ) -> None:
        super().__init__()
        if d_model % attention_heads != 0:
            raise ValueError("d_model must be divisible by attention_heads")
        if input_mode not in INPUT_MODES:
            raise ValueError(f"unknown input_mode: {input_mode}")
        self.input_mode = input_mode
        self.max_blocks = max_blocks
        self.block_size = block_size
        self.use_latent = "latent" in input_mode
        self.use_process = "process" in input_mode
        self.use_certificate = "certificate" in input_mode
        self.use_task = not input_mode.endswith("_no_task")

        if self.use_latent:
            self.slot_norm = nn.LayerNorm(latent_dim)
            self.slot_adapter = nn.Linear(latent_dim, d_model)
            self.slot_pos = nn.Embedding(block_size, d_model)
            self.block_pos = nn.Embedding(max_blocks, d_model)
            self.intra_block = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=attention_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
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
            self.latent_attn = nn.MultiheadAttention(
                d_model,
                attention_heads,
                dropout=dropout,
                batch_first=True,
            )
        if self.use_process:
            self.process_mlp = nn.Sequential(
                nn.LayerNorm(process_dim),
                nn.Linear(process_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
        if input_mode == "envelope_only":
            self.envelope_mlp = nn.Sequential(
                nn.LayerNorm(envelope_dim),
                nn.Linear(envelope_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
        if self.use_certificate:
            self.certificate_mlp = nn.Sequential(
                nn.LayerNorm(certificate_dim),
                nn.Linear(certificate_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
        if self.use_task:
            self.task_embedding = nn.Embedding(task_count, d_model)

        component_count = 0
        component_count += int(self.use_latent)
        component_count += int(self.use_process)
        component_count += int(input_mode == "envelope_only")
        component_count += int(self.use_certificate)
        component_count += int(self.use_task)
        if component_count == 0:
            raise ValueError("input_mode disables all inputs")
        self.head = nn.Sequential(
            nn.LayerNorm(component_count * d_model),
            nn.Linear(component_count * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        latent_blocks: torch.Tensor,
        process_features: torch.Tensor,
        block_mask: torch.Tensor,
        envelope_features: torch.Tensor,
        certificate_features: torch.Tensor,
        task_idx: torch.Tensor,
    ) -> torch.Tensor:
        components = []
        if self.use_latent:
            components.append(self.encode_latent(latent_blocks, block_mask))
        if self.use_process:
            components.append(masked_mean(self.process_mlp(process_features), block_mask))
        if self.input_mode == "envelope_only":
            components.append(self.envelope_mlp(envelope_features))
        if self.use_certificate:
            components.append(self.certificate_mlp(certificate_features))
        if self.use_task:
            components.append(self.task_embedding(task_idx))
        return self.head(torch.cat(components, dim=-1)).squeeze(-1)

    def encode_latent(self, latent_blocks: torch.Tensor, block_mask: torch.Tensor) -> torch.Tensor:
        batch_size, max_blocks, block_size, _ = latent_blocks.shape
        device = latent_blocks.device
        slot_pos = self.slot_pos(torch.arange(block_size, device=device)).view(1, 1, block_size, -1)
        block_pos = self.block_pos(torch.arange(max_blocks, device=device)).view(1, max_blocks, 1, -1)
        tokens = self.slot_adapter(self.slot_norm(latent_blocks)) + slot_pos + block_pos
        flat = tokens.reshape(batch_size * max_blocks, block_size, -1)
        encoded = self.intra_block(flat).reshape(batch_size, max_blocks, block_size, -1)
        block_summary = encoded.mean(dim=2)
        block_summary = self.inter_block(block_summary, src_key_padding_mask=~block_mask.bool())
        query = self.latent_query.unsqueeze(0).expand(batch_size, -1, -1)
        pooled, _ = self.latent_attn(
            query,
            block_summary,
            block_summary,
            key_padding_mask=~block_mask.bool(),
            need_weights=False,
        )
        return pooled.squeeze(1)


def train_latent_receiver(config: LatentReceiverTrainConfig) -> dict[str, Any]:
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
    summary_path = output_dir / "summary.json"

    packets = load_packets(Path(config.packets_jsonl), config.max_packets)
    if not packets:
        raise ValueError("no packets loaded")
    splits = split_packet_indices(packets, config)
    tensors_by_split, metadata = build_receiver_tensors(packets, splits, config)
    train_data, valid_data, test_data, norm_stats = make_datasets(tensors_by_split)

    device = resolve_device(config.device)
    require_cuda_training(device, "train_cola_latent_receiver.py")
    device_info = device_metadata(device)
    model = LatentReceiverCompatibilityModel(
        latent_dim=metadata["latent_dim"],
        process_dim=metadata["process_dim"],
        envelope_dim=metadata["envelope_dim"],
        certificate_dim=metadata["certificate_dim"],
        max_blocks=metadata["max_blocks"],
        block_size=metadata["block_size"],
        task_count=len(OFFICIAL_COLA_TASKS),
        d_model=config.d_model,
        attention_heads=config.attention_heads,
        inter_layers=config.inter_layers,
        dropout=config.dropout,
        input_mode=config.input_mode,
    ).to(device)
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
    )
    valid_loader = DataLoader(valid_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    target_weight = pos_weight(tensors_by_split["train"]["label"]).to(device)

    run = init_experiment(
        stage="cola-latent-receiver",
        experiment_name=config.experiment_name,
        description="P2-C same-substrate latent receiver compatibility diagnostic.",
        config={
            **asdict(config),
            "device_info": device_info,
            "online_input_policy": metadata["online_input_policy"],
            "controls": metadata["control_types"],
            "split_sizes": metadata["split_sizes"],
            "example_counts": metadata["example_counts"],
        },
        mode=config.swanlab_mode,
        tags=["cola", "official-benchmark", "p2", "latent-receiver", "compatibility"],
    )

    best_metric = -math.inf
    best_step = 0
    global_step = 0
    history: list[dict[str, Any]] = []
    start_time = time.time()

    try:
        with metrics_path.open("w", encoding="utf-8") as metrics_f:
            for epoch in range(config.epochs):
                model.train()
                for batch in train_loader:
                    global_step += 1
                    batch = [item.to(device) for item in batch]
                    loss, batch_metrics = compute_loss(model, batch, target_weight)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_metrics = {
                        "loss": float(loss.item()),
                        **{f"train_{key}": value for key, value in batch_metrics.items()},
                        "epoch": float(epoch),
                    }
                    log_metrics(train_metrics, step=global_step, prefix="train")
                    write_metric(metrics_f, "train", global_step, train_metrics)
                    if global_step % config.valid_interval == 0:
                        valid_metrics = evaluate(model, valid_loader, device, metadata["control_types"])
                        log_metrics(valid_metrics, step=global_step, prefix="valid")
                        write_metric(metrics_f, "valid", global_step, valid_metrics)
                        selected = select_metric(valid_metrics, config.selection_metric)
                        if selected > best_metric:
                            best_metric = selected
                            best_step = global_step
                            save_checkpoint(
                                checkpoint_dir / "best_checkpoint.pt",
                                model=model,
                                optimizer=optimizer,
                                config=config,
                                norm_stats=norm_stats,
                                metadata=metadata,
                                step=global_step,
                                metric=best_metric,
                            )
                        history.append({"step": global_step, "valid": valid_metrics})

            valid_metrics = evaluate(model, valid_loader, device, metadata["control_types"])
            test_metrics = evaluate(model, test_loader, device, metadata["control_types"])
            log_metrics(valid_metrics, step=global_step, prefix="valid")
            log_metrics(test_metrics, step=global_step, prefix="test")
            write_metric(metrics_f, "valid", global_step, valid_metrics)
            write_metric(metrics_f, "test", global_step, test_metrics)
            selected = select_metric(valid_metrics, config.selection_metric)
            if selected > best_metric:
                best_metric = selected
                best_step = global_step
                save_checkpoint(
                    checkpoint_dir / "best_checkpoint.pt",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    norm_stats=norm_stats,
                    metadata=metadata,
                    step=global_step,
                    metric=best_metric,
                )
            save_checkpoint(
                checkpoint_dir / "last_checkpoint.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                norm_stats=norm_stats,
                metadata=metadata,
                step=global_step,
                metric=selected,
            )

        summary = {
            "created_at": int(time.time()),
            "config": asdict(config),
            "device_info": device_info,
            "online_input_policy": metadata["online_input_policy"],
            "control_types": metadata["control_types"],
            "split_sizes": metadata["split_sizes"],
            "example_counts": metadata["example_counts"],
            "best_step": best_step,
            "best_metric_name": f"valid/{config.selection_metric}",
            "best_metric": best_metric,
            "last_valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
            "history": history,
            "metrics_jsonl": str(metrics_path),
            "best_checkpoint": str(checkpoint_dir / "best_checkpoint.pt"),
            "last_checkpoint": str(checkpoint_dir / "last_checkpoint.pt"),
            "elapsed_seconds": time.time() - start_time,
            "swanlab_run_id": getattr(run, "id", None),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary
    finally:
        finish_experiment()


def validate_config(config: LatentReceiverTrainConfig) -> None:
    if config.swanlab_mode != "cloud":
        raise ValueError("all deep-learning training experiments must use SwanLab cloud")
    if config.valid_interval > 10:
        raise ValueError("valid_interval must be <= 10 steps")
    if config.input_mode not in INPUT_MODES:
        raise ValueError("unknown input_mode")
    if config.selection_metric not in SELECTION_METRICS:
        raise ValueError("unknown selection_metric")
    if config.d_model % config.attention_heads != 0:
        raise ValueError("d_model must be divisible by attention_heads")
    if not 0 < config.train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0 <= config.valid_ratio < 1:
        raise ValueError("valid_ratio must be in [0, 1)")
    if config.train_ratio + config.valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio must leave a test split")
    if config.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if config.max_packets < 0:
        raise ValueError("max_packets must be non-negative")
    if not Path(config.packets_jsonl).exists():
        raise FileNotFoundError(config.packets_jsonl)
    normalize_control_types(config.control_types)


def load_packets(path: Path, max_packets: int) -> list[dict[str, Any]]:
    packets = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            packets.append(json.loads(line))
            if max_packets and len(packets) >= max_packets:
                break
    return packets


def split_packet_indices(packets: list[dict[str, Any]], config: LatentReceiverTrainConfig) -> dict[str, list[int]]:
    splits = {"train": [], "valid": [], "test": []}
    for index, packet in enumerate(packets):
        key = str(packet["sample_key"])
        value = stable_uniform(f"{config.seed}:{key}")
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


def build_receiver_tensors(
    packets: list[dict[str, Any]],
    splits: dict[str, list[int]],
    config: LatentReceiverTrainConfig,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    control_types = normalize_control_types(config.control_types)
    first_block = packets[0]["latent_memory"]["blocks"][0]
    max_blocks = max(int(packet["agent_a"]["max_block_budget"]) for packet in packets)
    block_size = int(first_block["latent_ref"]["shape"][0])
    latent_dim = int(first_block["latent_ref"]["shape"][1])
    process_dim = len(PROCESS_FEATURE_FIELDS)
    envelope_dim = 4
    certificate_dim = certificate_feature_dim()
    task_to_idx = {task: idx for idx, task in enumerate(OFFICIAL_COLA_TASKS)}
    tensors_by_split = {}
    example_counts = {}
    warnings: dict[str, list[dict[str, Any]]] = {}

    for split_name, indices in splits.items():
        split_packets = [packets[index] for index in indices]
        split_packet_indexes = build_packet_indexes(split_packets)
        split_tensors, split_warnings = build_split_tensors(
            split_packets=split_packets,
            control_types=control_types,
            task_to_idx=task_to_idx,
            max_blocks=max_blocks,
            block_size=block_size,
            latent_dim=latent_dim,
            process_dim=process_dim,
            envelope_dim=envelope_dim,
            certificate_dim=certificate_dim,
            config=config,
            packet_indexes=split_packet_indexes,
        )
        tensors_by_split[split_name] = split_tensors
        example_counts[split_name] = int(split_tensors["label"].numel())
        warnings[split_name] = split_warnings[:50]

    metadata = {
        "control_types": control_types,
        "process_feature_fields": PROCESS_FEATURE_FIELDS,
        "readiness_score_fields": READINESS_SCORE_FIELDS,
        "readiness_margin_fields": READINESS_MARGIN_FIELDS,
        "readiness_threshold_fields": READINESS_THRESHOLD_FIELDS,
        "risk_certificate_fields": RISK_CERTIFICATE_FIELDS,
        "task_to_idx": task_to_idx,
        "max_blocks": max_blocks,
        "block_size": block_size,
        "latent_dim": latent_dim,
        "process_dim": process_dim,
        "envelope_dim": envelope_dim,
        "certificate_dim": certificate_dim,
        "split_sizes": {name: len(indices) for name, indices in splits.items()},
        "example_counts": example_counts,
        "control_generation_warnings": warnings,
        "online_input_policy": (
            "Decoder-free compatibility receiver. Online inputs are selected by input_mode from "
            "latent block tensors, process features, readiness/risk certificate numbers, envelope "
            "features, and optional task id. Decoded text, token ids, gold answers, scorer outputs, "
            "selected/final correctness, and control_type are never model inputs."
        ),
    }
    return tensors_by_split, metadata


def build_split_tensors(
    *,
    split_packets: list[dict[str, Any]],
    control_types: list[str],
    task_to_idx: dict[str, int],
    max_blocks: int,
    block_size: int,
    latent_dim: int,
    process_dim: int,
    envelope_dim: int,
    certificate_dim: int,
    config: LatentReceiverTrainConfig,
    packet_indexes: dict[str, dict[Any, list[int]]],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    example_count = len(split_packets) * len(control_types)
    latent_blocks = torch.zeros(example_count, max_blocks, block_size, latent_dim, dtype=torch.float32)
    process_features = torch.zeros(example_count, max_blocks, process_dim, dtype=torch.float32)
    block_mask = torch.zeros(example_count, max_blocks, dtype=torch.bool)
    envelope_features = torch.zeros(example_count, envelope_dim, dtype=torch.float32)
    certificate_features = torch.zeros(example_count, certificate_dim, dtype=torch.float32)
    task_idx = torch.zeros(example_count, dtype=torch.long)
    labels = torch.zeros(example_count, dtype=torch.float32)
    control_type_idx = torch.zeros(example_count, dtype=torch.long)

    rng = random.Random(config.seed + len(split_packets))
    torch_generator = torch.Generator(device="cpu").manual_seed(config.seed + len(split_packets))
    shard_cache = ShardCache(config.max_cached_shards)
    rotation_mats: dict[tuple[int, int], torch.Tensor] = {}
    warnings: list[dict[str, Any]] = []
    write_idx = 0

    for packet_index, packet in enumerate(split_packets):
        matched_blocks = load_packet_blocks(packet, shard_cache)
        packet_process = packet_process_tensor(packet, max_blocks, process_dim)
        packet_mask = packet_block_mask(packet, max_blocks)
        packet_envelope = torch.tensor(envelope_feature_values(packet), dtype=torch.float32)
        packet_certificate = torch.tensor(certificate_feature_values(packet), dtype=torch.float32)
        for control_index, control_type in enumerate(control_types):
            if control_type == "matched":
                control_blocks = matched_blocks
                label = 1.0
            else:
                control_blocks, _, warning = build_control_blocks(
                    control_type=control_type,
                    packet_index=packet_index,
                    packet=packet,
                    packets=split_packets,
                    packet_indexes=packet_indexes,
                    matched_blocks=matched_blocks,
                    shard_cache=shard_cache,
                    rng=rng,
                    torch_generator=torch_generator,
                    noise_std=config.noise_std,
                    rotation_mats=rotation_mats,
                )
                label = 0.0
                if warning is not None and len(warnings) < 100:
                    warnings.append(warning)
            if control_blocks is not None:
                for block_offset, block in enumerate(control_blocks):
                    latent_blocks[write_idx, block_offset] = block
            process_features[write_idx] = packet_process
            block_mask[write_idx] = packet_mask
            envelope_features[write_idx] = packet_envelope
            certificate_features[write_idx] = packet_certificate
            task_idx[write_idx] = task_to_idx[str(packet["task"])]
            labels[write_idx] = label
            control_type_idx[write_idx] = control_index
            write_idx += 1

    return {
        "latent_blocks": latent_blocks,
        "process_features": process_features,
        "block_mask": block_mask,
        "envelope_features": envelope_features,
        "certificate_features": certificate_features,
        "task_idx": task_idx,
        "label": labels,
        "control_type_idx": control_type_idx,
    }, warnings


def packet_process_tensor(packet: dict[str, Any], max_blocks: int, process_dim: int) -> torch.Tensor:
    result = torch.zeros(max_blocks, process_dim, dtype=torch.float32)
    for block in packet["latent_memory"]["blocks"]:
        block_index = int(block["block_number"]) - 1
        features = block["process_features"]
        result[block_index] = torch.tensor(
            [safe_float(features.get(field)) for field in PROCESS_FEATURE_FIELDS],
            dtype=torch.float32,
        )
    return result


def packet_block_mask(packet: dict[str, Any], max_blocks: int) -> torch.Tensor:
    mask = torch.zeros(max_blocks, dtype=torch.bool)
    mask[: int(packet["latent_memory"]["block_count"])] = True
    return mask


def envelope_feature_values(packet: dict[str, Any]) -> list[float]:
    selected = safe_float(packet["agent_a"].get("selected_block"))
    max_budget = max(safe_float(packet["agent_a"].get("max_block_budget")), 1.0)
    return [
        selected / max_budget,
        safe_float(packet["latent_memory"].get("block_count")) / max_budget,
        max_budget / 4.0,
        1.0 if packet.get("protocol_version") == "cola_agent_latent_comm_v2" else 0.0,
    ]


def certificate_feature_values(packet: dict[str, Any]) -> list[float]:
    readiness = packet["readiness_state"]
    risk = packet["risk_certificate"]
    values: list[float] = []
    for field in READINESS_SCORE_FIELDS:
        values.append(safe_float(readiness.get("scores", {}).get(field)))
    for field in READINESS_MARGIN_FIELDS:
        values.append(safe_float(readiness.get("margins", {}).get(field)))
    for field in READINESS_THRESHOLD_FIELDS:
        threshold = readiness.get("thresholds", {}).get(field)
        values.append(safe_float(threshold))
        values.append(1.0 if threshold is None else 0.0)
    for field in RISK_CERTIFICATE_FIELDS:
        values.append(bool_or_float(risk.get(field)))
    values.extend(envelope_feature_values(packet))
    return values


def certificate_feature_dim() -> int:
    return (
        len(READINESS_SCORE_FIELDS)
        + len(READINESS_MARGIN_FIELDS)
        + 2 * len(READINESS_THRESHOLD_FIELDS)
        + len(RISK_CERTIFICATE_FIELDS)
        + 4
    )


def make_datasets(
    tensors_by_split: dict[str, dict[str, torch.Tensor]],
) -> tuple[TensorDataset, TensorDataset, TensorDataset, dict[str, torch.Tensor]]:
    train = tensors_by_split["train"]
    process_mean, process_std = masked_feature_stats(train["process_features"], train["block_mask"])
    cert_mean = train["certificate_features"].mean(dim=0, keepdim=True)
    cert_std = train["certificate_features"].std(dim=0, keepdim=True).clamp_min(1e-6)
    env_mean = train["envelope_features"].mean(dim=0, keepdim=True)
    env_std = train["envelope_features"].std(dim=0, keepdim=True).clamp_min(1e-6)
    norm_stats = {
        "process_mean": process_mean,
        "process_std": process_std,
        "certificate_mean": cert_mean,
        "certificate_std": cert_std,
        "envelope_mean": env_mean,
        "envelope_std": env_std,
    }

    def dataset(split_name: str) -> TensorDataset:
        tensors = tensors_by_split[split_name]
        process = (tensors["process_features"] - process_mean.view(1, 1, -1)) / process_std.view(1, 1, -1)
        process = process.masked_fill(~tensors["block_mask"].unsqueeze(-1), 0.0)
        certificate = (tensors["certificate_features"] - cert_mean) / cert_std
        envelope = (tensors["envelope_features"] - env_mean) / env_std
        return TensorDataset(
            tensors["latent_blocks"],
            process,
            tensors["block_mask"],
            envelope,
            certificate,
            tensors["task_idx"],
            tensors["label"],
            tensors["control_type_idx"],
        )

    return dataset("train"), dataset("valid"), dataset("test"), norm_stats


def masked_feature_stats(features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    active = features[mask.bool()]
    mean = active.mean(dim=0, keepdim=True)
    std = active.std(dim=0, keepdim=True).clamp_min(1e-6)
    return mean, std


def compute_loss(
    model: LatentReceiverCompatibilityModel,
    batch: list[torch.Tensor],
    target_weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    latent_blocks, process_features, block_mask, envelope_features, certificate_features, task_idx, labels, _ = batch
    logits = model(
        latent_blocks,
        process_features,
        block_mask,
        envelope_features,
        certificate_features,
        task_idx,
    )
    loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=target_weight)
    with torch.no_grad():
        prob = torch.sigmoid(logits).detach().cpu()
        target = labels.detach().cpu()
        metrics = binary_metrics(prob, target)
    return loss, metrics


@torch.no_grad()
def evaluate(
    model: LatentReceiverCompatibilityModel,
    loader: DataLoader,
    device: torch.device,
    control_types: list[str],
) -> dict[str, float]:
    model.eval()
    probs = []
    labels = []
    control_indices = []
    losses = []
    for batch in loader:
        batch = [item.to(device) for item in batch]
        latent_blocks, process_features, block_mask, envelope_features, certificate_features, task_idx, label, control_idx = batch
        logits = model(
            latent_blocks,
            process_features,
            block_mask,
            envelope_features,
            certificate_features,
            task_idx,
        )
        losses.append(F.binary_cross_entropy_with_logits(logits, label).detach().cpu())
        probs.append(torch.sigmoid(logits).detach().cpu())
        labels.append(label.detach().cpu())
        control_indices.append(control_idx.detach().cpu())
    prob = torch.cat(probs)
    target = torch.cat(labels)
    control_idx = torch.cat(control_indices)
    metrics = {"loss": float(torch.stack(losses).mean().item()), **binary_metrics(prob, target)}
    control_aurocs = []
    control_gaps = []
    matched_mask = control_idx == control_types.index("matched")
    matched_mean = float(prob[matched_mask].mean().item()) if matched_mask.any() else float("nan")
    metrics["matched_score_mean"] = matched_mean
    for index, control_type in enumerate(control_types):
        current_mask = control_idx == index
        current_mean = float(prob[current_mask].mean().item()) if current_mask.any() else float("nan")
        metrics[f"{control_type}_score_mean"] = current_mean
        if control_type == "matched":
            continue
        pair_mask = matched_mask | current_mask
        pair_auc = binary_auroc(prob[pair_mask], target[pair_mask])
        pair_gap = matched_mean - current_mean
        metrics[f"{control_type}_auroc"] = pair_auc
        metrics[f"{control_type}_score_gap"] = pair_gap
        if not math.isnan(pair_auc):
            control_aurocs.append(pair_auc)
        if not math.isnan(pair_gap):
            control_gaps.append(pair_gap)
    metrics["mean_control_auroc"] = float(sum(control_aurocs) / len(control_aurocs)) if control_aurocs else float("nan")
    metrics["mean_control_gap"] = float(sum(control_gaps) / len(control_gaps)) if control_gaps else float("nan")
    return metrics


def binary_metrics(prob: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "compatibility_accuracy": binary_accuracy(prob, target),
        "compatibility_auroc": binary_auroc(prob, target),
        "compatibility_auprc": binary_auprc(prob, target),
        "positive_rate": float((prob >= 0.5).float().mean().item()),
        "target_positive_rate": float(target.float().mean().item()),
    }


def select_metric(metrics: dict[str, float], name: str) -> float:
    if name == "compatibility_mean_control_auroc":
        return float(metrics["mean_control_auroc"])
    if name == "compatibility_mean_control_gap":
        return float(metrics["mean_control_gap"])
    return float(metrics[name])


def save_checkpoint(
    path: Path,
    *,
    model: LatentReceiverCompatibilityModel,
    optimizer: torch.optim.Optimizer,
    config: LatentReceiverTrainConfig,
    norm_stats: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    step: int,
    metric: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "norm_stats": norm_stats,
            "metadata": metadata,
            "step": step,
            "metric": metric,
            "model_class": "LatentReceiverCompatibilityModel",
        },
        path,
    )


def write_metric(handle: Any, split: str, step: int, metrics: dict[str, float]) -> None:
    handle.write(
        json.dumps(
            {
                "created_at": int(time.time()),
                "split": split,
                "step": step,
                "metrics": metrics,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
    )
    handle.flush()


def stable_uniform(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) / float(16**16)


def safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def bool_or_float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return safe_float(value)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float().unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def parse_args() -> LatentReceiverTrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets-jsonl", default=LatentReceiverTrainConfig.packets_jsonl)
    parser.add_argument("--output-dir", default=LatentReceiverTrainConfig.output_dir)
    parser.add_argument("--input-mode", choices=sorted(INPUT_MODES), default=LatentReceiverTrainConfig.input_mode)
    parser.add_argument("--control-types", default=LatentReceiverTrainConfig.control_types)
    parser.add_argument("--seed", type=int, default=LatentReceiverTrainConfig.seed)
    parser.add_argument("--train-ratio", type=float, default=LatentReceiverTrainConfig.train_ratio)
    parser.add_argument("--valid-ratio", type=float, default=LatentReceiverTrainConfig.valid_ratio)
    parser.add_argument("--max-packets", type=int, default=LatentReceiverTrainConfig.max_packets)
    parser.add_argument("--batch-size", type=int, default=LatentReceiverTrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=LatentReceiverTrainConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=LatentReceiverTrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=LatentReceiverTrainConfig.weight_decay)
    parser.add_argument("--dropout", type=float, default=LatentReceiverTrainConfig.dropout)
    parser.add_argument("--d-model", type=int, default=LatentReceiverTrainConfig.d_model)
    parser.add_argument("--attention-heads", type=int, default=LatentReceiverTrainConfig.attention_heads)
    parser.add_argument("--inter-layers", type=int, default=LatentReceiverTrainConfig.inter_layers)
    parser.add_argument("--valid-interval", type=int, default=LatentReceiverTrainConfig.valid_interval)
    parser.add_argument("--selection-metric", choices=sorted(SELECTION_METRICS), default=LatentReceiverTrainConfig.selection_metric)
    parser.add_argument("--max-cached-shards", type=int, default=LatentReceiverTrainConfig.max_cached_shards)
    parser.add_argument("--noise-std", type=float, default=LatentReceiverTrainConfig.noise_std)
    parser.add_argument("--num-workers", type=int, default=LatentReceiverTrainConfig.num_workers)
    parser.add_argument("--device", default=LatentReceiverTrainConfig.device)
    parser.add_argument("--swanlab-mode", default=LatentReceiverTrainConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=LatentReceiverTrainConfig.experiment_name)
    args = parser.parse_args()
    return LatentReceiverTrainConfig(
        packets_jsonl=args.packets_jsonl,
        output_dir=args.output_dir,
        input_mode=args.input_mode,
        control_types=args.control_types,
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
        valid_interval=args.valid_interval,
        selection_metric=args.selection_metric,
        max_cached_shards=args.max_cached_shards,
        noise_std=args.noise_std,
        num_workers=args.num_workers,
        device=args.device,
        swanlab_mode=args.swanlab_mode,
        experiment_name=args.experiment_name,
    )


def main() -> None:
    summary = train_latent_receiver(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
