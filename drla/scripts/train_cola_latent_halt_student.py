"""Train LatentHaltStudent-v1 for decoder-as-teacher Cola halt.

This is the Phase P1 model. Decoder/scorer/text-stability fields are used only
as offline teacher targets. The online inputs are raw latent trajectories plus
latent/process/budget features; decoded text, EOS probabilities,
prediction-stability fields, and official correctness are never fed as input
features.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from drla.scripts.train_cola_continuation_risk_model import (
    is_strict_prefix,
    prediction_stability_reference,
)
from drla.scripts.train_cola_readiness_model import (
    OFFICIAL_COLA_TASKS,
    ReadinessTrainConfig,
    add_derived_stability_features,
    binary_auprc,
    binary_auroc,
    device_metadata,
    load_training_rows,
    parse_tasks,
    pos_weight,
    require_cuda_training,
    resolve_device,
    safe_float,
    stable_uniform,
)
from drla.tracking import finish_experiment, init_experiment, log_metrics


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

PROCESS_FEATURE_FIELD_MODES = {
    "full": PROCESS_FEATURE_FIELDS,
    "no_block_budget": [
        "latent_norm_mean",
        "latent_norm_std",
        "latent_delta_norm",
        "latent_delta_missing",
        "latent_cosine_to_prev",
        "latent_cosine_missing",
        "denoise_drift_norm_mean",
    ],
}

CHOICE_ANSWER_TASKS = {"mmlu", "obqa", "race", "siqa"}

def process_feature_fields_for_mode(mode: str) -> list[str]:
    if mode not in PROCESS_FEATURE_FIELD_MODES:
        raise ValueError("unknown process_feature_mode")
    return list(PROCESS_FEATURE_FIELD_MODES[mode])


BASE_BINARY_TARGETS = [
    "readiness",
    "correctness",
    "prediction_change",
    "contentful",
    "decoder_stop",
]
BINARY_TARGETS = BASE_BINARY_TARGETS


def binary_targets_for_config(config: "LatentHaltStudentTrainConfig") -> list[str]:
    targets = list(BASE_BINARY_TARGETS)
    if config.use_completion_risk:
        targets.append("completion_risk")
    if config.use_empty_answer_risk:
        targets.append("empty_answer_risk")
    if config.use_answer_format_risk:
        targets.append("answer_format_risk")
    return targets


def binary_target_tensor_names(config: "LatentHaltStudentTrainConfig") -> list[str]:
    tensor_names = {
        "readiness": "y_ready",
        "correctness": "y_correct",
        "prediction_change": "y_prediction_change",
        "contentful": "y_contentful",
        "decoder_stop": "y_decoder_stop",
        "completion_risk": "y_completion_risk",
        "empty_answer_risk": "y_empty_answer_risk",
        "answer_format_risk": "y_answer_format_risk",
    }
    return [tensor_names[name] for name in binary_targets_for_config(config)]


def loss_weight_for_target(config: "LatentHaltStudentTrainConfig", target: str) -> float:
    return float(getattr(config, f"{target}_loss_weight"))


@dataclass(frozen=True)
class LatentHaltStudentTrainConfig:
    labels_dir: str = "/data1/luyifei/drla/outputs/cola_readiness_frontiers/official8_full_b64_bs12_t16_seed66_20260524"
    output_dir: str = "/data1/luyifei/drla/outputs/cola_latent_halt_student/official8_full_b64_bs12_seed66_d64_pma4_seed20260524"
    tasks: str = ",".join(OFFICIAL_COLA_TASKS)
    seed: int = 20260524
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    batch_size: int = 512
    epochs: int = 20
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    d_model: int = 64
    attention_heads: int = 4
    inter_layers: int = 2
    pooling_mode: str = "pma4_last"
    task_conditioning: str = "none"
    process_interaction_mode: str = "process_token"
    process_feature_mode: str = "full"
    readout_context_mode: str = "none"
    readiness_loss_weight: float = 1.0
    correctness_loss_weight: float = 0.5
    prediction_change_loss_weight: float = 0.75
    contentful_loss_weight: float = 0.25
    decoder_stop_loss_weight: float = 0.25
    completion_risk_loss_weight: float = 0.75
    empty_answer_risk_loss_weight: float = 0.75
    answer_format_risk_loss_weight: float = 0.75
    future_gain_loss_weight: float = 0.25
    use_completion_risk: bool = False
    use_empty_answer_risk: bool = False
    use_answer_format_risk: bool = False
    readiness_target_mode: str = "oracle_frontier"
    teacher_decisions_jsonl: str = ""
    selection_metric: str = "readiness_auroc"
    valid_interval: int = 50
    num_workers: int = 0
    device: str = "auto"
    swanlab_mode: str = "cloud"
    experiment_name: str = "official8-latent-halt-student-v1"


class LatentHaltStudent(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        process_dim: int,
        max_blocks: int,
        block_size: int,
        d_model: int,
        attention_heads: int,
        inter_layers: int,
        dropout: float,
        pooling_mode: str,
        task_conditioning: str,
        process_interaction_mode: str,
        readout_context_mode: str,
        task_count: int,
        binary_targets: list[str] | None = None,
    ):
        super().__init__()
        if pooling_mode not in {"pma4_last", "pma1", "mean_max", "all_tokens"}:
            raise ValueError("pooling_mode must be one of: pma4_last, pma1, mean_max, all_tokens")
        if task_conditioning not in {"none", "query", "embedding"}:
            raise ValueError("task_conditioning must be one of: none, query, embedding")
        if process_interaction_mode not in {"process_token", "film"}:
            raise ValueError("process_interaction_mode must be one of: process_token, film")
        if readout_context_mode not in {"none", "last_process_query"}:
            raise ValueError("readout_context_mode must be one of: none, last_process_query")
        if d_model % attention_heads != 0:
            raise ValueError("d_model must be divisible by attention_heads")

        self.max_blocks = max_blocks
        self.block_size = block_size
        self.d_model = d_model
        self.pooling_mode = pooling_mode
        self.task_conditioning = task_conditioning
        self.process_interaction_mode = process_interaction_mode
        self.readout_context_mode = readout_context_mode
        self.binary_targets = list(binary_targets or BASE_BINARY_TARGETS)
        self.query_names = [*self.binary_targets, "future_gain"]

        self.slot_norm = nn.LayerNorm(latent_dim)
        self.slot_adapter = nn.Linear(latent_dim, d_model)
        self.slot_pos = nn.Embedding(block_size, d_model)
        self.block_pos = nn.Embedding(max_blocks, d_model)
        if process_interaction_mode == "process_token":
            self.process_mlp = nn.Sequential(
                nn.LayerNorm(process_dim),
                nn.Linear(process_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
            self.tokens_per_block = block_size + 1
        else:
            self.film_mlp = nn.Sequential(
                nn.LayerNorm(process_dim),
                nn.Linear(process_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 2 * d_model),
            )
            last_linear = self.film_mlp[-1]
            if isinstance(last_linear, nn.Linear):
                nn.init.zeros_(last_linear.weight)
                nn.init.zeros_(last_linear.bias)
            self.tokens_per_block = block_size

        self.intra_block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=attention_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        if pooling_mode in {"pma4_last", "pma1"}:
            pool_count = 4 if pooling_mode == "pma4_last" else 1
            self.pool_queries = nn.Parameter(torch.randn(pool_count, d_model) * 0.02)
            self.pool_attn = nn.MultiheadAttention(
                d_model,
                attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.summary_tokens_per_block = pool_count + (1 if pooling_mode == "pma4_last" else 0)
        elif pooling_mode == "mean_max":
            self.summary_tokens_per_block = 2
        else:
            self.summary_tokens_per_block = self.tokens_per_block

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

        query_count = len(self.query_names)
        self.readout_queries = nn.Parameter(torch.randn(query_count, d_model) * 0.02)
        self.readout_attn = nn.MultiheadAttention(
            d_model,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        if task_conditioning == "query":
            self.task_query_offsets = nn.Embedding(task_count, query_count * d_model)
        elif task_conditioning == "embedding":
            self.task_embedding = nn.Embedding(task_count, d_model)
        if readout_context_mode == "last_process_query":
            self.readout_process_mlp = nn.Sequential(
                nn.LayerNorm(process_dim),
                nn.Linear(process_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, query_count * d_model),
            )
            last_linear = self.readout_process_mlp[-1]
            if isinstance(last_linear, nn.Linear):
                nn.init.zeros_(last_linear.weight)
                nn.init.zeros_(last_linear.bias)

        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, 1),
                )
                for name in self.query_names
            }
        )

    def forward(
        self,
        latent_blocks: torch.Tensor,
        process_features: torch.Tensor,
        block_mask: torch.Tensor,
        task_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, max_blocks, block_size, _ = latent_blocks.shape
        device = latent_blocks.device
        slot_pos = self.slot_pos(torch.arange(block_size, device=device)).view(1, 1, block_size, -1)
        block_pos = self.block_pos(torch.arange(max_blocks, device=device)).view(1, max_blocks, 1, -1)

        slot_tokens = self.slot_adapter(self.slot_norm(latent_blocks)) + slot_pos + block_pos
        if self.process_interaction_mode == "process_token":
            process_tokens = self.process_mlp(process_features).unsqueeze(2) + block_pos
            block_tokens = torch.cat([slot_tokens, process_tokens], dim=2)
        else:
            scale_shift = self.film_mlp(process_features).view(batch_size, max_blocks, 2, self.d_model)
            scale = torch.tanh(scale_shift[:, :, 0, :]).unsqueeze(2)
            shift = scale_shift[:, :, 1, :].unsqueeze(2)
            block_tokens = slot_tokens * (1.0 + scale) + shift
        tokens_per_block = block_tokens.shape[2]

        intra_in = block_tokens.reshape(batch_size * max_blocks, tokens_per_block, self.d_model)
        intra_out = self.intra_block(intra_in).reshape(batch_size, max_blocks, tokens_per_block, self.d_model)
        block_summary = self.pool_blocks(intra_out)

        summary_per_block = block_summary.shape[2]
        memory = block_summary.reshape(batch_size, max_blocks * summary_per_block, self.d_model)
        if self.task_conditioning == "embedding":
            memory = memory + self.task_embedding(task_idx).unsqueeze(1)

        key_padding_mask = ~block_mask.bool().repeat_interleave(summary_per_block, dim=1)
        causal_mask = self.causal_block_mask(max_blocks, summary_per_block, device)
        memory = self.inter_block(memory, mask=causal_mask, src_key_padding_mask=key_padding_mask)

        queries = self.readout_queries.unsqueeze(0).expand(batch_size, -1, -1)
        if self.task_conditioning == "query":
            offset = self.task_query_offsets(task_idx).view(batch_size, len(self.query_names), self.d_model)
            queries = queries + offset
        if self.readout_context_mode == "last_process_query":
            last_block_idx = block_mask.long().sum(dim=1).clamp_min(1) - 1
            batch_idx = torch.arange(batch_size, device=device)
            last_process = process_features[batch_idx, last_block_idx]
            process_offset = self.readout_process_mlp(last_process).view(
                batch_size,
                len(self.query_names),
                self.d_model,
            )
            queries = queries + process_offset

        readout, _ = self.readout_attn(
            queries,
            memory,
            memory,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        outputs = {}
        for index, name in enumerate(self.query_names):
            outputs[name] = self.heads[name](readout[:, index]).squeeze(-1)
        return outputs

    def pool_blocks(self, intra_out: torch.Tensor) -> torch.Tensor:
        batch_size, max_blocks, tokens_per_block, d_model = intra_out.shape
        flat = intra_out.reshape(batch_size * max_blocks, tokens_per_block, d_model)
        if self.pooling_mode in {"pma4_last", "pma1"}:
            queries = self.pool_queries.unsqueeze(0).expand(batch_size * max_blocks, -1, -1)
            pooled, _ = self.pool_attn(queries, flat, flat, need_weights=False)
            if self.pooling_mode == "pma4_last":
                last_slot = intra_out[:, :, self.block_size - 1, :].reshape(batch_size * max_blocks, 1, d_model)
                pooled = torch.cat([pooled, last_slot], dim=1)
            return pooled.reshape(batch_size, max_blocks, pooled.shape[1], d_model)
        if self.pooling_mode == "mean_max":
            mean_token = flat.mean(dim=1, keepdim=True)
            max_token = flat.max(dim=1, keepdim=True).values
            pooled = torch.cat([mean_token, max_token], dim=1)
            return pooled.reshape(batch_size, max_blocks, 2, d_model)
        return intra_out

    @staticmethod
    def causal_block_mask(max_blocks: int, summary_per_block: int, device: torch.device) -> torch.Tensor:
        block_ids = torch.arange(max_blocks, device=device).repeat_interleave(summary_per_block)
        return block_ids.unsqueeze(0) > block_ids.unsqueeze(1)


def train_latent_halt_student(config: LatentHaltStudentTrainConfig) -> dict[str, Any]:
    validate_config(config)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"

    rows = load_training_rows(ReadinessTrainConfig(labels_dir=config.labels_dir, tasks=config.tasks, seed=config.seed))
    tensors, metadata = build_student_tensors(rows, config)
    splits = split_indices(metadata["sample_keys"], config)
    train_data, valid_data, test_data, norm_stats = make_split_datasets(tensors, splits, config)
    binary_targets = binary_targets_for_config(config)

    device = resolve_device(config.device)
    require_cuda_training(device, "train_cola_latent_halt_student.py")
    device_info = device_metadata(device)
    model = LatentHaltStudent(
        latent_dim=tensors["latent_blocks"].shape[-1],
        process_dim=tensors["process_features"].shape[-1],
        max_blocks=tensors["latent_blocks"].shape[1],
        block_size=tensors["latent_blocks"].shape[2],
        d_model=config.d_model,
        attention_heads=config.attention_heads,
        inter_layers=config.inter_layers,
        dropout=config.dropout,
        pooling_mode=config.pooling_mode,
        task_conditioning=config.task_conditioning,
        process_interaction_mode=config.process_interaction_mode,
        readout_context_mode=config.readout_context_mode,
        task_count=len(OFFICIAL_COLA_TASKS),
        binary_targets=binary_targets,
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
    train_idx = torch.tensor(splits["train"], dtype=torch.long)
    target_weights = {
        target: pos_weight(tensors[tensor_name][train_idx]).to(device)
        for target, tensor_name in zip(binary_targets, binary_target_tensor_names(config), strict=True)
    }

    run = init_experiment(
        stage="cola-latent-halt-student",
        experiment_name=config.experiment_name,
        description="LatentHaltStudent-v1 decoder-as-teacher halt model with latent/process-only online inputs.",
        config={
            **asdict(config),
            "device_info": device_info,
            "process_feature_fields": metadata["process_feature_fields"],
            "binary_targets": binary_targets,
            "teacher_targets": metadata["teacher_targets"],
            "readiness_target_mode": metadata["readiness_target_mode"],
            "teacher_decisions_jsonl": metadata["teacher_decisions_jsonl"],
            "online_input_policy": metadata["online_input_policy"],
            "num_rows": len(rows),
            "split_sizes": {name: len(indices) for name, indices in splits.items()},
            "max_blocks": int(tensors["latent_blocks"].shape[1]),
            "block_size": int(tensors["latent_blocks"].shape[2]),
        },
        mode=config.swanlab_mode,
        tags=["cola", "official-benchmark", "latent-halt-student", "decoder-as-teacher"],
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
                    loss, batch_metrics = compute_loss(model, batch, target_weights, config)
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
                        valid_metrics = evaluate(model, valid_loader, device, binary_targets)
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

            valid_metrics = evaluate(model, valid_loader, device, binary_targets)
            test_metrics = evaluate(model, test_loader, device, binary_targets)
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
            "process_feature_fields": metadata["process_feature_fields"],
            "binary_targets": binary_targets,
            "teacher_targets": metadata["teacher_targets"],
            "readiness_target_mode": metadata["readiness_target_mode"],
            "teacher_decisions_jsonl": metadata["teacher_decisions_jsonl"],
            "online_input_policy": metadata["online_input_policy"],
            "num_rows": len(rows),
            "split_sizes": {name: len(indices) for name, indices in splits.items()},
            "best_step": best_step,
            "best_metric_name": f"valid/{config.selection_metric}",
            "best_metric": best_metric,
            "last_valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
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


def validate_config(config: LatentHaltStudentTrainConfig) -> None:
    parse_tasks(config.tasks)
    if config.swanlab_mode != "cloud":
        raise ValueError("all deep-learning training experiments must use SwanLab cloud")
    if config.valid_interval > 100:
        raise ValueError("valid_interval must be <= 100 steps")
    if config.pooling_mode not in {"pma4_last", "pma1", "mean_max", "all_tokens"}:
        raise ValueError("unknown pooling_mode")
    if config.task_conditioning not in {"none", "query", "embedding"}:
        raise ValueError("unknown task_conditioning")
    if config.process_interaction_mode not in {"process_token", "film"}:
        raise ValueError("unknown process_interaction_mode")
    if config.process_feature_mode not in PROCESS_FEATURE_FIELD_MODES:
        raise ValueError("unknown process_feature_mode")
    if config.readout_context_mode not in {"none", "last_process_query"}:
        raise ValueError("unknown readout_context_mode")
    if config.readiness_target_mode not in {
        "oracle_frontier",
        "p0_teacher_halt",
        "p0_teacher_action",
        "answer_identity_halt",
        "answer_identity_action",
    }:
        raise ValueError("unknown readiness_target_mode")
    if config.readiness_target_mode in {"p0_teacher_halt", "p0_teacher_action"} and not config.teacher_decisions_jsonl:
        raise ValueError("teacher_decisions_jsonl is required for P0 teacher readiness target modes")
    if config.teacher_decisions_jsonl and not Path(config.teacher_decisions_jsonl).exists():
        raise FileNotFoundError(config.teacher_decisions_jsonl)
    if config.selection_metric not in {
        "readiness_auroc",
        "prediction_change_auroc",
        "readiness_prediction_change_mean_auroc",
        "readiness_prediction_change_completion_mean_auroc",
        "readiness_prediction_change_completion_contentful_mean_auroc",
        "readiness_prediction_change_completion_empty_mean_auroc",
        "readiness_prediction_change_completion_format_mean_auroc",
    }:
        raise ValueError("unknown selection_metric")
    for name, value in {
        "readiness_loss_weight": config.readiness_loss_weight,
        "correctness_loss_weight": config.correctness_loss_weight,
        "prediction_change_loss_weight": config.prediction_change_loss_weight,
        "contentful_loss_weight": config.contentful_loss_weight,
        "decoder_stop_loss_weight": config.decoder_stop_loss_weight,
        "completion_risk_loss_weight": config.completion_risk_loss_weight,
        "empty_answer_risk_loss_weight": config.empty_answer_risk_loss_weight,
        "answer_format_risk_loss_weight": config.answer_format_risk_loss_weight,
        "future_gain_loss_weight": config.future_gain_loss_weight,
    }.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if config.d_model % config.attention_heads != 0:
        raise ValueError("d_model must be divisible by attention_heads")
    if not 0 < config.train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0 <= config.valid_ratio < 1:
        raise ValueError("valid_ratio must be in [0, 1)")
    if config.train_ratio + config.valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio must leave a test split")
    if config.epochs < 1:
        raise ValueError("epochs must be >= 1 for a training experiment")


def build_student_tensors(
    rows: list[dict[str, Any]],
    config: LatentHaltStudentTrainConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    add_derived_stability_features(rows)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(f"{row['task']}::{row['sample_id']}", []).append(row)
    for sample_rows in grouped.values():
        sample_rows.sort(key=lambda item: int(item["block_index"]))
    teacher_halt_blocks = (
        load_teacher_halt_blocks(Path(config.teacher_decisions_jsonl))
        if config.readiness_target_mode in {"p0_teacher_halt", "p0_teacher_action"}
        else {}
    )

    first_row = rows[0]
    max_blocks = max(int(row["max_block_budget"]) for row in rows)
    block_size = int(first_row["latent_block_shape"][0])
    latent_dim = int(first_row["latent_block_shape"][1])
    process_feature_fields = process_feature_fields_for_mode(config.process_feature_mode)
    process_dim = len(process_feature_fields)
    row_count = len(rows)

    latent_blocks = torch.zeros(row_count, max_blocks, block_size, latent_dim, dtype=torch.float32)
    process_features = torch.zeros(row_count, max_blocks, process_dim, dtype=torch.float32)
    block_mask = torch.zeros(row_count, max_blocks, dtype=torch.bool)
    task_idx = torch.zeros(row_count, dtype=torch.long)
    y_ready = torch.zeros(row_count, dtype=torch.float32)
    y_correct = torch.zeros(row_count, dtype=torch.float32)
    y_future = torch.zeros(row_count, dtype=torch.float32)
    y_prediction_change = torch.zeros(row_count, dtype=torch.float32)
    y_contentful = torch.zeros(row_count, dtype=torch.float32)
    y_decoder_stop = torch.zeros(row_count, dtype=torch.float32)
    y_completion_risk = torch.zeros(row_count, dtype=torch.float32)
    y_empty_answer_risk = torch.zeros(row_count, dtype=torch.float32)
    y_answer_format_risk = torch.zeros(row_count, dtype=torch.float32)
    sample_keys: list[str] = []

    latent_cache: dict[str, torch.Tensor] = {}
    task_to_idx = {task: idx for idx, task in enumerate(OFFICIAL_COLA_TASKS)}
    write_idx = 0

    for sample_key, sample_rows in grouped.items():
        sample_latents = torch.zeros(max_blocks, block_size, latent_dim, dtype=torch.float32)
        sample_process = torch.zeros(max_blocks, process_dim, dtype=torch.float32)
        stability_prediction = prediction_stability_reference(sample_rows)
        final_prediction = normalize_text(sample_rows[-1].get("scored_prediction"))
        answer_identity_block = first_answer_identity_block(sample_rows, stability_prediction, final_prediction)
        teacher_halt_block = teacher_halt_blocks.get(sample_key)
        if config.readiness_target_mode in {"p0_teacher_halt", "p0_teacher_action"} and teacher_halt_block is None:
            raise KeyError(f"missing P0 teacher halt decision for {sample_key}")
        for row in sample_rows:
            block_index = int(row["block_index"])
            latent_path = row["latent_batch_path"]
            if latent_path not in latent_cache:
                latent_cache[latent_path] = torch.load(latent_path, map_location="cpu")["latent_blocks"]
            sample_latents[block_index] = latent_cache[latent_path][
                int(row["latent_batch_sample_index"]),
                int(row["latent_batch_block_index"]),
            ].float()
            sample_process[block_index] = torch.tensor(
                row_process_features(row, process_feature_fields),
                dtype=torch.float32,
            )

        for row in sample_rows:
            block_index = int(row["block_index"])
            prefix_len = block_index + 1
            latent_blocks[write_idx, :prefix_len] = sample_latents[:prefix_len]
            process_features[write_idx, :prefix_len] = sample_process[:prefix_len]
            block_mask[write_idx, :prefix_len] = True
            task_idx[write_idx] = task_to_idx[row["task"]]
            if config.readiness_target_mode == "p0_teacher_halt":
                y_ready[write_idx] = float(int(row["block_number"]) >= int(teacher_halt_block))
            elif config.readiness_target_mode == "p0_teacher_action":
                y_ready[write_idx] = float(int(row["block_number"]) == int(teacher_halt_block))
            elif config.readiness_target_mode == "answer_identity_halt":
                y_ready[write_idx] = float(int(row["block_number"]) >= answer_identity_block)
            elif config.readiness_target_mode == "answer_identity_action":
                y_ready[write_idx] = float(int(row["block_number"]) == answer_identity_block)
            else:
                y_ready[write_idx] = float(row.get("is_at_or_after_oracle_frontier", False))
            y_correct[write_idx] = float(row.get("official_correct", False))
            y_future[write_idx] = float(row.get("future_gain_correct", 0.0))
            prediction = normalize_text(row.get("scored_prediction"))
            y_prediction_change[write_idx] = float(bool(prediction or stability_prediction) and prediction != stability_prediction)
            y_contentful[write_idx] = float(bool(prediction))
            y_empty_answer_risk[write_idx] = float(not prediction)
            y_answer_format_risk[write_idx] = answer_format_risk_target(str(row["task"]), prediction)
            y_decoder_stop[write_idx] = float(
                bool(row.get("contains_eos")) or bool(row.get("contains_im_end")) or bool(row.get("contains_stop"))
            )
            y_completion_risk[write_idx] = completion_risk_target(
                prediction=prediction,
                stability_prediction=stability_prediction,
                final_prediction=final_prediction,
            )
            sample_keys.append(sample_key)
            write_idx += 1

    tensors = {
        "latent_blocks": latent_blocks,
        "process_features": process_features,
        "block_mask": block_mask,
        "task_idx": task_idx,
        "y_ready": y_ready,
        "y_correct": y_correct,
        "y_future": y_future,
        "y_prediction_change": y_prediction_change,
        "y_contentful": y_contentful,
        "y_decoder_stop": y_decoder_stop,
        "y_completion_risk": y_completion_risk,
        "y_empty_answer_risk": y_empty_answer_risk,
        "y_answer_format_risk": y_answer_format_risk,
    }
    metadata = {
        "sample_keys": sample_keys,
        "task_to_idx": task_to_idx,
        "process_feature_fields": process_feature_fields,
        "teacher_targets": {
            "readiness": readiness_target_description(config),
            "correctness": "official scorer correctness for offline distillation only",
            "future_gain": "future_gain_correct from future blocks",
            "prediction_change": "current scored_prediction differs from prediction-stability reference",
            "contentful": "current task-scored prediction is non-empty",
            "decoder_stop": "contains EOS/im_end/stop token in decoder output",
            "completion_risk": (
                "current scored_prediction is empty or a strict prefix of the "
                "prediction-stability/final reference; offline decoder teacher only"
            ),
            "empty_answer_risk": "current task-scored prediction is empty; offline decoder teacher only",
            "answer_format_risk": "choice-task scored prediction is not a single A-E option; offline decoder teacher only",
        },
        "online_input_policy": (
            f"raw latent prefixes, block mask, selected process features ({config.process_feature_mode}), "
            f"process interaction ({config.process_interaction_mode}), "
            f"readout context ({config.readout_context_mode}), and optional task conditioning only; "
            "no decoded text, decoder logits/probabilities, prediction-stability fields, official scorer outputs, "
            "or gold answers are input features"
        ),
        "pooling_mode": config.pooling_mode,
        "task_conditioning": config.task_conditioning,
        "process_interaction_mode": config.process_interaction_mode,
        "process_feature_mode": config.process_feature_mode,
        "readout_context_mode": config.readout_context_mode,
        "readiness_target_mode": config.readiness_target_mode,
        "teacher_decisions_jsonl": config.teacher_decisions_jsonl,
    }
    return tensors, metadata


def row_process_features(row: dict[str, Any], fields: list[str]) -> list[float]:
    block_number = safe_float(row.get("block_number"))
    max_budget = safe_float(row.get("max_block_budget"))
    delta = row.get("latent_delta_norm")
    cosine = row.get("latent_cosine_to_prev")
    values = {
        "block_number": block_number,
        "max_block_budget": max_budget,
        "remaining_blocks": max_budget - block_number,
        "block_fraction": block_number / max(max_budget, 1.0),
        "latent_norm_mean": safe_float(row.get("latent_norm_mean")),
        "latent_norm_std": safe_float(row.get("latent_norm_std")),
        "latent_delta_norm": safe_float(delta),
        "latent_delta_missing": 1.0 if delta is None else 0.0,
        "latent_cosine_to_prev": safe_float(cosine),
        "latent_cosine_missing": 1.0 if cosine is None else 0.0,
        "denoise_drift_norm_mean": safe_float(row.get("denoise_drift_norm_mean")),
    }
    return [values[field] for field in fields]


def load_teacher_halt_blocks(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            sample_key = str(item["sample_key"])
            chosen = item.get("chosen") or {}
            result[sample_key] = int(chosen["block_number"])
    if not result:
        raise ValueError(f"empty teacher decisions file: {path}")
    return result


def readiness_target_description(config: LatentHaltStudentTrainConfig) -> str:
    if config.readiness_target_mode == "p0_teacher_halt":
        return (
            "at or after Phase P0 joint-readiness riskcap04 teacher halt block; "
            "offline decoder-probed teacher policy only"
        )
    if config.readiness_target_mode == "p0_teacher_action":
        return (
            "exact Phase P0 joint-readiness riskcap04 teacher halt block as an online stop-action target; "
            "offline decoder-probed teacher policy only"
        )
    if config.readiness_target_mode == "answer_identity_halt":
        return (
            "at or after the first block whose scored_prediction matches the prediction-stability/final "
            "reference; offline decoder/text teacher only"
        )
    if config.readiness_target_mode == "answer_identity_action":
        return (
            "exact first block whose scored_prediction matches the prediction-stability/final reference "
            "as an online stop-action target; offline decoder/text teacher only"
        )
    return "is_at_or_after_oracle_frontier from offline oracle labels"


def make_split_datasets(
    tensors: dict[str, torch.Tensor],
    splits: dict[str, list[int]],
    config: LatentHaltStudentTrainConfig,
) -> tuple[TensorDataset, TensorDataset, TensorDataset, dict[str, torch.Tensor]]:
    train_idx = torch.tensor(splits["train"], dtype=torch.long)
    train_process = tensors["process_features"][train_idx]
    train_mask = tensors["block_mask"][train_idx].bool()
    process_mean = train_process[train_mask].mean(dim=0, keepdim=True)
    process_std = train_process[train_mask].std(dim=0, keepdim=True).clamp_min(1e-6)
    norm_process = (tensors["process_features"] - process_mean.view(1, 1, -1)) / process_std.view(1, 1, -1)
    norm_process = norm_process.masked_fill(~tensors["block_mask"].unsqueeze(-1), 0.0)
    norm_stats = {"process_mean": process_mean, "process_std": process_std}

    def dataset(indices: list[int]) -> TensorDataset:
        idx = torch.tensor(indices, dtype=torch.long)
        return TensorDataset(
            tensors["latent_blocks"][idx],
            norm_process[idx],
            tensors["block_mask"][idx],
            tensors["task_idx"][idx],
            tensors["y_future"][idx],
            *[tensors[name][idx] for name in binary_target_tensor_names(config)],
        )

    return dataset(splits["train"]), dataset(splits["valid"]), dataset(splits["test"]), norm_stats


def split_indices(sample_keys: list[str], config: LatentHaltStudentTrainConfig) -> dict[str, list[int]]:
    splits = {"train": [], "valid": [], "test": []}
    for idx, key in enumerate(sample_keys):
        value = stable_uniform(f"{config.seed}:{key}")
        if value < config.train_ratio:
            splits["train"].append(idx)
        elif value < config.train_ratio + config.valid_ratio:
            splits["valid"].append(idx)
        else:
            splits["test"].append(idx)
    for name, indices in splits.items():
        if not indices:
            raise ValueError(f"empty split: {name}")
    return splits


def compute_loss(
    model: LatentHaltStudent,
    batch: list[torch.Tensor],
    target_weights: dict[str, torch.Tensor],
    config: LatentHaltStudentTrainConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    (
        latent_blocks,
        process_features,
        block_mask,
        task_idx,
        y_future,
        *binary_values,
    ) = batch
    binary_targets = binary_targets_for_config(config)
    batch_targets = dict(zip(binary_targets, binary_values, strict=True))
    outputs = model(latent_blocks, process_features, block_mask, task_idx)
    losses = {
        name: F.binary_cross_entropy_with_logits(
            outputs[name],
            target,
            pos_weight=target_weights[name],
        )
        for name, target in batch_targets.items()
    }
    losses.update({"future_gain": F.mse_loss(outputs["future_gain"], y_future)})
    loss = sum(loss_weight_for_target(config, name) * losses[name] for name in binary_targets)
    loss = loss + config.future_gain_loss_weight * losses["future_gain"]
    with torch.no_grad():
        metrics = {f"{name}_loss": float(value.item()) for name, value in losses.items()}
        for name, target in batch_targets.items():
            metrics.update(
                {
                    f"{name}_{key}": value
                    for key, value in binary_threshold_metrics(
                        torch.sigmoid(outputs[name]).cpu(),
                        target.cpu(),
                    ).items()
                }
            )
    return loss, metrics


def binary_threshold_metrics(prob: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = (prob >= 0.5).float()
    target = target.float()
    tp = float(((pred == 1) & (target == 1)).sum().item())
    tn = float(((pred == 0) & (target == 0)).sum().item())
    fp = float(((pred == 1) & (target == 0)).sum().item())
    fn = float(((pred == 0) & (target == 1)).sum().item())
    total = max(float(target.numel()), 1.0)

    def div(num: float, den: float) -> float:
        return num / den if den > 0 else 0.0

    positive_rate = float(target.mean().item()) if target.numel() else 0.0
    predicted_positive_rate = float(pred.mean().item()) if pred.numel() else 0.0
    recall = div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    precision = div(tp, tp + fp)
    f1 = div(2.0 * precision * recall, precision + recall)
    accuracy = div(tp + tn, total)
    majority_baseline_accuracy = max(positive_rate, 1.0 - positive_rate)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_rate": predicted_positive_rate,
        "positive_rate": positive_rate,
        "majority_baseline_accuracy": majority_baseline_accuracy,
        "accuracy_lift_vs_majority": accuracy - majority_baseline_accuracy,
    }


@torch.no_grad()
def evaluate(
    model: LatentHaltStudent,
    loader: DataLoader,
    device: torch.device,
    binary_targets: list[str],
) -> dict[str, float]:
    model.eval()
    probs: dict[str, list[torch.Tensor]] = {name: [] for name in binary_targets}
    targets: dict[str, list[torch.Tensor]] = {name: [] for name in binary_targets}
    future_preds: list[torch.Tensor] = []
    future_targets: list[torch.Tensor] = []
    losses: list[float] = []

    for batch in loader:
        batch = [item.to(device) for item in batch]
        (
            latent_blocks,
            process_features,
            block_mask,
            task_idx,
            y_future,
            *binary_values,
        ) = batch
        outputs = model(latent_blocks, process_features, block_mask, task_idx)
        batch_targets = dict(zip(binary_targets, binary_values, strict=True))
        loss = sum(
            F.binary_cross_entropy_with_logits(outputs[name], target)
            for name, target in batch_targets.items()
        ) + 0.25 * F.mse_loss(outputs["future_gain"], y_future)
        losses.append(float(loss.item()))
        for name, target in batch_targets.items():
            probs[name].append(torch.sigmoid(outputs[name]).cpu())
            targets[name].append(target.cpu())
        future_preds.append(outputs["future_gain"].cpu())
        future_targets.append(y_future.cpu())

    result = {"loss": sum(losses) / max(len(losses), 1)}
    for name in binary_targets:
        prob = torch.cat(probs[name])
        target = torch.cat(targets[name])
        threshold_metrics = binary_threshold_metrics(prob, target)
        result.update({f"{name}_{key}": value for key, value in threshold_metrics.items()})
        result[f"{name}_auroc"] = binary_auroc(prob, target)
        result[f"{name}_auprc"] = binary_auprc(prob, target)
        result[f"{name}_brier"] = float(torch.mean((prob - target) ** 2).item())
    future = torch.cat(future_preds)
    y_future = torch.cat(future_targets)
    result["future_gain_mse"] = float(torch.mean((future - y_future) ** 2).item())
    return result


def save_checkpoint(
    path: Path,
    *,
    model: LatentHaltStudent,
    optimizer: torch.optim.Optimizer,
    config: LatentHaltStudentTrainConfig,
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
            "process_feature_fields": PROCESS_FEATURE_FIELDS,
            "norm_stats": norm_stats,
            "metadata": {key: value for key, value in metadata.items() if key != "sample_keys"},
            "step": step,
            "metric": metric,
        },
        path,
    )


def write_metric(metrics_f: Any, split: str, step: int, metrics: dict[str, float]) -> None:
    metrics_f.write(
        json.dumps(
            {"created_at": int(time.time()), "split": split, "step": step, "metrics": metrics},
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )
    metrics_f.flush()


def select_metric(metrics: dict[str, float], selection_metric: str) -> float:
    if selection_metric == "readiness_prediction_change_mean_auroc":
        value = mean_metric(metrics, ["readiness_auroc", "prediction_change_auroc"])
    elif selection_metric == "readiness_prediction_change_completion_mean_auroc":
        value = mean_metric(
            metrics,
            ["readiness_auroc", "prediction_change_auroc", "completion_risk_auroc"],
        )
    elif selection_metric == "readiness_prediction_change_completion_contentful_mean_auroc":
        value = mean_metric(
            metrics,
            ["readiness_auroc", "prediction_change_auroc", "completion_risk_auroc", "contentful_auroc"],
        )
    elif selection_metric == "readiness_prediction_change_completion_empty_mean_auroc":
        value = mean_metric(
            metrics,
            ["readiness_auroc", "prediction_change_auroc", "completion_risk_auroc", "empty_answer_risk_auroc"],
        )
    elif selection_metric == "readiness_prediction_change_completion_format_mean_auroc":
        value = mean_metric(
            metrics,
            ["readiness_auroc", "prediction_change_auroc", "completion_risk_auroc", "answer_format_risk_auroc"],
        )
    else:
        value = metrics.get(selection_metric, float("nan"))
    if math.isnan(value):
        return -metrics["loss"]
    return value


def mean_metric(metrics: dict[str, float], names: list[str]) -> float:
    values = [metrics.get(name, float("nan")) for name in names]
    if any(math.isnan(value) for value in values):
        return float("nan")
    return sum(values) / len(values)


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def first_answer_identity_block(
    sample_rows: list[dict[str, Any]],
    stability_prediction: str,
    final_prediction: str,
) -> int:
    reference = stability_prediction or final_prediction
    if reference:
        for row in sample_rows:
            if normalize_text(row.get("scored_prediction")) == reference:
                return int(row["block_number"])
    return int(sample_rows[-1]["block_number"])


def completion_risk_target(
    *,
    prediction: str,
    stability_prediction: str,
    final_prediction: str,
) -> float:
    reference = stability_prediction or final_prediction
    if not reference or prediction == reference:
        return 0.0
    if not prediction:
        return 1.0
    return float(is_strict_prefix(prediction, reference))


def answer_format_risk_target(task: str, prediction: str) -> float:
    if task not in CHOICE_ANSWER_TASKS:
        return 0.0
    return float(re.fullmatch(r"[a-e]", prediction, flags=re.IGNORECASE) is None)


def parse_args() -> LatentHaltStudentTrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", default=LatentHaltStudentTrainConfig.labels_dir)
    parser.add_argument("--output-dir", default=LatentHaltStudentTrainConfig.output_dir)
    parser.add_argument("--tasks", default=LatentHaltStudentTrainConfig.tasks)
    parser.add_argument("--seed", type=int, default=LatentHaltStudentTrainConfig.seed)
    parser.add_argument("--train-ratio", type=float, default=LatentHaltStudentTrainConfig.train_ratio)
    parser.add_argument("--valid-ratio", type=float, default=LatentHaltStudentTrainConfig.valid_ratio)
    parser.add_argument("--batch-size", type=int, default=LatentHaltStudentTrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=LatentHaltStudentTrainConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=LatentHaltStudentTrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=LatentHaltStudentTrainConfig.weight_decay)
    parser.add_argument("--dropout", type=float, default=LatentHaltStudentTrainConfig.dropout)
    parser.add_argument("--d-model", type=int, default=LatentHaltStudentTrainConfig.d_model)
    parser.add_argument("--attention-heads", type=int, default=LatentHaltStudentTrainConfig.attention_heads)
    parser.add_argument("--inter-layers", type=int, default=LatentHaltStudentTrainConfig.inter_layers)
    parser.add_argument("--pooling-mode", default=LatentHaltStudentTrainConfig.pooling_mode)
    parser.add_argument("--task-conditioning", default=LatentHaltStudentTrainConfig.task_conditioning)
    parser.add_argument("--process-interaction-mode", default=LatentHaltStudentTrainConfig.process_interaction_mode)
    parser.add_argument("--process-feature-mode", default=LatentHaltStudentTrainConfig.process_feature_mode)
    parser.add_argument("--readout-context-mode", default=LatentHaltStudentTrainConfig.readout_context_mode)
    parser.add_argument("--readiness-loss-weight", type=float, default=LatentHaltStudentTrainConfig.readiness_loss_weight)
    parser.add_argument("--correctness-loss-weight", type=float, default=LatentHaltStudentTrainConfig.correctness_loss_weight)
    parser.add_argument(
        "--prediction-change-loss-weight",
        type=float,
        default=LatentHaltStudentTrainConfig.prediction_change_loss_weight,
    )
    parser.add_argument("--contentful-loss-weight", type=float, default=LatentHaltStudentTrainConfig.contentful_loss_weight)
    parser.add_argument(
        "--decoder-stop-loss-weight",
        type=float,
        default=LatentHaltStudentTrainConfig.decoder_stop_loss_weight,
    )
    parser.add_argument(
        "--completion-risk-loss-weight",
        type=float,
        default=LatentHaltStudentTrainConfig.completion_risk_loss_weight,
    )
    parser.add_argument(
        "--empty-answer-risk-loss-weight",
        type=float,
        default=LatentHaltStudentTrainConfig.empty_answer_risk_loss_weight,
    )
    parser.add_argument(
        "--answer-format-risk-loss-weight",
        type=float,
        default=LatentHaltStudentTrainConfig.answer_format_risk_loss_weight,
    )
    parser.add_argument("--future-gain-loss-weight", type=float, default=LatentHaltStudentTrainConfig.future_gain_loss_weight)
    parser.add_argument("--use-completion-risk", action="store_true")
    parser.add_argument("--use-empty-answer-risk", action="store_true")
    parser.add_argument("--use-answer-format-risk", action="store_true")
    parser.add_argument("--readiness-target-mode", default=LatentHaltStudentTrainConfig.readiness_target_mode)
    parser.add_argument("--teacher-decisions-jsonl", default=LatentHaltStudentTrainConfig.teacher_decisions_jsonl)
    parser.add_argument("--selection-metric", default=LatentHaltStudentTrainConfig.selection_metric)
    parser.add_argument("--valid-interval", type=int, default=LatentHaltStudentTrainConfig.valid_interval)
    parser.add_argument("--num-workers", type=int, default=LatentHaltStudentTrainConfig.num_workers)
    parser.add_argument("--device", default=LatentHaltStudentTrainConfig.device)
    parser.add_argument("--swanlab-mode", default=LatentHaltStudentTrainConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=LatentHaltStudentTrainConfig.experiment_name)
    args = parser.parse_args()
    return LatentHaltStudentTrainConfig(
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        tasks=args.tasks,
        seed=args.seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        d_model=args.d_model,
        attention_heads=args.attention_heads,
        inter_layers=args.inter_layers,
        pooling_mode=args.pooling_mode,
        task_conditioning=args.task_conditioning,
        process_interaction_mode=args.process_interaction_mode,
        process_feature_mode=args.process_feature_mode,
        readout_context_mode=args.readout_context_mode,
        readiness_loss_weight=args.readiness_loss_weight,
        correctness_loss_weight=args.correctness_loss_weight,
        prediction_change_loss_weight=args.prediction_change_loss_weight,
        contentful_loss_weight=args.contentful_loss_weight,
        decoder_stop_loss_weight=args.decoder_stop_loss_weight,
        completion_risk_loss_weight=args.completion_risk_loss_weight,
        empty_answer_risk_loss_weight=args.empty_answer_risk_loss_weight,
        answer_format_risk_loss_weight=args.answer_format_risk_loss_weight,
        future_gain_loss_weight=args.future_gain_loss_weight,
        use_completion_risk=args.use_completion_risk,
        use_empty_answer_risk=args.use_empty_answer_risk,
        use_answer_format_risk=args.use_answer_format_risk,
        readiness_target_mode=args.readiness_target_mode,
        teacher_decisions_jsonl=args.teacher_decisions_jsonl,
        selection_metric=args.selection_metric,
        valid_interval=args.valid_interval,
        num_workers=args.num_workers,
        device=args.device,
        swanlab_mode=args.swanlab_mode,
        experiment_name=args.experiment_name,
    )


def main() -> None:
    summary = train_latent_halt_student(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
