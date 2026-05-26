"""Train the Stage C block-causal latent prior MVP."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer

from drla.data.answer_judge import judge
from drla.data.stage_c import LatentCacheCollator, LatentCacheDataset
from drla.models.stage_b import StageBModelConfig, StageBReasoningAutoencoder, block_masked_mse
from drla.models.stage_c import BlockCausalPrior, StageCPriorConfig, rollout_prior
from drla.tracking import finish_experiment, init_experiment, log_metrics
from drla.training.stage_b_autoencoder import StageBTrainConfig, move_batch, weighted_loss as stage_b_weighted_loss


@dataclass(frozen=True)
class StageCPriorTrainConfig:
    stage_b_checkpoint: str = "/data1/luyifei/drla/outputs/stage_b_full_vocab_formal/checkpoint.pt"
    train_latent_dir: str = "/data1/luyifei/drla/outputs/stage_b_full_vocab_formal/latents/train"
    eval_latent_dir: str = "/data1/luyifei/drla/outputs/stage_b_full_vocab_formal/latents/test"
    output_dir: str = "/data1/luyifei/drla/outputs/stage_c_prior_mvp"
    tokenizer_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    local_files_only: bool = True
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    batch_size: int = 8
    eval_batch_size: int = 8
    epochs: int = 3
    max_steps: int | None = None
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    decode_weight: float = 0.0
    decode_warmup_steps: int = 0
    decode_ramp_steps: int = 1000
    previous_mix_final: float = 0.0
    previous_mix_warmup_steps: int = 0
    previous_mix_ramp_steps: int = 1000
    fixed_b_values: tuple[int, ...] = (4, 8, 16, 32)
    condition_encoder: str = "learned"
    condition_model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    condition_dtype: str = "bfloat16"
    prior_mode: str = "deterministic"
    flow_time_max: float = 1000.0
    flow_time_loc: float = 1.0
    flow_time_scale: float = 0.0
    flow_denoise_steps: int = 16
    joint_stage_b: bool = False
    stage_b_lr_ratio: float = 1.0
    stage_b_vae_weight: float = 0.1
    stage_b_ref_weight: float = 0.1
    stage_b_question_latent_weight: float = 0.0
    seed: int = 42
    device: str = "auto"
    swanlab_mode: str | None = None


def ensure_cloud_training(mode: str | None) -> None:
    requested = mode or os.getenv("SWANLAB_MODE") or "cloud"
    if requested != "cloud":
        raise ValueError(
            "Training must be logged to SwanLab cloud. "
            "Unset SWANLAB_MODE or use --swanlab-mode cloud."
        )


def train_stage_c_prior(config: StageCPriorTrainConfig) -> dict[str, Any]:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name,
        trust_remote_code=True,
        local_files_only=config.local_files_only,
    )
    stage_b = load_stage_b_decoder(config.stage_b_checkpoint, device=device)
    set_stage_b_trainability(stage_b, trainable=config.joint_stage_b)
    condition_encoder = load_condition_encoder(config, device=device)

    model_config = StageCPriorConfig(
        vocab_size=stage_b.config.vocab_size,
        b_max=stage_b.config.b_max,
        block_size=stage_b.config.block_size,
        latent_dim=stage_b.config.latent_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        pad_id=stage_b.config.pad_id,
        condition_dim=condition_encoder.hidden_size if condition_encoder is not None else None,
    )
    model = BlockCausalPrior(model_config).to(device)
    condition_init = initialize_prior_condition(model, stage_b)
    validate_prior_mode(config)
    flow_schedule = (
        make_flow_schedule(config, device=device)
        if config.prior_mode == "flow"
        else None
    )
    optimizer = make_stage_c_optimizer(model, stage_b, config)

    train_loader = make_loader(
        config.train_latent_dir,
        pad_id=stage_b.config.pad_id,
        batch_size=config.batch_size,
        shuffle=True,
        max_samples=config.max_train_samples,
        seed=config.seed,
    )
    eval_loader = make_loader(
        config.eval_latent_dir,
        pad_id=stage_b.config.pad_id,
        batch_size=config.eval_batch_size,
        shuffle=False,
        max_samples=config.max_eval_samples,
        seed=config.seed,
    )

    ensure_cloud_training(config.swanlab_mode)

    init_experiment(
        stage="stage-c",
        experiment_name=f"stage-c-block-causal-{config.prior_mode}-prior-mvp",
        description="Block-causal latent prior MVP over Stage B cached gold latents.",
        config={**asdict(config), "model_config": asdict(model_config), "condition_init": condition_init},
        mode=config.swanlab_mode,
    )

    global_step = 0
    final_train_metrics: dict[str, float] = {}
    final_eval_metrics: dict[str, float] = {}
    best_eval_metrics: dict[str, float] = {}
    best_eval_step = 0
    best_metric_name = primary_eval_metric_name(config)
    best_score = float("-inf")
    best_checkpoint_path = output_dir / "best_checkpoint.pt"
    started_at = int(time.time())
    try:
        for _epoch in range(config.epochs):
            model.train()
            stage_b.train(config.joint_stage_b)
            for batch in train_loader:
                global_step += 1
                batch = move_batch(batch, device)
                previous_mix_p = scheduled_value(
                    global_step,
                    final_value=config.previous_mix_final,
                    warmup_steps=config.previous_mix_warmup_steps,
                    ramp_steps=config.previous_mix_ramp_steps,
                )
                question_features = encode_condition_features(condition_encoder, batch)
                prior_batch, stage_b_joint_metrics = prepare_prior_training_batch(
                    stage_b,
                    batch,
                    tokenizer=tokenizer,
                    config=config,
                )
                previous_context = build_previous_context(
                    model,
                    prior_batch,
                    previous_mix_p,
                    question_features,
                    config=config,
                    flow_schedule=flow_schedule,
                )
                pred, prior_loss, prior_extra_metrics = training_prior_prediction(
                    model,
                    prior_batch,
                    previous_context,
                    question_features,
                    config=config,
                    flow_schedule=flow_schedule,
                )
                decode_loss = decoder_recon_loss(stage_b, pred, prior_batch)
                active_decode_weight = scheduled_value(
                    global_step,
                    final_value=config.decode_weight,
                    warmup_steps=config.decode_warmup_steps,
                    ramp_steps=config.decode_ramp_steps,
                )
                stage_b_joint_loss = stage_b_joint_metrics.pop("stage_b_joint_loss_tensor")
                loss = prior_loss + active_decode_weight * decode_loss + stage_b_joint_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters(model, stage_b), config.grad_clip)
                optimizer.step()
                final_train_metrics = {
                    "loss": float(loss.detach().cpu().item()),
                    "prior_mse": float(prior_loss.detach().cpu().item()),
                    "decode_loss": float(decode_loss.detach().cpu().item()),
                    "decode_weight": float(active_decode_weight),
                    "weighted_decode_loss": float((active_decode_weight * decode_loss).detach().cpu().item()),
                    "previous_mix_p": float(previous_mix_p),
                    **prior_extra_metrics,
                    **stage_b_joint_metrics,
                }
                log_metrics(final_train_metrics, step=global_step, prefix="train")
                if config.max_steps is not None and global_step >= config.max_steps:
                    break
            final_eval_metrics = evaluate_prior(
                model,
                stage_b,
                eval_loader,
                tokenizer=tokenizer,
                config=config,
                device=device,
                condition_encoder=condition_encoder,
                flow_schedule=flow_schedule,
            )
            log_metrics(final_eval_metrics, step=global_step, prefix="valid")
            current_score = primary_eval_score(final_eval_metrics, config)
            if current_score > best_score:
                best_score = current_score
                best_eval_step = global_step
                best_eval_metrics = dict(final_eval_metrics)
                save_stage_c_checkpoint(
                    best_checkpoint_path,
                    model=model,
                    stage_b=stage_b,
                    model_config=model_config,
                    train_config=config,
                    global_step=global_step,
                )
            if config.max_steps is not None and global_step >= config.max_steps:
                break
    finally:
        finish_experiment()

    checkpoint_path = output_dir / "checkpoint.pt"
    save_stage_c_checkpoint(
        checkpoint_path,
        model=model,
        stage_b=stage_b,
        model_config=model_config,
        train_config=config,
        global_step=global_step,
    )
    summary = {
        "created_at": started_at,
        "finished_at": int(time.time()),
        "config": asdict(config),
        "checkpoint_path": str(checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path) if best_eval_metrics else None,
        "global_step": global_step,
        "condition_init": condition_init,
        "train": final_train_metrics,
        "valid": final_eval_metrics,
        "best_valid": {
            "step": best_eval_step,
            "metric": best_metric_name,
            "value": best_eval_metrics.get(best_metric_name) if best_eval_metrics else None,
            "metrics": best_eval_metrics,
        },
        "stage_c_gate": {
            "formal_gate_evaluable": (
                config.max_train_samples is None
                and config.max_eval_samples is None
                and model_config.b_max == 32
            ),
            "notes": stage_c_gate_notes(config),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def save_stage_c_checkpoint(
    path: Path,
    *,
    model: BlockCausalPrior,
    stage_b: StageBReasoningAutoencoder | None = None,
    model_config: StageCPriorConfig,
    train_config: StageCPriorTrainConfig,
    global_step: int,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "global_step": global_step,
    }
    if stage_b is not None:
        payload["stage_b_model_state_dict"] = stage_b.state_dict()
        payload["stage_b_model_config"] = asdict(stage_b.config)
    torch.save(payload, path)


def primary_eval_metric_name(config: StageCPriorTrainConfig) -> str:
    fixed_b = max(config.fixed_b_values)
    return f"rollout_fixed_b{fixed_b}_acc"


def primary_eval_score(metrics: dict[str, float], config: StageCPriorTrainConfig) -> float:
    metric_name = primary_eval_metric_name(config)
    if metric_name in metrics:
        return metrics[metric_name]
    fallback_name = metric_name.replace("rollout_", "")
    if fallback_name in metrics:
        return metrics[fallback_name]
    return -metrics.get("prior_rollout_mse", float("inf"))


def trainable_parameters(
    model: BlockCausalPrior, stage_b: StageBReasoningAutoencoder
) -> list[torch.nn.Parameter]:
    return [
        parameter
        for module in (model, stage_b)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]


def set_stage_b_trainability(stage_b: StageBReasoningAutoencoder, *, trainable: bool) -> None:
    stage_b.train(trainable)
    for parameter in stage_b.parameters():
        parameter.requires_grad_(trainable)


def make_stage_c_optimizer(
    model: BlockCausalPrior,
    stage_b: StageBReasoningAutoencoder,
    config: StageCPriorTrainConfig,
) -> torch.optim.Optimizer:
    groups: list[dict[str, Any]] = [
        {"params": list(model.parameters()), "lr": config.lr}
    ]
    if config.joint_stage_b:
        stage_b_params = [parameter for parameter in stage_b.parameters() if parameter.requires_grad]
        groups.append({"params": stage_b_params, "lr": config.lr * config.stage_b_lr_ratio})
    return torch.optim.AdamW(groups, lr=config.lr, weight_decay=config.weight_decay)


def build_stage_b_batch_from_stage_c(
    stage_b: StageBReasoningAutoencoder,
    batch: dict[str, Any],
    *,
    tokenizer: Any,
) -> dict[str, Any]:
    batch_size = batch["question_ids"].shape[0]
    capacity = stage_b.config.capacity
    device = batch["question_ids"].device
    target_input_ids = torch.full(
        (batch_size, capacity),
        stage_b.config.pad_id,
        dtype=torch.long,
        device=device,
    )
    target_mask = torch.zeros((batch_size, capacity), dtype=torch.long, device=device)
    copy_len = min(capacity, batch["target_ids"].shape[1])
    if copy_len > 0:
        target_input_ids[:, :copy_len] = batch["target_ids"][:, :copy_len]
        target_mask[:, :copy_len] = batch["target_mask"][:, :copy_len]
    target_labels = target_input_ids.clone()
    target_labels[target_mask == 0] = -100

    answer_rows = [
        tokenizer.encode(str(answer), add_special_tokens=False)[: stage_b.config.max_answer_len]
        for answer in batch["answer_norms"]
    ]
    answer_input_ids = torch.full(
        (batch_size, stage_b.config.max_answer_len),
        stage_b.config.pad_id,
        dtype=torch.long,
        device=device,
    )
    answer_mask = torch.zeros(
        (batch_size, stage_b.config.max_answer_len), dtype=torch.long, device=device
    )
    for row_index, answer_ids in enumerate(answer_rows):
        if answer_ids:
            values = torch.tensor(answer_ids, dtype=torch.long, device=device)
            answer_input_ids[row_index, : values.numel()] = values
            answer_mask[row_index, : values.numel()] = 1
    answer_labels = answer_input_ids.clone()

    validate_stage_b_token_range(stage_b, batch["question_ids"], target_input_ids, answer_input_ids)
    return {
        "ids": batch["ids"],
        "question_ids": batch["question_ids"],
        "question_mask": batch["question_mask"],
        "target_input_ids": target_input_ids,
        "target_labels": target_labels,
        "target_mask": target_mask,
        "answer_input_ids": answer_input_ids,
        "answer_labels": answer_labels,
        "answer_mask": answer_mask,
        "answer_norms": batch["answer_norms"],
        "b_star": batch["b_star"],
        "block_mask": batch["block_mask"],
        "noop_mask": batch["noop_mask"],
    }


def validate_stage_b_token_range(
    stage_b: StageBReasoningAutoencoder, *token_tensors: torch.Tensor
) -> None:
    max_token = max(int(tensor.max().detach().cpu().item()) for tensor in token_tensors)
    if max_token >= stage_b.config.vocab_size:
        raise ValueError(
            "Joint Stage B training requires token ids in the Stage B vocabulary. "
            f"Saw token id {max_token}, vocab_size={stage_b.config.vocab_size}."
        )


def prepare_prior_training_batch(
    stage_b: StageBReasoningAutoencoder,
    batch: dict[str, Any],
    *,
    tokenizer: Any,
    config: StageCPriorTrainConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    zero = batch["z_blocks"].new_zeros(())
    if not config.joint_stage_b:
        return batch, {
            "stage_b_joint_loss_tensor": zero,
            "stage_b_vae_loss": 0.0,
            "stage_b_ref_loss": 0.0,
            "stage_b_joint_loss": 0.0,
        }
    stage_b_batch = build_stage_b_batch_from_stage_c(stage_b, batch, tokenizer=tokenizer)
    outputs = stage_b(stage_b_batch)
    vae_config = StageBTrainConfig(
        question_latent_weight=config.stage_b_question_latent_weight
    )
    vae_loss = stage_b_weighted_loss(outputs, vae_config)
    ref_loss = block_masked_mse(outputs["z_blocks"], batch["z_blocks"], batch["block_mask"])
    joint_loss = config.stage_b_vae_weight * vae_loss + config.stage_b_ref_weight * ref_loss
    prior_batch = dict(batch)
    prior_batch["z_blocks"] = outputs["z_blocks"]
    prior_batch["block_mask"] = stage_b_batch["block_mask"]
    prior_batch["noop_mask"] = stage_b_batch["noop_mask"]
    return prior_batch, {
        "stage_b_joint_loss_tensor": joint_loss,
        "stage_b_vae_loss": float(vae_loss.detach().cpu().item()),
        "stage_b_ref_loss": float(ref_loss.detach().cpu().item()),
        "stage_b_joint_loss": float(joint_loss.detach().cpu().item()),
    }


@torch.no_grad()
def prepare_prior_eval_batch(
    stage_b: StageBReasoningAutoencoder,
    batch: dict[str, Any],
    *,
    tokenizer: Any,
    config: StageCPriorTrainConfig,
) -> dict[str, Any]:
    if not config.joint_stage_b:
        return batch
    stage_b_batch = build_stage_b_batch_from_stage_c(stage_b, batch, tokenizer=tokenizer)
    outputs = stage_b(stage_b_batch)
    prior_batch = dict(batch)
    prior_batch["z_blocks"] = outputs["z_blocks"]
    prior_batch["block_mask"] = stage_b_batch["block_mask"]
    prior_batch["noop_mask"] = stage_b_batch["noop_mask"]
    return prior_batch


def initialize_prior_condition(
    model: BlockCausalPrior, stage_b: StageBReasoningAutoencoder
) -> dict[str, Any]:
    if model.config.condition_dim is not None:
        return {
            "external_condition_encoder": True,
            "condition_dim": model.config.condition_dim,
            "question_embedding_from_stage_b": False,
        }
    if model.question_embedding.weight.shape != stage_b.token_embedding.weight.shape:
        return {
            "question_embedding_from_stage_b": False,
            "reason": "shape_mismatch",
            "prior_shape": list(model.question_embedding.weight.shape),
            "stage_b_shape": list(stage_b.token_embedding.weight.shape),
        }
    with torch.no_grad():
        model.question_embedding.weight.copy_(stage_b.token_embedding.weight)
    return {"question_embedding_from_stage_b": True}


class FrozenConditionEncoder:
    def __init__(
        self,
        model_name: str,
        *,
        device: torch.device,
        local_files_only: bool,
        dtype_name: str,
    ) -> None:
        model_config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True, local_files_only=local_files_only
        )
        self.hidden_size = int(model_config.hidden_size)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
            torch_dtype=resolve_condition_dtype(dtype_name),
        ).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def encode(self, question_ids: torch.Tensor, question_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            input_ids=question_ids,
            attention_mask=question_mask,
            use_cache=False,
        )
        return outputs.last_hidden_state.float()


def load_condition_encoder(
    config: StageCPriorTrainConfig, *, device: torch.device
) -> FrozenConditionEncoder | None:
    if config.condition_encoder == "learned":
        return None
    if config.condition_encoder != "qwen":
        raise ValueError(f"Unknown condition encoder: {config.condition_encoder}")
    return FrozenConditionEncoder(
        config.condition_model_name,
        device=device,
        local_files_only=config.local_files_only,
        dtype_name=config.condition_dtype,
    )


def encode_condition_features(
    condition_encoder: FrozenConditionEncoder | None, batch: dict[str, Any]
) -> torch.Tensor | None:
    if condition_encoder is None:
        return None
    return condition_encoder.encode(batch["question_ids"], batch["question_mask"])


def resolve_condition_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_name in {"fp16", "float16"}:
        return torch.float16
    if dtype_name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported condition dtype: {dtype_name}")


def make_loader(
    latent_dir: str,
    *,
    pad_id: int,
    batch_size: int,
    shuffle: bool,
    max_samples: int | None,
    seed: int,
) -> DataLoader:
    dataset = LatentCacheDataset(latent_dir, max_samples=max_samples)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=LatentCacheCollator(pad_id=pad_id),
        generator=generator,
    )


def scheduled_value(
    step: int,
    *,
    final_value: float,
    warmup_steps: int,
    ramp_steps: int,
) -> float:
    if final_value <= 0 or step <= warmup_steps:
        return 0.0
    if ramp_steps <= 0:
        return final_value
    progress = min(1.0, (step - warmup_steps) / ramp_steps)
    return final_value * progress


def validate_prior_mode(config: StageCPriorTrainConfig) -> None:
    if config.prior_mode not in {"deterministic", "flow"}:
        raise ValueError(f"Unknown prior_mode: {config.prior_mode}")
    if config.joint_stage_b and config.prior_mode != "flow":
        raise ValueError("Cola-style joint Stage B training requires --prior-mode flow")
    if config.joint_stage_b and config.stage_b_lr_ratio <= 0:
        raise ValueError("stage_b_lr_ratio must be positive when joint_stage_b is enabled")
    if config.stage_b_vae_weight < 0 or config.stage_b_ref_weight < 0:
        raise ValueError("Stage B joint loss weights must be non-negative")
    if config.flow_time_max <= 0:
        raise ValueError("flow_time_max must be positive")
    if config.flow_time_scale < 0:
        raise ValueError("flow_time_scale must be non-negative")
    if config.flow_denoise_steps <= 0:
        raise ValueError("flow_denoise_steps must be positive")


def make_flow_schedule(
    config: StageCPriorTrainConfig, *, device: torch.device
) -> dict[str, torch.Tensor]:
    validate_prior_mode(config)
    times = torch.linspace(
        0.0,
        1.0,
        config.flow_denoise_steps + 1,
        device=device,
        dtype=torch.float32,
    )
    return {"infer_times": times}


def sample_flow_times(
    batch_size: int,
    config: StageCPriorTrainConfig,
    *,
    device: torch.device,
) -> torch.Tensor:
    if config.flow_time_scale == 0:
        raw = torch.full((batch_size,), config.flow_time_loc, device=device)
    else:
        raw = torch.randn(batch_size, device=device) * config.flow_time_scale
        raw = raw + config.flow_time_loc
    return torch.sigmoid(raw).clamp(1e-4, 1.0 - 1e-4)


def expand_flow_time(times: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    values = times.to(device=target.device, dtype=target.dtype)
    while values.dim() < target.dim():
        values = values.unsqueeze(-1)
    return values


def flow_interpolate(
    clean_blocks: torch.Tensor, noise: torch.Tensor, times: torch.Tensor
) -> torch.Tensor:
    t = expand_flow_time(times, clean_blocks)
    return t * clean_blocks + (1.0 - t) * noise


def flow_velocity_from_x(
    pred_clean: torch.Tensor, current: torch.Tensor, times: torch.Tensor
) -> torch.Tensor:
    t = expand_flow_time(times, current)
    return (pred_clean - current) / (1.0 - t).clamp_min(1e-4)


def flow_matching_loss(
    pred_clean: torch.Tensor,
    clean_blocks: torch.Tensor,
    current: torch.Tensor,
    noise: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    pred_velocity = flow_velocity_from_x(pred_clean, current, times)
    target_velocity = clean_blocks - noise
    return torch.nn.functional.mse_loss(pred_velocity, target_velocity)


def masked_flow_matching_loss(
    pred_clean: torch.Tensor,
    clean_blocks: torch.Tensor,
    current: torch.Tensor,
    noise: torch.Tensor,
    times: torch.Tensor,
    block_mask: torch.Tensor,
) -> torch.Tensor:
    pred_velocity = flow_velocity_from_x(pred_clean, current, times)
    target_velocity = clean_blocks - noise
    mask = block_mask.to(dtype=torch.bool).view(pred_clean.shape[0], pred_clean.shape[1], 1, 1)
    squared = torch.square(pred_velocity - target_velocity)
    return squared.masked_select(mask.expand_as(squared)).mean()


def sample_active_flow_blocks(block_mask: torch.Tensor) -> torch.Tensor:
    active_indices: list[torch.Tensor] = []
    for row in block_mask.to(dtype=torch.bool):
        candidates = torch.nonzero(row, as_tuple=False).flatten()
        if candidates.numel() == 0:
            candidates = torch.arange(row.numel(), device=row.device)
        selected = candidates[torch.randint(candidates.numel(), (1,), device=row.device)]
        active_indices.append(selected)
    return torch.cat(active_indices)


def active_block_mask(block_indices: torch.Tensor, b_max: int) -> torch.Tensor:
    mask = torch.zeros(block_indices.shape[0], b_max, dtype=torch.bool, device=block_indices.device)
    mask.scatter_(1, block_indices.view(-1, 1), True)
    return mask


def flow_euler_step(
    current: torch.Tensor,
    pred_clean: torch.Tensor,
    current_time: torch.Tensor,
    next_time: float,
) -> torch.Tensor:
    velocity = flow_velocity_from_x(pred_clean, current, current_time)
    dt = torch.as_tensor(next_time, device=current.device, dtype=current.dtype)
    dt = dt - current_time.to(device=current.device, dtype=current.dtype)
    while dt.dim() < current.dim():
        dt = dt.unsqueeze(-1)
    return current + dt * velocity


def training_prior_prediction(
    model: BlockCausalPrior,
    batch: dict[str, Any],
    previous_context: torch.Tensor,
    question_features: torch.Tensor | None,
    *,
    config: StageCPriorTrainConfig,
    flow_schedule: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if config.prior_mode == "deterministic":
        pred = model(
            batch["question_ids"],
            batch["question_mask"],
            previous_context,
            question_features,
        )
        return pred, model.loss(pred, batch["z_blocks"]), {}
    if flow_schedule is None:
        raise ValueError("flow_schedule is required for flow prior")
    times = sample_flow_times(
        batch["z_blocks"].shape[0], config, device=batch["z_blocks"].device
    )
    noise = torch.randn_like(batch["z_blocks"])
    noisy_blocks = flow_interpolate(batch["z_blocks"], noise, times)
    block_indices = sample_active_flow_blocks(batch["block_mask"])
    current_block_mask = active_block_mask(block_indices, model.config.b_max)
    noisy_blocks = torch.where(
        current_block_mask.view(noisy_blocks.shape[0], noisy_blocks.shape[1], 1, 1),
        noisy_blocks,
        torch.zeros_like(noisy_blocks),
    )
    pred = model(
        batch["question_ids"],
        batch["question_mask"],
        previous_context,
        question_features,
        noisy_blocks,
        times * config.flow_time_max,
    )
    prior_loss = masked_flow_matching_loss(
        pred, batch["z_blocks"], noisy_blocks, noise, times, current_block_mask
    )
    metrics = {
        "flow_t_mean": float(times.detach().mean().cpu().item()),
        "flow_t_std": float(times.detach().std(unbiased=False).cpu().item()),
        "flow_block_mean": float(block_indices.detach().float().mean().cpu().item() + 1.0),
    }
    return pred, prior_loss, metrics


@torch.no_grad()
def flow_teacher_predict(
    model: BlockCausalPrior,
    batch: dict[str, Any],
    question_features: torch.Tensor | None,
    *,
    config: StageCPriorTrainConfig,
    flow_schedule: dict[str, torch.Tensor],
) -> torch.Tensor:
    sample = torch.randn_like(batch["z_blocks"])
    times = flow_schedule["infer_times"].to(device=sample.device)
    batch_size = sample.shape[0]
    previous_context = batch["z_blocks"].detach()
    for block_index in range(model.config.b_max):
        current = sample[:, block_index]
        for current_time, next_time in zip(times[:-1], times[1:]):
            noisy_blocks = torch.zeros_like(sample)
            noisy_blocks[:, block_index] = current
            time_tensor = torch.full(
                (batch_size,), float(current_time.item()), device=sample.device, dtype=torch.float32
            )
            pred_clean = model(
                batch["question_ids"],
                batch["question_mask"],
                previous_context,
                question_features,
                noisy_blocks,
                time_tensor * config.flow_time_max,
            )[:, block_index]
            current = flow_euler_step(current, pred_clean, time_tensor, float(next_time.item()))
        sample[:, block_index] = current
    return sample


@torch.no_grad()
def rollout_flow_prior(
    model: BlockCausalPrior,
    question_ids: torch.Tensor,
    question_mask: torch.Tensor,
    question_features: torch.Tensor | None,
    *,
    config: StageCPriorTrainConfig,
    flow_schedule: dict[str, torch.Tensor],
) -> torch.Tensor:
    generated = torch.zeros(
        question_ids.shape[0],
        model.config.b_max,
        model.config.block_size,
        model.config.latent_dim,
        device=question_ids.device,
    )
    times = flow_schedule["infer_times"].to(device=question_ids.device)
    batch_size = question_ids.shape[0]
    for block_index in range(model.config.b_max):
        current = torch.randn(
            batch_size,
            model.config.block_size,
            model.config.latent_dim,
            device=question_ids.device,
        )
        for current_time, next_time in zip(times[:-1], times[1:]):
            noisy_blocks = torch.zeros_like(generated)
            noisy_blocks[:, block_index] = current
            time_tensor = torch.full(
                (batch_size,),
                float(current_time.item()),
                device=question_ids.device,
                dtype=torch.float32,
            )
            pred = model(
                question_ids,
                question_mask,
                generated,
                question_features,
                noisy_blocks,
                time_tensor * config.flow_time_max,
            )[:, block_index]
            current = flow_euler_step(current, pred, time_tensor, float(next_time.item()))
        generated[:, block_index] = current
    return generated


def stage_c_gate_notes(config: StageCPriorTrainConfig) -> str:
    if config.prior_mode == "flow" and config.joint_stage_b:
        return (
            "Cola-style Flow Matching prior with joint Stage B encoder/decoder "
            "updates and cached-latent reference regularization."
        )
    if config.prior_mode == "flow":
        return (
            "Fixed-Stage-B Flow Matching prior diagnostic; main Cola-style claim "
            "still requires joint Stage B training."
        )
    return (
        "Deterministic block-causal x0 regression baseline; use flow mode "
        "with joint Stage B for the main Cola-style prior claim."
    )


@torch.no_grad()
def build_previous_context(
    model: BlockCausalPrior,
    batch: dict[str, Any],
    mix_probability: float,
    question_features: torch.Tensor | None = None,
    *,
    config: StageCPriorTrainConfig | None = None,
    flow_schedule: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    if mix_probability <= 0:
        return batch["z_blocks"].detach()
    if config is not None and config.prior_mode == "flow":
        if flow_schedule is None:
            raise ValueError("flow_schedule is required for flow scheduled sampling")
        generated = rollout_flow_prior(
            model,
            batch["question_ids"],
            batch["question_mask"],
            question_features,
            config=config,
            flow_schedule=flow_schedule,
        ).detach()
    else:
        generated = model(
            batch["question_ids"], batch["question_mask"], batch["z_blocks"], question_features
        ).detach()
    block_mask = torch.rand(
        batch["z_blocks"].shape[0],
        model.config.b_max,
        1,
        1,
        device=batch["z_blocks"].device,
    ) < mix_probability
    return torch.where(block_mask, generated, batch["z_blocks"]).detach()


@torch.no_grad()
def evaluate_prior(
    model: BlockCausalPrior,
    stage_b: StageBReasoningAutoencoder,
    loader: DataLoader,
    *,
    tokenizer: Any,
    config: StageCPriorTrainConfig,
    device: torch.device,
    condition_encoder: "FrozenConditionEncoder | None" = None,
    flow_schedule: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    model.eval()
    stage_b.eval()
    totals: dict[str, float] = {}
    latent_stats: dict[str, dict[str, float]] = {}
    teacher_block_mse_sum = torch.zeros(stage_b.config.b_max, dtype=torch.float64)
    rollout_block_mse_sum = torch.zeros(stage_b.config.b_max, dtype=torch.float64)
    count = 0
    gold_correct = {b: 0 for b in config.fixed_b_values}
    teacher_correct = {b: 0 for b in config.fixed_b_values}
    rollout_correct = {b: 0 for b in config.fixed_b_values}
    for batch in loader:
        batch = move_batch(batch, device)
        question_features = encode_condition_features(condition_encoder, batch)
        batch = prepare_prior_eval_batch(stage_b, batch, tokenizer=tokenizer, config=config)
        if config.prior_mode == "flow":
            if flow_schedule is None:
                raise ValueError("flow_schedule is required for flow evaluation")
            teacher_pred = flow_teacher_predict(
                model,
                batch,
                question_features,
                config=config,
                flow_schedule=flow_schedule,
            )
            rolled = rollout_flow_prior(
                model,
                batch["question_ids"],
                batch["question_mask"],
                question_features,
                config=config,
                flow_schedule=flow_schedule,
            )
        else:
            teacher_pred = model(
                batch["question_ids"],
                batch["question_mask"],
                batch["z_blocks"],
                question_features,
            )
            rolled = rollout_prior(
                model, batch["question_ids"], batch["question_mask"], question_features
            )
        batch_size = batch["z_blocks"].shape[0]
        teacher_block_mse_sum += blockwise_mse(teacher_pred, batch["z_blocks"]).sum(dim=0).detach().cpu()
        rollout_block_mse_sum += blockwise_mse(rolled, batch["z_blocks"]).sum(dim=0).detach().cpu()
        totals["prior_teacher_mse"] = totals.get("prior_teacher_mse", 0.0) + torch.nn.functional.mse_loss(
            teacher_pred, batch["z_blocks"]
        ).item() * batch_size
        totals["prior_rollout_mse"] = totals.get("prior_rollout_mse", 0.0) + torch.nn.functional.mse_loss(
            rolled, batch["z_blocks"]
        ).item() * batch_size
        totals["teacher_gold_cosine"] = totals.get("teacher_gold_cosine", 0.0) + samplewise_cosine(
            teacher_pred, batch["z_blocks"]
        ).sum().item()
        totals["rollout_gold_cosine"] = totals.get("rollout_gold_cosine", 0.0) + samplewise_cosine(
            rolled, batch["z_blocks"]
        ).sum().item()
        accumulate_latent_stats(latent_stats, "gold", batch["z_blocks"])
        accumulate_latent_stats(latent_stats, "teacher", teacher_pred)
        accumulate_latent_stats(latent_stats, "rollout", rolled)
        count += batch_size
        for b in config.fixed_b_values:
            gold_texts = decode_fixed_b(stage_b, batch["z_blocks"], batch, tokenizer=tokenizer, fixed_b=b)
            teacher_texts = decode_fixed_b(stage_b, teacher_pred, batch, tokenizer=tokenizer, fixed_b=b)
            rollout_texts = decode_fixed_b(stage_b, rolled, batch, tokenizer=tokenizer, fixed_b=b)
            for gold_text, teacher_text, rollout_text, gold in zip(
                gold_texts, teacher_texts, rollout_texts, batch["answer_norms"]
            ):
                gold_correct[b] += int(judge(gold_text, gold)["correct"])
                teacher_correct[b] += int(judge(teacher_text, gold)["correct"])
                rollout_correct[b] += int(judge(rollout_text, gold)["correct"])

    metrics = {key: value / max(count, 1) for key, value in totals.items()}
    for block_index in range(stage_b.config.b_max):
        block = block_index + 1
        metrics[f"prior_teacher_block{block}_mse"] = float(
            teacher_block_mse_sum[block_index].item() / max(count, 1)
        )
        metrics[f"prior_rollout_block{block}_mse"] = float(
            rollout_block_mse_sum[block_index].item() / max(count, 1)
        )
    metrics.update(finalize_latent_stats(latent_stats))
    for b, value in gold_correct.items():
        metrics[f"gold_fixed_b{b}_acc"] = value / max(count, 1)
    for b, value in teacher_correct.items():
        metrics[f"teacher_fixed_b{b}_acc"] = value / max(count, 1)
    for b, value in rollout_correct.items():
        metrics[f"rollout_fixed_b{b}_acc"] = value / max(count, 1)
        metrics[f"fixed_b{b}_acc"] = metrics[f"rollout_fixed_b{b}_acc"]
    return metrics


def blockwise_mse(pred_blocks: torch.Tensor, target_blocks: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred_blocks - target_blocks) ** 2, dim=(2, 3))


def samplewise_cosine(pred_blocks: torch.Tensor, target_blocks: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(
        pred_blocks.flatten(start_dim=1),
        target_blocks.flatten(start_dim=1),
        dim=1,
    )


def accumulate_latent_stats(
    stats: dict[str, dict[str, float]], prefix: str, z_blocks: torch.Tensor
) -> None:
    values = z_blocks.detach().float()
    entry = stats.setdefault(
        prefix,
        {
            "sum": 0.0,
            "sumsq": 0.0,
            "numel": 0.0,
            "norm_sum": 0.0,
            "norm_sumsq": 0.0,
            "norm_count": 0.0,
        },
    )
    entry["sum"] += values.sum().item()
    entry["sumsq"] += torch.square(values).sum().item()
    entry["numel"] += float(values.numel())
    norms = values.norm(dim=-1)
    entry["norm_sum"] += norms.sum().item()
    entry["norm_sumsq"] += torch.square(norms).sum().item()
    entry["norm_count"] += float(norms.numel())


def finalize_latent_stats(stats: dict[str, dict[str, float]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for prefix, entry in stats.items():
        mean = entry["sum"] / max(entry["numel"], 1.0)
        norm_mean = entry["norm_sum"] / max(entry["norm_count"], 1.0)
        var = max(entry["sumsq"] / max(entry["numel"], 1.0) - mean * mean, 0.0)
        norm_var = max(
            entry["norm_sumsq"] / max(entry["norm_count"], 1.0) - norm_mean * norm_mean,
            0.0,
        )
        metrics[f"{prefix}_latent_mean"] = mean
        metrics[f"{prefix}_latent_std"] = math.sqrt(var)
        metrics[f"{prefix}_latent_norm_mean"] = norm_mean
        metrics[f"{prefix}_latent_norm_std"] = math.sqrt(norm_var)
    return metrics


@torch.no_grad()
def decode_fixed_b(
    stage_b: StageBReasoningAutoencoder,
    z_blocks: torch.Tensor,
    batch: dict[str, Any],
    *,
    tokenizer: Any,
    fixed_b: int,
) -> list[str]:
    fixed_b = min(fixed_b, stage_b.config.b_max)
    z_used = stage_b.zero_trailing_blocks(z_blocks, torch.ones_like(batch["noop_mask"]))
    z_used[:, :fixed_b] = z_blocks[:, :fixed_b]
    q_pool = stage_b.encode_question(batch["question_ids"], batch["question_mask"])
    logits = stage_b.decode_tokens(z_used, q_pool)
    pred = logits.argmax(dim=-1)[:, : fixed_b * stage_b.config.block_size]
    texts: list[str] = []
    for row in pred.detach().cpu().tolist():
        texts.append(tokenizer.decode([int(token_id) for token_id in row], skip_special_tokens=True))
    return texts


def decoder_recon_loss(
    stage_b: StageBReasoningAutoencoder,
    z_blocks: torch.Tensor,
    batch: dict[str, Any],
) -> torch.Tensor:
    q_pool = stage_b.encode_question(batch["question_ids"], batch["question_mask"])
    logits = stage_b.decode_tokens(z_blocks, q_pool)
    labels = torch.full(
        logits.shape[:2],
        -100,
        dtype=torch.long,
        device=logits.device,
    )
    copy_len = min(logits.shape[1], batch["target_ids"].shape[1])
    labels[:, :copy_len] = batch["target_ids"][:, :copy_len]
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[:, :copy_len] = batch["target_mask"][:, :copy_len].bool()
    labels[~mask] = -100
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )


def load_stage_b_decoder(checkpoint_path: str, *, device: torch.device) -> StageBReasoningAutoencoder:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = StageBReasoningAutoencoder(StageBModelConfig(**checkpoint["model_config"]))
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    allowed_missing_prefixes = ("question_latent_pos.", "question_to_latent.")
    disallowed_missing = [
        name for name in missing if not name.startswith(allowed_missing_prefixes)
    ]
    if unexpected or disallowed_missing:
        raise RuntimeError(
            "Stage B checkpoint is incompatible: "
            f"missing={disallowed_missing}, unexpected={unexpected}"
        )
    return model.to(device)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def parse_args() -> StageCPriorTrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-b-checkpoint", default=StageCPriorTrainConfig.stage_b_checkpoint)
    parser.add_argument("--train-latent-dir", default=StageCPriorTrainConfig.train_latent_dir)
    parser.add_argument("--eval-latent-dir", default=StageCPriorTrainConfig.eval_latent_dir)
    parser.add_argument("--output-dir", default=StageCPriorTrainConfig.output_dir)
    parser.add_argument("--tokenizer-name", default=StageCPriorTrainConfig.tokenizer_name)
    parser.add_argument("--allow-tokenizer-download", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--hidden-dim", type=int, default=StageCPriorTrainConfig.hidden_dim)
    parser.add_argument("--num-layers", type=int, default=StageCPriorTrainConfig.num_layers)
    parser.add_argument("--num-heads", type=int, default=StageCPriorTrainConfig.num_heads)
    parser.add_argument("--dropout", type=float, default=StageCPriorTrainConfig.dropout)
    parser.add_argument("--batch-size", type=int, default=StageCPriorTrainConfig.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=StageCPriorTrainConfig.eval_batch_size)
    parser.add_argument("--epochs", type=int, default=StageCPriorTrainConfig.epochs)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--lr", type=float, default=StageCPriorTrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=StageCPriorTrainConfig.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=StageCPriorTrainConfig.grad_clip)
    parser.add_argument("--decode-weight", type=float, default=StageCPriorTrainConfig.decode_weight)
    parser.add_argument("--decode-warmup-steps", type=int, default=StageCPriorTrainConfig.decode_warmup_steps)
    parser.add_argument("--decode-ramp-steps", type=int, default=StageCPriorTrainConfig.decode_ramp_steps)
    parser.add_argument("--previous-mix-final", type=float, default=StageCPriorTrainConfig.previous_mix_final)
    parser.add_argument("--previous-mix-warmup-steps", type=int, default=StageCPriorTrainConfig.previous_mix_warmup_steps)
    parser.add_argument("--previous-mix-ramp-steps", type=int, default=StageCPriorTrainConfig.previous_mix_ramp_steps)
    parser.add_argument("--fixed-b-values", type=int, nargs="+", default=list(StageCPriorTrainConfig.fixed_b_values))
    parser.add_argument("--condition-encoder", choices=["learned", "qwen"], default=StageCPriorTrainConfig.condition_encoder)
    parser.add_argument("--condition-model-name", default=StageCPriorTrainConfig.condition_model_name)
    parser.add_argument("--condition-dtype", default=StageCPriorTrainConfig.condition_dtype)
    parser.add_argument("--prior-mode", choices=["deterministic", "flow"], default=StageCPriorTrainConfig.prior_mode)
    parser.add_argument("--flow-time-max", type=float, default=StageCPriorTrainConfig.flow_time_max)
    parser.add_argument("--flow-time-loc", type=float, default=StageCPriorTrainConfig.flow_time_loc)
    parser.add_argument("--flow-time-scale", type=float, default=StageCPriorTrainConfig.flow_time_scale)
    parser.add_argument("--flow-denoise-steps", type=int, default=StageCPriorTrainConfig.flow_denoise_steps)
    parser.add_argument("--joint-stage-b", action="store_true")
    parser.add_argument("--stage-b-lr-ratio", type=float, default=StageCPriorTrainConfig.stage_b_lr_ratio)
    parser.add_argument("--stage-b-vae-weight", type=float, default=StageCPriorTrainConfig.stage_b_vae_weight)
    parser.add_argument("--stage-b-ref-weight", type=float, default=StageCPriorTrainConfig.stage_b_ref_weight)
    parser.add_argument(
        "--stage-b-question-latent-weight",
        type=float,
        default=StageCPriorTrainConfig.stage_b_question_latent_weight,
    )
    parser.add_argument("--seed", type=int, default=StageCPriorTrainConfig.seed)
    parser.add_argument("--device", default=StageCPriorTrainConfig.device)
    parser.add_argument("--swanlab-mode")
    args = parser.parse_args()
    return StageCPriorTrainConfig(
        stage_b_checkpoint=args.stage_b_checkpoint,
        train_latent_dir=args.train_latent_dir,
        eval_latent_dir=args.eval_latent_dir,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer_name,
        local_files_only=not args.allow_tokenizer_download,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        max_steps=args.max_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        decode_weight=args.decode_weight,
        decode_warmup_steps=args.decode_warmup_steps,
        decode_ramp_steps=args.decode_ramp_steps,
        previous_mix_final=args.previous_mix_final,
        previous_mix_warmup_steps=args.previous_mix_warmup_steps,
        previous_mix_ramp_steps=args.previous_mix_ramp_steps,
        fixed_b_values=tuple(args.fixed_b_values),
        condition_encoder=args.condition_encoder,
        condition_model_name=args.condition_model_name,
        condition_dtype=args.condition_dtype,
        prior_mode=args.prior_mode,
        flow_time_max=args.flow_time_max,
        flow_time_loc=args.flow_time_loc,
        flow_time_scale=args.flow_time_scale,
        flow_denoise_steps=args.flow_denoise_steps,
        joint_stage_b=args.joint_stage_b,
        stage_b_lr_ratio=args.stage_b_lr_ratio,
        stage_b_vae_weight=args.stage_b_vae_weight,
        stage_b_ref_weight=args.stage_b_ref_weight,
        stage_b_question_latent_weight=args.stage_b_question_latent_weight,
        seed=args.seed,
        device=args.device,
        swanlab_mode=args.swanlab_mode,
    )


def main() -> None:
    summary = train_stage_c_prior(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
