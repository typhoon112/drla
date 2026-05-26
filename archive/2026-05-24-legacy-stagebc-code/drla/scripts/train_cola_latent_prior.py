"""Diagnostic custom prior in official Cola VAE latent space.

The main Stage C path is official Cola DiT LoRA/adapter training. This
script is kept to reproduce small-prior diagnostics and overfit checks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

from drla.data.answer_judge import judge
from drla.models.stage_c import BlockCausalPrior, StageCPriorConfig
from drla.tracking import finish_experiment, init_experiment, log_metrics
from drla.training.stage_b_autoencoder import move_batch

try:
    from cola_dlm import ColaTextVAEModel
except ImportError as exc:  # pragma: no cover - integration-only dependency.
    raise ImportError("Set PYTHONPATH to the official Cola-DLM code directory before running this script.") from exc


@dataclass(frozen=True)
class ColaLatentPriorConfig:
    vae_path: str = "/data1/luyifei/drla/models/Cola-DLM/cola_dlm/cola_vae"
    tokenizer_path: str = "/data1/luyifei/drla/models/Cola-DLM/tokenizer.json"
    train_jsonl: str = "/data1/luyifei/drla/outputs/cola_gsm8k_eval/gsm8k_qa_test_64.jsonl"
    eval_jsonl: str = "/data1/luyifei/drla/outputs/cola_gsm8k_eval/gsm8k_qa_test_64.jsonl"
    output_dir: str = "/data1/luyifei/drla/outputs/cola_latent_prior_overfit64"
    max_train_samples: int = 64
    max_eval_samples: int = 64
    b_max: int = 32
    block_size: int = 16
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    question_layers: int = 1
    dropout: float = 0.1
    batch_size: int = 4
    eval_batch_size: int = 4
    max_steps: int = 300
    eval_every: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    flow_time_max: float = 1000.0
    flow_time_loc: float = 1.0
    flow_time_scale: float = 0.0
    flow_denoise_steps: int = 16
    max_question_len: int = 384
    pad_token_id: int = 100277
    seed: int = 42
    device: str = "auto"
    swanlab_mode: str | None = None
    experiment_name: str = "cola-latent-prior-flow"
    resume_checkpoint: str | None = None
    condition_encoder: str = "learned"
    condition_model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    condition_dtype: str = "bfloat16"
    local_files_only: bool = True
    max_condition_len: int = 384

    @property
    def capacity(self) -> int:
        return self.b_max * self.block_size


class ColaLatentDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class ColaLatentCollator:
    def __init__(self, *, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        question_ids, question_mask = pad_sequences(
            [row["question_ids"] for row in rows], pad_id=self.pad_token_id
        )
        batch = {
            "ids": [row["id"] for row in rows],
            "question_ids": question_ids,
            "question_mask": question_mask,
            "z_blocks": torch.stack([row["z_blocks"] for row in rows]),
            "token_mask": torch.stack([row["token_mask"] for row in rows]),
            "block_mask": torch.stack([row["block_mask"] for row in rows]),
            "full_ids": [row["full_ids"] for row in rows],
            "full_lengths": torch.tensor([len(row["full_ids"]) for row in rows], dtype=torch.long),
            "answer_norms": [row["answer_norm"] for row in rows],
            "gold_texts": [row["gold_text"] for row in rows],
        }
        if "question_features" in rows[0]:
            question_features, question_feature_mask = pad_feature_sequences(
                [row["question_features"] for row in rows]
            )
            batch["question_features"] = question_features
            batch["question_feature_mask"] = question_feature_mask
        return batch


class FrozenTextConditionEncoder:
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
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return outputs.last_hidden_state.float()


def ensure_cloud_training(mode: str | None) -> str:
    requested = mode or os.getenv("SWANLAB_MODE") or "cloud"
    if requested != "cloud":
        raise ValueError("Training must be logged to SwanLab cloud. Use --swanlab-mode cloud.")
    return requested


def train_cola_latent_prior(config: ColaLatentPriorConfig) -> dict[str, Any]:
    set_seed(config.seed)
    mode = ensure_cloud_training(config.swanlab_mode)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    condition_encoder, condition_tokenizer = load_condition_encoder(config, device=device)
    vae = ColaTextVAEModel.from_pretrained(config.vae_path).to(device)
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    train_rows = build_latent_rows(
        Path(config.train_jsonl),
        tokenizer=tokenizer,
        vae=vae,
        config=config,
        max_samples=config.max_train_samples,
        device=device,
        condition_encoder=condition_encoder,
        condition_tokenizer=condition_tokenizer,
    )
    eval_rows = build_latent_rows(
        Path(config.eval_jsonl),
        tokenizer=tokenizer,
        vae=vae,
        config=config,
        max_samples=config.max_eval_samples,
        device=device,
        condition_encoder=condition_encoder,
        condition_tokenizer=condition_tokenizer,
    )
    if not train_rows or not eval_rows:
        raise ValueError("No Cola latent rows were built; check capacity and input JSONL.")

    prior_config = StageCPriorConfig(
        vocab_size=tokenizer.get_vocab_size(),
        b_max=config.b_max,
        block_size=config.block_size,
        latent_dim=vae.latent_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        pad_id=config.pad_token_id,
        max_question_len=config.max_question_len,
        question_layers=config.question_layers,
        condition_dim=condition_encoder.hidden_size if condition_encoder is not None else None,
    )
    if condition_encoder is not None:
        del condition_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    model = BlockCausalPrior(prior_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    global_step = 0
    resume_metrics: dict[str, float] = {}
    if config.resume_checkpoint:
        resume_payload = torch.load(config.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(resume_payload["model_state_dict"])
        global_step = int(resume_payload.get("global_step", 0))
        resume_metrics = dict(resume_payload.get("metrics") or {})

    train_loader = make_loader(train_rows, config=config, batch_size=config.batch_size, shuffle=True)
    eval_loader = make_loader(eval_rows, config=config, batch_size=config.eval_batch_size, shuffle=False)
    flow_schedule = build_flow_schedule(config, device=device)
    effective_eval_every = resolve_eval_every(config.eval_every)

    run = init_experiment(
        stage="cola-latent-prior",
        experiment_name=config.experiment_name,
        description="Block-causal flow prior trained directly on official Cola VAE GSM8K latents.",
        config={
            **asdict(config),
            "swanlab_mode": mode,
            "num_train_rows": len(train_rows),
            "num_eval_rows": len(eval_rows),
            "effective_eval_every": effective_eval_every,
            "prior_config": asdict(prior_config),
        },
        mode=mode,
        tags=["cola", "flow", "gsm8k", "latent-prior"],
    )

    best_score = float("-inf")
    best_metrics: dict[str, float] = {}
    best_step = 0
    best_checkpoint_path = output_dir / "best_checkpoint.pt"
    if resume_metrics:
        best_score = float(resume_metrics.get("rollout_answer_accuracy", float("-inf")))
        best_metrics = dict(resume_metrics)
        best_step = global_step
        save_checkpoint(
            best_checkpoint_path,
            model=model,
            prior_config=prior_config,
            train_config=config,
            global_step=global_step,
            metrics=best_metrics,
        )
    final_train_metrics: dict[str, float] = {}
    final_eval_metrics: dict[str, float] = {}
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    train_iter = cycling_loader(train_loader)
    started_at = int(time.time())
    try:
        while global_step < config.max_steps:
            global_step += 1
            batch = move_batch(next(train_iter), device)
            model.train()
            pred, loss, train_metrics = training_step_prediction(model, batch, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            final_train_metrics = {
                "loss": float(loss.detach().cpu().item()),
                **train_metrics,
                "latent_mse": masked_latent_mse(pred, batch["z_blocks"], batch["token_mask"]),
            }
            log_metrics(final_train_metrics, step=global_step, prefix="train")
            append_metrics_jsonl(metrics_path, step=global_step, split="train", metrics=final_train_metrics)

            if (
                global_step == 1
                or global_step % effective_eval_every == 0
                or global_step == config.max_steps
            ):
                final_eval_metrics = evaluate_prior(
                    model,
                    vae,
                    eval_loader,
                    tokenizer=tokenizer,
                    config=config,
                    flow_schedule=flow_schedule,
                    device=device,
                )
                log_metrics(final_eval_metrics, step=global_step, prefix="valid")
                append_metrics_jsonl(metrics_path, step=global_step, split="valid", metrics=final_eval_metrics)
                score = final_eval_metrics["rollout_answer_accuracy"]
                if score > best_score:
                    best_score = score
                    best_step = global_step
                    best_metrics = dict(final_eval_metrics)
                    save_checkpoint(
                        best_checkpoint_path,
                        model=model,
                        prior_config=prior_config,
                        train_config=config,
                        global_step=global_step,
                        metrics=final_eval_metrics,
                    )
    finally:
        finish_experiment()

    checkpoint_path = output_dir / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        prior_config=prior_config,
        train_config=config,
        global_step=global_step,
        metrics=final_eval_metrics,
    )
    summary = {
        "created_at": int(time.time()),
        "started_at": started_at,
        "global_step": global_step,
        "checkpoint": str(best_checkpoint_path),
        "selected_checkpoint": str(best_checkpoint_path),
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(checkpoint_path),
        "metrics_jsonl": str(metrics_path),
        "best_step": best_step,
        "best_valid": best_metrics,
        "final_train": final_train_metrics,
        "final_valid": final_eval_metrics,
        "effective_eval_every": effective_eval_every,
        "swanlab_run_id": getattr(run, "id", None),
        "config": asdict(config),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


@torch.no_grad()
def build_latent_rows(
    jsonl_path: Path,
    *,
    tokenizer: Tokenizer,
    vae: Any,
    config: ColaLatentPriorConfig,
    max_samples: int,
    device: torch.device,
    condition_encoder: FrozenTextConditionEncoder | None = None,
    condition_tokenizer: Any | None = None,
) -> list[dict[str, Any]]:
    raw_rows = read_jsonl(jsonl_path, max_samples)
    rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    input_ids_list: list[torch.Tensor] = []
    encode_batch_size = max(1, min(config.eval_batch_size, 8))

    def flush() -> None:
        if not pending:
            return
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            enc = vae.encode(input_ids_list)
        condition_features = None
        condition_lengths: list[int] = []
        if condition_encoder is not None:
            if condition_tokenizer is None:
                raise ValueError("condition_tokenizer is required when condition_encoder is enabled")
            encoded = condition_tokenizer(
                [item["_condition_prompt"] for item in pending],
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=config.max_condition_len,
                return_tensors="pt",
            )
            condition_input_ids = encoded["input_ids"].to(device)
            condition_attention_mask = encoded["attention_mask"].to(device)
            condition_features = condition_encoder.encode(
                condition_input_ids, condition_attention_mask
            ).detach().cpu()
            condition_lengths = condition_attention_mask.sum(dim=1).detach().cpu().tolist()
        for item, latents in zip(pending, enc.latents_list):
            z = ((latents - vae.shifting_factor) * vae.scaling_factor).float().cpu()
            item["z_blocks"] = z.view(config.b_max, config.block_size, vae.latent_dim)
        if condition_features is not None:
            for item, features, length in zip(pending, condition_features, condition_lengths):
                item["question_features"] = features[: int(length)].contiguous()
        for item in pending:
            item.pop("_condition_prompt", None)
            rows.append(item)
        pending.clear()
        input_ids_list.clear()

    for row in raw_rows:
        prompt = normalize_prompt(str(row.get("question") or row.get("prompt") or ""))
        answer = str(row.get("ground_truth") or row.get("answer") or "").strip()
        target_text = str(row.get("target_text") or answer).strip()
        gold_text = f"{prompt} {target_text}".strip()
        question_ids = tokenizer.encode(prompt).ids
        full_ids = tokenizer.encode(gold_text).ids
        if len(question_ids) > config.max_question_len or len(full_ids) > config.capacity:
            continue
        padded_full_ids = full_ids + [config.pad_token_id] * (config.capacity - len(full_ids))
        token_mask = torch.zeros(config.capacity, dtype=torch.bool)
        token_mask[: len(full_ids)] = True
        token_mask_blocks = token_mask.view(config.b_max, config.block_size)
        block_mask = token_mask_blocks.any(dim=1)
        pending.append(
            {
                "id": row.get("id"),
                "question_ids": question_ids,
                "full_ids": full_ids,
                "token_mask": token_mask_blocks,
                "block_mask": block_mask,
                "answer_norm": answer,
                "gold_text": gold_text,
                "_condition_prompt": prompt,
            }
        )
        input_ids_list.append(torch.tensor(padded_full_ids, dtype=torch.long, device=device))
        if len(pending) >= encode_batch_size:
            flush()
    flush()
    return rows


def training_step_prediction(
    model: BlockCausalPrior,
    batch: dict[str, Any],
    config: ColaLatentPriorConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    times = sample_flow_times(batch["z_blocks"].shape[0], config, device=batch["z_blocks"].device)
    noise = torch.randn_like(batch["z_blocks"])
    noisy_blocks = flow_interpolate(batch["z_blocks"], noise, times)
    block_indices = sample_active_blocks(batch["block_mask"])
    current_block_mask = active_block_mask(block_indices, config.b_max)
    noisy_blocks = torch.where(
        current_block_mask.view(noisy_blocks.shape[0], noisy_blocks.shape[1], 1, 1),
        noisy_blocks,
        torch.zeros_like(noisy_blocks),
    )
    question_ids, question_mask, question_features = prior_condition_inputs(batch)
    pred = model(
        question_ids,
        question_mask,
        batch["z_blocks"].detach(),
        question_features=question_features,
        noisy_blocks=noisy_blocks,
        timesteps=times * config.flow_time_max,
    )
    active_token_mask = batch["token_mask"] & current_block_mask.view(current_block_mask.shape[0], -1, 1)
    loss = token_masked_flow_loss(pred, batch["z_blocks"], noisy_blocks, noise, times, active_token_mask)
    metrics = {
        "flow_t_mean": float(times.detach().mean().cpu().item()),
        "flow_t_std": float(times.detach().std(unbiased=False).cpu().item()),
        "flow_block_mean": float(block_indices.detach().float().mean().cpu().item() + 1.0),
    }
    return pred, loss, metrics


@torch.no_grad()
def evaluate_prior(
    model: BlockCausalPrior,
    vae: Any,
    loader: DataLoader,
    *,
    tokenizer: Tokenizer,
    config: ColaLatentPriorConfig,
    flow_schedule: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    vae.eval()
    totals = {
        "teacher_latent_mse": 0.0,
        "rollout_latent_mse": 0.0,
        "teacher_token_accuracy": 0.0,
        "rollout_token_accuracy": 0.0,
        "teacher_answer_accuracy": 0.0,
        "rollout_answer_accuracy": 0.0,
        "gold_answer_accuracy": 0.0,
    }
    count = 0

    for batch in loader:
        batch = move_batch(batch, device)
        batch_size = batch["z_blocks"].shape[0]
        teacher = teacher_flow_predict(model, batch, config=config, flow_schedule=flow_schedule)
        rollout = rollout_flow_prior(model, batch, config=config, flow_schedule=flow_schedule)
        gold_eval = decode_and_judge(vae, batch["z_blocks"], batch, tokenizer=tokenizer, config=config)
        teacher_eval = decode_and_judge(vae, teacher, batch, tokenizer=tokenizer, config=config)
        rollout_eval = decode_and_judge(vae, rollout, batch, tokenizer=tokenizer, config=config)
        metrics = {
            "teacher_latent_mse": masked_latent_mse(teacher, batch["z_blocks"], batch["token_mask"]),
            "rollout_latent_mse": masked_latent_mse(rollout, batch["z_blocks"], batch["token_mask"]),
            "teacher_token_accuracy": teacher_eval["token_accuracy"],
            "rollout_token_accuracy": rollout_eval["token_accuracy"],
            "teacher_answer_accuracy": teacher_eval["answer_accuracy"],
            "rollout_answer_accuracy": rollout_eval["answer_accuracy"],
            "gold_answer_accuracy": gold_eval["answer_accuracy"],
        }
        for key, value in metrics.items():
            totals[key] += value * batch_size
        count += batch_size

    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def teacher_flow_predict(
    model: BlockCausalPrior,
    batch: dict[str, Any],
    *,
    config: ColaLatentPriorConfig,
    flow_schedule: torch.Tensor,
) -> torch.Tensor:
    sample = torch.randn_like(batch["z_blocks"])
    output = sample.clone()
    question_ids, question_mask, question_features = prior_condition_inputs(batch)
    for block_index in range(config.b_max):
        current = sample[:, block_index]
        for current_time, next_time in zip(flow_schedule[:-1], flow_schedule[1:]):
            noisy_blocks = torch.zeros_like(sample)
            noisy_blocks[:, block_index] = current
            time_tensor = torch.full(
                (sample.shape[0],),
                float(current_time.item()),
                device=sample.device,
                dtype=torch.float32,
            )
            pred_clean = model(
                question_ids,
                question_mask,
                batch["z_blocks"],
                question_features=question_features,
                noisy_blocks=noisy_blocks,
                timesteps=time_tensor * config.flow_time_max,
            )[:, block_index]
            current = flow_euler_step(current, pred_clean, time_tensor, float(next_time.item()))
        output[:, block_index] = current
    return output


@torch.no_grad()
def rollout_flow_prior(
    model: BlockCausalPrior,
    batch: dict[str, Any],
    *,
    config: ColaLatentPriorConfig,
    flow_schedule: torch.Tensor,
) -> torch.Tensor:
    generated = torch.zeros_like(batch["z_blocks"])
    question_ids, question_mask, question_features = prior_condition_inputs(batch)
    for block_index in range(config.b_max):
        current = torch.randn(
            generated.shape[0],
            config.block_size,
            generated.shape[-1],
            device=generated.device,
        )
        for current_time, next_time in zip(flow_schedule[:-1], flow_schedule[1:]):
            noisy_blocks = torch.zeros_like(generated)
            noisy_blocks[:, block_index] = current
            time_tensor = torch.full(
                (generated.shape[0],),
                float(current_time.item()),
                device=generated.device,
                dtype=torch.float32,
            )
            pred_clean = model(
                question_ids,
                question_mask,
                generated,
                question_features=question_features,
                noisy_blocks=noisy_blocks,
                timesteps=time_tensor * config.flow_time_max,
            )[:, block_index]
            current = flow_euler_step(current, pred_clean, time_tensor, float(next_time.item()))
        generated[:, block_index] = current
    return generated


@torch.no_grad()
def decode_and_judge(
    vae: Any,
    z_blocks: torch.Tensor,
    batch: dict[str, Any],
    *,
    tokenizer: Tokenizer,
    config: ColaLatentPriorConfig,
) -> dict[str, float]:
    vae.set_kv_cache(False)
    flat = z_blocks.reshape(z_blocks.shape[0] * config.capacity, z_blocks.shape[-1])
    txt_shape = torch.full((z_blocks.shape[0], 1), config.capacity, dtype=torch.long, device=z_blocks.device)
    with torch.autocast(device_type=z_blocks.device.type, dtype=torch.bfloat16, enabled=z_blocks.device.type == "cuda"):
        logits = vae.decode(flat, txt_shape=txt_shape, txt_q_shape=txt_shape, update_kv=False)
    pred_flat = logits.argmax(dim=-1).squeeze(0).detach().cpu()

    token_correct = 0
    token_total = 0
    answer_correct = 0
    for i, (full_ids, length, gold) in enumerate(
        zip(batch["full_ids"], batch["full_lengths"].detach().cpu().tolist(), batch["answer_norms"])
    ):
        start = i * config.capacity
        pred_ids = pred_flat[start : start + int(length)].tolist()
        token_correct += sum(int(a == b) for a, b in zip(pred_ids, full_ids))
        token_total += int(length)
        decoded = tokenizer.decode(pred_ids, skip_special_tokens=False)
        answer_correct += int(judge(answer_part(decoded), gold)["correct"])
    return {
        "token_accuracy": token_correct / max(token_total, 1),
        "answer_accuracy": answer_correct / max(z_blocks.shape[0], 1),
    }


def token_masked_flow_loss(
    pred_clean: torch.Tensor,
    clean: torch.Tensor,
    current: torch.Tensor,
    noise: torch.Tensor,
    times: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    pred_velocity = flow_velocity_from_x(pred_clean, current, times)
    target_velocity = clean - noise
    squared = torch.square(pred_velocity - target_velocity)
    mask = token_mask.to(dtype=torch.bool).unsqueeze(-1).expand_as(squared)
    return squared.masked_select(mask).mean()


def masked_latent_mse(pred: torch.Tensor, target: torch.Tensor, token_mask: torch.Tensor) -> float:
    squared = torch.square(pred.detach() - target.detach())
    mask = token_mask.to(dtype=torch.bool).unsqueeze(-1).expand_as(squared)
    return float(squared.masked_select(mask).mean().cpu().item())


def flow_interpolate(clean: torch.Tensor, noise: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    t = expand_time(times, clean)
    return t * clean + (1.0 - t) * noise


def flow_velocity_from_x(pred_clean: torch.Tensor, current: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    t = expand_time(times, current)
    return (pred_clean - current) / (1.0 - t).clamp_min(1e-4)


def flow_euler_step(current: torch.Tensor, pred_clean: torch.Tensor, current_time: torch.Tensor, next_time: float) -> torch.Tensor:
    velocity = flow_velocity_from_x(pred_clean, current, current_time)
    dt = torch.as_tensor(next_time, device=current.device, dtype=current.dtype)
    dt = dt - current_time.to(device=current.device, dtype=current.dtype)
    while dt.dim() < current.dim():
        dt = dt.unsqueeze(-1)
    return current + dt * velocity


def expand_time(times: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    values = times.to(device=target.device, dtype=target.dtype)
    while values.dim() < target.dim():
        values = values.unsqueeze(-1)
    return values


def sample_flow_times(batch_size: int, config: ColaLatentPriorConfig, *, device: torch.device) -> torch.Tensor:
    if config.flow_time_scale == 0:
        raw = torch.full((batch_size,), config.flow_time_loc, device=device)
    else:
        raw = torch.randn(batch_size, device=device) * config.flow_time_scale + config.flow_time_loc
    return torch.sigmoid(raw).clamp(1e-4, 1.0 - 1e-4)


def sample_active_blocks(block_mask: torch.Tensor) -> torch.Tensor:
    active_indices: list[torch.Tensor] = []
    for row in block_mask.to(dtype=torch.bool):
        candidates = torch.nonzero(row, as_tuple=False).flatten()
        if candidates.numel() == 0:
            candidates = torch.arange(row.numel(), device=row.device)
        active_indices.append(candidates[torch.randint(candidates.numel(), (1,), device=row.device)])
    return torch.cat(active_indices)


def active_block_mask(block_indices: torch.Tensor, b_max: int) -> torch.Tensor:
    mask = torch.zeros(block_indices.shape[0], b_max, dtype=torch.bool, device=block_indices.device)
    mask.scatter_(1, block_indices.view(-1, 1), True)
    return mask


def build_flow_schedule(config: ColaLatentPriorConfig, *, device: torch.device) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, config.flow_denoise_steps + 1, device=device, dtype=torch.float32)


def normalize_prompt(prompt: str) -> str:
    prompt = prompt.rstrip()
    if "Answer:" not in prompt:
        prompt = f"Question: {prompt}\nAnswer:"
    return prompt


def answer_part(text: str) -> str:
    if "Answer:" in text:
        return text.rsplit("Answer:", 1)[-1]
    return text


def pad_sequences(values: Sequence[Sequence[int]], *, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    length = max(len(value) for value in values)
    output = torch.full((len(values), length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(values), length), dtype=torch.long)
    for i, value in enumerate(values):
        if value:
            output[i, : len(value)] = torch.tensor(value, dtype=torch.long)
            mask[i, : len(value)] = 1
    return output, mask


def pad_feature_sequences(values: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    length = max(value.shape[0] for value in values)
    dim = values[0].shape[-1]
    output = torch.zeros((len(values), length, dim), dtype=torch.float32)
    mask = torch.zeros((len(values), length), dtype=torch.long)
    for i, value in enumerate(values):
        if value.numel():
            output[i, : value.shape[0]] = value.float()
            mask[i, : value.shape[0]] = 1
    return output, mask


def prior_condition_inputs(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if "question_features" in batch:
        return batch["question_ids"], batch["question_feature_mask"], batch["question_features"]
    return batch["question_ids"], batch["question_mask"], None


def make_loader(rows: list[dict[str, Any]], *, config: ColaLatentPriorConfig, batch_size: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        ColaLatentDataset(rows),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=ColaLatentCollator(pad_token_id=config.pad_token_id),
        generator=generator,
    )


def cycling_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def read_jsonl(path: Path, max_samples: int) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if max_samples > 0 and len(rows) >= max_samples:
                    break
    return rows


def append_metrics_jsonl(path: Path, *, step: int, split: str, metrics: dict[str, float]) -> None:
    record = {
        "time": int(time.time()),
        "step": step,
        "split": split,
        "metrics": metrics,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def resolve_condition_dtype(dtype_name: str) -> torch.dtype | str:
    normalized = dtype_name.lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported condition dtype: {dtype_name}")


def load_condition_encoder(
    config: ColaLatentPriorConfig,
    *,
    device: torch.device,
) -> tuple[FrozenTextConditionEncoder | None, Any | None]:
    if config.condition_encoder == "learned":
        return None, None
    if config.condition_encoder != "qwen":
        raise ValueError(f"Unsupported condition encoder: {config.condition_encoder}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.condition_model_name,
        trust_remote_code=True,
        local_files_only=config.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoder = FrozenTextConditionEncoder(
        config.condition_model_name,
        device=device,
        local_files_only=config.local_files_only,
        dtype_name=config.condition_dtype,
    )
    return encoder, tokenizer


def resolve_eval_every(eval_every: int) -> int:
    return min(max(int(eval_every), 1), 100)


def save_checkpoint(
    path: Path,
    *,
    model: BlockCausalPrior,
    prior_config: StageCPriorConfig,
    train_config: ColaLatentPriorConfig,
    global_step: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(prior_config),
            "train_config": asdict(train_config),
            "global_step": global_step,
            "metrics": metrics,
        },
        path,
    )


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> ColaLatentPriorConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-path", default=ColaLatentPriorConfig.vae_path)
    parser.add_argument("--tokenizer-path", default=ColaLatentPriorConfig.tokenizer_path)
    parser.add_argument("--train-jsonl", default=ColaLatentPriorConfig.train_jsonl)
    parser.add_argument("--eval-jsonl", default=ColaLatentPriorConfig.eval_jsonl)
    parser.add_argument("--output-dir", default=ColaLatentPriorConfig.output_dir)
    parser.add_argument("--max-train-samples", type=int, default=ColaLatentPriorConfig.max_train_samples)
    parser.add_argument("--max-eval-samples", type=int, default=ColaLatentPriorConfig.max_eval_samples)
    parser.add_argument("--b-max", type=int, default=ColaLatentPriorConfig.b_max)
    parser.add_argument("--block-size", type=int, default=ColaLatentPriorConfig.block_size)
    parser.add_argument("--hidden-dim", type=int, default=ColaLatentPriorConfig.hidden_dim)
    parser.add_argument("--num-layers", type=int, default=ColaLatentPriorConfig.num_layers)
    parser.add_argument("--num-heads", type=int, default=ColaLatentPriorConfig.num_heads)
    parser.add_argument("--question-layers", type=int, default=ColaLatentPriorConfig.question_layers)
    parser.add_argument("--dropout", type=float, default=ColaLatentPriorConfig.dropout)
    parser.add_argument("--batch-size", type=int, default=ColaLatentPriorConfig.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=ColaLatentPriorConfig.eval_batch_size)
    parser.add_argument("--max-steps", type=int, default=ColaLatentPriorConfig.max_steps)
    parser.add_argument("--eval-every", type=int, default=ColaLatentPriorConfig.eval_every)
    parser.add_argument("--lr", type=float, default=ColaLatentPriorConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=ColaLatentPriorConfig.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=ColaLatentPriorConfig.grad_clip)
    parser.add_argument("--flow-time-max", type=float, default=ColaLatentPriorConfig.flow_time_max)
    parser.add_argument("--flow-time-loc", type=float, default=ColaLatentPriorConfig.flow_time_loc)
    parser.add_argument("--flow-time-scale", type=float, default=ColaLatentPriorConfig.flow_time_scale)
    parser.add_argument("--flow-denoise-steps", type=int, default=ColaLatentPriorConfig.flow_denoise_steps)
    parser.add_argument("--max-question-len", type=int, default=ColaLatentPriorConfig.max_question_len)
    parser.add_argument("--pad-token-id", type=int, default=ColaLatentPriorConfig.pad_token_id)
    parser.add_argument("--seed", type=int, default=ColaLatentPriorConfig.seed)
    parser.add_argument("--device", default=ColaLatentPriorConfig.device)
    parser.add_argument("--swanlab-mode", default=ColaLatentPriorConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=ColaLatentPriorConfig.experiment_name)
    parser.add_argument("--resume-checkpoint", default=ColaLatentPriorConfig.resume_checkpoint)
    parser.add_argument(
        "--condition-encoder",
        choices=["learned", "qwen"],
        default=ColaLatentPriorConfig.condition_encoder,
    )
    parser.add_argument("--condition-model-name", default=ColaLatentPriorConfig.condition_model_name)
    parser.add_argument("--condition-dtype", default=ColaLatentPriorConfig.condition_dtype)
    parser.add_argument("--max-condition-len", type=int, default=ColaLatentPriorConfig.max_condition_len)
    parser.add_argument(
        "--allow-condition-download",
        action="store_true",
        help="Allow downloading the frozen condition encoder from Hugging Face.",
    )
    args = parser.parse_args()
    return ColaLatentPriorConfig(
        vae_path=args.vae_path,
        tokenizer_path=args.tokenizer_path,
        train_jsonl=args.train_jsonl,
        eval_jsonl=args.eval_jsonl,
        output_dir=args.output_dir,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        b_max=args.b_max,
        block_size=args.block_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        question_layers=args.question_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        flow_time_max=args.flow_time_max,
        flow_time_loc=args.flow_time_loc,
        flow_time_scale=args.flow_time_scale,
        flow_denoise_steps=args.flow_denoise_steps,
        max_question_len=args.max_question_len,
        pad_token_id=args.pad_token_id,
        seed=args.seed,
        device=args.device,
        swanlab_mode=args.swanlab_mode,
        experiment_name=args.experiment_name,
        resume_checkpoint=args.resume_checkpoint,
        condition_encoder=args.condition_encoder,
        condition_model_name=args.condition_model_name,
        condition_dtype=args.condition_dtype,
        local_files_only=not args.allow_condition_download,
        max_condition_len=args.max_condition_len,
    )


def main() -> None:
    summary = train_cola_latent_prior(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
