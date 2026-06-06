"""Train a CoLA DiT LoRA interface adapter on Phase A calibration SFT pairs.

This is a deep-learning training script: it requires CUDA/GPU, SwanLab cloud
logging, local ``metrics.jsonl``, and best/last checkpoints.  It trains only
LoRA parameters on the official CoLA DiT prior with the VAE frozen.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.scripts.train_cola_readiness_model import device_metadata, require_cuda_training, resolve_device
from drla.tracking import finish_experiment, init_experiment, log_metrics


DEFAULT_SFT_DIR = (
    "/data1/luyifei/drla/outputs/p2_phase_a_cola_interface_sft/"
    "musique_calibration_qwen_teacher_v1_20260605"
)
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p2_phase_a_cola_dit_lora/"
    "musique_interface_lora_v1_20260605"
)
DEFAULT_COLA_DIT_PATH = "/data1/luyifei/drla/models/Cola-DLM/cola_dlm/cola_dit"
DEFAULT_COLA_VAE_PATH = "/data1/luyifei/drla/models/Cola-DLM/cola_dlm/cola_vae"
DEFAULT_COLA_TOKENIZER_PATH = "/data1/luyifei/drla/models/Cola-DLM/tokenizer.json"
DEFAULT_COLA_CODE_PATH = "/data1/luyifei/Cola-DLM/code"


@dataclass
class TrainConfig:
    sft_dir: str = DEFAULT_SFT_DIR
    output_dir: str = DEFAULT_OUTPUT_DIR
    cola_dit_path: str = DEFAULT_COLA_DIT_PATH
    cola_vae_path: str = DEFAULT_COLA_VAE_PATH
    cola_tokenizer_path: str = DEFAULT_COLA_TOKENIZER_PATH
    cola_code_path: str = DEFAULT_COLA_CODE_PATH
    device: str = "cuda"
    roles: str = "solver_full_info,solver_textmas_matched,evidence_agent_teacher"
    seed: int = 20260605
    batch_size: int = 1
    epochs: int = 1
    max_train_steps: int = 0
    valid_interval: int = 10
    max_valid_batches: int = 0
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = "proj_qkv,proj_out,proj_in"
    init_lora_path: str = ""
    flow_T: float = 1000.0
    pad_token_id: int = 100277
    target_eos_token_id: int = 100257
    append_target_eos: int = 1
    max_total_tokens: int = 0
    max_total_blocks: int = 0
    swanlab_mode: str = "cloud"
    experiment_name: str = "p2a-cola-dit-lora-musique-interface-v1"
    save_interval_checkpoints: bool = False


class BlockExampleDataset(Dataset):
    def __init__(
        self,
        *,
        pairs: list[dict[str, Any]],
        tokenizer: Any,
        split: str,
        roles: set[str],
        block_size: int,
        pad_token_id: int,
        target_eos_token_id: int | None,
        max_total_tokens: int = 0,
        max_total_blocks: int = 0,
    ) -> None:
        self.items: list[dict[str, Any]] = []
        self.pairs: list[dict[str, Any]] = []
        self.skipped: Counter[str] = Counter()
        self.block_size = block_size
        for pair in pairs:
            if str(pair.get("split", "")) != split or str(pair.get("role", "")) not in roles:
                continue
            prompt_ids = tokenizer.encode(str(pair["prompt_text"])).ids
            target_ids = tokenizer.encode(" " + str(pair["target_text"]).strip()).ids
            if target_eos_token_id is not None:
                target_ids = target_ids + [target_eos_token_id]
            if not prompt_ids or not target_ids:
                self.skipped["empty_prompt_or_target"] += 1
                continue
            ids = prompt_ids + target_ids
            total_tokens = len(ids)
            total_blocks = (total_tokens + block_size - 1) // block_size
            if max_total_tokens and total_tokens > max_total_tokens:
                self.skipped["over_max_total_tokens"] += 1
                continue
            if max_total_blocks and total_blocks > max_total_blocks:
                self.skipped["over_max_total_blocks"] += 1
                continue
            labels = [1] * len(prompt_ids) + [2] * len(target_ids)
            pad_len = (block_size - len(ids) % block_size) % block_size
            ids = ids + [pad_token_id] * pad_len
            labels = labels + [3] * pad_len
            pair_index = len(self.pairs)
            stored_pair = {
                **pair,
                "input_ids": ids,
                "token_labels": labels,
                "prompt_token_count": len(prompt_ids),
                "target_token_count": len(target_ids),
            }
            self.pairs.append(stored_pair)
            target_blocks = sorted({index // block_size for index, label in enumerate(labels) if label == 2})
            for block_index in target_blocks:
                self.items.append({"pair_index": pair_index, "block_start": block_index * block_size})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, int]:
        return self.items[index]

    def pair_for(self, pair_index: int) -> dict[str, Any]:
        return self.pairs[pair_index]


def main() -> None:
    summary = train(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainConfig()
    for field, value in asdict(defaults).items():
        arg = "--" + field.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(arg, action="store_true", default=value)
        elif isinstance(value, int):
            parser.add_argument(arg, type=int, default=value)
        elif isinstance(value, float):
            parser.add_argument(arg, type=float, default=value)
        else:
            parser.add_argument(arg, default=value)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def train(config: TrainConfig) -> dict[str, Any]:
    validate_config(config)
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    if config.cola_code_path and config.cola_code_path not in sys.path:
        sys.path.insert(0, config.cola_code_path)
    from tokenizers import Tokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    from cola_dlm import ColaDiTModel, ColaTextVAEModel

    tokenizer = Tokenizer.from_file(config.cola_tokenizer_path)
    roles = {role.strip() for role in config.roles.split(",") if role.strip()}
    pairs = read_jsonl(Path(config.sft_dir) / "sft_pairs.jsonl")

    device = resolve_device(config.device)
    require_cuda_training(device, "train_p2_phase_a_cola_dit_lora.py")
    dit = ColaDiTModel.from_pretrained(config.cola_dit_path).to(device)
    vae = ColaTextVAEModel.from_pretrained(config.cola_vae_path).to(device).eval()
    for param in vae.parameters():
        param.requires_grad_(False)

    if config.init_lora_path:
        dit = PeftModel.from_pretrained(dit, config.init_lora_path, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=[name.strip() for name in config.lora_target_modules.split(",") if name.strip()],
            bias="none",
        )
        dit = get_peft_model(dit, lora_config)
    dit.train()
    trainable_params = sum(param.numel() for param in dit.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in dit.parameters())

    block_size = int(dit.base_model.model.block_size if hasattr(dit, "base_model") else dit.block_size)
    train_ds = BlockExampleDataset(
        pairs=pairs,
        tokenizer=tokenizer,
        split="train",
        roles=roles,
        block_size=block_size,
        pad_token_id=config.pad_token_id,
        target_eos_token_id=config.target_eos_token_id if config.append_target_eos else None,
        max_total_tokens=config.max_total_tokens,
        max_total_blocks=config.max_total_blocks,
    )
    valid_ds = BlockExampleDataset(
        pairs=pairs,
        tokenizer=tokenizer,
        split="valid",
        roles=roles,
        block_size=block_size,
        pad_token_id=config.pad_token_id,
        target_eos_token_id=config.target_eos_token_id if config.append_target_eos else None,
        max_total_tokens=config.max_total_tokens,
        max_total_blocks=config.max_total_blocks,
    )
    if not train_ds or not valid_ds:
        raise ValueError(f"empty train/valid block dataset: train={len(train_ds)} valid={len(valid_ds)}")
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: batch,
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda batch: batch,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(dit.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    run = init_experiment(
        stage="p2a-cola-dit-lora-interface",
        config={
            **asdict(config),
            **device_metadata(device),
            "train_block_examples": len(train_ds),
            "valid_block_examples": len(valid_ds),
            "train_skipped": dict(train_ds.skipped),
            "valid_skipped": dict(valid_ds.skipped),
            "trainable_params": trainable_params,
            "total_params": total_params,
        },
        experiment_name=config.experiment_name,
        tags=["cola", "p2a", "musique", "dit-lora", "flow-matching"],
        mode=config.swanlab_mode,
    )

    best_valid_loss = float("inf")
    best_step = 0
    global_step = 0
    interval_checkpoints: list[dict[str, Any]] = []
    metrics_f = metrics_path.open("w", encoding="utf-8")
    try:
        stop = False
        for _epoch in range(config.epochs):
            for batch in train_loader:
                global_step += 1
                dit.train()
                optimizer.zero_grad(set_to_none=True)
                loss = compute_flow_loss(batch, train_ds, vae, dit, device, config)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dit.parameters(), config.grad_clip_norm)
                optimizer.step()
                train_metrics = {"loss": float(loss.detach().item())}
                write_metrics(metrics_f, "train", global_step, train_metrics)
                log_metrics(train_metrics, step=global_step, prefix="train")
                if global_step % config.valid_interval == 0:
                    valid_metrics = evaluate(valid_loader, valid_ds, vae, dit, device, config)
                    write_metrics(metrics_f, "valid", global_step, valid_metrics)
                    log_metrics(valid_metrics, step=global_step, prefix="valid")
                    if config.save_interval_checkpoints:
                        interval_checkpoint = checkpoint_dir / f"valid_step_{global_step}.pt"
                        interval_adapter = checkpoint_dir / f"valid_step_{global_step}_adapter"
                        save_checkpoint(
                            interval_checkpoint,
                            interval_adapter,
                            dit,
                            optimizer,
                            config,
                            global_step,
                            valid_metrics["loss"],
                            train_ds,
                            valid_ds,
                        )
                        interval_checkpoints.append(
                            {
                                "step": global_step,
                                "valid_loss": valid_metrics["loss"],
                                "checkpoint": str(interval_checkpoint),
                                "adapter": str(interval_adapter),
                            }
                        )
                    if valid_metrics["loss"] < best_valid_loss:
                        best_valid_loss = valid_metrics["loss"]
                        best_step = global_step
                        save_checkpoint(
                            checkpoint_dir / "best_checkpoint.pt",
                            checkpoint_dir / "best_adapter",
                            dit,
                            optimizer,
                            config,
                            global_step,
                            best_valid_loss,
                            train_ds,
                            valid_ds,
                        )
                if config.max_train_steps and global_step >= config.max_train_steps:
                    stop = True
                    break
            if stop:
                break
        final_valid_metrics = evaluate(valid_loader, valid_ds, vae, dit, device, config)
        write_metrics(metrics_f, "valid", global_step, final_valid_metrics)
        log_metrics(final_valid_metrics, step=global_step, prefix="valid")
        if final_valid_metrics["loss"] < best_valid_loss:
            best_valid_loss = final_valid_metrics["loss"]
            best_step = global_step
            save_checkpoint(
                checkpoint_dir / "best_checkpoint.pt",
                checkpoint_dir / "best_adapter",
                dit,
                optimizer,
                config,
                global_step,
                best_valid_loss,
                train_ds,
                valid_ds,
            )
        save_checkpoint(
            checkpoint_dir / "last_checkpoint.pt",
            checkpoint_dir / "last_adapter",
            dit,
            optimizer,
            config,
            global_step,
            final_valid_metrics["loss"],
            train_ds,
            valid_ds,
        )
    finally:
        metrics_f.close()
        finish_experiment()

    summary = {
        "created_at": int(time.time()),
        "status": "pass",
        "config": asdict(config),
        "swanlab_run_id": getattr(run, "id", None),
        "train_block_examples": len(train_ds),
        "valid_block_examples": len(valid_ds),
        "train_pairs": len(train_ds.pairs),
        "valid_pairs": len(valid_ds.pairs),
        "train_skipped": dict(train_ds.skipped),
        "valid_skipped": dict(valid_ds.skipped),
        "block_size": block_size,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "best_step": best_step,
        "best_valid_loss": best_valid_loss,
        "final_valid_metrics": final_valid_metrics,
        "interval_checkpoints": interval_checkpoints,
        "artifacts": {
            "metrics_jsonl": str(metrics_path),
            "best_checkpoint": str(checkpoint_dir / "best_checkpoint.pt"),
            "last_checkpoint": str(checkpoint_dir / "last_checkpoint.pt"),
            "best_adapter": str(checkpoint_dir / "best_adapter"),
            "last_adapter": str(checkpoint_dir / "last_adapter"),
            "summary_json": str(output_dir / "summary.json"),
        },
        "execution_boundary": [
            "deep-learning training",
            "CUDA/GPU required",
            "SwanLab cloud required",
            "calibration-only Phase A SFT pairs",
            "frozen official CoLA VAE",
            "LoRA-only official CoLA DiT adaptation",
            "no held-out data",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def compute_flow_loss(
    batch: list[dict[str, int]],
    dataset: BlockExampleDataset,
    vae: Any,
    dit: Any,
    device: torch.device,
    config: TrainConfig,
) -> torch.Tensor:
    model_input, timestep, txt_shape, loss_mask, target_velocity = build_flow_batch(
        batch, dataset, vae, device, config
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = dit(
            txt=model_input.to(torch.bfloat16),
            txt_shape=txt_shape,
            txt_q_shape=txt_shape,
            timestep=timestep.to(torch.bfloat16),
            update_kv=False,
            use_kv_cache=False,
        ).txt_sample
    return F.mse_loss(pred.float()[loss_mask], target_velocity[loss_mask])


@torch.no_grad()
def evaluate(
    loader: DataLoader,
    dataset: BlockExampleDataset,
    vae: Any,
    dit: Any,
    device: torch.device,
    config: TrainConfig,
) -> dict[str, float]:
    dit.eval()
    losses = []
    for batch_index, batch in enumerate(loader):
        if config.max_valid_batches and batch_index >= config.max_valid_batches:
            break
        loss = compute_flow_loss(batch, dataset, vae, dit, device, config)
        losses.append(float(loss.detach().item()))
    dit.train()
    return {"loss": sum(losses) / len(losses) if losses else float("inf"), "num_batches": float(len(losses))}


def build_flow_batch(
    batch: list[dict[str, int]],
    dataset: BlockExampleDataset,
    vae: Any,
    device: torch.device,
    config: TrainConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pairs = [dataset.pair_for(int(item["pair_index"])) for item in batch]
    block_starts = [int(item["block_start"]) for item in batch]
    input_ids_list = [
        torch.tensor(pair["input_ids"], dtype=torch.long, device=device)
        for pair in pairs
    ]
    labels_list = [
        torch.tensor(pair["token_labels"], dtype=torch.long, device=device)
        for pair in pairs
    ]
    scale = vae.scaling_factor
    shift = vae.shifting_factor
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        enc = vae.encode(input_ids_list)
        latents_list = [((lat - shift) * scale).float() for lat in enc.latents_list]

    model_inputs = []
    timesteps = []
    target_velocities = []
    loss_masks = []
    seq_lens = []
    block_size = int(getattr(dataset, "block_size", 0) or len(labels_list[0]))
    for latents, labels, block_start in zip(latents_list, labels_list, block_starts):
        seq_len = block_start + block_size
        z0_seq = latents[:seq_len].clone()
        label_seq = labels[:seq_len]
        current = slice(block_start, seq_len)
        current_labels = label_seq[current]
        target_mask_block = current_labels == 2
        if not bool(target_mask_block.any()):
            raise ValueError("selected block has no target-token latent positions")
        t_scalar = torch.rand((), device=device, dtype=torch.float32) * float(config.flow_T)
        z1_block = torch.randn(block_size, z0_seq.shape[-1], device=device, dtype=torch.float32)
        z0_block = z0_seq[current].clone()
        alpha = t_scalar / float(config.flow_T)
        zt_block = z0_block.clone()
        zt_block[target_mask_block] = (1.0 - alpha) * z0_block[target_mask_block] + alpha * z1_block[target_mask_block]
        z0_seq[current] = zt_block
        timestep = torch.zeros(seq_len, device=device, dtype=torch.float32)
        timestep_block = timestep[current]
        timestep_block[target_mask_block] = t_scalar
        timestep[current] = timestep_block
        target_velocity = torch.zeros_like(z0_seq)
        target_velocity_block = target_velocity[current]
        target_velocity_block[target_mask_block] = z1_block[target_mask_block] - z0_block[target_mask_block]
        target_velocity[current] = target_velocity_block
        loss_mask = torch.zeros(seq_len, device=device, dtype=torch.bool)
        loss_mask_block = loss_mask[current]
        loss_mask_block[target_mask_block] = True
        loss_mask[current] = loss_mask_block
        model_inputs.append(z0_seq)
        timesteps.append(timestep)
        target_velocities.append(target_velocity)
        loss_masks.append(loss_mask)
        seq_lens.append(seq_len)

    return (
        torch.cat(model_inputs, dim=0),
        torch.cat(timesteps, dim=0),
        torch.tensor([[length] for length in seq_lens], dtype=torch.long, device=device),
        torch.cat(loss_masks, dim=0),
        torch.cat(target_velocities, dim=0),
    )


def save_checkpoint(
    checkpoint_path: Path,
    adapter_dir: Path,
    model: Any,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    step: int,
    metric: float,
    train_ds: BlockExampleDataset,
    valid_ds: BlockExampleDataset,
) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    trainable_state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    torch.save(
        {
            "config": asdict(config),
            "global_step": step,
            "selection_metric": metric,
            "optimizer": optimizer.state_dict(),
            "trainable_state_dict": trainable_state,
            "adapter_dir": str(adapter_dir),
            "train_block_examples": len(train_ds),
            "valid_block_examples": len(valid_ds),
        },
        checkpoint_path,
    )


def validate_config(config: TrainConfig) -> None:
    if config.valid_interval > 10:
        raise ValueError("valid_interval must be <= 10 steps")
    if config.swanlab_mode != "cloud":
        raise ValueError("training must use SwanLab cloud; pass --swanlab-mode cloud")
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    if config.flow_T <= 0:
        raise ValueError("flow_T must be positive")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{line_no}")
        rows.append(value)
    return rows


def write_metrics(handle: Any, split: str, step: int, metrics: dict[str, float]) -> None:
    handle.write(json.dumps({"split": split, "step": step, **metrics}, sort_keys=True) + "\n")
    handle.flush()


if __name__ == "__main__":
    main()
