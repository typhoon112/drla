"""Train the Stage B deterministic reasoning latent autoencoder."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from drla.data.answer_judge import judge
from drla.data.stage_b import (
    StageBCollator,
    StageBDataset,
    VocabularyMapper,
    build_local_vocab,
    load_stage_b_examples,
)
from drla.models.stage_b import StageBModelConfig, StageBReasoningAutoencoder
from drla.tracking import finish_experiment, init_experiment, log_metrics


DEFAULT_TOKENIZER = "Qwen/Qwen3-4B-Instruct-2507"


@dataclass(frozen=True)
class StageBTrainConfig:
    data_dir: str = "/data1/luyifei/drla/data/stage_a"
    output_dir: str = "/data1/luyifei/drla/outputs/stage_b_autoencoder"
    tokenizer_name: str = DEFAULT_TOKENIZER
    train_split: str = "train"
    eval_split: str = "test"
    local_files_only: bool = True
    compact_vocab: bool = False
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    b_max: int = 32
    block_size: int = 16
    latent_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    max_answer_len: int = 16
    batch_size: int = 4
    eval_batch_size: int = 8
    epochs: int = 3
    max_steps: int | None = None
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    recon_weight: float = 0.3
    answer_weight: float = 1.0
    noop_weight: float = 0.1
    kd_weight: float = 0.1
    verifier_weight: float = 0.1
    question_latent_weight: float = 0.0
    latent_noise_std: float = 0.0
    seed: int = 42
    device: str = "auto"
    swanlab_mode: str | None = None
    cache_latents: bool = True


def ensure_cloud_training(mode: str | None) -> None:
    requested = mode or os.getenv("SWANLAB_MODE") or "cloud"
    if requested != "cloud":
        raise ValueError(
            "Training must be logged to SwanLab cloud. "
            "Unset SWANLAB_MODE or use --swanlab-mode cloud."
        )


def train_stage_b(config: StageBTrainConfig) -> dict[str, Any]:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name,
        trust_remote_code=True,
        local_files_only=config.local_files_only,
    )
    train_examples = load_stage_b_examples(
        config.data_dir,
        config.train_split,
        tokenizer=tokenizer,
        b_max=config.b_max,
        block_size=config.block_size,
        max_samples=config.max_train_samples,
    )
    eval_examples = load_stage_b_examples(
        config.data_dir,
        config.eval_split,
        tokenizer=tokenizer,
        b_max=config.b_max,
        block_size=config.block_size,
        max_samples=config.max_eval_samples,
    )
    vocab_mapper = build_local_vocab(train_examples + eval_examples) if config.compact_vocab else None
    pad_id = vocab_mapper.pad_id if vocab_mapper else int(tokenizer.pad_token_id or 0)
    vocab_size = vocab_mapper.vocab_size if vocab_mapper else len(tokenizer)

    train_loader = make_loader(
        train_examples,
        vocab_mapper=vocab_mapper,
        pad_id=pad_id,
        config=config,
        batch_size=config.batch_size,
        shuffle=True,
    )
    eval_loader = make_loader(
        eval_examples,
        vocab_mapper=vocab_mapper,
        pad_id=pad_id,
        config=config,
        batch_size=config.eval_batch_size,
        shuffle=False,
    )

    model_config = StageBModelConfig(
        vocab_size=vocab_size,
        b_max=config.b_max,
        block_size=config.block_size,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        max_answer_len=config.max_answer_len,
        pad_id=pad_id,
        latent_noise_std=config.latent_noise_std,
    )
    model = StageBReasoningAutoencoder(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    ensure_cloud_training(config.swanlab_mode)

    init_experiment(
        stage="stage-b",
        experiment_name=f"stage-b-autoencoder-bs{config.block_size}-ld{config.latent_dim}",
        description="Deterministic reasoning latent autoencoder with answer-ready losses.",
        config={**asdict(config), "vocab_size": vocab_size},
        mode=config.swanlab_mode,
    )

    global_step = 0
    final_train_metrics: dict[str, float] = {}
    final_eval_metrics: dict[str, float] = {}
    best_eval_metrics: dict[str, float] = {}
    best_eval_step = 0
    best_metric_name = primary_eval_metric_name()
    best_score = float("-inf")
    best_checkpoint_path = output_dir / "best_checkpoint.pt"
    started_at = int(time.time())
    try:
        for _epoch in range(config.epochs):
            model.train()
            for batch in train_loader:
                global_step += 1
                batch = move_batch(batch, device)
                outputs = model(batch)
                loss = weighted_loss(outputs, config)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

                final_train_metrics = scalar_metrics(outputs, loss=loss)
                log_metrics(final_train_metrics, step=global_step, prefix="train")
                if config.max_steps is not None and global_step >= config.max_steps:
                    break

            final_eval_metrics = evaluate_model(
                model,
                eval_loader,
                tokenizer=tokenizer,
                vocab_mapper=vocab_mapper,
                config=config,
                device=device,
            )
            log_metrics(final_eval_metrics, step=global_step, prefix="valid")
            current_score = primary_eval_score(final_eval_metrics)
            if current_score > best_score:
                best_score = current_score
                best_eval_step = global_step
                best_eval_metrics = dict(final_eval_metrics)
                save_stage_b_checkpoint(
                    best_checkpoint_path,
                    model=model,
                    model_config=model_config,
                    train_config=config,
                    vocab_mapper=vocab_mapper,
                    global_step=global_step,
                )
            if config.max_steps is not None and global_step >= config.max_steps:
                break
    finally:
        finish_experiment()

    checkpoint_path = output_dir / "checkpoint.pt"
    save_stage_b_checkpoint(
        checkpoint_path,
        model=model,
        model_config=model_config,
        train_config=config,
        vocab_mapper=vocab_mapper,
        global_step=global_step,
    )

    cache_checkpoint_path = checkpoint_path
    if best_eval_metrics:
        best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        cache_checkpoint_path = best_checkpoint_path

    latent_dirs: dict[str, str] = {}
    if config.cache_latents:
        latent_dirs[config.train_split] = str(
            cache_latents(
                model,
                train_loader,
                output_dir / "latents" / config.train_split,
                device=device,
                checkpoint_path=cache_checkpoint_path,
            )
        )
        latent_dirs[config.eval_split] = str(
            cache_latents(
                model,
                eval_loader,
                output_dir / "latents" / config.eval_split,
                device=device,
                checkpoint_path=cache_checkpoint_path,
            )
        )

    gate_evaluable = (
        not config.compact_vocab
        and config.train_split != config.eval_split
        and config.max_train_samples is None
        and config.max_eval_samples is None
        and config.b_max == 32
    )
    gate_metrics = best_eval_metrics or final_eval_metrics
    observed_acc = gate_metrics.get("recon_judge_acc", 0.0)
    answer_head_acc = gate_metrics.get("answer_judge_acc", 0.0)
    q_only_acc = gate_metrics.get("q_only_answer_judge_acc", 0.0)
    summary = {
        "created_at": started_at,
        "finished_at": int(time.time()),
        "config": asdict(config),
        "vocab_size": vocab_size,
        "checkpoint_path": str(checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path) if best_eval_metrics else None,
        "latent_checkpoint_path": str(cache_checkpoint_path),
        "latent_dirs": latent_dirs,
        "global_step": global_step,
        "train": final_train_metrics,
        "valid": final_eval_metrics,
        "best_valid": {
            "step": best_eval_step,
            "metric": best_metric_name,
            "value": best_eval_metrics.get(best_metric_name) if best_eval_metrics else None,
            "metrics": best_eval_metrics,
        },
        "stage_b_gate": {
            "gold_latent_answer_acc_target": 0.90,
            "gold_latent_answer_acc_observed": observed_acc,
            "answer_head_acc_observed": answer_head_acc,
            "q_only_answer_acc_observed": q_only_acc,
            "gate_evaluable": gate_evaluable,
            "ready_for_stage_c": bool(
                gate_evaluable and observed_acc >= 0.90 and observed_acc > q_only_acc
            ),
            "notes": (
                "Formal Stage B gate requires full train/test data, non-compact vocab, "
                "B_max=32, and held-out validation. Smoke and overfit runs are diagnostics only."
            ),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def save_stage_b_checkpoint(
    path: Path,
    *,
    model: StageBReasoningAutoencoder,
    model_config: StageBModelConfig,
    train_config: StageBTrainConfig,
    vocab_mapper: VocabularyMapper | None,
    global_step: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "vocab_mapper": vocab_mapper.to_dict() if vocab_mapper else None,
            "global_step": global_step,
        },
        path,
    )


def primary_eval_metric_name() -> str:
    return "recon_judge_acc"


def primary_eval_score(metrics: dict[str, float]) -> float:
    metric_name = primary_eval_metric_name()
    if metric_name in metrics:
        return metrics[metric_name]
    return -metrics.get("loss", float("inf"))


def make_loader(
    examples: list[Any],
    *,
    vocab_mapper: VocabularyMapper | None,
    pad_id: int,
    config: StageBTrainConfig,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = StageBDataset(
        examples, vocab_mapper=vocab_mapper, max_answer_len=config.max_answer_len
    )
    collator = StageBCollator(
        pad_id=pad_id,
        b_max=config.b_max,
        block_size=config.block_size,
        max_answer_len=config.max_answer_len,
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        generator=generator,
    )


def weighted_loss(outputs: dict[str, torch.Tensor], config: StageBTrainConfig) -> torch.Tensor:
    return (
        config.answer_weight * outputs["l_answer"]
        + config.recon_weight * outputs["l_recon"]
        + config.noop_weight * outputs["l_noop"]
        + config.kd_weight * outputs["l_kd"]
        + config.verifier_weight * outputs["l_verifier"]
        + config.question_latent_weight * outputs["l_question_latent"]
    )


@torch.no_grad()
def evaluate_model(
    model: StageBReasoningAutoencoder,
    loader: DataLoader,
    *,
    tokenizer: Any,
    vocab_mapper: VocabularyMapper | None,
    config: StageBTrainConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    judge_correct = 0
    recon_judge_correct = 0
    q_only_judge_correct = 0
    seq_correct = 0
    q_only_seq_correct = 0
    token_correct = 0
    token_total = 0

    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(batch)
        loss = weighted_loss(outputs, config)
        metrics = scalar_metrics(outputs, loss=loss)
        batch_size = batch["answer_input_ids"].shape[0]
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        count += batch_size

        answer_pred = outputs["answer_logits"].argmax(dim=-1)
        recon_pred = outputs["token_logits"].argmax(dim=-1)
        q_only_pred = q_only_answer_logits(model, batch).argmax(dim=-1)
        answer_mask = batch["answer_mask"].bool()
        labels = batch["answer_input_ids"]
        token_correct += ((answer_pred == labels) & answer_mask).sum().item()
        token_total += answer_mask.sum().item()
        seq_correct += sequence_matches(answer_pred, labels, answer_mask)
        q_only_seq_correct += sequence_matches(q_only_pred, labels, answer_mask)

        pred_texts = decode_predictions(answer_pred, tokenizer, vocab_mapper, pad_id=model.config.pad_id)
        recon_texts = decode_recon_predictions(
            recon_pred, batch["target_mask"], tokenizer, vocab_mapper
        )
        q_only_texts = decode_predictions(q_only_pred, tokenizer, vocab_mapper, pad_id=model.config.pad_id)
        for pred_text, recon_text, q_only_text, gold in zip(
            pred_texts, recon_texts, q_only_texts, batch["answer_norms"]
        ):
            judge_correct += int(judge(pred_text, gold)["correct"])
            recon_judge_correct += int(judge(recon_text, gold)["correct"])
            q_only_judge_correct += int(judge(q_only_text, gold)["correct"])

    averaged = {key: value / max(count, 1) for key, value in totals.items()}
    averaged["answer_token_acc"] = token_correct / max(token_total, 1)
    averaged["answer_seq_acc"] = seq_correct / max(count, 1)
    averaged["q_only_answer_seq_acc"] = q_only_seq_correct / max(count, 1)
    averaged["answer_judge_acc"] = judge_correct / max(count, 1)
    averaged["recon_judge_acc"] = recon_judge_correct / max(count, 1)
    averaged["q_only_answer_judge_acc"] = q_only_judge_correct / max(count, 1)
    averaged["answer_head_minus_q_only_judge_acc"] = (
        averaged["answer_judge_acc"] - averaged["q_only_answer_judge_acc"]
    )
    averaged["q_plus_z_minus_q_only_judge_acc"] = (
        averaged["recon_judge_acc"] - averaged["q_only_answer_judge_acc"]
    )
    return averaged


@torch.no_grad()
def q_only_answer_logits(
    model: StageBReasoningAutoencoder, batch: dict[str, Any]
) -> torch.Tensor:
    q_pool = model.encode_question(batch["question_ids"], batch["question_mask"])
    z_blocks = torch.zeros(
        batch["question_ids"].shape[0],
        model.config.b_max,
        model.config.block_size,
        model.config.latent_dim,
        device=batch["question_ids"].device,
    )
    return model.decode_answer(z_blocks, q_pool, batch["block_mask"])


@torch.no_grad()
def cache_latents(
    model: StageBReasoningAutoencoder,
    loader: DataLoader,
    output_dir: Path,
    *,
    device: torch.device,
    checkpoint_path: Path,
) -> Path:
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(batch)
        z_blocks = outputs["z_blocks"].detach().cpu()
        block_mask = batch["block_mask"].detach().cpu()
        noop_mask = batch["noop_mask"].detach().cpu()
        for i, sample_id in enumerate(batch["ids"]):
            torch.save(
                {
                    "id": sample_id,
                    "z_blocks": z_blocks[i],
                    "question_ids": torch.tensor(batch["original_question_ids"][i]),
                    "target_ids": torch.tensor(batch["original_target_ids"][i]),
                    "answer_norm": batch["answer_norms"][i],
                    "B_star": int(batch["b_star"][i].detach().cpu().item()),
                    "block_mask": block_mask[i],
                    "noop_mask": noop_mask[i],
                    "model_checkpoint": str(checkpoint_path),
                },
                output_dir / f"{sample_id}.pt",
            )
    return output_dir


def scalar_metrics(outputs: dict[str, torch.Tensor], *, loss: torch.Tensor) -> dict[str, float]:
    keys = [
        "l_recon",
        "l_answer",
        "l_noop",
        "l_kd",
        "l_verifier",
        "l_question_latent",
        "latent_norm_mean",
        "latent_norm_std",
    ]
    metrics = {"loss": float(loss.detach().cpu().item())}
    for key in keys:
        metrics[key] = float(outputs[key].detach().cpu().item())
    return metrics


def sequence_matches(pred_ids: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> int:
    matches = ((pred_ids == labels) | ~mask).all(dim=1)
    return int(matches.sum().item())


def decode_predictions(
    pred_ids: torch.Tensor,
    tokenizer: Any,
    vocab_mapper: VocabularyMapper | None,
    *,
    pad_id: int,
) -> list[str]:
    texts: list[str] = []
    for row in pred_ids.detach().cpu().tolist():
        clipped: list[int] = []
        for token_id in row:
            token_id = int(token_id)
            if token_id == pad_id:
                break
            clipped.append(token_id)
        if vocab_mapper is not None:
            original_ids = vocab_mapper.decode(clipped)
        else:
            original_ids = [token_id for token_id in clipped if token_id >= 0]
        texts.append(tokenizer.decode(original_ids, skip_special_tokens=True))
    return texts


def decode_recon_predictions(
    pred_ids: torch.Tensor,
    target_mask: torch.Tensor,
    tokenizer: Any,
    vocab_mapper: VocabularyMapper | None,
) -> list[str]:
    texts: list[str] = []
    pred_rows = pred_ids.detach().cpu().tolist()
    lengths = target_mask.detach().cpu().sum(dim=1).tolist()
    for row, length in zip(pred_rows, lengths):
        clipped = [int(token_id) for token_id in row[: int(length)]]
        if vocab_mapper is not None:
            original_ids = vocab_mapper.decode(clipped)
        else:
            original_ids = [token_id for token_id in clipped if token_id >= 0]
        texts.append(tokenizer.decode(original_ids, skip_special_tokens=True))
    return texts


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


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


def parse_args() -> StageBTrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=StageBTrainConfig.data_dir)
    parser.add_argument("--output-dir", default=StageBTrainConfig.output_dir)
    parser.add_argument("--tokenizer-name", default=StageBTrainConfig.tokenizer_name)
    parser.add_argument("--train-split", default=StageBTrainConfig.train_split)
    parser.add_argument("--eval-split", default=StageBTrainConfig.eval_split)
    parser.add_argument("--allow-tokenizer-download", action="store_true")
    parser.add_argument("--compact-vocab", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--b-max", type=int, default=StageBTrainConfig.b_max)
    parser.add_argument("--block-size", type=int, default=StageBTrainConfig.block_size)
    parser.add_argument("--latent-dim", type=int, default=StageBTrainConfig.latent_dim)
    parser.add_argument("--hidden-dim", type=int, default=StageBTrainConfig.hidden_dim)
    parser.add_argument("--num-layers", type=int, default=StageBTrainConfig.num_layers)
    parser.add_argument("--num-heads", type=int, default=StageBTrainConfig.num_heads)
    parser.add_argument("--dropout", type=float, default=StageBTrainConfig.dropout)
    parser.add_argument("--max-answer-len", type=int, default=StageBTrainConfig.max_answer_len)
    parser.add_argument("--batch-size", type=int, default=StageBTrainConfig.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=StageBTrainConfig.eval_batch_size)
    parser.add_argument("--epochs", type=int, default=StageBTrainConfig.epochs)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--lr", type=float, default=StageBTrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=StageBTrainConfig.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=StageBTrainConfig.grad_clip)
    parser.add_argument("--recon-weight", type=float, default=StageBTrainConfig.recon_weight)
    parser.add_argument("--answer-weight", type=float, default=StageBTrainConfig.answer_weight)
    parser.add_argument("--noop-weight", type=float, default=StageBTrainConfig.noop_weight)
    parser.add_argument("--kd-weight", type=float, default=StageBTrainConfig.kd_weight)
    parser.add_argument("--verifier-weight", type=float, default=StageBTrainConfig.verifier_weight)
    parser.add_argument(
        "--question-latent-weight",
        type=float,
        default=StageBTrainConfig.question_latent_weight,
    )
    parser.add_argument("--latent-noise-std", type=float, default=StageBTrainConfig.latent_noise_std)
    parser.add_argument("--seed", type=int, default=StageBTrainConfig.seed)
    parser.add_argument("--device", default=StageBTrainConfig.device)
    parser.add_argument("--swanlab-mode")
    parser.add_argument("--no-cache-latents", action="store_true")
    args = parser.parse_args()
    return StageBTrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer_name,
        train_split=args.train_split,
        eval_split=args.eval_split,
        local_files_only=not args.allow_tokenizer_download,
        compact_vocab=args.compact_vocab,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        b_max=args.b_max,
        block_size=args.block_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_answer_len=args.max_answer_len,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        max_steps=args.max_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        recon_weight=args.recon_weight,
        answer_weight=args.answer_weight,
        noop_weight=args.noop_weight,
        kd_weight=args.kd_weight,
        verifier_weight=args.verifier_weight,
        question_latent_weight=args.question_latent_weight,
        latent_noise_std=args.latent_noise_std,
        seed=args.seed,
        device=args.device,
        swanlab_mode=args.swanlab_mode,
        cache_latents=not args.no_cache_latents,
    )


def main() -> None:
    summary = train_stage_b(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
