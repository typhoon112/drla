"""Train D7.7 Dream layer receiver with TextMAS teacher alignment.

This trainer keeps the D7 V7 online boundary: the student solver receives no
decoded Agent A/B text at runtime and is conditioned only on D6 latent packets.
During training only, the same-row TextMAS message channel is used as a frozen
teacher distribution. The objective combines matched answer-token CE, a
teacher-logit alignment anchor, and zero/shuffled corruption separation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.scripts.p3_run_dream_text_encoded_packet_eval import (  # noqa: E402
    DEFAULT_TEXTMAS_GENERATIONS,
    load_textmas_messages,
)
from drla.scripts.p3_train_dream_layer_conditioned_receiver import (  # noqa: E402
    DreamLayerConditionedReceiver,
    LayerReceiverConfig,
    collate_one,
    load_row_packets,
    load_training_rows,
    move_batch,
    parse_selected_layers,
    receiver_loss,
    save_checkpoint,
    set_seed,
    split_rows,
    write_metrics,
)
from drla.scripts.p3_train_dream_layer_receiver_corruption_aware import make_layer_config, parse_corruption_types  # noqa: E402
from drla.scripts.p3_train_dream_soft_prefix_adapter import resolve_mask_token_id  # noqa: E402
from drla.scripts.run_p2_phase_c_text_agents import make_solver_messages  # noqa: E402
from drla.scripts.train_cola_readiness_model import device_metadata, require_cuda_training, resolve_device  # noqa: E402
from drla.tracking import finish_experiment, init_experiment, log_metrics  # noqa: E402


DEFAULT_INIT_CHECKPOINT = (
    "/data1/luyifei/drla/outputs/p3_dream_layer_receivers/"
    "dream_layer_receiver_v7_v4init_zeroshuf_textmas_matched200_seed20260607_20260607/"
    "best_checkpoint.pt"
)
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_layer_receivers/"
    "dream_layer_receiver_d77_text_teacher_v7init_textmas_matched200_seed20260617"
)


class TextTeacherLayerReceiverDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        textmas_messages: dict[str, list[dict[str, str]]],
        tokenizer: Any,
        config: LayerReceiverConfig,
    ) -> None:
        self.rows = rows
        self.textmas_messages = textmas_messages
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        row_id = str(row["row_id"])
        agent_messages = self.textmas_messages.get(row_id)
        if agent_messages is None:
            raise ValueError(f"missing TextMAS teacher messages for row_id={row_id}")
        prompt_ids = self.tokenizer.apply_chat_template(
            make_solver_messages(row["online_input_fields"], upstream_messages=[]),
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).input_ids[0]
        teacher_prompt_ids = self.tokenizer.apply_chat_template(
            make_solver_messages(row["online_input_fields"], upstream_messages=agent_messages),
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
            raise ValueError(f"empty target ids for row_id={row_id}")
        return {
            "packets": load_row_packets(row, self.config),
            "prompt_ids": prompt_ids.to(torch.long),
            "teacher_prompt_ids": teacher_prompt_ids.to(torch.long),
            "target_ids": target_ids.to(torch.long),
        }


def main() -> None:
    summary = train(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", default=LayerReceiverConfig.manifest_json)
    parser.add_argument("--online-inputs-jsonl", default=LayerReceiverConfig.online_inputs_jsonl)
    parser.add_argument("--packet-dir", default=LayerReceiverConfig.packet_dir)
    parser.add_argument("--model-path", default=LayerReceiverConfig.model_path)
    parser.add_argument("--textmas-generations-jsonl", default=DEFAULT_TEXTMAS_GENERATIONS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=LayerReceiverConfig.device)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=LayerReceiverConfig.dtype)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--train-ratio", type=float, default=LayerReceiverConfig.train_ratio)
    parser.add_argument("--valid-ratio", type=float, default=LayerReceiverConfig.valid_ratio)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--valid-interval", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--input-tokens-per-agent", type=int, default=LayerReceiverConfig.input_tokens_per_agent)
    parser.add_argument("--max-target-tokens", type=int, default=LayerReceiverConfig.max_target_tokens)
    parser.add_argument("--hidden-size", type=int, default=LayerReceiverConfig.hidden_size)
    parser.add_argument("--d-model", type=int, default=LayerReceiverConfig.d_model)
    parser.add_argument("--num-memory-layers", type=int, default=LayerReceiverConfig.num_memory_layers)
    parser.add_argument("--num-heads", type=int, default=LayerReceiverConfig.num_heads)
    parser.add_argument("--dropout", type=float, default=LayerReceiverConfig.dropout)
    parser.add_argument("--selected-layers", default=LayerReceiverConfig.selected_layers)
    parser.add_argument("--corruption-margin", type=float, default=0.2)
    parser.add_argument("--corruption-loss-weight", type=float, default=0.1)
    parser.add_argument("--corruption-types", default="zero,shuffled_row")
    parser.add_argument("--teacher-kl-weight", type=float, default=0.3)
    parser.add_argument("--teacher-cosine-weight", type=float, default=0.01)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--init-checkpoint", default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--swanlab-mode", default=LayerReceiverConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default="p3-dream-layer-receiver-d77-text-teacher-v7init")
    return parser.parse_args()


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.swanlab_mode != "cloud":
        raise ValueError("D7.7 text-teacher receiver training must use SwanLab cloud")
    if args.valid_interval > 10:
        raise ValueError("valid_interval must be <= 10 for current P3 training")
    if args.batch_size != 1:
        raise ValueError("This script currently requires --batch-size 1 because prompts have variable lengths")
    if args.teacher_temperature <= 0:
        raise ValueError("teacher_temperature must be > 0")
    set_seed(args.seed)
    rng = random.Random(args.seed + 23)
    device = resolve_device(args.device)
    require_cuda_training(device, "p3_train_dream_layer_receiver_text_teacher.py")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = make_layer_config(args)
    model_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[base_config.dtype]
    tokenizer = AutoTokenizer.from_pretrained(base_config.model_path, trust_remote_code=True)
    dream = AutoModel.from_pretrained(base_config.model_path, torch_dtype=model_dtype, trust_remote_code=True).to(device).eval()
    for param in dream.parameters():
        param.requires_grad_(False)

    rows, metadata = load_training_rows(base_config)
    textmas_messages = load_textmas_messages(Path(args.textmas_generations_jsonl))
    missing_messages = [row["row_id"] for row in rows if str(row["row_id"]) not in textmas_messages]
    if missing_messages:
        raise ValueError(f"missing TextMAS teacher messages: {missing_messages[:5]}")
    metadata["selected_layers"] = parse_selected_layers(base_config.selected_layers)
    corruption_types = parse_corruption_types(args.corruption_types)
    metadata.update(
        {
            "objective": "matched CE + TextMAS teacher KL/cosine anchor + corruption separation",
            "textmas_generations_jsonl": args.textmas_generations_jsonl,
            "teacher_kl_weight": args.teacher_kl_weight,
            "teacher_cosine_weight": args.teacher_cosine_weight,
            "teacher_temperature": args.teacher_temperature,
            "corruption_margin": args.corruption_margin,
            "corruption_loss_weight": args.corruption_loss_weight,
            "corruption_types": corruption_types,
            "init_checkpoint": args.init_checkpoint,
            "online_student_prompt": "no-message solver prompt",
            "teacher_prompt": "same-row TextMAS decoded Agent messages, training-only",
        }
    )
    splits = split_rows(rows, base_config.seed, base_config.train_ratio, base_config.valid_ratio)
    datasets = {
        name: TextTeacherLayerReceiverDataset(items, textmas_messages, tokenizer, base_config)
        for name, items in splits.items()
    }
    loaders = {
        name: DataLoader(dataset, batch_size=1, shuffle=(name == "train"), collate_fn=collate_one_text_teacher)
        for name, dataset in datasets.items()
    }
    receiver = DreamLayerConditionedReceiver(base_config).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        receiver.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.AdamW(receiver.parameters(), lr=base_config.learning_rate, weight_decay=base_config.weight_decay)
    run = init_experiment(
        stage="p3-dream-layer-receiver-text-teacher",
        config={
            **asdict(base_config),
            "textmas_generations_jsonl": args.textmas_generations_jsonl,
            "teacher_kl_weight": args.teacher_kl_weight,
            "teacher_cosine_weight": args.teacher_cosine_weight,
            "teacher_temperature": args.teacher_temperature,
            "corruption_margin": args.corruption_margin,
            "corruption_loss_weight": args.corruption_loss_weight,
            "corruption_types": corruption_types,
            "init_checkpoint": args.init_checkpoint,
            **device_metadata(device),
            "metadata": metadata,
            "split_sizes": {name: len(items) for name, items in splits.items()},
        },
        experiment_name=args.experiment_name,
        tags=["dream", "p3", "latentmas", "layer-receiver", "text-teacher", "swanlab-cloud"],
        mode=base_config.swanlab_mode,
    )

    metrics_path = output_dir / "metrics.jsonl"
    metrics_f = metrics_path.open("w", encoding="utf-8")
    best_metric = float("-inf")
    best_step = 0
    global_step = 0
    final_valid_metrics: dict[str, float] = {}
    final_test_metrics: dict[str, float] = {}
    try:
        for epoch in range(base_config.epochs):
            receiver.train()
            for batch in loaders["train"]:
                global_step += 1
                batch = move_batch(batch, device)
                shuffled_batch = move_batch(
                    collate_one_text_teacher([datasets["train"][rng.randrange(len(datasets["train"]))]]),
                    device,
                )
                optimizer.zero_grad(set_to_none=True)
                loss, train_metrics = text_teacher_loss(
                    receiver=receiver,
                    dream=dream,
                    batch=batch,
                    shuffled_packets=shuffled_batch["packets"],
                    tokenizer=tokenizer,
                    corruption_types=corruption_types,
                    corruption_margin=args.corruption_margin,
                    corruption_weight=args.corruption_loss_weight,
                    teacher_kl_weight=args.teacher_kl_weight,
                    teacher_cosine_weight=args.teacher_cosine_weight,
                    temperature=args.teacher_temperature,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(receiver.parameters(), base_config.grad_clip_norm)
                optimizer.step()
                train_metrics = {"loss": float(loss.detach().item()), **train_metrics, "epoch": float(epoch)}
                write_metrics(metrics_f, "train", global_step, train_metrics)
                log_metrics(train_metrics, step=global_step, prefix="train")
                if global_step % base_config.valid_interval == 0:
                    valid_metrics = evaluate_text_teacher_dataset(
                        receiver, dream, datasets["valid"], tokenizer, device, corruption_types, args
                    )
                    write_metrics(metrics_f, "valid", global_step, valid_metrics)
                    log_metrics(valid_metrics, step=global_step, prefix="valid")
                    if valid_metrics["selection_metric"] > best_metric:
                        best_metric = valid_metrics["selection_metric"]
                        best_step = global_step
                        save_checkpoint(output_dir / "best_checkpoint.pt", receiver, optimizer, base_config, metadata, best_step, best_metric)
                if base_config.max_train_steps and global_step >= base_config.max_train_steps:
                    break
            if base_config.max_train_steps and global_step >= base_config.max_train_steps:
                break
        final_valid_metrics = evaluate_text_teacher_dataset(receiver, dream, datasets["valid"], tokenizer, device, corruption_types, args)
        final_test_metrics = evaluate_text_teacher_dataset(receiver, dream, datasets["test"], tokenizer, device, corruption_types, args)
        write_metrics(metrics_f, "valid", global_step, final_valid_metrics)
        write_metrics(metrics_f, "test", global_step, final_test_metrics)
        log_metrics(final_valid_metrics, step=global_step, prefix="valid")
        log_metrics(final_test_metrics, step=global_step, prefix="test")
        if final_valid_metrics["selection_metric"] > best_metric:
            best_metric = final_valid_metrics["selection_metric"]
            best_step = global_step
            save_checkpoint(output_dir / "best_checkpoint.pt", receiver, optimizer, base_config, metadata, best_step, best_metric)
        save_checkpoint(output_dir / "last_checkpoint.pt", receiver, optimizer, base_config, metadata, global_step, final_valid_metrics["selection_metric"])
    finally:
        metrics_f.close()
        finish_experiment()

    best_valid_metrics = evaluate_text_teacher_checkpoint(
        output_dir / "best_checkpoint.pt", dream, datasets["valid"], tokenizer, device, corruption_types, args
    )
    best_test_metrics = evaluate_text_teacher_checkpoint(
        output_dir / "best_checkpoint.pt", dream, datasets["test"], tokenizer, device, corruption_types, args
    )
    summary = {
        "created_at": int(time.time()),
        "status": "pass",
        "config": asdict(base_config),
        "textmas_generations_jsonl": args.textmas_generations_jsonl,
        "teacher_kl_weight": args.teacher_kl_weight,
        "teacher_cosine_weight": args.teacher_cosine_weight,
        "teacher_temperature": args.teacher_temperature,
        "corruption_margin": args.corruption_margin,
        "corruption_loss_weight": args.corruption_loss_weight,
        "corruption_types": corruption_types,
        "init_checkpoint": args.init_checkpoint,
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
            "P3 D7.7 text-teacher layer receiver deep-learning training",
            "CUDA/GPU required",
            "SwanLab cloud required",
            "Dream frozen; layer receiver initialized from V7 and updated",
            "online student prompt contains no decoded Agent messages",
            "TextMAS decoded Agent messages are training-only teacher context",
            "gold answers are supervised loss targets only, not runtime inputs",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def text_teacher_loss(
    *,
    receiver: DreamLayerConditionedReceiver,
    dream: Any,
    batch: dict[str, torch.Tensor],
    shuffled_packets: torch.Tensor,
    tokenizer: Any,
    corruption_types: list[str],
    corruption_margin: float,
    corruption_weight: float,
    teacher_kl_weight: float,
    teacher_cosine_weight: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    matched_logits = receiver_target_logits(receiver, dream, batch, tokenizer)
    target_ids = batch["target_ids"]
    matched_ce = F.cross_entropy(matched_logits.reshape(-1, matched_logits.shape[-1]).float(), target_ids.reshape(-1))
    with torch.no_grad():
        teacher_logits = dream_target_logits(dream, batch, tokenizer)
    teacher_kl = kl_to_teacher(matched_logits, teacher_logits, temperature)
    teacher_cosine_loss = 1.0 - F.cosine_similarity(
        matched_logits.float().reshape(-1, matched_logits.shape[-1]),
        teacher_logits.float().reshape(-1, teacher_logits.shape[-1]),
        dim=-1,
    ).mean()
    corrupt_losses = {}
    if "zero" in corruption_types:
        corrupt_losses["zero"] = receiver_loss(receiver, dream, {**batch, "packets": torch.zeros_like(batch["packets"])}, tokenizer)[0]
    if "agent_swap" in corruption_types:
        corrupt_losses["agent_swap"] = receiver_loss(receiver, dream, {**batch, "packets": batch["packets"].flip(dims=[1])}, tokenizer)[0]
    if "shuffled_row" in corruption_types:
        corrupt_losses["shuffled_row"] = receiver_loss(receiver, dream, {**batch, "packets": shuffled_packets}, tokenizer)[0]
    margin_terms = {
        name: F.relu(matched_ce.detach() + corruption_margin - corrupt_loss)
        for name, corrupt_loss in corrupt_losses.items()
    }
    corruption_loss = sum(margin_terms.values()) / len(margin_terms) if margin_terms else matched_ce.new_zeros(())
    loss = matched_ce + teacher_kl_weight * teacher_kl + teacher_cosine_weight * teacher_cosine_loss + corruption_weight * corruption_loss
    with torch.no_grad():
        pred = matched_logits.argmax(dim=-1)
        token_acc = (pred == target_ids).float().mean()
        first_acc = (pred[:, :1] == target_ids[:, :1]).float().mean()
    metrics = {
        "matched_ce": float(matched_ce.detach().item()),
        "teacher_kl": float(teacher_kl.detach().item()),
        "teacher_cosine_loss": float(teacher_cosine_loss.detach().item()),
        "corruption_loss": float(corruption_loss.detach().item()),
        "token_accuracy": float(token_acc.detach().item()),
        "first_token_accuracy": float(first_acc.detach().item()),
    }
    for name, corrupt_loss in corrupt_losses.items():
        metrics[f"{name}_ce"] = float(corrupt_loss.detach().item())
        metrics[f"{name}_ce_margin"] = float((corrupt_loss.detach() - matched_ce.detach()).item())
        metrics[f"{name}_margin_violation"] = float(margin_terms[name].detach().item())
    return loss, metrics


@torch.no_grad()
def evaluate_text_teacher_dataset(
    receiver: DreamLayerConditionedReceiver,
    dream: Any,
    dataset: TextTeacherLayerReceiverDataset,
    tokenizer: Any,
    device: torch.device,
    corruption_types: list[str],
    args: argparse.Namespace,
) -> dict[str, float]:
    receiver.eval()
    matched = []
    teacher_kls = []
    teacher_cosines = []
    zero = []
    shuffled = []
    agent_swap = []
    token_acc = []
    first_acc = []
    for idx in range(len(dataset)):
        batch = move_batch(collate_one_text_teacher([dataset[idx]]), device)
        matched_logits = receiver_target_logits(receiver, dream, batch, tokenizer)
        target_ids = batch["target_ids"]
        matched_ce = F.cross_entropy(matched_logits.reshape(-1, matched_logits.shape[-1]).float(), target_ids.reshape(-1))
        teacher_logits = dream_target_logits(dream, batch, tokenizer)
        matched.append(float(matched_ce.item()))
        teacher_kls.append(float(kl_to_teacher(matched_logits, teacher_logits, args.teacher_temperature).item()))
        teacher_cosines.append(
            float(
                F.cosine_similarity(
                    matched_logits.float().reshape(-1, matched_logits.shape[-1]),
                    teacher_logits.float().reshape(-1, teacher_logits.shape[-1]),
                    dim=-1,
                ).mean().item()
            )
        )
        pred = matched_logits.argmax(dim=-1)
        token_acc.append(float((pred == target_ids).float().mean().item()))
        first_acc.append(float((pred[:, :1] == target_ids[:, :1]).float().mean().item()))
        if "zero" in corruption_types:
            zero.append(float(receiver_loss(receiver, dream, {**batch, "packets": torch.zeros_like(batch["packets"])}, tokenizer)[0].item()))
        if "agent_swap" in corruption_types:
            agent_swap.append(float(receiver_loss(receiver, dream, {**batch, "packets": batch["packets"].flip(dims=[1])}, tokenizer)[0].item()))
        if "shuffled_row" in corruption_types:
            shuffled_item = move_batch(collate_one_text_teacher([dataset[(idx + 1) % len(dataset)]]), device)
            shuffled.append(float(receiver_loss(receiver, dream, {**batch, "packets": shuffled_item["packets"]}, tokenizer)[0].item()))
    out = {
        "matched_ce": mean(matched),
        "teacher_kl": mean(teacher_kls),
        "teacher_cosine": mean(teacher_cosines),
        "token_accuracy": mean(token_acc),
        "first_token_accuracy": mean(first_acc),
        "num_rows": float(len(matched)),
    }
    if zero:
        out["zero_ce"] = mean(zero)
        out["zero_ce_margin"] = out["zero_ce"] - out["matched_ce"]
    if shuffled:
        out["shuffled_row_ce"] = mean(shuffled)
        out["shuffled_row_ce_margin"] = out["shuffled_row_ce"] - out["matched_ce"]
    if agent_swap:
        out["agent_swap_ce"] = mean(agent_swap)
        out["agent_swap_ce_margin"] = out["agent_swap_ce"] - out["matched_ce"]
    out["selection_metric"] = (
        -out["matched_ce"]
        - args.teacher_kl_weight * out["teacher_kl"]
        + args.teacher_cosine_weight * out["teacher_cosine"]
        + 0.05 * out.get("zero_ce_margin", 0.0)
        + 0.05 * out.get("shuffled_row_ce_margin", 0.0)
    )
    return out


def receiver_target_logits(
    receiver: DreamLayerConditionedReceiver,
    dream: Any,
    batch: dict[str, torch.Tensor],
    tokenizer: Any,
) -> torch.Tensor:
    prompt_ids = batch["prompt_ids"]
    target_ids = batch["target_ids"]
    mask_token_id = resolve_mask_token_id(dream, tokenizer)
    masked_target = torch.full_like(target_ids, mask_token_id)
    input_ids = torch.cat([prompt_ids, masked_target], dim=1)
    prompt_len = prompt_ids.shape[1]
    logits = receiver.forward_logits(dream, input_ids, batch["packets"], condition_start=prompt_len)
    shifted = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    return shifted[:, prompt_len : prompt_len + target_ids.shape[1], :]


def dream_target_logits(dream: Any, batch: dict[str, torch.Tensor], tokenizer: Any) -> torch.Tensor:
    teacher_prompt_ids = batch["teacher_prompt_ids"]
    target_ids = batch["target_ids"]
    mask_token_id = resolve_mask_token_id(dream, tokenizer)
    masked_target = torch.full_like(target_ids, mask_token_id)
    input_ids = torch.cat([teacher_prompt_ids, masked_target], dim=1)
    prompt_len = teacher_prompt_ids.shape[1]
    logits = dream_forward_logits(dream, input_ids)
    shifted = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    return shifted[:, prompt_len : prompt_len + target_ids.shape[1], :]


def dream_forward_logits(dream: Any, input_ids: torch.Tensor) -> torch.Tensor:
    hidden_states = dream.get_input_embeddings()(input_ids)
    seq_len = hidden_states.shape[1]
    position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
    position_embeddings = dream.model.rotary_emb(hidden_states, position_ids)
    for decoder_layer in dream.model.layers:
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=position_embeddings,
        )
        hidden_states = layer_outputs[0]
    hidden_states = dream.model.norm(hidden_states)
    return dream.lm_head(hidden_states)


def kl_to_teacher(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
    per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    return per_token.mean() * (temperature**2)


def evaluate_text_teacher_checkpoint(
    path: Path,
    dream: Any,
    dataset: TextTeacherLayerReceiverDataset,
    tokenizer: Any,
    device: torch.device,
    corruption_types: list[str],
    args: argparse.Namespace,
) -> dict[str, float]:
    checkpoint = torch.load(path, map_location=device)
    config = LayerReceiverConfig(**checkpoint["config"])
    receiver = DreamLayerConditionedReceiver(config).to(device)
    receiver.load_state_dict(checkpoint["model_state"])
    return evaluate_text_teacher_dataset(receiver, dream, dataset, tokenizer, device, corruption_types, args)


def collate_one_text_teacher(items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    if len(items) != 1:
        raise ValueError("collate_one_text_teacher requires batch_size=1")
    item = items[0]
    return {
        "packets": item["packets"].unsqueeze(0),
        "prompt_ids": item["prompt_ids"].unsqueeze(0),
        "teacher_prompt_ids": item["teacher_prompt_ids"].unsqueeze(0),
        "target_ids": item["target_ids"].unsqueeze(0),
    }


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


if __name__ == "__main__":
    main()
