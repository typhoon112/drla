"""Train D7.18 denoising-aligned Dream layer receiver.

D7.16 learned teacher-forced answer CE/margins but barely changed token transfer
decisions during sampled Dream denoising. This trainer keeps the same frozen
Dream layer-conditioned receiver architecture, but trains on partial-denoising
answer states and adds matched-vs-hard-control gold-token logit margins on
masked positions.
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
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.scripts.p3_train_dream_layer_conditioned_receiver import (  # noqa: E402
    DreamLayerConditionedReceiver,
    LayerReceiverConfig,
    LayerReceiverDataset,
    collate_one,
    load_training_rows,
    move_batch,
    parse_selected_layers,
    save_checkpoint,
    set_seed,
    split_rows,
    write_metrics,
)
from drla.scripts.p3_train_dream_layer_receiver_corruption_aware import (  # noqa: E402
    make_layer_config,
    parse_corruption_types,
    sample_corrupt_batch,
)
from drla.scripts.p3_train_dream_soft_prefix_adapter import resolve_mask_token_id  # noqa: E402
from drla.scripts.train_cola_readiness_model import device_metadata, require_cuda_training, resolve_device  # noqa: E402
from drla.tracking import finish_experiment, init_experiment, log_metrics  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_layer_receivers/"
    "dream_layer_receiver_d718_denoising_aligned_seed20260618"
)


def main() -> None:
    summary = train(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", default=LayerReceiverConfig.manifest_json)
    parser.add_argument("--online-inputs-jsonl", default=LayerReceiverConfig.online_inputs_jsonl)
    parser.add_argument("--packet-dir", default=LayerReceiverConfig.packet_dir)
    parser.add_argument("--model-path", default=LayerReceiverConfig.model_path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=LayerReceiverConfig.device)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=LayerReceiverConfig.dtype)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--train-ratio", type=float, default=LayerReceiverConfig.train_ratio)
    parser.add_argument("--valid-ratio", type=float, default=LayerReceiverConfig.valid_ratio)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--valid-interval", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
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
    parser.add_argument("--mask-ratios", default="1.0,0.75,0.5,0.25")
    parser.add_argument("--decision-margin", type=float, default=1.0)
    parser.add_argument("--decision-loss-weight", type=float, default=0.5)
    parser.add_argument("--control-top-margin", type=float, default=0.5)
    parser.add_argument("--control-top-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--decision-control-weights",
        default="",
        help="Optional per-control weights such as zero:0.2,shuffled_row:4.0.",
    )
    parser.add_argument(
        "--top-control-weights",
        default="",
        help="Optional per-control top-margin weights such as zero:0.1,shuffled_row:2.0.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["mean_hard_margin", "row_binding"],
        default="mean_hard_margin",
    )
    parser.add_argument("--corruption-types", default="zero,shuffled_row")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--swanlab-mode", default=LayerReceiverConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default="p3-dream-layer-receiver-d718-denoising-aligned")
    return parser.parse_args()


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.swanlab_mode != "cloud":
        raise ValueError("D7.18 denoising-aligned receiver training must use SwanLab cloud")
    if args.valid_interval > 10:
        raise ValueError("valid_interval must be <= 10 for current P3 training")
    if args.batch_size != 1:
        raise ValueError("This script currently requires --batch-size 1 because prompts have variable lengths")
    set_seed(args.seed)
    rng = random.Random(args.seed + 1818)
    device = resolve_device(args.device)
    require_cuda_training(device, "p3_train_dream_layer_receiver_denoising_aligned.py")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = make_layer_config(args)
    mask_ratios = parse_mask_ratios(args.mask_ratios)
    corruption_types = parse_corruption_types(args.corruption_types)
    if "agent_swap" in corruption_types:
        raise ValueError("agent_swap is diagnostic only for the current homogeneous evidence-agent protocol")
    decision_control_weights = parse_control_weights(args.decision_control_weights, corruption_types)
    top_control_weights = parse_control_weights(args.top_control_weights, corruption_types)

    model_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[config.dtype]
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    dream = AutoModel.from_pretrained(config.model_path, torch_dtype=model_dtype, trust_remote_code=True).to(device).eval()
    for param in dream.parameters():
        param.requires_grad_(False)
    mask_token_id = resolve_mask_token_id(dream, tokenizer)

    rows, metadata = load_training_rows(config)
    metadata["selected_layers"] = parse_selected_layers(config.selected_layers)
    metadata["objective"] = "partial-denoising matched CE + matched-vs-hard-control decision margins"
    metadata["mask_ratios"] = mask_ratios
    metadata["decision_margin"] = args.decision_margin
    metadata["decision_loss_weight"] = args.decision_loss_weight
    metadata["control_top_margin"] = args.control_top_margin
    metadata["control_top_loss_weight"] = args.control_top_loss_weight
    metadata["decision_control_weights"] = decision_control_weights
    metadata["top_control_weights"] = top_control_weights
    metadata["selection_mode"] = args.selection_mode
    metadata["corruption_types"] = corruption_types
    metadata["init_checkpoint"] = args.init_checkpoint
    splits = split_rows(rows, config.seed, config.train_ratio, config.valid_ratio)
    empty_splits = [name for name, items in splits.items() if not items]
    if empty_splits:
        raise ValueError(f"empty data splits {empty_splits}; adjust --train-ratio/--valid-ratio")
    datasets = {name: LayerReceiverDataset(items, tokenizer, config) for name, items in splits.items()}
    loaders = {
        name: DataLoader(dataset, batch_size=1, shuffle=(name == "train"), collate_fn=collate_one)
        for name, dataset in datasets.items()
    }

    receiver = DreamLayerConditionedReceiver(config).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        receiver.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.AdamW(receiver.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    run = init_experiment(
        stage="p3-dream-layer-receiver-denoising-aligned",
        config={
            **asdict(config),
            "mask_ratios": mask_ratios,
            "decision_margin": args.decision_margin,
            "decision_loss_weight": args.decision_loss_weight,
            "control_top_margin": args.control_top_margin,
            "control_top_loss_weight": args.control_top_loss_weight,
            "decision_control_weights": decision_control_weights,
            "top_control_weights": top_control_weights,
            "selection_mode": args.selection_mode,
            "corruption_types": corruption_types,
            "init_checkpoint": args.init_checkpoint,
            **device_metadata(device),
            "metadata": metadata,
            "split_sizes": {name: len(items) for name, items in splits.items()},
        },
        experiment_name=args.experiment_name,
        tags=["dream", "p3", "latentmas", "layer-receiver", "denoising-aligned", "swanlab-cloud"],
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
            receiver.train()
            for batch in loaders["train"]:
                global_step += 1
                batch = move_batch(batch, device)
                shuffled_batch = sample_corrupt_batch(datasets["train"], batch, rng, device)
                optimizer.zero_grad(set_to_none=True)
                loss, train_metrics = denoising_aligned_loss(
                    receiver=receiver,
                    dream=dream,
                    batch=batch,
                    shuffled_packets=shuffled_batch["packets"],
                    mask_token_id=mask_token_id,
                    mask_ratio=rng.choice(mask_ratios),
                    decision_margin=args.decision_margin,
                    decision_loss_weight=args.decision_loss_weight,
                    decision_control_weights=decision_control_weights,
                    control_top_margin=args.control_top_margin,
                    control_top_loss_weight=args.control_top_loss_weight,
                    top_control_weights=top_control_weights,
                    corruption_types=corruption_types,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(receiver.parameters(), config.grad_clip_norm)
                optimizer.step()
                train_metrics = {"loss": float(loss.detach().item()), **train_metrics, "epoch": float(epoch)}
                write_metrics(metrics_f, "train", global_step, train_metrics)
                log_metrics(train_metrics, step=global_step, prefix="train")
                if global_step % config.valid_interval == 0:
                    valid_metrics = evaluate_denoising_dataset(
                        receiver=receiver,
                        dream=dream,
                        dataset=datasets["valid"],
                        mask_token_id=mask_token_id,
                        device=device,
                        mask_ratios=mask_ratios,
                        decision_margin=args.decision_margin,
                        decision_control_weights=decision_control_weights,
                        control_top_margin=args.control_top_margin,
                        top_control_weights=top_control_weights,
                        corruption_types=corruption_types,
                        selection_mode=args.selection_mode,
                    )
                    write_metrics(metrics_f, "valid", global_step, valid_metrics)
                    log_metrics(valid_metrics, step=global_step, prefix="valid")
                    if valid_metrics["selection_metric"] > best_metric:
                        best_metric = valid_metrics["selection_metric"]
                        best_step = global_step
                        save_checkpoint(output_dir / "best_checkpoint.pt", receiver, optimizer, config, metadata, best_step, best_metric)
                if config.max_train_steps and global_step >= config.max_train_steps:
                    break
            if config.max_train_steps and global_step >= config.max_train_steps:
                break
        final_valid_metrics = evaluate_denoising_dataset(
            receiver=receiver,
            dream=dream,
            dataset=datasets["valid"],
            mask_token_id=mask_token_id,
            device=device,
            mask_ratios=mask_ratios,
            decision_margin=args.decision_margin,
            decision_control_weights=decision_control_weights,
            control_top_margin=args.control_top_margin,
            top_control_weights=top_control_weights,
            corruption_types=corruption_types,
            selection_mode=args.selection_mode,
        )
        final_test_metrics = evaluate_denoising_dataset(
            receiver=receiver,
            dream=dream,
            dataset=datasets["test"],
            mask_token_id=mask_token_id,
            device=device,
            mask_ratios=mask_ratios,
            decision_margin=args.decision_margin,
            decision_control_weights=decision_control_weights,
            control_top_margin=args.control_top_margin,
            top_control_weights=top_control_weights,
            corruption_types=corruption_types,
            selection_mode=args.selection_mode,
        )
        write_metrics(metrics_f, "valid", global_step, final_valid_metrics)
        write_metrics(metrics_f, "test", global_step, final_test_metrics)
        log_metrics(final_valid_metrics, step=global_step, prefix="valid")
        log_metrics(final_test_metrics, step=global_step, prefix="test")
        if final_valid_metrics["selection_metric"] > best_metric:
            best_metric = final_valid_metrics["selection_metric"]
            best_step = global_step
            save_checkpoint(output_dir / "best_checkpoint.pt", receiver, optimizer, config, metadata, best_step, best_metric)
        save_checkpoint(output_dir / "last_checkpoint.pt", receiver, optimizer, config, metadata, global_step, final_valid_metrics["selection_metric"])
    finally:
        metrics_f.close()
        finish_experiment()

    best_valid_metrics = evaluate_checkpoint_denoising(
        output_dir / "best_checkpoint.pt",
        dream,
        datasets["valid"],
        mask_token_id,
        device,
        mask_ratios,
        args.decision_margin,
        decision_control_weights,
        args.control_top_margin,
        top_control_weights,
        corruption_types,
        args.selection_mode,
    )
    best_test_metrics = evaluate_checkpoint_denoising(
        output_dir / "best_checkpoint.pt",
        dream,
        datasets["test"],
        mask_token_id,
        device,
        mask_ratios,
        args.decision_margin,
        decision_control_weights,
        args.control_top_margin,
        top_control_weights,
        corruption_types,
        args.selection_mode,
    )
    summary = {
        "created_at": int(time.time()),
        "status": "pass",
        "config": asdict(config),
        "mask_ratios": mask_ratios,
        "decision_margin": args.decision_margin,
        "decision_loss_weight": args.decision_loss_weight,
        "control_top_margin": args.control_top_margin,
        "control_top_loss_weight": args.control_top_loss_weight,
        "decision_control_weights": decision_control_weights,
        "top_control_weights": top_control_weights,
        "selection_mode": args.selection_mode,
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
            "P3 D7.18 denoising-aligned layer receiver deep-learning training",
            "CUDA/GPU required",
            "SwanLab cloud required",
            "Dream frozen; only packet memory encoder and layer conditioners update",
            "training states are partial-denoising answer states, not only all-mask targets",
            "matched packet is trained against hard controls on masked-position decision margins",
            "gold answers are supervised loss targets only, not runtime inputs",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def denoising_aligned_loss(
    *,
    receiver: DreamLayerConditionedReceiver,
    dream: Any,
    batch: dict[str, torch.Tensor],
    shuffled_packets: torch.Tensor,
    mask_token_id: int,
    mask_ratio: float,
    decision_margin: float,
    decision_loss_weight: float,
    decision_control_weights: dict[str, float],
    control_top_margin: float,
    control_top_loss_weight: float,
    top_control_weights: dict[str, float],
    corruption_types: list[str],
    deterministic_mask_index: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    input_ids, target_ids, answer_mask, prompt_len = make_partial_state(
        batch,
        mask_token_id,
        mask_ratio,
        deterministic_mask_index=deterministic_mask_index,
    )
    matched_logits = answer_logits(
        receiver.forward_logits(dream, input_ids, batch["packets"], condition_start=prompt_len),
        prompt_len,
        target_ids.shape[1],
    )
    matched_mask_logits = matched_logits[answer_mask]
    gold = target_ids[answer_mask]
    matched_ce = F.cross_entropy(matched_mask_logits.float(), gold.reshape(-1))
    matched_gold = matched_mask_logits.gather(-1, gold.reshape(-1, 1)).squeeze(-1)
    matched_top = matched_mask_logits.argmax(dim=-1)
    matched_token_acc = (matched_top == gold.reshape(-1)).float().mean()

    control_metrics: dict[str, float] = {}
    decision_terms: dict[str, torch.Tensor] = {}
    control_top_terms: dict[str, torch.Tensor] = {}
    for name, packets in control_packets(batch, shuffled_packets, corruption_types).items():
        control_logits = answer_logits(
            receiver.forward_logits(dream, input_ids, packets, condition_start=prompt_len),
            prompt_len,
            target_ids.shape[1],
        )
        control_mask_logits = control_logits[answer_mask]
        control_gold = control_mask_logits.gather(-1, gold.reshape(-1, 1)).squeeze(-1)
        control_top = control_mask_logits.max(dim=-1).values
        gold_margin = matched_gold - control_gold
        top_margin = matched_gold - control_top
        decision_terms[name] = F.relu(decision_margin - gold_margin).mean()
        control_top_terms[name] = F.relu(control_top_margin - top_margin).mean()
        with torch.no_grad():
            control_metrics[f"{name}_gold_margin"] = float(gold_margin.float().mean().item())
            control_metrics[f"{name}_top_margin"] = float(top_margin.float().mean().item())
            control_metrics[f"{name}_gold_margin_violation_rate"] = float((gold_margin < decision_margin).float().mean().item())
            control_metrics[f"{name}_top_margin_violation_rate"] = float((top_margin < control_top_margin).float().mean().item())
            control_metrics[f"{name}_decision_loss"] = float(decision_terms[name].detach().item())
            control_metrics[f"{name}_control_top_loss"] = float(control_top_terms[name].detach().item())

    decision_loss = weighted_mean_tensors(decision_terms, decision_control_weights, matched_ce)
    control_top_loss = weighted_mean_tensors(control_top_terms, top_control_weights, matched_ce)
    loss = matched_ce + decision_loss_weight * decision_loss + control_top_loss_weight * control_top_loss
    metrics = {
        "matched_ce": float(matched_ce.detach().item()),
        "matched_token_accuracy": float(matched_token_acc.detach().item()),
        "mask_ratio": float(mask_ratio),
        "masked_tokens": float(gold.numel()),
        "decision_loss": float(decision_loss.detach().item()),
        "control_top_loss": float(control_top_loss.detach().item()),
        **control_metrics,
    }
    return loss, metrics


@torch.no_grad()
def evaluate_denoising_dataset(
    *,
    receiver: DreamLayerConditionedReceiver,
    dream: Any,
    dataset: LayerReceiverDataset,
    mask_token_id: int,
    device: torch.device,
    mask_ratios: list[float],
    decision_margin: float,
    decision_control_weights: dict[str, float],
    control_top_margin: float,
    top_control_weights: dict[str, float],
    corruption_types: list[str],
    selection_mode: str,
) -> dict[str, float]:
    receiver.eval()
    totals: dict[str, list[float]] = {}
    for idx in range(len(dataset)):
        batch = move_batch(collate_one([dataset[idx]]), device)
        shuffled_item = dataset[(idx + 1) % len(dataset)]
        shuffled_batch = move_batch(collate_one([shuffled_item]), device)
        for ratio_index, mask_ratio in enumerate(mask_ratios):
            _, metrics = denoising_aligned_loss(
                receiver=receiver,
                dream=dream,
                batch=batch,
                shuffled_packets=shuffled_batch["packets"],
                mask_token_id=mask_token_id,
                mask_ratio=mask_ratio,
                decision_margin=decision_margin,
                decision_loss_weight=1.0,
                decision_control_weights=decision_control_weights,
                control_top_margin=control_top_margin,
                control_top_loss_weight=1.0,
                top_control_weights=top_control_weights,
                corruption_types=corruption_types,
                deterministic_mask_index=idx * len(mask_ratios) + ratio_index,
            )
            for key, value in metrics.items():
                totals.setdefault(key, []).append(float(value))
    out = {key: mean(values) for key, values in totals.items()}
    hard_margins = [out.get(f"{name}_gold_margin", 0.0) for name in corruption_types]
    out["hard_gold_margin_mean"] = mean(hard_margins)
    out["selection_metric"] = compute_selection_metric(out, selection_mode)
    out["num_rows"] = float(len(dataset))
    out["num_mask_ratios"] = float(len(mask_ratios))
    return out


def evaluate_checkpoint_denoising(
    path: Path,
    dream: Any,
    dataset: LayerReceiverDataset,
    mask_token_id: int,
    device: torch.device,
    mask_ratios: list[float],
    decision_margin: float,
    decision_control_weights: dict[str, float],
    control_top_margin: float,
    top_control_weights: dict[str, float],
    corruption_types: list[str],
    selection_mode: str,
) -> dict[str, float]:
    checkpoint = torch.load(path, map_location=device)
    config = LayerReceiverConfig(**checkpoint["config"])
    receiver = DreamLayerConditionedReceiver(config).to(device)
    receiver.load_state_dict(checkpoint["model_state"])
    return evaluate_denoising_dataset(
        receiver=receiver,
        dream=dream,
        dataset=dataset,
        mask_token_id=mask_token_id,
        device=device,
        mask_ratios=mask_ratios,
        decision_margin=decision_margin,
        decision_control_weights=decision_control_weights,
        control_top_margin=control_top_margin,
        top_control_weights=top_control_weights,
        corruption_types=corruption_types,
        selection_mode=selection_mode,
    )


def make_partial_state(
    batch: dict[str, torch.Tensor],
    mask_token_id: int,
    mask_ratio: float,
    *,
    deterministic_mask_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    prompt_ids = batch["prompt_ids"]
    target_ids = batch["target_ids"]
    prompt_len = prompt_ids.shape[1]
    answer_state = target_ids.clone()
    if mask_ratio >= 1.0:
        answer_mask = torch.ones_like(target_ids, dtype=torch.bool)
    elif mask_ratio <= 0.0:
        answer_mask = torch.zeros_like(target_ids, dtype=torch.bool)
    elif deterministic_mask_index is not None:
        answer_mask = deterministic_answer_mask(target_ids, mask_ratio, deterministic_mask_index)
    else:
        answer_mask = torch.rand(target_ids.shape, device=target_ids.device) < mask_ratio
    if not bool(answer_mask.any()):
        answer_mask[:, torch.randint(0, target_ids.shape[1], (1,), device=target_ids.device)] = True
    answer_state[answer_mask] = mask_token_id
    input_ids = torch.cat([prompt_ids, answer_state], dim=1)
    return input_ids, target_ids, answer_mask, prompt_len


def deterministic_answer_mask(target_ids: torch.Tensor, mask_ratio: float, offset: int) -> torch.Tensor:
    if target_ids.shape[0] != 1:
        raise ValueError("deterministic_answer_mask currently expects batch_size=1")
    length = target_ids.shape[1]
    num_mask = max(1, min(length, int(round(length * mask_ratio))))
    positions = (torch.arange(num_mask, device=target_ids.device) * length // num_mask + offset) % length
    mask = torch.zeros_like(target_ids, dtype=torch.bool)
    mask[:, positions] = True
    return mask


def answer_logits(logits: torch.Tensor, prompt_len: int, target_len: int) -> torch.Tensor:
    shifted = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    return shifted[:, prompt_len : prompt_len + target_len, :]


def control_packets(
    batch: dict[str, torch.Tensor],
    shuffled_packets: torch.Tensor,
    corruption_types: list[str],
) -> dict[str, torch.Tensor]:
    packets = {}
    if "zero" in corruption_types:
        packets["zero"] = torch.zeros_like(batch["packets"])
    if "shuffled_row" in corruption_types:
        packets["shuffled_row"] = shuffled_packets
    return packets


def parse_mask_ratios(value: str) -> list[float]:
    ratios = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not ratios:
        raise ValueError("mask_ratios must not be empty")
    for ratio in ratios:
        if ratio <= 0.0 or ratio > 1.0:
            raise ValueError(f"mask ratios must be in (0, 1], got {ratio}")
    return ratios


def parse_control_weights(value: str, corruption_types: list[str]) -> dict[str, float]:
    weights = {name: 1.0 for name in corruption_types}
    if not value.strip():
        return weights
    allowed = set(corruption_types)
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"control weight must be name:value, got {item!r}")
        name, raw_weight = item.split(":", 1)
        name = name.strip()
        if name not in allowed:
            raise ValueError(f"unknown control weight {name!r}; allowed={sorted(allowed)}")
        weight = float(raw_weight.strip())
        if weight < 0.0:
            raise ValueError(f"control weight must be non-negative, got {name}:{weight}")
        weights[name] = weight
    if sum(weights.values()) <= 0.0:
        raise ValueError("at least one control weight must be positive")
    return weights


def weighted_mean_tensors(
    terms: dict[str, torch.Tensor],
    weights: dict[str, float],
    fallback: torch.Tensor,
) -> torch.Tensor:
    active = [(name, term, float(weights.get(name, 1.0))) for name, term in terms.items()]
    active = [(name, term, weight) for name, term, weight in active if weight > 0.0]
    if not active:
        return fallback.new_zeros(())
    numerator = sum(term * weight for _, term, weight in active)
    denominator = sum(weight for _, _, weight in active)
    return numerator / denominator


def compute_selection_metric(metrics: dict[str, float], mode: str) -> float:
    if mode == "mean_hard_margin":
        return (
            -metrics.get("matched_ce", 0.0)
            + 0.10 * metrics.get("hard_gold_margin_mean", 0.0)
            - 0.05 * metrics.get("decision_loss", 0.0)
            - 0.02 * metrics.get("control_top_loss", 0.0)
        )
    if mode == "row_binding":
        return (
            -metrics.get("matched_ce", 0.0)
            + 0.80 * metrics.get("shuffled_row_gold_margin", 0.0)
            + 0.20 * metrics.get("shuffled_row_top_margin", 0.0)
            + 0.05 * metrics.get("zero_gold_margin", 0.0)
            - 0.10 * metrics.get("shuffled_row_gold_margin_violation_rate", 0.0)
            - 0.05 * metrics.get("shuffled_row_top_margin_violation_rate", 0.0)
        )
    raise ValueError(f"unknown selection mode: {mode}")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
