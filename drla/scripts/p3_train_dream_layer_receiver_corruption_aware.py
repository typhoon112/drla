"""Train D7 V5 corruption-aware Dream layer-conditioned receiver.

V4 improved receiver generation over no-message but leaked a receiver prior:
zero and corrupted packets kept most of the gain. This trainer keeps the V4
layer-conditioned receiver architecture and changes the objective so the
matched packet must beat zero, shuffled-row, and agent-swap packets on the same
answer-token CE target.
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
    evaluate_checkpoint,
    evaluate_dataset,
    load_training_rows,
    move_batch,
    parse_selected_layers,
    receiver_loss,
    save_checkpoint,
    set_seed,
    split_rows,
    write_metrics,
)
from drla.scripts.train_cola_readiness_model import device_metadata, require_cuda_training, resolve_device  # noqa: E402
from drla.tracking import finish_experiment, init_experiment, log_metrics  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_layer_receivers/"
    "dream_layer_receiver_v2_corruptaware_textmas_matched200_seed20260607"
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
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--train-ratio", type=float, default=LayerReceiverConfig.train_ratio)
    parser.add_argument("--valid-ratio", type=float, default=LayerReceiverConfig.valid_ratio)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--valid-interval", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
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
    parser.add_argument("--corruption-margin", type=float, default=0.5)
    parser.add_argument("--corruption-loss-weight", type=float, default=0.5)
    parser.add_argument("--corruption-types", default="zero,shuffled_row,agent_swap")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--swanlab-mode", default=LayerReceiverConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default="p3-dream-layer-receiver-v2-corruptaware-textmas-matched200")
    return parser.parse_args()


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.swanlab_mode != "cloud":
        raise ValueError("D7 corruption-aware receiver training must use SwanLab cloud")
    if args.valid_interval > 10:
        raise ValueError("valid_interval must be <= 10 for current P3 training")
    if args.batch_size != 1:
        raise ValueError("This script currently requires --batch-size 1 because prompts have variable lengths")
    set_seed(args.seed)
    rng = random.Random(args.seed + 17)
    device = resolve_device(args.device)
    require_cuda_training(device, "p3_train_dream_layer_receiver_corruption_aware.py")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = make_layer_config(args)
    model_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[base_config.dtype]
    tokenizer = AutoTokenizer.from_pretrained(base_config.model_path, trust_remote_code=True)
    dream = AutoModel.from_pretrained(base_config.model_path, torch_dtype=model_dtype, trust_remote_code=True).to(device).eval()
    for param in dream.parameters():
        param.requires_grad_(False)
    rows, metadata = load_training_rows(base_config)
    metadata["selected_layers"] = parse_selected_layers(base_config.selected_layers)
    corruption_types = parse_corruption_types(args.corruption_types)
    metadata["objective"] = f"matched CE + corruption margin against {corruption_types} packets"
    metadata["corruption_margin"] = args.corruption_margin
    metadata["corruption_loss_weight"] = args.corruption_loss_weight
    metadata["corruption_types"] = corruption_types
    metadata["init_checkpoint"] = args.init_checkpoint
    splits = split_rows(rows, base_config.seed, base_config.train_ratio, base_config.valid_ratio)
    datasets = {name: LayerReceiverDataset(items, tokenizer, base_config) for name, items in splits.items()}
    loaders = {
        name: DataLoader(dataset, batch_size=1, shuffle=(name == "train"), collate_fn=collate_one)
        for name, dataset in datasets.items()
    }
    receiver = DreamLayerConditionedReceiver(base_config).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        receiver.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.AdamW(receiver.parameters(), lr=base_config.learning_rate, weight_decay=base_config.weight_decay)
    run = init_experiment(
        stage="p3-dream-layer-receiver-corruption-aware",
        config={
            **asdict(base_config),
            "corruption_margin": args.corruption_margin,
            "corruption_loss_weight": args.corruption_loss_weight,
            "corruption_types": corruption_types,
            "init_checkpoint": args.init_checkpoint,
            **device_metadata(device),
            "metadata": metadata,
            "split_sizes": {name: len(items) for name, items in splits.items()},
        },
        experiment_name=args.experiment_name,
        tags=["dream", "p3", "latentmas", "layer-receiver", "corruption-aware", "swanlab-cloud"],
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
                corrupt_batch = sample_corrupt_batch(datasets["train"], batch, rng, device)
                optimizer.zero_grad(set_to_none=True)
                loss, train_metrics = corruption_aware_loss(
                    receiver=receiver,
                    dream=dream,
                    batch=batch,
                    shuffled_packets=corrupt_batch["packets"],
                    tokenizer=tokenizer,
                    margin=args.corruption_margin,
                    corruption_weight=args.corruption_loss_weight,
                    corruption_types=corruption_types,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(receiver.parameters(), base_config.grad_clip_norm)
                optimizer.step()
                train_metrics = {"loss": float(loss.detach().item()), **train_metrics, "epoch": float(epoch)}
                write_metrics(metrics_f, "train", global_step, train_metrics)
                log_metrics(train_metrics, step=global_step, prefix="train")
                if global_step % base_config.valid_interval == 0:
                    valid_metrics = evaluate_dataset(receiver, dream, datasets["valid"], tokenizer, device)
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
        final_valid_metrics = evaluate_dataset(receiver, dream, datasets["valid"], tokenizer, device)
        final_test_metrics = evaluate_dataset(receiver, dream, datasets["test"], tokenizer, device)
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

    best_valid_metrics = evaluate_checkpoint(output_dir / "best_checkpoint.pt", dream, datasets["valid"], tokenizer, device)
    best_test_metrics = evaluate_checkpoint(output_dir / "best_checkpoint.pt", dream, datasets["test"], tokenizer, device)
    summary = {
        "created_at": int(time.time()),
        "status": "pass",
        "config": asdict(base_config),
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
            "P3 D7 V5 corruption-aware layer receiver deep-learning training",
            "CUDA/GPU required",
            "SwanLab cloud required",
            "Dream frozen; only packet memory encoder and layer conditioners update",
            "matched packet is trained against the corruption types specified by --corruption-types",
            "gold answers are supervised loss targets only, not runtime inputs",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def corruption_aware_loss(
    *,
    receiver: DreamLayerConditionedReceiver,
    dream: Any,
    batch: dict[str, torch.Tensor],
    shuffled_packets: torch.Tensor,
    tokenizer: Any,
    margin: float,
    corruption_weight: float,
    corruption_types: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    matched_loss, matched_metrics = receiver_loss(receiver, dream, batch, tokenizer)
    corrupt_losses = {}
    if "zero" in corruption_types:
        zero_batch = {**batch, "packets": torch.zeros_like(batch["packets"])}
        corrupt_losses["zero"] = receiver_loss(receiver, dream, zero_batch, tokenizer)[0]
    if "agent_swap" in corruption_types:
        swap_batch = {**batch, "packets": batch["packets"].flip(dims=[1])}
        corrupt_losses["agent_swap"] = receiver_loss(receiver, dream, swap_batch, tokenizer)[0]
    if "shuffled_row" in corruption_types:
        shuffled_batch = {**batch, "packets": shuffled_packets}
        corrupt_losses["shuffled_row"] = receiver_loss(receiver, dream, shuffled_batch, tokenizer)[0]

    margin_terms = {
        name: F.relu(matched_loss.detach() + margin - corrupt_loss)
        for name, corrupt_loss in corrupt_losses.items()
    }
    corruption_loss = sum(margin_terms.values()) / len(margin_terms) if margin_terms else matched_loss.new_zeros(())
    loss = matched_loss + corruption_weight * corruption_loss
    metrics = {
        "matched_ce": float(matched_loss.detach().item()),
        "token_accuracy": matched_metrics["token_accuracy"],
        "first_token_accuracy": matched_metrics["first_token_accuracy"],
        "corruption_loss": float(corruption_loss.detach().item()),
    }
    for name, corrupt_loss in corrupt_losses.items():
        metrics[f"{name}_ce"] = float(corrupt_loss.detach().item())
        metrics[f"{name}_ce_margin"] = float((corrupt_loss.detach() - matched_loss.detach()).item())
        metrics[f"{name}_margin_violation"] = float(margin_terms[name].detach().item())
    return loss, metrics


def parse_corruption_types(value: str) -> list[str]:
    allowed = {"zero", "shuffled_row", "agent_swap"}
    result = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(result) - allowed)
    if invalid:
        raise ValueError(f"invalid corruption types: {invalid}; allowed={sorted(allowed)}")
    if not result:
        raise ValueError("at least one corruption type is required")
    return result


def sample_corrupt_batch(
    dataset: LayerReceiverDataset,
    batch: dict[str, torch.Tensor],
    rng: random.Random,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    item = dataset[rng.randrange(len(dataset))]
    corrupt = move_batch(collate_one([item]), device)
    return {**batch, "packets": corrupt["packets"]}


def make_layer_config(args: argparse.Namespace) -> LayerReceiverConfig:
    return LayerReceiverConfig(
        manifest_json=args.manifest_json,
        online_inputs_jsonl=args.online_inputs_jsonl,
        packet_dir=args.packet_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_train_steps=args.max_train_steps,
        valid_interval=args.valid_interval,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        input_tokens_per_agent=args.input_tokens_per_agent,
        max_target_tokens=args.max_target_tokens,
        hidden_size=args.hidden_size,
        d_model=args.d_model,
        num_memory_layers=args.num_memory_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        selected_layers=args.selected_layers,
        swanlab_mode=args.swanlab_mode,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()
