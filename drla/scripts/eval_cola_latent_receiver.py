"""Evaluate a P2 latent receiver compatibility checkpoint.

This script is local-only.  It reconstructs matched/corrupted packet examples
with the training split protocol, loads a saved receiver checkpoint, and writes
fresh compatibility metrics.  It does not update weights and must not use
SwanLab.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from drla.scripts.audit_cola_agent_latent_packet_distribution import (
    build_packet_indexes,
    normalize_control_types,
)
from drla.scripts.train_cola_latent_receiver import (
    LatentReceiverCompatibilityModel,
    LatentReceiverTrainConfig,
    build_split_tensors,
    evaluate,
    load_packets,
    split_packet_indices,
)
from drla.scripts.train_cola_readiness_model import resolve_device
from drla.tracking import require_swanlab_disabled_for_non_training


@dataclass(frozen=True)
class LatentReceiverEvalConfig:
    checkpoint: str
    output_dir: str
    packets_jsonl: str = ""
    eval_split: str = "test"
    control_types: str = ""
    max_packets: int = 0
    batch_size: int = 512
    num_workers: int = 0
    device: str = "auto"
    swanlab_mode: str = "disabled"


def main() -> None:
    summary = eval_latent_receiver(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> LatentReceiverEvalConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--packets-jsonl", default="")
    parser.add_argument("--eval-split", choices=["train", "valid", "test", "all"], default="test")
    parser.add_argument("--control-types", default="")
    parser.add_argument("--max-packets", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--swanlab-mode", default="disabled")
    args = parser.parse_args()
    if args.max_packets < 0:
        raise ValueError("max-packets must be non-negative")
    return LatentReceiverEvalConfig(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        packets_jsonl=args.packets_jsonl,
        eval_split=args.eval_split,
        control_types=args.control_types,
        max_packets=args.max_packets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        swanlab_mode=args.swanlab_mode,
    )


def eval_latent_receiver(config: LatentReceiverEvalConfig) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        config.swanlab_mode,
        script_kind="P2 latent receiver evaluation",
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
    train_config = LatentReceiverTrainConfig(**checkpoint["config"])
    packets_jsonl = config.packets_jsonl or train_config.packets_jsonl
    control_types = normalize_control_types(config.control_types or train_config.control_types)
    eval_train_config = LatentReceiverTrainConfig(
        **{
            **checkpoint["config"],
            "packets_jsonl": packets_jsonl,
            "control_types": ",".join(control_types),
            "max_packets": config.max_packets,
            "swanlab_mode": "cloud",
        },
    )
    packets = load_packets(Path(packets_jsonl), config.max_packets)
    splits = split_packet_indices(packets, eval_train_config)
    if config.eval_split == "all":
        indices = list(range(len(packets)))
    else:
        indices = splits[config.eval_split]
    eval_packets = [packets[index] for index in indices]
    split_tensors, warnings = build_split_tensors(
        split_packets=eval_packets,
        control_types=control_types,
        task_to_idx=checkpoint["metadata"]["task_to_idx"],
        max_blocks=checkpoint["metadata"]["max_blocks"],
        block_size=checkpoint["metadata"]["block_size"],
        latent_dim=checkpoint["metadata"]["latent_dim"],
        process_dim=checkpoint["metadata"]["process_dim"],
        envelope_dim=checkpoint["metadata"]["envelope_dim"],
        certificate_dim=checkpoint["metadata"]["certificate_dim"],
        config=eval_train_config,
        packet_indexes=build_packet_indexes(eval_packets),
    )
    dataset = make_eval_dataset(split_tensors, checkpoint["norm_stats"])
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    device = resolve_device(config.device)
    model = LatentReceiverCompatibilityModel(
        latent_dim=checkpoint["metadata"]["latent_dim"],
        process_dim=checkpoint["metadata"]["process_dim"],
        envelope_dim=checkpoint["metadata"]["envelope_dim"],
        certificate_dim=checkpoint["metadata"]["certificate_dim"],
        max_blocks=checkpoint["metadata"]["max_blocks"],
        block_size=checkpoint["metadata"]["block_size"],
        task_count=len(checkpoint["metadata"]["task_to_idx"]),
        d_model=train_config.d_model,
        attention_heads=train_config.attention_heads,
        inter_layers=train_config.inter_layers,
        dropout=train_config.dropout,
        input_mode=train_config.input_mode,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate(model, loader, device, control_types)
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps(
            {
                "created_at": int(time.time()),
                "split": config.eval_split,
                "metrics": metrics,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    control_path = output_dir / "control_metrics.csv"
    write_control_metrics(control_path, metrics, control_types)
    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "checkpoint": str(Path(config.checkpoint)),
        "train_config": checkpoint["config"],
        "eval_split": config.eval_split,
        "num_packets": len(eval_packets),
        "num_examples": int(split_tensors["label"].numel()),
        "control_types": control_types,
        "metrics": metrics,
        "control_generation_warnings": warnings[:50],
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "metrics_jsonl": str(metrics_path),
            "control_metrics_csv": str(control_path),
        },
        "interpretation": (
            "Local-only checkpoint re-evaluation for P2-C compatibility. "
            "This does not train and does not prove downstream task utility."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def make_eval_dataset(tensors: dict[str, torch.Tensor], norm_stats: dict[str, torch.Tensor]) -> TensorDataset:
    process = (
        tensors["process_features"] - norm_stats["process_mean"].view(1, 1, -1)
    ) / norm_stats["process_std"].view(1, 1, -1)
    process = process.masked_fill(~tensors["block_mask"].unsqueeze(-1), 0.0)
    certificate = (
        tensors["certificate_features"] - norm_stats["certificate_mean"]
    ) / norm_stats["certificate_std"]
    envelope = (tensors["envelope_features"] - norm_stats["envelope_mean"]) / norm_stats["envelope_std"]
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


def write_control_metrics(path: Path, metrics: dict[str, float], control_types: list[str]) -> None:
    rows = []
    for control_type in control_types:
        row = {
            "control_type": control_type,
            "score_mean": metrics.get(f"{control_type}_score_mean", ""),
            "auroc": metrics.get(f"{control_type}_auroc", ""),
            "score_gap": metrics.get(f"{control_type}_score_gap", ""),
        }
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
