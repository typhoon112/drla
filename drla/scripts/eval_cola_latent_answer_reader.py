"""Evaluate a P2 latent answer reader checkpoint.

This script is local-only.  It rebuilds the packet split used by
``train_cola_latent_answer_reader.py``, loads a checkpoint, retrieves answer
texts for the requested split, and scores them with the official Cola scorer.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from drla.scripts.audit_cola_sequential_latent_mas import load_official_scorer
from drla.scripts.train_cola_latent_answer_reader import (
    LatentAnswerReaderConfig,
    LatentAnswerReaderModel,
    build_examples,
    build_tensors,
    evaluate_reader,
    load_packets,
    make_datasets,
    split_indices,
)
from drla.scripts.train_cola_readiness_model import resolve_device
from drla.tracking import require_swanlab_disabled_for_non_training


@dataclass(frozen=True)
class LatentAnswerReaderEvalConfig:
    checkpoint: str
    output_dir: str
    eval_split: str = "test"
    batch_size: int = 512
    num_workers: int = 0
    device: str = "auto"
    swanlab_mode: str = "disabled"


def main() -> None:
    summary = eval_latent_answer_reader(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> LatentAnswerReaderEvalConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--swanlab-mode", default="disabled")
    args = parser.parse_args()
    return LatentAnswerReaderEvalConfig(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        eval_split=args.eval_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        swanlab_mode=args.swanlab_mode,
    )


def eval_latent_answer_reader(config: LatentAnswerReaderEvalConfig) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        config.swanlab_mode,
        script_kind="P2 latent answer reader evaluation",
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
    train_config = LatentAnswerReaderConfig(**checkpoint["config"])
    scorer = load_official_scorer(Path(train_config.acc_calc_script))
    packets = load_packets(Path(train_config.packets_jsonl), train_config.max_packets)
    examples = build_examples(packets, train_config, scorer)
    splits = split_indices(examples, train_config)
    tensors_by_split, metadata = build_tensors(examples, splits, train_config)
    _, valid_ds, test_ds, _ = make_datasets(tensors_by_split)
    if config.eval_split == "valid":
        dataset = valid_ds
    elif config.eval_split == "test":
        dataset = test_ds
    else:
        dataset, _, _, _ = make_datasets(tensors_by_split)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    device = resolve_device(config.device)
    model = LatentAnswerReaderModel(
        latent_dim=metadata["latent_dim"],
        process_dim=metadata["process_dim"],
        max_blocks=metadata["max_blocks"],
        block_size=metadata["block_size"],
        task_count=len(metadata["task_to_idx"]),
        d_model=train_config.d_model,
        attention_heads=train_config.attention_heads,
        inter_layers=train_config.inter_layers,
        dropout=train_config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate_reader(
        model,
        loader,
        tensors_by_split[config.eval_split],
        examples,
        splits[config.eval_split],
        scorer,
        device,
        train_config,
    )
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps(
            {"created_at": int(time.time()), "split": config.eval_split, "metrics": metrics},
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "checkpoint": str(Path(config.checkpoint)),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_metric": checkpoint.get("metric"),
        "train_config": checkpoint["config"],
        "eval_split": config.eval_split,
        "split_size": len(splits[config.eval_split]),
        "metrics": metrics,
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "metrics_jsonl": str(metrics_path),
        },
        "interpretation": (
            "Local-only latent-answer reader checkpoint evaluation. This does not train "
            "and should be compared against text_selected on the same split."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    main()
