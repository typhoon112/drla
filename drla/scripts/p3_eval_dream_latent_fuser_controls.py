"""Evaluate D7 Dream latent fuser corruption controls.

This local-only script loads the D7 best checkpoint and compares matched
latent packets against shuffled-row, agent-swap, and zero-packet controls. It
does not train, generate, score text, or create SwanLab runs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.scripts.p3_train_dream_latent_fuser import (  # noqa: E402
    DreamLatentFuser,
    FuserConfig,
    load_rows,
    load_tensor,
    resolve_device,
    select_evenly_spaced,
    split_rows,
)


DEFAULT_FUSER_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_latent_fusers/"
    "dream_latent_fuser_v1_textmas_matched200_seed20260606_20260606"
)
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_latent_fuser_controls/"
    "dream_latent_fuser_v1_textmas_matched200_controls_20260606"
)


@dataclass(frozen=True)
class EvalConfig:
    fuser_dir: str = DEFAULT_FUSER_DIR
    checkpoint_name: str = "best_checkpoint.pt"
    output_dir: str = DEFAULT_OUTPUT_DIR
    device: str = "auto"
    bootstrap_samples: int = 1000
    overwrite: bool = False


def main() -> None:
    summary = evaluate(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fuser-dir", default=EvalConfig.fuser_dir)
    parser.add_argument("--checkpoint-name", default=EvalConfig.checkpoint_name)
    parser.add_argument("--output-dir", default=EvalConfig.output_dir)
    parser.add_argument("--device", default=EvalConfig.device)
    parser.add_argument("--bootstrap-samples", type=int, default=EvalConfig.bootstrap_samples)
    parser.add_argument("--overwrite", action="store_true")
    return EvalConfig(**vars(parser.parse_args()))


def evaluate(config: EvalConfig) -> dict[str, Any]:
    created_at = int(time.time())
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(config.fuser_dir) / config.checkpoint_name
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_config = FuserConfig(**checkpoint["config"])
    device = resolve_device(config.device)
    model = DreamLatentFuser(
        hidden_size=train_config.hidden_size,
        d_model=train_config.d_model,
        input_tokens_per_agent=train_config.input_tokens_per_agent,
        prefix_len=train_config.prefix_len,
        num_layers=train_config.num_layers,
        num_heads=train_config.num_heads,
        dropout=train_config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rows, metadata = load_rows(train_config)
    splits = split_rows(rows, train_config.seed, train_config.train_ratio, train_config.valid_ratio)
    all_rows = []
    metrics = {}
    for split_name, split_rows_ in splits.items():
        rows_out, split_metrics = evaluate_split(model, split_rows_, train_config, device, config.bootstrap_samples, created_at)
        for row in rows_out:
            row["split"] = split_name
        all_rows.extend(rows_out)
        metrics[split_name] = split_metrics

    rows_path = output_dir / "control_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "created_at": created_at,
        "status": "pass",
        "config": asdict(config),
        "fuser_dir": config.fuser_dir,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("step"),
        "metadata": metadata,
        "split_sizes": {name: len(items) for name, items in splits.items()},
        "metrics": metrics,
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "metrics_jsonl": str(metrics_path),
            "control_rows_jsonl": str(rows_path),
        },
        "execution_boundary": [
            "local-only P3 D7 latent fuser corruption-control evaluation",
            "no optimizer, backward, or weight update",
            "no model generation",
            "no SwanLab run",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


@torch.no_grad()
def evaluate_split(
    model: DreamLatentFuser,
    rows: list[dict[str, Any]],
    train_config: FuserConfig,
    device: torch.device,
    bootstrap_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row_outputs = []
    row_tensors = [materialize_row(row, train_config) for row in rows]
    for index, row in enumerate(rows):
        matched_inputs, target = row_tensors[index]
        shuffled_inputs, _ = row_tensors[(index + 1) % len(row_tensors)] if len(row_tensors) > 1 else row_tensors[index]
        variants = {
            "matched": matched_inputs,
            "shuffled_row": shuffled_inputs,
            "agent_swap": matched_inputs.flip(dims=[0]),
            "zero_packet": torch.zeros_like(matched_inputs),
        }
        for variant, inputs in variants.items():
            pred = model(inputs.unsqueeze(0).to(device))
            target_device = target.unsqueeze(0).to(device)
            row_outputs.append(
                {
                    "row_id": row["row_id"],
                    "sample_id": row["sample_id"],
                    "variant": variant,
                    "mse": float((pred - target_device).pow(2).flatten(1).mean().detach().cpu().item()),
                    "cosine": float(F.cosine_similarity(pred.flatten(1), target_device.flatten(1), dim=-1).detach().cpu().item()),
                }
            )
    metrics = summarize_rows(row_outputs, bootstrap_samples, seed)
    return row_outputs, metrics


def materialize_row(row: dict[str, Any], config: FuserConfig) -> tuple[torch.Tensor, torch.Tensor]:
    agent_tensors = []
    for agent_id in ["agent_a", "agent_b"]:
        tensor = load_tensor(row["agent_hidden_refs"][agent_id])
        agent_tensors.append(select_evenly_spaced(tensor, config.input_tokens_per_agent))
    target = select_evenly_spaced(load_tensor(row["solver_hidden_ref"]), config.prefix_len)
    return torch.stack(agent_tensors, dim=0).to(torch.float32), target.to(torch.float32)


def summarize_rows(rows: list[dict[str, Any]], bootstrap_samples: int, seed: int) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_row_variant: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        by_variant[row["variant"]].append(row)
        by_row_variant[(row["row_id"], row["variant"])] = row
    summary = {
        variant: {
            "num_rows": len(items),
            "mse": mean([float(item["mse"]) for item in items]),
            "cosine": mean([float(item["cosine"]) for item in items]),
        }
        for variant, items in sorted(by_variant.items())
    }
    margins = {}
    row_ids = sorted({row["row_id"] for row in rows})
    for variant in ["shuffled_row", "agent_swap", "zero_packet"]:
        mse_margins = [
            float(by_row_variant[(row_id, variant)]["mse"]) - float(by_row_variant[(row_id, "matched")]["mse"])
            for row_id in row_ids
        ]
        cosine_margins = [
            float(by_row_variant[(row_id, "matched")]["cosine"]) - float(by_row_variant[(row_id, variant)]["cosine"])
            for row_id in row_ids
        ]
        margins[variant] = {
            "mse_margin_vs_matched": mean(mse_margins),
            "cosine_margin_vs_matched": mean(cosine_margins),
            "mse_margin_vs_matched_95ci": bootstrap_ci(mse_margins, bootstrap_samples, seed + len(variant)),
            "cosine_margin_vs_matched_95ci": bootstrap_ci(cosine_margins, bootstrap_samples, seed + 17 + len(variant)),
        }
    return {"by_variant": summary, "margins": margins}


def bootstrap_ci(values: list[float], samples: int, seed: int) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(mean(draw))
    means.sort()
    return [means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
