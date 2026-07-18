"""Evaluate D7 V2 contrastive latent fuser controls.

This local-only evaluator checks whether matched packets retrieve their solver
latent target better than shuffled, agent-swapped, or zero packets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.scripts.p3_train_dream_latent_fuser import load_rows, load_tensor, select_evenly_spaced, split_rows  # noqa: E402
from drla.scripts.p3_train_dream_latent_fuser_contrastive import (  # noqa: E402
    ContrastiveConfig,
    ContrastiveLatentFuser,
    resolve_device,
)


DEFAULT_FUSER_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_latent_fusers/"
    "dream_latent_fuser_v2_contrastive_textmas_matched200_seed20260606_20260606"
)
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_latent_fuser_controls/"
    "dream_latent_fuser_v2_contrastive_textmas_matched200_controls_20260606"
)


@dataclass(frozen=True)
class EvalConfig:
    fuser_dir: str = DEFAULT_FUSER_DIR
    checkpoint_name: str = "best_checkpoint.pt"
    output_dir: str = DEFAULT_OUTPUT_DIR
    device: str = "auto"
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
    parser.add_argument("--overwrite", action="store_true")
    return EvalConfig(**vars(parser.parse_args()))


def evaluate(config: EvalConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(config.fuser_dir) / config.checkpoint_name
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_config = ContrastiveConfig(**checkpoint["config"])
    device = resolve_device(config.device)
    model = ContrastiveLatentFuser(train_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rows, metadata = load_rows_for_config(train_config)
    splits = split_rows(rows, train_config.seed, train_config.train_ratio, train_config.valid_ratio)
    metrics = {}
    rows_out = []
    for split_name, split_rows_ in splits.items():
        split_metrics, split_outputs = evaluate_split(model, split_rows_, train_config, device)
        metrics[split_name] = split_metrics
        for item in split_outputs:
            rows_out.append({"split": split_name, **item})
    rows_path = output_dir / "control_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows_out:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "created_at": int(time.time()),
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
            "local-only P3 D7 contrastive latent fuser controls",
            "no optimizer, backward, or weight update",
            "no model generation",
            "no SwanLab run",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


@torch.no_grad()
def evaluate_split(
    model: ContrastiveLatentFuser,
    rows: list[dict[str, Any]],
    config: ContrastiveConfig,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    packets, targets = materialize(rows, config)
    packets = packets.to(device)
    targets = targets.to(device)
    target_emb = model.target_encoder(targets)
    variants = {
        "matched": packets,
        "shuffled_row": packets.roll(shifts=1, dims=0),
        "agent_swap": packets.flip(dims=[1]),
        "zero_packet": torch.zeros_like(packets),
    }
    metrics = {}
    row_outputs = []
    labels = torch.arange(len(rows), device=device)
    random_top1 = 1.0 / len(rows) if rows else 0.0
    for name, variant_packets in variants.items():
        packet_emb = model.packet_encoder(variant_packets)
        logits = packet_emb @ target_emb.T / model.temperature
        top1 = (logits.argmax(dim=1) == labels).float()
        ranks = []
        for idx in range(logits.shape[0]):
            order = torch.argsort(logits[idx], descending=True)
            rank = int((order == idx).nonzero(as_tuple=True)[0].item()) + 1
            ranks.append(rank)
            row_outputs.append(
                {
                    "row_id": rows[idx]["row_id"],
                    "sample_id": rows[idx]["sample_id"],
                    "variant": name,
                    "rank": rank,
                    "target_logit": float(logits[idx, idx].detach().cpu().item()),
                    "best_logit": float(logits[idx].max().detach().cpu().item()),
                }
            )
        diag = logits.diag()
        offdiag = logits[~torch.eye(logits.shape[0], dtype=torch.bool, device=device)]
        metrics[name] = {
            "top1": float(top1.mean().detach().cpu().item()),
            "mean_rank": sum(ranks) / len(ranks) if ranks else 0.0,
            "mrr": sum(1.0 / rank for rank in ranks) / len(ranks) if ranks else 0.0,
            "diag_logit_mean": float(diag.mean().detach().cpu().item()),
            "offdiag_logit_mean": float(offdiag.mean().detach().cpu().item()) if offdiag.numel() else 0.0,
            "diag_minus_offdiag": float((diag.mean() - offdiag.mean()).detach().cpu().item()) if offdiag.numel() else 0.0,
            "random_top1": random_top1,
        }
    metrics["margins"] = {
        control: {
            "top1_margin": metrics["matched"]["top1"] - metrics[control]["top1"],
            "mrr_margin": metrics["matched"]["mrr"] - metrics[control]["mrr"],
            "diag_minus_offdiag_margin": metrics["matched"]["diag_minus_offdiag"] - metrics[control]["diag_minus_offdiag"],
        }
        for control in ["shuffled_row", "agent_swap", "zero_packet"]
    }
    return metrics, row_outputs


def materialize(rows: list[dict[str, Any]], config: ContrastiveConfig) -> tuple[torch.Tensor, torch.Tensor]:
    packet_items = []
    target_items = []
    for row in rows:
        agents = []
        for agent_id in ["agent_a", "agent_b"]:
            agents.append(select_evenly_spaced(load_tensor(row["agent_hidden_refs"][agent_id]), config.input_tokens_per_agent))
        packet_items.append(torch.stack(agents, dim=0).to(torch.float32))
        target_items.append(select_evenly_spaced(load_tensor(row["solver_hidden_ref"]), config.target_tokens).to(torch.float32))
    return torch.stack(packet_items, dim=0), torch.stack(target_items, dim=0)


def load_rows_for_config(config: ContrastiveConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from types import SimpleNamespace

    return load_rows(SimpleNamespace(packet_dir=config.packet_dir, trace_dir=config.trace_dir))


if __name__ == "__main__":
    main()
