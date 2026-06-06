"""Evaluate a saved CoLA-latent candidate ranker checkpoint.

This script is local-only. It restores the train/valid split from the
checkpoint config, evaluates the selected checkpoint on the nonheldout valid
split and calibration candidate set, and writes local metrics/predictions.
It never trains and must not create a SwanLab run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.scripts.train_cola_readiness_model import resolve_device
from drla.scripts.train_p2_phase_a_cola_latent_candidate_ranker import (
    CandidateRankDataset,
    LatentCandidateRanker,
    TrainConfig,
    evaluate,
    load_manifest_evidence,
    read_jsonl,
    split_train_valid,
    write_jsonl,
)
from drla.tracking import require_swanlab_disabled_for_non_training


DEFAULT_CHECKPOINT_PATH = (
    "/data1/luyifei/drla/outputs/p2_phase_a_cola_latent_candidate_ranker/"
    "musique_top128_step500_seed20260606_20260606/checkpoints/best_checkpoint.pt"
)
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p2_phase_a_cola_latent_candidate_ranker_evals/"
    "musique_top128_step500_best_full_eval_20260606"
)


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--swanlab-mode", default="disabled")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        args.swanlab_mode,
        script_kind="eval_p2_phase_a_cola_latent_candidate_ranker.py",
    )
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = TrainConfig(**checkpoint["config"])
    feature_dim = int(checkpoint["feature_dim"])
    if "feature_schema_version" not in checkpoint["config"]:
        config.feature_schema_version = 1 if feature_dim == 28 else 2
    interaction_mode = str(checkpoint.get("interaction_mode", getattr(config, "interaction_mode", "pooled")))
    interaction_dim = int(checkpoint.get("interaction_dim", getattr(config, "interaction_dim", 64)))
    if config.cola_code_path and config.cola_code_path not in sys.path:
        sys.path.insert(0, config.cola_code_path)
    from tokenizers import Tokenizer
    from cola_dlm import ColaTextVAEModel

    train_rows = read_jsonl(Path(config.train_candidates_jsonl))
    if config.max_train_samples:
        train_rows = train_rows[: config.max_train_samples]
    _, valid_source = split_train_valid(train_rows, config)
    eval_rows = read_jsonl(Path(config.eval_candidates_jsonl))
    if config.max_eval_samples:
        eval_rows = eval_rows[: config.max_eval_samples]

    train_evidence = load_manifest_evidence(Path(config.train_manifest_json))
    eval_evidence = load_manifest_evidence(Path(config.eval_manifest_json))
    valid_ds = CandidateRankDataset(
        rows=valid_source,
        evidence_by_sample=train_evidence,
        max_candidates=config.max_candidates,
        require_positive=False,
    )
    eval_ds = CandidateRankDataset(
        rows=eval_rows,
        evidence_by_sample=eval_evidence,
        max_candidates=config.max_candidates,
        require_positive=False,
    )
    valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False, collate_fn=lambda rows: rows)
    eval_loader = DataLoader(eval_ds, batch_size=config.batch_size, shuffle=False, collate_fn=lambda rows: rows)

    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(config.cola_tokenizer_path)
    vae = ColaTextVAEModel.from_pretrained(config.cola_vae_path).to(device).eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    model = LatentCandidateRanker(
        latent_dim=int(checkpoint["latent_dim"]),
        feature_dim=feature_dim,
        hidden_dim=config.hidden_dim,
        interaction_dim=interaction_dim,
        interaction_mode=interaction_mode,
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    valid_metrics, valid_predictions = evaluate(
        valid_loader,
        model,
        tokenizer,
        vae,
        device,
        config,
        max_batches=args.max_batches,
    )
    eval_metrics, eval_predictions = evaluate(
        eval_loader,
        model,
        tokenizer,
        vae,
        device,
        config,
        max_batches=args.max_batches,
    )

    valid_predictions_path = output_dir / "valid_predictions.jsonl"
    eval_predictions_path = output_dir / "eval_predictions.jsonl"
    metrics_path = output_dir / "metrics.jsonl"
    write_jsonl(valid_predictions_path, valid_predictions)
    write_jsonl(eval_predictions_path, eval_predictions)
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"split": "valid", **valid_metrics}, ensure_ascii=False, sort_keys=True) + "\n")
        f.write(json.dumps({"split": "eval", **eval_metrics}, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "created_at": int(time.time()),
        "status": "pass",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("global_step", -1)),
        "checkpoint_selection_metric": float(checkpoint.get("selection_metric", float("nan"))),
        "feature_schema_version": config.feature_schema_version,
        "interaction_mode": interaction_mode,
        "interaction_dim": interaction_dim,
        "output_dir": str(output_dir),
        "valid_metrics": valid_metrics,
        "eval_metrics": eval_metrics,
        "dataset": {
            "valid_rows_used": len(valid_ds),
            "eval_rows_used": len(eval_ds),
            "max_batches": args.max_batches,
        },
        "artifacts": {
            "metrics_jsonl": str(metrics_path),
            "valid_predictions": str(valid_predictions_path),
            "eval_predictions": str(eval_predictions_path),
        },
        "execution_boundary": [
            "local-only checkpoint evaluation",
            "no optimizer or backward",
            "no SwanLab run",
            "no held-out data",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    main()
