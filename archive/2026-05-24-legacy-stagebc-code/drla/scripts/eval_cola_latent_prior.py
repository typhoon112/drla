"""Evaluate a saved Cola-latent prior checkpoint on a JSONL split."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from drla.models.stage_c import BlockCausalPrior, StageCPriorConfig
from drla.scripts.train_cola_latent_prior import (
    ColaLatentCollator,
    ColaLatentDataset,
    ColaLatentPriorConfig,
    build_flow_schedule,
    build_latent_rows,
    evaluate_prior,
    load_condition_encoder,
    resolve_device,
)
from drla.tracking import finish_experiment, init_experiment, log_metrics

try:
    from cola_dlm import ColaTextVAEModel
except ImportError as exc:  # pragma: no cover - integration-only dependency.
    raise ImportError("Set PYTHONPATH to the official Cola-DLM code directory before running this script.") from exc


def evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    eval_jsonl: Path,
    summary_json: Path,
    max_eval_samples: int,
    eval_batch_size: int,
    device_name: str,
    swanlab_mode: str,
    experiment_name: str,
) -> dict[str, object]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_config = ColaLatentPriorConfig(**payload["train_config"])
    config = ColaLatentPriorConfig(
        **{
            **asdict(train_config),
            "eval_jsonl": str(eval_jsonl),
            "max_eval_samples": max_eval_samples,
            "eval_batch_size": eval_batch_size,
            "device": device_name,
            "swanlab_mode": swanlab_mode,
            "experiment_name": experiment_name,
        }
    )
    device = resolve_device(config.device)
    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    vae = ColaTextVAEModel.from_pretrained(config.vae_path).to(device)
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    condition_encoder, condition_tokenizer = load_condition_encoder(config, device=device)
    rows = build_latent_rows(
        eval_jsonl,
        tokenizer=tokenizer,
        vae=vae,
        config=config,
        max_samples=config.max_eval_samples,
        device=device,
        condition_encoder=condition_encoder,
        condition_tokenizer=condition_tokenizer,
    )
    if condition_encoder is not None:
        del condition_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    loader = DataLoader(
        ColaLatentDataset(rows),
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=ColaLatentCollator(pad_token_id=config.pad_token_id),
    )
    model_config = StageCPriorConfig(**payload["model_config"])
    model = BlockCausalPrior(model_config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    flow_schedule = build_flow_schedule(config, device=device)
    metrics = evaluate_prior(
        model,
        vae,
        loader,
        tokenizer=tokenizer,
        config=config,
        flow_schedule=flow_schedule,
        device=device,
    )

    run = init_experiment(
        stage="cola-latent-prior-eval",
        experiment_name=config.experiment_name,
        description="Evaluation-only run for a saved Cola latent prior checkpoint.",
        config={
            **asdict(config),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_step": payload.get("global_step"),
            "num_eval_rows": len(rows),
        },
        mode=config.swanlab_mode,
        tags=["cola", "flow", "gsm8k", "latent-prior", "eval"],
    )
    try:
        log_metrics(metrics, prefix="valid")
    finally:
        finish_experiment()

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": payload.get("global_step"),
        "eval_jsonl": str(eval_jsonl),
        "num_eval_rows": len(rows),
        "metrics": metrics,
        "swanlab_run_id": getattr(run, "id", None),
        "config": asdict(config),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--max-eval-samples", type=int, required=True)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--swanlab-mode", default="cloud")
    parser.add_argument("--experiment-name", default="cola-latent-prior-eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_checkpoint(
        checkpoint_path=Path(args.checkpoint_path),
        eval_jsonl=Path(args.eval_jsonl),
        summary_json=Path(args.summary_json),
        max_eval_samples=args.max_eval_samples,
        eval_batch_size=args.eval_batch_size,
        device_name=args.device,
        swanlab_mode=args.swanlab_mode,
        experiment_name=args.experiment_name,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
