"""Train a state-conditioned P2-E latent fuser.

This is the richer follow-up to the shallow receiver-state action selector.
The model keeps sender-level latent representations from the score-target
latent fuser and uses calibrated receiver-state features as side information.

Online inputs remain decoder-free:

* sanitized sender latent/process/certificate packet fields
* calibrated state verifier outputs
* fuser logits/probabilities/confidence
* train-split task/global priors

Decoded answers and official scores are offline supervision/evaluation labels
only.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from drla.scripts.audit_cola_hierarchical_aggregation_potential import build_groups
from drla.scripts.audit_cola_sequential_latent_mas import load_official_scorer
from drla.scripts.eval_cola_hierarchical_state_policy import (
    attach_priors,
    build_fuser_model,
    build_rows,
    build_state_model,
    collect_outputs,
    fit_priors,
    load_calibrators,
    validate_compatible_configs,
)
from drla.scripts.train_cola_hierarchical_latent_fuser import (
    HierarchicalLatentFuser,
    HierarchicalLatentFuserConfig,
    build_tensors,
    compute_loss,
    make_datasets,
    read_jsonl,
    selection_metric,
    split_groups,
)
from drla.scripts.train_cola_hierarchical_state_verifier import HierarchicalStateVerifierConfig
from drla.scripts.train_cola_readiness_model import device_metadata, require_cuda_training, resolve_device
from drla.scripts.train_cola_receiver_state_action_selector import build_feature_tensor, feature_names_for_mode, mean
from drla.tracking import finish_experiment, init_experiment, log_metrics


@dataclass(frozen=True)
class StateConditionedLatentFuserConfig:
    fuser_checkpoint: str = (
        "/data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/"
        "p2e_hierarchical_fuser_score_full_seed20260529_20260529/checkpoints/best_checkpoint.pt"
    )
    state_checkpoint: str = (
        "/data1/luyifei/drla/outputs/cola_hierarchical_state_verifier/"
        "p2e_state_verifier_full_seed20260529_20260529/checkpoints/best_checkpoint.pt"
    )
    calibration_report: str = (
        "/data1/luyifei/drla/outputs/cola_hierarchical_state_verifier/"
        "p2e_state_verifier_full_seed20260529_20260529_calibration_ablation/"
        "calibration_report.json"
    )
    output_dir: str = (
        "/data1/luyifei/drla/outputs/cola_state_conditioned_latent_fuser/"
        "p2e_state_conditioned_latent_fuser_v1"
    )
    feature_mode: str = "state_fuser_prior"
    seed: int = 20260529
    batch_size: int = 256
    epochs: int = 24
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    state_hidden_dim: int = 128
    dropout: float = 0.1
    freeze_backbone: bool = True
    rank_loss_weight: float = 0.5
    target_mode: str = "score"
    valid_interval: int = 10
    num_workers: int = 0
    device: str = "auto"
    swanlab_mode: str = "cloud"
    experiment_name: str = "p2e-state-conditioned-latent-fuser-v1"


class StateConditionedLatentFuser(nn.Module):
    def __init__(
        self,
        *,
        backbone: HierarchicalLatentFuser,
        state_feature_dim: int,
        d_model: int,
        state_hidden_dim: int,
        dropout: float,
        freeze_backbone: bool,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(state_feature_dim),
            nn.Linear(state_feature_dim, state_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(state_hidden_dim, d_model),
            nn.GELU(),
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        final = self.delta_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(
        self,
        latent_blocks: torch.Tensor,
        process_features: torch.Tensor,
        block_mask: torch.Tensor,
        certificate_features: torch.Tensor,
        task_idx: torch.Tensor,
        state_features: torch.Tensor,
    ) -> torch.Tensor:
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                sender_state = self.backbone.encode_senders(
                    latent_blocks=latent_blocks,
                    process_features=process_features,
                    block_mask=block_mask,
                    certificate_features=certificate_features,
                    task_idx=task_idx,
                )
                base_logits = self.backbone.head(sender_state).squeeze(-1)
        else:
            sender_state = self.backbone.encode_senders(
                latent_blocks=latent_blocks,
                process_features=process_features,
                block_mask=block_mask,
                certificate_features=certificate_features,
                task_idx=task_idx,
            )
            base_logits = self.backbone.head(sender_state).squeeze(-1)
        state = self.state_encoder(state_features).unsqueeze(1).expand(-1, sender_state.shape[1], -1)
        delta = self.delta_head(torch.cat([sender_state, state], dim=-1)).squeeze(-1)
        return base_logits + delta


def main() -> None:
    summary = train_state_conditioned_fuser(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> StateConditionedLatentFuserConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fuser-checkpoint", default=StateConditionedLatentFuserConfig.fuser_checkpoint)
    parser.add_argument("--state-checkpoint", default=StateConditionedLatentFuserConfig.state_checkpoint)
    parser.add_argument("--calibration-report", default=StateConditionedLatentFuserConfig.calibration_report)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--feature-mode",
        choices=["state_only", "fuser_only", "prior_only", "state_fuser", "state_prior", "state_fuser_prior"],
        default=StateConditionedLatentFuserConfig.feature_mode,
    )
    parser.add_argument("--seed", type=int, default=StateConditionedLatentFuserConfig.seed)
    parser.add_argument("--batch-size", type=int, default=StateConditionedLatentFuserConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=StateConditionedLatentFuserConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=StateConditionedLatentFuserConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=StateConditionedLatentFuserConfig.weight_decay)
    parser.add_argument("--state-hidden-dim", type=int, default=StateConditionedLatentFuserConfig.state_hidden_dim)
    parser.add_argument("--dropout", type=float, default=StateConditionedLatentFuserConfig.dropout)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--unfreeze-backbone", action="store_false", dest="freeze_backbone")
    parser.add_argument("--rank-loss-weight", type=float, default=StateConditionedLatentFuserConfig.rank_loss_weight)
    parser.add_argument("--target-mode", choices=["score", "task_aware_score"], default=StateConditionedLatentFuserConfig.target_mode)
    parser.add_argument("--valid-interval", type=int, default=StateConditionedLatentFuserConfig.valid_interval)
    parser.add_argument("--num-workers", type=int, default=StateConditionedLatentFuserConfig.num_workers)
    parser.add_argument("--device", default=StateConditionedLatentFuserConfig.device)
    parser.add_argument("--swanlab-mode", default=StateConditionedLatentFuserConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=StateConditionedLatentFuserConfig.experiment_name)
    args = parser.parse_args()
    return StateConditionedLatentFuserConfig(
        fuser_checkpoint=args.fuser_checkpoint,
        state_checkpoint=args.state_checkpoint,
        calibration_report=args.calibration_report,
        output_dir=args.output_dir,
        feature_mode=args.feature_mode,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        state_hidden_dim=args.state_hidden_dim,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        rank_loss_weight=args.rank_loss_weight,
        target_mode=args.target_mode,
        valid_interval=args.valid_interval,
        num_workers=args.num_workers,
        device=args.device,
        swanlab_mode=args.swanlab_mode,
        experiment_name=args.experiment_name,
    )


def train_state_conditioned_fuser(config: StateConditionedLatentFuserConfig) -> dict[str, Any]:
    validate_config(config)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    data = build_data(config)
    train_ds, valid_ds, test_ds, norm_stats = make_state_conditioned_datasets(data)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    device = resolve_device(config.device)
    require_cuda_training(device, "train_cola_state_conditioned_latent_fuser.py")
    model = build_model(config, data, device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    labels = data["train"]["labels"]
    pos = labels.sum().clamp_min(1.0)
    neg = labels.numel() - pos
    pos_weight = (neg / pos).to(device)

    run = init_experiment(
        stage="p2e-state-conditioned-latent-fuser",
        config={**asdict(config), **device_metadata(device), "pos_weight": float(pos_weight.detach().cpu())},
        experiment_name=config.experiment_name,
        tags=["cola", "official-benchmark", "p2e", "latent-fuser", "state-conditioned"],
        mode=config.swanlab_mode,
    )

    best_metric = float("-inf")
    best_step = 0
    global_step = 0
    metrics_f = metrics_path.open("w", encoding="utf-8")
    try:
        for _epoch in range(config.epochs):
            model.train()
            for batch in train_loader:
                global_step += 1
                batch = [item.to(device) for item in batch]
                optimizer.zero_grad(set_to_none=True)
                logits = model(*batch[:5], batch[9])
                loss, train_metrics = compute_loss(
                    logits=logits,
                    labels=batch[5],
                    target_scores=batch[6],
                    train_targets=batch[7],
                    group_weight=batch[8],
                    pos_weight=pos_weight,
                    rank_loss_weight=config.rank_loss_weight,
                    target_mode=config.target_mode,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
                optimizer.step()
                train_metrics = {"loss": float(loss.detach().item()), **train_metrics}
                write_metrics(metrics_f, "train", global_step, train_metrics)
                log_metrics(train_metrics, step=global_step, prefix="train")
                if global_step % config.valid_interval == 0:
                    valid_metrics = evaluate(model, valid_loader, data["valid"], device)
                    write_metrics(metrics_f, "valid", global_step, valid_metrics)
                    log_metrics(valid_metrics, step=global_step, prefix="valid")
                    current = selection_metric(valid_metrics, config.target_mode)
                    if current > best_metric:
                        best_metric = current
                        best_step = global_step
                        save_checkpoint(checkpoint_dir / "best_checkpoint.pt", model, optimizer, config, data["metadata"], norm_stats, best_step, best_metric)

        valid_metrics = evaluate(model, valid_loader, data["valid"], device)
        test_metrics = evaluate(model, test_loader, data["test"], device)
        write_metrics(metrics_f, "valid", global_step, valid_metrics)
        write_metrics(metrics_f, "test", global_step, test_metrics)
        log_metrics(valid_metrics, step=global_step, prefix="valid")
        log_metrics(test_metrics, step=global_step, prefix="test")
        current = selection_metric(valid_metrics, config.target_mode)
        if current > best_metric:
            best_metric = current
            best_step = global_step
            save_checkpoint(checkpoint_dir / "best_checkpoint.pt", model, optimizer, config, data["metadata"], norm_stats, best_step, best_metric)
        save_checkpoint(checkpoint_dir / "last_checkpoint.pt", model, optimizer, config, data["metadata"], norm_stats, global_step, current)
    finally:
        metrics_f.close()
        finish_experiment()

    best_checkpoint = torch.load(checkpoint_dir / "best_checkpoint.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    best_valid_metrics = evaluate(model, valid_loader, data["valid"], device)
    best_test_metrics = evaluate(model, test_loader, data["test"], device)
    write_predictions(output_dir / "test_predictions.jsonl", model, test_loader, data["test"], device)

    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "swanlab_run_id": getattr(run, "id", None),
        "split_sizes": {split: int(data[split]["labels"].shape[0]) for split in ["train", "valid", "test"]},
        "metadata": data["metadata"],
        "best_step": best_step,
        "best_valid_selection_metric": best_metric,
        "final_valid_metrics": valid_metrics,
        "final_test_metrics": test_metrics,
        "best_valid_metrics": best_valid_metrics,
        "best_test_metrics": best_test_metrics,
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "metrics_jsonl": str(metrics_path),
            "best_checkpoint": str(checkpoint_dir / "best_checkpoint.pt"),
            "last_checkpoint": str(checkpoint_dir / "last_checkpoint.pt"),
            "test_predictions_jsonl": str(output_dir / "test_predictions.jsonl"),
        },
        "interpretation": (
            "State-conditioned latent fuser. The model starts from the score-target "
            "latent fuser and learns a residual sender-logit correction from "
            "calibrated state side information."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_data(config: StateConditionedLatentFuserConfig) -> dict[str, Any]:
    fuser_checkpoint = torch.load(config.fuser_checkpoint, map_location="cpu")
    state_checkpoint = torch.load(config.state_checkpoint, map_location="cpu")
    fuser_config = HierarchicalLatentFuserConfig(**fuser_checkpoint["config"])
    state_config = HierarchicalStateVerifierConfig(**state_checkpoint["config"])
    validate_compatible_configs(state_config, fuser_config)
    scorer = load_official_scorer(Path(fuser_config.acc_calc_script))
    packets = read_jsonl(Path(fuser_config.packets_jsonl))
    groups = build_groups(packets, fuser_config, scorer)
    if fuser_config.max_groups:
        groups = groups[: fuser_config.max_groups]
    splits = split_groups(groups, fuser_config)
    tensors_by_split, metadata = build_tensors(groups, splits, fuser_config)
    raw_train, raw_valid, raw_test, raw_norm_stats = make_datasets(tensors_by_split, fuser_config)
    raw_datasets = {"train": raw_train, "valid": raw_valid, "test": raw_test}

    device = resolve_device(config.device)
    state_model = build_state_model(state_checkpoint, state_config, metadata, device)
    fuser_model = build_fuser_model(fuser_checkpoint, fuser_config, metadata, device)
    calibrators = load_calibrators(Path(config.calibration_report))
    rows_by_split = {}
    outputs_by_split = {}
    for split in ["train", "valid", "test"]:
        loader = DataLoader(raw_datasets[split], batch_size=config.batch_size, shuffle=False, num_workers=0)
        outputs = collect_outputs(state_model, fuser_model, loader, device, calibrators)
        outputs_by_split[split] = outputs
        rows_by_split[split] = build_rows(
            split=split,
            groups=groups,
            indices=splits[split],
            tensors=tensors_by_split[split],
            outputs=outputs,
            scorer=scorer,
        )
    priors = fit_priors(rows_by_split["train"])
    for split in ["train", "valid", "test"]:
        attach_priors(rows_by_split[split], priors)

    feature_names = feature_names_for_mode(config.feature_mode, sender_count=metadata["sender_count"])
    data: dict[str, Any] = {"groups": groups, "splits": splits, "raw_norm_stats": raw_norm_stats}
    for split, raw_dataset in [("train", raw_train), ("valid", raw_valid), ("test", raw_test)]:
        raw_tensors = raw_dataset.tensors
        data[split] = {
            "raw_tensors": raw_tensors,
            "state_features": build_feature_tensor(rows_by_split[split], outputs_by_split[split], config.feature_mode),
            "labels": tensors_by_split[split]["labels"].float(),
            "target_scores": tensors_by_split[split]["target_scores"].float(),
            "baseline_rows": rows_by_split[split],
        }
    data["metadata"] = {
        **metadata,
        "state_feature_dim": len(feature_names),
        "state_feature_names": feature_names,
        "feature_mode": config.feature_mode,
        "fuser_checkpoint": config.fuser_checkpoint,
        "state_checkpoint": config.state_checkpoint,
        "calibration_report": config.calibration_report,
        "base_fuser_config": asdict(fuser_config),
    }
    return data


def make_state_conditioned_datasets(data: dict[str, Any]) -> tuple[TensorDataset, TensorDataset, TensorDataset, dict[str, torch.Tensor]]:
    train_state = data["train"]["state_features"]
    state_mean = train_state.mean(dim=0, keepdim=True)
    state_std = train_state.std(dim=0, keepdim=True).clamp_min(1e-6)
    norm_stats = {**data["raw_norm_stats"], "state_mean": state_mean, "state_std": state_std}

    def dataset(split: str) -> TensorDataset:
        raw = data[split]["raw_tensors"]
        state = (data[split]["state_features"] - state_mean) / state_std
        return TensorDataset(*raw, state)

    return dataset("train"), dataset("valid"), dataset("test"), norm_stats


def build_model(config: StateConditionedLatentFuserConfig, data: dict[str, Any], device: torch.device) -> StateConditionedLatentFuser:
    fuser_checkpoint = torch.load(config.fuser_checkpoint, map_location="cpu")
    base_config = HierarchicalLatentFuserConfig(**fuser_checkpoint["config"])
    metadata = data["metadata"]
    backbone = HierarchicalLatentFuser(
        latent_dim=metadata["latent_dim"],
        process_dim=metadata["process_dim"],
        certificate_dim=metadata["certificate_dim"],
        max_blocks=metadata["max_blocks"],
        block_size=metadata["block_size"],
        sender_count=metadata["sender_count"],
        task_count=len(metadata["task_to_idx"]),
        d_model=base_config.d_model,
        attention_heads=base_config.attention_heads,
        inter_layers=base_config.inter_layers,
        sender_layers=base_config.sender_layers,
        dropout=base_config.dropout,
    )
    backbone.load_state_dict(fuser_checkpoint["model_state_dict"])
    return StateConditionedLatentFuser(
        backbone=backbone,
        state_feature_dim=metadata["state_feature_dim"],
        d_model=base_config.d_model,
        state_hidden_dim=config.state_hidden_dim,
        dropout=config.dropout,
        freeze_backbone=config.freeze_backbone,
    ).to(device)


@torch.no_grad()
def evaluate(
    model: StateConditionedLatentFuser,
    loader: DataLoader,
    split_data: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    logits_all = []
    for batch in loader:
        batch = [item.to(device) for item in batch]
        logits_all.append(model(*batch[:5], batch[9]).cpu())
    logits = torch.cat(logits_all, dim=0)
    labels = split_data["labels"]
    target_scores = split_data["target_scores"]
    rows = split_data["baseline_rows"]
    selected = logits.argmax(dim=1)
    row_ids = torch.arange(labels.shape[0])
    selected_correct = labels[row_ids, selected].float()
    selected_score = target_scores[row_ids, selected].float()
    oracle_any = (labels.sum(dim=1) > 0).float()
    oracle_best_score = target_scores.max(dim=1).values.float()
    any_mask = oracle_any.bool()
    return {
        "model_selected_accuracy": float(selected_correct.mean().item()),
        "model_mean_official_score": float(selected_score.mean().item()),
        "model_mean_target_score": float(selected_score.mean().item()),
        "single_sender_first_accuracy": mean(row["first_correct"] for row in rows),
        "single_sender_first_mean_official_score": mean(row["first_score"] for row in rows),
        "text_majority_selected_accuracy": mean(row["text_majority_correct"] for row in rows),
        "text_majority_mean_official_score": mean(row["text_majority_score"] for row in rows),
        "base_fuser_accuracy": mean(row["fuser_correct"] for row in rows),
        "base_fuser_mean_official_score": mean(row["fuser_score"] for row in rows),
        "oracle_any_selected_accuracy": float(oracle_any.mean().item()),
        "oracle_best_selected_mean_official_score": float(oracle_best_score.mean().item()),
        "model_selects_correct_when_any_correct": float(selected_correct[any_mask].mean().item()) if any_mask.any() else 0.0,
        "num_groups": float(labels.shape[0]),
    }


def write_predictions(
    path: Path,
    model: StateConditionedLatentFuser,
    loader: DataLoader,
    split_data: dict[str, Any],
    device: torch.device,
) -> None:
    model.eval()
    rows = split_data["baseline_rows"]
    labels = split_data["labels"]
    target_scores = split_data["target_scores"]
    offset = 0
    with path.open("w", encoding="utf-8") as handle, torch.no_grad():
        for batch in loader:
            batch_size = batch[0].shape[0]
            batch = [item.to(device) for item in batch]
            logits = model(*batch[:5], batch[9]).cpu()
            probs = torch.softmax(logits, dim=1)
            selected = probs.argmax(dim=1)
            for idx in range(batch_size):
                row_idx = offset + idx
                selected_idx = int(selected[idx].item())
                record = {
                    **rows[row_idx],
                    "state_conditioned_selected_index": selected_idx,
                    "state_conditioned_selected_correct": float(labels[row_idx, selected_idx].item()),
                    "state_conditioned_selected_score": float(target_scores[row_idx, selected_idx].item()),
                    "state_conditioned_sender_probs": [float(value) for value in probs[idx].tolist()],
                }
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            offset += batch_size


def save_checkpoint(
    path: Path,
    model: StateConditionedLatentFuser,
    optimizer: torch.optim.Optimizer,
    config: StateConditionedLatentFuserConfig,
    metadata: dict[str, Any],
    norm_stats: dict[str, torch.Tensor],
    step: int,
    metric: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "metadata": metadata,
            "norm_stats": norm_stats,
            "step": step,
            "metric": metric,
        },
        path,
    )


def write_metrics(handle: Any, split: str, step: int, metrics: dict[str, float]) -> None:
    handle.write(json.dumps({"created_at": int(time.time()), "split": split, "step": step, "metrics": metrics}, sort_keys=True) + "\n")
    handle.flush()


def validate_config(config: StateConditionedLatentFuserConfig) -> None:
    if config.valid_interval > 10:
        raise ValueError("valid_interval must be <= 10")
    if config.swanlab_mode != "cloud":
        raise ValueError("training must use SwanLab cloud; pass --swanlab-mode cloud")
    if config.target_mode not in {"score", "task_aware_score"}:
        raise ValueError("target_mode must be score or task_aware_score")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")


if __name__ == "__main__":
    main()
