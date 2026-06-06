"""Train a P2-E post-request latent selector.

This selector is used after the receiver has requested additional sender
packets.  It consumes the first sender plus the requested sender packets and
selects which latent answer packet should be trusted.

Compared with the earlier hierarchical fuser, this model is anchor-aware: every
candidate sender state is represented together with its difference/product
relative to the first sender.  The loss combines continuous official-score
regression, listwise ranking, pairwise ranking, and gain-over-first regression.

Decoded answers and official scores are labels/evaluation references only.
They are not model inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from drla.scripts.audit_cola_hierarchical_aggregation_potential import (
    build_groups,
    choose_text_majority_selected,
)
from drla.scripts.audit_cola_sequential_latent_mas import load_official_scorer, score_text_with_official_rules
from drla.scripts.train_cola_hierarchical_latent_fuser import (
    HierarchicalLatentFuser,
    HierarchicalLatentFuserConfig,
    build_tensors,
    make_datasets,
    read_jsonl,
    split_groups,
)
from drla.scripts.train_cola_readiness_model import device_metadata, require_cuda_training, resolve_device
from drla.scripts.train_cola_request_more_policy import compute_fuser_refs, normalize_tensors
from drla.tracking import finish_experiment, init_experiment, log_metrics


@dataclass(frozen=True)
class PostRequestSelectorConfig:
    packets_jsonl: str = (
        "/data1/luyifei/drla/outputs/cola_agent_latent_comm/"
        "p2_agent_latent_comm_v2_locked_seed66_67_68_split20260601_20260529/"
        "agent_latent_comm_packets_test.jsonl"
    )
    output_dir: str = (
        "/data1/luyifei/drla/outputs/cola_post_request_selector/"
        "p2e_post_request_selector_v1"
    )
    fuser_checkpoint: str = (
        "/data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/"
        "p2e_hierarchical_fuser_score_full_seed20260529_20260529/checkpoints/"
        "best_checkpoint.pt"
    )
    request_policy_dir: str = (
        "/data1/luyifei/drla/outputs/cola_request_more_policy/"
        "p2e_request_more_policy_fuser_gain_full_seed20260529_20260529"
    )
    data_root: str = "/data1/luyifei/Cola-DLM/code/generate_task_data"
    acc_calc_script: str = "/data1/luyifei/Cola-DLM/code/scripts/acc_calc.py"
    seed: int = 20260529
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    max_groups: int = 0
    batch_size: int = 256
    epochs: int = 24
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    d_model: int = 128
    attention_heads: int = 4
    inter_layers: int = 2
    sender_layers: int = 2
    listwise_loss_weight: float = 1.0
    score_loss_weight: float = 1.0
    pairwise_loss_weight: float = 0.5
    gain_loss_weight: float = 0.5
    listwise_temperature: float = 0.25
    selection_output: str = "score"
    request_selection_mode: str = "target_request_rate"
    request_target_value: float = 0.25
    request_signal: str = "gain_pred"
    valid_interval: int = 10
    max_cached_shards: int = 1024
    num_workers: int = 0
    device: str = "auto"
    swanlab_mode: str = "cloud"
    experiment_name: str = "p2e-post-request-selector-v1"


class AnchorAwarePostRequestSelector(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        process_dim: int,
        certificate_dim: int,
        max_blocks: int,
        block_size: int,
        sender_count: int,
        task_count: int,
        d_model: int,
        attention_heads: int,
        inter_layers: int,
        sender_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = HierarchicalLatentFuser(
            latent_dim=latent_dim,
            process_dim=process_dim,
            certificate_dim=certificate_dim,
            max_blocks=max_blocks,
            block_size=block_size,
            sender_count=sender_count,
            task_count=task_count,
            d_model=d_model,
            attention_heads=attention_heads,
            inter_layers=inter_layers,
            sender_layers=sender_layers,
            dropout=dropout,
        )
        self.anchor_fusion = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.post_request_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=attention_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=sender_layers,
        )
        self.rank_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.score_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.gain_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        latent_blocks: torch.Tensor,
        process_features: torch.Tensor,
        block_mask: torch.Tensor,
        certificate_features: torch.Tensor,
        task_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        sender_state = self.backbone.encode_senders(
            latent_blocks=latent_blocks,
            process_features=process_features,
            block_mask=block_mask,
            certificate_features=certificate_features,
            task_idx=task_idx,
        )
        anchor = sender_state[:, :1].expand_as(sender_state)
        features = torch.cat([sender_state, anchor, sender_state - anchor, sender_state * anchor], dim=-1)
        candidate_state = self.anchor_fusion(features)
        candidate_state = self.post_request_encoder(candidate_state)
        score_logit = self.score_head(candidate_state).squeeze(-1)
        return {
            "rank_logits": self.rank_head(candidate_state).squeeze(-1),
            "score_pred": torch.sigmoid(score_logit),
            "gain_pred": self.gain_head(candidate_state).squeeze(-1),
        }


def main() -> None:
    summary = train_post_request_selector(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> PostRequestSelectorConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets-jsonl", default=PostRequestSelectorConfig.packets_jsonl)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fuser-checkpoint", default=PostRequestSelectorConfig.fuser_checkpoint)
    parser.add_argument("--request-policy-dir", default=PostRequestSelectorConfig.request_policy_dir)
    parser.add_argument("--data-root", default=PostRequestSelectorConfig.data_root)
    parser.add_argument("--acc-calc-script", default=PostRequestSelectorConfig.acc_calc_script)
    parser.add_argument("--seed", type=int, default=PostRequestSelectorConfig.seed)
    parser.add_argument("--train-ratio", type=float, default=PostRequestSelectorConfig.train_ratio)
    parser.add_argument("--valid-ratio", type=float, default=PostRequestSelectorConfig.valid_ratio)
    parser.add_argument("--max-groups", type=int, default=PostRequestSelectorConfig.max_groups)
    parser.add_argument("--batch-size", type=int, default=PostRequestSelectorConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=PostRequestSelectorConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=PostRequestSelectorConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=PostRequestSelectorConfig.weight_decay)
    parser.add_argument("--dropout", type=float, default=PostRequestSelectorConfig.dropout)
    parser.add_argument("--d-model", type=int, default=PostRequestSelectorConfig.d_model)
    parser.add_argument("--attention-heads", type=int, default=PostRequestSelectorConfig.attention_heads)
    parser.add_argument("--inter-layers", type=int, default=PostRequestSelectorConfig.inter_layers)
    parser.add_argument("--sender-layers", type=int, default=PostRequestSelectorConfig.sender_layers)
    parser.add_argument("--listwise-loss-weight", type=float, default=PostRequestSelectorConfig.listwise_loss_weight)
    parser.add_argument("--score-loss-weight", type=float, default=PostRequestSelectorConfig.score_loss_weight)
    parser.add_argument("--pairwise-loss-weight", type=float, default=PostRequestSelectorConfig.pairwise_loss_weight)
    parser.add_argument("--gain-loss-weight", type=float, default=PostRequestSelectorConfig.gain_loss_weight)
    parser.add_argument("--listwise-temperature", type=float, default=PostRequestSelectorConfig.listwise_temperature)
    parser.add_argument("--selection-output", choices=["score", "rank"], default=PostRequestSelectorConfig.selection_output)
    parser.add_argument("--request-selection-mode", default=PostRequestSelectorConfig.request_selection_mode)
    parser.add_argument("--request-target-value", type=float, default=PostRequestSelectorConfig.request_target_value)
    parser.add_argument("--request-signal", default=PostRequestSelectorConfig.request_signal)
    parser.add_argument("--valid-interval", type=int, default=PostRequestSelectorConfig.valid_interval)
    parser.add_argument("--max-cached-shards", type=int, default=PostRequestSelectorConfig.max_cached_shards)
    parser.add_argument("--num-workers", type=int, default=PostRequestSelectorConfig.num_workers)
    parser.add_argument("--device", default=PostRequestSelectorConfig.device)
    parser.add_argument("--swanlab-mode", default=PostRequestSelectorConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=PostRequestSelectorConfig.experiment_name)
    args = parser.parse_args()
    return PostRequestSelectorConfig(
        packets_jsonl=args.packets_jsonl,
        output_dir=args.output_dir,
        fuser_checkpoint=args.fuser_checkpoint,
        request_policy_dir=args.request_policy_dir,
        data_root=args.data_root,
        acc_calc_script=args.acc_calc_script,
        seed=args.seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        max_groups=args.max_groups,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        d_model=args.d_model,
        attention_heads=args.attention_heads,
        inter_layers=args.inter_layers,
        sender_layers=args.sender_layers,
        listwise_loss_weight=args.listwise_loss_weight,
        score_loss_weight=args.score_loss_weight,
        pairwise_loss_weight=args.pairwise_loss_weight,
        gain_loss_weight=args.gain_loss_weight,
        listwise_temperature=args.listwise_temperature,
        selection_output=args.selection_output,
        request_selection_mode=args.request_selection_mode,
        request_target_value=args.request_target_value,
        request_signal=args.request_signal,
        valid_interval=args.valid_interval,
        max_cached_shards=args.max_cached_shards,
        num_workers=args.num_workers,
        device=args.device,
        swanlab_mode=args.swanlab_mode,
        experiment_name=args.experiment_name,
    )


def train_post_request_selector(config: PostRequestSelectorConfig) -> dict[str, Any]:
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

    device = resolve_device(config.device)
    require_cuda_training(device, "train_cola_post_request_selector.py")
    scorer = load_official_scorer(Path(config.acc_calc_script))
    fuser_checkpoint = torch.load(config.fuser_checkpoint, map_location="cpu")
    fuser_train_config = HierarchicalLatentFuserConfig(**fuser_checkpoint["config"])
    data_config = replace(
        fuser_train_config,
        packets_jsonl=config.packets_jsonl,
        output_dir=config.output_dir,
        data_root=config.data_root,
        acc_calc_script=config.acc_calc_script,
        seed=config.seed,
        train_ratio=config.train_ratio,
        valid_ratio=config.valid_ratio,
        max_groups=config.max_groups,
        max_cached_shards=config.max_cached_shards,
        swanlab_mode="disabled",
    )
    groups = build_groups(read_jsonl(Path(config.packets_jsonl)), data_config, scorer)
    if config.max_groups:
        groups = groups[: config.max_groups]
    splits = split_groups(groups, data_config)
    tensors_by_split, metadata = build_tensors(groups, splits, data_config)
    normalized = normalize_tensors(tensors_by_split, fuser_checkpoint["norm_stats"])
    fuser_refs = compute_fuser_refs(
        fuser_checkpoint=fuser_checkpoint,
        fuser_config=fuser_train_config,
        metadata=metadata,
        normalized=normalized,
        tensors_by_split=tensors_by_split,
        device=device,
        batch_size=config.batch_size,
    )
    request_policy = load_request_policy(config, {name: len(indices) for name, indices in splits.items()})
    train_ds, valid_ds, test_ds = make_selector_datasets(normalized, tensors_by_split)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = AnchorAwarePostRequestSelector(
        latent_dim=metadata["latent_dim"],
        process_dim=metadata["process_dim"],
        certificate_dim=metadata["certificate_dim"],
        max_blocks=metadata["max_blocks"],
        block_size=metadata["block_size"],
        sender_count=metadata["sender_count"],
        task_count=len(metadata["task_to_idx"]),
        d_model=config.d_model,
        attention_heads=config.attention_heads,
        inter_layers=config.inter_layers,
        sender_layers=config.sender_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    run = init_experiment(
        stage="p2e-post-request-selector",
        config={**asdict(config), **device_metadata(device)},
        experiment_name=config.experiment_name,
        tags=["cola", "official-benchmark", "p2e", "post-request", "latent-selector"],
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
                outputs = model(*batch[:5])
                loss, train_metrics = compute_loss(outputs=outputs, target_scores=batch[6], config=config)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_metrics = {"loss": float(loss.detach().item()), **train_metrics}
                write_metrics(metrics_f, "train", global_step, train_metrics)
                log_metrics(train_metrics, step=global_step, prefix="train")
                if global_step % config.valid_interval == 0:
                    valid_metrics = evaluate_model(
                        model,
                        valid_loader,
                        tensors_by_split["valid"],
                        groups,
                        splits["valid"],
                        fuser_refs["valid"],
                        request_policy.get("valid"),
                        scorer,
                        device,
                        config,
                    )
                    write_metrics(metrics_f, "valid", global_step, valid_metrics)
                    log_metrics(numeric_metrics(valid_metrics), step=global_step, prefix="valid")
                    current = valid_metrics["selection_metric"]
                    if current > best_metric:
                        best_metric = current
                        best_step = global_step
                        save_checkpoint(checkpoint_dir / "best_checkpoint.pt", model, optimizer, config, metadata, best_step, best_metric)

        valid_metrics = evaluate_model(
            model,
            valid_loader,
            tensors_by_split["valid"],
            groups,
            splits["valid"],
            fuser_refs["valid"],
            request_policy.get("valid"),
            scorer,
            device,
            config,
        )
        test_metrics = evaluate_model(
            model,
            test_loader,
            tensors_by_split["test"],
            groups,
            splits["test"],
            fuser_refs["test"],
            request_policy.get("test"),
            scorer,
            device,
            config,
        )
        write_metrics(metrics_f, "valid", global_step, valid_metrics)
        write_metrics(metrics_f, "test", global_step, test_metrics)
        log_metrics(numeric_metrics(valid_metrics), step=global_step, prefix="valid")
        log_metrics(numeric_metrics(test_metrics), step=global_step, prefix="test")
        if valid_metrics["selection_metric"] > best_metric:
            best_metric = valid_metrics["selection_metric"]
            best_step = global_step
            save_checkpoint(checkpoint_dir / "best_checkpoint.pt", model, optimizer, config, metadata, best_step, best_metric)
        save_checkpoint(checkpoint_dir / "last_checkpoint.pt", model, optimizer, config, metadata, global_step, valid_metrics["selection_metric"])
    finally:
        metrics_f.close()
        finish_experiment()

    best_valid_metrics, best_test_metrics, predictions = evaluate_best_checkpoint(
        checkpoint_dir / "best_checkpoint.pt",
        config,
        metadata,
        valid_loader,
        test_loader,
        tensors_by_split,
        groups,
        splits,
        fuser_refs,
        request_policy,
        scorer,
        device,
    )
    artifacts = write_outputs(output_dir, config, predictions)
    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "fuser_train_config": asdict(fuser_train_config),
        "request_policy": request_policy.get("metadata", {}),
        "swanlab_run_id": getattr(run, "id", None),
        "split_sizes": {name: len(indices) for name, indices in splits.items()},
        "metadata": metadata,
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
            **artifacts,
        },
        "interpretation": (
            "P2-E post-request latent selector. The model is anchor-aware and "
            "evaluates both always-request selection and request-policy-gated "
            "selection against fuser/text/oracle controls."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def validate_config(config: PostRequestSelectorConfig) -> None:
    if config.valid_interval > 10:
        raise ValueError("valid_interval must be <= 10 steps")
    if config.swanlab_mode != "cloud":
        raise ValueError("training must use SwanLab cloud; pass --swanlab-mode cloud")
    if config.selection_output not in {"score", "rank"}:
        raise ValueError("selection_output must be score or rank")
    if config.listwise_temperature <= 0:
        raise ValueError("listwise_temperature must be positive")
    if not 0.0 < config.train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0.0 <= config.valid_ratio < 1.0:
        raise ValueError("valid_ratio must be in [0, 1)")
    if config.train_ratio + config.valid_ratio >= 1.0:
        raise ValueError("train_ratio + valid_ratio must be < 1")


def make_selector_datasets(
    normalized: dict[str, dict[str, torch.Tensor]],
    tensors_by_split: dict[str, dict[str, torch.Tensor]],
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    def dataset(split: str) -> TensorDataset:
        tensors = normalized[split]
        raw = tensors_by_split[split]
        return TensorDataset(
            tensors["latent_blocks"],
            tensors["process_features"],
            tensors["block_mask"],
            tensors["certificate_features"],
            tensors["task_idx"],
            raw["labels"],
            raw["target_scores"],
        )

    return dataset("train"), dataset("valid"), dataset("test")


def compute_loss(
    *,
    outputs: dict[str, torch.Tensor],
    target_scores: torch.Tensor,
    config: PostRequestSelectorConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    score_loss = F.mse_loss(outputs["score_pred"], target_scores)
    target_dist = F.softmax(target_scores / config.listwise_temperature, dim=1)
    listwise_loss = -(target_dist * F.log_softmax(outputs["rank_logits"], dim=1)).sum(dim=1).mean()
    pairwise_loss = compute_pairwise_loss(outputs["rank_logits"], target_scores)
    gain_target = target_scores - target_scores[:, :1]
    gain_loss = F.mse_loss(outputs["gain_pred"], gain_target)
    loss = (
        config.score_loss_weight * score_loss
        + config.listwise_loss_weight * listwise_loss
        + config.pairwise_loss_weight * pairwise_loss
        + config.gain_loss_weight * gain_loss
    )
    with torch.no_grad():
        selected = select_index(outputs, config.selection_output)
        row_ids = torch.arange(target_scores.shape[0], device=target_scores.device)
        selected_score = target_scores[row_ids, selected].mean()
        oracle_score = target_scores.max(dim=1).values.mean()
    return loss, {
        "score_loss": float(score_loss.detach().item()),
        "listwise_loss": float(listwise_loss.detach().item()),
        "pairwise_loss": float(pairwise_loss.detach().item()),
        "gain_loss": float(gain_loss.detach().item()),
        "batch_selected_score": float(selected_score.detach().item()),
        "batch_oracle_score": float(oracle_score.detach().item()),
    }


def compute_pairwise_loss(rank_logits: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
    diff_target = target_scores.unsqueeze(2) - target_scores.unsqueeze(1)
    diff_logit = rank_logits.unsqueeze(2) - rank_logits.unsqueeze(1)
    mask = diff_target.abs() > 1e-6
    if not mask.any():
        return torch.zeros((), device=rank_logits.device)
    signed_margin = diff_logit * diff_target.sign()
    weights = diff_target.abs()
    return (F.softplus(-signed_margin) * weights)[mask].mean()


@torch.no_grad()
def evaluate_model(
    model: AnchorAwarePostRequestSelector,
    loader: DataLoader,
    tensors: dict[str, torch.Tensor],
    groups: list[dict[str, Any]],
    indices: list[int],
    fuser_refs: dict[str, torch.Tensor],
    request_policy: dict[str, torch.Tensor] | None,
    scorer: Any,
    device: torch.device,
    config: PostRequestSelectorConfig,
) -> dict[str, float]:
    model.eval()
    outputs = {"rank_logits": [], "score_pred": [], "gain_pred": []}
    for batch in loader:
        batch = [item.to(device) for item in batch]
        batch_outputs = model(*batch[:5])
        for key in outputs:
            outputs[key].append(batch_outputs[key].cpu())
    cat_outputs = {key: torch.cat(value, dim=0) for key, value in outputs.items()}
    predictions = score_predictions(cat_outputs, tensors, groups, indices, fuser_refs, request_policy, scorer, config)
    return aggregate(predictions)


def score_predictions(
    outputs: dict[str, torch.Tensor],
    tensors: dict[str, torch.Tensor],
    groups: list[dict[str, Any]],
    indices: list[int],
    fuser_refs: dict[str, torch.Tensor],
    request_policy: dict[str, torch.Tensor] | None,
    scorer: Any,
    config: PostRequestSelectorConfig,
) -> list[dict[str, Any]]:
    selected = select_index(outputs, config.selection_output)
    labels = tensors["labels"]
    target_scores = tensors["target_scores"]
    request_mask = request_policy["request_mask"].bool() if request_policy is not None else torch.ones(len(indices), dtype=torch.bool)
    predictions = []
    for local_idx, group_index in enumerate(indices):
        group = groups[group_index]
        chosen_member = group["members"][int(selected[local_idx])]
        chosen_score = score_text_with_official_rules(
            task=str(group["task"]),
            text=str(chosen_member["selected_prediction"]),
            ground_truth=group["ground_truth"],
            choices=group["choices"],
            scorer=scorer,
        )
        first_member = group["members"][0]
        text_choice = choose_text_majority_selected(group["members"])
        text_score = score_text_with_official_rules(
            task=str(group["task"]),
            text=str(text_choice["prediction"]),
            ground_truth=group["ground_truth"],
            choices=group["choices"],
            scorer=scorer,
        )
        requested = bool(request_mask[local_idx].item())
        gated_correct = float(bool(chosen_score["correct"])) if requested else float(bool(first_member["selected_correct"]))
        gated_score = float(chosen_score["score"]) if requested else float(first_member.get("selected_score", float(bool(first_member["selected_correct"]))))
        predictions.append(
            {
                "task": group["task"],
                "sample_key": group["sample_key"],
                "sample_id": group["sample_id"],
                "model_selected_index": int(selected[local_idx]),
                "model_correct": float(bool(chosen_score["correct"])),
                "model_score": float(chosen_score["score"]),
                "request_policy_requested": float(requested),
                "request_policy_model_correct": gated_correct,
                "request_policy_model_score": gated_score,
                "first_correct": float(bool(first_member["selected_correct"])),
                "first_score": float(first_member.get("selected_score", float(bool(first_member["selected_correct"])))),
                "fuser_correct": float(fuser_refs["fuser_correct"][local_idx].item()),
                "fuser_score": float(fuser_refs["fuser_score"][local_idx].item()),
                "text_majority_correct": float(bool(text_score["correct"])),
                "text_majority_score": float(text_score["score"]),
                "oracle_any_correct": float(labels[local_idx].sum().item() > 0),
                "oracle_best_score": float(target_scores[local_idx].max().item()),
            }
        )
    return predictions


def aggregate(rows: list[dict[str, Any]], *, task: str = "all") -> dict[str, float]:
    if not rows:
        return {"task": task, "count": 0.0, "selection_metric": 0.0}
    metrics = {
        "task": task,
        "count": float(len(rows)),
        "model_selected_accuracy": mean(row["model_correct"] for row in rows),
        "model_mean_official_score": mean(row["model_score"] for row in rows),
        "request_policy_rate": mean(row["request_policy_requested"] for row in rows),
        "request_policy_model_accuracy": mean(row["request_policy_model_correct"] for row in rows),
        "request_policy_model_score": mean(row["request_policy_model_score"] for row in rows),
        "first_accuracy": mean(row["first_correct"] for row in rows),
        "first_score": mean(row["first_score"] for row in rows),
        "fuser_accuracy": mean(row["fuser_correct"] for row in rows),
        "fuser_score": mean(row["fuser_score"] for row in rows),
        "text_majority_accuracy": mean(row["text_majority_correct"] for row in rows),
        "text_majority_score": mean(row["text_majority_score"] for row in rows),
        "oracle_any_accuracy": mean(row["oracle_any_correct"] for row in rows),
        "oracle_best_score": mean(row["oracle_best_score"] for row in rows),
    }
    metrics["model_fraction_of_oracle_score_gap_closed"] = safe_gap_fraction(
        metrics["model_mean_official_score"],
        metrics["first_score"],
        metrics["oracle_best_score"],
    )
    metrics["request_policy_fraction_of_oracle_score_gap_closed"] = safe_gap_fraction(
        metrics["request_policy_model_score"],
        metrics["first_score"],
        metrics["oracle_best_score"],
    )
    metrics["selection_metric"] = metrics["model_mean_official_score"]
    return metrics


def select_index(outputs: dict[str, torch.Tensor], selection_output: str) -> torch.Tensor:
    if selection_output == "rank":
        return outputs["rank_logits"].argmax(dim=1)
    return outputs["score_pred"].argmax(dim=1)


def safe_gap_fraction(model_score: float, base_score: float, oracle_score: float) -> float:
    denom = oracle_score - base_score
    if denom <= 0:
        return 0.0
    return float((model_score - base_score) / denom)


def load_request_policy(config: PostRequestSelectorConfig, expected_sizes: dict[str, int]) -> dict[str, Any]:
    policy_dir = Path(config.request_policy_dir)
    if not policy_dir.exists():
        return {}
    policy_rows = read_csv(policy_dir / "policy_metrics.csv")
    threshold = None
    for row in policy_rows:
        if (
            str(row.get("selection_mode")) == config.request_selection_mode
            and str(row.get("signal")) == config.request_signal
            and abs(float(row.get("target_value", "nan")) - config.request_target_value) < 1e-9
        ):
            threshold = float(row["threshold"])
            break
    if threshold is None:
        return {}
    out: dict[str, Any] = {
        "metadata": {
            "request_policy_dir": str(policy_dir),
            "selection_mode": config.request_selection_mode,
            "target_value": config.request_target_value,
            "signal": config.request_signal,
            "threshold": threshold,
        }
    }
    for split in ["valid", "test"]:
        path = policy_dir / f"{split}_predictions.jsonl"
        if not path.exists():
            continue
        rows = read_jsonl(path)
        if len(rows) != expected_sizes.get(split):
            continue
        signal_values = torch.tensor([float(row[config.request_signal]) for row in rows], dtype=torch.float32)
        out[split] = {
            "signal": signal_values,
            "request_mask": signal_values >= threshold,
        }
    return out


def evaluate_best_checkpoint(
    path: Path,
    config: PostRequestSelectorConfig,
    metadata: dict[str, Any],
    valid_loader: DataLoader,
    test_loader: DataLoader,
    tensors_by_split: dict[str, dict[str, torch.Tensor]],
    groups: list[dict[str, Any]],
    splits: dict[str, list[int]],
    fuser_refs: dict[str, dict[str, torch.Tensor]],
    request_policy: dict[str, Any],
    scorer: Any,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float], dict[str, list[dict[str, Any]]]]:
    checkpoint = torch.load(path, map_location="cpu")
    model = AnchorAwarePostRequestSelector(
        latent_dim=metadata["latent_dim"],
        process_dim=metadata["process_dim"],
        certificate_dim=metadata["certificate_dim"],
        max_blocks=metadata["max_blocks"],
        block_size=metadata["block_size"],
        sender_count=metadata["sender_count"],
        task_count=len(metadata["task_to_idx"]),
        d_model=config.d_model,
        attention_heads=config.attention_heads,
        inter_layers=config.inter_layers,
        sender_layers=config.sender_layers,
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    valid_predictions = score_predictions_from_model(
        model,
        valid_loader,
        tensors_by_split["valid"],
        groups,
        splits["valid"],
        fuser_refs["valid"],
        request_policy.get("valid"),
        scorer,
        device,
        config,
    )
    test_predictions = score_predictions_from_model(
        model,
        test_loader,
        tensors_by_split["test"],
        groups,
        splits["test"],
        fuser_refs["test"],
        request_policy.get("test"),
        scorer,
        device,
        config,
    )
    return aggregate(valid_predictions), aggregate(test_predictions), {"valid": valid_predictions, "test": test_predictions}


@torch.no_grad()
def score_predictions_from_model(
    model: AnchorAwarePostRequestSelector,
    loader: DataLoader,
    tensors: dict[str, torch.Tensor],
    groups: list[dict[str, Any]],
    indices: list[int],
    fuser_refs: dict[str, torch.Tensor],
    request_policy: dict[str, torch.Tensor] | None,
    scorer: Any,
    device: torch.device,
    config: PostRequestSelectorConfig,
) -> list[dict[str, Any]]:
    model.eval()
    outputs = {"rank_logits": [], "score_pred": [], "gain_pred": []}
    for batch in loader:
        batch = [item.to(device) for item in batch]
        batch_outputs = model(*batch[:5])
        for key in outputs:
            outputs[key].append(batch_outputs[key].cpu())
    cat_outputs = {key: torch.cat(value, dim=0) for key, value in outputs.items()}
    return score_predictions(cat_outputs, tensors, groups, indices, fuser_refs, request_policy, scorer, config)


def write_outputs(
    output_dir: Path,
    config: PostRequestSelectorConfig,
    predictions: dict[str, list[dict[str, Any]]],
) -> dict[str, str]:
    valid_predictions_path = output_dir / "valid_predictions.jsonl"
    test_predictions_path = output_dir / "test_predictions.jsonl"
    per_task_path = output_dir / "test_per_task_metrics.csv"
    write_jsonl(valid_predictions_path, predictions["valid"])
    write_jsonl(test_predictions_path, predictions["test"])
    per_task = [
        aggregate([row for row in predictions["test"] if row["task"] == task], task=task)
        for task in sorted({row["task"] for row in predictions["test"]})
    ]
    write_csv(per_task_path, per_task)
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "valid_predictions_jsonl": str(valid_predictions_path),
        "test_predictions_jsonl": str(test_predictions_path),
        "test_per_task_metrics_csv": str(per_task_path),
        "config_json": str(config_path),
    }


def save_checkpoint(
    path: Path,
    model: AnchorAwarePostRequestSelector,
    optimizer: torch.optim.Optimizer,
    config: PostRequestSelectorConfig,
    metadata: dict[str, Any],
    step: int,
    metric: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "metadata": metadata,
            "step": step,
            "metric": metric,
        },
        path,
    )


def write_metrics(handle: Any, split: str, step: int, metrics: dict[str, float]) -> None:
    handle.write(json.dumps({"created_at": int(time.time()), "split": split, "step": step, "metrics": metrics}, sort_keys=True) + "\n")
    handle.flush()


def numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items() if isinstance(value, int | float)}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values: Any) -> float:
    values = [float(value) for value in values]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


if __name__ == "__main__":
    main()
