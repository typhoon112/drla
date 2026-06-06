"""Train a CoLA-latent candidate answer ranker for Phase A.

This is a deep-learning training script. It freezes the official CoLA VAE,
encodes only online question/evidence/candidate text into CoLA latent
representations, and trains a lightweight ranker to select an evidence-derived
candidate answer. Gold/scorer fields are used only as offline supervision and
evaluation targets.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.evaluation.p2_phase_c_scorers import score_qa_answer
from drla.scripts.train_cola_readiness_model import device_metadata, require_cuda_training, resolve_device
from drla.tracking import finish_experiment, init_experiment, log_metrics


DEFAULT_TRAIN_CANDIDATES_JSONL = (
    "/data1/luyifei/drla/outputs/p2_phase_a_candidate_answers/"
    "musique_train_candidate_answers_10000_seed20260606_20260606/candidates.jsonl"
)
DEFAULT_EVAL_CANDIDATES_JSONL = (
    "/data1/luyifei/drla/outputs/p2_phase_a_candidate_answers/"
    "musique_calibration_candidate_answers_200_seed20260606_20260606/candidates.jsonl"
)
DEFAULT_TRAIN_MANIFEST_JSON = (
    "/data1/luyifei/drla/outputs/p2_phase_c_manifests/"
    "musique_interface_train_manifest_10000_seed20260606/manifest.json"
)
DEFAULT_EVAL_MANIFEST_JSON = (
    "/data1/luyifei/drla/outputs/p2_phase_c_manifests/"
    "musique_calibration_manifest_200_seed20260601/manifest.json"
)
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p2_phase_a_cola_latent_candidate_ranker/"
    "musique_top128_train10000_calib200_seed20260606"
)
DEFAULT_COLA_VAE_PATH = "/data1/luyifei/drla/models/Cola-DLM/cola_dlm/cola_vae"
DEFAULT_COLA_TOKENIZER_PATH = "/data1/luyifei/drla/models/Cola-DLM/tokenizer.json"
DEFAULT_COLA_CODE_PATH = "/data1/luyifei/Cola-DLM/code"

RULES_V1 = [
    "title",
    "date_phrase",
    "quantity_phrase",
    "number_word",
    "language_phrase",
    "century_phrase",
    "season_year_phrase",
    "capitalized_phrase",
    "other",
]
RULES_V2 = [
    "title",
    "capitalized_full_span",
    "capitalized_subspan",
    "capitalized_single",
    "number_or_year",
    "number_word",
    "date_phrase",
    "quoted_span",
    "quantity_phrase",
    "language_phrase",
    "century_phrase",
    "season_phrase",
    "season_year_phrase",
    "other",
]
RULES = RULES_V2
QTYPES = ["who", "what", "where", "when", "which", "how_many", "language", "other"]
EVIDENCE_KINDS = ["support", "distractor", "unknown"]


@dataclass
class TrainConfig:
    train_candidates_jsonl: str = DEFAULT_TRAIN_CANDIDATES_JSONL
    eval_candidates_jsonl: str = DEFAULT_EVAL_CANDIDATES_JSONL
    train_manifest_json: str = DEFAULT_TRAIN_MANIFEST_JSON
    eval_manifest_json: str = DEFAULT_EVAL_MANIFEST_JSON
    output_dir: str = DEFAULT_OUTPUT_DIR
    cola_vae_path: str = DEFAULT_COLA_VAE_PATH
    cola_tokenizer_path: str = DEFAULT_COLA_TOKENIZER_PATH
    cola_code_path: str = DEFAULT_COLA_CODE_PATH
    device: str = "cuda"
    seed: int = 20260606
    max_candidates: int = 128
    max_train_samples: int = 0
    max_eval_samples: int = 0
    valid_ratio: float = 0.1
    batch_size: int = 1
    epochs: int = 1
    max_train_steps: int = 0
    valid_interval: int = 10
    max_valid_batches: int = 0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    hidden_dim: int = 192
    interaction_dim: int = 64
    interaction_mode: str = "pooled"
    feature_schema_version: int = 2
    dropout: float = 0.1
    max_context_tokens: int = 0
    max_candidate_tokens: int = 96
    swanlab_mode: str = "cloud"
    experiment_name: str = "p2a-cola-latent-candidate-ranker-top128"


class CandidateRankDataset(Dataset):
    def __init__(
        self,
        *,
        rows: list[dict[str, Any]],
        evidence_by_sample: dict[str, str],
        max_candidates: int,
        require_positive: bool,
    ) -> None:
        self.items: list[dict[str, Any]] = []
        self.skipped: Counter[str] = Counter()
        for row in rows:
            item = build_item(row, evidence_by_sample, max_candidates=max_candidates)
            if not item["candidates"]:
                self.skipped["no_candidates"] += 1
                continue
            if require_positive and not any(label > 0.5 for label in item["labels"]):
                self.skipped["no_primary_positive"] += 1
                continue
            self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


class LatentCandidateRanker(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        feature_dim: int,
        hidden_dim: int,
        interaction_dim: int,
        interaction_mode: str,
        dropout: float,
    ) -> None:
        super().__init__()
        if interaction_mode not in {"pooled", "late_maxsim"}:
            raise ValueError("interaction_mode must be pooled or late_maxsim")
        self.interaction_mode = interaction_mode
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.pool_query = nn.Parameter(torch.randn(hidden_dim) / math.sqrt(hidden_dim))
        self.interaction_proj = nn.Linear(hidden_dim, interaction_dim)
        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        interaction_feature_dim = 4 if interaction_mode == "late_maxsim" else 0
        joint_dim = hidden_dim * 4 + hidden_dim // 2 + interaction_feature_dim
        self.joint = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.score_head = nn.Linear(hidden_dim // 2, 1)

    def encode_sequences(self, sequences: list[torch.Tensor]) -> list[torch.Tensor]:
        if not sequences:
            raise ValueError("cannot encode an empty sequence list")
        lengths = torch.tensor([seq.shape[0] for seq in sequences], device=sequences[0].device)
        padded = pad_sequence(sequences, batch_first=True)
        mask = torch.arange(padded.shape[1], device=padded.device).unsqueeze(0) < lengths.unsqueeze(1)
        hidden = self.latent_proj(self.latent_norm(padded))
        return [hidden[index, : int(length.item())] for index, length in enumerate(lengths)]

    def pool_encoded(self, encoded: list[torch.Tensor]) -> torch.Tensor:
        if not encoded:
            raise ValueError("cannot pool an empty sequence list")
        lengths = torch.tensor([seq.shape[0] for seq in encoded], device=encoded[0].device)
        padded = pad_sequence(encoded, batch_first=True)
        mask = torch.arange(padded.shape[1], device=padded.device).unsqueeze(0) < lengths.unsqueeze(1)
        scores = torch.matmul(padded, self.pool_query)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(padded * weights.unsqueeze(-1), dim=1)

    def pool(self, sequences: list[torch.Tensor]) -> torch.Tensor:
        return self.pool_encoded(self.encode_sequences(sequences))

    def late_interaction_features(
        self,
        context_encoded: list[torch.Tensor],
        candidate_encoded: list[torch.Tensor],
        sample_indices: list[int],
    ) -> torch.Tensor:
        if self.interaction_mode != "late_maxsim":
            return torch.empty((len(candidate_encoded), 0), device=context_encoded[0].device)
        rows = []
        for cand_seq, sample_index in zip(candidate_encoded, sample_indices):
            ctx_seq = context_encoded[sample_index]
            cand_proj = F.normalize(self.interaction_proj(cand_seq), p=2, dim=-1)
            ctx_proj = F.normalize(self.interaction_proj(ctx_seq), p=2, dim=-1)
            sim = cand_proj @ ctx_proj.transpose(0, 1)
            cand_to_context = sim.max(dim=1).values
            context_to_cand = sim.max(dim=0).values
            rows.append(
                torch.stack(
                    [
                        cand_to_context.mean(),
                        cand_to_context.max(),
                        context_to_cand.mean(),
                        sim.mean(),
                    ]
                )
            )
        return torch.stack(rows, dim=0)

    def forward(
        self,
        context_latents: list[torch.Tensor],
        candidate_latents: list[list[torch.Tensor]],
        candidate_features: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        context_encoded = self.encode_sequences(context_latents)
        context_h = self.pool_encoded(context_encoded)
        flat_candidates: list[torch.Tensor] = []
        sample_indices: list[int] = []
        for sample_index, seqs in enumerate(candidate_latents):
            for seq in seqs:
                flat_candidates.append(seq)
                sample_indices.append(sample_index)
        candidate_encoded = self.encode_sequences(flat_candidates)
        candidate_h = self.pool_encoded(candidate_encoded)
        sample_index_tensor = torch.tensor(sample_indices, dtype=torch.long, device=context_h.device)
        context_for_candidate = context_h.index_select(0, sample_index_tensor)
        features = torch.cat(candidate_features, dim=0)
        feature_h = self.feature_encoder(features)
        interaction_features = self.late_interaction_features(context_encoded, candidate_encoded, sample_indices)
        joint = torch.cat(
            [
                context_for_candidate,
                candidate_h,
                context_for_candidate * candidate_h,
                torch.abs(context_for_candidate - candidate_h),
                feature_h,
                interaction_features,
            ],
            dim=-1,
        )
        logits = self.score_head(self.joint(joint)).squeeze(-1)
        outputs = []
        offset = 0
        for seqs in candidate_latents:
            count = len(seqs)
            outputs.append(logits[offset : offset + count])
            offset += count
        return outputs


def main() -> None:
    summary = train(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainConfig()
    for field, value in asdict(defaults).items():
        arg = "--" + field.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(arg, action="store_true", default=value)
        elif isinstance(value, int):
            parser.add_argument(arg, type=int, default=value)
        elif isinstance(value, float):
            parser.add_argument(arg, type=float, default=value)
        else:
            parser.add_argument(arg, default=value)
    return TrainConfig(**vars(parser.parse_args()))


def train(config: TrainConfig) -> dict[str, Any]:
    validate_config(config)
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    if config.cola_code_path and config.cola_code_path not in sys.path:
        sys.path.insert(0, config.cola_code_path)
    from tokenizers import Tokenizer
    from cola_dlm import ColaTextVAEModel

    train_rows = read_jsonl(Path(config.train_candidates_jsonl))
    eval_rows = read_jsonl(Path(config.eval_candidates_jsonl))
    if config.max_train_samples:
        train_rows = train_rows[: config.max_train_samples]
    if config.max_eval_samples:
        eval_rows = eval_rows[: config.max_eval_samples]
    train_source, valid_source = split_train_valid(train_rows, config)
    train_evidence = load_manifest_evidence(Path(config.train_manifest_json))
    eval_evidence = load_manifest_evidence(Path(config.eval_manifest_json))

    train_ds = CandidateRankDataset(
        rows=train_source,
        evidence_by_sample=train_evidence,
        max_candidates=config.max_candidates,
        require_positive=True,
    )
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
    if len(train_ds) == 0:
        raise ValueError("no train rows with primary-positive candidate labels")

    device = resolve_device(config.device)
    require_cuda_training(device, "train_p2_phase_a_cola_latent_candidate_ranker.py")
    tokenizer = Tokenizer.from_file(config.cola_tokenizer_path)
    vae = ColaTextVAEModel.from_pretrained(config.cola_vae_path).to(device).eval()
    for param in vae.parameters():
        param.requires_grad_(False)

    latent_dim = infer_latent_dim(train_ds[0], tokenizer, vae, device, config)
    feature_dim = len(
        candidate_feature_vector(
            train_ds[0]["candidates"][0],
            train_ds[0]["question"],
            schema_version=config.feature_schema_version,
        )
    )
    model = LatentCandidateRanker(
        latent_dim=latent_dim,
        feature_dim=feature_dim,
        hidden_dim=config.hidden_dim,
        interaction_dim=config.interaction_dim,
        interaction_mode=config.interaction_mode,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    run = init_experiment(
        stage="p2_phase_a",
        config={
            **asdict(config),
            "train_rows_source": len(train_source),
            "valid_rows_source": len(valid_source),
            "train_rows_used": len(train_ds),
            "valid_rows_used": len(valid_ds),
            "eval_rows_used": len(eval_ds),
            "device": device_metadata(device),
            "latent_dim": latent_dim,
            "feature_dim": feature_dim,
        },
        experiment_name=config.experiment_name,
        description="CoLA VAE latent candidate-answer ranker for MuSiQue Phase A",
        tags=["drla", "cola", "p2-phase-a", "latent-candidate-ranker", "musique"],
        mode=config.swanlab_mode,
    )
    best_metric = -1.0
    best_step = 0
    global_step = 0
    final_valid_metrics: dict[str, float] = {}
    final_eval_metrics: dict[str, float] = {}

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda rows: rows,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda rows: rows,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda rows: rows,
    )

    try:
        with metrics_path.open("w", encoding="utf-8") as metrics_f:
            stop = False
            for _epoch in range(config.epochs):
                for batch in train_loader:
                    model.train()
                    encoded = encode_batch(batch, tokenizer, vae, device, config)
                    logits_list = model(
                        encoded["context_latents"],
                        encoded["candidate_latents"],
                        encoded["candidate_features"],
                    )
                    loss = ranking_loss(logits_list, encoded["labels"])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if config.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                    optimizer.step()
                    global_step += 1
                    train_metrics = {"loss": float(loss.detach().item())}
                    write_metrics(metrics_f, "train", global_step, train_metrics)
                    log_metrics(train_metrics, step=global_step, prefix="train")
                    if global_step % config.valid_interval == 0:
                        valid_metrics, _ = evaluate(
                            valid_loader,
                            model,
                            tokenizer,
                            vae,
                            device,
                            config,
                            max_batches=config.max_valid_batches,
                        )
                        write_metrics(metrics_f, "valid", global_step, valid_metrics)
                        log_metrics(valid_metrics, step=global_step, prefix="valid")
                        current = valid_metrics["selected_primary"]
                        if current > best_metric or (
                            math.isclose(current, best_metric) and valid_metrics["loss"] < final_valid_metrics.get("loss", float("inf"))
                        ):
                            best_metric = current
                            best_step = global_step
                            save_checkpoint(
                                checkpoint_dir / "best_checkpoint.pt",
                                model,
                                optimizer,
                                config,
                                global_step,
                                best_metric,
                                latent_dim,
                                feature_dim,
                            )
                        final_valid_metrics = valid_metrics
                    if config.max_train_steps and global_step >= config.max_train_steps:
                        stop = True
                        break
                if stop:
                    break
            final_valid_metrics, valid_predictions = evaluate(
                valid_loader,
                model,
                tokenizer,
                vae,
                device,
                config,
                max_batches=0,
            )
            write_metrics(metrics_f, "valid", global_step, final_valid_metrics)
            log_metrics(final_valid_metrics, step=global_step, prefix="valid")
            if final_valid_metrics["selected_primary"] > best_metric:
                best_metric = final_valid_metrics["selected_primary"]
                best_step = global_step
                save_checkpoint(
                    checkpoint_dir / "best_checkpoint.pt",
                    model,
                    optimizer,
                    config,
                    global_step,
                    best_metric,
                    latent_dim,
                    feature_dim,
                )
            final_eval_metrics, eval_predictions = evaluate(
                eval_loader,
                model,
                tokenizer,
                vae,
                device,
                config,
                max_batches=0,
            )
            write_metrics(metrics_f, "eval", global_step, final_eval_metrics)
            log_metrics(final_eval_metrics, step=global_step, prefix="eval")
            last_full_valid_metrics = final_valid_metrics
            last_full_eval_metrics = final_eval_metrics
            last_valid_predictions = valid_predictions
            last_eval_predictions = eval_predictions
            save_checkpoint(
                checkpoint_dir / "last_checkpoint.pt",
                model,
                optimizer,
                config,
                global_step,
                final_valid_metrics["selected_primary"],
                latent_dim,
                feature_dim,
            )
            best_checkpoint = torch.load(checkpoint_dir / "best_checkpoint.pt", map_location=device)
            model.load_state_dict(best_checkpoint["model_state_dict"])
            best_full_valid_metrics, best_valid_predictions = evaluate(
                valid_loader,
                model,
                tokenizer,
                vae,
                device,
                config,
                max_batches=0,
            )
            write_metrics(metrics_f, "best_valid", global_step, best_full_valid_metrics)
            log_metrics(best_full_valid_metrics, step=global_step, prefix="best_valid")
            best_full_eval_metrics, best_eval_predictions = evaluate(
                eval_loader,
                model,
                tokenizer,
                vae,
                device,
                config,
                max_batches=0,
            )
            write_metrics(metrics_f, "best_eval", global_step, best_full_eval_metrics)
            log_metrics(best_full_eval_metrics, step=global_step, prefix="best_eval")
        write_jsonl(output_dir / "valid_predictions.jsonl", best_valid_predictions)
        write_jsonl(output_dir / "eval_predictions.jsonl", best_eval_predictions)
        write_jsonl(output_dir / "best_valid_predictions.jsonl", best_valid_predictions)
        write_jsonl(output_dir / "best_eval_predictions.jsonl", best_eval_predictions)
        write_jsonl(output_dir / "last_valid_predictions.jsonl", last_valid_predictions)
        write_jsonl(output_dir / "last_eval_predictions.jsonl", last_eval_predictions)
    finally:
        finish_experiment()

    summary = {
        "created_at": int(time.time()),
        "status": "pass",
        "swanlab_run_id": getattr(run, "id", None),
        "config": asdict(config),
        "dataset": {
            "train_source_rows": len(train_source),
            "valid_source_rows": len(valid_source),
            "eval_source_rows": len(eval_rows),
            "train_rows_used": len(train_ds),
            "valid_rows_used": len(valid_ds),
            "eval_rows_used": len(eval_ds),
            "train_skipped": dict(train_ds.skipped),
            "valid_skipped": dict(valid_ds.skipped),
            "eval_skipped": dict(eval_ds.skipped),
        },
        "model": {
            "latent_dim": latent_dim,
            "feature_dim": feature_dim,
            "hidden_dim": config.hidden_dim,
            "interaction_dim": config.interaction_dim,
            "interaction_mode": config.interaction_mode,
            "trainable_params": sum(param.numel() for param in model.parameters() if param.requires_grad),
        },
        "best_step": best_step,
        "best_valid_selected_primary": best_metric,
        "final_valid_metrics": best_full_valid_metrics,
        "final_eval_metrics": best_full_eval_metrics,
        "best_full_valid_metrics": best_full_valid_metrics,
        "best_full_eval_metrics": best_full_eval_metrics,
        "last_full_valid_metrics": last_full_valid_metrics,
        "last_full_eval_metrics": last_full_eval_metrics,
        "artifacts": {
            "metrics_jsonl": str(metrics_path),
            "best_checkpoint": str(checkpoint_dir / "best_checkpoint.pt"),
            "last_checkpoint": str(checkpoint_dir / "last_checkpoint.pt"),
            "valid_predictions": str(output_dir / "valid_predictions.jsonl"),
            "eval_predictions": str(output_dir / "eval_predictions.jsonl"),
            "best_valid_predictions": str(output_dir / "best_valid_predictions.jsonl"),
            "best_eval_predictions": str(output_dir / "best_eval_predictions.jsonl"),
            "last_valid_predictions": str(output_dir / "last_valid_predictions.jsonl"),
            "last_eval_predictions": str(output_dir / "last_eval_predictions.jsonl"),
        },
        "execution_boundary": [
            "deep-learning training with CUDA and SwanLab cloud",
            "official CoLA VAE is frozen and used only to encode online text into latent representations",
            "online model inputs exclude gold, aliases, teacher correctness, scorer outputs, and held-out data",
            "gold/scorer fields are offline supervision and evaluation only",
            "best checkpoint is selected on nonheldout valid selected_primary",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def validate_config(config: TrainConfig) -> None:
    if config.valid_interval > 10:
        raise ValueError("valid_interval must be <= 10 steps")
    if config.swanlab_mode != "cloud":
        raise ValueError("training must use SwanLab cloud; pass --swanlab-mode cloud")
    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if config.max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")
    if config.feature_schema_version not in {1, 2}:
        raise ValueError("feature_schema_version must be 1 or 2")
    if config.interaction_mode not in {"pooled", "late_maxsim"}:
        raise ValueError("interaction_mode must be pooled or late_maxsim")
    if config.interaction_dim < 1:
        raise ValueError("interaction_dim must be >= 1")
    if not 0.0 < config.valid_ratio < 0.5:
        raise ValueError("valid_ratio must be in (0, 0.5)")


def split_train_valid(rows: list[dict[str, Any]], config: TrainConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(rows)
    rng = random.Random(config.seed)
    rng.shuffle(rows)
    valid_count = max(1, int(round(len(rows) * config.valid_ratio)))
    return rows[valid_count:], rows[:valid_count]


def load_manifest_evidence(path: Path) -> dict[str, str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for sample in manifest.get("samples", []):
        result[str(sample["sample_id"])] = str(sample.get("metadata", {}).get("full_info_observation", ""))
    return result


def build_item(row: dict[str, Any], evidence_by_sample: dict[str, str], *, max_candidates: int) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    question = str(row.get("question", ""))
    gold = str(row.get("gold_answer", ""))
    aliases = [str(alias) for alias in row.get("answer_aliases", []) or []]
    candidates = list(row.get("candidates", []))[:max_candidates]
    labels = []
    label_scores = []
    for candidate in candidates:
        score = score_qa_answer(candidate.get("text", ""), gold, aliases)
        labels.append(float(score.primary_score))
        label_scores.append(score.to_dict())
    return {
        "sample_id": sample_id,
        "question": question,
        "context_text": make_context_text(question, evidence_by_sample.get(sample_id, "")),
        "gold_answer": gold,
        "answer_aliases": aliases,
        "candidates": candidates,
        "labels": labels,
        "label_scores": label_scores,
        "oracle_primary": float(max(labels) if labels else 0.0),
    }


def make_context_text(question: str, evidence: str) -> str:
    return (
        "Question:\n"
        f"{question.strip()}\n\n"
        "Evidence:\n"
        f"{evidence.strip()}\n\n"
        "Select the best final answer from the evidence-derived candidates."
    )


def make_candidate_text(candidate: dict[str, Any]) -> str:
    return (
        "Candidate:\n"
        f"{str(candidate.get('text', '')).strip()}\n"
        f"Source: {str(candidate.get('source_title', '')).strip()}\n"
        f"Rule: {str(candidate.get('rule', '')).strip()}\n"
        f"Evidence kind: {str(candidate.get('evidence_kind', '')).strip()}"
    )


def encode_batch(
    batch: list[dict[str, Any]],
    tokenizer: Any,
    vae: Any,
    device: torch.device,
    config: TrainConfig,
) -> dict[str, Any]:
    texts: list[str] = []
    layout: list[tuple[int, int]] = []
    for item in batch:
        start = len(texts)
        texts.append(item["context_text"])
        for candidate in item["candidates"]:
            texts.append(make_candidate_text(candidate))
        layout.append((start, len(item["candidates"])))
    input_ids_list = [
        torch.tensor(
            truncate_ids(
                tokenizer.encode(text).ids,
                config.max_context_tokens if index == start else config.max_candidate_tokens,
            ),
            dtype=torch.long,
            device=device,
        )
        for start, count in layout
        for index, text in enumerate(texts[start : start + count + 1], start=start)
    ]
    scale = vae.scaling_factor
    shift = vae.shifting_factor
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        enc = vae.encode(input_ids_list)
        latents = [((latent - shift) * scale).float().detach() for latent in enc.latents_list]
    context_latents: list[torch.Tensor] = []
    candidate_latents: list[list[torch.Tensor]] = []
    candidate_features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    offset = 0
    for item in batch:
        count = len(item["candidates"])
        context_latents.append(latents[offset])
        offset += 1
        candidate_latents.append(latents[offset : offset + count])
        offset += count
        feature_rows = [
            candidate_feature_vector(
                candidate,
                item["question"],
                schema_version=config.feature_schema_version,
            )
            for candidate in item["candidates"]
        ]
        candidate_features.append(torch.tensor(feature_rows, dtype=torch.float32, device=device))
        labels.append(torch.tensor(item["labels"], dtype=torch.float32, device=device))
    return {
        "context_latents": context_latents,
        "candidate_latents": candidate_latents,
        "candidate_features": candidate_features,
        "labels": labels,
    }


def truncate_ids(ids: list[int], max_tokens: int) -> list[int]:
    if not ids:
        return [0]
    if max_tokens and len(ids) > max_tokens:
        return ids[:max_tokens]
    return ids


def infer_latent_dim(
    item: dict[str, Any],
    tokenizer: Any,
    vae: Any,
    device: torch.device,
    config: TrainConfig,
) -> int:
    encoded = encode_batch([item], tokenizer, vae, device, config)
    return int(encoded["context_latents"][0].shape[-1])


def ranking_loss(logits_list: list[torch.Tensor], labels_list: list[torch.Tensor]) -> torch.Tensor:
    losses = []
    for logits, labels in zip(logits_list, labels_list):
        positives = labels > 0.5
        if not bool(positives.any()):
            continue
        losses.append(torch.logsumexp(logits, dim=0) - torch.logsumexp(logits[positives], dim=0))
    if not losses:
        raise ValueError("batch contains no primary-positive candidate labels")
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(
    loader: DataLoader,
    model: LatentCandidateRanker,
    tokenizer: Any,
    vae: Any,
    device: torch.device,
    config: TrainConfig,
    *,
    max_batches: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    losses = []
    predictions = []
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        encoded = encode_batch(batch, tokenizer, vae, device, config)
        logits_list = model(
            encoded["context_latents"],
            encoded["candidate_latents"],
            encoded["candidate_features"],
        )
        try:
            losses.append(float(ranking_loss(logits_list, encoded["labels"]).detach().item()))
        except ValueError:
            pass
        for item, logits in zip(batch, logits_list):
            pred_index = int(torch.argmax(logits).item())
            candidate = item["candidates"][pred_index]
            score = score_qa_answer(candidate.get("text", ""), item["gold_answer"], item["answer_aliases"])
            predictions.append(
                {
                    "sample_id": item["sample_id"],
                    "question": item["question"],
                    "prediction": candidate.get("text", ""),
                    "candidate_rank": candidate.get("rank"),
                    "candidate_rule": candidate.get("rule"),
                    "candidate_source_title": candidate.get("source_title"),
                    "gold_answer": item["gold_answer"],
                    "primary_score": score.primary_score,
                    "exact_match": score.exact_match,
                    "token_f1": score.token_f1,
                    "oracle_primary": item["oracle_primary"],
                    "selected_label": item["labels"][pred_index],
                    "num_candidates": len(item["candidates"]),
                }
            )
    metrics = summarize_predictions(predictions, losses)
    model.train()
    return metrics, predictions


def summarize_predictions(predictions: list[dict[str, Any]], losses: list[float]) -> dict[str, float]:
    n = max(1, len(predictions))
    return {
        "loss": float(sum(losses) / len(losses)) if losses else float("inf"),
        "num_samples": float(len(predictions)),
        "selected_primary": sum(float(row["primary_score"]) for row in predictions) / n,
        "selected_exact": sum(float(row["exact_match"]) for row in predictions) / n,
        "selected_token_f1": sum(float(row["token_f1"]) for row in predictions) / n,
        "oracle_primary": sum(float(row["oracle_primary"]) for row in predictions) / n,
        "selected_positive_label": sum(float(row["selected_label"]) for row in predictions) / n,
    }


def candidate_feature_vector(candidate: dict[str, Any], question: str, *, schema_version: int = 2) -> list[float]:
    text = str(candidate.get("text", ""))
    rank = max(1, int(candidate.get("rank", 999)))
    rule = str(candidate.get("rule", "other"))
    rules = RULES_V1 if schema_version == 1 else RULES_V2
    if rule not in rules:
        rule = "other"
    qtype = question_type(question)
    evidence_kind = str(candidate.get("evidence_kind", "unknown"))
    if evidence_kind not in EVIDENCE_KINDS:
        evidence_kind = "unknown"
    if schema_version == 1:
        token_count = max(1, len(text.split()))
        values = [
            1.0 / rank,
            math.log(rank + 1.0),
            math.log(float(candidate.get("occurrences", 1)) + 1.0),
            math.log(len(text) + 1.0),
            float(token_count),
            float(bool(candidate.get("has_support_occurrence"))),
            float(any(char.isdigit() for char in text)),
            float(token_count <= 4),
        ]
        values.extend(float(rule == item) for item in rules)
        values.extend(float(qtype == item) for item in QTYPES)
        values.extend(float(evidence_kind == item) for item in EVIDENCE_KINDS)
        return values
    candidate_tokens = token_set(text)
    question_tokens = token_set(question)
    source_title_tokens = token_set(candidate.get("source_title", ""))
    overlap = candidate_tokens & question_tokens
    token_count = max(1, len(candidate_tokens))
    is_numeric = any(char.isdigit() for char in text)
    is_date_like = rule in {"date_phrase", "century_phrase", "season_phrase", "season_year_phrase"}
    is_entity_like = rule in {
        "title",
        "capitalized_full_span",
        "capitalized_subspan",
        "capitalized_single",
        "quoted_span",
    }
    values = [
        1.0 / rank,
        math.log(rank + 1.0),
        math.log(float(candidate.get("occurrences", 1)) + 1.0),
        math.log(len(text) + 1.0),
        float(token_count),
        float(bool(candidate.get("has_support_occurrence"))),
        float(is_numeric),
        float(token_count <= 4),
        float(len(overlap)),
        float(len(overlap) / token_count),
        float(bool(candidate_tokens and candidate_tokens <= question_tokens)),
        float(len(source_title_tokens & question_tokens)),
        float(qtype in {"when", "how_many"} and is_numeric),
        float(qtype == "when" and is_date_like),
        float(qtype == "how_many" and (is_numeric or rule in {"quantity_phrase", "number_word"})),
        float(qtype in {"who", "where", "what", "which"} and is_entity_like),
    ]
    values.extend(float(rule == item) for item in rules)
    values.extend(float(qtype == item) for item in QTYPES)
    values.extend(float(evidence_kind == item) for item in EVIDENCE_KINDS)
    return values


def token_set(text: Any) -> set[str]:
    return {
        token
        for token in "".join(char.lower() if char.isalnum() else " " for char in str(text)).split()
        if token
    }


def question_type(question: str) -> str:
    q = question.lower().strip()
    if q.startswith("who"):
        return "who"
    if q.startswith("where"):
        return "where"
    if q.startswith("when"):
        return "when"
    if q.startswith("which"):
        return "which"
    if q.startswith("how many") or q.startswith("how much"):
        return "how_many"
    if " language " in f" {q} ":
        return "language"
    if q.startswith("what"):
        return "what"
    return "other"


def save_checkpoint(
    path: Path,
    model: LatentCandidateRanker,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    step: int,
    metric: float,
    latent_dim: int,
    feature_dim: int,
) -> None:
    torch.save(
        {
            "config": asdict(config),
            "global_step": step,
            "selection_metric": metric,
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "latent_dim": latent_dim,
            "feature_dim": feature_dim,
            "interaction_dim": config.interaction_dim,
            "interaction_mode": config.interaction_mode,
        },
        path,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_metrics(metrics_f: Any, split: str, step: int, metrics: dict[str, float]) -> None:
    row = {"split": split, "step": step, **metrics}
    metrics_f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metrics_f.flush()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
