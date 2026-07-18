"""Evaluate D5 Dream readiness student as an online halt policy.

This local-only script loads a trained ``DreamStepReadinessStudent`` checkpoint
and an offline D4 frontier, chooses halt thresholds on the validation split
only, then reports locked test behavior. It does not train, tune on held-out,
start SwanLab, or use decoded/gold/scorer fields as online model inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.scripts.p3_train_dream_readiness_student import (  # noqa: E402
    FEATURE_NAMES,
    DreamStepReadinessStudent,
    TrainConfig,
    event_features,
    read_jsonl,
    resolve_device,
    split_by_sample,
)


DEFAULT_FRONTIER_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_readiness_frontiers/"
    "musique_calibration_trace_full200_steps64_stride4_hidden_summary_frontier_with_hidden_20260606"
)
DEFAULT_CHECKPOINT = (
    "/data1/luyifei/drla/outputs/p3_dream_readiness_students/"
    "dream_step_readiness_student_v1_full200_with_hidden_seed20260606_20260606/best_checkpoint.pt"
)
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_readiness_policy_eval/"
    "dream_step_readiness_student_v1_full200_with_hidden_policy_eval_20260606"
)


@dataclass(frozen=True)
class EvalConfig:
    frontier_dir: str = DEFAULT_FRONTIER_DIR
    checkpoint: str = DEFAULT_CHECKPOINT
    output_dir: str = DEFAULT_OUTPUT_DIR
    device: str = "auto"
    risk_accuracy_drop_cap: float = 0.02
    bootstrap_samples: int = 1000
    ready_threshold_min: float = 0.05
    ready_threshold_max: float = 0.95
    ready_threshold_step: float = 0.05
    final_match_thresholds: str = "0.0,0.5,0.7,0.9"
    prediction_change_max_values: str = "1.0,0.5,0.3,0.1"
    future_gain_max_values: str = "999.0,0.10,0.05,0.0,-0.05"
    overwrite: bool = False


def main() -> None:
    summary = evaluate_policy(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-dir", default=EvalConfig.frontier_dir)
    parser.add_argument("--checkpoint", default=EvalConfig.checkpoint)
    parser.add_argument("--output-dir", default=EvalConfig.output_dir)
    parser.add_argument("--device", default=EvalConfig.device)
    parser.add_argument("--risk-accuracy-drop-cap", type=float, default=EvalConfig.risk_accuracy_drop_cap)
    parser.add_argument("--bootstrap-samples", type=int, default=EvalConfig.bootstrap_samples)
    parser.add_argument("--ready-threshold-min", type=float, default=EvalConfig.ready_threshold_min)
    parser.add_argument("--ready-threshold-max", type=float, default=EvalConfig.ready_threshold_max)
    parser.add_argument("--ready-threshold-step", type=float, default=EvalConfig.ready_threshold_step)
    parser.add_argument("--final-match-thresholds", default=EvalConfig.final_match_thresholds)
    parser.add_argument("--prediction-change-max-values", default=EvalConfig.prediction_change_max_values)
    parser.add_argument("--future-gain-max-values", default=EvalConfig.future_gain_max_values)
    parser.add_argument("--overwrite", action="store_true")
    return EvalConfig(**vars(parser.parse_args()))


def evaluate_policy(config: EvalConfig) -> dict[str, Any]:
    created_at = int(time.time())
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(config.checkpoint, map_location="cpu")
    train_config = TrainConfig(**checkpoint["config"])
    device = resolve_device(config.device)
    model = DreamStepReadinessStudent(
        feature_dim=len(FEATURE_NAMES),
        d_model=train_config.d_model,
        num_layers=train_config.num_layers,
        num_heads=train_config.num_heads,
        dropout=train_config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    sequences, frontier_summary = load_frontier_sequences(Path(config.frontier_dir))
    splits = split_by_sample(
        sequences,
        seed=train_config.seed,
        train_ratio=train_config.train_ratio,
        valid_ratio=train_config.valid_ratio,
    )
    feature_stats = checkpoint["feature_stats"]
    predictions = {
        split_name: predict_sequences(model, split_sequences, feature_stats, device)
        for split_name, split_sequences in splits.items()
    }
    grid = build_policy_grid(config)
    valid_grid_rows = [evaluate_policy_on_sequences(predictions["valid"], policy) for policy in grid]
    selected_policy_row = select_policy(valid_grid_rows, config.risk_accuracy_drop_cap)
    selected_policy = selected_policy_row["policy"]
    selected_metrics = {
        split_name: evaluate_policy_on_sequences(
            split_predictions,
            selected_policy,
            bootstrap_samples=config.bootstrap_samples,
            bootstrap_seed=created_at + split_index,
        )
        for split_index, (split_name, split_predictions) in enumerate(predictions.items())
    }
    grid_selected_metrics = {
        split_name: strip_bootstrap(metric)
        for split_name, split_predictions in predictions.items()
        for metric in [selected_metrics[split_name]]
    }
    calibration_metrics = {
        split_name: calibration_report(split_predictions)
        for split_name, split_predictions in predictions.items()
    }

    write_jsonl(output_dir / "valid_policy_grid.jsonl", valid_grid_rows)
    decision_rows = []
    for split_name, split_predictions in predictions.items():
        decision_rows.extend(policy_decision_rows(split_name, split_predictions, selected_policy))
    write_jsonl(output_dir / "policy_decisions.jsonl", decision_rows)
    event_rows = []
    for split_name, split_predictions in predictions.items():
        event_rows.extend(event_prediction_rows(split_name, split_predictions))
    write_jsonl(output_dir / "event_predictions.jsonl", event_rows)
    metrics = {
        "selected_policy_valid": selected_policy_row,
        "selected_policy_by_split": selected_metrics,
        "selected_policy_by_split_without_bootstrap": grid_selected_metrics,
        "calibration_by_split": calibration_metrics,
    }
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "created_at": created_at,
        "status": "pass",
        "config": asdict(config),
        "frontier_summary": frontier_summary,
        "checkpoint": str(config.checkpoint),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_selection_metric": checkpoint.get("selection_metric"),
        "feature_names": FEATURE_NAMES,
        "split_sizes": {name: len(items) for name, items in splits.items()},
        "num_valid_policy_candidates": len(valid_grid_rows),
        "selected_policy": selected_policy,
        "metrics": metrics,
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "metrics_jsonl": str(metrics_path),
            "valid_policy_grid_jsonl": str(output_dir / "valid_policy_grid.jsonl"),
            "policy_decisions_jsonl": str(output_dir / "policy_decisions.jsonl"),
            "event_predictions_jsonl": str(output_dir / "event_predictions.jsonl"),
        },
        "execution_boundary": [
            "local-only P3 D5 online halt policy evaluation",
            "no optimizer, backward, or weight update",
            "no SwanLab run",
            "thresholds selected on validation split only",
            "test split is report-only",
        ],
        "online_policy_inputs": [
            "ready probability from D5 student",
            "final_match probability from D5 student",
            "prediction_change probability from D5 student",
            "future_gain prediction from D5 student",
        ],
        "forbidden_online_inputs": [
            "gold_answer",
            "answer_aliases",
            "decoded step text",
            "step score",
            "final score",
            "oracle frontier labels",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def load_frontier_sequences(frontier_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = read_jsonl(frontier_dir / "frontier_events.jsonl")
    summary = json.loads((frontier_dir / "summary.json").read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[str(event["row_id"])].append(event)
    sequences = []
    for row_id, row_events in grouped.items():
        row_events = sorted(row_events, key=lambda item: int(item.get("trace_event_index", 0)))
        first = row_events[0]
        final_event = row_events[-1]
        sequences.append(
            {
                "row_id": row_id,
                "sample_id": first.get("sample_id", ""),
                "condition": first.get("condition", ""),
                "events": row_events,
                "features": [event_features(event) for event in row_events],
                "ready": [float(event.get("answer_ready_correct_and_final_stable", False)) for event in row_events],
                "final_primary_score": float(final_event.get("final_primary_score", 0.0)),
                "final_step": float(final_event.get("step") or 0.0),
                "final_event_index": int(final_event.get("trace_event_index", len(row_events) - 1)),
            }
        )
    metadata = {
        "frontier_dir": str(frontier_dir),
        "num_sequences": len(sequences),
        "num_events": len(events),
        "condition_counts": dict(Counter(str(item["condition"]) for item in sequences)),
        "frontier_status": summary.get("status"),
        "frontier_metrics": summary.get("metrics", {}),
    }
    return sequences, metadata


@torch.no_grad()
def predict_sequences(
    model: DreamStepReadinessStudent,
    sequences: list[dict[str, Any]],
    feature_stats: dict[str, list[float]],
    device: torch.device,
) -> list[dict[str, Any]]:
    mean = torch.tensor(feature_stats["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(feature_stats["std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    predictions = []
    for sequence in sequences:
        features = torch.tensor(sequence["features"], dtype=torch.float32, device=device)
        features = ((features - mean) / std).unsqueeze(0)
        padding_mask = torch.zeros((1, features.shape[1]), dtype=torch.bool, device=device)
        outputs = model(features, padding_mask)
        ready_prob = torch.sigmoid(outputs["ready_logit"])[0].detach().cpu().tolist()
        final_match_prob = torch.sigmoid(outputs["final_match_logit"])[0].detach().cpu().tolist()
        prediction_change_prob = torch.sigmoid(outputs["prediction_change_logit"])[0].detach().cpu().tolist()
        future_gain_pred = outputs["future_gain"][0].detach().cpu().tolist()
        event_predictions = []
        for index, event in enumerate(sequence["events"]):
            is_final_event = index == len(sequence["events"]) - 1
            step_score = float(event.get("step_primary_score", 0.0))
            selected_score = sequence["final_primary_score"] if is_final_event else step_score
            event_predictions.append(
                {
                    "row_id": sequence["row_id"],
                    "sample_id": sequence["sample_id"],
                    "condition": sequence["condition"],
                    "event_index": index,
                    "trace_event_index": int(event.get("trace_event_index", index)),
                    "step": float(event.get("step") or 0.0),
                    "ready_prob": float(ready_prob[index]),
                    "final_match_prob": float(final_match_prob[index]),
                    "prediction_change_prob": float(prediction_change_prob[index]),
                    "future_gain_pred": float(future_gain_pred[index]),
                    "ready_label": float(event.get("answer_ready_correct_and_final_stable", False)),
                    "step_primary_score": step_score,
                    "selected_primary_score": selected_score,
                    "is_final_event": is_final_event,
                }
            )
        predictions.append(
            {
                "row_id": sequence["row_id"],
                "sample_id": sequence["sample_id"],
                "condition": sequence["condition"],
                "events": event_predictions,
                "final_primary_score": sequence["final_primary_score"],
                "final_step": sequence["final_step"],
                "final_event_index": sequence["final_event_index"],
            }
        )
    return predictions


def build_policy_grid(config: EvalConfig) -> list[dict[str, float]]:
    ready_thresholds = float_range(config.ready_threshold_min, config.ready_threshold_max, config.ready_threshold_step)
    final_match_thresholds = parse_float_list(config.final_match_thresholds)
    prediction_change_max_values = parse_float_list(config.prediction_change_max_values)
    future_gain_max_values = parse_float_list(config.future_gain_max_values)
    policies = []
    for ready_threshold in ready_thresholds:
        for final_match_threshold in final_match_thresholds:
            for prediction_change_max in prediction_change_max_values:
                for future_gain_max in future_gain_max_values:
                    policies.append(
                        {
                            "ready_threshold": ready_threshold,
                            "final_match_threshold": final_match_threshold,
                            "prediction_change_max": prediction_change_max,
                            "future_gain_max": future_gain_max,
                        }
                    )
    return policies


def evaluate_policy_on_sequences(
    predictions: list[dict[str, Any]],
    policy: dict[str, float],
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    decisions = [select_event(sequence, policy) for sequence in predictions]
    final_scores = [float(sequence["final_primary_score"]) for sequence in predictions]
    selected_scores = [float(decision["selected_primary_score"]) for decision in decisions]
    final_steps = [float(sequence["final_step"]) for sequence in predictions]
    selected_steps = [float(decision["step"]) for decision in decisions]
    final_indices = [float(sequence["final_event_index"]) for sequence in predictions]
    selected_indices = [float(decision["trace_event_index"]) for decision in decisions]
    selected_accuracy = mean(selected_scores)
    final_accuracy = mean(final_scores)
    row = {
        "policy": policy,
        "num_rows": len(predictions),
        "final_accuracy": final_accuracy,
        "selected_accuracy": selected_accuracy,
        "accuracy_drop_vs_final": final_accuracy - selected_accuracy,
        "selected_accuracy_minus_final": selected_accuracy - final_accuracy,
        "mean_final_step": mean(final_steps),
        "mean_selected_step": mean(selected_steps),
        "mean_step_savings": mean([f - s for f, s in zip(final_steps, selected_steps)]),
        "mean_step_savings_rate": safe_div(mean([f - s for f, s in zip(final_steps, selected_steps)]), mean(final_steps)),
        "mean_final_event_index": mean(final_indices),
        "mean_selected_event_index": mean(selected_indices),
        "mean_event_savings": mean([f - s for f, s in zip(final_indices, selected_indices)]),
        "halt_before_final_rate": mean([float(not decision["is_final_event"]) for decision in decisions]),
    }
    row["per_condition"] = per_condition_metrics(predictions, decisions)
    if bootstrap_samples > 0:
        row["paired_bootstrap_ci"] = paired_bootstrap_ci(predictions, decisions, bootstrap_samples, bootstrap_seed)
    return row


def select_policy(valid_rows: list[dict[str, Any]], risk_accuracy_drop_cap: float) -> dict[str, Any]:
    for row in valid_rows:
        row["risk_cap_satisfied"] = row["accuracy_drop_vs_final"] <= risk_accuracy_drop_cap
    feasible = [row for row in valid_rows if row["risk_cap_satisfied"]]
    if feasible:
        selected = max(
            feasible,
            key=lambda row: (
                row["mean_step_savings"],
                row["selected_accuracy"],
                -row["accuracy_drop_vs_final"],
                row["halt_before_final_rate"],
            ),
        )
        selected["selection_rule"] = "max_valid_step_savings_under_accuracy_drop_cap"
        return selected
    selected = min(
        valid_rows,
        key=lambda row: (
            max(0.0, row["accuracy_drop_vs_final"] - risk_accuracy_drop_cap),
            row["accuracy_drop_vs_final"],
            -row["mean_step_savings"],
        ),
    )
    selected["selection_rule"] = "no_policy_met_cap_minimize_valid_risk_then_max_savings"
    return selected


def select_event(sequence: dict[str, Any], policy: dict[str, float]) -> dict[str, Any]:
    for event in sequence["events"]:
        if (
            event["ready_prob"] >= policy["ready_threshold"]
            and event["final_match_prob"] >= policy["final_match_threshold"]
            and event["prediction_change_prob"] <= policy["prediction_change_max"]
            and event["future_gain_pred"] <= policy["future_gain_max"]
        ):
            return event
    return sequence["events"][-1]


def per_condition_metrics(predictions: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result = {}
    conditions = sorted({str(sequence["condition"]) for sequence in predictions})
    for condition in conditions:
        indices = [idx for idx, sequence in enumerate(predictions) if sequence["condition"] == condition]
        final_scores = [float(predictions[idx]["final_primary_score"]) for idx in indices]
        selected_scores = [float(decisions[idx]["selected_primary_score"]) for idx in indices]
        final_steps = [float(predictions[idx]["final_step"]) for idx in indices]
        selected_steps = [float(decisions[idx]["step"]) for idx in indices]
        result[condition] = {
            "num_rows": float(len(indices)),
            "final_accuracy": mean(final_scores),
            "selected_accuracy": mean(selected_scores),
            "accuracy_drop_vs_final": mean(final_scores) - mean(selected_scores),
            "mean_step_savings": mean([f - s for f, s in zip(final_steps, selected_steps)]),
            "halt_before_final_rate": mean([float(not decisions[idx]["is_final_event"]) for idx in indices]),
        }
    return result


def paired_bootstrap_ci(
    predictions: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    num_samples: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = random.Random(seed)
    n = len(predictions)
    if n == 0:
        return {}
    accuracy_drop = []
    step_savings = []
    halt_rate = []
    for _ in range(num_samples):
        indices = [rng.randrange(n) for _ in range(n)]
        final_scores = [float(predictions[idx]["final_primary_score"]) for idx in indices]
        selected_scores = [float(decisions[idx]["selected_primary_score"]) for idx in indices]
        final_steps = [float(predictions[idx]["final_step"]) for idx in indices]
        selected_steps = [float(decisions[idx]["step"]) for idx in indices]
        accuracy_drop.append(mean(final_scores) - mean(selected_scores))
        step_savings.append(mean([f - s for f, s in zip(final_steps, selected_steps)]))
        halt_rate.append(mean([float(not decisions[idx]["is_final_event"]) for idx in indices]))
    return {
        "accuracy_drop_vs_final_95ci": quantile_interval(accuracy_drop),
        "mean_step_savings_95ci": quantile_interval(step_savings),
        "halt_before_final_rate_95ci": quantile_interval(halt_rate),
    }


def quantile_interval(values: list[float], lo: float = 0.025, hi: float = 0.975) -> list[float]:
    ordered = sorted(values)
    return [ordered[int(lo * (len(ordered) - 1))], ordered[int(hi * (len(ordered) - 1))]]


def strip_bootstrap(row: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(row)
    cleaned.pop("paired_bootstrap_ci", None)
    return cleaned


def calibration_report(predictions: list[dict[str, Any]], num_bins: int = 10) -> dict[str, float]:
    scores = []
    labels = []
    for sequence in predictions:
        for event in sequence["events"]:
            scores.append(float(event["ready_prob"]))
            labels.append(float(event["ready_label"]))
    return {
        "ready_brier": mean([(score - label) ** 2 for score, label in zip(scores, labels)]),
        "ready_ece_10bin": expected_calibration_error(scores, labels, num_bins),
        "ready_positive_rate": mean(labels),
        "ready_mean_prob": mean(scores),
        "num_events": float(len(labels)),
    }


def policy_decision_rows(
    split_name: str,
    predictions: list[dict[str, Any]],
    policy: dict[str, float],
) -> list[dict[str, Any]]:
    rows = []
    for sequence in predictions:
        decision = select_event(sequence, policy)
        rows.append(
            {
                "split": split_name,
                "row_id": sequence["row_id"],
                "sample_id": sequence["sample_id"],
                "condition": sequence["condition"],
                "final_primary_score": sequence["final_primary_score"],
                "selected_primary_score": decision["selected_primary_score"],
                "final_step": sequence["final_step"],
                "selected_step": decision["step"],
                "final_event_index": sequence["final_event_index"],
                "selected_trace_event_index": decision["trace_event_index"],
                "halt_before_final": not decision["is_final_event"],
                "ready_prob": decision["ready_prob"],
                "final_match_prob": decision["final_match_prob"],
                "prediction_change_prob": decision["prediction_change_prob"],
                "future_gain_pred": decision["future_gain_pred"],
            }
        )
    return rows


def event_prediction_rows(split_name: str, predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for sequence in predictions:
        for event in sequence["events"]:
            rows.append({"split": split_name, **event})
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def expected_calibration_error(scores: list[float], labels: list[float], num_bins: int) -> float:
    if not scores:
        return 0.0
    ece = 0.0
    for bin_index in range(num_bins):
        lo = bin_index / num_bins
        hi = (bin_index + 1) / num_bins
        if bin_index == num_bins - 1:
            in_bin = [idx for idx, score in enumerate(scores) if lo <= score <= hi]
        else:
            in_bin = [idx for idx, score in enumerate(scores) if lo <= score < hi]
        if not in_bin:
            continue
        confidence = mean([scores[idx] for idx in in_bin])
        accuracy = mean([labels[idx] for idx in in_bin])
        ece += (len(in_bin) / len(scores)) * abs(confidence - accuracy)
    return ece


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def float_range(start: float, stop: float, step: float) -> list[float]:
    values = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 10))
        current += step
    return values


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if abs(denominator) > 1e-12 else 0.0


if __name__ == "__main__":
    main()
