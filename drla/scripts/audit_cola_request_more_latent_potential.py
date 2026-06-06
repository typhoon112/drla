"""Audit P2-E request-more-latent potential on locked sender groups.

This local-only audit asks a different question from sender reranking:

* If the receiver starts with one latent sender packet, how much value could
  additional sender evidence add?
* Can decoder-free online signals decide when to request more evidence?

Decoded answers and official scores are offline labels only. Online request
signals are restricted to readiness/certificate-style packet scores and
train-split task/global priors.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from drla.scripts.audit_cola_hierarchical_aggregation_potential import (
    build_groups,
    read_jsonl,
    score_value,
)
from drla.scripts.audit_cola_sequential_latent_mas import load_official_scorer
from drla.scripts.train_cola_hierarchical_latent_fuser import split_groups
from drla.tracking import require_swanlab_disabled_for_non_training


REQUEST_SIGNALS: dict[str, str] = {
    "readiness": "low",
    "correctness": "low",
    "answer_identity_stability": "low",
    "contentful": "low",
    "completion_risk": "high",
    "future_gain": "high",
    "prediction_change": "high",
    "train_task_helpful_rate": "high",
    "train_task_gain_mean": "high",
    "train_global_helpful_rate": "high",
    "train_global_gain_mean": "high",
}


@dataclass(frozen=True)
class RequestMoreLatentPotentialConfig:
    packets_jsonl: str = (
        "/data1/luyifei/drla/outputs/cola_agent_latent_comm/"
        "p2_agent_latent_comm_v2_locked_seed66_67_68_split20260601_20260529/"
        "agent_latent_comm_packets_test.jsonl"
    )
    output_dir: str = (
        "/data1/luyifei/drla/outputs/cola_request_more_latent/"
        "p2e_request_more_latent_potential_v1"
    )
    data_root: str = "/data1/luyifei/Cola-DLM/code/generate_task_data"
    acc_calc_script: str = "/data1/luyifei/Cola-DLM/code/scripts/acc_calc.py"
    seed: int = 20260529
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    max_groups: int = 0
    target_request_rates: str = "0.10,0.25,0.50"
    target_helpful_precisions: str = "0.50,0.60,0.70"
    swanlab_mode: str = "disabled"


def main() -> None:
    summary = audit_request_more_latent(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> RequestMoreLatentPotentialConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets-jsonl", default=RequestMoreLatentPotentialConfig.packets_jsonl)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default=RequestMoreLatentPotentialConfig.data_root)
    parser.add_argument("--acc-calc-script", default=RequestMoreLatentPotentialConfig.acc_calc_script)
    parser.add_argument("--seed", type=int, default=RequestMoreLatentPotentialConfig.seed)
    parser.add_argument("--train-ratio", type=float, default=RequestMoreLatentPotentialConfig.train_ratio)
    parser.add_argument("--valid-ratio", type=float, default=RequestMoreLatentPotentialConfig.valid_ratio)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--target-request-rates", default=RequestMoreLatentPotentialConfig.target_request_rates)
    parser.add_argument("--target-helpful-precisions", default=RequestMoreLatentPotentialConfig.target_helpful_precisions)
    parser.add_argument("--swanlab-mode", default=RequestMoreLatentPotentialConfig.swanlab_mode)
    args = parser.parse_args()
    return RequestMoreLatentPotentialConfig(
        packets_jsonl=args.packets_jsonl,
        output_dir=args.output_dir,
        data_root=args.data_root,
        acc_calc_script=args.acc_calc_script,
        seed=args.seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        max_groups=args.max_groups,
        target_request_rates=args.target_request_rates,
        target_helpful_precisions=args.target_helpful_precisions,
        swanlab_mode=args.swanlab_mode,
    )


def audit_request_more_latent(config: RequestMoreLatentPotentialConfig) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        config.swanlab_mode,
        script_kind="P2-E request-more-latent potential audit",
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scorer = load_official_scorer(Path(config.acc_calc_script))
    groups = build_groups(read_jsonl(Path(config.packets_jsonl)), config, scorer)
    if config.max_groups:
        groups = groups[: config.max_groups]
    splits = split_groups(groups, config)
    rows_by_split = {split: [group_to_row(groups[index], split) for index in indices] for split, indices in splits.items()}
    priors = fit_train_priors(rows_by_split["train"])
    for split in ["train", "valid", "test"]:
        attach_priors(rows_by_split[split], priors)

    prefix_metrics = {split: aggregate_prefix_metrics(rows) for split, rows in rows_by_split.items()}
    request_rates = parse_float_list(config.target_request_rates)
    helpful_precisions = parse_float_list(config.target_helpful_precisions)
    policy_rows = evaluate_request_policies(
        valid_rows=rows_by_split["valid"],
        test_rows=rows_by_split["test"],
        request_rates=request_rates,
        helpful_precisions=helpful_precisions,
    )
    per_task = aggregate_by_task(rows_by_split["test"])
    artifacts = write_outputs(output_dir, config, rows_by_split, prefix_metrics, policy_rows, per_task, priors)
    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "split_sizes": {split: len(indices) for split, indices in splits.items()},
        "prefix_metrics": prefix_metrics,
        "policy_metrics": policy_rows,
        "test_per_task": per_task,
        "priors": priors,
        "artifacts": artifacts,
        "interpretation": (
            "Local-only request-more-latent potential audit. Prefix/oracle rows "
            "use offline labels to estimate additional-evidence upper bounds; "
            "request policy thresholds are selected only on valid and reported "
            "on held-out test."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def group_to_row(group: dict[str, Any], split: str) -> dict[str, Any]:
    members = group["members"]
    if len(members) < 3:
        raise ValueError("request-more audit expects at least three locked senders per group")
    first = members[0]
    prefix2 = members[:2]
    prefix3 = members[:3]
    best2 = best_member(prefix2)
    best3 = best_member(prefix3)
    readiness2 = choose_member_by_score(prefix2, "readiness", reverse=True)
    readiness3 = choose_member_by_score(prefix3, "readiness", reverse=True)
    row = {
        "split": split,
        "task": group["task"],
        "sample_key": group["sample_key"],
        "sample_id": group["sample_id"],
        "first_correct": float(bool(first["selected_correct"])),
        "first_score": float(first["selected_score"]),
        "prefix2_oracle_correct": float(bool(best2["selected_correct"])),
        "prefix2_oracle_score": float(best2["selected_score"]),
        "prefix3_oracle_correct": float(bool(best3["selected_correct"])),
        "prefix3_oracle_score": float(best3["selected_score"]),
        "prefix2_readiness_correct": float(bool(readiness2["selected_correct"])),
        "prefix2_readiness_score": float(readiness2["selected_score"]),
        "prefix3_readiness_correct": float(bool(readiness3["selected_correct"])),
        "prefix3_readiness_score": float(readiness3["selected_score"]),
        "first_to_prefix2_gain": float(best2["selected_score"]) - float(first["selected_score"]),
        "first_to_prefix3_gain": float(best3["selected_score"]) - float(first["selected_score"]),
        "prefix2_to_prefix3_gain": float(best3["selected_score"]) - float(best2["selected_score"]),
        "first_to_prefix2_exact_gain": float(bool(best2["selected_correct"])) - float(bool(first["selected_correct"])),
        "first_to_prefix3_exact_gain": float(bool(best3["selected_correct"])) - float(bool(first["selected_correct"])),
        "prefix2_to_prefix3_exact_gain": float(bool(best3["selected_correct"])) - float(bool(best2["selected_correct"])),
        "request_to_2_helpful": float(float(best2["selected_score"]) > float(first["selected_score"]) + 1e-12),
        "request_to_3_helpful": float(float(best3["selected_score"]) > float(first["selected_score"]) + 1e-12),
        "request_2_to_3_helpful": float(float(best3["selected_score"]) > float(best2["selected_score"]) + 1e-12),
        "num_correct_senders": float(sum(float(bool(member["selected_correct"])) for member in prefix3)),
        "num_unique_predictions": float(len({str(member["selected_prediction"]).strip().lower() for member in prefix3})),
    }
    for signal in REQUEST_SIGNALS:
        if signal.startswith("train_"):
            continue
        row[signal] = float(score_value(first, signal))
    return row


def best_member(members: list[dict[str, Any]]) -> dict[str, Any]:
    return max(members, key=lambda row: (float(row["selected_score"]), float(bool(row["selected_correct"])), -int(row["sender_seed"])))


def choose_member_by_score(members: list[dict[str, Any]], name: str, *, reverse: bool) -> dict[str, Any]:
    multiplier = 1.0 if reverse else -1.0
    return max(members, key=lambda row: (multiplier * score_value(row, name), -int(row["sender_seed"])))


def fit_train_priors(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    global_prior = {
        "helpful_rate": mean(row["request_to_3_helpful"] for row in train_rows),
        "gain_mean": mean(row["first_to_prefix3_gain"] for row in train_rows),
    }
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        by_task[str(row["task"])].append(row)
    task_priors = {}
    for task, rows in sorted(by_task.items()):
        task_priors[task] = {
            "count": len(rows),
            "helpful_rate": mean(row["request_to_3_helpful"] for row in rows),
            "gain_mean": mean(row["first_to_prefix3_gain"] for row in rows),
        }
    return {"global": global_prior, "task": task_priors}


def attach_priors(rows: list[dict[str, Any]], priors: dict[str, Any]) -> None:
    for row in rows:
        task_prior = priors["task"].get(str(row["task"]), priors["global"])
        row["train_task_helpful_rate"] = float(task_prior["helpful_rate"])
        row["train_task_gain_mean"] = float(task_prior["gain_mean"])
        row["train_global_helpful_rate"] = float(priors["global"]["helpful_rate"])
        row["train_global_gain_mean"] = float(priors["global"]["gain_mean"])


def aggregate_prefix_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "num_groups": float(len(rows)),
        "first_accuracy": mean(row["first_correct"] for row in rows),
        "first_score": mean(row["first_score"] for row in rows),
        "prefix2_oracle_accuracy": mean(row["prefix2_oracle_correct"] for row in rows),
        "prefix2_oracle_score": mean(row["prefix2_oracle_score"] for row in rows),
        "prefix3_oracle_accuracy": mean(row["prefix3_oracle_correct"] for row in rows),
        "prefix3_oracle_score": mean(row["prefix3_oracle_score"] for row in rows),
        "prefix2_readiness_accuracy": mean(row["prefix2_readiness_correct"] for row in rows),
        "prefix2_readiness_score": mean(row["prefix2_readiness_score"] for row in rows),
        "prefix3_readiness_accuracy": mean(row["prefix3_readiness_correct"] for row in rows),
        "prefix3_readiness_score": mean(row["prefix3_readiness_score"] for row in rows),
        "request_to_2_helpful_rate": mean(row["request_to_2_helpful"] for row in rows),
        "request_to_3_helpful_rate": mean(row["request_to_3_helpful"] for row in rows),
        "request_2_to_3_helpful_rate": mean(row["request_2_to_3_helpful"] for row in rows),
        "mean_first_to_prefix2_gain": mean(row["first_to_prefix2_gain"] for row in rows),
        "mean_first_to_prefix3_gain": mean(row["first_to_prefix3_gain"] for row in rows),
        "mean_prefix2_to_prefix3_gain": mean(row["prefix2_to_prefix3_gain"] for row in rows),
    }


def evaluate_request_policies(
    *,
    valid_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    request_rates: list[float],
    helpful_precisions: list[float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for signal, direction in REQUEST_SIGNALS.items():
        for rate in request_rates:
            threshold = choose_threshold_for_rate(valid_rows, signal, direction, rate)
            out.append(policy_row("target_request_rate", rate, signal, direction, threshold, valid_rows, test_rows))
        for precision in helpful_precisions:
            threshold = choose_threshold_for_helpful_precision(valid_rows, signal, direction, precision)
            out.append(policy_row("target_helpful_precision", precision, signal, direction, threshold, valid_rows, test_rows))
    return out


def policy_row(
    mode: str,
    target: float,
    signal: str,
    direction: str,
    threshold: float,
    valid_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "selection_mode": mode,
        "target_value": target,
        "signal": signal,
        "direction": direction,
        "threshold": threshold,
        **prefixed("valid", request_policy_metrics(valid_rows, signal, direction, threshold)),
        **prefixed("test", request_policy_metrics(test_rows, signal, direction, threshold)),
    }


def request_policy_metrics(rows: list[dict[str, Any]], signal: str, direction: str, threshold: float) -> dict[str, float]:
    selected = [should_request(row, signal, direction, threshold) for row in rows]
    requested = [row for row, is_selected in zip(rows, selected) if is_selected]
    oracle_scores = [
        row["prefix3_oracle_score"] if is_selected else row["first_score"]
        for row, is_selected in zip(rows, selected)
    ]
    oracle_correct = [
        row["prefix3_oracle_correct"] if is_selected else row["first_correct"]
        for row, is_selected in zip(rows, selected)
    ]
    readiness_scores = [
        row["prefix3_readiness_score"] if is_selected else row["first_score"]
        for row, is_selected in zip(rows, selected)
    ]
    readiness_correct = [
        row["prefix3_readiness_correct"] if is_selected else row["first_correct"]
        for row, is_selected in zip(rows, selected)
    ]
    return {
        "request_rate": mean(float(value) for value in selected),
        "avg_sender_budget": 1.0 + 2.0 * mean(float(value) for value in selected),
        "helpful_precision": mean(row["request_to_3_helpful"] for row in requested),
        "oracle_after_request_accuracy": mean(oracle_correct),
        "oracle_after_request_score": mean(oracle_scores),
        "readiness_after_request_accuracy": mean(readiness_correct),
        "readiness_after_request_score": mean(readiness_scores),
        "mean_score_gain_vs_first_oracle": mean(score - row["first_score"] for score, row in zip(oracle_scores, rows)),
        "mean_score_gain_vs_first_readiness": mean(score - row["first_score"] for score, row in zip(readiness_scores, rows)),
    }


def choose_threshold_for_rate(rows: list[dict[str, Any]], signal: str, direction: str, target_rate: float) -> float:
    values = sorted((float(row[signal]) for row in rows), reverse=(direction == "high"))
    if not values:
        return 1.0
    k = max(1, min(len(values), int(round(target_rate * len(values)))))
    return float(values[k - 1])


def choose_threshold_for_helpful_precision(rows: list[dict[str, Any]], signal: str, direction: str, target_precision: float) -> float:
    thresholds = sorted({float(row[signal]) for row in rows}, reverse=(direction == "high"))
    best_threshold = 1.0 if direction == "high" else -1.0
    best_rate = -1.0
    for threshold in thresholds:
        requested = [row for row in rows if should_request(row, signal, direction, threshold)]
        if not requested:
            continue
        precision = mean(row["request_to_3_helpful"] for row in requested)
        rate = len(requested) / len(rows)
        if precision >= target_precision and rate > best_rate:
            best_threshold = threshold
            best_rate = rate
    return float(best_threshold)


def should_request(row: dict[str, Any], signal: str, direction: str, threshold: float) -> bool:
    value = float(row[signal])
    if direction == "high":
        return value >= threshold
    if direction == "low":
        return value <= threshold
    raise ValueError(f"unknown direction: {direction}")


def aggregate_by_task(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task"])].append(row)
    out = []
    for task, task_rows in sorted(by_task.items()):
        out.append({"task": task, **aggregate_prefix_metrics(task_rows)})
    return out


def write_outputs(
    output_dir: Path,
    config: RequestMoreLatentPotentialConfig,
    rows_by_split: dict[str, list[dict[str, Any]]],
    prefix_metrics: dict[str, dict[str, float]],
    policy_rows: list[dict[str, Any]],
    per_task: list[dict[str, Any]],
    priors: dict[str, Any],
) -> dict[str, str]:
    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as handle:
        now = int(time.time())
        for split, metrics in prefix_metrics.items():
            handle.write(json.dumps({"created_at": now, "kind": "prefix", "split": split, "metrics": metrics}, sort_keys=True) + "\n")
        for row in policy_rows:
            handle.write(json.dumps({"created_at": now, "kind": "policy", "metrics": row}, sort_keys=True) + "\n")
    policy_path = output_dir / "request_policy_metrics.csv"
    write_csv(policy_path, policy_rows)
    per_task_path = output_dir / "test_per_task_prefix_metrics.csv"
    write_csv(per_task_path, per_task)
    rows_path = output_dir / "request_more_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for split in ["train", "valid", "test"]:
            for row in rows_by_split[split]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    priors_path = output_dir / "train_priors.json"
    priors_path.write_text(json.dumps(priors, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "summary_json": str(output_dir / "summary.json"),
        "metrics_jsonl": str(metrics_path),
        "request_policy_metrics_csv": str(policy_path),
        "test_per_task_prefix_metrics_csv": str(per_task_path),
        "request_more_rows_jsonl": str(rows_path),
        "train_priors_json": str(priors_path),
        "config_json": str(config_path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def mean(values: Any) -> float:
    values = [float(value) for value in values]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


if __name__ == "__main__":
    main()
