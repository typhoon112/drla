"""Audit calibration and risk-control for a P2-E joint request/select policy.

This is a local-only evaluation script.  It reads saved valid/test prediction
JSONL files, selects request thresholds on valid, and reports held-out test
metrics.  No optimizer/backward is used, so SwanLab must be disabled.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from drla.tracking import require_swanlab_disabled_for_non_training


@dataclass(frozen=True)
class JointPolicyCalibrationConfig:
    policy_dir: str = (
        "/data1/luyifei/drla/outputs/cola_joint_request_select_policy/"
        "p2e_joint_request_select_score_full_seed20260529_20260529"
    )
    output_dir: str = (
        "/data1/luyifei/drla/outputs/cola_joint_request_select_policy/"
        "p2e_joint_request_select_score_full_seed20260529_20260529_calibration"
    )
    target_request_rates: str = "0.10,0.25,0.50"
    target_model_helpful_precisions: str = "0.50,0.60,0.70"
    target_loss_risk_uppers: str = "0.10,0.20,0.30"
    risk_bound_z: float = 1.96
    n_bins: int = 10
    swanlab_mode: str = "disabled"


def main() -> None:
    summary = audit_joint_policy_calibration(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> JointPolicyCalibrationConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", default=JointPolicyCalibrationConfig.policy_dir)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-request-rates", default=JointPolicyCalibrationConfig.target_request_rates)
    parser.add_argument("--target-model-helpful-precisions", default=JointPolicyCalibrationConfig.target_model_helpful_precisions)
    parser.add_argument("--target-loss-risk-uppers", default=JointPolicyCalibrationConfig.target_loss_risk_uppers)
    parser.add_argument("--risk-bound-z", type=float, default=JointPolicyCalibrationConfig.risk_bound_z)
    parser.add_argument("--n-bins", type=int, default=JointPolicyCalibrationConfig.n_bins)
    parser.add_argument("--swanlab-mode", default=JointPolicyCalibrationConfig.swanlab_mode)
    args = parser.parse_args()
    return JointPolicyCalibrationConfig(
        policy_dir=args.policy_dir,
        output_dir=args.output_dir,
        target_request_rates=args.target_request_rates,
        target_model_helpful_precisions=args.target_model_helpful_precisions,
        target_loss_risk_uppers=args.target_loss_risk_uppers,
        risk_bound_z=args.risk_bound_z,
        n_bins=args.n_bins,
        swanlab_mode=args.swanlab_mode,
    )


def audit_joint_policy_calibration(config: JointPolicyCalibrationConfig) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        config.swanlab_mode,
        script_kind="P2-E joint request/select calibration audit",
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_dir = Path(config.policy_dir)
    valid_rows = enrich_rows(read_jsonl(policy_dir / "valid_predictions.jsonl"))
    test_rows = enrich_rows(read_jsonl(policy_dir / "test_predictions.jsonl"))
    if not valid_rows or not test_rows:
        raise ValueError("valid/test predictions must be non-empty")

    calibration = {
        "valid": calibration_metrics(valid_rows, config),
        "test": calibration_metrics(test_rows, config),
    }
    policy_rows = evaluate_threshold_sweep(valid_rows, test_rows, config)
    per_task = aggregate_by_task(test_rows)
    artifacts = write_outputs(output_dir, config, calibration, policy_rows, per_task)
    best_rows = {
        "best_test_model_score": max(policy_rows, key=lambda row: float(row["test_model_after_request_score"])),
        "best_test_score_gain_vs_fuser": max(policy_rows, key=lambda row: float(row["test_score_gain_vs_always_fuser"])),
        "best_test_loss_risk_controlled": best_risk_controlled(policy_rows),
    }
    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "policy_dir": str(policy_dir),
        "split_sizes": {"valid": len(valid_rows), "test": len(test_rows)},
        "baselines": {"valid": baseline_metrics(valid_rows), "test": baseline_metrics(test_rows)},
        "calibration": calibration,
        "best_rows": best_rows,
        "artifacts": artifacts,
        "interpretation": (
            "Local-only calibration/risk-control audit. Thresholds are selected "
            "on valid and reported on held-out test. Labels/scores are used only "
            "for offline calibration and evaluation."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item["model_gain"] = float(item["model_score"]) - float(item["first_score"])
        item["oracle_gain"] = float(item["oracle_best_score"]) - float(item["first_score"])
        item["fuser_gain"] = float(item["fuser_score"]) - float(item["first_score"])
        item["model_loss"] = float(float(item["model_score"]) < float(item["first_score"]) - 1e-12)
        item["oracle_loss"] = float(float(item["oracle_best_score"]) < float(item["first_score"]) - 1e-12)
        out.append(item)
    return out


def calibration_metrics(rows: list[dict[str, Any]], config: JointPolicyCalibrationConfig) -> dict[str, Any]:
    prob = [float(row["request_prob"]) for row in rows]
    model_target = [float(row["model_request_helpful"]) for row in rows]
    oracle_target = [float(row["oracle_request_helpful"]) for row in rows]
    gain_pred = [float(row["request_gain_pred"]) for row in rows]
    model_gain = [float(row["model_gain"]) for row in rows]
    oracle_gain = [float(row["oracle_gain"]) for row in rows]
    return {
        "request_prob_vs_model_helpful": binary_calibration(prob, model_target, config.n_bins),
        "request_prob_vs_oracle_helpful": binary_calibration(prob, oracle_target, config.n_bins),
        "request_gain_vs_model_gain": regression_metrics(gain_pred, model_gain),
        "request_gain_vs_oracle_gain": regression_metrics(gain_pred, oracle_gain),
    }


def binary_calibration(prob: list[float], target: list[float], n_bins: int) -> dict[str, Any]:
    brier = mean((p - y) ** 2 for p, y in zip(prob, target))
    ece = 0.0
    mce = 0.0
    bins = []
    total = len(prob)
    for idx in range(n_bins):
        lo = idx / n_bins
        hi = (idx + 1) / n_bins
        selected = [
            (p, y)
            for p, y in zip(prob, target)
            if (p >= lo and (p < hi or (idx == n_bins - 1 and p <= hi)))
        ]
        if not selected:
            bins.append({"bin": idx, "lo": lo, "hi": hi, "count": 0, "confidence": 0.0, "accuracy": 0.0, "gap": 0.0})
            continue
        confidence = mean(p for p, _y in selected)
        accuracy = mean(y for _p, y in selected)
        gap = abs(confidence - accuracy)
        ece += len(selected) / total * gap
        mce = max(mce, gap)
        bins.append(
            {
                "bin": idx,
                "lo": lo,
                "hi": hi,
                "count": len(selected),
                "confidence": confidence,
                "accuracy": accuracy,
                "gap": gap,
            }
        )
    return {
        "count": total,
        "target_mean": mean(target),
        "prob_mean": mean(prob),
        "brier": brier,
        "ece": ece,
        "mce": mce,
        "auroc": auroc(prob, target),
        "bins": bins,
    }


def regression_metrics(pred: list[float], target: list[float]) -> dict[str, float]:
    errors = [p - y for p, y in zip(pred, target)]
    return {
        "pred_mean": mean(pred),
        "target_mean": mean(target),
        "mae": mean(abs(value) for value in errors),
        "rmse": math.sqrt(mean(value * value for value in errors)),
        "corr": pearson(pred, target),
    }


def evaluate_threshold_sweep(
    valid_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    config: JointPolicyCalibrationConfig,
) -> list[dict[str, Any]]:
    policy_rows: list[dict[str, Any]] = []
    for signal in ["request_prob", "request_gain_pred"]:
        for rate in parse_float_list(config.target_request_rates):
            threshold = choose_threshold_for_rate(valid_rows, signal, rate)
            policy_rows.append(policy_row("target_request_rate", rate, signal, threshold, True, valid_rows, test_rows, config))
        for precision in parse_float_list(config.target_model_helpful_precisions):
            threshold = choose_threshold_for_model_helpful_precision(valid_rows, signal, precision)
            policy_rows.append(
                policy_row(
                    "target_model_helpful_precision",
                    precision,
                    signal,
                    threshold,
                    threshold is not None,
                    valid_rows,
                    test_rows,
                    config,
                )
            )
        for risk_upper in parse_float_list(config.target_loss_risk_uppers):
            threshold = choose_threshold_for_loss_risk(valid_rows, signal, risk_upper, config.risk_bound_z)
            policy_rows.append(
                policy_row(
                    "target_loss_wilson_upper",
                    risk_upper,
                    signal,
                    threshold,
                    threshold is not None,
                    valid_rows,
                    test_rows,
                    config,
                )
            )
    return policy_rows


def policy_row(
    mode: str,
    target: float,
    signal: str,
    threshold: float | None,
    selected: bool,
    valid_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    config: JointPolicyCalibrationConfig,
) -> dict[str, Any]:
    return {
        "selection_mode": mode,
        "target_value": target,
        "signal": signal,
        "threshold": threshold if threshold is not None else "",
        "selected": selected,
        **prefixed("valid", request_metrics(valid_rows, signal, threshold, config.risk_bound_z)),
        **prefixed("test", request_metrics(test_rows, signal, threshold, config.risk_bound_z)),
    }


def request_metrics(rows: list[dict[str, Any]], signal: str, threshold: float | None, z: float) -> dict[str, float]:
    threshold_value = float("inf") if threshold is None else threshold
    requested = [float(row[signal]) >= threshold_value for row in rows]
    requested_rows = [row for row, is_requested in zip(rows, requested) if is_requested]
    model_correct = [row["model_correct"] if is_requested else row["first_correct"] for row, is_requested in zip(rows, requested)]
    model_score = [row["model_score"] if is_requested else row["first_score"] for row, is_requested in zip(rows, requested)]
    oracle_correct = [row["oracle_any_correct"] if is_requested else row["first_correct"] for row, is_requested in zip(rows, requested)]
    oracle_score = [row["oracle_best_score"] if is_requested else row["first_score"] for row, is_requested in zip(rows, requested)]
    request_count = len(requested_rows)
    loss_count = sum(int(float(row["model_score"]) < float(row["first_score"]) - 1e-12) for row in requested_rows)
    return {
        "request_rate": mean(float(value) for value in requested),
        "avg_sender_budget": 1.0 + 2.0 * mean(float(value) for value in requested),
        "request_count": float(request_count),
        "model_helpful_precision": mean(row["model_request_helpful"] for row in requested_rows),
        "oracle_helpful_precision": mean(row["oracle_request_helpful"] for row in requested_rows),
        "model_loss_rate_on_requested": safe_div(loss_count, request_count),
        "model_loss_wilson_upper": wilson_upper(loss_count, request_count, z),
        "model_after_request_accuracy": mean(model_correct),
        "model_after_request_score": mean(model_score),
        "oracle_after_request_accuracy": mean(oracle_correct),
        "oracle_after_request_score": mean(oracle_score),
        "score_gain_vs_first": mean(score - row["first_score"] for score, row in zip(model_score, rows)),
        "score_gain_vs_text": mean(model_score) - mean(row["text_majority_score"] for row in rows),
        "score_gain_vs_always_fuser": mean(model_score) - mean(row["fuser_score"] for row in rows),
        "accuracy_gain_vs_text": mean(model_correct) - mean(row["text_majority_correct"] for row in rows),
        "accuracy_gain_vs_always_fuser": mean(model_correct) - mean(row["fuser_correct"] for row in rows),
    }


def choose_threshold_for_rate(rows: list[dict[str, Any]], signal: str, target_rate: float) -> float:
    values = sorted(float(row[signal]) for row in rows)[::-1]
    if not values:
        return 1.0
    k = max(1, min(len(values), int(round(target_rate * len(values)))))
    return float(values[k - 1])


def choose_threshold_for_model_helpful_precision(rows: list[dict[str, Any]], signal: str, target_precision: float) -> float | None:
    thresholds = sorted({float(row[signal]) for row in rows}, reverse=True)
    best_threshold = None
    best_rate = -1.0
    for threshold in thresholds:
        requested = [row for row in rows if float(row[signal]) >= threshold]
        if not requested:
            continue
        precision = mean(row["model_request_helpful"] for row in requested)
        rate = len(requested) / len(rows)
        if precision >= target_precision and rate > best_rate:
            best_threshold = float(threshold)
            best_rate = rate
    return best_threshold


def choose_threshold_for_loss_risk(rows: list[dict[str, Any]], signal: str, risk_upper: float, z: float) -> float | None:
    thresholds = sorted({float(row[signal]) for row in rows}, reverse=True)
    best_threshold = None
    best_rate = -1.0
    for threshold in thresholds:
        requested = [row for row in rows if float(row[signal]) >= threshold]
        if not requested:
            continue
        loss_count = sum(int(float(row["model_score"]) < float(row["first_score"]) - 1e-12) for row in requested)
        upper = wilson_upper(loss_count, len(requested), z)
        rate = len(requested) / len(rows)
        if upper <= risk_upper and rate > best_rate:
            best_threshold = float(threshold)
            best_rate = rate
    return best_threshold


def aggregate_by_task(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks = sorted({str(row["task"]) for row in rows})
    out = []
    for task in tasks:
        task_rows = [row for row in rows if str(row["task"]) == task]
        out.append({"task": task, **baseline_metrics(task_rows)})
    return out


def baseline_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "count": float(len(rows)),
        "first_accuracy": mean(row["first_correct"] for row in rows),
        "first_score": mean(row["first_score"] for row in rows),
        "text_majority_accuracy": mean(row["text_majority_correct"] for row in rows),
        "text_majority_score": mean(row["text_majority_score"] for row in rows),
        "fuser_accuracy": mean(row["fuser_correct"] for row in rows),
        "fuser_score": mean(row["fuser_score"] for row in rows),
        "always_request_model_accuracy": mean(row["model_correct"] for row in rows),
        "always_request_model_score": mean(row["model_score"] for row in rows),
        "oracle_any_accuracy": mean(row["oracle_any_correct"] for row in rows),
        "oracle_best_score": mean(row["oracle_best_score"] for row in rows),
    }


def best_risk_controlled(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if str(row["selection_mode"]) == "target_loss_wilson_upper" and bool(row.get("selected"))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (float(row["test_model_after_request_score"]), float(row["test_request_rate"])))


def write_outputs(
    output_dir: Path,
    config: JointPolicyCalibrationConfig,
    calibration: dict[str, Any],
    policy_rows: list[dict[str, Any]],
    per_task: list[dict[str, Any]],
) -> dict[str, str]:
    calibration_path = output_dir / "calibration_metrics.json"
    calibration_path.write_text(json.dumps(calibration, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    policy_path = output_dir / "risk_policy_metrics.csv"
    write_csv(policy_path, policy_rows)
    per_task_path = output_dir / "test_per_task_baselines.csv"
    write_csv(per_task_path, per_task)
    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"created_at": int(time.time()), "kind": "calibration", "metrics": calibration}, sort_keys=True) + "\n")
        for row in policy_rows:
            handle.write(json.dumps({"created_at": int(time.time()), "kind": "policy", "metrics": row}, sort_keys=True) + "\n")
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "summary_json": str(output_dir / "summary.json"),
        "calibration_metrics_json": str(calibration_path),
        "risk_policy_metrics_csv": str(policy_path),
        "test_per_task_baselines_csv": str(per_task_path),
        "metrics_jsonl": str(metrics_path),
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def wilson_upper(successes: int, n: int, z: float) -> float:
    if n <= 0:
        return 1.0
    phat = successes / n
    denom = 1.0 + z * z / n
    center = phat + z * z / (2.0 * n)
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n)
    return min(1.0, (center + margin) / denom)


def auroc(score: list[float], target: list[float]) -> float:
    pairs = sorted(zip(score, target), key=lambda item: item[0])
    pos = sum(1 for _score, y in pairs if y >= 0.5)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return 0.5
    rank_sum = 0.0
    for rank, (_score, y) in enumerate(pairs, start=1):
        if y >= 0.5:
            rank_sum += rank
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def pearson(x: list[float], y: list[float]) -> float:
    if not x or not y or len(x) != len(y):
        return 0.0
    mx = mean(x)
    my = mean(y)
    vx = [value - mx for value in x]
    vy = [value - my for value in y]
    denom = math.sqrt(sum(value * value for value in vx) * sum(value * value for value in vy))
    if denom <= 0:
        return 0.0
    return sum(a * b for a, b in zip(vx, vy)) / denom


def safe_div(num: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return float(num / denom)


def mean(values: Any) -> float:
    values = [float(value) for value in values]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


if __name__ == "__main__":
    main()
