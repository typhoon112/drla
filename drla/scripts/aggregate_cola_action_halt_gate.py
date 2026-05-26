"""Aggregate causal action-to-halt gate diagnostics for Cola latent halt students.

This script is intentionally local-only: it combines existing student-only
evaluation artifacts and never creates SwanLab runs.  The gate may use only
latent-student scores available at the block where the aggressive action policy
would stop.  Labels and decoded text are used only for calibration/evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


TASKS = ["lambada", "mmlu", "obqa", "hellaswag", "race", "siqa", "squad", "story_cloze"]


@dataclass(frozen=True)
class ActionHaltGateConfig:
    action_root_glob: str = (
        "/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/"
        "cross_task_full_b64_bs12_seed*_d64_pma4_answer_identity_action_completionrisk_"
        "targetcal_cap128_boundarypen02_subseeds_20260525"
    )
    halt_root_glob: str = (
        "/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/"
        "cross_task_full_b64_bs12_seed*_d64_pma4_answer_identity_halt_completionrisk_"
        "targetcalstrict_cap128_boundarypen02_subseeds_20260525"
    )
    output_dir: str = (
        "/data1/luyifei/drla/outputs/cola_experiment_summaries/"
        "official8_full_b64_bs12_p1_action_halt_latent_gate_diagnostic_20260525"
    )
    split: str = "test"
    calibration_split: str = "valid"


@dataclass(frozen=True)
class GateRule:
    name: str
    max_action_block: int | None = None
    risk_ge: float | None = None
    contentful_lt: float | None = None
    pred_change_ge: float | None = None
    readiness_lt: float | None = None
    future_gain_ge: float | None = None
    correctness_lt: float | None = None
    op: str = "or"


def main() -> None:
    summary = aggregate_action_halt_gate(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def aggregate_action_halt_gate(config: ActionHaltGateConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_eval_pairs(config)
    if not pairs:
        raise FileNotFoundError("no matched action/halt evaluation directories found")

    valid_cases: list[dict[str, Any]] = []
    test_cases: list[dict[str, Any]] = []
    pair_rows = []
    for pair in pairs:
        valid = build_cases(pair, split=config.calibration_split)
        test = build_cases(pair, split=config.split)
        valid_cases.extend(valid)
        test_cases.extend(test)
        pair_rows.append(
            {
                "seed": pair["seed"],
                "subseed": pair["subseed"],
                "task": pair["task"],
                "valid_samples": len(valid),
                "test_samples": len(test),
                "action_dir": str(pair["action_dir"]),
                "halt_dir": str(pair["halt_dir"]),
            }
        )

    rules = build_gate_rules()
    valid_sweep = evaluate_rules(valid_cases, rules)
    test_sweep = evaluate_rules(test_cases, rules)
    selected_by_valid_safety = select_rule(valid_sweep, mode="valid_loss_mismatch_then_blocks")
    selected_by_valid_cost = select_rule(valid_sweep, mode="valid_loss_blocks_then_mismatch")
    selected_safety_test = find_rule_metrics(test_sweep, selected_by_valid_safety["rule_name"])
    selected_cost_test = find_rule_metrics(test_sweep, selected_by_valid_cost["rule_name"])
    best_test_loss = select_rule(test_sweep, mode="test_loss_then_blocks")
    best_test_cost_limited = select_rule(
        test_sweep,
        mode="test_loss_under_action_plus_0p10_blocks",
        action_avg_blocks=policy_metrics(test_cases, "action")["avg_blocks"],
    )

    task_rows = build_task_rows(
        test_cases,
        rules,
        selected_by_valid_safety["rule_name"],
        selected_by_valid_cost["rule_name"],
    )
    write_csv(output_dir / "matched_eval_pairs.csv", pair_rows)
    write_csv(output_dir / "global_valid_gate_sweep.csv", valid_sweep)
    write_csv(output_dir / "global_test_gate_sweep.csv", test_sweep)
    write_csv(output_dir / "task_policy_summary.csv", task_rows)

    summary = {
        "created_at": int(time.time()),
        "route": "official8 P1 action-to-halt latent-only causal gate diagnostic",
        "config": asdict(config),
        "num_eval_pairs": len(pairs),
        "num_valid_cases": len(valid_cases),
        "num_test_cases": len(test_cases),
        "allowed_inference_inputs": [
            "action selected block number",
            "action student_readiness",
            "action student_prediction_change",
            "action student_contentful",
            "action student_completion_risk",
            "action student_future_gain",
            "action student_correctness head",
        ],
        "forbidden_inference_inputs": [
            "decoded selected_prediction",
            "decoded final_prediction",
            "official_correct",
            "task scorer result",
            "future block outputs before they are generated",
        ],
        "policies": {
            "action": policy_metrics(test_cases, "action"),
            "halt_original": policy_metrics(test_cases, "halt_original"),
            "causal_always_defer_after_action": policy_metrics(test_cases, "fallback"),
            "causal_oracle_defer_after_action": oracle_causal_metrics(test_cases),
            "valid_safety_selected_gate": selected_safety_test,
            "valid_cost_selected_gate": selected_cost_test,
            "best_test_gate_by_loss": best_test_loss,
            "best_test_gate_under_action_plus_0p10_blocks": best_test_cost_limited,
        },
        "selected_rules_from_valid": {
            "safety_loss_mismatch_blocks": selected_by_valid_safety,
            "cost_loss_blocks_mismatch": selected_by_valid_cost,
        },
        "artifacts": {
            "matched_eval_pairs_csv": str(output_dir / "matched_eval_pairs.csv"),
            "global_valid_gate_sweep_csv": str(output_dir / "global_valid_gate_sweep.csv"),
            "global_test_gate_sweep_csv": str(output_dir / "global_test_gate_sweep.csv"),
            "task_policy_summary_csv": str(output_dir / "task_policy_summary.csv"),
        },
        "readout": build_readout(
            action=policy_metrics(test_cases, "action"),
            halt=policy_metrics(test_cases, "halt_original"),
            fallback=policy_metrics(test_cases, "fallback"),
            oracle=oracle_causal_metrics(test_cases),
            selected=selected_safety_test,
            selected_cost=selected_cost_test,
            best=best_test_loss,
            best_limited=best_test_cost_limited,
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def find_eval_pairs(config: ActionHaltGateConfig) -> list[dict[str, Any]]:
    action_roots = sorted(Path("/").glob(config.action_root_glob.lstrip("/")))
    halt_roots = sorted(Path("/").glob(config.halt_root_glob.lstrip("/")))
    halt_by_seed = {extract_seed(path): path for path in halt_roots}
    pairs: list[dict[str, Any]] = []
    for action_root in action_roots:
        seed = extract_seed(action_root)
        halt_root = halt_by_seed.get(seed)
        if halt_root is None:
            continue
        for action_dir in sorted(action_root.glob("subseed*/leave_*_out_eval_*_test")):
            subseed = action_dir.parent.name
            task = extract_task(action_dir.name)
            halt_dir = halt_root / subseed / action_dir.name
            if not halt_dir.exists():
                continue
            pairs.append(
                {
                    "seed": seed,
                    "subseed": subseed.replace("subseed", ""),
                    "task": task,
                    "action_dir": action_dir,
                    "halt_dir": halt_dir,
                }
            )
    return pairs


def extract_seed(path: Path) -> str:
    match = re.search(r"_seed(\d+)_", path.name)
    if not match:
        raise ValueError(f"cannot parse seed from {path}")
    return match.group(1)


def extract_task(dirname: str) -> str:
    match = re.match(r"leave_(.+)_out_eval_(.+)_test$", dirname)
    if not match:
        raise ValueError(f"cannot parse task from {dirname}")
    left, right = match.groups()
    if left != right:
        raise ValueError(f"mismatched leave/eval task in {dirname}")
    return left


def build_cases(pair: dict[str, Any], *, split: str) -> list[dict[str, Any]]:
    action_dir = Path(pair["action_dir"])
    halt_dir = Path(pair["halt_dir"])
    action_summary = load_json(action_dir / "summary.json")
    halt_summary = load_json(halt_dir / "summary.json")
    action_thresholds = action_summary["selected_valid"]
    halt_thresholds = halt_summary["selected_valid"]
    action_scores = load_scores_by_sample(action_dir / "student_scores.jsonl", split=split)
    halt_scores = load_scores_by_sample(halt_dir / "student_scores.jsonl", split=split)
    action_decisions = load_or_reconstruct_decisions(
        action_dir / f"halt_decisions_{split}.jsonl",
        scores=action_scores,
        thresholds=action_thresholds,
    )
    halt_decisions = load_or_reconstruct_decisions(
        halt_dir / f"halt_decisions_{split}.jsonl",
        scores=halt_scores,
        thresholds=halt_thresholds,
    )

    cases = []
    for sample_key, action in sorted(action_decisions.items()):
        halt_original = halt_decisions.get(sample_key)
        score_rows = halt_scores.get(sample_key)
        if halt_original is None or not score_rows:
            raise ValueError(f"missing matched halt data for {sample_key} under {halt_dir}")
        fallback = choose_causal_fallback(
            score_rows=score_rows,
            action_selected_block=int(action["selected_block"]),
            thresholds=halt_thresholds,
        )
        cases.append(
            {
                "seed": pair["seed"],
                "subseed": pair["subseed"],
                "task": pair["task"],
                "sample_key": sample_key,
                "action": normalize_decision(action, policy="action"),
                "halt_original": normalize_decision(halt_original, policy="halt_original"),
                "fallback": normalize_decision(fallback, policy="fallback"),
            }
        )
    return cases


def load_or_reconstruct_decisions(
    path: Path,
    *,
    scores: dict[str, list[dict[str, Any]]],
    thresholds: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if path.exists():
        return load_decisions(path)
    return {
        sample_key: choose_policy_from_scores(score_rows=rows, thresholds=thresholds)
        for sample_key, rows in scores.items()
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_decisions(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            result[str(item["sample_key"])] = item
    return result


def load_scores_by_sample(path: Path, *, split: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("split") != split:
                continue
            sample_key = f"{item['task']}::{item['sample_id']}"
            grouped.setdefault(sample_key, []).append(item)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["block_number"]))
    return grouped


def choose_causal_fallback(
    *,
    score_rows: list[dict[str, Any]],
    action_selected_block: int,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    final = score_rows[-1]
    selected = final
    for row in score_rows:
        block = int(row["block_number"])
        if block <= action_selected_block:
            continue
        if row_passes_thresholds(row, thresholds):
            selected = row
            break
    return decision_from_score_row(
        selected=selected,
        final=final,
        stable=prediction_stability_row(score_rows),
        thresholds=thresholds,
    )


def choose_policy_from_scores(
    *,
    score_rows: list[dict[str, Any]],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    final = score_rows[-1]
    selected = final
    for row in score_rows:
        if row_passes_thresholds(row, thresholds):
            selected = row
            break
    return decision_from_score_row(
        selected=selected,
        final=final,
        stable=prediction_stability_row(score_rows),
        thresholds=thresholds,
    )


def row_passes_thresholds(row: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    checks = [
        float(row["student_readiness"]) >= float(thresholds["readiness_threshold"]),
        float(row["student_prediction_change"]) <= float(thresholds["risk_threshold"]),
        float(row["student_contentful"]) >= float(thresholds["contentful_threshold"]),
    ]
    if thresholds.get("correctness_threshold") is not None:
        checks.append(float(row["student_correctness"]) >= float(thresholds["correctness_threshold"]))
    if thresholds.get("completion_risk_threshold") is not None:
        checks.append(float(row["student_completion_risk"]) <= float(thresholds["completion_risk_threshold"]))
    if thresholds.get("empty_answer_risk_threshold") is not None and "student_empty_answer_risk" in row:
        checks.append(
            float(row["student_empty_answer_risk"]) <= float(thresholds["empty_answer_risk_threshold"])
        )
    if thresholds.get("answer_format_risk_threshold") is not None and "student_answer_format_risk" in row:
        checks.append(
            float(row["student_answer_format_risk"]) <= float(thresholds["answer_format_risk_threshold"])
        )
    return all(checks)


def prediction_stability_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    previous = ""
    streak = 0
    for row in rows:
        prediction = normalize_text(row.get("scored_prediction"))
        if prediction and prediction == previous:
            streak += 1
        else:
            previous = prediction
            streak = 1
        if prediction and streak >= 2:
            return row
    return rows[-1]


def decision_from_score_row(
    *,
    selected: dict[str, Any],
    final: dict[str, Any],
    stable: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    selected_correct = bool(selected["official_correct"])
    final_correct = bool(final["official_correct"])
    stable_correct = bool(stable["official_correct"])
    return {
        "sample_key": f"{selected['task']}::{selected['sample_id']}",
        "task": selected["task"],
        "sample_id": selected["sample_id"],
        "selected_block": int(selected["block_number"]),
        "final_block": int(final["block_number"]),
        "prediction_stability_block": int(stable["block_number"]),
        "selected_correct": selected_correct,
        "final_correct": final_correct,
        "prediction_stability_correct": stable_correct,
        "loss_vs_final": bool(final_correct and not selected_correct),
        "gain_vs_final": bool(selected_correct and not final_correct),
        "loss_vs_prediction_stability": bool(stable_correct and not selected_correct),
        "gain_vs_prediction_stability": bool(selected_correct and not stable_correct),
        "selected_prediction": selected.get("scored_prediction"),
        "final_prediction": final.get("scored_prediction"),
        "prediction_stability_prediction": stable.get("scored_prediction"),
        "readiness_threshold": thresholds.get("readiness_threshold"),
        "risk_threshold": thresholds.get("risk_threshold"),
        "contentful_threshold": thresholds.get("contentful_threshold"),
        "correctness_threshold": thresholds.get("correctness_threshold"),
        "completion_risk_threshold": thresholds.get("completion_risk_threshold"),
        "student_readiness": selected.get("student_readiness"),
        "student_prediction_change": selected.get("student_prediction_change"),
        "student_contentful": selected.get("student_contentful"),
        "student_correctness": selected.get("student_correctness"),
        "student_future_gain": selected.get("student_future_gain"),
        "student_completion_risk": selected.get("student_completion_risk"),
    }


def normalize_decision(item: dict[str, Any], *, policy: str) -> dict[str, Any]:
    result = dict(item)
    result["policy"] = policy
    result["_prediction_mismatch_vs_final"] = (
        normalize_text(result.get("selected_prediction")) != normalize_text(result.get("final_prediction"))
    )
    result["_prediction_mismatch_vs_prediction_stability"] = (
        normalize_text(result.get("selected_prediction"))
        != normalize_text(result.get("prediction_stability_prediction"))
    )
    return result


def build_gate_rules() -> list[GateRule]:
    rules = [GateRule(name="no_gate"), GateRule(name="always_defer_before_final")]
    # Keep the first diagnostic grid deliberately compact.  The rare loss events
    # are better studied with interpretable high-signal thresholds first; dense
    # sweeps can be added once a family looks promising.
    risk_values = [0.2, 0.4, 0.6, 0.8, 0.9]
    contentful_values = [0.1, 0.2, 0.4, 0.7]
    pred_change_values = risk_values
    readiness_values = [0.1, 0.2, 0.4, 0.7]
    future_gain_values = [-0.01, 0.0, 0.01, 0.02]
    correctness_values = [0.1, 0.2, 0.4, 0.7]
    max_blocks = [None, 1, 2]

    for max_block in max_blocks:
        suffix = "" if max_block is None else f"_b{max_block}"
        for value in risk_values:
            rules.append(GateRule(name=f"risk_ge_{value:g}{suffix}", max_action_block=max_block, risk_ge=value))
        for value in contentful_values:
            rules.append(
                GateRule(name=f"contentful_lt_{value:g}{suffix}", max_action_block=max_block, contentful_lt=value)
            )
        for value in pred_change_values:
            rules.append(
                GateRule(
                    name=f"predchange_ge_{value:g}{suffix}",
                    max_action_block=max_block,
                    pred_change_ge=value,
                )
            )
        for value in readiness_values:
            rules.append(
                GateRule(name=f"readiness_lt_{value:g}{suffix}", max_action_block=max_block, readiness_lt=value)
            )
        for value in future_gain_values:
            rules.append(
                GateRule(name=f"futuregain_ge_{value:g}{suffix}", max_action_block=max_block, future_gain_ge=value)
            )
        for value in correctness_values:
            rules.append(
                GateRule(
                    name=f"correctness_lt_{value:g}{suffix}",
                    max_action_block=max_block,
                    correctness_lt=value,
                )
            )
    for max_block in max_blocks:
        suffix = "" if max_block is None else f"_b{max_block}"
        for risk in risk_values:
            for contentful in contentful_values:
                rules.append(
                    GateRule(
                        name=f"risk_ge_{risk:g}_or_contentful_lt_{contentful:g}{suffix}",
                        max_action_block=max_block,
                        risk_ge=risk,
                        contentful_lt=contentful,
                    )
                )
        for risk in risk_values:
            for readiness in readiness_values:
                rules.append(
                    GateRule(
                        name=f"risk_ge_{risk:g}_or_readiness_lt_{readiness:g}{suffix}",
                        max_action_block=max_block,
                        risk_ge=risk,
                        readiness_lt=readiness,
                    )
                )
        for risk in risk_values:
            for pred_change in pred_change_values:
                rules.append(
                    GateRule(
                        name=f"risk_ge_{risk:g}_or_predchange_ge_{pred_change:g}{suffix}",
                        max_action_block=max_block,
                        risk_ge=risk,
                        pred_change_ge=pred_change,
                    )
                )
    return rules


def evaluate_rules(cases: list[dict[str, Any]], rules: Iterable[GateRule]) -> list[dict[str, Any]]:
    rows = []
    for rule in rules:
        rows.append(metrics_for_rule(cases, rule))
    return rows


def metrics_for_rule(cases: list[dict[str, Any]], rule: GateRule) -> dict[str, Any]:
    n = len(cases)
    if n == 0:
        raise ValueError("empty cases")
    defer_count = 0
    rescued_action_losses = 0
    introduced_losses = 0
    selected_correct = 0
    final_correct = 0
    stability_correct = 0
    selected_blocks = 0.0
    final_blocks = 0.0
    losses_final = 0
    gains_final = 0
    losses_stability = 0
    gains_stability = 0
    mismatch_final = 0
    mismatch_stability = 0
    for case in cases:
        use_fallback = should_defer(case["action"], rule)
        if use_fallback:
            defer_count += 1
        decision = case["fallback"] if use_fallback else case["action"]
        if bool(case["action"]["loss_vs_final"]) and not bool(decision["loss_vs_final"]):
            rescued_action_losses += 1
        if not bool(case["action"]["loss_vs_final"]) and bool(decision["loss_vs_final"]):
            introduced_losses += 1
        selected_correct += int(bool(decision["selected_correct"]))
        final_correct += int(bool(decision["final_correct"]))
        stability_correct += int(bool(decision["prediction_stability_correct"]))
        selected_blocks += float(decision["selected_block"])
        final_blocks += float(decision["final_block"])
        losses_final += int(bool(decision["loss_vs_final"]))
        gains_final += int(bool(decision["gain_vs_final"]))
        losses_stability += int(bool(decision["loss_vs_prediction_stability"]))
        gains_stability += int(bool(decision["gain_vs_prediction_stability"]))
        mismatch_final += int(bool(decision["_prediction_mismatch_vs_final"]))
        mismatch_stability += int(bool(decision["_prediction_mismatch_vs_prediction_stability"]))
    avg_blocks = selected_blocks / n
    fixed_final_avg_blocks = final_blocks / n
    accuracy = selected_correct / n
    final_accuracy = final_correct / n
    stability_accuracy = stability_correct / n
    metrics = {
        "num_samples": n,
        "accuracy": accuracy,
        "fixed_final_accuracy": final_accuracy,
        "prediction_stability_accuracy": stability_accuracy,
        "accuracy_drop_vs_final": final_accuracy - accuracy,
        "accuracy_drop_vs_prediction_stability": stability_accuracy - accuracy,
        "avg_blocks": avg_blocks,
        "fixed_final_avg_blocks": fixed_final_avg_blocks,
        "block_saving_vs_final": fixed_final_avg_blocks - avg_blocks,
        "block_saving_fraction_vs_final": (fixed_final_avg_blocks - avg_blocks)
        / max(fixed_final_avg_blocks, 1e-9),
        "losses_vs_final": losses_final,
        "gains_vs_final": gains_final,
        "losses_vs_prediction_stability": losses_stability,
        "gains_vs_prediction_stability": gains_stability,
        "prediction_mismatch_vs_final": mismatch_final,
        "prediction_mismatch_rate_vs_final": mismatch_final / n,
        "prediction_mismatch_vs_prediction_stability": mismatch_stability,
        "prediction_mismatch_rate_vs_prediction_stability": mismatch_stability / n,
    }
    metrics.update(
        {
            "rule_name": rule.name,
            "defer_count": defer_count,
            "defer_rate": defer_count / max(len(cases), 1),
            "rescued_action_losses": rescued_action_losses,
            "introduced_losses_vs_action": introduced_losses,
        }
    )
    return metrics


def should_defer(action: dict[str, Any], rule: GateRule) -> bool:
    if rule.name == "no_gate":
        return False
    if int(action["selected_block"]) >= int(action["final_block"]):
        return False
    if rule.max_action_block is not None and int(action["selected_block"]) > rule.max_action_block:
        return False
    if rule.name == "always_defer_before_final":
        return True

    checks = []
    if rule.risk_ge is not None:
        checks.append(float(action.get("student_completion_risk", 0.0)) >= rule.risk_ge)
    if rule.contentful_lt is not None:
        checks.append(float(action.get("student_contentful", 1.0)) < rule.contentful_lt)
    if rule.pred_change_ge is not None:
        checks.append(float(action.get("student_prediction_change", 0.0)) >= rule.pred_change_ge)
    if rule.readiness_lt is not None:
        checks.append(float(action.get("student_readiness", 1.0)) < rule.readiness_lt)
    if rule.future_gain_ge is not None:
        checks.append(float(action.get("student_future_gain", -1.0)) >= rule.future_gain_ge)
    if rule.correctness_lt is not None:
        checks.append(float(action.get("student_correctness", 1.0)) < rule.correctness_lt)
    if not checks:
        return False
    return any(checks) if rule.op == "or" else all(checks)


def policy_metrics(cases: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    return summarize_decisions([case[policy] for case in cases])


def summarize_decisions(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(decisions)
    if n == 0:
        raise ValueError("empty decisions")
    avg_blocks = sum(float(item["selected_block"]) for item in decisions) / n
    final_blocks = sum(float(item["final_block"]) for item in decisions) / n
    mismatch_final = sum(bool(item["_prediction_mismatch_vs_final"]) for item in decisions)
    mismatch_stability = sum(bool(item["_prediction_mismatch_vs_prediction_stability"]) for item in decisions)
    accuracy = sum(bool(item["selected_correct"]) for item in decisions) / n
    final_accuracy = sum(bool(item["final_correct"]) for item in decisions) / n
    stability_accuracy = sum(bool(item["prediction_stability_correct"]) for item in decisions) / n
    return {
        "num_samples": n,
        "accuracy": accuracy,
        "fixed_final_accuracy": final_accuracy,
        "prediction_stability_accuracy": stability_accuracy,
        "accuracy_drop_vs_final": final_accuracy - accuracy,
        "accuracy_drop_vs_prediction_stability": stability_accuracy - accuracy,
        "avg_blocks": avg_blocks,
        "fixed_final_avg_blocks": final_blocks,
        "block_saving_vs_final": final_blocks - avg_blocks,
        "block_saving_fraction_vs_final": (final_blocks - avg_blocks) / max(final_blocks, 1e-9),
        "losses_vs_final": sum(bool(item["loss_vs_final"]) for item in decisions),
        "gains_vs_final": sum(bool(item["gain_vs_final"]) for item in decisions),
        "losses_vs_prediction_stability": sum(bool(item["loss_vs_prediction_stability"]) for item in decisions),
        "gains_vs_prediction_stability": sum(bool(item["gain_vs_prediction_stability"]) for item in decisions),
        "prediction_mismatch_vs_final": mismatch_final,
        "prediction_mismatch_rate_vs_final": mismatch_final / n,
        "prediction_mismatch_vs_prediction_stability": mismatch_stability,
        "prediction_mismatch_rate_vs_prediction_stability": mismatch_stability / n,
    }


def oracle_causal_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    selected = []
    defer_count = 0
    for case in cases:
        action = case["action"]
        fallback = case["fallback"]
        use_fallback = (not bool(action["selected_correct"])) and bool(fallback["selected_correct"])
        if use_fallback:
            defer_count += 1
        selected.append(fallback if use_fallback else action)
    metrics = summarize_decisions(selected)
    metrics.update({"defer_count": defer_count, "defer_rate": defer_count / max(len(cases), 1)})
    return metrics


def select_rule(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    action_avg_blocks: float | None = None,
) -> dict[str, Any]:
    if mode == "valid_loss_mismatch_then_blocks":
        return min(
            rows,
            key=lambda row: (
                int(row["losses_vs_final"]),
                int(row["prediction_mismatch_vs_final"]),
                float(row["avg_blocks"]),
                -int(row["rescued_action_losses"]),
                row["rule_name"],
            ),
        )
    if mode == "valid_loss_blocks_then_mismatch":
        return min(
            rows,
            key=lambda row: (
                int(row["losses_vs_final"]),
                float(row["avg_blocks"]),
                int(row["prediction_mismatch_vs_final"]),
                -int(row["rescued_action_losses"]),
                row["rule_name"],
            ),
        )
    if mode == "test_loss_then_blocks":
        return min(
            rows,
            key=lambda row: (
                int(row["losses_vs_final"]),
                int(row["prediction_mismatch_vs_final"]),
                float(row["avg_blocks"]),
                row["rule_name"],
            ),
        )
    if mode == "test_loss_under_action_plus_0p10_blocks":
        if action_avg_blocks is None:
            raise ValueError("action_avg_blocks is required for cost-limited selection")
        eligible = [row for row in rows if float(row["avg_blocks"]) <= action_avg_blocks + 0.10]
        if not eligible:
            eligible = rows
        return select_rule(eligible, mode="test_loss_then_blocks")
    raise ValueError(f"unknown selection mode: {mode}")


def find_rule_metrics(rows: list[dict[str, Any]], rule_name: str) -> dict[str, Any]:
    for row in rows:
        if row["rule_name"] == rule_name:
            return row
    raise KeyError(rule_name)


def build_task_rows(
    cases: list[dict[str, Any]],
    rules: list[GateRule],
    safety_selected_rule_name: str,
    cost_selected_rule_name: str,
) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_task.setdefault(case["task"], []).append(case)

    safety_selected_rule = next(rule for rule in rules if rule.name == safety_selected_rule_name)
    cost_selected_rule = next(rule for rule in rules if rule.name == cost_selected_rule_name)
    rows = []
    for task in TASKS:
        task_cases = by_task.get(task, [])
        if not task_cases:
            continue
        for name, metrics in [
            ("action", policy_metrics(task_cases, "action")),
            ("halt_original", policy_metrics(task_cases, "halt_original")),
            ("causal_always_defer_after_action", policy_metrics(task_cases, "fallback")),
            ("valid_safety_selected_gate", metrics_for_rule(task_cases, safety_selected_rule)),
            ("valid_cost_selected_gate", metrics_for_rule(task_cases, cost_selected_rule)),
            ("causal_oracle_defer_after_action", oracle_causal_metrics(task_cases)),
        ]:
            rows.append({"task": task, "policy": name, **metrics})
    return rows


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_readout(
    *,
    action: dict[str, Any],
    halt: dict[str, Any],
    fallback: dict[str, Any],
    oracle: dict[str, Any],
    selected: dict[str, Any],
    selected_cost: dict[str, Any],
    best: dict[str, Any],
    best_limited: dict[str, Any],
) -> list[str]:
    return [
        (
            "Action baseline: "
            f"{action['losses_vs_final']} losses, {action['prediction_mismatch_vs_final']} mismatches, "
            f"{action['avg_blocks']:.3f}/4 blocks."
        ),
        (
            "Strict halt baseline: "
            f"{halt['losses_vs_final']} losses, {halt['prediction_mismatch_vs_final']} mismatches, "
            f"{halt['avg_blocks']:.3f}/4 blocks."
        ),
        (
            "Causal always-defer-after-action fallback: "
            f"{fallback['losses_vs_final']} losses, {fallback['prediction_mismatch_vs_final']} mismatches, "
            f"{fallback['avg_blocks']:.3f}/4 blocks."
        ),
        (
            "Valid safety-selected latent gate: "
            f"{selected['rule_name']} -> {selected['losses_vs_final']} losses, "
            f"{selected['prediction_mismatch_vs_final']} mismatches, {selected['avg_blocks']:.3f}/4 blocks."
        ),
        (
            "Valid cost-selected latent gate: "
            f"{selected_cost['rule_name']} -> {selected_cost['losses_vs_final']} losses, "
            f"{selected_cost['prediction_mismatch_vs_final']} mismatches, {selected_cost['avg_blocks']:.3f}/4 blocks."
        ),
        (
            "Best test-sweep gate is diagnostic only: "
            f"{best['rule_name']} -> {best['losses_vs_final']} losses, "
            f"{best['prediction_mismatch_vs_final']} mismatches, {best['avg_blocks']:.3f}/4 blocks."
        ),
        (
            "Best cost-limited test-sweep gate is diagnostic only: "
            f"{best_limited['rule_name']} -> {best_limited['losses_vs_final']} losses, "
            f"{best_limited['prediction_mismatch_vs_final']} mismatches, {best_limited['avg_blocks']:.3f}/4 blocks."
        ),
        (
            "Causal oracle defer-after-action upper bound: "
            f"{oracle['losses_vs_final']} losses, {oracle['prediction_mismatch_vs_final']} mismatches, "
            f"{oracle['avg_blocks']:.3f}/4 blocks."
        ),
    ]


def parse_args() -> ActionHaltGateConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-root-glob", default=ActionHaltGateConfig.action_root_glob)
    parser.add_argument("--halt-root-glob", default=ActionHaltGateConfig.halt_root_glob)
    parser.add_argument("--output-dir", default=ActionHaltGateConfig.output_dir)
    parser.add_argument("--split", default=ActionHaltGateConfig.split)
    parser.add_argument("--calibration-split", default=ActionHaltGateConfig.calibration_split)
    args = parser.parse_args()
    return ActionHaltGateConfig(
        action_root_glob=args.action_root_glob,
        halt_root_glob=args.halt_root_glob,
        output_dir=args.output_dir,
        split=args.split,
        calibration_split=args.calibration_split,
    )


if __name__ == "__main__":
    main()
