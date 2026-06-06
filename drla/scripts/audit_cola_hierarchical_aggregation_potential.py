"""Audit P2-E hierarchical aggregation potential from locked P2 packets.

This is a local-only diagnostic.  It groups the three locked P1 sender packets
for the same official8 sample and compares simple aggregation policies:

* single-sender baselines
* text majority over decoded selected predictions
* latent-readiness-state rankers that choose one sender
* oracle any-correct upper bounds

The decoded predictions and correctness labels used here are offline audit
references.  They must not be treated as online latent receiver inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from drla.scripts.audit_cola_sequential_latent_mas import load_official_scorer, score_text_with_official_rules
from drla.scripts.run_cola_sequential_latent_mas import DecisionSampleIdCache, TaskDataCache
from drla.scripts.train_cola_latent_answer_reader import normalize_answer
from drla.tracking import require_swanlab_disabled_for_non_training


@dataclass(frozen=True)
class AggregationPotentialConfig:
    packets_jsonl: str = (
        "/data1/luyifei/drla/outputs/cola_agent_latent_comm/"
        "p2_agent_latent_comm_v2_locked_seed66_67_68_split20260601_20260529/"
        "agent_latent_comm_packets_test.jsonl"
    )
    output_dir: str = (
        "/data1/luyifei/drla/outputs/cola_hierarchical_aggregation/"
        "p2e_aggregation_potential_locked_seed66_67_68"
    )
    data_root: str = "/data1/luyifei/Cola-DLM/code/generate_task_data"
    acc_calc_script: str = "/data1/luyifei/Cola-DLM/code/scripts/acc_calc.py"
    max_groups: int = 0
    swanlab_mode: str = "disabled"


def main() -> None:
    summary = audit_aggregation_potential(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> AggregationPotentialConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets-jsonl", default=AggregationPotentialConfig.packets_jsonl)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default=AggregationPotentialConfig.data_root)
    parser.add_argument("--acc-calc-script", default=AggregationPotentialConfig.acc_calc_script)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--swanlab-mode", default=AggregationPotentialConfig.swanlab_mode)
    args = parser.parse_args()
    return AggregationPotentialConfig(
        packets_jsonl=args.packets_jsonl,
        output_dir=args.output_dir,
        data_root=args.data_root,
        acc_calc_script=args.acc_calc_script,
        max_groups=args.max_groups,
        swanlab_mode=args.swanlab_mode,
    )


def audit_aggregation_potential(config: AggregationPotentialConfig) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        config.swanlab_mode,
        script_kind="P2-E hierarchical aggregation potential audit",
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scorer = load_official_scorer(Path(config.acc_calc_script))
    packets = read_jsonl(Path(config.packets_jsonl))
    groups = build_groups(packets, config, scorer)
    if config.max_groups:
        groups = groups[: config.max_groups]
    if not groups:
        raise ValueError("no complete packet groups found")

    predictions = []
    for group in groups:
        predictions.extend(evaluate_group(group, scorer))

    metrics = aggregate_metrics(predictions)
    per_task = aggregate_metrics(predictions, by_task=True)
    group_stats = summarize_groups(groups)
    artifacts = write_outputs(output_dir, config, metrics, per_task, group_stats, predictions)
    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "num_groups": len(groups),
        "num_packet_records": sum(len(group["members"]) for group in groups),
        "group_stats": group_stats,
        "metrics": metrics,
        "artifacts": artifacts,
        "interpretation": (
            "Local-only P2-E potential audit. Text-majority and oracle rows use "
            "decoded audit references; latent-state rankers only rank senders by "
            "P1 latent-readiness heads, but their chosen decoded answer is still "
            "an offline evaluation proxy."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_groups(
    packets: list[dict[str, Any]],
    config: AggregationPotentialConfig,
    scorer: Any,
) -> list[dict[str, Any]]:
    decision_cache: dict[str, dict[str, dict[str, Any]]] = {}
    data_cache = TaskDataCache(Path(config.data_root))
    sample_cache = DecisionSampleIdCache()
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for packet in packets:
        task = str(packet["task"])
        sample_key = str(packet["sample_key"])
        halt_path = str(packet["audit_refs"]["halt_decisions_jsonl"])
        if halt_path not in decision_cache:
            decision_cache[halt_path] = {str(row["sample_key"]): row for row in read_jsonl(Path(halt_path))}
        decision = decision_cache[halt_path][sample_key]
        sample_id = sample_cache.resolve(packet)
        raw_item = data_cache.get(task, sample_id)
        selected_prediction = str(decision.get("selected_prediction", ""))
        final_prediction = str(decision.get("final_prediction", ""))
        prediction_stability_prediction = str(decision.get("prediction_stability_prediction", ""))
        selected_score = score_prediction(task, selected_prediction, raw_item, scorer)
        final_score = score_prediction(task, final_prediction, raw_item, scorer)
        prediction_stability_score = score_prediction(task, prediction_stability_prediction, raw_item, scorer)
        buckets[(task, sample_key)].append(
            {
                "packet": packet,
                "decision": decision,
                "task": task,
                "sample_key": sample_key,
                "sample_id": sample_id,
                "sender_seed": parse_seed(packet),
                "ground_truth": raw_item.get("ground_truth", raw_item.get("answer", "")),
                "choices": raw_item.get("choices", []),
                "selected_prediction": selected_prediction,
                "final_prediction": final_prediction,
                "prediction_stability_prediction": prediction_stability_prediction,
                "selected_score": float(selected_score["score"]),
                "selected_correct": bool(selected_score["correct"]),
                "final_score": float(final_score["score"]),
                "final_correct": bool(final_score["correct"]),
                "prediction_stability_score": float(prediction_stability_score["score"]),
                "prediction_stability_correct": bool(prediction_stability_score["correct"]),
                "scores": dict(packet.get("readiness_state", {}).get("scores", {})),
                "margins": dict(packet.get("readiness_state", {}).get("margins", {})),
            }
        )

    groups = []
    for (task, sample_key), members in sorted(buckets.items()):
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda row: (row["sender_seed"], row["selected_prediction"]))
        first = members[0]
        groups.append(
            {
                "task": task,
                "sample_key": sample_key,
                "sample_id": first["sample_id"],
                "ground_truth": first["ground_truth"],
                "choices": first["choices"],
                "members": members,
            }
        )
    return groups


def evaluate_group(group: dict[str, Any], scorer: Any) -> list[dict[str, Any]]:
    members = group["members"]
    methods: list[tuple[str, Callable[[list[dict[str, Any]]], dict[str, Any]], str]] = [
        ("single_sender_first", choose_first, "single_sender"),
        ("text_majority_selected", choose_text_majority_selected, "text_aggregation"),
        ("readiness_max_selected", lambda rows: choose_by_score(rows, "readiness", reverse=True), "latent_state_ranker"),
        ("correctness_head_max_selected", lambda rows: choose_by_score(rows, "correctness", reverse=True), "latent_state_ranker"),
        ("completion_risk_min_selected", lambda rows: choose_by_score(rows, "completion_risk", reverse=False), "latent_state_ranker"),
        ("future_gain_min_selected", lambda rows: choose_by_score(rows, "future_gain", reverse=False), "latent_state_ranker"),
        ("answer_identity_stability_max_selected", lambda rows: choose_by_score(rows, "answer_identity_stability", reverse=True), "latent_state_ranker"),
        ("prediction_change_min_selected", lambda rows: choose_by_score(rows, "prediction_change", reverse=False), "latent_state_ranker"),
        ("contentful_max_selected", lambda rows: choose_by_score(rows, "contentful", reverse=True), "latent_state_ranker"),
        ("oracle_any_selected_correct", choose_oracle_selected, "oracle_upper_bound"),
        ("oracle_any_final_correct", choose_oracle_final, "oracle_upper_bound"),
    ]
    out = []
    for method, chooser, method_type in methods:
        chosen = chooser(members)
        prediction = str(chosen["prediction"])
        score = score_text_with_official_rules(
            task=str(group["task"]),
            text=prediction,
            ground_truth=group["ground_truth"],
            choices=group["choices"],
            scorer=scorer,
        )
        out.append(
            {
                "task": group["task"],
                "sample_key": group["sample_key"],
                "sample_id": group["sample_id"],
                "method": method,
                "method_type": method_type,
                "correct": int(bool(score["correct"])),
                "score": float(score["score"]),
                "prediction": prediction,
                "processed_prediction": score["processed_generation"],
                "selected_sender_seed": chosen.get("sender_seed"),
                "selected_block": chosen.get("selected_block"),
                "vote_count": chosen.get("vote_count"),
                "group_size": len(members),
                "num_unique_selected_predictions": len({normalize_answer(row["selected_prediction"]) for row in members}),
                "num_selected_correct_senders": sum(int(bool(row["selected_correct"])) for row in members),
                "num_final_correct_senders": sum(int(bool(row["final_correct"])) for row in members),
            }
        )
    return out


def choose_first(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row = rows[0]
    return chosen_row(row, row["selected_prediction"])


def choose_text_majority_selected(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[normalize_answer(row["selected_prediction"])].append(row)
    best_rows = max(
        buckets.values(),
        key=lambda items: (len(items), mean(score_value(row, "readiness") for row in items), -min(row["sender_seed"] for row in items)),
    )
    row = max(best_rows, key=lambda item: (score_value(item, "readiness"), -item["sender_seed"]))
    chosen = chosen_row(row, row["selected_prediction"])
    chosen["vote_count"] = len(best_rows)
    return chosen


def choose_by_score(rows: list[dict[str, Any]], name: str, *, reverse: bool) -> dict[str, Any]:
    multiplier = 1.0 if reverse else -1.0
    row = max(rows, key=lambda item: (multiplier * score_value(item, name), -item["sender_seed"]))
    return chosen_row(row, row["selected_prediction"])


def choose_oracle_selected(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = [row for row in rows if row["selected_correct"]]
    row = correct[0] if correct else rows[0]
    return chosen_row(row, row["selected_prediction"])


def choose_oracle_final(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = [row for row in rows if row["final_correct"]]
    row = correct[0] if correct else rows[0]
    return chosen_row(row, row["final_prediction"])


def chosen_row(row: dict[str, Any], prediction: str) -> dict[str, Any]:
    return {
        "prediction": prediction,
        "sender_seed": row["sender_seed"],
        "selected_block": row["decision"].get("selected_block"),
    }


def score_value(row: dict[str, Any], name: str) -> float:
    value = row["scores"].get(name)
    if value is None:
        value = row["margins"].get(name)
    return float(value if value is not None else 0.0)


def score_prediction(task: str, prediction: str, raw_item: dict[str, Any], scorer: Any) -> dict[str, Any]:
    return score_text_with_official_rules(
        task=task,
        text=prediction,
        ground_truth=raw_item.get("ground_truth", raw_item.get("answer", "")),
        choices=raw_item.get("choices", []),
        scorer=scorer,
    )


def aggregate_metrics(rows: list[dict[str, Any]], *, by_task: bool = False) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["method"]), str(row["task"]) if by_task else "all")
        buckets[key].append(row)
    out = []
    for (method, task), items in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1])):
        count = len(items)
        out.append(
            {
                "method": method,
                "task": task,
                "count": count,
                "accuracy": mean(float(row["correct"]) for row in items),
                "mean_score": mean(float(row["score"]) for row in items),
                "mean_vote_count": mean(float(row["vote_count"] or 0) for row in items),
                "mean_unique_selected_predictions": mean(float(row["num_unique_selected_predictions"]) for row in items),
                "mean_selected_correct_senders": mean(float(row["num_selected_correct_senders"]) for row in items),
                "mean_final_correct_senders": mean(float(row["num_final_correct_senders"]) for row in items),
            }
        )
    return out


def summarize_groups(groups: list[dict[str, Any]]) -> dict[str, Any]:
    group_sizes = Counter(len(group["members"]) for group in groups)
    task_counts = Counter(str(group["task"]) for group in groups)
    unique_selected = [
        len({normalize_answer(row["selected_prediction"]) for row in group["members"]}) for group in groups
    ]
    selected_correct_counts = [
        sum(int(bool(row["selected_correct"])) for row in group["members"]) for group in groups
    ]
    final_correct_counts = [
        sum(int(bool(row["final_correct"])) for row in group["members"]) for group in groups
    ]
    return {
        "group_size_hist": dict(sorted(group_sizes.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "mean_unique_selected_predictions": mean(float(value) for value in unique_selected),
        "selected_correct_count_hist": dict(sorted(Counter(selected_correct_counts).items())),
        "final_correct_count_hist": dict(sorted(Counter(final_correct_counts).items())),
    }


def write_outputs(
    output_dir: Path,
    config: AggregationPotentialConfig,
    metrics: list[dict[str, Any]],
    per_task: list[dict[str, Any]],
    group_stats: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> dict[str, str]:
    metrics_jsonl = output_dir / "metrics.jsonl"
    with metrics_jsonl.open("w", encoding="utf-8") as handle:
        now = int(time.time())
        for row in metrics:
            handle.write(json.dumps({"created_at": now, "split": "all", "metrics": row}, sort_keys=True) + "\n")

    write_csv(output_dir / "aggregate_metrics.csv", metrics)
    write_csv(output_dir / "per_task_metrics.csv", per_task)
    predictions_path = output_dir / "aggregation_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    (output_dir / "group_stats.json").write_text(
        json.dumps(group_stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "summary_json": str(output_dir / "summary.json"),
        "config_json": str(output_dir / "config.json"),
        "metrics_jsonl": str(metrics_jsonl),
        "aggregate_metrics_csv": str(output_dir / "aggregate_metrics.csv"),
        "per_task_metrics_csv": str(output_dir / "per_task_metrics.csv"),
        "group_stats_json": str(output_dir / "group_stats.json"),
        "aggregation_predictions_jsonl": str(predictions_path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_seed(packet: dict[str, Any]) -> int:
    checkpoint = str(packet.get("agent_a", {}).get("checkpoint", ""))
    match = re.search(r"seed(\d+)", checkpoint)
    if match:
        return int(match.group(1))
    return 0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


if __name__ == "__main__":
    main()
