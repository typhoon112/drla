"""Sample-level diagnostics for Cola continuation-risk gated halt policies."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from drla.scripts.eval_cola_adaptive_halt import group_rows, resolve_split_indices
from drla.scripts.eval_cola_risk_gated_halt import (
    RiskGatedHaltEvalConfig,
    choose_risk_gated_row,
    hydrate_from_eval_summary,
    materialize_rows,
    normalize_text,
    predict_readiness_for_rows,
    predict_risk_for_rows,
    single_choice_scope_to_params,
)
from drla.scripts.train_cola_readiness_model import (
    ReadinessTrainConfig,
    build_tensors,
    load_training_rows,
    resolve_device,
)


@dataclass(frozen=True)
class RiskGatedDecisionAnalysisConfig:
    eval_summary_path: str
    output_dir: str = "/data1/luyifei/drla/outputs/cola_risk_gated_halt_analysis/debug"
    max_examples: int = 25
    batch_size: int = 512
    device: str = "auto"


def analyze_risk_gated_decisions(config: RiskGatedDecisionAnalysisConfig) -> dict[str, Any]:
    summary = json.loads(Path(config.eval_summary_path).read_text(encoding="utf-8"))
    eval_config = hydrate_from_eval_summary(RiskGatedHaltEvalConfig(**summary["config"]))
    calibrated = summary["calibrated_risk_gated"]
    risk_threshold = float(calibrated["risk_threshold"])
    entropy_max = calibrated.get("entropy_max")
    top_prob_min = calibrated.get("top_prob_min")
    readiness_threshold = float(summary["readiness_threshold"])
    if "single_choice_guard_scope" in calibrated:
        require_stable_single_choice, stable_single_choice_max_block = single_choice_scope_to_params(
            calibrated["single_choice_guard_scope"]
        )
    else:
        require_stable_single_choice = eval_config.require_stable_single_choice
        stable_single_choice_max_block = eval_config.stable_single_choice_max_block

    readiness_checkpoint = torch.load(eval_config.readiness_checkpoint_path, map_location="cpu")
    readiness_config = ReadinessTrainConfig(**readiness_checkpoint["config"])
    feature_fields = readiness_checkpoint.get("feature_fields") or readiness_checkpoint.get("metadata", {}).get(
        "feature_fields"
    )
    labels_config = replace(
        readiness_config,
        labels_dir=eval_config.eval_labels_dir or readiness_config.labels_dir,
        tasks=eval_config.eval_tasks or readiness_config.tasks,
    )
    rows = load_training_rows(labels_config)
    tensors, metadata = build_tensors(rows, labels_config, feature_fields=feature_fields)
    indices = resolve_split_indices(metadata["sample_keys"], labels_config, eval_config.split)

    device = resolve_device(config.device)
    readiness_probs = predict_readiness_for_rows(
        checkpoint=readiness_checkpoint,
        train_config=readiness_config,
        tensors=tensors,
        indices=indices,
        device=device,
        batch_size=config.batch_size,
    )
    risk_checkpoint = torch.load(eval_config.risk_checkpoint_path, map_location="cpu")
    risk_probs = predict_risk_for_rows(
        checkpoint=risk_checkpoint,
        rows=rows,
        indices=indices,
        device=device,
        batch_size=config.batch_size,
    )
    eval_rows = materialize_rows(rows, metadata["sample_keys"], indices, readiness_probs, risk_probs)
    grouped = group_rows(eval_rows)

    records = []
    for sample_key, sample_rows in sorted(grouped.items()):
        sample_rows.sort(key=lambda item: int(item["block_index"]))
        (
            chosen,
            prefix_skips,
            shape_skips,
            fragment_skips,
            uncertainty_skips,
            single_choice_skips,
        ) = choose_risk_gated_row(
            sample_rows,
            readiness_threshold=readiness_threshold,
            risk_threshold=risk_threshold,
            entropy_max=entropy_max,
            top_prob_min=top_prob_min,
            use_oracle_prefix=False,
            require_contentful_prediction=eval_config.require_contentful_prediction,
            require_fragment_complete_prediction=eval_config.require_fragment_complete_prediction,
            require_stable_single_choice=require_stable_single_choice,
            stable_single_choice_max_block=stable_single_choice_max_block,
        )
        stability = choose_prediction_stability_row(sample_rows)
        final = sample_rows[-1]
        records.append(
            build_record(
                sample_key=sample_key,
                rows=sample_rows,
                chosen=chosen,
                stability=stability,
                final=final,
                prefix_skips=prefix_skips,
                shape_skips=shape_skips,
                fragment_skips=fragment_skips,
                uncertainty_skips=uncertainty_skips,
                single_choice_skips=single_choice_skips,
            )
        )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = output_dir / "risk_gated_decisions.jsonl"
    per_task_path = output_dir / "per_task_summary.csv"
    summary_path = output_dir / "summary.json"
    with decisions_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    per_task_rows = summarize_per_task(records)
    write_per_task_csv(per_task_path, per_task_rows)
    result = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "eval_summary_path": config.eval_summary_path,
        "swanlab_run_id": summary.get("swanlab_run_id"),
        "split": eval_config.split,
        "readiness_threshold": readiness_threshold,
        "risk_threshold": risk_threshold,
        "entropy_max": entropy_max,
        "top_prob_min": top_prob_min,
        "num_samples": len(records),
        "overall": summarize_records(records),
        "per_task_summary_csv": str(per_task_path),
        "risk_gated_decisions_jsonl": str(decisions_path),
        "loss_examples_vs_final": collect_examples(records, key="lost_final_correct", limit=config.max_examples),
        "loss_examples_vs_prediction_stability": collect_examples(
            records,
            key="lost_prediction_stability_correct",
            limit=config.max_examples,
        ),
        "gain_examples_vs_final": collect_examples(records, key="gained_over_final", limit=config.max_examples),
    }
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def choose_prediction_stability_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    previous = None
    streak = 0
    for row in rows:
        prediction = normalize_text(row.get("scored_prediction"))
        if prediction and prediction == previous:
            streak += 1
        else:
            streak = 1
            previous = prediction
        if prediction and streak >= 2:
            return row
    return rows[-1]


def build_record(
    *,
    sample_key: str,
    rows: list[dict[str, Any]],
    chosen: dict[str, Any],
    stability: dict[str, Any],
    final: dict[str, Any],
    prefix_skips: int,
    shape_skips: int,
    fragment_skips: int,
    uncertainty_skips: int,
    single_choice_skips: int,
) -> dict[str, Any]:
    chosen_prediction = normalize_text(chosen.get("scored_prediction"))
    final_prediction = normalize_text(final.get("scored_prediction"))
    stability_prediction = normalize_text(stability.get("scored_prediction"))
    final_correct = bool(final["official_correct"])
    stability_correct = bool(stability["official_correct"])
    chosen_correct = bool(chosen["official_correct"])
    return {
        "sample_key": sample_key,
        "task": final["task"],
        "sample_id": final["sample_id"],
        "target": final.get("scored_target"),
        "chosen": row_snapshot(chosen),
        "prediction_stability": row_snapshot(stability),
        "final": row_snapshot(final),
        "chosen_prediction_char_len": len(chosen_prediction),
        "chosen_prediction_word_count": len(chosen_prediction.split()),
        "chosen_is_strict_prefix_of_final": is_strict_prefix(chosen_prediction, final_prediction),
        "chosen_is_strict_prefix_of_stability": is_strict_prefix(chosen_prediction, stability_prediction),
        "chosen_has_terminal_punct": chosen_prediction.endswith((".", "!", "?", "\"", "'")),
        "chosen_ends_alnum": bool(chosen_prediction[-1:].isalnum()),
        "halted_before_final": int(chosen["block_index"]) < int(final["block_index"]),
        "halted_before_prediction_stability": int(chosen["block_index"]) < int(stability["block_index"]),
        "lost_final_correct": final_correct and not chosen_correct,
        "lost_prediction_stability_correct": stability_correct and not chosen_correct,
        "gained_over_final": chosen_correct and not final_correct,
        "gained_over_prediction_stability": chosen_correct and not stability_correct,
        "prefix_skips": prefix_skips,
        "shape_guard_skips": shape_skips,
        "fragment_guard_skips": fragment_skips,
        "uncertainty_guard_skips": uncertainty_skips,
        "single_choice_guard_skips": single_choice_skips,
        "blocks": [row_snapshot(row) for row in rows],
    }


def row_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "block_number": int(row["block_number"]),
        "block_index": int(row["block_index"]),
        "readiness_prob": float(row["readiness_prob"]),
        "continuation_risk_prob": float(row["continuation_risk_prob"]),
        "official_correct": bool(row["official_correct"]),
        "scored_prediction": row.get("scored_prediction"),
        "token_entropy_mean": row.get("token_entropy_mean"),
        "token_top_prob_mean": row.get("token_top_prob_mean"),
        "same_text_streak": row.get("same_text_streak"),
        "scored_prediction_same_streak": row.get("scored_prediction_same_streak"),
        "processed_generation_same_streak": row.get("processed_generation_same_streak"),
        "contains_stop": bool(row.get("contains_stop")),
        "contains_eos": bool(row.get("contains_eos")),
        "contains_im_end": bool(row.get("contains_im_end")),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, float]:
    n = max(len(records), 1)
    return {
        "accuracy": sum(int(item["chosen"]["official_correct"]) for item in records) / n,
        "fixed_final_accuracy": sum(int(item["final"]["official_correct"]) for item in records) / n,
        "prediction_stability_accuracy": sum(int(item["prediction_stability"]["official_correct"]) for item in records) / n,
        "avg_blocks": sum(int(item["chosen"]["block_number"]) for item in records) / n,
        "avg_prediction_stability_blocks": sum(
            int(item["prediction_stability"]["block_number"]) for item in records
        ) / n,
        "loss_count_vs_final": float(sum(int(item["lost_final_correct"]) for item in records)),
        "loss_count_vs_prediction_stability": float(
            sum(int(item["lost_prediction_stability_correct"]) for item in records)
        ),
        "gain_count_vs_final": float(sum(int(item["gained_over_final"]) for item in records)),
        "gain_count_vs_prediction_stability": float(
            sum(int(item["gained_over_prediction_stability"]) for item in records)
        ),
        "pre_stability_halt_rate": sum(int(item["halted_before_prediction_stability"]) for item in records) / n,
        "prefix_loss_vs_final_count": float(
            sum(int(item["lost_final_correct"] and item["chosen_is_strict_prefix_of_final"]) for item in records)
        ),
        "prefix_loss_vs_prediction_stability_count": float(
            sum(
                int(item["lost_prediction_stability_correct"] and item["chosen_is_strict_prefix_of_stability"])
                for item in records
            )
        ),
    }


def summarize_per_task(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["task"], []).append(record)
    rows = []
    for task, task_records in sorted(grouped.items()):
        rows.append({"task": task, **summarize_records(task_records)})
    return rows


def write_per_task_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "task",
        "accuracy",
        "fixed_final_accuracy",
        "prediction_stability_accuracy",
        "avg_blocks",
        "avg_prediction_stability_blocks",
        "loss_count_vs_final",
        "loss_count_vs_prediction_stability",
        "gain_count_vs_final",
        "gain_count_vs_prediction_stability",
        "pre_stability_halt_rate",
        "prefix_loss_vs_final_count",
        "prefix_loss_vs_prediction_stability_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_examples(records: list[dict[str, Any]], *, key: str, limit: int) -> list[dict[str, Any]]:
    examples = [record for record in records if record[key]]
    examples.sort(
        key=lambda item: (
            not item["halted_before_prediction_stability"],
            int(item["chosen"]["block_number"]),
            item["task"],
            str(item["sample_id"]),
        )
    )
    return [
        {
            "sample_key": item["sample_key"],
            "task": item["task"],
            "sample_id": item["sample_id"],
            "target": item["target"],
            "chosen_block": item["chosen"]["block_number"],
            "chosen_prediction": item["chosen"]["scored_prediction"],
            "stability_block": item["prediction_stability"]["block_number"],
            "stability_prediction": item["prediction_stability"]["scored_prediction"],
            "final_prediction": item["final"]["scored_prediction"],
            "chosen_readiness_prob": item["chosen"]["readiness_prob"],
            "chosen_risk_prob": item["chosen"]["continuation_risk_prob"],
            "prefix_final": item["chosen_is_strict_prefix_of_final"],
            "prefix_stability": item["chosen_is_strict_prefix_of_stability"],
        }
        for item in examples[:limit]
    ]


def is_strict_prefix(value: str, final_value: str) -> bool:
    return bool(value and final_value and value != final_value and final_value.startswith(value))


def parse_args() -> RiskGatedDecisionAnalysisConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-summary-path", required=True)
    parser.add_argument("--output-dir", default=RiskGatedDecisionAnalysisConfig.output_dir)
    parser.add_argument("--max-examples", type=int, default=RiskGatedDecisionAnalysisConfig.max_examples)
    parser.add_argument("--batch-size", type=int, default=RiskGatedDecisionAnalysisConfig.batch_size)
    parser.add_argument("--device", default=RiskGatedDecisionAnalysisConfig.device)
    args = parser.parse_args()
    return RiskGatedDecisionAnalysisConfig(
        eval_summary_path=args.eval_summary_path,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        device=args.device,
    )


def main() -> None:
    summary = analyze_risk_gated_decisions(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
