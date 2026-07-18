"""Audit receiver generation metrics by the checkpoint training split.

This local-only diagnostic reconstructs the train/valid/test split stored in a
D7 receiver checkpoint config and reports existing generation results by split.
It does not load Dream, generate, train, score new outputs, or create SwanLab
runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.scripts.p3_train_dream_latent_fuser import split_rows  # noqa: E402
from drla.scripts.p3_train_dream_layer_conditioned_receiver import LayerReceiverConfig  # noqa: E402
from drla.scripts.p3_train_dream_soft_prefix_adapter import load_training_rows, read_jsonl  # noqa: E402


DEFAULT_CHECKPOINT = (
    "/data1/luyifei/drla/outputs/p3_dream_layer_receivers/"
    "dream_layer_receiver_v7_v4init_zeroshuf_textmas_matched200_seed20260607_20260607/best_checkpoint.pt"
)
DEFAULT_RUN_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/"
    "dream_layer_receiver_v7_v4init_zeroshuf_eval_best200_merged_20260607"
)
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_receiver_generalization_audits/"
    "dream_receiver_v7_calibration_split_generalization_20260617"
)


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = LayerReceiverConfig(**checkpoint["config"])
    rows, row_metadata = load_training_rows(config)
    splits = split_rows(rows, config.seed, config.train_ratio, config.valid_ratio)
    split_by_row = {row["row_id"]: split for split, split_rows_ in splits.items() for row in split_rows_}
    split_sizes = {split: len(items) for split, items in splits.items()}

    generations_path = Path(args.run_dir) / "generations.jsonl"
    generations = read_jsonl(generations_path)
    records: list[dict[str, Any]] = []
    missing_split_rows = []
    for generation in generations:
        row_id = str(generation.get("row_id", ""))
        split = split_by_row.get(row_id)
        if split is None:
            missing_split_rows.append(row_id)
            split = "missing_split"
        record = {
            "split": split,
            "row_id": row_id,
            "sample_id": str(generation.get("sample_id", "")),
            "condition": canonical_condition(str(generation.get("condition", ""))),
            "primary_score": float(generation.get("primary_score", 0.0)),
            "exact_match": float(generation.get("exact_match", 0.0)),
            "token_f1": float(generation.get("token_f1", 0.0)),
            "prediction": str(generation.get("prediction", "")),
            "status": str(generation.get("status", "")),
        }
        records.append(record)

    condition_metrics = condition_summary(records)
    paired = paired_summary(records)
    hard_gate = hard_gate_by_split(paired)
    write_jsonl(output_dir / "rows.jsonl", records)
    write_jsonl(output_dir / "condition_metrics.jsonl", condition_metrics)
    write_jsonl(output_dir / "paired_metrics.jsonl", paired)
    metrics = {
        "num_generations": len(records),
        "num_missing_split_rows": len(set(missing_split_rows)),
        "hard_gate_all_nontrain_pass": all(
            item["hard_gate_pass"] for item in hard_gate if item["split"] in {"valid", "test"}
        ),
    }
    summary = {
        "created_at": int(time.time()),
        "status": "pass" if not missing_split_rows else "fail",
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint.get("step"),
        "run_dir": args.run_dir,
        "generations_jsonl": str(generations_path),
        "output_dir": str(output_dir),
        "split_sizes": split_sizes,
        "row_metadata": row_metadata,
        "metrics": metrics,
        "condition_metrics": condition_metrics,
        "paired_metrics": paired,
        "hard_gate_by_split": hard_gate,
        "missing_split_rows_preview": sorted(set(missing_split_rows))[:20],
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "metrics_jsonl": str(output_dir / "metrics.jsonl"),
            "rows_jsonl": str(output_dir / "rows.jsonl"),
            "condition_metrics_jsonl": str(output_dir / "condition_metrics.jsonl"),
            "paired_metrics_jsonl": str(output_dir / "paired_metrics.jsonl"),
        },
        "execution_boundary": [
            "local-only P3 Dream receiver split-generalization audit",
            "reads existing generation outputs only",
            "no Dream model loading or generation",
            "no optimizer, backward, or weight update",
            "no SwanLab run",
        ],
    }
    (output_dir / "metrics.jsonl").write_text(
        json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def canonical_condition(condition: str) -> str:
    for prefix in ("layer_receiver_", "text_interface_", "text_adapter_"):
        if condition.startswith(prefix):
            return condition[len(prefix) :]
    return condition


def condition_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[(record["split"], record["condition"])].append(record)
    rows = []
    for (split, condition), items in sorted(buckets.items()):
        rows.append(
            {
                "split": split,
                "condition": condition,
                "num_rows": len(items),
                "primary_score_mean": mean([item["primary_score"] for item in items]),
                "exact_match_mean": mean([item["exact_match"] for item in items]),
                "token_f1_mean": mean([item["token_f1"] for item in items]),
                "num_primary_correct": int(sum(item["primary_score"] for item in items)),
            }
        )
    return rows


def paired_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(item["split"], item["row_id"], item["condition"]): item for item in records}
    row_ids_by_split: dict[str, set[str]] = defaultdict(set)
    for item in records:
        row_ids_by_split[item["split"]].add(item["row_id"])
    comparisons = ["no_message", "zero", "shuffled_row", "agent_swap"]
    rows = []
    for split, row_ids in sorted(row_ids_by_split.items()):
        for control in comparisons:
            deltas = []
            wins = ties = losses = 0
            for row_id in row_ids:
                matched = by_key.get((split, row_id, "matched"))
                other = by_key.get((split, row_id, control))
                if matched is None or other is None:
                    continue
                delta = matched["primary_score"] - other["primary_score"]
                deltas.append(delta)
                if delta > 0:
                    wins += 1
                elif delta < 0:
                    losses += 1
                else:
                    ties += 1
            rows.append(
                {
                    "split": split,
                    "comparison": f"matched_minus_{control}",
                    "control_condition": control,
                    "num_paired": len(deltas),
                    "primary_delta_mean": mean(deltas),
                    "primary_win_count": wins,
                    "primary_tie_count": ties,
                    "primary_loss_count": losses,
                }
            )
    return rows


def hard_gate_by_split(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hard_controls = {"no_message", "zero", "shuffled_row"}
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in paired:
        if item["control_condition"] in hard_controls:
            by_split[item["split"]].append(item)
    return [
        {
            "split": split,
            "hard_gate_pass": bool(items) and all(item["primary_delta_mean"] > 0.0 for item in items),
            "deltas": {item["comparison"]: item["primary_delta_mean"] for item in items},
        }
        for split, items in sorted(by_split.items())
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
