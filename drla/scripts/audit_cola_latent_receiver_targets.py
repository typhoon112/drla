"""Audit offline accept/defer targets for P2 Cola latent receiver training.

This script is local-only.  It reads sanitized P2 packets plus their offline
halt-decision audit refs to measure whether the proposed receiver target is
statistically usable.  Offline correctness/loss fields are used only to create
this audit artifact; they must not become receiver inputs.
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


@dataclass(frozen=True)
class LatentReceiverTargetAuditConfig:
    packets_jsonl: str
    output_dir: str
    split_name: str = "test"
    max_packets: int = 0
    min_unsafe_events: int = 100
    min_unsafe_rate: float = 0.01
    max_unsafe_examples: int = 100


def main() -> None:
    summary = audit_receiver_targets(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> LatentReceiverTargetAuditConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--max-packets", type=int, default=0)
    parser.add_argument("--min-unsafe-events", type=int, default=100)
    parser.add_argument("--min-unsafe-rate", type=float, default=0.01)
    parser.add_argument("--max-unsafe-examples", type=int, default=100)
    args = parser.parse_args()
    if args.max_packets < 0:
        raise ValueError("max-packets must be non-negative")
    if args.min_unsafe_events < 0:
        raise ValueError("min-unsafe-events must be non-negative")
    if not 0.0 <= args.min_unsafe_rate <= 1.0:
        raise ValueError("min-unsafe-rate must be in [0, 1]")
    if args.max_unsafe_examples < 0:
        raise ValueError("max-unsafe-examples must be non-negative")
    return LatentReceiverTargetAuditConfig(
        packets_jsonl=args.packets_jsonl,
        output_dir=args.output_dir,
        split_name=args.split_name,
        max_packets=args.max_packets,
        min_unsafe_events=args.min_unsafe_events,
        min_unsafe_rate=args.min_unsafe_rate,
        max_unsafe_examples=args.max_unsafe_examples,
    )


def audit_receiver_targets(config: LatentReceiverTargetAuditConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    packets = load_packets(Path(config.packets_jsonl), config.max_packets)
    if not packets:
        raise ValueError("no packets loaded")

    decision_cache: dict[str, dict[str, dict[str, Any]]] = {}
    by_task: dict[str, dict[str, Any]] = defaultdict(init_bucket)
    by_selected_block: dict[int, dict[str, Any]] = defaultdict(init_bucket)
    by_receiver_action: dict[str, dict[str, Any]] = defaultdict(init_bucket)
    unsafe_examples: list[dict[str, Any]] = []
    counts = init_bucket()
    missing_decisions = 0

    for packet in packets:
        halt_path = str(packet["audit_refs"]["halt_decisions_jsonl"])
        if halt_path not in decision_cache:
            decision_cache[halt_path] = load_decisions(Path(halt_path))
        decision = decision_cache[halt_path].get(str(packet["sample_key"]))
        if decision is None:
            missing_decisions += 1
            continue
        target = target_from_decision(decision)
        update_bucket(counts, target)
        update_bucket(by_task[str(packet["task"])], target)
        update_bucket(by_selected_block[int(packet["agent_a"]["selected_block"])], target)
        update_bucket(by_receiver_action[str(packet["agent_b_contract"]["receiver_action"])], target)
        if target["unsafe"] and len(unsafe_examples) < config.max_unsafe_examples:
            unsafe_examples.append(
                {
                    "sample_key": packet["sample_key"],
                    "task": packet["task"],
                    "selected_block": packet["agent_a"]["selected_block"],
                    "final_block": decision["final_block"],
                    "loss_vs_final": target["loss_vs_final"],
                    "loss_vs_prediction_stability": target["loss_vs_prediction_stability"],
                    "receiver_action": packet["agent_b_contract"]["receiver_action"],
                    "halt_decisions_jsonl": halt_path,
                },
            )

    target_distribution_rows = []
    for name, buckets in [
        ("task", by_task),
        ("selected_block", by_selected_block),
        ("receiver_action", by_receiver_action),
    ]:
        for key, bucket in sorted(buckets.items(), key=lambda item: str(item[0])):
            row = bucket_to_row(bucket)
            row["group"] = name
            row["value"] = key
            target_distribution_rows.append(row)

    status = "pass"
    warnings = []
    unsafe_rate = safe_ratio(counts["unsafe"], counts["total"])
    if counts["unsafe"] < config.min_unsafe_events:
        warnings.append("unsafe_event_count_below_minimum")
        status = "warn"
    if unsafe_rate < config.min_unsafe_rate:
        warnings.append("unsafe_rate_below_minimum")
        status = "warn"
    if missing_decisions:
        warnings.append("missing_halt_decisions")
        status = "warn"

    paths = write_outputs(
        output_dir=output_dir,
        config=config,
        counts=counts,
        target_distribution_rows=target_distribution_rows,
        unsafe_examples=unsafe_examples,
    )
    summary = {
        "created_at": int(time.time()),
        "status": status,
        "warnings": warnings,
        "config": asdict(config),
        "inputs": {
            "packets_jsonl": str(Path(config.packets_jsonl)),
            "packets_loaded": len(packets),
            "halt_decision_files": len(decision_cache),
            "missing_decisions": missing_decisions,
        },
        "overall": bucket_to_row(counts),
        "by_task": {str(key): bucket_to_row(bucket) for key, bucket in sorted(by_task.items())},
        "artifacts": paths,
        "interpretation": (
            "A sparse unsafe target makes plain accept/defer BCE a weak P2-C main objective. "
            "Use accept/defer as an auxiliary rare-event/risk audit unless richer boundary "
            "examples or a balanced receiver-readability objective are constructed."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def load_packets(path: Path, max_packets: int) -> list[dict[str, Any]]:
    packets = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            packets.append(json.loads(line))
            if max_packets and len(packets) >= max_packets:
                break
    return packets


def load_decisions(path: Path) -> dict[str, dict[str, Any]]:
    decisions = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            decisions[str(row["sample_key"])] = row
    return decisions


def target_from_decision(decision: dict[str, Any]) -> dict[str, Any]:
    loss_vs_final = bool(decision.get("loss_vs_final", False))
    loss_vs_prediction_stability = bool(decision.get("loss_vs_prediction_stability", False))
    unsafe = loss_vs_final or loss_vs_prediction_stability
    return {
        "accept": not unsafe,
        "unsafe": unsafe,
        "loss_vs_final": loss_vs_final,
        "loss_vs_prediction_stability": loss_vs_prediction_stability,
        "loss_vs_both": loss_vs_final and loss_vs_prediction_stability,
    }


def init_bucket() -> dict[str, int]:
    return {
        "total": 0,
        "accept": 0,
        "unsafe": 0,
        "loss_vs_final": 0,
        "loss_vs_prediction_stability": 0,
        "loss_vs_both": 0,
    }


def update_bucket(bucket: dict[str, int], target: dict[str, Any]) -> None:
    bucket["total"] += 1
    for key in ["accept", "unsafe", "loss_vs_final", "loss_vs_prediction_stability", "loss_vs_both"]:
        bucket[key] += int(bool(target[key]))


def bucket_to_row(bucket: dict[str, int]) -> dict[str, Any]:
    total = bucket["total"]
    return {
        "total": total,
        "accept": bucket["accept"],
        "unsafe": bucket["unsafe"],
        "loss_vs_final": bucket["loss_vs_final"],
        "loss_vs_prediction_stability": bucket["loss_vs_prediction_stability"],
        "loss_vs_both": bucket["loss_vs_both"],
        "accept_rate": safe_ratio(bucket["accept"], total),
        "unsafe_rate": safe_ratio(bucket["unsafe"], total),
        "naive_accept_accuracy": safe_ratio(bucket["accept"], total),
    }


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def write_outputs(
    output_dir: Path,
    config: LatentReceiverTargetAuditConfig,
    counts: dict[str, int],
    target_distribution_rows: list[dict[str, Any]],
    unsafe_examples: list[dict[str, Any]],
) -> dict[str, str]:
    target_distribution_path = output_dir / "target_distribution.csv"
    unsafe_examples_path = output_dir / "unsafe_examples.jsonl"
    metrics_path = output_dir / "metrics.jsonl"
    write_csv(target_distribution_path, target_distribution_rows)
    with unsafe_examples_path.open("w", encoding="utf-8") as handle:
        for row in unsafe_examples:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metric = {
        "created_at": int(time.time()),
        "split_name": config.split_name,
        **bucket_to_row(counts),
    }
    metrics_path.write_text(
        json.dumps(metric, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "summary_json": str(output_dir / "summary.json"),
        "target_distribution_csv": str(target_distribution_path),
        "unsafe_examples_jsonl": str(unsafe_examples_path),
        "metrics_jsonl": str(metrics_path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
