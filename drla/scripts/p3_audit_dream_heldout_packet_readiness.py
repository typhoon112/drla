"""Preflight held-out D6 packet readiness for P3 Dream receiver eval.

This local-only audit checks whether the locked MuSiQue held-out split has the
latent packet substrate required by the D7 layer-conditioned receiver. It does
not train, load models, run generation, or create SwanLab runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_HELDOUT_MANIFEST_JSON = (
    "/data1/luyifei/drla/outputs/p2_phase_c_manifests/"
    "musique_heldout_manifest_800_seed20260605/manifest.json"
)
DEFAULT_HELDOUT_ONLINE_INPUTS_JSONL = (
    "/data1/luyifei/drla/outputs/p2_phase_c_control_inputs/"
    "musique_heldout_controls_800_seed20260605_v1_strict_wrong/online_inputs.jsonl"
)
DEFAULT_HELDOUT_TEXTMAS_AGGREGATE_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_textmas_aggregates/"
    "dream_textmas_gate_heldout800_maxctx4096_merged_20260606"
)
DEFAULT_CALIBRATION_TRACE_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_traces/"
    "musique_calibration_trace_textmas_matched200_steps64_stride4_hidden_tensor_merged_20260606"
)
DEFAULT_CALIBRATION_TRACE_ROOT = "/data1/luyifei/drla/outputs/p3_dream_traces"
DEFAULT_CALIBRATION_HIDDEN_GLOB = (
    "musique_calibration_trace_textmas_matched200_steps64_stride4_hidden_tensor_20260606_shard*/"
    "hidden_refs/*.pt"
)
DEFAULT_CALIBRATION_PACKET_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_latent_packets/"
    "dream_textmas_matched200_agent_ab_suffix_tensor_packets_v1_20260606"
)
DEFAULT_RECEIVER_CHECKPOINT = (
    "/data1/luyifei/drla/outputs/p3_dream_layer_receivers/"
    "dream_layer_receiver_v7_v4init_zeroshuf_textmas_matched200_seed20260607_20260607/"
    "best_checkpoint.pt"
)
DEFAULT_EXPECTED_HELDOUT_TRACE_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_traces/"
    "musique_heldout_trace_textmas_matched800_steps64_stride4_hidden_tensor_merged_20260617"
)
DEFAULT_EXPECTED_HELDOUT_PACKET_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_latent_packets/"
    "dream_textmas_heldout800_agent_ab_suffix_tensor_packets_v1_20260617"
)
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_heldout_packet_preflights/"
    "dream_heldout_packet_readiness_preflight_20260617"
)
REQUIRED_EVAL_CHECKS = {
    "heldout_manifest_exists",
    "heldout_online_inputs_exists",
    "heldout_textmas_aggregate_exists",
    "receiver_checkpoint_exists",
    "heldout_trace_ready",
    "heldout_packet_ready",
}


def main() -> None:
    summary = run_audit(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout-manifest-json", default=DEFAULT_HELDOUT_MANIFEST_JSON)
    parser.add_argument("--heldout-online-inputs-jsonl", default=DEFAULT_HELDOUT_ONLINE_INPUTS_JSONL)
    parser.add_argument("--heldout-textmas-aggregate-dir", default=DEFAULT_HELDOUT_TEXTMAS_AGGREGATE_DIR)
    parser.add_argument("--calibration-trace-dir", default=DEFAULT_CALIBRATION_TRACE_DIR)
    parser.add_argument("--calibration-trace-root", default=DEFAULT_CALIBRATION_TRACE_ROOT)
    parser.add_argument("--calibration-hidden-glob", default=DEFAULT_CALIBRATION_HIDDEN_GLOB)
    parser.add_argument("--calibration-packet-dir", default=DEFAULT_CALIBRATION_PACKET_DIR)
    parser.add_argument("--receiver-checkpoint", default=DEFAULT_RECEIVER_CHECKPOINT)
    parser.add_argument("--expected-heldout-trace-dir", default=DEFAULT_EXPECTED_HELDOUT_TRACE_DIR)
    parser.add_argument("--expected-heldout-packet-dir", default=DEFAULT_EXPECTED_HELDOUT_PACKET_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-free-gib-after-trace", type=float, default=50.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.heldout_manifest_json)
    online_path = Path(args.heldout_online_inputs_jsonl)
    aggregate_dir = Path(args.heldout_textmas_aggregate_dir)
    calibration_trace_dir = Path(args.calibration_trace_dir)
    calibration_packet_dir = Path(args.calibration_packet_dir)
    receiver_checkpoint = Path(args.receiver_checkpoint)
    expected_trace_dir = Path(args.expected_heldout_trace_dir)
    expected_packet_dir = Path(args.expected_heldout_packet_dir)

    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    heldout_condition_counts = count_conditions(online_path) if online_path.exists() else Counter()
    aggregate_condition_metrics = read_condition_metrics(aggregate_dir / "condition_metrics.csv")
    calibration_trace_summary = read_json(calibration_trace_dir / "summary.json") if (calibration_trace_dir / "summary.json").exists() else {}
    calibration_packet_summary = read_json(calibration_packet_dir / "summary.json") if (calibration_packet_dir / "summary.json").exists() else {}
    heldout_trace_summary = read_json(expected_trace_dir / "summary.json") if (expected_trace_dir / "summary.json").exists() else {}
    heldout_packet_summary = read_json(expected_packet_dir / "summary.json") if (expected_packet_dir / "summary.json").exists() else {}

    hidden_stats = calibration_hidden_stats(
        Path(args.calibration_trace_root),
        args.calibration_hidden_glob,
    )
    estimate = estimate_heldout_cost(
        heldout_matched_rows=int(heldout_condition_counts.get("textmas_matched", 0)),
        calibration_trace_summary=calibration_trace_summary,
        calibration_packet_summary=calibration_packet_summary,
        hidden_stats=hidden_stats,
        output_dir=output_dir,
        min_free_gib_after_trace=args.min_free_gib_after_trace,
    )

    checks = build_checks(
        manifest_path=manifest_path,
        online_path=online_path,
        aggregate_dir=aggregate_dir,
        calibration_trace_dir=calibration_trace_dir,
        calibration_packet_dir=calibration_packet_dir,
        receiver_checkpoint=receiver_checkpoint,
        expected_trace_dir=expected_trace_dir,
        expected_packet_dir=expected_packet_dir,
        heldout_trace_summary=heldout_trace_summary,
        heldout_packet_summary=heldout_packet_summary,
        estimate=estimate,
    )
    missing_required = [
        item for item in checks if item["check"] in REQUIRED_EVAL_CHECKS and not item["pass"]
    ]
    failed_advisory = [
        item for item in checks if item["check"] not in REQUIRED_EVAL_CHECKS and not item["pass"]
    ]
    can_run_v7_heldout_eval = not missing_required
    status = "ready" if can_run_v7_heldout_eval else "blocked"

    metrics = {
        "status_ready": int(status == "ready"),
        "heldout_num_manifest_samples": len(manifest.get("samples", [])),
        "heldout_online_textmas_matched_rows": int(heldout_condition_counts.get("textmas_matched", 0)),
        "calibration_hidden_total_gib": round(hidden_stats["total_bytes"] / 1024**3, 3),
        "estimated_heldout_trace_hidden_gib": estimate["estimated_trace_hidden_gib"],
        "estimated_heldout_packet_referenced_gib": estimate["estimated_packet_referenced_gib"],
        "free_gib_now": estimate["free_gib_now"],
        "estimated_free_gib_after_trace": estimate["estimated_free_gib_after_trace"],
        "num_missing_required_checks": len(missing_required),
        "num_failed_advisory_checks": len(failed_advisory),
    }
    write_jsonl(output_dir / "metrics.jsonl", [metrics])
    write_csv(output_dir / "checks.csv", checks)

    summary = {
        "created_at": int(time.time()),
        "status": status,
        "can_run_v7_heldout_eval": can_run_v7_heldout_eval,
        "paths": {
            "heldout_manifest_json": str(manifest_path),
            "heldout_online_inputs_jsonl": str(online_path),
            "heldout_textmas_aggregate_dir": str(aggregate_dir),
            "calibration_trace_dir": str(calibration_trace_dir),
            "calibration_packet_dir": str(calibration_packet_dir),
            "receiver_checkpoint": str(receiver_checkpoint),
            "expected_heldout_trace_dir": str(expected_trace_dir),
            "expected_heldout_packet_dir": str(expected_packet_dir),
        },
        "heldout_dataset": {
            "num_manifest_samples": len(manifest.get("samples", [])),
            "online_condition_counts": dict(sorted(heldout_condition_counts.items())),
            "textmas_aggregate_condition_metrics": aggregate_condition_metrics,
        },
        "calibration_reference": {
            "trace_summary": select_keys(calibration_trace_summary, ["status", "num_rows", "num_samples", "num_traces"]),
            "packet_metrics": calibration_packet_summary.get("metrics", {}),
            "hidden_file_stats": hidden_stats,
        },
        "existing_heldout_substrate": {
            "trace_summary": select_keys(heldout_trace_summary, ["status", "num_rows", "num_samples", "num_traces"]),
            "packet_metrics": heldout_packet_summary.get("metrics", {}),
        },
        "cost_estimate": estimate,
        "checks": checks,
        "missing_required_checks": missing_required,
        "failed_advisory_checks": failed_advisory,
        "next_required_artifacts": [
            str(expected_trace_dir) if not check_pass(checks, "heldout_trace_ready") else "",
            str(expected_packet_dir) if not check_pass(checks, "heldout_packet_ready") else "",
        ],
        "recommended_sequence": [
            "Run held-out textmas_matched Dream suffix_tensor trace shards only if disk/GPU budget is accepted.",
            "Merge held-out trace shards with p3_merge_dream_trace_shards.py.",
            "Build held-out D6 packets with p3_build_dream_latent_packets.py using the merged held-out trace.",
            "Run V7 layer-conditioned receiver held-out eval with no decoded Agent A/B text in solver prompt.",
            "Audit held-out generation controls with p3_audit_dream_receiver_generation_controls.py.",
        ],
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "metrics_jsonl": str(output_dir / "metrics.jsonl"),
            "checks_csv": str(output_dir / "checks.csv"),
        },
        "execution_boundary": [
            "local-only P3 held-out packet readiness audit",
            "no model loading or generation",
            "no optimizer, backward, or weight update",
            "no SwanLab run",
            "gold/scorer fields are not used for tuning",
        ],
    }
    summary["next_required_artifacts"] = [item for item in summary["next_required_artifacts"] if item]
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def count_conditions(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in read_jsonl_stream(path):
        counts[str(row.get("condition", ""))] += 1
    return counts


def read_condition_metrics(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {str(row.get("condition", "")): {key: parse_scalar(value) for key, value in row.items()} for row in reader}


def calibration_hidden_stats(root: Path, pattern: str) -> dict[str, Any]:
    total_bytes = 0
    count = 0
    for path in root.glob(pattern):
        if not path.is_file():
            continue
        count += 1
        total_bytes += path.stat().st_size
    return {
        "root": str(root),
        "glob": pattern,
        "num_files": count,
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / 1024**3, 3),
        "mean_file_bytes": round(total_bytes / count, 3) if count else 0.0,
    }


def estimate_heldout_cost(
    *,
    heldout_matched_rows: int,
    calibration_trace_summary: dict[str, Any],
    calibration_packet_summary: dict[str, Any],
    hidden_stats: dict[str, Any],
    output_dir: Path,
    min_free_gib_after_trace: float,
) -> dict[str, Any]:
    calibration_rows = int(calibration_trace_summary.get("num_rows", 0) or 0)
    bytes_per_row = hidden_stats["total_bytes"] / calibration_rows if calibration_rows else 0.0
    estimated_trace_bytes = bytes_per_row * heldout_matched_rows

    packet_metrics = calibration_packet_summary.get("metrics", {})
    calibration_packet_groups = int(packet_metrics.get("num_packet_groups", 0) or 0)
    referenced_per_group = (
        float(packet_metrics.get("total_referenced_hidden_bytes", 0) or 0) / calibration_packet_groups
        if calibration_packet_groups
        else 0.0
    )
    estimated_packet_referenced_bytes = referenced_per_group * heldout_matched_rows

    usage = shutil.disk_usage(output_dir)
    free_after = usage.free - estimated_trace_bytes
    enough_disk = free_after >= min_free_gib_after_trace * 1024**3
    return {
        "heldout_matched_rows": heldout_matched_rows,
        "calibration_rows": calibration_rows,
        "raw_trace_hidden_bytes_per_row": round(bytes_per_row, 3),
        "estimated_trace_hidden_bytes": round(estimated_trace_bytes),
        "estimated_trace_hidden_gib": round(estimated_trace_bytes / 1024**3, 3),
        "estimated_packet_referenced_bytes": round(estimated_packet_referenced_bytes),
        "estimated_packet_referenced_gib": round(estimated_packet_referenced_bytes / 1024**3, 3),
        "free_gib_now": round(usage.free / 1024**3, 3),
        "estimated_free_gib_after_trace": round(free_after / 1024**3, 3),
        "min_free_gib_after_trace": min_free_gib_after_trace,
        "disk_budget_pass": bool(enough_disk),
    }


def build_checks(
    *,
    manifest_path: Path,
    online_path: Path,
    aggregate_dir: Path,
    calibration_trace_dir: Path,
    calibration_packet_dir: Path,
    receiver_checkpoint: Path,
    expected_trace_dir: Path,
    expected_packet_dir: Path,
    heldout_trace_summary: dict[str, Any],
    heldout_packet_summary: dict[str, Any],
    estimate: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        check("heldout_manifest_exists", manifest_path.exists(), str(manifest_path)),
        check("heldout_online_inputs_exists", online_path.exists(), str(online_path)),
        check("heldout_textmas_aggregate_exists", (aggregate_dir / "summary.json").exists(), str(aggregate_dir)),
        check("calibration_trace_reference_exists", (calibration_trace_dir / "summary.json").exists(), str(calibration_trace_dir)),
        check("calibration_packet_reference_exists", (calibration_packet_dir / "summary.json").exists(), str(calibration_packet_dir)),
        check("receiver_checkpoint_exists", receiver_checkpoint.exists(), str(receiver_checkpoint)),
        check("heldout_trace_ready", heldout_trace_summary.get("status") == "pass", str(expected_trace_dir)),
        check("heldout_packet_ready", heldout_packet_summary.get("status") == "pass", str(expected_packet_dir)),
        check("disk_budget_for_full_trace", bool(estimate["disk_budget_pass"]), str(expected_trace_dir)),
    ]


def check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"check": name, "pass": bool(passed), "detail": detail}


def check_pass(checks: list[dict[str, Any]], name: str) -> bool:
    return any(item["check"] == name and item["pass"] for item in checks)


def select_keys(row: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: row.get(key) for key in keys if key in row}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def read_jsonl_stream(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            yield value


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_scalar(value: Any) -> Any:
    if value is None:
        return value
    text = str(value)
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return value


if __name__ == "__main__":
    main()
