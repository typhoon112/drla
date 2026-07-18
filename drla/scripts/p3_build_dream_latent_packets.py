"""Build D6 Dream latent packets from Agent A/B tensor traces.

This local-only script constructs audit-ready latent packet metadata from
``textmas_matched`` Dream traces. It uses the D5 student only to choose an
online step and to attach decoder-free readiness heads. It does not decode
packet tensors, train, score answers, or create SwanLab runs.
"""

from __future__ import annotations

import argparse
import json
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
)


DEFAULT_TRACE_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_traces/"
    "musique_calibration_trace_textmas_matched200_steps64_stride4_hidden_tensor_merged_20260606"
)
DEFAULT_POLICY_EVAL_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_readiness_policy_eval/"
    "dream_step_readiness_student_v1_full200_with_hidden_policy_eval_20260606"
)
DEFAULT_CHECKPOINT = (
    "/data1/luyifei/drla/outputs/p3_dream_readiness_students/"
    "dream_step_readiness_student_v1_full200_with_hidden_seed20260606_20260606/best_checkpoint.pt"
)
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_latent_packets/"
    "dream_textmas_matched200_agent_ab_suffix_tensor_packets_v1_20260606"
)

FORBIDDEN_PACKET_KEYS = {
    "decoded_text",
    "decoded_probe_text",
    "gold_answer",
    "answer_aliases",
    "scorer_output",
    "correctness",
    "step_prediction",
    "step_score",
    "step_primary_score",
    "final_prediction",
    "final_primary_score",
    "primary_score",
    "selected_answer_text",
}


@dataclass(frozen=True)
class PacketConfig:
    trace_dir: str = DEFAULT_TRACE_DIR
    policy_eval_dir: str = DEFAULT_POLICY_EVAL_DIR
    checkpoint: str = DEFAULT_CHECKPOINT
    output_dir: str = DEFAULT_OUTPUT_DIR
    device: str = "auto"
    agent_ids: str = "agent_a,agent_b"
    packet_version: str = "p3_dream_packet_v1_suffix_tensor"
    selection_mode: str = "d5_policy_transfer"
    overwrite: bool = False


def main() -> None:
    summary = build_packets(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> PacketConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", default=PacketConfig.trace_dir)
    parser.add_argument("--policy-eval-dir", default=PacketConfig.policy_eval_dir)
    parser.add_argument("--checkpoint", default=PacketConfig.checkpoint)
    parser.add_argument("--output-dir", default=PacketConfig.output_dir)
    parser.add_argument("--device", default=PacketConfig.device)
    parser.add_argument("--agent-ids", default=PacketConfig.agent_ids)
    parser.add_argument("--packet-version", default=PacketConfig.packet_version)
    parser.add_argument("--selection-mode", default=PacketConfig.selection_mode)
    parser.add_argument("--overwrite", action="store_true")
    return PacketConfig(**vars(parser.parse_args()))


def build_packets(config: PacketConfig) -> dict[str, Any]:
    created_at = int(time.time())
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(config.checkpoint, map_location="cpu")
    train_config = TrainConfig(**checkpoint["config"])
    policy_summary = json.loads((Path(config.policy_eval_dir) / "summary.json").read_text(encoding="utf-8"))
    selected_policy = policy_summary["selected_policy"]
    model = load_student(checkpoint, train_config, resolve_device(config.device))

    trace_dir = Path(config.trace_dir)
    trace_summary = json.loads((trace_dir / "summary.json").read_text(encoding="utf-8"))
    generations = read_jsonl(trace_dir / "generations.jsonl")
    traces = read_jsonl(trace_dir / "traces.jsonl")
    traces_by_call = {str(trace.get("call_id", "")): trace for trace in traces}
    target_agent_ids = {item.strip() for item in config.agent_ids.split(",") if item.strip()}

    packets = []
    packet_groups = []
    missing_refs = []
    missing_traces = []
    forbidden_hits = []
    for row in generations:
        if row.get("status") != "ok":
            continue
        row_packets = []
        for call_id in row.get("trace_call_ids", []):
            trace = traces_by_call.get(str(call_id))
            if trace is None:
                missing_traces.append({"row_id": row.get("row_id", ""), "call_id": call_id})
                continue
            if trace.get("agent_role") != "evidence_agent" or trace.get("agent_id") not in target_agent_ids:
                continue
            packet, ref_error = build_trace_packet(
                trace=trace,
                row=row,
                model=model,
                feature_stats=checkpoint["feature_stats"],
                selected_policy=selected_policy,
                config=config,
            )
            if ref_error:
                missing_refs.append(ref_error)
            hits = audit_forbidden_keys(packet)
            if hits:
                forbidden_hits.extend({"packet_id": packet["packet_id"], "key": key} for key in hits)
            packets.append(packet)
            row_packets.append(packet)
        packet_groups.append(
            {
                "row_id": row.get("row_id", ""),
                "sample_id": row.get("sample_id", ""),
                "condition": row.get("condition", ""),
                "packet_ids": [packet["packet_id"] for packet in row_packets],
                "packet_ids_by_agent": {packet["agent_id"]: packet["packet_id"] for packet in row_packets},
                "solver_call_id": str(row.get("trace_call_ids", [""])[-1]) if row.get("trace_call_ids") else "",
            }
        )

    write_jsonl(output_dir / "packets.jsonl", packets)
    write_jsonl(output_dir / "packet_groups.jsonl", packet_groups)
    metrics = compute_metrics(packets, packet_groups, missing_refs, missing_traces, forbidden_hits)
    (output_dir / "metrics.jsonl").write_text(
        json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    status = "pass" if not missing_refs and not missing_traces and not forbidden_hits else "fail"
    summary = {
        "created_at": created_at,
        "status": status,
        "config": asdict(config),
        "trace_summary": {
            "status": trace_summary.get("status"),
            "num_rows": trace_summary.get("num_rows"),
            "num_samples": trace_summary.get("num_samples"),
            "num_traces": trace_summary.get("num_traces"),
            "manifest_json": trace_summary.get("manifest_json"),
            "online_inputs_jsonl": trace_summary.get("online_inputs_jsonl"),
        },
        "policy_eval_summary": {
            "status": policy_summary.get("status"),
            "selected_policy": selected_policy,
            "policy_eval_dir": config.policy_eval_dir,
        },
        "checkpoint": config.checkpoint,
        "checkpoint_step": checkpoint.get("step"),
        "feature_names": FEATURE_NAMES,
        "metrics": metrics,
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "metrics_jsonl": str(output_dir / "metrics.jsonl"),
            "packets_jsonl": str(output_dir / "packets.jsonl"),
            "packet_groups_jsonl": str(output_dir / "packet_groups.jsonl"),
        },
        "execution_boundary": [
            "local-only P3 D6 latent packet construction",
            "no model generation",
            "no optimizer, backward, or weight update",
            "no SwanLab run",
            "packet tensors are referenced, not copied",
        ],
        "packet_online_payload_fields": [
            "hidden_ref",
            "hidden_shape",
            "hidden_dtype",
            "selected_step",
            "readiness_state",
            "process_state",
        ],
        "forbidden_packet_fields": sorted(FORBIDDEN_PACKET_KEYS),
        "scope_note": (
            "D5 policy was trained on solver readiness labels; applying it to "
            "evidence-agent traces is a D6 packet step-selection heuristic, not "
            "a new evidence-agent readiness claim."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def load_student(
    checkpoint: dict[str, Any],
    train_config: TrainConfig,
    device: torch.device,
) -> DreamStepReadinessStudent:
    model = DreamStepReadinessStudent(
        feature_dim=len(FEATURE_NAMES),
        d_model=train_config.d_model,
        num_layers=train_config.num_layers,
        num_heads=train_config.num_heads,
        dropout=train_config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


@torch.no_grad()
def build_trace_packet(
    trace: dict[str, Any],
    row: dict[str, Any],
    model: DreamStepReadinessStudent,
    feature_stats: dict[str, list[float]],
    selected_policy: dict[str, float],
    config: PacketConfig,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    device = next(model.parameters()).device
    events = sorted(trace.get("step_summaries", []) or [], key=lambda item: int(item.get("trace_event_index", 0)))
    enriched_events = []
    for event in events:
        enriched = dict(event)
        enriched["condition"] = row.get("condition", "")
        enriched_events.append(enriched)
    features = torch.tensor([event_features(event) for event in enriched_events], dtype=torch.float32, device=device)
    mean = torch.tensor(feature_stats["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(feature_stats["std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    outputs = model(((features - mean) / std).unsqueeze(0), torch.zeros((1, features.shape[0]), dtype=torch.bool, device=device))
    event_states = []
    for index, event in enumerate(enriched_events):
        event_states.append(
            {
                "event_index": index,
                "trace_event_index": int(event.get("trace_event_index", index)),
                "step": float(event.get("step") or 0.0),
                "hidden_ref": str(event.get("hidden_ref", "")),
                "ready_prob": float(torch.sigmoid(outputs["ready_logit"])[0, index].detach().cpu().item()),
                "final_match_prob": float(torch.sigmoid(outputs["final_match_logit"])[0, index].detach().cpu().item()),
                "prediction_change_prob": float(torch.sigmoid(outputs["prediction_change_logit"])[0, index].detach().cpu().item()),
                "future_gain_pred": float(outputs["future_gain"][0, index].detach().cpu().item()),
                "num_mask_tokens": event.get("num_mask_tokens", None),
                "changed_suffix_tokens_vs_prev_hook": event.get("changed_suffix_tokens_vs_prev_hook", None),
                "top1_prob_mean": event.get("top1_prob_mean", None),
                "top2_margin_mean": event.get("top2_margin_mean", None),
                "entropy_mean": event.get("entropy_mean", None),
            }
        )
    selected = select_event_state(event_states, selected_policy)
    tensor_meta, ref_error = tensor_metadata(selected["hidden_ref"], trace, row)
    packet_id = f"{row.get('row_id')}::{trace.get('agent_id')}::{trace.get('call_id')}::event{selected['trace_event_index']:04d}"
    packet = {
        "packet_id": packet_id,
        "packet_version": config.packet_version,
        "selection_mode": config.selection_mode,
        "row_id": row.get("row_id", ""),
        "sample_id": row.get("sample_id", ""),
        "condition": row.get("condition", ""),
        "agent_role": trace.get("agent_role", ""),
        "agent_id": trace.get("agent_id", ""),
        "source_call_id": trace.get("call_id", ""),
        "source_shard_dir": trace.get("source_shard_dir", ""),
        "selected_trace_event_index": selected["trace_event_index"],
        "selected_event_index": selected["event_index"],
        "selected_step": selected["step"],
        "hidden_ref": selected["hidden_ref"],
        **tensor_meta,
        "readiness_state": {
            "ready_prob": selected["ready_prob"],
            "final_match_prob": selected["final_match_prob"],
            "prediction_change_prob": selected["prediction_change_prob"],
            "future_gain_pred": selected["future_gain_pred"],
        },
        "process_state": {
            "num_mask_tokens": selected["num_mask_tokens"],
            "changed_suffix_tokens_vs_prev_hook": selected["changed_suffix_tokens_vs_prev_hook"],
            "top1_prob_mean": selected["top1_prob_mean"],
            "top2_margin_mean": selected["top2_margin_mean"],
            "entropy_mean": selected["entropy_mean"],
        },
    }
    return packet, ref_error


def select_event_state(events: list[dict[str, Any]], policy: dict[str, float]) -> dict[str, Any]:
    for event in events:
        if (
            event["ready_prob"] >= policy["ready_threshold"]
            and event["final_match_prob"] >= policy["final_match_threshold"]
            and event["prediction_change_prob"] <= policy["prediction_change_max"]
            and event["future_gain_pred"] <= policy["future_gain_max"]
            and event["hidden_ref"]
        ):
            return event
    for event in reversed(events):
        if event["hidden_ref"]:
            return event
    return events[-1]


def tensor_metadata(hidden_ref: str, trace: dict[str, Any], row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not hidden_ref:
        return {"hidden_shape": [], "hidden_dtype": "", "hidden_file_size_bytes": 0}, {
            "row_id": row.get("row_id", ""),
            "call_id": trace.get("call_id", ""),
            "reason": "empty_hidden_ref",
        }
    path = Path(hidden_ref)
    if not path.exists():
        return {"hidden_shape": [], "hidden_dtype": "", "hidden_file_size_bytes": 0}, {
            "row_id": row.get("row_id", ""),
            "call_id": trace.get("call_id", ""),
            "hidden_ref": hidden_ref,
            "reason": "missing_hidden_ref_file",
        }
    tensor = torch.load(path, map_location="cpu")
    if isinstance(tensor, dict) and "tensor" in tensor:
        hidden = tensor["tensor"]
    elif isinstance(tensor, dict) and "hidden" in tensor:
        hidden = tensor["hidden"]
    else:
        hidden = tensor
    return {
        "hidden_shape": list(hidden.shape) if torch.is_tensor(hidden) else [],
        "hidden_dtype": str(hidden.dtype) if torch.is_tensor(hidden) else "",
        "hidden_file_size_bytes": path.stat().st_size,
    }, None


def audit_forbidden_keys(value: Any, prefix: str = "") -> list[str]:
    hits = []
    if isinstance(value, dict):
        for key, child in value.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            if key in FORBIDDEN_PACKET_KEYS:
                hits.append(full_key)
            hits.extend(audit_forbidden_keys(child, full_key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(audit_forbidden_keys(child, f"{prefix}[{index}]"))
    return hits


def compute_metrics(
    packets: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    missing_refs: list[dict[str, Any]],
    missing_traces: list[dict[str, Any]],
    forbidden_hits: list[dict[str, str]],
) -> dict[str, Any]:
    selected_steps = [float(packet["selected_step"]) for packet in packets]
    hidden_sizes = [float(packet.get("hidden_file_size_bytes", 0)) for packet in packets]
    return {
        "num_packet_groups": len(groups),
        "num_packets": len(packets),
        "num_missing_refs": len(missing_refs),
        "num_missing_traces": len(missing_traces),
        "num_forbidden_key_hits": len(forbidden_hits),
        "missing_refs_preview": missing_refs[:10],
        "missing_traces_preview": missing_traces[:10],
        "forbidden_hits_preview": forbidden_hits[:10],
        "agent_counts": dict(Counter(packet.get("agent_id", "") for packet in packets)),
        "condition_counts": dict(Counter(packet.get("condition", "") for packet in packets)),
        "mean_selected_step": mean(selected_steps),
        "min_selected_step": min(selected_steps) if selected_steps else None,
        "max_selected_step": max(selected_steps) if selected_steps else None,
        "mean_hidden_file_size_bytes": mean(hidden_sizes),
        "total_referenced_hidden_bytes": sum(hidden_sizes),
        "packet_groups_with_two_agents": sum(1 for group in groups if len(group.get("packet_ids", [])) == 2),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
