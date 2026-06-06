"""Build decoder-free Agent A -> Agent B latent communication packets.

The packet is a protocol artifact, not a new training run.  It turns locked
P1 LatentHaltStudent decisions into Agent-A messages containing only latent
memory references, latent/process features, P1 student readiness state, and
fold-level calibration certificates.  Decoded text, scorer outputs, gold
answers, and prediction-stability references are kept out of the packet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PROCESS_FEATURE_FIELDS = [
    "block_number",
    "max_block_budget",
    "remaining_blocks",
    "block_fraction",
    "latent_norm_mean",
    "latent_norm_std",
    "latent_delta_norm",
    "latent_delta_missing",
    "latent_cosine_to_prev",
    "latent_cosine_missing",
    "denoise_drift_norm_mean",
]

FORBIDDEN_PACKET_KEYS = {
    "choices",
    "decode_text_so_far",
    "decode_token_ids_so_far",
    "final_correct",
    "final_prediction",
    "ground_truth",
    "latest_block_token_ids",
    "official_correct",
    "official_processed_generation",
    "official_score",
    "official_score_if_decodable",
    "prediction_stability_block",
    "prediction_stability_correct",
    "prediction_stability_prediction",
    "sample_id",
    "scored_prediction",
    "scored_target",
    "selected_correct",
    "selected_prediction",
}


@dataclass(frozen=True)
class AgentLatentCommPacketConfig:
    eval_roots: list[str] = field(default_factory=list)
    summary_jsons: list[str] = field(default_factory=list)
    output_dir: str = (
        "/data1/luyifei/drla/outputs/cola_agent_latent_comm/"
        "p2_agent_latent_comm_v1"
    )
    split_name: str = "test"
    protocol_version: str = "cola_agent_latent_comm_v1"
    communication_boundary: str = "single_handoff"
    sender_role: str = "solver"
    receiver_role: str = "solver"
    phase: str = "reasoning"
    prefix_contract: str = "shared_context_reencode"
    consume_mode: str = "replay_latent_blocks"
    max_packets_per_summary: int = 0
    strict_local_only: bool = True
    check_latent_files: bool = True


def main() -> None:
    summary = build_agent_latent_comm_packets(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> AgentLatentCommPacketConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", action="append", default=[])
    parser.add_argument("--summary-json", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", default="test")
    parser.add_argument(
        "--protocol-version",
        choices=["cola_agent_latent_comm_v1", "cola_agent_latent_comm_v2"],
        default="cola_agent_latent_comm_v1",
    )
    parser.add_argument(
        "--communication-boundary",
        choices=["single_handoff", "sequential_chain", "hierarchical_aggregation"],
        default="single_handoff",
    )
    parser.add_argument("--sender-role", default="solver")
    parser.add_argument(
        "--receiver-role",
        choices=["solver", "reviewer", "aggregator", "verifier"],
        default="solver",
    )
    parser.add_argument(
        "--phase",
        choices=["reasoning", "review", "aggregation", "verification"],
        default="reasoning",
    )
    parser.add_argument(
        "--prefix-contract",
        choices=["shared_context_reencode", "prefix_latent_ref", "kv_cache_ref"],
        default="shared_context_reencode",
    )
    parser.add_argument("--consume-mode", default="replay_latent_blocks")
    parser.add_argument("--max-packets-per-summary", type=int, default=0)
    parser.add_argument("--allow-nonlocal-eval", action="store_true")
    parser.add_argument("--skip-latent-file-check", action="store_true")
    args = parser.parse_args()
    if args.max_packets_per_summary < 0:
        raise ValueError("max_packets_per_summary must be non-negative")
    return AgentLatentCommPacketConfig(
        eval_roots=args.eval_root,
        summary_jsons=args.summary_json,
        output_dir=args.output_dir,
        split_name=args.split_name,
        protocol_version=args.protocol_version,
        communication_boundary=args.communication_boundary,
        sender_role=args.sender_role,
        receiver_role=args.receiver_role,
        phase=args.phase,
        prefix_contract=args.prefix_contract,
        consume_mode=args.consume_mode,
        max_packets_per_summary=args.max_packets_per_summary,
        strict_local_only=not args.allow_nonlocal_eval,
        check_latent_files=not args.skip_latent_file_check,
    )


def build_agent_latent_comm_packets(config: AgentLatentCommPacketConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    packet_path = output_dir / f"agent_latent_comm_packets_{config.split_name}.jsonl"
    metric_path = output_dir / "metrics.jsonl"
    schema_path = output_dir / "packet_schema.json"
    summary_path = output_dir / "summary.json"

    summary_paths = discover_summary_paths(config)
    if not summary_paths:
        raise FileNotFoundError("no eval summary.json files found")

    latent_exists_cache: dict[str, bool] = {}
    counts = init_counts()
    forbidden_examples: list[dict[str, Any]] = []
    missing_latent_files: list[str] = []
    task_counts: dict[str, int] = {}
    receiver_actions: dict[str, int] = {}
    calibration_unsatisfied = 0
    v2_field_coverage = init_v2_field_coverage()

    with packet_path.open("w", encoding="utf-8") as packet_f, metric_path.open(
        "w",
        encoding="utf-8",
    ) as metric_f:
        for summary_index, summary_json in enumerate(summary_paths):
            eval_summary = json.loads(summary_json.read_text(encoding="utf-8"))
            validate_eval_summary(eval_summary, summary_json, config)
            labels_dir = Path(eval_summary["train_config"]["labels_dir"])
            task = str(eval_summary["eval_tasks"])
            label_rows = load_label_rows(labels_dir, task)
            halt_decisions_path = Path(eval_summary["artifacts"][f"halt_decisions_{config.split_name}"])
            risk_certificate = risk_certificate_from_summary(eval_summary)
            if not risk_certificate["calibration_joint_risk_satisfied"]:
                calibration_unsatisfied += 1

            emitted_for_summary = 0
            for decision in read_jsonl(halt_decisions_path):
                packet = build_packet(
                    decision=decision,
                    label_rows=label_rows,
                    eval_summary=eval_summary,
                    summary_json=summary_json,
                    halt_decisions_path=halt_decisions_path,
                    risk_certificate=risk_certificate,
                    latent_exists_cache=latent_exists_cache,
                    check_latent_files=config.check_latent_files,
                    config=config,
                )
                counts["packets"] += 1
                counts["latent_blocks"] += packet["latent_memory"]["block_count"]
                task_counts[packet["task"]] = task_counts.get(packet["task"], 0) + 1
                action = packet_receiver_action(packet)
                receiver_actions[action] = receiver_actions.get(action, 0) + 1
                missing_latent_files.extend(packet.pop("_missing_latent_files"))
                update_v2_field_coverage(packet, v2_field_coverage)

                forbidden = find_forbidden_keys(packet, FORBIDDEN_PACKET_KEYS)
                if forbidden and len(forbidden_examples) < 20:
                    forbidden_examples.append(
                        {
                            "sample_key": packet["sample_key"],
                            "task": packet["task"],
                            "forbidden_keys": sorted(forbidden),
                        }
                    )
                counts["forbidden_key_hits"] += len(forbidden)
                packet_f.write(json.dumps(packet, ensure_ascii=False, sort_keys=True) + "\n")
                emitted_for_summary += 1
                if (
                    config.max_packets_per_summary
                    and emitted_for_summary >= config.max_packets_per_summary
                ):
                    break

            metric_f.write(
                json.dumps(
                    {
                        "created_at": int(time.time()),
                        "summary_index": summary_index,
                        "summary_json": str(summary_json),
                        "task": task,
                        "packets_emitted": emitted_for_summary,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    schema = packet_schema()
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    summary = {
        "created_at": int(time.time()),
        "protocol_version": config.protocol_version,
        "config": asdict(config),
        "num_eval_summaries": len(summary_paths),
        "counts": counts,
        "task_counts": task_counts,
        "receiver_actions": receiver_actions,
        "calibration_unsatisfied_summaries": calibration_unsatisfied,
        "v2_field_coverage": v2_field_coverage,
        "latent_file_check": {
            "enabled": config.check_latent_files,
            "unique_paths_checked": len(latent_exists_cache),
            "missing_count": len(set(missing_latent_files)),
            "missing_examples": sorted(set(missing_latent_files))[:20],
        },
        "online_input_audit": {
            "forbidden_key_hits": counts["forbidden_key_hits"],
            "forbidden_examples": forbidden_examples,
            "packet_forbidden_keys": sorted(FORBIDDEN_PACKET_KEYS),
            "status": "pass" if counts["forbidden_key_hits"] == 0 else "fail",
        },
        "artifacts": {
            "packets_jsonl": str(packet_path),
            "metrics_jsonl": str(metric_path),
            "packet_schema_json": str(schema_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if counts["forbidden_key_hits"]:
        raise ValueError("agent packet contains decoder/eval-only fields")
    if config.check_latent_files and missing_latent_files:
        raise FileNotFoundError("missing latent files in packet refs")
    return summary


def discover_summary_paths(config: AgentLatentCommPacketConfig) -> list[Path]:
    paths = [Path(path) for path in config.summary_jsons]
    patterns = [
        f"subseed*/leave_*_out_eval_*_{config.split_name}/summary.json",
        f"leave_*_out_eval_*_{config.split_name}/summary.json",
    ]
    for root_text in config.eval_roots:
        root = Path(root_text)
        for pattern in patterns:
            paths.extend(sorted(root.glob(pattern)))
    unique: dict[str, Path] = {}
    for path in paths:
        unique[str(path)] = path
    return list(unique.values())


def validate_eval_summary(
    summary: dict[str, Any],
    summary_json: Path,
    config: AgentLatentCommPacketConfig,
) -> None:
    if config.strict_local_only:
        if summary.get("swanlab_mode") != "disabled" or summary.get("swanlab_run_id") is not None:
            raise ValueError(f"eval summary must be local-only: {summary_json}")
    artifacts = summary.get("artifacts", {})
    key = f"halt_decisions_{config.split_name}"
    if key not in artifacts:
        raise KeyError(f"{summary_json} missing artifacts.{key}")
    if "train_config" not in summary or "labels_dir" not in summary["train_config"]:
        raise KeyError(f"{summary_json} missing train_config.labels_dir")


def load_label_rows(labels_dir: Path, task: str) -> dict[str, list[dict[str, Any]]]:
    path = labels_dir / f"{task}_readiness_labels.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        sample_key = f"{row['task']}::{row['sample_id']}"
        grouped.setdefault(sample_key, []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item["block_number"]))
    return grouped


def build_packet(
    decision: dict[str, Any],
    label_rows: dict[str, list[dict[str, Any]]],
    eval_summary: dict[str, Any],
    summary_json: Path,
    halt_decisions_path: Path,
    risk_certificate: dict[str, Any],
    latent_exists_cache: dict[str, bool],
    check_latent_files: bool,
    config: AgentLatentCommPacketConfig,
) -> dict[str, Any]:
    sample_key = str(decision["sample_key"])
    if sample_key not in label_rows:
        raise KeyError(f"missing label rows for {sample_key}")
    selected_block = int(decision["selected_block"])
    rows = [row for row in label_rows[sample_key] if int(row["block_number"]) <= selected_block]
    if len(rows) != selected_block:
        raise ValueError(f"{sample_key} has {len(rows)} rows for selected_block={selected_block}")
    latent_blocks, missing_latent_files = build_latent_blocks(
        rows,
        latent_exists_cache,
        check_latent_files,
    )
    agent_readiness_state = sanitize_readiness_state(decision["readiness_state"])
    receiver_hint = build_receiver_hint(agent_readiness_state, risk_certificate)
    base_packet = {
        "protocol_version": config.protocol_version,
        "created_at": int(time.time()),
        "sample_key": sample_key,
        "task": str(decision["task"]),
        "split_seed": eval_summary.get("split_seed"),
        "agent_a": {
            "name": "cola_p1_latent_halt_student",
            "checkpoint": eval_summary.get("checkpoint"),
            "selected_block": selected_block,
            "max_block_budget": int(decision["final_block"]),
            "halt_action": "halt" if agent_readiness_state["halt_candidate_found"] else "final_fallback",
        },
        "latent_memory": {
            "encoding": "cola_latent_block_refs",
            "block_count": len(latent_blocks),
            "blocks": latent_blocks,
        },
        "readiness_state": agent_readiness_state,
        "risk_certificate": risk_certificate,
        "audit_refs": {
            "eval_summary_json": str(summary_json),
            "halt_decisions_jsonl": str(halt_decisions_path),
            "labels_dir": eval_summary["train_config"]["labels_dir"],
        },
        "_missing_latent_files": missing_latent_files,
    }
    if config.protocol_version == "cola_agent_latent_comm_v1":
        base_packet["agent_b_receiver_hint"] = receiver_hint
        return base_packet
    base_packet["communication_boundary"] = build_communication_boundary(config)
    base_packet["prefix_contract"] = build_prefix_contract(
        rows=rows,
        decision=decision,
        eval_summary=eval_summary,
        config=config,
    )
    base_packet["agent_b_contract"] = build_agent_b_contract(
        receiver_hint=receiver_hint,
        config=config,
    )
    return base_packet


def build_latent_blocks(
    rows: list[dict[str, Any]],
    latent_exists_cache: dict[str, bool],
    check_latent_files: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    blocks = []
    missing_files = []
    for row in rows:
        latent_path = str(row["latent_batch_path"])
        if check_latent_files:
            exists = latent_exists_cache.get(latent_path)
            if exists is None:
                exists = Path(latent_path).exists()
                latent_exists_cache[latent_path] = exists
            if not exists:
                missing_files.append(latent_path)
        blocks.append(
            {
                "block_number": int(row["block_number"]),
                "latent_ref": {
                    "path": latent_path,
                    "batch_sample_index": int(row["latent_batch_sample_index"]),
                    "batch_block_index": int(row["latent_batch_block_index"]),
                    "shape": row["latent_block_shape"],
                },
                "process_features": process_features(row),
            }
        )
    return blocks, missing_files


def process_features(row: dict[str, Any]) -> dict[str, float | int]:
    features: dict[str, float | int] = {}
    max_block_budget = int(row["max_block_budget"])
    block_number = int(row["block_number"])
    derived = {
        "remaining_blocks": max_block_budget - block_number,
        "block_fraction": block_number / max(max_block_budget, 1),
        "latent_delta_missing": row.get("latent_delta_norm") is None,
        "latent_cosine_missing": row.get("latent_cosine_to_prev") is None,
    }
    for field_name in PROCESS_FEATURE_FIELDS:
        value = derived[field_name] if field_name in derived else row.get(field_name)
        if isinstance(value, bool):
            features[field_name] = int(value)
        elif value is None:
            features[field_name] = 0.0
        elif isinstance(value, int):
            features[field_name] = value
        else:
            features[field_name] = float(value)
    return features


def sanitize_readiness_state(state: dict[str, Any]) -> dict[str, Any]:
    sanitized = {
        "version": "cola_agent_latent_comm_v1_readiness_state",
        "source_version": state.get("version"),
        "halt_candidate_found": bool(state["halt_candidate_found"]),
        "fallback_to_final": bool(state["fallback_to_final"]),
        "selected_block": int(state["selected_block"]),
        "final_block": int(state["final_block"]),
        "scores": state["scores"],
        "thresholds": state["thresholds"],
        "margins": state["margins"],
        "score_source": "p1_student_heads_predicted_from_latent_prefix_and_process_features",
        "online_inputs": "latent_prefix_and_process_features_only",
        "stripped_eval_only_fields": ["prediction_stability_block"],
    }
    return sanitized


def risk_certificate_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    selected_valid = summary["selected_valid"]
    return {
        "calibration_scope": selected_valid.get("calibration_scope"),
        "loss_risk_target": selected_valid.get("calibration_loss_risk_target"),
        "mismatch_risk_target": selected_valid.get("calibration_mismatch_risk_target"),
        "risk_bound_z": selected_valid.get("calibration_risk_bound_z"),
        "loss_upper_max": selected_valid.get("loss_upper_max"),
        "mismatch_upper_max": selected_valid.get("mismatch_upper_max"),
        "loss_risk_satisfied": selected_valid.get("calibration_loss_risk_satisfied"),
        "mismatch_risk_satisfied": selected_valid.get("calibration_mismatch_risk_satisfied"),
        "calibration_joint_risk_satisfied": (
            selected_valid.get("calibration_loss_risk_satisfied") is True
            and selected_valid.get("calibration_mismatch_risk_satisfied") is True
        ),
        "selection_note": selected_valid.get("selection_note"),
    }


def build_receiver_hint(
    readiness_state: dict[str, Any],
    risk_certificate: dict[str, Any],
) -> dict[str, Any]:
    if not readiness_state["halt_candidate_found"]:
        action = "accept_final_budget_message"
    elif not risk_certificate["calibration_joint_risk_satisfied"]:
        action = "accept_with_uncertified_calibration"
    else:
        action = "accept_latent_message"
    return {
        "agent_b_v0_policy": "risk_certificate_aware_accept_or_flag",
        "action": action,
        "uses_decoder_online": False,
        "inputs_used": ["readiness_state", "risk_certificate"],
    }


def build_communication_boundary(config: AgentLatentCommPacketConfig) -> dict[str, str]:
    return {
        "pattern": config.communication_boundary,
        "handoff_mode": "one_shot",
        "sender_role": config.sender_role,
        "receiver_role": config.receiver_role,
        "phase": config.phase,
        "channel_substitution": "text_tokens_to_cola_latent_packet",
    }


def build_prefix_contract(
    rows: list[dict[str, Any]],
    decision: dict[str, Any],
    eval_summary: dict[str, Any],
    config: AgentLatentCommPacketConfig,
) -> dict[str, Any]:
    first = rows[0]
    latent_shape = list(first["latent_block_shape"])
    max_block_budget = int(decision["final_block"])
    config_digest = str(first.get("config_digest", ""))
    context_key = f"{decision['task']}::{decision['sample_key']}::{first.get('input_jsonl', '')}"
    sender_prompt_key = f"sender::{context_key}::{config.sender_role}"
    receiver_prompt_key = f"receiver::{context_key}::{config.receiver_role}::{config.consume_mode}"
    return {
        "mode": config.prefix_contract,
        "input_context_hash": stable_hash(context_key),
        "sender_prompt_hash": stable_hash(sender_prompt_key),
        "receiver_prompt_hash": stable_hash(receiver_prompt_key),
        "config_digest": config_digest,
        "model_id": "official_cola_dlm_same_substrate",
        "tokenizer_id": "cola_dlm_tokenizer",
        "vae_id": "cola_text_vae",
        "dit_id": "cola_dit",
        "block_size": latent_shape[0] if latent_shape else None,
        "patch_size": None,
        "latent_dim": latent_shape[-1] if latent_shape else None,
        "latent_scaling": None,
        "max_block_budget": max_block_budget,
        "consume_mode": config.consume_mode,
        "contract_note": "receiver re-encodes task context under shared Cola substrate; raw prompt text is not serialized in packet",
    }


def build_agent_b_contract(
    receiver_hint: dict[str, Any],
    config: AgentLatentCommPacketConfig,
) -> dict[str, Any]:
    return {
        "receiver_name": "cola_b_same_substrate_receiver",
        "consume_mode": config.consume_mode,
        "expected_payload": "cola_latent_block_refs",
        "uses_decoder_online": False,
        "receiver_action": receiver_hint["action"],
        "inputs_used": [
            "communication_boundary",
            "prefix_contract",
            "latent_memory",
            "readiness_state",
            "risk_certificate",
        ],
    }


def packet_receiver_action(packet: dict[str, Any]) -> str:
    if "agent_b_contract" in packet:
        return str(packet["agent_b_contract"]["receiver_action"])
    return str(packet["agent_b_receiver_hint"]["action"])


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def find_forbidden_keys(value: Any, forbidden: set[str]) -> set[str]:
    hits: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in forbidden:
                hits.add(key)
            hits.update(find_forbidden_keys(child, forbidden))
    elif isinstance(value, list):
        for child in value:
            hits.update(find_forbidden_keys(child, forbidden))
    return hits


def init_counts() -> dict[str, int]:
    return {
        "packets": 0,
        "latent_blocks": 0,
        "forbidden_key_hits": 0,
    }


def init_v2_field_coverage() -> dict[str, int]:
    return {
        "communication_boundary": 0,
        "prefix_contract": 0,
        "agent_b_contract": 0,
    }


def update_v2_field_coverage(packet: dict[str, Any], coverage: dict[str, int]) -> None:
    for key in coverage:
        if key in packet:
            coverage[key] += 1


def packet_schema() -> dict[str, Any]:
    return {
        "protocol_version": "cola_agent_latent_comm_v1_or_v2",
        "required_top_level_fields": [
            "sample_key",
            "task",
            "agent_a",
            "latent_memory",
            "readiness_state",
            "risk_certificate",
        ],
        "v2_required_top_level_fields": [
            "sample_key",
            "task",
            "communication_boundary",
            "prefix_contract",
            "agent_a",
            "latent_memory",
            "readiness_state",
            "risk_certificate",
            "agent_b_contract",
            "audit_refs",
        ],
        "online_allowed_inputs": [
            "communication_boundary",
            "prefix_contract",
            "latent_memory.blocks[*].latent_ref",
            "latent_memory.blocks[*].process_features",
            "readiness_state.scores",
            "readiness_state.thresholds",
            "readiness_state.margins",
            "risk_certificate",
            "agent_b_contract",
        ],
        "online_forbidden_inputs": sorted(FORBIDDEN_PACKET_KEYS),
    }


if __name__ == "__main__":
    main()
