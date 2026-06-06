"""Audit P2 Cola Agent A -> Agent B latent packet distribution.

This is a local-only audit.  It loads the latent refs inside sanitized P2
packets, checks that packet metadata matches the underlying trace shards, and
builds matched/corrupted controls for the next receiver-readability stage.
It does not train a receiver and does not use decoded text, gold answers, or
official scorer outputs as online packet inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from drla.scripts.build_cola_agent_latent_comm_packets import (
    FORBIDDEN_PACKET_KEYS,
    find_forbidden_keys,
)


DEFAULT_CONTROL_TYPES = [
    "matched",
    "metadata_only",
    "shuffle",
    "cross_task",
    "wrong_block",
    "noise",
    "rotation",
]

LATENT_STAT_FIELDS = [
    "element_mean",
    "element_std",
    "token_norm_mean",
    "token_norm_std",
    "l2_norm",
    "max_abs",
]


@dataclass(frozen=True)
class PacketDistributionAuditConfig:
    packets_jsonl: str
    output_dir: str
    num_control_samples: int = 0
    control_types: str = ",".join(DEFAULT_CONTROL_TYPES)
    max_packets: int = 0
    seed: int = 20260529
    expected_protocol_version: str = "cola_agent_latent_comm_v2"
    expected_latent_shape: str = "16,16"
    noise_std: float = 1.0
    max_cached_shards: int = 1024
    num_packet_examples: int = 20
    min_control_auroc: float = 0.95
    fail_on_audit_warnings: bool = False


class ShardCache:
    def __init__(self, max_cached_shards: int) -> None:
        self.max_cached_shards = max_cached_shards
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.loads = 0

    def load(self, path: str) -> dict[str, Any]:
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(obj, dict):
            raise ValueError(f"latent shard is not a dict: {path}")
        if "latent_blocks" not in obj:
            raise ValueError(f"latent shard lacks latent_blocks: {path}")
        self.cache[path] = obj
        self.loads += 1
        if self.max_cached_shards > 0:
            while len(self.cache) > self.max_cached_shards:
                self.cache.popitem(last=False)
        return obj


def main() -> None:
    summary = audit_packet_distribution(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if summary["status"] != "pass" and summary["config"]["fail_on_audit_warnings"]:
        raise SystemExit(1)


def parse_args() -> PacketDistributionAuditConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-control-samples", type=int, default=0)
    parser.add_argument(
        "--control-types",
        default=",".join(DEFAULT_CONTROL_TYPES),
        help="Comma-separated controls. matched is always added as the baseline.",
    )
    parser.add_argument("--max-packets", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument(
        "--expected-protocol-version",
        default="cola_agent_latent_comm_v2",
    )
    parser.add_argument("--expected-latent-shape", default="16,16")
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--max-cached-shards", type=int, default=1024)
    parser.add_argument("--num-packet-examples", type=int, default=20)
    parser.add_argument("--min-control-auroc", type=float, default=0.95)
    parser.add_argument("--fail-on-audit-warnings", action="store_true")
    args = parser.parse_args()
    if args.num_control_samples < 0:
        raise ValueError("num-control-samples must be non-negative")
    if args.max_packets < 0:
        raise ValueError("max-packets must be non-negative")
    if args.noise_std < 0:
        raise ValueError("noise-std must be non-negative")
    if args.max_cached_shards < 0:
        raise ValueError("max-cached-shards must be non-negative")
    if args.num_packet_examples < 0:
        raise ValueError("num-packet-examples must be non-negative")
    if not 0.0 <= args.min_control_auroc <= 1.0:
        raise ValueError("min-control-auroc must be in [0, 1]")
    return PacketDistributionAuditConfig(
        packets_jsonl=args.packets_jsonl,
        output_dir=args.output_dir,
        num_control_samples=args.num_control_samples,
        control_types=args.control_types,
        max_packets=args.max_packets,
        seed=args.seed,
        expected_protocol_version=args.expected_protocol_version,
        expected_latent_shape=args.expected_latent_shape,
        noise_std=args.noise_std,
        max_cached_shards=args.max_cached_shards,
        num_packet_examples=args.num_packet_examples,
        min_control_auroc=args.min_control_auroc,
        fail_on_audit_warnings=args.fail_on_audit_warnings,
    )


def audit_packet_distribution(config: PacketDistributionAuditConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    packets = load_packets(Path(config.packets_jsonl), config.max_packets)
    if not packets:
        raise ValueError("no packets loaded")

    rng = random.Random(config.seed)
    torch_generator = torch.Generator(device="cpu").manual_seed(config.seed)
    expected_shape = parse_shape(config.expected_latent_shape)
    control_types = normalize_control_types(config.control_types)
    audit_indices = choose_audit_indices(len(packets), config.num_control_samples, rng)
    packet_indexes = build_packet_indexes(packets)

    shard_cache = ShardCache(config.max_cached_shards)
    structural_counts = init_structural_counts()
    structural_examples: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    alignment_diffs: dict[str, list[float]] = defaultdict(list)
    distribution_accumulator: dict[tuple[str, str, int], dict[str, list[float]]] = defaultdict(
        make_metric_lists,
    )
    control_rows: list[dict[str, Any]] = []
    control_generation_warnings: list[dict[str, Any]] = []
    rotation_mats: dict[tuple[int, int], torch.Tensor] = {}

    with torch.no_grad():
        for packet_index in audit_indices:
            packet = packets[packet_index]
            structural = validate_packet_structure(
                packet=packet,
                packet_index=packet_index,
                expected_protocol_version=config.expected_protocol_version,
                expected_shape=expected_shape,
                shard_cache=shard_cache,
            )
            update_structural_counts(structural_counts, structural)
            if structural["errors"] and len(structural_examples) < 20:
                structural_examples.append(structural)

            matched_blocks = load_packet_blocks(packet, shard_cache)
            matched_block_stats = compute_block_stats(matched_blocks)
            record_native_alignment(packet, matched_blocks, matched_block_stats, alignment_diffs)
            for block_stat in matched_block_stats:
                add_distribution_stats(
                    distribution_accumulator,
                    control_type="matched",
                    task=packet["task"],
                    block_number=int(block_stat["block_number"]),
                    stats=block_stat,
                )

            if len(examples) < config.num_packet_examples:
                examples.append(
                    packet_example(
                        packet_index=packet_index,
                        packet=packet,
                        structural=structural,
                        matched_block_stats=matched_block_stats,
                    ),
                )

            matched_row = summarize_control_row(
                packet_index=packet_index,
                packet=packet,
                control_type="matched",
                source_packet=None,
                matched_blocks=matched_blocks,
                control_blocks=matched_blocks,
            )
            control_rows.append(matched_row)

            for control_type in control_types:
                if control_type == "matched":
                    continue
                control_blocks, source_packet, warning = build_control_blocks(
                    control_type=control_type,
                    packet_index=packet_index,
                    packet=packet,
                    packets=packets,
                    packet_indexes=packet_indexes,
                    matched_blocks=matched_blocks,
                    shard_cache=shard_cache,
                    rng=rng,
                    torch_generator=torch_generator,
                    noise_std=config.noise_std,
                    rotation_mats=rotation_mats,
                )
                if warning is not None and len(control_generation_warnings) < 50:
                    control_generation_warnings.append(warning)
                control_rows.append(
                    summarize_control_row(
                        packet_index=packet_index,
                        packet=packet,
                        control_type=control_type,
                        source_packet=source_packet,
                        matched_blocks=matched_blocks,
                        control_blocks=control_blocks,
                    ),
                )
                if control_blocks:
                    for block_stat in compute_block_stats(control_blocks):
                        add_distribution_stats(
                            distribution_accumulator,
                            control_type=control_type,
                            task=packet["task"],
                            block_number=int(block_stat["block_number"]),
                            stats=block_stat,
                        )

    distribution_rows = aggregate_distribution_stats(distribution_accumulator)
    ood_rows = compute_ood_rows(control_rows, config.min_control_auroc)
    paths = write_outputs(
        output_dir=output_dir,
        config=config,
        distribution_rows=distribution_rows,
        control_rows=control_rows,
        ood_rows=ood_rows,
        examples=examples,
    )
    alignment_summary = summarize_alignment_diffs(alignment_diffs)
    structural_pass = structural_counts["packets_with_errors"] == 0
    alignment_pass = alignment_summary["max_abs_diff"] <= 1e-5
    forbidden_pass = structural_counts["forbidden_key_hits"] == 0
    controls_pass = all(row["auroc_pass"] for row in ood_rows if row["control_type"] != "matched")
    status = "pass" if structural_pass and alignment_pass and forbidden_pass and controls_pass else "warn"
    summary = {
        "created_at": int(time.time()),
        "status": status,
        "config": asdict(config),
        "inputs": {
            "packets_jsonl": str(Path(config.packets_jsonl)),
            "packets_loaded": len(packets),
            "packets_audited": len(audit_indices),
            "control_types": control_types,
        },
        "structural": structural_counts,
        "native_alignment": alignment_summary,
        "control_generation_warnings": control_generation_warnings,
        "ood_detection": {
            "min_control_auroc": min(
                [row["auroc_pair_distance"] for row in ood_rows if row["control_type"] != "matched"]
                or [1.0],
            ),
            "min_required_auroc": config.min_control_auroc,
            "rows": ood_rows,
        },
        "shard_cache": {
            "loads": shard_cache.loads,
            "cached_shards": len(shard_cache.cache),
            "max_cached_shards": config.max_cached_shards,
        },
        "artifacts": paths,
        "interpretation": (
            "P2-B distribution audit checks packet/tensor consistency and corrupted-control "
            "separation. It does not prove Agent B can use the packet; that requires P2-C."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
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


def parse_shape(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("expected-latent-shape must not be empty")
    return tuple(int(part) for part in parts)


def normalize_control_types(value: str) -> list[str]:
    controls = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(controls) - set(DEFAULT_CONTROL_TYPES))
    if unknown:
        raise ValueError(f"unknown control types: {unknown}")
    if "matched" not in controls:
        controls.insert(0, "matched")
    return controls


def choose_audit_indices(num_packets: int, num_control_samples: int, rng: random.Random) -> list[int]:
    if num_control_samples <= 0 or num_control_samples >= num_packets:
        return list(range(num_packets))
    return sorted(rng.sample(range(num_packets), num_control_samples))


def build_packet_indexes(packets: list[dict[str, Any]]) -> dict[str, dict[Any, list[int]]]:
    by_task_block_count: dict[tuple[str, int], list[int]] = defaultdict(list)
    by_block_count: dict[int, list[int]] = defaultdict(list)
    by_task: dict[str, list[int]] = defaultdict(list)
    for index, packet in enumerate(packets):
        task = str(packet["task"])
        block_count = int(packet["latent_memory"]["block_count"])
        by_task_block_count[(task, block_count)].append(index)
        by_block_count[block_count].append(index)
        by_task[task].append(index)
    return {
        "by_task_block_count": dict(by_task_block_count),
        "by_block_count": dict(by_block_count),
        "by_task": dict(by_task),
    }


def init_structural_counts() -> dict[str, int]:
    return {
        "packets_checked": 0,
        "packets_with_errors": 0,
        "forbidden_key_hits": 0,
        "missing_top_level_fields": 0,
        "protocol_mismatch": 0,
        "block_count_mismatch": 0,
        "selected_block_mismatch": 0,
        "shape_mismatch": 0,
        "budget_mismatch": 0,
        "latent_ref_load_errors": 0,
        "task_mismatch": 0,
        "config_digest_mismatch": 0,
        "sample_id_mismatch": 0,
    }


def validate_packet_structure(
    packet: dict[str, Any],
    packet_index: int,
    expected_protocol_version: str,
    expected_shape: tuple[int, ...],
    shard_cache: ShardCache,
) -> dict[str, Any]:
    errors: list[str] = []
    required = [
        "protocol_version",
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
    ]
    missing = [key for key in required if key not in packet]
    if missing:
        errors.append(f"missing_top_level_fields:{','.join(missing)}")
    if packet.get("protocol_version") != expected_protocol_version:
        errors.append("protocol_mismatch")

    forbidden = find_forbidden_keys(packet, FORBIDDEN_PACKET_KEYS)
    if forbidden:
        errors.append(f"forbidden_keys:{','.join(sorted(forbidden))}")

    latent_memory = packet.get("latent_memory", {})
    blocks = latent_memory.get("blocks", [])
    block_count = int(latent_memory.get("block_count", -1))
    selected_block = int(packet.get("agent_a", {}).get("selected_block", -1))
    readiness_selected = int(packet.get("readiness_state", {}).get("selected_block", -1))
    max_budget = int(packet.get("agent_a", {}).get("max_block_budget", -1))
    prefix_budget = int(packet.get("prefix_contract", {}).get("max_block_budget", -1))
    if block_count != len(blocks):
        errors.append("block_count_mismatch")
    if selected_block != block_count or readiness_selected != selected_block:
        errors.append("selected_block_mismatch")
    if selected_block > max_budget or max_budget != prefix_budget:
        errors.append("budget_mismatch")

    for offset, block in enumerate(blocks):
        ref = block.get("latent_ref", {})
        ref_shape = tuple(ref.get("shape", []))
        block_number = int(block.get("block_number", -1))
        if ref_shape != expected_shape or ref_shape != tuple(block.get("process_features", {}).get("latent_block_shape", ref_shape)):
            errors.append("shape_mismatch")
        if block_number != offset + 1:
            errors.append("block_number_mismatch")
        try:
            shard = shard_cache.load(str(ref["path"]))
            latent_blocks = shard["latent_blocks"]
            sample_index = int(ref["batch_sample_index"])
            block_index = int(ref["batch_block_index"])
            if sample_index < 0 or sample_index >= latent_blocks.shape[0]:
                raise IndexError("batch_sample_index out of range")
            if block_index < 0 or block_index >= latent_blocks.shape[1]:
                raise IndexError("batch_block_index out of range")
            z = latent_blocks[sample_index, block_index]
            if tuple(z.shape) != ref_shape:
                errors.append("shape_mismatch")
            if str(shard.get("task")) != str(packet.get("task")):
                errors.append("task_mismatch")
            shard_digest = str(shard.get("config_digest", ""))
            packet_digest = str(packet.get("prefix_contract", {}).get("config_digest", ""))
            if shard_digest and packet_digest and shard_digest != packet_digest:
                errors.append("config_digest_mismatch")
            expected_sample_id = sample_id_from_key(str(packet.get("sample_key", "")))
            sample_ids = shard.get("sample_ids", [])
            if expected_sample_id is not None and sample_index < len(sample_ids):
                if int(sample_ids[sample_index]) != expected_sample_id:
                    errors.append("sample_id_mismatch")
        except Exception as exc:  # noqa: BLE001 - keep audit examples compact.
            errors.append(f"latent_ref_load_error:{type(exc).__name__}:{exc}")

    return {
        "packet_index": packet_index,
        "sample_key": packet.get("sample_key"),
        "task": packet.get("task"),
        "errors": sorted(set(errors)),
        "forbidden_key_count": len(forbidden),
    }


def sample_id_from_key(sample_key: str) -> int | None:
    if "::" not in sample_key:
        return None
    tail = sample_key.rsplit("::", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return None


def update_structural_counts(counts: dict[str, int], structural: dict[str, Any]) -> None:
    counts["packets_checked"] += 1
    if structural["errors"]:
        counts["packets_with_errors"] += 1
    counts["forbidden_key_hits"] += int(structural["forbidden_key_count"])
    for error in structural["errors"]:
        for key in [
            "missing_top_level_fields",
            "protocol_mismatch",
            "block_count_mismatch",
            "selected_block_mismatch",
            "shape_mismatch",
            "budget_mismatch",
            "latent_ref_load_errors",
            "task_mismatch",
            "config_digest_mismatch",
            "sample_id_mismatch",
        ]:
            if error.startswith(key) or error.startswith(key.rstrip("s")):
                counts[key] += 1


def load_packet_blocks(packet: dict[str, Any], shard_cache: ShardCache) -> list[torch.Tensor]:
    blocks = []
    for block in packet["latent_memory"]["blocks"]:
        blocks.append(load_block_from_ref(block["latent_ref"], shard_cache))
    return blocks


def load_block_from_ref(
    ref: dict[str, Any],
    shard_cache: ShardCache,
    override_block_index: int | None = None,
) -> torch.Tensor:
    shard = shard_cache.load(str(ref["path"]))
    block_index = int(ref["batch_block_index"] if override_block_index is None else override_block_index)
    z = shard["latent_blocks"][int(ref["batch_sample_index"]), block_index]
    return z.detach().cpu().float().clone()


def compute_block_stats(blocks: list[torch.Tensor]) -> list[dict[str, float]]:
    stats = []
    previous = None
    for index, block in enumerate(blocks):
        token_norm = block.norm(dim=-1)
        row = {
            "block_number": float(index + 1),
            "element_mean": float(block.mean()),
            "element_std": float(block.std(unbiased=False)),
            "token_norm_mean": float(token_norm.mean()),
            "token_norm_std": float(token_norm.std(unbiased=False)),
            "l2_norm": float(block.norm()),
            "max_abs": float(block.abs().max()),
            "delta_norm": 0.0,
            "cosine_to_prev": 0.0,
        }
        if previous is not None:
            row["delta_norm"] = float((block - previous).norm())
            row["cosine_to_prev"] = float(
                torch.nn.functional.cosine_similarity(
                    block.reshape(1, -1),
                    previous.reshape(1, -1),
                )[0],
            )
        stats.append(row)
        previous = block
    return stats


def record_native_alignment(
    packet: dict[str, Any],
    blocks: list[torch.Tensor],
    block_stats: list[dict[str, float]],
    alignment_diffs: dict[str, list[float]],
) -> None:
    _ = blocks
    packet_blocks = packet["latent_memory"]["blocks"]
    for packet_block, stats in zip(packet_blocks, block_stats, strict=True):
        features = packet_block["process_features"]
        compare_named(alignment_diffs, "latent_norm_mean", stats["token_norm_mean"], features)
        compare_named(alignment_diffs, "latent_norm_std", stats["token_norm_std"], features)
        compare_named(alignment_diffs, "latent_delta_norm", stats["delta_norm"], features)
        compare_named(alignment_diffs, "latent_cosine_to_prev", stats["cosine_to_prev"], features)


def compare_named(
    alignment_diffs: dict[str, list[float]],
    field: str,
    value: float,
    features: dict[str, Any],
) -> None:
    if field not in features or features[field] is None:
        return
    alignment_diffs[field].append(abs(float(features[field]) - value))


def make_metric_lists() -> dict[str, list[float]]:
    fields = list(LATENT_STAT_FIELDS) + ["delta_norm", "cosine_to_prev"]
    return {field: [] for field in fields}


def add_distribution_stats(
    accumulator: dict[tuple[str, str, int], dict[str, list[float]]],
    control_type: str,
    task: str,
    block_number: int,
    stats: dict[str, float],
) -> None:
    bucket = accumulator[(control_type, task, block_number)]
    for field, values in bucket.items():
        if field in stats:
            values.append(float(stats[field]))


def build_control_blocks(
    control_type: str,
    packet_index: int,
    packet: dict[str, Any],
    packets: list[dict[str, Any]],
    packet_indexes: dict[str, dict[Any, list[int]]],
    matched_blocks: list[torch.Tensor],
    shard_cache: ShardCache,
    rng: random.Random,
    torch_generator: torch.Generator,
    noise_std: float,
    rotation_mats: dict[tuple[int, int], torch.Tensor],
) -> tuple[list[torch.Tensor] | None, dict[str, Any] | None, dict[str, Any] | None]:
    if control_type == "metadata_only":
        return None, None, None
    if control_type == "noise":
        noisy = []
        for block in matched_blocks:
            scale = block.std(unbiased=False).clamp_min(1e-8)
            noisy.append(block + torch.randn(block.shape, generator=torch_generator) * scale * noise_std)
        return noisy, packet, None
    if control_type == "rotation":
        rotated = [rotate_block(block, rotation_mats, torch_generator) for block in matched_blocks]
        return rotated, packet, None
    if control_type == "wrong_block":
        wrong_blocks = []
        warnings = []
        for block in packet["latent_memory"]["blocks"]:
            ref = block["latent_ref"]
            shard = shard_cache.load(str(ref["path"]))
            num_blocks = int(shard["latent_blocks"].shape[1])
            current = int(ref["batch_block_index"])
            replacement = current + 1 if current + 1 < num_blocks else current - 1
            if replacement < 0:
                replacement = current
                warnings.append("no_alternate_block")
            wrong_blocks.append(load_block_from_ref(ref, shard_cache, override_block_index=replacement))
        warning = None
        if warnings:
            warning = {
                "packet_index": packet_index,
                "sample_key": packet["sample_key"],
                "control_type": control_type,
                "warnings": sorted(set(warnings)),
            }
        return wrong_blocks, packet, warning
    source_index, warning = choose_source_packet(
        control_type=control_type,
        packet_index=packet_index,
        packet=packet,
        packets=packets,
        packet_indexes=packet_indexes,
        rng=rng,
    )
    if source_index is None:
        return matched_blocks, packet, warning
    source_packet = packets[source_index]
    return load_packet_blocks(source_packet, shard_cache), source_packet, warning


def choose_source_packet(
    control_type: str,
    packet_index: int,
    packet: dict[str, Any],
    packets: list[dict[str, Any]],
    packet_indexes: dict[str, dict[Any, list[int]]],
    rng: random.Random,
) -> tuple[int | None, dict[str, Any] | None]:
    task = str(packet["task"])
    block_count = int(packet["latent_memory"]["block_count"])
    if control_type == "shuffle":
        candidates = [
            idx
            for idx in packet_indexes["by_task_block_count"].get((task, block_count), [])
            if idx != packet_index
        ]
    elif control_type == "cross_task":
        candidates = [
            idx
            for idx in packet_indexes["by_block_count"].get(block_count, [])
            if idx != packet_index and str(packets[idx]["task"]) != task
        ]
    else:
        raise ValueError(f"unsupported replacement control: {control_type}")
    warning = None
    if not candidates:
        candidates = [idx for idx in packet_indexes["by_block_count"].get(block_count, []) if idx != packet_index]
        warning = {
            "packet_index": packet_index,
            "sample_key": packet["sample_key"],
            "control_type": control_type,
            "warnings": ["fell_back_to_any_task_same_block_count"],
        }
    if not candidates:
        warning = {
            "packet_index": packet_index,
            "sample_key": packet["sample_key"],
            "control_type": control_type,
            "warnings": ["no_replacement_available"],
        }
        return None, warning
    return rng.choice(candidates), warning


def rotate_block(
    block: torch.Tensor,
    rotation_mats: dict[tuple[int, int], torch.Tensor],
    torch_generator: torch.Generator,
) -> torch.Tensor:
    shape = tuple(block.shape)
    if len(shape) != 2:
        raise ValueError(f"rotation control expects 2D latent block, got {shape}")
    key = (shape[-1], shape[-1])
    if key not in rotation_mats:
        matrix = torch.randn(key, generator=torch_generator)
        q, _ = torch.linalg.qr(matrix)
        rotation_mats[key] = q.float()
    return block @ rotation_mats[key]


def summarize_control_row(
    packet_index: int,
    packet: dict[str, Any],
    control_type: str,
    source_packet: dict[str, Any] | None,
    matched_blocks: list[torch.Tensor],
    control_blocks: list[torch.Tensor] | None,
) -> dict[str, Any]:
    source_sample_key = "" if source_packet is None else str(source_packet["sample_key"])
    source_task = "" if source_packet is None else str(source_packet["task"])
    row = {
        "packet_index": packet_index,
        "sample_key": packet["sample_key"],
        "task": packet["task"],
        "block_count": int(packet["latent_memory"]["block_count"]),
        "control_type": control_type,
        "payload_present": control_blocks is not None,
        "source_sample_key": source_sample_key,
        "source_task": source_task,
        "l2_distance_to_matched_mean": "",
        "l2_distance_per_element_mean": "",
        "cosine_to_matched_mean": "",
        "token_norm_ratio_mean": "",
        "pair_ood_score": "",
    }
    if control_blocks is None:
        row["pair_ood_score"] = math.inf
        return row
    distances = []
    per_element_distances = []
    cosines = []
    norm_ratios = []
    for matched, control in zip(matched_blocks, control_blocks, strict=True):
        diff = control - matched
        distances.append(float(diff.norm()))
        per_element_distances.append(float(diff.norm() / math.sqrt(float(diff.numel()))))
        cosines.append(float(torch.nn.functional.cosine_similarity(control.reshape(1, -1), matched.reshape(1, -1))[0]))
        matched_norm = float(matched.norm())
        control_norm = float(control.norm())
        norm_ratios.append(control_norm / matched_norm if matched_norm else math.inf)
    row["l2_distance_to_matched_mean"] = mean(distances)
    row["l2_distance_per_element_mean"] = mean(per_element_distances)
    row["cosine_to_matched_mean"] = mean(cosines)
    row["token_norm_ratio_mean"] = mean(norm_ratios)
    row["pair_ood_score"] = mean(per_element_distances)
    return row


def aggregate_distribution_stats(
    accumulator: dict[tuple[str, str, int], dict[str, list[float]]],
) -> list[dict[str, Any]]:
    rows = []
    for (control_type, task, block_number), values_by_field in sorted(accumulator.items()):
        count = len(next(iter(values_by_field.values()))) if values_by_field else 0
        row: dict[str, Any] = {
            "control_type": control_type,
            "task": task,
            "block_number": block_number,
            "count": count,
        }
        for field, values in values_by_field.items():
            row[f"{field}_mean"] = mean(values)
            row[f"{field}_std"] = population_std(values)
            row[f"{field}_min"] = min(values) if values else ""
            row[f"{field}_max"] = max(values) if values else ""
        rows.append(row)
    return rows


def compute_ood_rows(control_rows: list[dict[str, Any]], min_control_auroc: float) -> list[dict[str, Any]]:
    matched_scores = [
        finite_score(row["pair_ood_score"])
        for row in control_rows
        if row["control_type"] == "matched"
    ]
    rows = [
        {
            "control_type": "matched",
            "num_matched": len(matched_scores),
            "num_control": 0,
            "matched_score_mean": mean(matched_scores),
            "control_score_mean": "",
            "auroc_pair_distance": 0.5,
            "auroc_pass": True,
        },
    ]
    for control_type in sorted({row["control_type"] for row in control_rows if row["control_type"] != "matched"}):
        control_scores = [
            finite_score(row["pair_ood_score"])
            for row in control_rows
            if row["control_type"] == control_type
        ]
        auroc = rank_auc(matched_scores, control_scores)
        rows.append(
            {
                "control_type": control_type,
                "num_matched": len(matched_scores),
                "num_control": len(control_scores),
                "matched_score_mean": mean(matched_scores),
                "control_score_mean": mean(control_scores),
                "auroc_pair_distance": auroc,
                "auroc_pass": auroc >= min_control_auroc,
            },
        )
    return rows


def finite_score(value: Any) -> float:
    if value == "":
        return 1e9
    score = float(value)
    if math.isfinite(score):
        return score
    return 1e9


def rank_auc(negative_scores: list[float], positive_scores: list[float]) -> float:
    if not negative_scores or not positive_scores:
        return 0.5
    values = [(score, 0) for score in negative_scores] + [(score, 1) for score in positive_scores]
    values.sort(key=lambda item: item[0])
    rank_sum_pos = 0.0
    index = 0
    rank = 1
    while index < len(values):
        end = index + 1
        while end < len(values) and values[end][0] == values[index][0]:
            end += 1
        avg_rank = (rank + rank + (end - index) - 1) / 2.0
        for _, label in values[index:end]:
            if label == 1:
                rank_sum_pos += avg_rank
        rank += end - index
        index = end
    n_pos = len(positive_scores)
    n_neg = len(negative_scores)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def summarize_alignment_diffs(alignment_diffs: dict[str, list[float]]) -> dict[str, Any]:
    per_field = {}
    max_abs_diff = 0.0
    for field, values in sorted(alignment_diffs.items()):
        field_max = max(values) if values else 0.0
        max_abs_diff = max(max_abs_diff, field_max)
        per_field[field] = {
            "count": len(values),
            "mean_abs_diff": mean(values),
            "max_abs_diff": field_max,
        }
    return {
        "max_abs_diff": max_abs_diff,
        "per_field": per_field,
    }


def packet_example(
    packet_index: int,
    packet: dict[str, Any],
    structural: dict[str, Any],
    matched_block_stats: list[dict[str, float]],
) -> dict[str, Any]:
    return {
        "packet_index": packet_index,
        "sample_key": packet["sample_key"],
        "task": packet["task"],
        "selected_block": packet["agent_a"]["selected_block"],
        "block_count": packet["latent_memory"]["block_count"],
        "receiver_action": packet["agent_b_contract"]["receiver_action"],
        "structural_errors": structural["errors"],
        "first_block_stats": matched_block_stats[0] if matched_block_stats else {},
        "latent_refs": [
            {
                "block_number": block["block_number"],
                "path": block["latent_ref"]["path"],
                "batch_sample_index": block["latent_ref"]["batch_sample_index"],
                "batch_block_index": block["latent_ref"]["batch_block_index"],
            }
            for block in packet["latent_memory"]["blocks"]
        ],
    }


def write_outputs(
    output_dir: Path,
    config: PacketDistributionAuditConfig,
    distribution_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    ood_rows: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> dict[str, str]:
    distribution_path = output_dir / "distribution_stats.csv"
    control_path = output_dir / "control_stats.csv"
    ood_path = output_dir / "ood_detection.csv"
    examples_path = output_dir / "packet_examples.jsonl"
    metrics_path = output_dir / "metrics.jsonl"

    write_csv(distribution_path, distribution_rows)
    write_csv(control_path, control_rows)
    write_csv(ood_path, ood_rows)
    with examples_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")
    with metrics_path.open("w", encoding="utf-8") as handle:
        for row in ood_rows:
            metric = {
                "created_at": int(time.time()),
                "metric_type": "ood_detection",
                "control_type": row["control_type"],
                "auroc_pair_distance": row["auroc_pair_distance"],
                "auroc_pass": row["auroc_pass"],
                "min_required_auroc": config.min_control_auroc,
            }
            handle.write(json.dumps(metric, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "summary_json": str(output_dir / "summary.json"),
        "distribution_stats_csv": str(distribution_path),
        "control_stats_csv": str(control_path),
        "ood_detection_csv": str(ood_path),
        "packet_examples_jsonl": str(examples_path),
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


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / len(values))


if __name__ == "__main__":
    main()
