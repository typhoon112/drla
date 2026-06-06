"""Build paired Agent-A raw-text and latent messages for P2-D channel eval.

This is a local-only protocol builder.  It reads sanitized P2 latent packets
and attaches the raw Agent-A text emitted at the same trajectory/depth from the
native Cola block trace.  It must not use P1 ``selected_prediction`` as the
text message; that field is a scorer/task answer extraction, not the raw
agent-to-agent channel payload.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from drla.scripts.run_cola_sequential_latent_mas import OFFICIAL_COLA_TASKS
from drla.tracking import require_swanlab_disabled_for_non_training


@dataclass(frozen=True)
class AgentChannelMessageConfig:
    packets_jsonl: str = (
        "/data1/luyifei/drla/outputs/cola_agent_latent_comm/"
        "p2_agent_latent_comm_v2_locked_seed66_67_68_split20260601_20260529/"
        "agent_latent_comm_packets_test.jsonl"
    )
    output_dir: str = (
        "/data1/luyifei/drla/outputs/cola_agent_channel_eval/"
        "p2d_agent_channel_messages"
    )
    max_packets: int = 0
    max_packets_per_task: int = 0
    dedupe_sample_key: bool = False
    seed: int = 20260531
    swanlab_mode: str = "disabled"


def main() -> None:
    summary = build_agent_channel_messages(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> AgentChannelMessageConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets-jsonl", default=AgentChannelMessageConfig.packets_jsonl)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-packets", type=int, default=0)
    parser.add_argument("--max-packets-per-task", type=int, default=0)
    parser.add_argument("--dedupe-sample-key", action="store_true")
    parser.add_argument("--seed", type=int, default=AgentChannelMessageConfig.seed)
    parser.add_argument("--swanlab-mode", default=AgentChannelMessageConfig.swanlab_mode)
    args = parser.parse_args()
    if args.max_packets < 0 or args.max_packets_per_task < 0:
        raise ValueError("packet limits must be non-negative")
    return AgentChannelMessageConfig(
        packets_jsonl=args.packets_jsonl,
        output_dir=args.output_dir,
        max_packets=args.max_packets,
        max_packets_per_task=args.max_packets_per_task,
        dedupe_sample_key=bool(args.dedupe_sample_key),
        seed=args.seed,
        swanlab_mode=args.swanlab_mode,
    )


def build_agent_channel_messages(config: AgentChannelMessageConfig) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        config.swanlab_mode,
        script_kind="P2-D Agent channel message builder",
    )
    rng = random.Random(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    message_path = output_dir / "agent_channel_messages.jsonl"
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"

    packets_all = read_jsonl(Path(config.packets_jsonl))
    packets = select_packets(packets_all, config, rng)
    if not packets:
        raise ValueError("no packets selected")

    trace_cache = TraceCache()
    counts = {
        "packets_input": len(packets_all),
        "messages_written": 0,
        "missing_raw_text": 0,
        "selected_prediction_used_as_text": 0,
    }
    task_counts: dict[str, int] = defaultdict(int)
    examples: list[dict[str, Any]] = []

    with message_path.open("w", encoding="utf-8") as msg_f, metrics_path.open("w", encoding="utf-8") as metrics_f:
        for packet_index, packet in enumerate(packets):
            message = build_message(packet, packet_index, trace_cache, config.packets_jsonl)
            counts["messages_written"] += 1
            task_counts[str(message["task"])] += 1
            if not str(message["a_raw_text_message_t"]).strip():
                counts["missing_raw_text"] += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "sample_key": message["sample_key"],
                        "task": message["task"],
                        "handoff_depth": message["handoff_depth"],
                        "raw_text_preview": str(message["a_raw_text_message_t"])[:160],
                    },
                )
            msg_f.write(json.dumps(message, ensure_ascii=False, sort_keys=True) + "\n")
            metrics_f.write(
                json.dumps(
                    {
                        "created_at": int(time.time()),
                        "packet_index": packet_index,
                        "sample_key": message["sample_key"],
                        "task": message["task"],
                        "handoff_depth": message["handoff_depth"],
                        "raw_text_chars": len(str(message["a_raw_text_message_t"])),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
            )

    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "counts": counts,
        "task_counts": dict(sorted(task_counts.items())),
        "examples": examples,
        "artifacts": {
            "summary_json": str(summary_path),
            "metrics_jsonl": str(metrics_path),
            "agent_channel_messages_jsonl": str(message_path),
        },
        "online_input_audit": {
            "a_text_message_source": "native_trace.decode_text_so_far_at_handoff_depth",
            "selected_prediction_used_as_text": False,
            "status": "pass" if counts["missing_raw_text"] == 0 and counts["selected_prediction_used_as_text"] == 0 else "fail",
        },
        "interpretation": (
            "Paired channel messages for corrected P2-D evaluation. The raw text "
            "message is native trace decode_text_so_far at the same handoff depth "
            "as the latent packet; selected_prediction is not used as text input."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if summary["online_input_audit"]["status"] != "pass":
        raise RuntimeError("channel message online input audit failed")
    return summary


def build_message(
    packet: dict[str, Any],
    packet_index: int,
    trace_cache: "TraceCache",
    packets_jsonl: str,
) -> dict[str, Any]:
    task = str(packet["task"])
    sample_key = str(packet["sample_key"])
    sample_id = sample_id_from_key(sample_key)
    handoff_depth = int(packet["agent_a"]["selected_block"])
    trace_row = trace_cache.get(packet=packet, task=task, sample_id=sample_id, block_number=handoff_depth)
    if trace_row is None:
        raise KeyError(f"native trace row not found for {sample_key} block {handoff_depth}")
    raw_text = str(trace_row.get("decode_text_so_far", ""))
    if raw_text == "":
        raise ValueError(f"empty raw text message for {sample_key} block {handoff_depth}")
    return {
        "protocol_version": "cola_agent_channel_message_v1",
        "packet_index": packet_index,
        "packet_sample_key": sample_key,
        "sample_key": sample_key,
        "sample_id": sample_id,
        "task": task,
        "handoff_depth": handoff_depth,
        "max_block_budget": int(packet["agent_a"]["max_block_budget"]),
        "a_raw_text_message_t": raw_text,
        "a_raw_text_source": {
            "kind": "native_trace_decode_text_so_far",
            "trace_jsonl": str(trace_cache.trace_path_for_packet(packet, task)),
            "block_number": handoff_depth,
            "sample_id": sample_id,
        },
        "a_latent_packet_ref": {
            "packets_jsonl": packets_jsonl,
            "sample_key": sample_key,
            "packet_index_hint": packet_index,
            "latent_block_count": int(packet["latent_memory"]["block_count"]),
        },
        "audit_refs": {
            "halt_decisions_jsonl": packet["audit_refs"]["halt_decisions_jsonl"],
            "eval_summary_json": packet["audit_refs"]["eval_summary_json"],
        },
        "constraints": {
            "selected_prediction_used_as_text": False,
            "gold_or_scorer_used_online": False,
            "same_handoff_depth_as_latent": True,
        },
    }


class TraceCache:
    def __init__(self) -> None:
        self.cache: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}

    def get(self, *, packet: dict[str, Any], task: str, sample_id: Any, block_number: int) -> dict[str, Any] | None:
        trace_path = self.trace_path_for_packet(packet, task)
        key = str(trace_path)
        if key not in self.cache:
            rows = {}
            for row in read_jsonl(trace_path):
                rows[(str(row.get("sample_id")), int(row.get("block_number", 0)))] = row
            self.cache[key] = rows
        return self.cache[key].get((str(sample_id), int(block_number)))

    @staticmethod
    def trace_path_for_packet(packet: dict[str, Any], task: str) -> Path:
        blocks = packet["latent_memory"]["blocks"]
        if not blocks:
            raise ValueError(f"packet has no latent blocks: {packet.get('sample_key')}")
        latent_path = Path(blocks[-1]["latent_ref"]["path"])
        try:
            latents_index = latent_path.parts.index("latents")
        except ValueError as exc:
            raise ValueError(f"latent path lacks /latents/ segment: {latent_path}") from exc
        root = Path(*latent_path.parts[:latents_index])
        return root / f"{task}_traces.jsonl"


def select_packets(
    packets: list[dict[str, Any]],
    config: AgentChannelMessageConfig,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if config.max_packets_per_task > 0:
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for packet in packets:
            by_task[str(packet["task"])].append(packet)
        selected = []
        for task in OFFICIAL_COLA_TASKS:
            candidates = list(by_task.get(task, []))
            if config.dedupe_sample_key:
                candidates = dedupe_packets_by_sample_key(candidates, rng)
            if not candidates:
                continue
            selected.extend(rng.sample(candidates, min(config.max_packets_per_task, len(candidates))))
        selected.sort(key=lambda item: (str(item["task"]), str(item["sample_key"])))
        return selected[: config.max_packets] if config.max_packets else selected
    if config.dedupe_sample_key:
        packets = dedupe_packets_by_sample_key(packets, rng)
    if config.max_packets and config.max_packets < len(packets):
        return sorted(rng.sample(packets, config.max_packets), key=lambda item: str(item["sample_key"]))
    return packets


def dedupe_packets_by_sample_key(packets: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for packet in packets:
        grouped[str(packet["sample_key"])].append(packet)
    return [rng.choice(grouped[sample_key]) for sample_key in sorted(grouped)]


def sample_id_from_key(sample_key: str) -> int | str:
    suffix = sample_key.rsplit("::", 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return suffix


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
