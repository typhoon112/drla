"""Merge sharded corrected P2-D Agent-B channel-eval outputs.

The sharded runner writes one ``generations.jsonl`` per message range. This
local-only helper combines those rows, validates duplicate/missing channel rows,
and writes a single eval root that can be passed to
``aggregate_cola_channel_eval.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from drla.tracking import require_swanlab_disabled_for_non_training


@dataclass(frozen=True)
class MergeChannelEvalShardsConfig:
    shard_root: str
    output_dir: str
    expected_messages: int = 0
    expected_channels: int = 0
    swanlab_mode: str = "disabled"


def main() -> None:
    summary = merge_channel_eval_shards(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> MergeChannelEvalShardsConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-messages", type=int, default=0)
    parser.add_argument("--expected-channels", type=int, default=0)
    parser.add_argument("--swanlab-mode", default=MergeChannelEvalShardsConfig.swanlab_mode)
    args = parser.parse_args()
    if args.expected_messages < 0 or args.expected_channels < 0:
        raise ValueError("expected counts must be non-negative")
    return MergeChannelEvalShardsConfig(
        shard_root=args.shard_root,
        output_dir=args.output_dir,
        expected_messages=args.expected_messages,
        expected_channels=args.expected_channels,
        swanlab_mode=args.swanlab_mode,
    )


def merge_channel_eval_shards(config: MergeChannelEvalShardsConfig) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        config.swanlab_mode,
        script_kind="P2-D Agent-B channel-eval shard merge",
    )
    shard_root = Path(config.shard_root)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dirs = sorted(path for path in shard_root.iterdir() if path.is_dir() and path.name.startswith("shard"))
    if not shard_dirs:
        raise FileNotFoundError(f"no shard directories found in {shard_root}")

    rows: list[dict[str, Any]] = []
    shard_summaries = []
    for shard_dir in shard_dirs:
        generations_path = shard_dir / "generations.jsonl"
        summary_path = shard_dir / "summary.json"
        if not generations_path.exists():
            raise FileNotFoundError(generations_path)
        shard_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        message_start = int(shard_summary.get("config", {}).get("message_start") or 0)
        shard_rows = normalize_shard_message_index(read_jsonl(generations_path), message_start)
        rows.extend(shard_rows)
        shard_summaries.append(
            {
                "shard": shard_dir.name,
                "generations": len(shard_rows),
                "summary_num_generations": shard_summary.get("num_generations"),
                "message_start": message_start,
                "message_end": shard_summary.get("config", {}).get("message_end"),
                "elapsed_seconds": shard_summary.get("elapsed_seconds"),
            },
        )

    rows.sort(key=lambda row: (int(row.get("message_index", -1)), str(row.get("channel", ""))))
    duplicate_keys = find_duplicate_keys(rows)
    message_channels = count_message_channels(rows)
    channel_counts = Counter(str(row.get("channel", "")) for row in rows)
    sample_counts = Counter(str(row.get("sample_key", "")) for row in rows)
    row_contracts = sorted({str(row.get("agent_b_input_contract", "")) for row in rows if row.get("agent_b_input_contract")})
    row_scopes = sorted({str(row.get("score_output_scope", "")) for row in rows if row.get("score_output_scope")})
    missing_message_rows = []
    if config.expected_channels:
        missing_message_rows = [
            {"message_index": message_index, "channel_count": count}
            for message_index, count in sorted(message_channels.items())
            if count != config.expected_channels
        ]
    if duplicate_keys:
        raise RuntimeError(f"duplicate message/channel rows found: {duplicate_keys[:5]}")
    if missing_message_rows:
        raise RuntimeError(f"message rows with unexpected channel count: {missing_message_rows[:10]}")
    if config.expected_messages and len(message_channels) != config.expected_messages:
        raise RuntimeError(
            f"expected {config.expected_messages} messages, found {len(message_channels)}",
        )
    if config.expected_channels and len(rows) != len(message_channels) * config.expected_channels:
        raise RuntimeError("row count does not equal messages * channels")

    generations_path = output_dir / "generations.jsonl"
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(generations_path, rows)
    summary = {
        "created_at": int(time.time()),
        "config": {
            **asdict(config),
            "agent_b_input_contract": row_contracts[0] if len(row_contracts) == 1 else "",
            "score_output_scope": row_scopes[0] if len(row_scopes) == 1 else "",
        },
        "shard_root": str(shard_root),
        "shards": shard_summaries,
        "num_generations": len(rows),
        "num_messages": len(message_channels),
        "channels": sorted(channel_counts),
        "channel_counts": dict(sorted(channel_counts.items())),
        "unique_samples": len(sample_counts),
        "duplicate_keys": 0,
        "missing_message_rows": 0,
        "artifacts": {
            "summary_json": str(summary_path),
            "metrics_jsonl": str(metrics_path),
            "generations_jsonl": str(generations_path),
        },
        "interpretation": (
            "Merged corrected P2-D Agent-B channel-eval shards. This file is "
            "ready for aggregate_cola_channel_eval.py."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with metrics_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"created_at": int(time.time()), "metrics": summary}, sort_keys=True) + "\n")
    return summary


def find_duplicate_keys(rows: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        counts[
            (
                str(row.get("sample_key", "")),
                str(row.get("message_index", "")),
                str(row.get("channel", "")),
            )
        ] += 1
    return [key for key, count in counts.items() if count > 1]


def normalize_shard_message_index(rows: list[dict[str, Any]], message_start: int) -> list[dict[str, Any]]:
    if message_start == 0:
        return rows
    output = []
    for row in rows:
        item = dict(row)
        item["message_index"] = int(item.get("message_index", 0)) + message_start
        output.append(item)
    return output


def count_message_channels(rows: list[dict[str, Any]]) -> dict[int, int]:
    channels_by_message: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        channels_by_message[int(row.get("message_index", -1))].add(str(row.get("channel", "")))
    return {message_index: len(channels) for message_index, channels in channels_by_message.items()}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
