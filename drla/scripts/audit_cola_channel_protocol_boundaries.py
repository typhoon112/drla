"""Audit P2-D Agent-B channel evaluation input/output boundaries.

This local-only audit classifies channel-eval artifacts by whether the receiver
and scorer respect the LatentMAS-aligned Agent-A -> Agent-B handoff boundary.
A valid canonical Agent-B communication evaluation must satisfy:

* ``agent_b_input_contract == message_only`` in the eval summary/config.
* every generation row has ``agent_b_input_contract == message_only``.
* ``score_output_scope == receiver_only`` in the eval summary/config.
* every generation row has ``score_output_scope == receiver_only``.
* scorer-visible Agent-A text-message tokens sum to zero.
* scorer-visible Agent-A replay blocks sum to zero.

Artifacts produced before ``score_output_scope`` existed are classified as
``legacy_all_visible_inferred`` and must be treated as replay-output /
decodability diagnostics, not Agent-B communication evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from drla.tracking import require_swanlab_disabled_for_non_training


@dataclass(frozen=True)
class ProtocolBoundaryAuditConfig:
    eval_roots: tuple[str, ...]
    output_dir: str
    swanlab_mode: str = "disabled"


def main() -> None:
    summary = audit_protocol_boundaries(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> ProtocolBoundaryAuditConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--swanlab-mode", default="disabled")
    args = parser.parse_args()
    return ProtocolBoundaryAuditConfig(
        eval_roots=tuple(args.eval_root),
        output_dir=args.output_dir,
        swanlab_mode=args.swanlab_mode,
    )


def audit_protocol_boundaries(config: ProtocolBoundaryAuditConfig) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        config.swanlab_mode,
        script_kind="P2-D channel protocol-boundary audit",
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [audit_eval_root(Path(root)) for root in config.eval_roots]
    csv_path = output_dir / "protocol_boundary_audit.csv"
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    write_csv(csv_path, rows)
    with metrics_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"created_at": int(time.time()), "metrics": row}, sort_keys=True) + "\n")
    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "rows": rows,
        "valid_message_only_receiver_only_roots": [
            row["eval_root"] for row in rows if row["status"] == "pass"
        ],
        "invalid_or_legacy_roots": [row["eval_root"] for row in rows if row["status"] != "pass"],
        "artifacts": {
            "summary_json": str(summary_path),
            "metrics_jsonl": str(metrics_path),
            "protocol_boundary_audit_csv": str(csv_path),
        },
        "interpretation": (
            "Only rows with status=pass may support canonical message_only "
            "Agent-B communication claims. shared_context or legacy rows can "
            "support only diagnostics."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def audit_eval_root(eval_root: Path) -> dict[str, Any]:
    summary_path = eval_root / "summary.json"
    generations_path = eval_root / "generations.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not generations_path.exists():
        raise FileNotFoundError(generations_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = summary.get("config", {})
    generations = read_jsonl(generations_path)
    row_scopes = sorted({str(row.get("score_output_scope", "legacy_all_visible_inferred")) for row in generations})
    row_contracts = sorted({str(row.get("agent_b_input_contract", "legacy_or_unspecified")) for row in generations})
    configured_scope = str(config.get("score_output_scope") or (row_scopes[0] if len(row_scopes) == 1 else "legacy_all_visible_inferred"))
    configured_contract = str(config.get("agent_b_input_contract") or (row_contracts[0] if len(row_contracts) == 1 else "legacy_or_unspecified"))
    visible_text_tokens = sum(int(row.get("scorer_visible_text_message_tokens", 0)) for row in generations)
    visible_replay_blocks = sum(int(row.get("scorer_visible_replay_blocks", 0)) for row in generations)
    channel_count = len({str(row.get("channel", "")) for row in generations})
    status = "pass"
    reasons = []
    if configured_contract != "message_only":
        status = "fail"
        reasons.append(f"configured_contract={configured_contract}")
    if row_contracts != ["message_only"]:
        status = "fail"
        reasons.append(f"row_contracts={','.join(row_contracts)}")
    if configured_scope != "receiver_only":
        status = "fail"
        reasons.append(f"configured_scope={configured_scope}")
    if row_scopes != ["receiver_only"]:
        status = "fail"
        reasons.append(f"row_scopes={','.join(row_scopes)}")
    if visible_text_tokens != 0:
        status = "fail"
        reasons.append(f"visible_text_tokens={visible_text_tokens}")
    if visible_replay_blocks != 0:
        status = "fail"
        reasons.append(f"visible_replay_blocks={visible_replay_blocks}")
    return {
        "eval_root": str(eval_root),
        "status": status,
        "failure_reasons": ";".join(reasons),
        "configured_agent_b_input_contract": configured_contract,
        "row_agent_b_input_contracts": ",".join(row_contracts),
        "configured_score_output_scope": configured_scope,
        "row_score_output_scopes": ",".join(row_scopes),
        "num_messages": int(summary.get("num_messages", 0)),
        "num_generations": int(summary.get("num_generations", len(generations))),
        "channel_count": channel_count,
        "scorer_visible_text_message_tokens": visible_text_tokens,
        "scorer_visible_replay_blocks": visible_replay_blocks,
        "claim_allowed": (
            "agent_b_communication"
            if status == "pass"
            else "decodability_or_replay_output_diagnostic_only"
        ),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
