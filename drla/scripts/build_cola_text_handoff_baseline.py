"""Build text-channel handoff baselines for a P2-D latent replay subset.

This script is local-only and performs no model generation.  It materializes
the decoded text already available from the P1 halt-decision audit references
for the same samples used by a ``run_cola_sequential_latent_mas.py`` output.

The resulting ``tasks_text_*`` directories are official-scorer compatible and
serve as strong text-channel baselines:

* ``text_selected``: Agent A's selected/halt text is handed off directly.
* ``text_final``: full-budget text diagnostic.
* ``text_prediction_stability``: prediction-stability text diagnostic.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from drla.tracking import require_swanlab_disabled_for_non_training


@dataclass(frozen=True)
class TextHandoffBaselineConfig:
    eval_root: str
    packets_jsonl: str = (
        "/data1/luyifei/drla/outputs/cola_agent_latent_comm/"
        "p2_agent_latent_comm_v2_locked_seed66_67_68_split20260601_20260529/"
        "agent_latent_comm_packets_test.jsonl"
    )
    output_dir: str = ""
    swanlab_mode: str = "disabled"


def main() -> None:
    summary = build_text_handoff_baseline(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> TextHandoffBaselineConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--packets-jsonl", default=TextHandoffBaselineConfig.packets_jsonl)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--swanlab-mode", default=TextHandoffBaselineConfig.swanlab_mode)
    args = parser.parse_args()
    return TextHandoffBaselineConfig(
        eval_root=args.eval_root,
        packets_jsonl=args.packets_jsonl,
        output_dir=args.output_dir,
        swanlab_mode=args.swanlab_mode,
    )


def build_text_handoff_baseline(config: TextHandoffBaselineConfig) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        config.swanlab_mode,
        script_kind="P2-D text handoff baseline builder",
    )
    eval_root = Path(config.eval_root)
    output_dir = Path(config.output_dir) if config.output_dir else eval_root / "text_handoff_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    generations_path = eval_root / "generations.jsonl"
    if not generations_path.exists():
        raise FileNotFoundError(generations_path)

    matched_rows = [row for row in read_jsonl(generations_path) if row.get("control_type") == "matched"]
    if not matched_rows:
        raise ValueError(f"no matched rows found in {generations_path}")

    packet_refs = load_packet_refs(Path(config.packets_jsonl))
    decision_cache = DecisionRowCache()
    control_counts = {"text_selected": 0, "text_final": 0, "text_prediction_stability": 0}
    for row in matched_rows:
        sample_key = str(row["sample_key"])
        decision_ref = packet_refs.get(sample_key)
        if decision_ref is None:
            raise KeyError(f"packet reference not found for {sample_key}")
        decision = decision_cache.get(decision_ref, sample_key)
        for control, field in [
            ("text_selected", "selected_prediction"),
            ("text_final", "final_prediction"),
            ("text_prediction_stability", "prediction_stability_prediction"),
        ]:
            baseline_row = {
                "id": row["id"],
                "sample_key": sample_key,
                "task": row["task"],
                "control_type": control,
                "prompt": row.get("prompt", ""),
                "generate": decision.get(field, ""),
                "ground_truth": row.get("ground_truth", ""),
                "choices": row.get("choices", []),
                "source_prediction_field": field,
                "sender_selected_block": decision.get("selected_block"),
                "final_block": decision.get("final_block"),
                "prediction_stability_block": decision.get("prediction_stability_block"),
            }
            write_task_row(output_dir, control, str(row["task"]), baseline_row)
            control_counts[control] += 1

    summary = {
        "config": asdict(config),
        "num_samples": len(matched_rows),
        "control_counts": control_counts,
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "tasks_dirs": {
                control: str(output_dir / f"tasks_{control}") for control in sorted(control_counts)
            },
        },
        "interpretation": (
            "Direct decoded-text handoff baselines for the exact P2-D replay subset. "
            "These are text-channel diagnostics, not latent receiver inputs."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def load_packet_refs(path: Path) -> dict[str, str]:
    refs = {}
    for row in read_jsonl(path):
        refs[str(row["sample_key"])] = str(row["audit_refs"]["halt_decisions_jsonl"])
    return refs


class DecisionRowCache:
    def __init__(self) -> None:
        self.cache: dict[str, dict[str, dict[str, Any]]] = {}

    def get(self, path: str, sample_key: str) -> dict[str, Any]:
        if path not in self.cache:
            self.cache[path] = {str(row["sample_key"]): row for row in read_jsonl(Path(path))}
        if sample_key not in self.cache[path]:
            raise KeyError(f"halt decision row not found for {sample_key} in {path}")
        return self.cache[path][sample_key]


def write_task_row(output_dir: Path, control: str, task: str, row: dict[str, Any]) -> None:
    task_dir = output_dir / f"tasks_{control}"
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / f"{task}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
