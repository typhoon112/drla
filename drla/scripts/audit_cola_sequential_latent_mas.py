"""Audit P2-D sequential latent communication replay outputs.

This script is local-only.  It reads a completed
``run_cola_sequential_latent_mas.py`` output directory, combines the generated
texts with official scorer correct/wrong files, and compares each output to the
offline P1 halt-decision references and, when available, the raw native trace
``decode_text_so_far`` at the selected block.  The reference fields are used
only for post-hoc fidelity diagnostics; they are not online receiver inputs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from drla.tracking import require_swanlab_disabled_for_non_training


OFFICIAL_COLA_TASKS = [
    "lambada",
    "mmlu",
    "obqa",
    "hellaswag",
    "race",
    "siqa",
    "squad",
    "story_cloze",
]


@dataclass(frozen=True)
class SequentialLatentMasAuditConfig:
    eval_root: str
    packets_jsonl: str = (
        "/data1/luyifei/drla/outputs/cola_agent_latent_comm/"
        "p2_agent_latent_comm_v2_locked_seed66_67_68_split20260601_20260529/"
        "agent_latent_comm_packets_test.jsonl"
    )
    trace_root: str = (
        "/data1/luyifei/drla/outputs/cola_block_traces/"
        "tasks_official8_full_b64_t16_seed66_bs12_merged_20260524"
    )
    acc_calc_script: str = "/data1/luyifei/Cola-DLM/code/scripts/acc_calc.py"
    output_dir: str = ""
    swanlab_mode: str = "disabled"


def main() -> None:
    summary = audit_sequential_latent_mas(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> SequentialLatentMasAuditConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--packets-jsonl", default=SequentialLatentMasAuditConfig.packets_jsonl)
    parser.add_argument("--trace-root", default=SequentialLatentMasAuditConfig.trace_root)
    parser.add_argument("--acc-calc-script", default=SequentialLatentMasAuditConfig.acc_calc_script)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--swanlab-mode", default=SequentialLatentMasAuditConfig.swanlab_mode)
    args = parser.parse_args()
    return SequentialLatentMasAuditConfig(
        eval_root=args.eval_root,
        packets_jsonl=args.packets_jsonl,
        trace_root=args.trace_root,
        acc_calc_script=args.acc_calc_script,
        output_dir=args.output_dir,
        swanlab_mode=args.swanlab_mode,
    )


def audit_sequential_latent_mas(config: SequentialLatentMasAuditConfig) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        config.swanlab_mode,
        script_kind="P2-D sequential latent communication audit",
    )
    eval_root = Path(config.eval_root)
    output_dir = Path(config.output_dir) if config.output_dir else eval_root / "sequential_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    generations_path = eval_root / "generations.jsonl"
    if not generations_path.exists():
        raise FileNotFoundError(generations_path)

    packet_refs = load_packet_refs(Path(config.packets_jsonl))
    decision_cache = DecisionRowCache()
    trace_cache = TraceRowCache(Path(config.trace_root)) if config.trace_root else None
    scorer = load_official_scorer(Path(config.acc_calc_script))
    score_status = load_score_status(eval_root)
    rows: list[dict[str, Any]] = []
    generations = annotate_packet_indices(read_jsonl(generations_path))
    for generation in generations:
        sample_key = str(generation["sample_key"])
        decision_ref = packet_refs.get(sample_key)
        if decision_ref is None:
            raise KeyError(f"packet reference not found for {sample_key}")
        decision = decision_cache.get(decision_ref, sample_key)
        trace_row = None
        if trace_cache is not None:
            trace_row = trace_cache.get(
                task=str(generation["task"]),
                sample_id=generation["id"],
                block_number=int(decision["selected_block"]),
            )
        row = build_audit_row(generation, decision, score_status.pop(generation), trace_row, scorer)
        rows.append(row)

    by_control = aggregate_by_control(rows)
    paired = build_paired_rows(rows, baseline_control="matched")
    write_csv(output_dir / "fidelity_by_control.csv", by_control)
    write_csv(output_dir / "paired_score_changes.csv", paired)
    write_jsonl(output_dir / "sample_audit.jsonl", rows)
    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as handle:
        for row in by_control:
            handle.write(json.dumps({"control_type": row["control_type"], "metrics": row}, sort_keys=True) + "\n")

    summary = {
        "config": asdict(config),
        "num_generations": len(rows),
        "num_controls": len(by_control),
        "fidelity_by_control": by_control,
        "paired_score_changes": paired,
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "fidelity_by_control_csv": str(output_dir / "fidelity_by_control.csv"),
            "paired_score_changes_csv": str(output_dir / "paired_score_changes.csv"),
            "sample_audit_jsonl": str(output_dir / "sample_audit.jsonl"),
            "metrics_jsonl": str(metrics_path),
        },
        "interpretation": (
            "Offline audit only. P1 selected/final/prediction-stability references "
            "and raw trace decode_text_so_far measure replay fidelity and must not "
            "be used as online receiver inputs."
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


def annotate_packet_indices(generations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    controls = []
    for row in generations:
        control = str(row.get("control_type", ""))
        if control and control not in controls:
            controls.append(control)
    group_size = len(controls) if controls else 1
    for index, row in enumerate(generations):
        row["generation_index"] = index
        row["packet_index"] = index // group_size
    return generations


class DecisionRowCache:
    def __init__(self) -> None:
        self.cache: dict[str, dict[str, dict[str, Any]]] = {}

    def get(self, path: str, sample_key: str) -> dict[str, Any]:
        if path not in self.cache:
            rows = {}
            for row in read_jsonl(Path(path)):
                rows[str(row["sample_key"])] = row
            self.cache[path] = rows
        if sample_key not in self.cache[path]:
            raise KeyError(f"halt decision row not found for {sample_key} in {path}")
        return self.cache[path][sample_key]


class TraceRowCache:
    def __init__(self, trace_root: Path) -> None:
        self.trace_root = trace_root
        self.cache: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}

    def get(self, *, task: str, sample_id: Any, block_number: int) -> dict[str, Any] | None:
        if task not in self.cache:
            path = self.trace_root / f"{task}_traces.jsonl"
            if not path.exists():
                self.cache[task] = {}
            else:
                rows = {}
                for row in read_jsonl(path):
                    rows[(str(row.get("sample_id")), int(row.get("block_number", 0)))] = row
                self.cache[task] = rows
        return self.cache[task].get((str(sample_id), int(block_number)))


class ScoreStatusLookup:
    def __init__(self) -> None:
        self.exact: dict[tuple[str, str, str, str, str], deque[int]] = defaultdict(deque)
        self.fallback: dict[tuple[str, str, str, str], deque[int]] = defaultdict(deque)

    def add(self, row: dict[str, Any], *, control: str, task: str, value: int) -> None:
        base = (control, task, str(row.get("id")), str(row.get("sample_key", "")))
        self.fallback[base].append(value)
        self.exact[(*base, str(row.get("generate", "")))].append(value)

    def pop(self, generation: dict[str, Any]) -> int | None:
        base = (
            str(generation["control_type"]),
            str(generation["task"]),
            str(generation["id"]),
            str(generation.get("sample_key", "")),
        )
        exact = (*base, str(generation.get("generate", "")))
        if self.exact.get(exact):
            value = self.exact[exact].popleft()
            if self.fallback.get(base):
                try:
                    self.fallback[base].remove(value)
                except ValueError:
                    pass
            return value
        if self.fallback.get(base):
            return self.fallback[base].popleft()
        return None


def load_score_status(eval_root: Path) -> ScoreStatusLookup:
    status = ScoreStatusLookup()
    for control_dir in sorted(eval_root.glob("tasks_*")):
        control = control_dir.name.removeprefix("tasks_")
        for task in OFFICIAL_COLA_TASKS:
            for suffix, value in (("correct", 1), ("wrong", 0)):
                path = control_dir / f"{task}_{suffix}.jsonl"
                if not path.exists():
                    continue
                for row in read_jsonl(path):
                    status.add(row, control=control, task=task, value=value)
    return status


def build_audit_row(
    generation: dict[str, Any],
    decision: dict[str, Any],
    correct: int | None,
    trace_row: dict[str, Any] | None,
    scorer: Any,
) -> dict[str, Any]:
    task = str(generation["task"])
    answer_prefix = extract_answer_prefix(str(generation.get("generate", "")))
    native_trace_text = "" if trace_row is None else str(trace_row.get("decode_text_so_far", ""))
    native_trace_prefix = extract_answer_prefix(native_trace_text)
    replay_score = score_text_with_official_rules(
        task=task,
        text=str(generation.get("generate", "")),
        ground_truth=generation.get("ground_truth", ""),
        choices=generation.get("choices", []),
        scorer=scorer,
    )
    native_score = score_text_with_official_rules(
        task=task,
        text=native_trace_text,
        ground_truth=generation.get("ground_truth", ""),
        choices=generation.get("choices", []),
        scorer=scorer,
    )
    return {
        "control_type": generation["control_type"],
        "task": task,
        "id": generation["id"],
        "sample_key": generation["sample_key"],
        "generation_index": generation.get("generation_index"),
        "packet_index": generation.get("packet_index"),
        "source_sample_key": generation.get("source_sample_key", ""),
        "correct": correct,
        "selected_block": decision.get("selected_block"),
        "final_block": decision.get("final_block"),
        "selected_reference_correct": int(bool(decision.get("selected_correct"))),
        "final_reference_correct": int(bool(decision.get("final_correct"))),
        "prediction_stability_reference_correct": int(bool(decision.get("prediction_stability_correct"))),
        "replay_blocks_consumed": generation.get("replay_blocks_consumed"),
        "receiver_blocks_generated": generation.get("receiver_blocks_generated"),
        "total_blocks": generation.get("total_blocks"),
        "nonempty": int(bool(str(generation.get("generate", "")).strip())),
        "answer_prefix_chars": len(answer_prefix),
        "native_trace_selected_found": int(trace_row is not None),
        "native_trace_answer_prefix_chars": len(native_trace_prefix),
        "answer_prefix_agrees_selected": int(answers_agree(answer_prefix, decision.get("selected_prediction"))),
        "answer_prefix_agrees_final": int(answers_agree(answer_prefix, decision.get("final_prediction"))),
        "answer_prefix_agrees_prediction_stability": int(
            answers_agree(answer_prefix, decision.get("prediction_stability_prediction"))
        ),
        "answer_prefix_agrees_native_trace_selected": int(answers_agree(answer_prefix, native_trace_prefix)),
        "native_trace_agrees_selected_prediction": int(
            answers_agree(native_trace_prefix, decision.get("selected_prediction"))
        ),
        "official_processed_generation": replay_score["processed_generation"],
        "official_prediction": replay_score["prediction"],
        "official_score": replay_score["score"],
        "official_correct_from_text": int(replay_score["correct"]),
        "official_prediction_agrees_selected": int(
            normalize_text(replay_score["prediction"]) == normalize_text(decision.get("selected_prediction"))
        ),
        "official_prediction_agrees_final": int(
            normalize_text(replay_score["prediction"]) == normalize_text(decision.get("final_prediction"))
        ),
        "official_prediction_agrees_prediction_stability": int(
            normalize_text(replay_score["prediction"])
            == normalize_text(decision.get("prediction_stability_prediction"))
        ),
        "native_trace_official_processed_generation": native_score["processed_generation"],
        "native_trace_official_prediction": native_score["prediction"],
        "native_trace_official_score": native_score["score"],
        "native_trace_official_correct": int(native_score["correct"]),
        "native_trace_official_prediction_agrees_selected": int(
            normalize_text(native_score["prediction"]) == normalize_text(decision.get("selected_prediction"))
        ),
        "official_prediction_agrees_native_trace": int(
            normalize_text(replay_score["prediction"]) == normalize_text(native_score["prediction"])
        ),
    }


def score_text_with_official_rules(
    *,
    task: str,
    text: str,
    ground_truth: Any,
    choices: Any,
    scorer: Any,
) -> dict[str, Any]:
    generation = scorer.process_line({"generate": text}).get("generate", "")
    choices = choices or []
    if task == "lambada":
        prediction = scorer.get_first_word(generation)
        target = scorer.get_first_word(ground_truth)
        score = 1.0 if prediction == target else 0.0
    elif task in {"mmlu", "obqa", "race", "siqa"}:
        prediction = scorer.extract_mmlu_choice_letter(generation, choices)
        target = scorer.extract_gt_mmlu_choice_letter(ground_truth, choices)
        if prediction and target and prediction == target:
            score = 1.0
        else:
            score = scorer.calculate_similarity(generation, ground_truth)
    else:
        prediction = scorer.extract_answer_segment(generation)
        target = scorer.extract_answer_segment(ground_truth)
        score = scorer.calculate_similarity(generation, ground_truth)
    return {
        "processed_generation": generation,
        "prediction": prediction,
        "target": target,
        "score": float(score),
        "correct": bool(float(score) >= 1.0),
    }


def extract_answer_prefix(text: str) -> str:
    value = text.strip()
    if value.lower().startswith("answer:"):
        value = value.split(":", 1)[1].strip()
    markers = [
        "\n\nQuestion:",
        "\nQuestion:",
        "\n\nStory:",
        "\nStory:",
        "\n\nContext:",
        "\nContext:",
        "\n\nRead the following",
        "\nRead the following",
    ]
    cut = len(value)
    for marker in markers:
        idx = value.find(marker)
        if idx > 0:
            cut = min(cut, idx)
    return value[:cut].strip()


def answers_agree(generated: str, reference: Any) -> bool:
    gen = normalize_text(generated)
    ref = normalize_text(reference)
    if not gen or not ref:
        return False
    return gen == ref or gen.startswith(ref) or ref.startswith(gen)


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def aggregate_by_control(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["control_type"])].append(row)
    out = []
    for control, items in sorted(buckets.items()):
        count = len(items)
        scored = [row for row in items if row["correct"] is not None]
        out.append(
            {
                "control_type": control,
                "count": count,
                "scored_count": len(scored),
                "accuracy": mean([row["correct"] for row in scored]),
                "selected_reference_accuracy": mean([row["selected_reference_correct"] for row in items]),
                "nonempty_rate": mean([row["nonempty"] for row in items]),
                "native_trace_selected_found_rate": mean([row["native_trace_selected_found"] for row in items]),
                "selected_prediction_agreement_rate": mean(
                    [row["answer_prefix_agrees_selected"] for row in items]
                ),
                "final_prediction_agreement_rate": mean([row["answer_prefix_agrees_final"] for row in items]),
                "prediction_stability_agreement_rate": mean(
                    [row["answer_prefix_agrees_prediction_stability"] for row in items]
                ),
                "native_trace_selected_agreement_rate": mean(
                    [row["answer_prefix_agrees_native_trace_selected"] for row in items]
                ),
                "native_trace_to_selected_prediction_agreement_rate": mean(
                    [row["native_trace_agrees_selected_prediction"] for row in items]
                ),
                "official_prediction_agreement_rate": mean(
                    [row["official_prediction_agrees_selected"] for row in items]
                ),
                "official_prediction_to_native_trace_rate": mean(
                    [row["official_prediction_agrees_native_trace"] for row in items]
                ),
                "native_trace_official_prediction_agreement_rate": mean(
                    [row["native_trace_official_prediction_agrees_selected"] for row in items]
                ),
                "native_trace_official_accuracy": mean(
                    [row["native_trace_official_correct"] for row in items]
                ),
                "correct_selected_preservation_rate": conditional_mean(
                    items,
                    value_key="correct",
                    condition_key="selected_reference_correct",
                    condition_value=1,
                ),
                "correct_selected_prediction_match_rate": conditional_mean(
                    items,
                    value_key="official_prediction_agrees_selected",
                    condition_key="selected_reference_correct",
                    condition_value=1,
                ),
                "incorrect_selected_prediction_reproduction_rate": conditional_mean(
                    items,
                    value_key="official_prediction_agrees_selected",
                    condition_key="selected_reference_correct",
                    condition_value=0,
                ),
                "avg_total_blocks": mean([row["total_blocks"] for row in items]),
                "avg_replay_blocks": mean([row["replay_blocks_consumed"] for row in items]),
            }
        )
    return out


def build_paired_rows(rows: list[dict[str, Any]], baseline_control: str) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_key[
            (
                str(row["task"]),
                str(row["id"]),
                str(row["sample_key"]),
                str(row["packet_index"]),
            )
        ][str(row["control_type"])] = row
    controls = sorted({str(row["control_type"]) for row in rows if row["control_type"] != baseline_control})
    out = []
    for control in controls:
        comparable = 0
        baseline_wins = 0
        baseline_losses = 0
        same = 0
        for control_rows in by_key.values():
            if baseline_control not in control_rows or control not in control_rows:
                continue
            base_correct = control_rows[baseline_control].get("correct")
            other_correct = control_rows[control].get("correct")
            if base_correct is None or other_correct is None:
                continue
            comparable += 1
            if base_correct > other_correct:
                baseline_wins += 1
            elif base_correct < other_correct:
                baseline_losses += 1
            else:
                same += 1
        out.append(
            {
                "baseline_control": baseline_control,
                "control_type": control,
                "comparable": comparable,
                "baseline_wins": baseline_wins,
                "baseline_losses": baseline_losses,
                "same": same,
                "net_wins": baseline_wins - baseline_losses,
            }
        )
    return out


def mean(values: list[Any]) -> float:
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def conditional_mean(
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    condition_key: str,
    condition_value: Any,
) -> float:
    values = [row.get(value_key) for row in rows if row.get(condition_key) == condition_value]
    return mean(values)


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_official_scorer(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("cola_official_acc_calc_for_p2d_audit", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
