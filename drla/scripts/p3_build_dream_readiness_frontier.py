"""Build offline P3 Dream step-readiness frontier labels from traces.

This local-only D4 script consumes ``p3_collect_dream_step_traces.py`` artifacts
and scores solver step probes against the manifest gold/aliases. Gold/scorer
fields are used only to create offline teacher labels; they must not become
online student or communication inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.evaluation.p2_phase_c_scorers import score_qa_answer  # noqa: E402
from drla.scripts.run_p2_phase_c_text_agents import extract_final_answer, read_jsonl, write_jsonl  # noqa: E402


DEFAULT_OUTPUT_ROOT = "/data1/luyifei/drla/outputs/p3_dream_readiness_frontiers"


def main() -> None:
    summary = build_frontier(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--manifest-json", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--prediction-extraction-mode",
        choices=["default", "first_segment"],
        default="first_segment",
    )
    parser.add_argument("--correct-threshold", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_frontier(args: argparse.Namespace) -> dict[str, Any]:
    created_at = int(time.time())
    trace_dir = Path(args.trace_dir)
    trace_summary = json.loads((trace_dir / "summary.json").read_text(encoding="utf-8"))
    manifest_json = args.manifest_json or trace_summary.get("manifest_json", "")
    if not manifest_json:
        raise ValueError("--manifest-json is required when trace summary lacks manifest_json")
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(trace_dir, created_at)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(manifest_json).read_text(encoding="utf-8"))
    samples = {str(sample["sample_id"]): sample for sample in manifest.get("samples", [])}
    generations = read_jsonl(trace_dir / "generations.jsonl")
    traces = read_jsonl(trace_dir / "traces.jsonl")
    traces_by_call = {str(trace.get("call_id", "")): trace for trace in traces}

    event_records = []
    row_records = []
    missing_solver_calls = []
    for row in generations:
        if row.get("status") != "ok":
            continue
        call_ids = [str(item) for item in row.get("trace_call_ids", [])]
        if not call_ids:
            continue
        solver_call_id = call_ids[-1]
        trace = traces_by_call.get(solver_call_id)
        if trace is None or trace.get("agent_role") != "solver":
            missing_solver_calls.append({"row_id": row.get("row_id", ""), "solver_call_id": solver_call_id})
            continue
        sample = samples.get(str(row.get("sample_id", "")))
        if sample is None:
            raise ValueError(f"unknown sample_id in generation row: {row.get('sample_id')}")
        row_event_records = build_row_event_records(row, trace, sample, args)
        event_records.extend(row_event_records)
        row_records.append(build_row_frontier_record(row, trace, row_event_records, args))

    write_jsonl(output_dir / "frontier_events.jsonl", event_records)
    write_jsonl(output_dir / "frontier_rows.jsonl", row_records)
    metrics = compute_metrics(event_records, row_records, missing_solver_calls)
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "created_at": created_at,
        "status": "pass" if not missing_solver_calls else "warn",
        "trace_dir": str(trace_dir),
        "manifest_json": manifest_json,
        "frontier_events_jsonl": str(output_dir / "frontier_events.jsonl"),
        "frontier_rows_jsonl": str(output_dir / "frontier_rows.jsonl"),
        "metrics_jsonl": str(metrics_path),
        "num_generation_rows": len(generations),
        "num_trace_calls": len(traces),
        "num_frontier_events": len(event_records),
        "num_frontier_rows": len(row_records),
        "num_missing_solver_calls": len(missing_solver_calls),
        "missing_solver_calls_preview": missing_solver_calls[:10],
        "metrics": metrics,
        "execution_boundary": [
            "local-only P3 Dream D4 readiness frontier build",
            "no model generation",
            "no optimizer or backward",
            "no SwanLab run",
            "gold/scorer fields are offline teacher labels only",
        ],
        "online_student_allowed_feature_sources": [
            "hidden_summary",
            "hidden_ref",
            "logit confidence summaries",
            "mask/process/change features",
            "step index",
        ],
        "forbidden_online_fields": [
            "gold_answer",
            "answer_aliases",
            "step_prediction",
            "step_score",
            "final_prediction",
            "final_score",
            "oracle labels",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary["summary_json"] = str(output_dir / "summary.json")
    return summary


def build_row_event_records(
    row: dict[str, Any],
    trace: dict[str, Any],
    sample: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    scoring = sample.get("scoring", {})
    gold = scoring.get("gold_answer", "")
    aliases = scoring.get("answer_aliases", []) or []
    final_prediction = str(row.get("prediction", ""))
    final_score = float(row.get("primary_score", 0.0))
    records = []
    step_summaries = trace.get("step_summaries", []) or []
    for index, event in enumerate(step_summaries):
        decoded = str(event.get("decoded_probe_text", ""))
        step_prediction = extract_final_answer(decoded, mode=args.prediction_extraction_mode)
        step_score = score_qa_answer(step_prediction, gold, aliases).to_dict()
        next_decoded = str(step_summaries[index + 1].get("decoded_probe_text", "")) if index + 1 < len(step_summaries) else ""
        next_prediction = extract_final_answer(next_decoded, mode=args.prediction_extraction_mode) if next_decoded else ""
        records.append(
            {
                "row_id": row.get("row_id", ""),
                "sample_id": row.get("sample_id", ""),
                "condition": row.get("condition", ""),
                "call_id": trace.get("call_id", ""),
                "agent_role": trace.get("agent_role", ""),
                "agent_id": trace.get("agent_id", ""),
                "trace_event_index": event.get("trace_event_index", index),
                "step": event.get("step", None),
                "has_hidden_summary": bool(event.get("has_hidden_summary", False)),
                "has_hidden_ref": bool(event.get("has_hidden_ref", False)),
                "has_logit_stats": bool(event.get("has_logit_stats", False)),
                "hidden_summary": event.get("hidden_summary", {}) or {},
                "hidden_ref": event.get("hidden_ref", ""),
                "num_mask_tokens": event.get("num_mask_tokens", None),
                "changed_suffix_tokens_vs_prev_hook": event.get("changed_suffix_tokens_vs_prev_hook", None),
                "top1_prob_mean": event.get("top1_prob_mean", None),
                "top2_margin_mean": event.get("top2_margin_mean", None),
                "entropy_mean": event.get("entropy_mean", None),
                "step_prediction": step_prediction,
                "step_primary_score": step_score["primary_score"],
                "step_token_f1": step_score["token_f1"],
                "step_exact_match": step_score["exact_match"],
                "final_prediction": final_prediction,
                "final_primary_score": final_score,
                "prediction_matches_final": normalize_answer_text(step_prediction) == normalize_answer_text(final_prediction),
                "next_prediction": next_prediction,
                "prediction_changes_next_event": (
                    normalize_answer_text(step_prediction) != normalize_answer_text(next_prediction)
                    if next_prediction
                    else False
                ),
                "answer_ready_correct": float(step_score["primary_score"]) >= args.correct_threshold,
                "answer_ready_correct_and_final_stable": (
                    float(step_score["primary_score"]) >= args.correct_threshold
                    and normalize_answer_text(step_prediction) == normalize_answer_text(final_prediction)
                ),
                "future_gain_vs_final": final_score - float(step_score["primary_score"]),
            }
        )
    return records


def build_row_frontier_record(
    row: dict[str, Any],
    trace: dict[str, Any],
    events: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    first_correct = first_event(events, "answer_ready_correct")
    first_correct_stable = first_event(events, "answer_ready_correct_and_final_stable")
    first_final_match = first_matching_final(events)
    final_event = events[-1] if events else {}
    return {
        "row_id": row.get("row_id", ""),
        "sample_id": row.get("sample_id", ""),
        "condition": row.get("condition", ""),
        "solver_call_id": trace.get("call_id", ""),
        "num_events": len(events),
        "final_prediction": row.get("prediction", ""),
        "final_primary_score": row.get("primary_score", 0.0),
        "final_event_step": final_event.get("step", None),
        "final_event_primary_score": final_event.get("step_primary_score", None),
        "first_correct_event_index": first_correct.get("trace_event_index") if first_correct else None,
        "first_correct_step": first_correct.get("step") if first_correct else None,
        "first_correct_stable_event_index": first_correct_stable.get("trace_event_index") if first_correct_stable else None,
        "first_correct_stable_step": first_correct_stable.get("step") if first_correct_stable else None,
        "first_final_match_event_index": first_final_match.get("trace_event_index") if first_final_match else None,
        "first_final_match_step": first_final_match.get("step") if first_final_match else None,
        "oracle_has_correct_before_final": bool(first_correct and first_correct.get("trace_event_index") != final_event.get("trace_event_index")),
        "oracle_has_correct_stable_before_final": bool(
            first_correct_stable and first_correct_stable.get("trace_event_index") != final_event.get("trace_event_index")
        ),
        "mean_future_gain_vs_final": mean([float(event.get("future_gain_vs_final", 0.0)) for event in events]),
        "num_prediction_change_events": sum(1 for event in events if event.get("prediction_changes_next_event")),
        "hidden_summary_coverage": mean([float(event.get("has_hidden_summary", False)) for event in events]),
        "logit_stats_coverage": mean([float(event.get("has_logit_stats", False)) for event in events]),
    }


def first_event(events: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    for event in events:
        if event.get(key):
            return event
    return None


def first_matching_final(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in events:
        if event.get("prediction_matches_final"):
            return event
    return None


def compute_metrics(
    event_records: list[dict[str, Any]],
    row_records: list[dict[str, Any]],
    missing_solver_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    condition_counts = Counter(str(row.get("condition", "")) for row in row_records)
    per_condition_final = {
        condition: mean([float(row.get("final_primary_score", 0.0)) for row in row_records if row.get("condition") == condition])
        for condition in sorted(condition_counts)
    }
    return {
        "num_events": len(event_records),
        "num_rows": len(row_records),
        "num_missing_solver_calls": len(missing_solver_calls),
        "condition_counts": dict(condition_counts),
        "final_primary_by_condition": per_condition_final,
        "event_hidden_summary_coverage": mean([float(event.get("has_hidden_summary", False)) for event in event_records]),
        "event_logit_stats_coverage": mean([float(event.get("has_logit_stats", False)) for event in event_records]),
        "row_oracle_correct_before_final_rate": mean(
            [float(row.get("oracle_has_correct_before_final", False)) for row in row_records]
        ),
        "row_oracle_correct_stable_before_final_rate": mean(
            [float(row.get("oracle_has_correct_stable_before_final", False)) for row in row_records]
        ),
        "mean_first_correct_step": mean_defined([row.get("first_correct_step") for row in row_records]),
        "mean_first_correct_stable_step": mean_defined([row.get("first_correct_stable_step") for row in row_records]),
    }


def normalize_answer_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def mean_defined(values: list[Any]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return mean(cleaned) if cleaned else None


def default_output_dir(trace_dir: Path, created_at: int) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(created_at))
    return Path(DEFAULT_OUTPUT_ROOT) / f"{trace_dir.name}_frontier_{stamp}"


if __name__ == "__main__":
    main()
