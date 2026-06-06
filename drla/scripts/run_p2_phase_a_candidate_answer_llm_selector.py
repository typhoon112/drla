"""Run a local LLM candidate-answer selector for Phase A diagnostics.

This script is a local-only diagnostic. It asks a capable local model to select
the final short answer from evidence-derived candidates and the online evidence
context. It never trains, never logs to SwanLab, and must not read held-out data
for prompt/objective selection.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.evaluation.p2_phase_c_scorers import score_qa_answer
from drla.scripts.run_p2_phase_c_text_agents import (
    LocalTransformersProvider,
    extract_final_answer,
)


DEFAULT_CANDIDATES_JSONL = (
    "/data1/luyifei/drla/outputs/p2_phase_a_candidate_answers/"
    "musique_calibration_candidate_answers_200_seed20260606_20260606/candidates.jsonl"
)
DEFAULT_MANIFEST_JSON = (
    "/data1/luyifei/drla/outputs/p2_phase_c_manifests/"
    "musique_calibration_manifest_200_seed20260601/manifest.json"
)
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p2_phase_a_candidate_selectors/"
    "musique_candidate_selector_qwen3_8b_fp8_calib200_top128_20260606"
)
DEFAULT_LOCAL_MODEL_PATH = "/data1/luyifei/drla/models/Qwen3-8B-FP8"


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-jsonl", default=DEFAULT_CANDIDATES_JSONL)
    parser.add_argument("--manifest-json", default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--local-model-path", default=DEFAULT_LOCAL_MODEL_PATH)
    parser.add_argument("--local-device-map", default="auto")
    parser.add_argument("--local-dtype", default="auto")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-candidates-per-sample", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite and not args.resume:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = read_jsonl(Path(args.candidates_jsonl))
    rows = shard_rows(all_rows, num_shards=args.num_shards, shard_index=args.shard_index)
    if args.row_offset:
        rows = rows[args.row_offset :]
    if args.max_samples:
        rows = rows[: args.max_samples]
    samples = load_manifest_samples(Path(args.manifest_json))

    predictions_path = output_dir / "predictions.jsonl"
    predictions = read_jsonl(predictions_path) if args.resume and predictions_path.exists() else []
    completed = {str(row.get("sample_id", "")) for row in predictions}
    if not args.resume:
        predictions_path.write_text("", encoding="utf-8")

    provider = LocalTransformersProvider(
        model_path=args.local_model_path,
        device_map=args.local_device_map,
        dtype=args.local_dtype,
        enable_thinking=False,
    )
    start_time = time.time()
    for index, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"])
        if sample_id in completed:
            continue
        sample = samples.get(sample_id)
        if sample is None:
            raise ValueError(f"sample_id not found in manifest: {sample_id}")
        prediction = run_one(row, sample, provider, args)
        append_jsonl(predictions_path, prediction)
        predictions.append(prediction)
        completed.add(sample_id)
        if args.progress_interval and len(predictions) % args.progress_interval == 0:
            elapsed = time.time() - start_time
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "completed_predictions": len(predictions),
                        "scheduled_rows": len(rows),
                        "current_index": index,
                        "elapsed_seconds": round(elapsed, 2),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )

    metrics = compute_metrics(predictions)
    summary = {
        "created_at": int(time.time()),
        "status": "pass",
        "candidates_jsonl": args.candidates_jsonl,
        "manifest_json": args.manifest_json,
        "output_dir": str(output_dir),
        "predictions_jsonl": str(predictions_path),
        "local_model_path": args.local_model_path,
        "num_input_rows": len(all_rows),
        "num_scheduled_rows": len(rows),
        "num_predictions": len(predictions),
        "row_offset": args.row_offset,
        "max_samples": args.max_samples,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "max_candidates_per_sample": args.max_candidates_per_sample,
        "metrics": metrics,
        "execution_boundary": [
            "local-only LLM candidate selector diagnostic",
            "no deep-learning optimizer/backward",
            "no SwanLab run",
            "no held-out data",
            "gold labels used only for offline scoring",
            "candidate strings are evidence-derived; gold is not injected as a candidate",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.jsonl").write_text(
        json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def run_one(
    row: dict[str, Any],
    sample: dict[str, Any],
    provider: LocalTransformersProvider,
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidates = list(row.get("candidates", []))[: args.max_candidates_per_sample]
    messages = make_messages(row, sample, candidates)
    raw_output = provider.chat(messages, args)
    prediction = resolve_prediction_to_candidate(
        extract_final_answer(raw_output, mode="default"),
        candidates,
    )
    scoring = score_qa_answer(
        prediction,
        row.get("gold_answer", ""),
        row.get("answer_aliases", []) or [],
    ).to_dict()
    return {
        "sample_id": row["sample_id"],
        "question": row.get("question", ""),
        "raw_output": raw_output,
        "prediction": prediction,
        "gold_answer": row.get("gold_answer", ""),
        "score": scoring,
        "primary_score": scoring["primary_score"],
        "token_f1": scoring["token_f1"],
        "exact_match": scoring["exact_match"],
        "oracle_gold_covered_kept": row.get("audit", {}).get("gold_covered_kept"),
        "oracle_gold_best_rank_kept": row.get("audit", {}).get("gold_best_rank_kept"),
        "num_candidates": len(candidates),
        "matched_candidate": best_candidate_match(prediction, candidates),
    }


def make_messages(
    row: dict[str, Any],
    sample: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> list[dict[str, str]]:
    system = (
        "You are a candidate-answer selector for evidence-split QA. Use only "
        "the question, evidence, and candidate list. Return exactly one line in "
        "the format 'Final answer: <short answer>'. Prefer an answer from the "
        "candidate list. Do not explain."
    )
    payload = {
        "question": row.get("question", ""),
        "full_evidence": sample.get("metadata", {}).get("full_info_observation", ""),
        "candidates": [
            {
                "id": index,
                "text": candidate.get("text", ""),
                "rule": candidate.get("rule", ""),
                "evidence_kind": candidate.get("evidence_kind", ""),
                "source_title": candidate.get("source_title", ""),
                "rank": candidate.get("rank", index),
            }
            for index, candidate in enumerate(candidates, start=1)
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)},
    ]


def best_candidate_match(prediction: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    best = {"candidate_text": "", "candidate_rank": None, "primary_score": 0.0, "token_f1": 0.0}
    for candidate in candidates:
        score = score_qa_answer(prediction, candidate.get("text", ""), []).to_dict()
        if score["primary_score"] > best["primary_score"] or score["token_f1"] > best["token_f1"]:
            best = {
                "candidate_text": candidate.get("text", ""),
                "candidate_rank": candidate.get("rank"),
                "primary_score": score["primary_score"],
                "token_f1": score["token_f1"],
            }
    return best


def resolve_prediction_to_candidate(prediction: str, candidates: list[dict[str, Any]]) -> str:
    stripped = prediction.strip().strip('"')
    match = re.search(r"(?i)\b(?:candidate|option|id)?\s*#?\s*(\d{1,3})\b", stripped)
    if match:
        index = int(match.group(1))
        if 1 <= index <= len(candidates):
            candidate_text = str(candidates[index - 1].get("text", "")).strip()
            if candidate_text:
                return candidate_text
    prefix_match = re.match(r"^\s*\d{1,3}\s*[\).:-]\s*(.+)$", stripped)
    if prefix_match:
        return prefix_match.group(1).strip()
    return stripped


def compute_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [float(row["primary_score"]) for row in predictions]
    token_f1 = [float(row["token_f1"]) for row in predictions]
    exact = [float(row["exact_match"]) for row in predictions]
    covered_rows = [row for row in predictions if row.get("oracle_gold_covered_kept")]
    return {
        "selected_primary": mean(primary),
        "selected_token_f1": mean(token_f1),
        "selected_exact_match": mean(exact),
        "oracle_coverage_kept": mean([row.get("oracle_gold_covered_kept") for row in predictions]),
        "selected_given_covered": mean([row["primary_score"] for row in covered_rows]),
        "num_predictions": len(predictions),
    }


def load_manifest_samples(path: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {str(sample["sample_id"]): sample for sample in manifest.get("samples", [])}


def shard_rows(
    rows: list[dict[str, Any]],
    *,
    num_shards: int,
    shard_index: int,
) -> list[dict[str, Any]]:
    if num_shards == 1:
        return rows
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_no}")
        rows.append(value)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def mean(values: list[Any]) -> float:
    values = list(values)
    return sum(float(value) for value in values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
