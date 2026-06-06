"""Build non-heldout candidate-constrained short-answer SFT pairs for CoLA.

This script prepares data only. It does not train, call models, inspect
held-out data, or create SwanLab runs.

The objective is to turn the completed train-source semantic candidate teacher
into a CoLA-compatible short-answer interface. Online prompts contain only the
question, online evidence, and evidence-derived candidate texts/metadata.
Gold labels, alias flags, teacher correctness, and scorer outputs are used only
for offline filtering and summary metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST_JSON = (
    "/data1/luyifei/drla/outputs/p2_phase_c_manifests/"
    "musique_interface_train_manifest_10000_seed20260606/manifest.json"
)
DEFAULT_CANDIDATES_JSONL = (
    "/data1/luyifei/drla/outputs/p2_phase_a_candidate_answers/"
    "musique_train_candidate_answers_10000_seed20260606_20260606/candidates.jsonl"
)
DEFAULT_TEACHER_PREDICTIONS_JSONL = (
    "/data1/luyifei/drla/outputs/p2_phase_a_candidate_selectors/"
    "musique_candidate_selector_qwen3_8b_fp8_train10000_top128_all10_aggregate_20260606/"
    "predictions.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p2_phase_a_cola_interface_sft/"
    "musique_candidate_constrained_short_answer_train10000_top128_qwen_teacher_20260606"
)


def main() -> None:
    summary = build_sft(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--candidates-jsonl", default=DEFAULT_CANDIDATES_JSONL)
    parser.add_argument("--teacher-predictions-jsonl", default=DEFAULT_TEACHER_PREDICTIONS_JSONL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=20260606)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all allowed source samples.")
    parser.add_argument("--max-candidates", type=int, default=128)
    parser.add_argument(
        "--roles",
        default="solver_candidate_gold_covered,solver_candidate_teacher_correct",
        help=(
            "Comma-separated roles to emit. Supported: "
            "solver_candidate_gold_covered, solver_candidate_teacher_correct, "
            "solver_candidate_teacher_all."
        ),
    )
    parser.add_argument(
        "--target-prefix",
        choices=["answer_only", "final_answer"],
        default="answer_only",
        help="Target format. final_answer writes 'Final answer: <answer>'.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_sft(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 < args.valid_fraction < 0.5:
        raise ValueError("--valid-fraction must be in (0, 0.5)")
    roles = {role.strip() for role in str(args.roles).split(",") if role.strip()}
    supported_roles = {
        "solver_candidate_gold_covered",
        "solver_candidate_teacher_correct",
        "solver_candidate_teacher_all",
    }
    unknown = roles - supported_roles
    if unknown:
        raise ValueError(f"unsupported roles: {sorted(unknown)}")
    if not roles:
        raise ValueError("--roles must contain at least one role")
    if args.max_candidates < 1:
        raise ValueError("--max-candidates must be positive")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
    samples = list(manifest.get("samples", []))
    if args.max_samples:
        samples = samples[: args.max_samples]
    assert_allowed_source_splits(samples)

    candidate_by_sample = {
        str(row.get("sample_id", "")): row for row in read_jsonl(Path(args.candidates_jsonl))
    }
    teacher_by_sample = {
        str(row.get("sample_id", "")): row for row in read_jsonl(Path(args.teacher_predictions_jsonl))
    }
    split_by_sample = make_sample_splits(
        [str(sample["sample_id"]) for sample in samples],
        valid_fraction=args.valid_fraction,
        seed=args.split_seed,
    )

    pairs: list[dict[str, Any]] = []
    drops: Counter[str] = Counter()
    audit_counts: Counter[str] = Counter()
    role_target_lengths: dict[str, list[int]] = defaultdict(list)
    role_candidate_counts: dict[str, list[int]] = defaultdict(list)

    for sample in samples:
        sample_id = str(sample["sample_id"])
        question = str(sample.get("question", "")).strip()
        context = full_evidence(sample)
        candidate_row = candidate_by_sample.get(sample_id)
        teacher_row = teacher_by_sample.get(sample_id)
        if not question or not context:
            drops["missing_question_or_context"] += 1
            continue
        if not candidate_row:
            drops["missing_candidates"] += 1
            continue

        candidates = list(candidate_row.get("candidates", []))[: args.max_candidates]
        if not candidates:
            drops["empty_candidates"] += 1
            continue
        prompt_text = make_candidate_prompt(question=question, context=context, candidates=candidates)
        split = split_by_sample[sample_id]
        audit_counts["samples_with_candidates"] += 1
        if bool(candidate_row.get("audit", {}).get("gold_covered_kept")):
            audit_counts["samples_gold_covered_source_kept"] += 1
        gold_candidate = best_gold_candidate(candidates)
        if gold_candidate:
            audit_counts["samples_gold_covered_retained"] += 1
        if teacher_row and float(teacher_row.get("primary_score", 0.0) or 0.0) >= 1.0:
            audit_counts["samples_teacher_primary_correct"] += 1

        if "solver_candidate_gold_covered" in roles:
            if gold_candidate:
                target_text = format_target(str(gold_candidate.get("text", "")).strip(), args.target_prefix)
                pairs.append(
                    make_pair(
                        sample=sample,
                        split=split,
                        role="solver_candidate_gold_covered",
                        source_condition=f"candidate_top{args.max_candidates}_full_evidence_gold_covered",
                        target_source="best_gold_or_alias_candidate_text",
                        prompt_text=prompt_text,
                        target_text=target_text,
                        candidates=candidates,
                        teacher_row=teacher_row,
                    )
                )
                role_target_lengths["solver_candidate_gold_covered"].append(len(target_text))
                role_candidate_counts["solver_candidate_gold_covered"].append(len(candidates))
            else:
                drops["gold_covered_role_without_gold_candidate"] += 1

        if "solver_candidate_teacher_correct" in roles:
            if not teacher_row:
                drops["missing_teacher_for_teacher_correct_role"] += 1
            elif float(teacher_row.get("primary_score", 0.0) or 0.0) >= 1.0:
                if not teacher_target_in_candidates(teacher_row, candidates):
                    drops["teacher_correct_target_not_in_retained_candidates"] += 1
                    continue
                audit_counts["samples_teacher_primary_correct_retained"] += 1
                target_text = format_target(str(teacher_row.get("prediction", "")).strip(), args.target_prefix)
                if target_text:
                    pairs.append(
                        make_pair(
                            sample=sample,
                            split=split,
                            role="solver_candidate_teacher_correct",
                            source_condition=f"candidate_top{args.max_candidates}_full_evidence_qwen_teacher_correct",
                            target_source="qwen3_8b_fp8_teacher_prediction_primary_correct",
                            prompt_text=prompt_text,
                            target_text=target_text,
                            candidates=candidates,
                            teacher_row=teacher_row,
                        )
                    )
                    role_target_lengths["solver_candidate_teacher_correct"].append(len(target_text))
                    role_candidate_counts["solver_candidate_teacher_correct"].append(len(candidates))
                else:
                    drops["empty_teacher_correct_prediction"] += 1

        if "solver_candidate_teacher_all" in roles:
            if not teacher_row:
                drops["missing_teacher_for_teacher_all_role"] += 1
            else:
                if not teacher_target_in_candidates(teacher_row, candidates):
                    drops["teacher_all_target_not_in_retained_candidates"] += 1
                    continue
                target_text = format_target(str(teacher_row.get("prediction", "")).strip(), args.target_prefix)
                if target_text:
                    pairs.append(
                        make_pair(
                            sample=sample,
                            split=split,
                            role="solver_candidate_teacher_all",
                            source_condition=f"candidate_top{args.max_candidates}_full_evidence_qwen_teacher_all",
                            target_source="qwen3_8b_fp8_teacher_prediction_all_noisy",
                            prompt_text=prompt_text,
                            target_text=target_text,
                            candidates=candidates,
                            teacher_row=teacher_row,
                        )
                    )
                    role_target_lengths["solver_candidate_teacher_all"].append(len(target_text))
                    role_candidate_counts["solver_candidate_teacher_all"].append(len(candidates))
                else:
                    drops["empty_teacher_all_prediction"] += 1

    if not pairs:
        raise ValueError("no SFT pairs were emitted")

    pair_path = output_dir / "sft_pairs.jsonl"
    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    write_jsonl(pair_path, pairs)
    write_jsonl(train_path, [row for row in pairs if row["split"] == "train"])
    write_jsonl(valid_path, [row for row in pairs if row["split"] == "valid"])

    role_counts = Counter(str(row["role"]) for row in pairs)
    split_counts = Counter(str(row["split"]) for row in pairs)
    target_lengths = [len(str(row.get("target_text", ""))) for row in pairs]
    prompt_lengths = [len(str(row.get("prompt_text", ""))) for row in pairs]
    metrics = {
        "num_samples": len(samples),
        "num_pairs": len(pairs),
        "num_train_pairs": split_counts["train"],
        "num_valid_pairs": split_counts["valid"],
        "num_roles": len(role_counts),
        "samples_with_candidates": audit_counts["samples_with_candidates"],
        "samples_gold_covered_source_kept": audit_counts["samples_gold_covered_source_kept"],
        "samples_gold_covered_retained": audit_counts["samples_gold_covered_retained"],
        "samples_teacher_primary_correct": audit_counts["samples_teacher_primary_correct"],
        "samples_teacher_primary_correct_retained": audit_counts[
            "samples_teacher_primary_correct_retained"
        ],
        "gold_covered_source_kept_rate": audit_counts["samples_gold_covered_source_kept"]
        / max(1, len(samples)),
        "gold_covered_retained_rate": audit_counts["samples_gold_covered_retained"] / max(1, len(samples)),
        "teacher_primary_correct_rate": audit_counts["samples_teacher_primary_correct"] / max(1, len(samples)),
        "teacher_primary_correct_retained_rate": audit_counts[
            "samples_teacher_primary_correct_retained"
        ]
        / max(1, len(samples)),
        "prompt_chars_mean": safe_mean(prompt_lengths),
        "prompt_chars_p95": percentile(prompt_lengths, 0.95),
        "target_chars_mean": safe_mean(target_lengths),
    }
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "created_at": int(time.time()),
        "status": "pass",
        "manifest_json": args.manifest_json,
        "candidates_jsonl": args.candidates_jsonl,
        "teacher_predictions_jsonl": args.teacher_predictions_jsonl,
        "output_dir": str(output_dir),
        "sft_pairs_jsonl": str(pair_path),
        "train_jsonl": str(train_path),
        "valid_jsonl": str(valid_path),
        "metrics_jsonl": str(metrics_path),
        "num_samples": len(samples),
        "num_pairs": len(pairs),
        "split_counts": dict(sorted(split_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "drops": dict(sorted(drops.items())),
        "audit_counts": dict(sorted(audit_counts.items())),
        "role_target_char_stats": {
            role: describe(values) for role, values in sorted(role_target_lengths.items())
        },
        "role_candidate_count_stats": {
            role: describe(values) for role, values in sorted(role_candidate_counts.items())
        },
        "valid_fraction": args.valid_fraction,
        "split_seed": args.split_seed,
        "max_candidates": args.max_candidates,
        "roles": sorted(roles),
        "target_prefix": args.target_prefix,
        "metrics": metrics,
        "execution_boundary": [
            "non-heldout Phase A SFT data preparation",
            "source split must be calibration/calib/train",
            "online prompt includes question, full online evidence, and evidence-derived candidate text/metadata only",
            "gold and aliases are used only for offline filtering/coverage",
            "teacher scores are used only for offline filtering/summary",
            "no model generation",
            "no optimizer or backward",
            "no SwanLab run",
            "no held-out data",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_json"] = str(summary_path)
    return summary


def make_pair(
    *,
    sample: dict[str, Any],
    split: str,
    role: str,
    source_condition: str,
    target_source: str,
    prompt_text: str,
    target_text: str,
    candidates: list[dict[str, Any]],
    teacher_row: dict[str, Any] | None,
) -> dict[str, Any]:
    sample_id = str(sample["sample_id"])
    return {
        "pair_id": stable_id(sample_id, role, source_condition),
        "sample_id": sample_id,
        "split": split,
        "role": role,
        "agent_id": "solver",
        "cola_task_name": "p2_phase_c_musique_candidate_constrained",
        "source_condition": source_condition,
        "source_row_id": str(sample.get("source", {}).get("row_id", "")),
        "target_source": target_source,
        "context": full_evidence(sample),
        "question": str(sample.get("question", "")).strip(),
        "target_text": target_text,
        "prompt_text": prompt_text,
        "candidate_count": len(candidates),
        "teacher_prediction": str((teacher_row or {}).get("prediction", "")).strip(),
        "teacher_primary_score_offline": float((teacher_row or {}).get("primary_score", 0.0) or 0.0),
        "online_input_boundary": [
            "question",
            "full online evidence",
            "candidate text",
            "candidate rank/rule/source metadata",
            "no gold labels",
            "no alias flags",
            "no teacher correctness",
            "no scorer output",
            "no held-out data",
        ],
    }


def make_candidate_prompt(
    *,
    question: str,
    context: str,
    candidates: list[dict[str, Any]],
) -> str:
    candidate_lines = []
    for index, candidate in enumerate(candidates, start=1):
        text = one_line(candidate.get("text", ""))
        rule = one_line(candidate.get("rule", ""))
        source_title = one_line(candidate.get("source_title", ""))
        evidence_index = one_line(candidate.get("evidence_index", ""))
        meta_parts = []
        if rule:
            meta_parts.append(f"rule={rule}")
        if source_title:
            meta_parts.append(f"source={source_title}")
        if evidence_index:
            meta_parts.append(f"evidence={evidence_index}")
        meta = f" ({'; '.join(meta_parts)})" if meta_parts else ""
        candidate_lines.append(f"[{index}] {text}{meta}")
    return (
        "Select the best final answer from the candidate list using the evidence. "
        "Write only one short answer.\n\n"
        f"Question: {question}\n\n"
        f"Evidence:\n{context}\n\n"
        "Candidates:\n"
        + "\n".join(candidate_lines)
        + "\n\nFinal answer:"
    )


def best_gold_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    gold_candidates = [
        candidate
        for candidate in candidates
        if bool(candidate.get("is_gold_or_alias")) and str(candidate.get("text", "")).strip()
    ]
    if not gold_candidates:
        return None
    return min(gold_candidates, key=lambda item: int(item.get("rank", 10**9) or 10**9))


def teacher_target_in_candidates(teacher_row: dict[str, Any], candidates: list[dict[str, Any]]) -> bool:
    matched = teacher_row.get("matched_candidate", {})
    if isinstance(matched, dict):
        rank = matched.get("candidate_rank")
        try:
            if rank is not None and int(rank) <= len(candidates):
                return True
        except (TypeError, ValueError):
            pass
        matched_text = normalize(str(matched.get("candidate_text", "")))
        if matched_text and any(matched_text == normalize(candidate.get("text", "")) for candidate in candidates):
            return True
    prediction = normalize(str(teacher_row.get("prediction", "")))
    return bool(prediction and any(prediction == normalize(candidate.get("text", "")) for candidate in candidates))


def format_target(answer: str, target_prefix: str) -> str:
    answer = one_line(answer)
    if not answer:
        return ""
    if target_prefix == "answer_only":
        return answer
    if target_prefix == "final_answer":
        return f"Final answer: {answer}"
    raise ValueError(f"unknown target_prefix: {target_prefix}")


def assert_allowed_source_splits(samples: list[dict[str, Any]]) -> None:
    bad = [
        str(sample.get("sample_id", ""))
        for sample in samples
        if str(sample.get("split", "")).lower() not in {"calibration", "calib", "train"}
    ]
    if bad:
        raise ValueError(f"held-out/test/valid samples are forbidden for Phase A SFT prep: {bad[:5]}")


def make_sample_splits(sample_ids: list[str], *, valid_fraction: float, seed: int) -> dict[str, str]:
    shuffled = list(sample_ids)
    random.Random(seed).shuffle(shuffled)
    valid_count = max(1, int(round(len(shuffled) * valid_fraction)))
    valid_ids = set(shuffled[:valid_count])
    return {sample_id: ("valid" if sample_id in valid_ids else "train") for sample_id in sample_ids}


def full_evidence(sample: dict[str, Any]) -> str:
    metadata = sample.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("full_info_observation"):
        return str(metadata["full_info_observation"]).strip()
    return "\n".join(
        str(view.get("private_observation", "")).strip()
        for view in sample.get("agent_views", [])
        if str(view.get("private_observation", "")).strip()
    )


def stable_id(*parts: str) -> str:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"p2a_candidate_sft_{digest}"


def one_line(value: Any) -> str:
    return " ".join(str(value).strip().split())


def normalize(value: Any) -> str:
    return " ".join(
        "".join(ch.lower() if ch.isalnum() else " " for ch in str(value)).split()
    )


def safe_mean(values: list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[index])


def describe(values: list[int]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "mean": float(statistics.fmean(values)),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": float(max(values)),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_no}")
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
