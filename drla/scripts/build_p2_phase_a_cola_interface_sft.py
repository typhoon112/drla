"""Build non-heldout SFT pairs for Phase A CoLA interface adaptation.

This script prepares data only.  It does not train, call models, inspect held-out
generations, or create SwanLab runs.  The intended next consumer is a supervised
CoLA task-format / role-interface adapter trained on GPU with SwanLab cloud.
Allowed source splits are calibration/calib/train.  Held-out/test/valid
benchmark records are intentionally forbidden here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST_JSON = (
    "/data1/luyifei/drla/outputs/p2_phase_c_manifests/"
    "musique_calibration_manifest_200_seed20260601/manifest.json"
)
DEFAULT_TEACHER_GENERATIONS_JSONL = (
    "/data1/luyifei/drla/outputs/p2_phase_c_text_agent_runs/"
    "musique_local_qwen3_8b_fp8_v1_strict_wrong_full200_merged_20260601/generations.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p2_phase_a_cola_interface_sft/"
    "musique_calibration_qwen_teacher_v1_20260605"
)


def main() -> None:
    summary = build_sft(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--teacher-generations-jsonl", default=DEFAULT_TEACHER_GENERATIONS_JSONL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260605)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all allowed source samples.")
    parser.add_argument(
        "--solver-context-mode",
        choices=["full_evidence", "support_only", "both"],
        default="full_evidence",
        help="Which solver context variants to emit for answer-focused curriculum diagnostics.",
    )
    parser.add_argument(
        "--solver-target-mode",
        choices=["answer_only", "final_answer_then_support"],
        default="answer_only",
        help=(
            "Solver target text. final_answer_then_support keeps the first line "
            "scorable as Final answer while adding selected support supervision."
        ),
    )
    parser.add_argument(
        "--disable-teacher-roles",
        action="store_true",
        help="Emit only solver pairs and skip solver_textmas_matched/evidence-agent teacher pairs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_sft(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 < args.valid_fraction < 0.5:
        raise ValueError("--valid-fraction must be in (0, 0.5)")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
    samples = list(manifest.get("samples", []))
    if args.max_samples:
        samples = samples[: args.max_samples]
    assert_allowed_source_splits(samples)
    teacher_rows = read_jsonl(Path(args.teacher_generations_jsonl))
    teacher_by_key = {
        (str(row.get("sample_id", "")), str(row.get("condition", ""))): row for row in teacher_rows
    }
    split_by_sample = make_sample_splits(
        [str(sample["sample_id"]) for sample in samples],
        valid_fraction=args.valid_fraction,
        seed=args.split_seed,
    )

    pairs: list[dict[str, Any]] = []
    drops = Counter()
    for sample in samples:
        sample_id = str(sample["sample_id"])
        split = split_by_sample[sample_id]
        gold_answer = str(sample.get("scoring", {}).get("gold_answer", "")).strip()
        question = str(sample.get("question", "")).strip()
        if not gold_answer or not question:
            drops["missing_question_or_gold"] += 1
            continue

        solver_contexts = []
        if args.solver_context_mode in {"full_evidence", "both"}:
            solver_contexts.append(
                {
                    "role": "solver_full_info",
                    "context": full_evidence(sample),
                    "source_condition": "single_full_info",
                    "target_source": "manifest_gold_answer",
                }
            )
        if args.solver_context_mode in {"support_only", "both"}:
            support_context = support_only_evidence(sample)
            if support_context:
                solver_contexts.append(
                    {
                        "role": "solver_support_only",
                        "context": support_context,
                        "source_condition": "single_full_info_support_only_diagnostic",
                        "target_source": "manifest_gold_answer_support_only_context",
                    }
                )
            else:
                drops["missing_support_only_context"] += 1

        for item in solver_contexts:
            pairs.append(
                make_solver_pair(
                    sample=sample,
                    split=split,
                    role=item["role"],
                    context=item["context"],
                    target_text=solver_target_text(
                        sample,
                        gold_answer=gold_answer,
                        mode=args.solver_target_mode,
                    ),
                    source_condition=item["source_condition"],
                    source_row_id="",
                    target_source=solver_target_source(item["target_source"], args.solver_target_mode),
                )
            )

        if args.disable_teacher_roles:
            continue

        matched_row = teacher_by_key.get((sample_id, "textmas_matched"))
        if not matched_row:
            drops["missing_textmas_matched_teacher_row"] += 1
            continue
        agent_messages = matched_row.get("agent_messages", [])
        if not isinstance(agent_messages, list) or not agent_messages:
            drops["missing_teacher_agent_messages"] += 1
            continue
        pairs.append(
            make_solver_pair(
                sample=sample,
                split=split,
                role="solver_textmas_matched",
                context=format_agent_message_context(agent_messages),
                target_text=gold_answer,
                source_condition="textmas_matched",
                source_row_id=str(matched_row.get("row_id", "")),
                target_source="manifest_gold_answer",
            )
        )
        observations = observations_by_agent(sample)
        for message in agent_messages:
            if not isinstance(message, dict):
                drops["malformed_teacher_agent_message"] += 1
                continue
            agent_id = str(message.get("agent_id", ""))
            role_name = str(message.get("role", ""))
            observation = observations.get(agent_id) or observations.get(role_name)
            target_text = str(message.get("message", "")).strip()
            if not observation or not target_text:
                drops["missing_observation_or_teacher_message"] += 1
                continue
            pairs.append(
                make_evidence_agent_pair(
                    sample=sample,
                    split=split,
                    agent_id=agent_id,
                    role_name=role_name,
                    private_observation=observation,
                    target_text=target_text,
                    source_row_id=str(matched_row.get("row_id", "")),
                )
            )

    pair_path = output_dir / "sft_pairs.jsonl"
    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    write_jsonl(pair_path, pairs)
    write_jsonl(train_path, [row for row in pairs if row["split"] == "train"])
    write_jsonl(valid_path, [row for row in pairs if row["split"] == "valid"])

    role_counts = Counter(str(row["role"]) for row in pairs)
    split_counts = Counter(str(row["split"]) for row in pairs)
    metrics = {
        "num_samples": len(samples),
        "num_pairs": len(pairs),
        "num_train_pairs": split_counts["train"],
        "num_valid_pairs": split_counts["valid"],
        "num_roles": len(role_counts),
    }
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "created_at": int(time.time()),
        "status": "pass",
        "manifest_json": args.manifest_json,
        "teacher_generations_jsonl": args.teacher_generations_jsonl,
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
        "valid_fraction": args.valid_fraction,
        "split_seed": args.split_seed,
        "solver_context_mode": args.solver_context_mode,
        "solver_target_mode": args.solver_target_mode,
        "disable_teacher_roles": bool(args.disable_teacher_roles),
        "execution_boundary": [
            "non-heldout Phase A SFT data preparation",
            "source split must be calibration/calib/train",
            "no model generation",
            "no optimizer or backward",
            "no SwanLab run",
            "teacher agent messages are from a non-heldout Qwen3-8B-FP8 teacher-message run",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_json"] = str(summary_path)
    return summary


def make_solver_pair(
    *,
    sample: dict[str, Any],
    split: str,
    role: str,
    context: str,
    target_text: str,
    source_condition: str,
    source_row_id: str,
    target_source: str,
) -> dict[str, Any]:
    sample_id = str(sample["sample_id"])
    question = str(sample.get("question", "")).strip()
    return {
        "pair_id": stable_id(sample_id, role, source_condition),
        "sample_id": sample_id,
        "split": split,
        "role": role,
        "agent_id": "solver",
        "cola_task_name": "squad",
        "source_condition": source_condition,
        "source_row_id": source_row_id,
        "target_source": target_source,
        "context": context,
        "question": question,
        "target_text": target_text,
        "prompt_text": f"Context: {context}\nQuestion: {question}\nAnswer:",
        "online_input_boundary": [
            "question",
            "context",
            "no scorer output",
            "no held-out data",
        ],
    }


def make_evidence_agent_pair(
    *,
    sample: dict[str, Any],
    split: str,
    agent_id: str,
    role_name: str,
    private_observation: str,
    target_text: str,
    source_row_id: str,
) -> dict[str, Any]:
    sample_id = str(sample["sample_id"])
    prompt_text = (
        "Read the evidence below and write only the useful facts.\n\n"
        f"Evidence:\n{private_observation}\n\n"
        "Useful facts:"
    )
    return {
        "pair_id": stable_id(sample_id, "evidence_agent_teacher", agent_id or role_name),
        "sample_id": sample_id,
        "split": split,
        "role": "evidence_agent_teacher",
        "agent_id": agent_id,
        "agent_role": role_name,
        "cola_task_name": "p2_phase_c_musique",
        "source_condition": "textmas_matched",
        "source_row_id": source_row_id,
        "target_source": "qwen3_8b_fp8_teacher_agent_message",
        "context": private_observation,
        "question": "write useful facts from private evidence",
        "target_text": target_text,
        "prompt_text": prompt_text,
        "online_input_boundary": [
            "private_observation",
            "no scorer output",
            "no held-out data",
        ],
    }


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
        return str(metadata["full_info_observation"])
    return "\n".join(
        str(view.get("private_observation", "")).strip()
        for view in sample.get("agent_views", [])
        if str(view.get("private_observation", "")).strip()
    )


def solver_target_text(sample: dict[str, Any], *, gold_answer: str, mode: str) -> str:
    if mode == "answer_only":
        return gold_answer
    if mode == "final_answer_then_support":
        support = compact_selected_support(sample, gold_answer=gold_answer)
        if support:
            return f"Final answer: {gold_answer}\nSelected support: {support}"
        return f"Final answer: {gold_answer}"
    raise ValueError(f"unknown solver_target_mode: {mode}")


def solver_target_source(base: str, mode: str) -> str:
    return base if mode == "answer_only" else f"{base}_{mode}"


def support_only_evidence(sample: dict[str, Any]) -> str:
    source = full_evidence(sample)
    lines = [
        line.strip()
        for line in source.splitlines()
        if re.search(r"\(\s*support\s*\)", line, flags=re.IGNORECASE)
    ]
    return "\n".join(line for line in lines if line)


def compact_selected_support(sample: dict[str, Any], *, gold_answer: str, max_chars: int = 900) -> str:
    support_lines = [
        re.sub(r"\s+", " ", line.strip())
        for line in support_only_evidence(sample).splitlines()
        if line.strip()
    ]
    if not support_lines:
        return ""
    aliases = [
        str(alias)
        for alias in sample.get("scoring", {}).get("answer_aliases", []) or []
        if str(alias).strip()
    ]
    answer_forms = [gold_answer, *aliases]
    answer_norms = {normalize_for_match(text) for text in answer_forms if normalize_for_match(text)}
    answer_bearing = [
        line for line in support_lines if any(answer in normalize_for_match(line) for answer in answer_norms)
    ]
    ordered = answer_bearing + [line for line in support_lines if line not in set(answer_bearing)]
    selected: list[str] = []
    total = 0
    for line in ordered:
        if total and total + 1 + len(line) > max_chars and selected:
            break
        selected.append(line)
        total += len(line) + (1 if total else 0)
    return " ".join(selected)


def normalize_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def observations_by_agent(sample: dict[str, Any]) -> dict[str, str]:
    observations = {}
    for view in sample.get("agent_views", []):
        observation = str(view.get("private_observation", "")).strip()
        agent_id = str(view.get("agent_id", ""))
        role = str(view.get("role", ""))
        if agent_id:
            observations[agent_id] = observation
        if role:
            observations[role] = observation
    return observations


def format_agent_message_context(agent_messages: list[Any]) -> str:
    lines = []
    for message in agent_messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or message.get("agent_id") or "agent")
        content = str(message.get("message", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def stable_id(*parts: str) -> str:
    text = "::".join(parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"p2a_sft_{digest}"


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
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
