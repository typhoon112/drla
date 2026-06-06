"""Audit P2 capability-gate protocol-repair failures.

This local-only post-processor reads completed ``generations.jsonl`` files from
``run_cola_p2_capability_gate.py`` and compares single-solver versus Role
TextMAS outcomes on the same calibration examples.  It is meant to separate
base-solver floor, role-protocol harm, role-protocol help, and parser failures
before any held-out gate or P2 main table is attempted.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p2_capability_gate/"
    "audit_protocol_repair_failures_20260601"
)


def main() -> None:
    summary = audit_protocol_repair_failures(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-jsonl", action="append", default=[])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if not args.generation_jsonl:
        raise ValueError("Pass at least one --generation-jsonl")
    return args


def audit_protocol_repair_failures(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_records(args.generation_jsonl)
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = (
            str(record["task"]),
            str(record.get("prompt_variant", "")),
            str(record["id"]),
        )
        mode = str(record["mode"])
        if mode in grouped[key]:
            raise ValueError(f"Duplicate record for {key} mode={mode}")
        grouped[key][mode] = record

    pair_rows = []
    example_rows = []
    prediction_rows = []
    for (task, variant), items in sorted(group_by_task_variant(grouped).items()):
        pairs = [modes for modes in items if "single" in modes and "role_textmas" in modes]
        singles = [modes["single"] for modes in items if "single" in modes]
        roles = [modes["role_textmas"] for modes in items if "role_textmas" in modes]
        row = summarize_pairs(task, variant, pairs, singles, roles)
        pair_rows.append(row)
        prediction_rows.extend(prediction_count_rows(task, variant, singles, roles))
        example_rows.extend(select_examples(task, variant, pairs))

    pair_csv = output_dir / "paired_failure_summary.csv"
    prediction_csv = output_dir / "prediction_counts.csv"
    examples_jsonl = output_dir / "failure_examples.jsonl"
    write_csv(pair_csv, pair_rows)
    write_csv(prediction_csv, prediction_rows)
    with examples_jsonl.open("w", encoding="utf-8") as f:
        for row in example_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "created_at": int(time.time()),
        "generation_jsonls": [str(Path(path)) for path in args.generation_jsonl],
        "num_records": len(records),
        "num_task_variant_rows": len(pair_rows),
        "paired_failure_summary_csv": str(pair_csv),
        "prediction_counts_csv": str(prediction_csv),
        "failure_examples_jsonl": str(examples_jsonl),
        "pair_rows": pair_rows,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_json"] = str(summary_path)
    return summary


def read_records(paths: list[str]) -> list[dict[str, Any]]:
    records = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"generation-jsonl does not exist: {path}")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def group_by_task_variant(
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]],
) -> dict[tuple[str, str], list[dict[str, dict[str, Any]]]]:
    by_task_variant: dict[tuple[str, str], list[dict[str, dict[str, Any]]]] = defaultdict(list)
    for (task, variant, _row_id), modes in grouped.items():
        by_task_variant[(task, variant)].append(modes)
    return by_task_variant


def summarize_pairs(
    task: str,
    variant: str,
    pairs: list[dict[str, dict[str, Any]]],
    singles: list[dict[str, Any]],
    roles: list[dict[str, Any]],
) -> dict[str, Any]:
    single_correct = sum(is_correct(record) for record in singles)
    role_correct = sum(is_correct(record) for record in roles)
    single_parseable = sum(is_parseable(record) for record in singles)
    role_parseable = sum(is_parseable(record) for record in roles)
    counts = Counter(pair_category(pair["single"], pair["role_textmas"]) for pair in pairs)
    n_single = len(singles)
    n_role = len(roles)
    n_pair = len(pairs)
    single_acc = single_correct / n_single if n_single else None
    role_acc = role_correct / n_role if n_role else None
    return {
        "task": task,
        "prompt_variant": variant,
        "num_single": n_single,
        "num_role": n_role,
        "num_paired": n_pair,
        "single_accuracy": single_acc,
        "role_accuracy": role_acc,
        "role_minus_single_accuracy": (
            role_acc - single_acc if role_acc is not None and single_acc is not None else None
        ),
        "single_parseable_rate": single_parseable / n_single if n_single else None,
        "role_parseable_rate": role_parseable / n_role if n_role else None,
        "both_correct": counts["both_correct"],
        "single_only_correct": counts["single_only_correct"],
        "role_only_correct": counts["role_only_correct"],
        "both_wrong_parseable": counts["both_wrong_parseable"],
        "role_unparseable": counts["role_unparseable"],
        "single_unparseable": counts["single_unparseable"],
        "both_unparseable": counts["both_unparseable"],
    }


def pair_category(single: dict[str, Any], role: dict[str, Any]) -> str:
    single_ok = is_correct(single)
    role_ok = is_correct(role)
    single_parse = is_parseable(single)
    role_parse = is_parseable(role)
    if single_ok and role_ok:
        return "both_correct"
    if single_ok and not role_ok:
        return "single_only_correct"
    if role_ok and not single_ok:
        return "role_only_correct"
    if not single_parse and not role_parse:
        return "both_unparseable"
    if not role_parse:
        return "role_unparseable"
    if not single_parse:
        return "single_unparseable"
    return "both_wrong_parseable"


def is_correct(record: dict[str, Any]) -> bool:
    return bool(record.get("score", {}).get("correct", False))


def is_parseable(record: dict[str, Any]) -> bool:
    return bool(record.get("score", {}).get("parseable", False))


def prediction_count_rows(
    task: str,
    variant: str,
    singles: list[dict[str, Any]],
    roles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for mode, records in [("single", singles), ("role_textmas", roles)]:
        counts = Counter(prediction(record) for record in records)
        for value, count in sorted(counts.items()):
            rows.append(
                {
                    "task": task,
                    "prompt_variant": variant,
                    "mode": mode,
                    "prediction": value,
                    "count": count,
                    "fraction": count / len(records) if records else 0.0,
                }
            )
    return rows


def prediction(record: dict[str, Any]) -> str:
    value = str(record.get("score", {}).get("prediction", "")).strip()
    return value if value else "<none>"


def select_examples(
    task: str,
    variant: str,
    pairs: list[dict[str, dict[str, Any]]],
    *,
    per_category: int = 3,
) -> list[dict[str, Any]]:
    rows = []
    counts: Counter[str] = Counter()
    for pair in pairs:
        category = pair_category(pair["single"], pair["role_textmas"])
        if counts[category] >= per_category:
            continue
        counts[category] += 1
        single = pair["single"]
        role = pair["role_textmas"]
        rows.append(
            {
                "task": task,
                "prompt_variant": variant,
                "category": category,
                "id": single["id"],
                "single_prediction": prediction(single),
                "single_correct": is_correct(single),
                "role_prediction": prediction(role),
                "role_correct": is_correct(role),
                "role_parseable": is_parseable(role),
                "single_generate_head": str(single.get("generate", ""))[:400],
                "role_generate_head": str(role.get("generate", ""))[:400],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
