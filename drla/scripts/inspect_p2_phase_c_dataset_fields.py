"""Dry-inspect P2 Phase C dataset fields without constructing a benchmark.

This script is local-only.  It reads local JSON/JSONL files that were already
downloaded or intentionally provided for preview, then writes structural field
summaries, record hashes, and source/license metadata.  It does not run models,
train adapters, create SwanLab runs, tune prompts, construct manifests, or
inspect held-out generations.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT_DIR = "/data1/luyifei/drla/outputs/p2_phase_c_data_source_audits/field_inspect_20260601"

KEYWORD_GROUPS = {
    "sample_id": ["id", "_id", "qid", "question_id", "sample_id", "source_sample_id"],
    "question": ["question", "query", "prompt"],
    "answer": ["answer", "gold_answer", "correct_answer", "target", "label"],
    "aliases": ["alias", "aliases", "answer_aliases", "acceptable_answers"],
    "options": ["options", "choices", "candidates"],
    "context": ["context", "paragraph", "paragraphs", "passage", "passages", "full_context"],
    "evidence": [
        "evidence",
        "evidences",
        "evidence_id",
        "evidences_id",
        "support",
        "supports",
        "support_idx",
        "support_id",
        "supporting_facts",
        "is_supporting",
        "paragraph_support_idx",
        "facts",
    ],
    "title": ["title", "titles"],
    "sentences": ["sentence", "sentences"],
    "split": ["split", "subset", "partition"],
    "license": ["license", "licence"],
}


def main() -> None:
    args = parse_args()
    summary = run_selfcheck(args) if args.selfcheck else inspect_sources(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", action="append", default=[], help="Local JSON/JSONL file to inspect.")
    parser.add_argument("--records-key", default="", help="Optional dot path to a list of records inside JSON files.")
    parser.add_argument("--source-name", default="unknown_source")
    parser.add_argument("--source-version", default="unknown_version")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--license", default="unknown")
    parser.add_argument("--max-records", type=int, default=20)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.max_records <= 0:
        raise ValueError("--max-records must be positive")
    if not args.selfcheck and not args.input_file:
        raise ValueError("Pass at least one --input-file, or use --selfcheck")
    return args


def inspect_sources(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    file_summaries = []
    for input_file in args.input_file:
        path = Path(input_file)
        records, file_warnings = load_records(path, args.records_key, args.max_records)
        warnings.extend({"file": str(path), **warning} for warning in file_warnings)
        all_records.extend(records)
        file_summaries.append(
            {
                "path": str(path),
                "records_loaded": len(records),
                "sha256": file_sha256(path),
            }
        )

    field_stats = collect_field_stats(all_records)
    candidate_fields = find_candidate_fields(field_stats)
    record_hashes = [
        stable_record_hash(record)
        for record in all_records[: min(len(all_records), args.max_records)]
    ]
    audit = {
        "source": {
            "name": args.source_name,
            "version": args.source_version,
            "url": args.source_url,
            "license": args.license,
        },
        "input_files": file_summaries,
        "records_loaded": len(all_records),
        "records_key": args.records_key,
        "max_records": args.max_records,
        "field_stats": field_stats,
        "candidate_fields": candidate_fields,
        "record_hashes": record_hashes,
        "private_view_shardability_hint": infer_shardability(candidate_fields),
        "shortcut_risk_hint": infer_shortcut_risk(candidate_fields),
        "warnings": warnings,
        "execution_boundary": [
            "local-only data field dry inspection",
            "reads local preview files only",
            "writes field summaries and hashes, not benchmark samples",
            "does not construct manifests",
            "does not run model generation",
            "does not run optimizer or backward",
            "does not create SwanLab runs",
            "does not inspect held-out generations or tune prompts",
        ],
    }
    status = "pass" if all_records else "fail"
    metrics = {
        "num_records_loaded": len(all_records),
        "num_input_files": len(file_summaries),
        "num_field_paths": len(field_stats),
        "num_warnings": len(warnings),
        "status_pass": int(status == "pass"),
    }

    field_summary_path = output_dir / "field_summary.json"
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    field_summary_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "created_at": int(time.time()),
        "status": status,
        "field_summary_json": str(field_summary_path),
        "metrics_jsonl": str(metrics_path),
        "records_loaded": len(all_records),
        "candidate_fields": candidate_fields,
        "private_view_shardability_hint": audit["private_view_shardability_hint"],
        "shortcut_risk_hint": audit["shortcut_risk_hint"],
        "warnings": warnings,
        "execution_boundary": audit["execution_boundary"],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_json"] = str(summary_path)
    if status != "pass":
        raise ValueError("No records were loaded for field inspection")
    return summary


def run_selfcheck(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    with tempfile.TemporaryDirectory() as tmpdir:
        preview_path = Path(tmpdir) / "preview.jsonl"
        rows = [
            {
                "id": "toy_1",
                "question": "Which color follows from both clues?",
                "answer": "blue",
                "supporting_facts": [{"title": "sky", "sentences": ["The daytime sky is blue."]}],
                "context": [{"title": "distractor", "sentences": ["Grass can be green."]}],
            },
            {
                "id": "toy_2",
                "question": "Which color follows from both clues?",
                "answer": "red",
                "supporting_facts": [{"title": "fruit", "sentences": ["Ripe strawberries can be red."]}],
                "context": [{"title": "distractor", "sentences": ["Clouds can be white."]}],
            },
        ]
        with preview_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        inspect_args = argparse.Namespace(
            input_file=[str(preview_path)],
            records_key="",
            source_name="selfcheck_toy",
            source_version="v0",
            source_url="",
            license="not_applicable",
            max_records=args.max_records,
            output_dir=str(output_dir),
            overwrite=True,
        )
        summary = inspect_sources(inspect_args)
    required_groups = {"sample_id", "question", "answer", "evidence", "context", "title", "sentences"}
    present_groups = {
        group
        for group, paths in summary["candidate_fields"].items()
        if paths
    }
    missing = sorted(required_groups - present_groups)
    if missing:
        raise AssertionError(f"Field inspection self-check missing candidate groups: {missing}")
    summary["selfcheck"] = {"status": "pass", "required_groups": sorted(required_groups)}
    return summary


def load_records(path: Path, records_key: str, max_records: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(path)
    if path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz"):
        return read_jsonl(path, max_records), warnings
    if path.name.endswith(".json") or path.name.endswith(".json.gz"):
        obj = json.loads(read_text(path))
        selected = select_records(obj, records_key, warnings)
        return normalize_records(selected, max_records), warnings
    raise ValueError(f"Unsupported file type for dry inspection: {path}")


def read_jsonl(path: Path, max_records: int) -> list[dict[str, Any]]:
    records = []
    with open_text(path) as f:
        for line_no, line in enumerate(f, start=1):
            if len(records) >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            records.append(row)
    return records


def select_records(obj: Any, records_key: str, warnings: list[dict[str, Any]]) -> Any:
    if records_key:
        selected = obj
        for key in records_key.split("."):
            if isinstance(selected, dict) and key in selected:
                selected = selected[key]
            else:
                raise KeyError(f"records_key not found: {records_key}")
        return selected
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        list_keys = [key for key, value in obj.items() if isinstance(value, list)]
        if len(list_keys) == 1:
            warnings.append(
                {
                    "path": list_keys[0],
                    "message": "records_key was empty; using the only list-valued top-level key",
                }
            )
            return obj[list_keys[0]]
        warnings.append(
            {
                "path": "<root>",
                "message": "records_key was empty; treating the JSON object as one record",
            }
        )
        return [obj]
    raise ValueError("JSON root must be an object or list")


def normalize_records(selected: Any, max_records: int) -> list[dict[str, Any]]:
    if not isinstance(selected, list):
        raise ValueError("selected records must be a list")
    records = []
    for index, row in enumerate(selected[:max_records]):
        if not isinstance(row, dict):
            raise ValueError(f"selected record at index {index} is not an object")
        records.append(row)
    return records


def collect_field_stats(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for record in records:
        for path, value in walk_fields(record):
            entry = stats.setdefault(
                path,
                {
                    "count": 0,
                    "types": Counter(),
                    "string_lengths": [],
                    "list_lengths": [],
                    "numeric_values": [],
                },
            )
            entry["count"] += 1
            entry["types"][type_name(value)] += 1
            if isinstance(value, str):
                entry["string_lengths"].append(len(value))
            elif isinstance(value, list):
                entry["list_lengths"].append(len(value))
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                entry["numeric_values"].append(float(value))
    return {path: summarize_entry(entry) for path, entry in sorted(stats.items())}


def walk_fields(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if prefix:
        yield prefix, value
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from walk_fields(child, child_prefix)
    elif isinstance(value, list):
        for child in value[:5]:
            child_prefix = f"{prefix}[]" if prefix else "[]"
            yield from walk_fields(child, child_prefix)


def summarize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "count": entry["count"],
        "types": dict(sorted(entry["types"].items())),
    }
    if entry["string_lengths"]:
        summary["string_length"] = summarize_numbers(entry["string_lengths"])
    if entry["list_lengths"]:
        summary["list_length"] = summarize_numbers(entry["list_lengths"])
    if entry["numeric_values"]:
        summary["numeric_value"] = summarize_numbers(entry["numeric_values"])
    return summary


def summarize_numbers(values: list[float]) -> dict[str, float]:
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(statistics.fmean(values)),
    }


def find_candidate_fields(field_stats: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    candidates: dict[str, list[str]] = {group: [] for group in KEYWORD_GROUPS}
    for path in field_stats:
        leaf = path.split(".")[-1].replace("[]", "").lower()
        for group, keywords in KEYWORD_GROUPS.items():
            if leaf in keywords:
                candidates[group].append(path)
    return {group: sorted(set(paths)) for group, paths in candidates.items()}


def infer_shardability(candidate_fields: dict[str, list[str]]) -> str:
    if candidate_fields.get("evidence") and (candidate_fields.get("context") or candidate_fields.get("sentences")):
        return "possible_evidence_split"
    if candidate_fields.get("context") or candidate_fields.get("sentences"):
        return "possible_context_split_needs_support_labels"
    return "unknown_no_evidence_like_fields_detected"


def infer_shortcut_risk(candidate_fields: dict[str, list[str]]) -> str:
    has_question = bool(candidate_fields.get("question"))
    has_answer = bool(candidate_fields.get("answer"))
    has_evidence = bool(candidate_fields.get("evidence") or candidate_fields.get("context"))
    if has_question and has_answer and not has_evidence:
        return "high_question_answer_without_evidence_fields"
    if has_question and has_answer and has_evidence:
        return "must_measure_question_only_and_shuffled_controls"
    return "unknown_requires_manual_review"


def stable_record_hash(record: dict[str, Any]) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open_binary(path) as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_text(path: Path) -> str:
    with open_text(path) as f:
        return f.read()


def open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def open_binary(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rb")
    return path.open("rb")


def type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


if __name__ == "__main__":
    main()
