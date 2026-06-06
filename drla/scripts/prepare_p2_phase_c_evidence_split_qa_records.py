"""Prepare P2 Phase C evidence-split QA sample records.

This script converts local HotpotQA / MuSiQue / 2Wiki-style JSONL rows into
``p2_phase_c_manifest_v0`` sample records.  It is local-only: it does not
download data, run models, train adapters, inspect held-out generations, or
create SwanLab runs.  The output is a manifest-ready records JSONL that must
still pass ``build_p2_phase_c_manifest.py`` and leakage/scorer audits before
any model evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = "/data1/luyifei/drla/outputs/p2_phase_c_records/evidence_split_qa_20260601"

REQUIRED_BASELINES = [
    "single_q_only",
    "single_full_info",
    "textmas_matched",
    "textmas_no_message",
    "textmas_shuffled_message",
    "textmas_wrong_evidence_or_wrong_shard",
    "textmas_compressed_state",
]


def main() -> None:
    summary = prepare_records(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--source-type", required=True, choices=["hotpotqa", "musique", "2wiki"])
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--source-url", default="")
    parser.add_argument("--license", default="unknown")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means no cap after filtering.")
    parser.add_argument("--max-evidence-chars-per-item", type=int, default=0, help="0 means no truncation.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_records(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = read_jsonl(Path(args.input_jsonl))
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for raw_index, row in enumerate(raw_rows):
        if args.max_samples and len(records) >= args.max_samples:
            break
        built = build_record(args, row, raw_index)
        if built["record"] is None:
            skipped.append(built["skip"])
            continue
        records.append(built["record"])

    records_path = output_dir / "records.jsonl"
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(records_path, records)
    shortcut_audit = audit_answer_string_in_shards(records)
    metrics = {
        "num_input_rows": len(raw_rows),
        "num_output_records": len(records),
        "num_skipped": len(skipped),
        "answer_string_in_agent_a": shortcut_audit["answer_string_in_agent_a"],
        "answer_string_in_agent_b": shortcut_audit["answer_string_in_agent_b"],
        "answer_string_in_both_agents": shortcut_audit["answer_string_in_both_agents"],
        "status_pass": int(bool(records)),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "created_at": int(time.time()),
        "status": "pass" if records else "fail",
        "input_jsonl": args.input_jsonl,
        "records_jsonl": str(records_path),
        "metrics_jsonl": str(metrics_path),
        "num_input_rows": len(raw_rows),
        "num_output_records": len(records),
        "num_skipped": len(skipped),
        "skip_reasons": count_skip_reasons(skipped),
        "skipped_examples": skipped[:10],
        "shortcut_audit": shortcut_audit,
        "source_type": args.source_type,
        "task_name": args.task_name,
        "split": args.split,
        "execution_boundary": [
            "local-only evidence-split QA record preparation",
            "reads local source JSONL only",
            "writes manifest-ready sample records",
            "does not download data",
            "does not run model generation",
            "does not run optimizer or backward",
            "does not create SwanLab runs",
            "does not inspect held-out generations or tune prompts",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_json"] = str(summary_path)
    if not records:
        raise ValueError(f"No records could be built from {args.input_jsonl}")
    return summary


def build_record(args: argparse.Namespace, row: dict[str, Any], raw_index: int) -> dict[str, Any]:
    if args.source_type == "hotpotqa":
        parsed = parse_hotpotqa(row)
    elif args.source_type == "musique":
        parsed = parse_musique(row)
    elif args.source_type == "2wiki":
        parsed = parse_2wiki(row)
    else:
        raise ValueError(f"unknown source_type: {args.source_type}")

    if parsed["skip_reason"]:
        return {"record": None, "skip": {"raw_index": raw_index, "reason": parsed["skip_reason"]}}

    support_items = parsed["support_items"]
    distractor_items = parsed["distractor_items"]
    if len(support_items) < 2:
        return {
            "record": None,
            "skip": {
                "raw_index": raw_index,
                "source_sample_id": parsed["source_sample_id"],
                "reason": "need_at_least_two_support_items_for_split_evidence",
            },
        }

    shard_a, shard_b = split_support_items(support_items)
    add_distractors(shard_a, shard_b, distractor_items)
    source_sample_id = parsed["source_sample_id"] or f"row_{raw_index}"
    sample_id = f"p2c_{args.source_type}_{args.split}_{slug(source_sample_id)}"
    source = {
        "name": args.source_name,
        "version": args.source_version,
        "source_sample_id": str(source_sample_id),
    }
    if args.source_url:
        source["url"] = args.source_url

    full_info_observation = format_evidence_items(
        support_items + distractor_items,
        max_chars_per_item=args.max_evidence_chars_per_item,
    )
    record = {
        "sample_id": sample_id,
        "family": "evidence_split_qa",
        "task_name": args.task_name,
        "split": args.split,
        "source": source,
        "question": parsed["question"],
        "public_context": (
            "Evidence is split across agents. Each agent must report only the "
            "useful information from its private shard; the final solver must "
            "combine the messages to answer."
        ),
        "agent_views": [
            make_agent_view("agent_a", "evidence_holder_a", shard_a, args.max_evidence_chars_per_item),
            make_agent_view("agent_b", "evidence_holder_b", shard_b, args.max_evidence_chars_per_item),
        ],
        "scoring": make_scoring(parsed),
        "leakage_audit": {
            "gold_in_online_prompt": False,
            "scorer_output_in_online_prompt": False,
            "full_evidence_available_to_split_agent": False,
            "heldout_used_for_prompt_repair": False,
            "notes": "Gold/scorer/full evidence are offline-only. Split agents receive complementary private evidence shards.",
        },
        "baselines_required": REQUIRED_BASELINES,
        "metadata": {
            "license": args.license,
            "source_type": args.source_type,
            "raw_index": raw_index,
            "raw_record_sha256": stable_record_hash(row),
            "support_count": len(support_items),
            "distractor_count": len(distractor_items),
            "agent_a_support_count": count_support_items(shard_a),
            "agent_b_support_count": count_support_items(shard_b),
            "full_info_observation": full_info_observation,
            "source_fields": parsed["source_fields"],
            "is_calibration_draft": args.split == "calibration",
            "use_for_model_eval": False,
        },
    }
    return {"record": record, "skip": None}


def parse_hotpotqa(row: dict[str, Any]) -> dict[str, Any]:
    question = as_str(row.get("question"))
    answer = row.get("answer")
    source_sample_id = as_str(row.get("id"))
    if not question or answer in (None, ""):
        return skipped("missing_question_or_answer", source_sample_id)
    context = row.get("context", {})
    support = row.get("supporting_facts", {})
    title_to_sentences = context_title_to_sentences(context)
    support_pairs = support_title_sent_pairs(support)
    support_items = []
    used_titles = set()
    for title, sent_id in support_pairs:
        sentences = title_to_sentences.get(title)
        if not sentences:
            continue
        used_titles.add(title)
        text = select_sentence_or_all(sentences, sent_id)
        support_items.append({"title": title, "text": text, "kind": "support"})
    distractors = [
        {"title": title, "text": " ".join(sentences), "kind": "distractor"}
        for title, sentences in title_to_sentences.items()
        if title not in used_titles
    ]
    return parsed(
        source_sample_id=source_sample_id,
        question=question,
        answer=answer,
        aliases=[],
        support_items=dedupe_items(support_items),
        distractor_items=distractors[:4],
        source_fields=["id", "question", "answer", "context", "supporting_facts"],
    )


def parse_musique(row: dict[str, Any]) -> dict[str, Any]:
    question = as_str(row.get("question"))
    answer = row.get("answer")
    source_sample_id = as_str(row.get("id"))
    if not question or answer in (None, ""):
        return skipped("missing_question_or_answer", source_sample_id)
    paragraphs = row.get("paragraphs", [])
    support_indices = set()
    for step in row.get("question_decomposition", []) or []:
        if isinstance(step, dict) and isinstance(step.get("paragraph_support_idx"), int):
            support_indices.add(step["paragraph_support_idx"])
    support_items = []
    distractors = []
    for paragraph in paragraphs if isinstance(paragraphs, list) else []:
        if not isinstance(paragraph, dict):
            continue
        idx = paragraph.get("idx")
        item = {
            "title": as_str(paragraph.get("title")) or f"paragraph_{idx}",
            "text": as_str(paragraph.get("paragraph_text")),
            "kind": "support" if paragraph.get("is_supporting") is True or idx in support_indices else "distractor",
        }
        if not item["text"]:
            continue
        if item["kind"] == "support":
            support_items.append(item)
        else:
            distractors.append(item)
    aliases = [str(alias) for alias in row.get("answer_aliases", []) if isinstance(alias, str)]
    return parsed(
        source_sample_id=source_sample_id,
        question=question,
        answer=answer,
        aliases=aliases,
        support_items=dedupe_items(support_items),
        distractor_items=distractors[:4],
        source_fields=[
            "id",
            "question",
            "answer",
            "answer_aliases",
            "paragraphs",
            "paragraphs.is_supporting",
            "question_decomposition.paragraph_support_idx",
        ],
    )


def parse_2wiki(row: dict[str, Any]) -> dict[str, Any]:
    question = as_str(row.get("question"))
    answer = row.get("answer")
    source_sample_id = as_str(row.get("id"))
    if not question or answer in (None, ""):
        return skipped("missing_question_or_answer", source_sample_id)
    context = row.get("context", {})
    support = row.get("supporting_facts", {})
    title_to_sentences = context_title_to_sentences(context)
    support_pairs = support_title_sent_pairs(support)
    support_items = []
    used_titles = set()
    for title, sent_id in support_pairs:
        sentences = title_to_sentences.get(title)
        if not sentences:
            continue
        used_titles.add(title)
        text = select_sentence_or_all(sentences, sent_id)
        support_items.append({"title": title, "text": text, "kind": "support"})
    distractors = [
        {"title": title, "text": " ".join(sentences), "kind": "distractor"}
        for title, sentences in title_to_sentences.items()
        if title not in used_titles
    ]
    return parsed(
        source_sample_id=source_sample_id,
        question=question,
        answer=answer,
        aliases=[],
        support_items=dedupe_items(support_items),
        distractor_items=distractors[:4],
        source_fields=["id", "question", "answer", "context", "supporting_facts", "evidences"],
    )


def parsed(
    *,
    source_sample_id: str,
    question: str,
    answer: Any,
    aliases: list[str],
    support_items: list[dict[str, str]],
    distractor_items: list[dict[str, str]],
    source_fields: list[str],
) -> dict[str, Any]:
    return {
        "skip_reason": "",
        "source_sample_id": source_sample_id,
        "question": question,
        "answer": answer,
        "aliases": aliases,
        "support_items": support_items,
        "distractor_items": distractor_items,
        "source_fields": source_fields,
    }


def skipped(reason: str, source_sample_id: str = "") -> dict[str, Any]:
    return {
        "skip_reason": reason,
        "source_sample_id": source_sample_id,
        "question": "",
        "answer": "",
        "aliases": [],
        "support_items": [],
        "distractor_items": [],
        "source_fields": [],
    }


def context_title_to_sentences(context: Any) -> dict[str, list[str]]:
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        if isinstance(titles, list) and isinstance(sentences, list):
            result = {}
            for title, sent_list in zip(titles, sentences):
                if isinstance(sent_list, list):
                    result[as_str(title)] = [as_str(sent) for sent in sent_list if as_str(sent)]
            return result
    if isinstance(context, list):
        result = {}
        for item in context:
            if not isinstance(item, dict):
                continue
            title = as_str(item.get("title"))
            sent_list = item.get("sentences", [])
            if title and isinstance(sent_list, list):
                result[title] = [as_str(sent) for sent in sent_list if as_str(sent)]
        return result
    return {}


def support_title_sent_pairs(support: Any) -> list[tuple[str, int | None]]:
    if isinstance(support, dict):
        titles = support.get("title", [])
        sent_ids = support.get("sent_id", [])
        if isinstance(titles, list):
            pairs = []
            for index, title in enumerate(titles):
                sent_id = sent_ids[index] if isinstance(sent_ids, list) and index < len(sent_ids) else None
                pairs.append((as_str(title), sent_id if isinstance(sent_id, int) else None))
            return pairs
    if isinstance(support, list):
        pairs = []
        for item in support:
            if isinstance(item, dict):
                sent_id = item.get("sent_id")
                pairs.append((as_str(item.get("title")), sent_id if isinstance(sent_id, int) else None))
            elif isinstance(item, (list, tuple)) and item:
                sent_id = item[1] if len(item) > 1 and isinstance(item[1], int) else None
                pairs.append((as_str(item[0]), sent_id))
        return pairs
    return []


def select_sentence_or_all(sentences: list[str], sent_id: int | None) -> str:
    if isinstance(sent_id, int) and 0 <= sent_id < len(sentences):
        return sentences[sent_id]
    return " ".join(sentences)


def split_support_items(support_items: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    shard_a = []
    shard_b = []
    for index, item in enumerate(support_items):
        (shard_a if index % 2 == 0 else shard_b).append(item)
    return shard_a, shard_b


def add_distractors(
    shard_a: list[dict[str, str]],
    shard_b: list[dict[str, str]],
    distractor_items: list[dict[str, str]],
) -> None:
    for index, item in enumerate(distractor_items[:4]):
        (shard_a if index % 2 == 0 else shard_b).append(item)


def make_agent_view(
    agent_id: str,
    role: str,
    evidence_items: list[dict[str, str]],
    max_chars_per_item: int,
) -> dict[str, Any]:
    return {
        "agent_id": agent_id,
        "role": role,
        "private_observation": format_evidence_items(evidence_items, max_chars_per_item=max_chars_per_item),
        "allowed_output_contract": (
            "Summarize only the useful information in this private evidence shard. "
            "Do not guess the final answer unless the shard itself proves it; do "
            "not include gold labels or scorer information."
        ),
        "forbidden_fields": ["gold_answer", "scorer_output", "full_evidence"],
    }


def make_scoring(parsed_row: dict[str, Any]) -> dict[str, Any]:
    scoring = {
        "type": "normalized_f1",
        "gold_answer": parsed_row["answer"],
    }
    if parsed_row["aliases"]:
        scoring["answer_aliases"] = parsed_row["aliases"]
    return scoring


def format_evidence_items(evidence_items: list[dict[str, str]], max_chars_per_item: int) -> str:
    lines = []
    for index, item in enumerate(evidence_items, start=1):
        text = item["text"]
        if max_chars_per_item and len(text) > max_chars_per_item:
            text = text[:max_chars_per_item].rstrip() + " ..."
        lines.append(f"[{index}] ({item['kind']}) {item['title']}: {text}")
    return "\n".join(lines)


def dedupe_items(items: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    deduped = []
    for item in items:
        key = (item["title"], item["text"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def count_support_items(items: list[dict[str, str]]) -> int:
    return sum(1 for item in items if item.get("kind") == "support")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_record_hash(record: dict[str, Any]) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value[:120] or "unknown"


def as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def count_skip_reasons(skipped: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in skipped:
        reason = str(item.get("reason", "unknown"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def audit_answer_string_in_shards(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "answer_string_in_agent_a": 0,
        "answer_string_in_agent_b": 0,
        "answer_string_in_both_agents": 0,
    }
    for record in records:
        answer = as_str(record.get("scoring", {}).get("gold_answer")).lower()
        views = record.get("agent_views", [])
        if not answer or len(views) < 2:
            continue
        in_a = answer in as_str(views[0].get("private_observation")).lower()
        in_b = answer in as_str(views[1].get("private_observation")).lower()
        counts["answer_string_in_agent_a"] += int(in_a)
        counts["answer_string_in_agent_b"] += int(in_b)
        counts["answer_string_in_both_agents"] += int(in_a and in_b)
    total = len(records)
    return {
        **counts,
        "num_records": total,
        "answer_string_in_agent_a_rate": counts["answer_string_in_agent_a"] / total if total else 0.0,
        "answer_string_in_agent_b_rate": counts["answer_string_in_agent_b"] / total if total else 0.0,
        "answer_string_in_both_agents_rate": counts["answer_string_in_both_agents"] / total if total else 0.0,
        "interpretation": (
            "Answer strings may naturally appear in support evidence. This is not "
            "a gold-label field leak by itself, but it requires single_q_only, "
            "no_message, shuffled_message, and wrong_evidence controls before "
            "claiming communication necessity."
        ),
    }


if __name__ == "__main__":
    main()
