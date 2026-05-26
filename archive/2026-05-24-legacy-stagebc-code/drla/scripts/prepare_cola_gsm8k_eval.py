"""Prepare GSM8K JSONL inputs for official Cola-DLM inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def format_question(question: str, *, qa_template: bool) -> str:
    if not qa_template:
        return question
    return f"Question: {question}\nAnswer:"


def convert_gsm8k_for_cola(
    input_jsonl: Path,
    output_jsonl: Path,
    *,
    max_samples: int | None,
    qa_template: bool = False,
    include_target_text: bool = False,
) -> dict[str, Any]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_jsonl.open(encoding="utf-8") as src, output_jsonl.open("w", encoding="utf-8") as dst:
        for line in src:
            if max_samples is not None and count >= max_samples:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            out = {
                "id": row["id"],
                "question": format_question(row["question"], qa_template=qa_template),
                "answer": row["final_answer"],
                "ground_truth": row["final_answer"],
            }
            if include_target_text:
                out["target_text"] = row["target_text"]
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1
    return {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "num_samples": count,
        "qa_template": qa_template,
        "include_target_text": include_target_text,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", default="/data1/luyifei/drla/data/stage_a/gsm8k_test.jsonl")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--qa-template", action="store_true", help="Format question as a direct QA prompt.")
    parser.add_argument("--include-target-text", action="store_true", help="Also include concise solution target text.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = convert_gsm8k_for_cola(
        Path(args.input_jsonl),
        Path(args.output_jsonl),
        max_samples=args.max_samples,
        qa_template=args.qa_template,
        include_target_text=args.include_target_text,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
