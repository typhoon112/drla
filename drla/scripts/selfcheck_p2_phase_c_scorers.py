"""Run local self-checks for P2 Phase C scorer helpers.

This script is local-only.  It does not read benchmark data, run models,
inspect held-out rows, train adapters, or create SwanLab runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.evaluation.p2_phase_c_scorers import (
    normalize_qa_text,
    qa_token_f1,
    score_qa_answer,
    score_structured_exact,
)


DEFAULT_OUTPUT_DIR = "/data1/luyifei/drla/outputs/p2_phase_c_scorer_selfcheck/selfcheck_20260601"


def main() -> None:
    summary = run_selfcheck(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_selfcheck(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    checks = [
        ("normalize_articles_punctuation", normalize_qa_text("The Blue, Sky!") == "blue sky"),
        ("qa_alias_match", score_qa_answer("the color blue", "blue", ["the color blue"]).primary_score == 1.0),
        ("qa_token_f1_partial", 0.0 < qa_token_f1("blue sky", "blue ocean") < 1.0),
        (
            "structured_exact_order_sensitive",
            score_structured_exact(["a", "b"], ["b", "a"]).primary_score == 0.0,
        ),
        (
            "structured_exact_order_insensitive",
            score_structured_exact(["a", "b"], ["b", "a"], sort_lists=True).primary_score == 1.0,
        ),
    ]
    failed = [name for name, passed in checks if not passed]
    metrics = {
        "num_checks": len(checks),
        "num_failed": len(failed),
    }
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "created_at": int(time.time()),
        "status": "pass" if not failed else "fail",
        "checks": [{"name": name, "passed": passed} for name, passed in checks],
        "failed_checks": failed,
        "metrics_jsonl": str(metrics_path),
        "execution_boundary": [
            "local-only scorer self-check",
            "no benchmark data",
            "no model generation",
            "no optimizer or backward",
            "no SwanLab run",
            "no held-out inspection",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_json"] = str(summary_path)
    if failed:
        raise AssertionError(f"P2 Phase C scorer self-check failed: {failed}")
    return summary


if __name__ == "__main__":
    main()
