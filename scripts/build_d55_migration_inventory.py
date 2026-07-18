"""Create the exact file list for the D5.5-preserving migration archive."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
P3_PRE_D55_OUTPUT_DIRS = {
    "p3_dream_models",
    "p3_dream_protocol_audits",
    "p3_dream_readiness_frontiers",
    "p3_dream_readiness_policy_eval",
    "p3_dream_readiness_students",
    "p3_dream_textmas_aggregates",
    "p3_dream_textmas_runs",
}
POST_D55_TRACE_MARKERS = (
    "textmas_matched200_steps64_stride4_hidden_tensor",
    "selected_suffix_tensor",
    "heldout_trace_textmas",
    "train2000_trace",
    "validdiag50_trace",
)
HEAVY_POST_D55_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".npy", ".npz"}
V7_DIR = (
    "outputs/p3_dream_layer_receivers/"
    "dream_layer_receiver_v7_v4init_zeroshuf_textmas_matched200_seed20260607_20260607"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-list", required=True, help="NUL-delimited tar input, preferably outside repo")
    parser.add_argument(
        "--inventory-tsv",
        default=str(REPO_ROOT / "migration/ARCHIVE_INVENTORY.tsv"),
    )
    parser.add_argument(
        "--summary-json",
        default=str(REPO_ROOT / "migration/ARCHIVE_INVENTORY_SUMMARY.json"),
    )
    return parser.parse_args()


def classify(relative_path: Path) -> tuple[bool, str]:
    parts = relative_path.parts
    relative_text = relative_path.as_posix()
    if not parts:
        return False, "invalid"
    if parts[0] == "migration" and len(parts) > 1 and parts[1] == "post_d55_smoke":
        return True, "post_d55_smoke"
    if parts[0] == "models":
        return True, "model_weights"
    if parts[0] == "archive":
        return True, "historical_archive"
    if parts[0] == ".conda":
        return True, "conda_environment"
    if parts[0] == ".cache":
        return True, "local_cache"
    if parts[0] != "outputs":
        return True, "repository_code_docs_git"
    if len(parts) < 2 or not parts[1].startswith("p3_"):
        return True, "historical_and_mixed_outputs"
    if parts[1] in P3_PRE_D55_OUTPUT_DIRS:
        return True, "p3_pre_d55_reproducibility"
    if parts[1] == "p3_dream_traces" and len(parts) >= 3:
        run_name = parts[2]
        if not any(marker in run_name for marker in POST_D55_TRACE_MARKERS):
            return True, "p3_pre_d55_reproducibility"
    if relative_text == V7_DIR or relative_text.startswith(V7_DIR + "/"):
        return True, "post_d55_v7_runnable_checkpoint"
    if relative_path.suffix.lower() not in HEAVY_POST_D55_SUFFIXES:
        return True, "post_d55_logs_and_evidence"
    return False, "post_d55_heavy_omitted"


def main() -> None:
    args = parse_args()
    file_list_path = Path(args.file_list)
    inventory_path = Path(args.inventory_tsv)
    summary_path = Path(args.summary_json)
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    file_list_path.parent.mkdir(parents=True, exist_ok=True)

    included: list[tuple[str, int, str]] = []
    omitted_counts: Counter[str] = Counter()
    omitted_bytes: Counter[str] = Counter()
    category_bytes: defaultdict[str, int] = defaultdict(int)
    category_counts: Counter[str] = Counter()

    for root, dirnames, filenames in os.walk(REPO_ROOT, followlinks=False):
        root_path = Path(root)
        for name in list(dirnames):
            candidate = root_path / name
            if candidate.is_symlink():
                filenames.append(name)
                dirnames.remove(name)
        for name in filenames:
            path = root_path / name
            relative = path.relative_to(REPO_ROOT)
            include, category = classify(relative)
            size = path.lstat().st_size
            if include:
                included.append((relative.as_posix(), size, category))
                category_bytes[category] += size
                category_counts[category] += 1
            else:
                omitted_counts[category] += 1
                omitted_bytes[category] += size

    generated_relatives = {
        inventory_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        summary_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
    }
    included = [row for row in included if row[0] not in generated_relatives]
    included.sort(key=lambda row: row[0])

    with inventory_path.open("w", encoding="utf-8") as handle:
        handle.write("path\tsize_bytes\tretention_class\n")
        for path, size, category in included:
            handle.write(f"{path}\t{size}\t{category}\n")

    inventory_entry = (
        inventory_path.relative_to(REPO_ROOT).as_posix(),
        inventory_path.stat().st_size,
        "archive_metadata",
    )
    included.append(inventory_entry)
    category_bytes["archive_metadata"] += inventory_entry[1]
    category_counts["archive_metadata"] += 1

    summary = {
        "status": "pass",
        "policy_boundary": "preserve all historical/CoLA/P0-P2 and P3 through D5.5; compact P3 after D5.5",
        "repo_root_at_build_time": str(REPO_ROOT),
        "included_file_count": len(included) + 1,
        "included_apparent_bytes_excluding_this_summary": sum(size for _, size, _ in included),
        "included_by_class": {
            category: {"file_count": category_counts[category], "apparent_bytes": category_bytes[category]}
            for category in sorted(category_counts)
        },
        "omitted_by_class": {
            category: {"file_count": omitted_counts[category], "apparent_bytes": omitted_bytes[category]}
            for category in sorted(omitted_counts)
        },
        "heavy_post_d55_suffixes": sorted(HEAVY_POST_D55_SUFFIXES),
        "retained_v7_directory": V7_DIR,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_entry = (
        summary_path.relative_to(REPO_ROOT).as_posix(),
        summary_path.stat().st_size,
        "archive_metadata",
    )
    included.append(summary_entry)

    with file_list_path.open("wb") as handle:
        for path, _, _ in included:
            handle.write(path.encode("utf-8") + b"\0")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
