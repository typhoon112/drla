"""Run P3 Dream trace collection shards across GPUs.

This local-only launcher executes ``p3_collect_dream_step_traces.py`` shards for
D3 trace collection. It starts inference-only jobs, does not train, and does
not create SwanLab runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "drla/scripts/p3_collect_dream_step_traces.py"


def main() -> None:
    summary = run_queue(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--online-inputs-jsonl", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--num-rows", type=int, required=True)
    parser.add_argument("--conditions", default="single_full_info,textmas_matched")
    parser.add_argument("--shard-size", type=int, default=20)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--end-shard", type=int, default=0, help="Exclusive. 0 means all shards.")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--dream-steps", type=int, default=64)
    parser.add_argument("--snapshot-stride", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--alg", default="entropy")
    parser.add_argument("--alg-temp", type=float, default=0.0)
    parser.add_argument("--prediction-extraction-mode", default="first_segment")
    parser.add_argument(
        "--hidden-capture-mode",
        default="summary",
        choices=["none", "summary", "suffix_tensor", "selected_suffix_tensor"],
    )
    parser.add_argument("--hidden-save-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--policy-eval-dir", default="")
    parser.add_argument("--readiness-checkpoint", default="")
    parser.add_argument("--decode-step-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--skip-complete", action="store_true")
    return parser.parse_args()


def run_queue(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    queue_dir = output_root / f"{args.run_name}_queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    total_shards = (args.num_rows + args.shard_size - 1) // args.shard_size
    end_shard = args.end_shard or total_shards
    if args.start_shard < 0 or end_shard > total_shards or args.start_shard >= end_shard:
        raise ValueError(f"Invalid shard range: start={args.start_shard}, end={end_shard}, total={total_shards}")
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must list at least one GPU id")

    shard_specs = [make_shard_spec(args, shard_idx, total_shards) for shard_idx in range(args.start_shard, end_shard)]
    if args.skip_complete:
        shard_specs = [spec for spec in shard_specs if not shard_complete(Path(spec["output_dir"]))]

    (queue_dir / "shard_manifest.json").write_text(
        json.dumps({"created_at": int(time.time()), "shards": shard_specs}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    pending = list(shard_specs)
    running: dict[subprocess.Popen[str], dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    started_at = time.time()

    while pending or running:
        for gpu in gpu_ids:
            if not pending:
                break
            if any(spec["gpu"] == gpu for spec in running.values()):
                continue
            spec = pending.pop(0)
            spec["gpu"] = gpu
            command = build_command(args, spec, gpu)
            log_path = queue_dir / f"shard{spec['shard_index']:03d}_gpu{gpu}.log"
            log_handle = log_path.open("w", encoding="utf-8")
            log_handle.write(" ".join(command) + "\n\n")
            log_handle.flush()
            proc = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            spec["pid"] = proc.pid
            spec["log_path"] = str(log_path)
            spec["started_at"] = int(time.time())
            spec["_log_handle"] = log_handle
            running[proc] = spec

        time.sleep(args.poll_seconds)
        still_running: dict[subprocess.Popen[str], dict[str, Any]] = {}
        for proc, spec in running.items():
            rc = proc.poll()
            if rc is None:
                still_running[proc] = spec
                continue
            spec["returncode"] = rc
            spec["finished_at"] = int(time.time())
            spec["_log_handle"].close()
            spec.pop("_log_handle", None)
            if rc == 0 and shard_complete(Path(spec["output_dir"])):
                completed.append(spec)
            else:
                failed.append(spec)
        running = still_running
        progress = {
            "elapsed_seconds": round(time.time() - started_at, 1),
            "pending": len(pending),
            "running": len(running),
            "completed": len(completed),
            "failed": len(failed),
        }
        print(json.dumps(progress, ensure_ascii=False, sort_keys=True), flush=True)
        write_progress(queue_dir, progress, completed, failed, list(running.values()))

    status = "pass" if not failed else "fail"
    summary = {
        "created_at": int(started_at),
        "finished_at": int(time.time()),
        "status": status,
        "run_name": args.run_name,
        "queue_dir": str(queue_dir),
        "num_requested_shards": len(shard_specs),
        "num_completed": len(completed),
        "num_failed": len(failed),
        "completed_shards": strip_runtime_fields(completed),
        "failed_shards": strip_runtime_fields(failed),
        "execution_boundary": [
            "local-only P3 Dream trace shard queue",
            "launches inference/trace shards only",
            "no optimizer or backward",
            "no SwanLab run",
        ],
    }
    (queue_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if failed:
        raise RuntimeError(f"{len(failed)} shards failed; see {queue_dir}")
    return summary


def make_shard_spec(args: argparse.Namespace, shard_idx: int, total_shards: int) -> dict[str, Any]:
    row_offset = shard_idx * args.shard_size
    max_rows = min(args.shard_size, args.num_rows - row_offset)
    output_dir = Path(args.output_root) / (
        f"{args.run_name}_shard{shard_idx:03d}_rows{row_offset:04d}_{row_offset + max_rows - 1:04d}"
    )
    return {
        "shard_index": shard_idx,
        "total_shards": total_shards,
        "row_offset": row_offset,
        "max_rows": max_rows,
        "output_dir": str(output_dir),
        "gpu": "",
    }


def build_command(args: argparse.Namespace, spec: dict[str, Any], gpu: str) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "--manifest-json",
        args.manifest_json,
        "--online-inputs-jsonl",
        args.online_inputs_jsonl,
        "--model-path",
        args.model_path,
        "--device",
        f"cuda:{gpu}",
        "--dtype",
        args.dtype,
        "--conditions",
        args.conditions,
        "--row-offset",
        str(spec["row_offset"]),
        "--max-rows",
        str(spec["max_rows"]),
        "--max-samples",
        "0",
        "--max-tokens",
        str(args.max_tokens),
        "--max-context-tokens",
        str(args.max_context_tokens),
        "--dream-steps",
        str(args.dream_steps),
        "--snapshot-stride",
        str(args.snapshot_stride),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--alg",
        args.alg,
        "--alg-temp",
        str(args.alg_temp),
        "--prediction-extraction-mode",
        args.prediction_extraction_mode,
        "--hidden-capture-mode",
        args.hidden_capture_mode,
        "--hidden-save-dtype",
        args.hidden_save_dtype,
        "--output-dir",
        spec["output_dir"],
    ]
    if args.policy_eval_dir:
        command.extend(["--policy-eval-dir", args.policy_eval_dir])
    if args.readiness_checkpoint:
        command.extend(["--readiness-checkpoint", args.readiness_checkpoint])
    command.append("--decode-step-text" if args.decode_step_text else "--no-decode-step-text")
    return command


def shard_complete(output_dir: Path) -> bool:
    summary_path = output_dir / "summary.json"
    generations_path = output_dir / "generations.jsonl"
    traces_path = output_dir / "traces.jsonl"
    if not summary_path.exists() or not generations_path.exists() or not traces_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if summary.get("status") != "pass" or int(summary.get("num_errors", 0)) != 0:
        return False
    expected_rows = int(summary.get("num_rows", 0))
    actual_rows = count_jsonl(generations_path)
    actual_traces = count_jsonl(traces_path)
    return expected_rows > 0 and actual_rows == expected_rows and actual_traces > 0


def count_jsonl(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def write_progress(
    queue_dir: Path,
    progress: dict[str, Any],
    completed: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    running: list[dict[str, Any]],
) -> None:
    payload = {
        **progress,
        "completed_indices": [spec["shard_index"] for spec in completed],
        "failed_indices": [spec["shard_index"] for spec in failed],
        "running_indices": [spec["shard_index"] for spec in running],
    }
    (queue_dir / "progress.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def strip_runtime_fields(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = []
    for spec in specs:
        item = dict(spec)
        item.pop("_log_handle", None)
        clean.append(item)
    return clean


if __name__ == "__main__":
    main()
