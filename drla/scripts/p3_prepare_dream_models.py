"""Prepare local Dream checkpoints for P3 Dream-DLM LatentMAS.

This local-only utility downloads or verifies Dream model snapshots and writes
reproducible environment artifacts. It does not load model weights into GPU,
run generation, train, or create SwanLab runs.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "Dream-org/Dream-v0-Instruct-7B"
DEFAULT_LOCAL_DIR = "/data1/luyifei/drla/models/Dream-v0-Instruct-7B"
DEFAULT_OUTPUT_ROOT = "/data1/luyifei/drla/outputs/p3_dream_models"


def main() -> None:
    summary = prepare_model(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite-artifacts", action="store_true")
    parser.add_argument(
        "--required-files",
        default="config.json,tokenizer_config.json,generation_config.json,model.safetensors.index.json",
    )
    return parser.parse_args()


def prepare_model(args: argparse.Namespace) -> dict[str, Any]:
    created_at = int(time.time())
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.repo_id, created_at)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite_artifacts:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite-artifacts: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    env_before = collect_environment(args, local_dir)

    token = os.environ.get(args.token_env) or None
    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(local_dir),
        token=token,
        local_files_only=args.local_files_only,
        max_workers=args.max_workers,
    )
    local_dir = Path(snapshot_path).resolve()

    files = scan_files(local_dir)
    required_files = [item.strip() for item in args.required_files.split(",") if item.strip()]
    missing_required = [name for name in required_files if not (local_dir / name).exists()]
    model_shards = sorted(path.name for path in local_dir.glob("*.safetensors"))
    status = "pass" if not missing_required and model_shards else "fail"

    metrics = {
        "status_pass": int(status == "pass"),
        "num_files": files["num_files"],
        "total_size_bytes": files["total_size_bytes"],
        "num_safetensors": len(model_shards),
        "num_missing_required": len(missing_required),
    }
    summary = {
        "created_at": created_at,
        "status": status,
        "repo_id": args.repo_id,
        "revision": args.revision,
        "local_dir": str(local_dir),
        "local_files_only": args.local_files_only,
        "required_files": required_files,
        "missing_required_files": missing_required,
        "model_shards": model_shards,
        "files": files,
        "environment": env_before,
        "execution_boundary": [
            "local-only P3 Dream model preparation",
            "downloads or verifies HuggingFace snapshot files",
            "does not load weights into GPU",
            "does not run generation",
            "does not run optimizer or backward",
            "does not create SwanLab runs",
        ],
    }
    write_json(output_dir / "environment.json", env_before)
    write_json(output_dir / "summary.json", summary)
    append_jsonl(output_dir / "metrics.jsonl", metrics)
    summary["summary_json"] = str(output_dir / "summary.json")
    summary["metrics_jsonl"] = str(output_dir / "metrics.jsonl")
    summary["environment_json"] = str(output_dir / "environment.json")
    if status != "pass":
        raise RuntimeError(f"Dream checkpoint preparation failed: missing={missing_required}, shards={model_shards}")
    return summary


def default_output_dir(repo_id: str, created_at: int) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(created_at))
    safe_repo = repo_id.replace("/", "_").replace(":", "_")
    return Path(DEFAULT_OUTPUT_ROOT) / f"{safe_repo}_prepare_{stamp}"


def collect_environment(args: argparse.Namespace, local_dir: Path) -> dict[str, Any]:
    import huggingface_hub

    try:
        import torch
    except Exception as exc:  # pragma: no cover - diagnostic only
        torch_info: dict[str, Any] = {"import_error": repr(exc)}
    else:
        torch_info = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            torch_info["devices"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_properties(index).name,
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ]

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "huggingface_hub_version": huggingface_hub.__version__,
        "torch": torch_info,
        "hf_xet_high_performance": os.environ.get("HF_XET_HIGH_PERFORMANCE", ""),
        "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
        "hf_home": os.environ.get("HF_HOME", ""),
        "token_env": args.token_env,
        "token_env_set": bool(os.environ.get(args.token_env)),
        "target_local_dir": str(local_dir),
        "target_local_dir_exists": local_dir.exists(),
    }


def scan_files(root: Path) -> dict[str, Any]:
    files = []
    total_size = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        total_size += size
        files.append({"path": str(path.relative_to(root)), "size_bytes": size})
    return {
        "num_files": len(files),
        "total_size_bytes": total_size,
        "total_size_gib": round(total_size / 1024**3, 3),
        "sample": files[:50],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
