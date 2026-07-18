"""Verify the D5.5 migration payload and its retained post-D5.5 smoke fixture."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
V7_CHECKPOINT = REPO_ROOT / (
    "outputs/p3_dream_layer_receivers/"
    "dream_layer_receiver_v7_v4init_zeroshuf_textmas_matched200_seed20260607_20260607/"
    "best_checkpoint.pt"
)
D5_CHECKPOINT = REPO_ROOT / (
    "outputs/p3_dream_readiness_students/"
    "dream_step_readiness_student_v1_full200_with_hidden_seed20260606_20260606/"
    "best_checkpoint.pt"
)
BUNDLE_DIR = REPO_ROOT / "migration/post_d55_smoke"
FORBIDDEN_PACKET_FIELDS = {
    "answer_aliases",
    "correctness",
    "decoded_probe_text",
    "decoded_text",
    "final_prediction",
    "final_primary_score",
    "gold_answer",
    "primary_score",
    "scorer_output",
    "selected_answer_text",
    "step_prediction",
    "step_primary_score",
    "step_score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-gpu", action="store_true", help="also run one-row V7 generation")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs/p3_archive_smoke_runs/v7_one_row"),
    )
    return parser.parse_args()


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def verify_model_indexes() -> int:
    indexes = sorted((REPO_ROOT / "models").rglob("*.safetensors.index.json"))
    for index_path in indexes:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for shard_name in set(index.get("weight_map", {}).values()):
            require(index_path.parent / shard_name)
    return len(indexes)


def verify_checkpoint(path: Path) -> dict:
    require(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state" not in checkpoint or "config" not in checkpoint:
        raise ValueError(f"checkpoint is missing model_state/config: {path}")
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "step": checkpoint.get("global_step"),
        "num_state_tensors": len(checkpoint["model_state"]),
    }


def verify_packet_bundle() -> dict:
    packets = [
        json.loads(line)
        for line in (BUNDLE_DIR / "packets/packets.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(packets) != 2 or {row.get("agent_id") for row in packets} != {"agent_a", "agent_b"}:
        raise ValueError("smoke bundle must contain exactly one agent_a and one agent_b packet")
    for packet in packets:
        forbidden = FORBIDDEN_PACKET_FIELDS.intersection(packet)
        if forbidden:
            raise ValueError(f"forbidden online packet fields: {sorted(forbidden)}")
        tensor_path = REPO_ROOT / packet["hidden_ref"]
        require(tensor_path)
        tensor_payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
        tensor = tensor_payload if isinstance(tensor_payload, torch.Tensor) else tensor_payload.get("tensor")
        if tensor is None or list(tensor.shape) != packet["hidden_shape"]:
            raise ValueError(f"hidden tensor shape mismatch: {tensor_path}")
    return {"num_packets": len(packets), "packet_refs": "pass", "forbidden_fields": "pass"}


def run_full_gpu_smoke(output_dir: Path) -> dict:
    command = [
        sys.executable,
        str(REPO_ROOT / "drla/scripts/p3_run_dream_layer_receiver_eval.py"),
        "--checkpoint",
        str(V7_CHECKPOINT),
        "--manifest-json",
        str(BUNDLE_DIR / "manifest.json"),
        "--online-inputs-jsonl",
        str(BUNDLE_DIR / "online_inputs.jsonl"),
        "--packet-dir",
        str(BUNDLE_DIR / "packets"),
        "--model-path",
        str(REPO_ROOT / "models/Dream-v0-Instruct-7B"),
        "--output-dir",
        str(output_dir),
        "--device",
        "cuda:0",
        "--max-rows",
        "1",
        "--max-tokens",
        "32",
        "--dream-steps",
        "8",
        "--conditions",
        "no_message,layer_receiver_matched,layer_receiver_shuffled_row,layer_receiver_zero",
        "--overwrite",
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "pass" or summary.get("num_generations") != 4:
        raise ValueError("full GPU smoke did not produce four successful condition rows")
    return {"status": "pass", "num_generations": 4, "output_dir": str(output_dir)}


def main() -> None:
    args = parse_args()
    required_paths = [
        REPO_ROOT / "AGENT.md",
        REPO_ROOT / "docs/cola_archive/README.md",
        REPO_ROOT / "docs/current/CURRENT_EXPERIMENT_STATUS.md",
        REPO_ROOT / "models/Dream-v0-Instruct-7B/model.safetensors.index.json",
        REPO_ROOT / "outputs/p3_dream_readiness_frontiers",
        REPO_ROOT / "outputs/p3_dream_readiness_policy_eval",
        BUNDLE_DIR / "manifest.json",
        BUNDLE_DIR / "online_inputs.jsonl",
    ]
    for path in required_paths:
        require(path)

    report = {
        "status": "pass",
        "model_indexes_checked": verify_model_indexes(),
        "d5_checkpoint": verify_checkpoint(D5_CHECKPOINT),
        "v7_checkpoint": verify_checkpoint(V7_CHECKPOINT),
        "packet_bundle": verify_packet_bundle(),
    }
    if args.full_gpu:
        report["full_gpu_smoke"] = run_full_gpu_smoke(Path(args.output_dir))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
