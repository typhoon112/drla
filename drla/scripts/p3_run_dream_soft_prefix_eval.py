"""Run D7 Dream soft-prefix receiver generation controls.

This local-only evaluator loads a trained embedding-space soft-prefix adapter
and tests whether D6 agent latent packets can condition Dream receiver
generation. It never decodes agent packets into text and never inserts Agent
A/B text messages into the solver prompt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.evaluation.p2_phase_c_scorers import score_qa_answer  # noqa: E402
from drla.scripts.p3_run_dream_latent_prefix_eval import (  # noqa: E402
    aggregate,
    latent_prefix_diffusion_generate,
    resolve_mask_token_id,
)
from drla.scripts.p3_train_dream_latent_fuser import load_tensor, select_evenly_spaced  # noqa: E402
from drla.scripts.p3_train_dream_soft_prefix_adapter import (  # noqa: E402
    DEFAULT_MANIFEST_JSON,
    DEFAULT_MODEL_PATH,
    DEFAULT_ONLINE_INPUTS_JSONL,
    DEFAULT_PACKET_DIR,
    DreamSoftPrefixAdapter,
    SoftPrefixConfig,
)
from drla.scripts.run_p2_phase_c_text_agents import (  # noqa: E402
    append_jsonl,
    extract_final_answer,
    make_solver_messages,
    read_jsonl,
)


DEFAULT_CHECKPOINT = (
    "/data1/luyifei/drla/outputs/p3_dream_soft_prefix_adapters/"
    "dream_soft_prefix_adapter_v1_textmas_matched200_seed20260607/best_checkpoint.pt"
)
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_soft_prefix_runs/"
    "dream_soft_prefix_eval_textmas_matched200_20260607"
)


def main() -> None:
    summary = run_eval(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest-json", default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--online-inputs-jsonl", default=DEFAULT_ONLINE_INPUTS_JSONL)
    parser.add_argument("--packet-dir", default=DEFAULT_PACKET_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--dream-steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--alg", default="entropy")
    parser.add_argument("--alg-temp", type=float, default=0.0)
    parser.add_argument("--prediction-extraction-mode", choices=["default", "first_segment"], default="first_segment")
    parser.add_argument(
        "--conditions",
        default="no_message,soft_prefix_matched,soft_prefix_shuffled_row,soft_prefix_agent_swap,soft_prefix_zero",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    generations_path = output_dir / "generations.jsonl"
    generations_path.write_text("", encoding="utf-8")

    manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
    samples = {str(sample["sample_id"]): sample for sample in manifest.get("samples", [])}
    all_rows = [
        row
        for row in read_jsonl(Path(args.online_inputs_jsonl))
        if row.get("condition") == "textmas_matched"
    ]
    rows = all_rows[args.row_offset :]
    if args.max_rows:
        rows = rows[: args.max_rows]
    if not rows:
        raise ValueError("no textmas_matched rows selected")
    packet_groups, packets = load_packets(Path(args.packet_dir))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required when --device starts with cuda")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    dream = AutoModel.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True).to(device).eval()
    adapter = load_adapter(Path(args.checkpoint), dream, device).eval()

    generations = []
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    started = time.time()
    for row_index, row in enumerate(rows, start=1):
        sample = samples.get(str(row.get("sample_id", "")))
        if sample is None:
            raise ValueError(f"unknown sample_id: {row.get('sample_id')}")
        for condition in conditions:
            prefix = build_soft_prefix(condition, row, rows, packet_groups, packets, adapter, device, dtype)
            messages = make_solver_messages(row.get("online_input_fields", {}), upstream_messages=[])
            input_ids = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            ).input_ids.to(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            call_start = time.time()
            with torch.no_grad():
                output_ids = latent_prefix_diffusion_generate(
                    model=dream,
                    input_ids=input_ids,
                    prefix_embeds=prefix,
                    mask_token_id=resolve_mask_token_id(dream, tokenizer),
                    max_new_tokens=args.max_tokens,
                    steps=args.dream_steps,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    alg=args.alg,
                    alg_temp=args.alg_temp,
                )
            generated = output_ids[0, input_ids.shape[1] :].detach().cpu().tolist()
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            prediction = extract_final_answer(text, mode=args.prediction_extraction_mode)
            scoring = sample.get("scoring", {})
            score = score_qa_answer(prediction, scoring.get("gold_answer", ""), scoring.get("answer_aliases", []) or []).to_dict()
            record = {
                "row_index": row_index,
                "row_id": row.get("row_id", ""),
                "sample_id": row.get("sample_id", ""),
                "condition": condition,
                "status": "ok",
                "raw_final_output": text,
                "prediction": prediction,
                "score": score,
                "primary_score": score["primary_score"],
                "token_f1": score["token_f1"],
                "exact_match": score["exact_match"],
                "input_tokens": int(input_ids.shape[-1]),
                "prefix_tokens": int(prefix.shape[1]) if prefix is not None else 0,
                "max_tokens": args.max_tokens,
                "dream_steps": args.dream_steps,
                "elapsed_seconds": round(time.time() - call_start, 3),
                "peak_memory_gib": round((torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0) / 1024**3, 3),
            }
            append_jsonl(generations_path, record)
            generations.append(record)
    metrics = aggregate(generations)
    (output_dir / "metrics.jsonl").write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "created_at": int(time.time()),
        "status": "pass",
        "checkpoint": args.checkpoint,
        "output_dir": str(output_dir),
        "manifest_json": args.manifest_json,
        "online_inputs_jsonl": args.online_inputs_jsonl,
        "packet_dir": args.packet_dir,
        "model_path": args.model_path,
        "num_rows": len(rows),
        "num_generations": len(generations),
        "metrics": metrics,
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "metrics_jsonl": str(output_dir / "metrics.jsonl"),
            "generations_jsonl": str(generations_path),
        },
        "elapsed_seconds": round(time.time() - started, 3),
        "execution_boundary": [
            "local-only P3 D7 soft-prefix receiver generation evaluation",
            "no optimizer, backward, or weight update",
            "no SwanLab run",
            "no agent decoded text messages inserted into solver prompt",
            "gold/scorer used only for offline evaluation",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def load_adapter(checkpoint_path: Path, dream: Any, device: torch.device) -> DreamSoftPrefixAdapter:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = SoftPrefixConfig(**checkpoint["config"])
    adapter = DreamSoftPrefixAdapter(config, embed_size=int(dream.get_input_embeddings().embedding_dim)).to(device)
    adapter.load_state_dict(checkpoint["model_state"])
    return adapter


def build_soft_prefix(
    condition: str,
    row: dict[str, Any],
    rows: list[dict[str, Any]],
    packet_groups: dict[str, dict[str, Any]],
    packets: dict[str, dict[str, Any]],
    adapter: DreamSoftPrefixAdapter,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if condition == "no_message":
        return None
    group_row = row
    if condition == "soft_prefix_shuffled_row":
        index = rows.index(row)
        group_row = rows[(index + 1) % len(rows)]
    group = packet_groups[str(group_row["row_id"])]
    agent_ids = ["agent_a", "agent_b"]
    tensors = []
    for agent_id in agent_ids:
        packet_id = group["packet_ids_by_agent"][agent_id]
        packet = packets[packet_id]
        tensor = load_tensor(packet["hidden_ref"])
        tensor = select_evenly_spaced(tensor, adapter.input_tokens_per_agent)
        tensors.append(tensor)
    packet_tensor = torch.stack(tensors, dim=0).unsqueeze(0).to(device=device, dtype=torch.float32)
    if condition == "soft_prefix_agent_swap":
        packet_tensor = packet_tensor.flip(dims=[1])
    with torch.no_grad():
        prefix = adapter(packet_tensor).to(dtype=dtype)
    if condition == "soft_prefix_zero":
        prefix = torch.zeros_like(prefix)
    if condition not in {
        "soft_prefix_matched",
        "soft_prefix_shuffled_row",
        "soft_prefix_agent_swap",
        "soft_prefix_zero",
    }:
        raise ValueError(f"unknown condition: {condition}")
    return prefix


def load_packets(packet_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    packets = {str(packet["packet_id"]): packet for packet in read_jsonl(packet_dir / "packets.jsonl")}
    groups = {str(group["row_id"]): group for group in read_jsonl(packet_dir / "packet_groups.jsonl")}
    return groups, packets


if __name__ == "__main__":
    main()
