"""Run D7 V4 Dream layer-conditioned receiver generation controls.

This local-only evaluator loads a trained layer-conditioned receiver and tests
whether D6 agent latent packets can condition Dream generation through native
hidden-layer cross-attention adapters. It never decodes agent packets into text
and never inserts Agent A/B text into the solver prompt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.evaluation.p2_phase_c_scorers import score_qa_answer  # noqa: E402
from drla.scripts.p3_run_dream_latent_prefix_eval import aggregate, sample_tokens  # noqa: E402
from drla.scripts.p3_train_dream_layer_conditioned_receiver import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_TRAIN_OUTPUT_DIR,
    DreamLayerConditionedReceiver,
    LayerReceiverConfig,
    load_row_packets,
    split_rows,
)
from drla.scripts.p3_train_dream_soft_prefix_adapter import (  # noqa: E402
    DEFAULT_MANIFEST_JSON,
    DEFAULT_MODEL_PATH,
    DEFAULT_ONLINE_INPUTS_JSONL,
    DEFAULT_PACKET_DIR,
    load_training_rows,
    resolve_mask_token_id,
)
from drla.scripts.run_p2_phase_c_text_agents import (  # noqa: E402
    append_jsonl,
    extract_final_answer,
    make_solver_messages,
    read_jsonl,
)


DEFAULT_CHECKPOINT = f"{DEFAULT_TRAIN_OUTPUT_DIR}/best_checkpoint.pt"
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/"
    "dream_layer_receiver_eval_textmas_matched200_20260607"
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
    parser.add_argument("--split", choices=["all", "train", "valid", "test"], default="all")
    parser.add_argument("--exclude-sample-ids", default="")
    parser.add_argument("--exclude-sample-ids-file", default="")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--dream-steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--alg", default="entropy")
    parser.add_argument("--alg-temp", type=float, default=0.0)
    parser.add_argument("--prediction-extraction-mode", choices=["default", "first_segment"], default="first_segment")
    parser.add_argument(
        "--conditions",
        default="no_message,layer_receiver_matched,layer_receiver_shuffled_row,layer_receiver_agent_swap,layer_receiver_zero",
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
    all_online_rows = [
        row
        for row in read_jsonl(Path(args.online_inputs_jsonl))
        if row.get("condition") == "textmas_matched"
    ]

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_config = LayerReceiverConfig(**checkpoint["config"])
    config = replace(
        checkpoint_config,
        manifest_json=args.manifest_json,
        online_inputs_jsonl=args.online_inputs_jsonl,
        packet_dir=args.packet_dir,
        model_path=args.model_path,
    )
    rows, _ = load_training_rows(config)
    rows_by_id = {str(row["row_id"]): row for row in rows}
    selected_online = select_online_rows(
        all_online_rows=all_online_rows,
        training_rows=rows,
        config=config,
        split=args.split,
        row_offset=args.row_offset,
        max_rows=args.max_rows,
        exclude_sample_ids=load_excluded_sample_ids(args),
    )
    if not selected_online:
        raise ValueError("no textmas_matched rows selected")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required when --device starts with cuda")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    dream = AutoModel.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True).to(device).eval()
    receiver = DreamLayerConditionedReceiver(config).to(device).eval()
    receiver.load_state_dict(checkpoint["model_state"])

    generations = []
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    started = time.time()
    for row_index, online_row in enumerate(selected_online, start=1):
        row_id = str(online_row.get("row_id", ""))
        sample = samples.get(str(online_row.get("sample_id", "")))
        train_row = rows_by_id.get(row_id)
        if sample is None or train_row is None:
            raise ValueError(f"missing sample or packet row for row_id={row_id}")
        for condition in conditions:
            packets = build_packets(condition, train_row, selected_online, rows_by_id, config, device)
            messages = make_solver_messages(online_row.get("online_input_fields", {}), upstream_messages=[])
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
                output_ids = conditioned_diffusion_generate(
                    dream=dream,
                    receiver=receiver,
                    input_ids=input_ids,
                    packets=packets,
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
                "row_id": row_id,
                "sample_id": online_row.get("sample_id", ""),
                "condition": condition,
                "status": "ok",
                "raw_final_output": text,
                "prediction": prediction,
                "score": score,
                "primary_score": score["primary_score"],
                "token_f1": score["token_f1"],
                "exact_match": score["exact_match"],
                "input_tokens": int(input_ids.shape[-1]),
                "prefix_tokens": 0,
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
        "checkpoint_data_config": {
            "manifest_json": checkpoint_config.manifest_json,
            "online_inputs_jsonl": checkpoint_config.online_inputs_jsonl,
            "packet_dir": checkpoint_config.packet_dir,
            "model_path": checkpoint_config.model_path,
        },
        "runtime_data_config": {
            "manifest_json": config.manifest_json,
            "online_inputs_jsonl": config.online_inputs_jsonl,
            "packet_dir": config.packet_dir,
            "model_path": config.model_path,
        },
        "output_dir": str(output_dir),
        "manifest_json": args.manifest_json,
        "online_inputs_jsonl": args.online_inputs_jsonl,
        "packet_dir": args.packet_dir,
        "model_path": args.model_path,
        "num_rows": len(selected_online),
        "num_generations": len(generations),
        "selection": {
            "split": args.split,
            "row_offset": args.row_offset,
            "max_rows": args.max_rows,
            "num_excluded_sample_ids": len(load_excluded_sample_ids(args)),
        },
        "metrics": metrics,
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "metrics_jsonl": str(output_dir / "metrics.jsonl"),
            "generations_jsonl": str(generations_path),
        },
        "elapsed_seconds": round(time.time() - started, 3),
        "execution_boundary": [
            "local-only P3 D7 V4 layer-conditioned receiver generation evaluation",
            "no optimizer, backward, or weight update",
            "no SwanLab run",
            "no agent decoded text messages inserted into solver prompt",
            "gold/scorer used only for offline evaluation",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def load_excluded_sample_ids(args: argparse.Namespace) -> set[str]:
    ids = {item.strip() for item in str(args.exclude_sample_ids).split(",") if item.strip()}
    if args.exclude_sample_ids_file:
        for line in Path(args.exclude_sample_ids_file).read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if item:
                ids.add(item)
    return ids


def select_online_rows(
    *,
    all_online_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    config: LayerReceiverConfig,
    split: str,
    row_offset: int,
    max_rows: int,
    exclude_sample_ids: set[str],
) -> list[dict[str, Any]]:
    row_ids: set[str] | None = None
    if split != "all":
        split_map = split_rows(training_rows, config.seed, config.train_ratio, config.valid_ratio)
        row_ids = {str(row["row_id"]) for row in split_map[split]}
    selected = []
    for row in all_online_rows:
        if row_ids is not None and str(row.get("row_id", "")) not in row_ids:
            continue
        if str(row.get("sample_id", "")) in exclude_sample_ids:
            continue
        selected.append(row)
    selected = selected[row_offset:]
    if max_rows:
        selected = selected[:max_rows]
    return selected


def conditioned_diffusion_generate(
    *,
    dream: Any,
    receiver: DreamLayerConditionedReceiver,
    input_ids: torch.Tensor,
    packets: torch.Tensor | None,
    mask_token_id: int,
    max_new_tokens: int,
    steps: int,
    temperature: float,
    top_p: float,
    alg: str,
    alg_temp: float,
) -> torch.Tensor:
    device = input_ids.device
    max_length = input_ids.shape[1] + max_new_tokens
    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
    timesteps = torch.linspace(1, getattr(dream.generation_config, "eps", 1e-3), steps + 1, device=device)
    for i in range(steps):
        mask_index = x == mask_token_id
        if packets is None:
            logits = dream(x).logits
        else:
            logits = receiver.forward_logits(dream, x, packets, condition_start=input_ids.shape[1])
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        mask_logits = logits[mask_index]
        if mask_logits.numel() == 0:
            break
        t = timesteps[i]
        s = timesteps[i + 1]
        if alg != "entropy":
            raise ValueError("This evaluator currently implements alg=entropy only")
        confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, neg_entropy=True)
        num_mask_token = mask_index.sum() / mask_index.shape[0]
        number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
        if number_transfer_tokens > 0:
            full_confidence = torch.full_like(x, -torch.inf, device=device, dtype=logits.dtype)
            full_confidence[mask_index] = confidence
            if alg_temp == 0:
                _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
            else:
                probs = F.softmax(full_confidence / alg_temp, dim=-1)
                transfer_index = torch.multinomial(probs, num_samples=number_transfer_tokens)
            x_new = torch.zeros_like(x, device=device, dtype=torch.long) + mask_token_id
            x_new[mask_index] = x0.clone()
            row_indices = torch.arange(x.size(0), device=device).unsqueeze(1).expand_as(transfer_index)
            x[row_indices, transfer_index] = x_new[row_indices, transfer_index]
    return x


def build_packets(
    condition: str,
    row: dict[str, Any],
    selected_online_rows: list[dict[str, Any]],
    rows_by_id: dict[str, dict[str, Any]],
    config: LayerReceiverConfig,
    device: torch.device,
) -> torch.Tensor | None:
    if condition == "no_message":
        return None
    packet_row = row
    if condition == "layer_receiver_shuffled_row":
        online_ids = [str(item["row_id"]) for item in selected_online_rows]
        index = online_ids.index(str(row["row_id"]))
        packet_row = rows_by_id[online_ids[(index + 1) % len(online_ids)]]
    packets = load_row_packets(packet_row, config).unsqueeze(0).to(device)
    if condition == "layer_receiver_agent_swap":
        packets = packets.flip(dims=[1])
    if condition == "layer_receiver_zero":
        packets = torch.zeros_like(packets)
    if condition not in {
        "layer_receiver_matched",
        "layer_receiver_shuffled_row",
        "layer_receiver_agent_swap",
        "layer_receiver_zero",
    }:
        raise ValueError(f"unknown condition: {condition}")
    return packets


if __name__ == "__main__":
    main()
