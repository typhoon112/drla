"""Audit the D7 Dream receiver interface distribution.

This local-only diagnostic compares the hidden states produced when Agent B
receives decoded TextMAS messages through its normal tokenizer/embedding path
against the D6 latent packets and the V7 receiver memory/injection states. It
does not generate answers, train a model, call SwanLab, or score with gold.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.scripts.p3_train_dream_layer_conditioned_receiver import (  # noqa: E402
    DreamLayerConditionedReceiver,
    LayerReceiverConfig,
    load_row_packets,
)
from drla.scripts.p3_train_dream_soft_prefix_adapter import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    load_training_rows,
    resolve_mask_token_id,
)
from drla.scripts.run_p2_phase_c_text_agents import append_jsonl, make_solver_messages, read_jsonl  # noqa: E402


DEFAULT_CHECKPOINT = (
    "/data1/luyifei/drla/outputs/p3_dream_layer_receivers/"
    "dream_layer_receiver_v7_v4init_zeroshuf_textmas_matched200_seed20260607_20260607/best_checkpoint.pt"
)
DEFAULT_TRACE_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_traces/"
    "musique_calibration_trace_textmas_matched200_steps64_stride4_hidden_tensor_merged_20260606"
)
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_receiver_interface_audits/"
    "dream_receiver_interface_audit_v7_best20_20260617"
)


def main() -> None:
    summary = run_audit(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    parser.add_argument("--manifest-json", default=None)
    parser.add_argument("--online-inputs-jsonl", default=None)
    parser.add_argument("--packet-dir", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument("--receiver-mask-tokens", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    rows_path.write_text("", encoding="utf-8")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_config = LayerReceiverConfig(**checkpoint["config"])
    config = replace(
        checkpoint_config,
        manifest_json=args.manifest_json or checkpoint_config.manifest_json,
        online_inputs_jsonl=args.online_inputs_jsonl or checkpoint_config.online_inputs_jsonl,
        packet_dir=args.packet_dir or checkpoint_config.packet_dir,
        model_path=args.model_path or checkpoint_config.model_path,
    )
    rows, row_metadata = load_training_rows(config)
    selected_rows = rows[args.row_offset :]
    if args.max_rows:
        selected_rows = selected_rows[: args.max_rows]
    if not selected_rows:
        raise ValueError("no rows selected")

    trace_generations = {
        str(row.get("row_id", "")): row for row in read_jsonl(Path(args.trace_dir) / "generations.jsonl")
    }
    missing_trace_rows = [
        {"row_id": row["row_id"], "reason": "missing_trace_generation"}
        for row in selected_rows
        if row["row_id"] not in trace_generations
    ]
    if missing_trace_rows:
        raise ValueError(f"missing TextMAS trace generations: {missing_trace_rows[:5]}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required when --device starts with cuda")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    dream = AutoModel.from_pretrained(config.model_path, torch_dtype=dtype, trust_remote_code=True).to(device).eval()
    receiver = DreamLayerConditionedReceiver(config).to(device).eval()
    receiver.load_state_dict(checkpoint["model_state"])
    mask_token_id = resolve_mask_token_id(dream, tokenizer)

    started = time.time()
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(selected_rows, start=1):
        trace_row = trace_generations[row["row_id"]]
        agent_messages = trace_row.get("agent_messages", [])
        no_message_ids = encode_solver_prompt(tokenizer, row["online_input_fields"], [], device)
        textmas_ids = encode_solver_prompt(tokenizer, row["online_input_fields"], agent_messages, device)
        packets = load_row_packets(row, config).unsqueeze(0).to(device)
        with torch.no_grad():
            no_hidden = last_hidden(dream, no_message_ids)
            text_hidden = last_hidden(dream, textmas_ids)
            memory = receiver.memory_encoder(packets)
            receiver_stats = inspect_receiver_injection(
                dream=dream,
                receiver=receiver,
                input_ids=no_message_ids,
                packets=packets,
                memory=memory,
                mask_token_id=mask_token_id,
                mask_tokens=args.receiver_mask_tokens,
            )
        packet_flat = packets[0].reshape(-1, packets.shape[-1])
        record = {
            "row_index": row_index,
            "row_id": row["row_id"],
            "sample_id": row["sample_id"],
            "status": "ok",
            "agent_message_count": len(agent_messages),
            "agent_message_chars": sum(len(str(item.get("message", ""))) for item in agent_messages),
            "no_message_tokens": int(no_message_ids.shape[-1]),
            "textmas_tokens": int(textmas_ids.shape[-1]),
            "textmas_extra_tokens": int(textmas_ids.shape[-1] - no_message_ids.shape[-1]),
            "packet_shape": list(packets.shape),
            "metrics": {
                "packet_all": tensor_stats(packet_flat),
                "packet_agent_a": tensor_stats(packets[0, 0]),
                "packet_agent_b": tensor_stats(packets[0, 1]),
                "receiver_memory": tensor_stats(memory[0]),
                "no_message_prompt_hidden": tensor_stats(no_hidden),
                "textmas_prompt_hidden": tensor_stats(text_hidden),
                "textmas_last128_hidden": tensor_stats(text_hidden[-128:]),
                "packet_mean_cos_to_no_message_prompt": mean_cosine(packet_flat, no_hidden),
                "packet_mean_cos_to_textmas_prompt": mean_cosine(packet_flat, text_hidden),
                "packet_mean_cos_to_textmas_last128": mean_cosine(packet_flat, text_hidden[-128:]),
                "textmas_prompt_mean_cos_to_no_message_prompt": mean_cosine(text_hidden, no_hidden),
                "receiver": receiver_stats,
            },
        }
        append_jsonl(rows_path, record)
        records.append(record)

    flat_metrics = flatten_metrics(records)
    summary = {
        "created_at": int(time.time()),
        "status": "pass",
        "checkpoint": args.checkpoint,
        "trace_dir": args.trace_dir,
        "model_path": config.model_path,
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
        "num_rows": len(records),
        "row_metadata": row_metadata,
        "receiver_config": asdict(config),
        "metrics": flat_metrics,
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "metrics_jsonl": str(output_dir / "metrics.jsonl"),
            "rows_jsonl": str(rows_path),
        },
        "elapsed_seconds": round(time.time() - started, 3),
        "execution_boundary": [
            "local-only P3 D7 interface/distribution audit",
            "no optimizer, backward, generation, or weight update",
            "no SwanLab run",
            "decoded TextMAS Agent messages used only to inspect AgentB text-interface hidden states",
            "no gold/scorer fields used",
        ],
    }
    (output_dir / "metrics.jsonl").write_text(
        json.dumps(flat_metrics, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def encode_solver_prompt(tokenizer: Any, fields: dict[str, Any], upstream_messages: list[dict[str, str]], device: torch.device) -> torch.Tensor:
    return tokenizer.apply_chat_template(
        make_solver_messages(fields, upstream_messages=upstream_messages),
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).input_ids.to(device)


def last_hidden(dream: Any, input_ids: torch.Tensor) -> torch.Tensor:
    outputs = dream(input_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[-1][0]
    return hidden.detach().to(torch.float32)


def inspect_receiver_injection(
    *,
    dream: Any,
    receiver: DreamLayerConditionedReceiver,
    input_ids: torch.Tensor,
    packets: torch.Tensor,
    memory: torch.Tensor,
    mask_token_id: int,
    mask_tokens: int,
) -> dict[str, Any]:
    device = input_ids.device
    prompt_len = int(input_ids.shape[-1])
    x = F.pad(input_ids, (0, mask_tokens), value=mask_token_id)
    hidden_states = dream.get_input_embeddings()(x)
    seq_len = hidden_states.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    position_embeddings = dream.model.rotary_emb(hidden_states, position_ids)
    condition_mask = torch.zeros(hidden_states.shape[:2], device=device, dtype=hidden_states.dtype)
    condition_mask[:, prompt_len:] = 1.0
    condition_mask = condition_mask.unsqueeze(-1)
    layer_metrics: dict[str, Any] = {}
    for layer_idx, decoder_layer in enumerate(dream.model.layers):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=position_embeddings,
        )[0]
        layer_key = str(layer_idx)
        if layer_key in receiver.conditioners:
            conditioner = receiver.conditioners[layer_key]
            hidden_float = hidden_states.to(torch.float32)
            memory_float = memory.to(torch.float32)
            query = conditioner.query_proj(conditioner.query_norm(hidden_float))
            packet_memory = conditioner.packet_norm(memory_float)
            conditioned, _ = conditioner.cross_attn(query, packet_memory, packet_memory, need_weights=False)
            delta = conditioner.out(conditioned)
            gate = torch.sigmoid(conditioner.gate_logit).to(dtype=hidden_states.dtype)
            masked_hidden = hidden_states[:, prompt_len:].detach().to(torch.float32)
            masked_delta = (gate * delta[:, prompt_len:].to(dtype=hidden_states.dtype)).detach().to(torch.float32)
            hidden_norm = masked_hidden.norm(dim=-1).mean().item()
            delta_norm = masked_delta.norm(dim=-1).mean().item()
            layer_metrics[layer_key] = {
                "gate": float(gate.item()),
                "masked_hidden_norm_mean": hidden_norm,
                "gated_delta_norm_mean": delta_norm,
                "delta_to_hidden_norm_ratio": delta_norm / max(hidden_norm, 1e-8),
                "delta_stats": tensor_stats(masked_delta[0]),
                "masked_hidden_stats": tensor_stats(masked_hidden[0]),
            }
            hidden_states = hidden_states + gate * delta.to(dtype=hidden_states.dtype) * condition_mask
    hidden_states = dream.model.norm(hidden_states)
    return {
        "mask_tokens": float(mask_tokens),
        "selected_layers": layer_metrics,
        "final_masked_hidden": tensor_stats(hidden_states[0, prompt_len:]),
    }


def tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    if tensor.numel() == 0:
        return {}
    x = tensor.detach().to(torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    flat = x.reshape(-1, x.shape[-1])
    norms = flat.norm(dim=-1)
    return {
        "tokens": float(flat.shape[0]),
        "hidden": float(flat.shape[1]),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "abs_mean": float(flat.abs().mean().item()),
        "rms": float(torch.sqrt((flat * flat).mean()).item()),
        "token_norm_mean": float(norms.mean().item()),
        "token_norm_std": float(norms.std(unbiased=False).item()),
        "feature_std_mean": float(flat.std(dim=0, unbiased=False).mean().item()),
    }


def mean_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 or right.numel() == 0:
        return 0.0
    left_flat = left.detach().to(torch.float32).reshape(-1, left.shape[-1])
    right_flat = right.detach().to(torch.float32).reshape(-1, right.shape[-1])
    if left_flat.shape[-1] != right_flat.shape[-1]:
        return 0.0
    return float(F.cosine_similarity(left_flat.mean(dim=0), right_flat.mean(dim=0), dim=0).item())


def flatten_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for record in records:
        collect_numeric("", record["metrics"], buckets)
        for key in ["agent_message_count", "agent_message_chars", "no_message_tokens", "textmas_tokens", "textmas_extra_tokens"]:
            buckets[key].append(float(record[key]))
    return {key: sum(values) / len(values) for key, values in sorted(buckets.items()) if values}


def collect_numeric(prefix: str, value: Any, buckets: dict[str, list[float]]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            collect_numeric(next_prefix, item, buckets)
    elif isinstance(value, (int, float)):
        buckets[prefix].append(float(value))


if __name__ == "__main__":
    main()
