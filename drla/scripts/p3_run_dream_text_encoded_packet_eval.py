"""Evaluate Dream receiver with text-encoded Agent messages as latent packets.

This local-only diagnostic tests the user's hypothesis that if Agent A/B text
messages are converted through Dream's own hidden space, the resulting latent
packets should be easier for the receiver to consume than raw Agent-side suffix
hidden states. The agent text is never inserted into the final solver prompt;
it is used only to build continuous packet tensors.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.evaluation.p2_phase_c_scorers import score_qa_answer  # noqa: E402
from drla.scripts.p3_collect_dream_step_traces import find_last_layer_module  # noqa: E402
from drla.scripts.p3_run_dream_latent_prefix_eval import aggregate  # noqa: E402
from drla.scripts.p3_run_dream_layer_receiver_eval import conditioned_diffusion_generate  # noqa: E402
from drla.scripts.p3_train_dream_latent_fuser import select_evenly_spaced  # noqa: E402
from drla.scripts.p3_train_dream_layer_conditioned_receiver import (  # noqa: E402
    DreamLayerConditionedReceiver,
    LayerReceiverConfig,
)
from drla.scripts.p3_train_dream_soft_prefix_adapter import (  # noqa: E402
    DEFAULT_MANIFEST_JSON,
    DEFAULT_MODEL_PATH,
    DEFAULT_ONLINE_INPUTS_JSONL,
    resolve_mask_token_id,
)
from drla.scripts.run_p2_phase_c_text_agents import (  # noqa: E402
    append_jsonl,
    extract_final_answer,
    make_solver_messages,
    read_jsonl,
)


DEFAULT_CHECKPOINT = (
    "/data1/luyifei/drla/outputs/p3_dream_layer_receivers/"
    "dream_layer_receiver_v7_v4init_zeroshuf_textmas_matched200_seed20260607_20260607/"
    "best_checkpoint.pt"
)
DEFAULT_TEXTMAS_GENERATIONS = (
    "/data1/luyifei/drla/outputs/p3_dream_textmas_runs/"
    "dream_textmas_gate_full200_merged_20260606/generations.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_text_encoded_packet_runs/"
    "dream_text_encoded_packet_eval_v7_20260617"
)


def main() -> None:
    summary = run_eval(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--text-packet-adapter-checkpoint", default="")
    parser.add_argument("--textmas-generations-jsonl", default=DEFAULT_TEXTMAS_GENERATIONS)
    parser.add_argument("--manifest-json", default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--online-inputs-jsonl", default=DEFAULT_ONLINE_INPUTS_JSONL)
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
        default="no_message,text_encoded_matched,text_encoded_shuffled_row,text_encoded_agent_swap,text_encoded_zero",
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
    selected_online = [
        row
        for row in read_jsonl(Path(args.online_inputs_jsonl))
        if row.get("condition") == "textmas_matched"
    ][args.row_offset :]
    if args.max_rows:
        selected_online = selected_online[: args.max_rows]
    if not selected_online:
        raise ValueError("no textmas_matched rows selected")
    textmas_by_row = load_textmas_messages(Path(args.textmas_generations_jsonl))

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = LayerReceiverConfig(**checkpoint["config"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required when --device starts with cuda")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    dream = AutoModel.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True).to(device).eval()
    receiver = DreamLayerConditionedReceiver(config).to(device).eval()
    receiver.load_state_dict(checkpoint["model_state"])
    text_packet_adapter = None
    if args.text_packet_adapter_checkpoint:
        text_packet_adapter = load_text_packet_adapter(Path(args.text_packet_adapter_checkpoint), device)
    _, last_layer = find_last_layer_module(dream)
    if last_layer is None:
        raise RuntimeError("Could not find Dream last layer for text packet encoding")

    packet_cache: dict[str, torch.Tensor] = {}
    generations = []
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    started = time.time()
    for row_index, online_row in enumerate(selected_online, start=1):
        row_id = str(online_row.get("row_id", ""))
        sample = samples.get(str(online_row.get("sample_id", "")))
        if sample is None:
            raise ValueError(f"missing sample for row_id={row_id}")
        for condition in conditions:
            packets = build_text_packets(
                condition,
                row_id,
                selected_online,
                textmas_by_row,
                packet_cache,
                dream,
                tokenizer,
                last_layer,
                config,
                device,
                text_packet_adapter,
            )
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
        "text_packet_adapter_checkpoint": args.text_packet_adapter_checkpoint,
        "textmas_generations_jsonl": args.textmas_generations_jsonl,
        "output_dir": str(output_dir),
        "num_rows": len(selected_online),
        "num_generations": len(generations),
        "metrics": metrics,
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "metrics_jsonl": str(output_dir / "metrics.jsonl"),
            "generations_jsonl": str(generations_path),
        },
        "elapsed_seconds": round(time.time() - started, 3),
        "execution_boundary": [
            "local-only P3 text-encoded latent packet diagnostic",
            "no optimizer, backward, or weight update",
            "no SwanLab run",
            "agent text messages are encoded into continuous packets only",
            "agent text messages are not inserted into the final solver prompt",
            "gold/scorer used only for offline evaluation",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def load_textmas_messages(path: Path) -> dict[str, list[dict[str, str]]]:
    rows = {}
    for row in read_jsonl(path):
        if row.get("condition") != "textmas_matched":
            continue
        messages = row.get("agent_messages", [])
        if not isinstance(messages, list) or not messages:
            continue
        rows[str(row.get("row_id", ""))] = messages
    if not rows:
        raise ValueError(f"no textmas_matched agent_messages in {path}")
    return rows


def build_text_packets(
    condition: str,
    row_id: str,
    selected_online_rows: list[dict[str, Any]],
    textmas_by_row: dict[str, list[dict[str, str]]],
    packet_cache: dict[str, torch.Tensor],
    dream: Any,
    tokenizer: Any,
    last_layer: nn.Module,
    config: LayerReceiverConfig,
    device: torch.device,
    text_packet_adapter: nn.Module | None = None,
) -> torch.Tensor | None:
    if condition == "no_message":
        return None
    use_text_packet_adapter = condition.startswith("text_adapter_")
    raw_condition = condition
    if use_text_packet_adapter:
        raw_condition = "text_encoded_" + condition.removeprefix("text_adapter_")
    packet_row_id = row_id
    if raw_condition == "text_encoded_shuffled_row":
        online_ids = [str(item["row_id"]) for item in selected_online_rows]
        index = online_ids.index(row_id)
        packet_row_id = online_ids[(index + 1) % len(online_ids)]
    packets = get_or_encode_text_packets(packet_row_id, textmas_by_row, packet_cache, dream, tokenizer, last_layer, config, device)
    if raw_condition == "text_encoded_agent_swap":
        packets = packets.flip(dims=[1])
    if raw_condition == "text_encoded_zero":
        packets = torch.zeros_like(packets)
    if raw_condition not in {
        "text_encoded_matched",
        "text_encoded_shuffled_row",
        "text_encoded_agent_swap",
        "text_encoded_zero",
    }:
        raise ValueError(f"unknown condition: {condition}")
    if use_text_packet_adapter:
        if text_packet_adapter is None:
            raise ValueError(f"{condition} requires --text-packet-adapter-checkpoint")
        with torch.no_grad():
            packets = text_packet_adapter(packets.to(device))
    return packets


def load_text_packet_adapter(path: Path, device: torch.device) -> nn.Module:
    from drla.scripts.p3_train_dream_text_packet_adapter import (  # noqa: PLC0415
        TextPacketAdapter,
        TextPacketAdapterConfig,
    )

    checkpoint = torch.load(path, map_location=device)
    config = TextPacketAdapterConfig(**checkpoint["config"])
    adapter = TextPacketAdapter(config).to(device).eval()
    adapter.load_state_dict(checkpoint["model_state"])
    return adapter


def get_or_encode_text_packets(
    row_id: str,
    textmas_by_row: dict[str, list[dict[str, str]]],
    packet_cache: dict[str, torch.Tensor],
    dream: Any,
    tokenizer: Any,
    last_layer: nn.Module,
    config: LayerReceiverConfig,
    device: torch.device,
) -> torch.Tensor:
    if row_id not in packet_cache:
        messages = textmas_by_row.get(row_id)
        if messages is None:
            raise ValueError(f"missing TextMAS agent messages for row_id={row_id}")
        packet_cache[row_id] = encode_agent_messages(messages, dream, tokenizer, last_layer, config, device)
    return packet_cache[row_id]


def encode_agent_messages(
    messages: list[dict[str, str]],
    dream: Any,
    tokenizer: Any,
    last_layer: nn.Module,
    config: LayerReceiverConfig,
    device: torch.device,
) -> torch.Tensor:
    by_agent = {str(item.get("agent_id", "")): str(item.get("message", "")) for item in messages}
    tensors = []
    for agent_id in ["agent_a", "agent_b"]:
        text = by_agent.get(agent_id, "").strip()
        if not text:
            tensors.append(torch.zeros(config.input_tokens_per_agent, config.hidden_size, dtype=torch.float32))
            continue
        hidden = encode_text_hidden(text, dream, tokenizer, last_layer, device)
        tensors.append(select_evenly_spaced(hidden, config.input_tokens_per_agent))
    return torch.stack(tensors, dim=0).unsqueeze(0).to(device)


def encode_text_hidden(text: str, dream: Any, tokenizer: Any, last_layer: nn.Module, device: torch.device) -> torch.Tensor:
    captured: list[torch.Tensor] = []

    def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            hidden = output[0] if isinstance(output, tuple) else output
        if torch.is_tensor(hidden) and hidden.ndim >= 3:
            captured.append(hidden[0].detach().float().cpu())

    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    if ids.numel() == 0:
        raise ValueError("cannot encode empty agent message")
    handle = last_layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            dream(ids)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("last-layer hook did not capture hidden states")
    return captured[-1]


if __name__ == "__main__":
    main()
