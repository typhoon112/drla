"""Run D7 Dream latent-prefix receiver generation controls.

This local-only evaluator tests whether upstream latent packets can directly
condition a Dream solver generation path. It never decodes agent packets into
text and never inserts Agent A/B text messages. The solver prompt is the
no-message online input; packet tensors are prepended as continuous embeddings
inside a custom Dream diffusion loop.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.evaluation.p2_phase_c_scorers import score_qa_answer  # noqa: E402
from drla.scripts.run_p2_phase_c_text_agents import extract_final_answer, make_solver_messages, read_jsonl, write_jsonl  # noqa: E402


DEFAULT_MANIFEST_JSON = (
    "/data1/luyifei/drla/outputs/p2_phase_c_manifests/"
    "musique_calibration_manifest_200_seed20260601/manifest.json"
)
DEFAULT_ONLINE_INPUTS_JSONL = (
    "/data1/luyifei/drla/outputs/p2_phase_c_control_inputs/"
    "musique_calibration_controls_200_seed20260601_v1_strict_wrong/online_inputs.jsonl"
)
DEFAULT_PACKET_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_latent_packets/"
    "dream_textmas_matched200_agent_ab_suffix_tensor_packets_v1_20260606"
)
DEFAULT_MODEL_PATH = "/data1/luyifei/drla/models/Dream-v0-Instruct-7B"
DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_latent_prefix_runs/"
    "dream_latent_prefix_eval_textmas_matched200_20260606"
)


def main() -> None:
    summary = run_eval(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--online-inputs-jsonl", default=DEFAULT_ONLINE_INPUTS_JSONL)
    parser.add_argument("--packet-dir", default=DEFAULT_PACKET_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument("--prefix-tokens-per-agent", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--dream-steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--alg", default="entropy")
    parser.add_argument("--alg-temp", type=float, default=0.0)
    parser.add_argument("--prediction-extraction-mode", choices=["default", "first_segment"], default="first_segment")
    parser.add_argument("--conditions", default="no_message,latent_matched,latent_shuffled_row,latent_agent_swap,latent_zero")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
    samples = {str(sample["sample_id"]): sample for sample in manifest.get("samples", [])}
    all_rows = [
        row
        for row in read_jsonl(Path(args.online_inputs_jsonl))
        if row.get("condition") == "textmas_matched"
    ]
    selected_rows = all_rows[args.row_offset :]
    if args.max_rows:
        selected_rows = selected_rows[: args.max_rows]
    if not selected_rows:
        raise ValueError("no textmas_matched rows selected")
    packet_groups, packets = load_packets(Path(args.packet_dir))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required when --device starts with cuda")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True).to(device).eval()

    generations = []
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    started = time.time()
    for row_index, row in enumerate(selected_rows, start=1):
        sample = samples.get(str(row.get("sample_id", "")))
        if sample is None:
            raise ValueError(f"unknown sample_id: {row.get('sample_id')}")
        for condition in conditions:
            prefix = build_prefix(condition, row, selected_rows, packet_groups, packets, args, device, dtype)
            messages = make_solver_messages(row.get("online_input_fields", {}), upstream_messages=[])
            input_ids = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            ).input_ids.to(device)
            torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
            call_start = time.time()
            with torch.no_grad():
                output_ids = latent_prefix_diffusion_generate(
                    model=model,
                    input_ids=input_ids,
                    prefix_embeds=prefix,
                    mask_token_id=resolve_mask_token_id(model, tokenizer),
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
            generations.append(
                {
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
            )
    write_jsonl(output_dir / "generations.jsonl", generations)
    metrics = aggregate(generations)
    (output_dir / "metrics.jsonl").write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "created_at": int(time.time()),
        "status": "pass",
        "output_dir": str(output_dir),
        "manifest_json": args.manifest_json,
        "online_inputs_jsonl": args.online_inputs_jsonl,
        "packet_dir": args.packet_dir,
        "model_path": args.model_path,
        "num_rows": len(selected_rows),
        "num_generations": len(generations),
        "metrics": metrics,
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "metrics_jsonl": str(output_dir / "metrics.jsonl"),
            "generations_jsonl": str(output_dir / "generations.jsonl"),
        },
        "elapsed_seconds": round(time.time() - started, 3),
        "execution_boundary": [
            "local-only P3 D7 latent-prefix receiver generation evaluation",
            "no optimizer, backward, or weight update",
            "no SwanLab run",
            "no agent decoded text messages inserted into solver prompt",
            "gold/scorer used only for offline evaluation",
        ],
        "scope_note": (
            "This evaluator tests raw D6 suffix tensors as continuous Dream input "
            "prefixes. These tensors are last-layer hidden states, so failure may "
            "indicate representation-space mismatch rather than absence of latent signal."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def latent_prefix_diffusion_generate(
    *,
    model,
    input_ids: torch.Tensor,
    prefix_embeds: torch.Tensor | None,
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
    timesteps = torch.linspace(1, getattr(model.generation_config, "eps", 1e-3), steps + 1, device=device)
    token_embed = model.get_input_embeddings()
    prefix_len = int(prefix_embeds.shape[1]) if prefix_embeds is not None else 0
    for i in range(steps):
        mask_index = x == mask_token_id
        embeds = token_embed(x)
        combined = torch.cat([prefix_embeds, embeds], dim=1) if prefix_embeds is not None else embeds
        logits = model(inputs_embeds=combined).logits
        shifted = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        token_logits = shifted[:, prefix_len : prefix_len + x.shape[1], :]
        mask_logits = token_logits[mask_index]
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
            full_confidence = torch.full_like(x, -torch.inf, device=device, dtype=token_logits.dtype)
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


def sample_tokens(logits: torch.Tensor, temperature: float, top_p: float, neg_entropy: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    try:
        x0 = torch.distributions.Categorical(probs=probs).sample() if temperature > 0 else probs.argmax(dim=-1)
        confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
    except Exception:
        confidence, x0 = probs.max(dim=-1)
    if neg_entropy:
        log_probs = torch.log(probs + 1e-10)
        confidence = torch.sum(probs * log_probs, dim=-1)
    return confidence, x0


def top_p_logits(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumulative_probs > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def build_prefix(
    condition: str,
    row: dict[str, Any],
    rows: list[dict[str, Any]],
    packet_groups: dict[str, dict[str, Any]],
    packets: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if condition == "no_message":
        return None
    group_row = row
    if condition == "latent_shuffled_row":
        index = rows.index(row)
        group_row = rows[(index + 1) % len(rows)]
    group = packet_groups[str(group_row["row_id"])]
    agent_ids = ["agent_a", "agent_b"]
    if condition == "latent_agent_swap":
        agent_ids = list(reversed(agent_ids))
    tensors = []
    for agent_id in agent_ids:
        packet_id = group["packet_ids_by_agent"][agent_id]
        packet = packets[packet_id]
        tensor = load_hidden_tensor(packet["hidden_ref"])
        tensor = select_evenly_spaced(tensor, args.prefix_tokens_per_agent)
        tensors.append(tensor)
    prefix = torch.cat(tensors, dim=0).unsqueeze(0).to(device=device, dtype=dtype)
    if condition == "latent_zero":
        prefix = torch.zeros_like(prefix)
    return prefix


def load_packets(packet_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    packets = {str(packet["packet_id"]): packet for packet in read_jsonl(packet_dir / "packets.jsonl")}
    groups = {str(group["row_id"]): group for group in read_jsonl(packet_dir / "packet_groups.jsonl")}
    return groups, packets


def load_hidden_tensor(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    tensor = obj["tensor"] if isinstance(obj, dict) and "tensor" in obj else obj
    if not torch.is_tensor(tensor):
        raise TypeError(f"not a tensor ref: {path}")
    return tensor.to(torch.float32)


def select_evenly_spaced(tensor: torch.Tensor, length: int) -> torch.Tensor:
    if tensor.shape[0] == length:
        return tensor
    indices = torch.linspace(0, tensor.shape[0] - 1, length).round().long()
    return tensor.index_select(0, indices)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    conditions = sorted({str(row["condition"]) for row in rows})
    return {
        condition: {
            "num_rows": sum(1 for row in rows if row["condition"] == condition),
            "primary_score_mean": mean([float(row["primary_score"]) for row in rows if row["condition"] == condition]),
            "token_f1_mean": mean([float(row["token_f1"]) for row in rows if row["condition"] == condition]),
            "exact_match_mean": mean([float(row["exact_match"]) for row in rows if row["condition"] == condition]),
            "elapsed_seconds_mean": mean([float(row["elapsed_seconds"]) for row in rows if row["condition"] == condition]),
            "prefix_tokens_mean": mean([float(row["prefix_tokens"]) for row in rows if row["condition"] == condition]),
        }
        for condition in conditions
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def resolve_mask_token_id(model: Any, tokenizer: Any) -> int:
    for value in [
        getattr(getattr(model, "generation_config", None), "mask_token_id", None),
        getattr(getattr(model, "config", None), "mask_token_id", None),
        getattr(tokenizer, "mask_token_id", None),
    ]:
        if value is not None:
            return int(value)
    raise ValueError("Cannot resolve Dream mask_token_id")


if __name__ == "__main__":
    main()
