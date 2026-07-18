"""Probe Dream diffusion generation and online state visibility for P3.

This local-only probe loads a local Dream checkpoint, runs one or more short
prompts through ``diffusion_generate``, and records which state streams can be
observed without training: output history, token hook snapshots, logits summary,
and an optional last-layer hidden-state hook. It does not run optimizer/backward
or create SwanLab runs.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL_PATH = "/data1/luyifei/drla/models/Dream-v0-Instruct-7B"
DEFAULT_OUTPUT_ROOT = "/data1/luyifei/drla/outputs/p3_dream_models"


def main() -> None:
    summary = probe_generation(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--prompt", default="Answer briefly: What is the capital of France?")
    parser.add_argument("--messages-json", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--alg", default="entropy")
    parser.add_argument("--alg-temp", type=float, default=0.0)
    parser.add_argument("--history-sample-limit", type=int, default=8)
    parser.add_argument("--capture-hidden-summary", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def probe_generation(args: argparse.Namespace) -> dict[str, Any]:
    created_at = int(time.time())
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(created_at)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for Dream probe when --device starts with cuda")

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)
    env = collect_environment(args)
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = model.to(device).eval()

    hook_state: dict[str, Any] = {
        "token_snapshots": [],
        "logit_snapshots": [],
        "hidden_snapshots": [],
        "hidden_hook_module": "",
    }
    hidden_handle = None
    if args.capture_hidden_summary:
        module_name, module = find_last_layer_module(model)
        hook_state["hidden_hook_module"] = module_name
        if module is not None:
            hidden_handle = module.register_forward_hook(make_hidden_hook(hook_state))

    messages = load_messages(args)
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    )
    input_ids = inputs.input_ids.to(device=device)
    attention_mask = inputs.attention_mask.to(device=device)
    input_token_count = int(input_ids.shape[-1])
    mask_token_id = resolve_mask_token_id(model, tokenizer)
    if input_token_count + args.max_new_tokens > 2048:
        raise ValueError(f"Dream context would exceed 2048: prompt={input_token_count}, new={args.max_new_tokens}")

    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    with torch.no_grad():
        output = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            output_history=True,
            return_dict_in_generate=True,
            steps=args.steps,
            temperature=args.temperature,
            top_p=args.top_p,
            alg=args.alg,
            alg_temp=args.alg_temp,
            generation_tokens_hook_func=make_tokens_hook(hook_state, args.history_sample_limit, mask_token_id),
            generation_logits_hook_func=make_logits_hook(hook_state, args.history_sample_limit, mask_token_id),
        )
    if hidden_handle is not None:
        hidden_handle.remove()

    elapsed = time.time() - start
    sequences = getattr(output, "sequences", output)
    history = getattr(output, "history", None)
    generated_text = decode_generated(tokenizer, input_ids, sequences)
    history_summary = summarize_history(tokenizer, input_ids, history, args.history_sample_limit)
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

    generation = {
        "messages": messages,
        "input_token_count": input_token_count,
        "mask_token_id": mask_token_id,
        "max_new_tokens": args.max_new_tokens,
        "steps": args.steps,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "alg": args.alg,
        "alg_temp": args.alg_temp,
        "generated_text": generated_text,
        "sequence_shape": list(sequences.shape),
        "history_available": history is not None,
        "history_length": len(history) if history is not None else 0,
        "history_summary": history_summary,
        "token_hook_snapshots": hook_state["token_snapshots"],
        "logit_hook_snapshots": hook_state["logit_snapshots"],
        "hidden_hook_module": hook_state["hidden_hook_module"],
        "hidden_snapshots": hook_state["hidden_snapshots"][: args.history_sample_limit],
    }
    metrics = {
        "status_pass": 1,
        "input_token_count": input_token_count,
        "max_new_tokens": args.max_new_tokens,
        "steps": args.steps,
        "history_available": int(history is not None),
        "history_length": len(history) if history is not None else 0,
        "num_token_hook_snapshots": len(hook_state["token_snapshots"]),
        "num_logit_hook_snapshots": len(hook_state["logit_snapshots"]),
        "num_hidden_snapshots": len(hook_state["hidden_snapshots"]),
        "elapsed_seconds": round(elapsed, 3),
        "peak_memory_gib": round(peak_memory / 1024**3, 3),
    }
    summary = {
        "created_at": created_at,
        "status": "pass",
        "model_path": args.model_path,
        "environment": env,
        "generation": generation,
        "metrics": metrics,
        "online_state_visibility": {
            "output_history_tokens": history is not None,
            "generation_tokens_hook": len(hook_state["token_snapshots"]) > 0,
            "generation_logits_hook": len(hook_state["logit_snapshots"]) > 0,
            "hidden_hook": len(hook_state["hidden_snapshots"]) > 0,
            "decoded_text_is_offline_probe_only": True,
            "scorer_or_gold_used": False,
        },
        "execution_boundary": [
            "local-only P3 Dream substrate probe",
            "model inference only",
            "does not run optimizer or backward",
            "does not create SwanLab runs",
            "decoded text is recorded only as an offline probe field",
        ],
    }
    write_json(output_dir / "environment.json", env)
    write_json(output_dir / "summary.json", summary)
    append_jsonl(output_dir / "metrics.jsonl", metrics)
    append_jsonl(output_dir / "sample_generations.jsonl", generation)
    summary["summary_json"] = str(output_dir / "summary.json")
    summary["metrics_jsonl"] = str(output_dir / "metrics.jsonl")
    summary["sample_generations_jsonl"] = str(output_dir / "sample_generations.jsonl")
    return summary


def default_output_dir(created_at: int) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(created_at))
    return Path(DEFAULT_OUTPUT_ROOT) / f"dream_instruct_7b_generation_probe_{stamp}"


def resolve_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def collect_environment(args: argparse.Namespace) -> dict[str, Any]:
    import transformers

    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": props.total_memory,
                }
            )
    return {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": cuda_devices,
        "hf_xet_high_performance": os.environ.get("HF_XET_HIGH_PERFORMANCE", ""),
        "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
        "model_path_exists": Path(args.model_path).exists(),
    }


def load_messages(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.messages_json:
        messages = json.loads(Path(args.messages_json).read_text(encoding="utf-8"))
        if not isinstance(messages, list):
            raise ValueError("--messages-json must contain a list")
        return messages
    return [{"role": "user", "content": args.prompt}]


def resolve_mask_token_id(model: Any, tokenizer: Any) -> int | None:
    for value in (
        getattr(getattr(model, "generation_config", None), "mask_token_id", None),
        getattr(getattr(model, "config", None), "mask_token_id", None),
        getattr(tokenizer, "mask_token_id", None),
    ):
        if value is not None:
            return int(value)
    return None


def make_tokens_hook(hook_state: dict[str, Any], limit: int, mask_token_id: int | None):
    def hook(step: int | None, x: torch.Tensor, logits: torch.Tensor | None) -> torch.Tensor:
        if len(hook_state["token_snapshots"]) < limit:
            hook_state["token_snapshots"].append(
                {
                    "step": step,
                    "shape": list(x.shape),
                    "num_mask_tokens": int((x == mask_token_id).sum().item()) if mask_token_id is not None else None,
                }
            )
        return x

    return hook


def make_logits_hook(hook_state: dict[str, Any], limit: int, mask_token_id: int | None):
    def hook(step: int, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if len(hook_state["logit_snapshots"]) < limit:
            with torch.no_grad():
                masked = logits[x == mask_token_id] if mask_token_id is not None else logits.reshape(-1, logits.shape[-1])
                sample = masked[: min(masked.shape[0], 128)].float()
                probs = torch.softmax(sample, dim=-1)
                top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
                entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1)
                hook_state["logit_snapshots"].append(
                    {
                        "step": int(step),
                        "shape": list(logits.shape),
                        "sample_size": int(sample.shape[0]),
                        "top1_prob_mean": float(top2[:, 0].mean().item()) if sample.numel() else None,
                        "top1_top2_margin_mean": float((top2[:, 0] - top2[:, 1]).mean().item())
                        if top2.shape[-1] > 1
                        else None,
                        "entropy_mean": float(entropy.mean().item()) if sample.numel() else None,
                    }
                )
        return logits

    return hook


def make_hidden_hook(hook_state: dict[str, Any]):
    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        tensor = first_tensor(output)
        if tensor is None:
            return
        with torch.no_grad():
            sample = tensor.detach().float()
            hook_state["hidden_snapshots"].append(
                {
                    "shape": list(tensor.shape),
                    "mean_abs": float(sample.abs().mean().item()),
                    "rms": float(torch.sqrt((sample * sample).mean()).item()),
                }
            )

    return hook


def first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if hasattr(value, "last_hidden_state") and isinstance(value.last_hidden_state, torch.Tensor):
        return value.last_hidden_state
    return None


def find_last_layer_module(model: torch.nn.Module) -> tuple[str, torch.nn.Module | None]:
    candidates: list[tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            lower = name.lower()
            if "layer" in lower or "block" in lower or "h" == lower.split(".")[-1]:
                candidates.append((f"{name}.{len(module) - 1}", module[-1]))
    if candidates:
        return candidates[-1]
    return "", None


def decode_generated(tokenizer: Any, input_ids: torch.Tensor, sequences: torch.Tensor) -> str:
    prompt_len = int(input_ids.shape[-1])
    tokens = sequences[0, prompt_len:].detach().cpu().tolist()
    text = tokenizer.decode(tokens, skip_special_tokens=False)
    eos = getattr(tokenizer, "eos_token", None)
    if eos:
        text = text.split(eos)[0]
    return text


def summarize_history(tokenizer: Any, input_ids: torch.Tensor, history: Any, limit: int) -> list[dict[str, Any]]:
    if history is None:
        return []
    prompt_len = int(input_ids.shape[-1])
    selected_indices = select_history_indices(len(history), limit)
    rows = []
    for index in selected_indices:
        seq = history[index]
        tokens = seq[0, prompt_len:].detach().cpu().tolist()
        text = tokenizer.decode(tokens, skip_special_tokens=False)
        rows.append(
            {
                "history_index": int(index),
                "shape": list(seq.shape),
                "decoded_prefix": text[:300],
            }
        )
    return rows


def select_history_indices(length: int, limit: int) -> list[int]:
    if length <= 0 or limit <= 0:
        return []
    if length <= limit:
        return list(range(length))
    indices = {0, length - 1}
    for i in range(1, limit - 1):
        indices.add(round(i * (length - 1) / (limit - 1)))
    return sorted(indices)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
