"""Run Phase C capable-text-agent controls from locked online inputs.

This script performs local-only model evaluation for Phase C.  It reads
``build_p2_phase_c_control_inputs.py`` outputs, calls an OpenAI-compatible chat
completion endpoint when configured, scores final answers offline, and writes
reproducible artifacts.  It never trains, never writes SwanLab runs, and never
uses gold/scorer fields as online inputs.

If no external model credentials are available, use ``--selfcheck`` only.  The
self-check exercises prompt routing, sequential TextMAS flow, scoring, and
artifact writing with a deterministic toy provider; it is not an experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.evaluation.p2_phase_c_scorers import score_qa_answer


DEFAULT_OUTPUT_DIR = "/data1/luyifei/drla/outputs/p2_phase_c_text_agent_runs/text_agent_run_20260601"
DEFAULT_COLA_DIT_PATH = "/data1/luyifei/drla/models/Cola-DLM/cola_dlm/cola_dit"
DEFAULT_COLA_VAE_PATH = "/data1/luyifei/drla/models/Cola-DLM/cola_dlm/cola_vae"
DEFAULT_COLA_TOKENIZER_PATH = "/data1/luyifei/drla/models/Cola-DLM/tokenizer.json"
DEFAULT_COLA_CODE_PATH = "/data1/luyifei/Cola-DLM/code"


def main() -> None:
    args = parse_args()
    summary = run_selfcheck(args) if args.selfcheck else run_eval(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", default="")
    parser.add_argument("--online-inputs-jsonl", default="")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--provider",
        default="openai_compatible",
        choices=["openai_compatible", "local_transformers", "cola_dlm", "mock_selfcheck"],
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", ""))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--local-model-path",
        default="/data1/luyifei/drla/models/Qwen3-4B-Instruct-2507-git",
        help="Local HuggingFace model path for --provider local_transformers.",
    )
    parser.add_argument("--local-device-map", default="auto")
    parser.add_argument("--local-dtype", default="auto")
    parser.add_argument(
        "--local-enable-thinking",
        action="store_true",
        help="Enable model thinking mode when the tokenizer chat template supports it. Default is disabled.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--cola-dit-path", default=DEFAULT_COLA_DIT_PATH)
    parser.add_argument("--cola-dit-lora-path", default="")
    parser.add_argument("--cola-agent-dit-lora-path", default="")
    parser.add_argument("--cola-solver-dit-lora-path", default="")
    parser.add_argument("--cola-vae-path", default=DEFAULT_COLA_VAE_PATH)
    parser.add_argument("--cola-tokenizer-path", default=DEFAULT_COLA_TOKENIZER_PATH)
    parser.add_argument("--cola-code-path", default=DEFAULT_COLA_CODE_PATH)
    parser.add_argument("--cola-device", default="auto")
    parser.add_argument(
        "--cola-prompt-style",
        choices=["chat_join", "plain_qa_v1", "squad_template_v1", "candidate_constrained_v1"],
        default="chat_join",
    )
    parser.add_argument("--cola-timestep-num", type=int, default=16)
    parser.add_argument("--cola-guidance-scale", type=float, default=7.0)
    parser.add_argument("--cola-noise-seed", default="66")
    parser.add_argument("--cola-top-k", type=int, default=50)
    parser.add_argument("--cola-top-p", type=float, default=0.9)
    parser.add_argument("--cola-repetition-penalty", type=float, default=1.1)
    parser.add_argument("--cola-pad-token-id", type=int, default=100277)
    parser.add_argument("--cola-eos-token-id", type=int, default=100257)
    parser.add_argument("--cola-im-end-token-id", type=int, default=100265)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--prediction-extraction-mode",
        choices=["default", "first_segment"],
        default="default",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument("--conditions", default="", help="Comma-separated condition filter.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing generations.jsonl in output-dir.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if not args.selfcheck and (not args.manifest_json or not args.online_inputs_jsonl):
        raise ValueError("Pass --manifest-json and --online-inputs-jsonl, or use --selfcheck")
    if not args.selfcheck and args.provider == "mock_selfcheck":
        raise ValueError("mock_selfcheck provider is only allowed with --selfcheck")
    return args


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite and not args.resume:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    generations_path = output_dir / "generations.jsonl"
    manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
    samples = {str(sample["sample_id"]): sample for sample in manifest.get("samples", [])}
    rows = read_jsonl(Path(args.online_inputs_jsonl))
    rows = filter_rows(rows, args.conditions, args.max_rows, args.row_offset)
    provider = make_provider(args)

    generations = read_jsonl(
        generations_path,
        allow_truncated_final_line=args.resume,
    ) if args.resume and generations_path.exists() else []
    if args.resume:
        write_jsonl(generations_path, generations)
    else:
        generations_path.write_text("", encoding="utf-8")
    restore_agent_cache(provider, generations)
    completed = {str(row.get("row_id", "")) for row in generations}
    for row in rows:
        if str(row.get("row_id", "")) in completed:
            continue
        sample = samples.get(str(row.get("sample_id", "")))
        if sample is None:
            raise ValueError(f"unknown sample_id in online inputs: {row.get('sample_id')}")
        result = run_condition(row, sample, samples, provider, args)
        result["row_index"] = len(generations) + 1
        append_jsonl(generations_path, result)
        generations.append(result)

    return write_outputs(args, output_dir, manifest, rows, generations, selfcheck=args.provider == "mock_selfcheck")


def run_selfcheck(args: argparse.Namespace) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        manifest_path, online_inputs_path = write_selfcheck_inputs(tmp)
        self_args = argparse.Namespace(
            **{
                **vars(args),
                "manifest_json": str(manifest_path),
                "online_inputs_jsonl": str(online_inputs_path),
                "provider": "mock_selfcheck",
                "model": "mock_selfcheck",
                "output_dir": args.output_dir,
                "overwrite": args.overwrite,
                "resume": args.resume,
                "max_rows": args.max_rows,
                "row_offset": args.row_offset,
                "conditions": args.conditions,
            }
        )
        summary = run_eval(self_args)
    summary["selfcheck"] = {
        "status": "pass",
        "meaning": "Toy provider exercised routing/scoring only; not an experiment.",
    }
    return summary


def run_condition(
    row: dict[str, Any],
    sample: dict[str, Any],
    samples: dict[str, dict[str, Any]],
    provider: "ChatProvider",
    args: argparse.Namespace,
) -> dict[str, Any]:
    condition = str(row["condition"])
    fields = row.get("online_input_fields", {})
    agent_messages: list[dict[str, str]] = []
    if condition in {"single_q_only", "single_full_info", "textmas_no_message"}:
        final_answer = provider.chat(make_solver_messages(fields, upstream_messages=[]), args)
    elif condition in {"textmas_matched", "textmas_compressed_state"}:
        agent_messages = get_or_make_agent_messages(
            cache_key=f"{row['sample_id']}::{condition}",
            observations=fields.get("agent_private_observations", []),
            provider=provider,
            args=args,
            compressed=condition == "textmas_compressed_state",
        )
        final_answer = provider.chat(make_solver_messages(fields, upstream_messages=agent_messages), args)
    elif condition == "textmas_shuffled_message":
        control_id = str(row.get("control_source_sample_id", ""))
        control_sample = samples.get(control_id)
        if control_sample is None:
            raise ValueError(f"missing control sample for shuffled row: {control_id}")
        agent_messages = get_or_make_agent_messages(
            cache_key=f"{control_id}::textmas_matched",
            observations=sample_agent_observations(control_sample),
            provider=provider,
            args=args,
            compressed=False,
        )
        final_answer = provider.chat(make_solver_messages(fields, upstream_messages=agent_messages), args)
    elif condition == "textmas_wrong_evidence_or_wrong_shard":
        agent_messages = get_or_make_agent_messages(
            cache_key=f"{row['row_id']}::wrong_evidence",
            observations=fields.get("agent_private_observations", []),
            provider=provider,
            args=args,
            compressed=False,
        )
        final_answer = provider.chat(make_solver_messages(fields, upstream_messages=agent_messages), args)
    else:
        raise ValueError(f"unknown condition: {condition}")

    prediction = extract_final_answer(final_answer, mode=args.prediction_extraction_mode)
    scoring = sample.get("scoring", {})
    score = score_qa_answer(
        prediction,
        scoring.get("gold_answer", ""),
        scoring.get("answer_aliases", []) or [],
    ).to_dict()
    return {
        "row_id": row["row_id"],
        "sample_id": row["sample_id"],
        "task_name": row.get("task_name", ""),
        "split": row.get("split", ""),
        "condition": condition,
        "control_source_sample_id": row.get("control_source_sample_id", "")
        or row.get("online_input_fields", {}).get("shuffled_message_source_sample_id", "")
        or row.get("online_input_fields", {}).get("wrong_evidence_source_sample_id", ""),
        "model": args.model,
        "provider": provider.name,
        "online_input_fields": row.get("online_input_fields", {}),
        "agent_messages": agent_messages,
        "raw_final_output": final_answer,
        "prediction": prediction,
        "score": score,
        "primary_score": score["primary_score"],
        "token_f1": score["token_f1"],
        "exact_match": score["exact_match"],
        "prompt_contract_version": row.get("prompt_contract_version", ""),
    }


def restore_agent_cache(provider: "ChatProvider", generations: list[dict[str, Any]]) -> None:
    for row in generations:
        messages = row.get("agent_messages", [])
        if not messages:
            continue
        condition = str(row.get("condition", ""))
        sample_id = str(row.get("sample_id", ""))
        row_id = str(row.get("row_id", ""))
        if condition in {"textmas_matched", "textmas_compressed_state"}:
            provider.agent_cache[f"{sample_id}::{condition}"] = messages
        elif condition == "textmas_wrong_evidence_or_wrong_shard":
            provider.agent_cache[f"{row_id}::wrong_evidence"] = messages
        elif condition == "textmas_shuffled_message":
            fields = row.get("online_input_fields", {})
            control_id = str(fields.get("shuffled_message_source_sample_id", ""))
            if control_id:
                provider.agent_cache[f"{control_id}::textmas_matched"] = messages


def get_or_make_agent_messages(
    *,
    cache_key: str,
    observations: list[dict[str, Any]],
    provider: "ChatProvider",
    args: argparse.Namespace,
    compressed: bool,
) -> list[dict[str, str]]:
    if cache_key in provider.agent_cache:
        return provider.agent_cache[cache_key]
    messages = []
    for observation in observations:
        content = provider.chat(make_agent_messages(observation, compressed=compressed), args)
        messages.append(
            {
                "agent_id": str(observation.get("agent_id", "")),
                "role": str(observation.get("role", "")),
                "message": content,
            }
        )
    provider.agent_cache[cache_key] = messages
    return messages


def make_agent_messages(observation: dict[str, Any], *, compressed: bool) -> list[dict[str, str]]:
    system = (
        "You are an evidence agent. Use only your private observation. Do not use "
        "gold labels or scorer outputs. Report only information useful for the final solver."
    )
    if compressed:
        system += " Output compact JSON with useful_facts, uncertainty, and answer_hint."
    user = json.dumps(
        {
            "agent_id": observation.get("agent_id", ""),
            "role": observation.get("role", ""),
            "private_observation": observation.get("private_observation", ""),
            "allowed_output_contract": observation.get("allowed_output_contract", ""),
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def make_solver_messages(fields: dict[str, Any], upstream_messages: list[dict[str, str]]) -> list[dict[str, str]]:
    system = (
        "You are the final solver for an evidence-split QA protocol. Use only "
        "the provided online input and upstream messages. Return exactly one line "
        "in the format 'Final answer: <short answer>'. The answer must be only the "
        "entity, date, number, title, or short phrase requested. Do not explain. "
        "Do not mention hidden labels or scorer outputs."
    )
    payload = {
        "question": fields.get("question", ""),
        "public_context": fields.get("public_context", ""),
        "full_evidence": fields.get("full_evidence", ""),
        "upstream_messages": upstream_messages or fields.get("upstream_messages", []),
        "solver_message_contract": fields.get("solver_message_contract", ""),
        "agent_output_schema": fields.get("agent_output_schema", None),
        "candidate_answers": fields.get("candidate_answers", None),
    }
    payload = {key: value for key, value in payload.items() if value not in ("", None)}
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)},
    ]


def sample_agent_observations(sample: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "agent_id": view.get("agent_id", ""),
            "role": view.get("role", ""),
            "private_observation": view.get("private_observation", ""),
            "allowed_output_contract": view.get("allowed_output_contract", ""),
        }
        for view in sample.get("agent_views", [])
    ]


def extract_final_answer(text: str, *, mode: str = "default") -> str:
    cleaned = text.strip()
    if mode == "first_segment":
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if not lines:
            return ""
        first = lines[0].strip('"')
        first = strip_generation_special_tokens(first)
        match = re.search(r"(?i)\b(?:final\s+answer|answer)\b\s*(?:is|=|:|：)?\s*(.+)$", first)
        return match.group(1).strip().strip('"') if match else first
    if mode != "default":
        raise ValueError(f"unknown prediction extraction mode: {mode}")
    patterns = [
        r"(?i)final answer\s*:\s*(.+)$",
        r"(?i)final answer\s+is\s*:?\s*(.+)$",
        r"(?i)answer\s*:\s*(.+)$",
        r"(?i)answer\s+is\s*:?\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.MULTILINE)
        if match:
            return match.group(1).strip().strip('"')
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return lines[-1].strip('"') if lines else ""


def strip_generation_special_tokens(text: str) -> str:
    return re.split(r"<\|endoftext\|>|<\|im_end\|>", text, maxsplit=1)[0].strip()


def make_provider(args: argparse.Namespace) -> "ChatProvider":
    if args.provider == "mock_selfcheck":
        return MockSelfcheckProvider()
    if args.provider == "openai_compatible":
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            raise EnvironmentError(
                f"{args.api_key_env} is not set; cannot run capable text-agent evaluation. "
                "Use --selfcheck for local routing validation only."
            )
        if not args.model:
            raise ValueError("--model or OPENAI_MODEL is required for openai_compatible provider")
        return OpenAICompatibleProvider(base_url=args.base_url, api_key=api_key)
    if args.provider == "local_transformers":
        model_path = args.local_model_path or args.model
        if not model_path:
            raise ValueError("--local-model-path or --model is required for local_transformers provider")
        args.model = args.model or model_path
        return LocalTransformersProvider(
            model_path=model_path,
            device_map=args.local_device_map,
            dtype=args.local_dtype,
            enable_thinking=args.local_enable_thinking,
        )
    if args.provider == "cola_dlm":
        args.model = args.model or "official_cola_dlm"
        return ColaDLMProvider(
            dit_path=args.cola_dit_path,
            dit_lora_path=args.cola_dit_lora_path,
            agent_dit_lora_path=args.cola_agent_dit_lora_path,
            solver_dit_lora_path=args.cola_solver_dit_lora_path,
            vae_path=args.cola_vae_path,
            tokenizer_path=args.cola_tokenizer_path,
            cola_code_path=args.cola_code_path,
            device_arg=args.cola_device,
            prompt_style=args.cola_prompt_style,
            noise_seed=args.cola_noise_seed,
        )
    raise ValueError(f"unknown provider: {args.provider}")


class ChatProvider:
    name = "base"

    def __init__(self) -> None:
        self.agent_cache: dict[str, list[dict[str, str]]] = {}

    def chat(self, messages: list[dict[str, str]], args: argparse.Namespace) -> str:
        raise NotImplementedError


class OpenAICompatibleProvider(ChatProvider):
    name = "openai_compatible"

    def __init__(self, *, base_url: str, api_key: str) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def chat(self, messages: list[dict[str, str]], args: argparse.Namespace) -> str:
        payload = {
            "model": args.model,
            "messages": messages,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=args.timeout_seconds) as client:
            response = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        return str(data["choices"][0]["message"]["content"]).strip()


class LocalTransformersProvider(ChatProvider):
    name = "local_transformers"

    def __init__(self, *, model_path: str, device_map: str, dtype: str, enable_thinking: bool) -> None:
        super().__init__()
        self.enable_thinking = enable_thinking
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - dependency/environment guard.
            raise RuntimeError(
                "local_transformers provider requires torch and transformers in the active env"
            ) from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_kwargs = {"device_map": device_map, "trust_remote_code": True}
        if dtype and dtype != "auto":
            model_kwargs["dtype"] = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        else:
            model_kwargs["dtype"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def chat(self, messages: list[dict[str, str]], args: argparse.Namespace) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            except TypeError:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            prompt = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
            prompt += "\nassistant:"
        inputs = self.tokenizer([prompt], return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        do_sample = args.temperature > 0
        generate_kwargs = {
            **inputs,
            "max_new_tokens": args.max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = args.temperature
        with self.torch.no_grad():
            output = self.model.generate(**generate_kwargs)
        generated = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


class ColaDLMProvider(ChatProvider):
    name = "cola_dlm"

    def __init__(
        self,
        *,
        dit_path: str,
        dit_lora_path: str,
        agent_dit_lora_path: str,
        solver_dit_lora_path: str,
        vae_path: str,
        tokenizer_path: str,
        cola_code_path: str,
        device_arg: str,
        prompt_style: str,
        noise_seed: str,
    ) -> None:
        super().__init__()
        import sys

        import torch

        from drla.scripts.collect_cola_block_traces import load_cola_symbols, resolve_device

        if cola_code_path and cola_code_path not in sys.path:
            sys.path.insert(0, cola_code_path)
        cola = load_cola_symbols()
        from cola_dlm import generate_task_repaint_inference

        self.torch = torch
        self.generate_task_repaint_inference = generate_task_repaint_inference
        self.device = resolve_device(device_arg)
        if self.device.type != "cuda":
            raise RuntimeError(f"cola_dlm provider requires CUDA for official Cola generation: {self.device}")
        if noise_seed:
            os.environ["COLA_INFER_PER_SAMPLE_NOISE_SEED"] = str(noise_seed)
        self.tokenizer = cola["Tokenizer"].from_file(tokenizer_path)
        solver_lora = solver_dit_lora_path or dit_lora_path
        agent_lora = agent_dit_lora_path or dit_lora_path
        self.solver_dit = self.load_dit(cola["ColaDiTModel"], dit_path, solver_lora)
        self.agent_dit = (
            self.solver_dit
            if agent_lora == solver_lora
            else self.load_dit(cola["ColaDiTModel"], dit_path, agent_lora)
        )
        self.vae = cola["ColaTextVAEModel"].from_pretrained(vae_path).to(self.device).eval()
        self.prompt_style = prompt_style
        self.call_index = 0

    def load_dit(self, dit_cls: Any, dit_path: str, lora_path: str) -> Any:
        dit = dit_cls.from_pretrained(dit_path).to(self.device).eval()
        if lora_path:
            from peft import PeftModel

            dit = PeftModel.from_pretrained(dit, lora_path).merge_and_unload().to(self.device).eval()
        return dit

    def chat(self, messages: list[dict[str, str]], args: argparse.Namespace) -> str:
        self.call_index += 1
        prompt_item = self.format_prompt_item(messages)
        task_name = str(prompt_item.pop("_task_name", "p2_phase_c_musique"))
        dit = self.agent_dit if is_evidence_agent_messages(messages) else self.solver_dit
        with self.torch.no_grad():
            outputs = self.generate_task_repaint_inference(
                dit=dit,
                vae=self.vae,
                tokenizer=self.tokenizer,
                prompts=[
                    {
                        "id": f"phase_c_cola_{self.call_index}",
                        **prompt_item,
                        "answer": "",
                        "ground_truth": "",
                    }
                ],
                task_name=task_name,
                device=self.device,
                timestep_num=args.cola_timestep_num,
                guidance_scale=args.cola_guidance_scale,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.cola_top_k,
                top_p=args.cola_top_p,
                repetition_penalty=args.cola_repetition_penalty,
                pad_token_id=args.cola_pad_token_id,
                eos_token_id=args.cola_eos_token_id,
                im_end_token_id=args.cola_im_end_token_id,
                is_sft=False,
            )
        return str(outputs[0].get("generate", "")).strip()

    def format_prompt_item(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        if self.prompt_style == "candidate_constrained_v1":
            return format_cola_candidate_constrained_item(messages)
        if self.prompt_style == "squad_template_v1":
            return format_cola_squad_template_item(messages)
        return {"_task_name": "p2_phase_c_musique", "question": self.format_prompt(messages)}

    def format_prompt(self, messages: list[dict[str, str]]) -> str:
        if self.prompt_style == "plain_qa_v1":
            return format_cola_plain_qa_prompt(messages)
        prompt = "\n\n".join(
            f"{message.get('role', 'user').upper()}:\n{message.get('content', '')}"
            for message in messages
        )
        return prompt + "\n\nASSISTANT:"


def format_cola_plain_qa_prompt(messages: list[dict[str, str]]) -> str:
    user_text = "\n".join(message.get("content", "") for message in messages if message.get("role") == "user")
    payload = parse_json_object(user_text)
    if is_evidence_agent_messages(messages):
        observation = str(payload.get("private_observation", user_text)).strip()
        return (
            "Read the evidence below and write only the useful facts.\n\n"
            f"Evidence:\n{observation}\n\n"
            "Useful facts:"
        )
    question = str(payload.get("question", "")).strip()
    full_evidence = str(payload.get("full_evidence", "")).strip()
    upstream_messages = payload.get("upstream_messages", [])
    lines = []
    if question:
        lines.append(f"Question: {question}")
    else:
        lines.append(user_text.strip())
    if full_evidence:
        lines.append(f"\nEvidence:\n{full_evidence}")
    if isinstance(upstream_messages, list) and upstream_messages:
        notes = []
        for item in upstream_messages:
            if isinstance(item, dict):
                role = str(item.get("role") or item.get("agent_id") or "agent")
                message = str(item.get("message", "")).strip()
                if message:
                    notes.append(f"- {role}: {message}")
        if notes:
            lines.append("\nUseful facts from agents:\n" + "\n".join(notes))
    lines.append("\nAnswer with only the short final answer.\nAnswer:")
    return "\n".join(lines)


def format_cola_squad_template_item(messages: list[dict[str, str]]) -> dict[str, Any]:
    user_text = "\n".join(message.get("content", "") for message in messages if message.get("role") == "user")
    payload = parse_json_object(user_text)
    if is_evidence_agent_messages(messages):
        observation = str(payload.get("private_observation", user_text)).strip()
        return {
            "_task_name": "p2_phase_c_musique",
            "question": (
                "Read the evidence below and write only the useful facts.\n\n"
                f"Evidence:\n{observation}\n\n"
                "Useful facts:"
            ),
        }

    question = str(payload.get("question", "")).strip()
    context_parts = []
    full_evidence = str(payload.get("full_evidence", "")).strip()
    if full_evidence:
        context_parts.append(full_evidence)
    upstream_messages = payload.get("upstream_messages", [])
    if isinstance(upstream_messages, list) and upstream_messages:
        notes = []
        for item in upstream_messages:
            if isinstance(item, dict):
                role = str(item.get("role") or item.get("agent_id") or "agent")
                message = str(item.get("message", "")).strip()
                if message:
                    notes.append(f"{role}: {message}")
        if notes:
            context_parts.append("\n".join(notes))
    if not context_parts:
        context_parts.append(str(payload.get("public_context", "")).strip())
    return {
        "_task_name": "squad",
        "context": "\n\n".join(part for part in context_parts if part),
        "question": question or user_text.strip(),
    }


def format_cola_candidate_constrained_item(messages: list[dict[str, str]]) -> dict[str, Any]:
    user_text = "\n".join(message.get("content", "") for message in messages if message.get("role") == "user")
    payload = parse_json_object(user_text)
    if is_evidence_agent_messages(messages):
        return format_cola_squad_template_item(messages)

    candidates = payload.get("candidate_answers", [])
    if not isinstance(candidates, list) or not candidates:
        return format_cola_squad_template_item(messages)
    question = str(payload.get("question", "")).strip()
    full_evidence = str(payload.get("full_evidence", "")).strip()
    if not full_evidence:
        return format_cola_squad_template_item(messages)
    candidate_lines = []
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            continue
        text = one_line(candidate.get("text", ""))
        if not text:
            continue
        meta_parts = []
        for field, label in [("rule", "rule"), ("source_title", "source"), ("evidence_index", "evidence")]:
            value = one_line(candidate.get(field, ""))
            if value:
                meta_parts.append(f"{label}={value}")
        meta = f" ({'; '.join(meta_parts)})" if meta_parts else ""
        candidate_lines.append(f"[{index}] {text}{meta}")
    if not candidate_lines:
        return format_cola_squad_template_item(messages)
    prompt = (
        "Select the best final answer from the candidate list using the evidence. "
        "Write only one short answer.\n\n"
        f"Question: {question}\n\n"
        f"Evidence:\n{full_evidence}\n\n"
        "Candidates:\n"
        + "\n".join(candidate_lines)
        + "\n\nFinal answer:"
    )
    return {"_task_name": "p2_phase_c_musique", "question": prompt}


def one_line(value: Any) -> str:
    return " ".join(str(value).strip().split())


def is_evidence_agent_messages(messages: list[dict[str, str]]) -> bool:
    system_text = "\n".join(message.get("content", "") for message in messages if message.get("role") == "system")
    return "evidence agent" in system_text.lower()


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


class MockSelfcheckProvider(ChatProvider):
    name = "mock_selfcheck"

    def chat(self, messages: list[dict[str, str]], args: argparse.Namespace) -> str:
        text = "\n".join(message.get("content", "") for message in messages)
        lowered = text.lower()
        if "clear daytime sky" in lowered or "blue" in lowered:
            return "Final answer: blue"
        if "ripe strawberries" in lowered or "red" in lowered:
            return "Final answer: red"
        return "Final answer: unknown"


def write_outputs(
    args: argparse.Namespace,
    output_dir: Path,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    *,
    selfcheck: bool,
) -> dict[str, Any]:
    generations_path = output_dir / "generations.jsonl"
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    condition_csv_path = output_dir / "condition_metrics.csv"
    write_jsonl(generations_path, generations)
    condition_metrics = aggregate_condition_metrics(generations)
    write_condition_csv(condition_csv_path, condition_metrics)
    metrics = {
        "num_rows": len(rows),
        "num_generations": len(generations),
        "mean_primary_score": mean([row["primary_score"] for row in generations]),
        "mean_token_f1": mean([row["token_f1"] for row in generations]),
        "selfcheck": int(selfcheck),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "created_at": int(time.time()),
        "status": "pass",
        "provider": args.provider,
        "model": args.model,
        "run_config": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "conditions": args.conditions,
            "row_offset": args.row_offset,
            "max_rows": args.max_rows,
            "prediction_extraction_mode": args.prediction_extraction_mode,
            "cola_prompt_style": args.cola_prompt_style,
            "cola_timestep_num": args.cola_timestep_num,
            "cola_guidance_scale": args.cola_guidance_scale,
            "cola_noise_seed": args.cola_noise_seed,
            "cola_dit_lora_path": args.cola_dit_lora_path,
            "cola_agent_dit_lora_path": args.cola_agent_dit_lora_path,
            "cola_solver_dit_lora_path": args.cola_solver_dit_lora_path,
        },
        "manifest_json": args.manifest_json,
        "online_inputs_jsonl": args.online_inputs_jsonl,
        "generations_jsonl": str(generations_path),
        "metrics_jsonl": str(metrics_path),
        "condition_metrics_csv": str(condition_csv_path),
        "num_rows": len(rows),
        "condition_metrics": condition_metrics,
        "execution_boundary": [
            "local-only Phase C text-agent evaluation",
            "no optimizer or backward",
            "no SwanLab run",
            "gold/scorer fields used only for offline scoring after generation",
            "selfcheck is not an experiment" if selfcheck else "model outputs generated by configured provider",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_json"] = str(summary_path)
    return summary


def aggregate_condition_metrics(generations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in generations:
        grouped.setdefault(str(row["condition"]), []).append(row)
    metrics = {}
    for condition, rows in sorted(grouped.items()):
        metrics[condition] = {
            "num_rows": len(rows),
            "primary_score_mean": mean([row["primary_score"] for row in rows]),
            "token_f1_mean": mean([row["token_f1"] for row in rows]),
            "exact_match_mean": mean([row["exact_match"] for row in rows]),
        }
    return metrics


def write_condition_csv(path: Path, metrics: dict[str, dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["condition", "num_rows", "primary_score_mean", "token_f1_mean", "exact_match_mean"],
        )
        writer.writeheader()
        for condition, values in metrics.items():
            writer.writerow({"condition": condition, **values})


def filter_rows(rows: list[dict[str, Any]], conditions: str, max_rows: int, row_offset: int = 0) -> list[dict[str, Any]]:
    if conditions:
        allowed = {condition.strip() for condition in conditions.split(",") if condition.strip()}
        rows = [row for row in rows if row.get("condition") in allowed]
    if row_offset:
        if row_offset < 0:
            raise ValueError("--row-offset must be non-negative")
        rows = rows[row_offset:]
    if max_rows:
        rows = rows[:max_rows]
    return rows


def read_jsonl(path: Path, *, allow_truncated_final_line: bool = False) -> list[dict[str, Any]]:
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_no, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if allow_truncated_final_line and line_no == len(lines):
                break
            raise
        if not isinstance(row, dict):
            raise ValueError(f"Expected object at {path}:{line_no}")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_selfcheck_inputs(tmp: Path) -> tuple[Path, Path]:
    manifest = {
        "samples": [
            {
                "sample_id": "toy_blue",
                "task_name": "toy_evidence_split",
                "split": "calibration",
                "question": "Which color is implied?",
                "public_context": "Toy selfcheck only.",
                "agent_views": [
                    {
                        "agent_id": "agent_a",
                        "role": "evidence_holder_a",
                        "private_observation": "The clue refers to a clear daytime sky.",
                        "allowed_output_contract": "Summarize the clue.",
                    },
                    {
                        "agent_id": "agent_b",
                        "role": "evidence_holder_b",
                        "private_observation": "The answer is not red.",
                        "allowed_output_contract": "Summarize the clue.",
                    },
                ],
                "scoring": {"gold_answer": "blue", "answer_aliases": []},
            },
            {
                "sample_id": "toy_red",
                "task_name": "toy_evidence_split",
                "split": "calibration",
                "question": "Which color is implied?",
                "public_context": "Toy selfcheck only.",
                "agent_views": [
                    {
                        "agent_id": "agent_a",
                        "role": "evidence_holder_a",
                        "private_observation": "The clue refers to ripe strawberries.",
                        "allowed_output_contract": "Summarize the clue.",
                    },
                    {
                        "agent_id": "agent_b",
                        "role": "evidence_holder_b",
                        "private_observation": "The answer is not blue.",
                        "allowed_output_contract": "Summarize the clue.",
                    },
                ],
                "scoring": {"gold_answer": "red", "answer_aliases": []},
            },
        ]
    }
    rows = []
    conditions = ["single_q_only", "single_full_info", "textmas_matched", "textmas_no_message"]
    for sample in manifest["samples"]:
        for condition in conditions:
            fields = {"question": sample["question"], "public_context": sample["public_context"]}
            if condition == "single_full_info":
                fields["full_evidence"] = "\n".join(view["private_observation"] for view in sample["agent_views"])
            if condition == "textmas_matched":
                fields["agent_private_observations"] = sample_agent_observations(sample)
            if condition == "textmas_no_message":
                fields["upstream_messages"] = []
            rows.append(
                {
                    "row_id": f"{sample['sample_id']}::{condition}",
                    "sample_id": sample["sample_id"],
                    "task_name": sample["task_name"],
                    "split": sample["split"],
                    "condition": condition,
                    "online_input_fields": fields,
                    "used_for_prompt_repair": False,
                }
            )
    manifest_path = tmp / "manifest.json"
    rows_path = tmp / "online_inputs.jsonl"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(rows_path, rows)
    return manifest_path, rows_path


if __name__ == "__main__":
    main()
