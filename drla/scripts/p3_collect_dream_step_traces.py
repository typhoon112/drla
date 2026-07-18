"""Collect P3 Dream denoising-step traces for offline readiness labels.

This local-only D3 script runs Dream-v0-Instruct-7B on locked MuSiQue
evidence-split rows and records per-call denoising process summaries. It does
not train, run backward, update weights, or create SwanLab runs. Decoded text
and scorer fields are saved only as offline teacher/evaluation fields; online
student/communication features must come from token/logit/process summaries and
future latent refs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.evaluation.p2_phase_c_scorers import score_qa_answer  # noqa: E402
from drla.scripts.run_p2_phase_c_text_agents import (  # noqa: E402
    append_jsonl,
    extract_final_answer,
    filter_rows,
    make_agent_messages,
    make_solver_messages,
    read_jsonl,
    sample_agent_observations,
    write_jsonl,
)
from drla.scripts.p3_train_dream_readiness_student import (  # noqa: E402
    DreamStepReadinessStudent,
    FEATURE_NAMES,
    TrainConfig,
    event_features,
)


DEFAULT_MANIFEST_JSON = (
    "/data1/luyifei/drla/outputs/p2_phase_c_manifests/"
    "musique_calibration_manifest_200_seed20260601/manifest.json"
)
DEFAULT_ONLINE_INPUTS_JSONL = (
    "/data1/luyifei/drla/outputs/p2_phase_c_control_inputs/"
    "musique_calibration_controls_200_seed20260601_v1_strict_wrong/online_inputs.jsonl"
)
DEFAULT_MODEL_PATH = "/data1/luyifei/drla/models/Dream-v0-Instruct-7B"
DEFAULT_OUTPUT_ROOT = "/data1/luyifei/drla/outputs/p3_dream_traces"
DEFAULT_POLICY_EVAL_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_readiness_policy_eval/"
    "dream_step_readiness_student_v1_full200_with_hidden_policy_eval_20260606"
)
DEFAULT_READINESS_CHECKPOINT = (
    "/data1/luyifei/drla/outputs/p3_dream_readiness_students/"
    "dream_step_readiness_student_v1_full200_with_hidden_seed20260606_20260606/best_checkpoint.pt"
)


def main() -> None:
    summary = collect_traces(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--online-inputs-jsonl", default=DEFAULT_ONLINE_INPUTS_JSONL)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model", default="Dream-v0-Instruct-7B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--dream-steps", type=int, default=64)
    parser.add_argument("--snapshot-stride", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--alg", default="entropy")
    parser.add_argument("--alg-temp", type=float, default=0.0)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--conditions", default="single_full_info,textmas_matched")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument(
        "--prediction-extraction-mode",
        choices=["default", "first_segment"],
        default="first_segment",
    )
    parser.add_argument("--logit-sample-positions", type=int, default=256)
    parser.add_argument("--decode-step-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--hidden-capture-mode",
        choices=["none", "summary", "suffix_tensor", "selected_suffix_tensor"],
        default="summary",
        help="Capture last-layer hidden summaries or saved suffix tensors as online latent features.",
    )
    parser.add_argument("--hidden-save-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--policy-eval-dir", default=DEFAULT_POLICY_EVAL_DIR)
    parser.add_argument("--readiness-checkpoint", default=DEFAULT_READINESS_CHECKPOINT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def collect_traces(args: argparse.Namespace) -> dict[str, Any]:
    created_at = int(time.time())
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(created_at)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
    samples = {str(sample["sample_id"]): sample for sample in manifest.get("samples", [])}
    rows = filter_rows(read_jsonl(Path(args.online_inputs_jsonl)), args.conditions, 0, args.row_offset)
    rows = filter_by_sample_count(rows, args.max_samples)
    if args.max_rows:
        rows = rows[: args.max_rows]
    if not rows:
        raise ValueError("No rows selected")

    provider = DreamTraceProvider(args, output_dir)
    generations_path = output_dir / "generations.jsonl"
    generations_path.write_text("", encoding="utf-8")
    errors = []
    generations = []

    for row_index, row in enumerate(rows, start=1):
        sample = samples.get(str(row.get("sample_id", "")))
        if sample is None:
            raise ValueError(f"unknown sample_id in online inputs: {row.get('sample_id')}")
        try:
            result = run_traced_condition(row, sample, samples, provider, args)
            result["row_index"] = row_index
        except Exception as exc:
            result = error_result(row, args, provider, exc, row_index)
            errors.append(result)
        append_jsonl(generations_path, result)
        generations.append(result)

    return write_outputs(args, output_dir, manifest, rows, generations, errors, provider, created_at)


def run_traced_condition(
    row: dict[str, Any],
    sample: dict[str, Any],
    samples: dict[str, dict[str, Any]],
    provider: "DreamTraceProvider",
    args: argparse.Namespace,
) -> dict[str, Any]:
    condition = str(row["condition"])
    fields = row.get("online_input_fields", {})
    agent_messages: list[dict[str, str]] = []
    row_context = {
        "row_id": row.get("row_id", ""),
        "sample_id": row.get("sample_id", ""),
        "condition": condition,
    }
    if condition in {"single_q_only", "single_full_info", "textmas_no_message"}:
        final_answer = provider.chat_trace(
            make_solver_messages(fields, upstream_messages=[]),
            {**row_context, "agent_role": "solver", "agent_id": "final_solver"},
            args,
        )
    elif condition in {"textmas_matched", "textmas_compressed_state"}:
        agent_messages = trace_agent_messages(
            provider,
            fields.get("agent_private_observations", []),
            row_context,
            args,
            compressed=condition == "textmas_compressed_state",
        )
        final_answer = provider.chat_trace(
            make_solver_messages(fields, upstream_messages=agent_messages),
            {**row_context, "agent_role": "solver", "agent_id": "final_solver"},
            args,
        )
    elif condition == "textmas_shuffled_message":
        control_id = str(row.get("control_source_sample_id", ""))
        control_sample = samples.get(control_id)
        if control_sample is None:
            raise ValueError(f"missing control sample for shuffled row: {control_id}")
        agent_messages = trace_agent_messages(
            provider,
            sample_agent_observations(control_sample),
            {**row_context, "control_source_sample_id": control_id},
            args,
            compressed=False,
        )
        final_answer = provider.chat_trace(
            make_solver_messages(fields, upstream_messages=agent_messages),
            {**row_context, "agent_role": "solver", "agent_id": "final_solver"},
            args,
        )
    elif condition == "textmas_wrong_evidence_or_wrong_shard":
        agent_messages = trace_agent_messages(
            provider,
            fields.get("agent_private_observations", []),
            row_context,
            args,
            compressed=False,
        )
        final_answer = provider.chat_trace(
            make_solver_messages(fields, upstream_messages=agent_messages),
            {**row_context, "agent_role": "solver", "agent_id": "final_solver"},
            args,
        )
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
        "model": args.model,
        "provider": provider.name,
        "status": "ok",
        "agent_messages": agent_messages,
        "raw_final_output": final_answer,
        "prediction": prediction,
        "score": score,
        "primary_score": score["primary_score"],
        "token_f1": score["token_f1"],
        "exact_match": score["exact_match"],
        "trace_call_ids": provider.current_row_call_ids(),
        "offline_teacher_fields_present": ["raw_final_output", "prediction", "score"],
        "online_feature_boundary": "trace step summaries exclude gold/scorer fields",
    }


def trace_agent_messages(
    provider: "DreamTraceProvider",
    observations: list[dict[str, Any]],
    row_context: dict[str, str],
    args: argparse.Namespace,
    *,
    compressed: bool,
) -> list[dict[str, str]]:
    messages = []
    for observation in observations:
        agent_id = str(observation.get("agent_id", ""))
        content = provider.chat_trace(
            make_agent_messages(observation, compressed=compressed),
            {**row_context, "agent_role": "evidence_agent", "agent_id": agent_id},
            args,
        )
        messages.append(
            {
                "agent_id": agent_id,
                "role": str(observation.get("role", "")),
                "message": content,
            }
        )
    return messages


class DreamTraceProvider:
    name = "dream_trace"

    def __init__(self, args: argparse.Namespace, output_dir: Path) -> None:
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("DreamTraceProvider requires CUDA when --device starts with cuda")
        self.device = torch.device(args.device)
        self.dtype = resolve_dtype(args.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device).eval()
        self.mask_token_id = resolve_mask_token_id(self.model, self.tokenizer)
        self.call_index = 0
        self.trace_path = output_dir / "traces.jsonl"
        self.call_metrics_path = output_dir / "dream_trace_call_metrics.jsonl"
        self.trace_path.write_text("", encoding="utf-8")
        self.call_metrics_path.write_text("", encoding="utf-8")
        self.call_metrics: list[dict[str, Any]] = []
        self._row_call_ids: list[str] = []
        self.hidden_ref_dir = output_dir / "hidden_refs"
        if args.hidden_capture_mode in {"suffix_tensor", "selected_suffix_tensor"}:
            self.hidden_ref_dir.mkdir(parents=True, exist_ok=True)
        self.readiness_model: DreamStepReadinessStudent | None = None
        self.readiness_feature_stats: dict[str, list[float]] = {}
        self.selected_policy: dict[str, float] = {}
        if args.hidden_capture_mode == "selected_suffix_tensor":
            (
                self.readiness_model,
                self.readiness_feature_stats,
                self.selected_policy,
            ) = load_readiness_selector(args, self.device)
        self.hidden_hook_module_name = ""
        self.hidden_hook_module: nn.Module | None = None
        if args.hidden_capture_mode != "none":
            self.hidden_hook_module_name, self.hidden_hook_module = find_last_layer_module(self.model)
            if self.hidden_hook_module is None:
                raise RuntimeError("Could not find a Dream transformer layer for hidden capture")

    def current_row_call_ids(self) -> list[str]:
        ids = self._row_call_ids
        self._row_call_ids = []
        return ids

    def chat_trace(self, messages: list[dict[str, str]], context: dict[str, str], args: argparse.Namespace) -> str:
        self.call_index += 1
        call_id = f"call_{self.call_index:06d}"
        self._row_call_ids.append(call_id)
        call_start = time.time()
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        input_ids = inputs.input_ids.to(device=self.device)
        attention_mask = inputs.attention_mask.to(device=self.device)
        input_tokens = int(input_ids.shape[-1])
        if input_tokens + args.max_tokens > args.max_context_tokens:
            raise ValueError(
                f"Dream context exceeds {args.max_context_tokens}: input_tokens={input_tokens}, "
                f"max_tokens={args.max_tokens}"
            )
        trace_state: dict[str, Any] = {
            "step_summaries": [],
            "previous_tokens": None,
            "pending_logits": {},
            "hidden_events": [],
            "pending_hidden_events": [],
        }
        torch.cuda.reset_peak_memory_stats(self.device) if self.device.type == "cuda" else None
        hidden_handle = None
        if self.hidden_hook_module is not None:
            hidden_handle = self.hidden_hook_module.register_forward_hook(
                make_hidden_hook(
                    trace_state,
                    input_tokens,
                    call_id,
                    self.hidden_ref_dir,
                    args,
                )
            )
        with torch.no_grad():
            try:
                output = self.model.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_tokens,
                    output_history=True,
                    return_dict_in_generate=True,
                    steps=args.dream_steps,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    alg=args.alg,
                    alg_temp=args.alg_temp,
                    generation_tokens_hook_func=make_tokens_hook(
                        trace_state,
                        self.tokenizer,
                        input_tokens,
                        self.mask_token_id,
                        args,
                    ),
                    generation_logits_hook_func=make_logits_hook(
                        trace_state,
                        input_tokens,
                        self.mask_token_id,
                        args,
                    ),
                )
            finally:
                if hidden_handle is not None:
                    hidden_handle.remove()
        sequences = getattr(output, "sequences", output)
        generated_ids = sequences[0, input_tokens:].detach().cpu().tolist()
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if args.hidden_capture_mode == "selected_suffix_tensor":
            self.attach_selected_hidden_ref(trace_state, call_id, context, args)
        for event_index, item in enumerate(trace_state["step_summaries"]):
            item["trace_event_index"] = event_index
            item["has_logit_stats"] = "top1_prob_mean" in item
            item["has_hidden_summary"] = "hidden_summary" in item
            item["has_hidden_ref"] = "hidden_ref" in item
        peak_memory = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
        metric = {
            "call_id": call_id,
            "row_id": context.get("row_id", ""),
            "sample_id": context.get("sample_id", ""),
            "condition": context.get("condition", ""),
            "agent_role": context.get("agent_role", ""),
            "agent_id": context.get("agent_id", ""),
            "input_tokens": input_tokens,
            "max_tokens": args.max_tokens,
            "dream_steps": args.dream_steps,
            "num_step_summaries": len(trace_state["step_summaries"]),
            "elapsed_seconds": round(time.time() - call_start, 3),
            "peak_memory_gib": round(peak_memory / 1024**3, 3),
            "output_chars": len(text),
        }
        trace_record = {
            **metric,
            "model": args.model,
            "provider": self.name,
            "trace_version": "p3_dream_step_trace_v1",
            "generation_config": generation_config(args),
            "hidden_capture": {
                "mode": args.hidden_capture_mode,
                "module": self.hidden_hook_module_name,
                "save_dtype": args.hidden_save_dtype,
                "num_hidden_events": len(trace_state["hidden_events"]),
            },
            "step_summaries": trace_state["step_summaries"],
            "final_generated_text": text,
            "offline_teacher_fields_present": ["final_generated_text"],
            "online_feature_boundary": "hidden/logit/process summaries are online features; decoded_probe_text is offline teacher-only",
        }
        append_jsonl(self.trace_path, trace_record)
        append_jsonl(self.call_metrics_path, metric)
        self.call_metrics.append(metric)
        return text

    @torch.no_grad()
    def attach_selected_hidden_ref(
        self,
        trace_state: dict[str, Any],
        call_id: str,
        context: dict[str, str],
        args: argparse.Namespace,
    ) -> None:
        if self.readiness_model is None:
            return
        events = trace_state.get("step_summaries", [])
        if not events:
            return
        device = next(self.readiness_model.parameters()).device
        enriched_events = []
        for event in events:
            enriched = dict(event)
            enriched["condition"] = context.get("condition", "")
            enriched_events.append(enriched)
        features = torch.tensor([event_features(event) for event in enriched_events], dtype=torch.float32, device=device)
        mean = torch.tensor(self.readiness_feature_stats["mean"], dtype=torch.float32, device=device)
        std = torch.tensor(self.readiness_feature_stats["std"], dtype=torch.float32, device=device).clamp_min(1e-6)
        outputs = self.readiness_model(
            ((features - mean) / std).unsqueeze(0),
            torch.zeros((1, features.shape[0]), dtype=torch.bool, device=device),
        )
        event_states = []
        for index, event in enumerate(events):
            event_states.append(
                {
                    "event_index": index,
                    "trace_event_index": index,
                    "step": float(event.get("step") or 0.0),
                    "has_hidden_tensor": torch.is_tensor(event.get("_hidden_tensor")),
                    "ready_prob": float(torch.sigmoid(outputs["ready_logit"])[0, index].detach().cpu().item()),
                    "final_match_prob": float(torch.sigmoid(outputs["final_match_logit"])[0, index].detach().cpu().item()),
                    "prediction_change_prob": float(torch.sigmoid(outputs["prediction_change_logit"])[0, index].detach().cpu().item()),
                    "future_gain_pred": float(outputs["future_gain"][0, index].detach().cpu().item()),
                }
            )
        selected = select_event_state_with_tensor(event_states, self.selected_policy)
        selected_index = int(selected["event_index"])
        for index, event in enumerate(events):
            tensor = event.pop("_hidden_tensor", None)
            if index != selected_index or not torch.is_tensor(tensor):
                continue
            ref_path = self.hidden_ref_dir / f"{call_id}_selected_hidden_event{index:04d}.pt"
            torch.save(
                {
                    "call_id": call_id,
                    "hidden_event_index": index,
                    "input_tokens": event.get("hidden_summary", {}).get("input_tokens", None),
                    "tensor_scope": "generated_suffix",
                    "selection_mode": "d5_policy_selected_suffix_tensor",
                    "tensor": tensor.to(dtype=resolve_torch_dtype(args.hidden_save_dtype)).cpu(),
                },
                ref_path,
            )
            event["hidden_ref"] = str(ref_path)
            event["selected_suffix_tensor_policy"] = {
                "ready_prob": selected["ready_prob"],
                "final_match_prob": selected["final_match_prob"],
                "prediction_change_prob": selected["prediction_change_prob"],
                "future_gain_pred": selected["future_gain_pred"],
            }


def make_tokens_hook(
    trace_state: dict[str, Any],
    tokenizer: Any,
    input_tokens: int,
    mask_token_id: int | None,
    args: argparse.Namespace,
):
    def hook(step: int | None, x: torch.Tensor, logits: torch.Tensor | None) -> torch.Tensor:
        step_int = int(step) if step is not None else len(trace_state["step_summaries"])
        if step_int % max(args.snapshot_stride, 1) != 0 and step_int != args.dream_steps - 1:
            trace_state["previous_tokens"] = x.detach().cpu()
            return x
        suffix = x[0, input_tokens:].detach().cpu()
        previous = trace_state.get("previous_tokens")
        changed = None
        if previous is not None and previous.shape == x.detach().cpu().shape:
            changed = int((suffix != previous[0, input_tokens:]).sum().item())
        record = {
            "step": step_int,
            "token_shape": list(x.shape),
            "suffix_tokens": int(suffix.numel()),
            "num_mask_tokens": int((suffix == mask_token_id).sum().item()) if mask_token_id is not None else None,
            "changed_suffix_tokens_vs_prev_hook": changed,
        }
        if args.decode_step_text:
            record["decoded_probe_text"] = tokenizer.decode(suffix.tolist(), skip_special_tokens=True).strip()
        pending = trace_state.get("pending_logits", {}).pop(step_int, None)
        if pending:
            record.update(pending)
        hidden_event = consume_pending_hidden_event(trace_state)
        if hidden_event:
            record.update(hidden_event)
        trace_state["step_summaries"].append(record)
        trace_state["previous_tokens"] = x.detach().cpu()
        return x

    return hook


def make_logits_hook(
    trace_state: dict[str, Any],
    input_tokens: int,
    mask_token_id: int | None,
    args: argparse.Namespace,
):
    def hook(step: int, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        step_int = int(step)
        suffix_logits = logits[0, input_tokens:, :]
        suffix_tokens = x[0, input_tokens:]
        if mask_token_id is not None:
            mask = suffix_tokens == mask_token_id
            selected = suffix_logits[mask]
            selected_scope = "masked_suffix"
        else:
            selected = suffix_logits.reshape(-1, suffix_logits.shape[-1])
            selected_scope = "suffix"
        if selected.numel() == 0:
            selected = suffix_logits.reshape(-1, suffix_logits.shape[-1])
            selected_scope = "suffix_fallback"
        selected = selected[: args.logit_sample_positions].float()
        probs = torch.softmax(selected, dim=-1)
        top2 = torch.topk(probs, k=2, dim=-1).values
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        stats = {
            "logit_scope": selected_scope,
            "logit_positions_sampled": int(selected.shape[0]),
            "top1_prob_mean": round(float(top2[:, 0].mean().item()), 6),
            "top1_prob_min": round(float(top2[:, 0].min().item()), 6),
            "top2_margin_mean": round(float((top2[:, 0] - top2[:, 1]).mean().item()), 6),
            "entropy_mean": round(float(entropy.mean().item()), 6),
        }
        matching = [item for item in trace_state["step_summaries"] if item.get("step") == step_int]
        if matching:
            matching[-1].update(stats)
        else:
            trace_state.setdefault("pending_logits", {})[step_int] = stats
        return logits

    return hook


def make_hidden_hook(
    trace_state: dict[str, Any],
    input_tokens: int,
    call_id: str,
    hidden_ref_dir: Path,
    args: argparse.Namespace,
):
    def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            hidden = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(hidden) or hidden.ndim < 3:
            return
        event_index = len(trace_state["hidden_events"])
        suffix = hidden[0, input_tokens:, :].detach()
        if suffix.numel() == 0:
            return
        suffix_float = suffix.float()
        summary = {
            "event_index": event_index,
            "shape": list(suffix.shape),
            "mean": round(float(suffix_float.mean().item()), 6),
            "std": round(float(suffix_float.std(unbiased=False).item()), 6),
            "abs_mean": round(float(suffix_float.abs().mean().item()), 6),
            "l2_mean": round(float(torch.linalg.vector_norm(suffix_float, dim=-1).mean().item()), 6),
            "last_token_l2": round(float(torch.linalg.vector_norm(suffix_float[-1], dim=-1).item()), 6),
        }
        hidden_event: dict[str, Any] = {"hidden_summary": summary}
        if args.hidden_capture_mode == "suffix_tensor":
            ref_path = hidden_ref_dir / f"{call_id}_hidden_event{event_index:04d}.pt"
            torch.save(
                {
                    "call_id": call_id,
                    "hidden_event_index": event_index,
                    "input_tokens": input_tokens,
                    "tensor_scope": "generated_suffix",
                    "tensor": suffix.to(dtype=resolve_torch_dtype(args.hidden_save_dtype)).cpu(),
                },
                ref_path,
            )
            hidden_event["hidden_ref"] = str(ref_path)
        elif args.hidden_capture_mode == "selected_suffix_tensor":
            hidden_event["_hidden_tensor"] = suffix.to(dtype=resolve_torch_dtype(args.hidden_save_dtype)).cpu()
        trace_state["hidden_events"].append(hidden_event)
        trace_state.setdefault("pending_hidden_events", []).append(hidden_event)

    return hook


def load_readiness_selector(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DreamStepReadinessStudent, dict[str, list[float]], dict[str, float]]:
    checkpoint = torch.load(args.readiness_checkpoint, map_location="cpu")
    train_config = TrainConfig(**checkpoint["config"])
    model = DreamStepReadinessStudent(
        feature_dim=len(FEATURE_NAMES),
        d_model=train_config.d_model,
        num_layers=train_config.num_layers,
        num_heads=train_config.num_heads,
        dropout=train_config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    policy_summary = json.loads((Path(args.policy_eval_dir) / "summary.json").read_text(encoding="utf-8"))
    return model, checkpoint["feature_stats"], policy_summary["selected_policy"]


def select_event_state_with_tensor(events: list[dict[str, Any]], policy: dict[str, float]) -> dict[str, Any]:
    for event in events:
        if (
            event["ready_prob"] >= policy["ready_threshold"]
            and event["final_match_prob"] >= policy["final_match_threshold"]
            and event["prediction_change_prob"] <= policy["prediction_change_max"]
            and event["future_gain_pred"] <= policy["future_gain_max"]
            and event["has_hidden_tensor"]
        ):
            return event
    for event in reversed(events):
        if event["has_hidden_tensor"]:
            return event
    return events[-1]


def consume_pending_hidden_event(trace_state: dict[str, Any]) -> dict[str, Any] | None:
    pending = trace_state.get("pending_hidden_events", [])
    if not pending:
        return None
    hidden_event = pending[-1]
    pending.clear()
    return hidden_event


def find_last_layer_module(model: Any) -> tuple[str, nn.Module | None]:
    candidates = [
        "model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "backbone.layers",
        "layers",
    ]
    for path in candidates:
        module: Any = model
        ok = True
        for part in path.split("."):
            if not hasattr(module, part):
                ok = False
                break
            module = getattr(module, part)
        if ok and isinstance(module, (nn.ModuleList, list)) and len(module) > 0:
            return f"{path}.{len(module) - 1}", module[-1]
    for name, module in reversed(list(model.named_modules())):
        if "layers." in name or ".h." in name:
            return name, module
    return "", None


def default_output_dir(created_at: int) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(created_at))
    return Path(DEFAULT_OUTPUT_ROOT) / f"musique_calibration_dream_instruct_step_trace_{stamp}"


def resolve_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    return resolve_torch_dtype(name)


def resolve_torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def resolve_mask_token_id(model: Any, tokenizer: Any) -> int | None:
    for value in (
        getattr(getattr(model, "generation_config", None), "mask_token_id", None),
        getattr(getattr(model, "config", None), "mask_token_id", None),
        getattr(tokenizer, "mask_token_id", None),
    ):
        if value is not None:
            return int(value)
    return None


def filter_by_sample_count(rows: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    if max_samples <= 0:
        return rows
    selected_ids = []
    seen = set()
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if sample_id not in seen:
            selected_ids.append(sample_id)
            seen.add(sample_id)
        if len(selected_ids) >= max_samples:
            break
    allowed = set(selected_ids)
    return [row for row in rows if str(row.get("sample_id", "")) in allowed]


def generation_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_tokens": args.max_tokens,
        "dream_steps": args.dream_steps,
        "snapshot_stride": args.snapshot_stride,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "alg": args.alg,
        "alg_temp": args.alg_temp,
        "max_context_tokens": args.max_context_tokens,
        "prediction_extraction_mode": args.prediction_extraction_mode,
        "hidden_capture_mode": args.hidden_capture_mode,
        "hidden_save_dtype": args.hidden_save_dtype,
    }


def error_result(
    row: dict[str, Any],
    args: argparse.Namespace,
    provider: DreamTraceProvider,
    exc: Exception,
    row_index: int,
) -> dict[str, Any]:
    return {
        "row_id": row.get("row_id", ""),
        "sample_id": row.get("sample_id", ""),
        "task_name": row.get("task_name", ""),
        "split": row.get("split", ""),
        "condition": row.get("condition", ""),
        "model": args.model,
        "provider": provider.name,
        "row_index": row_index,
        "status": "error",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "trace_call_ids": provider.current_row_call_ids(),
        "primary_score": 0.0,
        "token_f1": 0.0,
        "exact_match": 0.0,
    }


def write_outputs(
    args: argparse.Namespace,
    output_dir: Path,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    provider: DreamTraceProvider,
    created_at: int,
) -> dict[str, Any]:
    metrics = {
        "num_rows": len(rows),
        "num_generations": len(generations),
        "num_errors": len([row for row in generations if row.get("status") == "error"]),
        "num_trace_calls": provider.call_index,
        "mean_primary_score": mean([float(row.get("primary_score", 0.0)) for row in generations]),
        "mean_token_f1": mean([float(row.get("token_f1", 0.0)) for row in generations]),
        "max_input_tokens": max([int(item.get("input_tokens", 0)) for item in provider.call_metrics] or [0]),
        "max_peak_memory_gib": max([float(item.get("peak_memory_gib", 0.0)) for item in provider.call_metrics] or [0.0]),
        "num_hidden_refs": len(list(provider.hidden_ref_dir.glob("*.pt"))) if provider.hidden_ref_dir.exists() else 0,
    }
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "created_at": created_at,
        "status": "pass" if not errors else "warn",
        "model": args.model,
        "model_path": args.model_path,
        "provider": provider.name,
        "manifest_json": args.manifest_json,
        "online_inputs_jsonl": args.online_inputs_jsonl,
        "num_manifest_samples": len(manifest.get("samples", [])),
        "num_rows": len(rows),
        "num_errors": metrics["num_errors"],
        "generation_config": generation_config(args),
        "generations_jsonl": str(output_dir / "generations.jsonl"),
        "traces_jsonl": str(provider.trace_path),
        "dream_trace_call_metrics_jsonl": str(provider.call_metrics_path),
        "hidden_ref_dir": str(provider.hidden_ref_dir) if provider.hidden_ref_dir.exists() else "",
        "metrics_jsonl": str(metrics_path),
        "metrics": metrics,
        "execution_boundary": [
            "local-only P3 Dream D3 trace collection",
            "model inference only",
            "does not run optimizer or backward",
            "does not create SwanLab runs",
            "decoded/scorer fields are offline teacher/eval only",
        ],
        "forbidden_online_fields": [
            "gold_answer",
            "answer_aliases",
            "score",
            "primary_score",
            "exact_match",
            "token_f1",
            "final_generated_text",
            "decoded_probe_text",
        ],
    }
    write_jsonl(output_dir / "selected_rows.jsonl", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary["summary_json"] = str(output_dir / "summary.json")
    return summary


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
