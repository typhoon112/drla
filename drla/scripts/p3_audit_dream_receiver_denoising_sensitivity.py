"""Audit D7 receiver packet influence during Dream denoising.

This local-only diagnostic compares matched packets against no-message, zero,
shuffled-row, and agent-swap controls on the same intermediate denoising state.
It does not train, update weights, decode agent packets into text, or create a
SwanLab run. The goal is to distinguish teacher-forcing CE success from actual
inference-time packet influence.
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
from drla.scripts.p3_run_dream_latent_prefix_eval import sample_tokens  # noqa: E402
from drla.scripts.p3_run_dream_layer_receiver_eval import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    build_packets,
    load_excluded_sample_ids,
    select_online_rows,
)
from drla.scripts.p3_train_dream_layer_conditioned_receiver import (  # noqa: E402
    DreamLayerConditionedReceiver,
    LayerReceiverConfig,
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


DEFAULT_OUTPUT_DIR = (
    "/data1/luyifei/drla/outputs/p3_dream_receiver_sensitivity_audits/"
    "dream_receiver_denoising_sensitivity_20260618"
)


def main() -> None:
    summary = run_audit(parse_args())
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
    parser.add_argument("--split", choices=["all", "train", "valid", "test"], default="valid")
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument("--exclude-sample-ids", default="")
    parser.add_argument("--exclude-sample-ids-file", default="")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--dream-steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--alg-temp", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument(
        "--conditions",
        default="no_message,layer_receiver_zero,layer_receiver_shuffled_row,layer_receiver_agent_swap",
        help="Controls compared against layer_receiver_matched on the same denoising state.",
    )
    parser.add_argument("--prediction-extraction-mode", choices=["default", "first_segment"], default="first_segment")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    step_metrics_path = output_dir / "step_metrics.jsonl"
    row_metrics_path = output_dir / "row_metrics.jsonl"
    step_metrics_path.write_text("", encoding="utf-8")
    row_metrics_path.write_text("", encoding="utf-8")

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
    mask_token_id = resolve_mask_token_id(dream, tokenizer)
    controls = [item.strip() for item in args.conditions.split(",") if item.strip()]

    started = time.time()
    row_records: list[dict[str, Any]] = []
    step_records: list[dict[str, Any]] = []
    for row_index, online_row in enumerate(selected_online, start=1):
        row_record, row_steps = audit_row(
            args=args,
            row_index=row_index,
            online_row=online_row,
            samples=samples,
            rows_by_id=rows_by_id,
            selected_online=selected_online,
            config=config,
            tokenizer=tokenizer,
            dream=dream,
            receiver=receiver,
            controls=controls,
            mask_token_id=mask_token_id,
            device=device,
        )
        append_jsonl(row_metrics_path, row_record)
        row_records.append(row_record)
        for step_record in row_steps:
            append_jsonl(step_metrics_path, step_record)
            step_records.append(step_record)

    summary = {
        "created_at": int(time.time()),
        "status": "pass",
        "checkpoint": args.checkpoint,
        "output_dir": str(output_dir),
        "manifest_json": args.manifest_json,
        "online_inputs_jsonl": args.online_inputs_jsonl,
        "packet_dir": args.packet_dir,
        "model_path": args.model_path,
        "selection": {
            "split": args.split,
            "row_offset": args.row_offset,
            "max_rows": args.max_rows,
            "num_excluded_sample_ids": len(load_excluded_sample_ids(args)),
        },
        "num_rows": len(row_records),
        "num_step_records": len(step_records),
        "controls": controls,
        "aggregate": aggregate_records(row_records, step_records, controls),
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "metrics_jsonl": str(output_dir / "metrics.jsonl"),
            "row_metrics_jsonl": str(row_metrics_path),
            "step_metrics_jsonl": str(step_metrics_path),
        },
        "elapsed_seconds": round(time.time() - started, 3),
        "execution_boundary": [
            "local-only P3 Dream receiver denoising sensitivity audit",
            "no optimizer, backward, or weight update",
            "no SwanLab run",
            "matched and control packets are compared on the same denoising state",
            "no decoded agent text messages inserted into solver prompt",
            "gold/scorer used only for offline final matched-trajectory evaluation",
        ],
    }
    (output_dir / "metrics.jsonl").write_text(
        json.dumps(summary["aggregate"], ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


@torch.no_grad()
def audit_row(
    *,
    args: argparse.Namespace,
    row_index: int,
    online_row: dict[str, Any],
    samples: dict[str, dict[str, Any]],
    rows_by_id: dict[str, dict[str, Any]],
    selected_online: list[dict[str, Any]],
    config: LayerReceiverConfig,
    tokenizer: Any,
    dream: Any,
    receiver: DreamLayerConditionedReceiver,
    controls: list[str],
    mask_token_id: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row_id = str(online_row.get("row_id", ""))
    sample = samples.get(str(online_row.get("sample_id", "")))
    packet_row = rows_by_id.get(row_id)
    if sample is None or packet_row is None:
        raise ValueError(f"missing sample or packet row for row_id={row_id}")
    packets_by_condition = {
        "layer_receiver_matched": build_packets(
            "layer_receiver_matched", packet_row, selected_online, rows_by_id, config, device
        )
    }
    for control in controls:
        packets_by_condition[control] = build_packets(control, packet_row, selected_online, rows_by_id, config, device)

    messages = make_solver_messages(online_row.get("online_input_fields", {}), upstream_messages=[])
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).input_ids.to(device)
    max_length = input_ids.shape[1] + args.max_tokens
    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
    timesteps = torch.linspace(1, getattr(dream.generation_config, "eps", 1e-3), args.dream_steps + 1, device=device)
    step_records: list[dict[str, Any]] = []

    for step_index in range(args.dream_steps):
        mask_index = x == mask_token_id
        num_masked = int(mask_index.sum().item())
        if num_masked == 0:
            break
        matched_logits = shifted_logits(
            receiver.forward_logits(
                dream,
                x,
                packets_by_condition["layer_receiver_matched"],
                condition_start=input_ids.shape[1],
            )
        )
        matched_mask_logits = matched_logits[mask_index]
        matched_top_logits, matched_top_ids = matched_mask_logits.max(dim=-1)
        torch.manual_seed(args.seed + row_index * 1000 + step_index)
        confidence, sampled_x0 = sample_tokens(
            matched_mask_logits,
            temperature=args.temperature,
            top_p=args.top_p,
            neg_entropy=True,
        )
        t = timesteps[step_index]
        s = timesteps[step_index + 1]
        num_transfer = int(mask_index.sum() / mask_index.shape[0] * (1 - s / t))
        if step_index == args.dream_steps - 1:
            num_transfer = int(mask_index.sum() / mask_index.shape[0])
        transfer_index = transfer_positions(x, mask_index, confidence, num_transfer, args.alg_temp)
        matched_flat_transfer = mask_flat_transfer_mask(mask_index, transfer_index)
        for control in controls:
            control_packets = packets_by_condition[control]
            if control_packets is None:
                control_logits = shifted_logits(dream(x).logits)
            else:
                control_logits = shifted_logits(
                    receiver.forward_logits(dream, x, control_packets, condition_start=input_ids.shape[1])
                )
            control_mask_logits = control_logits[mask_index]
            record = compare_logits(
                row_index=row_index,
                row_id=row_id,
                sample_id=str(online_row.get("sample_id", "")),
                step_index=step_index,
                control=control,
                num_masked=num_masked,
                num_transfer=num_transfer,
                matched_top_ids=matched_top_ids,
                matched_top_logits=matched_top_logits,
                sampled_x0=sampled_x0,
                transfer_mask=matched_flat_transfer,
                control_mask_logits=control_mask_logits,
            )
            step_records.append(record)
        if num_transfer > 0:
            x_new = torch.full_like(x, mask_token_id)
            x_new[mask_index] = sampled_x0.clone()
            row_indices = torch.arange(x.size(0), device=device).unsqueeze(1).expand_as(transfer_index)
            x[row_indices, transfer_index] = x_new[row_indices, transfer_index]

    generated = x[0, input_ids.shape[1] :].detach().cpu().tolist()
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    prediction = extract_final_answer(text, mode=args.prediction_extraction_mode)
    scoring = sample.get("scoring", {})
    score = score_qa_answer(prediction, scoring.get("gold_answer", ""), scoring.get("answer_aliases", []) or []).to_dict()
    row_record = {
        "row_index": row_index,
        "row_id": row_id,
        "sample_id": online_row.get("sample_id", ""),
        "condition": "layer_receiver_matched_shared_state",
        "status": "ok",
        "raw_final_output": text,
        "prediction": prediction,
        "score": score,
        "primary_score": score["primary_score"],
        "token_f1": score["token_f1"],
        "exact_match": score["exact_match"],
        "input_tokens": int(input_ids.shape[-1]),
        "max_tokens": args.max_tokens,
        "dream_steps": args.dream_steps,
        "actual_steps": len({record["step_index"] for record in step_records}),
    }
    return row_record, step_records


def shifted_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def transfer_positions(
    x: torch.Tensor,
    mask_index: torch.Tensor,
    confidence: torch.Tensor,
    num_transfer: int,
    alg_temp: float,
) -> torch.Tensor:
    if num_transfer <= 0:
        return torch.empty((x.size(0), 0), dtype=torch.long, device=x.device)
    full_confidence = torch.full_like(x, -torch.inf, device=x.device, dtype=confidence.dtype)
    full_confidence[mask_index] = confidence
    if alg_temp == 0:
        _, transfer_index = torch.topk(full_confidence, num_transfer)
    else:
        probs = F.softmax(full_confidence / alg_temp, dim=-1)
        transfer_index = torch.multinomial(probs, num_samples=num_transfer)
    return transfer_index


def mask_flat_transfer_mask(mask_index: torch.Tensor, transfer_index: torch.Tensor) -> torch.Tensor:
    flat_mask_positions = torch.nonzero(mask_index[0], as_tuple=False).flatten()
    if transfer_index.numel() == 0:
        return torch.zeros(flat_mask_positions.shape[0], dtype=torch.bool, device=mask_index.device)
    selected_positions = transfer_index[0]
    return (flat_mask_positions[:, None] == selected_positions[None, :]).any(dim=1)


def compare_logits(
    *,
    row_index: int,
    row_id: str,
    sample_id: str,
    step_index: int,
    control: str,
    num_masked: int,
    num_transfer: int,
    matched_top_ids: torch.Tensor,
    matched_top_logits: torch.Tensor,
    sampled_x0: torch.Tensor,
    transfer_mask: torch.Tensor,
    control_mask_logits: torch.Tensor,
) -> dict[str, Any]:
    control_top_logits, control_top_ids = control_mask_logits.max(dim=-1)
    control_on_matched_top = control_mask_logits.gather(-1, matched_top_ids.unsqueeze(-1)).squeeze(-1)
    top1_same = control_top_ids == matched_top_ids
    if transfer_mask.any():
        transfer_same = control_top_ids[transfer_mask] == sampled_x0[transfer_mask]
        transfer_top1_same_rate = float(transfer_same.float().mean().item())
    else:
        transfer_top1_same_rate = 0.0
    return {
        "row_index": row_index,
        "row_id": row_id,
        "sample_id": sample_id,
        "step_index": step_index,
        "control": control,
        "num_masked": num_masked,
        "num_transfer": num_transfer,
        "top1_same_rate": float(top1_same.float().mean().item()),
        "top1_disagree_rate": float((~top1_same).float().mean().item()),
        "transfer_top1_same_rate": transfer_top1_same_rate,
        "transfer_top1_disagree_rate": float(1.0 - transfer_top1_same_rate) if transfer_mask.any() else 0.0,
        "matched_top_logit_mean": float(matched_top_logits.float().mean().item()),
        "control_top_logit_mean": float(control_top_logits.float().mean().item()),
        "matched_top_minus_control_same_token_logit_mean": float(
            (matched_top_logits - control_on_matched_top).float().mean().item()
        ),
        "matched_top_minus_control_top_logit_mean": float(
            (matched_top_logits - control_top_logits).float().mean().item()
        ),
    }


def aggregate_records(
    row_records: list[dict[str, Any]],
    step_records: list[dict[str, Any]],
    controls: list[str],
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "matched_shared_state_primary": mean([float(row["primary_score"]) for row in row_records]),
        "matched_shared_state_exact": mean([float(row["exact_match"]) for row in row_records]),
        "matched_shared_state_token_f1": mean([float(row["token_f1"]) for row in row_records]),
        "num_rows": len(row_records),
        "num_step_records": len(step_records),
    }
    for control in controls:
        records = [record for record in step_records if record["control"] == control]
        aggregate[control] = {
            "num_records": len(records),
            "top1_same_rate_mean": mean([float(record["top1_same_rate"]) for record in records]),
            "top1_disagree_rate_mean": mean([float(record["top1_disagree_rate"]) for record in records]),
            "transfer_top1_same_rate_mean": mean(
                [float(record["transfer_top1_same_rate"]) for record in records if record["num_transfer"] > 0]
            ),
            "transfer_top1_disagree_rate_mean": mean(
                [float(record["transfer_top1_disagree_rate"]) for record in records if record["num_transfer"] > 0]
            ),
            "matched_top_minus_control_same_token_logit_mean": mean(
                [float(record["matched_top_minus_control_same_token_logit_mean"]) for record in records]
            ),
            "matched_top_minus_control_top_logit_mean": mean(
                [float(record["matched_top_minus_control_top_logit_mean"]) for record in records]
            ),
        }
    return aggregate


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
