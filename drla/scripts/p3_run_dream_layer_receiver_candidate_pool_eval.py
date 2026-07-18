"""Run D7.8 matched-channel candidate-pool diagnostics for Dream receivers.

This local-only evaluator samples multiple final-answer candidates from each
online channel and reports both online-selectable metrics (first candidate,
majority vote) and an offline oracle ceiling. It never inserts decoded Agent
text into the solver prompt and never trains a model.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drla.evaluation.p2_phase_c_scorers import normalize_qa_text, score_qa_answer  # noqa: E402
from drla.scripts.p3_run_dream_layer_receiver_eval import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT_DIR,
    build_packets,
    conditioned_diffusion_generate,
)
from drla.scripts.p3_train_dream_layer_conditioned_receiver import (  # noqa: E402
    DreamLayerConditionedReceiver,
    LayerReceiverConfig,
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
    parser.add_argument(
        "--output-dir",
        default=str(Path(DEFAULT_OUTPUT_DIR).with_name("dream_layer_receiver_candidate_pool_eval_20260617")),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--dream-steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--alg", default="entropy")
    parser.add_argument("--alg-temp", type=float, default=0.0)
    parser.add_argument("--prediction-extraction-mode", choices=["default", "first_segment"], default="first_segment")
    parser.add_argument(
        "--conditions",
        default="no_message,layer_receiver_matched,layer_receiver_shuffled_row,layer_receiver_agent_swap,layer_receiver_zero",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_candidates < 1:
        raise ValueError("--num-candidates must be >= 1")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not (args.overwrite or args.resume):
        raise FileExistsError(f"output_dir is not empty; pass --overwrite or --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    generations_path = output_dir / "generations.jsonl"
    if args.overwrite:
        generations_path.write_text("", encoding="utf-8")
    elif not generations_path.exists():
        generations_path.write_text("", encoding="utf-8")

    manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
    samples = {str(sample["sample_id"]): sample for sample in manifest.get("samples", [])}
    all_online_rows = [
        row
        for row in read_jsonl(Path(args.online_inputs_jsonl))
        if row.get("condition") == "textmas_matched"
    ]
    selected_online = all_online_rows[args.row_offset :]
    if args.max_rows:
        selected_online = selected_online[: args.max_rows]
    if not selected_online:
        raise ValueError("no textmas_matched rows selected")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = LayerReceiverConfig(**checkpoint["config"])
    rows, _ = load_training_rows(config)
    rows_by_id = {str(row["row_id"]): row for row in rows}

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required when --device starts with cuda")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    dream = AutoModel.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True).to(device).eval()
    receiver = DreamLayerConditionedReceiver(config).to(device).eval()
    receiver.load_state_dict(checkpoint["model_state"])

    generations: list[dict[str, Any]] = []
    if args.resume and generations_path.exists():
        generations = read_jsonl(generations_path)
    completed = {
        (str(item.get("row_id", "")), str(item.get("condition", "")), int(item.get("candidate_index", -1)))
        for item in generations
    }
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    started = time.time()
    for row_index, online_row in enumerate(selected_online, start=1):
        row_id = str(online_row.get("row_id", ""))
        sample = samples.get(str(online_row.get("sample_id", "")))
        train_row = rows_by_id.get(row_id)
        if sample is None or train_row is None:
            raise ValueError(f"missing sample or packet row for row_id={row_id}")
        scoring = sample.get("scoring", {})
        for condition in conditions:
            packets = build_packets(condition, train_row, selected_online, rows_by_id, config, device)
            messages = make_solver_messages(online_row.get("online_input_fields", {}), upstream_messages=[])
            input_ids = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            ).input_ids.to(device)
            for candidate_index in range(args.num_candidates):
                key = (row_id, condition, candidate_index)
                if key in completed:
                    continue
                set_candidate_seed(args.seed, row_index, condition, candidate_index)
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
                text, decode_status = safe_decode(tokenizer, generated)
                prediction = extract_final_answer(text, mode=args.prediction_extraction_mode)
                score = score_qa_answer(
                    prediction,
                    scoring.get("gold_answer", ""),
                    scoring.get("answer_aliases", []) or [],
                ).to_dict()
                record = {
                    "row_index": row_index,
                    "row_id": row_id,
                    "sample_id": online_row.get("sample_id", ""),
                    "condition": condition,
                    "candidate_index": candidate_index,
                    "status": "ok",
                    "decode_status": decode_status,
                    "raw_final_output": text,
                    "prediction": prediction,
                    "normalized_prediction": normalize_qa_text(prediction),
                    "score": score,
                    "primary_score": score["primary_score"],
                    "token_f1": score["token_f1"],
                    "exact_match": score["exact_match"],
                    "input_tokens": int(input_ids.shape[-1]),
                    "prefix_tokens": 0,
                    "max_tokens": args.max_tokens,
                    "dream_steps": args.dream_steps,
                    "elapsed_seconds": round(time.time() - call_start, 3),
                    "peak_memory_gib": round(
                        (torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0) / 1024**3,
                        3,
                    ),
                }
                append_jsonl(generations_path, record)
                generations.append(record)
                completed.add(key)

    metrics = aggregate_candidate_pool(generations)
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
        "num_rows": len(selected_online),
        "num_candidates": args.num_candidates,
        "num_generations": len(generations),
        "resumed": bool(args.resume),
        "metrics": metrics,
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "metrics_jsonl": str(output_dir / "metrics.jsonl"),
            "generations_jsonl": str(generations_path),
        },
        "elapsed_seconds": round(time.time() - started, 3),
        "execution_boundary": [
            "local-only P3 D7.8 layer-receiver candidate-pool evaluation",
            "no optimizer, backward, or weight update",
            "no SwanLab run",
            "no agent decoded text messages inserted into solver prompt",
            "first and majority selectors use only online-visible generated candidate text",
            "oracle metrics use gold/scorer only offline to estimate candidate-source ceiling",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def aggregate_candidate_pool(generations: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in generations:
        by_group[(str(item["condition"]), str(item["row_id"]))].append(item)

    condition_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    for (condition, _row_id), items in sorted(by_group.items()):
        items = sorted(items, key=lambda item: int(item["candidate_index"]))
        first = items[0]
        majority = select_majority(items)
        oracle = max(items, key=lambda item: (float(item["primary_score"]), float(item["token_f1"])))
        unique_predictions = len({str(item.get("normalized_prediction", "")) for item in items})
        condition_rows[condition].append(
            {
                "first_primary": float(first["primary_score"]),
                "first_token_f1": float(first["token_f1"]),
                "majority_primary": float(majority["primary_score"]),
                "majority_token_f1": float(majority["token_f1"]),
                "oracle_primary": float(oracle["primary_score"]),
                "oracle_token_f1": float(oracle["token_f1"]),
                "unique_predictions": float(unique_predictions),
                "candidate_count": float(len(items)),
                "oracle_candidate_index": float(oracle["candidate_index"]),
                "majority_count": float(
                    Counter(str(item.get("normalized_prediction", "")) for item in items).most_common(1)[0][1]
                ),
            }
        )
    return {condition: mean_metrics(rows) for condition, rows in sorted(condition_rows.items())}


def safe_decode(tokenizer: Any, token_ids: list[int]) -> tuple[str, str]:
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True).strip(), "normal"
    except TypeError:
        tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
        if isinstance(tokens, str):
            tokens = [tokens]
        safe_tokens = [token for token in tokens if isinstance(token, str)]
        try:
            return tokenizer.convert_tokens_to_string(safe_tokens).strip(), "filtered_none_tokens"
        except TypeError:
            return "".join(safe_tokens).strip(), "joined_filtered_none_tokens"


def select_majority(items: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(item.get("normalized_prediction", "")) for item in items)
    best_count = max(counts.values())
    for item in items:
        if counts[str(item.get("normalized_prediction", ""))] == best_count:
            return item
    return items[0]


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(rows[0].keys())
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in keys} | {"num_rows": float(len(rows))}


def set_candidate_seed(seed: int, row_index: int, condition: str, candidate_index: int) -> None:
    condition_hash = sum((idx + 1) * ord(char) for idx, char in enumerate(condition)) % 100000
    value = seed + row_index * 1009 + candidate_index * 9176 + condition_hash
    random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


if __name__ == "__main__":
    main()
