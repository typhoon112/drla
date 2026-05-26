"""Evaluate official Cola VAE gold-latent reconstruction on GSM8K prompts."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from drla.data.answer_judge import judge
from drla.tracking import finish_experiment, init_experiment, log_metrics

try:
    from cola_dlm import ColaTextVAEModel
except ImportError as exc:  # pragma: no cover - exercised in integration only.
    raise ImportError("Set PYTHONPATH to the official Cola-DLM code directory before running this script.") from exc


@dataclass(frozen=True)
class ColaVAEReconConfig:
    vae_path: str
    tokenizer_path: str
    input_jsonl: str
    summary_json: str
    max_samples: int = 64
    batch_size: int = 2
    device: str = "cuda"
    pad_token_id: int = 100277
    swanlab_mode: str = "cloud"
    experiment_name: str = "cola-vae-gsm8k-reconstruction"


def read_jsonl(path: Path, max_samples: int) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if max_samples > 0 and len(rows) >= max_samples:
                    break
    return rows


def build_gold_text(row: dict[str, Any]) -> tuple[str, str]:
    question = str(row.get("question") or row.get("prompt") or "").rstrip()
    gold = str(row.get("ground_truth") or row.get("answer") or "").strip()
    target_text = str(row.get("target_text") or gold).strip()
    if "Answer:" not in question:
        question = f"Question: {question}\nAnswer:"
    return f"{question} {target_text}".strip(), gold


def answer_part(text: str) -> str:
    if "Answer:" in text:
        return text.rsplit("Answer:", 1)[-1]
    return text


def pad_to_chunk(ids: list[int], *, chunk: int, pad_token_id: int) -> tuple[list[int], int]:
    pad_len = (chunk - len(ids) % chunk) % chunk
    return ids + [pad_token_id] * pad_len, len(ids)


def evaluate_reconstruction(config: ColaVAEReconConfig) -> dict[str, Any]:
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    vae = ColaTextVAEModel.from_pretrained(config.vae_path).to(device)
    vae.eval()

    rows = read_jsonl(Path(config.input_jsonl), config.max_samples)
    chunk = vae.patch_size * vae.block_size

    total_tokens = 0
    correct_tokens = 0
    exact_sequences = 0
    judged = []

    with torch.no_grad():
        for start in range(0, len(rows), config.batch_size):
            batch = rows[start : start + config.batch_size]
            input_ids_list = []
            raw_lengths = []
            raw_ids_list = []
            gold_texts = []
            gold_answers = []
            for row in batch:
                text, gold = build_gold_text(row)
                ids = tokenizer.encode(text).ids
                padded, raw_len = pad_to_chunk(ids, chunk=chunk, pad_token_id=config.pad_token_id)
                input_ids_list.append(torch.tensor(padded, dtype=torch.long, device=device))
                raw_lengths.append(raw_len)
                raw_ids_list.append(ids)
                gold_texts.append(text)
                gold_answers.append(gold)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, logits = vae(input_ids_list)
            pred_ids_flat = logits.argmax(dim=-1).squeeze(0).detach().cpu()

            offset = 0
            for row, raw_len, raw_ids, gold_text, gold_answer in zip(
                batch, raw_lengths, raw_ids_list, gold_texts, gold_answers
            ):
                padded_len = input_ids_list[len(judged) % config.batch_size].numel()
                pred_ids = pred_ids_flat[offset : offset + raw_len].tolist()
                offset += padded_len

                token_matches = sum(int(a == b) for a, b in zip(pred_ids, raw_ids))
                total_tokens += raw_len
                correct_tokens += token_matches
                exact = pred_ids == raw_ids
                exact_sequences += int(exact)

                decoded = tokenizer.decode(pred_ids, skip_special_tokens=False)
                answer_eval = judge(answer_part(decoded), gold_answer)
                judged.append(
                    {
                        "id": row.get("id"),
                        "token_accuracy": token_matches / max(raw_len, 1),
                        "sequence_exact": exact,
                        "correct": answer_eval["correct"],
                        "answer_found": answer_eval["answer_found"],
                        "gold": gold_answer,
                        "gold_text": gold_text,
                        "decoded": decoded,
                        "decoded_answer": answer_part(decoded),
                        "pred_norm": answer_eval["pred_norm"],
                        "gold_norm": answer_eval["gold_norm"],
                    }
                )

    count = len(judged)
    correct = sum(int(item["correct"]) for item in judged)
    answer_found = sum(int(item["answer_found"]) for item in judged)
    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "num_samples": count,
        "token_accuracy": correct_tokens / max(total_tokens, 1),
        "sequence_exact_rate": exact_sequences / max(count, 1),
        "answer_accuracy": correct / max(count, 1),
        "answer_found_rate": answer_found / max(count, 1),
        "correct": correct,
        "answer_found": answer_found,
        "examples": judged[:10],
    }

    summary_path = Path(config.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    run = init_experiment(
        stage="cola-vae-reconstruction",
        experiment_name=config.experiment_name,
        description="Official Cola VAE reconstruction on GSM8K gold QA text.",
        config=asdict(config),
        mode=config.swanlab_mode,
        tags=["cola", "vae", "gsm8k", "reconstruction"],
    )
    try:
        log_metrics(
            {
                "token_accuracy": summary["token_accuracy"],
                "sequence_exact_rate": summary["sequence_exact_rate"],
                "answer_accuracy": summary["answer_accuracy"],
                "answer_found_rate": summary["answer_found_rate"],
                "num_samples": summary["num_samples"],
            },
            prefix="valid",
        )
    finally:
        finish_experiment()
    summary["swanlab_run_id"] = getattr(run, "id", None)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> ColaVAEReconConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--max-samples", type=int, default=ColaVAEReconConfig.max_samples)
    parser.add_argument("--batch-size", type=int, default=ColaVAEReconConfig.batch_size)
    parser.add_argument("--device", default=ColaVAEReconConfig.device)
    parser.add_argument("--pad-token-id", type=int, default=ColaVAEReconConfig.pad_token_id)
    parser.add_argument("--swanlab-mode", default=ColaVAEReconConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=ColaVAEReconConfig.experiment_name)
    args = parser.parse_args()
    return ColaVAEReconConfig(
        vae_path=args.vae_path,
        tokenizer_path=args.tokenizer_path,
        input_jsonl=args.input_jsonl,
        summary_json=args.summary_json,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        device=args.device,
        pad_token_id=args.pad_token_id,
        swanlab_mode=args.swanlab_mode,
        experiment_name=args.experiment_name,
    )


def main() -> None:
    summary = evaluate_reconstruction(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
