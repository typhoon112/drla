"""Collect block-wise Cola rollout traces for readiness / halt research.

This script keeps the official Cola VAE + DiT generation path intact and
adds per-block trace artifacts. The traces are not used as scientific
evidence by themselves; they are the data substrate for oracle readiness
frontiers and later halt/readiness models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import zlib
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from drla.tracking import finish_experiment, init_experiment, log_metrics
from drla.tracking import require_swanlab_disabled_for_non_training


OFFICIAL_COLA_TASKS = [
    "lambada",
    "mmlu",
    "obqa",
    "hellaswag",
    "race",
    "siqa",
    "squad",
    "story_cloze",
]


@dataclass(frozen=True)
class ColaBlockTraceConfig:
    dit_path: str = "/data1/luyifei/drla/models/Cola-DLM/cola_dlm/cola_dit"
    vae_path: str = "/data1/luyifei/drla/models/Cola-DLM/cola_dlm/cola_vae"
    tokenizer_path: str = "/data1/luyifei/drla/models/Cola-DLM/tokenizer.json"
    input_jsonl: str = "/data1/luyifei/Cola-DLM/code/generate_task_data/lambada.jsonl"
    output_dir: str = "/data1/luyifei/drla/outputs/cola_block_traces/tasks_block_trace"
    task_name: str = "lambada"
    batch_size: int = 20
    max_samples: int = 0
    start_index: int = 0
    end_index: int = 0
    max_new_tokens: int = 32
    timestep_num: int = 16
    guidance_scale: float = 7.0
    temperature: float = 0.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    pad_token_id: int = 100277
    eos_token_id: int | None = 100257
    im_end_token_id: int | None = 100265
    seed: int = 20260524
    per_sample_noise_seed: int | None = 66
    device: str = "auto"
    rank: int = 0
    world_size: int = 1
    save_latents: bool = True
    swanlab_mode: str = "disabled"
    experiment_name: str = "official-cola-block-traces"
    is_sft: bool = False
    im_start_token_id: int | None = None
    user_token_id: int | None = None
    assistant_token_id: int | None = None
    newline_token_id: int | None = None


def collect_cola_block_traces(config: ColaBlockTraceConfig) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        config.swanlab_mode,
        script_kind="Cola block trace collection",
    )
    if config.task_name not in OFFICIAL_COLA_TASKS:
        raise ValueError(
            f"task_name must be one of official Cola tasks {OFFICIAL_COLA_TASKS}; "
            f"got {config.task_name!r}"
        )
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.start_index < 0:
        raise ValueError("start_index must be non-negative")
    if config.end_index and config.end_index <= config.start_index:
        raise ValueError("end_index must be 0 or greater than start_index")
    if config.world_size <= 0:
        raise ValueError("world_size must be positive")
    if not 0 <= config.rank < config.world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_rank{config.rank}" if config.world_size > 1 else ""
    trace_path = output_dir / f"{config.task_name}{suffix}_traces.jsonl"
    generation_path = output_dir / f"{config.task_name}{suffix}.jsonl"
    metrics_path = output_dir / f"{config.task_name}{suffix}_metrics.jsonl"
    summary_path = output_dir / f"{config.task_name}{suffix}_summary.json"
    latent_dir = output_dir / "latents" / config.task_name
    if config.save_latents:
        latent_dir.mkdir(parents=True, exist_ok=True)
    config_digest = config_fingerprint(config)

    raw_data = read_jsonl(Path(config.input_jsonl))
    if config.max_samples > 0:
        raw_data = raw_data[: config.max_samples]
    if config.start_index or config.end_index:
        raw_data = raw_data[config.start_index : config.end_index or None]
    shard_data = raw_data[config.rank :: config.world_size]

    set_seed(config.seed)
    device = resolve_device(config.device)
    cola = load_cola_symbols()
    tokenizer = cola["Tokenizer"].from_file(config.tokenizer_path)

    run = None
    if config.swanlab_mode != "disabled":
        run = init_experiment(
            stage="cola-block-trace",
            experiment_name=config.experiment_name,
            description="Official Cola block-wise rollout traces for readiness / halt research.",
            config={**asdict(config), "official_tasks": OFFICIAL_COLA_TASKS},
            mode=config.swanlab_mode,
            tags=["cola", "official-benchmark", "block-trace", "readiness"],
        )

    processed = 0
    trace_rows = 0
    total_blocks = 0
    stop_reasons: dict[str, int] = {}
    start_time = time.time()

    try:
        dit = cola["ColaDiTModel"].from_pretrained(config.dit_path).to(device)
        vae = cola["ColaTextVAEModel"].from_pretrained(config.vae_path).to(device)

        with trace_path.open("w", encoding="utf-8") as trace_f, generation_path.open(
            "w", encoding="utf-8"
        ) as gen_f, metrics_path.open("w", encoding="utf-8") as metrics_f:
            for batch_start in range(0, len(shard_data), config.batch_size):
                batch = shard_data[batch_start : batch_start + config.batch_size]
                batch_index = batch_start // config.batch_size
                result = generate_traced_batch(
                    dit=dit,
                    vae=vae,
                    tokenizer=tokenizer,
                    prompts=batch,
                    config=config,
                    device=device,
                    apply_prompt_template=cola["apply_prompt_template"],
                    sample_with_strategies=cola["sample_with_strategies"],
                    latent_dir=latent_dir if config.save_latents else None,
                    batch_index=batch_index,
                    global_batch_start=config.start_index + batch_start,
                    config_digest=config_digest,
                )

                for row in result["trace_records"]:
                    trace_f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                trace_f.flush()

                for row in result["generations"]:
                    gen_f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                gen_f.flush()

                processed += len(batch)
                trace_rows += len(result["trace_records"])
                total_blocks += sum(row["num_blocks"] for row in result["generations"])
                stop_reason = result["stop_reason"]
                stop_reasons[stop_reason] = stop_reasons.get(stop_reason, 0) + len(batch)

                elapsed = max(time.time() - start_time, 1e-6)
                metrics = {
                    "samples": processed,
                    "trace_rows": trace_rows,
                    "avg_blocks_per_sample": total_blocks / max(processed, 1),
                    "samples_per_second": processed / elapsed,
                }
                if run is not None:
                    log_metrics(
                        metrics,
                        step=processed,
                        prefix="trace",
                    )
                metrics_f.write(
                    json.dumps(
                        {
                            "created_at": int(time.time()),
                            "step": processed,
                            "task": config.task_name,
                            "rank": config.rank,
                            "world_size": config.world_size,
                            "batch_index": batch_index,
                            "config_digest": config_digest,
                            "swanlab_run_id": getattr(run, "id", None) if run is not None else None,
                            "metrics": metrics,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                metrics_f.flush()

        summary = {
            "created_at": int(time.time()),
            "config": asdict(config),
            "config_digest": config_digest,
            "official_tasks": OFFICIAL_COLA_TASKS,
            "num_input_samples": len(raw_data),
            "num_shard_samples": len(shard_data),
            "num_processed_samples": processed,
            "num_trace_rows": trace_rows,
            "avg_blocks_per_sample": total_blocks / max(processed, 1),
            "stop_reasons": stop_reasons,
            "trace_jsonl": str(trace_path),
            "generation_jsonl": str(generation_path),
            "metrics_jsonl": str(metrics_path),
            "latent_dir": str(latent_dir) if config.save_latents else None,
            "swanlab_run_id": getattr(run, "id", None) if run is not None else None,
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary
    finally:
        if run is not None:
            finish_experiment()


@torch.no_grad()
def generate_traced_batch(
    *,
    dit: Any,
    vae: Any,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    config: ColaBlockTraceConfig,
    device: torch.device,
    apply_prompt_template: Callable[..., str],
    sample_with_strategies: Callable[..., torch.Tensor],
    latent_dir: Path | None,
    batch_index: int,
    global_batch_start: int,
    config_digest: str,
) -> dict[str, Any]:
    dit.eval()
    vae.eval()

    scale = vae.scaling_factor
    shift = vae.shifting_factor
    patch_size = int(vae.patch_size)
    block_size = int(dit.block_size)
    chunk = patch_size * block_size
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    batch_prompts_text: list[str] = []
    input_ids_list: list[torch.Tensor] = []
    token_labels_list: list[torch.Tensor] = []
    prompt_len_remainders: list[int] = []

    for item in prompts:
        prompt_str = apply_prompt_template(
            task=config.task_name,
            context=item.get("context", ""),
            question=item.get("question", ""),
            answer=item.get("ground_truth", item.get("answer", "")),
            choices=item.get("choices"),
        )
        batch_prompts_text.append(prompt_str)

        ids = tokenizer.encode(prompt_str).ids
        if config.is_sft and all(
            token_id is not None
            for token_id in [
                config.im_start_token_id,
                config.user_token_id,
                config.assistant_token_id,
                config.newline_token_id,
                config.im_end_token_id,
            ]
        ):
            ids = (
                [config.im_start_token_id, config.user_token_id, config.newline_token_id]
                + ids
                + [
                    config.im_end_token_id,
                    config.newline_token_id,
                    config.im_start_token_id,
                    config.assistant_token_id,
                    config.newline_token_id,
                ]
            )

        if patch_size > 1:
            prompt_len_remainders.append(len(ids) % patch_size)

        p_pad_len = (chunk - len(ids) % chunk) % chunk
        t_labels = [1] * len(ids) + [3] * p_pad_len
        ids = ids + [config.pad_token_id] * p_pad_len
        input_ids_list.append(torch.tensor(ids, dtype=torch.long, device=device))
        token_labels_list.append(torch.tensor(t_labels, dtype=torch.long, device=device))

    batch_size = len(input_ids_list)
    if batch_size == 0:
        return {"generations": [], "trace_records": [], "stop_reason": "empty_batch"}

    with autocast_context(device):
        enc = vae.encode(input_ids_list)
        latents_list = [((lat - shift) * scale).float() for lat in enc.latents_list]

    latent_labels_list: list[torch.Tensor] = []
    for t_labels in token_labels_list:
        n_patches = t_labels.shape[0] // patch_size
        reshaped = t_labels.view(n_patches, patch_size)
        c1 = (reshaped == 1).any(dim=1)
        c2 = (reshaped == 2).any(dim=1)
        lat = torch.full((n_patches,), 3, dtype=torch.long, device=device)
        lat[c2] = 2
        lat[c1] = 1
        latent_labels_list.append(lat)

    prompt_latent_counts = [int((lat == 1).sum().item()) for lat in latent_labels_list]
    prefix_list: list[torch.Tensor] = []
    first_block_latents_list: list[torch.Tensor] = []
    first_block_labels_list: list[torch.Tensor] = []
    first_block_prompt_token_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
    force_complete_prefix_only = False

    for sample_idx in range(batch_size):
        num_ones = prompt_latent_counts[sample_idx]
        lat_total_i = latents_list[sample_idx].shape[0]
        pad_placeholder = latents_list[sample_idx][lat_total_i - block_size : lat_total_i].clone()

        if num_ones % block_size != 0:
            prefix_fill_latents = block_size - (num_ones % block_size)
            if config.max_new_tokens < prefix_fill_latents:
                force_complete_prefix_only = True
            start_idx = (num_ones // block_size) * block_size
            if start_idx + block_size <= lat_total_i:
                block_latents = latents_list[sample_idx][start_idx : start_idx + block_size].clone()
                block_labels = latent_labels_list[sample_idx][start_idx : start_idx + block_size].clone()
                block_labels[block_labels == 3] = 2
                token_start = start_idx * patch_size
                token_end = min(token_start + block_size * patch_size, token_labels_list[sample_idx].shape[0])
                first_block_prompt_token_counts[sample_idx] = (
                    token_labels_list[sample_idx][token_start:token_end] == 1
                ).sum()
                prefix_list.append(latents_list[sample_idx][:start_idx].clone())
                first_block_latents_list.append(block_latents)
                first_block_labels_list.append(block_labels)
            else:
                prefix_list.append(latents_list[sample_idx][:num_ones].clone())
                first_block_latents_list.append(pad_placeholder)
                first_block_labels_list.append(torch.full((block_size,), 2, dtype=torch.long, device=device))
        else:
            prefix_list.append(latents_list[sample_idx][:num_ones].clone())
            first_block_latents_list.append(pad_placeholder)
            first_block_labels_list.append(torch.full((block_size,), 2, dtype=torch.long, device=device))

    timesteps = torch.linspace(1000, 0, config.timestep_num + 1, dtype=torch.float32)
    prefix_lens = [p.shape[0] for p in prefix_list]
    txt_shape_prefix = shape_tensor(prefix_lens, device)

    kv_cache_enabled = False
    try:
        for block in dit.blocks:
            block.set_kv_cache(True)
        vae.set_kv_cache(True)
        kv_cache_enabled = True

        if any(p.shape[0] > 0 for p in prefix_list):
            txt_prefix = torch.cat(prefix_list, dim=0).to(model_dtype)
            ts_prefix = torch.zeros(txt_prefix.shape[0], device=device, dtype=model_dtype)
            with autocast_context(device):
                _ = dit(
                    txt=txt_prefix,
                    txt_shape=txt_shape_prefix,
                    txt_q_shape=txt_shape_prefix,
                    timestep=ts_prefix,
                    update_kv=True,
                    use_kv_cache=True,
                )
            with autocast_context(device):
                _ = vae.decode(
                    z=torch.cat(prefix_list, dim=0),
                    txt_shape=txt_shape_prefix,
                    txt_q_shape=txt_shape_prefix,
                    update_kv=True,
                )

        txt_shape_cum = shape_tensor(prefix_lens, device)
        first_block_latents_flatten = torch.cat(first_block_latents_list, dim=0)
        first_block_labels_flatten = torch.cat(first_block_labels_list, dim=0)
        flat_mask = first_block_labels_flatten == 1
        cfg_scale_first_block = (
            torch.tensor(
                [config.guidance_scale if pl > 0 else 1.0 for pl in prefix_lens],
                device=device,
                dtype=model_dtype,
            )
            .repeat_interleave(block_size)
            .unsqueeze(-1)
        )

        txt_q_shape = shape_tensor([block_size] * batch_size, device)
        context_ids: torch.Tensor | None = None
        eos_status = torch.zeros(batch_size, dtype=torch.bool, device=device)
        trace_records: list[dict[str, Any]] = []
        latent_history: list[torch.Tensor] = []
        previous_latent_blocks: torch.Tensor | None = None
        previous_texts: list[str | None] = [None] * batch_size
        same_text_streaks = [0] * batch_size
        first_stop_blocks: list[int | None] = [None] * batch_size
        stop_reason = "unknown"
        step = 0

        while True:
            txt_shape_cum = txt_shape_cum + block_size
            latent_dim = first_block_latents_flatten.shape[-1]
            if config.per_sample_noise_seed is None:
                txt = torch.randn(batch_size * block_size, latent_dim, device=device)
            else:
                noise_3d = torch.empty(batch_size, block_size, latent_dim, device=device)
                for sample_idx, item in enumerate(prompts):
                    sid = item.get("id", global_batch_start + sample_idx)
                    try:
                        sid_int = int(sid)
                    except (TypeError, ValueError):
                        sid_int = zlib.crc32(str(sid).encode("utf-8")) & 0xFFFFFFFF
                    generator = torch.Generator(device=device)
                    generator.manual_seed(
                        int(config.per_sample_noise_seed) + sid_int * 1_000 + int(step) * 10_000_000
                    )
                    noise_3d[sample_idx] = torch.randn(
                        block_size,
                        latent_dim,
                        device=device,
                        generator=generator,
                    )
                txt = noise_3d.view(batch_size * block_size, latent_dim)
            drift_norms: list[float] = []

            for t_curr, t_next in zip(timesteps[:-1], timesteps[1:]):
                ts_batch = torch.full((txt.shape[0],), float(t_curr), device=device)
                dt = (float(t_curr) - float(t_next)) / 1000.0

                if step == 0:
                    ts_batch[flat_mask] = 0
                    txt[flat_mask] = first_block_latents_flatten[flat_mask]

                txt_model = txt.to(model_dtype)
                ts_model = ts_batch.to(model_dtype)
                with autocast_context(device):
                    drift_cond = dit(
                        txt=txt_model,
                        txt_shape=txt_shape_cum,
                        txt_q_shape=txt_q_shape,
                        timestep=ts_model,
                        update_kv=False,
                        use_kv_cache=True,
                    ).txt_sample
                    drift_uncond = dit(
                        txt=txt_model,
                        txt_shape=txt_q_shape,
                        txt_q_shape=txt_q_shape,
                        timestep=ts_model,
                        update_kv=False,
                        use_kv_cache=False,
                    ).txt_sample

                scale_t = cfg_scale_first_block if step == 0 else config.guidance_scale
                drift = scale_t * (drift_cond - drift_uncond) + drift_uncond
                drift_norms.append(float(drift.float().norm(dim=-1).mean().item()))
                txt_next = txt - drift * dt

                if step == 0:
                    txt_next[flat_mask] = first_block_latents_flatten[flat_mask]
                txt = txt_next

            with autocast_context(device):
                decoded = vae.decode(
                    z=txt,
                    txt_shape=txt_shape_cum,
                    txt_q_shape=txt_q_shape,
                    update_kv=True,
                )
            decoded_logits = decoded.view(batch_size, block_size * patch_size, -1)
            probe_stats = compute_logit_probe_stats(
                decoded_logits=decoded_logits,
                eos_token_id=config.eos_token_id,
                im_end_token_id=config.im_end_token_id,
            )

            one_block_ids = sample_with_strategies(
                decoded_logits,
                generated_ids=context_ids,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
            )
            context_ids = one_block_ids if context_ids is None else torch.cat([context_ids, one_block_ids], dim=1)

            current_latent_blocks = txt.detach().float().view(batch_size, block_size, latent_dim).cpu()
            latent_history.append(current_latent_blocks)
            latent_stats = compute_latent_stats(current_latent_blocks, previous_latent_blocks)
            previous_latent_blocks = current_latent_blocks

            already_stopped = eos_status.detach().cpu().tolist()
            context_ids_cpu = context_ids.detach().cpu()
            block_ids_cpu = one_block_ids.detach().cpu()
            trim_counts = first_block_prompt_token_counts.detach().cpu().tolist()

            for sample_idx, item in enumerate(prompts):
                trim_count = max(0, min(int(trim_counts[sample_idx]), context_ids_cpu.shape[1]))
                decoded_ids = context_ids_cpu[sample_idx, trim_count:].tolist()
                probe_text = tokenizer.decode(decoded_ids, skip_special_tokens=False)
                stripped_text = probe_text.strip()
                previous_text = previous_texts[sample_idx]
                answer_changed = previous_text is not None and stripped_text != previous_text
                if previous_text is not None and stripped_text == previous_text:
                    same_text_streaks[sample_idx] += 1
                else:
                    same_text_streaks[sample_idx] = 1
                previous_texts[sample_idx] = stripped_text

                block_token_ids = block_ids_cpu[sample_idx].tolist()
                contains_eos = token_id_present(block_token_ids, config.eos_token_id)
                contains_im_end = token_id_present(block_token_ids, config.im_end_token_id)
                contains_stop = contains_eos or contains_im_end
                if contains_stop and first_stop_blocks[sample_idx] is None:
                    first_stop_blocks[sample_idx] = step

                trace_records.append(
                    {
                        "trace_version": "cola_block_trace_v1",
                        "task": config.task_name,
                        "sample_id": item.get("id", global_batch_start + sample_idx),
                        "seed": config.seed,
                        "per_sample_noise_seed": config.per_sample_noise_seed,
                        "rank": config.rank,
                        "world_size": config.world_size,
                        "input_jsonl": config.input_jsonl,
                        "config_digest": config_digest,
                        "batch_index": batch_index,
                        "batch_sample_index": sample_idx,
                        "block_index": step,
                        "block_number": step + 1,
                        "max_block_budget": math.ceil(
                            config.max_new_tokens / max(block_size * patch_size, 1)
                        ),
                        "latent_batch_path": None,
                        "latent_batch_sample_index": sample_idx,
                        "latent_batch_block_index": step,
                        "latent_block_shape": [block_size, latent_dim],
                        "latent_norm_mean": latent_stats[sample_idx]["norm_mean"],
                        "latent_norm_std": latent_stats[sample_idx]["norm_std"],
                        "latent_delta_norm": latent_stats[sample_idx]["delta_norm"],
                        "latent_cosine_to_prev": latent_stats[sample_idx]["cosine_to_prev"],
                        "denoise_drift_norm_mean": sum(drift_norms) / max(len(drift_norms), 1),
                        "latest_block_token_ids": block_token_ids,
                        "decode_token_ids_so_far": decoded_ids,
                        "decode_text_so_far": probe_text,
                        "answer_text_nonempty": bool(stripped_text),
                        "answer_changed": answer_changed,
                        "same_text_streak": same_text_streaks[sample_idx],
                        "already_stopped_before_block": bool(already_stopped[sample_idx]),
                        "contains_eos": contains_eos,
                        "contains_im_end": contains_im_end,
                        "contains_stop": contains_stop,
                        "first_stop_block_index": first_stop_blocks[sample_idx],
                        "official_score_if_decodable": None,
                        "future_gain_label": None,
                        **probe_stats[sample_idx],
                    }
                )

            for sample_idx in range(one_block_ids.shape[0]):
                if token_id_present(block_ids_cpu[sample_idx].tolist(), config.eos_token_id):
                    eos_status[sample_idx] = True
                if token_id_present(block_ids_cpu[sample_idx].tolist(), config.im_end_token_id):
                    eos_status[sample_idx] = True

            txt_model = txt.to(model_dtype)
            with autocast_context(device):
                _ = dit(
                    txt=txt_model,
                    txt_shape=txt_shape_cum,
                    txt_q_shape=txt_q_shape,
                    timestep=torch.zeros(txt.shape[0], device=device, dtype=model_dtype),
                    update_kv=True,
                    use_kv_cache=True,
                )

            step += 1
            if eos_status.all():
                stop_reason = "all_stop_tokens"
                break
            if force_complete_prefix_only and step >= 1:
                stop_reason = "force_complete_prefix_only"
                break
            if step * block_size * patch_size >= config.max_new_tokens:
                stop_reason = "max_new_tokens"
                break

        latent_path: Path | None = None
        if latent_dir is not None:
            latent_path = latent_dir / f"{config.task_name}_rank{config.rank}_batch{batch_index:06d}.pt"
            latent_tensor = torch.stack(latent_history, dim=1)
            torch.save(
                {
                    "trace_version": "cola_block_trace_v1",
                    "task": config.task_name,
                    "sample_ids": [item.get("id", global_batch_start + idx) for idx, item in enumerate(prompts)],
                    "latent_blocks": latent_tensor,
                    "latent_block_shape": [block_size, latent_dim],
                    "num_blocks": step,
                    "patch_size": patch_size,
                    "block_size": block_size,
                    "config_digest": config_digest,
                    "config": asdict(config),
                },
                latent_path,
            )
            for row in trace_records:
                if row["batch_index"] == batch_index:
                    row["latent_batch_path"] = str(latent_path)

        context_ids_cpu = context_ids.detach().cpu()
        trim_counts = first_block_prompt_token_counts.detach().cpu().tolist()
        trimmed_ids: list[list[int]] = []
        for sample_idx, trim_count in enumerate(trim_counts):
            trim_count = max(0, min(int(trim_count), context_ids_cpu.shape[1]))
            trimmed_ids.append(context_ids_cpu[sample_idx, trim_count:].tolist())
        generated_texts = tokenizer.decode_batch(trimmed_ids, skip_special_tokens=False)

        generations: list[dict[str, Any]] = []
        for sample_idx, gen_text in enumerate(generated_texts):
            item = prompts[sample_idx]
            row = {
                "id": item.get("id", global_batch_start + sample_idx),
                "task": config.task_name,
                "seed": config.seed,
                "per_sample_noise_seed": config.per_sample_noise_seed,
                "rank": config.rank,
                "world_size": config.world_size,
                "config_digest": config_digest,
                "prompt": batch_prompts_text[sample_idx],
                "generate": gen_text,
                "ground_truth": item.get("answer", item.get("ground_truth", "")),
                "num_blocks": step,
                "stop_reason": stop_reason,
                "first_stop_block_index": first_stop_blocks[sample_idx],
                "latent_batch_path": str(latent_path) if latent_path else None,
                "latent_batch_sample_index": sample_idx,
            }
            if item.get("choices"):
                row["choices"] = item["choices"]
            if patch_size > 1 and sample_idx < len(prompt_len_remainders):
                row["prompt_len_mod_patch_size"] = prompt_len_remainders[sample_idx]
            generations.append(row)

        return {
            "generations": generations,
            "trace_records": trace_records,
            "stop_reason": stop_reason,
        }
    finally:
        if kv_cache_enabled:
            for block in dit.blocks:
                block.set_kv_cache(False)
            vae.set_kv_cache(False)


def compute_logit_probe_stats(
    *,
    decoded_logits: torch.Tensor,
    eos_token_id: int | None,
    im_end_token_id: int | None,
) -> list[dict[str, float | int | None]]:
    logits = torch.nan_to_num(decoded_logits.detach().float(), nan=0.0, posinf=1e4, neginf=-1e4)
    batch_size, block_tokens, vocab_size = logits.shape
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1).mean(dim=1)
    top_prob_mean = probs.max(dim=-1).values.mean(dim=1)

    eos_probs = get_token_probs(probs, eos_token_id)
    im_end_probs = get_token_probs(probs, im_end_token_id)
    stop_ids = [
        token_id
        for token_id in [eos_token_id, im_end_token_id]
        if token_id is not None and 0 <= token_id < vocab_size
    ]

    stop_prob_max = torch.zeros(batch_size, dtype=probs.dtype, device=probs.device)
    if stop_ids:
        stop_prob_tensor = probs[:, :, stop_ids].amax(dim=(1, 2))
        stop_prob_max = stop_prob_tensor
    non_stop_probs = probs.clone()
    for token_id in stop_ids:
        non_stop_probs[:, :, token_id] = -1.0
    top_non_stop_prob = non_stop_probs.amax(dim=(1, 2))
    stop_margin = stop_prob_max - top_non_stop_prob

    rows: list[dict[str, float | int | None]] = []
    for sample_idx in range(batch_size):
        eos_max, eos_pos = token_prob_max_and_position(eos_probs, sample_idx, block_tokens)
        im_end_max, im_end_pos = token_prob_max_and_position(im_end_probs, sample_idx, block_tokens)
        rows.append(
            {
                "token_entropy_mean": float(entropy[sample_idx].item()),
                "token_top_prob_mean": float(top_prob_mean[sample_idx].item()),
                "eos_prob_max": eos_max,
                "eos_prob_argmax_pos": eos_pos,
                "im_end_prob_max": im_end_max,
                "im_end_prob_argmax_pos": im_end_pos,
                "stop_prob_max": float(stop_prob_max[sample_idx].item()) if stop_ids else None,
                "stop_prob_margin_vs_non_stop": float(stop_margin[sample_idx].item()) if stop_ids else None,
            }
        )
    return rows


def compute_latent_stats(
    current_blocks: torch.Tensor,
    previous_blocks: torch.Tensor | None,
) -> list[dict[str, float | None]]:
    norms = torch.linalg.vector_norm(current_blocks, dim=-1)
    rows: list[dict[str, float | None]] = []
    flat_current = current_blocks.reshape(current_blocks.shape[0], -1)
    flat_previous = previous_blocks.reshape(previous_blocks.shape[0], -1) if previous_blocks is not None else None
    for sample_idx in range(current_blocks.shape[0]):
        row: dict[str, float | None] = {
            "norm_mean": float(norms[sample_idx].mean().item()),
            "norm_std": float(norms[sample_idx].std(unbiased=False).item()),
            "delta_norm": None,
            "cosine_to_prev": None,
        }
        if flat_previous is not None:
            delta = flat_current[sample_idx] - flat_previous[sample_idx]
            row["delta_norm"] = float(torch.linalg.vector_norm(delta).item())
            row["cosine_to_prev"] = float(
                F.cosine_similarity(
                    flat_current[sample_idx].unsqueeze(0),
                    flat_previous[sample_idx].unsqueeze(0),
                    dim=1,
                ).item()
            )
        rows.append(row)
    return rows


def get_token_probs(probs: torch.Tensor, token_id: int | None) -> torch.Tensor | None:
    if token_id is None or not 0 <= token_id < probs.shape[-1]:
        return None
    return probs[:, :, token_id]


def token_prob_max_and_position(
    token_probs: torch.Tensor | None,
    sample_idx: int,
    block_tokens: int,
) -> tuple[float | None, int | None]:
    if token_probs is None:
        return None, None
    sample_probs = token_probs[sample_idx]
    value, pos = sample_probs.max(dim=0)
    position = int(pos.item())
    if not 0 <= position < block_tokens:
        position = None
    return float(value.item()), position


def token_id_present(token_ids: list[int], token_id: int | None) -> bool:
    return token_id is not None and token_id in token_ids


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"input_jsonl does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def shape_tensor(lens: list[int], device: torch.device) -> torch.LongTensor:
    return torch.tensor([[int(length)] for length in lens], dtype=torch.long, device=device)


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_fingerprint(config: ColaBlockTraceConfig) -> str:
    payload = json.dumps(asdict(config), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def load_cola_symbols() -> dict[str, Any]:
    try:
        from tokenizers import Tokenizer

        from cola_dlm import ColaDiTModel, ColaTextVAEModel
        from cola_dlm.inference import apply_prompt_template, sample_with_strategies
    except ImportError as exc:
        raise ImportError(
            "Official Cola modules are required. Run with "
            "PYTHONPATH=/data1/luyifei/Cola-DLM/code or install Cola-DLM in the active environment."
        ) from exc
    return {
        "Tokenizer": Tokenizer,
        "ColaDiTModel": ColaDiTModel,
        "ColaTextVAEModel": ColaTextVAEModel,
        "apply_prompt_template": apply_prompt_template,
        "sample_with_strategies": sample_with_strategies,
    }


def parse_args() -> ColaBlockTraceConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit-path", default=ColaBlockTraceConfig.dit_path)
    parser.add_argument("--vae-path", default=ColaBlockTraceConfig.vae_path)
    parser.add_argument("--tokenizer-path", default=ColaBlockTraceConfig.tokenizer_path)
    parser.add_argument("--input-jsonl", default=ColaBlockTraceConfig.input_jsonl)
    parser.add_argument("--output-dir", default=ColaBlockTraceConfig.output_dir)
    parser.add_argument("--task-name", default=ColaBlockTraceConfig.task_name, choices=OFFICIAL_COLA_TASKS)
    parser.add_argument("--batch-size", type=int, default=ColaBlockTraceConfig.batch_size)
    parser.add_argument("--max-samples", type=int, default=ColaBlockTraceConfig.max_samples)
    parser.add_argument("--start-index", type=int, default=ColaBlockTraceConfig.start_index)
    parser.add_argument("--end-index", type=int, default=ColaBlockTraceConfig.end_index)
    parser.add_argument("--max-new-tokens", type=int, default=ColaBlockTraceConfig.max_new_tokens)
    parser.add_argument("--timestep-num", type=int, default=ColaBlockTraceConfig.timestep_num)
    parser.add_argument("--guidance-scale", type=float, default=ColaBlockTraceConfig.guidance_scale)
    parser.add_argument("--temperature", type=float, default=ColaBlockTraceConfig.temperature)
    parser.add_argument("--top-k", type=int, default=ColaBlockTraceConfig.top_k)
    parser.add_argument("--top-p", type=float, default=ColaBlockTraceConfig.top_p)
    parser.add_argument("--repetition-penalty", type=float, default=ColaBlockTraceConfig.repetition_penalty)
    parser.add_argument("--pad-token-id", type=int, default=ColaBlockTraceConfig.pad_token_id)
    parser.add_argument("--eos-token-id", type=int, default=ColaBlockTraceConfig.eos_token_id)
    parser.add_argument("--im-end-token-id", type=int, default=ColaBlockTraceConfig.im_end_token_id)
    parser.add_argument("--seed", type=int, default=ColaBlockTraceConfig.seed)
    parser.add_argument("--per-sample-noise-seed", type=int, default=ColaBlockTraceConfig.per_sample_noise_seed)
    parser.add_argument("--disable-per-sample-noise-seed", action="store_true")
    parser.add_argument("--device", default=ColaBlockTraceConfig.device)
    parser.add_argument("--rank", type=int, default=ColaBlockTraceConfig.rank)
    parser.add_argument("--world-size", type=int, default=ColaBlockTraceConfig.world_size)
    parser.add_argument("--no-save-latents", action="store_true")
    parser.add_argument("--swanlab-mode", default=ColaBlockTraceConfig.swanlab_mode)
    parser.add_argument("--experiment-name", default=ColaBlockTraceConfig.experiment_name)
    parser.add_argument("--is-sft", action="store_true")
    parser.add_argument("--im-start-token-id", type=int, default=ColaBlockTraceConfig.im_start_token_id)
    parser.add_argument("--user-token-id", type=int, default=ColaBlockTraceConfig.user_token_id)
    parser.add_argument("--assistant-token-id", type=int, default=ColaBlockTraceConfig.assistant_token_id)
    parser.add_argument("--newline-token-id", type=int, default=ColaBlockTraceConfig.newline_token_id)
    args = parser.parse_args()
    return ColaBlockTraceConfig(
        dit_path=args.dit_path,
        vae_path=args.vae_path,
        tokenizer_path=args.tokenizer_path,
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        task_name=args.task_name,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        start_index=args.start_index,
        end_index=args.end_index,
        max_new_tokens=args.max_new_tokens,
        timestep_num=args.timestep_num,
        guidance_scale=args.guidance_scale,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=args.pad_token_id,
        eos_token_id=args.eos_token_id,
        im_end_token_id=args.im_end_token_id,
        seed=args.seed,
        per_sample_noise_seed=None if args.disable_per_sample_noise_seed else args.per_sample_noise_seed,
        device=args.device,
        rank=args.rank,
        world_size=args.world_size,
        save_latents=not args.no_save_latents,
        swanlab_mode=args.swanlab_mode,
        experiment_name=args.experiment_name,
        is_sft=args.is_sft,
        im_start_token_id=args.im_start_token_id,
        user_token_id=args.user_token_id,
        assistant_token_id=args.assistant_token_id,
        newline_token_id=args.newline_token_id,
    )


def main() -> None:
    summary = collect_cola_block_traces(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
