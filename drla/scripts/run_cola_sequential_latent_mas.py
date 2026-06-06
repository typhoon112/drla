"""Run P2-D sequential Cola latent communication smoke/control experiments.

This script is local-only: it performs generation/evaluation-style replay, not
training.  Agent B re-encodes the shared task context, consumes/replays Agent A
latent packet blocks into the official Cola DiT/VAE KV caches, and continues
generation for the remaining block budget.  It writes one official-scorer
compatible ``tasks_<control>/<task>.jsonl`` directory per control.

The runner is intentionally explicit about its current limitation: replaying
latent blocks updates the VAE decode cache and reconstructs token context for
continuation.  This is a P2-D replay diagnostic, not yet a heterogeneous or
decoder-free inter-agent deployment claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
import zlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from drla.scripts.audit_cola_agent_latent_packet_distribution import (
    DEFAULT_CONTROL_TYPES,
    ShardCache,
    build_control_blocks,
    build_packet_indexes,
    load_packet_blocks,
    normalize_control_types,
)
from drla.scripts.collect_cola_block_traces import (
    ColaBlockTraceConfig,
    autocast_context,
    compute_logit_probe_stats,
    load_cola_symbols,
    read_jsonl,
    resolve_device,
    set_seed,
    shape_tensor,
    token_id_present,
)
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
class SequentialLatentMasConfig:
    packets_jsonl: str = (
        "/data1/luyifei/drla/outputs/cola_agent_latent_comm/"
        "p2_agent_latent_comm_v2_locked_seed66_67_68_split20260601_20260529/"
        "agent_latent_comm_packets_test.jsonl"
    )
    output_dir: str = (
        "/data1/luyifei/drla/outputs/cola_sequential_latent_mas/"
        "p2d_sequential_latent_mas"
    )
    dit_path: str = ColaBlockTraceConfig.dit_path
    vae_path: str = ColaBlockTraceConfig.vae_path
    tokenizer_path: str = ColaBlockTraceConfig.tokenizer_path
    cola_code_path: str = "/data1/luyifei/Cola-DLM/code"
    data_root: str = "/data1/luyifei/Cola-DLM/code/generate_task_data"
    control_types: str = "matched,metadata_only,shuffle,cross_task,wrong_block,noise,rotation"
    max_packets: int = 0
    max_packets_per_task: int = 0
    seed: int = 20260529
    max_new_tokens: int = 64
    timestep_num: int = 16
    guidance_scale: float = 7.0
    temperature: float = 0.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    pad_token_id: int = 100277
    eos_token_id: int | None = 100257
    im_end_token_id: int | None = 100265
    per_sample_noise_seed: int | None = 66
    receiver_budget_mode: str = "remaining"
    fixed_receiver_blocks: int = 1
    receiver_context_mode: str = "full_prompt"
    noise_std: float = 1.0
    max_cached_shards: int = 1024
    device: str = "auto"
    swanlab_mode: str = "disabled"


@dataclass
class PromptState:
    prompt_text: str
    prefix: torch.Tensor
    prefix_len: int
    first_block_latents: torch.Tensor
    first_block_labels: torch.Tensor
    first_block_prompt_token_count: int
    block_size: int
    patch_size: int
    latent_dim: int


def main() -> None:
    summary = run_sequential_latent_mas(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> SequentialLatentMasConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets-jsonl", default=SequentialLatentMasConfig.packets_jsonl)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dit-path", default=SequentialLatentMasConfig.dit_path)
    parser.add_argument("--vae-path", default=SequentialLatentMasConfig.vae_path)
    parser.add_argument("--tokenizer-path", default=SequentialLatentMasConfig.tokenizer_path)
    parser.add_argument("--cola-code-path", default=SequentialLatentMasConfig.cola_code_path)
    parser.add_argument("--data-root", default=SequentialLatentMasConfig.data_root)
    parser.add_argument("--control-types", default=SequentialLatentMasConfig.control_types)
    parser.add_argument("--max-packets", type=int, default=0)
    parser.add_argument("--max-packets-per-task", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SequentialLatentMasConfig.seed)
    parser.add_argument("--max-new-tokens", type=int, default=SequentialLatentMasConfig.max_new_tokens)
    parser.add_argument("--timestep-num", type=int, default=SequentialLatentMasConfig.timestep_num)
    parser.add_argument("--guidance-scale", type=float, default=SequentialLatentMasConfig.guidance_scale)
    parser.add_argument("--temperature", type=float, default=SequentialLatentMasConfig.temperature)
    parser.add_argument("--top-k", type=int, default=SequentialLatentMasConfig.top_k)
    parser.add_argument("--top-p", type=float, default=SequentialLatentMasConfig.top_p)
    parser.add_argument("--repetition-penalty", type=float, default=SequentialLatentMasConfig.repetition_penalty)
    parser.add_argument("--pad-token-id", type=int, default=SequentialLatentMasConfig.pad_token_id)
    parser.add_argument("--eos-token-id", type=int, default=SequentialLatentMasConfig.eos_token_id)
    parser.add_argument("--im-end-token-id", type=int, default=SequentialLatentMasConfig.im_end_token_id)
    parser.add_argument("--per-sample-noise-seed", type=int, default=SequentialLatentMasConfig.per_sample_noise_seed)
    parser.add_argument("--disable-per-sample-noise-seed", action="store_true")
    parser.add_argument(
        "--receiver-budget-mode",
        choices=["remaining", "fixed"],
        default=SequentialLatentMasConfig.receiver_budget_mode,
    )
    parser.add_argument("--fixed-receiver-blocks", type=int, default=SequentialLatentMasConfig.fixed_receiver_blocks)
    parser.add_argument(
        "--receiver-context-mode",
        choices=["full_prompt", "empty_prompt"],
        default=SequentialLatentMasConfig.receiver_context_mode,
    )
    parser.add_argument("--noise-std", type=float, default=SequentialLatentMasConfig.noise_std)
    parser.add_argument("--max-cached-shards", type=int, default=SequentialLatentMasConfig.max_cached_shards)
    parser.add_argument("--device", default=SequentialLatentMasConfig.device)
    parser.add_argument("--swanlab-mode", default=SequentialLatentMasConfig.swanlab_mode)
    args = parser.parse_args()
    if args.max_packets < 0 or args.max_packets_per_task < 0:
        raise ValueError("packet limits must be non-negative")
    if args.fixed_receiver_blocks < 0:
        raise ValueError("fixed_receiver_blocks must be non-negative")
    return SequentialLatentMasConfig(
        packets_jsonl=args.packets_jsonl,
        output_dir=args.output_dir,
        dit_path=args.dit_path,
        vae_path=args.vae_path,
        tokenizer_path=args.tokenizer_path,
        cola_code_path=args.cola_code_path,
        data_root=args.data_root,
        control_types=args.control_types,
        max_packets=args.max_packets,
        max_packets_per_task=args.max_packets_per_task,
        seed=args.seed,
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
        per_sample_noise_seed=None if args.disable_per_sample_noise_seed else args.per_sample_noise_seed,
        receiver_budget_mode=args.receiver_budget_mode,
        fixed_receiver_blocks=args.fixed_receiver_blocks,
        receiver_context_mode=args.receiver_context_mode,
        noise_std=args.noise_std,
        max_cached_shards=args.max_cached_shards,
        device=args.device,
        swanlab_mode=args.swanlab_mode,
    )


def run_sequential_latent_mas(config: SequentialLatentMasConfig) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        config.swanlab_mode,
        script_kind="P2-D sequential latent communication runner",
    )
    set_seed(config.seed)
    rng = random.Random(config.seed)
    torch_generator = torch.Generator(device="cpu").manual_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    comparison_path = output_dir / "control_comparison.csv"
    generations_path = output_dir / "generations.jsonl"
    summary_path = output_dir / "summary.json"

    add_cola_code_path(config.cola_code_path)
    packets_all = load_packets(Path(config.packets_jsonl))
    packets = select_packets(packets_all, config, rng)
    if not packets:
        raise ValueError("no packets selected")
    control_types = normalize_control_types(config.control_types)
    packet_indexes = build_packet_indexes(packets_all)
    data_cache = TaskDataCache(Path(config.data_root))
    sample_id_cache = DecisionSampleIdCache()
    shard_cache = ShardCache(config.max_cached_shards)
    rotation_mats: dict[tuple[int, int], torch.Tensor] = {}
    control_generation_warnings: list[dict[str, Any]] = []

    device = resolve_device(config.device)
    cola = load_cola_symbols()
    tokenizer = cola["Tokenizer"].from_file(config.tokenizer_path)
    dit = cola["ColaDiTModel"].from_pretrained(config.dit_path).to(device)
    vae = cola["ColaTextVAEModel"].from_pretrained(config.vae_path).to(device)
    dit.eval()
    vae.eval()

    rows: list[dict[str, Any]] = []
    by_control: dict[str, dict[str, Any]] = defaultdict(init_control_bucket)
    start_time = time.time()

    try:
        with generations_path.open("w", encoding="utf-8") as all_gen_f:
            for packet_index, packet in enumerate(packets):
                task = str(packet["task"])
                sample_id = sample_id_cache.resolve(packet)
                raw_item = data_cache.get(task, sample_id)
                matched_blocks = load_packet_blocks(packet, shard_cache)
                matched_text = None
                for control_type in control_types:
                    if control_type == "matched":
                        replay_blocks = matched_blocks
                        source_packet = packet
                        warning = None
                    else:
                        replay_blocks, source_packet, warning = build_control_blocks(
                            control_type=control_type,
                            packet_index=packet_index,
                            packet=packet,
                            packets=packets_all,
                            packet_indexes=packet_indexes,
                            matched_blocks=matched_blocks,
                            shard_cache=shard_cache,
                            rng=rng,
                            torch_generator=torch_generator,
                            noise_std=config.noise_std,
                            rotation_mats=rotation_mats,
                        )
                    if warning is not None and len(control_generation_warnings) < 100:
                        control_generation_warnings.append(warning)
                    result = run_one_control(
                        packet=packet,
                        sample_id=sample_id,
                        raw_item=raw_item,
                        replay_blocks=replay_blocks,
                        control_type=control_type,
                        source_packet=source_packet,
                        dit=dit,
                        vae=vae,
                        tokenizer=tokenizer,
                        apply_prompt_template=cola["apply_prompt_template"],
                        sample_with_strategies=cola["sample_with_strategies"],
                        config=config,
                        device=device,
                    )
                    if control_type == "matched":
                        matched_text = result["generate"]
                    result["matched_generate_reference"] = matched_text if matched_text is not None else ""
                    result["agreement_with_matched"] = (
                        normalize_text(result["generate"]) == normalize_text(matched_text)
                        if matched_text is not None
                        else True
                    )
                    rows.append(result)
                    all_gen_f.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                    write_task_generation(output_dir, control_type, task, result)
                    update_control_bucket(by_control[control_type], result)

    finally:
        clear_kv_cache(dit, vae)

    control_rows = [bucket_to_row(control, bucket) for control, bucket in sorted(by_control.items())]
    write_csv(comparison_path, control_rows)
    with metrics_path.open("w", encoding="utf-8") as metrics_f:
        for row in control_rows:
            metrics_f.write(
                json.dumps(
                    {
                        "created_at": int(time.time()),
                        "control_type": row["control_type"],
                        "metrics": row,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
            )
    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "device": str(device),
        "num_input_packets": len(packets_all),
        "num_selected_packets": len(packets),
        "control_types": control_types,
        "control_generation_warnings": control_generation_warnings,
        "control_comparison": control_rows,
        "elapsed_seconds": time.time() - start_time,
        "artifacts": {
            "summary_json": str(summary_path),
            "metrics_jsonl": str(metrics_path),
            "generations_jsonl": str(generations_path),
            "control_comparison_csv": str(comparison_path),
            "tasks_dirs": {
                control: str(output_dir / f"tasks_{control}") for control in control_types
            },
        },
        "interpretation": (
            "P2-D replay diagnostic. Matched-vs-corrupted generation differences are "
            "sequential communication evidence only after official scoring/text baseline "
            "comparisons are added; this runner itself is local-only and non-training."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def run_one_control(
    *,
    packet: dict[str, Any],
    sample_id: Any,
    raw_item: dict[str, Any],
    replay_blocks: list[torch.Tensor] | None,
    control_type: str,
    source_packet: dict[str, Any] | None,
    dit: Any,
    vae: Any,
    tokenizer: Any,
    apply_prompt_template: Any,
    sample_with_strategies: Any,
    config: SequentialLatentMasConfig,
    device: torch.device,
) -> dict[str, Any]:
    clear_kv_cache(dit, vae)
    enable_kv_cache(dit, vae)
    task = str(packet["task"])
    state = build_prompt_state(
        raw_item=raw_item,
        task=task,
        tokenizer=tokenizer,
        vae=vae,
        block_size=int(dit.block_size),
        apply_prompt_template=apply_prompt_template,
        config=config,
        device=device,
        latent_dim=int(packet["prefix_contract"]["latent_dim"]),
    )
    block_size = state.block_size
    patch_size = state.patch_size
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    txt_shape_cum = shape_tensor([state.prefix_len], device)
    txt_q_shape = shape_tensor([block_size], device)
    context_ids: torch.Tensor | None = None
    eos_status = torch.zeros(1, dtype=torch.bool, device=device)
    replay_count = 0
    generated_blocks = 0
    probe_stats_last: dict[str, Any] = {}

    with torch.no_grad():
        if state.prefix_len > 0:
            prefix = state.prefix.to(device)
            txt_shape_prefix = shape_tensor([state.prefix_len], device)
            with autocast_context(device):
                _ = dit(
                    txt=prefix.to(model_dtype),
                    txt_shape=txt_shape_prefix,
                    txt_q_shape=txt_shape_prefix,
                    timestep=torch.zeros(prefix.shape[0], device=device, dtype=model_dtype),
                    update_kv=True,
                    use_kv_cache=True,
                )
                _ = vae.decode(
                    z=prefix,
                    txt_shape=txt_shape_prefix,
                    txt_q_shape=txt_shape_prefix,
                    update_kv=True,
                )

        if replay_blocks:
            for block in replay_blocks:
                txt_shape_cum = txt_shape_cum + block_size
                z = block.to(device)
                decoded_logits = decode_and_sample_block(
                    z=z,
                    vae=vae,
                    tokenizer=tokenizer,
                    sample_with_strategies=sample_with_strategies,
                    config=config,
                    txt_shape_cum=txt_shape_cum,
                    txt_q_shape=txt_q_shape,
                    context_ids=context_ids,
                )
                one_block_ids = decoded_logits["one_block_ids"]
                context_ids = one_block_ids if context_ids is None else torch.cat([context_ids, one_block_ids], dim=1)
                probe_stats_last = decoded_logits["probe_stats"]
                with autocast_context(device):
                    _ = dit(
                        txt=z.to(model_dtype),
                        txt_shape=txt_shape_cum,
                        txt_q_shape=txt_q_shape,
                        timestep=torch.zeros(z.shape[0], device=device, dtype=model_dtype),
                        update_kv=True,
                        use_kv_cache=True,
                    )
                if token_id_present(one_block_ids[0].detach().cpu().tolist(), config.eos_token_id):
                    eos_status[0] = True
                if token_id_present(one_block_ids[0].detach().cpu().tolist(), config.im_end_token_id):
                    eos_status[0] = True
                replay_count += 1

        receiver_blocks = receiver_budget(packet, replay_count, config)
        for _ in range(receiver_blocks):
            if eos_status.all():
                break
            txt_shape_cum = txt_shape_cum + block_size
            absolute_step = replay_count + generated_blocks
            z = denoise_next_block(
                dit=dit,
                state=state,
                config=config,
                device=device,
                sample_seed_id=stable_sample_seed_id(sample_id),
                txt_shape_cum=txt_shape_cum,
                txt_q_shape=txt_q_shape,
                absolute_step=absolute_step,
                use_first_block_fill=(replay_count == 0 and absolute_step == 0),
            )
            decoded_logits = decode_and_sample_block(
                z=z,
                vae=vae,
                tokenizer=tokenizer,
                sample_with_strategies=sample_with_strategies,
                config=config,
                txt_shape_cum=txt_shape_cum,
                txt_q_shape=txt_q_shape,
                context_ids=context_ids,
            )
            one_block_ids = decoded_logits["one_block_ids"]
            context_ids = one_block_ids if context_ids is None else torch.cat([context_ids, one_block_ids], dim=1)
            probe_stats_last = decoded_logits["probe_stats"]
            with autocast_context(device):
                _ = dit(
                    txt=z.to(model_dtype),
                    txt_shape=txt_shape_cum,
                    txt_q_shape=txt_q_shape,
                    timestep=torch.zeros(z.shape[0], device=device, dtype=model_dtype),
                    update_kv=True,
                    use_kv_cache=True,
                )
            if token_id_present(one_block_ids[0].detach().cpu().tolist(), config.eos_token_id):
                eos_status[0] = True
            if token_id_present(one_block_ids[0].detach().cpu().tolist(), config.im_end_token_id):
                eos_status[0] = True
            generated_blocks += 1

    generated_text = decode_context_text(
        context_ids=context_ids,
        tokenizer=tokenizer,
        trim_count=state.first_block_prompt_token_count,
    )
    source_sample_key = "" if source_packet is None else str(source_packet.get("sample_key", ""))
    output_id = raw_item["id"] if "id" in raw_item else sample_id
    return {
        "id": output_id,
        "sample_key": packet["sample_key"],
        "task": task,
        "control_type": control_type,
        "source_sample_key": source_sample_key,
        "sender_selected_block": int(packet["agent_a"]["selected_block"]),
        "receiver_blocks_generated": generated_blocks,
        "replay_blocks_consumed": replay_count,
        "total_blocks": replay_count + generated_blocks,
        "prompt": state.prompt_text,
        "receiver_context_mode": config.receiver_context_mode,
        "generate": generated_text,
        "ground_truth": raw_item.get("ground_truth", raw_item.get("answer", "")),
        "choices": raw_item.get("choices", []),
        "stop_reason": "stop_token" if bool(eos_status.item()) else "receiver_budget",
        "latent_elements_received": replay_count * state.block_size * state.latent_dim,
        "decode_replay_required": bool(replay_blocks),
        **{f"last_{key}": value for key, value in probe_stats_last.items()},
    }


def build_prompt_state(
    *,
    raw_item: dict[str, Any],
    task: str,
    tokenizer: Any,
    vae: Any,
    block_size: int,
    apply_prompt_template: Any,
    config: SequentialLatentMasConfig,
    device: torch.device,
    latent_dim: int,
) -> PromptState:
    patch_size = int(vae.patch_size)
    chunk = patch_size * block_size
    if config.receiver_context_mode == "empty_prompt":
        return PromptState(
            prompt_text="",
            prefix=torch.empty((0, latent_dim), dtype=torch.float32, device=device),
            prefix_len=0,
            first_block_latents=torch.zeros((block_size, latent_dim), dtype=torch.float32, device=device),
            first_block_labels=torch.full((block_size,), 2, dtype=torch.long, device=device),
            first_block_prompt_token_count=0,
            block_size=block_size,
            patch_size=patch_size,
            latent_dim=latent_dim,
        )
    prompt_text = apply_prompt_template(
        task=task,
        context=raw_item.get("context", ""),
        question=raw_item.get("question", ""),
        answer=raw_item.get("ground_truth", raw_item.get("answer", "")),
        choices=raw_item.get("choices"),
    )
    ids = tokenizer.encode(prompt_text).ids
    pad_len = (chunk - len(ids) % chunk) % chunk
    token_labels = torch.tensor([1] * len(ids) + [3] * pad_len, dtype=torch.long, device=device)
    input_ids = torch.tensor(ids + [config.pad_token_id] * pad_len, dtype=torch.long, device=device)
    with autocast_context(device):
        enc = vae.encode([input_ids])
        latents = ((enc.latents_list[0] - vae.shifting_factor) * vae.scaling_factor).float()
    n_patches = token_labels.shape[0] // patch_size
    reshaped = token_labels.view(n_patches, patch_size)
    latent_labels = torch.full((n_patches,), 3, dtype=torch.long, device=device)
    latent_labels[(reshaped == 1).any(dim=1)] = 1
    num_prompt_latents = int((latent_labels == 1).sum().item())
    latent_total = latents.shape[0]
    pad_placeholder = latents[latent_total - block_size : latent_total].clone()
    if num_prompt_latents % block_size != 0:
        start_idx = (num_prompt_latents // block_size) * block_size
        block_latents = latents[start_idx : start_idx + block_size].clone()
        block_labels = latent_labels[start_idx : start_idx + block_size].clone()
        block_labels[block_labels == 3] = 2
        token_start = start_idx * patch_size
        token_end = min(token_start + block_size * patch_size, token_labels.shape[0])
        first_prompt_count = int((token_labels[token_start:token_end] == 1).sum().item())
        prefix = latents[:start_idx].clone()
    else:
        prefix = latents[:num_prompt_latents].clone()
        block_latents = pad_placeholder
        block_labels = torch.full((block_size,), 2, dtype=torch.long, device=device)
        first_prompt_count = 0
    return PromptState(
        prompt_text=prompt_text,
        prefix=prefix.detach().float(),
        prefix_len=int(prefix.shape[0]),
        first_block_latents=block_latents.detach().float(),
        first_block_labels=block_labels.detach(),
        first_block_prompt_token_count=first_prompt_count,
        block_size=block_size,
        patch_size=patch_size,
        latent_dim=int(latents.shape[-1]),
    )


def denoise_next_block(
    *,
    dit: Any,
    state: PromptState,
    config: SequentialLatentMasConfig,
    device: torch.device,
    sample_seed_id: int,
    txt_shape_cum: torch.Tensor,
    txt_q_shape: torch.Tensor,
    absolute_step: int,
    use_first_block_fill: bool,
) -> torch.Tensor:
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if config.per_sample_noise_seed is None:
        txt = torch.randn(state.block_size, state.latent_dim, device=device)
    else:
        generator = torch.Generator(device=device)
        generator.manual_seed(
            int(config.per_sample_noise_seed) + sample_seed_id * 1_000 + int(absolute_step) * 10_000_000
        )
        txt = torch.randn(state.block_size, state.latent_dim, device=device, generator=generator)
    flat_mask = state.first_block_labels.to(device) == 1
    first_latents = state.first_block_latents.to(device)
    timesteps = torch.linspace(1000, 0, config.timestep_num + 1, dtype=torch.float32)
    cfg_scale_first = config.guidance_scale if state.prefix_len > 0 else 1.0
    for t_curr, t_next in zip(timesteps[:-1], timesteps[1:]):
        ts_batch = torch.full((txt.shape[0],), float(t_curr), device=device)
        dt = (float(t_curr) - float(t_next)) / 1000.0
        if use_first_block_fill:
            ts_batch[flat_mask] = 0
            txt[flat_mask] = first_latents[flat_mask]
        with autocast_context(device):
            drift_cond = dit(
                txt=txt.to(model_dtype),
                txt_shape=txt_shape_cum,
                txt_q_shape=txt_q_shape,
                timestep=ts_batch.to(model_dtype),
                update_kv=False,
                use_kv_cache=True,
            ).txt_sample
            drift_uncond = dit(
                txt=txt.to(model_dtype),
                txt_shape=txt_q_shape,
                txt_q_shape=txt_q_shape,
                timestep=ts_batch.to(model_dtype),
                update_kv=False,
                use_kv_cache=False,
            ).txt_sample
        scale = cfg_scale_first if use_first_block_fill else config.guidance_scale
        drift = scale * (drift_cond - drift_uncond) + drift_uncond
        txt_next = txt - drift * dt
        if use_first_block_fill:
            txt_next[flat_mask] = first_latents[flat_mask]
        txt = txt_next
    return txt.detach().float()


def decode_and_sample_block(
    *,
    z: torch.Tensor,
    vae: Any,
    tokenizer: Any,
    sample_with_strategies: Any,
    config: SequentialLatentMasConfig,
    txt_shape_cum: torch.Tensor,
    txt_q_shape: torch.Tensor,
    context_ids: torch.Tensor | None,
) -> dict[str, Any]:
    if z.dim() != 2:
        raise ValueError(f"expected one latent block with shape [block, dim], got {tuple(z.shape)}")
    with autocast_context(z.device):
        decoded = vae.decode(
            z=z,
            txt_shape=txt_shape_cum,
            txt_q_shape=txt_q_shape,
            update_kv=True,
        )
    logits = decoded.view(1, z.shape[0] * int(vae.patch_size), -1)
    one_block_ids = sample_with_strategies(
        logits,
        generated_ids=context_ids,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
    )
    probe = compute_logit_probe_stats(
        decoded_logits=logits,
        eos_token_id=config.eos_token_id,
        im_end_token_id=config.im_end_token_id,
    )[0]
    _ = tokenizer
    return {"one_block_ids": one_block_ids, "probe_stats": probe}


def decode_context_text(context_ids: torch.Tensor | None, tokenizer: Any, trim_count: int) -> str:
    if context_ids is None:
        return ""
    ids = context_ids.detach().cpu()[0].tolist()
    trim_count = max(0, min(int(trim_count), len(ids)))
    return tokenizer.decode(ids[trim_count:], skip_special_tokens=False)


def receiver_budget(packet: dict[str, Any], replay_count: int, config: SequentialLatentMasConfig) -> int:
    if config.receiver_budget_mode == "fixed":
        return int(config.fixed_receiver_blocks)
    max_budget = int(packet["agent_a"]["max_block_budget"])
    return max(0, max_budget - int(replay_count))


def enable_kv_cache(dit: Any, vae: Any) -> None:
    for block in dit.blocks:
        block.set_kv_cache(True)
    vae.set_kv_cache(True)


def clear_kv_cache(dit: Any, vae: Any) -> None:
    for block in dit.blocks:
        block.set_kv_cache(False)
    vae.set_kv_cache(False)


def load_packets(path: Path) -> list[dict[str, Any]]:
    packets = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                packets.append(json.loads(line))
    return packets


def add_cola_code_path(path: str) -> None:
    if not path:
        return
    code_path = Path(path)
    if code_path.exists() and str(code_path) not in sys.path:
        sys.path.insert(0, str(code_path))


def select_packets(
    packets: list[dict[str, Any]],
    config: SequentialLatentMasConfig,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if config.max_packets_per_task > 0:
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for packet in packets:
            by_task[str(packet["task"])].append(packet)
        selected = []
        for task in OFFICIAL_COLA_TASKS:
            candidates = list(by_task.get(task, []))
            if not candidates:
                continue
            count = min(config.max_packets_per_task, len(candidates))
            selected.extend(rng.sample(candidates, count))
        selected.sort(key=lambda item: (str(item["task"]), str(item["sample_key"])))
        return selected[: config.max_packets] if config.max_packets else selected
    if config.max_packets and config.max_packets < len(packets):
        return sorted(rng.sample(packets, config.max_packets), key=lambda item: str(item["sample_key"]))
    return packets


class TaskDataCache:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.cache: dict[str, dict[Any, dict[str, Any]]] = {}

    def get(self, task: str, sample_id: Any) -> dict[str, Any]:
        if task not in self.cache:
            path = self.data_root / f"{task}.jsonl"
            rows = read_jsonl(path)
            self.cache[task] = {row.get("id", idx): row for idx, row in enumerate(rows)}
        if sample_id not in self.cache[task]:
            raise KeyError(f"sample id {sample_id} not found for task {task}")
        return self.cache[task][sample_id]


def sample_id_from_key(sample_key: str) -> int:
    return int(sample_key.rsplit("::", 1)[-1])


class DecisionSampleIdCache:
    def __init__(self) -> None:
        self.cache: dict[str, dict[str, Any]] = {}

    def resolve(self, packet: dict[str, Any]) -> Any:
        sample_key = str(packet["sample_key"])
        try:
            return sample_id_from_key(sample_key)
        except ValueError:
            pass
        decision_path = str(packet["audit_refs"]["halt_decisions_jsonl"])
        if decision_path not in self.cache:
            self.cache[decision_path] = {}
            with Path(decision_path).open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if "sample_id" in row:
                        self.cache[decision_path][str(row["sample_key"])] = row["sample_id"]
        if sample_key not in self.cache[decision_path]:
            raise KeyError(f"sample_id not found for {sample_key} in {decision_path}")
        return self.cache[decision_path][sample_key]


def stable_sample_seed_id(sample_id: Any) -> int:
    try:
        return int(sample_id)
    except (TypeError, ValueError):
        return int(zlib.crc32(str(sample_id).encode("utf-8")))


def write_task_generation(output_dir: Path, control_type: str, task: str, row: dict[str, Any]) -> None:
    task_dir = output_dir / f"tasks_{control_type}"
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / f"{task}.jsonl"
    scorer_row = {
        key: value
        for key, value in row.items()
        if key
        in {
            "id",
            "prompt",
            "generate",
            "ground_truth",
            "choices",
            "sample_key",
            "control_type",
            "sender_selected_block",
            "receiver_blocks_generated",
            "replay_blocks_consumed",
            "total_blocks",
            "latent_elements_received",
            "receiver_context_mode",
        }
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(scorer_row, ensure_ascii=False, sort_keys=True) + "\n")


def init_control_bucket() -> dict[str, Any]:
    return {
        "count": 0,
        "nonempty": 0,
        "agreements_with_matched": 0,
        "total_blocks": 0,
        "replay_blocks": 0,
        "receiver_blocks": 0,
        "latent_elements": 0,
        "generated_chars": 0,
    }


def update_control_bucket(bucket: dict[str, Any], row: dict[str, Any]) -> None:
    bucket["count"] += 1
    bucket["nonempty"] += int(bool(str(row["generate"]).strip()))
    bucket["agreements_with_matched"] += int(bool(row["agreement_with_matched"]))
    bucket["total_blocks"] += int(row["total_blocks"])
    bucket["replay_blocks"] += int(row["replay_blocks_consumed"])
    bucket["receiver_blocks"] += int(row["receiver_blocks_generated"])
    bucket["latent_elements"] += int(row["latent_elements_received"])
    bucket["generated_chars"] += len(str(row["generate"]))


def bucket_to_row(control_type: str, bucket: dict[str, Any]) -> dict[str, Any]:
    count = max(int(bucket["count"]), 1)
    return {
        "control_type": control_type,
        "count": int(bucket["count"]),
        "nonempty_rate": bucket["nonempty"] / count,
        "agreement_with_matched_rate": bucket["agreements_with_matched"] / count,
        "avg_total_blocks": bucket["total_blocks"] / count,
        "avg_replay_blocks": bucket["replay_blocks"] / count,
        "avg_receiver_blocks": bucket["receiver_blocks"] / count,
        "avg_latent_elements_received": bucket["latent_elements"] / count,
        "avg_generated_chars": bucket["generated_chars"] / count,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def normalize_text(value: str | None) -> str:
    return " ".join(str(value or "").strip().lower().split())


if __name__ == "__main__":
    main()
