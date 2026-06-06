"""Run corrected P2-D Agent-B channel-equivalent evaluation.

This local-only generation script compares receiver conditions under the same
sample, handoff depth, receiver block budget, and official scorer format.

By default it follows the LatentMAS-style handoff boundary:

* ``none``: B receives an empty message/input.
* ``text``: B receives only Agent A raw text message at depth t.
* ``latent_matched``: B receives only Agent A matched latent packet at depth t.
* ``latent_*`` corruptions: B receives only corrupted latent payloads at depth t.

``--agent-b-input-contract shared_context`` additionally gives B the original
task prompt.  That mode is diagnostic only, not the canonical Agent-A -> Agent-B
communication protocol.

The raw text message must come from
``build_cola_agent_channel_messages.py`` and must not be P1
``selected_prediction``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from drla.scripts.audit_cola_agent_latent_packet_distribution import (
    ShardCache,
    build_control_blocks,
    build_packet_indexes,
    load_packet_blocks,
)
from drla.scripts.collect_cola_block_traces import (
    ColaBlockTraceConfig,
    autocast_context,
    load_cola_symbols,
    read_jsonl,
    resolve_device,
    set_seed,
    shape_tensor,
    token_id_present,
)
from drla.scripts.run_cola_sequential_latent_mas import (
    OFFICIAL_COLA_TASKS,
    PromptState,
    TaskDataCache,
    build_prompt_state,
    clear_kv_cache,
    decode_and_sample_block,
    decode_context_text,
    denoise_next_block,
    enable_kv_cache,
    stable_sample_seed_id,
)
from drla.tracking import require_swanlab_disabled_for_non_training


CHANNELS = [
    "none",
    "text",
    "latent_matched",
    "latent_matched_cache_only",
    "latent_matched_dit_only_cache",
    "latent_matched_vae_only_cache",
    "latent_shuffle",
    "latent_shuffle_cache_only",
    "latent_shuffle_dit_only_cache",
    "latent_shuffle_vae_only_cache",
    "latent_cross_task",
    "latent_cross_task_cache_only",
    "latent_cross_task_dit_only_cache",
    "latent_cross_task_vae_only_cache",
    "latent_wrong_block",
    "latent_wrong_block_cache_only",
    "latent_wrong_block_dit_only_cache",
    "latent_wrong_block_vae_only_cache",
    "latent_noise",
    "latent_noise_cache_only",
    "latent_noise_dit_only_cache",
    "latent_noise_vae_only_cache",
    "latent_rotation",
    "latent_rotation_cache_only",
    "latent_rotation_dit_only_cache",
    "latent_rotation_vae_only_cache",
]


@dataclass(frozen=True)
class AgentBChannelEvalConfig:
    channel_messages_jsonl: str
    packets_jsonl: str = (
        "/data1/luyifei/drla/outputs/cola_agent_latent_comm/"
        "p2_agent_latent_comm_v2_locked_seed66_67_68_split20260601_20260529/"
        "agent_latent_comm_packets_test.jsonl"
    )
    output_dir: str = (
        "/data1/luyifei/drla/outputs/cola_agent_channel_eval/"
        "p2d_agent_b_channel_eval"
    )
    dit_path: str = ColaBlockTraceConfig.dit_path
    vae_path: str = ColaBlockTraceConfig.vae_path
    tokenizer_path: str = ColaBlockTraceConfig.tokenizer_path
    cola_code_path: str = "/data1/luyifei/Cola-DLM/code"
    data_root: str = "/data1/luyifei/Cola-DLM/code/generate_task_data"
    channels: str = ",".join(CHANNELS)
    message_start: int = 0
    message_end: int = 0
    max_messages: int = 0
    seed: int = 20260531
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
    noise_std: float = 1.0
    max_cached_shards: int = 1024
    score_output_scope: str = "receiver_only"
    agent_b_input_contract: str = "message_only"
    device: str = "auto"
    swanlab_mode: str = "disabled"


def main() -> None:
    summary = run_agent_b_channel_eval(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> AgentBChannelEvalConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel-messages-jsonl", required=True)
    parser.add_argument("--packets-jsonl", default=AgentBChannelEvalConfig.packets_jsonl)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dit-path", default=AgentBChannelEvalConfig.dit_path)
    parser.add_argument("--vae-path", default=AgentBChannelEvalConfig.vae_path)
    parser.add_argument("--tokenizer-path", default=AgentBChannelEvalConfig.tokenizer_path)
    parser.add_argument("--cola-code-path", default=AgentBChannelEvalConfig.cola_code_path)
    parser.add_argument("--data-root", default=AgentBChannelEvalConfig.data_root)
    parser.add_argument("--channels", default=AgentBChannelEvalConfig.channels)
    parser.add_argument("--message-start", type=int, default=AgentBChannelEvalConfig.message_start)
    parser.add_argument("--message-end", type=int, default=AgentBChannelEvalConfig.message_end)
    parser.add_argument("--max-messages", type=int, default=0)
    parser.add_argument("--seed", type=int, default=AgentBChannelEvalConfig.seed)
    parser.add_argument("--timestep-num", type=int, default=AgentBChannelEvalConfig.timestep_num)
    parser.add_argument("--guidance-scale", type=float, default=AgentBChannelEvalConfig.guidance_scale)
    parser.add_argument("--temperature", type=float, default=AgentBChannelEvalConfig.temperature)
    parser.add_argument("--top-k", type=int, default=AgentBChannelEvalConfig.top_k)
    parser.add_argument("--top-p", type=float, default=AgentBChannelEvalConfig.top_p)
    parser.add_argument("--repetition-penalty", type=float, default=AgentBChannelEvalConfig.repetition_penalty)
    parser.add_argument("--pad-token-id", type=int, default=AgentBChannelEvalConfig.pad_token_id)
    parser.add_argument("--eos-token-id", type=int, default=AgentBChannelEvalConfig.eos_token_id)
    parser.add_argument("--im-end-token-id", type=int, default=AgentBChannelEvalConfig.im_end_token_id)
    parser.add_argument("--per-sample-noise-seed", type=int, default=AgentBChannelEvalConfig.per_sample_noise_seed)
    parser.add_argument("--disable-per-sample-noise-seed", action="store_true")
    parser.add_argument("--noise-std", type=float, default=AgentBChannelEvalConfig.noise_std)
    parser.add_argument("--max-cached-shards", type=int, default=AgentBChannelEvalConfig.max_cached_shards)
    parser.add_argument(
        "--score-output-scope",
        choices=["receiver_only", "legacy_all_visible"],
        default=AgentBChannelEvalConfig.score_output_scope,
        help=(
            "receiver_only scores only Agent-B tokens generated after the handoff. "
            "legacy_all_visible reproduces the historical leaky behavior where "
            "A text/replay tokens are included in generate."
        ),
    )
    parser.add_argument(
        "--agent-b-input-contract",
        choices=["message_only", "shared_context"],
        default=AgentBChannelEvalConfig.agent_b_input_contract,
        help=(
            "message_only means Agent B's prompt/state is only Agent A's output "
            "or latent packet. shared_context additionally gives B the original "
            "task prompt and is diagnostic only."
        ),
    )
    parser.add_argument("--device", default=AgentBChannelEvalConfig.device)
    parser.add_argument("--swanlab-mode", default=AgentBChannelEvalConfig.swanlab_mode)
    args = parser.parse_args()
    if (
        args.message_start < 0
        or args.message_end < 0
        or args.max_messages < 0
        or args.noise_std < 0
        or args.max_cached_shards < 0
    ):
        raise ValueError("limits/std/cache size must be non-negative")
    if args.message_end and args.message_end < args.message_start:
        raise ValueError("--message-end must be greater than or equal to --message-start")
    channels = normalize_channels(args.channels)
    return AgentBChannelEvalConfig(
        channel_messages_jsonl=args.channel_messages_jsonl,
        packets_jsonl=args.packets_jsonl,
        output_dir=args.output_dir,
        dit_path=args.dit_path,
        vae_path=args.vae_path,
        tokenizer_path=args.tokenizer_path,
        cola_code_path=args.cola_code_path,
        data_root=args.data_root,
        channels=",".join(channels),
        message_start=args.message_start,
        message_end=args.message_end,
        max_messages=args.max_messages,
        seed=args.seed,
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
        noise_std=args.noise_std,
        max_cached_shards=args.max_cached_shards,
        score_output_scope=args.score_output_scope,
        agent_b_input_contract=args.agent_b_input_contract,
        device=args.device,
        swanlab_mode=args.swanlab_mode,
    )


def run_agent_b_channel_eval(config: AgentBChannelEvalConfig) -> dict[str, Any]:
    require_swanlab_disabled_for_non_training(
        config.swanlab_mode,
        script_kind="P2-D Agent-B channel-equivalent evaluation",
    )
    set_seed(config.seed)
    rng = random.Random(config.seed)
    torch_generator = torch.Generator(device="cpu").manual_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    generations_path = output_dir / "generations.jsonl"
    comparison_path = output_dir / "channel_comparison.csv"
    summary_path = output_dir / "summary.json"

    add_cola_code_path(config.cola_code_path)
    channels = normalize_channels(config.channels)
    messages_all = read_jsonl(Path(config.channel_messages_jsonl))
    message_end = config.message_end if config.message_end else len(messages_all)
    messages = messages_all[config.message_start : message_end]
    if config.max_messages:
        messages = messages[: config.max_messages]
    if not messages:
        raise ValueError("no channel messages loaded")
    packets = read_jsonl(Path(config.packets_jsonl))
    packets_by_key = {str(packet["sample_key"]): (idx, packet) for idx, packet in enumerate(packets)}
    packet_indexes = build_packet_indexes(packets)
    data_cache = TaskDataCache(Path(config.data_root))
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

    by_channel: dict[str, dict[str, Any]] = defaultdict(init_bucket)
    rows: list[dict[str, Any]] = []
    start_time = time.time()

    try:
        with generations_path.open("w", encoding="utf-8") as gen_f:
            for local_message_index, message in enumerate(messages):
                message_index = config.message_start + local_message_index
                sample_key = str(message["sample_key"])
                if sample_key not in packets_by_key:
                    raise KeyError(f"packet not found for message sample_key {sample_key}")
                packet_index, packet = packets_by_key[sample_key]
                task = str(message["task"])
                sample_id = message["sample_id"]
                raw_item = data_cache.get(task, sample_id)
                matched_blocks = load_packet_blocks(packet, shard_cache)
                receiver_blocks = max(0, int(message["max_block_budget"]) - int(message["handoff_depth"]))
                task_prompt = cola["apply_prompt_template"](
                    task=task,
                    context=raw_item.get("context", ""),
                    question=raw_item.get("question", ""),
                    answer=raw_item.get("ground_truth", raw_item.get("answer", "")),
                    choices=raw_item.get("choices"),
                )
                for channel in channels:
                    replay_blocks = None
                    source_packet = None
                    base_channel, _ = split_latent_channel_mode(channel)
                    if base_channel == "latent_matched":
                        replay_blocks = matched_blocks
                        source_packet = packet
                    elif base_channel.startswith("latent_"):
                        control_type = base_channel.removeprefix("latent_")
                        replay_blocks, source_packet, warning = build_control_blocks(
                            control_type=control_type,
                            packet_index=packet_index,
                            packet=packet,
                            packets=packets,
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
                    result = run_one_channel(
                        message=message,
                        packet=packet,
                        raw_item=raw_item,
                        task_prompt=task_prompt,
                        channel=channel,
                        replay_blocks=replay_blocks,
                        source_packet=source_packet,
                        receiver_blocks=receiver_blocks,
                        dit=dit,
                        vae=vae,
                        tokenizer=tokenizer,
                        apply_prompt_template=cola["apply_prompt_template"],
                        sample_with_strategies=cola["sample_with_strategies"],
                        config=config,
                        device=device,
                    )
                    result["message_index"] = message_index
                    rows.append(result)
                    gen_f.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                    write_task_generation(output_dir, channel, task, result)
                    update_bucket(by_channel[channel], result)
    finally:
        clear_kv_cache(dit, vae)

    comparison_rows = [bucket_to_row(channel, bucket) for channel, bucket in sorted(by_channel.items())]
    write_csv(comparison_path, comparison_rows)
    with metrics_path.open("w", encoding="utf-8") as metrics_f:
        for row in comparison_rows:
            metrics_f.write(json.dumps({"created_at": int(time.time()), "channel": row["channel"], "metrics": row}, sort_keys=True) + "\n")

    summary = {
        "created_at": int(time.time()),
        "config": asdict(config),
        "device": str(device),
        "channels": channels,
        "num_messages": len(messages),
        "num_generations": len(rows),
        "channel_comparison": comparison_rows,
        "control_generation_warnings": control_generation_warnings,
        "elapsed_seconds": time.time() - start_time,
        "artifacts": {
            "summary_json": str(summary_path),
            "metrics_jsonl": str(metrics_path),
            "generations_jsonl": str(generations_path),
            "channel_comparison_csv": str(comparison_path),
            "tasks_dirs": {channel: str(output_dir / f"tasks_{channel}") for channel in channels},
        },
        "interpretation": (
            "Corrected P2-D Agent-B channel-equivalent generation. Only final "
            "Agent-B outputs are scorer inputs; text channel uses raw native "
            "trace message text, not selected_prediction."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def run_one_channel(
    *,
    message: dict[str, Any],
    packet: dict[str, Any],
    raw_item: dict[str, Any],
    task_prompt: str,
    channel: str,
    replay_blocks: list[torch.Tensor] | None,
    source_packet: dict[str, Any] | None,
    receiver_blocks: int,
    dit: Any,
    vae: Any,
    tokenizer: Any,
    apply_prompt_template: Any,
    sample_with_strategies: Any,
    config: AgentBChannelEvalConfig,
    device: torch.device,
) -> dict[str, Any]:
    clear_kv_cache(dit, vae)
    enable_kv_cache(dit, vae)
    task = str(message["task"])
    latent_dim = int(packet["prefix_contract"]["latent_dim"])
    raw_text = str(message["a_raw_text_message_t"])
    if config.agent_b_input_contract == "message_only":
        prompt_text = raw_text if channel == "text" else ""
        prompt_state_mode = "empty_prompt"
    elif config.agent_b_input_contract == "shared_context":
        prompt_text = task_prompt + raw_text if channel == "text" else task_prompt
        prompt_state_mode = "full_prompt"
    else:
        raise ValueError(f"unknown agent_b_input_contract: {config.agent_b_input_contract}")
    if channel == "text":
        state = build_prompt_state_from_prompt_text(
            prompt_text=prompt_text,
            tokenizer=tokenizer,
            vae=vae,
            block_size=int(dit.block_size),
            pad_token_id=config.pad_token_id,
            device=device,
            latent_dim=latent_dim,
        )
    else:
        state = build_prompt_state(
            raw_item=raw_item,
            task=task,
            tokenizer=tokenizer,
            vae=vae,
            block_size=int(dit.block_size),
            apply_prompt_template=apply_prompt_template,
            config=PromptOnlyConfig(config, receiver_context_mode=prompt_state_mode),
            device=device,
            latent_dim=latent_dim,
        )

    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    dit_shape_cum = shape_tensor([state.prefix_len], device)
    vae_shape_cum = shape_tensor([state.prefix_len], device)
    txt_q_shape = shape_tensor([state.block_size], device)
    context_ids: torch.Tensor | None = None
    receiver_token_start = 0
    eos_status = torch.zeros(1, dtype=torch.bool, device=device)
    replay_stop_token_seen = False
    replay_count = 0
    dit_replay_count = 0
    generated_blocks = 0
    probe_stats_last: dict[str, Any] = {}
    _, replay_mode = split_latent_channel_mode(channel)
    emit_replay_text = replay_mode == "decode_and_emit"
    update_replay_vae = replay_mode in {"decode_and_emit", "cache_only", "vae_only_cache"}
    update_replay_dit = replay_mode in {"decode_and_emit", "cache_only", "dit_only_cache"}

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
                z = block.to(device)
                if not emit_replay_text:
                    if update_replay_vae:
                        vae_shape_cum = vae_shape_cum + state.block_size
                        with autocast_context(device):
                            _ = vae.decode(
                                z=z,
                                txt_shape=vae_shape_cum,
                                txt_q_shape=txt_q_shape,
                                update_kv=True,
                            )
                else:
                    vae_shape_cum = vae_shape_cum + state.block_size
                    decoded = decode_and_sample_block(
                        z=z,
                        vae=vae,
                        tokenizer=tokenizer,
                        sample_with_strategies=sample_with_strategies,
                        config=config,
                        txt_shape_cum=vae_shape_cum,
                        txt_q_shape=txt_q_shape,
                        context_ids=context_ids,
                    )
                    one_block_ids = decoded["one_block_ids"]
                    context_ids = one_block_ids if context_ids is None else torch.cat([context_ids, one_block_ids], dim=1)
                    probe_stats_last = decoded["probe_stats"]
                if update_replay_dit:
                    dit_shape_cum = dit_shape_cum + state.block_size
                    with autocast_context(device):
                        _ = dit(
                            txt=z.to(model_dtype),
                            txt_shape=dit_shape_cum,
                            txt_q_shape=txt_q_shape,
                            timestep=torch.zeros(z.shape[0], device=device, dtype=model_dtype),
                            update_kv=True,
                            use_kv_cache=True,
                        )
                    dit_replay_count += 1
                if emit_replay_text:
                    replay_tokens = one_block_ids[0].detach().cpu().tolist()
                    replay_stop_token_seen = replay_stop_token_seen or token_id_present(
                        replay_tokens, config.eos_token_id
                    )
                    replay_stop_token_seen = replay_stop_token_seen or token_id_present(
                        replay_tokens, config.im_end_token_id
                    )
                    if config.score_output_scope == "legacy_all_visible" and replay_stop_token_seen:
                        eos_status[0] = True
                replay_count += 1
        receiver_token_start = 0 if context_ids is None else int(context_ids.shape[1])

        for _ in range(receiver_blocks):
            if eos_status.all():
                break
            dit_shape_cum = dit_shape_cum + state.block_size
            vae_shape_cum = vae_shape_cum + state.block_size
            absolute_step = int(message["handoff_depth"]) + generated_blocks
            z = denoise_next_block(
                dit=dit,
                state=state,
                config=config,
                device=device,
                sample_seed_id=stable_sample_seed_id(message["sample_id"]),
                txt_shape_cum=dit_shape_cum,
                txt_q_shape=txt_q_shape,
                absolute_step=absolute_step,
                use_first_block_fill=(dit_replay_count == 0 and generated_blocks == 0),
            )
            decoded = decode_and_sample_block(
                z=z,
                vae=vae,
                tokenizer=tokenizer,
                sample_with_strategies=sample_with_strategies,
                config=config,
                txt_shape_cum=vae_shape_cum,
                txt_q_shape=txt_q_shape,
                context_ids=context_ids,
            )
            one_block_ids = decoded["one_block_ids"]
            context_ids = one_block_ids if context_ids is None else torch.cat([context_ids, one_block_ids], dim=1)
            probe_stats_last = decoded["probe_stats"]
            with autocast_context(device):
                _ = dit(
                    txt=z.to(model_dtype),
                    txt_shape=dit_shape_cum,
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

    legacy_trim_count = 0 if (not emit_replay_text and replay_count > 0) else state.first_block_prompt_token_count
    all_visible_text = decode_context_text(
        context_ids=context_ids,
        tokenizer=tokenizer,
        trim_count=legacy_trim_count,
    )
    receiver_ids = None if context_ids is None else context_ids[:, receiver_token_start:]
    receiver_trim_count = state.first_block_prompt_token_count if replay_count == 0 else 0
    receiver_text = decode_context_text(
        context_ids=receiver_ids,
        tokenizer=tokenizer,
        trim_count=receiver_trim_count,
    )
    if config.score_output_scope == "legacy_all_visible":
        generated_text = raw_text + all_visible_text if channel == "text" else all_visible_text
    else:
        generated_text = receiver_text
    source_sample_key = "" if source_packet is None else str(source_packet.get("sample_key", ""))
    output_id = raw_item["id"] if "id" in raw_item else message["sample_id"]
    text_token_count = len(tokenizer.encode(raw_text).ids) if channel == "text" else 0
    latent_elements = replay_count * state.block_size * state.latent_dim
    return {
        "id": output_id,
        "sample_key": message["sample_key"],
        "task": task,
        "channel": channel,
        "source_sample_key": source_sample_key,
        "handoff_depth": int(message["handoff_depth"]),
        "receiver_budget_blocks": int(receiver_blocks),
        "receiver_blocks_generated": int(generated_blocks),
        "replay_blocks_consumed": int(replay_count),
        "total_blocks": int(replay_count + generated_blocks),
        "prompt": task_prompt if config.agent_b_input_contract == "shared_context" else "",
        "channel_prompt": prompt_text,
        "a_raw_text_message_t": raw_text if channel == "text" else "",
        "generate": generated_text,
        "ground_truth": raw_item.get("ground_truth", raw_item.get("answer", "")),
        "choices": raw_item.get("choices", []),
        "stop_reason": "stop_token" if bool(eos_status.item()) else "receiver_budget",
        "text_message_tokens_received": text_token_count,
        "text_message_chars_received": len(raw_text) if channel == "text" else 0,
        "latent_elements_received": latent_elements,
        "decode_replay_required": bool(replay_blocks),
        "replay_decode_mode": replay_mode,
        "replay_blocks_decoded_to_text": int(replay_count if emit_replay_text else 0),
        "replay_stop_token_seen": bool(replay_stop_token_seen),
        "score_output_scope": config.score_output_scope,
        "scorer_visible_text_message_tokens": int(text_token_count if config.score_output_scope == "legacy_all_visible" and channel == "text" else 0),
        "scorer_visible_replay_blocks": int(replay_count if config.score_output_scope == "legacy_all_visible" and emit_replay_text else 0),
        "receiver_token_start": int(receiver_token_start),
        "agent_b_input_contract": config.agent_b_input_contract,
        **{f"last_{key}": value for key, value in probe_stats_last.items()},
    }


class PromptOnlyConfig:
    """Small duck-typed wrapper for build_prompt_state."""

    def __init__(self, config: AgentBChannelEvalConfig, *, receiver_context_mode: str) -> None:
        self.receiver_context_mode = receiver_context_mode
        self.pad_token_id = config.pad_token_id


def build_prompt_state_from_prompt_text(
    *,
    prompt_text: str,
    tokenizer: Any,
    vae: Any,
    block_size: int,
    pad_token_id: int,
    device: torch.device,
    latent_dim: int,
) -> PromptState:
    patch_size = int(vae.patch_size)
    chunk = patch_size * block_size
    ids = tokenizer.encode(prompt_text).ids
    pad_len = (chunk - len(ids) % chunk) % chunk
    token_labels = torch.tensor([1] * len(ids) + [3] * pad_len, dtype=torch.long, device=device)
    input_ids = torch.tensor(ids + [pad_token_id] * pad_len, dtype=torch.long, device=device)
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
        latent_dim=int(latents.shape[-1]) if latents.numel() else latent_dim,
    )


def normalize_channels(value: str) -> list[str]:
    channels = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(channels) - set(CHANNELS))
    if unknown:
        raise ValueError(f"unknown channels: {unknown}")
    return channels


def split_latent_channel_mode(channel: str) -> tuple[str, str]:
    if channel.endswith("_dit_only_cache"):
        return channel.removesuffix("_dit_only_cache"), "dit_only_cache"
    if channel.endswith("_vae_only_cache"):
        return channel.removesuffix("_vae_only_cache"), "vae_only_cache"
    if channel.endswith("_cache_only"):
        return channel.removesuffix("_cache_only"), "cache_only"
    return channel, "decode_and_emit"


def write_task_generation(output_dir: Path, channel: str, task: str, row: dict[str, Any]) -> None:
    task_dir = output_dir / f"tasks_{channel}"
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
            "channel",
            "handoff_depth",
            "receiver_budget_blocks",
            "receiver_blocks_generated",
            "replay_blocks_consumed",
            "total_blocks",
            "latent_elements_received",
            "text_message_tokens_received",
            "text_message_chars_received",
            "score_output_scope",
            "scorer_visible_text_message_tokens",
            "scorer_visible_replay_blocks",
            "receiver_token_start",
            "agent_b_input_contract",
        }
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(scorer_row, ensure_ascii=False, sort_keys=True) + "\n")


def init_bucket() -> dict[str, Any]:
    return {
        "count": 0,
        "nonempty": 0,
        "total_blocks": 0,
        "replay_blocks": 0,
        "receiver_blocks": 0,
        "latent_elements": 0,
        "text_tokens": 0,
        "text_chars": 0,
        "generated_chars": 0,
    }


def update_bucket(bucket: dict[str, Any], row: dict[str, Any]) -> None:
    bucket["count"] += 1
    bucket["nonempty"] += int(bool(str(row["generate"]).strip()))
    bucket["total_blocks"] += int(row["total_blocks"])
    bucket["replay_blocks"] += int(row["replay_blocks_consumed"])
    bucket["receiver_blocks"] += int(row["receiver_blocks_generated"])
    bucket["latent_elements"] += int(row["latent_elements_received"])
    bucket["text_tokens"] += int(row["text_message_tokens_received"])
    bucket["text_chars"] += int(row["text_message_chars_received"])
    bucket["generated_chars"] += len(str(row["generate"]))


def bucket_to_row(channel: str, bucket: dict[str, Any]) -> dict[str, Any]:
    count = max(int(bucket["count"]), 1)
    return {
        "channel": channel,
        "count": int(bucket["count"]),
        "nonempty_rate": bucket["nonempty"] / count,
        "avg_total_blocks": bucket["total_blocks"] / count,
        "avg_replay_blocks": bucket["replay_blocks"] / count,
        "avg_receiver_blocks": bucket["receiver_blocks"] / count,
        "avg_latent_elements_received": bucket["latent_elements"] / count,
        "avg_text_message_tokens_received": bucket["text_tokens"] / count,
        "avg_text_message_chars_received": bucket["text_chars"] / count,
        "avg_generated_chars": bucket["generated_chars"] / count,
    }


def add_cola_code_path(path: str) -> None:
    if path:
        code_path = Path(path)
        if code_path.exists() and str(code_path) not in sys.path:
            sys.path.insert(0, str(code_path))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
