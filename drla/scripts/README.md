# Script Status

Last updated: 2026-06-17

## Main Path

The current default path is P3 Dream-DLM LatentMAS. Use
Dream-v0-Instruct-7B as the homogeneous DLM substrate, first pass the MuSiQue
evidence-split TextMAS capability gate, then migrate P0/P1 readiness/halt to
Dream denoising steps, and only then compare receiver-only TextMAS vs
LatentMAS.

The previous official Cola block-wise readiness / halt analysis through P1 and
same-substrate agent-agent latent communication in P2 is archived evidence and
protocol-boundary reference, not the default next experiment.

Current P3 entrypoints:

- `p3_prepare_dream_models.py`: local-only Dream checkpoint download/verification.
  It writes `summary.json`, `environment.json`, and `metrics.jsonl`; it does not
  load weights into GPU, generate, train, or create SwanLab runs.
- `p3_probe_dream_generation.py`: local-only Dream generation/state probe. It
  verifies `diffusion_generate`, `output_history`, token/logit hooks, optional
  hidden-state hook summaries, generated text, and peak memory; decoded text is
  offline probe-only.
- `p3_run_dream_textmas_gate.py`: local-only Dream TextMAS capability-gate
  runner. It consumes locked MuSiQue evidence-split online inputs, calls Dream
  `diffusion_generate` directly, preserves Agent A -> Solver data flow, scores
  offline, and writes `generations.jsonl`, `condition_metrics.csv`,
  `summary.json`, `metrics.jsonl`, and `dream_call_metrics.jsonl`.
- `p3_run_dream_textmas_shard_queue.py`: local-only shard queue wrapper for
  Dream TextMAS evaluation. It schedules deterministic row shards across visible
  GPUs, writes a queue summary, and passes the locked `--max-context-tokens`
  value through to each shard. It does not train or create SwanLab runs.
- `p3_merge_dream_textmas_shards.py`: local-only shard merger for Dream TextMAS
  runs. It validates duplicate rows, expected row/sample counts, condition
  coverage, and writes a merged `generations.jsonl`.
- `p3_audit_dream_static_context_lengths.py`: local-only tokenizer/static prompt
  length audit. Run it before held-out, longer prompts, or new benchmarks to
  catch context overflows before expensive generation. It does not call models,
  train, or create SwanLab runs.
- `p3_freeze_dream_textmas_protocol.py`: local-only calibration protocol locker.
  It records the admitted calibration gate, leakage-audit status, input hashes,
  model path, generation config, parser, scorer, and held-out one-shot rules.
- `p3_collect_dream_step_traces.py`: local-only D3 Dream trace collector. It
  runs locked MuSiQue rows through Dream, records denoising hook events with
  token/process/logit summaries, keeps decoded/scorer fields offline-only, and
  writes `traces.jsonl`, `generations.jsonl`, `dream_trace_call_metrics.jsonl`,
  `summary.json`, and `metrics.jsonl`. It does not train or create SwanLab runs.
  It supports `--hidden-capture-mode summary|suffix_tensor|selected_suffix_tensor`;
  summary mode stores online last-layer hidden statistics, suffix_tensor mode
  writes all hidden suffix `.pt` refs under `hidden_refs/`, and
  selected_suffix_tensor mode uses a D5 readiness policy to write only the
  selected evidence-agent suffix tensor per Dream call. For large packet
  substrate, prefer selected_suffix_tensor; full suffix_tensor is a small-scale
  audit/debug mode unless an explicit storage budget is available.
- `p3_build_dream_readiness_frontier.py`: local-only D4 frontier builder. It
  reads Dream traces, scores solver step probes offline, and writes
  `frontier_events.jsonl`, `frontier_rows.jsonl`, `summary.json`, and
  `metrics.jsonl`. Gold/scorer-derived labels are teacher-only and forbidden as
  online student inputs.
- `p3_train_dream_readiness_student.py`: D5 deep-learning trainer for
  DreamStepReadinessStudent-v1. It trains a causal trajectory Transformer over
  decoder-free hidden/logit/process features with multi-head outputs
  (`ready`, `future_gain`, `prediction_change`, `final_match`). It requires
  CUDA/GPU, SwanLab cloud, `valid_interval <= 10`, local `metrics.jsonl`, and
  `best_checkpoint.pt` / `last_checkpoint.pt`.
- `p3_eval_dream_readiness_policy.py`: local-only D5.5 online halt
  calibration / risk-control evaluator. It loads the D5 best checkpoint,
  selects policy thresholds on the validation split only, reports test
  step-saving and accuracy-drop with paired bootstrap CIs, and never trains or
  creates SwanLab runs.
- `p3_build_dream_latent_packets.py`: local-only D6 latent packet builder. It
  reads `textmas_matched` suffix-tensor traces, applies the D5 policy to
  upstream evidence-agent traces as a step-selection heuristic, and writes
  `packets.jsonl`, `packet_groups.jsonl`, `metrics.jsonl`, and `summary.json`.
  Packet tensors are referenced, not copied; decoded/gold/scorer fields are
  audited as forbidden packet payload. It accepts both full suffix_tensor traces
  and compact selected_suffix_tensor traces when each selected event has a
  `hidden_ref`.
- `p3_train_dream_latent_fuser.py`: D7 V1 MSE latent fuser trainer. It maps
  agent_a/agent_b packets to a solver latent prefix with solver-state
  distillation. This run is useful diagnostic evidence, but V1 does not pass
  held-out corruption controls and must not be cited as the final D7 receiver.
- `p3_eval_dream_latent_fuser_controls.py`: local-only D7 V1 corruption-control
  evaluator for matched, shuffled-row, agent-swap, and zero-packet controls.
- `p3_train_dream_latent_fuser_contrastive.py`: D7 V2 contrastive latent
  receiver/fuser trainer. It aligns packet embeddings to same-row solver latent
  embeddings with symmetric InfoNCE. It requires CUDA/GPU, SwanLab cloud,
  `valid_interval <= 10`, local `metrics.jsonl`, and best/last checkpoints.
- `p3_eval_dream_latent_fuser_contrastive_controls.py`: local-only D7 V2
  retrieval/corruption evaluator for matched, shuffled-row, agent-swap, and
  zero-packet controls.
- `p3_run_dream_latent_prefix_eval.py`: local-only D7 receiver-side generation
  diagnostic. It prepends latent packets as continuous Dream inputs in a custom
  diffusion loop without inserting Agent A/B decoded text. Current raw
  suffix-tensor prefix diagnostic fails, showing representation-space mismatch;
  use it as evidence for training an embedding-space soft-prefix adapter.
- `p3_train_dream_soft_prefix_adapter.py`: D7 V3 embedding-space soft-prefix
  adapter trainer. It freezes Dream, maps D6 agent_a/agent_b suffix tensors to
  Dream input-embedding prefix vectors, and trains through receiver
  answer-token CE on the no-message solver prompt. It is deep-learning
  training and must use CUDA/GPU, SwanLab cloud, `valid_interval <= 10`,
  local `metrics.jsonl`, `best_checkpoint.pt`, and `last_checkpoint.pt`.
- `p3_run_dream_soft_prefix_eval.py`: local-only D7 V3 receiver-side generation
  evaluator for a trained soft-prefix adapter. It runs no optimizer/backward,
  creates no SwanLab run, and tests no-message, matched, shuffled-row,
  agent-swap, and zero-prefix controls without inserting Agent A/B decoded text.
- `p3_train_dream_layer_conditioned_receiver.py`: D7 V4 native
  layer-conditioned receiver trainer. It freezes Dream and injects learned
  cross-attention residual adapters after selected Dream layers, conditioning
  generated/masked positions on D6 packet tensors. It is deep-learning
  training and must use CUDA/GPU, SwanLab cloud, `valid_interval <= 10`,
  local `metrics.jsonl`, `best_checkpoint.pt`, and `last_checkpoint.pt`.
- `p3_run_dream_layer_receiver_eval.py`: local-only D7 V4 receiver generation
  evaluator. It runs the custom layer-conditioned denoising loop for
  no-message, matched, shuffled-row, agent-swap, and zero-packet controls
  without inserting decoded agent text or creating SwanLab runs. Runtime
  `--manifest-json`, `--online-inputs-jsonl`, `--packet-dir`, and
  `--model-path` intentionally override checkpoint training-data config; the
  summary records both `checkpoint_data_config` and `runtime_data_config` so a
  held-out eval cannot silently replay calibration data. Use `--split valid`
  or `--split test` to evaluate checkpoint-defined nontrain sample-id splits,
  and `--exclude-sample-ids` / `--exclude-sample-ids-file` to remove known
  diagnostic-manifest overlaps.
- `p3_train_dream_layer_receiver_corruption_aware.py`: D7 V5 trainer using the
  same layer-conditioned receiver architecture as V4, but with a
  matched-vs-corruption margin objective against zero, shuffled-row, and
  agent-swap packets. It is deep-learning training and must use CUDA/GPU,
  SwanLab cloud, `valid_interval <= 10`, local `metrics.jsonl`, and best/last
  checkpoints. It now supports `--init-checkpoint` and `--corruption-types`.
  The 2026-06-07 V7 run initializes from V4 best and trains only
  zero/shuffled-row corruptions; it is the strongest historical calibration
  aggregate receiver diagnostic with full200 matched primary `0.215`, zero
  `0.095`, shuffled-row `0.180`, no-message `0.035`, and agent-swap `0.210`.
  The 2026-06-17 D7.16 train2000 continuation improves loss-level margins but
  fails checkpoint-defined valid generation controls (`matched=0.040`,
  `no_message=0.055`, `zero=0.045`, `shuffled_row=0.035`), so this trainer's
  teacher-forcing CE/margin metrics must not be treated as D8 admission.
- `p3_merge_dream_receiver_candidate_generations.py`: local-only D7 V6 shard
  merger/auditor for receiver-generated candidate pools. It validates unique
  row/condition coverage, missing pairs, duplicates, forbidden payload keys,
  writes a merged `generations.jsonl`, and creates no SwanLab run.
- `p3_train_dream_receiver_answer_reranker.py`: D7 V6 receiver-side answer
  reranker trainer. Candidate text comes from receiver-generated outputs only;
  gold/scorer/aliases are offline labels and metrics only. It requires
  CUDA/GPU, SwanLab cloud, `valid_interval <= 10`, local `metrics.jsonl`, and
  best/last checkpoints. The 2026-06-07 full200 run is a negative diagnostic:
  best test matched primary `0.20` is below zero/shuffled `0.25`, so do not run
  D8 from this reranker.
- `p3_run_dream_text_encoded_packet_eval.py`: local-only D7.5 diagnostic for
  the "text encoder latent should work" hypothesis. It reads real TextMAS
  `agent_messages`, encodes each message through Dream last-layer hidden
  states, and feeds those tensors to the V7 receiver without inserting agent
  text into the final solver prompt. The 2026-06-17 merged20 result is
  `text_encoded_matched=0.05` while same-row TextMAS matched is `0.40`, so this
  direct text-hidden packet path does not recover text communication. It also
  supports optional `--text-packet-adapter-checkpoint` with `text_adapter_*`
  conditions; this remains local-only evaluation.
- `p3_train_dream_text_packet_adapter.py`: D7.6 text-packet adapter trainer.
  It freezes Dream and the V7 receiver, encodes real TextMAS Agent messages
  into Dream last-layer hidden packets, and trains a lightweight Transformer
  adapter into the receiver packet space. It is deep-learning training and must
  use CUDA/GPU, SwanLab cloud, `valid_interval <= 10`, local `metrics.jsonl`,
  `best_checkpoint.pt`, and `last_checkpoint.pt`. The 2026-06-17 full run
  improves loss/token-F1 but not primary accuracy: `text_adapter_matched=0.05`
  primary / `0.17` token-F1 on the same 20 rows where TextMAS matched is
  `0.40` / `0.42`; do not run D8 from this diagnostic.
- `p3_train_dream_layer_receiver_text_teacher.py`: D7.7 TextMAS-teacher layer
  receiver trainer. It initializes from V7, freezes Dream, updates the layer
  receiver on D6 latent packets, and uses same-row decoded TextMAS messages
  only as a training-time teacher distribution. It is deep-learning training
  and must use CUDA/GPU, SwanLab cloud, `valid_interval <= 10`, local
  `metrics.jsonl`, `best_checkpoint.pt`, and `last_checkpoint.pt`. The
  2026-06-17 full run has strong loss-level margins but negative generation
  controls: best20 matched primary `0.05` / token-F1 `0.1283`, below V7 same-row
  matched `0.10` / `0.19`; do not run D8 from this diagnostic.
- `p3_run_dream_layer_receiver_candidate_pool_eval.py`: D7.8 local-only
  matched-channel candidate-pool evaluator. It samples multiple online
  candidates from V7/D7 receiver conditions, reports first/majority plus
  offline oracle ceiling, never trains, never logs to SwanLab, and never inserts
  decoded Agent text into the solver prompt. It supports `--resume` and
  `safe_decode` for Dream tokenizer edge cases. The 2026-06-17 V7 best20 c8 run
  completed 800 generations: matched oracle primary `0.15`, zero oracle primary
  `0.15`, matched first/majority primary `0.05`; do not run D8 from this
  diagnostic.
- `p3_audit_dream_receiver_interface_distribution.py`: D7.9 local-only
  interface/distribution auditor. It compares no-message AgentB prompt hidden,
  TextMAS AgentB prompt hidden, D6 packet tensors, V7 receiver memory, and V7
  gated layer deltas. It does no generation/training/scoring and never logs to
  SwanLab. The 2026-06-17 V7 best20 audit shows packet-to-TextMAS-last128
  cosine `0.5868`, packet-to-no-message cosine `0.2947`, but V7 memory compresses
  packets to 256-d and layer27 delta/hidden norm ratio is only `0.0472`; next
  receiver work should be text-interface-aligned, not D8.
- `p3_train_dream_text_interface_receiver.py`: D7.10-D7.12 text-interface
  virtual message receiver trainer. It maps D6 packet tensors into continuous
  `prefix_len x 3584` virtual message tokens inserted at AgentB's solver
  interface. Dream is frozen; decoded TextMAS Agent messages are training-only
  hidden/logit teachers. D7.11 additionally supports `--init-checkpoint`,
  `--corrupt-unlikelihood-weight`, `--logit-contrast-weight`,
  `--hidden-contrast-weight`, and `--contrast-temperature` to make
  packet-specific negative controls first-class training targets. D7.12 adds
  `--negative-loss-warmup-steps`, `--selection-token-accuracy-weight`,
  `--selection-margin-target`, and `--selection-margin-overflow-penalty` so
  corruption margins are capped risk controls rather than unbounded checkpoint
  rewards. It is deep-learning training and must use CUDA/GPU, SwanLab cloud,
  `valid_interval <= 10`, local `metrics.jsonl`, `best_checkpoint.pt`, and
  `last_checkpoint.pt`. The 2026-06-17 D7.10 v1 p96d1024 run reached best valid
  matched CE `2.5406`, but generation controls remained non-packet-specific:
  matched/zero/shuffled/agent_swap primary are all `0.05`. The 2026-06-17
  D7.11 packet-specific run suppressed zero-packet generation but also damaged
  matched generation: best/last generation matched primary are both `0.00`. The
  2026-06-17 D7.12 balanced run restores matched primary to `0.05`; its last
  checkpoint suppresses zero/shuffled-row to `0.00`, but agent_swap remains tied
  with matched. Do not run D8 from these diagnostics.
- `p3_run_dream_text_interface_receiver_eval.py`: D7.10-D7.12 local-only
  generation evaluator for text-interface receivers. It inserts virtual latent
  prefixes generated from packets, never inserts decoded Agent text, never
  trains, and never logs to SwanLab. Use it for best/last checkpoint generation
  controls before citing any text-interface receiver.
- `p3_audit_dream_receiver_generation_controls.py`: D7.13 local-only paired
  receiver-control auditor. It reads existing receiver `generations.jsonl`
  files, canonicalizes matched/no-message/zero/shuffled-row/agent-swap
  condition names, and writes condition means, paired matched-vs-control
  bootstrap CIs, and correct-row overlaps. It never loads a model, generates,
  trains, or logs to SwanLab. The 2026-06-17 audit shows V7 full200 is the only
  current receiver with positive primary paired CIs against hard controls
  no-message/zero/shuffled-row; D7.10-D7.12 text-interface runs fail this gate.
- `p3_audit_dream_heldout_packet_readiness.py`: D7.14 local-only held-out
  packet-readiness preflight. It checks whether locked held-out800 has the
  suffix-tensor trace and D6 packet substrate required by V7 receiver eval,
  estimates trace storage from the calibration tensor run, and writes
  `summary.json`, `metrics.jsonl`, and `checks.csv`. It never loads a model,
  generates, trains, or logs to SwanLab. The 2026-06-17 initial artifact showed
  V7 held-out eval was blocked by missing held-out suffix-tensor trace and
  missing held-out D6 packet manifest; after local-only trace/merge/packet
  construction the same preflight reports `status=ready` and
  `can_run_v7_heldout_eval=true`.
- `p3_audit_dream_receiver_split_generalization.py`: D7.15 local-only
  split-generalization auditor. It reconstructs the train/valid/test split
  stored in a receiver checkpoint config and reports existing generation
  outputs by split, including matched-vs-control deltas. It never loads Dream,
  generates, trains, or logs to SwanLab. The 2026-06-17 V7 audit shows the
  full200 hard-gate signal is train-dominated: train passes, valid/test fail.
- `p3_audit_dream_receiver_denoising_sensitivity.py`: D7.17 local-only
  inference-time sensitivity auditor. It loads a trained layer receiver and
  compares matched packets against no-message, zero, shuffled-row, and
  agent-swap controls on the same intermediate Dream denoising state. It writes
  row-level and step-level JSONL metrics, never trains or logs to SwanLab, and
  never inserts decoded agent text into the solver prompt. Use it after
  generation-gate failures to check whether packets actually change top-token
  or transfer-token decisions.
- `p3_train_dream_layer_receiver_denoising_aligned.py`: D7.18 deep-learning
  trainer. It keeps Dream frozen and trains the layer-conditioned receiver on
  partial-denoising answer states instead of only all-mask answer targets. It
  adds matched-vs-zero and matched-vs-shuffled-row gold-token decision margins
  on masked positions, requires CUDA/GPU and SwanLab cloud, enforces
  `valid_interval <= 10` and non-empty train/valid/test splits, and writes
  local `metrics.jsonl`, `best_checkpoint.pt`, and `last_checkpoint.pt`. It
  also supports D7.19 row-binding mode through `--decision-control-weights`,
  `--top-control-weights`, and `--selection-mode row_binding`. D7.18-v1 screen200 improved
  matched CE and zero separation but did not solve shuffled-row / row-specific
  packet binding.
- `p3_run_dream_trace_shard_queue.py`: local-only queue launcher for D3 trace
  collection shards. It runs inference-only shards across GPUs, keeps
  `swanlab` disabled, and writes queue progress/summary.
- `p3_merge_dream_trace_shards.py`: local-only merger for D3 trace shards. It
  remaps shard-local `call_000001` style ids into globally unique call ids so
  D4 frontier building cannot attach a generation row to the wrong solver trace.

Current P3 D0/D1 artifacts:

```text
/data1/luyifei/drla/outputs/p3_dream_models/
Dream-org_Dream-v0-Instruct-7B_prepare_20260606_144309

/data1/luyifei/drla/outputs/p3_dream_models/
dream_instruct_7b_generation_probe_20260606_144650

/data1/luyifei/drla/outputs/p3_dream_textmas_runs/
dream_textmas_gate_full200_merged_20260606

/data1/luyifei/drla/outputs/p3_dream_textmas_aggregates/
dream_textmas_gate_full200_merged_20260606

/data1/luyifei/drla/outputs/p3_dream_protocol_audits/
dream_textmas_protocol_lock_calibration_full200_maxctx4096_20260606

/data1/luyifei/drla/outputs/p3_dream_textmas_runs/
dream_textmas_gate_heldout800_maxctx4096_merged_20260606

/data1/luyifei/drla/outputs/p3_dream_textmas_aggregates/
dream_textmas_gate_heldout800_maxctx4096_merged_20260606

/data1/luyifei/drla/outputs/p3_dream_protocol_audits/
dream_textmas_gate_heldout800_maxctx4096_merged_20260606
```

Current P3 D2 locked held-out result:

```text
samples / rows:
  800 / 5600

row errors / leakage errors:
  0 / 0

primary scores:
  single_full_info = 0.49625
  single_q_only = 0.025
  textmas_matched = 0.38125
  textmas_no_message = 0.0225
  textmas_shuffled_message = 0.045
  textmas_wrong_evidence_or_wrong_shard = 0.045
  textmas_compressed_state = 0.35

paired CI lower:
  full_info_vs_question_only = 0.435
  matched_vs_no_message = 0.32375
  matched_vs_shuffled = 0.29875
  matched_vs_wrong_evidence = 0.29875

protocol:
  max_context_tokens = 4096
  old maxctx2048 held-out = invalid diagnostic only because of context overflow
```

Current P3 D3 trace smoke:

```text
/data1/luyifei/drla/outputs/p3_dream_traces/
musique_calibration_trace_smoke1_steps16_stride4_20260606

scope:
  1 calibration sample / single_full_info / 1 solver call
  dream_steps = 16
  max_tokens = 32
  snapshot_stride = 4

status:
  pass
  num_errors = 0
  trace events = [0, 0, 4, 8, 12, 15]
  event fields include trace_event_index and has_logit_stats
```

Current P3 D3 trace pilot:

```text
/data1/luyifei/drla/outputs/p3_dream_traces/
musique_calibration_trace_pilot2_steps32_stride4_20260606

scope:
  2 calibration samples
  conditions = single_full_info,textmas_matched
  dream_steps = 32
  max_tokens = 64
  snapshot_stride = 4

status:
  pass
  num_errors = 0
  generation rows = 4
  Dream trace calls = 8
  solver calls = 4
  evidence-agent calls = 4
  forbidden gold/alias fields in traces = false
```

Current P3 D3 hidden/state validation:

```text
/data1/luyifei/drla/outputs/p3_dream_traces/
musique_calibration_trace_hidden_summary_smoke1_steps16_stride4_20260606

status:
  pass
  hidden_capture = summary
  hidden module = model.layers.27
  no tensor refs written

/data1/luyifei/drla/outputs/p3_dream_traces/
musique_calibration_trace_hidden_tensor_smoke1_steps8_stride4_20260606

status:
  pass
  hidden_capture = suffix_tensor
  hidden_refs files = 8
  loaded tensor example shape = (16, 3584), dtype = torch.float16
```

Current P3 D3/D4 subset artifact:

```text
D3 trace:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_calibration_trace_subset20_steps64_stride4_hidden_summary_20260606

D4 frontier:
  /data1/luyifei/drla/outputs/p3_dream_readiness_frontiers/
  musique_calibration_trace_subset20_steps64_stride4_hidden_summary_frontier_20260606

scope:
  20 calibration samples
  conditions = single_full_info,textmas_matched
  dream_steps = 64
  max_tokens = 128
  snapshot_stride = 4
  hidden_capture = summary

trace status:
  pass
  generation rows = 40
  Dream trace calls = 80
  errors = 0
  max_peak_memory_gib = 15.387

frontier status:
  pass
  solver frontier rows = 40
  solver frontier events = 720
  hidden/logit coverage = 0.9444
  oracle correct-stable-before-final row rate = 0.5
```

Current P3 D3/D4 full calibration artifact:

```text
D3 queue:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_calibration_trace_full200_steps64_stride4_hidden_summary_20260606_queue

D3 merged trace:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_calibration_trace_full200_steps64_stride4_hidden_summary_merged_20260606

D4 frontier:
  /data1/luyifei/drla/outputs/p3_dream_readiness_frontiers/
  musique_calibration_trace_full200_steps64_stride4_hidden_summary_frontier_20260606

scope:
  200 calibration samples
  conditions = single_full_info,textmas_matched
  dream_steps = 64
  max_tokens = 128
  snapshot_stride = 4
  hidden_capture = summary

queue / merge status:
  queue status = pass
  shards = 20 / 20 completed
  merged generation rows = 400
  merged samples = 200
  merged Dream trace calls = 800
  duplicate row/call ids = 0
  missing trace ids = 0

frontier status:
  pass
  solver frontier rows = 400
  solver frontier events = 7200
  hidden/logit coverage = 0.9444
  single_full_info final primary = 0.59
  textmas_matched final primary = 0.465
  oracle correct-stable-before-final row rate = 0.5275
```

Current P3 D5 readiness student:

```text
superseded diagnostic:
  /data1/luyifei/drla/outputs/p3_dream_readiness_students/
  dream_step_readiness_student_v1_full200_seed20260606_20260606
  reason = initial frontier omitted hidden_summary payload, so hidden features
  were zeroed. Do not cite as the current D5 result.

corrected frontier:
  /data1/luyifei/drla/outputs/p3_dream_readiness_frontiers/
  musique_calibration_trace_full200_steps64_stride4_hidden_summary_frontier_with_hidden_20260606

current student:
  /data1/luyifei/drla/outputs/p3_dream_readiness_students/
  dream_step_readiness_student_v1_full200_with_hidden_seed20260606_20260606

SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/0vw4qvu08rajphqgllk64

training:
  CUDA/GPU = yes
  SwanLab mode = cloud
  valid_interval = 10
  global_step = 400
  best_step = 350
  best_checkpoint.pt = present
  last_checkpoint.pt = present

best checkpoint metrics:
  valid ready_auroc = 0.9181
  valid ready_accuracy_at_05 = 0.8458
  test ready_auroc = 0.7946
  test ready_accuracy_at_05 = 0.7306
  test final_match_auroc = 0.9997
  test prediction_change_auroc = 0.9997
```

Current P3 D5.5 readiness policy eval:

```text
artifact:
  /data1/luyifei/drla/outputs/p3_dream_readiness_policy_eval/
  dream_step_readiness_student_v1_full200_with_hidden_policy_eval_20260606

boundary:
  local-only
  no optimizer/backward/weight update
  no SwanLab run
  thresholds selected on validation split only
  test split is report-only

selected policy:
  ready_threshold = 0.05
  final_match_threshold = 0.7
  prediction_change_max = 1.0
  future_gain_max = 999.0

test result:
  final_accuracy = 0.50
  selected_accuracy = 0.50
  accuracy_drop_vs_final = 0.00
  mean_step_savings = 54.95 / 63 Dream steps
  halt_before_final_rate = 0.95
  paired bootstrap 95% CI for mean_step_savings = [50.525, 58.1]
  paired bootstrap 95% CI for accuracy_drop_vs_final = [0.0, 0.0]

scope note:
  This is the D5 internal train/valid/test split from calibration full200,
  not a new held-out TextMAS benchmark claim.
```

Current P3 D6 latent packet substrate:

```text
tensor trace queue:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_calibration_trace_textmas_matched200_steps64_stride4_hidden_tensor_20260606_queue

tensor trace merged:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_calibration_trace_textmas_matched200_steps64_stride4_hidden_tensor_merged_20260606

trace status:
  condition = textmas_matched
  rows / samples / traces = 200 / 200 / 600
  shards completed = 20 / 20
  failed shards = 0
  hidden tensor refs = 38400
  tensor shard storage = about 33G

packet artifact:
  /data1/luyifei/drla/outputs/p3_dream_latent_packets/
  dream_textmas_matched200_agent_ab_suffix_tensor_packets_v1_20260606

packet status:
  packet_version = p3_dream_packet_v1_suffix_tensor
  packet groups = 200
  packets = 400
  agent_a packets = 200
  agent_b packets = 200
  packet groups with both agents = 200
  missing refs / missing traces / forbidden key hits = 0 / 0 / 0
  mean selected step = 21.98
  tensor shape example = [128, 3584]
  tensor dtype = torch.float16

scope note:
  D5 policy was trained on solver readiness labels; applying it to evidence
  agent traces is a D6 packet step-selection heuristic, not a new evidence-agent
  readiness claim. The packet is ready for receiver/fuser experiments but does
  not by itself prove latent communication.
```

Current P3 D7 receiver/fuser status:

```text
V1 MSE fuser diagnostic:
  /data1/luyifei/drla/outputs/p3_dream_latent_fusers/
  dream_latent_fuser_v1_textmas_matched200_seed20260606_20260606

V1 SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/hg2otd5swqd3pzudh0k3b

V1 result:
  train controls show packet use
  valid/test controls do not pass
  test zero_packet MSE is better than matched MSE
  conclusion = overfit / average-state distillation diagnostic, not accepted D7

V2 contrastive fuser:
  /data1/luyifei/drla/outputs/p3_dream_latent_fusers/
  dream_latent_fuser_v2_contrastive_textmas_matched200_seed20260606_20260606

V2 SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/k07305m849l5jnjyijlk4

V2 training:
  objective = symmetric InfoNCE packet-to-solver latent alignment
  CUDA/GPU = yes
  SwanLab cloud = yes
  valid_interval = 10
  global_step = 800
  best_step = 780
  best_checkpoint.pt = present
  last_checkpoint.pt = present

V2 best checkpoint:
  valid top1_i2t = 0.71875
  valid top1_t2i = 0.65625
  test top1_i2t = 0.75
  test top1_t2i = 0.625

V2 controls:
  /data1/luyifei/drla/outputs/p3_dream_latent_fuser_controls/
  dream_latent_fuser_v2_contrastive_textmas_matched200_controls_20260606

V2 control result:
  valid matched top1 = 0.60
  valid shuffled_row top1 = 0.05
  valid zero_packet top1 = 0.05
  test matched top1 = 0.55
  test shuffled_row top1 = 0.00
  test zero_packet top1 = 0.05
  random top1 = 0.05 on 20-row valid/test splits

caveat:
  agent_swap remains close to matched, so V2 proves row-specific latent signal
  but not ordered agent-role sensitivity. This is a receiver/fuser alignment
  result, not yet final answer generation.

Raw latent-prefix generation diagnostic:
  /data1/luyifei/drla/outputs/p3_dream_latent_prefix_runs/
  dream_latent_prefix_eval_diag20_steps64_prefix8_20260606

raw-prefix setup:
  local-only, no SwanLab
  max_rows = 20
  max_tokens = 128
  dream_steps = 64
  prefix = raw D6 suffix tensors, 8 tokens per agent
  solver prompt = no upstream text messages

raw-prefix result:
  no_message primary = 0.00
  latent_matched primary = 0.00
  latent_shuffled_row primary = 0.00
  latent_agent_swap primary = 0.00
  latent_zero primary = 0.00
  all variants have token_f1_mean = 0.0333

interpretation:
  Directly prepending last-layer suffix hidden states to Dream input embeddings
  does not work. This is not evidence that latent communication has no signal;
  V2 proves row-specific latent alignment. It is evidence that the receiver
  needs an embedding-space soft-prefix adapter or native layer/KV integration.

V3 embedding-space soft-prefix adapter:
  /data1/luyifei/drla/outputs/p3_dream_soft_prefix_adapters/
  dream_soft_prefix_adapter_v1_textmas_matched200_seed20260607_20260607

V3 SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/1qoue1655x7820f73wvmm

V3 training:
  script = drla/scripts/p3_train_dream_soft_prefix_adapter.py
  objective = map D6 suffix tensors to Dream input-embedding soft prefix and
              optimize receiver answer-token CE
  CUDA/GPU = yes
  SwanLab cloud = yes
  valid_interval = 10
  global_step = 480
  best_step = 460
  best_checkpoint.pt = present
  last_checkpoint.pt = present

V3 best checkpoint loss-level result:
  valid matched_ce = 2.9495
  valid zero_ce = 6.1258
  test matched_ce = 2.7128
  test zero_ce = 5.7548
  test agent_swap_ce = 2.6983

V3 soft-prefix generation controls:
  /data1/luyifei/drla/outputs/p3_dream_soft_prefix_runs/
  dream_soft_prefix_eval_v1_best20_20260607

V3 generation result:
  no_message primary = 0.00, token_f1 = 0.0333
  soft_prefix_zero primary = 0.00, token_f1 = 0.0333
  soft_prefix_shuffled_row primary = 0.00, token_f1 = 0.0833
  soft_prefix_matched primary = 0.00, token_f1 = 0.1083
  soft_prefix_agent_swap primary = 0.00, token_f1 = 0.1083

interpretation:
  The soft-prefix adapter learns a non-zero loss-level conditioning signal and
  changes receiver outputs, but it fails the answer-generation gate and does
  not distinguish matched from agent-swap. Treat V3 as a negative/diagnostic
  receiver result; do not scale the same shallow input-prefix path as the next
  main experiment.

V4 layer-conditioned receiver:
  /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
  dream_layer_receiver_v1_textmas_matched200_seed20260607_20260607

V4 SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/0798gogc7xnpd3hiqeyya

V4 training:
  script = drla/scripts/p3_train_dream_layer_conditioned_receiver.py
  objective = condition generated/masked Dream hidden states on D6 packets via
              cross-attention residual adapters at layers [7,14,21,27]
  CUDA/GPU = yes
  SwanLab cloud = yes
  valid_interval = 10
  global_step = 480
  best_step = 480
  best_checkpoint.pt = present
  last_checkpoint.pt = present

V4 best checkpoint loss-level result:
  test matched_ce = 1.5232
  test zero_ce = 1.8341
  test shuffled_row_ce = 1.6898
  test agent_swap_ce = 1.4997

V4 20-row generation controls:
  /data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/
  dream_layer_receiver_eval_v1_best20_20260607
  matched primary = 0.10
  no_message / zero / shuffled_row / agent_swap primary = 0.00

V4 50-row generation controls:
  /data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/
  dream_layer_receiver_eval_v1_best50_20260607
  no_message primary = 0.08, token_f1 = 0.0893
  zero primary = 0.20, token_f1 = 0.2593
  shuffled_row primary = 0.18, token_f1 = 0.2293
  agent_swap primary = 0.18, token_f1 = 0.2493
  matched primary = 0.20, token_f1 = 0.2693

interpretation:
  V4 improves receiver generation over no-message, but 50-row controls show
  most gain survives zero/corrupted packets. Treat this as receiver-prior
  leakage, not packet-specific LatentMAS success. Do not run D8 from V4.

V5 corruption-aware layer receiver:
  /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
  dream_layer_receiver_v2_corruptaware_textmas_matched200_seed20260607_20260607

V5 SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/gotls7ez7sgvnzyv2vmha

V5 training:
  script = drla/scripts/p3_train_dream_layer_receiver_corruption_aware.py
  architecture = same as V4 layer-conditioned receiver
  objective = matched CE + margin against zero/shuffled-row/agent-swap CE
  CUDA/GPU = yes
  SwanLab cloud = yes
  valid_interval = 10
  global_step = 480
  best_step = 400
  best_checkpoint.pt = present
  last_checkpoint.pt = present

V5 best checkpoint loss-level result:
  test matched_ce = 2.1024
  test zero_ce = 5.7847
  test shuffled_row_ce = 2.1638
  test agent_swap_ce = 2.0692

V5 20-row generation controls:
  /data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/
  dream_layer_receiver_v2_corruptaware_eval_best20_20260607
  no_message primary = 0.00, token_f1 = 0.0333
  zero primary = 0.00, token_f1 = 0.0333
  matched primary = 0.00, token_f1 = 0.0500
  agent_swap primary = 0.00, token_f1 = 0.0500
  shuffled_row primary = 0.00, token_f1 = 0.0667

interpretation:
  V5 suppresses the zero-packet prior at CE level but removes the matched
  answer-generation gain. Treat this as a negative diagnostic; do not expand
  V5 to 50-row controls or D8. Next receiver repair should use answer-selection
  / reranking or a two-stage objective that preserves matched generation while
  enforcing corruption separation.

D7.10 text-interface virtual message receiver:
  /data1/luyifei/drla/outputs/p3_dream_text_interface_receivers/
  dream_text_interface_receiver_d710_v1_p96d1024_seed20260617_20260617

D7.10 SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/eb1evnoegho5ez5f6qvje

D7.10 training:
  script = drla/scripts/p3_train_dream_text_interface_receiver.py
  objective = map D6 packets to continuous virtual message prefixes at the
              AgentB solver interface, with TextMAS hidden/logit teacher
  CUDA/GPU = yes
  SwanLab cloud = yes
  valid_interval = 10
  global_step = 480
  best_step = 460
  best_checkpoint.pt = present
  last_checkpoint.pt = present

D7.10 best checkpoint loss-level result:
  valid matched_ce = 2.5406
  valid token_accuracy = 0.5683
  test matched_ce = 2.8319
  test token_accuracy = 0.5141
  test zero_margin = 0.0603
  test shuffled_row_margin = 0.0262
  test agent_swap_margin = -0.0024

D7.10 generation controls:
  /data1/luyifei/drla/outputs/p3_dream_text_interface_receiver_runs/
  dream_text_interface_receiver_d710_v1_best20_20260617
  no_message primary = 0.00, token_f1 = 0.0333
  text_interface_matched primary = 0.05, token_f1 = 0.1500
  text_interface_zero primary = 0.05, token_f1 = 0.1333
  text_interface_shuffled_row primary = 0.05, token_f1 = 0.1333
  text_interface_agent_swap primary = 0.05, token_f1 = 0.1583

D7.10 interpretation:
  The receiver learns a text-interface-shaped teacher signal, but the single
  correct row is shared by matched, zero, shuffled-row, and agent-swap. This is
  receiver prior, not packet-specific latent communication.

D7.11 packet-specific text-interface receiver:
  /data1/luyifei/drla/outputs/p3_dream_text_interface_receivers/
  dream_text_interface_receiver_d711_v1_packet_specific_seed20260617_20260617

D7.11 SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/fopwvqmgi8fqfhxodxruv

D7.11 training:
  script = drla/scripts/p3_train_dream_text_interface_receiver.py
  init_checkpoint = D7.10 best checkpoint
  objective = matched teacher alignment plus corrupted unlikelihood, logit
              contrast, hidden contrast, and CE margins against zero,
              shuffled-row, and agent-swap packets
  CUDA/GPU = yes
  SwanLab cloud = yes
  valid_interval = 10
  global_step = 480
  best_step = 230
  best_checkpoint.pt = present
  last_checkpoint.pt = present

D7.11 best checkpoint loss-level result:
  valid matched_ce = 2.4047
  valid token_accuracy = 0.5529
  valid zero_margin = 6.7841
  valid shuffled_row_margin = 0.0021
  valid agent_swap_margin = -0.0240
  test matched_ce = 2.7797
  test token_accuracy = 0.5281
  test zero_margin = 6.3762
  test shuffled_row_margin = 0.0303
  test agent_swap_margin = 0.0030

D7.11 generation controls:
  /data1/luyifei/drla/outputs/p3_dream_text_interface_receiver_runs/
  dream_text_interface_receiver_d711_v1_best20_20260617
  no_message primary = 0.00, token_f1 = 0.0333
  text_interface_matched primary = 0.00, token_f1 = 0.0667
  text_interface_zero primary = 0.00, token_f1 = 0.0000
  text_interface_shuffled_row primary = 0.00, token_f1 = 0.0667
  text_interface_agent_swap primary = 0.00, token_f1 = 0.0583

D7.11 interpretation:
  Packet-specific negative objectives suppress the zero-packet prior, but they
  also collapse matched answer generation. D7.11 is a partial negative
  diagnostic; do not run D8. The next D7 design must balance matched generation
  preservation with packet-specific separation, especially agent-swap/role
  semantics.

D7.12 balanced text-interface receiver:
  /data1/luyifei/drla/outputs/p3_dream_text_interface_receivers/
  dream_text_interface_receiver_d712_balanced_v1_seed20260617_20260617

D7.12 SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/fbuzdxnxs5ow5plfc64eo

D7.12 training:
  script = drla/scripts/p3_train_dream_text_interface_receiver.py
  init_checkpoint = D7.10 best checkpoint
  objective = stronger matched teacher alignment plus weaker corruption losses,
              negative-loss warmup, and capped-margin checkpoint selection
  CUDA/GPU = yes
  SwanLab cloud = yes
  valid_interval = 10
  global_step = 480
  best_step = 140
  best_checkpoint.pt = present
  last_checkpoint.pt = present

D7.12 best checkpoint loss-level result:
  valid matched_ce = 2.7577
  valid token_accuracy = 0.5867
  valid zero_margin = 0.6173
  valid shuffled_row_margin = 0.1005
  valid agent_swap_margin = 0.0059
  test matched_ce = 3.2509
  test token_accuracy = 0.5082
  test zero_margin = 0.5295
  test shuffled_row_margin = -0.0046
  test agent_swap_margin = 0.0203

D7.12 generation controls:
  /data1/luyifei/drla/outputs/p3_dream_text_interface_receiver_runs/
  dream_text_interface_receiver_d712_balanced_v1_best20_20260617
  best: matched/zero/shuffled_row/agent_swap primary = 0.05
  best: no_message primary = 0.00

  /data1/luyifei/drla/outputs/p3_dream_text_interface_receiver_runs/
  dream_text_interface_receiver_d712_balanced_v1_last20_20260617
  last: matched primary = 0.05
  last: zero primary = 0.00
  last: shuffled_row primary = 0.00
  last: agent_swap primary = 0.05

D7.12 interpretation:
  Balanced training avoids D7.11's matched-generation collapse but still does
  not pass the receiver gate. Best is prior-driven across all virtual-prefix
  controls. Last separates zero/shuffled-row, but agent_swap remains tied with
  matched. Under homogeneous evidence-agent roles, agent_swap should be treated
  as a symmetry/role diagnostic unless a later protocol makes A/B roles
  asymmetric; zero and shuffled-row remain the hard corruption controls.

D7.13 unified receiver-control audit:
  /data1/luyifei/drla/outputs/p3_dream_receiver_control_audits/
  dream_receiver_generation_control_audit_v7_d710_d711_d712_20260617

D7.13 script:
  drla/scripts/p3_audit_dream_receiver_generation_controls.py

D7.13 result:
  hard controls = no_message, zero, shuffled_row
  diagnostic controls = agent_swap
  V7 full200 hard_gate_pass = true on primary paired CI
  D7.10/D7.11/D7.12 text-interface hard_gate_pass = false

V7 paired primary deltas:
  matched - no_message = 0.180, CI = [0.120, 0.240]
  matched - zero = 0.120, CI = [0.065, 0.180]
  matched - shuffled_row = 0.035, CI = [0.005, 0.070]
  matched - agent_swap = 0.005, CI = [-0.010, 0.025]

D7.13 interpretation:
  V7 is the current strongest receiver under the revised hard-control taxonomy,
  but it is still not enough for D8 because matched-vs-shuffled is small,
  token-F1 CI crosses zero, and agent_swap remains a symmetry/role diagnostic.
```

```text
D7.14 locked held-out V7 receiver audit:
  /data1/luyifei/drla/outputs/p3_dream_receiver_control_audits/
  dream_receiver_generation_control_audit_v7_heldout800_20260617

D7.14 substrate:
  trace = /data1/luyifei/drla/outputs/p3_dream_traces/
    musique_heldout_trace_textmas_matched800_steps64_stride4_hidden_tensor_merged_20260617
  packets = /data1/luyifei/drla/outputs/p3_dream_latent_packets/
    dream_textmas_heldout800_agent_ab_suffix_tensor_packets_v1_20260617
  eval = /data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/
    dream_layer_receiver_v7_heldout800_merged_20260617

D7.14 result:
  hard controls = no_message, zero, shuffled_row
  diagnostic controls = agent_swap
  hard_gate_pass = false
  matched primary = 0.02500
  no_message primary = 0.02375
  zero primary = 0.02875
  shuffled_row primary = 0.02375
  agent_swap primary = 0.02375

D7.14 paired primary deltas:
  matched - no_message = +0.00125, CI = [-0.00750, +0.01000]
  matched - zero = -0.00375, CI = [-0.01000, +0.00125]
  matched - shuffled_row = +0.00125, CI = [-0.00375, +0.00750]
  matched - agent_swap = +0.00125, CI = [0.00000, +0.00375]

D7.14 interpretation:
  V7 calibration full200 does not transfer to locked held-out800. Matched only
  improves token-F1 over no_message; it does not beat zero or shuffled-row on
  primary score. Treat this as a negative/blocked D8 result and diagnose
  distribution shift, receiver prior, step-selection heuristic, and injection
  strength before claiming latent communication.
```

```text
D7.15 V7 failure localization:
  /data1/luyifei/drla/outputs/p3_dream_receiver_generalization_audits/
  dream_receiver_v7_d714_failure_localization_20260617

Interface finding:
  packet_mean_cos_to_textmas_last128:
    calibration = 0.61227
    held-out = 0.62397
  layer27 delta_to_hidden_norm_ratio:
    calibration = 0.04937
    held-out = 0.05064

Split-generalization finding:
  train hard_gate_pass = true
    matched = 0.24375, zero = 0.10000, shuffled_row = 0.18750
  valid hard_gate_pass = false
    matched = 0.00000, zero = 0.00000, shuffled_row = 0.00000
  test hard_gate_pass = false
    matched = 0.20000, zero = 0.15000, shuffled_row = 0.30000

D7.15 interpretation:
  Held-out failure is not explained by an obvious packet/TextMAS hidden
  distribution break. V7 full200 success is train-dominated and does not pass
  nontrain calibration. Any new receiver must pass a nontrain calibration gate
  before another locked held-out run.
```

```text
D7.16 compact selected-suffix train2000 substrate:
  trace = /data1/luyifei/drla/outputs/p3_dream_traces/
          musique_train2000_trace_textmas_matched_selected_suffix_tensor_merged_20260617
  packets = /data1/luyifei/drla/outputs/p3_dream_latent_packets/
            dream_textmas_train2000_selected_suffix_tensor_packets_20260617
  trace rows / calls = 2000 / 6000
  packet groups / packets = 2000 / 4000
  missing refs / missing traces / forbidden key hits = 0 / 0 / 0
  mean selected step = 35.31025

D7.16 train2000 receiver:
  /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
  dream_layer_receiver_d716_train2000_v7init_zeroshuf_seed20260617

D7.16 SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/7134z0gui6w8jek33rdt0

D7.16 training:
  init = V7 V4-initialized zeroshuf best checkpoint
  train / valid / test = 1600 / 200 / 200
  valid_interval = 10
  best_step = 600
  best_checkpoint.pt = present
  last_checkpoint.pt = present
  valid matched_ce = 2.5632873698417096
  valid zero_ce_margin = 2.8676398239191623
  valid shuffled_row_ce_margin = 0.03806205857545164

D7.16 checkpoint-defined valid generation:
  run = /data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/
        dream_layer_receiver_d716_train2000_v7init_zeroshuf_valid200_20260617
  audit = /data1/luyifei/drla/outputs/p3_dream_receiver_control_audits/
          dream_receiver_generation_control_audit_d716_train2000_valid200_20260617
  hard_gate_pass = false
  primary no_message / matched / zero / shuffled_row / agent_swap =
    0.055 / 0.040 / 0.045 / 0.035 / 0.040
  matched - no_message primary_delta = -0.015, CI = [-0.040, +0.005]
  matched - zero primary_delta = -0.005, CI = [-0.035, +0.025]
  matched - shuffled_row primary_delta = +0.005, CI = [0.000, +0.015]

D7.16 interpretation:
  Larger compact packet data and V7 initialization improve teacher-forced
  CE/margin metrics, but not sampled denoising generation. Matched packet does
  not become a stable answer source; prediction-similarity audit shows matched
  and agent_swap have 70% identical predictions and 100% identical primary
  scores. Do not run held-out or D8 from this checkpoint. The next receiver
  needs an inference-aligned objective/injection repair, not just more steps of
  the same CE objective.

D7.17 denoising sensitivity audit:
  script = /data1/luyifei/drla/drla/scripts/
           p3_audit_dream_receiver_denoising_sensitivity.py
  smoke = /data1/luyifei/drla/outputs/p3_dream_receiver_sensitivity_audits/
          dream_receiver_d716_valid2_steps8_smoke_20260618
  valid50 = /data1/luyifei/drla/outputs/p3_dream_receiver_sensitivity_audits/
            dream_receiver_d716_valid50_steps64_max128_20260618
  split = checkpoint-defined valid
  rows / steps / max_tokens = 50 / 64 / 128
  step-control records = 12800
  matched_shared_state primary = 0.020
  matched vs no_message top1_disagree / transfer_disagree =
    0.008185529597103596 / 0.006093749925494194
  matched vs zero top1_disagree / transfer_disagree =
    0.008204307057021652 / 0.007031249916180969
  matched vs shuffled_row top1_disagree / transfer_disagree =
    0.002695536487735808 / 0.0018229165952652693
  matched vs agent_swap top1_disagree / transfer_disagree =
    0.0016112422474543564 / 0.0012499999348074197

D7.17 interpretation:
  D7.16 fails below the final scorer: matched packets almost never alter the
  token that Dream writes during denoising. The current receiver can learn
  teacher-forced CE/margins, but its online perturbation is too weak or too
  poorly targeted to affect transfer decisions. Do not continue by simply
  increasing steps of the same answer-token CE objective; the next receiver
  needs trajectory-level/guidance-style or stronger injection training.

D7.18-v1 denoising-aligned screen:
  script = /data1/luyifei/drla/drla/scripts/
           p3_train_dream_layer_receiver_denoising_aligned.py
  train = /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
          dream_layer_receiver_d718_screen200_denoising_aligned_seed20260618
  SwanLab = https://swanlab.cn/@Lyfff/drla-mvp/runs/5f7ynp5opf6j61cqpbxyy
  sensitivity = /data1/luyifei/drla/outputs/p3_dream_receiver_sensitivity_audits/
                dream_receiver_d718_screen200_valid50_steps64_max128_20260618
  valid50_generation = /data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/
                       dream_layer_receiver_d718_screen200_valid50_20260618
  valid50_audit = /data1/luyifei/drla/outputs/p3_dream_receiver_control_audits/
                  dream_receiver_generation_control_audit_d718_screen200_valid50_20260618

D7.18-v1 training result:
  best_step = 200
  valid matched_ce = 1.7278096367617877
  valid zero_gold_margin = 2.8303708081692456
  valid shuffled_row_gold_margin = 0.03000616851146333
  test matched_ce = 1.6313846607612956
  test zero_gold_margin = 2.898188375737518
  test shuffled_row_gold_margin = 0.041638789478165565

D7.18-v1 sensitivity / generation:
  valid50 matched_shared_state primary = 0.100
  transfer disagreement vs no_message = 0.010885416604578495
  transfer disagreement vs shuffled_row = 0.001302083283662796
  valid50 generation primary no_message / matched / zero / shuffled_row =
    0.08 / 0.12 / 0.12 / 0.10
  hard_gate_pass = false

D7.18-v1 interpretation:
  Partial-denoising training is directionally useful: it improves matched CE,
  no-message sensitivity, and matched valid50 generation. It is still not a
  row-specific latent communication receiver because zero ties matched on
  primary and shuffled-row transfer disagreement does not improve. Do not run
  held-out or D8 from this checkpoint. Next repair should make shuffled-row /
  row-specific packet binding a first-class objective.

D7.19 row-binding weighted screen:
  train = /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
          dream_layer_receiver_d719_screen200_row_binding_seed20260618
  SwanLab = https://swanlab.cn/@Lyfff/drla-mvp/runs/ub06nr5p8ddq3p2fgdenw
  sensitivity = /data1/luyifei/drla/outputs/p3_dream_receiver_sensitivity_audits/
                dream_receiver_d719_screen200_valid50_steps64_max128_20260618
  valid50_generation = /data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/
                       dream_layer_receiver_d719_screen200_valid50_20260618
  valid50_audit = /data1/luyifei/drla/outputs/p3_dream_receiver_control_audits/
                  dream_receiver_generation_control_audit_d719_screen200_valid50_20260618

D7.19 training result:
  best_step = 200
  valid matched_ce = 1.7381578401603024
  valid zero_gold_margin = 2.977695965440944
  valid shuffled_row_gold_margin = 0.06667653079697629
  test matched_ce = 1.6379975606159973
  test zero_gold_margin = 2.9921453844895587
  test shuffled_row_gold_margin = 0.06340354728978127

D7.19 sensitivity / generation:
  valid50 matched_shared_state primary = 0.100
  transfer disagreement vs no_message = 0.010833333283662795
  transfer disagreement vs shuffled_row = 0.0016666666232049464
  valid50 generation primary no_message / matched / zero / shuffled_row =
    0.08 / 0.10 / 0.12 / 0.10
  matched - shuffled_row primary_delta = 0.00 with 50 / 50 ties
  hard_gate_pass = false

D7.19 interpretation:
  Up-weighting shuffled-row margins gives a small teacher-forced row-binding
  improvement but does not create generation-time packet specificity. Matched
  ties shuffled-row exactly and is below zero on valid50 primary. Do not run
  held-out or D8 from this checkpoint; next work should revisit row-identity
  architecture, stronger fusion/injection, or trajectory-level guidance rather
  than just increasing the same CE/margin weights.
```

Historical P1/P2 entrypoints below remain valid only when explicitly
reproducing or auditing the CoLA line.

P1 readiness / halt evaluation uses the official Cola 8-task benchmark suite:

```text
lambada, mmlu, obqa, hellaswag, race, siqa, squad, story_cloze
```

P2 agent-agent communication must not use official8-only evidence as the main
MAS claim.  The active route is capability-gated, role-conditioned MAS.  Pure
evaluation/generation scripts must keep `swanlab_mode=disabled`; only scripts
with backward/optimizer training loops may create SwanLab cloud runs.

Post-Family1 current next-plan:

```text
/data1/luyifei/drla/docs/current/
P2_Locked_Complete_Execution_Scheme_2026-06-01.md

/data1/luyifei/drla/docs/current/
P2_Post_Family1_Complete_Execution_Plan_2026-06-01.md

Branch B Family 1 is stopped. The locked route is Phase C true MAS
benchmark/protocol validation, then Phase A CoLA substrate/interface adaptation,
then Phase E CoLA TextMAS vs LatentMAS. Do not execute the historical Planned
P2-D/E items below as the current next experiment. The only allowed early work is
docs/artifact indexing, existing-result aggregation, manifest/scorer/dry-inspect
tools, and code hygiene that does not run held-out, training, fuser/adapter, or
P2 main tables.
```

## Active Scripts

- `prepare_cola_official_benchmarks.py`: prepares the official Cola 8-task benchmark JSONL files under `/data1/luyifei/Cola-DLM/code/generate_task_data`.
- `collect_cola_block_traces.py`: runs official Cola block-wise generation and records per-block raw latent shards, decoder probe stats, EOS/im_end probe stats, generated text so far, answer-stability signals, and local `metrics.jsonl`. It has no training loop and only accepts `swanlab_mode=disabled`. Default output is score-ready under `/data1/luyifei/drla/outputs/cola_block_traces/tasks_block_trace`.
- `merge_cola_block_trace_segments.py`: merges OOM/resume trace segments into one score-ready root, validating generation id order and trace sample ids. Use it for full prepared splits when one task is collected in multiple segments.
- `eval_cola_benchmarks.py`: wraps official Cola `scripts/acc_calc.py`, parses the 8-task CSV, and writes local summaries. It has no training loop and only accepts `swanlab_mode=disabled`.
- `prepare_cola_p2_candidate_benchmarks.py`: prepares P2 capability-gate JSONL files for ARC-E, ARC-C, GSM8K, MBPP+, HumanEval+, GPQA-Diamond, and MedQA. Current pinned sources are `allenai/ai2_arc`, `openai/gsm8k`, `evalplus/*`, `hendrydong/gpqa_diamond_mc`, and `GBaker/MedQA-USMLE-4-options`.
- `prepare_cola_p2_official8_role_candidates.py`: safe Branch-B data-prep script. It converts existing official CoLA benchmark JSONL files into the normalized P2 gate schema for official8-compatible role-candidate triage. Default tasks are `obqa,mmlu,race,hellaswag,siqa,story_cloze`; `squad` and `lambada` are supported but excluded by default. It performs no generation, no training, no held-out inspection, and creates no SwanLab run. Current full artifact is `/data1/luyifei/drla/outputs/p2_benchmark_redesign/official8_role_candidates_20260601` with 33296 rows.
- `build_cola_p2_locked_splits.py`: builds deterministic calibration/held-out partitions for P2-D2. Prompt/protocol repair may use calibration rows only; held-out rows are locked for later capability gates and P2 main-table evaluation.
- `run_cola_p2_capability_gate.py`: local-only CoLA generation/evaluation gate for candidate P2 benchmarks. It runs single-solver and Planner→Critic→Refiner→Solver text-MAS protocols, writes `generations.jsonl`, `metrics.jsonl`, `task_summary.csv`, and `summary.json`, and never logs to SwanLab because it has no training loop. Code tasks must use `--enable-code-execution` for main gate claims; syntax-only mode is only a pre-gate. `--prompt-variant generic_v1|cola_fewshot_v1|answer_state_v1|answer_state_structured_v1|role_plan_ignore_v1` is for calibration-only prompt repair unless a variant later passes held-out gate. `--single-prompt-variant` and `--role-prompt-variant` may override it when single baseline and Role TextMAS need different prompt contracts; this must be recorded in the output config and should not be used to cite held-out results unless the exact split-prompt protocol is locked first.
- `aggregate_cola_p2_capability_gate.py`: local-only post-processor for completed P2 capability-gate summaries. It writes task-level and mode-level CSVs plus aggregate `summary.json` for document updates and paper-table triage, including `single_prompt_variant` and `role_prompt_variant` when present.
- `run_cola_p2_official8_native_single_gate.py`: local-only Branch-B alignment audit. It recovers raw official8 rows from `/data1/luyifei/Cola-DLM/code/generate_task_data`, groups by native `task_name`, calls Cola's original `apply_prompt_template`, and scores with acc_calc-style choice-text / first-segment preprocessing. It never runs Role TextMAS, never touches held-out unless explicitly pointed there for a locked gate, and never logs to SwanLab.
- `rescore_cola_p2_official8_native_single_gate.py`: local-only post-processor for native single generations. It recomputes official acc_calc-style scores without rerunning GPU generation; use it when scoring logic changes.
- `audit_cola_p2_protocol_repair_failures.py`: local-only paired failure taxonomy for P2-D3 prompt/protocol repair. It compares single-solver and Role TextMAS outcomes on the same calibration examples, writes paired role-help/role-harm/parser-failure summaries, and must not be run on held-out during prompt repair.
- `fetch_p2_phase_c_hf_rows.py`: local-only Phase C data-source row fetcher. It uses HuggingFace datasets-server `/rows` with seeded multi-block offsets, writes source `rows.jsonl`, `fetch_manifest.json`, `summary.json`, and `metrics.jsonl`. It does not construct benchmarks, run models, train, inspect held-out generations, or create SwanLab runs.
- `prepare_p2_phase_c_evidence_split_qa_records.py`: local-only evidence-split QA record builder for HotpotQA / MuSiQue / 2Wiki-style JSONL rows. It creates manifest-ready `p2_phase_c_manifest_v0` sample records with split agent private views, offline scoring, leakage-audit booleans, and shortcut-risk summary. It does not download data, run models, train, or create SwanLab runs.
- `build_p2_phase_c_control_inputs.py`: local-only Phase C control-input packager. It expands a validated manifest into `single_q_only`, `single_full_info`, `textmas_matched`, `textmas_no_message`, `textmas_shuffled_message`, `textmas_wrong_evidence_or_wrong_shard`, and `textmas_compressed_state` online inputs, writes prompt previews, and runs the leakage auditor. It does not call models, train, inspect held-out generations, or create SwanLab runs. Current default prompt contract is `p2_phase_c_evidence_split_v1_strict_wrong_evidence`; the wrong-evidence condition gives all evidence agents private shards from a non-self control sample. Older v0 packages that replaced only one shard are weak-control diagnostics, not admission artifacts.
- `preflight_p2_phase_c_text_agent_run.py`: local-only Phase C/Phase A text-agent preflight. It checks manifest/control consistency, estimates solver/evidence-agent chat calls, records OpenAI-compatible, local-transformers, or `cola_dlm` readiness, and writes `run_plan.json`, `summary.json`, and `metrics.jsonl`. It never calls models, trains, inspects held-out generations, or creates SwanLab runs.
- `run_p2_phase_c_text_agents.py`: local-only Phase C/Phase A text-agent runner. It consumes control inputs, runs sequential evidence-agent -> solver flows through an OpenAI-compatible chat/completions endpoint, `--provider local_transformers`, or `--provider cola_dlm`, scores final answers offline, and writes `generations.jsonl`, `condition_metrics.csv`, `summary.json`, and `metrics.jsonl`. It never trains or logs to SwanLab. `--selfcheck --provider mock_selfcheck` is only a toy routing/scoring check and must not be cited as an experiment. `--resume` restores existing `generations.jsonl` and agent-message cache for long calibration runs; if a disk-full or process interruption leaves a truncated final JSONL line, resume rewrites a clean prefix before appending. Local-transformers runs are real model outputs, but small local smokes are wiring checks, not Phase C admission evidence. Qwen-style thinking mode is disabled by default through `enable_thinking=False`; pass `--local-enable-thinking` only for diagnostics where `<think>` content is intentionally part of the protocol. `cola_dlm` runs use official CoLA VAE/DiT generation under the same Phase C online inputs/scorer and are Phase A substrate/interface diagnostics unless run at locked gate scale. Supported CoLA prompt styles are `chat_join`, `plain_qa_v1`, and `squad_template_v1`; `--cola-dit-lora-path` loads a trained DiT LoRA adapter, `--cola-noise-seed` defaults to deterministic seed 66, and `--prediction-extraction-mode first_segment` is the CoLA-style parser that cuts generated `<|endoftext|>` / `<|im_end|>` boundaries. 2026-06-05 MuSiQue prompt-only smokes with these styles are negative diagnostics, while the same local weights pass an official SQuAD 20-sample sanity at 5/20 primary matches.
- `build_p2_phase_a_cola_interface_sft.py`: local-only Phase A SFT data builder. It uses MuSiQue calibration manifest plus admitted Qwen3-8B-FP8 calibration TextMAS generations to create CoLA interface SFT pairs for `solver_full_info`, `solver_textmas_matched`, and `evidence_agent_teacher`. It writes `sft_pairs.jsonl`, sample-level `train.jsonl` / `valid.jsonl`, `summary.json`, and `metrics.jsonl`; it does not train, call models, inspect held-out data, or create SwanLab runs. Current artifact: `/data1/luyifei/drla/outputs/p2_phase_a_cola_interface_sft/musique_calibration_qwen_teacher_v1_20260605`.
- `build_p2_phase_a_candidate_answer_sets.py`: local-only Phase A candidate-answer builder. It extracts answer candidates from evidence text only, labels them with gold/aliases only for offline coverage/scoring, and writes `candidates.jsonl`, `summary.json`, and `metrics.jsonl`. It must not inject gold as a candidate, inspect held-out during repair, train, call models, or create SwanLab runs.
- `train_p2_phase_a_candidate_answer_selector.py`: local-only sklearn candidate selector diagnostic. It trains logistic or HistGradientBoosting selectors over evidence-derived candidate features to test whether shallow ranking can pick the gold answer from the current candidate pool. It does not run deep-learning optimization, call models, use held-out data, or create SwanLab runs. Current best shallow calibration result is qtype logistic top128 at `selected_primary=0.175`, below the locked `0.20` solver floor.
- `run_p2_phase_a_candidate_answer_llm_selector.py`: local-only semantic candidate-selector diagnostic. It uses a capable local LLM such as Qwen3-8B-FP8 to choose a short answer from evidence-derived candidates plus online evidence/question, then scores offline. It performs inference only, never training, never SwanLab, and must not use held-out for prompt/objective selection. It supports deterministic `--num-shards` / `--shard-index` and `--progress-interval` for long train-source teacher generation. The 2026-06-06 calibration200 top128 run reached `selected_primary=0.445`, showing the candidate interface is semantically usable even though shallow selectors fail. The completed train10k top128 all10 aggregate reached `selected_primary=0.4336`, `selected_exact_match=0.3893`, `selected_token_f1=0.4941`, `oracle_coverage_kept=0.8012`; it is allowed only as nonheldout teacher evidence for constrained short-answer CoLA repair, not as a CoLA or Phase E result.
- `aggregate_p2_phase_a_candidate_answer_llm_selector.py`: local-only semantic selector shard aggregator. It merges one or more `predictions.jsonl` files or directories, deduplicates by sample id, recomputes offline metrics, and writes aggregate `predictions.jsonl`, `summary.json`, and `metrics.jsonl`. It does not call models, train, inspect held-out data, or create SwanLab runs.
- `build_p2_phase_a_candidate_constrained_sft.py`: local-only Phase A candidate-constrained short-answer SFT builder. It reads the nonheldout MuSiQue train manifest, evidence-derived candidate answers, and the completed Qwen semantic selector teacher to emit CoLA-compatible short-answer pairs. Online prompts contain only question, full online evidence, and candidate text/rank/rule/source metadata; gold/aliases and teacher correctness are offline-only filtering/statistics. Current full artifact: `/data1/luyifei/drla/outputs/p2_phase_a_cola_interface_sft/musique_candidate_constrained_short_answer_train10000_top128_qwen_teacher_20260606` with 12348 pairs (`solver_candidate_gold_covered=8012`, `solver_candidate_teacher_correct=4336`), prompt token p95 about 4577 and block p95 about 287 under the official CoLA tokenizer/block size.
- `train_p2_phase_a_cola_dit_lora.py`: Phase A deep-learning training script for official CoLA DiT LoRA interface adaptation. It freezes the official CoLA VAE, trains only LoRA parameters on block-level Flow-Matching targets from calibration SFT pairs, appends target EOS by default, requires CUDA/GPU and SwanLab cloud, enforces `valid_interval <= 10`, and writes local `metrics.jsonl`, `best_checkpoint.pt`, `last_checkpoint.pt`, `best_adapter/`, and `last_adapter/`. Current positive calibration adapters are the solver adapter `/data1/luyifei/drla/outputs/p2_phase_a_cola_dit_lora/musique_solver_interface_lora_v1_epoch3_eos_20260605` and the evidence-agent adapter `/data1/luyifei/drla/outputs/p2_phase_a_cola_dit_lora/musique_evidence_agent_lora_v1_from_solver_epoch2_20260605`. Together they pass the 200-sample MuSiQue calibration Role TextMAS gate at `/data1/luyifei/drla/outputs/p2_phase_c_text_agent_aggregates/musique_cola_dlm_dual_lora_evidence_epoch2_solver_epoch3eos_squadtemplate_v1_firstseg_seed66_calibration_full200_20260605`, but fail the identical 800-sample locked held-out gate at `/data1/luyifei/drla/outputs/p2_phase_c_text_agent_aggregates/musique_cola_dlm_dual_lora_evidence_epoch2_solver_epoch3eos_squadtemplate_v1_firstseg_seed66_heldout800_20260605` because `single_full_info=0.0875 < 0.2`; do not start Phase E from this checkpoint and do not use held-out for further tuning.
- `train_p2_phase_a_cola_latent_candidate_ranker.py`: Phase A deep-learning trainer for a frozen-CoLA-VAE latent candidate-answer ranker. It encodes only online question/evidence/candidate text into CoLA latent representations, trains a lightweight ranker, requires CUDA/GPU and SwanLab cloud, enforces `valid_interval <= 10`, writes `metrics.jsonl`, `best_checkpoint.pt`, `last_checkpoint.pt`, and evaluates the saved best checkpoint for final summary metrics. It supports historical `--interaction-mode pooled` and the current `--interaction-mode late_maxsim`, which adds ColBERT-style candidate-token to context-token MaxSim features. The first top128 step500 schema-v2 pooled screen at `/data1/luyifei/drla/outputs/p2_phase_a_cola_latent_candidate_ranker/musique_top128_step500_schema_v2_seed20260606_20260606` reached best-checkpoint calibration200 `selected_primary=0.095`; the matched late-maxsim screen at `/data1/luyifei/drla/outputs/p2_phase_a_cola_latent_candidate_ranker/musique_top128_step500_latemaxsim_seed20260606_20260606` improved to `0.120`, still below the shallow qtype selector `0.175` and Qwen semantic selector `0.445`. This is useful evidence that token-level interaction helps but does not pass Phase A.
- `eval_p2_phase_a_cola_latent_candidate_ranker.py`: local-only checkpoint evaluator for CoLA-latent candidate rankers. It restores the saved split/config, evaluates `best_checkpoint.pt` or `last_checkpoint.pt` on nonheldout valid plus calibration candidates, writes local `metrics.jsonl`, predictions, and `summary.json`, and must run with `--swanlab-mode disabled`.
- `aggregate_p2_phase_c_text_agent_results.py`: local-only Phase C result aggregator and admission gate. It computes condition metrics, paired bootstrap confidence intervals, and the locked criteria for `single_full_info > single_q_only` and `textmas_matched > no_message/shuffled/wrong_evidence`. It does not run models, train, inspect held-out generations, or create SwanLab runs.
- `build_p2_phase_c_manifest.py`: local-only Phase C manifest packager. It reads normalized sample records JSONL, writes `manifest.json`, `samples.jsonl`, `summary.json`, and `metrics.jsonl`, then runs `validate_p2_phase_c_manifest.py`. It does not download data, generate samples, run models, or create SwanLab runs.
- `validate_p2_phase_c_manifest.py`: local-only Phase C manifest hygiene checker. It validates the `p2_phase_c_manifest_v0` schema, split presence, required baselines, agent private-view fields, scorer fields, and explicit leakage-audit booleans. It writes `summary.json` and `metrics.jsonl`, does not run models, and must not create SwanLab runs.
- `inspect_p2_phase_c_dataset_fields.py`: local-only Phase C data-source dry inspector. It reads local JSON/JSONL preview files, writes field path/type/length summaries, record hashes, source/license metadata, `summary.json`, and `metrics.jsonl`. It does not download data, construct manifests, run models, tune prompts, inspect held-out generations, train, or create SwanLab runs.
- `selfcheck_p2_phase_c_scorers.py`: local-only scorer self-check for Phase C QA and structured scorers. It uses hardcoded toy assertions only, writes `summary.json` and `metrics.jsonl`, and must not be cited as an experiment.
- `audit_p2_phase_c_run_leakage.py`: local-only run-artifact leakage auditor for future Phase C protocol runs. It checks `generations.jsonl` online-input fields against the manifest, blocks explicit gold/scorer/full-evidence fields in forbidden conditions, requires non-self control sample ids for shuffled/wrong-message controls, and writes `leakage_audit.json`, `summary.json`, and `metrics.jsonl`.
- `build_cola_readiness_frontier.py`: uses official scorer rules to build offline oracle readiness/frontier labels from per-block traces.
- `train_cola_readiness_model.py`: trains the multi-signal readiness/halt model over raw latent blocks plus process/probe/stability features. It writes SwanLab metrics, local `metrics.jsonl`, `best_checkpoint.pt`, and `last_checkpoint.pt`.
- `train_cola_latent_halt_student.py`: trains Phase P1 `LatentHaltStudent-v1`. Decoder/scorer/stability fields are teacher targets only; online inputs are raw latent prefix blocks plus selected latent/process features, with CUDA/GPU, SwanLab cloud, local `metrics.jsonl`, and best/last checkpoints. It supports explicit teacher-loss weights, `--selection-metric`, `--process-feature-mode full|no_block_budget`, `--process-interaction-mode process_token|film|trajectory_token`, `--readout-context-mode none|last_process_query`, `--use-completion-risk`, `--use-empty-answer-risk`, `--use-answer-format-risk`, `--use-answer-identity-stability`, and P0 teacher targets via `--readiness-target-mode oracle_frontier|p0_teacher_halt|p0_teacher_action`. `trajectory_token` keeps the normal process token and appends a per-block trajectory token built from pooled block state, previous-block delta, and process state before causal inter-block attention. Current selection metrics include `readiness_prediction_change_completion_contentful_mean_auroc`, `readiness_prediction_change_completion_empty_mean_auroc`, `readiness_prediction_change_completion_format_mean_auroc`, `readiness_prediction_change_completion_identity_mean_auroc`, and `readiness_prediction_change_completion_empty_identity_mean_auroc` for boundary/identity diagnostics. The full official8 `empty_answer_risk + answer_identity_stability` run is a negative result; do not treat the combined metric as a default without the matching aggregate evidence.
- `eval_cola_latent_halt_student.py`: evaluates a `LatentHaltStudent-v1` checkpoint as a student-only halt policy. It calibrates thresholds on valid, reports test accuracy-cost, saves threshold sweeps, student scores, and halt decisions, and audits answer-text mismatch versus final/prediction-stability outputs. It has no training loop and only accepts `swanlab_mode=disabled`. `--require-zero-calibration-mismatch` adds a stricter valid-set text-stability calibration guard on top of loss calibration. `--max-calibration-mismatches` / `--max-calibration-mismatch-rate` provide a softer risk cap that can keep zero-loss policies from selecting overly aggressive low-block thresholds. `--calibration-boundary-risk-penalty` changes threshold selection from pure min-block to `avg_blocks + penalty * boundary_risk_slack`, which is useful when finite target calibration over-selects permissive risk thresholds. `--max-calibration-samples-per-task` plus `--calibration-subsample-seed` runs deterministic target-calibration sample-cap diagnostics. `--empty-answer-risk-thresholds`, `--answer-format-risk-thresholds`, and `--answer-identity-stability-thresholds` are only active for checkpoints trained with the matching heads. `--calibration-scope per_task` requires the selected threshold to satisfy the same constraints on every calibration task instead of only on the pooled valid split.
- `aggregate_cola_latent_halt_student_loto.py`: aggregates P1 leave-one-task-out eval summaries with loss/mismatch rates, Wilson upper bounds, per-task risk buckets, and cross-seed recurrence. It has no training loop and must stay local-only.
- `aggregate_cola_latent_halt_student_subseed_loto.py`: aggregates nested calibration-subseed LOTO eval summaries such as `subseed*/leave_*_out_eval_*_test/summary.json`. It writes repeated-sample micro metrics, seed/task/subseed CSVs, and is local-only.
- `analyze_latent_halt_risk_control.py`: local-only formal risk-control audit for existing P1 threshold sweeps. It selects thresholds using validation-set Wilson upper bounds for loss/mismatch risk, then reports held-out official8/cross-seed trade-offs. Use `--allow-legacy-swanlab-eval` only for historical eval summaries created before the local-only eval rule.
- `build_cola_agent_latent_comm_packets.py`: builds sanitized P2 Agent A -> Agent B latent communication packets from locked P1 halt decisions and latent refs. It supports `cola_agent_latent_comm_v1` and `cola_agent_latent_comm_v2`; v2 adds `communication_boundary`, `prefix_contract`, and `agent_b_contract` for same-substrate handoff. It has no training loop and must stay local-only; online packet fields must exclude decoded text, token ids, scorer outputs, gold/target labels, and selected/final/prediction-stability correctness.
- `audit_cola_agent_latent_packet_distribution.py`: local-only P2-B audit. It loads v2 packet latent refs, checks packet/tensor consistency, native trace feature alignment, forbidden fields, and matched-vs-corrupted controls (`metadata_only`, `shuffle`, `cross_task`, `wrong_block`, `noise`, `rotation`). It has no training loop and must stay local-only.
- `audit_cola_latent_receiver_targets.py`: local-only P2-C target feasibility audit. It reads packet `audit_refs` and offline halt decisions to measure accept/defer label sparsity; these offline fields are target diagnostics only and must not become receiver inputs.
- `train_cola_latent_receiver.py`: trains the P2-C same-substrate latent receiver compatibility model. It uses balanced matched-vs-corrupted packet payload labels, requires CUDA/GPU and SwanLab cloud, writes local `metrics.jsonl`, `best_checkpoint.pt`, and `last_checkpoint.pt`, and never feeds decoded text, scorer outputs, correctness labels, or `control_type` to the model.
- `eval_cola_latent_receiver.py`: local-only P2-C checkpoint evaluator. It reconstructs the train/valid/test split, rebuilds matched/corrupted examples, loads a saved best checkpoint, and writes fresh compatibility metrics with `swanlab_mode=disabled`.
- `aggregate_cola_latent_receiver.py`: local-only aggregation for P2-C receiver ablations. It reads receiver `summary.json` files and writes a comparison CSV/summary; it must use `swanlab_mode=disabled`.
- `eval_cola_adaptive_halt.py`: sweeps adaptive halt thresholds and compares fixed-B, oracle halt, prediction-stability, learned threshold-only halt, aggressive readiness-or-stability, and stability-gated guarded policies. It has no training loop and only accepts `swanlab_mode=disabled`.
- `analyze_cola_halt_decisions.py`: reconstructs per-sample halt decisions from a saved eval summary and checkpoint, then writes `halt_decisions.jsonl`, `policy_comparison.csv`, `readiness_bins.csv`, and `summary.json`.
- `aggregate_cola_halt_decision_analysis.py`: aggregates per-task halt decision diagnostics across leave-one-task-out runs.
- `train_cola_continuation_risk_model.py`: trains a non-gold continuation-risk model for gating readiness halt before prediction stability. `--target-mode strict_prefix` detects prefix/incomplete answers; `--target-mode prediction_change` detects whether the current task-scored prediction will differ from the rollout prediction-stability reference.
- `eval_cola_risk_gated_halt.py`: evaluates readiness halt gated by continuation-risk probability. It has no training loop and only accepts `swanlab_mode=disabled`. `risk_threshold_selection_mode=min_blocks` is aggressive; `first_saving` is safer. `--readiness-threshold-values` lets validation jointly sweep readiness and risk thresholds. `--require-zero-calibration-loss` prevents valid-set loss/gain cancellation by requiring zero losses versus prediction-stability when such rows exist. `--require-contentful-prediction` adds a non-gold answer-shape guard against punctuation-only early halt. `--require-fragment-complete-prediction` guards non-stable decoded answer fragments such as pure-number prefixes, unfinished abbreviations, trailing initials, and short hyphen fragments. `--require-stable-single-choice` requires single `A`-`E` predictions to reach prediction stability before halt, addressing multiple-choice letter flips; `--stable-single-choice-max-block 1` or `2` narrows that guard to early blocks. Use `--stable-single-choice-guard-scopes off,1,2,3,all` to let validation select the guard scope jointly with risk threshold.
- `eval_cola_risk_gated_halt.py` also supports strict uncertainty calibration through `--entropy-max-values` and `--top-prob-min-values`. These sweep non-gold decoder confidence guards in addition to risk threshold; defaults preserve the old behavior.
- `analyze_cola_risk_gated_halt_decisions.py`: reconstructs sample-level risk-gated halt decisions and writes per-sample loss/gain diagnostics against fixed-final and prediction-stability.
- `aggregate_cola_risk_gated_halt.py`: aggregates same-task and leave-one-task-out risk-gated halt summaries plus common threshold sweeps.

## P2 Latent Communication Scripts

Current P2 canonical design:

```text
/data1/luyifei/drla/docs/current/P2_Latent_MAS_Communication_Implementation_Plan_2026-05-29.md
/data1/luyifei/drla/docs/current/P2_Locked_Complete_Execution_Scheme_2026-06-01.md
/data1/luyifei/drla/docs/current/P2_Post_Family1_Complete_Execution_Plan_2026-06-01.md
/data1/luyifei/drla/docs/current/P2_Phase_C_Benchmark_Protocol_Preparation_2026-06-01.md
```

Implemented:

- `build_cola_agent_latent_comm_packets.py`: local-only packet builder. It creates the v1 sanitized latent message substrate and the current P2-A v2 same-substrate single-handoff packets from locked P1 outputs, then writes packet summaries, `packet_schema.json`, `metrics.jsonl`, latent-ref checks, forbidden-field audits, and v2 field-coverage audits.
- `audit_cola_agent_latent_packet_distribution.py`: local-only packet distribution auditor. It validates P2-B packet/tensor consistency and writes `summary.json`, `distribution_stats.csv`, `control_stats.csv`, `ood_detection.csv`, `packet_examples.jsonl`, and `metrics.jsonl`.
- `audit_cola_latent_receiver_targets.py`: local-only target audit for P2-C. It shows whether accept/defer labels are dense enough to support a supervised receiver objective.
- `train_cola_latent_receiver.py`: P2-C compatibility receiver trainer. Full runs use CUDA/GPU, SwanLab cloud, `valid_interval<=10`, `metrics.jsonl`, `best_checkpoint.pt`, and `last_checkpoint.pt`.
- `eval_cola_latent_receiver.py`: P2-C local-only checkpoint evaluator for best-checkpoint rescoring.
- `aggregate_cola_latent_receiver.py`: P2-C local-only ablation aggregator.
- `run_cola_sequential_latent_mas.py`: P2-D local-only replay runner. It re-encodes the shared task context, consumes/replays Agent A latent packet blocks into official Cola DiT/VAE caches, continues generation under the receiver budget, and writes scorer-ready `tasks_<control>` directories plus `summary.json`, `metrics.jsonl`, `generations.jsonl`, and `control_comparison.csv`.
- `audit_cola_sequential_latent_mas.py`: P2-D local-only post-hoc audit. It combines replay generations, official scorer correct/wrong files, and offline P1 halt-decision references to report paired matched-vs-control score changes and answer-prefix fidelity. P1 reference fields are audit-only and must not become online receiver inputs.
- `build_cola_text_handoff_baseline.py`: P2-D local-only direct-answer handoff diagnostic. For the same replay subset, it materializes `text_selected`, `text_final`, and `text_prediction_stability` direct-handoff outputs from P1 halt-decision references into scorer-ready `tasks_text_*` directories. It does not feed text into Agent B and must not be used as the main B_text channel baseline.
- `train_cola_latent_answer_reader.py`: P2-D receiver-side latent answer-state reader trainer. It uses CUDA/GPU and SwanLab cloud, writes `metrics.jsonl`, `best_checkpoint.pt`, and `last_checkpoint.pt`, and trains a lightweight latent/process-feature encoder against P1 teacher selected_prediction text states.
- `eval_cola_latent_answer_reader.py`: P2-D local-only best-checkpoint evaluator for the latent answer reader. It restores the packet split and candidate answer pool, retrieves answer states, scores them with the official Cola scorer, and writes local `summary.json` / `metrics.jsonl`.
- `audit_cola_hierarchical_aggregation_potential.py`: P2-E local-only aggregation-potential audit. It groups the three locked seed66/67/68 same-sample sender packets and compares single-sender, text majority, simple latent-readiness rankers, and oracle any-correct upper bounds.
- `train_cola_hierarchical_latent_fuser.py`: P2-E hierarchical latent fuser trainer. Full runs use CUDA/GPU, SwanLab cloud, `valid_interval<=10`, `metrics.jsonl`, `best_checkpoint.pt`, and `last_checkpoint.pt`. It supports `--target-mode correct`, `--target-mode score`, `--target-mode task_aware_score`, and `--task-loss-weighting balanced`; all modes keep online inputs decoder-free and use decoded/scored text only as offline supervision.
- `eval_cola_hierarchical_latent_fuser.py`: P2-E local-only best-checkpoint evaluator with aggregate and per-task metrics for the hierarchical latent fuser.
- `train_cola_hierarchical_state_verifier.py`: P2-E multi-sender latent-state utility verifier. It predicts group-level `any_correct` and `best_score` from decoder-free latent/process/certificate inputs, uses SwanLab cloud for training, and writes `metrics.jsonl`, `best_checkpoint.pt`, `last_checkpoint.pt`, and summary best/final metrics.
- `eval_cola_hierarchical_state_verifier.py`: P2-E local-only calibration and input-ablation evaluator for the latent-state verifier. It restores the locked split, evaluates zero-latent / zero-process / zero-certificate ablations, fits valid-only post-hoc calibrators, and writes `summary.json`, `metrics.jsonl`, `ablation_metrics.csv`, and `calibration_report.json`.
- `eval_cola_hierarchical_state_policy.py`: P2-E local-only receiver-state policy audit. It combines the calibrated latent-state verifier with the score-target fuser v2, selects thresholds on valid, and reports held-out test gates against text-majority, task-prior, and global-prior controls. It writes `summary.json`, `metrics.jsonl`, `policy_metrics.csv`, `per_group_states.jsonl`, and `train_priors.json`.
- `train_cola_receiver_state_action_selector.py`: P2-E structured receiver-state action selector trainer. It consumes calibrated state/fuser/prior features, uses offline selected-score labels, requires CUDA/GPU and SwanLab cloud, and writes `metrics.jsonl`, `best_checkpoint.pt`, `last_checkpoint.pt`, and `test_predictions.jsonl`. It supports direct sender logits and residual-fuser logits.
- `eval_cola_receiver_state_action_selector.py`: P2-E local-only evaluator for action selector checkpoints. It rebuilds the locked split, restores feature normalization, selects thresholds only on valid, and reports held-out test aggregate/gated metrics.
- `train_cola_state_conditioned_latent_fuser.py`: P2-E state-conditioned sender-level latent fuser trainer. It initializes from score-target fuser v2, keeps raw sender-level latent/process/certificate inputs, adds calibrated state/fuser/prior side features, and learns a residual sender-logit delta. Training requires CUDA/GPU and SwanLab cloud.
- `audit_cola_request_more_latent_potential.py`: P2-E local-only audit for first-sender vs prefix2/prefix3 additional latent evidence and oracle/request-readiness upper bounds.
- `train_cola_request_more_policy.py`: P2-E first-sender request-more trainer. It predicts whether requesting remaining sender packets is useful; training requires CUDA/GPU and SwanLab cloud.
- `train_cola_post_request_selector.py`: P2-E anchor-aware post-request selector trainer. It selects among first/requested sender packets after a request action; training requires CUDA/GPU and SwanLab cloud.
- `train_cola_joint_request_select_policy.py`: P2-E joint request-and-select trainer. The request head sees only the first sender; the selector sees requested sender packets only after request; training requires CUDA/GPU and SwanLab cloud. It supports the original target025 checkpoint selection and `--checkpoint-selection-mode valid_rate_frontier`; the strict 2026-05-31 frontier audit is negative and should not replace the current canonical target025 aggregate.
- `eval_cola_joint_request_select_policy.py`: P2-E local-only checkpoint evaluator for any saved joint request/select checkpoint, including `last_checkpoint.pt`. It reconstructs valid/test predictions, selects request thresholds on valid, and writes local `summary.json`, `metrics.jsonl`, `policy_metrics.csv`, and prediction JSONL files.
- `audit_cola_joint_policy_calibration.py`: P2-E local-only calibration/risk-control audit for saved joint policy predictions. It selects request thresholds on valid, reports held-out test, writes `summary.json`, `metrics.jsonl`, `calibration_metrics.json`, and `risk_policy_metrics.csv`, and must use `swanlab_mode=disabled`.

Corrected P2-D channel-equivalent scripts:

- `build_cola_agent_channel_messages.py`: local-only builder for paired `A_raw_text_message_t` and `A_latent_packet_t` from the same Agent A trajectory/depth. It uses native trace `decode_text_so_far` at the selected block, not `selected_prediction`, for the text message.
- `run_cola_agent_b_channel_eval.py`: local-only Agent-B evaluator for the LatentMAS-aligned `B_none(empty input)`, `B_text(A_raw_text_message_t)`, `B_latent(A_latent_packet_t)`, and `B_corrupt(corrupted_latent_packet_t)` conditions under the same receiver budget and official scorer format. The default `--agent-b-input-contract message_only` gives B only Agent A's output/message; `--agent-b-input-contract shared_context` additionally gives B the original prompt and is diagnostic only. Latent blocks are loaded directly from packet shards, not re-encoded from text. The default `--score-output-scope receiver_only` makes the scorer see only Agent-B tokens generated after handoff; A text-message tokens and A latent replay decoded tokens are excluded from final `generate`. `--score-output-scope legacy_all_visible` exists only to reproduce historical diagnostics. Replay modes are `decode_and_emit`, `_cache_only` (VAE decoder cache + DiT cache, no replay text), `_dit_only_cache`, and `_vae_only_cache`.
- `merge_cola_agent_b_channel_eval_shards.py`: local-only merger for sharded Agent-B channel eval outputs. It validates expected messages/channels, duplicate keys, and writes a single `generations.jsonl` for aggregation.
- `aggregate_cola_channel_eval.py`: local-only post-hoc official scorer aggregate for Agent-B final-output score, paired win/loss/tie, cost, and decision-rule readouts. Gold/scorer outputs are not online Agent A/B inputs.
- `audit_cola_channel_projection_gap.py`: local-only paired audit for the decoder semantic-projection gap between decode-and-emit `latent_matched` and receiver-native/cache-only `latent_matched_cache_only`.
- `audit_cola_channel_protocol_boundaries.py`: local-only gate for P2-D Agent-B communication claims. It classifies eval roots as `receiver_only` pass or legacy/fail, and verifies that scorer-visible A text-message tokens and A replay blocks are zero.

Current P2-A locked v2 packet artifact:

```text
/data1/luyifei/drla/outputs/cola_agent_latent_comm/
p2_agent_latent_comm_v2_locked_seed66_67_68_split20260601_20260529

packets / latent refs / unique latent files:
  14940 / 27399 / 8850

missing latent files / forbidden decoder-eval fields:
  0 / 0

v2 coverage:
  communication_boundary = 14940 / 14940
  prefix_contract = 14940 / 14940
  agent_b_contract = 14940 / 14940
```

Current P2-B locked distribution audit artifact:

```text
/data1/luyifei/drla/outputs/cola_agent_latent_comm/
p2_agent_latent_comm_v2_distribution_audit_locked_seed66_67_68_split20260601_20260529

status:
  pass

packets audited / native aligned latent blocks:
  14940 / 27399

structural errors / forbidden fields / latent load errors:
  0 / 0 / 0

native alignment max_abs_diff:
  0.0

min corrupted-control pair-distance AUROC:
  1.0
```

Current P2-C target feasibility artifact:

```text
/data1/luyifei/drla/outputs/cola_agent_latent_comm/
p2_latent_receiver_target_audit_locked_seed66_67_68_split20260601_20260529

status:
  warn

offline accept / unsafe:
  14936 / 4

unsafe rate:
  0.0268%

naive all-accept accuracy:
  99.9732%
```

Current P2-C receiver compatibility artifact:

```text
/data1/luyifei/drla/outputs/cola_latent_receiver/
p2c_receiver_compat_bestckpt_eval_aggregate_seed20260529_20260529

best input mode:
  latent_process_certificate

best SwanLab run:
  50bumvfh2pgp0olw60jvr

test mean-control AUROC:
  0.9194

hardest listed control:
  shuffle AUROC = 0.6205

negative controls:
  envelope_only ~= 0.5003
  process_only ~= 0.5000
  certificate_only ~= 0.5000
```

Early P2-D sequential replay artifact:

```text
/data1/luyifei/drla/outputs/cola_sequential_latent_mas/
p2d_replay_eval_official8_5per_task_controls_20260529

scope:
  official8, 5 packets per task, 40 packets total
  matched, metadata_only, shuffle, cross_task, wrong_block, noise, rotation
  local-only, swanlab disabled, no training

official scorer:
  matched = 17.5%
  metadata_only = 20.0%
  wrong_block = 5.0%
  shuffle/cross_task/noise/rotation = 0.0%

offline answer-prefix fidelity:
  matched = 37.5%
  metadata_only = 27.5%
  noise = 7.5%
  shuffle/cross_task/wrong_block/rotation = 2.5%
```

Current P2-D direct-answer diagnostic and replay-only diagnostic:

```text
direct-answer diagnostic:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_eval_official8_5per_task_controls_20260529/text_handoff_baseline

official scorer:
  text_selected = 30.0%
  text_final = 30.0%
  text_prediction_stability = 30.0%

replay-only:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_only_eval_official8_5per_task_controls_20260529

matched replay-only:
  scorer accuracy = 17.5%
  answer-prefix fidelity = 37.5%
  replay-only vs native trace selected-block raw text = 60.0%
  native trace selected-block raw text vs P1 selected_prediction = 27.5%
```

Current expanded P2-D diagnostic:

```text
replay-only:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_only_eval_official8_20per_task_controls_20260529

replay+continue:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_continue_eval_official8_20per_task_controls_20260529

direct-answer diagnostic:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_only_eval_official8_20per_task_controls_20260529/text_handoff_baseline

official scorer:
  replay-only matched = 24.38%
  replay+continue matched = 24.38%
  replay+continue metadata_only = 21.88%
  corrupted latent controls = 0.00% to 1.88%
  text_selected = 25.00%

audit:
  selected_reference_accuracy = 25.0%
  matched official_prediction_agrees_selected = 65.0%
  matched correct_selected_preservation_rate = 80.0%
```

Planned:

- Fresh 50-per-task P2-D validation is now the main sequential replay evidence:

```text
replay-only:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_only_eval_official8_50per_task_seed20260530_controls_20260529

replay+continue:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_continue_eval_official8_50per_task_seed20260530_controls_20260529

scope:
  50 packets per official task
  400 packets total
  fresh selection seed = 20260530
  matched, metadata_only, shuffle, cross_task, wrong_block, noise, rotation
  local-only, swanlab disabled

official scorer:
  replay-only matched = 24.25%
  replay-only metadata_only = 0.00%
  replay-only corrupted controls = 0.00% to 3.75%
  replay+continue matched = 24.25%
  replay+continue metadata_only = 23.75%
  replay+continue corrupted controls = 0.25% to 4.25%
  text_selected = 23.50%

audit:
  duplicate-safe scorer correctness lookup is required because task ids/sample_keys repeat
  replay-only matched net wins = +97 vs metadata_only, +82 to +97 vs corrupted controls
  replay+continue matched net wins = +2 vs metadata_only, +80 to +96 vs corrupted controls
```

Next:

- Treat matched-vs-corrupted as the strong current P2-D evidence.
- Do not claim text superiority: the current `text_selected` rows are direct-answer diagnostics, not Agent-B text-message baselines, and metadata_only+continue is close to matched.
- Main next step is scaling the corrected LatentMAS-aligned Agent-B channel-equivalent evaluation: `B_none(empty input)`, `B_text(A_raw_text_message_t)`, `B_latent(A_latent_packet_t)`, and `B_corrupt(corrupted_latent_packet_t)` under `--agent-b-input-contract message_only --score-output-scope receiver_only`.
- `run_cola_sequential_latent_mas.py` supports `--receiver-context-mode full_prompt|empty_prompt`.
- Corrected channel-equivalent smoke, 2026-05-31:

```text
messages:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_channel_messages_smoke_1per_task_20260531

eval:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_smoke_1per_task_20260531

aggregate:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_smoke_1per_task_20260531/channel_eval_aggregate

scope:
  official8, 1 message per task, 8 channels, 64 final Agent-B generations
  swanlab_mode=disabled, CUDA generation, no training

official scorer smoke readout:
  latent_matched = 25.00% accuracy, mean score 0.5754
  text_raw_message = 25.00% accuracy, mean score 0.5799
  none = 25.00% accuracy, mean score 0.6059
  corrupted latent controls = 0.00% accuracy, mean score 0.1720-0.2005

interpretation:
  protocol path is executable and matched latent is well above corrupted
  controls, but 8 samples are smoke only. Do not treat this as a paper number.
```
- Legacy all-visible official8 50/task unique-sample diagnostic, 2026-05-31:

```text
messages:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_channel_messages_official8_50per_task_seed20260531_unique_20260531

sharded eval:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_official8_50per_task_seed20260531_unique_20260531_sharded6

merged eval:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_official8_50per_task_seed20260531_unique_20260531_merged

aggregate:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_official8_50per_task_seed20260531_unique_20260531_merged/channel_eval_aggregate

scope:
  official8, 50 unique sample_key per task, 400 messages
  8 channels, 3200 final Agent-B generations
  swanlab_mode=disabled, CUDA sharded generation, no training

official scorer:
  latent_matched = 23.50%, mean score 0.4850
  text_raw_message = 23.00%, mean score 0.5011
  none = 19.75%, mean score 0.4410
  corrupted controls = 0.00% to 4.25%, mean score 0.1699-0.2499

paired deltas:
  matched - corrupted score_delta = +0.2351 to +0.3151, all CI95 lower > +0.1959
  matched - none score_delta = +0.0440, CI95 [+0.0111, +0.0758]
  matched - text score_delta = -0.0161, CI95 [-0.0406, +0.0076]

interpretation:
  This table used the historical all-visible output scope, so A text-message
  tokens and A decoded latent replay tokens could be scored directly. It is a
  decodability/replay-output diagnostic only, not valid Agent-B communication
  evidence. The earlier non-deduped 50/task run had only 375 unique sample_key
  and is protocol-audit only.
```
- Receiver-native cache-only audit, 2026-05-31:

```text
artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_official8_50per_task_cache_only_seed20260531_unique_20260531_merged/channel_eval_aggregate

protocol:
  replay latent blocks update VAE/DiT KV cache
  replay blocks are not sampled/emitted into output text
  replay_blocks_decoded_to_text = 0
  same 400 unique messages as the formal decode-and-emit run

official scorer:
  latent_matched_cache_only = 1.50%, mean score 0.1916
  none = 19.75%, mean score 0.4410
  text_raw_message = 23.00%, mean score 0.5011
  corrupted cache-only controls = 0.75% to 3.25%, mean score 0.1559-0.2261

paired deltas:
  matched_cache_only - none score_delta = -0.2493, CI95 [-0.2851, -0.2152]
  matched_cache_only - text score_delta = -0.3094, CI95 [-0.3484, -0.2686]

interpretation:
  current positive P2-D evidence is decode-and-emit payload readability, not
  receiver-native no-text latent reasoning. Future receiver work should consume
  latent state natively instead of only replaying sender latent through decoder.
```
- Corrected receiver-only channel smoke, 2026-05-31:

```text
artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_smoke_receiver_only_1per_task_20260531/channel_eval_aggregate

protocol:
  --score-output-scope receiver_only
  scorer-visible A text message tokens = 0
  scorer-visible A replay blocks = 0

official scorer:
  none = 25.00%, mean_score 0.6059
  text = 12.50%, mean_score 0.2996
  latent_matched = 0.00%, mean_score 0.2082
  latent_matched_cache_only = 0.00%, mean_score 0.2082
  latent_matched_dit_only_cache = 0.00%, mean_score 0.2082
  latent_matched_vae_only_cache = 0.00%, mean_score 0.3373

interpretation:
  Historical all-visible decode-and-emit results are downgraded to decodability
  diagnostics. They must not be cited as Agent-B communication results.
```
- LatentMAS-aligned message-only receiver-only smoke after replay-EOS fix, 2026-05-31:

```text
artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_smoke_message_only_receiver_only_eosfix_1per_task_20260531/channel_eval_aggregate

protocol:
  --agent-b-input-contract message_only
  --score-output-scope receiver_only

code fix:
  replay EOS/im_end from Agent A no longer stops Agent B under receiver_only.
  Only B-generated stop tokens terminate receiver generation.

boundary audit:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_protocol_boundary_audit_message_only_eosfix_smoke_20260531
  status = pass
  scorer-visible A text message tokens = 0
  scorer-visible A replay blocks = 0

official scorer:
  none = 0.00%, mean_score 0.1527
  text = 0.00%, mean_score 0.2022
  latent_matched = 0.00%, mean_score 0.2457
  latent_matched_cache_only = 0.00%, mean_score 0.2478
  latent_matched_dit_only_cache = 0.00%, mean_score 0.2405
  latent_matched_vae_only_cache = 0.00%, mean_score 0.1559

paired smoke deltas:
  latent_matched - none score_delta = +0.0930
  latent_matched - text score_delta = +0.0435

interpretation:
  This is the canonical protocol smoke, not a quality claim. The formal
  official8 50/task result below supersedes it for channel-quality discussion.
```
- Formal LatentMAS-aligned P2-D 50/task result, 2026-05-31:

```text
artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_official8_50per_task_message_only_receiver_only_eosfix_seed20260531_unique_20260531_merged

aggregate:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_official8_50per_task_message_only_receiver_only_eosfix_seed20260531_unique_20260531_merged/channel_eval_aggregate

audit:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_protocol_boundary_audit_message_only_receiver_only_eosfix_50task_20260531

scope:
  official8, 50 unique sample_key per task
  400 messages, 11 channels, 4400 Agent-B generations
  duplicate_keys = 0
  missing_message_rows = 0
  protocol audit = pass

official scorer:
  latent_matched = 1.75%, mean_score 0.1874
  latent_matched_cache_only = 1.75%, mean_score 0.1912
  latent_matched_dit_only_cache = 1.25%, mean_score 0.1873
  latent_matched_vae_only_cache = 0.00%, mean_score 0.1395
  none = 0.00%, mean_score 0.1378
  text = 1.25%, mean_score 0.1795
  latent_cross_task = 0.25%, mean_score 0.1535
  latent_shuffle = 0.25%, mean_score 0.1663
  latent_wrong_block = 4.00%, mean_score 0.1950
  latent_noise = 0.00%, mean_score 0.1390
  latent_rotation = 0.00%, mean_score 0.1297

paired readout:
  matched - none score_delta = +0.0496, CI95 [+0.0340, +0.0659]
  matched - text score_delta = +0.0079, CI95 [-0.0082, +0.0259]
  matched - wrong_block score_delta = -0.0076, CI95 [-0.0286, +0.0140]
  matched - cache_only score_delta = -0.0038, CI95 [-0.0064, -0.0012]

interpretation:
  Valid receiver-only Agent-B communication evidence exists for marginal utility
  over empty input. Latent is competitive with text but does not significantly
  beat text. The all-corrupt gate fails because wrong_block is anomalously
  strong, so payload-specific robustness remains unresolved.
```
- Decoder semantic-projection gap audit, 2026-05-31:

```text
artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_projection_gap_official8_50per_task_seed20260531_unique_20260531

scope:
  same 400 unique official8 50/task messages
  paired latent_matched decode-and-emit vs latent_matched_cache_only

result:
  projection_score_gain = +0.2934, CI95 [+0.2551, +0.3320]
  projection_accuracy_gain = +22.00pp, CI95 [+17.75pp, +26.25pp]
```

- Direct DiT/VAE cache ablation smoke, 2026-05-31:

```text
artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_smoke_direct_dit_1per_task_20260531_v2/channel_eval_aggregate

scores:
  latent_matched = 25.00%, mean_score 0.5754
  latent_matched_cache_only = 0.00%, mean_score 0.2082
  latent_matched_dit_only_cache = 0.00%, mean_score 0.2082
  latent_matched_vae_only_cache = 0.00%, mean_score 0.3373

interpretation:
  DiT-only direct replay is executable but does not recover decode-and-emit
  utility on the 8-sample smoke. This is protocol evidence only.
```
- Message-only marginal-utility diagnostic:

```text
artifact:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_message_only_replay_only_official8_50per_task_seed20260531_controls_20260529

protocol:
  receiver_context_mode = empty_prompt
  receiver_budget_mode = fixed
  fixed_receiver_blocks = 0

official scorer:
  matched latent = 12.50%
  metadata_only = 0.00%
  corrupted controls = 0.00% to 2.00%
  text_selected = 26.25%

audit:
  matched net wins = +50 vs metadata_only, +42 to +50 vs corrupted controls
```

This shows latent packets alone carry recoverable signal, but current
message-only replay is far worse than the direct-answer diagnostic. The next
main experiment is not more direct-handoff scaling; it is channel-equivalent
Agent-B evaluation plus receiver-native no-text latent consumption.

Receiver-side latent answer reader diagnostic:

```text
training artifact:
  /data1/luyifei/drla/outputs/cola_latent_answer_reader/
  p2_answer_reader_full_seed20260529_20260529

SwanLab cloud run:
  x6yc77eedf77z27ego0ve

best checkpoint test:
  answer_key_top1 = 15.20%
  official_top1_accuracy = 10.61%
  selected_reference_accuracy = 22.11%
```

This confirms the negative boundary: latent packets are readable, but the
current lightweight answer reader is not better than the direct-answer
diagnostic and does not establish text-channel competitiveness.

P2-E hierarchical aggregation:

```text
potential audit artifact:
  /data1/luyifei/drla/outputs/cola_hierarchical_aggregation/
  p2e_aggregation_potential_locked_seed66_67_68_split20260601_20260529

potential audit:
  single_sender_first = 20.74%
  text_majority_selected = 21.55%
  best simple latent-state ranker = 21.39%
  oracle_any_selected_correct = 33.13%

learned fuser artifact:
  /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
  p2e_hierarchical_fuser_full_seed20260529_20260529

SwanLab cloud run:
  ljv0m43x48a49j1at6gx9

best checkpoint test:
  model_selected_accuracy = 20.74%
  single_sender_first_accuracy = 22.38%
  text_majority_selected_accuracy = 23.41%
  oracle_any_selected_accuracy = 34.70%
```

P2-E v1 is a useful negative result: multi-sender packets have oracle headroom,
but the current decoder-free latent fuser does not beat single-sender or text
majority baselines on held-out test.

P2-E score-target fuser v2:

```text
supervision audit:
  non-binary official-score sender predictions = 7764 / 14940 = 52.0%
  partial-best groups = 2570 / 4980 = 51.6%

learned fuser artifact:
  /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
  p2e_hierarchical_fuser_score_full_seed20260529_20260529

SwanLab cloud run:
  o5fjvuiqk82nk9c5hihn0

best checkpoint test:
  model_selected_accuracy = 23.41%
  single_sender_first_accuracy = 22.38%
  text_majority_selected_accuracy = 23.41%
  oracle_any_selected_accuracy = 34.70%
  model_mean_official_score = 0.3685
  single_sender_first_mean_official_score = 0.3553
  text_majority_mean_official_score = 0.3622
  oracle_best_selected_mean_official_score = 0.4951
```

This is the strongest P2-E result so far: score-target latent fusing beats the
single fixed sender and matches text majority on exact test accuracy, while
beating both on mean official score. It is still below oracle and uneven across
tasks, so do not claim robust latent-over-text superiority yet.

P2-E negative follow-ups:

```text
task-balanced v3:
  artifact:
    /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
    p2e_hierarchical_fuser_score_taskbalanced_full_seed20260529_20260529
  SwanLab cloud run:
    k2ujjjdrmcyzutwnwbyyf
  test:
    model_selected_accuracy = 22.38%
    model_mean_official_score = 0.3654

task-aware v4:
  artifact:
    /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
    p2e_hierarchical_fuser_taskaware_score_full_seed20260529_20260529_metricfix
  SwanLab cloud run:
    fdjwe4vfq70syq0oz9tro
  test:
    model_selected_accuracy = 22.18%
    model_mean_official_score = 0.3605
```

Both are below score-target v2. Keep v2 as the current P2-E best model.

P2-E latent-state utility verifier:

```text
artifact:
  /data1/luyifei/drla/outputs/cola_hierarchical_state_verifier/
  p2e_state_verifier_full_seed20260529_20260529

SwanLab cloud run:
  brtvqv9yd3h2gbcsu25n5

best checkpoint test:
  any_auroc = 0.7054
  any_accuracy_at_0_5 = 58.52%
  any_brier = 0.2208
  best_score_corr = 0.3078
  best_score_rmse = 0.4029

same-test heuristic baselines:
  max_correctness_head any_auroc = 0.4717, best_score_corr = -0.0769
  max_readiness any_auroc = 0.4776, best_score_corr = -0.1232

calibration / ablation artifact:
  /data1/luyifei/drla/outputs/cola_hierarchical_state_verifier/
  p2e_state_verifier_full_seed20260529_20260529_calibration_ablation

calibrated test:
  any_brier = 0.1985
  ece_10 = 0.0232
  any_prob_mean = 0.3418 vs target 0.3470

input / prior ablation:
  full any_auroc = 0.7054, best_score_corr = 0.3078
  zero_latent any_auroc = 0.6731, best_score_corr = 0.2317
  task_prior any_auroc = 0.6399, best_score_corr = 0.2399
  global_prior any_auroc = 0.5142, best_score_corr = 0.0000
```

This supports latent-state utility readability beyond simple certificate
heuristics and beyond task-prior alone. Calibration improves probability
scale, but this is not yet a downstream text-superiority result.

P2-E calibrated receiver-state policy audit:

```text
artifact:
  /data1/luyifei/drla/outputs/cola_hierarchical_state_policy/
  p2e_state_policy_score_fuser_v2_locked_seed20260529_20260529

always-on test:
  single_sender_first accuracy = 22.38%, score = 0.3553
  score-target latent fuser v2 accuracy = 23.41%, score = 0.3685
  text_majority_selected accuracy = 23.41%, score = 0.3622
  oracle_any_selected accuracy = 34.70%, score = 0.4951

state_any_prob gate, valid target any precision 0.60:
  test coverage = 13.35%
  test any precision = 61.54%
  accepted fuser accuracy = 53.85%
  accepted fuser score = 0.5503
  accepted text accuracy = 49.23%
  fuser-else-first fallback accuracy = 23.00%
  fuser-else-first fallback score = 0.3609

state_any_prob gate, valid target any precision 0.65:
  test coverage = 11.29%
  accepted fuser accuracy = 56.36%
  accepted fuser score = 0.5636

train task-prior any gate, valid target any precision 0.60:
  test coverage = 12.32%
  accepted fuser accuracy = 55.00%
  accepted fuser score = 0.5500
```

The state policy audit says the calibrated latent state can find high-utility
subsets, but the current fuser/fallback action does not yet beat always-on
fuser overall. Keep this as receiver-state evidence, not final communication
superiority.

P2-E structured receiver-state action selector:

```text
direct selector:
  artifact = /data1/luyifei/drla/outputs/cola_receiver_state_action_selector/
             p2e_state_action_selector_state_fuser_prior_seed20260529_20260529
  SwanLab = 7emwtma3xyvrmvlv1hibb
  best test accuracy = 21.97%
  best test score = 0.3554

residual selector:
  artifact = /data1/luyifei/drla/outputs/cola_receiver_state_action_selector/
             p2e_state_action_selector_residual_state_fuser_prior_seed20260529_20260529
  SwanLab = 2z5uj588g6kkkv6dpte97
  best test accuracy = 22.79%
  best test score = 0.3626

reference baselines on the same held-out test:
  score-target latent fuser v2 accuracy = 23.41%, score = 0.3685
  text_majority_selected accuracy = 23.41%, score = 0.3622
```

The shallow action selector is negative: residual-fuser output is safer than
direct logits, but neither variant beats the raw fuser v2. Treat the compressed
state tuple as a risk/readiness side signal, not as a replacement for
sender-level latent selection.

P2-E state-conditioned sender-level latent fuser:

```text
frozen residual:
  artifact = /data1/luyifei/drla/outputs/cola_state_conditioned_latent_fuser/
             p2e_state_conditioned_fuser_frozen_state_fuser_prior_seed20260529_20260529
  SwanLab = 1ua24n9yo4tsrq4inahb9
  best test accuracy = 23.00%
  best test score = 0.3654

unfrozen residual:
  artifact = /data1/luyifei/drla/outputs/cola_state_conditioned_latent_fuser/
             p2e_state_conditioned_fuser_unfrozen_state_fuser_prior_seed20260529_20260529
  SwanLab = je3suuujcleox4x40lahd
  best test accuracy = 23.41%
  best test score = 0.3651

reference:
  score-target latent fuser v2 accuracy = 23.41%, score = 0.3685
  text_majority_selected accuracy = 23.41%, score = 0.3622
```

This preserves raw sender-level latent states but still does not beat fuser v2.
The next P2-E branch should test request-more-latent / additional evidence
rather than more reranking over the same three senders.

P2-E request-more-latent potential audit:

```text
script = /data1/luyifei/drla/drla/scripts/audit_cola_request_more_latent_potential.py
artifact = /data1/luyifei/drla/outputs/cola_request_more_latent/
           p2e_request_more_latent_potential_locked_seed20260529_20260529
mode = local-only, swanlab_mode=disabled
split = train / valid / test = 3998 / 495 / 487 grouped samples
```

Held-out test:

```text
first sender only = 22.38%, score 0.3553
prefix2 oracle = 30.39%, score 0.4458
prefix3 oracle = 34.70%, score 0.4951
prefix2 readiness selector = 23.61%, score 0.3661
prefix3 readiness selector = 22.38%, score 0.3518

request first -> prefix3 helpful rate = 33.47%
mean first -> prefix3 score gain = +0.1398
```

Interpretation: additional sender latent evidence has a strong oracle upper
bound, but current decoder-free request/readiness heuristics do not close much
of that gap. The next non-degenerate branch is a learned request policy or
sequential aggregator, with first-only, text-majority, always-request,
task-prior, global-prior, and oracle-after-request controls.

P2-E learned request-more-latent policy:

```text
script = /data1/luyifei/drla/drla/scripts/train_cola_request_more_policy.py
online input = first sender sanitized latent/process/certificate/task fields
action = stop after first sender or request remaining sender packets
post-request selector = score-target latent fuser v2
```

Smoke:

```text
artifact = /data1/luyifei/drla/outputs/cola_request_more_policy/
           p2e_request_more_policy_smoke_256groups_seed20260529_20260529
SwanLab = zh936nbxnbrow88z7savu
```

Full runs:

```text
fuser_gain:
  artifact = /data1/luyifei/drla/outputs/cola_request_more_policy/
             p2e_request_more_policy_fuser_gain_full_seed20260529_20260529
  SwanLab = zn7zl11z11ghmfenr8wr4
  best step = 350
  test helpful AUROC = 0.6823
  best practical policy = target_request_rate 0.25 on gain_pred
  request rate / avg sender budget = 23.00% / 1.46
  fuser-after-request accuracy / score = 23.20% / 0.3689

oracle_gain:
  artifact = /data1/luyifei/drla/outputs/cola_request_more_policy/
             p2e_request_more_policy_oracle_gain_full_seed20260529_20260529
  SwanLab = at0w7v8gsewja1vudb3jx
  best step = 350
  test helpful AUROC = 0.6461
  best practical fuser-after-request accuracy / score = 22.79% / 0.3660

references:
  first sender only = 22.38%, score 0.3553
  text majority = 23.41%, score 0.3622
  always request + fuser v2 = 23.41%, score 0.3685
  always request oracle upper bound = 34.70%, score 0.4951
```

Interpretation: the learned request policy gives the first narrow positive
budget-efficiency result. It slightly improves mean score over always-request
fuser while using about half the sender budget, but exact accuracy is slightly
lower and the score delta is tiny. The next bottleneck is post-request
aggregation/selection, not request detection alone.

P2-E post-request anchor-aware selector:

```text
script = /data1/luyifei/drla/drla/scripts/train_cola_post_request_selector.py
online input = first sender + requested sender latent/process/certificate/task fields
architecture = anchor-aware sender states + post-request Transformer
loss = score MSE + listwise ranking + pairwise ranking + gain regression
request-gated eval = previous fuser_gain request policy, target_request_rate 0.25 on gain_pred
```

Smoke:

```text
artifact = /data1/luyifei/drla/outputs/cola_post_request_selector/
           p2e_post_request_selector_anchor_smoke_256groups_seed20260529_20260529_rerun
SwanLab = r5jsobe8gh78m1ehuh818
```

Full runs:

```text
anchor score selection:
  artifact = /data1/luyifei/drla/outputs/cola_post_request_selector/
             p2e_post_request_selector_anchor_score_full_seed20260529_20260529
  SwanLab = r7qj2vu48vws5lnp5edxf
  standalone accuracy / score = 22.18% / 0.3572
  request-gated accuracy / score = 23.00% / 0.3668

anchor rank selection:
  artifact = /data1/luyifei/drla/outputs/cola_post_request_selector/
             p2e_post_request_selector_anchor_rank_full_seed20260529_20260529
  SwanLab = 089s7vnjawqpd4p7m20ak
  standalone accuracy / score = 21.77% / 0.3500
  request-gated accuracy / score = 23.41% / 0.3697

references:
  first sender only = 22.38%, score 0.3553
  text majority = 23.41%, score 0.3622
  always request + fuser v2 = 23.41%, score 0.3685
  request policy + fuser v2 = 23.20%, score 0.3689
  oracle upper bound = 34.70%, score 0.4951
```

Interpretation: standalone post-request selector is negative, but the
request-gated anchor-rank variant is a narrow positive budget-efficiency
result. It suggests that the next branch should jointly train request and
post-request selection, instead of composing two independently trained modules.

P2-E joint request-and-select policy:

```text
script = /data1/luyifei/drla/drla/scripts/train_cola_joint_request_select_policy.py
request head input = first sender latent/process/certificate/task fields only
selector head input = first + requested sender fields, only after request
loss = request BCE + request gain + score MSE + listwise rank + pairwise rank + gain + budgeted utility
```

Smoke:

```text
artifact = /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
           p2e_joint_request_select_smoke_256groups_seed20260529_20260529
SwanLab = munwuikpa4v8m8g9nysqf
```

Full runs:

```text
joint rank:
  artifact = /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
             p2e_joint_request_select_rank_full_seed20260529_20260529
  SwanLab = g126cuz3w32r9g76jizcs
  best gated policy = target_request_rate 0.50 on request_prob
  request rate / budget = 47.64% / 1.95
  accuracy / score = 23.82% / 0.3733

joint score:
  artifact = /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
             p2e_joint_request_select_score_full_seed20260529_20260529
  SwanLab = 2t0j6t9v33qfkfn71w2mi
  best gated policy = target_request_rate 0.50 on request_prob
  request rate / budget = 47.64% / 1.95
  accuracy / score = 24.02% / 0.3735

references:
  first sender only = 22.38%, score 0.3553
  text majority = 23.41%, score 0.3622
  always request + fuser v2 = 23.41%, score 0.3685
  request policy + fuser v2 = 23.20%, score 0.3689
  request policy + anchor-rank selector = 23.41%, score 0.3697
  oracle upper bound = 34.70%, score 0.4951
```

Interpretation: joint request-and-select is the current strongest P2-E
budgeted latent result. It is a locked-split, same-substrate win over text
majority and fuser baselines in mean score, with a small exact-accuracy gain.
Keep the claim narrow because the oracle gap is still large.

P2-E joint policy calibration / risk-control audit:

```text
script = /data1/luyifei/drla/drla/scripts/audit_cola_joint_policy_calibration.py
artifact = /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
           p2e_joint_request_select_score_full_seed20260529_20260529_calibration_risk
mode = local-only, swanlab_mode=disabled
split = valid 495 groups / test 487 groups
```

Readout:

```text
request_prob -> model_request_helpful:
  valid AUROC / ECE = 0.6959 / 0.2363
  test  AUROC / ECE = 0.7164 / 0.2226

request_prob -> oracle_request_helpful:
  valid AUROC / ECE = 0.7089 / 0.1050
  test  AUROC / ECE = 0.6825 / 0.1132

best utility row:
  target_request_rate = 0.50 on request_prob
  test request rate / budget = 47.64% / 1.95
  test accuracy / score = 24.02% / 0.3735
  requested model-loss Wilson95 upper = 35.02%

strict loss-risk caps:
  0.10 / 0.20: no non-trivial valid threshold
  0.30 / 0.40: almost-always request, test score drops to 0.3494
```

Interpretation: `request_prob` has useful ranking signal but is over-confident
as a calibrated probability. The current best policy is valid-selected
budgeted utility, not a risk-certified communication policy.

P2-E strict fresh split replication:

```text
overlap audit:
  seed30 test groups = 495
  seed30 test overlap with seed29 train/valid/test = 406 / 51 / 38

non-canonical diagnostic:
  artifact = /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
             p2e_joint_request_select_score_full_seed20260530_20260529
  SwanLab = jg65cg65hvzqxdybjc298
  issue = reused seed29 fuser checkpoint and norm_stats
```

Strict seed30 same-split fuser:

```text
training artifact = /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
                    p2e_hierarchical_fuser_score_full_seed20260530_20260529
SwanLab = qweuypbg1ugls3io0s9j0
best eval artifact = /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
                     p2e_hierarchical_fuser_score_full_seed20260530_20260529_best_eval_test
test fuser accuracy / score = 22.63% / 0.3694
```

Strict seed30 same-split joint:

```text
training artifact = /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
                    p2e_joint_request_select_score_full_seed20260530_fuserseed20260530_20260529
SwanLab = n4wu1f4ghzfwe6mhvltei
calibration artifact = /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
                       p2e_joint_request_select_score_full_seed20260530_fuserseed20260530_20260529_calibration_risk

best gated policy:
  target_request_rate = 0.50
  signal = request_gain_pred
  request rate / budget = 50.51% / 2.01
  accuracy / score = 23.64% / 0.3796

references:
  first sender = 22.42%, score 0.3669
  text majority = 22.63%, score 0.3642
  same-split fuser best = 22.63%, score 0.3694
  oracle upper bound = 35.96%, score 0.5071

calibration:
  request_prob model-helpful AUROC / ECE = 0.6255 / 0.2697
  requested model-loss Wilson95 upper = 25.40%
```

Interpretation: the strict seed30 rerun replicates the positive utility
direction against text and same-split fuser, while confirming that probability
calibration and risk certification remain weak. The earlier seed30 diagnostic
must not be used as canonical because of seed29 fuser/norm_stats overlap.

Strict seed31 same-split replication:

```text
same-split fuser:
  artifact = /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
             p2e_hierarchical_fuser_score_full_seed20260531_20260529
  SwanLab = 64h605uhcjpse62n84l8v
  best eval = /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
              p2e_hierarchical_fuser_score_full_seed20260531_20260529_best_eval_test
  test fuser accuracy / score = 17.75% / 0.3184

same-split joint:
  artifact = /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
             p2e_joint_request_select_score_full_seed20260531_fuserseed20260531_20260529
  SwanLab = xh19j75yervdr7l603rt4
  calibration artifact = /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
                         p2e_joint_request_select_score_full_seed20260531_fuserseed20260531_20260529_calibration_risk
```

Strict seed31 result:

```text
first sender = 16.96%, score 0.3074
text majority = 17.75%, score 0.3180
same-split fuser best = 17.75%, score 0.3184
joint selected = 19.33%, score 0.3426
oracle = 29.78%, score 0.4515

best selected policy:
  target_loss_wilson_upper = 0.30
  signal = request_prob
  request rate / budget = 99.80% / 3.00
  requested model-loss Wilson95 upper = 26.99%
```

Three-strict-seed aggregate:

```text
artifact = /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
           p2e_joint_request_select_strict_seed29_30_31_summary_20260529

macro mean:
  first sender = 20.59%, score 0.3432
  text majority = 21.26%, score 0.3481
  same-split fuser = 21.26%, score 0.3521
  joint selected = 22.33%, score 0.3652
  oracle = 33.48%, score 0.4846
  request rate / budget = 65.98% / 2.32
  request_prob model-helpful AUROC / ECE = 0.6688 / 0.2571
```

Checkpoint-selection audit:

```text
script = /data1/luyifei/drla/drla/scripts/eval_cola_joint_request_select_policy.py

last-checkpoint best policy:
  seed29 = 23.00%, score 0.3653
  seed30 = 24.85%, score 0.3919
  seed31 = 17.75%, score 0.3252
```

Interpretation: three strict seeds now support a positive macro utility
direction for latent joint request-select over text majority and same-split
fuser, but seed31 shows that the budget advantage is not stable. Last-checkpoint
eval is mixed, which motivated the valid-frontier audit below. Since that audit
is negative, the next target should be corrected channel-equivalent P2-D
evaluation or a genuinely new calibration/objective, not more shallow checkpoint
or threshold tuning.

P2-E valid-frontier checkpoint selection audit:

```text
script = /data1/luyifei/drla/drla/scripts/train_cola_joint_request_select_policy.py
mode = --checkpoint-selection-mode valid_rate_frontier
request rates = 0.10,0.25,0.50,0.75

strict frontier runs:
  seed29 = p2e_joint_request_select_score_frontier_full_seed20260529_fuserseed20260529_20260529
  seed30 = p2e_joint_request_select_score_frontier_full_seed20260530_fuserseed20260530_20260529
  seed31 = p2e_joint_request_select_score_frontier_full_seed20260531_fuserseed20260531_20260529

aggregate = /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
            p2e_joint_request_select_frontier_strict_seed29_30_31_summary_20260531
```

Canonical frontier readout, using valid-selected policy rows only:

```text
old target025 strict aggregate = 22.33%, score 0.3652
valid-frontier canonical = 21.46%, score 0.3505
request rate / budget = 49.73% / 1.99
score delta vs old joint = -0.0147
```

Interpretation: valid-frontier checkpoint selection is a completed negative
audit. It is useful evidence that the checkpoint-selection issue is real, but
it does not improve over the current canonical target025 strict aggregate. Do
not use test-best frontier rows as paper numbers.

P2 script policy:

```text
train scripts:
  CUDA/GPU required
  SwanLab cloud required
  write metrics.jsonl, best_checkpoint.pt, last_checkpoint.pt

eval/audit/packet-build scripts:
  swanlab_mode=disabled
  no optimizer/backward
  write local summary.json / metrics.jsonl / audit CSVs
```

## Current Official-Protocol Results

The current 2026-05-24 summary artifact is:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_readiness_halt_20260524
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_b64_readiness_halt_20260524
```

The first includes full-model multi-seed, signal ablations, EOS-only/fixed-B baselines, valid-calibrated thresholds, and leave-one-task-out transfer results for the 32-token / 2-block protocol. The second covers the 64-token / 4-block trace, score, frontier, full readiness model, calibrated adaptive halt eval, trace dynamics, the non-gold `prediction_stability` halt baseline, and stability-feature model comparison.

`train_cola_readiness_model.py` now derives non-gold `scored_prediction` / `official_processed_generation` stability features. Checkpoint feature fields are preserved for backward-compatible eval of older 22-feature checkpoints.

The b64 leave-one-task-out summary for `process_no_task_stability_features` is:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_b64_cross_task_stability_features_20260524
```

It shows positive but modest cross-task transfer and a remaining calibration problem, especially on SQuAD.

The strict calibration comparison is:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_b64_cross_task_calibration_comparison_20260524
```

Tightening tolerance from `0.01` to `0.0` mostly preserves accuracy but becomes too conservative (`3.65/4` blocks), so the next calibration work should be stability-aware rather than threshold-only.

The guarded calibration comparison is:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_b64_cross_task_guarded_calibration_20260524
```

`adaptive_or_prediction_stability` is an aggressive control, not a true guard: it can stop before the task-scored prediction is stable. The true `stability_guarded_adaptive` policy preserves the cross-task prediction-stability baseline (`27.03%`, `2.41/4` blocks) but currently selects threshold `0.0`, so it adds no extra saving. The aggressive control reaches `26.28%` at `1.75/4` blocks under calibrated held-out evaluation. A conservative aggregate OR threshold `0.75` preserves accuracy but saves only `0.005` extra block versus prediction-stability, confirming that useful pre-stability readiness is still poorly calibrated for held-out tasks.

Sample-level halt diagnostics are:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_b64_halt_decision_analysis_20260524
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_b64_halt_decision_analysis_strict_20260524
```

The 1% calibrated adaptive/OR policies lose `61` final-correct held-out samples, all before prediction stability; `57/61` are strict prefix losses. The strict 0% OR variant still loses `8` LAMBADA final-correct samples while averaging `2.286/4` blocks, and all `8/8` are prefix losses, so simple threshold tightening does not solve task-robust safe early halt.

Continuation-risk gated halt results are:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_b64_risk_gated_halt_20260524
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_b64_risk_gated_halt_first_saving_20260524
```

The `min_blocks` calibration is useful as an aggressive upper bound: cross-task accuracy `26.95%`, average blocks `1.864/4`, still `0.08` percentage points below prediction-stability. The `first_saving` calibration is the current safe default: cross-task aggregate accuracy matches fixed-final / prediction-stability at `27.03%` while using `2.018/4` average blocks. It still has one SQuAD task-level loss offset by one MMLU gain, so task-level safety is not solved.

The current strongest 1k-protocol result adds the non-gold contentful-prediction guard:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_b64_risk_gated_halt_first_saving_content_guard_20260524
/data1/luyifei/drla/outputs/cola_risk_gated_halt_analysis/first_saving_content_guard_squad_20260524
```

It reaches aggregate `27.04%` accuracy at `2.019/4` blocks and removes the SQuAD quote-only early-stop loss. Treat the slight aggregate gain over fixed-final as a diagnostic early-correct case, not as a primary accuracy-improvement claim.

The full prepared-split protocol is now unified on `batch_size=12`, because batch-invariance diagnostics showed MMLU output can change across batch sizes. Do not mix `bs20`, `bs12`, and `bs1` outputs in a main trace result. The diagnostic artifact is:

```text
/data1/luyifei/drla/outputs/cola_batch_invariance/batch_size_invariance_summary_20260524.json
```

Full prepared-split artifacts:

```text
trace: /data1/luyifei/drla/outputs/cola_block_traces/tasks_official8_full_b64_t16_seed66_bs12_merged_20260524
score: /data1/luyifei/drla/outputs/cola_official_benchmarks/full_b64_bs12_trace_score_20260524/summary.json
frontier: /data1/luyifei/drla/outputs/cola_readiness_frontiers/official8_full_b64_bs12_t16_seed66_20260524
readiness model: /data1/luyifei/drla/outputs/cola_readiness_model/official8_full_b64_bs12_t16_seed66_20260524
risk model: /data1/luyifei/drla/outputs/cola_continuation_risk_model/official8_full_b64_bs12_process_no_task_seed20260524
adaptive halt: /data1/luyifei/drla/outputs/cola_adaptive_halt/official8_full_b64_bs12_t16_seed66_20260524
risk-gated halt: /data1/luyifei/drla/outputs/cola_risk_gated_halt/official8_full_b64_bs12_first_saving_content_guard_seed20260524
cross-seed summary: /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_seed_20260524
```

Merge summary: `49,019` generation rows and `196,076` trace rows, `ok=true`. Official scorer task-average accuracy is `25.26%` (SwanLab `gpnfpuu93117wy3nub94o`). On the full-split test partition, first-saving risk-gated halt with content guard reaches `20.392%` micro accuracy at `1.880/4` blocks versus fixed-final `20.351%` at `4/4` (SwanLab `fgrqk019ypjtc6hd52suf`). Treat this as a block-budget result, not as a benchmark accuracy-improvement claim.

Seed67 repeats the full protocol with `seed=20260525`, `per_sample_noise_seed=67`, and uniform `batch_size=12`. Its merged trace is:

```text
/data1/luyifei/drla/outputs/cola_block_traces/tasks_official8_full_b64_t16_seed67_bs12_merged_20260524
```

Seed67 official scorer task-average is `24.99%`; first-saving risk-gated halt with content guard reaches `21.039%` micro accuracy at `1.873/4` blocks versus fixed-final `21.018%` at `4/4`.

Seed68 repeats the full protocol with `seed=20260526`, `per_sample_noise_seed=68`, and uniform `batch_size=12`. Its merged trace is:

```text
/data1/luyifei/drla/outputs/cola_block_traces/tasks_official8_full_b64_t16_seed68_bs12_merged_20260524
```

Seed68 official scorer task-average is `24.96%`; first-saving risk-gated halt with content guard reaches `20.798%` micro accuracy at `1.842/4` blocks versus fixed-final `20.860%` at `4/4`. The three-seed summary averages `20.743%` risk-gated micro accuracy at `1.865/4` blocks versus `20.743%` fixed-final at `4/4`, about `53.38%` block saving. Treat this as a block-budget/readiness result, not as a benchmark accuracy-improvement claim.

Strict uncertainty calibration artifact:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_uncertainty_calibration_20260524
```

With `accuracy_drop_tolerance=0.0`, strict first-saving calibration averages `20.750%` micro accuracy at `2.291/4` blocks; strict min-blocks gives `20.750%` at `2.281/4`. Both match prediction-stability accuracy on all three seeds. The selected `entropy_max` and `top_prob_min` are `none` for every seed, so mean token entropy/top-prob does not currently improve the guard.

Choice-stability guard artifact:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_choice_guard_20260524
```

The sample-level diagnostics show the remaining `0.01`-tolerance content-guard losses are single-letter multiple-choice flips, not prefix losses. With `--require-stable-single-choice`, the 3-seed policy matches prediction-stability accuracy (`20.750%`) at `2.210/4` blocks and has zero observed losses versus prediction-stability. Narrowing it to `--stable-single-choice-max-block 1` keeps zero observed losses, improves to `20.757%` at `2.137/4` blocks, and is the current better safety-cost point among these guards. The current audit rerun also saves `calibration_risk_threshold_sweep.csv` and selected valid metrics for every seed.

Validation-selected choice-scope artifact:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_choice_scope_sweep_20260524
```

With `accuracy_drop_tolerance=0.01`, validation selects `single_choice_guard_scope=off` and preserves the cheaper content-guard policy, but sample diagnostics still show observed single-letter losses. With strict `accuracy_drop_tolerance=0.0`, validation selects `single_choice_guard_scope=1` and `risk_threshold=0.01` on all three seeds, matching prediction-stability with zero observed losses at `2.137/4` blocks. This makes block1-only a validation-selected safety setting, not a test-tuned rule.

Prediction-change risk artifact:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_prediction_change_risk_20260524
```

The `prediction_change` target is still non-gold: it asks whether the current scored prediction differs from the future prediction-stability reference in the same rollout. Three formal SwanLab training runs reach test AUROC about `0.997`. Under strict `accuracy_drop_tolerance=0.0`, first-saving calibration selects `single_choice_guard_scope=off` on all seeds and reaches `20.750%` at `1.958/4` blocks with zero observed losses versus prediction-stability. Strict `min_blocks` is cheaper (`1.904/4`) but seed67 has loss/gain cancellation, so it remains diagnostic rather than the safety-first policy.

Leave-one-task-out prediction-change risk transfer artifacts:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_prediction_change_risk_20260524
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_prediction_change_risk_seed67_20260524
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_prediction_change_risk_seed68_20260524
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_prediction_change_risk_cross_seed_20260524
```

This trains readiness/risk on 7 tasks with `process_no_task`, calibrates only on the 7-task valid split, and evaluates the held-out task on `split=all`. Across seed66/67/68, strict first-saving risk-gated halt averages `21.596% +/- 0.029%` weighted micro accuracy at `2.499 +/- 0.005/4` blocks, matching prediction-stability accuracy while saving about `1.501` blocks versus fixed-final. Sample diagnostics show zero observed losses versus both fixed-final and prediction-stability across all three seeds. Treat this as replicated transfer evidence for the current prepared split, not as a claim that official Cola benchmark accuracy improved.

Joint readiness-threshold calibration artifact:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_joint_readiness_riskcap04_cross_seed_20260524
```

This sweeps readiness thresholds on valid, caps `risk_threshold_end=0.4`, requires zero valid loss versus prediction-stability, uses the contentful-prediction guard, and applies block2 single-choice stability guard. Across seed66/67/68 it keeps zero observed losses versus prediction-stability and reaches the same weighted micro accuracy at `2.118/4` average blocks, saving `0.394` block beyond prediction-stability.

Decoder-dependency status: this is the current best decoder-probed baseline, not a final latent-only agent communication policy. The readiness/risk labels and features still use decoder probe, decoded text dynamics, task-scored prediction, prediction-stability, and official scorer outputs; the eval guard also uses `scored_prediction`. Future default experiments should prioritize latent-primary halt variants that use these decoder-side signals only as teacher labels or offline diagnostics, with online inputs restricted toward latent trajectory, process features, and learned latent verifier proxies.

SQuAD held-out answer-shape diagnostic:

```text
/data1/luyifei/drla/outputs/cola_risk_gated_halt/cross_task_full_b64_bs12_prediction_change_shape_features_fragment_guard_v2_joint_readiness_min_blocks_choice2_zeroloss_seed20260526/leave_squad_out_eval_squad_all
/data1/luyifei/drla/outputs/cola_risk_gated_halt_analysis/cross_task_full_b64_bs12_prediction_change_shape_features_fragment_guard_v2_joint_readiness_min_blocks_choice2_zeroloss_20260526/leave_squad_out_eval_squad_all
```

The shape-feature risk model alone still loses SQuAD prefix-fragment cases without riskcap. Adding `--require-fragment-complete-prediction` v2 reaches zero observed losses versus fixed-final and prediction-stability at `1.908/4` blocks on SQuAD `split=all` (SwanLab run `639qsu4cpfseme3cpomf2`). Treat this as a diagnostic readiness signal, not as a final cross-task replacement for riskcap04 until it is validated across tasks/seeds.

Seed68 full 8-task fragment-completeness checks:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_fragmentguardv2_scope_sweep_seed68_20260524
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_fragmentguardv2_choice2_noriskcap_seed68_20260524
```

The scope-sweep/no-riskcap run saves blocks (`1.895/4`) but loses `21` MMLU samples, all single-choice flips. The fixed block2/no-riskcap run fixes MMLU but loses `34` SQuAD samples. This keeps riskcap04 as the current safety baseline and motivates full answer-shape feature training rather than more hand-written guards.

Full 8-task answer-shape risk check:

```text
/data1/luyifei/drla/outputs/cola_continuation_risk_model/cross_task_full_b64_bs12_prediction_change_shape_features_process_no_task_seed20260526
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_shape_features_fragmentguardv2_choice2_noriskcap_seed68_20260524
/data1/luyifei/drla/outputs/cola_risk_gated_halt_analysis/cross_task_full_b64_bs12_prediction_change_shape_features_fragmentguardv2_choice2_noriskcap_seed20260526
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_shape_features_fragmentguardv2_choice2_noriskcap_cross_seed_20260524
```

All `24` seed66/67/68 leave-one-task-out 38-feature risk checkpoints are trained with `best_checkpoint.pt`. The per-task valid-selected no-riskcap policy averages weighted micro accuracy `21.596%` at `2.153/4` blocks, but has `19` observed losses versus prediction-stability, offset by `19` gains. The losses are free-text prefix/completion failures: SQuAD numeric/entity fragments, HellaSwag `[substeps]` / `[step]` continuations, and StoryCloze short sentence prefixes. Post-hoc zero-loss aggregate rows average `2.191/4` blocks, still worse than joint-readiness riskcap04 (`2.118/4`, zero observed losses). Riskcap04 remains the current cross-seed safety baseline.

Fragment-completeness v3 riskcap04 check:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_shape_features_fragmentguardv3_choice2_riskcap04_cross_seed_20260524
```

The v3 guard adds non-stable completion checks for replacement characters, trailing continuation markers, bare currency amounts, numeric range prefixes, connector endings, short unstable phrases, and truncated hyphen tokens. No-riskcap v3 was stopped early because seed66/HellaSwag produced `25` losses versus prediction-stability. With `risk_threshold_end=0.4` restored, seed66/67/68 official8 stays at zero observed losses and averages `21.596%` weighted micro accuracy at `2.245/4` blocks, saving `0.267` block versus prediction-stability. This is safer than no-riskcap but still worse than the current joint-readiness riskcap04 baseline (`2.118/4`, zero losses), so keep v3 as a diagnostic completion signal rather than a new default policy.

Terminology: `trace collection` is not training and not the final evaluation itself. It runs official Cola block-wise rollout to record per-block latent/probe/text/stability signals. The scorer, readiness frontier, readiness/risk training, and halt evaluations are downstream consumers of that trace.

## Trace Smoke Command

Use disabled SwanLab only for engineering smoke tests:

```bash
source /data1/luyifei/drla/scripts/activate_conda.sh
PYTHONPATH=/data1/luyifei/Cola-DLM/code:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 \
python -m drla.scripts.collect_cola_block_traces \
  --task-name lambada \
  --input-jsonl /data1/luyifei/Cola-DLM/code/generate_task_data/lambada.jsonl \
  --output-dir /data1/luyifei/drla/outputs/smoke_cola_block_traces/tasks_smoke \
  --max-samples 1 \
  --batch-size 1 \
  --max-new-tokens 16 \
  --timestep-num 1 \
  --swanlab-mode disabled
```

Full trace/evaluation runs must use `swanlab_mode=disabled` and `tasks_<alias>` output directories so the official `scripts/acc_calc.py` can discover `lambada.jsonl`, `mmlu.jsonl`, etc. No-training scripts reject any non-disabled SwanLab mode; only scripts with a real optimizer/backward training loop may use SwanLab cloud.

## Archived Legacy Code

The old self-built Stage A/B/C path, GSM8K diagnostic wrappers, and custom small Cola-latent-prior scripts were moved to:

```text
/data1/luyifei/drla/archive/2026-05-24-legacy-stagebc-code
```

Old Stage A/GSM8K data, Stage B/C outputs, custom-prior outputs, and old smoke artifacts were moved to:

```text
/data1/luyifei/drla/archive/2026-05-24-legacy-stagebc-artifacts
```

Both archives are reproducibility-only. Do not use them as the main experimental path.
