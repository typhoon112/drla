# P3 Dream-DLM LatentMAS 实施计划

更新日期：2026-06-18

> 状态：当前 Dream 系列主线工程计划。本文把
> `/data1/luyifei/drla/docs/current/P3_Dream_DLM_Latent_MAS_Experiment_Design_2026-06-06.md`
> 落成可执行阶段、artifact 命名、训练/评估边界和验收标准。2026-06-06 已完成
> D0-D2.5 Dream TextMAS capability gate、D3/D4 full200 trace/frontier、
> D5 hidden/logit/process readiness student 和 D5.5 online halt policy
> calibration/risk-control、D6 suffix-tensor latent packet construction。当前有效锁定协议
> 是 maxctx4096；D7 已完成 V1-V7 receiver diagnostics、D7.10-D7.12 text-interface
> diagnostics、D7.13 receiver-control audit、D7.14 held-out packet readiness /
> locked held-out receiver eval，以及 D7.15 failure-localization audit。V7
> layer-conditioned zeroshuf 是 calibration train-dominated receiver；locked
> held-out800 hard gate 未通过，且 calibration valid/test 也未通过 hard gate。不要从
> D7.13 calibration aggregate 直接进入 D8 主表。D7.16 已完成 compact
> train2000 selected-suffix substrate、V7-init receiver 训练和 checkpoint-defined
> valid200 generation audit；loss-level margin 更强，但 valid generation hard gate
> 失败。D7.17 denoising sensitivity audit 进一步显示 matched packet 几乎不改变
> inference-time top/transfer token decisions。D7.18 partial-denoising repair
> 改善 matched CE / no-message sensitivity 但 row-specific shuffled binding 不足。
> D7.19 row-binding weighted repair 只小幅提升 shuffled-row margin，generation hard
> gate 仍失败，不能进入 held-out 或 D8。

## 0a. 2026-06-06 执行状态

已完成：

```text
D0 Dream-v0-Instruct-7B 下载与文件验收
D1 Dream diffusion_generate / output_history / token-logit-hidden hook probe
D2.1 calibration smoke10
D2.2 calibration pilot50
D2.3 calibration full200
D2.4 protocol freeze at maxctx2048, then static context audit exposed held-out overflow
D2.4-v2 protocol freeze at maxctx4096
D2.5-v2 locked held-out800
D3.0 trace collector smoke1
D3.1 hidden/state validation
D3.2 subset20 trace
D3.3 full200 trace queue/merge
D4 subset20 and full200 readiness frontier
D4 corrected frontier with hidden payload
D5 DreamStepReadinessStudent-v1 training with SwanLab cloud
D5.5 online halt policy calibration/risk-control eval
D6 textmas_matched suffix-tensor trace and latent packet manifest
D7 V1 MSE fuser diagnostic
D7 V2 contrastive fuser alignment
D7 raw latent-prefix generation diagnostic
D7 V3 embedding-space soft-prefix adapter and generation controls
D7 V4 native layer-conditioned receiver and 20/50-row generation controls
D7 V5 corruption-aware layer receiver and 20-row generation controls
D7 V6 receiver-generated candidate merge/audit and answer-reranker diagnostic
D7 V7 V4-initialized zero/shuffled corruption fine-tune and full200 controls
D7.5 text-encoded packet diagnostic for AgentB-side hidden compatibility
D7.6 text-packet adapter training and 20-row generation controls
D7.7 TextMAS-teacher layer receiver training and best/last generation controls
D7.8 V7 matched-channel candidate-pool diagnostic
D7.9 V7 interface/distribution audit
D7.10 text-interface virtual-message receiver training and generation controls
D7.11 packet-specific text-interface receiver with unlikelihood/contrastive losses
D7.12 balanced text-interface receiver with capped corruption margins
D7.13 unified receiver-control audit for V7 and D7.10-D7.12
D7.14 held-out D6 packet readiness, substrate build, V7 held-out eval/audit
D7.15 V7 interface distribution and split-generalization failure localization
D7.16 compact selected-suffix tensor substrate, train2000 receiver, and valid gate negative
D7.17 denoising sensitivity audit for D7.16 failure localization
D7.18 partial-denoising receiver repair and valid gate negative
D7.19 row-binding weighted receiver repair and valid gate negative
```

关键 artifact：

```text
model:
  /data1/luyifei/drla/models/Dream-v0-Instruct-7B

D0:
  /data1/luyifei/drla/outputs/p3_dream_models/
  Dream-org_Dream-v0-Instruct-7B_prepare_20260606_144309

D1:
  /data1/luyifei/drla/outputs/p3_dream_models/
  dream_instruct_7b_generation_probe_20260606_144650

D2 full200 merged:
  /data1/luyifei/drla/outputs/p3_dream_textmas_runs/
  dream_textmas_gate_full200_merged_20260606

D2 full200 aggregate:
  /data1/luyifei/drla/outputs/p3_dream_textmas_aggregates/
  dream_textmas_gate_full200_merged_20260606

D2 protocol lock:
  /data1/luyifei/drla/outputs/p3_dream_protocol_audits/
  dream_textmas_protocol_lock_calibration_full200_maxctx4096_20260606

D2 held-out800 maxctx4096 merged:
  /data1/luyifei/drla/outputs/p3_dream_textmas_runs/
  dream_textmas_gate_heldout800_maxctx4096_merged_20260606

D2 held-out800 maxctx4096 aggregate:
  /data1/luyifei/drla/outputs/p3_dream_textmas_aggregates/
  dream_textmas_gate_heldout800_maxctx4096_merged_20260606

D2 held-out800 maxctx4096 leakage audit:
  /data1/luyifei/drla/outputs/p3_dream_protocol_audits/
  dream_textmas_gate_heldout800_maxctx4096_merged_20260606

D3 trace smoke1:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_calibration_trace_smoke1_steps16_stride4_20260606

D3 trace pilot2:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_calibration_trace_pilot2_steps32_stride4_20260606

D3 hidden summary smoke1:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_calibration_trace_hidden_summary_smoke1_steps16_stride4_20260606

D3 hidden tensor smoke1:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_calibration_trace_hidden_tensor_smoke1_steps8_stride4_20260606

D3 trace subset20:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_calibration_trace_subset20_steps64_stride4_hidden_summary_20260606

D4 frontier subset20:
  /data1/luyifei/drla/outputs/p3_dream_readiness_frontiers/
  musique_calibration_trace_subset20_steps64_stride4_hidden_summary_frontier_20260606

D3 trace full200 queue:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_calibration_trace_full200_steps64_stride4_hidden_summary_20260606_queue

D3 trace full200 merged:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_calibration_trace_full200_steps64_stride4_hidden_summary_merged_20260606

D4 frontier full200:
  /data1/luyifei/drla/outputs/p3_dream_readiness_frontiers/
  musique_calibration_trace_full200_steps64_stride4_hidden_summary_frontier_20260606

D4 corrected frontier with hidden payload:
  /data1/luyifei/drla/outputs/p3_dream_readiness_frontiers/
  musique_calibration_trace_full200_steps64_stride4_hidden_summary_frontier_with_hidden_20260606

D5 current readiness student:
  /data1/luyifei/drla/outputs/p3_dream_readiness_students/
  dream_step_readiness_student_v1_full200_with_hidden_seed20260606_20260606

D5.5 online halt policy eval:
  /data1/luyifei/drla/outputs/p3_dream_readiness_policy_eval/
  dream_step_readiness_student_v1_full200_with_hidden_policy_eval_20260606

D6 textmas_matched suffix-tensor trace queue:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_calibration_trace_textmas_matched200_steps64_stride4_hidden_tensor_20260606_queue

D6 textmas_matched suffix-tensor trace merged:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_calibration_trace_textmas_matched200_steps64_stride4_hidden_tensor_merged_20260606

D6 latent packet manifest:
  /data1/luyifei/drla/outputs/p3_dream_latent_packets/
  dream_textmas_matched200_agent_ab_suffix_tensor_packets_v1_20260606

D7.13 receiver-control audit:
  /data1/luyifei/drla/outputs/p3_dream_receiver_control_audits/
  dream_receiver_generation_control_audit_v7_d710_d711_d712_20260617

D7.14 held-out D6 packet readiness preflight:
  /data1/luyifei/drla/outputs/p3_dream_heldout_packet_preflights/
  dream_heldout_packet_readiness_preflight_20260617

D7.14 held-out textmas_matched suffix-tensor trace:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_heldout_trace_textmas_matched800_steps64_stride4_hidden_tensor_merged_20260617

D7.14 held-out D6 packet manifest:
  /data1/luyifei/drla/outputs/p3_dream_latent_packets/
  dream_textmas_heldout800_agent_ab_suffix_tensor_packets_v1_20260617

D7.14 V7 held-out receiver eval:
  /data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/
  dream_layer_receiver_v7_heldout800_merged_20260617

D7.14 V7 held-out receiver-control audit:
  /data1/luyifei/drla/outputs/p3_dream_receiver_control_audits/
  dream_receiver_generation_control_audit_v7_heldout800_20260617

D7.15 V7 failure localization:
  /data1/luyifei/drla/outputs/p3_dream_receiver_generalization_audits/
  dream_receiver_v7_d714_failure_localization_20260617

D7.16 validdiag50 compact selected-suffix trace:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_validdiag50_trace_textmas_matched_selected_suffix_tensor_merged_20260617

D7.16 validdiag50 compact selected-suffix packets:
  /data1/luyifei/drla/outputs/p3_dream_latent_packets/
  dream_textmas_validdiag50_selected_suffix_tensor_packets_20260617

D7.16 train2000 compact selected-suffix trace:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_train2000_trace_textmas_matched_selected_suffix_tensor_merged_20260617

D7.16 train2000 compact selected-suffix packets:
  /data1/luyifei/drla/outputs/p3_dream_latent_packets/
  dream_textmas_train2000_selected_suffix_tensor_packets_20260617

D7.16 train2000 receiver:
  /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
  dream_layer_receiver_d716_train2000_v7init_zeroshuf_seed20260617

D7.16 train2000 receiver valid generation:
  /data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/
  dream_layer_receiver_d716_train2000_v7init_zeroshuf_valid200_20260617

D7.16 train2000 receiver valid control audit:
  /data1/luyifei/drla/outputs/p3_dream_receiver_control_audits/
  dream_receiver_generation_control_audit_d716_train2000_valid200_20260617

D7.17 D7.16 denoising sensitivity audit:
  /data1/luyifei/drla/outputs/p3_dream_receiver_sensitivity_audits/
  dream_receiver_d716_valid50_steps64_max128_20260618
```

D2 full200 结果：

```text
rows / samples:
  1400 / 200

generation row errors / leakage errors:
  0 / 0

primary_score_mean:
  single_full_info = 0.59
  single_q_only = 0.045
  textmas_matched = 0.465
  textmas_no_message = 0.05
  textmas_shuffled_message = 0.11
  textmas_wrong_evidence_or_wrong_shard = 0.10

paired primary delta CI lower:
  full_info_vs_question_only = 0.475
  matched_vs_no_message = 0.34
  matched_vs_shuffled = 0.28
  matched_vs_wrong_evidence = 0.295
```

冻结配置：

```text
model_path = /data1/luyifei/drla/models/Dream-v0-Instruct-7B
dtype = bfloat16
max_tokens = 128
dream_steps = 64
temperature = 0.2
top_p = 0.95
alg = entropy
alg_temp = 0.0
max_context_tokens = 4096
prediction_extraction_mode = first_segment
scorer = drla.evaluation.p2_phase_c_scorers.score_qa_answer
```

D2 held-out800 maxctx4096 结果：

```text
rows / samples:
  5600 / 800

generation row errors / leakage errors:
  0 / 0

primary_score_mean:
  single_full_info = 0.49625
  single_q_only = 0.025
  textmas_matched = 0.38125
  textmas_no_message = 0.0225
  textmas_shuffled_message = 0.045
  textmas_wrong_evidence_or_wrong_shard = 0.045
  textmas_compressed_state = 0.35

paired primary delta CI lower:
  full_info_vs_question_only = 0.435
  matched_vs_no_message = 0.32375
  matched_vs_shuffled = 0.29875
  matched_vs_wrong_evidence = 0.29875

parseable_rate:
  1.0 for all conditions

leakage audit:
  status = pass
  num_errors = 0
  num_warnings = 2640
```

重要边界：

```text
旧 maxctx2048 held-out 因 single_full_info context overflow 废弃为诊断结果，
不能作为正式 held-out 指标引用。

Dream model config 支持更长 context；4096 是当前协议上限，不是模型硬上限。
任何新 held-out、更长 prompt 或新 benchmark 先做 static context audit，再做
locked evaluation。
```

当前 D5.5 结果：

```text
selected policy:
  ready_threshold = 0.05
  final_match_threshold = 0.7
  prediction_change_max = 1.0
  future_gain_max = 999.0

valid:
  final_accuracy = 0.65
  selected_accuracy = 0.65
  accuracy_drop_vs_final = 0.0
  mean_step_savings = 53.175 / 63
  halt_before_final_rate = 0.925

internal test:
  final_accuracy = 0.50
  selected_accuracy = 0.50
  accuracy_drop_vs_final = 0.0
  mean_step_savings = 54.95 / 63
  halt_before_final_rate = 0.95
  paired bootstrap 95% CI for accuracy_drop_vs_final = [0.0, 0.0]
  paired bootstrap 95% CI for mean_step_savings = [50.525, 58.1]

boundary:
  local-only eval
  no SwanLab run
  thresholds selected on valid only
  test is report-only
  this is an internal D5 split, not a new held-out TextMAS claim
```

下一步：

```text
D7 receiver/fuser integration.
如果需要 projector/fuser/adapter，这是 deep-learning training，必须 GPU/CUDA +
SwanLab cloud，valid_interval <= 10，并保存 metrics.jsonl、best_checkpoint.pt、
last_checkpoint.pt。
```

## 0b. 当前工作区体检

2026-06-06 检查结果：

```text
git status --short at initial P3 planning time:
  clean

current note:
  D2 maxctx4096 docs, D3 trace scripts, D4 frontier script, and script index
  updates are active WIP until committed/archived.

CoLA archive:
  /data1/luyifei/drla/docs/cola_archive/README.md 已存在并接入 DOCS_INDEX

当前 DRLA/CoLA/Dream 训练进程:
  none

其它用户/目录训练进程:
  CITS 相关训练在 /data1/dingkuiye/CITS 下运行，非本项目进程
```

## 1. 执行总原则

必须遵守：

```text
1. 先能力 gate，再 latent communication。
2. 先 calibration，再 frozen protocol，再 held-out。
3. 所有 deep-learning training 必须 GPU/CUDA + SwanLab cloud。
4. 所有 training 必须写 metrics.jsonl、best_checkpoint.pt、last_checkpoint.pt。
5. eval / scoring / aggregation / audit 不创建 SwanLab run。
6. held-out 不能用于 prompt repair、threshold tuning、adapter selection。
7. scorer 只看 Agent B / final solver handoff 后输出。
8. smoke test 只验证链路，不得作为架构成败结论。
```

Dream 主线不再继续 CoLA adapter 修补。CoLA 线仅作为 P0/P1 readiness 证据和 P2
failure boundary。

## 2. 目录规划

新增输出根：

```text
/data1/luyifei/drla/outputs/p3_dream_models/
/data1/luyifei/drla/outputs/p3_dream_capability_gate/
/data1/luyifei/drla/outputs/p3_dream_textmas_runs/
/data1/luyifei/drla/outputs/p3_dream_textmas_aggregates/
/data1/luyifei/drla/outputs/p3_dream_traces/
/data1/luyifei/drla/outputs/p3_dream_readiness_frontiers/
/data1/luyifei/drla/outputs/p3_dream_readiness_students/
/data1/luyifei/drla/outputs/p3_dream_readiness_policy_eval/
/data1/luyifei/drla/outputs/p3_dream_latent_packets/
/data1/luyifei/drla/outputs/p3_dream_latentmas_runs/
/data1/luyifei/drla/outputs/p3_dream_latentmas_aggregates/
/data1/luyifei/drla/outputs/p3_dream_protocol_audits/
```

建议新增脚本前缀：

```text
drla/scripts/p3_prepare_dream_models.py
drla/scripts/p3_probe_dream_generation.py
drla/scripts/p3_run_dream_textmas_gate.py
drla/scripts/p3_aggregate_dream_textmas_gate.py
drla/scripts/p3_collect_dream_step_traces.py
drla/scripts/p3_build_dream_readiness_frontier.py
drla/scripts/p3_train_dream_readiness_student.py
drla/scripts/p3_eval_dream_readiness_policy.py
drla/scripts/p3_build_dream_latent_packets.py
drla/scripts/p3_run_dream_latentmas.py
drla/scripts/p3_aggregate_dream_latentmas.py
drla/scripts/p3_audit_dream_protocol_boundaries.py
```

脚本命名可调整，但必须保持 P3 / dream / textmas / latentmas 边界清晰。

## 3. Phase D0: 模型下载与环境确认

目标：

```text
下载并固定 Dream-v0-Instruct-7B
可选下载 Dream-v0-Base-7B
确认 transformers trust_remote_code 路径可加载
确认 diffusion_generate 可运行并能输出 history
```

模型路径建议：

```text
/data1/luyifei/drla/models/Dream-v0-Instruct-7B
/data1/luyifei/drla/models/Dream-v0-Base-7B
```

如果 Hugging Face 下载慢，优先使用本机已配置的镜像策略。下载本身不是训练，不创建
SwanLab run。

验收：

```text
model_config.json / tokenizer files / safetensors present
single prompt generation succeeds
output_history=True returns intermediate states or recoverable intermediate tokens
peak VRAM recorded
summary.json + metrics.jsonl written locally
```

产物：

```text
outputs/p3_dream_models/dream_instruct_7b_load_probe_YYYYMMDD/
  summary.json
  metrics.jsonl
  sample_generations.jsonl
  environment.json
```

## 4. Phase D1: Dream substrate probe

目标：

```text
确定 Dream 可以暴露哪些在线可用 state：
  hidden states
  logits
  denoising step history
  token confidence
  mask / changed-token trajectory
  final decode text
```

必须区分：

```text
offline teacher fields:
  decoded text, scorer output, gold correctness, final answer identity

online student / communication fields:
  hidden state, logits summary, process features, step index, uncertainty map
```

验收：

```text
可以在不改模型权重的情况下收集 step snapshots
可以控制 steps / max_new_tokens / temperature / alg
可以限制 prompt <= 2048 context
可以记录 hidden/logit/process features 且不会 OOM
```

## 5. Phase D2: Dream TextMAS capability gate

目标：

```text
先证明 Dream-v0-Instruct-7B 在 MuSiQue evidence-split QA 上具备 true MAS 能力。
```

数据：

优先复用现有 MuSiQue strict protocol：

```text
calibration:
  outputs/p2_phase_c_control_inputs/musique_calibration_controls_200_seed20260601_v1_strict_wrong

held-out:
  outputs/p2_phase_c_control_inputs/musique_heldout_controls_800_seed20260605_v1_strict_wrong
```

当前有效协议使用 `max_context_tokens=4096`。2048 是过窄诊断上限；它在
held-out single_full_info 上暴露 context overflow，不能作为正式协议继续使用。
任何新协议先用 `p3_audit_dream_static_context_lengths.py` 做 calibration/held-out
静态长度审计。

条件：

```text
single_q_only
single_full_info
textmas_matched
textmas_no_message
textmas_shuffled_message
textmas_wrong_evidence_or_wrong_shard
textmas_compressed_state
```

执行顺序：

```text
D2.1 calibration smoke 10 samples
D2.2 calibration pilot 50 samples
D2.3 calibration full 200 samples
D2.4 freeze prompt/parser/control definition
D2.5 held-out 800 samples once
```

进入下一阶段 gate：

```text
single_full_info - single_q_only paired CI lower > 0
textmas_matched - no_message paired CI lower > 0
textmas_matched - shuffled paired CI lower > 0
textmas_matched - wrong_evidence paired CI lower > 0
parseable_rate >= 0.95
leakage_audit errors = 0
```

如果 Dream Instruct 不过 gate：

```text
不要训练 latent receiver
不要进入 LatentMAS
先判断是 context compression、prompt format、还是 base capability 问题
Base checkpoint 只做诊断，不作为替代主线
```

## 6. Phase D3: Dream P0 trace collection

目标：

```text
收集 Dream denoising step traces，构建 P0 teacher labels。
```

trace 单位：

```text
sample_id
condition
agent_role
step_index
total_steps
decoded_text_so_far or decoded_current_sequence
hidden_state_ref
logits_summary
process_features
scorer fields offline only
```

建议 snapshot：

```text
steps = 64 or 128 first
snapshot_stride = 4 or 8
always include final step
```

只在 calibration/train split 做 trace 设计和 feature 选择。held-out trace 只能在协议冻结后跑。

产物：

```text
outputs/p3_dream_traces/musique_calibration_dream_instruct_steps64_stride4_YYYYMMDD/
  traces.jsonl
  latent_refs/
  metrics.jsonl
  summary.json
  scorer_outputs/
```

当前工具验收：

```text
D3.0 smoke artifact:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_calibration_trace_smoke1_steps16_stride4_20260606

scope:
  calibration single_full_info
  1 sample / 1 solver call
  dream_steps = 16
  max_tokens = 32
  snapshot_stride = 4

result:
  status = pass
  num_errors = 0
  max_input_tokens = 963
  max_peak_memory_gib = 14.816
  trace events = [0, 0, 4, 8, 12, 15]
  every event has trace_event_index and has_logit_stats flag

D3.0 pilot artifact:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_calibration_trace_pilot2_steps32_stride4_20260606

scope:
  calibration single_full_info,textmas_matched
  2 samples / 4 generation rows / 8 Dream trace calls
  dream_steps = 32
  max_tokens = 64
  snapshot_stride = 4

result:
  status = pass
  num_errors = 0
  solver trace calls = 4
  evidence-agent trace calls = 4
  trace events = 80
  forbidden gold/alias fields in traces = false

D3.1 hidden/state validation:
  hidden summary smoke = pass
  hidden tensor smoke = pass
  hidden module = model.layers.27
  suffix tensor example shape = (16, 3584)
  suffix tensor dtype = torch.float16

D3.2 subset20 trace:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_calibration_trace_subset20_steps64_stride4_hidden_summary_20260606
  scope = 20 calibration samples, single_full_info,textmas_matched
  generation rows = 40
  Dream trace calls = 80
  errors = 0
  hidden/logit event coverage = 1360/1440 = 0.9444
  max_peak_memory_gib = 15.387

D3.3 full200 trace queue:
  queue status = pass
  shards completed = 20/20
  failed shards = 0
  merged generation rows = 400
  merged samples = 200
  merged Dream trace calls = 800
  duplicate row/call ids = 0
  missing trace ids = 0
```

说明：Dream 在 step 0 可能触发多个 hook event；D3 schema 使用
`trace_event_index` 区分 hook event，用 `step` 表示 denoising step，用
`has_logit_stats` 标注该 event 是否带 confidence summary。不要把重复 step 误判为
重复样本或数据泄漏。

## 7. Phase D4: P0 readiness frontier

目标：

```text
把 Dream step trace 转成 oracle readiness / risk / future-gain labels。
```

labels：

```text
answer_identity_stable
answer_ready_vs_final
answer_ready_vs_prediction_stability
prediction_change_risk
empty_or_format_risk
future_gain
receiver_usefulness
```

teacher policies：

```text
prediction_stability
risk_gated_readiness
future_gain_min_cost
receiver_usefulness_oracle
```

验收：

```text
frontier summary per task / split / condition
loss vs final and prediction-stability
answer-change density across steps
no gold fields in online feature schema
```

当前 subset20 frontier 验收：

```text
artifact:
  /data1/luyifei/drla/outputs/p3_dream_readiness_frontiers/
  musique_calibration_trace_subset20_steps64_stride4_hidden_summary_frontier_20260606

status:
  pass
  frontier rows = 40
  frontier events = 720
  missing solver calls = 0
  event hidden_summary_coverage = 0.9444
  event logit_stats_coverage = 0.9444

subset metrics:
  single_full_info final primary = 0.6
  textmas_matched final primary = 0.4
  oracle correct-stable-before-final row rate = 0.5

full200 frontier:
  /data1/luyifei/drla/outputs/p3_dream_readiness_frontiers/
  musique_calibration_trace_full200_steps64_stride4_hidden_summary_frontier_20260606

full200 status:
  pass
  frontier rows = 400
  frontier events = 7200
  missing solver calls = 0
  event hidden_summary_coverage = 0.9444
  event logit_stats_coverage = 0.9444

full200 metrics:
  single_full_info final primary = 0.59
  textmas_matched final primary = 0.465
  oracle correct-stable-before-final row rate = 0.5275
  mean first correct-stable step = 4.7204
```

说明：这是离线 teacher/frontier 诊断，不是在线 halt policy。full200 frontier 是当前
D5 student 训练输入；`step_prediction`、`step_score`、`final_prediction`、
`final_score` 和 oracle labels 只能作为监督/评估，不能作为在线特征。

## 8. Phase D5: P1 DreamStepReadinessStudent

目标：

```text
训练 decoder-supervised、在线 decoder-free 的 Dream step-readiness student。
```

模型草案：

```text
DreamStepReadinessStudent-v1

Inputs:
  selected hidden states or compressed state refs
  logits/confidence summaries
  mask/change/step process features
  previous-step deltas

Architecture:
  Step-State Encoder
  PMA token pooler
  causal trajectory transformer
  process feature encoder
  multi-head readout

Heads:
  halt_action
  continuation_risk
  answer_identity_stability
  future_gain
  receiver_usefulness
```

训练规范：

```text
GPU/CUDA required
SwanLab cloud required
valid_interval <= 10 for early runs, <= 100 maximum for long runs
save best_checkpoint.pt by validation selection metric
also save last_checkpoint.pt
write metrics.jsonl
record config.json and feature_schema.json
```

初始 selection metric：

```text
valid/risk_controlled_step_saving
with constraints:
  calibration loss <= target
  mismatch <= target
  receiver_usefulness AUROC tracked
```

不能只看 readiness_accuracy。CoLA P1 已经表明单一 accuracy 容易早期刷高但不代表
可用早停策略。

当前 D5 验收：

```text
superseded diagnostic:
  /data1/luyifei/drla/outputs/p3_dream_readiness_students/
  dream_step_readiness_student_v1_full200_seed20260606_20260606
  reason = initial D4 frontier omitted hidden_summary payload, so hidden
  features were zero. Do not cite as current D5 result.

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
  SwanLab cloud = yes
  valid_interval = 10
  global_step = 400
  best_step = 350
  best_checkpoint.pt = present
  last_checkpoint.pt = present

best checkpoint:
  valid ready_auroc = 0.9181
  valid ready_accuracy_at_05 = 0.8458
  test ready_auroc = 0.7946
  test ready_accuracy_at_05 = 0.7306
  test final_match_auroc = 0.9997
  test prediction_change_auroc = 0.9997
```

## 8.5. Phase D5.5: online halt calibration / risk-control

目标：

```text
把 D5 student 从“teacher label 预测器”审计成真实 online halt policy。
```

边界：

```text
local-only
no optimizer / backward / weight update
no SwanLab run
thresholds selected on validation split only
test split report-only
gold/scorer/oracle/decoded text never used as online policy inputs
```

当前策略选择：

```text
script:
  drla/scripts/p3_eval_dream_readiness_policy.py

artifact:
  /data1/luyifei/drla/outputs/p3_dream_readiness_policy_eval/
  dream_step_readiness_student_v1_full200_with_hidden_policy_eval_20260606

policy:
  ready_threshold = 0.05
  final_match_threshold = 0.7
  prediction_change_max = 1.0
  future_gain_max = 999.0
```

当前结果：

```text
valid:
  rows = 40
  final_accuracy = 0.65
  selected_accuracy = 0.65
  accuracy_drop_vs_final = 0.0
  mean_selected_step = 9.825
  mean_step_savings = 53.175 / 63
  halt_before_final_rate = 0.925

internal test:
  rows = 40
  final_accuracy = 0.50
  selected_accuracy = 0.50
  accuracy_drop_vs_final = 0.0
  mean_selected_step = 8.05
  mean_step_savings = 54.95 / 63
  halt_before_final_rate = 0.95
  paired bootstrap 95% CI for accuracy_drop_vs_final = [0.0, 0.0]
  paired bootstrap 95% CI for mean_step_savings = [50.525, 58.1]
```

解释：

```text
这是 D5 内部 split 上的 online halt policy 审计，说明 D5 student 不只是 AUROC
高，而是在当前 calibration-derived split 上能以多头状态阈值复现 final accuracy 并
大幅提前停止。它不是新的 held-out benchmark claim；D6 仍需 packet audit 和
Agent B receiver/fuser corruption controls。
```

## 9. Phase D6: Latent packet construction

目标：

```text
从 Agent A 的 selected Dream step 构造可审计 latent packet。
```

packet schema：

```text
sample_id
agent_role
selected_step
total_steps
latent_refs
state_type
process_features
readiness_heads
packet_version
forbidden_field_audit
```

禁止字段：

```text
decoded_text
gold_answer
scorer_output
correctness
final_prediction
prediction_stability_prediction
selected_answer_text
```

packet variants：

```text
p3_dream_packet_v1_hidden:
  selected hidden states / selected layers / selected positions

p3_dream_packet_v2_embedding:
  denoising state z_t or x_pred style embeddings if accessible

p3_dream_packet_v3_compressed:
  PMA-compressed latent tokens plus process/certificate heads
```

主 LatentMAS claim 至少要包含 v1 或 v2 这类被 B 真实消费的 latent state。
v3 只能作为 compact state / policy diagnostic，不能单独证明 latent communication。

当前 D6 验收：

```text
script:
  drla/scripts/p3_build_dream_latent_packets.py

tensor trace:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_calibration_trace_textmas_matched200_steps64_stride4_hidden_tensor_merged_20260606

trace scope:
  condition = textmas_matched
  rows = 200
  samples = 200
  Dream trace calls = 600
  hidden tensor refs = 38400
  failed shards = 0

packet artifact:
  /data1/luyifei/drla/outputs/p3_dream_latent_packets/
  dream_textmas_matched200_agent_ab_suffix_tensor_packets_v1_20260606

packet version:
  p3_dream_packet_v1_suffix_tensor

packet scope:
  packet groups = 200
  packets = 400
  agent_a packets = 200
  agent_b packets = 200
  packet groups with both agents = 200
  mean selected step = 21.98
  example hidden shape = [128, 3584]
  hidden dtype = torch.float16

audit:
  missing refs = 0
  missing traces = 0
  forbidden packet key hits = 0
```

注意：

```text
D5 policy was trained on solver readiness labels. Applying it to evidence-agent
traces is currently a D6 step-selection heuristic, not an independent
evidence-agent readiness claim. D7 receiver/fuser must include corruption
controls before any LatentMAS success claim.
```

Compact substrate update, 2026-06-17:

```text
new trace mode:
  --hidden-capture-mode selected_suffix_tensor

purpose:
  keep step summaries/logit/process features, but write only the D5-policy-
  selected evidence-agent suffix tensor for each Dream call

why:
  full suffix_tensor stores all step tensors and does not scale to 2000+ rows
  on the current /data1 budget

validated artifacts:
  direct smoke1 = pass
  queue smoke1 merge = pass
  validdiag50 queue/merge = pass
  validdiag50 packet build = pass

validdiag50 packet checks:
  packet groups = 50
  packets = 100
  packet groups with both agents = 50
  missing refs = 0
  missing traces = 0
  forbidden packet key hits = 0
  mean selected step = 33.58
  mean selected hidden file size = 919611 bytes
```

Interpretation: selected-suffix mode is now the default engineering substrate
for large nonheldout receiver material. It does not by itself prove receiver
generalization, but it removes the storage bottleneck that previously forced
small full-tensor traces. Any new deep-learning receiver/fuser trained from this
material must still use CUDA/GPU, SwanLab cloud, `valid_interval <= 10`,
`metrics.jsonl`, `best_checkpoint.pt`, and `last_checkpoint.pt`.

Split note: the validdiag50 compact artifact is only an engineering validation
artifact. It overlaps train2000 by 3 sample ids, so it is not a clean
receiver-selection split unless those samples are filtered or a disjoint
validdiag is rebuilt. The locked held-out800 manifest remains disjoint from both
and is still report-only.

train2000 compact substrate status:

```text
trace queue:
  completed shards = 40 / 40
  failed shards = 0

merged trace:
  rows = 2000
  samples = 2000
  trace calls = 6000
  status = pass

packet artifact:
  packet groups = 2000
  packets = 4000
  both-agent groups = 2000
  missing refs = 0
  missing traces = 0
  forbidden packet key hits = 0
  mean selected step = 35.31025
```

## 10. Phase D7: Dream LatentMAS receiver

目标：

```text
让 Agent B 直接消费 Agent A latent packet，而不是 decode 成 text 再重新 encode。
```

receiver modes：

```text
native_continuous_prefix:
  将 A packet 映射/追加为 B 的 continuous input prefix

cross_attention_fuser:
  B 在 denoising过程中 cross-attend 到 A packet

state_conditioned_denoising:
  A packet 作为 conditioning state 影响 B 的 logits / denoising updates
```

训练边界：

```text
如果需要 projector/fuser/adapter:
  只用 train/calibration split
  GPU + SwanLab cloud
  valid_interval <= 10
  best_checkpoint.pt / last_checkpoint.pt
```

eval 边界：

```text
local-only
swanlab disabled
receiver-only scoring
paired controls required
```

当前 D7 结果：

```text
V1 MSE fuser:
  script = drla/scripts/p3_train_dream_latent_fuser.py
  artifact =
    /data1/luyifei/drla/outputs/p3_dream_latent_fusers/
    dream_latent_fuser_v1_textmas_matched200_seed20260606_20260606
  SwanLab =
    https://swanlab.cn/@Lyfff/drla-mvp/runs/hg2otd5swqd3pzudh0k3b
  global_step = 800
  best_step = 570
  status = diagnostic only

V1 controls:
  script = drla/scripts/p3_eval_dream_latent_fuser_controls.py
  artifact =
    /data1/luyifei/drla/outputs/p3_dream_latent_fuser_controls/
    dream_latent_fuser_v1_textmas_matched200_controls_20260606
  result = train shows packet use, valid/test do not pass;
           test zero_packet MSE is better than matched MSE
  interpretation = MSE prefix distillation learned average solver-state
                   components and is not accepted as D7 receiver success

V2 contrastive fuser:
  script = drla/scripts/p3_train_dream_latent_fuser_contrastive.py
  objective = symmetric InfoNCE packet-to-solver latent alignment
  artifact =
    /data1/luyifei/drla/outputs/p3_dream_latent_fusers/
    dream_latent_fuser_v2_contrastive_textmas_matched200_seed20260606_20260606
  SwanLab =
    https://swanlab.cn/@Lyfff/drla-mvp/runs/k07305m849l5jnjyijlk4
  global_step = 800
  best_step = 780
  best_checkpoint.pt = present
  last_checkpoint.pt = present
  metrics.jsonl = present

V2 best checkpoint:
  valid packet->target top1 = 0.71875
  valid target->packet top1 = 0.65625
  test packet->target top1 = 0.75
  test target->packet top1 = 0.625

V2 controls:
  script = drla/scripts/p3_eval_dream_latent_fuser_contrastive_controls.py
  artifact =
    /data1/luyifei/drla/outputs/p3_dream_latent_fuser_controls/
    dream_latent_fuser_v2_contrastive_textmas_matched200_controls_20260606
  random top1 = 0.05 on 20-row valid/test splits
  valid matched top1 = 0.60
  valid shuffled_row top1 = 0.05
  valid zero_packet top1 = 0.05
  test matched top1 = 0.55
  test shuffled_row top1 = 0.00
  test zero_packet top1 = 0.05
```

解释：

```text
V2 proves row-specific latent alignment between upstream packets and solver
latent states under shuffled/zero controls. It does not yet prove ordered
agent-role sensitivity, because agent_swap remains close to matched. It also
does not yet prove final answer generation. The next D7 action should turn
V2 embeddings/prefix into a receiver-side generation or answer-selection path
and keep the same corruption controls.
```

Raw latent-prefix generation diagnostic:

```text
script:
  drla/scripts/p3_run_dream_latent_prefix_eval.py

artifact:
  /data1/luyifei/drla/outputs/p3_dream_latent_prefix_runs/
  dream_latent_prefix_eval_diag20_steps64_prefix8_20260606

boundary:
  local-only
  no optimizer / backward / weight update
  no SwanLab run
  no Agent A/B decoded text inserted into solver prompt
  gold/scorer used only for offline evaluation

setup:
  rows = 20 textmas_matched calibration rows
  max_tokens = 128
  dream_steps = 64
  prefix_tokens_per_agent = 8
  prefix source = raw D6 suffix tensors

result:
  no_message primary = 0.0
  latent_matched primary = 0.0
  latent_shuffled_row primary = 0.0
  latent_agent_swap primary = 0.0
  latent_zero primary = 0.0
  all variants token_f1_mean = 0.0333
```

解释：

```text
Raw D6 suffix tensors are last-layer hidden states, while Dream `inputs_embeds`
expects embedding-space vectors. Directly prepending raw suffix tensors to the
input embedding stream is not a valid final receiver. The failure is evidence
for representation-space mismatch, not evidence against the latent signal,
because V2 contrastive alignment already shows row-specific packet signal.

Immediate follow-up:
  embedding-space soft-prefix adapter has now been tested below. Because it
  does not pass receiver answer-generation controls, the next repair should
  prioritize native layer/KV integration, cross-attention receiver conditioning,
  or a receiver-side answer-selection path rather than simply scaling raw or
  shallow input-prefix variants.
```

Embedding-space soft-prefix adapter diagnostic:

```text
training script:
  drla/scripts/p3_train_dream_soft_prefix_adapter.py

training artifact:
  /data1/luyifei/drla/outputs/p3_dream_soft_prefix_adapters/
  dream_soft_prefix_adapter_v1_textmas_matched200_seed20260607_20260607

SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/1qoue1655x7820f73wvmm

training boundary:
  CUDA/GPU
  SwanLab cloud
  valid_interval = 10
  Dream frozen
  adapter input = D6 agent_a/agent_b suffix tensors
  runtime prompt = no-message solver prompt
  gold answer tokens are supervised loss targets only

training result:
  global_step = 480
  best_step = 460
  best_checkpoint.pt = present
  last_checkpoint.pt = present
  metrics.jsonl = present

best checkpoint loss-level metrics:
  valid matched_ce = 2.9495
  valid zero_ce = 6.1258
  test matched_ce = 2.7128
  test zero_ce = 5.7548
  test token_accuracy = 0.5607
  test agent_swap_ce = 2.6983

interpretation:
  The adapter learns a non-zero answer-token conditioning signal, because
  matched CE is far below zero-prefix CE. However agent_swap CE is close to
  matched CE, so the loss-level signal still does not prove ordered role
  sensitivity or true receiver communication.
```

Soft-prefix receiver generation controls:

```text
eval script:
  drla/scripts/p3_run_dream_soft_prefix_eval.py

artifact:
  /data1/luyifei/drla/outputs/p3_dream_soft_prefix_runs/
  dream_soft_prefix_eval_v1_best20_20260607

boundary:
  local-only
  no optimizer / backward / weight update
  no SwanLab run
  no Agent A/B decoded text inserted into solver prompt
  gold/scorer used only for offline evaluation

setup:
  checkpoint = V3 best_checkpoint.pt
  rows = 20 textmas_matched calibration rows
  max_tokens = 128
  dream_steps = 64
  prefix_len = 16
  conditions =
    no_message
    soft_prefix_matched
    soft_prefix_shuffled_row
    soft_prefix_agent_swap
    soft_prefix_zero

result:
  no_message primary = 0.0, token_f1 = 0.0333
  soft_prefix_zero primary = 0.0, token_f1 = 0.0333
  soft_prefix_shuffled_row primary = 0.0, token_f1 = 0.0833
  soft_prefix_matched primary = 0.0, token_f1 = 0.1083
  soft_prefix_agent_swap primary = 0.0, token_f1 = 0.1083
```

解释：

```text
V3 soft-prefix influences receiver output and improves weak token overlap over
no-message/zero, but it fails the answer-generation gate and does not separate
matched from agent_swap. Do not claim LatentMAS success from V3 and do not
scale this shallow input-embedding prefix as the next main path. The next D7
repair should prioritize native layer/KV integration, cross-attention receiver
conditioning, or a receiver-side answer-selection path that consumes latent
packets and is tested with matched / shuffled-row / zero / agent-swap controls.
```

Native layer-conditioned receiver diagnostic:

```text
training script:
  drla/scripts/p3_train_dream_layer_conditioned_receiver.py

training artifact:
  /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
  dream_layer_receiver_v1_textmas_matched200_seed20260607_20260607

SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/0798gogc7xnpd3hiqeyya

architecture:
  Dream frozen
  packet memory encoder over D6 agent_a/agent_b suffix tensors
  cross-attention residual adapters injected after Dream layers [7, 14, 21, 27]
  conditioning mask applies only to generated/masked positions

training boundary:
  CUDA/GPU
  SwanLab cloud
  valid_interval = 10
  global_step = 480
  best_step = 480
  best_checkpoint.pt = present
  last_checkpoint.pt = present
  metrics.jsonl = present

best checkpoint loss-level metrics:
  valid matched_ce = 2.1264
  valid zero_ce = 2.3628
  valid shuffled_row_ce = 2.2680
  valid agent_swap_ce = 2.0907
  test matched_ce = 1.5232
  test zero_ce = 1.8341
  test shuffled_row_ce = 1.6898
  test agent_swap_ce = 1.4997

interpretation:
  V4 learns a stronger receiver-side CE signal than V3 and separates matched
  from zero/shuffled at the loss level. However agent_swap remains close to or
  better than matched, so ordered role sensitivity is still not established.
```

V4 receiver generation controls:

```text
eval script:
  drla/scripts/p3_run_dream_layer_receiver_eval.py

20-row artifact:
  /data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/
  dream_layer_receiver_eval_v1_best20_20260607

20-row result:
  no_message primary = 0.00, token_f1 = 0.0333
  layer_receiver_zero primary = 0.00, token_f1 = 0.0500
  layer_receiver_shuffled_row primary = 0.00, token_f1 = 0.0500
  layer_receiver_agent_swap primary = 0.00, token_f1 = 0.0750
  layer_receiver_matched primary = 0.10, token_f1 = 0.1750

50-row artifact:
  /data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/
  dream_layer_receiver_eval_v1_best50_20260607

50-row result:
  no_message primary = 0.08, token_f1 = 0.0893
  layer_receiver_zero primary = 0.20, token_f1 = 0.2593
  layer_receiver_shuffled_row primary = 0.18, token_f1 = 0.2293
  layer_receiver_agent_swap primary = 0.18, token_f1 = 0.2493
  layer_receiver_matched primary = 0.20, token_f1 = 0.2693
```

解释：

```text
V4 is the first P3 receiver that improves answer generation over no-message at
50-row scale, but it does not pass packet-specific communication controls:
zero and corrupted packets retain most of the gain. This implies receiver-prior
leakage: the trained receiver can behave like an answer prior even when the
packet is absent or corrupted. Do not run D8 main comparison from V4.

Next D7 repair:
  train a corruption-aware receiver where matched packets must beat zero,
  shuffled-row, and agent-swap packets during training, or switch to a
  receiver-side answer-selection/reranking objective with explicit
  matched-vs-corruption risk controls. A valid D7 receiver must show a stable
  matched advantage over zero/shuffled/agent-swap before any held-out or D8
  TextMAS-vs-LatentMAS table.
```

Corruption-aware layer receiver diagnostic:

```text
training script:
  drla/scripts/p3_train_dream_layer_receiver_corruption_aware.py

training artifact:
  /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
  dream_layer_receiver_v2_corruptaware_textmas_matched200_seed20260607_20260607

SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/gotls7ez7sgvnzyv2vmha

architecture:
  same as V4 layer-conditioned receiver

objective:
  matched answer-token CE
  + margin loss requiring matched CE to be lower than:
    zero-packet CE
    shuffled-row packet CE
    agent-swap packet CE

training boundary:
  CUDA/GPU
  SwanLab cloud
  valid_interval = 10
  global_step = 480
  best_step = 400
  best_checkpoint.pt = present
  last_checkpoint.pt = present
  metrics.jsonl = present

best checkpoint loss-level metrics:
  test matched_ce = 2.1024
  test zero_ce = 5.7847
  test shuffled_row_ce = 2.1638
  test agent_swap_ce = 2.0692

final checkpoint loss-level metrics:
  test matched_ce = 1.8225
  test zero_ce = 5.4697
  test shuffled_row_ce = 1.9194
  test agent_swap_ce = 1.8251
```

V5 receiver generation controls:

```text
eval script:
  drla/scripts/p3_run_dream_layer_receiver_eval.py

20-row artifact:
  /data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/
  dream_layer_receiver_v2_corruptaware_eval_best20_20260607

20-row result:
  no_message primary = 0.00, token_f1 = 0.0333
  layer_receiver_zero primary = 0.00, token_f1 = 0.0333
  layer_receiver_matched primary = 0.00, token_f1 = 0.0500
  layer_receiver_agent_swap primary = 0.00, token_f1 = 0.0500
  layer_receiver_shuffled_row primary = 0.00, token_f1 = 0.0667
```

解释：

```text
V5 confirms the V4 diagnosis. Corruption-aware CE can suppress zero-packet
receiver-prior leakage at the loss level, but this version also removes the
matched answer-generation gain. Do not expand V5 to 50 rows or D8.

Next D7 repair:
  do not merely increase corruption margin. Build a receiver-side
  answer-selection/reranking or two-stage objective that preserves matched
  generation usefulness while enforcing matched-vs-corruption separation.
  A valid receiver must pass generation controls, not only CE controls.
```

Receiver-side answer-reranker diagnostic:

```text
candidate merge/audit script:
  drla/scripts/p3_merge_dream_receiver_candidate_generations.py

training script:
  drla/scripts/p3_train_dream_receiver_answer_reranker.py

merged candidate artifact:
  /data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/
  dream_layer_receiver_eval_v1_best200_candidates_merged_20260607

merge validation:
  200 unique rows
  1000 generations
  5 conditions
  duplicate row/condition pairs = 0
  missing row/condition pairs = 0
  forbidden payload key hits = 0

full200 receiver-generation metrics:
  layer_receiver_matched primary = 0.195
  layer_receiver_agent_swap primary = 0.190
  layer_receiver_shuffled_row primary = 0.185
  layer_receiver_zero primary = 0.115
  no_message primary = 0.045

candidate ceiling with alias-aware offline labels:
  all-condition oracle primary = 0.230
  matched-only oracle primary = 0.195
  matched + no_message oracle primary = 0.220
  matched + zero oracle primary = 0.220
  all-condition average candidate count = 2.71
```

V6 answer-reranker training:

```text
smoke artifact:
  /data1/luyifei/drla/outputs/p3_dream_answer_rerankers/
  dream_receiver_answer_reranker_v1_smoke1_seed20260607_20260607

smoke SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/3tjatglw6lu25g49kxapd

full artifact:
  /data1/luyifei/drla/outputs/p3_dream_answer_rerankers/
  dream_receiver_answer_reranker_v1_seed20260607_20260607

full SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/38fxxq335128q0vachhuh

training boundary:
  CUDA/GPU
  SwanLab cloud
  valid_interval = 10
  global_step = 3200
  best_step = 700
  best_checkpoint.pt = present
  last_checkpoint.pt = present
  metrics.jsonl = present

best checkpoint test controls:
  matched_primary = 0.20
  zero_primary = 0.25
  shuffled_row_primary = 0.25
  agent_swap_primary = 0.20
  matched_minus_zero = -0.05
  matched_minus_shuffled = -0.05
  matched_minus_agent_swap = 0.0

final checkpoint test controls:
  matched_primary = 0.15
  zero_primary = 0.15
  shuffled_row_primary = 0.15
  agent_swap_primary = 0.15
```

解释：

```text
V6 is a negative diagnostic. It confirms that post-hoc candidate selection
over weak receiver-generated answers does not recover packet-specific latent
communication. The all-condition candidate pool is also a diagnostic ceiling,
not a clean online protocol, because it includes outputs from corrupted/no
message controls. Do not run held-out or D8 from V6.

Next D7 repair:
  return to receiver alignment/fusion itself. Candidate selection remains
  admissible only if the online candidate source is produced under the matched
  latent channel or by an explicitly defined matched-channel sampling policy.
  The receiver must show matched > zero/shuffled/agent-swap before D8.
```

V4-initialized zero/shuffled corruption fine-tune:

```text
training script:
  drla/scripts/p3_train_dream_layer_receiver_corruption_aware.py

new script support:
  --init-checkpoint
  --corruption-types

motivation:
  agent_swap is not necessarily a strict negative under the current evidence-agent
  protocol, because Agent A and Agent B are homogeneous evidence holders and
  swapping packet order preserves the same evidence set. Therefore V7 treats
  agent_swap as a diagnostic and trains only zero/shuffled-row corruption.

init checkpoint:
  /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
  dream_layer_receiver_v1_textmas_matched200_seed20260607_20260607/
  best_checkpoint.pt

smoke artifact:
  /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
  dream_layer_receiver_v7_v4init_zeroshuf_smoke1_seed20260607_20260607

smoke SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/hxpqk3c71f3avad3y62vw

full artifact:
  /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
  dream_layer_receiver_v7_v4init_zeroshuf_textmas_matched200_seed20260607_20260607

full SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/m2wnxo0pwzp49dcyl0d3n

training boundary:
  CUDA/GPU
  SwanLab cloud
  valid_interval = 10
  global_step = 480
  best_step = 470
  best_checkpoint.pt = present
  last_checkpoint.pt = present
  metrics.jsonl = present

objective:
  matched CE + 0.1 * margin(matched vs zero/shuffled_row)
  corruption_margin = 0.2
  learning_rate = 2e-5

best checkpoint test loss-level metrics:
  matched_ce = 1.1486
  zero_ce = 1.6930
  shuffled_row_ce = 1.3863
  agent_swap_ce = 1.1135
  zero_ce_margin = 0.5444
  shuffled_row_ce_margin = 0.2377
  agent_swap_ce_margin = -0.0352
```

V7 generation controls:

```text
50-row artifact:
  /data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/
  dream_layer_receiver_v7_v4init_zeroshuf_eval_best50_20260607

50-row result:
  no_message primary = 0.08
  zero primary = 0.16
  shuffled_row primary = 0.16
  agent_swap primary = 0.22
  matched primary = 0.22

full200 merged artifact:
  /data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/
  dream_layer_receiver_v7_v4init_zeroshuf_eval_best200_merged_20260607

full200 merge validation:
  200 unique rows
  1000 generations
  duplicate row/condition pairs = 0
  missing row/condition pairs = 0
  forbidden payload key hits = 0

full200 result:
  no_message primary = 0.035, token_f1 = 0.0487
  zero primary = 0.095, token_f1 = 0.1358
  shuffled_row primary = 0.180, token_f1 = 0.2486
  agent_swap primary = 0.210, token_f1 = 0.2777
  matched primary = 0.215, token_f1 = 0.2789
```

解释：

```text
V7 is the current strongest D7 receiver. It preserves matched generation and
clearly suppresses the zero-packet receiver prior, while improving matched over
shuffled-row by a small margin. It still does not justify D8 by itself: the
matched-vs-shuffled margin is small, and agent_swap is tied with matched. Before
D8, decide whether agent_swap should be treated as a symmetry diagnostic rather
than a corruption negative under the current homogeneous evidence-agent setup,
then run a locked held-out receiver/packet evaluation or a stronger calibration
risk audit.
```

Text-encoded packet diagnostic:

```text
script:
  drla/scripts/p3_run_dream_text_encoded_packet_eval.py

purpose:
  test whether real TextMAS Agent A/B messages become usable latent packets if
  encoded through Dream last-layer hidden states, without inserting the text
  into the final solver prompt.

input text source:
  /data1/luyifei/drla/outputs/p3_dream_textmas_runs/
  dream_textmas_gate_full200_merged_20260606/generations.jsonl

receiver checkpoint:
  /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
  dream_layer_receiver_v7_v4init_zeroshuf_textmas_matched200_seed20260607_20260607/
  best_checkpoint.pt

smoke5 artifact:
  /data1/luyifei/drla/outputs/p3_dream_text_encoded_packet_runs/
  dream_text_encoded_packet_eval_v7_smoke5_20260617

merged20 artifact:
  /data1/luyifei/drla/outputs/p3_dream_text_encoded_packet_runs/
  dream_text_encoded_packet_eval_v7_merged20_20260617

merged20 validation:
  20 unique rows
  100 generations
  duplicate row/condition pairs = 0
  missing row/condition pairs = 0
  forbidden payload key hits = 0

merged20 result:
  text_encoded_matched primary = 0.05
  text_encoded_agent_swap primary = 0.05
  text_encoded_shuffled_row primary = 0.00
  text_encoded_zero primary = 0.00
  no_message primary = 0.00

same-row TextMAS reference:
  textmas_matched primary = 0.40
  textmas_matched token_f1 = 0.42
```

解释：

```text
This diagnostic directly tests the "latent should match text-encoder latent"
hypothesis. Under the current V7 receiver/injection path, text-encoded packets
do not recover the TextMAS channel. Therefore the next repair should not assume
that any same-model hidden state is automatically consumable by the receiver.
It should train or architect an explicit mapping into the receiver's expected
fusion state, preferably using paired text-channel hidden states as a teacher.
```

Text-packet adapter alignment repair:

```text
training script:
  drla/scripts/p3_train_dream_text_packet_adapter.py

eval script:
  drla/scripts/p3_run_dream_text_encoded_packet_eval.py

purpose:
  test whether a learned lightweight mapping from same-model text hidden packets
  into V7 receiver packet space can recover TextMAS communication without
  inserting Agent text into the final solver prompt.

training input:
  real D2 TextMAS agent_messages from
  /data1/luyifei/drla/outputs/p3_dream_textmas_runs/
  dream_textmas_gate_full200_merged_20260606/generations.jsonl

empty-message rule:
  four full200 textmas_matched rows have one empty Agent message. These are
  encoded as zero packets, preserving the fact that the Agent sent no content
  instead of fabricating helper text.

receiver checkpoint:
  /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
  dream_layer_receiver_v7_v4init_zeroshuf_textmas_matched200_seed20260607_20260607/
  best_checkpoint.pt

full training artifact:
  /data1/luyifei/drla/outputs/p3_dream_text_packet_adapters/
  dream_text_packet_adapter_v1_v7_seed20260617_20260617

full SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/ui9l5tllope0w1pik527t

training boundary:
  CUDA/GPU
  SwanLab cloud
  valid_interval = 10
  global_step = 800
  best_step = 760
  best_checkpoint.pt = present
  last_checkpoint.pt = present
  metrics.jsonl = present
  Dream frozen
  V7 receiver frozen
```

Loss-level result:

```text
best valid:
  matched_ce = 0.7869
  zero_ce = 1.4472
  shuffled_row_ce = 1.1283
  zero_ce_margin = 0.6603
  shuffled_row_ce_margin = 0.3414

best test:
  matched_ce = 0.9926
  zero_ce = 1.7216
  shuffled_row_ce = 1.5200
  zero_ce_margin = 0.7291
  shuffled_row_ce_margin = 0.5274
```

Generation eval:

```text
artifact:
  /data1/luyifei/drla/outputs/p3_dream_text_encoded_packet_runs/
  dream_text_packet_adapter_v1_eval20_20260617

scope:
  20 rows
  100 generations
  no optimizer/backward
  no SwanLab run

result:
  no_message primary = 0.00, token_f1 = 0.0333
  text_adapter_zero primary = 0.00, token_f1 = 0.075
  text_adapter_shuffled_row primary = 0.00, token_f1 = 0.075
  text_adapter_agent_swap primary = 0.05, token_f1 = 0.125
  text_adapter_matched primary = 0.05, token_f1 = 0.17

same-row references:
  raw text_encoded_matched primary = 0.05, token_f1 = 0.125
  TextMAS matched primary = 0.40, token_f1 = 0.42
```

解释：

```text
D7.6 is a useful half-positive diagnostic, not an admitted D8 receiver.
The adapter learns a receiver-compatible CE interface and improves token-F1
over raw text-hidden packets, but it does not improve primary accuracy and
remains far below the decoded TextMAS channel. The matched-vs-control gap is
also not strong enough because agent_swap remains tied on primary score.

Conclusion:
  same-model hidden distribution similarity is not sufficient for latent
  communication. The receiver must be trained or architected around an end-task
  communication objective with strong matched-vs-zero/shuffled controls, and
  likely needs a richer online receiver protocol than a post-hoc packet adapter.
```

TextMAS-teacher layer receiver:

```text
training script:
  drla/scripts/p3_train_dream_layer_receiver_text_teacher.py

generation eval script:
  drla/scripts/p3_run_dream_layer_receiver_eval.py

purpose:
  test whether a training-only decoded TextMAS teacher distribution can transfer
  the strong text-channel behavior into the latent receiver while keeping the
  online student prompt decoder/text-free.

student online input:
  no-message solver prompt + D6 latent packets

teacher training input:
  same-row TextMAS decoded Agent messages in the solver prompt

full training artifact:
  /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
  dream_layer_receiver_d77_text_teacher_v7init_textmas_matched200_seed20260617_20260617

full SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/z6rl2y32n03xsdqpwlcs9

training boundary:
  CUDA/GPU
  SwanLab cloud
  valid_interval = 10
  global_step = 480
  best_step = 440
  best_checkpoint.pt = present
  last_checkpoint.pt = present
  metrics.jsonl = present
  Dream frozen
  receiver initialized from V7 and updated
  Agent decoded text is teacher-only, not online student input
```

Loss-level result:

```text
best valid:
  matched_ce = 0.7733
  zero_ce = 1.6413
  shuffled_row_ce = 1.3716
  zero_ce_margin = 0.8680
  shuffled_row_ce_margin = 0.5983
  teacher_kl = 2.1931
  teacher_cosine = 0.8635

best test:
  matched_ce = 0.8260
  zero_ce = 1.8806
  shuffled_row_ce = 1.5745
  zero_ce_margin = 1.0546
  shuffled_row_ce_margin = 0.7485
  teacher_kl = 2.2488
  teacher_cosine = 0.8487
```

Generation eval:

```text
best checkpoint artifact:
  /data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/
  dream_layer_receiver_d77_text_teacher_eval_best20_20260617

best checkpoint result:
  no_message primary = 0.00, token_f1 = 0.0333
  layer_receiver_zero primary = 0.05, token_f1 = 0.1083
  layer_receiver_shuffled_row primary = 0.00, token_f1 = 0.0833
  layer_receiver_agent_swap primary = 0.05, token_f1 = 0.1283
  layer_receiver_matched primary = 0.05, token_f1 = 0.1283

last checkpoint artifact:
  /data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/
  dream_layer_receiver_d77_text_teacher_eval_last20_20260617

last checkpoint result:
  no_message primary = 0.00, token_f1 = 0.0333
  layer_receiver_zero primary = 0.05, token_f1 = 0.1333
  layer_receiver_shuffled_row primary = 0.00, token_f1 = 0.0833
  layer_receiver_agent_swap primary = 0.05, token_f1 = 0.1333
  layer_receiver_matched primary = 0.00, token_f1 = 0.0833

same-row references:
  V7 layer_receiver_matched primary = 0.10, token_f1 = 0.19
  D7.6 text_adapter_matched primary = 0.05, token_f1 = 0.17
  TextMAS matched primary = 0.40, token_f1 = 0.42
```

解释：

```text
D7.7 is a negative generation diagnostic despite strong loss-level metrics.
Teacher-forcing CE/KL and corruption margins do not reliably transfer to Dream
diffusion generation behavior. This is important evidence that the next D7
repair should not use answer-token CE or teacher KL as the main success proxy.

Next direction:
  optimize/evaluate online matched-channel generation directly. Candidate
  selection/reranking remains admissible only if all candidates are generated
  from the matched latent channel, not from zero/shuffled/corrupted controls.
  A generation-time alignment objective or an online matched-channel
  self-consistency/selection protocol is a better next test than another
  teacher-forcing-only receiver loss.
```

### D7.8 V7 matched-channel candidate-pool diagnostic

目的：检验当前最强 raw receiver V7 在 online matched latent channel 下是否至少能
稳定采样到正确答案。该实验不是训练，不上 SwanLab，不做 optimizer/backward，不把
decoded Agent text 插入 solver prompt。它只在同一 20-row comparison set 上对每个
condition 采 8 个候选，并汇总：

```text
online-visible metrics:
  first_primary / first_token_f1
  majority_primary / majority_token_f1

offline diagnostic ceiling:
  oracle_primary / oracle_token_f1
```

脚本：

```text
/data1/luyifei/drla/drla/scripts/p3_run_dream_layer_receiver_candidate_pool_eval.py
```

完整 artifact：

```text
/data1/luyifei/drla/outputs/p3_dream_layer_receiver_candidate_pools/
dream_layer_receiver_v7_candidate_pool_best20_c8_20260617
```

运行边界：

```text
checkpoint = V7 best_checkpoint.pt
rows = 20
candidates_per_row_per_condition = 8
conditions:
  no_message
  layer_receiver_matched
  layer_receiver_shuffled_row
  layer_receiver_agent_swap
  layer_receiver_zero
num_generations = 800
status = pass
SwanLab = disabled
Agent decoded text in solver prompt = no
oracle scorer/gold = offline ceiling only
```

结果：

```text
no_message:
  first_primary = 0.00
  majority_primary = 0.00
  oracle_primary = 0.00
  oracle_token_f1 = 0.00

layer_receiver_matched:
  first_primary = 0.05
  majority_primary = 0.05
  oracle_primary = 0.15
  oracle_token_f1 = 0.195

layer_receiver_agent_swap:
  first_primary = 0.05
  majority_primary = 0.05
  oracle_primary = 0.05
  oracle_token_f1 = 0.1843

layer_receiver_shuffled_row:
  first_primary = 0.00
  majority_primary = 0.00
  oracle_primary = 0.00
  oracle_token_f1 = 0.115

layer_receiver_zero:
  first_primary = 0.00
  majority_primary = 0.00
  oracle_primary = 0.15
  oracle_token_f1 = 0.15
```

行级审计：

```text
matched oracle-correct rows = 3 / 20
zero oracle-correct rows = 3 / 20
matched ∩ zero oracle-correct rows = 2
matched unique oracle-correct rows = 1
zero unique oracle-correct rows = 1
matched-correct also agent_swap-correct rows = 1
```

解释：

```text
D7.8 shows weak candidate-source signal, but not a stable packet-specific
receiver channel. matched oracle_primary reaches 0.15, but zero also reaches
0.15, and matched has only one unique oracle-correct row. first/majority
online selectors remain at 0.05, far below same-row TextMAS matched 0.40.

Do not run D8 from V7/D7.8. Do not train a reranker as the next main step unless
matched candidates first show a clear and stable advantage over zero/shuffled/
agent_swap. The next step should be an interface/distribution audit: compare
TextMAS-conditioned AgentB hidden states with D6 suffix packet tensors and the
receiver injection states, then localize why text -> AgentB tokenizer/embedding/
diffusion works while direct packet injection does not.
```

### D7.9 V7 interface/distribution audit

目的：回答“如果 AgentA decoded text 经过 AgentB tokenizer/embedding/DIT 能 work，
为什么 AgentA latent packet 直接给 B 不 work”。该实验只做本地 forward/statistics，
不训练、不生成答案、不评分、不上 SwanLab。

脚本：

```text
/data1/luyifei/drla/drla/scripts/p3_audit_dream_receiver_interface_distribution.py
```

artifact：

```text
smoke2:
  /data1/luyifei/drla/outputs/p3_dream_receiver_interface_audits/
  dream_receiver_interface_audit_v7_smoke2_20260617

best20:
  /data1/luyifei/drla/outputs/p3_dream_receiver_interface_audits/
  dream_receiver_interface_audit_v7_best20_20260617
```

审计对象：

```text
no-message AgentB solver prompt hidden
TextMAS AgentB solver prompt hidden with decoded Agent messages
D6 agent_a/agent_b suffix packet tensors
V7 PacketMemoryEncoder output
V7 selected-layer gated cross-attention deltas on masked solver positions
```

best20 结果：

```text
TextMAS extra tokens entering AgentB prompt = 238.9

mean-vector cosine:
  packet -> no-message prompt hidden = 0.2947
  packet -> TextMAS full prompt hidden = 0.5149
  packet -> TextMAS last128 hidden = 0.5868
  TextMAS prompt -> no-message prompt hidden = 0.8608

token_norm_mean:
  packet all = 223.25
  no-message prompt hidden = 324.41
  TextMAS prompt hidden = 320.54
  V7 receiver memory after 3584->256 projection = 34.82

receiver gates:
  layer7 = 0.1211
  layer14 = 0.1221
  layer21 = 0.1245
  layer27 = 0.1235

gated delta / masked hidden norm ratio:
  layer7 = 0.3116
  layer14 = 0.2531
  layer21 = 0.2495
  layer27 = 0.0472
```

解释与执行结论：

```text
D6 latent packet is not random: its mean direction is closer to TextMAS
AgentB prompt hidden than to no-message hidden. The failure is therefore likely
an interface/fusion failure, not absence of all signal.

Current V7 receiver is not equivalent to feeding AgentA latent as AgentB text
hidden. It compresses the packet into 256-d memory and injects small gated
cross-attention deltas into generated/masked positions only. By layer27, the
packet update is about 4.7% of masked hidden norm, so it can weakly perturb
generation but not reliably reproduce TextMAS communication.

Next receiver design should be text-interface-aligned: preserve more token-level
structure, bridge packets toward AgentB message-hidden states, or condition at
the native hidden/input interface with stronger generation-time controls. Do
not proceed to D8 from V7/D7.8/D7.9.
```

### D7.10 Text-interface virtual-message receiver

目的：把 D7.9 的诊断变成一个可训练 receiver。D7.9 说明 D6 packet 与
TextMAS AgentB prompt hidden 有相关性，但 V7 通过 256-d memory 和小 gate 注入太弱。
D7.10 因此把 packet 映射成 AgentB solver 接口处的连续 virtual message tokens。
该设计参考 continuous prompt / prefix-tuning 的接口直觉：冻结 backbone 时，连续向量
必须放在模型能自然 attend 的位置；同时遵守当前 LatentMAS 目标，runtime 不插入
decoded Agent text。

脚本：

```text
training:
  /data1/luyifei/drla/drla/scripts/p3_train_dream_text_interface_receiver.py

generation eval:
  /data1/luyifei/drla/drla/scripts/p3_run_dream_text_interface_receiver_eval.py
```

训练边界：

```text
Dream frozen
CUDA/GPU required
SwanLab cloud required
valid_interval <= 10
metrics.jsonl required
best_checkpoint.pt required
last_checkpoint.pt required
online student input = no-message solver prompt + latent packets
TextMAS decoded Agent messages = training-only hidden/logit teacher
gold answer = supervised CE target only
```

架构：

```text
input packet:
  agent_a / agent_b D6 suffix tensors
  2 x 32 x 3584

receiver:
  LayerNorm + 3584 -> d_model projection
  agent embedding + packet positional embedding
  2-layer Transformer memory encoder
  learned prefix queries cross-attend to packet memory
  MLP outputs prefix_len x 3584 virtual message tokens

generation:
  [no-message AgentB prompt tokens]
  [continuous virtual message prefix from latent packet]
  [Dream diffusion masked answer tokens]
```

Loss：

```text
matched answer CE
TextMAS teacher KL on answer logits
prefix final-hidden alignment to TextMAS prompt last prefix_len hidden
prefix input-embedding alignment to TextMAS prompt last prefix_len input embeddings
zero / shuffled-row / agent-swap corruption margin
```

Smoke：

```text
artifact:
  /data1/luyifei/drla/outputs/p3_dream_text_interface_receivers/
  dream_text_interface_receiver_d710_smoke2c_seed20260617_20260617

SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/eey3s8ssjeq9b79brmeh1

status:
  pass
  global_step = 2
  best/last/metrics present
```

Full v1 training：

```text
artifact:
  /data1/luyifei/drla/outputs/p3_dream_text_interface_receivers/
  dream_text_interface_receiver_d710_v1_p96d1024_seed20260617_20260617

SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/eb1evnoegho5ez5f6qvje

config:
  prefix_len = 96
  d_model = 1024
  global_step = 480
  best_step = 460
  valid_interval = 10
```

Loss-level result：

```text
best valid:
  matched_ce = 2.5406
  token_accuracy = 0.5683
  hidden_cosine_loss = 0.5361
  zero_ce_margin = 0.0403
  shuffled_row_ce_margin = -0.0042
  agent_swap_ce_margin = -0.0223

best test:
  matched_ce = 2.8319
  token_accuracy = 0.5141
  hidden_cosine_loss = 0.5487
  zero_ce_margin = 0.0603
  shuffled_row_ce_margin = 0.0262
  agent_swap_ce_margin = -0.0024
```

Generation controls：

```text
best checkpoint artifact:
  /data1/luyifei/drla/outputs/p3_dream_text_interface_receiver_runs/
  dream_text_interface_receiver_d710_v1_best20_20260617

best checkpoint:
  no_message primary = 0.00, token_f1 = 0.0333
  text_interface_matched primary = 0.05, token_f1 = 0.1500
  text_interface_zero primary = 0.05, token_f1 = 0.1333
  text_interface_shuffled_row primary = 0.05, token_f1 = 0.1333
  text_interface_agent_swap primary = 0.05, token_f1 = 0.1583

last checkpoint artifact:
  /data1/luyifei/drla/outputs/p3_dream_text_interface_receiver_runs/
  dream_text_interface_receiver_d710_v1_last20_20260617

last checkpoint:
  no_message primary = 0.00, token_f1 = 0.0333
  text_interface_matched primary = 0.05, token_f1 = 0.0750
  text_interface_zero primary = 0.05, token_f1 = 0.0500
  text_interface_shuffled_row primary = 0.05, token_f1 = 0.1000
  text_interface_agent_swap primary = 0.05, token_f1 = 0.1083
```

解释：

```text
D7.10 is a complete negative diagnostic. It proves that text-interface virtual
tokens are trainable and can improve some token-F1, but they do not yet produce
packet-specific communication. In the best checkpoint, the only primary-correct
row is correct under matched, zero, shuffled-row, and agent-swap simultaneously.
This is receiver prior, not latent communication.

Do not proceed to D8 from D7.10. Do not simply increase training steps. The next
D7 experiment must make packet specificity a first-class objective: corrupted
packets should not be able to produce the same correct answer prior. Candidate
selection/reranking remains inadmissible until matched generation itself is
clearly above zero/shuffled/agent-swap.
```

### D7.11 Packet-specific text-interface objective

目的：直接处理 D7.10 暴露的 receiver prior。D7.10 的 virtual-message prefix 能训练，
但 zero/shuffled/agent_swap 也能生成同一正确答案。D7.11 不改 runtime 架构，仍然是
no-message AgentB prompt + latent virtual prefix；只在训练目标中加入负样本约束。

新增 objective：

```text
corrupt_unlikelihood:
  corrupted packets should assign low probability to the same gold answer tokens

logit_contrast:
  matched answer-token CE should beat zero / shuffled-row / agent-swap CE

hidden_contrast:
  matched prefix hidden should be closer to TextMAS teacher hidden than corrupted
  prefix hidden
```

参考动机：

```text
unlikelihood training:
  lower probability of known negative tokens/sequences

contrastive objectives:
  keep positive pairs close and push negative pairs away
```

Smoke：

```text
artifact:
  /data1/luyifei/drla/outputs/p3_dream_text_interface_receivers/
  dream_text_interface_receiver_d711_smoke2_seed20260617_20260617

SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/drls43zfkxm6c2f8s037t

status:
  pass
  init_checkpoint = D7.10 v1 best
  global_step = 2
```

Full v1：

```text
artifact:
  /data1/luyifei/drla/outputs/p3_dream_text_interface_receivers/
  dream_text_interface_receiver_d711_v1_packet_specific_seed20260617_20260617

SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/fopwvqmgi8fqfhxodxruv

config:
  init_checkpoint = D7.10 v1 best
  prefix_len = 96
  d_model = 1024
  corrupt_unlikelihood_weight = 0.5
  logit_contrast_weight = 1.0
  hidden_contrast_weight = 0.5
  corruption_weight = 0.5
  corruption_margin = 0.5
  global_step = 480
  best_step = 230
```

Loss-level result：

```text
best valid:
  matched_ce = 2.4047
  token_accuracy = 0.5529
  zero_ce_margin = 6.7841
  shuffled_row_ce_margin = 0.0021
  agent_swap_ce_margin = -0.0240

best test:
  matched_ce = 2.7797
  token_accuracy = 0.5281
  zero_ce_margin = 6.3762
  shuffled_row_ce_margin = 0.0303
  agent_swap_ce_margin = 0.0030
```

Generation controls：

```text
best checkpoint artifact:
  /data1/luyifei/drla/outputs/p3_dream_text_interface_receiver_runs/
  dream_text_interface_receiver_d711_v1_best20_20260617

best checkpoint:
  no_message primary = 0.00, token_f1 = 0.0333
  text_interface_matched primary = 0.00, token_f1 = 0.0667
  text_interface_zero primary = 0.00, token_f1 = 0.0000
  text_interface_shuffled_row primary = 0.00, token_f1 = 0.0667
  text_interface_agent_swap primary = 0.00, token_f1 = 0.0583

last checkpoint artifact:
  /data1/luyifei/drla/outputs/p3_dream_text_interface_receiver_runs/
  dream_text_interface_receiver_d711_v1_last20_20260617

last checkpoint:
  no_message primary = 0.00, token_f1 = 0.0333
  text_interface_matched primary = 0.00, token_f1 = 0.0500
  text_interface_zero primary = 0.00, token_f1 = 0.0000
  text_interface_shuffled_row primary = 0.00, token_f1 = 0.0333
  text_interface_agent_swap primary = 0.00, token_f1 = 0.0500
```

解释：

```text
D7.11 is a partial negative diagnostic. It solves the zero-prior failure but
does not preserve matched generation. The result is not packet-specific latent
communication and cannot enter D8.

Do not simply increase negative weights. The next D7 design needs a balanced
packet-specific objective: keep matched generation/teacher alignment strong
while separating zero, shuffled-row, and especially agent-swap. Agent-swap is
the hard remaining control because the current packet representation does not
strongly encode agent role/order semantics.
```

### D7.12 Balanced text-interface objective

目的：验证 D7.11 的失败是否主要来自负样本目标过强。D7.12 仍使用
text-interface virtual message receiver，不改变在线数据流；它从 D7.10 best 初始化，
增强 positive teacher alignment，同时降低负样本权重，并加入 negative-loss warmup 与
capped-margin checkpoint selection。

新增机制：

```text
negative_loss_warmup:
  ramp corruption/unlikelihood/contrastive losses over training steps

capped-margin selection:
  corruption margins are useful only up to a target value; very large margins
  are penalized so best checkpoint is not selected only because zero CE explodes

token-accuracy-aware selection:
  validation token accuracy contributes to checkpoint selection so matched
  generation quality cannot be ignored
```

Smoke：

```text
artifact:
  /data1/luyifei/drla/outputs/p3_dream_text_interface_receivers/
  dream_text_interface_receiver_d712_balanced_smoke2_seed20260617_20260617

SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/qoatplup9a5ekdr8bwydr

status:
  pass
  global_step = 2
```

Full v1：

```text
artifact:
  /data1/luyifei/drla/outputs/p3_dream_text_interface_receivers/
  dream_text_interface_receiver_d712_balanced_v1_seed20260617_20260617

SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/fbuzdxnxs5ow5plfc64eo

config:
  init_checkpoint = D7.10 v1 best
  prefix_len = 96
  d_model = 1024
  learning_rate = 3e-5
  hidden_align_weight = 0.8
  teacher_kl_weight = 0.3
  corruption_weight = 0.15
  corruption_margin = 0.2
  corrupt_unlikelihood_weight = 0.05
  logit_contrast_weight = 0.2
  hidden_contrast_weight = 0.1
  negative_loss_warmup_steps = 160
  selection_token_accuracy_weight = 0.8
  selection_margin_target = 1.0
  selection_margin_overflow_penalty = 0.3
  global_step = 480
  best_step = 140
```

Loss-level result：

```text
best valid:
  matched_ce = 2.7577
  token_accuracy = 0.5867
  zero_ce_margin = 0.6173
  shuffled_row_ce_margin = 0.1005
  agent_swap_ce_margin = 0.0059

best test:
  matched_ce = 3.2509
  token_accuracy = 0.5082
  zero_ce_margin = 0.5295
  shuffled_row_ce_margin = -0.0046
  agent_swap_ce_margin = 0.0203
```

Generation controls：

```text
best checkpoint artifact:
  /data1/luyifei/drla/outputs/p3_dream_text_interface_receiver_runs/
  dream_text_interface_receiver_d712_balanced_v1_best20_20260617

best checkpoint:
  no_message primary = 0.00, token_f1 = 0.0333
  text_interface_matched primary = 0.05, token_f1 = 0.0750
  text_interface_zero primary = 0.05, token_f1 = 0.0500
  text_interface_shuffled_row primary = 0.05, token_f1 = 0.0500
  text_interface_agent_swap primary = 0.05, token_f1 = 0.0750

last checkpoint artifact:
  /data1/luyifei/drla/outputs/p3_dream_text_interface_receiver_runs/
  dream_text_interface_receiver_d712_balanced_v1_last20_20260617

last checkpoint:
  no_message primary = 0.00, token_f1 = 0.0333
  text_interface_matched primary = 0.05, token_f1 = 0.0500
  text_interface_zero primary = 0.00, token_f1 = 0.0000
  text_interface_shuffled_row primary = 0.00, token_f1 = 0.0583
  text_interface_agent_swap primary = 0.05, token_f1 = 0.0500
```

Row-level audit：

```text
best checkpoint:
  matched-correct row is also correct under zero, shuffled-row, and agent-swap

last checkpoint:
  matched-correct row is also correct under agent-swap
  zero and shuffled-row are no longer correct
```

解释：

```text
D7.12 is a partial diagnostic. It avoids the D7.11 matched-generation collapse,
but it still does not establish packet-specific latent communication. The best
checkpoint remains prior-driven across all virtual-prefix conditions. The last
checkpoint separates zero/shuffled-row in generation, but agent_swap remains tied
with matched.

This supports the earlier V7 observation: under the current homogeneous
evidence-agent protocol, agent_swap may be a symmetry diagnostic rather than a
strict corruption negative, because swapping A/B preserves the same evidence
set. Do not keep optimizing agent_swap as a hard negative unless the benchmark
or prompt protocol makes A/B roles asymmetric. For the current protocol,
zero-packet and shuffled-row are the hard corruption controls; agent_swap should
be reported separately as a role/order sensitivity diagnostic.
```

### D7.13 Unified receiver-control audit

目的：把 V7 layer receiver 与 D7.10-D7.12 text-interface branch 放到同一 paired
control contract 下，避免只看不同 run 的平均值。审计只读取已有 `generations.jsonl`，
不加载模型、不生成、不训练、不上 SwanLab。

```text
script:
  /data1/luyifei/drla/drla/scripts/
  p3_audit_dream_receiver_generation_controls.py

artifact:
  /data1/luyifei/drla/outputs/p3_dream_receiver_control_audits/
  dream_receiver_generation_control_audit_v7_d710_d711_d712_20260617

hard controls:
  no_message
  zero
  shuffled_row

diagnostic controls:
  agent_swap
```

V7 full200 paired result：

```text
condition means:
  matched primary = 0.215
  no_message primary = 0.035
  zero primary = 0.095
  shuffled_row primary = 0.180
  agent_swap primary = 0.210

paired primary deltas:
  matched - no_message = 0.180, CI = [0.120, 0.240]
  matched - zero = 0.120, CI = [0.065, 0.180]
  matched - shuffled_row = 0.035, CI = [0.005, 0.070]
  matched - agent_swap = 0.005, CI = [-0.010, 0.025]

row overlap:
  matched correct rows = 43
  zero correct rows = 19
  shuffled_row correct rows = 36
  agent_swap correct rows = 42
  matched unique over zero = 31
  matched unique over shuffled_row = 9
  matched unique over agent_swap = 2
```

Text-interface branch audit：

```text
D7.10 best/last:
  hard gate = fail
  matched primary = 0.05
  zero/shuffled-row primary = 0.05

D7.11 best/last:
  hard gate = fail
  matched primary = 0.00

D7.12 best:
  hard gate = fail
  matched primary = 0.05
  zero/shuffled-row primary = 0.05

D7.12 last:
  hard gate = fail on 20-row paired CI
  matched primary = 0.05
  zero/shuffled-row primary = 0.00
```

解释：

```text
Under the revised control taxonomy, V7 is the current strongest receiver and
the only receiver that passes primary paired CI against no_message, zero, and
shuffled-row on full200. It still does not justify D8 by itself because the
matched-vs-shuffled margin is small, token-F1 matched-vs-shuffled CI crosses
zero, and agent_swap remains tied as a role/order symmetry diagnostic.

Next steps should use V7 as the baseline receiver for stricter locked
calibration/risk audit or held-out packet evaluation. Do not continue scaling
D7.10-D7.12 text-interface objectives unless the design can recover V7-level
matched generation while keeping zero/shuffled-row separated.
```

### D7.14 Held-out D6 packet readiness, eval, and audit

目的：在不破坏 locked held-out 边界的前提下，检查并补齐 held-out800 的
`textmas_matched` suffix-tensor trace 与 D6 latent packet substrate，然后用当前
calibration 最强的 V7 layer-conditioned zeroshuf receiver 做一次真正 held-out
generation-control audit。trace、merge、packet build、receiver eval、audit 都是
local-only；不加载 optimizer、不训练、不上 SwanLab。

```text
script:
  /data1/luyifei/drla/drla/scripts/
  p3_audit_dream_heldout_packet_readiness.py

artifact:
  /data1/luyifei/drla/outputs/p3_dream_heldout_packet_preflights/
  dream_heldout_packet_readiness_preflight_20260617

initial status:
  blocked before substrate construction
  can_run_v7_heldout_eval = false before substrate construction
```

可用项：

```text
held-out manifest = present
held-out online inputs = present
held-out TextMAS aggregate = present
calibration suffix-tensor trace reference = present
calibration D6 packet reference = present
V7 best checkpoint = present
```

缺失项：

```text
held-out suffix-tensor trace:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_heldout_trace_textmas_matched800_steps64_stride4_hidden_tensor_merged_20260617

held-out D6 packet manifest:
  /data1/luyifei/drla/outputs/p3_dream_latent_packets/
  dream_textmas_heldout800_agent_ab_suffix_tensor_packets_v1_20260617
```

成本估算：

```text
held-out textmas_matched rows = 800
calibration suffix-tensor hidden refs = 38,400 files / 32.883 GiB
estimated held-out raw suffix-tensor trace hidden refs = 131.533 GiB
estimated held-out selected packet refs = 1.370 GiB
free disk before trace = 211.552 GiB
estimated free disk after trace = 80.019 GiB
disk_budget_for_full_trace = pass with min_free_gib_after_trace=50
```

补齐 held-out substrate：

```text
trace queue:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_heldout_trace_textmas_matched800_steps64_stride4_hidden_tensor_20260617_queue
  status = pass, completed shards = 80 / 80, failed shards = 0

merged trace:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_heldout_trace_textmas_matched800_steps64_stride4_hidden_tensor_merged_20260617
  status = pass, rows = 800, samples = 800, traces = 2400
  duplicates / missing trace ids = 0 / 0

D6 packet manifest:
  /data1/luyifei/drla/outputs/p3_dream_latent_packets/
  dream_textmas_heldout800_agent_ab_suffix_tensor_packets_v1_20260617
  status = pass
  packet groups = 800
  packets = 1600
  agent_a / agent_b = 800 / 800
  missing refs / traces / forbidden keys = 0 / 0 / 0
  mean selected step = 21.534375
```

Post-substrate preflight：

```text
status = ready
can_run_v7_heldout_eval = true
missing required checks = []
advisory disk_budget_for_full_trace = false after construction
```

注意：最终 disk-budget advisory false 只说明在已生成 131 GiB 级 trace 后，当前空闲
磁盘不足以再次原样重跑一遍 full held-out trace；它不是 held-out substrate 的失败项。

Receiver eval reproducibility guard：

```text
script:
  /data1/luyifei/drla/drla/scripts/
  p3_run_dream_layer_receiver_eval.py

fix:
  runtime --manifest-json / --online-inputs-jsonl / --packet-dir / --model-path
  override checkpoint training-data config

summary now records:
  checkpoint_data_config
  runtime_data_config
```

这个 guard 很重要：V7 checkpoint 本身来自 calibration full200，如果 eval 脚本错误地
继续读取 checkpoint config，就会把 held-out eval 伪装成 calibration replay。2026-06-17
held-out smoke 已确认 runtime config 指向 locked held-out manifest、online inputs 和
held-out D6 packets。

V7 held-out eval：

```text
checkpoint:
  /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
  dream_layer_receiver_v7_v4init_zeroshuf_textmas_matched200_seed20260607_20260607/
  best_checkpoint.pt

merged eval artifact:
  /data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/
  dream_layer_receiver_v7_heldout800_merged_20260617

merge validation:
  status = pass
  generations = 4000
  rows = 800
  conditions = 5
  duplicates / missing / forbidden payload hits = 0 / 0 / 0

condition primary_score_mean:
  matched = 0.02500
  no_message = 0.02375
  zero = 0.02875
  shuffled_row = 0.02375
  agent_swap = 0.02375

condition token_f1_mean:
  matched = 0.09736
  no_message = 0.06618
  zero = 0.09462
  shuffled_row = 0.09646
  agent_swap = 0.09342
```

Held-out paired control audit：

```text
artifact:
  /data1/luyifei/drla/outputs/p3_dream_receiver_control_audits/
  dream_receiver_generation_control_audit_v7_heldout800_20260617

status:
  pass
  hard_gate_pass = false

matched - no_message:
  primary delta = +0.00125, CI = [-0.00750, +0.01000]
  token-F1 delta = +0.03117, CI = [+0.01848, +0.04353]

matched - zero:
  primary delta = -0.00375, CI = [-0.01000, +0.00125]
  token-F1 delta = +0.00274, CI = [-0.00645, +0.01182]

matched - shuffled_row:
  primary delta = +0.00125, CI = [-0.00375, +0.00750]
  token-F1 delta = +0.00090, CI = [-0.00672, +0.00866]

matched - agent_swap:
  primary delta = +0.00125, CI = [0.00000, +0.00375]
  token-F1 delta = +0.00394, CI = [-0.00186, +0.01009]
```

解释：

```text
V7 calibration full200 的 matched-over-control 结论没有迁移到 locked held-out800。
Held-out 上 matched 与 no_message/shuffled-row 基本打平，且低于 zero；token-F1 只相对
no_message 有稳定提升，不能区别 zero/shuffled-row。D7.14 因此是一个严肃的负结果：
V7 不能进入 D8 主表，不能 claim latent communication success。下一步应诊断
calibration-to-heldout 分布迁移、zero receiver prior、packet step-selection heuristic
和 receiver 注入强度，而不是引用 D7.13 calibration 结果替代 held-out。
```

### D7.15 Failure localization after V7 held-out failure

目的：在 D7.14 失败后，先定位失败来源，而不是继续调权重。D7.15 只聚合和读取
已有 artifact，不训练、不生成、不上 SwanLab。它回答两个问题：

1. held-out D6 packets 是否在接口分布上明显偏离 calibration；
2. V7 full200 的 matched-over-control 是否真正来自 nontrain generalization，还是主要来自 train split。

```text
interface audit script:
  /data1/luyifei/drla/drla/scripts/
  p3_audit_dream_receiver_interface_distribution.py

split-generalization audit script:
  /data1/luyifei/drla/drla/scripts/
  p3_audit_dream_receiver_split_generalization.py

calibration interface audit:
  /data1/luyifei/drla/outputs/p3_dream_receiver_interface_audits/
  dream_receiver_interface_audit_v7_calibration_full200_20260617

held-out interface audit:
  /data1/luyifei/drla/outputs/p3_dream_receiver_interface_audits/
  dream_receiver_interface_audit_v7_heldout800_20260617

split-generalization audit:
  /data1/luyifei/drla/outputs/p3_dream_receiver_generalization_audits/
  dream_receiver_v7_calibration_split_generalization_20260617

combined readout:
  /data1/luyifei/drla/outputs/p3_dream_receiver_generalization_audits/
  dream_receiver_v7_d714_failure_localization_20260617
```

接口分布结论：

```text
packet_mean_cos_to_textmas_last128:
  calibration = 0.61227
  held-out = 0.62397

packet_mean_cos_to_textmas_prompt:
  calibration = 0.57396
  held-out = 0.58021

receiver_memory_rms:
  calibration = 2.26080
  held-out = 2.05353

delta_to_hidden_norm_ratio:
  layer7  calibration = 0.32080, held-out = 0.33114
  layer14 calibration = 0.26166, held-out = 0.27403
  layer21 calibration = 0.26140, held-out = 0.27498
  layer27 calibration = 0.04937, held-out = 0.05064
```

解释：held-out packets 与 TextMAS hidden 的接口统计没有明显变差，甚至
packet-to-TextMAS-last128 cosine 略高；receiver 注入比例也没有在 held-out 变弱。Layer27
注入在两个 split 都只有约 5% hidden norm，是结构瓶颈，但不能单独解释 held-out 崩溃。

Split-generalization 结论：

```text
V7 training split sizes:
  train = 160
  valid = 20
  test = 20

train primary:
  matched = 0.24375
  no_message = 0.03125
  zero = 0.10000
  shuffled_row = 0.18750
  hard_gate_pass = true

valid primary:
  matched = 0.00000
  no_message = 0.00000
  zero = 0.00000
  shuffled_row = 0.00000
  hard_gate_pass = false

test primary:
  matched = 0.20000
  no_message = 0.10000
  zero = 0.15000
  shuffled_row = 0.30000
  hard_gate_pass = false
```

解释：V7 full200 aggregate 的正向信号主要由 train split 支撑；nontrain split 没有通过
hard gate，test 上 shuffled_row 还高于 matched。这说明 D7.13 aggregate 不能作为
receiver 泛化证据，D7.14 held-out 失败也不是意外。

后续门槛：

```text
Any new receiver must first pass nontrain calibration split:
  matched > no_message
  matched > zero
  matched > shuffled_row
  paired deltas positive on valid/test or a fresh nonheldout split

Only after this nontrain gate passes may we run locked held-out800.
Held-out800 is now report-only for this V7 branch and must not be used for
threshold/model/prompt/objective tuning.
```

Implementation guard added after D7.16: receiver generation eval must use
checkpoint-defined sample-id splits when judging nontrain behavior. Use
`p3_run_dream_layer_receiver_eval.py --split valid` and `--split test` for the
post-training gate, and reserve `row_offset/max_rows` for smoke/runtime slicing
only. If a diagnostic manifest overlaps the training manifest, remove those
sample ids with `--exclude-sample-ids` or rebuild the diagnostic split before
claiming nontrain performance.

D7.16 train2000 receiver result:

```text
training:
  output_dir = /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
               dream_layer_receiver_d716_train2000_v7init_zeroshuf_seed20260617
  SwanLab = https://swanlab.cn/@Lyfff/drla-mvp/runs/7134z0gui6w8jek33rdt0
  init = V7 V4-initialized zeroshuf best checkpoint
  train / valid / test = 1600 / 200 / 200
  valid_interval = 10
  best_step = 600
  checkpoints = best_checkpoint.pt and last_checkpoint.pt

loss-level valid:
  matched_ce = 2.5632873698417096
  zero_ce_margin = 2.8676398239191623
  shuffled_row_ce_margin = 0.03806205857545164
  token_accuracy = 0.6134504672139883

loss-level test:
  matched_ce = 2.470198732819408
  zero_ce_margin = 2.879840268287808
  shuffled_row_ce_margin = 0.03120649160817246
  token_accuracy = 0.6136382107436656
```

D7.16 valid200 generation-control audit:

```text
hard_gate_pass = false

primary:
  no_message = 0.055
  matched = 0.040
  zero = 0.045
  shuffled_row = 0.035
  agent_swap = 0.040

paired primary deltas:
  matched - no_message = -0.015, CI = [-0.040, +0.005]
  matched - zero = -0.005, CI = [-0.035, +0.025]
  matched - shuffled_row = +0.005, CI = [0.000, +0.015]
  matched - agent_swap = 0.000, diagnostic only

prediction similarity:
  matched vs shuffled_row identical prediction = 58.0%
  matched vs agent_swap identical prediction = 70.0%
  matched vs agent_swap identical primary score = 100.0%
```

解释：D7.16 不是数据量不足的简单正例，而是目标错配证据。teacher-forcing CE
和 zero/shuffled margin 能被优化，但 sampled Dream denoising generation 仍主要由
base/no-message prior 与弱 packet-conditioned perturbation 主导；matched latent packet
没有稳定成为答案来源。下一轮 D7 receiver 必须改为 inference-aligned 设计：训练目标
需要直接约束 packet-conditioned denoising trajectory / generated answer behavior，
或者增强在线注入位置与强度，让 hard controls 在生成时显著失效；不能只增加同一
teacher-forcing CE 训练步数后进入 held-out。

D7.17 denoising sensitivity audit:

```text
script:
  /data1/luyifei/drla/drla/scripts/
  p3_audit_dream_receiver_denoising_sensitivity.py

artifact:
  /data1/luyifei/drla/outputs/p3_dream_receiver_sensitivity_audits/
  dream_receiver_d716_valid50_steps64_max128_20260618

scope:
  local-only
  checkpoint-defined valid50
  64 denoising steps
  128 generated-token budget
  12800 step-control records
  matched and controls compared on the same intermediate denoising state
```

D7.17 result:

```text
matched_shared_state:
  primary = 0.020
  token_f1 = 0.1297979797979798

matched vs no_message:
  top1_disagree = 0.008185529597103596
  transfer_top1_disagree = 0.006093749925494194

matched vs zero:
  top1_disagree = 0.008204307057021652
  transfer_top1_disagree = 0.007031249916180969

matched vs shuffled_row:
  top1_disagree = 0.002695536487735808
  transfer_top1_disagree = 0.0018229165952652693

matched vs agent_swap:
  top1_disagree = 0.0016112422474543564
  transfer_top1_disagree = 0.0012499999348074197
```

解释：D7.17 把 D7.16 的失败定位到 denoising decision 层。当前 receiver 即使在
matched packet 条件下，也几乎不会改变每步 top token 或实际 transfer token；与
shuffled-row/agent-swap 尤其接近。这说明 packet signal 不是完全无法进入 logits，
而是没有强到足以改变 Dream 的离散写入决策。下一轮实验必须选择一个能直接影响
inference-time denoising decisions 的方案，例如 trajectory-level imitation/guidance、
step-wise selected-token contrast、或者更强的 layer injection/gating 机制；继续扩展同一
teacher-forced answer CE 目标是不合理的。

### D7.18 proposed receiver repair

目的：把 D7.17 的定位转成新的训练目标。D7.18 不应继续只训练
`prompt + all-mask target -> gold answer CE`，而要让 receiver 在更接近 Dream
inference 的 partial-denoising states 上学会改变离散写入决策。

最低设计要求：

```text
training state:
  random or scheduled partial-mask answer states, not only all-mask targets
  include multiple mask ratios / denoising-time buckets
  prompt remains no-message solver prompt
  packet input remains decoder-free agent_a/agent_b latent tensors

positive objective:
  matched packet predicts gold answer tokens on masked answer positions
  matched packet must improve selected-token logit margins in partial states

negative controls:
  zero packet
  shuffled-row packet
  agent-swap remains diagnostic unless role asymmetry is introduced

decision-level loss:
  optimize matched-vs-control gold-token logit margin at masked positions
  optionally add top-token / selected-token contrast where controls would keep
  the same base prior token

selection:
  checkpoint selected by nontrain valid metric that includes matched CE and
  decision-level matched-vs-hard-control margins
  final admission still requires local-only valid generation hard gate
```

验收标准：

```text
required training hygiene:
  CUDA/GPU
  SwanLab cloud
  valid_interval <= 10
  metrics.jsonl
  best_checkpoint.pt
  last_checkpoint.pt

required local eval before held-out:
  checkpoint-defined valid generation hard gate
  paired audit against no_message / zero / shuffled_row
  denoising sensitivity audit showing transfer-token disagreement rises
  materially from D7.17 baseline

forbidden shortcuts:
  no held-out tuning
  no decoded agent text in solver prompt
  no gold/scorer/oracle as online features
  no D8 unless valid generation hard gate passes
```

D7.18 成功不是 loss 下降本身，而是 matched packet 在 valid split 上开始改变
Dream denoising transfer decisions，并且这种改变提升 final answer correctness while
hard controls fail.

D7.18-v1 screen result on 2026-06-18:

```text
script:
  /data1/luyifei/drla/drla/scripts/
  p3_train_dream_layer_receiver_denoising_aligned.py

training artifact:
  /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
  dream_layer_receiver_d718_screen200_denoising_aligned_seed20260618

SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/5f7ynp5opf6j61cqpbxyy

best_step:
  200

valid:
  matched_ce = 1.7278096367617877
  zero_gold_margin = 2.8303708081692456
  shuffled_row_gold_margin = 0.03000616851146333
  selection_metric = -1.6408261639043147

test:
  matched_ce = 1.6313846607612956
  zero_gold_margin = 2.898188375737518
  shuffled_row_gold_margin = 0.041638789478165565
  selection_metric = -1.5380326717021715
```

D7.18-v1 sensitivity and generation:

```text
sensitivity artifact:
  /data1/luyifei/drla/outputs/p3_dream_receiver_sensitivity_audits/
  dream_receiver_d718_screen200_valid50_steps64_max128_20260618

valid50 generation artifact:
  /data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/
  dream_layer_receiver_d718_screen200_valid50_20260618

valid50 generation-control audit:
  /data1/luyifei/drla/outputs/p3_dream_receiver_control_audits/
  dream_receiver_generation_control_audit_d718_screen200_valid50_20260618

sensitivity transfer_top1_disagree:
  no_message = 0.010885416604578495
  zero = 0.0054687499441206456
  shuffled_row = 0.001302083283662796
  agent_swap = 0.0009374999534338713

generation primary:
  no_message = 0.08
  matched = 0.12
  zero = 0.12
  shuffled_row = 0.10
  agent_swap = 0.08

hard_gate_pass:
  false
```

解释：D7.18-v1 证明 partial-denoising 训练方向比 D7.16 更接近问题核心：
matched CE 明显下降，matched generation valid50 也升到 0.12。但是它仍然不是
row-specific latent communication receiver。zero control 与 matched primary 打平，
shuffled-row transfer disagreement 没有提升，说明 average hard margin 被 zero
主导。不要从 D7.18-v1 进入 held-out 或 D8。下一步 D7.19 应把 shuffled-row /
row-specific packet binding 作为主目标，而不是继续增加 D7.18-v1 的训练步数。

D7.19 row-binding weighted repair on 2026-06-18:

```text
script:
  /data1/luyifei/drla/drla/scripts/
  p3_train_dream_layer_receiver_denoising_aligned.py

new knobs:
  --decision-control-weights zero:0.15,shuffled_row:4.0
  --top-control-weights zero:0.05,shuffled_row:2.0
  --selection-mode row_binding

training artifact:
  /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
  dream_layer_receiver_d719_screen200_row_binding_seed20260618

SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/ub06nr5p8ddq3p2fgdenw

best_step:
  200

valid:
  matched_ce = 1.7381578401603024
  zero_gold_margin = 2.977695965440944
  shuffled_row_gold_margin = 0.06667653079697629
  shuffled_row_top_margin = -1.063897340404801
  selection_metric = -1.8808453747144656

test:
  matched_ce = 1.6379975606159973
  zero_gold_margin = 2.9921453844895587
  shuffled_row_gold_margin = 0.06340354728978127
  shuffled_row_top_margin = -0.9445282942068297
  selection_metric = -1.7558944928309617
```

D7.19 sensitivity and generation:

```text
sensitivity artifact:
  /data1/luyifei/drla/outputs/p3_dream_receiver_sensitivity_audits/
  dream_receiver_d719_screen200_valid50_steps64_max128_20260618

valid50 generation artifact:
  /data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/
  dream_layer_receiver_d719_screen200_valid50_20260618

valid50 generation-control audit:
  /data1/luyifei/drla/outputs/p3_dream_receiver_control_audits/
  dream_receiver_generation_control_audit_d719_screen200_valid50_20260618

sensitivity transfer_top1_disagree:
  no_message = 0.010833333283662795
  zero = 0.0064062499441206455
  shuffled_row = 0.0016666666232049464
  agent_swap = 0.0008333333022892475

generation primary:
  no_message = 0.08
  matched = 0.10
  zero = 0.12
  shuffled_row = 0.10
  agent_swap = 0.10

paired hard controls:
  matched - no_message = +0.02, CI = [0.00, +0.06]
  matched - zero = -0.02, CI = [-0.08, +0.04]
  matched - shuffled_row = 0.00, CI = [0.00, 0.00]

hard_gate_pass:
  false
```

解释：D7.19 验证了“把 shuffled-row loss 单独加权”这条朴素修复不够。它确实把
valid shuffled-row gold margin 从 D7.18-v1 的 `0.0300` 提到 `0.0667`，也把
matched-vs-shuffled transfer disagreement 从 `0.00130` 提到 `0.00167`，但最终
generation 完全没有获得 row-specific advantage：matched 与 shuffled-row 在 50
个 valid 样本上 primary 全平，且 matched 低于 zero。不要从 D7.19 进入 held-out
或 D8。下一步必须先做更广角复盘：row-identity architecture、packet-to-state
retrieval/binding、stronger fusion/gating、trajectory-level packet-conditioned guidance，
或者直接让 nontrain generation matched-vs-shuffled 差异进入选模/训练闭环。

Post-D7.19 design implication:

```text
Literature-aligned reading:
  Coconut-style continuous thoughts are fed back where the model naturally
  consumes the next input embedding.
  CoLaR trains a latent head to predict next compressed embeddings rather than
  only supervising final answer CE.
  LatentMAS-style latent collaboration relies on working-memory / latent
  realignment so downstream agents consume upstream state as state, not as a
  weak late-layer perturbation.
  TarMAC-style continuous communication uses sender/receiver attention and
  recipient binding; message identity is part of the architecture.

Implication for our next receiver:
  A stronger D7.20 should introduce explicit row/recipient binding before or
  inside generation, for example query-key packet retrieval, receiver working
  memory, state realignment, or earlier/stronger gated injection.
  Selection should include a nontrain matched-vs-shuffled generation signal or
  a close proxy; teacher-forced margins alone are insufficient.

Do not:
  keep rerunning the same D7.18/D7.19 objective with larger shuffled weights
  or more steps unless a new architecture changes how AgentB/receiver consumes
  the latent packet.
```

## 11. Phase D8: Main comparison

条件：

```text
Dream TextMAS held-out gate passed
P1 readiness student passed calibration/risk audit
latent packet audit passed
receiver integration passed corruption controls
receiver nontrain calibration split hard gate passed
receiver held-out generation-control hard gate passed
```

主表 rows：

```text
single_q_only
single_full_info
textmas_matched
textmas_no_message
textmas_shuffled_message
textmas_wrong_evidence
latent_matched
latent_metadata_only
latent_shuffled
latent_wrong_sample
latent_wrong_step
latent_noise
```

主指标：

```text
primary score
exact match
token F1
paired CI lower
decoded tokens
denoising steps
wall-clock
peak VRAM
packet bytes
```

成功判据：

```text
latent_matched > latent_shuffled / wrong_sample / noise / metadata_only
latent_matched > no_message
latent protocol passes receiver-only audit
latent communication saves decoded tokens or A denoising steps
```

强成功判据：

```text
latent_matched >= textmas_matched - small_margin
or latent_matched > textmas_matched
with lower token / latency / communication cost
```

## 12. Risk-control protocol

沿用并升级 CoLA P1 风险控制：

```text
calibration selects thresholds only
held-out reports only
Wilson / bootstrap upper bounds for loss and mismatch
no loss/gain cancellation as safety claim
per-task and per-condition risk buckets
loss case audit mandatory
```

对 Dream 新增：

```text
step-saving vs answer stability
step-saving vs receiver usefulness
uncertainty calibration by denoising time
packet corruption AUROC
latent injection OOD audit
```

## 13. 文档更新触发条件

每完成一个阶段必须更新：

```text
docs/current/CURRENT_EXPERIMENT_STATUS.md
docs/DOCS_INDEX.md if new canonical docs/scripts are added
drla/scripts/README.md if new scripts are added
```

每个 deep-learning training 完成后必须记录：

```text
SwanLab run URL / run id
local output_dir
best checkpoint metric
best_checkpoint.pt path
last_checkpoint.pt path
valid frequency
held-out usage boundary
```

## 14. 初始执行清单

第一批只做 D0-D2，不启动 P1/receiver training：

```text
1. 下载 / 链接 Dream-v0-Instruct-7B。
2. Dream generation probe，确认 2048 context、output_history、hidden/logit access。
3. 构建 Dream MuSiQue calibration smoke10。
4. 如 context 超限，仅在 calibration 上压缩 prompt/evidence。
5. 跑 calibration pilot50。
6. 若 gate 有希望，跑 calibration full200。
7. 冻结 prompt/parser/control。
8. 只在通过 calibration 后跑 held-out800。
```

只有 D2 held-out 通过，才进入 D3-D8。

## 15. Sources

Sources checked on 2026-06-06:

- Dream paper: `https://arxiv.org/abs/2508.15487`
- Dream GitHub: `https://github.com/DreamLM/Dream`
- Dream-v0-Instruct-7B model card: `https://huggingface.co/Dream-org/Dream-v0-Instruct-7B`
- Dream-v0-Base-7B model card: `https://huggingface.co/Dream-org/Dream-v0-Base-7B`
- MuSiQue paper: `https://arxiv.org/abs/2108.00573`
- LatentMAS paper: `https://arxiv.org/abs/2511.20639`

Additional sources checked on 2026-06-07 after V6:

- Beyond tokens: a unified framework for latent communication in LLM-based
  multi-agent systems: `https://arxiv.org/abs/2606.05711`
- HyLaT: Efficient Multi-Agent Communication via Hybrid Latent-Text Protocol:
  `https://arxiv.org/abs/2605.25421`
- Learning to Communicate: Toward End-to-End Optimization of Multi-Agent
  Language Systems: `https://arxiv.org/abs/2604.21794`
