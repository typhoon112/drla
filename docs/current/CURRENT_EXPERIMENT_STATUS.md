# Current Experiment Status

Last updated: 2026-06-18

> 状态：当前快照。瘦身前完整历史流水见 `/data1/luyifei/drla/docs/full_history/2026-05-29_pre_slim/CURRENT_EXPERIMENT_STATUS.md`。

## Current Canonical Docs

```text
docs/DOCS_INDEX.md: 文档系统入口
docs/current/P3_Dream_DLM_Latent_MAS_Experiment_Design_2026-06-06.md: Dream-DLM LatentMAS 当前实验设计
docs/current/P3_Dream_DLM_Latent_MAS_Implementation_Plan_2026-06-06.md: Dream-DLM LatentMAS 当前实施计划
docs/current/P2_Latent_MAS_Communication_Implementation_Plan_2026-05-29.md: 历史 P2 canonical，供边界参考
docs/current/P2_Benchmark_and_Agent_Baseline_Redesign_2026-06-01.md: 历史 P2 benchmark/agent baseline 路线修订
docs/current/P2_Next_Phase_Execution_Plan_2026-06-01.md: 历史 P2 下一阶段执行锁定方案
docs/current/P2_D4_Branch_Decision_Audit_2026-06-01.md: 历史 P2 分支决策审计
docs/current/P2_Branch_B_Execution_Plan_2026-06-01.md: 历史 P2 Branch B 执行锁定方案
docs/current/P2_Branch_B_Calibration_Report_2026-06-01.md: 历史 P2 Branch B calibration 报告
docs/current/P2_Official8_Native_Alignment_Audit_2026-06-01.md: 历史 P2 official8 native prompt/eval 对齐审计
docs/current/P2_Post_Family1_Branch_Decision_Memo_2026-06-01.md: 历史 P2 Family 1 stop 后分支决策备忘录
docs/current/P2_Post_Family1_Complete_Execution_Plan_2026-06-01.md: 历史 P2 Family 1 stop 后完整执行方案
docs/current/P2_Locked_Complete_Execution_Scheme_2026-06-01.md: 历史 P2 锁定执行方案，当前已被 P3 Dream-DLM supersede
docs/current/P2_Phase_C_Benchmark_Protocol_Preparation_2026-06-01.md: 历史 Phase C benchmark/protocol 安全准备
docs/current/P2_Phase_C_Data_Source_and_Runner_Design_2026-06-01.md: 历史 Phase C 数据源与 runner 设计
docs/current/P2_Phase_C_Data_Source_Field_License_Audit_2026-06-01.md: 历史 Phase C 数据源字段/license 审计
docs/current/P2_Benchmark_Redesign_Candidate_Inventory_2026-06-01.md: 历史 Branch B benchmark redesign 候选清单
docs/p1_archive/P1_Final_Archive_and_P2_Plan_2026-05-27.md: P1 locked archive
docs/p1_archive/P1_Model_Comparison_Report_2026-05-27.md: P1 主表、消融和泄漏审计
docs/p0_reports/cola_adaptive_halt_paper_report_zh.md: P0 riskcap04 中文 canonical 报告
docs/cola_archive/README.md: CoLA 线完整归档入口，含 P0/P1/P2 artifact、权重、代码和复现边界
```

## Current Stage

```text
P1 locked evaluation completed.
CoLA P0/P1 archive completed on 2026-06-06. Use
`/data1/luyifei/drla/docs/cola_archive/README.md` as the canonical entry for
P0 adaptive halt teacher, P1 LatentHaltStudent-v1, checkpoint paths, summary
metrics, and P2 CoLA diagnostic freeze notes.
P3 Dream-DLM LatentMAS planning completed on 2026-06-06. The new main design is
homogeneous Dream-v0-Instruct-7B agents on MuSiQue evidence-split QA, with
Dream denoising-step readiness / halt as the P0/P1 migration target.
P3 D0/D1 local-only substrate preparation has started. Dream-v0-Instruct-7B was
downloaded and verified at `/data1/luyifei/drla/models/Dream-v0-Instruct-7B`.
The D0 artifact is
`/data1/luyifei/drla/outputs/p3_dream_models/Dream-org_Dream-v0-Instruct-7B_prepare_20260606_144309`.
The D1 generation/state probe artifact is
`/data1/luyifei/drla/outputs/p3_dream_models/dream_instruct_7b_generation_probe_20260606_144650`.
The probe generated the expected Paris answer with `steps=16`,
`max_new_tokens=32`, `output_history=True`, and confirmed online visibility of
history tokens, token hook snapshots, logits summaries, and a last-layer hidden
hook (`model.layers.27`) with peak allocated memory about 14.448 GiB on
`cuda:0`.
P3 D2 Dream TextMAS capability gate completed through maxctx4096 locked
held-out800. D2.1 smoke10 artifact:
`/data1/luyifei/drla/outputs/p3_dream_textmas_runs/dream_textmas_gate_samples10_20260606_145101`;
aggregate:
`/data1/luyifei/drla/outputs/p3_dream_textmas_aggregates/dream_textmas_gate_samples10_20260606_145101`.
D2.2 pilot50 merged artifact:
`/data1/luyifei/drla/outputs/p3_dream_textmas_runs/dream_textmas_gate_pilot50_merged_20260606`;
aggregate:
`/data1/luyifei/drla/outputs/p3_dream_textmas_aggregates/dream_textmas_gate_pilot50_merged_20260606`.
D2.3 full200 merged artifact:
`/data1/luyifei/drla/outputs/p3_dream_textmas_runs/dream_textmas_gate_full200_merged_20260606`;
aggregate:
`/data1/luyifei/drla/outputs/p3_dream_textmas_aggregates/dream_textmas_gate_full200_merged_20260606`;
leakage audit:
`/data1/luyifei/drla/outputs/p3_dream_protocol_audits/dream_textmas_gate_full200_merged_20260606`.
Full200 uses 200 calibration samples / 1400 rows with zero generation row
errors and zero leakage-audit errors. Aggregate gate is admitted:
`single_full_info=0.59`, `single_q_only=0.045`, `textmas_matched=0.465`,
`textmas_no_message=0.05`, `textmas_shuffled_message=0.11`,
`textmas_wrong_evidence_or_wrong_shard=0.10`; paired CI lower bounds are
`full_info_vs_question_only=0.475`, `matched_vs_no_message=0.34`,
`matched_vs_shuffled=0.28`, and `matched_vs_wrong_evidence=0.295`.
The initial maxctx2048 held-out queue exposed three `single_full_info` context
overflow rows and is therefore invalid as a formal held-out result. It is kept
only as a protocol-coverage diagnostic. Static context audits showed that
maxctx4096 covers both calibration full200 and held-out800 under the current
prompt/control schema.
The current protocol lock artifact is
`/data1/luyifei/drla/outputs/p3_dream_protocol_audits/dream_textmas_protocol_lock_calibration_full200_maxctx4096_20260606`;
it locks Dream-v0-Instruct-7B, `bfloat16`, `max_tokens=128`, `dream_steps=64`,
`temperature=0.2`, `top_p=0.95`, `alg=entropy`, `alg_temp=0.0`,
`max_context_tokens=4096`, parser `first_segment`, scorer
`score_qa_answer`, and the existing MuSiQue strict control definitions.
D2.5 maxctx4096 held-out800 merged artifact:
`/data1/luyifei/drla/outputs/p3_dream_textmas_runs/dream_textmas_gate_heldout800_maxctx4096_merged_20260606`;
aggregate:
`/data1/luyifei/drla/outputs/p3_dream_textmas_aggregates/dream_textmas_gate_heldout800_maxctx4096_merged_20260606`;
leakage audit:
`/data1/luyifei/drla/outputs/p3_dream_protocol_audits/dream_textmas_gate_heldout800_maxctx4096_merged_20260606`.
Held-out uses 800 samples / 5600 rows with zero generation row errors and zero
leakage-audit errors. Aggregate gate is admitted:
`single_full_info=0.49625`, `single_q_only=0.025`,
`textmas_matched=0.38125`, `textmas_no_message=0.0225`,
`textmas_shuffled_message=0.045`,
`textmas_wrong_evidence_or_wrong_shard=0.045`,
`textmas_compressed_state=0.35`; all conditions have `parseable_rate=1.0`.
Paired CI lower bounds are `full_info_vs_question_only=0.435`,
`matched_vs_no_message=0.32375`, `matched_vs_shuffled=0.29875`, and
`matched_vs_wrong_evidence=0.29875`. Dream TextMAS capability gate is therefore
complete; the next main step is D3 Dream denoising-step trace/frontier on
calibration/train split.
D3 trace collector has been added and smoke-tested at
`/data1/luyifei/drla/outputs/p3_dream_traces/musique_calibration_trace_smoke1_steps16_stride4_20260606`.
The smoke uses 1 calibration `single_full_info` row, `dream_steps=16`,
`max_tokens=32`, `snapshot_stride=4`, and records 1 solver trace call with
zero errors. The trace contains hook events `[0, 0, 4, 8, 12, 15]`; every event
has `trace_event_index` and `has_logit_stats` so D4 can distinguish repeated
Dream hook events from duplicated samples. This is a local-only tool/protocol
smoke, not a training result.
D3 trace pilot2 also completed at
`/data1/luyifei/drla/outputs/p3_dream_traces/musique_calibration_trace_pilot2_steps32_stride4_20260606`.
It uses 2 calibration samples, conditions `single_full_info,textmas_matched`,
`dream_steps=32`, `max_tokens=64`, and `snapshot_stride=4`. It produced 4
generation rows and 8 Dream trace calls with zero errors: 4 solver calls and 4
evidence-agent calls. All trace events have `trace_event_index` and
`has_logit_stats`; no `gold_answer` or `answer_aliases` fields appear in
`traces.jsonl`. This verifies the D3 collector can trace both direct solver and
Agent A -> solver data flow.
The D3 collector was then upgraded to save online Dream state features:
`--hidden-capture-mode summary` records last-layer generated-suffix hidden
statistics, and `--hidden-capture-mode suffix_tensor` writes suffix hidden
tensor refs under `hidden_refs/`. Hidden summary smoke:
`/data1/luyifei/drla/outputs/p3_dream_traces/musique_calibration_trace_hidden_summary_smoke1_steps16_stride4_20260606`;
hidden tensor smoke:
`/data1/luyifei/drla/outputs/p3_dream_traces/musique_calibration_trace_hidden_tensor_smoke1_steps8_stride4_20260606`.
The tensor smoke produced 8 hidden refs; a loaded example has shape
`(16, 3584)`, dtype `torch.float16`, scope `generated_suffix`.
D3 subset20 trace completed at
`/data1/luyifei/drla/outputs/p3_dream_traces/musique_calibration_trace_subset20_steps64_stride4_hidden_summary_20260606`.
It uses 20 calibration samples, conditions `single_full_info,textmas_matched`,
`dream_steps=64`, `max_tokens=128`, `snapshot_stride=4`, hidden summary mode,
and produced 40 generation rows / 80 Dream trace calls with zero errors.
Trace role distribution is 40 solver calls and 40 evidence-agent calls; event
hidden/logit coverage is 1360/1440 = 0.9444, and no gold/alias fields appear in
trace events.
D4 readiness frontier builder has been added and run on the subset20 trace:
`/data1/luyifei/drla/outputs/p3_dream_readiness_frontiers/musique_calibration_trace_subset20_steps64_stride4_hidden_summary_frontier_20260606`.
It writes 40 solver frontier rows and 720 solver step events, with zero missing
solver calls. On this subset, final primary scores are `single_full_info=0.6`
and `textmas_matched=0.4`; 50% of rows have an oracle correct-and-final-stable
step before the final event. This is an offline teacher/frontier diagnostic,
not an online policy result. Next D3/D4 action should shard full calibration
trace/frontier collection; single-process subset20 runtime shows full200 should
not be run serially. No Dream training has run yet in P3.
Full calibration D3/D4 has now completed with sharding. Queue artifact:
`/data1/luyifei/drla/outputs/p3_dream_traces/musique_calibration_trace_full200_steps64_stride4_hidden_summary_20260606_queue`;
merged trace:
`/data1/luyifei/drla/outputs/p3_dream_traces/musique_calibration_trace_full200_steps64_stride4_hidden_summary_merged_20260606`;
frontier:
`/data1/luyifei/drla/outputs/p3_dream_readiness_frontiers/musique_calibration_trace_full200_steps64_stride4_hidden_summary_frontier_20260606`.
The queue completed 20/20 shards with zero failed shards. The merge has 400
generation rows, 200 samples, 800 Dream trace calls, zero duplicate row/call
ids, and zero missing trace ids after shard-local call-id remapping. The
frontier has 400 solver rows and 7200 solver step events with zero missing
solver calls. Final primary by condition is `single_full_info=0.59` and
`textmas_matched=0.465`, matching the D2 calibration aggregate for these two
conditions. Hidden/logit event coverage is 0.9444. The oracle
correct-and-final-stable-before-final row rate is 0.5275, with mean first
correct-stable step about 4.72.
During D5 preparation, the first student run
`/data1/luyifei/drla/outputs/p3_dream_readiness_students/dream_step_readiness_student_v1_full200_seed20260606_20260606`
was marked superseded because the initial D4 frontier did not preserve the
`hidden_summary` payload, causing hidden feature statistics to be zero. The D4
frontier builder was fixed and the corrected frontier is
`/data1/luyifei/drla/outputs/p3_dream_readiness_frontiers/musique_calibration_trace_full200_steps64_stride4_hidden_summary_frontier_with_hidden_20260606`.
It keeps 6800/7200 events with non-empty hidden summaries while preserving the
same aggregate metrics. This corrected frontier is the current D5 training input;
all labels are offline teacher labels and must not be used as online features.
D5 DreamStepReadinessStudent-v1 with hidden/logit/process features completed at
`/data1/luyifei/drla/outputs/p3_dream_readiness_students/dream_step_readiness_student_v1_full200_with_hidden_seed20260606_20260606`.
This is a CUDA/GPU training run with SwanLab cloud run
`https://swanlab.cn/@Lyfff/drla-mvp/runs/0vw4qvu08rajphqgllk64`.
It used a causal trajectory Transformer over decoder-free features, multi-head
outputs for `ready`, `future_gain`, `prediction_change`, and `final_match`,
`valid_interval=10`, and saved both `best_checkpoint.pt` and
`last_checkpoint.pt`. Training finished at `global_step=400`, best step 350.
Best checkpoint metrics: valid `ready_auroc=0.9181`,
valid `ready_accuracy_at_05=0.8458`, test `ready_auroc=0.7946`, test
`ready_accuracy_at_05=0.7306`, test `final_match_auroc=0.9997`, and test
`prediction_change_auroc=0.9997`. This is the current D5 readiness student
artifact.
D5.5 local-only online halt calibration / risk-control evaluation completed at
`/data1/luyifei/drla/outputs/p3_dream_readiness_policy_eval/dream_step_readiness_student_v1_full200_with_hidden_policy_eval_20260606`.
It loads the D5 `best_checkpoint.pt`, selects thresholds on validation only,
and reports the internal test split without retuning. Selected policy:
`ready_threshold=0.05`, `final_match_threshold=0.7`,
`prediction_change_max=1.0`, `future_gain_max=999.0`. Internal test result:
final accuracy `0.50`, selected accuracy `0.50`, accuracy drop vs final `0.0`,
mean selected step `8.05`, mean step savings `54.95/63`, halt-before-final
rate `0.95`. Paired bootstrap 95% CIs: accuracy drop `[0.0, 0.0]`, mean step
savings `[50.525, 58.1]`, halt-before-final rate `[0.875, 1.0]`. This is a
D5 internal split policy audit, not a new held-out TextMAS claim. Next work can
start D6 latent packet construction, while preserving the rule that Agent A
latent state must be consumed by Agent B and must not bypass B through decoded
text or scorer-visible helper fields.
D6 latent packet substrate and packet manifest completed. Because the earlier
full200 trace only stored `hidden_summary`, a new local-only `suffix_tensor`
trace was run for `textmas_matched` rows:
`/data1/luyifei/drla/outputs/p3_dream_traces/musique_calibration_trace_textmas_matched200_steps64_stride4_hidden_tensor_20260606_queue`.
It completed 20/20 shards with zero failures. The merged trace is
`/data1/luyifei/drla/outputs/p3_dream_traces/musique_calibration_trace_textmas_matched200_steps64_stride4_hidden_tensor_merged_20260606`,
with 200 rows, 200 samples, 600 Dream trace calls, and 38,400 hidden tensor refs.
The tensor shards occupy about 33G. D6 packet builder output:
`/data1/luyifei/drla/outputs/p3_dream_latent_packets/dream_textmas_matched200_agent_ab_suffix_tensor_packets_v1_20260606`.
It builds `p3_dream_packet_v1_suffix_tensor` packets for upstream evidence agents:
200 packet groups, 400 packets, 200 `agent_a` packets, 200 `agent_b` packets,
and all 200 groups contain both agents. Packet audit status is pass:
missing refs `0`, missing traces `0`, forbidden packet key hits `0`. Mean
selected step is `21.98`; selected tensor example shape is `[128, 3584]`,
dtype `torch.float16`. Important caveat: the D5 policy was trained on solver
readiness labels, so applying it to evidence-agent traces is a D6 packet
step-selection heuristic, not a new evidence-agent readiness claim. The next
main work is D7 receiver/fuser integration with corruption controls.
D7 receiver/fuser integration has started and produced two alignment runs.
V1 MSE solver-state distillation:
`/data1/luyifei/drla/outputs/p3_dream_latent_fusers/dream_latent_fuser_v1_textmas_matched200_seed20260606_20260606`,
SwanLab run `https://swanlab.cn/@Lyfff/drla-mvp/runs/hg2otd5swqd3pzudh0k3b`.
It trained with CUDA/GPU, SwanLab cloud, `valid_interval=10`, `global_step=800`,
`best_step=570`, and saved `best_checkpoint.pt`, `last_checkpoint.pt`, and
`metrics.jsonl`. V1 is diagnostic only: train controls show packet use, but
valid/test controls do not pass; on test, `zero_packet` MSE is better than
matched MSE. This suggests V1 learned an average solver-state component and
does not prove receiver latent communication.
D7 V2 contrastive receiver/fuser:
`/data1/luyifei/drla/outputs/p3_dream_latent_fusers/dream_latent_fuser_v2_contrastive_textmas_matched200_seed20260606_20260606`,
SwanLab run `https://swanlab.cn/@Lyfff/drla-mvp/runs/k07305m849l5jnjyijlk4`.
It uses symmetric InfoNCE to align agent packet embeddings with same-row solver
latent embeddings. Training used CUDA/GPU, SwanLab cloud, `valid_interval=10`,
`global_step=800`, `best_step=780`, and saved required checkpoints/logs. Best
checkpoint metrics: valid packet-to-target top1 `0.71875`, target-to-packet
top1 `0.65625`; test packet-to-target top1 `0.75`, target-to-packet top1
`0.625`. V2 local-only controls completed at
`/data1/luyifei/drla/outputs/p3_dream_latent_fuser_controls/dream_latent_fuser_v2_contrastive_textmas_matched200_controls_20260606`.
On 20-row valid/test splits, random top1 is `0.05`; valid matched top1 is
`0.60` vs shuffled-row `0.05` and zero-packet `0.05`; test matched top1 is
`0.55` vs shuffled-row `0.00` and zero-packet `0.05`. Agent-swap remains close
to matched, so the current claim is row-specific latent alignment, not ordered
agent-role sensitivity. V2 is the current accepted D7 alignment result, but it
is not yet final answer generation or a complete LatentMAS main-table result.
D7 raw latent-prefix receiver generation diagnostic completed at
`/data1/luyifei/drla/outputs/p3_dream_latent_prefix_runs/dream_latent_prefix_eval_diag20_steps64_prefix8_20260606`.
This local-only evaluator prepends raw D6 suffix tensors as continuous Dream
inputs in a custom diffusion loop, with no Agent A/B decoded text messages in
the solver prompt. Scope: 20 `textmas_matched` rows, `max_tokens=128`,
`dream_steps=64`, `prefix_tokens_per_agent=8`, conditions `no_message`,
`latent_matched`, `latent_shuffled_row`, `latent_agent_swap`, and `latent_zero`.
Result: all variants have primary score `0.0`; all variants have token-F1 mean
`0.0333`. The generated answers are essentially unchanged across matched,
shuffled, zero, and no-message variants. Interpretation: directly prepending
last-layer suffix hidden states to Dream input embeddings fails because the
spaces are not aligned. This does not contradict V2 latent alignment; it
motivated the V3 embedding-space soft-prefix adapter below and remains evidence
against raw last-layer hidden prefixes.
D7 V3 embedding-space soft-prefix adapter completed on 2026-06-07. Training
script:
`/data1/luyifei/drla/drla/scripts/p3_train_dream_soft_prefix_adapter.py`.
Training artifact:
`/data1/luyifei/drla/outputs/p3_dream_soft_prefix_adapters/dream_soft_prefix_adapter_v1_textmas_matched200_seed20260607_20260607`.
SwanLab run:
`https://swanlab.cn/@Lyfff/drla-mvp/runs/1qoue1655x7820f73wvmm`.
It freezes Dream-v0-Instruct-7B, maps D6 agent_a/agent_b suffix tensors into a
16-token Dream input-embedding prefix, and trains the adapter through Dream
answer-token CE on the no-message solver prompt. Training obeyed CUDA/GPU,
SwanLab cloud, `valid_interval=10`, `global_step=480`, `best_step=460`, and
wrote `metrics.jsonl`, `best_checkpoint.pt`, and `last_checkpoint.pt`.
Best checkpoint loss-level result: valid matched CE `2.9495` vs zero CE
`6.1258`; test matched CE `2.7128` vs zero CE `5.7548`. This shows the
adapter learns a non-zero answer-token conditioning signal. However,
agent-swap CE remains close to matched (`best_test agent_swap_ce=2.6983` vs
matched `2.7128`), so the loss-level result does not prove ordered role
sensitivity.
D7 V3 receiver-side generation controls completed locally with
`/data1/luyifei/drla/drla/scripts/p3_run_dream_soft_prefix_eval.py`.
Evaluation artifact:
`/data1/luyifei/drla/outputs/p3_dream_soft_prefix_runs/dream_soft_prefix_eval_v1_best20_20260607`.
Scope: 20 `textmas_matched` calibration rows, `max_tokens=128`,
`dream_steps=64`, conditions `no_message`, `soft_prefix_matched`,
`soft_prefix_shuffled_row`, `soft_prefix_agent_swap`, and `soft_prefix_zero`.
No optimizer/backward/SwanLab run was used; no Agent A/B decoded text was
inserted into the solver prompt. Result: primary score is `0.0` for all
conditions. Token-F1 means are `no_message=0.0333`, `soft_prefix_zero=0.0333`,
`soft_prefix_shuffled_row=0.0833`, `soft_prefix_matched=0.1083`, and
`soft_prefix_agent_swap=0.1083`. Interpretation: V3 soft-prefix influences
receiver output and improves weak token overlap over no-message/zero, but it
does not pass the answer-generation gate and does not separate matched from
agent-swap. Do not claim LatentMAS success from V3. The next D7 repair should
move beyond a shallow input-embedding prefix toward native layer/KV integration,
cross-attention receiver conditioning, or an answer-selection receiver that is
explicitly trained/evaluated with matched-vs-shuffled-vs-zero controls.
D7 V4 native layer-conditioned receiver completed on 2026-06-07. Training
script:
`/data1/luyifei/drla/drla/scripts/p3_train_dream_layer_conditioned_receiver.py`.
This freezes Dream and injects packet conditioning at selected Dream layers
`[7, 14, 21, 27]` through learned cross-attention residual adapters over the
generated/masked positions. Training artifact:
`/data1/luyifei/drla/outputs/p3_dream_layer_receivers/dream_layer_receiver_v1_textmas_matched200_seed20260607_20260607`.
SwanLab run:
`https://swanlab.cn/@Lyfff/drla-mvp/runs/0798gogc7xnpd3hiqeyya`.
Training obeyed CUDA/GPU, SwanLab cloud, `valid_interval=10`,
`global_step=480`, `best_step=480`, and wrote `metrics.jsonl`,
`best_checkpoint.pt`, and `last_checkpoint.pt`. Best test loss-level result:
matched CE `1.5232`, zero CE `1.8341`, shuffled-row CE `1.6898`, agent-swap
CE `1.4997`. This is better than V3 on matched CE and gives positive
zero/shuffled margins, but agent-swap remains slightly better than matched,
so ordered role sensitivity is still not learned.
D7 V4 20-row receiver generation controls:
`/data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/dream_layer_receiver_eval_v1_best20_20260607`.
Result: `layer_receiver_matched` primary `0.10`, token-F1 `0.175`; all four
controls `no_message`, `layer_receiver_shuffled_row`, `layer_receiver_agent_swap`,
and `layer_receiver_zero` have primary `0.0`. This was the first P3 receiver
generation result with matched-only answer accuracy on 20 calibration rows.
Because 20 rows is too small, it was immediately expanded.
D7 V4 50-row receiver generation controls:
`/data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/dream_layer_receiver_eval_v1_best50_20260607`.
Scope: 50 `textmas_matched` calibration rows, 250 generations, local-only,
no optimizer/backward/SwanLab, and no Agent A/B decoded text inserted into the
solver prompt. Result: `layer_receiver_matched` primary `0.20`, token-F1
`0.2693`; `layer_receiver_zero` primary `0.20`, token-F1 `0.2593`;
`layer_receiver_agent_swap` primary `0.18`, token-F1 `0.2493`;
`layer_receiver_shuffled_row` primary `0.18`, token-F1 `0.2293`;
`no_message` primary `0.08`, token-F1 `0.0893`. Interpretation: V4 is a real
receiver-side generation improvement over no-message, but the 50-row controls
show most of the gain can be produced by the trained receiver with zero or
corrupted packets. Therefore V4 is not accepted as packet-specific LatentMAS
success. The next D7 repair must suppress receiver-prior leakage and optimize
matched-vs-corruption separation directly, for example with corruption-aware
training losses, explicit packet-use regularization, or a receiver-side
answer-selection/reranking objective whose zero/shuffled/agent-swap controls
cannot inherit the same learned answer prior.
D7 V5 corruption-aware layer receiver completed on 2026-06-07. Training script:
`/data1/luyifei/drla/drla/scripts/p3_train_dream_layer_receiver_corruption_aware.py`.
It keeps the V4 architecture but changes the objective to
`matched CE + margin(matched, zero/shuffled-row/agent-swap)`, so corrupted
packets are explicitly penalized when they predict the same gold answer too
well. Training artifact:
`/data1/luyifei/drla/outputs/p3_dream_layer_receivers/dream_layer_receiver_v2_corruptaware_textmas_matched200_seed20260607_20260607`.
SwanLab run:
`https://swanlab.cn/@Lyfff/drla-mvp/runs/gotls7ez7sgvnzyv2vmha`.
Training obeyed CUDA/GPU, SwanLab cloud, `valid_interval=10`,
`global_step=480`, `best_step=400`, and wrote `metrics.jsonl`,
`best_checkpoint.pt`, and `last_checkpoint.pt`. Best test loss-level result:
matched CE `2.1024`, zero CE `5.7847`, shuffled-row CE `2.1638`,
agent-swap CE `2.0692`. Final test result: matched CE `1.8225`, zero CE
`5.4697`, shuffled-row CE `1.9194`, agent-swap CE `1.8251`. Interpretation:
V5 strongly suppresses the zero-packet receiver prior and gives a modest
matched-vs-shuffled loss margin, but agent-swap remains essentially tied with
matched.
D7 V5 20-row receiver generation controls:
`/data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/dream_layer_receiver_v2_corruptaware_eval_best20_20260607`.
Scope: 20 `textmas_matched` calibration rows, 100 generations, local-only,
no optimizer/backward/SwanLab, and no Agent A/B decoded text inserted into the
solver prompt. Result: primary score is `0.0` for all conditions. Token-F1:
`no_message=0.0333`, `zero=0.0333`, `matched=0.0500`, `agent_swap=0.0500`,
`shuffled_row=0.0667`. Interpretation: V5 removes much of the V4
receiver-prior leakage, but it also removes the matched answer-generation gain.
Do not expand V5 to 50 rows or D8. The next D7 repair should not simply
increase the corruption margin; it should use a two-stage or multi-objective
receiver-side answer-selection/reranking setup that preserves matched
generation usefulness while enforcing matched-vs-corruption separation.
D7 V6 receiver-side answer-reranker diagnostic completed on 2026-06-07.
Candidate generation was built from receiver-generated outputs only, not from
private evidence text. The candidate merge/audit script is
`/data1/luyifei/drla/drla/scripts/p3_merge_dream_receiver_candidate_generations.py`.
Merged candidate artifact:
`/data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/dream_layer_receiver_eval_v1_best200_candidates_merged_20260607`.
The merge validates 200 unique rows, 1000 generations, 5 conditions, zero
duplicate row/condition pairs, zero missing row/condition pairs, and zero
forbidden payload keys. Full200 generation metrics: `layer_receiver_matched`
primary `0.195`, `agent_swap=0.190`, `shuffled_row=0.185`, `zero=0.115`,
`no_message=0.045`. Candidate-pool ceiling audit with aliases: all-condition
oracle primary `0.230`, matched-only oracle primary `0.195`, matched+no-message
oracle primary `0.220`, and matched+zero oracle primary `0.220`; average
all-condition candidate count is `2.71`. This means all-condition candidate
pool is a diagnostic upper bound and is not a clean online single matched-run
protocol.
D7 V6 training script:
`/data1/luyifei/drla/drla/scripts/p3_train_dream_receiver_answer_reranker.py`.
Smoke training artifact:
`/data1/luyifei/drla/outputs/p3_dream_answer_rerankers/dream_receiver_answer_reranker_v1_smoke1_seed20260607_20260607`,
SwanLab run `https://swanlab.cn/@Lyfff/drla-mvp/runs/3tjatglw6lu25g49kxapd`.
Full training artifact:
`/data1/luyifei/drla/outputs/p3_dream_answer_rerankers/dream_receiver_answer_reranker_v1_seed20260607_20260607`,
SwanLab run `https://swanlab.cn/@Lyfff/drla-mvp/runs/38fxxq335128q0vachhuh`.
Training obeyed CUDA/GPU, SwanLab cloud, `valid_interval=10`,
`global_step=3200`, `best_step=700`, and wrote `metrics.jsonl`,
`best_checkpoint.pt`, and `last_checkpoint.pt`. Best test metrics:
`matched_primary=0.20`, `zero_primary=0.25`, `shuffled_row_primary=0.25`,
`agent_swap_primary=0.20`, `matched_minus_zero=-0.05`,
`matched_minus_shuffled=-0.05`, `matched_minus_agent_swap=0.0`.
Final test metrics are also not packet-specific: matched/zero/shuffled/agent_swap
all have primary `0.15`. Interpretation: V6 is a negative diagnostic. It
confirms that a post-hoc candidate reranker over weak receiver-generated
candidates does not recover packet-specific LatentMAS behavior; it tends to
select by candidate text prior and candidate-pool ceiling rather than by
matched latent packet. Do not run held-out or D8 from V6. The next D7 repair
must return to receiver alignment/fusion itself, with an online-valid candidate
source if answer selection remains in scope.
Literature context checked on 2026-06-07 is consistent with this diagnosis:
recent latent-communication work emphasizes what state is communicated
(embedding/hidden/KV), which layer/alignment is used, and how the receiver
fuses the state; hybrid and trainable MAS protocols further emphasize
interactive receiver alignment rather than a disconnected post-hoc reranker.
D7 V7 V4-initialized zero/shuffled corruption fine-tune completed on
2026-06-07. Motivation: `agent_swap` is not necessarily a strict negative
because Agent A/B are both evidence agents and swapping packet order can
preserve the same evidence set. V7 therefore starts from V4 best checkpoint and
trains only against `zero` and `shuffled_row` corruptions, keeping agent-swap
as a diagnostic. Training script:
`/data1/luyifei/drla/drla/scripts/p3_train_dream_layer_receiver_corruption_aware.py`
with new `--init-checkpoint` and `--corruption-types` support. Smoke artifact:
`/data1/luyifei/drla/outputs/p3_dream_layer_receivers/dream_layer_receiver_v7_v4init_zeroshuf_smoke1_seed20260607_20260607`,
SwanLab run `https://swanlab.cn/@Lyfff/drla-mvp/runs/hxpqk3c71f3avad3y62vw`.
Full training artifact:
`/data1/luyifei/drla/outputs/p3_dream_layer_receivers/dream_layer_receiver_v7_v4init_zeroshuf_textmas_matched200_seed20260607_20260607`,
SwanLab run `https://swanlab.cn/@Lyfff/drla-mvp/runs/m2wnxo0pwzp49dcyl0d3n`.
Training obeyed CUDA/GPU, SwanLab cloud, `valid_interval=10`,
`global_step=480`, `best_step=470`, and wrote `metrics.jsonl`,
`best_checkpoint.pt`, and `last_checkpoint.pt`. Best test loss-level metrics:
matched CE `1.1486`, zero CE `1.6930`, shuffled-row CE `1.3863`,
agent-swap CE `1.1135`; zero margin `0.5444`, shuffled margin `0.2377`,
agent-swap margin `-0.0352`.
D7 V7 generation controls completed locally. 50-row artifact:
`/data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/dream_layer_receiver_v7_v4init_zeroshuf_eval_best50_20260607`.
50-row result: matched primary `0.22`, zero `0.16`, shuffled-row `0.16`,
agent-swap `0.22`, no-message `0.08`. Full200 merged artifact:
`/data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/dream_layer_receiver_v7_v4init_zeroshuf_eval_best200_merged_20260607`.
The merge validates 200 unique rows, 1000 generations, zero duplicate pairs,
zero missing pairs, and zero forbidden payload keys. Full200 result:
matched primary `0.215`, token-F1 `0.2789`; zero primary `0.095`, token-F1
`0.1358`; shuffled-row primary `0.180`, token-F1 `0.2486`; agent-swap primary
`0.210`, token-F1 `0.2777`; no-message primary `0.035`, token-F1 `0.0487`.
Interpretation: V7 is the current strongest D7 receiver. It clearly suppresses
the zero-packet receiver prior and modestly improves matched over shuffled-row,
while preserving matched generation. However the matched-vs-shuffled margin is
small and agent-swap is essentially tied with matched. Do not run D8 yet
without an explicit decision that agent-swap is a symmetry diagnostic rather
than a corruption negative, plus a locked held-out receiver/packet evaluation
or stronger calibration risk audit.
P3 text-encoded packet diagnostic was added on 2026-06-17 to test the hypothesis
that AgentB-side text-encoder hidden states should be easier to consume than
AgentA suffix tensors. Script:
`/data1/luyifei/drla/drla/scripts/p3_run_dream_text_encoded_packet_eval.py`.
It uses real D2 TextMAS `agent_messages`, encodes each agent message through
Dream last-layer hidden states, and feeds the resulting continuous packets to
the V7 receiver. The agent text is not inserted into the final solver prompt;
it is used only to construct diagnostic packets. Smoke5 artifact:
`/data1/luyifei/drla/outputs/p3_dream_text_encoded_packet_runs/dream_text_encoded_packet_eval_v7_smoke5_20260617`.
Merged20 artifact:
`/data1/luyifei/drla/outputs/p3_dream_text_encoded_packet_runs/dream_text_encoded_packet_eval_v7_merged20_20260617`.
The merged audit validates 20 unique rows, 100 generations, zero duplicate
pairs, zero missing pairs, and zero forbidden payload keys. Result:
`text_encoded_matched` primary `0.05`, `text_encoded_agent_swap=0.05`,
`text_encoded_shuffled_row=0.00`, `text_encoded_zero=0.00`, `no_message=0.00`.
On exactly the same 20 rows, the original TextMAS matched text channel has
primary `0.40` and token-F1 `0.42`. Interpretation: simply converting Agent
text into Dream hidden states and injecting those states through the current
V7 receiver does not recover TextMAS. This supports the diagnosis that the
missing component is not only AgentA suffix-hidden distribution mismatch; the
receiver/injection path itself needs explicit alignment or training on the
intended text-encoded / communication state distribution.
D7.6 text-packet adapter training was added on 2026-06-17 as the smallest
explicit receiver-side alignment repair for the D7.5 failure. Training script:
`/data1/luyifei/drla/drla/scripts/p3_train_dream_text_packet_adapter.py`.
It freezes Dream and the V7 receiver, encodes real TextMAS Agent messages into
Dream last-layer hidden packets, and trains a lightweight Transformer adapter
from text-hidden packet space into the V7 receiver packet space. Empty Agent
messages in four full200 rows are encoded as zero packets, preserving the fact
that the text channel carried no content instead of fabricating helper text.
Smoke artifact:
`/data1/luyifei/drla/outputs/p3_dream_text_packet_adapters/dream_text_packet_adapter_v1_v7_smoke1_seed20260617_20260617`,
SwanLab run `https://swanlab.cn/@Lyfff/drla-mvp/runs/wz0266vdk2jdcodgen1wi`.
Full training artifact:
`/data1/luyifei/drla/outputs/p3_dream_text_packet_adapters/dream_text_packet_adapter_v1_v7_seed20260617_20260617`,
SwanLab run `https://swanlab.cn/@Lyfff/drla-mvp/runs/ui9l5tllope0w1pik527t`.
Training obeyed CUDA/GPU, SwanLab cloud, `valid_interval=10`,
`global_step=800`, `best_step=760`, and wrote `metrics.jsonl`,
`best_checkpoint.pt`, and `last_checkpoint.pt`. Best loss-level metrics:
valid matched CE `0.7869`, valid zero CE `1.4472`, valid shuffled-row CE
`1.1283`; test matched CE `0.9926`, test zero CE `1.7216`, test shuffled-row
CE `1.5200`. Local-only generation eval artifact:
`/data1/luyifei/drla/outputs/p3_dream_text_encoded_packet_runs/dream_text_packet_adapter_v1_eval20_20260617`.
On the same 20 rows as D7.5, `text_adapter_matched` primary is `0.05` and
token-F1 is `0.17`; `text_adapter_agent_swap` primary is `0.05`, token-F1
`0.125`; `text_adapter_shuffled_row` primary is `0.00`, token-F1 `0.075`;
`text_adapter_zero` primary is `0.00`, token-F1 `0.075`; `no_message` primary
is `0.00`, token-F1 `0.0333`. Compared with D7.5 raw text-hidden matched
`0.05` primary / `0.125` token-F1, the adapter improves semantic overlap but
does not improve primary accuracy. Compared with same-row TextMAS matched
`0.40` primary / `0.42` token-F1, D7.6 still fails to recover text
communication. Interpretation: a receiver-side mapping into V7 packet space is
learnable at the CE level and slightly improves answer-token similarity, but it
does not yet create a robust, controllable Agent A/B -> receiver communication
protocol. Do not run D8 from D7.6; the next repair must explicitly optimize
end-task receiver behavior and matched-vs-control separation, not only
answer-token CE alignment.
D7.7 TextMAS-teacher layer receiver training was added on 2026-06-17 to test
whether explicit decoded-text teacher alignment can transfer TextMAS behavior
into the latent receiver without inserting Agent text at runtime. Training
script:
`/data1/luyifei/drla/drla/scripts/p3_train_dream_layer_receiver_text_teacher.py`.
It initializes from the V7 receiver, freezes Dream, trains the layer receiver
on D6 latent packets, and adds a training-only TextMAS teacher distribution
from same-row decoded Agent messages. The online student prompt remains the
no-message solver prompt; Agent text appears only in the teacher forward pass.
Smoke artifact:
`/data1/luyifei/drla/outputs/p3_dream_layer_receivers/dream_layer_receiver_d77_text_teacher_smoke1_seed20260617_20260617`,
SwanLab run `https://swanlab.cn/@Lyfff/drla-mvp/runs/2qlw93kku9oyotoge9w8l`.
Full training artifact:
`/data1/luyifei/drla/outputs/p3_dream_layer_receivers/dream_layer_receiver_d77_text_teacher_v7init_textmas_matched200_seed20260617_20260617`,
SwanLab run `https://swanlab.cn/@Lyfff/drla-mvp/runs/z6rl2y32n03xsdqpwlcs9`.
Training obeyed CUDA/GPU, SwanLab cloud, `valid_interval=10`,
`global_step=480`, `best_step=440`, and wrote `metrics.jsonl`,
`best_checkpoint.pt`, and `last_checkpoint.pt`. Best loss-level metrics:
valid matched CE `0.7733`, zero CE `1.6413`, shuffled-row CE `1.3716`,
teacher KL `2.1931`; test matched CE `0.8260`, zero CE `1.8806`,
shuffled-row CE `1.5745`, teacher KL `2.2488`. This is a clear loss-level
improvement over V7/D7.6. However local-only generation controls do not improve:
best checkpoint artifact
`/data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/dream_layer_receiver_d77_text_teacher_eval_best20_20260617`
has `layer_receiver_matched` primary `0.05`, token-F1 `0.1283`;
`agent_swap` primary `0.05`, token-F1 `0.1283`; `zero` primary `0.05`,
token-F1 `0.1083`; `shuffled_row` primary `0.00`, token-F1 `0.0833`;
`no_message` primary `0.00`, token-F1 `0.0333`. Last checkpoint artifact
`/data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/dream_layer_receiver_d77_text_teacher_eval_last20_20260617`
is worse for matched primary (`0.00`). On the same 20 rows, V7 matched was
`0.10` primary / `0.19` token-F1 and TextMAS matched was `0.40` primary /
`0.42` token-F1. Interpretation: teacher-forcing TextMAS KL/CE improves
offline logits and corruption margins but still does not transfer to the
diffusion generation behavior needed for latent communication. Do not run D8
from D7.7. The next D7 repair should stop treating teacher-forced answer-token
loss as sufficient evidence and instead optimize/evaluate online matched-channel
generation behavior directly, for example through matched-channel candidate
generation/selection with no corrupted-control candidates, or a generation-time
alignment objective.

D7.8 matched-channel candidate-pool diagnostic was added on 2026-06-17 to test
whether the current strongest raw receiver, V7, at least samples correct answers
under the online matched latent channel. Script:
`/data1/luyifei/drla/drla/scripts/p3_run_dream_layer_receiver_candidate_pool_eval.py`.
This is local-only evaluation, not training: no optimizer, no backward, no
SwanLab run, no decoded Agent text inserted into the solver prompt. It samples
8 candidates per row per condition on the same 20-row comparison set and reports
online-visible first/majority metrics plus an offline oracle ceiling. Full
artifact:
`/data1/luyifei/drla/outputs/p3_dream_layer_receiver_candidate_pools/dream_layer_receiver_v7_candidate_pool_best20_c8_20260617`.
The run completed 800 generations. One Dream tokenizer `None` token decode
edge case was fixed with `safe_decode`, and the script now supports `--resume`
so partial runs can continue without discarding generated rows.

D7.8 metrics:

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

Row-level audit: matched oracle-correct rows = 3/20, zero oracle-correct rows =
3/20. Two rows overlap between matched and zero; matched has only one unique
oracle-correct row, and one matched-correct row is also correct under
agent_swap. Therefore V7 matched latent channel has weak candidate-source
signal, but it is not packet-specific or stable enough to justify D8 or a
reranker as the main next step. Compared with same-row TextMAS matched
`0.40` primary / `0.42` token-F1, V7/D7.8 remains far behind.

Interpretation: the next D7 step should not assume that “same Dream model
latent distribution” is already compatible across agents. TextMAS works because
Agent A output is decoded to text and then re-enters Agent B through AgentB's
tokenizer, embedding stack, and diffusion generation dynamics. Direct D6 suffix
packets bypass that learned text-to-hidden interface. Before another receiver
training run, run a local-only interface/distribution audit comparing
TextMAS-conditioned AgentB hidden states with D6 packet tensors and the receiver
injection states; only after the mismatch is localized should we train a new
receiver/fuser.

D7.9 interface/distribution audit was added and run on 2026-06-17. Script:
`/data1/luyifei/drla/drla/scripts/p3_audit_dream_receiver_interface_distribution.py`.
This is local-only forward/statistics work: no generation, no optimizer, no
backward, no SwanLab run, no gold/scorer fields. It compares the same 20 rows
across:

```text
no-message AgentB solver prompt hidden states
TextMAS AgentB solver prompt hidden states with decoded Agent messages
D6 agent_a/agent_b suffix packet tensors
V7 PacketMemoryEncoder output
V7 selected-layer gated cross-attention deltas on masked solver positions
```

Artifacts:

```text
smoke2:
  /data1/luyifei/drla/outputs/p3_dream_receiver_interface_audits/
  dream_receiver_interface_audit_v7_smoke2_20260617

best20:
  /data1/luyifei/drla/outputs/p3_dream_receiver_interface_audits/
  dream_receiver_interface_audit_v7_best20_20260617
```

D7.9 best20 key metrics:

```text
TextMAS extra tokens entering AgentB prompt:
  mean = 238.9

mean-vector cosine:
  D6 packet -> no-message prompt hidden = 0.2947
  D6 packet -> TextMAS full prompt hidden = 0.5149
  D6 packet -> TextMAS last128 hidden = 0.5868
  TextMAS prompt -> no-message prompt hidden = 0.8608

token_norm_mean:
  D6 packet all = 223.25
  AgentB no-message prompt hidden = 324.41
  AgentB TextMAS prompt hidden = 320.54
  V7 receiver memory, after 3584->256 projection/Transformer = 34.82

receiver selected-layer gates:
  layer7 gate = 0.1211
  layer14 gate = 0.1221
  layer21 gate = 0.1245
  layer27 gate = 0.1235

gated delta / masked hidden norm ratio:
  layer7 = 0.3116
  layer14 = 0.2531
  layer21 = 0.2495
  layer27 = 0.0472
```

Interpretation: D6 packets are not pure noise; their mean direction is much
closer to TextMAS prompt hidden than to no-message prompt hidden. However, the
current V7 receiver does not pass the packet as AgentB's natural text-interface
hidden state. It compresses two 32-token 3584-d agent tensors into 64 tokens of
256-d memory, then injects the signal through small gated cross-attention
deltas. By the final selected layer, the packet-induced update is only about
4.7% of the masked hidden norm. This supports the observed D7.8 pattern:
matched has weak candidate-source signal but not enough stable causal influence
on Dream generation. The next receiver should be text-interface-aligned, for
example by learning a packet-to-AgentB-message-hidden bridge or stronger native
conditioning that preserves token-level structure and has generation-time
controls. Do not proceed to D8 from V7/D7.8/D7.9.

D7.10 text-interface virtual-message receiver was implemented and run on
2026-06-17. Motivation: D7.9 showed that D6 packets are closer to TextMAS
AgentB prompt hidden states than to no-message hidden states, but V7's 256-d
memory/gated cross-attention path was too weak. D7.10 therefore maps packet
tensors into continuous virtual message tokens at AgentB's solver interface.
This follows the general lesson from prefix-tuning / continuous prompt work
that continuous vectors must be placed where the frozen model can naturally
attend to them as virtual tokens, while preserving the LatentMAS/Coconut
requirement that runtime communication stays latent and decoder-free.

D7.10 scripts:

```text
training:
  /data1/luyifei/drla/drla/scripts/p3_train_dream_text_interface_receiver.py

generation eval:
  /data1/luyifei/drla/drla/scripts/p3_run_dream_text_interface_receiver_eval.py
```

Model design:

```text
inputs:
  D6 agent_a / agent_b suffix tensors
  shape after selection = 2 agents x 32 tokens x 3584 hidden

receiver:
  packet LayerNorm + 3584 -> d_model projection
  agent embedding + packet positional embedding
  2-layer Transformer memory encoder
  learned prefix queries cross-attend to packet memory
  MLP outputs prefix_len continuous virtual message tokens in 3584-d space

online generation:
  no-message AgentB solver prompt
  + virtual message prefix from latent packet
  + Dream diffusion masked answer generation
  decoded Agent text is not inserted

training-only teacher:
  TextMAS decoded Agent messages define TextMAS prompt hidden/logit targets
  gold answer tokens define supervised CE target
```

Smoke training passed:

```text
artifact:
  /data1/luyifei/drla/outputs/p3_dream_text_interface_receivers/
  dream_text_interface_receiver_d710_smoke2c_seed20260617_20260617

SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/eey3s8ssjeq9b79brmeh1

status:
  pass
  global_step = 2
  valid_interval = 1
  best_checkpoint.pt = present
  last_checkpoint.pt = present
  metrics.jsonl = present
```

Full D7.10 v1 training:

```text
artifact:
  /data1/luyifei/drla/outputs/p3_dream_text_interface_receivers/
  dream_text_interface_receiver_d710_v1_p96d1024_seed20260617_20260617

SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/eb1evnoegho5ez5f6qvje

config:
  prefix_len = 96
  d_model = 1024
  num_heads = 8
  global_step = 480
  best_step = 460
  valid_interval = 10
  SwanLab cloud = yes
  best_checkpoint.pt / last_checkpoint.pt / metrics.jsonl = present
```

Best checkpoint loss-level metrics:

```text
valid:
  matched_ce = 2.5406
  token_accuracy = 0.5683
  hidden_cosine_loss = 0.5361
  hidden_mse = 22.0105
  zero_ce_margin = 0.0403
  shuffled_row_ce_margin = -0.0042
  agent_swap_ce_margin = -0.0223

test:
  matched_ce = 2.8319
  token_accuracy = 0.5141
  hidden_cosine_loss = 0.5487
  zero_ce_margin = 0.0603
  shuffled_row_ce_margin = 0.0262
  agent_swap_ce_margin = -0.0024
```

Generation controls:

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

Row-level audit: for the best checkpoint, the only primary-correct row is
correct under matched, zero, shuffled-row, and agent-swap simultaneously.
Therefore D7.10 v1 does not provide packet-specific latent communication. It is
a useful negative diagnostic: virtual-message prefixing can train and slightly
improve token-F1, but the current objective still lets the receiver learn a
shared answer prior that survives corrupted packets. Do not proceed to D8 from
D7.10. Next D7 work must make the objective explicitly packet-specific at
generation time, for example by penalizing zero/shuffled/agent-swap answers that
match the same gold, using matched-vs-corruption contrastive hidden/logit
targets, or changing the receiver so corrupted packets cannot produce the same
virtual message prior.

D7.11 packet-specific text-interface receiver was implemented and run on
2026-06-17 by extending the D7.10 trainer with optional negative/contrastive
losses. It keeps the same online architecture and runtime boundary as D7.10,
but initializes from D7.10 best checkpoint and adds:

```text
corrupt_unlikelihood:
  penalize zero/shuffled/agent_swap packets for assigning high probability to
  the same gold answer tokens

logit_contrast:
  InfoNCE-style contrast where matched answer-token CE should beat zero,
  shuffled-row, and agent-swap CE

hidden_contrast:
  only matched prefix hidden should be closer to TextMAS teacher hidden than
  corrupted prefix hidden
```

Reference motivation: unlikelihood training explicitly lowers the probability
of negative tokens/sequences, and contrastive objectives are designed to keep
positive pairs close while pushing negatives apart. In this experiment those
ideas are applied only to training-time corrupted latent packets; runtime
communication stays decoder-free and does not insert Agent text.

Smoke:

```text
artifact:
  /data1/luyifei/drla/outputs/p3_dream_text_interface_receivers/
  dream_text_interface_receiver_d711_smoke2_seed20260617_20260617

SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/drls43zfkxm6c2f8s037t

status:
  pass
  global_step = 2
  init_checkpoint = D7.10 v1 best
  best/last/metrics present
```

Full D7.11 v1 training:

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
  corruption_weight = 0.5
  corruption_margin = 0.5
  corrupt_unlikelihood_weight = 0.5
  logit_contrast_weight = 1.0
  hidden_contrast_weight = 0.5
  contrast_temperature = 0.5
  global_step = 480
  best_step = 230
  valid_interval = 10
```

Loss-level result:

```text
best valid:
  matched_ce = 2.4047
  token_accuracy = 0.5529
  zero_ce = 9.1889
  zero_ce_margin = 6.7841
  shuffled_row_ce_margin = 0.0021
  agent_swap_ce_margin = -0.0240
  logit_contrast_loss = 1.1155
  hidden_contrast_loss = 1.2194

best test:
  matched_ce = 2.7797
  token_accuracy = 0.5281
  zero_ce = 9.1559
  zero_ce_margin = 6.3762
  shuffled_row_ce_margin = 0.0303
  agent_swap_ce_margin = 0.0030
```

Generation controls:

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

Interpretation: D7.11 fixed only half of the D7.10 failure. It successfully
suppresses the zero/shared virtual-prefix prior at the loss level and in
generation, but it also suppresses matched generation; matched primary drops
from D7.10's prior-driven 0.05 to 0.00. It still does not produce
packet-specific latent communication. The next experiment should not simply
increase negative weights. It needs a balanced objective or architecture that
preserves matched generation quality while preventing corrupted packets from
sharing the same answer prior. In particular, agent_swap remains a hard
diagnostic because evidence-agent order and role semantics are not strongly
distinguished by the current packet representation.

D7.12 balanced text-interface receiver was implemented and run on 2026-06-17.
It extends the D7.10/D7.11 trainer with negative-loss warmup and a checkpoint
selection metric that treats corruption margins as capped risk controls rather
than unbounded rewards. The goal was to preserve matched generation while
separating hard corrupted controls.

Script changes:

```text
drla/scripts/p3_train_dream_text_interface_receiver.py

new options:
  --negative-loss-warmup-steps
  --selection-token-accuracy-weight
  --selection-margin-target
  --selection-margin-overflow-penalty
```

Smoke:

```text
artifact:
  /data1/luyifei/drla/outputs/p3_dream_text_interface_receivers/
  dream_text_interface_receiver_d712_balanced_smoke2_seed20260617_20260617

SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/qoatplup9a5ekdr8bwydr

status:
  pass
  global_step = 2
  valid_interval = 1
  best/last/metrics present
```

Full D7.12 v1:

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
  valid_interval = 10
```

Loss-level result:

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

Generation controls:

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

Row-level audit:

```text
best checkpoint:
  matched-correct row = p2c_musique_calibration_2hop__585569_12907
  the same row is also correct under zero, shuffled-row, and agent-swap

last checkpoint:
  matched-correct row = p2c_musique_calibration_2hop__585569_12907
  the same row is also correct under agent-swap
  zero and shuffled-row are no longer correct on that row
```

Interpretation: D7.12 confirms that warmup and capped-margin selection can
avoid the D7.11 failure mode of destroying matched generation. It still does
not pass the D7 receiver gate: best remains prior-driven across all virtual
prefix conditions, and last only separates zero/shuffled while agent_swap stays
tied with matched. This reinforces the earlier V7 note that agent_swap may not
be a strict negative under homogeneous evidence-agent roles; until the protocol
has asymmetric roles, agent_swap should be reported as a symmetry/role
diagnostic rather than optimized as a hard corruption. The hard generation
controls remain zero and shuffled-row.

D7.13 receiver-control audit was added on 2026-06-17 to put V7 and the
D7.10-D7.12 text-interface branch under one paired metric contract.

```text
script:
  /data1/luyifei/drla/drla/scripts/
  p3_audit_dream_receiver_generation_controls.py

artifact:
  /data1/luyifei/drla/outputs/p3_dream_receiver_control_audits/
  dream_receiver_generation_control_audit_v7_d710_d711_d712_20260617

boundary:
  local-only
  no model loading or generation
  no optimizer/backward/weight update
  no SwanLab run

hard controls:
  no_message
  zero
  shuffled_row

diagnostic controls:
  agent_swap
```

Main audit result:

```text
V7 full200:
  matched primary = 0.215
  no_message primary = 0.035
  zero primary = 0.095
  shuffled_row primary = 0.180
  agent_swap primary = 0.210

V7 paired primary deltas:
  matched - no_message = 0.180, 95% bootstrap CI = [0.120, 0.240]
  matched - zero = 0.120, 95% bootstrap CI = [0.065, 0.180]
  matched - shuffled_row = 0.035, 95% bootstrap CI = [0.005, 0.070]
  matched - agent_swap = 0.005, 95% bootstrap CI = [-0.010, 0.025]

V7 row overlap:
  matched correct rows = 43 / 200
  zero correct rows = 19 / 200
  shuffled_row correct rows = 36 / 200
  agent_swap correct rows = 42 / 200
  matched unique over zero = 31 rows
  matched unique over shuffled_row = 9 rows
  matched unique over agent_swap = 2 rows
```

Comparison with text-interface receivers:

```text
D7.10 best/last:
  matched primary = 0.05
  zero/shuffled-row also = 0.05
  hard gate = fail

D7.11 best/last:
  matched primary = 0.00
  hard gate = fail

D7.12 best:
  matched primary = 0.05
  zero/shuffled-row also = 0.05
  hard gate = fail

D7.12 last:
  matched primary = 0.05
  zero/shuffled-row = 0.00
  paired CI lower remains 0.00 on 20 rows
  hard gate = fail
```

Interpretation: under the revised control taxonomy, V7 is the current strongest
receiver and the only one with positive paired primary CIs against the hard
controls on full200. However, it is not enough for D8: matched-vs-shuffled is
small, token-F1 matched-vs-shuffled has a CI crossing zero, and agent_swap
remains tied as a symmetry/role diagnostic. The text-interface branch is useful
as diagnostic evidence but should not be the next main receiver path unless its
matched generation can approach V7 while keeping zero/shuffled separated.

D7.14 held-out D6 packet readiness / evaluation was completed on 2026-06-17 to
check whether the current strongest receiver, V7 layer-conditioned zeroshuf,
generalizes from calibration full200 to the locked D2.5 held-out800 split
without breaking the protocol.

```text
script:
  /data1/luyifei/drla/drla/scripts/
  p3_audit_dream_heldout_packet_readiness.py

artifact:
  /data1/luyifei/drla/outputs/p3_dream_heldout_packet_preflights/
  dream_heldout_packet_readiness_preflight_20260617

boundary:
  local-only
  no model loading or generation
  no optimizer/backward/weight update
  no SwanLab run
```

Initial preflight result:

```text
status = blocked before substrate construction
can_run_v7_heldout_eval = false before substrate construction

available:
  held-out manifest = present
  held-out online inputs = present
  held-out TextMAS aggregate = present
  calibration suffix-tensor trace reference = present
  calibration D6 packet reference = present
  V7 best checkpoint = present

missing:
  held-out suffix-tensor trace =
    /data1/luyifei/drla/outputs/p3_dream_traces/
    musique_heldout_trace_textmas_matched800_steps64_stride4_hidden_tensor_merged_20260617
  held-out D6 packet manifest =
    /data1/luyifei/drla/outputs/p3_dream_latent_packets/
    dream_textmas_heldout800_agent_ab_suffix_tensor_packets_v1_20260617

held-out online rows:
  textmas_matched = 800
  all seven D2.5 conditions = 800 each

storage estimate:
  calibration suffix-tensor hidden refs = 38,400 files / 32.883 GiB
  estimated held-out raw suffix-tensor trace hidden refs = 131.533 GiB
  estimated held-out D6 selected packet refs = 1.370 GiB
  free disk before trace = 211.552 GiB
  estimated free disk after trace = 80.019 GiB
  disk budget check = pass with min_free_gib_after_trace=50
```

Held-out substrate construction:

```text
trace queue:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_heldout_trace_textmas_matched800_steps64_stride4_hidden_tensor_20260617_queue

trace queue status:
  pass
  completed shards = 80 / 80
  failed shards = 0

merged trace:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_heldout_trace_textmas_matched800_steps64_stride4_hidden_tensor_merged_20260617

merged trace status:
  pass
  rows = 800
  samples = 800
  traces = 2400
  duplicate row/call ids = 0
  missing trace ids = 0

held-out D6 packet manifest:
  /data1/luyifei/drla/outputs/p3_dream_latent_packets/
  dream_textmas_heldout800_agent_ab_suffix_tensor_packets_v1_20260617

packet status:
  pass
  packet groups = 800
  packets = 1600
  agent_a packets = 800
  agent_b packets = 800
  groups with two agents = 800
  missing refs / traces / forbidden keys = 0 / 0 / 0
  mean selected step = 21.534375
  total referenced hidden bytes = 1.471 GiB
```

Post-substrate preflight result:

```text
status = ready
can_run_v7_heldout_eval = true
missing required checks = []
advisory disk_budget_for_full_trace = false after construction
```

The disk-budget failure in the final preflight is advisory only: it asks whether
the full trace could be regenerated again from the now-lower free disk, not
whether the already-built held-out trace/packet substrate is valid.

The V7 held-out receiver eval script was also fixed before the full run. The
checkpoint config still records calibration packet paths, but runtime
`--manifest-json`, `--online-inputs-jsonl`, `--packet-dir`, and `--model-path`
now override those paths. Summaries write both `checkpoint_data_config` and
`runtime_data_config`, and the held-out smoke confirmed runtime data points to
the locked held-out split.

Held-out V7 receiver evaluation:

```text
checkpoint:
  /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
  dream_layer_receiver_v7_v4init_zeroshuf_textmas_matched200_seed20260607_20260607/
  best_checkpoint.pt

full eval shards:
  /data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/
  dream_layer_receiver_v7_heldout800_shard000_rows0000_0266_20260617
  dream_layer_receiver_v7_heldout800_shard001_rows0267_0533_20260617
  dream_layer_receiver_v7_heldout800_shard002_rows0534_0799_20260617

merged eval:
  /data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/
  dream_layer_receiver_v7_heldout800_merged_20260617

merge validation:
  status = pass
  generations = 4000
  unique rows = 800
  conditions = 5
  duplicates = 0
  missing pairs = 0
  forbidden payload hits = 0
```

Held-out condition means:

```text
primary_score_mean:
  matched = 0.02500
  no_message = 0.02375
  zero = 0.02875
  shuffled_row = 0.02375
  agent_swap = 0.02375

exact_match_mean:
  matched = 0.02125
  no_message = 0.01750
  zero = 0.02500
  shuffled_row = 0.02000
  agent_swap = 0.02000

token_f1_mean:
  matched = 0.09736
  no_message = 0.06618
  zero = 0.09462
  shuffled_row = 0.09646
  agent_swap = 0.09342
```

Held-out receiver-control audit:

```text
artifact:
  /data1/luyifei/drla/outputs/p3_dream_receiver_control_audits/
  dream_receiver_generation_control_audit_v7_heldout800_20260617

status:
  pass
  hard_gate_pass = false

hard controls:
  no_message
  zero
  shuffled_row

diagnostic controls:
  agent_swap
```

Paired held-out deltas:

```text
matched - no_message:
  primary delta = +0.00125, CI = [-0.00750, +0.01000]
  wins / ties / losses = 7 / 787 / 6
  token-F1 delta = +0.03117, CI = [+0.01848, +0.04353]

matched - zero:
  primary delta = -0.00375, CI = [-0.01000, +0.00125]
  wins / ties / losses = 1 / 795 / 4
  token-F1 delta = +0.00274, CI = [-0.00645, +0.01182]

matched - shuffled_row:
  primary delta = +0.00125, CI = [-0.00375, +0.00750]
  wins / ties / losses = 3 / 795 / 2
  token-F1 delta = +0.00090, CI = [-0.00672, +0.00866]

matched - agent_swap:
  primary delta = +0.00125, CI = [0.00000, +0.00375]
  wins / ties / losses = 1 / 799 / 0
  token-F1 delta = +0.00394, CI = [-0.00186, +0.01009]
```

Interpretation: V7 full200 calibration controls do not transfer to locked
held-out800. On held-out, matched does not beat the hard controls in primary
score: it is effectively tied with no-message and shuffled-row and is below
zero. Token-F1 improves over no-message, but not over zero or shuffled-row.
Therefore V7 held-out is a negative/blocked D8 result, not a latent
communication success. The next legitimate step is not to cite D7.13
calibration as held-out evidence; it is to diagnose why calibration packet
signal collapses under held-out distribution shift and why zero/receiver prior
remains competitive.

D7.15 failure-localization audit was added on 2026-06-17 after the V7 held-out
failure. It tests two hypotheses without training, generation, scoring new
outputs, or SwanLab: whether held-out packets are visibly out-of-distribution
at the interface level, and whether V7's calibration full200 gain is actually
concentrated in its training split.

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

combined failure-localization artifact:
  /data1/luyifei/drla/outputs/p3_dream_receiver_generalization_audits/
  dream_receiver_v7_d714_failure_localization_20260617
```

Interface comparison:

```text
packet_mean_cos_to_textmas_last128:
  calibration full200 = 0.61227
  held-out800 = 0.62397

packet_mean_cos_to_textmas_prompt:
  calibration full200 = 0.57396
  held-out800 = 0.58021

packet_mean_cos_to_no_message_prompt:
  calibration full200 = 0.37380
  held-out800 = 0.35674

receiver memory rms:
  calibration full200 = 2.26080
  held-out800 = 2.05353

layer delta/hidden norm ratio:
  layer7  calibration = 0.32080, held-out = 0.33114
  layer14 calibration = 0.26166, held-out = 0.27403
  layer21 calibration = 0.26140, held-out = 0.27498
  layer27 calibration = 0.04937, held-out = 0.05064
```

Interpretation: held-out packet/TextMAS hidden interface statistics are broadly
similar to calibration, and held-out injection ratios are not weaker. Layer27
injection remains tiny in both splits, but the held-out failure is not explained
by an obvious raw packet/TextMAS distribution mismatch.

V7 calibration full200 split-generalization audit:

```text
split sizes:
  train = 160
  valid = 20
  test = 20

train primary:
  matched = 0.24375
  no_message = 0.03125
  zero = 0.10000
  shuffled_row = 0.18750
  agent_swap = 0.23750
  hard_gate_pass = true

valid primary:
  matched = 0.00000
  no_message = 0.00000
  zero = 0.00000
  shuffled_row = 0.00000
  agent_swap = 0.00000
  hard_gate_pass = false

test primary:
  matched = 0.20000
  no_message = 0.10000
  zero = 0.15000
  shuffled_row = 0.30000
  agent_swap = 0.20000
  hard_gate_pass = false
```

Interpretation: D7.13 V7 full200 looked promising because the aggregate was
dominated by the 160-row training split. Nontrain splits do not pass: valid is
all zero, and test has shuffled_row above matched. This reframes D7.14 as a
generalization/selection failure rather than a missing-substrate or obvious
held-out hidden-distribution failure. Next receiver work must be selected by
nontrain calibration or a fresh non-heldout split before any locked held-out
run. Do not use held-out800 for tuning this branch.

D7.16 compact selected-suffix tensor substrate was added on 2026-06-17 to fix
the full `suffix_tensor` storage bottleneck before training any new receiver.
This is local-only trace/packet construction, not a training result and not a
LatentMAS success claim. The collector now supports
`--hidden-capture-mode selected_suffix_tensor`, which keeps per-step summaries
but writes only the D5-policy-selected evidence-agent suffix tensor for each
Dream call. This preserves the actual online tensor consumed by packet builders
without storing every denoising step's full suffix tensor.

Validation artifacts:

```text
direct smoke trace:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_calibration_trace_selected_suffix_tensor_smoke1_20260617

queue smoke merged trace:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_calibration_trace_selected_suffix_tensor_queue_smoke1_merged_20260617

queue smoke packets:
  /data1/luyifei/drla/outputs/p3_dream_latent_packets/
  dream_textmas_selected_suffix_tensor_queue_smoke1_packets_20260617

validdiag50 selected trace queue:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_validdiag50_trace_textmas_matched_selected_suffix_tensor_20260617_queue

validdiag50 selected trace merged:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_validdiag50_trace_textmas_matched_selected_suffix_tensor_merged_20260617

validdiag50 selected packets:
  /data1/luyifei/drla/outputs/p3_dream_latent_packets/
  dream_textmas_validdiag50_selected_suffix_tensor_packets_20260617
```

validdiag50 packet summary:

```text
trace rows / samples / calls:
  50 / 50 / 150

packet groups / packets:
  50 / 100

packet groups with both agents:
  50

selected step:
  min = 4
  mean = 33.58
  max = 63

tensor storage:
  mean selected hidden file size = 919611 bytes
  total referenced hidden bytes = 91961100

audit:
  missing refs = 0
  missing traces = 0
  forbidden packet key hits = 0
```

Storage check: the selected validdiag50 shards are about 27M each, while an old
full suffix tensor 10-row shard was about 1.7G. Therefore future large
non-heldout packet substrate should use selected-suffix mode by default. Full
suffix tensor traces remain useful for interface audits but should not be scaled
without an explicit storage budget.

Split boundary note: validdiag50 is an engineering validation set for compact
trace/packet construction, not a clean receiver-selection split against
train2000. A sample-id audit found 3 overlapping samples between train2000 and
validdiag50, while both have 0 overlap with the locked held-out800 manifest
`/data1/luyifei/drla/outputs/p2_phase_c_manifests/musique_heldout_manifest_800_seed20260605/manifest.json`.
If validdiag is used as a nontrain receiver gate, rebuild it disjointly or
filter the overlapping sample ids first.

The full nonheldout train2000 compact substrate was then collected and merged:

```text
trace queue:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_train2000_trace_textmas_matched_selected_suffix_tensor_20260617_queue

merged trace:
  /data1/luyifei/drla/outputs/p3_dream_traces/
  musique_train2000_trace_textmas_matched_selected_suffix_tensor_merged_20260617

packet artifact:
  /data1/luyifei/drla/outputs/p3_dream_latent_packets/
  dream_textmas_train2000_selected_suffix_tensor_packets_20260617
```

train2000 compact substrate summary:

```text
queue:
  requested shards = 40
  completed shards = 40
  failed shards = 0

merged trace:
  rows = 2000
  samples = 2000
  traces = 6000
  missing trace ids = 0
  duplicate row/call ids = 0

packet build:
  packet groups = 2000
  packets = 4000
  packet groups with both agents = 2000
  missing refs = 0
  missing traces = 0
  forbidden packet key hits = 0
  mean selected step = 35.31025
  min selected step = 0
  max selected step = 63
  total referenced hidden bytes = 3678444000
```

This artifact is now the main nonheldout receiver-training substrate. It still
does not prove latent communication: it only gives a larger decoder-free packet
input set. Any receiver trained on it must be evaluated with checkpoint-defined
valid/test splits and hard controls before held-out can be touched.

D7.16 train2000 receiver training was then run on top of the compact packet
substrate. It initializes from the earlier V7 V4-initialized zeroshuf receiver,
keeps Dream frozen, trains only the packet memory encoder and layer conditioners,
uses CUDA/GPU plus SwanLab cloud, and logs local `metrics.jsonl`,
`best_checkpoint.pt`, and `last_checkpoint.pt`. This is a training result, but
not a LatentMAS success claim unless generation controls pass.

Training artifact:

```text
output_dir:
  /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
  dream_layer_receiver_d716_train2000_v7init_zeroshuf_seed20260617

SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/7134z0gui6w8jek33rdt0

best checkpoint:
  /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
  dream_layer_receiver_d716_train2000_v7init_zeroshuf_seed20260617/
  best_checkpoint.pt

last checkpoint:
  /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
  dream_layer_receiver_d716_train2000_v7init_zeroshuf_seed20260617/
  last_checkpoint.pt
```

Training configuration:

```text
init_checkpoint:
  /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
  dream_layer_receiver_v7_v4init_zeroshuf_textmas_matched200_seed20260607_20260607/
  best_checkpoint.pt

split:
  train / valid / test = 1600 / 200 / 200

valid_interval:
  10

max_train_steps:
  600

learning_rate:
  3e-5

corruption_types:
  zero, shuffled_row

corruption_loss_weight / margin:
  0.5 / 0.5
```

Best checkpoint loss-level metrics:

```text
best_step:
  600

valid:
  matched_ce = 2.5632873698417096
  zero_ce_margin = 2.8676398239191623
  shuffled_row_ce_margin = 0.03806205857545164
  agent_swap_ce_margin = 0.0025234181527049593
  token_accuracy = 0.6134504672139883

test:
  matched_ce = 2.470198732819408
  zero_ce_margin = 2.879840268287808
  shuffled_row_ce_margin = 0.03120649160817246
  agent_swap_ce_margin = -0.005069603230804187
  token_accuracy = 0.6136382107436656
```

Interpretation at this level: D7.16 successfully learns the teacher-forcing
answer-token CE objective and strongly separates zero packets. It weakly
separates shuffled-row packets and does not learn an ordered agent-role signal.
This is useful but insufficient, because the real LatentMAS gate is sampled
Dream generation under hard controls.

D7.16 checkpoint-defined valid generation gate was run local-only:

```text
run_dir:
  /data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/
  dream_layer_receiver_d716_train2000_v7init_zeroshuf_valid200_20260617

audit_dir:
  /data1/luyifei/drla/outputs/p3_dream_receiver_control_audits/
  dream_receiver_generation_control_audit_d716_train2000_valid200_20260617

selection:
  --split valid
  max_rows = 200

execution boundary:
  local-only eval/audit
  no optimizer/backward/weight update
  no SwanLab
  no decoded agent text inserted into solver prompt
```

Generation metrics:

```text
no_message:
  primary = 0.055
  exact = 0.025
  token_f1 = 0.08172222222222222

matched:
  primary = 0.040
  exact = 0.025
  token_f1 = 0.10878282828282826

zero:
  primary = 0.045
  exact = 0.030
  token_f1 = 0.10434126984126983

shuffled_row:
  primary = 0.035
  exact = 0.025
  token_f1 = 0.1118247863247863

agent_swap:
  primary = 0.040
  exact = 0.025
  token_f1 = 0.11955205905205903
```

Paired hard-control audit:

```text
hard_gate_pass:
  false

matched - no_message:
  primary_delta_mean = -0.015
  primary_delta_ci = [-0.040, 0.005]
  wins / ties / losses = 1 / 195 / 4
  token_f1_delta_mean = +0.027060606060606056

matched - zero:
  primary_delta_mean = -0.005
  primary_delta_ci = [-0.035, 0.025]
  wins / ties / losses = 4 / 191 / 5
  token_f1_delta_mean = +0.004441558441558442

matched - shuffled_row:
  primary_delta_mean = +0.005
  primary_delta_ci = [0.000, 0.015]
  wins / ties / losses = 1 / 199 / 0
  token_f1_delta_mean = -0.003041958041958042

matched - agent_swap:
  diagnostic only
  primary_delta_mean = 0.000
  token_f1_delta_mean = -0.010769230769230769
```

Prediction-similarity diagnostic was saved at:

```text
/data1/luyifei/drla/outputs/p3_dream_receiver_control_audits/
dream_receiver_generation_control_audit_d716_train2000_valid200_20260617/
prediction_similarity.json
```

Key similarity findings:

```text
matched vs no_message:
  identical prediction = 72 / 200 = 36.0%
  identical primary score = 195 / 200 = 97.5%
  matched-unique correct rows = 1
  no_message-unique correct rows = 4

matched vs zero:
  identical prediction = 67 / 200 = 33.5%
  identical primary score = 191 / 200 = 95.5%

matched vs shuffled_row:
  identical prediction = 116 / 200 = 58.0%
  identical primary score = 199 / 200 = 99.5%

matched vs agent_swap:
  identical prediction = 140 / 200 = 70.0%
  identical primary score = 200 / 200 = 100.0%
```

D7.16 conclusion: the larger compact train2000 substrate and V7-initialized
corruption-aware training improve loss-level packet/control separation, but the
effect still does not become a stable answer source during sampled denoising
generation. In particular, matched does not beat no-message or zero on valid
primary, barely beats shuffled-row on primary, and is indistinguishable from
agent-swap on correctness. Do not run held-out or D8 from this checkpoint. The
next receiver experiment should not be "more of the same" teacher-forcing CE;
it should align the objective and injection mechanism with inference-time Dream
denoising, for example by supervising packet-conditioned changes along rollout
states, strengthening online conditioning where generation actually branches,
and keeping hard controls in the training/evaluation contract.

D7.17 denoising sensitivity audit was added on 2026-06-18 to locate the D7.16
failure inside the Dream generation loop. This is a local-only audit: it loads
the D7.16 best checkpoint, uses checkpoint-defined valid rows, compares matched
packets against controls on the same intermediate denoising state, and writes
row/step metrics without training or SwanLab. It does not insert decoded agent
text into the solver prompt. Gold/scorer fields are used only for offline final
matched-trajectory scoring.

Script:

```text
/data1/luyifei/drla/drla/scripts/
p3_audit_dream_receiver_denoising_sensitivity.py
```

Smoke artifact:

```text
/data1/luyifei/drla/outputs/p3_dream_receiver_sensitivity_audits/
dream_receiver_d716_valid2_steps8_smoke_20260618
```

Valid50 artifact:

```text
/data1/luyifei/drla/outputs/p3_dream_receiver_sensitivity_audits/
dream_receiver_d716_valid50_steps64_max128_20260618
```

Valid50 configuration:

```text
checkpoint:
  /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
  dream_layer_receiver_d716_train2000_v7init_zeroshuf_seed20260617/
  best_checkpoint.pt

split:
  checkpoint-defined valid

rows / steps / max_tokens:
  50 / 64 / 128

step-control records:
  12800

execution boundary:
  local-only
  no optimizer/backward/weight update
  no SwanLab
  no decoded agent text inserted into solver prompt
```

Aggregate sensitivity metrics:

```text
matched_shared_state:
  primary = 0.020
  exact = 0.020
  token_f1 = 0.1297979797979798

matched vs no_message:
  top1_disagree = 0.008185529597103596
  transfer_top1_disagree = 0.006093749925494194
  top1_same = 0.9918144636973739
  transfer_top1_same = 0.9939062500745058

matched vs zero:
  top1_disagree = 0.008204307057021652
  transfer_top1_disagree = 0.007031249916180969
  top1_same = 0.9917956862505526
  transfer_top1_same = 0.992968750083819

matched vs shuffled_row:
  top1_disagree = 0.002695536487735808
  transfer_top1_disagree = 0.0018229165952652693
  top1_same = 0.9973044545017182
  transfer_top1_same = 0.9981770834047348

matched vs agent_swap:
  top1_disagree = 0.0016112422474543564
  transfer_top1_disagree = 0.0012499999348074197
  top1_same = 0.9983887483365834
  transfer_top1_same = 0.9987500000651925
```

Step-band readout:

```text
steps 0-15:
  no_message transfer_disagree = 0.01875
  zero transfer_disagree = 0.02125
  shuffled_row transfer_disagree = 0.001875
  agent_swap transfer_disagree = 0.00000

steps 16-31:
  no_message transfer_disagree = 0.000625
  zero transfer_disagree = 0.000625
  shuffled_row transfer_disagree = 0.00000
  agent_swap transfer_disagree = 0.00000

steps 32-47:
  no_message transfer_disagree = 0.00000
  zero transfer_disagree = 0.00000
  shuffled_row transfer_disagree = 0.00000
  agent_swap transfer_disagree = 0.00000

steps 48-63:
  no_message transfer_disagree = 0.00500
  zero transfer_disagree = 0.00625
  shuffled_row transfer_disagree = 0.005417
  agent_swap transfer_disagree = 0.00500
```

D7.17 conclusion: D7.16's packet-conditioned receiver rarely changes the token
that Dream would write at each denoising step. Even zero/no-message controls
agree with matched on more than 99% of transfer decisions; shuffled-row and
agent-swap are even closer. This localizes the failure below the final scorer:
the current receiver can improve teacher-forced answer CE/margins, but its
online perturbation is not strong or targeted enough to change denoising
decisions. The next receiver design should directly optimize or guide
packet-conditioned denoising decisions under hard controls, not merely extend
the same answer-token CE objective.

D7.18-v1 denoising-aligned receiver was implemented and run on 2026-06-18.
It keeps the D7.16 frozen-Dream layer-conditioned receiver architecture but
changes the training state and objective:

```text
training state:
  partial-denoising answer states
  mask ratios = 1.0, 0.75, 0.5, 0.25

positive objective:
  matched packet CE on masked answer positions

hard-control objective:
  matched-vs-zero gold-token logit margin
  matched-vs-shuffled-row gold-token logit margin
  optional matched-gold-vs-control-top margin

diagnostic boundary:
  agent_swap remains diagnostic only
  no decoded agent text in solver prompt
  gold answers are supervised targets only
```

Script:

```text
/data1/luyifei/drla/drla/scripts/
p3_train_dream_layer_receiver_denoising_aligned.py
```

Smoke artifacts:

```text
/data1/luyifei/drla/outputs/p3_dream_layer_receivers/
dream_layer_receiver_d718_smoke2_denoising_aligned_seed20260618

/data1/luyifei/drla/outputs/p3_dream_layer_receivers/
dream_layer_receiver_d718_smoke1_deterministic_eval_seed20260618
```

The first smoke exposed that validation masks must be deterministic for stable
checkpoint selection; the trainer was patched so evaluation uses deterministic
partial-mask patterns while training remains randomly scheduled.

D7.18-v1 screen200 training artifact:

```text
output_dir:
  /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
  dream_layer_receiver_d718_screen200_denoising_aligned_seed20260618

SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/5f7ynp5opf6j61cqpbxyy

best checkpoint:
  /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
  dream_layer_receiver_d718_screen200_denoising_aligned_seed20260618/
  best_checkpoint.pt

last checkpoint:
  /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
  dream_layer_receiver_d718_screen200_denoising_aligned_seed20260618/
  last_checkpoint.pt
```

Training configuration:

```text
init_checkpoint:
  D7.16 train2000 best checkpoint

split:
  train / valid / test = 1600 / 200 / 200

max_train_steps:
  200

valid_interval:
  10

learning_rate:
  3e-5

decision_margin / decision_loss_weight:
  1.0 / 0.5

control_top_margin / control_top_loss_weight:
  0.5 / 0.1

corruption_types:
  zero, shuffled_row
```

Best checkpoint metrics:

```text
best_step:
  200

valid:
  matched_ce = 1.7278096367617877
  matched_token_accuracy = 0.6994468848593534
  hard_gold_margin_mean = 1.4301884883403544
  zero_gold_margin = 2.8303708081692456
  shuffled_row_gold_margin = 0.03000616851146333
  decision_loss = 0.570474853515625
  control_top_loss = 1.3755816650390624
  selection_metric = -1.6408261639043147

test:
  matched_ce = 1.6313846607612956
  matched_token_accuracy = 0.7104631761461496
  hard_gold_margin_mean = 1.4699135826078418
  zero_gold_margin = 2.898188375737518
  shuffled_row_gold_margin = 0.041638789478165565
  decision_loss = 0.5647116088867188
  control_top_loss = 1.2701894378662109
  selection_metric = -1.5380326717021715
```

Training interpretation: D7.18-v1 validates the partial-denoising objective in
one sense: matched CE and zero-packet margins improve substantially compared
with D7.16. However, row-specific shuffled-row margin barely moves. The average
hard margin is dominated by zero separation, so the objective is still not
forcing enough sample-specific packet binding.

D7.18-v1 denoising sensitivity audit:

```text
artifact:
  /data1/luyifei/drla/outputs/p3_dream_receiver_sensitivity_audits/
  dream_receiver_d718_screen200_valid50_steps64_max128_20260618

scope:
  local-only
  checkpoint-defined valid50
  64 denoising steps
  128 generated-token budget
  12800 step-control records
```

Sensitivity metrics:

```text
matched_shared_state:
  primary = 0.100
  exact = 0.040
  token_f1 = 0.21272168284789644

matched vs no_message:
  top1_disagree = 0.01940115470002638
  transfer_top1_disagree = 0.010885416604578495

matched vs zero:
  top1_disagree = 0.01982994086603867
  transfer_top1_disagree = 0.0054687499441206456

matched vs shuffled_row:
  top1_disagree = 0.004630239804100711
  transfer_top1_disagree = 0.001302083283662796

matched vs agent_swap:
  top1_disagree = 0.0019041202749940566
  transfer_top1_disagree = 0.0009374999534338713
```

Sensitivity interpretation: D7.18-v1 increases no-message disagreement relative
to D7.17 (`0.00609 -> 0.01089` on transfer decisions), but shuffled-row
disagreement does not improve (`0.00182 -> 0.00130`). This confirms the training
readout: the new objective makes the receiver more sensitive to packet presence,
but not more sensitive to which row's packet is present.

D7.18-v1 valid50 generation controls:

```text
run:
  /data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/
  dream_layer_receiver_d718_screen200_valid50_20260618

audit:
  /data1/luyifei/drla/outputs/p3_dream_receiver_control_audits/
  dream_receiver_generation_control_audit_d718_screen200_valid50_20260618
```

Generation metrics:

```text
no_message:
  primary = 0.08
  exact = 0.02
  token_f1 = 0.116984126984127

matched:
  primary = 0.12
  exact = 0.06
  token_f1 = 0.23272168284789643

zero:
  primary = 0.12
  exact = 0.04
  token_f1 = 0.16306459948320412

shuffled_row:
  primary = 0.10
  exact = 0.04
  token_f1 = 0.212

agent_swap:
  primary = 0.08
  exact = 0.02
  token_f1 = 0.19570716510903427
```

Paired audit:

```text
hard_gate_pass:
  false

matched - no_message:
  primary_delta_mean = +0.04
  primary_delta_ci = [0.00, +0.10]
  wins / ties / losses = 2 / 48 / 0

matched - zero:
  primary_delta_mean = 0.00
  primary_delta_ci = [-0.08, +0.08]
  wins / ties / losses = 2 / 46 / 2

matched - shuffled_row:
  primary_delta_mean = +0.02
  primary_delta_ci = [0.00, +0.06]
  wins / ties / losses = 1 / 49 / 0
```

D7.18-v1 conclusion: the partial-denoising/decision-margin repair is directionally
useful, but it is not a valid receiver. It improves matched generation over
no-message and slightly over shuffled-row on valid50, but it ties zero on primary
and does not materially increase shuffled-row transfer-token disagreement.
Therefore do not run held-out or D8 from D7.18-v1. The next repair should make
row-specific shuffled-row separation a first-class objective, for example by
up-weighting shuffled-row margins separately from zero, adding row-contrastive
packet binding, or using retrieval-style packet-to-answer/state alignment before
or alongside denoising decision losses.

D7.19 row-binding weighted denoising-aligned receiver was implemented and run on
2026-06-18. It keeps the D7.18 partial-denoising training state, but changes the
objective and checkpoint selection so shuffled-row / row-specific binding is no
longer averaged away by the easier zero-packet separation.

Code changes:

```text
script:
  /data1/luyifei/drla/drla/scripts/
  p3_train_dream_layer_receiver_denoising_aligned.py

new training controls:
  --decision-control-weights
  --top-control-weights
  --selection-mode row_binding

extra guard:
  non-empty train/valid/test split check
```

D7.19 screen200 training artifact:

```text
output_dir:
  /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
  dream_layer_receiver_d719_screen200_row_binding_seed20260618

SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/ub06nr5p8ddq3p2fgdenw

best checkpoint:
  /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
  dream_layer_receiver_d719_screen200_row_binding_seed20260618/
  best_checkpoint.pt

last checkpoint:
  /data1/luyifei/drla/outputs/p3_dream_layer_receivers/
  dream_layer_receiver_d719_screen200_row_binding_seed20260618/
  last_checkpoint.pt
```

Training configuration:

```text
init_checkpoint:
  D7.18-v1 best checkpoint

split:
  train / valid / test = 1600 / 200 / 200

max_train_steps:
  200

valid_interval:
  10

learning_rate:
  2e-5

decision_margin / decision_loss_weight:
  1.0 / 1.0

control_top_margin / control_top_loss_weight:
  0.5 / 0.2

decision_control_weights:
  zero = 0.15
  shuffled_row = 4.0

top_control_weights:
  zero = 0.05
  shuffled_row = 2.0

selection_mode:
  row_binding
```

Best checkpoint metrics:

```text
best_step:
  200

valid:
  matched_ce = 1.7381578401603024
  matched_token_accuracy = 0.6977861981652677
  zero_gold_margin = 2.977695965440944
  shuffled_row_gold_margin = 0.06667653079697629
  shuffled_row_top_margin = -1.063897340404801
  decision_loss = 0.9688546752929688
  control_top_loss = 1.6566759109497071
  selection_metric = -1.8808453747144656

test:
  matched_ce = 1.6379975606159973
  matched_token_accuracy = 0.7087592168152332
  zero_gold_margin = 2.9921453844895587
  shuffled_row_gold_margin = 0.06340354728978127
  shuffled_row_top_margin = -0.9445282942068297
  decision_loss = 0.9919075441360473
  control_top_loss = 1.5656114578247071
  selection_metric = -1.7558944928309617
```

Training interpretation: D7.19 does move the loss-level row-binding readout in
the intended direction. Valid shuffled-row gold margin rises from D7.18-v1's
`0.0300` to `0.0667`, and test shuffled-row gold margin rises from `0.0416` to
`0.0634`. However, the absolute margin is still tiny and shuffled-row top margin
remains strongly negative. This means the matched packet often improves the
gold-token logit only slightly, while the control top token remains competitive.

D7.19 denoising sensitivity audit:

```text
artifact:
  /data1/luyifei/drla/outputs/p3_dream_receiver_sensitivity_audits/
  dream_receiver_d719_screen200_valid50_steps64_max128_20260618

scope:
  local-only
  checkpoint-defined valid50
  64 denoising steps
  128 generated-token budget
  12800 step-control records
```

Sensitivity metrics:

```text
matched_shared_state:
  primary = 0.100
  exact = 0.040
  token_f1 = 0.19994871794871794

matched vs no_message:
  top1_disagree = 0.023465993327263276
  transfer_top1_disagree = 0.010833333283662795

matched vs zero:
  top1_disagree = 0.018538119594741147
  transfer_top1_disagree = 0.0064062499441206455

matched vs shuffled_row:
  top1_disagree = 0.004466023079294246
  transfer_top1_disagree = 0.0016666666232049464

matched vs agent_swap:
  top1_disagree = 0.0033019181326380933
  transfer_top1_disagree = 0.0008333333022892475
```

Sensitivity interpretation: compared with D7.18-v1, D7.19 slightly improves
matched-vs-shuffled-row transfer disagreement (`0.00130 -> 0.00167`) and
matched-vs-zero transfer disagreement (`0.00547 -> 0.00641`), but no-message is
essentially unchanged (`0.01089 -> 0.01083`). The row-binding loss therefore
creates a small logit/decision perturbation but still does not produce a strong
row-specific packet effect during sampled Dream denoising.

D7.19 valid50 generation controls:

```text
run:
  /data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/
  dream_layer_receiver_d719_screen200_valid50_20260618

audit:
  /data1/luyifei/drla/outputs/p3_dream_receiver_control_audits/
  dream_receiver_generation_control_audit_d719_screen200_valid50_20260618
```

Generation metrics:

```text
no_message:
  primary = 0.08
  exact = 0.02
  token_f1 = 0.116984126984127

matched:
  primary = 0.10
  exact = 0.04
  token_f1 = 0.19994871794871794

zero:
  primary = 0.12
  exact = 0.04
  token_f1 = 0.16015873015873017

shuffled_row:
  primary = 0.10
  exact = 0.04
  token_f1 = 0.1986153846153846

agent_swap:
  primary = 0.10
  exact = 0.04
  token_f1 = 0.18636108908273855
```

Paired audit:

```text
hard_gate_pass:
  false

matched - no_message:
  primary_delta_mean = +0.02
  primary_delta_ci = [0.00, +0.06]
  wins / ties / losses = 1 / 49 / 0

matched - zero:
  primary_delta_mean = -0.02
  primary_delta_ci = [-0.08, +0.04]
  wins / ties / losses = 1 / 47 / 2

matched - shuffled_row:
  primary_delta_mean = 0.00
  primary_delta_ci = [0.00, 0.00]
  wins / ties / losses = 0 / 50 / 0

matched - agent_swap:
  primary_delta_mean = 0.00
  primary_delta_ci = [0.00, 0.00]
  wins / ties / losses = 0 / 50 / 0
```

D7.19 conclusion: row-binding weighted loss is a useful diagnostic but not a
valid receiver. It improves the teacher-forced shuffled-row margin and creates a
small sensitivity uptick, yet generation remains non-packet-specific: matched is
identical to shuffled-row on primary for all 50 rows and is below zero-packet
primary. Do not run held-out or D8 from D7.19. The next receiver attempt should
not merely increase the same CE/margin weights or steps; it should first revisit
row-identity architecture, stronger fusion/injection, trajectory-level
packet-conditioned guidance, or a receiver objective whose nontrain gate is
directly tied to matched-vs-shuffled generation differences.

For new experiments, P3 Dream-DLM supersedes the old P2 CoLA route. P2 entries
below remain historical evidence and protocol-boundary references.
P2 packet v1 substrate completed.
P2-A packet v2 rebuild completed.
P2-B distribution audit completed.
P2-C decoder-free compatibility completed.
P2-D message_only receiver-only official8 completed as channel diagnostic.
P2 main goal reset on 2026-06-01:
  same-substrate role-conditioned Cola A -> Cola B latent communication,
  with capability-gated benchmarks and paper-level text-vs-latent controls.
P2-D1 capability-gate scripts and candidate benchmark data prepared.
P2-D1 formal full gate completed: no candidate task admitted for P2 main.
P2-D2 locked calibration/held-out split completed.
P2-D3 initial prompt-variant calibration sweep completed; no task admitted.
P2-D3.1 answer-state protocol repair completed on calibration; no task admitted.
P2-D4 branch decision audit completed: recommendation is benchmark redesign
first, unless ARC/GPQA/MedQA/GSM8K are non-negotiable and require substrate
adaptation.
Branch B benchmark redesign candidate inventory completed as safe prep work.
Branch B execution plan locked: frozen official CoLA first, official8-compatible
role candidates first, Single + Role capability gate before any text-vs-latent
main table.
Branch B official8-compatible calibration first pass completed: no task admitted;
next step is official CoLA prompt/eval alignment audit before any held-out gate
or latent/text main experiment.
Official8 native prompt/eval alignment audit completed: native official Single
Solver still admits no task on calibration; Branch B Family 1 should stop.
Post-Family1 branch decision memo completed.
Post-Family1 complete execution plan completed. Old next-phase/Branch B documents
are superseded as current next-step guidance. Do not rerun held-out, P2 main
tables, or fuser/adapter training from the stopped Family 1 path.
P2 locked complete execution scheme completed: the default route is now Branch C
true MAS benchmark/protocol validation -> Branch A CoLA substrate/interface
adaptation -> Phase E CoLA TextMAS vs LatentMAS paper-level comparison.
Branch B Family 2 remains diagnostic only and must not become the default route.
Phase C benchmark/protocol preparation doc and manifest schema completed as
safe prep only; manifest validator completed; no model run, no held-out, no
training, no branch execution.
Phase C manifest builder skeleton and records example completed as safe prep
only; no data download, no benchmark synthesis, no model run.
Phase C data-source field dry-inspection tool completed as safe prep only;
self-check and schema-example inspection pass; no real dataset download, no
benchmark synthesis, no model run.
Phase C tiny first-rows dry inspections completed for HotpotQA, MuSiQue mirror,
and 2Wiki mirror. All three expose evidence-split-compatible fields in preview;
this is field/schema evidence only, not benchmark admission and not a model run.
Phase C evidence-split QA row fetcher and record builder completed as safe prep
only. Seeded 300-row source pulls from HotpotQA and MuSiQue were converted into
200-record calibration-only manifest drafts. Both manifest audits pass with
zero errors. HotpotQA has 271 warnings and MuSiQue has 207 warnings, mainly
because answer strings naturally appear in support evidence; these warnings are
shortcut-risk signals requiring single/no-message/shuffled/wrong-evidence
controls before any benchmark admission.
Phase C control online-input package builder completed as safe prep only.
HotpotQA and MuSiQue calibration drafts were expanded into 7-condition control
packages, 1400 rows each. Leakage audits pass with zero errors. Warnings remain
high because evidence text often contains the answer string; this reinforces
the need for controls rather than weakening the no-leakage result.
Phase C capable text-agent runner completed as safe prep only. Runner selfcheck
passes on toy rows and validates sequential agent-message routing plus offline
scoring. No real capable-agent calibration run has been executed because
OPENAI_API_KEY and OPENAI_MODEL are unset in the current environment.
Phase C capable text-agent preflight completed as safe prep only. HotpotQA and
MuSiQue calibration control packages each estimate 2600 chat calls
(1400 solver calls + 1200 evidence-agent calls, 600 unique agent-cache keys).
Both preflights report ready_to_run_model=false because OPENAI_API_KEY and
OPENAI_MODEL are unset. No model generation has been run.
Phase C local-transformers provider added for real local model engineering
runs when no OpenAI-compatible endpoint is configured. Local
`/data1/luyifei/drla/models/Qwen3-4B-Instruct-2507-git` loads on GPU0 and
passes minimal generation. HotpotQA local Qwen3-4B preflight reports
ready_to_run_model=true. A 14-row HotpotQA smoke over the first 2 calibration
samples and all 7 conditions completed with real local model outputs:
single_full_info 2/2, single_q_only 0/2, textmas_matched 1/2, no_message 0/2,
shuffled 0/2, wrong_evidence 0/2, compressed_state 2/2. Aggregate gate remains
admitted=false because only 2 pairs gives paired CI lower bound 0. This is
engineering evidence only, not Phase C benchmark admission. The first smoke
also exposed and fixed a runner artifact bug: control source ids were present
inside online_input_fields but not promoted to generations.jsonl
control_source_sample_id. After the fix, run-level leakage audit passes with
0 errors and 7 evidence-string warnings.
Qwen3-8B-FP8 was downloaded to `/tmp/drla_models/Qwen3-8B-FP8` and linked at
`/data1/luyifei/drla/models/Qwen3-8B-FP8` after `/data1` space was too low and
`/data2` was not writable by the current user. Direct Xet download stalled at
large shards; disabling Xet resumed and completed the download. The runner now
passes `enable_thinking=false` to Qwen-style chat templates by default, matching
short-answer evaluation. HotpotQA Qwen3-8B-FP8 70-row pilot over the first 10
calibration samples completed: single_full_info 8/10, single_q_only 1/10,
textmas_matched 6/10, no_message 1/10, shuffled 0/10, wrong_evidence 3/10,
compressed_state 5/10. Aggregate remains admitted=false: matched-vs-shuffled
has positive CI lower bound (+0.3), but matched-vs-no_message and
matched-vs-wrong_evidence have CI lower bound 0. Leakage audit passes with
0 errors and 32 evidence-string warnings. This is a real local-model pilot, not
benchmark admission.
MuSiQue Qwen3-8B-FP8 initial 70-row pilot exposed a protocol issue: without
strict answer-format prompting, single_full_info was 0/10 while several outputs
contained correct answers inside longer sentences. The solver prompt was
repaired on calibration only to require exactly `Final answer: <short answer>`;
the parser now also accepts "final answer is ..." and "answer is ...". This is
format repair, not a scoring relaxation.
The wrong-evidence control was also strengthened from v0 "replace one shard" to
v1 strict "all evidence agents receive non-self control-sample private shards".
New v1 control packages were built for MuSiQue and HotpotQA under
`*_v1_strict_wrong`, both with 1400 rows and passing construction audit.
MuSiQue Qwen3-8B-FP8 v1 strict 70-row pilot over the first 10 calibration
samples completed and passes the pilot gate: single_full_info 6/10,
single_q_only 1/10, textmas_matched 7/10, no_message 1/10, shuffled 1/10,
wrong_evidence 1/10, compressed_state 6/10; paired CI lower bounds are +0.2
for full_info-vs-question and +0.3 for matched-vs-no/shuffled/wrong. Leakage
audit passes with 0 errors and 31 evidence-string warnings.
HotpotQA Qwen3-8B-FP8 v1 strict 70-row pilot still fails the pilot gate:
single_full_info 8/10, single_q_only 4/10, textmas_matched 8/10, no_message
4/10, shuffled 1/10, wrong_evidence 2/10, compressed_state 8/10; paired CI
lower is 0 for full_info-vs-question and matched-vs-no_message. Leakage audit
passes with 0 errors and 28 evidence-string warnings. This pilot made MuSiQue
the next full-calibration candidate, while HotpotQA stays diagnostic because
shortcut/no-message baselines are too strong.
MuSiQue Qwen3-8B-FP8 v1 strict full calibration over 200 calibration samples
completed and passes the Phase C calibration admission gate. Four 350-row
shards were merged into
`outputs/p2_phase_c_text_agent_runs/musique_local_qwen3_8b_fp8_v1_strict_wrong_full200_merged_20260601/generations.jsonl`
with 1400 unique rows. Aggregate artifact:
`outputs/p2_phase_c_text_agent_aggregates/musique_local_qwen3_8b_fp8_v1_strict_wrong_full200_20260601`.
Leakage audit:
`outputs/p2_phase_c_leakage_audits/musique_local_qwen3_8b_fp8_v1_strict_wrong_full200_20260601`.
Primary scores: single_full_info 0.425, single_q_only 0.080,
textmas_matched 0.450, no_message 0.080, shuffled 0.060,
wrong_evidence 0.070, compressed_state 0.420. Paired bootstrap CI lower bounds
are +0.265 for full_info-vs-question, +0.295 for matched-vs-no_message,
+0.320 for matched-vs-shuffled, and +0.315 for matched-vs-wrong_evidence.
Parseable rate is 1.0 for all conditions. Run-level leakage audit passes with
0 errors and 633 evidence-string warnings; these warnings mean answer aliases
appear in online evidence text and are treated as shortcut-risk warnings, not
hidden-label leakage. This is calibration admission only; the next locked step
is MuSiQue held-out evaluation with identical schema, prompt contract, scorer,
conditions and gate, with no held-out prompt repair.
MuSiQue Qwen3-8B-FP8 v1 strict locked held-out evaluation completed on
2026-06-05 and passes the Phase C admission gate. Held-out source rows were
fetched from `bdsaglam/musique` answerable/validation:
`outputs/p2_phase_c_data_source_audits/hf_rows_musique_answerable_validation_1000_seed20260605`.
The held-out manifest contains 800 samples:
`outputs/p2_phase_c_manifests/musique_heldout_manifest_800_seed20260605`.
Strict control inputs contain 5600 rows:
`outputs/p2_phase_c_control_inputs/musique_heldout_controls_800_seed20260605_v1_strict_wrong`.
Eight 700-row shards were merged into
`outputs/p2_phase_c_text_agent_runs/musique_local_qwen3_8b_fp8_v1_strict_wrong_heldout800_merged_20260605/generations.jsonl`
with 5600 unique rows. Aggregate artifact:
`outputs/p2_phase_c_text_agent_aggregates/musique_local_qwen3_8b_fp8_v1_strict_wrong_heldout800_20260605`.
Leakage audit:
`outputs/p2_phase_c_leakage_audits/musique_local_qwen3_8b_fp8_v1_strict_wrong_heldout800_20260605`.
Held-out primary scores: single_full_info 0.48625, single_q_only 0.03875,
textmas_matched 0.37375, no_message 0.03875, shuffled 0.04750,
wrong_evidence 0.04875, compressed_state 0.38250. Paired bootstrap CI lower
bounds are +0.41250 for full_info-vs-question, +0.30125 for
matched-vs-no_message, +0.29375 for matched-vs-shuffled, and +0.29250 for
matched-vs-wrong_evidence. Parseable rate is 1.0 for all conditions.
Run-level leakage audit passes with 0 errors and 2640 evidence-string
warnings. Phase C benchmark/protocol validation is therefore complete for
MuSiQue evidence-split QA v1 strict. This is a capable text-agent protocol
result, not a CoLA result; the next stage is Phase A CoLA substrate/interface
adaptation on this locked protocol, with CoLA Single Solver and CoLA Role
TextMAS gates required before CoLA TextMAS vs LatentMAS main comparison.
Phase A CoLA runner plumbing started on 2026-06-05. The Phase C runner now
supports `--provider cola_dlm`, which uses the official CoLA VAE/DiT under the
same MuSiQue online inputs, parser, QA scorer, resume behavior, and artifact
schema as the Qwen capable-text-agent runs. Preflight passes for the 14-row
held-out smoke:
`outputs/p2_phase_c_text_agent_preflights/musique_cola_dlm_v1_strict_wrong_heldout_smoke14_preflight_20260605`.
The real CoLA smoke run also completes:
`outputs/p2_phase_c_text_agent_runs/musique_cola_dlm_v1_strict_wrong_heldout_smoke14_20260605`.
All seven conditions score 0/2 in this smoke. Raw outputs show JSON/chat-style
prompt drift and malformed answer strings rather than meaningful evidence
integration. This is a Phase A interface diagnostic, not a CoLA main result
and not a latent-communication failure. The next Phase A action should use
calibration only to build a CoLA task-format / role-interface adapter or
prompt/template adapter, then rerun CoLA Single and Role TextMAS gates before
any LatentMAS comparison.
Phase A CoLA prompt-only interface diagnostics continued on 2026-06-05, using
calibration-only MuSiQue smoke rows. `plain_qa_v1` completed at
`outputs/p2_phase_c_text_agent_runs/musique_cola_dlm_plainqa_v1_strict_wrong_calibration_smoke14_20260605`
and scored 0/2 primary on all seven conditions. `squad_template_v1`, which
routes final solver calls through CoLA's official SQuAD prompt template shape,
completed at
`outputs/p2_phase_c_text_agent_runs/musique_cola_dlm_squadtemplate_v1_strict_wrong_calibration_smoke14_20260605`
and also scored 0/2 primary on all seven conditions, with only small token-F1
overlap in matched/compressed states. Raw outputs show evidence copying,
question continuation, and answer-format drift; they do not show stable final
answer extraction. To verify that the official CoLA substrate itself is not
broken, an independent official SQuAD 20-sample sanity run was executed at
`outputs/p2_phase_a_official_cola_sanity/official_squad_smoke20_20260605`;
the same local CoLA weights reached 5/20 primary matches (0.25), consistent in
scale with the existing official SQuAD full baseline (30.90%). Therefore the
current failure is a MuSiQue role/interface/task-distribution mismatch, not a
bad model load or CUDA path failure. Phase A should stop prompt-only repair and
move to calibration-only supervised CoLA task-format / role-interface
adaptation; any training must run on GPU with SwanLab cloud, local
`metrics.jsonl`, `best_checkpoint.pt`, `last_checkpoint.pt`, and
`valid_interval <= 10`. Held-out MuSiQue remains locked and must not be used
for adapter design or prompt repair.
Phase A calibration-only supervised interface adaptation started on 2026-06-05.
`build_p2_phase_a_cola_interface_sft.py` produced 800 SFT pairs at
`outputs/p2_phase_a_cola_interface_sft/musique_calibration_qwen_teacher_v1_20260605`:
200 `solver_full_info`, 200 `solver_textmas_matched`, and 400
`evidence_agent_teacher` pairs distilled from the admitted Qwen3-8B-FP8
calibration TextMAS run. The split is sample-level train/valid = 640/160 pairs,
with no drops and no held-out data. `train_p2_phase_a_cola_dit_lora.py`
implements a LoRA-only official CoLA DiT Flow-Matching adapter with frozen VAE,
GPU-only training, SwanLab cloud, local `metrics.jsonl`, `best_checkpoint.pt`,
`last_checkpoint.pt`, and adapter directories. The first all-role 100-step pilot
proved the gradient/checkpoint/SwanLab path but did not improve exact QA. A
solver-only 3-epoch run without explicit EOS improved token overlap but not
primary accuracy. The solver-only 3-epoch run with target EOS is the current
best Phase A adapter:
`outputs/p2_phase_a_cola_dit_lora/musique_solver_interface_lora_v1_epoch3_eos_20260605`
with SwanLab run `lmw46365bo8dneyheqw49`, best step 1100, best valid
Flow-Matching loss 0.14344, and 6.34M trainable LoRA params over a 1.836B-param
CoLA DiT. The Phase C runner now supports deterministic `--cola-noise-seed`
and `--cola-dit-lora-path`, plus `--prediction-extraction-mode first_segment`
that strips generated `<|endoftext|>` / `<|im_end|>` boundaries for CoLA-style
outputs.

Deterministic calibration solver100 comparison, seed 66, `squad_template_v1`,
`first_segment`, no held-out:

```text
Frozen official CoLA:
  single_full_info primary: 0.00 / 50
  single_q_only primary:   0.00 / 50

CoLA DiT LoRA solver-interface epoch3 + EOS:
  single_full_info primary: 0.36 / 50
  single_q_only primary:   0.10 / 50
  paired mean diff: +0.26
  paired bootstrap 95% CI lower: +0.12
```

Interpretation: Phase A Single Solver interface adaptation is now working on
calibration and shows real evidence-use above question-only. This is not yet a
Role TextMAS gate and not a LatentMAS result. Next Phase A work should train or
otherwise adapt the evidence-agent / role-message interface with EOS-aware
targets, then rerun the full seven-condition calibration gate before touching
held-out or Phase E.
Phase A role-interface adaptation continued on 2026-06-05. A role-specific
evidence-agent adapter was trained from the solver-interface adapter:
`outputs/p2_phase_a_cola_dit_lora/musique_evidence_agent_lora_v1_from_solver_epoch2_20260605`
with SwanLab run `yra5lf4h00711kbb87m3z`, best step 3200, best valid
Flow-Matching loss 0.25338, and final valid Flow-Matching loss 0.36942. The
first all-role shared-adapter attempt
`outputs/p2_phase_a_cola_dit_lora/musique_role_interface_lora_v1_from_solver_epoch1_20260605`
is a negative diagnostic because it damaged solver quality in 14-row smoke
evaluation; the current admitted calibration setup therefore uses role-specific
adapters: evidence-agent adapter for upstream evidence messages, solver
adapter for final answers.

Full 200-sample MuSiQue calibration Role TextMAS gate with these dual adapters
completed and passes the admission gate:

```text
run:
  outputs/p2_phase_c_text_agent_runs/
  musique_cola_dlm_dual_lora_evidence_epoch2_solver_epoch3eos_squadtemplate_v1_firstseg_seed66_calibration_full200_merged_20260605

aggregate:
  outputs/p2_phase_c_text_agent_aggregates/
  musique_cola_dlm_dual_lora_evidence_epoch2_solver_epoch3eos_squadtemplate_v1_firstseg_seed66_calibration_full200_20260605

leakage audit:
  outputs/p2_phase_c_leakage_audits/
  musique_cola_dlm_dual_lora_evidence_epoch2_solver_epoch3eos_squadtemplate_v1_firstseg_seed66_calibration_full200_20260605

condition primary scores:
  single_full_info: 0.490
  single_q_only: 0.115
  textmas_matched: 0.465
  textmas_no_message: 0.095
  textmas_shuffled_message: 0.095
  textmas_wrong_evidence_or_wrong_shard: 0.125
  textmas_compressed_state: 0.470

paired bootstrap CI lower:
  single_full_info - single_q_only: +0.300
  textmas_matched - no_message: +0.290
  textmas_matched - shuffled_message: +0.295
  textmas_matched - wrong_evidence: +0.265

gate:
  admitted=true on calibration
  parseable_rate=1.0 for all conditions
  leakage audit status=pass
  leakage errors=0
  leakage warnings=633
```

Interpretation: CoLA Single Solver and CoLA Role TextMAS are now both positive
on calibration under the locked MuSiQue v1 strict protocol. This is still not
the final CoLA claim and not a LatentMAS result. The next locked step is the
identical held-out run with the same prompt style, parser, seed, evidence-agent
adapter, solver adapter, scorer and controls; no further prompt repair or
adapter selection may use held-out.
The identical locked held-out run completed on 2026-06-05 and does not pass the
Phase A admission gate:

```text
run:
  outputs/p2_phase_c_text_agent_runs/
  musique_cola_dlm_dual_lora_evidence_epoch2_solver_epoch3eos_squadtemplate_v1_firstseg_seed66_heldout800_merged_20260605

aggregate:
  outputs/p2_phase_c_text_agent_aggregates/
  musique_cola_dlm_dual_lora_evidence_epoch2_solver_epoch3eos_squadtemplate_v1_firstseg_seed66_heldout800_20260605

leakage audit:
  outputs/p2_phase_c_leakage_audits/
  musique_cola_dlm_dual_lora_evidence_epoch2_solver_epoch3eos_squadtemplate_v1_firstseg_seed66_heldout800_20260605

condition primary scores:
  single_full_info: 0.0875
  single_q_only: 0.0025
  textmas_matched: 0.15875
  textmas_no_message: 0.00125
  textmas_shuffled_message: 0.01000
  textmas_wrong_evidence_or_wrong_shard: 0.01000
  textmas_compressed_state: 0.16000

paired bootstrap CI lower:
  single_full_info - single_q_only: +0.06500
  textmas_matched - no_message: +0.13250
  textmas_matched - shuffled_message: +0.12250
  textmas_matched - wrong_evidence: +0.12375

gate:
  admitted=false on held-out
  failed gate: single_full_info_above_floor
  threshold: 0.20000
  observed single_full_info: 0.08750
  parseable_rate: 0.99625 for single_full_info, 1.0 for matched
  leakage audit status=pass
  leakage errors=0
  leakage warnings=2640
```

Interpretation: the held-out result preserves a strong role-message effect
relative to no-message / shuffled / wrong-evidence controls, but the adapted
CoLA solver's absolute held-out capability is below the locked floor. The
current dual-adapter Phase A result is therefore a positive calibration result
and a useful failure diagnostic, not an admitted CoLA Role TextMAS benchmark
result. Phase E CoLA TextMAS vs LatentMAS must not start from this checkpoint.
This held-out aggregate may be cited as a failed locked eval, but held-out rows
must not be used for prompt repair, adapter selection, threshold choice, or any
training data construction.

Phase A train-only generalization repair attempt completed on 2026-06-05. To
avoid relying on the small calibration set for adapter training, 2000 MuSiQue
answerable/train rows were fetched, converted into the locked evidence-split
records, expanded into v1 strict controls, and used only to collect Qwen3-8B-FP8
teacher evidence-agent messages. The merged teacher-message artifact is:
`outputs/p2_phase_a_teacher_agent_messages/musique_qwen3_8b_fp8_interface_train_textmas_matched_2000_seed20260606_t96_merged_20260605`.
The derived CoLA interface SFT artifact is:
`outputs/p2_phase_a_cola_interface_sft/musique_interface_train_qwen_teacher_2000_t96_seed20260606_20260605`
with 8000 pairs: 2000 `solver_full_info`, 2000 `solver_textmas_matched`, and
4000 `evidence_agent_teacher`. No held-out rows were used.

Two role-specific CoLA DiT LoRA adapters were trained with GPU + SwanLab cloud:

```text
solver adapter:
  outputs/p2_phase_a_cola_dit_lora/
  musique_solver_interface_lora_train2000_t96_epoch2_seed20260606_20260605
  SwanLab: hv2i6pj6edx4x0i2zqjt3
  best_step: 3500
  best_valid_loss: 0.13262649088341277

evidence-agent adapter:
  outputs/p2_phase_a_cola_dit_lora/
  musique_evidence_agent_lora_train2000_t96_from_solver_epoch1_seed20260606_20260605
  SwanLab: kfu5ph91sj21pplmgtzzu
  best_step: 20300
  best_valid_loss: 0.21131337011232973
```

Full 200-sample calibration rerun with these train-only adapters completed and
does **not** pass the Phase A admission gate:

```text
merged run:
  outputs/p2_phase_c_text_agent_runs/
  musique_cola_dlm_dual_lora_train2000_t96_seed20260606_squadtemplate_v1_firstseg_seed66_calibration_full200_merged_20260605

aggregate:
  outputs/p2_phase_c_text_agent_aggregates/
  musique_cola_dlm_dual_lora_train2000_t96_seed20260606_squadtemplate_v1_firstseg_seed66_calibration_full200_20260605

condition primary scores:
  single_full_info: 0.130
  single_q_only: 0.000
  textmas_matched: 0.110
  textmas_no_message: 0.000
  textmas_shuffled_message: 0.010
  textmas_wrong_evidence_or_wrong_shard: 0.005
  textmas_compressed_state: 0.130

paired bootstrap CI lower:
  single_full_info - single_q_only: +0.085
  textmas_matched - no_message: +0.070
  textmas_matched - shuffled_message: +0.060
  textmas_matched - wrong_evidence: +0.065

gate:
  admitted=false on calibration
  failed gate: single_full_info_above_floor
  threshold: 0.200
  observed single_full_info: 0.130
  parseable_rate=1.0 for single_full_info and matched
```

Interpretation: the train-only 2000 adapter preserves a real communication
effect over no-message/shuffled/wrong controls, but it does not recover enough
absolute CoLA solver capability even on calibration. Therefore held-out must
not be run for this checkpoint, and Phase E still must not start. The next
Phase A work should repair CoLA full-info solver generalization under
non-heldout data only, rather than tuning on held-out or lowering the gate.

Additional train-source diagnostic for the same solver adapter completed on
2026-06-05. It evaluates only the first 100 train-source samples under
`single_full_info` and `single_q_only`, using the locked `squad_template_v1`,
`first_segment`, seed 66, and the train-only solver best adapter:

```text
merged run:
  outputs/p2_phase_c_text_agent_runs/
  musique_cola_dlm_solver_train2000_t96_train_source_single_diag100_merged_20260605

rows:
  train split only
  100 single_full_info
  100 single_q_only

readout:
  single_full_info primary: 0.240
  single_full_info exact: 0.220
  single_full_info token_f1: 0.4079
  single_q_only primary: 0.000
  paired diff: +0.240
  bootstrap CI lower: +0.160
```

Interpretation: the train-only solver adapter is not a dead training run; on
its source distribution it learns usable evidence-conditioned answering above
question-only. The failure mode is weak transfer from train-source MuSiQue to
the locked calibration set, so the next repair should increase non-heldout
coverage/capacity/checkpoint selection for full-info solver capability before
retrying Role TextMAS or any held-out run.

Follow-up full-info coverage repair attempt was started on 2026-06-05:

```text
source rows:
  outputs/p2_phase_c_data_source_audits/
  hf_dataset_rows_musique_answerable_train_10000_seed20260606
  backend: datasets.load_dataset
  rows: 10000 / 19938 train rows

records:
  outputs/p2_phase_c_records/
  musique_interface_train_records_10000_seed20260606
  output records: 10000

manifest:
  outputs/p2_phase_c_manifests/
  musique_interface_train_manifest_10000_seed20260606

SFT:
  outputs/p2_phase_a_cola_interface_sft/
  musique_interface_train_fullinfo_10000_seed20260606_20260605
  role_counts: solver_full_info 10000, solver_textmas_matched 861,
               evidence_agent_teacher 1722
```

The intended solver_full_info-only continuation run used GPU + SwanLab cloud:

```text
run:
  outputs/p2_phase_a_cola_dit_lora/
  musique_solver_fullinfo_lora_train10000_from_train2000_epoch1_lr5e5_seed20260606_20260605
SwanLab:
  mludvhrmz5yadn25niij0
config:
  init_lora_path = train2000 solver best_adapter
  roles = solver_full_info
  lr = 5e-5
  valid_interval = 100
  max_valid_batches = 300
```

This run failed at runtime before completion:

```text
status:
  failed_runtime_nvml_assert
failure artifact:
  outputs/p2_phase_a_cola_dit_lora/
  musique_solver_fullinfo_lora_train10000_from_train2000_epoch1_lr5e5_seed20260606_20260605/
  summary_runtime_failure.json
last_train_step:
  527
best_valid_metric:
  step 500, valid Flow-Matching loss 0.18355972192327802
failure:
  PyTorch CUDA caching allocator NVML internal assert
```

The partial step500 best adapter was evaluated on calibration solver100:

```text
merged eval:
  outputs/p2_phase_c_text_agent_runs/
  musique_cola_dlm_solver_fullinfo_train10000_partial_step500_calibration_solver100_merged_20260605

readout:
  single_full_info primary: 0.030
  single_q_only primary: 0.000
  paired diff: +0.030
  CI lower: 0.000
```

Interpretation: this partial checkpoint is not usable. More importantly,
Flow-Matching valid loss alone is not a reliable selector for downstream QA
capability on this adapter. Before another large full-info run, the trainer
should preserve more interval checkpoints and/or add a non-heldout solver
capability selection diagnostic; do not use the crashed step500 adapter for
Role TextMAS, held-out, or Phase E.

Trainer support for this was added immediately afterward:
`drla/scripts/train_p2_phase_a_cola_dit_lora.py` now accepts
`--save-interval-checkpoints`. When enabled, every validation interval writes
`checkpoints/valid_step_<step>.pt` and
`checkpoints/valid_step_<step>_adapter`, while still preserving the normal
`best_checkpoint.pt`, `last_checkpoint.pt`, `best_adapter`, and `last_adapter`.
This should be enabled for the next full-info repair run so checkpoint
selection can be audited with non-heldout solver diagnostics instead of relying
only on Flow-Matching valid loss.
Phase C text-agent aggregation/admission gate completed as safe prep only.
Aggregator selfcheck admits a complete toy control set and rejects the runner
selfcheck artifact because shuffled/wrong controls are absent. This locks gate
logic before real capable-agent outputs exist.
Phase C data-source field/license audit completed as safe prep only. It now
records both tiny preview field checks and seeded calibration-draft artifacts;
no held-out/test data has been used and no model run has started.
Phase C data-source / runner design completed as safe prep only.
Phase C offline scorer helpers and scorer self-check completed as safe prep
only; no benchmark data, no model run, no held-out, no training.
Phase C run-level leakage auditor completed as safe prep only.
P2 locked complete execution scheme updated with the current anti-disturbance
plan: benchmark/agent baselines must shift to true MAS + capable TextMAS first;
CoLA is a suitable shared latent substrate but frozen official weights are not
assumed to satisfy new benchmarks; TextMAS/LatentMAS must both follow
Agent-A-output-as-Agent-B-input with receiver-only scoring; no fuser/adapter or
main text-vs-latent table may start before Phase C and Phase A gates pass.
```

```text
evidence order:
  1. packet validity and distribution compatibility
  2. Agent B readability under matched-vs-corrupted controls
  3. downstream task utility
  4. cost-quality comparison against text-channel baseline
  5. role-conditioned sequential/hierarchical envelopes
  6. naturally decomposable MAS benchmarks
```

`docs/parked/DRLA_Multiscale_Block_Halt_Design.md` 是 parked try，不纳入当前实验路线。

## 2026-06-01 P2 Route Reset

```text
official8 solver-to-solver message_only:
  keep as P2-D0 channel diagnostic only.

main P2 benchmark protocol:
  Planner -> Critic -> Refiner -> Solver,
  downstream roles see q + previous text/latent message,
  scorer sees final Solver output only.

benchmark admission:
  every new benchmark must pass Single CoLA Solver and Role TextMAS capability
  gates before entering main paper-level P2 tables.

latent handoff:
  context_plus_thought / full_working_memory is canonical;
  thought_only is an ablation.

fuser:
  train only after no-fuser context-visible/full-working-memory experiments
  show robust need, e.g. wrong_block remains strong or latent underperforms
  TextMAS under matched budgets.
```

## 2026-06-01 P2-D1 Capability Gate Setup

```text
data preparation script:
  /data1/luyifei/drla/drla/scripts/prepare_cola_p2_candidate_benchmarks.py

gate evaluation script:
  /data1/luyifei/drla/drla/scripts/run_cola_p2_capability_gate.py

full candidate data:
  /data1/luyifei/drla/outputs/p2_capability_gate/data_20260601

combined jsonl:
  /data1/luyifei/drla/outputs/p2_capability_gate/data_20260601/p2_candidate_benchmarks.jsonl

rows:
  ARC-Easy validation = 570
  ARC-Challenge validation = 299
  GSM8K test = 1319
  MBPP+ test = 378
  HumanEval+ test = 164
  GPQA-Diamond test = 198
  MedQA test = 1273
  total = 4201

unprepared:
  none
```

Interpretation:

```text
These data are a capability-gate substrate, not yet a paper-level result.
Code tasks need an execution gate before entering main P2 tables.
Pure gate/eval runs stay local-only with swanlab_mode=disabled.
```

Formal full gate:

```text
aggregate:
  /data1/luyifei/drla/outputs/p2_capability_gate/aggregate_full_20260601

admitted_tasks:
  []

summary:
  ARC-Easy: single 20.18%, role 14.21%, not admitted
  ARC-Challenge: single 23.08%, role 10.70%, not admitted
  GPQA-Diamond: single 21.21%, role 12.12%, not admitted
  MedQA: single 23.02%, role 17.83%, not admitted
  GSM8K: single 2.05% passes single-only gate, role 1.74% fails, not admitted
  HumanEval+: single/role execution pass 0%, not admitted
  MBPP+: single/role execution pass 0%, not admitted
```

Conclusion:

```text
Do not run P2 text-vs-latent main tables on these candidate tasks yet.
The current bottleneck is CoLA base capability / prompt protocol, not a proven
latent-communication failure.
Historical next step at that time was locked calibration/held-out partitioning
plus prompt/protocol repair. That path has since been executed through
P2-D3/D3.1/D3.2 and Branch B Family 1, and remains admitted_tasks=[].
Current next-plan is P2_Post_Family1_Complete_Execution_Plan_2026-06-01.md.
```

## 2026-06-01 P2-D2 Locked Split

```text
script:
  /data1/luyifei/drla/drla/scripts/build_cola_p2_locked_splits.py

output:
  /data1/luyifei/drla/outputs/p2_capability_gate/locked_splits_seed20260602_20260601

input_sha256:
  fffe88201e6643a57541c2f986496a7b836f1c1112629500a49afc3b9f623fd0

split_seed:
  20260602

calibration / heldout / overlap:
  842 / 3359 / 0
```

Per-task split:

```text
arc_easy: 114 calibration / 456 heldout
arc_challenge: 60 / 239
gpqa_diamond: 40 / 158
medqa: 255 / 1018
gsm8k: 264 / 1055
humanevalplus: 33 / 131
mbppplus: 76 / 302
```

Usage boundary:

```text
Historical P2-D3 prompt/protocol repair used calibration rows only.
Held-out rows were not to be inspected at sample level during repair.
This split is not a current trigger for held-out gate because the repair path
did not produce admitted_tasks.
```

## 2026-06-01 P2-D3 Initial Prompt Repair

```text
prompt variants:
  generic_v1
  cola_fewshot_v1

calibration-only aggregate:
  /data1/luyifei/drla/outputs/p2_capability_gate/
  aggregate_calibration_noncode_single_prompt_variants_20260601

gpqa single+role aggregate:
  /data1/luyifei/drla/outputs/p2_capability_gate/
  aggregate_calibration_gpqa_generic_single_role_20260601
```

Result:

```text
cola_fewshot_v1 does not improve the candidate set.
generic_v1 GPQA-Diamond single-mode calibration passes:
  accuracy = 32.50%
  parseable = 92.50%

GPQA-Diamond Role TextMAS still fails:
  accuracy = 25.00%
  parseable = 50.00%

admitted_tasks after P2-D3 initial repair:
  []
```

Conclusion:

```text
Do not use held-out yet.
Do not enter P2 main text-vs-latent tables.
Historical next repair target was Role TextMAS protocol quality on calibration.
That repair path has since completed and did not admit a task.

Superseded locked next-phase plan:
  /data1/luyifei/drla/docs/current/P2_Next_Phase_Execution_Plan_2026-06-01.md

Current canonical next-plan:
  /data1/luyifei/drla/docs/current/
  P2_Post_Family1_Complete_Execution_Plan_2026-06-01.md

Historical immediate next step:
  answer-state repair has now been evaluated on calibration.
  Do not inspect held-out, run latent-vs-text main tables, or train fuser yet.
```

## 2026-06-01 P2-D3.1 Answer-State Repair

```text
implemented prompt variants:
  answer_state_v1
  answer_state_structured_v1

aggregate:
  /data1/luyifei/drla/outputs/p2_capability_gate/
  aggregate_calibration_protocol_repair_answer_state_20260601

failure taxonomy:
  /data1/luyifei/drla/outputs/p2_capability_gate/
  audit_protocol_repair_failures_all_20260601

all repair aggregate:
  /data1/luyifei/drla/outputs/p2_capability_gate/
  aggregate_calibration_protocol_repair_all_20260601
```

Result:

```text
admitted_tasks:
  []

GPQA:
  generic_v1 role = 25.00% acc / 50.00% parse
  answer_state_v1 role = 5.00% acc / 40.00% parse
  answer_state_structured_v1 role = 15.00% acc / 67.50% parse
  role_plan_ignore_v1 role = 2.50% acc / 35.00% parse

ARC-Challenge:
  answer_state_structured_v1 role = 26.67% acc / 85.00% parse

ARC-Easy:
  answer_state_structured_v1 role = 17.54% acc / 85.09% parse

MedQA:
  answer_state_structured_v1 role = 24.31% acc / 92.94% parse

paired taxonomy:
  GPQA generic role-minus-single = -7.50 pp
  GPQA structured role-minus-single = -5.00 pp
  GPQA role_plan_ignore role-minus-single = -22.50 pp
  ARC-Challenge structured role-minus-single = +6.67 pp, but parse = 85.00%
  MedQA structured role-minus-single = +4.31 pp, but near random floor
```

Interpretation:

```text
Structured answer-state reduces raw role-message pollution but does not solve
the capability gate. The current evidence points to base CoLA capability plus
role protocol mismatch, not to a latent communication result.

Failure taxonomy and local LatentMAS/Coconut review are complete enough to say
prompt-only Role TextMAS repair is weak under the current CoLA substrate. Next
step is a branch decision: substrate adaptation or benchmark redesign. Held-out
remains untouched.
```

## 2026-06-01 P2-D4 Branch Decision Audit

```text
doc:
  /data1/luyifei/drla/docs/current/
  P2_D4_Branch_Decision_Audit_2026-06-01.md

status:
  completed as historical decision audit; Branch B first was executed and
  later stopped as Family 1.

historical recommendation:
  benchmark redesign / capability-matched task selection.

post-Family1 update:
  current canonical next-plan is
  P2_Post_Family1_Complete_Execution_Plan_2026-06-01.md.
  Recommended scientific order is Branch C -> Branch A.

not started:
  held-out gate
  P2 main text-vs-latent table
  LoRA/fuser/adapter training
```

Safe prep completed:

```text
doc:
  /data1/luyifei/drla/docs/current/
  P2_Benchmark_Redesign_Candidate_Inventory_2026-06-01.md

execution lock:
  /data1/luyifei/drla/docs/current/
  P2_Branch_B_Execution_Plan_2026-06-01.md

recommended first Branch B pass:
  official8-compatible MCQ role gate on obqa, mmlu, race, hellaswag, siqa,
  story_cloze, with new manifest and fresh calibration/held-out gates.

boundary:
  This does not admit a benchmark or execute a main text-vs-latent table.
```

Branch B official8-compatible data prepared:

```text
script:
  /data1/luyifei/drla/drla/scripts/prepare_cola_p2_official8_role_candidates.py

smoke output:
  /data1/luyifei/drla/outputs/p2_benchmark_redesign/
  official8_role_candidates_smoke_20260601

full output:
  /data1/luyifei/drla/outputs/p2_benchmark_redesign/
  official8_role_candidates_20260601

combined jsonl:
  /data1/luyifei/drla/outputs/p2_benchmark_redesign/
  official8_role_candidates_20260601/p2_official8_role_candidates.jsonl

rows:
  official8_obqa = 500
  official8_mmlu = 14042
  official8_race = 4887
  official8_hellaswag = 10042
  official8_siqa = 1954
  official8_story_cloze = 1871
  total = 33296

schema check:
  multiple_choice rows = 33296
  bad ground_truth / choices rows = 0

result meaning:
  data conversion only. No model generation, no training, no held-out result.
```

Branch B calibration first pass:

```text
report:
  /data1/luyifei/drla/docs/current/
  P2_Branch_B_Calibration_Report_2026-06-01.md

split:
  /data1/luyifei/drla/outputs/p2_benchmark_redesign/
  official8_role_candidates_splits_seed20260603_20260601

calibration / heldout / overlap:
  1600 / 31696 / 0

aggregate:
  /data1/luyifei/drla/outputs/p2_benchmark_redesign/
  aggregate_calibration_official8_branchb_prompt_gate_20260601

formal prompt families checked:
  generic_v1
  answer_state_structured_v1
  cola_fewshot_v1

task-isolated candidate check:
  official8_mmlu with single_prompt_variant=generic_v1 and
  role_prompt_variant=answer_state_structured_v1

admitted_tasks:
  []
```

Key facts:

```text
generic_v1 all-task single run had one marginal pass:
  official8_mmlu = 27.67% acc / 91.33% parse

task-isolated MMLU split-prompt run did not reproduce that pass:
  single = 21.00% acc / 88.00% parse
  role_textmas = 20.00% acc / 86.00% parse

interpretation:
  no held-out gate yet.
  no P2 text-vs-latent main table.
  next step is official CoLA prompt/eval alignment audit because current
  normalized gate prompt/parser is not stable enough.
```

Official8 native prompt/eval alignment audit:

```text
report:
  /data1/luyifei/drla/docs/current/
  P2_Official8_Native_Alignment_Audit_2026-06-01.md

native single script:
  /data1/luyifei/drla/drla/scripts/run_cola_p2_official8_native_single_gate.py

rescore script:
  /data1/luyifei/drla/drla/scripts/rescore_cola_p2_official8_native_single_gate.py

native generation:
  /data1/luyifei/drla/outputs/p2_benchmark_redesign/
  eval_calibration_official8_native_single_full32_seed66_20260601

rescored result:
  /data1/luyifei/drla/outputs/p2_benchmark_redesign/
  eval_calibration_official8_native_single_full32_seed66_rescored_20260601

admitted_tasks:
  []
```

Native single calibration:

```text
official8_hellaswag = 1.33% acc / 100.00% parse
official8_mmlu = 18.33% acc / 75.00% parse
official8_obqa = 25.00% acc / 83.00% parse
official8_race = 23.00% acc / 83.33% parse
official8_siqa = 30.67% acc / 93.00% parse
official8_story_cloze = 29.33% acc / 100.00% parse
```

Interpretation:

```text
The normalized P2 prompt/parser was not the only cause. Native official
template/scoring still does not admit a task under random_floor+margin and
parse gates. Branch B Family 1 should stop; next decision is Branch A
substrate adaptation, Branch C external capable TextMAS protocol validation,
or a genuinely new Branch B Family 2 design with a robust native single gate.
```

Post-Family1 branch decision memo:

```text
doc:
  /data1/luyifei/drla/docs/current/
  P2_Post_Family1_Branch_Decision_Memo_2026-06-01.md

status:
  decision memo only, not execution

external sources checked:
  LatentMAS: https://arxiv.org/abs/2511.20639
  Interlat: https://arxiv.org/abs/2511.09149
  Thought Communication: https://arxiv.org/abs/2510.20733
  CoLA DLM: https://arxiv.org/abs/2605.06548
  CoLA model card: https://huggingface.co/ByteDance-Seed/Cola-DLM
  Silo-Bench: https://papers.cool/arxiv/2603.01045
  CRAFT: https://huggingface.co/papers/2603.25268
  CoSMAC: https://openreview.net/forum?id=yGzAhl1o4i

recommendation:
  Branch C first, then return to CoLA latent through adapter/translator.

boundary:
  This recommendation requires user confirmation. Do not auto-start Branch C,
  Branch A training, or Branch B Family 2 replacement.
```

Post-Family1 complete execution plan:

```text
doc:
  /data1/luyifei/drla/docs/current/
  P2_Post_Family1_Complete_Execution_Plan_2026-06-01.md

status:
  current canonical next-plan

route:
  Phase C external capable TextMAS benchmark/protocol validation
  -> Phase A CoLA substrate/interface adaptation
  -> Phase E CoLA TextMAS vs LatentMAS main comparison

guardrail:
  Branch B Family 2 is diagnostic-only; no held-out/main/fuser execution until
  the chosen branch meets its gate.
```

Phase C benchmark/protocol safe prep:

```text
doc:
  /data1/luyifei/drla/docs/current/
  P2_Phase_C_Benchmark_Protocol_Preparation_2026-06-01.md

data source / runner design:
  /data1/luyifei/drla/docs/current/
  P2_Phase_C_Data_Source_and_Runner_Design_2026-06-01.md

data source field/license audit:
  /data1/luyifei/drla/docs/current/
  P2_Phase_C_Data_Source_Field_License_Audit_2026-06-01.md

schema:
  /data1/luyifei/drla/configs/p2_phase_c_manifest_schema.json

schema example:
  /data1/luyifei/drla/configs/p2_phase_c_manifest_example.json

records example:
  /data1/luyifei/drla/configs/p2_phase_c_records_example.jsonl

manifest builder:
  /data1/luyifei/drla/drla/scripts/build_p2_phase_c_manifest.py

validator:
  /data1/luyifei/drla/drla/scripts/validate_p2_phase_c_manifest.py

scorer helpers:
  /data1/luyifei/drla/drla/evaluation/p2_phase_c_scorers.py

scorer self-check:
  /data1/luyifei/drla/drla/scripts/selfcheck_p2_phase_c_scorers.py

run-level leakage audit:
  /data1/luyifei/drla/drla/scripts/audit_p2_phase_c_run_leakage.py

candidate families:
  evidence-split multi-hop QA
  scalable distributed-state synthesis
  planner-coder-tester-reviewer code workflow

status:
  safe prep only; no execution until Branch C is explicitly selected.
```

Smoke checks completed:

```text
data smoke:
  /data1/luyifei/drla/outputs/p2_capability_gate/data_smoke_20260601
  2 samples per prepared task, schema OK

generation smoke:
  /data1/luyifei/drla/outputs/p2_capability_gate/eval_smoke_arc_easy_both_20260601
  single + role_textmas, 1 ARC-E sample, CUDA path OK, SwanLab disabled

result meaning:
  parser/output schema works.
  This is a smoke run and cannot admit or reject ARC-E scientifically.
```

## P1 Locked Result

```text
summary:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/
  official8_full_b64_bs12_p1_locked_seed66_67_68_split20260601_riskcert_20260527/summary.json

loss_case_audit:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/
  official8_full_b64_bs12_p1_locked_seed66_67_68_split20260601_riskcert_20260527/loss_case_audit.json

seeds:
  66, 67, 68

split seed:
  20260601

repeated target-test decisions:
  14940

selected / fixed-final / prediction-stability accuracy:
  20.930% / 20.950% / 20.957%

average selected blocks:
  1.834 / 4

losses vs final / prediction stability:
  3 / 4

mismatches vs final / prediction stability:
  85 / 91

calibration joint risk satisfied:
  21 / 24 folds
```

Interpretation：

```text
P1 supports entering P2.
Risk-control is observed-low-risk + partially certified.
It is not fully certified formal safety.
```

## P2 Packet v1/v2

```text
script:
  /data1/luyifei/drla/drla/scripts/build_cola_agent_latent_comm_packets.py

output:
  /data1/luyifei/drla/outputs/cola_agent_latent_comm/
  p2_agent_latent_comm_v1_locked_seed66_67_68_split20260601_20260527

packets:
  14940

latent block refs:
  27399

unique latent files checked:
  8850

missing latent files:
  0

forbidden decoder/eval fields:
  0
```

Packet v1 proves sanitized packet construction, not Agent B readability or superiority over text handoff.

P2-A packet v2：

```text
output:
  /data1/luyifei/drla/outputs/cola_agent_latent_comm/
  p2_agent_latent_comm_v2_locked_seed66_67_68_split20260601_20260529

protocol:
  cola_agent_latent_comm_v2

packets:
  14940

latent block refs:
  27399

unique latent files checked:
  8850

missing latent files:
  0

forbidden decoder/eval fields:
  0

v2 field coverage:
  communication_boundary = 14940 / 14940
  prefix_contract = 14940 / 14940
  agent_b_contract = 14940 / 14940
```

Packet v2 proves schema/contract coverage for same-substrate single-handoff packets. It still does not prove Agent B readability or superiority over text handoff.

## P2-B Distribution Audit

```text
script:
  /data1/luyifei/drla/drla/scripts/audit_cola_agent_latent_packet_distribution.py

output:
  /data1/luyifei/drla/outputs/cola_agent_latent_comm/
  p2_agent_latent_comm_v2_distribution_audit_locked_seed66_67_68_split20260601_20260529

status:
  pass

packets audited:
  14940

latent blocks aligned to native trace:
  27399

structural errors / forbidden fields / latent load errors:
  0 / 0 / 0

native alignment max_abs_diff:
  0.0

corrupted controls:
  metadata_only, shuffle, cross_task, wrong_block, noise, rotation

pair-distance AUROC:
  min = 1.0

control warning:
  1 shuffle packet fell back to any-task same-block-count replacement
  because no same-task same-block-count substitute existed.
```

Interpretation：

```text
P2-B supports E1: packet refs are loadable, metadata matches latent shards,
matched latent stats exactly align with native trace process features, and
corrupted payloads are separable under audit-time pair-distance checks.

It still does not prove Agent B can read/use the latent packet. P2-C must train
or evaluate a receiver where matched latent beats metadata-only and corrupted
controls under the same receiver contract.
```

## P2-C Target Feasibility Audit

```text
script:
  /data1/luyifei/drla/drla/scripts/audit_cola_latent_receiver_targets.py

output:
  /data1/luyifei/drla/outputs/cola_agent_latent_comm/
  p2_latent_receiver_target_audit_locked_seed66_67_68_split20260601_20260529

status:
  warn

packets:
  14940

offline accept / unsafe:
  14936 / 4

unsafe rate:
  0.0268%

loss_vs_final / loss_vs_prediction_stability:
  3 / 4

naive all-accept accuracy:
  99.9732%
```

Interpretation：

```text
Do not train a plain accept/defer BCE receiver as the main P2-C evidence on
this locked set. The unsafe class is too sparse; a constant accept policy would
look excellent and would not prove Agent B reads latent payloads.

For P2-C, accept/defer should be an auxiliary rare-event/risk audit unless
richer boundary examples are constructed. The main receiver-readability claim
needs a balanced matched-vs-corrupted or context-payload compatibility setup
that still respects the decoder-free online contract.
```

## P2-C Receiver Compatibility

P2-C main objective was revised from sparse `accept/defer` BCE to balanced
decoder-free compatibility classification:

```text
positive:
  matched packet latent payload

negative:
  metadata_only, shuffle, cross_task, wrong_block, noise, rotation

forbidden online inputs:
  decoded text, token ids, gold/scorer outputs, correctness labels, control_type
```

Implemented scripts：

```text
train:
  /data1/luyifei/drla/drla/scripts/train_cola_latent_receiver.py

aggregate:
  /data1/luyifei/drla/drla/scripts/aggregate_cola_latent_receiver.py
```

Full ablation aggregate：

```text
output:
  /data1/luyifei/drla/outputs/cola_latent_receiver/
  p2c_receiver_compat_bestckpt_eval_aggregate_seed20260529_20260529

table:
  receiver_ablation_summary.csv
```

Full ablation results, evaluated from each run's `best_checkpoint.pt`：

| input_mode | SwanLab | test mean AUROC | shuffle AUROC | interpretation |
|---|---:|---:|---:|---|
| envelope_only | poysm6qvybl7nipzqpzfj | 0.4999 | 0.4999 | negative control, no signal |
| process_only | 8nchledz4d108uka1m9zl | 0.5000 | 0.5000 | negative control, no signal |
| certificate_only | j5imgo3417t9ieqz4ngbm | 0.5000 | 0.5000 | certificate does not leak control label |
| latent_only | q3c18vawupb3lbtdai29n | 0.8963 | 0.5040 | detects distribution/structure controls, weak on same-task shuffle |
| latent_process | b8tek4rtj4vvph3q8z4r4 | 0.9115 | 0.5687 | process features help compatibility |
| latent_process_certificate | 50bumvfh2pgp0olw60jvr | 0.9194 | 0.6205 | best mean-control AUROC |
| latent_process_certificate_no_task | co6r6l6qw4lzxnu4mxkn2 | 0.9127 | 0.6511 | stronger shuffle, weaker cross-task/mean |

Interpretation：

```text
P2-C gives controlled decoder-free evidence that receiver-side model inputs
containing matched latent payload are useful: latent_process_certificate beats
envelope/process/certificate-only controls by about +0.419 mean AUROC.

However, this is still compatibility/readability evidence, not downstream task
utility. The hardest control remains same-task shuffle: the best mean model has
shuffle AUROC 0.6205, and no-task improves shuffle to 0.6511 while lowering
mean/cross-task. P2-D should therefore focus on whether this compatibility
signal helps actual receiver continuation, not claim text-channel superiority.
```

## P2-D Sequential Latent Replay

Implemented scripts：

```text
runner:
  /data1/luyifei/drla/drla/scripts/run_cola_sequential_latent_mas.py

audit:
  /data1/luyifei/drla/drla/scripts/audit_cola_sequential_latent_mas.py

text baseline:
  /data1/luyifei/drla/drla/scripts/build_cola_text_handoff_baseline.py
```

The runner is local-only generation/eval, not training. It re-encodes the
shared context, replays Agent A packet latent blocks into official Cola
DiT/VAE caches, then continues generation under the same block budget.

Protocol correction, 2026-05-31：

```text
The implemented text baseline above is now classified as a direct decoded-answer
handoff diagnostic, not the main P2-D text-channel baseline. It writes
selected_prediction/final_prediction/prediction_stability_prediction directly
to the official scorer and does not feed text into Agent B.

The next canonical P2-D evaluation must compare Agent B outputs under
LatentMAS-aligned message_only receiver conditions:

  B_none(empty input)
  B_text(A_raw_text_message_t)
  B_latent(A_latent_packet_t)
  B_corrupt(corrupted_latent_packet_t)

Only final Agent B outputs are scored. selected_prediction is not a valid
A_text_message_t because it is a task/scorer-extracted answer state rather than
the raw text emitted at the communication boundary.

shared_context, where B also receives the original benchmark prompt q, is kept
only as a diagnostic. It must not be used as the main Agent-A -> Agent-B
communication claim.
```

Corrected channel-equivalent scripts：

```text
build_cola_agent_channel_messages.py:
  implemented. Construct paired A_raw_text_message_t and A_latent_packet_t
  from the same Agent A trajectory/depth t. The text payload is native trace
  decode_text_so_far at the selected block, not selected_prediction.

run_cola_agent_b_channel_eval.py:
  implemented. Run B_none/B_text/B_latent/B_corrupt with same receiver
  budget and scorer-ready final outputs.

aggregate_cola_channel_eval.py:
  implemented. Score final Agent-B outputs post-hoc with official scorer
  rules and report paired win/loss/tie, cost, and decision-rule readouts.
```

Corrected channel-equivalent smoke, 2026-05-31：

```text
message artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_channel_messages_smoke_1per_task_20260531

eval artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_smoke_1per_task_20260531

aggregate artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_smoke_1per_task_20260531/channel_eval_aggregate

scope:
  official8, 1 message per task, 8 channels, 64 final Agent-B generations
  swanlab_mode=disabled, CUDA generation, no optimizer/backward

online-input audit:
  A_text_message_source = native_trace.decode_text_so_far_at_handoff_depth
  selected_prediction_used_as_text = false
  gold_or_scorer_used_online = false

official scorer smoke readout:
  latent_matched = 25.00% accuracy, mean score 0.5754
  text_raw_message = 25.00% accuracy, mean score 0.5799
  none = 25.00% accuracy, mean score 0.6059
  corrupted latent controls = 0.00% accuracy, mean score 0.1720-0.2005

decision-rule smoke:
  matched beats all corrupted controls by score = true
  matched beats none = false
  matched is score-competitive with raw text at tolerance 0.01 = true

interpretation:
  the corrected protocol is executable and separates matched latent from
  corrupted controls. The 8-sample run is only an implementation smoke test,
  not a scientific estimate or paper number.
```

Corrected channel-equivalent official8 50/task unique-sample result, 2026-05-31：

```text
message artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_channel_messages_official8_50per_task_seed20260531_unique_20260531

sharded generation artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_official8_50per_task_seed20260531_unique_20260531_sharded6

merged eval artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_official8_50per_task_seed20260531_unique_20260531_merged

aggregate artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_official8_50per_task_seed20260531_unique_20260531_merged/channel_eval_aggregate

scope:
  official8, 50 unique sample_key per task
  400 messages, 8 channels, 3200 final Agent-B generations
  swanlab_mode=disabled, CUDA sharded generation, no optimizer/backward

protocol audit:
  selected_prediction_used_as_text = false
  unique sample_key = 400 / 400
  merged duplicate keys = 0
  each channel count = 400
```

Official scorer channel summary：

```text
latent_matched:
  accuracy = 23.50%
  mean_score = 0.4850

B_text_raw_message:
  accuracy = 23.00%
  mean_score = 0.5011

B_none:
  accuracy = 19.75%
  mean_score = 0.4410

corrupted latent controls:
  accuracy = 0.00% to 4.25%
  mean_score = 0.1699 to 0.2499
```

Paired deltas, latent_matched minus compare：

```text
vs corrupted controls:
  score_delta = +0.2351 to +0.3151
  score_delta_ci95 lower bounds all > +0.1959
  accuracy_delta = +19.25pp to +23.50pp
  accuracy_delta_ci95 lower bounds all > +15.00pp

vs B_none:
  score_delta = +0.0440, CI95 [+0.0111, +0.0758]
  accuracy_delta = +3.75pp, CI95 [-0.25pp, +7.50pp]

vs B_text_raw_message:
  score_delta = -0.0161, CI95 [-0.0406, +0.0076]
  accuracy_delta = +0.50pp, CI95 [-2.50pp, +3.75pp]
```

Interpretation, superseded for Agent-B communication claims：

```text
This table is no longer valid as an Agent-B communication result. A later
data-flow audit found that the historical all-visible output scope allowed A
raw text/replay decoded tokens to enter final `generate` and be scored. Keep it
only as evidence that A's matched latent packet is decodable/task-relevant under
Cola VAE replay.

An earlier 50/task non-deduped run found only 375 unique sample_key among 400
messages and is kept only as a protocol audit. Do not cite it as the formal
P2-D number.
```

Receiver-native cache-only audit, 2026-05-31：

```text
goal:
  test whether Agent B can benefit from A latent blocks without decoding/emitting
  A replay blocks as text.

channels:
  B_none
  B_text(A_raw_text_message_t)
  latent_matched_cache_only
  latent_*_cache_only corrupted controls

implementation:
  replay latent blocks update VAE/DiT KV cache
  replay blocks are not sampled into output text
  replay_blocks_decoded_to_text = 0

artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_official8_50per_task_cache_only_seed20260531_unique_20260531_merged/channel_eval_aggregate

scope:
  same 400 unique messages as the decode-and-emit official8 50/task run
  8 channels, 3200 final Agent-B generations
```

Official scorer cache-only summary：

```text
latent_matched_cache_only:
  accuracy = 1.50%
  mean_score = 0.1916

B_none:
  accuracy = 19.75%
  mean_score = 0.4410

B_text_raw_message:
  accuracy = 23.00%
  mean_score = 0.5011

corrupted cache-only controls:
  accuracy = 0.75% to 3.25%
  mean_score = 0.1559 to 0.2261
```

Paired cache-only deltas, latent_matched_cache_only minus compare：

```text
vs B_none:
  score_delta = -0.2493, CI95 [-0.2851, -0.2152]
  accuracy_delta = -18.25pp, CI95 [-22.50pp, -14.25pp]

vs B_text_raw_message:
  score_delta = -0.3094, CI95 [-0.3484, -0.2686]
  accuracy_delta = -21.50pp, CI95 [-25.50pp, -17.00pp]

vs corrupted cache-only controls:
  matched beats cross_task/noise by score
  matched does not beat all controls; wrong_block cache_only is stronger
```

Interpretation：

```text
The current P2-D positive result should be described precisely:

  latent packets are useful as same-substrate, decodable communication payloads
  under decode-and-emit replay.

It should not yet be described as receiver-native no-text latent reasoning.
When replay blocks are not emitted as text, matched latent does not beat B_none
or raw text and does not cleanly dominate corrupted cache-only controls.
This points the next work toward a receiver architecture/objective that consumes
latent state natively, rather than only replaying the sender's latent trajectory.
```

Implementation clarification, 2026-05-31：

```text
The latent packet is not re-created by text encoding.  Agent-B loads A's latent
blocks directly from the packet latent shard.  In the replay runner, those
blocks enter Cola DiT directly as `dit(txt=z, update_kv=True)`.

The earlier cache-only path was not DiT-only: it updated both VAE decoder KV
cache and DiT KV cache while emitting no replay text.  New smoke-only ablation
channels now separate:

  latent_*_cache_only      = VAE decoder cache + DiT cache, no replay text
  latent_*_dit_only_cache  = DiT cache only
  latent_*_vae_only_cache  = VAE decoder cache only
```

Decoder semantic-projection gap audit：

```text
script:
  /data1/luyifei/drla/drla/scripts/audit_cola_channel_projection_gap.py

artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_projection_gap_official8_50per_task_seed20260531_unique_20260531

scope:
  same 400 unique official8 50/task messages
  paired latent_matched decode-and-emit vs latent_matched_cache_only

result:
  decode_emit_mean_score = 0.4850
  cache_only_mean_score = 0.1916
  projection_score_gain = +0.2934, CI95 [+0.2551, +0.3320]
  decode_emit_accuracy = 23.50%
  cache_only_accuracy = 1.50%
  projection_accuracy_gain = +22.00pp, CI95 [+17.75pp, +26.25pp]
```

Direct DiT/VAE cache ablation smoke：

```text
artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_smoke_direct_dit_1per_task_20260531_v2/channel_eval_aggregate

scope:
  official8, 1 message per task, protocol smoke only

scores:
  latent_matched = 25.00%, mean_score 0.5754
  latent_matched_cache_only = 0.00%, mean_score 0.2082
  latent_matched_dit_only_cache = 0.00%, mean_score 0.2082
  latent_matched_vae_only_cache = 0.00%, mean_score 0.3373

interpretation:
  Direct DiT-only replay is executable but does not recover decode-and-emit
  utility on this tiny smoke.  The failure is not because latent blocks were
  accidentally routed through a text encoder; the missing piece is a learned
  receiver interface/alignment objective.
```

Data-flow correction：

```text
The latent decode-and-emit channel is not:
  A latent -> decoder text -> Agent-B VAE encoder -> Agent-B DiT

The text channel is the one that calls `vae.encode(text)`.

The latent decode-and-emit channel is:
  A latent z -> VAE decoder logits/tokens -> final scorer-visible `generate`
  A latent z -> DiT KV cache

Thus the decode-and-emit score can be high because A's decoded replay tokens
already expose the answer or useful answer prefix.  This remains useful
evidence that the latent packet is decodable and task-relevant, but it should
not be cited as proof that Agent B natively understands A's latent without
decoder-mediated emission.
```

Corrected receiver-only smoke, 2026-05-31：

```text
code change:
  run_cola_agent_b_channel_eval.py now defaults to:
    --score-output-scope receiver_only

scoring rule:
  scorer sees only Agent-B tokens generated after handoff.
  A text message and A latent replay decoded tokens may condition B according
  to the channel, but they are excluded from final `generate`.

legacy behavior:
  --score-output-scope legacy_all_visible
  kept only to reproduce historical diagnostics.

artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_smoke_receiver_only_1per_task_20260531/channel_eval_aggregate

leak audit:
  generations = 48
  score_output_scope = receiver_only
  sum(scorer_visible_text_message_tokens) = 0
  sum(scorer_visible_replay_blocks) = 0

official scorer smoke:
  none = 25.00%, mean_score 0.6059
  text = 12.50%, mean_score 0.2996
  latent_matched = 0.00%, mean_score 0.2082
  latent_matched_cache_only = 0.00%, mean_score 0.2082
  latent_matched_dit_only_cache = 0.00%, mean_score 0.2082
  latent_matched_vae_only_cache = 0.00%, mean_score 0.3373

interpretation:
  The previous decode-and-emit benefit collapses when replay tokens are not
  scored directly. This confirms the old result bypassed the intended Agent-B
  answer-generation boundary. Current status: P2-D communication claim is
  reset to "not established"; the only retained evidence is latent
  decodability and negative/weak native receiver consumption.
```

LatentMAS-aligned message-only receiver-only smoke after replay-EOS fix, 2026-05-31：

```text
why:
  LatentMAS text MAS appends one agent's output to the next agent, and latent
  MAS transfers the predecessor's latent working memory/KV state. Therefore the
  canonical P2-D handoff should make Agent B observe only Agent A's message or
  latent packet. Giving B the original benchmark prompt is shared_context
  diagnostic, not the main communication protocol.

artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_smoke_message_only_receiver_only_eosfix_1per_task_20260531/channel_eval_aggregate

protocol:
  agent_b_input_contract = message_only
  score_output_scope = receiver_only

code fix:
  replay EOS/im_end from Agent A is recorded, but no longer stops Agent B in
  receiver_only mode. Only B's own generated stop tokens can end the receiver
  continuation.

boundary audit:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_protocol_boundary_audit_message_only_eosfix_smoke_20260531
  status = pass
  scorer_visible_text_message_tokens = 0
  scorer_visible_replay_blocks = 0

official scorer smoke:
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
  This is a leakage-free protocol smoke only. It confirms the corrected path is
  executable, but it is too small for channel-quality claims. The next formal
  P2-D artifact must be official8 50/task message_only + receiver_only.
```

Formal LatentMAS-aligned P2-D result, 2026-05-31：

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
```

Official scorer：

| channel | accuracy | mean score |
|---|---:|---:|
| `latent_matched` | 1.75% | 0.1874 |
| `latent_matched_cache_only` | 1.75% | 0.1912 |
| `latent_matched_dit_only_cache` | 1.25% | 0.1873 |
| `latent_matched_vae_only_cache` | 0.00% | 0.1395 |
| `none` | 0.00% | 0.1378 |
| `text` | 1.25% | 0.1795 |
| `latent_cross_task` | 0.25% | 0.1535 |
| `latent_shuffle` | 0.25% | 0.1663 |
| `latent_wrong_block` | 4.00% | 0.1950 |
| `latent_noise` | 0.00% | 0.1390 |
| `latent_rotation` | 0.00% | 0.1297 |

Paired comparisons：

```text
latent_matched - none:
  score_delta = +0.0496, CI95 [+0.0340, +0.0659]
  accuracy_delta = +1.75 pp, CI95 [+0.50, +3.00]

latent_matched - text:
  score_delta = +0.0079, CI95 [-0.0082, +0.0259]
  accuracy_delta = +0.50 pp, CI95 [-1.00, +2.00]

latent_matched - latent_wrong_block:
  score_delta = -0.0076, CI95 [-0.0286, +0.0140]
  accuracy_delta = -2.25 pp, CI95 [-4.50, -0.25]

latent_matched - latent_matched_cache_only:
  score_delta = -0.0038, CI95 [-0.0064, -0.0012]
```

Interpretation：

```text
Current P2-D status:
  valid receiver-only Agent-B communication evidence exists.
  latent_matched has significant marginal utility over empty input.
  latent_matched is competitive with text, but does not significantly beat text.
  the all-corrupt-control gate is not passed because wrong_block is anomalously
  strong.

Scientific claim allowed:
  "Under a LatentMAS-aligned message_only + receiver_only protocol, Cola latent
  handoff carries usable signal for Agent B beyond empty input."

Scientific claim not allowed:
  "Latent communication beats text communication."
  "B specifically understands the matched payload better than every corrupted
  latent control."
```

Smoke artifact：

```text
/data1/luyifei/drla/outputs/cola_sequential_latent_mas/
p2d_replay_smoke_official8_1per_task_controls_fixsample2_20260529

packets:
  8 total, 1 per official task

status:
  pass

official scorer:
  matched = 12.5%
  metadata_only = 12.5%
  cross_task/noise/rotation/shuffle/wrong_block = 0.0%
```

Early 5-per-task diagnostic artifact：

```text
/data1/luyifei/drla/outputs/cola_sequential_latent_mas/
p2d_replay_eval_official8_5per_task_controls_20260529

packets:
  40 total, 5 per official task

controls:
  matched, metadata_only, shuffle, cross_task, wrong_block, noise, rotation

runner status:
  pass
  no control generation warnings
  all controls nonempty

official scorer accuracy:
  matched = 17.5%
  metadata_only = 20.0%
  wrong_block = 5.0%
  shuffle/cross_task/noise/rotation = 0.0%

offline answer-prefix fidelity to P1 selected/final/prediction-stability references:
  matched = 37.5%
  metadata_only = 27.5%
  noise = 7.5%
  shuffle/cross_task/wrong_block/rotation = 2.5%
```

Direct-answer handoff diagnostic on the same 40 samples：

```text
artifact:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_eval_official8_5per_task_controls_20260529/text_handoff_baseline

official scorer:
  text_selected = 30.0%
  text_final = 30.0%
  text_prediction_stability = 30.0%
```

Replay-only mismatch diagnostic：

```text
artifact:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_only_eval_official8_5per_task_controls_20260529

setting:
  receiver_budget_mode = fixed
  fixed_receiver_blocks = 0

official scorer:
  matched = 17.5%
  metadata_only = 0.0%
  wrong_block = 5.0%
  shuffle/cross_task/noise/rotation = 0.0%

offline answer-prefix fidelity:
  matched = 37.5%
  noise = 7.5%
  shuffle/cross_task/wrong_block/rotation = 2.5%

raw trace audit:
  matched replay-only vs native trace selected-block raw text = 60.0%
  native trace selected-block raw text vs P1 selected_prediction = 27.5%
```

Expanded 20-per-task P2-D diagnostic：

```text
replay-only:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_only_eval_official8_20per_task_controls_20260529

sequential continuation:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_continue_eval_official8_20per_task_controls_20260529

text baseline:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_only_eval_official8_20per_task_controls_20260529/text_handoff_baseline

scope:
  official8
  20 packets per task
  160 packets total
  local-only, swanlab disabled, no training

official scorer:
  replay-only matched = 24.38%
  replay-only metadata_only = 0.00%
  replay-only shuffle/cross_task/wrong_block = 1.88% / 0.62% / 0.62%
  replay-only noise/rotation = 0.00% / 0.00%
  replay+continue matched = 24.38%
  replay+continue metadata_only = 21.88%
  text_selected = 25.00%
  text_final = 25.00%
  text_prediction_stability = 25.00%

expanded audit:
  selected_reference_accuracy = 25.0%
  matched official_prediction_agrees_selected = 65.0%
  matched correct_selected_preservation_rate = 80.0%
  matched incorrect_selected_prediction_reproduction_rate = 60.8%
  matched vs native trace selected-block raw text = 56.9%
  native trace official prediction agreement with selected_prediction = 45.6%
```

Interpretation：

```text
P2-D currently supports executable same-substrate latent replay and shows that
matched latent carries more native-answer signal than metadata-only/corrupted
controls under offline fidelity metrics.

The 160-sample diagnostic remains useful latent-readability evidence. Matched
latent replay is far above corrupted latent controls and numerically near the
direct-answer text_selected diagnostic (24.38% vs 25.00%). Sequential
continuation preserves the matched replay score rather than degrading it.
Matched+continue also modestly beats metadata_only continue (24.38% vs 21.88%).

This supports E2/E3-style same-substrate latent-readability claims on the
diagnostic subset:
B can read/use matched latent payload and the payload carries task information
well beyond corrupted controls. It still does not support E4 text superiority:
text_selected is a direct-answer diagnostic, not a valid Agent-B text-message
baseline, and the matched-vs-metadata gain is small.

Next step should repair the channel comparison rather than scale the old direct
handoff diagnostic: run B_text(A_raw_text_message_t) and
B_latent(A_latent_packet_t) under the same message_only receiver budget and
receiver_only scorer, plus B_none(empty input) and corrupted latent controls.
```

Fresh 50-per-task P2-D validation, 2026-05-29：

```text
replay-only:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_only_eval_official8_50per_task_seed20260530_controls_20260529

replay+continue:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_continue_eval_official8_50per_task_seed20260530_controls_20260529

scope:
  official8
  fresh selection seed = 20260530
  50 packets per task
  400 packets total
  controls = matched, metadata_only, shuffle, cross_task, wrong_block, noise, rotation
  local-only, swanlab disabled, no training
```

Official scorer：

```text
replay-only:
  matched = 24.25%
  metadata_only = 0.00%
  corrupted latent controls = 0.00% to 3.75%
  text_selected/text_final/text_prediction_stability = 23.50%

replay+continue:
  matched = 24.25%
  metadata_only = 23.75%
  corrupted latent controls = 0.25% to 4.25%
  text_selected/text_final/text_prediction_stability = 23.50%
```

Duplicate-safe audit：

```text
reason for audit fix:
  the 50-per-task fresh subset contains duplicate task ids/sample_keys
  old audit keys could overwrite scorer correctness and undercount paired rows

replay-only matched:
  official_prediction_agrees_selected = 63.5%
  correct_selected_preservation_rate = 75.5%
  native_trace_selected_agreement_rate = 51.25%
  paired net wins vs metadata_only / corrupted controls:
    +97 vs metadata_only
    +82 to +97 vs corrupted latent controls

replay+continue matched:
  official_prediction_agrees_selected = 63.25%
  correct_selected_preservation_rate = 75.5%
  native_trace_selected_agreement_rate = 50.5%
  paired net wins:
    +2 vs metadata_only
    +80 to +96 vs corrupted latent controls
```

Interpretation：

```text
The fresh 400-packet result stabilizes P2-D E2/E3 evidence:
matched latent packets are readable/useful under the same Cola substrate and
beat corrupted latent payloads by a large margin.

It does not prove E4 text superiority. Matched is only +0.75 points over
the direct-answer diagnostic, and metadata_only+continue is only 0.50 points
below matched.
This means shared-context continuation can solve much of this subset without
the latent payload. Future P2 work must keep metadata/text controls explicit
and should use the corrected Agent-B channel-equivalent protocol before making
any text-vs-latent claim.
```

Message-only marginal-utility diagnostic：

```text
artifact:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_message_only_replay_only_official8_50per_task_seed20260531_controls_20260529

protocol:
  receiver_context_mode = empty_prompt
  receiver_budget_mode = fixed
  fixed_receiver_blocks = 0

scope:
  official8
  fresh selection seed = 20260531
  50 packets per task
  400 packets total
  controls = matched, metadata_only, shuffle, cross_task, wrong_block, noise, rotation
  local-only, swanlab disabled, no training
```

Official scorer：

```text
matched latent = 12.50%
metadata_only = 0.00%
corrupted latent controls = 0.00% to 2.00%
text_selected/text_final/text_prediction_stability = 26.25%
```

Audit：

```text
matched official_prediction_agrees_selected = 16.25%
matched correct_selected_preservation_rate = 19.05%
matched native_trace_selected_agreement_rate = 7.5%
paired net wins:
  +50 vs metadata_only
  +42 to +50 vs corrupted latent controls
```

Interpretation：

```text
This stricter protocol removes B's full task-context re-encoding. It shows that
the matched latent packet alone carries recoverable signal, because it beats
metadata_only and corrupted controls. But it is much worse than direct text
handoff on the same 400 samples (12.50% vs 26.25%).

Current answer to "latent communication better than text communication":
  No, not under the current replay/message interface.

Current supported claim:
  Same-substrate latent packets are readable and useful, but text-channel
  superiority is not established. The next improvement target is corrected
  Agent-B channel evaluation plus receiver-native no-text latent consumption,
  not stronger wording.
```

## P2-D Receiver-Side Latent Answer Reader

```text
training artifact:
  /data1/luyifei/drla/outputs/cola_latent_answer_reader/
  p2_answer_reader_full_seed20260529_20260529

best-checkpoint eval artifacts:
  valid:
    /data1/luyifei/drla/outputs/cola_latent_answer_reader/
    p2_answer_reader_full_seed20260529_20260529_best_eval_valid
  test:
    /data1/luyifei/drla/outputs/cola_latent_answer_reader/
    p2_answer_reader_full_seed20260529_20260529_best_eval_test

training:
  SwanLab cloud run = x6yc77eedf77z27ego0ve
  CUDA/GPU
  epochs = 12
  batch_size = 256
  valid_interval = 50
  best_checkpoint.pt selected by valid official_top1_accuracy
```

Best checkpoint：

```text
valid:
  answer_key_top1 = 13.54%
  official_top1_accuracy = 11.11%
  selected_reference_accuracy = 21.48%

test:
  answer_key_top1 = 15.20%
  official_top1_accuracy = 10.61%
  selected_reference_accuracy = 22.11%
```

Interpretation：

```text
The latent answer reader is a useful diagnostic, but not a positive text-beating
result. It learns some latent-to-answer-state signal, yet its best checkpoint is
well below the same-split teacher/text reference and below the 400-packet
message-only replay matched-latent result.

Current answer to "latent communication better than text communication":
  No.

Current supported claim:
  Latent packets contain recoverable answer/task signal under same-substrate
  controls. The currently tested receiver interfaces do not yet beat ordinary
  text handoff.
```

## P2-E Hierarchical Aggregation

Potential audit：

```text
script:
  /data1/luyifei/drla/drla/scripts/audit_cola_hierarchical_aggregation_potential.py

artifact:
  /data1/luyifei/drla/outputs/cola_hierarchical_aggregation/
  p2e_aggregation_potential_locked_seed66_67_68_split20260601_20260529

scope:
  official8 held-out packet set
  4980 same-sample groups
  3 sender packets per group: seed66 / seed67 / seed68
  local-only, swanlab disabled
```

Aggregate result：

```text
single_sender_first = 20.74%
prediction_change_min_selected = 21.39%
text_majority_selected = 21.55%
oracle_any_selected_correct = 33.13%
oracle_any_final_correct = 33.15%
```

Interpretation：

```text
There is meaningful multi-sender headroom: at least one of the three selected
answers is correct for 33.13% of groups. But naive text majority and simple
latent-readiness/risk rankers barely improve over a single sender.
```

Learned latent fuser v1：

```text
train script:
  /data1/luyifei/drla/drla/scripts/train_cola_hierarchical_latent_fuser.py

eval script:
  /data1/luyifei/drla/drla/scripts/eval_cola_hierarchical_latent_fuser.py

training artifact:
  /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
  p2e_hierarchical_fuser_full_seed20260529_20260529

best-checkpoint eval:
  valid:
    /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
    p2e_hierarchical_fuser_full_seed20260529_20260529_best_eval_valid
  test:
    /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
    p2e_hierarchical_fuser_full_seed20260529_20260529_best_eval_test

training:
  SwanLab cloud run = ljv0m43x48a49j1at6gx9
  CUDA/GPU
  epochs = 24
  batch_size = 256
  valid_interval = 50
  best_checkpoint.pt selected by valid model_selected_accuracy
```

Best checkpoint：

```text
valid:
  model_selected_accuracy = 23.03%
  single_sender_first_accuracy = 23.23%
  text_majority_selected_accuracy = 22.42%
  oracle_any_selected_accuracy = 33.33%

test:
  model_selected_accuracy = 20.74%
  single_sender_first_accuracy = 22.38%
  text_majority_selected_accuracy = 23.41%
  oracle_any_selected_accuracy = 34.70%
```

Interpretation：

```text
This is a negative but useful P2-E result. The fuser reads only decoder-free
latent/process/certificate fields and can identify a correct sender in 59.76%
of test groups where any sender is correct, but it still underperforms the
single fixed sender and text-majority baselines on held-out test accuracy.

Current answer to "latent communication better than text communication":
  Still no.

Current P2-E lesson:
  Multi-sender latent packets contain complementary signal, but the current
  sparse selected-correct sender-selection objective is not enough to realize
  the oracle headroom.
```

Score-target latent fuser v2：

```text
reason:
  selected_correct is too sparse. An audit found 7764 / 14940 sender predictions
  have non-binary official scores, and 2570 / 4980 groups have a partial-utility
  best sender without exact correctness.

train script:
  /data1/luyifei/drla/drla/scripts/train_cola_hierarchical_latent_fuser.py

new option:
  --target-mode score

training artifact:
  /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
  p2e_hierarchical_fuser_score_full_seed20260529_20260529

best-checkpoint eval:
  valid:
    /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
    p2e_hierarchical_fuser_score_full_seed20260529_20260529_best_eval_valid
  test:
    /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
    p2e_hierarchical_fuser_score_full_seed20260529_20260529_best_eval_test

training:
  SwanLab cloud run = o5fjvuiqk82nk9c5hihn0
  CUDA/GPU
  epochs = 24
  batch_size = 256
  valid_interval = 50
  best_checkpoint.pt selected by valid model_mean_official_score
```

Best checkpoint：

```text
valid:
  model_selected_accuracy = 22.63%
  single_sender_first_accuracy = 23.23%
  text_majority_selected_accuracy = 22.42%
  oracle_any_selected_accuracy = 33.33%
  model_mean_official_score = 0.3756
  single_sender_first_mean_official_score = 0.3711
  text_majority_mean_official_score = 0.3617
  oracle_best_selected_mean_official_score = 0.4904

test:
  model_selected_accuracy = 23.41%
  single_sender_first_accuracy = 22.38%
  text_majority_selected_accuracy = 23.41%
  oracle_any_selected_accuracy = 34.70%
  model_mean_official_score = 0.3685
  single_sender_first_mean_official_score = 0.3553
  text_majority_mean_official_score = 0.3622
  oracle_best_selected_mean_official_score = 0.4951
```

Interpretation：

```text
This is the strongest P2-E result so far, but still not a broad text-superiority
result. Compared with v1, the score-target objective is better aligned with the
official utility surface. On held-out test it beats a single fixed sender in
both exact accuracy and mean score, ties text majority in exact accuracy, and
beats text majority in mean score.

However, per-task results remain uneven, and the oracle gap is still large.
Current supported claim:
  richer utility supervision makes decoder-free latent fusing useful on some
  aggregate metrics, but a robust latent-over-text communication claim is not
  established yet.
```

Task-balanced / task-aware follow-ups：

```text
task-balanced v3:
  artifact:
    /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
    p2e_hierarchical_fuser_score_taskbalanced_full_seed20260529_20260529
  best eval:
    /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
    p2e_hierarchical_fuser_score_taskbalanced_full_seed20260529_20260529_best_eval_test
  SwanLab cloud run:
    k2ujjjdrmcyzutwnwbyyf
  test:
    model_selected_accuracy = 22.38%
    model_mean_official_score = 0.3654
    text_majority_selected_accuracy = 23.41%
    text_majority_mean_official_score = 0.3622

task-aware v4:
  artifact:
    /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
    p2e_hierarchical_fuser_taskaware_score_full_seed20260529_20260529_metricfix
  best eval:
    /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
    p2e_hierarchical_fuser_taskaware_score_full_seed20260529_20260529_metricfix_best_eval_test
  SwanLab cloud run:
    fdjwe4vfq70syq0oz9tro
  test:
    model_selected_accuracy = 22.18%
    model_mean_official_score = 0.3605
    text_majority_selected_accuracy = 23.41%
    text_majority_mean_official_score = 0.3622
```

Interpretation：

```text
Both follow-ups are negative relative to score-target v2. Simple task-balanced
loss weighting does not improve micro or macro behavior. Task-aware exact/score
targets also underperform v2 after fixing checkpoint selection to use mean
official score for task_aware_score.

Current best P2-E model remains score-target v2:
  /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
  p2e_hierarchical_fuser_score_full_seed20260529_20260529
```

Latent-state utility verifier：

```text
train script:
  /data1/luyifei/drla/drla/scripts/train_cola_hierarchical_state_verifier.py

artifact:
  /data1/luyifei/drla/outputs/cola_hierarchical_state_verifier/
  p2e_state_verifier_full_seed20260529_20260529

SwanLab cloud run:
  brtvqv9yd3h2gbcsu25n5

objective:
  predict group-level any_correct and best_score from decoder-free multi-sender
  latent/process/certificate inputs.
```

Best checkpoint：

```text
valid:
  any_auroc = 0.7204
  any_accuracy_at_0_5 = 59.80%
  any_brier = 0.2172
  best_score_corr = 0.3212
  best_score_rmse = 0.3932

test:
  any_auroc = 0.7054
  any_accuracy_at_0_5 = 58.52%
  any_brier = 0.2208
  best_score_corr = 0.3078
  best_score_rmse = 0.4029
```

Same-test heuristic baselines：

```text
max_correctness_head:
  any_auroc = 0.4717
  best_score_corr = -0.0769
  best_score_rmse = 0.4471

max_readiness:
  any_auroc = 0.4776
  best_score_corr = -0.1232
  best_score_rmse = 0.6466

max_answer_identity_stability:
  any_auroc = 0.5079
  best_score_corr = -0.0762
  best_score_rmse = 0.6044
```

Calibration / input-prior ablation：

```text
eval artifact:
  /data1/luyifei/drla/outputs/cola_hierarchical_state_verifier/
  p2e_state_verifier_full_seed20260529_20260529_calibration_ablation

raw full test:
  any_auroc = 0.7054
  any_brier = 0.2208
  ece_10 = 0.1444
  any_prob_mean = 0.4894
  any_target_mean = 0.3470
  best_score_corr = 0.3078
  best_score_rmse = 0.4029

valid-fitted calibrated test:
  any_auroc = 0.7054
  any_accuracy_at_0_5 = 67.15%
  any_brier = 0.1985
  ece_10 = 0.0232
  any_prob_mean = 0.3418
  any_target_mean = 0.3470

zero_latent test:
  any_auroc = 0.6731
  best_score_corr = 0.2317
  best_score_rmse = 0.4132

valid task-prior-only test:
  any_auroc = 0.6399
  any_brier = 0.2050
  best_score_corr = 0.2399
  best_score_rmse = 0.4124

valid global-prior-only test:
  any_auroc = 0.5142
  any_brier = 0.2268
  best_score_corr = 0.0000
  best_score_rmse = 0.4235
```

Risk-control diagnostic：

```text
target precision 0.6:
  threshold = 0.7323
  test precision = 0.609
  test recall = 0.231
  test coverage = 0.131

target precision 0.7 / 0.8:
  no non-trivial held-out coverage
```

Interpretation：

```text
This is positive P2-E evidence for direct latent-state utility modeling. The
verifier predicts whether the group contains an exact-correct sender much
better than simple certificate/readiness heuristics. It also obtains a moderate
positive correlation with best available selected-score utility.

The task-prior baseline is important: benchmark identity alone already provides
non-trivial signal, but it does not explain the full model. Full AUROC 0.7054
is above task-prior 0.6399, and zeroing raw latent blocks drops the verifier to
0.6731. The current best reading is therefore: raw latent states add measurable
utility information beyond task difficulty and certificate heuristics.

Limit:
  This is not yet a downstream communication win. Calibration is now much
  better, but high-precision risk-control coverage remains low, and best-score
  regression is only moderate. Frame this as latent-state readability / utility
  prediction, not as final latent-over-text agent communication.
```

Receiver-state policy audit：

```text
eval script:
  /data1/luyifei/drla/drla/scripts/eval_cola_hierarchical_state_policy.py

artifact:
  /data1/luyifei/drla/outputs/cola_hierarchical_state_policy/
  p2e_state_policy_score_fuser_v2_locked_seed20260529_20260529

state checkpoint:
  /data1/luyifei/drla/outputs/cola_hierarchical_state_verifier/
  p2e_state_verifier_full_seed20260529_20260529/checkpoints/best_checkpoint.pt

fuser checkpoint:
  /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
  p2e_hierarchical_fuser_score_full_seed20260529_20260529/checkpoints/best_checkpoint.pt

local-only:
  swanlab disabled
  no training
  thresholds selected on valid and reported on held-out test
```

Always-on test baselines：

```text
single_sender_first:
  accuracy = 22.38%
  score = 0.3553

score-target latent fuser v2:
  accuracy = 23.41%
  score = 0.3685

text_majority_selected:
  accuracy = 23.41%
  score = 0.3622

oracle_any_selected:
  accuracy = 34.70%
  best score = 0.4951
```

Key state-gate results on test：

```text
state_any_prob, valid target any precision 0.60:
  coverage = 13.35%
  any precision = 61.54%
  accepted fuser accuracy = 53.85%
  accepted fuser score = 0.5503
  accepted text accuracy = 49.23%
  fuser-else-first fallback accuracy = 23.00%
  fuser-else-first fallback score = 0.3609

state_any_prob, valid target any precision 0.65:
  coverage = 11.29%
  any precision = 63.64%
  accepted fuser accuracy = 56.36%
  accepted fuser score = 0.5636
  accepted text accuracy = 50.91%

train task-prior any, valid target any precision 0.60:
  coverage = 12.32%
  any precision = 61.67%
  accepted fuser accuracy = 55.00%
  accepted fuser score = 0.5500
  accepted text accuracy = 48.33%

state_any_prob, valid target coverage 0.25:
  coverage = 22.18%
  any precision = 52.78%
  accepted fuser accuracy = 42.59%
  accepted fuser score = 0.5462

train task-prior any, valid target coverage 0.25:
  coverage = 49.69%
  any precision = 47.11%
  accepted fuser accuracy = 30.17%
  accepted fuser score = 0.3477
```

Interpretation：

```text
The calibrated state verifier is useful as a receiver-side state / risk signal:
it can pick a subset where latent fuser exact accuracy rises from 23.41% to
roughly 54-56%.

But this is still not a final downstream communication win. The fuser-else-first
fallback policy does not beat always-on fuser by score, and high-confidence
task-prior gating is competitive at very low coverage. The bottleneck is now
the action/selector that consumes the state, not whether the state contains
signal.
```

Structured receiver-state action selector：

```text
train script:
  /data1/luyifei/drla/drla/scripts/train_cola_receiver_state_action_selector.py

eval script:
  /data1/luyifei/drla/drla/scripts/eval_cola_receiver_state_action_selector.py

direct selector artifact:
  /data1/luyifei/drla/outputs/cola_receiver_state_action_selector/
  p2e_state_action_selector_state_fuser_prior_seed20260529_20260529

direct selector SwanLab:
  7emwtma3xyvrmvlv1hibb

residual selector artifact:
  /data1/luyifei/drla/outputs/cola_receiver_state_action_selector/
  p2e_state_action_selector_residual_state_fuser_prior_seed20260529_20260529

residual selector SwanLab:
  2z5uj588g6kkkv6dpte97

local held-out eval artifacts:
  p2e_state_action_selector_state_fuser_prior_seed20260529_20260529_best_eval
  p2e_state_action_selector_residual_state_fuser_prior_seed20260529_20260529_best_eval
```

Best-checkpoint held-out test：

```text
score-target latent fuser v2:
  accuracy = 23.41%
  score = 0.3685

text_majority_selected:
  accuracy = 23.41%
  score = 0.3622

direct state action selector:
  accuracy = 21.97%
  score = 0.3554

residual state action selector:
  accuracy = 22.79%
  score = 0.3626
```

Valid-selected gate audit on held-out test：

```text
direct selector, valid target any precision 0.60:
  coverage = 9.45%
  test any precision = 58.70%
  accepted selector accuracy = 39.13%
  accepted selector score = 0.4217
  fallback-first score = 0.3472

residual selector, valid target any precision 0.60:
  coverage = 9.24%
  test any precision = 60.00%
  accepted selector accuracy = 40.00%
  accepted selector score = 0.4252
  fallback-first score = 0.3466
```

Interpretation：

```text
This is a negative result for shallow structured-state action selection. The
residual selector is less harmful than direct logits and slightly exceeds
text-majority score, but it still trails the raw score-target latent fuser v2.

Therefore the current compressed state tuple is useful for risk/readiness
auditing, but not sufficient to replace raw latent sender-choice features. The
next P2-E step should not be more threshold tuning or another shallow MLP over
the same state. It should either introduce a true request-more-latent action
or train a richer sender-level selector that still keeps calibrated state as
side information.
```

State-conditioned sender-level latent fuser：

```text
train script:
  /data1/luyifei/drla/drla/scripts/train_cola_state_conditioned_latent_fuser.py

frozen residual artifact:
  /data1/luyifei/drla/outputs/cola_state_conditioned_latent_fuser/
  p2e_state_conditioned_fuser_frozen_state_fuser_prior_seed20260529_20260529

frozen residual SwanLab:
  1ua24n9yo4tsrq4inahb9

unfrozen residual artifact:
  /data1/luyifei/drla/outputs/cola_state_conditioned_latent_fuser/
  p2e_state_conditioned_fuser_unfrozen_state_fuser_prior_seed20260529_20260529

unfrozen residual SwanLab:
  je3suuujcleox4x40lahd
```

Best-checkpoint held-out test：

```text
score-target latent fuser v2:
  accuracy = 23.41%
  score = 0.3685

text_majority_selected:
  accuracy = 23.41%
  score = 0.3622

frozen state-conditioned fuser:
  accuracy = 23.00%
  score = 0.3654

unfrozen state-conditioned fuser:
  accuracy = 23.41%
  score = 0.3651
```

Interpretation：

```text
This is another negative result for "same action space, better selector". Even
when sender-level raw latent states are preserved and the state tuple is added
as side information, held-out test score remains below the original fuser v2.

The state side signal is useful for auditing/risk, but not yet a robust
sender-choice improvement. The next credible branch is request-more-latent /
additional evidence, not another reranker over the same three available
senders.
```

Request-more-latent potential audit：

```text
script:
  /data1/luyifei/drla/drla/scripts/audit_cola_request_more_latent_potential.py

artifact:
  /data1/luyifei/drla/outputs/cola_request_more_latent/
  p2e_request_more_latent_potential_locked_seed20260529_20260529

mode:
  local-only audit; swanlab_mode=disabled

split sizes:
  train / valid / test = 3998 / 495 / 487 grouped samples
```

Held-out test prefix/additional-evidence upper bound：

| policy / bound | exact acc | mean official score |
|---|---:|---:|
| first sender only | 22.38% | 0.3553 |
| prefix2 oracle | 30.39% | 0.4458 |
| prefix3 oracle | 34.70% | 0.4951 |
| prefix2 readiness selector | 23.61% | 0.3661 |
| prefix3 readiness selector | 22.38% | 0.3518 |

Marginal utility on held-out test：

```text
request first -> prefix2 helpful rate = 24.23%
request first -> prefix3 helpful rate = 33.47%
request prefix2 -> prefix3 helpful rate = 13.35%

mean first -> prefix2 score gain = +0.0905
mean first -> prefix3 score gain = +0.1398
mean prefix2 -> prefix3 score gain = +0.0493
```

Best valid-threshold practical policies observed in the audit：

| selection rule | test request rate | avg sender budget | helpful precision | oracle-after-request score | readiness-after-request score |
|---|---:|---:|---:|---:|---:|
| train task gain prior, target request rate 0.10/0.25 | 34.50% | 1.69 | 29.17% | 0.4232 | 0.3640 |
| contentful low, target request rate 0.50 | 52.36% | 2.05 | 32.55% | 0.4297 | 0.3604 |
| completion_risk high, target request rate 0.10 | 11.29% | 1.23 | 29.09% | 0.3707 | 0.3593 |
| readiness low, target helpful precision 0.50 | 1.03% | 1.02 | 60.00% | 0.3592 | 0.3571 |

Interpretation：

```text
Additional latent evidence has real offline value: the prefix3 oracle reaches
34.70% exact and 0.4951 mean score, far above first-sender and text-majority
baselines. This supports the request-more-latent direction.

However the current decoder-free practical selectors do not close much of that
gap. Simple readiness selection over additional senders falls back to about
first-sender quality, and the best thresholded request policies mostly improve
oracle-after-request upper bounds rather than practical readiness-after-request
quality.

This is not a text-superiority result. It is evidence that the next meaningful
experiment should train a learned request/additional-evidence policy or a
sequential aggregator, instead of continuing to rerank the same fixed three
senders with shallow scalar features.
```

Learned request-more-latent policy：

```text
script:
  /data1/luyifei/drla/drla/scripts/train_cola_request_more_policy.py

online input:
  first sender sanitized latent/process/certificate/task fields only

after-request practical aggregator:
  score-target latent fuser v2

training requirements:
  CUDA/GPU required
  SwanLab cloud required
  metrics.jsonl + best_checkpoint.pt + last_checkpoint.pt
```

Smoke：

```text
artifact:
  /data1/luyifei/drla/outputs/cola_request_more_policy/
  p2e_request_more_policy_smoke_256groups_seed20260529_20260529

SwanLab:
  zh936nbxnbrow88z7savu

status:
  passed pipeline/checkpoint/policy-table smoke
```

Full `fuser_gain` target：

```text
artifact:
  /data1/luyifei/drla/outputs/cola_request_more_policy/
  p2e_request_more_policy_fuser_gain_full_seed20260529_20260529

SwanLab:
  zn7zl11z11ghmfenr8wr4

best step:
  350

test helpful AUROC / gain corr:
  0.6823 / 0.0216
```

Full `oracle_gain` target：

```text
artifact:
  /data1/luyifei/drla/outputs/cola_request_more_policy/
  p2e_request_more_policy_oracle_gain_full_seed20260529_20260529

SwanLab:
  at0w7v8gsewja1vudb3jx

best step:
  350

test helpful AUROC / gain corr:
  0.6461 / 0.0879
```

Held-out test comparison：

| policy | request rate | avg sender budget | exact acc | mean official score |
|---|---:|---:|---:|---:|
| first sender only | 0.00% | 1.00 | 22.38% | 0.3553 |
| text majority | n/a | 3 text answers | 23.41% | 0.3622 |
| always request + fuser v2 | 100.00% | 3.00 | 23.41% | 0.3685 |
| `fuser_gain` best practical policy | 23.00% | 1.46 | 23.20% | 0.3689 |
| `oracle_gain` best practical policy | 25.67% | 1.51 | 22.79% | 0.3660 |
| always request oracle upper bound | 100.00% | 3.00 | 34.70% | 0.4951 |

Best practical `fuser_gain` policy：

```text
selection:
  target_request_rate = 0.25
  signal = gain_pred

test request rate:
  22.9979%

avg sender budget:
  1.45996

target helpful precision:
  32.14%

oracle-after-request:
  accuracy = 26.28%
  score = 0.4010

fuser-after-request:
  accuracy = 23.20%
  score = 0.3689
```

Interpretation：

```text
This is the first positive request-more-latent efficiency result: a learned
first-sender latent policy can identify a useful subset of request cases
better than random, and the best fuser_gain policy slightly exceeds always-on
fuser mean score while using about half the sender budget.

The result is deliberately framed as budget-efficiency evidence, not a strong
quality win. Exact accuracy is slightly below always-request/text-majority, and
the absolute score gain over always-request fuser is tiny.

The oracle gap remains large. Oracle-after-request reaches 0.4010 under the
best practical request subset and 0.4951 under always-request oracle, but the
current fuser-after-request path only reaches 0.3689. Therefore the next
scientific bottleneck is not request detection alone; it is request +
post-request aggregation/selection.
```

Post-request anchor-aware selector：

```text
script:
  /data1/luyifei/drla/drla/scripts/train_cola_post_request_selector.py

online input:
  first sender + requested sender sanitized latent/process/certificate/task fields

design:
  sender encoder from hierarchical fuser backbone
  anchor-aware candidate features:
    sender_state, first_sender_state, difference, product
  post-request Transformer over sender candidates
  heads:
    score_pred
    rank_logits
    gain_pred

loss:
  score MSE + listwise rank + pairwise rank + gain-over-first MSE

request-gated eval:
  use previous fuser_gain request policy
  target_request_rate = 0.25
  signal = gain_pred
```

Smoke：

```text
artifact:
  /data1/luyifei/drla/outputs/cola_post_request_selector/
  p2e_post_request_selector_anchor_smoke_256groups_seed20260529_20260529_rerun

SwanLab:
  r5jsobe8gh78m1ehuh818

status:
  passed pipeline/checkpoint/per-task/prediction smoke
```

Full variants：

| variant | SwanLab | best step | standalone acc | standalone score | request-gated acc | request-gated score |
|---|---|---:|---:|---:|---:|---:|
| anchor score selection | `r7qj2vu48vws5lnp5edxf` | 350 | 22.18% | 0.3572 | 23.00% | 0.3668 |
| anchor rank selection | `089s7vnjawqpd4p7m20ak` | 200 | 21.77% | 0.3500 | 23.41% | 0.3697 |

References on the same held-out test：

| reference | acc | score |
|---|---:|---:|
| first sender only | 22.38% | 0.3553 |
| text majority | 23.41% | 0.3622 |
| always request + fuser v2 | 23.41% | 0.3685 |
| request policy + fuser v2 | 23.20% | 0.3689 |
| request policy + anchor rank selector | 23.41% | 0.3697 |
| oracle upper bound | 34.70% | 0.4951 |

Interpretation：

```text
The standalone anchor-aware selector is negative: when it must always choose
among all senders, both score and rank variants trail the original fuser v2.

The request-gated anchor-rank variant is a narrow positive budget-efficiency
result. It keeps the request rate at 22.9979%, matches text/fuser exact
accuracy at 23.41%, and slightly improves mean score over request+fuser v2
and always-request fuser v2.

This still does not close the oracle gap. The result suggests that post-request
ranking is useful only inside the restricted request subset; globally, the
selector overfits or misranks many always-request cases. The next branch should
couple request and post-request selection more tightly, instead of training the
two modules independently.
```

Joint request-and-select policy：

```text
script:
  /data1/luyifei/drla/drla/scripts/train_cola_joint_request_select_policy.py

online stages:
  request head:
    sees only the first sender packet
  selector head:
    sees first + requested sender packets only after request

training objective:
  request BCE on oracle helpfulness
  request gain regression
  selector score MSE
  selector listwise ranking
  selector pairwise ranking
  selector gain-over-first regression
  differentiable budgeted expected utility

checkpoint selection:
  valid target_request_rate=0.25 budgeted score during training

policy table:
  valid-selected thresholds reported on held-out test
```

Smoke：

```text
artifact:
  /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
  p2e_joint_request_select_smoke_256groups_seed20260529_20260529

SwanLab:
  munwuikpa4v8m8g9nysqf

status:
  passed pipeline/checkpoint/policy-table/per-task/prediction smoke
```

Full variants：

| variant | SwanLab | best step | always-request acc | always-request score | best gated request rate | best gated budget | best gated acc | best gated score |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| joint rank selection | `g126cuz3w32r9g76jizcs` | 250 | 23.41% | 0.3660 | 47.64% | 1.95 | 23.82% | 0.3733 |
| joint score selection | `2t0j6t9v33qfkfn71w2mi` | 250 | 21.77% | 0.3473 | 47.64% | 1.95 | 24.02% | 0.3735 |

Same held-out test references：

| reference | request rate | budget | acc | score |
|---|---:|---:|---:|---:|
| first sender only | 0.00% | 1.00 | 22.38% | 0.3553 |
| text majority | n/a | 3 text answers | 23.41% | 0.3622 |
| always request + fuser v2 | 100.00% | 3.00 | 23.41% | 0.3685 |
| request policy + fuser v2 | 23.00% | 1.46 | 23.20% | 0.3689 |
| request policy + anchor-rank selector | 23.00% | 1.46 | 23.41% | 0.3697 |
| joint score, best gated | 47.64% | 1.95 | 24.02% | 0.3735 |
| oracle upper bound | 100.00% | 3.00 | 34.70% | 0.4951 |

Best joint score row：

```text
selection:
  target_request_rate = 0.50
  signal = request_prob

held-out test:
  request rate = 47.64%
  average sender budget = 1.9528
  model helpful precision = 35.34%
  model-after-request accuracy / score = 24.02% / 0.3735
  oracle-after-request accuracy / score = 26.49% / 0.4137
```

Joint policy calibration / risk-control audit：

```text
script:
  /data1/luyifei/drla/drla/scripts/audit_cola_joint_policy_calibration.py

artifact:
  /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
  p2e_joint_request_select_score_full_seed20260529_20260529_calibration_risk

mode:
  local-only, swanlab_mode=disabled
  no optimizer/backward
  valid thresholds only; held-out test is report-only

split:
  valid = 495 groups
  test = 487 groups
```

Calibration readout：

```text
request_prob -> model_request_helpful:
  valid AUROC / ECE = 0.6959 / 0.2363
  test  AUROC / ECE = 0.7164 / 0.2226
  test prob mean / target mean = 0.4464 / 0.2238

request_prob -> oracle_request_helpful:
  valid AUROC / ECE = 0.7089 / 0.1050
  test  AUROC / ECE = 0.6825 / 0.1132

request_gain_pred:
  test corr with model gain = 0.0010
  test corr with oracle gain = 0.1315
```

Risk-control readout：

```text
best utility row remains target_request_rate=0.50 on request_prob:
  test request rate / budget = 47.64% / 1.9528
  test accuracy / score = 24.02% / 0.3735
  requested model-loss rate = 28.88%
  requested model-loss Wilson95 upper = 35.02%

strict conditional loss-risk caps:
  target upper 0.10 / 0.20: no non-trivial valid threshold
  target upper 0.30 / 0.40: selects almost-always request and drops test score to 0.3494
```

Interpretation：

```text
This is the strongest P2-E budgeted latent communication result so far. Joint
request-and-select beats text majority and always-request fuser v2 in mean
score, and it slightly improves exact accuracy over both while using less than
two sender packets on average.

The result is still modest and not a broad claim that latent communication is
universally better than text. The oracle upper bound remains much higher
at 34.70% / 0.4951, so the bottleneck is now calibrated joint budgeted utility
and better exploitation of requested packets.

The standalone always-request score-selection model is poor, but its gated
policy is best. This reinforces the core lesson: the selector should be judged
inside the request distribution, not as a global reranker.

Calibration adds an important qualifier: request_prob is a useful ranking
signal, but it is over-confident as a probability and current conditional
loss-risk control does not certify the high-scoring threshold. The current
0.50 request-rate policy should therefore be described as valid-selected
budgeted utility, not as a formally risk-certified communication policy.
```

Fresh seed / fresh split robustness, seed30：

```text
overlap audit:
  seed30 test groups = 495
  overlap with seed29 split:
    seed29 train = 406
    seed29 valid = 51
    seed29 test = 38

contaminated diagnostic:
  /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
  p2e_joint_request_select_score_full_seed20260530_20260529

reason it is non-canonical:
  it reused seed29 fuser checkpoint and seed29 fuser norm_stats, so most of its
  seed30 test groups were seen by the referenced fuser/norm pipeline.
```

Strict seed30 fuser baseline：

```text
training artifact:
  /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
  p2e_hierarchical_fuser_score_full_seed20260530_20260529

SwanLab:
  qweuypbg1ugls3io0s9j0

best-checkpoint local eval:
  /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
  p2e_hierarchical_fuser_score_full_seed20260530_20260529_best_eval_test

test:
  fuser = 22.63%, score 0.3694
  first sender = 22.42%, score 0.3669
  text majority = 22.63%, score 0.3642
  oracle = 35.96%, score 0.5071
```

Strict seed30 joint request-select：

```text
training artifact:
  /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
  p2e_joint_request_select_score_full_seed20260530_fuserseed20260530_20260529

SwanLab:
  n4wu1f4ghzfwe6mhvltei

calibration artifact:
  /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
  p2e_joint_request_select_score_full_seed20260530_fuserseed20260530_20260529_calibration_risk

best policy:
  selection_mode = target_request_rate
  target_value = 0.50
  signal = request_gain_pred

held-out test:
  request rate / budget = 50.51% / 2.0101
  joint acc / score = 23.64% / 0.3796
  text acc / score = 22.63% / 0.3642
  fuser acc / score = 22.63% / 0.3694
  score gain vs text / fuser = +0.0153 / +0.0102
  requested model-loss Wilson95 upper = 25.40%
```

Fresh-seed interpretation：

```text
The strict seed30 rerun preserves the positive direction: joint latent
request-select beats text majority and the same-split fuser best checkpoint in
mean score and exact accuracy.

The result also shows why the overlap audit matters. The first seed30 run that
reused seed29 fuser/norm_stats produced a misleading fuser baseline because
406/495 fresh test groups were in the seed29 train split.

Calibration remains the weak point. On strict seed30, request_prob has only
test AUROC 0.6255 and ECE 0.2697 against model_request_helpful; the best
utility policy uses request_gain_pred instead, but the conditional loss upper
is still 25.40%. This is a replicated utility win, not a certified safe policy.

Checkpoint-selection note:
  strict seed30 final checkpoint reports always-request 24.04% / 0.3852, but
  the official row remains the best checkpoint selected by valid target025
  protocol. This suggests the selection metric should be audited, not that we
  should claim the final checkpoint result as canonical.
```

Strict seed31 replication：

```text
same-split fuser:
  artifact:
    /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
    p2e_hierarchical_fuser_score_full_seed20260531_20260529
  SwanLab:
    64h605uhcjpse62n84l8v
  best-checkpoint eval:
    /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
    p2e_hierarchical_fuser_score_full_seed20260531_20260529_best_eval_test

same-split joint:
  artifact:
    /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
    p2e_joint_request_select_score_full_seed20260531_fuserseed20260531_20260529
  SwanLab:
    xh19j75yervdr7l603rt4
  calibration:
    /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
    p2e_joint_request_select_score_full_seed20260531_fuserseed20260531_20260529_calibration_risk
```

Strict seed31 held-out test：

```text
first sender = 16.96%, score 0.3074
text majority = 17.75%, score 0.3180
same-split fuser best = 17.75%, score 0.3184
joint best selected = 19.33%, score 0.3426
oracle upper bound = 29.78%, score 0.4515

best selected policy:
  selection_mode = target_loss_wilson_upper
  target_value = 0.30
  signal = request_prob
  request rate / budget = 99.80% / 2.9961
  score gain vs text / fuser = +0.0245 / +0.0242
  requested model-loss Wilson95 upper = 26.99%
```

Three-strict-seed aggregate：

```text
artifact:
  /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
  p2e_joint_request_select_strict_seed29_30_31_summary_20260529

strict seeds:
  seed29, seed30, seed31

macro mean:
  first sender = 20.59%, score 0.3432
  text majority = 21.26%, score 0.3481
  same-split fuser = 21.26%, score 0.3521
  joint selected = 22.33%, score 0.3652
  oracle = 33.48%, score 0.4846
  joint request rate / budget = 65.98% / 2.3196
  requested model-loss Wilson95 upper = 29.13%
  request_prob model-helpful AUROC / ECE = 0.6688 / 0.2571
```

Fresh-seed aggregate interpretation：

```text
Across three strict seeds, joint latent request-select preserves a positive
macro utility direction over text majority and same-split fuser. The gain is
stronger in official score than exact accuracy.

Budget efficiency is not stable yet. Seed29/seed30 are budgeted at about half
request rate, but seed31's selected best utility policy is almost always
request. Calibration remains weak across seeds, so this is replicated utility
evidence, not a formal risk-certified or consistently budget-efficient policy.

Checkpoint-selection audit:
  local evaluator:
    /data1/luyifei/drla/drla/scripts/eval_cola_joint_request_select_policy.py

  seed29 last checkpoint best policy = 23.00% / 0.3653
  seed30 last checkpoint best policy = 24.85% / 0.3919
  seed31 last checkpoint best policy = 17.75% / 0.3252

The last-checkpoint audit is mixed: seed30 last is better than the selected
best checkpoint, but seed29/seed31 last are worse. This confirms the need for
a better checkpoint-selection metric rather than switching to last checkpoint.
```

Valid-frontier checkpoint selection audit, 2026-05-31：

```text
purpose:
  test whether selecting checkpoints by a valid utility frontier fixes the
  target025 checkpoint-selection sensitivity without leaking test information

trainer support:
  /data1/luyifei/drla/drla/scripts/train_cola_joint_request_select_policy.py
  --checkpoint-selection-mode valid_rate_frontier
  --checkpoint-selection-request-rates 0.10,0.25,0.50,0.75

local evaluator / audit:
  /data1/luyifei/drla/drla/scripts/eval_cola_joint_request_select_policy.py
  /data1/luyifei/drla/drla/scripts/audit_cola_joint_policy_calibration.py

strict frontier artifacts:
  seed29:
    /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
    p2e_joint_request_select_score_frontier_full_seed20260529_fuserseed20260529_20260529
    SwanLab = r0saruhorycaaf2sqskbs
  seed30:
    /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
    p2e_joint_request_select_score_frontier_full_seed20260530_fuserseed20260530_20260529
    SwanLab = d1dnxi4ojqb9v4quro5zw
  seed31:
    /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
    p2e_joint_request_select_score_frontier_full_seed20260531_fuserseed20260531_20260529
    SwanLab = 5eou0eit48tdhoac6hkwy

aggregate:
  /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
  p2e_joint_request_select_frontier_strict_seed29_30_31_summary_20260531
```

Leak-safe canonical selection rule：

```text
For the frontier aggregate, each seed chooses the policy row by maximum
valid_model_after_request_score among valid-selected rows, then reports held-out
test metrics. Test-best rows are recorded only as diagnostics and must not be
used as canonical results.
```

Frontier macro readout：

```text
old target025 strict aggregate:
  joint selected = 22.33%, score 0.3652
  request rate / budget = 65.98% / 2.3196
  loss Wilson95 upper = 29.13%

valid-frontier canonical aggregate:
  selected = 21.46%, score 0.3505
  request rate / budget = 49.73% / 1.9946
  loss Wilson95 upper = 30.69%
  score delta vs old joint = -0.0147
  acc delta vs old joint = -0.87 pp

test-best diagnostic aggregate, non-canonical:
  selected = 22.20%, score 0.3584
  request rate / budget = 59.71% / 2.1942
  score delta vs old joint = -0.0069
```

Frontier interpretation：

```text
Valid-frontier checkpoint selection is complete for strict seed29/30/31 and is
not an improvement over the current canonical target025 strict aggregate.
It lowers request budget but loses more score than it saves, and calibration is
still weak. Do not replace the current canonical P2-E joint request-select
result with frontier. Treat frontier as a useful negative checkpoint-selection
audit and stop spending more runs on shallow frontier/threshold tuning unless a
new model objective or calibration mechanism is introduced.
```

## 2026-06-06 Phase A Full-Info Repair Update

```text
environment issue:
  nvidia-smi currently reports NVML driver/library mismatch, but PyTorch CUDA
  still sees 8 RTX5090 devices. Short allocator tests pass with
  PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync. Use this allocator setting
  for long CoLA train/eval until the machine NVML stack is repaired.

10k length audit:
  source SFT:
    outputs/p2_phase_a_cola_interface_sft/
    musique_interface_train_fullinfo_10000_seed20260606_20260605

  solver_full_info train examples:
    n = 9000
    token median/p90/p95/p99/max = 782 / 1134 / 1262 / 1549 / 2237
    latent block median/p90/p95/p99/max = 49 / 71 / 79 / 97 / 140

  train2000 reference max block length = 98
  calibration max block length is within the cap used below

repair run:
  outputs/p2_phase_a_cola_dit_lora/
  musique_solver_fullinfo_lora_train10000_cap96_from_train2000_step2000_lr1e5_interval_seed20260606_20260606

  SwanLab:
    y92wrmkp38hcuc8x4y5h0

  config:
    init_lora_path = train2000 solver best_adapter
    roles = solver_full_info
    lr = 1e-5
    max_train_steps = 2000
    max_total_blocks = 96
    save_interval_checkpoints = true
    valid_interval = 100
    max_valid_batches = 50
    PYTORCH_CUDA_ALLOC_CONF = backend:cudaMallocAsync

  dataset after cap96:
    train_pairs = 8902
    valid_pairs = 988
    train_skipped_over_max_total_blocks = 98
    valid_skipped_over_max_total_blocks = 12

  training status:
    pass
    best_step = 900
    best_valid_loss = 0.04698952
    final_valid_loss = 0.05179715
    interval adapters saved at each validation step
```

Non-heldout SFT-valid diagnostic:

```text
screen:
  outputs/p2_phase_c_text_agent_aggregates/
  validdiag50_solver_candidate_screen_20260606

split:
  SFT valid split only, 50 samples
  conditions = single_full_info, single_q_only
  no held-out rows

readout:
  cap96_step900:   single_full_info 0.10, single_q_only 0.02, paired diff +0.08
  cap96_step1300:  single_full_info 0.08, single_q_only 0.00, paired diff +0.08
  cap96_last:      single_full_info 0.08, single_q_only 0.00, paired diff +0.08
  base_train2000:  single_full_info 0.02, single_q_only 0.00, paired diff +0.02
```

Calibration solver100 precheck:

```text
aggregate:
  outputs/p2_phase_c_text_agent_aggregates/
  calibration_solver100_cap96_candidate_precheck_20260606

split:
  locked calibration, 100 samples
  conditions = single_full_info, single_q_only
  no Role TextMAS gate, no held-out, no SwanLab

candidate results:
  cap96_last:
    single_full_info primary = 0.08
    single_q_only primary = 0.01
    paired diff = +0.07
    paired bootstrap 95% CI = [+0.02, +0.13]

  cap96_step1300:
    single_full_info primary = 0.06
    single_q_only primary = 0.01
    paired diff = +0.05
    paired bootstrap 95% CI = [0.00, +0.10]

  cap96_step900:
    single_full_info primary = 0.05
    single_q_only primary = 0.01
    paired diff = +0.04
    paired bootstrap 95% CI = [0.00, +0.09]

gate:
  failed_precheck_no_candidate_reaches_full_info_floor
  floor = single_full_info primary >= 0.20
```

Interpretation:

```text
The 10k cap96 run fixes the training stability problem and produces much lower
Flow-Matching valid loss, but it does not recover downstream QA capability on
locked calibration. The best calibration solver100 full-info score is only
0.08, below both the locked 0.20 floor and the earlier train2000 calibration
full gate score of 0.13.

This confirms the earlier warning: Flow-Matching valid loss and nonheldout
SFT-valid diagnostics are not enough to select a useful downstream QA adapter.
The cap96 continuation must not enter full Role TextMAS gate, held-out, or
Phase E TextMAS-vs-LatentMAS comparison.
```

Objective-mismatch diagnostic:

```text
artifact:
  outputs/p2_phase_a_diagnostics/
  fullinfo_objective_mismatch_20260606

compared runs:
  cap96_last_solver100
  cap96_step1300_solver100
  cap96_step900_solver100
  train2000_dual_full200

cap96_last single_full_info taxonomy:
  primary_acc = 0.080
  token_f1_mean = 0.181
  wrong_primary = 0.920
  eos_tail_after_stop = 0.780
  not_copied_from_evidence = 0.530
  wrong_or_extra_support_entity = 0.310
  partial_gold_overlap = 0.240
  overlong_prediction = 0.210
  distractor_copy = 0.110
  high_f1_but_not_primary = 0.110

train2000_dual_full200 single_full_info taxonomy:
  primary_acc = 0.130
  token_f1_mean = 0.264
  wrong_primary = 0.870
  eos_tail_after_stop = 0.740
  partial_gold_overlap = 0.375
  overlong_prediction = 0.290

SFT length audit for 10k solver_full_info:
  target_tokens median/p90/p95/p99/max = 2 / 4 / 6 / 9 / 15
  context_tokens median/p90/p95/p99/max = 533 / 791 / 876 / 1049 / 1438
  prompt_tokens median/p90/p95/p99/max = 551 / 814 / 899 / 1076 / 1471
  answer_string_in_context_rate = 1.000
```

Diagnostic interpretation:

```text
Most cap96 failures are answer-shaped, not empty/parser failures. The adapter
often selects the wrong support/distractor entity, emits the right answer plus
extra tokens, or produces plausible evidence-adjacent spans. Raw EOS-tail
artifacts are common, but first_segment parsing already strips them, so they
are not the dominant scoring failure.

The central mismatch is that the supervised target is a very short final
answer, while the prompt/context is hundreds of tokens long and contains many
answer-like spans. Full-sequence Flow-Matching loss can improve without
learning the final-answer extraction policy needed by the scorer.
```

Support-only diagnostic and curriculum attempt:

```text
new diagnostic control builder:
  drla/scripts/build_p2_phase_c_support_only_solver_controls.py

support-only calibration controls:
  outputs/p2_phase_c_control_inputs/
  musique_calibration_solver_support_only_diag_200_seed20260606
  rows = 400
  samples = 200
  conditions = single_full_info, single_q_only
  diagnostic_context_mode = support_only_solver_single_full_info
  boundary = local-only diagnostic, not locked gate

pre-curriculum support-only eval:
  aggregate:
    outputs/p2_phase_c_text_agent_aggregates/
    calibration_solver100_support_only_diag_cap96_last_20260606
  merged run:
    outputs/p2_phase_c_text_agent_runs/
    calibration_solver100_support_only_diag_cap96_last_merged_20260606
  cap96_last support-only single_full_info = 0.16
  cap96_last support-only single_q_only = 0.01
  paired diff = +0.15
  paired bootstrap 95% CI = [+0.08, +0.23]

SFT support-only artifact:
  outputs/p2_phase_a_cola_interface_sft/
  musique_interface_train_support_only_10000_seed20260606_20260606
  role_counts: solver_support_only = 10000
  split: train 9000, valid 1000
  no teacher-role pairs

support-only SFT length audit:
  prompt_tokens train median/p90/p95/p99/max = 210 / 396 / 465 / 585 / 867
  context_tokens train median/p90/p95/p99/max = 192 / 372 / 436 / 560 / 828
  target_tokens train median/p90/p95/p99/max = 2 / 4 / 6 / 9 / 15

support-only curriculum training:
  outputs/p2_phase_a_cola_dit_lora/
  musique_solver_support_only_lora_train10000_from_cap96_last_step1000_lr1e5_interval_seed20260606_20260606
  SwanLab: g4yipez7s8mk16i2y88lf
  init_lora_path: cap96 last_adapter
  roles: solver_support_only
  max_train_steps: 1000
  valid_interval: 100
  save_interval_checkpoints: true
  best_step: 500
  best_valid_loss: 0.1216601644147886
```

Support-only curriculum solver100 comparison:

```text
aggregate:
  outputs/p2_phase_c_text_agent_aggregates/
  support_only_curriculum_solver100_comparison_20260606

candidates:
  support_curriculum_best_step500:
    support-only single_full_info = 0.17
    single_q_only = 0.01
    paired diff = +0.16
    paired bootstrap 95% CI = [+0.08, +0.24]
    support-only floor 0.20: fail

  support_curriculum_last_step1000:
    support-only single_full_info = 0.17
    single_q_only = 0.01
    paired diff = +0.16
    paired bootstrap 95% CI = [+0.08, +0.24]
    support-only floor 0.20: fail

  cap96_last_pre_curriculum:
    support-only single_full_info = 0.16
    single_q_only = 0.01
    paired diff = +0.15
    paired bootstrap 95% CI = [+0.08, +0.23]
```

Support-only interpretation:

```text
Removing distractors improves cap96_last from locked full-evidence 0.08 to
support-only 0.16, so distractor/evidence selection is a real failure factor.
However, the 1000-step support-only curriculum improves only to 0.17 and still
does not pass even the diagnostic 0.20 floor. This path does not justify
Role TextMAS, held-out, or Phase E.

The next repair should not be "more of the same" support-only Flow-Matching.
It needs a stronger answer-selection objective or auxiliary signal that teaches
which support span answers the question, not merely a shorter evidence prompt.
```

Answer-support target attempt:

```text
builder update:
  drla/scripts/build_p2_phase_a_cola_interface_sft.py
  --solver-target-mode final_answer_then_support

target shape:
  line 1: Final answer: <gold answer>
  line 2: Selected support: <answer-bearing support line first, then compact support>

input boundary:
  prompt still contains ordinary full_evidence
  no gold span marker is inserted into the online prompt
  gold/support are used only as non-heldout SFT targets

SFT artifact:
  outputs/p2_phase_a_cola_interface_sft/
  musique_interface_train_answer_support_10000_seed20260606_20260606
  role_counts: solver_full_info = 10000
  train/valid = 9000 / 1000
  first_line_bad = 0
  answer_in_selected_support_rate = 0.9796
  target_tokens train median/p90/p95/p99/max = 126 / 189 / 219 / 296 / 408

training:
  outputs/p2_phase_a_cola_dit_lora/
  musique_solver_answer_support_lora_train10000_from_cap96_last_step1000_lr1e5_cap96_interval_seed20260606_20260606
  SwanLab: 8rtj42x4ey9zadeu8y936
  init_lora_path: cap96 last_adapter
  max_total_blocks = 96
  train_pairs = 8592
  valid_pairs = 949
  train_skipped_over_max_total_blocks = 408
  valid_skipped_over_max_total_blocks = 51
  best_step = 1000
  best_valid_loss = 0.2581918811961077
```

Full-evidence solver100 screen:

```text
aggregate:
  outputs/p2_phase_c_text_agent_aggregates/
  answer_support_best_fullevidence_solver100_20260606

merged run:
  outputs/p2_phase_c_text_agent_runs/
  answer_support_best_fullevidence_solver100_merged_20260606

readout:
  single_full_info primary = 0.05
  single_q_only primary = 0.01
  paired diff = +0.04
  paired bootstrap 95% CI = [0.00, +0.09]
  floor = single_full_info >= 0.20
  status = fail_solver100_full_info_floor
```

Answer-support interpretation:

```text
The target construction passed its data audit, but downstream full-evidence QA
got worse than cap96_last (0.05 vs 0.08). Raw outputs show the model often emits
support-style long spans or entity lists in the first segment, so the objective
pulls generation toward evidence reproduction instead of short-answer
selection.

Do not run Role TextMAS, held-out, or Phase E from this answer-support
checkpoint. Do not continue the same "Final answer + Selected support" target
without a mechanism that keeps the first segment short-answer disciplined.
```

Candidate-answer selection diagnostic:

```text
candidate builder:
  drla/scripts/build_p2_phase_a_candidate_answer_sets.py

selector trainer:
  drla/scripts/train_p2_phase_a_candidate_answer_selector.py

boundary:
  local-only sklearn diagnostic
  no deep-learning optimizer/backward
  no model generation
  no SwanLab run
  no held-out data
  gold/aliases used only as train/eval labels and coverage audit
```

Candidate coverage:

```text
train top128:
  outputs/p2_phase_a_candidate_answers/
  musique_train_candidate_answers_10000_seed20260606_20260606
  oracle coverage kept = 0.8012

calibration top128:
  outputs/p2_phase_a_candidate_answers/
  musique_calibration_candidate_answers_200_seed20260606_20260606
  oracle coverage kept = 0.7150

train top256:
  outputs/p2_phase_a_candidate_answers/
  musique_train_candidate_answers_10000_seed20260606_top256_20260606
  oracle coverage kept = 0.8182

calibration top256:
  outputs/p2_phase_a_candidate_answers/
  musique_calibration_candidate_answers_200_seed20260606_top256_20260606
  oracle coverage kept = 0.7150
```

Selector results on calibration200:

```text
logistic basic top128:
  outputs/p2_phase_a_candidate_selectors/
  musique_candidate_selector_train10000_eval_calib200_20260606
  selected_primary = 0.070
  candidate_auc = 0.8555

logistic qtype top128:
  outputs/p2_phase_a_candidate_selectors/
  musique_candidate_selector_qtype_train10000_eval_calib200_20260606
  selected_primary = 0.175
  selected_given_covered = 0.2448
  candidate_auc = 0.8863

logistic qtype top256:
  outputs/p2_phase_a_candidate_selectors/
  musique_candidate_selector_qtype_top256_train10000_eval_calib200_20260606
  selected_primary = 0.170
  selected_given_covered = 0.2378
  candidate_auc = 0.9303

hist_gbdt top128:
  outputs/p2_phase_a_candidate_selectors/
  musique_candidate_selector_histgbdt_train10000_eval_calib200_20260606
  selected_primary = 0.135
  selected_given_covered = 0.1888
  candidate_auc = 0.9118

hist_gbdt top256:
  outputs/p2_phase_a_candidate_selectors/
  musique_candidate_selector_histgbdt_top256_train10000_eval_calib200_20260606
  selected_primary = 0.115
  selected_given_covered = 0.1608
  candidate_auc = 0.9403
```

Candidate-selector interpretation:

```text
The best shallow selector is qtype logistic top128 at 0.175 primary. It is
close to, but still below, the locked solver100 precheck floor of 0.20.

The oracle kept coverage is much higher than selected accuracy, so the problem
is not only candidate extraction. The current rule-derived candidates contain
useful signal, but shallow feature-based ranking cannot reliably select the
answer from distractor/entity-rich MuSiQue contexts.

Increasing candidates from top128 to top256 does not improve calibration
coverage or selected accuracy. HistGradientBoosting improves AUC but worsens
top-1 selected accuracy, which means pairwise candidate separability is not
enough for the downstream top-1 solver objective.

Do not continue by simply adding more rules, larger candidate pools, or more
of the same shallow selector. The next repair needs a stronger semantic
answer-selection/reranking signal or a CoLA-native constrained interface that
keeps the scored first segment short-answer disciplined.
```

Qwen semantic candidate-selector diagnostic:

```text
script:
  drla/scripts/run_p2_phase_a_candidate_answer_llm_selector.py

model:
  local Qwen3-8B-FP8

input:
  calibration candidate top128
  full online evidence from calibration manifest
  question
  no gold or scorer fields as online input

boundary:
  local-only LLM inference diagnostic
  no deep-learning optimizer/backward
  no SwanLab run
  no held-out data
  gold/aliases used only for offline scoring

smoke20:
  outputs/p2_phase_a_candidate_selectors/
  musique_candidate_selector_qwen3_8b_fp8_calib20_top128_20260606
  selected_primary = 0.600
  oracle_coverage_kept = 0.950

full calibration200:
  outputs/p2_phase_a_candidate_selectors/
  musique_candidate_selector_qwen3_8b_fp8_calib200_top128_20260606
  selected_primary = 0.445
  selected_exact_match = 0.395
  selected_token_f1 = 0.4485
  oracle_coverage_kept = 0.715
  selected_given_covered = 0.6154
```

Qwen selector interpretation:

```text
This is a positive diagnostic for the candidate protocol. With the same top128
candidate pool that shallow selectors fail to rank, a capable semantic model
reaches 0.445 primary on calibration200, roughly at the same scale as the
previous Qwen full-info/text-MAS capable-agent calibration numbers.

The split is informative:
  gold covered by kept candidates: 143 / 200
  primary when covered: 0.6154
  primary when not covered: 0.0175

Therefore the candidate protocol is semantically usable, but requires a
stronger semantic selection signal. The next repair should use this as teacher
evidence for nonheldout short-answer selection/distillation, rather than
continuing shallow rerankers or long support-generating targets.
```

Train-source semantic teacher full10k:

```text
script update:
  drla/scripts/run_p2_phase_a_candidate_answer_llm_selector.py
  added deterministic --num-shards / --shard-index support
  added --progress-interval for long local inference runs

aggregate script:
  drla/scripts/aggregate_p2_phase_a_candidate_answer_llm_selector.py

completed shards:
  outputs/p2_phase_a_candidate_selectors/
  musique_candidate_selector_qwen3_8b_fp8_train10000_top128_shard00of10_20260606
  ...
  musique_candidate_selector_qwen3_8b_fp8_train10000_top128_shard09of10_20260606

aggregate:
  outputs/p2_phase_a_candidate_selectors/
  musique_candidate_selector_qwen3_8b_fp8_train10000_top128_all10_aggregate_20260606

input:
  train-source MuSiQue manifest = 10000 samples
  top128 evidence-derived candidates
  num_shards = 10
  scheduled rows = 10000
  unique sample ids after aggregate = 10000

metrics:
  selected_primary = 0.4336
  selected_exact_match = 0.3893
  selected_token_f1 = 0.4941
  oracle_coverage_kept = 0.8012
  selected_given_covered = 0.4908
  candidate_exact_or_high_f1_rate = 0.8634
  num_oracle_covered = 8012 / 10000

boundary:
  local-only Qwen inference
  no training / no optimizer / no SwanLab
  no held-out data
  gold used only for offline score attached to teacher artifacts
```

Train-source teacher interpretation:

```text
The full 10000-sample train-source semantic teacher is now complete and stable
relative to calibration (0.4336 vs 0.445 primary). It confirms that the
candidate interface is usable when the selector has semantic capacity, while
the remaining gap to the oracle candidate ceiling is mostly answer-selection
and answer-coverage pressure.

This artifact is allowed as nonheldout teacher evidence for building a
constrained short-answer CoLA target or CoLA-native answer selector. It is not
a CoLA result, not a Phase E communication result, and not held-out evidence.
```

Candidate-constrained short-answer CoLA SFT artifact:

```text
script:
  drla/scripts/build_p2_phase_a_candidate_constrained_sft.py

output:
  outputs/p2_phase_a_cola_interface_sft/
  musique_candidate_constrained_short_answer_train10000_top128_qwen_teacher_20260606

input:
  nonheldout MuSiQue train manifest = 10000 samples
  evidence-derived candidate answers top128
  completed Qwen semantic selector train10k teacher aggregate

online prompt boundary:
  question
  full online evidence
  candidate text/rank/rule/source metadata
  no gold label
  no alias flag
  no teacher correctness
  no scorer output
  no held-out data

roles:
  solver_candidate_gold_covered = 8012 pairs
  solver_candidate_teacher_correct = 4336 pairs
  total = 12348 pairs
  train = 11133 pairs
  valid = 1215 pairs

CoLA tokenizer/block audit:
  block_size = 16
  token mean = 4030.18
  token p95 = 4577
  token max = 5833
  block p95 = 287
  block max = 365

boundary:
  local-only data preparation
  no model generation
  no optimizer/backward
  no SwanLab
  no held-out data
```

Candidate-constrained CoLA training feasibility:

```text
top128 full prompt smoke:
  output:
    outputs/p2_phase_a_cola_dit_lora/
    musique_candidate_constrained_short_answer_both_top128_smoke10_seed20260606_20260606
  SwanLab:
    z81yej0pzf2guim4r9bxk
  result:
    OOM on first DiT forward

top64 cap224 smoke:
  SwanLab:
    rzvd8bvdej7hl14gpvru0
  result:
    OOM on first DiT forward

top64 cap160 smoke:
  SwanLab:
    wlro4n428qe5c3j7szjic
  result:
    OOM on DiT forward

top32 cap128 smoke:
  SwanLab:
    ht3wj04pzawndh6wubo7v
  result:
    forward passes but backward OOM

top32 cap112 smoke:
  output:
    outputs/p2_phase_a_cola_dit_lora/
    musique_candidate_constrained_short_answer_both_top32_cap112_smoke10_seed20260606_20260606
  SwanLab:
    plxdvnjhtk86fkf60zvp8
  result:
    pass
    best_checkpoint.pt and last_checkpoint.pt written
    train_pairs after cap = 6793
    valid_pairs after cap = 736
    train_block_examples = 9476
    valid_block_examples = 1017

decision:
  top32 + max_total_blocks=112 is the first feasible candidate-constrained
  short-answer CoLA training branch. Top64/top128 are currently too long for
  the slow-attention CoLA DiT training path without evidence compression or
  attention/memory engineering.

training logging policy update:
  The completed step900 run used valid_interval=100 and is retained as a
  historical artifact. For subsequent CoLA LoRA training, use valid_interval=10
  so SwanLab and local metrics show 10:1 train/valid dynamics. Save both
  best_checkpoint.pt / best_adapter and last_checkpoint.pt / last_adapter; use
  best_adapter for downstream solver screens unless explicitly comparing last.
```

Candidate-constrained top32-cap112 training and solver100 screen:

```text
training output:
  outputs/p2_phase_a_cola_dit_lora/
  musique_candidate_constrained_short_answer_both_top32_cap112_step900_seed20260606_20260606

SwanLab:
  imprfr8cdudi791eer012

training config:
  sft = musique_candidate_constrained_short_answer_train10000_top32_qwen_teacher_20260606
  roles = solver_candidate_gold_covered, solver_candidate_teacher_correct
  max_total_blocks = 112
  max_train_steps = 900
  valid_interval = 100
  save_interval_checkpoints = true
  note: this is a historical completed run before the policy update to
        valid_interval <= 10

training result:
  status = pass
  best_step = 700
  best_valid_loss = 0.211184
  final_valid_loss = 0.217415
  train_pairs_after_cap = 6793
  valid_pairs_after_cap = 736
  train_skipped_over_cap = 1340
  valid_skipped_over_cap = 159
  saved:
    best_checkpoint.pt / best_adapter
    last_checkpoint.pt / last_adapter
    valid_step_100 ... valid_step_900 interval checkpoints

solver100 screen:
  aggregate:
    outputs/p2_phase_c_text_agent_aggregates/
    candidate_constrained_top32_cap112_best_solver100_20260606
  merged generations:
    outputs/p2_phase_c_text_agent_runs/
    candidate_constrained_top32_cap112_best_solver100_merged_20260606
  adapter:
    step900 run best_adapter
  conditions:
    single_full_info, single_q_only
  calibration samples:
    100 paired samples / 200 rows

solver100 metrics:
  single_full_info primary = 0.01
  single_full_info exact = 0.01
  single_full_info token_f1 = 0.0345
  single_q_only primary = 0.00
  paired diff = +0.01
  paired bootstrap 95% CI = [0.00, 0.03]
  parseable_rate = 1.00

decision:
  fail_solver100_full_info_floor
  floor = single_full_info primary >= 0.20
  Do not run Role TextMAS gate, held-out, or Phase E from this adapter.

interpretation:
  The short-answer candidate objective is learnable in Flow-Matching loss, but
  it does not transfer to real CoLA solver generation under the locked
  squad_template_v1 first-segment evaluator. The likely issue is not logging or
  checkpoint selection; it is objective/interface mismatch under the current
  CoLA generation path.
```

Candidate-prompt payload repair and answer-only target follow-up:

```text
bug fix:
  run_p2_phase_c_text_agents.py now passes candidate_answers through
  make_solver_messages and supports cola_prompt_style=candidate_constrained_v1.
  The earlier candidate-prompt smoke before this fix was not a valid candidate
  interface test because candidates were dropped from the solver payload.

fixed SFT artifact:
  outputs/p2_phase_a_cola_interface_sft/
  musique_candidate_constrained_short_answer_train10000_top32_answeronly_qwen_teacher_20260606

fixed target:
  answer_only, not "Final answer: X".
  The prompt already ends with "Final answer:", so this removes the duplicate
  "Final answer: Final answer: X" training string from the previous artifact.

fixed SFT counts:
  total pairs = 9028
  train pairs = 8133
  valid pairs = 895
  solver_candidate_gold_covered = 5895
  solver_candidate_teacher_correct = 3133

valid10 training artifact:
  outputs/p2_phase_a_cola_dit_lora/
  musique_candidate_constrained_short_answer_top32_answeronly_cap112_step700_valid10_seed20260606_20260606

SwanLab:
  4377spe8huweh6c2xx857

training readout:
  status = pass
  max_total_blocks = 112
  max_train_steps = 700
  valid_interval = 10
  num valid points = 71
  best_step = 490
  best_valid_loss = 0.222344
  final valid losses at step 700 = 0.440427 and 0.331650
  saved best_checkpoint.pt / best_adapter and last_checkpoint.pt / last_adapter

candidate-prompt smoke after payload fix:
  previous duplicate-target adapter:
    outputs/p2_phase_c_text_agent_runs/
    candidate_constrained_top32_cap112_best_candidateprompt_smoke20_payloadfix_20260606
    single_full_info = 0/10 primary, token_f1 = 0.02
    single_q_only = 0/10 primary

  answer-only adapter:
    outputs/p2_phase_c_text_agent_runs/
    candidate_constrained_top32_answeronly_cap112_best_candidateprompt_smoke20_20260606
    single_full_info = 0/10 primary, token_f1 = 0.04
    single_q_only = 0/10 primary

diagnosis:
  This removes the known surface confounds:
    sparse valid logging
    last-vs-best checkpoint ambiguity
    duplicate final-answer target
    dropped candidate payload

  The candidate-constrained CoLA DiT generation path still fails to select the
  answer even when the prompt is candidate-constrained and best_adapter is used.
  Therefore the next repair should not be a longer version of the same DiT
  short-answer generation run.

decision:
  Stop expanding candidate-constrained DiT generation unless a deeper
  architecture/interface reason is introduced and documented first.
  Continue Phase A through a CoLA-native candidate answer selector/ranker that
  uses frozen official CoLA VAE latent substrate and supervised candidate
  ranking/selection, with CUDA + SwanLab cloud, metrics.jsonl,
  valid_interval <= 10, best_checkpoint.pt, and last_checkpoint.pt.
```

CoLA-native latent candidate-ranker screens:

```text
new training entry:
  drla/scripts/train_p2_phase_a_cola_latent_candidate_ranker.py

new local evaluator:
  drla/scripts/eval_p2_phase_a_cola_latent_candidate_ranker.py

model:
  frozen official CoLA VAE encodes online question/evidence context and each
  evidence-derived candidate string into latent_dim=16 sequences.
  A lightweight attention-pooling ranker scores candidates from:
    context latent pool
    candidate latent pool
    pairwise product / absolute difference
    online candidate metadata features
  Online inputs exclude gold, aliases, teacher correctness, scorer outputs,
  and held-out data. Gold/scorer labels are offline supervision only.

smoke:
  top16 smoke3:
    output = outputs/p2_phase_a_cola_latent_candidate_ranker/
      musique_top16_smoke3_seed20260606_20260606
    SwanLab = q5jd4pd724hka6btiux45
    status = pass

  top128 smoke2:
    output = outputs/p2_phase_a_cola_latent_candidate_ranker/
      musique_top128_smoke2_seed20260606_20260606
    SwanLab = u9bi8jieeh5tgocl1oq7i
    status = pass

step500 v1:
  output:
    outputs/p2_phase_a_cola_latent_candidate_ranker/
    musique_top128_step500_seed20260606_20260606
  SwanLab:
    rrp6ra498uc5l9eb4jyyr
  best-checkpoint eval:
    outputs/p2_phase_a_cola_latent_candidate_ranker_evals/
    musique_top128_step500_best_full_eval_20260606
  best step = 490
  valid200 selected_primary = 0.09
  calibration200 selected_primary = 0.04

implementation diagnosis:
  v1 collapsed many actual candidate extraction rules into "other"
  because the feature schema omitted high-frequency rules such as
  capitalized_full_span, capitalized_subspan, capitalized_single,
  number_or_year, quoted_span, and season_phrase.

step500 schema-v2:
  output:
    outputs/p2_phase_a_cola_latent_candidate_ranker/
    musique_top128_step500_schema_v2_seed20260606_20260606
  SwanLab:
    0zdqy1z8irphsuvquij5f
  best-checkpoint eval:
    outputs/p2_phase_a_cola_latent_candidate_ranker_evals/
    musique_top128_step500_schema_v2_best_full_eval_20260606
  best step = 450
  train rows used = 7827 primary-positive rows
  valid rows = 200
  calibration rows = 200
  valid200 oracle_primary = 0.815
  valid200 selected_primary = 0.125
  valid200 selected_token_f1 = 0.1560
  calibration200 oracle_primary = 0.715
  calibration200 selected_primary = 0.095
  calibration200 selected_token_f1 = 0.1132

  calibration comparison:
  rank-1 candidate baseline selected_primary = 0.070
  shallow qtype logistic top128 selected_primary = 0.175
  Qwen3-8B-FP8 semantic selector top128 selected_primary = 0.445
  top128 oracle_primary = 0.715

batch-size feasibility:
  top128 batch_size=4 smoke:
    output = outputs/p2_phase_a_cola_latent_candidate_ranker/
      musique_top128_batch4_smoke2_schema_v2_seed20260606_20260606
    SwanLab = 8xnh29izydxcw7euqa1ci
    status = fail_oom
    failure site = frozen CoLA VAE encode slow attention
    allocated = 19.42 GiB
    requested = 11.17 GiB
    device limit = 31.36 GiB
  implication:
    Full-context top128 ranker training cannot simply raise batch size on the
    current slow-attention VAE path. More training coverage should use
    batch_size=1 with gradient accumulation, cached latent representations, or
    a more memory-efficient context/candidate interaction design.

decision:
  The CoLA VAE latent-ranker path is trainable and handles top128 candidates
  without DiT prompt-length OOM, but the current independent-pooling ranker is
  still below the shallow selector and far below the semantic teacher.
  Do not claim Phase A pass from this model, and do not simply scale this exact
  architecture by step count. The next repair should add richer
  context-candidate interaction and/or distill Qwen semantic selection signals
  with a stronger objective.
```

Late-interaction ranker follow-up:

```text
motivation:
  The independent-pooling ranker collapses context and candidate latent
  sequences before scoring. This is analogous to a single-vector bi-encoder and
  risks erasing answer-bearing local matches. ColBERT-style late interaction
  keeps token-level vectors and uses MaxSim-style matching, which is a more
  natural fit for candidate answer selection over latent token/block sequences.

implementation:
  train_p2_phase_a_cola_latent_candidate_ranker.py now supports:
    interaction_mode = pooled | late_maxsim
    interaction_dim = 64 by default
    feature_schema_version = 1 | 2
  The evaluator is backward-compatible with old 28-dim v1 checkpoints and
  restores interaction settings from the checkpoint.

smoke:
  output:
    outputs/p2_phase_a_cola_latent_candidate_ranker/
    musique_top32_latemaxsim_smoke2_seed20260606_20260606
  SwanLab:
    uj4evk55b25bdzsaphghm
  status = pass
  best/full eval is automatically written in summary.

top128 step500 late_maxsim:
  output:
    outputs/p2_phase_a_cola_latent_candidate_ranker/
    musique_top128_step500_latemaxsim_seed20260606_20260606
  SwanLab:
    myo0ofc3aivcxmxijbs20
  interaction:
    late_maxsim, interaction_dim=64
  best_step = 460
  train rows used = 7827 primary-positive rows
  valid rows = 200
  calibration rows = 200
  valid200 oracle_primary = 0.815
  valid200 selected_primary = 0.100
  valid200 selected_token_f1 = 0.1308
  calibration200 oracle_primary = 0.715
  calibration200 selected_primary = 0.120
  calibration200 selected_token_f1 = 0.1568
  last calibration200 selected_primary = 0.110

comparison:
  pooled schema-v2 best calibration200 selected_primary = 0.095
  late_maxsim best calibration200 selected_primary = 0.120
  shallow qtype logistic top128 selected_primary = 0.175
  Qwen3-8B-FP8 semantic selector top128 selected_primary = 0.445
  top128 oracle_primary = 0.715

diagnosis:
  Late interaction improves over independent pooling (+0.025 primary,
  +0.0437 token-F1 on calibration200), so preserving token-level latent
  interaction is directionally useful. However it is still below shallow
  feature ranking and far below the semantic teacher. The remaining gap is not
  solved by MaxSim alone; it likely needs teacher distillation, candidate-list
  normalization across candidates, or a stronger context-candidate cross
  interaction with cached latents.

decision:
  Do not claim Phase A pass and do not run held-out or Phase E from this model.
  Next repair should prioritize Qwen semantic selector distillation and/or
  cached-latent cross-candidate training rather than only increasing steps of
  the same late_maxsim architecture.
```

## Next Work

```text
1. Stop expanding the current cap96 full-info continuation path.
   It has a clean SwanLab/log/checkpoint artifact, but fails the calibration
   solver100 capability floor.

2. Stop expanding the current support-only curriculum path.
   It confirms distractor/evidence-selection pressure but does not pass the
   diagnostic support-only floor after training.

3. Stop expanding the answer-support generative target path.
   It passed data audit but worsened full-evidence solver100 and encouraged
   support-span reproduction in the scored first segment.

4. Stop expanding the current rule-candidate + shallow-selector path.
   It is useful evidence about the failure mode, but its best calibration
   selected_primary is 0.175 and it does not pass the 0.20 solver floor.

5. Build a CoLA-native candidate answer selector/ranker from the completed
   nonheldout train-source Qwen semantic teacher and evidence-derived
   candidate sets. The first frozen-VAE independent-pooling ranker is a weak
   diagnostic (`calibration200 selected_primary=0.095`); late_maxsim improves
   it to `0.120` but remains below shallow qtype `0.175`. The next version
   should use Qwen semantic teacher distillation and/or cached-latent
   cross-candidate interaction rather than only increasing steps. Do not use
   held-out and do not lower the locked floor.

6. Keep using downstream solver100 screens for checkpoint selection; FM valid
   loss alone remains insufficient.

7. Phase E remains blocked until CoLA Single Solver and CoLA Role TextMAS pass
   the locked gates on the admitted MuSiQue protocol.
```

Current hard boundary:

```text
No held-out tuning.
No Phase E TextMAS-vs-LatentMAS run from cap96 checkpoints.
No claim that latent communication failed from these adapter failures.
The current failure is CoLA task/interface adaptation, not a valid LatentMAS
comparison.
```
