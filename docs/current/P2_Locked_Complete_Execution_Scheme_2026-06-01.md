# P2 Locked Complete Execution Scheme

更新日期：2026-06-06

> 状态：当前最高优先级执行方案。本文把 post-Family1 后的路线从“分支待选”
> 固化为默认执行顺序，防止后续实验退回 old official8-only、Family 1 repair、
> frozen-CoLA 盲目找题或提前训练 fuser/adapter。本文不记录实验结果本身，
> 而是定义后续实验能否启动、能否进入主表、能否做 claim 的完整协议。

## 0. 一句话路线

```text
先用 capable text agents 锁定 true MAS benchmark/protocol
-> 再做 CoLA substrate / role interface adaptation
-> 最后在同一 locked benchmark 上比较 CoLA TextMAS vs CoLA LatentMAS
```

核心目标保持不变：

```text
same-substrate Cola A -> Cola B latent communication
role-conditioned MAS
capability-gated benchmark
matched TextMAS vs LatentMAS paper-level comparison
```

## 0.1 2026-06-01 完整方案锁定补丁

本文后续执行按下面四个判断固定，不再被 old official8、旧 direct-answer
handoff 或 premature fuser training 扰动：

```text
1. benchmark 必须先换成 true MAS benchmark。
   official8 / GSM8K / ARC / GPQA 等单问答任务只能证明 solver capability，
   不能作为 agent-to-agent communication 主 claim。

2. agent baseline 必须先换成 capable TextMAS。
   当前要先证明任务与协议本身合理：A/B 等 role agents 持有互补私有信息，
   final Solver/Agent B 只能通过上游消息或允许的 public context 作答。

3. CoLA 架构适合作为 shared latent substrate，但当前 frozen official CoLA
   权重不自动适配新 benchmark。
   它可以提供 text <-> latent 与 block-causal latent reasoning 接口；但
   instruction following、role-message parsing、evidence integration 和
   final answer formatting 必须通过 Phase A gate 证明或通过 adapter/LoRA
   建立。不能因为 frozen CoLA 在新任务上低分就否定 latent communication。

4. LatentMAS 主实验只在同一 locked benchmark、同一 role protocol、
   同一 scorer 边界下比较。
   TextMAS: Agent A 的文本输出作为 Agent B 输入。
   LatentMAS: Agent A 的 latent packet/state 作为 Agent B 输入。
   scorer 只看 Agent B / final Solver 在 handoff 后生成的最终答案。
```

必须坚持的研究问题是：

```text
在同一 CoLA substrate 上，A 的 latent state 是否能成为 B 可消费的通信状态，
并在自然需要通信的任务上达到 readable / useful / competitive 三层证据。
```

当前不再追求：

```text
让 DiT LoRA 刷官方 benchmark accuracy。
在单问答数据上构造看似 multi-agent 的同质 solver relay。
用 direct decoded answer 或 replay-visible tokens 绕过 Agent B。
在 text baseline 未成立前训练 latent receiver/fuser。
```

但当前结论已经锁定：

```text
frozen official CoLA + 当前候选任务:
  admitted_tasks = []

official8-only solver-to-solver message_only:
  只保留为 channel diagnostic

下一步:
  不再继续 Family 1 held-out / P2 main table / fuser training
  先进入 true MAS benchmark/protocol validation
```

## 0.2 2026-06-01 Phase C 校准结果锁定

Phase C 现在已有一个 calibration-level admitted protocol：

```text
benchmark/protocol:
  MuSiQue evidence-split QA
  prompt contract: p2_phase_c_evidence_split_v1_strict_wrong_evidence
  model: local Qwen3-8B-FP8, local_transformers provider
  split: calibration, 200 samples
  conditions: 7
  total online rows: 1400

artifact root:
  outputs/p2_phase_c_text_agent_runs/
  musique_local_qwen3_8b_fp8_v1_strict_wrong_full200_merged_20260601

aggregate:
  outputs/p2_phase_c_text_agent_aggregates/
  musique_local_qwen3_8b_fp8_v1_strict_wrong_full200_20260601

leakage audit:
  outputs/p2_phase_c_leakage_audits/
  musique_local_qwen3_8b_fp8_v1_strict_wrong_full200_20260601
```

Primary readout：

```text
single_full_info: 0.425 primary score, 0.390 exact match
single_q_only: 0.080 primary score, 0.065 exact match
textmas_matched: 0.450 primary score, 0.395 exact match
textmas_no_message: 0.080 primary score, 0.065 exact match
textmas_shuffled_message: 0.060 primary score, 0.045 exact match
textmas_wrong_evidence_or_wrong_shard: 0.070 primary score, 0.060 exact match
textmas_compressed_state: 0.420 primary score, 0.355 exact match

paired bootstrap CI lower:
  single_full_info - single_q_only: +0.265
  textmas_matched - no_message: +0.295
  textmas_matched - shuffled_message: +0.320
  textmas_matched - wrong_evidence: +0.315

gate:
  admitted=true on calibration
  parseable_rate=1.0 for all conditions
  run-level leakage audit status=pass
  leakage errors=0
```

解释：

```text
这证明 MuSiQue evidence-split 协议在 capable local text agent 上确实需要
上游通信；Agent B/Solver 不是只靠问题、空消息、错配消息或错误 evidence
就能达到 matched 水平。

这还不是 paper 主结论，也不是 CoLA latent communication 结论。
它只是 Phase C calibration admission，下一步必须做 locked held-out evaluation。
```

HotpotQA 当前状态：

```text
HotpotQA Qwen3-8B-FP8 v1 strict 10-sample pilot fails gate.
原因是 single_q_only / no_message shortcut 太强，CI lower 对 no_message 为 0。
HotpotQA 暂时只保留为诊断，不作为下一步主 admission candidate。
```

泄漏 warning 口径：

```text
Phase C evidence QA 中，答案字符串出现在 online evidence text 是常见现象；
它是 shortcut-risk warning，不等价于 hidden gold label 泄漏。
是否可用必须由 no-message / shuffled / wrong-evidence controls 判定。
MuSiQue full calibration 的这些 controls 均显著低于 matched。
```

后续执行顺序现在锁定为：

```text
1. Freeze current MuSiQue calibration protocol and artifacts.
2. Build locked MuSiQue held-out split with the same schema, prompt contract,
   scorer, conditions and gate; no held-out prompt repair.
3. Run capable TextMAS held-out evaluation once.
4. If held-out passes, freeze this as Phase C admitted benchmark/protocol.
5. Enter Phase A: CoLA substrate/interface adaptation on the locked protocol.
6. Only after CoLA Single Solver and CoLA Role TextMAS pass gate, enter Phase E
   same-benchmark TextMAS vs LatentMAS comparison.
```

明确禁止：

```text
不能把 calibration admission 当最终 paper number。
不能在 held-out 后再改 prompt/parser/control definition。
不能跳过 CoLA Single/Role TextMAS gate 直接训练 latent fuser/receiver。
不能把 HotpotQA pilot、official8 channel diagnostic 或 decode-and-emit replay
混入 MuSiQue locked protocol 的主表。
```

## 0.3 2026-06-05 Phase C locked held-out 结果锁定

MuSiQue evidence-split QA v1 strict 已通过 locked held-out：

```text
benchmark/protocol:
  MuSiQue evidence-split QA
  prompt contract: p2_phase_c_evidence_split_v1_strict_wrong_evidence
  model: local Qwen3-8B-FP8, local_transformers provider
  split: heldout from bdsaglam/musique answerable/validation
  samples: 800
  conditions: 7
  total online rows: 5600

merged run:
  outputs/p2_phase_c_text_agent_runs/
  musique_local_qwen3_8b_fp8_v1_strict_wrong_heldout800_merged_20260605

aggregate:
  outputs/p2_phase_c_text_agent_aggregates/
  musique_local_qwen3_8b_fp8_v1_strict_wrong_heldout800_20260605

leakage audit:
  outputs/p2_phase_c_leakage_audits/
  musique_local_qwen3_8b_fp8_v1_strict_wrong_heldout800_20260605
```

Held-out primary readout：

```text
single_full_info: 0.48625 primary score, 0.45000 exact match
single_q_only: 0.03875 primary score, 0.03250 exact match
textmas_matched: 0.37375 primary score, 0.34000 exact match
textmas_no_message: 0.03875 primary score, 0.03250 exact match
textmas_shuffled_message: 0.04750 primary score, 0.04000 exact match
textmas_wrong_evidence_or_wrong_shard: 0.04875 primary score, 0.04125 exact match
textmas_compressed_state: 0.38250 primary score, 0.35375 exact match

paired bootstrap CI lower:
  single_full_info - single_q_only: +0.41250
  textmas_matched - no_message: +0.30125
  textmas_matched - shuffled_message: +0.29375
  textmas_matched - wrong_evidence: +0.29250

gate:
  admitted=true on locked held-out
  parseable_rate=1.0 for all conditions
  run-level leakage audit status=pass
  leakage errors=0
  leakage warnings=2640
```

解释：

```text
Phase C 的 true-MAS benchmark/protocol validation 已完成：
MuSiQue held-out 证明 capable TextMAS 的 matched communication 明显优于
question-only / no-message / shuffled-message / wrong-evidence controls。

这仍不是 CoLA latent communication 结果；它证明的是实验场和协议成立。
现在可以进入 Phase A: CoLA substrate/interface adaptation。
```

后续执行顺序更新为：

```text
1. Freeze MuSiQue v1 strict as the current admitted Phase C benchmark/protocol.
2. Start Phase A on this locked protocol:
   CoLA Single Solver capability gate.
   CoLA Role TextMAS capability gate.
   If needed, task-format / role-interface adapter or LoRA.
3. Only after CoLA Single and CoLA Role TextMAS pass gate, enter Phase E:
   same-benchmark CoLA TextMAS vs CoLA LatentMAS comparison.
```

新的禁止项：

```text
不能再回头改 MuSiQue held-out prompt/parser/control definition 来追分。
不能把 Qwen held-out score 写成 CoLA score。
不能跳过 CoLA Single/Role TextMAS gate 直接 claim LatentMAS。
不能因为 frozen CoLA 初跑低分就否定 latent communication；那首先是
substrate/interface adaptation 问题。
```

## 0.4 2026-06-05 Phase A CoLA 接口诊断锁定

Phase A 已验证两件事：

```text
1. official CoLA VAE/DiT 已能被 Phase C runner 调用。
   它使用相同 MuSiQue online inputs、parser、QA scorer、resume 行为和
   artifact schema；这一步是 CoLA substrate plumbing，不是主实验结论。

2. frozen official CoLA 的 prompt-only MuSiQue role/interface 适配失败。
   chat_join、plain_qa_v1、squad_template_v1 三种 smoke 均未在 14-row
   smoke 上产生 primary-score 命中。
```

关键 artifact：

```text
chat_join held-out smoke:
  outputs/p2_phase_c_text_agent_runs/
  musique_cola_dlm_v1_strict_wrong_heldout_smoke14_20260605

plain_qa_v1 calibration smoke:
  outputs/p2_phase_c_text_agent_runs/
  musique_cola_dlm_plainqa_v1_strict_wrong_calibration_smoke14_20260605

squad_template_v1 calibration smoke:
  outputs/p2_phase_c_text_agent_runs/
  musique_cola_dlm_squadtemplate_v1_strict_wrong_calibration_smoke14_20260605

official CoLA SQuAD sanity:
  outputs/p2_phase_a_official_cola_sanity/
  official_squad_smoke20_20260605
```

读数：

```text
MuSiQue prompt-only CoLA smoke:
  all seven conditions primary_score_mean = 0.0
  raw outputs show evidence copying, continuation drift, and answer-format drift

Official SQuAD 20-sample sanity with same local CoLA weights:
  primary matches = 5/20
  primary score mean = 0.25
  existing full official SQuAD baseline in local eval_output = 30.90%
```

解释：

```text
这说明 CoLA 权重与官方推理链路不是坏的；失败来自 MuSiQue evidence-split
role protocol 与 frozen CoLA 当前 instruction/format 分布不匹配。

因此下一步不是继续手调 prompt，也不是据此否定 latent communication。
下一步是 calibration-only 的 CoLA task-format / role-interface adaptation。
```

Phase A 后续门槛现在锁定为：

```text
1. 只用 calibration 设计和训练 adapter / LoRA / role-interface。
2. 所有有 optimizer/backward 的深度学习训练必须 GPU + SwanLab cloud，
   写 local metrics.jsonl，保存 best_checkpoint.pt 和 last_checkpoint.pt，
   valid_interval <= 10 step。
3. 训练完成后先跑 CoLA Single Solver gate 和 CoLA Role TextMAS gate。
4. 只有这两个 gate 通过，才能进入 Phase E 的 same-benchmark TextMAS vs
   LatentMAS 主比较。
5. Held-out MuSiQue 不得用于 adapter 设计、prompt repair 或 threshold 选择。
```

2026-06-05 追加结果：

```text
Phase A SFT data:
  outputs/p2_phase_a_cola_interface_sft/
  musique_calibration_qwen_teacher_v1_20260605
  pairs: 800 total = 200 full-info solver + 200 matched solver
         + 400 evidence-agent teacher pairs
  split: sample-level train/valid = 640/160
  source: calibration only, Qwen3-8B-FP8 admitted TextMAS teacher messages

Current best adapter:
  outputs/p2_phase_a_cola_dit_lora/
  musique_solver_interface_lora_v1_epoch3_eos_20260605
  method: official CoLA DiT LoRA Flow-Matching, frozen VAE, target EOS included
  SwanLab run: lmw46365bo8dneyheqw49
  best step: 1100
  best valid FM loss: 0.14344
  trainable params: 6.34M / 1.836B
```

Deterministic calibration solver100 result, seed 66：

```text
frozen CoLA + squad_template_v1 + first_segment:
  single_full_info primary = 0.00 / 50
  single_q_only primary   = 0.00 / 50

solver-interface LoRA epoch3 + EOS:
  single_full_info primary = 0.36 / 50
  single_q_only primary   = 0.10 / 50
  paired full-info minus q-only mean diff = +0.26
  paired bootstrap 95% CI lower = +0.12
```

解释：

```text
CoLA Single Solver interface adaptation has a positive calibration result.
This is still not a Role TextMAS gate and not a LatentMAS result.

The next Phase A step is evidence-agent / role-message adaptation, then a full
seven-condition calibration gate with matched/no-message/shuffled/wrong controls.
Held-out remains locked until the full calibration gate passes.
```

2026-06-05 追加结果：Phase A Role TextMAS calibration gate 已通过。

```text
evidence-agent adapter:
  outputs/p2_phase_a_cola_dit_lora/
  musique_evidence_agent_lora_v1_from_solver_epoch2_20260605
  method: official CoLA DiT LoRA Flow-Matching, frozen VAE, target EOS included
  SwanLab run: yra5lf4h00711kbb87m3z
  best step: 3200
  best valid FM loss: 0.25338

solver adapter:
  outputs/p2_phase_a_cola_dit_lora/
  musique_solver_interface_lora_v1_epoch3_eos_20260605

calibration merged run:
  outputs/p2_phase_c_text_agent_runs/
  musique_cola_dlm_dual_lora_evidence_epoch2_solver_epoch3eos_squadtemplate_v1_firstseg_seed66_calibration_full200_merged_20260605

aggregate:
  outputs/p2_phase_c_text_agent_aggregates/
  musique_cola_dlm_dual_lora_evidence_epoch2_solver_epoch3eos_squadtemplate_v1_firstseg_seed66_calibration_full200_20260605

leakage audit:
  outputs/p2_phase_c_leakage_audits/
  musique_cola_dlm_dual_lora_evidence_epoch2_solver_epoch3eos_squadtemplate_v1_firstseg_seed66_calibration_full200_20260605
```

Calibration primary readout:

```text
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

执行约束随之更新：

```text
1. Phase A calibration gate 已满足，可以进入同一设置的 locked held-out run。
2. Held-out run 必须使用相同 prompt style、first_segment parser、seed 66、
   evidence-agent adapter、solver adapter、scorer 和 v1 strict controls。
3. Held-out 不得用于 prompt repair、adapter 选择、threshold selection 或
   任何形式的再调参。
4. 只有 held-out 也通过，才进入 Phase E 的 CoLA TextMAS vs LatentMAS 主比较。
```

2026-06-05 locked held-out 结果：Phase A gate 未通过。

```text
held-out merged run:
  outputs/p2_phase_c_text_agent_runs/
  musique_cola_dlm_dual_lora_evidence_epoch2_solver_epoch3eos_squadtemplate_v1_firstseg_seed66_heldout800_merged_20260605

aggregate:
  outputs/p2_phase_c_text_agent_aggregates/
  musique_cola_dlm_dual_lora_evidence_epoch2_solver_epoch3eos_squadtemplate_v1_firstseg_seed66_heldout800_20260605

leakage audit:
  outputs/p2_phase_c_leakage_audits/
  musique_cola_dlm_dual_lora_evidence_epoch2_solver_epoch3eos_squadtemplate_v1_firstseg_seed66_heldout800_20260605
```

Held-out primary readout:

```text
single_full_info: 0.08750
single_q_only: 0.00250
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
  leakage audit status=pass
  leakage errors=0
  leakage warnings=2640
```

结论：

```text
1. Role-message signal generalizes: matched remains clearly above no-message,
   shuffled-message and wrong-evidence controls.
2. Absolute CoLA solver capability does not generalize enough:
   single_full_info falls below the locked floor.
3. Phase E must not start from this dual-adapter checkpoint.
4. Do not lower the gate after seeing held-out.
5. Do not use this held-out split for prompt repair, adapter selection,
   threshold choice, or new SFT target construction.
6. Next CoLA adaptation attempt must be justified from calibration/training
   sources only and must pass a fresh locked evaluation before paper-level
   TextMAS vs LatentMAS comparison.
```

2026-06-05 train-only generalization repair 结果：Phase A calibration gate 未通过。

```text
motivation:
  The previous calibration-trained dual adapter passed calibration but failed
  locked held-out. A stricter repair attempt therefore used 2000 MuSiQue
  answerable/train rows and Qwen3-8B-FP8 teacher evidence-agent messages,
  without held-out rows, to train role-specific CoLA DiT LoRA adapters.

SFT artifact:
  outputs/p2_phase_a_cola_interface_sft/
  musique_interface_train_qwen_teacher_2000_t96_seed20260606_20260605
  pairs: 8000 total
  roles: 2000 solver_full_info, 2000 solver_textmas_matched,
         4000 evidence_agent_teacher

solver adapter:
  outputs/p2_phase_a_cola_dit_lora/
  musique_solver_interface_lora_train2000_t96_epoch2_seed20260606_20260605
  SwanLab run: hv2i6pj6edx4x0i2zqjt3
  best_step: 3500
  best_valid_loss: 0.13262649088341277

evidence-agent adapter:
  outputs/p2_phase_a_cola_dit_lora/
  musique_evidence_agent_lora_train2000_t96_from_solver_epoch1_seed20260606_20260605
  SwanLab run: kfu5ph91sj21pplmgtzzu
  best_step: 20300
  best_valid_loss: 0.21131337011232973
```

Full 200-sample calibration gate:

```text
merged run:
  outputs/p2_phase_c_text_agent_runs/
  musique_cola_dlm_dual_lora_train2000_t96_seed20260606_squadtemplate_v1_firstseg_seed66_calibration_full200_merged_20260605

aggregate:
  outputs/p2_phase_c_text_agent_aggregates/
  musique_cola_dlm_dual_lora_train2000_t96_seed20260606_squadtemplate_v1_firstseg_seed66_calibration_full200_20260605

primary:
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
```

结论：

```text
1. The communication differential remains real: matched is above no-message,
   shuffled and wrong-evidence controls with positive CI lower bounds.
2. The absolute CoLA solver capability is still below the locked floor even
   on calibration. This checkpoint must not enter held-out.
3. Phase E remains blocked. Do not start TextMAS vs LatentMAS from this model.
4. Next work should target non-heldout full-info solver generalization and
   adapter selection, not gate lowering or held-out tuning.
```

Train-source diagnostic for the same solver adapter:

```text
merged run:
  outputs/p2_phase_c_text_agent_runs/
  musique_cola_dlm_solver_train2000_t96_train_source_single_diag100_merged_20260605

split:
  train only
  100 samples x {single_full_info, single_q_only}

readout:
  single_full_info primary: 0.240
  single_q_only primary: 0.000
  paired diff: +0.240
  bootstrap CI lower: +0.160
```

Implication:

```text
The train-only solver adapter is learning evidence-conditioned answering on
its source distribution, but it does not yet generalize enough to the locked
calibration set. The next admissible repair is to improve non-heldout full-info
solver generalization and checkpoint selection. Held-out remains forbidden for
repair decisions, and Phase E remains blocked.
```

10k full-info continuation attempt:

```text
data:
  outputs/p2_phase_c_data_source_audits/
  hf_dataset_rows_musique_answerable_train_10000_seed20260606
  backend: datasets.load_dataset

SFT:
  outputs/p2_phase_a_cola_interface_sft/
  musique_interface_train_fullinfo_10000_seed20260606_20260605
  solver_full_info pairs: 10000

training run:
  outputs/p2_phase_a_cola_dit_lora/
  musique_solver_fullinfo_lora_train10000_from_train2000_epoch1_lr5e5_seed20260606_20260605
  SwanLab: mludvhrmz5yadn25niij0
  status: failed_runtime_nvml_assert at train step 527
  failure summary:
    summary_runtime_failure.json

partial step500 calibration solver100:
  single_full_info primary: 0.030
  single_q_only primary: 0.000
  paired diff: +0.030
  CI lower: 0.000
```

Implication:

```text
The partial checkpoint is unusable. The run also shows that lower
Flow-Matching valid loss can select a worse downstream QA checkpoint.
Before another large full-info repair, update the training/selection protocol
to preserve interval checkpoints and/or evaluate a non-heldout solver
capability diagnostic. Do not use this crashed run for Role TextMAS, held-out,
or Phase E.
```

Implementation note:

```text
drla/scripts/train_p2_phase_a_cola_dit_lora.py now supports
--save-interval-checkpoints.

When enabled, every valid interval saves:
  checkpoints/valid_step_<step>.pt
  checkpoints/valid_step_<step>_adapter

This is required for the next full-info repair run unless a stronger
non-heldout checkpoint-selection mechanism replaces it.
```

## 0.5 2026-06-06 10k Full-Info Repair Gate 结果锁定

The 10k full-info repair path has now been tested under a safer length and
checkpoint-selection protocol. It does not pass the Phase A solver capability
floor and must not be expanded into Role TextMAS, held-out, or Phase E.

Training/run artifact:

```text
outputs/p2_phase_a_cola_dit_lora/
musique_solver_fullinfo_lora_train10000_cap96_from_train2000_step2000_lr1e5_interval_seed20260606_20260606

SwanLab:
  y92wrmkp38hcuc8x4y5h0

method:
  official CoLA DiT LoRA Flow-Matching
  frozen VAE
  roles = solver_full_info
  init_lora_path = train2000 solver best_adapter
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
```

Why cap96 was used:

```text
10k train solver_full_info length audit:
  token median/p90/p95/p99/max = 782 / 1134 / 1262 / 1549 / 2237
  latent block median/p90/p95/p99/max = 49 / 71 / 79 / 97 / 140

The previous long run crashed with a PyTorch CUDA allocator/NVML internal
assert at step 527. The current machine still reports NVML driver/library
mismatch through nvidia-smi, while PyTorch CUDA itself works. Short allocator
tests pass with backend:cudaMallocAsync, so long CoLA train/eval should use
that setting until the machine NVML stack is fixed.

Cap96 removes the extreme tail while preserving the calibration length range
and the train2000 reference range.
```

Checkpoint-selection diagnostics:

```text
nonheldout SFT-valid screen:
  outputs/p2_phase_c_text_agent_aggregates/
  validdiag50_solver_candidate_screen_20260606

  cap96_step900:   full-info 0.10, q-only 0.02, paired diff +0.08
  cap96_step1300:  full-info 0.08, q-only 0.00, paired diff +0.08
  cap96_last:      full-info 0.08, q-only 0.00, paired diff +0.08

locked calibration solver100 precheck:
  outputs/p2_phase_c_text_agent_aggregates/
  calibration_solver100_cap96_candidate_precheck_20260606

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
```

Gate decision:

```text
status:
  failed_precheck_no_candidate_reaches_full_info_floor

locked floor:
  single_full_info primary >= 0.20

best observed:
  cap96_last single_full_info primary = 0.08

decision:
  do not run full Role TextMAS gate for this cap96 solver
  do not run held-out for this cap96 solver
  do not enter Phase E TextMAS-vs-LatentMAS from this path
```

Interpretation:

```text
The 10k cap96 run fixed training stability and produced much lower
Flow-Matching valid loss, but downstream QA capability on locked calibration
did not recover. This is evidence that current full-sequence Flow-Matching
valid loss is misaligned with final-answer utility for this MuSiQue interface.

The failure is CoLA task/interface adaptation, not a valid negative result for
latent communication. Phase E remains blocked.
```

Objective-mismatch diagnostic:

```text
artifact:
  outputs/p2_phase_a_diagnostics/
  fullinfo_objective_mismatch_20260606

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

SFT length audit for 10k solver_full_info:
  target_tokens median/p90/p95/p99/max = 2 / 4 / 6 / 9 / 15
  context_tokens median/p90/p95/p99/max = 533 / 791 / 876 / 1049 / 1438
  prompt_tokens median/p90/p95/p99/max = 551 / 814 / 899 / 1076 / 1471
  answer_string_in_context_rate = 1.000
```

Diagnostic interpretation:

```text
Most failures are answer-shaped, not empty/parser failures. The adapter often
selects the wrong support/distractor entity, emits the right answer plus extra
tokens, or produces plausible evidence-adjacent spans. EOS-tail artifacts are
common in raw output but are already stripped by first_segment parsing.

The main issue is objective mismatch: full-sequence Flow-Matching over long
prompt/context-to-short-answer examples can reduce valid loss without learning
the evidence selection and final-answer extraction policy required by the QA
scorer.
```

Support-only diagnostic and curriculum gate:

```text
support-only controls:
  outputs/p2_phase_c_control_inputs/
  musique_calibration_solver_support_only_diag_200_seed20260606
  conditions = single_full_info, single_q_only
  diagnostic only; not a locked gate

pre-curriculum support-only eval:
  outputs/p2_phase_c_text_agent_aggregates/
  calibration_solver100_support_only_diag_cap96_last_20260606

  cap96_last support-only single_full_info = 0.16
  cap96_last support-only single_q_only = 0.01
  paired diff = +0.15
  paired bootstrap 95% CI = [+0.08, +0.23]

support-only SFT:
  outputs/p2_phase_a_cola_interface_sft/
  musique_interface_train_support_only_10000_seed20260606_20260606
  role_counts: solver_support_only = 10000
  no teacher-role pairs
  train/valid = 9000 / 1000

support-only curriculum training:
  outputs/p2_phase_a_cola_dit_lora/
  musique_solver_support_only_lora_train10000_from_cap96_last_step1000_lr1e5_interval_seed20260606_20260606
  SwanLab = g4yipez7s8mk16i2y88lf
  best_step = 500
  best_valid_loss = 0.1216601644147886

support-only curriculum solver100 aggregate:
  outputs/p2_phase_c_text_agent_aggregates/
  support_only_curriculum_solver100_comparison_20260606

  support_curriculum_best_step500:
    support-only single_full_info = 0.17
    single_q_only = 0.01
    paired diff = +0.16
    paired bootstrap 95% CI = [+0.08, +0.24]
    diagnostic floor 0.20: fail

  support_curriculum_last_step1000:
    support-only single_full_info = 0.17
    single_q_only = 0.01
    paired diff = +0.16
    paired bootstrap 95% CI = [+0.08, +0.24]
    diagnostic floor 0.20: fail
```

Support-only decision:

```text
Removing distractors improves the cap96 solver from locked full-evidence 0.08
to support-only 0.16, so distractor/evidence selection is a real failure mode.
However, support-only curriculum training only reaches 0.17 and still fails
the diagnostic 0.20 floor.

Therefore:
  do not run Role TextMAS gate from support-only curriculum checkpoints
  do not run held-out from support-only curriculum checkpoints
  do not enter Phase E from support-only curriculum checkpoints
  do not repeat the same support-only Flow-Matching objective as the next repair

Next repair must add a stronger answer-selection/span-selection signal, not
only shorter support-only prompts.
```

Answer-support generative target result:

```text
SFT builder:
  drla/scripts/build_p2_phase_a_cola_interface_sft.py
  --solver-target-mode final_answer_then_support

SFT artifact:
  outputs/p2_phase_a_cola_interface_sft/
  musique_interface_train_answer_support_10000_seed20260606_20260606
  role_counts: solver_full_info = 10000
  first_line_bad = 0
  answer_in_selected_support_rate = 0.9796
  target_tokens train median/p90/p95/p99/max = 126 / 189 / 219 / 296 / 408

training:
  outputs/p2_phase_a_cola_dit_lora/
  musique_solver_answer_support_lora_train10000_from_cap96_last_step1000_lr1e5_cap96_interval_seed20260606_20260606
  SwanLab = 8rtj42x4ey9zadeu8y936
  best_step = 1000
  best_valid_loss = 0.2581918811961077

full-evidence solver100 aggregate:
  outputs/p2_phase_c_text_agent_aggregates/
  answer_support_best_fullevidence_solver100_20260606

readout:
  single_full_info primary = 0.05
  single_q_only primary = 0.01
  paired diff = +0.04
  paired bootstrap 95% CI = [0.00, +0.09]
  locked floor 0.20: fail
```

Answer-support decision:

```text
This target passed data audit but worsened real full-evidence solver100
relative to cap96_last. Raw outputs show long support-style spans or entity
lists in the scored first segment, so the objective encourages evidence
reproduction rather than disciplined short-answer selection.

Therefore:
  do not run Role TextMAS gate from answer-support checkpoints
  do not run held-out from answer-support checkpoints
  do not enter Phase E from answer-support checkpoints
  do not repeat the same "Final answer + Selected support" generative target
```

Candidate-answer selection diagnostic:

```text
scope:
  nonheldout local diagnostic only
  no deep-learning training
  no SwanLab run
  no model generation
  no held-out data

builder:
  drla/scripts/build_p2_phase_a_candidate_answer_sets.py

selector:
  drla/scripts/train_p2_phase_a_candidate_answer_selector.py
```

Readout:

```text
oracle coverage kept:
  train top128 = 0.8012
  calibration top128 = 0.7150
  train top256 = 0.8182
  calibration top256 = 0.7150

calibration selected_primary:
  logistic basic top128 = 0.070
  logistic qtype top128 = 0.175
  logistic qtype top256 = 0.170
  hist_gbdt top128 = 0.135
  hist_gbdt top256 = 0.115

best shallow selector:
  logistic qtype top128
  selected_primary = 0.175
  locked solver100 floor = 0.20
  status = fail_floor
```

Decision:

```text
This branch is useful as a failure diagnostic but not as an admitted solver
repair. Candidate coverage is substantially higher than selected accuracy,
which indicates answer-selection/reranking is the bottleneck. Increasing the
candidate budget to top256 and replacing logistic with HistGradientBoosting do
not improve top-1 calibration accuracy.

Therefore:
  do not run Role TextMAS gate from shallow candidate-selector outputs
  do not run held-out from this selector branch
  do not enter Phase E from this selector branch
  do not continue by merely adding more candidate rules or a larger pool
```

Qwen semantic candidate-selector diagnostic:

```text
script:
  drla/scripts/run_p2_phase_a_candidate_answer_llm_selector.py

model:
  local Qwen3-8B-FP8

input boundary:
  calibration top128 evidence-derived candidates
  calibration full online evidence and question
  no gold/scorer fields as online input
  gold/aliases used only for offline scoring
  no held-out data

full calibration200 artifact:
  outputs/p2_phase_a_candidate_selectors/
  musique_candidate_selector_qwen3_8b_fp8_calib200_top128_20260606

readout:
  selected_primary = 0.445
  selected_exact_match = 0.395
  selected_token_f1 = 0.4485
  oracle_coverage_kept = 0.715
  selected_given_covered = 0.6154
```

Decision:

```text
This is a positive diagnostic for the candidate protocol and a negative
diagnostic for shallow selectors. A semantic model can exploit the same
candidate/evidence interface that logistic/GBDT cannot rank. The candidate
pool still has an oracle ceiling problem, but the immediate repair target is
semantic answer selection/distillation, not more rule candidates.

Therefore:
  Qwen semantic selector outputs may be used as nonheldout teacher evidence
  for a constrained short-answer CoLA target or CoLA-native answer selector.
  They are not CoLA results and must not be cited as Phase E TextMAS/LatentMAS.
```

Train-source teacher construction status:

```text
selector sharding support:
  drla/scripts/run_p2_phase_a_candidate_answer_llm_selector.py
  --num-shards / --shard-index

selector aggregate:
  drla/scripts/aggregate_p2_phase_a_candidate_answer_llm_selector.py

completed:
  train10k top128 shard00of10 ... shard09of10
  outputs/p2_phase_a_candidate_selectors/
  musique_candidate_selector_qwen3_8b_fp8_train10000_top128_shard00of10_20260606
  ...
  musique_candidate_selector_qwen3_8b_fp8_train10000_top128_shard09of10_20260606

aggregate:
  outputs/p2_phase_a_candidate_selectors/
  musique_candidate_selector_qwen3_8b_fp8_train10000_top128_all10_aggregate_20260606

readout:
  num_predictions = 10000
  unique sample ids = 10000
  selected_primary = 0.4336
  selected_exact_match = 0.3893
  selected_token_f1 = 0.4941
  oracle_coverage_kept = 0.8012
  selected_given_covered = 0.4908
  candidate_exact_or_high_f1_rate = 0.8634
```

Decision:

```text
The complete train-source teacher confirms that the semantic selector direction
transfers from calibration to train-source data. It is now allowed as
nonheldout teacher evidence for a constrained short-answer CoLA target or
CoLA-native answer selector.

It remains a Qwen teacher artifact, not a CoLA result and not a Phase E
communication result. Held-out must stay locked for final admission only.
```

Candidate-constrained short-answer SFT construction:

```text
script:
  drla/scripts/build_p2_phase_a_candidate_constrained_sft.py

artifact:
  outputs/p2_phase_a_cola_interface_sft/
  musique_candidate_constrained_short_answer_train10000_top128_qwen_teacher_20260606

roles:
  solver_candidate_gold_covered = 8012 pairs
  solver_candidate_teacher_correct = 4336 pairs
  total pairs = 12348
  train pairs = 11133
  valid pairs = 1215

online prompt includes:
  question
  full online evidence
  evidence-derived candidate text/rank/rule/source metadata

online prompt excludes:
  gold label
  alias flag
  teacher correctness
  scorer output
  held-out data

length audit under official CoLA tokenizer:
  block_size = 16
  tokens p95 = 4577
  blocks p95 = 287
  tokens max = 5833
  blocks max = 365

status:
  ready for GPU+SwanLab shape/memory smoke and then real candidate-constrained
  short-answer CoLA DiT LoRA training if feasible.

training logging policy:
  subsequent CoLA LoRA training must use valid_interval <= 10 for a 10:1
  train/valid observation cadence, write metrics.jsonl, and save both
  best_checkpoint.pt / best_adapter and last_checkpoint.pt / last_adapter.
  Downstream solver screens should use best_adapter by default.
```

Candidate-constrained top32-cap112 result:

```text
training artifact:
  outputs/p2_phase_a_cola_dit_lora/
  musique_candidate_constrained_short_answer_both_top32_cap112_step900_seed20260606_20260606

SwanLab:
  imprfr8cdudi791eer012

training readout:
  status = pass
  best_step = 700
  best_valid_loss = 0.211184
  last/final_valid_loss = 0.217415
  best_adapter and last_adapter both saved
  interval checkpoints valid_step_100 ... valid_step_900 saved
  note: this completed run used valid_interval=100 before the later policy
        update to valid_interval <= 10

solver100 aggregate:
  outputs/p2_phase_c_text_agent_aggregates/
  candidate_constrained_top32_cap112_best_solver100_20260606

solver100 readout:
  single_full_info primary = 0.01
  single_q_only primary = 0.00
  paired diff = +0.01
  paired bootstrap 95% CI = [0.00, 0.03]
  parseable_rate = 1.00

decision:
  fails locked solver100 floor of single_full_info >= 0.20
  do not run Role TextMAS, held-out, or Phase E from this adapter
```

Candidate-prompt payload repair and answer-only target follow-up:

```text
payload fix:
  run_p2_phase_c_text_agents.py now passes candidate_answers into the solver
  payload and supports cola_prompt_style=candidate_constrained_v1.

answer-only SFT:
  outputs/p2_phase_a_cola_interface_sft/
  musique_candidate_constrained_short_answer_train10000_top32_answeronly_qwen_teacher_20260606
  total pairs = 9028
  train/valid = 8133 / 895
  solver_candidate_gold_covered = 5895
  solver_candidate_teacher_correct = 3133

answer-only valid10 training:
  outputs/p2_phase_a_cola_dit_lora/
  musique_candidate_constrained_short_answer_top32_answeronly_cap112_step700_valid10_seed20260606_20260606
  SwanLab = 4377spe8huweh6c2xx857
  valid_interval = 10
  best_step = 490
  best_valid_loss = 0.222344
  best_adapter and last_adapter saved

candidate-prompt smoke:
  duplicate-target adapter after payload fix:
    single_full_info = 0/10 primary
  answer-only adapter:
    single_full_info = 0/10 primary

decision:
  This removes the known confounds around payload, target text, valid cadence,
  and checkpoint selection. The current candidate-constrained DiT generation
  path still does not become a usable CoLA solver. Do not scale this exact path
  further without a deeper architecture/interface change.
```

CoLA-native latent candidate-ranker screen:

```text
trainer:
  drla/scripts/train_p2_phase_a_cola_latent_candidate_ranker.py

evaluator:
  drla/scripts/eval_p2_phase_a_cola_latent_candidate_ranker.py

contract:
  frozen official CoLA VAE encodes only online question/evidence/candidate
  text into latent representations.
  Gold, aliases, teacher correctness, and scorer output are offline labels
  only.
  Training uses CUDA + SwanLab cloud + metrics.jsonl + valid_interval <= 10
  + best_checkpoint.pt + last_checkpoint.pt.

schema-v2 top128 step500:
  training output:
    outputs/p2_phase_a_cola_latent_candidate_ranker/
    musique_top128_step500_schema_v2_seed20260606_20260606
  SwanLab:
    0zdqy1z8irphsuvquij5f
  best eval:
    outputs/p2_phase_a_cola_latent_candidate_ranker_evals/
    musique_top128_step500_schema_v2_best_full_eval_20260606

best-checkpoint metrics:
  valid200 oracle_primary = 0.815
  valid200 selected_primary = 0.125
  calibration200 oracle_primary = 0.715
  calibration200 selected_primary = 0.095

calibration references:
  rank-1 candidate baseline = 0.070
  shallow qtype logistic top128 = 0.175
  Qwen3-8B-FP8 semantic selector top128 = 0.445
  top128 oracle = 0.715

batch-size feasibility:
  top128 batch_size=4 smoke OOMs inside frozen CoLA VAE slow attention
  before ranker backward. Current full-context top128 training should assume
  batch_size=1 unless gradient accumulation, latent caching, or a more
  memory-efficient interaction design is introduced.

decision:
  This confirms the top128 frozen-VAE ranker path is feasible and trainable,
  but the current independent-pooling architecture is below the shallow
  selector and does not pass Phase A. Do not scale this exact architecture by
  steps alone. The next ranker must introduce stronger context-candidate
  interaction and/or teacher-distillation targets.
```

Late-interaction ranker update:

```text
design basis:
  The independent-pooling ranker is analogous to a single-vector dual encoder.
  ColBERT-style late interaction keeps token-level vectors and applies MaxSim
  matching, which better preserves local answer-bearing evidence. This is a
  natural next step for CoLA latent token/block sequences.

implementation:
  train_p2_phase_a_cola_latent_candidate_ranker.py supports
  --interaction-mode late_maxsim and --interaction-dim.
  It also records feature_schema_version and evaluator compatibility for older
  pooled checkpoints.

top128 step500 late_maxsim:
  output:
    outputs/p2_phase_a_cola_latent_candidate_ranker/
    musique_top128_step500_latemaxsim_seed20260606_20260606
  SwanLab:
    myo0ofc3aivcxmxijbs20
  best step:
    460
  best valid200:
    selected_primary = 0.100
    oracle_primary = 0.815
  best calibration200:
    selected_primary = 0.120
    selected_token_f1 = 0.1568
    oracle_primary = 0.715

comparison:
  pooled schema-v2 best calibration200 = 0.095
  late_maxsim best calibration200 = 0.120
  shallow qtype logistic top128 = 0.175
  Qwen semantic selector top128 = 0.445
  top128 oracle = 0.715

decision:
  Late interaction is directionally better than independent pooling but still
  not enough. It does not pass Phase A and must not unlock held-out or Phase E.
  Next repair should prioritize Qwen semantic selector distillation and/or
  cached-latent cross-candidate interaction, not just more steps of the same
  architecture.
```

Next allowed repair directions:

```text
1. Stop expanding both failed paths:
   cap96 full-info continuation
   support-only Flow-Matching curriculum
   answer-support generative target
   rule-candidate + shallow-selector reranking

2. Continue with semantic answer-selection distillation:
   use the completed nonheldout train10k Qwen semantic selector artifact and
   the candidate-constrained short-answer SFT data to train a CoLA-native
   candidate answer selector/ranker. The first frozen-VAE independent-pooling
   ranker is below shallow features; late_maxsim helps but remains below
   shallow qtype. The next version should use Qwen teacher distillation and/or
   cached-latent cross-candidate interaction. The online input must stay
   question/evidence/candidate-only; gold, aliases, teacher correctness, and
   scorer output remain offline supervision only.

3. Keep downstream solver100 screens as the checkpoint-selection evidence.
   Flow-Matching valid loss alone is insufficient.

4. Held-out remains locked. Do not use held-out to choose prompts,
   checkpoints, objectives, or thresholds. Do not lower the locked floor.
```

## 1. 不可违反的实验前提

### 1.1 任务必须天然需要通信

普通单问答不能再硬拆成 multi-agent 主实验。P2 主 benchmark 必须满足至少一种：

```text
partial-information / evidence-split reasoning
distributed state integration
role-specific expert aggregation
planner-coder-tester-reviewer workflow
```

如果一个样本被单个 agent 直接回答完就结束，它只能作为 solver capability
诊断，不能支撑 agent-to-agent communication claim。

### 1.2 CoLA 架构适合作为 substrate，当前权重不自动适合新任务

CoLA 的价值是：

```text
Text VAE maps text <-> continuous latent sequence.
Block-causal DiT operates on latent sequence.
Same official Cola A/B gives a shared latent substrate.
```

但当前 frozen official CoLA checkpoint 不应被假设已经满足新 MAS benchmark：

```text
它不是 instruction-tuned / RLHF-aligned solver。
它对 prompt template / parser / task format 敏感。
Family 1 和 native official8 audit 已经显示当前 admitted_tasks=[]。
```

因此，若新 benchmark 上 CoLA single solver 接近 floor，结论不是 latent
communication 失败，而是 substrate capability / interface 尚未建立。

执行含义：

```text
Phase C:
  先用 capable text agents 验证 benchmark/protocol，不依赖 CoLA 解题能力。

Phase A:
  只在 Phase C 通过后，才训练或适配 CoLA 的 task-format / role interface。

Phase E:
  只有 CoLA Single 和 CoLA Role TextMAS 都过 gate，才比较 CoLA TextMAS
  与 CoLA LatentMAS。
```

### 1.3 先证明 text MAS benchmark 合理，再比较 latent

每个候选任务必须先建立：

```text
Single capable solver baseline
Role TextMAS baseline
no-message / shuffled-message / wrong-message controls
stable parser / scorer
locked calibration / held-out split
no gold / scorer / selected_prediction leakage
```

这些不成立时，不能进入 CoLA LatentMAS 主比较。

### 1.4 Agent B 的输入边界必须明确

允许两种协议，必须在 manifest 中写明：

```text
message_only:
  Agent B / Solver receives only upstream agent message or latent packet.
  适合测试 self-contained communication。

public_context_plus_message:
  Agent B / Solver receives public task context plus upstream message/latent.
  public_context 只能包含公开问题、角色说明或任务规则；
  private evidence / hidden state 必须只从 upstream message/latent 来。
```

禁止：

```text
Agent B 看到 gold answer。
Agent B 看到 scorer output。
Agent B 看到 selected_prediction / oracle block。
把 Agent A decoded replay tokens 直接拼进 final answer 给 scorer。
在 latent 条件下让 scorer 看到 A 的可读文本，但 text 条件没有同等预算。
把 Agent A 的文本/latent 当作 final answer 直接评分，而不让 Agent B 生成。
```

最小合格数据流：

```text
TextMAS:
  Agent A private observation -> A text message
  Agent B / final Solver receives allowed public context + A text message
  Agent B generates final answer
  scorer evaluates only Agent B final answer

LatentMAS:
  Agent A private observation -> A latent packet/state
  Agent B receives allowed public context + A latent packet/state
  Agent B generates final answer
  scorer evaluates only Agent B final answer

Diagnostic-only:
  decoded answer handoff
  replay-visible text
  legacy_all_visible scoring
  official8 message_only channel diagnostics
```

## 2. Phase C: true MAS benchmark/protocol validation

### C0. 安全准备

允许的工作：

```text
文档整理
schema / manifest / scorer 草案
本地字段 dry inspection 工具
license / attribution 审计
leakage audit 工具
已有结果聚合
```

禁止的工作：

```text
不下载或构建 full benchmark 后直接跑模型
不看 held-out 样本内容做 prompt 修补
不启动训练
不创建 SwanLab run
不把 smoke test 写成科学结论
```

当前已有安全准备入口：

```text
docs/current/P2_Phase_C_Benchmark_Protocol_Preparation_2026-06-01.md
docs/current/P2_Phase_C_Data_Source_and_Runner_Design_2026-06-01.md
docs/current/P2_Phase_C_Data_Source_Field_License_Audit_2026-06-01.md
configs/p2_phase_c_manifest_schema.json
drla/scripts/validate_p2_phase_c_manifest.py
drla/scripts/build_p2_phase_c_manifest.py
drla/scripts/inspect_p2_phase_c_dataset_fields.py
drla/scripts/fetch_p2_phase_c_hf_rows.py
drla/scripts/prepare_p2_phase_c_evidence_split_qa_records.py
drla/scripts/build_p2_phase_c_control_inputs.py
drla/scripts/preflight_p2_phase_c_text_agent_run.py
drla/scripts/run_p2_phase_c_text_agents.py
drla/scripts/aggregate_p2_phase_c_text_agent_results.py
drla/evaluation/p2_phase_c_scorers.py
drla/scripts/audit_p2_phase_c_run_leakage.py
```

### C1. 数据源 dry inspection

优先级：

```text
1. MuSiQue
2. HotpotQA distractor
3. 2WikiMultiHopQA
4. scalable distributed-state synthetic tasks
5. code workflow second batch
```

dry inspection 必须只输出字段摘要：

```text
source revision / URL / hash
license / attribution requirements
sample keys
answer / aliases fields
evidence / support fields
split names
private-view shardability
possible shortcut risk
```

不允许在 dry inspection 阶段调 prompt、挑样本或看 held-out 具体内容。

### C2. 构建 Phase C manifest

每条样本必须包含：

```text
sample_id
task_family
split: calibration | heldout | test
public_context
private_views for each role
expected_answer / aliases, offline only
scorer config, offline only
source license / attribution metadata
leakage_audit booleans
```

每个 benchmark family 至少要求：

```text
calibration >= 200
heldout >= 800
test 或 locked heldout 不用于 prompt repair
overlap = 0
split hash recorded
```

小样本只允许验证 builder/parser/scorer，不允许推出架构成败结论。

### C3. Capable TextMAS validation

先使用 capable text agents，而不是 frozen CoLA，验证 benchmark/protocol。

必须跑的条件：

```text
single_q_only:
  solver only sees public question.

single_full_info:
  solver sees question + all evidence/state shards.

textmas_matched:
  role agents see private shards and send messages to final Solver.

textmas_no_message:
  final Solver sees no upstream message under the same contract.

textmas_shuffled_message:
  final Solver receives message from another sample.

textmas_wrong_evidence_or_wrong_shard:
  one upstream agent receives irrelevant or mismatched private state.

textmas_compressed_state:
  upstream message is constrained to compact typed state.
```

最低准入：

```text
single_full_info > single_q_only
textmas_matched > textmas_no_message
textmas_matched > shuffled / wrong-message controls
parseable_rate >= 0.95
failure taxonomy 不是 parser/format failure 主导
matched-vs-control paired CI lower bound > 0
```

Phase C 只证明 benchmark/protocol 合理，不证明 CoLA latent communication。

当前已准备好的 calibration candidates：

```text
HotpotQA calibration controls:
  200 samples, 7 conditions, 1400 online-input rows.

MuSiQue calibration controls:
  200 samples, 7 conditions, 1400 online-input rows.

Both:
  manifest/control leakage audits pass with zero errors.
  warnings mostly mean answer string appears naturally in evidence text.
  These warnings are shortcut-risk signals, so matched-vs-control gates are
  mandatory before admitting either benchmark.
```

capable text-agent real run 前必须先做 preflight：

```text
drla/scripts/preflight_p2_phase_c_text_agent_run.py

It must report:
  ready_to_run_model = true
  OPENAI_API_KEY or chosen compatible key is set
  OPENAI_MODEL or --model is set
  condition counts match expected control package
  estimated call budget is recorded
```

若 preflight 未通过，只能运行 `--selfcheck --provider mock_selfcheck` 验证
路由/计分逻辑，不能生成实验 claim。

### C4. Phase C 产物

通过后必须保存：

```text
manifest.json
samples.jsonl
source_registry.json
split_report.json
scorer_report.json
leakage_audit.json
generation_config.json
generations.jsonl
task_summary.csv
summary.json
metrics.jsonl
failure_taxonomy.csv
```

所有 pure eval / generation / audit 产物 local-only，不上 SwanLab。

## 3. Phase A: CoLA substrate / interface adaptation

触发条件：

```text
Phase C 至少有一个 admitted benchmark/protocol。
或者用户明确要求某个 benchmark 不可替换。
```

目标：

```text
让 CoLA 在 locked benchmark 上具备基本 solver 与 role interface 能力，
再进入 latent-vs-text comparison。
```

### A1. CoLA Single Solver adaptation

训练对象：

```text
task-format adapter / LoRA / lightweight interface module
```

训练目标：

```text
CoLA single solver can solve single_full_info and/or public benchmark format
above floor on validation.
```

适配内容优先级：

```text
1. prompt/template/parser/interface adapter。
2. lightweight LoRA/adapter for task formatting and evidence integration。
3. only if needed, role-message compression / state-interface distillation。
```

禁止把 Phase A 写成“提高 CoLA 官方 8 benchmark accuracy”。Phase A 的目的
是让 CoLA 能在 Phase C locked benchmark 上承担 same-substrate A/B 角色。

要求：

```text
CUDA only
SwanLab cloud
metrics.jsonl
valid_interval <= 10 step
best_checkpoint.pt
last_checkpoint.pt
no held-out prompt repair
```

2026-06-05 Phase A smoke 诊断：

```text
runner:
  drla/scripts/run_p2_phase_c_text_agents.py --provider cola_dlm

smoke artifact:
  outputs/p2_phase_c_text_agent_runs/
  musique_cola_dlm_v1_strict_wrong_heldout_smoke14_20260605

scope:
  held-out first 2 samples * 7 conditions = 14 rows
  local-only eval, no training, SwanLab disabled

result:
  all seven conditions score 0/2
  raw outputs show JSON/chat-style prompt drift and malformed answer strings

interpretation:
  official frozen CoLA can be called under the Phase C runner, but the current
  Phase C JSON/chat prompt is not a working CoLA interface.
  This is an interface/task-format adaptation signal, not a latent
  communication failure.
```

Phase A 下一步：

```text
Use calibration split only to build a CoLA task-format / role-interface
adapter or prompt/template adapter.
Then rerun CoLA Single Solver and CoLA Role TextMAS gates on the locked
MuSiQue protocol.
Do not modify held-out prompt/parser/control definition.
```

### A2. CoLA Role TextMAS interface adaptation

训练对象：

```text
role prompt / state interface / compact message adapter
```

目标：

```text
CoLA Role TextMAS passes the same capability gate that capable TextMAS passed.
```

关键边界：

```text
如果 Role TextMAS 不过 gate，不允许开始 LatentMAS main table。
如果 only Single Solver 过 gate，最多报告 substrate adaptation 中间结果。
```

### A3. Latent interface adaptation

先跑 no-fuser / direct same-substrate handoff：

```text
matched latent packet
wrong-sample latent
wrong-block latent
shuffled latent
noise / metadata-only controls
```

只有满足以下条件才训练 receiver/fuser：

```text
TextMAS 已过 gate。
no-fuser latent 明显弱于 TextMAS。
matched latent 与 corrupt controls 有可解释差异。
错误来自 interface mismatch，而不是 solver floor 或 scorer/parser。
```

receiver/fuser 的 claim 必须写成：

```text
adapted-CoLA latent communication
```

不能写成：

```text
frozen official CoLA no-fuser latent communication solved
```

## 4. Phase E: paper-level CoLA TextMAS vs LatentMAS

触发条件：

```text
Phase C admitted benchmark exists.
CoLA Single Solver passes gate.
CoLA Role TextMAS passes gate.
locked split exists.
leakage audit passes.
```

主表条件：

```text
Single CoLA Solver
CoLA TextMAS
CoLA LatentMAS no-fuser
CoLA LatentMAS with receiver/fuser, only if Phase A3 triggered
latent corrupt controls
text corrupt controls
```

主指标：

```text
accuracy / task score
paired delta vs Single
paired delta vs TextMAS
matched vs corrupt-control delta
parseable / nonempty / format adherence
token count
latent block count
wall-clock cost
confidence interval
failure taxonomy
```

可发表 claim 分级：

```text
Readable:
  matched latent > corrupt controls.

Useful:
  matched latent > no-message / none under paired comparison.

Competitive:
  latent is Pareto competitive with text in quality/cost.

True MAS:
  task naturally requires communication and ablation hurts.
```

任何表格必须明确区分：

```text
capable external text agents
frozen official CoLA
adapted CoLA
no-fuser latent
receiver/fuser latent
```

## 5. 当前不再做的事情

```text
1. 不继续 Branch B Family 1 held-out gate。
2. 不把 official8-only solver-to-solver message_only 当 P2 主实验。
3. 不在 admitted_tasks=[] 时跑 text-vs-latent main table。
4. 不在 Role TextMAS baseline 未成立时训练 latent fuser/adapter。
5. 不把 low accuracy 解释成 latent communication failure。
6. 不用几十条 smoke test 证明或否定架构。
7. 不让 eval-only 脚本上 SwanLab。
8. 不把 decoder/gold/scorer/oracle 字段作为在线 receiver input。
9. 不把 Agent A decoded output 直接拼进 final answer 给 scorer。
```

## 6. 下一步实施清单

按顺序执行：

```text
Step 1:
  完成本地字段 dry-inspection 工具，只读已下载或允许预览的数据。
  验证输出 field_summary.json / metrics.jsonl。
  当前入口:
    drla/scripts/inspect_p2_phase_c_dataset_fields.py

Step 2:
  用 MuSiQue + HotpotQA 各构建一个 calibration-only manifest draft。
  运行 manifest validator 和 leakage schema check。
  当前入口:
    drla/scripts/fetch_p2_phase_c_hf_rows.py
    drla/scripts/prepare_p2_phase_c_evidence_split_qa_records.py
    drla/scripts/build_p2_phase_c_manifest.py

Step 3:
  在 calibration split 上跑 capable Single / TextMAS / controls。
  只记录 local metrics，不上 SwanLab。
  执行前必须先构造并通过审计:
    drla/scripts/build_p2_phase_c_control_inputs.py
  执行前必须先跑 preflight:
    drla/scripts/preflight_p2_phase_c_text_agent_run.py
  模型运行入口:
    drla/scripts/run_p2_phase_c_text_agents.py
  当前机器若没有 OPENAI_API_KEY/OPENAI_MODEL 或兼容端点，不允许用 mock
  provider 伪造实验；只能运行 selfcheck 验证路由和计分逻辑。
  允许使用真实 local_transformers provider 做工程预检或 calibration 运行，
  但必须明确模型路径、GPU、样本规模和它是否足够作为 capable-agent baseline。
  小规模 local smoke 只能验证 wiring，不能作为 benchmark admission。
  结果聚合与准入:
    drla/scripts/aggregate_p2_phase_c_text_agent_results.py

Step 4:
  选择通过 Phase C gate 的 benchmark/protocol。
  锁定 prompt、parser、scorer、split hash。

Step 5:
  进入 Phase A，训练 CoLA task-format / role interface adapter。
  训练必须 GPU + SwanLab cloud + best/last checkpoint。

Step 6:
  通过 CoLA Single + Role gate 后，进入 Phase E 主表。
```

若 Phase C evidence-split QA 控制项显示 question-only 或 shuffled/wrong
message 也很强，则不要为了保留该数据集而降低标准。直接转入
distributed-state v0 或 code-workflow v0 的非 toy benchmark 构造，并保留失败
报告。

每个 step 完成后必须写：

```text
result summary
artifact paths
failed cases
leakage audit status
下一步是否满足 gate
```

## 7. 文档优先级

当前执行优先级：

```text
1. 本文
2. P2_Post_Family1_Complete_Execution_Plan_2026-06-01.md
3. P2_Phase_C_Benchmark_Protocol_Preparation_2026-06-01.md
4. P2_Phase_C_Data_Source_and_Runner_Design_2026-06-01.md
5. P2_Phase_C_Data_Source_Field_License_Audit_2026-06-01.md
```

历史文档仍可引用证据，但不能覆盖本文：

```text
P2_Next_Phase_Execution_Plan_2026-06-01.md
P2_D4_Branch_Decision_Audit_2026-06-01.md
P2_Branch_B_Execution_Plan_2026-06-01.md
P2_Branch_B_Calibration_Report_2026-06-01.md
P2_Official8_Native_Alignment_Audit_2026-06-01.md
```
