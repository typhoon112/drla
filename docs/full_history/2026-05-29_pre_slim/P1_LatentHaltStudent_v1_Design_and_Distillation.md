# P1 LatentHaltStudent-v1 Design and Distillation Plan

> 状态：P1 实验笔记。本文保留 LatentHaltStudent-v1 的设计、蒸馏和消融记录；当前 P2 主线以 2026-05-29 P2 实施文档为准。

更新时间：2026-05-26

本文档整理当前 DRLA 工作区状态、P0 实验进展、P1 阶段的最优候选模型设计、判别器架构、训练策略，以及如何使用 P0 decoder-probed teacher 进行蒸馏。核心原则是：保留 P0 作为 teacher / upper-bound / safety baseline，同时把最终在线 halt 决策迁移到只读 latent/process trajectory 的 P1 student。

## 1. 当前工作区状态

当前工作区位于：

```text
/data1/luyifei/drla
```

Git 根目录是：

```text
/data1/luyifei
```

当前 `/data1/luyifei/drla` 不是一个整理好的 Python package，而是一个实验工作区。主要结构如下：

```text
configs/     环境依赖、SwanLab 配置
docs/        文档系统、实验状态、历史计划和报告
drla/        实验脚本与轻量 package 壳
models/      本地 HuggingFace 权重与 config
outputs/     trace、frontier、checkpoint、summary 等实验产物
scripts/     环境激活、报告渲染、绘图工具
tests/       少量测试入口
archive/     历史 Stage/GSM8K/custom-prior 代码和产物
```

需要注意：

1. `drla/models/` 和 `drla/training/` 目前基本是空壳。
2. 已实现的 trainable halt / risk 模型在 `drla/scripts/` 内部，而不是独立模型包中。
3. `configs/` 目前不是实验模型配置系统，主要是 conda/pip/SwanLab 配置。
4. P1 的 `LatentHaltStudent-v1` 已实现并完成多轮 same-split、leave-one-task-out、结构消融和 P0-teacher 蒸馏诊断；最新结果见第 5 节和 `docs/CURRENT_EXPERIMENT_STATUS.md`。

## 2. 当前实验阶段总览

当前主线已经从旧的自建 latent prior / GSM8K MVP 转为 official Cola block-wise readiness / halt：

```text
official Cola baseline
-> block-wise rollout trace
-> per-block decoder probe
-> oracle readiness frontier
-> Phase P0 decoder-probed readiness baseline
-> Phase P1 decoder-as-teacher LatentHaltStudent-v1
-> adaptive halt accuracy-cost frontier
```

历史自建 Stage B/C small-prior 和 GSM8K 诊断代码已经归档：

```text
/data1/luyifei/drla/archive/2026-05-24-legacy-stagebc-code
/data1/luyifei/drla/archive/2026-05-24-legacy-stagebc-artifacts
```

这些归档只保留复现价值，不作为当前主线架构有效性的主要证据。

## 3. P0 当前最优模型与实验进展

严格来说，当前真正完成并可称为最优的模型/策略属于 P0，而不是 P1。

P0 当前最优安全成本点是：

```text
joint-readiness riskcap04
```

对应 cross-seed summary：

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_joint_readiness_riskcap04_cross_seed_20260524/summary.json
```

关键结果：

```text
任务集：official8
协议：full prepared split, batch_size=12, seed66/67/68
weighted micro accuracy：约 21.596%
平均 block 使用量：2.118 / 4
相对 fixed-final observed loss：0
相对 prediction-stability observed loss：0
相对 prediction-stability 节省 block：约 0.394
相对 fixed-final 节省 block：约 1.882
```

P0 的实际 trainable 组件包括两个脚本内 MLP。

### 3.1 ReadinessModel

源码：

```text
/data1/luyifei/drla/drla/scripts/train_cola_readiness_model.py
```

功能：

```text
输入：
  raw latent block 或 zero latent
  process/probe/stability features
  optional task one-hot

输出：
  readiness_logits
  correctness_logits
  future_gain
```

当前支持的 `signal_mode`：

```text
full
process_only
process_no_task
latent_only
```

需要注意：`process_no_task` 会置零 raw latent 和 task one-hot，主要依赖 process/probe/stability features。这是 P0 cross-task transfer 中常用的设置，但仍不是最终 latent-only P1，因为其中很多 features 来自 decoder/text/probe。

### 3.2 ContinuationRiskModel

源码：

```text
/data1/luyifei/drla/drla/scripts/train_cola_continuation_risk_model.py
```

功能：

```text
输入：
  process / probe / prediction-shape features
  optional task one-hot

输出：
  continuation_risk_prob
```

当前主要 target：

```text
strict_prefix:
  当前 task-scored prediction 是否是 final prediction 的 strict prefix

prediction_change:
  当前 task-scored prediction 是否会变成 prediction-stability reference 之外的答案
```

`prediction_change` 是当前更重要的 P0 risk target，因为它能覆盖 MMLU/RACE 等多选任务中的单字母翻转，而 `strict_prefix` 更偏自由文本 prefix/incomplete answer 风险。

### 3.3 Risk-Gated Halt Policy

源码：

```text
/data1/luyifei/drla/drla/scripts/eval_cola_risk_gated_halt.py
```

该 policy 不是第三个 trainable network，而是一个 validation-calibrated sequential decision rule：

```text
for each block b:
  if readiness_prob >= readiness_threshold:
    if continuation_risk_prob < risk_threshold:
      if content / fragment / single-choice / uncertainty guards pass:
        halt at b
  if prediction_stability_reached:
    halt at b
force stop at B_max
```

`riskcap04` 的含义：

```text
risk_threshold_end = 0.4
```

它不是模型结构，而是校准约束。作用是防止 validation 选择过高 risk threshold，例如 0.9，从而在 held-out task 上放过 prefix/completion failure。

## 4. 为什么 P0 不是最终目标

P0 的结论很强，但它不是最终 agent-to-agent latent communication 的在线策略。

原因：

1. P0 training labels 来自 decoder output、task scorer、prediction-stability reference、future blocks 和 official correctness。
2. P0 readiness / risk feature fields 包含 EOS/im_end/stop probe、decoded text dynamics、scored prediction dynamics、prediction length/shape、decode length 等 decoder/text-derived features。
3. P0 eval policy 还直接使用 `scored_prediction`、contentful guard、fragment guard、single-choice stability guard 和 `prediction_stability_reached(row)`。

因此，P0 应被解释为：

```text
decoder-probed / text-stability-supervised risk-gated halt baseline
```

而不是：

```text
final agent-to-agent latent-only halt policy
```

P0 的保留价值是：

1. 证明 official Cola latent rollout 中存在可由 decoder probe 暴露的 answer-readiness 结构。
2. 生成 dense teacher targets。
3. 提供当前最强 safety/cost upper bound。
4. 提供 P1 student 必须对齐或解释差距的 baseline。
5. 提供样本级失败诊断集。

## 5. P1 当前最优模型状态

P1 已经有可复现实验结果。当前主模型方案仍是：

```text
decoder-as-teacher, LatentHaltStudent-v1
```

P1 的核心目标：

```text
训练期：
  使用 P0 decoder/scorer/stability/future-gain teacher 构造监督

推理期：
  只读取 latent prefix + process/budget features
  不读取 decoded text
  不读取 scored_prediction
  不读取 prediction-stability
  不读取 EOS/im_end probability
  不读取 official correctness
```

也就是说，P1 不是丢弃 decoder 信息，而是把 decoder 从在线输入变成离线 teacher。

当前最强 P1 路线是：

```text
d64_pma4_last + process_token + full process features
readiness_target_mode = p0_teacher_action
use_completion_risk = true
```

训练产物：

```text
train root:
  /data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_p0teach_action_completionrisk_20260525/leave_squad_out

SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/wkirim527d08abt606u8e

best checkpoint:
  step 2420
  valid mean AUROC(readiness-action / prediction-change / completion-risk) = 0.9897
```

SQuAD LOTO 本地 student-only eval：

```text
eval root:
  /data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_p0teach_action_completionrisk_refined_textaudit_20260525/leave_squad_out_eval_squad_all

selected thresholds:
  readiness = 0.1
  prediction_change risk = 0.15
  completion_risk = 0.15

SQuAD held-out all split:
  accuracy = 22.034%
  avg_blocks = 2.049 / 4
  losses_vs_final = 0 / 10570
  losses_vs_prediction_stability = 0 / 10570
  text_mismatch_vs_final = 841 / 10570
```

解释：

1. `p0_teacher_action + completion_risk` 是目前第一个在 SQuAD LOTO 上明显优于 prediction-stability cost 的 P1 student：prediction-stability 是 `2.509/4` blocks，而该模型是 `2.049/4` blocks。
2. `841` 个 text mismatch 全部发生在 fixed-final 本来也错误的样本上；正确样本没有被提前停坏，所以 correctness frontier 是强正结果。
3. 这还不是 agent latent communication 的终点，因为 answer-text identity / latent answer stability 仍未被充分约束。若要求校准集 zero loss + zero text mismatch，并显式加入 fixed-final 候选，策略会退回 `4.0/4` blocks。
4. 下一阶段不能只继续调阈值，应把 P0 action distillation 扩展到 all-task / multi-seed，并加入 latent-level answer identity 或 sequence-level stability 目标。

### 5.1 Seed68 All-Task Action Distillation Update

seed68 all-8 LOTO 已完成：

```text
train root:
  /data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_p0teach_action_completionrisk_20260525

primary aggregate:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_seed68_p0teacher_action_completionrisk_20260525/summary.json

mismatch-cap3 aggregate:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_seed68_p0teacher_action_completionrisk_mismatchcap3_20260525/summary.json

strict fixed-final fallback aggregate:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_seed68_p0teacher_action_completionrisk_zero_mismatch_with_final_20260525/summary.json
```

结果：

```text
primary:
  accuracy = 21.569%
  avg_blocks = 2.089 / 4
  losses_vs_prediction_stability = 16 / 49019
  text_mismatch_vs_prediction_stability = 949 / 49019

mismatch-cap3:
  accuracy = 21.586%
  avg_blocks = 2.157 / 4
  losses_vs_prediction_stability = 8 / 49019
  text_mismatch_vs_prediction_stability = 701 / 49019

strict zero-mismatch + fixed-final candidate:
  accuracy = 21.600%
  avg_blocks = 3.522 / 4
  losses_vs_final = 0 / 49019
  losses_vs_prediction_stability = 1 / 49019
```

seen-task all-8 训练诊断：

```text
train root:
  /data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/official8_full_b64_bs12_seed68_d64_pma4_p0teach_action_completionrisk_alltasks_20260525

SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/7klq88o6qe9cbjcgaod98

mismatch-cap3 eval:
  /data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/official8_full_b64_bs12_seed68_d64_pma4_p0teach_action_completionrisk_alltasks_mismatchcap3_20260525/test
```

seen-task test 结果：

```text
accuracy = fixed-final = prediction-stability = 22.028%
avg_blocks = 1.928 / 4
losses_vs_prediction_stability = 0 / 4912
text_mismatch_vs_prediction_stability = 5 / 4912
```

解释：

1. seen-task 结果证明 P1 latent/process-only student 能学到 decoder teacher 暴露出的 answer-readiness / completion-risk 信号。
2. LOTO primary 和 mismatch-cap3 仍有 losses，说明跨任务 calibration 和 completion semantics 还没有解决。
3. Story Cloze 的 loss 样本主要是当前 block decoded answer 是正确答案前缀但未完成；目标任务校准会把 `contentful_threshold` 提到 `0.9` 并在 Story Cloze test 上消除 loss。
4. Target-task calibration 的样本量诊断显示：16/32/64/128 个目标 valid 样本分别得到 `7/5/2/0` 个 held-out test loss（128-shot 指 subsample seed `20260525`），平均 blocks 约 `1.95-1.99/4`。但 128-shot 换 5 个 subsample seed 后，仍有 3 个 seed 在同一个 MMLU 样本 `mmlu::5615` 上各掉 1 个 loss；该样本的 block 2 decoded answer 为空，final/prediction-stability 为 `D` 且正确。
5. 因此下一步不是再盲扫 d_model。先把 target-calibration 的选择规则从单纯 `min_blocks` 改成带 boundary-risk cost 的选择性预测问题；随后继续把 answer identity / completion stability / empty-answer boundary 作为 latent-level 监督目标。

### 5.2 Target Calibration And Contentful Boundary Update

新增的 eval 诊断：

```text
sample-cap sweep:
  /data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_p0teach_action_completionrisk_targetcal_caps_20260525/cap_sweep_aggregate.json

128-shot subsample robustness:
  /data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_p0teach_action_completionrisk_targetcal_cap128_subseeds_20260525/cap128_subseed_aggregate.json
```

关键结论：

```text
target-valid cap 16:  loss = 7 / 4912, avg_blocks = 1.949 / 4
target-valid cap 32:  loss = 5 / 4912, avg_blocks = 1.958 / 4
target-valid cap 64:  loss = 2 / 4912, avg_blocks = 1.966 / 4
target-valid cap 128: loss = 0 / 4912, avg_blocks = 1.987 / 4  (subsample seed 20260525)
```

128-shot target calibration 很强，但还不是安全结论：5 个 subsample seed 中只有 2 个 seed 全任务 zero-loss，另外 3 个 seed 都在 `mmlu::5615` 出现同一个 empty-answer 早停边界。硬性提高 `contentful_threshold` 可以挡住这个样本，但代价过大：MMLU-only `contentful_threshold >= 0.35` 需要约 `3.34/4` blocks；`contentful_threshold >= 0.5` 对多个任务退化到 `3.5-3.9/4` blocks。

MMLU-focused `contentfulw1` 训练也已完成：

```text
train root:
  /data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_p0teach_action_completionrisk_contentfulw1_20260525/leave_mmlu_out

SwanLab:
  https://swanlab.cn/@Lyfff/drla-mvp/runs/l1o11p6a4dbviyg4i72ba
```

它把 `contentful_loss_weight` 提到 `1.0`，并用 `readiness_prediction_change_completion_contentful_mean_auroc` 选择 checkpoint；best step `2150`，held-out test `contentful_auroc=0.9318`。但同样的 MMLU 128-shot target calibration 仍在 `4/5` subsample seed 上损失 `mmlu::5615`，所以“简单加大 contentful loss”应暂时拒绝为主线修复。

随后做了两个更窄的 MMLU boundary head 诊断：

```text
summary:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_mmlu_boundary_head_diagnostics_20260525/summary.json

empty-answer risk train:
  /data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_p0teach_action_completionrisk_emptyrisk_20260525/leave_mmlu_out
  SwanLab: https://swanlab.cn/@Lyfff/drla-mvp/runs/92wq5v4knddw7zqx0pd8b

answer-format risk train:
  /data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_p0teach_action_completionrisk_formatrisk_20260525/leave_mmlu_out
  SwanLab: https://swanlab.cn/@Lyfff/drla-mvp/runs/ukv4l7py57ujlfyeywo51
```

结果：

```text
empty-answer risk:
  test AUROC = 0.9373
  free calibration: 5 / 5 subsample seeds lose mmlu::5615, mean blocks = 1.933 / 4
  forced empty_answer_risk <= 0.3: 0 / 5 losses, mean blocks = 3.185 / 4

answer-format risk:
  test AUROC = 0.9409
  free calibration: 5 / 5 subsample seeds lose mmlu::5615, mean blocks = 1.776 / 4
```

结论：这两个 rare-event heads 都能学习 decoder teacher target，但旧的 threshold selector 会为了 valid zero-loss + min blocks 把它们的阈值选到 `1.0`，等价于关闭 gate。强制打开 gate 又比 prediction-stability 还贵。因此它们应作为诊断/特征候选保留，不作为主线 halt 修复。

### 5.3 Boundary-Risk Calibration Update

新增的 eval-only 校准选择器：

```text
--calibration-boundary-risk-penalty 0.2
selection_score = avg_blocks + penalty * boundary_risk_slack
boundary_risk_slack = risk_threshold + completion_risk_threshold
                    + optional empty_answer_risk_threshold
                    + optional answer_format_risk_threshold
```

设计动机：旧选择器在 valid zero-loss/mismatch-cap 约束下继续最小化 `avg_blocks`，容易选择过宽的 risk-side 阈值；这在 128-shot target calibration 中表现为同一个 MMLU 空答案边界样本 `mmlu::5615` 反复泄漏。新选择器仍然保持在线 latent/process-only，不给模型增加 decoder 输入，只是在离线 valid calibration 中偏好更保守的 risk-side gate。

主要结果：

```text
MMLU 128-shot target calibration, 5 subsample seeds:
  old min-block selector: 3 / 5 seeds lose mmlu::5615, mean blocks = 2.014 / 4
  boundary penalty 0.2:  0 / 5 losses, 0 mismatches, mean blocks = 2.038 / 4
  prediction-stability: mean blocks = 2.584 / 4

official8 seed68 LOTO, 5 subsample seeds, 40 eval summaries:
  accuracy = 22.048% micro
  losses_vs_prediction_stability = 0
  avg_blocks = 2.005 / 4
  block saving vs prediction-stability = 0.496
  text mismatches vs prediction-stability = 326
```

Artifacts:

```text
MMLU-only:
  /data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_p0teach_action_completionrisk_mmlu_targetcal_cap128_boundarypen02_subseeds_20260525/mmlu_targetcal_cap128_boundarypen02_subseed_summary.json

official8 aggregate:
  /data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_p0teach_action_completionrisk_targetcal_cap128_boundarypen02_subseeds_20260525/cap128_boundarypen02_subseed_aggregate.json
```

解释：这个结果把 P1 当前最佳点从“128-shot target-cal 仍有少数 MMLU 边界 loss”推进到“同一 seed68 LOTO / 5 个 calibration subsample seed 下 zero-loss，且成本仍明显低于 prediction-stability”。但它仍不是最终 agent latent communication 证据，因为 calibration label 和 mismatch audit 来自 decoder/scorer，且 SQuAD/HellaSwag 仍有大量 text mismatch。下一阶段仍应显式建模：

```text
empty-answer risk
answer-identity / option-identity stability
strict prefix / incomplete free-form answer boundary
multi-seed calibration robustness
```

这些目标仍来自 decoder teacher，但在线输入保持 latent/process-only；目标是让学生学到 latent 与 decoder answer-boundary 之间的映射，而不是推理时调用 decoder。

### 5.5 Answer-Identity Action/Halt 消融（2026-05-25）

为避免继续把 P0 teacher 的单一阈值策略当成最终目标，P1 增加了两个直接来自 answer identity 的 teacher target：

```text
answer_identity_action:
  仅首次 decoded/scored answer identity 匹配 prediction-stability/final answer 的 block 为正例。

answer_identity_halt:
  从首次 answer identity 匹配开始，后续 block 全部为 halt-ready 正例。
```

两者仍是 decoder-as-teacher：训练标签来自离线 decoder/scorer/probe，但在线学生输入只包含 latent prefix、process features、block mask 和可选 task index，不包含 decoded text、prediction-stability、gold answer 或 official correctness。

完整同口径评估：

```text
Protocol:
  seeds = 66, 67, 68
  held-out tasks = official 8-task LOTO
  calibration subsamples = 20260525..20260529
  eval summaries = 120
  calibration = target-task valid cap128 + boundary-risk penalty 0.2
  eval SwanLab = disabled
```

结果：

```text
P0 teacher action + completion_risk:
  accuracy = 22.473%
  avg_blocks = 1.908 / 4
  losses_vs_prediction_stability = 55
  mismatches_vs_prediction_stability = 886

answer_identity_action + completion_risk:
  accuracy = 22.481%
  avg_blocks = 1.742 / 4
  losses_vs_prediction_stability = 47
  mismatches_vs_prediction_stability = 617
  remaining losses: lambada 32, squad 15

answer_identity_action + completion_risk + fine contentful grid:
  accuracy = 22.486%
  avg_blocks = 1.710 / 4
  losses_vs_prediction_stability = 47
  mismatches_vs_prediction_stability = 638
  interpretation: contentful threshold granularity saves a little cost,
                  but does not fix remaining boundary losses

answer_identity_halt + completion_risk, strict target-task calibration:
  accuracy = 22.498%
  avg_blocks = 1.824 / 4
  losses_vs_prediction_stability = 31
  mismatches_vs_prediction_stability = 699
  remaining losses: lambada 15, hellaswag 10, squad 1, story_cloze 5
```

Interpretation:

- The earlier `answer_identity_halt` aggregate that used train-task calibration is not comparable with the strict target-task action baseline; keep it only as a calibration diagnostic.
- `answer_identity_action` is cheaper and has lower text mismatch; strict `answer_identity_halt` has fewer correctness losses but costs more blocks and more mismatch.
- Fine contentful thresholds do not solve the remaining LAMBADA/SQuAD losses. The existing contentful head is useful as a weak answer-formedness signal, but not as a standalone gate.
- A diagnostic task-wise action/halt choice reaches `16` losses, `569` mismatches, and `1.832/4` blocks by using strict halt for LAMBADA/MMLU/OBQA/RACE/SIQA/SQuAD and action for HellaSwag/Story Cloze. This is only an upper-bound diagnostic, not a deployable policy, because it hard-codes task-level choices.
- A causal action->halt diagnostic was run to remove the task-hardcoding from the previous hybrid. It lets `answer_identity_action` propose a stop, then either accepts it or continues to the strict halt policy after that block. Online gate inputs are restricted to latent student heads and block number; decoded text/scorer/correctness/future blocks are not inference features.
- The diagnostic result is two-sided: valid-cost calibration selects `no_gate` because the valid split has zero action losses, while valid-safety calibration selects `always_defer_before_final` and gets zero test losses at `2.722/4` blocks. Test-sweep evidence shows the latent heads do carry boundary signal: `readiness_lt_0.7_b2` rescues `30/47` action losses with no introduced losses, reaching `17` losses, `454` final-text mismatches, and `1.838/4` blocks; `contentful_lt_0.7_b2` reaches zero losses at `2.454/4` blocks but defers too often.
- Learned action-to-halt gate v1 trains an MLP over action selected block and action student heads. It uses source train split group-stratified internal validation because source official-valid and target-valid are often zero-positive. All 8 LOTO runs used SwanLab cloud and best-checkpoint evaluation. Aggregate summary:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_learned_action_halt_gate_loto_seed20260526/summary.json

action:
  losses = 47
  mismatches = 612
  avg_blocks = 1.742 / 4

strict halt:
  losses = 31
  mismatches = 694
  avg_blocks = 1.824 / 4

learned gate v1, source-valid cost selection:
  losses = 0
  mismatches = 339
  avg_blocks = 2.255 / 4

learned gate v1, source-valid safety selection:
  losses = 0
  mismatches = 143
  avg_blocks = 2.703 / 4

learned gate v1, test cost-limited diagnostic:
  losses = 33
  mismatches = 522
  avg_blocks = 1.777 / 4
```

Interpretation: learned gate v1 proves the latent student heads can learn a safety boundary, but weighted BCE over rare positive labels is too conservative when calibrated for safety and not discriminative enough under a tight block budget. This matches the implementation plan's warning that dynamic halt should optimize a frontier/reward, not a plain stop-logit classification. The next model should learn a rare-event-aware latent answer-type / completion-boundary gate or mixture policy with explicit early-stop-wrong penalty, false-defer/block-cost penalty, and ranking/calibration on boundary samples, without using decoded text, scorer outputs, or task-hardcoded routing at inference.

Learned gate v2 adds cost-weighted BCE and chooses the best checkpoint by source-validation policy frontier instead of `valid/auprc`:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_learned_action_halt_gate_costw200_block5_policycost_loto_seed20260526/summary.json

source-valid cost selection:
  losses = 10
  mismatches = 315
  avg_blocks = 2.220 / 4

source-valid safety selection:
  losses = 0
  mismatches = 130
  avg_blocks = 2.722 / 4

test cost-limited diagnostic:
  losses = 25
  mismatches = 461
  avg_blocks = 1.782 / 4
```

v2 improves the tight-cost frontier over v1 (`33 -> 25` losses, `522 -> 461` mismatches), but it still does not dominate the hand-rule diagnostic or strict halt. The failure mode is now clearer: the binary rescue target is too sparse and does not teach a smooth utility surface.

A LAMBADA-only `utility_mse` diagnostic was run before any 8-task expansion. It trains on fallback-minus-action utility rather than only the binary rescue label:

```text
sigmoid-threshold utility_mse:
  SwanLab = aviy0kiujga36ka4oag9n
  summary = /data1/luyifei/drla/outputs/cola_action_halt_gate/official8_full_b64_bs12_loto_lambada_seed20260526_utilitymse_policycost_besteval/summary.json
  source-valid cost = 0 losses / 0 mismatches / 3.405 blocks
  tight-cost diagnostic = 23 losses / 45 mismatches / 2.438 blocks

raw-score utility_mse:
  SwanLab = 2680kl7lod603onsqgz3h
  summary = /data1/luyifei/drla/outputs/cola_action_halt_gate/official8_full_b64_bs12_loto_lambada_seed20260526_utilitymse_rawscore_policycost_besteval/summary.json
  source-valid cost = 0 losses / 0 mismatches / 3.323 blocks
  tight-cost diagnostic = 24 losses / 49 mismatches / 2.485 blocks

v2 LAMBADA reference:
  SwanLab = 5rvvs3njy67yez8zecx7w
  source-valid cost = 0 losses / 0 mismatches / 3.227 blocks
  tight-cost diagnostic = 15 losses / 28 mismatches / 2.484 blocks
```

Interpretation: raw utility score semantics are methodologically cleaner than sigmoid-threshold utility, but naive scalar utility regression still underperforms v2 on LAMBADA. Do not expand this exact `utility_mse` variant to all 8 tasks as a mainline result.

The follow-up `utility_pairwise` objective directly ranks `defer to halt` against `accept action`:

```text
utility_pairwise LAMBADA:
  SwanLab = s62d23sfr1vlc4hxhapih
  summary = /data1/luyifei/drla/outputs/cola_action_halt_gate/official8_full_b64_bs12_loto_lambada_seed20260526_utilitypairwise_policycost_besteval/summary.json
  source-valid cost = 0 losses / 0 mismatches / 3.250 blocks
  tight-cost diagnostic = 15 losses / 36 mismatches / 2.478 blocks

utility_pairwise SQuAD:
  SwanLab = 3clg9aw2x6t8rg89i1oiz
  summary = /data1/luyifei/drla/outputs/cola_action_halt_gate/official8_full_b64_bs12_loto_squad_seed20260526_utilitypairwise_policycost_besteval/summary.json
  source-valid cost = 10 losses / 272 mismatches / 2.163 blocks
  tight-cost diagnostic = 11 losses / 387 mismatches / 2.026 blocks

utility_pairwise LAMBADA, mismatch_penalty=1.0:
  SwanLab = rbws51gf6jki9n9hqax8d
  summary = /data1/luyifei/drla/outputs/cola_action_halt_gate/official8_full_b64_bs12_loto_lambada_seed20260526_utilitypairwise_mismatch1_policycost_besteval/summary.json
  source-valid cost = 0 losses / 0 mismatches / 3.337 blocks
  tight-cost diagnostic = 24 losses / 51 mismatches / 2.476 blocks
```

Interpretation: pairwise ranking is healthier than scalar utility MSE and can match v2 on LAMBADA tight loss, but it worsens text mismatch and does not transfer cleanly to SQuAD. Increasing the scalar mismatch penalty to `1.0` makes the policy more conservative without improving the tight frontier. Do not continue by simply raising the same penalty or expanding this variant to all 8 tasks. The next variant should prioritize frontier-aware calibration, rare-boundary validation construction, or richer latent/process interaction, with reward terms aligned to the implementation plan:

```text
utility(decision) =
  correctness_reward
  - block_cost
  - mismatch_or_instability_penalty
  - early_stop_wrong_penalty
```

The online features may stay latent-only; decoded text/scorer/correctness remain label/eval sources only.

The next calibration diagnostic tried rare-boundary internal validation instead of ordinary stratified train validation. This keeps the model and online features unchanged, but changes checkpoint/threshold evidence: all positive rescue groups are kept, boundary-negative groups are oversampled, and ordinary negatives are downsampled. Dry-runs stayed local-only; formal runs used SwanLab cloud.

```text
validation_source = train_boundary_stratified
tasks = lambada, squad, hellaswag

partial aggregate:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_learned_action_halt_gate_boundaryvalid_partial_seed20260526/summary.json

boundary-valid partial aggregate:
  action = 47 losses / 601 mismatches / 1.989 blocks
  source-valid cost = 10 losses / 258 mismatches / 2.640 blocks
  source-valid cost-limited = 10 losses / 430 mismatches / 2.186 blocks
  tight test-cost diagnostic = 25 losses / 434 mismatches / 2.065 blocks

v2 full aggregate with backfilled cost-limited source selection:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_learned_action_halt_gate_costw200_block5_policycost_loto_seed20260526_costlimited_backfill/summary.json
  source-valid cost = 10 losses / 315 mismatches / 2.220 blocks
  source-valid cost-limited = 10 losses / 465 mismatches / 1.859 blocks
  tight test-cost diagnostic = 25 losses / 461 mismatches / 1.782 blocks
```

Task-level readout:

- LAMBADA boundary-valid keeps source-valid cost safe but does not improve the tight frontier versus v2: tight remains `15` losses / `28` mismatches, with slightly higher blocks (`2.493` vs `2.484`).
- SQuAD boundary-valid improves the tight mismatch frontier at the same loss: `10` losses / `323` mismatches / `2.055` blocks, versus v2 `10` / `356` / `2.056`.
- HellaSwag boundary-valid worsens the tight mismatch frontier: `0` losses / `83` mismatches / `1.855` blocks, versus v2 `0` / `66` / `1.863`.

Interpretation: rare-boundary validation is methodologically useful because it makes validation curves less blind to sparse action failures, but it is not enough as a standalone fix. It mostly changes threshold/checkpoint selection and therefore still moves along the same safety-cost-mismatch surface. Do not expand this exact variant to all 8 tasks unless it is paired with a changed objective or richer latent/process interaction.

The follow-up `utility_soft_bce` objective tried to make the accept/defer training signal smoother. Instead of regressing raw utility or using only a binary rescue label, it maps fallback-minus-action utility delta to a soft BCE target:

```text
soft_target = sigmoid(utility_delta / temperature)
loss = weighted BCE(gate_logit, soft_target)
```

This is closer to a selective-prediction / learning-to-defer view, but the LAMBADA diagnostics show it still does not beat v2:

```text
utility_soft_bce, temperature=1.0:
  SwanLab = k83y47qe0ojzv1247mh9a
  summary = /data1/luyifei/drla/outputs/cola_action_halt_gate/official8_full_b64_bs12_loto_lambada_seed20260526_utilitysoftbce_boundaryvalid_policycost_besteval/summary.json
  source-valid cost = 0 losses / 0 mismatches / 3.320 blocks
  source-valid cost-limited = 5 losses / 5 mismatches / 3.213 blocks
  tight-cost diagnostic = 23 losses / 45 mismatches / 2.456 blocks
  issue = soft targets are too flat; test predicted-positive rate is 0.606

utility_soft_bce, temperature=0.1:
  SwanLab = p8g60owojtdbluovrstk7
  summary = /data1/luyifei/drla/outputs/cola_action_halt_gate/official8_full_b64_bs12_loto_lambada_seed20260526_utilitysoftbce_temp01_boundaryvalid_policycost_besteval/summary.json
  source-valid cost = 0 losses / 0 mismatches / 3.405 blocks
  source-valid cost-limited = 5 losses / 5 mismatches / 3.276 blocks
  tight-cost diagnostic = 32 losses / 59 mismatches / 2.416 blocks
  issue = sharper targets reduce loss/predicted-positive rate but the useful tight-cost frontier collapses

v2 LAMBADA reference:
  SwanLab = 5rvvs3njy67yez8zecx7w
  source-valid cost = 0 losses / 0 mismatches / 3.227 blocks
  tight-cost diagnostic = 15 losses / 28 mismatches / 2.484 blocks
```

Interpretation: the failure is no longer just label sparsity. The 11-dimensional action-head feature vector can separate some safety cases, but these scalar objectives do not learn a deployable utility frontier. Do not expand `utility_soft_bce` to all 8 tasks as-is.

Artifacts:

```text
answer_identity_action:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_answer_identity_action_completionrisk_boundarypen02_cross_seed_20260525/summary.json
  /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_answer_identity_action_completionrisk_boundarypen02_cross_seed_20260525/loss_analysis.json

answer_identity_halt:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_answer_identity_halt_completionrisk_targetcalstrict_boundarypen02_cross_seed_20260525/summary.json
  /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_answer_identity_halt_completionrisk_targetcalstrict_boundarypen02_cross_seed_20260525/loss_analysis.json

fine-contentful diagnostic:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_answer_identity_action_completionrisk_finecontentful_targetcalstrict_boundarypen02_cross_seed_20260525/summary.json

task-wise diagnostic hybrid:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_answer_identity_action_halt_strict_taskwise_hybrid_diagnostic_20260525/summary.json

causal action->halt latent gate diagnostic:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_action_halt_latent_gate_diagnostic_20260525/summary.json

learned action->halt gate v1:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_learned_action_halt_gate_loto_seed20260526/summary.json

learned action->halt gate v2:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_learned_action_halt_gate_costw200_block5_policycost_loto_seed20260526/summary.json

learned action->halt gate utility_mse diagnostics:
  /data1/luyifei/drla/outputs/cola_action_halt_gate/official8_full_b64_bs12_loto_lambada_seed20260526_utilitymse_policycost_besteval/summary.json
  /data1/luyifei/drla/outputs/cola_action_halt_gate/official8_full_b64_bs12_loto_lambada_seed20260526_utilitymse_rawscore_policycost_besteval/summary.json

learned action->halt gate utility_pairwise diagnostics:
  /data1/luyifei/drla/outputs/cola_action_halt_gate/official8_full_b64_bs12_loto_lambada_seed20260526_utilitypairwise_policycost_besteval/summary.json
  /data1/luyifei/drla/outputs/cola_action_halt_gate/official8_full_b64_bs12_loto_squad_seed20260526_utilitypairwise_policycost_besteval/summary.json
  /data1/luyifei/drla/outputs/cola_action_halt_gate/official8_full_b64_bs12_loto_lambada_seed20260526_utilitypairwise_mismatch1_policycost_besteval/summary.json
```

## 6. P1 判别器架构

`LatentHaltStudent-v1` 应采用 attention-based latent trajectory model，而不是 `raw latent -> binary stop` 或 `slot MLP -> mean pool -> MLP concat`。

### 6.1 输入

在线允许输入：

```text
latent trajectory:
  z_{1...b}, up to current block b
  每个 latent slot 为 R^16
  slot position
  block position
  current and previous block latent tokens
  latent norm / delta / cosine / drift
  denoising residual / velocity if available

process / budget:
  block_idx
  max_block_budget
  remaining_budget
  block_fraction
  recent marginal latent change
  optional task embedding
```

在线禁止输入：

```text
decoded answer text
scored_prediction
official_processed_generation
prediction_stability
EOS/im_end probability
token entropy from decoder logits
official correctness
gold answer
future block information
```

### 6.2 默认模型规格

```text
d_model = 64
attention_heads = 4
dropout = 0.1
```

结构：

```text
slot_adapter:
  standardize or LayerNorm over each R^16 latent slot
  Linear(16 -> d_model)
  add slot_pos_embedding
  add block_pos_embedding

process_token:
  MLP(process / budget features -> d_model)
  append to each block's slot tokens

intra_block_encoder:
  one lightweight self-attention layer over:
    [16 slot tokens + 1 process token]

block_pooler:
  K=4 learned pooling queries cross-attend to intra-block tokens
  keep last_slot token explicitly
  per-block output:
    [pool_1, pool_2, pool_3, pool_4, last_slot]

inter_block_encoder:
  2-layer causal Transformer over block summary tokens
  current block can attend only to previous/current blocks
  future block attention is forbidden

readout_queries:
  q_halt
  q_risk
  q_stability
  q_decoder_proxy

heads:
  multi-task MLP heads from readout query states
```

### 6.3 当前输出 Heads

当前已实现的输出：

```python
{
    "readiness": float,
    "correctness": float,
    "prediction_change": float,
    "contentful": float,
    "decoder_stop": float,
    "completion_risk": float,  # optional, enabled by --use-completion-risk
    "future_gain": float,
}
```

含义：

```text
readiness:
  默认是 oracle frontier at/after label；
  p0_teacher_halt 模式下是 at/after P0 chosen halt block；
  p0_teacher_action 模式下是 exact P0 chosen halt block stop-action label。

correctness:
  official scorer correctness，只用于离线蒸馏/eval，不作为在线输入。

prediction_change:
  当前 scored_prediction 是否不同于 prediction-stability reference。

contentful:
  当前 task-scored prediction 是否非空。

decoder_stop:
  当前 decoder output 是否包含 EOS/im_end/stop。

completion_risk:
  当前 prediction 是否为空或是 stability/final reference 的 strict prefix。

future_gain:
  继续 rollout 后 correctness 的未来增益 proxy。
```

未来可扩展输出：

```python
{
    "prediction_change_prob": float,
    "completion_risk_prob": float,
    "p_answerable": float,
    "p_stable": float,
    "p_correct_est": float,
    "p_continuation_risk": float,
    "uncertainty": float,
    "expected_future_gain": float,
    "p_stop": float,
}
```

每个 head 的用途：

```text
stop_proxy_logits:
  模仿 decoder EOS/im_end/stop readiness

entropy_proxy:
  模仿 decoder uncertainty，但推理期不读 decoder logits

prediction_change_prob:
  预测当前 answer proxy 是否还会变化

completion_risk_prob:
  预测当前 latent 是否对应 prefix / fragment / incomplete answer

p_answerable:
  当前 latent prefix 是否已经有足够 answer evidence

p_stable:
  当前 answer state 是否会在未来 block 保持稳定

p_correct_est:
  训练期 correctness proxy，评估时只能作为 student 输出，不能读 gold

p_continuation_risk:
  是否应该继续生成以避免 incomplete/changed answer

uncertainty:
  校准 halt 阈值时使用

expected_future_gain:
  继续生成后预期能带来的收益

p_stop:
  最终 halt head
```

## 7. 训练策略

### 7.1 训练数据

P1 训练应复用已有 full split traces 和 frontier labels：

```text
seed66:
  /data1/luyifei/drla/outputs/cola_block_traces/tasks_official8_full_b64_t16_seed66_bs12_merged_20260524
  /data1/luyifei/drla/outputs/cola_readiness_frontiers/official8_full_b64_bs12_t16_seed66_20260524

seed67:
  /data1/luyifei/drla/outputs/cola_block_traces/tasks_official8_full_b64_t16_seed67_bs12_merged_20260524
  /data1/luyifei/drla/outputs/cola_readiness_frontiers/official8_full_b64_bs12_t16_seed67_20260524

seed68:
  /data1/luyifei/drla/outputs/cola_block_traces/tasks_official8_full_b64_t16_seed68_bs12_merged_20260524
  /data1/luyifei/drla/outputs/cola_readiness_frontiers/official8_full_b64_bs12_t16_seed68_20260524
```

每个 training example 是：

```text
(sample_id, task, seed, block b)

input:
  latent prefix z_1...z_b
  process / budget features visible at b

targets:
  teacher labels derived offline from block b and future trace
```

### 7.2 Split 策略

必须至少报告：

```text
same-task split:
  train / valid / test within all tasks

leave-one-task-out:
  train on 7 tasks
  calibrate on 7-task valid
  evaluate held-out task split=all

cross-seed:
  repeat on seed66 / seed67 / seed68
```

P1 的主要可信结果应以 leave-one-task-out + cross-seed 为准，而不是只看 same-task random split。

### 7.3 优化策略

默认训练方式：

```text
optimizer: AdamW
valid_interval <= 100 steps
save best_checkpoint.pt
save last_checkpoint.pt
log to SwanLab cloud
write local metrics.jsonl
select best by validation objective
```

禁止事项：

```text
不能只报告 last checkpoint
不能关闭 SwanLab cloud 训练
不能在 test 上调阈值
不能把 decoder-derived fields 作为 student-only eval 输入
不能把 P1 结果覆盖 P0 riskcap04 baseline
```

### 7.4 默认 Loss

推荐 supervised distillation loss：

```text
L =
  BCE(p_answerable, y_answerable)
  + BCE(p_stable, y_stable)
  + BCE(p_correct_est, y_correct)
  + BCE(p_continuation_risk, y_continuation_risk)
  + BCE(p_stop, y_stop)
  + 0.2 * MSE(expected_future_gain, future_gain)
  + calibration_regularizer
```

也可以加入 soft teacher KL / BCE：

```text
L_soft =
  BCE(p_answerable, p_ready_teacher)
  + BCE(p_continuation_risk, p_risk_teacher)
  + BCE(p_stop, p_halt_teacher)
```

综合版本：

```text
L_total =
  L_hard
  + lambda_ready_soft * BCE(p_answerable, p_ready_teacher)
  + lambda_risk_soft * BCE(p_continuation_risk, p_risk_teacher)
  + lambda_halt_soft * BCE(p_stop, p_halt_teacher)
  + lambda_calib * calibration_regularizer
```

初始建议权重：

```text
lambda_ready_soft = 0.5
lambda_risk_soft = 0.5
lambda_halt_soft = 1.0
lambda_calib = 0.05
```

这些权重必须通过 validation ablation 调整，不能固定后直接当结论。

## 8. P0 Teacher 蒸馏方案

P0 teacher 不是单个网络，而是组合 teacher：

```text
official Cola decoder probe
official scorer
oracle readiness frontier
prediction-stability reference
ReadinessModel
ContinuationRiskModel
riskcap04 calibrated halt policy
sample-level diagnostics
```

### 8.1 Teacher Target 来源

| Target | 来源 | 是否可做 P1 训练标签 | 是否可做 P1 在线输入 |
|---|---|---:|---:|
| EOS/im_end/stop proxy | decoder probe | yes | no |
| token entropy/top probability | decoder logits/probe | yes | no |
| answer_changed | decoded text dynamics | yes | no |
| prediction_change | scored prediction future reference | yes | no |
| official_correct | official scorer | yes, offline only | no |
| oracle readiness frontier | scorer/future blocks | yes | no |
| future_gain | future block correctness/stability | yes | no |
| continuation risk | P0 risk target/model | yes | no |
| riskcap04 halt action | calibrated P0 policy | yes | no |
| latent norm/delta/cosine | latent trace | yes | yes |
| block_idx/remaining_budget | trace metadata | yes | yes |

### 8.2 Hard Labels

Recommended hard targets:

```text
y_answerable(b):
  1 if answer_found_b or oracle/readiness criteria says block b is at or after frontier

y_stable(b):
  1 if normalized task-scored answer remains stable for future r probes

y_correct(b):
  official scorer correctness at block b
  training only

y_continuation_risk(b):
  1 if current task-scored answer will change or is strict prefix/incomplete

y_stop(b):
  1 if P0 riskcap04 policy would halt at block b
  0 before that block

future_gain(b):
  max reward among future blocks minus reward at b
```

### 8.3 Soft Labels

Recommended soft targets:

```text
p_ready_teacher(b):
  sigmoid(P0 ReadinessModel logits)

p_risk_teacher(b):
  sigmoid(P0 ContinuationRiskModel logits)

p_halt_teacher(b):
  smoothed version of P0 riskcap04 decision
```

Possible smoothing for `p_halt_teacher`:

```text
if b < teacher_halt_block:
  p_halt_teacher = 0.0
if b == teacher_halt_block:
  p_halt_teacher = 1.0
if b > teacher_halt_block:
  p_halt_teacher = 1.0
```

For softer curriculum:

```text
p_halt_teacher(b) = sigmoid(alpha * (b - teacher_halt_block + 0.5))
```

with:

```text
alpha = 2.0 to 5.0
```

### 8.4 Distillation Pipeline

Recommended pipeline:

```text
1. Load frontier labels for all official8 tasks and seeds.
2. Load raw latent shards referenced by latent_batch_path.
3. For each sample and block b, construct latent prefix z_1...z_b.
4. Run or load P0 ReadinessModel probabilities.
5. Run or load P0 ContinuationRiskModel probabilities.
6. Reconstruct P0 riskcap04 selected halt block from eval summaries or policy replay.
7. Write P1 distillation dataset:
     inputs.pt
     labels.jsonl
     metadata.json
8. Train LatentHaltStudent-v1 on train split.
9. Select best checkpoint by valid safety/cost/calibration objective.
10. Calibrate halt thresholds only on valid split.
11. Evaluate student-only on held-out task / test.
12. Compare against P0 riskcap04 and prediction-stability.
```

### 8.5 Student-Only Eval Rule

During P1 student-only evaluation, each row may contain teacher-derived columns for auditing, but the decision function must only receive:

```text
latent prefix up to current block
process / budget features available at current block
student model outputs
validation-selected thresholds
```

The decision function must not receive:

```text
scored_prediction
decode_text_so_far
official_processed_generation
prediction_stability_reached
EOS/im_end probability
token entropy/top probability from decoder
official correctness
future block rows
gold answer
```

## 9. P1 Halt Rule

Recommended initial rule:

```text
stop if:
  block_idx >= B_min
  p_stop >= tau_stop
  p_stable >= tau_stable
  p_continuation_risk <= tau_risk
  completion_risk_prob <= tau_completion
  uncertainty <= tau_uncertainty
  expected_future_gain <= tau_gain

force stop if:
  block_idx == B_max
```

Default calibration grid:

```text
tau_stop:        0.1, 0.2, ..., 0.9
tau_stable:      0.5, 0.6, 0.7, 0.8, 0.9
tau_risk:        0.01, 0.05, 0.1, 0.2, 0.4
tau_completion:  0.05, 0.1, 0.2, 0.4
tau_uncertainty: validation percentiles
tau_gain:        0.0, 0.01, 0.02, 0.05
```

Selection objective:

```text
primary:
  zero observed valid loss vs prediction-stability or fixed-final

secondary:
  enforce a small validation mismatch cap when zero-loss rows are too aggressive

tertiary:
  minimize average blocks

quaternary:
  better calibration and lower future_gain error
```

This mirrors the P0 lesson that aggregate accuracy alone can hide loss/gain cancellation.

当前脚本支持：

```text
--require-zero-calibration-loss
--require-zero-calibration-mismatch
--max-calibration-mismatches
--max-calibration-mismatch-rate
```

`--require-zero-calibration-mismatch` 是硬安全诊断，常常退回 fixed-final。`--max-calibration-mismatches 3` 是当前更实用的中间策略：在 seen-task test 上能保持 zero loss 和明显 block saving，但 LOTO 仍不够安全。

## 10. Required Ablations

P1 cannot be accepted as a single architecture run. Required ablations:

```text
d_model:
  16 / 32 / 64 / 128

slot_adapter:
  identity+linear
  linear only
  2-layer MLP

normalization:
  train-set mean/std
  LayerNorm

pooling:
  no pooling full-slot
  mean+max
  PMA K=1
  PMA K=4 + last_slot

process interaction:
  concat MLP
  process token attention
  FiLM gating

readout:
  last token
  pooled state
  task-specific readout queries

features:
  with / without block_idx
  with / without remaining_budget
  with / without latent trajectory features

targets:
  with / without decoder-proxy targets
  with / without EOS/im_end teacher targets
  with / without answer stability targets
  with / without future gain head
  hard labels only vs hard + soft teacher labels
```

## 11. Evaluation Metrics

P1 must report:

```text
accuracy
average blocks
median blocks
p90 blocks
block saving vs fixed-final
block saving vs prediction-stability
loss count vs fixed-final
loss count vs prediction-stability
gain count vs fixed-final
gain count vs prediction-stability
early-stop wrong rate
forced-stop rate
oracle gap
readiness AUROC/AUPRC
risk AUROC/AUPRC
future gain MSE
ECE / calibration bins
per-task metrics
cross-seed mean/std
sample-level failure diagnostics
```

Most important comparison:

```text
P1 student-only
vs
P0 joint-readiness riskcap04
```

P1 should not be reported as successful merely because it beats fixed-final on average blocks. It must either approach P0 riskcap04's safety/cost frontier or clearly explain the remaining gap.

## 12. Implementation Plan

Recommended new files:

```text
drla/models/latent_halt_student.py
drla/scripts/build_p1_distillation_dataset.py
drla/scripts/train_latent_halt_student.py
drla/scripts/eval_latent_halt_student.py
drla/scripts/aggregate_latent_halt_student.py
```

### 12.1 `drla/models/latent_halt_student.py`

Should contain:

```text
LatentHaltStudentConfig
LatentHaltStudentModel
SlotAdapter
ProcessTokenEncoder
IntraBlockEncoder
PmaBlockPooler
CausalBlockEncoder
ReadoutQueryHeads
```

### 12.2 `build_p1_distillation_dataset.py`

Responsibilities:

```text
load P0 labels/frontiers
load latent shards
materialize latent prefixes
load/replay P0 teacher probabilities
write compact trainable tensors
write metadata and feature schema
verify no future/decoder fields enter online_input
```

### 12.3 `train_latent_halt_student.py`

Responsibilities:

```text
train multi-task P1 student
log SwanLab cloud
write metrics.jsonl
save best_checkpoint.pt and last_checkpoint.pt
validate every <=100 steps
support architecture ablations
```

### 12.4 `eval_latent_halt_student.py`

Responsibilities:

```text
load student checkpoint
run sequential student-only halt replay
calibrate thresholds on valid split
evaluate held-out task / test split
write summary.json and per-sample decisions
assert forbidden fields are not used by policy
```

## 13. Leakage Audit Checklist

Before accepting any P1 result, verify:

```text
[ ] eval policy does not read scored_prediction
[ ] eval policy does not read decoded text
[ ] eval policy does not read official_processed_generation
[ ] eval policy does not call prediction_stability_reached(row)
[ ] eval policy does not read EOS/im_end/stop probabilities
[ ] eval policy does not read token entropy/top probability from decoder
[ ] eval policy does not read official correctness
[ ] eval policy does not read future block rows
[ ] thresholds selected only on validation split
[ ] held-out task evaluation uses no held-out labels for calibration
[ ] P0 riskcap04 remains reported as baseline
[ ] loss/gain counts are reported, not only aggregate accuracy
```

## 14. Current Best Answer to "P1 最优模型是什么"

当前应这样表述：

```text
P1 当前最强已实现路线是：
  LatentHaltStudent-v1
  d64_pma4_last
  process_token
  full process features
  p0_teacher_action + completion_risk

它使用 P0 decoder-probed riskcap04 作为 offline teacher，
训练 attention-based latent student，
并在 student-only eval 中禁止读取 decoder/text/scorer-derived online features。
```

如果需要和当前实际结果并列：

```text
当前实际最优已完成结果：
  P0 joint-readiness riskcap04
  约 21.596% weighted micro accuracy
  2.118/4 blocks
  zero observed loss vs fixed-final and prediction-stability

当前 P1 seen-task 结果：
  seed68 all-8 test
  22.028% micro accuracy
  1.928/4 blocks
  zero observed loss vs prediction-stability

当前 P1 LOTO 结果：
  seed66/67/68 all-8 LOTO
  target-task 128-shot calibration
  boundary-risk penalty 0.2
  22.473% micro accuracy
  1.908/4 blocks
  55 observed losses / 120 eval summaries vs prediction-stability

结论：
  P1 已经证明 latent/process-only student 可学习 decoder teacher，
  但 seed68 zero-loss 不跨 seed 稳定；
  LOTO safety/cost frontier 仍弱于 P0 riskcap04。
```

## 15. 下一步建议

优先级：

1. 加入 latent-level answer identity / completion-stability / completion-boundary 目标，重点覆盖 Story Cloze、HellaSwag、SQuAD、LAMBADA 这类 prefix/incomplete answer 风险。
2. 对 seed66/67/68 的 loss samples 做训练目标复盘：现有 completion-risk/correctness heads 能诊断一部分前缀风险，但 hard gate 成本太高，不能作为主解。
3. 系统化 task/target calibration：区分 pure LOTO、target-valid calibration、少样本 target calibration 和同任务 test calibration，不要混为一个结论。
4. 改进校准选择器：把 zero-loss、mismatch cap、Wilson upper bound、block cost 组成显式安全成本目标，而不是只贪最少 block。
5. 与 P0 riskcap04、prediction-stability、fixed-final 并列报告，并明确 P1 仍是 latent/process-only online 输入。
6. 在上述闭环完成后，再考虑更复杂架构或 RL-style policy tuning。

不建议继续优先做：

```text
更多手写 decoder/text guard
no-riskcap shape-risk 替代 riskcap04
LoRA/adapter/RL-style policy tuning
只优化 aggregate accuracy 的 threshold sweep
```

原因是当前瓶颈不是 P0 缺少更多规则，而是尚未把 P0 teacher 暴露出的 completion/stability/readiness 信号迁移成 P1 latent-primary student。
