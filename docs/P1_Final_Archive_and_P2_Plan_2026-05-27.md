# P1 阶段归档与 P2 计划

更新时间：2026-05-27

## 结论摘要

P1 阶段可以视为完成：`LatentHaltStudent-v1` 已经在 official8、3 seeds、leave-one-task-out、target-valid calibration 协议下充分学习到了 P0 decoder-probed readiness 信号。当前最好 student-only 路线是：

```text
trajectory_token + answer_identity_action + completion_risk + answer_identity_stability
```

同基准结果：

```text
Cola fixed-final accuracy: 22.534%
P1 selected accuracy:     22.528%
P1 selected blocks:       1.812 / 4
Loss vs Cola final:       4 / 73,645 repeated decisions = 0.0054%
Mismatch vs Cola final:   601 / 73,645 = 0.816%
Saved blocks vs final:    2.188
```

这不表示 P1 提升了 official Cola benchmark accuracy。P1 的贡献是：在同一批 held-out target-test samples 上，学习何时可以提前 halt，并尽量保持 final-block correctness。

## 代码状态

P1 归档时的代码基线：

```text
commit: 38ba1e6cd9f9d48c2d7f56787cc466c1b375fbc0
branch: main
repo:   /data1/luyifei/drla
```

关键代码入口：

| 用途 | 路径 |
|---|---|
| P1 student 训练 | `/data1/luyifei/drla/drla/scripts/train_cola_latent_halt_student.py` |
| P1 student eval / threshold sweep | `/data1/luyifei/drla/drla/scripts/eval_cola_latent_halt_student.py` |
| P1 subseed LOTO 聚合 | `/data1/luyifei/drla/drla/scripts/aggregate_cola_latent_halt_student_subseed_loto.py` |
| learned action->halt gate | `/data1/luyifei/drla/drla/scripts/train_cola_action_halt_gate.py` |
| gate 聚合 | `/data1/luyifei/drla/drla/scripts/aggregate_cola_action_halt_gate.py` |
| P0 risk-gated teacher | `/data1/luyifei/drla/drla/scripts/eval_cola_risk_gated_halt.py` |

环境入口：

```bash
source /data1/luyifei/drla/scripts/activate_conda.sh
```

所有深度学习训练实验必须继续使用 CUDA/GPU + SwanLab cloud；无训练过程的 eval / aggregation 必须 `swanlab_mode=disabled`。

## 数据与 Trace

主线数据只使用 Cola 官方 8 个 benchmark：

```text
lambada, mmlu, obqa, hellaswag, race, siqa, squad, story_cloze
```

当前 P1 使用的是 official Cola full prepared split 的 block-wise trace / readiness frontier，统一协议：

```text
b64 / bs12 / t16
seeds: 66, 67, 68
max blocks: 4
每个原始样本有 4 行 block trace
split 按 sample_key = task::sample_id 做 deterministic hash
```

主要 trace / label roots：

```text
/data1/luyifei/drla/outputs/cola_readiness_frontiers/official8_full_b64_bs12_t16_seed66_20260524
/data1/luyifei/drla/outputs/cola_readiness_frontiers/official8_full_b64_bs12_t16_seed67_20260524
/data1/luyifei/drla/outputs/cola_readiness_frontiers/official8_full_b64_bs12_t16_seed68_20260524
```

Official Cola full benchmark scorer summaries：

```text
/data1/luyifei/drla/outputs/cola_official_benchmarks/full_b64_bs12_trace_score_20260524/summary.json
/data1/luyifei/drla/outputs/cola_official_benchmarks/full_b64_bs12_seed67_trace_score_20260524/summary.json
/data1/luyifei/drla/outputs/cola_official_benchmarks/full_b64_bs12_seed68_trace_score_20260524/summary.json
```

Official benchmark anchor：

| Task | Official Cola acc mean +/- std (%) |
|---|---:|
| LAMBADA | 51.867 +/- 0.615 |
| MMLU | 20.593 +/- 0.314 |
| OBQA | 22.867 +/- 0.499 |
| HellaSwag | 4.087 +/- 0.066 |
| RACE | 20.740 +/- 0.404 |
| SIQA | 29.717 +/- 0.844 |
| SQuAD | 22.450 +/- 0.299 |
| StoryCloze | 28.220 +/- 0.482 |
| Unweighted task average | 25.070 +/- 0.135 |

## 训练/测试协议

P1 使用 LOTO，即 leave-one-task-out。

以 `leave_mmlu_out` 为例：

```text
训练任务：lambada, obqa, hellaswag, race, siqa, squad, story_cloze
留出任务：mmlu
```

每个任务内部按 `sample_key = task::sample_id` stable hash 划分：

```text
train_ratio = 0.8
valid_ratio = 0.1
test_ratio  = 0.1
```

对于目标任务：

```text
target valid = threshold calibration only
target test  = reported held-out evaluation
```

当前主结果是：

```text
3 trace seeds x 8 held-out tasks x 5 target-calibration subseeds
= 120 eval summaries
= 73,645 repeated decisions
```

注意：`73,645` 是 repeated decisions。底层 target-test samples 在 5 个 calibration subsample seeds 下重复评估，用来观察 threshold calibration 稳定性。

## 最佳 P1 权重与配置归档

最佳 student-only 路线训练 root：

```text
/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed66_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_20260527
/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed67_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_20260527
/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_20260527
```

每个 seed 下都有 8 个目录：

```text
leave_lambada_out
leave_mmlu_out
leave_obqa_out
leave_hellaswag_out
leave_race_out
leave_siqa_out
leave_squad_out
leave_story_cloze_out
```

每个 `leave_*_out` 目录都已核对存在：

```text
summary.json
metrics.jsonl
checkpoints/best_checkpoint.pt
checkpoints/last_checkpoint.pt
```

完整性检查结果：

```text
24 / 24 training summaries present
24 / 24 metrics.jsonl present
24 / 24 best_checkpoint.pt present
24 / 24 last_checkpoint.pt present
24 / 24 SwanLab run ids present
```

最佳路线统一关键 config：

```text
batch_size = 1024
epochs = 20
learning_rate = 2e-4
weight_decay = 1e-4
dropout = 0.1
d_model = 64
attention_heads = 4
inter_layers = 2
pooling_mode = pma4_last
process_feature_mode = full
process_interaction_mode = trajectory_token
readiness_target_mode = answer_identity_action
use_completion_risk = true
use_answer_identity_stability = true
completion_risk_loss_weight = 0.75
answer_identity_stability_loss_weight = 0.75
future_gain_loss_weight = 0.25
selection_metric = readiness_prediction_change_completion_identity_mean_auroc
valid_interval = 50
swanlab_mode = cloud
```

Checkpoint 内容包含：

```text
model_state_dict
config
norm_stats
metadata
```

其中 `metadata.online_input_policy` 明确：在线输入只使用 raw latent prefixes、block mask、process features、trajectory/process interaction 和可选 task conditioning；不输入 decoded text、decoder logits/probabilities、prediction-stability、official scorer outputs 或 gold answers。

## Eval 与聚合产物

最佳 P1 target-calibration eval roots：

```text
/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed66_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_targetcal_cap128_boundarypen02_subseeds_20260527
/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed67_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_targetcal_cap128_boundarypen02_subseeds_20260527
/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_targetcal_cap128_boundarypen02_subseeds_20260527
```

最佳 P1 aggregate summary：

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_trajtok_answer_identity_action_completionrisk_identitystable_boundarypen02_cross_seed_20260527/summary.json
```

该目录同时保存：

```text
eval_summary_rows.csv
seed_summary.csv
subseed_summary.csv
task_summary.csv
seed_task_summary.csv
```

重要对照与消融 summaries：

```text
# 同口径 P1 消融
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_answer_identity_action_completionrisk_boundarypen02_cross_seed_20260525/summary.json
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_answer_identity_halt_completionrisk_targetcalstrict_boundarypen02_cross_seed_20260525/summary.json
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_trajtok_answer_identity_action_completionrisk_boundarypen02_cross_seed_20260527/summary.json
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_trajtok_answer_identity_action_completionrisk_identitystable_contentful05_boundarypen02_cross_seed_20260527/summary.json
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_trajtok_answer_identity_action_completionrisk_emptyrisk_identitystable_boundarypen02_cross_seed_20260527/summary.json

# learned gate 参照
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_learned_action_halt_gate_costw200_block5_policycost_loto_seed20260526_costlimited_backfill/summary.json

# P0 teacher / upper bound
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_joint_readiness_riskcap04_cross_seed_20260524/summary.json
```

## 同基准指标结果

当前 paper-style 主表应使用 same-split target-test 口径，而不是直接混用 official full benchmark mean。

Protocol：

```text
official8 full, b64/bs12, seeds 66/67/68
leave-one-task-out
128-shot target-valid calibration
held-out target-test evaluation
N = 73,645 repeated decisions
```

| 方法 | 类型 | Acc (%) | Delta vs Cola (pp) | Blocks | Saved | Loss vs Cola | Mismatch vs Cola |
|---|---|---:|---:|---:|---:|---:|---:|
| Cola fixed-final | baseline | 22.534 | +0.000 | 4.000 | 0.000 | 0 | 0 |
| Prediction-stability | decoder/text baseline | 22.534 | +0.000 | 2.508 | 1.492 | 0 | 5 |
| P1 answer-id action + completion | student | 22.481 | -0.053 | 1.742 | 2.258 | 47 | 612 |
| P1 answer-id halt + completion | student | 22.498 | -0.035 | 1.824 | 2.176 | 31 | 694 |
| P1 traj-token + action + completion | student | 22.492 | -0.042 | 1.711 | 2.289 | 41 | 801 |
| P1 traj-token + identity-stability | student | 22.528 | -0.005 | 1.812 | 2.188 | 4 | 601 |
| P1 identity-stability + contentful>=0.5 | safety diagnostic | 22.534 | +0.000 | 2.924 | 1.076 | 0 | 146 |
| P1 identity-stability + empty-risk | negative ablation | 22.508 | -0.026 | 1.829 | 2.171 | 24 | 618 |
| P1 learned gate v2 cost-limited | post-hoc gate | 22.527 | -0.007 | 1.859 | 2.141 | 10 | 465 |
| P1 learned gate v2 safety | post-hoc gate | 22.534 | +0.000 | 2.722 | 1.278 | 0 | 130 |

最佳 P1 逐任务结果：

| Task | P1 selected acc (%) | Cola fixed acc (%) | Blocks | PS blocks | Losses | Mismatches |
|---|---:|---:|---:|---:|---:|---:|
| LAMBADA | 53.756 | 53.756 | 3.254 | 2.018 | 0 | 5 |
| MMLU | 22.278 | 22.288 | 1.456 | 2.605 | 2 | 7 |
| OBQA | 27.027 | 27.027 | 1.209 | 2.405 | 0 | 0 |
| HellaSwag | 3.980 | 3.980 | 1.799 | 2.689 | 0 | 93 |
| RACE | 20.040 | 20.040 | 1.583 | 2.528 | 0 | 5 |
| SIQA | 32.437 | 32.437 | 1.337 | 2.308 | 0 | 0 |
| SQuAD | 22.894 | 22.907 | 1.864 | 2.492 | 2 | 494 |
| StoryCloze | 30.769 | 30.769 | 1.565 | 2.425 | 0 | 2 |

## 公平性与泄漏结论

当前 P1 主结果没有发现直接 train/test leakage：

- P1 checkpoint 是 LOTO 训练，目标任务不进入训练任务集合。
- split 按 `task::sample_id` 做 stable hash，同一原始样本的 4 个 block 不会跨 split。
- target valid 只用于 threshold calibration；target test 才用于报告。
- eval 脚本在 valid sweep 中选 threshold，再到 test sweep 中取同一 threshold 的 matching row。
- 在线 P1 输入不包含 decoded text、decoder stop probe、task scorer、gold answer 或 correctness。

必须保留的 caveat：

- 当前主结果是 `128-shot target-valid calibration`，不是 zero-shot target transfer。
- P1 是 decoder-supervised student，训练标签来自 decoder/text/scorer，但推理输入不含这些信号。
- P1 阶段已经在同一 held-out test protocol 上做过多轮消融与路线选择；若写投稿主结果，应冻结方案后再跑一次 fresh seed/fresh split 的 locked evaluation。
- 不能把当前 P1 target-test result 直接和 official Cola full benchmark mean 混成一张主表；official full benchmark 是全量官方锚点，same-split fixed-final 才是 P1 公平 baseline。

## 复现入口

### 训练一个 LOTO student

示例为 `seed66 / leave_mmlu_out`。正式复现需要对 `seed in {66,67,68}` 和 8 个 held-out tasks 全部展开。

```bash
source /data1/luyifei/drla/scripts/activate_conda.sh

python -m drla.scripts.train_cola_latent_halt_student \
  --labels-dir /data1/luyifei/drla/outputs/cola_readiness_frontiers/official8_full_b64_bs12_t16_seed66_20260524 \
  --output-dir /data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed66_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_20260527/leave_mmlu_out \
  --tasks lambada,obqa,hellaswag,race,siqa,squad,story_cloze \
  --seed 66 \
  --batch-size 1024 \
  --d-model 64 \
  --attention-heads 4 \
  --inter-layers 2 \
  --pooling-mode pma4_last \
  --process-feature-mode full \
  --process-interaction-mode trajectory_token \
  --readiness-target-mode answer_identity_action \
  --use-completion-risk \
  --use-answer-identity-stability \
  --selection-metric readiness_prediction_change_completion_identity_mean_auroc \
  --valid-interval 50 \
  --swanlab-mode cloud
```

### Eval 一个 target task

```bash
python -m drla.scripts.eval_cola_latent_halt_student \
  --checkpoint /data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed66_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_20260527/leave_mmlu_out/checkpoints/best_checkpoint.pt \
  --output-dir /data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed66_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_targetcal_cap128_boundarypen02_subseeds_20260527/subseed20260525/leave_mmlu_out_eval_mmlu_test \
  --labels-dir /data1/luyifei/drla/outputs/cola_readiness_frontiers/official8_full_b64_bs12_t16_seed66_20260524 \
  --calibration-tasks mmlu \
  --eval-tasks mmlu \
  --eval-split test \
  --require-zero-calibration-loss \
  --max-calibration-mismatches 3 \
  --max-calibration-samples-per-task 128 \
  --calibration-subsample-seed 20260525 \
  --calibration-boundary-risk-penalty 0.2 \
  --swanlab-mode disabled
```

### 聚合 3 seeds / 5 calibration subseeds

```bash
python -m drla.scripts.aggregate_cola_latent_halt_student_subseed_loto \
  --eval-root /data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed66_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_targetcal_cap128_boundarypen02_subseeds_20260527 \
  --eval-root /data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed67_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_targetcal_cap128_boundarypen02_subseeds_20260527 \
  --eval-root /data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_targetcal_cap128_boundarypen02_subseeds_20260527 \
  --output-dir /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_trajtok_answer_identity_action_completionrisk_identitystable_boundarypen02_cross_seed_20260527
```

## 相关文档

- `/data1/luyifei/drla/docs/P1_Model_Comparison_Report_2026-05-27.md`：P1 paper-style 对比、泄漏审计与主表。
- `/data1/luyifei/drla/docs/DRLA_Implementation_Plan.md`：当前主线、P0/P1 协议和下一阶段约束。
- `/data1/luyifei/drla/docs/CURRENT_EXPERIMENT_STATUS.md`：截至 P1 的详细实验流水与 artifact 索引。
- `/data1/luyifei/drla/docs/AGENT_CONTEXT_REFERENCE.md`：从 AGENT 移出的运行细节和命令索引。
- `/data1/luyifei/drla/docs/Diffusion_Latent_Reasoning_Framework.md`：架构背景，尤其 6.2 的 block-level answer-enough halt。

## P2 阶段思路

根据 `docs/DRLA_Implementation_Plan.md`，下一阶段不应继续做“再堆一个二分类 head”的局部调参。P2 应从 P1 的结论出发，推进到更接近最终目标的 decoder-free latent communication。

优先级修订：P1 本身已经是 `decoder-as-teacher, latent-student` 的 decoder-free online-input verifier。进入 agent-to-agent 前，不再需要重新发明一个 verifier；应先把 P1 verifier v1 冻结，并完成：

```text
1. fresh seed / fresh split locked evaluation
2. calibration / risk-control certificate
3. readiness_state 接口
```

完成这三点后即可开始 agent-to-agent latent communication v1。

### P2.0 冻结 P1 后做 locked evaluation

目的：把 P1 的阶段性结果从“开发集充分证据”升级成“投稿可引用结果”。

要求：

- 冻结 P1 student 架构、target labels、threshold selection rule。
- 不再根据 locked test 结果改模型。
- 使用 fresh split 或 fresh seed。
- 输出同基准主表、per-task 表、loss/mismatch case audit。

原因：当前 P1 代码没有直接 test leakage，但同一 held-out test protocol 已经被多轮消融和人工决策复用。投稿主结论需要 locked evaluation。

实施方式：

```text
复用当前 24 个 P1 best checkpoints
使用 --split-seed-override 生成 fresh target valid/test split
保持 threshold grid / calibration rule 不变
只跑 local-only eval 和 aggregation
```

2026-05-27 已完成三 seed P1 locked audit v1：

```text
model:
  cross_task_full_b64_bs12_seed{66,67,68}_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_20260527
split_seed:
  20260601
eval_root:
  /data1/luyifei/drla/outputs/cola_latent_halt_student_eval_locked/cross_task_full_b64_bs12_seed{66,67,68}_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_split20260601_riskcert_20260527/subseed20260601
aggregate_summary:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_locked_seed66_67_68_split20260601_riskcert_20260527/summary.json
loss_case_audit:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_locked_seed66_67_68_split20260601_riskcert_20260527/loss_case_audit.json
micro:
  folds = 24
  repeated_samples = 14940
  selected_accuracy = 20.930%
  fixed_final_accuracy = 20.950%
  prediction_stability_accuracy = 20.957%
  avg_blocks = 1.834 / 4
  prediction_stability_avg_blocks = 2.501 / 4
  losses_vs_final = 3
  losses_vs_prediction_stability = 4
  mismatches_vs_final = 85
  mismatches_vs_prediction_stability = 91
risk_certificate:
  calibration_joint_risk_satisfied = 21 / 24
  max_calibration_upper_bound = 0.0838
```

这组 locked result 不能再反向用于修改 P1 模型或阈值。3 个未满足 certificate target 的 fold 全部是 OBQA：observed loss/mismatch 为 0，但 target-valid 样本太少，Wilson upper bound 仍为 `0.0838`，略高于当前 `0.08` target。主要 loss case 是 LAMBADA `b -> borneo`、SQuAD `1568– -> 1568–1609` 的 prefix boundary，以及 RACE 中 final/policy 为空但 prediction-stability 为 `A` 的 case。结论应写成 observed-low-risk + partially certified，而不是 fully certified safety。

### P2.1 改进校准与风险控制协议

当前 calibration 是经验型：

```text
target-valid 128 samples
+ zero observed loss
+ mismatch cap
+ boundary penalty
-> 选平均 block 最低的 threshold
```

改进版先不改变模型，而是在 eval summary 中加入 risk certificate：

```text
observed loss rate
Wilson upper bound for loss
observed mismatch rate
Wilson upper bound for mismatch
selected thresholds
whether requested risk targets are satisfied
```

已开始实现的 eval 参数：

```text
--calibration-loss-risk-target
--calibration-mismatch-risk-target
--calibration-risk-bound-z
```

如果设置 risk target，threshold selection 会优先选择满足 calibration risk upper bound 的候选；如果没有满足候选，则 summary 会明确记录 unsatisfied/fallback 情况。

### P2.2 输出 readiness_state 接口

当前 P1 多头输出不应只在 policy 内部被压成 stop/continue。agent-to-agent v1 应把多头收敛信息传给下一个 agent。

`halt_decisions_*.jsonl` 已开始输出：

```text
readiness_state = {
  halt_candidate_found,
  fallback_to_final,
  selected_block,
  final_block,
  prediction_stability_block,
  scores: {
    readiness,
    prediction_change,
    contentful,
    correctness,
    future_gain,
    completion_risk,
    answer_identity_stability,
    ...
  },
  thresholds: {...},
  margins: {...}
}
```

这样下一个 agent 可以同时接收：

```text
latent memory
halt action
readiness_state
risk certificate
```

### P2.3 准备 agent-to-agent latent communication 最小闭环

完成 locked eval、risk certificate 和 readiness_state 后，开始 agent communication：

```text
Agent A:
  produce latent blocks
  send accumulated latent memory + halt/verifier state

Agent B:
  consume latent memory directly or through shared Cola substrate
  decide continue / ask more / decode final
```

这一步的在线协议不能依赖 decoder text。Decoder 只能作为离线 teacher、debugger 和最终 evaluation tool。

### P2.4 LoRA/adapter/RL-style policy 只作为后续工具

实施文档已经明确：DiT LoRA / adapter 不作为提升 official benchmark accuracy 的主目标。P2 若使用 LoRA 或 RL-style policy，应服务于：

- 接口适配；
- verifier/readiness 信号增强；
- block-budget policy 微调；
- latent communication robustness；

而不是声称让 official Cola benchmark 变强。
