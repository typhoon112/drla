# P1 LatentHaltStudent 复现归档

更新时间：2026-06-06

## 阶段定位

P1 是 CoLA 线最重要的 decoder-free 早停判别器阶段。它回答的问题是：

```text
只看当前可见 latent prefix 与 process features，轻量学生模型能否学习 P0 的 answer-readiness / halt 信号？
```

P1 的训练标签来自 decoder/text/scorer-derived teacher 信号，但在线推理输入不包含 decoded text、decoder stop probe、task scorer、gold answer 或 correctness。因此 P1 是 decoder-supervised latent student，而不是 decoder-dependent online policy。

## 最优 P1 student-only 结果

Canonical summary：

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/
official8_full_b64_bs12_p1_trajtok_answer_identity_action_completionrisk_identitystable_boundarypen02_cross_seed_20260527/
summary.json
```

核心指标：

| 指标 | 数值 |
|---|---:|
| Eval summaries | 120 |
| Repeated decisions | 73,645 |
| Selected accuracy | `22.52835%` |
| Fixed-final same-split accuracy | `22.53378%` |
| Prediction-stability same-split accuracy | `22.53378%` |
| Avg selected blocks | `1.8121/4` |
| Prediction-stability avg blocks | `2.5080/4` |
| Saving vs prediction-stability | `0.6959` blocks |
| Losses vs final | `4` |
| Losses vs prediction-stability | `4` |
| Mismatches vs final | `601` |
| Mismatches vs prediction-stability | `606` |

Loss task 分布：

| Task | Losses | Mismatches | Avg blocks |
|---|---:|---:|---:|
| MMLU | 2 | 7 | 1.4561 |
| SQuAD | 2 | 494 | 1.8635 |

解释：这是当前 P1 latent/process-only student 的最佳低 loss / 低成本 frontier。它不提升 fixed-final accuracy，而是在同一样本上尽量保留 final/prediction-stability correctness 的同时提前停止。

## Fresh split locked risk audit

Canonical summary：

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/
official8_full_b64_bs12_p1_locked_seed66_67_68_split20260601_riskcert_20260527/
summary.json
```

核心指标：

| 指标 | 数值 |
|---|---:|
| Eval summaries | 24 |
| Repeated decisions | 14,940 |
| Selected accuracy | `20.93039%` |
| Fixed-final same-split accuracy | `20.95047%` |
| Prediction-stability same-split accuracy | `20.95716%` |
| Avg selected blocks | `1.8339/4` |
| Prediction-stability avg blocks | `2.5011/4` |
| Saving vs prediction-stability | `0.6672` blocks |
| Losses vs final | `3` |
| Losses vs prediction-stability | `4` |
| Mismatches vs final | `85` |
| Mismatches vs prediction-stability | `91` |
| Calibration joint-risk satisfied | `21/24` |

Loss task 分布：

| Task | Losses | Mismatches | Avg blocks |
|---|---:|---:|---:|
| LAMBADA | 1 | 1 | 3.2335 |
| RACE | 1 | 4 | 1.5657 |
| SQuAD | 2 | 68 | 2.0006 |

解释：locked riskcert 是 fresh split / fresh seed 视角的低风险审计。它支持“observed-low-risk”，但由于仍有 `3/24` calibration folds 未满足 joint-risk 约束，不能写成完全形式化认证。

## 最优模型架构

模型名：

```text
LatentHaltStudent-v1
```

最佳配置：

```text
width = d64
pooling = PMA with 4 queries
process_feature_mode = full
process_interaction_mode = trajectory_token
readiness target = answer_identity_action
auxiliary heads = completion_risk, answer_identity_stability
boundary penalty = 0.2
target calibration cap = 128 examples per target task valid split
```

在线输入：

- 当前 block 及之前所有可见 latent blocks；
- latent norm、delta、cosine、drift；
- block index / remaining budget 等 process features；
- causal inter-block attention 中的 trajectory token；
- 不输入 decoded answer text；
- 不输入 decoder EOS/im_end probe；
- 不输入 task scorer result；
- 不输入 gold/correctness。

离线 teacher labels：

- `answer_identity_action`：当前 block 是否已足以得到与 stable/final reference 一致的 answer identity；
- `completion_risk`：当前 decoded answer 是否为空、prefix、continuation-incomplete 或将继续变化；
- `answer_identity_stability`：当前 answer identity 是否已经稳定到 reference；
- correctness/loss/mismatch 只用于离线评估与 threshold calibration，不作为在线输入。

## 权重与训练产物

最佳 P1 route 的训练产物共有：

```text
best_checkpoint.pt: 24
metrics.jsonl: 24
summary.json: 24
```

结构为 `3 seeds x 8 leave-one-task-out`：

```text
/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/
cross_task_full_b64_bs12_seed66_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_20260527/
leave_<task>_out/

/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/
cross_task_full_b64_bs12_seed67_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_20260527/
leave_<task>_out/

/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/
cross_task_full_b64_bs12_seed68_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_20260527/
leave_<task>_out/
```

每个 `leave_<task>_out` 目录应包含：

```text
checkpoints/best_checkpoint.pt
checkpoints/last_checkpoint.pt
metrics.jsonl
summary.json
```

注意：归档不复制 2.9G 的权重目录，只锁定路径。复现时必须读取这些 checkpoint 或按同配置重训。

## Eval 产物

最佳 P1 cross-seed eval roots：

```text
/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/
cross_task_full_b64_bs12_seed66_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_targetcal_cap128_boundarypen02_subseeds_20260527

/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/
cross_task_full_b64_bs12_seed67_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_targetcal_cap128_boundarypen02_subseeds_20260527

/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/
cross_task_full_b64_bs12_seed68_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_targetcal_cap128_boundarypen02_subseeds_20260527
```

Locked riskcert eval roots：

```text
/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_locked/
cross_task_full_b64_bs12_seed66_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_split20260601_riskcert_20260527

/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_locked/
cross_task_full_b64_bs12_seed67_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_split20260601_riskcert_20260527

/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_locked/
cross_task_full_b64_bs12_seed68_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_split20260601_riskcert_20260527
```

Aggregated CSVs：

```text
eval_summary_rows.csv
seed_summary.csv
seed_task_summary.csv
subseed_summary.csv
task_summary.csv
loss_case_audit.json
```

## 代码入口

训练：

```text
/data1/luyifei/drla/drla/scripts/train_cola_latent_halt_student.py
```

评估：

```text
/data1/luyifei/drla/drla/scripts/eval_cola_latent_halt_student.py
```

聚合：

```text
/data1/luyifei/drla/drla/scripts/aggregate_cola_latent_halt_student_loto.py
/data1/luyifei/drla/drla/scripts/aggregate_cola_latent_halt_student_subseed_loto.py
/data1/luyifei/drla/drla/scripts/analyze_latent_halt_risk_control.py
```

## 复现顺序

环境：

```bash
cd /data1/luyifei/drla
source /data1/luyifei/drla/scripts/activate_conda.sh
```

推荐复现顺序：

1. 确认 P0/official8 traces 已存在。
2. 对每个 seed 和 leave-out task 训练 P1 student。
3. 训练时必须使用 CUDA/GPU、SwanLab cloud、本地 `metrics.jsonl`、`best_checkpoint.pt`、`last_checkpoint.pt`。
4. 用 `best_checkpoint.pt` 做 local-only eval；eval 不创建 SwanLab run。
5. 对 target valid 做 threshold calibration；target test 只用于最终报告。
6. 聚合 seed/task/subseed CSV 与 summary。
7. 对 locked split/riskcert 结果单独聚合，不要把它和 pre-locked ablation 结果混成一个主表。

## 数据隔离与泄漏审计

当前 P1 数据隔离合理，但表述必须精确：

- LOTO checkpoint 训练时不使用 held-out target task 的 train rows。
- sample split 按 `task::sample_id` 做 deterministic split，同一样本的 block rows 不跨 split。
- target valid 只用于 threshold calibration；target test 用于最终报告。
- P1 在线输入不包含 decoded text、decoder stop probe、scorer、gold answer 或 correctness。
- 当前主结果是 target-calibrated LOTO，不是 zero-shot target-task evaluation。
- 多轮消融使用过同一套 held-out protocol，因此作为投稿主结论时应标注为阶段性 paper-style evidence；locked riskcert 结果是后续更严格口径。

## 可引用表述

可以写：

```text
P1 learns a decoder-supervised but decoder-free online latent readiness policy.
The best latent/process-only student reaches 4 losses over 73,645 repeated
decisions while using 1.812/4 blocks on average.
```

不要写：

```text
P1 improves official CoLA benchmark accuracy.
P1 never uses decoder-derived information at training time.
P1 is a complete agent-to-agent latent communication result.
```

## 阶段停止理由

P1 已充分学习 P0 信号，继续在 official8 上做局部小消融的边际价值低。当前最有意义的后续不是继续调 P1 小头，而是把 P1 作为 decoder-free latent readiness / packet substrate，用于更合理的 multi-agent benchmark 或下一代 communication interface。
