# P1 阶段模型对比报告

> 状态：P1/P0 归档与 paper-style 对比。本文用于 P1 主表、消融、泄漏审计和 P0 teacher 对照，不作为当前 P2 实施方案。

更新时间：2026-05-27

## 报告范围

本文总结当前 Phase P1 latent-halt 实验，并将其与 official Cola full benchmark、Phase P0 decoder-probed teacher、以及主要 P1 消融实验进行对比。

需要先明确三个评估口径：

- **Official Cola scorer**：在官方 8 个 benchmark 的 prepared split 上完整跑 4 个 block 后打分。它衡量的是冻结 Cola substrate 的最终答案精度，P1 halt 不改变这个能力。
- **P1 halt evaluation**：leave-one-task-out 学生早停评估，在 held-out test partition 上用 5 个 target-calibration subseeds 聚合。accuracy、loss、mismatch 都是和同一样本上的 fixed-final / prediction-stability 基线比较。
- **P0 teacher**：基于 decoder probe、文本稳定性和 scorer 信号得到的 readiness/risk policy。它是 P1 的 teacher 和上界诊断，不是最终 latent-only agent communication policy。

## Official Cola Full Benchmark

Official Cola full benchmark 使用 `b64`、`bs12`、seeds `66/67/68`，并在官方 8-task scorer 上评估。

| 任务 | Official Cola accuracy, mean +/- std (%) |
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

数据来源：

```text
/data1/luyifei/drla/outputs/cola_official_benchmarks/full_b64_bs12_trace_score_20260524/summary.json
/data1/luyifei/drla/outputs/cola_official_benchmarks/full_b64_bs12_seed67_trace_score_20260524/summary.json
/data1/luyifei/drla/outputs/cola_official_benchmarks/full_b64_bs12_seed68_trace_score_20260524/summary.json
```

## P1 主要模型

除特别说明外，下表均为 official8 full、3 seeds、LOTO-style P1 evaluation。Losses 和 mismatches 均相对于同一评估样本上的 prediction-stability。

| 模型 / 策略 | 在线输入 | Accuracy | Blocks | Losses | Mismatches | 解释 |
|---|---|---:|---:|---:|---:|---|
| P1 baseline `d64_pma4` | latent/process only | 21.556% | 2.048/4 | 58 | 1612 | 已经学到部分 P0-like 信号，但 HellaSwag/SQuAD 风险较高。 |
| `answer_identity_action + completion_risk` | latent/process only | 22.481% | 1.742/4 | 47 | 617 | 比 baseline 更便宜、mismatch 更低，但 correctness loss 仍偏多。 |
| `answer_identity_halt + completion_risk` | latent/process only | 22.498% | 1.824/4 | 31 | 699 | fair target-task strict aggregate；loss 少于 action route，但 block 成本和 mismatch 更高。 |
| `trajectory_token + action + completion_risk` | latent/process only | 22.492% | 1.711/4 | 41 | 806 | 说明 trajectory/delta 有帮助，但不是干净胜出。 |
| `trajectory_token + action + completion_risk + answer_identity_stability` | latent/process only | 22.528% | 1.812/4 | 4 | 606 | 当前 P1 student-only 的低 loss / 低成本 frontier 最优点。 |
| 上一行 + hard `contentful>=0.5` | latent/process + calibrated student contentful head | 22.534% | 2.924/4 | 0 | 151 | 安全诊断点很好，但成本比 prediction-stability 还高，不适合默认策略。 |
| 上一行 + `empty_answer_risk` | latent/process only | 22.508% | 1.829/4 | 24 | 623 | 负结果；把 empty 风险转移成 prefix/continuation miss。 |
| Learned action->halt gate v2, cost-limited | scalar P1 heads | 22.534% | 1.859/4 | 10 | 465 | mismatch 低于 best student，但 loss 更多。 |
| Learned action->halt gate v2, safety | scalar P1 heads | 22.534% | 2.722/4 | 0 | 130 | 很强的 safety reference，但计算成本较高。 |
| P0 joint-readiness riskcap04 teacher | decoder-probed/text-derived features | 21.596% weighted full split | 2.118/4 | 0 | 非同口径 text-mismatch audit | teacher / upper-bound diagnostic；不是 decoder-free。 |

当前最优 P1 student-only 模型：

```text
trajectory_token + answer_identity_action + completion_risk + answer_identity_stability
```

对应 artifact：

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_trajtok_answer_identity_action_completionrisk_identitystable_boundarypen02_cross_seed_20260527/summary.json
```

它将早期 action route 的 loss 从 `47` 降到 `4`，同时 block 成本仍接近最强低成本策略。它还没有超过 decoder-probed 或 safety-gated 的零 loss 基线，但已经是当前 latent/process-only student 中最好的点。

## P1 学生模型架构与训练策略

P1 student 保持 official Cola VAE/DiT substrate 冻结。它不重新训练 Cola，也不改变第 4 个 block 的最终答案分布。P1 只学习一个基于当前可见 latent block prefix 的在线 halt/readiness policy。

当前最优学生模型：

```text
LatentHaltStudent-v1
width d64
PMA pooling with 4 queries
process_token full features
trajectory_token interaction
answer_identity_action readiness target
completion_risk auxiliary head
answer_identity_stability auxiliary head
boundary penalty 0.2 in target calibration
```

在线输入：

- 当前 block 及之前所有可见 latent blocks；
- latent norm、delta、cosine、drift，以及 block-budget features；
- 推理时不输入 decoded answer text，不输入 decoder EOS/im_end probe，也不输入 task scorer result。

离线 teacher labels：

- `answer_identity_action`：当前 block 第一次 decode/scored answer identity 与 prediction-stability/final reference 一致的位置；
- `completion_risk`：当前 decoded answer 为空，或是 stable/final reference 的 strict prefix / incomplete continuation；
- `answer_identity_stability`：当前 block 的 answer identity 已经等于 stable/final reference。

因此，P1 可以理解为一个 decoder-supervised latent student：训练和评估时使用 decoder/text 信号构造监督标签，但在线推理时不把 decoder/text 当作输入。这正是从 P0 teacher 过渡到未来 decoder-free agent latent communication policy 的中间桥梁。

训练协议：

- 官方 8 个任务：`lambada, mmlu, obqa, hellaswag, race, siqa, squad, story_cloze`；
- seeds `66/67/68`，leave-one-task-out；
- 所有 P1 training 都使用 CUDA/GPU、SwanLab cloud、本地 `metrics.jsonl`、`best_checkpoint.pt` 和 `last_checkpoint.pt`；
- 最优 trajectory/identity-stability route 使用 `valid_interval=50`；
- eval aggregation 使用 local-only、SwanLab disabled，因为它不是训练 run。

## 与 Official Cola 的对比方式

这里有两个公平比较方式：

1. **Official Cola final-answer accuracy**：完整跑 4 个 block，再在官方 prepared benchmark 上评分。P1 不以提升这个数字为目标。
2. **Same-split halt comparison**：在同一样本上比较 P1 selected block 与 fixed-final / prediction-stability answer。它衡量的是学生模型能否更早停下，并尽量不改变 correctness。

### 数据隔离与泄漏审计

当前 P1 LOTO protocol 的代码层面数据隔离是合理的：

- P1 student checkpoint 按 `leave-one-task-out` 训练。以 `leave_mmlu_out` 为例，训练只使用另外 7 个任务，不使用 MMLU。
- split 是按 `sample_key = task::sample_id` 的 deterministic hash 做的，同一个原始样本的 block 1/2/3/4 会进入同一个 split，不会出现 block 级泄漏。
- target task 的 `valid` split 只用于 threshold calibration；target task 的 `test` split 才用于报告结果。
- eval 脚本先在 `valid` sweep 中选 threshold，再用同一个 threshold 去 `test` sweep 中取 matching row；没有在该脚本内用 target-test loss 选择 threshold。
- P1 在线输入只包含 latent prefix 与 process features，不输入 decoded text、decoder stop probe、task scorer、gold answer 或 correctness。

但有三个必须在论文中如实说明的 caveat：

- **这不是 zero-shot target-task evaluation**：当前主结果使用 `128-shot target-valid calibration`，且 calibration 目标依赖 decoder/scorer/text-derived 标签。因此它是 target-calibrated LOTO，不是完全无标签迁移。
- **P1 是 decoder-supervised student**：训练标签来自 decoder/text/scorer 信号；推理时不输入这些信号。这个设定合理，但不能写成“完全不使用 decoder 信息训练”。
- **当前 P1 阶段存在 test-protocol 复用风险**：我们在同一套 held-out test protocol 上做了多轮消融、人工分析和路线选择。代码没有直接 test leakage，但如果要作为最终投稿主结论，最好冻结模型/阈值选择规则后，在 fresh seed 或 fresh split 上做一次 locked evaluation。

因此，下面的主表可以作为当前 P1 阶段的 paper-style comparison，但论文表述应标注为：

```text
official8 full, b64/bs12, seeds 66/67/68,
leave-one-task-out, 128-shot target-valid calibration,
held-out target-test evaluation,
N = 14,729 target-test decisions x 5 calibration repeats = 73,645 repeated decisions.
```

### 同基准主表

下表把最优质的几个 P1 模型/策略和同一基准下的 Cola final-block baseline 放在一起。这里的 `Cola fixed-final` 不是 official full benchmark mean，而是同一批 P1 held-out target-test samples 上跑满 block 4 的 final-block baseline。

| 方法 | 类型 | Acc (%) | ΔAcc vs Cola (pp) | Blocks | Saved blocks | Loss vs Cola | Mismatch vs Cola | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Cola fixed-final | baseline | 22.534 | +0.000 | 4.000 | 0.000 | 0 (0.0000%) | 0 (0.000%) | 同一样本跑满 block 4 |
| Prediction-stability | decoder/text baseline | 22.534 | +0.000 | 2.508 | 1.492 | 0 (0.0000%) | 5 (0.007%) | 非 student；用于早停上界参照 |
| P1 answer-id action + completion | student | 22.481 | -0.053 | 1.742 | 2.258 | 47 (0.0638%) | 612 (0.831%) | 低成本早期路线 |
| P1 answer-id halt + completion | student | 22.498 | -0.035 | 1.824 | 2.176 | 31 (0.0421%) | 694 (0.942%) | 更保守的 halt target |
| P1 traj-token + action + completion | student | 22.492 | -0.042 | 1.711 | 2.289 | 41 (0.0557%) | 801 (1.088%) | trajectory/delta 架构消融 |
| P1 traj-token + identity-stability | student | 22.528 | -0.005 | 1.812 | 2.188 | 4 (0.0054%) | 601 (0.816%) | 最佳低 loss/cost student |
| P1 identity-stability + contentful>=0.5 | safety diagnostic | 22.534 | +0.000 | 2.924 | 1.076 | 0 (0.0000%) | 146 (0.198%) | 零 loss 诊断，成本高 |
| P1 identity-stability + empty-risk | negative ablation | 22.508 | -0.026 | 1.829 | 2.171 | 24 (0.0326%) | 618 (0.839%) | 负消融 |
| P1 learned gate v2 cost-limited | post-hoc gate | 22.527 | -0.007 | 1.859 | 2.141 | 10 (0.0136%) | 465 (0.631%) | source-valid 选择；二阶段策略 |
| P1 learned gate v2 safety | post-hoc gate | 22.534 | +0.000 | 2.722 | 1.278 | 0 (0.0000%) | 130 (0.177%) | source-valid 选择；安全参照 |

建议论文主张聚焦在两行：

- `P1 traj-token + identity-stability`：最佳 student-only low-loss/cost frontier，只有 `4/73,645` repeated losses，同时平均节省 `2.188` blocks。
- `P1 learned gate v2 safety` 或 `contentful>=0.5`：作为 safety/cost trade-off 参照，而不是主 student-only 结果。

不要把 `best_test_gate_by_loss`、`causal_oracle_defer_after_action`、以及任何 test-selected oracle policy 放入主表；这些只适合做 upper-bound diagnostic。旧的 `d64_pma4` baseline 聚合使用 full/all split `147,057` samples，也不应和当前 target-test repeated protocol 混在主表中。

下表是当前最优 P1 student 的逐任务 same-split halt comparison。它应该被解读为“P1 在节省 block 的同时保留了多少 correctness”，而不是“P1 提升了 Cola 官方 benchmark 精度”。

| 任务 | Official Cola full acc mean (%) | P1 same-split fixed acc (%) | P1 selected acc (%) | 相对 same-split fixed 的变化 | 相对 final 节省 blocks |
|---|---:|---:|---:|---:|---:|
| LAMBADA | 51.867 | 53.756 | 53.756 | +0.000 | 0.746 |
| MMLU | 20.593 | 22.288 | 22.278 | -0.010 | 2.544 |
| OBQA | 22.867 | 27.027 | 27.027 | +0.000 | 2.791 |
| HellaSwag | 4.087 | 3.980 | 3.980 | +0.000 | 2.201 |
| RACE | 20.740 | 20.040 | 20.040 | +0.000 | 2.417 |
| SIQA | 29.717 | 32.437 | 32.437 | +0.000 | 2.663 |
| SQuAD | 22.450 | 22.907 | 22.894 | -0.013 | 2.136 |
| StoryCloze | 28.220 | 30.769 | 30.769 | +0.000 | 2.435 |

由于 official full scorer 和 P1 LOTO repeated-sample protocol 的数据划分与样本权重不完全相同，official 列是 benchmark anchor，而 fixed-vs-selected delta 才是同口径 halt 指标。在该 halt 口径下，最佳 P1 student 只损失 `4/73,645 = 0.00543%` repeated samples，同时相对 final-block decoding 平均节省 `2.188` blocks，相对 prediction-stability 平均节省 `0.696` blocks。

## 当前最优 P1 模型逐任务结果

下表使用 P1 held-out test repeated-sample protocol，不是 full official scorer。`Fixed` 是同一样本上的 final block accuracy。`PS blocks` 是 prediction-stability average blocks。

| 任务 | P1 best acc (%) | Fixed acc (%) | Blocks | PS blocks | Losses | Mismatches |
|---|---:|---:|---:|---:|---:|---:|
| LAMBADA | 53.756 | 53.756 | 3.254 | 2.018 | 0 | 5 |
| MMLU | 22.278 | 22.288 | 1.456 | 2.605 | 2 | 7 |
| OBQA | 27.027 | 27.027 | 1.209 | 2.405 | 0 | 0 |
| HellaSwag | 3.980 | 3.980 | 1.799 | 2.689 | 0 | 93 |
| RACE | 20.040 | 20.040 | 1.583 | 2.528 | 0 | 5 |
| SIQA | 32.437 | 32.437 | 1.337 | 2.308 | 0 | 0 |
| SQuAD | 22.894 | 22.907 | 1.864 | 2.492 | 2 | 494 |
| StoryCloze | 30.769 | 30.769 | 1.565 | 2.425 | 0 | 2 |

解释：

- 当前最优 P1 模型在 6/8 个任务上没有观察到 correctness loss。
- 残余 correctness loss 只出现在 MMLU 和 SQuAD。
- SQuAD 仍贡献了大部分 text-identity mismatch，即 `494/606`。因此当前未解决的核心问题不是 benchmark accuracy，而是 answer-boundary identity。

## P0 Teacher 逐任务参考

P0 joint-readiness riskcap04 使用 decoder-probed 和 text-stability-supervised 信号。它的输入比 P1 丰富，因此应被视为 teacher / upper bound，而不是最终部署策略。

| 任务 | P0 risk-gated acc (%) | Fixed acc (%) | P0 blocks | PS blocks | Losses |
|---|---:|---:|---:|---:|---:|
| LAMBADA | 51.866 | 51.866 | 1.670 | 2.015 | 0 |
| MMLU | 20.598 | 20.593 | 2.616 | 2.620 | 0 |
| OBQA | 22.867 | 22.867 | 2.351 | 2.351 | 0 |
| HellaSwag | 4.089 | 4.089 | 1.788 | 2.676 | 0 |
| RACE | 20.756 | 20.742 | 2.516 | 2.517 | 0 |
| SIQA | 29.717 | 29.717 | 2.334 | 2.334 | 0 |
| SQuAD | 22.453 | 22.450 | 1.865 | 2.504 | 0 |
| StoryCloze | 28.220 | 28.220 | 1.486 | 2.439 | 0 |

P1 已经学到相当多的 P0 结构：例如 SQuAD 上 P1 blocks 为 `1.864`，几乎等于 P0 的 `1.865`；很多多选任务上 P1 还更便宜。但 P0 仍能做到 zero observed loss，因为它允许使用 decoder/text-derived features。

## 主要消融结论

### 架构消融

| 消融项 | 结果 | 结论 |
|---|---|---|
| `all_tokens` pooling | seed68 上 2 losses / 17 mismatches / 3.979 blocks | 太保守，不适合作为默认策略。 |
| `pma1` | seed68 上 5 losses / 315 mismatches / 2.793 blocks | 单个 PMA query 对 evidence 压缩过强。 |
| `mean_max` | seed68 上 4 losses / 115 mismatches / 2.837 blocks | mismatch 较低，但 block 成本偏高。 |
| `d128_pma4` | seed68 上 4 losses / 285 mismatches / 2.432 blocks | 单纯扩容不是关键杠杆。 |
| `d32_pma4` | seed68 上 27 losses / 263 mismatches / 2.126 blocks | 更便宜，但明显不安全。 |
| `no_block_budget` | seed68 上 8 losses / 389 mismatches / 1.831 blocks | block/budget features 仍是重要 calibration anchor。 |
| `film` process interaction | seed68 上 8 losses / 190 mismatches / 2.260 blocks | 简单 FiLM 不能替代 process/trajectory tokens。 |
| `trajectory_token` | cross-seed 41 losses / 806 mismatches / 1.711 blocks | trajectory/delta 是正向信号，但需要更好的 identity objective。 |

### 目标函数与校准消融

| 消融项 | 结果 | 结论 |
|---|---|---|
| `answer_identity_action + completion_risk` | 47 losses / 617 mismatches / 1.742 blocks | 低成本方向有效，但仍不够安全。 |
| `answer_identity_halt + completion_risk` | 31 losses / 699 mismatches / 1.824 blocks | loss 少于 action route，但 block 和 mismatch 更高。 |
| `answer_identity_stability` head | 4 losses / 606 mismatches / 1.812 blocks | 当前 P1 student-only 最优点。 |
| Hard `contentful>=0.5` | 0 losses / 151 mismatches / 2.924 blocks | 很有价值的安全诊断，但太贵。 |
| `empty_answer_risk` head | 24 losses / 623 mismatches / 1.829 blocks | 负迁移 / proxy mismatch，不应继续简单堆二分类辅助头。 |
| Wilson risk control on current sweeps | strict targets 下 0 个 complete folds 被选中 | 当前 calibration size / scores 还不能认证低风险。 |

## Paper 级别结论

1. **P1 没有提升 Cola 官方 benchmark accuracy，也不应该把这作为目标。** 冻结 Official Cola 的 full-benchmark task average 约为 `25.07%`。P1 改变的是何时停止，而不是 Cola 最终能生成什么答案。
2. **P1 确实学到了有意义的 P0 readiness 信号。** 最佳 latent/process-only student 在 `1.812/4` blocks 下，只产生 `4/73,645` repeated-sample losses。
3. **P0 仍是安全上界。** P0 riskcap04 在 `2.118/4` blocks 下 zero observed loss，但依赖 decoder/text features，因此不能直接作为最终 agent latent-communication policy。
4. **当前主要风险是 answer identity boundary。** SQuAD mismatch 和 prefix/continuation case 占主导；简单添加 `empty_answer_risk` 这种窄二分类 head 不够。
5. **当前最稳妥的科学表述是：** P1 已经充分学习到了 P0 decoder-probed readiness signal，足以支持继续走 latent-student 路线；但最终部署前仍需要更强的 answer-identity risk modeling，或更严格的 calibration / risk-control protocol。

## P1 阶段推荐停止点

如果用于论文或阶段报告，建议把以下几个模型作为 P1 headline：

1. `trajectory_token + answer_identity_action + completion_risk + answer_identity_stability`：最佳 latent/process-only student。
2. P0 joint-readiness riskcap04：decoder-probed teacher / upper bound。
3. Learned action->halt gate v2 safety：额外的 safety-cost reference。
4. `empty_answer_risk`：负消融，用来说明继续堆窄二分类 auxiliary head 不是正确方向。

不要把 P1 表述为提升 official Cola benchmark accuracy。更准确的表述是：P1 在 student-only online-input policy 下，显著减少 latent block budget，同时近似保持 final-answer correctness。
