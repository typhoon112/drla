# P1 Progress and Literature Synthesis, 2026-05-27

> 状态：P1 实验笔记。本文是 2026-05-27 的 P1 文献复盘与决策记录，不作为当前 P2 实施方案。

本记录用于给下一步实验提供依据。本轮决策刻意不参考 `docs/DRLA_Multiscale_Block_Halt_Design.md`，只基于已经完成的 P0/P1 实验证据、官方 Cola 资料、工作区论文和网络检索到的 primary sources。

## 1. 当前主线判断

当前路线仍是：

```text
official Cola VAE + official Cola DiT
-> block-wise rollout trace
-> decoder/probe teacher
-> P0 decoder-probed safety/cost frontier
-> P1 latent/process-only student
-> adaptive halt / answer-readiness frontier
```

主研究对象不是提升 Cola 官方 benchmark 精度，而是让在线判别器只凭已经生成的 latent/process trajectory 判断：继续生成 block 是否还能带来 answer correctness / stability 的实质收益。

## 2. 已有实验证据

### P0 decoder/probe teacher

P0 `joint-readiness riskcap04` 是当前安全上界和 teacher source：

- Summary: `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_joint_readiness_riskcap04_cross_seed_20260524/summary.json`
- 三 seed weighted fixed-final micro accuracy: `0.215930`
- prediction-stability: `0.215957` at `2.512/4` blocks
- risk-gated P0: `0.215964` at `2.118/4` blocks
- observed losses vs fixed-final / prediction-stability: `0`

解释：P0 已证明 decoder/probe/stability 信号里存在 answer-readiness，但它不是最终 online policy，因为它依赖 decoder/text-derived signals。

### P1 LatentHaltStudent baseline

P1 原始 LOTO student-only 结果：

- Summary: `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_cross_seed_20260525/summary.json`
- selected accuracy: `0.215563`
- avg blocks: `2.048/4`
- losses vs prediction-stability: `58/147057 = 0.0394%`
- text mismatches vs prediction-stability: `1612/147057 = 1.096%`

解释：latent/process-only student 有跨任务信号，但 safety 不足；失败主要集中在 completion boundary 和 text/answer stability。

### P1 answer-identity / completion-risk 系列

严格对比下，answer-identity 分支比早期 P1 更便宜，但仍有 correctness loss：

- `p0_teacher_action + completion_risk`: `0.224727`, `1.908/4`, `55` losses, `886` mismatches
- `answer_identity_action + completion_risk`: `0.224808`, `1.742/4`, `47` losses, `617` mismatches
- `answer_identity_halt + completion_risk`: `0.224903`, `1.747/4`, `37` losses, `1263` mismatches

解释：action 分支更便宜、mismatch 少；halt 分支 correctness loss 少但 mismatch 多。二者的差异说明下一步不应把 action/halt 当成二选一标签，而应显式建模“继续是否值得”。

### Learned action->halt gate v2

当前较强 gate 是 cost-weighted BCE / policy-cost 选择的 action->halt gate：

- Summary: `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_learned_action_halt_gate_costw200_block5_policycost_loto_seed20260526_costlimited_backfill/summary.json`
- source-valid safety selected gate: `0` losses, `135` mismatches, `2.722/4`
- source-valid cost selected gate: `10` losses, `320` mismatches, `2.220/4`
- source-valid cost-limited selected gate: `10` losses, `470` mismatches, `1.859/4`
- best test under action+0.10 blocks: `25` losses, `466` mismatches, `1.782/4`

解释：latent student heads 中确实有 rescue signal，但单一 BCE / cost-weighted BCE / scalar utility objective 仍然不能稳定对齐 loss、mismatch 和 block cost 三者。

### 近期负结果

以下方向已作为负结果或局部诊断保留，不应盲目扩展：

- `utility_mse`
- `utility_pairwise`
- `utility_pairwise --utility-mismatch-penalty 1.0`
- `utility_soft_bce` with temperature `1.0` / `0.1`
- 只靠 `train_boundary_stratified` 的三任务 partial 扩展
- 简单宽度、pooling、FiLM、no-budget、stability loss 加权等架构微调

共同结论：问题不是数据链路坏了，也不是 latent 完全没有信号；问题是 rare-boundary decision objective 和 calibration 没有对齐真实 frontier。

## 3. 文献与网络检索结论

### Cola DLM

官方 Cola DLM 采用 Text VAE + block-causal DiT prior + conditional decoder，论文明确把 text generation 视为 hierarchical latent diffusion language modeling，并在 8 个 benchmark 上评估。这确认我们应该基于 official Cola latent substrate 做 readiness/halt，而不是自己训练小 prior 或用 LoRA 刷 benchmark 分数。

Sources:

- https://arxiv.org/abs/2605.06548
- https://huggingface.co/ByteDance-Seed/Cola-DLM

### COCONUT

COCONUT 把 LLM 最后一层 hidden state 当作 continuous thought，再反馈为下一步 input embedding；论文也指出 latent mode 何时终止是关键问题，可用 binary classifier 或固定 latent length。对本项目的启发是：halt 本身是 latent reasoning 系统的一等公民，不能只作为后处理阈值。

Source: https://arxiv.org/abs/2412.06769

### CODI

CODI 用 teacher/student self-distillation 把 explicit CoT 压缩到 continuous space，并对齐 answer-generating state 附近的 hidden activations；同时论文警告不要让 teacher hidden state 直接携带 exact answer shortcut。对本项目的启发是：P0 decoder signal 可以做 teacher，但 P1 在线输入必须严格不含 decoded text / gold scorer / future blocks。

Source: https://arxiv.org/abs/2502.21074

### CoLaR

CoLaR 使用随机 compression factor、probabilistic latent head 和 RL-style exploration-exploitation，强调 dynamic latent compression / termination 不是固定长度或单一 deterministic score。对本项目的启发是：action->halt gate 应该学习收益、风险和成本的分解，并用 frontier-aware calibration 选策略，而不是只学一个 defer 概率。

Source: https://arxiv.org/abs/2505.16552

### Latent-space survey

综述把 latent space 定位为 reasoning、planning、memory、collaboration 和 communication 的通用 substrate，同时指出 latent reasoning 的核心风险是 evaluability、controllability 和 interpretability。对本项目的启发是：最终 agent latent communication 不能依赖 decoder 作为在线信号，但训练阶段可以把 decoder/scorer 作为 teacher 和审计工具，用于逼迫 student 学 latent->answer-readiness mapping。

Source: https://arxiv.org/abs/2604.02029

### Dynamic early exit

Dynamic early-exit 工作在显式 CoT 中通过模型行为和 trial-answer confidence 判断是否停止，说明“看当前状态能否稳定回答”是自然问题设定。但它仍以 text/answer confidence 为主，不能直接替代 latent-only halt；我们借鉴的是 transition-point monitoring 和 confidence/risk 校准思想。

Source: https://arxiv.org/abs/2504.15895

## 4. 下一步实验选择

下一步不继续扩大 scalar utility / soft BCE，也不直接扩一个仅靠 boundary-valid 的 partial 结果。更合理的实验是：

```text
P1 learned action->halt gate
+ decomposed expected-utility objective
+ boundary-stratified validation
+ policy-cost-limited checkpoint selection
```

Online inputs 保持不变，只能使用：

- action policy 已经决定的 selected block number/fraction
- action student latent-head scores:
  - readiness
  - prediction_change
  - contentful
  - correctness
  - future_gain
  - completion_risk

不得使用 decoded text、gold answer、official correctness、future block output 或 task-hardcoded routing 作为在线输入。

### 目标分解

本轮新增的目标不是把所有因素压成一个二分类标签，而是用多头学习：

1. `rescue_loss_prob`: defer 是否能救回 action 的 correctness loss
2. `mismatch_rescue_prob`: defer 是否能修复 action 的 final-text mismatch
3. `introduced_loss_prob`: defer 是否引入 correctness loss
4. `introduced_mismatch_prob`: defer 是否引入 text mismatch
5. `extra_block_cost`: defer 相比 action 多用多少 block

策略分数：

```text
expected_defer_utility
= rescue_loss_prob * correctness_swing
+ mismatch_rescue_prob * mismatch_value
- introduced_loss_prob * correctness_swing
- introduced_mismatch_prob * mismatch_value
- extra_block_cost * block_cost
```

这样做的理由：

- v2 的主要价值来自少量 rescue 和大量 mismatch cleanup，不能只学 `fallback correct && action wrong`。
- Dry-run 显示当前 strict halt fallback 很少比 action 更差，因此 harm heads 可能稀疏；但保留 harm heads 可防止未来协议或更激进 fallback 下出现 silent regression。
- 该目标更接近 CoLaR 的 utility / cost trade-off，也更符合 survey 对 latent controllability/evaluability 的要求。

## 5. 验收口径

正式实验必须：

- 使用 official 8-task leave-one-task-out，而不是只看 LAMBADA 或三任务 partial。
- 所有训练上 SwanLab cloud；dry-run / aggregation / eval 不上云。
- valid interval `<=100` step。
- 保存 `metrics.jsonl`、`best_checkpoint.pt`、`last_checkpoint.pt`、`summary.json`。
- 与 P0 riskcap04、prediction-stability、action baseline、halt baseline、v2 learned gate 同表比较。
- 同时报告:
  - accuracy
  - avg blocks
  - losses vs final / prediction-stability
  - prediction/text mismatches
  - defer rate
  - rescued losses
  - per-task threshold and failure cases

## 6. 本轮实验结果

Dry-run 通过后，正式运行了四个 held-out tasks：

- LAMBADA: SwanLab `uctnvfdlndbtnxwif04zr`
- MMLU: SwanLab `13ag2qdsqr7u9wef3it4c`
- OBQA: SwanLab `2u6tev5kow9fp2lusfgi8`
- HellaSwag: SwanLab `l59gluqdvnfqg5gsh00oi`

Partial aggregate:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_decomposed_expectedutility_boundaryvalid_partial4_seed20260527/summary.json
```

四任务同口径对比旧 v2：

| Policy | v2 losses / mismatches / blocks | decomposed losses / mismatches / blocks | 结论 |
|---|---:|---:|---|
| action | `32 / 180 / 1.731` | `32 / 180 / 1.731` | 同一输入 action baseline |
| source-valid cost | `0 / 50 / 2.396` | `0 / 54 / 2.413` | 新目标更保守且 mismatch 更高 |
| source-valid cost-limited | `0 / 103 / 1.880` | `0 / 103 / 1.871` | 基本持平，只省 `0.009` block |
| test tight-cost diagnostic | `15 / 99 / 1.769` | `19 / 120 / 1.758` | 新目标更便宜但更不安全 |

Race 已启动但在数据构造/训练前被中断，未形成有效训练结论；对应目录只有 run-status / 空 train log，不应纳入统计。

结论：当前 decomposed objective 能学到 rare-boundary signal，但简单地把五个 head 线性合成为 expected utility 还不够。主要问题是：

1. `introduced_loss` 与 `introduced_mismatch` 在当前 strict halt fallback 下几乎全零，harm-side 监督太弱。
2. `mismatch_rescue` 频率远高于 correctness rescue，但默认 `utility_mismatch_penalty=0.1` 过低，导致 mismatch cleanup 没有被足够重视。
3. `policy_cost_limited` 的阈值选择仍只看当前 sweep 排序，没有显式校准 head probability，也没有 per-task reliability / calibration error 约束。
4. 在 test_positive_rate 为 0 的任务上，source-valid cost 容易学成过度 defer；在 LAMBADA 这类有 rescue 的任务上，tight-cost 反而漏掉更多关键 rescue。

## 7. 当时决策与后续复核

四任务 partial 后的当时判断是：不要直接把原始 `decomposed_expected_utility` 当成主线扩展到 official8 全量，应先修正 selector / calibration，而不是再添加一个 head 或继续扫 learning rate。优先候选：

1. 保留五个 decomposed heads，但训练后单独做 validation calibration：
   - 校准每个 head 的 reliability。
   - 在 source-valid 上学习或搜索 utility weights，而不是手写 `0.1` mismatch value。
   - selection objective 同时约束 `loss == 0`、`mismatch`、`avg_blocks`，避免只靠 raw expected utility 阈值。
2. 将 `mismatch_rescue` 与 `correctness_rescue` 分成两个策略层：
   - 第一层只处理 correctness rescue / harm。
   - 第二层在 correctness-safe frontier 内处理 text/answer-stability mismatch cleanup。
3. 回到 richer latent/process interaction，而不是继续只在 11 个 action-student scalar heads 上做 gate；当前 scalar heads 可能已经丢掉了 answer-boundary 的局部结构。

用户随后要求补齐数据，不做 partial 妥协。因此本轮继续补跑剩余 held-out tasks，并用相同 local-only 校准脚本做 full official8 复核；结论见第 9 节。

## 8. Post-hoc 校准复查

已追加一个本地-only 校准分析，不启动 optimizer、不创建 SwanLab run，只加载四个 decomposed best checkpoint，在 source validation 上搜索 utility weights、constrained head thresholds 和阈值，再应用到 held-out test。

Artifact:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_decomposed_expectedutility_calibrated_partial4_seed20260527/summary.json
```

该脚本解析到 `device=cuda`，但主要耗时在阈值扫描和 sample metric 聚合；这不是训练，也不应上云。

本轮补充查阅了 selective prediction / risk control / early-exit confidence 的资料。关键启发是：

- SelectiveNet 把 reject/accept 作为和主任务联合优化的 risk-coverage 问题，而不是只在预训练 confidence 上手写阈值。
- Learn-then-Test 把校准看成显式 risk control：先学习模型，再用校准集选择满足风险约束的策略。
- Neural calibration 论文说明神经网络 confidence 往往不是可靠概率，需要用 reliability / ECE 等诊断。
- BranchyNet 类 early-exit 工作使用“早层足够 confident 则退出”的思想，但这类 confidence threshold 需要额外校准，不能直接替代本项目的 latent-only halt risk。

Sources: SelectiveNet https://arxiv.org/abs/1901.09192; Learn-then-Test https://arxiv.org/abs/2110.01052; neural calibration https://proceedings.mlr.press/v70/guo17a.html; BranchyNet https://arxiv.org/abs/1709.01686.

四任务 scalar utility 同口径对比旧 v2：

| Policy | v2 losses / mismatches / blocks | calibrated decomposed losses / mismatches / blocks | 结论 |
|---|---:|---:|---|
| action | `32 / 180 / 1.731` | `32 / 180 / 1.731` | 同一 action baseline |
| source-valid cost | `0 / 50 / 2.396` | `0 / 54 / 2.375` | 省 `0.021` block，但 mismatch 更高 |
| source-valid cost-limited | `0 / 103 / 1.880` | `0 / 103 / 1.876` | 几乎持平，只省 `0.004` block |
| test tight-cost diagnostic | `15 / 99 / 1.769` | `27 / 125 / 1.751` | 更便宜但明显更不安全 |

constrained two-stage selector 结果：

| Policy | result losses / mismatches / blocks | 对比 |
|---|---:|---|
| source-valid constrained safety | `0 / 28 / 2.674` | 与 always-defer/prediction-stability 安全点几乎相同 |
| source-valid constrained cost-limited | `0 / 98 / 1.882` | 比 scalar cost-limited 少 `5` mismatches，但 block 多 `0.006`；仍只是与 v2 持平附近 |
| test constrained tight-cost diagnostic | `23 / 113 / 1.758` | 比 scalar tight-cost 好，但仍差于 v2 `15 / 99 / 1.769` |

source-task-robust selector 结果：

| Policy | result losses / mismatches / blocks | 结论 |
|---|---:|---|
| source-valid task-robust safety | `0 / 28 / 2.663` | 与 pooled safety 相同，说明 safety 点主要还是 always-defer-like |
| source-valid task-robust cost-limited | `0 / 93 / 1.876` | 当前四任务 deployable 最好点；超过 old v2 cost-limited `0 / 103 / 1.880` |
| source-valid task-robust constrained cost-limited | `0 / 103 / 1.873` | 更便宜但 mismatch 回到 pooled scalar 水平 |

这里的 task-robust 指：阈值仍只用 source validation 选择，但排序时先看 source tasks 的 worst-case loss/mismatch/defer，而不是 pooled source-valid 总体指标。它没有使用 held-out test 作为选择信号。

Head diagnostics 暴露了核心问题：

| Held-out task | rescue positives / AUPRC / top5% recall | mismatch positives / AUPRC / top5% recall | cost head |
|---|---:|---:|---:|
| HellaSwag | `0 / nan / 0.000` | `88 / 0.2510 / 0.477` | `MAE 0.035, r 0.817` |
| LAMBADA | `32 / 0.0287 / 0.406` | `59 / 0.0186 / 0.237` | `MAE 0.133, r 0.758` |
| MMLU | `0 / nan / 0.000` | `5 / 0.0002 / 0.000` | `MAE 0.059, r -0.180` |
| OBQA | `0 / nan / 0.000` | `0 / nan / 0.000` | `MAE 0.056, r 0.619` |

Source-valid 与 held-out test 的事件分布漂移很明显：例如 MMLU source-valid 中 rescue/mismatch rescue positives 分别是 `215/905`，但 held-out test 只有 `0/5`；OBQA test 两类 rescue 都是 `0`，但 heads 仍给出较高 mean probability。也就是说，当前 source-valid calibration 会把其他任务的 rare-boundary 频率外推到 held-out task，导致过度 defer。

结论：简单 post-hoc 标量 utility-weight calibration 不能把 decomposed heads 转成优于 v2 的策略；constrained selector 只带来小幅改善，说明 selector 形式不是唯一瓶颈。task-robust risk calibration 是一个值得继续验证的改进，因为它在四任务 deployable cost-limited 口径上超过 old v2，但这只是 partial，不足以重新启动主线结论。head 中有可用信号，因为 source-valid safety 仍能做到 `0` losses，HellaSwag mismatch rescue 也有排序能力；但当前 scalar student heads 的概率不可靠、跨任务分布漂移大、MMLU/OBQA 上几乎没有可救事件。下一步应先补完 remaining held-out tasks，再决定是否把 risk-controlled calibration 作为正式方向：

1. 补跑 RACE/SIQA/SQuAD/StoryCloze 的 `decomposed_expected_utility` formal runs，必须 SwanLab cloud + CUDA。
2. 用同一个 local-only calibration script 重新聚合 official8，比较 pooled、constrained、task-robust 与 old v2。
3. 若 task-robust 在 official8 仍成立，再把 risk-controlled calibration 正式纳入主线；否则转向更丰富 latent/process interaction，不再继续扫 selector。

## 9. Full Official8 补齐结果

补跑的剩余 held-out tasks 均使用 SwanLab cloud、`valid_interval=50`、`best_checkpoint.pt` 和 `last_checkpoint.pt`：

- RACE: `77amuogees617ovgiycmn`
- SIQA: `tdng7woa0awf4m4j0tp12`
- SQuAD: `qs2beb5h1rj6uo2b7q4kk`
- Story Cloze: `buwt1lx9we4ez2lzcbnfl`

Full official8 local-only calibration artifact:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_decomposed_expectedutility_calibrated_official8_seed20260527/summary.json
```

与 old v2 backfilled aggregate 的同口径对比：

| Policy | old v2 losses / mismatches / blocks | decomposed official8 losses / mismatches / blocks | 结论 |
|---|---:|---:|---|
| action | `47 / 612 / 1.742` | `47 / 612 / 1.742` | 同一 action baseline |
| source-valid cost-limited | `10 / 465 / 1.859` | `11 / 493 / 1.847` | 新模型更便宜 `0.012` block，但多 `1` loss 和 `28` mismatches |
| source-valid constrained cost-limited | n/a | `11 / 465 / 1.857` | mismatch 追平 v2，但仍多 `1` loss，block 只省 `0.002` |
| source-valid task-robust cost-limited | n/a | `11 / 482 / 1.848` | 四任务优势没有扩展到 full official8 |
| source-valid task-robust constrained cost-limited | n/a | `11 / 495 / 1.843` | 更便宜但 mismatch 更差 |
| source-valid safety | full accuracy, `130` mismatches, `2.722` | full accuracy, `140` mismatches, `2.658` | 同 accuracy 下省 block，但 mismatch 稍差 |
| best-test gate by loss | full accuracy, `130` mismatches, `2.501` | full accuracy, `130` mismatches, `2.366` | test-only 上界更便宜，不可作为 deployable 选择 |

复核结论：

1. 四任务 source-task-robust 的改善没有在 official8 成立，不能把它提升为主线。
2. Decomposed heads 仍然有诊断价值：它们能分开看 correctness rescue、mismatch cleanup、introduced harm 和 cost，帮助解释为什么 selector 会过度 defer 或漏救。
3. 当前瓶颈不只是 threshold selector，而是 action-student 11 个 scalar heads 的信息不足与跨任务概率校准不可靠。继续扫 utility weights / threshold grids 很可能只是在 loss、mismatch、blocks 之间搬动误差。
4. 下一步应该回到 richer latent/process interaction 或 formal risk-control objective：让模型直接看更丰富的 latent trajectory / block-local evidence，并把 risk constraint 明确放进训练或校准协议，而不是只在 frozen scalar heads 上做后处理。
5. 关于“为什么看起来像 CPU”：训练脚本解析到 CUDA 后模型参数和 batch 会放到 GPU；但 action->halt gate 只有 11 维输入、hidden=64，GPU 计算量很小，CPU 侧数据、SwanLab logging、valid 和 threshold sweep 会占主要时间。为避免误用，训练脚本已增加 CUDA fail-fast：非 dry-run 训练若解析到 CPU 会直接报错，并在新 summary/SwanLab config 中写入 resolved device。

## 10. Formal Risk-Control 诊断

为了避免继续在阈值上局部调参，本轮新增 local-only 风险控制审计：

```text
/data1/luyifei/drla/drla/scripts/analyze_latent_halt_risk_control.py
```

输入是已有 P1 d64 LOTO 的 `threshold_sweep_valid.csv` / `threshold_sweep_eval.csv`，覆盖 seed66/67/68 official8 共 `24` 个 held-out folds。脚本对 valid sweep 的 `losses_vs_prediction_stability` 和 `prediction_mismatch_vs_prediction_stability` 计算 Wilson 95% upper bound，只用 valid 选择阈值，再把同一阈值映射到 held-out eval。该分析不启动 optimizer，不创建 SwanLab run。

Artifact:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_latent_halt_wilson_risk_control_cross_seed_20260527/summary.json
```

关键结果：

| Policy | Complete folds | Held-out losses / mismatches / blocks | 结论 |
|---|---:|---:|---|
| empirical min-blocks | `24/24` | `1853 / 18926 / 1.559` | 便宜但明显不安全 |
| empirical zero-loss valid | `24/24` | `58 / 1612 / 2.048` | 当前经验校准基线 |
| Wilson loss `<=0.00025` | `0/24` | n/a | 当前 valid size 无法认证 low-loss target |
| Wilson loss `<=0.0005` | `0/24` | n/a | 同上 |
| Wilson loss `<=0.001` | `18/24` | n/a | 仍无法覆盖 official8 |
| Wilson loss `<=0.002` | `24/24` | `111 / 2983 / 1.825` | 完整但 held-out loss 高于经验 zero-loss |
| Wilson loss `<=0.002`, mismatch `<=0.005` | `24/24` | `104 / 2834 / 1.834` | 稍降 mismatch/loss，但仍不优于经验 zero-loss |

解释：

1. Learn-then-Test / selective prediction 的启发是正确的：早停应该被看成 risk-control / risk-coverage 问题，而不是固定 confidence 阈值。但当前 P1 valid split 每个 fold 只有约几千 calibration samples，若要证明 `<=0.00025` 或 `<=0.0005` 的低 loss 风险，0 个 valid loss 也不够。
2. 放宽 Wilson target 到 `0.002` 可以选出完整策略，却在 held-out 上比经验 zero-loss valid 更差，说明当前 student score 的跨任务排序和校准仍不够稳。
3. 因此 formal risk-control 不是“再调一个阈值”的捷径；它反而证明了现有阈值 sweep 缺乏可认证余量。下一步应提升 latent/process readout 或重新设计 calibration 数据协议，例如更大的 source calibration、跨 seed calibration、或把 risk constraint 作为训练目标的一部分。

Sources: Learn-then-Test https://arxiv.org/abs/2110.01052; SelectiveNet https://arxiv.org/abs/1901.09192; BranchyNet https://arxiv.org/abs/1709.01686.

## 11. Trajectory-Token Readout 实验

在 formal risk-control 诊断之后，本轮没有继续扫 scalar selector，而是测试 richer latent/process interaction。依据来自三类文献信号：

- Coconut 把 continuous thought 看成可承载多条候选路径的连续推理状态，说明 answer-enough 不应只看单个 block 表征，而要看推理状态如何从探索走向承诺。参考：https://arxiv.org/abs/2412.06769
- CODI 通过 hidden-state distillation 把语言 CoT 压到连续空间，支持“decoder/teacher 可以离线提供监督，在线 student 只读 latent/process”。参考：https://arxiv.org/abs/2502.21074
- CoLaR 强调 dynamic latent compression、非确定 latent head 和推理速度调节，进一步说明 block budget/trajectory 的动态性是核心，而不是固定阈值后处理。参考：https://arxiv.org/abs/2505.16552

实现：

```text
process_interaction_mode = trajectory_token
raw latent slots -> slot adapter + intra-block attention + PMA/last-slot pooling
trajectory token = MLP([pooled block state, current-prev block delta, process state])
pooled tokens + trajectory token -> causal inter-block Transformer -> multi-head readout
```

该 token 只使用当前 prefix 内可见 blocks 和 process features，不使用 decoder output、scorer、gold answer 或 future block，因此符合 P1 online-input 限制。

完整实验协议：

- 任务：official8 full LOTO，seed66/67/68。
- 训练目标：`answer_identity_action + completion_risk`。
- 训练数：`24` 个正式训练，全部 SwanLab cloud + CUDA，`valid_interval=50`，保存 `best_checkpoint.pt` 和 `last_checkpoint.pt`。
- 训练根目录：

```text
/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed{66,67,68}_d64_pma4_trajtok_answer_identity_action_completionrisk_20260527
```

- 评估协议：target-task valid calibration，cap128，5 个 calibration subsample seeds，boundary-risk penalty `0.2`，本地-only eval。
- 聚合脚本：

```text
/data1/luyifei/drla/drla/scripts/aggregate_cola_latent_halt_student_subseed_loto.py
```

- 聚合 summary：

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_trajtok_answer_identity_action_completionrisk_boundarypen02_cross_seed_20260527/summary.json
```

同口径对比：

| Route | Losses | Mismatches | Blocks | 解释 |
|---|---:|---:|---:|---|
| old `answer_identity_action + completion_risk` | `47` | `617` | `1.742/4` | 旧 single-student balanced route |
| `trajectory_token + answer_identity_action + completion_risk` | `41` | `806` | `1.711/4` | 更便宜、loss 更少，但 mismatch 更高 |
| `answer_identity_halt + completion_risk` strict | `31` | `699` | `1.824/4` | loss 更低，但 block 更贵 |

按任务拆解：

| Task | Old loss | Traj loss | Old mismatch | Traj mismatch | 结论 |
|---|---:|---:|---:|---:|---|
| LAMBADA | `32` | `28` | `59` | `64` | 小幅改善 prefix loss |
| MMLU | `0` | `5` | `5` | `14` | `mmlu::5615` empty-answer risk 重新出现 |
| HellaSwag | `0` | `6` | `116` | `300` | continuation/incomplete sentence 风险增加 |
| SQuAD | `15` | `2` | `426` | `423` | 明显改善 answer-boundary loss |

解释：

1. trajectory token 是正向信号：它把 SQuAD loss 从 `15` 降到 `2`，说明 block-to-block latent delta 确实能帮助读出 answer-boundary。
2. 它不是净胜：MMLU/HellaSwag 新增 loss，且 mismatch 从 `617` 增到 `806`。这说明更激进地读出 trajectory 会更早停，但不一定保留 answer text identity。
3. Wilson risk-control 仍失败。Trajectory summary 在 `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_trajtok_answer_identity_action_completionrisk_wilson_risk_control_20260527/summary.json`；在 128-shot target calibration folds 下，Wilson loss targets `<=0.00025/0.0005/0.001/0.002` 都无法覆盖完整 `120/120` folds。经验 zero-loss-min-block 也有 `442` held-out losses，不能作为 safety policy。

下一步判断：

- 不应把 `trajectory_token` 直接设为默认主线；它应作为下一代学生模型的组件候选。
- 更合理的下一步是训练 trajectory-level answer identity/stability objective：让模型显式预测“当前 answer identity 是否会被后续 block 改写/补全”，而不是只用 action label + completion head 间接表达。
- 另一路是扩大或重设 calibration 数据协议，否则 formal risk-control 仍会因为 finite valid size 无法认证低 loss 风险。

## 12. Trajectory-Level Answer Identity/Stability Teacher

上一节的结论已经被本轮实验直接验证：只给 trajectory token 加 action/completion 目标，会更早停但更容易改写 answer identity；因此这次把 decoder-as-teacher 信号改成更贴近最终需求的中间目标。

### 12.1 设计依据

这个目标不是把 decoder 信号拿来做在线特征，而是把训练好的 decoder 当成离线 teacher：P1 student 在线仍然只能看当前 prefix 内的 latent blocks 和 latent/process features，但训练时学习一个“当前 latent prefix 是否已经编码出稳定 answer identity”的轻量读出器。

这和几篇相关工作的可借鉴点是一致的：

- Coconut 说明 continuous thought 的有效性要看状态轨迹如何从探索走向可答状态，不能只看单点 latent。
- CODI 支持用 teacher hidden/state 或输出侧监督蒸馏连续空间中的推理状态。
- CoLaR 的 dynamic termination / latent compression 提醒我们，halt 目标应是 answer-state readiness 与 cost 的共同建模。
- Latent space survey 的风险提示是：latent 可控性与可评估性必须通过 probe、teacher label、calibration audit 和失败样本解释闭环确认，不能靠小实验正负结果下结论。

### 12.2 实现

新增训练开关：

```text
--use-answer-identity-stability
--selection-metric readiness_prediction_change_completion_identity_mean_auroc
```

新增 target：

```text
answer_identity_stability = 1
iff current scored_prediction == rollout prediction-stability / final reference
```

这意味着 student 学到的不是“是否正确”，而是“当前 block 的 answer identity 是否已经稳定到未来不会被补全或改写”。它仍然是 decoder/text-derived teacher target；不能把 `scored_prediction`、prediction-stability、final answer 或 official correctness 作为在线输入。

模型结构沿用上一轮：

```text
latent slots
-> slot adapter + intra-block attention + PMA4 + last-slot
-> trajectory token from pooled state + previous-block delta + process state
-> causal inter-block Transformer
-> multi-head readout
```

### 12.3 完整实验协议

- 任务：official8 full leave-one-task-out。
- Seeds：`66, 67, 68`。
- 训练数：`24` 个正式训练。
- 训练约束：全部 CUDA/GPU + SwanLab cloud，`valid_interval=50`，写 `metrics.jsonl`、`best_checkpoint.pt`、`last_checkpoint.pt`。
- Eval：`120` 个 local-only eval，target-task valid calibration，cap128，5 个 calibration subseeds，boundary-risk penalty `0.2`。

Artifacts:

```text
training roots:
/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed{66,67,68}_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_20260527

eval roots:
/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed{66,67,68}_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_targetcal_cap128_boundarypen02_subseeds_20260527

aggregate:
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_trajtok_answer_identity_action_completionrisk_identitystable_boundarypen02_cross_seed_20260527/summary.json
```

### 12.4 结果

| Route | Losses | Mismatches | Blocks | 结论 |
|---|---:|---:|---:|---|
| old `answer_identity_action + completion_risk` | `47` | `617` | `1.742/4` | 旧 action route |
| `trajectory_token + answer_identity_action + completion_risk` | `41` | `806` | `1.711/4` | 便宜但 mismatch 更差 |
| learned action->halt v2 cost-limited | `10` | `465` | `1.859/4` | mismatch 更低但 loss 更多 |
| new identity-stability trajectory student | `4` | `606` | `1.812/4` | 当前 P1 student-only 低 loss/cost 最好点 |
| v2 safety | `0` | `130` | `2.722/4` | 更安全但明显更贵 |

Task-level repeated-sample loss 分布：

| Task | Losses | Mismatches | Avg blocks |
|---|---:|---:|---:|
| LAMBADA | `0` | `5` | `3.254` |
| MMLU | `2` | `7` | `1.456` |
| OBQA | `0` | `0` | `1.209` |
| HellaSwag | `0` | `93` | `1.799` |
| RACE | `0` | `5` | `1.583` |
| SIQA | `0` | `0` | `1.337` |
| SQuAD | `2` | `494` | `1.864` |
| StoryCloze | `0` | `2` | `1.565` |

4 个 repeated losses 实际对应 2 个 unique boundary misses：

- seed67 SQuAD：block1 为空答案，future/final 为 `1873`。
- seed68 MMLU：block1 为空答案，future/final 为 `C`。

这说明新 head 把大部分 prefix/completion 风险压住了，但低阈值 calibration 仍会放过空答案边界。

### 12.5 诊断和边界

额外 local-only 诊断：

```text
hard contentful>=0.5 gate:
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_trajtok_answer_identity_action_completionrisk_identitystable_contentful05_boundarypen02_cross_seed_20260527/summary.json
```

结果为 `0` losses / `151` mismatches / `2.924/4` blocks。它证明 residual risk 主要是 empty-answer boundary，但这个 guard 比 prediction-stability 还贵，不应作为默认路线。

Formal risk-control 仍失败：

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_trajtok_answer_identity_action_completionrisk_identitystable_wilson_risk_control_20260527/summary.json
```

Wilson loss targets `<=0.00025/0.0005/0.001/0.002` 都选不出完整 `120/120` folds。结论是：answer identity/stability 是有效 teacher signal，但当前 valid calibration size 与 score 排序还不足以给严格低风险认证。

### 12.6 当前判断

1. 新结果是 P1 student-only 的实质进展，不是 smoke test。
2. 它支持“用 decoder 训练轻量 early-stop decoder/proxy，再在线只读 latent”的路线。
3. 它还不是最终 agent latent communication 答案，因为训练 label 仍来自 decoder/text teacher，且 calibration 仍靠离线阈值搜索。
4. 下一步应集中处理 empty-answer boundary、text-identity mismatch 和 formal calibration，而不是继续做小的 scalar threshold sweep。

## 13. Empty-Answer Auxiliary Head 负结果

### 13.1 为什么跑这一组

上一组 identity-stability route 只剩 `4` 个 repeated losses，而且它们对应两个 unique empty-answer boundary misses。因此最直接的假设是：加入一个显式 `empty_answer_risk` teacher head，可能让 student 在 latent 上读出“当前 answer 还没有成形”的状态。

这组实验不是小样本验证，而是完整 official8 full LOTO：

- 训练：seed66/67/68 x 8 held-out tasks，共 `24` 个正式训练。
- 训练约束：全部 CUDA/GPU + SwanLab cloud，`valid_interval=50`，`metrics.jsonl`，`best_checkpoint.pt`，`last_checkpoint.pt`。
- Eval：`120` 个 local-only target-calibration eval，cap128，5 个 calibration subseeds，boundary penalty `0.2`。

Artifacts:

```text
training roots:
/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed{66,67,68}_d64_pma4_trajtok_answer_identity_action_completionrisk_emptyrisk_identitystable_20260527

eval roots:
/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed{66,67,68}_d64_pma4_trajtok_answer_identity_action_completionrisk_emptyrisk_identitystable_targetcal_cap128_boundarypen02_subseeds_20260527

aggregate:
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_trajtok_answer_identity_action_completionrisk_emptyrisk_identitystable_boundarypen02_cross_seed_20260527/summary.json
```

### 13.2 结果

| Route | Losses | Mismatches | Blocks | 结论 |
|---|---:|---:|---:|---|
| identity-stability route | `4` | `606` | `1.812/4` | 当前最好 P1 student-only 点 |
| + `empty_answer_risk` | `24` | `623` | `1.829/4` | 更贵且更不安全 |

按 task，loss 主要落在：

- LAMBADA: `5` losses
- HellaSwag: `5` losses
- SQuAD: `14` losses

unique 失败样本只有 `5` 个，但每个被 calibration subseeds 重复选中：

| Task | Early answer | Stable/final answer | 类型 |
|---|---|---|---|
| LAMBADA | `quarant` | `quarantined` | word prefix |
| HellaSwag | `... walking` | `... walking him.` | continuation |
| SQuAD | `John` | `John Vanderbank's workshop` | entity prefix |
| SQuAD | `27` | `27 September 2001` | date prefix |
| SQuAD | `Metro Trains` | `Metro Trains Melbourne` | entity prefix |

这说明新增 head 没有解决“answer enough”的核心问题，反而把错误从 empty-answer 转成 prefix/continuation identity risk。

### 13.3 Calibration 诊断

在有 loss 的 folds 中，selected threshold 经常是：

```text
empty_answer_risk_threshold = 1.0
answer_identity_stability_threshold = 0.1
contentful_threshold = 0.0
```

也就是说，valid calibration 为了省 block 常常选择完全放开 empty-risk gate。即便强行做 post-hoc empty threshold cap，也不理想：

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_trajtok_answer_identity_action_completionrisk_emptyrisk_identitystable_empty_cap_diagnostic_20260527/summary.json
```

只有 `empty_answer_risk_threshold <= 0.005` 能覆盖完整 `120/120` folds 且达到 `0` losses，但成本升到 `3.808/4` blocks，几乎退回 final-block policy。

Formal risk-control 也没有改善：

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_trajtok_answer_identity_action_completionrisk_emptyrisk_identitystable_wilson_risk_control_20260527/summary.json
```

Wilson loss targets `<=0.00025/0.0005/0.001/0.002` 全部是 `0/120` complete folds；经验 zero-loss-min-block 仍有 `139` held-out losses。

### 13.4 文献解释

这次负结果和 auxiliary loss / multi-task negative transfer 文献一致：辅助目标如果只是局部 proxy，而不是和主 decision 同构，可能通过 shared representation 改坏主任务。`empty_answer_risk` 是一个太窄的 proxy，它只覆盖“空答案”，但真实风险是 prefix/continuation answer identity 尚未稳定。

参考点：

- Adapting Auxiliary Losses Using Gradient Similarity 指出辅助 loss 是否有益取决于它和主任务梯度是否一致；冲突辅助信号会伤害主目标。https://arxiv.org/abs/1812.02224
- Multi-task learning negative transfer 相关工作强调固定辅助权重和不匹配辅助任务可能产生 negative transfer。https://arxiv.org/abs/2012.09575
- Early-Exiting with Risk Control / Learn-then-Test 的启发是：早停要作为 risk-control 问题处理，不能只靠 confidence/proxy threshold。https://openreview.net/pdf?id=hACHuDzi1U 和 https://arxiv.org/abs/2110.01052

### 13.5 下一步判断

1. `empty_answer_risk` head 不应成为默认路线。
2. 当前最好点仍是 identity-stability route：`4` losses / `606` mismatches / `1.812/4` blocks。
3. 下一步应把 prefix/continuation answer-identity risk 设计成与 halt action 同构的目标，例如直接预测“若现在停，answer identity 会不会被未来 block 补全/改写”，而不是继续堆窄二分类头。
4. Calibration 需要从“选最低 block 的 permissive threshold”转向更强的 risk-control protocol；当前 128-shot target calibration 对极低 loss 风险没有认证能力。
