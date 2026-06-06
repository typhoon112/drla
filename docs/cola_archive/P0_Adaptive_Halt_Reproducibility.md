# P0 Adaptive Halt 复现归档

更新时间：2026-06-06

## 阶段定位

P0 是 CoLA 线的 teacher / upper-bound 诊断阶段。它回答的问题是：

```text
在 official CoLA block-wise rollout 中，是否存在可以提前停止且尽量不损失答案正确性的 readiness 信号？
```

P0 不要求推理时 decoder-free。它允许使用 decoder probe、文本稳定性、scorer-derived prediction-change、answer-shape 等非 gold 或离线 derived 信号来构造 early-halt teacher。因此 P0 是 P1 的监督来源和安全上界，不是最终 agent-to-agent latent communication 策略。

## Canonical 结果

主结果路径：

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/
official8_full_b64_bs12_cross_task_joint_readiness_riskcap04_cross_seed_20260524/
summary.json
```

核心协议：

```text
tasks = lambada, mmlu, obqa, hellaswag, race, siqa, squad, story_cloze
trace seeds = 66, 67, 68
split = prepared full split
heldout protocol = leave-one-task-out
risk target = prediction_change
readiness thresholds = 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.65,0.7,0.75,0.8
risk threshold end = 0.4
risk threshold selection = min_blocks
require_zero_calibration_loss = true
contentful guard = true
single-choice guard scope = block <= 2
```

聚合指标：

| 指标 | 数值 |
|---|---:|
| Seeds | 3 |
| Samples per seed | 49,019 |
| Fixed-final accuracy | `21.59299% +/- 0.02872%` |
| Prediction-stability accuracy | `21.59571% +/- 0.02947%` |
| Risk-gated halt accuracy | `21.59639% +/- 0.02963%` |
| Prediction-stability avg blocks | `2.5116/4` |
| Risk-gated avg blocks | `2.1179/4` |
| Saving vs final | `1.8821` blocks |
| Saving vs prediction-stability | `0.3937` blocks |
| Losses vs final, all seeds | `0` |
| Losses vs prediction-stability, all seeds | `0` |

解释：P0 的主张不是 accuracy improvement，而是相对 prediction-stability 的安全省 block。aggregate 中存在少量 gains，但主结论必须基于 sample-level zero-loss 约束，而不是 gain/loss cancellation。

## 关键负结果

no-riskcap shape-feature 路线不能作为安全策略：

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/
official8_full_b64_bs12_cross_task_shape_features_fragmentguardv2_choice2_noriskcap_cross_seed_20260524/
summary.json
```

它平均 block 很低，但相对 prediction-stability 存在 `19` losses 和 `19` gains，属于 loss/gain cancellation。这个结果应保留为 failure-surface 诊断，不能替代 riskcap04。

## 依赖数据与中间产物

Official CoLA 8-task traces：

```text
/data1/luyifei/drla/outputs/cola_block_traces/
tasks_official8_full_b64_t16_seed66_bs12_merged_20260524

/data1/luyifei/drla/outputs/cola_block_traces/
tasks_official8_full_b64_t16_seed67_bs12_merged_20260524

/data1/luyifei/drla/outputs/cola_block_traces/
tasks_official8_full_b64_t16_seed68_bs12_merged_20260524
```

Official benchmark anchor：

```text
/data1/luyifei/drla/outputs/cola_official_benchmarks/
full_b64_bs12_trace_score_20260524/summary.json

/data1/luyifei/drla/outputs/cola_official_benchmarks/
full_b64_bs12_seed67_trace_score_20260524/summary.json

/data1/luyifei/drla/outputs/cola_official_benchmarks/
full_b64_bs12_seed68_trace_score_20260524/summary.json
```

P0 report and figures：

```text
/data1/luyifei/drla/docs/p0_reports/cola_adaptive_halt_paper_report_zh.md
/data1/luyifei/drla/outputs/paper_report_20260525/cola_adaptive_halt_paper_report_zh.pdf
/data1/luyifei/drla/outputs/paper_report_20260525/figure_data.json
/data1/luyifei/drla/outputs/paper_report_20260525/figures/
```

## 代码入口

数据准备与 trace：

```text
/data1/luyifei/drla/drla/scripts/prepare_cola_official_benchmarks.py
/data1/luyifei/drla/drla/scripts/collect_cola_block_traces.py
/data1/luyifei/drla/drla/scripts/merge_cola_block_trace_segments.py
/data1/luyifei/drla/drla/scripts/eval_cola_benchmarks.py
```

P0 label/model/eval：

```text
/data1/luyifei/drla/drla/scripts/build_cola_readiness_frontier.py
/data1/luyifei/drla/drla/scripts/train_cola_readiness_model.py
/data1/luyifei/drla/drla/scripts/train_cola_continuation_risk_model.py
/data1/luyifei/drla/drla/scripts/eval_cola_risk_gated_halt.py
/data1/luyifei/drla/drla/scripts/aggregate_cola_risk_gated_halt.py
/data1/luyifei/drla/drla/scripts/analyze_cola_risk_gated_halt_decisions.py
```

## 复现顺序

复现时先确认环境：

```bash
cd /data1/luyifei/drla
source /data1/luyifei/drla/scripts/activate_conda.sh
```

建议复现顺序：

1. 准备 official 8-task benchmark JSONL。
2. 收集或复用 seed66/67/68 full b64 bs12 block traces。
3. 用 official scorer 评估 final block baseline，确认 official benchmark anchor。
4. 从 traces 构建 oracle readiness/frontier labels。
5. 训练 readiness model 与 continuation-risk model。
6. 运行 risk-gated halt eval，使用 riskcap04、zero-calibration-loss、contentful 和 single-choice guard。
7. 聚合 cross-seed LOTO 结果，核对 canonical summary。

训练脚本必须使用 CUDA/GPU、SwanLab cloud、本地 `metrics.jsonl`、`best_checkpoint.pt`、`last_checkpoint.pt`。纯 eval / aggregation / figure 脚本必须本地执行，不应新建 SwanLab run。

## 可引用表述

可以写：

```text
P0 demonstrates that decoder/probe/scorer-derived readiness and prediction-change
risk signals can safely reduce CoLA block budget under offline LOTO replay.
```

不要写：

```text
P0 improves official CoLA benchmark accuracy.
P0 is decoder-free.
P0 is already the deployed latent communication policy.
```

## 残余风险

- P0 是 offline replay，不是 online generation-loop integration。
- P0 使用 decoder/text/probe/scorer-derived features，不能直接代表无 decoder 的 agent-to-agent latent communication。
- 三个 seeds 是同 prepared split 上的 trace/frontier variants，支持 replicated evidence，但不是独立数据集泛化证明。
