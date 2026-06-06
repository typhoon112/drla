# P2-D1 Capability Gate Report

更新日期：2026-06-01

> 状态：当前 P2 benchmark 准入报告。本文记录 official CoLA 权重在候选 P2 benchmark 上的 no-training 能力门结果。结论是：当前 7 个候选任务没有任何一个通过“Single CoLA Solver + Role TextMAS”双门准入，因此不能直接进入 P2 text-vs-latent MAS 主表。

## 1. 目的

P2 主线不能再把 official8 solver-to-solver diagnostic 当作真实 MAS 证据。新 benchmark 必须先证明 CoLA base 权重能在该任务上产出可解析、非随机、可作为下游 Agent-B 输入的答案。

Capability gate 只回答一个问题：

```text
当前 official CoLA 权重是否足以承载该 benchmark 的 P2 MAS 主实验？
```

它不回答：

```text
latent communication 是否成功。
latent 是否优于 text。
P1/P2 架构是否失败。
```

## 2. 数据

数据准备脚本：

```text
/data1/luyifei/drla/drla/scripts/prepare_cola_p2_candidate_benchmarks.py
```

完整数据目录：

```text
/data1/luyifei/drla/outputs/p2_capability_gate/data_20260601
```

Prepared rows:

| Task | Source | Split | Rows | Type |
|---|---:|---:|---:|---|
| arc_easy | allenai/ai2_arc / ARC-Easy | validation | 570 | multiple_choice |
| arc_challenge | allenai/ai2_arc / ARC-Challenge | validation | 299 | multiple_choice |
| gsm8k | openai/gsm8k / main | test | 1319 | numeric |
| mbppplus | evalplus/mbppplus | test | 378 | code |
| humanevalplus | evalplus/humanevalplus | test | 164 | code |
| gpqa_diamond | hendrydong/gpqa_diamond_mc | test | 198 | multiple_choice |
| medqa | GBaker/MedQA-USMLE-4-options | test | 1273 | multiple_choice |

Total:

```text
4201 rows
7 tasks prepared
0 unprepared tasks
```

## 3. 协议

评估脚本：

```text
/data1/luyifei/drla/drla/scripts/run_cola_p2_capability_gate.py
```

聚合脚本：

```text
/data1/luyifei/drla/drla/scripts/aggregate_cola_p2_capability_gate.py
```

聚合结果：

```text
/data1/luyifei/drla/outputs/p2_capability_gate/aggregate_full_20260601
```

Gate modes:

```text
Single CoLA Solver:
  q -> CoLA -> answer

Role TextMAS:
  Planner(q) -> Critic(q, planner) -> Refiner(q, planner, critic) -> Solver(q, refiner)
```

准入要求：

```text
Single gate_pass = true
Role TextMAS gate_pass = true
not a smoke run
no gold/scorer/selected_prediction in online prompts
code tasks must use execution gate, not syntax-only pre-gate
```

本轮全部是 no-training local-only evaluation：

```text
swanlab_mode=disabled
no optimizer
no backward
no checkpoint
```

## 4. 结果

聚合 summary：

```text
/data1/luyifei/drla/outputs/p2_capability_gate/aggregate_full_20260601/summary.json
```

Task summary：

| Task | Samples | Single Acc | Single Parse | Single Pass | Role Acc | Role Parse | Role Pass | Admitted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| arc_easy | 570 | 20.18% | 83.86% | no | 14.21% | 57.37% | no | no |
| arc_challenge | 299 | 23.08% | 87.63% | no | 10.70% | 59.20% | no | no |
| gpqa_diamond | 198 | 21.21% | 84.34% | no | 12.12% | 40.40% | no | no |
| medqa | 1273 | 23.02% | 92.77% | no | 17.83% | 61.74% | no | no |
| gsm8k | 1319 | 2.05% | 99.77% | yes | 1.74% | 82.34% | no | no |
| humanevalplus | 164 | 0.00% | 0.00% | no | 0.00% | 0.00% | no | no |
| mbppplus | 378 | 0.00% | 0.00% | no | 0.00% | 0.00% | no | no |

Admitted tasks:

```text
[]
```

## 5. 解释

本轮最重要的结论不是“latent communication 失败”，而是：

```text
当前 official CoLA 权重 + 当前通用 prompt protocol
不能直接承载这些新 P2 benchmark 的主实验。
```

具体看：

```text
ARC-E / ARC-C / GPQA-Diamond / MedQA:
  Single accuracy 低于或接近 multiple-choice 随机 floor，parseable 也不足。

GSM8K:
  Single numeric parseable 很高，且勉强超过当前极低 margin；
  但 Role TextMAS parseable/accuracy 都不达标，因此不能作为 MAS 主表任务。

HumanEval+ / MBPP+:
  execution gate 下 parseable 和 pass 都为 0。
  当前 CoLA 不是可直接使用的代码生成 solver。
```

因此，后续不能在这些任务上直接做：

```text
TextMAS vs LatentMAS 主对比
latent > text claim
agent-to-agent latent communication paper table
```

否则会把 base model capability floor 错当成 communication-channel 结果。

## 6. 数据泄漏边界

本轮是一次 no-training benchmark gate，不更新模型、不调阈值、不选择 latent policy。

但后续如果要修 prompt 或协议，不能继续在这些 test rows 上反复调参后再把同一结果当主表。下一步必须建立：

```text
calibration/dev split:
  用于 prompt/protocol repair 与 parser audit。

locked held-out split:
  只用于最终 capability gate 或 P2 主表。
```

对于只有 test split 的任务，如 GPQA-Diamond、MBPP+、HumanEval+，需要在本地固定 deterministic calibration/held-out partition，并把 split seed 写进 artifact。

## 7. 下一步

合理路线不是马上训练 latent fuser，也不是在未准入任务上跑 P2 主表。下一步应先做 benchmark/protocol 修复：

```text
P2-D2:
  Build locked calibration/held-out partitions for the 7 candidate tasks.

P2-D3:
  Prompt/protocol repair on calibration only:
    task-specific answer formatting
    CoLA-compatible prompt templates
    role contract simplification
    parser audit

P2-D4:
  Re-run locked held-out capability gate.

Only if a task passes:
  run TextMAS vs LatentMAS under the same role protocol and budget.
```

如果修复后仍无任务准入，则 P2 主线应转向 naturally decomposable tasks with externally capable text agents, while keeping CoLA latent experiments as same-substrate diagnostics, or require CoLA-side instruction/task adaptation before MAS claims.

## 8. P2-D2 Locked Split

P2-D2 已完成：

```text
/data1/luyifei/drla/outputs/p2_capability_gate/locked_splits_seed20260602_20260601
```

```text
split_seed = 20260602
calibration = 842
heldout = 3359
overlap = 0
```

后续 prompt/protocol repair 只能使用 calibration。held-out 只用于修复后的 locked capability gate。
