# P2-D2 Locked Split Protocol

更新日期：2026-06-01

> 状态：当前 P2 benchmark 修复协议。本文固定 7 个候选任务的 calibration / held-out partition。后续 prompt/protocol repair 只能使用 calibration；held-out 只用于重新验证 capability gate 或进入 P2 主表前的 locked evaluation。

## 1. 背景

P2-D1 full capability gate 的结果是：

```text
/data1/luyifei/drla/outputs/p2_capability_gate/aggregate_full_20260601

admitted_tasks = []
```

这说明当前 official CoLA 权重加通用 prompt/protocol 无法直接承载这些新 P2 benchmark。下一步不能直接跑 latent-vs-text 主表，而应先在 calibration split 上修 prompt/protocol，再用 held-out split 重新验证。

## 2. Split Artifact

构建脚本：

```text
/data1/luyifei/drla/drla/scripts/build_cola_p2_locked_splits.py
```

输出目录：

```text
/data1/luyifei/drla/outputs/p2_capability_gate/locked_splits_seed20260602_20260601
```

输入数据：

```text
/data1/luyifei/drla/outputs/p2_capability_gate/data_20260601/p2_candidate_benchmarks.jsonl
sha256 = fffe88201e6643a57541c2f986496a7b836f1c1112629500a49afc3b9f623fd0
```

Split config:

```text
split_seed = 20260602
calibration_fraction = 0.2
min_calibration_per_task = 20
max_calibration_per_task = 300
multiple_choice tasks are stratified by ground_truth label
```

## 3. Sizes

| Task | Type | Total | Calibration | Held-out |
|---|---:|---:|---:|---:|
| arc_easy | multiple_choice | 570 | 114 | 456 |
| arc_challenge | multiple_choice | 299 | 60 | 239 |
| gpqa_diamond | multiple_choice | 198 | 40 | 158 |
| medqa | multiple_choice | 1273 | 255 | 1018 |
| gsm8k | numeric | 1319 | 264 | 1055 |
| humanevalplus | code | 164 | 33 | 131 |
| mbppplus | code | 378 | 76 | 302 |

Total:

```text
calibration = 842
heldout = 3359
total = 4201
overlap = 0
```

## 4. Usage Rules

Calibration split:

```text
May be used for prompt/protocol repair.
May be used for parser audit.
May be used to select answer format templates and role contracts.
May be used for error taxonomy.
```

Held-out split:

```text
Must not be inspected at sample level during repair.
Must not be used to choose prompts, parsers, role wording, thresholds, or budgets.
May be used only for locked capability gate after repair.
May be used for P2 main text-vs-latent tables only after the repaired protocol passes held-out gate.
```

Historical note:

```text
P2-D1 already ran a baseline full gate on all rows before this split existed.
From this point onward, prompt/protocol decisions must be based only on
calibration rows and aggregate P2-D1 facts, not held-out sample-level outputs.
```

## 5. Next Step

P2-D3 should repair the benchmark protocol on calibration only:

```text
task-specific answer formatting
CoLA-compatible prompt templates
role contract simplification
parser robustness audit
shorter answer-only generations where appropriate
code prompt repair before any execution-gate claim
```

P2-D4 then reruns locked held-out capability gate. Only tasks that pass both:

```text
Single CoLA Solver gate
Role TextMAS gate
```

can enter P2 main text-vs-latent communication experiments.
