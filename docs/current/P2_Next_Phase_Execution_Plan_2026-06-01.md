# P2 下一阶段完整执行方案

更新日期：2026-06-01

> 状态：历史阶段性执行锁定方案。本文记录 P2-D3.1 到 Branch B Family 1
> 之前的 gate 纪律和禁止事项，但“继续 repair Role TextMAS -> held-out gate”
> 的下一步表述已被
> `/data1/luyifei/drla/docs/current/P2_Post_Family1_Complete_Execution_Plan_2026-06-01.md`
> supersede。Branch B Family 1 已停止；当前不得直接跑 held-out、P2 main table
> 或 fuser/adapter training。

## 1. 当前事实

已经完成的事实：

```text
P1:
  locked evaluation completed.
  P1 latent-readiness / halt student 已经充分学习 P0 teacher 信号。

P2 packet:
  v1/v2 packet build, distribution audit, decoder-free receiver compatibility
  已完成，证明 matched latent packet 在受控诊断下有可读性信号。

P2-D official8:
  corrected channel-equivalent official8 结果只能作为 channel diagnostic。
  official8 不再作为真实 MAS 主 benchmark。

P2-D1 full capability gate:
  7 个候选任务 admitted_tasks = []。

P2-D2 locked split:
  calibration = 842
  heldout = 3359
  overlap = 0

P2-D3 initial prompt repair:
  cola_fewshot_v1 没有改善候选集。
  GPQA-Diamond generic_v1 只通过 single calibration。
  Role TextMAS 仍失败。

P2-D3.1 answer-state repair:
  answer_state_v1, answer_state_structured_v1, and role_plan_ignore_v1
  已在 calibration 上评估。
  structured answer-state 减少 raw role-message 污染，但 admitted_tasks 仍为 []。
  role_plan_ignore_v1 也为负，提示 prompt-only repair 证据弱。
```

当前最重要结论：

```text
现在不能把新 benchmark 上的低分解释成 latent communication 失败。
当前瓶颈首先是 official CoLA 权重 + 当前 prompt/protocol 的 base capability 和 Role TextMAS 可用性。
```

## 2. 总目标

P2 的目标不是证明 DiT LoRA 提高 CoLA benchmark 精度，也不是在 official8 上继续刷 receiver-only diagnostic。

P2 主目标：

```text
在 same-substrate CoLA A -> CoLA B 条件下，
验证 role-conditioned agent-to-agent communication 中，
latent working memory 是否能作为 text message 的替代或补充，
并在 matched budgets、matched roles、matched scorer 下带来可解释的质量/成本前沿。
```

主 claim 分级：

| Claim | 最低证据 |
|---|---|
| Latent packet readable | matched latent beats metadata/corrupt controls |
| Latent packet useful | B_latent beats B_none under same role protocol |
| Latent competitive with text | B_latent 与 B_text 在质量/成本上 Pareto competitive |
| True MAS utility | 任务结构本身需要角色分工或信息分布，latent 改善最终 Solver 输出 |

## 3. 不可退回的旧路线

禁止把以下内容重新作为主线：

```text
official8 solver-to-solver message_only 主表
GSM8K 小样本 MVP / overfit 证明
legacy Stage B/C small prior
直接用 selected_prediction / scorer-extracted answer 作为 Agent-A message
在 admitted_tasks=[] 的任务上跑 latent-vs-text 主表
没有 held-out gate 就把 calibration 结果写成 paper table
```

这些内容仍可作为历史、smoke、diagnostic 或 engineering audit，但不能承载 P2 主 claim。

## 4. 执行阶段

### P2-D3.1 Role Protocol Repair

目的：

```text
在 calibration split 上修复 Role TextMAS 协议，
让至少一个任务同时通过 Single CoLA Solver + Role TextMAS calibration gate。
```

优先实现 `answer_state_v1`：

```text
Planner:
  输入 q。
  输出 compact candidate answer state，而不是长篇 reasoning。

Critic:
  输入 q + planner candidate answer state。
  输出 corrected candidate / uncertainty / missing evidence。

Refiner:
  输入 q + previous answer states。
  输出 refined candidate answer state。

Solver:
  输入 q + compact upstream answer state。
  输出 final answer only。
```

为什么先做 compact answer-state：

```text
当前 Role TextMAS 主要失败在 parseability 和角色输出噪声。
如果上游 agent 输出大段自由文本，CoLA 的非 instruction-tuned 特性会被放大。
compact answer-state 更接近 agent-to-agent state interface，也更接近后续 latent packet 需要表达的内容。
```

使用边界：

```text
只能读取 calibration.jsonl。
可以看 calibration 错误类型和 aggregate。
禁止查看 heldout sample-level outputs。
纯 eval local-only，swanlab_mode=disabled。
```

通过标准：

```text
至少一个任务:
  single_gate_pass = true
  role_textmas_gate_pass = true
  parseable_rate >= 0.90
  nonempty_rate >= 0.95
  accuracy > random/task floor + margin
```

### P2-D4 Held-out Locked Capability Gate

触发条件：

```text
P2-D3.1 在 calibration 上找到候选协议。
```

执行：

```text
在 heldout.jsonl 上重跑 Single CoLA Solver + Role TextMAS。
不能再调 prompt、parser、阈值或预算。
```

结果分类：

| Held-out 结果 | 下一步 |
|---|---|
| 有 admitted task | 进入 P2-E text-vs-latent 主实验 |
| 只有 single pass，role fail | 继续 calibration-only role protocol repair |
| single 也 fail | 不在该任务上做通信主表，转入 substrate adaptation 或换 benchmark |

### P2-E Channel-correct Text-vs-Latent MAS

只对 held-out admitted tasks 运行。

统一协议：

```text
Single:
  q -> Solver -> final answer

TextMAS:
  Planner(q) -> Critic(q, planner_text_state)
  -> Refiner(q, text_states) -> Solver(q, final_text_state)

LatentMAS no-fuser:
  Planner/previous role emits latent working memory。
  Next role receives q + role instruction + latent packet。
  Agent B must generate its own final output。

Corrupt controls:
  wrong_sample, same_task_wrong_sample, wrong_block, shuffle, noise, rotation
```

关键边界：

```text
Scorer 只能看 final Solver output。
Agent B 不能直接收到 selected_prediction、official scorer 输出、gold answer。
Text channel 只能使用 raw boundary message / compact answer-state，不使用 scorer-extracted answer。
Latent channel 优先使用 context_plus_thought 或 full_working_memory。
thought_only 只是 ablation。
```

主要指标：

```text
accuracy / task score
paired delta vs Single
paired delta vs TextMAS
matched latent vs corrupted controls
token count
latent block count
wall-clock time
parseable / nonempty / format adherence
confidence interval
failure taxonomy
```

最低继续条件：

```text
B_latent_matched beats B_corrupt controls。
B_latent_matched beats or ties B_none under paired comparison。
```

若 latent 不如 text，但稳定强于 corrupt：

```text
结论是 latent carries usable signal but no text superiority yet。
随后才允许触发 fuser/adapter，而不是直接宣称失败。
```

### P2-F Lightweight Fuser / Receiver Adapter

触发条件：

```text
no-fuser latent 明显强于 corrupt，但弱于 TextMAS；
或 wrong_block / same-task wrong-sample 异常强；
或 full_working_memory 可用而 thought_only 不可用。
```

训练目标：

```text
A_latent + B_context -> B_usable_state
```

可用 teacher：

```text
TextMAS hidden/logit/answer-state teacher
matched-vs-corrupt contrastive labels
P1/P0 readiness/certification signals
decoder/probe signals for offline distillation only
```

线上禁止：

```text
不能把 decoded A replay text、gold、scorer output、selected_prediction 作为 receiver 输入。
```

训练规范：

```text
CUDA only。
SwanLab cloud。
metrics.jsonl。
valid_interval <= 10 step。
best_checkpoint.pt + last_checkpoint.pt。
每个阶段性优化后必须做全局复盘，不允许只看一个小曲线继续局部调参。
```

### P2-G True MAS Benchmark

在同一套协议稳定后，才进入更强 claim：

```text
evidence-split multi-hop QA
planner-coder-tester-reviewer code workflows
distributed information gathering
role-specific expert aggregation
```

准入原则：

```text
任务必须天然需要 agent 分工。
CoLA 或选定 substrate 必须先过 single/role capability gate。
如果 CoLA 权重无法承载任务，不能用该任务证明 latent communication 失败。
```

## 5. 如果新 benchmark 继续不过 gate

这不是退化，而是科学分流。

### 分支 A: CoLA substrate adaptation

适用：

```text
目标仍然是 same-substrate CoLA latent communication，
但 official CoLA 权重对候选 benchmark 不可用。
```

允许做：

```text
prompt/protocol repair
answer-state interface repair
task-format adapter
LoRA/adapter/fuser 作为 substrate-capability or receiver-interface adaptation
```

禁止做：

```text
把 adapter 提升官方 benchmark accuracy 当主目标。
把 adaptation 后的结果与 frozen CoLA official numbers 混成同一 baseline。
没有训练日志、best checkpoint、held-out split 就报告主结果。
```

### 分支 B: 换入能力匹配的 MAS benchmark

适用：

```text
当前候选任务与 CoLA 权重能力不匹配，
但还有更适合 CoLA answer format 或更自然分工的任务。
```

要求：

```text
先建 manifest。
先跑 Single + Role TextMAS gate。
先锁 calibration/held-out。
不能因为想快就回到小样本 MVP。
```

### 分支 C: 外部 capable text agents

适用：

```text
要先验证 MAS protocol 和 benchmark 本身是否合理。
```

边界：

```text
这只能验证 text MAS task/protocol，不证明 CoLA latent communication。
如果后续要接 CoLA latent，需要 translator/shared codec/adapter，并另行声明 claim 范围。
```

## 6. 文档与 Artifact 纪律

每个阶段必须留下：

```text
input manifest
script path
exact command or config
output directory
metrics.jsonl or summary.json
split seed
prompt/protocol version
scorer/parser version
leakage audit result
结论边界
```

训练实验额外要求：

```text
SwanLab cloud run id
best_checkpoint.pt
last_checkpoint.pt
valid interval <= 10 step
```

纯评估要求：

```text
swanlab_mode=disabled
no optimizer
no backward
不创建无训练曲线的 SwanLab run
```

## 7. 当前 supersede 后的下一步

已完成：

```text
1. answer_state_v1 implemented and evaluated on GPQA calibration.
2. answer_state_structured_v1 implemented and evaluated on GPQA, ARC-E/C,
   and MedQA calibration.
3. aggregate_calibration_protocol_repair_answer_state_20260601 written.
4. audit_protocol_repair_failures_answer_state_20260601 written.
5. admitted_tasks remains [].
```

supersede 后的下一步：

```text
1. Branch B Family 1 official8-compatible calibration 已完成，admitted_tasks=[]。
2. Official8 native prompt/eval alignment audit 已完成，native Single Solver
   仍 admitted_tasks=[]。
3. Branch B Family 1 已满足 stop condition。
4. 当前 canonical next-plan 是：
   /data1/luyifei/drla/docs/current/
   P2_Post_Family1_Complete_Execution_Plan_2026-06-01.md
5. 不再执行本文早前的 held-out gate / P2-E main table / fuser trigger，
   除非 post-Family1 方案中的分支条件被重新满足。
```

P2-D4 branch decision audit:

```text
doc:
  /data1/luyifei/drla/docs/current/
  P2_D4_Branch_Decision_Audit_2026-06-01.md

historical recommendation:
  Branch B first was executed as Family 1 and stopped.

current boundary:
  Do not start Branch A/C/Family2 execution until explicitly selected.
  Recommended scientific order in the post-Family1 plan is Branch C -> Branch A.
```

Branch B Family 1 executed and stopped:

```text
doc:
  /data1/luyifei/drla/docs/current/
  P2_Branch_B_Execution_Plan_2026-06-01.md

first candidate family:
  official8-compatible role candidates:
    obqa, mmlu, race, hellaswag, siqa, story_cloze

prepared data:
  /data1/luyifei/drla/outputs/p2_benchmark_redesign/
  official8_role_candidates_20260601

prepared rows:
  33296

meaning:
  data was used for calibration-only capability gate.
  It did not produce an admitted benchmark and is not a text-vs-latent result.
```

当前仍不执行：

```text
不跑 held-out。
不跑 latent-vs-text 主表。
不训练 fuser。
不声称 CoLA 权重已经满足新 benchmark。
```

## 8. 判断标准

如果下一步成功：

```text
至少一个 admitted held-out benchmark -> 进入 P2-E。
```

如果下一步失败：

```text
记录失败类型:
  base solver floor
  role protocol collapse
  parser/format failure
  task/substrate mismatch

然后选择 substrate adaptation 或 benchmark redesign，
而不是把 floor result 解释成 latent communication failure。
```
