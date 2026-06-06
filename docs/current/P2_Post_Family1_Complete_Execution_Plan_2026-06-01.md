# P2 Family 1 Stop 后完整执行方案

更新日期：2026-06-01

> 状态：post-Family1 后的完整路线背景与执行依据。当前最高优先级锁定方案为
> `P2_Locked_Complete_Execution_Scheme_2026-06-01.md`；后续执行以该 locked
> scheme 为准。本文 supersede 早前
> `P2_Next_Phase_Execution_Plan_2026-06-01.md` 中“继续 repair Role TextMAS
> -> held-out gate”的下一步表述，也 supersede
> `P2_D4_Branch_Decision_Audit_2026-06-01.md` 中“Branch B first”的旧默认。
> Branch B Family 1 已执行并停止；后续不得直接跑 held-out、P2 main table、
> fuser/adapter training，除非本文的分支条件被满足并得到明确选择。

## 1. 当前结论

P2 的科学目标保持不变：

```text
same-substrate Cola A -> Cola B latent communication
role-conditioned MAS
capability-gated benchmark
matched TextMAS vs LatentMAS comparison
```

当前不可绕过的实验事实：

```text
P2-D1 7-task candidate gate:
  admitted_tasks = []

P2-D3/D3.1/D3.2 prompt/protocol repair:
  admitted_tasks = []

Branch B Family 1 official8-compatible calibration:
  admitted_tasks = []

Official8 native prompt/eval alignment audit:
  native Single Solver admitted_tasks = []
```

因此当前不能做：

```text
held-out gate
P2 text-vs-latent main table
latent fuser / adapter training
把低分解释成 latent communication failure
把 frozen official CoLA 当作已经满足新 MAS benchmark 的 substrate
```

根因判断：

```text
当前瓶颈不是 latent packet 是否存在信号，而是 benchmark/protocol/base capability
没有形成可解释比较前提。若 Single Solver 或 Role TextMAS 自身接近 floor，
communication medium 的优劣没有科学解释力。
```

## 2. 路线原则

### 2.1 先固定任务对象，再比较通信介质

P2 不能再把普通单问答硬拆成多 agent。主 benchmark 应该天然需要通信：

```text
partial-information / evidence-split reasoning
planner-coder-tester-reviewer code workflow
distributed state synthesis
role-specific expert aggregation
```

任务必须先证明：

```text
Single baseline above floor
Role TextMAS above floor
communication ablation causes measurable drop
parser/scorer stable
no gold/scorer leakage
locked calibration/held-out split
```

### 2.2 CoLA 权重不是所有新 benchmark 的免费 solver

CoLA 架构适合作为 latent substrate：

```text
Text VAE maps text <-> continuous latent sequence.
Block-causal DiT operates on latent sequence.
Same official CoLA A/B makes same-substrate handoff meaningful.
```

但当前 official CoLA checkpoint 不自动满足 ARC、GPQA、MedQA、GSM8K、
EvalPlus 或更强 MAS tasks。Family 1 和 native official8 audit 已经说明：
即使在 official8-compatible/native 口径下，frozen CoLA 也没有给出可直接进入
P2 main table 的 admitted task。

### 2.3 Latent 比较前必须有 text baseline

每个进入 P2 main 的任务至少需要：

```text
Single Solver:
  q -> final answer

TextMAS:
  role-conditioned agents exchange text state
  final Solver output is scored

LatentMAS:
  same roles and budgets
  exchange latent working memory / packet
  final Solver output is scored

Controls:
  none, wrong_sample, same_task_wrong_sample, wrong_block, shuffle/noise
```

Scorer 只能看 final Solver output。Agent B 不能看到 gold answer、scorer output、
selected_prediction 或 Agent-A decoded replay tokens 作为最终答案。

## 3. 推荐完整方案

当前推荐路线是：

```text
Phase C: 先用 capable text agents 锁定 true MAS benchmark/protocol
-> Phase A: 再把选定 benchmark 接回 CoLA substrate/interface adaptation
-> Phase E: 最后做 CoLA TextMAS vs LatentMAS 主比较
```

Branch B Family 2 只保留为诊断分支，不作为主线。

### Phase C: 外部 capable TextMAS benchmark/protocol validation

目的：

```text
把“什么任务真的需要 agent-to-agent communication”固定下来，
避免继续把 CoLA base floor 误判成 benchmark/protocol 或 latent channel 失败。
```

执行：

```text
C1. 选择 2-3 个 naturally decomposable tasks。
C2. 建 manifest、scorer、parser、leakage audit。
C3. 建 calibration/held-out split，held-out 只用于 locked eval。
C4. 跑 capable Single Solver 与 capable Role TextMAS。
C5. 做 communication ablation:
    no message / shuffled message / wrong evidence / compressed state。
C6. 只保留 Role TextMAS 相对 Single 有可解释增益，或至少不坍塌且
    ablation 有明显 drop 的任务。
```

候选任务族：

```text
evidence-split multi-hop QA:
  每个 agent 只拿到部分证据，Solver 必须整合上游信息。

planner-coder-tester-reviewer code workflow:
  Planner 产出规格，Coder 写代码，Tester 产出失败信号，Reviewer 修正。

distributed state synthesis:
  不同 agent 持有不同事实、表格或约束，最终答案需要汇总。
```

输出：

```text
locked benchmark manifest
prompt/protocol version
single/textMAS/ablation metrics
failure taxonomy
calibration/held-out split hash
no-leakage audit
```

注意：Phase C 只证明 benchmark/protocol 合理，不证明 CoLA latent communication。

### Phase A: CoLA substrate/interface adaptation

触发条件：

```text
Phase C 已经锁定至少一个合理 MAS benchmark；
或用户明确要求 ARC/GPQA/MedQA/GSM8K/EvalPlus 等任务不可替换。
```

目的：

```text
让 CoLA 在目标 benchmark 上具备基本 solver 和 role interface 能力，
再进入 latent-vs-text communication 比较。
```

执行顺序：

```text
A1. 构建 CoLA 版本 train/valid/test split。
A2. 训练 task-format adapter / LoRA，让 Single CoLA Solver 过 valid gate。
A3. 训练或调 role-state interface，让 Role TextMAS 过 valid gate。
A4. locked test gate，只报告 best checkpoint。
A5. 若 no-fuser latent handoff 可读但弱于 text，再训练 receiver/fuser。
```

训练规范：

```text
CUDA only
SwanLab cloud
metrics.jsonl
valid_interval <= 10 step
best_checkpoint.pt
last_checkpoint.pt
train/valid/test locked split
每个阶段性优化后先全局复盘，再决定下一轮
```

claim 边界：

```text
这是 adapted-CoLA latent communication，不是 frozen official CoLA no-fuser claim。
所有表格必须区分 frozen official CoLA、adapted CoLA、capable text agent。
```

### Phase E: CoLA TextMAS vs LatentMAS main comparison

触发条件：

```text
Single CoLA Solver gate pass
Role TextMAS gate pass
locked held-out/test split exists
no gold/scorer/selected_prediction leakage
```

主比较：

```text
Single CoLA Solver
CoLA TextMAS
CoLA LatentMAS no-fuser
CoLA LatentMAS with receiver/fuser, only if triggered
corrupt latent controls
```

主要指标：

```text
accuracy / task score
paired delta vs Single
paired delta vs TextMAS
matched latent vs corrupt controls
parseable / nonempty / format adherence
token count
latent block count
wall-clock cost
confidence interval
failure taxonomy
```

最低可发表结论分级：

```text
Readable:
  matched latent > corrupt controls.

Useful:
  matched latent > none under paired comparison.

Competitive:
  latent is Pareto competitive with text in quality/cost.

True MAS:
  task structure naturally requires communication and ablation hurts.
```

## 4. Branch B Family 2 的边界

Branch B Family 2 仅在以下条件下作为诊断：

```text
用户明确要求继续 frozen official CoLA。
任务先 robust pass native Single CoLA Solver gate。
Role TextMAS 再 pass calibration gate。
任务不能只是普通 QA 改 prompt，必须有 communication value。
```

它不作为当前主线，原因：

```text
Family 1 已经在 official8-compatible 与 native official8 口径下停止。
继续找 frozen-CoLA-friendly 小任务很容易退化成 channel diagnostic，
而不是 true MAS evidence。
```

## 5. 防扰乱规则

后续执行必须遵守：

```text
1. 没有 admitted task，不跑 P2 main table。
2. 没有 Single + Role baseline，不比较 latent vs text。
3. 没有 communication ablation，不声称 true MAS utility。
4. 没有 locked split，不报告 paper table。
5. 没有训练曲线的 eval 脚本，不上 SwanLab。
6. 任何训练必须 SwanLab cloud + metrics.jsonl + best/last checkpoint。
7. pure eval / audit / aggregation local-only with swanlab_mode=disabled。
8. decoder/gold/scorer/oracle 只允许 offline label/eval，不允许作为在线 receiver input。
9. 阶段性优化后必须做全局复盘，必要时查文献，再决定下一步。
```

## 6. 当前锁定的执行选择

2026-06-01 locked scheme 已将默认执行顺序固化为：

```text
Branch C first:
  先锁定 true MAS benchmark/protocol，
  再回到 CoLA substrate/interface adaptation。
```

备选分支只在用户明确改写主目标时启用：

```text
Branch A first:
  直接进入 CoLA adaptation。
  适合用户明确要求 CoLA latent substrate 立即成为下一阶段核心。

Branch B Family 2:
  只做 frozen-CoLA diagnostic，不建议作为主线。
```

在开始 Phase C 模型运行前，允许先做：

```text
文档整理
artifact 索引
已有结果聚合
不触碰 held-out 的 manifest/scorer 草案
不训练、不跑 main table 的代码卫生检查
```

下一步执行细节以
`/data1/luyifei/drla/docs/current/P2_Locked_Complete_Execution_Scheme_2026-06-01.md`
为准。
