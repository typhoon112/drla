# P2 Family 1 Stop 后分支决策备忘录

更新日期：2026-06-01

> 状态：post-Family1 分支决策备忘录。2026-06-01 locked scheme 已将默认路线
> 固化为 Branch C true MAS benchmark/protocol validation -> Branch A CoLA
> substrate/interface adaptation -> Phase E 主比较。本文不启动训练、不跑 held-out、
> 不进入 P2 main table，也不训练 fuser/adapter；它保留 Branch B Family 1
> 停止后的本地证据、外部文献依据和分支论证，防止后续实验退回不严谨局部试错。

## 1. 当前不可绕过的事实

P2 目标仍然是：

```text
same-substrate Cola A -> Cola B latent communication
role-conditioned MAS
capability-gated benchmark
matched TextMAS vs LatentMAS comparison
```

但当前 evidence chain 是：

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

因此当前禁止：

```text
held-out gate
P2 text-vs-latent main table
latent fuser / adapter training
claim latent communication failed
claim frozen official CoLA supports the new benchmark
```

这不是“保守退化”，而是实验前提没有满足：

```text
如果 Single CoLA Solver 不能稳定过 gate，通信介质比较没有解释力。
如果 Role TextMAS 不能过 gate，LatentMAS vs TextMAS 的 baseline 也不成立。
```

## 2. 外部依据

LatentMAS：

```text
https://arxiv.org/abs/2511.20639

LatentMAS 的前提是 agent 能产生可用 latent thoughts / working memory，
并在多 benchmark 上比较 single、text MAS、latent MAS。它不是在 base
agent 接近 floor 的任务上直接证明 latent channel。
```

Interlat：

```text
https://arxiv.org/abs/2511.09149

Interlat 使用 last hidden states 作为 agent thought，并引入 learned
compression。它支持“latent channel 可能需要接口/压缩/适配”的方向，
但这属于 adaptation claim，不是 frozen no-fuser claim。
```

Thought Communication：

```text
https://arxiv.org/abs/2510.20733

该工作把 communication 视为 latent thought sharing/recovery 问题，强调
共享/私有 latent thoughts 的可识别结构。这支持我们不要把 answer label
当作通信对象，而应评估可用工作记忆/状态。
```

CoLA DLM：

```text
https://arxiv.org/abs/2605.06548
https://huggingface.co/ByteDance-Seed/Cola-DLM

CoLA 是 Text VAE + block-causal DiT 的 continuous latent diffusion LM。
HuggingFace model card 明确指出 checkpoint 不是 instruction-tuned/RLHF，
prompt 格式敏感，且开源 reference benchmark 分数较低。
```

MAS benchmark literature：

```text
Silo-Bench:
  https://papers.cool/arxiv/2603.01045
  强调 distributed information synthesis，指出 agent 常能交换信息但无法整合。

CRAFT:
  https://huggingface.co/papers/2603.25268
  严格 partial information 下评估 pragmatic communication，说明强 reasoning
  不等于强 coordination。

CoSMAC:
  https://openreview.net/forum?id=yGzAhl1o4i
  在需要通信/协调的场景中评估 LLM agents，而不是普通单问答后硬拆 agent。
```

对 DRLA 的含义：

```text
1. benchmark 应该天然需要 agent communication / distributed state。
2. base solver capability 与 role TextMAS capability 必须先成立。
3. latent interface 不可在 base floor 上训练/比较，否则无法解释。
4. 若要继续使用 CoLA，必须接受 substrate/interface adaptation 的 claim 边界。
```

## 3. 分支选项

### Branch A: CoLA Substrate Adaptation

适用条件：

```text
我们坚持以 CoLA 为最终 substrate。
目标 benchmark 包含 ARC/GPQA/MedQA/GSM8K/EvalPlus 或更强 MAS tasks。
接受“adapted CoLA latent communication”作为下一阶段 claim，而不是
frozen official CoLA no-fuser claim。
```

核心目标：

```text
让 CoLA 至少通过:
  Single CoLA Solver gate
  Role TextMAS gate
然后再比较 TextMAS vs LatentMAS。
```

可能技术路线：

```text
task-format adapter / LoRA:
  解决 prompt/task format 与 CoLA open checkpoint 能力不匹配。

role-protocol adapter:
  学会 Planner/Critic/Refiner/Solver 的 compact state interface。

latent receiver / working-memory adapter:
  在 no-fuser controls 之后，学习 A_latent + B_context -> B_usable_state。

teacher distillation:
  使用 capable TextMAS 或 official trace 作为 teacher。
```

优点：

```text
最贴近最终 CoLA latent communication 目标。
可以正面解决 base capability floor。
后续 latent channel 的可解释性更强。
```

风险：

```text
训练成本高。
claim 边界改变：不再是 frozen official CoLA。
需要严格 train/valid/test、SwanLab cloud、best checkpoint、valid<=100 step。
```

最低执行纪律：

```text
CUDA only
SwanLab cloud
metrics.jsonl
valid_interval <= 10 step
best_checkpoint.pt + last_checkpoint.pt
locked train/valid/test split
no held-out prompt repair
no reporting from last checkpoint only
```

### Branch C: External Capable TextMAS First

适用条件：

```text
我们先验证 benchmark/protocol 是否自然、可评价、确实需要 agent 分工。
不急于把 CoLA latent 接上。
```

核心目标：

```text
使用 capable text agents 建立:
  Single capable solver baseline
  Role TextMAS baseline
  task difficulty / coordination requirement / scorer stability
然后再回到 CoLA latent adapter 或 translator。
```

优点：

```text
最快验证 MAS benchmark 是否合理。
避免把 CoLA base floor 错当 benchmark/protocol 失败。
可以借鉴 Silo-Bench/CRAFT 的 partial-information 设计。
```

风险：

```text
它不证明 CoLA latent communication。
后续仍需要 CoLA latent translator/shared codec/adapter。
```

适合任务族：

```text
partial-information QA / evidence-split reasoning
planner-coder-tester-reviewer code workflows
distributed state synthesis tasks
role-specific expert aggregation
```

### Branch B Family 2: 新的 frozen-CoLA-compatible task design

适用条件：

```text
我们仍坚持 frozen official CoLA，不想先训练/adapt。
但 Family 1 official8-compatible role candidates 已停止。
```

必须满足：

```text
native Single CoLA Solver 先 robust pass calibration
Role TextMAS 再 pass calibration
task 本身必须有 communication value，不只是普通 QA 改 prompt
```

可尝试方向：

```text
very short answer-text matching tasks
CoLA reference-output reconstruction diagnostics
controlled but sufficiently large synthetic communication tasks
```

注意：

```text
这条路科学风险最大。因为 CoLA open checkpoint 参考分数低，过 gate 的任务
可能太简单，容易变成 channel diagnostic 而不是 true MAS evidence。
```

## 4. 推荐排序

在不替用户擅自决定的前提下，当前证据支持以下排序：

```text
1. Branch C:
   先验证 true MAS benchmark/protocol。

2. Branch A:
   如果用户坚持 CoLA latent substrate 是下一阶段核心，则进入 adaptation。

3. Branch B Family 2:
   只作为补充诊断，不建议作为主线。
```

理由：

```text
Branch B Family 1 已证明 frozen official CoLA 不能支撑当前 gate。
继续 prompt-only repair 很可能是在 base floor 上局部调参。

Branch C 能先把“我们到底该评估什么 agent-to-agent communication”说清楚。
这符合 Silo-Bench/CRAFT/CoSMAC 对天然通信任务的要求。

Branch A 是最终回到 CoLA latent 的必要路线，但它需要接受训练成本和 claim
边界改变。
```

## 5. 若选择 Branch C

第一阶段只做 benchmark/protocol validation，不碰 CoLA latent：

```text
C1. 选择 2-3 个 naturally decomposable tasks。
C2. 建 manifest、scorer、leakage audit。
C3. 跑 capable Single Solver 与 capable Role TextMAS。
C4. 只保留 Role TextMAS 相对 Single 有可解释增益或至少不坍塌的任务。
C5. 输出 MAS benchmark locked protocol。
```

验收：

```text
Single baseline above floor
Role TextMAS above floor and parse/scorer stable
communication ablation causes measurable drop
no gold/scorer leakage
calibration/held-out locked
```

之后才接：

```text
CoLA latent translator / adapter / receiver
TextMAS teacher -> CoLA latent state distillation
LatentMAS vs TextMAS under same benchmark
```

## 6. 若选择 Branch A

第一阶段是 substrate-capability adaptation，不是 latent-vs-text 主表：

```text
A1. 选择目标 benchmark family。
A2. 构建 train/valid/test split。
A3. 训练 minimal task-format adapter/LoRA，使 Single CoLA Solver 过 valid gate。
A4. 训练/调 Role TextMAS interface，使 Role TextMAS 过 valid gate。
A5. locked test gate。
A6. 然后才进入 no-fuser latent handoff 与 fuser trigger。
```

训练必须：

```text
SwanLab cloud
GPU/CUDA
metrics.jsonl
valid <= 100 step
best_checkpoint.pt
last_checkpoint.pt
```

## 7. 必须由用户确认的问题

继续前需要选择：

```text
Option 1:
  Branch C first.
  先建立强 MAS benchmark/protocol，再回到 CoLA latent。

Option 2:
  Branch A first.
  直接进入 CoLA substrate adaptation，接受 adapted-CoLA claim。

Option 3:
  Branch B Family 2.
  继续寻找 frozen-CoLA-compatible 诊断任务，但不作为主线 claim。
```

我的建议：

```text
优先 Branch C。
原因是当前最大的未知不是 latent packet 构造，而是 benchmark/protocol 是否
真的表达 agent-to-agent communication。Branch C 可以先把这个科学对象固定，
再决定 CoLA 需要怎样 adaptation。
```

但这一步涉及研究路线取舍，不能由 agent 自动替用户决定。
