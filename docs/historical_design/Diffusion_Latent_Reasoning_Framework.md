# Diffusion Latent Reasoning Framework Digest

> 状态：长期背景设计 digest。瘦身前完整框架设想见 `/data1/luyifei/drla/docs/full_history/2026-05-29_pre_slim/Diffusion_Latent_Reasoning_Framework.md`。当前 P2 实施方案不以本文为准。

## 1. 背景问题

早期 DRLA 设想要回答：

```text
latent-space reasoning 是否能成为 reasoning、planning、memory 和 agent communication 的 substrate？
```

这个方向仍有概念价值，但当前工程主线已经收敛到：

```text
official Cola VAE/DiT substrate
-> P0 decoder-probed readiness
-> P1 decoder-as-teacher latent halt student
-> P2 same-substrate agent-agent latent communication
```

本文不再维护完整训练方案或 broad multi-agent claim。

## 2. 三种 latent 必须区分

### text latent

```text
文本 token / hidden state / decoded CoT 的连续表示。
```

风险：容易只是压缩 text CoT，不一定是真正的 latent reasoning substrate。

### embedding latent

```text
LLM embedding / prefix embedding / adapter hidden state。
```

风险：依赖具体模型接口，跨模型不天然对齐。

### reasoning latent

```text
模型内部用于推进推理状态的 latent trajectory。
```

P2 当前使用的是 Cola VAE/DLM latent blocks，属于 same-substrate reasoning/communication latent 的可控子问题。

## 3. 对当前 P2 有用的边界

### 3.1 不能直接说 latent 是通用语言

Text 是公共接口；latent 不是。P2 只能先 claim：

```text
same-substrate Cola A -> Cola B
```

异构 latent communication 需要：

```text
adapter
translator
shared codec
KV/cache alignment
```

### 3.2 latent reasoning 不等于压缩 CoT

P2 不是把 A 的 text answer 压缩成 latent 再发给 B，而是在同一 Cola substrate 下直接传递 A 的 latent packet。

### 3.3 halt/readiness 是 communication readiness 的前置诊断

P1 的 latent halt student 说明：

```text
latent/process trajectory 中存在 answer-readiness signal。
```

但 P2 仍需单独证明：

```text
Agent B can read/use latent packet.
latent packet has downstream utility.
latent handoff can compete with text handoff.
```

## 4. 早期框架的保留价值

仍可保留的概念：

```text
block-wise latent trajectory
answer-readiness / communication-readiness
decoder-as-teacher, latent-student
budget-aware halt policy
matched-vs-corrupted latent controls
cost-quality frontier
```

应降级的旧设想：

```text
from-scratch Cola-like prior
GSM8K-only MVP as main evidence
direct broad claim of multi-agent latent communication
unverified heterogeneous latent transfer
fixed full DRLA architecture as implementation target
```

## 5. 当前 canonical 文档

```text
P0 adaptive halt / riskcap04:
  /data1/luyifei/drla/docs/p0_reports/cola_adaptive_halt_paper_report_zh.md

P1 latent halt student:
  /data1/luyifei/drla/docs/p1_archive/P1_Model_Comparison_Report_2026-05-27.md
  /data1/luyifei/drla/docs/p1_archive/P1_Final_Archive_and_P2_Plan_2026-05-27.md

P2 latent communication:
  /data1/luyifei/drla/docs/current/P2_Latent_MAS_Communication_Implementation_Plan_2026-05-29.md
```

## 6. If This Document Is Reused

Use it only for:

```text
concept vocabulary
latent type distinctions
historical motivation
paper background
```

Do not use it as:

```text
current implementation roadmap
current P2 experimental protocol
evidence that heterogeneous latent communication works
```
