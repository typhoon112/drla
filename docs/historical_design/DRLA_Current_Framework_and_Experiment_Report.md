# DRLA 当前框架与实验进展报告 Digest

> 状态：历史快照 digest。本文记录 2026-05-25 时的 P0/P1 框架状态；瘦身前完整报告见 `/data1/luyifei/drla/docs/full_history/2026-05-29_pre_slim/DRLA_Current_Framework_and_Experiment_Report.md`。当前 P2 主线以 2026-05-29 P2 实施文档为准。

更新时间：2026-05-25

## 1. 当时主线

2026-05-25 时，项目已从：

```text
自建小 prior / GSM8K MVP / 用 LoRA 提 Cola 精度
```

转为：

```text
official Cola VAE + official Cola DiT
-> block-wise latent rollout
-> decoder-side probe
-> answer stability / task scorer / future gain
-> block-level answer-readiness / halt
```

核心问题：

```text
累计到当前 block 的 latent memory 是否已经足以生成正确且稳定的答案？
继续生成更多 latent blocks 是否仍有实质收益？
```

## 2. 当时框架

```text
prompt / task input
  -> official Cola VAE encode prefix latent

for block = 1 ... B_max:
  official Cola DiT samples current latent block
  append current block to accumulated latent memory
  decoder probe collects answer/stability/process signals
  readiness / halt model estimates whether to continue

final accumulated blocks
  -> official Cola decoder
  -> answer text
  -> official task scorer
```

当时已经明确：decoder-side signals 是 teacher / diagnostic，不是最终 agent-latent communication 的 online dependency。

## 3. 当时关键结论

```text
1. Cola block-wise latent rollout 中存在 answer-readiness signal。
2. EOS-only 几乎接近 fixed-final，不足以作为 halt condition。
3. prediction_stability 是强 non-gold halt baseline。
4. joint-readiness riskcap04 是当时最好 P0 safety/cost baseline。
5. no-riskcap shape-risk / fragment guard 会出现 loss-gain cancellation。
6. 手写 completion guards 有诊断价值，但不应成为最终能力。
7. 下一步应把 decoder-side stability/completion/risk 蒸馏进 latent-student。
```

## 4. 后续如何演进

这份快照之后，路线已经演进为：

```text
P0 decoder-probed readiness baseline
-> P1 decoder-as-teacher LatentHaltStudent-v1
-> P1 locked evaluation
-> P2 sanitized latent packet substrate
-> P2 same-substrate Agent A -> Agent B latent communication
```

当前 canonical：

```text
P0 report:
  /data1/luyifei/drla/docs/p0_reports/cola_adaptive_halt_paper_report_zh.md

P1 archive:
  /data1/luyifei/drla/docs/p1_archive/P1_Final_Archive_and_P2_Plan_2026-05-27.md
  /data1/luyifei/drla/docs/p1_archive/P1_Model_Comparison_Report_2026-05-27.md

P2 implementation:
  /data1/luyifei/drla/docs/current/P2_Latent_MAS_Communication_Implementation_Plan_2026-05-29.md
```

## 5. 使用边界

Use this document for:

```text
理解 2026-05-25 时的框架状态
追溯 P0 -> P1 过渡动机
写 background 时引用历史判断
```

Do not use it for:

```text
当前 P2 实施路线
当前实验下一步
heterogeneous latent communication claim
```
