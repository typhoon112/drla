# P1 Progress and Literature Synthesis Digest, 2026-05-27

> 状态：P1 实验笔记 digest。瘦身前完整记录见 `/data1/luyifei/drla/docs/full_history/2026-05-29_pre_slim/P1_Progress_and_Literature_Synthesis_2026-05-27.md`。P1 主结果以 `P1_Final_Archive...` 和 `P1_Model_Comparison...` 为准。

## 1. 当时主线判断

2026-05-27 的核心判断：

```text
P0 decoder-probed readiness 不能作为最终在线策略。
P1 decoder-as-teacher latent student 是正确桥梁。
P1 不应继续堆窄二分类辅助头。
P1 应冻结当前最好 route，做 locked evaluation，然后进入 P2 packet communication。
```

本轮决策明确不参考 `DRLA_Multiscale_Block_Halt_Design.md`，因为它是不成熟 try。

## 2. 文献结论

Relevant anchors：

```text
Cola DLM:
  使用 official Cola VAE/DiT 作为 substrate，保留 block-causal latent rollout。

COCONUT:
  continuous thought 可行，但不要把本项目退化成 text-CoT compression。

CODI:
  answer-ready distillation 思路有用；decoder/text 可以作为 teacher。

CoLaR:
  dynamic budget / RL-style policy 有启发，但不是 P1 主杠杆。

Latent-space survey:
  latent space 可用于 reasoning/planning/memory/communication，但 evaluability 和 controllability 是核心风险。

Dynamic early exit:
  必须区分 average accuracy 和 sample-level losses；不能用 gain/loss cancellation 报安全。
```

对本项目的结论：

```text
训练阶段可以使用 decoder/scorer 作为 teacher 和 audit。
最终 online agent communication 不能依赖 decoder text。
P2 必须验证 Agent B 是否真正使用 latent payload。
```

## 3. P1 关键证据

P1 逐步从 baseline 走到最优 route：

```text
d64_pma4 baseline:
  有 transfer 信号，但 HellaSwag/SQuAD risk 较高。

answer_identity_action + completion_risk:
  低成本有效，但 loss 偏多。

trajectory_token:
  显式 trajectory/delta readout 是正向信号，但会带来 mismatch。

answer_identity_stability:
  明确正向，成为 P1 student-only 最优点。

empty_answer_risk:
  负结果，出现 proxy mismatch / negative transfer。
```

当前 P1 headline：

```text
trajectory_token
+ answer_identity_action
+ completion_risk
+ answer_identity_stability
```

主结果见：

```text
/data1/luyifei/drla/docs/p1_archive/P1_Model_Comparison_Report_2026-05-27.md
/data1/luyifei/drla/docs/p1_archive/P1_Final_Archive_and_P2_Plan_2026-05-27.md
```

## 4. 停止继续局部调参的理由

以下方向没有成为主线：

```text
all_tokens / pma1 / mean_max
blind width sweeps
stabilityw2
no_block_budget
simple film
last_process_query
isolated rare-event heads
decomposed_expected_utility
post-hoc scalar utility-weight calibration
source-task-robust scalar calibration
```

共同问题：

```text
局部更便宜但不安全；
valid/test loss-gain cancellation；
teacher proxy mismatch；
无法通过 strict Wilson-style risk-control；
不能稳定减少 answer identity mismatch。
```

因此下一步不是继续 P1 调参，而是：

```text
freeze P1
run locked evaluation
emit readiness_state
build sanitized latent communication packets
enter P2 Agent A -> Agent B latent message validation
```

## 5. P2 过渡结论

P1 locked result 已完成：

```text
summary:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/
  official8_full_b64_bs12_p1_locked_seed66_67_68_split20260601_riskcert_20260527/summary.json
```

P2 packet v1 已完成：

```text
output:
  /data1/luyifei/drla/outputs/cola_agent_latent_comm/
  p2_agent_latent_comm_v1_locked_seed66_67_68_split20260601_20260527

packets:
  14940

forbidden decoder/eval fields:
  0
```

这支持进入 P2，但只证明 packet substrate 可构造，不证明：

```text
Agent B can read it
latent handoff improves downstream utility
latent beats text-channel handoff
heterogeneous latent communication is solved
```

当前 P2 canonical：`/data1/luyifei/drla/docs/current/P2_Latent_MAS_Communication_Implementation_Plan_2026-05-29.md`
