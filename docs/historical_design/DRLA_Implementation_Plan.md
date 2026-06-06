# DRLA P0/P1 Historical Roadmap Digest

> 状态：历史实施日志 / P0-P1 roadmap digest。瘦身前完整长文档见 `/data1/luyifei/drla/docs/full_history/2026-05-29_pre_slim/DRLA_Implementation_Plan.md`。当前 P2 实施方案以 `/data1/luyifei/drla/docs/current/P2_Latent_MAS_Communication_Implementation_Plan_2026-05-29.md` 为准。

## 1. 路线修订摘要

2026-05-24 起，项目主线从：

```text
自建小 prior / GSM8K MVP / 用 LoRA 提 Cola 精度
```

修订为：

```text
official Cola VAE + official Cola DiT as latent substrate
-> block-wise latent rollout
-> decoder-side probe / answer stability / task scorer / future gain
-> block-level answer-readiness / halt
-> decoder-as-teacher, latent-student halt
-> P2 same-substrate agent-agent latent communication
```

核心目标不是提升 released Cola official benchmark accuracy，而是判断：

```text
当前 accumulated latent memory 是否已经足以生成正确且稳定的答案？
继续生成更多 latent blocks 是否还有实质收益？
这个 readiness 能否最终服务 agent-agent latent communication？
```

## 2. 不变约束

```text
main data:
  official Cola 8 tasks only
  lambada, mmlu, obqa, hellaswag, race, siqa, squad, story_cloze

main protocol:
  full prepared split
  b64 / bs12 / t16
  max blocks = 4
  seeds = 66, 67, 68

training:
  CUDA/GPU required
  SwanLab cloud required
  metrics.jsonl + best_checkpoint.pt + last_checkpoint.pt required

eval / trace / frontier / aggregation:
  swanlab_mode=disabled
  local artifacts only
```

GSM8K 只允许作为 OOD/math diagnostic。DiT LoRA / adapter 只作为后续接口适配、probe 或 policy tuning 工具，不作为主线 accuracy 目标。

## 3. P0 / P1 结论

P0 是 decoder-probed readiness baseline，不是最终 latent-only policy。

P0 使用 decoder output、task-scored prediction、prediction-stability 和 official scorer 构造 teacher / diagnostic signals。它证明 Cola latent rollout 中存在 answer-readiness structure，并提供 P1 teacher / safety-cost upper bound。

```text
P0 canonical:
  /data1/luyifei/drla/docs/p0_reports/cola_adaptive_halt_paper_report_zh.md

P0 key artifact:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/
  official8_full_b64_bs12_cross_task_joint_readiness_riskcap04_cross_seed_20260524/summary.json
```

P1 是 `decoder-as-teacher, LatentHaltStudent-v1`。Training labels 来自 decoder/scorer/stability/future blocks，online inputs 只用 raw latent prefix blocks 与 latent/process/budget features。

```text
best P1 route:
  trajectory_token + answer_identity_action + completion_risk + answer_identity_stability

locked summary:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/
  official8_full_b64_bs12_p1_locked_seed66_67_68_split20260601_riskcert_20260527/summary.json

selected / fixed-final / prediction-stability accuracy:
  20.930% / 20.950% / 20.957%

avg blocks:
  1.834 / 4

losses vs final / prediction-stability:
  3 / 4

risk certificate:
  21 / 24 folds satisfied
```

P1 canonical docs：

```text
/data1/luyifei/drla/docs/p1_archive/P1_Final_Archive_and_P2_Plan_2026-05-27.md
/data1/luyifei/drla/docs/p1_archive/P1_Model_Comparison_Report_2026-05-27.md
```

## 4. 为什么停止 P1 局部调参

已验证的负结果包括：

```text
all_tokens
pma1
mean_max
d32_pma4 / d128_pma4
stabilityw2
no_block_budget
simple film
last_process_query
empty_answer_risk stacking
decomposed_expected_utility
post-hoc scalar utility calibration
source-task-robust scalar calibration
```

共同结论：

```text
继续堆窄二分类辅助头或只调阈值，容易产生 proxy mismatch / negative transfer。
P1 已足以支持进入 P2 packet communication。
下一步应验证 latent message 是否可被 Agent B 读取和使用。
```

## 5. P2 过渡

P1 已输出 readiness_state 和 risk_certificate，P2 packet v1 已完成：

```text
script:
  /data1/luyifei/drla/drla/scripts/build_cola_agent_latent_comm_packets.py

output:
  /data1/luyifei/drla/outputs/cola_agent_latent_comm/
  p2_agent_latent_comm_v1_locked_seed66_67_68_split20260601_20260527

packets:
  14940

missing latent files:
  0

forbidden decoder/eval fields:
  0
```

这证明 protocol 和 sanitized packet substrate 成立，但还没有证明 Agent B readability、downstream utility 或 text-channel superiority。

当前 P2 canonical：

```text
/data1/luyifei/drla/docs/current/P2_Latent_MAS_Communication_Implementation_Plan_2026-05-29.md
```

## 6. Historical Notes

旧 Stage/GSM8K/custom-prior 路线已降级，只用于复现：

```text
/data1/luyifei/drla/archive/2026-05-24-legacy-stagebc-code
/data1/luyifei/drla/archive/2026-05-24-legacy-stagebc-artifacts
```

如果需要完整逐轮实验流水，请读：

```text
/data1/luyifei/drla/docs/full_history/2026-05-29_pre_slim/DRLA_Implementation_Plan.md
```
