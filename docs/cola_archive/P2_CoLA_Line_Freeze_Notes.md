# P2 CoLA 线冻结说明

更新时间：2026-06-06

## 为什么要归档而不是继续

P2 的原始目标是 agent-to-agent latent communication。P0/P1 提供了很好的 latent readiness substrate，但后续 CoLA MAS 线暴露出两个边界：

1. Official8 本身不是天然 multi-agent benchmark。很多任务被单个 agent 回答完就结束，不要求分布式 evidence / role 分工。
2. Frozen CoLA 直接迁移到 MuSiQue/HotpotQA 等 evidence-split multi-agent 协议时，prompt-only/interface-only 能力不足；即使经过 calibration-only LoRA 适配，held-out gate 仍未达到进入 text-vs-latent 主表的能力门槛。

因此，CoLA P2 当前应作为诊断线冻结，不应继续用它做主 claim。P0/P1 的价值保留：它们证明了 CoLA latent block 中存在可学习 readiness 信号，并提供了 decoder-free early-halt student 与 packet substrate。

## 已完成且保留的 P2 CoLA 诊断

Packet v2 substrate：

```text
/data1/luyifei/drla/outputs/cola_agent_latent_comm/
p2_agent_latent_comm_v2_locked_seed66_67_68_split20260601_20260529
```

关键状态：

```text
packets = 14940
latent refs = 27399
unique latent files = 8850
missing latent files = 0
forbidden decoder/eval fields = 0
```

Distribution audit：

```text
/data1/luyifei/drla/outputs/cola_agent_latent_comm/
p2_agent_latent_comm_v2_distribution_audit_locked_seed66_67_68_split20260601_20260529
```

关键状态：

```text
status = pass
native alignment max_abs_diff = 0.0
min corrupted-control pair-distance AUROC = 1.0
```

Receiver compatibility diagnostic：

```text
/data1/luyifei/drla/outputs/cola_latent_receiver/
p2c_receiver_compat_bestckpt_eval_aggregate_seed20260529_20260529
```

关键状态：

```text
best input mode = latent_process_certificate
test mean-control AUROC = 0.9194
hardest listed control = shuffle AUROC 0.6205
```

解释：latent packet 在结构上可区分、可审计，说明 substrate 不是随机噪声。但这不等于 Agent B 能自然把 Agent A latent 当作可用推理输入。

## 已识别的协议问题

早期 direct-answer / latent_matched 结果存在 scorer-visible shortcut 风险：Agent A 的文本或 replay decoded tokens 被拼进最终 scorer-visible output，导致 Agent A 到 B 的通信被越过。

当前正确边界是：

```text
Agent A output/message -> Agent B input -> evaluate Agent B generated result only
```

因此所有 P2-D channel claim 必须满足：

- `score_output_scope = receiver_only`
- Agent A text-message tokens 不进入最终 scorer-visible answer；
- Agent A latent replay decoded tokens 不进入最终 scorer-visible answer；
- Latent packet 直接作为 latent/channel payload 进入 Agent B 的 receiver/continuation path，而不是先 decode 成答案再拼给 scorer；
- Gold/scorer/correctness 只允许离线评估，不允许成为 Agent A/B online input。

用于审计的脚本：

```text
/data1/luyifei/drla/drla/scripts/audit_cola_channel_protocol_boundaries.py
/data1/luyifei/drla/drla/scripts/audit_cola_channel_projection_gap.py
```

## Phase A CoLA interface adaptation 边界

已尝试的方向包括：

```text
/data1/luyifei/drla/outputs/p2_phase_a_cola_interface_sft/
/data1/luyifei/drla/outputs/p2_phase_a_cola_dit_lora/
/data1/luyifei/drla/outputs/p2_phase_a_cola_latent_candidate_ranker/
```

当前结论：

- official CoLA substrate 本身没有坏，official SQuAD smoke 与 historical official SQuAD baseline 数量级一致。
- MuSiQue role/evidence-split 协议下，prompt-only CoLA 失败主要是 task/interface mismatch。
- calibration-only LoRA 能改善 calibration，但 held-out gate 未达 `single_full_info >= 0.2` 等进入主表的能力门槛。
- Candidate-ranker 诊断显示 token-level interaction 有帮助，但 CoLA-latent shallow/ranker 路线仍明显低于 Qwen semantic selector。

所以不要从当前 CoLA Phase A checkpoint 直接进入 Phase E TextMAS vs LatentMAS 主表。

## 后续如果重启 P2 CoLA

必须先满足以下条件：

1. 使用真正支持 multi-agent 分工的 benchmark/protocol。
2. 有严格的 capability gate：single full-info、text-matched、no-message、shuffled、wrong-evidence controls 都必须通过。
3. Frozen CoLA 或 CoLA adapter 必须先在 held-out protocol 上证明基本 solver/role ability。
4. LatentMAS 对比必须是 receiver-only scoring。
5. 不允许使用 held-out 做 prompt repair、adapter selection 或 threshold tuning。

当前默认执行路线仍以：

```text
/data1/luyifei/drla/docs/current/P2_Locked_Complete_Execution_Scheme_2026-06-01.md
```

为最高优先级。
