# P2 Channel Evaluation Protocol Audit, 2026-05-31

## 1. Executive Verdict

本次审计确认：P2-D 旧的 `latent_matched` 50/task 表存在重大协议问题，不能再作为 Agent-A -> Agent-B communication evidence 引用。

问题不是 latent packet 是否有信息，而是旧 runner 的 scorer-visible 输出边界错误：

```text
旧 legacy all-visible 行为：
  Agent A replay tokens / text message
  -> 进入 final generate
  -> official scorer 直接评分

严格 receiver-only 行为：
  Agent A message 可以作为 Agent B 输入/状态条件
  Agent A message 本身不能进入 final generate
  official scorer 只看 Agent B 在 handoff 之后新生成的 answer
```

因此当前严谨状态是：

```text
保留：
  A latent packet 可被 Cola VAE decoder 解码出 task-relevant 内容。
  matched latent 明显不同于 corrupted latent。

撤销/降级：
  旧 50/task `latent_matched` 表不证明 Agent B 成功消费 latent。
  旧 50/task `latent_matched` 表不证明 latent communication 优于 text communication。

当前唯一通过 communication 边界的结果：
  receiver-only 1/task smoke，仅证明新协议可执行。
  尚无正式 50/task receiver-only Agent-B communication table。
```

## 2. 当前代码中的实际 Agent A -> B 数据流

### 2.1 Message Builder

脚本：

```text
/data1/luyifei/drla/drla/scripts/build_cola_agent_channel_messages.py
```

实际做法：

```text
packet sample_key
-> 找同一个 sample / selected_block 的 native trace
-> 读取 trace_row["decode_text_so_far"] 作为 A_raw_text_message_t
-> 记录 latent packet ref / block count
```

重要约束：

```text
selected_prediction 不作为 text message。
gold/scorer output 不作为在线输入。
text 与 latent 使用同一个 handoff_depth。
```

### 2.2 Agent-B Text Channel

代码位置：

```text
/data1/luyifei/drla/drla/scripts/run_cola_agent_b_channel_eval.py
```

当前 text channel：

```text
base_prompt + A_raw_text_message_t
-> tokenizer
-> Cola VAE encoder
-> B prefix latent/cache
-> B 继续 denoise/generate receiver blocks
-> receiver_only 模式下 scorer 只看 B 新生成 tokens
```

这符合“Agent A 输出作为 Agent B 输入，评估 Agent B 输出”的基本协议，尽管 prompt 形式仍较粗糙，后续应加入明确的 receiver role / message delimiter。

### 2.3 Agent-B Latent Channel

当前 latent matched decode-and-emit 数据流：

```text
A latent z 直接从 packet shard 读取，不由 text encode 得到。

legacy decode-and-emit replay:
  z -> VAE decoder logits/tokens
  z -> DiT KV cache
  decoded replay tokens 进入 context_ids

receiver generation:
  B 继续 denoise/generate remaining blocks
```

关键更正：

```text
latent channel 不是：
  z -> decoder text -> B encoder -> B DiT

旧 legacy all-visible 模式实际是：
  z -> decoder tokens -> final generate -> scorer
  z -> DiT cache
```

所以旧 `latent_matched` work 的主要原因可以是：A latent 被 decoder 还原成答案/答案前缀后，直接被 scorer 看到。这绕过了 Agent B answer-generation 边界。

### 2.4 Receiver-Only 修正

runner 已新增：

```text
--score-output-scope receiver_only
--score-output-scope legacy_all_visible
```

默认值已改为：

```text
receiver_only
```

receiver-only 规则：

```text
A text message 可以进入 B 输入，但不进入 final generate。
A latent replay 可以更新 B 的 VAE/DiT cache，但 replay decoded tokens 不进入 final generate。
scorer 只看 Agent B handoff 之后新生成 tokens。
```

## 3. Artifact Protocol Boundary Audit

审计脚本：

```text
/data1/luyifei/drla/drla/scripts/audit_cola_channel_protocol_boundaries.py
```

审计 artifact：

```text
/data1/luyifei/drla/outputs/cola_agent_channel_eval/
p2d_protocol_boundary_audit_20260531
```

结果：

| artifact | scope | messages | generations | status | allowed claim |
|---|---:|---:|---:|---|---|
| `p2d_agent_b_channel_eval_official8_50per_task_seed20260531_unique_20260531_merged` | `legacy_all_visible_inferred` | 400 | 3200 | fail | decodability/replay-output only |
| `p2d_agent_b_channel_eval_official8_50per_task_cache_only_seed20260531_unique_20260531_merged` | `legacy_all_visible_inferred` | 400 | 3200 | fail | decodability/replay-output only |
| `p2d_agent_b_channel_eval_smoke_receiver_only_1per_task_20260531` | `receiver_only` | 8 | 48 | pass | Agent-B communication smoke only |

注意：旧 artifact 没有 `score_output_scope` 字段，审计脚本按 `legacy_all_visible_inferred` 处理。它们不满足当前 Agent-B communication 边界。

## 4. Receiver-Only Smoke Result

artifact：

```text
/data1/luyifei/drla/outputs/cola_agent_channel_eval/
p2d_agent_b_channel_eval_smoke_receiver_only_1per_task_20260531/channel_eval_aggregate
```

leak audit：

```text
generations = 48
score_output_scope = receiver_only
sum(scorer_visible_text_message_tokens) = 0
sum(scorer_visible_replay_blocks) = 0
```

official scorer smoke：

| channel | accuracy | mean score |
|---|---:|---:|
| `none` | 25.00% | 0.6059 |
| `text` | 12.50% | 0.2996 |
| `latent_matched` | 0.00% | 0.2082 |
| `latent_matched_cache_only` | 0.00% | 0.2082 |
| `latent_matched_dit_only_cache` | 0.00% | 0.2082 |
| `latent_matched_vae_only_cache` | 0.00% | 0.3373 |

解释：

```text
旧 decode-and-emit 优势在 receiver-only smoke 中消失。
这说明旧优势主要来自 replay decoded tokens 直接被 scorer 看到。
当前尚无正式 receiver-only 50/task Agent-B communication 结论。
```

## 5. 与参考论文对照

### Coconut

Coconut 的关键是把 last hidden state 作为 continuous thought，再作为后续 input embedding 回流；并且这个机制通过训练/课程学习获得。它不是把 latent decode 成文本后直接给 scorer，也不是把外来 latent 随便塞进 cache 后期待模型自然理解。

对本项目的约束：

```text
如果要声称 native latent reasoning/communication，必须证明 receiver 在不暴露 replay text 给 scorer 的情况下使用 latent state。
```

参考：`https://arxiv.org/abs/2412.06769`

### CODI

CODI 用 self-distillation 对齐 explicit/implicit reasoning 的 hidden activations，尤其围绕 answer-generating token 的 hidden state。它说明 continuous thought 能 work 的关键是 representation alignment / distillation supervision。

对本项目的约束：

```text
如果 A latent 要成为 B 的输入，B 需要 learned receiver interface / distillation target。
直接 replay 并让 scorer 看 replay text，不等价于 learned latent communication。
```

参考：`https://arxiv.org/abs/2502.21074`

### CoLaR

CoLaR 通过 next compressed embedding prediction / latent head / RL-style dynamic compression 来训练 latent reasoning。它不是后处理拼接，也不是把未训练过的外部 latent 注入后直接做最终 claims。

对本项目的约束：

```text
需要把 latent channel 当成可训练接口或策略，而不是只测 decoder 可读性。
```

参考：`https://arxiv.org/abs/2505.16552`

### Continuous multi-agent communication

CommNet 类工作把 continuous communication 与 agent policy 一起训练，通信向量的意义来自任务损失和反向传播，而不是天然共享。

对本项目的约束：

```text
Agent-A latent -> Agent-B response 的路径必须有 receiver-side objective。
否则只说明 latent 可读/可扰动，不说明 agent communication 成立。
```

参考：`https://proceedings.neurips.cc/paper/6398-learning-multiagent-communication-with-backpropagation`

## 6. 当前可保留与必须撤销的结论

### 可保留

```text
P2-A packet construction / latent refs / no forbidden fields：保留。
P2-B packet distribution and corrupted-control audit：保留。
P2-C compatibility receiver can distinguish matched vs corrupted：保留为 latent separability diagnostic。
旧 latent_matched replay-output result：保留为 Cola decoder decodability diagnostic。
```

### 必须撤销或降级

```text
旧 50/task latent_matched > B_none：撤销为 Agent-B communication claim。
旧 50/task latent_matched vs text：撤销为 text-vs-latent communication comparison。
projection gap audit：降级为 scorer-visible replay leakage diagnostic。
```

### 当前未知

```text
receiver-only 50/task text vs latent communication：未跑正式表。
latent direct DiT/KV consumption 是否可用：未建立。
learned receiver/fuser 是否能把 A latent 转成 B 可用状态：未建立。
```

## 7. 下一步严格实验协议

所有后续 Agent-A -> Agent-B 评估必须先通过 protocol-boundary audit。

LatentMAS 对照后的协议选择：

```text
canonical:
  message_only

diagnostic only:
  shared_context
```

理由：

```text
LatentMAS 的 text MAS 描述是：一个 agent 完成 generation 后，
自然语言输出被直接追加/传给下一 agent；latent MAS 则把前一 agent
的 latent working memory / KV cache 传给下一 agent，最后只有最终
agent decode answer。

因此如果本项目要研究 Agent-A -> Agent-B communication，Agent B 的
主要观测应来自 Agent A output / latent packet，而不是让 B 再独立看到
原始 benchmark prompt。`shared_context` 可以作为控制诊断，但不能替代
纯 handoff 主协议。
```

硬性 gate：

```text
agent_b_input_contract = message_only
score_output_scope = receiver_only
sum(scorer_visible_text_message_tokens) = 0
sum(scorer_visible_replay_blocks) = 0
selected_prediction 不作为 message 输入
gold/scorer output 不作为 online input
same q / same handoff_depth / same B budget / same scorer
```

正式 receiver-only P2-D 表应包含：

```text
B_none(empty input) -> score only B receiver output
B_text(A_raw_text_message_t) -> score only B receiver output
B_latent_matched(A_latent_packet_t) -> score only B receiver output
B_latent_corrupt(corrupted_packet_t) -> score only B receiver output
B_latent_dit_only_cache / vae_only_cache ablations
```

预期解释规则：

```text
matched > corrupt:
  B uses matched latent state more than corrupted state.

matched > none:
  latent message gives marginal value to B.

matched vs text:
  only valid if both are receiver_only and scorer sees neither A text nor A replay.
```

若 receiver-only latent 仍失败，则下一步不应继续重复 replay，而应训练：

```text
learned latent receiver / adapter / fuser
目标：A latent -> B usable receiver state / answer-readiness / response policy
监督：receiver-only downstream score、CODI-style hidden-state distillation、
      replay decoder teacher signal，但 teacher signal 只用于训练，不作为 online scorer-visible output。
```

## 8. Current Status

```text
P2-D communication claim:
  reset to not established.

P2-D protocol:
  repaired in code.
  shared_context + receiver_only smoke is executable.
  message_only + receiver_only smoke is executable and is now canonical.

Next required formal artifact:
  official8 50/task message_only + receiver_only channel-equivalent evaluation.

Scientific posture:
  Do not cite legacy all-visible tables as communication evidence.
  Do not cite shared_context no-message results as pure Agent-A -> B handoff.
  Do not claim latent > text until receiver-only paired comparison passes.
```

Message-only receiver-only smoke, 2026-05-31：

```text
artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_smoke_message_only_receiver_only_1per_task_20260531

protocol:
  agent_b_input_contract = message_only
  score_output_scope = receiver_only

boundary audit:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_protocol_boundary_audit_message_only_smoke_20260531

audit result:
  status = pass
  scorer_visible_text_message_tokens = 0
  scorer_visible_replay_blocks = 0
```

Smoke official scorer：

| channel | accuracy | mean score |
|---|---:|---:|
After the replay-EOS fix, the current canonical smoke is:

```text
artifact:
  p2d_agent_b_channel_eval_smoke_message_only_receiver_only_eosfix_1per_task_20260531

code fix:
  replay EOS/im_end is recorded as replay_stop_token_seen, but in receiver_only
  mode it does not stop Agent B. Only Agent-B-generated stop tokens terminate
  receiver generation.

boundary audit:
  p2d_protocol_boundary_audit_message_only_eosfix_smoke_20260531
  status = pass
```

| channel | accuracy | mean score |
|---|---:|---:|
| `none` | 0.00% | 0.1527 |
| `text` | 0.00% | 0.2022 |
| `latent_matched` | 0.00% | 0.2457 |
| `latent_matched_cache_only` | 0.00% | 0.2478 |
| `latent_matched_dit_only_cache` | 0.00% | 0.2405 |
| `latent_matched_vae_only_cache` | 0.00% | 0.1559 |

Interpretation：

```text
This is only a protocol smoke.  It confirms the LatentMAS-aligned handoff path
is executable and leakage-free.  It is too small to establish channel quality.
```

## 9. Formal Message-Only Receiver-Only Result

```text
artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_official8_50per_task_message_only_receiver_only_eosfix_seed20260531_unique_20260531_merged

aggregate:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_official8_50per_task_message_only_receiver_only_eosfix_seed20260531_unique_20260531_merged/channel_eval_aggregate

boundary audit:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_protocol_boundary_audit_message_only_receiver_only_eosfix_50task_20260531

scope:
  official8, 50 unique sample_key per task
  400 messages, 11 channels, 4400 Agent-B generations
  merge duplicate_keys = 0
  missing_message_rows = 0
  audit status = pass
```

Boundary result：

```text
configured_agent_b_input_contract = message_only
configured_score_output_scope = receiver_only
row_agent_b_input_contracts = message_only
row_score_output_scopes = receiver_only
scorer_visible_text_message_tokens = 0
scorer_visible_replay_blocks = 0
claim_allowed = agent_b_communication
```

Official scorer summary：

| channel | accuracy | mean score |
|---|---:|---:|
| `latent_matched` | 1.75% | 0.1874 |
| `latent_matched_cache_only` | 1.75% | 0.1912 |
| `latent_matched_dit_only_cache` | 1.25% | 0.1873 |
| `latent_matched_vae_only_cache` | 0.00% | 0.1395 |
| `none` | 0.00% | 0.1378 |
| `text` | 1.25% | 0.1795 |
| `latent_cross_task` | 0.25% | 0.1535 |
| `latent_shuffle` | 0.25% | 0.1663 |
| `latent_wrong_block` | 4.00% | 0.1950 |
| `latent_noise` | 0.00% | 0.1390 |
| `latent_rotation` | 0.00% | 0.1297 |

Paired readout：

```text
matched > none:
  score_delta = +0.0496, CI95 [+0.0340, +0.0659]
  accuracy_delta = +1.75 pp, CI95 [+0.50, +3.00]

matched vs text:
  score_delta = +0.0079, CI95 [-0.0082, +0.0259]
  accuracy_delta = +0.50 pp, CI95 [-1.00, +2.00]

matched vs wrong_block:
  score_delta = -0.0076, CI95 [-0.0286, +0.0140]
  accuracy_delta = -2.25 pp, CI95 [-4.50, -0.25]
```

Conclusion：

```text
Valid communication evidence now exists for marginal utility over empty input.
Text superiority is not established in either direction.  The all-corrupt gate
fails because wrong_block is anomalously strong.  This result should be framed
as "receiver-only latent handoff has signal but payload-specific control
robustness is unresolved", not as "latent beats text".
```
