# P2 Agent-Agent Latent Communication 实施文档

更新日期：2026-05-29

## 0. 执行摘要

P2 应该被定义为一个通信通道替换问题，而不是一个多智能体架构设计问题。

最小研究对象是：

```text
Agent A -> message channel -> Agent B
```

当前 LLM-based multi-agent systems 里，agent 之间的 message channel 基本都是 text token：

```text
Agent A internal state -> decode to text tokens -> Agent B encodes text -> Agent B responds
```

P2 要做的是把中间的 text message 替换成 Cola latent packet：

```text
Agent A internal state -> Cola latent packet -> Agent B consumes latent -> Agent B responds
```

Agent B 收到 message 之后做什么，不是 communication 本身的定义。B 可以验证、总结、继续推理、调用工具、执行动作，或者聚合多个 agent 的输出。这些都是 receiver-side behavior。P2 的核心问题更窄，也更清晰：

```text
agent-agent communication medium 能不能从 text tokens 换成 latent packets？
```

P2 有两个中心假设：

```text
H1. 可读性 / 有效性：
    Agent B 能够读取并使用 Agent A 发来的 latent packet。

H2. 通信优势：
    相比 text-token handoff，latent packet 更高效、更少损失，或者带来更好的 downstream utility。
```

Sequential 和 hierarchical multi-agent setting 仍然有价值，但它们只是围绕同一个 channel-substitution 问题的实验外壳。single-handoff receiver 是最干净的诊断实验；sequential 和 hierarchical chain 用来测试这个 latent channel 在更大通信模式下是否仍然有效。

## 1. 研究定位

### 1.1 这里的 MAS 指什么

本文档里的 MAS 不指完整的 AutoGen、MetaGPT、ChatDev、AgentVerse 等系统架构设计。这里的 MAS 只指最小 multi-agent communication setting：

```text
两个或更多 agent
一个 agent 发出 message
另一个 agent 接收 message
接收方基于这个 message 决定后续行为
```

也就是说，P2 的研究对象不是 orchestration framework，而是其中的 message channel：

```text
text-token channel vs latent-packet channel
```

现有 MAS 工作的意义在于：它们告诉我们 agent-agent message 在真实系统里通常出现在哪里。我们不需要复刻完整 MAS 框架，只需要抓住它们的通信边界。

### 1.2 核心 claim

P2 研究 same-substrate latent communication。一个 Cola-based sender 把原本会发给另一个 agent 的 text-token message 替换为 distribution-complete Cola latent packet；一个 Cola-based receiver 直接消费这个 latent packet，并产生自己的下游响应。

目标是验证 receiver-native latent handoff 是否能够：

- 保持语义分布一致性；
- 携带可测量的通信效用；
- 降低 text-mediated handoff 的成本；
- 避免 `decode-to-text -> re-encode-from-text` 这一步中的信息损失。

这个问题可以在 single-handoff、sequential-chain、hierarchical-aggregation 等设置里测试，但这些设置不是核心 claim。核心 claim 是：

```text
text message -> latent packet
```

### 1.3 为什么 text 容易，latent 困难

Text 是 LLM agents 的公共接口。即使 Agent A 和 Agent B 是不同模型，通信契约也很清楚：

```text
Agent A 把内部状态 decode 成 text tokens。
Agent B 把 text tokens tokenize/encode 成自己的内部状态。
```

这条路径低效且有损，但它跨异构模型非常稳健。

Latent packet 不同。latent 不是天然公共接口：

```text
Agent A latent space 不一定等于 Agent B latent space。
```

latent 是否可读，取决于：

- 模型架构；
- tokenizer 和 prompt template；
- VAE/DLM latent scaling；
- layer/block position；
- context/prefix state；
- hidden dimensionality；
- 训练分布；
- receiver-side consumption mechanism。

因此 P2 必须区分三层 claim：

| 层级 | 设置 | P2 中的地位 |
|---|---|---|
| Level 1 | same-substrate Cola A -> Cola B | P2 主范围 |
| Level 2 | same-family / near-homogeneous agents | 可能扩展，需要 calibration 或 adapter |
| Level 3 | heterogeneous agents，例如 Qwen -> Llama 或 VLM -> LLM | 后续工作，需要 translator/adapter/shared codec/KV alignment |

除非我们真正实现并评估 adapter，否则 P2 的主 claim 应该限定在 Level 1。

### 1.4 P2 不是什么

P2 不是：

- 一个新的通用 multi-agent framework；
- 声称所有 agent communication 都应该 latent 化；
- 替代 human-readable outputs、tool inputs、code、PRD、test logs 或 retrieval documents；
- 通过 tuning base Cola model 来提升 official Cola benchmark accuracy。

P2 是：

- 一个 communication-substrate replacement study；
- 以 agent-agent message passing 为研究对象；
- 初期限定在 same-substrate Cola agents；
- 必须用 matched-vs-corrupted latent controls 评估。

### 1.5 和现有 MAS 工作的关系

现有 LLM MAS 系统提供的是 text-token handoff 的实验外壳：

| 实验外壳 | 代表系统 | 原本的 text-token message | P2 的 channel substitution |
|---|---|---|---|
| Single handoff | CAMEL, AutoGen | instruction, solution, feedback | 用 A latent packet 替代 A text response |
| Sequential chain | ChatDev, MetaGPT, LatentMAS | 上一个 agent 输出成为下一个 agent 输入 | A-to-B text handoff 替换为 latent handoff |
| Shared message pool | MetaGPT | structured published messages | typed latent message entries |
| Planner/evaluator loop | AgentVerse, AutoAgents | plan, critique, evaluation | latent evidence packet 作为 message payload |
| Hierarchical aggregation | LatentMAS-style hierarchy | 多个 agent 输出给 aggregator | 多个 latent packets 给 aggregator |

最相关的比较对象是 LatentMAS。LatentMAS 在 sequential 和 hierarchical collaboration 中传递 hidden-state 或 KV-cache working memory，而不是 text。P2 可以借用类似实验外壳，但概念对象仍然是 channel replacement。

两者区别应写清楚：

```text
LatentMAS:
  在 sequential/hierarchical systems 中进行 hidden-state/KV-cache latent collaboration。

P2:
  把 Cola VAE/DLM latent packets 作为 same-substrate agent-agent messages，
  并显式审计 distribution-readability 和 matched-vs-corrupted utility。
```

## 2. 当前起点

### 2.1 P1 locked result

P1 已经在 official Cola 8-task prepared split 上得到 locked latent halt student result：

```text
aggregate summary:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/
  official8_full_b64_bs12_p1_locked_seed66_67_68_split20260601_riskcert_20260527/summary.json

seeds:
  66, 67, 68

split seed:
  20260601

repeated target-test decisions:
  14940

P1 selected accuracy:
  20.930%

fixed-final accuracy:
  20.950%

prediction-stability accuracy:
  20.957%

average selected blocks:
  1.834 / 4

losses vs final:
  3

losses vs prediction stability:
  4

calibration joint risk satisfied:
  21 / 24 folds
```

解释：P1 提供了一个 frozen latent/process online-input readiness state。它还不是 Agent B，但它提供了有用的 communication depth selector 和 risk certificate。

### 2.2 当前 P2 packet v1

当前 protocol-level packet substrate：

```text
script:
  /data1/luyifei/drla/drla/scripts/build_cola_agent_latent_comm_packets.py

output:
  /data1/luyifei/drla/outputs/cola_agent_latent_comm/
  p2_agent_latent_comm_v1_locked_seed66_67_68_split20260601_20260527

packets:
  14940

latent block refs:
  27399

unique latent files checked:
  8850

missing latent files:
  0

forbidden decoder/eval fields:
  0
```

v1 online packet 允许字段：

```text
latent_memory.blocks[*].latent_ref
latent_memory.blocks[*].process_features
readiness_state.scores
readiness_state.thresholds
readiness_state.margins
risk_certificate
```

v1 online packet 禁止字段：

```text
decoded text
token ids
gold answer
official score
scored prediction
selected/final/prediction-stability correctness
prediction-stability prediction
prediction_stability_block
```

v1 证明了 sanitized latent packet construction 是可行的。但它还没有证明：

- packet 是 distribution-complete 的；
- Agent B 能把 packet 当成 native Cola context 消费；
- latent payload 对 Agent B 有 causal effect；
- latent packet 在效率、保真度或 task utility 上优于 text-token handoff；
- 同一个 channel 在 sequential 或 hierarchical experimental envelopes 中仍然有效。

## 3. P2 设计原则

### 3.1 保留显式 envelope，只替换 cognitive payload

真实的 agent-agent message 应该保持结构化：

```text
message = explicit envelope + latent cognitive payload
```

Envelope 字段保持可读、可审计：

```text
sender
receiver
task_id
role
phase
handoff_type
block_idx
model_id
config_digest
prompt_hash
risk_certificate
payload_type
```

Latent payload 承载内部 cognitive state：

```text
z_pre or deterministic prefix contract
z_1 ... z_t
process features
optional receiver/fuser features
```

这样可以避免过度 claim。Tool calls、人类可读输出、代码、PRD、检索文档仍然可以是显式 text 或 structured artifacts。P2 只替换 agent-to-agent cognitive message payload。

不要用 receiver 收到 message 之后做什么来定义 communication。receiver 可以继续推理、总结、验证、执行或调用工具。P2 只问 message medium 能不能从 text tokens 换成 latent packets。

### 3.2 Same-substrate first

P2 从 homogeneous Cola agents 开始：

```text
Agent A: Cola VAE/DLM sender
Agent B: Cola VAE/DLM receiver
```

这可以先避开异构模型 latent alignment 问题。但 same-substrate 不等于自动同分布。分布有效性至少需要：

- same model and tokenizer；
- same VAE/DLM config；
- same latent scaling and block geometry；
- known block position and prefix contract；
- correct prompt or prefix latent state；
- matched role and phase metadata。

异构问题必须作为核心 limitation 明确写出，而不是藏起来。Text 广泛可读，是因为每个 LLM 都有 text encoder/tokenizer interface。Latent packet 只有在明确 latent-interface contract 下才可读。Heterogeneous latent communication 需要额外 adapter、translator、shared codec 或 KV alignment layer，不是 P2 主 claim。

### 3.3 先做 one-shot handoff，不急着做 streaming

P2 初始通信事件应该是：

```text
Agent A reasons to block t
Agent A emits one latent packet
Agent B consumes the packet
Agent B continues, aggregates, or verifies
```

这不是 block-by-block streaming。后续可以加入 request-more extension：

```text
B consumes z_1...z_t
B requests final/fallback if uncertain
```

但 P2.2 不应该首先被定义为“Agent A 每推理一个 block 就流式输入 Agent B”。核心是 B 能否稳定接收并使用 A 的 latent packet。

### 3.4 必须有 corrupted controls

任何“B 理解了 latent communication”的 claim 都必须比较：

```text
matched latent
metadata-only
shuffled latent
cross-task latent
wrong-block latent
Gaussian-noised latent
rotated latent
text handoff baseline
```

如果 matched latent 不能优于 corrupted latent，那么 receiver 没有真正使用 latent payload。

### 3.5 证据阶梯与允许 claim

P2 应该沿着证据阶梯推进。每一级回答不同问题，支持不同强度的 claim。

| 层级 | 问题 | 必需证据 | 允许 claim |
|---|---|---|---|
| E0 packet validity | packet 是否干净且可加载 | refs 可加载，forbidden fields 不存在，schema complete | latent packets 可以安全构造 |
| E1 distribution compatibility | A 的 packet 对 B 是否足够 native | matched packet stats 与 native Cola traces 对齐 | same-substrate latent handoff 是 distribution-specified 的 |
| E2 readability | B 是否使用了 payload | matched latent 优于 metadata-only 和 corrupted controls | B 能读/用 A 的 latent packet |
| E3 task utility | B 的 downstream response 是否改善 | matched latent 优于 no-message/corrupted-message receiver | latent handoff 携带有用 task information |
| E4 communication advantage | 是否优于 text handoff | 和 text-channel baseline 做 cost-quality comparison | latent communication 更高效、更少损失或 Pareto-competitive |
| E5 envelope generality | 是否能推广到更大通信模式 | sequential/hierarchical envelopes 通过 E2-E4 controls | latent communication 不只适用于 single handoff |

这个阶梯很重要，因为“Agent B 能读 A 的 latent”和“latent beats text communication”不是同一个 claim。实验只达到 E2/E3 时，不能写成 E4。

## 4. Packet v2 规范

### 4.1 目标

把 v1 packets 升级为 distribution-aware、channel-substitution messages。

### 4.2 新 top-level schema

```json
{
  "protocol_version": "cola_agent_latent_comm_v2",
  "created_at": 0,
  "sample_key": "task::sample_id",
  "task": "hellaswag",
  "communication_boundary": {
    "pattern": "single_handoff | sequential_chain | hierarchical_aggregation",
    "handoff_mode": "one_shot",
    "sender_role": "solver",
    "receiver_role": "solver | reviewer | aggregator | verifier",
    "phase": "reasoning | review | aggregation | verification"
  },
  "prefix_contract": {
    "mode": "shared_context_reencode | prefix_latent_ref | kv_cache_ref",
    "input_context_hash": "...",
    "sender_prompt_hash": "...",
    "receiver_prompt_hash": "...",
    "config_digest": "...",
    "model_id": "Cola-DLM",
    "tokenizer_id": "...",
    "vae_id": "...",
    "dit_id": "...",
    "block_size": 16,
    "patch_size": 4,
    "latent_dim": 16,
    "latent_scaling": "official_cola_shift_scale",
    "max_block_budget": 4
  },
  "agent_a": {
    "name": "cola_agent_sender",
    "checkpoint": "...",
    "selected_block": 1,
    "halt_source": "p1_latent_halt_student | fixed_block | final | oracle_diagnostic"
  },
  "latent_memory": {
    "encoding": "cola_latent_block_refs",
    "block_count": 1,
    "blocks": []
  },
  "readiness_state": {},
  "risk_certificate": {},
  "agent_b_contract": {
    "consume_mode": "replay_latent_blocks | latent_receiver_encoder | latent_fuser",
    "uses_decoder_online": false,
    "allowed_actions": ["continue", "aggregate", "accept", "defer"]
  },
  "audit_refs": {}
}
```

### 4.3 Prefix contract modes

Mode 1, `shared_context_reencode`：

```text
B 能访问原始 task context 和 receiver role prompt。
B 确定性地重新 encode 自己的 prefix。
Packet 携带 hashes 和 config metadata，不携带 prefix latent。
```

这是第一阶段实现目标，因为可以复用已有 traces。

Mode 2, `prefix_latent_ref`：

```text
Packet 携带或引用 z_pre。
B 不需要为了 shared context 再运行 encoder。
```

如果要更强地 claim “中间 agent 可以完全在 latent space 通信，不需要 encoder”，这个模式是必要的。

Mode 3, `kv_cache_ref`：

```text
Packet 携带 DiT/VAE/LLM KV-cache references。
B 直接继承 working memory。
```

这更接近 LatentMAS，建议在 v2a 之后考虑。

### 4.4 实现任务

修改：

```text
/data1/luyifei/drla/drla/scripts/build_cola_agent_latent_comm_packets.py
```

新增 CLI flags：

```bash
--protocol-version cola_agent_latent_comm_v2
--communication-boundary sequential_chain
--sender-role solver
--receiver-role solver
--prefix-contract shared_context_reencode
--consume-mode replay_latent_blocks
```

新输出目录：

```text
/data1/luyifei/drla/outputs/cola_agent_latent_comm/
p2_agent_latent_comm_v2_locked_seed66_67_68_split20260601_YYYYMMDD
```

通过标准：

```text
packet count = 14940
latent refs exist = true
forbidden key hits = 0
100% packets have communication_boundary
100% packets have prefix_contract
100% packets have agent_b_contract
packet_schema.json documents v2 fields
```

## 5. Distribution audit

### 5.1 目标

回答：

```text
Agent A 的 latent packet 是否足够 native、足够 well-specified，能被 Agent B 接收？
```

### 5.2 新脚本

```text
/data1/luyifei/drla/drla/scripts/audit_cola_agent_latent_packet_distribution.py
```

输入：

```text
--packets-jsonl
--output-dir
--num-control-samples
--control-types matched,metadata_only,shuffle,cross_task,wrong_block,noise,rotation
```

输出：

```text
summary.json
distribution_stats.csv
control_stats.csv
ood_detection.csv
packet_examples.jsonl
```

### 5.3 Audit checks

结构检查：

```text
block_count == selected_block
latent_block_shape == [16, 16]
selected_block <= max_block_budget
task/config_digest/seed consistency
all refs loadable
```

分布检查：

```text
latent_norm_mean
latent_norm_std
latent_delta_norm
latent_cosine_to_prev
denoise_drift_norm_mean
block-position conditional statistics
task-conditional statistics
seed-conditional statistics
```

Corruption controls：

```text
metadata_only: keep envelope, remove latent payload
shuffle: same task/seed, different sample latent
cross_task: different task latent
wrong_block: replace z_t with z_k
noise: z + sigma * eps
rotation: random orthogonal transform
```

通过标准：

```text
matched packets pass structural checks
matched latent stats align with native trace stats
corrupted controls are separable from matched latent by distribution diagnostics
no decoder/eval-only fields appear
```

建议报告：

```text
matched-vs-corrupted AUROC
per-task norm/delta/cosine tables
worst-task examples
```

## 6. Layer 1 诊断实验：Single-Handoff Receiver

### 6.1 目的

这个诊断实验回答最核心的问题：receiver 是否能使用 Agent A 的 latent message。它先隔离 channel 的可读性和 causal utility，再加入 sequential 或 hierarchical wrapper。

### 6.2 设置

```text
Agent A: Cola sender
Agent B: latent receiver
Input: one v2 latent packet
Output: accept / defer
```

target construction 只允许 offline：

```text
accept = selected packet does not lose vs final/prediction-stability
defer = selected packet is risky and should use final/fallback
```

Targets 可以 offline 使用 decoder/scorer/gold，但这些字段绝不能进入 online packet 或 receiver input。

### 6.3 新脚本

```text
/data1/luyifei/drla/drla/scripts/train_cola_latent_receiver.py
/data1/luyifei/drla/drla/scripts/eval_cola_latent_receiver.py
/data1/luyifei/drla/drla/scripts/aggregate_cola_latent_receiver.py
```

训练命令形态：

```bash
python -m drla.scripts.train_cola_latent_receiver \
  --packets-jsonl /path/to/agent_latent_comm_packets_train.jsonl \
  --labels-dir /path/to/readiness_frontiers \
  --output-dir /path/to/receiver_train \
  --input-mode latent_process_certificate \
  --objective accept_defer_bce \
  --device auto \
  --swanlab-mode cloud
```

评估命令形态：

```bash
python -m drla.scripts.eval_cola_latent_receiver \
  --packets-jsonl /path/to/agent_latent_comm_packets_test.jsonl \
  --checkpoint /path/to/best_checkpoint.pt \
  --control-types matched,metadata_only,shuffle,cross_task,noise \
  --output-dir /path/to/receiver_eval \
  --swanlab-mode disabled
```

### 6.4 Receiver input modes

必须跑这些 ablations：

```text
envelope_only
process_only
certificate_only
latent_only
latent_process
latent_process_certificate
latent_process_certificate_no_task
```

必须跑这些 corrupted controls：

```text
latent_process_certificate_matched
latent_process_certificate_shuffled
latent_process_certificate_cross_task
latent_process_certificate_noised
metadata_only
```

### 6.5 Receiver architecture

第一个 receiver 保持简单：

```text
latent block encoder:
  per-block PMA or mean pooling over [16, 16] latent block

trajectory encoder:
  small causal Transformer or GRU over selected blocks

process encoder:
  MLP over process features

certificate encoder:
  MLP over readiness scores, margins, thresholds, risk bounds

head:
  accept/defer logit
```

Base Cola model 保持 frozen。这个 receiver 不是 generation adapter，而是 diagnostic readout。

### 6.6 Metrics

```text
accept accuracy
unsafe-packet AUROC
unsafe-packet AUPRC
ECE / calibration bins
losses vs final under selected threshold
mismatches vs final
average accepted blocks
accept rate
matched-vs-corrupted score gap
```

成功标准：

```text
matched latent + certificate > certificate_only
matched latent + certificate > shuffled/cross_task/noised controls
calibrated accept policy does not exceed P1 locked loss risk
receiver remains decoder-free online
```

## 7. Layer 2 实验外壳：Sequential Latent Communication

### 7.1 目的

这是第一个更大的 communication envelope。它问：

```text
Agent B 能否从 Agent A 的 latent packet 继续推理，而不是从 Agent A 的 text output 继续？
```

### 7.2 Text baseline

Sequential text-channel baseline：

```text
Agent 1 receives task + role prompt.
Agent 1 produces text reasoning/answer.
Agent 2 receives task + role prompt + Agent 1 text.
Agent 2 produces final answer.
```

对 official8，text baseline 应保持轻量，因为当前 Cola official benchmark 不是 long-CoT benchmark。baseline 可以包括：

```text
single Cola final
text-channel baseline with A output as decoded selected/final text
text-channel baseline with A output as short text answer only
```

### 7.3 Latent variant

Sequential latent communication：

```text
Agent 1 receives task + sender role prompt.
Agent 1 generates latent blocks z_1...z_t.
Agent 1 emits v2 latent packet.
Agent 2 receives task + receiver role prompt + latent packet.
Agent 2 replays or consumes latent packet.
Agent 2 continues Cola latent generation.
Final decoder produces answer.
```

Communication depth variants：

```text
t = 1
t = 2
t = P1-selected
t = prediction-stability block diagnostic
t = final block
```

Receiver continuation budget：

```text
B continues 1 block
B continues 2 blocks
B continues to final budget
```

### 7.4 B 如何消费 A 的 packet

实现选项 A，replay-based：

```text
B deterministically re-encodes its own receiver prompt and task context.
B loads A's latent blocks from packet refs.
B updates Cola DiT/VAE cache by replaying A blocks as zero-timestep latent memory.
B continues generation with B's own block budget.
```

需要一个 utility function：

```text
prefill_cola_receiver_from_latent_packet(...)
```

可能位置：

```text
/data1/luyifei/drla/drla/scripts/run_cola_sequential_latent_mas.py
```

实现选项 B，receiver encoder：

```text
B uses a small latent fuser to compress A's packet into receiver prefix slots.
B conditions generation on those slots.
```

这个选项更灵活，但会引入 trainable components。建议在 replay-based smoke tests 之后再做。

### 7.5 新脚本

```text
/data1/luyifei/drla/drla/scripts/run_cola_sequential_latent_mas.py
```

命令形态：

```bash
python -m drla.scripts.run_cola_sequential_latent_mas \
  --packets-jsonl /path/to/agent_latent_comm_packets_test.jsonl \
  --input-jsonl-root /data1/luyifei/Cola-DLM/code/generate_task_data \
  --output-dir /path/to/sequential_latent_mas_eval \
  --sender-depth p1_selected \
  --receiver-continue-blocks 2 \
  --consume-mode replay_latent_blocks \
  --control-type matched \
  --swanlab-mode disabled
```

Control runs：

```bash
--control-type metadata_only
--control-type shuffled
--control-type cross_task
--control-type wrong_block
--control-type noise
```

### 7.6 Outputs

```text
generations/{task}.jsonl
trace/{task}_trace.jsonl
metrics.jsonl
summary.json
communication_cost.csv
control_comparison.csv
```

### 7.7 Metrics

Task metrics：

```text
official scorer accuracy
weighted micro accuracy
per-task accuracy
loss/gain vs single Cola final
loss/gain vs text-channel baseline
```

Communication metrics：

```text
Agent A blocks transmitted
Agent B blocks generated
latent elements transmitted
intermediate text tokens transmitted
end-to-end runtime
```

Control metrics：

```text
matched latent vs metadata-only
matched latent vs shuffled latent
matched latent vs cross-task latent
matched latent vs noised latent
```

成功标准：

```text
matched latent beats corrupted controls
matched latent reduces text handoff tokens substantially
matched latent is competitive with the text-channel baseline or improves over single-agent under at least one validated setting
P1-selected depth is close to final-depth latent handoff at lower block cost
```

## 8. Layer 3 扩展：Hierarchical Latent Communication

### 8.1 目的

这个设置测试多个 latent messages 是否可以被 aggregate。

### 8.2 Text baseline

Hierarchical text-channel baseline：

```text
Agent 1 text solution
Agent 2 text solution
Agent 3 text solution
  -> Aggregator reads text outputs
  -> final answer
```

### 8.3 Latent variant

Hierarchical latent communication：

```text
Agent 1 latent packet
Agent 2 latent packet
Agent 3 latent packet
  -> latent aggregator consumes packet set
  -> final decode
```

Candidate sender diversity：

```text
different noise seeds
different role prompts
different selected depths
different task-specialized prompts
```

### 8.4 Fusion modes

Mode 1，concatenation：

```text
Concatenate packet latents in role order.
Aggregator consumes the concatenated latent memory.
```

Mode 2，attention fuser：

```text
Train a small packet-set fuser over packet embeddings.
Base Cola remains frozen.
```

Mode 3，top-k receiver：

```text
Use single-handoff receiver scores to choose top-k packets.
Aggregator consumes selected packet(s).
```

### 8.5 新脚本

```text
/data1/luyifei/drla/drla/scripts/run_cola_hierarchical_latent_mas.py
```

命令形态：

```bash
python -m drla.scripts.run_cola_hierarchical_latent_mas \
  --packet-roots /path/to/agent1_packets /path/to/agent2_packets /path/to/agent3_packets \
  --fusion-mode concat \
  --aggregator-continue-blocks 2 \
  --output-dir /path/to/hierarchical_latent_mas_eval \
  --swanlab-mode disabled
```

### 8.6 成功标准

最低成功：

```text
matched multi-packet latent aggregation beats shuffled/cross-task packet controls
```

更强成功：

```text
hierarchical latent handoff matches or improves hierarchical text handoff with lower intermediate text cost
```

## 9. Evaluation protocol

### 9.1 Datasets

默认使用：

```text
official Cola 8 tasks:
  lambada
  mmlu
  obqa
  hellaswag
  race
  siqa
  squad
  story_cloze
```

沿用当前 full prepared split protocol：

```text
b64
bs12
t16
max blocks = 4
seeds = 66, 67, 68
split seed = 20260601 for locked packet tests
```

后续可选扩展：

```text
GSM8K / math reasoning
code generation
```

在 official8 latent communication controls 没有理解清楚前，不建议迁移到可选任务。

### 9.2 Splits

对于 trained receiver/fuser components：

```text
source tasks train
source valid checkpoint selection
target valid threshold calibration only
target test final report
```

对于 sequential 和 hierarchical generation experiments：

```text
use locked target-test packets
do not tune after inspecting target-test results
run matched and corrupted controls under identical sample sets
```

### 9.3 Main tables

Table 1，packet validity：

```text
protocol version
packet count
latent refs
missing refs
forbidden key hits
prefix contract coverage
```

Table 2，distribution audit：

```text
matched
metadata-only
shuffled
cross-task
wrong-block
noise
rotation
```

Table 3，single-handoff receiver diagnostic：

```text
input mode
unsafe AUROC
unsafe AUPRC
accept rate
loss vs final
avg blocks
matched-vs-corrupted gap
```

Table 4，sequential communication envelope：

```text
method
accuracy
loss/gain vs single
loss/gain vs text-channel baseline
A blocks sent
B blocks generated
text tokens transmitted
runtime
```

Table 5，hierarchical communication envelope：

```text
fusion mode
accuracy
control gap
communication cost
runtime
```

### 9.4 Communication advantage 的操作性定义

P2 应该在同一个 communication boundary 上比较 text 和 latent，不能给其中一方额外信息。

Matched comparison：

```text
Agent A reasons to block t.

Text-channel baseline:
  decode A's state at block t into text
  feed that text to Agent B
  B produces downstream response

Latent-channel variant:
  send A's latent packet at block t
  B consumes the latent packet
  B produces downstream response
```

Efficiency metrics：

```text
intermediate text tokens transmitted
latent elements or compressed bytes transmitted
decode/re-encode calls avoided
wall-clock runtime
Agent B continuation blocks
```

Fidelity metrics：

```text
downstream accuracy or task score
agreement with Agent A final-depth answer
agreement with single-agent final answer
loss/gain relative to full text handoff
loss/gain relative to final-depth latent handoff
matched-vs-corrupted downstream gap
```

最强的“更无损”证据不是 latent token 少，而是在同样 sender depth 和 receiver budget 下，latent handoff 比 `decode-to-text -> re-encode` handoff 更好地保留下游 task utility，或者以更低 communication cost 达到同等 utility。

## 10. Implementation roadmap

### Milestone P2-A: Packet v2

任务：

- 给 `build_cola_agent_latent_comm_packets.py` 增加 v2 schema fields；
- 通过 `--protocol-version` 保留 v1 behavior；
- 写出 `packet_schema.json`；
- 重新构建 locked packets。

Artifacts：

```text
outputs/cola_agent_latent_comm/p2_agent_latent_comm_v2_...
```

退出标准：

```text
14940 packets
0 missing latent refs
0 forbidden key hits
v2 schema complete
```

### Milestone P2-B: Distribution audit

任务：

- 实现 `audit_cola_agent_latent_packet_distribution.py`；
- 加载 packet refs 并计算 latent statistics；
- 生成 corrupted controls；
- 报告 matched-vs-control gaps。

退出标准：

```text
distribution summary produced
per-task stats produced
controls generated deterministically
matched/corrupted separability reported
```

### Milestone P2-C: Single-handoff receiver diagnostic

任务：

- 实现 receiver dataset builder；
- 在 input-mode ablations 下训练 receiver；
- 评估 matched 和 corrupted controls；
- 聚合 LOTO/cross-seed results。

退出标准：

```text
matched latent improves over certificate-only
matched latent improves over shuffled/cross-task controls
calibrated loss risk is comparable to P1 locked result
```

### Milestone P2-D: Sequential latent communication

任务：

- 实现 minimal sequential runner；
- 实现 replay-based packet consumption；
- 在 small subset 上跑 smoke；
- 在 official8 locked test packets 上跑 matched/corrupted controls；
- 跑 matched-depth text-channel vs latent-channel comparisons；
- 和 single Cola、text handoff baselines 比较。

退出标准：

```text
matched latent sequential handoff beats corrupted latent controls
communication cost is lower than text handoff
fidelity metrics are reported at the same sender depth and receiver budget
P1-selected depth has useful cost-quality trade-off
```

### Milestone P2-E: Hierarchical latent communication

任务：

- 创建 multi-packet sender variants；
- 先实现 concat fusion；
- 只有 concat 有意义时再加 attention fuser；
- 比较 matched vs corrupted multi-packet controls。

退出标准：

```text
matched multi-packet aggregation beats corrupted controls
hierarchical text comparison is reported
```

### Milestone P2-F: Writeup and claim audit

任务：

- 准备 method diagrams；
- 准备 tables and ablations；
- 写 related work，并定位到 existing MAS 和 latent-communication work；
- 把每个结果映射到 Section 3.5 的 evidence ladder；
- 分离 same-substrate claims 和 heterogeneous-agent future work；
- 审计所有 claim 是否被证据支持。

退出标准：

```text
No claim that all agent communication is replaced.
No claim that base Cola benchmark accuracy improved unless directly shown.
No claim that hetero-agent latent communication is solved.
All online inputs remain decoder-free where claimed.
```

## 11. Engineering notes

### 11.1 Device and logging policy

Training scripts：

```text
must use CUDA/GPU
must use SwanLab cloud
must write metrics.jsonl
must save best_checkpoint.pt and last_checkpoint.pt
```

Pure eval/audit scripts：

```text
swanlab_mode=disabled
local summaries only
no optimizer/backward
```

### 11.2 Avoiding leakage

Online packet 和 receiver inputs 禁止包含：

```text
decoded text
decode token ids
latest block token ids
official score
scored prediction
gold answer
selected/final correctness
prediction-stability correctness
prediction-stability prediction
prediction_stability_block
```

Offline labels 可以使用这些字段做 supervision 和 analysis，但 train/eval dataset builder 必须显式分离：

```text
online_inputs
offline_targets
audit_only_fields
```

### 11.3 Reproducibility

每个新 artifact 都应该写出：

```text
config
git commit if available
input packet root
input trace root
checkpoint paths
seed
split seed
control generation seed
forbidden key audit
```

## 12. 风险与缓解

### Risk 1: Packet under-specified

症状：

```text
B cannot reproduce or consume A's latent context.
```

缓解：

```text
add prefix_contract
add deterministic context hashes
later add prefix_latent_ref or kv_cache_ref
```

### Risk 2: Receiver 使用 metadata，而不是 latent

症状：

```text
matched latent ~= shuffled latent
```

缓解：

```text
mandatory corrupted controls
latent-only and certificate-only ablations
matched-vs-corrupted score gap
```

### Risk 3: Sequential replay 不够 native

症状：

```text
replayed latent blocks harm B generation or behave like noise
```

缓解：

```text
compare replay vs learned fuser
consider saving prefix latents
consider saving KV-cache refs
run small distribution probes before full eval
```

### Risk 4: Official8 太弱，暴露不出 channel benefits

症状：

```text
single-agent baseline already saturates the useful behavior, and the multi-agent wrapper adds little
```

缓解：

```text
still use official8 for continuity and controls
then extend to reasoning-heavy tasks only after substrate is validated
```

### Risk 5: Claim 漂移到 heterogeneous communication

症状：

```text
paper sounds like Qwen-to-Llama or arbitrary-agent latent communication
```

缓解：

```text
state same-substrate limitation clearly
cite hetero alignment as future work
do not claim hetero without adapter experiments
```

## 13. 预期贡献

最低可信贡献：

```text
Sanitized Cola latent packets as agent-agent messages.
Distribution audit for receiver-native latent message validity.
Controlled evidence that Agent B uses matched latent payload.
```

强 P2 贡献：

```text
Sequential communication replaces text handoff with Cola latent handoff,
beats corrupted controls,
and reduces communication cost while staying competitive with the text-channel baseline.
```

非常强的贡献：

```text
Sequential and hierarchical communication envelopes both work,
P1-selected packet depth approaches final-depth utility,
and latent handoff beats text handoff on cost-quality Pareto.
```

## 14. Related work anchors

后续写 paper 时使用这些 verified anchors：

- AutoGen：multi-agent conversation framework，message-passing abstractions，dynamic group chat。https://arxiv.org/abs/2308.08155
- CAMEL：role-playing communicative agents，instruction-solution messages。https://arxiv.org/abs/2303.17760
- ChatDev：chat chain 和 software-development MAS 中的 communicative dehallucination。https://arxiv.org/abs/2307.07924
- MetaGPT：SOP-based structured communication，message pool，subscription。https://arxiv.org/abs/2308.00352
- AgentVerse：expert recruitment，collaborative decision-making，evaluation feedback。https://arxiv.org/abs/2308.10848
- AutoAgents：planner/observer/action observer 和 dynamic agent generation。https://arxiv.org/abs/2309.17288
- Five Ws of Multi-Agent Communication：跨 MARL、emergent language、LLM communication 的 who/what/when/why/how framing。https://arxiv.org/abs/2602.11583
- CommNet：cooperative MARL 中的 continuous communication。https://arxiv.org/abs/1605.07736
- DIAL：differentiable inter-agent communication。https://arxiv.org/abs/1605.06676
- TarMAC：targeted multi-agent communication。https://arxiv.org/abs/1810.11187
- IC3Net：learning when to communicate。https://arxiv.org/abs/1812.09755
- SchedNet：bandwidth constraints 下的 communication scheduling。https://arxiv.org/abs/1902.01554
- LatentMAS：sequential/hierarchical latent collaboration，latent thoughts 和 working-memory transfer。https://arxiv.org/abs/2511.20639
- Interlat：latent-space agent communication，adapter/curriculum evidence。https://arxiv.org/abs/2511.09149
- Cache-to-Cache：通过 cache transfer 进行 direct semantic communication。https://arxiv.org/abs/2510.03215
- K-V Cache Alignment：通过 K/V cache alignment 做 latent-space communication。https://arxiv.org/abs/2601.06123
- Vision Wormhole：heterogeneous latent-space communication 和 alignment warnings。https://arxiv.org/abs/2602.15382

## 15. 立即行动顺序

推荐 coding order：

```text
1. Add v2 packet schema fields.
2. Build p2_agent_latent_comm_v2 locked packets.
3. Implement distribution audit with corrupted controls.
4. Train/evaluate single-handoff receiver ablations.
5. Implement sequential latent communication smoke test.
6. Run matched/corrupted sequential controls.
7. Add text handoff baseline.
8. Decide whether hierarchical communication is ready for full run.
```

Step 6 之后的关键决策：

```text
If matched latent does not beat corrupted latent:
  fix packet/receiver distribution before expanding.

If matched latent beats corrupted but not text handoff:
  claim latent communication utility and cost diagnostics, not superiority.

If matched latent beats corrupted and is competitive with text handoff:
  proceed to the hierarchical communication envelope and paper-style framing.
```
