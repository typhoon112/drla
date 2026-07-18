# P3 Dream-DLM LatentMAS 实验设计

更新日期：2026-06-06

> 状态：当前 Dream 系列主线实验设计。本文定义科学问题、模型与 benchmark 选择、
> P0/P1 早停判别器如何迁移到 Dream-DLM、以及 latent communication 主实验的
> 公平对照。具体工程执行见
> `/data1/luyifei/drla/docs/current/P3_Dream_DLM_Latent_MAS_Implementation_Plan_2026-06-06.md`。

## 0. 一句话路线

```text
以 Dream-v0-Instruct-7B 为同构 DLM substrate
先在 true MAS benchmark 上证明 TextMAS protocol 成立
再训练 Dream denoising-step readiness / halt 判别器
最后比较 receiver-only TextMAS vs LatentMAS
```

这不是继续 CoLA P2 的权重修补，而是保留 CoLA P0/P1 学到的早停判别器思想，
更换为更可能具备 instruction / QA / role-following 能力的 DLM substrate。

## 1. 为什么从 CoLA 转向 Dream

CoLA 线已经归档，结论很清楚：

- P0 证明 CoLA latent block 中存在可学习的 readiness / halt 信号。
- P1 证明 decoder-supervised、推理时 decoder-free 的轻量 student 可以学习 P0 信号。
- P2 证明 official8 不是天然 MAS benchmark，且 frozen / adapter CoLA 在 MuSiQue
  role/evidence-split 协议上的绝对能力不足，不能进入 TextMAS vs LatentMAS 主表。

因此下一阶段不应继续在 CoLA 权重上做局部修补，而应选择一个更适合 agent 任务的
DLM 作为同构通信 substrate。

## 2. 模型选择

### 2.1 主模型

```text
Dream-v0-Instruct-7B
```

理由：

- 它是开放 diffusion large language model，官方提供 Instruct checkpoint。
- 具备 chat / instruct 使用入口，更可能通过 MuSiQue role protocol。
- 官方 `diffusion_generate()` 暴露 `steps`、`output_history`、remasking strategy 等
  denoising trajectory 控制点，天然适合 P0/P1-style trace collection。
- 官方 GitHub 说明运行 Dream 至少需要 20GB GPU memory；8 张 RTX 5090 具备足够
  资源做 inference、trace collection、LoRA/adapter 和轻量判别器训练。

### 2.2 诊断模型

```text
Dream-v0-Base-7B
```

Base 不作为第一主线 MAS 能力模型，只用于理解 Instruct tuning 对 denoising
trajectory、readiness 和 communication 的影响。

### 2.3 同构优先

第一阶段使用两个完全相同的 Dream-v0-Instruct-7B：

```text
Agent A = Dream-v0-Instruct-7B
Agent B = Dream-v0-Instruct-7B
same tokenizer
same architecture
same weights
different role prompt / private evidence
different communication channel
```

异构通信暂不做。跨 Dream 与其他 DLM/AR-LLM substrate 的 latent space 不对齐，
失败原因会混入 translator / projector / architecture mismatch。异构通信应等
同构 Dream LatentMAS 成立后再单独设计。

## 3. Benchmark 选择

### 3.1 主 benchmark

```text
MuSiQue evidence-split QA
```

理由：

- MuSiQue 本身面向 connected multihop reasoning，设计目标是减少 shortcut。
- 工作区已有历史 TextMAS 证据：MuSiQue strict evidence-split calibration 和
  held-out 都通过 gate。这只说明 benchmark/protocol 可用于 true MAS，不代表
  P3 继续使用历史 text-agent 模型。
- 它能自然构造 role agents：不同 evidence agents 持有互补私有 evidence，final
  solver 必须依赖上游消息。
- 它支持明确控制组：no-message、shuffled-message、wrong-evidence、compressed-state。

### 3.2 备选 benchmark

```text
2WikiMultiHopQA evidence-split QA
```

作为第二候选。它同样适合 evidence split，但需要先做 field/license/scorer 和
shortcut-risk 审计。

### 3.3 暂不作为主线

```text
HotpotQA
official8
GSM8K
ARC / GPQA / MedQA
普通 single-agent coding/math benchmark
```

HotpotQA 在当前工作区已有 no-message / shortcut 风险偏强的诊断结果。official8、
GSM8K、ARC 等任务可以测 solver capability，但不天然要求 Agent A -> B 通信。

## 4. MAS benchmark 与普通 benchmark 的结构区别

普通 single-agent benchmark 通常是：

```text
question + all context -> solver -> answer
```

本项目需要的 true MAS benchmark 必须满足：

```text
1. 信息分布式：
   Agent A / evidence agents 只看到部分 private evidence。

2. 通信必要：
   final solver / Agent B 没有上游消息时显著变差。

3. 控制组强：
   no_message、shuffled_message、wrong_evidence、metadata_only 必须低于 matched。

4. receiver-only scoring：
   scorer 只看 Agent B / final solver 在 handoff 后生成的答案。

5. paired comparison：
   同一个 sample 同时评估 TextMAS 与 LatentMAS。
```

这几个条件是 CoLA P2 最大踩坑之后的硬约束。

## 5. Dream DLM 中的 latent 是什么

Dream 与 CoLA 不同：

```text
CoLA:
  Text VAE latent blocks + block-causal DiT + decoder

Dream:
  discrete diffusion / iterative denoising over token sequence
  Transformer internal hidden states / logits / confidence trajectory
```

Dream 的外显 token 是 denoising 过程中的草稿；latent communication 关心的是生成
草稿的内部连续状态：

```text
hidden states
attention / residual states
logit distributions
token confidence map
mask / remasking / token-change dynamics
denoising step trajectory
```

训练阶段可以 decode 中间 step 来构造 teacher label；在线判别和 latent
communication 不应把 decoded text 当输入。

## 6. P0/P1 如何迁移到 Dream

### 6.1 从 CoLA block 到 Dream denoising step

```text
CoLA block b
  -> Dream denoising step t

CoLA block-readiness
  -> Dream step-readiness

CoLA latent block tensor
  -> Dream hidden/state/logit/mask trajectory
```

Dream 不是少数 4 个 block，而是多个 denoising steps。因此需要采样 checkpoint：

```text
steps = 32 / 64 / 128 / 256 / 512
snapshot stride = every 4 / 8 / 16 steps, plus final step
```

具体 stride 由显存、速度和 answer-change 密度决定。

### 6.2 P0 teacher

P0 是离线 teacher，不是最终部署策略。它可以使用 decoder/text/scorer-derived 信号：

```text
for each sample, each selected denoising step:
  decode current sequence
  extract short answer
  score against gold / aliases
  compare answer identity with final and prediction-stability reference
  measure whether future steps change answer identity
  measure whether Agent B would benefit from receiving this state
```

P0 labels：

```text
answer_readiness:
  current step is enough to produce stable/correct final answer

prediction_change_risk:
  current answer identity or extracted answer will change later

future_gain:
  expected correctness / stability gain from continuing denoising

receiver_usefulness:
  whether passing this state helps Agent B versus no-message / shuffled controls
```

### 6.3 P1 DreamStepReadinessStudent

P1 是在线 student。训练标签来自 P0，但在线输入只包含 Dream latent/process state。

候选架构：

```text
Step-State Encoder
  token hidden states / selected layers / logits summary

Token or Slot Pooler
  PMA queries or answer-region-aware pooling

Trajectory Encoder
  causal Transformer over denoising-step snapshots

Process Feature Encoder
  mask ratio, changed-token ratio, entropy, confidence, step index, remaining steps

Readiness Heads
  halt_action
  answer_identity_stability
  continuation_risk
  future_gain
  receiver_usefulness
```

在线判定：

```text
halt if readiness high
and continuation risk low
and uncertainty / calibration constraints satisfied
else continue denoising
```

P1 的目标不是直接二分类“对/错”，而是学习 denoising trajectory 中是否已经形成
足以通信和作答的 answer-ready state。

## 7. Latent communication 主实验

### 7.1 Agent 数据流

正确数据流：

```text
Agent A private evidence
  -> Dream A denoising
  -> P1 readiness selects step/state
  -> A sends text message or latent packet
  -> Agent B consumes message/state with its own public question/private context
  -> Agent B generates final answer
  -> scorer evaluates Agent B final output only
```

禁止数据流：

```text
A decoded answer直接拼进 scorer-visible final answer
A replay decoded tokens 进入 B final generation
使用 gold/scorer/correctness 作为 A/B online input
在 held-out 上调 prompt / threshold / adapter
```

### 7.2 TextMAS baseline

```text
textmas_matched:
  A emits concise evidence/answer-state text message
  B receives that message and answers

textmas_no_message:
  B receives empty message

textmas_shuffled_message:
  B receives another sample's A message

textmas_wrong_evidence:
  A is fed wrong/non-self evidence shard

textmas_compressed_state:
  A emits constrained structured state, not full CoT
```

TextMAS 必须先在 Dream-v0-Instruct-7B 上通过 calibration 和 locked held-out gate。

### 7.3 LatentMAS variants

按保守到激进排序：

```text
latent_process_certificate:
  P1 heads + process state + compact latent summaries
  主要用于 receiver policy / confidence，不应作为唯一主 latent channel

latent_hidden_packet:
  selected layers' hidden states at readiness step
  compressed or selected positions

latent_embedding_packet:
  Dream denoising state / predicted clean embedding x_pred
  需要 receiver-side injection / alignment

latent_state_with_adapter:
  small projector / fuser maps A state to B-consumable embedding/state
  训练只用 calibration/train split
```

主 claim 应至少包含一个真正被 B 消费的 latent state，而不是只传 P1 scalar heads。

### 7.4 Receiver integration

优先级：

```text
1. Same-substrate native injection:
   将 A 的 aligned hidden/embedding packet 作为 B 的额外 continuous prefix 或
   working-memory state。

2. Lightweight receiver adapter:
   小 projector / cross-attention fuser，把 A packet 映射到 B 可消费的 embedding
   或 conditioning states。

3. Decoder-text bridge:
   只作为 diagnostic，不作为 LatentMAS 主结果。
```

Receiver 必须通过 corruption controls：

```text
latent_matched > latent_shuffled
latent_matched > latent_wrong_sample
latent_matched > latent_noise
latent_matched > metadata_only
```

## 8. 主指标

准确性：

```text
primary score
exact match
token F1
paired bootstrap CI lower bounds
```

通信有效性：

```text
latent_matched - no_message
latent_matched - shuffled
latent_matched - wrong_evidence
latent_matched - textmas_matched
```

成本：

```text
Dream denoising steps used by A
Dream denoising steps used by B
decoded output tokens
wall-clock latency
GPU memory peak
packet size in floats / bytes
```

早停质量：

```text
loss vs final
loss vs prediction-stability
mismatch vs final
future-gain calibration
risk-control upper bounds
```

## 9. 成功标准

最低进入 LatentMAS 主表的条件：

```text
Dream TextMAS held-out admitted=true
single_full_info > single_q_only with positive paired CI lower
textmas_matched > no_message / shuffled / wrong_evidence with positive paired CI lower
latent protocol passes receiver-only audit
latent_matched beats latent corruption controls
```

强成功标准：

```text
latent_matched >= textmas_matched - small_margin
or latent_matched > textmas_matched
while using fewer decoded tokens / lower communication bytes / fewer A denoising steps
```

早停判别器成功标准：

```text
P1 student preserves P0 readiness signal
low observed loss vs final / prediction-stability
calibrated risk below locked threshold
block/step saving is meaningful on held-out
```

## 10. 主要风险

```text
Dream context length 2048:
  MuSiQue evidence must be support-only / compressed. Long evidence prompts may fail.

Dream instruction ability:
  Instruct checkpoint likely works better than Base, but must pass TextMAS gate.

Hidden-state injection:
  A's state may be out-of-distribution for B unless aligned or adapted.

Latent shortcut:
  scalar readiness heads alone may classify packet quality but not transmit usable evidence.

Held-out leakage:
  prompt/parser/adapter/threshold must freeze before held-out.

Over-small smoke tests:
  smoke can validate wiring only, never prove architecture success/failure.
```

## 11. Sources

Sources checked on 2026-06-06:

- Dream paper: `https://arxiv.org/abs/2508.15487`
- Dream GitHub: `https://github.com/DreamLM/Dream`
- Dream-v0-Instruct-7B model card: `https://huggingface.co/Dream-org/Dream-v0-Instruct-7B`
- Dream-v0-Base-7B model card: `https://huggingface.co/Dream-org/Dream-v0-Base-7B`
- MuSiQue paper: `https://arxiv.org/abs/2108.00573`
- LatentMAS paper: `https://arxiv.org/abs/2511.20639`
- CoLA archive entry: `/data1/luyifei/drla/docs/cola_archive/README.md`
