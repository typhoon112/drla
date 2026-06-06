# Diffusion Latent Reasoning Framework

> 状态：长期背景设计。本文用于理解早期 DRLA latent reasoning 框架设想，不作为当前 P2 实施方案。

## 0. 摘要

本文提出一个新的 latent-space reasoning 框架：**DiT Reasoning Latent Agent, DRLA**。

核心主张是：

```text
用 Cola-style block-causal DiT / Flow prior 在 latent space 中逐块生成 reasoning latent，
替代 hidden-state autoregressive latent thought，
并在每个 latent block 后用 halt-verifier 判断 answer 是否已经 enough。
```

这意味着推理主体不再是：

```text
h_k -> align -> e_{k+1} -> LLM forward -> h_{k+1}
```

而是：

```text
question
  -> conditional block-causal DiT reasoning prior
  -> z_block_1, z_block_2, ...
  -> 每个 block 后 halt / verifier
  -> enough 后 final answer decoder
```

本文的关键取舍是：**不把所有论文平均融合**。Cola-DLM、ELF、CODI、CoLaR、Coconut、LatentMAS 都提供了有价值的局部经验，但新的主干应围绕一个更自然的抽象建立：**reasoning latent 是服务答案形成、验证和通信的隐式推理状态，而不是文本 token 的连续替身**。

---

## 1. 问题重新定义

### 1.1 不能把 latent 直接喂回 encoder

当前最大的接口问题来自这个直觉：

```text
去掉 decoder 后，
把 DiT 生成的 latent 直接放回 encoder，
让系统继续推理。
```

这个设计通常会产生分布错配。原因是 encoder 的输入接口通常是：

```text
token ids / token embeddings / text-conditioned hidden states
```

而 DiT 输出的 latent 是：

```text
由 prior 生成的连续样本
```

即使这个 latent 的 shape 与 encoder embedding 看起来相似，它也不一定处在 encoder 训练时见过的输入分布上。更严重的是，如果这个 latent 来自 VAE 或 diffusion prior，它的语义角色也可能不同：

```text
encoder input embedding: 用于读取文本
VAE text latent: 用于重建文本
DiT prior latent: 用于拟合 clean latent 分布
reasoning latent: 用于支持答案形成
```

因此，新框架不应让 latent 在 encoder 和 decoder 之间硬循环，而应显式定义：

```text
谁产生 latent
latent 服务什么目标
谁消费 latent
消费接口如何训练
```

DRLA 的答案是：

```text
Reasoning Encoder 产生训练目标 latent
DiT Prior 学习从 question 生成该 latent
Answer Decoder / Readout 消费该 latent
Halt-Verifier 判断该 latent 是否 answer-ready
```

### 1.2 latent reasoning 不等于压缩 CoT token

一个常见误解是：

```text
latent reasoning = 把一段 CoT token 压缩成更少 latent token
```

这只是可行路线之一，但不是最本质的目标。

显式 CoT 是人类可读的推理轨迹，它包含很多表达性、格式性和教学性内容。模型真正需要的可能不是逐句复现这些文本，而是到达一个能支持答案生成的内部状态。这个状态可以包含：

- 题目约束的绑定结果。
- 中间变量或子问题结构。
- 候选解的隐式集合。
- 对答案的置信度。
- 对错误路径的排除信息。
- 可供 verifier 使用的证据。

因此，reasoning latent 应定义为：

```text
z_reason = compact answer-useful latent state
```

而不是：

```text
z_text = compressed CoT text representation
```

CODI 的重要启发正在这里：它不强迫 student latent 对齐每个 CoT token，而是对齐 teacher 在即将回答位置的 hidden activation。这个位置更接近 **answer-ready representation**。DRLA 应保留这个思想，并把它推广为 reasoning latent 的主要监督信号之一。

### 1.3 动态停止不能依赖 EOS

在 token 自回归模型里，停止由 EOS token 或 `max_new_tokens` 控制。

但 latent space 中没有天然 EOS：

```text
z_k ∈ R^d
```

它不是词表分布，也没有一个固定坐标能表示“结束”。因此，动态停止必须换成一个可训练的判别问题：

```text
当前 latent 是否已经足以支持正确、稳定的最终答案？
```

这比判断“是否像结束 token”更强，也更适合推理任务。停止标准应包含：

- 当前 latent 是否 answer-ready。
- decoder probe 是否能产生正确答案。
- 多次 probe 或连续 block 的答案是否稳定。
- verifier 是否认为答案正确。
- 继续 denoise / refine 的预期收益是否足够小。

---

## 2. 三种 latent 必须区分

### 2.1 text latent

text latent 服务于文本重建或文本生成。

典型代表是 Cola-DLM：

```text
text tokens
  -> Text VAE Encoder
  -> z_text
  -> DiT prior
  -> z_hat_text
  -> VAE Decoder
  -> text tokens
```

这个 latent 的核心约束是：

```text
decoder 能从它还原文本
prior 能生成 decoder-valid latent
```

它适合语言建模，但不自动等价于推理状态。

### 2.2 embedding latent

embedding latent 服务于连续 embedding flow。

典型代表是 ELF：

```text
tokens
  -> frozen T5 encoder
  -> contextual embeddings
  -> bottleneck
  -> flow / DiT
  -> clean embeddings
  -> shared decode mode + unembedding
  -> tokens
```

它的优势是使用预训练 contextual embedding，语义基础强。它的风险是没有显式区分：

```text
文本语义
推理过程
答案就绪状态
```

所以 ELF 能证明 whole-latent flow 可行，但不能直接作为 reasoning latent 的完整答案。

### 2.3 reasoning latent

reasoning latent 服务于答案形成、验证和 agent 通信。

DRLA 中的 reasoning latent 是一串按 block 组织的连续状态：

```text
z_reason = [z_1, z_2, ..., z_B]
z_b ∈ R^{block_size × d}
```

其中 `B` 是动态生成出的 block 数，`block_size` 是每个 block 内的 latent slot 数，`d` 是每个 slot 的维度。

它不要求与输入 prompt token 数一致，但它的总容量应随推理和答案信息量增长。例如：

```text
prompt: 40 个中文字符
z_reason: B 个 blocks，每块 16 × d
answer: 350 个中文字符
```

三者长度不需要一一对应；但如果 decoder 是 token-aligned VAE/Cola-style decoder，那么 latent block 数必须足够承载输出文本或推理信息。DRLA 因此采用 **adaptive block generation**：先生成最小数量的 blocks，然后每新增一个 block 就判断是否已经 answer enough。

---

## 3. 对现有路线的取舍

### 3.1 Cola-DLM：保留 block-causal prior，不照搬 text-latent 目标

Cola-DLM 的重要贡献是把文本生成拆成：

```text
Text VAE: text <-> latent
DiT prior: noise -> latent
Decoder: latent -> text
```

它证明了 continuous latent prior 可以作为语言建模的核心结构。尤其重要的是，Cola 把 diffusion 看成 **latent prior transport**，而不是 token-level observation recovery。

DRLA 应保留这个思想：

```text
DiT 不直接恢复 token，而是把 Gaussian noise transport 到有意义的 latent region。
```

但 DRLA 不应照搬 Cola 的 text-latent 目标。

Cola 的默认结构里，VAE patch size 通常为 1，意味着：

```text
1 token -> 1 latent vector
block size 16 -> 每块 decode 16 个 token
```

这对文本生成合理，但对推理不一定最自然。DRLA 应保留 **按 block 生成、按 block 停止** 的机制，但把 block 的语义从普通 text continuation 改成 answer-oriented reasoning / answer latent。也就是说：

```text
Cola: block = text latent continuation unit
DRLA: block = reasoning / answer latent continuation unit
```

因此，Cola 是 DRLA 最重要的结构参考：**block-causal DiT prior + accumulated latent blocks + decoder**。DRLA 的创新点不在于去掉 block，而在于把 block-level generation 用于 latent reasoning，并在 block 之间加入 answer-enough 判别。

### 3.2 ELF：保留 x-prediction 等训练技巧，不采用整段固定 latent 主干

ELF 的关键价值是证明整段 continuous embedding flow 可以训出来。它的配套技巧包括：

- `x-prediction`：直接预测 clean embedding。
- bottleneck：把高维 embedding 投影到较低维。
- shared denoise/decode：同一网络根据 mode 做去噪或最终解码。
- CFG：增强条件控制。
- SDE sampler：少步采样时减少误差积累。

这些对 DRLA 非常重要。特别是 `x-prediction`：

```text
DiT(z_t, q, t) -> z_0
```

比预测噪声或 velocity 更符合 reasoning latent 的目标，因为我们最终需要的是 clean answer-ready state。

但 DRLA 不应直接把 T5 contextual embedding 当 reasoning latent。T5 embedding 是文本语义表示，不保证包含完成推理所需的结构，也不保证适合 halt/verifier。

DRLA 应借鉴 ELF 的训练技巧：

```text
x-prediction
bottleneck
SDE / CFG sampling
```

但 DRLA 的主干不应是一次性 whole-latent denoising，而应是 Cola-style block-causal generation：每个 block 内部可以使用 ELF 式 x-prediction / SDE sampling，block 之间则由 causal prior 和 halt-verifier 控制。

### 3.3 CODI：保留 answer-ready distillation

CODI 最值得保留的是训练信号设计。

它发现可以用显式 CoT teacher 的 answer-position hidden state 来监督 implicit continuous CoT。这个信号比逐 token 模仿 CoT 更接近推理完成状态。

DRLA 中可定义：

```text
h_teacher_answer = teacher 在 "The answer is:" / 答案触发位置的 hidden state
z_reason = Reasoning Encoder(q, solution_trace, y)
```

训练目标之一：

```text
Align(Pool(z_reason), h_teacher_answer)
```

这能减少 reasoning latent 学成普通 text latent 的风险。

### 3.4 CoLaR：保留动态预算与 RL，不照搬 token compression

CoLaR 的核心价值是：

- reasoning length 可以动态控制。
- latent path 可以是 probabilistic。
- RL 可以优化正确率与长度的 tradeoff。

但 CoLaR 的压缩对象仍主要是 CoT token embedding：

```text
c 个 CoT token embedding -> 1 个 compressed latent
```

DRLA 不应把它作为主结构，因为这仍然把 reasoning latent 绑定到文本 CoT。更自然的做法是把 CoLaR 的思想迁移到 reasoning latent 上：

```text
probabilistic reasoning latent prior
adaptive block budget
reward = correctness - cost
```

### 3.5 Coconut / LatentMAS：作为 baseline，不作为主干

Coconut 和 LatentMAS 的主干是 hidden-state autoregressive latent thought：

```text
last hidden state -> input embedding -> next hidden state
```

这证明了 continuous thought 可行，也证明了 latent communication 可以减少文本通信成本。

但这条路线仍然保留了自回归结构：

```text
一步 latent thought 依赖上一步 latent thought
```

DRLA 的目标是用 DiT 替代这条自回归 latent loop。也就是说，Coconut / LatentMAS 应作为对比基线和通信启发，而不是新框架主干。

---

## 4. 推荐框架：DiT Reasoning Latent Agent

### 4.1 总体架构

DRLA 的主干应采用 **Cola-style block-causal latent generation**。也就是说，DiT 不一次性生成固定长度 latent，也不按 token 自回归生成，而是一次生成一个 latent block；每个 block 内部可以并行 denoise，block 之间保持因果依赖。

DRLA 包含五个模块：

```text
1. Reasoning Encoder
2. Block-Causal DiT Reasoning Prior
3. Answer Decoder / Readout
4. Block-level Halt-Verifier Head
5. Optional Reasoning Latent Bus
```

训练期：

```text
question q
solution trace r
final answer y
        |
        v
Reasoning Encoder E_R
        |
        v
clean reasoning latent blocks z_R^{1:B*}
        |
        +-------------------------+
        |                         |
        v                         v
DiT learns p(z_R^b | q, z_R^{<b})      Decoder learns p(y | q, z_R^{1:B*})
```

推理期：

```text
question q
   |
   v
Condition Encoder C(q)
   |
   v
for b = 1 ... B_max:
    sample Gaussian noise for current block
    DiT denoises z_b conditioned on C(q) and previous blocks z_{<b}
    append z_b to latent memory
    if b >= B_min:
        decoder probe + halt-verifier
        stop if answer enough

final accumulated latent blocks z_{1:B_stop}
   -> final answer decoder
```

这里的关键是：**动态停止发生在 block 级别，而不是 token 级别，也不主要发生在单个 block 的 denoising step 级别**。单个 block 内仍可使用固定或轻量自适应 denoising steps；真正决定答案长度和推理预算的是 `B_stop`。

### 4.2 Reasoning Encoder

Reasoning Encoder 的输入是：

```text
q: question / task
r: solution trace / CoT / proof / program trace
y: final answer
```

输出：

```text
z_R^{1:B*} = E_R(q, r, y)
z_R^b ∈ R^{block_size × d}
```

推荐初始配置：

```text
block_size = 16
B_min = 2 到 4
B_max = 根据任务设置，例如 16 / 32 / 64
d = 256 或 512
```

`B*` 是训练样本对应的目标 block 数。它可以来自 solution trace 的 latent 分块，也可以来自 answer/proof 的长度和信息量。对于 token-aligned decoder，如果 `patch_size = 1`，那么一个 block 约对应 `block_size` 个 token；如果 `patch_size = 2`，一个 block 约对应 `2 × block_size` 个 token。

Reasoning Encoder 不应只做文本自编码。它的目标应包含：

```text
1. 支持 final answer readout
2. 对齐 teacher answer-ready hidden state
3. 保持 latent space 平滑可拟合
4. 保留 verifier 所需的可判别信息
```

建议损失：

```text
L_answer = CE(Decoder(q, z_R), y)
L_kd = || Project(Pool(z_R)) - h_teacher_answer ||_1
L_info = checkpoint prediction / subgoal prediction
L_reg = KL or variance regularization

L_encoder = L_answer + λ_kd L_kd + λ_info L_info + λ_reg L_reg
```

其中 `checkpoint prediction` 可以来自结构化中间监督，例如数学题中的关键变量、代码题中的测试结果、证明题中的中间命题。若没有结构化标签，可先只使用 `L_answer + L_kd + L_reg`。

### 4.3 Block-Causal DiT Reasoning Prior

DiT 的目标是学习：

```text
p_ψ(z_R^{1:B} | q, m)
  = ∏_{b=1}^{B} p_ψ(z_R^b | q, m, z_R^{<b})
```

其中 `m` 是可选外部 memory，例如其他 agent 的 reasoning latent。

训练时从 Reasoning Encoder 得到 clean latent blocks。对每个 block 采样噪声：

```text
z_0^b = z_R^b
ε ~ N(0, I)
z_t^b = α_t z_0^b + σ_t ε
```

DiT 采用 `x-prediction`：

```text
z_pred^b = DiT_ψ(z_t^b, t, C(q), m, z_R^{<b})
L_dit = Σ_b ||z_pred^b - z_0^b||^2
```

选择 `x-prediction` 的原因：

- reasoning latent 的最终消费对象是 answer decoder，需要 clean state。
- 高维语义空间中直接预测 clean data 通常更稳定。
- halt/verifier 也需要判断当前 latent 到 clean answer-ready state 的接近程度。

这个 prior 与 Cola 的思想一致：当前 block 的生成条件包括 prompt latent / condition 和之前已经生成的 clean blocks。区别在于，DRLA 的 blocks 不是普通文本 continuation 的 text latent，而是 answer-oriented reasoning / answer latent blocks。

### 4.4 Answer Decoder / Readout

Answer Decoder 负责把：

```text
q + z_hat_R^{1:B}
```

转换成 variable-length final answer。

它不应让输出长度绑定到 prompt 长度，但需要与累计 latent blocks 的容量相匹配。DRLA 的长度控制来自：

```text
B_stop = halt-verifier 认为 answer enough 时的 block 数
```

推荐两种实现：

#### 方案 A：Cola-style latent decoder

```text
z_hat_R^{1:B}
  -> latent decoder
  -> token logits / answer tokens
```

如果 decoder 是 token-aligned，则大致有：

```text
decoded_token_capacity ≈ B × block_size × patch_size
```

例如：

```text
block_size = 16
patch_size = 1
350 tokens answer -> 至少约 22 个 blocks
```

这个方案最接近 Cola，也最适合验证“DiT 替代自回归推理 / 生成”的主张。

#### 方案 B：LLM cross-attention / prefix readout

```text
LLM decoder hidden states
  cross-attend to z_hat_R^{1:B}
  generate final answer autoregressively
```

优点：

- 支持长答案。
- 支持自然语言解释、代码、格式化输出。
- 与现有 LLM 能力兼容。

方案 B 更容易做 MVP，但它会引入一个风险：LLM decoder 可能绕过 latent 自己完成推理。因此如果使用方案 B，必须做 `q only / z only / q + z` 消融。

DRLA 的主推荐是：**若目标是忠于 Cola 式 latent generation，则优先采用方案 A；若目标是先验证 reasoning latent 是否有用，可以用方案 B 降低工程难度。**

关键原则：

```text
block-causal DiT 负责 latent reasoning / answer latent 生成；
halt-verifier 负责决定生成多少 blocks；
decoder 只消费累计 blocks 输出最终文本。
```

### 4.5 Halt-Verifier Head

Halt-Verifier Head 在每个 block 生成后判断：

```text
累计 latent blocks 是否已经足以生成正确且稳定的答案？
```

输出：

```text
H(q, z^{1:b}, probe_features) -> {
  p_answerable,
  p_correct,
  uncertainty,
  expected_gain,
  p_stable
}
```

输入特征不应只包含 raw latent。推荐使用：

```text
latent_summary = [
  mean_pool(z^{1:b}),
  max_pool(z^{1:b}),
  selected attention pooled slots,
  latest_block_summary,
  block_index_embedding,
  delta_between_recent_blocks
]

decoder_probe_features = [
  answer_found,
  answer_logprob,
  answer_entropy,
  eos_reached,
  generated_length,
  answer_changed_from_previous_probe,
  same_answer_streak
]

verifier_features = [
  verifier_score,
  self_consistency_score,
  constraint_satisfaction_score
]
```

停止条件：

```text
stop if:
  b >= B_min
  and
  p_answerable > τ_answerable
  and p_correct > τ_correct
  and uncertainty < τ_uncertainty
  and p_stable > τ_stable

force stop if:
  b == B_max
```

---

## 5. 训练方案

### Stage 0：数据构造

需要构造训练样本：

```text
(q, r, y)
```

其中：

- `q` 是问题或任务。
- `r` 是 solution trace，可以是 CoT、proof、program trace、tool-use trace、multi-agent transcript。
- `y` 是最终答案。

可以从三类来源获得：

```text
1. 人类标注数据集中的 CoT / solution
2. 强 teacher LLM 生成的 solution trace
3. TextMAS / debate / verifier pipeline 生成的成功轨迹
```

训练时必须过滤：

- final answer 错误的 trace。
- trace 与 answer 不一致的样本。
- 明显冗长、模板化、无效的 CoT。

### Stage 1：Reasoning Encoder + Answer Readout

目标：让 `z_R^{1:B*}` 成为一串可被 decoder 消费的 answer-ready latent blocks。

训练：

```text
z_R^{1:B*} = E_R(q, r, y)
y_hat = Decoder(q, z_R^{1:B*})

L = CE(y_hat, y)
  + λ_kd KD(Pool(z_R^{1:B*}), h_teacher_answer)
  + λ_reg Regularization(z_R^{1:B*})
```

成功标准：

```text
给定 gold z_R^{1:B*}，decoder 能高准确率恢复 final answer。
```

这一步非常关键。如果 gold latent 都不能被 decoder 使用，后面的 DiT prior 没有意义。

### Stage 2：Block-Causal DiT Reasoning Prior

目标：让 DiT 从问题条件和已有 latent blocks 逐块生成 `z_R^{1:B}`。

训练：

```text
z_0^{1:B*} = stopgrad(E_R(q, r, y))  # 初期推荐冻结 encoder 目标

for b in 1 ... B*:
    z_t^b = noise(z_0^b, t)
    z_pred^b = DiT(z_t^b, t, C(q), z_0^{<b})

L_dit = Σ_b ||z_pred^b - z_0^b||^2
```

建议先冻结 Reasoning Encoder，只训练 DiT。等 prior 稳定后，再考虑 joint training：

```text
L_joint = L_dit + λ_answer CE(Decoder(q, z_pred^{1:B*}), y)
```

这里要避免 decoder 过强导致 DiT latent 被忽略。可以通过 latent dropout / prompt dropout / answer-only control 对比来检测。

### Stage 3：Halt-Verifier Head

目标：学习 accumulated latent blocks 上的 answer-enough 停止边界。

对每个样本 rollout 到 `B_max` 个 blocks：

```text
z^1 -> z^2 -> ... -> z^{B_max}
```

在每个 block 生成后做 decoder probe，通常从 `B_min` 开始：

```text
answer_b = Decode(q, z^{1:b})
correct_b = Judge(answer_b, y)
stable_b = SemanticStable(answer_b, answer_{b+1:b+r})
```

定义 earliest stable correct block：

```text
b* = min b such that:
  b >= B_min
  and correct_b = 1
  and stable_b = 1
  and verifier_score_b > threshold
```

标签：

```text
y_answerable(b) = 1 if decoder can extract valid answer from z^{1:b}
y_correct(b) = correct_b
y_stable(b) = stable_b
y_stop(b) = 1 if b >= b*
```

损失：

```text
L_halt =
  BCE(p_answerable, y_answerable)
  + BCE(p_correct, y_correct)
  + BCE(p_stable, y_stable)
  + BCE(p_stop, y_stop)
  + MSE(expected_gain, future_reward_gain)
```

若没有任何 block 正确：

```text
y_stop(b) = 0 for b < B_max
y_stop(B_max) = 1
y_correct(b) = 0
```

这能避免模型学会无限继续。

### Stage 4：RL 优化

在 SFT / supervised prior 稳定后，引入 RL 优化：

```text
same q -> sample multiple latent paths / stopping points
```

奖励：

```text
R =
  1(final answer correct)
  - α * normalized_blocks
  - β * decoder_tokens
  - γ * latent_bandwidth
  - δ * early_stop_wrong_penalty
```

其中 early-stop wrong penalty 必须较大：

```text
if stopped early and answer wrong:
  penalty = large
```

否则模型会为了省 block 学会过早停止。

---

## 6. 推理流程与动态停止

### 6.1 Block-causal DiT reasoning

基础推理流程：

```text
q
  -> condition C(q)
  -> for each block b:
       initialize current block noise
       DiT denoise current block conditioned on previous blocks
       append clean block to latent memory
       if b >= B_min: decoder probe + halt check
  -> final answer decode
```

伪代码：

```python
def drla_infer(q):
    cond = condition_encoder(q)
    blocks = []
    probe_history = []

    for b in range(1, B_max + 1):
        z_b = sample_gaussian(shape=(block_size, d))
        z_b = dit_denoise_block(z_b, cond=cond, prev_blocks=blocks)
        blocks.append(z_b)

        if b >= B_min:
            answer_probe = decoder_probe(q, blocks)
            features = build_halt_features(q, blocks, answer_probe, probe_history)
            score = halt_verifier(features)
            probe_history.append(answer_probe)

            if halt(score):
                break

    return answer_decoder(q, blocks)
```

### 6.2 两种停止层级

DRLA 中有两种动态停止。

#### 层级 A：block-level answer-enough halt

判断：

```text
累计到当前 block 的 latent memory 是否已经足够 answer-ready？
```

这是 DRLA 的主停止机制，也是最接近 Cola block-causal 生成的版本。

#### 层级 B：within-block denoising halt

在单个 block 内部，也可以判断 denoising steps 是否足够：

```text
z_t^b -> z_{t-1}^b -> ... -> z_0^b
```

但这只是采样效率优化，不负责主要的答案长度和推理预算控制。

判断：

```text
当前 block 是否已经 denoise 到足够 clean？
```

建议 MVP 先固定每个 block 的 denoising steps，只学习 block-level halt。等 block halt 稳定后，再研究 within-block adaptive denoising。

### 6.3 为什么要 decoder probe

直接从 high-dimensional latent 判断是否完成非常难。latent 本身信息量大、分布复杂，单纯 MLP 判别器容易学到伪相关。

因此 halt head 应该接入 decoder-side signal：

```text
latent 本身告诉我们“内部状态像不像完成”
decoder probe 告诉我们“这个状态能不能实际产出答案”
verifier 告诉我们“产出的答案是否可信”
```

这比只看 latent 更可靠。

---

## 7. 多 Agent 扩展

### 7.1 主通信协议：reasoning latent slots

在新框架中，多 agent 通信的自然单位不应是完整 KV cache，而应是：

```text
message_i = {
  z_i^{1:B_i}: B_i × block_size × d reasoning latent blocks,
  confidence,
  uncertainty,
  role_embedding,
  optional metadata
}
```

原因：

- KV cache 绑定具体模型结构、层数、head 数和 tokenizer。
- KV cache 体积大，传输成本高。
- reasoning latent 更像任务级语义协议，可跨 agent 共享。
- halt/verifier 可以直接对 message 评分。

KV cache 可以保留为辅助：

```text
z_i -> KV prefix adapter -> receiver LLM
```

但不作为主协议。

### 7.2 Planner-Solver-Verifier 架构

推荐的多 agent 数据流：

```text
Question
   |
   v
Planner DiT
   -> z_plan
   |
   v
Solver DiT conditioned on z_plan
   -> z_solution
   |
   v
Verifier DiT conditioned on z_plan + z_solution
   -> z_verify
   |
   v
Final Answer Decoder
   -> answer
```

各 agent 的 latent 语义：

```text
z_plan: 问题分解、约束、策略
z_solution: 具体求解状态
z_verify: 检查、反例、置信度、修正建议
```

### 7.3 Reasoning Latent Bus

当 agent 数量增加时，可以引入共享 latent bus：

```text
Agent_i local z_i
  -> write adapter
  -> Shared Reasoning Latent Bus Σ
  -> read adapter
  -> Agent_j condition
```

这比 pairwise KV fuser 更可扩展。每个新 agent 只需学习：

```text
local z <-> shared Σ
```

而不是为每一对 agent 训练 adapter。

### 7.4 与 LatentMAS / C2C / KV Alignment 的关系

DRLA 不否定 KV communication，而是重新定位它。

```text
LatentMAS: KV 是主通信载体
C2C: 异构 KV 转换与融合
KV Alignment: 多模型共享 KV latent space
DRLA: reasoning latent 是主通信载体，KV 是可选兼容层
```

如果需要连接现有 LLM agent，可以采用：

```text
z_reason -> KV prefix adapter -> receiver past_key_values
```

但长期目标应是：

```text
agent 间共享 reasoning latent，而不是共享模型内部缓存。
```

---

## 8. 关键实验设计

### 8.1 Baselines

必须比较以下路线：

```text
1. Text CoT
2. No-CoT direct answer
3. Coconut / LatentMAS-style autoregressive latent
4. Cola-style text latent DiT
5. ELF-style embedding flow
6. DRLA reasoning latent DiT
```

其中 `4` 和 `5` 可以是简化复现或近似 baseline。核心目标是证明：

```text
reasoning latent DiT 优于普通 text latent / embedding latent，
至少在 reasoning accuracy-cost tradeoff 上更合理。
```

### 8.2 Ablations

关键消融：

```text
with / without CODI-style answer-ready distillation
fixed block budget vs adaptive block halt
decoder-only halt vs verifier-assisted halt
text latent target vs reasoning latent target
block-causal DiT vs one-shot whole-latent DiT
x-prediction vs v-prediction vs noise-prediction
frozen encoder target vs joint encoder-DiT training
with / without latent bottleneck
with / without RL budget optimization
single agent vs Planner-Solver-Verifier
reasoning latent communication vs KV communication
```

### 8.3 Metrics

必须报告：

```text
Accuracy / pass@1
End-to-end latency
Generated text tokens
Generated latent blocks
Within-block denoising steps
Decoder probe count
Early-stop wrong rate
Forced-stop rate
Answer stability
Latent bandwidth
GPU memory peak
Verifier false-positive rate
```

其中最重要的不是单一 accuracy，而是：

```text
accuracy vs reasoning cost frontier
```

### 8.4 推荐任务

第一阶段建议选择可自动判分的任务：

```text
GSM8K / GSM-Hard
MATH / MATH-500
ProntoQA
ARC-C
OpenBookQA
HumanEval / MBPP for code
```

原因：

- final answer 可判定。
- CoT / solution trace 可由 teacher 生成。
- early-stop wrong rate 可量化。

---

## 9. 风险与缓解

### 9.1 reasoning latent 学成 text latent

风险：

```text
Reasoning Encoder 只是压缩 solution text，
没有学到 answer-ready reasoning state。
```

缓解：

- 使用 CODI-style answer-ready hidden-state distillation。
- 加入 final-answer readout loss，而不是只重建 CoT。
- 加入 checkpoint / verifier supervision。
- 做 text reconstruction probe，避免 reconstruction 目标过强。

判断标准：

```text
如果去掉 solution trace 表面词汇后 latent 仍能支持答案，
说明它更接近 reasoning latent。
```

### 9.2 DiT prior 与 encoder posterior 不匹配

风险：

```text
Encoder 产生的 z_R 是 decoder-valid，
但 DiT 生成的 z_hat_R 落在 decoder 不熟悉区域。
```

缓解：

- 先冻结 encoder 训练 DiT，稳定后再 joint training。
- 使用 decoder-validity loss。
- 在 DiT 输出上训练 answer decoder probe。
- 使用 prior hit / decoder probe success 作为诊断指标。

判断标准：

```text
gold z_R decode accuracy 高，
generated z_hat_R decode accuracy 低，
则瓶颈在 prior。
```

### 9.3 halt head 误停

风险：

```text
模型为了省 block 提前停止，导致错误答案。
```

缓解：

- p_answerable + p_correct + p_stable 多头判别。
- 引入 decoder probe 和 verifier score。
- 使用 earliest stable correct block 标签。
- RL 中重罚 early-stop wrong。
- 设置 `B_min` 和 conservative threshold。

判断标准：

```text
early-stop wrong rate 必须单独报告，
不能被平均 latency 掩盖。
```

### 9.4 decoder 过强导致 latent 被忽略

风险：

```text
Answer Decoder 直接依赖 question 自己解题，
z_R 变成无用旁路。
```

缓解：

- question dropout / prompt dropout。
- latent ablation 测试。
- decoder frozen 或限制 decoder capacity。
- 对比 `q only`、`z only`、`q + z` 三种设置。

判断标准：

```text
q + z 显著优于 q only，
且 z ablation 会明显降分。
```

### 9.5 多 agent latent 污染

风险：

```text
上游 agent 的错误 latent 误导下游 agent。
```

缓解：

- 每个 latent message 携带 confidence / uncertainty。
- receiver 使用 gated conditioning。
- verifier agent 检查 message。
- 训练时加入 bad-message dropout。

判断标准：

```text
加入弱 agent 后不能显著拖垮强 receiver。
```

---

## 10. 推荐路线图

### Phase 1：单 agent DRLA proof-of-concept

目标：

```text
证明 reasoning latent DiT 可以从 question 生成可用 answer-ready latent。
```

实现：

- 数据：GSM8K 或 ProntoQA。
- Teacher：生成 solution trace 和 answer-ready hidden state。
- Reasoning Encoder：输出 `B* × block_size × d` latent blocks。
- Answer Decoder：优先采用 Cola-style latent decoder；MVP 可用 latent-to-prefix adapter 降低难度。
- DiT：block-causal x-prediction，每次生成一个 block。

核心对比：

```text
gold z_R decode
DiT z_hat_R decode
Text CoT
No-CoT
autoregressive latent baseline
```

成功标准：

```text
DiT-generated z_hat_R 明显提升 answer accuracy，
且比显式 CoT 使用更少生成 token。
```

### Phase 2：adaptive block halt

目标：

```text
证明动态停止可以减少生成的 latent blocks，
同时控制 early-stop wrong rate。
```

实现：

- Rollout 到 `B_max` 个 latent blocks。
- 每个 block 后 decoder probe。
- 构造 earliest stable correct block。
- 训练 Halt-Verifier Head。

核心对比：

```text
fixed 4 / 8 / 16 / 32 blocks
adaptive block halt
oracle block halt
```

成功标准：

```text
adaptive halt 接近 oracle frontier，
并显著低于固定大 block 数的平均成本。
```

### Phase 3：multi-agent reasoning latent communication

目标：

```text
证明 reasoning latent slots 可以作为 agent 间通信协议。
```

实现：

```text
Planner -> z_plan
Solver conditioned on z_plan -> z_solution
Verifier conditioned on z_plan + z_solution -> z_verify
Final decoder -> answer
```

核心对比：

```text
TextMAS
LatentMAS KV communication
DRLA reasoning latent communication
single DRLA
```

成功标准：

```text
DRLA multi-agent 在 accuracy-cost frontier 上优于 TextMAS 和 KV-only latent communication。
```

### Phase 4：RL optimization

目标：

```text
优化 correctness、block cost、decoder cost 和 communication bandwidth。
```

实现：

- 对同一问题采样多个 latent paths。
- 使用 correctness 和 cost-based reward。
- 强化 halt policy 和 probabilistic latent prior。

成功标准：

```text
在相同 accuracy 下减少 generated blocks / latent bandwidth，
或在相同 cost 下提高 accuracy。
```

---

## 11. 结论

DRLA 的本质不是把 Cola、ELF、Coconut、CODI、CoLaR 平均混合，而是重新定义 latent reasoning 的对象。

最终取舍如下：

```text
不采用 hidden-state autoregressive latent thought 作为主干；
不采用 token-aligned text latent 作为推理 latent；
不把 T5 embedding flow 直接等同于 reasoning；
不依赖 EOS 解决动态停止。
```

保留的核心洞察是：

```text
Cola: diffusion 应建模 latent prior transport
Cola: block-causal latent generation 是最自然的长度控制方式
ELF: x-prediction / bottleneck / SDE sampling 等训练技巧可借鉴
CODI: answer-ready hidden state 是强监督信号
CoLaR: 动态预算和 RL 对 latent reasoning 很重要
LatentMAS: agent 间 latent communication 有系统价值
```

因此，最自然的新框架是：

```text
Reasoning Encoder 从 solution trace 中学习 answer-ready latent blocks；
DiT 从 question 条件逐块生成 reasoning / answer latent；
Halt-Verifier 在 block generation 过程中判断何时停止；
Answer Decoder 只负责最终答案表达；
多 agent 之间传 accumulated reasoning latent blocks，而不是默认传完整 KV cache。
```

这条路线把“推理”和“表达”拆开：

```text
DiT latent prior: 负责隐式推理状态生成
Decoder / readout: 负责最终文本表达
Verifier / halt: 负责停止和可靠性
```

如果目标是构建真正绕过自然语言 CoT 的 latent-space reasoning system，这比“压缩 CoT 文本”或“KV cache 传递”更接近本质。

---

## 12. 参考材料

本方案基于当前工作区内的调研文档、论文原文和源码阅读：

- `Latent_Space_Agent_Communication_Reading_Report.md`
- `Adaptive_Latent_Communication_MAS_Design.md`
- Cola-DLM paper and code: `/data1/luyifei/Cola-DLM/`
- ELF paper source: `/data1/luyifei/ELF/`
- Coconut / CODI / CoLaR papers: `/data1/luyifei/latent_reasoning_papers/`
