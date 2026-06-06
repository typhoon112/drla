# DRLA 多尺度递归 Block 与 Sub-block Halt 设计备忘

Last updated: 2026-05-26

> 状态：不成熟 try / 当前不考虑。本文只记录一个多尺度递归 block 与 sub-block halt 设想，不纳入当前实验路线、P2 优先级或主文档入口，也不能作为架构有效性的证据。

## 0. 一句话结论

`block_size=1` 在 Cola 消融中仍然 competitive，说明 token/latent-position 级别的连续 latent 推理不是死路；`block_size=16` 更强，说明局部语义聚合仍然重要。

因此，一个值得认真评估的方向是：

```text
big block = 16
small decision granularity = 1 / 2 / 4 / 8 / 16

用小粒度提供 halt/readiness 判断点，
用逐级扩大的 block 恢复 Cola block16 的局部语义连续性。
```

但它不能直接作为当前 released Cola checkpoint 的 inference trick 使用。当前可立即开展的是 post-hoc sub-block readiness 诊断；若诊断显示显著价值，再训练或微调真正的 multi-scale Cola prior。

## 1. 背景事实

当前 official Cola-DLM released 配置是：

```text
VAE patch_size = 1
VAE latent_dim = 16
DiT block_size = 16
```

因此在当前配置中：

```text
1 generated DiT block = 16 latent positions
patch_size = 1 时，1 latent position 对应 1 token-position 的 decoder logits
```

注意：实验名里的 `b64` 指的是 `max_new_tokens=64`，不是 `block_size=64`。当前 `b64` 协议实际是最多生成 `64 / 16 = 4` 个 Cola DiT blocks。

Cola 论文的 block size 消融显示：

```text
block_size = 16 最强
block_size = 1 仍然 competitive，但弱于 16，尤其在 MMLU 等任务上
block_size = 64 / 128 明显变差
```

合理解释是：

```text
block_size = 1:
  更接近 autoregressive latent generation
  halt 粒度最细
  但缺少 within-block bidirectional semantic aggregation

block_size = 16:
  保留局部 block 内语义聚合
  并行生成效率更好
  但 halt 只能每 16 个 token-position 判断一次
```

这正好暴露出当前 DRLA halt 研究的一个结构性问题：

```text
Cola 的最优 generation block 粒度
不一定等于最优 halt/readiness 判断粒度。
```

## 2. 核心设想

设定一个 big block 覆盖 16 个 latent positions，但在内部引入多尺度递归过程：

```text
scale schedule: 1 -> 2 -> 4 -> 8 -> 16
```

每个 scale 都产生一个可 probe / 可 halt 的中间状态：

```text
z_1       -> probe/halt
z_1:2     -> probe/halt
z_1:4     -> probe/halt
z_1:8     -> probe/halt
z_1:16    -> probe/halt, big block complete
```

完成一个 big block 后，再进入下一个 big block：

```text
big block 0: positions 1..16
big block 1: positions 17..32
big block 2: positions 33..48
...
```

目标不是提高 Cola 官方 benchmark accuracy，而是研究：

```text
累计到当前 sub-block / scale 的 latent 是否已经足以生成正确且稳定的答案？
继续扩展到更大 scale 是否还有 future gain？
```

## 3. 两种可能实现语义

用户提出的 `concat` 需要区分两种语义。二者差别很大。

### 3.1 Prefix-expansion concat

这里的 concat 沿 sequence dimension 发生：

```text
先生成 position 1
再生成 position 2
concat -> prefix length 2
再生成 positions 3..4
concat -> prefix length 4
再生成 positions 5..8
concat -> prefix length 8
再生成 positions 9..16
concat -> prefix length 16
```

优点：

- 语义最自然。
- decoder 仍然看到一串 latent positions。
- halt 可以发生在 prefix length 1 / 2 / 4 / 8 / 16。

问题：

- 这会把原来一次 block16 solve 拆成多次 sequential solves。
- 若每个 scale 都跑完整 denoising，critical path 会明显变长。
- 当前 Cola checkpoint 没按这种 prefix-expansion schedule 训练，直接运行会产生分布错配。

### 3.2 Multi-resolution refinement concat

这里的 concat 表示把同一段内容的多尺度表征拼接起来：

```text
scale1 representation
concat scale2 representation
concat scale4 representation
...
```

这更像 coarse-to-fine / next-scale prediction。

优点：

- 更接近多尺度语义组织。
- 可以让模型先形成粗粒度 answer-ready state，再逐步补局部细节。

问题：

- VAE decoder 不能直接消费这种 multi-resolution feature。
- 需要 projection、scale embedding 或多尺度 VAE/DiT 训练。
- 不能简单把不同 scale latent 拼到一起就交给 released decoder。

结论：

```text
如果短期要做，优先考虑 prefix-expansion sub-block 诊断。
如果长期要做，multi-resolution refinement 是更强但更重的模型设计。
```

## 4. 当前 checkpoint 能否直接跑

结论：不能作为严肃实验直接跑。

原因：

1. 当前 Cola DiT 使用固定 `block_size=16`。block-causal attention mask 假设文本长度是固定 block size 的倍数，block-wise query 也是固定 block size。
2. 论文中 `block_size=1` 是另一个训练设定，不是当前 `block_size=16` checkpoint 的 runtime knob。
3. 当前 DiT 学到的是：

```text
previous clean blocks + current noisy 16-position block -> current clean block
```

而不是：

```text
previous clean blocks + current noisy 1/2/4/8-position prefix -> recursive expansion
```

4. 如果在同一个 big block 内多次更新 latent，KV cache、decoder cache、position indexing、attention mask 都需要重新定义。

因此当前可行性分三档：

| 方案 | 能否立即做 | 科学有效性 | 说明 |
| --- | --- | --- | --- |
| 直接把 current checkpoint 改成 variable block inference | 技术上可 hack，严肃性不足 | 低 | 分布错配，不可作为结论 |
| post-hoc sub-block readiness 诊断 | 可以立即做 | 中高 | 不改 prior，只评估细粒度 halt 的潜在价值 |
| 训练/微调 multi-scale Cola prior | 需要新工程与训练 | 高 | 真正验证该架构是否 work |

## 5. 效率账

当前 Cola block16 的理想形态是：

```text
一次 block solve -> 16 latent/token positions
```

如果每个 block 使用 `N` 个 denoising steps，那么一个 16-position block 的 sequential denoising depth 约为：

```text
T_block16 ~= N
```

递归 scale `1 -> 2 -> 4 -> 8 -> 16` 若每个 scale 都完整 solve，则：

```text
T_recursive ~= 5N
```

从 query token 数看，处理量近似从：

```text
16
```

变为：

```text
1 + 2 + 4 + 8 + 16 = 31
```

也就是 query-side token 量约 `31 / 16 = 1.94x`，但由于多了 5 个 sequential stages，真实延迟风险可能接近 `5x` 的 critical-path 代价。

因此该方案只有在以下条件下才可能赢：

```text
大量样本能在 1/2/4/8 阶段 halt
或者后续 scale 使用更少 denoising steps
或者小 scale probe 非常便宜，不完整调用 DiT
或者训练出可复用前一 scale 计算的 cache/refinement 结构
```

否则它会提升 halt 粒度，但牺牲 Cola block diffusion 的并行效率。

## 6. 与 halt 判别器的关系

这个想法与 DRLA 的主问题高度相关。

当前 block-level halt 问的是：

```text
看完第 b 个 16-position block 后，答案是否已经 enough？
```

multi-scale sub-block halt 问的是：

```text
看完当前 big block 内 1/2/4/8/16 个 latent positions 后，答案是否已经 enough？
```

它能提供更密集的 readiness 观测点：

```text
decoder EOS/im_end probe
answer stability
scored prediction stability
continuation risk
future gain
latent/process trajectory
denoising confidence dynamics
```

尤其是 SQuAD / LAMBADA 等容易出现 prefix answer 的任务，sub-block halt 可能帮助识别：

```text
当前答案只是最终答案的严格前缀
当前答案格式尚未闭合
当前 decoder 已想结束但 task answer 尚不稳定
继续生成只会重复或引入噪声
```

但是它也会引入更难的校准问题：

```text
scale=1 的 readiness 分布
scale=2 的 readiness 分布
scale=4 的 readiness 分布
scale=8 的 readiness 分布
scale=16 的 readiness 分布
```

这些分布不一定共享同一个阈值。

## 7. 相关工作与启发

这些工作不等价于本方案，但提供了有用证据。

### 7.1 Block Diffusion

Source: https://arxiv.org/abs/2503.09573

Block Diffusion 把语言生成放在 autoregressive 与 diffusion 之间，使用 block-wise autoregressive factorization 和 block 内 diffusion/parallel denoising。

启发：

- block size 是质量、效率、可控性之间的 trade-off。
- block diffusion 可以支持 flexible-length generation、KV cache 和 parallel token sampling。
- 我们的 sub-block halt 可以看成在 fixed block diffusion 外增加更细的 control/readiness 层。

### 7.2 AdaBlock-dLLM

Source: https://arxiv.org/abs/2509.26432

AdaBlock-dLLM 反对固定 block size，提出根据 denoising 过程中的 confidence dynamics 自适应调整 block size。

启发：

- 固定 block 会带来 late decoding overhead 和 premature decoding error。
- denoising trajectory 本身包含可用于调度的信号。
- 这支持我们把 denoising-process features 纳入 readiness/halt 判别器，而不是只看最终 decoded answer。

### 7.3 Dynamic Chunking for Diffusion Language Models

Source: https://arxiv.org/abs/2605.15676

DCDM 认为固定位置 block 会切断语义连续性，提出 content-defined semantic chunks。

启发：

- block 边界不应只由位置决定。
- 更自然的 halt 判别器也不应只问“到第几个固定 block”，而应问“当前语义单元是否 answer-enough”。
- 长期版本可以从 `1/2/4/8/16` 固定 scale 走向 content-aware chunking。

### 7.4 Dynamic Chunking for End-to-End Hierarchical Sequence Modeling

Source: https://arxiv.org/abs/2507.07955

该方向强调 end-to-end 学习 content/context-dependent segmentation。

启发：

- 多尺度 chunk 不是手工切分就能最优，需要模型学习边界。
- 我们可以先用固定 scale 作为诊断，再考虑 learnable boundary / dynamic chunk。

### 7.5 VAR / Next-Scale Prediction

Source: https://arxiv.org/abs/2404.02905

VAR 在视觉生成中使用 next-scale prediction，以 coarse-to-fine scale 作为生成顺序。

启发：

- `1 -> 2 -> 4 -> 8 -> 16` 的思想与 next-scale generation 精神相似。
- 但 VAR 的 tokenizer/model 是为多尺度训练的；这提醒我们不能把多尺度递归强行套到 released Cola checkpoint 上。

### 7.6 Blockwise Parallel Decoding

Source: https://arxiv.org/abs/1811.03115

Blockwise Parallel Decoding 一次预测多个未来 step，再用 scorer 接受可信 prefix。

启发：

- “先并行 draft，再验证最长可接受 prefix”与 readiness/halt 目标非常接近。
- 我们可以把 sub-block halt 看成 Cola latent 版的 prefix acceptance / validation。

## 8. 推荐实验路线

### Phase A: post-hoc sub-block readiness oracle

目标：不改 Cola prior，先回答“细粒度 halt 是否真的有价值”。

输入：

```text
existing official8 b64 block traces
per-block decoded logits / token ids / scored predictions
raw latent shards if available
```

做法：

```text
for each generated 16-position block:
    construct prefixes at positions 1, 2, 4, 8, 16
    compute decoder-side signals
    compute task-scored prediction if output is parseable
    compute prediction stability / text stability
    compute future gain relative to later prefixes and later blocks
```

记录指标：

```text
scale
block index
prefix length
EOS/im_end logits or indicators
token entropy / confidence
answer text
official processed prediction
correctness
prediction changed later?
strict-prefix / completion-risk label
future gain
oracle earliest scale
```

注意：

```text
这一步不能证明 variable-block generation 会 work。
它只能证明：如果拥有 sub-block 观测点，halt/readiness 理论上能否更早、更稳。
```

### Phase B: sub-block halt student

目标：训练一个读 sub-block signals / latent prefix 的 readiness model。

输入可以分三档：

```text
teacher-rich:
  decoder probe + answer stability + scorer + process features

process-only:
  token confidence / denoising confidence / position / scale / block index

latent-student:
  latent prefix + process trajectory + learned readout
```

训练规范：

```text
所有训练必须上 SwanLab cloud
本地写 metrics.jsonl
valid_interval <= 100 step
保存 best_checkpoint.pt
不能只看 last checkpoint
```

评估：

```text
official 8 tasks:
  lambada, mmlu, obqa, hellaswag, race, siqa, squad, story_cloze

primary:
  accuracy / correctness stability
  average token/block budget
  loss versus prediction-stability baseline
  text mismatch / strict-prefix risk

secondary:
  AUROC / calibration / per-task threshold stability
  leave-one-task-out transfer
```

### Phase C: true multi-scale Cola prior

只有 Phase A/B 显示明显价值时，才进入该阶段。

需要改动：

```text
generalized block-causal mask:
  support query length 1/2/4/8/16

scale embedding:
  tell DiT current scale and target prefix geometry

position/boundary protocol:
  define whether scale expansion writes new positions or refines old positions

KV/cache semantics:
  define whether previous sub-scale states are committed, replaced, or reused

training objective:
  clean latent target
  flow matching / diffusion loss
  scale curriculum
  readiness/future-gain auxiliary objective if needed
```

训练策略：

```text
start from official Cola VAE + DiT
prefer adapter/LoRA or partial fine-tuning for schedule adaptation
do not frame LoRA as benchmark accuracy improvement
full official8 evaluation required
small smoke tests only check code path, not architecture validity
```

## 9. 验收标准

该方向成立的最低标准不是“某个小实验 work”，而是：

```text
1. sub-block oracle frontier 优于当前 block16 halt/readiness frontier
2. improvement 不只是 EOS 或长度伪相关
3. 能减少 average generated budget
4. accuracy / scored prediction stability 不显著下降
5. strict-prefix / incomplete-answer 风险可控
6. 至少在 official 8-task 上分任务分析
7. 能与 prediction-stability baseline 和当前 risk-gated/readiness policies 对齐比较
8. 若训练模型，必须有 SwanLab 曲线、metrics.jsonl、best checkpoint、valid <= 100 step
```

推荐决策门：

```text
Gate A:
  如果 post-hoc sub-block oracle 都不能超过 block16 baseline，
  不进入 multi-scale prior 训练。

Gate B:
  如果 oracle 好，但 student 学不到，
  优先改 readout / calibration / teacher objective，
  不急着改 DiT。

Gate C:
  如果 multi-scale prior 的额外 sequential cost 大于 halt 节省，
  该方向只作为 interpretability/control 研究，不作为默认 inference path。
```

## 10. 当前判断

这个方向值得保留为 DRLA 的重要升级路线，因为它精确对应当前问题：

```text
generation block size 与 halt decision granularity 不必相同。
```

但它不能绕过严肃训练和评估：

```text
block_size=1 competitive
  != current block16 checkpoint 可以 runtime 改成 1

sub-block prefix 可判断
  != sub-block generation 一定可行

小样本 smoke test 成功
  != 架构在 official Cola benchmark 上成立
```

最稳妥的下一步是：

```text
先基于已有 official8 b64 traces 做 sub-block readiness oracle，
量化 1/2/4/8/16 这些判断点到底能带来多少额外 budget saving，
再决定是否进入真正 multi-scale Cola prior 的训练工程。
```
