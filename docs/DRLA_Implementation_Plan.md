# DRLA 具体实施方案

## 2026-05-24 路线修订：official Cola + answer-readiness / halt

当前主线从“自建小 prior / 从零训练 Cola-like prior / 把 DiT LoRA 当作主目标”修订为：

```text
official Cola VAE + official Cola DiT 作为冻结或半冻结 latent substrate
+ block-wise latent rollout
+ decoder-side probe
+ answer stability / task scorer / future gain
+ block-level answer-readiness / halt 判别器
```

核心目标不是让 released Cola 在官方 benchmark 上变强，而是读懂 Cola 的 latent 推理过程，并学习：

```text
累计到当前 block 的 latent memory 是否已经足以生成正确且稳定的答案？
继续生成更多 latent blocks 是否还有实质收益？
```

这意味着：

- DiT LoRA / adapter 降级为辅助或诊断工具，可用于接口适配、probe 研究或后续 policy 调优，不作为主线精度目标。
- halt / half 判别器统一按 `halt / answer-enough / answer-readiness` 理解。
- halt 判别器不得退化成 raw latent 二分类；短期可以用 decoder probe、answer stability、task scorer 等信号做 teacher / offline label，但最终 agent-to-agent latent 通信的在线 halt 输入必须优先收敛到 latent trajectory / process features / learned latent verifier proxy，不能依赖 decoder 输出。
- decoder 信息不能被丢弃。正确路线是 `decoder-as-teacher, latent-student halt`：训练期用 decoder probe / scorer / stability 提供密集监督，让判别器学习 latent 到 decoder-side readiness 信号的映射；推理期只读 latent / process trajectory，由 student 近似这些 answer-enough 信号。
- 已完成的 decoder-dependent / decoder-probed readiness 架构和实验结果是前一阶段核心成果，不能删除或视作失败路线。它们负责证明 readiness 信号存在、提供 teacher labels、作为 upper-bound / safety baseline，并为下一阶段 student-only 结果提供对照。
- 小规模实验只能验证工程链路，不得作为架构成败结论。完整 rollout、probe、label、train、eval、调优闭环必须搭完。

硬性实验规范：

```text
所有深度学习训练实验必须上 SwanLab cloud，训练 smoke/probe/ablation 也不例外
无训练过程的 eval/trace/frontier/aggregation 不上 SwanLab，只允许 `swanlab_mode=disabled`，只写本地 artifact
所有训练必须写本地 metrics.jsonl
valid 间隔不得超过 100 step
保存 best_checkpoint.pt，不能只保存 last checkpoint
评测必须记录完整动态曲线，不能只看最后 summary
trace 协议必须固定 batch size；batch size 不只是速度参数
```

主评测口径固定为 Cola 官方 8 个 benchmark：

```text
lambada, mmlu, obqa, hellaswag, race, siqa, squad, story_cloze
```

评测必须使用 `/data1/luyifei/Cola-DLM/code/scripts/acc_calc.py` 的 task-specific 规则。GSM8K 只保留为 OOD/math diagnostic，不再作为主线数据或主结论来源。

2026-05-24 诊断显示 official Cola batched inference 不是完全 batch-invariant：LAMBADA `bs20` vs `bs1` 在单点样本一致，但 MMLU `bs12` vs `bs1` 在单点样本不一致。因此主 trace 协议必须把 batch size 写入 run name/config，并且同一个主结果不得混合不同 batch size 的输出。当前 full prepared split 使用统一 `batch_size=12`。

---

## 0. 实施目标

本文档把 `docs/Diffusion_Latent_Reasoning_Framework.md` 中 6.2 节的 block-level answer-enough halt 落到可执行实验。

目标系统：

```text
prompt / task input
  -> official Cola VAE encode prefix latent z_pre

for b = 1 ... B_max:
    official Cola DiT samples current latent block z_b
    append z_b to latent memory

    decoder probe:
      decode accumulated blocks z_1 ... z_b
      collect token/logit/EOS/answer/stability signals

    readiness model:
      estimate answerability, stability, uncertainty, future gain
      decide continue or halt

final accumulated blocks
  -> official Cola VAE decoder
  -> answer text
  -> official task scorer
```

第一阶段要证明的不是“模型更聪明”，而是：

```text
1. Cola block-wise latent trajectory 中存在可观测 answer-readiness 信号。
2. 这些信号不只是输出长度或 EOS 的伪相关。
3. readiness / halt model 能缩短平均 block budget。
4. 缩短预算时 correctness 和 answer stability 不发生不可接受损失。
5. 信号能在 official 8-task benchmark 上分任务分析，并具备一定跨任务泛化。
```

明确不做的主线：

```text
不从零训练 Cola 级 VAE + DiT
不把 DiT LoRA 作为提高 official benchmark accuracy 的主目标
不以 GSM8K 小样本 overfit / underfit 判断架构成败
不使用 raw latent -> stop/continue 的直接二分类作为主模型
不只保存最后一轮 summary
```

---

## 1. Cola latent 事实与判别器边界

基于当前 released Cola 权重与开源推理代码：

```text
VAE patch_size = 1
VAE latent_dim = 16
DiT block_size = 16
```

因此当前配置下：

```text
1 个 generated block = 16 个 latent positions
1 个 latent position = 16 维 latent vector
patch_size = 1 时，1 个 latent position 对应 1 个 token-position logits
```

但这不等于“一个 latent vector 就是一个 token embedding”。VAE decoder 是带上下文和 KV cache 的 transformer decoder，某个位置 decode 成 EOS / im_end 取决于：

```text
当前 latent vector
prefix latent
previous generated blocks
decoder attention / KV state
position
sampling strategy
```

所以可以研究 `EOS/im_end latent probe`，但不能把它理解为全局固定的 EOS latent 原型。

合理做法：

```text
z_b,i + context summary + decoder state proxy
  -> EOS/im_end probability
  -> stop-token margin
  -> token entropy
  -> answer-readiness evidence
```

EOS/im_end probe 是 decoder-side halt evidence，不等于 task verifier。它只能说明模型是否倾向结束生成，不能说明答案是否正确。最终 halt 必须同时结合 answer stability 与 task scorer / verifier 信号。

---

## 2. 文献依据与可借鉴点

这些论文与当前实验不完全同题。它们多在 LLM hidden state / embedding space 中训练 latent CoT，而不是在 Cola-DLM diffusion latent block 上训练 halt 判别器。因此不能直接声称它们证明本架构有效。

它们的作用是提供设计依据和避坑经验。

### 2.1 COCONUT

可借鉴点：

- continuous thought 不应被迫逐字压缩被替换的 CoT；目标是帮助未来答案生成。
- latent thought 可以通过 decoder probe 解释，这支持我们做 per-block decoder probe。
- latent mode 的停止难做；固定长度只是简化方案，不应成为我们的最终主线。
- curriculum 很关键，直接进入最终 latent 阶段容易失败。
- 一次加入太多 continuous thoughts 会导致 loss spike，说明 latent 训练和预算调度必须看完整曲线。

对本项目的要求：

```text
不要用单次小样本结果判断 latent 架构。
必须记录 train/valid 动态、loss spike、readiness 曲线和 block budget 曲线。
```

### 2.2 CODI

可借鉴点：

- 关键监督不是每个 CoT token，而是 answer-generating hidden state。
- feature-level distillation 能把 explicit CoT 的 reasoning 信息转移到 continuous thought。
- 只用最终答案监督太稀疏，容易学不到连续推理状态。
- token-level probe 会漏掉多 token 实体，关键 probe token 选择也可能偏。

对本项目的要求：

```text
readiness model 应关注 answer-generating state。
per-block decoder probe 不只保存文本，还要保存 logits、margin、entropy、EOS/im_end、answer prefix 和多 token answer 状态。
直接 raw latent 二分类不能作为主判别器。
```

### 2.3 CoLaR

可借鉴点：

- dynamic chain length 和 termination 是 latent reasoning 的核心问题。
- deterministic latent reasoning 在困难任务上探索能力不足。
- probabilistic latent head / uncertainty / RL-style reward 能帮助探索和压缩预算。
- 只用最终 answer supervision 太稀疏，dense reasoning supervision 很重要。
- reward 如果设计不好，会出现过早停止、长度冲到上限或 collapse。

对本项目的要求：

```text
halt reward 不能只奖励短。
必须同时度量 correctness、stability、future gain、uncertainty 和 average blocks。
optional RL-style policy 调优必须在 oracle/readiness 信号可靠之后再做。
```

### 2.4 Latent Space Survey

可借鉴点：

- latent space 是 machine-native、continuous、efficient、high-fidelity 的内部计算空间。
- 代价是 evaluability、controllability、interpretability 都更困难。
- 当前项目属于 component / auxiliary latent control：不重训 backbone，而是给 latent computation 增加可观测、可校准的控制接口。

对本项目的要求：

```text
readiness / halt 判别器本质上是 latent computation 的控制界面。
实验必须同时报告效果、可解释性、校准和泄漏风险。
```

### 2.5 综合设计结论：decoder-as-teacher, latent-student halt

当前模型修订不是“去掉 decoder 信息”，而是把 decoder 从在线输入改成训练期 teacher。

文献对应关系：

- COCONUT 说明 continuous thought 可被 decoder / LM head probe，但 latent reasoning 需要 curriculum 和完整 loss 曲线监控；因此 decoder probe 是合理观测工具，但不能等价为最终 halt 输入。
- CODI 说明只靠最终答案监督太稀疏，answer-generating hidden state / feature-level distillation 是让 student 学到 continuous reasoning 的关键；因此 readiness model 应学习 decoder 暴露出的 answer-ready 状态，而不是直接二分类 raw latent。
- CoLaR 说明 next compressed embedding / latent head 提供比最终 answer 更密集的监督，probabilistic head 可表达 uncertainty；因此 halt student 应有多头 proxy，而不是单个 stop logit。
- Latent Space Survey 说明 latent trajectory 的主要风险是 evaluability、controllability、interpretability；因此 decoder/scorer/stability 应作为可观测 teacher 和审计工具，而最终在线策略必须显式报告 decoder dependency。

训练图式：

```text
offline trace:
  latent prefix z_1...z_b
  + decoder probe / task scorer / prediction stability / future gain
  -> teacher targets

train:
  attention-based latent-student(z_1...z_b)
  -> decoder-proxy heads
  -> stability / completion heads
  -> continuation-risk / future-gain heads
  -> halt/readiness head

inference:
  attention-based latent-student(z_1...z_b)
  -> halt or continue
```

梯度路径必须是可收敛的 supervised distillation，而不是对不可微 scorer 或 sampled text 反传：

```text
L = L_halt
  + lambda_stop * L_decoder_stop_proxy
  + lambda_entropy * L_decoder_uncertainty_proxy
  + lambda_stability * L_prediction_change_proxy
  + lambda_correct * L_current_correct_proxy
  + lambda_future * L_future_gain
  + lambda_risk * L_continuation_risk
```

其中 teacher target 可以来自 decoder/scorer/gold/offline future blocks；online feature 不得直接读取 decoded text、prediction-stability、EOS/im_end probability 或 official correctness。

这一路线的含义是：我们训练的不是通用文本 decoder，而是一个轻量化、目标明确为早停的 `answer-readiness decoder proxy`。

---

## 3. 数据与评测口径

### 3.1 主数据

主数据使用 Cola 官方 8-task benchmark：

```text
lambada
mmlu
obqa
hellaswag
race
siqa
squad
story_cloze
```

每个任务必须保留：

```text
task_name
sample_id
prompt text
choices if any
ground truth / answer
official scorer input
official scorer output
```

官方数据准备与评测必须与 Cola release 对齐。任何自定义 judge 只能做辅助分析，不能替代 official scorer。

### 3.2 诊断数据

GSM8K 只作为 OOD/math diagnostic：

```text
允许用于检查数学 answer extraction、长推理输出、readiness 对复杂计算的响应。
不允许作为主线 benchmark。
不允许用 GSM8K 小样本结果支持或否定最终架构。
```

### 3.3 运行分组

每个实验至少区分：

```text
official Cola baseline
fixed-B rollout
oracle readiness halt
adaptive readiness halt
diagnostic ablations
```

所有 run name 必须包含：

```text
stage
task set
B_max
seed
readiness model version
whether LoRA/adapter is used
```

---

## 4. 完整链路

### 4.1 Stage A：official Cola baseline

目标：

```text
不训练，复现 official Cola 在 8-task benchmark 上的 baseline。
```

必须记录：

```text
per-task accuracy
task-average accuracy
generation config
max_new_tokens
temperature / top_p / top_k
guidance_scale
timestep_num
seed
runtime / latency
SwanLab run id
```

通过标准：

```text
8-task 数据全部可跑
acc_calc.py 可解析所有输出
本地 metrics.jsonl 与 SwanLab summary 一致
```

### 4.2 Stage B：block-wise rollout trace collection

目标：

```text
在 official Cola 生成过程中保存每个 block 的 latent、decoder probe 和 scorer 相关信号。
```

每个样本每个 block 保存：

```python
{
    "trace_version": "cola_block_trace_v1",
    "task": str,
    "sample_id": str | int,
    "seed": int,
    "per_sample_noise_seed": int | None,
    "rank": int,
    "world_size": int,
    "input_jsonl": str,
    "config_digest": str,
    "block_index": int,
    "block_number": int,
    "max_block_budget": int,
    "latent_batch_path": str,
    "latent_batch_sample_index": int,
    "latent_batch_block_index": int,
    "latent_block_shape": [16, 16],
    "latent_norm_mean": float,
    "latent_norm_std": float,
    "latent_delta_norm": float | None,
    "latent_cosine_to_prev": float | None,
    "denoise_drift_norm_mean": float | None,
    "decode_text_so_far": str,
    "decode_token_ids_so_far": list[int],
    "latest_block_token_ids": list[int],
    "eos_prob_max": float | None,
    "eos_prob_argmax_pos": int | None,
    "im_end_prob_max": float | None,
    "im_end_prob_argmax_pos": int | None,
    "stop_prob_max": float | None,
    "stop_prob_margin_vs_non_stop": float | None,
    "token_entropy_mean": float | None,
    "token_top_prob_mean": float | None,
    "answer_text_nonempty": bool,
    "answer_changed": bool,
    "same_answer_streak": int,
    "contains_eos": bool,
    "contains_im_end": bool,
    "contains_stop": bool,
    "first_stop_block_index": int | None,
    "official_score_if_decodable": float | None,
    "future_gain_label": float | None,
}
```

注意：

```text
official_score_if_decodable 只用于离线标签和评估。
训练 halt 推理特征时不得把 gold correctness 直接喂给模型。
```

### 4.3 Stage C：oracle readiness frontier

目标：

```text
构造每个样本的 earliest stable correct block，作为 oracle frontier 和弱监督标签。
```

默认 oracle 定义：

```text
b* = earliest b such that:
  b >= B_min
  official scorer marks decoded answer correct
  normalized answer 在未来 r 个 probe 中保持稳定
  stop-token 或 answer-ending evidence 不与继续生成强冲突
```

默认值：

```text
B_min = 1
stability_window r = 2
B_max 按官方生成预算和任务长度设定
```

如果不存在 `b*`：

```text
该样本 oracle stop = B_max
early blocks 全部标记为 not-ready
```

必须产出：

```text
earliest stable correct block histogram
oracle accuracy-cost frontier
fixed-B accuracy-cost frontier
oracle gap
per-task oracle advantage
answer stability by block
EOS/im_end vs correctness correlation
```

### 4.4 Stage D：LatentHaltStudent-v1 readiness / halt model

主模型不是 `raw latent -> binary stop`，也不是继续把 decoder-derived features 直接喂给 halt policy。主模型应升级为 `LatentHaltStudent-v1`：训练期从 decoder/scorer/stability teacher 学信号，推理期只读 latent/process trajectory。

`LatentHaltStudent-v1` 采用小型 attention-based latent trajectory model，而不是简单的 `slot MLP -> mean pool -> MLP concat`。

原因：

- Cola latent slot 已经是密集 VAE latent，`16 -> 128` 的强 MLP 可能扭曲 latent 几何；升维只应作为 attention working width，不应被理解成重新提取语义。
- block 内 EOS/im_end、fragment、answer-ending evidence 可能集中在少数 slot，mean pooling 容易抹平局部信号。
- process / budget features 的含义依赖 block 位置和 latent trajectory，最后简单 concat MLP 不足以表达这种条件交互。

在线输入信号分组：

```text
latent trajectory:
  raw latent slots z_{b,s}, each R^16
  slot position and block position
  current block tokens
  accumulated past block tokens
  norm / delta / cosine / drift
  denoising residual / velocity if available
  cross-seed variance if multiple rollouts exist

process / budget:
  block_idx
  max_block_budget
  remaining budget
  recent marginal latent change
  optional task embedding
```

默认模型规格：

```text
LatentHaltStudent-v1:
  d_model = 64
  attention_heads = 4
  dropout = 0.1

  slot_adapter:
    train-set standardization or LayerNorm over R^16
    Linear(16, d_model)
    + slot_pos_embedding
    + block_pos_embedding

  process_token:
    MLP(process / budget features -> d_model)
    appended to each block's slot tokens

  intra_block_encoder:
    1 lightweight self-attention layer over:
      [16 slot tokens + 1 process token]

  block_pooler:
    K=4 learned pooling queries cross-attend to intra-block tokens
    keep last_slot token explicitly
    output per block:
      [pool_1, pool_2, pool_3, pool_4, last_slot]

  inter_block_encoder:
    2-layer causal Transformer over all block summary tokens up to current block
    current block can attend only to previous/current blocks, never future blocks

  readout_queries:
    q_halt, q_risk, q_stability, q_decoder_proxy
    cross-attend to causal encoded block tokens

  heads:
    multi-task MLP heads from readout query states
```

默认超参只是首版，不是结论。必须做结构消融：

```text
d_model: 16 / 32 / 64 / 128
slot_adapter: identity+linear / linear only / 2-layer MLP
normalization: train-set mean/std vs LayerNorm
pooling: no pooling full-slot / mean+max / PMA K=1 / PMA K=4 + last_slot
process interaction: concat MLP / process token attention / FiLM gating
readout: last token / pooled state / task-specific readout queries
with / without block_idx and remaining_budget
```

训练期 teacher targets：

```text
decoder-stop proxy:
  EOS/im_end probability and margin
  token entropy
  token top probability
  stop-token presence

decoded-text / stability proxy:
  answer_found
  normalized answer presence
  answer_changed
  same_answer_streak
  generated length
  answer token logprob / margin if available

prediction-stability proxy:
  same answer across recent blocks
  same answer across stochastic rollouts
  low marginal token/logit change

scorer / verifier proxy:
  official correctness as offline label only
  oracle readiness frontier
  future gain
  continuation / incomplete-answer risk
  contentful answer-shape guard
  rule-based format consistency
  task-specific non-gold checks
```

输出：

```python
{
    "stop_proxy_logits": float,
    "entropy_proxy": float,
    "prediction_change_prob": float,
    "completion_risk_prob": float,
    "p_answerable": float,
    "p_stable": float,
    "p_correct_est": float,
    "p_continuation_risk": float,
    "uncertainty": float,
    "expected_future_gain": float,
    "p_stop": float,
}
```

训练标签：

```text
y_answerable(b) = answer_found_b
y_stable(b) = normalized answer 在未来 r 个 probes 中保持不变
y_correct(b) = official scorer correctness，仅训练用
y_continuation_risk(b) = 1 if 当前 task-scored answer 是 final answer 的 strict prefix / incomplete form
y_stop(b) = 1 if b >= b*, else 0
future_gain(b) = max_{k>b} reward(k) - reward(b)
```

训练时可以用 decoder/scorer/gold 构造 label；评估 `student-only` 时禁止把这些 teacher 字段作为决策输入。

推荐模型结构：

```text
slot-level local attention
+ attention pooling within each block
+ causal attention across block summaries
+ task-specific readout queries
+ multi-task distillation heads
```

设计依据：CODI 的 answer-state distillation 支持 feature-level teacher；CoLaR 的 latent head 支持 dense latent prediction 和 uncertainty；COCONUT 的 loss spike 警告要求逐步训练和密集日志；综述要求显式处理 evaluability / controllability / interpretability。

默认 reward：

```text
reward(b) =
  correctness_score_b
  + λ_stable * stability_score_b
  - λ_block * b / B_max
  - λ_probe * probe_cost_b
  - λ_wrong * early_wrong_stop_b
```

默认损失：

```text
L =
  BCE(p_answerable, y_answerable)
  + BCE(p_stable, y_stable)
  + BCE(p_correct_est, y_correct)
  + BCE(p_continuation_risk, y_continuation_risk)
  + BCE(p_stop, y_stop)
  + 0.2 * MSE(expected_future_gain, future_gain)
  + calibration_regularizer
```

推理停止规则：

```text
stop if:
  block_idx >= B_min
  p_stop >= τ_stop
  p_stable >= τ_stable
  p_continuation_risk <= τ_risk
  student completion/contentfulness proxy passes threshold
  uncertainty <= τ_uncertainty
  expected_future_gain <= τ_gain

force stop if block_idx == B_max
```

阈值只能在 validation set 上调，不能在 test set 上调。

### 4.5 Stage E：adaptive halt frontier

目标：

```text
比较 fixed-B、oracle halt、adaptive halt 的 accuracy-cost frontier。
```

必须报告：

```text
per-task accuracy
task-average accuracy
average blocks
median blocks
p90 blocks
latency
probe count
early-stop wrong rate
forced-stop rate
oracle gap
ECE / calibration
AUROC / AUPRC for readiness labels
future gain prediction error
```

通过标准不是“accuracy 更高”，而是：

```text
adaptive halt 在可接受 accuracy / stability 损失内显著减少 block budget
readiness score 校准良好
错误早停可解释且可被阈值控制
跨任务表现不是只在单一任务上成立
```

### 4.6 Stage F：optional LoRA / adapter / RL-style policy tuning

只有在 Stage A-E 完整链路完成后，才考虑这个阶段。

允许用途：

```text
改善 probe 可读性
改善 readiness feature separability
做轻量 task adapter
探索 uncertainty-aware sampling
优化 halt policy 的 accuracy-cost tradeoff
```

禁止用途：

```text
把 official benchmark accuracy 提升作为主线目标
用 LoRA 单独跑出一点涨跌就宣布架构有效或无效
覆盖 official Cola baseline 而不报告 frozen-base 对照
```

---

## 5. Baselines 与消融

### 5.1 必须包含的 baselines

```text
official Cola default generation
fixed B = 1 / 2 / 4 / 8 / 16 / B_max
EOS/im_end-only stop
length-only stop
oracle readiness halt
adaptive readiness halt
latent-only diagnostic halt
decoder-probe-only halt
latent-student halt
task-specific verifier proxy only
```

其中 `latent-only diagnostic halt` 不能退化为 naive raw latent 二分类；下一阶段必须升级为 `latent-student halt`：用 decoder probe / scorer / stability 构造 teacher labels，但在线输入限制为 latent trajectory、process features 和 learned latent verifier proxy。decoder-probe-only halt 只作为 teacher / upper-bound diagnostic，不是最终 agent 通信策略。

### 5.2 必须包含的消融

```text
teacher-probed upper bound vs student-only halt
latent/process-only student vs current riskcap04
with / without decoder-proxy distillation targets
with / without EOS/im_end teacher targets
with / without answer stability teacher targets
with / without latent trajectory features
with / without future gain head
with / without block index / budget features
single-task training vs multi-task training
cross-task train/test split
fixed threshold vs calibrated threshold
single seed rollout vs multi-seed self-consistency
frozen official Cola vs optional LoRA/adapter
```

### 5.3 关键图表

```text
accuracy vs average blocks
accuracy vs latency
per-task readiness frontier
earliest stable correct block histogram
EOS/im_end probability vs correctness
answer stability by block index
future gain prediction calibration
adaptive halt confusion matrix
early-stop wrong case table
```

---

## 6. 工程与日志规范

### 6.1 SwanLab 与本地日志

所有训练必须同时写：

```text
SwanLab cloud
outputs/.../metrics.jsonl
outputs/.../config.json
outputs/.../summary.json
outputs/.../checkpoints/best_checkpoint.pt
outputs/.../checkpoints/last_checkpoint.pt
```

无训练过程的 trace、frontier、eval、threshold sweep 和 aggregation 不上 SwanLab，只允许 `swanlab_mode=disabled`，只写：

```text
outputs/.../metrics.jsonl
outputs/.../config.json 或 summary 内 config
outputs/.../summary.json
必要的 jsonl/csv 诊断文件
```

训练日志必须包含完整动态，不只保存最后结果：

```text
train/loss
train/lr
train/grad_norm
valid/loss
valid/per_task_accuracy
valid/average_blocks
valid/early_stop_wrong_rate
valid/oracle_gap
valid/ece
valid/future_gain_error
```

保存策略：

```text
best_checkpoint.pt 按 validation 主指标保存
last_checkpoint.pt 只用于恢复训练
summary 必须同时记录 best step 和 last step
```

valid 频率：

```text
valid_interval <= 100 step
```

### 6.2 输出文件

推荐结构：

```text
outputs/
  cola_readiness/
    runs/
      <run_id>/
        config.json
        metrics.jsonl
        summary.json
        traces/
        probes/
        labels/
        checkpoints/
          best_checkpoint.pt
          last_checkpoint.pt
```

每个 trace 文件必须能反查：

```text
model paths
git commit / dirty state
dataset version
task name
seed
generation config
batch size
SwanLab run id
```

---

## 7. 主要风险与处理

### 7.1 EOS/im_end 与 correctness 混淆

症状：

```text
模型学会遇到 EOS/im_end 就停，但答案错误率高。
```

处理：

```text
EOS/im_end 只作为子特征。
stop policy 必须同时要求 stability、uncertainty、future_gain。
报告 EOS-only baseline。
```

### 7.2 raw latent 二分类退化

症状：

```text
训练集 halt 准，跨任务或跨 seed 失效。
```

处理：

```text
naive raw-latent-only 只作为 diagnostic。
下一阶段优先训练 latent-student halt：decoder probe / stability / scorer 只做 teacher label。
最终在线输入不得依赖 decoded text、prediction stability 或 decoder stop-token probe。
做 cross-task split、length-confound 和 decoder-dependency audit。
```

### 7.3 scorer 泄漏

症状：

```text
训练时把 official correctness 作为推理输入，导致线上不可用。
```

处理：

```text
official scorer 只用于 label 和评估。
当前 decoder-probed baseline 可使用 decoder logits / generated text 做诊断和 teacher。
最终 agent-latent 通信策略的在线推理特征必须来自 latent、process trajectory、learned latent verifier proxy。
```

### 7.4 halt 只学输出长度

症状：

```text
halt block 与 generated length 强相关，但与题目难度和 correctness 弱相关。
```

处理：

```text
加入同长度不同难度分析。
加入 per-task length-controlled eval。
报告 length-only baseline。
```

### 7.5 小实验误导

症状：

```text
小样本 work 后过早扩大结论，或小样本不 work 后否定架构。
```

处理：

```text
小样本只做 smoke test。
主结论必须来自 official 8-task 完整链路。
报告样本规模、任务覆盖、seed 数和预算。
```

### 7.6 Optional LoRA 过拟合

症状：

```text
LoRA 改变了 Cola latent 分布，readiness probe 在 frozen base 上失效。
```

处理：

```text
LoRA 必须有 frozen official Cola 对照。
报告 latent drift。
LoRA run 不得覆盖主线 frozen-base 结论。
```

---

## 8. 里程碑

### P0：decoder-probed readiness baseline（已完成前一阶段）

定位：

```text
decoder-dependent / decoder-probed readiness 架构
不是最终 agent-to-agent latent-only halt policy
但它是当前主线的前一阶段成果和 teacher / upper-bound baseline
```

保留价值：

```text
1. 证明 official Cola latent rollout 中存在可由 decoder 暴露的 answer-readiness 结构。
2. 产出 decoder probe、prediction-stability、scorer、future-gain 等 dense teacher targets。
3. 提供当前最强 safety/cost baseline：joint-readiness riskcap04。
4. 作为 LatentHaltStudent-v1 的训练监督、校准参照和失败诊断工具。
5. 作为后续 student-only 结果必须接近或解释差距的 upper-bound / safety frontier。
```

禁止事项：

```text
不得删除 decoder-probed 脚本、summary、checkpoint、SwanLab run 或样本诊断。
不得把它重新命名成失败路线。
不得用 student-only 新模型结果覆盖 riskcap04 baseline；必须并列表达。
```

### M0：official baseline

完成标准：

```text
8-task 数据全部齐备
official Cola baseline 可复现
acc_calc.py 评测通过
SwanLab + metrics.jsonl + summary.json 完整
```

### M1：block-wise trace

完成标准：

```text
每个任务至少完成固定 seed rollout
每个 block 保存 latent / logits / text / answer / stability 信号
trace 可复现 final output
trace schema 有版本号
```

### M2：oracle readiness frontier

完成标准：

```text
生成 earliest stable correct block 标签
报告 fixed-B vs oracle frontier
证明 oracle frontier 不是单纯长度规则
分任务展示哪些任务存在 halt 空间
```

### M3：readiness model

完成标准：

```text
多头 readiness model 训练完成
valid 间隔 <= 100 step
best_checkpoint.pt 按 validation 指标保存
校准指标和 future gain 指标可用
latent-only diagnostic 明确弱于或不替代多信号模型
```

### M4：adaptive halt

完成标准：

```text
adaptive halt 与 fixed-B / oracle / EOS-only / length-only 完整对比
average blocks 明显下降
correctness / stability 损失在预设阈值内
early-stop wrong cases 可解释
```

### M5：跨任务与鲁棒性

完成标准：

```text
single-task 与 multi-task 对比完成
cross-task train/test split 完成
不同 seed rollout 稳定性完成
不同 B_max / threshold sensitivity 完成
```

### M6：optional policy tuning

进入条件：

```text
M0-M5 完成
readiness signal 明确存在
adaptive halt 有可优化空间
```

允许内容：

```text
LoRA/adapter for probe separability
uncertainty-aware sampling
RL-style halt policy tuning
```

---

## 9. 历史诊断路线

旧的 Stage B/C 自建 encoder、latent decoder、小 Block-DiT prior 不删除，但降级为历史诊断路线。

保留价值：

```text
验证数据、decoder、judge、SwanLab、checkpoint、valid 频率等工程链路
构造可控小环境分析 loss spike、scheduled sampling、probe schema
为 readiness model 做单元测试
```

限制：

```text
不得作为主线结论来源
不得用 GSM8K 小样本结果判断 official Cola readiness 架构
不得把 overfit64 成功解释成泛化成功
不得把 train512/test256 失败解释成最终架构失败
```

如果继续运行历史诊断实验，必须在 run name 和文档中标注：

```text
diagnostic-only
not official Cola benchmark
not evidence for final architecture validity
```

---

## 10. 下一步执行顺序

当前推荐顺序：

```text
1. 已完成：更新文档与 AGENT.md，锁定 official Cola + readiness/halt 路线。
2. 已完成：核对 official 8-task 数据与 acc_calc.py 评测链路。
3. 已完成：实现 block-wise rollout trace 采集入口，并通过 1-sample smoke test。
4. 已完成：trace 中记录 per-block decoder probe、EOS/im_end/logits proxy、answer stability、raw latent shard。
5. 已完成：记录 official Cola frozen reference baseline；历史 SwanLab run `cca5o9r6t2sbchaye2stz` 是无训练记录污染，后续不要重复。
6. 已完成：对 official-protocol 8-task 跑 block-wise rollout trace；此类 trace 无训练过程，只允许 `swanlab_mode=disabled`，只保留本地 artifacts。
7. 已完成：构造 oracle readiness frontier。
8. 已完成：训练多信号 readiness / halt model，并保存 `best_checkpoint.pt` 与 `last_checkpoint.pt`。
9. 已完成：评估 adaptive halt frontier。
10. 已完成：EOS-only、feature/latent/process ablation、per-task threshold、leave-one-task-out cross-task ablation。
11. 已完成：样本级 halt 诊断，定位主要错误为 pre-stability strict prefix / incomplete answer halt。
12. 已完成：训练 continuation-risk gate，并评估 `min_blocks` 与 `first_saving` calibration。
13. 已完成：诊断 first-saving 下残余 SQuAD loss；根因是 quote-only / punctuation-only early halt，已加入非 gold contentful answer-shape guard。
14. 当前 1k official-protocol 最安全 learned policy：first-saving risk-gated halt + contentful prediction guard，aggregate `27.04%`、`2.019/4` blocks。该 `+0.013pp` 只能视为早停诊断现象，不是 benchmark accuracy 提升目标。
15. 已完成：full prepared split 统一 `batch_size=12` trace，merged artifact `ok=true`，`49019` samples / `196076` trace rows。
16. 已完成：full split scorer、frontier、readiness full model、continuation-risk model、adaptive halt、first-saving risk-gated halt。full split test 上 risk-gated halt + content guard 为 `20.392%` micro accuracy、`1.880/4` blocks，对照 fixed-final `20.351%`、`4/4` blocks；这是 block-budget 结果，不是 benchmark accuracy 提升主张。
17. 已完成：第二个 full split seed (`seed=20260525`, `per_sample_noise_seed=67`) 的 trace、merge、scorer、frontier、readiness/risk model、adaptive halt、risk-gated halt。seed67 risk-gated halt + content guard 为 `21.039%` micro accuracy、`1.873/4` blocks，对照 fixed-final `21.018%`、`4/4` blocks。
18. 已完成：第三个 full split seed (`seed=20260526`, `per_sample_noise_seed=68`) 的 trace、merge、scorer、frontier、readiness/risk model、adaptive halt、risk-gated halt。seed68 risk-gated halt + content guard 为 `20.798%` micro accuracy、`1.842/4` blocks，对照 fixed-final `20.860%`、`4/4` blocks。
19. 当前 cross-seed summary 在 `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_seed_20260524`。3 seeds 平均 risk-gated halt 为 `20.743%` micro accuracy、`1.865/4` blocks，对照 fixed-final `20.743%`、`4/4` blocks，约 `53.38%` block saving；`-0.0005pp` 平均 accuracy delta 不作为准确率变化主张。
20. 已完成第一轮 uncertainty/stability-aware calibration：`eval_cola_risk_gated_halt.py` 支持 `entropy_max/top_prob_min` 网格；full bs12 三 seed strict `0.0`-drop summary 在 `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_uncertainty_calibration_20260524`。结果：strict first-saving 为 `20.750%`、`2.291/4` blocks，strict min-blocks 为 `20.750%`、`2.281/4` blocks；均匹配 prediction-stability accuracy，但 entropy/top-prob guard 未被任何 seed 选中。
21. 已完成 risk-gated 样本级诊断与 single-choice stability guard：`analyze_cola_risk_gated_halt_decisions.py` 显示 `0.01` tolerance content guard 的剩余 loss 主要是 MMLU/RACE 单字母选项在 block 1 翻转；`--require-stable-single-choice` 后三 seed 均对齐 prediction-stability accuracy，summary 在 `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_choice_guard_20260524`。全量 choice guard 平均 `20.750%`、`2.210/4` blocks，observed loss 为 `0`；block1-only guard (`--stable-single-choice-max-block 1`) 平均 `20.757%`、`2.137/4` blocks，observed loss 仍为 `0`，是当前更好的 safety-cost 点。最新 calibration audit 已保存 valid/test threshold sweep，三 seed 均选择 `risk_threshold=0.01`。
22. 已完成 guard scope validation sweep：`eval_cola_risk_gated_halt.py` 支持 `--stable-single-choice-guard-scopes off,1,2,3,all`，把 single-choice guard 适用 block 纳入 valid sweep，而不是用 test 诊断手工选择。summary 在 `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_choice_scope_sweep_20260524`。`accuracy_drop_tolerance=0.01` 时 valid 三 seed 选择 `scope=off`，更省 block 但 test 仍有 `2/2/3` 个 observed loss；严格 `accuracy_drop_tolerance=0.0` 时 valid 三 seed 选择 `scope=1` 和 `risk_threshold=0.01`，得到 `20.757%`、`2.137/4` blocks，observed loss 为 `0`。因此 block1-only guard 是 strict safety objective 下的 valid-selected policy，而不是 test-tuned rule。
23. 已完成 prediction-change risk target：旧 `strict_prefix` risk 只能学习 prefix/incomplete 风险，不能学习 MMLU/RACE 单字母翻转。`train_cola_continuation_risk_model.py` 现在支持 `--target-mode prediction_change`，标签是当前 task-scored prediction 是否不同于同一 rollout 的 prediction-stability reference，仍然不使用 gold correctness。三 seed 正式训练 SwanLab runs 为 `hxz587d12946ramelk3xm`、`lyeomqgvkwvu6rijaz0lq`、`cik39cl5n5ux7x9r63q1l`，test AUROC 约 `0.997`。
24. 已完成 prediction-change risk halt eval：summary 在 `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_prediction_change_risk_20260524`。严格 `accuracy_drop_tolerance=0.0` + first-saving calibration 三 seed 均 valid-select `single_choice_guard_scope=off`、`risk_threshold=0.01`，平均 `20.750%`、`1.958/4` blocks，对 prediction-stability 的 observed loss 为 `0`。这优于旧 strict block1 guard 的 `2.137/4` blocks，说明可学习的 prediction-change risk 能替代硬 single-choice guard 的一部分。严格 min-blocks 虽可到 `1.904/4` blocks，但 seed67 有 sample-level loss/gain 抵消，保留为 diagnostic，不作为 safety-first policy。
25. 已完成 seed66/seed67/seed68 leave-one-task-out prediction-change risk transfer：readiness/risk 都用 `process_no_task` 在 7 个任务训练，阈值只在 7-task valid 校准，再评估 held-out task 的 `split=all`。per-seed summary 分别在 `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_prediction_change_risk_20260524`、`/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_prediction_change_risk_seed67_20260524`、`/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_prediction_change_risk_seed68_20260524`；cross-seed summary 在 `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_prediction_change_risk_cross_seed_20260524`。三 seed 平均 fixed-final weighted micro accuracy 为 `21.593% +/- 0.029%`、`4/4` blocks，prediction-stability 为 `21.596% +/- 0.029%`、`2.512/4` blocks，strict first-saving risk-gated halt 为 `21.596% +/- 0.029%`、`2.499 +/- 0.005/4` blocks。样本级诊断显示三 seed 合计对 fixed-final 和 prediction-stability 的 observed loss 都为 `0`，对 fixed-final 有 `5` 个 gain，对 prediction-stability 有 `1` 个 gain。这是当前 prepared full-split protocol 的 replicated transfer evidence，不是 benchmark accuracy 提升结论。
26. 已完成 joint readiness-threshold calibration：`eval_cola_risk_gated_halt.py` 现在支持 `--readiness-threshold-values` 和 `--require-zero-calibration-loss`。诊断显示，只用 aggregate accuracy 做 valid 校准会掩盖 loss/gain 抵消；seed68 的 joint min-blocks 无额外 guard 可到 `1.827/4` blocks，但 MMLU 有 `16` 个 observed loss，SQuAD 有自由文本 prefix loss。当前安全候选为 joint-readiness riskcap04：valid sweep readiness thresholds，`risk_threshold_end=0.4`，`min_blocks`，`require_zero_calibration_loss=true`，contentful prediction guard，block2 single-choice guard。cross-seed summary 在 `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_joint_readiness_riskcap04_cross_seed_20260524`。三 seed 平均 risk-gated weighted micro accuracy 为 `21.596% +/- 0.030%`、`2.118 +/- 0.010/4` blocks，对照 prediction-stability `21.596% +/- 0.029%`、`2.512/4` blocks；对 fixed-final 和 prediction-stability 的 observed loss 均为 `0`。这是当前更好的 cross-task cost/safety 点。
27. 已完成 SQuAD held-out answer-shape 诊断：加入 38 维 answer-shape risk features 后，若不使用 riskcap，cross-task 校准仍有 `15` 个 SQuAD prefix-fragment loss。`eval_cola_risk_gated_halt.py` 现在支持 `--require-fragment-complete-prediction`；v2 guard 对 non-stable decoded answer 的纯数字前缀、未完成缩写、尾部首字母和短连字符片段延迟 halt。seed68 leave-SQuAD-out 评估在 `/data1/luyifei/drla/outputs/cola_risk_gated_halt/cross_task_full_b64_bs12_prediction_change_shape_features_fragment_guard_v2_joint_readiness_min_blocks_choice2_zeroloss_seed20260526/leave_squad_out_eval_squad_all`，SwanLab run `639qsu4cpfseme3cpomf2`，结果为 `22.100%` SQuAD accuracy、`1.908/4` blocks，对 fixed-final 和 prediction-stability 的 observed loss 均为 `0`。这说明 fragment completeness 是有效 non-gold readiness 信号；但它仍是 held-out diagnostic，不能直接替代三 seed riskcap04 baseline。
28. 已完成 seed68 full 8-task fragment-completeness 验证。scope sweep/no-riskcap 在 `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_fragmentguardv2_scope_sweep_seed68_20260524`，结果 `21.590%` weighted micro accuracy、`1.895/4` blocks，但对 prediction-stability 有 `21` 个 observed loss，全部来自 MMLU 单选翻转。fixed block2 single-choice guard/no-riskcap 在 `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_fragmentguardv2_choice2_noriskcap_seed68_20260524`，结果 `21.559%`、`2.143/4` blocks，但 SQuAD 有 `34` 个 observed loss。结论：fragment-completeness 是有效 non-gold readiness 信号，但当前规则 + 现有 prediction-change risk 不能替代 riskcap04。
29. 已完成 seed66/67/68 full 8-task 38-feature answer-shape risk 扩展。`24` 个 leave-one-task-out risk checkpoint 均已训练并保存 `best_checkpoint.pt`，目录为 `/data1/luyifei/drla/outputs/cola_continuation_risk_model/cross_task_full_b64_bs12_prediction_change_shape_features_process_no_task_seed{20260524,20260525,20260526}`。cross-seed no-riskcap + fragmentguardv2 + block2 single-choice guard summary 在 `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_shape_features_fragmentguardv2_choice2_noriskcap_cross_seed_20260524`：per-task valid-selected policy 平均 `21.596%` weighted micro accuracy、`2.153/4` blocks，但对 prediction-stability 有 `19` 个 observed loss，同时有 `19` 个 gain，属于 loss/gain cancellation，不是安全策略。
30. 已完成 shape-risk no-riskcap loss 诊断：失败集中在自由文本 prefix/completion，而不是 MMLU/RACE 单选翻转。seed66 SQuAD 有 `7` loss；seed67 HellaSwag 有 `2` loss、SQuAD 有 `6` loss；seed68 HellaSwag 有 `1` loss、StoryCloze 有 `3` loss。代表样本包括 `AS-` vs `AS-205`、`Metro:` vs `Metro: All Change`、`$20` vs `$20 billion`、`Laverne &` vs `Laverne & Shirley`，以及 HellaSwag `[substeps]` / `[step]` 后的后续动作句缺失。post-hoc zero-loss frontier 平均 `2.191/4` blocks，仍弱于 riskcap04 的 `2.118/4` zero-loss baseline。
31. 已完成 fragment-completeness v3 复验：v3 guard 延迟 non-stable decoded answer 中的 replacement char、尾部 continuation marker、bare currency amount、numeric range prefix、connector ending、short unstable phrase 和 truncated hyphen token。no-riskcap v3 被 seed66/HellaSwag fail-fast 拒绝：`25` 个 loss，`1.669/4` blocks。恢复 `risk_threshold_end=0.4` 后，seed66/67/68 official8 全量评估为 `0` observed loss，cross-seed summary 在 `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_shape_features_fragmentguardv3_choice2_riskcap04_cross_seed_20260524`，平均 weighted micro accuracy `21.596%`、`2.245 +/- 0.003/4` blocks，saving vs prediction-stability `0.267`。这安全但比旧 joint-readiness riskcap04 的 `2.118/4` 更贵；v3 只作为 completion/stability 诊断信号，不替换当前 baseline。
32. 已完成当前 best result 的 decoder-dependency audit：`joint-readiness riskcap04` 不是 decoder-free / latent-only。它的 readiness/risk training labels 来自 decoder output、task-scored prediction、prediction-stability reference 和 official scorer；readiness/risk checkpoint feature fields 包含 EOS/im_end/stop probe、decoded answer dynamics、scored prediction dynamics、prediction length/shape 等 decoder/text-derived 特征；eval policy 还显式使用 `require_contentful_prediction=true`、block2 single-choice guard 和 `prediction_stability_reached(row)`。因此它只能称为 `decoder-probed / text-stability-supervised risk-gated halt baseline`，不能称为最终 agent-to-agent latent-only halt policy。
33. 已完成 Phase P1 `LatentHaltStudent-v1` same-split 首轮。三 seed test readiness AUROC 平均 `0.7317`，prediction-change AUROC 平均 `0.9953`；strict student-only halt 平均 `20.737%` micro accuracy、`2.324/4` blocks，比 prediction-stability 只多省 `0.193` block，并且 seed67 出现 `2` 个 observed loss。结论：latent/process-only student 能学习 teacher 信号，但不是 safety-complete baseline。
34. 已完成 Phase P1 seed66/67/68 leave-one-task-out eval。seed summaries 在 `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_seed{66,67,68}_20260525/summary.json`；cross-seed aggregate 在 `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_cross_seed_20260525/summary.json`。三 seed 聚合 student-only 为 `21.556%`、`2.048/4` blocks；prediction-stability 为 `21.596%`、`2.512/4` blocks。P1 多省 `0.463` block，但相对 prediction-stability 有 `58/147057 = 0.0394%` losses 和 `1612/147057 = 1.0962%` text mismatches。seed68 eval 是本地-only；seed66/67 eval run id 是 2026-05-25 logging policy 修复前的历史 no-training SwanLab 记录，后续不要重复。
35. P1 LOTO 解释必须同时报告 loss count、loss rate、mismatch rate 和跨 seed 重现性。`lambada` 为 `3/15459 = 0.0194%` loss、`mmlu` 为 `2/42126 = 0.0047%` loss、`race` 为 `2/14661 = 0.0136%` loss，不能因为有 1-3 个 loss 就直接等同系统性失败。真正稳定暴露 completion/stability 风险的是 `hellaswag`（`12/30126 = 0.0398%` loss、`361/30126 = 1.1983%` mismatch）和 `squad`（`39/31710 = 0.1230%` loss、`1221/31710 = 3.8505%` mismatch）。结论：P1 有 cross-task 信号，但当前版本不能替代 P0 joint-readiness riskcap04。
36. 已新增 `drla/scripts/aggregate_cola_latent_halt_student_loto.py`，用于汇总 P1 LOTO 的 per-seed、per-task、cross-seed 结果。该脚本无训练过程，默认本地 artifact；它输出 `loss_rate_vs_prediction_stability`、`prediction_mismatch_rate_vs_prediction_stability`、Wilson upper bound、risk bucket 和 recurrence，防止只看 SwanLab 曲线上单个 loss 点造成误判。
37. 已完成 Phase P1 seed68 pooling ablation `all_tokens`。Training root `/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d64_alltokens_20260525`，eval root `/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_alltokens_strict_textaudit_20260525`，comparison summary `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_ablation_seed68_pooling_20260525/summary.json`。all_tokens 相对 pma4_last 把 mismatch 从 `43` 降到 `17`，loss 仍为 `2/49019 = 0.0041%`，但 blocks 从 `2.325/4` 增到 `3.979/4`，比 prediction-stability 的 `2.512/4` 还贵。结论：去掉 PMA pooling 主要导致保守校准，不是当前更优 halt policy；瓶颈不只是 PMA 信息压缩。
38. 已完成 Phase P1 seed68 pooling ablation `d64_pma1`。Training root `/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d64_pma1_20260525`，eval root `/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_pma1_strict_textaudit_20260525`，aggregate summary `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_seed68_d64_pma1_20260525/summary.json`，architecture comparison `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_ablation_seed68_arch_20260525/summary.json`。pma1 得到 `21.592%` micro accuracy、`2.793/4` blocks、`5/49019 = 0.0102%` loss 和 `315/49019 = 0.6426%` mismatch；它比 `d64_pma4_last` 更贵、更不安全，也比 prediction-stability 更贵。结论：单 query PMA 压缩过强，当前 evidence 支持保留 PMA K=4 + explicit last-slot readout。
39. 已完成 Phase P1 seed68 pooling ablation `d64_mean_max`。Training root `/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d64_meanmax_20260525`，eval root `/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_meanmax_strict_textaudit_20260525`，aggregate summary `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_seed68_d64_meanmax_20260525/summary.json`，architecture comparison `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_ablation_seed68_arch_20260525/summary.json`。mean_max 得到 `21.594%` micro accuracy、`2.837/4` blocks、`4/49019 = 0.0082%` loss 和 `115/49019 = 0.2346%` mismatch；它比 `pma1/d128` mismatch 少，但仍比 `d64_pma4_last` 更贵且 loss 更多，也比 prediction-stability 更贵。结论：简单 mean/max pooling 不是缺失 readout。
40. 已完成 Phase P1 seed68 capacity ablation `d128_pma4_last`。Training root `/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d128_pma4_20260525`，eval root `/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d128_pma4_strict_textaudit_20260525`，architecture comparison `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_ablation_seed68_arch_20260525/summary.json`。d128 得到 `21.594%` micro accuracy、`2.432/4` blocks、`4/49019 = 0.0082%` loss 和 `285/49019 = 0.5814%` mismatch；它比 `d64_pma4_last` 更不安全，且只比 prediction-stability 省 `0.080` block。结论：单纯扩容不是当前主杠杆。
40a. 已完成 Phase P1 seed68 capacity ablation `d32_pma4_last`。Training root `/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d32_pma4_20260525`，eval root `/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d32_pma4_strict_textaudit_20260525`，aggregate summary `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_seed68_d32_pma4_20260525/summary.json`。d32 得到 `21.553%` micro accuracy、`2.126/4` blocks、`27/49019 = 0.0551%` loss 和 `263/49019 = 0.5365%` mismatch；SQuAD 贡献 `24` loss 和 `209` mismatch。结论：缩小 width 会更便宜但明显不安全，当前 reference 仍是 `d64_pma4_last process_token full`，不要继续盲扫 width。
41. 已完成 Phase P1 seed68 calibration ablation `zero-loss-zero-mismatch valid calibration`。Eval root `/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_zero_mismatch_calib_20260525`，aggregate summary `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_seed68_d64_pma4_zero_mismatch_calib_20260525/summary.json`，calibration comparison `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_calibration_seed68_20260525/summary.json`。它把 held-out loss 从 `2` 降到 `1`、mismatch 从 `43` 降到 `22`，但 blocks 从 `2.325/4` 增到 `3.872/4`，比 prediction-stability 还贵 `1.361` blocks。结论：text-stability 校准是有效 safety diagnostic，但不能直接作为默认策略。
41a. 已完成 Phase P1 seed68 calibration ablation `d64_pma4_last per_task calibration`。代码在 `drla/scripts/eval_cola_latent_halt_student.py` 增加 `--calibration-scope pooled|per_task`；`per_task` 要求每个 calibration task 都满足同一 accuracy/loss/mismatch 约束，防止 pooled valid 把单任务风险平均掉。Eval root `/data1/luyifei/drla/outputs/cola_latent_halt_student_eval/cross_task_full_b64_bs12_seed68_d64_pma4_per_task_calib_20260525`，aggregate summary `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_seed68_d64_pma4_per_task_calib_20260525/summary.json`。结果与 pooled d64 基本一致：`21.598%` micro accuracy、`2.325/4` blocks、`2/49019 = 0.0041%` loss、`43/49019 = 0.0877%` mismatch。结论：d64 的主要缺口不是 pooled 阈值泄漏，而是 latent/process-only student 对边界 completion/stability 的可校准读出不足。
41b. 已完成 Phase P1 seed68 readout-context diagnostic `last_process_query`，只跑最高风险的 leave-SQuAD-out 后被证据否决。代码在 `train/eval_cola_latent_halt_student.py` 增加 `--readout-context-mode none|last_process_query`；该模式给 final readout queries 加上当前最后可见 block process features 的零初始化条件偏置。Training root `/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_readoutctx_20260525/leave_squad_out`，SwanLab run `j4wmhdpn4v6qmei2lptja`，best step `1500`，best valid readiness AUROC `0.7487`。Pooled 与 per_task eval 都得到 SQuAD `21.977%` accuracy、`1.755/4` blocks、`6/10570 = 0.0568%` loss、`223/10570 = 2.1097%` mismatch；baseline d64 SQuAD 是 `22.034%`、`3.999/4` blocks、`0` loss、`5` mismatch。失败样本仍是 `October 16,`、`15`、`1568-`、`194` 这类 prefix/completion boundary。结论：浅层 process-conditioned query 不会自然学会 answer-enough，不扩展到 8-task LOTO。
41c. 已完成 Phase P1 seed68 explicit completion-boundary diagnostic `d64_pma4_last + completion_risk`，只先跑最高风险的 leave-SQuAD-out。代码在 `train/eval_cola_latent_halt_student.py` 增加 `--use-completion-risk` 和 `--completion-risk-thresholds`；completion-risk teacher target 定义为当前 `scored_prediction` 为空，或是 prediction-stability/final reference 的 strict prefix，仍然只作为离线 decoder teacher label，在线输入不含 decoded text。Training root `/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_completionrisk_20260525/leave_squad_out`，SwanLab run `gsarp1b2dzfmhh6nluhws`，best step `1750`，best valid mean AUROC(readiness/prediction-change/completion-risk) `0.9140`；held-in test completion-risk AUROC `0.9967`，说明该局部边界标签可学习。
41d. completion-risk SQuAD eval 结论：default strict root `/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_completionrisk_strict_textaudit_20260525/leave_squad_out_eval_squad_all` 选择 `readiness=0.9`、`risk=0.1`、`completion_risk=0.2`，SQuAD 为 `22.034%` accuracy、`3.9996/4` blocks、`0/10570` loss、`2/10570` mismatch；相比 baseline d64 SQuAD 的 `5` mismatch 有改善，但几乎没有 compute saving。refined threshold root `/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_completionrisk_refined_textaudit_20260525/leave_squad_out_eval_squad_all` 仍是 `0` loss，但 mismatch 增到 `6`、blocks `3.9986/4`。risk-only probe root `/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_completionrisk_riskonly_probe_20260525/leave_squad_out_eval_squad_all` 证明只靠 `prediction_change + completion_risk + contentful` 不安全：valid `2/4008` loss，SQuAD `10/10570` loss 和 `354/10570` mismatch，虽然 blocks 降到 `1.686/4`。结论：completion-risk 是有价值的辅助蒸馏信号，但“只加单个 completion head”不是主线解法；下一步应蒸馏 P0 teacher halt policy 或改更强 latent-process interaction / learned stability-aware readout。
41e. 已完成 Phase P1 seed68 P0-teacher distillation SQuAD 诊断。`train_cola_latent_halt_student.py` 支持 `--readiness-target-mode p0_teacher_halt|p0_teacher_action` 和 `--teacher-decisions-jsonl`。teacher decisions 来自 P0 riskcap04：`/data1/luyifei/drla/outputs/cola_risk_gated_halt_teacher/cross_task_full_b64_bs12_joint_readiness_riskcap04_seed20260526/train_tasks_for_leave_squad_out_exact/analysis/risk_gated_decisions.jsonl`，7-task teacher 平均 halt 为 `2.111/4` blocks 且 zero loss。`p0_teacher_halt + completion_risk` refined SQuAD eval 为 `22.034%`、`3.529/4` blocks、`0/10570` loss、`74` mismatch。当前最强是 `p0_teacher_action + completion_risk`：training root `/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_p0teach_action_completionrisk_20260525/leave_squad_out`，SwanLab run `wkirim527d08abt606u8e`，best step `2420`，valid mean AUROC `0.9897`；refined SQuAD eval root `/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_p0teach_action_completionrisk_refined_textaudit_20260525/leave_squad_out_eval_squad_all`，结果为 `22.034%` accuracy、`2.049/4` blocks、`0/10570` correctness loss、`841/10570` text mismatch。所有 mismatch 都在 fixed-final 本来也错误的样本上。结论：stop-action 蒸馏是当前最强 P1 方向，但还需要 latent-level answer identity / stability objective。
42. 已完成 Phase P1 seed68 teacher-objective ablation `stabilityw2`。Training root `/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_stabilityw2_20260525`，eval root `/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_stabilityw2_strict_textaudit_20260525`，aggregate summary `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_seed68_d64_pma4_stabilityw2_20260525/summary.json`，objective comparison `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_stability_objective_seed68_20260525/summary.json`。它将 `prediction_change_loss_weight` 提到 `2.0` 并按 `readiness_prediction_change_mean_auroc` 选 best checkpoint；结果为 `21.590%` micro accuracy、`2.860/4` blocks、`6/49019 = 0.0122%` loss 和 `62/49019 = 0.1265%` mismatch，比 `d64_pma4_last` baseline 更贵且更不安全。严格 zero-mismatch calibration 可压到 `2` loss / `32` mismatch，但 blocks 升到 `3.980/4`。结论：单纯加大 prediction-change teacher loss 不是当前主线。
43. 已完成 Phase P1 seed68 process-feature ablation `no_block_budget`。代码支持 `--process-feature-mode full|no_block_budget`；`no_block_budget` 移除 block/budget 位置特征，保留 latent norm/delta/cosine/drift。Training root `/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_nobudget_20260525`，eval root `/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_nobudget_strict_textaudit_20260525`，aggregate summary `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_seed68_d64_pma4_nobudget_20260525/summary.json`，process-feature comparison `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_process_feature_ablation_seed68_20260525/summary.json`。结果为 `21.586%` micro accuracy、`1.831/4` blocks、`8/49019 = 0.0163%` loss 和 `389/49019 = 0.7936%` mismatch；zero-mismatch calibration 后为 `3` loss、`124` mismatch、`3.495/4` blocks。结论：block/budget features 当前是重要 calibration anchor，不能硬删；但它们也可能是位置捷径，后续应以更强 latent-process interaction 替代。
44. 已完成 Phase P1 seed68 process-interaction ablation `film`。代码支持 `--process-interaction-mode process_token|film`；FiLM 用 process features 调制 slot tokens，而不是追加 process token。Training root `/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_film_20260525`，primary aggregate `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_seed68_d64_pma4_film_20260525/summary.json`，zero-mismatch aggregate `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_seed68_d64_pma4_film_zero_mismatch_calib_20260525/summary.json`，process-interaction comparison `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_process_interaction_ablation_seed68_20260525/summary.json`。Primary 为 `21.586%` micro accuracy、`2.260/4` blocks、`8/49019 = 0.0163%` loss、`190/49019 = 0.3876%` mismatch；zero-mismatch calibration 后仍有 `8` loss、`150` mismatch，且 blocks 升至 `3.421/4`。SQuAD 保留 `7` loss 和 `134` mismatch。结论：简单 FiLM gating 不是当前主线赢家，seed68 reference 仍是 `d64_pma4_last process_token full`。
45. 已修正 P1 训练指标解释：`valid/readiness_accuracy` 是 `0.5` 固定阈值 accuracy，在 readiness 正例率约 `17-22%` 且使用 weighted BCE 时容易被多数类/阈值漂移误导。后续读取 SwanLab 时优先看 `readiness_auroc/auprc/brier`、`readiness_balanced_accuracy`、`readiness_predicted_positive_rate`、`readiness_accuracy_lift_vs_majority`，最终仍以 held-out halt eval 为准。
46. 下一步优先级：不要继续把主要精力放在手写 decoder/text guard 上，也不要直接移除 `risk_threshold_end=0.4`。应以 `p0_teacher_action + completion_risk` 作为 P1 正向分支，扩展到 official8 all LOTO / 多 seed，并显式加入 answer identity / sequence-level stability objective；目标是保留 SQuAD `0` correctness loss 与低 block 成本，同时减少 text/latent answer mismatch。并行可以测更强 latent-process interaction 或 transferable task conditioning，但不要直接采用 `all_tokens`、`pma1`、`mean_max`、硬 zero-mismatch calibration、单纯扩容或缩小 width、单纯加大 prediction-change loss 权重、硬删除 block/budget features、简单 FiLM gating、浅层 readout process query、只加单个 completion-risk head，或只改 pooled/per-task 阈值校准。
46a. 2026-05-27 已完成文献复盘后的一次 P1 action->halt gate 负结果：`decomposed_expected_utility` 五头目标预测 correctness rescue、mismatch rescue、introduced loss、introduced mismatch 和 extra block cost。Full official8 LOTO 已补齐，正式训练均使用 SwanLab cloud；calibration summary 为 `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_decomposed_expectedutility_calibrated_official8_seed20260527/summary.json`。同口径对比 old v2 backfilled source-valid cost-limited `10` losses / `465` mismatches / `1.859/4` blocks：new source-valid cost-limited 为 `11` / `493` / `1.847/4`，constrained cost-limited 为 `11` / `465` / `1.857/4`，task-robust cost-limited 为 `11` / `482` / `1.848/4`。结论：四任务 task-robust 改善没有扩展到 official8，简单多头 expected utility 和后处理 selector 不应作为主线。
46b. 已新增 formal risk-control local audit：`drla/scripts/analyze_latent_halt_risk_control.py`。它读取现有 P1 LOTO threshold sweeps，在 valid 上用 Wilson 95% upper bound 约束 `losses_vs_prediction_stability` / mismatch，再应用到 held-out eval。Cross-seed summary 在 `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_latent_halt_wilson_risk_control_cross_seed_20260527/summary.json`。结果：经验 zero-loss valid 为 `58` losses / `1612` mismatches / `2.048/4` blocks；Wilson target `<=0.00025` 与 `<=0.0005` 完全无法覆盖 24 folds，`<=0.001` 只覆盖 `18/24`，`<=0.002` 虽覆盖全部但 held-out 为 `104-111` losses、`2834-2983` mismatches、`1.825-1.834/4` blocks。结论：现有 P1 score 和 valid calibration size 不足以支撑形式化低风险保证；下一步必须改 latent/process readout 或 calibration 数据协议，而不是继续调阈值。
47. 如果更改 batch size、block count、decoding 或任务集合，必须重新跑 trace/frontier/train/eval，而不是复用旧结论。在完整链路证据进一步充分后，再考虑 LoRA/adapter/RL-style policy tuning。
```

每一步都必须产出可复查 artifact。没有 trace、label、metric、checkpoint 的实验不进入主结论。
