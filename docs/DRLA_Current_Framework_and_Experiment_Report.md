# DRLA 当前框架与实验进展报告

更新时间：2026-05-25

## 1. 当前研究主线

当前主线已经从“自建小 prior / GSM8K MVP / 用 LoRA 提 Cola 精度”转为：

```text
official Cola VAE + official Cola DiT 作为冻结或半冻结 latent substrate
+ block-wise latent rollout
+ decoder-side probe
+ answer stability / task scorer / future gain
+ block-level answer-readiness / halt 判别器
```

核心目标不是提升 released Cola 在官方 benchmark 上的 accuracy，而是研究：

```text
累计到当前 block 的 latent memory 是否已经足以生成正确且稳定的答案？
继续生成更多 latent blocks 是否仍有实质收益？
```

因此，当前实验关注的是 answer-readiness / halt，而不是训练一个新的 Cola、重训大模型，或用 DiT LoRA 刷官方 benchmark 分数。

## 2. 当前框架

整体推理与判别链路如下：

```text
prompt / task input
  -> official Cola VAE encode prefix latent

for block = 1 ... B_max:
    official Cola DiT samples current latent block
    append current block to accumulated latent memory

    decoder probe:
      decode accumulated blocks
      collect EOS / im_end / stop-token / answer text / logit-like process signals

    readiness / halt model:
      estimate answerability, stability, uncertainty, continuation risk, future gain
      decide whether to halt

final accumulated blocks
  -> official Cola decoder
  -> answer text
  -> official task scorer
```

当前 halt 判别器不被定义为 raw latent 二分类器。它是一个多信号 readiness model，输入和辅助信号包括：

- latent / process features
- decoder probe 输出
- EOS / im_end / stop token 信号
- task-scored prediction
- answer stability / prediction stability
- continuation risk
- future gain / oracle frontier
- contentful prediction guard
- single-choice stability guard
- fragment / completion diagnostic guard

其中 `prediction_stability` 是当前非常强的 non-gold baseline：如果 task-scored prediction 连续稳定出现，就说明当前答案很可能已经足够。

## 3. 最终部署目标：agent 间 latent 通信

必须明确：当前项目最终目标不是“让单个模型少 decode 几次”，而是实现 agent 之间通过 latent space 直接通信。也就是说，一个 agent 产生 latent blocks，另一个 agent 或后续模块直接消费这些 latent，而不是每一步都 decode 成文本。

因此，早停的最终意义是：

```text
在真正推理和 agent-to-agent latent 通信时，
不依赖 decoder 相关信号，
直接从累计 latent / process trajectory 中判断：
当前 latent message 是否已经足够稳定、足够完整、足以被下游 agent 使用。
```

这对当前实验解释有一个重要限制：

- 当前 decoder probe、EOS/im_end、answer text、task scorer、prediction-stability 等信号，主要是训练期监督、离线诊断和 oracle 构造工具。
- 它们可以帮助我们理解 latent block 中什么时候出现 answer-readiness。
- 它们可以作为 teacher / probe label，训练 latent-only 或 latent-primary 的 halt model。
- 但它们不能直接成为最终 agent-latent 通信场景下的在线输入，因为 agent 之间没有 decoder，也不应该为了判断是否继续通信而把 latent message 解码成文本。

因此，当前阶段的 decoder-side 信号应被理解为“可观测代理变量”，不是最终部署依赖。后续模型路线应逐步从：

```text
decoder-probed readiness / text-stability supervised halt
```

迁移到：

```text
latent trajectory / process-feature / downstream latent-consumption readiness halt
```

最终目标是让 halt 判别器在不解码的情况下判断 latent message 是否已经 answer-ready / communication-ready。

### 3.1 模型修订：decoder-as-teacher, latent-student halt

最新设计不是丢弃 decoder 信息。decoder 本身已经是 trained latent decoder，它包含大量关于 latent 是否可被解码成完整答案、是否接近 EOS/im_end、答案是否稳定的知识。真正需要避免的是在最终 agent-to-agent latent 通信推理时直接读取 decoder 输出。

因此下一步模型应按如下方式修改：

```text
训练期：
  decoder / scorer / prediction-stability / future blocks
  -> 产生 per-block teacher targets

推理期：
  latent prefix z_1...z_b + process/budget features
  -> attention-based latent-student readiness model
  -> halt / continue
```

这等价于训练一个轻量化、目标明确为早停的 `answer-readiness decoder proxy`，而不是训练通用文本 decoder。

当前替换后的具体架构是 `LatentHaltStudent-v1`，不是简单 MLP：

```text
slot_adapter:
  standardize or LayerNorm over each R^16 latent slot
  Linear(16 -> d_model), default d_model=64
  + slot position embedding
  + block position embedding

process_token:
  MLP(block_idx, remaining_budget, norm, delta, cosine, drift -> d_model)
  appended to each block as an extra token

intra_block_encoder:
  one lightweight self-attention layer over 16 slot tokens + process token

block_pooler:
  K=4 learned pooling queries cross-attend to intra-block tokens
  keep last_slot token explicitly
  per block output = [pool_1, pool_2, pool_3, pool_4, last_slot]

inter_block_encoder:
  2-layer causal Transformer over block summary tokens up to current block

readout_queries:
  q_halt, q_risk, q_stability, q_decoder_proxy
  cross-attend to causal block states
```

设计理由：

- 不默认 `16 -> 128` 强 MLP；latent 本身已经信息密集，升维只作为 attention working width，默认 `d_model=64`，`128` 只能作为消融。
- 不默认 mean pooling；block 内局部 ending / fragment / stop evidence 可能集中在少数 slot，所以采用 attention pooling + last slot 保留。
- 不把 process features 最后 concat 一下；process / budget 与 latent trajectory 的含义强交互，所以作为 process token 进入 block-level attention。

建议的 student 多头输出：

```text
decoder-stop proxy:
  EOS/im_end/stop probability, entropy, top probability

stability / completion proxy:
  answer will change, same-answer streak proxy, fragment / completion risk

scorer proxy:
  current-correctness estimate, future gain, continuation risk

halt head:
  calibrated readiness / answer-enough probability
```

关键约束：

- decoder probe、task scorer、prediction-stability 和 future-gain 可以作为 teacher label。
- online input 必须逐步限制为 latent trajectory、block/process dynamics、learned latent verifier proxy。
- `student-only` eval 时不得读取 `scored_prediction`、decoded answer text、EOS/im_end probability、prediction-stability 或 official correctness 作为决策输入。
- 当前 riskcap04 继续作为 teacher-probed upper bound / safety baseline，而不是最终部署策略。

注意：这不是废弃前一阶段 decoder-dependent 架构。相反，前一阶段 decoder-probed readiness baseline 是当前路线成立的关键证据：它证明 latent rollout 中存在可被 decoder probe 暴露的 readiness 结构，并提供 teacher labels、upper-bound frontier、safety calibration 和样本级失败诊断。`LatentHaltStudent-v1` 是在这个 baseline 之上做蒸馏和在线输入收缩，而不是替换掉或删除它。

文献依据：

- COCONUT 支持 continuous thought 可 probe、可反传，但警告 latent 训练需要 curriculum，突然增加 latent thoughts 会 loss spike。
- CODI 支持 feature-level self-distillation，尤其是从 answer-generating hidden state 向 continuous student 转移推理信息。
- CoLaR 支持 dense latent prediction / probabilistic latent head / uncertainty，而不是只用最终答案监督。
- Set Transformer / Perceiver 系列支持用 learned queries / cross-attention 从 dense inputs 中读出少量任务相关 latent，而不是手写 mean pooling。
- Latent Space Survey 强调 latent trajectory 的 evaluability、controllability、interpretability 风险，因此 teacher-probed 训练和 student-only 审计必须分开报告。

## 4. 早停评估方式

当前实验包含两个阶段，必须区分：

### 4.1 Trace 收集

trace 收集阶段会先把所有 block 都跑出来，例如 block 1 到 block 4。这样做是为了离线构造：

- oracle readiness frontier
- future gain label
- per-block decoder probe
- answer text dynamics
- risk/readiness 训练样本
- loss/gain 复盘样本

这个阶段本身不是在线早停，也不声称节省真实 wall-clock 推理时间。

### 4.2 Halt 策略评估

halt 评估是在完整 trace 上离线重放在线策略：

```text
block 1 输出 -> 判别器判断 -> 可以停则选 block 1
否则
block 2 输出 -> 判别器判断 -> 可以停则选 block 2
否则
block 3 输出 -> 判别器判断 -> 可以停则选 block 3
否则
block 4 / final
```

判别器在 block k 做决策时，只允许使用 block 1 到 block k 的信息。未来 block 只能用于离线 label、oracle、诊断和结果比较，不能作为当前 block 的决策输入。

因此，`2.118/4 blocks` 这类指标表示：如果把该策略部署进真实 block-wise 推理循环，平均会在约 2.118 个 block 停下。它不是“看完所有 block 后倒推选择”。

### 4.3 当前评估与最终部署的差异

当前评估中，部分强 baseline 会使用 decoder-derived 信号，例如 prediction-stability、task-scored prediction 和 fragment-completeness guard。这些结果证明“latent rollout 中存在可被 decoder probe 暴露的 readiness 结构”，但还不等于最终 agent-to-agent latent communication 已经可以完全 latent-only 早停。

因此，当前结果应按两层解读：

1. **科学诊断层**：decoder probe 证明某些 block 已经包含足够答案信息。
2. **部署目标层**：下一步需要训练不依赖 decoder 的 latent-primary readiness model，让它从 latent/process trajectory 中预测这些 probe 暴露出的 readiness 状态。

最终验收不能停留在 decoder-probed halt；必须验证 latent-only 或 latent-primary halt 在不解码时仍能接近当前 riskcap04 的 safety/cost frontier。

## 5. 数据与实验协议

主评测口径固定为 Cola 官方 8 个 benchmark：

```text
lambada, mmlu, obqa, hellaswag, race, siqa, squad, story_cloze
```

当前 full prepared split 协议：

- `batch_size=12`
- seeds：`66`, `67`, `68`
- 每个 seed：`49019` samples
- 每个 seed：`196076` trace rows
- 统一使用 official Cola task-specific scorer
- GSM8K 只保留为 OOD/math diagnostic，不作为主线结论

batch size 是协议变量，不只是速度参数。此前诊断显示 official Cola batched inference 并非完全 batch-invariant，所以主结果不得混用不同 batch size。

正式训练和评估规范：

- 所有正式训练必须上 SwanLab cloud
- 所有训练必须写本地 `metrics.jsonl`
- valid 间隔不得超过 `100 step`
- 必须保存 `best_checkpoint.pt`
- 不允许只报告 last checkpoint
- 每个主结论必须有 trace、label、metric、checkpoint 或 summary artifact 支撑

## 6. 当前最好 baseline

当前最好的 cross-task safety/cost baseline 是：

```text
joint-readiness riskcap04
```

核心配置：

- readiness thresholds 在 valid split 上 sweep
- `risk_threshold_end=0.4`
- `risk_threshold_selection_mode=min_blocks`
- `require_zero_calibration_loss=true`
- contentful prediction guard
- block2 single-choice guard

三 seed cross-seed 结果：

| 指标 | 数值 |
|---|---:|
| weighted micro accuracy | `21.596% +/- 0.030%` |
| average blocks | `2.118 +/- 0.010 / 4` |
| prediction-stability average blocks | `2.512 / 4` |
| saving vs prediction-stability | `0.394 block/sample` |
| saving vs fixed-final | `1.882 blocks/sample` |
| observed loss vs fixed-final | `0` |
| observed loss vs prediction-stability | `0` |

Artifact：

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_joint_readiness_riskcap04_cross_seed_20260524/summary.json
```

结论：当前已经有一个三 seed、official8、leave-one-task-out transfer setting 下的零 observed loss 早停策略，可以比 prediction-stability 进一步节省 block budget。

### 6.0 阶段定位：前一阶段 decoder-probed readiness baseline

这一阶段不应被扔掉。它的定位是：

```text
Phase P0:
  decoder-dependent / decoder-probed readiness baseline

Phase P1:
  decoder-as-teacher, LatentHaltStudent-v1
```

P0 的价值：

- 证明 readiness 信号不是凭空假设，而是在 official Cola block-wise rollout 中可被 decoder/scorer/stability 观测到。
- 给 P1 提供 dense teacher targets：EOS/im_end/stop proxy、entropy/top-prob、prediction-change、completion risk、official correctness、future gain、oracle frontier。
- 给 P1 提供必须对照的 safety/cost frontier：当前 joint-readiness riskcap04 是 `2.118/4` blocks、zero observed loss 的 baseline。
- 给 P1 提供失败样本和诊断集合，例如 SQuAD prefix、HellaSwag continuation、MMLU single-choice flip。

因此后续报告必须并列：

```text
teacher-probed riskcap04 baseline
student-only LatentHaltStudent-v1 result
gap analysis and decoder-dependency reduction
```

最新 P1 seed68 消融结论分成两类。第一类是负结果：`all_tokens` 和 zero-mismatch calibration 会退回接近 final block；`pma1`、`mean_max`、`d128_pma4_last` 更贵或更不安全；`d32_pma4_last` 虽更便宜到 `2.126/4` blocks，但 loss 升到 `27/49019`、mismatch 升到 `263/49019`，SQuAD 是主要风险源；`stabilityw2`、`no_block_budget`、简单 `film`、`per_task` calibration、`last_process_query` 和单个 `completion_risk` head 都没有成为默认解。第二类是正结果：`p0_teacher_action + completion_risk` 把 P0 riskcap04 的 exact chosen halt block 蒸馏成 stop-action target，在 leave-SQuAD-out 上达到 `22.034%` accuracy、`2.049/4` blocks、`0/10570` correctness loss，明显优于 prediction-stability 的 `2.509/4` blocks。它的问题是 `841/10570` text mismatch，但这些 mismatch 全部发生在 fixed-final 本来也错误的样本上。结论：P0 action distillation 是当前最有前景的 P1 方向；下一步不应再靠调阈值或浅层结构小改，而应扩展到全 8-task / 多 seed，并加入 latent-level answer identity 或 sequence-level stability objective。

### 6.1 Decoder 依赖审查

需要明确：当前最好结果 `joint-readiness riskcap04` 不是 decoder-free / latent-only 策略。它在训练标签、模型输入和评估 guard 三个层面都使用了 decoder-derived 或 text-derived 信号。

1. **训练标签依赖 decoder / scorer**
   readiness frontier、oracle ready、future gain、prediction-change target 都来自完整 block trace 的 decoder 输出、task-scored prediction、prediction-stability reference 或 official scorer。这是训练期 teacher / offline label，不是最终 latent communication 的可部署输入。

2. **readiness model 输入包含 decoder-derived features**
   当前 best readiness checkpoint 的 `feature_fields` 包含：

   ```text
   token_entropy_mean, token_top_prob_mean,
   eos_prob_max, im_end_prob_max, stop_prob_max,
   stop_prob_margin_vs_non_stop,
   answer_text_nonempty, answer_changed,
   same_text_streak,
   scored_prediction_nonempty, scored_prediction_changed,
   scored_prediction_same_streak,
   processed_generation_changed, processed_generation_same_streak,
   already_stopped_before_block,
   contains_eos, contains_im_end, contains_stop
   ```

   其中大部分来自 decoder probe、decoded text、stop-token probe 或 task-scored prediction dynamics。虽然 readiness model 也使用 latent features，例如 latent norm、latent delta、latent cosine 等，但当前 best policy 不是纯 latent 输入。

3. **continuation-risk model 输入也包含 decoder-derived features**
   当前 `prediction_change` risk checkpoint 的 `feature_fields` 包含 EOS/im_end/stop、answer text dynamics、scored prediction dynamics、processed generation dynamics、prediction length / word count / punctuation shape、decode length 等。这些同样依赖 decoder-side observation。

4. **halt policy 的 guard 直接使用 decoded prediction**
   当前 best summary 的 protocol 显式打开：

   ```text
   require_contentful_prediction=true
   require_stable_single_choice=true
   stable_single_choice_max_block=2
   ```

   在 `eval_cola_risk_gated_halt.py` 中，这些 guard 直接读取 `scored_prediction`，并调用 `prediction_stability_reached(row)`。因此它们是 text-derived online guard，不是 latent-only guard。

因此，当前 best result 的正确解释是：

```text
decoder-probed / text-stability-supervised risk-gated halt baseline
```

而不是：

```text
final agent-to-agent latent-only halt policy
```

它证明了 official Cola block-wise latent rollout 中存在可被 decoder probe 暴露的 readiness 结构，也证明这些结构可以通过 risk/readiness calibration 转化为 block-saving 策略。但它还没有证明最终 agent 间 latent 通信场景下，不使用 decoder 信号也能达到同样 safety/cost frontier。

## 7. 关键实验进展

### 7.1 Prediction-change continuation risk

`prediction_change` 是一个 non-gold continuation-risk target。它判断当前 task-scored prediction 是否会相对 rollout prediction-stability reference 发生改变。

它主要用于补齐旧 `strict_prefix` target 的盲区。旧 target 能学习 prefix / incomplete-answer 风险，但不擅长 MMLU / RACE 这类单选答案翻转。

三 seed leave-one-task-out transfer 结果：

| 指标 | 数值 |
|---|---:|
| weighted micro accuracy | `21.596% +/- 0.029%` |
| average blocks | `2.499 +/- 0.005 / 4` |
| prediction-stability average blocks | `2.512 / 4` |
| observed loss vs fixed-final | `0` |
| observed loss vs prediction-stability | `0` |

Artifact：

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_prediction_change_risk_cross_seed_20260524/summary.json
```

判断：这是有效 transfer evidence，但单独省 block 很少，只比 prediction-stability 省约 `0.013 block/sample`。它证明 learned prediction-change risk 有用，但不能单独作为最终 cost/safety baseline。

### 7.2 Joint-readiness riskcap04

`eval_cola_risk_gated_halt.py` 已支持 joint readiness-threshold calibration：

- `--readiness-threshold-values`
- `--require-zero-calibration-loss`

这个改动很重要，因为只用 aggregate accuracy 做 valid calibration 会掩盖 loss/gain cancellation。此前 seed68 joint min-blocks 无额外 safety cap 可到 `1.827/4` blocks，但 MMLU 有 `16` 个 observed loss，SQuAD 也有 prefix loss。

加入 riskcap04 后，三 seed official8 达到当前最好安全成本点：

```text
21.596% accuracy
2.118 / 4 blocks
0 observed loss
```

判断：这是当前主 baseline。

### 7.3 Answer-shape risk 与 fragment-completeness v2

我们训练了 38-feature answer-shape risk model，seed66/67/68 共 `24` 个 leave-one-task-out risk checkpoint，均保存 `best_checkpoint.pt`。

no-riskcap + fragmentguardv2 + block2 single-choice guard 的 cross-seed 结果：

| 指标 | 数值 |
|---|---:|
| weighted micro accuracy | `21.596%` |
| average blocks | `2.153 / 4` |
| observed loss vs prediction-stability | `19` |
| gain vs prediction-stability | `19` |

Artifact：

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_shape_features_fragmentguardv2_choice2_noriskcap_cross_seed_20260524/summary.json
```

这些结果属于 loss/gain cancellation，不是安全策略。

主要失败模式集中在自由文本 prefix / completion，而不是 multiple-choice 翻转：

- `AS-` vs `AS-205`
- `Metro:` vs `Metro: All Change`
- `$20` vs `$20 billion`
- `Laverne &` vs `Laverne & Shirley`
- HellaSwag `[step]` / `[substeps]` 后续动作句缺失
- StoryCloze 短句 prefix

post-hoc zero-loss frontier 平均约 `2.191/4` blocks，仍弱于 riskcap04 的 `2.118/4` zero-loss baseline。

判断：answer-shape features 有诊断价值，但 no-riskcap 版本不能作为 baseline。

### 7.4 Fragment-completeness v3

v3 guard 扩展了 non-stable decoded answer 的 completion 检查，覆盖：

- replacement character
- trailing continuation marker
- bare currency amount
- numeric range prefix
- connector ending
- short unstable phrase
- truncated hyphen token

no-riskcap v3 被 fail-fast 拒绝：

```text
seed66 / HellaSwag
25 losses vs prediction-stability
1.669 / 4 blocks
```

恢复 riskcap04 后，三 seed official8 全量结果：

| 指标 | 数值 |
|---|---:|
| weighted micro accuracy | `21.596%` |
| average blocks | `2.245 +/- 0.003 / 4` |
| saving vs prediction-stability | `0.267 block/sample` |
| observed loss vs fixed-final | `0` |
| observed loss vs prediction-stability | `0` |

Artifact：

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_shape_features_fragmentguardv3_choice2_riskcap04_cross_seed_20260524/summary.json
```

判断：v3 是安全的 diagnostic completion signal，但它比旧 joint-readiness riskcap04 的 `2.118/4` 更贵，所以不能替换当前 baseline。

## 8. 已降级或归档的旧路线

旧 Stage A/B/C、自建小 prior、GSM8K MVP 路线已经归档为历史诊断。

归档路径：

```text
/data1/luyifei/drla/archive/2026-05-24-legacy-stagebc-code
/data1/luyifei/drla/archive/2026-05-24-legacy-stagebc-artifacts
```

原因：

- 小 prior 可以 overfit 小样本，但不能泛化。
- 最好 target-text b32 20k run 在 `train512/test256` 只有 `7/256 = 2.73%` rollout answer accuracy。
- GSM8K 可用于 decoder / judge / OOD math diagnostic，但不再作为 Cola 主线 benchmark。

判断：旧路线可复查，但不能作为当前架构有效性的主证据。

## 9. 当前关键结论

1. Cola block-wise latent rollout 中存在可观测的 answer-readiness 信号。
2. EOS-only 几乎接近 fixed-final，不足以作为 halt 条件。
3. `prediction_stability` 是强 non-gold halt baseline。
4. learned continuation-risk + readiness calibration 可以在零 observed loss 下减少 block budget。
5. 当前最好 baseline 是 joint-readiness riskcap04，而不是 no-riskcap shape-risk 或 fragmentguardv3。
6. riskcap04 当前不能移除。去掉它会出现 loss/gain cancellation 或 HellaSwag/SQuAD/StoryCloze prefix-completion loss。
7. 手写 completion guard 有诊断价值，但继续堆规则不是最终方向。
8. 当前 decoder-side 信号是训练期 probe / teacher，不是最终 agent-latent 通信的在线依赖。
9. 下一步应把 decoder-side completion / stability / risk 信号蒸馏进 latent-student，使它在不解码时预测 answer-enough。

## 10. 当前风险与限制

当前结论的边界：

- 结果基于 current prepared full split，batch size 固定为 `12`。
- 当前 block 数为 `4`，如果改变 block count，需要重跑 trace/frontier/train/eval。
- 当前 decoding protocol 若改变，已有 halt 结论不能直接复用。
- 当前三 seed 能替代早期 two-seed caveat，但还不能声称广泛统计泛化。
- observed loss 为 0 不等于理论安全，只表示当前 official8 / 三 seed / 当前 split protocol 下没有观测到 loss。
- v3 guard 是手写规则，不应继续作为最终模型能力的替代物。
- 当前最好 baseline 仍使用 decoder-derived probe / text-stability 相关信号作为监督和评估依据，尚未证明最终 latent-only agent 通信早停已经完成。

## 11. 下一步实验方向

下一步不应做更小 MVP，也不应继续简单堆 v4/v5 手写 guard。更合理的方向是：

1. 保持 joint-readiness riskcap04 作为前一阶段 decoder-probed baseline、teacher source 和 safety frontier。
2. 训练 `LatentHaltStudent-v1` completion/stability/readiness signal，并逐步限制最终在线输入为 latent/process trajectory，覆盖：
   - 数字单位未完成
   - 实体连接符未完成
   - 公式 / 区间未完成
   - `[step]` continuation
   - 短句 prefix
   - task-scored answer 未来是否变化
3. 用 decoder probe / task scorer / prediction-stability / future gain / P0 riskcap04 chosen halt block 构造 teacher labels，但训练 student-only 或 latent-primary halt model；当前最强 SQuAD 诊断是 `p0_teacher_action + completion_risk`。下一步应扩展 action distillation 到全 8-task / 多 seed，并增加 answer identity / sequence-level stability objective，而不是继续盲扫 `d_model`、只换阈值，或只给 readout query 加浅层 process 偏置。
4. 在 official8、三 seed、leave-one-task-out transfer setting 下复验。
5. 继续使用 zero observed loss 校准，而不是 aggregate accuracy-only calibration。
6. 如果改变 batch size、block count、decoding 或任务集合，必须重跑完整链路。

完整链路仍然必须包括：

```text
trace -> scorer -> frontier labels -> readiness/risk train -> halt eval -> aggregate -> sample-level diagnostics -> docs
```

## 12. 一句话总结

当前框架已经从“训练一个小 latent prior”转为“在 official Cola latent diffusion 推理过程中，可靠判断 latent message 是否已经足够”。P0 最好结果是 joint-readiness riskcap04：三 seed official8 零 observed loss，平均从 `4` blocks 降到 `2.118` blocks。P1 最新正结果是 P0 action distillation：SQuAD LOTO 在 `2.049/4` blocks 保持 fixed-final accuracy 且零 correctness loss。剩余核心问题不是“能不能早停”，而是把 decoder-probed answer identity / stability 也迁移成 latent-primary 可学习信号，最终服务 agent-to-agent latent communication。
