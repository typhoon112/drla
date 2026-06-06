# P1 LatentHaltStudent-v1 Design Digest

> 状态：P1 实验笔记 digest。瘦身前完整设计与实验流水见 `/data1/luyifei/drla/docs/full_history/2026-05-29_pre_slim/P1_LatentHaltStudent_v1_Design_and_Distillation.md`。P1 主表与 paper-style 结论见 `/data1/luyifei/drla/docs/p1_archive/P1_Model_Comparison_Report_2026-05-27.md`。

## 1. P1 定位

P1 是从 P0 decoder-probed readiness 到 P2 latent communication 的桥梁：

```text
P0:
  decoder/text/scorer/probe features can be used online
  proves readiness exists
  provides teacher and safety/cost upper bound

P1:
  decoder/text/scorer/future blocks are teacher labels only
  online input is latent/process trajectory only
  learns a decoder-supervised latent halt student

P2:
  uses P1 readiness_state + latent memory as agent-agent packet substrate
```

P1 不提升 official Cola final accuracy。P1 只学习更早选择 latent block，并尽量保持 fixed-final / prediction-stability correctness。

## 2. 在线输入与禁止字段

P1 online inputs：

```text
raw latent prefix blocks z_1...z_b
latent norm / delta / cosine / drift
block index and remaining budget
process features available at current block
```

P1 online forbidden：

```text
decoded answer text
decoder EOS/im_end probe as inference feature
task scorer result
gold answer
official correctness
future block information
prediction-stability reference
```

Decoder/scorer/text signals 只用于 offline teacher labels、calibration labels、audit 和 final evaluation。

## 3. Architecture v1

当前 P1 student architecture：

```text
slot_adapter:
  standardize or LayerNorm each R16 latent slot
  Linear(16 -> d_model), default d_model=64
  slot/block position embeddings

process_token:
  MLP over block_idx, remaining_budget, norm, delta, cosine, drift
  appended to each block

intra_block_encoder:
  one lightweight self-attention layer over slot tokens + process token

block_pooler:
  PMA K=4 learned pooling queries
  explicit last_slot retained

trajectory_token:
  pooled block state + previous-block delta + process state
  enters causal inter-block Transformer

inter_block_encoder:
  2-layer causal Transformer over block summaries

readout:
  task-specific / head-specific queries
```

Important negative architecture results:

```text
all_tokens:
  too conservative; often falls back to final

pma1:
  over-compresses evidence

mean_max:
  lower mismatch but too expensive

d32 / d128:
  simple width sweep is not the key lever

no_block_budget:
  cheaper but unsafe; budget features remain calibration anchors

film:
  simple FiLM process interaction does not beat process/trajectory tokens
```

## 4. Teacher Targets

Main teacher signals：

```text
answer_identity_action:
  stop action at first block whose scored prediction identity matches stable/final reference

completion_risk:
  decoded answer empty or strict prefix/incomplete continuation of stable/final reference

answer_identity_stability:
  current answer identity already equals stable/final reference

prediction_change:
  current task-scored prediction differs from rollout stability reference

future_gain / correctness:
  auxiliary offline targets only
```

Do not continue stacking narrow binary heads without evidence. `empty_answer_risk` was a negative result because it shifted failures from empty answer to prefix/continuation boundary.

## 5. Best P1 Route

Current best student-only route：

```text
trajectory_token
+ answer_identity_action
+ completion_risk
+ answer_identity_stability
```

Development aggregate：

```text
summary:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/
  official8_full_b64_bs12_p1_trajtok_answer_identity_action_completionrisk_identitystable_boundarypen02_cross_seed_20260527/summary.json

losses:
  4

mismatches:
  606

average blocks:
  1.812 / 4
```

Locked result：

```text
summary:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/
  official8_full_b64_bs12_p1_locked_seed66_67_68_split20260601_riskcert_20260527/summary.json

selected accuracy:
  20.930%

fixed-final accuracy:
  20.950%

prediction-stability accuracy:
  20.957%

avg blocks:
  1.834 / 4

losses vs final:
  3

losses vs prediction stability:
  4

risk certificate:
  21 / 24 folds satisfied
```

Interpretation：

```text
P1 learned useful decoder-probed readiness signals from latent/process inputs.
P1 supports P2 packet construction.
P1 is not fully certified safety.
P1 does not solve agent-agent latent communication by itself.
```

## 6. P2 Interface Output

P1 eval emits `readiness_state` into halt decisions:

```text
readiness_state:
  halt_candidate_found
  fallback_to_final
  selected_block
  final_block
  scores
  thresholds
  margins
  risk certificate references
```

P2 packet builder consumes:

```text
latent_memory.blocks[*].latent_ref
latent_memory.blocks[*].process_features
readiness_state.scores
readiness_state.thresholds
readiness_state.margins
risk_certificate
```

P2 packet forbids decoded/eval-only fields.

## 7. Reproducibility Pointers

Train script：

```text
/data1/luyifei/drla/drla/scripts/train_cola_latent_halt_student.py
```

Eval script：

```text
/data1/luyifei/drla/drla/scripts/eval_cola_latent_halt_student.py
```

Aggregate scripts：

```text
/data1/luyifei/drla/drla/scripts/aggregate_cola_latent_halt_student_loto.py
/data1/luyifei/drla/drla/scripts/aggregate_cola_latent_halt_student_subseed_loto.py
```

P1 archive：

```text
/data1/luyifei/drla/docs/p1_archive/P1_Final_Archive_and_P2_Plan_2026-05-27.md
```

P1 comparison：

```text
/data1/luyifei/drla/docs/p1_archive/P1_Model_Comparison_Report_2026-05-27.md
```
