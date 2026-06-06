# P1 阶段归档与 P2 过渡记录

> 状态：P1 frozen archive + P2 过渡记录。瘦身前完整版本见 `/data1/luyifei/drla/docs/full_history/2026-05-29_pre_slim/P1_Final_Archive_and_P2_Plan_2026-05-27.md`。P2 最新 canonical 实施方案见 `/data1/luyifei/drla/docs/current/P2_Latent_MAS_Communication_Implementation_Plan_2026-05-29.md`。

更新时间：2026-05-27

## 1. P1 结论

P1 阶段可以视为完成。`LatentHaltStudent-v1` 已经在 official8、3 seeds、leave-one-task-out、target-valid calibration 协议下学习到 P0 decoder-probed readiness signal。

当前 P1 student-only 最好路线：

```text
trajectory_token
+ answer_identity_action
+ completion_risk
+ answer_identity_stability
```

它不表示 P1 提升了 official Cola benchmark accuracy。P1 的贡献是：在同一批 held-out target-test samples 上，学习何时可以提前 halt，并尽量保持 final-block correctness。

## 2. P1 Locked Result

Frozen setup：

```text
model:
  cross_task_full_b64_bs12_seed{66,67,68}_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_20260527

split_seed:
  20260601

aggregate_summary:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/
  official8_full_b64_bs12_p1_locked_seed66_67_68_split20260601_riskcert_20260527/summary.json

loss_case_audit:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/
  official8_full_b64_bs12_p1_locked_seed66_67_68_split20260601_riskcert_20260527/loss_case_audit.json
```

Metrics：

```text
folds:
  24

repeated target-test decisions:
  14940

selected accuracy:
  20.930%

fixed-final accuracy:
  20.950%

prediction-stability accuracy:
  20.957%

avg selected blocks:
  1.834 / 4

prediction-stability avg blocks:
  2.501 / 4

losses vs final:
  3

losses vs prediction-stability:
  4

mismatches vs final:
  85

mismatches vs prediction-stability:
  91

calibration_joint_risk_satisfied:
  21 / 24 folds
```

Interpretation：

```text
Observed-low-risk + partially certified.
Not fully certified formal safety.
Do not tune P1 model or thresholds based on locked result.
```

3 个未满足 certificate target 的 fold 全部是 OBQA：observed loss/mismatch 为 0，但 target-valid 样本太少，Wilson upper bound 仍超过当前 target。

## 3. Code and Protocol

Key scripts：

```text
P1 train:
  /data1/luyifei/drla/drla/scripts/train_cola_latent_halt_student.py

P1 eval:
  /data1/luyifei/drla/drla/scripts/eval_cola_latent_halt_student.py

P1 aggregate:
  /data1/luyifei/drla/drla/scripts/aggregate_cola_latent_halt_student_subseed_loto.py

P2 packet builder:
  /data1/luyifei/drla/drla/scripts/build_cola_agent_latent_comm_packets.py
```

Protocol：

```text
official Cola 8 tasks
b64 / bs12 / t16
seeds: 66, 67, 68
max blocks: 4
split by sample_key = task::sample_id
target valid = threshold calibration only
target test = reported held-out evaluation
```

Training policy：

```text
CUDA/GPU required
SwanLab cloud required
metrics.jsonl + best_checkpoint.pt + last_checkpoint.pt required
```

Eval / aggregation：

```text
swanlab_mode=disabled
local artifacts only
```

## 4. P2 Packet v1

P2.3 protocol-level packet substrate completed：

```text
script:
  /data1/luyifei/drla/drla/scripts/build_cola_agent_latent_comm_packets.py

output:
  /data1/luyifei/drla/outputs/cola_agent_latent_comm/
  p2_agent_latent_comm_v1_locked_seed66_67_68_split20260601_20260527

packets:
  14940

latent_block_refs:
  27399

unique_latent_files_checked:
  8850

missing_latent_files:
  0

forbidden_decoder_or_eval_fields:
  0
```

Allowed online packet fields：

```text
latent_memory.blocks[*].latent_ref
latent_memory.blocks[*].process_features
readiness_state.scores
readiness_state.thresholds
readiness_state.margins
risk_certificate
```

Explicitly forbidden：

```text
decoded text
token ids
scored prediction
official score
gold / target
selected/final/prediction-stability prediction or correctness
prediction_stability_block
```

This proves：

```text
latent message protocol exists
packet sanitization works
artifact chain is auditable
```

This does not prove：

```text
Agent B can read the latent packet
latent handoff improves downstream task utility
latent handoff beats text handoff
heterogeneous latent communication is solved
```

## 5. P2 Supersession

P2.0-P2.4 in the original archive were transition notes. Current P2 canonical is now：

```text
/data1/luyifei/drla/docs/current/P2_Latent_MAS_Communication_Implementation_Plan_2026-05-29.md
```

P2 next steps：

```text
1. Packet v2 schema.
2. Distribution audit with corrupted controls.
3. Single-handoff Agent B receiver diagnostic.
4. Matched-depth text-channel vs latent-channel comparison.
5. Sequential communication envelope only after channel validation.
```

## 6. Related Docs

```text
P1 comparison:
  /data1/luyifei/drla/docs/p1_archive/P1_Model_Comparison_Report_2026-05-27.md

P1 design digest:
  /data1/luyifei/drla/docs/p1_archive/P1_LatentHaltStudent_v1_Design_and_Distillation.md

Current status:
  /data1/luyifei/drla/docs/current/CURRENT_EXPERIMENT_STATUS.md

P2 canonical:
  /data1/luyifei/drla/docs/current/P2_Latent_MAS_Communication_Implementation_Plan_2026-05-29.md
```
