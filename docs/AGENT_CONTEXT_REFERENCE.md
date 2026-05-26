# AGENT 上下文参考

本文档保存从 `AGENT.md` 移出的运行细节，用来减少默认上下文长度。只有需要 artifact 路径、脚本入口或当前结果细节时再读。

## 关键流程

当前执行顺序以 `docs/DRLA_Implementation_Plan.md` 为准：

```text
official Cola baseline
-> block-wise rollout trace
-> per-block decoder probe
-> oracle readiness frontier
-> Phase P0 decoder-probed readiness baseline
-> Phase P1 decoder-as-teacher LatentHaltStudent-v1
-> adaptive halt accuracy-cost frontier
-> 可选 LoRA/adapter/RL-style policy tuning
```

历史自建 Stage B/C small-prior 代码和输出归档在：

```text
/data1/luyifei/drla/archive/2026-05-24-legacy-stagebc-code
/data1/luyifei/drla/archive/2026-05-24-legacy-stagebc-artifacts
```

这些归档只用于复现，不作为最终架构证据。

## 当前 Trace 入口

当前 trace 入口：

```text
/data1/luyifei/drla/drla/scripts/collect_cola_block_traces.py
```

它沿用 official Cola VAE/DiT block-wise inference，并额外记录 per-block trace JSONL、score-ready generation JSONL、本地 `metrics.jsonl`、raw latent shard `.pt`、decoder logits probe、EOS/im_end probe、answer text/stability signals。

Full prepared-split protocol 必须显式记录：

```text
seed
per_sample_noise_seed
batch_size
SwanLab run id
model path
dataset version
generation config
```

主结论不得混用 batch size。当前 full prepared split 使用 `batch_size=12`。Batch-invariance 诊断文件：

```text
/data1/luyifei/drla/outputs/cola_batch_invariance/batch_size_invariance_summary_20260524.json
```

## 当前 Phase P0 基线

当前 decoder-probed Phase P0 safety/cost baseline：

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_joint_readiness_riskcap04_cross_seed_20260524
```

阶段解释：

```text
Phase P0 = decoder-dependent / decoder-probed readiness baseline
Phase P1 = decoder-as-teacher, LatentHaltStudent-v1 student-only halt
```

P0 必须保留为：

```text
readiness existence evidence
dense teacher-label source
safety/cost upper-bound baseline
sample-level diagnostic set
```

P1 结果必须和 P0 并列报告 gap analysis，不得覆盖 P0。

## 主要 Full-Split 产物

完整 artifact 索引见 `docs/CURRENT_EXPERIMENT_STATUS.md`。常用路径：

```text
full bs12 merged traces:
  /data1/luyifei/drla/outputs/cola_block_traces/tasks_official8_full_b64_t16_seed66_bs12_merged_20260524
  /data1/luyifei/drla/outputs/cola_block_traces/tasks_official8_full_b64_t16_seed67_bs12_merged_20260524
  /data1/luyifei/drla/outputs/cola_block_traces/tasks_official8_full_b64_t16_seed68_bs12_merged_20260524

full bs12 score:
  /data1/luyifei/drla/outputs/cola_official_benchmarks/full_b64_bs12_trace_score_20260524/summary.json

full bs12 cross-seed summary:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_seed_20260524

full bs12 cross-task prediction-change risk cross-seed:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_prediction_change_risk_cross_seed_20260524

full bs12 shape-risk fragmentguardv3 riskcap04 cross-seed:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_shape_features_fragmentguardv3_choice2_riskcap04_cross_seed_20260524

Phase P1 LatentHaltStudent seed66 same-split:
  train: /data1/luyifei/drla/outputs/cola_latent_halt_student/official8_full_b64_bs12_seed66_d64_pma4_seed20260524
  eval:  /data1/luyifei/drla/outputs/cola_latent_halt_student_eval/official8_full_b64_bs12_seed66_d64_pma4_best_strict_textaudit_20260525
  3-seed summary: /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_same_split_20260525

Phase P1 LatentHaltStudent leave-one-task-out:
  train roots: /data1/luyifei/drla/outputs/cola_latent_halt_student/cross_task_full_b64_bs12_seed{66,67,68}_d64_pma4_20260525
  eval roots:  /data1/luyifei/drla/outputs/cola_latent_halt_student_eval/cross_task_full_b64_bs12_seed{66,67,68}_d64_pma4_strict_textaudit_20260525
  seed summaries: /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_seed{66,67,68}_20260525/summary.json
  cross-seed summary: /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_cross_seed_20260525/summary.json

Phase P1 seed68 pooling ablation all_tokens:
  train root: /data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d64_alltokens_20260525
  eval root:  /data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_alltokens_strict_textaudit_20260525
  summary:    /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_seed68_d64_alltokens_20260525/summary.json
  comparison: /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_ablation_seed68_pooling_20260525/summary.json

Phase P1 seed68 pooling ablation d64_pma1:
  train root: /data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d64_pma1_20260525
  eval root:  /data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_pma1_strict_textaudit_20260525
  summary:    /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_seed68_d64_pma1_20260525/summary.json
  comparison: /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_ablation_seed68_arch_20260525/summary.json

Phase P1 seed68 pooling ablation d64_mean_max:
  train root: /data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d64_meanmax_20260525
  eval root:  /data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_meanmax_strict_textaudit_20260525
  summary:    /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_seed68_d64_meanmax_20260525/summary.json
  comparison: /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_ablation_seed68_arch_20260525/summary.json

Phase P1 seed68 capacity ablation d128_pma4:
  train root: /data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d128_pma4_20260525
  eval root:  /data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d128_pma4_strict_textaudit_20260525
  summary:    /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_seed68_d128_pma4_20260525/summary.json
  comparison: /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_ablation_seed68_arch_20260525/summary.json

Phase P1 seed68 capacity ablation d32_pma4:
  train root: /data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d32_pma4_20260525
  eval root:  /data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d32_pma4_strict_textaudit_20260525
  summary:    /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_seed68_d32_pma4_20260525/summary.json

Phase P1 seed68 calibration ablation zero-loss-zero-mismatch:
  eval root:  /data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_zero_mismatch_calib_20260525
  summary:    /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_seed68_d64_pma4_zero_mismatch_calib_20260525/summary.json
  comparison: /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_calibration_seed68_20260525/summary.json

Phase P1 seed68 calibration ablation d64 per_task:
  eval root:  /data1/luyifei/drla/outputs/cola_latent_halt_student_eval/cross_task_full_b64_bs12_seed68_d64_pma4_per_task_calib_20260525
  summary:    /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_seed68_d64_pma4_per_task_calib_20260525/summary.json

Phase P1 seed68 readout-context diagnostic, leave SQuAD out only:
  train root: /data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_readoutctx_20260525/leave_squad_out
  eval root:  /data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_readoutctx_strict_textaudit_20260525/leave_squad_out_eval_squad_all
  per-task eval root: /data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_readoutctx_per_task_calib_20260525/leave_squad_out_eval_squad_all

Phase P1 seed68 P0-teacher action distillation, leave SQuAD out only:
  P0 teacher decisions:
    /data1/luyifei/drla/outputs/cola_risk_gated_halt_teacher/cross_task_full_b64_bs12_joint_readiness_riskcap04_seed20260526/train_tasks_for_leave_squad_out_exact/analysis/risk_gated_decisions.jsonl
  p0_teacher_halt + completion_risk train:
    /data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_p0teach_completionrisk_20260525/leave_squad_out
  p0_teacher_halt + completion_risk refined eval:
    /data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_p0teach_completionrisk_refined_textaudit_20260525/leave_squad_out_eval_squad_all
  p0_teacher_action + completion_risk train:
    /data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_p0teach_action_completionrisk_20260525/leave_squad_out
  p0_teacher_action + completion_risk refined eval:
    /data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_p0teach_action_completionrisk_refined_textaudit_20260525/leave_squad_out_eval_squad_all
  p0_teacher_action zero-mismatch audit:
    /data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_p0teach_action_completionrisk_zero_mismatch_with_final_20260525/leave_squad_out_eval_squad_all
```

关键 SwanLab runs：

```text
reference baseline: cca5o9r6t2sbchaye2stz
trace scoring: mgcjtq4h7z1swne7bp170
oracle frontier: ann4meosdsnlnilmh8wsk
readiness model: szibnlfjrec1mq1j6kkk9
adaptive halt: gpbfpw22r4tl6la6l9r4j
P1 student train seed66: 4s6ie43k9v6a8a8fgor1p
P1 student train seed67/68: 6lw6w81ham2ic5fa5twzc / fz9ctcc9i4cb2hga0yju6
historical P1 no-training eval runs, do not repeat: cmjq5ic0nwpajl751f5lw / 7ukb091l1sbt7bj6644k6 / tg6zgoeblngg89uoazzwh
```

## 当前结论

在 official protocol 上，process/probe/stability features 比 raw latent alone 更有信息。EOS-only 几乎等同 fixed-final，不足以作为 halt baseline。`prediction_stability` 是强 non-gold halt baseline。

当前最强 Phase P0 policy 是 joint-readiness riskcap04：它在 valid 上 sweep readiness thresholds，要求相对 prediction-stability 的 zero valid loss，限制 `risk_threshold_end=0.4`，并使用 block2 single-choice guard。它在 seed66/67/68 上达到 `21.596% +/- 0.030%` weighted micro accuracy、`2.118 +/- 0.010/4` blocks，并且相对 fixed-final 和 prediction-stability 都是 zero observed loss。

Cross-seed 38-feature answer-shape no-riskcap 已完成全部 `24` 个 leave-one-task-out checkpoints，但 valid-selected policy 相对 prediction-stability 有 `19` losses 和 `19` gains。它只能作为诊断，不能作为安全策略。

Fragmentguardv3 捕获了更多 completion fragments；no-riskcap v3 在 seed66/HellaSwag 上 early fail，出现 `25` losses。Riskcap04+v3 恢复 zero loss，但成本为 `2.245/4` blocks，弱于 riskcap04。

Phase P1 已完成 seed66/67/68 same-split 首轮：`LatentHaltStudent-v1` 用 latent/process/budget-only online inputs，训练期 teacher targets 来自 decoder/scorer/stability/future gain。三种子 test readiness AUROC 平均 `0.7317`，prediction-change AUROC 平均 `0.9953`。strict student-only eval 平均 micro accuracy `20.737%`、`2.324/4` blocks，相比 prediction-stability 平均 `2.516/4` blocks 只多省 `0.193` block；seed67 出现 `2` 个 test correctness losses，三种子共有 `9` 个 selected predictions 与 prediction-stability text 不一致。因此 P1 信号存在，但还不能替代 P0 riskcap04；下一步必须做 leave-one-task-out、结构消融和更严格稳定性校准。

Phase P1 seed66/67/68 leave-one-task-out 已完成。seed summaries: `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_seed{66,67,68}_20260525/summary.json`；cross-seed summary: `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_cross_seed_20260525/summary.json`。三 seed 聚合：student-only 为 `21.556%` micro accuracy、`2.048/4` blocks；prediction-stability 为 `21.596%`、`2.512/4` blocks。P1 多省 `0.463` block，但相对 prediction-stability 有 `58/147057 = 0.0394%` losses 和 `1612/147057 = 1.0962%` text mismatches。

P1 LOTO 解释必须同时看 loss count、loss rate、mismatch rate 和跨 seed 复现性。`lambada` 是 `3/15459 = 0.0194%`、`mmlu` 是 `2/42126 = 0.0047%`、`race` 是 `2/14661 = 0.0136%`，属于低频边界或 mismatch watch，不能和系统性失败混为一谈。真正稳定暴露 completion/stability 风险的是 `hellaswag`（`12/30126 = 0.0398%` loss、`1.1983%` mismatch）和 `squad`（`39/31710 = 0.1230%` loss、`3.8505%` mismatch）。结论：P1 latent/process-only student 有 transfer 信号，但当前版本仍不能替代 P0 joint-readiness riskcap04。

P1 seed68 pooling ablation `all_tokens` 已完成。它把 selected mismatch 从 `43` 降到 `17`，loss 仍为 `2/49019 = 0.0041%`，但平均 blocks 从 `2.325/4` 增到 `3.979/4`，比 prediction-stability 的 `2.512/4` 还贵。结论：当前全 slot attention 不是更好的 halt policy；它主要让阈值校准退回 final block，说明瓶颈不只是 PMA pooling 信息损失。

P1 seed68 pooling ablation `d64_pma1` 已完成。它得到 `21.592%` micro accuracy、`2.793/4` blocks，loss 为 `5/49019 = 0.0102%`，text mismatch 为 `315/49019 = 0.6426%`；相比 `d64_pma4_last` 更贵、更不安全，也比 prediction-stability 的 `2.512/4` blocks 更贵。结论：单 query PMA 压缩过强，当前应保留 PMA K=4 + explicit last-slot readout。

P1 seed68 pooling ablation `d64_mean_max` 已完成。它得到 `21.594%` micro accuracy、`2.837/4` blocks，loss 为 `4/49019 = 0.0082%`，text mismatch 为 `115/49019 = 0.2346%`；它比 `pma1/d128` mismatch 少，但仍比 `d64_pma4_last` 更贵且 loss 更多，也比 prediction-stability 更贵。结论：简单 mean/max pooling 不是缺失 readout。

P1 seed68 capacity ablation `d128_pma4_last` 已完成。它得到 `21.594%` micro accuracy、`2.432/4` blocks，loss 为 `4/49019 = 0.0082%`，text mismatch 为 `285/49019 = 0.5814%`；相比 seed68 `d64_pma4_last` 更不安全，且只比 prediction-stability 省 `0.080` block。结论：单纯扩容不是下一步主杠杆，应先处理校准、readout、teacher objective 和 stability-aware threshold。

P1 seed68 capacity ablation `d32_pma4_last` 已完成。它得到 `21.553%` micro accuracy、`2.126/4` blocks，loss 为 `27/49019 = 0.0551%`，text mismatch 为 `263/49019 = 0.5365%`；SQuAD 是主要风险源。结论：单纯缩小 width 会变便宜但不安全，当前 seed68 reference 仍是 `d64_pma4_last process_token full`。

P1 seed68 calibration ablation `zero-loss-zero-mismatch valid calibration` 已完成。它把 held-out loss 从 `2` 降到 `1`、mismatch 从 `43` 降到 `22`，但 blocks 从 `2.325/4` 增到 `3.872/4`，比 prediction-stability 还贵 `1.361` blocks。结论：text-stability 校准是有效 safety diagnostic，但不能直接作为默认策略；应把该信号蒸馏进 learned stability-aware readout/teacher objective。

P1 seed68 calibration ablation `d64_pma4_last per_task calibration` 已完成。它要求每个 calibration task 都满足约束，避免 pooled valid 抵消风险；结果和 pooled d64 几乎相同：`21.598%` micro accuracy、`2.325/4` blocks、`2/49019 = 0.0041%` loss、`43/49019 = 0.0877%` mismatch。结论：d64 的阈值问题不是简单 pooled leakage；下一步应改变 learned readout / latent-process interaction，而不是继续调阈值。

P1 seed68 readout-context diagnostic `last_process_query` 已完成 leave-SQuAD-out 单任务否决。训练使用 SwanLab run `j4wmhdpn4v6qmei2lptja`，best step `1500`，best valid readiness AUROC `0.7487`。held-out SQuAD pooled/per_task eval 都是 `21.977%` accuracy、`1.755/4` blocks、`6/10570` loss、`223/10570` mismatch；baseline d64 SQuAD 是 `22.034%`、`3.999/4` blocks、`0` loss、`5` mismatch。失败仍是 `October 16,`、`15`、`1568-`、`194` 等前缀 completion boundary。结论：浅层 process-conditioned readout query 会过早 halt，不扩展到 8-task 主实验。

P1 seed68 explicit completion-boundary diagnostic `d64_pma4_last + completion_risk` 已完成 leave-SQuAD-out。训练使用 SwanLab run `gsarp1b2dzfmhh6nluhws`，training root `/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_completionrisk_20260525/leave_squad_out`，best step `1750`，best valid mean AUROC(readiness/prediction-change/completion-risk) `0.9140`。completion-risk teacher target 是当前 scored prediction 为空，或是 prediction-stability/final reference 的 strict prefix；它只做离线 decoder teacher label，不进入在线输入。held-in test completion-risk AUROC `0.9967`，说明该局部边界信号可学。SQuAD default strict eval root `/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_completionrisk_strict_textaudit_20260525/leave_squad_out_eval_squad_all`：`22.034%` accuracy、`3.9996/4` blocks、`0/10570` loss、`2/10570` mismatch。refined threshold root 有 `0` loss 但 `6` mismatch；risk-only probe root 有 `10/10570` loss、`354/10570` mismatch、`1.686/4` blocks。结论：completion-risk 是有效辅助蒸馏信号，但单个辅助 head 不能替代 answer-readiness / P0 teacher-policy 蒸馏。

P1 seed68 P0-teacher distillation 已完成 leave-SQuAD-out 单任务诊断。`p0_teacher_halt` 使用 P0 riskcap04 chosen block 的 at/after 标签，`p0_teacher_action` 只把 P0 chosen block 本身标为 stop action；两者在线输入都仍是 latent/process-only。`p0_teacher_halt + completion_risk` refined SQuAD eval 为 `22.034%` accuracy、`3.529/4` blocks、`0/10570` loss、`74` mismatch。当前最强是 `p0_teacher_action + completion_risk`，SwanLab run `wkirim527d08abt606u8e`，best step `2420`，best valid mean AUROC `0.9897`；SQuAD refined eval 为 `22.034%` accuracy、`2.049/4` blocks、`0/10570` correctness loss、`841/10570` text mismatch。所有 mismatch 都发生在 fixed-final 本来也错误的样本上，说明 correctness/cost frontier 很强，但 answer identity/stability 仍未解决。若校准强制 zero loss + zero text mismatch 并加入 `readiness=1.01` fixed-final 候选，策略会退回 `4.0/4` blocks。

P1 seed68 teacher-objective ablation `stabilityw2` 已完成。Training root `/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_stabilityw2_20260525`，eval root `/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_stabilityw2_strict_textaudit_20260525`，aggregate summary `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_seed68_d64_pma4_stabilityw2_20260525/summary.json`，objective comparison `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_stability_objective_seed68_20260525/summary.json`。它将 `prediction_change_loss_weight` 提到 `2.0` 并用 `readiness_prediction_change_mean_auroc` 选 best checkpoint，但结果为 `21.590%` micro accuracy、`2.860/4` blocks、`6/49019 = 0.0122%` loss、`62/49019 = 0.1265%` mismatch；比 `d64_pma4_last` baseline 更贵且更不安全。严格 zero-mismatch calibration 可压到 `2` loss / `32` mismatch，但代价是 `3.980/4` blocks。结论：不要把“加大 prediction-change loss 权重”作为主线；后续应换成更好的结构交互、task conditioning 或校准目标。

P1 seed68 process-feature ablation `no_block_budget` 已完成。代码支持 `--process-feature-mode full|no_block_budget`；`no_block_budget` 移除 `block_number/max_block_budget/remaining_blocks/block_fraction`，保留 latent norm/delta/cosine/drift。Training root `/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_nobudget_20260525`，eval root `/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_nobudget_strict_textaudit_20260525`，aggregate summary `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_seed68_d64_pma4_nobudget_20260525/summary.json`，process-feature comparison `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_process_feature_ablation_seed68_20260525/summary.json`。结果为 `21.586%` micro accuracy、`1.831/4` blocks、`8/49019 = 0.0163%` loss、`389/49019 = 0.7936%` mismatch；zero-mismatch calibration 后为 `3` loss、`124` mismatch、`3.495/4` blocks。结论：block/budget 目前是重要 calibration anchor，不能硬删；但它也可能是位置捷径，后续应以更强 latent-process interaction 替代，而不是把它当最终 agent-communication 证据。

P1 seed68 process-interaction ablation `film` 已完成。代码支持 `--process-interaction-mode process_token|film`；FiLM 用 process features 调制 slot tokens，而不是追加 process token。Training root `/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_film_20260525`，primary eval root `/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/cross_task_full_b64_bs12_seed68_d64_pma4_film_strict_textaudit_20260525`，primary aggregate `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_seed68_d64_pma4_film_20260525/summary.json`，zero-mismatch aggregate `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_loto_seed68_d64_pma4_film_zero_mismatch_calib_20260525/summary.json`，process-interaction comparison `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_latent_halt_student_process_interaction_ablation_seed68_20260525/summary.json`。Primary 得到 `21.586%` micro accuracy、`2.260/4` blocks、`8/49019 = 0.0163%` loss、`190/49019 = 0.3876%` mismatch；zero-mismatch calibration 后仍有 `8` loss、`150` mismatch，且 blocks 升到 `3.421/4`。SQuAD 保留 `7` loss 和 `134` mismatch。结论：FiLM 不是当前主线赢家，seed68 reference 仍是 `d64_pma4_last process_token full`。

## 常用命令

```bash
source /data1/luyifei/drla/scripts/activate_conda.sh
python -m pip check
swanlab verify
```

完整 smoke commands 和脚本说明见：

```text
/data1/luyifei/drla/drla/scripts/README.md
```

凡是启动 optimizer、更新模型参数或产生可比较训练结论的深度学习训练、微调、probe、readiness/halt、LoRA/adapter、ablation 实验，都必须使用 SwanLab cloud；训练 smoke 也不例外。反过来，纯 eval、threshold sweep、trace collection、frontier building、aggregation、`py_compile`、数据格式检查等无训练过程脚本不要上 SwanLab；它们代码层只接受 `swanlab_mode=disabled`，只写本地 artifact。历史上已经产生的 no-training SwanLab run 只能视为记录污染，不能当训练曲线或收敛证据。
