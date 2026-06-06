# CoLA 线代码与 Artifact Manifest

更新时间：2026-06-06

本文列出 CoLA 线可复现所需的代码入口、结果目录、权重目录和报告文档。归档策略是保留原路径，不移动大体积 artifact。

## 目录体量快照

| 路径 | 大小 / 数量 |
|---|---:|
| `/data1/luyifei/drla/outputs/cola_experiment_summaries` | 20M |
| `/data1/luyifei/drla/outputs/cola_official_benchmarks` | 44K |
| `/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation` | 2.9G |
| `/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_locked` | 533M |
| `/data1/luyifei/drla/outputs/paper_report_20260525` | 37M |
| P1 all ablation `best_checkpoint.pt` | 216 |
| P1 all ablation `metrics.jsonl` | 216 |
| P1 all ablation `summary.json` | 216 |
| P1 best route `best_checkpoint.pt` | 24 |
| P1 best route `metrics.jsonl` | 24 |
| P1 best route `summary.json` | 24 |

## Official CoLA baseline artifacts

```text
/data1/luyifei/drla/outputs/cola_official_benchmarks/
full_b64_bs12_trace_score_20260524/summary.json

/data1/luyifei/drla/outputs/cola_official_benchmarks/
full_b64_bs12_seed67_trace_score_20260524/summary.json

/data1/luyifei/drla/outputs/cola_official_benchmarks/
full_b64_bs12_seed68_trace_score_20260524/summary.json
```

Reference CSVs：

```text
accuracy_summary.csv
reference_summary.json
```

## Trace roots

Canonical full traces：

```text
/data1/luyifei/drla/outputs/cola_block_traces/tasks_official8_full_b64_t16_seed66_bs12_merged_20260524
/data1/luyifei/drla/outputs/cola_block_traces/tasks_official8_full_b64_t16_seed67_bs12_merged_20260524
/data1/luyifei/drla/outputs/cola_block_traces/tasks_official8_full_b64_t16_seed68_bs12_merged_20260524
```

Segment/resume roots are preserved under:

```text
/data1/luyifei/drla/outputs/cola_block_traces/
```

Do not delete segment roots until merged roots and downstream summaries have been backed up externally.

## P0 canonical artifacts

Main:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/
official8_full_b64_bs12_cross_task_joint_readiness_riskcap04_cross_seed_20260524/
summary.json
```

Important diagnostics:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/
official8_full_b64_bs12_cross_task_prediction_change_risk_cross_seed_20260524/
summary.json

/data1/luyifei/drla/outputs/cola_experiment_summaries/
official8_full_b64_bs12_cross_task_shape_features_fragmentguardv2_choice2_noriskcap_cross_seed_20260524/
summary.json

/data1/luyifei/drla/outputs/cola_experiment_summaries/
official8_full_b64_bs12_cross_task_shape_features_fragmentguardv3_choice2_riskcap04_cross_seed_20260524/
summary.json
```

Report package:

```text
/data1/luyifei/drla/docs/p0_reports/cola_adaptive_halt_paper_report_zh.md
/data1/luyifei/drla/outputs/paper_report_20260525/cola_adaptive_halt_paper_report_zh.pdf
/data1/luyifei/drla/outputs/paper_report_20260525/figure_data.json
/data1/luyifei/drla/outputs/paper_report_20260525/figures/
```

## P1 canonical artifacts

Best student aggregate:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/
official8_full_b64_bs12_p1_trajtok_answer_identity_action_completionrisk_identitystable_boundarypen02_cross_seed_20260527/
summary.json
```

Best student aggregate CSVs:

```text
eval_summary_rows.csv
seed_summary.csv
seed_task_summary.csv
subseed_summary.csv
task_summary.csv
```

Best student checkpoint roots:

```text
/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/
cross_task_full_b64_bs12_seed66_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_20260527/

/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/
cross_task_full_b64_bs12_seed67_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_20260527/

/data1/luyifei/drla/outputs/cola_latent_halt_student_ablation/
cross_task_full_b64_bs12_seed68_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_20260527/
```

Best student eval roots:

```text
/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/
cross_task_full_b64_bs12_seed66_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_targetcal_cap128_boundarypen02_subseeds_20260527

/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/
cross_task_full_b64_bs12_seed67_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_targetcal_cap128_boundarypen02_subseeds_20260527

/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_ablation/
cross_task_full_b64_bs12_seed68_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_targetcal_cap128_boundarypen02_subseeds_20260527
```

Locked riskcert aggregate:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/
official8_full_b64_bs12_p1_locked_seed66_67_68_split20260601_riskcert_20260527/
summary.json
```

Locked eval roots:

```text
/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_locked/
cross_task_full_b64_bs12_seed66_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_split20260601_riskcert_20260527

/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_locked/
cross_task_full_b64_bs12_seed67_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_split20260601_riskcert_20260527

/data1/luyifei/drla/outputs/cola_latent_halt_student_eval_locked/
cross_task_full_b64_bs12_seed68_d64_pma4_trajtok_answer_identity_action_completionrisk_identitystable_split20260601_riskcert_20260527
```

## P2 CoLA diagnostic artifacts

Packet substrate:

```text
/data1/luyifei/drla/outputs/cola_agent_latent_comm/
p2_agent_latent_comm_v2_locked_seed66_67_68_split20260601_20260529
```

Distribution audit:

```text
/data1/luyifei/drla/outputs/cola_agent_latent_comm/
p2_agent_latent_comm_v2_distribution_audit_locked_seed66_67_68_split20260601_20260529
```

Channel eval and projection-gap diagnostics:

```text
/data1/luyifei/drla/outputs/cola_agent_channel_eval/
/data1/luyifei/drla/outputs/cola_sequential_latent_mas/
```

Phase A CoLA interface adaptation diagnostics:

```text
/data1/luyifei/drla/outputs/p2_phase_a_cola_dit_lora/
/data1/luyifei/drla/outputs/p2_phase_a_cola_interface_sft/
/data1/luyifei/drla/outputs/p2_phase_a_cola_latent_candidate_ranker/
```

These are diagnostic/freeze artifacts, not P0/P1 canonical early-halt results.

## Code entrypoints

Official benchmark and trace:

```text
/data1/luyifei/drla/drla/scripts/prepare_cola_official_benchmarks.py
/data1/luyifei/drla/drla/scripts/collect_cola_block_traces.py
/data1/luyifei/drla/drla/scripts/merge_cola_block_trace_segments.py
/data1/luyifei/drla/drla/scripts/eval_cola_benchmarks.py
```

P0:

```text
/data1/luyifei/drla/drla/scripts/build_cola_readiness_frontier.py
/data1/luyifei/drla/drla/scripts/train_cola_readiness_model.py
/data1/luyifei/drla/drla/scripts/train_cola_continuation_risk_model.py
/data1/luyifei/drla/drla/scripts/eval_cola_risk_gated_halt.py
/data1/luyifei/drla/drla/scripts/aggregate_cola_risk_gated_halt.py
```

P1:

```text
/data1/luyifei/drla/drla/scripts/train_cola_latent_halt_student.py
/data1/luyifei/drla/drla/scripts/eval_cola_latent_halt_student.py
/data1/luyifei/drla/drla/scripts/aggregate_cola_latent_halt_student_loto.py
/data1/luyifei/drla/drla/scripts/aggregate_cola_latent_halt_student_subseed_loto.py
/data1/luyifei/drla/drla/scripts/analyze_latent_halt_risk_control.py
```

P2 diagnostics:

```text
/data1/luyifei/drla/drla/scripts/build_cola_agent_latent_comm_packets.py
/data1/luyifei/drla/drla/scripts/audit_cola_agent_latent_packet_distribution.py
/data1/luyifei/drla/drla/scripts/run_cola_agent_b_channel_eval.py
/data1/luyifei/drla/drla/scripts/aggregate_cola_channel_eval.py
/data1/luyifei/drla/drla/scripts/audit_cola_channel_protocol_boundaries.py
/data1/luyifei/drla/drla/scripts/train_p2_phase_a_cola_dit_lora.py
/data1/luyifei/drla/drla/scripts/train_p2_phase_a_cola_latent_candidate_ranker.py
```

## Reproducibility checklist

- Use `/data1/luyifei/drla/scripts/activate_conda.sh`.
- Use GPU for every deep-learning training run.
- Deep-learning training must log to SwanLab cloud and local `metrics.jsonl`.
- Training checkpoints must include `best_checkpoint.pt` and `last_checkpoint.pt`.
- Pure eval, scoring, aggregation, audit, and report generation must be local-only and should not create SwanLab runs.
- For P1, report from `best_checkpoint.pt`, not `last_checkpoint.pt`.
- Do not compare official full benchmark task average directly to P1 same-split selected accuracy as if they were the same metric.
- Do not delete trace segment/resume roots until an external backup exists.
- Do not cite legacy P2 channel results where scorer saw Agent A text/replay tokens.
