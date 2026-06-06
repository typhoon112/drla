# CoLA 线完整归档入口

更新时间：2026-06-06

本文是工作区内 CoLA 线的归档入口。目标是把当前实验结果、代码入口、权重、配置和报告文档整理成可复现索引，尤其锁定 P0 与 P1 两个早停判别器阶段。

## 一句话结论

CoLA 线的有效核心已经收敛到：

```text
P0: decoder/probe/scorer 辅助的 adaptive halt teacher
P1: decoder-supervised、推理时 decoder-free 的 LatentHaltStudent-v1
```

P0 证明了 official CoLA block rollout 上存在安全早停信号；P1 证明轻量 latent/process-only 学生能学习到大部分 P0 readiness 信号。P2 的 CoLA multi-agent communication 方向已有重要负结果和接口诊断，但当前不应继续把 frozen CoLA 直接作为主线做 MAS claim。

## Canonical 阅读顺序

1. `/data1/luyifei/drla/docs/cola_archive/README.md`
2. `/data1/luyifei/drla/docs/cola_archive/P0_Adaptive_Halt_Reproducibility.md`
3. `/data1/luyifei/drla/docs/cola_archive/P1_Latent_Halt_Student_Reproducibility.md`
4. `/data1/luyifei/drla/docs/cola_archive/CODE_AND_ARTIFACT_MANIFEST.md`
5. `/data1/luyifei/drla/docs/cola_archive/P2_CoLA_Line_Freeze_Notes.md`
6. `/data1/luyifei/drla/docs/p0_reports/cola_adaptive_halt_paper_report_zh.md`
7. `/data1/luyifei/drla/docs/p1_archive/P1_Model_Comparison_Report_2026-05-27.md`

机器可读索引：

```text
/data1/luyifei/drla/docs/cola_archive/cola_line_manifest.json
```

## 阶段状态

| 阶段 | 状态 | 主结论 | Canonical artifact |
|---|---|---|---|
| Official CoLA baseline | 已归档 | official 8-task full benchmark 是能力锚点，不是 P1 要提升的目标。 | `/data1/luyifei/drla/outputs/cola_official_benchmarks/full_b64_bs12_trace_score_20260524/summary.json` |
| P0 adaptive halt teacher | 已归档 / 可引用 | `joint-readiness + prediction-change risk + riskcap04` 在 3 seeds full split 上 0 observed losses，同时从 prediction-stability 再省约 `0.394` blocks。 | `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_joint_readiness_riskcap04_cross_seed_20260524/summary.json` |
| P1 latent halt student | 已归档 / 推荐停止点 | 最优 student-only 结果为 `4` losses、`606` mismatches、`1.812/4` blocks，说明 P1 已充分学习 P0 信号。 | `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_trajtok_answer_identity_action_completionrisk_identitystable_boundarypen02_cross_seed_20260527/summary.json` |
| P1 locked risk audit | 已归档 / 证书视角 | fresh split locked riskcert 结果为 `4` losses、`91` mismatches、`1.834/4` blocks；是 observed-low-risk 证据，不是完全形式化认证。 | `/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_locked_seed66_67_68_split20260601_riskcert_20260527/summary.json` |
| P2 CoLA MAS/interface | 冻结为诊断线 | CoLA 有 latent/channel 信号，但当前 frozen/interface 能力低于进入 text-vs-latent 主表的门槛。 | `/data1/luyifei/drla/docs/cola_archive/P2_CoLA_Line_Freeze_Notes.md` |

## 不要混淆的口径

- Official CoLA benchmark：完整生成 4 个 block 后打分，衡量冻结 CoLA 的最终能力。
- P0 halt replay：使用 decoder/probe/text/scorer derived features 的 teacher-style adaptive halt。
- P1 halt evaluation：同一样本上比较 selected block 与 fixed-final / prediction-stability，衡量省 block 时是否损 correctness。
- P1 不提升 CoLA benchmark accuracy，也不应该被写成提升 CoLA 官方精度。
- P1 在线输入不包含 decoded text、decoder stop probe、task scorer、gold answer 或 correctness；这些只用于离线监督标签与评估。
- P2 CoLA communication 不能用 early legacy scorer-visible A tokens 的结果作为主 claim。

## 当前建议

P0/P1 作为 CoLA 早停判别器线已经可以归档。后续若写论文或继续研究，应把 P0/P1 作为 evidence 和 teacher/student 设计基础，而不是继续在 official8 上用小消融榨取局部最优。

下一阶段若回到 agent-to-agent latent communication，应以 `/data1/luyifei/drla/docs/current/P2_Locked_Complete_Execution_Scheme_2026-06-01.md` 为准，优先使用真正支持 multi-agent 分工的 benchmark/protocol，并先通过 capability gate。
