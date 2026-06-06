# DRLA 文档导航

更新日期：2026-06-06

本文档是 DRLA workspace 的文档入口。默认先读“当前入口”，需要追溯证据或历史路线时再读归档和实验笔记。

## 当前入口

| 文档 | 状态 | 用途 | 何时阅读 |
|---|---|---|---|
| `/data1/luyifei/drla/docs/DOCS_INDEX.md` | 当前文档导航 | 当前入口、归档、历史设计和工程运行文档的唯一导航 | 不确定该读哪份文档时 |
| `/data1/luyifei/drla/AGENT.md` | 当前 agent 约束 | 语言、环境、实验硬约束、当前主线边界 | 每次新会话或准备执行实验/改代码前 |
| `/data1/luyifei/drla/docs/current/CURRENT_EXPERIMENT_STATUS.md` | 当前状态快照 | P1/P2 当前快照、artifact 指针、下一步 | 需要确认当前阶段和最近结果时 |
| `/data1/luyifei/drla/docs/current/P2_Locked_Complete_Execution_Scheme_2026-06-01.md` | 当前最高优先级 P2 方案 | post-Family1 后锁定的完整执行顺序：Phase C true MAS validation -> Phase A CoLA adaptation -> Phase E CoLA TextMAS vs LatentMAS；同时锁定 benchmark/agent baseline 更换、CoLA 权重边界、Agent A->B 数据流和禁止事项 | 准备继续任何 P2 实验、下载/构建新 benchmark、训练 adapter/fuser 或写主表前 |
| `/data1/luyifei/drla/docs/current/P2_Latent_MAS_Communication_Implementation_Plan_2026-05-29.md` | P2 substrate/method canonical | same-substrate agent-agent latent communication 的实施方案；执行顺序以 locked scheme 为准 | 设计或实现 P2 packet、audit、receiver、sequential communication 时 |
| `/data1/luyifei/drla/docs/current/P2_Benchmark_and_Agent_Baseline_Redesign_2026-06-01.md` | 当前 P2 路线修订 | benchmark 准入、role-conditioned MAS baseline、CoLA 权重能力 gate、fuser 触发条件 | 设计或引用任何 P2 text-vs-latent / MAS 主实验前 |
| `/data1/luyifei/drla/docs/current/P2_Next_Phase_Execution_Plan_2026-06-01.md` | 历史执行纪律 | P2-D3.1 到 Branch B 前的 gate、禁止事项和 artifact 纪律，下一步表述已被 post-Family1 plan supersede | 追溯 P2-D3.1/旧 Branch B 纪律时 |
| `/data1/luyifei/drla/docs/current/P2_D1_Capability_Gate_Report_2026-06-01.md` | 当前 P2-D1 gate 报告 | 7 个候选 benchmark 的 full capability gate 结果与 admitted_tasks=[] 结论 | 决定下一步是否能进入 P2 主表前 |
| `/data1/luyifei/drla/docs/current/P2_D2_Locked_Split_Protocol_2026-06-01.md` | 当前 P2-D2 split 协议 | calibration/held-out split、split_seed、使用边界、防泄漏规则 | 做 prompt/protocol repair 或 held-out gate 前 |
| `/data1/luyifei/drla/docs/current/P2_D3_Prompt_Repair_Calibration_Report_2026-06-01.md` | 当前 P2-D3 calibration 报告 | prompt variant calibration 结果、GPQA single-only 正例和 Role TextMAS 失败结论 | 继续修 prompt/protocol 前 |
| `/data1/luyifei/drla/docs/current/P2_D4_Branch_Decision_Audit_2026-06-01.md` | 历史分支审计 | substrate adaptation vs benchmark redesign 的旧证据与文献对照，Branch B first 已执行并停止 | 追溯为什么尝试 Branch B Family 1 时 |
| `/data1/luyifei/drla/docs/current/P2_Branch_B_Execution_Plan_2026-06-01.md` | 历史 Branch B Family 1 方案 | frozen CoLA compatible benchmark redesign、official8-compatible role candidates、gate、主实验边界和 stop conditions | 追溯 Family 1 数据口径与停止条件时 |
| `/data1/luyifei/drla/docs/current/P2_Branch_B_Calibration_Report_2026-06-01.md` | Branch B calibration 报告 | official8-compatible split、single prompt sweep、MMLU role gate、admitted_tasks=[] | 追溯 Branch B 第一轮为什么未进入 held-out gate 时 |
| `/data1/luyifei/drla/docs/current/P2_Official8_Native_Alignment_Audit_2026-06-01.md` | 当前 native 对齐审计 | official CoLA native prompt/template/scorer 对齐、native single calibration、Family 1 stop condition | 判断是否继续 frozen official8-compatible Branch B 前 |
| `/data1/luyifei/drla/docs/current/P2_Post_Family1_Branch_Decision_Memo_2026-06-01.md` | 当前分支决策备忘录 | Family 1 停止后的 Branch A/C/Family2 证据、文献依据、推荐排序和需要用户确认的问题 | 开始任何新 P2 分支、训练或 benchmark 替换前 |
| `/data1/luyifei/drla/docs/current/P2_Post_Family1_Complete_Execution_Plan_2026-06-01.md` | post-Family1 背景方案 | Family 1 停止后的证据、Phase C/A/E 顺序、防扰乱规则和分支触发条件；已被 locked scheme 固化为默认路线 | 追溯为什么当前路线锁定为 Phase C -> A -> E 时 |
| `/data1/luyifei/drla/docs/current/P2_Phase_C_Benchmark_Protocol_Preparation_2026-06-01.md` | Phase C 安全准备 | true MAS benchmark/protocol 候选、manifest schema、baseline/ablation/scorer/leakage 规则 | 用户选择 Branch C 前后准备 manifest/scorer 时 |
| `/data1/luyifei/drla/docs/current/P2_Phase_C_Data_Source_and_Runner_Design_2026-06-01.md` | Phase C 数据源/runner 设计 | evidence-split QA、distributed-state、code workflow 的数据源排序、preflight/runner/scorer/failure taxonomy、Agent A->B 数据流与泄漏检查 | 选择 Branch C 后开始构建数据和 runner 前 |
| `/data1/luyifei/drla/docs/current/P2_Phase_C_Data_Source_Field_License_Audit_2026-06-01.md` | Phase C 数据源审计 | MuSiQue、HotpotQA、2WikiMultiHopQA 的字段、license、风险和 dry-inspection 要求 | 写真实数据 builder 或下载数据前 |
| `/data1/luyifei/drla/docs/current/P2_Benchmark_Redesign_Candidate_Inventory_2026-06-01.md` | Branch B 预备清单 | frozen CoLA compatible benchmark families、准入协议和 stop conditions | 接受 benchmark redesign 分支后准备新 manifest 前 |
| `/data1/luyifei/drla/docs/current/P2_Channel_Evaluation_Protocol_Audit_2026-05-31.md` | 当前 P2-D 协议审计 | Agent-A -> B 数据流、scorer 边界、legacy 表降级、receiver-only gate | 引用或重跑任何 P2-D text/latent communication 结果前 |
| `/data1/luyifei/drla/drla/scripts/README.md` | 当前脚本索引 | active scripts、训练/eval 边界、P2 脚本规划 | 准备运行或新增脚本时 |

## P1 / P0 归档与论文证据

| 文档 | 状态 | 用途 | 何时阅读 |
|---|---|---|---|
| `/data1/luyifei/drla/docs/cola_archive/README.md` | CoLA 线完整归档入口 | P0/P1/P2 CoLA 线的 canonical 结果、权重/代码/文档索引和复现边界 | 系统复现或整理 CoLA 线时先读 |
| `/data1/luyifei/drla/docs/cola_archive/P0_Adaptive_Halt_Reproducibility.md` | P0 复现归档 | decoder/probe/scorer 辅助 adaptive halt teacher 的协议、指标、artifact 和脚本入口 | 复现或引用 P0 riskcap04 teacher 时 |
| `/data1/luyifei/drla/docs/cola_archive/P1_Latent_Halt_Student_Reproducibility.md` | P1 复现归档 | LatentHaltStudent-v1 最优结果、locked risk audit、权重路径和泄漏边界 | 复现或引用 P1 早停判别器时 |
| `/data1/luyifei/drla/docs/cola_archive/CODE_AND_ARTIFACT_MANIFEST.md` | CoLA artifact manifest | 代码入口、summary/checkpoint/trace/report 路径、目录体量和复现 checklist | 查找权重、summary、CSV、脚本时 |
| `/data1/luyifei/drla/docs/cola_archive/P2_CoLA_Line_Freeze_Notes.md` | P2 CoLA 诊断冻结说明 | 为什么当前 CoLA MAS/interface 线冻结为诊断、哪些结果不可作为主 claim | 追溯 P2 CoLA communication 负结果和边界时 |
| `/data1/luyifei/drla/docs/p1_archive/P1_Final_Archive_and_P2_Plan_2026-05-27.md` | P1 归档 + P2 过渡记录 | P1 locked result、packet v1 substrate、阶段切换证据 | 复现 P1 或追溯 P2 起点时 |
| `/data1/luyifei/drla/docs/p1_archive/P1_Model_Comparison_Report_2026-05-27.md` | P1 paper-style 对比 | P1 主表、消融、泄漏审计、paper 级结论 | 写 P1/P2 background 或方法对比时 |
| `/data1/luyifei/drla/docs/p0_reports/cola_adaptive_halt_paper_report_zh.md` | 中文 canonical 报告草稿 | P0 adaptive halt / riskcap04 论文式证据 | 写安全早停背景或 P0 teacher 上界时 |
| `/data1/luyifei/drla/docs/p0_reports/cola_adaptive_halt_paper_report.md` | 英文旧草稿 stub | 指向中文 canonical 与瘦身前英文原文 | 需要英文表述参考时 |

## 历史设计与实验笔记

| 文档 | 状态 | 用途 | 何时阅读 |
|---|---|---|---|
| `/data1/luyifei/drla/docs/historical_design/DRLA_Implementation_Plan.md` | P0/P1 roadmap digest | official Cola + readiness/halt 路线的历史摘要 | 追溯 P0/P1 决策和旧 artifact 时 |
| `/data1/luyifei/drla/docs/p1_archive/P1_LatentHaltStudent_v1_Design_and_Distillation.md` | P1 设计 digest | LatentHaltStudent-v1 架构、蒸馏、主要正负结果 | 需要 P1 architecture 细节时 |
| `/data1/luyifei/drla/docs/p1_archive/P1_Progress_and_Literature_Synthesis_2026-05-27.md` | P1 文献/决策 digest | 2026-05-27 文献复盘和 P1 决策依据 | 追溯为何停止局部调参时 |
| `/data1/luyifei/drla/docs/historical_design/DRLA_Current_Framework_and_Experiment_Report.md` | 历史快照 digest | 2026-05-25 时的框架和实验状态摘要 | 只用于理解 P0/P1 过渡背景 |
| `/data1/luyifei/drla/docs/historical_design/Diffusion_Latent_Reasoning_Framework.md` | 长期背景设计 digest | 早期 DRLA latent reasoning 框架设想摘要 | 需要宏观架构背景时 |

## 工程运行

| 文档 | 状态 | 用途 | 何时阅读 |
|---|---|---|---|
| `/data1/luyifei/drla/docs/engineering/AGENT_CONTEXT_REFERENCE.md` | agent 详细上下文 | 从 AGENT.md 移出的路径、artifact、命令细节 | AGENT.md 信息不够时再读 |
| `/data1/luyifei/drla/docs/engineering/SWANLAB_TRACKING.md` | logging 规范 | SwanLab cloud/disabled 边界和 tracking helper 用法 | 新增训练脚本或排查 run 记录时 |
| `/data1/luyifei/drla/ENVIRONMENT.md` | 环境说明 | conda、cache、Python user base 策略 | 配环境或查 dependency/cache 规则时 |

## 当前不关注

| 文档 | 状态 | 用途 | 何时阅读 |
|---|---|---|---|
| `/data1/luyifei/drla/docs/parked/DRLA_Multiscale_Block_Halt_Design.md` | 不成熟 try / 当前不考虑 | 多尺度递归 block 与 sub-block halt 设想 | 暂不用于当前实验路线；只有重新讨论该方向时再读 |

## 瘦身前原文快照

| 目录 | 状态 | 用途 | 何时阅读 |
|---|---|---|---|
| `/data1/luyifei/drla/docs/full_history/2026-05-29_pre_slim/` | 只读历史快照 | 保存本轮瘦身前的长文档原文 | 需要核对被压缩掉的实验流水或旧设计细节时 |
