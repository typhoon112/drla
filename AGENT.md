# AGENT.md

DRLA 实验工程：以 official Cola VAE/DiT 作为 latent substrate。当前锁定主线是 P2 same-substrate Cola A -> Cola B latent communication，但必须先完成 true MAS benchmark/protocol validation，再做 CoLA substrate/interface adaptation，最后进入 CoLA TextMAS vs LatentMAS 主表；旧 official8 solver-to-solver message_only 只作为 channel diagnostic。不要把工作退回 legacy Stage B/C、GSM8K 小样本 overfit、自建 small prior，或 LoRA 刷 benchmark 分数。

**语言**：默认用中文回复用户；代码、路径、schema、命令和指标名保留英文。
**环境**：先运行 `source /data1/luyifei/drla/scripts/activate_conda.sh`。

---

## 硬性约束

1. **训练必须 GPU + SwanLab cloud**。任何会启动 optimizer、backward、更新权重或产生可比较训练结论的实验，包括 smoke training、probe、adapter/LoRA、readiness/halt、ablation，都必须使用 CUDA 且 `swanlab_mode=cloud` / `SWANLAB_MODE=cloud`。纯 eval、trace collection、frontier building、threshold sweep、packet build、audit、aggregation、`py_compile` 和数据格式检查必须 local-only，使用 `swanlab_mode=disabled`。

2. **训练产物不可省略**。所有训练必须写本地 `metrics.jsonl`，当前主线 valid 间隔 `<=10` step，并保存 `best_checkpoint.pt` 和 `last_checkpoint.pt`。只看 last checkpoint、离线 stdout 或 SwanLab summary 会漏掉关键失败形态。

3. **阶段性实验后先全局复盘**。继续下一轮优化前，至少比较 frontier、loss/mismatch rate、失败样本类型、跨 task/seed 复现性、decoder 依赖是否减少，以及结果是否改变主假设。项目已经多次出现“小改动局部更便宜但整体更不安全”的情况，不要只根据短期曲线连续调参。

4. **P2 benchmark 不能再 official8-only**。P2 主实验必须优先读 `/data1/luyifei/drla/docs/current/P2_Locked_Complete_Execution_Scheme_2026-06-01.md`。旧 `/data1/luyifei/drla/docs/current/P2_Next_Phase_Execution_Plan_2026-06-01.md`、`/data1/luyifei/drla/docs/current/P2_D4_Branch_Decision_Audit_2026-06-01.md` 和 `/data1/luyifei/drla/docs/current/P2_Branch_B_Execution_Plan_2026-06-01.md` 只保留阶段性历史与 gate 细节，不能覆盖 locked scheme。`official8` 只保留为 substrate/P1/channel diagnostic。2026-06-05，MuSiQue evidence-split QA v1 strict 已通过 capable text-agent calibration 和 locked held-out admission，Phase C benchmark/protocol validation 对该协议已完成。下一步是 Phase A：在同一 locked MuSiQue 协议上做 CoLA Single Solver capability gate 和 CoLA Role TextMAS capability gate；必要时做 task-format / role-interface adapter 或 LoRA。不要继续把 Qwen held-out score 当作 CoLA 结果，也不要跳过 CoLA gate 直接跑 P2 main table 或 latent fuser/receiver。若 ARC/GSM8K/GPQA/MedQA/EvalPlus 是非退让目标，才直接进入对应 substrate adaptation。后续 protocol repair 只能使用 calibration split；held-out 结果已经锁定，不能再为追分修改 prompt/parser/control definition。当前 official8 full prepared split 是 `batch_size=12`；official8 trace 结论不得混用 `bs20`、`bs12`、`bs1`。

5. **Gold/scorer/oracle 只用于离线 label/eval**。Official scorer correctness、gold answers、oracle block 不能成为 inference features。P0 的 decoder logits/text、`scored_prediction`、prediction-stability 等只能作为诊断、teacher label 或 text-channel baseline 的显式组成部分；最终在线 latent halt / latent communication 不能偷用这些字段。

6. **Trace protocol 是实验协议的一部分**。Trace run 不启动 optimizer，只允许 `swanlab_mode=disabled`；必须保留 model path、dataset version、task、seed、generation config、per-block latent/probe/answer/stability fields 和本地 `metrics.jsonl`。没有 trace 就无法重建 oracle frontier 或解释 halt error。

7. **P0/P1 不要被误删或重启**。P0 `joint-readiness riskcap04` 是 readiness 存在性证据、teacher-label 来源、safety/cost upper-bound 和 P1 诊断集。P1 locked evaluation 与 packet v1 substrate 已完成；不要继续堆零碎二分类 head，除非先读 P1 archive 并说明为什么要重新打开。

8. **P2 claim 只限 same-substrate latent communication，并且主协议是 role-conditioned MAS**。默认 baseline 是 `Planner -> Critic -> Refiner -> Solver` 或自然分工的 hierarchical experts。下游 role 可以看到原始问题 `q` 加上上游 text/latent message；这不是泄漏。泄漏是 scorer 或在线 receiver 看到 gold/scorer output、`selected_prediction`、或 Agent-A decoded replay tokens。不要声称 heterogeneous-agent latent communication 已解决。

9. **Agent A -> B 数据流不能绕过 B**。TextMAS 必须让 Agent A 的文本输出成为 Agent B / final Solver 的输入；LatentMAS 必须让 Agent A 的 latent packet/state 成为 Agent B / final Solver 的输入；scorer 只看 B 在 handoff 后生成的 final answer。`single_full_info` 只是 solvability control，不是 MAS 协议。direct-answer handoff、replay-visible text、legacy all-visible scoring 只能作为 diagnostic。

10. **CoLA 架构适合作为 substrate，但 frozen official 权重不自动适配新 benchmark**。Phase C 先用 capable text agents 验证 true MAS benchmark/protocol；Phase A 才训练/适配 CoLA task-format、role interface 或 latent interface；Phase E 才做同一 locked benchmark 上的 CoLA TextMAS vs LatentMAS 主表。不要因为 frozen CoLA 在新任务上低分就否定 latent communication。

11. **`DRLA_Multiscale_Block_Halt_Design.md` 是 parked try**。它只保留为历史想法，当前不纳入实验路线、脚本优先级或 P2 claim。

12. **不要污染共享 Python 环境**。依赖安装必须进入 `/data1/luyifei/drla/.conda/drla-mvp` 和项目内 cache。默认使用 `HF_XET_HIGH_PERFORMANCE=1`；Hugging Face 直连慢时再切 mirror，不要因为本地缺权重缩小科学实验。

13. **代码改动前使用 `karpathy-guidelines`**。写代码、审查或重构前读取 `/data1/luyifei/.codex/skills/karpathy-guidelines/SKILL.md`，保持改动小、假设显式、验证标准清楚。

---

## 当前流程

```text
official Cola trace
-> P0 decoder-probed readiness baseline
-> P1 decoder-as-teacher LatentHaltStudent-v1
-> P2 role-conditioned same-substrate agent-agent latent communication
```

---

## 最小命令

```bash
source /data1/luyifei/drla/scripts/activate_conda.sh
swanlab verify
```

---

## 只读入口

默认不要把 AGENT 当实验记录。需要更多上下文时从这里进入：

- `/data1/luyifei/drla/docs/DOCS_INDEX.md`：唯一文档导航入口。
- `/data1/luyifei/drla/docs/current/CURRENT_EXPERIMENT_STATUS.md`：当前 P1/P2 快照和下一步。
- `/data1/luyifei/drla/docs/current/P2_Locked_Complete_Execution_Scheme_2026-06-01.md`：当前最高优先级 P2 锁定执行方案，优先级高于旧 next-phase/Branch B 文档。
- `/data1/luyifei/drla/docs/current/P2_Latent_MAS_Communication_Implementation_Plan_2026-05-29.md`：当前 P2 canonical 实施方案。
- `/data1/luyifei/drla/docs/current/P2_Benchmark_and_Agent_Baseline_Redesign_2026-06-01.md`：P2 benchmark/agent baseline 路线修订。
- `/data1/luyifei/drla/docs/current/P2_Next_Phase_Execution_Plan_2026-06-01.md`：历史 P2-D3.1/Branch B 前执行纪律，已被 post-Family1 plan supersede。
- `/data1/luyifei/drla/docs/current/P2_D4_Branch_Decision_Audit_2026-06-01.md`：历史 prompt-only repair 失败后的分支审计，Branch B first 已执行并停止。
- `/data1/luyifei/drla/docs/current/P2_Branch_B_Execution_Plan_2026-06-01.md`：历史 Branch B Family 1 执行方案，不能作为当前默认路线。
- `/data1/luyifei/drla/docs/current/P2_Branch_B_Calibration_Report_2026-06-01.md`：Branch B official8-compatible calibration 结果。
- `/data1/luyifei/drla/docs/current/P2_Official8_Native_Alignment_Audit_2026-06-01.md`：official CoLA native prompt/eval 对齐审计与 Family 1 stop condition。
- `/data1/luyifei/drla/docs/current/P2_Post_Family1_Branch_Decision_Memo_2026-06-01.md`：Family 1 停止后的 Branch A/C/Family2 决策备忘录。
- `/data1/luyifei/drla/docs/current/P2_Post_Family1_Complete_Execution_Plan_2026-06-01.md`：Family 1 停止后的完整执行背景，已由 locked scheme 固化为默认路线。
- `/data1/luyifei/drla/docs/engineering/AGENT_CONTEXT_REFERENCE.md`：artifact 路径、历史 readout 和运行细节。
- `/data1/luyifei/drla/drla/scripts/README.md`：脚本入口、train/eval/audit 边界。
