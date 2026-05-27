# AGENT.md

DRLA 实验工程：以 official Cola VAE/DiT 作为 latent substrate，研究 block-level answer-readiness / halt；不要把主线退回从零训练小 prior 或用 LoRA 刷 benchmark 分数。

**语言**：默认用中文回复用户。代码、配置、命令和指标名保持英文。
**环境**：使用 `source /data1/luyifei/drla/scripts/activate_conda.sh`。

---

## 硬性约束

**最高优先级：所有深度学习训练实验必须 SwanLab 上云，并且必须使用 CUDA/GPU。** 任何会启动 optimizer、更新模型参数或产生可比较训练结论的实验，包括 smoke training、probe、readiness/halt、LoRA/adapter、微调和 ablation，都必须使用 `swanlab_mode=cloud` / `SWANLAB_MODE=cloud`，并确认 resolved device 不是 CPU。纯 eval、threshold sweep、trace collection、frontier building、aggregation、`py_compile` 和数据格式检查不应使用 SwanLab；这些脚本代码层只接受 `swanlab_mode=disabled`，必须写本地 artifact、`metrics.jsonl` 和 `summary.json`。本地分析脚本可以因为阈值扫描/指标聚合出现高 CPU 占用，但这不是训练。

**高优先级：每个阶段性优化实验结束后必须先做全局复盘，再决定下一步。** 复盘至少比较 P0/P1 frontier、loss/mismatch rate、失败样本类型、跨任务/跨 seed 复现性、decoder 依赖是否减少，以及该结果是否改变主假设。遇到架构、训练策略或 latent reasoning 机理不确定时，先自行查阅论文和网络资料形成证据，再继续实验或向用户确认；不要只根据短期曲线和单点指标连续调参。
原因：本项目已经多次出现“小改动局部更便宜但整体更不安全”的情况，缺少广角复盘会陷入局部最优。

1. **主线是 official Cola + readiness/halt**。主评测只用 Cola 官方 8 个任务：`lambada, mmlu, obqa, hellaswag, race, siqa, squad, story_cloze`。GSM8K 只允许作为 OOD/math diagnostic。
   原因：小样本 GSM8K / overfit 结果已经误导过架构判断。

2. **保留 Phase P0 decoder-probed 成果**。`joint-readiness riskcap04` 以及相关 decoder-dependent 脚本、summary、checkpoint、SwanLab run、样本诊断，是前一阶段核心结果，不是废弃工作。
   原因：P0 是 readiness 存在性证据、teacher-label 来源、safety/cost upper bound 和 P1 诊断集。

3. **最终目标是 agent latent communication，不是 decoder-probed halt**。P0 可以把 decoder logits/text 用作诊断和 teacher label；最终在线 halt 不能依赖 decoder outputs、`scored_prediction`、prediction-stability、gold answers 或 official correctness。
   原因：否则会把“通过 decoder 看见 readiness”误报成“agent 可以在 latent space 里通信和早停”。

4. **P1 模型是 `decoder-as-teacher, LatentHaltStudent-v1`**。首版默认：R16 slot standardization/LayerNorm + Linear 到 `d_model=64`，每个 block 加 process token，1 层 intra-block self-attention，PMA K=4 + explicit `last_slot`，2 层 causal inter-block Transformer，task-specific readout queries。
   原因：raw latent 二分类和 mean pooling 容易学到长度/任务捷径，或抹掉局部 stop/completion evidence。

5. **P1 必须做结构消融与校准复盘**。已完成的负结果包括 `all_tokens`、`pma1`、`mean_max`、`d32_pma4`、`d128_pma4`、`stabilityw2`、`no_block_budget`、简单 `film` gating、只改 `per_task` 校准、SQuAD 单任务否决的 `last_process_query` readout context、只加 `completion_risk` 辅助 head 的 SQuAD 诊断，以及 MMLU-focused `contentfulw1`、`empty_answer_risk`、`answer_format_risk` 训练诊断。当前公平口径：`answer_identity_action + completion_risk` 为 `47` losses / `617` mismatches / `1.742/4` blocks；`answer_identity_halt + completion_risk` 为 `31` / `699` / `1.824/4`。`trajectory_token + answer_identity_action + completion_risk` 为 `41` / `806` / `1.711/4`；继续加入 `answer_identity_stability` teacher head 后，full official8 3-seed LOTO / 5 calibration subseeds 为 `4` losses / `606` mismatches / `1.812/4` blocks，是当前 P1 student-only 低 loss/cost frontier 的最好点，但 mismatch 仍高于 v2 cost-limited，且 Wilson risk-control 仍无法认证完整 folds。再加入 `empty_answer_risk` head 是负结果：`24` / `623` / `1.829/4`，失败从 empty-answer 转成 prefix/continuation。learned action->halt gate v2 的 fair source-valid cost 为 `10` / `315` / `2.220/4`，backfilled source-cost-limited 为 `10` / `465` / `1.859/4`，safety 为 `0` / `130` / `2.722/4`。LAMBADA `utility_mse`、raw utility-score threshold、`utility_pairwise`、`utility_pairwise --utility-mismatch-penalty 1.0`、`utility_soft_bce` temperature `1.0/0.1`、2026-05-27 official8 `decomposed_expected_utility`、post-hoc scalar utility-weight calibration、constrained head-threshold selector、source-task-robust scalar calibration 都没有明确超过 v2 的 deployable source-valid cost-limited 口径。结论：下一步应学习更同构的 prefix/continuation answer-identity risk 或改 calibration 协议；不要继续叠加零碎二分类辅助头。
   原因：架构选择不能只靠直觉定。

6. **EOS/im_end 不是 correctness**。EOS/im_end 只能说明 decoder 倾向结束文本生成；halt 还必须看 stability、future gain 和 non-gold verifier/proxy signals。
   原因：否则错误但已结束的答案会显得很安全。

7. **Gold scorer 只用于 label/eval**。Official scorer correctness、gold answers、oracle block 只能做离线 label/metric，不能做 inference features。
   原因：label leakage 会让 valid/test 结果失效。

8. **Trace protocol 是实验的一部分**。Trace run 必须保留 model path、dataset version、task、seed、generation config、per-block latent/probe/answer/stability fields 和本地 `metrics.jsonl`；trace 不启动 optimizer，只允许 `swanlab_mode=disabled`。
   原因：没有 trace 就无法重建 oracle frontier 或解释 halt error。

9. **Batch size 是协议参数**。主 trace 结论不得混用 `bs20`、`bs12`、`bs1`；当前 full prepared split 是 `batch_size=12`。
   原因：2026-05-24 诊断发现 MMLU 输出会随 batch size 改变。

10. **训练日志和 checkpoint 也不可省略**。所有深度学习训练必须同时写本地 `metrics.jsonl`，valid 间隔 `<=100` step，并保存 `best_checkpoint.pt` 和 `last_checkpoint.pt`。
    原因：训练过程动态、valid 最优点和失败形态都必须可追溯；只看 last checkpoint、离线日志或 SwanLab summary 会错过关键证据。

11. **readiness/halt 闭环完成前不要做 multi-agent**。Multi-agent latent communication 必须在 official baseline、trace、oracle frontier、readiness model、adaptive halt frontier 之后。
    原因：否则 prior、decoder、halt、communication 的失败会混在一起。

12. **不要污染共享 Python 环境**。通过 activation script 使用 `/data1/luyifei/drla/.conda/drla-mvp`；依赖安装走该环境和项目内 cache。
    原因：共享环境漂移会破坏复现。

13. **HF 下载策略**。默认用 `HF_XET_HIGH_PERFORMANCE=1`；Hugging Face 直连慢时再用 mirror。不要因为本地缺权重而缩小实验。
    原因：网络/下载问题不是科学约束。

14. **写代码时使用 `karpathy-guidelines`**。代码编辑、审查或重构前，读取 `/data1/luyifei/.codex/skills/karpathy-guidelines/SKILL.md`。
    原因：本项目容易被过度抽象和未验证改动拖偏。

---

## 主线流程

```text
official Cola baseline
-> block-wise rollout trace
-> per-block decoder probe
-> oracle readiness frontier
-> Phase P0 decoder-probed readiness baseline
-> Phase P1 LatentHaltStudent-v1
-> adaptive halt accuracy-cost frontier
```

已归档的 Stage/GSM8K/custom-prior 工作只用于复现：

```text
/data1/luyifei/drla/archive/2026-05-24-legacy-stagebc-code
/data1/luyifei/drla/archive/2026-05-24-legacy-stagebc-artifacts
```

---

## 常用命令

```bash
source /data1/luyifei/drla/scripts/activate_conda.sh
python -m pip check
swanlab verify
```

---

## 文档索引

- `/data1/luyifei/drla/docs/AGENT_CONTEXT_REFERENCE.md`：从 AGENT 移出的运行细节、artifact 路径、当前 readout 和常用命令。
- `/data1/luyifei/drla/docs/DRLA_Implementation_Plan.md`：当前研究计划、P0/P1 协议、标签、里程碑和风险。
- `/data1/luyifei/drla/docs/DRLA_Current_Framework_and_Experiment_Report.md`：当前框架与实验进展报告。
- `/data1/luyifei/drla/docs/P1_Progress_and_Literature_Synthesis_2026-05-27.md`：2026-05-27 P1 进展、文献复盘和 decomposed expected-utility official8 复核。
- `/data1/luyifei/drla/docs/CURRENT_EXPERIMENT_STATUS.md`：active scripts、最新 artifact、当前结果和下一组实验。
- `/data1/luyifei/drla/docs/Diffusion_Latent_Reasoning_Framework.md`：架构背景，尤其 block-level answer-enough halt。
- `/data1/luyifei/drla/docs/SWANLAB_TRACKING.md`：SwanLab logging 和 resume 约定。
- `/data1/luyifei/drla/ENVIRONMENT.md`：环境、cache 和安装策略。
- `/data1/luyifei/drla/drla/scripts/README.md`：脚本入口和 smoke commands。
