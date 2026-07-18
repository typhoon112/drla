# AGENT.md

DRLA 实验工程。当前默认主线是 **P3 Dream-DLM LatentMAS**：以
Dream-v0-Instruct-7B 同构双 agent 为 substrate，在 MuSiQue evidence-split QA
上研究 decoder-free latent readiness / halt 与 agent-to-agent latent
communication。CoLA P0/P1/P2 已归档为证据线，不能继续当作当前主线修补。

**语言**：默认用中文回复用户；代码、路径、schema、命令和指标名保留英文。
**环境**：先运行 `source /data1/luyifei/drla/scripts/activate_conda.sh`。
**更新**：2026-06-18。

---

## 硬性约束

1. **当前路线只认 P3 Dream**。新实验默认读取 P3 Dream design/implementation
   docs。不要退回 Qwen、ELF、legacy Stage B/C、GSM8K 小样本、CoLA adapter 修补、
   official8-only 或 LoRA 刷 benchmark 分数。若脚本调用 Qwen/CoLA/ELF，先确认它是
   历史归档、离线 teacher 诊断，还是当前路线污染。

2. **第一阶段是同构双 Dream agent**。Agent A/B 使用同一个
   Dream-v0-Instruct-7B，只允许 role prompt、private evidence 和 communication
   channel 不同。Dream-v0-Base-7B、异构模型、多模型 routing 只能作为后续诊断。

3. **主 benchmark 是 MuSiQue evidence-split QA**。Agent A/B 必须持有互补私有
   evidence，final receiver/solver 必须依赖上游消息。2Wiki 只做备选；Hotpot、
   official8、GSM8K、ARC/GPQA 等不能默认替代主线。

4. **Agent A -> B/receiver 数据流不能绕过 B**。TextMAS 必须让 Agent A 文本输出成为
   Agent B 或 final receiver 的输入；LatentMAS 必须让 Agent A latent packet/state
   成为 Agent B 或 receiver 的输入；scorer 只看 B/receiver 生成的 final answer。
   禁止把 Agent A decoded answer、replay tokens 或 scorer-visible helper text 直接拼进
   最终答案。

5. **Gold/scorer/oracle/decoder 只用于离线 label/eval**。Gold answers、official scorer、
   oracle frontier、decoder logits/text、prediction-stability、scored_prediction 可以
   训练 teacher label 或做诊断；最终在线 readiness / latent communication 不能偷用这些
   字段。训练阶段可以学习 latent 与 decoder 信号的映射，但推理接口必须 decoder-free。

6. **深度学习训练必须 GPU + SwanLab cloud**。凡是启动 optimizer/backward、更新权重或
   产生可比较训练结论的实验，包括 smoke training、probe、adapter/LoRA、
   readiness/halt、receiver/fuser、reranker、ablation，都必须使用 CUDA 且
   `swanlab_mode=cloud` / `SWANLAB_MODE=cloud`。纯 eval、trace collection、frontier
   building、threshold sweep、packet build、audit、aggregation、`py_compile` 和数据格式
   检查必须 local-only，不要上云制造空曲线。

7. **训练产物不可省略**。所有训练必须写本地 `metrics.jsonl`，当前主线
   `valid_interval <= 10`，并保存 `best_checkpoint.pt` 和 `last_checkpoint.pt`。
   不要只看 last checkpoint、stdout 或 SwanLab summary。

8. **Calibration / held-out 边界不能破**。Prompt repair、threshold tuning、adapter
   selection、receiver/fuser/reranker selection 只能用 calibration 或 train split。
   Held-out 只能做 locked evaluation，不能为了追分改 prompt、parser、control
   definition、candidate source 或 threshold。

9. **D2 maxctx4096 protocol lock 当前有效**。TextMAS 对照必须继承
   `dream_textmas_protocol_lock_calibration_full200_maxctx4096_20260606`。旧 2048
   held-out 因 `single_full_info` context overflow 只保留为诊断。任何新 held-out、
   更长 prompt 或新 benchmark 先做 static context audit。

10. **D5/D6 结论不能过度解释**。D5 readiness student 只认 corrected frontier with
    hidden 的版本；D5.5 是 internal split policy audit，不是 held-out TextMAS claim。
    D6 packets 是真实 evidence-agent suffix tensors，但 D5 policy 迁移到 evidence-agent
    trace 只是 packet step-selection heuristic，不是 evidence-agent readiness 证明。

11. **D7 进入 D8 前必须通过 packet-specific receiver gate**。已知结论：
    V1 MSE fuser 没过 corruption controls；V2 contrastive fuser 只证明 row-specific
    latent alignment；raw latent-prefix 与 V3 soft-prefix generation 未通过；V4
    layer-conditioned receiver 有 receiver-prior 泄漏；V5 corruption-aware receiver 压住
    zero prior 但损伤 matched generation；V6 answer-reranker 没学出 matched 超过
    zero/shuffled 的 packet-specific 选择；V7 是当前最强 receiver，matched 明显超过
    zero/no-message、略高于 shuffled，但 matched-vs-shuffled margin 小且 agent_swap
    与 matched 基本持平；D7.5 text-encoded packet 诊断显示 same-model text hidden
    直接注入 V7 receiver 仍远低于 TextMAS；D7.6 text-packet adapter 训练能改善
    loss/token-F1 但 20-row primary 仍只有 0.05，远低于 same-row TextMAS 0.40；
    D7.7 TextMAS teacher KL 让 loss-level margin 更强，但 generation matched 仍只有
    0.05，且不超过 zero/agent_swap；D7.8 V7 matched-channel candidate-pool
    诊断显示 8-candidate oracle primary matched=0.15、zero=0.15、agent_swap=0.05，
    matched 只有 1 条独有 oracle-correct row；D7.9 interface audit 显示 D6
    packet 与 TextMAS prompt hidden 有弱到中等相关性，但经 256-d memory 和约
    0.12 gate 的 cross-attention 后，layer27 delta/hidden norm ratio 仅约 0.047；
    D7.10 text-interface virtual message receiver 能把 CE 学到 matched≈2.54，但
    20-row generation matched=zero=shuffled=agent_swap=0.05，唯一正确 row 是所有
    virtual-prefix 条件共享的 receiver prior；D7.11 加入 corrupted unlikelihood /
    logit contrast / hidden contrast 后，zero margin 被拉到 valid≈6.78，但 best/last
    generation matched primary 都是 0.00，说明负样本目标压住 prior 的同时伤掉了
    matched generation；D7.12 balanced warmup/selection 从 D7.10 best 初始化，best
    checkpoint 恢复到 matched=0.05 但 zero/shuffled/agent_swap 也都是 0.05，last
    checkpoint 把 zero/shuffled 压到 0.00 但 matched=agent_swap=0.05，正确行仍是同一
    行；D7.13 receiver-control audit 统一复算 V7 与 D7.10-D7.12 后显示，若 hard
    controls 定义为 no_message/zero/shuffled-row，V7 full200 是唯一 primary paired
    CI 全部为正的 receiver，但 matched-vs-shuffled margin 很小且 token-F1 CI 仍跨 0；
    D7.14 已补齐 held-out suffix-tensor trace 与 held-out D6 packet，并完成 V7 locked
    held-out800 eval/audit：hard_gate_pass=false，matched primary=0.02500，
    no_message=0.02375，zero=0.02875，shuffled_row=0.02375。不要把 V7 full200
    calibration controls 写成 held-out，也不要从 V7 直接跑 D8 主表。D7.15 failure
    localization 显示 held-out packet/TextMAS hidden 接口统计没有明显变差；V7 full200
    hard gate 主要由 train split 支撑，valid/test 不通过。D7.16 train2000 compact
    selected-suffix receiver 从 V7 初始化、valid_interval=10、SwanLab cloud 训练完成；
    它把 matched CE 和 zero/shuffled margin 学得更强，但 checkpoint-defined valid200
    generation gate 仍失败：matched=0.040、no_message=0.055、zero=0.045、
    shuffled_row=0.035、agent_swap=0.040。D7.17 denoising sensitivity audit 在
    valid50 shared-state 上进一步显示 matched 与 no_message 的 transfer-token
    disagreement 只有 0.61%，与 shuffled_row 只有 0.18%，与 agent_swap 只有 0.12%。
    不要从 D7.16/D7.17 进入 D8 或 held-out。

12. **下一步默认方向**。先做 text-conditioned hidden state 与 D6 latent packet 的
    接口/分布审计，弄清 TextMAS 经过 AgentB tokenizer/embedding/DIT 后可用，而直接
    packet 注入不可用的具体断点。D7.9 已定位到当前 receiver 的压缩/门控/后层注入
    可能过弱；D7.10 说明仅做 virtual message prefix 仍会退化为 receiver prior；
    D7.11 说明只加强 negative packet-specific loss 会牺牲 matched generation；D7.12
    说明 balanced objective 能保住 positive，但仍未超过 zero/shuffled/agent-swap 的
    generation gate。当前 homogeneous evidence-agent 协议下，agent_swap 不一定是 strict
    corruption，因为交换 A/B 仍保留同一证据集合；除非后续协议引入非对称角色，否则
    zero/shuffled-row 是硬负样本，agent_swap 作为 symmetry/role diagnostic 报告。D7.14
    说明 V7 的 calibration packet signal 没有迁移到 held-out；下一步应诊断
    calibration-to-heldout 分布迁移、zero receiver prior、packet step-selection
    heuristic 和 layer injection 强度，或设计明确非对称角色后再把 agent_swap 当硬负样本。
    D7.15 已将主要问题定位为 nontrain generalization failure；D7.16/D7.17 进一步说明
    teacher-forcing CE / corruption margin 能学到，但 sampled Dream denoising generation
    没有把 matched packet 稳定当作答案来源，也没有显著改变每步 top/transfer token。
    D7.18-v1 partial-denoising / decision-margin trainer 已完成 screen200：valid
    matched_ce 从 D7.16 的 2.56 降到 1.73，zero_gold_margin 到 2.83，但
    shuffled_row_gold_margin 仍只有 0.03；valid50 sensitivity 显示 no_message
    transfer disagreement 升到 1.09%，shuffled_row 反而只有 0.13%；valid50
    generation hard_gate=false，matched=0.12、zero=0.12、shuffled=0.10、
    no_message=0.08。D7.19 在 D7.18 脚本中加入 per-control loss weights 和
    `selection_mode=row_binding`，从 D7.18 best 初始化、valid_interval=10、SwanLab
    cloud screen200：valid shuffled_row_gold_margin 提到 0.0667，但 valid50
    sensitivity 仅把 matched-vs-shuffled transfer disagreement 提到 0.00167；
    generation hard_gate=false，matched=0.10、zero=0.12、shuffled=0.10、
    no_message=0.08，matched-vs-shuffled 50/50 全平。任何新 receiver 必须先过
    nontrain calibration gate，再考虑
    locked held-out。当前没有已通过 held-out gate 的 D8 receiver；V7 是历史
    calibration aggregate 最强，D7.18-v1 是“partial-denoising 目标有效但
    row-specific binding 不足”的诊断，D7.19 是“单纯加权 row-binding loss 仍不能
    转成生成时 packet-specific answer”的诊断；二者都不是 D8 receiver。
    receiver-side selection/reranking 仍可做，但 candidate source 必须是在线 matched latent
    channel 可产生的候选。候选或监督信号不能来自 receiver 在线不可见的
    private evidence/gold/scorer 字段，也不能把 corrupted-control outputs 当作在线 matched
    候选。

13. **阶段性实验后先全局复盘**。继续下一轮优化前，比较 frontier、loss/mismatch rate、
    失败样本类型、跨 split/seed 泛化、decoder 依赖是否减少、通信是否真的经过 B，以及
    结果是否改变主假设。必要时重新查论文和当前文献，不要只根据短期曲线连续调参。

14. **CoLA 线已归档但不能丢**。P0/P1/P2 CoLA 是 readiness、teacher/student、
    receiver-only 边界和失败模式证据；不要重启为当前主线，也不要删除权重、config、
    metrics 或文档。追溯入口是 `/data1/luyifei/drla/docs/cola_archive/README.md`。

15. **不要污染共享 Python 环境**。依赖安装必须进入
    `/data1/luyifei/drla/.conda/drla-mvp` 和项目内 cache。默认使用
    `HF_XET_HIGH_PERFORMANCE=1`；Hugging Face 直连慢时再切 mirror，不要因为本地缺权重
    缩小科学实验。

16. **代码改动前使用 `karpathy-guidelines`**。写代码、审查或重构前读取
    `/data1/luyifei/.codex/skills/karpathy-guidelines/SKILL.md`，保持改动小、假设显式、
    验证标准清楚。

## 当前锚点

- D2 TextMAS capability gate 已通过；正式协议是 maxctx4096 locked held-out800。
- D3/D4 full200 trace/frontier 已完成；D5 readiness student 与 D5.5 internal online
  halt calibration 已完成。
- D6 `p3_dream_packet_v1_suffix_tensor` packets 已完成并通过 packet audit。
- D7 仍在 receiver integration 阶段；V7 仍是当前 calibration generation 最强 raw receiver。
  D7.5/D7.6/D7.7 证明 text hidden、text-hidden adapter、teacher-forcing
  TextMAS KL 都还没有恢复 TextMAS。matched 必须稳定超过 zero/shuffled/agent-swap，
  并接近 TextMAS channel，才能进入 receiver-only LatentMAS 或 TextMAS-vs-LatentMAS
  主表。
- D7.14 preflight artifact:
  `/data1/luyifei/drla/outputs/p3_dream_heldout_packet_preflights/dream_heldout_packet_readiness_preflight_20260617`。
- D7.14 held-out artifacts:
  `/data1/luyifei/drla/outputs/p3_dream_traces/musique_heldout_trace_textmas_matched800_steps64_stride4_hidden_tensor_merged_20260617`，
  `/data1/luyifei/drla/outputs/p3_dream_latent_packets/dream_textmas_heldout800_agent_ab_suffix_tensor_packets_v1_20260617`，
  `/data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/dream_layer_receiver_v7_heldout800_merged_20260617`，
  `/data1/luyifei/drla/outputs/p3_dream_receiver_control_audits/dream_receiver_generation_control_audit_v7_heldout800_20260617`。
  结论：locked held-out800 hard gate fail；matched 与 no_message/shuffled_row 基本打平且低于 zero，不能 claim latent communication success。
- D7.15 failure-localization artifact:
  `/data1/luyifei/drla/outputs/p3_dream_receiver_generalization_audits/dream_receiver_v7_d714_failure_localization_20260617`。
  结论：held-out interface stats 不明显坏于 calibration；V7 full200 hard gate 是 train-dominated，valid/test fail。
- D7.16 compact selected-suffix substrate:
  `selected_suffix_tensor` trace mode 已通过 smoke1、queue smoke1 和 validdiag50。
  它只保存 D5 policy 实际选中的 evidence-agent suffix tensor，避免 full
  `suffix_tensor` 为每个 step 保存所有 tensor 造成磁盘爆炸。validdiag50 artifact:
  `/data1/luyifei/drla/outputs/p3_dream_traces/musique_validdiag50_trace_textmas_matched_selected_suffix_tensor_merged_20260617`，
  packet artifact:
  `/data1/luyifei/drla/outputs/p3_dream_latent_packets/dream_textmas_validdiag50_selected_suffix_tensor_packets_20260617`。
  结果：50 groups / 100 packets，missing refs/traces=0，forbidden decoder/gold/scorer
  packet key hits=0。后续大规模 receiver substrate 默认使用该 compact mode；
  full suffix tensor 只作为小样本诊断，不再默认扩到 2000+。
- D7.16 train2000 compact substrate 已完成：
  `/data1/luyifei/drla/outputs/p3_dream_traces/musique_train2000_trace_textmas_matched_selected_suffix_tensor_merged_20260617`
  和
  `/data1/luyifei/drla/outputs/p3_dream_latent_packets/dream_textmas_train2000_selected_suffix_tensor_packets_20260617`。
  结果：2000 rows / 6000 trace calls，2000 groups / 4000 packets，missing
  refs/traces=0，forbidden packet key hits=0，mean selected step=35.31025。
- D7.16 train2000 receiver 训练与 valid gate 已完成：
  `/data1/luyifei/drla/outputs/p3_dream_layer_receivers/dream_layer_receiver_d716_train2000_v7init_zeroshuf_seed20260617`，
  SwanLab run `7134z0gui6w8jek33rdt0`。best_step=600，best/last checkpoint
  均已保存，valid_interval=10。valid loss-level matched_ce=2.5633，
  zero_margin=2.8676，shuffled_margin=0.0381；但 valid200 generation gate
  失败，run 在
  `/data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/dream_layer_receiver_d716_train2000_v7init_zeroshuf_valid200_20260617`，
  audit 在
  `/data1/luyifei/drla/outputs/p3_dream_receiver_control_audits/dream_receiver_generation_control_audit_d716_train2000_valid200_20260617`。
  matched primary=0.040，no_message=0.055，zero=0.045，shuffled_row=0.035。
  prediction-similarity audit 显示 matched 与 shuffled_row 有 58% 完全同答、
  与 agent_swap 有 70% 完全同答，primary score 与 agent_swap 100% 相同。
  下一步不能只加数据或步数；必须改成 inference-aligned receiver 目标/注入机制，
  让 matched packet 在 denoising generation 中成为真实答案来源。
- D7.17 denoising sensitivity audit 已完成：
  `/data1/luyifei/drla/outputs/p3_dream_receiver_sensitivity_audits/dream_receiver_d716_valid50_steps64_max128_20260618`。
  结果：valid50 shared-state、64 step、128 token、12800 step-control records。
  matched_shared_state primary=0.02；matched vs no_message top1_disagree=0.00819、
  transfer_disagree=0.00609；matched vs zero top1_disagree=0.00820、
  transfer_disagree=0.00703；matched vs shuffled_row top1_disagree=0.00270、
  transfer_disagree=0.00182；matched vs agent_swap top1_disagree=0.00161、
  transfer_disagree=0.00125。结论：当前 packet 通道没有显著改变 Dream denoising
  的写 token 决策；下一步必须让训练目标/注入机制直接作用于 inference-time
  denoising decisions。
- D7.18-v1 denoising-aligned screen200 已完成：
  `/data1/luyifei/drla/outputs/p3_dream_layer_receivers/dream_layer_receiver_d718_screen200_denoising_aligned_seed20260618`，
  SwanLab run `5f7ynp5opf6j61cqpbxyy`。best_step=200，best/last checkpoint 已保存。
  valid matched_ce=1.7278，zero_gold_margin=2.8304，但
  shuffled_row_gold_margin=0.0300。sensitivity artifact:
  `/data1/luyifei/drla/outputs/p3_dream_receiver_sensitivity_audits/dream_receiver_d718_screen200_valid50_steps64_max128_20260618`；
  matched vs no_message transfer_disagree=0.0109，vs shuffled_row=0.0013。
  valid50 generation artifact:
  `/data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/dream_layer_receiver_d718_screen200_valid50_20260618`；
  audit:
  `/data1/luyifei/drla/outputs/p3_dream_receiver_control_audits/dream_receiver_generation_control_audit_d718_screen200_valid50_20260618`。
  hard_gate_pass=false；matched primary=0.12、no_message=0.08、zero=0.12、
  shuffled_row=0.10。不要从 D7.18-v1 跑 held-out 或 D8；下一步应把
  shuffled-row / row-specific binding 设为主目标，而不是继续平均 hard margin。
- D7.19 row-binding weighted screen200 已完成：
  `/data1/luyifei/drla/outputs/p3_dream_layer_receivers/dream_layer_receiver_d719_screen200_row_binding_seed20260618`，
  SwanLab run `ub06nr5p8ddq3p2fgdenw`。best_step=200，best/last checkpoint 已保存。
  valid matched_ce=1.7382，zero_gold_margin=2.9777，shuffled_row_gold_margin=0.0667。
  sensitivity artifact:
  `/data1/luyifei/drla/outputs/p3_dream_receiver_sensitivity_audits/dream_receiver_d719_screen200_valid50_steps64_max128_20260618`；
  matched vs no_message transfer_disagree=0.0108，vs shuffled_row=0.0017。
  valid50 generation artifact:
  `/data1/luyifei/drla/outputs/p3_dream_layer_receiver_runs/dream_layer_receiver_d719_screen200_valid50_20260618`；
  audit:
  `/data1/luyifei/drla/outputs/p3_dream_receiver_control_audits/dream_receiver_generation_control_audit_d719_screen200_valid50_20260618`。
  hard_gate_pass=false；matched primary=0.10、no_message=0.08、zero=0.12、
  shuffled_row=0.10、agent_swap=0.10。matched-vs-shuffled primary_delta=0.00，
  50/50 ties；matched-vs-zero primary_delta=-0.02。不要从 D7.19 跑 held-out 或
  D8；下一步不能只继续加权同一 CE/margin objective，应先做 row-identity
  architecture / stronger fusion-injection / trajectory-level objective 设计复盘。

## 文档入口

默认不要把 AGENT 当实验记录。需要更多上下文时从这里进入：

- `/data1/luyifei/drla/docs/DOCS_INDEX.md`：文档导航入口。
- `/data1/luyifei/drla/docs/current/CURRENT_EXPERIMENT_STATUS.md`：当前状态快照和 artifact 路径。
- `/data1/luyifei/drla/docs/current/P3_Dream_DLM_Latent_MAS_Experiment_Design_2026-06-06.md`：P3 Dream 科学问题、模型、benchmark、对照和防泄漏边界。
- `/data1/luyifei/drla/docs/current/P3_Dream_DLM_Latent_MAS_Implementation_Plan_2026-06-06.md`：P3 Dream D0-D8 执行阶段、训练/评估边界和验收标准。
- `/data1/luyifei/drla/docs/cola_archive/README.md`：CoLA P0/P1/P2 完整归档入口。
- `/data1/luyifei/drla/docs/engineering/AGENT_CONTEXT_REFERENCE.md`：长 artifact 清单、历史 readout 和运行细节。
- `/data1/luyifei/drla/drla/scripts/README.md`：脚本入口、train/eval/audit 边界。
