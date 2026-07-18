# P3 Dream D5.5 后实验归档结论

本文是迁移后的最小研究上下文。它记录 D5.5 后做过什么、结论是什么，以及哪些路线
不应在没有新假设时重复。详细指标和生成记录仍可从 `outputs/p3_*` 下保留的 JSON/JSONL/CSV
文件追溯。

## 边界与当前结论

- D5.5 完成的是 internal split readiness policy audit，不是 held-out TextMAS claim。
- D6 将真实 evidence-agent suffix hidden tensor 按 D5 policy 选择步骤并封装成 packet；
  该步骤选择只是迁移 heuristic，不证明 evidence-agent readiness。
- D7 始终处于 receiver integration。没有任何 receiver 通过 locked held-out gate，
  当前不能进入 D8。
- V7 是 calibration 上最强的历史 receiver，只因它适合验证端到端接口而保留 checkpoint；
  不是因为它已经建立成功的 latent communication claim。

## 阶段结论与禁止重复项

| 阶段 | 做了什么 | 结论 | 没有新机制时不要重复 |
|---|---|---|---|
| D6 | full suffix tensor、D5 policy step selection、packet audit | packet schema 和防泄漏检查通过，但产生约 165 GiB full tensor | 不再扩 full-suffix trace；需要新数据时使用 compact selected-suffix |
| D7 V1 | MSE latent fuser | 未通过 corruption controls | 不要只优化重建/MSE |
| D7 V2 | contrastive fuser | 只证明 row-specific latent alignment，没有证明生成可用 | 不要把 alignment 当通信成功 |
| D7 V3–V5 | raw/soft prefix、layer-conditioned/corruption-aware receiver | raw/soft prefix 失败；V4 有 receiver prior；V5 压住 prior 但损伤 matched | 不要只加强 zero negative loss |
| D7 V6–V7 | reranker 与 V7 receiver | V7 calibration matched 超过 zero/no-message，但 shuffled margin 小，agent swap 几乎相同 | 不要从 calibration V7 直接跑 D8 |
| D7.5–D7.9 | text-hidden、adapter、teacher KL、candidate pool、interface audit | TextMAS hidden/teacher 仍未恢复 TextMAS；packet 经压缩、gate、后层注入后作用过弱 | 不要继续只调小 gate 或同一后层注入 |
| D7.10 | virtual message receiver | CE 可下降，但 generation 退化为 receiver prior | 不要把 teacher-forcing loss 当生成成功 |
| D7.11 | unlikelihood/logit/hidden contrast | zero margin 增强，但 matched generation 被一起破坏 | 不要单纯放大负样本目标 |
| D7.12 | balanced warmup/selection | 能保住少量 positive，但 matched 未超过 zero/shuffled/agent-swap | 不要继续同一 objective 微调权重 |
| D7.13 | 统一 control audit | V7 full200 是唯一 calibration primary paired CI 全正者，但 shuffled/token-F1 证据弱 | calibration 结果不得写成 held-out |
| D7.14 | V7 locked heldout800 | hard gate fail；matched=0.025、no_message=0.02375、zero=0.02875、shuffled=0.02375 | 不要复用 V7 做 held-out/D8 claim |
| D7.15 | split/generalization localization | 接口统计未明显恶化；主要是 nontrain generalization failure | 不要只做 calibration 追分 |
| D7.16 | train2000 compact + CE/margin | loss 更好，但 valid200 generation fail；matched=0.040，no_message=0.055 | 不要只加数据、步数或 CE |
| D7.17 | denoising sensitivity | matched 对 transfer token 的影响极小；vs shuffled 约 0.18% disagreement | 必须直接影响 inference-time token decisions |
| D7.18 | partial-denoising objective | matched CE/zero margin 改善，row binding 仍弱；valid50 hard gate fail | 不要把方向性改善写成 receiver 成功 |
| D7.19 | weighted row-binding objective | shuffled margin略升，但 generation matched/shuffled 全平，hard gate fail | 不要继续只给同一 CE/margin 加权 |

## 后续若重新开启 D7

必须提出不同于现有失败族的新假设，优先考虑：

1. 显式 row-identity architecture，而不是仅靠 shuffled-row margin；
2. 更强、更早或 trajectory-level 的 fusion/injection；
3. 与 Dream inference-time denoising decision 对齐的目标；
4. 先通过 nontrain calibration gate，再进行 locked held-out；
5. agent-swap 在同构、对称证据协议下只作 symmetry diagnostic，不能默认当硬负样本。

任何新方案都必须保持 decoder-free 在线接口，不能让 gold/scorer/oracle/decoded agent answer
进入 receiver 在线输入。

## 保留的可执行锚点

- Dream 模型：`models/Dream-v0-Instruct-7B/`
- D5 student：`outputs/p3_dream_readiness_students/dream_step_readiness_student_v1_full200_with_hidden_seed20260606_20260606/`
- D5.5 policy audit：`outputs/p3_dream_readiness_policy_eval/dream_step_readiness_student_v1_full200_with_hidden_policy_eval_20260606/`
- V7 checkpoint：`outputs/p3_dream_layer_receivers/dream_layer_receiver_v7_v4init_zeroshuf_textmas_matched200_seed20260607_20260607/`
- 自包含 fixture：`migration/post_d55_smoke/`
- 验证入口：`scripts/verify_d55_migration.py`

Smoke fixture 只保留一个 packet group，不能用于统计结论、重新训练或新的 benchmark claim。
