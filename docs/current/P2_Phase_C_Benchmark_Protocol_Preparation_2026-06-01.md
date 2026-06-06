# P2 Phase C Benchmark / Protocol 准备方案

更新日期：2026-06-01

> 状态：Phase C 安全准备文档。当前 locked scheme 已将 Phase C 设为下一条
> 默认路线，但本文本身不跑模型、不触碰 held-out、不训练、不创建 SwanLab run。它把
> `/data1/luyifei/drla/docs/current/P2_Post_Family1_Complete_Execution_Plan_2026-06-01.md`
> 中允许的 manifest/scorer 草案具体化，并服务于
> `/data1/luyifei/drla/docs/current/P2_Locked_Complete_Execution_Scheme_2026-06-01.md`。

## 1. 为什么 Phase C 必要

当前 P2 不能继续在普通单问答上硬拆 multi-agent，也不能在 frozen official
CoLA base solver 接近 floor 时比较 latent vs text。Phase C 的目的不是证明
CoLA latent communication，而是先证明：

```text
1. 任务确实需要 agent-to-agent communication。
2. capable text agents 在该任务上能形成稳定 Single / Role TextMAS baseline。
3. no-message / wrong-message / shuffled-message ablation 会造成可测下降。
4. scorer/parser 稳定，且没有 gold/scorer/selected_prediction 泄漏。
```

只有这些成立，后续 Phase A 才有意义：

```text
把锁定 benchmark 接回 CoLA substrate/interface adaptation
-> 再做 CoLA TextMAS vs LatentMAS 主比较
```

## 2. 文献依据

Silo-Bench:

```text
https://arxiv.org/abs/2603.01045

核心启发：
  评估 distributed coordination 时，不能只看 agent 是否交换信息；
  要看它们是否能把 distributed state 集成为正确答案。

对 DRLA 的含义：
  Phase C 任务必须包含 no-message / wrong-message / shuffled-message ablation，
  并显式度量 integration failure，而不是只记录对话是否看起来合理。
```

CRAFT:

```text
https://arxiv.org/abs/2603.25268

核心启发：
  strict partial information 是评估 communication 的好设置；
  failure taxonomy 应拆成 grounding / belief modeling / pragmatic
  communication 等可解释类别。

对 DRLA 的含义：
  Phase C 样本应给不同 agent 不完整但互补的 private observations；
  失败分析不能只给 accuracy，还要记录缺信息、误整合、格式失败等类型。
```

CoSMAC:

```text
https://openreview.net/forum?id=yGzAhl1o4i

核心启发：
  communication and coordination 需要放在共享目标、局部观测、协作行动或
  决策场景中评估。

对 DRLA 的含义：
  若任务没有角色分工或局部信息约束，就不能支撑 true MAS claim。
```

LatentMAS:

```text
https://arxiv.org/abs/2511.20639

核心启发：
  latent channel 的比较对象是 strong single / text MAS baseline，并且最终只
  评价下游 agent 的输出。

对 DRLA 的含义：
  Phase C 需要先把 text MAS baseline 做稳；Phase E 才替换通信介质。
```

Interlat / DiffMAS:

```text
Interlat local source:
  /data1/luyifei/latent_reasoning_papers/agent_comm_downloads/
  2511.09149_Interlat/main.tex

DiffMAS:
  https://arxiv.org/abs/2604.21794

核心启发：
  latent receiver 通常需要 adapter、curriculum、matched-vs-mismatched
  separation 或 end-to-end communication optimization。

对 DRLA 的含义：
  Phase A 若接回 CoLA，不应假设 raw latent 直接塞给 Agent B 就天然可用；
  adapter/fuser 属于明确的 adapted-CoLA claim。
```

## 3. Phase C 候选任务族

### C-Family 1: evidence-split multi-hop QA

目标：

```text
不同 agent 只看到互补证据片段。
最终 Solver 必须整合多个 agent 的信息才能回答。
```

候选数据源：

```text
HotpotQA / 2WikiMultiHopQA / MuSiQue 风格 multi-hop QA。
```

推荐样本构造：

```text
public:
  question only

Agent A private view:
  evidence shard A

Agent B private view:
  evidence shard B

Solver input:
  question + agent messages

scorer:
  exact match / normalized F1 / answer alias match
```

必须有的 baseline：

```text
single_q_only:
  capable single solver only sees question; should be near evidence-free floor.

single_full_info:
  capable single solver sees question + all evidence; must be above floor.

textmas_split:
  agents see split evidence and communicate; should recover part of full-info score.

ablation_no_message:
  solver sees question but no upstream message.

ablation_shuffled_message:
  solver receives message from another sample.

ablation_wrong_evidence:
  one agent receives irrelevant evidence.
```

准入：

```text
single_full_info above floor
textmas_split above no_message with paired CI lower bound > 0
shuffled/wrong evidence significantly below matched textmas
parser/scorer stable
```

### C-Family 2: scalable distributed-state synthesis

目标：

```text
构造非玩具、可扩展、可确定评分的 distributed state 任务。
每个 agent 拥有 state shard，最终答案依赖全局聚合或约束求解。
```

任务形态：

```text
set / table / graph / constraint aggregation
multi-shard counting / max-min / join / consistency checking
path or dependency synthesis with hidden shard edges
```

为什么不是小 toy：

```text
样本规模、shard 数、噪声项、distractor 数和 reasoning depth 必须可扩展。
每个 family 至少需要 calibration + held-out 的成规模生成，
不能用几十条 smoke 得出科学结论。
```

必须有的 baseline：

```text
oracle_program:
  deterministic scorer and answer generator.

single_full_state:
  solver sees all shards; verifies language model can solve when information complete.

textmas_split:
  agents communicate partial state.

ablation_no_message / wrong_shard / shuffled_message:
  validates communication necessity.
```

准入：

```text
full-state solver above floor
split TextMAS recovers nontrivial score
wrong/shuffled controls drop
error taxonomy shows integration vs extraction vs format failures
```

### C-Family 3: planner-coder-tester-reviewer code workflow

目标：

```text
把 multi-agent 分工放在真实工程工作流中，而不是普通 QA 的同质化重复。
```

候选数据源：

```text
HumanEval+ / MBPP+ / 自有可执行单元测试任务。
```

推荐角色：

```text
Planner:
  problem decomposition and edge cases.

Coder:
  implementation.

Tester:
  generates or runs tests, reports failing cases only.

Reviewer:
  patches code based on test signal.
```

准入：

```text
single capable solver pass rate above floor
text workflow does not collapse parser/execution
unit-test execution is the scorer
tester/reviewer ablations produce measurable drop
```

注意：

```text
该 family 成本和工程复杂度最高。它适合作为第二个 Phase C family，
不建议作为第一批唯一入口。
```

## 4. 推荐第一批 Phase C 准备对象

推荐只准备 manifest/scorer 草案，不执行：

```text
P2C-evidence-split-QA-v0:
  首选。最贴近 partial-information communication，且后续可转为 CoLA
  adaptation 的文本任务。

P2C-distributed-state-v0:
  并行准备。它提供可控、确定评分、可扩展的通信必要性测试。

P2C-code-workflow-v0:
  暂列第二批。等前两类确认 protocol 后再接入执行式 scorer。
```

数据源和 runner 细化方案：

```text
/data1/luyifei/drla/docs/current/
P2_Phase_C_Data_Source_and_Runner_Design_2026-06-01.md
```

不推荐第一批执行：

```text
CRAFT / CoSMAC 原环境直接接入：
  科学上相关，但环境/交互成本高，且离 CoLA text-latent substrate 转换更远。

继续 official8-compatible Branch B Family 2:
  只作为 diagnostic，不作为主线 claim。
```

## 5. Manifest Schema

草案 schema：

```text
/data1/luyifei/drla/configs/p2_phase_c_manifest_schema.json
```

schema example：

```text
/data1/luyifei/drla/configs/p2_phase_c_manifest_example.json
```

该 example 只用于 validator/schema 自测，不是实验数据，不得进入任何实验统计。

records example：

```text
/data1/luyifei/drla/configs/p2_phase_c_records_example.jsonl
```

该 records example 只用于 builder/validator 自测，不是实验数据，不得进入任何
实验统计。

manifest builder skeleton：

```text
/data1/luyifei/drla/drla/scripts/build_p2_phase_c_manifest.py
```

该 builder 只打包 normalized sample records，不下载数据、不生成 benchmark 样本、
不跑模型。

本地校验脚本：

```text
/data1/luyifei/drla/drla/scripts/validate_p2_phase_c_manifest.py
```

脚本边界：

```text
local-only
no model generation
no optimizer/backward
no SwanLab run
no held-out inspection beyond manifest-level split counts
```

核心字段：

```text
sample_id
family
task_name
split
source
question
public_context
agent_views[]
scoring
leakage_audit
baselines_required[]
```

其中 `agent_views[]` 必须记录每个 agent 的：

```text
agent_id
role
private_observation
allowed_output_contract
forbidden_fields
```

## 6. Evaluation Protocol

每个 family 至少跑以下条件：

```text
single_q_only
single_full_info
textmas_matched
textmas_no_message
textmas_shuffled_message
textmas_wrong_evidence_or_wrong_shard
textmas_compressed_state
```

每个条件必须写：

```text
generations.jsonl
metrics.jsonl
summary.json
task_summary.csv
failure_taxonomy.json
leakage_audit.json
```

纯 eval / generation：

```text
swanlab_mode=disabled
no optimizer
no backward
no SwanLab cloud run
```

任何训练或 adapter/fuser：

```text
必须等 Phase A 触发。
CUDA only
SwanLab cloud
metrics.jsonl
valid_interval <= 10 step
best_checkpoint.pt + last_checkpoint.pt
```

## 7. 防泄漏规则

在线 agent 输入禁止包含：

```text
gold answer
scorer output
selected_prediction
official correctness
full evidence union, unless condition is single_full_info
held-out prompt/parser/debug observations during calibration
```

scorer 只能看：

```text
final Solver output
task id / sample id for lookup
gold answer inside offline scorer only
```

## 8. Phase C 成功标准

任务 family 进入 Phase A 的最低标准：

```text
1. full-info single solver above floor。
2. matched TextMAS > no-message，paired CI lower bound > 0。
3. matched TextMAS > shuffled/wrong-message control。
4. scorer/parser stable，parseable_rate >= 0.95。
5. failure taxonomy 可解释，主要失败不是格式崩溃。
6. calibration 与 held-out split 锁定，held-out 只报告不调参。
```

如果没有 family 通过：

```text
不能转入 CoLA latent-vs-text 主表。
应重新审查 benchmark 是否真的适合 agent communication，
而不是训练 CoLA 去拟合一个不自然的 protocol。
```

## 9. 下一步执行边界

当前允许：

```text
补 manifest schema
补 manifest validator
补 manifest builder skeleton
补 scorer 设计文档
检查已有 runner 是否可复用
准备数据源候选清单
```

当前不允许，除非用户明确选择 Branch C：

```text
下载/构建大规模 Phase C 数据
跑 capable text agents
看 held-out
训练 adapter/fuser
写 paper 主表
```
