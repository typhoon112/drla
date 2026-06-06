# P2 Phase C 数据源与 Runner 设计

更新日期：2026-06-01

> 状态：Phase C 数据源/runner 安全准备文档。当前 locked scheme 已将 Phase C
> 设为下一条默认路线，但本文本身不下载大规模数据、不跑模型、不看 held-out、
> 不训练、不创建 SwanLab run。它只把 Phase C 应优先接入的数据源、runner、
> scorer 和 failure taxonomy 设计固化下来，避免后续临时凭直觉选 benchmark。

## 1. Phase C 的判定目标

Phase C 不是为了证明 CoLA latent communication，而是为了筛出真正适合
agent-to-agent communication 的 benchmark/protocol：

```text
1. 任务天然需要多 agent 通信或 distributed state integration。
2. capable text agents 的 Single / Role TextMAS baseline 可用。
3. no-message / shuffled-message / wrong-evidence control 会显著下降。
4. scorer/parser 稳定，且在线 prompt 没有 gold/scorer 泄漏。
5. 通过后才能进入 Phase A: CoLA substrate/interface adaptation。
```

2026-06-01 当前执行结论：

```text
benchmark baseline 要换：
  official8/GSM8K/ARC/GPQA/MedQA 等单问答集合不再作为 P2 主 MAS benchmark。
  它们只保留为 CoLA substrate capability diagnostic。

agent baseline 要换：
  先使用 capable text agents 跑 true MAS 协议与 controls。
  通过后再让 CoLA 在同一协议上做 Single / Role TextMAS gate。

CoLA 架构是否满足新 benchmark：
  架构层面满足 shared latent substrate 研究需求，因为 VAE/DiT 提供 text-latent
  双向接口和 block-wise latent state。
  权重层面不保证满足新 benchmark，因为 frozen official CoLA 未必具备
  instruction-following、role-message parsing、evidence integration 和
  final answer formatting 能力。
```

所以 Phase C 的 immediate deliverable 是：

```text
找出一个 capable TextMAS 显著依赖通信、controls 明显下降、scorer/parser
稳定的 locked benchmark/protocol。
```

而不是：

```text
让 frozen CoLA 直接在新 benchmark 上取得高分。
证明 latent 已优于 text。
训练 latent receiver/fuser。
```

2026-06-05 Phase A runner 更新：

```text
run_p2_phase_c_text_agents.py 现在支持 --provider cola_dlm。
该 provider 使用 official CoLA VAE/DiT、本地 tokenizer 和官方
generate_task_repaint_inference，在同一 Phase C online inputs / parser /
scorer / artifact schema 下生成。

支持的 CoLA prompt styles:
  chat_join: 直接把 chat messages 拼接成纯文本。
  plain_qa_v1: calibration-only 的短 QA/Useful facts 模板。
  squad_template_v1: final solver 走 CoLA 官方 SQuAD prompt-template 形态。

当前诊断结果:
  三种 prompt-only style 在 MuSiQue 14-row smoke 上均未产生 primary-score
  命中；raw outputs 显示 evidence copying、continuation drift 和答案格式漂移。
  同一 local CoLA 权重在 official SQuAD 20-sample sanity 上有 5/20 primary
  matches，因此失败不是模型加载/CUDA 链路故障，而是 MuSiQue role/interface
  distribution mismatch。

下一步:
  Phase A 应停止 prompt-only 修补，转为 calibration-only supervised
  task-format / role-interface adaptation；训练必须走 GPU + SwanLab cloud，
  写 metrics.jsonl，保存 best_checkpoint.pt / last_checkpoint.pt，并保持
  valid_interval <= 10 step。
```

## 2. 外部资料核查

已核查的候选来源：

```text
HotpotQA:
  https://hotpotqa.github.io/
  https://arxiv.org/abs/1809.09600

2WikiMultiHopQA:
  https://huggingface.co/datasets/xanhho/2WikiMultihopQA
  https://arxiv.org/abs/2011.01060

MuSiQue:
  https://arxiv.org/abs/2108.00573
  https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00475/110996/

Silo-Bench:
  https://arxiv.org/abs/2603.01045

CRAFT:
  https://arxiv.org/abs/2603.25268

CoSMAC:
  https://openreview.net/forum?id=yGzAhl1o4i
```

对本项目的直接含义：

```text
HotpotQA / 2Wiki / MuSiQue:
  适合构造 evidence-split multi-hop QA。它们有问题、答案、上下文/证据，
  可把 supporting evidence 分给不同 agent，测试下游 Solver 是否整合消息。

Silo-Bench:
  适合借鉴 distributed state integration 的评价精神，但不应直接照搬成
  小 toy；我们需要可扩展样本规模、可控 shard 数和 deterministic scorer。

CRAFT / CoSMAC:
  强调 strict partial information、communication/coordination 和 failure
  taxonomy。它们更像设计约束，不建议第一批直接接入原环境。
```

## 3. 第一批候选排序

推荐排序：

```text
1. P2C-evidence-split-QA-v0
   主候选。最贴近 natural language agent communication，也最容易后续转为
   CoLA text/latent substrate adaptation。

2. P2C-distributed-state-v0
   并行候选。确定评分、可扩展、可严格控制 communication necessity。

3. P2C-code-workflow-v0
   第二批。科学价值高，但 execution scorer、sandbox、安全和成本更复杂。
```

暂不作为第一批：

```text
CRAFT / CoSMAC 原环境:
  环境/交互成本高，且与 CoLA latent substrate 的后续衔接更远。

Branch B Family 2 frozen-CoLA diagnostic:
  只保留为诊断，不作为 true MAS 主线。
```

## 4. P2C-evidence-split-QA-v0

### 4.1 数据源选择

优先级：

```text
MuSiQue:
  优点：构造目标就是减少 shortcut、要求 connected multihop reasoning。
  风险：需要确认本地/远程字段能稳定抽取 supporting paragraphs 与答案别名。

2WikiMultiHopQA:
  优点：有 structured/unstructured evidence 与 reasoning explanations。
  风险：需要字段适配和 answer normalization。

HotpotQA distractor:
  优点：数据成熟、supporting facts 字段清楚、规模大。
  风险：部分问题可能被参数知识或单一段落 shortcut 解决，需要 ablation 筛选。
```

第一批建议：

```text
先准备 MuSiQue + HotpotQA 两套 manifest 草案。
若 MuSiQue 字段抽取顺利，以 MuSiQue 为主。
若 MuSiQue 数据源/字段不稳定，用 HotpotQA 做 runner 验证。
```

2026-06-01 tiny first-rows dry inspection:

```text
HotpotQA distractor/train:
  possible_evidence_split.
  Fields include id, question, answer, context, supporting_facts,
  context.title, context.sentences, supporting_facts.title.

MuSiQue answerable/train mirror:
  possible_evidence_split.
  Fields include id, question, answer, answer_aliases, paragraphs,
  paragraphs.is_supporting, question_decomposition.paragraph_support_idx.
  Mirror/license metadata still needs confirmation against official release.

2WikiMultiHopQA mirror:
  possible_evidence_split.
  Fields include id, question, answer, context, supporting_facts, evidences,
  context.title, context.sentences, supporting_facts.title.
  Mirror/license metadata still needs confirmation against official release.
```

Implication:

```text
The first real builder can target MuSiQue + HotpotQA without inventing a new
schema. 2Wiki remains a strong second/backup family. This field audit does not
admit any benchmark and does not replace manifest/scorer/leakage gates.
```

2026-06-01 calibration-only manifest drafts:

```text
HotpotQA source rows:
  outputs/p2_phase_c_data_source_audits/
  hf_rows_hotpotqa_distractor_train_300_seed20260601
  rows: 300, seeded multi-block train sampling

HotpotQA records:
  outputs/p2_phase_c_records/
  hotpotqa_calibration_records_200_seed20260601
  records: 200, skipped: 0

HotpotQA manifest:
  outputs/p2_phase_c_manifests/
  hotpotqa_calibration_manifest_200_seed20260601
  validator: pass, errors: 0, warnings: 271

MuSiQue source rows:
  outputs/p2_phase_c_data_source_audits/
  hf_rows_musique_answerable_train_300_seed20260601
  rows: 300, seeded multi-block train sampling

MuSiQue records:
  outputs/p2_phase_c_records/
  musique_calibration_records_200_seed20260601
  records: 200, skipped: 0

MuSiQue manifest:
  outputs/p2_phase_c_manifests/
  musique_calibration_manifest_200_seed20260601
  validator: pass, errors: 0, warnings: 207
```

Warning interpretation:

```text
Most validator warnings are "gold answer string appears in private observation".
For evidence QA this often means the answer naturally appears inside support
evidence, not that the gold label field was inserted into the online prompt.
However, it is a real shortcut risk. These drafts cannot be admitted until
single_q_only, textmas_no_message, shuffled_message and wrong_evidence controls
show that matched communication is actually needed.
```

2026-06-01 control input packages:

```text
HotpotQA control inputs:
  outputs/p2_phase_c_control_inputs/
  hotpotqa_calibration_controls_200_seed20260601
  rows: 1400 = 200 samples * 7 conditions
  leakage audit: pass, errors 0, warnings 785

MuSiQue control inputs:
  outputs/p2_phase_c_control_inputs/
  musique_calibration_controls_200_seed20260601
  rows: 1400 = 200 samples * 7 conditions
  leakage audit: pass, errors 0, warnings 724
```

Conditions materialized:

```text
single_q_only
single_full_info
textmas_matched
textmas_no_message
textmas_shuffled_message
textmas_wrong_evidence_or_wrong_shard
textmas_compressed_state
```

Boundary:

```text
These are online-input packages and prompt contracts only. They do not contain
model outputs and do not admit the benchmark. The high warning counts are
expected for evidence QA where answer strings naturally appear in support
evidence; controls are mandatory before any communication claim.
```

2026-06-01 strict wrong-evidence v1 control packages:

```text
Reason:
  v0 wrong_evidence replaced only one evidence-agent shard. In 2-hop samples,
  the remaining correct shard was often sufficient, so wrong_evidence was too
  weak and could not cleanly test communication necessity.

Change:
  textmas_wrong_evidence_or_wrong_shard now gives all evidence agents private
  shards from a non-self control sample while preserving the target question.

Prompt contract:
  p2_phase_c_evidence_split_v1_strict_wrong_evidence

MuSiQue v1 control inputs:
  outputs/p2_phase_c_control_inputs/
  musique_calibration_controls_200_seed20260601_v1_strict_wrong
  rows: 1400 = 200 samples * 7 conditions
  build status: pass

HotpotQA v1 control inputs:
  outputs/p2_phase_c_control_inputs/
  hotpotqa_calibration_controls_200_seed20260601_v1_strict_wrong
  rows: 1400 = 200 samples * 7 conditions
  build status: pass
```

Boundary:

```text
Old v0 pilots remain useful diagnostics showing that the previous wrong-control
was too weak. They must not be used for benchmark admission.
```

2026-06-01 text-agent runner:

```text
script:
  /data1/luyifei/drla/drla/scripts/run_p2_phase_c_text_agents.py

preflight script:
  /data1/luyifei/drla/drla/scripts/preflight_p2_phase_c_text_agent_run.py

supported provider:
  openai_compatible chat/completions endpoint
  local_transformers HuggingFace causal LM provider for local engineering runs

required env for real model run:
  OPENAI_API_KEY
  OPENAI_MODEL
  optional OPENAI_BASE_URL

selfcheck artifact:
  outputs/p2_phase_c_text_agent_runs/selfcheck_20260601
  status: pass
  rows: 8 toy rows
  meaning: routing/scoring/cache logic only, not an experiment

current credential state:
  OPENAI_API_KEY unset
  OPENAI_MODEL unset
```

Local model provider, 2026-06-01:

```text
Implemented provider:
  run_p2_phase_c_text_agents.py --provider local_transformers

Local 4B model path:
  /data1/luyifei/drla/models/Qwen3-4B-Instruct-2507-git

Local 8B-FP8 model path:
  /data1/luyifei/drla/models/Qwen3-8B-FP8
  symlink target: /tmp/drla_models/Qwen3-8B-FP8

Status:
  both local models load on CUDA_VISIBLE_DEVICES=0 and pass minimal generation
  smoke.

Boundary:
  This enables real local model engineering checks when no API endpoint is
  configured. It is not automatically the final "capable text-agent" baseline
  for paper claims; full Phase C admission still requires calibration-scale
  controls and gate evidence.
```

Source note:

```text
Qwen/Qwen3-8B-FP8 official Hugging Face page lists Transformers support,
Apache-2.0 license, 8.2B parameters, native 32,768-token context, and the
enable_thinking switch. The runner defaults local thinking mode off for
short-answer eval protocol.
```

Preflight artifacts:

```text
HotpotQA:
  outputs/p2_phase_c_text_agent_preflights/
  hotpotqa_calibration_preflight_200_seed20260601
  ready_to_run_model: false
  reason: OPENAI_API_KEY unset, OPENAI_MODEL unset
  estimated chat calls: 2600
    solver calls: 1400
    agent calls: 1200
    unique agent-cache keys: 600

MuSiQue:
  outputs/p2_phase_c_text_agent_preflights/
  musique_calibration_preflight_200_seed20260601
  ready_to_run_model: false
  reason: OPENAI_API_KEY unset, OPENAI_MODEL unset
  estimated chat calls: 2600
    solver calls: 1400
    agent calls: 1200
    unique agent-cache keys: 600

HotpotQA local Qwen3-4B preflight:
  outputs/p2_phase_c_text_agent_preflights/
  hotpotqa_local_qwen3_4b_preflight_200_seed20260601
  ready_to_run_model: true
  provider: local_transformers
  estimated chat calls: 2600

HotpotQA local Qwen3-8B-FP8 preflight:
  outputs/p2_phase_c_text_agent_preflights/
  hotpotqa_local_qwen3_8b_fp8_preflight_200_seed20260601
  ready_to_run_model: true
  provider: local_transformers
  estimated chat calls: 2600
```

Boundary:

```text
OpenAI-compatible endpoint credentials are still unset, but local_transformers
now provides real local model generation for engineering and calibration runs.
Mock/selfcheck outputs must never be cited as benchmark evidence. Local model
smokes remain wiring checks unless they cover the locked calibration/held-out
conditions and pass the admission gate.
```

Runner resume rule:

```text
run_p2_phase_c_text_agents.py supports --resume.
It restores existing generations.jsonl and agent-message cache before
continuing, so long model runs can be resumed without duplicate row ids.
Partial/resume artifacts are execution hygiene only; they do not change the
scientific gate.
```

2026-06-01 local Qwen3-4B smoke:

```text
Run artifact:
  outputs/p2_phase_c_text_agent_runs/
  hotpotqa_local_qwen3_4b_smoke14_20260601

Aggregate artifact:
  outputs/p2_phase_c_text_agent_aggregates/
  hotpotqa_local_qwen3_4b_smoke14_20260601

Scope:
  HotpotQA calibration, first 2 samples, 7 conditions, 14 rows.

Result:
  single_full_info: 2/2
  single_q_only: 0/2
  textmas_matched: 1/2
  textmas_no_message: 0/2
  textmas_shuffled_message: 0/2
  textmas_wrong_evidence_or_wrong_shard: 0/2
  textmas_compressed_state: 2/2

Gate:
  admitted=false because paired CI lower bounds are 0 with only 2 pairs.

Leakage audit:
  outputs/p2_phase_c_leakage_audits/
  hotpotqa_local_qwen3_4b_smoke14_20260601
  status: pass
  errors: 0
  warnings: 7, answer string appears naturally in online evidence text.

Interpretation:
  This is a real-model engineering smoke for provider/routing/scoring/control
  wiring only. It is not benchmark admission evidence and must not be used as
  a scientific conclusion.
```

Implementation note:

```text
The runner now writes control_source_sample_id into generations.jsonl for
shuffled_message and wrong_evidence controls, so run-level leakage audit can
verify non-self control sources from completed artifacts.
```

2026-06-01 local Qwen3-8B-FP8 pilot:

```text
Run artifact:
  outputs/p2_phase_c_text_agent_runs/
  hotpotqa_local_qwen3_8b_fp8_pilot70_20260601

Aggregate artifact:
  outputs/p2_phase_c_text_agent_aggregates/
  hotpotqa_local_qwen3_8b_fp8_pilot70_20260601

Leakage audit:
  outputs/p2_phase_c_leakage_audits/
  hotpotqa_local_qwen3_8b_fp8_pilot70_20260601
  status: pass, errors: 0, warnings: 32

Scope:
  HotpotQA calibration, first 10 samples, 7 conditions, 70 rows.

Result:
  single_full_info: 8/10
  single_q_only: 1/10
  textmas_matched: 6/10
  textmas_no_message: 1/10
  textmas_shuffled_message: 0/10
  textmas_wrong_evidence_or_wrong_shard: 3/10
  textmas_compressed_state: 5/10

Paired deltas:
  full_info - question_only: mean +0.7, CI lower +0.4
  matched - no_message: mean +0.5, CI lower 0.0
  matched - shuffled: mean +0.6, CI lower +0.3
  matched - wrong_evidence: mean +0.3, CI lower 0.0

Gate:
  admitted=false. The pilot is too small for no_message/wrong_evidence CI and
  wrong_evidence remains a material shortcut/control-risk signal.

Interpretation:
  The evidence-split QA protocol is executable with real local 8B-FP8 outputs
  and has a meaningful matched-vs-shuffled signal, but HotpotQA cannot be
  admitted from this pilot. Next decision should prefer full calibration or a
  stricter MuSiQue/distributed-state family rather than relaxing controls.
```

2026-06-01 format-fix + strict-control pilots:

```text
Protocol repair:
  Solver prompt now requires exactly:
    Final answer: <short answer>
  The parser also accepts "final answer is ..." / "answer is ...".
  This is calibration-only format repair, not a scoring relaxation.

MuSiQue strict-control pilot:
  run:
    outputs/p2_phase_c_text_agent_runs/
    musique_local_qwen3_8b_fp8_v1_strict_wrong_pilot70_20260601
  aggregate:
    outputs/p2_phase_c_text_agent_aggregates/
    musique_local_qwen3_8b_fp8_v1_strict_wrong_pilot70_20260601
  leakage:
    outputs/p2_phase_c_leakage_audits/
    musique_local_qwen3_8b_fp8_v1_strict_wrong_pilot70_20260601
  scope: first 10 calibration samples, 7 conditions, 70 rows
  leakage: pass, errors 0, warnings 31

MuSiQue strict-control result:
  single_full_info: 0.6
  single_q_only: 0.1
  textmas_matched: 0.7
  textmas_no_message: 0.1
  textmas_shuffled_message: 0.1
  textmas_wrong_evidence_or_wrong_shard: 0.1
  textmas_compressed_state: 0.6
  paired CI lower:
    full_info - question_only: +0.2
    matched - no_message: +0.3
    matched - shuffled: +0.3
    matched - wrong_evidence: +0.3
  pilot gate: admitted=true

HotpotQA strict-control pilot:
  run:
    outputs/p2_phase_c_text_agent_runs/
    hotpotqa_local_qwen3_8b_fp8_v1_strict_wrong_pilot70_20260601
  aggregate:
    outputs/p2_phase_c_text_agent_aggregates/
    hotpotqa_local_qwen3_8b_fp8_v1_strict_wrong_pilot70_20260601
  leakage:
    outputs/p2_phase_c_leakage_audits/
    hotpotqa_local_qwen3_8b_fp8_v1_strict_wrong_pilot70_20260601
  scope: first 10 calibration samples, 7 conditions, 70 rows
  leakage: pass, errors 0, warnings 28

HotpotQA strict-control result:
  single_full_info: 0.8
  single_q_only: 0.4
  textmas_matched: 0.8
  textmas_no_message: 0.4
  textmas_shuffled_message: 0.1
  textmas_wrong_evidence_or_wrong_shard: 0.2
  textmas_compressed_state: 0.8
  paired CI lower:
    full_info - question_only: 0.0
    matched - no_message: 0.0
    matched - shuffled: +0.4
    matched - wrong_evidence: +0.2
  pilot gate: admitted=false
```

Interpretation:

```text
MuSiQue became the preferred evidence-split QA family for full calibration
because it passed the strict 10-sample pilot across all controls.
HotpotQA is stronger as a solver/evidence diagnostic, but question-only and
no-message shortcuts are too high in this pilot, so it must not be the next
admission candidate unless a larger calibration run contradicts this pattern.
```

2026-06-01 MuSiQue strict-control full calibration:

```text
Run shards:
  outputs/p2_phase_c_text_agent_runs/
  musique_local_qwen3_8b_fp8_v1_strict_wrong_full200_shard00_20260601
  musique_local_qwen3_8b_fp8_v1_strict_wrong_full200_shard01_20260601
  musique_local_qwen3_8b_fp8_v1_strict_wrong_full200_shard02_20260601
  musique_local_qwen3_8b_fp8_v1_strict_wrong_full200_shard03_20260601

Merged run:
  outputs/p2_phase_c_text_agent_runs/
  musique_local_qwen3_8b_fp8_v1_strict_wrong_full200_merged_20260601

Aggregate:
  outputs/p2_phase_c_text_agent_aggregates/
  musique_local_qwen3_8b_fp8_v1_strict_wrong_full200_20260601

Leakage audit:
  outputs/p2_phase_c_leakage_audits/
  musique_local_qwen3_8b_fp8_v1_strict_wrong_full200_20260601

Scope:
  MuSiQue calibration split, 200 samples, 7 conditions, 1400 generated rows.
  Provider: local_transformers.
  Model: /data1/luyifei/drla/models/Qwen3-8B-FP8.
```

Full calibration result:

```text
single_full_info:
  primary_score_mean: 0.425
  exact_match_mean: 0.390
  parseable_rate: 1.000

single_q_only:
  primary_score_mean: 0.080
  exact_match_mean: 0.065
  parseable_rate: 1.000

textmas_matched:
  primary_score_mean: 0.450
  exact_match_mean: 0.395
  parseable_rate: 1.000

textmas_no_message:
  primary_score_mean: 0.080
  exact_match_mean: 0.065
  parseable_rate: 1.000

textmas_shuffled_message:
  primary_score_mean: 0.060
  exact_match_mean: 0.045
  parseable_rate: 1.000

textmas_wrong_evidence_or_wrong_shard:
  primary_score_mean: 0.070
  exact_match_mean: 0.060
  parseable_rate: 1.000

textmas_compressed_state:
  primary_score_mean: 0.420
  exact_match_mean: 0.355
  parseable_rate: 1.000
```

Paired admission deltas:

```text
single_full_info - single_q_only:
  mean: +0.345
  bootstrap CI lower: +0.265
  bootstrap CI upper: +0.420

textmas_matched - textmas_no_message:
  mean: +0.370
  bootstrap CI lower: +0.295
  bootstrap CI upper: +0.440

textmas_matched - textmas_shuffled_message:
  mean: +0.390
  bootstrap CI lower: +0.320
  bootstrap CI upper: +0.460

textmas_matched - textmas_wrong_evidence_or_wrong_shard:
  mean: +0.380
  bootstrap CI lower: +0.315
  bootstrap CI upper: +0.450
```

Gate and leakage:

```text
aggregate status: pass
admitted: true on calibration
failed_gates: []
leakage audit status: pass
leakage errors: 0
leakage warnings: 633
```

Warning interpretation:

```text
The leakage warnings are "gold/alias string appears in online input text".
For evidence-split QA, this often means the answer naturally appears inside
support evidence, not that a hidden gold label field was inserted into online
prompts. This is why the no-message, shuffled-message and wrong-evidence
controls are mandatory. In this full MuSiQue calibration, all such controls
remain far below textmas_matched.
```

Decision:

```text
MuSiQue evidence-split QA v1 strict is admitted at calibration scale.
It is not yet a final paper benchmark until the same protocol passes locked
held-out evaluation without held-out prompt repair.

Next:
  build locked MuSiQue held-out split with identical schema/conditions/scorer;
  run one capable TextMAS held-out evaluation;
  if it passes, enter CoLA substrate/interface adaptation on this protocol.
```

2026-06-05 MuSiQue strict-control locked held-out:

```text
Source rows:
  outputs/p2_phase_c_data_source_audits/
  hf_rows_musique_answerable_validation_1000_seed20260605
  dataset: bdsaglam/musique
  config: answerable
  split: validation
  rows fetched: 1000

Records:
  outputs/p2_phase_c_records/
  musique_heldout_records_800_seed20260605
  records: 800
  split: heldout

Manifest:
  outputs/p2_phase_c_manifests/
  musique_heldout_manifest_800_seed20260605
  status: pass

Control inputs:
  outputs/p2_phase_c_control_inputs/
  musique_heldout_controls_800_seed20260605_v1_strict_wrong
  rows: 5600 = 800 samples * 7 conditions
  prompt contract: p2_phase_c_evidence_split_v1_strict_wrong_evidence
```

Execution artifacts:

```text
Sharded runs:
  outputs/p2_phase_c_text_agent_runs/
  musique_local_qwen3_8b_fp8_v1_strict_wrong_heldout800_shard00_20260605
  ...
  musique_local_qwen3_8b_fp8_v1_strict_wrong_heldout800_shard07_20260605

Merged run:
  outputs/p2_phase_c_text_agent_runs/
  musique_local_qwen3_8b_fp8_v1_strict_wrong_heldout800_merged_20260605
  rows: 5600
  unique row ids: 5600

Aggregate:
  outputs/p2_phase_c_text_agent_aggregates/
  musique_local_qwen3_8b_fp8_v1_strict_wrong_heldout800_20260605

Leakage audit:
  outputs/p2_phase_c_leakage_audits/
  musique_local_qwen3_8b_fp8_v1_strict_wrong_heldout800_20260605
```

Held-out result:

```text
single_full_info:
  primary_score_mean: 0.48625
  exact_match_mean: 0.45000
  parseable_rate: 1.000

single_q_only:
  primary_score_mean: 0.03875
  exact_match_mean: 0.03250
  parseable_rate: 1.000

textmas_matched:
  primary_score_mean: 0.37375
  exact_match_mean: 0.34000
  parseable_rate: 1.000

textmas_no_message:
  primary_score_mean: 0.03875
  exact_match_mean: 0.03250
  parseable_rate: 1.000

textmas_shuffled_message:
  primary_score_mean: 0.04750
  exact_match_mean: 0.04000
  parseable_rate: 1.000

textmas_wrong_evidence_or_wrong_shard:
  primary_score_mean: 0.04875
  exact_match_mean: 0.04125
  parseable_rate: 1.000

textmas_compressed_state:
  primary_score_mean: 0.38250
  exact_match_mean: 0.35375
  parseable_rate: 1.000
```

Held-out paired admission deltas:

```text
single_full_info - single_q_only:
  mean: +0.44750
  bootstrap CI lower: +0.41250
  bootstrap CI upper: +0.48250

textmas_matched - textmas_no_message:
  mean: +0.33500
  bootstrap CI lower: +0.30125
  bootstrap CI upper: +0.36875

textmas_matched - textmas_shuffled_message:
  mean: +0.32625
  bootstrap CI lower: +0.29375
  bootstrap CI upper: +0.36000

textmas_matched - textmas_wrong_evidence_or_wrong_shard:
  mean: +0.32500
  bootstrap CI lower: +0.29250
  bootstrap CI upper: +0.35875
```

Gate and leakage:

```text
aggregate status: pass
admitted: true on locked held-out
failed_gates: []
leakage audit status: pass
leakage errors: 0
leakage warnings: 2640
```

Decision:

```text
MuSiQue evidence-split QA v1 strict is now admitted at locked held-out scale.
Phase C benchmark/protocol validation is complete for this protocol.

Next:
  freeze MuSiQue v1 strict protocol and artifacts;
  enter Phase A: CoLA substrate/interface adaptation;
  require CoLA Single Solver and CoLA Role TextMAS gates before any
  CoLA TextMAS vs LatentMAS main comparison.
```

2026-06-01 text-agent aggregation / admission gate:

```text
script:
  /data1/luyifei/drla/drla/scripts/
  aggregate_p2_phase_c_text_agent_results.py

gate:
  required conditions present
  single_full_info primary mean >= 0.2
  single_full_info and textmas_matched parseable_rate >= 0.95
  paired bootstrap CI lower bound > 0 for:
    single_full_info - single_q_only
    textmas_matched - textmas_no_message
    textmas_matched - textmas_shuffled_message
    textmas_matched - textmas_wrong_evidence_or_wrong_shard

selfcheck pass artifact:
  outputs/p2_phase_c_text_agent_aggregates/selfcheck_pass_20260601
  admitted: true on toy complete controls

negative gate artifact:
  outputs/p2_phase_c_text_agent_aggregates/
  runner_selfcheck_negative_gate_20260601
  admitted: false because shuffled/wrong controls are absent
```

Boundary:

```text
The gate is locked before real capable-agent results are produced. This avoids
post-hoc threshold selection after seeing model outputs.
```

字段/license 审计：

```text
/data1/luyifei/drla/docs/current/
P2_Phase_C_Data_Source_Field_License_Audit_2026-06-01.md
```

### 4.2 样本构造

每条样本：

```text
public_context:
  empty or minimal instruction; 不包含完整证据。

question:
  原始问题。

agent_views:
  Agent A: supporting evidence shard A + limited distractors
  Agent B: supporting evidence shard B + limited distractors
  optional Agent C: third hop / disambiguation shard

scoring:
  normalized exact match + alias match
  optional token F1 for QA-style answers
```

关键要求：

```text
每个 split agent 不能看到完整证据 union。
single_full_info condition 可以看到 full evidence union，用于验证 task solvability。
textmas_matched condition 才把 agent messages 交给 Solver。
```

### 4.3 Baseline 条件

必须跑：

```text
single_q_only:
  question only.

single_full_info:
  question + all evidence shards.

textmas_matched:
  shard agents -> Solver.

textmas_no_message:
  Solver sees question only under same final prompt.

textmas_shuffled_message:
  Solver receives messages from another sample in same task/split.

textmas_wrong_evidence_or_wrong_shard:
  one upstream agent receives irrelevant evidence.

textmas_compressed_state:
  upstream messages constrained to compact typed state.
```

通过条件：

```text
single_full_info > single_q_only
textmas_matched > textmas_no_message
textmas_matched > shuffled/wrong-message controls
parseable_rate >= 0.95
failure taxonomy 不是主要由 parser/format failure 主导
```

## 5. P2C-distributed-state-v0

### 5.1 任务族

候选任务：

```text
set_join:
  Agent A/B/C 各持有集合片段，答案依赖交/并/差或条件过滤。

table_join:
  Agent A/B 各持有表的一部分，答案需要 join key 与聚合。

graph_path:
  Agent A/B 持有不同边集，答案依赖跨 shard 路径或连通性。

constraint_merge:
  每个 agent 持有局部约束，答案需要全局一致性检查。
```

### 5.2 非 toy 要求

不能用几十条手写 smoke 得出结论。每个任务族至少要求：

```text
calibration >= 200
heldout >= 800
至少 3 个 difficulty levels
至少 2 个 shard counts
包含 distractors / irrelevant rows
oracle_program 生成答案和 scorer
```

### 5.3 Runner 价值

该 family 的主要价值：

```text
可严格知道 single_q_only floor。
可控制 agent 之间信息互补性。
可定量调节 communication complexity。
可用 deterministic scorer 做低噪声判定。
```

风险：

```text
合成任务如果语义太浅，会变成工程诊断而不是 paper 主 claim。
必须用 difficulty / distractor / scale 保证不是 trivial toy。
```

## 6. P2C-code-workflow-v0

第二批候选。只有当前两类任务通过 Phase C 或被证明不合适后再启动。

推荐条件：

```text
Planner -> Coder -> Tester -> Reviewer
unit tests as scorer
no gold solution in online prompt
Tester 只能报告 failing case / observed output，不给 hidden solution
Reviewer 只能基于 test signal patch
```

通过条件：

```text
single capable solver above floor
workflow matched > no-tester / shuffled-test-report
unit-test execution stable
cost 可接受
```

## 7. Runner 设计

Phase C runner 应分成三层：

```text
manifest builder:
  只构造 json manifest，不跑模型。
  当前 skeleton:
    /data1/luyifei/drla/drla/scripts/build_p2_phase_c_manifest.py

protocol runner:
  读取 manifest，跑 selected condition 的 capable text agents。
  pure eval/generation，swanlab_mode=disabled。

preflight:
  在任何 model call 前检查 manifest/control consistency、env readiness 和
  call budget。preflight 不调用模型，不创建 SwanLab run。

aggregator:
  汇总 score、paired delta、CI、parseable、failure taxonomy、leakage audit。
```

推荐输出目录结构：

```text
outputs/p2_phase_c/<run_name>/
  config.json
  manifest.json
  generations.jsonl
  metrics.jsonl
  summary.json
  task_summary.csv
  condition_summary.csv
  paired_deltas.csv
  failure_taxonomy.json
  leakage_audit.json
```

每条 generation row 至少包含：

```text
sample_id
family
task_name
split
condition
agent_id / role
prompt_hash
input_contract
output_text
parsed_answer
score
parseable
failure_tags[]
online_input_fields
online_input_field_hashes
```

## 8. Scorer 设计

QA scorer：

```text
normalize:
  lower
  strip punctuation/articles/extra whitespace

metrics:
  exact_match
  alias_match
  token_f1

report:
  primary = exact_or_alias
  secondary = token_f1
```

当前本地 scorer helper：

```text
/data1/luyifei/drla/drla/evaluation/p2_phase_c_scorers.py
```

Distributed-state scorer：

```text
oracle answer:
  generated by deterministic program.

metrics:
  exact structured match
  optional partial credit for set/table answers

report:
  primary = exact structured match
```

当前自检脚本：

```text
/data1/luyifei/drla/drla/scripts/selfcheck_p2_phase_c_scorers.py
```

该自检只使用 hardcoded toy assertions，不能作为实验结果引用。

Code scorer：

```text
primary:
  unit test pass rate

secondary:
  syntax parse
  runtime error category
```

## 9. Failure Taxonomy

每个 final Solver output 至少打这些标签：

```text
format_failure:
  无法解析 final answer。

missing_private_info:
  需要的信息没有从上游 agent 传到 Solver。

wrong_integration:
  信息已出现，但 Solver 合并/推理错误。

wrong_message:
  上游 agent 摘要错误。

shortcut_guess:
  question-only 也答对，疑似不需要通信。

overreliance_on_prior:
  忽略 evidence/message，输出常识或参数记忆答案。

scorer_ambiguity:
  答案等价但 scorer 未覆盖，需 alias/scorer 修订；只能在 calibration 上修。
```

进入 Phase A 的 family，`wrong_integration` / `missing_private_info` 应该是有意义
比例，不能主要是 `format_failure` 或 `scorer_ambiguity`。

## 10. 泄漏检查

manifest-level：

```text
validate_p2_phase_c_manifest.py 必须 pass。
leakage_audit 四个核心布尔必须为 false。
```

run-level：

```text
online_input_field_hashes 记录每个 prompt 的输入字段 hash。
online_input_fields 记录可审计字段；若隐私/成本原因后续要删字段，必须先生成
leakage_audit.json 后再清理。
leakage_audit.json 检查 gold/scorer/full evidence 是否进入不该进入的 condition。
heldout split 不允许被用于 prompt/scorer 修订。
```

当前 run-level audit 脚本：

```text
/data1/luyifei/drla/drla/scripts/audit_p2_phase_c_run_leakage.py
```

设计边界：

```text
显式字段名 gold_answer / scorer_output / selected_prediction / full_evidence
进入在线输入时直接 fail。

证据文本中自然出现答案字符串只给 warning，因为 evidence-split QA 的 supporting
passage 可能本来包含答案；这类 warning 需要人工在 calibration 上确认。
```

必须阻断：

```text
gold answer in online prompt
scorer output in online prompt
full evidence in split-agent condition
shuffled/wrong controls 使用同 sample answer
heldout sample-level debug 用于改 prompt/parser
```

## 11. Agent A -> B 数据流锁定

Phase C / Phase E 的核心数据流必须满足：

```text
Agent A:
  sees only assigned private observation/shard.
  outputs text message in TextMAS, or latent packet/state in LatentMAS.

Agent B / final Solver:
  receives only allowed public context plus A's message/packet.
  must generate a new final answer after receiving the message/packet.

Scorer:
  sees only Agent B / final Solver final answer.
  never sees A decoded replay tokens, selected_prediction, gold/scorer fields,
  or oracle block labels.
```

`single_full_info` 是 solvability control，不是 MAS 协议：

```text
It checks whether a capable solver can answer when all evidence is visible.
It must not be confused with Agent B's online input in TextMAS/LatentMAS.
```

`single_q_only` / `no_message` / `shuffled_message` / `wrong_evidence` 是通信
必要性 controls：

```text
If these controls perform close to matched TextMAS, the benchmark is not
admitted even if matched TextMAS accuracy is high.
```

## 12. 如果 evidence-split QA 不过 gate

不要降低标准保留数据集。执行顺序改为：

```text
1. 写 failure report，记录是 shortcut、parser、prior knowledge 还是 scorer
   ambiguity 主导。
2. 保留 HotpotQA/MuSiQue artifacts 为 diagnostic。
3. 启动 P2C-distributed-state-v0 或 P2C-code-workflow-v0。
4. 每个新 family 仍需 calibration >= 200、heldout >= 800、controls、CI、
   leakage audit 和 failure taxonomy。
```

进入 distributed-state/code-workflow 不是妥协，而是为了确保 benchmark 真正
测 communication，而不是测单个 solver 的参数知识或答案字符串检索。

## 13. 当前允许的下一步

已经完成的安全准备：

```text
manifest schema / validator
dataset field inspection
seeded HotpotQA/MuSiQue calibration records
7-condition control input packages
runner selfcheck
aggregator gate selfchecks
preflight artifacts
scorer selfcheck
run-level leakage auditor selfcheck
```

下一步只允许以下两类：

```text
1. 设置真实 capable model endpoint/key 后，在 calibration split 上运行
   HotpotQA/MuSiQue capable text-agent controls。
2. 若 controls 不通过，写 failure report 并切换到 distributed-state/code
   workflow，而不是降低 gate。
```
