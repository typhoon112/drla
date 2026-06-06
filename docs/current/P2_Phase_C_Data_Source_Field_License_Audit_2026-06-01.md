# P2 Phase C 数据源字段与 License 审计

更新日期：2026-06-01

> 状态：安全准备文档。本文只记录公开资料核查，不下载数据、不构建真实
> manifest、不跑模型、不看 held-out、不训练、不创建 SwanLab run。真正接入任一
> 数据源前，还必须对本地下载后的首批 calibration records 做字段 dry inspection。

## 1. 审计结论

第一批 Phase C 推荐顺序保持不变：

```text
1. MuSiQue / HotpotQA evidence-split QA
2. distributed-state synthetic-but-scaled tasks
3. 2WikiMultiHopQA as backup/second evidence-split QA
4. code workflow second batch
```

原因：

```text
MuSiQue:
  最贴近 connected multihop reasoning 目标，但官方/原始格式存在字段名差异，
  接入前必须 dry inspect。

HotpotQA:
  字段最清楚，适合先验证 evidence-split QA builder/runner。
  但存在 shortcut/parametric knowledge 风险，需要 question-only 和 shuffled
  controls 过滤。

2WikiMultiHopQA:
  reasoning/evidence 解释更强，license 友好；适合作为第二个 QA family 或
  MuSiQue 字段不稳定时的替代。
```

## 2. Source Audit Table

| Source | License / terms | Public fields observed | Phase C value | Main risks | Current action |
|---|---|---|---|---|---|
| MuSiQue | CC BY 4.0 | 官方仓库说明包含 MuSiQue-Ans / MuSiQue-Full train/dev/test、single-hop IDs、official evaluator；字段名在 official/raw format 间会不同 | 构造目标就是 connected 2-4 hop QA，最适合 communication necessity | 字段名需 dry inspect；seed single-hop leakage caution；Google Drive download path | 第一优先，但先做 field dry inspection，不直接写真实 builder |
| HotpotQA | CC BY-SA 4.0 | HF card shows `id`, `question`, `answer`, `type`, `level`, `supporting_facts`, `context` | 字段清晰，supporting facts 可分 shard，适合验证 builder/runner | CC BY-SA 传播要求；部分问题可能 question-only/shortcut 可答 | 第一批 runner 验证候选 |
| 2WikiMultiHopQA | Apache-2.0 | GitHub notes paragraph info: `id`, `title`, `sentences`, `mentions`; eval with `evidences_id` / `answer_id`; paper emphasizes reasoning path evidence | evidence/reasoning-path 适合 failure taxonomy 和 split evidence | HF mirror字段需 dry inspect；可能需要 alias/id scorer | 第二候选或 MuSiQue backup |

## 3. MuSiQue Notes

公开依据：

```text
Repository:
  https://github.com/StonyBrookNLP/musique

Paper:
  https://arxiv.org/abs/2108.00573
```

关键信息：

```text
License:
  CC BY 4.0.

Dataset:
  MuSiQue-Ans and MuSiQue-Full train/dev/test.

Evaluation:
  official evaluate_v1.0.py reports answer_f1 and support_f1;
  MuSiQue-Full also reports group answer/support sufficiency metrics.

Leakage caution:
  dev/test single-hop question IDs are released because MuSiQue composes
  questions from seed single-hop datasets. If a model uses seed datasets,
  those IDs must be excluded.

Format caution:
  official released data and raw data used by code differ in field names.
```

Phase C implications:

```text
1. Do not assume field names before dry inspection.
2. Use MuSiQue-Ans first; MuSiQue-Full unanswerable contrast can be a later
   robustness extension.
3. Preserve support evidence IDs/paragraphs for agent shard assignment.
4. Record seed-leakage caution in manifest metadata.
```

Required dry inspection before real builder:

```text
sample keys
question field
answer / aliases field
paragraph/evidence field
support/evidence IDs
hop count / decomposition fields if available
split names
license metadata
```

## 4. HotpotQA Notes

公开依据：

```text
Dataset:
  https://huggingface.co/datasets/hotpotqa/hotpot_qa

Homepage:
  https://hotpotqa.github.io/

Paper:
  https://arxiv.org/abs/1809.09600
```

关键信息：

```text
License:
  CC BY-SA 4.0.

HF fields:
  id
  question
  answer
  type
  level
  supporting_facts
  context

Subsets:
  distractor
  fullwiki
```

Phase C implications:

```text
1. Use distractor first because context already contains support + distractors.
2. Build Agent A/B shards from supporting_facts titles/sentences plus distractors.
3. Keep single_q_only and shuffled-message controls because HotpotQA can contain
   shortcut-able examples.
4. CC BY-SA 4.0 means derived manifests/artifacts must preserve attribution and
   share-alike obligations.
```

Recommended initial construction:

```text
public_context:
  empty or only task instruction.

agent_views:
  two supporting titles/sentence groups split across Agent A/B;
  add small distractor paragraphs per agent.

single_full_info:
  question + all selected support/distractor context.

matched TextMAS:
  agent summaries -> final Solver.
```

## 5. 2WikiMultiHopQA Notes

公开依据：

```text
Repository:
  https://github.com/Alab-NII/2wikimultihop

Paper:
  https://arxiv.org/abs/2011.01060
```

关键信息：

```text
License:
  Apache-2.0.

Repository notes:
  para_with_hyperlink contains paragraph information:
    id
    title
    sentences
    mentions

Evaluation:
  updated evaluator can use evidences_id and answer_id.

Paper:
  dataset includes evidence information containing a reasoning path.
```

Phase C implications:

```text
1. Strong candidate for reasoning-path-aware failure taxonomy.
2. Apache-2.0 is easier for derived tooling/artifacts than share-alike sources.
3. Need dry inspect actual HF/GitHub data files before writing builder because
   field layout may differ between mirrors and raw files.
```

## 6. License / Attribution Discipline

For every real Phase C manifest:

```text
source.name
source.version
source.url
source.source_sample_id
metadata.license
metadata.attribution_required
metadata.derived_from
```

must be present either directly in each sample or in a manifest-level source
registry before paper-level reporting.

Rules:

```text
CC BY 4.0:
  preserve attribution.

CC BY-SA 4.0:
  preserve attribution and share-alike constraints for derived releases.

Apache-2.0:
  preserve license/notice obligations.
```

If source license/terms are unclear:

```text
do not run main experiment;
keep only as local dry-run candidate until clarified.
```

## 7. Field Dry Inspection Protocol

Before any real builder is written:

```text
1. Download only metadata / a tiny calibration preview when allowed.
2. Print field keys and nested key summaries.
3. Record exact source revision / URL / hash.
4. Confirm answer, alias, evidence, support IDs, split fields.
5. Confirm no held-out/test sample content is inspected for prompt/scorer tuning.
6. Update this audit doc with observed local field schema.
```

Dry inspection output must be local-only:

```text
outputs/p2_phase_c_data_source_audits/<source>_<date>/
  field_summary.json
  metrics.jsonl
  summary.json
```

Local tool:

```text
/data1/luyifei/drla/drla/scripts/inspect_p2_phase_c_dataset_fields.py
```

Current tool self-check:

```text
outputs/p2_phase_c_data_source_audits/field_inspect_selfcheck_20260601

status:
  pass

meaning:
  The script can detect question / answer / evidence / context / title /
  sentence-like fields from a local preview without saving raw sample content.
```

Schema-example dry inspection:

```text
outputs/p2_phase_c_data_source_audits/field_inspect_records_example_20260601

status:
  pass

meaning:
  The tool can inspect existing Phase C record examples. This is a tooling
  check only, not a dataset audit and not an experiment.
```

Important boundary:

```text
The inspector reports structural field paths, type/length summaries and record
hashes. It must not be used to pick examples, tune prompts, construct held-out
records, or infer benchmark success.
```

2026-06-01 tiny first-rows preview inspections:

```text
HotpotQA:
  artifact:
    outputs/p2_phase_c_data_source_audits/
    hotpotqa_distractor_field_inspect_20260601
  source preview:
    HF datasets-server first-rows, hotpotqa/hotpot_qa, distractor/train
  preview rows inspected:
    20
  detected fields:
    id, question, answer, context, supporting_facts,
    context.title, context.sentences,
    supporting_facts.title
  shardability:
    possible_evidence_split

MuSiQue:
  artifact:
    outputs/p2_phase_c_data_source_audits/
    musique_answerable_field_inspect_20260601
  source preview:
    HF datasets-server first-rows, bdsaglam/musique, answerable/train
  preview rows inspected:
    17
  detected fields:
    id, question, answer, answer_aliases, paragraphs,
    paragraphs.title, paragraphs.is_supporting,
    question_decomposition.question,
    question_decomposition.answer,
    question_decomposition.paragraph_support_idx
  shardability:
    possible_evidence_split
  caveat:
    mirror metadata/license still needs confirmation against official MuSiQue
    release before full manifest construction.

2WikiMultiHopQA:
  artifact:
    outputs/p2_phase_c_data_source_audits/
    2wiki_framolfese_field_inspect_20260601
  source preview:
    HF datasets-server first-rows, framolfese/2WikiMultihopQA, default/train
  preview rows inspected:
    20
  detected fields:
    id, question, answer, context, supporting_facts, evidences,
    context.title, context.sentences,
    supporting_facts.title
  shardability:
    possible_evidence_split
  caveat:
    mirror metadata/license still needs confirmation against official 2Wiki
    release before full manifest construction.
```

Interpretation:

```text
All three first-row previews expose the minimum fields needed for evidence-split
QA builder design. This is only field/schema evidence. It does not admit a
benchmark, does not construct train/heldout splits, and does not justify model
runs before manifest/leakage/scorer protocols are locked.
```

2026-06-01 seeded 300-row field inspections:

```text
HotpotQA:
  source rows:
    outputs/p2_phase_c_data_source_audits/
    hf_rows_hotpotqa_distractor_train_300_seed20260601
  field audit:
    outputs/p2_phase_c_data_source_audits/
    hotpotqa_distractor_train_300_field_inspect_20260601
  status:
    pass
  shardability:
    possible_evidence_split

MuSiQue:
  source rows:
    outputs/p2_phase_c_data_source_audits/
    hf_rows_musique_answerable_train_300_seed20260601
  field audit:
    outputs/p2_phase_c_data_source_audits/
    musique_answerable_train_300_field_inspect_20260601
  status:
    pass
  shardability:
    possible_evidence_split
```

2026-06-01 calibration-only manifest drafts:

```text
HotpotQA:
  records:
    outputs/p2_phase_c_records/
    hotpotqa_calibration_records_200_seed20260601
  manifest:
    outputs/p2_phase_c_manifests/
    hotpotqa_calibration_manifest_200_seed20260601
  validation:
    pass, errors 0, warnings 271

MuSiQue:
  records:
    outputs/p2_phase_c_records/
    musique_calibration_records_200_seed20260601
  manifest:
    outputs/p2_phase_c_manifests/
    musique_calibration_manifest_200_seed20260601
  validation:
    pass, errors 0, warnings 207
```

No SwanLab run is allowed for data inspection.

## 8. Current Decision Boundary

Allowed now:

```text
data-source field/license docs
tiny schema/metadata dry inspection scripts
manifest/scorer/leakage skeletons
```

Requires explicit Branch C confirmation:

```text
download full datasets
construct real Phase C manifests
run capable text agents
inspect held-out generations
report Phase C benchmark results
```
