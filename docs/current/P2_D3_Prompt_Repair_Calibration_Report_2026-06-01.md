# P2-D3 Prompt Repair Calibration Report

更新日期：2026-06-01

> 状态：P2-D3 初始 calibration-only prompt/protocol repair 记录。本文只使用 P2-D2 calibration split，不使用 held-out。当前没有任何任务在 calibration 上通过完整 Single + Role TextMAS 双门。

## 1. 输入边界

Calibration split:

```text
/data1/luyifei/drla/outputs/p2_capability_gate/locked_splits_seed20260602_20260601/calibration.jsonl
```

Held-out split:

```text
/data1/luyifei/drla/outputs/p2_capability_gate/locked_splits_seed20260602_20260601/heldout.jsonl
```

本轮只读取 calibration；held-out 未用于 prompt/protocol repair。

## 2. Prompt Variants

当前脚本支持：

```text
generic_v1:
  P2-D1 原始通用 prompt。

cola_fewshot_v1:
  参考 official CoLA MMLU/OBQA 风格，选择题用 few-shot answer-text prompt，
  numeric/code 也使用更接近 answer/completion 的格式。

answer_state_v1:
  Planner/Critic/Refiner/Solver 传递 compact candidate answer state，
  但下游仍看到上游 raw role text 的压缩版本。

answer_state_structured_v1:
  上游 raw role text 先被本地解析为 Candidate: X/unknown，
  下游只看到结构化 answer-state，不再看到完整 raw role message。

role_plan_ignore_v1:
  保留 generic_v1 的 Planner/Critic/Refiner plan semantics，
  但 final Solver 被明确告知 upstream state 可能 noisy/irrelevant，
  不有用时可以忽略。
```

执行脚本：

```text
/data1/luyifei/drla/drla/scripts/run_cola_p2_capability_gate.py
```

新增参数：

```text
--prompt-variant generic_v1|cola_fewshot_v1|answer_state_v1|answer_state_structured_v1|role_plan_ignore_v1
```

## 3. Non-code Single-mode Calibration Sweep

Generic baseline:

```text
/data1/luyifei/drla/outputs/p2_capability_gate/eval_calibration_noncode_single_generic_v1_20260601
```

CoLA few-shot variant:

```text
/data1/luyifei/drla/outputs/p2_capability_gate/eval_calibration_noncode_single_cola_fewshot_v1_20260601
```

Aggregate:

```text
/data1/luyifei/drla/outputs/p2_capability_gate/aggregate_calibration_noncode_single_prompt_variants_20260601
```

Result:

| Task | Variant | Single Acc | Parseable | Single Pass |
|---|---|---:|---:|---:|
| arc_easy | generic_v1 | 15.79% | 82.46% | no |
| arc_easy | cola_fewshot_v1 | 17.54% | 71.05% | no |
| arc_challenge | generic_v1 | 21.67% | 81.67% | no |
| arc_challenge | cola_fewshot_v1 | 13.33% | 63.33% | no |
| gpqa_diamond | generic_v1 | 32.50% | 92.50% | yes |
| gpqa_diamond | cola_fewshot_v1 | 20.00% | 70.00% | no |
| medqa | generic_v1 | 24.31% | 94.51% | no |
| medqa | cola_fewshot_v1 | 21.96% | 79.22% | no |
| gsm8k | generic_v1 | 1.89% | 98.86% | no |
| gsm8k | cola_fewshot_v1 | 1.89% | 100.00% | no |

Interpretation:

```text
cola_fewshot_v1 does not improve the candidate set.
generic_v1 remains the better baseline for most tasks.
GPQA-Diamond generic_v1 passes single-mode calibration only.
```

## 4. GPQA Role TextMAS Check

Because GPQA-Diamond passed single-mode calibration under `generic_v1`, we ran
Role TextMAS on the same calibration split:

```text
/data1/luyifei/drla/outputs/p2_capability_gate/eval_calibration_gpqa_role_textmas_generic_v1_20260601
```

Aggregate:

```text
/data1/luyifei/drla/outputs/p2_capability_gate/aggregate_calibration_gpqa_generic_single_role_20260601
```

Result:

```text
single accuracy = 32.50%
single parseable = 92.50%
single gate_pass = true

role_textmas accuracy = 25.00%
role_textmas parseable = 50.00%
role_textmas gate_pass = false

admitted_for_main = false
```

## 5. Conclusion

P2-D3 initial prompt repair did not produce an admitted benchmark.

Important boundary:

```text
Do not run held-out gate yet.
Do not run P2 text-vs-latent main table yet.
Do not treat GPQA single-only calibration pass as MAS readiness.
```

Next repair should focus on Role TextMAS protocol, not only single-solver prompt:

```text
simplify role messages
reduce planner/critic/refiner verbosity
make solver receive a compact answer-state rather than noisy generated prose
try role_textmas with shorter role budgets on calibration only
consider task-specific MCQ parser/answer-text scoring audit before held-out
```

## 6. P2-D3.1 Answer-State Repair Result

Implemented variants:

```text
answer_state_v1
answer_state_structured_v1
```

Script:

```text
/data1/luyifei/drla/drla/scripts/run_cola_p2_capability_gate.py
```

Artifacts:

```text
GPQA raw answer-state:
  /data1/luyifei/drla/outputs/p2_capability_gate/
  eval_calibration_gpqa_answer_state_v1_20260601

GPQA structured answer-state:
  /data1/luyifei/drla/outputs/p2_capability_gate/
  eval_calibration_gpqa_answer_state_structured_v1_20260601

ARC/MedQA structured answer-state:
  /data1/luyifei/drla/outputs/p2_capability_gate/
  eval_calibration_mcq_answer_state_structured_v1_20260601

Aggregate:
  /data1/luyifei/drla/outputs/p2_capability_gate/
  aggregate_calibration_protocol_repair_answer_state_20260601

Failure taxonomy:
  /data1/luyifei/drla/outputs/p2_capability_gate/
  audit_protocol_repair_failures_all_20260601

All repair aggregate:
  /data1/luyifei/drla/outputs/p2_capability_gate/
  aggregate_calibration_protocol_repair_all_20260601
```

Parser fix:

```text
During smoke, `Answer: <one option letter from A/B/C/D>` was found to be
parsed as A because the placeholder contained option labels. The parser now
strips answer placeholders before extracting a multiple-choice prediction.
This prevents format-template echo from being counted as a valid answer.
```

Calibration results:

| Task | Variant | Single Acc | Single Parse | Role Acc | Role Parse | Gate |
|---|---|---:|---:|---:|---:|---:|
| gpqa_diamond | generic_v1 | 32.50% | 92.50% | 25.00% | 50.00% | no |
| gpqa_diamond | answer_state_v1 | 20.00% | 90.00% | 5.00% | 40.00% | no |
| gpqa_diamond | answer_state_structured_v1 | 20.00% | 90.00% | 15.00% | 67.50% | no |
| gpqa_diamond | role_plan_ignore_v1 | 25.00% | 85.00% | 2.50% | 35.00% | no |
| arc_challenge | answer_state_structured_v1 | 20.00% | 88.33% | 26.67% | 85.00% | no |
| arc_easy | answer_state_structured_v1 | 23.68% | 71.05% | 17.54% | 85.09% | no |
| medqa | answer_state_structured_v1 | 20.00% | 94.90% | 24.31% | 92.94% | no |

Interpretation:

```text
answer_state_v1 is negative. It reduces role messages conceptually, but raw
role text still pollutes downstream Solver prompts.

answer_state_structured_v1 partly fixes message pollution. On GPQA role,
parseability improves from 40.00% to 67.50% vs raw answer_state_v1, and role
accuracy improves from 5.00% to 15.00%. It still underperforms generic_v1 role.

ARC-Challenge role under structured answer-state reaches 26.67%, but it still
misses the random-floor + margin gate and parseability is only 85.00%.

MedQA structured role passes parseability but stays at the random floor.

role_plan_ignore_v1 is also negative on GPQA. The instruction that upstream
state may be ignored does not stabilize CoLA's final Solver format; role
accuracy drops to 2.50% and parseability to 35.00%.

No task is admitted after P2-D3.1.
```

Paired failure taxonomy:

| Task | Variant | Role - Single Acc | Single-only Correct | Role-only Correct | Role Unparseable |
|---|---|---:|---:|---:|---:|
| gpqa_diamond | generic_v1 | -7.50 pp | 9 / 40 | 6 / 40 | 12 / 40 |
| gpqa_diamond | answer_state_v1 | -15.00 pp | 7 / 40 | 1 / 40 | 15 / 40 |
| gpqa_diamond | answer_state_structured_v1 | -5.00 pp | 6 / 40 | 4 / 40 | 9 / 40 |
| gpqa_diamond | role_plan_ignore_v1 | -22.50 pp | 10 / 40 | 1 / 40 | 14 / 40 |
| arc_challenge | answer_state_structured_v1 | +6.67 pp | 9 / 60 | 13 / 60 | 7 / 60 |
| arc_easy | answer_state_structured_v1 | -6.14 pp | 20 / 114 | 13 / 114 | 5 / 114 |
| medqa | answer_state_structured_v1 | +4.31 pp | 40 / 255 | 51 / 255 | 13 / 255 |

Taxonomy interpretation:

```text
Structured answer-state is not uniformly bad: ARC-Challenge and MedQA show
positive role-minus-single deltas on calibration. However, both remain at or
near random floor, and ARC-Challenge still fails parseability.

GPQA remains the only single-mode calibration pass under generic_v1, but every
Role TextMAS variant harms GPQA relative to its single solver.

This means the current blocker is not merely output parsing. The role protocol
is sometimes adding signal, sometimes adding harm, while the underlying CoLA
solver stays close to task floor on most candidate tasks.
```

Current boundary:

```text
Do not run held-out gate yet.
Do not run P2 text-vs-latent main tables.
Do not train fuser/adapter from these calibration failures.
```

## 7. Next Decision Point

Before another prompt tweak, use the completed failure review plus local paper
check:

```text
LatentMAS source:
  /data1/luyifei/latent_reasoning_papers/agent_comm_downloads/
  2511.20639_LatentMAS

Coconut text:
  /data1/luyifei/latent_reasoning_papers/2412.06769_Coconut.txt
```

Relevant observations:

```text
LatentMAS sequential MCQ prompts still use full plan/feedback/refined-plan
roles. Its latent path transfers layer-wise KV working memory containing both
context and newly generated latent thoughts. It is not a compressed answer
label channel.

LatentMAS also tells the final solver that latent information may contain
irrelevant content and can be ignored. This is important because our current
CoLA TextMAS prompts force the solver to consume noisy upstream text states.

Coconut's continuous-thought results rely on staged curriculum/internalization
of latent reasoning. This cautions against expecting a one-shot prompt-only
answer-state compression to create a reliable latent/role state interface.
```

Protocol implication:

```text
Do not continue blindly compressing roles into answer labels.
The first faithful repair attempt, role_plan_ignore_v1, is also negative for
CoLA. Therefore, further prompt-only Role TextMAS repair has weak evidence.
The next branch should be substrate adaptation or benchmark redesign unless a
new protocol is justified before being run on calibration.
```

Before another experiment:

```text
1. Compare generic_v1 vs answer_state_structured_v1 failure cases.
2. Separate base solver floor, parser/format failure, and role protocol collapse.
3. Decide whether to repair Role TextMAS prompt semantics or move to substrate
   adaptation / benchmark redesign.
4. If no calibration task can pass Single + Role gates, move to substrate
   adaptation or benchmark redesign rather than P2 main tables.
```

Previously planned design, now evaluated:

```text
Planner:
  produce a compact candidate answer state.

Critic:
  revise the candidate answer state and mark uncertainty.

Refiner:
  merge previous states into one refined candidate.

Solver:
  output final answer only.
```

Boundary:

```text
Use calibration split only.
Do not inspect held-out sample-level data.
Do not run P2 text-vs-latent main tables before held-out gate.
```

Primary target:

```text
GPQA-Diamond, because generic_v1 already passed single-mode calibration.
```

Fallback:

```text
GPQA, ARC, and MedQA calibration still fail after compact/structured
answer-state repair. The next step is not held-out; it is a failure taxonomy
and branch decision.
```
