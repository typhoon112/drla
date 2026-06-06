# P2 Benchmark Redesign 候选清单

更新日期：2026-06-01

> 状态：Branch B 的预备清单。本文不替换 benchmark、不跑 held-out、不证明任何 P2 主结论。它只说明如果接受 P2-D4 推荐的 benchmark redesign 分支，应该从哪些候选族开始、如何准入、如何避免再次把 base capability floor 误报成 communication 结果。

## 1. 设计目标

Branch B 的目标：

```text
保留 frozen official CoLA 作为 same-substrate latent communication substrate，
先找到 Single CoLA Solver 与 Role TextMAS 都能承载的任务，
再进入 text-vs-latent MAS 主实验。
```

Branch B 不做：

```text
不训练 LoRA / adapter / fuser。
不把 official8 solver-to-solver diagnostic 当主表。
不在 admitted_tasks=[] 的任务上跑 latent-vs-text。
不使用 held-out 做 prompt/protocol repair。
```

## 2. Candidate Families

### Family 1: Official8-compatible role diagnostics

Source:

```text
/data1/luyifei/Cola-DLM/code/generate_task_data

tasks:
  lambada
  mmlu
  obqa
  hellaswag
  race
  siqa
  squad
  story_cloze
```

Prepared artifact:

```text
script:
  /data1/luyifei/drla/drla/scripts/prepare_cola_p2_official8_role_candidates.py

smoke:
  /data1/luyifei/drla/outputs/p2_benchmark_redesign/
  official8_role_candidates_smoke_20260601

full:
  /data1/luyifei/drla/outputs/p2_benchmark_redesign/
  official8_role_candidates_20260601

combined:
  /data1/luyifei/drla/outputs/p2_benchmark_redesign/
  official8_role_candidates_20260601/p2_official8_role_candidates.jsonl

rows:
  official8_obqa = 500
  official8_mmlu = 14042
  official8_race = 4887
  official8_hellaswag = 10042
  official8_siqa = 1954
  official8_story_cloze = 1871
  total = 33296
```

This preparation only normalizes existing official CoLA JSONL rows into the
P2 gate schema. It does not run CoLA generation, inspect held-out data, or
admit any benchmark.

Why first:

```text
These are the only tasks with official CoLA substrate history. They are not
enough for the strongest true-MAS claim, but they are the cleanest way to
build a capability-matched role/channel benchmark before changing the
substrate.
```

Promising subfamilies:

| Task | Why useful | Main risk |
|---|---|---|
| `obqa` | short MCQ, already science QA-like | may be too close to old diagnostic |
| `mmlu` | broad MCQ, known CoLA format | subject mixture may hide failures |
| `race` | reading comprehension with evidence context | long prompts may harm CoLA |
| `hellaswag` | continuation MCQ, stable scorer | role decomposition may be artificial |
| `siqa` | commonsense/social MCQ | role outputs may not improve over single |
| `story_cloze` | two-choice narrative ending | limited answer space, weak MAS claim |
| `squad` | extractive QA | parsing/evaluation needs care |
| `lambada` | next-word prediction | weak MAS relevance |

Admissible claim:

```text
channel/protocol diagnostic under frozen CoLA, not final true-MAS benchmark.
```

### Family 2: Official8-derived decomposed tasks

Construction idea:

```text
Use official8-compatible source examples, but define roles that have natural
work:
  evidence selector
  option eliminator
  consistency checker
  final solver
```

Examples:

```text
RACE:
  Planner extracts relevant sentence/evidence.
  Critic checks distractors.
  Refiner writes compact evidence state.
  Solver answers.

OpenBookQA / MMLU:
  Planner states relevant fact.
  Critic checks option conflicts.
  Refiner eliminates impossible options.
  Solver answers.

Story Cloze:
  Planner states narrative causal constraint.
  Critic checks each ending.
  Refiner selects consistency signal.
  Solver answers.
```

Why useful:

```text
This is closer to MAS than plain single-answer QA, while staying near CoLA's
known task distribution.
```

Risk:

```text
If roles are synthetic and do not change final behavior, the result remains a
diagnostic rather than a strong MAS claim.
```

### Family 3: Capability-matched short MCQ outside official8

Candidate sources:

```text
short OpenBookQA-like MCQ
commonsense MCQ with short question/options
small-domain science QA where frozen CoLA single solver is above floor
```

Why lower priority:

```text
ARC/GPQA/MedQA already showed that "reasonable MCQ" does not guarantee CoLA
capability. Any new source needs the full D1/D2/D3 gate again.
```

### Family 4: Naturally decomposable but CoLA-uncertain tasks

Candidate sources:

```text
HotpotQA / 2WikiMultiHopQA / MuSiQue evidence-split QA
planner-coder-tester-reviewer code tasks
distributed information gathering tasks
```

Why not first:

```text
These are better true-MAS tasks, but frozen official CoLA is unlikely to pass
without adaptation. They should be used after either:
  a. Single + Role CoLA gate passes, or
  b. Branch A substrate adaptation is accepted.
```

## 3. Admission Protocol

Every candidate family needs:

```text
manifest.json
candidate JSONL
source dataset and split
answer_type
parser/scorer version
calibration/held-out split
Single CoLA Solver gate
Role TextMAS gate
leakage audit
```

Minimum gate:

```text
nonempty_rate >= 0.95
parseable_rate >= 0.90
accuracy > random/task floor + margin
Single gate_pass = true
Role gate_pass = true
not smoke
no gold/scorer/selected_prediction in online prompts
```

For official8-derived candidates:

```text
Do not mix old P1/P2 official8 diagnostics with redesigned role gates.
Use new artifact roots and protocol_version fields.
Report that this is a capability-matched diagnostic unless the role structure
is genuinely task-required.
```

## 4. Recommended First Redesign Pass

If Branch B is accepted, start with:

```text
B1:
  official8-compatible MCQ role gate on obqa, mmlu, race, hellaswag, siqa,
  story_cloze using official8_role_candidates_20260601.

B2:
  build deterministic calibration/held-out split and choose tasks where Single
  + Role pass on calibration.

B3:
  run locked held-out gate only for passed tasks.

B4:
  only then run TextMAS vs LatentMAS under corrected Agent-B channel protocol.
```

Why this order:

```text
It preserves the frozen CoLA claim boundary.
It avoids another prompt-only repair loop on tasks where CoLA is at floor.
It creates a clean bridge from P1/P2 packet diagnostics to role-conditioned
communication without pretending official8 is the final true-MAS benchmark.
```

## 5. Stop Conditions

Stop Branch B and reconsider Branch A if:

```text
No official8-compatible role task passes calibration.
Role pass requires excessive parser hacks or scorer leakage.
Held-out gate fails after calibration success.
The only passing tasks are too trivial to support even a diagnostic claim.
```

If any stop condition is hit:

```text
Do not keep prompt-tuning.
Return to P2-D4 branch decision and choose substrate adaptation or external
capable text-MAS validation.
```
