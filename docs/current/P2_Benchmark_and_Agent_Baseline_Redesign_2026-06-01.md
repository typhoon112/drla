# P2 Benchmark 与 Agent Baseline 重设方案

更新日期：2026-06-01

> 状态：当前 P2 路线修订。本文覆盖后续 P2 benchmark 选择、agent baseline 架构、CoLA 权重准入、latent receiver/fuser 触发条件和可发表证据标准。旧 official8 solver-to-solver 结果保留为 channel diagnostic，不再作为真实 MAS 主证据。
>
> 2026-06-01 post-Family1 更新：早前“先修复 Role TextMAS，再做 held-out gate”
> 的执行口径已经被
> `/data1/luyifei/drla/docs/current/P2_Post_Family1_Complete_Execution_Plan_2026-06-01.md`
> supersede。当前最高优先级执行方案是
> `/data1/luyifei/drla/docs/current/P2_Locked_Complete_Execution_Scheme_2026-06-01.md`。
> Branch B Family 1 已执行并停止；当前不能直接跑 held-out gate、
> P2 text-vs-latent 主表或 fuser/adapter training。
>
> P2-D4 后的 Branch B Family 1 路线见
> `/data1/luyifei/drla/docs/current/P2_Branch_B_Execution_Plan_2026-06-01.md`，
> 但该路线已在 official8-compatible calibration 与 native alignment audit 后停止。
> 当前锁定顺序是 Branch C 先验证 true MAS benchmark/protocol，再按需要进入
> Branch A CoLA substrate/interface adaptation，最后进入 Phase E 主比较。

## 1. 总结决策

当前 P2 不能继续把 `official8 + solver A -> solver B + message_only` 当作主线 multi-agent communication 评估。它只说明 latent packet 在严格 receiver-only 边界下有弱信号，不说明真实 agent collaboration 成立，也不说明 latent 优于 text。

后续主线改为：

```text
same-substrate Cola A -> Cola B
+ role-conditioned MAS protocol
+ context-visible or full-working-memory latent handoff
+ capability-gated benchmarks
+ matched-vs-corrupted and text-vs-latent paired comparison
```

保留的结论：

```text
P2-A/P2-B: packet construction and distribution audit are valid.
P2-C: decoder-free compatibility model shows matched latent separability.
P2-D official8 receiver-only: latent has marginal signal over empty input.
```

降级的结论：

```text
official8 solver-to-solver message_only is channel diagnostic only.
It is not the main MAS benchmark.
It does not prove latent > text.
It does not prove robust payload-specific communication because wrong_block remains anomalously strong.
```

## 2. 文献约束

LatentMAS:

```text
Sequential MAS uses Planner -> Critic -> Refiner -> Solver.
Each next role receives the original question q plus previous agent output.
Latent collaboration transfers layer-wise KV working memory, including input context and generated latent thoughts.
Benchmarks include GSM8K, AIME24/25, GPQA-Diamond, MedQA, ARC-E/C, MBPP+, HumanEval+.
```

Vision Wormhole:

```text
Keep the same role workflow, prompts, number of agents, and budgets.
Only replace the communication medium.
Final solver still receives the target question and latent/text reference.
```

C2C:

```text
Receiver has original question context.
Sharer provides contextual understanding.
Useful cache communication is learned through projection/fusion and residual gating.
Naive replacement can be destructive; layer/gate choice matters.
```

ThoughtComm:

```text
Latent communication can require a learned shared thought space and prefix adapter.
Shared/private thought routing matters when agents have different roles.
```

The Latent Space survey and MAS benchmark literature:

```text
Latent collaboration should be evaluated as semantic fidelity, shared cognition, and task utility.
Real MAS needs explicit architecture, role, protocol, communication content, and evaluation target.
Single-answer QA without role decomposition is weak evidence for multi-agent collaboration.
```

## 3. CoLA 架构与权重准入判断

CoLA 架构满足 P2 研究目标：

```text
Text VAE maps text <-> continuous latent sequences.
Block-causal DiT operates in latent space.
Same-substrate Cola A/B makes training-free latent handoff plausible.
```

CoLA 权重不自动满足所有新 benchmark：

```text
Released checkpoint has official reference only on:
  lambada, mmlu, obqa, hellaswag, race, siqa, squad, story_cloze

It is not instruction-tuned and not RLHF-aligned.
Prompt format and answer parser are sensitive.
New benchmark must pass a capability gate before entering main P2 tables.
```

Do not treat a failed high-difficulty task as evidence that latent communication fails. It may simply be CoLA base capability floor.

Executable capability gate:

```text
prepare:
  drla/scripts/prepare_cola_p2_candidate_benchmarks.py

evaluate:
  drla/scripts/run_cola_p2_capability_gate.py
```

Gate outputs:

```text
generations.jsonl
metrics.jsonl
task_summary.csv
summary.json
```

Current prepared candidate data, 2026-06-01:

```text
/data1/luyifei/drla/outputs/p2_capability_gate/data_20260601

ARC-Easy validation: 570
ARC-Challenge validation: 299
GSM8K test: 1319
MBPP+ test: 378
HumanEval+ test: 164
GPQA-Diamond test: 198
MedQA test: 1273
total prepared rows: 4201
```

Current protocol smoke:

```text
/data1/luyifei/drla/outputs/p2_capability_gate/eval_smoke_arc_easy_both_20260601

single + Role TextMAS, 1 ARC-E sample, CUDA generation path OK.
This smoke validates code/schema only; it is not a scientific result.
```

Rules:

```text
Pure gate/eval runs must use swanlab_mode=disabled.
The gate runs on CUDA because CoLA generation is GPU-native.
Smoke runs can validate formatting and parser code but cannot admit a benchmark.
Code tasks require --enable-code-execution for main gate claims; syntax-only pass is not enough.
GPQA-Diamond and MedQA now have pinned HF sources; changing them requires a new manifest.
```

Formal full gate result, 2026-06-01:

```text
/data1/luyifei/drla/outputs/p2_capability_gate/aggregate_full_20260601

admitted_tasks = []
```

Interpretation:

```text
None of the 7 candidate tasks currently passes both Single CoLA Solver and
Role TextMAS gates. These tasks must not enter P2 text-vs-latent main tables
until prompt/protocol repair is performed on a separate calibration split and
then re-evaluated on a locked held-out split.
```

Locked split for repair, 2026-06-01:

```text
/data1/luyifei/drla/outputs/p2_capability_gate/locked_splits_seed20260602_20260601

calibration = 842
heldout = 3359
split_seed = 20260602
overlap = 0
```

Prompt/protocol repair may use calibration only. Held-out is reserved for the
next locked capability gate.

Initial calibration repair result:

```text
/data1/luyifei/drla/outputs/p2_capability_gate/
aggregate_calibration_noncode_single_prompt_variants_20260601

/data1/luyifei/drla/outputs/p2_capability_gate/
aggregate_calibration_gpqa_generic_single_role_20260601
```

```text
cola_fewshot_v1 does not improve the candidate set.
generic_v1 GPQA-Diamond passes single-mode calibration only.
GPQA-Diamond Role TextMAS still fails.
admitted_tasks remains [].
```

## 4. 两个合法通信协议

### 4.1 Canonical: context-visible role MAS

This is the next main protocol.

```text
Agent B input = original question q + role instruction + previous agent message/latent state
Scorer input = only final Solver output
```

This is not leakage. It matches LatentMAS/Vision Wormhole/C2C practice. Leakage means scorer or online receiver sees gold answers, scorer outputs, selected_prediction, or Agent-A decoded replay as final answer.

### 4.2 Diagnostic: self-contained message-only

Message-only is valid only if one of the following is true:

```text
A text message explicitly contains q / evidence / role output needed by B.
A latent packet contains full working memory, including prefix/context state.
```

Current `decode_text_so_far` or selected latent blocks without full context are too weak to serve as the main MAS protocol. They remain stress tests.

## 5. Agent Baseline Architecture

The old baseline:

```text
solver A -> message_only -> solver B
```

is demoted to P2-D0 channel diagnostic.

The new default sequential baseline:

```text
Planner -> Critic -> Refiner -> Solver
```

Role contracts:

```text
Planner:
  input: q
  output: plan, no final answer unless task format requires concise answer sketch

Critic:
  input: q + planner message
  output: weaknesses, corrections, missing evidence

Refiner:
  input: q + planner message + critic message
  output: refined plan / reasoning state

Solver:
  input: q + final upstream text or latent state
  output: final answer only
```

Hierarchical alternative:

```text
Domain Expert A/B/C -> Summarizer/Solver
```

Use hierarchical only when task naturally benefits from multiple perspectives or evidence partitions.

## 6. Benchmark Ladder

### Tier 0: official8 diagnostic

Use for:

```text
CoLA official substrate comparison
P1 halt/readiness continuity
packet build/distribution audit
channel boundary smoke
matched-vs-corrupted latent sanity checks
```

Do not use for the main claim that latent MAS beats text MAS.

### Tier 1: LatentMAS-aligned capability-gated tasks

Candidate tasks:

```text
ARC-E / ARC-C
OpenBookQA / MMLU-style multiple choice
GPQA-Diamond / MedQA multiple choice
GSM8K short numeric answer
```

High-risk tasks:

```text
AIME24/25:
  likely floor effect; use only after single-CoLA baseline is nontrivial.

MBPP+ / HumanEval+:
  code generation requires exact syntax and executable functions.
  Use only after parseability and execution sanity pass.
```

### Tier 2: naturally decomposable MAS tasks

Use for the strongest agent-to-agent claim:

```text
HotpotQA / 2WikiMultiHopQA / MuSiQue with evidence split
code planner-coder-tester-reviewer workflows
MultiAgentBench-style interactive tasks
PeopleJoin-style distributed information gathering
```

### 6.4 P2-D1 capability-gate protocol

Before a benchmark enters P2 main text-vs-latent tables, run:

```text
Single CoLA Solver:
  q -> CoLA -> answer

Role TextMAS:
  Planner(q) -> Critic(q, planner) -> Refiner(q, planner, critic) -> Solver(q, refiner)
```

Admission is task-level, not global:

```text
nonempty_rate >= 0.95
parseable_rate >= 0.90
accuracy > task_random_floor + margin
no gold/scorer/selected_prediction fields in online prompts
not a max-sample smoke run
```

For code benchmarks:

```text
syntax and entry-point sanity are only a pre-gate.
Executable unit-test pass rate must be added before code enters a main P2 table.
```

If Single CoLA Solver is at floor, the task is a base-capability failure.  It
must not be used to claim that latent communication fails.  If Single CoLA
Solver is nontrivial but Role TextMAS collapses, first repair prompt/role
protocol before comparing communication media.

These tasks are preferred for paper-level claims because collaboration is required by the task structure, not imposed after the fact.

## 7. Capability Gate

Every new benchmark must pass a no-training CoLA gate before main P2 experiments.

Run:

```text
Single CoLA Solver
Role TextMAS baseline
Basic parser / format audit
```

Minimum criteria:

```text
nonempty_rate >= 95%
parseable_rate >= 90%
accuracy or task score clearly above trivial/random floor
seed/template variance is reportable and not dominating the signal
TextMAS changes behavior relative to Single Solver in a measurable way
```

If a task fails the gate:

```text
Do not use it as main evidence.
Keep it only as stress/OOD diagnostic.
Do not tune latent communication on a benchmark where CoLA cannot produce usable final answers.
```

Historical Branch B Family 1 first pass:

```text
script:
  /data1/luyifei/drla/drla/scripts/prepare_cola_p2_official8_role_candidates.py

artifact:
  /data1/luyifei/drla/outputs/p2_benchmark_redesign/
  official8_role_candidates_20260601

tasks:
  official8_obqa, official8_mmlu, official8_race, official8_hellaswag,
  official8_siqa, official8_story_cloze

rows:
  33296

meaning:
  data is ready for capability-gate split/eval only.
  It is not an admitted benchmark.
```

## 8. Paper-Level Comparison Matrix

For every admitted benchmark, compare:

```text
Single Solver:
  q -> final answer

TextMAS:
  q + role prompts + text messages -> final Solver answer

LatentMAS no-fuser, context-visible:
  q + role prompts + A latent working memory -> final Solver answer

LatentMAS no-fuser, thought-only:
  q + role prompts + generated latent blocks only -> final Solver answer

Latent corrupt controls:
  wrong_sample, same_task_wrong_sample, wrong_block, shuffle, noise, rotation

LatentMAS + lightweight fuser/gate:
  q + role prompts + fused A latent -> final Solver answer
```

Required metrics:

```text
accuracy / task score
paired delta vs Single Solver
paired delta vs TextMAS
matched vs corrupted deltas
token count
latent block count
wall-clock time
nonempty / parseable / format adherence
confidence intervals
failure-case taxonomy
```

## 9. Full Working Memory vs Thought-Only

Same-substrate CoLA makes training-free handoff plausible only when the transmitted state is complete enough.

Must distinguish:

```text
thought_only:
  A generated latent blocks only.

context_plus_thought:
  prefix/context state + A generated latent reasoning blocks.

full_working_memory:
  all receiver-consumable state required to reconstruct A's reasoning context.
```

Main no-fuser claim requires at least `context_plus_thought`. `thought_only` is an ablation, not the canonical communication protocol.

## 10. Fuser / Receiver Adapter Trigger

Do not train a fuser merely because latent dimensions look similar or different. Same-CoLA already gives approximate distribution compatibility.

Train a lightweight fuser only if no-fuser experiments show one of:

```text
matched latent does not beat corrupted controls robustly.
wrong_block or same-task wrong-sample remains anomalously strong.
latent is consistently worse than TextMAS under matched budgets.
full_working_memory handoff works but thought_only is not usable.
```

Fuser objective:

```text
A_latent + B_context -> B_usable_state
```

Recommended architecture:

```text
block/slot encoder over A latent blocks
B-context conditioned attention or pooling
residual fusion into B state
learnable gate over blocks/layers
corruption-aware contrastive loss
optional TextMAS teacher hidden/logit distillation
```

Training rules:

```text
All fuser/adapter training must use CUDA.
All training must log to SwanLab cloud.
valid_interval <= 10 step.
Save metrics.jsonl, best_checkpoint.pt, last_checkpoint.pt.
Use decoder/text teacher only for training labels or distillation.
Never expose A decoded replay text to scorer during final evaluation.
```

## 11. Implementation Phases

P2-D0: Freeze existing official8 channel diagnostic.

```text
Document current receiver-only result.
Audit wrong_block and same-task controls.
No new main claim.
```

P2-D1: Benchmark capability gate.

```text
Prepare candidate datasets.
Run Single CoLA Solver and Role TextMAS baseline.
Admit only tasks with nontrivial, parseable CoLA outputs.
```

P2-D2: Role-conditioned TextMAS.

```text
Implement Planner/Critic/Refiner/Solver prompts.
Keep question q visible to downstream roles.
Score only final Solver output.
```

P2-D3: Role-conditioned latent no-fuser.

```text
Replace text message content with latent working memory.
Compare context_plus_thought vs thought_only.
Keep budgets matched with TextMAS.
```

P2-D4: Control and leakage audit.

```text
wrong_sample, same_task_wrong_sample, wrong_block, shuffle, noise, rotation.
No scorer-visible A replay.
No gold/scorer/selected_prediction online.
```

P2-D5: Lightweight fuser/gate only if triggered.

```text
Train residual gated adapter.
Use TextMAS teacher and corruption contrastive objectives.
Evaluate with decoder-free online path.
```

P2-D6: Naturally decomposable MAS.

```text
Evidence-split multi-hop QA or role-separated code workflows.
Use this tier for the strongest paper claim.
```

## 12. Success Criteria

Minimal scientific claim:

```text
Matched latent carries usable signal beyond empty/corrupted input.
```

Strong same-substrate communication claim:

```text
Role-conditioned matched latent improves over Single Solver or no-message.
Matched latent robustly beats corrupted controls.
Result holds across admitted tasks and seeds.
```

Text-vs-latent claim:

```text
Latent is Pareto-competitive or better than TextMAS under matched role prompts,
matched budgets, same q, same scorer, and receiver-only final output.
```

True MAS claim:

```text
The task requires distributed roles or information partitioning, and latent
communication improves quality/cost relative to text or single-agent baselines.
```

## 13. Non-Negotiables

```text
Smoke tests verify engineering only; they do not prove architecture success/failure.
Do not draw conclusions from benchmarks where CoLA has floor-level ability.
Do not use official8 solver-to-solver message_only as the main MAS result.
Do not let scorer see A text, A replay tokens, gold answers, or selected_prediction.
Do not train without CUDA and SwanLab cloud.
Do not optimize locally from one small curve; after every phase, run a literature-aware global review.
```

## 14. Current Execution Lock

As of 2026-06-01:

```text
admitted_tasks = []
P2-D3.1 answer_state_v1 / answer_state_structured_v1 calibration repair:
  no task admitted
role_plan_ignore_v1 calibration repair:
  no task admitted
```

Therefore:

```text
Do next:
  Choose substrate adaptation or benchmark redesign before any further P2 main
  experiment.

Do only if a later calibration repair obtains dual-pass:
  P2-D4 held-out capability gate.

Do only after held-out admission:
  P2-E channel-correct TextMAS vs LatentMAS main experiments.

Do only after no-fuser evidence:
  P2-F fuser / receiver adapter training.
```

This lock is intentionally conservative: it prevents base-model capability
floor, prompt collapse, or parser failures from being misreported as latent
communication failures.
