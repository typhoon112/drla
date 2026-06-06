# P2-D4 分支决策审计

更新日期：2026-06-01

> 状态：历史分支决策审计。本文的证据仍有效，但“Branch B first”的默认建议
> 已被执行为 Branch B Family 1，并在 official8-compatible calibration 与 native
> alignment audit 后停止。当前下一步以
> `/data1/luyifei/drla/docs/current/P2_Post_Family1_Complete_Execution_Plan_2026-06-01.md`
> 为准。

## 1. 当前硬证据

Capability gate:

```text
P2-D1 full gate:
  admitted_tasks = []

P2-D2 split:
  calibration = 842
  heldout = 3359
  overlap = 0

P2-D3 prompt repair:
  generic_v1 GPQA-Diamond single calibration passes only.
  Role TextMAS fails.

P2-D3.1 / D3.2 protocol repair:
  answer_state_v1: negative
  answer_state_structured_v1: partial parse improvement, no admitted task
  role_plan_ignore_v1: negative
  all repair aggregate admitted_tasks = []
```

Key artifacts:

```text
all repair aggregate:
  /data1/luyifei/drla/outputs/p2_capability_gate/
  aggregate_calibration_protocol_repair_all_20260601

failure taxonomy:
  /data1/luyifei/drla/outputs/p2_capability_gate/
  audit_protocol_repair_failures_all_20260601
```

Representative calibration facts:

| Task | Best observed relevant row | Result |
|---|---|---|
| GPQA-Diamond | generic_v1 single | 32.50% acc / 92.50% parse, single-only pass |
| GPQA-Diamond | generic_v1 role | 25.00% acc / 50.00% parse, fail |
| GPQA-Diamond | role_plan_ignore_v1 role | 2.50% acc / 35.00% parse, fail |
| ARC-Challenge | answer_state_structured_v1 role | 26.67% acc / 85.00% parse, fail |
| MedQA | answer_state_structured_v1 role | 24.31% acc / 92.94% parse, fail |

Current conclusion:

```text
Current official CoLA substrate plus prompt-only Role TextMAS repair cannot
admit the candidate P2 benchmark set.

This is not a latent communication failure. It is a base-capability / protocol
readiness failure before the communication medium is even compared.
```

## 2. Literature Cross-check

Primary sources checked:

```text
LatentMAS:
  https://arxiv.org/abs/2511.20639
  local source:
    /data1/luyifei/latent_reasoning_papers/agent_comm_downloads/
    2511.20639_LatentMAS

Coconut:
  https://arxiv.org/abs/2412.06769
  local text:
    /data1/luyifei/latent_reasoning_papers/2412.06769_Coconut.txt

CODI:
  https://aclanthology.org/2025.emnlp-main.36.pdf
  local text:
    /data1/luyifei/latent_reasoning_papers/2502.21074_CODI.txt

LCGuard:
  https://arxiv.org/abs/2605.22786
```

Relevant takeaways:

```text
LatentMAS:
  The method assumes capable base agents and transfers latent working memory
  through layer-wise KV caches. It does not use compressed answer labels as the
  communication object. Its sequential MAS prompts keep plan / feedback /
  refined-plan semantics and only decode final answers.

Coconut:
  Continuous thoughts are not obtained by prompt wording. They are introduced
  through a staged training procedure that internalizes reasoning into latent
  steps.

CODI:
  Continuous reasoning can be learned through self-distillation and hidden-state
  alignment. This supports adaptation if the substrate must handle new tasks.

LCGuard:
  KV/latent communication can preserve richer task-relevant information, but
  opaque latent channels also carry sensitive/contextual information, so learned
  transformations and audits may be needed before broad deployment.
```

Implication for this project:

```text
Prompt-only Role TextMAS repair is not the natural path once the base CoLA
solver is near random on the target tasks.

If we keep current candidate benchmarks, we likely need substrate adaptation
or receiver/working-memory adaptation.

If we avoid adaptation, we need benchmark redesign: choose tasks where frozen
official CoLA can pass Single + Role gates, or tasks naturally aligned with
official CoLA's demonstrated format and capability.
```

## 3. Branch Options

### Branch A: Substrate Adaptation

Use when:

```text
We insist on ARC/GPQA/MedQA/GSM8K/EvalPlus-like benchmarks as the P2 target.
We need official CoLA to become capable enough to pass Single + Role gates.
```

Likely methods:

```text
task-format adapter or LoRA for CoLA solver capability
role-protocol adapter for Planner/Critic/Refiner/Solver
latent working-memory / receiver adapter after no-fuser controls
teacher distillation from capable TextMAS or CoLA official trace when available
```

Training obligations:

```text
CUDA only
SwanLab cloud
metrics.jsonl
valid_interval <= 10 step
best_checkpoint.pt and last_checkpoint.pt
locked calibration/held-out or train/valid/test split
no reporting on last checkpoint only
```

Risks:

```text
This changes the substrate. Comparisons must clearly distinguish:
  frozen official CoLA
  adapted CoLA
  adapted TextMAS
  adapted LatentMAS

It may blur the current same-substrate training-free claim unless reported as
an adaptation branch.
```

### Branch B: Benchmark Redesign

Use when:

```text
We want to preserve frozen official CoLA as the substrate.
We want P2 communication claims before undertaking task-capability training.
```

Requirements:

```text
new manifest
Single CoLA Solver gate
Role TextMAS gate
calibration/held-out split
no held-out inspection during repair
no smoke result as scientific conclusion
```

Benchmark families to consider:

```text
official8-derived role tasks:
  keep CoLA's known answer formats, but construct role-conditioned diagnostics.

evidence-split lightweight QA:
  only if frozen CoLA can answer final questions above floor.

multiple-choice tasks with shorter prompts and stable option labels:
  only after Single + Role calibration pass.

synthetic but not tiny toy tasks:
  acceptable only if they stress communication and have enough scale/diversity;
  not acceptable as "small works therefore large works" evidence.
```

Risks:

```text
May delay the true LatentMAS-style benchmark comparison.
May produce tasks that are diagnostically clean but less compelling than
ARC/GPQA/MedQA.
```

### Branch C: External Capable Text-MAS First

Use when:

```text
We need to validate the MAS benchmark/protocol independent of CoLA.
```

Boundary:

```text
This validates task/protocol design, not CoLA latent communication.
Any later CoLA latent claim requires a translator/shared codec/adapter and a
separate claim boundary.
```

## 4. Recommendation

Historical recommendation executed:

```text
benchmark redesign / capability-matched task selection.
```

Reason:

```text
The active P2 claim is same-substrate CoLA A -> CoLA B latent communication.
Before training or adapting the substrate, we should first determine whether
there exists a frozen-CoLA-compatible role benchmark where Single + Role gates
pass. This preserves the cleanest claim boundary.

Branch A becomes necessary if no such benchmark can be found or if the project
chooses ARC/GPQA/MedQA/GSM8K as non-negotiable target tasks.
```

Post-Family1 update:

```text
Branch B Family 1 was executed and stopped with admitted_tasks=[].
Do not treat Branch B first as the current default anymore.
Current recommended scientific order is Branch C -> Branch A, while Branch B
Family 2 remains diagnostic-only unless explicitly selected.
```

## 5. Immediate Safe Work Before Branch Execution

These are allowed without committing to A or B:

```text
1. Build a benchmark-candidate inventory document.
2. Define capability-gate manifest requirements for future candidate tasks.
3. Add script support for new manifests only if it does not inspect held-out.
4. Audit current prompt/parser failure modes.
5. Keep all pure eval local-only with swanlab_mode=disabled.
```

Completed safe prep:

```text
/data1/luyifei/drla/docs/current/
P2_Benchmark_Redesign_Candidate_Inventory_2026-06-01.md
```

These are not allowed yet:

```text
run held-out gate
run text-vs-latent main table
train LoRA/fuser/adapter
claim latent communication failure
claim frozen CoLA satisfies new benchmarks
```

## 6. Current Execution Decision

Current default after this audit:

```text
Branch B:
  preserve frozen CoLA and redesign benchmark/capability gate first.
```

Execution lock:

```text
/data1/luyifei/drla/docs/current/
P2_Branch_B_Execution_Plan_2026-06-01.md
```

Fallback branches remain valid but are not current defaults:

```text
Branch A:
  adapt CoLA substrate for target benchmarks, with full training discipline.
  Use this if ARC/GPQA/MedQA/GSM8K/EvalPlus are non-negotiable.

Branch C:
  validate MAS protocol with external capable text agents first, then return
  to CoLA latent via adapter/translator.
  Use this if benchmark/protocol naturalness must be validated independent of
  CoLA base capability.
```
