# P2 Branch B Calibration 报告

更新日期：2026-06-01

> 状态：Branch B 第一轮 calibration-only 结果。本文不使用 held-out，不运行
> text-vs-latent 主表，不训练 fuser/adapter。结论只用于判断 frozen official
> CoLA + 当前 normalized gate prompt 是否已经有任务可进入 locked held-out gate。

## 1. 执行范围

Branch B 当前目标：

```text
保留 frozen official CoLA
-> 用 official8-compatible role candidates 做 capability-matched triage
-> 先过 Single CoLA Solver + Role TextMAS calibration gate
-> 再考虑 held-out locked gate
```

本轮严格没有做：

```text
held-out generation
P2 text-vs-latent main table
latent fuser / adapter training
SwanLab cloud logging
```

## 2. 数据与 Split

Candidate data：

```text
script:
  /data1/luyifei/drla/drla/scripts/prepare_cola_p2_official8_role_candidates.py

full data:
  /data1/luyifei/drla/outputs/p2_benchmark_redesign/
  official8_role_candidates_20260601

combined jsonl:
  /data1/luyifei/drla/outputs/p2_benchmark_redesign/
  official8_role_candidates_20260601/p2_official8_role_candidates.jsonl

total rows:
  33296
```

Deterministic split：

```text
script:
  /data1/luyifei/drla/drla/scripts/build_cola_p2_locked_splits.py

output:
  /data1/luyifei/drla/outputs/p2_benchmark_redesign/
  official8_role_candidates_splits_seed20260603_20260601

split_seed:
  20260603

calibration / heldout / overlap:
  1600 / 31696 / 0
```

Per-task calibration：

| Task | Calibration | Held-out |
|---|---:|---:|
| `official8_hellaswag` | 300 | 9742 |
| `official8_mmlu` | 300 | 13742 |
| `official8_obqa` | 100 | 400 |
| `official8_race` | 300 | 4587 |
| `official8_siqa` | 300 | 1654 |
| `official8_story_cloze` | 300 | 1571 |

Held-out rows remain locked and were not inspected at sample level.

## 3. Script Update

Capability gate script now supports separated prompt variants：

```text
script:
  /data1/luyifei/drla/drla/scripts/run_cola_p2_capability_gate.py

new args:
  --single-prompt-variant
  --role-prompt-variant

backward compatibility:
  both default to --prompt-variant when not provided.
```

Reason：

```text
Single Solver 和 Role TextMAS 的 prompt contract 不应被强行绑定。
合理 MAS baseline 中，single solver prompt 与 role-state prompt 可以不同。
否则 single capability gate 会被 role-state prompt 错误压低。
```

Aggregator also records these prompt fields：

```text
/data1/luyifei/drla/drla/scripts/aggregate_cola_p2_capability_gate.py
```

## 4. Formal Calibration Artifacts

Aggregate：

```text
/data1/luyifei/drla/outputs/p2_benchmark_redesign/
aggregate_calibration_official8_branchb_prompt_gate_20260601

summary:
  summary.json
  mode_gate_summary.csv
  task_gate_summary.csv

admitted_tasks:
  []
```

Formal full64 runs included：

```text
eval_calibration_official8_single_structured_full64_20260601
eval_calibration_official8_single_fewshot_full64_20260601
eval_calibration_official8_single_generic_full64_20260601
eval_calibration_official8_obqa_structured_full64_20260601
eval_calibration_official8_mmlu_splitprompt_full64_20260601
```

Smoke / preliminary runs exist but must not be cited as science results：

```text
eval_calibration_official8_obqa_smoke_20260601
eval_calibration_official8_obqa_structured_smoke_20260601
eval_calibration_official8_obqa_splitprompt_smoke_20260601
eval_calibration_official8_obqa_structured_full_20260601
```

The `*_full_20260601` OBQA run used 32-token budget and is therefore only a
budget/protocol check, not the formal gate result.

## 5. Single Solver Calibration

### generic_v1

| Task | Accuracy | Parseable | Random floor | Gate |
|---|---:|---:|---:|---|
| `official8_hellaswag` | 19.00% | 69.67% | 25.00% | fail |
| `official8_mmlu` | 27.67% | 91.33% | 25.00% | pass in all-task run |
| `official8_obqa` | 27.00% | 82.00% | 25.00% | fail |
| `official8_race` | 24.00% | 81.33% | 25.00% | fail |
| `official8_siqa` | 31.33% | 87.00% | 33.33% | fail |
| `official8_story_cloze` | 42.67% | 81.33% | 50.00% | fail |

### answer_state_structured_v1

| Task | Accuracy | Parseable | Random floor | Gate |
|---|---:|---:|---:|---|
| `official8_hellaswag` | 21.33% | 83.00% | 25.00% | fail |
| `official8_mmlu` | 22.33% | 85.00% | 25.00% | fail |
| `official8_obqa` | 13.00% | 66.00% | 25.00% | fail |
| `official8_race` | 19.67% | 85.33% | 25.00% | fail |
| `official8_siqa` | 27.33% | 78.33% | 33.33% | fail |
| `official8_story_cloze` | 45.00% | 84.67% | 50.00% | fail |

### cola_fewshot_v1

| Task | Accuracy | Parseable | Random floor | Gate |
|---|---:|---:|---:|---|
| `official8_hellaswag` | 14.33% | 60.33% | 25.00% | fail |
| `official8_mmlu` | 18.00% | 64.33% | 25.00% | fail |
| `official8_obqa` | 21.00% | 76.00% | 25.00% | fail |
| `official8_race` | 18.00% | 72.67% | 25.00% | fail |
| `official8_siqa` | 21.33% | 56.00% | 33.33% | fail |
| `official8_story_cloze` | 35.00% | 66.00% | 50.00% | fail |

Interpretation：

```text
generic_v1 is the only single prompt with a calibration pass, and only for
official8_mmlu in the all-task run.

Most failures are not just accuracy failures; parseable_rate often falls below
0.90, which means current normalized prompt/parser contract is not yet stable.
```

## 6. MMLU Task-Isolated Role Gate

Because `official8_mmlu` was the only all-task single-pass candidate, it was
rerun as a task-isolated split-prompt gate：

```text
output:
  /data1/luyifei/drla/outputs/p2_benchmark_redesign/
  eval_calibration_official8_mmlu_splitprompt_full64_20260601

single_prompt_variant:
  generic_v1

role_prompt_variant:
  answer_state_structured_v1
```

Result：

| Mode | Accuracy | Parseable | Random floor | Gate |
|---|---:|---:|---:|---|
| single | 21.00% | 88.00% | 25.00% | fail |
| role_textmas | 20.00% | 86.00% | 25.00% | fail |

Conclusion：

```text
official8_mmlu is not admitted.
admitted_tasks remains [].
```

## 7. Important Reproducibility Finding

`official8_mmlu` passed single gate in the all-task generic run but failed in
the task-isolated split-prompt run.

Likely cause：

```text
Official CoLA generation uses diffusion/noise sampling. Even with temperature
0, output can depend on RNG consumption and run batching/order.
```

Operational rule：

```text
Do not admit marginal tasks from all-task aggregate alone.
For candidate admission, require task-isolated calibration gate or an explicitly
locked sharding/seed protocol.
```

This is a protocol stability issue, not a latent communication result.

## 8. Current Conclusion

Historical Branch B Family 1 first pass：

```text
calibration admitted_tasks = []
held-out remains untouched
no P2 text-vs-latent main table is allowed
no fuser/adapter training is allowed
```

The bottleneck is now narrower than before：

```text
The official8-compatible candidate family is not yet passing the normalized
Single + Role capability gate.

The main failure appears to be prompt/schema/parser alignment and CoLA base
generation stability under the new normalized gate, not latent communication.
```

## 9. Next Step

Do not continue directly to role/latent communication. The next rigorous step
is an official CoLA prompt/eval alignment audit：

```text
1. Compare normalized P2 prompts against the original official CoLA task prompt
   format used by `/data1/luyifei/Cola-DLM/code/generate_task_data` and
   official evaluation scripts.

2. Audit parser failures by task and prompt variant, especially placeholder
   echoes, option-text outputs, and long free-form answer lines.

3. Build a native-official8 prompt variant only if it faithfully matches the
   official CoLA benchmark interface.

4. Re-run single task-isolated calibration after prompt alignment.

5. Only if single gate passes robustly, re-run Role TextMAS calibration.
```

Until that audit is done, Branch B has no admitted task and P2 main evaluation
must remain locked.
