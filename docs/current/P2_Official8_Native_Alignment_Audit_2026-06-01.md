# P2 Official8 Native Alignment Audit

更新日期：2026-06-01

> 状态：Branch B official8-compatible calibration 后的 prompt/eval 对齐审计。
> 本文只使用 calibration split，不使用 held-out，不运行 Role TextMAS 主表，
> 不训练 fuser/adapter，不创建 SwanLab run。

## 1. 为什么做这个审计

Branch B 第一轮 normalized gate 得到：

```text
official8-compatible calibration admitted_tasks = []
```

但当时存在一个不确定点：

```text
当前 P2 gate 使用 normalized prompt：
  task_name = p2_generic
  choices = A/B/C/D label dict
  target = option letter

official CoLA benchmark 使用 native prompt：
  task_name = obqa/mmlu/race/...
  apply_prompt_template(task_name, ...)
  choices = option text list
  target = option text
```

因此必须确认：低分是否来自 P2 normalized gate 偏离 official CoLA 原始接口。

## 2. 官方接口事实

本地源码：

```text
/data1/luyifei/Cola-DLM/code/cola_dlm/inference.py
/data1/luyifei/Cola-DLM/code/scripts/acc_calc.py
/data1/luyifei/Cola-DLM/code/scripts/run_benchmark.sh
```

官方 generation：

```text
generate_task_repaint_inference(...)
  -> apply_prompt_template(task_name, context, question, answer, choices)
```

官方 MCQ scoring：

```text
mmlu/obqa/race/siqa:
  first preprocess generate by splitting at first meaningful newline
  extract choice letter OR match generated text to choice text
  fallback to exact normalized similarity against ground_truth text

hellaswag/story_cloze:
  first preprocess generate by splitting at first meaningful newline
  exact normalized similarity against ground_truth text
```

官方 run defaults：

```text
COLA_INFER_PER_SAMPLE_NOISE_SEED = 66
BATCH_SIZE = 20
MAX_SAMPLES = 1000
MAX_NEW_TOKENS = 32
temperature = 0.0
guidance_scale = 7.0
timestep_num = 16
```

Official local reference accuracy：

```text
/data1/luyifei/Cola-DLM/code/eval_output/accuracy_summary.csv

lambada = 50.80
mmlu = 19.30
obqa = 23.00
hellaswag = 10.70
race = 19.60
siqa = 28.90
squad = 30.90
story_cloze = 30.77
tasks_average = 26.75
```

This means official CoLA's released local output is already weak on several
choice tasks under raw accuracy; it should not be assumed to clear
`random_floor + margin`.

## 3. Implemented Native Audit Script

New script：

```text
/data1/luyifei/drla/drla/scripts/run_cola_p2_official8_native_single_gate.py
```

It does：

```text
1. read P2 calibration rows
2. recover raw official rows from /data1/luyifei/Cola-DLM/code/generate_task_data
3. group by source_task
4. call generate_task_repaint_inference with native task_name
5. use official acc_calc-style first-segment preprocessing and scoring
6. write generations.jsonl, metrics.jsonl, task_summary.csv, summary.json
```

Smoke bug fixed：

```text
smoke runs can have per-task gate_pass for code-path diagnostics,
but summary.admitted_tasks is always [] when max_samples_per_task > 0.
```

Companion rescore script：

```text
/data1/luyifei/drla/drla/scripts/rescore_cola_p2_official8_native_single_gate.py
```

Reason：

```text
Initial native scorer missed official acc_calc's first-meaningful-newline
preprocessing. The rescore script recomputes scores locally without rerunning
GPU generation.
```

## 4. Artifacts

Smoke：

```text
/data1/luyifei/drla/outputs/p2_benchmark_redesign/
eval_calibration_official8_native_single_smoke_20260601

meaning:
  code path only; not scientific evidence.
  admitted_tasks = []
```

Full native generation：

```text
/data1/luyifei/drla/outputs/p2_benchmark_redesign/
eval_calibration_official8_native_single_full32_seed66_20260601

rows:
  1600 calibration rows

settings:
  native task templates
  max_new_tokens = 32
  per_sample_noise_seed = 66
  SwanLab disabled
```

Rescored official-style result：

```text
/data1/luyifei/drla/outputs/p2_benchmark_redesign/
eval_calibration_official8_native_single_full32_seed66_rescored_20260601

admitted_tasks:
  []
```

## 5. Native Single Calibration Results

Rescored with official first-segment preprocessing：

| Task | Accuracy | Parseable | Random floor | Gate |
|---|---:|---:|---:|---|
| `official8_hellaswag` | 1.33% | 100.00% | 25.00% | fail |
| `official8_mmlu` | 18.33% | 75.00% | 25.00% | fail |
| `official8_obqa` | 25.00% | 83.00% | 25.00% | fail |
| `official8_race` | 23.00% | 83.33% | 25.00% | fail |
| `official8_siqa` | 30.67% | 93.00% | 33.33% | fail |
| `official8_story_cloze` | 29.33% | 100.00% | 50.00% | fail |

No task satisfies：

```text
nonempty_rate >= 0.95
parseable_rate >= 0.90
accuracy >= random_floor + 0.02
```

## 6. Interpretation

The audit resolves the previous ambiguity：

```text
Normalized P2 prompt/parser did introduce some distortion.
However, native official prompt/scoring does not produce an admitted task either.
```

The official local reference also supports this：

```text
mmlu = 19.30
obqa = 23.00
race = 19.60
siqa = 28.90
story_cloze = 30.77
```

These values are below or near random floors for the MCQ / binary tasks.
Therefore, this is not a latent communication failure. It is a base substrate
capability / benchmark suitability issue for Branch B Family 1.

## 7. Consequence for P2

Current state：

```text
Branch B Family 1 official8-compatible role candidates:
  calibration admitted_tasks = []

held-out:
  untouched

P2 text-vs-latent main table:
  not allowed

fuser/adapter training:
  not allowed
```

Family 1 stop condition is effectively met：

```text
No official8-compatible role task has a robust Single + Role calibration pass.
Even native official Single Solver does not admit a task under the current
gate criteria.
```

## 8. Recommended Next Decision

Do not keep prompt-tuning Family 1.

The next rigorous branch decision is：

```text
Branch A:
  substrate adaptation if ARC/GPQA/MedQA/GSM8K/EvalPlus or stronger MAS
  benchmarks are non-negotiable.

Branch C:
  validate MAS benchmark/protocol with external capable text agents first,
  then return to CoLA latent through adapter/translator.

Branch B Family 2:
  only if we can construct genuinely decomposable official8-derived tasks
  where a single native CoLA solver first passes calibration robustly.
```

Given the current evidence, Branch A or C is more scientifically meaningful
than continuing prompt-only repair on frozen official CoLA Family 1.
