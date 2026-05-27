# Script Status

Last updated: 2026-05-27

## Main Path

The main path is official Cola block-wise readiness / halt analysis. The active entrypoints cover benchmark preparation, trace collection, oracle frontier construction, multi-signal readiness training, and adaptive halt evaluation.

Main evaluation must use the official Cola 8-task benchmark suite:

```text
lambada, mmlu, obqa, hellaswag, race, siqa, squad, story_cloze
```

## Active Scripts

- `prepare_cola_official_benchmarks.py`: prepares the official Cola 8-task benchmark JSONL files under `/data1/luyifei/Cola-DLM/code/generate_task_data`.
- `collect_cola_block_traces.py`: runs official Cola block-wise generation and records per-block raw latent shards, decoder probe stats, EOS/im_end probe stats, generated text so far, answer-stability signals, and local `metrics.jsonl`. It has no training loop and only accepts `swanlab_mode=disabled`. Default output is score-ready under `/data1/luyifei/drla/outputs/cola_block_traces/tasks_block_trace`.
- `merge_cola_block_trace_segments.py`: merges OOM/resume trace segments into one score-ready root, validating generation id order and trace sample ids. Use it for full prepared splits when one task is collected in multiple segments.
- `eval_cola_benchmarks.py`: wraps official Cola `scripts/acc_calc.py`, parses the 8-task CSV, and writes local summaries. It has no training loop and only accepts `swanlab_mode=disabled`.
- `build_cola_readiness_frontier.py`: uses official scorer rules to build offline oracle readiness/frontier labels from per-block traces.
- `train_cola_readiness_model.py`: trains the multi-signal readiness/halt model over raw latent blocks plus process/probe/stability features. It writes SwanLab metrics, local `metrics.jsonl`, `best_checkpoint.pt`, and `last_checkpoint.pt`.
- `train_cola_latent_halt_student.py`: trains Phase P1 `LatentHaltStudent-v1`. Decoder/scorer/stability fields are teacher targets only; online inputs are raw latent prefix blocks plus selected latent/process features, with CUDA/GPU, SwanLab cloud, local `metrics.jsonl`, and best/last checkpoints. It supports explicit teacher-loss weights, `--selection-metric`, `--process-feature-mode full|no_block_budget`, `--process-interaction-mode process_token|film|trajectory_token`, `--readout-context-mode none|last_process_query`, `--use-completion-risk`, `--use-empty-answer-risk`, `--use-answer-format-risk`, `--use-answer-identity-stability`, and P0 teacher targets via `--readiness-target-mode oracle_frontier|p0_teacher_halt|p0_teacher_action`. `trajectory_token` keeps the normal process token and appends a per-block trajectory token built from pooled block state, previous-block delta, and process state before causal inter-block attention. Current selection metrics include `readiness_prediction_change_completion_contentful_mean_auroc`, `readiness_prediction_change_completion_empty_mean_auroc`, `readiness_prediction_change_completion_format_mean_auroc`, and `readiness_prediction_change_completion_identity_mean_auroc` for boundary/identity diagnostics.
- `eval_cola_latent_halt_student.py`: evaluates a `LatentHaltStudent-v1` checkpoint as a student-only halt policy. It calibrates thresholds on valid, reports test accuracy-cost, saves threshold sweeps, student scores, and halt decisions, and audits answer-text mismatch versus final/prediction-stability outputs. It has no training loop and only accepts `swanlab_mode=disabled`. `--require-zero-calibration-mismatch` adds a stricter valid-set text-stability calibration guard on top of loss calibration. `--max-calibration-mismatches` / `--max-calibration-mismatch-rate` provide a softer risk cap that can keep zero-loss policies from selecting overly aggressive low-block thresholds. `--calibration-boundary-risk-penalty` changes threshold selection from pure min-block to `avg_blocks + penalty * boundary_risk_slack`, which is useful when finite target calibration over-selects permissive risk thresholds. `--max-calibration-samples-per-task` plus `--calibration-subsample-seed` runs deterministic target-calibration sample-cap diagnostics. `--empty-answer-risk-thresholds`, `--answer-format-risk-thresholds`, and `--answer-identity-stability-thresholds` are only active for checkpoints trained with the matching heads. `--calibration-scope per_task` requires the selected threshold to satisfy the same constraints on every calibration task instead of only on the pooled valid split.
- `aggregate_cola_latent_halt_student_loto.py`: aggregates P1 leave-one-task-out eval summaries with loss/mismatch rates, Wilson upper bounds, per-task risk buckets, and cross-seed recurrence. It has no training loop and must stay local-only.
- `aggregate_cola_latent_halt_student_subseed_loto.py`: aggregates nested calibration-subseed LOTO eval summaries such as `subseed*/leave_*_out_eval_*_test/summary.json`. It writes repeated-sample micro metrics, seed/task/subseed CSVs, and is local-only.
- `analyze_latent_halt_risk_control.py`: local-only formal risk-control audit for existing P1 threshold sweeps. It selects thresholds using validation-set Wilson upper bounds for loss/mismatch risk, then reports held-out official8/cross-seed trade-offs. Use `--allow-legacy-swanlab-eval` only for historical eval summaries created before the local-only eval rule.
- `eval_cola_adaptive_halt.py`: sweeps adaptive halt thresholds and compares fixed-B, oracle halt, prediction-stability, learned threshold-only halt, aggressive readiness-or-stability, and stability-gated guarded policies. It has no training loop and only accepts `swanlab_mode=disabled`.
- `analyze_cola_halt_decisions.py`: reconstructs per-sample halt decisions from a saved eval summary and checkpoint, then writes `halt_decisions.jsonl`, `policy_comparison.csv`, `readiness_bins.csv`, and `summary.json`.
- `aggregate_cola_halt_decision_analysis.py`: aggregates per-task halt decision diagnostics across leave-one-task-out runs.
- `train_cola_continuation_risk_model.py`: trains a non-gold continuation-risk model for gating readiness halt before prediction stability. `--target-mode strict_prefix` detects prefix/incomplete answers; `--target-mode prediction_change` detects whether the current task-scored prediction will differ from the rollout prediction-stability reference.
- `eval_cola_risk_gated_halt.py`: evaluates readiness halt gated by continuation-risk probability. It has no training loop and only accepts `swanlab_mode=disabled`. `risk_threshold_selection_mode=min_blocks` is aggressive; `first_saving` is safer. `--readiness-threshold-values` lets validation jointly sweep readiness and risk thresholds. `--require-zero-calibration-loss` prevents valid-set loss/gain cancellation by requiring zero losses versus prediction-stability when such rows exist. `--require-contentful-prediction` adds a non-gold answer-shape guard against punctuation-only early halt. `--require-fragment-complete-prediction` guards non-stable decoded answer fragments such as pure-number prefixes, unfinished abbreviations, trailing initials, and short hyphen fragments. `--require-stable-single-choice` requires single `A`-`E` predictions to reach prediction stability before halt, addressing multiple-choice letter flips; `--stable-single-choice-max-block 1` or `2` narrows that guard to early blocks. Use `--stable-single-choice-guard-scopes off,1,2,3,all` to let validation select the guard scope jointly with risk threshold.
- `eval_cola_risk_gated_halt.py` also supports strict uncertainty calibration through `--entropy-max-values` and `--top-prob-min-values`. These sweep non-gold decoder confidence guards in addition to risk threshold; defaults preserve the old behavior.
- `analyze_cola_risk_gated_halt_decisions.py`: reconstructs sample-level risk-gated halt decisions and writes per-sample loss/gain diagnostics against fixed-final and prediction-stability.
- `aggregate_cola_risk_gated_halt.py`: aggregates same-task and leave-one-task-out risk-gated halt summaries plus common threshold sweeps.

## Current Official-Protocol Results

The current 2026-05-24 summary artifact is:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_readiness_halt_20260524
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_b64_readiness_halt_20260524
```

The first includes full-model multi-seed, signal ablations, EOS-only/fixed-B baselines, valid-calibrated thresholds, and leave-one-task-out transfer results for the 32-token / 2-block protocol. The second covers the 64-token / 4-block trace, score, frontier, full readiness model, calibrated adaptive halt eval, trace dynamics, the non-gold `prediction_stability` halt baseline, and stability-feature model comparison.

`train_cola_readiness_model.py` now derives non-gold `scored_prediction` / `official_processed_generation` stability features. Checkpoint feature fields are preserved for backward-compatible eval of older 22-feature checkpoints.

The b64 leave-one-task-out summary for `process_no_task_stability_features` is:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_b64_cross_task_stability_features_20260524
```

It shows positive but modest cross-task transfer and a remaining calibration problem, especially on SQuAD.

The strict calibration comparison is:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_b64_cross_task_calibration_comparison_20260524
```

Tightening tolerance from `0.01` to `0.0` mostly preserves accuracy but becomes too conservative (`3.65/4` blocks), so the next calibration work should be stability-aware rather than threshold-only.

The guarded calibration comparison is:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_b64_cross_task_guarded_calibration_20260524
```

`adaptive_or_prediction_stability` is an aggressive control, not a true guard: it can stop before the task-scored prediction is stable. The true `stability_guarded_adaptive` policy preserves the cross-task prediction-stability baseline (`27.03%`, `2.41/4` blocks) but currently selects threshold `0.0`, so it adds no extra saving. The aggressive control reaches `26.28%` at `1.75/4` blocks under calibrated held-out evaluation. A conservative aggregate OR threshold `0.75` preserves accuracy but saves only `0.005` extra block versus prediction-stability, confirming that useful pre-stability readiness is still poorly calibrated for held-out tasks.

Sample-level halt diagnostics are:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_b64_halt_decision_analysis_20260524
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_b64_halt_decision_analysis_strict_20260524
```

The 1% calibrated adaptive/OR policies lose `61` final-correct held-out samples, all before prediction stability; `57/61` are strict prefix losses. The strict 0% OR variant still loses `8` LAMBADA final-correct samples while averaging `2.286/4` blocks, and all `8/8` are prefix losses, so simple threshold tightening does not solve task-robust safe early halt.

Continuation-risk gated halt results are:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_b64_risk_gated_halt_20260524
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_b64_risk_gated_halt_first_saving_20260524
```

The `min_blocks` calibration is useful as an aggressive upper bound: cross-task accuracy `26.95%`, average blocks `1.864/4`, still `0.08` percentage points below prediction-stability. The `first_saving` calibration is the current safe default: cross-task aggregate accuracy matches fixed-final / prediction-stability at `27.03%` while using `2.018/4` average blocks. It still has one SQuAD task-level loss offset by one MMLU gain, so task-level safety is not solved.

The current strongest 1k-protocol result adds the non-gold contentful-prediction guard:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_b64_risk_gated_halt_first_saving_content_guard_20260524
/data1/luyifei/drla/outputs/cola_risk_gated_halt_analysis/first_saving_content_guard_squad_20260524
```

It reaches aggregate `27.04%` accuracy at `2.019/4` blocks and removes the SQuAD quote-only early-stop loss. Treat the slight aggregate gain over fixed-final as a diagnostic early-correct case, not as a primary accuracy-improvement claim.

The full prepared-split protocol is now unified on `batch_size=12`, because batch-invariance diagnostics showed MMLU output can change across batch sizes. Do not mix `bs20`, `bs12`, and `bs1` outputs in a main trace result. The diagnostic artifact is:

```text
/data1/luyifei/drla/outputs/cola_batch_invariance/batch_size_invariance_summary_20260524.json
```

Full prepared-split artifacts:

```text
trace: /data1/luyifei/drla/outputs/cola_block_traces/tasks_official8_full_b64_t16_seed66_bs12_merged_20260524
score: /data1/luyifei/drla/outputs/cola_official_benchmarks/full_b64_bs12_trace_score_20260524/summary.json
frontier: /data1/luyifei/drla/outputs/cola_readiness_frontiers/official8_full_b64_bs12_t16_seed66_20260524
readiness model: /data1/luyifei/drla/outputs/cola_readiness_model/official8_full_b64_bs12_t16_seed66_20260524
risk model: /data1/luyifei/drla/outputs/cola_continuation_risk_model/official8_full_b64_bs12_process_no_task_seed20260524
adaptive halt: /data1/luyifei/drla/outputs/cola_adaptive_halt/official8_full_b64_bs12_t16_seed66_20260524
risk-gated halt: /data1/luyifei/drla/outputs/cola_risk_gated_halt/official8_full_b64_bs12_first_saving_content_guard_seed20260524
cross-seed summary: /data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_seed_20260524
```

Merge summary: `49,019` generation rows and `196,076` trace rows, `ok=true`. Official scorer task-average accuracy is `25.26%` (SwanLab `gpnfpuu93117wy3nub94o`). On the full-split test partition, first-saving risk-gated halt with content guard reaches `20.392%` micro accuracy at `1.880/4` blocks versus fixed-final `20.351%` at `4/4` (SwanLab `fgrqk019ypjtc6hd52suf`). Treat this as a block-budget result, not as a benchmark accuracy-improvement claim.

Seed67 repeats the full protocol with `seed=20260525`, `per_sample_noise_seed=67`, and uniform `batch_size=12`. Its merged trace is:

```text
/data1/luyifei/drla/outputs/cola_block_traces/tasks_official8_full_b64_t16_seed67_bs12_merged_20260524
```

Seed67 official scorer task-average is `24.99%`; first-saving risk-gated halt with content guard reaches `21.039%` micro accuracy at `1.873/4` blocks versus fixed-final `21.018%` at `4/4`.

Seed68 repeats the full protocol with `seed=20260526`, `per_sample_noise_seed=68`, and uniform `batch_size=12`. Its merged trace is:

```text
/data1/luyifei/drla/outputs/cola_block_traces/tasks_official8_full_b64_t16_seed68_bs12_merged_20260524
```

Seed68 official scorer task-average is `24.96%`; first-saving risk-gated halt with content guard reaches `20.798%` micro accuracy at `1.842/4` blocks versus fixed-final `20.860%` at `4/4`. The three-seed summary averages `20.743%` risk-gated micro accuracy at `1.865/4` blocks versus `20.743%` fixed-final at `4/4`, about `53.38%` block saving. Treat this as a block-budget/readiness result, not as a benchmark accuracy-improvement claim.

Strict uncertainty calibration artifact:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_uncertainty_calibration_20260524
```

With `accuracy_drop_tolerance=0.0`, strict first-saving calibration averages `20.750%` micro accuracy at `2.291/4` blocks; strict min-blocks gives `20.750%` at `2.281/4`. Both match prediction-stability accuracy on all three seeds. The selected `entropy_max` and `top_prob_min` are `none` for every seed, so mean token entropy/top-prob does not currently improve the guard.

Choice-stability guard artifact:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_choice_guard_20260524
```

The sample-level diagnostics show the remaining `0.01`-tolerance content-guard losses are single-letter multiple-choice flips, not prefix losses. With `--require-stable-single-choice`, the 3-seed policy matches prediction-stability accuracy (`20.750%`) at `2.210/4` blocks and has zero observed losses versus prediction-stability. Narrowing it to `--stable-single-choice-max-block 1` keeps zero observed losses, improves to `20.757%` at `2.137/4` blocks, and is the current better safety-cost point among these guards. The current audit rerun also saves `calibration_risk_threshold_sweep.csv` and selected valid metrics for every seed.

Validation-selected choice-scope artifact:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_choice_scope_sweep_20260524
```

With `accuracy_drop_tolerance=0.01`, validation selects `single_choice_guard_scope=off` and preserves the cheaper content-guard policy, but sample diagnostics still show observed single-letter losses. With strict `accuracy_drop_tolerance=0.0`, validation selects `single_choice_guard_scope=1` and `risk_threshold=0.01` on all three seeds, matching prediction-stability with zero observed losses at `2.137/4` blocks. This makes block1-only a validation-selected safety setting, not a test-tuned rule.

Prediction-change risk artifact:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_prediction_change_risk_20260524
```

The `prediction_change` target is still non-gold: it asks whether the current scored prediction differs from the future prediction-stability reference in the same rollout. Three formal SwanLab training runs reach test AUROC about `0.997`. Under strict `accuracy_drop_tolerance=0.0`, first-saving calibration selects `single_choice_guard_scope=off` on all seeds and reaches `20.750%` at `1.958/4` blocks with zero observed losses versus prediction-stability. Strict `min_blocks` is cheaper (`1.904/4`) but seed67 has loss/gain cancellation, so it remains diagnostic rather than the safety-first policy.

Leave-one-task-out prediction-change risk transfer artifacts:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_prediction_change_risk_20260524
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_prediction_change_risk_seed67_20260524
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_prediction_change_risk_seed68_20260524
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_prediction_change_risk_cross_seed_20260524
```

This trains readiness/risk on 7 tasks with `process_no_task`, calibrates only on the 7-task valid split, and evaluates the held-out task on `split=all`. Across seed66/67/68, strict first-saving risk-gated halt averages `21.596% +/- 0.029%` weighted micro accuracy at `2.499 +/- 0.005/4` blocks, matching prediction-stability accuracy while saving about `1.501` blocks versus fixed-final. Sample diagnostics show zero observed losses versus both fixed-final and prediction-stability across all three seeds. Treat this as replicated transfer evidence for the current prepared split, not as a claim that official Cola benchmark accuracy improved.

Joint readiness-threshold calibration artifact:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_joint_readiness_riskcap04_cross_seed_20260524
```

This sweeps readiness thresholds on valid, caps `risk_threshold_end=0.4`, requires zero valid loss versus prediction-stability, uses the contentful-prediction guard, and applies block2 single-choice stability guard. Across seed66/67/68 it keeps zero observed losses versus prediction-stability and reaches the same weighted micro accuracy at `2.118/4` average blocks, saving `0.394` block beyond prediction-stability.

Decoder-dependency status: this is the current best decoder-probed baseline, not a final latent-only agent communication policy. The readiness/risk labels and features still use decoder probe, decoded text dynamics, task-scored prediction, prediction-stability, and official scorer outputs; the eval guard also uses `scored_prediction`. Future default experiments should prioritize latent-primary halt variants that use these decoder-side signals only as teacher labels or offline diagnostics, with online inputs restricted toward latent trajectory, process features, and learned latent verifier proxies.

SQuAD held-out answer-shape diagnostic:

```text
/data1/luyifei/drla/outputs/cola_risk_gated_halt/cross_task_full_b64_bs12_prediction_change_shape_features_fragment_guard_v2_joint_readiness_min_blocks_choice2_zeroloss_seed20260526/leave_squad_out_eval_squad_all
/data1/luyifei/drla/outputs/cola_risk_gated_halt_analysis/cross_task_full_b64_bs12_prediction_change_shape_features_fragment_guard_v2_joint_readiness_min_blocks_choice2_zeroloss_20260526/leave_squad_out_eval_squad_all
```

The shape-feature risk model alone still loses SQuAD prefix-fragment cases without riskcap. Adding `--require-fragment-complete-prediction` v2 reaches zero observed losses versus fixed-final and prediction-stability at `1.908/4` blocks on SQuAD `split=all` (SwanLab run `639qsu4cpfseme3cpomf2`). Treat this as a diagnostic readiness signal, not as a final cross-task replacement for riskcap04 until it is validated across tasks/seeds.

Seed68 full 8-task fragment-completeness checks:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_fragmentguardv2_scope_sweep_seed68_20260524
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_fragmentguardv2_choice2_noriskcap_seed68_20260524
```

The scope-sweep/no-riskcap run saves blocks (`1.895/4`) but loses `21` MMLU samples, all single-choice flips. The fixed block2/no-riskcap run fixes MMLU but loses `34` SQuAD samples. This keeps riskcap04 as the current safety baseline and motivates full answer-shape feature training rather than more hand-written guards.

Full 8-task answer-shape risk check:

```text
/data1/luyifei/drla/outputs/cola_continuation_risk_model/cross_task_full_b64_bs12_prediction_change_shape_features_process_no_task_seed20260526
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_shape_features_fragmentguardv2_choice2_noriskcap_seed68_20260524
/data1/luyifei/drla/outputs/cola_risk_gated_halt_analysis/cross_task_full_b64_bs12_prediction_change_shape_features_fragmentguardv2_choice2_noriskcap_seed20260526
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_shape_features_fragmentguardv2_choice2_noriskcap_cross_seed_20260524
```

All `24` seed66/67/68 leave-one-task-out 38-feature risk checkpoints are trained with `best_checkpoint.pt`. The per-task valid-selected no-riskcap policy averages weighted micro accuracy `21.596%` at `2.153/4` blocks, but has `19` observed losses versus prediction-stability, offset by `19` gains. The losses are free-text prefix/completion failures: SQuAD numeric/entity fragments, HellaSwag `[substeps]` / `[step]` continuations, and StoryCloze short sentence prefixes. Post-hoc zero-loss aggregate rows average `2.191/4` blocks, still worse than joint-readiness riskcap04 (`2.118/4`, zero observed losses). Riskcap04 remains the current cross-seed safety baseline.

Fragment-completeness v3 riskcap04 check:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_cross_task_shape_features_fragmentguardv3_choice2_riskcap04_cross_seed_20260524
```

The v3 guard adds non-stable completion checks for replacement characters, trailing continuation markers, bare currency amounts, numeric range prefixes, connector endings, short unstable phrases, and truncated hyphen tokens. No-riskcap v3 was stopped early because seed66/HellaSwag produced `25` losses versus prediction-stability. With `risk_threshold_end=0.4` restored, seed66/67/68 official8 stays at zero observed losses and averages `21.596%` weighted micro accuracy at `2.245/4` blocks, saving `0.267` block versus prediction-stability. This is safer than no-riskcap but still worse than the current joint-readiness riskcap04 baseline (`2.118/4`, zero losses), so keep v3 as a diagnostic completion signal rather than a new default policy.

Terminology: `trace collection` is not training and not the final evaluation itself. It runs official Cola block-wise rollout to record per-block latent/probe/text/stability signals. The scorer, readiness frontier, readiness/risk training, and halt evaluations are downstream consumers of that trace.

## Trace Smoke Command

Use disabled SwanLab only for engineering smoke tests:

```bash
source /data1/luyifei/drla/scripts/activate_conda.sh
PYTHONPATH=/data1/luyifei/Cola-DLM/code:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 \
python -m drla.scripts.collect_cola_block_traces \
  --task-name lambada \
  --input-jsonl /data1/luyifei/Cola-DLM/code/generate_task_data/lambada.jsonl \
  --output-dir /data1/luyifei/drla/outputs/smoke_cola_block_traces/tasks_smoke \
  --max-samples 1 \
  --batch-size 1 \
  --max-new-tokens 16 \
  --timestep-num 1 \
  --swanlab-mode disabled
```

Full trace/evaluation runs must use `swanlab_mode=disabled` and `tasks_<alias>` output directories so the official `scripts/acc_calc.py` can discover `lambada.jsonl`, `mmlu.jsonl`, etc. No-training scripts reject any non-disabled SwanLab mode; only scripts with a real optimizer/backward training loop may use SwanLab cloud.

## Archived Legacy Code

The old self-built Stage A/B/C path, GSM8K diagnostic wrappers, and custom small Cola-latent-prior scripts were moved to:

```text
/data1/luyifei/drla/archive/2026-05-24-legacy-stagebc-code
```

Old Stage A/GSM8K data, Stage B/C outputs, custom-prior outputs, and old smoke artifacts were moved to:

```text
/data1/luyifei/drla/archive/2026-05-24-legacy-stagebc-artifacts
```

Both archives are reproducibility-only. Do not use them as the main experimental path.
