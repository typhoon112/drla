# P1 Model Comparison Report

Last updated: 2026-05-27

## Scope

This report summarizes the current Phase P1 latent-halt experiments against the official Cola full benchmark traces, the Phase P0 decoder-probed teacher, and the main P1 ablations.

Important evaluation distinction:

- **Official Cola scorer**: full prepared-split benchmark scoring over the official 8 tasks. This measures the frozen Cola substrate's final-answer accuracy and is not changed by P1 halt.
- **P1 halt evaluation**: leave-one-task-out student halt on held-out test partitions with 5 target-calibration subseeds. Accuracy/loss/mismatch are compared against the same-split fixed-final and prediction-stability baselines.
- **P0 teacher**: decoder-probed/text-stability-supervised readiness/risk policy. It is the teacher and upper-bound diagnostic, not the final latent-only agent communication policy.

## Official Cola Full Benchmark

Official Cola full benchmark uses `b64`, `bs12`, seeds `66/67/68`, and the official 8-task scorer.

| Task | Official Cola accuracy, mean +/- std (%) |
|---|---:|
| LAMBADA | 51.867 +/- 0.615 |
| MMLU | 20.593 +/- 0.314 |
| OBQA | 22.867 +/- 0.499 |
| HellaSwag | 4.087 +/- 0.066 |
| RACE | 20.740 +/- 0.404 |
| SIQA | 29.717 +/- 0.844 |
| SQuAD | 22.450 +/- 0.299 |
| StoryCloze | 28.220 +/- 0.482 |
| Unweighted task average | 25.070 +/- 0.135 |

Artifact sources:

```text
/data1/luyifei/drla/outputs/cola_official_benchmarks/full_b64_bs12_trace_score_20260524/summary.json
/data1/luyifei/drla/outputs/cola_official_benchmarks/full_b64_bs12_seed67_trace_score_20260524/summary.json
/data1/luyifei/drla/outputs/cola_official_benchmarks/full_b64_bs12_seed68_trace_score_20260524/summary.json
```

## Main P1 Models

All rows below are full official8, 3 seeds, LOTO-style P1 evaluations unless noted. Losses and mismatches are versus prediction-stability on the same evaluation samples.

| Model / policy | Online inputs | Accuracy | Blocks | Losses | Mismatches | Interpretation |
|---|---|---:|---:|---:|---:|---|
| P1 baseline `d64_pma4` | latent/process only | 21.556% | 2.048/4 | 58 | 1612 | Learns P0-like signals but unsafe on HellaSwag/SQuAD. |
| `answer_identity_action + completion_risk` | latent/process only | 22.481% | 1.742/4 | 47 | 617 | Cheaper and lower mismatch than baseline, still too many losses. |
| `answer_identity_halt + completion_risk` | latent/process only | 22.498% | 1.824/4 | 31 | 699 | Fair target-task strict aggregate; fewer losses than action route, but more blocks and mismatch. |
| `trajectory_token + action + completion_risk` | latent/process only | 22.492% | 1.711/4 | 41 | 806 | Shows trajectory/delta helps, but not a clean win. |
| `trajectory_token + action + completion_risk + answer_identity_stability` | latent/process only | 22.528% | 1.812/4 | 4 | 606 | Current best P1 student-only low-loss/cost frontier. |
| Same as above + hard `contentful>=0.5` | latent/process + calibrated student contentful head | 22.534% | 2.924/4 | 0 | 151 | Safe diagnostic but costlier than prediction-stability. |
| Same as above + `empty_answer_risk` | latent/process only | 22.508% | 1.829/4 | 24 | 623 | Negative result; converts empty risk into prefix/continuation misses. |
| Learned action->halt gate v2, cost-limited | scalar P1 heads | 22.534% | 1.859/4 | 10 | 465 | Better mismatch than best student, but more losses. |
| Learned action->halt gate v2, safety | scalar P1 heads | 22.534% | 2.722/4 | 0 | 130 | Strong safety point, but expensive. |
| P0 joint-readiness riskcap04 teacher | decoder-probed/text-derived features | 21.596% weighted full split | 2.118/4 | 0 | not same text-mismatch audit | Teacher/upper-bound diagnostic; not decoder-free. |

Best current P1 student-only model:

```text
trajectory_token + answer_identity_action + completion_risk + answer_identity_stability
```

Artifact:

```text
/data1/luyifei/drla/outputs/cola_experiment_summaries/official8_full_b64_bs12_p1_trajtok_answer_identity_action_completionrisk_identitystable_boundarypen02_cross_seed_20260527/summary.json
```

It reduces losses from `47` to `4` versus the earlier action route while keeping block cost close to the strongest low-cost policies. It does not beat the zero-loss decoder-probed or safety-gated baselines, but it is the best latent/process-only student so far.

## P1 Student Architecture And Training Strategy

The P1 student keeps the official Cola VAE/DiT substrate frozen. It does not retrain Cola and does not change the final answer distribution at block 4. The student only learns an online halt/readiness policy over the visible latent block prefix.

Current best student:

```text
LatentHaltStudent-v1
width d64
PMA pooling with 4 queries
process_token full features
trajectory_token interaction
answer_identity_action readiness target
completion_risk auxiliary head
answer_identity_stability auxiliary head
boundary penalty 0.2 in target calibration
```

Online inputs:

- visible latent blocks up to the current block;
- latent norms/deltas/cosine/drift and block-budget features;
- no decoded answer text, no decoder EOS/im_end probe, and no task scorer result at inference.

Offline teacher labels:

- `answer_identity_action`: first block whose decoded/scored answer identity matches the prediction-stability/final reference;
- `completion_risk`: current decoded answer is empty or a strict prefix/incomplete continuation of the stable/final reference;
- `answer_identity_stability`: current block answer identity already equals the stable/final reference.

This means P1 is a decoder-supervised latent student: decoder/text signals are used to create labels during training/evaluation, but not as online inputs. This is the intended bridge from P0 teacher to a future decoder-free agent latent-communication policy.

Training protocol:

- official 8 tasks: `lambada, mmlu, obqa, hellaswag, race, siqa, squad, story_cloze`;
- seeds `66/67/68`, leave-one-task-out;
- all P1 training used CUDA/GPU, SwanLab cloud, local `metrics.jsonl`, `best_checkpoint.pt`, and `last_checkpoint.pt`;
- best trajectory/identity-stability route used `valid_interval=50`;
- eval aggregation used local-only SwanLab-disabled scripts because aggregation/eval is not a training run.

## Comparison To Official Cola

There are two fair comparisons:

1. **Official Cola final-answer accuracy**: run all 4 blocks and score the official prepared benchmark. P1 does not aim to improve this number.
2. **Same-split halt comparison**: compare P1 selected block against the same sample's fixed-final/prediction-stability answer. This measures whether the student can stop earlier without changing correctness.

The table below is the same-split halt comparison for the best P1 student. It should be read as "how much correctness does P1 preserve while saving blocks", not as a claim that P1 improves Cola's official benchmark accuracy.

| Task | Official Cola full acc mean (%) | P1 same-split fixed acc (%) | P1 selected acc (%) | Delta vs same-split fixed | Blocks saved vs final |
|---|---:|---:|---:|---:|---:|
| LAMBADA | 51.867 | 53.756 | 53.756 | +0.000 | 0.746 |
| MMLU | 20.593 | 22.288 | 22.278 | -0.010 | 2.544 |
| OBQA | 22.867 | 27.027 | 27.027 | +0.000 | 2.791 |
| HellaSwag | 4.087 | 3.980 | 3.980 | +0.000 | 2.201 |
| RACE | 20.740 | 20.040 | 20.040 | +0.000 | 2.417 |
| SIQA | 29.717 | 32.437 | 32.437 | +0.000 | 2.663 |
| SQuAD | 22.450 | 22.907 | 22.894 | -0.013 | 2.136 |
| StoryCloze | 28.220 | 30.769 | 30.769 | +0.000 | 2.435 |

Because the official full scorer and P1 LOTO repeated-sample protocol are not identical datasets/weightings, the official column is a benchmark anchor, while the fixed-vs-selected delta is the apples-to-apples halt metric. Under that halt metric, the best P1 student loses only `4/73,645 = 0.00543%` repeated samples while saving `2.188` blocks versus final-block decoding and `0.696` blocks versus prediction-stability.

## Per-Benchmark P1 Best Model

The table below uses the P1 held-out test repeated-sample protocol, not the full official scorer. `Fixed` is the same-split final block accuracy. `PS blocks` is prediction-stability average blocks.

| Task | P1 best acc (%) | Fixed acc (%) | Blocks | PS blocks | Losses | Mismatches |
|---|---:|---:|---:|---:|---:|---:|
| LAMBADA | 53.756 | 53.756 | 3.254 | 2.018 | 0 | 5 |
| MMLU | 22.278 | 22.288 | 1.456 | 2.605 | 2 | 7 |
| OBQA | 27.027 | 27.027 | 1.209 | 2.405 | 0 | 0 |
| HellaSwag | 3.980 | 3.980 | 1.799 | 2.689 | 0 | 93 |
| RACE | 20.040 | 20.040 | 1.583 | 2.528 | 0 | 5 |
| SIQA | 32.437 | 32.437 | 1.337 | 2.308 | 0 | 0 |
| SQuAD | 22.894 | 22.907 | 1.864 | 2.492 | 2 | 494 |
| StoryCloze | 30.769 | 30.769 | 1.565 | 2.425 | 0 | 2 |

Interpretation:

- The best P1 model preserves same-split final accuracy on 6/8 tasks with zero observed losses.
- Residual correctness losses are only MMLU and SQuAD under this repeated-sample protocol.
- SQuAD still dominates text-identity mismatch (`494/606`), so the unresolved problem is answer-boundary identity, not benchmark accuracy.

## P0 Teacher Per-Benchmark Reference

P0 joint-readiness riskcap04 is decoder-probed and text-stability-supervised. It uses richer signals than P1 and should be treated as teacher/upper bound, not final deployment policy.

| Task | P0 risk-gated acc (%) | Fixed acc (%) | P0 blocks | PS blocks | Losses |
|---|---:|---:|---:|---:|---:|
| LAMBADA | 51.866 | 51.866 | 1.670 | 2.015 | 0 |
| MMLU | 20.598 | 20.593 | 2.616 | 2.620 | 0 |
| OBQA | 22.867 | 22.867 | 2.351 | 2.351 | 0 |
| HellaSwag | 4.089 | 4.089 | 1.788 | 2.676 | 0 |
| RACE | 20.756 | 20.742 | 2.516 | 2.517 | 0 |
| SIQA | 29.717 | 29.717 | 2.334 | 2.334 | 0 |
| SQuAD | 22.453 | 22.450 | 1.865 | 2.504 | 0 |
| StoryCloze | 28.220 | 28.220 | 1.486 | 2.439 | 0 |

P1 has learned substantial P0 structure: it reaches similar cost on SQuAD (`1.864` vs P0 `1.865`) and cheaper cost on many multiple-choice tasks, while using only latent/process online inputs. But P0 still has zero observed losses because it is allowed to use decoder/text-derived features.

## Main Ablation Conclusions

### Architecture

| Ablation | Result | Conclusion |
|---|---|---|
| `all_tokens` pooling | 2 losses / 17 mismatches / 3.979 blocks on seed68 | Too conservative; not a useful default. |
| `pma1` | 5 losses / 315 mismatches / 2.793 blocks on seed68 | Single PMA query over-compresses evidence. |
| `mean_max` | 4 losses / 115 mismatches / 2.837 blocks on seed68 | Reduces mismatch but costlier than baseline. |
| `d128_pma4` | 4 losses / 285 mismatches / 2.432 blocks on seed68 | Blind capacity increase is not the lever. |
| `d32_pma4` | 27 losses / 263 mismatches / 2.126 blocks on seed68 | Smaller width is cheaper but unsafe. |
| `no_block_budget` | 8 losses / 389 mismatches / 1.831 blocks on seed68 | Block/budget features are still important calibration anchors. |
| `film` process interaction | 8 losses / 190 mismatches / 2.260 blocks on seed68 | Simple FiLM does not replace process/trajectory tokens. |
| `trajectory_token` | 41 losses / 806 mismatches / 1.711 blocks cross-seed | Positive architecture signal, but needs better identity objective. |

### Objectives and Calibration

| Ablation | Result | Conclusion |
|---|---|---|
| `answer_identity_action + completion_risk` | 47 losses / 617 mismatches / 1.742 blocks | Good low-cost route, still unsafe. |
| `answer_identity_halt + completion_risk` | 31 losses / 699 mismatches / 1.824 blocks | Fewer losses than action route, but costs more blocks and mismatch. |
| `answer_identity_stability` head | 4 losses / 606 mismatches / 1.812 blocks | Best P1 student-only point. |
| Hard `contentful>=0.5` | 0 losses / 151 mismatches / 2.924 blocks | Useful diagnostic, too expensive. |
| `empty_answer_risk` head | 24 losses / 623 mismatches / 1.829 blocks | Negative transfer/proxy mismatch. |
| Wilson risk control on current sweeps | strict targets select 0 complete folds | Current calibration size/scores cannot certify low risk. |

## Paper-Level Takeaway

1. **Cola accuracy is not improved by P1, and that is not the goal.** The official frozen Cola full-benchmark task average remains about `25.07%`. P1 changes when to stop, not what final answer Cola can produce.
2. **P1 learns meaningful P0 signals.** The best latent/process-only student reaches `1.812/4` blocks with only `4/73,645` repeated-sample losses against prediction-stability.
3. **P0 remains the safety upper bound.** P0 riskcap04 gets zero observed losses at `2.118/4` blocks but depends on decoder/text features, so it is not the final agent latent-communication policy.
4. **The main unresolved risk is answer identity boundary.** SQuAD mismatch and prefix/continuation cases dominate; adding narrow heads such as `empty_answer_risk` is not sufficient.
5. **Best current scientific statement:** P1 has learned enough of P0's decoder-probed readiness signal to support the latent-student route, but final deployment still needs stronger answer-identity risk modeling or a more rigorous calibration/risk-control protocol.

## Recommended Stopping Point For P1

For paper/reporting, use these as the P1 headline models:

1. `trajectory_token + answer_identity_action + completion_risk + answer_identity_stability` as the best latent/process-only student.
2. P0 joint-readiness riskcap04 as the decoder-probed teacher/upper bound.
3. Learned action->halt gate v2 safety as an additional safety-cost reference.
4. `empty_answer_risk` as a negative auxiliary-head ablation.

Do not present P1 as improving official Cola benchmark accuracy. Present it as reducing latent block budget while approximately preserving final-answer correctness under a student-only online-input policy.
