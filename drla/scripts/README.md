# Script Status

Last updated: 2026-06-06

## Main Path

The main path is official Cola block-wise readiness / halt analysis through P1, then same-substrate agent-agent latent communication in P2. The active entrypoints cover benchmark preparation, trace collection, oracle frontier construction, readiness / halt training and eval, P1 locked packet construction, P2-A packet v2 construction, and planned P2 latent-message audits.

P1 readiness / halt evaluation uses the official Cola 8-task benchmark suite:

```text
lambada, mmlu, obqa, hellaswag, race, siqa, squad, story_cloze
```

P2 agent-agent communication must not use official8-only evidence as the main
MAS claim.  The active route is capability-gated, role-conditioned MAS.  Pure
evaluation/generation scripts must keep `swanlab_mode=disabled`; only scripts
with backward/optimizer training loops may create SwanLab cloud runs.

Post-Family1 current next-plan:

```text
/data1/luyifei/drla/docs/current/
P2_Locked_Complete_Execution_Scheme_2026-06-01.md

/data1/luyifei/drla/docs/current/
P2_Post_Family1_Complete_Execution_Plan_2026-06-01.md

Branch B Family 1 is stopped. The locked route is Phase C true MAS
benchmark/protocol validation, then Phase A CoLA substrate/interface adaptation,
then Phase E CoLA TextMAS vs LatentMAS. Do not execute the historical Planned
P2-D/E items below as the current next experiment. The only allowed early work is
docs/artifact indexing, existing-result aggregation, manifest/scorer/dry-inspect
tools, and code hygiene that does not run held-out, training, fuser/adapter, or
P2 main tables.
```

## Active Scripts

- `prepare_cola_official_benchmarks.py`: prepares the official Cola 8-task benchmark JSONL files under `/data1/luyifei/Cola-DLM/code/generate_task_data`.
- `collect_cola_block_traces.py`: runs official Cola block-wise generation and records per-block raw latent shards, decoder probe stats, EOS/im_end probe stats, generated text so far, answer-stability signals, and local `metrics.jsonl`. It has no training loop and only accepts `swanlab_mode=disabled`. Default output is score-ready under `/data1/luyifei/drla/outputs/cola_block_traces/tasks_block_trace`.
- `merge_cola_block_trace_segments.py`: merges OOM/resume trace segments into one score-ready root, validating generation id order and trace sample ids. Use it for full prepared splits when one task is collected in multiple segments.
- `eval_cola_benchmarks.py`: wraps official Cola `scripts/acc_calc.py`, parses the 8-task CSV, and writes local summaries. It has no training loop and only accepts `swanlab_mode=disabled`.
- `prepare_cola_p2_candidate_benchmarks.py`: prepares P2 capability-gate JSONL files for ARC-E, ARC-C, GSM8K, MBPP+, HumanEval+, GPQA-Diamond, and MedQA. Current pinned sources are `allenai/ai2_arc`, `openai/gsm8k`, `evalplus/*`, `hendrydong/gpqa_diamond_mc`, and `GBaker/MedQA-USMLE-4-options`.
- `prepare_cola_p2_official8_role_candidates.py`: safe Branch-B data-prep script. It converts existing official CoLA benchmark JSONL files into the normalized P2 gate schema for official8-compatible role-candidate triage. Default tasks are `obqa,mmlu,race,hellaswag,siqa,story_cloze`; `squad` and `lambada` are supported but excluded by default. It performs no generation, no training, no held-out inspection, and creates no SwanLab run. Current full artifact is `/data1/luyifei/drla/outputs/p2_benchmark_redesign/official8_role_candidates_20260601` with 33296 rows.
- `build_cola_p2_locked_splits.py`: builds deterministic calibration/held-out partitions for P2-D2. Prompt/protocol repair may use calibration rows only; held-out rows are locked for later capability gates and P2 main-table evaluation.
- `run_cola_p2_capability_gate.py`: local-only CoLA generation/evaluation gate for candidate P2 benchmarks. It runs single-solver and Planner→Critic→Refiner→Solver text-MAS protocols, writes `generations.jsonl`, `metrics.jsonl`, `task_summary.csv`, and `summary.json`, and never logs to SwanLab because it has no training loop. Code tasks must use `--enable-code-execution` for main gate claims; syntax-only mode is only a pre-gate. `--prompt-variant generic_v1|cola_fewshot_v1|answer_state_v1|answer_state_structured_v1|role_plan_ignore_v1` is for calibration-only prompt repair unless a variant later passes held-out gate. `--single-prompt-variant` and `--role-prompt-variant` may override it when single baseline and Role TextMAS need different prompt contracts; this must be recorded in the output config and should not be used to cite held-out results unless the exact split-prompt protocol is locked first.
- `aggregate_cola_p2_capability_gate.py`: local-only post-processor for completed P2 capability-gate summaries. It writes task-level and mode-level CSVs plus aggregate `summary.json` for document updates and paper-table triage, including `single_prompt_variant` and `role_prompt_variant` when present.
- `run_cola_p2_official8_native_single_gate.py`: local-only Branch-B alignment audit. It recovers raw official8 rows from `/data1/luyifei/Cola-DLM/code/generate_task_data`, groups by native `task_name`, calls Cola's original `apply_prompt_template`, and scores with acc_calc-style choice-text / first-segment preprocessing. It never runs Role TextMAS, never touches held-out unless explicitly pointed there for a locked gate, and never logs to SwanLab.
- `rescore_cola_p2_official8_native_single_gate.py`: local-only post-processor for native single generations. It recomputes official acc_calc-style scores without rerunning GPU generation; use it when scoring logic changes.
- `audit_cola_p2_protocol_repair_failures.py`: local-only paired failure taxonomy for P2-D3 prompt/protocol repair. It compares single-solver and Role TextMAS outcomes on the same calibration examples, writes paired role-help/role-harm/parser-failure summaries, and must not be run on held-out during prompt repair.
- `fetch_p2_phase_c_hf_rows.py`: local-only Phase C data-source row fetcher. It uses HuggingFace datasets-server `/rows` with seeded multi-block offsets, writes source `rows.jsonl`, `fetch_manifest.json`, `summary.json`, and `metrics.jsonl`. It does not construct benchmarks, run models, train, inspect held-out generations, or create SwanLab runs.
- `prepare_p2_phase_c_evidence_split_qa_records.py`: local-only evidence-split QA record builder for HotpotQA / MuSiQue / 2Wiki-style JSONL rows. It creates manifest-ready `p2_phase_c_manifest_v0` sample records with split agent private views, offline scoring, leakage-audit booleans, and shortcut-risk summary. It does not download data, run models, train, or create SwanLab runs.
- `build_p2_phase_c_control_inputs.py`: local-only Phase C control-input packager. It expands a validated manifest into `single_q_only`, `single_full_info`, `textmas_matched`, `textmas_no_message`, `textmas_shuffled_message`, `textmas_wrong_evidence_or_wrong_shard`, and `textmas_compressed_state` online inputs, writes prompt previews, and runs the leakage auditor. It does not call models, train, inspect held-out generations, or create SwanLab runs. Current default prompt contract is `p2_phase_c_evidence_split_v1_strict_wrong_evidence`; the wrong-evidence condition gives all evidence agents private shards from a non-self control sample. Older v0 packages that replaced only one shard are weak-control diagnostics, not admission artifacts.
- `preflight_p2_phase_c_text_agent_run.py`: local-only Phase C/Phase A text-agent preflight. It checks manifest/control consistency, estimates solver/evidence-agent chat calls, records OpenAI-compatible, local-transformers, or `cola_dlm` readiness, and writes `run_plan.json`, `summary.json`, and `metrics.jsonl`. It never calls models, trains, inspects held-out generations, or creates SwanLab runs.
- `run_p2_phase_c_text_agents.py`: local-only Phase C/Phase A text-agent runner. It consumes control inputs, runs sequential evidence-agent -> solver flows through an OpenAI-compatible chat/completions endpoint, `--provider local_transformers`, or `--provider cola_dlm`, scores final answers offline, and writes `generations.jsonl`, `condition_metrics.csv`, `summary.json`, and `metrics.jsonl`. It never trains or logs to SwanLab. `--selfcheck --provider mock_selfcheck` is only a toy routing/scoring check and must not be cited as an experiment. `--resume` restores existing `generations.jsonl` and agent-message cache for long calibration runs; if a disk-full or process interruption leaves a truncated final JSONL line, resume rewrites a clean prefix before appending. Local-transformers runs are real model outputs, but small local smokes are wiring checks, not Phase C admission evidence. Qwen-style thinking mode is disabled by default through `enable_thinking=False`; pass `--local-enable-thinking` only for diagnostics where `<think>` content is intentionally part of the protocol. `cola_dlm` runs use official CoLA VAE/DiT generation under the same Phase C online inputs/scorer and are Phase A substrate/interface diagnostics unless run at locked gate scale. Supported CoLA prompt styles are `chat_join`, `plain_qa_v1`, and `squad_template_v1`; `--cola-dit-lora-path` loads a trained DiT LoRA adapter, `--cola-noise-seed` defaults to deterministic seed 66, and `--prediction-extraction-mode first_segment` is the CoLA-style parser that cuts generated `<|endoftext|>` / `<|im_end|>` boundaries. 2026-06-05 MuSiQue prompt-only smokes with these styles are negative diagnostics, while the same local weights pass an official SQuAD 20-sample sanity at 5/20 primary matches.
- `build_p2_phase_a_cola_interface_sft.py`: local-only Phase A SFT data builder. It uses MuSiQue calibration manifest plus admitted Qwen3-8B-FP8 calibration TextMAS generations to create CoLA interface SFT pairs for `solver_full_info`, `solver_textmas_matched`, and `evidence_agent_teacher`. It writes `sft_pairs.jsonl`, sample-level `train.jsonl` / `valid.jsonl`, `summary.json`, and `metrics.jsonl`; it does not train, call models, inspect held-out data, or create SwanLab runs. Current artifact: `/data1/luyifei/drla/outputs/p2_phase_a_cola_interface_sft/musique_calibration_qwen_teacher_v1_20260605`.
- `build_p2_phase_a_candidate_answer_sets.py`: local-only Phase A candidate-answer builder. It extracts answer candidates from evidence text only, labels them with gold/aliases only for offline coverage/scoring, and writes `candidates.jsonl`, `summary.json`, and `metrics.jsonl`. It must not inject gold as a candidate, inspect held-out during repair, train, call models, or create SwanLab runs.
- `train_p2_phase_a_candidate_answer_selector.py`: local-only sklearn candidate selector diagnostic. It trains logistic or HistGradientBoosting selectors over evidence-derived candidate features to test whether shallow ranking can pick the gold answer from the current candidate pool. It does not run deep-learning optimization, call models, use held-out data, or create SwanLab runs. Current best shallow calibration result is qtype logistic top128 at `selected_primary=0.175`, below the locked `0.20` solver floor.
- `run_p2_phase_a_candidate_answer_llm_selector.py`: local-only semantic candidate-selector diagnostic. It uses a capable local LLM such as Qwen3-8B-FP8 to choose a short answer from evidence-derived candidates plus online evidence/question, then scores offline. It performs inference only, never training, never SwanLab, and must not use held-out for prompt/objective selection. It supports deterministic `--num-shards` / `--shard-index` and `--progress-interval` for long train-source teacher generation. The 2026-06-06 calibration200 top128 run reached `selected_primary=0.445`, showing the candidate interface is semantically usable even though shallow selectors fail. The completed train10k top128 all10 aggregate reached `selected_primary=0.4336`, `selected_exact_match=0.3893`, `selected_token_f1=0.4941`, `oracle_coverage_kept=0.8012`; it is allowed only as nonheldout teacher evidence for constrained short-answer CoLA repair, not as a CoLA or Phase E result.
- `aggregate_p2_phase_a_candidate_answer_llm_selector.py`: local-only semantic selector shard aggregator. It merges one or more `predictions.jsonl` files or directories, deduplicates by sample id, recomputes offline metrics, and writes aggregate `predictions.jsonl`, `summary.json`, and `metrics.jsonl`. It does not call models, train, inspect held-out data, or create SwanLab runs.
- `build_p2_phase_a_candidate_constrained_sft.py`: local-only Phase A candidate-constrained short-answer SFT builder. It reads the nonheldout MuSiQue train manifest, evidence-derived candidate answers, and the completed Qwen semantic selector teacher to emit CoLA-compatible short-answer pairs. Online prompts contain only question, full online evidence, and candidate text/rank/rule/source metadata; gold/aliases and teacher correctness are offline-only filtering/statistics. Current full artifact: `/data1/luyifei/drla/outputs/p2_phase_a_cola_interface_sft/musique_candidate_constrained_short_answer_train10000_top128_qwen_teacher_20260606` with 12348 pairs (`solver_candidate_gold_covered=8012`, `solver_candidate_teacher_correct=4336`), prompt token p95 about 4577 and block p95 about 287 under the official CoLA tokenizer/block size.
- `train_p2_phase_a_cola_dit_lora.py`: Phase A deep-learning training script for official CoLA DiT LoRA interface adaptation. It freezes the official CoLA VAE, trains only LoRA parameters on block-level Flow-Matching targets from calibration SFT pairs, appends target EOS by default, requires CUDA/GPU and SwanLab cloud, enforces `valid_interval <= 10`, and writes local `metrics.jsonl`, `best_checkpoint.pt`, `last_checkpoint.pt`, `best_adapter/`, and `last_adapter/`. Current positive calibration adapters are the solver adapter `/data1/luyifei/drla/outputs/p2_phase_a_cola_dit_lora/musique_solver_interface_lora_v1_epoch3_eos_20260605` and the evidence-agent adapter `/data1/luyifei/drla/outputs/p2_phase_a_cola_dit_lora/musique_evidence_agent_lora_v1_from_solver_epoch2_20260605`. Together they pass the 200-sample MuSiQue calibration Role TextMAS gate at `/data1/luyifei/drla/outputs/p2_phase_c_text_agent_aggregates/musique_cola_dlm_dual_lora_evidence_epoch2_solver_epoch3eos_squadtemplate_v1_firstseg_seed66_calibration_full200_20260605`, but fail the identical 800-sample locked held-out gate at `/data1/luyifei/drla/outputs/p2_phase_c_text_agent_aggregates/musique_cola_dlm_dual_lora_evidence_epoch2_solver_epoch3eos_squadtemplate_v1_firstseg_seed66_heldout800_20260605` because `single_full_info=0.0875 < 0.2`; do not start Phase E from this checkpoint and do not use held-out for further tuning.
- `train_p2_phase_a_cola_latent_candidate_ranker.py`: Phase A deep-learning trainer for a frozen-CoLA-VAE latent candidate-answer ranker. It encodes only online question/evidence/candidate text into CoLA latent representations, trains a lightweight ranker, requires CUDA/GPU and SwanLab cloud, enforces `valid_interval <= 10`, writes `metrics.jsonl`, `best_checkpoint.pt`, `last_checkpoint.pt`, and evaluates the saved best checkpoint for final summary metrics. It supports historical `--interaction-mode pooled` and the current `--interaction-mode late_maxsim`, which adds ColBERT-style candidate-token to context-token MaxSim features. The first top128 step500 schema-v2 pooled screen at `/data1/luyifei/drla/outputs/p2_phase_a_cola_latent_candidate_ranker/musique_top128_step500_schema_v2_seed20260606_20260606` reached best-checkpoint calibration200 `selected_primary=0.095`; the matched late-maxsim screen at `/data1/luyifei/drla/outputs/p2_phase_a_cola_latent_candidate_ranker/musique_top128_step500_latemaxsim_seed20260606_20260606` improved to `0.120`, still below the shallow qtype selector `0.175` and Qwen semantic selector `0.445`. This is useful evidence that token-level interaction helps but does not pass Phase A.
- `eval_p2_phase_a_cola_latent_candidate_ranker.py`: local-only checkpoint evaluator for CoLA-latent candidate rankers. It restores the saved split/config, evaluates `best_checkpoint.pt` or `last_checkpoint.pt` on nonheldout valid plus calibration candidates, writes local `metrics.jsonl`, predictions, and `summary.json`, and must run with `--swanlab-mode disabled`.
- `aggregate_p2_phase_c_text_agent_results.py`: local-only Phase C result aggregator and admission gate. It computes condition metrics, paired bootstrap confidence intervals, and the locked criteria for `single_full_info > single_q_only` and `textmas_matched > no_message/shuffled/wrong_evidence`. It does not run models, train, inspect held-out generations, or create SwanLab runs.
- `build_p2_phase_c_manifest.py`: local-only Phase C manifest packager. It reads normalized sample records JSONL, writes `manifest.json`, `samples.jsonl`, `summary.json`, and `metrics.jsonl`, then runs `validate_p2_phase_c_manifest.py`. It does not download data, generate samples, run models, or create SwanLab runs.
- `validate_p2_phase_c_manifest.py`: local-only Phase C manifest hygiene checker. It validates the `p2_phase_c_manifest_v0` schema, split presence, required baselines, agent private-view fields, scorer fields, and explicit leakage-audit booleans. It writes `summary.json` and `metrics.jsonl`, does not run models, and must not create SwanLab runs.
- `inspect_p2_phase_c_dataset_fields.py`: local-only Phase C data-source dry inspector. It reads local JSON/JSONL preview files, writes field path/type/length summaries, record hashes, source/license metadata, `summary.json`, and `metrics.jsonl`. It does not download data, construct manifests, run models, tune prompts, inspect held-out generations, train, or create SwanLab runs.
- `selfcheck_p2_phase_c_scorers.py`: local-only scorer self-check for Phase C QA and structured scorers. It uses hardcoded toy assertions only, writes `summary.json` and `metrics.jsonl`, and must not be cited as an experiment.
- `audit_p2_phase_c_run_leakage.py`: local-only run-artifact leakage auditor for future Phase C protocol runs. It checks `generations.jsonl` online-input fields against the manifest, blocks explicit gold/scorer/full-evidence fields in forbidden conditions, requires non-self control sample ids for shuffled/wrong-message controls, and writes `leakage_audit.json`, `summary.json`, and `metrics.jsonl`.
- `build_cola_readiness_frontier.py`: uses official scorer rules to build offline oracle readiness/frontier labels from per-block traces.
- `train_cola_readiness_model.py`: trains the multi-signal readiness/halt model over raw latent blocks plus process/probe/stability features. It writes SwanLab metrics, local `metrics.jsonl`, `best_checkpoint.pt`, and `last_checkpoint.pt`.
- `train_cola_latent_halt_student.py`: trains Phase P1 `LatentHaltStudent-v1`. Decoder/scorer/stability fields are teacher targets only; online inputs are raw latent prefix blocks plus selected latent/process features, with CUDA/GPU, SwanLab cloud, local `metrics.jsonl`, and best/last checkpoints. It supports explicit teacher-loss weights, `--selection-metric`, `--process-feature-mode full|no_block_budget`, `--process-interaction-mode process_token|film|trajectory_token`, `--readout-context-mode none|last_process_query`, `--use-completion-risk`, `--use-empty-answer-risk`, `--use-answer-format-risk`, `--use-answer-identity-stability`, and P0 teacher targets via `--readiness-target-mode oracle_frontier|p0_teacher_halt|p0_teacher_action`. `trajectory_token` keeps the normal process token and appends a per-block trajectory token built from pooled block state, previous-block delta, and process state before causal inter-block attention. Current selection metrics include `readiness_prediction_change_completion_contentful_mean_auroc`, `readiness_prediction_change_completion_empty_mean_auroc`, `readiness_prediction_change_completion_format_mean_auroc`, `readiness_prediction_change_completion_identity_mean_auroc`, and `readiness_prediction_change_completion_empty_identity_mean_auroc` for boundary/identity diagnostics. The full official8 `empty_answer_risk + answer_identity_stability` run is a negative result; do not treat the combined metric as a default without the matching aggregate evidence.
- `eval_cola_latent_halt_student.py`: evaluates a `LatentHaltStudent-v1` checkpoint as a student-only halt policy. It calibrates thresholds on valid, reports test accuracy-cost, saves threshold sweeps, student scores, and halt decisions, and audits answer-text mismatch versus final/prediction-stability outputs. It has no training loop and only accepts `swanlab_mode=disabled`. `--require-zero-calibration-mismatch` adds a stricter valid-set text-stability calibration guard on top of loss calibration. `--max-calibration-mismatches` / `--max-calibration-mismatch-rate` provide a softer risk cap that can keep zero-loss policies from selecting overly aggressive low-block thresholds. `--calibration-boundary-risk-penalty` changes threshold selection from pure min-block to `avg_blocks + penalty * boundary_risk_slack`, which is useful when finite target calibration over-selects permissive risk thresholds. `--max-calibration-samples-per-task` plus `--calibration-subsample-seed` runs deterministic target-calibration sample-cap diagnostics. `--empty-answer-risk-thresholds`, `--answer-format-risk-thresholds`, and `--answer-identity-stability-thresholds` are only active for checkpoints trained with the matching heads. `--calibration-scope per_task` requires the selected threshold to satisfy the same constraints on every calibration task instead of only on the pooled valid split.
- `aggregate_cola_latent_halt_student_loto.py`: aggregates P1 leave-one-task-out eval summaries with loss/mismatch rates, Wilson upper bounds, per-task risk buckets, and cross-seed recurrence. It has no training loop and must stay local-only.
- `aggregate_cola_latent_halt_student_subseed_loto.py`: aggregates nested calibration-subseed LOTO eval summaries such as `subseed*/leave_*_out_eval_*_test/summary.json`. It writes repeated-sample micro metrics, seed/task/subseed CSVs, and is local-only.
- `analyze_latent_halt_risk_control.py`: local-only formal risk-control audit for existing P1 threshold sweeps. It selects thresholds using validation-set Wilson upper bounds for loss/mismatch risk, then reports held-out official8/cross-seed trade-offs. Use `--allow-legacy-swanlab-eval` only for historical eval summaries created before the local-only eval rule.
- `build_cola_agent_latent_comm_packets.py`: builds sanitized P2 Agent A -> Agent B latent communication packets from locked P1 halt decisions and latent refs. It supports `cola_agent_latent_comm_v1` and `cola_agent_latent_comm_v2`; v2 adds `communication_boundary`, `prefix_contract`, and `agent_b_contract` for same-substrate handoff. It has no training loop and must stay local-only; online packet fields must exclude decoded text, token ids, scorer outputs, gold/target labels, and selected/final/prediction-stability correctness.
- `audit_cola_agent_latent_packet_distribution.py`: local-only P2-B audit. It loads v2 packet latent refs, checks packet/tensor consistency, native trace feature alignment, forbidden fields, and matched-vs-corrupted controls (`metadata_only`, `shuffle`, `cross_task`, `wrong_block`, `noise`, `rotation`). It has no training loop and must stay local-only.
- `audit_cola_latent_receiver_targets.py`: local-only P2-C target feasibility audit. It reads packet `audit_refs` and offline halt decisions to measure accept/defer label sparsity; these offline fields are target diagnostics only and must not become receiver inputs.
- `train_cola_latent_receiver.py`: trains the P2-C same-substrate latent receiver compatibility model. It uses balanced matched-vs-corrupted packet payload labels, requires CUDA/GPU and SwanLab cloud, writes local `metrics.jsonl`, `best_checkpoint.pt`, and `last_checkpoint.pt`, and never feeds decoded text, scorer outputs, correctness labels, or `control_type` to the model.
- `eval_cola_latent_receiver.py`: local-only P2-C checkpoint evaluator. It reconstructs the train/valid/test split, rebuilds matched/corrupted examples, loads a saved best checkpoint, and writes fresh compatibility metrics with `swanlab_mode=disabled`.
- `aggregate_cola_latent_receiver.py`: local-only aggregation for P2-C receiver ablations. It reads receiver `summary.json` files and writes a comparison CSV/summary; it must use `swanlab_mode=disabled`.
- `eval_cola_adaptive_halt.py`: sweeps adaptive halt thresholds and compares fixed-B, oracle halt, prediction-stability, learned threshold-only halt, aggressive readiness-or-stability, and stability-gated guarded policies. It has no training loop and only accepts `swanlab_mode=disabled`.
- `analyze_cola_halt_decisions.py`: reconstructs per-sample halt decisions from a saved eval summary and checkpoint, then writes `halt_decisions.jsonl`, `policy_comparison.csv`, `readiness_bins.csv`, and `summary.json`.
- `aggregate_cola_halt_decision_analysis.py`: aggregates per-task halt decision diagnostics across leave-one-task-out runs.
- `train_cola_continuation_risk_model.py`: trains a non-gold continuation-risk model for gating readiness halt before prediction stability. `--target-mode strict_prefix` detects prefix/incomplete answers; `--target-mode prediction_change` detects whether the current task-scored prediction will differ from the rollout prediction-stability reference.
- `eval_cola_risk_gated_halt.py`: evaluates readiness halt gated by continuation-risk probability. It has no training loop and only accepts `swanlab_mode=disabled`. `risk_threshold_selection_mode=min_blocks` is aggressive; `first_saving` is safer. `--readiness-threshold-values` lets validation jointly sweep readiness and risk thresholds. `--require-zero-calibration-loss` prevents valid-set loss/gain cancellation by requiring zero losses versus prediction-stability when such rows exist. `--require-contentful-prediction` adds a non-gold answer-shape guard against punctuation-only early halt. `--require-fragment-complete-prediction` guards non-stable decoded answer fragments such as pure-number prefixes, unfinished abbreviations, trailing initials, and short hyphen fragments. `--require-stable-single-choice` requires single `A`-`E` predictions to reach prediction stability before halt, addressing multiple-choice letter flips; `--stable-single-choice-max-block 1` or `2` narrows that guard to early blocks. Use `--stable-single-choice-guard-scopes off,1,2,3,all` to let validation select the guard scope jointly with risk threshold.
- `eval_cola_risk_gated_halt.py` also supports strict uncertainty calibration through `--entropy-max-values` and `--top-prob-min-values`. These sweep non-gold decoder confidence guards in addition to risk threshold; defaults preserve the old behavior.
- `analyze_cola_risk_gated_halt_decisions.py`: reconstructs sample-level risk-gated halt decisions and writes per-sample loss/gain diagnostics against fixed-final and prediction-stability.
- `aggregate_cola_risk_gated_halt.py`: aggregates same-task and leave-one-task-out risk-gated halt summaries plus common threshold sweeps.

## P2 Latent Communication Scripts

Current P2 canonical design:

```text
/data1/luyifei/drla/docs/current/P2_Latent_MAS_Communication_Implementation_Plan_2026-05-29.md
/data1/luyifei/drla/docs/current/P2_Locked_Complete_Execution_Scheme_2026-06-01.md
/data1/luyifei/drla/docs/current/P2_Post_Family1_Complete_Execution_Plan_2026-06-01.md
/data1/luyifei/drla/docs/current/P2_Phase_C_Benchmark_Protocol_Preparation_2026-06-01.md
```

Implemented:

- `build_cola_agent_latent_comm_packets.py`: local-only packet builder. It creates the v1 sanitized latent message substrate and the current P2-A v2 same-substrate single-handoff packets from locked P1 outputs, then writes packet summaries, `packet_schema.json`, `metrics.jsonl`, latent-ref checks, forbidden-field audits, and v2 field-coverage audits.
- `audit_cola_agent_latent_packet_distribution.py`: local-only packet distribution auditor. It validates P2-B packet/tensor consistency and writes `summary.json`, `distribution_stats.csv`, `control_stats.csv`, `ood_detection.csv`, `packet_examples.jsonl`, and `metrics.jsonl`.
- `audit_cola_latent_receiver_targets.py`: local-only target audit for P2-C. It shows whether accept/defer labels are dense enough to support a supervised receiver objective.
- `train_cola_latent_receiver.py`: P2-C compatibility receiver trainer. Full runs use CUDA/GPU, SwanLab cloud, `valid_interval<=10`, `metrics.jsonl`, `best_checkpoint.pt`, and `last_checkpoint.pt`.
- `eval_cola_latent_receiver.py`: P2-C local-only checkpoint evaluator for best-checkpoint rescoring.
- `aggregate_cola_latent_receiver.py`: P2-C local-only ablation aggregator.
- `run_cola_sequential_latent_mas.py`: P2-D local-only replay runner. It re-encodes the shared task context, consumes/replays Agent A latent packet blocks into official Cola DiT/VAE caches, continues generation under the receiver budget, and writes scorer-ready `tasks_<control>` directories plus `summary.json`, `metrics.jsonl`, `generations.jsonl`, and `control_comparison.csv`.
- `audit_cola_sequential_latent_mas.py`: P2-D local-only post-hoc audit. It combines replay generations, official scorer correct/wrong files, and offline P1 halt-decision references to report paired matched-vs-control score changes and answer-prefix fidelity. P1 reference fields are audit-only and must not become online receiver inputs.
- `build_cola_text_handoff_baseline.py`: P2-D local-only direct-answer handoff diagnostic. For the same replay subset, it materializes `text_selected`, `text_final`, and `text_prediction_stability` direct-handoff outputs from P1 halt-decision references into scorer-ready `tasks_text_*` directories. It does not feed text into Agent B and must not be used as the main B_text channel baseline.
- `train_cola_latent_answer_reader.py`: P2-D receiver-side latent answer-state reader trainer. It uses CUDA/GPU and SwanLab cloud, writes `metrics.jsonl`, `best_checkpoint.pt`, and `last_checkpoint.pt`, and trains a lightweight latent/process-feature encoder against P1 teacher selected_prediction text states.
- `eval_cola_latent_answer_reader.py`: P2-D local-only best-checkpoint evaluator for the latent answer reader. It restores the packet split and candidate answer pool, retrieves answer states, scores them with the official Cola scorer, and writes local `summary.json` / `metrics.jsonl`.
- `audit_cola_hierarchical_aggregation_potential.py`: P2-E local-only aggregation-potential audit. It groups the three locked seed66/67/68 same-sample sender packets and compares single-sender, text majority, simple latent-readiness rankers, and oracle any-correct upper bounds.
- `train_cola_hierarchical_latent_fuser.py`: P2-E hierarchical latent fuser trainer. Full runs use CUDA/GPU, SwanLab cloud, `valid_interval<=10`, `metrics.jsonl`, `best_checkpoint.pt`, and `last_checkpoint.pt`. It supports `--target-mode correct`, `--target-mode score`, `--target-mode task_aware_score`, and `--task-loss-weighting balanced`; all modes keep online inputs decoder-free and use decoded/scored text only as offline supervision.
- `eval_cola_hierarchical_latent_fuser.py`: P2-E local-only best-checkpoint evaluator with aggregate and per-task metrics for the hierarchical latent fuser.
- `train_cola_hierarchical_state_verifier.py`: P2-E multi-sender latent-state utility verifier. It predicts group-level `any_correct` and `best_score` from decoder-free latent/process/certificate inputs, uses SwanLab cloud for training, and writes `metrics.jsonl`, `best_checkpoint.pt`, `last_checkpoint.pt`, and summary best/final metrics.
- `eval_cola_hierarchical_state_verifier.py`: P2-E local-only calibration and input-ablation evaluator for the latent-state verifier. It restores the locked split, evaluates zero-latent / zero-process / zero-certificate ablations, fits valid-only post-hoc calibrators, and writes `summary.json`, `metrics.jsonl`, `ablation_metrics.csv`, and `calibration_report.json`.
- `eval_cola_hierarchical_state_policy.py`: P2-E local-only receiver-state policy audit. It combines the calibrated latent-state verifier with the score-target fuser v2, selects thresholds on valid, and reports held-out test gates against text-majority, task-prior, and global-prior controls. It writes `summary.json`, `metrics.jsonl`, `policy_metrics.csv`, `per_group_states.jsonl`, and `train_priors.json`.
- `train_cola_receiver_state_action_selector.py`: P2-E structured receiver-state action selector trainer. It consumes calibrated state/fuser/prior features, uses offline selected-score labels, requires CUDA/GPU and SwanLab cloud, and writes `metrics.jsonl`, `best_checkpoint.pt`, `last_checkpoint.pt`, and `test_predictions.jsonl`. It supports direct sender logits and residual-fuser logits.
- `eval_cola_receiver_state_action_selector.py`: P2-E local-only evaluator for action selector checkpoints. It rebuilds the locked split, restores feature normalization, selects thresholds only on valid, and reports held-out test aggregate/gated metrics.
- `train_cola_state_conditioned_latent_fuser.py`: P2-E state-conditioned sender-level latent fuser trainer. It initializes from score-target fuser v2, keeps raw sender-level latent/process/certificate inputs, adds calibrated state/fuser/prior side features, and learns a residual sender-logit delta. Training requires CUDA/GPU and SwanLab cloud.
- `audit_cola_request_more_latent_potential.py`: P2-E local-only audit for first-sender vs prefix2/prefix3 additional latent evidence and oracle/request-readiness upper bounds.
- `train_cola_request_more_policy.py`: P2-E first-sender request-more trainer. It predicts whether requesting remaining sender packets is useful; training requires CUDA/GPU and SwanLab cloud.
- `train_cola_post_request_selector.py`: P2-E anchor-aware post-request selector trainer. It selects among first/requested sender packets after a request action; training requires CUDA/GPU and SwanLab cloud.
- `train_cola_joint_request_select_policy.py`: P2-E joint request-and-select trainer. The request head sees only the first sender; the selector sees requested sender packets only after request; training requires CUDA/GPU and SwanLab cloud. It supports the original target025 checkpoint selection and `--checkpoint-selection-mode valid_rate_frontier`; the strict 2026-05-31 frontier audit is negative and should not replace the current canonical target025 aggregate.
- `eval_cola_joint_request_select_policy.py`: P2-E local-only checkpoint evaluator for any saved joint request/select checkpoint, including `last_checkpoint.pt`. It reconstructs valid/test predictions, selects request thresholds on valid, and writes local `summary.json`, `metrics.jsonl`, `policy_metrics.csv`, and prediction JSONL files.
- `audit_cola_joint_policy_calibration.py`: P2-E local-only calibration/risk-control audit for saved joint policy predictions. It selects request thresholds on valid, reports held-out test, writes `summary.json`, `metrics.jsonl`, `calibration_metrics.json`, and `risk_policy_metrics.csv`, and must use `swanlab_mode=disabled`.

Corrected P2-D channel-equivalent scripts:

- `build_cola_agent_channel_messages.py`: local-only builder for paired `A_raw_text_message_t` and `A_latent_packet_t` from the same Agent A trajectory/depth. It uses native trace `decode_text_so_far` at the selected block, not `selected_prediction`, for the text message.
- `run_cola_agent_b_channel_eval.py`: local-only Agent-B evaluator for the LatentMAS-aligned `B_none(empty input)`, `B_text(A_raw_text_message_t)`, `B_latent(A_latent_packet_t)`, and `B_corrupt(corrupted_latent_packet_t)` conditions under the same receiver budget and official scorer format. The default `--agent-b-input-contract message_only` gives B only Agent A's output/message; `--agent-b-input-contract shared_context` additionally gives B the original prompt and is diagnostic only. Latent blocks are loaded directly from packet shards, not re-encoded from text. The default `--score-output-scope receiver_only` makes the scorer see only Agent-B tokens generated after handoff; A text-message tokens and A latent replay decoded tokens are excluded from final `generate`. `--score-output-scope legacy_all_visible` exists only to reproduce historical diagnostics. Replay modes are `decode_and_emit`, `_cache_only` (VAE decoder cache + DiT cache, no replay text), `_dit_only_cache`, and `_vae_only_cache`.
- `merge_cola_agent_b_channel_eval_shards.py`: local-only merger for sharded Agent-B channel eval outputs. It validates expected messages/channels, duplicate keys, and writes a single `generations.jsonl` for aggregation.
- `aggregate_cola_channel_eval.py`: local-only post-hoc official scorer aggregate for Agent-B final-output score, paired win/loss/tie, cost, and decision-rule readouts. Gold/scorer outputs are not online Agent A/B inputs.
- `audit_cola_channel_projection_gap.py`: local-only paired audit for the decoder semantic-projection gap between decode-and-emit `latent_matched` and receiver-native/cache-only `latent_matched_cache_only`.
- `audit_cola_channel_protocol_boundaries.py`: local-only gate for P2-D Agent-B communication claims. It classifies eval roots as `receiver_only` pass or legacy/fail, and verifies that scorer-visible A text-message tokens and A replay blocks are zero.

Current P2-A locked v2 packet artifact:

```text
/data1/luyifei/drla/outputs/cola_agent_latent_comm/
p2_agent_latent_comm_v2_locked_seed66_67_68_split20260601_20260529

packets / latent refs / unique latent files:
  14940 / 27399 / 8850

missing latent files / forbidden decoder-eval fields:
  0 / 0

v2 coverage:
  communication_boundary = 14940 / 14940
  prefix_contract = 14940 / 14940
  agent_b_contract = 14940 / 14940
```

Current P2-B locked distribution audit artifact:

```text
/data1/luyifei/drla/outputs/cola_agent_latent_comm/
p2_agent_latent_comm_v2_distribution_audit_locked_seed66_67_68_split20260601_20260529

status:
  pass

packets audited / native aligned latent blocks:
  14940 / 27399

structural errors / forbidden fields / latent load errors:
  0 / 0 / 0

native alignment max_abs_diff:
  0.0

min corrupted-control pair-distance AUROC:
  1.0
```

Current P2-C target feasibility artifact:

```text
/data1/luyifei/drla/outputs/cola_agent_latent_comm/
p2_latent_receiver_target_audit_locked_seed66_67_68_split20260601_20260529

status:
  warn

offline accept / unsafe:
  14936 / 4

unsafe rate:
  0.0268%

naive all-accept accuracy:
  99.9732%
```

Current P2-C receiver compatibility artifact:

```text
/data1/luyifei/drla/outputs/cola_latent_receiver/
p2c_receiver_compat_bestckpt_eval_aggregate_seed20260529_20260529

best input mode:
  latent_process_certificate

best SwanLab run:
  50bumvfh2pgp0olw60jvr

test mean-control AUROC:
  0.9194

hardest listed control:
  shuffle AUROC = 0.6205

negative controls:
  envelope_only ~= 0.5003
  process_only ~= 0.5000
  certificate_only ~= 0.5000
```

Early P2-D sequential replay artifact:

```text
/data1/luyifei/drla/outputs/cola_sequential_latent_mas/
p2d_replay_eval_official8_5per_task_controls_20260529

scope:
  official8, 5 packets per task, 40 packets total
  matched, metadata_only, shuffle, cross_task, wrong_block, noise, rotation
  local-only, swanlab disabled, no training

official scorer:
  matched = 17.5%
  metadata_only = 20.0%
  wrong_block = 5.0%
  shuffle/cross_task/noise/rotation = 0.0%

offline answer-prefix fidelity:
  matched = 37.5%
  metadata_only = 27.5%
  noise = 7.5%
  shuffle/cross_task/wrong_block/rotation = 2.5%
```

Current P2-D direct-answer diagnostic and replay-only diagnostic:

```text
direct-answer diagnostic:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_eval_official8_5per_task_controls_20260529/text_handoff_baseline

official scorer:
  text_selected = 30.0%
  text_final = 30.0%
  text_prediction_stability = 30.0%

replay-only:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_only_eval_official8_5per_task_controls_20260529

matched replay-only:
  scorer accuracy = 17.5%
  answer-prefix fidelity = 37.5%
  replay-only vs native trace selected-block raw text = 60.0%
  native trace selected-block raw text vs P1 selected_prediction = 27.5%
```

Current expanded P2-D diagnostic:

```text
replay-only:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_only_eval_official8_20per_task_controls_20260529

replay+continue:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_continue_eval_official8_20per_task_controls_20260529

direct-answer diagnostic:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_only_eval_official8_20per_task_controls_20260529/text_handoff_baseline

official scorer:
  replay-only matched = 24.38%
  replay+continue matched = 24.38%
  replay+continue metadata_only = 21.88%
  corrupted latent controls = 0.00% to 1.88%
  text_selected = 25.00%

audit:
  selected_reference_accuracy = 25.0%
  matched official_prediction_agrees_selected = 65.0%
  matched correct_selected_preservation_rate = 80.0%
```

Planned:

- Fresh 50-per-task P2-D validation is now the main sequential replay evidence:

```text
replay-only:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_only_eval_official8_50per_task_seed20260530_controls_20260529

replay+continue:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_continue_eval_official8_50per_task_seed20260530_controls_20260529

scope:
  50 packets per official task
  400 packets total
  fresh selection seed = 20260530
  matched, metadata_only, shuffle, cross_task, wrong_block, noise, rotation
  local-only, swanlab disabled

official scorer:
  replay-only matched = 24.25%
  replay-only metadata_only = 0.00%
  replay-only corrupted controls = 0.00% to 3.75%
  replay+continue matched = 24.25%
  replay+continue metadata_only = 23.75%
  replay+continue corrupted controls = 0.25% to 4.25%
  text_selected = 23.50%

audit:
  duplicate-safe scorer correctness lookup is required because task ids/sample_keys repeat
  replay-only matched net wins = +97 vs metadata_only, +82 to +97 vs corrupted controls
  replay+continue matched net wins = +2 vs metadata_only, +80 to +96 vs corrupted controls
```

Next:

- Treat matched-vs-corrupted as the strong current P2-D evidence.
- Do not claim text superiority: the current `text_selected` rows are direct-answer diagnostics, not Agent-B text-message baselines, and metadata_only+continue is close to matched.
- Main next step is scaling the corrected LatentMAS-aligned Agent-B channel-equivalent evaluation: `B_none(empty input)`, `B_text(A_raw_text_message_t)`, `B_latent(A_latent_packet_t)`, and `B_corrupt(corrupted_latent_packet_t)` under `--agent-b-input-contract message_only --score-output-scope receiver_only`.
- `run_cola_sequential_latent_mas.py` supports `--receiver-context-mode full_prompt|empty_prompt`.
- Corrected channel-equivalent smoke, 2026-05-31:

```text
messages:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_channel_messages_smoke_1per_task_20260531

eval:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_smoke_1per_task_20260531

aggregate:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_smoke_1per_task_20260531/channel_eval_aggregate

scope:
  official8, 1 message per task, 8 channels, 64 final Agent-B generations
  swanlab_mode=disabled, CUDA generation, no training

official scorer smoke readout:
  latent_matched = 25.00% accuracy, mean score 0.5754
  text_raw_message = 25.00% accuracy, mean score 0.5799
  none = 25.00% accuracy, mean score 0.6059
  corrupted latent controls = 0.00% accuracy, mean score 0.1720-0.2005

interpretation:
  protocol path is executable and matched latent is well above corrupted
  controls, but 8 samples are smoke only. Do not treat this as a paper number.
```
- Legacy all-visible official8 50/task unique-sample diagnostic, 2026-05-31:

```text
messages:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_channel_messages_official8_50per_task_seed20260531_unique_20260531

sharded eval:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_official8_50per_task_seed20260531_unique_20260531_sharded6

merged eval:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_official8_50per_task_seed20260531_unique_20260531_merged

aggregate:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_official8_50per_task_seed20260531_unique_20260531_merged/channel_eval_aggregate

scope:
  official8, 50 unique sample_key per task, 400 messages
  8 channels, 3200 final Agent-B generations
  swanlab_mode=disabled, CUDA sharded generation, no training

official scorer:
  latent_matched = 23.50%, mean score 0.4850
  text_raw_message = 23.00%, mean score 0.5011
  none = 19.75%, mean score 0.4410
  corrupted controls = 0.00% to 4.25%, mean score 0.1699-0.2499

paired deltas:
  matched - corrupted score_delta = +0.2351 to +0.3151, all CI95 lower > +0.1959
  matched - none score_delta = +0.0440, CI95 [+0.0111, +0.0758]
  matched - text score_delta = -0.0161, CI95 [-0.0406, +0.0076]

interpretation:
  This table used the historical all-visible output scope, so A text-message
  tokens and A decoded latent replay tokens could be scored directly. It is a
  decodability/replay-output diagnostic only, not valid Agent-B communication
  evidence. The earlier non-deduped 50/task run had only 375 unique sample_key
  and is protocol-audit only.
```
- Receiver-native cache-only audit, 2026-05-31:

```text
artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_official8_50per_task_cache_only_seed20260531_unique_20260531_merged/channel_eval_aggregate

protocol:
  replay latent blocks update VAE/DiT KV cache
  replay blocks are not sampled/emitted into output text
  replay_blocks_decoded_to_text = 0
  same 400 unique messages as the formal decode-and-emit run

official scorer:
  latent_matched_cache_only = 1.50%, mean score 0.1916
  none = 19.75%, mean score 0.4410
  text_raw_message = 23.00%, mean score 0.5011
  corrupted cache-only controls = 0.75% to 3.25%, mean score 0.1559-0.2261

paired deltas:
  matched_cache_only - none score_delta = -0.2493, CI95 [-0.2851, -0.2152]
  matched_cache_only - text score_delta = -0.3094, CI95 [-0.3484, -0.2686]

interpretation:
  current positive P2-D evidence is decode-and-emit payload readability, not
  receiver-native no-text latent reasoning. Future receiver work should consume
  latent state natively instead of only replaying sender latent through decoder.
```
- Corrected receiver-only channel smoke, 2026-05-31:

```text
artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_smoke_receiver_only_1per_task_20260531/channel_eval_aggregate

protocol:
  --score-output-scope receiver_only
  scorer-visible A text message tokens = 0
  scorer-visible A replay blocks = 0

official scorer:
  none = 25.00%, mean_score 0.6059
  text = 12.50%, mean_score 0.2996
  latent_matched = 0.00%, mean_score 0.2082
  latent_matched_cache_only = 0.00%, mean_score 0.2082
  latent_matched_dit_only_cache = 0.00%, mean_score 0.2082
  latent_matched_vae_only_cache = 0.00%, mean_score 0.3373

interpretation:
  Historical all-visible decode-and-emit results are downgraded to decodability
  diagnostics. They must not be cited as Agent-B communication results.
```
- LatentMAS-aligned message-only receiver-only smoke after replay-EOS fix, 2026-05-31:

```text
artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_smoke_message_only_receiver_only_eosfix_1per_task_20260531/channel_eval_aggregate

protocol:
  --agent-b-input-contract message_only
  --score-output-scope receiver_only

code fix:
  replay EOS/im_end from Agent A no longer stops Agent B under receiver_only.
  Only B-generated stop tokens terminate receiver generation.

boundary audit:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_protocol_boundary_audit_message_only_eosfix_smoke_20260531
  status = pass
  scorer-visible A text message tokens = 0
  scorer-visible A replay blocks = 0

official scorer:
  none = 0.00%, mean_score 0.1527
  text = 0.00%, mean_score 0.2022
  latent_matched = 0.00%, mean_score 0.2457
  latent_matched_cache_only = 0.00%, mean_score 0.2478
  latent_matched_dit_only_cache = 0.00%, mean_score 0.2405
  latent_matched_vae_only_cache = 0.00%, mean_score 0.1559

paired smoke deltas:
  latent_matched - none score_delta = +0.0930
  latent_matched - text score_delta = +0.0435

interpretation:
  This is the canonical protocol smoke, not a quality claim. The formal
  official8 50/task result below supersedes it for channel-quality discussion.
```
- Formal LatentMAS-aligned P2-D 50/task result, 2026-05-31:

```text
artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_official8_50per_task_message_only_receiver_only_eosfix_seed20260531_unique_20260531_merged

aggregate:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_official8_50per_task_message_only_receiver_only_eosfix_seed20260531_unique_20260531_merged/channel_eval_aggregate

audit:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_protocol_boundary_audit_message_only_receiver_only_eosfix_50task_20260531

scope:
  official8, 50 unique sample_key per task
  400 messages, 11 channels, 4400 Agent-B generations
  duplicate_keys = 0
  missing_message_rows = 0
  protocol audit = pass

official scorer:
  latent_matched = 1.75%, mean_score 0.1874
  latent_matched_cache_only = 1.75%, mean_score 0.1912
  latent_matched_dit_only_cache = 1.25%, mean_score 0.1873
  latent_matched_vae_only_cache = 0.00%, mean_score 0.1395
  none = 0.00%, mean_score 0.1378
  text = 1.25%, mean_score 0.1795
  latent_cross_task = 0.25%, mean_score 0.1535
  latent_shuffle = 0.25%, mean_score 0.1663
  latent_wrong_block = 4.00%, mean_score 0.1950
  latent_noise = 0.00%, mean_score 0.1390
  latent_rotation = 0.00%, mean_score 0.1297

paired readout:
  matched - none score_delta = +0.0496, CI95 [+0.0340, +0.0659]
  matched - text score_delta = +0.0079, CI95 [-0.0082, +0.0259]
  matched - wrong_block score_delta = -0.0076, CI95 [-0.0286, +0.0140]
  matched - cache_only score_delta = -0.0038, CI95 [-0.0064, -0.0012]

interpretation:
  Valid receiver-only Agent-B communication evidence exists for marginal utility
  over empty input. Latent is competitive with text but does not significantly
  beat text. The all-corrupt gate fails because wrong_block is anomalously
  strong, so payload-specific robustness remains unresolved.
```
- Decoder semantic-projection gap audit, 2026-05-31:

```text
artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_projection_gap_official8_50per_task_seed20260531_unique_20260531

scope:
  same 400 unique official8 50/task messages
  paired latent_matched decode-and-emit vs latent_matched_cache_only

result:
  projection_score_gain = +0.2934, CI95 [+0.2551, +0.3320]
  projection_accuracy_gain = +22.00pp, CI95 [+17.75pp, +26.25pp]
```

- Direct DiT/VAE cache ablation smoke, 2026-05-31:

```text
artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_smoke_direct_dit_1per_task_20260531_v2/channel_eval_aggregate

scores:
  latent_matched = 25.00%, mean_score 0.5754
  latent_matched_cache_only = 0.00%, mean_score 0.2082
  latent_matched_dit_only_cache = 0.00%, mean_score 0.2082
  latent_matched_vae_only_cache = 0.00%, mean_score 0.3373

interpretation:
  DiT-only direct replay is executable but does not recover decode-and-emit
  utility on the 8-sample smoke. This is protocol evidence only.
```
- Message-only marginal-utility diagnostic:

```text
artifact:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_message_only_replay_only_official8_50per_task_seed20260531_controls_20260529

protocol:
  receiver_context_mode = empty_prompt
  receiver_budget_mode = fixed
  fixed_receiver_blocks = 0

official scorer:
  matched latent = 12.50%
  metadata_only = 0.00%
  corrupted controls = 0.00% to 2.00%
  text_selected = 26.25%

audit:
  matched net wins = +50 vs metadata_only, +42 to +50 vs corrupted controls
```

This shows latent packets alone carry recoverable signal, but current
message-only replay is far worse than the direct-answer diagnostic. The next
main experiment is not more direct-handoff scaling; it is channel-equivalent
Agent-B evaluation plus receiver-native no-text latent consumption.

Receiver-side latent answer reader diagnostic:

```text
training artifact:
  /data1/luyifei/drla/outputs/cola_latent_answer_reader/
  p2_answer_reader_full_seed20260529_20260529

SwanLab cloud run:
  x6yc77eedf77z27ego0ve

best checkpoint test:
  answer_key_top1 = 15.20%
  official_top1_accuracy = 10.61%
  selected_reference_accuracy = 22.11%
```

This confirms the negative boundary: latent packets are readable, but the
current lightweight answer reader is not better than the direct-answer
diagnostic and does not establish text-channel competitiveness.

P2-E hierarchical aggregation:

```text
potential audit artifact:
  /data1/luyifei/drla/outputs/cola_hierarchical_aggregation/
  p2e_aggregation_potential_locked_seed66_67_68_split20260601_20260529

potential audit:
  single_sender_first = 20.74%
  text_majority_selected = 21.55%
  best simple latent-state ranker = 21.39%
  oracle_any_selected_correct = 33.13%

learned fuser artifact:
  /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
  p2e_hierarchical_fuser_full_seed20260529_20260529

SwanLab cloud run:
  ljv0m43x48a49j1at6gx9

best checkpoint test:
  model_selected_accuracy = 20.74%
  single_sender_first_accuracy = 22.38%
  text_majority_selected_accuracy = 23.41%
  oracle_any_selected_accuracy = 34.70%
```

P2-E v1 is a useful negative result: multi-sender packets have oracle headroom,
but the current decoder-free latent fuser does not beat single-sender or text
majority baselines on held-out test.

P2-E score-target fuser v2:

```text
supervision audit:
  non-binary official-score sender predictions = 7764 / 14940 = 52.0%
  partial-best groups = 2570 / 4980 = 51.6%

learned fuser artifact:
  /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
  p2e_hierarchical_fuser_score_full_seed20260529_20260529

SwanLab cloud run:
  o5fjvuiqk82nk9c5hihn0

best checkpoint test:
  model_selected_accuracy = 23.41%
  single_sender_first_accuracy = 22.38%
  text_majority_selected_accuracy = 23.41%
  oracle_any_selected_accuracy = 34.70%
  model_mean_official_score = 0.3685
  single_sender_first_mean_official_score = 0.3553
  text_majority_mean_official_score = 0.3622
  oracle_best_selected_mean_official_score = 0.4951
```

This is the strongest P2-E result so far: score-target latent fusing beats the
single fixed sender and matches text majority on exact test accuracy, while
beating both on mean official score. It is still below oracle and uneven across
tasks, so do not claim robust latent-over-text superiority yet.

P2-E negative follow-ups:

```text
task-balanced v3:
  artifact:
    /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
    p2e_hierarchical_fuser_score_taskbalanced_full_seed20260529_20260529
  SwanLab cloud run:
    k2ujjjdrmcyzutwnwbyyf
  test:
    model_selected_accuracy = 22.38%
    model_mean_official_score = 0.3654

task-aware v4:
  artifact:
    /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
    p2e_hierarchical_fuser_taskaware_score_full_seed20260529_20260529_metricfix
  SwanLab cloud run:
    fdjwe4vfq70syq0oz9tro
  test:
    model_selected_accuracy = 22.18%
    model_mean_official_score = 0.3605
```

Both are below score-target v2. Keep v2 as the current P2-E best model.

P2-E latent-state utility verifier:

```text
artifact:
  /data1/luyifei/drla/outputs/cola_hierarchical_state_verifier/
  p2e_state_verifier_full_seed20260529_20260529

SwanLab cloud run:
  brtvqv9yd3h2gbcsu25n5

best checkpoint test:
  any_auroc = 0.7054
  any_accuracy_at_0_5 = 58.52%
  any_brier = 0.2208
  best_score_corr = 0.3078
  best_score_rmse = 0.4029

same-test heuristic baselines:
  max_correctness_head any_auroc = 0.4717, best_score_corr = -0.0769
  max_readiness any_auroc = 0.4776, best_score_corr = -0.1232

calibration / ablation artifact:
  /data1/luyifei/drla/outputs/cola_hierarchical_state_verifier/
  p2e_state_verifier_full_seed20260529_20260529_calibration_ablation

calibrated test:
  any_brier = 0.1985
  ece_10 = 0.0232
  any_prob_mean = 0.3418 vs target 0.3470

input / prior ablation:
  full any_auroc = 0.7054, best_score_corr = 0.3078
  zero_latent any_auroc = 0.6731, best_score_corr = 0.2317
  task_prior any_auroc = 0.6399, best_score_corr = 0.2399
  global_prior any_auroc = 0.5142, best_score_corr = 0.0000
```

This supports latent-state utility readability beyond simple certificate
heuristics and beyond task-prior alone. Calibration improves probability
scale, but this is not yet a downstream text-superiority result.

P2-E calibrated receiver-state policy audit:

```text
artifact:
  /data1/luyifei/drla/outputs/cola_hierarchical_state_policy/
  p2e_state_policy_score_fuser_v2_locked_seed20260529_20260529

always-on test:
  single_sender_first accuracy = 22.38%, score = 0.3553
  score-target latent fuser v2 accuracy = 23.41%, score = 0.3685
  text_majority_selected accuracy = 23.41%, score = 0.3622
  oracle_any_selected accuracy = 34.70%, score = 0.4951

state_any_prob gate, valid target any precision 0.60:
  test coverage = 13.35%
  test any precision = 61.54%
  accepted fuser accuracy = 53.85%
  accepted fuser score = 0.5503
  accepted text accuracy = 49.23%
  fuser-else-first fallback accuracy = 23.00%
  fuser-else-first fallback score = 0.3609

state_any_prob gate, valid target any precision 0.65:
  test coverage = 11.29%
  accepted fuser accuracy = 56.36%
  accepted fuser score = 0.5636

train task-prior any gate, valid target any precision 0.60:
  test coverage = 12.32%
  accepted fuser accuracy = 55.00%
  accepted fuser score = 0.5500
```

The state policy audit says the calibrated latent state can find high-utility
subsets, but the current fuser/fallback action does not yet beat always-on
fuser overall. Keep this as receiver-state evidence, not final communication
superiority.

P2-E structured receiver-state action selector:

```text
direct selector:
  artifact = /data1/luyifei/drla/outputs/cola_receiver_state_action_selector/
             p2e_state_action_selector_state_fuser_prior_seed20260529_20260529
  SwanLab = 7emwtma3xyvrmvlv1hibb
  best test accuracy = 21.97%
  best test score = 0.3554

residual selector:
  artifact = /data1/luyifei/drla/outputs/cola_receiver_state_action_selector/
             p2e_state_action_selector_residual_state_fuser_prior_seed20260529_20260529
  SwanLab = 2z5uj588g6kkkv6dpte97
  best test accuracy = 22.79%
  best test score = 0.3626

reference baselines on the same held-out test:
  score-target latent fuser v2 accuracy = 23.41%, score = 0.3685
  text_majority_selected accuracy = 23.41%, score = 0.3622
```

The shallow action selector is negative: residual-fuser output is safer than
direct logits, but neither variant beats the raw fuser v2. Treat the compressed
state tuple as a risk/readiness side signal, not as a replacement for
sender-level latent selection.

P2-E state-conditioned sender-level latent fuser:

```text
frozen residual:
  artifact = /data1/luyifei/drla/outputs/cola_state_conditioned_latent_fuser/
             p2e_state_conditioned_fuser_frozen_state_fuser_prior_seed20260529_20260529
  SwanLab = 1ua24n9yo4tsrq4inahb9
  best test accuracy = 23.00%
  best test score = 0.3654

unfrozen residual:
  artifact = /data1/luyifei/drla/outputs/cola_state_conditioned_latent_fuser/
             p2e_state_conditioned_fuser_unfrozen_state_fuser_prior_seed20260529_20260529
  SwanLab = je3suuujcleox4x40lahd
  best test accuracy = 23.41%
  best test score = 0.3651

reference:
  score-target latent fuser v2 accuracy = 23.41%, score = 0.3685
  text_majority_selected accuracy = 23.41%, score = 0.3622
```

This preserves raw sender-level latent states but still does not beat fuser v2.
The next P2-E branch should test request-more-latent / additional evidence
rather than more reranking over the same three senders.

P2-E request-more-latent potential audit:

```text
script = /data1/luyifei/drla/drla/scripts/audit_cola_request_more_latent_potential.py
artifact = /data1/luyifei/drla/outputs/cola_request_more_latent/
           p2e_request_more_latent_potential_locked_seed20260529_20260529
mode = local-only, swanlab_mode=disabled
split = train / valid / test = 3998 / 495 / 487 grouped samples
```

Held-out test:

```text
first sender only = 22.38%, score 0.3553
prefix2 oracle = 30.39%, score 0.4458
prefix3 oracle = 34.70%, score 0.4951
prefix2 readiness selector = 23.61%, score 0.3661
prefix3 readiness selector = 22.38%, score 0.3518

request first -> prefix3 helpful rate = 33.47%
mean first -> prefix3 score gain = +0.1398
```

Interpretation: additional sender latent evidence has a strong oracle upper
bound, but current decoder-free request/readiness heuristics do not close much
of that gap. The next non-degenerate branch is a learned request policy or
sequential aggregator, with first-only, text-majority, always-request,
task-prior, global-prior, and oracle-after-request controls.

P2-E learned request-more-latent policy:

```text
script = /data1/luyifei/drla/drla/scripts/train_cola_request_more_policy.py
online input = first sender sanitized latent/process/certificate/task fields
action = stop after first sender or request remaining sender packets
post-request selector = score-target latent fuser v2
```

Smoke:

```text
artifact = /data1/luyifei/drla/outputs/cola_request_more_policy/
           p2e_request_more_policy_smoke_256groups_seed20260529_20260529
SwanLab = zh936nbxnbrow88z7savu
```

Full runs:

```text
fuser_gain:
  artifact = /data1/luyifei/drla/outputs/cola_request_more_policy/
             p2e_request_more_policy_fuser_gain_full_seed20260529_20260529
  SwanLab = zn7zl11z11ghmfenr8wr4
  best step = 350
  test helpful AUROC = 0.6823
  best practical policy = target_request_rate 0.25 on gain_pred
  request rate / avg sender budget = 23.00% / 1.46
  fuser-after-request accuracy / score = 23.20% / 0.3689

oracle_gain:
  artifact = /data1/luyifei/drla/outputs/cola_request_more_policy/
             p2e_request_more_policy_oracle_gain_full_seed20260529_20260529
  SwanLab = at0w7v8gsewja1vudb3jx
  best step = 350
  test helpful AUROC = 0.6461
  best practical fuser-after-request accuracy / score = 22.79% / 0.3660

references:
  first sender only = 22.38%, score 0.3553
  text majority = 23.41%, score 0.3622
  always request + fuser v2 = 23.41%, score 0.3685
  always request oracle upper bound = 34.70%, score 0.4951
```

Interpretation: the learned request policy gives the first narrow positive
budget-efficiency result. It slightly improves mean score over always-request
fuser while using about half the sender budget, but exact accuracy is slightly
lower and the score delta is tiny. The next bottleneck is post-request
aggregation/selection, not request detection alone.

P2-E post-request anchor-aware selector:

```text
script = /data1/luyifei/drla/drla/scripts/train_cola_post_request_selector.py
online input = first sender + requested sender latent/process/certificate/task fields
architecture = anchor-aware sender states + post-request Transformer
loss = score MSE + listwise ranking + pairwise ranking + gain regression
request-gated eval = previous fuser_gain request policy, target_request_rate 0.25 on gain_pred
```

Smoke:

```text
artifact = /data1/luyifei/drla/outputs/cola_post_request_selector/
           p2e_post_request_selector_anchor_smoke_256groups_seed20260529_20260529_rerun
SwanLab = r5jsobe8gh78m1ehuh818
```

Full runs:

```text
anchor score selection:
  artifact = /data1/luyifei/drla/outputs/cola_post_request_selector/
             p2e_post_request_selector_anchor_score_full_seed20260529_20260529
  SwanLab = r7qj2vu48vws5lnp5edxf
  standalone accuracy / score = 22.18% / 0.3572
  request-gated accuracy / score = 23.00% / 0.3668

anchor rank selection:
  artifact = /data1/luyifei/drla/outputs/cola_post_request_selector/
             p2e_post_request_selector_anchor_rank_full_seed20260529_20260529
  SwanLab = 089s7vnjawqpd4p7m20ak
  standalone accuracy / score = 21.77% / 0.3500
  request-gated accuracy / score = 23.41% / 0.3697

references:
  first sender only = 22.38%, score 0.3553
  text majority = 23.41%, score 0.3622
  always request + fuser v2 = 23.41%, score 0.3685
  request policy + fuser v2 = 23.20%, score 0.3689
  oracle upper bound = 34.70%, score 0.4951
```

Interpretation: standalone post-request selector is negative, but the
request-gated anchor-rank variant is a narrow positive budget-efficiency
result. It suggests that the next branch should jointly train request and
post-request selection, instead of composing two independently trained modules.

P2-E joint request-and-select policy:

```text
script = /data1/luyifei/drla/drla/scripts/train_cola_joint_request_select_policy.py
request head input = first sender latent/process/certificate/task fields only
selector head input = first + requested sender fields, only after request
loss = request BCE + request gain + score MSE + listwise rank + pairwise rank + gain + budgeted utility
```

Smoke:

```text
artifact = /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
           p2e_joint_request_select_smoke_256groups_seed20260529_20260529
SwanLab = munwuikpa4v8m8g9nysqf
```

Full runs:

```text
joint rank:
  artifact = /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
             p2e_joint_request_select_rank_full_seed20260529_20260529
  SwanLab = g126cuz3w32r9g76jizcs
  best gated policy = target_request_rate 0.50 on request_prob
  request rate / budget = 47.64% / 1.95
  accuracy / score = 23.82% / 0.3733

joint score:
  artifact = /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
             p2e_joint_request_select_score_full_seed20260529_20260529
  SwanLab = 2t0j6t9v33qfkfn71w2mi
  best gated policy = target_request_rate 0.50 on request_prob
  request rate / budget = 47.64% / 1.95
  accuracy / score = 24.02% / 0.3735

references:
  first sender only = 22.38%, score 0.3553
  text majority = 23.41%, score 0.3622
  always request + fuser v2 = 23.41%, score 0.3685
  request policy + fuser v2 = 23.20%, score 0.3689
  request policy + anchor-rank selector = 23.41%, score 0.3697
  oracle upper bound = 34.70%, score 0.4951
```

Interpretation: joint request-and-select is the current strongest P2-E
budgeted latent result. It is a locked-split, same-substrate win over text
majority and fuser baselines in mean score, with a small exact-accuracy gain.
Keep the claim narrow because the oracle gap is still large.

P2-E joint policy calibration / risk-control audit:

```text
script = /data1/luyifei/drla/drla/scripts/audit_cola_joint_policy_calibration.py
artifact = /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
           p2e_joint_request_select_score_full_seed20260529_20260529_calibration_risk
mode = local-only, swanlab_mode=disabled
split = valid 495 groups / test 487 groups
```

Readout:

```text
request_prob -> model_request_helpful:
  valid AUROC / ECE = 0.6959 / 0.2363
  test  AUROC / ECE = 0.7164 / 0.2226

request_prob -> oracle_request_helpful:
  valid AUROC / ECE = 0.7089 / 0.1050
  test  AUROC / ECE = 0.6825 / 0.1132

best utility row:
  target_request_rate = 0.50 on request_prob
  test request rate / budget = 47.64% / 1.95
  test accuracy / score = 24.02% / 0.3735
  requested model-loss Wilson95 upper = 35.02%

strict loss-risk caps:
  0.10 / 0.20: no non-trivial valid threshold
  0.30 / 0.40: almost-always request, test score drops to 0.3494
```

Interpretation: `request_prob` has useful ranking signal but is over-confident
as a calibrated probability. The current best policy is valid-selected
budgeted utility, not a risk-certified communication policy.

P2-E strict fresh split replication:

```text
overlap audit:
  seed30 test groups = 495
  seed30 test overlap with seed29 train/valid/test = 406 / 51 / 38

non-canonical diagnostic:
  artifact = /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
             p2e_joint_request_select_score_full_seed20260530_20260529
  SwanLab = jg65cg65hvzqxdybjc298
  issue = reused seed29 fuser checkpoint and norm_stats
```

Strict seed30 same-split fuser:

```text
training artifact = /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
                    p2e_hierarchical_fuser_score_full_seed20260530_20260529
SwanLab = qweuypbg1ugls3io0s9j0
best eval artifact = /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
                     p2e_hierarchical_fuser_score_full_seed20260530_20260529_best_eval_test
test fuser accuracy / score = 22.63% / 0.3694
```

Strict seed30 same-split joint:

```text
training artifact = /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
                    p2e_joint_request_select_score_full_seed20260530_fuserseed20260530_20260529
SwanLab = n4wu1f4ghzfwe6mhvltei
calibration artifact = /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
                       p2e_joint_request_select_score_full_seed20260530_fuserseed20260530_20260529_calibration_risk

best gated policy:
  target_request_rate = 0.50
  signal = request_gain_pred
  request rate / budget = 50.51% / 2.01
  accuracy / score = 23.64% / 0.3796

references:
  first sender = 22.42%, score 0.3669
  text majority = 22.63%, score 0.3642
  same-split fuser best = 22.63%, score 0.3694
  oracle upper bound = 35.96%, score 0.5071

calibration:
  request_prob model-helpful AUROC / ECE = 0.6255 / 0.2697
  requested model-loss Wilson95 upper = 25.40%
```

Interpretation: the strict seed30 rerun replicates the positive utility
direction against text and same-split fuser, while confirming that probability
calibration and risk certification remain weak. The earlier seed30 diagnostic
must not be used as canonical because of seed29 fuser/norm_stats overlap.

Strict seed31 same-split replication:

```text
same-split fuser:
  artifact = /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
             p2e_hierarchical_fuser_score_full_seed20260531_20260529
  SwanLab = 64h605uhcjpse62n84l8v
  best eval = /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
              p2e_hierarchical_fuser_score_full_seed20260531_20260529_best_eval_test
  test fuser accuracy / score = 17.75% / 0.3184

same-split joint:
  artifact = /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
             p2e_joint_request_select_score_full_seed20260531_fuserseed20260531_20260529
  SwanLab = xh19j75yervdr7l603rt4
  calibration artifact = /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
                         p2e_joint_request_select_score_full_seed20260531_fuserseed20260531_20260529_calibration_risk
```

Strict seed31 result:

```text
first sender = 16.96%, score 0.3074
text majority = 17.75%, score 0.3180
same-split fuser best = 17.75%, score 0.3184
joint selected = 19.33%, score 0.3426
oracle = 29.78%, score 0.4515

best selected policy:
  target_loss_wilson_upper = 0.30
  signal = request_prob
  request rate / budget = 99.80% / 3.00
  requested model-loss Wilson95 upper = 26.99%
```

Three-strict-seed aggregate:

```text
artifact = /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
           p2e_joint_request_select_strict_seed29_30_31_summary_20260529

macro mean:
  first sender = 20.59%, score 0.3432
  text majority = 21.26%, score 0.3481
  same-split fuser = 21.26%, score 0.3521
  joint selected = 22.33%, score 0.3652
  oracle = 33.48%, score 0.4846
  request rate / budget = 65.98% / 2.32
  request_prob model-helpful AUROC / ECE = 0.6688 / 0.2571
```

Checkpoint-selection audit:

```text
script = /data1/luyifei/drla/drla/scripts/eval_cola_joint_request_select_policy.py

last-checkpoint best policy:
  seed29 = 23.00%, score 0.3653
  seed30 = 24.85%, score 0.3919
  seed31 = 17.75%, score 0.3252
```

Interpretation: three strict seeds now support a positive macro utility
direction for latent joint request-select over text majority and same-split
fuser, but seed31 shows that the budget advantage is not stable. Last-checkpoint
eval is mixed, which motivated the valid-frontier audit below. Since that audit
is negative, the next target should be corrected channel-equivalent P2-D
evaluation or a genuinely new calibration/objective, not more shallow checkpoint
or threshold tuning.

P2-E valid-frontier checkpoint selection audit:

```text
script = /data1/luyifei/drla/drla/scripts/train_cola_joint_request_select_policy.py
mode = --checkpoint-selection-mode valid_rate_frontier
request rates = 0.10,0.25,0.50,0.75

strict frontier runs:
  seed29 = p2e_joint_request_select_score_frontier_full_seed20260529_fuserseed20260529_20260529
  seed30 = p2e_joint_request_select_score_frontier_full_seed20260530_fuserseed20260530_20260529
  seed31 = p2e_joint_request_select_score_frontier_full_seed20260531_fuserseed20260531_20260529

aggregate = /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
            p2e_joint_request_select_frontier_strict_seed29_30_31_summary_20260531
```

Canonical frontier readout, using valid-selected policy rows only:

```text
old target025 strict aggregate = 22.33%, score 0.3652
valid-frontier canonical = 21.46%, score 0.3505
request rate / budget = 49.73% / 1.99
score delta vs old joint = -0.0147
```

Interpretation: valid-frontier checkpoint selection is a completed negative
audit. It is useful evidence that the checkpoint-selection issue is real, but
it does not improve over the current canonical target025 strict aggregate. Do
not use test-best frontier rows as paper numbers.

P2 script policy:

```text
train scripts:
  CUDA/GPU required
  SwanLab cloud required
  write metrics.jsonl, best_checkpoint.pt, last_checkpoint.pt

eval/audit/packet-build scripts:
  swanlab_mode=disabled
  no optimizer/backward
  write local summary.json / metrics.jsonl / audit CSVs
```

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
