# P2 Agent-Agent Latent Communication 实施文档

更新日期：2026-06-01

> 状态：P2 latent communication substrate/method canonical。当前最高优先级执行顺序
> 见 `/data1/luyifei/drla/docs/current/P2_Locked_Complete_Execution_Scheme_2026-06-01.md`。
> 瘦身前完整版本见 `/data1/luyifei/drla/docs/full_history/2026-05-29_pre_slim/P2_Latent_MAS_Communication_Implementation_Plan_2026-05-29.md`。
>
> 2026-06-01 路线修订：后续 P2 主实验必须同时遵守 `/data1/luyifei/drla/docs/current/P2_Benchmark_and_Agent_Baseline_Redesign_2026-06-01.md`。旧 `official8 + solver-to-solver + message_only` 结果保留为 channel diagnostic，不再作为真实 MAS 或 latent > text 主证据。
>
> 2026-06-01 post-Family1 执行锁定：继续实验前必须先读
> `/data1/luyifei/drla/docs/current/P2_Locked_Complete_Execution_Scheme_2026-06-01.md`
> 和 `/data1/luyifei/drla/docs/current/P2_Post_Family1_Complete_Execution_Plan_2026-06-01.md`。
> P2-D4 后的 Branch B Family 1 已执行并停止，当前 admitted_tasks=[]。不能直接跑
> held-out、latent-vs-text 主表或 fuser training；下一步锁定为 Branch C 验证 true
> MAS benchmark/protocol，然后再回到 CoLA substrate/interface adaptation。
>
> 2026-06-01 anti-disturbance update：本文记录 substrate/method 历史与可复用
> 脚本，但执行顺序以 locked scheme 为准。后续主线不是继续 official8/GSM8K
> 单问答或旧 solver relay，而是先用 capable TextMAS 验证 true MAS benchmark。
> TextMAS/LatentMAS 的共同边界是 Agent A 输出作为 Agent B 输入，scorer 只看
> Agent B handoff 后生成的 final answer；direct decoded answer、replay-visible
> text 和 legacy all-visible 表只能作为 diagnostic。

## 0.1 2026-06-01 Benchmark / Agent Baseline Reset

Current decision:

```text
P2-D0:
  official8 solver-to-solver message_only receiver-only result is frozen as
  channel diagnostic.

P2 main:
  role-conditioned MAS with capability-gated benchmarks.
```

Canonical next protocol:

```text
Planner -> Critic -> Refiner -> Solver

Each downstream role receives:
  original question q
  role instruction
  previous role text message or latent working memory

Scorer sees:
  final Solver output only
```

This is not leakage. Leakage means scorer or online receiver sees gold answers,
scorer outputs, selected_prediction, or Agent-A decoded replay tokens as final
answer.

Benchmark rules:

```text
official8:
  retained for CoLA substrate diagnostics, P1 continuity, packet audit, and
  channel smoke/control tests.

new main benchmarks:
  must pass Single CoLA Solver and Role TextMAS capability gates before being
  used for paper-level P2 claims.

candidate tasks:
  ARC-E/C, OpenBookQA/MMLU-style MCQ, GPQA/MedQA MCQ, GSM8K short answer.

high-risk tasks:
  AIME24/25 and MBPP+/HumanEval+ require separate capability gates because
  CoLA may be floor-limited or format-limited.

true MAS tasks:
  evidence-split multi-hop QA and role-separated code workflows are preferred
  for the strongest agent-to-agent claim.
```

P2-D1 executable gate, 2026-06-01:

```text
prepare script:
  /data1/luyifei/drla/drla/scripts/prepare_cola_p2_candidate_benchmarks.py

eval script:
  /data1/luyifei/drla/drla/scripts/run_cola_p2_capability_gate.py

full prepared candidate data:
  /data1/luyifei/drla/outputs/p2_capability_gate/data_20260601

smoke eval:
  /data1/luyifei/drla/outputs/p2_capability_gate/eval_smoke_arc_easy_both_20260601
```

Current prepared tasks:

```text
ARC-Easy validation: 570
ARC-Challenge validation: 299
GSM8K test: 1319
MBPP+ test: 378
HumanEval+ test: 164
GPQA-Diamond test: 198
MedQA test: 1273
```

Pinned sources include `hendrydong/gpqa_diamond_mc` for GPQA-Diamond and
`GBaker/MedQA-USMLE-4-options` for MedQA.  Changing benchmark sources requires
a new manifest and should not be done silently.

Formal full gate, 2026-06-01:

```text
/data1/luyifei/drla/outputs/p2_capability_gate/aggregate_full_20260601

admitted_tasks = []
```

The current candidate tasks are blocked by CoLA base capability and prompt
protocol, not by a proven latent communication failure.  Do not run P2 main
text-vs-latent tables on these tasks until a calibration/held-out protocol
repair pass succeeds.

P2-D2 locked split, 2026-06-01:

```text
/data1/luyifei/drla/outputs/p2_capability_gate/locked_splits_seed20260602_20260601

calibration = 842
heldout = 3359
overlap = 0
split_seed = 20260602
```

P2-D3 prompt/protocol repair may inspect calibration rows only; held-out is
reserved for P2-D4 locked gate.

P2-D3 initial prompt repair, 2026-06-01:

```text
cola_fewshot_v1 does not improve the candidate set.
generic_v1 GPQA-Diamond passes single-mode calibration only.
GPQA-Diamond Role TextMAS still fails.
admitted_tasks remains [].
```

Locked next phase:

```text
P2-D3.1:
  completed; answer_state_v1, answer_state_structured_v1, and
  role_plan_ignore_v1 did not admit any calibration task.

P2-D3.2:
  failure taxonomy and first local protocol/literature review completed; choose
  substrate adaptation or benchmark redesign before another main branch.

P2-D4:
  rerun held-out capability gate only after calibration has a dual-pass task.

P2-E:
  run channel-correct TextMAS vs LatentMAS only on held-out admitted tasks.

P2-F:
  train fuser/adapter only if no-fuser latent has usable signal but fails text
  competitiveness or corrupt-control robustness.
```

Latent handoff rules:

```text
context_plus_thought or full_working_memory is canonical.
thought_only is an ablation.
no-fuser training-free replay must be tested before training a fuser.
lightweight fuser/gate is triggered only if matched latent fails robust
corrupted-control gates or remains worse than text under matched budgets.
```

## 0. 核心定义

P2 是 communication-channel substitution in a controlled MAS protocol，不是 AutoGen/MetaGPT 这类完整 multi-agent framework 设计。2026-06-01 后，channel substitution 必须放在 role-conditioned MAS baseline 中评估，不能再只依赖 solver-to-solver message-only 压力测试。

```text
Agent A -> message channel -> Agent B
```

当前 LLM agents 通常通过 text tokens 通信：

```text
Agent A internal state -> decode to text tokens -> Agent B encodes text -> Agent B responds
```

P2 替换为 Cola latent packet：

```text
Agent A internal state -> Cola latent packet -> Agent B consumes latent -> Agent B responds
```

Agent B 收到 message 后可以验证、总结、继续推理、调用工具或聚合；这些是 receiver-side behavior，不定义 communication 本身。

中心假设：

```text
H1 readability / validity:
  Agent B 能读取并使用 Agent A 的 latent packet。

H2 communication advantage:
  latent packet 相比 text-token handoff 更高效、更少损失，或 downstream utility 更好。
```

## 1. 研究边界

P2 的最小研究对象仍是 communication event：

```text
two or more agents
one agent emits a message
another agent receives it
the receiver conditions its next behavior on that message
```

但主实验必须把该 communication event 嵌入明确角色协议，例如 `Planner -> Critic -> Refiner -> Solver`。P2 不研究 AutoGen / MetaGPT / ChatDev / AgentVerse 的完整 orchestration framework；它们只提供可能的实验外壳。

主 claim 限定为：

```text
same-substrate Cola A -> Cola B latent communication
```

Text 是公共接口；latent 不是。Latent readability 依赖：

```text
model architecture
tokenizer and prompt template
VAE/DLM config and scaling
block position and prefix state
context hash / prompt hash
receiver-side consumption mechanism
training distribution
```

Claim 分层：

| Level | Setting | P2 地位 |
|---|---|---|
| Level 1 | same-substrate Cola A -> Cola B | 主范围 |
| Level 2 | same-family / near-homogeneous agents | 扩展，需要 calibration/adapters |
| Level 3 | heterogeneous agents | 后续工作，需要 translator/shared codec/KV alignment |

不要 claim heterogeneous-agent latent communication，除非实现并评估 adapter。

## 2. 当前起点

P1 locked result：

```text
summary:
  /data1/luyifei/drla/outputs/cola_experiment_summaries/
  official8_full_b64_bs12_p1_locked_seed66_67_68_split20260601_riskcert_20260527/summary.json

repeated decisions:
  14940

selected / fixed-final / prediction-stability accuracy:
  20.930% / 20.950% / 20.957%

avg selected blocks:
  1.834 / 4

losses vs final / prediction-stability:
  3 / 4

risk certificate:
  21 / 24 folds satisfied
```

P2 packet v1：

```text
script:
  /data1/luyifei/drla/drla/scripts/build_cola_agent_latent_comm_packets.py

packets / latent refs / missing / forbidden:
  14940 / 27399 / 0 / 0
```

v1 proves sanitized packet construction. It does not prove B readability, downstream utility, text-channel superiority, or hetero-agent communication.

## 3. Design Principles

Agent message：

```text
message = explicit envelope + latent cognitive payload
```

Envelope fields：

```text
sender, receiver, task_id, role, phase, handoff_type,
block_idx, model_id, config_digest, prompt_hash,
risk_certificate, payload_type
```

Latent payload：

```text
z_pre or deterministic prefix contract
z_1 ... z_t
process features
optional receiver/fuser features
```

P2 only replaces agent-to-agent cognitive message payload. Tool calls、人类输出、代码、PRD、检索文档仍可使用 text / structured artifacts。

Initial communication event：

```text
Agent A reasons to block t.
Agent A emits one latent packet.
Agent B consumes the packet.
Agent B continues / aggregates / verifies.
```

不先做 streaming。request-more 可以后续扩展。

Mandatory controls：

```text
matched latent
metadata-only
shuffled latent
cross-task latent
wrong-block latent
Gaussian-noised latent
rotated latent
text handoff baseline
```

Evidence ladder：

| Level | Question | Required evidence | Allowed claim |
|---|---|---|---|
| E0 | packet clean/loadable? | refs load, forbidden absent | packet construction works |
| E1 | distribution-compatible? | stats align with native traces | same-substrate handoff is specified |
| E2 | B uses payload? | matched beats metadata/corrupted | B can read/use packet |
| E3 | downstream useful? | matched beats no-message/corrupted | useful task information |
| E4 | better than text? | cost-quality vs text baseline | efficient/less lossy/Pareto |
| E5 | works in envelopes? | sequential/hierarchical controls pass | useful beyond single handoff |

## 4. Packet v2

Goal：把 v1 升级为 distribution-aware channel-substitution message。

Minimal top-level fields：

```text
protocol_version = cola_agent_latent_comm_v2
sample_key
task
communication_boundary
prefix_contract
agent_a
latent_memory
readiness_state
risk_certificate
agent_b_contract
audit_refs
```

`communication_boundary`：

```text
pattern:
  single_handoff | sequential_chain | hierarchical_aggregation
handoff_mode:
  one_shot
sender_role:
  solver
receiver_role:
  solver | reviewer | aggregator | verifier
phase:
  reasoning | review | aggregation | verification
```

`prefix_contract`：

```text
mode:
  shared_context_reencode | prefix_latent_ref | kv_cache_ref
input_context_hash
sender_prompt_hash
receiver_prompt_hash
config_digest
model_id / tokenizer_id / vae_id / dit_id
block_size / patch_size / latent_dim / latent_scaling / max_block_budget
```

First implementation mode：

```text
shared_context_reencode:
  B re-encodes task context and receiver prompt deterministically.
```

Builder update：

```text
/data1/luyifei/drla/drla/scripts/build_cola_agent_latent_comm_packets.py
```

New flags：

```bash
--protocol-version cola_agent_latent_comm_v2
--communication-boundary single_handoff
--sender-role solver
--receiver-role solver
--prefix-contract shared_context_reencode
--consume-mode replay_latent_blocks
```

Pass criteria：

```text
14940 packets
0 missing latent refs
0 forbidden key hits
100% communication_boundary / prefix_contract / agent_b_contract coverage
packet_schema.json documents v2 fields
```

Completed locked rebuild：

```text
output:
  /data1/luyifei/drla/outputs/cola_agent_latent_comm/
  p2_agent_latent_comm_v2_locked_seed66_67_68_split20260601_20260529

summary:
  /data1/luyifei/drla/outputs/cola_agent_latent_comm/
  p2_agent_latent_comm_v2_locked_seed66_67_68_split20260601_20260529/summary.json

protocol:
  cola_agent_latent_comm_v2

communication boundary:
  single_handoff

packets / latent refs / unique latent files:
  14940 / 27399 / 8850

missing latent refs / forbidden decoder-eval fields:
  0 / 0

v2 field coverage:
  communication_boundary = 14940 / 14940
  prefix_contract = 14940 / 14940
  agent_b_contract = 14940 / 14940
```

Interpretation：

```text
P2-A E0 is complete for same-substrate single-handoff packets.
This validates schema coverage, sanitization, and latent-ref existence.
It does not validate Agent B readability, matched-vs-corrupted separation,
or superiority over text-token handoff.
```

## 5. Distribution Audit

Script：

```text
audit_cola_agent_latent_packet_distribution.py
```

Inputs：

```text
--packets-jsonl
--output-dir
--num-control-samples
--control-types matched,metadata_only,shuffle,cross_task,wrong_block,noise,rotation
```

Checks：

```text
structural:
  block_count == selected_block
  latent_block_shape == [16, 16]
  selected_block <= max_block_budget
  config/task/seed consistency
  all refs loadable

distribution:
  latent norm/std/delta/cosine/drift
  block-position/task/seed conditional stats

controls:
  metadata_only, shuffle, cross_task, wrong_block, noise, rotation
```

Outputs：

```text
summary.json
distribution_stats.csv
control_stats.csv
ood_detection.csv
packet_examples.jsonl
```

Success：

```text
matched packets pass structural checks
matched stats align with native trace stats
corrupted controls are separable
no decoder/eval-only fields appear
```

Completed locked audit：

```text
output:
  /data1/luyifei/drla/outputs/cola_agent_latent_comm/
  p2_agent_latent_comm_v2_distribution_audit_locked_seed66_67_68_split20260601_20260529

summary:
  /data1/luyifei/drla/outputs/cola_agent_latent_comm/
  p2_agent_latent_comm_v2_distribution_audit_locked_seed66_67_68_split20260601_20260529/summary.json

status:
  pass

packets audited:
  14940

native aligned latent blocks:
  27399

structural errors / forbidden fields / latent load errors:
  0 / 0 / 0

native alignment max_abs_diff:
  0.0

controls:
  metadata_only, shuffle, cross_task, wrong_block, noise, rotation

pair-distance AUROC:
  min = 1.0

noted warning:
  1 shuffle control fell back to any-task same-block-count replacement.
```

Interpretation：

```text
P2-B E1 is complete for packet/tensor consistency and audit-time corrupted
control generation. Pair-distance separability is an audit sanity check only.
It does not prove receiver readability; P2-C must show matched latent improves
over metadata-only and corrupted controls under Agent B consumption.
```

## 6. Layer 1: Single-Handoff Receiver

Purpose：验证 B 是否使用 A 的 latent message。

Setting：

```text
Agent A: Cola sender
Agent B: latent receiver
Input: one v2 latent packet
Output: accept / defer
```

Offline targets：

```text
accept = selected packet does not lose vs final/prediction-stability
defer = selected packet is risky and should use final/fallback
```

Target feasibility audit：

```text
output:
  /data1/luyifei/drla/outputs/cola_agent_latent_comm/
  p2_latent_receiver_target_audit_locked_seed66_67_68_split20260601_20260529

packets:
  14940

offline accept / unsafe:
  14936 / 4

unsafe rate:
  0.0268%

naive all-accept accuracy:
  99.9732%
```

Implication：

```text
Do not use plain accept/defer BCE as the main P2-C proof on this locked set.
The label is too sparse to establish latent readability. Accept/defer remains
an auxiliary rare-event/risk audit unless richer boundary examples are built.

The main P2-C readability objective must be balanced enough to test whether
Agent B uses the latent payload, e.g. matched-vs-corrupted or context-payload
compatibility, while keeping decoder/gold/scorer fields out of online inputs.
```

Implemented P2-C v1 objective：

```text
balanced compatibility classification

positive:
  matched packet latent payload

negative:
  metadata_only, shuffle, cross_task, wrong_block, noise, rotation

model never receives:
  control_type
  decoded text / token ids
  gold answers / official scorer outputs
  selected/final/prediction-stability correctness labels
```

Scripts：

```text
train_cola_latent_receiver.py
eval_cola_latent_receiver.py
aggregate_cola_latent_receiver.py
```

Implemented：

```text
train:
  /data1/luyifei/drla/drla/scripts/train_cola_latent_receiver.py

aggregate:
  /data1/luyifei/drla/drla/scripts/aggregate_cola_latent_receiver.py
```

Ablations：

```text
envelope_only
process_only
certificate_only
latent_only
latent_process
latent_process_certificate
latent_process_certificate_no_task
```

Success：

```text
matched latent + certificate > certificate_only
matched latent + certificate > shuffled/cross_task/noised controls
calibrated loss risk comparable to P1 locked result
decoder-free online
```

Completed P2-C v1 ablation：

```text
aggregate:
  /data1/luyifei/drla/outputs/cola_latent_receiver/
  p2c_receiver_compat_bestckpt_eval_aggregate_seed20260529_20260529/summary.json

full train protocol:
  CUDA/GPU
  SwanLab cloud
  valid_interval = 50
  best_checkpoint.pt and last_checkpoint.pt saved for every run
```

| input_mode | SwanLab | test mean AUROC | shuffle AUROC |
|---|---:|---:|---:|
| envelope_only | poysm6qvybl7nipzqpzfj | 0.4999 | 0.4999 |
| process_only | 8nchledz4d108uka1m9zl | 0.5000 | 0.5000 |
| certificate_only | j5imgo3417t9ieqz4ngbm | 0.5000 | 0.5000 |
| latent_only | q3c18vawupb3lbtdai29n | 0.8963 | 0.5040 |
| latent_process | b8tek4rtj4vvph3q8z4r4 | 0.9115 | 0.5687 |
| latent_process_certificate | 50bumvfh2pgp0olw60jvr | 0.9194 | 0.6205 |
| latent_process_certificate_no_task | co6r6l6qw4lzxnu4mxkn2 | 0.9127 | 0.6511 |

Interpretation：

```text
P2-C v1 supports receiver-side latent payload readability under controlled
compatibility diagnostics: latent-bearing modes beat envelope/process/cert-only
negative controls, and certificate-only does not leak the control label.

The evidence is not yet downstream utility. Same-task shuffle remains the hard
control, so the next step must test continuation/utility under matched vs
corrupted packets rather than claiming text-channel superiority.
```

## 7. Layer 2: Sequential Latent Communication

Question：

```text
Can B continue reasoning from A's latent packet instead of A's text output?
```

Canonical text baseline：

```text
Agent 1 receives task + role prompt.
Agent 1 produces a raw text message from the same handoff trajectory/depth t.
Agent 2 receives task + receiver role prompt + Agent 1 raw text message.
Agent 2 produces final answer.
```

Canonical latent variant：

```text
Agent 1 generates z_1...z_t.
Agent 1 emits a distribution-complete v2 latent packet.
Agent 2 receives task + receiver role prompt + latent packet.
Agent 2 consumes the packet without using decoded message text as input.
Agent 2 continues Cola latent generation.
Final decoder produces answer.
```

Depths：

```text
t = 1, 2, P1-selected, prediction-stability diagnostic, final block
```

Consumption：

```text
canonical no-text latent consumption:
  B re-encodes receiver prompt/context.
  B loads A latent blocks and updates receiver-side Cola state/KV/cache.
  B does not sample/decode the latent message into intermediate text.
  B decodes only the final receiver output/action.

replay-based diagnostic:
  B re-encodes receiver prompt/context.
  B loads A latent blocks.
  B consumes/replays blocks, may decode block tokens to reconstruct context,
  and continues generation.

latent fuser later:
  used for accept/reject/request-more/verify policies after channel-correct
  single-handoff evaluation is fixed.
```

Runner：

```text
existing diagnostics:
  run_cola_sequential_latent_mas.py
  audit_cola_sequential_latent_mas.py
  build_cola_text_handoff_baseline.py

corrected channel-equivalent implementation:
  build_cola_agent_channel_messages.py
  run_cola_agent_b_channel_eval.py
  aggregate_cola_channel_eval.py
```

LatentMAS-aligned success gates：

```text
readability:
  B_latent_matched beats B_corrupted controls.

marginal utility:
  B_latent_matched beats B_none(empty input).

text competitiveness:
  B_latent_matched is competitive with or beats B_text_raw_message.

efficiency:
  quality is competitive at lower text-token/decode/re-encode/runtime/B-block cost.
```

Protocol correction, 2026-05-31：

```text
The historical text_selected/text_final/text_prediction_stability baselines are
direct decoded-answer handoff diagnostics. They do not feed text into Agent B
and therefore are not valid main baselines for the question:

  Does Agent B do better when receiving A's text message or A's latent packet?

They remain useful only as answer-state diagnostics.
```

LatentMAS-aligned channel-equivalent evaluation：

```text
For each same sample q and same Agent A handoff depth t:

  B_none(empty input) -> y_none
  B_text(A_raw_text_message_t) -> y_text
  B_latent(A_latent_packet_t) -> y_latent
  B_corrupt(corrupted_latent_packet_t) -> y_corrupt

The official scorer evaluates only y_none, y_text, y_latent, and y_corrupt.
Gold answers and scorer outputs never enter Agent A/B online inputs.

The canonical Agent-B input contract is message_only.  shared_context, where B
also sees the original benchmark prompt q, is allowed only as a diagnostic to
separate prompt/context effects from communication effects.
```

Text message construction：

```text
A_raw_text_message_t must come from the same A trajectory/depth as the latent
packet. It should be canonical replay/decode of z_1...z_t with prompt tokens
trimmed, or the native trace decode_text_so_far at the same selected block.

Do not use selected_prediction as A_text_message_t. selected_prediction is an
answer extracted by the task/scorer pipeline, not the raw agent-to-agent text
message.
```

Primary official8 scope：

```text
Current Cola/P2 artifacts are complete for official8 only. The first corrected
channel-equivalent evaluation should stay on official8 to avoid mixing protocol
repair with new-data plumbing. If B_none is too strong and message marginal
utility remains weak, then add reasoning-heavy datasets such as GSM8K/MATH with
new prompt/scorer/trace/packet support.
```

Corrected channel-equivalent smoke, 2026-05-31：

```text
scripts:
  build_cola_agent_channel_messages.py
  run_cola_agent_b_channel_eval.py
  aggregate_cola_channel_eval.py

message artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_channel_messages_smoke_1per_task_20260531

eval artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_smoke_1per_task_20260531

aggregate artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_smoke_1per_task_20260531/channel_eval_aggregate

scope:
  official8, 1 message per task, 8 channels, 64 final Agent-B generations
  local-only, swanlab_mode=disabled, CUDA generation, no training loop

online-input audit:
  A_text_message_t = native trace decode_text_so_far at the same selected block
  selected_prediction is not used as text
  gold/scorer outputs are not online Agent A/B inputs

official scorer smoke readout:
  latent_matched = 25.00% accuracy, mean score 0.5754
  B_text_raw_message = 25.00% accuracy, mean score 0.5799
  B_none = 25.00% accuracy, mean score 0.6059
  corrupted latent controls = 0.00% accuracy, mean score 0.1720-0.2005

decision-rule smoke:
  matched > all corrupted controls by score
  matched is score-competitive with raw text at tolerance 0.01
  matched does not beat B_none on this 8-sample smoke

interpretation:
  the corrected protocol is now executable. This smoke test is not a paper
  number; it only validates that the current route is no longer the historical
  selected_prediction direct-handoff route or shallow frontier tuning route.
```

Legacy all-visible official8 50/task unique-sample result, 2026-05-31：

```text
scripts:
  build_cola_agent_channel_messages.py --dedupe-sample-key
  run_cola_agent_b_channel_eval.py --message-start/--message-end
  merge_cola_agent_b_channel_eval_shards.py
  aggregate_cola_channel_eval.py --bootstrap-samples 2000

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
  official8, 50 unique sample_key per task
  400 messages, 8 channels, 3200 final Agent-B generations
  local-only, swanlab_mode=disabled, CUDA sharded generation, no training loop

protocol audit:
  A_text_message_t = native trace decode_text_so_far at the same selected block
  selected_prediction is not used as text
  gold/scorer outputs are not online Agent A/B inputs
  unique sample_key = 400 / 400
  merged duplicate keys = 0

critical caveat added after data-flow audit:
  this run used the historical all-visible scorer output.  In that mode, A raw
  text for the text channel and A decoded replay tokens for latent decode-and-
  emit could enter the final `generate` scored by the official scorer.  This
  means the table is valid only as a replay-output / decodability diagnostic.
  It must not be cited as proof of Agent-B communication or latent-vs-text
  channel superiority.
```

Official scorer summary：

```text
latent_matched:
  accuracy = 23.50%, mean_score = 0.4850

B_text_raw_message:
  accuracy = 23.00%, mean_score = 0.5011

B_none:
  accuracy = 19.75%, mean_score = 0.4410

corrupted latent controls:
  accuracy = 0.00% to 4.25%
  mean_score = 0.1699 to 0.2499
```

Paired latent_matched deltas：

```text
vs corrupted controls:
  score_delta = +0.2351 to +0.3151
  score_delta_ci95 lower bounds all > +0.1959
  accuracy_delta = +19.25pp to +23.50pp
  accuracy_delta_ci95 lower bounds all > +15.00pp

vs B_none:
  score_delta = +0.0440, CI95 [+0.0111, +0.0758]
  accuracy_delta = +3.75pp, CI95 [-0.25pp, +7.50pp]

vs B_text_raw_message:
  score_delta = -0.0161, CI95 [-0.0406, +0.0076]
  accuracy_delta = +0.50pp, CI95 [-2.50pp, +3.75pp]
```

Legacy decision-rule status：

```text
matched > corrupted:
  pass only as decodability/replay-output evidence.

matched > B_none:
  invalid as Agent-B communication evidence because replay output can be scored.

matched competitive with or beats B_text:
  invalid as Agent-B communication evidence because both message channels can
  leak A-message content into `generate`.

claim allowed:
  A's latent packet is decodable and task-relevant under Cola VAE replay.

claim not allowed yet:
  useful Agent-B communication.
  latent is better than raw text communication.
```

An earlier 50/task non-deduped channel eval found only 375 unique sample_key
among 400 messages. Treat it as a protocol audit that motivated
`--dedupe-sample-key`; do not cite it as the formal P2-D table.

Receiver-native cache-only audit, 2026-05-31：

```text
question:
  Is the current latent advantage coming from direct latent-state consumption,
  or from decoding/replaying A's latent blocks into text-like output?

cache-only channels:
  latent_matched_cache_only
  latent_shuffle_cache_only
  latent_cross_task_cache_only
  latent_wrong_block_cache_only
  latent_noise_cache_only
  latent_rotation_cache_only

protocol:
  replay latent blocks update VAE/DiT KV cache
  replay blocks are not sampled into output text
  replay_blocks_decoded_to_text = 0
  same 400 unique messages as the formal 50/task decode-and-emit run

artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_official8_50per_task_cache_only_seed20260531_unique_20260531_merged/channel_eval_aggregate
```

Cache-only result：

```text
latent_matched_cache_only:
  accuracy = 1.50%, mean_score = 0.1916

B_none:
  accuracy = 19.75%, mean_score = 0.4410

B_text_raw_message:
  accuracy = 23.00%, mean_score = 0.5011

corrupted cache-only controls:
  accuracy = 0.75% to 3.25%
  mean_score = 0.1559 to 0.2261
```

Cache-only paired deltas：

```text
vs B_none:
  score_delta = -0.2493, CI95 [-0.2851, -0.2152]
  accuracy_delta = -18.25pp, CI95 [-22.50pp, -14.25pp]

vs B_text_raw_message:
  score_delta = -0.3094, CI95 [-0.3484, -0.2686]
  accuracy_delta = -21.50pp, CI95 [-25.50pp, -17.00pp]

vs corrupted cache-only controls:
  matched does not dominate all controls; wrong_block_cache_only is stronger
  than matched by mean score.
```

Interpretation：

```text
Current P2-D establishes useful same-substrate latent communication only in the
decode-and-emit replay sense: the latent packet is a decodable payload that can
improve Agent-B final outputs over B_none and strongly beats corrupted payloads.

It does not establish receiver-native no-text latent reasoning. For the target
of agent-to-agent latent communication without decoder-mediated signals, the
next architecture should make Agent B consume latent state through a native
receiver/fuser/policy objective rather than only replaying A's latent trajectory
through the decoder path.
```

Important implementation clarification：

```text
The current Agent-B latent channel does not reconstruct A's latent packet by
encoding text.  `load_packet_blocks(...)` loads A latent blocks directly from
the packet latent shard.  During replay, each block z is passed to the Cola DiT
as `dit(txt=z, update_kv=True, use_kv_cache=True)`.

However, the earlier cache-only channel was not DiT-only.  It also called
`vae.decode(z, update_kv=True)` to keep the VAE decoder cache aligned, while
not sampling/emitting those replay tokens.  Thus the channel variants are:

decode_and_emit:
  z -> VAE decode/sample visible replay tokens
  z -> DiT KV cache
  decoded replay tokens are appended to `context_ids` / final `generate`
  decoded replay tokens are not re-encoded by Agent B's VAE encoder

cache_only:
  z -> VAE decoder KV cache, no visible replay tokens
  z -> DiT KV cache

dit_only_cache:
  z -> DiT KV cache only
  VAE decoder receives no replay cache and decodes only final receiver blocks

vae_only_cache:
  z -> VAE decoder KV cache only
  DiT receives no replay cache
```

Decoder semantic-projection gap audit, 2026-05-31：

```text
script:
  /data1/luyifei/drla/drla/scripts/audit_cola_channel_projection_gap.py

artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_projection_gap_official8_50per_task_seed20260531_unique_20260531

scope:
  same 400 unique messages as the formal official8 50/task result
  paired decode-and-emit latent_matched vs latent_matched_cache_only

result:
  decode_emit_mean_score = 0.4850
  cache_only_mean_score = 0.1916
  projection_score_gain = +0.2934, CI95 [+0.2551, +0.3320]
  decode_emit_accuracy = 23.50%
  cache_only_accuracy = 1.50%
  projection_accuracy_gain = +22.00pp, CI95 [+17.75pp, +26.25pp]
  same_prediction_rate = 8.25%
```

Direct DiT / VAE cache ablation smoke, 2026-05-31：

```text
script:
  run_cola_agent_b_channel_eval.py
  added channel suffixes:
    _dit_only_cache
    _vae_only_cache

artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_smoke_direct_dit_1per_task_20260531_v2/channel_eval_aggregate

scope:
  official8, 1 message per task, protocol smoke only

smoke scores:
  latent_matched = 25.00%, mean_score 0.5754
  latent_matched_cache_only = 0.00%, mean_score 0.2082
  latent_matched_dit_only_cache = 0.00%, mean_score 0.2082
  latent_matched_vae_only_cache = 0.00%, mean_score 0.3373
  latent_noise_dit_only_cache = 0.00%, mean_score 0.2427
  latent_noise_vae_only_cache = 0.00%, mean_score 0.3373

interpretation:
  This smoke confirms that direct DiT-only replay is executable and currently
  does not recover the decode-and-emit utility on the tiny 8-sample probe.  It
  is not a formal number, but it rules out the explanation that the previous
  cache-only result failed merely because latent blocks were accidentally
  routed through a text encoder.
```

Data-flow caveat：

```text
`latent_matched` working should not be interpreted as:

  A latent -> decoder text -> Agent-B encoder -> Agent-B latent -> DiT

That is the `text` channel's structure, not the latent channel's structure.
The actual latent decode-and-emit structure is:

  A latent z -> VAE decoder logits/tokens -> scorer-visible final output text
  A latent z -> DiT KV cache

Only the final receiver-generated blocks are newly denoised by Agent B after
the replay boundary.  Therefore a large part of the decode-and-emit gain can
come from A's decoded replay tokens already containing the answer or a useful
answer prefix.  This is valid evidence that A's latent packet is decodable and
task-relevant, but it is not evidence that Agent B learned to internally
understand A latent without a decoder-mediated message.
```

Corrected receiver-only smoke, 2026-05-31：

```text
implementation change:
  run_cola_agent_b_channel_eval.py now defaults to:
    --score-output-scope receiver_only

receiver_only rule:
  A text message can condition B's prompt/cache.
  A latent replay blocks can condition B's VAE/DiT cache according to channel.
  A text message tokens and A latent replay decoded tokens are excluded from
  final `generate`.
  The official scorer sees only Agent-B tokens generated after handoff.

legacy reproduction:
  --score-output-scope legacy_all_visible
  should be used only for historical debugging, not for communication claims.

artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_smoke_receiver_only_1per_task_20260531/channel_eval_aggregate

scope:
  official8, 1 message per task, protocol smoke only

leak audit:
  score_output_scope = receiver_only for all 48 generations
  sum(scorer_visible_text_message_tokens) = 0
  sum(scorer_visible_replay_blocks) = 0

smoke scores:
  none = 25.00%, mean_score 0.6059
  text = 12.50%, mean_score 0.2996
  latent_matched = 0.00%, mean_score 0.2082
  latent_matched_cache_only = 0.00%, mean_score 0.2082
  latent_matched_dit_only_cache = 0.00%, mean_score 0.2082
  latent_matched_vae_only_cache = 0.00%, mean_score 0.3373

interpretation:
  Once A's message/replay tokens are not scored directly, the previous
  decode-and-emit advantage disappears on the smoke.  This confirms that the
  old `latent_matched` score mostly measured decodable replay output rather
  than Agent-B's post-handoff answer generation.  A formal corrected
  receiver-only 50/task run is required before making any P2-D communication
  claim.
```

LatentMAS-aligned handoff correction, 2026-05-31：

```text
reference:
  LatentMAS builds sequential / hierarchical MAS where agent outputs are passed
  to the next agent.  In text MAS this is text output handoff; in latent MAS
  the preceding agent's latent working memory / KV cache is transferred, and
  only the final agent decodes the answer.

implication for this project:
  The canonical Agent-A -> Agent-B protocol should be message_only:

    B_none(empty input)
    B_text(A_raw_text_message_t)
    B_latent(A_latent_packet_t)

  `shared_context`, where B also receives the original benchmark prompt, is a
  diagnostic/control setting rather than the main communication protocol.
```

Implementation status：

```text
run_cola_agent_b_channel_eval.py now supports:
  --agent-b-input-contract message_only
  --agent-b-input-contract shared_context

default:
  message_only
```

Message-only + receiver-only smoke after replay-EOS fix：

```text
artifact:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_smoke_message_only_receiver_only_eosfix_1per_task_20260531/channel_eval_aggregate

protocol:
  agent_b_input_contract = message_only
  score_output_scope = receiver_only

code fix:
  In receiver_only mode, EOS/im_end decoded from Agent A replay is recorded as
  replay_stop_token_seen but no longer stops Agent B generation. Only stop
  tokens generated by Agent B itself stop receiver generation.

boundary audit:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_protocol_boundary_audit_message_only_eosfix_smoke_20260531

audit result:
  pass
  scorer-visible A text/replay = 0

smoke scores:
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
  protocol smoke only; the formal official8 50/task result below supersedes it
  for channel-quality discussion.
```

Formal official8 50/task message_only + receiver_only result after replay-EOS fix：

```text
generation:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_official8_50per_task_message_only_receiver_only_eosfix_seed20260531_unique_20260531_merged

aggregate:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_agent_b_channel_eval_official8_50per_task_message_only_receiver_only_eosfix_seed20260531_unique_20260531_merged/channel_eval_aggregate

protocol audit:
  /data1/luyifei/drla/outputs/cola_agent_channel_eval/
  p2d_protocol_boundary_audit_message_only_receiver_only_eosfix_50task_20260531

scope:
  official8, 50 unique sample_key per task
  400 messages, 11 channels, 4400 Agent-B generations
  merge duplicate_keys = 0
  missing_message_rows = 0

boundary:
  agent_b_input_contract = message_only
  score_output_scope = receiver_only
  scorer_visible_text_message_tokens = 0
  scorer_visible_replay_blocks = 0
  audit status = pass
```

Official scorer summary：

| channel | accuracy | mean score |
|---|---:|---:|
| `latent_matched` | 1.75% | 0.1874 |
| `latent_matched_cache_only` | 1.75% | 0.1912 |
| `latent_matched_dit_only_cache` | 1.25% | 0.1873 |
| `latent_matched_vae_only_cache` | 0.00% | 0.1395 |
| `none` | 0.00% | 0.1378 |
| `text` | 1.25% | 0.1795 |
| `latent_cross_task` | 0.25% | 0.1535 |
| `latent_shuffle` | 0.25% | 0.1663 |
| `latent_wrong_block` | 4.00% | 0.1950 |
| `latent_noise` | 0.00% | 0.1390 |
| `latent_rotation` | 0.00% | 0.1297 |

Paired decision readout：

```text
latent_matched - none:
  score_delta = +0.0496, CI95 [+0.0340, +0.0659]
  accuracy_delta = +1.75 pp, CI95 [+0.50, +3.00]

latent_matched - text:
  score_delta = +0.0079, CI95 [-0.0082, +0.0259]
  accuracy_delta = +0.50 pp, CI95 [-1.00, +2.00]

latent_matched - latent_wrong_block:
  score_delta = -0.0076, CI95 [-0.0286, +0.0140]
  accuracy_delta = -2.25 pp, CI95 [-4.50, -0.25]

latent_matched - latent_matched_cache_only:
  score_delta = -0.0038, CI95 [-0.0064, -0.0012]
  accuracy_delta = 0.00 pp
```

Interpretation：

```text
The corrected LatentMAS-aligned protocol now gives valid Agent-B communication
evidence, but the claim is narrow:

  established:
    matched latent has marginal utility over empty input under receiver_only.
    matched latent is score-competitive with raw text message.
    DiT/cache-side latent consumption is not dead; cache-only slightly exceeds
    decode-and-emit in mean score while scoring only B output.

  not established:
    latent does not significantly beat text.
    matched does not beat all corrupted controls because wrong_block is
    anomalously strong and has higher accuracy/score.
    therefore no paper-level "latent communication beats text" claim yet.

The next design question is why wrong_block and cache-only are strong: possible
causes include block-depth prior, task/template bias, receiver promptless
generation artifacts, or control construction that is not hard enough in the
intended direction.  The next experiment should audit wrong_block source
distribution and add stricter matched-depth / same-task / same-budget controls
before claiming payload-specific latent understanding.
```

Literature-design implication：

```text
Coconut feeds hidden states back as continuous thoughts only under an explicit
latent-mode training curriculum.  CODI uses feature-level self-distillation to
align continuous thoughts with an answer-generating hidden state.  CoLaR trains
a latent head / next-compressed-embedding objective and treats dynamic
termination as part of the model.  Multi-agent continuous communication work
such as CommNet learns the communication interface by backpropagation.

Therefore, the current negative receiver-native/cache-only result should be
treated as missing learned receiver-interface/alignment supervision, not as
evidence that A's latent packet has no information.
```

Early replay diagnostic, 2026-05-29：

```text
eval root:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_eval_official8_5per_task_controls_20260529

scope:
  official 8 tasks
  5 packets per task
  40 packets total
  controls = matched, metadata_only, shuffle, cross_task, wrong_block, noise, rotation
  local-only, swanlab disabled, no training

runner status:
  pass
  no control generation warnings
  all controls produce nonempty outputs
```

Official scorer result：

| control | scorer accuracy |
|---|---:|
| matched | 17.5% |
| metadata_only | 20.0% |
| wrong_block | 5.0% |
| shuffle | 0.0% |
| cross_task | 0.0% |
| noise | 0.0% |
| rotation | 0.0% |

Offline fidelity audit：

| control | selected/final/prediction-stability answer-prefix agreement |
|---|---:|
| matched | 37.5% |
| metadata_only | 27.5% |
| noise | 7.5% |
| shuffle | 2.5% |
| cross_task | 2.5% |
| wrong_block | 2.5% |
| rotation | 2.5% |

Early interpretation：

```text
P2-D replay path is executable and matched latent changes receiver generation.
Matched replay carries more native-answer signal than metadata-only and
corrupted latent controls under offline answer-prefix fidelity.

However, matched does not yet beat metadata_only on downstream scorer accuracy
on this 40-sample subset. Therefore this is evidence for replay readability /
fidelity signal, not evidence for communication advantage or text superiority.

Next P2-D work should separate two failure modes:
  1. replay-consumption mismatch versus native Cola continuation;
  2. weak downstream utility of P1-selected latent handoff.
```

Direct-answer handoff diagnostic on the same 40 samples：

```text
artifact:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_eval_official8_5per_task_controls_20260529/text_handoff_baseline

diagnostic construction:
  text_selected = P1 selected_prediction direct handoff
  text_final = full-budget final_prediction direct handoff
  text_prediction_stability = prediction_stability_prediction direct handoff

official scorer:
  text_selected = 30.0%
  text_final = 30.0%
  text_prediction_stability = 30.0%
```

Replay-only mismatch diagnostic：

```text
artifact:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_only_eval_official8_5per_task_controls_20260529

setting:
  receiver_budget_mode = fixed
  fixed_receiver_blocks = 0

matched scorer accuracy:
  17.5%

matched answer-prefix fidelity:
  37.5%

raw trace audit:
  matched replay-only vs native trace selected-block raw text = 60.0%
  native trace selected-block raw text vs P1 selected_prediction = 27.5%
```

Interpretation update：

```text
The score gap is already present in replay-only mode, so the main current
failure is not B's additional continuation. Matched replay partially recovers
the native raw selected-block trace, but the raw trace itself is not the same
object as P1 selected_prediction after task scorer / answer extraction.

The current direct-answer diagnostic is a clean answer handoff. It is not an
Agent-B text-message baseline. The current latent diagnostic is a raw latent
continuation/replay handoff. This is useful evidence, but it should not be
reported as an apples-to-apples text-vs-latent communication comparison until
B_text(A_raw_text_message_t) and B_latent(A_latent_packet_t) are evaluated under
the same message_only receiver budget and receiver_only scorer.

The next diagnostic should explicitly compare:
  raw native trace decode_text_so_far at selected block
  matched replay-only text
  P1 selected_prediction after task scorer / answer extraction
  B_text final output from raw A text message
  B_latent final output from no-text latent consumption

Until this is resolved, do not claim latent beats text. The next design is not
more blind replay scaling. It is a channel-equivalent Agent-B evaluation plus a
receiver-native no-text latent consumption path for accept/reject/request-more/
continue/verify actions.
```

Expanded P2-D diagnostic, 2026-05-29：

```text
replay-only artifact:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_only_eval_official8_20per_task_controls_20260529

sequential continuation artifact:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_continue_eval_official8_20per_task_controls_20260529

text baseline artifact:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_only_eval_official8_20per_task_controls_20260529/text_handoff_baseline

scope:
  official 8 tasks
  20 packets per task
  160 packets total
  controls = matched, metadata_only, shuffle, cross_task, wrong_block, noise, rotation
  local-only, swanlab disabled, no training
```

Official scorer result：

| setting | matched | metadata_only | shuffle | cross_task | wrong_block | noise | rotation | text_selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| replay-only | 24.38% | 0.00% | 1.88% | 0.62% | 0.62% | 0.00% | 0.00% | 25.00% |
| replay + continue | 24.38% | 21.88% | 1.88% | 0.62% | 0.62% | 0.00% | 0.00% | 25.00% |

Expanded official-extractor audit：

```text
selected_reference_accuracy:
  25.0%

matched replay-only official_prediction_agrees_selected:
  65.0%

matched replay-only correct_selected_preservation_rate:
  80.0%

matched replay-only incorrect_selected_prediction_reproduction_rate:
  60.8%

matched replay-only vs native trace selected-block raw text:
  56.9%

native trace official prediction agreement with selected_prediction:
  45.6%
```

Expanded interpretation：

```text
The 20-per-task diagnostic upgrades P2-D from "executable smoke" to positive
same-substrate channel evidence on a small but nontrivial subset:

  matched latent replay is near text_selected (24.38% vs 25.00%);
  matched latent replay is far above corrupted latent controls;
  continuation preserves the matched replay score rather than degrading it;
  matched+continue modestly beats metadata_only+continue (24.38% vs 21.88%).

This supports E2/E3-style claims on the diagnostic subset: B can read/use the
matched latent payload and the payload carries useful task information beyond
corrupted controls. It does not support E4 text-channel superiority: text_selected
is still slightly higher and the matched-vs-metadata gain is small.

Next work should validate the result at larger paired scale and across fresh
packet samples/seeds, then decide whether a learned state reader/fuser is needed
for the remaining gap. Do not jump to hierarchical communication until this
larger-scale P2-D evidence is stable.
```

Fresh 50-per-task P2-D validation, 2026-05-29：

```text
replay-only artifact:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_only_eval_official8_50per_task_seed20260530_controls_20260529

replay+continue artifact:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_replay_continue_eval_official8_50per_task_seed20260530_controls_20260529

text baselines:
  under each artifact's text_handoff_baseline/

scope:
  official 8 tasks
  fresh selection seed = 20260530
  50 packets per task
  400 packets total
  controls = matched, metadata_only, shuffle, cross_task, wrong_block, noise, rotation
  local-only, swanlab disabled, no training
```

Official scorer result：

| setting | matched | metadata_only | shuffle | cross_task | wrong_block | noise | rotation | text_selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| replay-only | 24.25% | 0.00% | 0.50% | 0.25% | 3.75% | 1.25% | 0.00% | 23.50% |
| replay + continue | 24.25% | 23.75% | 0.75% | 0.50% | 4.25% | 1.25% | 0.25% | 23.50% |

Duplicate-safe audit result：

```text
audit script fix:
  duplicate sample ids/sample_keys exist in the 50-per-task subset
  audit now keys scorer rows by sample_key + generate and groups paired rows by packet_index

matched replay-only:
  accuracy = 24.25%
  official_prediction_agrees_selected = 63.5%
  correct_selected_preservation_rate = 75.5%
  native_trace_selected_agreement_rate = 51.25%
  paired net wins vs metadata_only / wrong_block / shuffle / cross_task / noise / rotation:
    +97 / +82 / +95 / +96 / +92 / +97

matched replay+continue:
  accuracy = 24.25%
  official_prediction_agrees_selected = 63.25%
  correct_selected_preservation_rate = 75.5%
  native_trace_selected_agreement_rate = 50.5%
  paired net wins vs metadata_only / wrong_block / shuffle / cross_task / noise / rotation:
    +2 / +80 / +94 / +95 / +92 / +96
```

Fresh-scale interpretation：

```text
The 50-per-task fresh validation stabilizes the 20-per-task result:
matched latent replay remains competitive with text_selected and far above
corrupted latent controls. This supports E2/E3 same-substrate claims: the
matched Cola latent packet is readable/useful to another Cola instance, and
the signal is not explained by arbitrary latent payloads.

The result does not support E4 text-channel superiority. In replay-only,
matched is only +0.75 percentage points above text_selected. In replay+continue,
metadata_only also reaches 23.75%, only 0.50 points below matched, because
the receiver can still solve from the shared context when allowed to spend the
remaining block budget. Therefore, matched-vs-corrupted is the strong evidence;
matched-vs-text and matched-vs-metadata remain weak.

Next P2 work should either:
  1. define a stricter marginal-utility receiver protocol where context-only
     metadata cannot solve the task nearly as well, or
  2. move to hierarchical / aggregation envelopes while preserving metadata and
     corrupted controls, and avoid claiming text superiority until a real
     cost-quality Pareto advantage appears.
```

Message-only marginal-utility diagnostic, 2026-05-29：

```text
artifact:
  /data1/luyifei/drla/outputs/cola_sequential_latent_mas/
  p2d_message_only_replay_only_official8_50per_task_seed20260531_controls_20260529

protocol:
  receiver_context_mode = empty_prompt
  receiver_budget_mode = fixed
  fixed_receiver_blocks = 0

scope:
  official 8 tasks
  fresh selection seed = 20260531
  50 packets per task
  400 packets total
  controls = matched, metadata_only, shuffle, cross_task, wrong_block, noise, rotation
  local-only, swanlab disabled, no training
```

This protocol removes B's shared-context re-encoding and decodes only the
message payload. It is a stricter marginal-utility diagnostic, not the default
P2 contract. It answers whether the latent packet itself carries recoverable
answer/task signal.

Official scorer result：

| setting | matched latent | metadata_only | shuffle | cross_task | wrong_block | noise | rotation | text_selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| message-only replay | 12.50% | 0.00% | 1.75% | 0.25% | 2.00% | 0.00% | 0.25% | 26.25% |

Audit result：

```text
matched:
  official_prediction_agrees_selected = 16.25%
  correct_selected_preservation_rate = 19.05%
  native_trace_selected_agreement_rate = 7.5%
  paired net wins vs metadata_only / wrong_block / shuffle / cross_task / noise / rotation:
    +50 / +42 / +43 / +49 / +50 / +49
```

Interpretation：

```text
Message-only matched latent is clearly above metadata_only and corrupted latent
controls, so the packet itself contains recoverable signal. However it is far
below the direct-answer diagnostic on the exact same 400 packets (12.50% vs
26.25%).

Therefore the current replay interface supports a conservative claim:
  latent packets are readable/useful under same-substrate controls.

It does not support:
  latent communication is better than text communication.

The gap is likely an interface/consumption problem rather than evidence that the
latent packet contains no information: full-prompt replay reaches 24.25%, while
message-only reaches 12.50%. The next design should expose a receiver-side
state reader/fuser or an aggregation task where text and latent are compared
under a real cost-quality objective.
```

Receiver-side latent answer reader diagnostic, 2026-05-29：

```text
training artifact:
  /data1/luyifei/drla/outputs/cola_latent_answer_reader/
  p2_answer_reader_full_seed20260529_20260529

best-checkpoint eval artifacts:
  valid:
    /data1/luyifei/drla/outputs/cola_latent_answer_reader/
    p2_answer_reader_full_seed20260529_20260529_best_eval_valid
  test:
    /data1/luyifei/drla/outputs/cola_latent_answer_reader/
    p2_answer_reader_full_seed20260529_20260529_best_eval_test

training:
  CUDA/GPU training
  SwanLab cloud run = x6yc77eedf77z27ego0ve
  epochs = 12
  batch_size = 256
  valid_interval = 50
  d_model = 128
  inter_layers = 2
  best_checkpoint.pt selected by valid official_top1_accuracy
```

This diagnostic trains a lightweight latent-to-answer-state reader rather than
using the official decoder online. It encodes sanitized packet latents plus
process features, contrasts them against answer-text byte embeddings, retrieves
the nearest P1 teacher selected_prediction state, and then scores the retrieved
text with the official Cola scorer. The teacher text is a training/eval label
space, not an online receiver input.

Best-checkpoint metrics：

| split | answer_key_top1 | official_top1_accuracy | selected_reference_accuracy | gap vs selected |
|---|---:|---:|---:|---:|
| valid | 13.54% | 11.11% | 21.48% | -10.37 pp |
| test | 15.20% | 10.61% | 22.11% | -11.50 pp |

Interpretation：

```text
The answer reader learns non-trivial latent-to-answer associations, but it is
still far below the same-split text/teacher reference. It is also below the
message-only replay matched-latent scorer result on the 400-packet diagnostic
(10.61% best-checkpoint test vs 12.50% message-only replay).

Therefore it should be treated as a useful negative diagnostic:
  direct lightweight retrieval from latent packets is not yet enough.

It does not change the main P2-D conclusion:
  current latent communication is readable/useful vs corrupted controls, but is
  not better than ordinary text communication under the tested interfaces.
```

## 7.5 Layer 3: Hierarchical Aggregation

Motivation, 2026-05-29：

```text
P2-D shows that a single latent packet is readable but not text-superior.
The next question is whether multiple independent latent messages contain
complementary answer-state information that an aggregator can exploit.
```

This follows the same high-level lesson as self-consistency and multi-agent
debate: the value of multiple reasoning paths is normally realized by an
aggregation/selection step, not by inspecting one path in isolation. In P2 this
must be tested under strict packet controls, because text majority itself is a
strong baseline and decoded text must not become an online latent input.

P2-E potential audit：

```text
script:
  /data1/luyifei/drla/drla/scripts/audit_cola_hierarchical_aggregation_potential.py

artifact:
  /data1/luyifei/drla/outputs/cola_hierarchical_aggregation/
  p2e_aggregation_potential_locked_seed66_67_68_split20260601_20260529

protocol:
  local-only, swanlab disabled
  group the three locked seed66/67/68 sender packets for each same sample
  groups = 4980
  packets = 14940
```

Main results：

| method | accuracy |
|---|---:|
| single_sender_first | 20.74% |
| best simple latent-state ranker, prediction_change_min | 21.39% |
| text_majority_selected | 21.55% |
| oracle_any_selected_correct | 33.13% |
| oracle_any_final_correct | 33.15% |

Interpretation：

```text
There is real aggregation headroom: three senders contain at least one correct
selected answer for 33.13% of groups, far above a single fixed sender at 20.74%.
However naive text majority only reaches 21.55%, and simple readiness/risk
rankers only move slightly above the single-sender baseline.

Therefore P2-E is worth testing, but the online receiver needs a learned fuser
that can identify the useful sender/message. The oracle number is an upper
bound, not an allowed online method.
```

P2-E learned latent fuser v1：

```text
train script:
  /data1/luyifei/drla/drla/scripts/train_cola_hierarchical_latent_fuser.py

eval script:
  /data1/luyifei/drla/drla/scripts/eval_cola_hierarchical_latent_fuser.py

training artifact:
  /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
  p2e_hierarchical_fuser_full_seed20260529_20260529

best-checkpoint eval artifacts:
  valid:
    /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
    p2e_hierarchical_fuser_full_seed20260529_20260529_best_eval_valid
  test:
    /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
    p2e_hierarchical_fuser_full_seed20260529_20260529_best_eval_test

training:
  CUDA/GPU
  SwanLab cloud run = ljv0m43x48a49j1at6gx9
  epochs = 24
  batch_size = 256
  valid_interval = 50
  best_checkpoint.pt selected by valid model_selected_accuracy
```

Architecture：

```text
per sender:
  latent block slot transformer
  process-feature MLP
  readiness/risk certificate MLP
  task embedding

across senders:
  sender-position embedding
  sender-level Transformer
  sender selection head

online inputs:
  sanitized latent blocks
  process features
  readiness/certificate fields
  task id

forbidden online inputs:
  decoded selected_prediction
  official scorer result
  gold answer
```

Best-checkpoint eval：

| split | model selected | single sender | text majority | oracle any |
|---|---:|---:|---:|---:|
| valid | 23.03% | 23.23% | 22.42% | 33.33% |
| test | 20.74% | 22.38% | 23.41% | 34.70% |

Interpretation：

```text
The first learned hierarchical fuser is a negative result. It sees only
decoder-free latent/process/certificate fields, but it does not beat the
single fixed sender or the text-majority baseline on the held-out test split.
It does select a correct sender in 59.76% of the test groups where at least one
sender is correct, but this is still not enough to close the oracle gap.

This suggests that P2-E needs either a richer fuser objective, better
supervision than sparse selected_correct, or an aggregation task whose receiver
can use latent states directly instead of merely choosing which decoded answer
would have been correct.
```

P2-E supervision audit and score-target fuser v2, 2026-05-29：

```text
motivation:
  fuser v1 used sparse selected_correct labels.
  Official scorer values are often continuous partial utility, not only 0/1.

audit:
  sender predictions = 14940
  non-binary official scores = 7764 / 14940 = 52.0%
  groups whose best selected score is partial but not exact-correct:
    2570 / 4980 = 51.6%
```

This means the binary selected_correct objective discards much of the
supervision signal that should matter for communication utility.

Implemented update：

```text
train script:
  /data1/luyifei/drla/drla/scripts/train_cola_hierarchical_latent_fuser.py

new option:
  --target-mode score

target:
  selected official score in [0, 1]

loss:
  pointwise MSE(sigmoid(logit), selected_score)
  + listwise rank loss over sender selected_scores

online inputs:
  unchanged; still decoder-free latent/process/certificate fields only
```

Score-target v2 artifacts：

```text
training artifact:
  /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
  p2e_hierarchical_fuser_score_full_seed20260529_20260529

best-checkpoint eval artifacts:
  valid:
    /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
    p2e_hierarchical_fuser_score_full_seed20260529_20260529_best_eval_valid
  test:
    /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
    p2e_hierarchical_fuser_score_full_seed20260529_20260529_best_eval_test

training:
  CUDA/GPU
  SwanLab cloud run = o5fjvuiqk82nk9c5hihn0
  epochs = 24
  batch_size = 256
  valid_interval = 50
  best_checkpoint.pt selected by valid model_mean_official_score
```

Best-checkpoint eval：

| split | model acc | single acc | text majority acc | oracle any acc | model mean score | single mean score | text majority mean score | oracle best mean score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| valid | 22.63% | 23.23% | 22.42% | 33.33% | 0.3756 | 0.3711 | 0.3617 | 0.4904 |
| test | 23.41% | 22.38% | 23.41% | 34.70% | 0.3685 | 0.3553 | 0.3622 | 0.4951 |

Interpretation：

```text
Score-target v2 improves the P2-E picture but still does not justify a broad
"latent beats text" claim.

Positive:
  On held-out test, v2 beats single_sender_first in exact accuracy
  (23.41% vs 22.38%) and mean official score (0.3685 vs 0.3553).
  It ties text_majority_selected in exact accuracy (23.41%) and beats it in
  mean official score (0.3685 vs 0.3622).

Negative / limitation:
  The oracle best selected score is still much higher (0.4951), and per-task
  results are uneven. MMLU and story_cloze remain worse than text majority,
  while lambada, race, siqa, and squad improve on at least one utility axis.

Current claim:
  A richer utility target lets a decoder-free latent fuser recover some
  multi-sender value and match/beat text-majority on selected metrics, but the
  result is not yet robust enough for a main text-superiority claim.
```

Task-balanced and task-aware follow-up, 2026-05-29：

```text
implemented options:
  --task-loss-weighting balanced
  --target-mode task_aware_score

code:
  /data1/luyifei/drla/drla/scripts/train_cola_hierarchical_latent_fuser.py

motivation:
  v2 is uneven across tasks.
  Test gains appear on lambada/race/siqa/squad, while mmlu/story_cloze remain
  behind text majority.
```

Task-balanced score fuser v3：

```text
training artifact:
  /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
  p2e_hierarchical_fuser_score_taskbalanced_full_seed20260529_20260529

best eval:
  /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
  p2e_hierarchical_fuser_score_taskbalanced_full_seed20260529_20260529_best_eval_test

SwanLab cloud run:
  k2ujjjdrmcyzutwnwbyyf

test:
  model_selected_accuracy = 22.38%
  model_mean_official_score = 0.3654
  text_majority_selected_accuracy = 23.41%
  text_majority_mean_official_score = 0.3622
```

Task-aware score fuser v4：

```text
target:
  exact correctness for lambada/mmlu/obqa/race/siqa
  continuous score for hellaswag/squad/story_cloze

training artifact:
  /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
  p2e_hierarchical_fuser_taskaware_score_full_seed20260529_20260529_metricfix

best eval:
  /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
  p2e_hierarchical_fuser_taskaware_score_full_seed20260529_20260529_metricfix_best_eval_test

SwanLab cloud run:
  fdjwe4vfq70syq0oz9tro

test:
  model_selected_accuracy = 22.18%
  model_mean_official_score = 0.3605
  text_majority_selected_accuracy = 23.41%
  text_majority_mean_official_score = 0.3622
```

Interpretation：

```text
Both follow-ups are negative relative to score-target v2. Task-balanced loss
improves neither micro nor macro test performance. Task-aware exact/score
targets also underperform v2 after fixing checkpoint selection to use mean
official score.

Current best P2-E fuser remains:
  score-target v2
  /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
  p2e_hierarchical_fuser_score_full_seed20260529_20260529

The next improvement should not be simple loss reweighting. It should either
use a richer receiver-side state objective, learned calibration with more held-
out evidence, or a direct latent-state utility task rather than sender-choice
proxy supervision.
```

P2-E latent-state utility verifier, 2026-05-29：

```text
motivation:
  sender-choice fusers are limited by the proxy "which decoded answer would
  have been best?".
  A receiver also needs a decoder-free state estimate:
    does this group of latent messages already contain a useful answer state?
    how high is the best available latent-message utility?

train script:
  /data1/luyifei/drla/drla/scripts/train_cola_hierarchical_state_verifier.py

training artifact:
  /data1/luyifei/drla/outputs/cola_hierarchical_state_verifier/
  p2e_state_verifier_full_seed20260529_20260529

training:
  CUDA/GPU
  SwanLab cloud run = brtvqv9yd3h2gbcsu25n5
  epochs = 24
  batch_size = 256
  valid_interval = 50
  best_checkpoint.pt selected by valid any_auroc + (1 - best_score_rmse)

online inputs:
  sanitized latent blocks
  process features
  readiness/certificate fields
  task id

offline labels:
  any_correct = any sender selected answer is exact-correct
  best_score = max sender selected official score
```

Best-checkpoint metrics：

| split | any AUROC | any acc@0.5 | any brier | best-score corr | best-score RMSE | best-score MAE |
|---|---:|---:|---:|---:|---:|---:|
| valid | 0.7204 | 59.80% | 0.2172 | 0.3212 | 0.3932 | 0.3517 |
| test | 0.7054 | 58.52% | 0.2208 | 0.3078 | 0.4029 | 0.3640 |

Heuristic certificate baselines on the same test split：

| predictor | any AUROC | best-score corr | best-score RMSE |
|---|---:|---:|---:|
| max correctness head | 0.4717 | -0.0769 | 0.4471 |
| max readiness | 0.4776 | -0.1232 | 0.6466 |
| max answer-identity-stability | 0.5079 | -0.0762 | 0.6044 |
| min completion-risk inverse | 0.4687 | -0.0759 | 0.6585 |
| max contentful | 0.4837 | -0.0411 | 0.5501 |

Post-hoc calibration and ablation, 2026-05-29：

```text
eval script:
  /data1/luyifei/drla/drla/scripts/eval_cola_hierarchical_state_verifier.py

artifact:
  /data1/luyifei/drla/outputs/cola_hierarchical_state_verifier/
  p2e_state_verifier_full_seed20260529_20260529_calibration_ablation

local-only:
  swanlab_mode = disabled
  no optimizer/backward on model weights
  Platt / affine calibrators are fitted on valid and reported on held-out test
```

Test calibration：

| setting | any AUROC | any acc@0.5 | any brier | any ECE10 | any prob mean | target mean |
|---|---:|---:|---:|---:|---:|---:|
| raw full | 0.7054 | 58.52% | 0.2208 | 0.1444 | 0.4894 | 0.3470 |
| calibrated | 0.7054 | 67.15% | 0.1985 | 0.0232 | 0.3418 | 0.3470 |

Input / prior ablation on held-out test：

| predictor | any AUROC | any brier | best-score corr | best-score RMSE |
|---|---:|---:|---:|---:|
| full verifier | 0.7054 | 0.2208 | 0.3078 | 0.4029 |
| zero latent blocks | 0.6731 | 0.2208 | 0.2317 | 0.4132 |
| zero process features | 0.7046 | 0.2199 | 0.3040 | 0.4035 |
| zero certificates | 0.7065 | 0.2206 | 0.3113 | 0.4024 |
| valid task-prior only | 0.6399 | 0.2050 | 0.2399 | 0.4124 |
| valid global-prior only | 0.5142 | 0.2268 | 0.0000 | 0.4235 |

Risk-control threshold diagnostic：

```text
valid-fitted target precision 0.6:
  threshold = 0.7323
  valid precision = 0.629, recall = 0.236, coverage = 0.125
  test precision = 0.609, recall = 0.231, coverage = 0.131

target precision 0.7 / 0.8:
  no non-trivial held-out coverage yet
```

Interpretation：

```text
This is positive evidence for the richer state-objective direction. The model
predicts group-level latent utility substantially better than simple online
certificate/readiness heuristics. In particular, it reaches test any_correct
AUROC 0.7054, while the heuristic heads are near random.

The task-prior baseline shows that benchmark difficulty explains part of the
signal, but not all of it: task-prior AUROC is 0.6399, while the full verifier
reaches 0.7054, and zeroing raw latent blocks drops the verifier to 0.6731.
This supports a cautious claim that raw latent states contribute measurable
utility signal beyond task identity and simple certificate heads.

Calibration is useful but not enough to make this a final downstream
communication win. Platt calibration fixes probability scale on held-out test
(ECE 0.1444 -> 0.0232; Brier 0.2208 -> 0.1985), but high-precision risk
control is still weak: target precision 0.6 has only 13.1% test coverage, and
0.7/0.8 have no non-trivial coverage. The next step should use the calibrated
state verifier as an explicit receiver-side state signal, while keeping text
handoff and task-prior baselines in the same protocol.
```

P2-E calibrated receiver-state policy audit, 2026-05-29：

```text
eval script:
  /data1/luyifei/drla/drla/scripts/eval_cola_hierarchical_state_policy.py

artifact:
  /data1/luyifei/drla/outputs/cola_hierarchical_state_policy/
  p2e_state_policy_score_fuser_v2_locked_seed20260529_20260529

inputs:
  calibrated state verifier best_checkpoint.pt
  score-target fuser v2 best_checkpoint.pt

local-only:
  swanlab_mode = disabled
  thresholds selected on valid
  held-out test reported
```

Always-on held-out test baselines：

| policy | exact acc | mean official score |
|---|---:|---:|
| single sender first | 22.38% | 0.3553 |
| score-target latent fuser v2 | 23.41% | 0.3685 |
| text majority selected | 23.41% | 0.3622 |
| oracle any selected | 34.70% | 0.4951 |

Selected receiver-state gates on held-out test：

| gate | valid selection rule | test coverage | test any precision | accepted fuser acc | accepted fuser score | accepted text acc |
|---|---|---:|---:|---:|---:|---:|
| state_any_prob | target any precision 0.60 | 13.35% | 61.54% | 53.85% | 0.5503 | 49.23% |
| state_any_prob | target any precision 0.65 | 11.29% | 63.64% | 56.36% | 0.5636 | 50.91% |
| state_any_prob | target coverage 0.10 | 10.88% | 62.26% | 54.72% | 0.5472 | 49.06% |
| state_any_prob | target coverage 0.25 | 22.18% | 52.78% | 42.59% | 0.5462 | 42.59% |
| train task-prior any | target any precision 0.60 | 12.32% | 61.67% | 55.00% | 0.5500 | 48.33% |
| train task-prior any | target coverage 0.25 | 49.69% | 47.11% | 30.17% | 0.3477 | 31.82% |
| fuser confidence | target coverage 0.10 | 12.53% | 52.46% | 29.51% | 0.3203 | 29.51% |

Interpretation：

```text
The calibrated state signal is useful as a receiver state / risk signal:
state_any_prob can select a high-utility latent subset where fuser exact
accuracy rises from the always-on 23.41% to about 54-56%.

However, this is not yet a complete downstream policy win. At the strict
13.35% coverage gate, fuser-else-first fallback reaches only 23.00% exact and
0.3609 score, below always-on fuser score 0.3685. At high-confidence coverage,
train task-prior any is also competitive, so any claim must keep task-prior
controls in the table.

The clean next engineering target is not another threshold-only policy. It is
to expose the calibrated state tuple to the receiver:
  state_any_prob, state_best_score_pred, fuser_confidence, sender-choice logits,
  task-prior controls, and raw calibrated risk flags.

Then train/evaluate a receiver that consumes this structured state to request
more latent evidence, choose a sender, or abstain. The current audit says the
state is meaningful, but the sender selector / fallback action is still the
bottleneck.
```

P2-E structured receiver-state action selector, 2026-05-29：

```text
train script:
  /data1/luyifei/drla/drla/scripts/train_cola_receiver_state_action_selector.py

eval script:
  /data1/luyifei/drla/drla/scripts/eval_cola_receiver_state_action_selector.py

online features:
  state_any_prob
  state_best_score_pred
  state_raw_any_prob
  state_raw_best_score_pred
  fuser logits/probabilities/confidence/margin/entropy
  train task/global priors

offline labels:
  sender selected_correct / selected_score
  any_correct
  oracle best selected_score

training:
  CUDA/GPU
  SwanLab cloud
  valid_interval = 25
  best_checkpoint.pt selected by valid model_mean_score
```

Two full selector variants were trained：

| variant | SwanLab | output mode | feature mode | best step |
|---|---|---|---|---:|
| direct selector | 7emwtma3xyvrmvlv1hibb | direct learned sender logits | state_fuser_prior | 1100 |
| residual selector | 2z5uj588g6kkkv6dpte97 | fuser logits + learned delta | state_fuser_prior | 1100 |

Held-out test, best checkpoint：

| policy | exact acc | mean official score |
|---|---:|---:|
| single sender first | 22.38% | 0.3553 |
| text majority selected | 23.41% | 0.3622 |
| score-target latent fuser v2 | 23.41% | 0.3685 |
| direct state action selector | 21.97% | 0.3554 |
| residual state action selector | 22.79% | 0.3626 |

Valid-selected gate audit on held-out test：

| variant | valid rule | test coverage | test any precision | accepted selector acc | accepted selector score | fallback-first score |
|---|---|---:|---:|---:|---:|---:|
| direct | target any precision 0.60 | 9.45% | 58.70% | 39.13% | 0.4217 | 0.3472 |
| residual | target any precision 0.60 | 9.24% | 60.00% | 40.00% | 0.4252 | 0.3466 |

Interpretation：

```text
This is a negative but useful result. A learned selector over structured
state/fuser/prior features does not beat the raw score-target latent fuser v2.
The residual variant is better than the direct variant and roughly matches
text-majority score, but still trails fuser v2.

This suggests that the current bottleneck is not simply "combine the existing
state scalars with an MLP". The fuser's raw latent representation still carries
sender-choice information that the compressed structured state does not
preserve well enough.

Do not keep tuning this shallow state selector as the main path unless a new
action space is introduced. The next credible P2-E step should either:
  1. let the receiver request additional latent evidence and evaluate the value
     of that extra evidence, or
  2. train a richer selector that consumes sender-level latent states directly,
     with the calibrated state verifier used only as risk/control side
     information.
```

P2-E state-conditioned sender-level latent fuser, 2026-05-29：

```text
train script:
  /data1/luyifei/drla/drla/scripts/train_cola_state_conditioned_latent_fuser.py

design:
  initialize from score-target latent fuser v2
  keep sender-level latent states
  add calibrated state/fuser/prior side features
  output = base_fuser_logits + learned residual delta

online features:
  raw sender latent/process/certificate packet fields
  state_any_prob / state_best_score_pred
  fuser logits/probs/confidence/margin/entropy
  train task/global priors

offline labels:
  selected_score / selected_correct only
```

Two full variants were trained：

| variant | SwanLab | backbone | lr | best step |
|---|---|---|---:|---:|
| frozen residual | 1ua24n9yo4tsrq4inahb9 | frozen fuser v2 | 5e-4 | 384 |
| unfrozen residual | je3suuujcleox4x40lahd | full fuser v2 finetune | 1e-4 | 100 |

Held-out test, best checkpoint：

| policy | exact acc | mean official score |
|---|---:|---:|
| score-target latent fuser v2 | 23.41% | 0.3685 |
| text majority selected | 23.41% | 0.3622 |
| frozen state-conditioned fuser | 23.00% | 0.3654 |
| unfrozen state-conditioned fuser | 23.41% | 0.3651 |

Interpretation：

```text
This is a second negative result for simply adding calibrated state side
information to the current sender-choice task. Both variants improve or look
competitive on valid, but neither beats the original score-target fuser v2 on
held-out test score.

The conclusion is stronger than the shallow MLP result: even when raw
sender-level latent states are preserved, the current calibrated state tuple
does not provide a robust sender-choice improvement. It remains useful as a
risk/readiness signal, but it is not yet a selector-improving feature.

The next P2-E step should move to a genuinely different action space:
request-more-latent / additional sender evidence / sequential aggregation.
That tests whether latent communication helps when the receiver can acquire
more evidence, rather than asking a small selector to re-rank the same three
already-available senders.
```

P2-E request-more-latent potential audit, 2026-05-29：

```text
script:
  /data1/luyifei/drla/drla/scripts/audit_cola_request_more_latent_potential.py

artifact:
  /data1/luyifei/drla/outputs/cola_request_more_latent/
  p2e_request_more_latent_potential_locked_seed20260529_20260529

scope:
  local-only audit; swanlab_mode=disabled
  no optimizer/backward
  valid-selected thresholds, held-out test report

group split:
  train / valid / test = 3998 / 495 / 487
```

Question：

```text
If the receiver initially sees only the first latent sender packet, is it worth
requesting additional sender evidence?

This is a different action space from same-action sender reranking. The policy
first decides whether to spend more sender budget, then an offline upper bound
or simple decoder-free readiness selector estimates what the extra evidence
could buy.
```

Held-out test result：

| setting | exact acc | mean official score |
|---|---:|---:|
| first sender only | 22.38% | 0.3553 |
| prefix2 oracle | 30.39% | 0.4458 |
| prefix3 oracle | 34.70% | 0.4951 |
| prefix2 readiness selector | 23.61% | 0.3661 |
| prefix3 readiness selector | 22.38% | 0.3518 |

Marginal value：

| request action | helpful rate | mean score gain |
|---|---:|---:|
| first -> prefix2 | 24.23% | +0.0905 |
| first -> prefix3 | 33.47% | +0.1398 |
| prefix2 -> prefix3 | 13.35% | +0.0493 |

Representative valid-threshold policies on held-out test：

| rule | request rate | avg sender budget | helpful precision | oracle-after-request score | readiness-after-request score |
|---|---:|---:|---:|---:|---:|
| train task gain prior, target request rate 0.10/0.25 | 34.50% | 1.69 | 29.17% | 0.4232 | 0.3640 |
| contentful low, target request rate 0.50 | 52.36% | 2.05 | 32.55% | 0.4297 | 0.3604 |
| completion_risk high, target request rate 0.10 | 11.29% | 1.23 | 29.09% | 0.3707 | 0.3593 |
| readiness low, target helpful precision 0.50 | 1.03% | 1.02 | 60.00% | 0.3592 | 0.3571 |

Interpretation：

```text
The extra latent evidence is valuable in principle: prefix3 oracle matches the
known aggregation oracle bound at 34.70% / 0.4951. This is strong evidence that
the communication substrate contains useful complementary information across
senders.

The current practical request/readiness heuristics are not enough. They mostly
improve oracle-after-request upper bounds, while practical readiness-after-
request score remains near first-sender or text-majority quality.

Therefore the next P2-E experiment should train a learned request policy or
sequential aggregator that directly models marginal gain / expected utility.
Do not interpret this audit as proof that latent communication beats text; it
only justifies spending the next experiment on additional-evidence actions.
```

P2-E learned request-more-latent policy, 2026-05-29：

```text
script:
  /data1/luyifei/drla/drla/scripts/train_cola_request_more_policy.py

online input:
  first sender latent/process/certificate/task fields only

action:
  stop at first sender
  or request the remaining sender packets

post-request practical selector:
  existing score-target latent fuser v2

training:
  CUDA/GPU + SwanLab cloud
  valid_interval = 50
  best_checkpoint.pt selected by valid helpful AUROC + gain correlation
```

Full runs：

| target | SwanLab | artifact | best step | test helpful AUROC | test gain corr |
|---|---|---|---:|---:|---:|
| `fuser_gain` | `zn7zl11z11ghmfenr8wr4` | `/data1/luyifei/drla/outputs/cola_request_more_policy/p2e_request_more_policy_fuser_gain_full_seed20260529_20260529` | 350 | 0.6823 | 0.0216 |
| `oracle_gain` | `at0w7v8gsewja1vudb3jx` | `/data1/luyifei/drla/outputs/cola_request_more_policy/p2e_request_more_policy_oracle_gain_full_seed20260529_20260529` | 350 | 0.6461 | 0.0879 |

Held-out test baselines and best practical policies：

| policy | request rate | avg sender budget | exact acc | mean official score |
|---|---:|---:|---:|---:|
| first sender only | 0.00% | 1.00 | 22.38% | 0.3553 |
| text majority | n/a | 3 text answers | 23.41% | 0.3622 |
| always request + fuser v2 | 100.00% | 3.00 | 23.41% | 0.3685 |
| `fuser_gain` best practical policy | 23.00% | 1.46 | 23.20% | 0.3689 |
| `oracle_gain` best practical policy | 25.67% | 1.51 | 22.79% | 0.3660 |
| always request oracle upper bound | 100.00% | 3.00 | 34.70% | 0.4951 |

Best `fuser_gain` practical policy：

```text
valid selection rule:
  target_request_rate = 0.25
  signal = gain_pred

held-out test:
  request rate = 22.9979%
  average sender budget = 1.45996
  target helpful precision = 32.14%
  oracle-after-request accuracy / score = 26.28% / 0.4010
  fuser-after-request accuracy / score = 23.20% / 0.3689
```

Interpretation：

```text
The learned request policy is a small but real budget-efficiency improvement:
it uses about half the sender budget of always-request and slightly improves
mean score over always-request fuser v2. It also beats text-majority score.

It is not a strong quality improvement. Exact accuracy remains slightly lower
than always-request/text-majority, and the score gain over always-request fuser
is tiny.

The main unsolved gap is post-request aggregation. The request policy can find
cases with oracle value, but the current fuser does not fully exploit the
requested packets. The next step should improve the sequential/post-request
selector rather than only sharpening the request threshold.
```

P2-E post-request anchor-aware selector, 2026-05-29：

```text
script:
  /data1/luyifei/drla/drla/scripts/train_cola_post_request_selector.py

scope:
  after request-more-latent
  consume first sender + requested sender packets
  compare always-request selection and request-policy-gated selection

architecture:
  reuse hierarchical latent fuser sender encoder
  make each candidate anchor-aware:
    candidate state
    first-sender state
    candidate - first
    candidate * first
  post-request Transformer over candidates
  score, rank, and gain heads

loss:
  continuous score MSE
  listwise ranking loss
  pairwise ranking loss
  gain-over-first regression
```

Full runs：

| variant | SwanLab | artifact | best step |
|---|---|---|---:|
| anchor score selection | `r7qj2vu48vws5lnp5edxf` | `/data1/luyifei/drla/outputs/cola_post_request_selector/p2e_post_request_selector_anchor_score_full_seed20260529_20260529` | 350 |
| anchor rank selection | `089s7vnjawqpd4p7m20ak` | `/data1/luyifei/drla/outputs/cola_post_request_selector/p2e_post_request_selector_anchor_rank_full_seed20260529_20260529` | 200 |

Held-out test：

| policy | request rate | exact acc | mean official score |
|---|---:|---:|---:|
| first sender only | 0.00% | 22.38% | 0.3553 |
| text majority | n/a | 23.41% | 0.3622 |
| always request + fuser v2 | 100.00% | 23.41% | 0.3685 |
| request policy + fuser v2 | 23.00% | 23.20% | 0.3689 |
| anchor score selector, standalone | 100.00% | 22.18% | 0.3572 |
| request policy + anchor score selector | 23.00% | 23.00% | 0.3668 |
| anchor rank selector, standalone | 100.00% | 21.77% | 0.3500 |
| request policy + anchor rank selector | 23.00% | 23.41% | 0.3697 |
| oracle upper bound | 100.00% | 34.70% | 0.4951 |

Interpretation：

```text
Standalone post-request selection is negative. The anchor-aware selector is
not a replacement for fuser v2 when forced to choose on every sample.

Request-gated rank selection is narrowly positive: with the same request
policy as before, it matches text/fuser exact accuracy and slightly improves
mean score over request+fuser v2 and always-request fuser v2.

The gain is small. The scientific takeaway is not "new selector solves
aggregation", but "post-request ranking has value inside the requested subset,
and independent request + independent selector composition is likely too weak
to close the oracle gap."
```

Design implication from related methods：

```text
Learning-to-rank methods such as RankNet motivate pairwise/listwise supervision
for candidate selection, but our standalone negative result shows that ranking
alone is not enough under sparse/noisy latent utility labels.

Selective prediction / reject-option work motivates controlling coverage and
risk instead of always forcing a decision. Adaptive computation work similarly
frames extra computation as a budgeted action. For P2, the natural next step is
therefore a joint budgeted policy:
  decide whether to request
  then select/aggregate only under that request distribution
  optimize held-out budgeted utility directly
```

P2-E joint request-and-select policy, 2026-05-29：

```text
script:
  /data1/luyifei/drla/drla/scripts/train_cola_joint_request_select_policy.py

online contract:
  request head sees only first sender packet
  selector head sees requested sender packets only after request

loss:
  request BCE on oracle helpfulness
  request gain regression
  selector score MSE
  selector listwise ranking
  selector pairwise ranking
  selector gain-over-first regression
  differentiable budgeted expected utility

checkpoint selection:
  valid target_request_rate=0.25 model-after-request score

reporting:
  valid-selected thresholds
  held-out test metrics
```

Full runs：

| variant | SwanLab | artifact | best step |
|---|---|---|---:|
| joint rank selection | `g126cuz3w32r9g76jizcs` | `/data1/luyifei/drla/outputs/cola_joint_request_select_policy/p2e_joint_request_select_rank_full_seed20260529_20260529` | 250 |
| joint score selection | `2t0j6t9v33qfkfn71w2mi` | `/data1/luyifei/drla/outputs/cola_joint_request_select_policy/p2e_joint_request_select_score_full_seed20260529_20260529` | 250 |

Held-out test：

| policy | request rate | avg sender budget | exact acc | mean official score |
|---|---:|---:|---:|---:|
| first sender only | 0.00% | 1.00 | 22.38% | 0.3553 |
| text majority | n/a | 3 text answers | 23.41% | 0.3622 |
| always request + fuser v2 | 100.00% | 3.00 | 23.41% | 0.3685 |
| request policy + fuser v2 | 23.00% | 1.46 | 23.20% | 0.3689 |
| request policy + anchor-rank selector | 23.00% | 1.46 | 23.41% | 0.3697 |
| joint rank, best gated | 47.64% | 1.95 | 23.82% | 0.3733 |
| joint score, best gated | 47.64% | 1.95 | 24.02% | 0.3735 |
| oracle upper bound | 100.00% | 3.00 | 34.70% | 0.4951 |

Best current row：

```text
model:
  joint score selection

valid-selected threshold:
  selection_mode = target_request_rate
  target_value = 0.50
  signal = request_prob

held-out test:
  request rate = 47.64%
  average sender budget = 1.9528
  model helpful precision = 35.34%
  model-after-request accuracy = 24.02%
  model-after-request score = 0.3735
  oracle-after-request accuracy = 26.49%
  oracle-after-request score = 0.4137
```

Calibration / risk-control audit：

```text
script:
  /data1/luyifei/drla/drla/scripts/audit_cola_joint_policy_calibration.py

artifact:
  /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
  p2e_joint_request_select_score_full_seed20260529_20260529_calibration_risk

protocol:
  local-only, swanlab_mode=disabled
  thresholds selected on valid
  test used only after selection
```

Key results：

```text
request_prob ranking:
  model helpful AUROC = 0.6959 valid / 0.7164 test
  oracle helpful AUROC = 0.7089 valid / 0.6825 test

probability calibration:
  model helpful ECE = 0.2363 valid / 0.2226 test
  request_prob is over-confident:
    test prob mean = 0.4464
    test model-helpful target mean = 0.2238

gain regression:
  request_gain_pred corr with test model gain = 0.0010
  request_gain_pred corr with test oracle gain = 0.1315
```

Risk-control implication：

```text
The current best policy is utility-selected, not risk-certified:
  target_request_rate = 0.50 on request_prob
  test score = 0.3735
  requested model-loss Wilson95 upper = 35.02%

Strict conditional loss-risk caps do not currently produce a useful frontier:
  cap 0.10 / 0.20: no non-trivial valid threshold
  cap 0.30 / 0.40: almost-always request, test score drops to 0.3494
```

Fresh seed / fresh split robustness update：

```text
seed30 overlap audit:
  seed30 test groups = 495
  seed30 test vs seed29 split:
    seed29 train = 406
    seed29 valid = 51
    seed29 test = 38

non-canonical diagnostic:
  p2e_joint_request_select_score_full_seed20260530_20260529
  reason: reused seed29 fuser checkpoint / norm_stats
```

Strict seed30 protocol：

```text
same split seed:
  20260530

same-split fuser:
  artifact:
    /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
    p2e_hierarchical_fuser_score_full_seed20260530_20260529
  SwanLab:
    qweuypbg1ugls3io0s9j0
  best-checkpoint eval:
    /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
    p2e_hierarchical_fuser_score_full_seed20260530_20260529_best_eval_test

same-split joint:
  artifact:
    /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
    p2e_joint_request_select_score_full_seed20260530_fuserseed20260530_20260529
  SwanLab:
    n4wu1f4ghzfwe6mhvltei
  calibration:
    /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
    p2e_joint_request_select_score_full_seed20260530_fuserseed20260530_20260529_calibration_risk
```

Strict seed30 held-out test：

| policy | request rate | budget | acc | score |
|---|---:|---:|---:|---:|
| first sender only | 0.00% | 1.00 | 22.42% | 0.3669 |
| text majority | n/a | 3 text answers | 22.63% | 0.3642 |
| same-split fuser best | 100.00% | 3.00 | 22.63% | 0.3694 |
| joint best gated | 50.51% | 2.01 | 23.64% | 0.3796 |
| oracle upper bound | 100.00% | 3.00 | 35.96% | 0.5071 |

Strict seed30 calibration note：

```text
best joint policy:
  target_request_rate = 0.50
  signal = request_gain_pred

gains:
  score gain vs text = +0.0153
  score gain vs same-split fuser = +0.0102

risk:
  requested model-loss Wilson95 upper = 25.40%
  request_prob -> model_request_helpful AUROC / ECE = 0.6255 / 0.2697
```

Strict seed31 update：

```text
same-split fuser:
  artifact:
    /data1/luyifei/drla/outputs/cola_hierarchical_latent_fuser/
    p2e_hierarchical_fuser_score_full_seed20260531_20260529
  SwanLab:
    64h605uhcjpse62n84l8v

same-split joint:
  artifact:
    /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
    p2e_joint_request_select_score_full_seed20260531_fuserseed20260531_20260529
  SwanLab:
    xh19j75yervdr7l603rt4
```

Strict seed31 held-out test：

| policy | request rate | budget | acc | score |
|---|---:|---:|---:|---:|
| first sender only | 0.00% | 1.00 | 16.96% | 0.3074 |
| text majority | n/a | 3 text answers | 17.75% | 0.3180 |
| same-split fuser best | 100.00% | 3.00 | 17.75% | 0.3184 |
| joint best selected | 99.80% | 3.00 | 19.33% | 0.3426 |
| oracle upper bound | 100.00% | 3.00 | 29.78% | 0.4515 |

Three-strict-seed aggregate：

```text
artifact:
  /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
  p2e_joint_request_select_strict_seed29_30_31_summary_20260529

macro mean:
  first sender = 20.59%, score 0.3432
  text majority = 21.26%, score 0.3481
  same-split fuser = 21.26%, score 0.3521
  joint selected = 22.33%, score 0.3652
  oracle = 33.48%, score 0.4846

joint policy:
  request rate = 65.98%
  average sender budget = 2.3196
  requested model-loss Wilson95 upper = 29.13%
  request_prob model-helpful AUROC / ECE = 0.6688 / 0.2571
```

Checkpoint-selection audit：

```text
local evaluator:
  /data1/luyifei/drla/drla/scripts/eval_cola_joint_request_select_policy.py

last-checkpoint best policy:
  seed29 = 23.00%, score 0.3653
  seed30 = 24.85%, score 0.3919
  seed31 = 17.75%, score 0.3252
```

Interpretation：

```text
Joint request-and-select is now the strongest P2-E budgeted latent result. It
improves mean score over text majority, always-request fuser v2, independent
request+fuser, and independent request+anchor-rank. It also slightly improves
exact accuracy over text/fuser baselines while using less than two sender
packets on average.

The claim remains narrow: this is a same-substrate, budgeted multi-sender
latent communication win on the locked split, not a proof that latent
communication generally dominates text communication. The oracle gap is still
large. The calibration audit says request_prob is useful as a ranking signal,
but its probability scale and conditional loss-risk frontier are not yet strong
enough for a formal safety claim. The strict seed30 rerun reproduces the
positive utility direction against text and same-split fuser, but it also
confirms calibration weakness and exposes checkpoint-selection sensitivity.
The strict seed31 rerun adds a third positive utility direction, but its best
selected policy is almost always request, so budget efficiency is not yet
stable. Checkpoint-selection sensitivity motivated the valid-frontier audit
below; that audit is now complete and does not replace the canonical target025
strict aggregate.
```

P2-E valid-frontier checkpoint selection audit, 2026-05-31：

```text
motivation:
  audit whether selecting checkpoints by valid utility-frontier improves over
  the earlier target025 checkpoint-selection rule without switching to last
  checkpoint or using test data

implementation:
  train_cola_joint_request_select_policy.py
    --checkpoint-selection-mode valid_rate_frontier
    --checkpoint-selection-request-rates 0.10,0.25,0.50,0.75

strict seed artifacts:
  seed29 = p2e_joint_request_select_score_frontier_full_seed20260529_fuserseed20260529_20260529
  seed30 = p2e_joint_request_select_score_frontier_full_seed20260530_fuserseed20260530_20260529
  seed31 = p2e_joint_request_select_score_frontier_full_seed20260531_fuserseed20260531_20260529

aggregate:
  /data1/luyifei/drla/outputs/cola_joint_request_select_policy/
  p2e_joint_request_select_frontier_strict_seed29_30_31_summary_20260531
```

Leak-safe frontier aggregate protocol：

```text
canonical row per seed:
  choose max valid_model_after_request_score among valid-selected policy rows
  report held-out test only after the row is fixed

non-canonical diagnostic:
  test-best rows are kept in the artifact for debugging but must not be used as
  paper/canonical numbers
```

Frontier result：

```text
old target025 strict aggregate:
  joint selected = 22.33%, score 0.3652
  request rate / budget = 65.98% / 2.3196

valid-frontier canonical aggregate:
  selected = 21.46%, score 0.3505
  request rate / budget = 49.73% / 1.9946
  score delta vs old joint = -0.0147
  acc delta vs old joint = -0.87 pp

test-best diagnostic aggregate, non-canonical:
  selected = 22.20%, score 0.3584
  score delta vs old joint = -0.0069
```

Interpretation：

```text
Valid-frontier checkpoint selection does not improve the current canonical
joint request-select result. It reduces budget but gives up too much score, and
the calibration/risk-control weakness remains. Keep the old target025 strict
aggregate as the current P2-E canonical result. Treat frontier as a completed
negative checkpoint-selection audit, not as the next main branch.
```

## 8. Evaluation Protocol

```text
default:
  official Cola 8 tasks
  b64 / bs12 / t16
  max blocks = 4
  seeds = 66, 67, 68
  split seed = 20260601

canonical comparison:
  same sender depth t
  same receiver budget
  same communication boundary
  agent_b_input_contract = message_only
  score_output_scope = receiver_only
  B_none(empty input)
  B_text(A_raw_text_message_t)
  B_latent(A_latent_packet_t)
  B_corrupt(corrupted_latent_packet_t)

diagnostic comparison:
  shared_context can additionally give B q, but those rows cannot support the
  pure Agent-A -> Agent-B handoff claim

primary scores:
  score(B_latent) - score(B_corrupt): readability / payload use
  score(B_latent) - score(B_none): marginal utility
  score(B_latent) - score(B_text): text competitiveness

historical diagnostics:
  text_selected/text_final/text_prediction_stability direct handoff
  replay-based latent decoding
  latent answer reader retrieval

efficiency:
  text tokens, latent elements/bytes, decode/re-encode calls, runtime, B blocks

fidelity:
  downstream score, paired win/loss/tie, answer agreement,
  B_latent-vs-B_text gap, B_latent-vs-B_none gap, matched-vs-corrupted gap

receiver-native distribution:
  packet/native trace alignment
  receiver post-consumption latent/process feature distance
  native-vs-consumed OOD AUROC
  matched-vs-corrupt separability
```

Hierarchical/request-more extensions can continue as budgeted communication
diagnostics, but paper-level text-vs-latent claims must use the corrected
Agent-B channel-equivalent protocol above.

## 9. Roadmap

```text
P2-A Packet v2: completed; schema fields, packet_schema.json, locked packet rebuild
P2-B Distribution audit: completed; audit script, corrupted controls, matched-vs-control gaps
P2-C Single-handoff receiver: v1 compatibility ablations completed; standalone eval/risk calibration still pending
P2-D Sequential communication: replay runner and duplicate-safe local audit implemented; historical direct-answer text handoff and all-visible decode-and-emit tables are downgraded to diagnostics; old official8 50/task table measured decoder/replay visibility, not valid Agent-B communication; corrected B_none/B_text/B_latent/B_corrupt scripts now support LatentMAS-aligned message_only + receiver_only boundaries; 1-per-task message_only smoke and formal official8 50/task evaluation pass protocol audit; formal result shows matched latent has significant marginal utility over empty input and is competitive with text, but does not significantly beat text and fails the all-corrupt gate because wrong_block is anomalously strong
P2-D1 Benchmark capability gate: scripts implemented; full candidate JSONL prepared for ARC-E, ARC-C, GSM8K, MBPP+, HumanEval+, GPQA-Diamond, and MedQA; code-task execution gate implemented; ARC-E one-sample single+Role TextMAS smoke confirms CUDA generation path, parser, metrics.jsonl, task_summary.csv, and local-only SwanLab-disabled output schema; formal full 7-task gate completed with admitted_tasks=[] so these tasks are not yet eligible for P2 main text-vs-latent tables
P2-D2 Locked split: deterministic calibration/held-out split implemented and generated with split_seed=20260602; calibration=842, heldout=3359, overlap=0; future prompt/protocol repair is calibration-only and must be followed by held-out gate
P2-D3 Initial prompt repair: prompt variants implemented; non-code calibration single sweep shows cola_fewshot_v1 is not better than generic_v1; GPQA-Diamond generic_v1 single-mode passes calibration but Role TextMAS fails, so no task is admitted and held-out remains untouched
P2-E Hierarchical extension: aggregation potential audit completed; learned latent fuser v1 negative; score-target fuser v2 is current best sender-choice model and improves held-out mean official score while tying text-majority accuracy on test; task-balanced v3 and task-aware v4 are negative follow-ups; latent-state utility verifier shows positive group-level utility prediction vs certificate heuristics; calibrated state-policy audit shows high-quality accepted latent subsets but no overall downstream policy win yet
P2-F Writeup: map claims to evidence ladder, separate same-substrate from hetero future work
```

## 10. Risks and Claims

```text
risks:
  packet under-specified:
  fix prefix_contract / hashes / prefix refs
  receiver uses metadata not latent:
  use corrupted controls and ablations
  replay not native:
  add no-text receiver-native latent consumption and receiver-state distribution audits
  text baseline drift:
  do not use selected_prediction direct handoff as main B_text baseline
  require B_text(A_raw_text_message_t) under the same message_only receiver
  budget and receiver_only scorer
  official8 too weak:
  validate substrate first, add reasoning-heavy tasks later
  heterogeneous claim drift:
  state same-substrate limitation clearly

expected minimum:
  sanitized Cola latent packets as agent-agent messages
  distribution audit
  evidence B uses matched latent payload vs corrupted controls

expected strong:
  message_only + receiver_only B_latent beats corrupted controls and B_none
  B_latent is competitive with B_text_raw_message at lower communication/runtime cost

expected very strong:
  sequential + hierarchical envelopes work
  P1-selected depth approaches final-depth utility
  latent beats text on cost-quality Pareto
```
