# Legacy Stage B/C Code Archive

Archived on 2026-05-24 after the route changed to:

```text
official Cola VAE + official Cola DiT
+ block-wise rollout trace
+ decoder probe
+ oracle readiness frontier
+ multi-signal readiness / halt model
```

This archive preserves the old self-built Stage A/B/C and custom Cola-latent-prior code for reproducibility only.

## Archived Code

```text
drla/data/answer_judge.py
drla/data/gsm8k_stage_a.py
drla/data/stage_b.py
drla/data/stage_c.py
drla/models/stage_b.py
drla/models/stage_c.py
drla/training/stage_b_autoencoder.py
drla/training/stage_c_prior.py
drla/scripts/prepare_stage_a.py
drla/scripts/train_stage_b.py
drla/scripts/train_stage_c.py
drla/scripts/prepare_cola_gsm8k_eval.py
drla/scripts/eval_cola_outputs.py
drla/scripts/eval_cola_vae_reconstruction.py
drla/scripts/train_cola_latent_prior.py
drla/scripts/eval_cola_latent_prior.py
tests/test_stage_b.py
tests/test_stage_c.py
```

## Status

These files are diagnostic-only and are not evidence for final architecture validity.

Do not use them for the main experimental route unless explicitly restoring a historical run. The active route must use official Cola 8-task benchmarks and the readiness/halt pipeline described in `/data1/luyifei/drla/docs/DRLA_Implementation_Plan.md`.
