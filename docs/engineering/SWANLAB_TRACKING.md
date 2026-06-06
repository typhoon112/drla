# SwanLab Tracking

DRLA experiments use the project-local conda environment and the unified helper in
`drla.tracking`.

硬性约束：所有深度学习训练实验必须使用 SwanLab cloud。这里包括
smoke training、probe、readiness/halt model、LoRA/adapter、微调和 ablation。
无训练过程的脚本不要上 SwanLab：纯 eval、threshold sweep、trace collection、
frontier building、aggregation、`py_compile` 和数据格式检查只允许 `disabled`，
只写本地 artifact；传入任何非 disabled SwanLab mode 都应在脚本开头失败。

```bash
source /data1/luyifei/drla/scripts/activate_conda.sh
swanlab verify
```

## Minimal Usage

```python
from drla.tracking import finish_experiment, init_experiment, log_metrics

run = init_experiment(
    stage="stage-b",
    experiment_name="stage-b-autoencoder-bs16-ld128",
    description="Deterministic reasoning latent autoencoder.",
    config={
        "dataset": "cola-official-8tasks",
        "block_size": 16,
        "latent_dim": 128,
        "B_max": 32,
        "seed": 42,
    },
)

log_metrics(
    {
        "loss": 1.23,
        "l_answer": 0.45,
        "l_recon": 2.34,
        "answer_acc": 0.88,
    },
    step=100,
    prefix="train",
)

finish_experiment()
```

Defaults live in `/data1/luyifei/drla/configs/swanlab.yaml`.

## Stage Names

- `cola-benchmark`: official Cola 8-task benchmark evaluation (`lambada`, `mmlu`, `obqa`, `hellaswag`, `race`, `siqa`, `squad`, `story_cloze`).
- `stage-a`: legacy GSM8K target text, answer normalizer, judge, token/block stats.
- `stage-b`: reasoning encoder, latent decoder, answer-ready KD.
- `stage-c`: block-causal DiT prior and closed-loop fixed-B inference.
- `stage-d`: rollout probes, oracle halt, halt-verifier.
- `ablation`: controlled comparisons and go/no-go checks.

## Environment Overrides

```bash
export SWANLAB_MODE=cloud            # 只用于训练；无训练脚本不要设为 cloud
export SWANLAB_PROJECT=drla-mvp
export SWANLAB_EXPERIMENT_NAME=debug-stage-b
export SWANLAB_GROUP=stage-b-autoencoder
export SWANLAB_TAGS=debug,small-run
export SWANLAB_RUN_ID=<existing-run-id>
export SWANLAB_RESUME=allow          # or must
```

如果只是做不创建云端 run 的本地检查、trace、frontier、eval 或 aggregation：

```bash
SWANLAB_MODE=disabled python your_script.py
```

任何训练 run 都不能使用 `SWANLAB_MODE=disabled`、`offline` 或 `local`，
小型 smoke training 也不例外。任何无训练过程脚本都不要使用 `cloud`。

## Recommended Metric Names

Use namespaced keys so SwanLab charts stay comparable:

```text
train/loss
train/l_answer
train/l_recon
train/l_kd
train/l_noop
train/l_kl
train/lr
valid/answer_acc
valid/exact_match
valid/noop_consistency
valid/latent_norm_mean
valid/kl_per_block
prior/mse_block
prior/fixed_b4_acc
prior/fixed_b8_acc
prior/fixed_b16_acc
prior/fixed_b32_acc
halt/adaptive_acc
halt/avg_blocks
halt/early_stop_wrong_rate
halt/forced_stop_rate
halt/oracle_gap
```

## HuggingFace Trainer

For `transformers.Trainer`, keep the same SwanLab project by setting:

```python
from transformers import TrainingArguments

args = TrainingArguments(
    ...,
    report_to="swanlab",
    run_name="qwen-condition-probe",
)
```
