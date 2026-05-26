import math
import pytest
import torch

from drla.models.stage_b import StageBModelConfig, StageBReasoningAutoencoder
from drla.models.stage_c import (
    BlockCausalPrior,
    StageCPriorConfig,
    rollout_prior,
    sinusoidal_timestep_embedding,
)
from drla.training.stage_c_prior import (
    StageCPriorTrainConfig,
    active_block_mask,
    blockwise_mse,
    build_stage_b_batch_from_stage_c,
    finalize_latent_stats,
    flow_interpolate,
    flow_matching_loss,
    masked_flow_matching_loss,
    prepare_prior_training_batch,
    sample_active_flow_blocks,
    scheduled_value,
    validate_prior_mode,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [6] if text else []


def make_stage_c_batch_for_joint():
    return {
        "ids": ["sample"],
        "question_ids": torch.tensor([[1, 2, 0]]),
        "question_mask": torch.tensor([[1, 1, 0]]),
        "target_ids": torch.tensor([[4, 5, 0]]),
        "target_mask": torch.tensor([[1, 1, 0]]),
        "answer_norms": ["7"],
        "b_star": torch.tensor([1]),
        "block_mask": torch.tensor([[1, 0]]),
        "noop_mask": torch.tensor([[0, 1]]),
        "z_blocks": torch.randn(1, 2, 2, 4),
    }


def make_small_stage_b():
    return StageBReasoningAutoencoder(
        StageBModelConfig(
            vocab_size=16,
            b_max=2,
            block_size=2,
            latent_dim=4,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            max_answer_len=2,
            dropout=0.0,
        )
    )


def test_block_causal_prior_shapes_and_rollout():
    model = BlockCausalPrior(
        StageCPriorConfig(
            vocab_size=32,
            b_max=4,
            block_size=3,
            latent_dim=8,
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
        )
    )
    question_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    question_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    z_blocks = torch.randn(2, 4, 3, 8)

    pred = model(question_ids, question_mask, z_blocks)
    rolled = rollout_prior(model, question_ids, question_mask)
    loss = model.loss(pred, z_blocks)

    assert pred.shape == z_blocks.shape
    assert rolled.shape == z_blocks.shape
    assert torch.isfinite(loss)


def test_block_causal_mask_hides_future_blocks():
    model = BlockCausalPrior(StageCPriorConfig(vocab_size=8, b_max=3, block_size=2, latent_dim=4, hidden_dim=8, num_heads=2))
    mask = model.block_causal_mask(torch.device("cpu"))

    assert mask.shape == (6, 6)
    assert mask[0, 2]
    assert not mask[2, 0]
    assert not mask[2, 3]
    assert mask[2, 4]


def test_scheduled_value_warmup_and_ramp():
    assert scheduled_value(10, final_value=0.1, warmup_steps=20, ramp_steps=100) == 0.0
    assert scheduled_value(70, final_value=0.1, warmup_steps=20, ramp_steps=100) == 0.05
    assert scheduled_value(200, final_value=0.1, warmup_steps=20, ramp_steps=100) == 0.1


def test_flow_interpolate_and_x_prediction_loss_match_elf_objective():
    clean = torch.tensor([[[[2.0], [4.0]]]])
    noise = torch.tensor([[[[0.0], [2.0]]]])
    times = torch.tensor([0.25])

    noisy = flow_interpolate(clean, noise, times)
    loss = flow_matching_loss(clean, clean, noisy, noise, times)

    assert torch.allclose(noisy, torch.tensor([[[[0.5], [2.5]]]]))
    assert torch.allclose(loss, torch.tensor(0.0))


def test_flow_training_masks_to_current_block_only():
    pred = torch.zeros(1, 2, 1, 1)
    clean = torch.tensor([[[[1.0]], [[10.0]]]])
    noise = torch.zeros_like(clean)
    current = torch.zeros_like(clean)
    times = torch.tensor([0.5])
    mask = torch.tensor([[True, False]])

    loss = masked_flow_matching_loss(pred, clean, current, noise, times, mask)

    assert torch.allclose(loss, torch.tensor(1.0))


def test_active_flow_block_sampling_respects_block_mask():
    torch.manual_seed(0)
    block_mask = torch.tensor([[False, True, False], [False, False, False]])

    selected = sample_active_flow_blocks(block_mask)
    mask = active_block_mask(selected, b_max=3)

    assert selected[0].item() == 1
    assert mask.shape == block_mask.shape
    assert mask.sum(dim=1).tolist() == [1, 1]


def test_timestep_embedding_matches_cola_sin_cos_order():
    timesteps = torch.tensor([1.0])
    embedding = sinusoidal_timestep_embedding(timesteps, 4)
    expected = torch.tensor([[math.sin(1.0), math.sin(0.01), math.cos(1.0), math.cos(0.01)]])

    assert torch.allclose(embedding, expected)


def test_block_causal_prior_accepts_flow_noisy_blocks_and_timesteps():
    model = BlockCausalPrior(
        StageCPriorConfig(
            vocab_size=16,
            b_max=2,
            block_size=2,
            latent_dim=4,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
        )
    )
    question_ids = torch.tensor([[1, 2, 0]])
    question_mask = torch.tensor([[1, 1, 0]])
    previous = torch.zeros(1, 2, 2, 4)
    noisy = torch.randn_like(previous)
    timesteps = torch.tensor([731.0])

    pred = model(question_ids, question_mask, previous, noisy_blocks=noisy, timesteps=timesteps)

    assert pred.shape == previous.shape


def test_blockwise_mse_reports_each_block():
    pred = torch.tensor([[[[1.0], [3.0]], [[2.0], [4.0]]]])
    target = torch.tensor([[[[0.0], [1.0]], [[2.0], [2.0]]]])

    mse = blockwise_mse(pred, target)

    assert torch.allclose(mse, torch.tensor([[2.5, 2.0]]))


def test_finalize_latent_stats_reports_value_and_norm_stats():
    stats = {
        "gold": {
            "sum": 0.0,
            "sumsq": 4.0,
            "numel": 4.0,
            "norm_sum": 4.0,
            "norm_sumsq": 8.0,
            "norm_count": 2.0,
        }
    }

    metrics = finalize_latent_stats(stats)

    assert metrics["gold_latent_mean"] == 0.0
    assert metrics["gold_latent_std"] == 1.0
    assert metrics["gold_latent_norm_mean"] == 2.0
    assert metrics["gold_latent_norm_std"] == 0.0


def test_stage_c_builds_stage_b_batch_from_latent_cache_rows():
    stage_b = make_small_stage_b()
    batch = make_stage_c_batch_for_joint()

    stage_b_batch = build_stage_b_batch_from_stage_c(
        stage_b, batch, tokenizer=FakeTokenizer()
    )

    assert stage_b_batch["target_input_ids"].shape == (1, 4)
    assert stage_b_batch["target_mask"].tolist() == [[1, 1, 0, 0]]
    assert stage_b_batch["answer_input_ids"].tolist() == [[6, 0]]
    assert stage_b_batch["block_mask"].tolist() == [[1, 0]]


def test_prepare_prior_training_batch_recomputes_joint_stage_b_latents():
    torch.manual_seed(0)
    stage_b = make_small_stage_b()
    batch = make_stage_c_batch_for_joint()
    config = StageCPriorTrainConfig(
        prior_mode="flow",
        joint_stage_b=True,
        stage_b_vae_weight=0.1,
        stage_b_ref_weight=0.1,
    )

    prior_batch, metrics = prepare_prior_training_batch(
        stage_b, batch, tokenizer=FakeTokenizer(), config=config
    )

    assert prior_batch["z_blocks"].shape == batch["z_blocks"].shape
    assert prior_batch["z_blocks"].requires_grad
    assert torch.isfinite(metrics["stage_b_joint_loss_tensor"])
    assert metrics["stage_b_joint_loss"] > 0


def test_joint_stage_b_requires_flow_prior_mode():
    with pytest.raises(ValueError, match="requires --prior-mode flow"):
        validate_prior_mode(StageCPriorTrainConfig(joint_stage_b=True))


def test_question_encoder_uses_token_order():
    torch.manual_seed(0)
    model = BlockCausalPrior(
        StageCPriorConfig(
            vocab_size=16,
            b_max=2,
            block_size=2,
            latent_dim=4,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            max_question_len=8,
        )
    )
    model.eval()
    question_ids = torch.tensor([[1, 2, 3], [3, 2, 1]])
    question_mask = torch.ones_like(question_ids)
    z_blocks = torch.zeros(2, 2, 2, 4)

    with torch.no_grad():
        pred = model(question_ids, question_mask, z_blocks)

    assert not torch.allclose(pred[0], pred[1])


def test_external_question_features_drive_prior():
    torch.manual_seed(0)
    model = BlockCausalPrior(
        StageCPriorConfig(
            vocab_size=16,
            b_max=2,
            block_size=2,
            latent_dim=4,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            condition_dim=6,
        )
    )
    model.eval()
    question_ids = torch.tensor([[1, 2, 0]])
    question_mask = torch.tensor([[1, 1, 0]])
    z_blocks = torch.zeros(1, 2, 2, 4)
    features_a = torch.zeros(1, 3, 6)
    features_b = torch.ones(1, 3, 6)

    with torch.no_grad():
        pred_a = model(question_ids, question_mask, z_blocks, features_a)
        pred_b = model(question_ids, question_mask, z_blocks, features_b)

    assert pred_a.shape == z_blocks.shape
    assert not torch.allclose(pred_a, pred_b)
