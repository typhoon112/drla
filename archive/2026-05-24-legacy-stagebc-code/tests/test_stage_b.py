from drla.data.answer_judge import normalize_answer
import torch

from drla.data.stage_b import (
    StageBCollator,
    StageBDataset,
    StageBExample,
    build_local_vocab,
)
from drla.models.stage_b import StageBModelConfig, StageBReasoningAutoencoder
from drla.training.stage_b_autoencoder import (
    StageBTrainConfig,
    primary_eval_metric_name,
    primary_eval_score,
    sequence_matches,
    weighted_loss,
)


def make_examples():
    return [
        StageBExample(
            id="a",
            question_ids=[10, 11, 12],
            target_ids=[20, 21, 22, 23, 24],
            answer_ids=[30],
            answer_norm="7",
            b_star=2,
            b_max=4,
            block_size=3,
        ),
        StageBExample(
            id="b",
            question_ids=[10, 13],
            target_ids=[25, 26],
            answer_ids=[31, 32],
            answer_norm="12",
            b_star=1,
            b_max=4,
            block_size=3,
        ),
    ]


def test_stage_b_collator_masks_and_vocab():
    examples = make_examples()
    vocab = build_local_vocab(examples)
    dataset = StageBDataset(examples, vocab_mapper=vocab, max_answer_len=2)
    collator = StageBCollator(pad_id=vocab.pad_id, b_max=4, block_size=3, max_answer_len=2)
    batch = collator([dataset[0], dataset[1]])

    assert batch["target_input_ids"].shape == (2, 12)
    assert batch["answer_input_ids"].shape == (2, 2)
    assert batch["block_mask"].tolist() == [[1, 1, 0, 0], [1, 0, 0, 0]]
    assert batch["noop_mask"].tolist() == [[0, 0, 1, 1], [0, 1, 1, 1]]
    assert vocab.decode(vocab.encode([10, 30])) == [10, 30]


def test_stage_b_model_forward_losses_are_finite():
    examples = make_examples()
    vocab = build_local_vocab(examples)
    dataset = StageBDataset(examples, vocab_mapper=vocab, max_answer_len=2)
    collator = StageBCollator(pad_id=vocab.pad_id, b_max=4, block_size=3, max_answer_len=2)
    batch = collator([dataset[0], dataset[1]])
    model = StageBReasoningAutoencoder(
        StageBModelConfig(
            vocab_size=vocab.vocab_size,
            b_max=4,
            block_size=3,
            latent_dim=8,
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            max_answer_len=2,
            dropout=0.0,
        )
    )

    outputs = model(batch)
    loss = weighted_loss(outputs, StageBTrainConfig())

    assert outputs["z_blocks"].shape == (2, 4, 3, 8)
    assert outputs["question_latent_pred"].shape == outputs["z_blocks"].shape
    assert outputs["token_logits"].shape[:2] == (2, 12)
    assert outputs["answer_logits"].shape[:2] == (2, 2)
    assert torch.isfinite(outputs["l_question_latent"])
    assert torch.isfinite(loss)


def test_stage_b_question_latent_weight_changes_loss():
    outputs = {
        "l_answer": torch.tensor(1.0),
        "l_recon": torch.tensor(2.0),
        "l_noop": torch.tensor(3.0),
        "l_kd": torch.tensor(4.0),
        "l_verifier": torch.tensor(5.0),
        "l_question_latent": torch.tensor(7.0),
    }

    base = weighted_loss(outputs, StageBTrainConfig(question_latent_weight=0.0))
    shaped = weighted_loss(outputs, StageBTrainConfig(question_latent_weight=0.5))

    assert torch.allclose(shaped - base, torch.tensor(3.5))


def test_sequence_matches_ignores_padding():
    pred = torch.tensor([[1, 2, 9], [3, 8, 0]])
    labels = torch.tensor([[1, 2, 0], [3, 4, 0]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)

    assert sequence_matches(pred, labels, mask) == 1


def test_normalize_answer_handles_very_large_integers():
    value = "999999999999999999999999999999999999999999999999999999999999"

    assert normalize_answer(value) == value


def test_stage_b_primary_eval_metric_tracks_gold_latent_decode():
    assert primary_eval_metric_name() == "recon_judge_acc"
    assert primary_eval_score({"recon_judge_acc": 0.75, "loss": 10.0}) == 0.75
    assert primary_eval_score({"loss": 0.25}) == -0.25
