from __future__ import annotations

import torch

from src.models.ghost_model import GhostFutureModel


def test_model_forward_gru() -> None:
    vocab_sizes = [10, 5, 5, 4, 3, 6]
    model = GhostFutureModel(
        vocab_sizes=vocab_sizes,
        num_continuous=12,
        num_action_types=10,
        model_cfg={
            "backend": "gru",
            "d_model": 32,
            "num_layers": 1,
            "dropout": 0.1,
            "categorical_embedding_dim": 8,
            "continuous_hidden_dim": 16,
        },
    )
    cat_x = torch.zeros((2, 4, 6), dtype=torch.long)
    for i, size in enumerate(vocab_sizes):
        cat_x[:, :, i] = torch.randint(0, size, (2, 4), dtype=torch.long)
    batch = {
        "cat_x": cat_x,
        "cont_x": torch.randn(2, 4, 12),
        "mask": torch.tensor([[False, True, True, True], [True, True, True, True]], dtype=torch.bool),
    }
    out = model(batch)
    assert set(out.keys()) == {
        "action_logits",
        "zone_logits",
        "outcome_logit",
        "shot_next_5_logit",
        "turnover_next_3_logit",
    }
    assert out["action_logits"].shape == (2, 10)
    assert out["zone_logits"].shape == (2, 96)
