from __future__ import annotations

import torch

from src.generation.rollout import sample_ghost_futures


class DummyModel(torch.nn.Module):
    def forward(self, batch):  # type: ignore[override]
        bsz = batch["cat_x"].shape[0]
        return {
            "action_logits": torch.randn(bsz, 6),
            "zone_logits": torch.randn(bsz, 96),
            "outcome_logit": torch.randn(bsz),
            "shot_next_5_logit": torch.randn(bsz),
            "turnover_next_3_logit": torch.randn(bsz),
        }


def test_rollout_shapes() -> None:
    model = DummyModel()
    history = {
        "cat_x": torch.zeros((4, 6), dtype=torch.long),
        "cont_x": torch.zeros((4, 35), dtype=torch.float32),
        "mask": torch.tensor([False, True, True, True], dtype=torch.bool),
    }
    vocabs = {"event_type": {"idx_to_token": ["PAD", "UNK", "Pass", "Carry", "Shot", "Duel"]}}
    config = {"generation": {"stop_on_terminal_event": True}}

    futures = sample_ghost_futures(
        model=model,
        initial_history=history,
        dataset_or_feature_builder=None,
        vocabs=vocabs,
        config=config,
        num_samples=2,
        rollout_steps=3,
        temperature=1.0,
        top_k=5,
        device=torch.device("cpu"),
    )
    assert len(futures) == 2
    assert "events" in futures[0]
    assert len(futures[0]["events"]) >= 1

