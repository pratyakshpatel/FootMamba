from __future__ import annotations

import torch
import torch.nn as nn


class PredictionHeads(nn.Module):
    def __init__(self, d_model: int, num_action_types: int, num_zones: int = 96) -> None:
        super().__init__()
        self.action_head = nn.Linear(d_model, num_action_types)
        self.zone_head = nn.Linear(d_model, num_zones)
        self.outcome_head = nn.Linear(d_model, 1)
        self.shot_next_5_head = nn.Linear(d_model, 1)
        self.turnover_next_3_head = nn.Linear(d_model, 1)

    def forward(self, h_last: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "action_logits": self.action_head(h_last),
            "zone_logits": self.zone_head(h_last),
            "outcome_logit": self.outcome_head(h_last).squeeze(-1),
            "shot_next_5_logit": self.shot_next_5_head(h_last).squeeze(-1),
            "turnover_next_3_logit": self.turnover_next_3_head(h_last).squeeze(-1),
        }

