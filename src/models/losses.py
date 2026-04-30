from __future__ import annotations

import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    def __init__(self, cfg: dict[str, float]) -> None:
        super().__init__()
        self.action_w = float(cfg.get("action_weight", 1.0))
        self.zone_w = float(cfg.get("zone_weight", 1.0))
        self.outcome_w = float(cfg.get("outcome_weight", 0.5))
        self.shot_w = float(cfg.get("shot_next_5_weight", 0.8))
        self.turn_w = float(cfg.get("turnover_next_3_weight", 0.8))

        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        action_loss = self.ce(outputs["action_logits"], batch["target_action"])
        zone_loss = self.ce(outputs["zone_logits"], batch["target_zone"])
        outcome_loss = self.bce(outputs["outcome_logit"], batch["target_outcome"])
        shot_loss = self.bce(outputs["shot_next_5_logit"], batch["target_shot_next_5"])
        turn_loss = self.bce(outputs["turnover_next_3_logit"], batch["target_turnover_next_3"])

        total = (
            self.action_w * action_loss
            + self.zone_w * zone_loss
            + self.outcome_w * outcome_loss
            + self.shot_w * shot_loss
            + self.turn_w * turn_loss
        )
        components = {
            "action_loss": float(action_loss.detach().cpu().item()),
            "zone_loss": float(zone_loss.detach().cpu().item()),
            "outcome_loss": float(outcome_loss.detach().cpu().item()),
            "shot_loss": float(shot_loss.detach().cpu().item()),
            "turnover_loss": float(turn_loss.detach().cpu().item()),
            "total_loss": float(total.detach().cpu().item()),
        }
        return total, components
