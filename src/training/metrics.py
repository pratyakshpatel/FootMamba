from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def compute_metrics(batch_records: dict[str, list[np.ndarray]]) -> dict[str, float]:
    action_logits = np.concatenate(batch_records["action_logits"], axis=0)
    zone_logits = np.concatenate(batch_records["zone_logits"], axis=0)
    outcome_logit = np.concatenate(batch_records["outcome_logit"], axis=0)
    shot_logit = np.concatenate(batch_records["shot_next_5_logit"], axis=0)
    turn_logit = np.concatenate(batch_records["turnover_next_3_logit"], axis=0)

    action_true = np.concatenate(batch_records["target_action"], axis=0)
    zone_true = np.concatenate(batch_records["target_zone"], axis=0)
    outcome_true = np.concatenate(batch_records["target_outcome"], axis=0)
    shot_true = np.concatenate(batch_records["target_shot_next_5"], axis=0)
    turn_true = np.concatenate(batch_records["target_turnover_next_3"], axis=0)

    action_pred = action_logits.argmax(axis=1)
    zone_pred = zone_logits.argmax(axis=1)

    top3 = np.argsort(-action_logits, axis=1)[:, :3]
    top3_acc = float(np.mean([t in row for t, row in zip(action_true, top3)]))

    outcome_prob = 1.0 / (1.0 + np.exp(-outcome_logit))
    shot_prob = 1.0 / (1.0 + np.exp(-shot_logit))
    turn_prob = 1.0 / (1.0 + np.exp(-turn_logit))

    metrics = {
        "action_accuracy": float(np.mean(action_pred == action_true)),
        "action_top3_accuracy": top3_acc,
        "zone_accuracy": float(np.mean(zone_pred == zone_true)),
        "outcome_accuracy": float(np.mean((outcome_prob >= 0.5).astype(int) == outcome_true.astype(int))),
        "shot_next_5_auc": _safe_auc(shot_true, shot_prob),
        "turnover_next_3_auc": _safe_auc(turn_true, turn_prob),
        "outcome_brier": float(np.mean((outcome_prob - outcome_true) ** 2)),
        "shot_next_5_brier": float(np.mean((shot_prob - shot_true) ** 2)),
        "turnover_next_3_brier": float(np.mean((turn_prob - turn_true) ** 2)),
    }
    return metrics

