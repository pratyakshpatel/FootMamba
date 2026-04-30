from __future__ import annotations

from typing import Any

import torch

from src.data.build_sequences import CATEGORICAL_COLUMNS, CONTINUOUS_COLUMNS
from src.generation.constraints import apply_top_k, categorize_future, is_terminal_event, zone_id_to_center


def _sample_from_logits(logits: torch.Tensor, temperature: float, top_k: int) -> tuple[int, torch.Tensor]:
    logits = logits / max(temperature, 1e-6)
    logits = apply_top_k(logits, top_k)
    probs = torch.softmax(logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1).item()
    return int(idx), probs


def _append_synthetic_step(
    cat_seq: torch.Tensor,
    cont_seq: torch.Tensor,
    mask_seq: torch.Tensor,
    action_id: int,
    x: float,
    y: float,
    action_col: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cat_next = cat_seq.clone()
    cont_next = cont_seq.clone()
    mask_next = mask_seq.clone()

    cat_next = torch.roll(cat_next, shifts=-1, dims=0)
    cont_next = torch.roll(cont_next, shifts=-1, dims=0)
    mask_next = torch.roll(mask_next, shifts=-1, dims=0)

    cat_next[-1] = 0
    cat_next[-1, action_col] = action_id

    cont_next[-1] = 0.0
    # x, y, end_x, end_y indices in CONTINUOUS_COLUMNS
    x_idx = CONTINUOUS_COLUMNS.index("x")
    y_idx = CONTINUOUS_COLUMNS.index("y")
    ex_idx = CONTINUOUS_COLUMNS.index("end_x")
    ey_idx = CONTINUOUS_COLUMNS.index("end_y")
    cont_next[-1, x_idx] = x / 120.0
    cont_next[-1, y_idx] = y / 80.0
    cont_next[-1, ex_idx] = x / 120.0
    cont_next[-1, ey_idx] = y / 80.0
    mask_next[-1] = True
    return cat_next, cont_next, mask_next


@torch.no_grad()
def sample_ghost_futures(
    model: torch.nn.Module,
    initial_history: dict[str, torch.Tensor],
    dataset_or_feature_builder: Any,
    vocabs: dict[str, Any],
    config: dict[str, Any],
    num_samples: int,
    rollout_steps: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    action_vocab = vocabs["event_type"]["idx_to_token"] if isinstance(vocabs["event_type"], dict) else vocabs["event_type"].idx_to_token

    cat_x = initial_history["cat_x"].to(device)
    cont_x = initial_history["cont_x"].to(device)
    mask = initial_history["mask"].to(device)

    futures: list[dict[str, Any]] = []
    for _ in range(num_samples):
        cseq = cat_x.clone()
        qseq = cont_x.clone()
        mseq = mask.clone()
        events: list[dict[str, Any]] = []

        for step in range(1, rollout_steps + 1):
            batch = {
                "cat_x": cseq.unsqueeze(0),
                "cont_x": qseq.unsqueeze(0),
                "mask": mseq.unsqueeze(0),
            }
            out = model(batch)
            action_id, _ = _sample_from_logits(out["action_logits"][0], temperature=temperature, top_k=top_k)
            zone_id, _ = _sample_from_logits(out["zone_logits"][0], temperature=temperature, top_k=top_k)
            outcome_prob = torch.sigmoid(out["outcome_logit"][0]).item()
            shot_prob = torch.sigmoid(out["shot_next_5_logit"][0]).item()
            turnover_prob = torch.sigmoid(out["turnover_next_3_logit"][0]).item()

            action_name = action_vocab[action_id] if action_id < len(action_vocab) else "UNK"
            x, y = zone_id_to_center(zone_id)
            event = {
                "step": step,
                "event_type": action_name,
                "zone_id": int(zone_id),
                "x": float(x),
                "y": float(y),
                "outcome_prob": float(outcome_prob),
                "shot_next_5_prob": float(shot_prob),
                "turnover_next_3_prob": float(turnover_prob),
            }
            events.append(event)

            cseq, qseq, mseq = _append_synthetic_step(cseq, qseq, mseq, action_id=action_id, x=x, y=y)
            if config["generation"].get("stop_on_terminal_event", True) and is_terminal_event(action_name):
                break

        futures.append({"events": events, "ending_category": categorize_future(events)})

    return futures

