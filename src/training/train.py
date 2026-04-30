from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.metrics import compute_metrics
from src.utils.io import ensure_dir
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    grad_clip_norm: float = 1.0,
) -> dict[str, float]:
    model.train()
    losses = []
    records: dict[str, list[np.ndarray]] = {
        "action_logits": [],
        "zone_logits": [],
        "outcome_logit": [],
        "shot_next_5_logit": [],
        "turnover_next_3_logit": [],
        "target_action": [],
        "target_zone": [],
        "target_outcome": [],
        "target_shot_next_5": [],
        "target_turnover_next_3": [],
    }

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        outputs = model(batch)
        loss, _ = loss_fn(outputs, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        losses.append(float(loss.item()))
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        for k in ["action_logits", "zone_logits", "outcome_logit", "shot_next_5_logit", "turnover_next_3_logit"]:
            records[k].append(outputs[k].detach().cpu().numpy())
        for k in ["target_action", "target_zone", "target_outcome", "target_shot_next_5", "target_turnover_next_3"]:
            records[k].append(batch[k].detach().cpu().numpy())

    metrics = compute_metrics(records)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses = []
    records: dict[str, list[np.ndarray]] = {
        "action_logits": [],
        "zone_logits": [],
        "outcome_logit": [],
        "shot_next_5_logit": [],
        "turnover_next_3_logit": [],
        "target_action": [],
        "target_zone": [],
        "target_outcome": [],
        "target_shot_next_5": [],
        "target_turnover_next_3": [],
    }

    for batch in tqdm(loader, desc="val", leave=False):
        batch = _move_batch(batch, device)
        outputs = model(batch)
        loss, _ = loss_fn(outputs, batch)
        losses.append(float(loss.item()))
        for k in ["action_logits", "zone_logits", "outcome_logit", "shot_next_5_logit", "turnover_next_3_logit"]:
            records[k].append(outputs[k].detach().cpu().numpy())
        for k in ["target_action", "target_zone", "target_outcome", "target_shot_next_5", "target_turnover_next_3"]:
            records[k].append(batch[k].detach().cpu().numpy())

    metrics = compute_metrics(records)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    vocabs: dict[str, Any],
    path: str | Path,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "vocabs": vocabs,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=device)


def fit_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    config: dict[str, Any],
    vocabs: dict[str, Any],
    device: torch.device,
    output_path: str | Path,
) -> dict[str, Any]:
    epochs = int(config["training"]["epochs"])
    grad_clip_norm = float(config["training"].get("grad_clip_norm", 1.0))

    best_val = float("inf")
    best_metrics: dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_clip_norm=grad_clip_norm,
        )
        val_metrics = validate(model=model, loader=val_loader, loss_fn=loss_fn, device=device)
        LOGGER.info(
            "Epoch %d | train_loss=%.4f val_loss=%.4f action_acc=%.4f",
            epoch,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["action_accuracy"],
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_metrics = val_metrics
            best_path = Path(output_path).with_name(Path(output_path).stem + "_best.pt")
            save_checkpoint(model, optimizer, config, vocabs, best_path)

    save_checkpoint(model, optimizer, config, vocabs, output_path)
    return {"best_val_loss": best_val, "best_val_metrics": best_metrics}

