from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.models.losses import MultiTaskLoss
from src.training.train import validate
from src.utils.io import ensure_dir, save_json


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_cfg: dict[str, float],
    device: torch.device,
) -> dict[str, float]:
    loss_fn = MultiTaskLoss(loss_cfg)
    return validate(model=model, loader=loader, loss_fn=loss_fn, device=device)


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    save_json({k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()}, path)

