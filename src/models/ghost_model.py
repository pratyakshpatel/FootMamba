from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from src.models.event_encoder import EventEncoder
from src.models.gru_baseline import GRUBackbone
from src.models.heads import PredictionHeads
from src.models.mamba_model import MambaBackbone


class GhostFutureModel(nn.Module):
    def __init__(
        self,
        vocab_sizes: list[int],
        num_continuous: int,
        num_action_types: int,
        model_cfg: dict[str, object],
    ) -> None:
        super().__init__()
        d_model = int(model_cfg["d_model"])
        num_layers = int(model_cfg["num_layers"])
        dropout = float(model_cfg.get("dropout", 0.1))
        emb_dim = int(model_cfg.get("categorical_embedding_dim", 32))
        cont_hidden = int(model_cfg.get("continuous_hidden_dim", 128))
        backend = str(model_cfg.get("backend", "auto")).lower()

        self.encoder = EventEncoder(
            vocab_sizes=vocab_sizes,
            num_continuous=num_continuous,
            d_model=d_model,
            categorical_embedding_dim=emb_dim,
            continuous_hidden_dim=cont_hidden,
            dropout=dropout,
        )

        if backend == "mamba":
            self.backbone = MambaBackbone(d_model=d_model, num_layers=num_layers, dropout=dropout)
            self.backend = "mamba"
        elif backend == "gru":
            self.backbone = GRUBackbone(d_model=d_model, num_layers=num_layers, dropout=dropout)
            self.backend = "gru"
        else:
            try:
                self.backbone = MambaBackbone(d_model=d_model, num_layers=num_layers, dropout=dropout)
                self.backend = "mamba"
            except Exception:
                warnings.warn("Mamba backend unavailable; falling back to GRU.", RuntimeWarning, stacklevel=2)
                self.backbone = GRUBackbone(d_model=d_model, num_layers=num_layers, dropout=dropout)
                self.backend = "gru"

        self.heads = PredictionHeads(d_model=d_model, num_action_types=num_action_types, num_zones=96)

    @staticmethod
    def _last_valid(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # hidden: [B,K,D], mask: [B,K]
        bsz = hidden.size(0)
        lengths = mask.long().sum(dim=1).clamp(min=1)
        idx = lengths - 1
        return hidden[torch.arange(bsz, device=hidden.device), idx]

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cat_x = batch["cat_x"]
        cont_x = batch["cont_x"]
        mask = batch["mask"]
        x = self.encoder(cat_x, cont_x)
        hidden = self.backbone(x, mask=mask)
        last_hidden = self._last_valid(hidden, mask)
        return self.heads(last_hidden)
