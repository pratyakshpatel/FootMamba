from __future__ import annotations

import torch
import torch.nn as nn


class MambaBackbone(nn.Module):
    def __init__(self, d_model: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on environment
            raise ImportError(f"Failed to import mamba_ssm: {exc}") from exc

        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.blocks = nn.ModuleList(
            [
                Mamba(
                    d_model=d_model,
                    d_state=16,
                    d_conv=4,
                    expand=2,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for norm, block in zip(self.norms, self.blocks):
            x = x + self.dropout(block(norm(x)))
        return x

