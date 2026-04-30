from __future__ import annotations

import torch
import torch.nn as nn


class EventEncoder(nn.Module):
    def __init__(
        self,
        vocab_sizes: list[int],
        num_continuous: int,
        d_model: int,
        categorical_embedding_dim: int,
        continuous_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=size,
                    embedding_dim=categorical_embedding_dim,
                    padding_idx=0,
                )
                for size in vocab_sizes
            ]
        )
        self.cont_mlp = nn.Sequential(
            nn.Linear(num_continuous, continuous_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(continuous_hidden_dim, continuous_hidden_dim),
            nn.ReLU(),
        )
        total_dim = len(vocab_sizes) * categorical_embedding_dim + continuous_hidden_dim
        self.proj = nn.Linear(total_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cat_x: torch.Tensor, cont_x: torch.Tensor) -> torch.Tensor:
        # cat_x: [B, K, C_cat], cont_x: [B, K, C_cont]
        cat_embs = []
        for i, emb in enumerate(self.embeddings):
            cat_embs.append(emb(cat_x[:, :, i]))
        cat_concat = torch.cat(cat_embs, dim=-1)

        cont_x = torch.nan_to_num(cont_x, nan=0.0, posinf=0.0, neginf=0.0)
        cont_proj = self.cont_mlp(cont_x)
        x = torch.cat([cat_concat, cont_proj], dim=-1)
        x = self.proj(x)
        return self.dropout(x)

