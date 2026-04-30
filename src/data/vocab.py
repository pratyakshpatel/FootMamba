from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.io import save_json


PAD_TOKEN = "PAD"
UNK_TOKEN = "UNK"


@dataclass
class Vocab:
    token_to_idx: dict[str, int]
    idx_to_token: list[str]

    @classmethod
    def build(cls, values: list[str]) -> "Vocab":
        uniq = sorted(set(v for v in values if isinstance(v, str) and v))
        idx_to_token = [PAD_TOKEN, UNK_TOKEN] + uniq
        token_to_idx = {tok: i for i, tok in enumerate(idx_to_token)}
        return cls(token_to_idx=token_to_idx, idx_to_token=idx_to_token)

    def encode(self, token: str) -> int:
        return self.token_to_idx.get(token, self.token_to_idx[UNK_TOKEN])

    def decode(self, idx: int) -> str:
        if 0 <= idx < len(self.idx_to_token):
            return self.idx_to_token[idx]
        return UNK_TOKEN

    def to_dict(self) -> dict[str, object]:
        return {"token_to_idx": self.token_to_idx, "idx_to_token": self.idx_to_token}


def build_vocabs(events_df: pd.DataFrame, categorical_columns: list[str]) -> dict[str, Vocab]:
    vocabs: dict[str, Vocab] = {}
    for col in categorical_columns:
        values = events_df[col].fillna("unknown").astype(str).tolist() if col in events_df.columns else []
        vocabs[col] = Vocab.build(values)
    return vocabs


def vocabs_to_serializable(vocabs: dict[str, Vocab]) -> dict[str, dict[str, object]]:
    return {k: v.to_dict() for k, v in vocabs.items()}


def load_vocabs_dict(data: dict[str, dict[str, object]]) -> dict[str, Vocab]:
    out: dict[str, Vocab] = {}
    for col, v in data.items():
        token_to_idx = {str(k): int(i) for k, i in dict(v["token_to_idx"]).items()}
        idx_to_token = [str(x) for x in list(v["idx_to_token"])]
        out[col] = Vocab(token_to_idx=token_to_idx, idx_to_token=idx_to_token)
    return out


def save_vocabs(vocabs: dict[str, Vocab], path: str | Path) -> None:
    save_json(vocabs_to_serializable(vocabs), path)

