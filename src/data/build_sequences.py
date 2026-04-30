from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.vocab import Vocab

CATEGORICAL_COLUMNS = [
    "event_type",
    "team_name",
    "player_name",
    "play_pattern",
    "body_part",
    "outcome_name",
]

CONTINUOUS_COLUMNS = [
    "x",
    "y",
    "end_x",
    "end_y",
    "delta_x",
    "delta_y",
    "distance_moved",
    "minute",
    "second",
    "time_since_prev",
    "under_pressure",
    "same_team_as_prev",
    "possession_changed",
    "has_360",
    "num_visible_players",
    "num_visible_teammates",
    "num_visible_opponents",
    "num_visible_keepers",
    "num_visible_actors",
    "nearest_teammate_distance_to_ball",
    "nearest_opponent_distance_to_ball",
    "mean_teammate_distance_to_ball",
    "mean_opponent_distance_to_ball",
    "opponents_within_5",
    "opponents_within_10",
    "teammates_within_10",
    "teammates_ahead_of_ball",
    "opponents_ahead_of_ball",
    "opponents_between_ball_and_goal",
    "teammate_width",
    "opponent_width",
    "teammate_depth",
    "opponent_depth",
    "visible_area_size",
    "visible_area_fraction",
]


def outcome_to_binary(outcome: str, event_type: str = "") -> int:
    o = (outcome or "").lower()
    e = (event_type or "").lower()
    fail_keywords = ["incomplete", "out", "lost", "offside", "blocked", "saved", "failure"]
    success_keywords = ["complete", "success", "goal", "won"]
    if any(k in o for k in fail_keywords):
        return 0
    if any(k in o for k in success_keywords):
        return 1
    if o in {"unknown", ""} and ("pass" in e or "carry" in e or "dribble" in e or "ball receipt" in e):
        return 1
    # Unknown outcome is common in StatsBomb for non-failure events; avoid marking unknown as fail.
    return 1


def zone_id_from_xy(
    x: float,
    y: float,
    x_bins: int = 12,
    y_bins: int = 8,
    pitch_length: float = 120.0,
    pitch_width: float = 80.0,
) -> int:
    if not np.isfinite(x) or not np.isfinite(y):
        # Center-ish fallback (6,4) for a 12x8 grid.
        return min(x_bins * y_bins - 1, (y_bins // 2) * x_bins + (x_bins // 2))
    zone_x = int(np.clip((x / pitch_length) * x_bins, 0, x_bins - 1))
    zone_y = int(np.clip((y / pitch_width) * y_bins, 0, y_bins - 1))
    return int(zone_y * x_bins + zone_x)


def add_future_labels(
    events_df: pd.DataFrame,
    future_horizon: int = 5,
    turnover_horizon: int = 3,
) -> pd.DataFrame:
    events = events_df.copy()
    events["target_outcome_binary"] = [
        outcome_to_binary(o, e) for o, e in zip(events["outcome_name"], events["event_type"])
    ]
    events["target_zone_id"] = [
        zone_id_from_xy(x, y) for x, y in zip(events["x"], events["y"])
    ]

    shot_next = np.zeros(len(events), dtype=int)
    turnover_next = np.zeros(len(events), dtype=int)

    for _, idxs in events.groupby("match_id", sort=False).groups.items():
        idxs_list = list(idxs)
        sub = events.loc[idxs_list].reset_index()
        for i in range(len(sub)):
            origin_idx = int(sub.loc[i, "index"])
            own_team = str(sub.loc[i, "team_name"])
            own_possession = sub.loc[i, "possession"]

            look_end = min(len(sub), i + future_horizon + 1)
            look_df = sub.iloc[i:look_end]
            if pd.notna(own_possession):
                look_df = look_df[(look_df["possession"] == own_possession) | (look_df.index == i)]
            shot_next[origin_idx] = int(
                any("shot" in str(et).lower() for et in look_df["event_type"].tolist())
            )

            turn_end = min(len(sub), i + turnover_horizon + 1)
            turn_df = sub.iloc[i + 1 : turn_end]
            team_change = any(str(t) != own_team for t in turn_df["team_name"].tolist())
            outcome_fail = any(
                any(k in str(o).lower() for k in ["incomplete", "lost", "offside", "out", "failure"])
                for o in turn_df["outcome_name"].tolist()
            )
            turnover_next[origin_idx] = int(team_change or outcome_fail)

    events["target_shot_next_5_binary"] = shot_next
    events["target_turnover_next_3_binary"] = turnover_next
    return events


@dataclass
class DatasetMetadata:
    categorical_columns: list[str]
    continuous_columns: list[str]
    context_length: int


class FootballSequenceDataset(Dataset):
    def __init__(
        self,
        events_df: pd.DataFrame,
        vocabs: dict[str, Vocab],
        context_length: int,
        categorical_columns: list[str] | None = None,
        continuous_columns: list[str] | None = None,
        match_ids: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.categorical_columns = categorical_columns or CATEGORICAL_COLUMNS
        self.continuous_columns = continuous_columns or CONTINUOUS_COLUMNS
        self.context_length = int(context_length)
        self.vocabs = vocabs

        df = events_df.copy()
        if match_ids is not None:
            df = df[df["match_id"].isin(match_ids)].copy()
        df = df.sort_values(["match_id", "period", "time_seconds", "event_index"]).reset_index(drop=True)
        self.df = df

        # Categorical encodings
        cat_arrays = []
        for col in self.categorical_columns:
            vocab = vocabs[col]
            values = df[col].fillna("unknown").astype(str).tolist()
            cat_arrays.append([vocab.encode(v) for v in values])
        self.cat_matrix = np.stack(cat_arrays, axis=1).astype(np.int64)

        # Continuous features
        for col in self.continuous_columns:
            if col not in df.columns:
                df[col] = 0.0
        cont = df[self.continuous_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

        # Basic scaling for stable training.
        if "x" in cont.columns:
            cont["x"] /= 120.0
        if "y" in cont.columns:
            cont["y"] /= 80.0
        if "end_x" in cont.columns:
            cont["end_x"] /= 120.0
        if "end_y" in cont.columns:
            cont["end_y"] /= 80.0
        if "minute" in cont.columns:
            cont["minute"] /= 130.0
        if "second" in cont.columns:
            cont["second"] /= 60.0

        self.cont_matrix = cont.to_numpy(dtype=np.float32)

        self.target_action = np.array(self.cat_matrix[:, 0], dtype=np.int64)
        self.target_zone = np.array(df["target_zone_id"].fillna(0).astype(int).tolist(), dtype=np.int64)
        self.target_outcome = np.array(df["target_outcome_binary"].fillna(0).astype(float).tolist(), dtype=np.float32)
        self.target_shot5 = np.array(df["target_shot_next_5_binary"].fillna(0).astype(float).tolist(), dtype=np.float32)
        self.target_turn3 = np.array(
            df["target_turnover_next_3_binary"].fillna(0).astype(float).tolist(), dtype=np.float32
        )

        self.match_ids_arr = df["match_id"].astype(int).to_numpy()
        self.event_ids_arr = df["event_id"].astype(str).to_numpy()

        self.match_to_positions: dict[int, list[int]] = {}
        self.pos_to_rank: dict[int, int] = {}
        for match_id, idxs in df.groupby("match_id", sort=False).groups.items():
            positions = sorted(int(i) for i in idxs)
            self.match_to_positions[int(match_id)] = positions
            for rank, pos in enumerate(positions):
                self.pos_to_rank[pos] = rank

        self.target_positions = list(range(len(df)))

    def __len__(self) -> int:
        return len(self.target_positions)

    def _build_context(self, pos: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cat_context = np.zeros((self.context_length, self.cat_matrix.shape[1]), dtype=np.int64)
        cont_context = np.zeros((self.context_length, self.cont_matrix.shape[1]), dtype=np.float32)
        mask = np.zeros((self.context_length,), dtype=bool)

        match_id = int(self.match_ids_arr[pos])
        positions = self.match_to_positions[match_id]
        rank = self.pos_to_rank[pos]
        history_positions = positions[max(0, rank - self.context_length) : rank]

        if history_positions:
            start = self.context_length - len(history_positions)
            cat_context[start:] = self.cat_matrix[history_positions]
            cont_context[start:] = self.cont_matrix[history_positions]
            mask[start:] = True

        return (
            torch.tensor(cat_context, dtype=torch.long),
            torch.tensor(cont_context, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool),
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pos = self.target_positions[idx]
        cat_x, cont_x, mask = self._build_context(pos)
        return {
            "cat_x": cat_x,
            "cont_x": cont_x,
            "mask": mask,
            "target_action": torch.tensor(self.target_action[pos], dtype=torch.long),
            "target_zone": torch.tensor(self.target_zone[pos], dtype=torch.long),
            "target_outcome": torch.tensor(self.target_outcome[pos], dtype=torch.float32),
            "target_shot_next_5": torch.tensor(self.target_shot5[pos], dtype=torch.float32),
            "target_turnover_next_3": torch.tensor(self.target_turn3[pos], dtype=torch.float32),
            "meta_match_id": torch.tensor(int(self.match_ids_arr[pos]), dtype=torch.long),
            "meta_event_id": self.event_ids_arr[pos],
        }


def build_datasets_for_splits(
    events_df: pd.DataFrame,
    vocabs: dict[str, Vocab],
    splits: dict[str, list[int]],
    context_length: int,
) -> dict[str, FootballSequenceDataset]:
    return {
        split_name: FootballSequenceDataset(
            events_df=events_df,
            vocabs=vocabs,
            context_length=context_length,
            match_ids=match_ids,
        )
        for split_name, match_ids in splits.items()
    }
