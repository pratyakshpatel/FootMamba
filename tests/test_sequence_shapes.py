from __future__ import annotations

import pandas as pd

from src.data.build_sequences import FootballSequenceDataset, add_future_labels
from src.data.vocab import build_vocabs


def _toy_events() -> pd.DataFrame:
    rows = []
    for i in range(8):
        rows.append(
            {
                "match_id": 1,
                "event_id": f"e{i}",
                "event_index": i,
                "period": 1,
                "time_seconds": float(i),
                "minute": 0,
                "second": i,
                "possession": 1,
                "team_name": "A",
                "player_name": "P1",
                "event_type": "Pass" if i % 2 == 0 else "Carry",
                "play_pattern": "Regular Play",
                "body_part": "Foot",
                "outcome_name": "Complete",
                "under_pressure": 0,
                "x": 60.0,
                "y": 40.0,
                "end_x": 62.0,
                "end_y": 42.0,
                "delta_x": 2.0,
                "delta_y": 2.0,
                "distance_moved": 2.8,
                "time_since_prev": 1.0,
                "same_team_as_prev": 1,
                "possession_changed": 0,
                "has_360": 0,
                "num_visible_players": 0,
                "num_visible_teammates": 0,
                "num_visible_opponents": 0,
                "num_visible_keepers": 0,
                "num_visible_actors": 0,
                "nearest_teammate_distance_to_ball": 999.0,
                "nearest_opponent_distance_to_ball": 999.0,
                "mean_teammate_distance_to_ball": 999.0,
                "mean_opponent_distance_to_ball": 999.0,
                "opponents_within_5": 0,
                "opponents_within_10": 0,
                "teammates_within_10": 0,
                "teammates_ahead_of_ball": 0,
                "opponents_ahead_of_ball": 0,
                "opponents_between_ball_and_goal": 0,
                "teammate_width": 0.0,
                "opponent_width": 0.0,
                "teammate_depth": 0.0,
                "opponent_depth": 0.0,
                "visible_area_size": 0.0,
                "visible_area_fraction": 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_sequence_shapes() -> None:
    events = add_future_labels(_toy_events(), future_horizon=5, turnover_horizon=3)
    vocabs = build_vocabs(
        events,
        ["event_type", "team_name", "player_name", "play_pattern", "body_part", "outcome_name"],
    )
    ds = FootballSequenceDataset(events_df=events, vocabs=vocabs, context_length=4)
    sample = ds[3]
    assert sample["cat_x"].shape[0] == 4
    assert sample["cont_x"].shape[0] == 4
    assert sample["mask"].shape[0] == 4
    assert sample["target_action"].ndim == 0
    assert sample["target_zone"].ndim == 0

