from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.utils.geometry import euclidean_distance
from src.utils.statsbomb_safe import (
    safe_body_part,
    safe_end_location,
    safe_event_type,
    safe_location,
    safe_nested_name,
    safe_outcome,
    safe_play_pattern,
    safe_player,
    safe_team,
    safe_under_pressure,
    timestamp_to_seconds,
)

TERMINAL_KEYWORDS = {
    "shot",
    "goal",
    "foul committed",
    "foul won",
    "offside",
    "half end",
    "injury stoppage",
    "substitution",
    "tactical shift",
}


def _pick_event_subtype(row: pd.Series) -> str:
    for candidate in [
        "pass_type",
        "shot_type",
        "duel_type",
        "goalkeeper_type",
        "clearance_body_part",
    ]:
        if candidate in row.index:
            value = row.get(candidate)
            if isinstance(value, dict):
                name = value.get("name")
                if name:
                    return str(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return "unknown"


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(value)
    except Exception:
        return default


def build_events_dataframe(raw_events: pd.DataFrame) -> pd.DataFrame:
    if raw_events.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for idx, row in raw_events.iterrows():
        event_type = safe_event_type(row)
        event_type_l = event_type.lower()

        x, y = safe_location(row, "location")
        end_x, end_y = safe_end_location(row)
        if not np.isfinite(end_x) or not np.isfinite(end_y):
            end_x, end_y = x, y

        outcome = safe_outcome(row)
        match_id = row.get("match_id")
        if pd.isna(match_id):
            match_id = row.get("game_id", -1)

        rec = {
            "match_id": _to_int(match_id, -1),
            "event_id": str(row.get("id", f"event_{idx}")),
            "event_index": _to_int(row.get("index"), idx),
            "period": _to_int(row.get("period"), 0),
            "timestamp": str(row.get("timestamp", "")),
            "minute": _to_int(row.get("minute"), 0),
            "second": _to_int(row.get("second"), 0),
            "time_seconds": float(timestamp_to_seconds(row)),
            "possession": _to_int(row.get("possession"), -1),
            "team_name": safe_team(row),
            "player_name": safe_player(row),
            "event_type": event_type,
            "event_subtype": _pick_event_subtype(row),
            "play_pattern": safe_play_pattern(row),
            "body_part": safe_body_part(row),
            "outcome_name": outcome,
            "under_pressure": safe_under_pressure(row),
            "x": x,
            "y": y,
            "end_x": end_x,
            "end_y": end_y,
            "is_shot": int("shot" in event_type_l),
            "is_pass": int("pass" in event_type_l),
            "is_carry": int("carry" in event_type_l or "dribble" in event_type_l),
            "is_duel": int("duel" in event_type_l),
            "is_foul": int("foul" in event_type_l),
            "is_clearance": int("clearance" in event_type_l),
            "is_interception": int("interception" in event_type_l),
            "is_terminal_event": int(any(k in event_type_l for k in TERMINAL_KEYWORDS)),
        }
        rows.append(rec)

    events = pd.DataFrame(rows)
    events["x"] = events["x"].clip(0, 120)
    events["y"] = events["y"].clip(0, 80)
    events["end_x"] = events["end_x"].clip(0, 120)
    events["end_y"] = events["end_y"].clip(0, 80)

    events = events.sort_values(
        by=["match_id", "period", "time_seconds", "event_index"], kind="mergesort"
    ).reset_index(drop=True)

    events["delta_x"] = events["end_x"] - events["x"]
    events["delta_y"] = events["end_y"] - events["y"]
    events["distance_moved"] = [
        euclidean_distance(x1, y1, x2, y2)
        for x1, y1, x2, y2 in zip(events["x"], events["y"], events["end_x"], events["end_y"])
    ]

    events["prev_time_seconds"] = events.groupby("match_id")["time_seconds"].shift(1)
    events["time_since_prev"] = (events["time_seconds"] - events["prev_time_seconds"]).fillna(0.0)
    events["time_since_prev"] = events["time_since_prev"].clip(lower=0.0)

    prev_team = events.groupby("match_id")["team_name"].shift(1)
    prev_possession = events.groupby("match_id")["possession"].shift(1)
    events["same_team_as_prev"] = (events["team_name"] == prev_team).astype(int)
    events["possession_changed"] = (events["possession"] != prev_possession).fillna(0).astype(int)

    events = events.drop(columns=["prev_time_seconds"])
    return events

