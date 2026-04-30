from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.geometry import euclidean_distance, parse_visible_area, polygon_area

COUNT_COLUMNS = [
    "num_visible_players",
    "num_visible_teammates",
    "num_visible_opponents",
    "num_visible_keepers",
    "num_visible_actors",
    "opponents_within_5",
    "opponents_within_10",
    "teammates_within_10",
    "teammates_ahead_of_ball",
    "opponents_ahead_of_ball",
    "opponents_between_ball_and_goal",
]

DIST_COLUMNS = [
    "nearest_teammate_distance_to_ball",
    "nearest_opponent_distance_to_ball",
    "mean_teammate_distance_to_ball",
    "mean_opponent_distance_to_ball",
]

AREA_COLUMNS = ["visible_area_size", "visible_area_fraction"]


def _event_id_column(frames_df: pd.DataFrame) -> str | None:
    for c in ["event_id", "event_uuid", "id", "event"]:
        if c in frames_df.columns:
            return c
    return None


def _location_from_frame_row(row: pd.Series) -> tuple[float, float]:
    if "location" in row.index and isinstance(row["location"], (list, tuple)) and len(row["location"]) >= 2:
        return float(row["location"][0]), float(row["location"][1])
    if "x" in row.index and "y" in row.index:
        try:
            return float(row["x"]), float(row["y"])
        except Exception:
            return float("nan"), float("nan")
    return float("nan"), float("nan")


def _bool_from_any(value: object, default: bool = False) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return default


def build_360_features(
    frames_df: pd.DataFrame,
    events_df: pd.DataFrame,
    pitch_length: float = 120.0,
    pitch_width: float = 80.0,
) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()

    event_xy = (
        events_df[["event_id", "x", "y"]]
        .drop_duplicates(subset=["event_id"])
        .set_index("event_id")
        .to_dict(orient="index")
    )

    defaults = {
        "has_360": 0,
        **{c: 0 for c in COUNT_COLUMNS},
        **{c: 999.0 for c in DIST_COLUMNS},
        "teammate_width": 0.0,
        "opponent_width": 0.0,
        "teammate_depth": 0.0,
        "opponent_depth": 0.0,
        "visible_area_size": 0.0,
        "visible_area_fraction": 0.0,
    }

    if frames_df.empty:
        out = events_df[["event_id"]].drop_duplicates().copy()
        for k, v in defaults.items():
            out[k] = v
        return out

    event_id_col = _event_id_column(frames_df)
    if event_id_col is None:
        out = events_df[["event_id"]].drop_duplicates().copy()
        for k, v in defaults.items():
            out[k] = v
        return out

    features: list[dict[str, object]] = []
    grouped = frames_df.groupby(event_id_col, dropna=True)

    for event_id, group in grouped:
        event_id = str(event_id)
        ball = event_xy.get(event_id, {"x": np.nan, "y": np.nan})
        ball_x = float(ball.get("x", np.nan))
        ball_y = float(ball.get("y", np.nan))

        locs_teammate: list[tuple[float, float]] = []
        locs_opponent: list[tuple[float, float]] = []
        d_teammate: list[float] = []
        d_opponent: list[float] = []
        keepers = 0
        actors = 0
        valid_players = 0

        for _, row in group.iterrows():
            px, py = _location_from_frame_row(row)
            if not np.isfinite(px) or not np.isfinite(py):
                continue
            valid_players += 1
            teammate = _bool_from_any(row.get("teammate"), default=False)
            keeper = _bool_from_any(row.get("keeper"), default=False)
            actor = _bool_from_any(row.get("actor"), default=False)
            if keeper:
                keepers += 1
            if actor:
                actors += 1

            dist = euclidean_distance(px, py, ball_x, ball_y)
            if teammate:
                locs_teammate.append((px, py))
                if np.isfinite(dist):
                    d_teammate.append(dist)
            else:
                locs_opponent.append((px, py))
                if np.isfinite(dist):
                    d_opponent.append(dist)

        rec: dict[str, object] = {
            "event_id": event_id,
            "has_360": int(len(group) > 0),
            "num_visible_players": int(valid_players),
            "num_visible_teammates": int(len(locs_teammate)),
            "num_visible_opponents": int(len(locs_opponent)),
            "num_visible_keepers": int(keepers),
            "num_visible_actors": int(actors),
            "nearest_teammate_distance_to_ball": float(np.min(d_teammate)) if d_teammate else np.nan,
            "nearest_opponent_distance_to_ball": float(np.min(d_opponent)) if d_opponent else np.nan,
            "mean_teammate_distance_to_ball": float(np.mean(d_teammate)) if d_teammate else np.nan,
            "mean_opponent_distance_to_ball": float(np.mean(d_opponent)) if d_opponent else np.nan,
            "opponents_within_5": int(np.sum(np.array(d_opponent) <= 5.0)) if d_opponent else 0,
            "opponents_within_10": int(np.sum(np.array(d_opponent) <= 10.0)) if d_opponent else 0,
            "teammates_within_10": int(np.sum(np.array(d_teammate) <= 10.0)) if d_teammate else 0,
            "teammates_ahead_of_ball": int(sum(px > ball_x for px, _ in locs_teammate if np.isfinite(ball_x))),
            "opponents_ahead_of_ball": int(sum(px > ball_x for px, _ in locs_opponent if np.isfinite(ball_x))),
            "opponents_between_ball_and_goal": int(
                sum((px > ball_x) and (abs(py - ball_y) < 15.0) for px, py in locs_opponent)
                if np.isfinite(ball_x) and np.isfinite(ball_y)
                else 0
            ),
            "teammate_width": float(np.ptp([y for _, y in locs_teammate])) if len(locs_teammate) > 1 else 0.0,
            "opponent_width": float(np.ptp([y for _, y in locs_opponent])) if len(locs_opponent) > 1 else 0.0,
            "teammate_depth": float(np.ptp([x for x, _ in locs_teammate])) if len(locs_teammate) > 1 else 0.0,
            "opponent_depth": float(np.ptp([x for x, _ in locs_opponent])) if len(locs_opponent) > 1 else 0.0,
        }

        va_size = np.nan
        if "visible_area" in group.columns:
            for value in group["visible_area"].tolist():
                points = parse_visible_area(value)
                if points:
                    va_size = polygon_area(points)
                    break

        rec["visible_area_size"] = float(va_size) if np.isfinite(va_size) else np.nan
        rec["visible_area_fraction"] = (
            float(va_size / (pitch_length * pitch_width)) if np.isfinite(va_size) else np.nan
        )
        features.append(rec)

    feats = pd.DataFrame(features)
    all_events = events_df[["event_id"]].drop_duplicates()
    merged = all_events.merge(feats, on="event_id", how="left")

    merged["has_360"] = merged["has_360"].fillna(0).astype(int)
    for col in COUNT_COLUMNS:
        merged[col] = merged[col].fillna(0).astype(int)
    for col in DIST_COLUMNS:
        merged[col] = merged[col].fillna(999.0)

    merged["teammate_width"] = merged["teammate_width"].fillna(0.0)
    merged["opponent_width"] = merged["opponent_width"].fillna(0.0)
    merged["teammate_depth"] = merged["teammate_depth"].fillna(0.0)
    merged["opponent_depth"] = merged["opponent_depth"].fillna(0.0)
    merged["visible_area_size"] = merged["visible_area_size"].fillna(0.0)
    merged["visible_area_fraction"] = merged["visible_area_fraction"].fillna(0.0)

    return merged
