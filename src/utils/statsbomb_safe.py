from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.geometry import clamp_xy


def safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    if isinstance(obj, pd.Series):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_name(value: Any, default: str = "unknown") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, dict):
        name = value.get("name")
        return str(name) if name else default
    if isinstance(value, str):
        value = value.strip()
        return value if value else default
    return str(value)


def safe_nested_name(row: pd.Series, column: str, default: str = "unknown") -> str:
    flat_candidates = [f"{column}.name", f"{column}_name", column]
    for candidate in flat_candidates:
        if candidate in row.index:
            name = _extract_name(row.get(candidate), default=default)
            if name != default:
                return name
    return default


def _safe_numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def safe_location(row: pd.Series, column: str = "location") -> tuple[float, float]:
    candidates = [
        column,
        column.replace(".", "_"),
        f"{column}.location",
        f"{column}_location",
    ]
    value = None
    for candidate in candidates:
        if candidate in row.index:
            value = row.get(candidate)
            break

    if isinstance(value, dict):
        x = _safe_numeric(value.get("x"))
        y = _safe_numeric(value.get("y"))
        return clamp_xy(x, y)

    if isinstance(value, np.ndarray) and value.size >= 2:
        x = _safe_numeric(value[0])
        y = _safe_numeric(value[1])
        return clamp_xy(x, y)

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x = _safe_numeric(value[0])
        y = _safe_numeric(value[1])
        return clamp_xy(x, y)

    return float("nan"), float("nan")


def safe_end_location(row: pd.Series) -> tuple[float, float]:
    ordered = [
        "pass_end_location",
        "pass.end_location",
        "carry_end_location",
        "carry.end_location",
        "shot_end_location",
        "shot.end_location",
        "goalkeeper_end_location",
        "goalkeeper.end_location",
    ]
    for col in ordered:
        x, y = safe_location(row, col)
        if np.isfinite(x) and np.isfinite(y):
            return x, y
    return safe_location(row, "location")


def safe_bool(row: pd.Series, column: str, default: bool = False) -> bool:
    if column not in row.index:
        return default
    value = row.get(column)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return default


def safe_event_type(row: pd.Series) -> str:
    return safe_nested_name(row, "type", default="unknown")


def safe_outcome(row: pd.Series) -> str:
    candidates = [
        "outcome_name",
        "pass_outcome_name",
        "shot_outcome_name",
        "duel_outcome_name",
        "dribble_outcome_name",
        "interception_outcome_name",
        "goalkeeper_outcome_name",
        "substitution_outcome_name",
        "outcome",
        "pass_outcome",
        "shot_outcome",
        "duel_outcome",
        "dribble_outcome",
        "interception_outcome",
        "goalkeeper_outcome",
        "substitution_outcome",
    ]
    for column in candidates:
        if column in row.index:
            value = _extract_name(row.get(column), default="")
            if value:
                return value
    return "unknown"


def safe_player(row: pd.Series) -> str:
    return safe_nested_name(row, "player", default="unknown")


def safe_team(row: pd.Series) -> str:
    return safe_nested_name(row, "team", default="unknown")


def safe_play_pattern(row: pd.Series) -> str:
    return safe_nested_name(row, "play_pattern", default="unknown")


def safe_body_part(row: pd.Series) -> str:
    candidates = [
        "body_part_name",
        "pass_body_part_name",
        "shot_body_part_name",
        "goalkeeper_body_part_name",
        "body_part",
        "pass_body_part",
        "shot_body_part",
        "goalkeeper_body_part",
    ]
    for c in candidates:
        if c in row.index:
            text = _extract_name(row.get(c), default="")
            if text:
                return text
    return "unknown"


def safe_under_pressure(row: pd.Series) -> int:
    return int(safe_bool(row, "under_pressure", default=False))


def timestamp_to_seconds(row: pd.Series) -> float:
    ts = row.get("timestamp")
    if isinstance(ts, str) and ts:
        parts = ts.split(":")
        if len(parts) == 3:
            try:
                hours = float(parts[0])
                mins = float(parts[1])
                secs = float(parts[2])
                return hours * 3600 + mins * 60 + secs
            except ValueError:
                pass
    minute = _safe_numeric(row.get("minute"))
    second = _safe_numeric(row.get("second"))
    if np.isfinite(minute) and np.isfinite(second):
        return float(max(0.0, minute * 60.0 + second))
    return float("nan")
