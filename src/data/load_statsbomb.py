from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.utils.io import ensure_dir, save_json, save_parquet
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

try:
    from statsbombpy import sb
except Exception as exc:  # pragma: no cover - import path differs by environment
    sb = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def _require_statsbombpy() -> None:
    if sb is None:
        raise ImportError(f"statsbombpy is required but failed to import: {IMPORT_ERROR}")


def _safe_dataframe(result: Any) -> pd.DataFrame:
    if result is None:
        return pd.DataFrame()
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, dict):
        frames: list[pd.DataFrame] = []
        for key, value in result.items():
            if isinstance(value, pd.DataFrame):
                frame = value.copy()
                frame["lineup_team_key"] = str(key)
                frames.append(frame)
        if frames:
            return pd.concat(frames, ignore_index=True)
    try:
        return pd.DataFrame(result)
    except Exception:
        return pd.DataFrame()


def load_competitions() -> pd.DataFrame:
    _require_statsbombpy()
    try:
        return _safe_dataframe(sb.competitions(fmt="dataframe"))
    except Exception as exc:
        LOGGER.warning("Failed loading competitions: %s", exc)
        return pd.DataFrame()


def load_matches_for_competition(competition_id: int, season_id: int) -> pd.DataFrame:
    _require_statsbombpy()
    try:
        return _safe_dataframe(
            sb.matches(competition_id=competition_id, season_id=season_id, fmt="dataframe")
        )
    except Exception as exc:
        LOGGER.warning(
            "Failed loading matches for competition_id=%s season_id=%s: %s",
            competition_id,
            season_id,
            exc,
        )
        return pd.DataFrame()


def load_events(match_id: int) -> pd.DataFrame:
    _require_statsbombpy()
    try:
        return _safe_dataframe(sb.events(match_id=match_id, fmt="dataframe"))
    except Exception as exc:
        LOGGER.warning("Failed loading events for match_id=%s: %s", match_id, exc)
        return pd.DataFrame()


def load_frames(match_id: int) -> pd.DataFrame:
    _require_statsbombpy()
    try:
        return _safe_dataframe(sb.frames(match_id=match_id, fmt="dataframe"))
    except Exception as exc:
        LOGGER.warning("Failed loading frames for match_id=%s: %s", match_id, exc)
        return pd.DataFrame()


def load_lineups(match_id: int) -> pd.DataFrame:
    _require_statsbombpy()
    try:
        lineups = sb.lineups(match_id=match_id, fmt="dataframe")
        return _safe_dataframe(lineups)
    except Exception as exc:
        LOGGER.warning("Failed loading lineups for match_id=%s: %s", match_id, exc)
        return pd.DataFrame()


def _cache_paths(match_id: int, cache_dir: str | Path) -> dict[str, Path]:
    cache_dir = ensure_dir(cache_dir)
    prefix = cache_dir / f"match_{match_id}"
    return {
        "events": Path(f"{prefix}_events.parquet"),
        "frames": Path(f"{prefix}_frames.parquet"),
        "lineups": Path(f"{prefix}_lineups.parquet"),
        "meta": Path(f"{prefix}_meta.json"),
    }


def cache_match_data(match_id: int, cache_dir: str | Path) -> dict[str, Path]:
    paths = _cache_paths(match_id, cache_dir)
    events = load_events(match_id)
    frames = load_frames(match_id)
    lineups = load_lineups(match_id)

    save_parquet(events, paths["events"])
    save_parquet(frames, paths["frames"])
    save_parquet(lineups, paths["lineups"])
    save_json(
        {
            "match_id": int(match_id),
            "events_rows": int(len(events)),
            "frames_rows": int(len(frames)),
            "lineups_rows": int(len(lineups)),
        },
        paths["meta"],
    )
    return paths


def load_cached_or_fetch_match(match_id: int, cache_dir: str | Path) -> dict[str, pd.DataFrame]:
    paths = _cache_paths(match_id, cache_dir)
    if paths["events"].exists() and paths["frames"].exists() and paths["lineups"].exists():
        return {
            "events": pd.read_parquet(paths["events"]),
            "frames": pd.read_parquet(paths["frames"]),
            "lineups": pd.read_parquet(paths["lineups"]),
        }
    cache_match_data(match_id, cache_dir)
    return {
        "events": pd.read_parquet(paths["events"]),
        "frames": pd.read_parquet(paths["frames"]),
        "lineups": pd.read_parquet(paths["lineups"]),
    }


def find_matches_with_360(max_matches: int | None = None) -> list[dict[str, Any]]:
    competitions = load_competitions()
    if competitions.empty:
        return []

    required_cols = {"competition_id", "season_id"}
    if not required_cols.issubset(competitions.columns):
        LOGGER.warning("Competitions dataframe missing expected columns: %s", required_cols)
        return []

    found: list[dict[str, Any]] = []
    for _, comp in tqdm(competitions.iterrows(), total=len(competitions), desc="Scanning competitions"):
        competition_id = int(comp["competition_id"])
        season_id = int(comp["season_id"])
        matches = load_matches_for_competition(competition_id, season_id)
        if matches.empty or "match_id" not in matches.columns:
            continue

        for _, match in matches.iterrows():
            match_id = int(match.get("match_id"))
            try:
                frames = load_frames(match_id)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Frames fetch failed for match_id=%s: %s", match_id, exc)
                continue

            if frames.empty:
                continue

            found.append(
                {
                    "match_id": match_id,
                    "competition_id": competition_id,
                    "season_id": season_id,
                    "match_date": str(match.get("match_date", "")),
                    "home_team": str(match.get("home_team", "")),
                    "away_team": str(match.get("away_team", "")),
                    "frames_rows": int(len(frames)),
                }
            )
            if max_matches is not None and len(found) >= max_matches:
                return found
    return found

