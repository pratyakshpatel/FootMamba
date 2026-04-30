#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_statsbomb import (
    cache_match_data,
    find_matches_with_360,
    load_frames,
    load_matches_for_competition,
)
from src.utils.config import load_config
from src.utils.io import save_json
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _select_matches_from_competition(
    competition_id: int,
    season_id: int,
    max_matches: int | None,
) -> list[dict]:
    matches_df = load_matches_for_competition(competition_id=competition_id, season_id=season_id)
    if matches_df.empty or "match_id" not in matches_df.columns:
        return []

    if "match_status_360" in matches_df.columns:
        status = matches_df["match_status_360"].astype(str).str.lower()
        matches_df = matches_df[status.eq("available")].copy()
    matches_df = matches_df.sort_values("match_date")

    selected: list[dict] = []
    for _, row in matches_df.iterrows():
        match_id = int(row["match_id"])
        ok = False
        frames_rows = 0
        for attempt in range(3):
            try:
                frames = load_frames(match_id)
                if isinstance(frames, pd.DataFrame) and not frames.empty:
                    ok = True
                    frames_rows = int(len(frames))
                    break
            except Exception:
                time.sleep(0.4 * (attempt + 1))
        if not ok:
            continue

        selected.append(
            {
                "match_id": match_id,
                "competition_id": int(competition_id),
                "season_id": int(season_id),
                "match_date": str(row.get("match_date", "")),
                "home_team": str(row.get("home_team", "")),
                "away_team": str(row.get("away_team", "")),
                "frames_rows": frames_rows,
            }
        )
        if max_matches is not None and len(selected) >= max_matches:
            break
    return selected


def run_cache(
    config: dict,
    competition_id: int | None = None,
    season_id: int | None = None,
) -> list[dict]:
    max_matches = config["data"].get("max_matches")
    cache_dir = Path(config["data"]["cache_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    if competition_id is not None and season_id is not None:
        matches = _select_matches_from_competition(
            competition_id=competition_id,
            season_id=season_id,
            max_matches=max_matches,
        )
    else:
        matches = find_matches_with_360(max_matches=max_matches)
    if not matches:
        LOGGER.warning("No matches with 360 found.")
        return []

    for m in matches:
        try:
            cache_match_data(int(m["match_id"]), cache_dir=cache_dir)
        except Exception as exc:
            LOGGER.warning("Failed caching match_id=%s: %s", m["match_id"], exc)

    selected_path = processed_dir / "selected_matches.json"
    save_json({"matches": matches}, selected_path)
    LOGGER.info("Cached %d matches with 360. Metadata: %s", len(matches), selected_path)
    return matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--competition-id", type=int, default=None)
    parser.add_argument("--season-id", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    run_cache(config, competition_id=args.competition_id, season_id=args.season_id)


if __name__ == "__main__":
    main()
