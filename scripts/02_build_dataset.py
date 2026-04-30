#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.build_360_features import build_360_features
from src.data.build_events import build_events_dataframe
from src.data.build_sequences import CATEGORICAL_COLUMNS, add_future_labels, build_datasets_for_splits
from src.data.load_statsbomb import cache_match_data, find_matches_with_360, load_cached_or_fetch_match
from src.data.split import split_match_ids
from src.data.vocab import build_vocabs, save_vocabs, vocabs_to_serializable
from src.utils.config import load_config
from src.utils.io import ensure_dir, load_json, save_json, save_parquet
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _get_selected_matches(config: dict) -> list[dict]:
    processed_dir = Path(config["data"]["processed_dir"])
    selected_path = processed_dir / "selected_matches.json"
    if selected_path.exists():
        data = load_json(selected_path)
        return list(data.get("matches", []))

    matches = find_matches_with_360(max_matches=config["data"].get("max_matches"))
    for m in matches:
        cache_match_data(int(m["match_id"]), config["data"]["cache_dir"])
    save_json({"matches": matches}, selected_path)
    return matches


def run_build_dataset(config: dict) -> dict[str, str]:
    cache_dir = Path(config["data"]["cache_dir"])
    processed_dir = ensure_dir(config["data"]["processed_dir"])

    selected_matches = _get_selected_matches(config)
    if not selected_matches:
        raise RuntimeError("No selected matches available to build dataset.")

    all_events = []
    all_features = []
    for m in selected_matches:
        match_id = int(m["match_id"])
        raw = load_cached_or_fetch_match(match_id=match_id, cache_dir=cache_dir)
        events_raw = raw["events"]
        frames_raw = raw["frames"]
        if events_raw.empty:
            continue
        events = build_events_dataframe(events_raw)
        if len(events) < int(config["data"].get("min_events_per_match", 1)):
            continue
        feats = build_360_features(
            frames_df=frames_raw,
            events_df=events,
            pitch_length=float(config["features"]["pitch_length"]),
            pitch_width=float(config["features"]["pitch_width"]),
        )
        events = events.merge(feats, on="event_id", how="left")
        all_events.append(events)
        all_features.append(feats.assign(match_id=match_id))

    if not all_events:
        raise RuntimeError("No matches produced valid event data.")

    events_df = pd.concat(all_events, ignore_index=True)
    features_df = pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame()
    events_df = add_future_labels(
        events_df,
        future_horizon=int(config["features"]["future_horizon"]),
        turnover_horizon=int(config["features"]["turnover_horizon"]),
    )

    vocabs = build_vocabs(events_df, CATEGORICAL_COLUMNS)
    splits = split_match_ids(
        match_ids=events_df["match_id"].astype(int).tolist(),
        train_frac=float(config["data"]["train_frac"]),
        val_frac=float(config["data"]["val_frac"]),
        test_frac=float(config["data"]["test_frac"]),
        seed=int(config["project"]["seed"]),
    )
    _ = build_datasets_for_splits(
        events_df=events_df,
        vocabs=vocabs,
        splits=splits,
        context_length=int(config["features"]["context_length"]),
    )

    events_path = processed_dir / "events.parquet"
    feats_path = processed_dir / "frames_features.parquet"
    vocabs_path = processed_dir / "vocabs.json"
    splits_path = processed_dir / "splits.json"
    dataset_path = processed_dir / "dataset.pt"

    save_parquet(events_df, events_path)
    save_parquet(features_df, feats_path)
    save_vocabs(vocabs, vocabs_path)
    save_json(splits, splits_path)
    torch.save(
        {
            "events_path": str(events_path),
            "features_path": str(feats_path),
            "vocabs_path": str(vocabs_path),
            "splits_path": str(splits_path),
            "config_snapshot": config,
        },
        dataset_path,
    )
    LOGGER.info("Saved processed data to %s", processed_dir)
    return {
        "events_path": str(events_path),
        "features_path": str(feats_path),
        "vocabs_path": str(vocabs_path),
        "splits_path": str(splits_path),
        "dataset_path": str(dataset_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    run_build_dataset(config)


if __name__ == "__main__":
    main()
