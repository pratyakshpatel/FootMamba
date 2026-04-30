from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.data.build_sequences import (
    CATEGORICAL_COLUMNS,
    add_future_labels,
    build_datasets_for_splits,
)
from src.data.vocab import load_vocabs_dict
from src.generation.rollout import sample_ghost_futures
from src.generation.summarize_futures import summarize_futures
from src.models.ghost_model import GhostFutureModel
from src.utils.io import load_json, save_json
from src.visualization.ghost_futures_plot import plot_ghost_futures_case


def _load_raw_frames_for_match(cache_dir: Path, match_id: int) -> pd.DataFrame:
    path = cache_dir / f"match_{match_id}_frames.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _select_event(events: pd.DataFrame, match_id: int | None, event_id: str | None, event_index: int | None) -> pd.Series:
    df = events.copy()
    if match_id is not None:
        df = df[df["match_id"] == int(match_id)]
    if event_id is not None:
        df = df[df["event_id"] == str(event_id)]
    if event_index is not None:
        df = df[df["event_index"] == int(event_index)]
    if df.empty:
        df = events[(events["has_360"] > 0) & (events["event_type"].str.lower().str.contains("pass|carry", regex=True))]
    if df.empty:
        raise ValueError("No suitable event found for case study.")
    return df.iloc[0]


def make_case_study(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    match_id: int | None = None,
    event_id: str | None = None,
    event_index: int | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    processed_dir = Path(config["data"]["processed_dir"])
    events_path = processed_dir / "events.parquet"
    vocabs_path = processed_dir / "vocabs.json"
    splits_path = processed_dir / "splits.json"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing processed events file: {events_path}")
    if not vocabs_path.exists():
        raise FileNotFoundError(f"Missing vocabs file: {vocabs_path}")
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing splits file: {splits_path}")

    events = pd.read_parquet(events_path)
    events = add_future_labels(
        events,
        future_horizon=int(config["features"]["future_horizon"]),
        turnover_horizon=int(config["features"]["turnover_horizon"]),
    )
    vocabs = load_vocabs_dict(load_json(vocabs_path))
    splits = load_json(splits_path)

    events = events.sort_values(["match_id", "period", "time_seconds", "event_index"]).reset_index(drop=True)
    selected_event = _select_event(events, match_id, event_id, event_index)
    selected_match_id = int(selected_event["match_id"])
    selected_event_id = str(selected_event["event_id"])

    dataset = build_datasets_for_splits(
        events_df=events,
        vocabs=vocabs,
        splits={"full": [selected_match_id]},
        context_length=int(config["features"]["context_length"]),
    )["full"]

    match_events = events[events["match_id"] == selected_match_id].reset_index(drop=True)
    pos = match_events.index[match_events["event_id"] == selected_event_id]
    if len(pos) == 0:
        raise ValueError("Selected event not found in processed events.")
    sample = dataset[int(pos[0])]

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = GhostFutureModel(
        vocab_sizes=[len(vocabs[col].idx_to_token) for col in CATEGORICAL_COLUMNS],
        num_continuous=len(dataset.continuous_columns),
        num_action_types=len(vocabs["event_type"].idx_to_token),
        model_cfg=ckpt["config"]["model"],
    )
    model.load_state_dict(ckpt["model_state_dict"])

    device_cfg = str(config["project"]["device"]).lower()
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    model = model.to(device)

    history = {
        "cat_x": sample["cat_x"],
        "cont_x": sample["cont_x"],
        "mask": sample["mask"],
    }
    sampled = sample_ghost_futures(
        model=model,
        initial_history=history,
        dataset_or_feature_builder=dataset,
        vocabs=ckpt["vocabs"],
        config=config,
        num_samples=int(config["generation"]["num_samples"]),
        rollout_steps=int(config["generation"]["rollout_steps"]),
        temperature=float(config["generation"]["temperature"]),
        top_k=int(config["generation"]["top_k"]),
        device=device,
    )
    summary = summarize_futures(sampled)

    start = int(pos[0]) + 1
    end = start + int(config["generation"]["rollout_steps"])
    actual_next = match_events.iloc[start:end].copy()

    cache_dir = Path(config["data"]["cache_dir"])
    raw_frames = _load_raw_frames_for_match(cache_dir, selected_match_id)
    frame_id_col = None
    for c in ["event_id", "id", "event_uuid", "event"]:
        if c in raw_frames.columns:
            frame_id_col = c
            break
    if frame_id_col is None:
        current_frame = pd.DataFrame()
    else:
        current_frame = raw_frames[raw_frames[frame_id_col].astype(str) == selected_event_id].copy()

    out_dir = Path(config["visualization"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = out_dir / f"case_study_match_{selected_match_id}_{selected_event_id}.png"
    output_path = Path(output_path)
    summary_path = output_path.with_suffix(".json")

    plot_ghost_futures_case(
        current_event=selected_event,
        current_frame=current_frame,
        actual_future_events=actual_next,
        sampled_futures=sampled,
        summary=summary,
        output_path=output_path,
        config=config,
    )
    save_json(summary, summary_path)
    return {"png_path": str(output_path), "summary_path": str(summary_path), "summary": summary}
