#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.build_sequences import CATEGORICAL_COLUMNS, build_datasets_for_splits
from src.data.vocab import load_vocabs_dict
from src.models.ghost_model import GhostFutureModel
from src.training.evaluate import evaluate_loader, save_metrics
from src.utils.config import load_config
from src.utils.io import load_json


def run_evaluate(config: dict, checkpoint_path: str | Path, output_path: str | Path) -> dict[str, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    processed_dir = Path(config["data"]["processed_dir"])
    events = pd.read_parquet(processed_dir / "events.parquet")
    vocabs = load_vocabs_dict(load_json(processed_dir / "vocabs.json"))
    splits = load_json(processed_dir / "splits.json")
    datasets = build_datasets_for_splits(
        events_df=events,
        vocabs=vocabs,
        splits=splits,
        context_length=int(config["features"]["context_length"]),
    )
    eval_ds = datasets["test"] if len(datasets["test"]) > 0 else datasets["val"]
    if len(eval_ds) == 0:
        eval_ds = datasets["train"]
    if len(eval_ds) == 0:
        all_match_ids = sorted(set(events["match_id"].astype(int).tolist()))
        fallback = build_datasets_for_splits(
            events_df=events,
            vocabs=vocabs,
            splits={"train": all_match_ids, "val": all_match_ids, "test": all_match_ids},
            context_length=int(config["features"]["context_length"]),
        )
        eval_ds = fallback["test"]

    model_cfg = checkpoint["config"]["model"]
    model = GhostFutureModel(
        vocab_sizes=[len(vocabs[col].idx_to_token) for col in CATEGORICAL_COLUMNS],
        num_continuous=len(eval_ds.continuous_columns),
        num_action_types=len(vocabs["event_type"].idx_to_token),
        model_cfg=model_cfg,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device_cfg = str(config["project"]["device"]).lower()
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    model = model.to(device)

    loader = DataLoader(eval_ds, batch_size=int(config["training"]["batch_size"]), shuffle=False)
    metrics = evaluate_loader(model=model, loader=loader, loss_cfg=config["loss"], device=device)
    save_metrics(metrics, output_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/model.pt")
    parser.add_argument("--output", default="outputs/metrics/eval_metrics.json")
    args = parser.parse_args()

    config = load_config(args.config)
    metrics = run_evaluate(config=config, checkpoint_path=args.checkpoint, output_path=args.output)
    print(f"Saved metrics: {args.output}")
    print(metrics)


if __name__ == "__main__":
    main()
