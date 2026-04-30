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
from src.models.losses import MultiTaskLoss
from src.training.train import fit_model
from src.utils.config import load_config
from src.utils.io import ensure_dir, load_json, save_json
from src.utils.logging import get_logger
from src.utils.seed import set_seed

LOGGER = get_logger(__name__)


def _load_processed_paths(processed_path: Path, config: dict) -> dict[str, str]:
    if processed_path.suffix == ".pt" and processed_path.exists():
        meta = torch.load(processed_path, map_location="cpu")
        return {
            "events_path": meta["events_path"],
            "vocabs_path": meta["vocabs_path"],
            "splits_path": meta["splits_path"],
        }
    processed_dir = Path(config["data"]["processed_dir"])
    return {
        "events_path": str(processed_dir / "events.parquet"),
        "vocabs_path": str(processed_dir / "vocabs.json"),
        "splits_path": str(processed_dir / "splits.json"),
    }


def run_train(config: dict, processed_path: str | Path, output: str | Path) -> str:
    set_seed(int(config["project"]["seed"]))
    paths = _load_processed_paths(Path(processed_path), config)
    events = pd.read_parquet(paths["events_path"])
    vocabs = load_vocabs_dict(load_json(paths["vocabs_path"]))
    splits = load_json(paths["splits_path"])

    datasets = build_datasets_for_splits(
        events_df=events,
        vocabs=vocabs,
        splits=splits,
        context_length=int(config["features"]["context_length"]),
    )
    train_ds = datasets["train"]
    val_ds = datasets["val"] if len(datasets["val"]) > 0 else datasets["train"]

    if len(train_ds) == 0:
        # Defensive fallback for very small smoke subsets where split rows may collapse.
        all_match_ids = sorted(set(events["match_id"].astype(int).tolist()))
        fallback = build_datasets_for_splits(
            events_df=events,
            vocabs=vocabs,
            splits={"train": all_match_ids, "val": all_match_ids, "test": all_match_ids},
            context_length=int(config["features"]["context_length"]),
        )
        train_ds = fallback["train"]
        val_ds = fallback["val"]
        LOGGER.warning("Train split was empty; using all matches for train/val fallback.")
    elif len(val_ds) == 0:
        val_ds = train_ds
        LOGGER.warning("Validation split was empty; reusing train split for validation.")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
    )

    model = GhostFutureModel(
        vocab_sizes=[len(vocabs[col].idx_to_token) for col in CATEGORICAL_COLUMNS],
        num_continuous=len(train_ds.continuous_columns),
        num_action_types=len(vocabs["event_type"].idx_to_token),
        model_cfg=config["model"],
    )

    if config["project"]["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["project"]["device"])
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    loss_fn = MultiTaskLoss(config["loss"])

    output = Path(output)
    ensure_dir(output.parent)
    result = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
        vocabs=load_json(paths["vocabs_path"]),
        device=device,
        output_path=output,
    )
    metrics_dir = ensure_dir("outputs/metrics")
    save_json(result, metrics_dir / "train_summary.json")
    LOGGER.info("Training complete. Checkpoint: %s", output)
    return str(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--processed-path", default="data/processed/dataset.pt")
    parser.add_argument("--output", default="outputs/checkpoints/model.pt")
    args = parser.parse_args()

    config = load_config(args.config)
    run_train(config=config, processed_path=args.processed_path, output=args.output)


if __name__ == "__main__":
    main()
