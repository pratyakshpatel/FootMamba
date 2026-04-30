#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _run(cmd: list[str]) -> None:
    LOGGER.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    try:
        _run([sys.executable, "scripts/01_cache_statsbomb_data.py", "--config", args.config])
        _run([sys.executable, "scripts/02_build_dataset.py", "--config", args.config])
    except Exception as exc:
        raise RuntimeError(
            "Could not access StatsBomb Open Data. Check internet connection or run with cached data."
        ) from exc

    checkpoint_path = "outputs/checkpoints/smoke_model.pt"
    _run(
        [
            sys.executable,
            "scripts/03_train_model.py",
            "--config",
            args.config,
            "--processed-path",
            "data/processed/dataset.pt",
            "--output",
            checkpoint_path,
        ]
    )

    metrics_path = "outputs/metrics/smoke_eval_metrics.json"
    _run(
        [
            sys.executable,
            "scripts/04_evaluate_model.py",
            "--config",
            args.config,
            "--checkpoint",
            checkpoint_path,
            "--output",
            metrics_path,
        ]
    )

    _run(
        [
            sys.executable,
            "scripts/05_make_case_study.py",
            "--config",
            args.config,
            "--checkpoint",
            checkpoint_path,
            "--output",
            "outputs/case_studies/smoke_case_study.png",
        ]
    )

    processed_data_path = Path(config["data"]["processed_dir"]) / "events.parquet"
    print("Smoke test complete.")
    print(f"Processed data: {processed_data_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print("Case study PNG: outputs/case_studies/smoke_case_study.png")
    print("Case study JSON: outputs/case_studies/smoke_case_study.json")


if __name__ == "__main__":
    main()
