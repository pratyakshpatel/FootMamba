#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/model.pt")
    parser.add_argument("--match-id", type=int, default=None)
    parser.add_argument("--event-index", type=int, default=None)
    parser.add_argument("--event-id", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    from src.visualization.case_study import make_case_study

    config = load_config(args.config)
    out = make_case_study(
        config=config,
        checkpoint_path=args.checkpoint,
        match_id=args.match_id,
        event_index=args.event_index,
        event_id=args.event_id,
        output_path=args.output,
    )
    print(f"Case study PNG: {out['png_path']}")
    print(f"Case study JSON: {out['summary_path']}")


if __name__ == "__main__":
    main()
