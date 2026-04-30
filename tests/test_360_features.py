from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.build_360_features import build_360_features
from src.utils.geometry import polygon_area


def test_polygon_area_rectangle() -> None:
    pts = [(0, 0), (4, 0), (4, 3), (0, 3)]
    assert polygon_area(pts) == 12.0


def test_360_features_toy() -> None:
    events = pd.DataFrame(
        [
            {"event_id": "e1", "x": 60.0, "y": 40.0},
        ]
    )
    frames = pd.DataFrame(
        [
            {"id": "e1", "teammate": True, "actor": False, "keeper": False, "location": [62.0, 41.0]},
            {"id": "e1", "teammate": False, "actor": False, "keeper": False, "location": [55.0, 39.0]},
        ]
    )
    feats = build_360_features(frames, events)
    assert len(feats) == 1
    assert int(feats.loc[0, "has_360"]) == 1
    assert np.isfinite(float(feats.loc[0, "nearest_teammate_distance_to_ball"]))

