from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def clamp_xy(x: float, y: float, pitch_length: float = 120.0, pitch_width: float = 80.0) -> tuple[float, float]:
    if not np.isfinite(x) or not np.isfinite(y):
        return float("nan"), float("nan")
    return float(np.clip(x, 0.0, pitch_length)), float(np.clip(y, 0.0, pitch_width))


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    if not all(np.isfinite(v) for v in [x1, y1, x2, y2]):
        return float("nan")
    return float(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))


def polygon_area(points: Iterable[tuple[float, float]]) -> float:
    pts = [(float(x), float(y)) for x, y in points if np.isfinite(x) and np.isfinite(y)]
    if len(pts) < 3:
        return 0.0
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    area = 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return area


def parse_visible_area(value: object) -> list[tuple[float, float]]:
    if value is None:
        return []
    if isinstance(value, list):
        if not value:
            return []
        if isinstance(value[0], (list, tuple)) and len(value[0]) >= 2:
            return [(float(v[0]), float(v[1])) for v in value if len(v) >= 2]
        if len(value) % 2 == 0 and all(isinstance(v, (int, float)) for v in value):
            points = []
            for i in range(0, len(value), 2):
                points.append((float(value[i]), float(value[i + 1])))
            return points
    return []

