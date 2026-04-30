from __future__ import annotations

import torch

TERMINAL_EVENT_KEYWORDS = {
    "shot",
    "foul",
    "offside",
    "half end",
    "substitution",
    "injury stoppage",
    "tactical shift",
}


def is_terminal_event(event_type: str) -> bool:
    event_type = (event_type or "").lower()
    return any(k in event_type for k in TERMINAL_EVENT_KEYWORDS)


def zone_id_to_center(zone_id: int, x_bins: int = 12, y_bins: int = 8) -> tuple[float, float]:
    zone_id = int(max(0, min(zone_id, x_bins * y_bins - 1)))
    zone_x = zone_id % x_bins
    zone_y = zone_id // x_bins
    x = (zone_x + 0.5) * 120.0 / x_bins
    y = (zone_y + 0.5) * 80.0 / y_bins
    return float(x), float(y)


def center_to_zone_id(x: float, y: float, x_bins: int = 12, y_bins: int = 8) -> int:
    zone_x = int(min(max((x / 120.0) * x_bins, 0), x_bins - 1))
    zone_y = int(min(max((y / 80.0) * y_bins, 0), y_bins - 1))
    return int(zone_y * x_bins + zone_x)


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    values, indices = torch.topk(logits, k=k, dim=-1)
    masked = torch.full_like(logits, fill_value=-1e9)
    masked.scatter_(-1, indices, values)
    return masked


def categorize_future(events: list[dict[str, object]]) -> str:
    if not events:
        return "other"
    if any("shot" in str(e.get("event_type", "")).lower() for e in events):
        return "shot"
    if any(float(e.get("turnover_next_3_prob", 0.0)) > 0.5 for e in events):
        return "turnover"
    if any(float(e.get("x", 0.0)) > 102.0 for e in events):
        return "box_entry"

    xs = [float(e.get("x", 0.0)) for e in events]
    if xs and (xs[-1] - xs[0] < 4.0):
        return "recycle"
    return "other"

