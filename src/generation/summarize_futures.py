from __future__ import annotations

from collections import Counter
from typing import Any


def summarize_futures(sampled_futures: list[dict[str, Any]]) -> dict[str, Any]:
    categories = [f.get("ending_category", "other") for f in sampled_futures]
    counts = Counter(categories)
    total = max(len(sampled_futures), 1)
    percentages = {k: float(v) / total for k, v in counts.items()}

    avg_shot = 0.0
    avg_turn = 0.0
    n_events = 0
    for future in sampled_futures:
        for ev in future.get("events", []):
            avg_shot += float(ev.get("shot_next_5_prob", 0.0))
            avg_turn += float(ev.get("turnover_next_3_prob", 0.0))
            n_events += 1
    if n_events > 0:
        avg_shot /= n_events
        avg_turn /= n_events

    top_categories = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return {
        "num_futures": len(sampled_futures),
        "category_counts": dict(counts),
        "category_percentages": percentages,
        "top_categories": [{"category": k, "count": int(v)} for k, v in top_categories],
        "avg_shot_next_5_prob": float(avg_shot),
        "avg_turnover_next_3_prob": float(avg_turn),
    }

