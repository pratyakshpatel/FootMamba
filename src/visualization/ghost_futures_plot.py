from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.visualization.pitch_plot import draw_arrow, draw_base_pitch, plot_event_sequence, plot_freeze_frame


def plot_ghost_futures_case(
    current_event: pd.Series,
    current_frame: pd.DataFrame,
    actual_future_events: pd.DataFrame,
    sampled_futures: list[dict[str, Any]],
    summary: dict[str, Any],
    output_path: str | Path,
    config: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=int(config["visualization"]["figure_dpi"]))
    ax_a, ax_b, ax_c = axes

    plot_freeze_frame(
        ax=ax_a,
        frame_df=current_frame,
        event_row=current_event,
        show_visible_area=bool(config["visualization"].get("show_visible_area", True)),
    )
    plot_event_sequence(ax=ax_b, events_df=actual_future_events, title="Actual Next 5 Events", style="actual")

    draw_base_pitch(ax_c, title="Sampled Ghost Futures")
    max_draw = int(config["visualization"].get("max_ghost_futures_to_draw", 12))
    for i, future in enumerate(sampled_futures[:max_draw]):
        events = future.get("events", [])
        alpha = 0.25
        prev_x = float(current_event.get("x", 60.0))
        prev_y = float(current_event.get("y", 40.0))
        for ev in events:
            x1 = prev_x
            y1 = prev_y
            x2 = float(ev["x"])
            y2 = float(ev["y"])
            draw_arrow(
                ax_c,
                x1,
                y1,
                x2,
                y2,
                label=str(i + 1) if ev["step"] == 1 and i < 5 else None,
                alpha=alpha,
                linestyle=":",
                color="tab:purple",
            )
            et = str(ev.get("event_type", "")).lower()
            if "shot" in et:
                ax_c.scatter([x2], [y2], marker="*", s=55, color="gold", edgecolor="black")
            if float(ev.get("turnover_next_3_prob", 0.0)) > 0.5:
                ax_c.scatter([x2], [y2], marker="x", s=35, color="red")
            prev_x, prev_y = x2, y2

    text = (
        f"match_id={current_event.get('match_id', 'n/a')}\n"
        f"time={current_event.get('minute', 0)}:{current_event.get('second', 0)}\n"
        f"event={current_event.get('event_type', 'unknown')}\n"
        f"top={summary.get('top_categories', [])}\n"
        f"mean_shot={summary.get('avg_shot_next_5_prob', 0.0):.2f}\n"
        f"mean_turn={summary.get('avg_turnover_next_3_prob', 0.0):.2f}"
    )
    ax_c.text(
        2,
        78,
        text,
        fontsize=8,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
    )

    fig.suptitle("Mamba-360 Ghost Futures", fontsize=16, y=0.98)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(config["visualization"]["figure_dpi"]))
    plt.close(fig)
