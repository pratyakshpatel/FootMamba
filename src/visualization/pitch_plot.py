from __future__ import annotations

import numpy as np
import pandas as pd
from mplsoccer import Pitch

from src.utils.geometry import parse_visible_area


def draw_base_pitch(ax, title: str | None = None) -> None:
    pitch = Pitch(pitch_type="statsbomb")
    pitch.draw(ax=ax)
    if title:
        ax.set_title(title)


def draw_arrow(
    ax,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    label: str | None = None,
    alpha: float = 1.0,
    linestyle: str = "-",
    color: str = "tab:blue",
) -> None:
    if not np.isfinite([x1, y1, x2, y2]).all():
        return
    if abs(x2 - x1) < 1e-6 and abs(y2 - y1) < 1e-6:
        ax.scatter([x1], [y1], c=color, s=10, alpha=alpha)
        if label:
            ax.text(x1, y1, label, fontsize=8, color=color, alpha=alpha)
        return
    ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linestyle=linestyle, lw=1.5)
    ax.scatter([x2], [y2], c=color, s=8, alpha=alpha)
    if label:
        ax.text(x2, y2, label, fontsize=8, color=color, alpha=alpha)


def plot_freeze_frame(ax, frame_df: pd.DataFrame, event_row: pd.Series, show_visible_area: bool = True) -> None:
    draw_base_pitch(ax, title="Current 360 Freeze Frame")

    ex, ey = float(event_row.get("x", np.nan)), float(event_row.get("y", np.nan))
    if np.isfinite(ex) and np.isfinite(ey):
        ax.scatter([ex], [ey], c="gold", s=60, marker="o", edgecolor="black", label="Ball/Event")

    if frame_df is None or frame_df.empty:
        ax.text(60, 40, "No 360 frame available", ha="center", va="center")
        return

    team_x, team_y, opp_x, opp_y, actor_x, actor_y = [], [], [], [], [], []
    for _, row in frame_df.iterrows():
        loc = row.get("location")
        if not isinstance(loc, (list, tuple)) or len(loc) < 2:
            continue
        x, y = float(loc[0]), float(loc[1])
        if bool(row.get("actor", False)):
            actor_x.append(x)
            actor_y.append(y)
        if bool(row.get("teammate", False)):
            team_x.append(x)
            team_y.append(y)
        else:
            opp_x.append(x)
            opp_y.append(y)

    if team_x:
        ax.scatter(team_x, team_y, c="tab:blue", s=25, label="Teammates")
    if opp_x:
        ax.scatter(opp_x, opp_y, c="tab:red", s=25, label="Opponents")
    if actor_x:
        ax.scatter(actor_x, actor_y, c="tab:green", s=55, marker="*", label="Actor")

    if show_visible_area and "visible_area" in frame_df.columns:
        for value in frame_df["visible_area"]:
            pts = parse_visible_area(value)
            if pts:
                xs = [p[0] for p in pts] + [pts[0][0]]
                ys = [p[1] for p in pts] + [pts[0][1]]
                ax.plot(xs, ys, color="gray", alpha=0.5, linestyle="--", label="Visible area")
                break

    ax.legend(loc="upper right", fontsize=8)


def plot_event_sequence(ax, events_df: pd.DataFrame, title: str, style: str = "actual") -> None:
    draw_base_pitch(ax, title=title)
    if events_df is None or events_df.empty:
        ax.text(60, 40, "No events", ha="center", va="center")
        return

    color = "tab:blue" if style == "actual" else "gray"
    linestyle = "-" if style == "actual" else ":"
    alpha = 1.0 if style == "actual" else 0.3

    for i, (_, row) in enumerate(events_df.iterrows(), start=1):
        x1 = float(row.get("x", np.nan))
        y1 = float(row.get("y", np.nan))
        x2 = float(row.get("end_x", x1))
        y2 = float(row.get("end_y", y1))
        if not np.isfinite(x1) or not np.isfinite(y1):
            continue
        if not np.isfinite(x2) or not np.isfinite(y2):
            x2, y2 = x1, y1
        draw_arrow(ax, x1, y1, x2, y2, label=str(i), alpha=alpha, linestyle=linestyle, color=color)
