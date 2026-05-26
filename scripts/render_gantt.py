"""Render the project Gantt chart for the CRISP-DM timeline (Section 4.2).

Produces a single PNG showing one horizontal bar per CRISP-DM phase across
W1-W16, matching the phase plan in Table 2 of the report. Output is written
to outputs/figures/06_gantt_timeline.png and is committed alongside the
other report figures.

Usage:
    python scripts/render_gantt.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

FIG_PATH = Path(__file__).resolve().parent.parent / "outputs" / "figures" / "06_gantt_timeline.png"

PHASES = [
    ("Business Understanding",          1,  2),
    ("Data Understanding",              2,  4),
    ("Data Preparation (pipeline)",     4,  7),
    ("Data Preparation (GT)",           5,  7),
    ("Modeling",                        7, 10),
    ("Evaluation",                     10, 12),
    ("Deployment",                     12, 14),
    ("Reporting",                      14, 16),
]


def render(out_path: Path = FIG_PATH) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))

    labels = [p[0] for p in PHASES]
    starts = [p[1] for p in PHASES]
    durations = [p[2] - p[1] for p in PHASES]
    y_positions = list(range(len(PHASES)))

    bar_color = "#4F6D8E"
    ax.barh(
        y_positions,
        durations,
        left=starts,
        height=0.55,
        color=bar_color,
        edgecolor="#2B3E55",
        linewidth=0.8,
    )

    for y, start, dur in zip(y_positions, starts, durations):
        ax.text(
            start + dur / 2,
            y,
            f"W{start}-W{start + dur}",
            va="center",
            ha="center",
            fontsize=8.5,
            color="white",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()

    ax.set_xlim(0.5, 16.5)
    ax.set_xticks(range(1, 17))
    ax.set_xticklabels([f"W{w}" for w in range(1, 17)], fontsize=8)
    ax.set_xlabel("Project week", fontsize=10)

    ax.grid(axis="x", linestyle="--", linewidth=0.5, color="#CCCCCC", alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.set_title(
        "Project Gantt chart - CRISP-DM phases across illustrative weeks W1-W16",
        fontsize=11,
        pad=12,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    written = render()
    print(f"Wrote {written}")
