"""Render the project Gantt chart for the CRISP-DM timeline (Section 4.2).

Produces a single PNG showing one horizontal bar per CRISP-DM phase across
W1-W11, matching the phase plan in Table 2 of the report. Output is written
to outputs/figures/06_gantt_timeline.png and is committed alongside the
other report figures.

The schedule is intentionally compressed into eleven weeks to reflect the
actual MSc Final Project window; overlapping bars between the pipeline and
GT preparation tracks (W3-W5 and W4-W5) preserve the real concurrency.

Usage:
    python scripts/render_gantt.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

FIG_PATH = Path(__file__).resolve().parent.parent / "outputs" / "figures" / "06_gantt_timeline.png"

PHASES = [
    ("Business Understanding", 1, 2),
    ("Data Understanding", 2, 4),
    ("Data Preparation (pipeline)", 3, 5),
    ("Data Preparation (GT)", 4, 5),
    ("Modeling", 5, 7),
    ("Evaluation", 7, 9),
    ("Deployment", 9, 10),
    ("Reporting", 10, 11),
]

WEEK_MIN, WEEK_MAX = 1, 11


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

    ax.set_xlim(WEEK_MIN - 0.5, WEEK_MAX + 0.5)
    ax.set_xticks(range(WEEK_MIN, WEEK_MAX + 1))
    ax.set_xticklabels([f"W{w}" for w in range(WEEK_MIN, WEEK_MAX + 1)], fontsize=9)
    ax.set_xlabel("Project week", fontsize=10)

    ax.grid(axis="x", linestyle="--", linewidth=0.5, color="#CCCCCC", alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.set_title(
        f"Project Gantt chart - CRISP-DM phases across illustrative weeks W{WEEK_MIN}-W{WEEK_MAX}",
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
