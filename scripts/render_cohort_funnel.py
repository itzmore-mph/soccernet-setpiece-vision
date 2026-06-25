"""Render the cohort-attrition funnel figure (Figure 18).

Traces a single set-piece cohort through the pipeline funnel, from the 33
discovered clips down to the effective inferential unit. Each stage's count is
derived from the committed PC parquets, not hardcoded, so the figure stays
consistent with the validation tables.

Funnel stages (counts below are illustrative; the figure always reads the
current committed parquets, so the rendered numbers track whatever cohort
the pipeline currently produces):
    33 clips discovered (SoccerNet GSR corners + direct free kicks)
    33 clips calibrated (TVCalib, zero homography failures)
    21 clips with autonomous ball position (ball-detection coverage)
    651 pipeline PC frames
    639 paired frames (pipeline ∩ GT)
    21 matched clips (clip-level inferential unit)
    ~22-24 effective independent observations (ICC design effect)

Reads:
    outputs/pitch_control_soccana_tvcalib.parquet  (pipeline)
    outputs/pitch_control_gt_full.parquet           (GT reference)
    outputs/icc_per_metric.parquet                  (n_eff per metric)

Writes:
    outputs/figures/18_cohort_funnel.png            (committed figure)

Run: uv run python scripts/render_cohort_funnel.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"

PIPE_PATH = OUTPUTS_DIR / "pitch_control_soccana_tvcalib.parquet"
GT_PATH = OUTPUTS_DIR / "pitch_control_gt_full.parquet"
ICC_PATH = OUTPUTS_DIR / "icc_per_metric.parquet"
KEYS = ["split", "clip_id", "frame_idx"]

# Clips discovered with action_class in {Corner, Direct free-kick}. This is the
# top of the funnel and is fixed by the dataset scan (Section 7.2), not by a PC
# parquet, so it is stated as a constant.
N_DISCOVERED = 33
N_CALIBRATED = 33  # TVCalib: zero homography failures (Section 7.3)


def compute_stages() -> list[tuple[str, str, float]]:
    """Return (label, sublabel, value) per funnel stage, data-derived where possible."""
    pipe = pd.read_parquet(PIPE_PATH)
    gt = pd.read_parquet(GT_PATH)
    icc = pd.read_parquet(ICC_PATH)

    n_ball_clips = pipe["clip_id"].nunique()
    n_pipe_frames = len(pipe)
    n_paired = len(pipe[KEYS].merge(gt[KEYS], on=KEYS))
    n_common_clips = len(set(pipe["clip_id"]) & set(gt["clip_id"]))
    neff_lo = float(icc["n_eff"].min())
    neff_hi = float(icc["n_eff"].max())
    ball_pct = round(100 * n_ball_clips / N_DISCOVERED)
    design_effect = round(n_pipe_frames / ((neff_lo + neff_hi) / 2))

    return [
        ("Clips discovered", "SoccerNet GSR corners + free kicks", N_DISCOVERED),
        ("Clips calibrated", "TVCalib, zero homography failures", N_CALIBRATED),
        ("Clips with ball position", f"autonomous ball detection ({ball_pct}%)", n_ball_clips),
        ("Pipeline PC frames", f"{n_ball_clips} clips x ~31 frames", n_pipe_frames),
        ("Paired frames", "pipeline intersect GT", n_paired),
        ("Matched clips", "clip-level inferential unit", n_common_clips),
        ("Effective observations", f"ICC design effect ~{design_effect}", round((neff_lo + neff_hi) / 2)),
    ], (neff_lo, neff_hi)


def render(
    stages: list[tuple[str, str, float]],
    neff_range: tuple[float, float],
    n_ball_lost: int,
) -> Path:
    n = len(stages)
    fig, ax = plt.subplots(figsize=(9, 7))

    # Two visual scales: clip-count stages and frame-count stages differ by ~30x,
    # so bar widths are normalised within each regime and the count is annotated.
    max_w = 1.0
    colors = [
        "#1f77b4",
        "#1f77b4",
        "#2c7fb8",
        "#7fbf7b",
        "#7fbf7b",
        "#d95f0e",
        "#d95f0e",
    ]

    # Normalise widths: frames (stages 3-4) scaled to their own max; clips/obs to 33.
    frame_max = max(stages[3][2], stages[4][2])
    clip_max = N_DISCOVERED
    widths = []
    for i, (_label, _sub, val) in enumerate(stages):
        ref = frame_max if i in (3, 4) else clip_max
        widths.append(max_w * val / ref)

    y = list(range(n - 1, -1, -1))
    for i, (label, sub, val) in enumerate(stages):
        w = widths[i]
        left = (max_w - w) / 2
        ax.barh(y[i], w, left=left, height=0.62, color=colors[i], edgecolor="white", linewidth=1.5)
        val_str = f"{int(val)}" if i not in (6,) else f"~{int(neff_range[0])}-{int(neff_range[1])}"
        ax.text(
            0.5,
            y[i] + 0.10,
            f"{label}: {val_str}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white",
        )
        ax.text(0.5, y[i] - 0.16, sub, ha="center", va="center", fontsize=8.5, color="white", style="italic")

    # Attrition annotations on the right margin.
    drops = [
        (1, "0 lost: TVCalib succeeds on all 33"),
        (2, f"{n_ball_lost} lost: ball occluded at the spot"),
        (5, "pseudoreplication removed"),
        (6, "31 near-replicate frames -> ~1 obs/clip"),
    ]
    for idx, note in drops:
        ax.annotate(note, xy=(1.0, y[idx]), xytext=(1.06, y[idx]), fontsize=8, color="#444444", va="center", ha="left")

    ax.set_xlim(-0.02, 1.55)
    ax.set_ylim(-0.7, n - 0.3)
    ax.axis("off")
    ax.set_title(
        "Cohort attrition: from 33 discovered clips to the effective inferential unit\n"
        "Why effective sample size, not algorithmic quality, is the binding constraint",
        fontsize=12,
        fontweight="bold",
        loc="center",
    )
    fig.tight_layout()
    out = FIG_DIR / "18_cohort_funnel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    stages, neff_range = compute_stages()
    n_ball_lost = N_DISCOVERED - int(stages[2][2])
    out = render(stages, neff_range, n_ball_lost)
    print(f"Saved: {out}")
    print("Funnel stages:")
    for label, sub, val in stages:
        print(f"  {label}: {val}  ({sub})")
    print(f"  n_eff range: {neff_range[0]:.2f} - {neff_range[1]:.2f}")


if __name__ == "__main__":
    main()
