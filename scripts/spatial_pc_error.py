"""Spatial (cell-space) Pitch Control error map, pipeline vs GT.

The summary metrics collapse each 60x40 PC surface to five scalars. This script
recomputes both surfaces for every paired frame and accumulates the mean absolute
per-cell difference, turning the validation from five numbers into a spatial error
characterisation. It answers "where on the pitch does the pipeline disagree with
GT" and is expected to localise the error in the attacking third and penalty box.

The pipeline surface needs the video-derived detections parquet, so this script runs
only where it is present (locally or after regenerating from the SoccerNet GSR dataset). The OUTPUT is
aggregate (mean error per cell) and committed: detections in, aggregate out.

Reads:
    outputs/detections_soccana_tvcalib.parquet  (PRIVATE, pipeline detections)
    outputs/detections_gt_full.parquet           (GT detections)
    outputs/ball_positions.parquet               (PRIVATE, shared ball)

Writes:
    outputs/spatial_pc_error.parquet             (per-cell mean abs error, public)
    outputs/figures/17_spatial_pc_error.png      (mplsoccer heatmap)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch

from _pipeline_core import (
    GRID_NX,
    GRID_NY,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    pitch_control_surface,
    split_attack_defend,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

KEYS = ["split", "clip_id", "frame_idx"]
CLIP_KEYS = ["split", "clip_id"]


def _expand_balls(balls: pd.DataFrame, det: pd.DataFrame) -> pd.DataFrame:
    """Expand per-clip ball positions to per-frame rows (matches the PC scripts)."""
    if "frame_idx" in balls.columns and "ball_x_m" in balls.columns:
        return balls
    frames = det[KEYS].drop_duplicates()
    renamed = balls.rename(columns={"x_pitch": "ball_x_m", "y_pitch": "ball_y_m"})[CLIP_KEYS + ["ball_x_m", "ball_y_m"]]
    return frames.merge(renamed, on=CLIP_KEYS, how="inner")


def frame_surfaces(det: pd.DataFrame, team_col: str, balls: pd.DataFrame) -> dict[tuple, np.ndarray]:
    """Return {(split, clip, frame): 60x40 PC surface} replicating process_track.

    Each surface is oriented to a canonical attack-to-the-right frame: when the
    ball is in the left half (the team attacks left) the surface is mirrored
    horizontally. The flip decision uses the shared ball, so pipeline and GT
    surfaces for the same frame are always oriented identically. Without this,
    clips attacking opposite directions cancel and the spatial map is meaningless.
    """
    balls_f = _expand_balls(balls, det)
    ball_lookup = {
        (r.split, r.clip_id, r.frame_idx): (r.ball_x_m, r.ball_y_m)
        for r in balls_f.itertuples(index=False)
        if np.isfinite(r.ball_x_m) and np.isfinite(r.ball_y_m)
    }
    surfaces: dict[tuple, np.ndarray] = {}
    for (split, clip, frame), g in det.groupby(KEYS):
        key = (split, clip, frame)
        if key not in ball_lookup:
            continue
        ball_xy = ball_lookup[key]
        players_xy = g[["x_m", "y_m"]].to_numpy()
        teams = g[team_col].to_numpy()
        att_xy, def_xy, _ = split_attack_defend(players_xy, teams, ball_xy)
        if len(att_xy) == 0 or len(def_xy) == 0:
            continue
        surf = pitch_control_surface(att_xy, def_xy, ball_xy)
        if ball_xy[0] < PITCH_LENGTH_M / 2:
            surf = np.fliplr(surf)  # orient attack to the right
        surfaces[key] = surf
    return surfaces


def _canonical_frame_keys() -> set[tuple]:
    """Frame keys shared by the two committed PC parquets (the 662-frame cohort)."""
    pc_pipe = pd.read_parquet(OUTPUTS_DIR / "pitch_control_soccana_tvcalib.parquet")
    pc_gt = pd.read_parquet(OUTPUTS_DIR / "pitch_control_gt_full.parquet")
    kp = set(map(tuple, pc_pipe[KEYS].to_numpy()))
    kg = set(map(tuple, pc_gt[KEYS].to_numpy()))
    return kp & kg


def compute_cell_error(pipe_det: pd.DataFrame, gt_det: pd.DataFrame, balls: pd.DataFrame) -> pd.DataFrame:
    """Per-cell mean absolute PC difference over the canonical paired frames."""
    pipe_s = frame_surfaces(pipe_det, "team_kmeans", balls)
    gt_s = frame_surfaces(gt_det, "team", balls)
    # Restrict to the same frame cohort used by the rest of the validation.
    common = sorted(set(pipe_s) & set(gt_s) & _canonical_frame_keys())
    if not common:
        raise ValueError("No paired frames with computable surfaces in both tracks.")

    acc = np.zeros((GRID_NY, GRID_NX))
    for key in common:
        acc += np.abs(pipe_s[key] - gt_s[key])
    mean_err = acc / len(common)

    xs = np.linspace(0.0, PITCH_LENGTH_M, GRID_NX)
    ys = np.linspace(0.0, PITCH_WIDTH_M, GRID_NY)
    iy, ix = np.meshgrid(np.arange(GRID_NY), np.arange(GRID_NX), indexing="ij")
    return pd.DataFrame(
        {
            "iy": iy.ravel(),
            "ix": ix.ravel(),
            "x_m": xs[ix.ravel()],
            "y_m": ys[iy.ravel()],
            "mean_abs_err": mean_err.ravel(),
            "n_frames": len(common),
        }
    )


def render_heatmap(cells: pd.DataFrame) -> None:
    """mplsoccer pitch heatmap of the per-cell mean absolute PC error."""
    grid = cells.pivot(index="iy", columns="ix", values="mean_abs_err").to_numpy()
    pitch = Pitch(pitch_type="custom", pitch_length=PITCH_LENGTH_M, pitch_width=PITCH_WIDTH_M, line_zorder=2)
    fig, ax = pitch.draw(figsize=(9, 6))
    im = ax.imshow(
        grid,
        extent=(0, PITCH_LENGTH_M, 0, PITCH_WIDTH_M),
        origin="lower",
        cmap="magma",
        alpha=0.85,
        zorder=1,
        aspect="auto",
    )
    fig.colorbar(im, ax=ax, fraction=0.035, label="mean |PC pipeline - PC GT|")
    ax.set_title(
        f"Spatial Pitch Control error, pipeline vs GT (n={int(cells['n_frames'].iloc[0])} paired frames)\n"
        "oriented attack-to-right; right edge = attacking third",
        fontweight="bold",
    )
    out = FIG_DIR / "17_spatial_pc_error.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    pipe_det = pd.read_parquet(OUTPUTS_DIR / "detections_soccana_tvcalib.parquet")
    gt_det = pd.read_parquet(OUTPUTS_DIR / "detections_gt_full.parquet")
    balls = pd.read_parquet(OUTPUTS_DIR / "ball_positions.parquet")

    cells = compute_cell_error(pipe_det, gt_det, balls)
    out = OUTPUTS_DIR / "spatial_pc_error.parquet"
    cells.to_parquet(out, index=False)
    print(f"Paired frames: {int(cells['n_frames'].iloc[0])}")
    print(f"Per-cell mean abs error: overall={cells['mean_abs_err'].mean():.4f}  max={cells['mean_abs_err'].max():.4f}")

    # Thirds are attack-relative after orientation: x near 105 is the attacking third.
    third = PITCH_LENGTH_M / 3
    for name, lo, hi in [
        ("own third", 0, third),
        ("mid third", third, 2 * third),
        ("attacking third", 2 * third, PITCH_LENGTH_M),
    ]:
        sel = cells[(cells["x_m"] >= lo) & (cells["x_m"] < hi)]
        print(f"  {name}: mean abs error {sel['mean_abs_err'].mean():.4f}")
    print(f"\nSaved: {out}")

    render_heatmap(cells)


if __name__ == "__main__":
    main()
