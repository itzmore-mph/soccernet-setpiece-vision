"""PC for Soccana detections under TVCalib H.

Reads:  outputs/detections_soccana_tvcalib.parquet
Writes: outputs/pitch_control_soccana_tvcalib.parquet  (track='soccana_tvcalib')
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Ensure sibling scripts are importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _pipeline_core import process_track

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def _expand_balls_to_frames(balls_per_clip: pd.DataFrame, det: pd.DataFrame) -> pd.DataFrame:
    """Expand per-clip ball positions to per-frame rows matching process_track schema.

    The optimized pipeline produces ball_positions.parquet with one row per clip
    (columns: split, clip_id, x_pitch, y_pitch). process_track expects per-frame
    rows with columns: split, clip_id, frame_idx, ball_x_m, ball_y_m.

    For set-pieces the ball is stationary, so we broadcast the clip position to
    every frame present in the detections DataFrame.
    """
    # Handle both old schema (frame_idx, ball_x_m, ball_y_m) and new schema (x_pitch, y_pitch)
    if "frame_idx" in balls_per_clip.columns and "ball_x_m" in balls_per_clip.columns:
        return balls_per_clip  # Already in per-frame format

    # New schema: expand per-clip to per-frame
    frames = det[["split", "clip_id", "frame_idx"]].drop_duplicates()
    # Rename columns to match process_track expectations
    balls_renamed = balls_per_clip.rename(columns={"x_pitch": "ball_x_m", "y_pitch": "ball_y_m"})
    balls_renamed = balls_renamed[["split", "clip_id", "ball_x_m", "ball_y_m"]]
    # Merge: each frame gets the clip's ball position
    expanded = frames.merge(balls_renamed, on=["split", "clip_id"], how="inner")
    return expanded


def main():
    det = pd.read_parquet(OUTPUTS_DIR / "detections_soccana_tvcalib.parquet")
    balls_raw = pd.read_parquet(OUTPUTS_DIR / "ball_positions.parquet")
    balls = _expand_balls_to_frames(balls_raw, det)
    print(f"detections: {det.shape}  |  ball_positions (expanded): {balls.shape}")
    pc = process_track(det, track_name="soccana_tvcalib", team_col="team_kmeans", balls=balls)
    out = OUTPUTS_DIR / "pitch_control_soccana_tvcalib.parquet"
    pc.to_parquet(out, index=False)
    print(f"frames: {len(pc)}  clips: {pc['clip_id'].nunique()}  saved: {out}")
    print(pc[["pc_mean", "pc_at_ball", "pc_in_box", "pc_in_third", "pc_area_gt_0p5"]].mean().round(3))


if __name__ == "__main__":
    main()
