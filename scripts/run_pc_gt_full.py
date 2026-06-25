"""Compute GT pitch control over all 33 set-piece clips.

Uses the GT ball (annotation-derived, public) so the reference is independent of the
pipeline ball. Both inputs are committed public parquets, so this reproduces from
committed outputs without any raw video. Regenerate gt_ball_positions.parquet with
dump_gt_ball.py (needs the SoccerNet GSR dataset) only if it is missing.

Reads:
    outputs/detections_gt_full.parquet   (team col = 'team')
    outputs/gt_ball_positions.parquet    (GT ball per frame)

Writes:
    outputs/pitch_control_gt_full.parquet  (track='gt')
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


def main():
    gt = pd.read_parquet(OUTPUTS_DIR / "detections_gt_full.parquet")
    balls = pd.read_parquet(OUTPUTS_DIR / "gt_ball_positions.parquet")
    print(f"detections_gt_full: {gt.shape}  |  gt_ball_positions: {balls.shape}")
    pc = process_track(gt, track_name="gt", team_col="team", balls=balls)
    pc = pc.sort_values(["split", "clip_id", "frame_idx"]).reset_index(drop=True)
    out = OUTPUTS_DIR / "pitch_control_gt_full.parquet"
    pc.to_parquet(out, index=False)
    print(f"pitch_control_gt_full frames: {len(pc)}  |  clips: {pc['clip_id'].nunique()}  |  saved: {out}")
    print(pc[["pc_mean", "pc_at_ball", "pc_in_box", "pc_in_third", "pc_area_gt_0p5"]].mean().round(3))


if __name__ == "__main__":
    main()
