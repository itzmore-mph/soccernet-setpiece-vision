"""Compute GT pitch control over all 33 set-piece clips (not just 20 from baseline).

Reads:
    outputs/detections_gt_full.parquet  (team col = 'team', all 33 clips)
    outputs/ball_positions.parquet

Writes:
    outputs/pitch_control_gt_full.parquet  (track='gt', n=33 clips)
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
    balls = pd.read_parquet(OUTPUTS_DIR / "ball_positions.parquet")
    print(f"detections_gt_full: {gt.shape}  |  ball_positions: {balls.shape}")
    pc = process_track(gt, track_name="gt", team_col="team", balls=balls)
    out = OUTPUTS_DIR / "pitch_control_gt_full.parquet"
    pc.to_parquet(out, index=False)
    print(f"pitch_control_gt_full frames: {len(pc)}  |  clips: {pc['clip_id'].nunique()}  |  saved: {out}")
    print(pc[["pc_mean", "pc_at_ball", "pc_in_box", "pc_in_third", "pc_area_gt_0p5"]].mean().round(3))


if __name__ == "__main__":
    main()
