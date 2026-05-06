"""PC for Soccana detections under TVCalib H.

Reads:  outputs/detections_soccana_tvcalib.parquet
Writes: outputs/pitch_control_soccana_tvcalib.parquet  (track='soccana_tvcalib')
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from run_pc_tvcalib import process_track

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def main():
    det = pd.read_parquet(OUTPUTS_DIR / "detections_soccana_tvcalib.parquet")
    balls = pd.read_parquet(OUTPUTS_DIR / "ball_positions.parquet")
    print(f"detections: {det.shape}  |  ball_positions: {balls.shape}")
    pc = process_track(det, track_name="soccana_tvcalib", team_col="team_kmeans", balls=balls)
    out = OUTPUTS_DIR / "pitch_control_soccana_tvcalib.parquet"
    pc.to_parquet(out, index=False)
    print(f"frames: {len(pc)}  clips: {pc['clip_id'].nunique()}  saved: {out}")
    print(pc[["pc_mean", "pc_at_ball", "pc_in_box", "pc_in_third", "pc_area_gt_0p5"]].mean().round(3))


if __name__ == "__main__":
    main()
