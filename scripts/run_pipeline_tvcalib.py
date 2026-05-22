"""Phase 3 pipeline run: nb02 stages with TVCalib H replacing GT-pitch-line homography.

Reads:
    outputs/homographies_tvcalib.parquet
    SSD frames (SOCCERNET_LOCAL_DIR)

Writes:
    outputs/detections_pipeline_tvcalib.parquet  (mirrors detections_pipeline.parquet schema)
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from _pipeline_core import (
    DEVICE,
    discover_setpiece_clips,
    load_tvcalib_lookup,
    run_clip,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
GSR_ROOT = Path(os.getenv("SOCCERNET_LOCAL_DIR", "/Volumes/MPH-ExternalStorage/soccernet-gsr")) / "gamestate-2024"

YOLO_WEIGHTS = "yolov8x.pt"
YOLO_PERSON_CLASS = 0


def main() -> None:
    assert GSR_ROOT.exists(), f"SoccerNet GSR not mounted: {GSR_ROOT}"
    print("Loading TVCalib H lookup...")
    H_lookup = load_tvcalib_lookup(OUTPUTS_DIR)
    print(f"  {len(H_lookup)} (split, clip, frame) entries")

    clips = discover_setpiece_clips(GSR_ROOT)
    print(f"Set-piece clips: {len(clips)}")

    print(f"Loading YOLO {YOLO_WEIGHTS} on {DEVICE}...")
    yolo = YOLO(YOLO_WEIGHTS)

    all_rows: list[dict] = []
    skipped: list[tuple] = []

    for i, (_, clip) in enumerate(clips.iterrows()):
        rows, reason = run_clip(clip, yolo, H_lookup, player_class=YOLO_PERSON_CLASS)
        if reason is not None:
            skipped.append((clip["clip_id"], clip["action_position"], reason))
        else:
            all_rows.extend(rows)
        if (i + 1) % 5 == 0:
            print(f"  clips {i+1}/{len(clips)}  |  pipeline rows: {len(all_rows)}")

    df = pd.DataFrame(all_rows)
    out_path = OUTPUTS_DIR / "detections_pipeline_tvcalib.parquet"
    df.to_parquet(out_path, engine="pyarrow", index=False)
    print(f"\nDone. rows: {len(df)} | clips skipped: {len(skipped)} | saved: {out_path}")
    for c, fr, w in skipped[:15]:
        print(f"  skipped {c} (centre {fr}): {w}")
    print(f"Clips with rows: {df['clip_id'].nunique()}")


if __name__ == "__main__":
    main()
