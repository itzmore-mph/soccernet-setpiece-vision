"""Dump GT player detections for all 33 set-piece clips' frame windows.

Writes `outputs/detections_gt_full.parquet` containing ground-truth pitch
positions for every player annotation in the ±15-frame window around each
set-piece action.

Used by run_pc_gt_full.py and validation (ks_table_tvcalib.py).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
GSR = Path(os.getenv("SOCCERNET_LOCAL_DIR", "/Volumes/MPH-ExternalStorage/soccernet-gsr")) / "gamestate-2024"
SPLITS = ["train", "valid", "test", "challenge"]
TARGET_ACTIONS = {"Corner", "Direct free-kick"}
FRAME_WINDOW = 15
PITCH_L = 105.0
PITCH_W = 68.0


def main():
    rows = []
    n_clips = 0
    for split in SPLITS:
        split_dir = GSR / split
        if not split_dir.is_dir():
            continue
        for clip_dir in sorted(split_dir.glob("SNGS-*")):
            label_path = clip_dir / "Labels-GameState.json"
            if not label_path.is_file():
                continue
            try:
                with open(label_path) as f:
                    labels = json.load(f)
            except Exception:
                continue
            info = labels.get("info", {})
            if info.get("action_class") not in TARGET_ACTIONS:
                continue
            n_clips += 1
            n_frames = len(labels["images"])
            # action_position is a global broadcast frame number, not clip-local (1–750).
            # Clips start at the set-piece; use the first 2*FRAME_WINDOW+1 frames.
            centre = min(FRAME_WINDOW + 1, n_frames)
            lo = max(1, centre - FRAME_WINDOW)
            hi = min(n_frames, centre + FRAME_WINDOW)
            # build image_id index for this clip
            id_for_frame = {
                int(img["file_name"].removesuffix(".jpg")): img["image_id"]
                for img in labels["images"] if img.get("file_name")
            }
            # build per-image annotation index
            anns_by_image: dict = {}
            for a in labels["annotations"]:
                if a.get("category_id") not in (1, 2):
                    continue
                anns_by_image.setdefault(a.get("image_id"), []).append(a)
            for frame_idx in range(lo, hi + 1):
                image_id = id_for_frame.get(frame_idx)
                if image_id is None:
                    continue
                for a in anns_by_image.get(image_id, []):
                    bp = a.get("bbox_pitch")
                    if not bp:
                        continue
                    xc = bp.get("x_bottom_middle")
                    yc = bp.get("y_bottom_middle")
                    if xc is None or yc is None:
                        continue
                    x_m = float(xc) + PITCH_L / 2
                    y_m = float(yc) + PITCH_W / 2
                    _M = 2.0
                    if not (-_M <= x_m <= PITCH_L + _M and -_M <= y_m <= PITCH_W + _M):
                        continue
                    rows.append({
                        "split": split, "clip_id": clip_dir.name,
                        "action_class": info["action_class"],
                        "frame_idx": frame_idx,
                        "x_m": x_m,
                        "y_m": y_m,
                        "team": a["attributes"].get("team"),
                        "role": a["attributes"].get("role"),
                        "track_id": a.get("track_id"),
                    })
    df = pd.DataFrame(rows)
    out = OUTPUTS_DIR / "detections_gt_full.parquet"
    df.to_parquet(out, index=False)
    print(f"clips: {n_clips}  rows: {len(df)}  saved: {out}")
    print(f"clips with GT players: {df['clip_id'].nunique()}")


if __name__ == "__main__":
    main()
