"""
Dump ball metric-pitch positions for all (split, clip_id, frame_idx) combos
present in detections_soccana_tvcalib.parquet or detections_gt_full.parquet.

Output: outputs/ball_positions.parquet (split, clip_id, frame_idx, ball_x_m, ball_y_m).

Reason: nb02 + nb03 need ball positions for pitch control computation.
Caching to parquet lets downstream notebooks run offline (without the SSD mounted).
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GSR = Path(os.getenv("SOCCERNET_LOCAL_DIR", "/Volumes/MPH-ExternalStorage/soccernet-gsr")) / "gamestate-2024"
OUT = PROJECT_ROOT / "outputs"
PITCH_L, PITCH_W = 105.0, 68.0


def main() -> None:
    pipe = pd.read_parquet(OUT / "detections_soccana_tvcalib.parquet")
    gt = pd.read_parquet(OUT / "detections_gt_full.parquet")
    sources = [pipe[["split", "clip_id", "frame_idx"]],
               gt[["split", "clip_id", "frame_idx"]]]
    keys = pd.concat(sources).drop_duplicates().reset_index(drop=True)

    rows = []
    labels_cache: dict = {}
    image_id_cache: dict = {}  # (split, clip_id) -> {filename: image_id}
    ann_ball_cache: dict = {}  # (split, clip_id) -> {image_id: (bx, by)}
    for _, k in keys.iterrows():
        cache_key = (k["split"], k["clip_id"])
        if cache_key not in labels_cache:
            label_path = GSR / k["split"] / k["clip_id"] / "Labels-GameState.json"
            try:
                with open(label_path) as f:
                    labels_cache[cache_key] = json.load(f)
            except Exception:
                labels_cache[cache_key] = None
            # Pre-build lookup indices for this clip
            labels = labels_cache[cache_key]
            if labels is not None:
                image_id_cache[cache_key] = {
                    img.get("file_name"): img.get("image_id")
                    for img in labels["images"]
                    if img.get("file_name")
                }
                ball_lookup: dict = {}
                for a in labels["annotations"]:
                    if a.get("category_id") != 4:
                        continue
                    bp = a.get("bbox_pitch")
                    if not bp:
                        continue
                    xc = bp.get("x_bottom_middle")
                    yc = bp.get("y_bottom_middle")
                    if xc is None or yc is None:
                        continue
                    ball_lookup[a.get("image_id")] = (
                        float(xc) + PITCH_L / 2,
                        float(yc) + PITCH_W / 2,
                    )
                ann_ball_cache[cache_key] = ball_lookup

        bx, by = np.nan, np.nan
        labels = labels_cache[cache_key]
        if labels is not None:
            fname = f"{int(k['frame_idx']):06d}.jpg"
            image_id = image_id_cache.get(cache_key, {}).get(fname)
            if image_id is not None:
                ball_pos = ann_ball_cache.get(cache_key, {}).get(image_id)
                if ball_pos is not None:
                    bx, by = ball_pos
        rows.append({**k.to_dict(), "ball_x_m": bx, "ball_y_m": by})

    df = pd.DataFrame(rows)
    out_path = OUT / "ball_positions.parquet"
    df.to_parquet(out_path, index=False)
    print(f"saved {out_path}  rows={len(df)}  with_ball={df['ball_x_m'].notna().sum()}")


if __name__ == "__main__":
    main()
