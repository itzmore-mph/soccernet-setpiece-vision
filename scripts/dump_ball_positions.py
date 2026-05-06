"""
Dump ball metric-pitch positions for all (split, clip_id, frame_idx) combos
present in detections_pipeline.parquet or detections_gt.parquet.

Output: outputs/ball_positions.parquet (split, clip_id, frame_idx, ball_x_m, ball_y_m).

Reason: nb03 currently parses Labels-GameState.json from the SSD per frame.
Caching to parquet lets nb03 + nb04 run offline (without the SSD mounted).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

GSR = Path("/Volumes/MPH-ExternalStorage/soccernet-gsr/gamestate-2024")
OUT = Path("outputs")
PITCH_L, PITCH_W = 105.0, 68.0


def main() -> None:
    pipe = pd.read_parquet(OUT / "detections_pipeline.parquet")
    gt = pd.read_parquet(OUT / "detections_gt.parquet")
    sources = [pipe[["split", "clip_id", "frame_idx"]],
               gt[["split", "clip_id", "frame_idx"]]]
    tv_path = OUT / "detections_pipeline_tvcalib.parquet"
    if tv_path.is_file():
        tv = pd.read_parquet(tv_path)
        sources.append(tv[["split", "clip_id", "frame_idx"]])
    keys = pd.concat(sources).drop_duplicates().reset_index(drop=True)

    rows = []
    labels_cache: dict = {}
    for _, k in keys.iterrows():
        cache_key = (k["split"], k["clip_id"])
        if cache_key not in labels_cache:
            label_path = GSR / k["split"] / k["clip_id"] / "Labels-GameState.json"
            try:
                labels_cache[cache_key] = json.load(open(label_path))
            except Exception:
                labels_cache[cache_key] = None
        labels = labels_cache[cache_key]
        bx, by = np.nan, np.nan
        if labels is not None:
            fname = f"{int(k['frame_idx']):06d}.jpg"
            image_id = next(
                (img["image_id"] for img in labels["images"]
                 if img.get("file_name") == fname),
                None,
            )
            if image_id is not None:
                for a in labels["annotations"]:
                    if a.get("image_id") != image_id or a.get("category_id") != 4:
                        continue
                    bp = a.get("bbox_pitch")
                    if not bp:
                        continue
                    xc = bp.get("x_bottom_middle")
                    yc = bp.get("y_bottom_middle")
                    if xc is None or yc is None:
                        continue
                    bx = float(xc) + PITCH_L / 2
                    by = float(yc) + PITCH_W / 2
                    break
        rows.append({**k.to_dict(), "ball_x_m": bx, "ball_y_m": by})

    df = pd.DataFrame(rows)
    out_path = OUT / "ball_positions.parquet"
    df.to_parquet(out_path, index=False)
    print(f"saved {out_path}  rows={len(df)}  with_ball={df['ball_x_m'].notna().sum()}")


if __name__ == "__main__":
    main()
