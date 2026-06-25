"""Dump ground-truth ball positions for the GT set-piece cohort.

The GT ball comes from the SoccerNet GSR annotations (`Labels-GameState.json`,
ball = `category_id` 4, `bbox_pitch` centred coordinates), which are open source.
The output parquet is therefore annotation-derived and public (committed), unlike
the video-derived `ball_positions.parquet`.

`run_pc_gt_full.py` reads this file so the GT Pitch Control reference is computed
from the *GT* ball rather than the pipeline ball, and reproduces from committed
parquets without any raw video.

Reads (SoccerNet GSR dataset):
    <SOCCERNET_LOCAL_DIR>/gamestate-2024/<split>/<clip_id>/Labels-GameState.json
Reads (cohort):
    outputs/detections_gt_full.parquet  (defines the split/clip/frame cohort)
Writes:
    outputs/gt_ball_positions.parquet   (split, clip_id, frame_idx, ball_x_m, ball_y_m)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Ensure sibling scripts are importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _pipeline_core import (
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    image_id_for_frame,
    set_deterministic,
    verify_soccernet_data,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
BALL_CATEGORY_ID = 4


def parse_gt_ball(labels: dict, frame_idx: int) -> tuple[float, float] | None:
    """Return the GT ball position for a frame as metric pitch (x_m, y_m), top-left origin.

    Parameters
    ----------
    labels : dict
        Parsed `Labels-GameState.json` for a clip.
    frame_idx : int
        1-based frame index within the clip.

    Returns
    -------
    tuple[float, float] | None
        (x_m, y_m) in 0-105 x 0-68 metres, or None if the frame has no ball annotation.
    """
    image_id = image_id_for_frame(labels, frame_idx)
    if image_id is None:
        return None
    for a in labels["annotations"]:
        if a.get("image_id") != image_id or a.get("category_id") != BALL_CATEGORY_ID:
            continue
        bp = a.get("bbox_pitch")
        if not bp:
            continue
        x_centred = bp.get("x_bottom_middle")
        y_centred = bp.get("y_bottom_middle")
        if x_centred is None or y_centred is None:
            continue
        return float(x_centred) + PITCH_LENGTH_M / 2, float(y_centred) + PITCH_WIDTH_M / 2
    return None


def main() -> None:
    set_deterministic()
    gsr_root = verify_soccernet_data()
    cohort = pd.read_parquet(OUTPUTS_DIR / "detections_gt_full.parquet")[["split", "clip_id", "frame_idx"]]
    cohort = cohort.drop_duplicates().sort_values(["split", "clip_id", "frame_idx"])

    labels_cache: dict[tuple[str, str], dict] = {}
    rows = []
    for split, clip_id, frame_idx in cohort.itertuples(index=False):
        key = (split, clip_id)
        if key not in labels_cache:
            label_path = gsr_root / split / clip_id / "Labels-GameState.json"
            labels_cache[key] = json.load(open(label_path)) if label_path.is_file() else {}
        labels = labels_cache[key]
        bp = parse_gt_ball(labels, int(frame_idx)) if labels else None
        if bp is None:
            continue
        rows.append(
            {
                "split": split,
                "clip_id": clip_id,
                "frame_idx": int(frame_idx),
                "ball_x_m": bp[0],
                "ball_y_m": bp[1],
            }
        )

    balls = pd.DataFrame(rows)
    out = OUTPUTS_DIR / "gt_ball_positions.parquet"
    balls.to_parquet(out, index=False)
    print(f"gt_ball_positions: {balls.shape}  |  clips: {balls['clip_id'].nunique()}  |  saved: {out}")


if __name__ == "__main__":
    main()
