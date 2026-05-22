"""Render annotated broadcast clips with team-colored player bboxes.

Reads detections_soccana_tvcalib.parquet (must contain x1_px/y1_px/x2_px/y2_px
columns), loads broadcast frames from the SSD, and writes one MP4 per clip to
outputs/figures/annotated/.

Usage:
    python scripts/render_annotated_clips.py                      # all clips
    python scripts/render_annotated_clips.py --clip SNGS-066      # one clip
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures" / "annotated"
GSR_ROOT = Path(os.getenv("SOCCERNET_LOCAL_DIR", "/Volumes/MPH-ExternalStorage/soccernet-gsr")) / "gamestate-2024"

DETECTOR_PARQUETS = {
    "soccana": OUTPUTS_DIR / "detections_soccana_tvcalib.parquet",
}

# BGR colors: team 0 = blue, team 1 = red, unknown = grey
TEAM_COLORS = {0: (220, 80, 20), 1: (20, 60, 220), -1: (120, 120, 120)}
FONT = cv2.FONT_HERSHEY_SIMPLEX
FPS = 10


def discover_clip_path(clip_id: str) -> Path | None:
    for split in ("train", "valid", "test", "challenge"):
        p = GSR_ROOT / split / clip_id
        if p.is_dir():
            return p
    return None


def render_clip(
    clip_id: str,
    dets: pd.DataFrame,
    out_path: Path,
) -> int:
    clip_path = discover_clip_path(clip_id)
    if clip_path is None:
        print(f"  [skip] {clip_id}: clip dir not found on SSD")
        return 0

    frame_indices = sorted(dets["frame_idx"].unique())
    if not frame_indices:
        print(f"  [skip] {clip_id}: no frames in detections")
        return 0

    # read first frame to get dimensions
    first_frame_path = clip_path / "img1" / f"{frame_indices[0]:06d}.jpg"
    if not first_frame_path.is_file():
        print(f"  [skip] {clip_id}: frame {frame_indices[0]} not found")
        return 0
    sample = cv2.imread(str(first_frame_path))
    if sample is None:
        return 0
    h, w = sample.shape[:2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (w, h))

    frame_dets = dets.groupby("frame_idx")
    written = 0

    for fi in frame_indices:
        img_path = clip_path / "img1" / f"{fi:06d}.jpg"
        if not img_path.is_file():
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        if fi in frame_dets.groups:
            rows = frame_dets.get_group(fi)
            for _, row in rows.iterrows():
                team = int(row["team_kmeans"])
                color = TEAM_COLORS.get(team, TEAM_COLORS[-1])
                x1, y1 = int(row["x1_px"]), int(row["y1_px"])
                x2, y2 = int(row["x2_px"]), int(row["y2_px"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"T{team} #{int(row['track_id'])}"
                label_y = max(y1 - 6, 12)
                cv2.putText(frame, label, (x1, label_y), FONT, 0.45, color, 1, cv2.LINE_AA)

        writer.write(frame)
        written += 1

    writer.release()
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default=None, help="single clip ID, e.g. SNGS-066")
    args = parser.parse_args()

    parquet_path = DETECTOR_PARQUETS["soccana"]
    if not parquet_path.is_file():
        raise FileNotFoundError(f"parquet not found: {parquet_path}\nRun scripts/run_soccana_tvcalib.py first.")

    df = pd.read_parquet(parquet_path)

    required = {"x1_px", "y1_px", "x2_px", "y2_px"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"parquet missing pixel bbox columns {missing}.\n"
            "Re-run run_soccana_tvcalib.py to regenerate."
        )

    if args.clip:
        clip_ids = [args.clip]
    else:
        clip_ids = sorted(df["clip_id"].unique())

    print(f"clips to render: {len(clip_ids)}")
    assert GSR_ROOT.exists(), f"SoccerNet GSR not mounted: {GSR_ROOT}"

    for clip_id in clip_ids:
        clip_dets = df[df["clip_id"] == clip_id]
        action = clip_dets["action_class"].iloc[0] if len(clip_dets) else "unknown"
        out_path = FIGURES_DIR / f"{clip_id}_soccana.mp4"
        n = render_clip(clip_id, clip_dets, out_path)
        if n:
            print(f"  {clip_id} ({action}): {n} frames -> {out_path.relative_to(PROJECT_ROOT)}")

    print("Done.")


if __name__ == "__main__":
    main()
