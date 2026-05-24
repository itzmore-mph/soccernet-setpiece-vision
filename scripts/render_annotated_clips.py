"""Render detection-style annotated broadcast clips.

Shows bounding boxes with team-colored labels, confidence scores, and track IDs
— similar to raw YOLO output but with team assignment applied. This visualizes
the detection + tracking + team assignment pipeline stages before pitch control.

Reads detections_soccana_tvcalib.parquet, loads broadcast frames from the SSD,
and writes one MP4 per clip to outputs/figures/annotated/.

Usage:
    python scripts/render_annotated_clips.py                      # all clips
    python scripts/render_annotated_clips.py --clip SNGS-066      # one clip
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures" / "annotated"
GSR_ROOT = Path(os.getenv("SOCCERNET_LOCAL_DIR", "/Volumes/MPH-ExternalStorage/soccernet-gsr")) / "gamestate-2024"

# Team colors (BGR)
TEAM_COLORS = {
    0: (220, 120, 20),   # blue team
    1: (20, 40, 220),    # red team
    -1: (100, 100, 100), # unassigned
}
TEAM_NAMES = {0: "Team A", 1: "Team B", -1: "?"}
FONT = cv2.FONT_HERSHEY_SIMPLEX
FPS = 8


def discover_clip_path(clip_id: str) -> Path | None:
    for split in ("train", "valid", "test", "challenge"):
        p = GSR_ROOT / split / clip_id
        if p.is_dir():
            return p
    return None


def draw_detection_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    team: int,
    track_id: int,
    conf: float,
) -> None:
    """Draw a YOLO-style detection box with label background."""
    color = TEAM_COLORS.get(team, TEAM_COLORS[-1])
    thickness = 2

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Label text
    label = f"{TEAM_NAMES[team]} #{track_id} {conf:.2f}"

    # Label background
    (tw, th), baseline = cv2.getTextSize(label, FONT, 0.5, 1)
    label_y1 = max(y1 - th - 8, 0)
    label_y2 = y1
    cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 6, label_y2), color, -1)

    # Label text (white on colored background)
    cv2.putText(frame, label, (x1 + 3, label_y2 - 4),
                FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_info_bar(
    frame: np.ndarray,
    clip_id: str,
    action_class: str,
    frame_idx: int,
    n_team_a: int,
    n_team_b: int,
) -> None:
    """Draw info bar at the top of the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent black bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Info text
    info = f"{clip_id} | {action_class} | frame {frame_idx} | Team A: {n_team_a} | Team B: {n_team_b}"
    cv2.putText(frame, info, (10, 25), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Team color legend
    legend_x = w - 220
    cv2.rectangle(frame, (legend_x, 8), (legend_x + 15, 23), TEAM_COLORS[0], -1)
    cv2.putText(frame, "Team A", (legend_x + 20, 22), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (legend_x + 90, 8), (legend_x + 105, 23), TEAM_COLORS[1], -1)
    cv2.putText(frame, "Team B", (legend_x + 110, 22), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


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
    action_class = dets["action_class"].iloc[0] if not dets.empty else "unknown"

    for fi in frame_indices:
        img_path = clip_path / "img1" / f"{fi:06d}.jpg"
        if not img_path.is_file():
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        n_a, n_b = 0, 0
        if fi in frame_dets.groups:
            rows = frame_dets.get_group(fi)
            for _, row in rows.iterrows():
                team = int(row["team_kmeans"])
                x1, y1 = int(row["x1_px"]), int(row["y1_px"])
                x2, y2 = int(row["x2_px"]), int(row["y2_px"])
                conf = float(row["conf"]) if "conf" in row.index else 0.0
                track_id = int(row["track_id"])

                draw_detection_box(frame, x1, y1, x2, y2, team, track_id, conf)

                if team == 0:
                    n_a += 1
                elif team == 1:
                    n_b += 1

        draw_info_bar(frame, clip_id, action_class, fi, n_a, n_b)

        writer.write(frame)
        written += 1

    writer.release()
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default=None, help="single clip ID, e.g. SNGS-066")
    args = parser.parse_args()

    parquet_path = OUTPUTS_DIR / "detections_soccana_tvcalib.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(f"parquet not found: {parquet_path}\nRun scripts/run_soccana_tvcalib.py first.")

    df = pd.read_parquet(parquet_path)

    required = {"x1_px", "y1_px", "x2_px", "y2_px"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"parquet missing pixel bbox columns {missing}.\nRe-run run_soccana_tvcalib.py to regenerate.")

    if args.clip:
        clip_ids = [args.clip]
    else:
        clip_ids = sorted(df["clip_id"].unique())

    print(f"Clips to render: {len(clip_ids)}")
    assert GSR_ROOT.exists(), f"SoccerNet GSR not mounted: {GSR_ROOT}"

    for clip_id in clip_ids:
        clip_dets = df[df["clip_id"] == clip_id]
        out_path = FIGURES_DIR / f"{clip_id}_detections.mp4"
        n = render_clip(clip_id, clip_dets, out_path)
        if n:
            print(f"  {clip_id}: {n} frames → {out_path.relative_to(PROJECT_ROOT)}")

    print("Done.")


if __name__ == "__main__":
    main()
