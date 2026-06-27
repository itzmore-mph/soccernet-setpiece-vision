"""Render detection-style annotated broadcast clips.

Shows bounding boxes with team-colored labels, confidence scores, and track IDs.
Players colored by KMeans team assignment; referees in orange (Soccana class 2).

Reads detections_soccana_tvcalib.parquet, loads broadcast frames from the SoccerNet GSR dataset,
and writes one MP4 per clip to outputs/figures/annotated/.

Usage:
    python scripts/render_annotated_clips.py                      # all clips
    python scripts/render_annotated_clips.py --clip SNGS-066      # one clip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Ensure sibling scripts are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _pipeline_core import PITCH_LENGTH_M, PITCH_WIDTH_M, ensure_h264_playback, verify_soccernet_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures" / "annotated"
GSR_ROOT = verify_soccernet_data(str(PROJECT_ROOT / ".env"))

# Colors (BGR)
TEAM_COLORS = {
    0: (220, 120, 20),  # blue team
    1: (20, 40, 220),  # red team
    2: (30, 130, 255),  # third cluster (often referees/outliers)
    -1: (100, 100, 100),  # unassigned
}
REFEREE_COLOR = (30, 130, 255)  # orange
TEAM_NAMES = {0: "Team A", 1: "Team B", 2: "Other", -1: "?"}
FONT = cv2.FONT_HERSHEY_SIMPLEX
FPS = 8


def discover_clip_path(clip_id: str) -> Path | None:
    for split in ("train", "valid", "test", "challenge"):
        p = GSR_ROOT / split / clip_id
        if p.is_dir():
            return p
    return None


def remap_other_cluster(dets: pd.DataFrame) -> pd.DataFrame:
    """Reassign the KMeans "other" cluster (team_kmeans == 2) to Team A/B.

    KMeans is fit with k=3 and should collapse to two real teams plus
    referees, but on some clips a third jersey-colour cluster survives among
    non-referee detections (e.g. SNGS-122: 214/588 rows). For display only,
    each such row is assigned to whichever of team 0 / team 1 has the nearer
    mean HSV centroid in that frame, since the root cause is jersey-colour
    clustering, not pitch position. Does not touch the source parquet.
    """
    dets = dets.copy()
    hsv_cols = ["hsv_h", "hsv_s", "hsv_v"]

    for fi in dets["frame_idx"].unique():
        frame_mask = dets["frame_idx"] == fi
        other_mask = frame_mask & (dets["team_kmeans"] == 2) & ~dets["is_referee"]
        if not other_mask.any():
            continue
        centroids = {}
        for t in (0, 1):
            team_rows = dets[frame_mask & (dets["team_kmeans"] == t) & ~dets["is_referee"]]
            if not team_rows.empty:
                centroids[t] = team_rows[hsv_cols].mean().to_numpy()
        if len(centroids) < 2:
            continue
        for idx in dets.index[other_mask]:
            hsv = dets.loc[idx, hsv_cols].to_numpy(dtype=float)
            dists = {t: np.linalg.norm(hsv - c) for t, c in centroids.items()}
            dets.loc[idx, "team_kmeans"] = min(dists, key=dists.get)

    return dets


def load_homography_lookup() -> dict[tuple[str, str, int], np.ndarray]:
    df = pd.read_parquet(OUTPUTS_DIR / "homographies_tvcalib.parquet")
    out: dict = {}
    for _, r in df.iterrows():
        H = np.array(
            [
                [r["h00"], r["h01"], r["h02"]],
                [r["h10"], r["h11"], r["h12"]],
                [r["h20"], r["h21"], r["h22"]],
            ]
        )
        out[(r["split"], r["clip_id"], int(r["frame_idx"]))] = H
    return out


def project_ball_to_image(ball_x_m: float, ball_y_m: float, H_world_to_image: np.ndarray) -> tuple[float, float] | None:
    x_c, y_c = ball_x_m - PITCH_LENGTH_M / 2, ball_y_m - PITCH_WIDTH_M / 2
    proj = H_world_to_image @ np.array([x_c, y_c, 1.0])
    if abs(proj[2]) < 1e-8:
        return None
    return (proj[0] / proj[2], proj[1] / proj[2])


def draw_ball(frame: np.ndarray, ball_xy_px: tuple[float, float]) -> None:
    bx, by = int(ball_xy_px[0]), int(ball_xy_px[1])
    h, w = frame.shape[:2]
    if not (0 <= bx < w and 0 <= by < h):
        return
    cv2.circle(frame, (bx, by), 10, (0, 0, 0), 3)
    cv2.circle(frame, (bx, by), 10, (0, 255, 255), -1)
    cv2.circle(frame, (bx, by), 10, (255, 255, 255), 1)


def draw_detection_box(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    team: int,
    track_id: int,
    conf: float,
    is_referee: bool = False,
) -> None:
    """Draw a YOLO-style detection box with label background."""
    if is_referee:
        color = REFEREE_COLOR
        label = f"Referee {conf:.2f}"
    else:
        color = TEAM_COLORS.get(team, TEAM_COLORS[-1])
        label = f"{TEAM_NAMES[team]} #{track_id} {conf:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    (tw, th), baseline = cv2.getTextSize(label, FONT, 0.5, 1)
    label_y1 = max(y1 - th - 8, 0)
    label_y2 = y1
    cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 6, label_y2), color, -1)
    cv2.putText(frame, label, (x1 + 3, label_y2 - 4), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_info_bar(
    frame: np.ndarray,
    clip_id: str,
    action_class: str,
    frame_idx: int,
    n_team_a: int,
    n_team_b: int,
    n_refs: int = 0,
) -> None:
    """Draw info bar at the top of the frame."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    info = f"{clip_id} | {action_class} | frame {frame_idx} | A: {n_team_a}  B: {n_team_b}  Ref: {n_refs}"
    cv2.putText(frame, info, (10, 25), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    legend_x = w - 310
    cv2.rectangle(frame, (legend_x, 8), (legend_x + 15, 23), TEAM_COLORS[0], -1)
    cv2.putText(frame, "Team A", (legend_x + 20, 22), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (legend_x + 90, 8), (legend_x + 105, 23), TEAM_COLORS[1], -1)
    cv2.putText(frame, "Team B", (legend_x + 110, 22), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (legend_x + 180, 8), (legend_x + 195, 23), REFEREE_COLOR, -1)
    cv2.putText(frame, "Ref", (legend_x + 200, 22), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def render_clip(
    clip_id: str,
    dets: pd.DataFrame,
    out_path: Path,
    balls: pd.DataFrame | None = None,
    H_lookup: dict | None = None,
) -> int:
    clip_path = discover_clip_path(clip_id)
    if clip_path is None:
        print(f"  [skip] {clip_id}: clip dir not found in SoccerNet GSR dataset")
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
    split = dets["split"].iloc[0] if not dets.empty else None

    ball_xy_m = None
    if balls is not None:
        ball_row = balls[balls["clip_id"] == clip_id]
        x_col = "ball_x_m" if "ball_x_m" in ball_row.columns else "x_pitch"
        y_col = "ball_y_m" if "ball_y_m" in ball_row.columns else "y_pitch"
        if not ball_row.empty and pd.notna(ball_row[x_col].iloc[0]):
            ball_xy_m = (float(ball_row[x_col].iloc[0]), float(ball_row[y_col].iloc[0]))

    for fi in frame_indices:
        img_path = clip_path / "img1" / f"{fi:06d}.jpg"
        if not img_path.is_file():
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        n_a, n_b, n_refs = 0, 0, 0
        if fi in frame_dets.groups:
            rows = frame_dets.get_group(fi)
            has_ref_col = "is_referee" in rows.columns
            for _, row in rows.iterrows():
                team = int(row["team_kmeans"])
                x1, y1 = int(row["x1_px"]), int(row["y1_px"])
                x2, y2 = int(row["x2_px"]), int(row["y2_px"])
                conf = float(row["conf"]) if "conf" in row.index else 0.0
                track_id = int(row["track_id"])
                is_ref = bool(row["is_referee"]) if has_ref_col else False

                draw_detection_box(frame, x1, y1, x2, y2, team, track_id, conf, is_ref)

                if is_ref:
                    n_refs += 1
                elif team == 0:
                    n_a += 1
                elif team == 1:
                    n_b += 1

        if ball_xy_m is not None and H_lookup is not None:
            H = H_lookup.get((split, clip_id, fi))
            if H is not None:
                ball_px = project_ball_to_image(ball_xy_m[0], ball_xy_m[1], H)
                if ball_px is not None:
                    draw_ball(frame, ball_px)

        draw_info_bar(frame, clip_id, action_class, fi, n_a, n_b, n_refs)

        writer.write(frame)
        written += 1

    writer.release()
    if written:
        ensure_h264_playback(out_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default=None, help="single clip ID, e.g. SNGS-066")
    args = parser.parse_args()

    parquet_path = OUTPUTS_DIR / "detections_soccana_tvcalib.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(f"parquet not found: {parquet_path}\nRun scripts/run_optimized_pipeline.py first.")

    df = pd.read_parquet(parquet_path)

    required = {"x1_px", "y1_px", "x2_px", "y2_px"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"parquet missing pixel bbox columns {missing}.\nRe-run run_optimized_pipeline.py to regenerate."
        )

    if args.clip:
        clip_ids = [args.clip]
    else:
        clip_ids = sorted(df["clip_id"].unique())

    print(f"Clips to render: {len(clip_ids)}")
    verify_soccernet_data()

    balls_path = OUTPUTS_DIR / "ball_positions.parquet"
    balls = pd.read_parquet(balls_path) if balls_path.is_file() else None
    H_lookup = load_homography_lookup() if (OUTPUTS_DIR / "homographies_tvcalib.parquet").is_file() else None

    for clip_id in clip_ids:
        clip_dets = remap_other_cluster(df[df["clip_id"] == clip_id])
        out_path = FIGURES_DIR / f"{clip_id}_detections.mp4"
        n = render_clip(clip_id, clip_dets, out_path, balls=balls, H_lookup=H_lookup)
        if n:
            print(f"  {clip_id}: {n} frames → {out_path.relative_to(PROJECT_ROOT)}")

    print("Done.")


if __name__ == "__main__":
    main()
