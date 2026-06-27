"""Render a clean cold-open hook clip: broadcast frame fading into a Pitch
Control heatmap, no clip ID, no metric bars, no frame counter.

Built for the explanatory-video intro (Segment 0), where the cue is
"broadcast clip of a corner kick, then same frame with a Pitch Control
heatmap fading in over it" — deliberately stripped of the technical overlay
chrome used in render_pc_overlay.py, since the hook needs to read at a
glance, not like a dashboard.

Usage:
    python scripts/render_hook_clip.py --clip SNGS-116

Requires the local SoccerNet GSR dataset (reads broadcast JPEGs).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _pipeline_core import (
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    ensure_h264_playback,
    pitch_control_surface,
    split_attack_defend,
    verify_soccernet_data,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures" / "hook"
GSR_ROOT = verify_soccernet_data(str(PROJECT_ROOT / ".env"))

GRID_NX, GRID_NY = 60, 40
SOURCE_FPS = 6
OUTPUT_FPS = 24  # tween factor 4x for smoother playback
TWEEN = OUTPUT_FPS // SOURCE_FPS
TARGET_ALPHA = 0.35  # final heatmap strength; lower than the dashboard variant so
# saturated cells (common at a corner, where defenders cluster and attackers have
# open space) still show grass through instead of reading as a flat color block
FADE_IN_FRAMES = 8  # source frames over which the heatmap ramps from 0 -> TARGET_ALPHA
HOLD_BROADCAST_FRAMES = 3  # leading source frames shown with no heatmap at all
PLAYER_DOT_RADIUS = 8
TEAM_COLORS_BGR = {0: (220, 120, 20), 1: (20, 40, 220), 2: (30, 130, 255), -1: (100, 100, 100)}


def make_pc_colormap(pc_surface: np.ndarray) -> np.ndarray:
    """Pitch control [0,1] -> BGR colormap. 0=red (defend), 1=blue (attack).

    Linear intensity (no power curve) so saturated regions stay a visible
    gradient instead of crushing to a flat block once most of the grid is
    past ~0.8, which is common at a corner with few defenders far away.
    """
    pc_clipped = np.clip(pc_surface, 0, 1)
    att_intensity = np.where(pc_clipped > 0.5, (pc_clipped - 0.5) * 2.0, 0.0)
    def_intensity = np.where(pc_clipped < 0.5, (0.5 - pc_clipped) * 2.0, 0.0)
    ny, nx = pc_surface.shape
    img = np.zeros((ny, nx, 3), dtype=np.uint8)
    img[:, :, 0] = (att_intensity * 200).astype(np.uint8)
    img[:, :, 2] = (def_intensity * 200).astype(np.uint8)
    return img


def draw_players(frame: np.ndarray, dets: pd.DataFrame, H: np.ndarray, img_w: int, img_h: int) -> None:
    """Small team-coloured dots at detected player positions, no labels."""
    for _, row in dets.iterrows():
        cx = int((row["x1_px"] + row["x2_px"]) / 2)
        cy = int(row["y2_px"])
        if not (0 <= cx < img_w and 0 <= cy < img_h):
            continue
        color = TEAM_COLORS_BGR.get(int(row["team_kmeans"]), TEAM_COLORS_BGR[-1])
        cv2.circle(frame, (cx, cy), PLAYER_DOT_RADIUS, (0, 0, 0), 2)
        cv2.circle(frame, (cx, cy), PLAYER_DOT_RADIUS, color, -1)


def warp_pc_to_image(pc_surface: np.ndarray, H_world_to_image: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    ny, nx = pc_surface.shape
    pc_color = make_pc_colormap(pc_surface)
    T_topleft_to_centred = np.array([[1, 0, -PITCH_LENGTH_M / 2], [0, 1, -PITCH_WIDTH_M / 2], [0, 0, 1]])
    T_grid_to_topleft = np.array([[PITCH_LENGTH_M / nx, 0, 0], [0, PITCH_WIDTH_M / ny, 0], [0, 0, 1]])
    H_grid_to_image = H_world_to_image @ T_topleft_to_centred @ T_grid_to_topleft
    return cv2.warpPerspective(
        pc_color,
        H_grid_to_image,
        (img_w, img_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def project_ball_to_image(ball_x_m: float, ball_y_m: float, H_world_to_image: np.ndarray) -> tuple[float, float] | None:
    x_c, y_c = ball_x_m - PITCH_LENGTH_M / 2, ball_y_m - PITCH_WIDTH_M / 2
    proj = H_world_to_image @ np.array([x_c, y_c, 1.0])
    if abs(proj[2]) < 1e-8:
        return None
    px, py = proj[0] / proj[2], proj[1] / proj[2]
    return (px, py)


def load_homography_lookup() -> dict[tuple[str, str, int], np.ndarray]:
    df = pd.read_parquet(OUTPUTS_DIR / "homographies_tvcalib.parquet")
    out = {}
    for _, r in df.iterrows():
        H = np.array([[r["h00"], r["h01"], r["h02"]], [r["h10"], r["h11"], r["h12"]], [r["h20"], r["h21"], r["h22"]]])
        out[(r["split"], r["clip_id"], int(r["frame_idx"]))] = H
    return out


def render(clip_id: str, out_path: Path) -> int:
    dets_all = pd.read_parquet(OUTPUTS_DIR / "detections_soccana_tvcalib.parquet")
    balls = pd.read_parquet(OUTPUTS_DIR / "ball_positions.parquet")
    H_lookup = load_homography_lookup()

    dets = dets_all[dets_all["clip_id"] == clip_id]
    split = dets["split"].iloc[0]
    clip_path = GSR_ROOT / split / clip_id
    frame_indices = sorted(dets["frame_idx"].unique())

    ball_row = balls[balls["clip_id"] == clip_id]
    x_col = "ball_x_m" if "ball_x_m" in ball_row.columns else "x_pitch"
    y_col = "ball_y_m" if "ball_y_m" in ball_row.columns else "y_pitch"
    ball_x = float(ball_row[x_col].iloc[0])
    ball_y = float(ball_row[y_col].iloc[0])

    sample = cv2.imread(str(clip_path / "img1" / f"{frame_indices[0]:06d}.jpg"))
    img_h, img_w = sample.shape[:2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, OUTPUT_FPS, (img_w, img_h))

    prev_blended = None
    written = 0

    for idx, fi in enumerate(frame_indices):
        frame = cv2.imread(str(clip_path / "img1" / f"{fi:06d}.jpg"))
        if frame is None:
            continue

        H = H_lookup.get((split, clip_id, fi))
        f_dets = dets[dets["frame_idx"] == fi]

        ramp = np.clip((idx - HOLD_BROADCAST_FRAMES) / FADE_IN_FRAMES, 0.0, 1.0)
        alpha_now = TARGET_ALPHA * ramp

        if H is not None and alpha_now > 0 and not f_dets.empty:
            players_xy = f_dets[["x_m", "y_m"]].to_numpy()
            teams = f_dets["team_kmeans"].to_numpy()
            att_xy, def_xy, _ = split_attack_defend(players_xy, teams, (ball_x, ball_y))
            if len(att_xy) > 0 and len(def_xy) > 0:
                pc = pitch_control_surface(att_xy, def_xy, (ball_x, ball_y))
                pc_warped = warp_pc_to_image(pc, H, img_w, img_h)
                mask = np.clip(pc_warped.astype(np.float32).sum(axis=2) / 255.0, 0, 1)
                mask_3ch = np.stack([mask] * 3, axis=2)
                frame_f = frame.astype(np.float32)
                pc_f = pc_warped.astype(np.float32)
                blended = frame_f * (1 - alpha_now * mask_3ch) + pc_f * (alpha_now * mask_3ch)
                frame = blended.astype(np.uint8)

        if H is not None and not f_dets.empty:
            draw_players(frame, f_dets, H, img_w, img_h)

        if H is not None and ball_x is not None:
            ball_px = project_ball_to_image(ball_x, ball_y, H)
            if ball_px is not None:
                bx, by = int(ball_px[0]), int(ball_px[1])
                if 0 <= bx < img_w and 0 <= by < img_h:
                    cv2.circle(frame, (bx, by), 10, (0, 0, 0), 3)
                    cv2.circle(frame, (bx, by), 10, (0, 255, 255), -1)
                    cv2.circle(frame, (bx, by), 10, (255, 255, 255), 1)

        # Tween toward the previous output frame for smoother apparent motion
        if prev_blended is not None:
            for t in range(1, TWEEN):
                w = t / TWEEN
                tween_frame = cv2.addWeighted(prev_blended, 1 - w, frame, w, 0)
                writer.write(tween_frame)
                written += 1
        writer.write(frame)
        written += 1
        prev_blended = frame

    writer.release()
    if written:
        ensure_h264_playback(out_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Render clean cold-open hook clip")
    parser.add_argument("--clip", default="SNGS-116")
    args = parser.parse_args()
    out_path = FIGURES_DIR / f"{args.clip}_hook.mp4"
    n = render(args.clip, out_path)
    print(f"{args.clip}: {n} frames -> {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
