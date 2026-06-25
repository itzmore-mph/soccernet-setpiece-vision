"""Render pitch control overlay directly on broadcast frames.

Warps the pitch control heatmap back to image coordinates using the inverse
homography, then alpha-blends it onto the broadcast frame. Produces a single
MP4 per clip showing the evolving pitch control surface overlaid on the actual
broadcast footage — the most intuitive visualization of the pipeline output.

Features:
- Pitch control heatmap warped to broadcast perspective
- Team-coloured player dots at detected positions
- Ball marker
- pc_in_box metric bar at the bottom
- Frame counter

Usage:
    python scripts/render_pc_overlay.py                      # all clips
    python scripts/render_pc_overlay.py --clip SNGS-066      # single clip

Requires the local SoccerNet GSR dataset (reads broadcast JPEGs).
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

from _pipeline_core import (
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    pitch_control_surface,
    split_attack_defend,
    verify_soccernet_data,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures" / "overlay"
GSR_ROOT = verify_soccernet_data(str(PROJECT_ROOT / ".env"))

# Visualization parameters
GRID_NX, GRID_NY = 60, 40
FPS = 6
ALPHA = 0.35  # heatmap transparency (lower = more broadcast visible)
TEAM_COLORS_BGR = {0: (220, 80, 20), 1: (20, 60, 220), -1: (120, 120, 120)}
BALL_COLOR_BGR = (0, 255, 255)  # yellow


def load_homography_lookup() -> dict[tuple[str, str, int], np.ndarray]:
    """Load H_world_to_image (centred metres → pixels) from parquet."""
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


def make_pc_colormap(pc_surface: np.ndarray) -> np.ndarray:
    """Convert pitch control [0,1] array to BGR colormap image with alpha.

    0.0 = full red (defending), 0.5 = transparent, 1.0 = full blue (attacking).
    Areas near 0.5 are nearly transparent to avoid flooding the frame.
    """
    ny, nx = pc_surface.shape
    img = np.zeros((ny, nx, 3), dtype=np.uint8)

    pc_clipped = np.clip(pc_surface, 0, 1)

    # Intensity ramps from 0 at 0.5 to 1 at extremes, with power curve for contrast
    att_raw = np.where(pc_clipped > 0.5, (pc_clipped - 0.5) * 2.0, 0.0)
    def_raw = np.where(pc_clipped < 0.5, (0.5 - pc_clipped) * 2.0, 0.0)
    att_intensity = np.power(att_raw, 1.5)
    def_intensity = np.power(def_raw, 1.5)

    # BGR: Blue for attackers, Red for defenders
    img[:, :, 0] = (att_intensity * 220).astype(np.uint8)  # B
    img[:, :, 2] = (def_intensity * 220).astype(np.uint8)  # R

    return img


def warp_pc_to_image(
    pc_surface: np.ndarray,
    H_world_to_image: np.ndarray,
    img_w: int = 1920,
    img_h: int = 1080,
) -> np.ndarray:
    """Warp pitch control surface from pitch coordinates to image coordinates.

    The PC surface is defined on a grid over [0, 105] x [0, 68] (top-left origin).
    H_world_to_image maps centred coordinates (±52.5, ±34) to pixels.
    We need to compose with the top-left → centred transform.
    """
    ny, nx = pc_surface.shape

    # Create the colored PC image on the pitch grid
    pc_color = make_pc_colormap(pc_surface)

    # Transform: top-left metres → centred metres
    T_topleft_to_centred = np.array(
        [
            [1.0, 0.0, -PITCH_LENGTH_M / 2],
            [0.0, 1.0, -PITCH_WIDTH_M / 2],
            [0.0, 0.0, 1.0],
        ]
    )

    # Scale from PC grid pixels to pitch metres (top-left)
    # PC image is (ny, nx) representing [0, PITCH_LENGTH_M] x [0, PITCH_WIDTH_M]
    T_grid_to_topleft = np.array(
        [
            [PITCH_LENGTH_M / nx, 0.0, 0.0],
            [0.0, PITCH_WIDTH_M / ny, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Full transform: PC grid → top-left metres → centred metres → image pixels
    H_grid_to_image = H_world_to_image @ T_topleft_to_centred @ T_grid_to_topleft

    # Warp the colored PC image to broadcast frame coordinates
    warped = cv2.warpPerspective(
        pc_color,
        H_grid_to_image,
        (img_w, img_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return warped


def compute_pc_in_box(pc_surface: np.ndarray, ball_x: float) -> float:
    """Compute mean PC in the attacking penalty box."""
    ny, nx = pc_surface.shape
    xs = np.linspace(0, PITCH_LENGTH_M, nx)
    ys = np.linspace(0, PITCH_WIDTH_M, ny)
    XX, YY = np.meshgrid(xs, ys)

    if ball_x < PITCH_LENGTH_M / 2:
        box_mask = (XX <= 16.5) & (YY >= 13.84) & (YY <= 54.16)
    else:
        box_mask = (XX >= PITCH_LENGTH_M - 16.5) & (YY >= 13.84) & (YY <= 54.16)

    if box_mask.any():
        return float(pc_surface[box_mask].mean())
    return 0.5


def draw_metric_bar(frame: np.ndarray, pc_in_box: float, pc_at_ball: float) -> None:
    """Draw a metric bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_h = 40
    bar_y = h - bar_h

    # Semi-transparent black bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_y), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # PC in box bar (left side)
    bar_w = 200
    bar_x = 20
    cv2.rectangle(frame, (bar_x, bar_y + 8), (bar_x + bar_w, bar_y + 28), (60, 60, 60), -1)
    fill_w = int(bar_w * np.clip(pc_in_box, 0, 1))
    color = (220, 80, 20) if pc_in_box > 0.5 else (20, 60, 220)
    cv2.rectangle(frame, (bar_x, bar_y + 8), (bar_x + fill_w, bar_y + 28), color, -1)
    cv2.putText(
        frame,
        f"PC in box: {pc_in_box:.2f}",
        (bar_x + bar_w + 10, bar_y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # PC at ball (right side)
    bar_x2 = w // 2
    cv2.rectangle(frame, (bar_x2, bar_y + 8), (bar_x2 + bar_w, bar_y + 28), (60, 60, 60), -1)
    fill_w2 = int(bar_w * np.clip(pc_at_ball, 0, 1))
    color2 = (220, 80, 20) if pc_at_ball > 0.5 else (20, 60, 220)
    cv2.rectangle(frame, (bar_x2, bar_y + 8), (bar_x2 + fill_w2, bar_y + 28), color2, -1)
    cv2.putText(
        frame,
        f"PC at ball: {pc_at_ball:.2f}",
        (bar_x2 + bar_w + 10, bar_y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_players_and_ball(
    frame: np.ndarray,
    dets: pd.DataFrame,
    ball_xy_px: tuple[float, float] | None,
) -> None:
    """Draw player circles and ball on the frame."""
    for _, row in dets.iterrows():
        cx = int((row["x1_px"] + row["x2_px"]) / 2)
        cy = int(row["y2_px"])  # foot position
        team = int(row["team_kmeans"])
        color = TEAM_COLORS_BGR.get(team, TEAM_COLORS_BGR[-1])
        cv2.circle(frame, (cx, cy), 12, (0, 0, 0), 3)  # black outline
        cv2.circle(frame, (cx, cy), 12, color, -1)  # filled
        cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 2)  # white ring

    if ball_xy_px is not None:
        bx, by = int(ball_xy_px[0]), int(ball_xy_px[1])
        cv2.circle(frame, (bx, by), 14, (0, 0, 0), 3)
        cv2.circle(frame, (bx, by), 14, BALL_COLOR_BGR, -1)
        cv2.circle(frame, (bx, by), 14, (255, 255, 255), 2)


def project_ball_to_image(ball_x_m: float, ball_y_m: float, H_world_to_image: np.ndarray) -> tuple[float, float] | None:
    """Project ball pitch position (top-left metres) to image pixels."""
    # Convert top-left to centred
    x_c = ball_x_m - PITCH_LENGTH_M / 2
    y_c = ball_y_m - PITCH_WIDTH_M / 2
    pt = np.array([x_c, y_c, 1.0])
    proj = H_world_to_image @ pt
    if abs(proj[2]) < 1e-8:
        return None
    px = proj[0] / proj[2]
    py = proj[1] / proj[2]
    if 0 <= px <= 1920 and 0 <= py <= 1080:
        return (px, py)
    return None


def render_clip(
    clip_id: str,
    split: str,
    dets: pd.DataFrame,
    balls: pd.DataFrame,
    H_lookup: dict,
    out_path: Path,
) -> int:
    """Render one clip with PC overlay."""
    clip_path = GSR_ROOT / split / clip_id
    if not clip_path.is_dir():
        print(f"  [skip] {clip_id}: clip dir not found")
        return 0

    frame_indices = sorted(dets["frame_idx"].unique())
    if not frame_indices:
        return 0

    # Read first frame for dimensions
    first_path = clip_path / "img1" / f"{frame_indices[0]:06d}.jpg"
    if not first_path.is_file():
        print(f"  [skip] {clip_id}: frame not found")
        return 0
    sample = cv2.imread(str(first_path))
    if sample is None:
        return 0
    img_h, img_w = sample.shape[:2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (img_w, img_h))

    written = 0
    for fi in frame_indices:
        img_path = clip_path / "img1" / f"{fi:06d}.jpg"
        if not img_path.is_file():
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        # Get homography for this frame
        H = H_lookup.get((split, clip_id, fi))
        if H is None:
            writer.write(frame)
            written += 1
            continue

        # Get detections and ball for this frame
        f_dets = dets[dets["frame_idx"] == fi]
        # ball_positions.parquet has one row per clip with set-piece position
        ball_row = balls[balls["clip_id"] == clip_id]
        if not ball_row.empty:
            x_col = "ball_x_m" if "ball_x_m" in ball_row.columns else "x_pitch"
            y_col = "ball_y_m" if "ball_y_m" in ball_row.columns else "y_pitch"
            ball_x = float(ball_row[x_col].iloc[0]) if ball_row[x_col].notna().any() else None
            ball_y = float(ball_row[y_col].iloc[0]) if ball_x is not None else None
        else:
            ball_x = None
            ball_y = None

        # Compute pitch control
        if not f_dets.empty and ball_x is not None:
            players_xy = f_dets[["x_m", "y_m"]].to_numpy()
            teams = f_dets["team_kmeans"].to_numpy()
            att_xy, def_xy, _ = split_attack_defend(players_xy, teams, (ball_x, ball_y))

            if len(att_xy) > 0 and len(def_xy) > 0:
                pc = pitch_control_surface(att_xy, def_xy, (ball_x, ball_y))

                # Warp PC to image and overlay
                pc_warped = warp_pc_to_image(pc, H, img_w, img_h)

                # Create intensity-based mask: stronger overlay where PC deviates from 0.5
                # This keeps contested areas (near 0.5) nearly transparent
                mask = pc_warped.astype(np.float32).sum(axis=2) / 255.0
                mask = np.clip(mask, 0, 1)
                mask_3ch = np.stack([mask] * 3, axis=2)

                # Alpha blend with intensity-weighted transparency
                frame_float = frame.astype(np.float32)
                pc_float = pc_warped.astype(np.float32)
                blended = frame_float * (1 - ALPHA * mask_3ch) + pc_float * (ALPHA * mask_3ch)
                frame = blended.astype(np.uint8)

                # Metrics
                pc_in_box = compute_pc_in_box(pc, ball_x)
                # PC at ball
                nx, ny = GRID_NX, GRID_NY
                ix = int(np.clip(round((ball_x / PITCH_LENGTH_M) * (nx - 1)), 0, nx - 1))
                iy = int(np.clip(round((ball_y / PITCH_WIDTH_M) * (ny - 1)), 0, ny - 1))
                pc_at_ball = float(pc[iy, ix])

                draw_metric_bar(frame, pc_in_box, pc_at_ball)

        # Draw players and ball
        ball_px = project_ball_to_image(ball_x, ball_y, H) if ball_x is not None else None
        draw_players_and_ball(frame, f_dets, ball_px)

        # Frame label
        action = f_dets["action_class"].iloc[0] if not f_dets.empty else ""
        cv2.putText(
            frame,
            f"{clip_id} | {action} | frame {fi}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        written += 1

    writer.release()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Render pitch control overlay on broadcast frames")
    parser.add_argument("--clip", default=None, help="Single clip ID, e.g. SNGS-066")
    args = parser.parse_args()

    print("Loading data...")
    dets = pd.read_parquet(OUTPUTS_DIR / "detections_soccana_tvcalib.parquet")
    balls = pd.read_parquet(OUTPUTS_DIR / "ball_positions.parquet")
    H_lookup = load_homography_lookup()
    print(f"  detections: {len(dets)} rows, {dets['clip_id'].nunique()} clips")
    print(f"  homographies: {len(H_lookup)} entries")

    if args.clip:
        clip_ids = [args.clip]
    else:
        clip_ids = sorted(dets["clip_id"].unique())

    print(f"Rendering {len(clip_ids)} clips...")
    for clip_id in clip_ids:
        clip_dets = dets[dets["clip_id"] == clip_id]
        if clip_dets.empty:
            continue
        split = clip_dets["split"].iloc[0]
        action = clip_dets["action_class"].iloc[0]
        out_path = FIGURES_DIR / f"{clip_id}_pc_overlay.mp4"
        n = render_clip(clip_id, split, clip_dets, balls, H_lookup, out_path)
        if n:
            print(f"  {clip_id} ({action}): {n} frames → {out_path.relative_to(PROJECT_ROOT)}")

    print("Done.")


if __name__ == "__main__":
    main()
