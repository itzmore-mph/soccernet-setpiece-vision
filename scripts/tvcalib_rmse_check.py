"""Phase 1 RMSE sanity: TVCalib H vs GT-derived H on 5 SNGS-066 frames.

For each frame:
  1. Project GT player `bbox_pitch` foot points -> image via both Hs.
  2. Compare against actual `bbox_image` foot points.
  3. Pixel RMSE per frame, per H source.

Gate: if TVCalib RMSE > 2x GT-H RMSE on average, abort full integration.

Outputs:
  outputs/tvcalib_phase1_rmse.parquet
  prints summary to stdout
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TVCALIB_ROOT = PROJECT_ROOT.parent / "tvcalib"
SSD_ROOT = Path(os.getenv("SOCCERNET_LOCAL_DIR", "/Volumes/MPH-ExternalStorage/soccernet-gsr")) / "gamestate-2024"
CLIP = "SNGS-066"
SPLIT = "train"
FRAMES = [735, 738, 742, 746, 750]
STAGE_DIR = Path("/tmp/tvcalib_p1")
OUT_DIR = Path("/tmp/tvcalib_p1_out")

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
FRAME_W = 1920
FRAME_H = 1080

# pitch-line geometry (mirrors nb02)
PB_DEPTH = 16.5
PB_TOP = (PITCH_WIDTH_M - 40.32) / 2
PB_BOT = PITCH_WIDTH_M - PB_TOP
SIX_DEPTH = 5.5
SIX_TOP = (PITCH_WIDTH_M - 18.32) / 2
SIX_BOT = PITCH_WIDTH_M - SIX_TOP
GOAL_TOP = (PITCH_WIDTH_M - 7.32) / 2
GOAL_BOT = PITCH_WIDTH_M - GOAL_TOP
L = PITCH_LENGTH_M
W = PITCH_WIDTH_M

INTERSECTIONS_M = {
    ("Side line top", "Side line left"): (0.0, 0.0),
    ("Side line top", "Side line right"): (L, 0.0),
    ("Side line bottom", "Side line left"): (0.0, W),
    ("Side line bottom", "Side line right"): (L, W),
    ("Side line top", "Middle line"): (L / 2, 0.0),
    ("Side line bottom", "Middle line"): (L / 2, W),
    ("Side line left", "Big rect. left top"): (0.0, PB_TOP),
    ("Side line left", "Big rect. left bottom"): (0.0, PB_BOT),
    ("Big rect. left top", "Big rect. left main"): (PB_DEPTH, PB_TOP),
    ("Big rect. left bottom", "Big rect. left main"): (PB_DEPTH, PB_BOT),
    ("Side line left", "Small rect. left top"): (0.0, SIX_TOP),
    ("Side line left", "Small rect. left bottom"): (0.0, SIX_BOT),
    ("Small rect. left top", "Small rect. left main"): (SIX_DEPTH, SIX_TOP),
    ("Small rect. left bottom", "Small rect. left main"): (SIX_DEPTH, SIX_BOT),
    ("Side line left", "Goal left post left"): (0.0, GOAL_TOP),
    ("Side line left", "Goal left post right"): (0.0, GOAL_BOT),
    ("Side line right", "Big rect. right top"): (L, PB_TOP),
    ("Side line right", "Big rect. right bottom"): (L, PB_BOT),
    ("Big rect. right top", "Big rect. right main"): (L - PB_DEPTH, PB_TOP),
    ("Big rect. right bottom", "Big rect. right main"): (L - PB_DEPTH, PB_BOT),
    ("Side line right", "Small rect. right top"): (L, SIX_TOP),
    ("Side line right", "Small rect. right bottom"): (L, SIX_BOT),
    ("Small rect. right top", "Small rect. right main"): (L - SIX_DEPTH, SIX_TOP),
    ("Small rect. right bottom", "Small rect. right main"): (L - SIX_DEPTH, SIX_BOT),
    ("Side line right", "Goal right post left"): (L, GOAL_TOP),
    ("Side line right", "Goal right post right"): (L, GOAL_BOT),
}
CANDIDATE_LINES = sorted({n for pair in INTERSECTIONS_M.keys() for n in pair})


def line_polyline_to_image(pts_norm):
    if not pts_norm or len(pts_norm) < 2:
        return None
    return np.array([[p["x"] * FRAME_W, p["y"] * FRAME_H] for p in pts_norm], dtype=np.float32)


def fit_line_2d(pts):
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    return np.array([x0, y0], dtype=np.float64), np.array([vx, vy], dtype=np.float64)


def line_intersection(p1, d1, p2, d2):
    A = np.column_stack([d1, -d2])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    t = np.linalg.solve(A, (p2 - p1))
    return p1 + t[0] * d1


def homography_from_pitch_lines(pitch_ann_lines):
    """Pixel -> pitch (top-left metres) homography."""
    fitted = {}
    for name in CANDIDATE_LINES:
        if name in pitch_ann_lines:
            pts = line_polyline_to_image(pitch_ann_lines[name])
            if pts is not None and len(pts) >= 2:
                fitted[name] = fit_line_2d(pts)
    src_pts, dst_pts = [], []
    for (a, b), pitch_xy in INTERSECTIONS_M.items():
        if a in fitted and b in fitted:
            inter = line_intersection(fitted[a][0], fitted[a][1], fitted[b][0], fitted[b][1])
            if inter is not None and -200 <= inter[0] <= FRAME_W + 200 and -200 <= inter[1] <= FRAME_H + 200:
                src_pts.append(inter)
                dst_pts.append(pitch_xy)
    if len(src_pts) < 4:
        return None, len(src_pts)
    H, _ = cv2.findHomography(
        np.array(src_pts, dtype=np.float64),
        np.array(dst_pts, dtype=np.float64),
        method=cv2.RANSAC,
        ransacReprojThreshold=15.0,
    )
    return H, len(src_pts)


def find_pitch_ann(labels, image_id):
    for a in labels["annotations"]:
        if a.get("image_id") == image_id and a.get("category_id") == 5:
            return a.get("lines", {})
    return None


def image_id_for_frame(labels, frame_idx):
    target = f"{frame_idx:06d}.jpg"
    for img in labels["images"]:
        if img.get("file_name") == target:
            return img.get("image_id")
    return None


def project_pitch_to_image_via_nb02(H_pix_to_pitch_topleft, pitch_xy_topleft):
    """Apply inv(H) to project pitch (top-left metres) to image pixels."""
    H_inv = np.linalg.inv(H_pix_to_pitch_topleft)
    pts = np.column_stack([pitch_xy_topleft, np.ones(len(pitch_xy_topleft))])
    out = (H_inv @ pts.T).T
    return out[:, :2] / out[:, 2:3]


def project_pitch_to_image_via_tvcalib(H_world_to_image, pitch_xy_centred):
    """TVCalib H is world (centred metres) -> image. Apply directly."""
    pts = np.column_stack([pitch_xy_centred, np.ones(len(pitch_xy_centred))])
    out = (H_world_to_image @ pts.T).T
    return out[:, :2] / out[:, 2:3]


def stage_frames():
    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)
    STAGE_DIR.mkdir(parents=True)
    for f in FRAMES:
        src = SSD_ROOT / SPLIT / CLIP / "img1" / f"{f:06d}.jpg"
        if not src.is_file():
            raise FileNotFoundError(src)
        shutil.copy(src, STAGE_DIR / f"{CLIP}_{f:06d}.jpg")
    print(f"staged {len(FRAMES)} frames into {STAGE_DIR}")


def run_tvcalib():
    cache = OUT_DIR / "calib.json"
    if cache.is_file():
        print(f"using cached {cache}")
        return json.loads(cache.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(TVCALIB_ROOT / ".venv/bin/python"),
        str(TVCALIB_ROOT / "run_inference.py"),
        "--images_path", str(STAGE_DIR),
        "--output_dir", str(OUT_DIR),
        "--image_width", "1920",
        "--image_height", "1080",
        "--optim_steps", "2000",
    ]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, cwd=TVCALIB_ROOT, check=True)
    return json.loads((OUT_DIR / "calib.json").read_text())


def main():
    stage_frames()
    tvcalib_results = run_tvcalib()
    print(f"tvcalib produced {len(tvcalib_results)} results")

    labels_path = SSD_ROOT / SPLIT / CLIP / "Labels-GameState.json"
    labels = json.loads(labels_path.read_text())

    rows = []
    for frame_idx in FRAMES:
        image_id = image_id_for_frame(labels, frame_idx)
        if image_id is None:
            print(f"[skip] frame {frame_idx}: no image_id")
            continue

        # GT player anns
        players = []
        for a in labels["annotations"]:
            if a.get("image_id") != image_id:
                continue
            if a.get("category_id") != 1:  # player only (not goalkeeper/referee/etc)
                continue
            bp = a.get("bbox_pitch")
            bi = a.get("bbox_image")
            if not bp or not bi:
                continue
            players.append({
                "pitch_x_centred": bp["x_bottom_middle"],
                "pitch_y_centred": bp["y_bottom_middle"],
                "img_x": bi["x_center"],
                "img_y": bi["y"] + bi["h"],  # foot = bottom of bbox
            })
        # Try category_id 2 fallback if no category 1 (some labels use 2 for player)
        if not players:
            for a in labels["annotations"]:
                if a.get("image_id") != image_id:
                    continue
                if a.get("category_id") not in (1, 2):
                    continue
                bp = a.get("bbox_pitch")
                bi = a.get("bbox_image")
                if not bp or not bi:
                    continue
                players.append({
                    "pitch_x_centred": bp["x_bottom_middle"],
                    "pitch_y_centred": bp["y_bottom_middle"],
                    "img_x": bi["x_center"],
                    "img_y": bi["y"] + bi["h"],
                })

        if not players:
            print(f"[skip] frame {frame_idx}: no players with bbox_pitch + bbox_image")
            continue

        pitch_centred = np.array([[p["pitch_x_centred"], p["pitch_y_centred"]] for p in players])
        pitch_topleft = pitch_centred + np.array([52.5, 34.0])
        gt_img = np.array([[p["img_x"], p["img_y"]] for p in players])

        # GT-derived H from pitch lines
        pitch_lines = find_pitch_ann(labels, image_id)
        if pitch_lines is None:
            print(f"[skip] frame {frame_idx}: no pitch lines annotation")
            continue
        H_gt, n_inter = homography_from_pitch_lines(pitch_lines)
        if H_gt is None:
            print(f"[skip] frame {frame_idx}: H_gt None (only {n_inter} intersections)")
            continue
        pred_gt = project_pitch_to_image_via_nb02(H_gt, pitch_topleft)
        rmse_gt = float(np.sqrt(((pred_gt - gt_img) ** 2).sum(axis=1).mean()))

        # TVCalib H
        tv_key = f"{CLIP}_{frame_idx:06d}.jpg"
        if tv_key not in tvcalib_results:
            print(f"[skip] frame {frame_idx}: tvcalib key {tv_key} missing")
            continue
        H_tv = np.array(tvcalib_results[tv_key]["H_world_to_image"])
        loss_ndc = tvcalib_results[tv_key]["loss_ndc_total"]
        pred_tv = project_pitch_to_image_via_tvcalib(H_tv, pitch_centred)
        rmse_tv = float(np.sqrt(((pred_tv - gt_img) ** 2).sum(axis=1).mean()))

        rows.append({
            "frame_idx": frame_idx,
            "n_players": len(players),
            "n_intersections": n_inter,
            "rmse_gt_lines": rmse_gt,
            "rmse_tvcalib": rmse_tv,
            "tvcalib_loss_ndc": loss_ndc,
        })
        print(f"frame {frame_idx}: n={len(players)} | GT-H RMSE={rmse_gt:8.2f}px | "
              f"TVCalib RMSE={rmse_tv:8.2f}px | loss_ndc={loss_ndc:.4f}")

    df = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "outputs" / "tvcalib_phase1_rmse.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nsaved: {out_path}")
    print("\nsummary:")
    print(df.to_string(index=False))
    print(f"\nmean GT-H RMSE: {df['rmse_gt_lines'].mean():.2f}px")
    print(f"mean TVCalib RMSE: {df['rmse_tvcalib'].mean():.2f}px")
    ratio = df["rmse_tvcalib"].mean() / df["rmse_gt_lines"].mean()
    print(f"ratio TVCalib / GT-H: {ratio:.2f}x")
    if ratio > 2.0:
        print("\nGATE FAIL: TVCalib > 2x GT-H. Abort full integration recommended.")
    else:
        print("\nGATE PASS: TVCalib within 2x of GT-H. Proceed to Phase 2.")


if __name__ == "__main__":
    main()
