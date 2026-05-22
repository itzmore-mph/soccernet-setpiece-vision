"""Ablation runner: re-extract player detections with Adit-jain/soccana (YOLOv11n,
football-finetuned) instead of YOLOv8x. Mirrors nb02 logic end-to-end so output
schema matches `outputs/detections_pipeline.parquet`. Writes
`outputs/detections_soccana.parquet`.

Requires SSD mounted at GSR_ROOT. Run from repo root:
    python scripts/run_soccana_ablation.py
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.cluster import KMeans
from ultralytics import YOLO

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

GSR_ROOT = Path(os.getenv("SOCCERNET_LOCAL_DIR", "/Volumes/MPH-ExternalStorage/soccernet-gsr")) / "gamestate-2024"
SPLITS = ["train", "valid", "test", "challenge"]
TARGET_ACTIONS = {"Corner", "Direct free-kick"}

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
FRAME_W = 1920
FRAME_H = 1080

# --- Ablation: swap detector ---
# Soccana = YOLOv11n football-finetuned (Adit-jain/soccana on HF Hub).
# Weights live at Model/weights/best.pt inside the repo. Cached under
# ~/.cache/huggingface/ on first call.
SOCCANA_REPO = "Adit-jain/soccana"
SOCCANA_WEIGHTS_PATH_IN_REPO = "Model/weights/best.pt"
PLAYER_CLASS = 0                     # soccana class 0 = Player (excludes ball, referee)
YOLO_CONF = 0.40
DEVICE = os.getenv("TORCH_DEVICE", "mps")

FRAME_WINDOW = 15

assert GSR_ROOT.exists(), f"SoccerNet GSR not mounted: {GSR_ROOT}"

# ---------- Pitch geometry / homography (unchanged from nb02) ----------
PB_TOP = 34.0 - 40.32 / 2
PB_BOT = 34.0 + 40.32 / 2
SIX_TOP = 34.0 - 18.32 / 2
SIX_BOT = 34.0 + 18.32 / 2
GOAL_TOP = 34.0 - 7.32 / 2
GOAL_BOT = 34.0 + 7.32 / 2
PB_DEPTH, SIX_DEPTH = 16.5, 5.5
L, W = PITCH_LENGTH_M, PITCH_WIDTH_M

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
CANDIDATE_LINES = sorted({n for pair in INTERSECTIONS_M for n in pair})


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
        return None
    H, _ = cv2.findHomography(np.array(src_pts, dtype=np.float64),
                              np.array(dst_pts, dtype=np.float64),
                              method=cv2.RANSAC, ransacReprojThreshold=15.0)
    return H


def project_points(H, pts_xy):
    if len(pts_xy) == 0:
        return pts_xy
    homo = np.column_stack([pts_xy, np.ones(len(pts_xy))])
    out = (H @ homo.T).T
    return out[:, :2] / out[:, 2:3]


# ---------- HSV team assignment (unchanged) ----------
def jersey_hsv(image_bgr, bbox_xywh):
    x, y, w, h = bbox_xywh
    x1 = int(x + 0.25 * w); x2 = int(x + 0.75 * w)
    y1 = int(y + 0.15 * h); y2 = int(y + 0.45 * h)
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(image_bgr.shape[1] - 1, x2)
    y2 = min(image_bgr.shape[0] - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return np.array([np.nan, np.nan, np.nan])
    crop = image_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return hsv.reshape(-1, 3).mean(axis=0)


def _is_ref_like(c):
    h, s, v = float(c[0]), float(c[1]), float(c[2])
    if 20 <= h <= 65 and s > 80:
        return True
    if v < 50:
        return True
    return False


def assign_teams_kmeans(hsv_features, k=3, drop_frac=0.15):
    valid = ~np.isnan(hsv_features).any(axis=1)
    labels = np.full(len(hsv_features), -1, dtype=int)
    n_valid = int(valid.sum())
    if n_valid < 2:
        return labels
    if n_valid < k:
        km = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels[valid] = km.fit_predict(hsv_features[valid])
        return labels
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    raw = km.fit_predict(hsv_features[valid])
    sizes = np.bincount(raw, minlength=k)
    centroids = km.cluster_centers_
    order = np.argsort(-sizes)
    smallest = order[-1]
    drop = (sizes[smallest] / n_valid < drop_frac) or _is_ref_like(centroids[smallest])
    if drop:
        keep = {order[0]: 0, order[1]: 1}
        remap = np.array([keep.get(c, -1) for c in raw])
    else:
        km2 = KMeans(n_clusters=2, n_init=10, random_state=42)
        remap = km2.fit_predict(hsv_features[valid])
    labels[valid] = remap
    return labels


# ---------- Detector ----------
weights_path = hf_hub_download(repo_id=SOCCANA_REPO, filename=SOCCANA_WEIGHTS_PATH_IN_REPO)
yolo = YOLO(weights_path)
print(f"YOLO weights: {SOCCANA_REPO}::{SOCCANA_WEIGHTS_PATH_IN_REPO}")
print(f"Local cache : {weights_path}")
print(f"Device      : {DEVICE}")


def track_players(model, frame):
    res = model.track(
        source=frame, conf=YOLO_CONF, classes=[PLAYER_CLASS],
        tracker="bytetrack.yaml", persist=True, device=DEVICE, verbose=False,
    )[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    xyxy = res.boxes.xyxy.cpu().numpy()
    conf = res.boxes.conf.cpu().numpy()
    ids = (res.boxes.id.cpu().numpy().astype(int)
           if res.boxes.id is not None
           else np.full(len(conf), -1, dtype=int))
    x = xyxy[:, 0]; y = xyxy[:, 1]
    w = xyxy[:, 2] - xyxy[:, 0]
    h = xyxy[:, 3] - xyxy[:, 1]
    return np.column_stack([x, y, w, h, conf, ids]).astype(np.float32)


# ---------- Clip discovery + GSR helpers ----------
def discover_clips():
    rows = []
    for split in SPLITS:
        sd = GSR_ROOT / split
        if not sd.is_dir():
            continue
        for cd in sorted(sd.glob("SNGS-*")):
            lp = cd / "Labels-GameState.json"
            if not lp.is_file():
                continue
            try:
                with open(lp) as f:
                    info = json.load(f)["info"]
            except Exception:
                continue
            if info.get("action_class") in TARGET_ACTIONS:
                rows.append({
                    "split": split, "clip_id": cd.name, "clip_path": str(cd),
                    "action_class": info["action_class"],
                    "action_position": int(info.get("action_position", 375)),
                })
    return pd.DataFrame(rows)


def load_frame(clip_path, frame_idx):
    fp = clip_path / "img1" / f"{frame_idx:06d}.jpg"
    return cv2.imread(str(fp)) if fp.is_file() else None


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


# ---------- Main ----------
def main():
    clips = discover_clips()
    print(f"Set-piece clips found: {len(clips)}")

    pipeline_rows, skipped = [], []

    for i, clip in clips.iterrows():
        clip_path = Path(clip["clip_path"])
        try:
            with open(clip_path / "Labels-GameState.json") as f:
                labels = json.load(f)
        except Exception as e:
            skipped.append((clip["clip_id"], -1, f"labels: {e}"))
            continue

        n_frames = len(labels["images"])
        centre = max(1, min(int(clip["action_position"]), n_frames))
        lo = max(1, centre - FRAME_WINDOW)
        hi = min(n_frames, centre + FRAME_WINDOW)

        if hasattr(yolo, "predictor") and yolo.predictor is not None:
            if hasattr(yolo.predictor, "trackers") and yolo.predictor.trackers:
                yolo.predictor.trackers[0].reset()

        frame_detections, track_hsv_samples = [], {}

        for frame_idx in range(lo, hi + 1):
            image_id = image_id_for_frame(labels, frame_idx)
            if image_id is None:
                continue
            frame = load_frame(clip_path, frame_idx)
            if frame is None:
                continue
            pitch_lines = find_pitch_ann(labels, image_id)
            H = homography_from_pitch_lines(pitch_lines) if pitch_lines else None
            if H is None:
                track_players(yolo, frame)
                continue
            dets = track_players(yolo, frame)
            if len(dets) == 0:
                continue
            hsv_batch = np.array([jersey_hsv(frame, dets[k, :4]) for k in range(len(dets))])
            for k in range(len(dets)):
                tid = int(dets[k, 5])
                if tid < 0:
                    continue
                if not np.isnan(hsv_batch[k]).any():
                    track_hsv_samples.setdefault(tid, []).append(hsv_batch[k])
            frame_detections.append({
                "frame_idx": frame_idx, "image_id": image_id,
                "H": H, "dets": dets, "hsv_batch": hsv_batch,
            })

        if not frame_detections:
            skipped.append((clip["clip_id"], centre, "all frames failed homography or no detections"))
            continue

        track_ids = sorted(track_hsv_samples.keys())
        if track_ids:
            track_mean_hsv = np.array([
                np.nanmean(track_hsv_samples[tid], axis=0) for tid in track_ids
            ])
            clip_team_labels = assign_teams_kmeans(track_mean_hsv, k=3, drop_frac=0.15)
            track_team_map = {tid: int(clip_team_labels[j]) for j, tid in enumerate(track_ids)}
        else:
            track_team_map = {}

        for fd in frame_detections:
            dets, H, hsv_batch, frame_idx = fd["dets"], fd["H"], fd["hsv_batch"], fd["frame_idx"]
            feet = np.column_stack([dets[:, 0] + dets[:, 2] / 2, dets[:, 1] + dets[:, 3]])
            pitch_xy = project_points(H, feet)
            for k in range(len(dets)):
                tid = int(dets[k, 5])
                team_lbl = track_team_map.get(tid, -1)
                if team_lbl < 0:
                    continue
                x_m, y_m = float(pitch_xy[k, 0]), float(pitch_xy[k, 1])
                if not (0 <= x_m <= PITCH_LENGTH_M and 0 <= y_m <= PITCH_WIDTH_M):
                    continue
                pipeline_rows.append({
                    "split": clip["split"], "clip_id": clip["clip_id"],
                    "action_class": clip["action_class"], "frame_idx": frame_idx,
                    "track_id": tid, "x_m": x_m, "y_m": y_m,
                    "team_kmeans": team_lbl, "conf": float(dets[k, 4]),
                    "hsv_h": float(hsv_batch[k, 0]),
                    "hsv_s": float(hsv_batch[k, 1]),
                    "hsv_v": float(hsv_batch[k, 2]),
                })

        if (i + 1) % 5 == 0:
            print(f"  clips {i+1}/{len(clips)}  |  rows: {len(pipeline_rows)}")

    df = pd.DataFrame(pipeline_rows)
    out = OUTPUTS_DIR / "detections_soccana.parquet"
    df.to_parquet(out, engine="pyarrow", index=False)
    print(f"\nDone. Rows: {len(df)}  |  skipped clips: {len(skipped)}  |  saved: {out}")
    for c, fr, w in skipped[:10]:
        print(f"  skipped {c} (centre {fr}): {w}")


if __name__ == "__main__":
    main()
