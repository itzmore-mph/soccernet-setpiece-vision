"""Phase 5 Soccana detector under TVCalib autonomous H.

Mirrors run_soccana_ablation.py but reads pre-computed TVCalib H from
outputs/homographies_tvcalib.parquet instead of GT pitch lines.

Reads:
    outputs/homographies_tvcalib.parquet
    SSD frames

Writes:
    outputs/detections_soccana_tvcalib.parquet
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
GSR_ROOT = Path(os.getenv("SOCCERNET_LOCAL_DIR", "/Volumes/MPH-ExternalStorage/soccernet-gsr")) / "gamestate-2024"
SPLITS = ["train", "valid", "test", "challenge"]
TARGET_ACTIONS = {"Corner", "Direct free-kick"}

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
FRAME_WINDOW = 15

SOCCANA_REPO = "Adit-jain/soccana"
SOCCANA_WEIGHTS_PATH_IN_REPO = "Model/weights/best.pt"
PLAYER_CLASS = 0
YOLO_CONF = 0.40
DEVICE = "mps"

T_CENTRED_TO_TOPLEFT = np.array([
    [1.0, 0.0, PITCH_LENGTH_M / 2],
    [0.0, 1.0, PITCH_WIDTH_M / 2],
    [0.0, 0.0, 1.0],
])


def load_tvcalib_lookup():
    df = pd.read_parquet(OUTPUTS_DIR / "homographies_tvcalib.parquet")
    out = {}
    for _, r in df.iterrows():
        H = np.array([[r["h00"], r["h01"], r["h02"]],
                      [r["h10"], r["h11"], r["h12"]],
                      [r["h20"], r["h21"], r["h22"]]])
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            continue
        out[(r["split"], r["clip_id"], int(r["frame_idx"]))] = T_CENTRED_TO_TOPLEFT @ H_inv
    return out


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
                info = json.load(open(lp))["info"]
            except Exception:
                continue
            if info.get("action_class") in TARGET_ACTIONS:
                rows.append({
                    "split": split, "clip_id": cd.name, "clip_path": str(cd),
                    "action_class": info["action_class"],
                    "action_position": int(info.get("action_position", 375)),
                })
    return pd.DataFrame(rows)


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


def project_points(H, pts_xy):
    if len(pts_xy) == 0:
        return pts_xy
    homo = np.column_stack([pts_xy, np.ones(len(pts_xy))])
    out = (H @ homo.T).T
    return out[:, :2] / out[:, 2:3]


def load_frame(clip_path, frame_idx):
    fp = Path(clip_path) / "img1" / f"{frame_idx:06d}.jpg"
    return cv2.imread(str(fp)) if fp.is_file() else None


def image_id_for_frame(labels, frame_idx):
    target = f"{frame_idx:06d}.jpg"
    for img in labels["images"]:
        if img.get("file_name") == target:
            return img.get("image_id")
    return None


def main():
    assert GSR_ROOT.exists(), f"SoccerNet GSR not mounted: {GSR_ROOT}"
    H_LOOKUP = load_tvcalib_lookup()
    print(f"TVCalib H entries: {len(H_LOOKUP)}")

    clips = discover_clips()
    print(f"set-piece clips: {len(clips)}")

    weights_path = hf_hub_download(repo_id=SOCCANA_REPO, filename=SOCCANA_WEIGHTS_PATH_IN_REPO)
    yolo = YOLO(weights_path)
    print(f"Soccana weights: {weights_path}")

    pipeline_rows, skipped = [], []

    for i, clip in clips.iterrows():
        clip_path = Path(clip["clip_path"])
        try:
            labels = json.load(open(clip_path / "Labels-GameState.json"))
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
        n_homog_fail = 0

        for frame_idx in range(lo, hi + 1):
            image_id = image_id_for_frame(labels, frame_idx)
            if image_id is None:
                continue
            frame = load_frame(clip_path, frame_idx)
            if frame is None:
                continue
            H = H_LOOKUP.get((clip["split"], clip["clip_id"], frame_idx))
            if H is None:
                n_homog_fail += 1
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
            skipped.append((clip["clip_id"], centre,
                            f"all frames missing H ({n_homog_fail}) or no detections"))
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
            dets, H, hsv_batch, fi = fd["dets"], fd["H"], fd["hsv_batch"], fd["frame_idx"]
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
                    "action_class": clip["action_class"], "frame_idx": fi,
                    "track_id": tid, "x_m": x_m, "y_m": y_m,
                    "team_kmeans": team_lbl, "conf": float(dets[k, 4]),
                    "hsv_h": float(hsv_batch[k, 0]),
                    "hsv_s": float(hsv_batch[k, 1]),
                    "hsv_v": float(hsv_batch[k, 2]),
                })

        if (i + 1) % 5 == 0:
            print(f"  clips {i+1}/{len(clips)}  |  rows: {len(pipeline_rows)}")

    df = pd.DataFrame(pipeline_rows)
    out = OUTPUTS_DIR / "detections_soccana_tvcalib.parquet"
    df.to_parquet(out, engine="pyarrow", index=False)
    print(f"\nDone. Rows: {len(df)}  |  clips skipped: {len(skipped)}  |  saved: {out}")
    print(f"clips with rows: {df['clip_id'].nunique()}")


if __name__ == "__main__":
    main()
