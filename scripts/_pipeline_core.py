"""Shared helpers for the pipeline scripts.

Used by run_soccana_tvcalib.py, run_pc_soccana_tvcalib.py, run_pc_gt_full.py, etc.
Do not import directly from notebooks.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
FRAME_WINDOW = 15

SPLITS = ["train", "valid", "test", "challenge"]
TARGET_ACTIONS = {"Corner", "Direct free-kick"}
YOLO_CONF = 0.40
DEVICE = os.getenv("TORCH_DEVICE", "mps")

T_CENTRED_TO_TOPLEFT = np.array([
    [1.0, 0.0, PITCH_LENGTH_M / 2],
    [0.0, 1.0, PITCH_WIDTH_M / 2],
    [0.0, 0.0, 1.0],
])


def discover_setpiece_clips(gsr_root: Path) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        split_dir = gsr_root / split
        if not split_dir.is_dir():
            continue
        for clip_dir in sorted(split_dir.glob("SNGS-*")):
            label_path = clip_dir / "Labels-GameState.json"
            if not label_path.is_file():
                continue
            try:
                with open(label_path) as f:
                    info = json.load(f)["info"]
            except Exception:
                continue
            if info.get("action_class") in TARGET_ACTIONS:
                rows.append({
                    "split": split,
                    "clip_id": clip_dir.name,
                    "clip_path": str(clip_dir),
                    "action_class": info["action_class"],
                    "action_position": int(info.get("action_position", 375)),
                })
    return pd.DataFrame(rows)


def load_tvcalib_lookup(outputs_dir: Path) -> dict[tuple[str, str, int], np.ndarray]:
    df = pd.read_parquet(outputs_dir / "homographies_tvcalib.parquet")
    out: dict = {}
    for _, r in df.iterrows():
        H_world_to_image = np.array([
            [r["h00"], r["h01"], r["h02"]],
            [r["h10"], r["h11"], r["h12"]],
            [r["h20"], r["h21"], r["h22"]],
        ])
        try:
            H_image_to_world = np.linalg.inv(H_world_to_image)
        except np.linalg.LinAlgError:
            continue
        out[(r["split"], r["clip_id"], int(r["frame_idx"]))] = T_CENTRED_TO_TOPLEFT @ H_image_to_world
    return out


def jersey_hsv(image_bgr: np.ndarray, bbox_xywh: np.ndarray) -> np.ndarray:
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


def _is_ref_like(c: np.ndarray) -> bool:
    h, s, v = float(c[0]), float(c[1]), float(c[2])
    if 20 <= h <= 65 and s > 80:
        return True
    if v < 50:
        return True
    return False


def assign_teams_kmeans(hsv_features: np.ndarray, k: int = 3, drop_frac: float = 0.15) -> np.ndarray:
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
    drop_smallest = (sizes[smallest] / n_valid < drop_frac) or _is_ref_like(centroids[smallest])
    if drop_smallest:
        keep = {order[0]: 0, order[1]: 1}
        remap = np.array([keep.get(c, -1) for c in raw])
    else:
        km2 = KMeans(n_clusters=2, n_init=10, random_state=42)
        remap = km2.fit_predict(hsv_features[valid])
    labels[valid] = remap
    return labels


def track_players(yolo, image_bgr: np.ndarray, player_class: int = 0) -> np.ndarray:
    res = yolo.track(
        source=image_bgr,
        conf=YOLO_CONF,
        classes=[player_class],
        tracker="bytetrack.yaml",
        persist=True,
        device=DEVICE,
        verbose=False,
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


def project_points(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    if len(pts_xy) == 0:
        return pts_xy
    homo = np.column_stack([pts_xy, np.ones(len(pts_xy))])
    out = (H @ homo.T).T
    return out[:, :2] / out[:, 2:3]


def load_frame(clip_path: Path, frame_idx: int) -> np.ndarray | None:
    fp = clip_path / "img1" / f"{frame_idx:06d}.jpg"
    if not fp.is_file():
        return None
    return cv2.imread(str(fp))


def image_id_for_frame(labels: dict, frame_idx: int) -> int | None:
    target = f"{frame_idx:06d}.jpg"
    for img in labels["images"]:
        if img.get("file_name") == target:
            return img.get("image_id")
    return None


def reset_tracker(yolo) -> None:
    if hasattr(yolo, "predictor") and yolo.predictor is not None:
        if hasattr(yolo.predictor, "trackers") and yolo.predictor.trackers:
            yolo.predictor.trackers[0].reset()


def build_detection_rows(
    clip: pd.Series,
    frame_detections: list[dict],
    track_team_map: dict[int, int],
) -> list[dict]:
    rows = []
    for fd in frame_detections:
        dets = fd["dets"]; H = fd["H"]; hsv_batch = fd["hsv_batch"]; fi = fd["frame_idx"]
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
            rows.append({
                "split": clip["split"], "clip_id": clip["clip_id"],
                "action_class": clip["action_class"], "frame_idx": fi,
                "track_id": tid,
                "x_m": x_m, "y_m": y_m,
                "team_kmeans": team_lbl,
                "conf": float(dets[k, 4]),
                "hsv_h": float(hsv_batch[k, 0]),
                "hsv_s": float(hsv_batch[k, 1]),
                "hsv_v": float(hsv_batch[k, 2]),
                "x1_px": float(dets[k, 0]),
                "y1_px": float(dets[k, 1]),
                "x2_px": float(dets[k, 0] + dets[k, 2]),
                "y2_px": float(dets[k, 1] + dets[k, 3]),
            })
    return rows


def run_clip(
    clip: pd.Series,
    yolo,
    H_lookup: dict,
    player_class: int = 0,
) -> tuple[list[dict], str | None]:
    """Process one clip: detect, accumulate HSV, assign teams, project to pitch.

    Returns (detection_rows, skip_reason_or_None).
    """
    clip_path = Path(clip["clip_path"])
    try:
        with open(clip_path / "Labels-GameState.json") as f:
            labels_json = json.load(f)
    except Exception as e:
        return [], f"labels: {e}"

    n_frames = len(labels_json["images"])
    centre = max(1, min(int(clip["action_position"]), n_frames))
    lo = max(1, centre - FRAME_WINDOW)
    hi = min(n_frames, centre + FRAME_WINDOW)

    reset_tracker(yolo)

    frame_detections: list[dict] = []
    track_hsv_samples: dict[int, list] = {}
    n_homog_fail = 0

    for frame_idx in range(lo, hi + 1):
        image_id = image_id_for_frame(labels_json, frame_idx)
        if image_id is None:
            continue
        frame = load_frame(clip_path, frame_idx)
        if frame is None:
            continue

        H = H_lookup.get((clip["split"], clip["clip_id"], frame_idx))
        if H is None:
            n_homog_fail += 1
            track_players(yolo, frame, player_class)
            continue

        dets = track_players(yolo, frame, player_class)
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
        return [], f"all frames missing H ({n_homog_fail}) or no detections"

    track_ids = sorted(track_hsv_samples.keys())
    if track_ids:
        track_mean_hsv = np.array([
            np.nanmean(track_hsv_samples[tid], axis=0) for tid in track_ids
        ])
        clip_team_labels = assign_teams_kmeans(track_mean_hsv, k=3, drop_frac=0.15)
        track_team_map = {tid: int(clip_team_labels[j]) for j, tid in enumerate(track_ids)}
    else:
        track_team_map = {}

    rows = build_detection_rows(clip, frame_detections, track_team_map)
    return rows, None

# ---------- Pitch Control (Laurie Shaw / Spearman model) ----------

GRID_NX = 60
GRID_NY = 40
MAX_SPEED = 5.0
REACTION_TIME = 0.7
SIGMA = 0.45
TIME_TO_INTERCEPT_SIGMOID_K = np.pi / (np.sqrt(3.0) * SIGMA)


def time_to_intercept(player_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import cdist
    dist = cdist(player_xy, target_xy)
    return REACTION_TIME + dist / MAX_SPEED


def pitch_control_surface(
    att_xy: np.ndarray, def_xy: np.ndarray, ball_xy: tuple, nx: int = GRID_NX, ny: int = GRID_NY
) -> np.ndarray:
    xs = np.linspace(0.0, PITCH_LENGTH_M, nx)
    ys = np.linspace(0.0, PITCH_WIDTH_M, ny)
    grid = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T
    if len(att_xy) == 0 or len(def_xy) == 0:
        return np.full((ny, nx), 0.5)
    tti_att = time_to_intercept(att_xy, grid).min(axis=0)
    tti_def = time_to_intercept(def_xy, grid).min(axis=0)
    delta = tti_att - tti_def
    exponent = np.clip(TIME_TO_INTERCEPT_SIGMOID_K * delta, -500, 500)
    return (1.0 / (1.0 + np.exp(exponent))).reshape(ny, nx)


def split_attack_defend(
    players_xy: np.ndarray, team_labels: np.ndarray, ball_xy: tuple
) -> tuple[np.ndarray, np.ndarray, object]:
    bx, by = ball_xy
    teams = [t for t in np.unique(team_labels)
             if t is not None and t != -1 and not (isinstance(t, float) and np.isnan(t))]
    if len(teams) < 2:
        return players_xy, np.zeros((0, 2)), teams[0] if teams else None
    min_d: dict = {}
    for t in teams:
        mask = team_labels == t
        if mask.sum() == 0:
            continue
        d = np.linalg.norm(players_xy[mask] - np.array([bx, by]), axis=1)
        min_d[t] = d.min()
    att_team = min(min_d, key=min_d.get)
    def_team = [t for t in min_d if t != att_team][0]
    return players_xy[team_labels == att_team], players_xy[team_labels == def_team], att_team


def summarise_surface(pc: np.ndarray, ball_xy: tuple) -> dict:
    bx, by = ball_xy
    ny, nx = pc.shape
    xs = np.linspace(0.0, PITCH_LENGTH_M, nx)
    ys = np.linspace(0.0, PITCH_WIDTH_M, ny)
    ix = int(np.clip(round((bx / PITCH_LENGTH_M) * (nx - 1)), 0, nx - 1))
    iy = int(np.clip(round((by / PITCH_WIDTH_M) * (ny - 1)), 0, ny - 1))
    pc_at_ball = float(pc[iy, ix])
    XX, YY = np.meshgrid(xs, ys)
    pen_left = (XX <= 16.5) & (YY >= 13.84) & (YY <= 54.16)
    pen_right = (XX >= PITCH_LENGTH_M - 16.5) & (YY >= 13.84) & (YY <= 54.16)
    if bx < PITCH_LENGTH_M / 2:
        att_third = XX <= PITCH_LENGTH_M / 3
        att_box = pen_left
    else:
        att_third = XX >= 2 * PITCH_LENGTH_M / 3
        att_box = pen_right
    return {
        "pc_mean": float(pc.mean()),
        "pc_at_ball": pc_at_ball,
        "pc_in_box": float(pc[att_box].mean()) if att_box.any() else np.nan,
        "pc_in_third": float(pc[att_third].mean()) if att_third.any() else np.nan,
        "pc_area_gt_0p5": float((pc > 0.5).mean()),
    }


def process_track(df: pd.DataFrame, track_name: str, team_col: str, balls: pd.DataFrame) -> pd.DataFrame:
    """Compute pitch control for each frame in a detection DataFrame."""
    out = []
    for (split, clip_id, frame_idx), g in df.groupby(["split", "clip_id", "frame_idx"]):
        ball_row = balls[(balls["split"] == split) & (balls["clip_id"] == clip_id) & (balls["frame_idx"] == frame_idx)]
        if ball_row.empty or ball_row["ball_x_m"].isna().any():
            continue
        ball_xy = (float(ball_row["ball_x_m"].iloc[0]), float(ball_row["ball_y_m"].iloc[0]))
        players_xy = g[["x_m", "y_m"]].to_numpy()
        teams = g[team_col].to_numpy()
        att_xy, def_xy, _ = split_attack_defend(players_xy, teams, ball_xy)
        if len(att_xy) == 0 or len(def_xy) == 0:
            continue
        pc = pitch_control_surface(att_xy, def_xy, ball_xy)
        m = summarise_surface(pc, ball_xy)
        action_class = g["action_class"].iloc[0] if "action_class" in g.columns else None
        out.append({
            "split": split, "clip_id": clip_id, "frame_idx": int(frame_idx),
            "track": track_name, "action_class": action_class,
            "n_attackers": int(len(att_xy)), "n_defenders": int(len(def_xy)),
            "ball_x_m": ball_xy[0], "ball_y_m": ball_xy[1],
            **m,
        })
    return pd.DataFrame(out)
