"""Shared helpers for the pipeline scripts.

Used by run_soccana_tvcalib.py, run_pc_soccana_tvcalib.py, run_pc_gt_full.py, etc.
Do not import directly from notebooks.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
FRAME_WINDOW = 15
FITTING_WINDOW = 250

SPLITS = ["train", "valid", "test", "challenge"]
TARGET_ACTIONS = {"Corner", "Direct free-kick"}
YOLO_CONF = 0.25
DEVICE = os.getenv("TORCH_DEVICE", "mps")

T_CENTRED_TO_TOPLEFT = np.array([
    [1.0, 0.0, PITCH_LENGTH_M / 2],
    [0.0, 1.0, PITCH_WIDTH_M / 2],
    [0.0, 0.0, 1.0],
])


def verify_ssd_mount(env_path: str = ".env") -> Path:
    """Check SSD is mounted, return GSR root path or exit with a clear error.

    Loads SOCCERNET_LOCAL_DIR from the .env file (or environment) and verifies
    the path exists as a directory. Exits the process with a non-zero status
    and descriptive error message if the SSD is not mounted.

    Args:
        env_path: Path to the .env file (relative to cwd or absolute).

    Returns:
        Path to the gamestate-2024 directory on the SSD.
    """
    from dotenv import load_dotenv

    load_dotenv(env_path)
    ssd_base = os.environ.get(
        "SOCCERNET_LOCAL_DIR", "/Volumes/MPH-ExternalStorage/soccernet-gsr"
    )
    ssd_path = Path(ssd_base) / "gamestate-2024"
    if not ssd_path.is_dir():
        sys.exit(f"ERROR: SSD not mounted at {ssd_path}. Mount the drive and retry.")
    return ssd_path


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
    x1 = int(x + 0.25 * w)
    x2 = int(x + 0.75 * w)
    y1 = int(y + 0.15 * h)
    y2 = int(y + 0.45 * h)
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


def fit_global_kmeans(
    track_hsv_samples: dict[int, list[np.ndarray]],
) -> tuple[np.ndarray, dict]:
    """Fit KMeans(k=3) on track-mean HSV vectors from all fitting frames.

    Args:
        track_hsv_samples: {track_id: [hsv_vector_frame1, hsv_vector_frame2, ...]}
            Each hsv_vector is the output of jersey_hsv() for that track in that frame.

    Returns:
        centroids: np.ndarray of shape (3, D) — the 3 cluster centroids
        label_map: dict mapping {track_id: cluster_label} based on each track's mean HSV
    """
    # Compute mean HSV vector per track, skipping NaN samples
    track_ids: list[int] = []
    track_means: list[np.ndarray] = []
    for tid, samples in track_hsv_samples.items():
        if not samples:
            continue
        stacked = np.array(samples)
        # Filter out NaN rows (invalid jersey_hsv results)
        valid_mask = ~np.isnan(stacked).any(axis=1)
        if not valid_mask.any():
            continue
        mean_vec = stacked[valid_mask].mean(axis=0)
        track_ids.append(tid)
        track_means.append(mean_vec)

    if len(track_means) < 3:
        # Not enough tracks for k=3; return empty centroids and map
        n = len(track_means)
        if n == 0:
            return np.empty((0, 3), dtype=np.float64), {}
        # Fall back to k=n clusters
        X = np.array(track_means)
        km = KMeans(n_clusters=n, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        label_map = {tid: int(lbl) for tid, lbl in zip(track_ids, labels)}
        return km.cluster_centers_, label_map

    X = np.array(track_means)
    km = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    label_map = {tid: int(lbl) for tid, lbl in zip(track_ids, labels)}
    return km.cluster_centers_, label_map


def assign_teams_global_consensus(
    per_frame_labels: dict[int, dict[int, int]],
) -> dict[int, int]:
    """Compute mode label per track_id across all frames (cross-frame consensus).

    Args:
        per_frame_labels: {frame_idx: {track_id: cluster_label}}
            The cluster labels assigned to each track in each frame by the global
            KMeans model.

    Returns:
        consensus: {track_id: team_label}
            The most frequently occurring label for each track across all frames.
            Ties are broken by picking the smallest label (deterministic).
    """
    # Collect all labels per track across frames
    track_labels: dict[int, list[int]] = {}
    for _frame_idx, frame_map in per_frame_labels.items():
        for track_id, label in frame_map.items():
            track_labels.setdefault(track_id, []).append(label)

    # Compute mode per track; ties broken by smallest label
    consensus: dict[int, int] = {}
    for track_id, labels in track_labels.items():
        counts = Counter(labels)
        max_count = max(counts.values())
        # Among labels with max count, pick the smallest (deterministic tie-break)
        mode_label = min(lbl for lbl, cnt in counts.items() if cnt == max_count)
        consensus[track_id] = mode_label

    return consensus


def track_frame(
    yolo, image_bgr: np.ndarray, player_class: int = 0, referee_class: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Single tracking pass for players + referees.

    One yolo.track() call per frame keeps ByteTrack state intact.
    Returns:
        player_dets  — Nx6 [x, y, w, h, conf, track_id]
        referee_dets — Mx5 [x, y, w, h, conf]
    """
    res = yolo.track(
        source=image_bgr,
        conf=YOLO_CONF,
        classes=[player_class, referee_class],
        tracker="bytetrack.yaml",
        persist=True,
        device=DEVICE,
        verbose=False,
        augment=True,       # TTA - Test-Time Augmentation
        agnostic_nms=True,  # Class-agnostic NMS
    )[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32), np.zeros((0, 5), dtype=np.float32)

    xyxy = res.boxes.xyxy.cpu().numpy()
    conf = res.boxes.conf.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    ids = (res.boxes.id.cpu().numpy().astype(int)
           if res.boxes.id is not None
           else np.full(len(conf), -1, dtype=int))
    x = xyxy[:, 0]
    y = xyxy[:, 1]
    w = xyxy[:, 2] - xyxy[:, 0]
    h = xyxy[:, 3] - xyxy[:, 1]
    all_dets = np.column_stack([x, y, w, h, conf, ids]).astype(np.float32)

    player_dets = all_dets[cls == player_class]
    referee_dets = all_dets[cls == referee_class, :5]
    return player_dets, referee_dets


def detect_ball_frame(yolo_ball, image_bgr: np.ndarray) -> np.ndarray:
    """Detect ball using Soccana class=1 at conf=0.15 with ByteTrack.

    Args:
        yolo_ball: YOLO model instance configured for ball detection.
        image_bgr: BGR image frame as numpy array.

    Returns:
        np.ndarray of shape (N, 5): [x_center, y_center, w, h, conf]
        or empty (0, 5) array if no detections.
    """
    res = yolo_ball.track(
        source=image_bgr,
        conf=0.15,
        classes=[1],
        tracker="bytetrack.yaml",
        persist=True,
        device=DEVICE,
        verbose=False,
    )[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    xyxy = res.boxes.xyxy.cpu().numpy()
    conf = res.boxes.conf.cpu().numpy()

    x_center = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    y_center = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
    w = xyxy[:, 2] - xyxy[:, 0]
    h = xyxy[:, 3] - xyxy[:, 1]

    return np.column_stack([x_center, y_center, w, h, conf]).astype(np.float32)


def interpolate_ball_gaps(ball_df: pd.DataFrame, max_gap: int = 5) -> pd.DataFrame:
    """Linear interpolation of ball position for gaps <= max_gap consecutive frames.

    Args:
        ball_df: DataFrame with columns: frame_idx, x_pitch, y_pitch.
            Rows represent frames where ball was detected. Missing frames = gaps.
        max_gap: Maximum gap size (in frames) to interpolate. Gaps > max_gap
            are left unfilled.

    Returns:
        DataFrame with same columns plus interpolated rows for gaps <= max_gap.
        An 'interpolated' boolean column marks which rows were filled.
    """
    if ball_df.empty:
        return ball_df.assign(interpolated=pd.Series(dtype=bool))

    df = ball_df.sort_values("frame_idx").reset_index(drop=True)
    df["interpolated"] = False

    interpolated_rows: list[dict] = []

    for i in range(len(df) - 1):
        frame_a = int(df.loc[i, "frame_idx"])
        frame_b = int(df.loc[i + 1, "frame_idx"])
        gap_size = frame_b - frame_a - 1

        if gap_size < 1 or gap_size > max_gap:
            continue

        x_a, y_a = df.loc[i, "x_pitch"], df.loc[i, "y_pitch"]
        x_b, y_b = df.loc[i + 1, "x_pitch"], df.loc[i + 1, "y_pitch"]

        for j in range(1, gap_size + 1):
            t = j / (gap_size + 1)
            interpolated_rows.append(
                {
                    "frame_idx": frame_a + j,
                    "x_pitch": x_a + t * (x_b - x_a),
                    "y_pitch": y_a + t * (y_b - y_a),
                    "interpolated": True,
                }
            )

    if interpolated_rows:
        interp_df = pd.DataFrame(interpolated_rows)
        df = pd.concat([df, interp_df], ignore_index=True)

    return df.sort_values("frame_idx").reset_index(drop=True)


def compute_setpiece_ball_position(
    ball_detections: pd.DataFrame,
) -> tuple[float, float]:
    """Compute set-piece ball position with frame-1 priority logic.

    Args:
        ball_detections: DataFrame with columns: frame_idx, x_pitch, y_pitch.
            Ball positions projected to pitch coordinates (after interpolation).
            Only rows within pitch bounds should be passed.

    Returns:
        (x_pitch, y_pitch) tuple representing the set-piece ball position.

    Logic:
        1. If frame_idx == 1 has a valid detection within pitch bounds, use it
           (resting ball at set-piece start).
        2. Otherwise, use the median of valid detections in frames 1–5.
        3. If no valid detections in frames 1–5, use median of all valid detections.

    Raises:
        ValueError: If ball_detections is empty (no valid detections at all).
    """
    if ball_detections.empty:
        raise ValueError("No valid ball detections to compute set-piece position.")

    # Priority 1: frame_idx == 1
    frame1 = ball_detections[ball_detections["frame_idx"] == 1]
    if not frame1.empty:
        row = frame1.iloc[0]
        return (float(row["x_pitch"]), float(row["y_pitch"]))

    # Priority 2: median of frames 1–5
    frames_1_to_5 = ball_detections[ball_detections["frame_idx"].between(1, 5)]
    if not frames_1_to_5.empty:
        return (
            float(frames_1_to_5["x_pitch"].median()),
            float(frames_1_to_5["y_pitch"].median()),
        )

    # Priority 3: median of all valid detections
    return (
        float(ball_detections["x_pitch"].median()),
        float(ball_detections["y_pitch"].median()),
    )


def validate_ball_positions(
    autonomous_df: pd.DataFrame,
    gt_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare autonomous ball positions against GT positions.

    Merges autonomous and GT positions on clip_id, computes Euclidean distance
    error per clip, prints summary statistics, and flags clips with error > 5m.

    Args:
        autonomous_df: DataFrame with columns: clip_id, x_pitch, y_pitch
            (autonomous ball positions from the pipeline)
        gt_df: DataFrame with columns: clip_id, x_pitch, y_pitch
            (historical GT-derived positions from ball_positions.parquet)

    Returns:
        DataFrame with columns: clip_id, x_auto, y_auto, x_gt, y_gt,
        euclidean_error, flagged (True if error > 5m).
        Summary statistics are printed to stdout.
    """
    ERROR_THRESHOLD_M = 5.0

    # Merge on clip_id
    merged = autonomous_df.merge(
        gt_df,
        on="clip_id",
        suffixes=("_auto", "_gt"),
    )

    # Rename columns for clarity
    merged = merged.rename(
        columns={
            "x_pitch_auto": "x_auto",
            "y_pitch_auto": "y_auto",
            "x_pitch_gt": "x_gt",
            "y_pitch_gt": "y_gt",
        }
    )

    # Compute Euclidean distance error per clip
    merged["euclidean_error"] = np.sqrt(
        (merged["x_auto"] - merged["x_gt"]) ** 2
        + (merged["y_auto"] - merged["y_gt"]) ** 2
    )

    # Flag clips with error > threshold
    merged["flagged"] = merged["euclidean_error"] > ERROR_THRESHOLD_M

    # Print summary statistics
    n_clips = len(merged)
    n_flagged = int(merged["flagged"].sum())
    print("=" * 60)
    print("Ball Position Validation: Autonomous vs GT")
    print("=" * 60)
    print(f"  Clips compared:       {n_clips}")
    print(f"  Mean error (m):       {merged['euclidean_error'].mean():.3f}")
    print(f"  Median error (m):     {merged['euclidean_error'].median():.3f}")
    print(f"  Max error (m):        {merged['euclidean_error'].max():.3f}")
    print(f"  Clips > {ERROR_THRESHOLD_M}m error:   {n_flagged}")
    print("-" * 60)

    if n_flagged > 0:
        flagged_clips = merged.loc[merged["flagged"], ["clip_id", "euclidean_error"]]
        print("  Flagged clips for manual review:")
        for _, row in flagged_clips.iterrows():
            print(f"    {row['clip_id']}: {row['euclidean_error']:.3f} m")
        print("-" * 60)

    return merged[["clip_id", "x_auto", "y_auto", "x_gt", "y_gt", "euclidean_error", "flagged"]]


def project_points(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    if len(pts_xy) == 0:
        return pts_xy
    homo = np.column_stack([pts_xy, np.ones(len(pts_xy))])
    out = (H @ homo.T).T
    return out[:, :2] / out[:, 2:3]


def filter_pitch_bounds(coords: np.ndarray) -> np.ndarray:
    """Drop coordinates outside [0, 105] x [0, 68].

    Parameters
    ----------
    coords : np.ndarray
        Nx2 array of (x, y) pitch coordinates in metres.

    Returns
    -------
    np.ndarray
        Subset of *coords* where x ∈ [0, 105] and y ∈ [0, 68].
    """
    if len(coords) == 0:
        return coords
    mask = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] <= PITCH_LENGTH_M)
        & (coords[:, 1] >= 0)
        & (coords[:, 1] <= PITCH_WIDTH_M)
    )
    return coords[mask]


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
        H = fd["H"]
        fi = fd["frame_idx"]

        # --- players (tracked, participate in pitch control) ---
        dets = fd["dets"]
        hsv_batch = fd["hsv_batch"]
        if len(dets) > 0:
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
                    "is_referee": False,
                    "conf": float(dets[k, 4]),
                    "hsv_h": float(hsv_batch[k, 0]),
                    "hsv_s": float(hsv_batch[k, 1]),
                    "hsv_v": float(hsv_batch[k, 2]),
                    "x1_px": float(dets[k, 0]),
                    "y1_px": float(dets[k, 1]),
                    "x2_px": float(dets[k, 0] + dets[k, 2]),
                    "y2_px": float(dets[k, 1] + dets[k, 3]),
                })

        # --- referees (YOLO class 2, excluded from pitch control) ---
        ref_dets = fd.get("ref_dets")
        if ref_dets is not None and len(ref_dets) > 0:
            ref_feet = np.column_stack([
                ref_dets[:, 0] + ref_dets[:, 2] / 2,
                ref_dets[:, 1] + ref_dets[:, 3],
            ])
            ref_pitch_xy = project_points(H, ref_feet)
            for k in range(len(ref_dets)):
                x_m, y_m = float(ref_pitch_xy[k, 0]), float(ref_pitch_xy[k, 1])
                if not (0 <= x_m <= PITCH_LENGTH_M and 0 <= y_m <= PITCH_WIDTH_M):
                    continue
                rows.append({
                    "split": clip["split"], "clip_id": clip["clip_id"],
                    "action_class": clip["action_class"], "frame_idx": fi,
                    "track_id": -2,
                    "x_m": x_m, "y_m": y_m,
                    "team_kmeans": -1,
                    "is_referee": True,
                    "conf": float(ref_dets[k, 4]),
                    "hsv_h": float("nan"),
                    "hsv_s": float("nan"),
                    "hsv_v": float("nan"),
                    "x1_px": float(ref_dets[k, 0]),
                    "y1_px": float(ref_dets[k, 1]),
                    "x2_px": float(ref_dets[k, 0] + ref_dets[k, 2]),
                    "y2_px": float(ref_dets[k, 1] + ref_dets[k, 3]),
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
    # action_position in SoccerNet GSR is a global broadcast frame number (>>750),
    # not a clip-local index. Clips are numbered 1–750 with the set-piece at frame 1.
    centre = min(FRAME_WINDOW + 1, n_frames)
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
            track_frame(yolo, frame, player_class)  # keep ByteTrack warm
            continue

        dets, ref_dets = track_frame(yolo, frame, player_class)
        if len(dets) == 0 and len(ref_dets) == 0:
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
            "ref_dets": ref_dets,
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


def run_clip_global(
    clip: pd.Series,
    yolo,
    H_lookup: dict,
    player_class: int = 0,
) -> tuple[list[dict], str | None]:
    """Process one clip using global team assignment (250-frame fitting window).

    Workflow:
      1. Read frames 1–250 for HSV collection (fitting window).
      2. For each frame, run detection + tracking and extract jersey HSV per track.
      3. After all 250 frames: fit global KMeans on accumulated track HSV samples.
      4. Apply assign_teams_global_consensus() for stable team labels per track.
      5. Build detection rows ONLY for frames 1–31 (PC computation window).
      6. Apply the global consensus team labels to those detection rows.

    Args:
        clip: Series with clip_path, split, clip_id, action_class.
        yolo: YOLO model instance for player/referee detection.
        H_lookup: {(split, clip_id, frame_idx): 3x3 homography matrix}.
        player_class: YOLO class index for players (default 0).

    Returns:
        (detection_rows, skip_reason_or_None).
    """
    clip_path = Path(clip["clip_path"])
    try:
        with open(clip_path / "Labels-GameState.json") as f:
            labels_json = json.load(f)
    except Exception as e:
        return [], f"labels: {e}"

    n_frames = len(labels_json["images"])

    # PC computation window: frames 1–31 (centre at 16, ±FRAME_WINDOW)
    pc_centre = min(FRAME_WINDOW + 1, n_frames)
    pc_lo = max(1, pc_centre - FRAME_WINDOW)
    pc_hi = min(n_frames, pc_centre + FRAME_WINDOW)

    # Fitting window: frames 1–250 (or fewer if clip is shorter)
    fit_hi = min(n_frames, FITTING_WINDOW)

    reset_tracker(yolo)

    # --- Phase 1: Read all fitting frames, collect detections + HSV ---
    track_hsv_samples: dict[int, list[np.ndarray]] = {}
    # Store frame detections for the PC window (frames 1–31)
    pc_frame_detections: list[dict] = []
    n_homog_fail = 0

    for frame_idx in range(1, fit_hi + 1):
        image_id = image_id_for_frame(labels_json, frame_idx)
        if image_id is None:
            continue
        frame = load_frame(clip_path, frame_idx)
        if frame is None:
            continue

        H = H_lookup.get((clip["split"], clip["clip_id"], frame_idx))

        # Always run tracking to keep ByteTrack state consistent
        dets, ref_dets = track_frame(yolo, frame, player_class)

        if H is None:
            n_homog_fail += 1
        else:
            # Extract HSV for all player detections in this frame
            if len(dets) > 0:
                hsv_batch = np.array(
                    [jersey_hsv(frame, dets[k, :4]) for k in range(len(dets))]
                )
                for k in range(len(dets)):
                    tid = int(dets[k, 5])
                    if tid < 0:
                        continue
                    if not np.isnan(hsv_batch[k]).any():
                        track_hsv_samples.setdefault(tid, []).append(hsv_batch[k])
            else:
                hsv_batch = np.zeros((0, 3), dtype=np.float32)

            # Store frame data if within PC window
            if pc_lo <= frame_idx <= pc_hi:
                pc_frame_detections.append({
                    "frame_idx": frame_idx,
                    "image_id": image_id,
                    "H": H,
                    "dets": dets,
                    "hsv_batch": hsv_batch,
                    "ref_dets": ref_dets,
                })

    if not pc_frame_detections:
        return [], f"no valid PC frames (homog_fail={n_homog_fail})"

    # --- Phase 2: Fit global KMeans on accumulated HSV ---
    _centroids, label_map = fit_global_kmeans(track_hsv_samples)

    # --- Phase 3: Build per-frame label dict for consensus ---
    # Use the global KMeans label_map to assign each track a label per frame
    # where it appears. This feeds into consensus to get the mode label.
    per_frame_labels: dict[int, dict[int, int]] = {}
    for fd in pc_frame_detections:
        fi = fd["frame_idx"]
        frame_map: dict[int, int] = {}
        dets = fd["dets"]
        for k in range(len(dets)):
            tid = int(dets[k, 5])
            if tid in label_map:
                frame_map[tid] = label_map[tid]
        if frame_map:
            per_frame_labels[fi] = frame_map

    # Also include labels from fitting frames outside PC window for consensus
    # (the label_map already captures all tracks from the full fitting window)

    # --- Phase 4: Assign teams via global consensus ---
    if per_frame_labels:
        consensus_map = assign_teams_global_consensus(per_frame_labels)
    else:
        consensus_map = label_map  # fallback to direct label_map

    # --- Phase 5: Build detection rows for PC window only ---
    rows = build_detection_rows(clip, pc_frame_detections, consensus_map)
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
