"""
For each clip:
  Phase 1: Read all frames (1–250), collect player detections, ball detections, HSV features
  Phase 2: Global team assignment (fit KMeans, mode consensus)
  Phase 3: Build detection rows (frames 1–31 only for PC)
  Phase 4: Ball position computation (project, interpolate, set-piece position)

Outputs:
  - outputs/detections_soccana_tvcalib.parquet
  - outputs/ball_positions.parquet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

# PyTorch 2.6 changed weights_only default to True; ultralytics .pt weights
# contain arbitrary globals and require weights_only=False.
_orig_torch_load = torch.load
torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})

from ultralytics import YOLO  # noqa: E402

# Ensure sibling scripts are importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _pipeline_core import (  # noqa: E402
    DEVICE,
    FITTING_WINDOW,
    FRAME_WINDOW,
    assign_teams_global_consensus,
    build_detection_rows,
    compute_setpiece_ball_position,
    detect_ball_frame,
    discover_setpiece_clips,
    filter_pitch_bounds,
    fit_global_kmeans,
    interpolate_ball_gaps,
    jersey_hsv,
    load_frame,
    load_tvcalib_lookup,
    project_points,
    reset_tracker,
    set_deterministic,
    track_frame,
    verify_soccernet_data,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

SOCCANA_REPO = "Adit-jain/soccana"
SOCCANA_WEIGHTS_PATH_IN_REPO = "Model/weights/best.pt"
# Pin the exact HF commit so weights are reproducible across runs/machines.
# main @ 2025-08-30 (repo is stable; last change was a thumbnail upload).
SOCCANA_REVISION = "305936007fe7d19ea528d73d08ccd7e70d088adf"
PLAYER_CLASS = 0
REFEREE_CLASS = 2


def run_optimized_clip(
    clip: pd.Series,
    yolo_player,
    yolo_ball,
    H_lookup: dict,
) -> tuple[list[dict], pd.DataFrame, str | None]:
    """
    Phases:
      1. Read all frames (1–250), run player detection + ball detection + HSV extraction.
      2. Global team assignment: fit KMeans on track-mean HSV, mode consensus.
      3. Build detection rows for frames 1–31 only (PC computation window).
      4. Ball position computation: project, interpolate, compute set-piece position.

    Args:
        clip: Series with clip_path, split, clip_id, action_class.
        yolo_player: YOLO model for player/referee detection (classes=[0,2]).
        yolo_ball: YOLO model for ball detection (classes=[1]).
        H_lookup: {(split, clip_id, frame_idx): 3x3 homography matrix}.

    Returns:
        (detection_rows, ball_position_df, skip_reason_or_None)
        detection_rows: list of dicts for detections_soccana_tvcalib.parquet
        ball_position_df: DataFrame with clip ball position info
        skip_reason: None if successful, else a string describing why clip was skipped
    """
    clip_path = Path(clip["clip_path"])
    try:
        with open(clip_path / "Labels-GameState.json") as f:
            labels_json = json.load(f)
    except Exception as e:
        return [], pd.DataFrame(), f"labels: {e}"

    n_frames = len(labels_json["images"])

    # PC computation window: frames 1–31 (centre at 16, ±FRAME_WINDOW)
    pc_centre = min(FRAME_WINDOW + 1, n_frames)
    pc_lo = max(1, pc_centre - FRAME_WINDOW)
    pc_hi = min(n_frames, pc_centre + FRAME_WINDOW)

    # Fitting window: frames 1–250 (or fewer if clip is shorter)
    fit_hi = min(n_frames, FITTING_WINDOW)

    # Reset both trackers for this clip
    reset_tracker(yolo_player)
    reset_tracker(yolo_ball)

    frame_to_image_id = {
        int(img["file_name"].replace(".jpg", "").lstrip("0") or "0"): img.get("image_id")
        for img in labels_json["images"]
        if img.get("file_name", "").endswith(".jpg")
    }

    # --- Phase 1: Single video pass over frames 1–fit_hi ---
    track_hsv_samples: dict[int, list[np.ndarray]] = {}
    pc_frame_detections: list[dict] = []
    ball_raw_detections: list[dict] = []
    n_homog_fail = 0

    for frame_idx in range(1, fit_hi + 1):
        image_id = frame_to_image_id.get(frame_idx)
        if image_id is None:
            continue

        # Read each SoccerNet video frame exactly once
        frame = load_frame(clip_path, frame_idx)
        if frame is None:
            continue

        H = H_lookup.get((clip["split"], clip["clip_id"], frame_idx))

        # Player/referee detection (keeps ByteTrack state consistent)
        dets, ref_dets = track_frame(yolo_player, frame, PLAYER_CLASS, REFEREE_CLASS)

        # Ball detection (separate ByteTrack state)
        ball_dets = detect_ball_frame(yolo_ball, frame)

        if H is None:
            n_homog_fail += 1
        else:
            # Extract HSV for player detections
            if len(dets) > 0:
                hsv_batch = np.array([jersey_hsv(frame, dets[k, :4]) for k in range(len(dets))])
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
                pc_frame_detections.append(
                    {
                        "frame_idx": frame_idx,
                        "image_id": image_id,
                        "H": H,
                        "dets": dets,
                        "hsv_batch": hsv_batch,
                        "ref_dets": ref_dets,
                    }
                )

            # Store ball detections for all frames with valid H
            if len(ball_dets) > 0:
                # Take highest-confidence ball detection
                best_idx = ball_dets[:, 4].argmax()
                ball_centre = ball_dets[best_idx, :2].reshape(1, 2)
                pitch_xy = project_points(H, ball_centre)
                # Filter pitch bounds
                valid = filter_pitch_bounds(pitch_xy)
                if len(valid) > 0:
                    ball_raw_detections.append(
                        {
                            "frame_idx": frame_idx,
                            "x_pitch": float(valid[0, 0]),
                            "y_pitch": float(valid[0, 1]),
                        }
                    )

    if not pc_frame_detections:
        return [], pd.DataFrame(), f"no valid PC frames (homog_fail={n_homog_fail})"

    # --- Phase 2: Global team assignment ---
    _centroids, label_map = fit_global_kmeans(track_hsv_samples)

    # Build per-frame label dict for consensus
    per_frame_labels: dict[int, dict[int, int]] = {}
    for fd in pc_frame_detections:
        fi = fd["frame_idx"]
        frame_map: dict[int, int] = {}
        frame_dets = fd["dets"]
        for k in range(len(frame_dets)):
            tid = int(frame_dets[k, 5])
            if tid in label_map:
                frame_map[tid] = label_map[tid]
        if frame_map:
            per_frame_labels[fi] = frame_map

    # Assign teams via global consensus
    if per_frame_labels:
        consensus_map = assign_teams_global_consensus(per_frame_labels)
    else:
        consensus_map = label_map  # fallback

    # --- Phase 3: Build detection rows for PC window only ---
    detection_rows = build_detection_rows(clip, pc_frame_detections, consensus_map)

    # --- Phase 4: Ball position computation ---
    ball_position_df = pd.DataFrame()
    if ball_raw_detections:
        ball_df = pd.DataFrame(ball_raw_detections)
        # Interpolate gaps <= 5 frames
        ball_df = interpolate_ball_gaps(ball_df, max_gap=5)
        try:
            sp_x, sp_y = compute_setpiece_ball_position(ball_df)
            ball_position_df = pd.DataFrame(
                [
                    {
                        "split": clip["split"],
                        "clip_id": clip["clip_id"],
                        "action_class": clip["action_class"],
                        "x_pitch": sp_x,
                        "y_pitch": sp_y,
                        "n_raw_detections": len(ball_raw_detections),
                        "n_interpolated": (
                            int(ball_df["interpolated"].sum()) if "interpolated" in ball_df.columns else 0
                        ),
                    }
                ]
            )
        except ValueError:
            # No valid ball detections after filtering
            pass

    return detection_rows, ball_position_df, None


def main() -> None:
    """Run the optimized pipeline on all 33 set-piece clips."""
    # --- Deterministic run (seed + deterministic kernels) ---
    set_deterministic()

    # --- SoccerNet GSR data verification ---
    gsr_root = verify_soccernet_data()
    print(f"SoccerNet GSR data found: {gsr_root}")

    # --- Load homographies ---
    H_lookup = load_tvcalib_lookup(OUTPUTS_DIR)
    print(f"TVCalib H entries: {len(H_lookup)}")

    # --- Discover 33-clip cohort ---
    clips = discover_setpiece_clips(gsr_root)
    print(f"Set-piece clips: {len(clips)}")

    # --- Initialize YOLO models ---
    weights_path = hf_hub_download(
        repo_id=SOCCANA_REPO,
        filename=SOCCANA_WEIGHTS_PATH_IN_REPO,
        revision=SOCCANA_REVISION,
    )
    yolo_player = YOLO(weights_path)
    yolo_ball = YOLO(weights_path)
    print(f"Soccana weights: {weights_path}  device: {DEVICE}")
    print("  yolo_player: classes=[0,2], conf=0.25, imgsz=1280, agnostic_nms (no TTA)")
    print("  yolo_ball:   classes=[1], conf=0.15, imgsz=1280, ByteTrack")

    # --- Process clips ---
    all_detection_rows: list[dict] = []
    all_ball_positions: list[pd.DataFrame] = []
    skipped: list[tuple[str, str]] = []

    for i, (_, clip) in enumerate(clips.iterrows()):
        det_rows, ball_df, reason = run_optimized_clip(clip, yolo_player, yolo_ball, H_lookup)
        if reason is not None:
            skipped.append((clip["clip_id"], reason))
            print(f"  [{i+1}/{len(clips)}] {clip['clip_id']} — SKIPPED: {reason}")
        else:
            all_detection_rows.extend(det_rows)
            if not ball_df.empty:
                all_ball_positions.append(ball_df)
            print(
                f"  [{i+1}/{len(clips)}] {clip['clip_id']} — "
                f"{len(det_rows)} det rows, ball={'yes' if not ball_df.empty else 'no'}"
            )

    # --- Save outputs ---
    # Detections
    det_df = pd.DataFrame(all_detection_rows)
    det_out = OUTPUTS_DIR / "detections_soccana_tvcalib.parquet"
    det_df.to_parquet(det_out, engine="pyarrow", index=False)
    print(f"\nDetections saved: {det_out}")
    print(f"  Rows: {len(det_df)}  |  Clips: {det_df['clip_id'].nunique() if len(det_df) > 0 else 0}")

    # Ball positions
    if all_ball_positions:
        ball_df = pd.concat(all_ball_positions, ignore_index=True)
    else:
        ball_df = pd.DataFrame(
            columns=["split", "clip_id", "action_class", "x_pitch", "y_pitch", "n_raw_detections", "n_interpolated"]
        )
    ball_out = OUTPUTS_DIR / "ball_positions.parquet"
    ball_df.to_parquet(ball_out, engine="pyarrow", index=False)
    print(f"Ball positions saved: {ball_out}")
    print(f"  Clips with ball: {len(ball_df)}")

    # --- Summary ---
    print(f"\nDone. Clips processed: {len(clips)}  |  Skipped: {len(skipped)}")
    if skipped:
        print("Skipped clips:")
        for clip_id, reason in skipped:
            print(f"  {clip_id}: {reason}")


if __name__ == "__main__":
    main()
