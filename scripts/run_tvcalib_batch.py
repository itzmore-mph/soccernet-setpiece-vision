"""Batch TVCalib: compute autonomous homographies for all set-piece frames.

Discovers all SoccerNet GSR clips with action_class in {Corner, Direct free-kick},
stages the first ±15 frames (frames 1–31) of each clip into a temp dir, runs TVCalib
once over the lot, writes outputs/homographies_tvcalib.parquet keyed
(split, clip_id, frame_idx).

Requirements:
    - Local SoccerNet GSR dataset (SOCCERNET_LOCAL_DIR env var)
    - TVCalib repo at ../tvcalib/ with its own venv (.venv/) set up:
        cd ../tvcalib
        python3.11 -m venv .venv
        source .venv/bin/activate
        pip install torch==2.1.* torchvision kornia==0.8.2 pytorch-lightning==2.6.1
        pip install SoccerNet==0.1.62 opencv-python numpy
      Patches for PyTorch 2.x:
        - sncalib_dataset.py line 13: replace `from torch._six import string_classes`
          with `string_classes = (str, bytes)`
        - inference.py line 89: add `weights_only=False` to torch.load()
      Segmentation checkpoint: tvcalib/data/segment_localization/train_59.pt

Idempotent on stage (skips already-copied frames). TVCalib re-runs every call;
delete /tmp/tvcalib_batch_out/calib.json to force re-stage only.

Writes:
    outputs/homographies_tvcalib.parquet
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure sibling scripts are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _pipeline_core import verify_soccernet_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TVCALIB_ROOT = PROJECT_ROOT.parent / "tvcalib"
GSR_ROOT = Path(os.getenv("SOCCERNET_LOCAL_DIR", "data/soccernet-gsr")) / "gamestate-2024"
SPLITS = ["train", "valid", "test", "challenge"]
TARGET_ACTIONS = {"Corner", "Direct free-kick"}
FRAME_WINDOW = 15
STAGE_DIR = Path("/tmp/tvcalib_batch")
OUT_DIR = Path("/tmp/tvcalib_batch_out")


def discover_clips() -> list[dict]:
    rows = []
    for split in SPLITS:
        split_dir = GSR_ROOT / split
        if not split_dir.is_dir():
            continue
        for clip_dir in sorted(split_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            labels_path = clip_dir / "Labels-GameState.json"
            if not labels_path.is_file():
                continue
            try:
                labels = json.loads(labels_path.read_text())
            except Exception:
                continue
            info = labels.get("info", {})
            if info.get("action_class") not in TARGET_ACTIONS:
                continue
            n_frames = len(labels.get("images", []))
            # action_position is a global broadcast frame number, not clip-local (1–750).
            # Clips start at the set-piece; use the first 2*FRAME_WINDOW+1 frames.
            centre = min(FRAME_WINDOW + 1, n_frames)
            lo = max(1, centre - FRAME_WINDOW)
            hi = min(n_frames, centre + FRAME_WINDOW)
            rows.append({
                "split": split,
                "clip_id": clip_dir.name,
                "action_class": info["action_class"],
                "centre": centre,
                "lo": lo,
                "hi": hi,
            })
    return rows


def stage(clips: list[dict]) -> int:
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    n_total = 0
    n_new = 0
    for c in clips:
        for frame_idx in range(c["lo"], c["hi"] + 1):
            n_total += 1
            src = GSR_ROOT / c["split"] / c["clip_id"] / "img1" / f"{frame_idx:06d}.jpg"
            if not src.is_file():
                continue
            dst = STAGE_DIR / f"{c['split']}__{c['clip_id']}__{frame_idx:06d}.jpg"
            if dst.is_file():
                continue
            shutil.copy(src, dst)
            n_new += 1
    print(f"staged: {n_new} new, {n_total} total expected, dir={STAGE_DIR}")
    return n_total


def run_tvcalib() -> dict:
    cache = OUT_DIR / "calib.json"
    if cache.is_file():
        print(f"using cached {cache}")
        return json.loads(cache.read_text())

    tvcalib_python = TVCALIB_ROOT / ".venv/bin/python"
    tvcalib_script = TVCALIB_ROOT / "run_inference.py"

    if not tvcalib_python.is_file():
        raise FileNotFoundError(
            f"TVCalib venv not found at {tvcalib_python}\n"
            "Set up TVCalib in a sibling directory — see this script's docstring for instructions."
        )
    if not tvcalib_script.is_file():
        raise FileNotFoundError(f"TVCalib inference script not found at {tvcalib_script}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(tvcalib_python),
        str(tvcalib_script),
        "--images_path", str(STAGE_DIR),
        "--output_dir", str(OUT_DIR),
        "--image_width", "1920",
        "--image_height", "1080",
        "--batch_size_seg", "8",
        "--batch_size_calib", "256",
        "--optim_steps", "2000",
    ]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, cwd=TVCALIB_ROOT, check=True)
    return json.loads(cache.read_text())


def parse_results(results: dict) -> pd.DataFrame:
    rows = []
    for image_id, payload in results.items():
        # image_id e.g. "train__SNGS-066__000735.jpg"
        stem = image_id.removesuffix(".jpg")
        parts = stem.split("__")
        if len(parts) != 3:
            print(f"[skip] cannot parse: {image_id}")
            continue
        split, clip_id, frame_str = parts
        try:
            frame_idx = int(frame_str)
        except ValueError:
            continue
        H = np.array(payload["H_world_to_image"], dtype=np.float64)
        if H.shape != (3, 3):
            continue
        rows.append({
            "split": split,
            "clip_id": clip_id,
            "frame_idx": frame_idx,
            "h00": H[0, 0], "h01": H[0, 1], "h02": H[0, 2],
            "h10": H[1, 0], "h11": H[1, 1], "h12": H[1, 2],
            "h20": H[2, 0], "h21": H[2, 1], "h22": H[2, 2],
            "loss_ndc_total": float(payload.get("loss_ndc_total", float("nan"))),
        })
    return pd.DataFrame(rows)


def main():
    verify_soccernet_data()

    clips = discover_clips()
    print(f"discovered {len(clips)} set-piece clips")
    n_expected = stage(clips)
    results = run_tvcalib()
    print(f"tvcalib produced {len(results)} results (staged {n_expected})")

    df = parse_results(results)
    out_path = PROJECT_ROOT / "outputs" / "homographies_tvcalib.parquet"
    df.to_parquet(out_path, index=False)
    print(f"saved {len(df)} rows -> {out_path}")
    print("loss_ndc_total stats:")
    print(df["loss_ndc_total"].describe().round(4))
    print("clips covered:", df.groupby(["split", "clip_id"])["frame_idx"].count().describe().round(1))


if __name__ == "__main__":
    main()
