"""Phase 5 Soccana detector under TVCalib autonomous H.

Reads:
    outputs/homographies_tvcalib.parquet
    SSD frames (SOCCERNET_LOCAL_DIR)

Writes:
    outputs/detections_soccana_tvcalib.parquet
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import pandas as pd
from huggingface_hub import hf_hub_download

# PyTorch 2.6 changed weights_only default to True; ultralytics .pt weights
# contain arbitrary globals and require weights_only=False.
_orig_torch_load = torch.load
torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})

from ultralytics import YOLO

# Ensure sibling scripts are importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _pipeline_core import (
    DEVICE,
    discover_setpiece_clips,
    load_tvcalib_lookup,
    run_clip,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
GSR_ROOT = Path(os.getenv("SOCCERNET_LOCAL_DIR", "/Volumes/MPH-ExternalStorage/soccernet-gsr")) / "gamestate-2024"

SOCCANA_REPO = "Adit-jain/soccana"
SOCCANA_WEIGHTS_PATH_IN_REPO = "Model/weights/best.pt"
PLAYER_CLASS = 0


def main() -> None:
    assert GSR_ROOT.exists(), f"SoccerNet GSR not mounted: {GSR_ROOT}"
    H_lookup = load_tvcalib_lookup(OUTPUTS_DIR)
    print(f"TVCalib H entries: {len(H_lookup)}")

    clips = discover_setpiece_clips(GSR_ROOT)
    print(f"Set-piece clips: {len(clips)}")

    weights_path = hf_hub_download(repo_id=SOCCANA_REPO, filename=SOCCANA_WEIGHTS_PATH_IN_REPO)
    yolo = YOLO(weights_path)
    print(f"Soccana weights: {weights_path}  device: {DEVICE}")

    all_rows: list[dict] = []
    skipped: list[tuple] = []

    for i, (_, clip) in enumerate(clips.iterrows()):
        rows, reason = run_clip(clip, yolo, H_lookup, player_class=PLAYER_CLASS)
        if reason is not None:
            skipped.append((clip["clip_id"], clip["action_position"], reason))
        else:
            all_rows.extend(rows)
        if (i + 1) % 5 == 0:
            print(f"  clips {i+1}/{len(clips)}  |  rows: {len(all_rows)}")

    df = pd.DataFrame(all_rows)
    out = OUTPUTS_DIR / "detections_soccana_tvcalib.parquet"
    df.to_parquet(out, engine="pyarrow", index=False)
    print(f"\nDone. Rows: {len(df)}  |  clips skipped: {len(skipped)}  |  saved: {out}")
    print(f"Clips with rows: {df['clip_id'].nunique()}")


if __name__ == "__main__":
    main()
