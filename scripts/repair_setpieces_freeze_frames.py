"""Repair outputs/setpieces.parquet: re-attach StatsBomb 360 freeze frames.

The original nb01 run produced setpieces.parquet with 100% empty freeze_frame
arrays (likely a transient statsbombpy issue at execution time). This script
re-fetches frames per match and rebuilds the column. All other columns are
preserved.

Run from repo root:
    python scripts/repair_setpieces_freeze_frames.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsbombpy import sb
from statsbombpy.api_client import NoAuthWarning

warnings.filterwarnings("ignore", category=NoAuthWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SP_PATH = PROJECT_ROOT / "outputs" / "setpieces.parquet"

PITCH_LENGTH_M = 105
PITCH_WIDTH_M = 68
SB_X_MAX = 120
SB_Y_MAX = 80


def sb_to_m(x, y):
    return x * (PITCH_LENGTH_M / SB_X_MAX), y * (PITCH_WIDTH_M / SB_Y_MAX)


def convert_freeze_frame(frame_rows: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in frame_rows.iterrows():
        loc = r.get("location")
        if loc is None or (isinstance(loc, float) and np.isnan(loc)):
            continue
        x_m, y_m = sb_to_m(loc[0], loc[1])
        out.append({
            "player_x_m": float(x_m),
            "player_y_m": float(y_m),
            "teammate": bool(r.get("teammate", False)),
            "actor": bool(r.get("actor", False)),
            "keeper": bool(r.get("keeper", False)),
        })
    return out


def main():
    sp = pd.read_parquet(SP_PATH)
    print(f"Loaded {SP_PATH.name}: {sp.shape}")
    print(f"Empty FF before: {(sp['freeze_frame'].apply(len) == 0).sum()}/{len(sp)}")

    new_ff = []
    cache: dict[int, dict] = {}
    for _, row in sp.iterrows():
        mid = int(row["match_id"])
        if mid not in cache:
            try:
                fr = sb.frames(match_id=mid)
            except Exception as e:
                print(f"  match {mid} frames failed: {e}")
                fr = None
            if fr is not None and not fr.empty:
                key = "event_uuid" if "event_uuid" in fr.columns else "id"
                cache[mid] = {eid: grp for eid, grp in fr.groupby(key)}
            else:
                cache[mid] = {}

        ff_rows = cache[mid].get(row["event_id"])
        new_ff.append(convert_freeze_frame(ff_rows) if ff_rows is not None else [])

    sp = sp.copy()
    sp["freeze_frame"] = new_ff

    n_with = (sp["freeze_frame"].apply(len) > 0).sum()
    print(f"Empty FF after : {(sp['freeze_frame'].apply(len) == 0).sum()}/{len(sp)}")
    print(f"With FF        : {n_with}  ({100*n_with/len(sp):.1f}%)")

    sp.to_parquet(SP_PATH, engine="pyarrow", index=False)
    print(f"Saved          : {SP_PATH}")


if __name__ == "__main__":
    main()
