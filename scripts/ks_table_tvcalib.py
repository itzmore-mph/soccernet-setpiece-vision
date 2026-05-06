"""KS table for TVCalib autonomous-H pipeline vs full GT (33-clip cohort).

Compares pipeline detections under three H sources:
  - 'gt-leak' baseline: outputs/pitch_control.parquet (track='pipeline', 20 clips)
  - 'tvcalib' autonomous: outputs/pitch_control_tvcalib.parquet (33 clips)
GT reference:
  - outputs/pitch_control_gt_full.parquet (33-clip GT) for autonomous comparison
  - outputs/pitch_control.parquet (track='gt', 20 clips) for baseline comparison

Writes:
    outputs/validation_summary_tvcalib.parquet
    outputs/figures/13_ks_table_tvcalib.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

KS_ALPHA = 0.05
HIST_BINS = 12
METRICS = ["pc_mean", "pc_at_ball", "pc_in_box", "pc_in_third", "pc_area_gt_0p5"]


def hist_overlap(a, b, bins=HIST_BINS, lo=0.0, hi=1.0):
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges)
    pb, _ = np.histogram(b, bins=edges)
    pa = pa / pa.sum() if pa.sum() else pa
    pb = pb / pb.sum() if pb.sum() else pb
    return float(np.minimum(pa, pb).sum())


def compare(a, b, source, metric):
    if len(a) < 3 or len(b) < 3:
        return {"source": source, "metric": metric, "n": len(a), "n_gt": len(b),
                "ks_stat": np.nan, "ks_p": np.nan, "hist_overlap": np.nan,
                "mean": np.nan, "mean_gt": np.nan, "delta": np.nan, "passes_ks": False}
    ks = stats.ks_2samp(a, b)
    return {
        "source": source, "metric": metric,
        "n": len(a), "n_gt": len(b),
        "mean": float(a.mean()), "mean_gt": float(b.mean()),
        "delta": float(a.mean() - b.mean()),
        "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
        "hist_overlap": hist_overlap(a, b),
        "passes_ks": bool(ks.pvalue >= KS_ALPHA),
    }


def main():
    pc_base = pd.read_parquet(OUTPUTS_DIR / "pitch_control.parquet")
    pc_tv = pd.read_parquet(OUTPUTS_DIR / "pitch_control_tvcalib.parquet")
    pc_gt_full = pd.read_parquet(OUTPUTS_DIR / "pitch_control_gt_full.parquet")
    pc_soc_tv_path = OUTPUTS_DIR / "pitch_control_soccana_tvcalib.parquet"
    pc_soc_tv = pd.read_parquet(pc_soc_tv_path) if pc_soc_tv_path.is_file() else None

    gt_old = pc_base[pc_base["track"] == "gt"]
    pipe_old = pc_base[pc_base["track"] == "pipeline"]
    tv = pc_tv
    gt_full = pc_gt_full

    print(f"baseline pipeline frames : {len(pipe_old)} ({pipe_old['clip_id'].nunique()} clips)")
    print(f"baseline gt frames       : {len(gt_old)} ({gt_old['clip_id'].nunique()} clips)")
    print(f"tvcalib  pipeline frames : {len(tv)}  ({tv['clip_id'].nunique()} clips)")
    print(f"full     gt frames       : {len(gt_full)}  ({gt_full['clip_id'].nunique()} clips)")
    if pc_soc_tv is not None:
        print(f"soccana+tvcalib frames   : {len(pc_soc_tv)}  ({pc_soc_tv['clip_id'].nunique()} clips)")

    rows = []
    for metric in METRICS:
        gt_old_a = gt_old[metric].dropna().to_numpy()
        gt_full_a = gt_full[metric].dropna().to_numpy()
        rows.append(compare(pipe_old[metric].dropna().to_numpy(), gt_old_a,
                            "gt-leak yolov8x (vs gt 20)", metric))
        rows.append(compare(tv[metric].dropna().to_numpy(), gt_full_a,
                            "tvcalib yolov8x (vs gt 33)", metric))
        if pc_soc_tv is not None:
            rows.append(compare(pc_soc_tv[metric].dropna().to_numpy(), gt_full_a,
                                "tvcalib soccana (vs gt 33)", metric))

    df = pd.DataFrame(rows)
    df.to_parquet(OUTPUTS_DIR / "validation_summary_tvcalib.parquet", index=False)

    cols = ["source", "metric", "mean", "mean_gt", "delta", "ks_stat", "ks_p", "hist_overlap", "passes_ks"]
    print()
    print(df[cols].round(4).to_string(index=False))

    # Pretty figure
    fig, ax = plt.subplots(figsize=(13, 0.45 * len(df) + 1.4))
    ax.axis("off")
    cell = df[cols].round(3).astype(str).values
    table = ax.table(cellText=cell, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    ax.set_title(
        f"H-source ablation: GT-pitch-line leak vs TVCalib autonomous (KS alpha={KS_ALPHA})",
        fontsize=11, pad=12,
    )
    out_fig = FIGURES_DIR / "13_ks_table_tvcalib.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_fig}")

    base_pass = df[(df["source"].str.startswith("gt-leak")) & df["passes_ks"]]["metric"].tolist()
    tv_pass = df[(df["source"].str.startswith("tvcalib")) & df["passes_ks"]]["metric"].tolist()
    print(f"\ngt-leak passes KS: {base_pass or 'none'}")
    print(f"tvcalib passes KS: {tv_pass or 'none'}")


if __name__ == "__main__":
    main()
